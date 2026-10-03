"""Download and exactly reproduce the established monocular EuRoC benchmark."""

from __future__ import annotations

import argparse
import io
import json
import shutil
import struct
import zipfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pycolmap
import yaml
from scipy.spatial.transform import Rotation

from vidmap.datasets.layouts import get_dataset_layout
from vidmap.datasets.prepare.video_common import checksum, resumable_http_download
from vidmap.datasets.resources import dataset_asset

EUROC_PREPARATION_VERSION = 1
EUROC_LICENSE = "In Copyright - Non-Commercial Use Permitted"
EUROC_SOURCE = "https://www.research-collection.ethz.ch/items/bcaf173e-5dac-484b-bc37-faf97a594f1f"
EUROC_LAYOUT = get_dataset_layout("euroc")
_MODEL_FILES = ("cameras.bin", "images.bin", "points3D.bin")
_CAMERA_MODEL_PINHOLE = 1


@dataclass(frozen=True)
class SourceArchive:
    name: str
    filename: str
    url: str
    size: int
    sha256: str


@dataclass(frozen=True)
class SequenceSource:
    scene: str
    archive: str
    member: str
    size: int
    sha256: str
    raw_cam0_images: int
    canonical_images: int
    first_image: str
    last_image: str


@dataclass(frozen=True)
class CameraCalibration:
    width: int
    height: int
    intrinsics: np.ndarray
    distortion: np.ndarray
    body_to_camera: np.ndarray


@dataclass(frozen=True)
class ImagePose:
    image_id: int
    name: str
    camera_from_world: np.ndarray


@dataclass(frozen=True)
class EurocGroundTruth:
    timestamps: np.ndarray  # (N,) int64 nanoseconds
    positions: np.ndarray  # (N, 3) p_RS_R in meters
    quaternions: np.ndarray  # (N, 4) q_RS in wxyz
    velocities: np.ndarray  # (N, 3) v_RS_R in m/s
    gyro_biases: np.ndarray  # (N, 3) b_w_RS_S in rad/s
    accel_biases: np.ndarray  # (N, 3) b_a_RS_S in m/s^2

    def __iter__(self):
        return iter(
            (
                self.timestamps,
                self.positions,
                self.quaternions,
                self.velocities,
                self.gyro_biases,
                self.accel_biases,
            )
        )

    def __getitem__(self, idx):
        return (
            self.timestamps,
            self.positions,
            self.quaternions,
            self.velocities,
            self.gyro_biases,
            self.accel_biases,
        )[idx]


def source_inventory() -> tuple[dict[str, SourceArchive], dict[str, SequenceSource]]:
    """Return the explicit checksum-pinned official archive and sequence inventory."""
    payload = json.loads(dataset_asset("euroc", "source_archives.json").read_text(encoding="utf-8"))
    archives = {
        name: SourceArchive(
            name=name,
            filename=raw["filename"],
            url=raw["url"],
            size=int(raw["size"]),
            sha256=raw["sha256"],
        )
        for name, raw in payload["archives"].items()
    }
    sequences = {
        scene: SequenceSource(
            scene=scene,
            archive=raw["archive"],
            member=raw["member"],
            size=int(raw["size"]),
            sha256=raw["sha256"],
            raw_cam0_images=int(raw["raw_cam0_images"]),
            canonical_images=int(raw["canonical_images"]),
            first_image=raw["first_image"],
            last_image=raw["last_image"],
        )
        for scene, raw in payload["sequences"].items()
    }
    if tuple(sequences) != EUROC_LAYOUT.scenes:
        raise ValueError("EuRoC source inventory does not exactly match the registered sequence manifest")
    if any(sequence.archive not in archives for sequence in sequences.values()):
        raise ValueError("EuRoC sequence inventory references an unknown source archive")
    return archives, sequences


def parse_sensor(text: str) -> CameraCalibration:
    """Parse one official EuRoC cam0 ``sensor.yaml``."""
    payload = yaml.safe_load(text)
    camera_model = payload["camera_model"]
    distortion_model = payload["distortion_model"]
    if camera_model != "pinhole":
        raise ValueError(f"Unsupported EuRoC camera model {camera_model!r}")
    if distortion_model != "radial-tangential":
        raise ValueError(f"Unsupported EuRoC distortion model {distortion_model!r}")
    width, height = (int(value) for value in payload["resolution"])
    fx, fy, cx, cy = (float(value) for value in payload["intrinsics"])
    coefficients = tuple(float(value) for value in payload["distortion_coefficients"])
    if len(coefficients) != 4:
        raise ValueError(f"Expected four EuRoC distortion coefficients, got {len(coefficients)}")
    body_to_camera = np.asarray(payload["T_BS"]["data"], dtype=np.float64).reshape(4, 4)
    intrinsics = np.asarray([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
    distortion = np.asarray((*coefficients, 0.0), dtype=np.float64)
    if (
        width <= 0
        or height <= 0
        or not all(np.isfinite(values).all() for values in (intrinsics, distortion, body_to_camera))
    ):
        raise ValueError("EuRoC camera calibration contains invalid values")
    if not np.allclose(body_to_camera[3], [0.0, 0.0, 0.0, 1.0], rtol=0.0, atol=1e-12):
        raise ValueError("EuRoC T_BS is not a homogeneous transform")
    return CameraCalibration(width, height, intrinsics, distortion, body_to_camera)


def parse_image_index(text: str) -> tuple[tuple[int, str], ...]:
    """Parse the ordered official cam0 timestamp/filename CSV."""
    images = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line or line.startswith("#"):
            continue
        fields = line.split(",")
        if len(fields) != 2:
            raise ValueError(f"Invalid EuRoC image row {line_number}: {line!r}")
        timestamp, name = int(fields[0]), fields[1]
        if Path(name).name != name or not name.endswith(".png"):
            raise ValueError(f"Unsafe EuRoC image name at row {line_number}: {name!r}")
        images.append((timestamp, name))
    if not images or any(left[0] >= right[0] for left, right in zip(images, images[1:])):
        raise ValueError("EuRoC image timestamps must be non-empty and strictly increasing")
    if len({name for _, name in images}) != len(images):
        raise ValueError("EuRoC image index contains duplicate filenames")
    return tuple(images)


def parse_ground_truth(text: str) -> EurocGroundTruth:
    """Parse all 17 columns of EuRoC state_groundtruth_estimate0/data.csv."""
    values = np.loadtxt(io.StringIO(text), delimiter=",", skiprows=1, ndmin=2)
    if values.shape[1] < 17 or len(values) < 2 or not np.isfinite(values[:, :17]).all():
        raise ValueError("EuRoC ground-truth CSV does not contain 17 valid columns")
    timestamps = values[:, 0].astype(np.int64)
    if np.any(np.diff(timestamps) <= 0):
        raise ValueError("EuRoC ground-truth timestamps must be strictly increasing")
    positions = values[:, 1:4]
    quaternions = values[:, 4:8]  # w, x, y, z
    if np.any(np.linalg.norm(quaternions, axis=1) <= np.finfo(np.float64).eps):
        raise ValueError("EuRoC ground-truth CSV contains a zero quaternion")
    velocities = values[:, 8:11]
    gyro_biases = values[:, 11:14]
    accel_biases = values[:, 14:17]
    return EurocGroundTruth(
        timestamps=timestamps,
        positions=positions,
        quaternions=quaternions,
        velocities=velocities,
        gyro_biases=gyro_biases,
        accel_biases=accel_biases,
    )


def interpolate_world_from_body(
    timestamps: np.ndarray,
    positions: np.ndarray,
    quaternions_wxyz: np.ndarray,
    query_timestamp: int,
) -> np.ndarray | None:
    """Interpolate the official body pose at one camera timestamp."""
    index = int(np.searchsorted(timestamps, query_timestamp))
    if index == 0 or index >= len(timestamps):
        return None
    t0, t1 = timestamps[index - 1], timestamps[index]
    alpha = float((query_timestamp - t0) / (t1 - t0))
    position = (1.0 - alpha) * positions[index - 1] + alpha * positions[index]
    q0, q1 = quaternions_wxyz[index - 1], quaternions_wxyz[index]
    r0 = Rotation.from_quat([q0[1], q0[2], q0[3], q0[0]])
    r1 = Rotation.from_quat([q1[1], q1[2], q1[3], q1[0]])
    interpolated = Rotation.from_rotvec((1.0 - alpha) * r0.as_rotvec() + alpha * r1.as_rotvec())
    world_from_body = np.eye(4, dtype=np.float64)
    world_from_body[:3, :3] = interpolated.as_matrix()
    world_from_body[:3, 3] = position
    return world_from_body


def camera_from_world(world_from_body: np.ndarray, body_to_camera: np.ndarray) -> np.ndarray:
    """Convert an official world-from-body pose to COLMAP camera-from-world."""
    return body_to_camera @ np.linalg.inv(world_from_body)


def _canonical_poses(
    image_index: tuple[tuple[int, str], ...],
    ground_truth: EurocGroundTruth | Sequence[Any],
    body_to_camera: np.ndarray,
) -> tuple[ImagePose, ...]:
    timestamps, positions, quaternions = ground_truth[:3]
    result = []
    for timestamp, name in image_index:
        world_from_body = interpolate_world_from_body(timestamps, positions, quaternions, timestamp)
        if world_from_body is not None:
            result.append(
                ImagePose(
                    len(result),
                    name,
                    camera_from_world(world_from_body, body_to_camera),
                )
            )
    return tuple(result)


def _rectification(
    calibration: CameraCalibration,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    size = (calibration.width, calibration.height)
    rectified, _ = cv2.getOptimalNewCameraMatrix(
        calibration.intrinsics,
        calibration.distortion,
        size,
        alpha=0.0,
        newImgSize=size,
    )
    map1, map2 = cv2.initUndistortRectifyMap(
        calibration.intrinsics,
        calibration.distortion,
        None,
        rectified,
        size,
        cv2.CV_32FC1,
    )
    return rectified, map1, map2


def _rectify_png(encoded: bytes, map1: np.ndarray, map2: np.ndarray, size: tuple[int, int]) -> bytes:
    raw = cv2.imdecode(np.frombuffer(encoded, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if raw is None:
        raise ValueError("Could not decode an official EuRoC cam0 PNG")
    if (raw.shape[1], raw.shape[0]) != size:
        raise ValueError(f"EuRoC image/calibration size mismatch: {(raw.shape[1], raw.shape[0])} vs {size}")
    image = cv2.remap(raw, map1, map2, interpolation=cv2.INTER_LINEAR)
    success, output = cv2.imencode(".png", image)
    if not success:
        raise ValueError("Could not encode a rectified EuRoC PNG")
    return output.tobytes()


def _camera_bytes(calibration: CameraCalibration, rectified: np.ndarray) -> bytes:
    fx, fy, cx, cy = rectified[[0, 1, 0, 1], [0, 1, 2, 2]]
    return b"".join(
        (
            struct.pack("<Q", 1),
            struct.pack("<IiQQ", 1, _CAMERA_MODEL_PINHOLE, calibration.width, calibration.height),
            struct.pack("<dddd", fx, fy, cx, cy),
        )
    )


def _images_bytes(images: tuple[ImagePose, ...]) -> bytes:
    output = io.BytesIO()
    output.write(struct.pack("<Q", len(images)))
    for image in images:
        transform = image.camera_from_world
        quaternion_xyzw = Rotation.from_matrix(transform[:3, :3]).as_quat()
        qx, qy, qz, qw = quaternion_xyzw
        output.write(struct.pack("<I", image.image_id))
        output.write(struct.pack("<dddd", qw, qx, qy, qz))
        output.write(struct.pack("<ddd", *transform[:3, 3]))
        output.write(struct.pack("<I", 1))
        output.write(image.name.encode("utf-8") + b"\0")
        output.write(struct.pack("<Q", 0))
    return output.getvalue()


def canonical_testset_bytes(image_count: int) -> bytes:
    """Return the canonical full-trajectory EuRoC testset bytes."""
    return yaml.safe_dump({0: list(range(image_count))}, default_flow_style=True).encode("utf-8")


def _write_exact(destination: Path, contents: bytes) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        if not destination.is_file() or destination.read_bytes() != contents:
            raise RuntimeError(f"Existing EuRoC artifact differs from the canonical output: {destination}")
        return
    partial = destination.with_suffix(destination.suffix + ".part")
    if partial.exists() or partial.is_symlink():
        if not partial.is_file() or partial.read_bytes() != contents:
            raise RuntimeError(f"Existing EuRoC staging artifact is invalid: {partial}")
        partial.replace(destination)
        return
    with partial.open("xb") as handle:
        handle.write(contents)
    partial.replace(destination)


def _ensure_testset(sequence: SequenceSource) -> None:
    _write_exact(
        EUROC_LAYOUT.testsets / sequence.scene / "all.yaml",
        canonical_testset_bytes(sequence.canonical_images),
    )


def _validate_archive(path: Path, *, size: int, sha256: str) -> None:
    if not path.is_file() or path.stat().st_size != size:
        actual = path.stat().st_size if path.is_file() else "missing"
        raise ValueError(f"EuRoC archive size mismatch for {path}: expected {size}, got {actual}")
    actual_digest = checksum(path, "sha256")
    if actual_digest != sha256:
        raise ValueError(f"EuRoC SHA-256 mismatch for {path}: expected {sha256}, got {actual_digest}")


def _sequence_archive(sequence: SequenceSource, aggregate: SourceArchive, download_dir: Path) -> Path:
    destination = download_dir / "sequences" / f"{sequence.scene}.zip"
    if destination.exists():
        _validate_archive(destination, size=sequence.size, sha256=sequence.sha256)
        return destination
    outer_path = download_dir / aggregate.filename
    _validate_archive(outer_path, size=aggregate.size, sha256=aggregate.sha256)
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(".zip.part")
    offset = partial.stat().st_size if partial.exists() else 0
    if offset > sequence.size:
        raise ValueError(f"Oversized EuRoC nested-archive partial: {partial}")
    with zipfile.ZipFile(outer_path) as bundle:
        info = bundle.getinfo(sequence.member)
        if info.file_size != sequence.size:
            raise ValueError(
                f"EuRoC nested archive size mismatch for {sequence.member}: {info.file_size} != {sequence.size}"
            )
        with bundle.open(info) as source, partial.open("ab" if partial.exists() else "xb") as target:
            if offset:
                source.seek(offset)
            shutil.copyfileobj(source, target, length=8 * 1024 * 1024)
    _validate_archive(partial, size=sequence.size, sha256=sequence.sha256)
    partial.replace(destination)
    return destination


def _read_member(bundle: zipfile.ZipFile, member: str) -> bytes:
    return bundle.read(member)


def euroc_scene_is_prepared(scene_dir: Path) -> bool:
    """Check the exact established image/pose/testset boundary for one sequence."""
    _, sequences = source_inventory()
    if scene_dir.name not in sequences:
        return False
    sequence = sequences[scene_dir.name]
    images_dir = scene_dir / "images"
    rec_dir = scene_dir / "rec"
    testset = EUROC_LAYOUT.testsets / sequence.scene / "all.yaml"
    if not images_dir.is_dir() or not all((rec_dir / name).is_file() for name in _MODEL_FILES):
        return False
    image_paths = tuple(sorted(path for path in images_dir.iterdir() if path.is_file()))
    if (
        len(image_paths) != sequence.canonical_images
        or image_paths[0].name != sequence.first_image
        or image_paths[-1].name != sequence.last_image
        or not testset.is_file()
        or testset.read_bytes() != canonical_testset_bytes(sequence.canonical_images)
    ):
        return False
    reconstruction = pycolmap.Reconstruction(rec_dir)
    ordered = tuple(sorted(reconstruction.images.items()))
    camera = next(iter(reconstruction.cameras.values()), None)
    return (
        camera is not None
        and len(reconstruction.cameras) == 1
        and camera.model_name == "PINHOLE"
        and len(ordered) == sequence.canonical_images
        and tuple(image_id for image_id, _ in ordered) == tuple(range(sequence.canonical_images))
        and tuple(image.name for _, image in ordered) == tuple(path.name for path in image_paths)
    )


def _write_preparation_record(scene_dir: Path, sequence: SequenceSource, aggregate: SourceArchive) -> None:
    payload = {
        "dataset": "EuRoC MAV",
        "sequence": sequence.scene,
        "preparation_version": EUROC_PREPARATION_VERSION,
        "gt_evaluable": True,
        "camera": "cam0",
        "image_rectification": "opencv-radtan-alpha-0.0",
        "temporal_sampling": "all-cam0-frames-within-ground-truth-range",
        "source_archive": aggregate.filename,
        "source_archive_sha256": aggregate.sha256,
        "sequence_archive_sha256": sequence.sha256,
        "license": EUROC_LICENSE,
        "source": EUROC_SOURCE,
    }
    _write_exact(
        scene_dir / "preparation.json",
        (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )


def parse_euroc_imu(text: str) -> pycolmap.ImuMeasurements:
    """Parse EuRoC mav0/imu0/data.csv into pycolmap.ImuMeasurements."""
    values = np.loadtxt(io.StringIO(text), delimiter=",", skiprows=1, ndmin=2)
    measurements = pycolmap.ImuMeasurements()
    prev_ts = None
    for row in values:
        ts = int(round(row[0]))
        if prev_ts is not None and ts <= prev_ts:
            ts = prev_ts + 1
        prev_ts = ts
        gyro = row[1:4]
        accel = row[4:7]
        measurements.insert(pycolmap.ImuMeasurement(ts, gyro, accel))
    return measurements


def parse_euroc_imu_calibration(text: str) -> pycolmap.ImuCalibration:
    """Parse EuRoC mav0/imu0/sensor.yaml into pycolmap.ImuCalibration."""
    payload = yaml.safe_load(text)
    calib = pycolmap.ImuCalibration()
    if "rate_hz" in payload:
        calib.imu_rate = float(payload["rate_hz"])
    if "gyroscope_noise_density" in payload:
        calib.gyro_noise_density = float(payload["gyroscope_noise_density"])
    if "gyroscope_random_walk" in payload:
        calib.bias_gyro_random_walk_sigma = float(payload["gyroscope_random_walk"])
    if "accelerometer_noise_density" in payload:
        calib.accel_noise_density = float(payload["accelerometer_noise_density"])
    if "accelerometer_random_walk" in payload:
        calib.bias_accel_random_walk_sigma = float(payload["accelerometer_random_walk"])
    return calib


def load_euroc_imu_and_gt(
    scene_dir: Path | str,
) -> tuple[pycolmap.ImuMeasurements, pycolmap.ImuCalibration, EurocGroundTruth]:
    """Load IMU measurements, calibration, and ground-truth trajectory from a EuRoC scene directory."""
    path = Path(scene_dir)
    imu_file = None
    for candidate in (
        path / "mav0" / "imu0" / "data.csv",
        path / "imu" / "data.csv",
        path / "imu0" / "data.csv",
        path / "imu_data.csv",
    ):
        if candidate.is_file():
            imu_file = candidate
            break
    if imu_file is None:
        raise FileNotFoundError(f"Could not find EuRoC IMU data.csv in {path}")

    sensor_file = None
    for candidate in (
        path / "mav0" / "imu0" / "sensor.yaml",
        path / "imu" / "sensor.yaml",
        path / "imu0" / "sensor.yaml",
        path / "sensor.yaml",
    ):
        if candidate.is_file():
            sensor_file = candidate
            break

    gt_file = None
    for candidate in (
        path / "mav0" / "state_groundtruth_estimate0" / "data.csv",
        path / "state_groundtruth_estimate0" / "data.csv",
        path / "imu" / "state_groundtruth_estimate0.csv",
        path / "groundtruth.csv",
    ):
        if candidate.is_file():
            gt_file = candidate
            break
    if gt_file is None:
        raise FileNotFoundError(f"Could not find EuRoC groundtruth data.csv in {path}")

    measurements = parse_euroc_imu(imu_file.read_text(encoding="utf-8"))
    calib = (
        parse_euroc_imu_calibration(sensor_file.read_text(encoding="utf-8"))
        if sensor_file is not None
        else pycolmap.ImuCalibration()
    )
    gt = parse_ground_truth(gt_file.read_text(encoding="utf-8"))
    return measurements, calib, gt


def prepare_scene(scene: str, download_dir: Path) -> None:
    """Prepare one official EuRoC sequence without replacing existing output."""
    archives, sequences = source_inventory()
    if scene not in sequences:
        raise ValueError(f"Unknown EuRoC sequence {scene!r}")
    sequence = sequences[scene]
    final_scene = EUROC_LAYOUT.data_dir / scene
    if final_scene.exists() or final_scene.is_symlink():
        _ensure_testset(sequence)
        if euroc_scene_is_prepared(final_scene):
            return
        raise RuntimeError(f"Existing EuRoC scene is not the canonical prepared output: {final_scene}")

    aggregate = archives[sequence.archive]
    sequence_path = _sequence_archive(sequence, aggregate, download_dir)
    staging_scene = EUROC_LAYOUT.data_dir / ".euroc-preparing" / scene
    staging_scene.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(sequence_path) as bundle:
        calibration = parse_sensor(_read_member(bundle, "mav0/cam0/sensor.yaml").decode("utf-8"))
        image_index = parse_image_index(_read_member(bundle, "mav0/cam0/data.csv").decode("utf-8"))
        if len(image_index) != sequence.raw_cam0_images:
            raise ValueError(
                f"EuRoC {scene} raw cam0 count mismatch: {len(image_index)} != {sequence.raw_cam0_images}"
            )
        ground_truth = parse_ground_truth(
            _read_member(bundle, "mav0/state_groundtruth_estimate0/data.csv").decode("utf-8")
        )
        poses = _canonical_poses(image_index, ground_truth, calibration.body_to_camera)
        if (
            len(poses) != sequence.canonical_images
            or poses[0].name != sequence.first_image
            or poses[-1].name != sequence.last_image
        ):
            raise ValueError(f"EuRoC {scene} canonical GT-range inventory mismatch")
        rectified, map1, map2 = _rectification(calibration)
        for image in poses:
            contents = _read_member(bundle, f"mav0/cam0/data/{image.name}")
            _write_exact(
                staging_scene / "images" / image.name,
                _rectify_png(contents, map1, map2, (calibration.width, calibration.height)),
            )

        imu_csv = _read_member(bundle, "mav0/imu0/data.csv")
        imu_sensor = _read_member(bundle, "mav0/imu0/sensor.yaml")
        gt_csv = _read_member(bundle, "mav0/state_groundtruth_estimate0/data.csv")

        _write_exact(staging_scene / "mav0" / "imu0" / "data.csv", imu_csv)
        _write_exact(staging_scene / "mav0" / "imu0" / "sensor.yaml", imu_sensor)
        _write_exact(staging_scene / "mav0" / "state_groundtruth_estimate0" / "data.csv", gt_csv)
        _write_exact(staging_scene / "imu" / "data.csv", imu_csv)
        _write_exact(staging_scene / "imu" / "sensor.yaml", imu_sensor)
        _write_exact(staging_scene / "state_groundtruth_estimate0" / "data.csv", gt_csv)

    _write_exact(staging_scene / "rec" / "cameras.bin", _camera_bytes(calibration, rectified))
    _write_exact(staging_scene / "rec" / "images.bin", _images_bytes(poses))
    _write_exact(staging_scene / "rec" / "points3D.bin", struct.pack("<Q", 0))
    _write_preparation_record(staging_scene, sequence, aggregate)
    _ensure_testset(sequence)
    if not euroc_scene_is_prepared(staging_scene):
        raise RuntimeError(f"Prepared EuRoC scene failed final validation: {scene}")
    final_scene.parent.mkdir(parents=True, exist_ok=True)
    staging_scene.replace(final_scene)


def _validate_zip_response(response) -> None:
    content_type = response.headers["Content-Type"].lower()
    if "text/html" in content_type:
        raise RuntimeError("EuRoC official source returned unexpected HTML instead of a ZIP archive")


def main(
    scenes: tuple[str, ...] | list[str] | None = None,
    *,
    download: bool = True,
    delete_files: bool = False,
) -> None:
    """Download official bundles and prepare selected or all EuRoC sequences."""
    selected = EUROC_LAYOUT.select_scenes(scenes)
    archives, sequences = source_inventory()
    pending = []
    for scene in selected:
        final_scene = EUROC_LAYOUT.data_dir / scene
        if final_scene.exists() or final_scene.is_symlink():
            _ensure_testset(sequences[scene])
            if not euroc_scene_is_prepared(final_scene):
                raise RuntimeError(f"Existing EuRoC scene is not the canonical prepared output: {final_scene}")
        else:
            pending.append(scene)
    if not pending:
        return

    download_dir = EUROC_LAYOUT.data_dir / "downloads"
    needed_archives = tuple(dict.fromkeys(sequences[scene].archive for scene in pending))
    if download:
        for archive_name in needed_archives:
            archive = archives[archive_name]
            resumable_http_download(
                archive.url,
                download_dir / archive.filename,
                expected_size=archive.size,
                expected_digest=archive.sha256,
                response_validator=_validate_zip_response,
            )
    for archive_name in needed_archives:
        archive = archives[archive_name]
        _validate_archive(download_dir / archive.filename, size=archive.size, sha256=archive.sha256)
    for scene in pending:
        prepare_scene(scene, download_dir)
    if delete_files:
        for scene in pending:
            (download_dir / "sequences" / f"{scene}.zip").unlink(missing_ok=True)
        for archive_name in needed_archives:
            (download_dir / archives[archive_name].filename).unlink(missing_ok=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenes", nargs="+", choices=EUROC_LAYOUT.scenes)
    parser.add_argument(
        "--no-download",
        action="store_true",
        help="Use checksum-validated official bundles already staged",
    )
    parser.add_argument(
        "--delete-downloads",
        action="store_true",
        help="Delete validated source archives after successful preparation",
    )
    arguments = parser.parse_args()
    main(
        arguments.scenes,
        download=not arguments.no_download,
        delete_files=arguments.delete_downloads,
    )
