"""Download and exactly reproduce the established iOS-only LaMAR benchmark."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import logging
import re
import shutil
import struct
import tarfile
import zipfile
from collections.abc import Iterable
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pycolmap
import yaml
from PIL import Image

from vidmap.datasets.layouts import get_dataset_layout
from vidmap.datasets.prepare.helpers import filesystem as prep_fs
from vidmap.datasets.prepare.helpers.video_common import resumable_http_download
from vidmap.datasets.resources import dataset_asset

logger = logging.getLogger(__name__)
LAMAR_LAYOUT = get_dataset_layout("lamar")

_GROUND_TRUTH_MODELS_FILENAME = "lamar_ground_truth_models.tar.gz"
_GROUND_TRUTH_MODELS_SIZE = 11_689_090
_GROUND_TRUTH_MODELS_SHA256 = "60770ead917d936fcef7b0115a8d67cd8044e7dc6655321e7e837923b3736428"
_GROUND_TRUTH_MODELS_URL = (
    f"https://github.com/cvg/vidmap/releases/download/dataset-assets-v2/{_GROUND_TRUTH_MODELS_FILENAME}"
)
_PREPARATION_VERSION = 1
_EXPECTED_GT_IMAGE_COUNTS = {"CAB": 206_161, "HGE": 130_953, "LIN": 117_970}
_EXPECTED_SAMPLE_IMAGE_COUNTS = {"CAB": 36_334, "HGE": 27_289, "LIN": 30_866}
_MODEL_FILES = ("cameras.bin", "images.bin", "points3D.bin")
_SAMPLE_TARGET = re.compile(r"^(ios_.+)_\d{3}-\d+$")


@dataclass(frozen=True)
class SourceArchive:
    scene: str
    filename: str
    size: int
    sha256: str
    url: str


@dataclass(frozen=True)
class RawCamera:
    width: int
    height: int
    params: tuple[float, float, float, float]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _ground_truth_models_archive(downloads: Path, *, download: bool) -> Path:
    path = downloads / _GROUND_TRUTH_MODELS_FILENAME
    if not download and not path.is_file():
        raise FileNotFoundError(f"Missing LaMAR ground-truth archive: {path}")
    return resumable_http_download(
        _GROUND_TRUTH_MODELS_URL,
        path,
        expected_size=_GROUND_TRUTH_MODELS_SIZE,
        expected_digest=_GROUND_TRUTH_MODELS_SHA256,
    )


def source_inventory(profile: str = "sample-300") -> dict[str, tuple[SourceArchive, ...]]:
    """Load checksum-pinned official session archives for the requested profile."""
    payload = json.loads(dataset_asset("lamar", "source_archives.json").read_text(encoding="utf-8"))
    base_url = payload["base_url"]
    inventory = {}
    for scene in LAMAR_LAYOUT.scenes:
        required = _profile_sessions(scene, profile)
        entries = []
        for raw in payload["scenes"][scene]["archives"]:
            if Path(raw["filename"]).stem in required:
                entries.append(
                    SourceArchive(
                        scene=scene,
                        filename=raw["filename"],
                        size=int(raw["size"]),
                        sha256=raw["sha256"],
                        url=base_url.format(scene=scene) + raw["filename"],
                    )
                )
        selected = {Path(entry.filename).stem for entry in entries}
        if selected != required:
            raise ValueError(f"LaMAR {scene} source inventory does not cover {profile}: {sorted(required - selected)}")
        inventory[scene] = tuple(entries)
    return inventory


def parse_sensors(text: str) -> dict[int, RawCamera]:
    """Parse LaMAR iOS per-frame PINHOLE calibration rows."""
    cameras = {}
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line or line.startswith("#"):
            continue
        fields = [value.strip() for value in line.split(",")]
        if len(fields) >= 3 and fields[2] != "camera":
            continue
        if len(fields) != 10 or fields[2:4] != ["camera", "PINHOLE"] or not fields[0].startswith("cam_phone_"):
            raise ValueError(f"Invalid LaMAR sensor row {line_number}: {line!r}")
        timestamp = int(fields[0].removeprefix("cam_phone_"))
        if timestamp in cameras:
            raise ValueError(f"Duplicate LaMAR camera timestamp {timestamp}")
        camera = RawCamera(
            int(fields[4]),
            int(fields[5]),
            tuple(float(value) for value in fields[6:10]),
        )
        if camera.width <= 0 or camera.height <= 0 or not np.isfinite(camera.params).all():
            raise ValueError(f"Invalid LaMAR camera at timestamp {timestamp}")
        cameras[timestamp] = camera
    if not cameras:
        raise ValueError("LaMAR sensors file contains no phone cameras")
    return cameras


def parse_trajectories(text: str) -> dict[int, pycolmap.Rigid3d]:
    """Parse raw rig-to-world poses and return camera-from-world poses."""
    poses = {}
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line or line.startswith("#"):
            continue
        fields = [value.strip() for value in line.split(",")]
        if len(fields) != 9:
            raise ValueError(f"Invalid LaMAR trajectory row {line_number}: {line!r}")
        timestamp = int(fields[0])
        if timestamp in poses:
            raise ValueError(f"Duplicate LaMAR trajectory timestamp {timestamp}")
        values = np.asarray([float(value) for value in fields[2:]], dtype=np.float64)
        if not np.isfinite(values).all() or np.linalg.norm(values[:4]) <= np.finfo(np.float64).eps:
            raise ValueError(f"Invalid LaMAR pose at timestamp {timestamp}")
        qw, qx, qy, qz, tx, ty, tz = values
        world_from_camera = pycolmap.Rigid3d(pycolmap.Rotation3d([qx, qy, qz, qw]), [tx, ty, tz])
        poses[timestamp] = world_from_camera.inverse()
    if not poses:
        raise ValueError("LaMAR trajectories file contains no poses")
    return poses


def rotate_camera(camera: RawCamera, rotations: int) -> RawCamera:
    """Apply the LaMAR image-rotation convention to a raw camera calibration."""
    rotations = (-rotations) % 4
    width, height = camera.width, camera.height
    fx, fy, cx, cy = camera.params
    if rotations == 0:
        return camera
    if rotations == 1:
        cx, cy = cy, width - cx
    elif rotations == 2:
        cx, cy = width - cx, height - cy
    else:
        cx, cy = height - cy, cx
    if rotations % 2:
        return RawCamera(height, width, (fy, fx, cx, cy))
    return RawCamera(width, height, (fx, fy, cx, cy))


def infer_rotation(camera: RawCamera, target: pycolmap.Camera) -> int:
    """Infer the unique image rotation that maps raw calibration to the GT camera."""
    matches = []
    for rotations in range(4):
        candidate = rotate_camera(camera, rotations)
        if (candidate.width, candidate.height) == (target.width, target.height) and np.allclose(
            candidate.params, target.params, rtol=0.0, atol=1e-9
        ):
            matches.append(rotations)
    if len(matches) != 1:
        raise ValueError(f"Could not uniquely match LaMAR camera {camera} to {target}; rotations={matches}")
    return matches[0]


def generate_testsets(scene: str, image_ids: set[int]) -> dict[str, dict[str, list[int]]]:
    """Reproduce the public sample-300 benchmark set."""
    with gzip.open(dataset_asset("lamar", "ground_truth_subsequences.json.gz"), "rt", encoding="utf-8") as file:
        payload = json.load(file)[scene]
    result: dict[str, dict[str, list[int]]] = {"sample": {}}
    for subsequence, anchors in payload.items():
        if not anchors:
            raise ValueError(f"Empty LaMAR GT subsequence {scene}/{subsequence}")
        chunks = [anchors[index : index + 300] for index in range(0, len(anchors), 300)]
        chunks = [chunk for chunk in chunks if len(chunk) == 300][:20]
        for index, chunk in enumerate(chunks):
            result["sample"][f"{subsequence}-{index}"] = [
                image_id for image_id in range(chunk[0], chunk[-1] + 1) if image_id in image_ids
            ]
    return result


def canonical_testsets(scene: str, profile: str = "sample-300") -> dict[str, bytes]:
    """Read the exact canonical testset YAML bytes for one scene."""
    if profile != "sample-300":
        raise ValueError(f"Unknown LaMAR profile: {profile}")
    return {"sample": dataset_asset("lamar", "testsets", scene, "sample-300.yaml").read_bytes()}


@cache
def _parse_selection(contents: bytes) -> dict[str, list[int]]:
    return yaml.safe_load(contents)


def _sample_mapping(scene: str) -> dict[str, list[int]]:
    mapping = _parse_selection(canonical_testsets(scene)["sample"])
    if not isinstance(mapping, dict):
        raise ValueError(f"Bundled LaMAR sample-300 testset is malformed for {scene}")
    return mapping


def _sample_sessions(scene: str) -> set[str]:
    sessions = set()
    for target in _sample_mapping(scene):
        match = _SAMPLE_TARGET.fullmatch(target)
        if match is None:
            raise ValueError(f"Invalid LaMAR sample-300 target: {scene}/{target}")
        sessions.add(match.group(1))
    return sessions


def _selected_image_names(scene: str, ground_truth: pycolmap.Reconstruction, profile: str = "sample-300") -> set[str]:
    mapping = _parse_selection(next(iter(canonical_testsets(scene, profile).values())))
    image_ids = {image_id for values in mapping.values() for image_id in values}
    by_id = {int(image.image_id): str(image.name) for image in ground_truth.images.values()}
    missing = image_ids - by_id.keys()
    if missing:
        raise ValueError(f"LaMAR {scene} selection references missing image IDs: {sorted(missing)[:3]}")
    names = {by_id[image_id] for image_id in image_ids}
    expected = _EXPECTED_SAMPLE_IMAGE_COUNTS[scene]
    if len(names) != expected:
        raise ValueError(f"LaMAR {scene} {profile} image count mismatch: expected {expected}, got {len(names)}")
    sessions = {Path(name).parent.as_posix() for name in names}
    if sessions != _profile_sessions(scene, profile):
        raise ValueError(f"LaMAR {scene} {profile} session mapping is inconsistent")
    return names


def _ensure_testsets(scenes: Iterable[str], profile: str = "sample-300") -> None:
    """Install benchmark cases with explicit recording and variant references."""
    for scene in scenes:
        selections = {mode: {} for mode in canonical_testsets(scene, profile)}
        for recording in sorted(_profile_sessions(scene, profile)):
            manifest = json.loads((LAMAR_LAYOUT.recording_dir(scene, recording) / "preparation.json").read_text())
            for mode, cases in manifest["testsets"].items():
                for case, image_ids in cases.items():
                    if case in selections[mode]:
                        raise ValueError(f"Duplicate LaMAR benchmark case: {scene}/{case}")
                    selections[mode][case] = {
                        "recording": recording,
                        "variant": "native",
                        "image_ids": image_ids,
                    }
        for mode, canonical in canonical_testsets(scene, profile).items():
            expected = _parse_selection(canonical)
            cases = selections[mode]
            if {case: entry["image_ids"] for case, entry in cases.items()} != expected:
                raise ValueError(f"LaMAR {scene}/{mode} selection differs from the canonical asset")
            contents = yaml.safe_dump({case: cases[case] for case in expected}, sort_keys=False).encode()
            destination = LAMAR_LAYOUT.testset_path(scene, mode)
            if destination.exists() or destination.is_symlink():
                if not destination.is_file() or destination.read_bytes() != contents:
                    raise RuntimeError(f"Existing LaMAR testset differs from the prepared selection: {destination}")
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_suffix(destination.suffix + ".part")
            temporary.write_bytes(contents)
            temporary.replace(destination)


def _extract_ground_truth(scene: str, destination: Path, ground_truth_models: Path) -> None:
    destination.mkdir(parents=True)
    with tarfile.open(ground_truth_models) as archive:
        for filename in _MODEL_FILES:
            member = archive.getmember(f"{scene}/rec/{filename}")
            source = archive.extractfile(member)
            if source is None:
                raise ValueError(f"LaMAR ground-truth release asset is missing {scene}/rec/{filename}")
            with source, (destination / filename).open("xb") as target:
                shutil.copyfileobj(source, target, length=8 * 1024 * 1024)


def _copy_image(bundle: zipfile.ZipFile, member: str, destination: Path, rotations: int) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    if temporary.exists():
        raise RuntimeError(f"Unexpected LaMAR staging artifact: {temporary}")
    with bundle.open(member) as source:
        if rotations:
            operations = (
                Image.Transpose.ROTATE_270,
                Image.Transpose.ROTATE_180,
                Image.Transpose.ROTATE_90,
            )
            with Image.open(source) as image:
                image.transpose(operations[rotations - 1]).save(temporary, format="JPEG")
        else:
            with temporary.open("xb") as target:
                shutil.copyfileobj(source, target, length=8 * 1024 * 1024)
    temporary.rename(destination)


def _recording_testsets(scene, profile, ground_truth):
    result = {recording: {} for recording in _profile_sessions(scene, profile)}
    for mode, contents in canonical_testsets(scene, profile).items():
        cases = _parse_selection(contents)
        if profile == "sample-300" and generate_testsets(scene, set(ground_truth.images))[mode] != cases:
            raise ValueError(f"Canonical LaMAR {scene}/{mode} testset content mismatch")
        for case, image_ids in cases.items():
            recordings = {Path(ground_truth.images[image_id].name).parent.as_posix() for image_id in image_ids}
            if len(recordings) != 1:
                raise ValueError(f"LaMAR case must belong to one recording: {scene}/{case}")
            recording = recordings.pop()
            result[recording].setdefault(mode, {})[case] = image_ids
    return result


def _write_recording_model(destination, records, cameras):
    """Preserve source IDs and pose sentinels while flattening recording-local names."""
    destination.mkdir(parents=True)
    selected_cameras = sorted({values[-1] for values in records.values()})
    with (destination / "cameras.bin").open("xb") as stream:
        stream.write(struct.pack("<Q", len(selected_cameras)))
        for camera_id in selected_cameras:
            camera = cameras[camera_id]
            stream.write(struct.pack("<iiQQ", camera_id, int(camera.model), camera.width, camera.height))
            stream.write(struct.pack("<4d", *camera.params))
    with (destination / "images.bin").open("xb") as stream:
        stream.write(struct.pack("<Q", len(records)))
        for name, values in records.items():
            pose = values[1:8]
            stream.write(struct.pack("<i7di", values[0], *pose, values[-1]))
            stream.write(f"cam0/{Path(name).name}".encode() + b"\0" + struct.pack("<Q", 0))
    (destination / "points3D.bin").write_bytes(struct.pack("<Q", 0))


def _convert_recording(scene, archive, destination, ground_truth, records, testsets, profile):
    recording = archive.stem
    seen_names = set()
    rotated = 0
    with zipfile.ZipFile(archive) as bundle:
        prefix = f"{recording}/"
        sensors = parse_sensors(bundle.read(prefix + "sensors.txt").decode("utf-8"))
        poses = parse_trajectories(bundle.read(prefix + "trajectories.txt").decode("utf-8"))
        if set(sensors) != set(poses):
            raise ValueError(f"LaMAR {scene}/{recording} sensor/trajectory timestamp mismatch")
        members = set(bundle.namelist())
        for timestamp in poses:
            name = f"{recording}/{timestamp}.jpg"
            if name not in records:
                continue
            gt_image = ground_truth.images[records[name][0]]
            rotations = infer_rotation(sensors[timestamp], gt_image.camera)
            member = prefix + f"raw_data/images/{timestamp}.jpg"
            if member not in members:
                raise ValueError(f"LaMAR archive is missing {member}")
            _copy_image(bundle, member, destination / "native/images/cam0" / Path(name).name, rotations)
            rotated += bool(rotations)
            seen_names.add(name)
    if seen_names != records.keys():
        missing = sorted(records.keys() - seen_names)
        raise ValueError(f"LaMAR {scene}/{recording} omitted {len(missing)} selected images: {missing[:3]}")
    rec_dir = destination / "native/rec"
    _write_recording_model(rec_dir, records, ground_truth.cameras)
    manifest = {
        "dataset": "LaMAR",
        "data_version": "2.2",
        "device": "iOS",
        "ground_truth_asset_sha256": _GROUND_TRUTH_MODELS_SHA256,
        "images": len(seen_names),
        "profile": profile,
        "preparation_version": _PREPARATION_VERSION,
        "recording": recording,
        "variants": {"native": {"rig": None, "force_gt_keyframes": True}},
        "rotated_images": rotated,
        "scene": scene,
        "source_archives": [archive.name],
        "testsets": testsets,
        "model_sha256": {name: _sha256(rec_dir / name) for name in _MODEL_FILES},
    }
    with (destination / "preparation.json").open("x", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2, sort_keys=True)
        file.write("\n")


def _convert_scene(
    scene: str,
    archives: tuple[Path, ...],
    destination: Path,
    ground_truth_models: Path,
    *,
    profile: str = "sample-300",
) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=f".{scene}-", dir=destination) as temporary:
        source_model = Path(temporary) / "rec"
        _extract_ground_truth(scene, source_model, ground_truth_models)
        ground_truth = pycolmap.Reconstruction(source_model)
        if ground_truth.num_images() != _EXPECTED_GT_IMAGE_COUNTS[scene] or ground_truth.num_points3D():
            raise ValueError(f"LaMAR {scene} GT inventory mismatch")
        if any(camera.model != pycolmap.CameraModelId.PINHOLE for camera in ground_truth.cameras.values()):
            raise ValueError("Expected PINHOLE cameras in the LaMAR source model")
        required_names = _selected_image_names(scene, ground_truth, profile)
        testsets = _recording_testsets(scene, profile, ground_truth)
        records = {recording: {} for recording in testsets}
        with (source_model / "images.bin").open("rb") as stream:
            for name, values in _image_records(stream):
                if name in required_names:
                    records[Path(name).parent.as_posix()][name] = values
        for archive in archives:
            recording = archive.stem
            if recording not in testsets:
                raise ValueError(f"Unexpected LaMAR source recording: {recording}")
            target = destination / recording
            logger.info(f"Preparing LaMAR {scene}/{recording}")
            with prep_fs.scene_staging(target) as staging:
                _convert_recording(
                    scene,
                    archive,
                    staging,
                    ground_truth,
                    records[recording],
                    testsets[recording],
                    profile,
                )
                if not lamar_recording_is_prepared(staging, profile=profile, recording=recording):
                    raise RuntimeError(f"LaMAR preparation did not produce a complete recording: {recording}")
                if target.exists():
                    raise RuntimeError(f"LaMAR recording destination appeared during preparation: {target}")
                staging.rename(target)


def lamar_recording_is_prepared(
    recording_dir: Path, profile: str | None = None, *, recording: str | None = None
) -> bool:
    """Validate one recording independently of external benchmark selection files."""
    recording = recording_dir.name if recording is None else recording
    manifest_path = recording_dir / "preparation.json"
    rec_dir = recording_dir / "native/rec"
    if not manifest_path.is_file() or not prep_fs.complete_colmap_model(rec_dir):
        return False
    manifest = json.loads(manifest_path.read_text())
    if "preparation_version" not in manifest or manifest["preparation_version"] != _PREPARATION_VERSION:
        return False
    scene = manifest["scene"]
    if (
        scene not in LAMAR_LAYOUT.scenes
        or manifest["recording"] != recording
        or manifest["dataset"] != "LaMAR"
        or manifest["data_version"] != "2.2"
        or manifest["device"] != "iOS"
        or manifest["ground_truth_asset_sha256"] != _GROUND_TRUTH_MODELS_SHA256
        or (profile is not None and manifest["profile"] != profile)
        or manifest["source_archives"] != [f"{recording}.zip"]
        or manifest["variants"] != {"native": {"rig": None, "force_gt_keyframes": True}}
        or recording not in _profile_sessions(scene, manifest["profile"])
    ):
        return False
    if manifest["model_sha256"] != {name: _sha256(rec_dir / name) for name in _MODEL_FILES}:
        return False
    canonical = {
        mode: _parse_selection(contents) for mode, contents in canonical_testsets(scene, manifest["profile"]).items()
    }
    if set(manifest["testsets"]) != set(canonical):
        return False
    image_ids = set()
    for mode, cases in manifest["testsets"].items():
        if not cases or any(
            case not in canonical[mode] or ids != canonical[mode][case] for case, ids in cases.items()
        ):
            return False
        image_ids.update(image_id for ids in cases.values() for image_id in ids)
    with (rec_dir / "images.bin").open("rb") as stream:
        records = dict(_image_records(stream))
    if {values[0] for values in records.values()} != image_ids or manifest["images"] != len(records):
        return False
    images_dir = recording_dir / "native/images"
    names = {path.relative_to(images_dir).as_posix() for path in images_dir.rglob("*.jpg") if path.is_file()}
    return names == records.keys() and all(Path(name).parent.as_posix() == "cam0" for name in records)


def lamar_scene_is_prepared(dataset_root: Path, scene: str, profile: str = "sample-300") -> bool:
    """Return whether all recordings selected by one benchmark group are ready."""
    return all(
        lamar_recording_is_prepared(dataset_root / scene / recording, profile=profile)
        for recording in _profile_sessions(scene, profile)
    )


def _pending_recordings(scenes, profile):
    pending = {}
    for scene in scenes:
        missing = []
        for recording in sorted(_profile_sessions(scene, profile)):
            destination = LAMAR_LAYOUT.recording_dir(scene, recording)
            if destination.exists() or destination.is_symlink():
                if not lamar_recording_is_prepared(destination, profile=profile):
                    raise RuntimeError(f"LaMAR recording destination is partial or incompatible: {destination}")
            else:
                missing.append(recording)
        if missing:
            pending[scene] = set(missing)
    return pending


def _profile_sessions(scene: str, profile: str) -> set[str]:
    if profile != "sample-300":
        raise ValueError(f"Unknown LaMAR profile: {profile}")
    return _sample_sessions(scene)


def _image_records(stream):
    """Read COLMAP image headers without retaining 2D observations."""
    for _ in range(struct.unpack("<Q", stream.read(8))[0]):
        values = struct.unpack("<i7di", stream.read(64))
        name = bytearray()
        while (character := stream.read(1)) != b"\0":
            if not character:
                raise ValueError("Truncated COLMAP image name")
            name.extend(character)
        count = struct.unpack("<Q", stream.read(8))[0]
        stream.seek(count * 24, 1)
        yield name.decode(), values


def main(
    scenes: Iterable[str] | None = None,
    *,
    download: bool = True,
    delete_files: bool = False,
    profile: str = "sample-300",
) -> None:
    """Download official iOS sessions and prepare selected or all LaMAR scenes."""
    selected = LAMAR_LAYOUT.select_scenes(scenes)
    pending = _pending_recordings(selected, profile)
    if not pending:
        _ensure_testsets(selected, profile)
        return
    inventory = source_inventory(profile)
    downloads = LAMAR_LAYOUT.data_dir / "downloads"
    downloads.mkdir(parents=True, exist_ok=True)
    ground_truth_models = _ground_truth_models_archive(downloads, download=download)
    for scene, recordings in pending.items():
        archive_paths = []
        for source in inventory[scene]:
            if Path(source.filename).stem not in recordings:
                continue
            path = downloads / scene / source.filename
            if download:
                resumable_http_download(
                    source.url,
                    path,
                    expected_size=source.size,
                    expected_digest=source.sha256,
                )
            elif not path.is_file() or path.stat().st_size != source.size or _sha256(path) != source.sha256:
                raise FileNotFoundError(f"Missing checksum-valid LaMAR archive: {path}")
            archive_paths.append(path)
        _convert_scene(
            scene,
            tuple(archive_paths),
            LAMAR_LAYOUT.scene_dir(scene),
            ground_truth_models,
            profile=profile,
        )
        _ensure_testsets((scene,), profile)
        if delete_files:
            for path in archive_paths:
                path.unlink()
            scene_downloads = downloads / scene
            if not any(scene_downloads.iterdir()):
                scene_downloads.rmdir()
    _ensure_testsets(selected, profile)
    if delete_files:
        ground_truth_models.unlink()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenes", nargs="+", choices=LAMAR_LAYOUT.scenes)
    parser.add_argument("--profile", choices=["sample-300"], default="sample-300")
    parser.add_argument(
        "--no-download",
        action="store_true",
        help="Use checksum-valid archives already in downloads/",
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
        profile=arguments.profile,
    )
