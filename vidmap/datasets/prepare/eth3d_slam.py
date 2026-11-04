"""Download and prepare the ETH3D-SLAM mono and stereo benchmarks."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pycolmap
import yaml
from PIL import Image

from vidmap.datasets.base import CANONICAL_UNPOSED_IMAGES_FILENAME, DatasetLayout, PreparedTestset
from vidmap.datasets.eth3d_slam import ETH3D_SLAM_MANIFEST, ETH3D_SLAM_PREPARATION_VERSION
from vidmap.datasets.layouts import get_dataset_layout
from vidmap.datasets.prepare.helpers import filesystem as prep_fs
from vidmap.datasets.prepare.helpers.rig_capture import camera_options, frame_image_ids, write_frame_table
from vidmap.datasets.prepare.helpers.ros_benchmarks import timestamp_ns
from vidmap.datasets.prepare.helpers.video_common import resumable_http_download, safe_extract_zip
from vidmap.datasets.resources import dataset_asset

_URL_TEMPLATE = "https://www.eth3d.net/data/slam/datasets/{scene}_{modality}.zip"
_SOURCE_RECORD = ".source_archive.json"
ETH3D_SLAM_LAYOUT = get_dataset_layout("eth3d_slam")


@dataclass(frozen=True)
class SourceArchive:
    """One checksum-pinned archive from the official release."""

    scene: str
    split: str
    size: int
    http_etag: str
    sha256: str
    modality: str = "mono"

    @property
    def filename(self) -> str:
        return f"{self.scene}_{self.modality}.zip"

    @property
    def url(self) -> str:
        return _URL_TEMPLATE.format(scene=self.scene, modality=self.modality)

    @property
    def gt_available(self) -> bool:
        return self.split == "training"

    def record(self) -> dict[str, object]:
        return {
            "filename": self.filename,
            "http_etag": self.http_etag,
            "sha256": self.sha256,
            "size": self.size,
            "split": self.split,
            "url": self.url,
        }


def source_inventory(*, modality="mono") -> tuple[SourceArchive, ...]:
    """Read and validate the complete ground-truth benchmark inventory."""
    if modality not in {"mono", "stereo"}:
        raise ValueError(f"Unknown ETH3D-SLAM modality: {modality}")
    with dataset_asset("eth3d_slam", f"{modality}_archives.tsv").open(encoding="utf-8", newline="") as file:
        rows = tuple(csv.DictReader(file, delimiter="\t"))
    unordered = tuple(
        SourceArchive(
            scene=row["scene"],
            split=row["split"],
            size=int(row["size"]),
            http_etag=row["http_etag"],
            sha256=row["sha256"],
            modality=modality,
        )
        for row in rows
        if row["split"] == "training"
    )
    by_scene = {archive.scene: archive for archive in unordered}
    if len(by_scene) != len(unordered) or set(by_scene) != set(ETH3D_SLAM_MANIFEST.scenes):
        raise ValueError("ETH3D-SLAM source inventory must exactly match the registered scenes")
    archives = tuple(by_scene[scene] for scene in ETH3D_SLAM_MANIFEST.scenes)
    if any(
        archive.split != "training"
        or archive.size <= 0
        or not re.fullmatch(r"[0-9a-f]{64}", archive.sha256)
        or not archive.http_etag
        for archive in archives
    ):
        raise ValueError("ETH3D-SLAM source inventory contains an invalid archive record")
    return archives


def _source_archive(scene: str) -> SourceArchive:
    try:
        return next(archive for archive in source_inventory() if archive.scene == scene)
    except StopIteration as error:
        raise ValueError(f"Unknown ETH3D-SLAM scene {scene!r}") from error


def _read_unposed_names(scene_dir: Path) -> tuple[str, ...] | None:
    path = scene_dir / CANONICAL_UNPOSED_IMAGES_FILENAME
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        names = payload["image_names"]
    except (OSError, KeyError, TypeError, json.JSONDecodeError):
        return None
    if (
        not isinstance(names, list)
        or not all(isinstance(name, str) for name in names)
        or len(names) != len(set(names))
    ):
        return None
    return tuple(names)


def _scene_is_complete(layout: DatasetLayout, scene: str, *, mode="mono") -> bool:
    images_dir = layout.images_dir(scene, "0", "native")
    rec_dir = layout.reconstruction_dir(scene, "0", "native")
    if not images_dir.is_dir() or not prep_fs.complete_colmap_model(rec_dir):
        return False
    try:
        reconstruction = pycolmap.Reconstruction(rec_dir)
        testsets = layout.read_testsets(scene, "mono")
        preparation = json.loads((layout.recording_dir(scene, "0") / "preparation.json").read_text(encoding="utf-8"))
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError):
        return False
    if mode in {"stereo", "all"}:
        rig = preparation["variants"]["native"]["rig"]
        if rig is None or len(rig["cameras"]) != 2 or not (images_dir / "frames.csv").is_file():
            return False
        if not layout.testset_path(scene, "stereo").is_file():
            return False
        case = PreparedTestset("0", "native", frame_image_ids(layout.input_dir(scene, "0", "native")))
        if layout.read_testsets(scene, "stereo") != {"0": case}:
            return False
    image_ids = sorted(iid for iid, image in reconstruction.images.items() if image.name.startswith("cam0/"))
    names = {image.name for image in reconstruction.images.values()}
    unposed = _read_unposed_names(layout.input_dir(scene, "0", "native"))
    try:
        archive = _source_archive(scene)
    except ValueError:
        archive = None
    return (
        bool(reconstruction.cameras)
        and bool(image_ids)
        and testsets == {"0": PreparedTestset("0", "native", tuple(image_ids))}
        and unposed is not None
        and set(unposed) <= names
        and isinstance(preparation, Mapping)
        and preparation.get("preparation_version") == ETH3D_SLAM_PREPARATION_VERSION
        and set(preparation["variants"]) == {"native"}
        and preparation["variants"]["native"]["force_gt_keyframes"]
        and (archive is None or preparation.get("source") == archive.record())
        and (archive is None or preparation.get("gt_available") is archive.gt_available)
        and all(
            (images_dir / image.name).is_file() and (images_dir / image.name).suffix.lower() in prep_fs.IMAGE_SUFFIXES
            for image in reconstruction.images.values()
        )
    )


def _pending_scenes(layout: DatasetLayout, scenes: tuple[str, ...] | None = None, *, mode="mono") -> tuple[str, ...]:
    pending = []
    for scene in layout.scenes if scenes is None else scenes:
        destination = layout.recording_dir(scene, "0")
        testset = layout.testset_path(scene, "mono")
        legacy = layout.testset_path(scene, "all")
        if legacy.is_file() and not testset.exists():
            legacy.rename(testset)
        previous = destination.with_name(f".{destination.name}.preparing") / "previous"
        if previous.exists() and not destination.exists():
            previous.rename(destination)
        shutil.rmtree(destination.with_name(f".{destination.name}.preparing"), ignore_errors=True)
        testset.with_name(f".{testset.name}.preparing").unlink(missing_ok=True)
        if destination.exists() or testset.exists():
            if not _scene_is_complete(layout, scene):
                raise RuntimeError(f"ETH3D-SLAM scene destination is partial or incompatible: {destination}")
            if _scene_is_complete(layout, scene, mode=mode):
                continue
        pending.append(scene)
    return tuple(pending)


def _numeric_rows(path: Path, columns: int, label: str) -> np.ndarray:
    rows: list[list[float]] = []
    try:
        with path.open(encoding="utf-8") as file:
            for line_number, raw_line in enumerate(file, start=1):
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue
                fields = line.split()
                if len(fields) != columns:
                    raise ValueError(
                        f"Invalid {label} row at {path}:{line_number}: expected {columns} columns, got {len(fields)}"
                    )
                try:
                    row = [float(value) for value in fields]
                except ValueError as error:
                    raise ValueError(f"Invalid numeric value in {label} row at {path}:{line_number}") from error
                if not np.isfinite(row).all():
                    raise ValueError(f"Non-finite value in {label} row at {path}:{line_number}")
                rows.append(row)
    except UnicodeDecodeError as error:
        raise ValueError(f"Invalid UTF-8 in {label} file {path}: {error}") from error
    return np.asarray(rows, dtype=np.float64)


def _read_calibration(path: Path) -> tuple[float, float, float, float]:
    rows = _numeric_rows(path, 4, "ETH3D-SLAM calibration")
    if rows.shape != (1, 4) or rows[0, 0] <= 0 or rows[0, 1] <= 0:
        raise ValueError(f"ETH3D-SLAM calibration {path} must contain one valid PINHOLE parameter row")
    return tuple(rows[0])


def _read_groundtruth(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows = _numeric_rows(path, 8, "ETH3D-SLAM ground truth")
    if not len(rows):
        raise ValueError(f"ETH3D-SLAM ground truth {path} does not contain any poses")
    timestamps = rows[:, 0]
    if np.any(np.diff(timestamps) <= 0):
        raise ValueError(f"ETH3D-SLAM ground-truth timestamps must be strictly increasing: {path}")
    quaternions = rows[:, 4:8]
    if np.any(np.linalg.norm(quaternions, axis=1) <= np.finfo(np.float64).eps):
        raise ValueError(f"ETH3D-SLAM ground truth contains a zero quaternion: {path}")
    return timestamps, rows[:, 1:4], quaternions


def _source_images(source: Path, directory="rgb") -> tuple[tuple[Path, float], ...]:
    rgb_dir = source / directory
    if not rgb_dir.is_dir():
        raise RuntimeError(f"ETH3D-SLAM source scene is missing its RGB directory: {rgb_dir}")
    images = []
    for path in rgb_dir.iterdir():
        if not path.is_file() or path.suffix.lower() not in prep_fs.IMAGE_SUFFIXES:
            continue
        try:
            timestamp = float(path.stem)
        except ValueError as error:
            raise ValueError(f"ETH3D-SLAM image name must be a numeric timestamp: {path}") from error
        if not np.isfinite(timestamp):
            raise ValueError(f"ETH3D-SLAM image timestamp must be finite: {path}")
        images.append((path, timestamp))
    images.sort(key=lambda item: (item[1], item[0].name))
    if not images:
        raise RuntimeError(f"ETH3D-SLAM source scene contains no supported images: {rgb_dir}")
    if len({timestamp for _, timestamp in images}) != len(images):
        raise ValueError(f"ETH3D-SLAM source scene contains duplicate image timestamps: {rgb_dir}")
    return tuple(images)


def _image_dimensions(images: Sequence[tuple[Path, float]]) -> tuple[int, int]:
    dimensions = []
    for path, _ in images:
        try:
            with Image.open(path) as image:
                dimensions.append(image.size)
        except OSError as error:
            raise RuntimeError(f"Could not read ETH3D-SLAM image {path}: {error}") from error
    if len(set(dimensions)) != 1:
        raise ValueError("ETH3D-SLAM images assigned to one camera must have identical dimensions")
    width, height = dimensions[0]
    if width <= 0 or height <= 0:
        raise ValueError("ETH3D-SLAM images must have positive dimensions")
    return width, height


def ground_truth_poses_at_timestamps(
    image_times, timestamps, translations, quaternions, *, max_interpolation_timespan=1 / 75
):
    """Interpolate world-from-camera GT using the official ETH3D support rule.

    ETH3D excludes extrapolation and brackets wider than its capture-specific limit.
    In particular, a nearby endpoint does not make a tracking gap valid GT.
    """
    poses = []
    for timestamp in image_times:
        after = int(np.searchsorted(timestamps, timestamp, side="right"))
        if (
            after == 0
            or after == len(timestamps)
            or timestamps[after] - timestamps[after - 1] > max_interpolation_timespan
        ):
            poses.append(None)
            continue
        before = after - 1
        fraction = (timestamp - timestamps[before]) / (timestamps[after] - timestamps[before])
        endpoints = [
            pycolmap.Rigid3d(pycolmap.Rotation3d(quaternions[index]), translations[index]) for index in (before, after)
        ]
        poses.append(pycolmap.Rigid3d.interpolate(*endpoints, fraction))
    return poses


def _convert_scene(source: Path, destination: Path, *, gt_available: bool = True, previous=None) -> list[int]:
    streams = [_source_images(source)]
    if (source / "rgb2").is_dir():
        streams.append(_source_images(source, "rgb2"))
        if [path.name for path, _ in streams[0]] != [path.name for path, _ in streams[1]]:
            raise ValueError(f"ETH3D-SLAM camera streams have different frame names: {source}")
    poses = [None] * len(streams[0])
    if gt_available:
        timestamps, translations, quaternions = _read_groundtruth(source / "groundtruth.txt")
        poses = ground_truth_poses_at_timestamps(
            (timestamp for _, timestamp in streams[0]),
            timestamps,
            translations,
            quaternions,
            max_interpolation_timespan=1 / 11.25 if source.name.startswith("sfm_") else 1 / 75,
        )

    reconstruction = pycolmap.Reconstruction()
    rec_dir = destination / "rec"
    rec_dir.mkdir(parents=True)
    unposed_names, frame_streams = [], []
    for cid, images in enumerate(streams):
        calibration = _read_calibration(source / ("calibration.txt" if cid == 0 else "calibration2.txt"))
        width, height = _image_dimensions(images)
        camera = pycolmap.Camera(model="PINHOLE", width=width, height=height, params=calibration, camera_id=cid)
        reconstruction.add_camera_with_trivial_rig(camera)
        images_dir = destination / "images" / f"cam{cid}"
        images_dir.mkdir(parents=True)
        frame_streams.append({timestamp_ns(path.stem): f"cam{cid}/{path.name}" for path, _ in images})
        for index, (source_image, _) in enumerate(images):
            image_id = len(reconstruction.images)
            image = pycolmap.Image(camera_id=cid, name=f"cam{cid}/{source_image.name}", image_id=image_id)
            pose = poses[index] if cid == 0 else None
            if pose is None:
                unposed_names.append(image.name)
                # COLMAP omits unregistered images when writing; the sidecar removes
                # this placeholder pose on load while preserving the input image.
                reconstruction.add_image_with_trivial_frame(image, pycolmap.Rigid3d())
            else:
                reconstruction.add_image_with_trivial_frame(image, pose.inverse())
            existing = None if previous is None else previous / "images" / image.name
            if existing is not None and existing.is_file():
                os.link(existing, images_dir / source_image.name)
            else:
                shutil.copy2(source_image, images_dir / source_image.name)

    reconstruction.write_binary(rec_dir)
    if len(streams) > 1:
        write_frame_table(destination / "images", frame_streams)
    with (destination / CANONICAL_UNPOSED_IMAGES_FILENAME).open("w", encoding="utf-8") as file:
        json.dump({"image_names": unposed_names}, file, indent=2)
        file.write("\n")
    return list(range(len(streams[0])))


def _rig_calibration(source, reconstruction):
    cam1_from_cam0 = None
    if len(reconstruction.cameras) > 1:
        transform = np.loadtxt(source / "extrinsics_1_2.txt")
        if (
            transform.shape != (3, 4)
            or not np.isfinite(transform).all()
            or not np.allclose(transform[:, :3].T @ transform[:, :3], np.eye(3), atol=1e-6)
            or not np.isclose(np.linalg.det(transform[:, :3]), 1)
        ):
            raise ValueError("Invalid ETH3D camera transform")
        # The official matrix maps the second camera into the reference camera.
        cam1_from_cam0 = pycolmap.Rigid3d(transform).inverse()
    cameras = []
    for cid, camera in sorted(reconstruction.cameras.items()):
        # Source calibration uses integer pixel centres; COLMAP uses half-integer centres.
        camera.params[2:4] += 0.5
        cameras.append(camera_options(camera, cam1_from_cam0 if cid else None))
    return dict(cameras=cameras, frames="frames.csv" if len(cameras) > 1 else None)


def _prepare_scene(
    layout: DatasetLayout,
    scene: str,
    source: Path,
    archive: SourceArchive | None = None,
) -> None:
    gt_available = True if archive is None else archive.gt_available
    destination = layout.recording_dir(scene, "0")
    with prep_fs.scene_staging(destination, replace=destination.exists()) as temporary:
        staging = temporary / "recording"
        image_ids = _convert_scene(
            source,
            staging / "native",
            gt_available=gt_available,
            previous=destination / "native" if destination.exists() else None,
        )
        reconstruction = pycolmap.Reconstruction(staging / "native" / "rec")
        rig = _rig_calibration(source, reconstruction)
        preparation = {
            "gt_available": gt_available,
            "preparation_version": ETH3D_SLAM_PREPARATION_VERSION,
            "variants": {"native": {"rig": rig, "force_gt_keyframes": True}},
            "source": None if archive is None else archive.record(),
        }
        if len(rig["cameras"]) > 1:
            preparation["stereo_source"] = json.loads((source / _SOURCE_RECORD).read_text())[1]
        (staging / "preparation.json").write_text(json.dumps(preparation, indent=2, sort_keys=True) + "\n")
        cases = {"mono": PreparedTestset("0", "native", tuple(image_ids))}
        if len(rig["cameras"]) > 1:
            cases["stereo"] = PreparedTestset("0", "native", frame_image_ids(staging / "native"))
        pending_testsets = {}
        for mode, case in cases.items():
            path = layout.testset_path(scene, mode)
            if path.exists():
                if layout.read_testsets(scene, mode) != {"0": case}:
                    raise ValueError(f"Existing testset differs from prepared scene: {path}")
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary_testset = temporary / f"{mode}.yaml"
            temporary_testset.write_text(yaml.safe_dump({0: case.as_dict()}, sort_keys=False, default_flow_style=True))
            pending_testsets[path] = temporary_testset
        if not prep_fs.complete_colmap_model(staging / "native" / "rec") or not image_ids:
            raise RuntimeError(f"ETH3D-SLAM conversion did not produce a complete scene: {staging}")
        if destination.exists():
            destination.rename(temporary / "previous")
        staging.rename(destination)
        published = False
        try:
            for path, temporary_testset in pending_testsets.items():
                temporary_testset.replace(path)
            published = True
        finally:
            if not published:
                destination.rename(staging)
                for path in pending_testsets:
                    path.unlink(missing_ok=True)


def _extract_archive(archive: SourceArchive, downloaded: Path, root: Path, *, camera_archives=()) -> Path:
    destination = root / archive.scene
    sources = ((archive, downloaded), *camera_archives)
    expected_record = [item.record() for item, _ in sources]
    record_path = destination / _SOURCE_RECORD
    if destination.is_dir():
        try:
            record = json.loads(record_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            record = None
        if record == expected_record:
            _read_calibration(destination / "calibration.txt")
            _source_images(destination)
            if camera_archives:
                _read_calibration(destination / "calibration2.txt")
                _source_images(destination, "rgb2")
            return destination
        shutil.rmtree(destination)

    staging = root / f".{archive.scene}.extracting"
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True)
    try:
        for _, path in sources:
            safe_extract_zip(path, staging)
        extracted = staging / archive.scene
        if not extracted.is_dir():
            raise RuntimeError(f"ETH3D-SLAM archive {downloaded} did not contain {archive.scene}/")
        _read_calibration(extracted / "calibration.txt")
        _source_images(extracted)
        if camera_archives:
            _read_calibration(extracted / "calibration2.txt")
            _source_images(extracted, "rgb2")
        if archive.gt_available:
            _read_groundtruth(extracted / "groundtruth.txt")
        record_path = extracted / _SOURCE_RECORD
        record_path.write_text(
            json.dumps(expected_record, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        root.mkdir(parents=True, exist_ok=True)
        extracted.rename(destination)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return destination


def main(delete_files: bool = False, scenes: Iterable[str] | None = None, *, mode="mono") -> None:
    if mode not in {"mono", "stereo", "all"}:
        raise ValueError(f"Unknown ETH3D-SLAM mode: {mode}")
    layout = ETH3D_SLAM_LAYOUT
    layout.data_dir.mkdir(parents=True, exist_ok=True)
    selected = layout.select_scenes(scenes)
    pending = _pending_scenes(layout, selected, mode=mode)
    extra_sources = (
        {archive.scene: archive for archive in source_inventory(modality="stereo")}
        if mode in {"stereo", "all"}
        else {}
    )
    for scene in pending:
        archive = _source_archive(scene)
        sources = [archive, extra_sources[scene]] if mode in {"stereo", "all"} else [archive]
        downloaded = [
            resumable_http_download(
                item.url,
                layout.data_dir / "downloads" / item.filename,
                expected_size=item.size,
                expected_digest=item.sha256,
            )
            for item in sources
        ]
        source = _extract_archive(
            archive,
            downloaded[0],
            layout.data_dir / ".eth3d-slam-source",
            camera_archives=tuple(zip(sources[1:], downloaded[1:], strict=True)),
        )
        _prepare_scene(layout, scene, source, archive)
        if not _scene_is_complete(layout, scene, mode=mode):
            raise RuntimeError(f"ETH3D-SLAM preparation did not produce a complete scene: {scene}")
        shutil.rmtree(source)
        if delete_files:
            for path in downloaded:
                path.unlink()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare the ETH3D-SLAM mono or stereo benchmark")
    parser.add_argument("--mode", choices=("mono", "stereo", "all"), default="mono")
    parser.add_argument(
        "--scenes",
        nargs="+",
        choices=ETH3D_SLAM_LAYOUT.scenes,
        help="Scenes to prepare (default: all 55 benchmark scenes)",
    )
    parser.add_argument(
        "--delete-downloads",
        action="store_true",
        help="Delete validated source archives after successful preparation",
    )
    return parser


if __name__ == "__main__":
    arguments = build_parser().parse_args()
    main(delete_files=arguments.delete_downloads, scenes=arguments.scenes, mode=arguments.mode)
