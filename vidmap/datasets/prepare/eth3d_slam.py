"""Download and prepare the ground-truth ETH3D-SLAM monocular benchmark."""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pycolmap
import yaml
from PIL import Image

from vidmap.datasets.base import CANONICAL_UNPOSED_IMAGES_FILENAME, DatasetLayout
from vidmap.datasets.eth3d_slam import ETH3D_SLAM_MANIFEST
from vidmap.datasets.layouts import get_dataset_layout
from vidmap.datasets.prepare import filesystem as prep_fs
from vidmap.datasets.prepare.timestamp_matching import assign_images_to_timestamp_ids
from vidmap.datasets.prepare.video_common import resumable_http_download, safe_extract_zip
from vidmap.datasets.resources import dataset_asset

_URL_TEMPLATE = "https://www.eth3d.net/data/slam/datasets/{scene}_mono.zip"
_SOURCE_RECORD = ".source_archive.json"
ETH3D_SLAM_PREPARATION_VERSION = 1
ETH3D_SLAM_LAYOUT = get_dataset_layout("eth3d_slam")


@dataclass(frozen=True)
class SourceArchive:
    """One checksum-pinned archive from the official monocular release."""

    scene: str
    split: str
    size: int
    http_etag: str
    sha256: str

    @property
    def filename(self) -> str:
        return f"{self.scene}_mono.zip"

    @property
    def url(self) -> str:
        return _URL_TEMPLATE.format(scene=self.scene)

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


def source_inventory() -> tuple[SourceArchive, ...]:
    """Read and validate the complete ground-truth benchmark inventory."""
    with dataset_asset("eth3d_slam", "mono_archives.tsv").open(encoding="utf-8", newline="") as file:
        rows = tuple(csv.DictReader(file, delimiter="\t"))
    unordered = tuple(
        SourceArchive(
            scene=row["scene"],
            split=row["split"],
            size=int(row["size"]),
            http_etag=row["http_etag"],
            sha256=row["sha256"],
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


def _scene_is_complete(layout: DatasetLayout, scene: str) -> bool:
    images_dir = layout.images_dir(scene)
    rec_dir = layout.reconstruction_dir(scene)
    if not images_dir.is_dir() or not prep_fs.complete_colmap_model(rec_dir):
        return False
    try:
        reconstruction = pycolmap.Reconstruction(rec_dir)
        testsets = layout.read_testsets(scene, "all")
        preparation = json.loads((layout.scene_dir(scene) / "preparation.json").read_text(encoding="utf-8"))
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError):
        return False
    image_ids = sorted(reconstruction.images)
    names = {image.name for image in reconstruction.images.values()}
    unposed = _read_unposed_names(layout.scene_dir(scene))
    try:
        archive = _source_archive(scene)
    except ValueError:
        archive = None
    expected_unposed = None
    if archive is not None:
        expected_unposed = set() if archive.gt_available else names
    return (
        bool(reconstruction.cameras)
        and bool(image_ids)
        and testsets == {"0": image_ids}
        and unposed is not None
        and set(unposed) <= names
        and (expected_unposed is None or set(unposed) == expected_unposed)
        and isinstance(preparation, Mapping)
        and preparation.get("preparation_version") == ETH3D_SLAM_PREPARATION_VERSION
        and (archive is None or preparation.get("source") == archive.record())
        and (archive is None or preparation.get("gt_available") is archive.gt_available)
        and all(
            (images_dir / image.name).is_file() and (images_dir / image.name).suffix.lower() in prep_fs.IMAGE_SUFFIXES
            for image in reconstruction.images.values()
        )
    )


def _pending_scenes(layout: DatasetLayout, scenes: tuple[str, ...] | None = None) -> tuple[str, ...]:
    pending = []
    for scene in layout.scenes if scenes is None else scenes:
        destination = layout.scene_dir(scene)
        testset = layout.testset_path(scene, "all")
        shutil.rmtree(layout.data_dir / f".{scene}.preparing", ignore_errors=True)
        testset.with_name(f".{testset.name}.preparing").unlink(missing_ok=True)
        if destination.exists() or testset.exists():
            if not _scene_is_complete(layout, scene):
                raise RuntimeError(f"ETH3D-SLAM scene destination is partial or incompatible: {destination}")
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


def _source_images(source: Path) -> tuple[tuple[Path, float], ...]:
    rgb_dir = source / "rgb"
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


def _convert_scene(source: Path, destination: Path, *, gt_available: bool = True) -> list[int]:
    calibration = _read_calibration(source / "calibration.txt")
    images = _source_images(source)
    width, height = _image_dimensions(images)
    assigned_poses = None
    translations = quaternions = None
    if gt_available:
        timestamps, translations, quaternions = _read_groundtruth(source / "groundtruth.txt")
        assigned_poses = assign_images_to_timestamp_ids((timestamp for _, timestamp in images), timestamps)

    reconstruction = pycolmap.Reconstruction()
    camera = pycolmap.Camera(model="PINHOLE", width=width, height=height, params=calibration, camera_id=0)
    reconstruction.add_camera_with_trivial_rig(camera)
    images_dir = destination / "images"
    rec_dir = destination / "rec"
    images_dir.mkdir(parents=True)
    rec_dir.mkdir()

    unposed_names = []
    for image_id, (source_image, _) in enumerate(images):
        image = pycolmap.Image(camera_id=0, name=source_image.name, image_id=image_id)
        if assigned_poses is None:
            camera_from_world = pycolmap.Rigid3d()
            unposed_names.append(source_image.name)
        else:
            pose_id = assigned_poses[image_id]
            rotation = pycolmap.Rotation3d(quaternions[pose_id])
            camera_from_world = pycolmap.Rigid3d(rotation=rotation, translation=translations[pose_id]).inverse()
        reconstruction.add_image_with_trivial_frame(image, camera_from_world)
        shutil.copy2(source_image, images_dir / source_image.name)

    reconstruction.write_binary(rec_dir)
    with (destination / CANONICAL_UNPOSED_IMAGES_FILENAME).open("w", encoding="utf-8") as file:
        json.dump({"image_names": unposed_names}, file, indent=2)
        file.write("\n")
    return sorted(reconstruction.images)


def _prepare_scene(
    layout: DatasetLayout,
    scene: str,
    source: Path,
    archive: SourceArchive | None = None,
) -> None:
    gt_available = True if archive is None else archive.gt_available
    destination = layout.scene_dir(scene)
    testset = layout.testset_path(scene, "all")
    temporary_testset = testset.with_name(f".{testset.name}.preparing")
    testset.parent.mkdir(parents=True, exist_ok=True)
    temporary_testset.unlink(missing_ok=True)
    try:
        with prep_fs.scene_staging(destination) as staging:
            image_ids = _convert_scene(source, staging, gt_available=gt_available)
            with (staging / "preparation.json").open("w", encoding="utf-8") as file:
                json.dump(
                    {
                        "gt_available": gt_available,
                        "preparation_version": ETH3D_SLAM_PREPARATION_VERSION,
                        "source": None if archive is None else archive.record(),
                    },
                    file,
                    indent=2,
                    sort_keys=True,
                )
                file.write("\n")
            with temporary_testset.open("x", encoding="utf-8") as file:
                yaml.safe_dump({0: image_ids}, file, sort_keys=False, default_flow_style=True)
            if not prep_fs.complete_colmap_model(staging / "rec") or not image_ids:
                raise RuntimeError(f"ETH3D-SLAM conversion did not produce a complete scene: {staging}")
            if testset.exists() or testset.is_symlink():
                raise RuntimeError(f"ETH3D-SLAM testset destination appeared during preparation: {testset}")
            staging.rename(destination)
            try:
                temporary_testset.rename(testset)
            except BaseException:
                shutil.rmtree(destination, ignore_errors=True)
                raise
    finally:
        temporary_testset.unlink(missing_ok=True)


def _extract_archive(archive: SourceArchive, downloaded: Path, root: Path) -> Path:
    destination = root / archive.scene
    expected_record = archive.record()
    record_path = destination / _SOURCE_RECORD
    if destination.is_dir():
        try:
            record = json.loads(record_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            record = None
        if record == expected_record:
            _read_calibration(destination / "calibration.txt")
            _source_images(destination)
            return destination
        shutil.rmtree(destination)

    staging = root / f".{archive.scene}.extracting"
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True)
    try:
        safe_extract_zip(downloaded, staging)
        extracted = staging / archive.scene
        if not extracted.is_dir():
            raise RuntimeError(f"ETH3D-SLAM archive {downloaded} did not contain {archive.scene}/")
        _read_calibration(extracted / "calibration.txt")
        _source_images(extracted)
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


def main(delete_files: bool = False, scenes: Iterable[str] | None = None) -> None:
    layout = ETH3D_SLAM_LAYOUT
    layout.data_dir.mkdir(parents=True, exist_ok=True)
    selected = layout.select_scenes(scenes)
    pending = _pending_scenes(layout, selected)
    for scene in pending:
        archive = _source_archive(scene)
        print(f"Preparing ETH3D-SLAM scene {scene} ({archive.split})")
        downloaded = resumable_http_download(
            archive.url,
            layout.data_dir / "downloads" / archive.filename,
            expected_size=archive.size,
            expected_digest=archive.sha256,
        )
        source = _extract_archive(archive, downloaded, layout.data_dir / ".eth3d-slam-source")
        _prepare_scene(layout, scene, source, archive)
        if not _scene_is_complete(layout, scene):
            raise RuntimeError(f"ETH3D-SLAM preparation did not produce a complete scene: {scene}")
        shutil.rmtree(source)
        if delete_files:
            downloaded.unlink()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare the ground-truth ETH3D-SLAM monocular benchmark")
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
    main(delete_files=arguments.delete_downloads, scenes=arguments.scenes)
