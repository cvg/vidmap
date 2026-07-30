"""Convert CroCoDL captures to the VidMap dataset layout."""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypeAlias

import numpy as np
import pycolmap
import yaml
from tqdm import tqdm

from vidmap.datasets.base import validate_path_component
from vidmap.datasets.crocodl import benchmark_scene_name, parse_benchmark_scene, validate_ios_session
from vidmap.datasets.layouts import get_dataset_layout
from vidmap.datasets.prepare.video_common import resumable_http_download, safe_extract_zip
from vidmap.datasets.resources import dataset_asset

PoseKey: TypeAlias = tuple[int, str]
TestsetKey: TypeAlias = int | Literal["complete"]
TestsetData: TypeAlias = dict[TestsetKey, dict[str, list[int]]]

CROCODL_LAYOUT = get_dataset_layout("crocodl")
CROCODL_DATA_DIR = CROCODL_LAYOUT.data_dir

_CAMERA_PARAMETER_COUNTS = {
    "PINHOLE": 4,
    "SIMPLE_PINHOLE": 3,
    "RADIAL": 5,
    "SIMPLE_RADIAL": 4,
    "OPENCV": 8,
}
_TESTSET_LENGTHS = (100, 250, 500, 1500)
_SAMPLE_TESTSET = re.compile(r"sample-(\d+)\.yaml")
_HUGGING_FACE_REPOSITORY = "https://huggingface.co/datasets/CroCoDL"
CROCODL_PREPARATION_VERSION = 1
CROCODL_LICENSE = "No dataset license declared by the pinned official Hugging Face repositories"


class CroCoDLFormatError(ValueError):
    """A CroCoDL source file violates the conversion contract."""


@dataclass(frozen=True)
class CsvRow:
    values: tuple[str, ...]
    path: Path
    line_number: int
    session_id: str

    @property
    def source(self) -> str:
        return f"{self.path}:{self.line_number} (session {self.session_id!r})"

    def fail(self, message: str) -> CroCoDLFormatError:
        return CroCoDLFormatError(f"{self.source}: {message}")

    def require(self, count: int, name: str) -> None:
        if len(self.values) < count:
            raise self.fail(f"{name} requires at least {count} fields, found {len(self.values)}")

    def integer(self, index: int, name: str) -> int:
        try:
            return int(self.values[index])
        except ValueError as error:
            raise self.fail(f"invalid {name} {self.values[index]!r}") from error

    def floats(self, values: Sequence[str], name: str) -> tuple[float, ...]:
        try:
            parsed = tuple(float(value) for value in values)
        except ValueError as error:
            raise self.fail(f"invalid {name}") from error
        if not all(np.isfinite(value) for value in parsed):
            raise self.fail(f"{name} must contain only finite values")
        return parsed

    def pose(self, name: str) -> "Pose":
        self.require(9, name)
        qw, qx, qy, qz, tx, ty, tz = self.floats(self.values[2:9], f"{name} pose")
        rotation = (qx, qy, qz, qw)
        if not any(rotation):
            raise self.fail(f"{name} quaternion must be non-zero")
        return Pose(rotation, (tx, ty, tz))


@dataclass(frozen=True)
class CameraSensor:
    sensor_id: str
    model: str
    width: int
    height: int
    parameters: tuple[float, ...]
    source: str


@dataclass(frozen=True)
class Pose:
    rotation_xyzw: tuple[float, float, float, float]
    translation: tuple[float, float, float]

    def as_rigid3d(self) -> pycolmap.Rigid3d:
        return pycolmap.Rigid3d(pycolmap.Rotation3d(self.rotation_xyzw), self.translation)


@dataclass(frozen=True)
class CaptureImage:
    timestamp: int
    sensor_id: str
    relative_path: Path
    source: str

    @property
    def basename(self) -> str:
        return self.relative_path.name


@dataclass(frozen=True)
class ConversionResult:
    session_id: str
    testsets: TestsetData


@dataclass(frozen=True)
class OfficialIosArchive:
    """One checksum-pinned iOS capture archive in the official release."""

    location: str
    filename: str
    size: int
    sha256: str
    revision: str

    @property
    def session_id(self) -> str:
        return Path(self.filename).stem

    @property
    def url(self) -> str:
        return f"{_HUGGING_FACE_REPOSITORY}/{self.location}/resolve/{self.revision}/sessions/{self.filename}"


@dataclass(frozen=True)
class OfficialIosAuxiliaryArchive(OfficialIosArchive):
    """One explicitly excluded iOS localization map/query archive."""

    role: Literal["localization-map", "localization-query"]


def official_ios_inventory() -> tuple[OfficialIosArchive, ...]:
    """Read and validate the complete pinned 64-session iOS inventory."""
    with dataset_asset("crocodl", "ios_archives.tsv").open(encoding="utf-8", newline="") as file:
        rows = tuple(csv.DictReader(file, delimiter="\t"))
    archives = tuple(
        OfficialIosArchive(
            location=validate_path_component(row["location"], "CroCoDL location"),
            filename=validate_path_component(row["filename"], "CroCoDL archive filename"),
            size=int(row["size"]),
            sha256=row["sha256"],
            revision=row["revision"],
        )
        for row in rows
    )
    expected_locations = {scene.removeprefix("ios-") for scene in CROCODL_LAYOUT.scenes}
    if len(archives) != 64 or len({(archive.location, archive.session_id) for archive in archives}) != 64:
        raise ValueError("CroCoDL iOS source inventory must contain 64 unique capture sessions")
    if {archive.location for archive in archives} != expected_locations:
        raise ValueError("CroCoDL iOS source inventory does not exactly match the registered locations")
    if any(
        not archive.filename.startswith("ios_")
        or not archive.filename.endswith(".zip")
        or archive.session_id.endswith(("_map", "_query"))
        or archive.size <= 0
        or not re.fullmatch(r"[0-9a-f]{64}", archive.sha256)
        or not re.fullmatch(r"[0-9a-f]{40}", archive.revision)
        for archive in archives
    ):
        raise ValueError("CroCoDL iOS source inventory contains an invalid archive record")
    return archives


def official_ios_auxiliary_inventory() -> tuple[OfficialIosAuxiliaryArchive, ...]:
    """Read the complete non-runnable iOS map/query archive inventory."""
    with dataset_asset("crocodl", "auxiliary_ios_archives.tsv").open(encoding="utf-8", newline="") as file:
        rows = tuple(csv.DictReader(file, delimiter="\t"))
    archives = tuple(
        OfficialIosAuxiliaryArchive(
            location=validate_path_component(row["location"], "CroCoDL location"),
            filename=validate_path_component(row["filename"], "CroCoDL archive filename"),
            size=int(row["size"]),
            sha256=row["sha256"],
            revision=row["revision"],
            role=row["role"],
        )
        for row in rows
    )
    locations = {scene.removeprefix("ios-") for scene in CROCODL_LAYOUT.scenes}
    expected = {(location, f"ios_{kind}.zip") for location in locations for kind in ("map", "query")}
    if {(archive.location, archive.filename) for archive in archives} != expected or len(archives) != 8:
        raise ValueError("CroCoDL auxiliary iOS inventory must contain map and query archives for every location")
    if any(
        archive.role != f"localization-{Path(archive.filename).stem.removeprefix('ios_')}"
        or archive.size <= 0
        or not re.fullmatch(r"[0-9a-f]{64}", archive.sha256)
        or not re.fullmatch(r"[0-9a-f]{40}", archive.revision)
        for archive in archives
    ):
        raise ValueError("CroCoDL auxiliary iOS inventory contains an invalid archive record")
    return archives


def _official_archives_for_scene(scene: str) -> tuple[OfficialIosArchive, ...]:
    location = parse_benchmark_scene(scene)
    return tuple(archive for archive in official_ios_inventory() if archive.location == location)


def _rows(path: Path, session_id: str) -> Iterator[CsvRow]:
    if not path.exists():
        return
    with path.open("r", encoding="utf-8", newline="") as file:
        for line_number, values in enumerate(csv.reader(file), 1):
            values = tuple(value.strip() for value in values)
            if values and any(values) and not values[0].startswith("#"):
                yield CsvRow(values, path, line_number, session_id)


def _validate_sensor(sensor: CameraSensor) -> None:
    expected = _CAMERA_PARAMETER_COUNTS.get(sensor.model)
    context = f"{sensor.source}: sensor {sensor.sensor_id!r}"
    if expected is None:
        supported = ", ".join(sorted(_CAMERA_PARAMETER_COUNTS))
        raise CroCoDLFormatError(f"{context} uses unsupported camera model {sensor.model!r}; supported: {supported}")
    if sensor.width <= 0 or sensor.height <= 0:
        raise CroCoDLFormatError(f"{context} has invalid resolution {sensor.width}x{sensor.height}")
    if len(sensor.parameters) != expected:
        raise CroCoDLFormatError(
            f"{context} uses {sensor.model}, which requires {expected} parameters; found {len(sensor.parameters)}"
        )
    focal_count = 2 if sensor.model in {"PINHOLE", "OPENCV"} else 1
    if any(value <= 0 for value in sensor.parameters[:focal_count]):
        raise CroCoDLFormatError(f"{context} has non-positive focal length")


class CroCoDLCaptureReader:
    """Parse one raw capture session once for conversion."""

    def __init__(self, location_dir: Path, session_id: str):
        self.session_id = validate_path_component(session_id, "CroCoDL session ID")
        self.session_dir = Path(location_dir) / "sessions" / session_id
        if not self.session_dir.is_dir():
            raise FileNotFoundError(f"Session directory not found: {self.session_dir}")
        self.sensors = self._parse_sensors()
        self.trajectories = self._parse_trajectories(self.session_dir / "trajectories.txt")
        self.alignment_trajectories = self._parse_trajectories(
            self.session_dir / "proc" / "alignment_trajectories.txt"
        )
        self.images = self._parse_images()

    @property
    def poses(self) -> Mapping[PoseKey, Pose]:
        return self.alignment_trajectories or self.trajectories

    def image_path(self, image: CaptureImage) -> Path:
        return self.session_dir / "raw_data" / image.relative_path

    def _parse_sensors(self) -> dict[str, CameraSensor]:
        sensors: dict[str, CameraSensor] = {}
        for row in _rows(self.session_dir / "sensors.txt", self.session_id):
            row.require(3, "sensor row")
            sensor_id = row.values[0]
            if not sensor_id:
                raise row.fail("sensor ID cannot be empty")
            if row.values[2].casefold() != "camera":
                continue
            row.require(7, "camera sensor row")
            if sensor_id in sensors:
                raise row.fail(f"duplicate camera sensor ID {sensor_id!r}")
            sensor = CameraSensor(
                sensor_id,
                row.values[3].upper(),
                row.integer(4, "camera width"),
                row.integer(5, "camera height"),
                row.floats(row.values[6:], "camera parameters"),
                row.source,
            )
            _validate_sensor(sensor)
            sensors[sensor_id] = sensor
        return dict(sorted(sensors.items()))

    def _parse_trajectories(self, path: Path) -> dict[PoseKey, Pose]:
        trajectories: dict[PoseKey, Pose] = {}
        for row in _rows(path, self.session_id):
            pose = row.pose("trajectory row")
            key = (row.integer(0, "trajectory timestamp"), row.values[1])
            if not key[1]:
                raise row.fail("trajectory sensor ID cannot be empty")
            if key in trajectories:
                raise row.fail(f"duplicate trajectory for timestamp/sensor {key!r}")
            trajectories[key] = pose
        return dict(sorted(trajectories.items()))

    def _parse_images(self) -> tuple[CaptureImage, ...]:
        images = []
        for row in _rows(self.session_dir / "images.txt", self.session_id):
            row.require(3, "image row")
            if len(row.values) != 3:
                raise row.fail(f"image row requires exactly 3 fields, found {len(row.values)}")
            sensor_id, relative_path = row.values[1], Path(row.values[2])
            if not sensor_id:
                raise row.fail("image sensor ID cannot be empty")
            if not relative_path.name or relative_path.is_absolute() or ".." in relative_path.parts:
                raise row.fail(f"invalid relative image path {row.values[2]!r}")
            images.append(
                CaptureImage(
                    row.integer(0, "image timestamp"),
                    sensor_id,
                    relative_path,
                    row.source,
                )
            )
        return tuple(images)


def colmap_camera_from_sensor(sensor: CameraSensor, camera_id: int) -> pycolmap.Camera:
    _validate_sensor(sensor)
    if not 0 <= camera_id < pycolmap.INVALID_CAMERA_ID:
        raise ValueError(f"Camera ID must be in [0, {pycolmap.INVALID_CAMERA_ID}), got {camera_id}")
    try:
        camera = pycolmap.Camera(
            model=sensor.model,
            camera_id=camera_id,
            width=sensor.width,
            height=sensor.height,
            params=sensor.parameters,
        )
    except (TypeError, ValueError, RuntimeError) as error:
        raise CroCoDLFormatError(
            f"{sensor.source}: failed to construct camera for sensor {sensor.sensor_id!r}: {error}"
        ) from error
    if not camera.verify_params():
        raise CroCoDLFormatError(f"{sensor.source}: COLMAP rejected parameters for sensor {sensor.sensor_id!r}")
    return camera


def get_testset_data(
    reconstruction: pycolmap.Reconstruction,
    session_name: str,
    max_per_session: int = 50,
) -> TestsetData:
    """Build duration-calibrated testset subsequences for one reconstruction."""
    keyframes = sorted(image_id for image_id, image in reconstruction.images.items() if image.has_pose)
    if not keyframes:
        raise CroCoDLFormatError(f"session {session_name!r} has no converted images with valid poses")

    print(f"  Found {len(keyframes)} keyframes out of {len(reconstruction.images)} images")
    result: TestsetData = {}
    for length in _TESTSET_LENGTHS:
        chunks = [
            keyframes[start : start + length]
            for start in range(0, len(keyframes), length)
            if len(keyframes[start : start + length]) == length
        ][:max_per_session]
        if chunks:
            result[length] = {f"{session_name}-{index}": list(chunk) for index, chunk in enumerate(chunks)}
            print(f"  Found {len(chunks)} subsequences of length {length}")
    result["complete"] = {f"{session_name}-0": keyframes}
    return result


def _atomic_yaml(path: Path, data: Mapping[str, list[int]]) -> None:
    temporary = path.with_name(f".{path.name}.crocodl-tmp")
    if temporary.exists():
        temporary.unlink()
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as file:
            yaml.safe_dump(
                dict(sorted(data.items())),
                file,
                allow_unicode=False,
                default_flow_style=False,
                sort_keys=True,
            )
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def write_combined_testsets(
    testsets: Mapping[TestsetKey, Mapping[str, list[int]]],
    output_dir: Path,
    location: str,
    *,
    prune_stale: bool = True,
) -> None:
    location = validate_path_component(location, "CroCoDL location")
    if not testsets:
        raise CroCoDLFormatError(f"refusing to publish zero testsets for location {location!r}")
    output_dir = Path(output_dir)
    scene = benchmark_scene_name(location)
    if scene is None:
        raise CroCoDLFormatError(f"CroCoDL location {location!r} is not registered")
    output_dir.mkdir(parents=True, exist_ok=True)
    complete = testsets.get("complete")
    if not complete or any(not image_ids for image_ids in complete.values()):
        raise CroCoDLFormatError(f"refusing to publish empty complete testsets for scene {scene!r}")
    if any(
        not subsequences or any(not image_ids for image_ids in subsequences.values())
        for subsequences in testsets.values()
    ):
        raise CroCoDLFormatError(f"refusing to publish empty testsets for scene {scene!r}")

    scene_dir = output_dir / scene
    scene_dir.mkdir(parents=True, exist_ok=True)
    lengths = sorted(
        testsets,
        key=lambda key: (key == "complete", 0 if key == "complete" else key),
    )
    filenames = {"complete.yaml" if length == "complete" else f"sample-{length}.yaml" for length in lengths}
    for length in lengths:
        filename = "complete.yaml" if length == "complete" else f"sample-{length}.yaml"
        _atomic_yaml(scene_dir / filename, testsets[length])
    if prune_stale:
        for path in scene_dir.glob("sample-*.yaml"):
            if _SAMPLE_TESTSET.fullmatch(path.name) and path.name not in filenames:
                path.unlink()
    print(f"  {scene}: wrote {len(lengths)} testset files")


def _available_images(
    reader: CroCoDLCaptureReader,
    output_session_id: str,
) -> tuple[CaptureImage, ...]:
    existing = tuple(image for image in reader.images if reader.image_path(image).is_file())
    if missing := len(reader.images) - len(existing):
        print(f"  Ignoring {missing} missing source image(s) from {reader.session_id}")
    basenames: dict[str, Path] = {}
    for image in existing:
        previous = basenames.get(image.basename)
        if previous is not None:
            raise CroCoDLFormatError(
                f"{image.source}: session {output_session_id!r} maps both {previous} and {image.relative_path} "
                f"to output basename {image.basename!r}"
            )
        basenames[image.basename] = image.relative_path
    return existing


def _build_reconstruction(
    reader: CroCoDLCaptureReader,
    images: Sequence[CaptureImage],
) -> pycolmap.Reconstruction:
    sensor_ids = tuple(sorted({image.sensor_id for image in images}))
    shared_ios_camera = len(sensor_ids) > 1 and all("cam_phone_" in sensor_id for sensor_id in sensor_ids)
    camera_sensor_ids = (images[0].sensor_id,) if shared_ios_camera else sensor_ids
    pose_keys = {(image.timestamp, image.sensor_id) for image in images}
    if len(pose_keys) != len(images):
        raise CroCoDLFormatError(f"session {reader.session_id!r} contains duplicate timestamp/sensor image rows")

    reconstruction = pycolmap.Reconstruction()
    camera_ids = {sensor_id: camera_id for camera_id, sensor_id in enumerate(camera_sensor_ids)}
    for sensor_id, camera_id in camera_ids.items():
        try:
            sensor = reader.sensors[sensor_id]
        except KeyError as error:
            source = next(image.source for image in images if image.sensor_id == sensor_id)
            raise CroCoDLFormatError(f"{source}: image references unknown camera sensor {sensor_id!r}") from error
        reconstruction.add_camera_with_trivial_rig(colmap_camera_from_sensor(sensor, camera_id))

    for image_id, source in enumerate(images):
        pose = reader.poses.get((source.timestamp, source.sensor_id))
        if pose is None:
            raise CroCoDLFormatError(
                f"{source.source}: image {source.relative_path} has no direct pose for "
                f"{(source.timestamp, source.sensor_id)!r}"
            )
        cam_from_world = pose.as_rigid3d().inverse()
        reconstruction.add_image_with_trivial_frame(
            pycolmap.Image(
                # The established CroCoDL iOS boundary uses the first frame's
                # calibration as one shared camera. Official per-frame
                # intrinsics vary slightly, but changing that policy changes
                # the canonical mapper input and prior benchmark outputs.
                camera_id=0 if shared_ios_camera else camera_ids[source.sensor_id],
                name=source.basename,
                image_id=image_id,
            ),
            cam_from_world,
        )
    return reconstruction


def _materialize_images(
    reader: CroCoDLCaptureReader,
    images: Sequence[CaptureImage],
    output_dir: Path,
    *,
    symlink: bool,
    description: str,
) -> None:
    for image in tqdm(images, desc=f"Linking images ({description})", leave=False):
        source, destination = reader.image_path(image), output_dir / image.basename
        if symlink:
            destination.symlink_to(source.resolve(strict=True))
        else:
            shutil.copy2(source, destination)


@contextmanager
def _staged_session(destination: Path) -> Iterator[Path]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.with_name(f".{destination.name}.crocodl-staging")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir()
    try:
        yield staging
        if destination.exists():
            shutil.rmtree(destination)
        staging.rename(destination)
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def convert_single_camera_session(
    reader: CroCoDLCaptureReader,
    location: str,
    session_id: str,
    output_dir: Path,
    symlink_images: bool,
    provenance: Mapping[str, object] | None = None,
) -> ConversionResult | None:
    location = validate_path_component(location, "CroCoDL location")
    session_id = validate_ios_session(session_id)
    if reader.session_id != session_id:
        raise ValueError(f"reader session {reader.session_id!r} does not match requested session {session_id!r}")
    images = _available_images(reader, session_id)
    if not images:
        print(f"  Skipping {session_id}: no source images exist")
        return None
    reconstruction = _build_reconstruction(reader, images)
    testsets = get_testset_data(reconstruction, session_id)
    metadata = {
        "location": location,
        "session": session_id,
        "num_images": len(images),
        "num_cameras": len(reconstruction.cameras),
        "preparation_version": CROCODL_PREPARATION_VERSION,
    }
    if provenance is not None:
        metadata["source"] = dict(provenance)
    session_dir = Path(output_dir) / location / session_id
    with _staged_session(session_dir) as staging:
        images_dir, rec_dir = staging / "images", staging / "rec"
        images_dir.mkdir()
        rec_dir.mkdir()
        _materialize_images(
            reader,
            images,
            images_dir,
            symlink=symlink_images,
            description=session_id,
        )
        print(f"  Writing {session_id} ({len(reconstruction.images)} images)")
        reconstruction.write_binary(rec_dir)
        with (staging / "metadata.json").open("w", encoding="utf-8", newline="\n") as file:
            json.dump(metadata, file, indent=2, sort_keys=True)
            file.write("\n")
    return ConversionResult(session_id, testsets)


def _select_session_ids(
    sessions_dir: Path,
    requested: Sequence[str] | None,
) -> tuple[str, ...]:
    if requested is None:
        validated = {
            validate_ios_session(entry.name)
            for entry in sessions_dir.iterdir()
            if entry.is_dir() and entry.name.startswith("ios_")
        }
    else:
        validated = {validate_ios_session(name) for name in requested}
    selected = {name for name in validated if not name.endswith(("_map", "_query"))}
    return tuple(sorted(selected))


def _combine_testsets(
    results: Sequence[ConversionResult],
) -> TestsetData:
    combined: TestsetData = {}
    for result in sorted(results, key=lambda item: item.session_id):
        for length, subsequences in result.testsets.items():
            destination = combined.setdefault(length, {})
            if duplicates := destination.keys() & subsequences.keys():
                raise CroCoDLFormatError(f"duplicate testset subsequences: {sorted(duplicates)}")
            destination.update(subsequences)
    return combined


def _prepared_conversion_results(
    output_dir: Path,
    location: str,
    allowed_session_ids: Sequence[str] | None = None,
) -> list[ConversionResult]:
    """Recompute testsets from every prepared session in one converted location."""
    location_dir = Path(output_dir) / location
    allowed = None if allowed_session_ids is None else set(allowed_session_ids)
    results: list[ConversionResult] = []
    for session_dir in sorted(location_dir.iterdir()):
        if not session_dir.name.startswith("ios_"):
            continue
        if allowed is not None and session_dir.name not in allowed:
            continue
        metadata_path = session_dir / "metadata.json"
        if not session_dir.is_dir() or not metadata_path.is_file():
            continue
        try:
            with metadata_path.open(encoding="utf-8") as file:
                metadata = json.load(file)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise CroCoDLFormatError(f"invalid prepared CroCoDL metadata {metadata_path}: {error}") from error
        if not isinstance(metadata, Mapping):
            raise CroCoDLFormatError(f"invalid prepared CroCoDL metadata {metadata_path}: expected an object")
        try:
            session_id = validate_ios_session(metadata.get("session"))
        except (TypeError, ValueError) as error:
            raise CroCoDLFormatError(f"invalid prepared CroCoDL metadata {metadata_path}: {error}") from error
        if metadata.get("location") != location or session_id != session_dir.name:
            raise CroCoDLFormatError(f"prepared CroCoDL metadata does not match its location: {metadata_path}")
        reconstruction_dir = session_dir / "rec"
        images_dir = session_dir / "images"
        if not reconstruction_dir.is_dir() or not images_dir.is_dir():
            raise CroCoDLFormatError(f"prepared CroCoDL session is incomplete: {session_dir}")
        try:
            reconstruction = pycolmap.Reconstruction(reconstruction_dir)
        except (OSError, ValueError, RuntimeError) as error:
            raise CroCoDLFormatError(
                f"invalid prepared CroCoDL reconstruction {reconstruction_dir}: {error}"
            ) from error
        results.append(
            ConversionResult(
                session_id,
                get_testset_data(reconstruction, session_id),
            )
        )
    return results


def convert_location(
    capture_dir: Path,
    location: str,
    output_dir: Path,
    testsets_dir: Path,
    session_ids: Sequence[str] | None = None,
    symlink_images: bool = True,
    provenance_by_session: Mapping[str, Mapping[str, object]] | None = None,
    publication_session_ids: Sequence[str] | None = None,
) -> None:
    location = validate_path_component(location, "CroCoDL location")
    location_dir = Path(capture_dir) / location
    sessions_dir = location_dir / "sessions"
    if not sessions_dir.is_dir():
        raise FileNotFoundError(f"Sessions directory not found: {sessions_dir}")
    selected = _select_session_ids(sessions_dir, session_ids)
    if not selected:
        raise CroCoDLFormatError(f"location {location!r} has no sessions matching the requested selection")
    print(f"Converting {len(selected)} sessions from {location}")
    converted: list[ConversionResult] = []
    for session_id in tqdm(selected, desc="Processing sessions"):
        print(f"\nProcessing {session_id}...")
        reader = CroCoDLCaptureReader(location_dir, session_id)
        if not reader.images:
            print("  Skipping: no images found")
            continue
        result = convert_single_camera_session(
            reader,
            location,
            session_id,
            output_dir,
            symlink_images,
            provenance=None if provenance_by_session is None else provenance_by_session.get(session_id),
        )
        if result is not None:
            converted.append(result)

    if not converted:
        raise CroCoDLFormatError(f"location {location!r} produced no valid converted sessions")
    print("\nCombining testsets...")
    partial_run = session_ids is not None
    publication_results = (
        _prepared_conversion_results(output_dir, location, publication_session_ids) if partial_run else converted
    )
    write_combined_testsets(
        _combine_testsets(publication_results),
        Path(testsets_dir) / "crocodl",
        location,
        prune_stale=not partial_run,
    )
    print(f"\nConversion complete!\n  Converted {len(converted)} sessions")
    for result in converted:
        print(f"    - {result.session_id}")


def _official_session_is_prepared(archive: OfficialIosArchive) -> bool:
    session_dir = CROCODL_DATA_DIR / archive.location / archive.session_id
    metadata_path = session_dir / "metadata.json"
    images_dir = session_dir / "images"
    rec_dir = session_dir / "rec"
    if not metadata_path.is_file() or not images_dir.is_dir() or not rec_dir.is_dir():
        return False
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        source = metadata["source"]
        reconstruction = pycolmap.Reconstruction(rec_dir)
    except (
        OSError,
        KeyError,
        TypeError,
        ValueError,
        RuntimeError,
        json.JSONDecodeError,
    ):
        return False
    return (
        metadata.get("location") == archive.location
        and metadata.get("preparation_version") == CROCODL_PREPARATION_VERSION
        and metadata.get("session") == archive.session_id
        and source
        == {
            "filename": archive.filename,
            "revision": archive.revision,
            "sha256": archive.sha256,
            "size": archive.size,
            "url": archive.url,
        }
        and bool(reconstruction.images)
        and all((images_dir / image.name).is_file() for image in reconstruction.images.values())
    )


def official_scene_is_prepared(scene: str) -> bool:
    """Return whether every official iOS session and the aggregate testset exist."""
    archives = _official_archives_for_scene(scene)
    if not archives or not all(_official_session_is_prepared(archive) for archive in archives):
        return False
    testset = CROCODL_LAYOUT.testset_path(scene, "complete")
    try:
        payload = yaml.safe_load(testset.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError):
        return False
    expected = {f"{archive.session_id}-0" for archive in archives}
    return isinstance(payload, Mapping) and set(payload) == expected and all(payload[key] for key in expected)


def _extract_official_archive(archive: OfficialIosArchive, downloaded: Path) -> Path:
    sessions_dir = CROCODL_DATA_DIR / "capture" / archive.location / "sessions"
    destination = sessions_dir / archive.session_id
    source_record_path = destination / ".source_archive.json"
    if destination.is_dir():
        try:
            source_record = json.loads(source_record_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            source_record = None
        if source_record == _source_record(archive):
            CroCoDLCaptureReader(sessions_dir.parent, archive.session_id)
            return destination
        shutil.rmtree(destination)
    elif destination.exists():
        raise RuntimeError(f"CroCoDL capture destination is not a directory: {destination}")
    staging = sessions_dir.parent / f".{archive.session_id}.extracting"
    if staging.exists():
        shutil.rmtree(staging)
    staging_sessions = staging / "sessions"
    staging_sessions.mkdir(parents=True)
    try:
        safe_extract_zip(downloaded, staging_sessions)
        extracted = staging_sessions / archive.session_id
        if not extracted.is_dir():
            raise CroCoDLFormatError(f"Official CroCoDL archive {downloaded} does not contain {archive.session_id}/")
        CroCoDLCaptureReader(staging, archive.session_id)
        (extracted / ".source_archive.json").write_text(
            json.dumps(_source_record(archive), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        sessions_dir.mkdir(parents=True, exist_ok=True)
        extracted.rename(destination)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return destination


def _source_record(archive: OfficialIosArchive) -> dict[str, object]:
    return {
        "filename": archive.filename,
        "revision": archive.revision,
        "sha256": archive.sha256,
        "size": archive.size,
        "url": archive.url,
    }


def prepare_official_scene(scene: str, *, delete_files: bool = False) -> None:
    """Download and prepare every public non-map iOS capture at one location."""
    archives = _official_archives_for_scene(scene)
    official_session_ids = tuple(archive.session_id for archive in archives)
    location = parse_benchmark_scene(scene)
    pending = tuple(archive for archive in archives if not _official_session_is_prepared(archive))
    downloaded: dict[str, Path] = {}
    for archive in pending:
        destination = CROCODL_DATA_DIR / "downloads" / location / archive.filename
        downloaded[archive.session_id] = resumable_http_download(
            archive.url,
            destination,
            expected_size=archive.size,
            expected_digest=archive.sha256,
        )
        _extract_official_archive(archive, destination)

    if pending:
        convert_location(
            capture_dir=CROCODL_DATA_DIR / "capture",
            location=location,
            output_dir=CROCODL_DATA_DIR,
            testsets_dir=CROCODL_LAYOUT.testsets.parent,
            session_ids=[archive.session_id for archive in pending],
            symlink_images=False,
            provenance_by_session={archive.session_id: _source_record(archive) for archive in pending},
            publication_session_ids=official_session_ids,
        )
    else:
        write_combined_testsets(
            _combine_testsets(_prepared_conversion_results(CROCODL_DATA_DIR, location, official_session_ids)),
            CROCODL_LAYOUT.testsets,
            location,
            prune_stale=False,
        )

    if not official_scene_is_prepared(scene):
        raise RuntimeError(f"CroCoDL preparation did not produce the complete official iOS scene: {scene}")
    for archive in pending:
        shutil.rmtree(CROCODL_DATA_DIR / "capture" / location / "sessions" / archive.session_id)
        if delete_files:
            downloaded[archive.session_id].unlink()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download and prepare the official CroCoDL iOS benchmark")
    parser.add_argument(
        "--scenes",
        nargs="+",
        choices=CROCODL_LAYOUT.scenes,
        help="Aggregate iOS scenes to prepare (default: all four locations)",
    )
    parser.add_argument(
        "--delete-downloads",
        action="store_true",
        help="Delete validated source archives after successful preparation",
    )
    return parser


def main(scenes: Sequence[str] | None = None, *, delete_files: bool = False) -> None:
    selected = CROCODL_LAYOUT.select_scenes(scenes)
    CROCODL_DATA_DIR.mkdir(parents=True, exist_ok=True)
    for scene in selected:
        prepare_official_scene(scene, delete_files=delete_files)


if __name__ == "__main__":
    arguments = build_parser().parse_args()
    main(arguments.scenes, delete_files=arguments.delete_downloads)
