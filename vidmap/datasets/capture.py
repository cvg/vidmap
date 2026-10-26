# Strict integer checks intentionally reject bool.
# ruff: noqa: E721
"""Frontend captures with native COLMAP geometry and optional frame times."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING, Mapping

from PIL import Image

from vidmap.datasets.frame_names import timestamp_from_image_name, timestamp_ns_from_image_name
from vidmap.repro.hashing import sha256_file

if TYPE_CHECKING:
    import pycolmap


def _id(value: object, label: str) -> int:
    if type(value) is not int or not 0 <= value < 2**32 - 1:
        raise ValueError(f"{label} must be a nonnegative COLMAP integer ID")
    return value


def _name(value: object) -> str:
    if not isinstance(value, str) or not value or "\\" in value or "\x00" in value:
        raise ValueError("Image names must be nonempty relative POSIX paths")
    if Path(value).is_absolute() or any(part in ("", ".", "..") for part in value.split("/")):
        raise ValueError(f"Unsafe image name: {value!r}")
    return value


def _fingerprint(value) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def calibration_fingerprint(rec) -> str:
    """Identify native camera calibration without duplicating its persisted graph."""
    return _fingerprint(_calibration(rec))


def _validate_camera_frame(rec, frame):
    import pycolmap

    sensors = set(rec.rig(frame.rig_id).sensor_ids())
    if any(data.sensor_id.type != pycolmap.SensorType.CAMERA for data in frame.data_ids):
        raise ValueError("VidMap capture frames currently support camera sensors only")
    if len(frame.image_ids) != len(sensors) or {data.sensor_id for data in frame.image_ids} != sensors:
        raise ValueError("Incomplete capture frame; supply every camera exactly once")
    if any(rec.image(data.id).frame_id != frame.frame_id for data in frame.image_ids):
        raise ValueError("COLMAP frame and image inventories disagree")


@dataclass
class CaptureSnapshot:
    """Validated source bytes for one frontend execution; geometry remains native."""

    capture: PreparedCapture
    image_sha256: Mapping[str, str]

    def __post_init__(self):
        self.image_sha256 = dict(self.image_sha256)

    def image_content_fingerprint(self, names) -> str:
        # Same canonical JSON as the frontend cache contract, without rereading media.
        return _fingerprint([[name, self.image_sha256[name]] for name in names])

    def fingerprint(self) -> str:
        capture = self.capture
        identity = _fingerprint(
            {
                "images": dict(self.image_sha256),
                "calibration": calibration_fingerprint(capture.rec),
                "membership": [
                    [
                        image.image_id,
                        image.name,
                        image.camera_id,
                        image.frame_id,
                        capture.rec.frame(image.frame_id).rig_id,
                    ]
                    for _, image in sorted(capture.rec.images.items())
                ],
                "frame_order": capture.frame_ids,
                "frame_timestamps_ns": (
                    None if capture.frame_timestamps_ns is None else dict(capture.frame_timestamps_ns)
                ),
                "preserve_calibration": capture.preserve_calibration,
                "forced_keyframes": sorted(capture.forced_keyframe_names),
            }
        )
        if "source_fingerprint" in capture.provenance:
            identity = _fingerprint([identity, capture.provenance["source_fingerprint"]])
        return identity


@dataclass
class PreparedCapture:
    """Native capture geometry, borrowed read-only throughout a frontend execution."""

    rec: pycolmap.Reconstruction
    rgb_dir: Path
    frame_timestamps_ns: Mapping[int, int] | None
    provenance: Mapping
    preserve_calibration: bool = True
    scene: str = "capture"
    preview_image_dir: Path | None = None
    forced_keyframe_names: frozenset[str] = frozenset()
    _image_ids_by_name: Mapping[str, int] = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        if not self.rec.is_valid():
            raise ValueError("Invalid native capture reconstruction")
        if not self.rec.cameras or not self.rec.images:
            raise ValueError("A capture requires cameras and images")
        if len(self.rec.rigs) > 1 and any(rig.num_sensors() != 1 for rig in self.rec.rigs.values()):
            raise ValueError("A capture must have one fixed rig or a sequence of singleton frames")
        sensors = {sensor for rig in self.rec.rigs.values() for sensor in rig.sensor_ids()}
        if sensors != {camera.sensor_id for camera in self.rec.cameras.values()}:
            raise ValueError("COLMAP rig sensors contradict capture cameras")
        for rig in self.rec.rigs.values():
            for sensor in rig.non_ref_sensors:
                transform = rig.sensor_from_rig(sensor)
                if transform is None or not all(math.isfinite(value) for row in transform.matrix() for value in row):
                    raise ValueError("Every sensor requires a finite fixed transform")
        if self.frame_timestamps_ns is not None:
            times = self.frame_timestamps_ns
            if not isinstance(times, Mapping) or set(times) != set(self.rec.frames):
                raise ValueError("Capture timestamps must cover exactly the native frame IDs")
            if any(type(key) is not int or type(value) is not int for key, value in times.items()):
                raise ValueError("Capture timestamps and IDs must be integers")
            if len(set(self.frame_timestamps_ns.values())) != len(self.rec.frames):
                raise ValueError("Capture frame timestamps must increase")
        else:
            if any(rig.num_sensors() != 1 for rig in self.rec.rigs.values()):
                raise ValueError("Multi-camera captures require explicit timestamps")
        for frame in self.rec.frames.values():
            _validate_camera_frame(self.rec, frame)
        if any(frame.has_pose() for frame in self.rec.frames.values()):
            raise ValueError("Capture inputs must be unposed; GT is evaluation-only")
        image_ids = {_name(image.name): image.image_id for image in self.rec.images.values()}
        if len(image_ids) != len(self.rec.images):
            raise ValueError("Capture image names must be unique")
        self._image_ids_by_name = image_ids
        self.rgb_dir = Path(self.rgb_dir)
        if self.preview_image_dir is not None:
            self.preview_image_dir = Path(self.preview_image_dir)
        if self.frame_timestamps_ns is not None:
            self.frame_timestamps_ns = dict(self.frame_timestamps_ns)
        self.provenance = dict(self.provenance)

    def image(self, name: str) -> pycolmap.Image:
        """Resolve names in constant time; membership edits require a new capture."""
        image = self.rec.image(self._image_ids_by_name[name])
        if image.name != name:
            raise ValueError("Capture image names changed after preparation")
        return image

    @cached_property
    def frame_ids(self) -> tuple[int, ...]:
        if self.frame_timestamps_ns is not None:
            return tuple(sorted(self.rec.frames, key=self.frame_timestamps_ns.__getitem__))
        return tuple(
            image.frame_id
            for image in sorted(
                self.rec.images.values(), key=lambda image: (timestamp_from_image_name(image.name), image.image_id)
            )
        )

    @property
    def reference_camera_id(self) -> int:
        return next(iter(self.rec.rigs.values())).ref_sensor_id.id

    @property
    def imnames(self) -> tuple[str, ...]:
        return tuple(self.rec.image(data.id).name for fid in self.frame_ids for data in self.rec.frame(fid).image_ids)

    def snapshot(self) -> CaptureSnapshot:
        """Hash source bytes and validate prepared images before cache reuse."""
        images = {}
        root = self.rgb_dir.resolve(strict=True)
        for name in self.imnames:
            source = root / name
            if self.preserve_calibration:
                source = source.resolve(strict=True)
                if not source.is_relative_to(root):
                    raise ValueError("Capture image escapes the image directory")
                camera = self.rec.camera(self.image(name).camera_id)
                with Image.open(source) as pixels:
                    if pixels.size != (camera.width, camera.height):
                        raise ValueError(f"Capture image dimensions changed: {name}")
            images[name] = sha256_file(source)
        expected_images = self.provenance["image_sha256"] if "image_sha256" in self.provenance else {}
        for name, expected in expected_images.items():
            if name not in images or images[name] != expected:
                raise ValueError(f"Prepared capture image changed: {name}")
        return CaptureSnapshot(self, images)

    def panorama_metadata(self) -> dict | None:
        """Expose only the source media needed by the panoramic viewer."""
        if "panorama" not in self.provenance:
            return None
        return dict(self.provenance["panorama"])


def read_frame_table(path: Path) -> tuple[dict, ...]:
    """Read explicit CSV associations without inventing another geometry model."""
    columns = {"frame_id", "timestamp_ns", "camera_id", "image_name"}
    rows, frame_times, names, members = [], {}, set(), set()
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None or len(reader.fieldnames) != len(columns) or set(reader.fieldnames) != columns:
            raise ValueError("Frame CSV must contain exactly the required columns")
        for row in reader:
            item = {key: _name(value) if key == "image_name" else int(value) for key, value in row.items()}
            fid, cid = _id(item["frame_id"], "Frame ID"), _id(item["camera_id"], "Camera ID")
            timestamp = item["timestamp_ns"]
            if fid in frame_times and frame_times[fid] != timestamp:
                raise ValueError(f"Inconsistent timestamps for frame {fid}")
            if fid not in frame_times:
                if frame_times and timestamp <= next(reversed(frame_times.values())):
                    raise ValueError("Capture frame timestamps must increase")
                frame_times[fid] = timestamp
            if item["image_name"] in names or (fid, cid) in members:
                raise ValueError("Capture images and frame/camera memberships must be unique")
            names.add(item["image_name"])
            members.add((fid, cid))
            rows.append(item)
    if not rows:
        raise ValueError("Frame CSV must not be empty")
    return tuple(rows)


def resolve_capture_frames(root: Path, directories: tuple[str, ...], frames: str | None) -> tuple[dict, ...]:
    """Read explicit associations, or synchronize streams by filename timestamps."""
    from vidmap.datasets.local import select_images

    if frames is not None:
        return read_frame_table(root / frames)
    streams = []
    for directory in directories:
        stream = {}
        for name in select_images(root / directory, None):
            image_name = (Path(directory) / name).as_posix()
            timestamp = timestamp_ns_from_image_name(image_name)
            if timestamp in stream:
                raise ValueError(f"Duplicate timestamp in {directory!r}; set rig.frames to a frames.csv")
            stream[timestamp] = image_name
        streams.append(stream)
    timestamps = sorted(streams[0])
    if any(stream.keys() != streams[0].keys() for stream in streams[1:]):
        raise ValueError("Camera timestamps do not match exactly; set rig.frames to a frames.csv")
    return tuple(
        dict(frame_id=fid, timestamp_ns=timestamp, camera_id=cid, image_name=stream[timestamp])
        for fid, timestamp in enumerate(timestamps, start=1)
        for cid, stream in enumerate(streams)
    )


def build_capture_reconstruction(cameras, rig, rows):
    """Associate explicit rows with native frames, keeping timing in sidecar maps."""
    import pycolmap

    rec = pycolmap.Reconstruction()
    for camera in cameras:
        rec.add_camera(camera)
    rec.add_rig(rig)
    grouped = {}
    for row in rows:
        grouped.setdefault(row["frame_id"], []).append(row)
    frame_times = {}
    image_id = 1
    for fid, entries in grouped.items():
        frame = pycolmap.Frame(frame_id=fid, rig_id=rig.rig_id)
        images = []
        frame_times[fid] = entries[0]["timestamp_ns"]
        for entry in sorted(entries, key=lambda row: row["camera_id"]):
            image = pycolmap.Image(
                image_id=image_id, name=entry["image_name"], camera_id=entry["camera_id"], frame_id=fid
            )
            frame.add_data_id(image.data_id)
            images.append(image)
            image_id += 1
        rec.add_frame(frame)
        for image in images:
            rec.add_image(image)
    return rec, None if all(timestamp is None for timestamp in frame_times.values()) else frame_times


def _calibration(rec: pycolmap.Reconstruction) -> dict:
    return {
        "cameras": {
            str(camera_id): {
                "model": camera.model_name,
                "width": camera.width,
                "height": camera.height,
                "params": camera.params.tolist(),
                "has_prior_focal_length": camera.has_prior_focal_length,
            }
            for camera_id, camera in sorted(rec.cameras.items())
        },
        "rigs": {
            str(rig_id): {
                "reference": rig.ref_sensor_id.id,
                "sensors": {
                    str(sensor.id): (
                        None if rig.is_ref_sensor(sensor) else rig.sensor_from_rig(sensor).matrix().tolist()
                    )
                    for sensor in sorted(rig.sensor_ids(), key=lambda sensor: sensor.id)
                },
            }
            for rig_id, rig in sorted(rec.rigs.items())
        },
    }


def prepare_scene_capture(scene_parser) -> PreparedCapture:
    """Adapt a dataset's calibration and image inventory to the frontend capture."""
    if isinstance(scene_parser, PreparedCapture):
        return scene_parser

    return PreparedCapture(
        rec=copy_unposed_reconstruction(scene_parser.rec),
        rgb_dir=scene_parser.rgb_dir,
        frame_timestamps_ns=None,
        provenance={},
        preserve_calibration=False,
        forced_keyframe_names=frozenset(image.name for image in scene_parser.rec.images.values() if image.has_pose),
    )


def copy_unposed_reconstruction(source, reference_image_names=None) -> pycolmap.Reconstruction:
    """Copy image calibration and complete rig frames, excluding all input poses."""
    import pycolmap

    names = None if reference_image_names is None else set(reference_image_names)
    images = [image for image in source.images.values() if names is None or image.name in names]
    if names is not None and {image.name for image in images} != names:
        raise ValueError("Initial reconstruction selects unknown image names")
    selected_ids = {image.image_id for image in images}
    frame_ids = tuple(dict.fromkeys(image.frame_id for image in images))
    for frame_id in frame_ids:
        frame = source.frame(frame_id)
        if {data.id for data in frame.image_ids} - selected_ids:
            raise ValueError("Rig image selection must retain complete frames")
        _validate_camera_frame(source, frame)
    rec = pycolmap.Reconstruction()
    for camera_id in dict.fromkeys(image.camera_id for image in images):
        rec.add_camera(source.camera(camera_id))
    for rig_id in dict.fromkeys(source.frame(frame_id).rig_id for frame_id in frame_ids):
        rec.add_rig(source.rig(rig_id))
    for frame_id in frame_ids:
        original = source.frame(frame_id)
        frame = pycolmap.Frame(frame_id=frame_id, rig_id=original.rig_id)
        for data in original.image_ids:
            frame.add_data_id(data)
        rec.add_frame(frame)
    for image in images:
        rec.add_image(
            pycolmap.Image(
                image_id=image.image_id, name=image.name, camera_id=image.camera_id, frame_id=image.frame_id
            )
        )
    return rec
