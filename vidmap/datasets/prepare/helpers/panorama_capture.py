"""Render declared equirectangular frames into a fixed, zero-baseline rig."""

from __future__ import annotations

import math
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
from PIL import Image

from vidmap.datasets.capture import (
    CaptureSnapshot,
    PreparedCapture,
    build_capture_reconstruction,
    resolve_capture_frames,
)
from vidmap.datasets.prepare.helpers.filesystem import prepare_image_directory
from vidmap.frontend.cache.metadata import fingerprint
from vidmap.frontend.options.matching import RoMaImageOptions
from vidmap.frontend.options.rig import PanoramaOptions
from vidmap.repro.hashing import sha256_file


def camera_from_panorama(view):
    """Return the declared view rotation in COLMAP's +Z-forward, +Y-down frame."""
    import pycolmap

    return (
        pycolmap.Rotation3d(np.array([-math.radians(view["pitch_degrees"]), 0, 0]))
        * pycolmap.Rotation3d(np.array([0, -math.radians(view["yaw_degrees"]), 0]))
    ).matrix()


def _image_name(camera_id, frame):
    return f"cam{camera_id}/{Path(frame['image_name']).stem}.png"


def _geometry(identity):
    import pycolmap
    from pycolmap.panorama import create_pano_rig_config

    projection = identity["projection"]
    width, height = projection["image_size"]
    focal = width / (2 * math.tan(math.radians(projection["horizontal_fov_degrees"]) / 2))
    if not math.isfinite(focal) or focal <= 0:
        raise ValueError("Projection focal length must be finite and positive")
    ids = sorted(projection["views"])
    rotations = [camera_from_panorama(projection["views"][cid]) for cid in ids]
    config = create_pano_rig_config(rotations, ref_idx=0)
    cameras = [
        pycolmap.Camera(
            camera_id=cid,
            model="PINHOLE",
            width=width,
            height=height,
            params=[focal, focal, width / 2, height / 2],
            has_prior_focal_length=True,
        )
        for cid in ids
    ]
    rig = pycolmap.Rig(rig_id=1)
    rig.add_ref_sensor(cameras[0].sensor_id)
    for camera, entry in zip(cameras, config.cameras, strict=True):
        if not entry.ref_sensor:
            rig.add_sensor(camera.sensor_id, entry.cam_from_rig)
    return cameras, rig, rotations


def _image_size(
    projection: PanoramaOptions, panorama_size: tuple[int, int], image_options: RoMaImageOptions
) -> tuple[int, int]:
    """Resolve view dimensions using the tracking resize policy."""
    width = panorama_size[0] * projection.horizontal_fov_degrees / 360
    aspect_width, aspect_height = projection.aspect_ratio
    size = (width, width * aspect_height / aspect_width)
    resize_max = image_options.resize_max
    if resize_max and (image_options.resize_force or max(size) > resize_max):
        scale = resize_max / max(size)
        size = tuple(value * scale for value in size)
    size = tuple(round(value) for value in size)
    if min(size) < 1:
        raise ValueError("Panorama aspect ratio and tracking resolution must produce positive image dimensions")
    return size


def _inputs(image_dir, options, image_names, image_options):
    projection = asdict(options.projection)
    root = Path(image_dir).expanduser().resolve(strict=True)
    if not root.is_dir():
        raise ValueError("images must be a directory")
    frames = resolve_capture_frames(root, ("cam0",), options.frames)
    if any(row["camera_id"] != 0 or not row["image_name"].startswith("cam0/") for row in frames):
        raise ValueError("Every panorama frame must contain exactly one cam0/ image")
    if image_names is not None:
        selected = set(image_names)
        frames = tuple(frame for frame in frames if frame["image_name"] in selected)
        if not frames or {frame["image_name"] for frame in frames} != selected:
            raise ValueError("Panorama selection must contain known source images")
    if len({Path(frame["image_name"]).stem for frame in frames}) != len(frames):
        raise ValueError("Panorama filenames must have unique stems")
    records, panorama_size = [], None
    for row in frames:
        source = (root / row["image_name"]).resolve(strict=True)
        if not source.is_relative_to(root):
            raise ValueError("Source image must be inside images")
        with Image.open(source) as pixels:
            if pixels.width != 2 * pixels.height:
                raise ValueError("Panoramas must have 2:1 dimensions")
            if panorama_size is not None and pixels.size != panorama_size:
                raise ValueError("Panorama dimensions must be constant")
            panorama_size = pixels.size
        records.append(
            {key: value for key, value in row.items() if key != "camera_id"} | {"sha256": sha256_file(source)}
        )
    projection["image_size"] = _image_size(options.projection, panorama_size, image_options)
    identity = dict(
        schema_version=3,
        renderer="colmap-opencv-linear-wrap-v2",
        projection=projection,
        panorama_size=panorama_size,
        frames=records,
    )
    return identity, root


def _capture(output, identity, hashes, preview_image_dir, cameras, rig, rotations):
    panorama = {
        "panorama_size": list(identity["panorama_size"]),
        "rig_from_panorama_rotation": rotations[0].tolist(),
        "frames": [{key: frame[key] for key in ("frame_id", "image_name", "sha256")} for frame in identity["frames"]],
    }
    rows = [
        dict(
            frame_id=frame["frame_id"],
            timestamp_ns=frame["timestamp_ns"],
            camera_id=camera.camera_id,
            image_name=_image_name(camera.camera_id, frame),
        )
        for frame in identity["frames"]
        for camera in cameras
    ]
    rec, frame_times = build_capture_reconstruction(cameras, rig, rows)
    capture = PreparedCapture(
        rec=rec,
        rgb_dir=output / "images",
        frame_timestamps_ns=frame_times,
        provenance={
            "source_fingerprint": fingerprint(identity),
            "panorama": panorama,
            "image_sha256": hashes,
        },
        preview_image_dir=preview_image_dir,
    )

    return CaptureSnapshot(capture, hashes)


def _render(identity, source_root, target, cameras, rotations):
    import cv2
    from pycolmap.panorama import get_virtual_camera_rays, spherical_img_from_cam

    width, height = identity["projection"]["image_size"]
    rays = get_virtual_camera_rays(cameras[0])
    maps = []
    for camera, rotation in zip(cameras, rotations, strict=True):
        # Upstream rays are x-major. Convert COLMAP's .5 pixel centers to OpenCV.
        xy = spherical_img_from_cam(tuple(identity["panorama_size"]), rays @ rotation)
        xy = (xy.reshape(width, height, 2).transpose(1, 0, 2) - 0.5).astype(np.float32)
        xy[..., 1] = np.clip(xy[..., 1], 0, identity["panorama_size"][1] - 1)
        directory = target / f"cam{camera.camera_id}"
        directory.mkdir(parents=True)
        maps.append((directory, xy))

    for frame in identity["frames"]:
        with Image.open(source_root / frame["image_name"]) as image:
            pixels = np.asarray(image.convert("RGB"))
        for directory, xy in maps:
            rendered = cv2.remap(pixels, xy, None, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
            # Lossless either way; level 1 encodes ~4.5x faster than PIL's default 6 for ~8% larger files.
            Image.fromarray(rendered).save(directory / f"{Path(frame['image_name']).stem}.png", compress_level=1)


def prepare_panorama_capture(
    image_dir: str | Path, options, output_dir: str | Path, *, frontend_options, image_names=None
) -> CaptureSnapshot:
    """Render pinhole views at tracking resolution.

    Reuse output only when source files, frame associations and projection match.
    """
    output = Path(output_dir).expanduser().absolute()
    identity, preview_image_dir = _inputs(image_dir, options, image_names, frontend_options.tracks.images)
    cameras, rig, rotations = _geometry(identity)
    from vidmap.frontend.preparation.cameras import resolve_camera_preparation

    capture = _capture(output, identity, {}, preview_image_dir, cameras, rig, rotations).capture
    resolve_camera_preparation(capture, frontend_options)
    names = tuple(
        _image_name(cid, frame) for frame in identity["frames"] for cid in sorted(identity["projection"]["views"])
    )

    def write_images(image_dir):
        _render(identity, preview_image_dir, image_dir, cameras, rotations)
        if _inputs(preview_image_dir, options, image_names, frontend_options.tracks.images)[0] != identity:
            raise ValueError("Source capture changed while rendering")

    hashes = prepare_image_directory(output, identity, names, write_images)
    prepared = replace(capture, provenance={**capture.provenance, "image_sha256": hashes})
    return CaptureSnapshot(prepared, hashes)
