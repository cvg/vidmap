"""Load calibrated physical rigs and prepare images with COLMAP undistortion."""

from dataclasses import replace
from pathlib import Path

import numpy as np
from PIL import Image

from vidmap.datasets.capture import (
    CaptureSnapshot,
    PreparedCapture,
    build_capture_reconstruction,
    resolve_capture_frames,
)
from vidmap.datasets.prepare.filesystem import prepare_image_directory


def load_capture(image_dir: str | Path, options) -> PreparedCapture:
    """Load inline calibration and synchronized images from cam0/, cam1/, ... ."""
    import pycolmap

    root = Path(image_dir).expanduser().resolve(strict=True)
    if not root.is_dir():
        raise ValueError("Capture images must be a directory")
    if not options.cameras:
        raise ValueError("Physical rig input requires rig.cameras")
    rows = resolve_capture_frames(root, tuple(f"cam{cid}" for cid in range(len(options.cameras))), options.frames)
    sizes = {}
    for row in rows:
        camera_id, name = row["camera_id"], row["image_name"]
        if camera_id >= len(options.cameras) or not name.startswith(f"cam{camera_id}/"):
            raise ValueError(f"Image {name!r} must belong to a declared camera's camN/ directory")
        if camera_id not in sizes:
            with Image.open(root / name) as pixels:
                sizes[camera_id] = pixels.size
    if len(sizes) != len(options.cameras):
        raise ValueError("Every declared camera requires images")
    cameras = []
    rig = pycolmap.Rig(rig_id=1)
    for camera_id, source in enumerate(options.cameras):
        width, height = sizes[camera_id]
        camera = pycolmap.Camera(
            camera_id=camera_id,
            model=pycolmap.CameraModelId(source.camera_model_name),
            width=width,
            height=height,
            params=source.camera_params,
            has_prior_focal_length=True,
        )
        if not camera.verify_params():
            raise ValueError(f"Invalid parameter count for camera model {camera.model_name}")
        if any(camera.params[index] <= 0 for index in camera.focal_length_idxs()):
            raise ValueError("Camera focal lengths must be positive")
        cameras.append(camera)
    rig.add_ref_sensor(cameras[0].sensor_id)
    for camera, source in zip(cameras[1:], options.cameras[1:], strict=True):
        # YAML uses wxyz; Rotation3d's quaternion constructor takes xyzw.
        w, x, y, z = source.cam_from_rig_rotation
        rig.add_sensor(
            camera.sensor_id,
            pycolmap.Rigid3d(pycolmap.Rotation3d([x, y, z, w]), source.cam_from_rig_translation),
        )
    rec, frame_times = build_capture_reconstruction(cameras, rig, rows)
    return PreparedCapture(
        rec=rec,
        rgb_dir=root,
        frame_timestamps_ns=frame_times,
        kind="rig",
        provenance={},
    )


def prepare_rig_capture(
    capture: PreparedCapture, output_dir: str | Path, *, camera_ids_to_undistort: tuple[int, ...]
) -> CaptureSnapshot:
    """Undistort physical rigs once; pinhole-only captures keep their source images."""
    import pycolmap

    distorted = set(camera_ids_to_undistort)
    if not distorted:
        return capture.snapshot()

    source = capture.snapshot()
    options = pycolmap.UndistortCameraOptions()
    reconstruction = pycolmap.Reconstruction(capture.rec)
    for camera_id in distorted:
        original = capture.rec.camera(camera_id)
        camera = pycolmap.undistort_camera(options, original)
        camera.camera_id = camera_id
        camera.has_prior_focal_length = original.has_prior_focal_length
        reconstruction.cameras[camera_id] = camera
    # Appending a suffix preserves distinct source names and writes lossless pixels.
    for image in reconstruction.images.values():
        image.name += ".png"

    output = Path(output_dir).expanduser().absolute()
    identity = dict(
        schema_version=2,
        colmap_version=pycolmap.__version__,
        source_fingerprint=source.fingerprint(),
        undistort_camera_ids=sorted(distorted),
    )
    names = tuple(image.name for image in reconstruction.images.values())

    def write_images(image_dir):
        for image_id, image in capture.rec.images.items():
            with Image.open(capture.rgb_dir / image.name) as pixels:
                array = np.asarray(pixels.convert("RGB"))
            if image.camera_id in distorted:
                bitmap, _ = pycolmap.undistort_image(
                    options, pycolmap.Bitmap.from_array(array), capture.rec.camera(image.camera_id)
                )
                array = bitmap.to_array()
            target = image_dir / reconstruction.image(image_id).name
            target.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(array).save(target)
        if capture.snapshot().fingerprint() != identity["source_fingerprint"]:
            raise ValueError("Source capture changed while undistorting")

    hashes = prepare_image_directory(output, identity, names, write_images)
    prepared = replace(
        capture,
        rec=reconstruction,
        rgb_dir=output / "images",
        provenance={"source_fingerprint": identity["source_fingerprint"], "image_sha256": hashes},
    )
    return CaptureSnapshot(prepared, hashes)
