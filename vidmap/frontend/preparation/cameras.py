"""Resolve camera processing and allocate working COLMAP topology once."""

from dataclasses import dataclass, replace
from typing import Literal

import pycolmap

from vidmap.datasets.capture import PreparedCapture
from vidmap.frontend.options.preparation import CameraPriorEstimationOptions, CameraProcessingOptions
from vidmap.frontend.rig import resolve_rig_options


@dataclass(frozen=True)
class CameraPreparationPlan:
    allocation: Literal["input", "shared", "per_image"]
    camera_priors: CameraPriorEstimationOptions
    undistort_camera_ids: tuple[int, ...]


def resolve_camera_preparation(capture: PreparedCapture, options) -> CameraPreparationPlan:
    """Resolve complete camera settings before prediction or image rendering."""
    if options.rig is not None:
        resolve_rig_options(capture, options.rig)
    unknown = set(options.camera_processing) - set(capture.rec.cameras)
    if unknown:
        raise ValueError(f"Camera processing references unknown camera IDs: {sorted(unknown)}")
    default = CameraProcessingOptions(camera_priors=options.camera_priors)
    settings = {
        cid: options.camera_processing[cid] if cid in options.camera_processing else default
        for cid in capture.rec.cameras
    }
    priors = next(iter(settings.values())).camera_priors
    if any(setting.camera_priors != priors for setting in settings.values()):
        raise ValueError("Different per-camera calibration settings are not supported")
    if priors.initialization != options.camera_priors.initialization:
        raise ValueError("Camera processing cannot change the input calibration initialization mode")
    has_multi_sensor_rig = any(rig.num_sensors() != 1 for rig in capture.rec.rigs.values())
    if capture.preserve_calibration:
        if priors.initialization != "supplied":
            raise ValueError("Preserved capture calibration requires supplied initialization")
        if options.keyframes.selection.intrinsics_source != "ground_truth":
            raise ValueError("Preserved capture calibration requires supplied keyframing intrinsics")
        allocation = "input"
    else:
        if has_multi_sensor_rig:
            raise ValueError("Camera regrouping requires single-sensor, single-image frames")
        allocation = "shared" if priors.initialization == "predicted" else "per_image"
    if priors.inference == "selected_batch":
        num_cameras = 1
        if allocation == "input":
            num_cameras = len(capture.rec.cameras)
        elif allocation == "per_image":
            num_cameras = capture.rec.num_images()
        if num_cameras != 1:
            raise ValueError("Selected-batch calibration requires one working camera; select per_view inference")

    undistort = []
    for cid, camera in capture.rec.cameras.items():
        distorted = camera.model_name not in ("PINHOLE", "SIMPLE_PINHOLE")
        image_mode = settings[cid].image_mode
        if image_mode == "auto":
            image_mode = "undistort" if capture.preserve_calibration and distorted else "native"
        if image_mode == "undistort" and distorted:
            if not capture.preserve_calibration:
                raise ValueError("Undistortion requires supplied capture calibration")
            undistort.append(cid)
        elif distorted:
            if capture.preserve_calibration:
                raise ValueError("Native calibrated distorted cameras are unsupported; select undistort")
            raise ValueError("Native distorted source cameras are unsupported")
    return CameraPreparationPlan(allocation, priors, tuple(undistort))


def prepare_capture(capture: PreparedCapture, plan: CameraPreparationPlan) -> PreparedCapture:
    """Allocate stable working camera IDs before frame selection or prediction."""
    source = capture.rec
    if any(source.camera(cid).model_name not in ("PINHOLE", "SIMPLE_PINHOLE") for cid in plan.undistort_camera_ids):
        raise ValueError("Undistort capture images before constructing the frontend")
    if plan.allocation == "input":
        return capture
    rec = pycolmap.Reconstruction()
    for index, image in enumerate(sorted(source.images.values(), key=lambda image: image.name)):
        cid = 0 if plan.allocation == "shared" else index
        camera = source.camera(image.camera_id)
        if cid not in rec.cameras:
            rec.add_camera_with_trivial_rig(
                pycolmap.Camera(
                    camera_id=cid,
                    model=camera.model,
                    width=camera.width,
                    height=camera.height,
                    params=camera.params,
                    has_prior_focal_length=plan.camera_priors.initialization == "supplied",
                )
            )
        prepared = pycolmap.Image(image_id=image.image_id, name=image.name, camera_id=cid, frame_id=image.frame_id)
        frame = pycolmap.Frame(frame_id=image.frame_id, rig_id=cid)
        frame.add_data_id(prepared.data_id)
        rec.add_frame(frame)
        rec.add_image(prepared)
    return replace(capture, rec=rec)


def initialize_selected_camera_geometry(working_rec, source_rec, image_names, plan: CameraPreparationPlan):
    """Resolve shared image dimensions from the selected images, without regrouping."""
    if plan.allocation != "shared":
        return
    names = set(image_names)
    images = sorted(
        (image for image in source_rec.images.values() if image.name in names), key=lambda image: image.name
    )
    camera = source_rec.camera(images[0].camera_id)
    signature = (camera.model_name, camera.width, camera.height)
    if any(
        (other.model_name, other.width, other.height) != signature
        for other in (source_rec.camera(image.camera_id) for image in images)
    ):
        raise ValueError("Shared preparation requires identical camera models and dimensions")
    prepared = pycolmap.Camera(camera.todict())
    prepared.camera_id = 0
    prepared.has_prior_focal_length = plan.camera_priors.initialization == "supplied"
    working_rec.cameras[0] = prepared
