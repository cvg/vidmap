"""Initialize cameras from the extraction's calibration predictions."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import pycolmap

from vidmap.calibration import CalibrationEvidence, CalibrationObservation
from vidmap.frontend.cache.metadata import IncrementalArtifactContract


@dataclass(frozen=True)
class CalibrationResult:
    """Certified original predictions and their working-camera associations."""

    path: Path
    artifact: IncrementalArtifactContract
    evidence: CalibrationEvidence

    @classmethod
    def from_artifact(cls, *, path, reconstruction, image_names, options, artifact):
        images = {image.name: image for image in reconstruction.images.values()}
        shared = options.inference == "selected_batch"
        if shared:
            camera_ids = {images[name].camera_id for name in image_names}
            if len(camera_ids) != 1:
                raise ValueError("Selected shared calibration requires one camera")
            observations = (CalibrationObservation(camera_ids.pop(), "batch_calibration", None),)
        else:
            observations = tuple(CalibrationObservation(images[name].camera_id, name, name) for name in image_names)
        evidence = CalibrationEvidence(
            options.estimator,
            options.inference,
            "PINHOLE",
            "working_images",
            observations,
        )
        return cls(Path(path), artifact, evidence)


def apply_calibration(result: CalibrationResult, reconstruction: pycolmap.Reconstruction) -> None:
    """Initialize from original predictions without changing the stored targets."""
    evidence = result.evidence
    images = {image.name: image.camera_id for image in reconstruction.images.values()}
    if {observation.camera_id for observation in evidence.observations} != set(images.values()):
        raise ValueError("Calibration observations disagree with initialization cameras")
    if (
        evidence.inference == "per_view"
        and {observation.image_name: observation.camera_id for observation in evidence.observations} != images
    ):
        raise ValueError("Calibration observations disagree with initialization image membership")
    with h5py.File(result.path, "r") as hfile:
        apply_camera_priors(
            results=[hfile[observation.key] for observation in evidence.observations],
            shared=evidence.inference == "selected_batch",
            reconstruction=reconstruction,
        )


def _validate_intrinsics(values, *, label: str, positive: bool) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.shape != (2,) or not np.isfinite(values).all() or (positive and np.any(values <= 0)):
        requirement = "two positive finite values" if positive else "two finite values"
        raise ValueError(f"{label} must contain {requirement}, got {values!r}")
    return values


def _apply_calibration(camera: pycolmap.Camera, focal, principal_point) -> None:
    """Update model-declared focal/principal-point parameters and preserve distortion."""
    focal = _validate_intrinsics(focal, label="GeoCalib focal", positive=True)
    principal_point = _validate_intrinsics(principal_point, label="GeoCalib principal point", positive=False)
    focal_indices = tuple(camera.focal_length_idxs())
    principal_indices = tuple(camera.principal_point_idxs())
    if len(focal_indices) == 1:
        if not np.isclose(focal[0], focal[1], rtol=1e-6, atol=1e-6):
            raise ValueError(
                f"Camera model {camera.model.name} has one focal parameter but GeoCalib returned anisotropic focal"
            )
        focal_values = (float(focal.mean()),)
    elif len(focal_indices) == 2:
        focal_values = tuple(float(value) for value in focal)
    else:
        raise ValueError(f"Unsupported focal parameter layout for camera model {camera.model.name}")
    if len(principal_indices) != 2:
        raise ValueError(f"Unsupported principal-point layout for camera model {camera.model.name}")

    params = camera.params.copy()
    params[list(focal_indices)] = focal_values
    params[list(principal_indices)] = principal_point
    camera.params = params


def _apply_shared_focal(camera: pycolmap.Camera, focal: float, principal_point=None) -> None:
    if not np.isfinite(focal) or focal <= 0:
        raise ValueError(f"VGC focal prior must be positive and finite, got {focal!r}")
    params = camera.params.copy()
    params[list(camera.focal_length_idxs())] = float(focal)
    if principal_point is not None:
        principal_point = _validate_intrinsics(
            principal_point,
            label="VGC principal point",
            positive=False,
        )
        params[list(camera.principal_point_idxs())] = principal_point
    camera.params = params


def apply_camera_priors(*, results, shared, reconstruction):
    """Use the shared estimate or median predicted focal for the input cameras."""
    if not results:
        raise ValueError("Predicted initialization requires calibration results")
    cameras = {image.camera_id: reconstruction.cameras[image.camera_id] for image in reconstruction.images.values()}
    if any(c.model.name not in {"PINHOLE", "SIMPLE_PINHOLE"} for c in cameras.values()):
        raise ValueError("Unsupported camera model for calibration initialization")
    if len({(c.model.name, c.width, c.height) for c in cameras.values()}) != 1:
        raise ValueError("Shared initialization requires identical camera models and dimensions")
    camera = next(iter(cameras.values()))
    if any(tuple(result["image_size"]) != (camera.width, camera.height) for result in results):
        raise ValueError("Calibration dimensions disagree with cameras")
    if shared:
        if len(results) != 1:
            raise ValueError("Shared inference requires one camera result")
        K = np.asarray(results[0]["K"], dtype=np.float32)
        for camera in cameras.values():
            _apply_calibration(camera, K[[0, 1], [0, 1]], K[:2, 2])
    else:
        intrinsics = np.asarray([result["K"] for result in results], dtype=np.float32)
        median_focal = float(np.median((intrinsics[:, 0, 0] + intrinsics[:, 1, 1]).astype(np.float64) / 2.0))
        shared_principal_point = (camera.width / 2, camera.height / 2)
        for camera in cameras.values():
            _apply_shared_focal(camera, median_focal, shared_principal_point)
