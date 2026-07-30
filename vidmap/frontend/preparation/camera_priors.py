"""Initialise per-camera intrinsics from GeoCalib / VGC priors.

Reads the per-camera focal + principal point from the ``geocalib_batch`` H5
file when GeoCalib is enabled, else seeds intrinsics from
the median of GT cameras when view-graph calibration is enabled. Also stamps
"""

from __future__ import annotations

import logging
from pathlib import Path

import h5py
import numpy as np
import pycolmap

from vidmap.frontend.cache import CacheMetadataMismatch, IncrementalArtifactContract, validate_incremental_cache

logger = logging.getLogger(__name__)


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


def apply_camera_priors(
    *,
    use_geocalib: bool,
    view_graph_calibration: bool,
    geocalib_batch_path: Path | None,
    geocalib_batch_artifact: IncrementalArtifactContract | None,
    source_reconstruction: pycolmap.Reconstruction,
    reconstruction: pycolmap.Reconstruction,
) -> None:
    """Apply configured camera priors to the preparation reconstruction."""
    if use_geocalib:
        if geocalib_batch_path is None or geocalib_batch_artifact is None:
            raise CacheMetadataMismatch("GeoCalib is enabled but its batch artifact is unavailable")
        validate_incremental_cache(
            geocalib_batch_path,
            geocalib_batch_artifact.metadata,
            geocalib_batch_artifact.expected_items,
        )
        with h5py.File(geocalib_batch_path, "r") as fd:
            batch_grp = fd["batch_calibration"]
            focal = batch_grp["focal"][:]
            principal_point = batch_grp["principal_point"][:]

        cids = {image.camera_id for image in reconstruction.images.values()}
        for cid in cids:
            _apply_calibration(reconstruction.cameras[cid], focal, principal_point)
    elif view_graph_calibration:
        gt_focals = [camera.mean_focal_length() for camera in source_reconstruction.cameras.values()]
        if not gt_focals:
            raise ValueError("View-graph calibration requires at least one source camera")
        if not np.isfinite(gt_focals).all() or np.any(np.asarray(gt_focals) <= 0):
            raise ValueError("View-graph calibration source cameras have invalid focal lengths")
        median_focal = np.median(gt_focals)
        first_camera = next(iter(source_reconstruction.cameras.values()))
        shared_principal_point = (
            first_camera.principal_point_x,
            first_camera.principal_point_y,
        )

        cids = {image.camera_id for image in reconstruction.images.values()}
        for cid in cids:
            _apply_shared_focal(reconstruction.cameras[cid], median_focal, shared_principal_point)
        logger.info("View-graph calibration initialized with median GT focal %.1f", median_focal)
