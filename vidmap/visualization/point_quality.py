"""Geometry-only sparse-point quality used by visualization exporters."""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np
from tqdm import tqdm


def point_covariance_trace(model: Any, point3D: Any) -> float:
    """Compute the established triangulation score ``trace(inv(sum J.T @ J))``."""
    import pycolmap

    point = np.asarray(point3D.xyz, dtype=np.float64)
    hessian = np.zeros((3, 3), dtype=np.float64)
    for element in point3D.track.elements:
        if element.image_id not in model.images:
            continue
        image = model.images[element.image_id]
        camera = model.cameras[image.camera_id]
        params = np.asarray(camera.params, dtype=np.float64)
        if camera.model in (
            pycolmap.CameraModelId.SIMPLE_PINHOLE,
            pycolmap.CameraModelId.SIMPLE_RADIAL,
            pycolmap.CameraModelId.RADIAL,
            pycolmap.CameraModelId.SIMPLE_RADIAL_FISHEYE,
            pycolmap.CameraModelId.RADIAL_FISHEYE,
            pycolmap.CameraModelId.SIMPLE_DIVISION,
            pycolmap.CameraModelId.SIMPLE_FISHEYE,
        ):
            fx = fy = params[0]
        else:
            fx, fy = params[:2]
        cam_from_world = image.cam_from_world()
        rotation = np.asarray(cam_from_world.rotation.matrix(), dtype=np.float64)
        camera_point = rotation @ point + np.asarray(cam_from_world.translation, dtype=np.float64)
        depth = camera_point[2]
        if depth <= 0:
            continue
        jacobian = (
            np.asarray(
                [
                    [fx / depth, 0.0, -fx * camera_point[0] / depth**2],
                    [0.0, fy / depth, -fy * camera_point[1] / depth**2],
                ]
            )
            @ rotation
        )
        hessian += jacobian.T @ jacobian
    if np.linalg.matrix_rank(hessian) < 3:
        return float("inf")
    return float(np.trace(np.linalg.inv(hessian)))


def lowest_covariance_point_ids(
    model: Any,
    point_ids: Iterable[int],
    percentile: float,
) -> np.ndarray:
    """Retain the requested lowest-trace covariance percentile."""
    if not 0 < percentile <= 100:
        raise ValueError("point covariance percentile must be in (0, 100]")
    candidates = np.asarray(
        [int(point_id) for point_id in point_ids if model.point3D(int(point_id)).track.length() >= 2],
        dtype=np.int64,
    )
    scores = np.asarray(
        [
            point_covariance_trace(model, model.point3D(int(point_id)))
            for point_id in tqdm(
                candidates,
                desc="Computing point covariances",
                unit="point",
            )
        ],
        dtype=np.float64,
    )
    keep = int(len(candidates) * percentile / 100.0)
    order = np.lexsort((candidates, scores))
    return np.sort(candidates[order[:keep]])
