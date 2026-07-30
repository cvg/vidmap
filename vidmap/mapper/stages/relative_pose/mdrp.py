"""Pure single-pair MDRP pose estimation."""

from typing import Literal, TypeAlias, TypedDict

import numpy as np
import poselib
import pycolmap

MIN_MDRP_VALID_DEPTH_CORRESPONDENCES = 3


class InvalidMDRPResult(TypedDict):
    is_valid: Literal[False]
    invalid_reason: str


class ValidMDRPResult(TypedDict):
    is_valid: Literal[True]
    cam2_from_cam1: pycolmap.Rigid3d
    inliers: np.ndarray
    weight: float
    rel_depth_scale: float
    depth1_outliers: np.ndarray | None
    depth2_outliers: np.ndarray | None
    inlier_match_indices: np.ndarray | None


MDRPResult: TypeAlias = InvalidMDRPResult | ValidMDRPResult


def estimate_mdrp_pose_for_pair(
    pair_args,
    *,
    images,
    compute_reproj_error_outliers,
    reproj_outlier_threshold,
    ransac_options,
    bundle_options,
    camera_poselib_cache,
    image_feature_cache,
    image_depth_cache,
    image_valid_cache,
) -> tuple[int, MDRPResult]:
    """Estimate a relative pose for a single view-graph pair (MDRP).

    ``pair_args`` is ``(image_pair_id, image_id1, image_id2, matches)``; packing
    it as a single positional keeps the function compatible with
    ``thread_map``/``map`` which pass one element per call.

    """
    image_pair_id, image_id1, image_id2, matches = pair_args

    image1 = images[image_id1]
    image2 = images[image_id2]

    points2D_1 = image_feature_cache[image_id1][matches[:, 0]]
    points2D_2 = image_feature_cache[image_id2][matches[:, 1]]
    depths1 = image_depth_cache[image_id1][matches[:, 0]]
    depths2 = image_depth_cache[image_id2][matches[:, 1]]
    valid1 = image_valid_cache[image_id1][matches[:, 0]]
    valid2 = image_valid_cache[image_id2][matches[:, 1]]

    pair_label = f"{image_pair_id} ({image1.name} -> {image2.name})"
    values = {
        "points2D_1": points2D_1,
        "points2D_2": points2D_2,
        "depths1": depths1,
        "depths2": depths2,
    }
    for name, value in values.items():
        try:
            values[name] = np.asarray(value, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Malformed MDRP inputs for {pair_label}: {name} must be numeric") from exc
    points2D_1 = values["points2D_1"]
    points2D_2 = values["points2D_2"]
    depths1 = values["depths1"]
    depths2 = values["depths2"]
    valid1 = np.asarray(valid1, dtype=bool)
    valid2 = np.asarray(valid2, dtype=bool)

    if points2D_1.ndim != 2 or points2D_1.shape[1] != 2:
        raise ValueError(f"Malformed MDRP inputs for {pair_label}: points2D_1 must have shape (N, 2)")
    if points2D_2.ndim != 2 or points2D_2.shape[1] != 2:
        raise ValueError(f"Malformed MDRP inputs for {pair_label}: points2D_2 must have shape (N, 2)")
    for name, value in (
        ("depths1", depths1),
        ("depths2", depths2),
        ("valid1", valid1),
        ("valid2", valid2),
    ):
        if value.ndim != 1:
            raise ValueError(f"Malformed MDRP inputs for {pair_label}: {name} must have shape (N,)")

    lengths = {
        "points2D_1": points2D_1.shape[0],
        "points2D_2": points2D_2.shape[0],
        "depths1": depths1.shape[0],
        "depths2": depths2.shape[0],
        "valid1": valid1.shape[0],
        "valid2": valid2.shape[0],
    }
    if len(set(lengths.values())) != 1:
        raise ValueError(f"Malformed MDRP inputs for {pair_label}: inconsistent lengths {lengths}")

    valid = valid1 & valid2
    invalid_reason = None
    if np.count_nonzero(valid) < MIN_MDRP_VALID_DEPTH_CORRESPONDENCES:
        invalid_reason = "too_few_valid_depth_correspondences"
    elif not np.isfinite(points2D_1[valid]).all() or not np.isfinite(points2D_2[valid]).all():
        invalid_reason = "non_finite_valid_points"
    elif not np.isfinite(depths1[valid]).all() or not np.isfinite(depths2[valid]).all():
        invalid_reason = "non_finite_valid_depths"
    elif np.any(depths1[valid] <= 0) or np.any(depths2[valid] <= 0):
        invalid_reason = "non_positive_valid_depths"
    if invalid_reason is not None:
        return image_pair_id, {"is_valid": False, "invalid_reason": invalid_reason}

    camera_poselib1 = camera_poselib_cache[image1.camera_id]
    camera_poselib2 = camera_poselib_cache[image2.camera_id]
    pose_result = poselib.estimate_monodepth_relative_pose(
        points2D_1[valid],
        points2D_2[valid],
        depths1[valid],
        depths2[valid],
        camera_poselib1,
        camera_poselib2,
        ransac_options,
        bundle_options,
    )
    monodepth_geometry, inliers_info = pose_result
    pose_rel_calc = monodepth_geometry.pose
    scale = monodepth_geometry.scale
    q_coeffs = np.asarray(list(pose_rel_calc.q[1:]) + [pose_rel_calc.q[0]])
    rot = pycolmap.Rotation3d(q_coeffs.astype(np.float64))
    t_vec = np.asarray(pose_rel_calc.t).astype(np.float64)
    cam2_from_cam1 = pycolmap.Rigid3d(translation=t_vec, rotation=rot)
    inlier_mask = inliers_info["inliers"]
    inliers = np.where(inlier_mask)[0].astype(np.int32)
    weight = float(len(inliers) / len(matches))

    if compute_reproj_error_outliers:
        # Reprojection-error-based outlier detection (scale-only variant).
        # Unproject with depth, transform, reproject — depth outliers surface as big reprojection errors.
        R = pose_rel_calc.R
        t = np.asarray(pose_rel_calc.t)

        valid_indices = np.where(valid)[0]
        inlier_valid_indices = valid_indices[inlier_mask]
        pts1 = points2D_1[valid][inlier_mask]
        pts2 = points2D_2[valid][inlier_mask]
        d1 = depths1[valid][inlier_mask]
        d2 = depths2[valid][inlier_mask]

        # Direction 1 -> 2
        norm1 = np.array(camera_poselib1.unproject(pts1))
        P1_cam1 = np.column_stack([norm1 * d1[:, None], d1])
        P1_cam2 = (R @ P1_cam1.T).T + t
        with np.errstate(divide="ignore", invalid="ignore"):
            P1_cam2_norm = P1_cam2[:, :2] / P1_cam2[:, 2:3]
        pts2_proj = np.array(camera_poselib2.project(P1_cam2_norm))
        reproj_err_1to2 = np.linalg.norm(pts2_proj - pts2, axis=1)

        # Direction 2 -> 1
        d2_scaled = scale * d2
        norm2 = np.array(camera_poselib2.unproject(pts2))
        P2_cam2 = np.column_stack([norm2 * d2_scaled[:, None], d2_scaled])
        P2_cam1 = (R.T @ (P2_cam2 - t).T).T
        with np.errstate(divide="ignore", invalid="ignore"):
            P2_cam1_norm = P2_cam1[:, :2] / P2_cam1[:, 2:3]
        pts1_proj = np.array(camera_poselib1.project(P2_cam1_norm))
        reproj_err_2to1 = np.linalg.norm(pts1_proj - pts1, axis=1)

        threshold = reproj_outlier_threshold
        depth1_outlier = (reproj_err_1to2 > threshold) | ~np.isfinite(reproj_err_1to2)
        depth2_outlier = (reproj_err_2to1 > threshold) | ~np.isfinite(reproj_err_2to1)
    else:
        depth1_outlier = None
        depth2_outlier = None
        inlier_valid_indices = None

    result: ValidMDRPResult = {
        "is_valid": True,
        "cam2_from_cam1": cam2_from_cam1,
        "inliers": inliers,
        "weight": weight,
        "rel_depth_scale": scale,
        "depth1_outliers": depth1_outlier,
        "depth2_outliers": depth2_outlier,
        "inlier_match_indices": inlier_valid_indices,
    }
    return image_pair_id, result
