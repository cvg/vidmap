"""Single-pair relative pose estimation with optional depth support."""

from typing import Literal, TypeAlias, TypedDict

import numpy as np
import poselib
import pycolmap

MIN_MDRP_VALID_DEPTH_CORRESPONDENCES = 3


class InvalidPairPoseResult(TypedDict):
    is_valid: Literal[False]
    invalid_reason: str


class ValidPairPoseResult(TypedDict):
    is_valid: Literal[True]
    cam2_from_cam1: pycolmap.Rigid3d
    inliers: np.ndarray
    rel_depth_scale: float


PairPoseResult: TypeAlias = InvalidPairPoseResult | ValidPairPoseResult


def estimate_relative_pose_for_pair(
    pair_args,
    *,
    images,
    ransac_options,
    bundle_options,
    camera_poselib_cache,
    image_feature_cache,
    image_depth_cache,
    image_valid_cache,
    native_cameras=None,
) -> tuple[int, PairPoseResult]:
    """Estimate a relative pose for a single view-graph pair.

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
            raise ValueError(f"Malformed relative-pose inputs for {pair_label}: {name} must be numeric") from exc
    points2D_1 = values["points2D_1"]
    points2D_2 = values["points2D_2"]
    depths1 = values["depths1"]
    depths2 = values["depths2"]
    valid1 = np.asarray(valid1, dtype=bool)
    valid2 = np.asarray(valid2, dtype=bool)

    if points2D_1.ndim != 2 or points2D_1.shape[1] != 2:
        raise ValueError(f"Malformed relative-pose inputs for {pair_label}: points2D_1 must have shape (N, 2)")
    if points2D_2.ndim != 2 or points2D_2.shape[1] != 2:
        raise ValueError(f"Malformed relative-pose inputs for {pair_label}: points2D_2 must have shape (N, 2)")
    for name, value in (
        ("depths1", depths1),
        ("depths2", depths2),
        ("valid1", valid1),
        ("valid2", valid2),
    ):
        if value.ndim != 1:
            raise ValueError(f"Malformed relative-pose inputs for {pair_label}: {name} must have shape (N,)")

    lengths = {
        "points2D_1": points2D_1.shape[0],
        "points2D_2": points2D_2.shape[0],
        "depths1": depths1.shape[0],
        "depths2": depths2.shape[0],
        "valid1": valid1.shape[0],
        "valid2": valid2.shape[0],
    }
    if len(set(lengths.values())) != 1:
        raise ValueError(f"Malformed relative-pose inputs for {pair_label}: inconsistent lengths {lengths}")

    valid = valid1 & valid2
    invalid_reason = None
    if np.count_nonzero(valid) < MIN_MDRP_VALID_DEPTH_CORRESPONDENCES and native_cameras is None:
        invalid_reason = "too_few_valid_depth_correspondences"
    elif not np.isfinite(points2D_1[valid]).all() or not np.isfinite(points2D_2[valid]).all():
        invalid_reason = "non_finite_valid_points"
    elif not np.isfinite(depths1[valid]).all() or not np.isfinite(depths2[valid]).all():
        invalid_reason = "non_finite_valid_depths"
    elif np.any(depths1[valid] <= 0) or np.any(depths2[valid] <= 0):
        invalid_reason = "non_positive_valid_depths"
    if invalid_reason is not None:
        return image_pair_id, {"is_valid": False, "invalid_reason": invalid_reason}

    if native_cameras is not None:
        return image_pair_id, _estimate_with_depthless_matches(
            points2D_1,
            points2D_2,
            depths1,
            depths2,
            valid,
            native_cameras[image1.camera_id],
            native_cameras[image2.camera_id],
            ransac_options,
            bundle_options,
        )

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
    result: ValidPairPoseResult = {
        "is_valid": True,
        "cam2_from_cam1": cam2_from_cam1,
        "inliers": inliers,
        "rel_depth_scale": scale,
    }
    return image_pair_id, result


def _estimate_with_depthless_matches(points1, points2, depths1, depths2, valid, camera1, camera2, ransac, bundle):
    """Seed with depth when available, then refine all inliers in original pixels."""
    from vidmap.mapper.native.extension import native
    from vidmap.utils.camera_geometry import poselib_camera, undistorted_pixels

    threshold = ransac["max_epipolar_error"]
    inliers = np.zeros(len(points1), dtype=bool)
    scale = 1.0
    if valid.sum() >= MIN_MDRP_VALID_DEPTH_CORRESPONDENCES:
        geometry, _ = poselib.estimate_monodepth_relative_pose(
            undistorted_pixels(camera1, points1[valid]),
            undistorted_pixels(camera2, points2[valid]),
            depths1[valid],
            depths2[valid],
            poselib_camera(camera1, undistorted=True),
            poselib_camera(camera2, undistorted=True),
            ransac,
            bundle,
        )
        pose = pycolmap.Rigid3d(pycolmap.Rotation3d(geometry.pose.R), geometry.pose.t)
        scale = geometry.scale
        inliers = native.relative_pose_residuals(pose, camera1, points1, camera2, points2) <= threshold**2
    if inliers.sum() < 5:
        result = pycolmap.estimate_relative_pose(
            camera1,
            points1,
            camera2,
            points2,
            pycolmap.RANSACOptions(max_error=threshold, max_num_trials=ransac["max_iterations"]),
        )
        if result is None:
            return {"is_valid": False, "invalid_reason": "too_few_geometric_inliers"}
        pose, inliers, scale = result["cam2_from_cam1"], result["inlier_mask"], 1.0
    refined = pycolmap.refine_relative_pose(pose, camera1, points1, camera2, points2, inliers)
    if refined is None:
        return {"is_valid": False, "invalid_reason": "relative_pose_refinement_failed"}
    pose = refined["cam2_from_cam1"]
    inliers = native.relative_pose_residuals(pose, camera1, points1, camera2, points2) <= threshold**2
    return {
        "is_valid": True,
        "cam2_from_cam1": pose,
        "inliers": np.flatnonzero(inliers).astype(np.int32),
        "rel_depth_scale": scale,
    }
