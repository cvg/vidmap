"""Translate bundle-adjustment configuration into native options and records."""

from collections.abc import Sequence

import numpy as np

from vidmap.mapper.native.extension import native
from vidmap.mapper.native.losses import build_named_loss_config, build_typed_loss_config


def build_bundle_adjustment_options(
    *,
    image_order: Sequence[int],
    camera_ids: Sequence[int],
    variable_point3D_ids: Sequence[int],
    optimize_intrinsics: bool,
    refine_principal_point: bool,
    fix_rotations: bool,
    fix_all_poses: bool = False,
    reprojection_loss: str,
    reprojection_scale: float,
    reprojection_weight: float,
    num_threads: int | None,
):
    native_options = native.BundleAdjustmentOptions()
    native_options.image_order = list(image_order)
    native_options.constant_camera_ids = [] if optimize_intrinsics else list(camera_ids)
    native_options.variable_point3D_ids = list(variable_point3D_ids)
    native_options.reprojection_loss = build_named_loss_config(
        reprojection_loss,
        scale=reprojection_scale,
        weight=reprojection_weight,
    )
    native_options.refine_focal_length = True
    native_options.refine_principal_point = refine_principal_point
    native_options.refine_extra_params = True
    native_options.refine_points3D = True
    native_options.fix_first_pose = True
    native_options.fix_rotations = fix_rotations
    native_options.fix_all_poses = fix_all_poses
    native_options.use_log_depth_residual = True
    native_options.num_threads = -1 if num_threads is None else int(num_threads)
    return native_options


def make_intrinsics_prior_record(camera_id: int, values: np.ndarray, stddevs: np.ndarray):
    record = native.IntrinsicsPriorRecord()
    record.camera_id = int(camera_id)
    record.values = values
    record.stddevs = stddevs
    return record


def make_depth_constraint_record(
    image_id: int,
    point3D_id: int,
    depth: float,
    constraint_loss_type,
    scale: float,
    weight: float,
):
    record = native.DepthConstraintRecord()
    record.image_id = int(image_id)
    record.point3D_id = int(point3D_id)
    record.depth = float(depth)
    record.loss = build_typed_loss_config(constraint_loss_type, scale=scale, weight=weight)
    return record


def make_depth_scale_record(
    *,
    image_id: int,
    shift_scale: np.ndarray,
    fix_scale: bool,
    use_scale_prior: bool,
    scale_prior_stddev: float,
    scale_prior_loss: str,
    scale_prior_weight: float,
):
    record = native.DepthScaleRecord()
    record.image_id = int(image_id)
    record.shift_scale = np.asarray(shift_scale, dtype=np.float64)
    record.fix_shift = True
    record.fix_scale = fix_scale
    record.use_scale_prior = use_scale_prior
    record.scale_prior_stddev = scale_prior_stddev
    record.scale_prior_loss = build_named_loss_config(scale_prior_loss, weight=scale_prior_weight)
    return record
