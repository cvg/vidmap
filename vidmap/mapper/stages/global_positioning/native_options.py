"""Translate global-positioning configuration into native options."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from vidmap.mapper.native.extension import native
from vidmap.mapper.native.losses import build_named_loss_config, loss_config_from_options, native_loss_type
from vidmap.mapper.native.solver_backend import apply_solver_backend
from vidmap.mapper.options.positioning import GPOptions


@dataclass(frozen=True)
class GlobalPositioningTolerances:
    function: float
    gradient: float
    parameter: float

    @classmethod
    def capture(cls, native_options) -> GlobalPositioningTolerances:
        return cls(
            function=float(native_options.function_tolerance),
            gradient=float(native_options.gradient_tolerance),
            parameter=float(native_options.parameter_tolerance),
        )

    def restore(self, native_options) -> None:
        native_options.function_tolerance = self.function
        native_options.gradient_tolerance = self.gradient
        native_options.parameter_tolerance = self.parameter


def _metric_depth_residual_type(smooth_transition: bool, use_log_residual: bool):
    if smooth_transition:
        return native.MetricDepthResidualType.LOG_LINEAR
    if use_log_residual:
        return native.MetricDepthResidualType.LOG
    return native.MetricDepthResidualType.LINEAR


def apply_global_positioning_policy(native_options, options: GPOptions):
    common = options.common
    native_options.use_loop_closure_observations = common.use_lc_observations
    native_options.loss.type = native_loss_type(common.loss_function_type)
    native_options.loss.weight = common.loss_function_weight
    native_options.loss.scale = common.loss_function_scale
    native_options.random_seed = common.random_seed
    native_options.initialize_warm_start_scales = options.first_pass.initialize_warm_start_scales
    native_options.use_parameter_block_ordering = True
    native_options.apply_uncalibrated_loss_downweight = common.apply_uncalibrated_loss_downweight
    if common.num_threads is not None:
        native_options.num_threads = common.num_threads
    apply_solver_backend(native_options, options.solver_backend)
    native_options.min_num_views_per_track = options.track_filter.min_num_views_per_track
    native_options.random_init_scale = common.random_init_scale
    native_options.parameter_ordering = {
        "grouped": native.GlobalPositioningOrdering.GROUPED,
        "deterministic_singleton_groups": native.GlobalPositioningOrdering.SINGLETON,
    }[common.parameter_ordering_strategy]
    native_options.center_mode = {
        "frame": native.GlobalPositioningCenterMode.FRAME,
        "image": native.GlobalPositioningCenterMode.IMAGE,
    }[common.camera_center_strategy]
    return native_options


def build_first_global_positioning_options(options: GPOptions, *, depth_outliers_marked: bool):
    native_options = native.GlobalPositioningOptions()
    common = options.common
    first = options.first_pass
    native_options.use_metric_depth_constraint = common.use_metric_depth_constraint
    native_options.optimize_scales = common.optimize_depth_map_scales
    native_options.loss_loop_closure_geometry = loss_config_from_options(first.loss_lc_geometry)
    native_options.loss_loop_closure_depth = loss_config_from_options(first.loss_lc_depth)
    native_options.loss_normal_geometry = loss_config_from_options(first.loss_normal_geometry)
    native_options.loss_normal_depth = loss_config_from_options(first.loss_normal_depth)
    native_options.loss_scale_prior = build_named_loss_config(first.scale_reg_loss_name, weight=first.scale_reg_weight)
    native_options.scale_prior_stddev = first.scale_prior_stddev
    native_options.max_num_iterations = first.max_iterations
    native_options.use_log_depth_map_scales = common.use_log_scale_for_depth_map_scales
    native_options.metric_depth_residual_type = _metric_depth_residual_type(common.smooth_log_linear_transition, False)

    original_tolerances = GlobalPositioningTolerances.capture(native_options)
    if options.second_pass.enabled:
        for name, value in (
            ("function_tolerance", first.function_tolerance_when_second),
            ("gradient_tolerance", first.gradient_tolerance_when_second),
            ("parameter_tolerance", first.parameter_tolerance_when_second),
        ):
            if value is not None:
                setattr(native_options, name, value)

    native_options.zero_residual_behind_camera = common.zero_residuals_behind_camera
    native_options.log_linear_threshold = common.log_linear_threshold
    native_options.use_initial_positions = False
    native_options.generate_scales = True
    if depth_outliers_marked:
        native_options.loss_normal_depth_outlier = loss_config_from_options(
            options.track_filter.loss_normal_depth_outlier
        )
    return apply_global_positioning_policy(native_options, options), original_tolerances


def configure_second_global_positioning_options(
    native_options,
    options: GPOptions,
    first_result,
    original_tolerances: GlobalPositioningTolerances,
    initial_frame_centers: Mapping[int, np.ndarray],
):
    original_tolerances.restore(native_options)
    second = options.second_pass
    native_options.scale_prior_stddev = second.scale_prior_stddev
    native_options.loss_scale_prior = build_named_loss_config(
        second.scale_reg_loss_name, weight=second.scale_reg_weight
    )
    native_options.max_num_iterations = second.max_iterations
    native_options.metric_depth_residual_type = _metric_depth_residual_type(
        options.common.smooth_log_linear_transition,
        second.use_log_depth_residual,
    )
    native_options.loss_loop_closure_geometry = loss_config_from_options(second.loss_lc_geometry)
    native_options.loss_loop_closure_depth = loss_config_from_options(second.loss_lc_depth)
    native_options.loss_normal_geometry = loss_config_from_options(second.loss_normal_geometry)
    native_options.loss_normal_depth = loss_config_from_options(second.loss_normal_depth)
    native_options.use_initial_positions = True
    native_options.generate_scales = False
    native_options.filter_depth_outliers = False
    native_options.initial_depth_map_scales = first_result.depth_map_scales
    native_options.initial_frame_centers = dict(initial_frame_centers)
    return apply_global_positioning_policy(native_options, options)


def configure_temporal_acceleration_options(native_options, options: GPOptions, *, stage: str, prior_specs):
    """Configure one GP pass from explicit adjacent-triplet prior records."""
    temporal = options.temporal_acceleration
    if stage == "gp1":
        weight = temporal.first_pass_weight
        dead_zone = temporal.first_pass_dead_zone
        huber_width = temporal.first_pass_huber_width
    elif stage == "gp2":
        weight = temporal.second_pass_weight
        dead_zone = temporal.second_pass_dead_zone
        huber_width = temporal.second_pass_huber_width
    else:
        raise ValueError(f"Unsupported temporal-acceleration stage: {stage!r}")

    enabled = weight > 0.0
    priors = []
    if enabled:
        for spec in prior_specs:
            prior = native.TemporalAccelerationPrior()
            prior.prev_image_id = int(spec["prev_image_id"])
            prior.image_id = int(spec["image_id"])
            prior.next_image_id = int(spec["next_image_id"])
            prior.dt_prev = float(spec["dt_prev"])
            prior.dt_next = float(spec["dt_next"])
            prior.sqrt_observation_count = float(spec["sqrt_observation_count"])
            priors.append(prior)

    native_options.use_temporal_acceleration_prior = enabled
    native_options.temporal_acceleration_priors = priors
    native_options.temporal_acceleration_prior_stddev = temporal.stddev
    native_options.temporal_acceleration_prior_weight = weight
    native_options.temporal_acceleration_prior_loss_dead_zone = dead_zone / temporal.stddev
    native_options.temporal_acceleration_prior_loss_huber_width = huber_width / temporal.stddev
    return native_options
