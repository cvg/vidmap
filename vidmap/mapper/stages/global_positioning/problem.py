"""VidMap constraints and solve schedule on COLMAP's prepared GP problem."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pyceres
import pycolmap
from vidmap_native import global_positioning as gp_costs

from vidmap.mapper.native.extension import native
from vidmap.mapper.native.solver_backend import SolverDiagnostics, apply_solver_backend
from vidmap.mapper.options.positioning import GPOptions, LossConfig
from vidmap.mapper.playback_trace import SolverPlaybackOptions


@dataclass(frozen=True)
class TemporalAccelerationPrior:
    prev_image_id: int
    image_id: int
    next_image_id: int
    dt_prev: float
    dt_next: float
    sqrt_observation_count: float

    def __post_init__(self):
        ids = (self.prev_image_id, self.image_id, self.next_image_id)
        values = (self.dt_prev, self.dt_next, self.sqrt_observation_count)
        if len(set(ids)) != 3 or any(not np.isfinite(v) or v <= 0 for v in values):
            raise ValueError("invalid temporal acceleration prior")


@dataclass
class GlobalPositioningDiagnostics(SolverDiagnostics):
    num_bata_residuals: int = 0
    num_metric_depth_residuals: int = 0
    num_scale_prior_residuals: int = 0
    num_temporal_acceleration_residuals: int = 0
    num_regular_observations_used: int = 0
    num_loop_closure_observations_used: int = 0
    num_bata_scales: int = 0
    num_depth_map_scales: int = 0
    num_camera_centers: int = 0
    num_point3D_parameters: int = 0


@dataclass
class GlobalPositioningResult:
    success: bool = False
    depth_map_scales: dict = field(default_factory=dict)
    initial_frame_centers: dict = field(default_factory=dict)
    initial_point3D_xyz: dict = field(default_factory=dict)
    initial_bata_scales: dict = field(default_factory=dict)
    diagnostics: GlobalPositioningDiagnostics = field(default_factory=GlobalPositioningDiagnostics)


def _loss(config):
    return pyceres.LossFunction(dict(name=config.name, params=[config.scale], magnitude=config.weight))


def prepare_observations(reconstruction, sidecars, options: GPOptions, image_timeline=()):
    first = options.first_pass
    if first.sequential_support_warmup_rounds:
        if len(image_timeline) != reconstruction.num_images() or set(image_timeline) != set(reconstruction.images):
            raise ValueError("Sequential support requires an exact unique image timeline")
    return gp_costs.prepare_observations(
        reconstruction,
        sidecars,
        options.track_filter.min_num_views_per_track,
        options.common.use_lc_observations,
        image_timeline,
        first.sequential_support_observations_per_track if first.sequential_support_warmup_rounds else 0,
    )


def _geometry_selections(prepared, options, normal, uncalibrated, loop_loss, warmup):
    default_loss = warmup[0] if warmup else normal
    losses = [default_loss]
    groups = [[]]
    for (loop, calibrated, support), selection in prepared.items():
        downweight = options.common.apply_uncalibrated_loss_downweight and not calibrated
        if warmup and support:
            loss = warmup[1 if downweight else 0]
        elif loop and loop_loss is not None:
            loss = loop_loss
        else:
            loss = uncalibrated if downweight else normal
        index = next((i for i, value in enumerate(losses) if value is loss and (not loop or i != 0)), None)
        if index is None:
            index = len(losses)
            losses.append(loss)
            groups.append([])
        groups[index].append(selection)
    return [gp_costs.combine_observations(group) for group in groups], losses


def run_global_positioning(
    options,
    reconstruction,
    sidecars,
    prepared,
    pixel_stddev,
    *,
    second_pass=False,
    initial_depth_map_scales=None,
    temporal_priors=(),
    depth_outliers_marked=False,
    depth_outlier_masks=None,
    playback_options=None,
    playback_callback=None,
    capture_state=False,
):
    """Prepare, extend and solve one GP pass on the caller's working reconstruction."""
    common = options.common
    pass_options = options.second_pass if second_pass else options.first_pass
    warmup_rounds = 0 if second_pass else options.first_pass.sequential_support_warmup_rounds
    initial_depth_map_scales = initial_depth_map_scales or {}
    if any(not np.isfinite(value) or value <= 0 for value in initial_depth_map_scales.values()):
        raise ValueError("invalid initial depth-map scale")
    if options.solver_backend.linear_solver == "dense_schur":
        raise ValueError("dense_schur is not supported for global positioning")
    playback_options = playback_options or SolverPlaybackOptions()
    playback_options.validate()
    images = reconstruction.images
    for image in images.values():
        if image.has_pose and not image.is_ref_in_frame():
            raise ValueError("VidMap global positioning requires reference cameras")

    # GP1 initializes positions; GP2 starts from the GP1 result.
    gp_options = pycolmap.GlobalPositionerOptions(
        use_gpu=options.solver_backend.use_cuda,
        min_num_images_gpu_solver=0,
        random_seed=common.random_seed,
        generate_random_positions=not second_pass,
        generate_random_points=not second_pass,
        initialize_scales_from_geometry=second_pass and options.first_pass.initialize_warm_start_scales,
        optimize_scales=common.optimize_depth_map_scales,
        min_num_view_per_track=options.track_filter.min_num_views_per_track,
        fix_first_scale=False,
        experimental_observation_stddev=pixel_stddev,
        uncalibrated_observation_weight=1.0,
    )
    normal_geometry = _loss(
        pass_options.loss_normal_geometry
        if options.common.use_metric_depth_constraint
        else LossConfig(
            name=common.loss_function_type, scale=common.loss_function_scale, weight=common.loss_function_weight
        )
    )
    trivial = pyceres.TrivialLoss()
    warmup_losses = []
    if warmup_rounds:
        warmup = _loss(options.first_pass.sequential_support_loss)
        warmup_losses = [gp_costs.MutableLoss(warmup), gp_costs.MutableLoss(gp_costs.scaled_loss(warmup, 0.5))]
    uncalibrated_geometry = (
        gp_costs.scaled_loss(normal_geometry, 0.5)
        if options.common.apply_uncalibrated_loss_downweight and not options.common.use_metric_depth_constraint
        else normal_geometry
    )
    loop_config = pass_options.loss_lc_geometry
    loop_geometry = (
        _loss(loop_config)
        if (loop_config.name.upper() != "TRIVIAL" or loop_config.scale != 1.0 or loop_config.weight != 1.0)
        else None
    )

    # Create COLMAP's problem, then append the remaining observations.
    selections, geometry_losses = _geometry_selections(
        prepared, options, normal_geometry, uncalibrated_geometry, loop_geometry, warmup_losses
    )
    default_loss = geometry_losses[0]
    selected_tracks = gp_costs.selected_tracks(reconstruction, selections[0])
    gp_costs.swap_tracks(reconstruction, selected_tracks)
    try:
        # Eligibility was checked against the full tracks, before selecting support.
        gp_options.min_num_view_per_track = 1
        positioner = pycolmap.create_default_global_positioner(
            gp_options, pycolmap.PoseGraph(), reconstruction, default_loss
        )
    finally:
        gp_costs.swap_tracks(reconstruction, selected_tracks)
    del selected_tracks
    problem = positioner.problem
    centers = {
        image_id: positioner.frame_center_parameter_block(image.frame_id)
        for image_id, image in images.items()
        if image.has_pose
    }
    centers = {i: center for i, center in centers.items() if center is not None}
    if len({images[image_id].frame_id for image_id in centers}) != len(centers):
        raise ValueError("VidMap global positioning requires one image per frame")
    random = np.random.default_rng(common.random_seed if common.random_seed >= 0 else None)
    if gp_options.generate_random_points:
        gp_costs.initialize_points(problem, reconstruction, options.track_filter.min_num_views_per_track, random)

    result = GlobalPositioningResult()
    diagnostics = result.diagnostics
    if capture_state:
        point_ids, points_xyz, _ = native.point3D_table(reconstruction)
        result.initial_point3D_xyz = dict(zip(point_ids.tolist(), points_xyz))
    gp_costs.collect_default_observations(problem, reconstruction, centers, selections[0])

    extra_centers = {}
    for image_id, image in reconstruction.images.items():
        if image_id in centers or not image.has_pose:
            continue
        if gp_options.generate_random_positions:
            center = random.uniform(-100.0, 100.0, 3)
        else:
            center = image.projection_center().copy()
        extra_centers[image_id] = centers[image_id] = center

    # Keep appended parameter storage alive through both solves.
    _appended_scales = [
        gp_costs.append_observations(
            positioner,
            reconstruction,
            selection,
            centers,
            loss,
            pixel_stddev,
            gp_options.initialize_scales_from_geometry,
            gp_options.optimize_scales,
        )
        for selection, loss in zip(selections[1:], geometry_losses[1:])
        if len(selection)
    ]
    if not options.common.use_metric_depth_constraint and gp_options.optimize_scales:
        gp_costs.fix_first_observation_scale(problem, selections)
    regular, loop, num_points, has_support = gp_costs.observation_counts(selections)
    if warmup_rounds and not has_support:
        raise ValueError("Sequential support has no active observations")
    diagnostics.num_regular_observations_used = regular
    diagnostics.num_loop_closure_observations_used = loop
    diagnostics.num_bata_residuals = diagnostics.num_bata_scales = regular + loop
    if capture_state:
        result.initial_bata_scales = _scale_values(selections)

    # Add metric depth constraints and per-image scale priors.
    depth_scales = {}
    if options.common.use_metric_depth_constraint:
        normal_depth = _loss(pass_options.loss_normal_depth)
        loop_depth = _loss(pass_options.loss_lc_depth)
        outlier_depth = _loss(options.track_filter.loss_normal_depth_outlier) if depth_outliers_marked else trivial
        depth_groups, counts = gp_costs.split_depth_observations(selections, sidecars, depth_outlier_masks or {})
        for image_id in counts:
            value = initial_depth_map_scales.get(image_id, 1.0)
            depth_scales[image_id] = np.array(
                [np.log(max(1e-9, value)) if common.use_log_scale_for_depth_map_scales else value], dtype=float
            )
        for (loop, outlier), selection in depth_groups.items():
            loss = loop_depth if loop else outlier_depth if outlier else normal_depth
            gp_costs.append_depth_observations(
                problem,
                reconstruction,
                sidecars,
                selection,
                depth_scales,
                loss,
                common.use_log_scale_for_depth_map_scales,
                (
                    native.MetricDepthResidualType.LOG_LINEAR
                    if common.smooth_log_linear_transition
                    else (
                        native.MetricDepthResidualType.LOG
                        if second_pass and options.second_pass.use_log_depth_residual
                        else native.MetricDepthResidualType.LINEAR
                    )
                ),
                common.zero_residuals_behind_camera,
                common.log_linear_threshold,
            )
        diagnostics.num_metric_depth_residuals = sum(counts.values())
        prior_loss = _loss(LossConfig(name=pass_options.scale_reg_loss_name, weight=pass_options.scale_reg_weight))
        prior_cost = pyceres.factors.NormalPrior(
            np.array([0.0 if common.use_log_scale_for_depth_map_scales else 1.0]),
            np.array([[pass_options.scale_prior_stddev**2]]),
        )
        for image_id, count in counts.items():
            scale = depth_scales[image_id]
            problem.add_residual_block(prior_cost, gp_costs.scaled_loss(prior_loss, count), [scale])
            if not common.use_log_scale_for_depth_map_scales:
                problem.set_parameter_lower_bound(scale, 0, 1e-5)
        diagnostics.num_scale_prior_residuals = len(counts)
    # Add optional temporal acceleration priors.
    temporal = options.temporal_acceleration
    prefix = "second_pass" if second_pass else "first_pass"
    temporal_weight = getattr(temporal, prefix + "_weight")
    if temporal_weight > 0:
        if not temporal_priors:
            raise ValueError("enabled temporal acceleration requires priors and positive weight")
        loss = gp_costs.dead_zone_huber_loss(
            getattr(temporal, prefix + "_dead_zone") / temporal.stddev,
            getattr(temporal, prefix + "_huber_width") / temporal.stddev,
        )
        for prior in temporal_priors:
            cost = gp_costs.temporal_acceleration_cost(prior.dt_prev, prior.dt_next, temporal.stddev)
            weighted = gp_costs.scaled_loss(loss, temporal_weight * prior.sqrt_observation_count**2)
            problem.add_residual_block(
                cost,
                weighted,
                [centers[i] for i in (prior.prev_image_id, prior.image_id, prior.next_image_id)],
            )
            diagnostics.num_temporal_acceleration_residuals += 1

    for image_id in list(extra_centers):
        if not problem.has_parameter_block(extra_centers[image_id]):
            del extra_centers[image_id]
            del centers[image_id]
    if capture_state:
        result.initial_frame_centers = {images[i].frame_id: center.copy() for i, center in centers.items()}

    # Configure parameter ordering, solver options and playback.
    positioner.extend_parameter_block_ordering(
        [(block, 2) for arrays in (extra_centers, depth_scales) for block in arrays.values()]
    )
    solver_options = pyceres.SolverOptions(positioner.solver_options)
    apply_solver_backend(solver_options, options.solver_backend)
    if common.num_threads is not None and common.num_threads > 0:
        solver_options.num_threads = common.num_threads
    configure_solver_tolerances(solver_options, options, second_pass=second_pass)
    capture = _Playback(
        playback_callback,
        playback_options,
        reconstruction,
        centers,
        selections,
        geometry_losses,
        problem,
    )
    if playback_callback is not None:
        solver_options.update_state_every_iteration = True
        solver_options.callbacks = [capture]
    capture.emit("initial", -1)

    # GP1 warmup.
    if warmup_rounds:
        solver_options.max_num_iterations = warmup_rounds
        saved_radius = solver_options.max_trust_region_radius
        solver_options.max_trust_region_radius = 1.0e4
        _solve(problem, solver_options, capture, -1)
        solver_options.max_trust_region_radius = saved_radius
        warmup_losses[0].reset(normal_geometry)
        warmup_losses[1].reset(
            normal_geometry
            if options.common.use_metric_depth_constraint
            else gp_costs.scaled_loss(normal_geometry, 0.5)
        )
    # Main solve for the current pass (GP1 or GP2).
    solver_options.max_num_iterations = pass_options.max_iterations
    summary = _solve(problem, solver_options, capture, warmup_rounds)
    diagnostics.update(summary)
    capture.emit("final", warmup_rounds + diagnostics.num_iterations - 1)

    # Write optimized centers back to camera poses.
    # Finalize also writes factory centers whose observations were all rejected.
    inactive_translations = {
        image.frame_id: image.frame.rig_from_world.translation.copy()
        for image_id, image in images.items()
        if image.has_pose and image_id not in centers
    }
    if not positioner.finalize(summary):
        raise RuntimeError("Global positioning finalization failed")
    for frame_id, translation in inactive_translations.items():
        reconstruction.frame(frame_id).rig_from_world.translation = translation
    for image_id, center in extra_centers.items():
        pose = reconstruction.frame(images[image_id].frame_id).rig_from_world
        pose.translation = -(pose.rotation * center)

    result.success = True
    result.depth_map_scales = {
        i: float(np.exp(v[0]) if common.use_log_scale_for_depth_map_scales else v[0]) for i, v in depth_scales.items()
    }
    diagnostics.num_depth_map_scales = len(depth_scales)
    diagnostics.num_camera_centers = len(centers)
    diagnostics.num_point3D_parameters = num_points
    del capture, problem, positioner
    return result


def configure_solver_tolerances(solver, options: GPOptions, *, second_pass: bool) -> None:
    for name, default in (("function_tolerance", 1e-5), ("gradient_tolerance", 1e-10), ("parameter_tolerance", 1e-8)):
        value = (
            getattr(options.first_pass, name + "_when_second")
            if options.second_pass.enabled and not second_pass
            else None
        )
        setattr(solver, name, default if value is None else value)


def _scale_values(selections):
    values = {}
    for selection in selections:
        values.update(selection.scale_values())
    return values


class _Playback(pyceres.IterationCallback):
    def __init__(self, callback, options, reconstruction, centers, observations, losses, problem):
        super().__init__()
        self.callback, self.options = callback, options
        self.centers = centers
        self.reconstruction = reconstruction
        self.observations = observations
        self.losses = losses
        if callback is None:
            return
        self.image_ids = list(options.image_ids) or sorted(centers)
        self.point_ids = gp_costs.playback_point_ids(problem, reconstruction, list(options.point3D_ids), 200000)
        if any(i not in centers for i in self.image_ids):
            raise ValueError("Playback image is not active in global positioning")
        self.offset = 0

    def __call__(self, summary):
        iteration = summary.iteration + self.offset
        if iteration >= 0 and iteration % self.options.snapshot_every_n_iterations == 0:
            self.emit("iteration", iteration)
        return pyceres.CallbackReturnType.SOLVER_CONTINUE

    def emit(self, phase, iteration):
        if self.callback is None:
            return
        pairs, counts, scores = gp_costs.playback_edges(self.observations, self.losses, set(self.image_ids))
        image_ids = self.image_ids
        point_ids = self.point_ids
        camera_values, point_values = native.playback_coordinates(
            self.reconstruction, image_ids, point_ids, self.centers
        )
        self.callback(
            dict(
                phase=phase,
                iteration=iteration,
                image_ids=np.array(image_ids),
                centers=camera_values,
                point_ids=point_ids,
                points_xyz=point_values,
                lc_pairs=pairs,
                lc_support_count=counts,
                lc_raw_score=scores,
            )
        )


def _solve(problem, options, capture, offset):
    capture.offset = offset
    summary = pyceres.SolverSummary()
    pyceres.solve(options, problem, summary)
    if not summary.IsSolutionUsable():
        raise RuntimeError("Global positioning failed")
    return summary
