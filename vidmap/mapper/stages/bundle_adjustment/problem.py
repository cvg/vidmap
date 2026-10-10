"""Extend upstream bundle adjustment with VidMap's depth and focal priors."""

from dataclasses import dataclass, field

import numpy as np
import pyceres
import pycolmap
from vidmap_native import bundle_adjustment as ba_costs

from vidmap.mapper.native.extension import native
from vidmap.mapper.native.solver_backend import SolverDiagnostics
from vidmap.mapper.options.positioning import LossConfig
from vidmap.mapper.playback_trace import SolverPlaybackOptions


@dataclass
class DepthObservationBatch:
    image_id: int
    point_ids: np.ndarray
    depths: np.ndarray
    robust_scales: np.ndarray
    weights: np.ndarray
    robust_mask: np.ndarray
    loss_type: pycolmap.LossFunctionType


@dataclass
class DepthScale:
    image_id: int
    log_scale: float = 0.0
    fix_scale: bool = False
    use_scale_prior: bool = False
    scale_prior_stddev: float = 1.0
    scale_prior_loss: LossConfig = field(default_factory=lambda: LossConfig(name="trivial"))

    def validate(self):
        if (
            not np.isscalar(self.log_scale)
            or not np.isfinite(self.log_scale)
            or not np.isfinite(self.scale_prior_stddev)
            or self.scale_prior_stddev <= 0
        ):
            raise ValueError("invalid BA depth scale record")


@dataclass
class BundleAdjustmentDiagnostics(SolverDiagnostics):
    num_reprojection_residuals: int = 0
    num_depth_residuals: int = 0
    num_intrinsics_prior_residuals: int = 0
    num_relative_intrinsics_prior_residuals: int = 0
    num_scale_prior_residuals: int = 0


@dataclass
class BundleAdjustmentResult:
    success: bool = False
    log_depth_scales: dict[int, float] = field(default_factory=dict)
    diagnostics: BundleAdjustmentDiagnostics = field(default_factory=BundleAdjustmentDiagnostics)


def run_bundle_adjustment(
    options,
    config,
    depth_batches,
    depth_scales,
    intrinsics_priors,
    state,
    *,
    relative_intrinsics_priors=(),
    playback_callback=None,
    playback_options=None,
):
    """Solve a private reconstruction and publish its values only on success."""
    playback_options = playback_options or SolverPlaybackOptions()
    playback_options.validate()
    reconstruction = pycolmap.Reconstruction(state.reconstruction)
    image_ids = list(config.images)
    if len({p.camera_id for p in intrinsics_priors}) != len(intrinsics_priors):
        raise ValueError("duplicate BA log-focal prior camera")
    if len({s.image_id for s in depth_scales}) != len(depth_scales):
        raise ValueError("duplicate BA depth scale record")
    for record in (*depth_scales, *intrinsics_priors):
        record.validate()

    for image_id in image_ids:
        if not reconstruction.image(image_id).has_pose:
            raise ValueError("BA image does not have a pose")
    ceres_options = options.ceres
    owner = pycolmap.create_default_ceres_bundle_adjuster(options, config, reconstruction)
    # Keep this wrapper alive: PyCeres retains appended costs and arrays on it.
    problem = owner.problem
    diagnostics = BundleAdjustmentDiagnostics(num_reprojection_residuals=problem.num_residual_blocks())

    for prior in intrinsics_priors:
        camera = reconstruction.camera(prior.camera_id)
        params = camera.params
        if not problem.has_parameter_block(params):
            continue
        for index in camera.focal_length_idxs():
            problem.set_parameter_lower_bound(params, index, 1e-3)
        for focal, stddev in prior.observations:
            problem.add_residual_block(ba_costs.focal_prior_cost(camera, focal, stddev), prior.loss, [params])
            diagnostics.num_intrinsics_prior_residuals += 1

    # Relative log-focal priors between consecutive cameras (time-varying intrinsics).
    for prior in relative_intrinsics_priors:
        prior.validate()
        camera1, camera2 = reconstruction.camera(prior.camera_id1), reconstruction.camera(prior.camera_id2)
        params1, params2 = camera1.params, camera2.params
        if not (problem.has_parameter_block(params1) and problem.has_parameter_block(params2)):
            continue
        problem.add_residual_block(
            ba_costs.relative_focal_prior_cost(camera1, camera2, prior.target_log_ratio, prior.sigma_log_ratio),
            prior.loss,
            [params1, params2],
        )
        diagnostics.num_relative_intrinsics_prior_residuals += 1

    scale_records = {record.image_id: record for record in depth_scales}
    scales = {image_id: np.array([record.log_scale], dtype=float) for image_id, record in scale_records.items()}
    depth_losses = []
    active_images = set(image_ids)
    for batch in depth_batches:
        image_id = batch.image_id
        if image_id not in active_images:
            continue
        if image_id not in scales:
            raise ValueError("depth constraints require a scale record")
        losses, count = ba_costs.append_depth_observations(
            problem,
            reconstruction,
            image_id,
            batch.point_ids,
            batch.depths,
            batch.robust_scales,
            batch.weights,
            batch.robust_mask,
            batch.loss_type,
            scales[image_id],
            not options.refine_rig_from_world
            or config.has_constant_rig_from_world_pose(reconstruction.image(image_id).frame_id),
        )
        depth_losses.append(losses)
        diagnostics.num_depth_residuals += count

    for image_id, record in scale_records.items():
        scale = scales[image_id]
        if not problem.has_parameter_block(scale):
            continue
        if record.fix_scale:
            problem.set_parameter_block_constant(scale)
        if record.use_scale_prior and record.scale_prior_loss.weight > 0:
            problem.add_residual_block(
                pyceres.factors.NormalPrior(np.zeros(1), np.array([[record.scale_prior_stddev**2]])),
                pyceres.LossFunction(
                    dict(
                        name=record.scale_prior_loss.name,
                        params=[record.scale_prior_loss.scale],
                        magnitude=record.scale_prior_loss.weight,
                    )
                ),
                [scale],
            )
            diagnostics.num_scale_prior_residuals += 1

    solver = pyceres.SolverOptions(ceres_options.create_solver_options(config, problem))
    solver.minimizer_progress_to_stdout = False
    capture = _Playback(playback_callback, playback_options, reconstruction)
    if playback_callback is not None:
        solver.update_state_every_iteration = True
        solver.callbacks = [capture]
    capture.emit("initial", -1)
    summary = pyceres.SolverSummary()
    pyceres.solve(solver, problem, summary)
    diagnostics.update(summary)
    result = BundleAdjustmentResult(
        summary.IsSolutionUsable(), {image_id: float(scale[0]) for image_id, scale in scales.items()}, diagnostics
    )
    if result.success:
        capture.emit("final", diagnostics.num_iterations - 1)
        native.publish_geometry(reconstruction, state.reconstruction, update_poses=options.refine_rig_from_world)
    return result


class _Playback(pyceres.IterationCallback):
    def __init__(self, callback, options, reconstruction):
        super().__init__()
        self.callback, self.options, self.reconstruction = (
            callback,
            options,
            reconstruction,
        )
        if callback is None:
            return
        self.image_ids = list(options.image_ids) or sorted(reconstruction.reg_image_ids())
        points = sorted(reconstruction.point3D_ids())
        self.point_ids = list(options.point3D_ids) or points[:: max(1, (len(points) + 199999) // 200000)]
        if any(not reconstruction.image(i).has_pose for i in self.image_ids):
            raise ValueError("playback image is not available in bundle adjustment")
        for point_id in self.point_ids:
            reconstruction.point3D(point_id)
        self.image_ids = np.array(self.image_ids, dtype=np.uint32)
        self.point_ids = np.array(self.point_ids, dtype=np.uint64)

    def __call__(self, summary):
        if summary.iteration % self.options.snapshot_every_n_iterations == 0:
            self.emit("iteration", summary.iteration)
        return pyceres.CallbackReturnType.SOLVER_CONTINUE

    def emit(self, phase, iteration):
        if self.callback is not None:
            centers, points_xyz = native.playback_coordinates(self.reconstruction, self.image_ids, self.point_ids)
            self.callback(
                dict(
                    phase=phase,
                    iteration=iteration,
                    image_ids=self.image_ids.copy(),
                    centers=centers,
                    point_ids=self.point_ids.copy(),
                    points_xyz=points_xyz,
                    lc_pairs=np.empty((0, 2), dtype=np.uint32),
                    lc_raw_score=np.empty(0),
                    lc_support_count=np.empty(0, dtype=np.uint64),
                )
            )
