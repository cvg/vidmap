"""Unified normal and annealed bundle adjustment."""

from __future__ import annotations

import logging
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pycolmap

import vidmap.utils.multiview_geometry as multiview_geometry
from vidmap.mapper.checkpoints import reconstruction_checkpoint_directory
from vidmap.mapper.native.extension import native
from vidmap.mapper.native.losses import native_loss_type
from vidmap.mapper.native.state import SolveState
from vidmap.mapper.options.refinement import BAOptions, BATriangulationOptions
from vidmap.mapper.playback_trace import PlaybackTraceRecorder
from vidmap.mapper.replay.cache import ReplayCache
from vidmap.utils.multiview_geometry import point_has_positive_depth
from vidmap.utils.profiling import log_memory, record_timing, sync_time

from .native_options import (
    build_bundle_adjustment_options,
    make_depth_constraint_record,
    make_depth_scale_record,
    make_intrinsics_prior_record,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True, kw_only=True)
class BASolvePolicy:
    """Discrete scientific choices for one bundle-adjustment solve."""

    refinement: bool
    fix_scale: bool
    gross_outliers: bool
    fix_rotations: bool
    regularize_scale: bool
    fix_all_poses: bool = False
    fix_intrinsics: bool = False


@dataclass(kw_only=True)
class BundleAdjuster:
    """Run normal, annealed, and final BA on one live reconstruction."""

    solve_state: SolveState
    options: BAOptions
    depth_stddev_multiplier: float
    focal_uncertainty: float | None
    output_dir: Path
    replay: ReplayCache
    persist_intermediate_reconstructions: bool = False
    playback_trace: PlaybackTraceRecorder | None = None
    observation_graph: pycolmap.CorrespondenceGraph = field(init=False)
    observations: pycolmap.ObservationManager = field(init=False)
    triangulator: pycolmap.IncrementalTriangulator = field(init=False)
    triangulator_options: pycolmap.IncrementalTriangulatorOptions = field(init=False)
    shift_scale: dict = field(init=False)
    optimize_intrinsics: bool = field(init=False)
    truncation_multiplier: float = field(init=False)

    @property
    def reconstruction(self) -> pycolmap.Reconstruction:
        return self.solve_state.reconstruction

    @property
    def final_depth_map_scales(self) -> dict[int, float]:
        """Return the physical raw-depth multipliers used by final BA."""
        return {
            int(image_id): float(np.exp(np.asarray(values, dtype=np.float64)[1]))
            for image_id, values in self.shift_scale.items()
        }

    @property
    def annealing_intrinsics_prior_std_factor(self) -> float:
        return self.options.resolved_annealing_prior_std_factor

    def adjust(self) -> None:
        self.prepare_workspace()
        ba_start_time = sync_time()
        self.replay.write_ba_start_summary(self.reconstruction, self.solve_state)

        self.run_normal()
        if self.persist_intermediate_reconstructions:
            checkpoint_dir = self.output_dir / "rec-ba"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            self.reconstruction.write(checkpoint_dir)
            logger.info("Saved pre-annealing BA checkpoint: %s", checkpoint_dir)
        self.run_annealed()
        self.run_final()
        if self.options.post_annealing_point_refinement:
            with reconstruction_checkpoint_directory(
                self.output_dir,
                "rec-pre-point-refinement",
                persist=self.persist_intermediate_reconstructions,
            ) as checkpoint_dir:
                self.reconstruction.write(checkpoint_dir)
                logger.info("Saved pre-point-refinement checkpoint: %s", checkpoint_dir)
                if not self.run_post_annealing_point_refinement():
                    self.solve_state.import_checkpoint(pycolmap.Reconstruction(checkpoint_dir))
                    logger.warning(
                        "Post-annealing point-only refinement failed; restored the pre-refinement checkpoint"
                    )

        record_timing("bundle_adjustment", sync_time() - ba_start_time)
        log_memory("bundle_adjustment")

    def prepare_workspace(self) -> None:
        target_multiplier = self.options.depth.target_stddev_multiplier
        for image in self.solve_state.image_records().values():
            image.depth_stddevs = np.asarray(image.depth_stddevs) / self.depth_stddev_multiplier * target_multiplier
            self.solve_state.update_image(image)

        self.observation_graph = self.build_observation_graph()
        self.observations = pycolmap.ObservationManager(self.reconstruction, self.observation_graph)
        self.triangulator = pycolmap.IncrementalTriangulator(
            self.observation_graph,
            self.reconstruction,
            self.observations,
        )
        self.triangulator_options = self.build_triangulator_options(self.options.triangulation)
        self.optimize_intrinsics = bool(self.focal_uncertainty is not None)

    def build_observation_graph(self) -> pycolmap.CorrespondenceGraph:
        graph = pycolmap.CorrespondenceGraph()
        for image_id, image in self.reconstruction.images.items():
            graph.add_image(image_id, len(image.points2D))

        registered_image_ids = set(self.reconstruction.images.keys())
        for image_pair in self.solve_state.pair_records().values():
            image_id1, image_id2 = image_pair.image_id1, image_pair.image_id2
            if len(image_pair.inlier_indices) == 0:
                continue
            if image_id1 not in registered_image_ids or image_id2 not in registered_image_ids:
                continue
            geometry = pycolmap.TwoViewGeometry()
            geometry.inlier_matches = image_pair.all_matches[image_pair.inlier_indices].astype(np.uint32)
            geometry.config = pycolmap.TwoViewGeometryConfiguration.CALIBRATED
            graph.add_two_view_geometry(image_id1, image_id2, geometry)

        graph.finalize()
        return graph

    def build_triangulator_options(
        self,
        options: BATriangulationOptions,
    ) -> pycolmap.IncrementalTriangulatorOptions:
        return pycolmap.IncrementalTriangulatorOptions(
            {
                "min_angle": options.min_angle,
                "ignore_two_view_tracks": options.ignore_two_view_tracks,
                "create_max_angle_error": options.create_max_angle_error,
                "re_max_angle_error": options.re_max_angle_error,
                "re_max_trials": options.re_max_trials,
                "re_min_ratio": options.re_min_ratio,
            }
        )

    def run_normal(self) -> None:
        logger.info("Bundle adjustment start")
        iteration = 0
        while iteration < self.options.normal.iterations:
            if iteration == 0:
                self.shift_scale = {image_id: np.array([0.0, 0.0]) for image_id in self.reconstruction.images.keys()}
                self.shift_scale = self.reset_and_retriangulate(
                    max_error_multiplier=self.options.first_iteration_error_multiplier * self.options.multiply_errors,
                )
                self.solve_problem(
                    policy=BASolvePolicy(
                        refinement=False,
                        fix_scale=False,
                        gross_outliers=False,
                        fix_rotations=True,
                        regularize_scale=True,
                    ),
                    param_multiplier=self.options.depth.param_multiplier,
                    intrinsics_prior_std_factor=self.options.intrinsics.prior_std_factor,
                )

            self.shift_scale = self.reset_and_retriangulate(
                max_error_multiplier=(
                    self.options.first_iteration_error_multiplier * self.options.multiply_errors
                    if iteration == 0
                    else self.options.multiply_errors
                ),
            )
            solved = self.solve_problem(
                policy=BASolvePolicy(
                    refinement=False,
                    fix_scale=False,
                    gross_outliers=False,
                    fix_rotations=False,
                    regularize_scale=True,
                ),
                param_multiplier=self.options.depth.param_multiplier,
                intrinsics_prior_std_factor=self.options.intrinsics.prior_std_factor,
            )
            if solved:
                logger.debug(
                    "Global bundle adjustment iteration %s / %s, stage 2 finished",
                    iteration + 1,
                    self.options.normal.iterations,
                )
            logger.debug("Filtering tracks by reprojection")
            status = True
            filtered_num = 0
            total_observations = np.sum(
                [self.observations.num_observations(image_id) for image_id in self.reconstruction.images.keys()]
            )
            while status and iteration < self.options.normal.iterations:
                scaling = max(self.options.normal.iterations - iteration, 1)
                filtered_num += self.observations.filter_all_points3D(
                    scaling * self.options.normal.observation_filter_multiplier,
                    0,
                )
                logger.debug("Filtered %s obs of %s total obs", filtered_num, total_observations)
                if filtered_num > (
                    self.options.normal.convergence_filtered_point_ratio * len(self.reconstruction.points3D)
                ):
                    logger.debug("Filtered more than 0.1%% points; running BA again")
                    status = False
                else:
                    iteration += 1

            if status:
                logger.debug("Fewer than 0.1%% tracks were filtered; stopping normal BA iterations")
                break
            iteration += 1

    def run_annealed(self) -> None:
        logger.info("Annealed bundle adjustment start")
        refinement = self.options.annealing
        for iteration in range(self.options.annealing.iterations):
            self.shift_scale = self.estimate_depth_scales()
            logger.debug(
                "[Annealed BA iter %s] Computed shift/scale for %s images",
                iteration,
                len(self.shift_scale),
            )
            self.shift_scale = self.reset_and_retriangulate(
                max_error_multiplier=(
                    self.options.first_iteration_error_multiplier * self.options.multiply_errors
                    if iteration == 0
                    else self.options.multiply_errors
                ),
            )
            self.truncation_multiplier = self.estimate_truncation_multiplier()
            adapted_multiplier = refinement.depth_param_multiplier * self.truncation_multiplier
            solved = self.solve_problem(
                policy=BASolvePolicy(
                    refinement=True,
                    fix_scale=True,
                    gross_outliers=False,
                    fix_rotations=refinement.fix_rotations,
                    regularize_scale=False,
                ),
                param_multiplier=adapted_multiplier,
                intrinsics_prior_std_factor=self.annealing_intrinsics_prior_std_factor,
            )
            if solved:
                logger.debug(
                    "[Annealed BA iter %s] BA finished (fix_scale=True, param_mult=%.4f)",
                    iteration,
                    adapted_multiplier,
                )

            filter_max_reprojection = self.options.multiply_errors * refinement.observation_filter_multiplier
            observations_before = sum(
                self.observations.num_observations(image_id) for image_id in self.reconstruction.images.keys()
            )
            filtered_num = self.observations.filter_all_points3D(filter_max_reprojection, 0)
            logger.debug(
                "[Annealed BA iter %s] Filtered %s / %s observations",
                iteration,
                filtered_num,
                observations_before,
            )
            if filtered_num < refinement.convergence_filtered_point_ratio * len(self.reconstruction.points3D):
                logger.debug("[Annealed BA iter %s] Converged (< 0.1%% filtered)", iteration)
                break

    def run_final(self) -> None:
        refinement = self.options.annealing
        logger.info("[Annealed BA] Final refinement with aggressive outlier downweighting")
        self.shift_scale = self.estimate_depth_scales()
        self.shift_scale = self.reset_and_retriangulate(max_error_multiplier=self.options.multiply_errors)
        solved = self.solve_problem(
            policy=BASolvePolicy(
                refinement=True,
                fix_scale=True,
                gross_outliers=True,
                fix_rotations=refinement.fix_rotations,
                regularize_scale=False,
            ),
            param_multiplier=refinement.final_depth_param_multiplier * self.truncation_multiplier,
            intrinsics_prior_std_factor=self.annealing_intrinsics_prior_std_factor,
        )
        if solved:
            logger.info(
                "[Annealed BA] Final refinement done (param_mult=%s)",
                refinement.final_depth_param_multiplier,
            )

    def run_post_annealing_point_refinement(self) -> bool:
        """Retriangulate and refine only points under the normal BA model."""
        logger.info("Post-annealing point-only refinement start")
        self.shift_scale = self.reset_and_retriangulate(
            max_error_multiplier=self.options.multiply_errors,
        )
        solved = self.solve_problem(
            policy=BASolvePolicy(
                refinement=False,
                fix_scale=True,
                gross_outliers=False,
                fix_rotations=True,
                regularize_scale=False,
                fix_all_poses=True,
                fix_intrinsics=True,
            ),
            param_multiplier=self.options.depth.param_multiplier,
            intrinsics_prior_std_factor=self.options.intrinsics.prior_std_factor,
        )
        if solved:
            filtered_num = self.observations.filter_all_points3D(
                self.options.normal.observation_filter_multiplier,
                0,
            )
            logger.info("Post-annealing point-only refinement done")
            logger.info("Post-annealing point-only refinement filtered %d observations", filtered_num)
        return solved

    @staticmethod
    def image_point3D_ids(image) -> np.ndarray:
        return np.asarray([point.point3D_id for point in image.points2D], dtype=np.uint64)

    def points3D_xyz(self, point3D_ids) -> np.ndarray:
        return np.asarray(
            [self.reconstruction.point3D(int(point3D_id)).xyz for point3D_id in point3D_ids],
            dtype=np.float64,
        ).reshape((-1, 3))

    def small_triangulation_angle_mask(self, point3D_ids, min_angle: float) -> np.ndarray:
        native_ids = set(int(value) for value in self.solve_state.native_problem.point3D_ids)
        unique_point3D_ids = dict.fromkeys(int(value) for value in point3D_ids)
        records = [
            self.solve_state.native_problem.track(point3D_id)
            for point3D_id in unique_point3D_ids
            if point3D_id in native_ids
        ]
        result = native.filter_tracks_by_triangulation_angle(
            self.solve_state.native_problem,
            records,
            min_angle,
        )
        small_ids = {int(track.point3D_id) for track in result.tracks if len(track.observations) == 0}
        return np.asarray(
            [int(point3D_id) in small_ids for point3D_id in point3D_ids],
            dtype=bool,
        )

    def estimate_truncation_multiplier(self) -> float:
        whitened_residuals = []
        for image_id in self.reconstruction.reg_image_ids():
            image = self.reconstruction.images[image_id]
            point2D_indices = np.array(image.get_observation_point2D_idxs())
            if point2D_indices.size == 0:
                continue
            solve_image = self.solve_state.image(image_id)
            valid = np.asarray(solve_image.depth_validity, dtype=bool)[point2D_indices]
            valid_indices = point2D_indices[valid]
            if valid_indices.size == 0:
                continue
            depth_priors = np.asarray(solve_image.depth_values)[valid_indices]
            stddevs = np.asarray(solve_image.depth_stddevs)[valid_indices]
            point3D_ids = self.image_point3D_ids(image)[valid_indices].astype(np.int64)
            projected_depths = multiview_geometry.project_point_depths(self.reconstruction, image_id, point3D_ids)
            mask = (depth_priors > 0) & (projected_depths > 0)
            if mask.sum() == 0:
                continue
            log_stddevs = np.clip(stddevs[mask] / depth_priors[mask], 1e-6, None)
            log_distances = np.log(depth_priors[mask]) - np.log(projected_depths[mask])
            whitened_residuals.append(log_distances / log_stddevs)

        if not whitened_residuals:
            return 1.0
        residuals = np.concatenate(whitened_residuals)
        median = np.median(residuals)
        sigma = 1.4826 * np.median(np.abs(residuals - median))
        multiplier = max(sigma, self.options.annealing.mad_sigma_floor)
        logger.debug("MAD truncation_multiplier = %.3f", multiplier)
        return multiplier

    def solve_problem(
        self,
        *,
        policy: BASolvePolicy,
        param_multiplier: float,
        intrinsics_prior_std_factor: float,
    ) -> bool:
        depth_options = self.options.depth
        refinement_options = self.options.annealing
        keypoint_stddev = refinement_options.kp_std if policy.refinement else self.options.normal.kp_stddev
        reprojection_loss = (
            refinement_options.reproj_loss_name if policy.refinement else self.options.normal.reproj_loss_name
        )
        robust_risky_depth = (
            refinement_options.robust_risky_depth if policy.refinement else self.options.normal.robust_risky_depth
        )
        depth_magnitude_multiplier = (
            refinement_options.depth_magnitude_multiplier if policy.refinement else depth_options.magnitude_multiplier
        )
        depth_cutoff = (
            refinement_options.depth_cutoff_cauchy_scales if policy.refinement else depth_options.cutoff_cauchy_scales
        )

        self.solve_state.import_scene()
        depth_loss_type = native_loss_type(depth_options.reg_loss_name)
        optimized_image_ids = list(self.reconstruction.reg_image_ids())
        track_lengths = {
            point3D_id: self.reconstruction.points3D[point3D_id].track.length()
            for point3D_id in self.reconstruction.points3D.keys()
        }
        camera_ids = [self.reconstruction.images[image_id].camera_id for image_id in optimized_image_ids]
        variable_point3D_ids = [
            int(point3D_id)
            for point3D_id, track_length in track_lengths.items()
            if track_length < self.options.variable_point_track_length_threshold
        ]
        options = build_bundle_adjustment_options(
            image_order=optimized_image_ids,
            camera_ids=camera_ids,
            variable_point3D_ids=variable_point3D_ids,
            optimize_intrinsics=self.optimize_intrinsics and not policy.fix_intrinsics,
            refine_principal_point=self.options.intrinsics.refine_principal_point,
            fix_rotations=policy.fix_rotations,
            fix_all_poses=policy.fix_all_poses,
            reprojection_loss=reprojection_loss,
            reprojection_scale=self.options.reproj_loss_scale * keypoint_stddev,
            reprojection_weight=1 / keypoint_stddev**2,
            num_threads=self.options.num_threads,
        )

        intrinsics_priors = []
        if self.optimize_intrinsics and not policy.fix_intrinsics and self.options.intrinsics.use_prior:
            for camera_id in camera_ids:
                camera = self.reconstruction.cameras[camera_id]
                priors = camera.params.copy()
                effective_focal_uncertainty = (
                    self.focal_uncertainty if self.focal_uncertainty is not None else 0.05 * priors[0]
                )
                focal_stddev = effective_focal_uncertainty * intrinsics_prior_std_factor
                if camera.model == pycolmap.CameraModelId.SIMPLE_PINHOLE:
                    stddevs = np.array([focal_stddev, 5, 5])
                else:
                    stddevs = np.array([focal_stddev, focal_stddev, 5, 5])
                intrinsics_priors.append(make_intrinsics_prior_record(camera_id, priors, stddevs))

        depth_constraints = []
        depth_scales = []
        images_without_residuals = []
        for image_id in optimized_image_ids:
            image = self.reconstruction.images[image_id]
            point2D_indices = np.array(image.get_observation_point2D_idxs())
            if point2D_indices.size == 0:
                images_without_residuals.append(image_id)
                continue
            solve_image = self.solve_state.image(image_id)
            valid = np.asarray(solve_image.depth_validity, dtype=bool)[point2D_indices]
            depths = np.asarray(solve_image.depth_values)[point2D_indices]
            point2D_indices = point2D_indices[valid]
            depths = depths[valid]
            point3D_ids = self.image_point3D_ids(image)[point2D_indices].astype(np.int64)
            projected_depths = multiview_geometry.project_point_depths(self.reconstruction, image_id, point3D_ids)
            risky_track_lengths = (
                np.array([track_lengths[point3D_id] for point3D_id in point3D_ids])
                < depth_options.risky_track_length_threshold
            )
            risky_angles = self.small_triangulation_angle_mask(
                point3D_ids,
                depth_options.risky_triangulation_angle_deg,
            )
            risky_mask = risky_track_lengths | risky_angles

            mask = depths > 0
            variances = np.asarray(solve_image.depth_stddevs)[point2D_indices] ** 2
            if policy.gross_outliers:
                whitened = (
                    np.abs(np.log(depths).clip(1e-6, None) - np.log(projected_depths).clip(1e-6, None))
                    / variances**0.5
                )
                mask *= whitened < 2
            if depth_cutoff > 0 and param_multiplier > 0:
                log_residual = np.abs(np.log(depths.clip(1e-6, None)) - np.log(projected_depths.clip(1e-6, None)))
                cauchy_scales = param_multiplier * variances**0.5 / depths.clip(1e-6, None)
                mask *= log_residual < depth_cutoff * cauchy_scales
            if np.sum(mask) == 0:
                logger.debug("No valid points for depth regularization in image %d", image_id)
                continue

            depths = depths[mask]
            variances = variances[mask]
            inverse_uncertainty = 1 / variances.clip(1e-6, None)
            point3D_ids = point3D_ids[mask]
            risky_mask = risky_mask[mask]
            if param_multiplier == 0:
                continue
            params = param_multiplier * variances**0.5 / depths
            magnitudes = depth_magnitude_multiplier * depths**2 * inverse_uncertainty
            loss_types = (
                [depth_loss_type for _ in risky_mask]
                if robust_risky_depth
                else [native_loss_type("trivial") if risky else depth_loss_type for risky in risky_mask]
            )
            for point3D_id, depth, magnitude, param, constraint_loss_type in zip(
                point3D_ids,
                depths,
                magnitudes,
                params,
                loss_types,
            ):
                depth_constraints.append(
                    make_depth_constraint_record(
                        image_id,
                        point3D_id,
                        depth,
                        constraint_loss_type,
                        param,
                        magnitude,
                    )
                )

            depth_scales.append(
                make_depth_scale_record(
                    image_id=image_id,
                    shift_scale=self.shift_scale[image_id],
                    fix_scale=policy.fix_scale,
                    use_scale_prior=policy.regularize_scale,
                    scale_prior_stddev=depth_options.scale_std,
                    scale_prior_loss=depth_options.scale_reg_loss_name,
                    scale_prior_weight=np.sum(mask),
                )
            )

        logger.debug("Solving bundle-adjustment problem")
        playback_sink = None
        if self.playback_trace is not None:
            playback_sink = self.playback_trace.attach_bundle_adjustment(options, self.reconstruction)
        result = native.run_bundle_adjustment(
            options,
            depth_constraints,
            depth_scales,
            intrinsics_priors,
            self.solve_state.native_problem,
        )
        if not result.success:
            self.solve_state.import_scene()
            diagnostics = result.diagnostics
            if playback_sink is not None:
                playback_sink.finish_recovered(
                    self.reconstruction,
                    final_iteration=diagnostics.num_iterations - 1,
                )
            logger.warning(
                f"BA solve failed (termination_type={diagnostics.termination_type}, "
                f"iterations={diagnostics.num_iterations}, initial_cost={diagnostics.initial_cost:.8g}, "
                f"final_cost={diagnostics.final_cost:.8g}); restored the last valid reconstruction "
                "and continuing"
            )
            return False
        for image_id, values in result.depth_shift_scales.items():
            self.shift_scale[int(image_id)] = np.asarray(values).copy()
        if not policy.fix_all_poses:
            for image_id in images_without_residuals:
                image_record = self.solve_state.image(image_id)
                image_record.pose.has_pose = False
                self.solve_state.update_image(image_record)
        self.solve_state.export_cameras()
        self.solve_state.export_poses()
        self.solve_state.export_track_values()
        logger.debug(
            "BA solved: residuals=%d, initial_cost=%.8g, final_cost=%.8g",
            result.diagnostics.num_residual_blocks,
            result.diagnostics.initial_cost,
            result.diagnostics.final_cost,
        )
        return True

    def estimate_depth_scales(self) -> dict:
        shift_scale = {}
        for image_id in self.reconstruction.reg_image_ids():
            image = self.reconstruction.images[image_id]
            point2D_indices = np.array(image.get_observation_point2D_idxs())
            if point2D_indices.size == 0:
                continue
            solve_image = self.solve_state.image(image_id)
            valid_prior = np.asarray(solve_image.depth_validity, dtype=bool)[point2D_indices]
            if not np.any(valid_prior):
                continue
            point2D_indices = point2D_indices[valid_prior]
            observed_depths = np.asarray(solve_image.depth_values)[point2D_indices]
            point3D_ids = self.image_point3D_ids(image)[point2D_indices].astype(np.int64)
            mask = observed_depths > 0
            if mask.sum() == 0:
                logger.debug("No valid points for shift/scale estimation in image %d", image_id)
                continue
            points3D = self.points3D_xyz(point3D_ids)
            projected_depths = (image.cam_from_world() * points3D)[:, -1][mask]
            observed_depths = observed_depths[mask]
            proposed = np.median(np.log(projected_depths.clip(1e-6, None) / observed_depths.clip(1e-6, None)))
            shift_scale[image_id] = np.array([0.0, proposed])
        return shift_scale

    def reset_and_retriangulate(self, *, max_error_multiplier: float) -> dict:
        triangulation = self.options.triangulation
        filtered_num = self.observations.filter_all_points3D(
            max_reproj_error=self.options.multiply_errors * self.options.retriangulation_reproj_multiplier,
            min_tri_angle=triangulation.min_angle,
        )
        if max_error_multiplier > 1:
            options = deepcopy(self.triangulator_options)
            options.create_max_angle_error *= max_error_multiplier
            options.continue_max_angle_error *= max_error_multiplier
            options.merge_max_reproj_error *= max_error_multiplier
            options.complete_max_reproj_error *= max_error_multiplier
            options.max_transitivity = triangulation.relaxed_max_transitivity
        else:
            options = self.triangulator_options

        retriangulated = self.retriangulate(options)
        completed = self.triangulator.complete_all_tracks(options)
        merged = self.triangulator.merge_all_tracks(options)
        logger.info(
            "Filtered %d points, retriangulated %d, completed %d, merged %d",
            filtered_num,
            retriangulated,
            completed,
            merged,
        )
        return self.estimate_depth_scales()

    def retriangulate(self, options: pycolmap.IncrementalTriangulatorOptions):
        output = self.triangulator.retriangulate(options)
        self.solve_state.import_scene()
        point3D_ids = np.array(list(self.reconstruction.points3D.keys()))
        risky_mask = self.small_triangulation_angle_mask(
            point3D_ids,
            self.options.triangulation.min_angle,
        )
        count = 0
        for point3D_id in point3D_ids[risky_mask]:
            point3D = self.reconstruction.points3D[point3D_id]
            image_ids = [element.image_id for element in point3D.track.elements]
            point2D_indices = [element.point2D_idx for element in point3D.track.elements]
            cameras_from_world = [self.reconstruction.images[image_id].cam_from_world() for image_id in image_ids]
            self.observations.delete_point3D(point3D_id)
            for lift_index, image_id in enumerate(image_ids):
                lift_image = self.solve_state.image(image_id)
                point2D_index = point2D_indices[lift_index]
                if not lift_image.depth_validity[point2D_index]:
                    continue
                xy = lift_image.keypoints[point2D_index]
                depth = lift_image.depth_values[point2D_index] * np.exp(self.shift_scale[image_id][1])
                camera = self.reconstruction.cameras[lift_image.camera_id]
                xyz = self.reconstruction.image(image_id).cam_from_world().inverse() * (
                    np.concatenate([camera.cam_from_img(xy[None]), np.ones((1, 1))], -1) * depth
                )
                track = pycolmap.Track()
                for (
                    candidate_image_id,
                    candidate_point2D_index,
                    camera_from_world,
                ) in zip(
                    image_ids,
                    point2D_indices,
                    cameras_from_world,
                ):
                    if point_has_positive_depth(camera_from_world.matrix(), xyz):
                        track.add_element(candidate_image_id, candidate_point2D_index)
                self.observations.add_point3D(xyz[0], track)
                count += 1
                break
        logger.debug("Lifted %d points", count)
        self.solve_state.import_scene()
        return output
