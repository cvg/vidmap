"""Mapper entry point and complete mapping-stage sequence."""

from collections.abc import Callable
from pathlib import Path

from vidmap.mapper.focal_prior import load_focal_prior
from vidmap.mapper.inputs import MapperInputs
from vidmap.mapper.inputs.snapshot import calibration_artifact_name
from vidmap.utils.profiling import log_memory, record_timing, sync_time

from .checkpoints import remove_disabled_intermediate_reconstructions
from .inputs.database import remove_database_sidecars
from .inputs.loader import MappingProblemLoader
from .options import MapperOptions
from .playback_trace import PlaybackTraceOptions, PlaybackTraceRecorder
from .replay.cache import ReplayCache
from .stages.bundle_adjustment.adjuster import BundleAdjuster
from .stages.depth_consistency import DepthConsistencyFilter
from .stages.global_positioning.positioner import GlobalPositioner
from .stages.relative_pose.estimator import RelativePoseEstimator
from .stages.rotation_averaging import RotationAverager
from .stages.tracks import TrackBuilder
from .stages.vgc_filter import ViewGraphFilter
from .stages.view_graph_calibration import ViewGraphCalibrator


class Mapper:
    """Run global reconstruction from finalized persistent inputs."""

    def __init__(
        self,
        *,
        conf: MapperOptions,
        mapper_inputs: MapperInputs,
        sfm_outputs_dir: Path,
        persist_intermediate_reconstructions: bool = False,
    ) -> None:
        if not isinstance(conf, MapperOptions):
            raise TypeError(f"Expected MapperOptions, got {type(conf).__name__}")
        if not isinstance(mapper_inputs, MapperInputs):
            raise TypeError(f"Expected MapperInputs, got {type(mapper_inputs).__name__}")
        if not isinstance(persist_intermediate_reconstructions, bool):
            raise TypeError(
                "persist_intermediate_reconstructions must be bool, "
                f"got {type(persist_intermediate_reconstructions).__name__}"
            )
        sfm_outputs_dir = Path(sfm_outputs_dir).resolve()
        if mapper_inputs.directory == sfm_outputs_dir or mapper_inputs.directory in sfm_outputs_dir.parents:
            raise ValueError("Mapper outputs must not be inside the mapper-input directory")
        self.conf = conf
        self.mapper_inputs = mapper_inputs
        self.sfm_outputs_dir = sfm_outputs_dir
        self.persist_intermediate_reconstructions = persist_intermediate_reconstructions

    def run(
        self,
        *,
        save_playback_trace: bool = False,
        playback_trace_stride: int = 3,
        playback_trace_point_cap: int | None = None,
        overwrite_outputs: bool = False,
        on_inputs_validated: Callable[[], None] | None = None,
    ):
        boundary = self.mapper_inputs.frontend_identity()["boundary_options"]
        need_factors = self._validate_calibration_request(boundary)
        if save_playback_trace:
            PlaybackTraceRecorder.preflight(self.sfm_outputs_dir, replace=overwrite_outputs)
            playback_options = PlaybackTraceOptions(
                iteration_stride=playback_trace_stride,
                point_cap=playback_trace_point_cap,
            )
        else:
            playback_options = None
        replay = ReplayCache(self.conf.replay_cache, self.sfm_outputs_dir)
        working_database = self.sfm_outputs_dir / "database_complete.db"
        try:
            loader = MappingProblemLoader(
                options=self.conf.setup,
                inputs=self.mapper_inputs,
                sfm_outputs_dir=self.sfm_outputs_dir,
                replay=replay,
            )
            prior = {}

            def admit_calibration(state):
                nonlocal prior
                if need_factors:
                    prior = self._prepare_calibration(state, boundary)

            mapping_stage_inputs = loader.load(on_database_loaded=admit_calibration)

            if on_inputs_validated is not None:
                on_inputs_validated()
            remove_disabled_intermediate_reconstructions(
                self.sfm_outputs_dir,
                persist=self.persist_intermediate_reconstructions,
                post_point_refinement=self.conf.ba.post_annealing_point_refinement,
            )
            playback_trace = (
                PlaybackTraceRecorder.start(
                    self.sfm_outputs_dir,
                    working_database,
                    options=playback_options,
                    replace=overwrite_outputs,
                )
                if save_playback_trace
                else None
            )
            try:
                reconstruction = self._solve(mapping_stage_inputs, replay, playback_trace, prior)
                if playback_trace is not None:
                    playback_trace.finish(reconstruction)
            except BaseException:
                if playback_trace is not None:
                    playback_trace.abort()
                raise
            return reconstruction
        finally:
            remove_database_sidecars(working_database)
            working_database.unlink(missing_ok=True)

    def _validate_calibration_request(self, boundary) -> bool:
        """Check configuration before loading or publishing and resolve whether factors are needed."""
        calibration = self.conf.calibration
        predictor = boundary["estimator"]
        if calibration.vgc_focal_prior and (predictor != "da3" or not calibration.vgc_enabled):
            raise ValueError("VGC focal prior requires enabled DA3 VGC")
        need_factors = calibration.vgc_focal_prior or (
            calibration.optimize_intrinsics and self.conf.ba.focal_prior.enabled
        )
        if need_factors and predictor == "none":
            raise ValueError("Focal regularization requires original predictor factors")
        return need_factors

    def _prepare_calibration(self, solve_state, boundary):
        """Construct original-target factors from the validated mapper inputs."""
        calibration = self.conf.calibration
        predictor = boundary["estimator"]
        cameras = solve_state.reconstruction.cameras
        if any(c.model.name not in {"PINHOLE", "SIMPLE_PINHOLE"} for c in cameras.values()):
            raise ValueError("Calibration requires pinhole cameras")
        shared = boundary["inference"] == "selected_batch"
        path = self.mapper_inputs.directory / calibration_artifact_name(predictor, boundary["inference"])
        log_focal_stddev = calibration.da3_log_focal_stddev if predictor == "da3" else None
        return load_focal_prior(path, solve_state, log_focal_stddev=log_focal_stddev, shared=shared)

    def _solve(self, mapping_stage_inputs, replay, playback_trace, prior):
        solve_start_time = sync_time()
        solve_state = mapping_stage_inputs.solve_state

        calibration = self.conf.calibration

        # Remove view-graph observations that must not influence calibration or
        # any of the downstream global solves.
        view_graph_filter = ViewGraphFilter(
            solve_state=solve_state,
            options=self.conf.vgc.filter,
            calibration_enabled=calibration.vgc_enabled,
            initial_exclusion_ids=mapping_stage_inputs.vgc_exclusion_ids,
        )
        vgc_exclusion_ids = view_graph_filter.filter()

        # Calibrate the surviving view graph before relative rotations and
        # bearing vectors are estimated from it.
        vgc_start_time = sync_time()
        view_graph_calibrator = ViewGraphCalibrator(
            solve_state=solve_state,
            options=self.conf.vgc.calibration,
            enabled=calibration.vgc_enabled,
            consecutive_pair_ids=mapping_stage_inputs.consecutive_pair_ids,
            exclusion_ids=vgc_exclusion_ids,
            focal_prior=prior if calibration.vgc_focal_prior else None,
        )
        view_graph_calibrator.calibrate()

        # Bearings are shared by relative-pose estimation and the later global
        # positioning stage, so construct them once on the solve state.
        GlobalPositioner.prepare_bearings(solve_state, self.conf.gp.common.bearing_kp_stddev)

        # Estimate pair geometry first; rotation averaging consumes both its
        # filtered video edges and resolved inlier thresholds.
        relative_pose_estimator = RelativePoseEstimator(
            solve_state=solve_state,
            options=self.conf.mdrp,
            inlier_threshold_options=self.conf.inlier_thresholds,
            consecutive_pair_ids=mapping_stage_inputs.consecutive_pair_ids,
            replay=replay,
        )
        relative_pose = relative_pose_estimator.estimate()

        rotation_averager = RotationAverager(
            solve_state=solve_state,
            options=self.conf.ra,
            consecutive_pair_ids=mapping_stage_inputs.consecutive_pair_ids,
            filtered_consecutive_pair_ids=relative_pose.filtered_consecutive_pairs,
            replay=replay,
        )
        rotation_averager.average()

        # Mark inconsistent boundary depth before track construction so every
        # subsequent position solve sees the same filtered observations.
        depth_consistency_filter = DepthConsistencyFilter(
            solve_state=solve_state,
            options=self.conf.depth_consistency,
            consecutive_pair_ids=mapping_stage_inputs.consecutive_pair_ids,
            sequence_id_to_index=mapping_stage_inputs.sequence_id_to_index,
        )
        boundary_depth_outliers_marked = depth_consistency_filter.filter()

        track_builder = TrackBuilder(
            solve_state=solve_state,
            options=self.conf.tracks,
            boundary_depth_outliers_marked=boundary_depth_outliers_marked,
            replay=replay,
        )
        tracks = track_builder.build()

        record_timing("pre_global_positioning", sync_time() - vgc_start_time)
        log_memory("pre_global_positioning")

        # Solve positions from the fixed rotations and tracks, then refine the
        # complete reconstruction with bundle adjustment.
        global_positioner = GlobalPositioner(
            solve_state=solve_state,
            tracks=tracks,
            consecutive_pair_ids=mapping_stage_inputs.consecutive_pair_ids,
            sequence_id_to_index=mapping_stage_inputs.sequence_id_to_index,
            inlier_thresholds=relative_pose.inlier_thresholds,
            boundary_depth_outliers_marked=boundary_depth_outliers_marked,
            options=self.conf.gp,
            output_dir=self.sfm_outputs_dir,
            replay=replay,
            persist_intermediate_reconstructions=self.persist_intermediate_reconstructions,
            playback_trace=playback_trace,
        )
        global_positioner.position()

        bundle_adjuster = BundleAdjuster(
            solve_state=solve_state,
            options=self.conf.ba,
            depth_stddev_multiplier=self.conf.mdrp.depth_stddev_multiplier,
            optimize_intrinsics=calibration.optimize_intrinsics,
            focal_prior=prior,
            output_dir=self.sfm_outputs_dir,
            replay=replay,
            persist_intermediate_reconstructions=self.persist_intermediate_reconstructions,
            playback_trace=playback_trace,
        )
        bundle_adjuster.adjust()
        if self.mapper_inputs.full_depth_maps_path is not None:
            from vidmap.depth_artifacts import write_depth_scales

            write_depth_scales(
                self.sfm_outputs_dir,
                bundle_adjuster.final_depth_map_scales,
            )

        record_timing("solve_total", sync_time() - solve_start_time)
        log_memory("solve_total")
        return solve_state.reconstruction
