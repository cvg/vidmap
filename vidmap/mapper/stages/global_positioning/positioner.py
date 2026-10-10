"""Global positioning and BA-boundary reconstruction preparation."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path

import numpy as np
import pycolmap
from vidmap_native import global_positioning as gp_costs

from vidmap.mapper.checkpoints import reconstruction_checkpoint_directory
from vidmap.mapper.location_priors import LocationPriorSet
from vidmap.mapper.native.extension import native
from vidmap.mapper.native.state import SolveState
from vidmap.mapper.options.positioning import GPOptions
from vidmap.mapper.options.view_graph import InlierThresholdOptions
from vidmap.mapper.playback_trace import PlaybackTraceRecorder, SolverPlaybackOptions
from vidmap.mapper.replay.cache import ReplayCache
from vidmap.mapper.replay.evidence.canonical import snapshot_images, snapshot_tracks
from vidmap.mapper.replay.evidence.stages import (
    capture_gp_initial_state,
    gp_initial_state_summary,
    gp_input_rotations,
    gp_input_summary,
    gp_output_summary,
)
from vidmap.mapper.stages.global_positioning.problem import (
    TemporalAccelerationPrior,
    prepare_observations,
    run_global_positioning,
)
from vidmap.utils.profiling import log_memory, record_timing, sync_time

logger = logging.getLogger(__name__)
_TIMESTAMP = re.compile(r"^(\d+(?:\.\d+)?)(?:-|$)")


def _image_timestamps_seconds(names: list[str]) -> tuple[Decimal, ...]:
    """Parse one ordered canonical video timeline into exact elapsed seconds."""
    parsed = []
    for name in names:
        match = _TIMESTAMP.match(Path(name).stem)
        if match is None:
            raise ValueError(f"temporal acceleration requires image names containing numeric timestamps, got {name!r}")
        token = match.group(1)
        if "." not in token:
            raise ValueError(
                "temporal acceleration requires decimal-second image timestamps; "
                "integer timestamp units are ambiguous after keyframe selection"
            )
        timestamp = Decimal(token)
        if not timestamp.is_finite():
            raise ValueError(f"temporal acceleration timestamp is not finite: {name!r}")
        parsed.append(timestamp)
    deltas = [current - previous for previous, current in zip(parsed, parsed[1:])]
    if any(delta <= 0 for delta in deltas):
        raise ValueError("temporal acceleration requires strictly increasing image timestamps")

    origin = parsed[0] if parsed else Decimal(0)
    return tuple(timestamp - origin for timestamp in parsed)


def _build_temporal_acceleration_prior_specs(
    *,
    solve_state: SolveState,
    reconstruction,
    consecutive_pair_ids,
    sequence_id_to_index,
    coordinate: str = "timestamp",
) -> list:
    """Build smoothness priors across valid consecutive tracking edges."""
    ordered_image_ids = [image_id for image_id, _ in sorted(sequence_id_to_index.items(), key=lambda item: item[1])]
    if coordinate == "timestamp":
        timeline = _image_timestamps_seconds([reconstruction.image(image_id).name for image_id in ordered_image_ids])
        coordinates = dict(zip(ordered_image_ids, timeline, strict=True))
    elif coordinate == "keyframe_index":
        coordinates = {image_id: index for index, image_id in enumerate(ordered_image_ids)}
    else:
        raise ValueError(f"Unsupported temporal-acceleration coordinate: {coordinate!r}")
    registered_image_ids = {image_id for image_id in ordered_image_ids if reconstruction.image(image_id).has_pose}
    valid_edges = set()
    for pair_id in consecutive_pair_ids:
        image_id1, image_id2 = pycolmap.pair_id_to_image_pair(pair_id)
        if not solve_state.pose_graph.has_edge(image_id1, image_id2) or not solve_state.pose_graph.is_valid(pair_id):
            continue
        index1 = sequence_id_to_index[image_id1]
        index2 = sequence_id_to_index[image_id2]
        if abs(index1 - index2) == 1:
            valid_edges.add(frozenset((image_id1, image_id2)))

    observation_counts = gp_costs.track_observation_counts(reconstruction, solve_state.sidecars)
    for image_id in ordered_image_ids:
        if image_id not in observation_counts:
            observation_counts[image_id] = 0
    priors = []
    for prev_image_id, image_id, next_image_id in zip(
        ordered_image_ids,
        ordered_image_ids[1:],
        ordered_image_ids[2:],
    ):
        triplet = (prev_image_id, image_id, next_image_id)
        if any(candidate not in registered_image_ids for candidate in triplet):
            continue
        if frozenset((prev_image_id, image_id)) not in valid_edges:
            continue
        if frozenset((image_id, next_image_id)) not in valid_edges:
            continue
        mean_observation_count = sum(observation_counts[candidate] for candidate in triplet) / 3.0
        if mean_observation_count <= 0.0:
            continue
        dt_prev = float(coordinates[image_id] - coordinates[prev_image_id])
        dt_next = float(coordinates[next_image_id] - coordinates[image_id])
        if not np.isfinite(dt_prev) or not np.isfinite(dt_next) or dt_prev <= 0.0 or dt_next <= 0.0:
            raise ValueError(
                f"temporal acceleration produced invalid {coordinate} gaps for "
                f"{solve_state.image(prev_image_id).name!r}, {reconstruction.image(image_id).name!r}, "
                f"{solve_state.image(next_image_id).name!r}"
            )
        priors.append(
            TemporalAccelerationPrior(
                prev_image_id, image_id, next_image_id, dt_prev, dt_next, float(np.sqrt(mean_observation_count))
            )
        )
    return priors


def _depth_prior_outlier_masks(solve_state: SolveState, max_depth: float | None):
    """Prepare pass-local masks without changing the image sidecars."""
    masks = {}
    active = newly_marked = total = 0
    if max_depth is None:
        return masks
    for image_id in solve_state.image_order:
        data = solve_state.image_data(image_id)
        depth = np.asarray(data.depth_values)
        if depth.size == 0:
            continue
        previous = np.asarray(data.is_depth_outlier, dtype=bool)
        if previous.size == 0:
            previous = np.zeros(depth.shape, dtype=bool)
        if previous.shape != depth.shape:
            raise ValueError(f"image {image_id} depth outlier mask does not match depth values")
        threshold = np.isfinite(depth) & (depth > max_depth)
        masks[image_id] = np.asarray(previous | threshold, dtype=np.uint8)
        active += int(masks[image_id].sum())
        newly_marked += int((threshold & ~previous).sum())
        total += depth.size
    logger.info(
        "Depth-prior threshold %g m marked %d new outliers (%d/%d active)",
        max_depth,
        newly_marked,
        active,
        total,
    )
    return masks


@dataclass(kw_only=True)
class GlobalPositioner:
    """Run both positioning passes on upstream COLMAP problems."""

    solve_state: SolveState
    consecutive_pair_ids: list[int]
    sequence_id_to_index: dict[int, int]
    inlier_thresholds: InlierThresholdOptions
    boundary_depth_outliers_marked: bool
    options: GPOptions
    output_dir: Path
    replay: ReplayCache
    location_priors: LocationPriorSet | None = None
    persist_intermediate_reconstructions: bool = False
    playback_trace: PlaybackTraceRecorder | None = None

    @property
    def reconstruction(self) -> pycolmap.Reconstruction:
        return self.solve_state.reconstruction

    @staticmethod
    def prepare_bearings(state: SolveState) -> None:
        native.prepare_image_bearings(state.reconstruction, state.sidecars)

    @staticmethod
    def result_for_replay(result) -> dict:
        diagnostics = result.diagnostics
        return {
            "success": bool(result.success),
            "dmap_scale_map": dict(result.depth_map_scales),
            "debug_initial_frame_centers": dict(result.initial_frame_centers),
            "debug_initial_point3D_xyz": dict(result.initial_point3D_xyz),
            "debug_initial_bata_scales": dict(result.initial_bata_scales),
            "debug_diagnostics": {
                "num_bata_residuals": diagnostics.num_bata_residuals,
                "num_metric_depth_residuals": diagnostics.num_metric_depth_residuals,
                "num_scale_prior_residuals": diagnostics.num_scale_prior_residuals,
                **(
                    {
                        "num_temporal_acceleration_residuals": diagnostics.num_temporal_acceleration_residuals,
                    }
                    if diagnostics.num_temporal_acceleration_residuals
                    else {}
                ),
                "num_regular_observations_used": diagnostics.num_regular_observations_used,
                "num_lc_observations_used": diagnostics.num_loop_closure_observations_used,
                "num_bata_scales": diagnostics.num_bata_scales,
                "num_dmap_scales": diagnostics.num_depth_map_scales,
                "num_frame_centers": diagnostics.num_camera_centers,
                "num_point3D_xyz": diagnostics.num_point3D_parameters,
                "num_residual_blocks": diagnostics.num_residual_blocks,
                "num_parameter_blocks": diagnostics.num_parameter_blocks,
                "num_parameters": diagnostics.num_parameters,
                "num_iterations": diagnostics.num_iterations,
                "termination_type": diagnostics.termination_type,
                "initial_cost": diagnostics.initial_cost,
                "final_cost": diagnostics.final_cost,
            },
        }

    def run_pass(
        self,
        stage,
        working,
        prepared,
        replay_images,
        depth_masks,
        temporal_priors,
        *,
        initial_depth_map_scales=None,
    ):
        record = self.replay.write_enabled(stage)
        input_summary = None
        if record:
            tracks = snapshot_tracks(self.solve_state, reconstruction=working)
            input_summary = gp_input_summary(stage, self.solve_state, replay_images, tracks, working.cameras)
            self.replay.write_json(stage, "input_summary.json", input_summary)
            self.replay.write_json(stage, "input_rotations.json", gp_input_rotations(replay_images))
        callback = None
        playback_options = SolverPlaybackOptions()
        if self.playback_trace is not None:
            callback = self.playback_trace.attach_global_positioning(playback_options, stage)
        stddev = self.options.common.bearing_kp_stddev
        if stage == "gp2":
            stddev *= self.options.second_pass.relax_angular_stddevs
        location_priors = self._location_priors(stage)
        scale_prior_stddev = None
        if location_priors is not None and location_priors.num_active_anchors(working) >= 2:
            scale_prior_stddev = float(location_priors.options.gp_relaxed_scale_prior_stddev)
        result = run_global_positioning(
            self.options,
            working,
            self.solve_state.sidecars,
            prepared,
            stddev,
            second_pass=stage == "gp2",
            initial_depth_map_scales=initial_depth_map_scales,
            temporal_priors=temporal_priors,
            depth_outliers_marked=self.boundary_depth_outliers_marked
            or self.options.track_filter.depth_prior_outlier_max_depth is not None,
            depth_outlier_masks=depth_masks,
            playback_options=playback_options,
            playback_callback=callback,
            capture_state=record,
            location_priors=location_priors,
            scale_prior_stddev=scale_prior_stddev,
        )
        if record:
            replay_result = self.result_for_replay(result)
            if stage == "gp1":
                initial_state = capture_gp_initial_state(
                    stage, initial_depth_map_scales or {}, replay_result, input_summary, tracks
                )
                self.replay.write_pickle(stage, "initial_state.pkl", initial_state)
                self.replay.write_json(
                    stage,
                    "initial_summary.json",
                    gp_initial_state_summary(initial_state),
                )
            self.replay.write_json(
                stage,
                "output_summary.json",
                gp_output_summary(
                    stage,
                    self.solve_state,
                    replay_result,
                    reconstruction=working,
                    depth_outlier_masks=depth_masks,
                ),
            )
        return result

    def _location_priors(self, stage: str) -> LocationPriorSet | None:
        priors = self.location_priors
        if priors is None or not priors.options.enabled or not priors.options.use_in_global_positioning:
            return None
        if stage == "gp1" and not priors.options.use_in_gp1:
            return None
        return priors

    def filter_tracks(self, working):
        focal_priors = {
            image.camera.has_prior_focal_length for image in working.images.values() if image.num_points3D > 0
        }
        if len(focal_priors) > 1:
            raise ValueError("Post-GP filtering requires uniform camera focal-prior flags")
        max_angle_error = self.inlier_thresholds.max_angle_error * (2 if focal_priors == {False} else 1)
        observations = pycolmap.ObservationManager(working)
        point_ids = working.point3D_ids()
        observations.filter_points3D_with_large_reprojection_error(
            max_angle_error, point_ids, pycolmap.ReprojectionErrorType.ANGULAR
        )
        observations.filter_points3D_with_small_triangulation_angle(
            self.inlier_thresholds.min_triangulation_angle, point_ids
        )
        if self.replay.write_enabled("ba_start"):
            self.replay.capture_ba_start_tracks(snapshot_tracks(self.solve_state, reconstruction=working))

    def save_checkpoint_before_ba(self) -> None:
        if not self.persist_intermediate_reconstructions:
            return
        with reconstruction_checkpoint_directory(self.output_dir, "rec-gp", persist=True) as checkpoint_dir:
            self.reconstruction.write(checkpoint_dir)
            logger.info("Saved GP checkpoint: %s", checkpoint_dir)

    def position(self) -> None:
        start_time = sync_time()
        working = pycolmap.Reconstruction(self.reconstruction)
        replay_images = (
            snapshot_images(self.solve_state, reconstruction=working)
            if self.replay.write_enabled("gp1") or self.replay.write_enabled("gp2")
            else None
        )
        temporal_prior_specs = []
        temporal = self.options.temporal_acceleration
        if temporal.first_pass_weight > 0.0 or temporal.second_pass_weight > 0.0:
            temporal_prior_specs = _build_temporal_acceleration_prior_specs(
                solve_state=self.solve_state,
                reconstruction=working,
                consecutive_pair_ids=self.consecutive_pair_ids,
                sequence_id_to_index=self.sequence_id_to_index,
                coordinate=temporal.coordinate,
            )
            logger.info(
                "Temporal acceleration (%s coordinate) has %d valid adjacent triplets",
                temporal.coordinate,
                len(temporal_prior_specs),
            )

        max_depth = self.options.track_filter.depth_prior_outlier_max_depth
        depth_masks = _depth_prior_outlier_masks(self.solve_state, max_depth)
        image_timeline = [
            int(image_id) for image_id, _ in sorted(self.sequence_id_to_index.items(), key=lambda item: item[1])
        ]
        prepared = prepare_observations(working, self.solve_state.sidecars, self.options, image_timeline)
        # GP1: initialize and solve, with optional warmup.
        logger.info("Running first global positioning ...")
        result = self.run_pass("gp1", working, prepared, replay_images, depth_masks, temporal_prior_specs)
        # GP2: refine GP1 geometry and reuse its depth scales.
        if self.options.second_pass.enabled:
            if self.options.track_filter.depth_prior_outlier_stages == "gp1":
                depth_masks = {}
            initial_depth_map_scales = result.depth_map_scales
            if self._location_priors("gp2") is not None:
                # Bring GP1 into the prior frame (scale and translation) before refining with the priors.
                initial_depth_map_scales = self.location_priors.align_gp1_to_location_priors_4dof(
                    working, dict(initial_depth_map_scales)
                )
            logger.info("Running second global positioning ...")
            self.run_pass(
                "gp2",
                working,
                prepared,
                replay_images,
                depth_masks,
                temporal_prior_specs,
                initial_depth_map_scales=initial_depth_map_scales,
            )
        self.filter_tracks(working)
        self.solve_state.reconstruction = working
        self.save_checkpoint_before_ba()
        self.solve_state.sidecars.clear_tracks()
        record_timing("global_positioning", sync_time() - start_time)
        log_memory("global_positioning")
