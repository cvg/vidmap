"""Global positioning and BA-boundary reconstruction preparation."""

from __future__ import annotations

import copy
import logging
import re
import struct
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path

import numpy as np
import pycolmap

from vidmap.mapper.checkpoints import reconstruction_checkpoint_directory
from vidmap.mapper.native.extension import native
from vidmap.mapper.native.records import pose_record_to_pycolmap
from vidmap.mapper.native.state import SolveState
from vidmap.mapper.options.positioning import GPOptions
from vidmap.mapper.options.view_graph import InlierThresholdOptions
from vidmap.mapper.playback_trace import PlaybackTraceRecorder
from vidmap.mapper.replay.cache import ReplayCache
from vidmap.mapper.replay.evidence.canonical import ReplayImageSnapshot
from vidmap.mapper.replay.evidence.stages import (
    capture_gp_initial_state,
    gp_initial_state_summary,
    gp_input_rotations,
    gp_input_summary,
    gp_output_summary,
)
from vidmap.utils.profiling import log_memory, record_timing, sync_time

from .native_options import (
    GlobalPositioningTolerances,
    build_first_global_positioning_options,
    configure_second_global_positioning_options,
    configure_temporal_acceleration_options,
)

logger = logging.getLogger(__name__)
_TIMESTAMP = re.compile(r"^(\d+(?:\.\d+)?)(?:-|$)")


def angular_stds_to_xyz_covar(bearings, angular_stds):
    """Propagate diagonal image-plane angular covariance to unit bearings."""
    count = bearings.shape[0]
    image_covariance = np.zeros((count, 3, 3))
    image_covariance[:, 0, 0] = angular_stds[:, 0] ** 2
    image_covariance[:, 1, 1] = angular_stds[:, 1] ** 2

    bearing_z = np.abs(bearings[:, 2])[:, None, None]
    outer_product = np.einsum("ni,nj->nij", bearings, bearings)
    jacobian = bearing_z * (np.eye(3)[None, :, :] - outer_product)
    return np.einsum("nij,njk,nlk->nil", jacobian, image_covariance, jacobian)


def _copy_pose(pose: pycolmap.Rigid3d | None) -> pycolmap.Rigid3d | None:
    if pose is None:
        return None
    return pycolmap.Rigid3d(
        rotation=np.asarray(pose.rotation.quat, dtype=np.float64),
        translation=np.asarray(pose.translation, dtype=np.float64),
    )


def _image_center(record) -> np.ndarray:
    pose = pose_record_to_pycolmap(record.pose)
    return np.asarray(-pose.rotation.matrix().T @ pose.translation, dtype=np.float64)


def _track_observation_counts(tracks) -> dict[int, int]:
    counts: dict[int, int] = {}
    for track in tracks.values():
        for observations in (track.observations, track.loop_closure_observations):
            for image_id in np.asarray(observations)[:, 0]:
                key = int(image_id)
                if key not in counts:
                    counts[key] = 0
                counts[key] += 1
    return counts


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
    tracks,
    consecutive_pair_ids,
    sequence_id_to_index,
    coordinate: str = "timestamp",
) -> list[dict[str, float | int]]:
    """Build smoothness priors across valid consecutive tracking edges."""
    ordered_image_ids = [image_id for image_id, _ in sorted(sequence_id_to_index.items(), key=lambda item: item[1])]
    if coordinate == "timestamp":
        timeline = _image_timestamps_seconds([solve_state.image(image_id).name for image_id in ordered_image_ids])
        coordinates = dict(zip(ordered_image_ids, timeline, strict=True))
    elif coordinate == "keyframe_index":
        coordinates = {image_id: index for index, image_id in enumerate(ordered_image_ids)}
    else:
        raise ValueError(f"Unsupported temporal-acceleration coordinate: {coordinate!r}")
    registered_image_ids = {image_id for image_id in ordered_image_ids if solve_state.image(image_id).pose.has_pose}
    valid_edges = set()
    for pair_id in consecutive_pair_ids:
        pair = solve_state.pair(pair_id)
        if not pair.is_valid:
            continue
        index1 = sequence_id_to_index[int(pair.image_id1)]
        index2 = sequence_id_to_index[int(pair.image_id2)]
        if abs(index1 - index2) == 1:
            valid_edges.add(frozenset((int(pair.image_id1), int(pair.image_id2))))

    observation_counts = _track_observation_counts(tracks)
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
                f"{solve_state.image(prev_image_id).name!r}, {solve_state.image(image_id).name!r}, "
                f"{solve_state.image(next_image_id).name!r}"
            )
        priors.append(
            {
                "prev_image_id": prev_image_id,
                "image_id": image_id,
                "next_image_id": next_image_id,
                "dt_prev": dt_prev,
                "dt_next": dt_next,
                "sqrt_observation_count": float(np.sqrt(mean_observation_count)),
            }
        )
    return priors


def _mark_depth_prior_outliers(solve_state: SolveState, max_depth: float):
    """Temporarily OR a raw-depth threshold into native GP outlier masks."""
    previous_masks = {}
    active = 0
    newly_marked = 0
    total = 0
    for image_id in solve_state.image_order:
        image = solve_state.image(image_id)
        depth_values = np.asarray(image.depth_values, dtype=np.float64)
        if depth_values.size == 0:
            continue
        existing = np.asarray(image.is_depth_outlier, dtype=bool)
        if existing.size == 0:
            existing = np.zeros(depth_values.shape, dtype=bool)
        if existing.shape != depth_values.shape:
            raise ValueError(
                f"image {image_id} depth-outlier mask shape {existing.shape} "
                f"does not match depth values {depth_values.shape}"
            )
        threshold_mask = np.isfinite(depth_values) & (depth_values > max_depth)
        combined = existing | threshold_mask
        previous_masks[image_id] = existing.copy()
        image.is_depth_outlier = np.asarray(combined, dtype=np.uint8)
        solve_state.update_image(image)
        active += int(combined.sum())
        newly_marked += int((threshold_mask & ~existing).sum())
        total += int(combined.size)
    return previous_masks, {"active": active, "new": newly_marked, "total": total}


def _restore_depth_prior_outliers(solve_state: SolveState, previous_masks) -> None:
    for image_id, mask in previous_masks.items():
        image = solve_state.image(image_id)
        image.is_depth_outlier = np.asarray(mask, dtype=np.uint8)
        solve_state.update_image(image)


@dataclass(kw_only=True)
class GlobalPositioner:
    """Run the two native positioning passes and install their stock scene."""

    solve_state: SolveState
    tracks: dict[int, native.TrackRecord]
    consecutive_pair_ids: list[int]
    sequence_id_to_index: dict[int, int]
    inlier_thresholds: InlierThresholdOptions
    boundary_depth_outliers_marked: bool
    options: GPOptions
    output_dir: Path
    replay: ReplayCache
    persist_intermediate_reconstructions: bool = False
    playback_trace: PlaybackTraceRecorder | None = None
    first_pass_tolerances: GlobalPositioningTolerances = field(init=False)

    @property
    def reconstruction(self) -> pycolmap.Reconstruction:
        return self.solve_state.reconstruction

    @staticmethod
    def prepare_bearings(state: SolveState, bearing_kp_stddev: float) -> None:
        native.prepare_image_bearings(state.native_problem)
        for image_id in state.image_order:
            image = state.image(image_id)
            camera = state.reconstruction.camera(image.camera_id)
            fx, fy = camera.focal_length_x, camera.focal_length_y
            bearings = np.asarray(image.bearings)
            angular_stddevs = bearing_kp_stddev * np.ones((len(image.keypoints), 1)) / (fx, fy)
            bearing_covars = angular_stds_to_xyz_covar(bearings, angular_stddevs)
            image.angular_stddevs = np.sqrt(np.diagonal(bearing_covars, axis1=1, axis2=2))[:, :-1]
            state.update_image(image)

    @staticmethod
    def require_success(result: native.GlobalPositioningResult, stage: str) -> None:
        if not result.success:
            raise RuntimeError(f"{stage} global positioning failed")

    @staticmethod
    def result_for_replay(result: native.GlobalPositioningResult) -> dict:
        diagnostics = result.diagnostics
        return {
            "success": bool(result.success),
            "dmap_scale_map": dict(result.depth_map_scales),
            "debug_initial_frame_centers": dict(result.initial_frame_centers),
            "debug_initial_point3D_xyz": dict(result.initial_point3D_xyz),
            "debug_initial_bata_scales": dict(result.initial_bata_scales),
            "debug_final_bata_scales": dict(result.final_bata_scales),
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

    def snapshot_images(self) -> dict[int, ReplayImageSnapshot]:
        snapshots = {}
        for image_id, image in self.solve_state.image_records().items():
            pose = pose_record_to_pycolmap(image.pose)
            snapshots[image_id] = ReplayImageSnapshot(
                image_id=image_id,
                camera_id=image.camera_id,
                frame_id=image.frame_id,
                name=image.name,
                has_pose=image.pose.has_pose,
                cam_from_world=_copy_pose(pose),
                features=np.asarray(image.keypoints).copy(),
                features_undist=np.asarray(image.bearings).copy(),
                depth_priors=np.asarray(image.depth_values).copy(),
                depth_prior_stddevs=np.asarray(image.depth_stddevs).copy(),
                depth_prior_validity=np.asarray(image.depth_validity, dtype=bool).copy(),
                angular_stddevs=np.asarray(image.angular_stddevs).copy(),
                is_inlier=np.asarray(image.is_inlier, dtype=bool).copy(),
                is_track_anchor=np.asarray(image.is_track_anchor, dtype=bool).copy(),
                is_depth_outlier=np.asarray(image.is_depth_outlier, dtype=bool).copy(),
            )
        return snapshots

    def current_tracks(self) -> dict[int, native.TrackRecord]:
        return {
            int(point3D_id): self.solve_state.native_problem.track(point3D_id)
            for point3D_id in self.solve_state.native_problem.point3D_ids
        }

    def export_solved_scene(self) -> None:
        """Publish one solved pass to the pycolmap checkpoint.

        Positioning moves point coordinates and poses but never track membership, so the
        value-only track export is exact; it falls back to a full rebuild if ids diverge.
        """
        self.solve_state.export_cameras()
        self.solve_state.export_poses()
        self.solve_state.export_track_values()

    def first_pass(self, native_options, replay_images):
        input_summary = None
        if self.replay.write_enabled("gp1"):
            input_summary = gp_input_summary(
                "gp1",
                self.solve_state,
                replay_images,
                self.tracks,
                self.reconstruction.cameras,
            )
            self.replay.write_json("gp1", "input_summary.json", input_summary)
            self.replay.write_json("gp1", "input_rotations.json", gp_input_rotations(replay_images))

        if self.playback_trace is not None:
            self.playback_trace.attach_global_positioning(native_options, "gp1")
        result = native.run_global_positioning(native_options, self.solve_state.native_problem)
        self.export_solved_scene()
        replay_result = self.result_for_replay(result)
        if self.replay.write_enabled("gp1"):
            initial_state = capture_gp_initial_state("gp1", native_options, replay_result, input_summary, self.tracks)
            self.replay.write_pickle("gp1", "initial_state.pkl", initial_state)
            self.replay.write_json("gp1", "initial_summary.json", gp_initial_state_summary(initial_state))
            self.replay.write_json(
                "gp1",
                "output_summary.json",
                gp_output_summary("gp1", self.solve_state, replay_result),
            )
        self.require_success(result, "First")
        return result

    def second_pass(self, native_options, first_result, replay_images, temporal_prior_specs):
        logger.info("Running second global positioning ...")
        for image in self.solve_state.image_records().values():
            image.angular_stddevs = np.asarray(image.angular_stddevs) * self.options.second_pass.relax_angular_stddevs
            self.solve_state.update_image(image)
        initial_frame_centers = {}
        if self.options.second_pass.center_init_mode == "python_frame_centers":
            initial_frame_centers = {
                int(image.frame_id): _image_center(image)
                for image in self.solve_state.image_records().values()
                if image.pose.has_pose
            }
        configure_second_global_positioning_options(
            native_options,
            self.options,
            first_result,
            self.first_pass_tolerances,
            initial_frame_centers,
        )
        configure_temporal_acceleration_options(
            native_options,
            self.options,
            stage="gp2",
            prior_specs=temporal_prior_specs,
        )
        input_summary = None
        if self.replay.write_enabled("gp2"):
            input_summary = gp_input_summary(
                "gp2",
                self.solve_state,
                replay_images,
                self.current_tracks(),
                self.reconstruction.cameras,
            )
            self.replay.write_json("gp2", "input_rotations.json", gp_input_rotations(replay_images))
            self.replay.write_json("gp2", "input_summary.json", input_summary)
        if self.playback_trace is not None:
            self.playback_trace.attach_global_positioning(native_options, "gp2")
        result = native.run_global_positioning(native_options, self.solve_state.native_problem)
        self.export_solved_scene()
        replay_result = self.result_for_replay(result)
        if self.replay.write_enabled("gp2"):
            self.replay.write_json(
                "gp2",
                "output_summary.json",
                gp_output_summary("gp2", self.solve_state, replay_result),
            )
        self.require_success(result, "Second")
        return result

    def filter_tracks(self):
        tracks = list(self.current_tracks().values())
        angle_result = native.filter_tracks_by_angle(
            self.solve_state.native_problem,
            tracks,
            self.inlier_thresholds.max_angle_error,
        )
        triangulation_result = native.filter_tracks_by_triangulation_angle(
            self.solve_state.native_problem,
            angle_result.tracks,
            self.inlier_thresholds.min_triangulation_angle,
        )
        filtered = {int(track.point3D_id): track for track in triangulation_result.tracks}
        self.replay.capture_ba_start_tracks(filtered)
        if self.options.track_filter.skip_zero_observation_points:
            filtered = {track_id: track for track_id, track in filtered.items() if len(track.observations) > 0}
        self.solve_state.replace_tracks(list(filtered.values()))
        self.solve_state.export_tracks()
        if self.options.track_filter.update_point3d_errors:
            self.reconstruction.update_point_3d_errors()
            self.solve_state.import_tracks()
        return filtered

    @staticmethod
    def canonicalize_point_errors(reconstruction, decimals: int = 6) -> None:
        for point_id in reconstruction.point3D_ids():
            point = reconstruction.point3D(point_id)
            point.error = round(float(point.error), decimals)

    @staticmethod
    def rewrite_images_bin_sorted(reconstruction, checkpoint_dir) -> None:
        path = checkpoint_dir / "images.bin"
        image_ids = sorted(int(image_id) for image_id in reconstruction.reg_image_ids())
        with path.open("wb") as file_handle:
            file_handle.write(struct.pack("<Q", len(image_ids)))
            for image_id in image_ids:
                image = reconstruction.image(image_id)
                pose = image.cam_from_world()
                qvec = np.asarray(pose.rotation.quat, dtype=np.float64)
                tvec = np.asarray(pose.translation, dtype=np.float64)
                file_handle.write(struct.pack("<I", image_id))
                file_handle.write(struct.pack("<4d", *qvec.tolist()))
                file_handle.write(struct.pack("<3d", *tvec.tolist()))
                file_handle.write(struct.pack("<I", int(image.camera_id)))
                file_handle.write(str(image.name).encode("utf-8") + b"\x00")
                points2D = list(image.points2D)
                file_handle.write(struct.pack("<Q", len(points2D)))
                for point2D in points2D:
                    xy = np.asarray(point2D.xy, dtype=np.float64)
                    point3D_id = int(point2D.point3D_id)
                    if point3D_id > 2**63 - 1:
                        point3D_id = -1
                    file_handle.write(struct.pack("<ddq", float(xy[0]), float(xy[1]), point3D_id))

    def write_checkpoint(self, checkpoint_dir: Path) -> None:
        if self.options.common.canonical_checkpoint:
            self.canonicalize_point_errors(self.reconstruction)
            self.solve_state.import_tracks()
        self.reconstruction.write(checkpoint_dir)
        if self.options.common.canonical_checkpoint:
            self.rewrite_images_bin_sorted(self.reconstruction, checkpoint_dir)
        logger.info("Saved GP checkpoint: %s", checkpoint_dir)

    def roundtrip_checkpoint(self, checkpoint_dir) -> None:
        cameras = {camera_id: copy.deepcopy(camera) for camera_id, camera in self.reconstruction.cameras.items()}
        reconstruction = pycolmap.Reconstruction(checkpoint_dir)
        for camera_id, source in cameras.items():
            if camera_id in reconstruction.cameras:
                target = reconstruction.camera(camera_id)
                target.params = np.asarray(source.params, dtype=np.float64)
                target.has_prior_focal_length = source.has_prior_focal_length
            else:
                reconstruction.add_camera_with_trivial_rig(source)
        self.solve_state.import_checkpoint(reconstruction)
        logger.info("Round-tripped GP reconstruction before BA via %s", checkpoint_dir)
        return None

    def save_checkpoint_before_ba(self) -> None:
        checkpoint_required = (
            self.persist_intermediate_reconstructions
            or self.options.common.roundtrip_before_ba
            or self.options.common.canonical_checkpoint
        )
        if not checkpoint_required:
            return
        with reconstruction_checkpoint_directory(
            self.output_dir,
            "rec-gp",
            persist=self.persist_intermediate_reconstructions,
        ) as checkpoint_dir:
            self.write_checkpoint(checkpoint_dir)
            if self.options.common.roundtrip_before_ba:
                self.roundtrip_checkpoint(checkpoint_dir)

    def position(self) -> None:
        start_time = sync_time()
        logger.info("Running first global positioning ...")
        replay_images = (
            self.snapshot_images() if self.replay.write_enabled("gp1") or self.replay.write_enabled("gp2") else None
        )
        self.solve_state.replace_tracks(list(self.tracks.values()))

        temporal_prior_specs = []
        temporal = self.options.temporal_acceleration
        if temporal.first_pass_weight > 0.0 or temporal.second_pass_weight > 0.0:
            temporal_prior_specs = _build_temporal_acceleration_prior_specs(
                solve_state=self.solve_state,
                tracks=self.tracks,
                consecutive_pair_ids=self.consecutive_pair_ids,
                sequence_id_to_index=self.sequence_id_to_index,
                coordinate=temporal.coordinate,
            )
            logger.info(
                "Temporal acceleration (%s coordinate) has %d valid adjacent triplets",
                temporal.coordinate,
                len(temporal_prior_specs),
            )

        previous_depth_outlier_masks = None
        max_depth = self.options.track_filter.depth_prior_outlier_max_depth
        if max_depth is not None:
            previous_depth_outlier_masks, counts = _mark_depth_prior_outliers(self.solve_state, max_depth)
            logger.info(
                f"Depth-prior threshold {max_depth:g} m marked {counts['new']} new outliers "
                f"({counts['active']}/{counts['total']} active)"
            )

        native_options, self.first_pass_tolerances = build_first_global_positioning_options(
            self.options,
            depth_outliers_marked=self.boundary_depth_outliers_marked or max_depth is not None,
            image_timeline=[
                int(image_id)
                for image_id, _index in sorted(
                    self.sequence_id_to_index.items(),
                    key=lambda item: item[1],
                )
            ],
        )
        configure_temporal_acceleration_options(
            native_options,
            self.options,
            stage="gp1",
            prior_specs=temporal_prior_specs,
        )
        try:
            result = self.first_pass(native_options, replay_images)
            if (
                previous_depth_outlier_masks is not None
                and self.options.track_filter.depth_prior_outlier_stages == "gp1"
            ):
                _restore_depth_prior_outliers(self.solve_state, previous_depth_outlier_masks)
                previous_depth_outlier_masks = None
            if self.options.second_pass.enabled:
                self.second_pass(native_options, result, replay_images, temporal_prior_specs)
        finally:
            if previous_depth_outlier_masks is not None:
                _restore_depth_prior_outliers(self.solve_state, previous_depth_outlier_masks)

        self.filter_tracks()
        self.save_checkpoint_before_ba()
        record_timing("global_positioning", sync_time() - start_time)
        log_memory("global_positioning")
        return None
