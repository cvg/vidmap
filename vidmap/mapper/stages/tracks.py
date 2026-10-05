"""Track establishment and filtering."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from vidmap.mapper.native.extension import native
from vidmap.mapper.native.state import SolveState
from vidmap.mapper.options.positioning import MapperTrackOptions
from vidmap.mapper.replay.cache import ReplayCache
from vidmap.mapper.replay.evidence.canonical import ReplayTrackSnapshot, snapshot_images, snapshot_tracks
from vidmap.mapper.replay.evidence.scene import native_problem_summary, pose_graph_summary
from vidmap.mapper.replay.evidence.stages import capture_tracks_state, tracks_summary

logger = logging.getLogger(__name__)


def _build_native_options(options: MapperTrackOptions):
    native_options = native.TrackEstablishmentOptions()
    native_options.min_num_views_per_track = options.min_num_views_per_track
    native_options.max_num_views_per_track = options.max_num_views_per_track
    native_options.two_view_depth_gate = options.two_view_depth_gate
    return native_options


@dataclass(kw_only=True)
class TrackBuilder:
    solve_state: SolveState
    options: MapperTrackOptions
    boundary_depth_outliers_marked: bool
    replay: ReplayCache

    def build(self) -> None:
        state = self.solve_state
        opt_track = _build_native_options(self.options)
        record = self.replay.write_enabled("tracks")
        state.retriangulation_graph = native.create_correspondence_graph(
            state.reconstruction,
            state.sidecars,
            state.pair_order,
        )
        valid_pair_order = [pair_id for pair_id in state.pair_order if state.pose_graph.is_valid(pair_id)]
        result = native.establish_tracks(
            state.reconstruction,
            state.pose_graph,
            state.sidecars,
            state.reconstruction.reg_image_ids(),
            valid_pair_order,
            opt_track,
            loop_closure_second_pass=self.options.loop_closure_second_pass,
            include_loop_closure_observations=self.options.include_loop_closure_observations,
            capture_tracks=record,
        )
        logger.info("Before filter: %d, after filter: %d", result.num_full_tracks, result.num_tracks)
        if record:
            images = snapshot_images(state)
            tracks = snapshot_tracks(state)
            full_data = result.full_track_data
            full_tracks = {
                point_id: ReplayTrackSnapshot(
                    xyz=np.zeros(3),
                    observations=np.asarray(
                        [(el.image_id, el.point2D_idx) for el in track.elements], dtype=np.uint32
                    ).reshape((-1, 2)),
                    loop_closure_observations=np.asarray(full_data[point_id].loop_closure_observations),
                )
                for point_id, track in result.full_tracks.items()
            }
            args = (state, images, full_tracks, tracks, self.boundary_depth_outliers_marked)
            self.replay.write_pickle("tracks", "state.pkl", capture_tracks_state(*args))
            self.replay.write_json("tracks", "summary.json", tracks_summary(*args))
        if any(self.replay.write_enabled(stage) for stage in ("gp1", "gp2", "ba_start")):
            state.replay_graph_summary = native_problem_summary(state)
            state.replay_pose_graph_summary = pose_graph_summary(state)
        state.sidecars.clear_pairs()
