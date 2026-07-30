"""Track establishment and filtering."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from vidmap.mapper.native.extension import native
from vidmap.mapper.native.state import SolveState
from vidmap.mapper.options.positioning import MapperTrackOptions
from vidmap.mapper.replay.cache import ReplayCache
from vidmap.mapper.replay.evidence.stages import capture_tracks_state, tracks_summary

logger = logging.getLogger(__name__)


def _build_native_options(options: MapperTrackOptions):
    establishment_options = native.TrackEstablishmentOptions()
    establishment_options.min_num_views_per_track = options.min_num_views_per_track
    filtering_options = native.TrackProblemFilterOptions()
    filtering_options.min_num_views_per_track = options.min_num_views_per_track
    filtering_options.max_num_views_per_track = options.max_num_views_per_track
    filtering_options.two_view_depth_gate = options.two_view_depth_gate
    return establishment_options, filtering_options


@dataclass(kw_only=True)
class TrackBuilder:
    solve_state: SolveState
    options: MapperTrackOptions
    boundary_depth_outliers_marked: bool
    replay: ReplayCache

    def build(self) -> dict[int, native.TrackRecord]:
        state = self.solve_state
        images = state.image_records()
        opt_track, opt_track_filter = _build_native_options(self.options)

        registered_image_ids = [iid for iid in sorted(images) if images[iid].pose.has_pose]
        valid_pair_order = [pair_id for pair_id in state.pair_order if state.pair(pair_id).is_valid]
        tracks_full_records = native.establish_full_tracks(
            state.native_problem,
            registered_image_ids,
            valid_pair_order,
            opt_track,
            loop_closure_second_pass=self.options.loop_closure_second_pass,
        )
        track_records = native.filter_tracks_for_problem(
            state.native_problem,
            registered_image_ids,
            tracks_full_records,
            opt_track_filter,
        )
        tracks_full = {int(track.point3D_id): track for track in tracks_full_records}
        tracks = {int(track.point3D_id): track for track in track_records}

        logger.info(f"Before filter: {len(tracks_full)}, after filter: {len(tracks)}")
        if not self.options.include_loop_closure_observations:
            for track_id, track in list(tracks.items()):
                track.loop_closure_observations = np.empty((0, 2), dtype=np.uint32)
                tracks[track_id] = track
        state.replace_tracks(list(tracks.values()))
        needs_summary = self.replay.write_enabled("tracks")
        summary = (
            tracks_summary(
                state,
                images,
                tracks_full,
                tracks,
                None,
                self.boundary_depth_outliers_marked,
            )
            if needs_summary
            else None
        )
        if needs_summary:
            self.replay.write_pickle(
                "tracks",
                "state.pkl",
                capture_tracks_state(
                    state,
                    images,
                    tracks_full,
                    tracks,
                    None,
                    self.boundary_depth_outliers_marked,
                ),
            )
            self.replay.write_json("tracks", "summary.json", summary)

        return tracks
