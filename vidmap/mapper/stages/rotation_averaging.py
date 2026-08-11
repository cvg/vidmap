"""Rotation averaging and evaluation operations."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from vidmap.mapper.native.extension import native
from vidmap.mapper.native.state import SolveState
from vidmap.mapper.options.view_graph import RAOptions
from vidmap.mapper.replay.cache import ReplayCache
from vidmap.mapper.replay.evidence.canonical import rotation_artifact
from vidmap.mapper.replay.evidence.stages import ra_summary

logger = logging.getLogger(__name__)


def _build_native_options(options: RAOptions):
    native_options = native.RotationAveragingOptions()
    native_options.random_seed = options.random_seed
    native_options.max_rotation_error_deg = options.max_rotation_error_deg
    native_options.tracking_huber_scale = options.video_tracking_huber_scale
    native_options.loop_closure_cauchy_scale = options.video_lc_cauchy_scale
    native_options.skip_risky_loop_closure_pairs = options.filter_risky_loop_closure_pairs
    native_options.filter_unregistered_images = options.filter_unregistered_images
    native_options.num_threads = 1 if options.num_threads is None else int(options.num_threads)
    return native_options


@dataclass(kw_only=True)
class RotationAverager:
    solve_state: SolveState
    options: RAOptions
    consecutive_pair_ids: list[int]
    filtered_consecutive_pair_ids: set[int]
    replay: ReplayCache

    def run_pass(self, opt_ra: native.RotationAveragingOptions):
        """Run one native solve while preserving the caller's pose state."""
        state = self.solve_state
        translations = {
            image_id: np.asarray(image.pose.translation, dtype=np.float64).copy()
            for image_id, image in state.image_records().items()
            if image.pose.has_pose
        }
        result = native.run_video_rotation_averaging(
            opt_ra,
            state.image_order,
            state.pair_order,
            state.native_problem,
        )

        registered_image_ids = set(translations)
        if opt_ra.max_rotation_error_deg > 0.0 and result.success:
            registered_image_ids.intersection_update(int(value) for value in result.registered_image_ids)

        for image_id in state.image_order:
            record = state.image(image_id)
            if image_id not in registered_image_ids:
                record.pose = native.PoseRecord()
            else:
                record.pose.has_pose = True
                record.pose.translation = translations[image_id]
            state.update_image(record)
        state.export_poses()
        return result

    def average(self) -> None:
        state = self.solve_state
        rec = self.solve_state.reconstruction
        opt_ra = _build_native_options(self.options)
        for pass_index, image_order_passes in enumerate((1, 2)):
            opt_ra.image_order_passes = image_order_passes
            self.run_pass(opt_ra)

            if pass_index == 0:
                filtered_consecutive_pairs = [pid for pid in self.consecutive_pair_ids if not state.pair(pid).is_valid]
                if len(filtered_consecutive_pairs) > 0:
                    logger.warning(
                        f"{len(filtered_consecutive_pairs)} consecutive pairs were filtered out, continuing..."
                    )
        logger.info(f"{len(rec.reg_image_ids())} are within the connected component.")
        needs_summary = self.replay.write_enabled("ra")
        summary = (
            ra_summary(
                state,
                state.image_records(),
                self.filtered_consecutive_pair_ids,
            )
            if needs_summary
            else None
        )
        if needs_summary:
            self.replay.write_json("ra", "rotations.json", rotation_artifact(state.image_records()))
            self.replay.write_json("ra", "summary.json", summary)
