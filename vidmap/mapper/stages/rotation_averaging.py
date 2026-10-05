"""Rotation averaging and evaluation operations."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from vidmap.mapper.native.extension import native
from vidmap.mapper.native.state import SolveState
from vidmap.mapper.options.view_graph import RAOptions
from vidmap.mapper.replay.cache import ReplayCache
from vidmap.mapper.replay.evidence.canonical import rotation_artifact, snapshot_images
from vidmap.mapper.replay.evidence.stages import ra_summary

logger = logging.getLogger(__name__)


def _build_native_options(options: RAOptions):
    native_options = native.RotationAveragingOptions()
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
        registered_image_ids = {image_id for image_id, image in state.reconstruction.images.items() if image.has_pose}
        result = native.run_video_rotation_averaging(
            opt_ra,
            state.reconstruction,
            state.pose_graph,
            state.sidecars,
        )

        if opt_ra.max_rotation_error_deg > 0.0 and result.success:
            registered_image_ids.intersection_update(int(value) for value in result.registered_image_ids)

        registered_frames = set(state.reconstruction.reg_frame_ids())
        for image_id in state.image_order:
            image = state.image(image_id)
            frame = state.reconstruction.frame(image.frame_id)
            if image_id not in registered_image_ids:
                if image.has_pose:
                    if image.frame_id in registered_frames:
                        state.reconstruction.deregister_frame(image.frame_id)
                        registered_frames.discard(image.frame_id)
                    frame.reset_pose()
            elif image.frame_id not in registered_frames:
                state.reconstruction.register_frame(image.frame_id)
                registered_frames.add(image.frame_id)
        return result

    def average(self) -> None:
        state = self.solve_state
        rec = self.solve_state.reconstruction
        opt_ra = _build_native_options(self.options)
        for pass_index in range(2 if opt_ra.max_rotation_error_deg > 0.0 else 1):
            if not self.run_pass(opt_ra).success:
                raise RuntimeError("Rotation averaging failed")

            if pass_index == 0:
                filtered_consecutive_pairs = [
                    pid for pid in self.consecutive_pair_ids if not state.pose_graph.is_valid(pid)
                ]
                if len(filtered_consecutive_pairs) > 0:
                    logger.warning(
                        f"{len(filtered_consecutive_pairs)} consecutive pairs were filtered out, continuing..."
                    )
        logger.info(f"{len(rec.reg_image_ids())} are within the connected component.")
        if self.replay.write_enabled("ra"):
            images = snapshot_images(state)
            summary = ra_summary(state, images, self.filtered_consecutive_pair_ids)
            self.replay.write_json("ra", "rotations.json", rotation_artifact(images))
            self.replay.write_json("ra", "summary.json", summary)
