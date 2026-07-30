"""MDRP and inlier-filtering operations."""

from __future__ import annotations

import functools
import logging
from dataclasses import dataclass

import numpy as np
import poselib
import pycolmap
from tqdm import tqdm
from tqdm.contrib.concurrent import thread_map

from vidmap.mapper.native.extension import native
from vidmap.mapper.native.records import pose_record_from_pycolmap
from vidmap.mapper.native.state import SolveState
from vidmap.mapper.options.view_graph import InlierThresholdOptions, MDRPOptions
from vidmap.mapper.replay.cache import ReplayCache
from vidmap.mapper.replay.evidence.stages import capture_relative_pose_state, relative_pose_summary
from vidmap.utils.logging import progress_bars_enabled

from .mdrp import MDRPResult, ValidMDRPResult, estimate_mdrp_pose_for_pair
from .native_options import build_inlier_threshold_options

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RelativePoseResult:
    inlier_thresholds: InlierThresholdOptions
    filtered_consecutive_pairs: set[int]


@dataclass(kw_only=True)
class RelativePoseEstimator:
    solve_state: SolveState
    options: MDRPOptions
    inlier_threshold_options: InlierThresholdOptions
    consecutive_pair_ids: list[int]
    replay: ReplayCache

    def estimate(self) -> RelativePoseResult:
        state = self.solve_state
        rec = self.solve_state.reconstruction
        cameras = rec.cameras
        images = state.image_records()
        consecutive_pair_ids = self.consecutive_pair_ids
        inlier_thresholds = self.inlier_threshold_options
        native_inlier_thresholds = build_inlier_threshold_options(inlier_thresholds)
        filtered_consecutive_pairs = set()
        identity_pose = pycolmap.Rigid3d()
        for pair in state.pair_records().values():
            geometry = pair.geometry
            geometry.cam2_from_cam1 = pose_record_from_pycolmap(identity_pose)
            pair.geometry = geometry
            state.update_pair(pair)

        logger.info("Estimating relative poses using MDRP")
        ransac_options = poselib.RansacOptions(
            {
                "max_iterations": self.options.ransac_max_iterations,
                "max_epipolar_error": self.options.ransac_max_epipolar_error,
            }
        )
        bundle_options = poselib.BundleOptions()
        camera_poselib_cache = {
            camera_id: poselib.Camera(camera.model.name, camera.params.tolist(), camera.width, camera.height)
            for camera_id, camera in cameras.items()
        }
        image_feature_cache = {image_id: np.asarray(image.keypoints) for image_id, image in images.items()}
        image_depth_cache = {image_id: np.asarray(image.depth_values) for image_id, image in images.items()}
        image_valid_cache = {
            image_id: np.asarray(image.depth_validity, dtype=bool) for image_id, image in images.items()
        }

        valid_pair_ids = []
        for image_pair_id, image_pair in tqdm(
            state.pair_records().items(),
            desc="Filtering valid image pairs",
            disable=not progress_bars_enabled(),
        ):
            if image_pair.is_valid:
                valid_pair_ids.append(image_pair_id)
        logger.info("Estimating relative poses for %d image pairs", len(valid_pair_ids))

        write_replay = self.replay.write_enabled("relative_pose")
        worker = functools.partial(
            estimate_mdrp_pose_for_pair,
            images=images,
            compute_reproj_error_outliers=self.options.compute_reproj_error_outliers,
            reproj_outlier_threshold=self.options.reproj_outlier_threshold,
            ransac_options=ransac_options,
            bundle_options=bundle_options,
            camera_poselib_cache=camera_poselib_cache,
            image_feature_cache=image_feature_cache,
            image_depth_cache=image_depth_cache,
            image_valid_cache=image_valid_cache,
        )
        pair_args_list = []
        for image_pair_id in valid_pair_ids:
            image_pair = state.pair(image_pair_id)
            pair_args_list.append(
                (
                    image_pair_id,
                    image_pair.image_id1,
                    image_pair.image_id2,
                    image_pair.all_matches,
                )
            )

        results: dict[int, MDRPResult] = {}
        if self.options.max_workers == 1:
            for pair_args in tqdm(
                pair_args_list,
                desc="Estimating relative poses (MDRP)",
                disable=not progress_bars_enabled(),
            ):
                image_pair_id, result = worker(pair_args)
                results[image_pair_id] = result
        else:
            all_results = thread_map(
                worker,
                pair_args_list,
                max_workers=self.options.max_workers,
                desc="Estimating relative poses (MDRP)",
            )
            for image_pair_id, result in all_results:
                results[image_pair_id] = result
        logger.debug("Weight and inliers were assigned to image pairs unlike in upstream GLOMAP.")
        logger.info("Estimating relative pose done")

        valid_items: list[tuple[int, ValidMDRPResult]] = []
        for image_pair_id, result in results.items():
            if not result["is_valid"]:
                pair = state.pair(image_pair_id)
                pair.is_valid = False
                state.update_pair(pair)
            else:
                valid_items.append((image_pair_id, result))

        for image_pair_id, result in valid_items:
            pair = state.pair(image_pair_id)
            pair.is_valid = True
            geometry = pair.geometry
            geometry.cam2_from_cam1 = pose_record_from_pycolmap(result["cam2_from_cam1"])
            pair.geometry = geometry
            pair.inlier_indices = np.asarray(result["inliers"], dtype=np.int32)
            state.update_pair(pair)

        logger.info("Assigned %d/%d valid MDRP results", len(valid_items), len(results))

        valid_image_ids = set()
        for image_pair_id in results:
            pair = state.pair(image_pair_id)
            if pair.is_valid:
                valid_image_ids.update((pair.image_id1, pair.image_id2))
        for image_id in sorted(valid_image_ids):
            image = state.image(image_id)
            image.depth_values = np.asarray(image.depth_values, dtype=np.float64)
            image.depth_stddevs = np.asarray(image.depth_stddevs) * self.options.depth_stddev_multiplier
            state.update_image(image)

        filter_operations = (
            (
                native.score_image_pair_inliers,
                (native_inlier_thresholds, True, state.native_problem),
            ),
            (
                native.filter_pairs_by_inlier_count,
                (int(inlier_thresholds.min_inlier_num), state.native_problem),
            ),
            (
                native.filter_pairs_by_inlier_ratio,
                (inlier_thresholds.min_inlier_ratio, state.native_problem),
            ),
        )
        for operation, args in filter_operations:
            operation(*args)
            current = {pair_id for pair_id in consecutive_pair_ids if not state.pair(pair_id).is_valid}
            newly_filtered = len(current - filtered_consecutive_pairs)
            if newly_filtered:
                logger.warning(
                    "%d consecutive pairs were filtered out, continuing...",
                    newly_filtered,
                )
            filtered_consecutive_pairs = current

        summary = (
            relative_pose_summary(
                state,
                state.image_records(),
                filtered_consecutive_pairs,
                results,
                None,
            )
            if write_replay
            else None
        )
        if write_replay:
            self.replay.write_pickle(
                "relative_pose",
                "state.pkl",
                capture_relative_pose_state(
                    state,
                    state.image_records(),
                    consecutive_pair_ids,
                    filtered_consecutive_pairs,
                    results,
                    None,
                ),
            )
            self.replay.write_json("relative_pose", "summary.json", summary)

        return RelativePoseResult(
            inlier_thresholds=inlier_thresholds,
            filtered_consecutive_pairs=filtered_consecutive_pairs,
        )
