"""Depth-consistency classification and graph mutation."""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import TypedDict

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from tqdm import tqdm

from vidmap.mapper.native.records import pose_record_to_pycolmap
from vidmap.mapper.native.state import SolveState
from vidmap.mapper.options.positioning import DepthConsistencyOptions
from vidmap.utils.logging import progress_bars_enabled
from vidmap.utils.profiling import record_timing, sync_time

logger = logging.getLogger(__name__)

Keypoint = tuple[int, int]


class DepthConsistencyResult(TypedDict):
    is_outlier: np.ndarray | None
    tainted_keypoints: set[Keypoint]


def classify_depth_consistency(
    solve_state: SolveState,
    cameras,
    consecutive_pair_ids,
    depth_ratio_threshold=1.3,
) -> dict[int, DepthConsistencyResult]:
    results = {}
    camera_matrices = {camera_id: camera.calibration_matrix() for camera_id, camera in cameras.items()}
    image_data = {
        image_id: {
            "camera_id": image.camera_id,
            "features": np.asarray(image.keypoints),
            "depth_priors": np.asarray(image.depth_values),
            "depth_prior_validity": np.asarray(image.depth_validity, dtype=bool),
        }
        for image_id, image in solve_state.image_records().items()
    }

    pair_data = {}
    for pair_id in consecutive_pair_ids:
        image_pair = solve_state.pair(pair_id)
        cam2_from_cam1 = pose_record_to_pycolmap(image_pair.geometry.cam2_from_cam1)
        pair_data[pair_id] = {
            "id1": image_pair.image_id1,
            "id2": image_pair.image_id2,
            "rotation": np.asarray(cam2_from_cam1.rotation.matrix()),
            "translation": np.asarray(cam2_from_cam1.translation),
            "matches": np.asarray(image_pair.all_matches),
        }

    for pair_id in tqdm(
        consecutive_pair_ids,
        desc="Depth consistency check",
        disable=not progress_bars_enabled(),
    ):
        pair = pair_data[pair_id]
        image_id1 = pair["id1"]
        image_id2 = pair["id2"]
        data1 = image_data[image_id1]
        data2 = image_data[image_id2]
        calibration1 = camera_matrices[data1["camera_id"]]
        calibration2 = camera_matrices[data2["camera_id"]]
        keypoints1 = data1["features"]
        keypoints2 = data2["features"]
        rotation = pair["rotation"]
        center2 = -rotation.T @ pair["translation"]
        matches = pair["matches"]

        pair_result: DepthConsistencyResult = {
            "is_outlier": None,
            "tainted_keypoints": set(),
        }

        if len(matches) > 0:
            match_idx1 = matches[:, 0]
            match_idx2 = matches[:, 1]
            matched_keypoints1 = keypoints1[match_idx1][:, :2]
            matched_keypoints2 = keypoints2[match_idx2][:, :2]
            matched_depths1 = data1["depth_priors"][match_idx1]
            matched_depths2 = data2["depth_priors"][match_idx2]
            matched_valid1 = data1["depth_prior_validity"][match_idx1]
            matched_valid2 = data2["depth_prior_validity"][match_idx2]

            valid_for_ratio = matched_valid1 & matched_valid2 & (matched_depths1 > 0) & (matched_depths2 > 0)
            inverse_calibration2 = np.linalg.inv(calibration2)
            homogeneous_keypoints2 = np.hstack(
                [
                    matched_keypoints2[valid_for_ratio],
                    np.ones((valid_for_ratio.sum(), 1)),
                ]
            )
            rays2_unscaled = (inverse_calibration2 @ homogeneous_keypoints2.T).T
            points2_unscaled = rays2_unscaled * matched_depths2[valid_for_ratio, None]
            rotated_depths2 = (rotation.T @ points2_unscaled.T).T[:, 2]
            valid_depths1 = matched_depths1[valid_for_ratio]
            valid_reprojection = (rotated_depths2 > 0) & (valid_depths1 > 0)

            if valid_reprojection.sum() > 0:
                scale_ratios = (valid_depths1[valid_reprojection] - center2[2]) / rotated_depths2[valid_reprojection]
                scale = np.median(scale_ratios)
            else:
                scale = 1.0

            inverse_calibration1 = np.linalg.inv(calibration1)
            homogeneous_keypoints1 = np.hstack([matched_keypoints1, np.ones((len(matched_keypoints1), 1))])
            rays1 = (inverse_calibration1 @ homogeneous_keypoints1.T).T
            points_world1 = rays1 * matched_depths1[:, None]

            homogeneous_keypoints2 = np.hstack([matched_keypoints2, np.ones((len(matched_keypoints2), 1))])
            rays2 = (inverse_calibration2 @ homogeneous_keypoints2.T).T
            points_camera2 = rays2 * matched_depths2[:, None] * scale
            points_world2 = (rotation.T @ points_camera2.T).T + center2

            expected_depths1 = points_world2[:, 2]
            depth_ratio = expected_depths1 / (matched_depths1 + 1e-8)
            valid_both = matched_valid1 & matched_valid2 & (matched_depths1 > 0) & (expected_depths1 > 0)
            valid_both &= (np.abs(points_world1) < 100).all(axis=1) & (np.abs(points_world2) < 100).all(axis=1)
            is_inlier = (
                valid_both & (depth_ratio < depth_ratio_threshold) & (depth_ratio > 1.0 / depth_ratio_threshold)
            )
            is_outlier = valid_both & ~is_inlier

            pair_result["is_outlier"] = is_outlier

            for index in np.where(is_outlier)[0]:
                pair_result["tainted_keypoints"].add((image_id1, int(match_idx1[index])))
                pair_result["tainted_keypoints"].add((image_id2, int(match_idx2[index])))

        results[pair_id] = pair_result

    return results


@dataclass(kw_only=True)
class DepthConsistencyFilter:
    solve_state: SolveState
    options: DepthConsistencyOptions
    consecutive_pair_ids: list[int]
    sequence_id_to_index: dict[int, int]

    def filter(self) -> bool:
        state = self.solve_state
        seq_id_to_idx = self.sequence_id_to_index
        tainted_kp_to_boundary = {}
        boundary_depth_outliers_marked = False

        initial_tainted_keypoints = set()
        if self.options.enabled:
            depth_consistency_results = classify_depth_consistency(
                solve_state=state,
                cameras=state.reconstruction.cameras,
                consecutive_pair_ids=self.consecutive_pair_ids,
                depth_ratio_threshold=self.options.depth_ratio_threshold,
            )

            for pid, result in depth_consistency_results.items():
                initial_tainted_keypoints.update(result["tainted_keypoints"])

                if self.options.depth_outlier_propagation == "boundary" and result["tainted_keypoints"]:
                    pair = state.pair(pid)
                    if pair.image_id1 in seq_id_to_idx and pair.image_id2 in seq_id_to_idx:
                        idx1 = seq_id_to_idx[pair.image_id1]
                        idx2 = seq_id_to_idx[pair.image_id2]
                        boundary = min(idx1, idx2)
                        for kp in result["tainted_keypoints"]:
                            tainted_kp_to_boundary[kp] = boundary

                if result["is_outlier"] is not None:
                    pair = state.pair(pid)
                    are_lc = np.asarray(pair.are_loop_closure).copy()
                    outlier_indices = np.where(result["is_outlier"])[0]
                    for idx in outlier_indices:
                        if idx < len(are_lc):
                            are_lc[idx] = True
                    pair.are_loop_closure = np.asarray(are_lc, dtype=np.uint8)
                    state.update_pair(pair)

            logger.info(f"Depth consistency: {len(initial_tainted_keypoints)} tainted keypoints from depth outliers")

        boundary_lc_start = sync_time()
        if self.options.depth_outlier_propagation == "boundary" and len(tainted_kp_to_boundary) > 0:
            boundary_depth_outliers_marked = self.propagate_boundary(tainted_kp_to_boundary)
        elif len(initial_tainted_keypoints) > 0:
            self.propagate_forward(initial_tainted_keypoints)
        record_timing("boundary_lc", sync_time() - boundary_lc_start)

        return boundary_depth_outliers_marked

    def propagate_boundary(self, tainted_kp_to_boundary: dict[Keypoint, int]) -> bool:
        state = self.solve_state
        images = state.image_records()
        seq_id_to_idx = self.sequence_id_to_index

        logger.info(f"Building track membership for {len(tainted_kp_to_boundary)} tainted keypoints...")

        max_kps = max(len(img.keypoints) for img in images.values())
        kp_shift = max(max_kps + 1, 100000)

        all_enc1 = []
        all_enc2 = []
        for pair_id in tqdm(state.pair_order, disable=not progress_bars_enabled()):
            image_pair = state.pair(pair_id)
            if not image_pair.is_valid:
                continue
            inliers = np.asarray(image_pair.inlier_indices)
            if len(inliers) == 0:
                continue
            are_lc = np.asarray(image_pair.are_loop_closure, dtype=bool)
            non_lc_inliers = inliers[~are_lc[inliers]]
            if len(non_lc_inliers) == 0:
                continue
            matches = image_pair.all_matches
            id1 = image_pair.image_id1
            id2 = image_pair.image_id2
            all_enc1.append(id1 * kp_shift + matches[non_lc_inliers, 0].astype(np.int64))
            all_enc2.append(id2 * kp_shift + matches[non_lc_inliers, 1].astype(np.int64))

        if all_enc1:
            all_enc1 = np.concatenate(all_enc1)
            all_enc2 = np.concatenate(all_enc2)
        else:
            all_enc1 = np.array([], dtype=np.int64)
            all_enc2 = np.array([], dtype=np.int64)

        all_nodes = np.union1d(all_enc1, all_enc2)
        n = len(all_nodes)

        idx1 = np.searchsorted(all_nodes, all_enc1) if len(all_enc1) else np.array([], dtype=np.int64)
        idx2 = np.searchsorted(all_nodes, all_enc2) if len(all_enc2) else np.array([], dtype=np.int64)
        data = np.ones(len(idx1), dtype=np.int8)
        graph = csr_matrix((data, (idx1, idx2)), shape=(n, n))

        _, labels = connected_components(graph, directed=False)

        tainted_enc = {
            img_id * kp_shift + kp_idx: boundary for (img_id, kp_idx), boundary in tainted_kp_to_boundary.items()
        }
        tainted_component_boundary = {}
        for enc_kp, boundary in tainted_enc.items():
            pos = np.searchsorted(all_nodes, enc_kp)
            if pos >= n or all_nodes[pos] != enc_kp:
                continue
            comp = labels[pos]
            if comp not in tainted_component_boundary:
                tainted_component_boundary[comp] = boundary

        tainted_labels = np.array(list(tainted_component_boundary.keys()), dtype=np.int64)
        if len(tainted_labels):
            is_tainted = np.isin(labels, tainted_labels)
            tainted_nodes = all_nodes[is_tainted]
            tainted_comp_labels = labels[is_tainted]
            track_keypoints = {(int(k // kp_shift), int(k % kp_shift)) for k in tainted_nodes}
            kp_to_boundary = {
                (
                    int(k // kp_shift),
                    int(k % kp_shift),
                ): tainted_component_boundary[int(c)]
                for k, c in zip(tainted_nodes, tainted_comp_labels)
            }
        else:
            track_keypoints = set()
            kp_to_boundary = {}

        logger.info(f"Found {len(track_keypoints)} keypoints in bad tracks")

        tk_enc = np.array(
            sorted(img_id * kp_shift + kp_idx for img_id, kp_idx in track_keypoints),
            dtype=np.int64,
        )
        kb_enc_keys = np.array(
            [img_id * kp_shift + kp_idx for (img_id, kp_idx) in kp_to_boundary],
            dtype=np.int64,
        )
        kb_enc_vals = np.array([kp_to_boundary[k] for k in kp_to_boundary], dtype=np.int64)
        kb_order = np.argsort(kb_enc_keys)
        kb_enc_keys = kb_enc_keys[kb_order]
        kb_enc_vals = kb_enc_vals[kb_order]

        total_lc_marked = 0
        for pair_id in state.pair_order:
            image_pair = state.pair(pair_id)
            if not image_pair.is_valid:
                continue

            id1 = image_pair.image_id1
            id2 = image_pair.image_id2
            if id1 not in seq_id_to_idx or id2 not in seq_id_to_idx:
                continue
            idx1_seq = seq_id_to_idx[id1]
            idx2_seq = seq_id_to_idx[id2]

            min_seq, max_seq = min(idx1_seq, idx2_seq), max(idx1_seq, idx2_seq)

            matches = image_pair.all_matches
            are_lc = np.asarray(image_pair.are_loop_closure, dtype=bool).copy()
            inliers = np.asarray(image_pair.inlier_indices)
            if len(inliers) == 0:
                continue

            non_lc_inliers = inliers[~are_lc[inliers]]
            if len(non_lc_inliers) == 0:
                continue

            enc1 = id1 * kp_shift + matches[non_lc_inliers, 0].astype(np.int64)
            enc2 = id2 * kp_shift + matches[non_lc_inliers, 1].astype(np.int64)

            if len(tk_enc) == 0:
                continue

            pos1 = np.searchsorted(tk_enc, enc1)
            pos2 = np.searchsorted(tk_enc, enc2)
            in_track1 = (pos1 < len(tk_enc)) & (tk_enc[np.minimum(pos1, len(tk_enc) - 1)] == enc1)
            in_track2 = (pos2 < len(tk_enc)) & (tk_enc[np.minimum(pos2, len(tk_enc) - 1)] == enc2)
            in_track = in_track1 | in_track2
            if not np.any(in_track):
                continue

            candidate_idx = np.where(in_track)[0]
            candidate_enc1 = enc1[candidate_idx]
            candidate_enc2 = enc2[candidate_idx]

            bp1 = np.searchsorted(kb_enc_keys, candidate_enc1)
            bp1_valid = (bp1 < len(kb_enc_keys)) & (
                kb_enc_keys[np.minimum(bp1, len(kb_enc_keys) - 1)] == candidate_enc1
            )
            bp2 = np.searchsorted(kb_enc_keys, candidate_enc2)
            bp2_valid = (bp2 < len(kb_enc_keys)) & (
                kb_enc_keys[np.minimum(bp2, len(kb_enc_keys) - 1)] == candidate_enc2
            )

            has_boundary = bp1_valid | bp2_valid
            boundaries = np.where(bp1_valid, kb_enc_vals[np.minimum(bp1, len(kb_enc_vals) - 1)], 0)
            boundaries = np.where(
                ~bp1_valid & bp2_valid,
                kb_enc_vals[np.minimum(bp2, len(kb_enc_vals) - 1)],
                boundaries,
            )

            mark_mask = has_boundary & (boundaries >= min_seq) & (boundaries < max_seq)
            n_marked = int(np.sum(mark_mask))
            if n_marked > 0:
                are_lc[non_lc_inliers[candidate_idx[mark_mask]]] = True
                image_pair.are_loop_closure = np.asarray(are_lc, dtype=np.uint8)
                state.update_pair(image_pair)
                total_lc_marked += n_marked

        logger.info(f"Track-aware boundary LC marking: {total_lc_marked} matches marked LC")

        if self.options.mark_boundary_depth_outliers:
            depth_outliers_per_image = defaultdict(list)
            for kp in track_keypoints:
                image_id, kp_idx = kp
                if image_id not in seq_id_to_idx or kp not in kp_to_boundary:
                    continue
                kp_seq_idx = seq_id_to_idx[image_id]
                boundary = kp_to_boundary[kp]

                if kp_seq_idx > boundary:
                    depth_outliers_per_image[image_id].append(kp_idx)

            depth_outliers_marked = 0
            for image_id, kp_indices in depth_outliers_per_image.items():
                image = state.image(image_id)
                if len(image.is_depth_outlier) == 0:
                    arr = np.zeros(len(image.keypoints), dtype=bool)
                else:
                    arr = np.asarray(image.is_depth_outlier, dtype=bool).copy()

                for kp_idx in kp_indices:
                    arr[kp_idx] = True
                    depth_outliers_marked += 1

                image.is_depth_outlier = np.asarray(arr, dtype=np.uint8)
                state.update_image(image)

            logger.info(f"Marked {depth_outliers_marked} keypoints as depth outliers (after boundary)")
            return depth_outliers_marked > 0

        logger.info("Skipping depth outlier marking (mark_boundary_depth_outliers=false)")
        return False

    def propagate_forward(self, initial_tainted_keypoints: set[Keypoint]) -> None:
        state = self.solve_state
        logger.info(
            f"Propagating LC status forward from {len(initial_tainted_keypoints)} initial tainted keypoints..."
        )
        tainted_keypoints = initial_tainted_keypoints.copy()
        iteration = 0
        total_new_lc = 0

        while True:
            iteration += 1
            new_lc_count = 0
            newly_tainted = set()

            for pair_id in state.pair_order:
                image_pair = state.pair(pair_id)
                if not image_pair.is_valid:
                    continue
                matches = image_pair.all_matches
                are_lc = np.asarray(image_pair.are_loop_closure, dtype=bool).copy()
                inliers = image_pair.inlier_indices

                for idx in inliers:
                    if idx < len(are_lc) and not are_lc[idx]:
                        kp1 = (image_pair.image_id1, int(matches[idx, 0]))
                        kp2 = (image_pair.image_id2, int(matches[idx, 1]))
                        if kp1 in tainted_keypoints:
                            are_lc[idx] = True
                            new_lc_count += 1
                            newly_tainted.add(kp2)

                image_pair.are_loop_closure = np.asarray(are_lc, dtype=np.uint8)
                state.update_pair(image_pair)

            tainted_keypoints.update(newly_tainted)
            total_new_lc += new_lc_count

            logger.debug(
                "Propagation iteration %d: %d newly tainted keypoints, %d new LC matches",
                iteration,
                len(newly_tainted),
                new_lc_count,
            )

            if new_lc_count == 0:
                break

        logger.info(
            f"LC forward propagation converged after {iteration} iterations, "
            f"{total_new_lc} total matches marked LC, {len(tainted_keypoints)} total tainted keypoints"
        )
