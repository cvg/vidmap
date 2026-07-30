"""Two-stage VGC pair filtering.

Decomposes F -> E -> (R, t) in Python per pair, selects cheirality-best
via batched DLT triangulation of subsampled matches, rejects pairs with
too-forward motion (|t_z|/||t|| > max_abs_forward_translation_ratio) or too-low
median triangulation angle (< min_median_triangulation_angle_deg). Adds rejected
``pair_id``s to the returned exclusion set.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pycolmap
from tqdm import tqdm
from tqdm.contrib.concurrent import thread_map

from vidmap.mapper.native.state import SolveState
from vidmap.mapper.options.view_graph import VGCFilterOptions
from vidmap.utils.logging import progress_bars_enabled

logger = logging.getLogger(__name__)


def filter_vgc_pair(args):
    (
        pair_id,
        pair,
        features_cache,
        camera_id_cache,
        k_cache,
        w_matrix,
        subsample_size,
        min_matches,
        min_cheirality_points,
        strong_pair_match_count,
        min_angle_rad,
        max_forward_motion,
    ) = args
    matches = np.asarray(pair.all_matches)
    if len(matches) < min_matches:
        return pair_id, "few_matches", False

    feat1 = features_cache[pair.image_id1]
    feat2 = features_cache[pair.image_id2]
    K1 = k_cache[camera_id_cache[pair.image_id1]]
    K2 = k_cache[camera_id_cache[pair.image_id2]]
    F = np.asarray(pair.geometry.fundamental)
    E = K2.T @ F @ K1
    U, _S, Vt = np.linalg.svd(E)
    if np.linalg.det(U) < 0:
        U = -U
    if np.linalg.det(Vt) < 0:
        Vt = -Vt

    n = len(matches)
    if n > subsample_size:
        idx = np.linspace(0, n - 1, subsample_size, dtype=int)
        sub = matches[idx]
    else:
        sub = matches
    pts1 = feat1[sub[:, 0]]
    pts2 = feat2[sub[:, 1]]
    num_points = len(pts1)

    P1 = K1 @ np.hstack([np.eye(3), np.zeros((3, 1))])

    R_cands = [U @ w_matrix @ Vt, U @ w_matrix.T @ Vt]
    t_cands = [U[:, 2], -U[:, 2]]
    best_R, best_t, best_count, best_pts3d = None, None, -1, None
    for R_c in R_cands:
        for t_c in t_cands:
            P2 = K2 @ np.hstack([R_c, t_c.reshape(3, 1)])
            A = np.empty((num_points, 4, 4))
            A[:, 0] = pts1[:, 0:1] * P1[2] - P1[0]
            A[:, 1] = pts1[:, 1:2] * P1[2] - P1[1]
            A[:, 2] = pts2[:, 0:1] * P2[2] - P2[0]
            A[:, 3] = pts2[:, 1:2] * P2[2] - P2[1]
            _, _, Vh = np.linalg.svd(A)
            X_h = Vh[:, -1, :]
            X = X_h[:, :3] / X_h[:, 3:4]
            d1 = X[:, 2]
            d2 = (X @ R_c.T + t_c)[..., 2]
            pos = (d1 > 0) & (d2 > 0)
            positive_count = pos.sum()
            if positive_count > best_count:
                best_R, best_t, best_count, best_pts3d = (
                    R_c,
                    t_c,
                    int(positive_count),
                    X,
                )

    if best_count < min_cheirality_points:
        return pair_id, "decompose_fail", False

    R, t = best_R, best_t
    t_norm = np.linalg.norm(t)
    tz_ratio = abs(t[2]) / t_norm if t_norm > 1e-12 else 1.0

    pts3d = best_pts3d
    d1 = pts3d[:, 2]
    d2 = (pts3d @ R.T + t)[..., 2]
    valid = (d1 > 0) & (d2 > 0)
    pts_valid = pts3d[valid]
    if len(pts_valid) < min_cheirality_points:
        return pair_id, "low_tri", False
    proj_center2 = -R.T @ t
    ray1 = pts_valid
    ray2 = pts_valid - proj_center2
    n1 = np.linalg.norm(ray1, axis=1)
    n2 = np.linalg.norm(ray2, axis=1)
    good = (n1 > 1e-12) & (n2 > 1e-12)
    if good.sum() < min_cheirality_points:
        return pair_id, "low_tri", False
    cos_a = np.clip(
        np.sum(ray1[good] * ray2[good], axis=1) / (n1[good] * n2[good]),
        -1,
        1,
    )
    tri_angles = np.arccos(cos_a)
    median_angle = np.median(tri_angles)
    if tz_ratio > max_forward_motion:
        return pair_id, "forward", False
    if median_angle < min_angle_rad:
        return pair_id, "low_tri", False
    return pair_id, "kept", len(matches) >= strong_pair_match_count


@dataclass(kw_only=True)
class ViewGraphFilter:
    """Reject geometrically weak pairs before view-graph calibration."""

    solve_state: SolveState
    options: VGCFilterOptions
    calibration_enabled: bool
    initial_exclusion_ids: set[int]

    def filter(self) -> set[int]:
        cameras = self.solve_state.reconstruction.cameras
        images = self.solve_state.image_records()
        vgc_filtered_pair_ids = self.initial_exclusion_ids

        if not (self.options.enabled and self.calibration_enabled):
            return vgc_filtered_pair_ids

        _vgc_min_angle_rad = np.deg2rad(self.options.min_median_triangulation_angle_deg)
        _vgc_excluded = set()
        _vgc_forward = 0
        _vgc_low_tri = 0
        _vgc_few_matches = 0
        _vgc_decompose_fail = 0
        _vgc_strong_kept = 0
        _W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)

        _K_cache = {cid: cam.calibration_matrix() for cid, cam in cameras.items()}
        _features_cache = {image_id: np.asarray(image.keypoints) for image_id, image in images.items()}
        _camera_id_cache = {image_id: image.camera_id for image_id, image in images.items()}

        valid_configurations = {
            pycolmap.TwoViewGeometryConfiguration.CALIBRATED,
            pycolmap.TwoViewGeometryConfiguration.UNCALIBRATED,
        }
        _vgc_pairs = [
            (pid, p)
            for pid, p in self.solve_state.pair_records().items()
            if p.is_valid and p.geometry.configuration in valid_configurations
        ]
        _worker_args = [
            (
                pair_id,
                pair,
                _features_cache,
                _camera_id_cache,
                _K_cache,
                _W,
                self.options.subsample_size,
                self.options.min_matches,
                self.options.min_cheirality_points,
                self.options.strong_pair_match_count,
                _vgc_min_angle_rad,
                self.options.max_abs_forward_translation_ratio,
            )
            for pair_id, pair in _vgc_pairs
        ]
        if self.options.num_threads in (None, 1):
            _vgc_results = [
                filter_vgc_pair(args)
                for args in tqdm(_worker_args, desc="VGC pair filter", disable=not progress_bars_enabled())
            ]
        else:
            _vgc_results = thread_map(
                filter_vgc_pair,
                _worker_args,
                max_workers=self.options.num_threads,
                desc="VGC pair filter",
            )

        for pair_id, reason, strong_kept in _vgc_results:
            if reason == "few_matches":
                _vgc_few_matches += 1
                continue
            if reason == "decompose_fail":
                _vgc_decompose_fail += 1
                _vgc_excluded.add(pair_id)
                continue
            if reason == "forward":
                _vgc_excluded.add(pair_id)
                _vgc_forward += 1
                continue
            if reason == "low_tri":
                _vgc_excluded.add(pair_id)
                _vgc_low_tri += 1
                continue
            if strong_kept:
                _vgc_strong_kept += 1
        _vgc_total = len(_vgc_pairs)
        if _vgc_strong_kept < self.options.min_strong_kept_pairs:
            logger.info(
                "VGC filter has only %d strong pairs (at least %d matches; minimum %d); skipping filter",
                _vgc_strong_kept,
                self.options.strong_pair_match_count,
                self.options.min_strong_kept_pairs,
            )
            _vgc_excluded.clear()
        if _vgc_excluded:
            vgc_filtered_pair_ids = vgc_filtered_pair_ids | _vgc_excluded
        logger.info(
            "VGC filter kept %d/%d pairs (%d strong with at least %d matches); excluded %d "
            "(%d forward motion, %d triangulation angle below %.1f degrees, %d decomposition failures, "
            "%d below %d matches)",
            _vgc_total - len(_vgc_excluded),
            _vgc_total,
            _vgc_strong_kept,
            self.options.strong_pair_match_count,
            len(_vgc_excluded),
            _vgc_forward,
            _vgc_low_tri,
            self.options.min_median_triangulation_angle_deg,
            _vgc_decompose_fail,
            _vgc_few_matches,
            self.options.min_matches,
        )

        return vgc_filtered_pair_ids
