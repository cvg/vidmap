"""Rotation cycle consistency filtering for relative poses."""

from __future__ import annotations

import logging

import numpy as np
import pycolmap

from vidmap.mapper.native.state import SolveState

logger = logging.getLogger(__name__)


def filter_pairs_by_cycle_consistency(
    state: SolveState,
    *,
    min_triangles: int = 3,
    max_median_cycle_error_deg: float = 20.0,
    max_inconsistent_ratio: float = 0.6,
    triangle_error_threshold_deg: float = 15.0,
) -> int:
    """Filter image pairs whose relative rotation is inconsistent with 3-cycles in the view graph.

    For each valid pair (i, j) with an estimated relative pose, we examine all mutual
    neighbors k such that (i, k) and (k, j) are also valid pairs with relative poses
    (forming a triangle i-k-j). For each triangle, the cycle error is the angular
    distance between R_ji and R_jk * R_ki.
    If the pair has at least min_triangles triangles and the median cycle error exceeds
    max_median_cycle_error_deg (and the fraction of inconsistent triangles exceeds
    max_inconsistent_ratio), the pair is deemed an outlier and marked invalid.

    Returns the number of pairs filtered.
    """
    adj: dict[int, set[int]] = {}
    rot_lookup: dict[tuple[int, int], pycolmap.Rotation3d] = {}

    def has_relative_pose(pair_id: int) -> bool:
        return state.pose_graph.is_valid(pair_id) and state.pair_data(pair_id).has_relative_pose

    for pair_id in state.pair_order:
        if not has_relative_pose(pair_id):
            continue
        id1, id2 = pycolmap.pair_id_to_image_pair(pair_id)
        adj.setdefault(id1, set()).add(id2)
        adj.setdefault(id2, set()).add(id1)
        rotation = state.pose_graph.edges[pair_id].cam2_from_cam1.rotation
        rot_lookup[(id1, id2)] = rotation
        rot_lookup[(id2, id1)] = rotation.inverse()

    flagged_pairs = []
    for pair_id in state.pair_order:
        if not has_relative_pose(pair_id):
            continue
        id1, id2 = pycolmap.pair_id_to_image_pair(pair_id)
        common = adj.get(id1, set()) & adj.get(id2, set())
        if len(common) < min_triangles:
            continue

        R_21 = rot_lookup[(id1, id2)]
        errors = []
        for k in common:
            R_k1 = rot_lookup[(id1, k)]
            R_2k = rot_lookup[(k, id2)]
            R_pred = R_2k * R_k1
            err_deg = np.rad2deg((R_pred.inverse() * R_21).angle())
            errors.append(err_deg)

        median_error = float(np.median(errors))
        inconsistent_ratio = float(np.mean(np.array(errors) > triangle_error_threshold_deg))

        if median_error > max_median_cycle_error_deg and inconsistent_ratio > max_inconsistent_ratio:
            flagged_pairs.append((pair_id, id1, id2, len(common), median_error, inconsistent_ratio))

    for pair_id, id1, id2, n_tri, med_err, inc_ratio in flagged_pairs:
        state.pose_graph.set_invalid_edge(pair_id)
        logger.info(
            "Filtered pair %d (%d <-> %d) by cycle consistency: %d triangles, median_error=%.1f deg, inconsistent=%.1f%%",
            pair_id,
            id1,
            id2,
            n_tri,
            med_err,
            inc_ratio * 100.0,
        )

    if flagged_pairs:
        logger.warning(
            "Cycle consistency filtering invalidated %d inconsistent pairs",
            len(flagged_pairs),
        )

    return len(flagged_pairs)
