"""Low-level track-building helpers for sparse track frontend.

Lower-level numerical helpers used by sparse track state and propagation.
"""

from typing import NamedTuple

import numpy as np
import torch
import torch.nn.functional as F

from vidmap.frontend.correspondences import assign_keypoints
from vidmap.frontend.tracking.nms import prop_tracker_nms
from vidmap.frontend.tracking.sampling import bilinear_sample_2d, nn_sample_2d
from vidmap.utils.keypoint_scaling import scale_keypoints, unscale_keypoints

# Covariance scaling factor (converts from pixel^2 to scaled uncertainty space)
COV_SCALE = 3**2


def _normalize_for_grid_sample(coords, width, height):
    coords_norm = coords.clone()
    coords_norm[:, 0] = (coords_norm[:, 0] / (width - 1)) * 2 - 1 if width > 1 else 0
    coords_norm[:, 1] = (coords_norm[:, 1] / (height - 1)) * 2 - 1 if height > 1 else 0
    return coords_norm


def select_lc_matches_from_dense(
    kpts0,
    kpts1,
    matches_dense,
    certainty,
    *,
    source_size,
    target_size,
    lc_match_thresh,
    max_error=4,
):
    """Assign sparse source keypoints to sparse target keypoints via a dense RoMa field."""
    kpts0_np = np.asarray(kpts0, dtype=np.float32)
    kpts1_np = np.asarray(kpts1, dtype=np.float32)
    H, W = matches_dense.shape[:2]
    W0, H0 = source_size
    W1, H1 = target_size

    source_scale = np.array([W0 / W, H0 / H], dtype=np.float32)
    kpts0_sample = scale_keypoints(kpts0_np, source_scale)

    kpts0_scaled_t = torch.as_tensor(kpts0_sample, dtype=matches_dense.dtype, device=matches_dense.device)
    grid = _normalize_for_grid_sample(kpts0_scaled_t, W, H).unsqueeze(0).unsqueeze(0)

    matches_field_t = matches_dense.permute(2, 0, 1).unsqueeze(0)
    kpts1_sampled = F.grid_sample(matches_field_t, grid, mode="bilinear", padding_mode="zeros", align_corners=True)
    kpts1_sampled = kpts1_sampled.squeeze(0).squeeze(1).permute(1, 0).cpu().numpy()

    cert_field_t = certainty.unsqueeze(0).unsqueeze(0)
    certainty_sampled = (
        F.grid_sample(
            cert_field_t,
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )
        .squeeze()
        .cpu()
        .numpy()
    )
    certainty_sampled = np.atleast_1d(certainty_sampled).astype(np.float32)

    target_scale = np.array([W1 / W, H1 / H], dtype=np.float32)
    kpts1_query = unscale_keypoints(kpts1_sampled - 0.5, target_scale)
    kpts1_reference = kpts1_np

    mkp_ids1 = assign_keypoints(kpts1_query, kpts1_reference, max_error=max_error)
    valid = certainty_sampled > lc_match_thresh
    mkp_ids1[~valid] = -1

    valid_mask = mkp_ids1 >= 0
    if valid_mask.any():
        ref_indices = mkp_ids1[valid_mask]
        query_indices = np.where(valid_mask)[0]
        certainties = certainty_sampled[valid_mask]

        matches0 = np.full(len(kpts0_np), -1, dtype=np.int32)
        matching_scores0 = np.zeros(len(kpts0_np), dtype=np.float32)
        for ref_idx in np.unique(ref_indices):
            mask = ref_indices == ref_idx
            query_candidates = query_indices[mask]
            certainty_candidates = certainties[mask]
            best_idx = query_candidates[np.argmax(certainty_candidates)]
            matches0[best_idx] = ref_idx
            matching_scores0[best_idx] = certainty_sampled[best_idx]
    else:
        matches0 = np.full(len(kpts0_np), -1, dtype=np.int32)
        matching_scores0 = np.zeros(len(kpts0_np), dtype=np.float32)

    return {
        "matches0": matches0,
        "matching_scores0": matching_scores0,
    }


def select_keypoints_from_certainty(
    certainty,
    covs,
    prev_keypoints,
    nms_radius,
    sample_thresh,
    max_kps,
    bilinear=False,
):
    """
    Select keypoints from certainty map using probabilistic density sampling.

    Uses density-aware sampling to select distinctive keypoints, prioritizing
    previous tracked keypoints while sampling new ones based on certainty.

    Args:
        certainty: (H, W) certainty map from dense matcher
        covs: (H, W, 2, 2) covariance tensors
        prev_keypoints: Previously tracked keypoints to preserve (scaled coords)
        nms_radius: Suppression radius in pixels for density estimation
        sample_thresh: Minimum certainty threshold for new keypoints
        max_kps: Maximum number of keypoints to select
    Returns:
        ref_kps_nms: Selected keypoint coordinates
        certainty_nms: Certainty scores for selected keypoints
        covs_nms: Covariances for selected keypoints
    """
    check_certainties = certainty.clone()
    check_certainties[check_certainties <= sample_thresh] = 0

    queries = None
    if prev_keypoints is not None and prev_keypoints.shape[1] > 0:
        prev_keypoints = prev_keypoints[:max_kps]
        queries = torch.from_numpy(prev_keypoints).cuda()

    ref_kps_nms = prop_tracker_nms(
        check_certainties[None],
        nms_radius,
        queries[None] if queries is not None else None,
        num_corresp=max_kps,
    ).cpu()

    if prev_keypoints is not None:
        ref_kps_nms = torch.cat(
            [torch.tensor(prev_keypoints), ref_kps_nms],
            dim=0,
        )

    _s2d = bilinear_sample_2d if bilinear else nn_sample_2d
    certainty_np = certainty.cpu().numpy()
    covs_np = covs.cpu().numpy()
    kps_np = ref_kps_nms.cpu().numpy() if torch.is_tensor(ref_kps_nms) else ref_kps_nms
    certainty_nms = torch.from_numpy(_s2d(certainty_np, kps_np[:, 0], kps_np[:, 1]).astype(np.float32))
    covs_nms = torch.from_numpy(_s2d(covs_np, kps_np[:, 0], kps_np[:, 1]).astype(np.float32))

    return SelectedKeypoints(ref_kps_nms, certainty_nms, covs_nms)


def build_tracks_from_matches(ref_kps, ref_conf, ref_covar, matches_fine, certainty_01, covar_01, bilinear=False):
    """
    Compute destination keypoints and their quality metrics (RoMAv2 version).

    For each reference keypoint, looks up the corresponding match in the dense
    match field and computes accumulated confidence/covariance.

    Args:
        ref_kps: (N, 2) reference keypoint positions
        ref_conf: (N,) accumulated confidence from previous frames
        ref_covar: (N, 2, 2) accumulated covariance
        matches_fine: (H, W, 2) dense match field (corner-based coordinates)
        certainty_01: (H, W) match certainty map
        covar_01: (H, W, 2, 2) match covariance map

    Returns:
        kps1: (N, 2) matched keypoint locations in frame i+1
        conf1: (N,) updated confidence (minimum of accumulated and current)
        covar1: (N, 2, 2) updated covariance (accumulated + current)
    """
    _s2d = bilinear_sample_2d if bilinear else nn_sample_2d
    matches_np = matches_fine.cpu().numpy() if torch.is_tensor(matches_fine) else matches_fine
    ref_kps_np = ref_kps.cpu().numpy() if torch.is_tensor(ref_kps) else ref_kps
    H_match, W_match = matches_np.shape[:2]
    kps1 = torch.from_numpy(_s2d(matches_np, ref_kps_np[:, 0], ref_kps_np[:, 1]).astype(np.float32)) - 0.5
    kps1[:, 0].clamp_(-0.5, W_match - 0.5 - 1e-4)
    kps1[:, 1].clamp_(-0.5, H_match - 0.5 - 1e-4)
    conf1 = torch.minimum(torch.tensor(ref_conf), certainty_01)
    covar1 = ref_covar + np.array(covar_01)
    return PropagatedTracks(kps1, conf1, covar1)


class SelectedKeypoints(NamedTuple):
    keypoints: torch.Tensor
    certainty: torch.Tensor
    covariance: torch.Tensor


class SparseProjection(NamedTuple):
    matches: np.ndarray
    scores: np.ndarray
    saved_keypoints: np.ndarray
    target_keypoints: np.ndarray
    survivors: np.ndarray
    saved_survivors: np.ndarray


class PropagatedTracks(NamedTuple):
    keypoints: torch.Tensor
    confidence: torch.Tensor
    covariance: np.ndarray
