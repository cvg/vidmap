"""NMS / spatial-suppression helpers for sparse track frontend."""

import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial import cKDTree

from vidmap.frontend.tracking.sampling import kde_blind
from vidmap.utils.keypoint_scaling import scale_keypoints


def get_nms_mask(kp, scores, radius):
    """
    Apply greedy Non-Maximum Suppression to filter keypoints.

    Uses scipy's cKDTree for efficient spatial queries. Keypoints are processed
    in descending score order, suppressing neighbors within the given radius.

    Args:
        kp: (N, 2) keypoint coordinates
        scores: (N,) scores for each keypoint (higher = better)
        radius: Suppression radius in pixels

    Returns:
        Boolean mask (N,) indicating which keypoints to keep
    """
    idx = np.argsort(-scores)
    tree = cKDTree(kp[idx])
    adj = tree.query_ball_tree(tree, radius)

    keep_indices = np.ones(len(kp), bool)
    for i in range(len(idx)):
        if keep_indices[i]:
            for neighbor in adj[i]:
                if neighbor > i:
                    keep_indices[neighbor] = False

    # Map the sorted mask back to the original order
    mask = np.zeros(len(kp), bool)
    mask[idx] = keep_indices
    return mask


def nms_on_coordinates(pixel_coords, scores, nms_radius):
    """
    Removes points that have a higher-scoring neighbor within 'nms_radius'.

    Args:
        pixel_coords: (N, 2) tensor of (x, y) coordinates
        scores: (N,) tensor of confidence scores
        nms_radius: float, radius in pixels

    Returns:
        Filtered pixel_coords and scores
    """
    if len(pixel_coords) == 0:
        return pixel_coords, scores

    # Sort by score (highest to lowest)
    sort_idx = torch.argsort(scores, descending=True)
    sorted_coords = pixel_coords[sort_idx]

    # Compute pairwise distances
    dist = torch.cdist(sorted_coords.float(), sorted_coords.float())

    # Create suppression matrix
    is_close = dist < nms_radius
    triu_mask = torch.triu(torch.ones_like(is_close), diagonal=1).bool()
    suppression_mask = is_close & triu_mask

    # Determine survivors
    is_suppressed = suppression_mask.any(dim=0)
    kept_indices = sort_idx[~is_suppressed]

    return pixel_coords[kept_indices], scores[kept_indices]


# Blind density estimation
def prop_tracker_nms(
    scores,
    nms_radius: int,
    prev_keypoints=None,
    scales=None,
    num_corresp=2000,
):
    """
    Probabilistic NMS with blind density sampling (RoMAv2 style).

    Ensures existing points are respected and new points are evenly distributed
    by combining strict suppression around previous keypoints with density-based
    sampling for new keypoints.

    Args:
        scores: (1, H, W) confidence/certainty map
        nms_radius: Suppression radius in pixels
        prev_keypoints: (1, N, 2) previous keypoints to preserve
        scales: Scale factors if prev_keypoints need rescaling
        num_corresp: Target number of keypoints to return
    Returns:
        (N, 2) tensor of selected keypoint coordinates
    """
    if nms_radius < 0:
        raise ValueError(f"nms_radius must be nonnegative, got {nms_radius}")

    # Clone scores to avoid modifying input
    scores = scores.clone()

    H_A, W_A = scores.shape[1], scores.shape[2]
    device = scores.device

    # Handle previous keypoints
    if prev_keypoints is not None:
        kps_prev = prev_keypoints[0].clone()

        if scales is not None:
            kps_prev = scale_keypoints(kps_prev, scales)

        kps_prev_rounded = kps_prev.round().long()
        kps_prev_rounded[:, 1] = kps_prev_rounded[:, 1].clamp(0, H_A - 1)
        kps_prev_rounded[:, 0] = kps_prev_rounded[:, 0].clamp(0, W_A - 1)

        # Suppress regions around previous keypoints
        mask = torch.zeros((1, 1, H_A, W_A), device=device, dtype=torch.float32)
        mask[:, :, kps_prev_rounded[:, 1], kps_prev_rounded[:, 0]] = 1.0
        dilated_mask = F.max_pool2d(mask, kernel_size=2 * nms_radius + 1, stride=1, padding=nms_radius)[0]
        scores = scores * (1 - dilated_mask)

        # Normalize for density estimation
        priors = kps_prev.float()
        priors[:, 0] = (priors[:, 0] / (W_A - 1)) * 2 - 1
        priors[:, 1] = (priors[:, 1] / (H_A - 1)) * 2 - 1
    else:
        kps_prev = None
        kps_prev_rounded = None
        priors = None

    # Generate grid in normalized coordinates
    scores = scores[0]  # Remove batch dim

    y_grid, x_grid = torch.meshgrid(
        torch.linspace(-1, 1, H_A, device=device), torch.linspace(-1, 1, W_A, device=device), indexing="ij"
    )
    grid = torch.stack((x_grid, y_grid), dim=-1)
    matches = grid.reshape(-1, 2)
    confidence = scores.reshape(-1)

    # Boundary masking
    confidence_2d = confidence.view(H_A, W_A)
    confidence_2d[0, :] = 0
    confidence_2d[-1, :] = 0
    confidence_2d[:, 0] = 0
    confidence_2d[:, -1] = 0
    confidence = confidence_2d.reshape(-1)

    # Expansion sampling
    expansion_factor = 4
    total_expansion = min(expansion_factor * num_corresp, (confidence > 0).sum().item())

    if total_expansion > 0:
        corresp_inds = torch.multinomial(confidence, total_expansion, replacement=False)
        sampled_matches = matches[corresp_inds]
        sampled_confidence = confidence[corresp_inds]
    else:
        # No valid confidence values - return empty results
        sampled_matches = torch.empty((0, 2), device=device, dtype=torch.float32)
        sampled_confidence = torch.empty((0,), device=device, dtype=torch.float32)

    norm_blind_radius = nms_radius * (2.0 / max(H_A, W_A))
    density = kde_blind(sampled_matches, blind_radius=norm_blind_radius)

    if priors is not None:
        density += kde_blind(sampled_matches, neighbors=priors, blind_radius=0.0)

    # Balanced resampling
    p = 1 / (density + 1)

    base_samples = 20000.0
    base_threshold = 10.0
    density_threshold = (total_expansion / base_samples) * base_threshold
    p[density < max(1.0, density_threshold)] = 1e-7

    # Calculate remaining quota
    if kps_prev is not None:
        samples_needed = max(0, num_corresp - len(kps_prev))
    else:
        samples_needed = num_corresp

    num_samples = min(samples_needed, len(sampled_confidence))
    if num_samples > 0 and len(sampled_confidence) > 0:
        balanced_samples = torch.multinomial(
            p,
            num_samples=num_samples,
            replacement=False,
        )
        # Final output & NMS
        final_coords = sampled_matches[balanced_samples]
        final_scores = sampled_confidence[balanced_samples]
    else:
        # No samples needed or no valid confidence - return empty results
        final_coords = torch.empty((0, 2), device=device, dtype=torch.float32)
        final_scores = torch.empty((0,), device=device, dtype=torch.float32)

    # Convert back to pixel coordinates
    pixel_coords = (final_coords + 1) * 0.5 * torch.tensor([W_A - 1, H_A - 1], device=device)
    pixel_coords = pixel_coords.round()

    # Final strict NMS
    pixel_coords, _ = nms_on_coordinates(pixel_coords, final_scores, nms_radius)

    return pixel_coords
