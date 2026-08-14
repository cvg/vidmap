"""Eligibility filtering for tracks propagated from the previous frame."""

import logging

import numpy as np
import torch

from vidmap.frontend.tracking.nms import get_nms_mask
from vidmap.frontend.tracking.sampling import kde_blind, nn_sample_2d
from vidmap.utils.keypoint_scaling import scale_keypoints

logger = logging.getLogger(__name__)


def select_previous_track_mask(
    dmatches,
    dcertainties,
    *,
    previous_keypoints,
    track_length,
    history,
    options,
    scale_ratio,
    current_size,
):
    """Return the previous tracks eligible to extend into the current frame."""
    if previous_keypoints is None:
        return None

    mask_nms = get_nms_mask(
        previous_keypoints,
        track_length,
        options.nms_radius * scale_ratio[0],
    )
    scaled_prev_kps = scale_keypoints(previous_keypoints, scale_ratio)

    _s2d = nn_sample_2d
    dmatches_np = dmatches.cpu().numpy()
    dcertainties_np = dcertainties.cpu().numpy()
    warped_np = _s2d(dmatches_np, scaled_prev_kps[:, 0], scaled_prev_kps[:, 1]).astype(np.float32)

    current_width, current_height = current_size
    masked_canvas = (
        (warped_np[..., 0] >= 0)
        * (warped_np[..., 0] < (current_width - 1))
        * (warped_np[..., 1] >= 0)
        * (warped_np[..., 1] < (current_height - 1))
    )
    masked_prob = _s2d(dcertainties_np, scaled_prev_kps[:, 0], scaled_prev_kps[:, 1]) > options.min_conf
    base_mask = mask_nms & masked_canvas & masked_prob

    if base_mask.sum() >= options.tvg_min_inliers:
        import poselib

        ransac_kwargs = dict(
            max_iterations=options.tvg_max_iterations,
            max_epipolar_error=options.tvg_max_epipolar_error,
            min_iterations=50,
        )
        bundle_kwargs = {"max_iterations": 0}
        if options.tvg_min_iterations is not None:
            ransac_kwargs["min_iterations"] = options.tvg_min_iterations
        ransac_opt = poselib.RansacOptions(ransac_kwargs)
        bundle_opt = poselib.BundleOptions(bundle_kwargs)
        tvg_killed = 0

        for k in range(1, history.reach):
            survivor_idx = np.where(base_mask)[0]
            if len(survivor_idx) < options.tvg_min_inliers:
                break
            hist_pos = history.track[-k, survivor_idx]
            valid = (hist_pos[:, 0] >= 0) & (hist_pos[:, 1] >= 0)
            if valid.sum() < options.tvg_min_inliers:
                continue

            pts_src = np.ascontiguousarray(hist_pos[valid], dtype=np.float64)
            pts_dst = np.ascontiguousarray(warped_np[base_mask][valid], dtype=np.float64)
            F, info = poselib.estimate_fundamental(pts_src, pts_dst, ransac_opt, bundle_opt)
            inlier_mask = np.array(info["inliers"])

            outlier_local = np.where(valid)[0][~inlier_mask]
            base_mask[survivor_idx[outlier_local]] = False
            tvg_killed += len(outlier_local)

        if tvg_killed > 0:
            logger.debug(
                "TVG track filter removed %d tracks across %d hops",
                tvg_killed,
                history.reach - 1,
            )

    if options.density_thin_k > 0 and base_mask.sum() > 0:
        warped_survivors = warped_np[base_mask].astype(np.float64)
        warped_norm = warped_survivors.copy()
        warped_norm[:, 0] = (warped_norm[:, 0] / (current_width - 1)) * 2 - 1
        warped_norm[:, 1] = (warped_norm[:, 1] / (current_height - 1)) * 2 - 1

        density = kde_blind(
            torch.tensor(warped_norm),
            std=options.density_thin_std,
            blind_radius=0.0,
        ).numpy()
        d_ext = density - 1.0  # remove self-contribution
        p_survive = np.exp(-options.density_thin_k * d_ext)
        thin_keep = np.random.rand(len(density)) < p_survive

        # Apply thinning to base_mask
        survivor_indices = np.where(base_mask)[0]
        base_mask[survivor_indices[~thin_keep]] = False

    return base_mask
