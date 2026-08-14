"""Covariance-based refinement of continuing sparse tracks."""

import numpy as np

from vidmap.frontend.tracking.kernels import COV_SCALE
from vidmap.frontend.tracking.sampling import nn_sample_2d


def _cov_trace(cov):
    return COV_SCALE * (cov[..., 0, 0] + cov[..., 1, 1])


def refine_long_tracks_adaptive(
    kps1_np,
    track_length,
    tracks_mask,
    previous_mask,
    covar1,
    dense_hops,
    *,
    history,
    options,
    scheduled_hops,
    scale_ratio,
    current_size,
):
    """
    Select lower-covariance multi-hop paths for continuing sparse tracks.

    Candidate paths use earlier anchor frames and fall back to the sequential
    prediction unless the best covariance path also agrees geometrically.
    """
    track_length_masked = track_length[tracks_mask]
    prev_tracks_mask = previous_mask.copy()
    prev_tracks_mask[previous_mask] = tracks_mask[: previous_mask.sum()]

    n_prev = prev_tracks_mask.sum()
    tl_prev = track_length_masked[:n_prev]

    continuing = tl_prev > 0
    if not continuing.any():
        return kps1_np

    cont_indices = np.flatnonzero(continuing)
    track_indices = np.where(prev_tracks_mask)[0][cont_indices]

    max_hop = max(scheduled_hops)
    min_conf = float(options.min_conf)
    _s2d = nn_sample_2d

    seq_pred = kps1_np[cont_indices].copy()
    covar1_masked = covar1[tracks_mask][:n_prev][cont_indices]
    seq_cov_trace = _cov_trace(covar1_masked) * float(options.lt_cov_scale)

    best_cov = seq_cov_trace.copy()
    best_pred = seq_pred.copy()

    max_n_per_track = np.minimum(tl_prev[cont_indices].astype(int) + 1, max_hop)

    track_buf_len = history.track.shape[0]
    for n in sorted(scheduled_hops):
        if n == 1:
            continue
        anchor_idx = -(n - 1)
        if (n - 1) > track_buf_len:
            break

        eligible = max_n_per_track >= n
        if not eligible.any():
            break

        elig_idx = np.flatnonzero(eligible)
        tidx = track_indices[elig_idx]

        anchor_pos = history.track[anchor_idx, tidx]

        valid = (
            (anchor_pos[:, 0] >= 0)
            & (anchor_pos[:, 1] >= 0)
            & (anchor_pos[:, 0] < current_size[0])
            & (anchor_pos[:, 1] < current_size[1])
        )

        acc_cov = history.cov[anchor_idx, tidx]
        valid &= acc_cov[:, 0, 0] >= 0

        if not valid.any():
            continue

        v_idx = np.flatnonzero(valid)
        v_elig = elig_idx[v_idx]
        v_anchor = anchor_pos[v_idx]
        ax = v_anchor[:, 0].astype(np.float64)
        ay = v_anchor[:, 1].astype(np.float64)

        v_acc_cov = acc_cov[v_idx]
        acc_trace = _cov_trace(v_acc_cov)

        direct_cert = _s2d(dense_hops.certainty[-n], ax, ay)
        cert_ok = direct_cert >= min_conf
        if not cert_ok.any():
            continue

        c_idx = np.flatnonzero(cert_ok)
        c_elig = v_elig[c_idx]
        c_ax = ax[c_idx]
        c_ay = ay[c_idx]
        c_acc_trace = acc_trace[c_idx]

        direct_pred = _s2d(dense_hops.matches[-n], c_ax, c_ay) - 0.5
        direct_cov = _s2d(dense_hops.covariance[-n], c_ax, c_ay)
        direct_trace = _cov_trace(direct_cov)

        total_cov = c_acc_trace + direct_trace

        improves = total_cov < best_cov[c_elig]
        if not improves.any():
            continue

        imp_idx = np.flatnonzero(improves)
        best_cov[c_elig[imp_idx]] = total_cov[imp_idx]
        best_pred[c_elig[imp_idx]] = direct_pred[imp_idx]

    dist = np.linalg.norm(best_pred - seq_pred, axis=-1)
    thresh = (best_cov**0.5 + 1) * scale_ratio[0]
    accept = dist < thresh

    kps1_np[cont_indices[accept]] = best_pred[accept]
    return kps1_np
