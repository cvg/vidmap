"""Temporal filtering and smoothing for camera intrinsic sequences."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def smooth_temporal_focals(
    focals: Sequence[float] | np.ndarray,
    window_size: int = 5,
    gaussian_sigma: float = 1.0,
) -> np.ndarray:
    """Apply outlier rejection (median filter) and temporal Gaussian smoothing to focal lengths.

    Filters in log-space so relative zoom ratios are treated symmetrically.
    """
    arr = np.asarray(focals, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"Focal lengths must be a 1D sequence, got shape {arr.shape}")
    if len(arr) <= 2:
        return arr.copy()
    if np.any(arr <= 0) or not np.all(np.isfinite(arr)):
        raise ValueError("Focal lengths must be positive and finite")

    log_f = np.log(arr)
    n = len(log_f)

    # 1. 1D median filter to remove neural prediction spikes.
    w = min(window_size, n if n % 2 == 1 else n - 1)
    w = max(1, w)
    if w > 1:
        pad = w // 2
        padded = np.pad(log_f, pad, mode="edge")
        med = np.array([np.median(padded[i : i + w]) for i in range(n)], dtype=np.float64)
    else:
        med = log_f.copy()

    # 2. 1D Gaussian smoothing to ensure C^inf focal curve.
    if gaussian_sigma > 0 and n > 2:
        radius = int(np.ceil(3 * gaussian_sigma))
        radius = min(radius, n)
        x = np.arange(-radius, radius + 1, dtype=np.float64)
        kernel = np.exp(-0.5 * (x / gaussian_sigma) ** 2)
        kernel /= kernel.sum()
        padded_med = np.pad(med, radius, mode="edge")
        smooth_log = np.convolve(padded_med, kernel, mode="valid")
    else:
        smooth_log = med

    return np.exp(smooth_log)
