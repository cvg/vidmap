"""Bilinear/nearest-neighbor sampling and density helpers for sparse track frontend.

Pure numerical helpers; no on-disk IO, no model dispatch.
"""

import numpy as np
import torch


def bilinear_sample_2d(arr, x, y):
    """Bilinear interpolation on a numpy array of shape (H, W, ...).

    At integer coordinates, returns the exact cell value (no interpolation error).
    Coordinates outside [0, W-1] x [0, H-1] are clamped to the border.

    Args:
        arr: numpy array of shape (H, W) or (H, W, C) or (H, W, C1, C2), etc.
        x: (N,) float array of sub-pixel x coordinates (column index).
        y: (N,) float array of sub-pixel y coordinates (row index).

    Returns:
        Interpolated values of shape (N,) or (N, C) or (N, C1, C2), etc.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    H, W = arr.shape[:2]

    x = np.clip(x, 0, W - 1)
    y = np.clip(y, 0, H - 1)

    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    x1 = np.minimum(x0 + 1, W - 1)
    y1 = np.minimum(y0 + 1, H - 1)

    wx = x - x0
    wy = y - y0

    extra_dims = arr.ndim - 2
    for _ in range(extra_dims):
        wx = wx[..., np.newaxis]
        wy = wy[..., np.newaxis]

    v00 = arr[y0, x0]
    v01 = arr[y0, x1]
    v10 = arr[y1, x0]
    v11 = arr[y1, x1]

    return v00 * (1 - wx) * (1 - wy) + v01 * wx * (1 - wy) + v10 * (1 - wx) * wy + v11 * wx * wy


def bilinear_sample_3d(arr, t, x, y):
    """Bilinear interpolation in spatial dims of a temporal array.

    Integer temporal index `t` (supports negative indexing), bilinear in spatial.
    At integer spatial coordinates, returns the exact cell value.

    Args:
        arr: (T, H, W, ...) array — arbitrary trailing dims
        t: (N,) int array — temporal index (supports negative indexing)
        x: (N,) float array — spatial x coordinate (W dimension)
        y: (N,) float array — spatial y coordinate (H dimension)

    Returns:
        (N, ...) interpolated values
    """
    t = np.asarray(t, dtype=int)
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    H, W = arr.shape[1], arr.shape[2]
    extra_shape = arr.shape[3:]

    x = np.clip(x, 0, W - 1)
    y = np.clip(y, 0, H - 1)

    x0 = np.floor(x).astype(int)
    y0 = np.floor(y).astype(int)
    x1 = np.minimum(x0 + 1, W - 1)
    y1 = np.minimum(y0 + 1, H - 1)

    wx = x - x0
    wy = y - y0

    for _ in extra_shape:
        wx = wx[..., np.newaxis]
        wy = wy[..., np.newaxis]

    v00 = arr[t, y0, x0]
    v01 = arr[t, y0, x1]
    v10 = arr[t, y1, x0]
    v11 = arr[t, y1, x1]

    return v00 * (1 - wx) * (1 - wy) + v01 * wx * (1 - wy) + v10 * (1 - wx) * wy + v11 * wx * wy


def nn_sample_2d(arr, x, y):
    """Nearest-neighbor sampling on (H, W, ...) array. Same API as bilinear_sample_2d."""
    xi = np.clip((np.asarray(x) + 0.5).astype(np.int64), 0, arr.shape[1] - 1)
    yi = np.clip((np.asarray(y) + 0.5).astype(np.int64), 0, arr.shape[0] - 1)
    return arr[yi, xi]


def nn_sample_3d(arr, t, x, y):
    """Nearest-neighbor sampling on (T, H, W, ...) array. Same API as bilinear_sample_3d."""
    xi = np.clip((np.asarray(x) + 0.5).astype(np.int64), 0, arr.shape[2] - 1)
    yi = np.clip((np.asarray(y) + 0.5).astype(np.int64), 0, arr.shape[1] - 1)
    return arr[np.asarray(t, dtype=int), yi, xi]


def kde_blind(query, neighbors=None, std=0.1, blind_radius=0.0):
    if neighbors is None:
        neighbors = query
    dist_sq = torch.cdist(query, neighbors) ** 2
    weights = torch.exp(-dist_sq / (2 * std**2))

    if neighbors is query:
        dist = torch.sqrt(dist_sq)
        is_duplicate = (dist < blind_radius) & (dist > 1e-7)
        weights[is_duplicate] = 0.0

    return weights.sum(dim=-1)
