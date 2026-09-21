"""Audited FP32 bilinear RoMa correlation, without sampled-feature intermediates.

Contract: finite FP32 features, zero padding, align_corners=False, inference only.
Coordinate support/weight masks also preserve the pinned CUDA behavior for
out-of-range and non-finite warp coordinates. This is not a gradient operator.
"""

import torch
import triton
import triton.language as tl


def _validate(a, b, warp, window):
    assert a.ndim == 4 and b.shape == a.shape
    n, c, h, w = a.shape
    assert min(n, c, h, w) > 0 and c <= 256
    assert warp.shape == (n, h, w, 2)
    assert window.ndim == 3 and window.shape[0] == 1 and window.shape[2] == 2 and window.shape[1] > 0
    assert all(
        value.is_cuda and value.device == a.device and value.dtype == torch.float32 for value in (a, b, warp, window)
    )


@torch.library.custom_op("vidmap::roma_correlation", mutates_args=())
def fused_correlation(a: torch.Tensor, b: torch.Tensor, warp: torch.Tensor, window: torch.Tensor) -> torch.Tensor:
    _validate(a, b, warp, window)
    # Keep materialization inside the opaque operator, as in the audited path.
    # Exposing these copies to Inductor changes upstream layout/fusion choices.
    warp, window = warp.contiguous(), window.contiguous()
    n, c, h, w = a.shape
    k = window.shape[1]
    output = torch.empty((n, k, h, w), device=a.device, dtype=torch.float32)
    _correlation_kernel[(n * h * w, triton.cdiv(k, 8))](
        a,
        b,
        warp,
        window,
        output,
        c,
        h,
        w,
        k,
        *a.stride(),
        *b.stride(),
        triton.next_power_of_2(c),
        8,
        num_warps=4,
        enable_fp_fusion=False,
    )
    return output


@fused_correlation.register_fake
def _fake_correlation(a, b, warp, window):
    _validate(a, b, warp, window)
    return a.new_empty((a.shape[0], window.shape[1], a.shape[2], a.shape[3]))


def local_correlation(feature0, feature1, local_radius, warp, scale_factor):
    """Construct exactly the pinned RoMa window and call the fused operator."""
    h, w = feature0.shape[-2:]
    r = local_radius
    k = (2 * r + 1) ** 2
    yy, xx = torch.meshgrid(
        [
            torch.linspace(-2 * r / h, 2 * r / h, 2 * r + 1, device=feature0.device),
            torch.linspace(-2 * r / w, 2 * r / w, 2 * r + 1, device=feature0.device),
        ],
        indexing="ij",
    )
    window = torch.stack((xx, yy), dim=-1)[None].expand(1, 2 * r + 1, 2 * r + 1, 2).reshape(1, k, 2)
    return fused_correlation(feature0, feature1, warp, window)


@triton.jit
def _correlation_kernel(
    a,
    b,
    warp,
    window,
    output,
    C: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    K: tl.constexpr,
    A0: tl.constexpr,
    A1: tl.constexpr,
    A2: tl.constexpr,
    A3: tl.constexpr,
    B0: tl.constexpr,
    B1: tl.constexpr,
    B2: tl.constexpr,
    B3: tl.constexpr,
    CHANNELS: tl.constexpr,
    OFFSETS: tl.constexpr,
):
    pixel = tl.program_id(0)
    batch = pixel // (H * W)
    y = pixel % (H * W) // W
    x = pixel % W
    offsets = tl.program_id(1) * OFFSETS + tl.arange(0, OFFSETS)
    channels = tl.arange(0, CHANNELS)
    dtype: tl.constexpr = tl.float32
    gx = tl.load(warp + pixel * 2).to(dtype) + tl.load(window + offsets * 2, offsets < K, 0).to(dtype)
    gy = tl.load(warp + pixel * 2 + 1).to(dtype) + tl.load(window + offsets * 2 + 1, offsets < K, 0).to(dtype)
    sx = tl.fma(gx, W * 0.5, (W - 1) * 0.5)
    sy = tl.fma(gy, H * 0.5, (H - 1) * 0.5)
    supported = (sx > -1.0) & (sx < W) & (sy > -1.0) & (sy < H)
    sx = tl.where(supported, sx, -2.0)
    sy = tl.where(supported, sy, -2.0)
    ix = tl.floor(sx).to(tl.int32)
    iy = tl.floor(sy).to(tl.int32)
    dx = sx - ix
    dy = sy - iy
    channel_mask = channels[None, :] < C
    valid = (offsets[:, None] < K) & channel_mask
    address = b + batch * B0 + channels[None, :] * B1
    v00 = tl.load(
        address + iy[:, None] * B2 + ix[:, None] * B3,
        valid & (ix[:, None] >= 0) & (ix[:, None] < W) & (iy[:, None] >= 0) & (iy[:, None] < H),
        0,
    )
    v10 = tl.load(
        address + iy[:, None] * B2 + (ix[:, None] + 1) * B3,
        valid & (ix[:, None] + 1 >= 0) & (ix[:, None] + 1 < W) & (iy[:, None] >= 0) & (iy[:, None] < H),
        0,
    )
    v01 = tl.load(
        address + (iy[:, None] + 1) * B2 + ix[:, None] * B3,
        valid & (ix[:, None] >= 0) & (ix[:, None] < W) & (iy[:, None] + 1 >= 0) & (iy[:, None] + 1 < H),
        0,
    )
    v11 = tl.load(
        address + (iy[:, None] + 1) * B2 + (ix[:, None] + 1) * B3,
        valid & (ix[:, None] + 1 >= 0) & (ix[:, None] + 1 < W) & (iy[:, None] + 1 >= 0) & (iy[:, None] + 1 < H),
        0,
    )
    w00 = (1 - dx) * (1 - dy)
    w10 = dx * (1 - dy)
    w01 = (1 - dx) * dy
    w11 = dx * dy
    w00 = tl.where((ix >= 0) & (ix < W) & (iy >= 0) & (iy < H), w00, 0)
    w10 = tl.where((ix + 1 >= 0) & (ix + 1 < W) & (iy >= 0) & (iy < H), w10, 0)
    w01 = tl.where((ix >= 0) & (ix < W) & (iy + 1 >= 0) & (iy + 1 < H), w01, 0)
    w11 = tl.where((ix + 1 >= 0) & (ix + 1 < W) & (iy + 1 >= 0) & (iy + 1 < H), w11, 0)
    sampled = (
        v00.to(dtype) * w00[:, None]
        + v10.to(dtype) * w10[:, None]
        + v01.to(dtype) * w01[:, None]
        + v11.to(dtype) * w11[:, None]
    )
    source = tl.load(a + batch * A0 + channels * A1 + y * A2 + x * A3, channels < C, 0).to(dtype)
    scaled_source = source * tl.full((), C ** (-0.5), dtype)
    values = tl.sum(sampled * scaled_source[None, :], axis=1)
    tl.store(output + batch * K * H * W + offsets * H * W + y * W + x, values, offsets < K)
