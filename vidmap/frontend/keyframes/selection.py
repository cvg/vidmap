"""Keyframe scoring, propagation, and per-frame feature frontend."""

import numpy as np
import torch
import torch.nn.functional as F

from vidmap.frontend.h5_write_queue import save_features


def propagate_keypoints(kps, pair_match, pair_cert, oW, oH, certainty_threshold=0.02):
    """
    Propagate sparse keypoints through dense flow field.

    Args:
        kps: (N, 2) torch tensor or numpy array of keypoint positions [x, y]
        pair_match: (H, W, 2) dense flow field
        pair_cert: (H, W) certainty map
        oW, oH: Image dimensions
        certainty_threshold: Minimum certainty for visible keypoint

    Returns:
        new_kps: (N, 2) torch tensor of updated keypoint positions
        kps_cert: (N,) torch tensor of certainty values at keypoint locations
        visible_mask: (N,) bool tensor of visible keypoints
    """
    # Convert to torch if needed
    if not isinstance(kps, torch.Tensor):
        kps = torch.from_numpy(kps).to(pair_match.device)
    else:
        kps = kps.to(pair_match.device)

    N = kps.shape[0]
    if N == 0:
        return (
            kps,
            torch.zeros(0, device=kps.device),
            torch.zeros(0, dtype=torch.bool, device=kps.device),
        )

    # Normalize keypoint coordinates to [-1, 1] for grid_sample
    # pair_match is (H, W, 2), we need to sample at keypoint locations
    # grid_sample expects (N, H_out, W_out, 2) in normalized coords
    kps_norm = kps.clone().float()
    kps_norm[:, 0] = 2.0 * kps_norm[:, 0] / (oW - 1) - 1.0  # x
    kps_norm[:, 1] = 2.0 * kps_norm[:, 1] / (oH - 1) - 1.0  # y

    # Reshape for grid_sample: (1, N, 1, 2)
    grid = kps_norm.unsqueeze(0).unsqueeze(2)  # (1, N, 1, 2)

    # Sample flow: pair_match is (H, W, 2), need (1, 2, H, W) for grid_sample
    flow_for_sample = pair_match.permute(2, 0, 1).unsqueeze(0)  # (1, 2, H, W)
    sampled_flow = F.grid_sample(
        flow_for_sample,
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )  # (1, 2, N, 1)
    sampled_flow = sampled_flow.squeeze(0).squeeze(2).permute(1, 0)  # (N, 2)

    # Sample certainty: pair_cert is (H, W), need (1, 1, H, W)
    cert_for_sample = pair_cert.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
    sampled_cert = F.grid_sample(
        cert_for_sample,
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )  # (1, 1, N, 1)
    sampled_cert = sampled_cert.squeeze(0).squeeze(0).squeeze(1)  # (N,)

    # Update keypoint positions
    new_kps = sampled_flow

    # Check visibility: in bounds + high certainty
    in_bounds = (new_kps[:, 0] >= 0) & (new_kps[:, 0] < oW) & (new_kps[:, 1] >= 0) & (new_kps[:, 1] < oH)
    high_cert = sampled_cert > certainty_threshold
    visible_mask = in_bounds & high_cert

    return new_kps, sampled_cert, visible_mask


def _as_numpy_points(points):
    if isinstance(points, torch.Tensor):
        return points.detach().cpu().numpy()
    return np.asarray(points)


def _normalized_xy(kps, k):
    x = (kps[:, 0] - k[0, 2]) / k[0, 0]
    y = (kps[:, 1] - k[1, 2]) / k[1, 1]
    return np.stack([x, y], axis=1)


def compute_normalized_keypoint_motion_score(
    tracked_kps,
    original_kps,
    visible_mask,
    source_K,
    target_K,
    max_normalized_keypoint_drift=0.1,
):
    """Compute sparse motion score from calibrated normalized-camera drift."""
    original = _as_numpy_points(original_kps).astype(np.float64, copy=False)
    tracked = _as_numpy_points(tracked_kps).astype(np.float64, copy=False)
    visible = np.asarray(visible_mask, dtype=bool).reshape(-1)
    n = min(len(original), len(tracked), len(visible))
    if n == 0:
        return 1.0

    original = original[:n]
    tracked = tracked[:n]
    visible = visible[:n]
    num_visible = int(np.sum(visible))
    if num_visible == 0:
        return 1.0

    if source_K is None or target_K is None:
        return 1.0

    xy0 = _normalized_xy(original, np.asarray(source_K, dtype=np.float64))
    xy1 = _normalized_xy(tracked, np.asarray(target_K, dtype=np.float64))
    drift = np.linalg.norm(xy1 - xy0, axis=1)
    good_mask = visible & (drift <= float(max_normalized_keypoint_drift))
    frac_good = float(np.mean(good_mask))
    return 1.0 - frac_good


def compute_aliked_features_for_frame(scene_parser, output_path, image_name, aliked_model, salient_options):
    """Compute and persist ALIKED keypoints for one streaming frame."""
    from vidmap.frontend.image_dataset import resize_image
    from vidmap.utils.io import read_image

    if output_path is None:
        return

    image = read_image(scene_parser.rgb_dir / image_name).astype(np.float32)
    size = image.shape[:2][::-1]
    original_size = np.array(size)
    max_dim = max(size)
    preprocessing_conf = salient_options.preprocessing
    should_resize = preprocessing_conf.resize_max and (
        preprocessing_conf.resize_force or max_dim > preprocessing_conf.resize_max
    )
    if should_resize:
        scale = preprocessing_conf.resize_max / max_dim
        size_new = tuple(int(round(value * scale)) for value in size)
        image = resize_image(image, size_new, preprocessing_conf.interpolation)

    image = image[None] if preprocessing_conf.grayscale else image.transpose((2, 0, 1))
    image_tensor = torch.from_numpy(image / 255.0).unsqueeze(0).cuda()
    with torch.no_grad():
        keypoints = aliked_model({"image": image_tensor})["keypoints"][0].cpu().numpy()

    resized_size = np.array(image_tensor.shape[-2:][::-1])
    scales = (original_size / resized_size).astype(np.float32)
    keypoints = (keypoints + 0.5) * scales[None] - 0.5
    save_features(
        {"keypoints": keypoints, "image_size": original_size},
        str(output_path),
        image_name,
    )
