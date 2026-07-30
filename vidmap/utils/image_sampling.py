import numpy as np
import torch
from torch.nn import functional as F


def sample_at_keypoints(keypoints, image_data, x_scale, y_scale, mode="bilinear"):
    """Sample a dense image-sized array at keypoint coordinates."""
    sampling_grid = torch.tensor(keypoints * np.array([x_scale, y_scale]))
    if len(sampling_grid.shape) == 1:
        sampling_grid = sampling_grid[None]
    height, width = image_data.shape[:2]
    image_tensor = torch.tensor(image_data)[None, None]
    sampling_grid[:, 0] = (sampling_grid[:, 0] / (width - 1)) * 2 - 1
    sampling_grid[:, 1] = (sampling_grid[:, 1] / (height - 1)) * 2 - 1
    sampling_grid = sampling_grid[None, None].permute(0, 1, 2, 3)
    return F.grid_sample(image_tensor, sampling_grid, mode=mode, padding_mode="zeros", align_corners=True)[
        0, 0, 0
    ].numpy()
