def unscale_keypoints(keypoints, scales):
    return (keypoints + 0.5) * scales[None] - 0.5


def scale_keypoints(keypoints, scales):
    return (keypoints + 0.5) / scales[None] - 0.5
