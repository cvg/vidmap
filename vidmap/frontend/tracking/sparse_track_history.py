"""Fixed-size position and covariance history for sparse tracks."""

import numpy as np


class SparseTrackHistory:
    """Own the sliding position and covariance buffers used by sparse tracking."""

    def __init__(self, *, history_reach: int, max_keypoints: int):
        if history_reach < 1:
            raise ValueError("history_reach must be positive")
        self.reach = int(history_reach)
        capacity = max_keypoints * 10
        self.track = np.full((self.reach - 1, capacity, 2), -1, dtype=np.int32)
        self.cov = np.full((self.reach, capacity, 2, 2), 0, dtype=np.float32)
        self.cov[:, :, 0, 0] = -1
        self.cov[:, :, 1, 1] = -1

    def mask_tracks(self, mask):
        """Keep surviving tracks in their existing order and clear the tail."""
        num_kept = mask.sum()
        self.track[:, :num_kept] = self.track[:, : mask.shape[0]][:, mask]
        self.track[:, num_kept:] = -1

        cov_size = min(mask.shape[0], self.cov.shape[1])
        cov_mask = mask[:cov_size]
        cov_kept = cov_mask.sum()
        self.cov[:, :cov_kept] = self.cov[:, :cov_size][:, cov_mask]
        self.cov[:, cov_kept:] = 0

    def append_track(self, keypoints, covariances=None):
        """Shift one frame and append current positions and optional covariances."""
        if len(self.track):
            self.track[:-1] = self.track[1:]
            self.track[-1, : keypoints.shape[0]] = keypoints

        if covariances is not None:
            self.cov[:-1] = self.cov[1:]
            num_to_store = min(covariances.shape[0], self.cov.shape[1])
            self.cov[-1, :num_to_store] = covariances[:num_to_store]
            self.cov[-1, num_to_store:] = 0
