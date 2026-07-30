"""Select spatially distributed salient keypoints that start new tracks."""

import numpy as np
import torch

from vidmap.frontend.tracking.sampling import kde_blind


class SalientKeypointSelector:
    """Choose detector-proposed salient keypoints that complement RoMa certainty sampling."""

    def __init__(self, options):
        self.options = options

    def _density_budget_scale(self, samples_needed: int) -> float:
        reference = max(1.0, float(self.options.salient_density_ref_kps))
        actual = max(1.0, float(samples_needed))
        scale = max(1.0, np.sqrt(reference / actual))
        return min(float(self.options.salient_density_max_scale), scale)

    @staticmethod
    def _normalize_current_keypoints(keypoints, current_size):
        width, height = current_size
        keypoints_norm = keypoints.copy().astype(float)
        keypoints_norm[:, 0] = (keypoints_norm[:, 0] / (width - 1)) * 2 - 1.0
        keypoints_norm[:, 1] = (keypoints_norm[:, 1] / (height - 1)) * 2 - 1.0
        return keypoints_norm

    def select(
        self,
        candidate_keypoints,
        available_mask,
        confidence_mask,
        existing_keypoints,
        *,
        occupied_count,
        current_size,
        kf_id: int,
    ):
        """Return an input-aligned mask of candidates selected to start new tracks."""
        samples_needed = max(
            0,
            (self.options.max_kps - occupied_count) // 2,
        )
        selected_mask = np.zeros_like(available_mask, dtype=bool)
        valid_mask = available_mask & confidence_mask
        valid_count = int(valid_mask.sum())
        num_samples = min(max(0, int(samples_needed)), valid_count)
        if num_samples == 0:
            return selected_mask

        density_scale = self._density_budget_scale(num_samples)
        density_std = float(self.options.salient_density_std) * density_scale
        density_power = float(self.options.salient_density_power) * density_scale
        width, height = current_size
        blind_radius = self.options.nms_radius * (2.0 / max(height, width))

        candidate_keypoints_norm = self._normalize_current_keypoints(candidate_keypoints, current_size)
        valid_candidates_norm = torch.tensor(candidate_keypoints_norm[valid_mask])
        candidate_density = kde_blind(
            valid_candidates_norm,
            std=density_std,
            blind_radius=blind_radius,
        )

        if existing_keypoints is not None and len(existing_keypoints) > 0:
            existing_norm = self._normalize_current_keypoints(existing_keypoints, current_size)
            candidate_density += kde_blind(
                valid_candidates_norm,
                neighbors=torch.tensor(existing_norm),
                std=density_std,
                blind_radius=0.0,
            )

        sampling_probability = (1 / (candidate_density + 1)) ** density_power
        density_threshold = (self.options.salient_density_ref_kps * 2 / 20000.0) * 10.0
        sampling_probability[candidate_density < max(1.0, density_threshold)] = 1e-7

        torch.manual_seed(42 + kf_id)
        balanced_samples = torch.multinomial(
            sampling_probability,
            num_samples=num_samples,
            replacement=False,
        )

        valid_indices = np.flatnonzero(valid_mask)
        selected_mask[valid_indices[balanced_samples.cpu().numpy()]] = True
        return selected_mask
