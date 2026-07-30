"""
Build sparse tracks from dense matches using temporal propagation.

This module implements track propagation across keyframes by:
1. Loading dense matches between consecutive keyframe pairs
2. Selecting keypoints using NMS on certainty maps
3. Propagating tracks across frames with covariance-based quality filtering
4. Maintaining a sliding window of track history for long-track refinement

The implementation uses a hybrid CPU/GPU approach:
- GPU: dense matching, NMS, keypoint selection
- CPU/Numpy: track storage, long-term history, scipy-based NMS

Coordinate conventions:
- Sizes stored as (W, H) tuples
- Keypoints stored as (x, y) with pixel centers at integer + 0.5
- Dense match fields use pixel corners (integer coordinates)
- Array shapes follow (H, W) indexing order [y, x]
"""

from dataclasses import dataclass

import numpy as np
import torch

from vidmap.frontend.tracking.kernels import (
    SparseProjection,
    build_tracks_from_matches,
    select_keypoints_from_certainty,
)
from vidmap.frontend.tracking.long_track_refinement import refine_long_tracks_adaptive
from vidmap.frontend.tracking.previous_track_eligibility import select_previous_track_mask
from vidmap.frontend.tracking.salient_keypoint_selection import SalientKeypointSelector
from vidmap.frontend.tracking.sampling import nn_sample_2d
from vidmap.frontend.tracking.sparse_track_history import SparseTrackHistory
from vidmap.utils.keypoint_scaling import scale_keypoints, unscale_keypoints

_SALIENT_KEYPOINT_STD_PX = 2.0


@dataclass(frozen=True)
class SparseTrackTransition:
    """Serialized frame outputs produced by one numerical state transition."""

    source_keypoints: np.ndarray
    source_uncertainty: np.ndarray
    matches: np.ndarray
    matching_scores: np.ndarray


class SparseTrackState:
    """
    Temporal track propagation for sparse feature tracking.

    Propagates keypoint tracks across a sequence of keyframes by:
    - Selecting keypoints using NMS on dense match certainty
    - Maintaining track history in a sliding window
    - Filtering tracks based on geometric and photometric consistency
    - Accumulating confidence and covariance over time

    The tracker maintains keypoint and covariance history across frames
    - State: previous frame's keypoints, confidences, covariances, track lengths
    """

    def __init__(self, conf, *, original_size, current_size, scheduled_hops):
        """Initialize numerical propagation state from explicit image geometry."""
        self.conf = conf
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.original_size = original_size
        self.current_size = current_size
        self.scale_ratio = np.array(original_size) / np.array(current_size)
        self.scheduled_hops = scheduled_hops
        self.prev_keypoints = None
        self.prev_conf = None
        self.prev_covar = None
        self.track_length = None
        self.history = SparseTrackHistory(
            window=self.conf.window,
            max_keypoints=self.conf.max_kps,
        )
        self.salient_keypoint_selector = SalientKeypointSelector(self.conf)

    def _propagated_covar_to_original_pixels_3(self, covar_2x2):
        covariance = np.stack(
            [covar_2x2[:, 0, 0], covar_2x2[:, 1, 1], covar_2x2[:, 0, 1]],
            axis=-1,
        )
        scale_x, scale_y = self.scale_ratio
        covariance[:, 0] *= scale_x**2
        covariance[:, 1] *= scale_y**2
        covariance[:, 2] *= scale_x * scale_y
        return covariance

    @property
    def current_width(self):
        """Current processing image width (after any resizing)."""
        return self.current_size[0]

    @property
    def current_height(self):
        """Current processing image height (after any resizing)."""
        return self.current_size[1]

    def _build_track_state(self, kps0, kps_prev_mask):
        """
        Build accumulated confidence, covariance, and track lengths.

        For new tracks: initialize with defaults (conf=1, covar=0, length=0)
        For continuing tracks: use previous values and increment length

        Args:
            kps0: (N, 2) selected keypoints for current frame
            kps_prev_mask: Boolean mask indicating which prev keypoints continue

        Returns:
            conf0: (N,) accumulated confidence
            covar0: (N, 2, 2) accumulated covariance
            track_length: (N,) number of frames tracked
        """
        if kps_prev_mask is None:
            conf0 = np.ones(kps0.shape[0], dtype=np.float32)
            covar0 = np.zeros((kps0.shape[0], 2, 2), dtype=np.float32)
            track_length = np.zeros(kps0.shape[0], dtype=np.int32)
        else:
            num_prev = kps_prev_mask.sum()
            num_new = kps0.shape[0] - num_prev

            # Continuing tracks
            conf_prev = self.prev_conf[kps_prev_mask]
            covar_prev = self.prev_covar[kps_prev_mask]
            length_prev = self.track_length[kps_prev_mask] + 1

            # New tracks
            conf_new = np.ones(num_new, dtype=np.float32)
            covar_new = np.zeros((num_new, 2, 2), dtype=np.float32)
            length_new = np.zeros(num_new, dtype=np.int32)

            # Combine
            conf0 = np.concatenate([conf_prev, conf_new])
            covar0 = np.concatenate([covar_prev, covar_new])
            track_length = np.concatenate([length_prev, length_new])

        return conf0, covar0, track_length

    def _project_tracks(
        self,
        source_keypoints,
        propagated,
        certainty,
        previous_mask,
        track_length,
    ):
        """Validate propagated tracks and construct persisted match indices."""
        points_0 = source_keypoints.cpu().numpy()
        points_1 = propagated.keypoints.cpu().numpy()
        confidence = propagated.confidence.cpu().numpy()
        certainty_np = certainty.cpu().numpy() if torch.is_tensor(certainty) else certainty
        survivors = (
            (points_1[:, 0] >= -0.5)
            * (points_1[:, 0] < self.current_width - 0.5)
            * (points_1[:, 1] >= -0.5)
            * (points_1[:, 1] < self.current_height - 0.5)
        )
        survivors *= confidence > self.conf.min_conf
        assert np.all(survivors[track_length == 0]), "tracks_mask should be all True for tracks with length 0"

        sequential_threshold = self.conf.max_sequential_track_sigma_roma_px
        if sequential_threshold is not None:
            covariance = propagated.covariance
            covariance_np = covariance.cpu().numpy() if torch.is_tensor(covariance) else np.asarray(covariance)
            sigma_squared = covariance_np[:, 0, 0] + covariance_np[:, 1, 1]
            sequential_sigma_roma_px = np.sqrt(np.maximum(sigma_squared, 0.0))
            survivors &= sequential_sigma_roma_px <= sequential_threshold

        target_keypoints = points_1[survivors]
        if self.prev_keypoints is None:
            saved_keypoints = points_0[None]
            matched_sources = np.where(survivors)[0]
            matches = np.column_stack((matched_sources, np.arange(len(matched_sources))))
            matches0 = np.full(saved_keypoints.shape[1], -1, dtype=int)
            scores0 = np.zeros(saved_keypoints.shape[1], dtype=np.float32)
            matches0[matches[:, 0]] = matches[:, 1]
            scores0[matches[:, 0]] = certainty_np[survivors]
            saved_survivors = survivors.copy()
        else:
            scaled_previous = scale_keypoints(self.prev_keypoints, self.scale_ratio)
            new_keypoints = points_0[previous_mask.sum() :]
            saved_keypoints = np.vstack([scaled_previous, new_keypoints])[None]
            saved_survivors = np.ones(saved_keypoints.shape[1], dtype=bool)
            saved_survivors[np.where(~previous_mask)[0]] = False
            saved_survivors[saved_survivors] = survivors
            matches = np.vstack([np.where(saved_survivors)[0], np.arange(survivors.sum())]).T
            matches0 = np.full(saved_keypoints.shape[1], -1, dtype=int)
            scores0 = np.zeros(saved_keypoints.shape[1], dtype=np.float32)
            matches0[matches[:, 0]] = matches[:, 1]
            scores0[matches[:, 0]] = certainty_np[survivors]
        return SparseProjection(
            matches0,
            scores0,
            saved_keypoints,
            target_keypoints,
            survivors,
            saved_survivors,
        )

    def advance(self, *, kf_id, matches, certainty, covariance, salient_keypoints, dense_hops):
        """Advance one frame while preserving the established numerical operation order."""
        kps_prev_mask = select_previous_track_mask(
            matches,
            certainty,
            previous_keypoints=self.prev_keypoints,
            track_length=self.track_length,
            history=self.history,
            options=self.conf,
            scale_ratio=self.scale_ratio,
            current_size=self.current_size,
        )

        source_keypoints, certainty_01, covariance_01, _ = self._select_source_keypoints(
            kf_id=kf_id,
            certainty=certainty,
            covariance=covariance,
            salient_keypoints=salient_keypoints,
            previous_mask=kps_prev_mask,
        )
        source_confidence, source_covariance, track_length = self._build_track_state(
            source_keypoints,
            kps_prev_mask,
        )

        propagated = build_tracks_from_matches(
            source_keypoints,
            source_confidence,
            source_covariance,
            matches,
            certainty_01,
            covariance_01,
            bilinear=False,
        )
        target_confidence = propagated.confidence
        target_covariance = propagated.covariance
        projection = self._project_tracks(
            source_keypoints,
            propagated,
            certainty_01,
            kps_prev_mask,
            track_length,
        )
        target_keypoints_np = projection.target_keypoints
        survivors = projection.survivors
        if self.prev_keypoints is not None:
            target_keypoints_np = refine_long_tracks_adaptive(
                target_keypoints_np,
                track_length,
                survivors,
                kps_prev_mask,
                target_covariance,
                dense_hops,
                history=self.history,
                options=self.conf,
                scheduled_hops=self.scheduled_hops,
                scale_ratio=self.scale_ratio,
                current_size=self.current_size,
            )

        transition = self._serialize_source_transition(
            projection=projection,
            target_covariance=target_covariance,
        )
        self._commit_transition(
            source_keypoints=source_keypoints,
            source_covariance=source_covariance,
            target_keypoints=target_keypoints_np,
            target_confidence=target_confidence,
            target_covariance=target_covariance,
            track_length=track_length,
            survivors=survivors,
            previous_mask=kps_prev_mask,
        )
        return transition

    def _select_source_keypoints(self, *, kf_id, certainty, covariance, salient_keypoints, previous_mask):
        """Select continuing and new source keypoints for one transition."""

        keep_salient_mask = np.ones_like(salient_keypoints[..., 0], dtype=bool)
        check_existing_kps = None
        if self.prev_keypoints is not None:
            check_existing_kps = scale_keypoints(self.prev_keypoints[previous_mask], self.scale_ratio)
            if len(check_existing_kps) > 0 and len(salient_keypoints) > 0:
                diff = salient_keypoints[:, None, :] - check_existing_kps[None, :, :]
                sq_dists = (diff**2).sum(axis=-1)
                keep_salient_mask = (sq_dists > self.conf.nms_radius**2).all(axis=1)

        sample = nn_sample_2d
        certainty_np = certainty.cpu().numpy()
        salient_certainty = sample(
            certainty_np,
            salient_keypoints[:, 0],
            salient_keypoints[:, 1],
        )
        salient_conf_mask = salient_certainty > self.conf.min_conf
        selected_salient_mask = self.salient_keypoint_selector.select(
            salient_keypoints,
            keep_salient_mask,
            salient_conf_mask,
            check_existing_kps,
            occupied_count=(previous_mask.sum() if previous_mask is not None else 0),
            current_size=self.current_size,
            kf_id=kf_id,
        )
        selected_salient = salient_keypoints[selected_salient_mask]
        if self.prev_keypoints is not None:
            candidate_keypoints = np.concatenate([check_existing_kps, selected_salient])
        else:
            candidate_keypoints = selected_salient

        selected = select_keypoints_from_certainty(
            certainty,
            covariance,
            candidate_keypoints,
            self.conf.nms_radius,
            self.conf.min_conf,
            self.conf.max_kps,
            bilinear=False,
        )
        certainty_01 = selected.certainty.cpu()
        covariance_01 = selected.covariance.cpu()
        source_keypoints = selected.keypoints.cpu()
        return source_keypoints, certainty_01, covariance_01, selected_salient_mask

    def _serialize_source_transition(
        self,
        *,
        projection,
        target_covariance,
    ):
        """Serialize the committed source frame without mutating track state."""
        survivors = projection.survivors
        source_keypoints_to_save = unscale_keypoints(projection.saved_keypoints, self.scale_ratio)
        target_covariance_np = (
            target_covariance.cpu().numpy() if torch.is_tensor(target_covariance) else target_covariance
        )
        if self.prev_keypoints is None:
            source_covariance_to_save = target_covariance_np
        else:
            default_var = _SALIENT_KEYPOINT_STD_PX**2
            source_covariance_to_save = np.zeros(
                (source_keypoints_to_save.shape[1], 2, 2),
                dtype=np.float32,
            )
            source_covariance_to_save[:, 0, 0] = default_var
            source_covariance_to_save[:, 1, 1] = default_var
            source_covariance_to_save[projection.saved_survivors] = target_covariance_np[survivors]
        source_uncertainty = self._propagated_covar_to_original_pixels_3(source_covariance_to_save)
        matches0 = projection.matches
        scores0 = projection.scores
        return SparseTrackTransition(
            source_keypoints=source_keypoints_to_save,
            source_uncertainty=source_uncertainty,
            matches=matches0,
            matching_scores=scores0,
        )

    def _commit_transition(
        self,
        *,
        source_keypoints,
        source_covariance,
        target_keypoints,
        target_confidence,
        target_covariance,
        track_length,
        survivors,
        previous_mask,
    ):
        """Commit one validated transition and update history buffers."""
        self.prev_conf = target_confidence[survivors]
        self.prev_covar = target_covariance[survivors]
        self.prev_keypoints = unscale_keypoints(target_keypoints, self.scale_ratio)
        self.track_length = track_length[survivors]
        if previous_mask is not None:
            previous_survivors = previous_mask.copy()
            previous_survivors[previous_mask] = survivors[: previous_mask.sum()]
            self.history.mask_tracks(previous_survivors)
        self.history.append_track(source_keypoints[survivors], source_covariance[survivors])

    def serialize_current_target(self) -> tuple[np.ndarray, np.ndarray]:
        """Serialize the committed target frame only when it becomes an output."""
        if self.prev_keypoints is None or self.prev_covar is None:
            raise RuntimeError("Cannot serialize target keypoints before a tracking transition")
        keypoints = self.prev_keypoints[None].astype(np.float32)
        uncertainty = self._propagated_covar_to_original_pixels_3(self.prev_covar)
        return keypoints, uncertainty
