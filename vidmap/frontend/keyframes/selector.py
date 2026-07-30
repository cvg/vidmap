"""Keyframe selection state and calibration planning."""

import logging

import numpy as np
import torch

from vidmap.frontend.keyframes import selection as keyframe_utils
from vidmap.frontend.keyframes.selection import compute_aliked_features_for_frame
from vidmap.frontend.models.aliked import ALIKED

logger = logging.getLogger("vidmap.frontend.keyframes.processing")


def get_gt_frame_indices(sequence, scene_parser):
    """Return sequence indices for frames that have ground truth poses."""
    gt_names = set()
    total_images = 0
    for image in scene_parser.rec.images.values():
        total_images += 1
        if image.has_pose:
            gt_names.add(image.name)

    gt_indices = []
    for idx, name in enumerate(sequence):
        if name in gt_names:
            gt_indices.append(idx)

    msg = (
        f"get_gt_frame_indices: {len(gt_names)} GT frames out of {total_images} in scene_parser.rec, "
        f"{len(gt_indices)} matched in sequence of {len(sequence)}"
    )
    logger.info(msg)

    if len(gt_names) > 0 and len(gt_indices) == 0:
        seq_sample = list(sequence[:3]) if len(sequence) >= 3 else list(sequence)
        gt_sample = list(gt_names)[:3]
        msg2 = f"  Sequence names sample: {seq_sample}"
        msg3 = f"  GT names sample: {gt_sample}"
        logger.debug(msg2)
        logger.debug(msg3)

    return sorted(gt_indices)


def _build_sequence_camera_lookup(scene_parser):
    rec = scene_parser.rec
    name_to_image = {image.name: image for image in rec.images.values()}
    return rec.cameras, name_to_image


def _calibration_for_sequence_index(sequence, idx, cameras, name_to_image):
    if idx is None or idx < 0 or idx >= len(sequence):
        return None
    name = sequence[idx]
    image = name_to_image[name] if name in name_to_image else None
    if image is None or image.camera_id not in cameras:
        return None
    return np.asarray(cameras[image.camera_id].calibration_matrix(), dtype=np.float64)


def ground_truth_intrinsics_plan(scene_parser, sequence):
    """Return the ordered effective calibration plan used by GT keyframing."""
    cameras, name_to_image = _build_sequence_camera_lookup(scene_parser)
    return [
        [name, _calibration_for_sequence_index(sequence, index, cameras, name_to_image)]
        for index, name in enumerate(sequence)
    ]


def create_aliked_model(salient_options):
    from vidmap.frontend.models.aliked import ALIKEDOptions

    return (
        ALIKED(
            ALIKEDOptions(
                nms_radius=salient_options.nms_radius,
                max_num_keypoints=salient_options.max_num_keypoints,
                sub_pixel=salient_options.sub_pixel,
            )
        )
        .cuda()
        .eval()
    )


class KeyframeSelector:
    def __init__(
        self,
        scene_parser,
        salient_features_path,
        sequence,
        keyframe_options,
        salient_options,
        aliked_model,
        original_width,
        original_height,
        bootstrap_intrinsics,
    ):
        self.scene_parser = scene_parser
        self.salient_features_path = salient_features_path
        self.sequence = sequence
        self.conf = keyframe_options
        self.salient_options = salient_options
        self.aliked_model = aliked_model
        self.oW = original_width
        self.oH = original_height
        self.bootstrap_intrinsics = (
            np.asarray(bootstrap_intrinsics, dtype=np.float64) if bootstrap_intrinsics is not None else None
        )
        self.keyframe_ids = [0]
        self.seg_start_frame = 0
        self.pair_source_idx = 0
        self.last_good_frame_idx = None
        self.prev_matches_lr = None
        self.prev_cert_lr = None
        self.tracking_origin_frame_idx = 0
        self.ground_truth_intrinsics = None
        if keyframe_options.intrinsics_source == "ground_truth":
            self.ground_truth_intrinsics = [entry[1] for entry in ground_truth_intrinsics_plan(scene_parser, sequence)]
        self.gt_frame_indices = set()
        if keyframe_options.force_gt_keyframes:
            self.gt_frame_indices = set(get_gt_frame_indices(sequence, scene_parser))
            logger.info(
                "force_gt_keyframes enabled: forcing %d GT frame positions",
                len(self.gt_frame_indices),
            )

        logger.debug("Computing ALIKED features for initial keyframe: %s", sequence[0])
        self._compute_aliked_features(0)
        from vidmap.utils.io import get_keypoints

        self.tracked_kps = get_keypoints(salient_features_path, sequence[0])
        self.original_kps = self.tracked_kps.copy()
        self.ever_invisible = np.zeros(len(self.tracked_kps), dtype=bool)
        logger.debug("Loaded %d keypoints for tracking", len(self.tracked_kps))

    def _compute_aliked_features(self, frame_index):
        compute_aliked_features_for_frame(
            self.scene_parser,
            self.salient_features_path,
            self.sequence[frame_index],
            self.aliked_model,
            self.salient_options,
        )

    def _select_keyframe(self, is_gt_frame):
        if is_gt_frame:
            return self.pair_source_idx
        if self.last_good_frame_idx is not None and self.last_good_frame_idx > self.seg_start_frame:
            return self.last_good_frame_idx
        candidate = self.pair_source_idx
        if candidate == self.keyframe_ids[-1]:
            return min(self.pair_source_idx + 1, len(self.sequence) - 1)
        return candidate

    def _scoring_intrinsics(self, pair_target_frame_idx):
        if self.conf.intrinsics_source == "geocalib":
            return self.bootstrap_intrinsics, self.bootstrap_intrinsics
        source_K = self.ground_truth_intrinsics[self.tracking_origin_frame_idx]
        target_K = self.ground_truth_intrinsics[pair_target_frame_idx]
        return source_K, target_K

    def process_pair(self, pair_match_lr, pair_cert_lr):
        self._process_sparse_pair(pair_match_lr, pair_cert_lr)
        self.prev_matches_lr = pair_match_lr
        self.prev_cert_lr = pair_cert_lr
        self.pair_source_idx += 1

    def _propagate_and_update_visibility(self, pair_match_lr, pair_cert_lr):
        self.tracked_kps, _kps_cert, visible = keyframe_utils.propagate_keypoints(
            self.tracked_kps,
            pair_match_lr,
            pair_cert_lr,
            self.oW,
            self.oH,
            self.conf.certainty_threshold,
        )
        visible = visible.cpu().numpy() if isinstance(visible, torch.Tensor) else visible
        self.ever_invisible |= ~visible.astype(bool)
        self.visible_mask = visible & ~self.ever_invisible

    def _process_sparse_pair(self, pair_match_lr, pair_cert_lr):
        from vidmap.utils.io import get_keypoints

        self._propagate_and_update_visibility(pair_match_lr, pair_cert_lr)
        pair_target_frame_idx = min(self.pair_source_idx + 1, len(self.sequence) - 1)
        source_K, target_K = self._scoring_intrinsics(pair_target_frame_idx)
        motion_score = keyframe_utils.compute_normalized_keypoint_motion_score(
            self.tracked_kps,
            self.original_kps,
            self.visible_mask,
            source_K,
            target_K,
            self.conf.max_normalized_keypoint_drift,
        )

        is_gt_frame = self.pair_source_idx in self.gt_frame_indices and self.pair_source_idx != self.seg_start_frame
        if motion_score > self.conf.target_frac or is_gt_frame:
            new_keyframe = self._select_keyframe(is_gt_frame)
            if new_keyframe <= self.keyframe_ids[-1]:
                raise RuntimeError(
                    f"Keyframe selection must advance beyond {self.keyframe_ids[-1]}, got {new_keyframe}"
                )
            self.keyframe_ids.append(new_keyframe)
            self._compute_aliked_features(new_keyframe)
            self.tracked_kps = get_keypoints(self.salient_features_path, self.sequence[new_keyframe])
            self.ever_invisible = np.zeros(len(self.tracked_kps), dtype=bool)
            frames_skipped = self.pair_source_idx - new_keyframe
            if frames_skipped not in (-1, 0, 1):
                raise RuntimeError(
                    f"Keyframe rollback must select the current or previous pair source, got {new_keyframe} "
                    f"for source {self.pair_source_idx}"
                )
            if frames_skipped == 1 and self.prev_matches_lr is not None:
                self._propagate_and_update_visibility(self.prev_matches_lr, self.prev_cert_lr)
            if frames_skipped != -1:
                self._propagate_and_update_visibility(pair_match_lr, pair_cert_lr)
            self.seg_start_frame = new_keyframe if frames_skipped == -1 else self.pair_source_idx
            self.original_kps = (
                self.tracked_kps.clone() if isinstance(self.tracked_kps, torch.Tensor) else self.tracked_kps.copy()
            )
            self.tracking_origin_frame_idx = pair_target_frame_idx
            self.last_good_frame_idx = None
        else:
            self.last_good_frame_idx = self.pair_source_idx

    def finish(self):
        final_index = len(self.sequence) - 1
        if self.keyframe_ids[-1] != final_index:
            self.keyframe_ids.append(final_index)
            logger.debug("Adding final keyframe at index %d: %s", final_index, self.sequence[-1])
            self._compute_aliked_features(final_index)
        return self.keyframe_ids
