"""Native frame selection and salient-feature cache publication."""

import logging

import numpy as np

from vidmap.frontend.cache import (
    CacheMetadataMismatch,
    cache_is_valid,
    cache_metadata,
    read_single_dataset,
    semantic_config,
    write_single_dataset,
)
from vidmap.frontend.geocalib import keyframe_bootstrap_cache_identity
from vidmap.frontend.keyframes import selector as keyframe_selector
from vidmap.frontend.models.aliked import aliked_cache_identity
from vidmap.frontend.models.romav2 import romav2_cache_identity

_RUNTIME_CONFIG_FIELDS = frozenset({"num_workers"})

logger = logging.getLogger("vidmap.frontend.keyframes.processing")


def admitted_frames_cache_metadata(
    *,
    input_snapshot,
    sequence,
    frame_ids,
    tracker_options,
    track_options,
    lowres_options,
    highres_options,
    keyframe_options,
    salient_options,
):
    """Bind admission to the reference stream and its native frame timeline."""
    scene_parser = input_snapshot.capture
    image_fingerprint = input_snapshot.image_content_fingerprint
    intrinsics_source = keyframe_options.intrinsics_source
    keyframe_config = semantic_config(keyframe_options, exclude_fields=frozenset({"intrinsics_source"}))
    config = {
        "tracker": romav2_cache_identity(tracker_options),
        "lowres": lowres_options,
        "keyframes": keyframe_config,
        "salient_features": semantic_config(salient_options),
        "salient_feature_model": aliked_cache_identity(),
    }
    if keyframe_options.lookahead_pruning:
        config["lookahead_tracks"] = semantic_config(track_options)
        config["highres"] = highres_options
    if lowres_options.batch_size > 1:
        # Zero-pad the final partial batch to reuse the full-batch compiled graph.
        config["lowres_batch_padding"] = "zeros-to-configured-batch-size"
    ordered_inputs = {
        "sequence": sequence,
        "frame_ids": frame_ids,
        "image_content": image_fingerprint(sequence),
    }
    if keyframe_options.force_gt_keyframes:
        ordered_inputs["forced_keyframe_names"] = [
            name for name in sequence if name in scene_parser.forced_keyframe_names
        ]
    if intrinsics_source == "geocalib":
        bootstrap_config, bootstrap_inputs = keyframe_bootstrap_cache_identity(sequence)
        config["bootstrap_geocalib"] = bootstrap_config
        ordered_inputs["bootstrap"] = bootstrap_inputs
        ordered_inputs["bootstrap_image_content"] = image_fingerprint(bootstrap_inputs["images"])
    else:
        config["ground_truth_intrinsics"] = {"policy": "ordered-effective-reconstruction-calibration-v1"}
        ordered_inputs["ground_truth_intrinsics"] = keyframe_selector.ground_truth_intrinsics_plan(
            scene_parser, sequence
        )
    return cache_metadata(
        stage="admitted_frames",
        config=config,
        ordered_inputs=ordered_inputs,
        payload_format="ordered-native-frame-ids",
        nonsemantic_config_fields=_RUNTIME_CONFIG_FIELDS,
    )


def salient_feature_cache_metadata(*, salient_options):
    """Feature inference depends on its model; source bytes are bound per image."""
    return cache_metadata(
        stage="salient_features",
        config={
            "features": semantic_config(salient_options),
            "model": aliked_cache_identity(),
        },
        ordered_inputs={},
        payload_format="per-image-local-features",
        nonsemantic_config_fields=_RUNTIME_CONFIG_FIELDS,
    )


def load_cached_frame_ids(
    *, frames, scene_parser, keyframe_options, admitted_frames_path, force_recompute, admission_metadata
):
    """Read the selected native frame IDs; image streams are a topology projection."""
    if force_recompute or not cache_is_valid(admitted_frames_path, admission_metadata):
        return None
    values = read_single_dataset(admitted_frames_path, admission_metadata)
    if values.ndim != 1 or values.dtype != np.dtype("uint32"):
        raise CacheMetadataMismatch("Selected frame payload must be a uint32 vector")
    frame_ids = tuple(map(int, values))
    indices = frames.indices(frame_ids)
    if len(indices) < 2 or indices[0] != 0 or indices[-1] != len(frames.frame_ids) - 1:
        raise CacheMetadataMismatch("Selected frames must include both timeline endpoints")
    logger.info("Loaded %d admitted frames", len(frame_ids))
    if keyframe_options.force_gt_keyframes:
        gt_indices = keyframe_selector.get_gt_frame_indices(frames.streams[0], scene_parser)
        missing_gt = set(gt_indices) - set(indices)
        if missing_gt:
            raise AssertionError(
                f"Cached keyframes missing {len(missing_gt)} GT frames (out of {len(gt_indices)} total).\n"
                f"Missing GT indices: {sorted(missing_gt)}\n"
                "This should not happen with the fixed code. Re-run with --force-frontend to regenerate keyframes."
            )
    return frame_ids


def commit_keyframes(
    *,
    frames,
    keyframe_ids,
    gt_frame_indices,
    admitted_frames_path,
    keyframe_options,
    admission_metadata,
):
    frame_ids = tuple(frames.frame_ids[index] for index in keyframe_ids)
    count_message = f"Number of detected keyframes: {len(keyframe_ids)}"
    if keyframe_options.force_gt_keyframes:
        count_message += f" (including {sum(index in gt_frame_indices for index in keyframe_ids)} GT frames)"
    logger.debug("Detected keyframes (by sequence index): %s", keyframe_ids)
    logger.info(count_message)
    write_single_dataset(admitted_frames_path, np.asarray(frame_ids, dtype=np.uint32), admission_metadata)
    logger.info("Cached admitted frames to: %s", admitted_frames_path)
