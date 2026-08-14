"""Admitted track-pair and salient-feature cache identities and publication."""

import logging

from vidmap.frontend.cache import (
    cache_is_valid,
    cache_metadata,
    incremental_cache_is_complete,
    mark_incremental_cache_complete,
    ordered_files_fingerprint,
    prune_incremental_items,
    read_cache_metadata,
    read_pair_artifact,
    semantic_config,
    write_pair_artifact,
)
from vidmap.frontend.geocalib import keyframe_bootstrap_cache_identity
from vidmap.frontend.keyframes import selector as keyframe_selector
from vidmap.frontend.models.aliked import aliked_cache_identity
from vidmap.frontend.models.romav2 import romav2_cache_identity

_RUNTIME_CONFIG_FIELDS = frozenset({"num_workers"})

logger = logging.getLogger("vidmap.frontend.keyframes.processing")


def _ordered_timestamps(sequence, timestamps):
    values = []
    for name in sequence:
        value = timestamps[name]
        if hasattr(value, "isoformat"):
            value = value.isoformat()
        values.append([name, value])
    return values


def admitted_track_pairs_cache_metadata(
    *,
    scene_parser,
    sequence,
    timestamps,
    tracker_options,
    lowres_options,
    highres_options,
    keyframe_options,
    salient_options,
):
    """Build the admitted adjacent-pair identity from both selection passes."""
    intrinsics_source = keyframe_options.intrinsics_source
    keyframe_config = semantic_config(keyframe_options, exclude_fields=frozenset({"intrinsics_source"}))
    config = {
        "tracker": romav2_cache_identity(tracker_options),
        "lowres": lowres_options,
        "highres": highres_options,
        "keyframes": keyframe_config,
        "salient_features": semantic_config(salient_options),
        "salient_feature_model": aliked_cache_identity(),
    }
    ordered_inputs = {
        "sequence": sequence,
        "timestamps": _ordered_timestamps(sequence, timestamps),
        "image_content": ordered_files_fingerprint(scene_parser.rgb_dir, sequence),
    }
    if intrinsics_source == "geocalib":
        bootstrap_config, bootstrap_inputs = keyframe_bootstrap_cache_identity(sequence)
        config["bootstrap_geocalib"] = bootstrap_config
        ordered_inputs["bootstrap"] = bootstrap_inputs
        ordered_inputs["bootstrap_image_content"] = ordered_files_fingerprint(
            scene_parser.rgb_dir,
            bootstrap_inputs["images"],
        )
    else:
        config["ground_truth_intrinsics"] = {"policy": "ordered-effective-reconstruction-calibration-v1"}
        ordered_inputs["ground_truth_intrinsics"] = keyframe_selector.ground_truth_intrinsics_plan(
            scene_parser, sequence
        )
    return cache_metadata(
        stage="track_pairs",
        config=config,
        ordered_inputs=ordered_inputs,
        payload_format="ordered-image-pairs",
        nonsemantic_config_fields=_RUNTIME_CONFIG_FIELDS,
    )


def salient_feature_cache_metadata(*, salient_options, sequence, timestamps, track_pairs_metadata):
    """Build the salient-feature identity from the admitted pair plan."""
    track_pairs_fingerprint = track_pairs_metadata.get(
        "artifact_fingerprint",
        track_pairs_metadata["identity_fingerprint"],
    )
    return cache_metadata(
        stage="salient_features",
        config={
            "features": semantic_config(salient_options),
            "model": aliked_cache_identity(),
        },
        ordered_inputs={
            "sequence": sequence,
            "timestamps": _ordered_timestamps(sequence, timestamps),
        },
        upstream={"track_pairs": track_pairs_fingerprint},
        payload_format="per-image-local-features",
        nonsemantic_config_fields=_RUNTIME_CONFIG_FIELDS,
    )


def load_cached_names(
    *,
    scene_parser,
    sequence,
    timestamps,
    track_pairs_path,
    salient_features_path,
    force_recompute,
    keyframe_options,
    salient_options,
    track_pairs_metadata,
):
    if force_recompute or not cache_is_valid(
        track_pairs_path,
        track_pairs_metadata,
    ):
        return None
    pairs = tuple(read_pair_artifact(track_pairs_path, track_pairs_metadata))
    if not pairs:
        logger.info("Cached admitted track-pair plan is empty")
        return None
    names = (pairs[0][0], *(pair[1] for pair in pairs))
    if pairs != tuple(zip(names, names[1:])):
        logger.info("Cached admitted track pairs do not form one adjacent chain")
        return None
    positions = {name: index for index, name in enumerate(sequence)}
    if any(name not in positions for name in names):
        logger.info("Cached admitted track pairs contain images outside the source sequence")
        return None
    keyframe_ids = [positions[name] for name in names]
    last_frame_idx = len(sequence) - 1
    if (
        keyframe_ids[0] != 0
        or keyframe_ids[-1] != last_frame_idx
        or any(right <= left for left, right in zip(keyframe_ids, keyframe_ids[1:]))
    ):
        logger.info(
            f"Cached keyframes invalid: first={keyframe_ids[0] if keyframe_ids else None}, "
            f"last={keyframe_ids[-1] if keyframe_ids else None}, expected first=0, last={last_frame_idx}"
        )
        return None
    logger.debug("Loaded admitted keyframes (by sequence index): %s", keyframe_ids)
    logger.info("Loaded %d admitted keyframes from track pairs", len(keyframe_ids))
    if keyframe_options.force_gt_keyframes:
        gt_indices = keyframe_selector.get_gt_frame_indices(sequence, scene_parser)
        missing_gt = set(gt_indices) - set(keyframe_ids)
        if missing_gt:
            raise AssertionError(
                f"Cached keyframes missing {len(missing_gt)} GT frames (out of {len(gt_indices)} total).\n"
                f"Missing GT indices: {sorted(missing_gt)}\n"
                "This should not happen with the fixed code. Re-run with --force-frontend to regenerate keyframes."
            )

    salient_metadata = salient_feature_cache_metadata(
        salient_options=salient_options,
        sequence=sequence,
        timestamps=timestamps,
        track_pairs_metadata=read_cache_metadata(track_pairs_path),
    )
    expected_names = [sequence[index] for index in keyframe_ids]
    if incremental_cache_is_complete(
        salient_features_path,
        salient_metadata,
        expected_names,
    ):
        return names
    logger.info("Cached admission and track-propagation section is incomplete")
    return None


def commit_keyframes(
    *,
    sequence,
    timestamps,
    keyframe_ids,
    gt_frame_indices,
    track_pairs_path,
    salient_features_path,
    keyframe_options,
    salient_options,
    track_pairs_metadata,
):
    count_message = f"Number of detected keyframes: {len(keyframe_ids)}"
    if keyframe_options.force_gt_keyframes:
        count_message += f" (including {sum(index in gt_frame_indices for index in keyframe_ids)} GT frames)"
    logger.debug("Detected keyframes (by sequence index): %s", keyframe_ids)
    logger.info(count_message)
    expected_names = [sequence[index] for index in keyframe_ids]
    track_pairs = tuple(zip(expected_names, expected_names[1:]))
    write_pair_artifact(track_pairs_path, track_pairs, track_pairs_metadata)
    prune_incremental_items(salient_features_path, expected_names)
    final_metadata = salient_feature_cache_metadata(
        salient_options=salient_options,
        sequence=sequence,
        timestamps=timestamps,
        track_pairs_metadata=read_cache_metadata(track_pairs_path),
    )
    mark_incremental_cache_complete(salient_features_path, final_metadata, expected_names)
    logger.info("Cached admitted track pairs to: %s", track_pairs_path)
