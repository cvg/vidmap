"""Keyframe and salient-feature cache identities, repair, and publication."""

import logging

import torch

from vidmap.frontend.cache import (
    cache_is_valid,
    cache_metadata,
    incremental_cache_is_complete,
    inspect_incremental_items,
    mark_incremental_cache_complete,
    ordered_files_fingerprint,
    prepare_incremental_cache,
    prune_incremental_items,
    read_cache_metadata,
    read_keyframe_artifact,
    semantic_config,
    write_keyframe_artifact,
)
from vidmap.frontend.geocalib import keyframe_bootstrap_cache_identity
from vidmap.frontend.keyframes import selector as keyframe_selector
from vidmap.frontend.keyframes.selection import compute_aliked_features_for_frame
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


def keyframe_cache_metadata(
    *,
    scene_parser,
    sequence,
    timestamps,
    tracker_options,
    lowres_options,
    keyframe_options,
    salient_options,
):
    """Build the keyframe cache identity from explicit stage dependencies."""
    intrinsics_source = keyframe_options.intrinsics_source
    keyframe_config = semantic_config(keyframe_options, exclude_fields=frozenset({"intrinsics_source"}))
    config = {
        "tracker": romav2_cache_identity(tracker_options),
        "lowres": lowres_options,
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
        stage="keyframes",
        config=config,
        ordered_inputs=ordered_inputs,
        payload_format="ordered-keyframe-indices",
        nonsemantic_config_fields=_RUNTIME_CONFIG_FIELDS,
    )


def salient_feature_cache_metadata(*, salient_options, sequence, timestamps, keyframe_metadata):
    """Build the salient-feature identity from the committed keyframe artifact."""
    keyframe_fingerprint = keyframe_metadata.get(
        "artifact_fingerprint",
        keyframe_metadata["identity_fingerprint"],
    )
    return cache_metadata(
        stage="salient_features",
        config={"features": semantic_config(salient_options), "model": aliked_cache_identity()},
        ordered_inputs={
            "sequence": sequence,
            "timestamps": _ordered_timestamps(sequence, timestamps),
        },
        upstream={"keyframes": keyframe_fingerprint},
        payload_format="per-image-local-features",
        nonsemantic_config_fields=_RUNTIME_CONFIG_FIELDS,
    )


def load_or_repair_cached_ids(
    *,
    scene_parser,
    sequence,
    keyframes_path,
    salient_features_path,
    force_recompute,
    keyframe_options,
    salient_options,
    keyframes_metadata,
    salient_metadata,
):
    if force_recompute or not cache_is_valid(
        keyframes_path,
        keyframes_metadata,
    ):
        return None
    keyframe_ids = read_keyframe_artifact(keyframes_path, keyframes_metadata)
    last_frame_idx = len(sequence) - 1
    if len(keyframe_ids) < 2 or keyframe_ids[0] != 0 or keyframe_ids[-1] != last_frame_idx:
        logger.info(
            f"Cached keyframes invalid: first={keyframe_ids[0] if keyframe_ids else None}, "
            f"last={keyframe_ids[-1] if keyframe_ids else None}, expected first=0, last={last_frame_idx}"
        )
        return None
    logger.debug("Loaded cached keyframes (by sequence index): %s", keyframe_ids)
    logger.info("Loaded %d cached keyframes", len(keyframe_ids))
    if keyframe_options.force_gt_keyframes:
        gt_indices = keyframe_selector.get_gt_frame_indices(sequence, scene_parser)
        missing_gt = set(gt_indices) - set(keyframe_ids)
        if missing_gt:
            raise AssertionError(
                f"Cached keyframes missing {len(missing_gt)} GT frames (out of {len(gt_indices)} total).\n"
                f"Missing GT indices: {sorted(missing_gt)}\n"
                "This should not happen with the fixed code. Re-run with --force-frontend to regenerate keyframes."
            )

    expected_names = [sequence[index] for index in keyframe_ids]
    if incremental_cache_is_complete(
        salient_features_path,
        salient_metadata,
        expected_names,
    ):
        return keyframe_ids

    prepare_incremental_cache(
        salient_features_path,
        salient_metadata,
        overwrite=False,
    )
    prune_incremental_items(salient_features_path, expected_names)
    _present, missing = inspect_incremental_items(
        salient_features_path,
        expected_names,
        salient_metadata,
        repair_malformed=True,
    )
    if missing:
        aliked_model = keyframe_selector.create_aliked_model(salient_options)
        try:
            for name in missing:
                compute_aliked_features_for_frame(
                    scene_parser,
                    salient_features_path,
                    name,
                    aliked_model,
                    salient_options,
                )
        finally:
            del aliked_model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    mark_incremental_cache_complete(salient_features_path, salient_metadata, expected_names)
    return keyframe_ids


def prepare_salient_feature_cache(salient_features_path, salient_metadata, *, overwrite):
    prepare_incremental_cache(
        salient_features_path,
        salient_metadata,
        overwrite=overwrite,
    )


def commit_keyframes(
    *,
    sequence,
    timestamps,
    keyframe_ids,
    gt_frame_indices,
    keyframes_path,
    salient_features_path,
    keyframe_options,
    salient_options,
    keyframes_metadata,
    salient_metadata,
):
    count_message = f"Number of detected keyframes: {len(keyframe_ids)}"
    if keyframe_options.force_gt_keyframes:
        count_message += f" (including {sum(index in gt_frame_indices for index in keyframe_ids)} GT frames)"
    logger.debug("Detected keyframes (by sequence index): %s", keyframe_ids)
    logger.info(count_message)
    write_keyframe_artifact(keyframes_path, keyframe_ids, keyframes_metadata)
    expected_names = [sequence[index] for index in keyframe_ids]
    mark_incremental_cache_complete(
        salient_features_path,
        salient_metadata,
        expected_names,
    )
    final_metadata = salient_feature_cache_metadata(
        salient_options=salient_options,
        sequence=sequence,
        timestamps=timestamps,
        keyframe_metadata=read_cache_metadata(keyframes_path),
    )
    mark_incremental_cache_complete(
        salient_features_path,
        final_metadata,
        expected_names,
    )
    logger.info("Cached keyframes to: %s", keyframes_path)
