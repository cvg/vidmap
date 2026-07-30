"""Depth frontend and cache reuse."""

import logging
import sys
from contextlib import nullcontext
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import cv2
import h5py
import numpy as np
import torch
from tqdm import tqdm

from vidmap.datasets.base import DatasetParser
from vidmap.frontend.cache import (
    IncrementalArtifactContract,
    artifact_fingerprint,
    cache_metadata,
    certify_incremental_artifact,
    inspect_incremental_items,
    mark_incremental_cache_complete,
    prepare_incremental_cache,
    prune_incremental_items,
    read_cache_metadata,
    semantic_config,
)
from vidmap.frontend.depth_loading import create_da3_window_loader
from vidmap.frontend.h5_write_queue import H5WriteQueue
from vidmap.frontend.keyframes.processing import KeyframePlan
from vidmap.frontend.models.depth.da3_video import (
    DA3_MODEL_CHECKPOINT_SHA256,
    DA3_MODEL_CONFIG_SHA256,
    DA3_SOURCE_REVISION,
)
from vidmap.frontend.options.depth import Da3VideoOptions, DepthEstimationOptions
from vidmap.frontend.paths import FrontendPaths
from vidmap.utils.image_sampling import sample_at_keypoints
from vidmap.utils.logging import progress_bars_enabled

_RUNTIME_CONFIG_FIELDS = frozenset({"num_workers"})

logger = logging.getLogger(__name__)


def _release_depth_model(model) -> None:
    """Release an owned depth model without masking an active frontend failure."""
    active_error = sys.exc_info()[0] is not None
    cleanup_error = None
    if model is not None:
        try:
            model.cpu()
        except Exception as error:
            cleanup_error = error
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception as error:
        cleanup_error = cleanup_error or error
    if cleanup_error is not None and not active_error:
        raise cleanup_error


def da3_cache_identity(backend: Da3VideoOptions) -> dict:
    """Return the DA3 implementation identity that can affect depth payloads."""
    return {
        "config": semantic_config(backend),
        "source_revision": DA3_SOURCE_REVISION,
        "config_sha256": DA3_MODEL_CONFIG_SHA256,
        "checkpoint_sha256": DA3_MODEL_CHECKPOINT_SHA256,
    }


def _depth_options_cache_config(options):
    return semantic_config(options, exclude_fields=_RUNTIME_CONFIG_FIELDS)


def _prepare_full_depth_cache(
    path: Path,
    *,
    image_names,
    image_content_fingerprint: str,
    options: DepthEstimationOptions,
    overwrite: bool,
):
    """Prepare one resumable confidence-free full-depth cache."""
    metadata = cache_metadata(
        stage="full_depth",
        config={
            "options": _depth_options_cache_config(options),
            "backend": da3_cache_identity(options.backend),
            "native_resolution": True,
        },
        ordered_inputs={
            "images": image_names,
            "image_content": image_content_fingerprint,
        },
        payload_format="per-image-full-depth-v2",
        nonsemantic_config_fields=_RUNTIME_CONFIG_FIELDS,
    )
    prepare_incremental_cache(path, metadata, overwrite=overwrite)
    prune_incremental_items(path, image_names)
    _, missing = inspect_incremental_items(
        path,
        image_names,
        metadata,
        repair_malformed=True,
    )
    return metadata, missing


def _certify_full_depth_cache(path: Path, metadata, image_names) -> IncrementalArtifactContract:
    mark_incremental_cache_complete(path, metadata, image_names)
    return certify_incremental_artifact(path, metadata, image_names)


@dataclass(frozen=True)
class DepthFrontendResult:
    """Certified sampled depths and an optional full-map cache."""

    sampled: IncrementalArtifactContract
    full: IncrementalArtifactContract | None = None


def write_sampled_depth_cache(image_result, depth_path):
    """Sample predicted depth at image keypoints and write the depth, validity, and confidence H5 datasets."""

    image_name, depth_data = image_result
    depth_map = depth_data["depth_map"]  # (H, W)
    valid_map = depth_data["valid_map"].astype(np.float32)  # (H, W)
    conf_map = depth_data.get("conf_map", None)  # (H, W) or None
    keypoints = depth_data["keypoints"]  # (M, 2) in original image coordinates
    original_size = depth_data["original_size"]  # (width, height)

    # Sample depth at keypoint locations
    if len(keypoints) > 0:
        # Calculate scale factors
        depth_h, depth_w = depth_map.shape[:2]
        sx = (depth_w / original_size[0]).astype(np.float32)
        sy = (depth_h / original_size[1]).astype(np.float32)

        # Sample depth and valid at keypoints
        depths_kps = sample_at_keypoints(keypoints, depth_map, sx, sy)
        valid_kps = sample_at_keypoints(keypoints, valid_map, sx, sy, mode="nearest").astype(bool)
        conf_kps = sample_at_keypoints(keypoints, conf_map, sx, sy) if conf_map is not None else None
    else:
        # No keypoints, return empty arrays
        depths_kps = np.array([], dtype=np.float32)
        valid_kps = np.array([], dtype=bool)
        conf_kps = None

    # Save to H5 file
    with h5py.File(str(depth_path), "a", libver="latest") as h5_file_handle:
        if image_name in h5_file_handle:
            del h5_file_handle[image_name]
        image_group = h5_file_handle.create_group(image_name)
        image_group.create_dataset("depth", data=depths_kps)
        image_group.create_dataset("valid", data=valid_kps)
        if conf_kps is not None:
            image_group.create_dataset("conf", data=conf_kps)


def write_full_depth_cache(image_result, depth_path):
    """Write one full prediction-grid depth map and source-image dimensions."""
    image_name, depth_data = image_result
    depth_map = np.asarray(depth_data["depth_map"], dtype=np.float32)
    valid_map = np.asarray(depth_data["valid_map"], dtype=bool)
    original_size = np.asarray(depth_data["original_size"]).reshape(2)
    if depth_map.ndim != 2 or valid_map.shape != depth_map.shape:
        raise ValueError(f"Full depth payload for {image_name!r} has inconsistent shapes")
    with h5py.File(str(depth_path), "a", libver="latest") as h5_file_handle:
        if image_name in h5_file_handle:
            del h5_file_handle[image_name]
        image_group = h5_file_handle.create_group(image_name)
        image_group.create_dataset("depth", data=depth_map)
        image_group.create_dataset("valid", data=valid_map)
        image_group.attrs["original_width"] = int(original_size[0])
        image_group.attrs["original_height"] = int(original_size[1])


def _write_depth_results(image_result, *, sampled_path, full_path):
    """Write explicit optional sampled and full payloads for one image."""
    image_name, sampled_payload, full_payload = image_result
    if sampled_payload is not None:
        write_sampled_depth_cache((image_name, sampled_payload), sampled_path)
    if full_payload is not None:
        if full_path is None:
            raise ValueError("A full depth output path is required")
        write_full_depth_cache((image_name, full_payload), full_path)


def _run_da3_depth(
    scene_parser,
    image_names,
    pending_names,
    depth_path,
    backend,
    sparse_features_path,
    sampled_names=None,
    full_depth_path=None,
    full_names=frozenset(),
    *,
    num_workers,
):
    """Run the ordered DA3 sliding-window inference owned by the depth stage."""
    from vidmap.frontend.models.depth.da3_video import DA3_MODEL_ID, Da3Video

    sampled_names = frozenset(pending_names) if sampled_names is None else sampled_names
    window_size = backend.window_size
    logger.info(
        "Loading DA3 model: %s (window=%d, ref_view=%s, res=%s)",
        DA3_MODEL_ID,
        window_size,
        backend.ref_view_strategy,
        backend.process_res,
    )
    loader = create_da3_window_loader(
        scene_parser.rgb_dir,
        image_names,
        pending_names,
        window_size=window_size,
        process_res=backend.process_res,
        num_workers=num_workers,
    )
    windows = iter(loader)
    model = None
    try:
        model = Da3Video(backend)
        with (
            H5WriteQueue(
                partial(
                    _write_depth_results,
                    sampled_path=depth_path,
                    full_path=full_depth_path,
                )
            ) as writer,
            (
                h5py.File(str(sparse_features_path), "r") if sparse_features_path is not None else nullcontext({})
            ) as features,
            tqdm(
                total=len(pending_names),
                desc="Estimating DA3 depth maps",
                disable=not progress_bars_enabled(),
            ) as progress,
        ):
            for batch in windows:
                name = batch["name"]
                prediction = model.forward_multiview(batch["images"], batch["center_index"])
                keypoints = features[name]["keypoints"][:] if name in features else np.array([])
                original_width, original_height = (int(value) for value in batch["original_size"])
                depth_map = prediction["depth"]
                valid_map = prediction["valid"]
                conf_map = prediction.get("conf", None)
                write_sampled = name in sampled_names
                write_full = name in full_names
                original_size = np.array([original_width, original_height])
                full_payload = (
                    {
                        "depth_map": depth_map,
                        "valid_map": valid_map,
                        "original_size": original_size,
                    }
                    if write_full
                    else None
                )
                if write_sampled and depth_map.shape[:2] != (
                    original_height,
                    original_width,
                ):
                    output_size = (original_width, original_height)
                    depth_map = cv2.resize(depth_map, output_size, interpolation=cv2.INTER_LINEAR)
                    valid_map = cv2.resize(
                        valid_map.astype(np.uint8),
                        output_size,
                        interpolation=cv2.INTER_NEAREST,
                    ).astype(bool)
                    if conf_map is not None:
                        conf_map = cv2.resize(conf_map, output_size, interpolation=cv2.INTER_LINEAR)
                sampled_payload = (
                    {
                        "depth_map": depth_map,
                        "valid_map": valid_map,
                        "conf_map": conf_map,
                        "keypoints": keypoints,
                        "original_size": original_size,
                    }
                    if write_sampled
                    else None
                )
                writer.put(
                    (
                        name,
                        sampled_payload,
                        full_payload,
                    )
                )
                progress.update(1)
    finally:
        _release_depth_model(model)
    if sampled_names:
        logger.info("Sampled DA3 depth maps saved to: %s", depth_path)
    if full_names:
        logger.info("Full DA3 depth maps saved to: %s", full_depth_path)


class DepthEstimator:
    """Own depth metadata, incremental repair, DA3 execution, and cleanup."""

    def __init__(
        self,
        *,
        scene_parser: DatasetParser,
        paths: FrontendPaths,
        force_recompute: bool,
        keyframes: KeyframePlan,
        options: DepthEstimationOptions,
        image_content_fingerprint: str,
    ):
        self.scene_parser = scene_parser
        self.paths = paths
        self.force_recompute = force_recompute
        self.keyframes = keyframes
        self.options = options
        self.image_content_fingerprint = image_content_fingerprint

    def estimate(self, *, cache_full_depth_maps: bool = False) -> DepthFrontendResult:
        from vidmap.utils.profiling import log_memory, record_timing, sync_time

        image_names = self.keyframes.names
        logger.info("Estimating depth maps for %d keyframes", len(image_names))
        started = sync_time()
        sparse_features_metadata = read_cache_metadata(self.paths.sparse_features_path)
        metadata = cache_metadata(
            stage="depth",
            config={
                "options": _depth_options_cache_config(self.options),
                "backend": da3_cache_identity(self.options.backend),
            },
            ordered_inputs={
                "images": image_names,
                "image_content": self.image_content_fingerprint,
            },
            upstream={"sparse_features": artifact_fingerprint(sparse_features_metadata)},
            payload_format="per-image-keypoint-depth",
            nonsemantic_config_fields=_RUNTIME_CONFIG_FIELDS,
        )
        depth_path = self.paths.depth_maps_path
        full_depth_path = self.paths.full_depth_maps_path
        sparse_features_path = self.paths.sparse_features_path
        if not sparse_features_path.exists():
            raise FileNotFoundError(f"Features file not found: {sparse_features_path}")

        prepare_incremental_cache(
            depth_path,
            metadata,
            overwrite=self.force_recompute,
        )
        prune_incremental_items(depth_path, image_names)
        _, sampled_pending_names = inspect_incremental_items(
            depth_path,
            image_names,
            metadata,
            repair_malformed=True,
        )
        full_pending_names = []
        full_metadata = None
        if cache_full_depth_maps:
            if full_depth_path is None:
                raise ValueError("Full depth-map frontend path is unavailable")
            full_metadata, full_pending_names = _prepare_full_depth_cache(
                full_depth_path,
                image_names=image_names,
                image_content_fingerprint=self.image_content_fingerprint,
                options=self.options,
                overwrite=self.force_recompute,
            )
        sampled_names = frozenset(sampled_pending_names)
        full_names = frozenset(full_pending_names)
        pending_names = [name for name in image_names if name in sampled_names or name in full_names]
        if pending_names:
            logger.info("Estimating depth maps for %d images", len(pending_names))
            _run_da3_depth(
                self.scene_parser,
                image_names,
                pending_names,
                depth_path,
                self.options.backend,
                sparse_features_path,
                sampled_names,
                full_depth_path if cache_full_depth_maps else None,
                full_names,
                num_workers=self.options.num_workers,
            )
        else:
            logger.info("No depth maps to estimate; all already exist")

        mark_incremental_cache_complete(depth_path, metadata, image_names)
        full_artifact = None
        if cache_full_depth_maps:
            assert full_metadata is not None and full_depth_path is not None
            full_artifact = _certify_full_depth_cache(full_depth_path, full_metadata, image_names)
        record_timing("depth_frontend", sync_time() - started)
        log_memory("depth_frontend")
        return DepthFrontendResult(
            sampled=certify_incremental_artifact(depth_path, metadata, image_names),
            full=full_artifact,
        )


def cache_full_depth_maps_posthoc(
    *,
    scene_parser: DatasetParser,
    image_names,
    sampled_depth_path: Path,
    output_path: Path,
    options: DepthEstimationOptions,
    force_recompute: bool = False,
) -> IncrementalArtifactContract:
    """Infer only full prediction-grid depths for an existing frontend."""
    from vidmap.frontend.cache import ordered_files_fingerprint

    image_names = tuple(image_names)
    if not image_names:
        raise ValueError("Post-hoc full-depth frontend requires at least one finalized keyframe")
    output_path = Path(output_path)
    metadata, missing = _prepare_full_depth_cache(
        output_path,
        image_names=image_names,
        image_content_fingerprint=ordered_files_fingerprint(scene_parser.rgb_dir, image_names),
        options=options,
        overwrite=force_recompute,
    )
    if missing:
        logger.info("Post-hoc full-depth frontend: processing %d keyframes", len(missing))
        _run_da3_depth(
            scene_parser,
            image_names,
            missing,
            sampled_depth_path,
            options.backend,
            None,
            frozenset(),
            output_path,
            frozenset(missing),
            num_workers=options.num_workers,
        )
    else:
        logger.info(
            "Post-hoc full-depth frontend: all %d keyframes already cached",
            len(image_names),
        )
    return _certify_full_depth_cache(output_path, metadata, image_names)
