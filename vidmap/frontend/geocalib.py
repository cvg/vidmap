"""GeoCalib frontend and shared calibration helpers."""

from __future__ import annotations

import logging
import sys
from functools import cache, partial
from importlib.metadata import version
from pathlib import Path
from typing import Any, Mapping, Sequence

import h5py
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from vidmap.frontend.cache import (
    IncrementalArtifactContract,
    artifact_fingerprint,
    cache_metadata,
    certify_incremental_artifact,
    file_fingerprint,
    fingerprint,
    inspect_incremental_items,
    mark_incremental_cache_complete,
    prepare_incremental_cache,
    read_cache_metadata,
)
from vidmap.frontend.h5_write_queue import H5WriteQueue
from vidmap.frontend.options.preparation import CameraPriorEstimationOptions
from vidmap.utils.logging import progress_bars_enabled

_RUNTIME_CONFIG_FIELDS = frozenset({"num_workers"})
GEOCALIB_CHECKPOINT_SHA256 = "86d6aeacd8bbd974c59ce39f61854e00d36911c732ad89be471476fd708722ac"
KEYFRAME_BOOTSTRAP_SAMPLE_COUNT = 30
KEYFRAME_BOOTSTRAP_SAMPLING_POLICY = "uniform-sequence-index-v1"

logger = logging.getLogger(__name__)


@cache
def geocalib_cache_identity() -> dict[str, str]:
    """Describe the installed model implementation for cache provenance."""
    import geocalib

    actual_version = version("geocalib")
    package_root = Path(geocalib.__file__).resolve().parent
    source_files = [
        [path.relative_to(package_root).as_posix(), file_fingerprint(path)]
        for path in sorted(package_root.rglob("*.py"))
    ]
    return {
        "package_version": actual_version,
        "source_sha256": fingerprint(source_files),
        "checkpoint_sha256": GEOCALIB_CHECKPOINT_SHA256,
    }


def _load_geocalib_model():
    from geocalib import GeoCalib

    # GeoCalib downloads its release checkpoint on first construction.
    model = GeoCalib()
    _verify_geocalib_checkpoint()
    try:
        return model.to("cuda")
    except Exception:
        _release_geocalib_model(model)
        raise


def _release_geocalib_model(model) -> None:
    """Release an owned GeoCalib model without masking an active stage failure."""
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


def keyframe_bootstrap_sample_plan(
    sequence: Sequence[str],
) -> tuple[list[int], list[str]]:
    """Select the fixed ordered full-video sample used for keyframe calibration."""
    sequence_length = len(sequence)
    if sequence_length == 0:
        raise ValueError("Keyframe GeoCalib bootstrap requires a non-empty image sequence")
    if sequence_length <= KEYFRAME_BOOTSTRAP_SAMPLE_COUNT:
        indices = list(range(sequence_length))
    else:
        indices = np.linspace(
            0,
            sequence_length - 1,
            KEYFRAME_BOOTSTRAP_SAMPLE_COUNT,
            dtype=np.int64,
        ).tolist()
    if len(indices) != len(set(indices)) or indices != sorted(indices):
        raise RuntimeError(f"Invalid keyframe GeoCalib sample plan: {indices}")
    return indices, [sequence[index] for index in indices]


def keyframe_bootstrap_cache_identity(
    sequence: Sequence[str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return the semantic configuration and ordered inputs for bootstrap calibration."""
    indices, images = keyframe_bootstrap_sample_plan(sequence)
    return (
        {
            "model": geocalib_cache_identity(),
            "sample_count": KEYFRAME_BOOTSTRAP_SAMPLE_COUNT,
            "sampling_policy": KEYFRAME_BOOTSTRAP_SAMPLING_POLICY,
            "shared_intrinsics": True,
        },
        {"indices": indices, "images": images},
    )


def validate_keyframe_bootstrap_frame_dimensions(rgb_dir: Path, sequence: Sequence[str]) -> tuple[int, int]:
    expected_size = None
    expected_name = None
    for name in sequence:
        with Image.open(Path(rgb_dir) / name) as image:
            size = image.size
        if expected_size is None:
            expected_size = size
            expected_name = name
        elif size != expected_size:
            raise ValueError(
                "Keyframe GeoCalib bootstrap requires identical frame dimensions; "
                f"{name} has {size}, expected {expected_size} from {expected_name}"
            )
    if expected_size is None:
        raise ValueError("Keyframe GeoCalib bootstrap requires a non-empty image sequence")
    return expected_size


def _validated_shared_intrinsics(
    result: Mapping[str, Any],
    batch_size: int,
    *,
    expected_image_size: tuple[int, int] | None = None,
) -> np.ndarray:
    if "camera" not in result:
        raise RuntimeError("GeoCalib shared calibration did not return a camera")
    camera = result["camera"]
    calibration = camera.K
    if isinstance(calibration, torch.Tensor):
        calibration = calibration.detach().cpu().numpy()
    calibration = np.asarray(calibration)
    expected_shape = (batch_size, 3, 3)
    if calibration.shape != expected_shape:
        raise RuntimeError(
            f"GeoCalib shared calibration returned K with shape {calibration.shape}, expected {expected_shape}"
        )
    if not np.isfinite(calibration).all():
        raise RuntimeError("GeoCalib shared calibration returned non-finite intrinsics")
    if np.any(calibration[:, 0, 0] <= 0) or np.any(calibration[:, 1, 1] <= 0):
        raise RuntimeError("GeoCalib shared calibration returned non-positive focal lengths")
    homogeneous_row = np.array([0.0, 0.0, 1.0], dtype=calibration.dtype)
    if not np.allclose(calibration[:, 2, :], homogeneous_row, rtol=0.0, atol=1e-6):
        raise RuntimeError("GeoCalib shared calibration returned invalid homogeneous intrinsic matrices")
    if not np.allclose(calibration, calibration[:1], rtol=0.0, atol=1e-6):
        raise RuntimeError("GeoCalib shared calibration returned inconsistent intrinsics across the image stack")
    if expected_image_size is not None:
        camera_size = camera.size
        if isinstance(camera_size, torch.Tensor):
            camera_size = camera_size.detach().cpu().numpy()
        camera_size = np.asarray(camera_size)
        expected_sizes = np.repeat(np.asarray(expected_image_size)[None], batch_size, axis=0)
        if camera_size.shape != expected_sizes.shape or not np.allclose(
            camera_size,
            expected_sizes,
            rtol=0.0,
            atol=1e-3,
        ):
            raise RuntimeError(
                f"GeoCalib shared calibration returned image size {camera_size.tolist()}, "
                f"expected {expected_sizes.tolist()} in raw-image pixel coordinates"
            )
    return calibration


def calibrate_shared_intrinsics(model, image_paths: Sequence[Path], *, device: str = "cuda") -> Mapping[str, Any]:
    """Load one same-sized raw-image stack and run one shared GeoCalib forward pass."""
    if not image_paths:
        raise ValueError("GeoCalib shared calibration requires at least one image")
    images = []
    expected_shape = None
    for image_path in image_paths:
        image = model.load_image(Path(image_path))
        if not isinstance(image, torch.Tensor):
            raise TypeError(f"GeoCalib image {image_path} returned {type(image).__name__}, expected a tensor")
        if image.ndim != 3 or image.shape[0] != 3:
            raise ValueError(
                f"GeoCalib image {image_path} has incompatible tensor shape {tuple(image.shape)}; expected (3, H, W)"
            )
        shape = tuple(image.shape)
        if expected_shape is None:
            expected_shape = shape
        elif shape != expected_shape:
            raise ValueError(
                f"GeoCalib shared calibration requires identical frame dimensions; "
                f"{image_path} has {shape}, expected {expected_shape}"
            )
        images.append(image.to(device))

    batch = torch.stack(images).to(device)
    result = model.calibrate(batch, shared_intrinsics=True)
    if not isinstance(result, Mapping):
        raise RuntimeError(f"GeoCalib shared calibration returned {type(result).__name__}, expected a mapping")
    height, width = expected_shape[1:]
    _validated_shared_intrinsics(result, len(images), expected_image_size=(width, height))
    return result


@torch.no_grad()
def estimate_keyframe_bootstrap_intrinsics(rgb_dir: Path, sequence: Sequence[str]) -> np.ndarray:
    """Estimate the local K used only by normalized keyframe motion scoring."""
    indices, images = keyframe_bootstrap_sample_plan(sequence)
    model = _load_geocalib_model()
    try:
        result = calibrate_shared_intrinsics(model, [Path(rgb_dir) / name for name in images])
        calibration = _validated_shared_intrinsics(result, len(images))[0].astype(np.float64, copy=True)
    finally:
        _release_geocalib_model(model)
    logger.info(
        "Keyframe GeoCalib bootstrap: indices=%s images=%s K=%s",
        indices,
        images,
        calibration.tolist(),
    )
    return calibration


@cache
def _verify_geocalib_checkpoint() -> None:
    checkpoint = Path(torch.hub.get_dir()) / "geocalib/pinhole.tar"
    actual = file_fingerprint(checkpoint)
    if actual != GEOCALIB_CHECKPOINT_SHA256:
        raise RuntimeError(f"GeoCalib checkpoint has sha256 {actual}, expected {GEOCALIB_CHECKPOINT_SHA256}")


def write_image_geocalib_cache(image_result, geocalib_per_image_path):
    """Write one image's focal-uncertainty and confidence datasets to the GeoCalib H5 cache."""
    image_name, geocalib_result = image_result
    focal_uncertainty = geocalib_result["focal_uncertainty"]
    confidence = geocalib_result["confidence"]

    # Save individual result to per-image file
    with h5py.File(str(geocalib_per_image_path), "a", libver="latest") as h5_file_handle:
        if image_name in h5_file_handle:
            del h5_file_handle[image_name]
        image_group = h5_file_handle.create_group(image_name)
        image_group.create_dataset("focal_uncertainty", data=np.array(focal_uncertainty, dtype=np.float32))
        image_group.create_dataset("confidence", data=np.array(confidence, dtype=np.float32))


class CameraPriorEstimator:
    """Own per-image repair, shared calibration, metadata, and model lifetime."""

    def __init__(
        self,
        *,
        rgb_dir: Path,
        per_image_path: Path,
        batch_path: Path,
        force_recompute: bool,
        keyframe_names: Sequence[str],
        options: CameraPriorEstimationOptions,
        image_content_fingerprint: str,
    ):
        self.rgb_dir = Path(rgb_dir)
        self.per_image_path = per_image_path
        self.batch_path = batch_path
        self.force_recompute = force_recompute
        self.keyframe_names = tuple(keyframe_names)
        self.options = options
        self.image_content_fingerprint = image_content_fingerprint

    @torch.no_grad()
    def _repair_per_image(self, image_names, metadata) -> None:
        path = self.per_image_path
        present, _ = inspect_incremental_items(path, image_names, metadata, repair_malformed=True)
        present_names = set(present)
        pending_names = [name for name in image_names if name not in present_names]
        if not pending_names:
            logger.info("All individual geo-calibration results already exist")
            return

        logger.info("Computing focal uncertainty for %d images", len(pending_names))
        model = None
        try:
            model = _load_geocalib_model()
            with (
                H5WriteQueue(partial(write_image_geocalib_cache, geocalib_per_image_path=path)) as writer,
                tqdm(
                    total=len(pending_names),
                    desc="Computing focal uncertainty",
                    disable=not progress_bars_enabled(),
                ) as progress,
            ):
                for image_name in pending_names:
                    image = model.load_image(Path(self.rgb_dir) / image_name).to("cuda")
                    result = model.calibrate(image, shared_intrinsics=False)
                    focal_uncertainty = float(result["focal_uncertainty"].cpu().numpy().reshape(-1)[0])
                    writer.put(
                        (
                            image_name,
                            {
                                "focal_uncertainty": focal_uncertainty,
                                "confidence": float(1 / (focal_uncertainty**0.5)),
                            },
                        )
                    )
                    progress.update(1)
        finally:
            _release_geocalib_model(model)

    @torch.no_grad()
    def _repair_batch(self, image_names, metadata) -> None:
        batch_path = self.batch_path
        present, _ = inspect_incremental_items(
            batch_path,
            ["batch_calibration"],
            metadata,
            repair_malformed=True,
        )
        if present:
            logger.info("Batch calibration already exists; skipping")
            return

        confidences = []
        with h5py.File(str(self.per_image_path), "r") as hfile:
            for name in image_names:
                if name in hfile:
                    confidences.append((name, float(hfile[name]["confidence"][()])))
        confidences.sort(key=lambda item: item[1], reverse=True)
        selected = [name for name, _ in confidences[: self.options.max_images]]
        if not selected:
            logger.info("No images available for batch calibration")
            return

        model = None
        try:
            model = _load_geocalib_model()
            result = calibrate_shared_intrinsics(
                model,
                [self.rgb_dir / name for name in selected],
            )
            focal = result["camera"].f.cpu().numpy()[0]
            principal_point = result["camera"].c.cpu().numpy()[0]
            focal_uncertainty = result["focal_uncertainty"].cpu().numpy()
            with h5py.File(str(batch_path), "a", libver="latest") as hfile:
                if "batch_calibration" in hfile:
                    del hfile["batch_calibration"]
                group = hfile.create_group("batch_calibration")
                group.create_dataset("focal", data=focal.astype(np.float32))
                group.create_dataset("principal_point", data=principal_point.astype(np.float32))
                group.create_dataset("focal_uncertainty", data=focal_uncertainty.astype(np.float32))
                group.create_dataset(
                    "topk_images",
                    data=np.array([name.encode("utf-8") for name in selected], dtype=object),
                )
        finally:
            _release_geocalib_model(model)
        logger.info(
            "Batch calibration complete: focal=%s principal=%s",
            [int(value) for value in focal],
            [int(value) for value in principal_point],
        )

    def estimate(self) -> IncrementalArtifactContract:
        from vidmap.utils.profiling import log_memory, record_timing, sync_time

        image_names = self.keyframe_names
        logger.info("Estimating camera priors for %d keyframes", len(image_names))
        started = sync_time()
        per_image_metadata = cache_metadata(
            stage="geocalib_per_image",
            config={
                "model": geocalib_cache_identity(),
            },
            ordered_inputs={
                "images": image_names,
                "image_content": self.image_content_fingerprint,
            },
            payload_format="per-image-geocalibration",
            nonsemantic_config_fields=_RUNTIME_CONFIG_FIELDS,
        )
        prepare_incremental_cache(
            self.per_image_path,
            per_image_metadata,
            overwrite=self.force_recompute,
        )
        self._repair_per_image(image_names, per_image_metadata)
        mark_incremental_cache_complete(
            self.per_image_path,
            per_image_metadata,
            image_names,
        )
        batch_metadata = cache_metadata(
            stage="geocalib_batch",
            config={"options": {"topk": self.options.max_images}, "model": geocalib_cache_identity()},
            ordered_inputs={"images": image_names},
            upstream={"geocalib_per_image": artifact_fingerprint(read_cache_metadata(self.per_image_path))},
            payload_format="shared-intrinsics-geocalibration",
            nonsemantic_config_fields=_RUNTIME_CONFIG_FIELDS,
        )
        prepare_incremental_cache(
            self.batch_path,
            batch_metadata,
            overwrite=self.force_recompute,
        )
        logger.info("Selecting top-k images and batch calibrating")
        self._repair_batch(image_names, batch_metadata)
        mark_incremental_cache_complete(
            self.batch_path,
            batch_metadata,
            ["batch_calibration"],
        )
        logger.info("Per-image geo-calibration saved to: %s", self.per_image_path)
        logger.info("Batch geo-calibration saved to: %s", self.batch_path)
        record_timing("geocalib", sync_time() - started)
        log_memory("geocalib")
        return certify_incremental_artifact(self.batch_path, batch_metadata, ("batch_calibration",))
