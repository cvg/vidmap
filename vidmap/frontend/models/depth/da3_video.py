"""Depth Anything V3 multi-view depth estimation with sliding window."""

import importlib
import logging
from functools import cache

import numpy as np
import torch
from huggingface_hub import hf_hub_download

from vidmap.frontend.cache import file_fingerprint
from vidmap.frontend.options.depth import Da3VideoOptions
from vidmap.model_sources import import_model_package, model_package_root

from .da3_imports import optional_xformers_disabled
from .da3_inference import Da3Inference

DA3_MODEL_ID = "depth-anything/DA3NESTED-GIANT-LARGE-1.1"
DA3_MODEL_REVISION = "b2359bdf726fb44ef62acca04d629dcf158053e7"
DA3_MODEL_CONFIG_SHA256 = "09adf89474017e717bc05aa86fd3a378708ba8914b036d61874eced328069468"
DA3_MODEL_CHECKPOINT_SHA256 = "8ebe871a022ed58d2fc8fdfb2ebdb31d57b60fe39611c849095851a7b7c6020c"
DA3_SOURCE_REVISION = "b531937f10ae9f3d6f32b6050d13a9b96b3e1c5b"
DA3_PACKAGE_ROOT = model_package_root(
    "depth_anything_3",
    "third_party/Depth-Anything-3/src/depth_anything_3",
)
DA3_SOURCE = DA3_PACKAGE_ROOT.parent


def _configure_da3_logging() -> None:
    """Map DA3's print-based logger onto VidMap's runtime output policy."""
    logger_module = importlib.import_module("depth_anything_3.utils.logger")
    level_name = "DEBUG" if logging.getLogger("vidmap").isEnabledFor(logging.DEBUG) else "WARN"
    logger_module.logger.level = logger_module.LOG_LEVELS[level_name]


@cache
def verify_da3_model_snapshot() -> None:
    expected = {
        "config.json": DA3_MODEL_CONFIG_SHA256,
        "model.safetensors": DA3_MODEL_CHECKPOINT_SHA256,
    }
    for filename, expected_sha256 in expected.items():
        path = hf_hub_download(repo_id=DA3_MODEL_ID, filename=filename, revision=DA3_MODEL_REVISION)
        actual = file_fingerprint(path)
        if actual != expected_sha256:
            raise RuntimeError(f"DA3 {filename} has sha256 {actual}, expected {expected_sha256}")


class Da3Video(torch.nn.Module):
    """Estimate center-frame depth and intrinsics for independent image windows.

    Window records carry normalized frames (N, 3, H, W), a center index, and
    original and uncropped sizes. Windows in one batch share the image shape.
    """

    def __init__(self, conf: Da3VideoOptions):
        super().__init__()
        assert isinstance(conf, Da3VideoOptions), f"Expected Da3VideoOptions, got {type(conf).__name__}"
        self.conf = conf
        with optional_xformers_disabled():
            import_model_package("depth_anything_3", DA3_PACKAGE_ROOT)
            _configure_da3_logging()
            verify_da3_model_snapshot()
            self.model = Da3Inference.from_pretrained(DA3_MODEL_ID, revision=DA3_MODEL_REVISION)
        self.model = self.model.cuda().eval()
        for parameter in self.parameters():
            parameter.requires_grad = False
        self.model.configure_runtime(compile=conf.compile)

    def forward_windows(self, windows, *, batch_size):
        outputs = self.model.infer_windows(
            [window["images"] for window in windows],
            batch_size=batch_size,
            ref_view_strategy=self.conf.ref_view_strategy,
        )
        return [self._prediction(*output, window) for output, window in zip(outputs, windows)]

    def _prediction(self, depths, confidences, intrinsics, window):
        center_idx = window["center_index"]
        depth = depths[center_idx]
        conf = confidences[center_idx]
        valid = (depth > 0) & np.isfinite(depth)
        depth[np.isinf(depth)] = 2.0

        # Map intrinsics from the center crop back to original-image coordinates.
        height, width = depth.shape
        uw, uh = window["uncropped_size"]
        left = int(round((uw - width) / 2))
        top = int(round((uh - height) / 2))

        ow, oh = window["original_size"]
        scale = np.diag(np.asarray([ow / uw, oh / uh, 1.0], dtype=np.float32))
        K = np.asarray(intrinsics[center_idx], dtype=np.float32).copy()
        K[0, 2] += left
        K[1, 2] += top
        K = scale @ K
        calibration = {"K": K, "image_size": tuple(int(n) for n in window["original_size"])}

        return {
            "depth": depth,
            "conf": conf,
            "valid": valid,
            "calibration": calibration,
        }
