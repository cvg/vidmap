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
DA3_SOURCE_REVISION = "2c21ea849ceec7b469a3e62ea0c0e270afc3281a"
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
def _verify_da3_model_snapshot(model_id: str, revision: str) -> None:
    expected = {
        "config.json": DA3_MODEL_CONFIG_SHA256,
        "model.safetensors": DA3_MODEL_CHECKPOINT_SHA256,
    }
    for filename, expected_sha256 in expected.items():
        path = hf_hub_download(repo_id=model_id, filename=filename, revision=revision)
        actual = file_fingerprint(path)
        if actual != expected_sha256:
            raise RuntimeError(f"DA3 {filename} has sha256 {actual}, expected {expected_sha256}")


class Da3Video(torch.nn.Module):
    """
    Depth Anything 3 multi-view depth estimation with sliding window.

    Processes multiple frames at once using DA3's global attention to produce
    temporally consistent depth.

    Uses pretrained weights from HuggingFace Hub.

    Usage:
        model = Da3Video(Da3VideoOptions())
        images = prepared_window  # normalized tensor (N, 3, H, W)
        out = model.forward_multiview(images, center_idx=1)
        depth_B = out["depth"]  # (H, W) metric depth in meters
    """

    def __init__(self, conf: Da3VideoOptions):
        super().__init__()
        assert isinstance(conf, Da3VideoOptions), f"Expected Da3VideoOptions, got {type(conf).__name__}"
        self.conf = conf
        with optional_xformers_disabled():
            import_model_package("depth_anything_3", DA3_PACKAGE_ROOT)
            _configure_da3_logging()
            _verify_da3_model_snapshot(DA3_MODEL_ID, DA3_MODEL_REVISION)
            self.model = Da3Inference.from_pretrained(DA3_MODEL_ID, revision=DA3_MODEL_REVISION)
        self.model = self.model.cuda().eval()
        for parameter in self.parameters():
            parameter.requires_grad = False

    def forward(self, images: torch.Tensor):
        """Infer one prepared image."""
        return self.forward_multiview(images, 0)

    def forward_multiview(self, images: torch.Tensor, center_idx: int):
        """
        Process one prepared image window and return depth for its center frame.

        Args:
            images: Normalized tensor with shape (N, 3, H, W).
            center_idx: Index of the center frame to estimate depth for
        Returns:
            dict with:
                - depth: (H, W) float32 depth map for center frame
                - conf: (H, W) float32 confidence map for center frame
                - valid: (H, W) bool validity mask
        """
        depths, confidences = self.model.infer(
            images,
            ref_view_strategy=self.conf.ref_view_strategy,
        )

        depth = depths[center_idx]
        conf = confidences[center_idx] if confidences is not None else np.ones_like(depth)
        valid = (depth > 0) & np.isfinite(depth)
        depth[np.isinf(depth)] = 2.0

        return {"depth": depth, "conf": conf, "valid": valid}
