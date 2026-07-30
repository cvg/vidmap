"""Inference-only Depth Anything 3 interface."""

from __future__ import annotations

import numpy as np
import torch
from huggingface_hub import PyTorchModelHubMixin


class Da3Inference(torch.nn.Module, PyTorchModelHubMixin):
    """DA3 model construction, checkpoint loading, and inference used by VidMap."""

    def __init__(self, model_name: str = "da3-large", config: dict[str, object] | None = None):
        super().__init__()
        assert config is None or isinstance(config, dict)
        from depth_anything_3.cfg import create_object, load_config
        from depth_anything_3.registry import MODEL_REGISTRY

        self.model = create_object(load_config(MODEL_REGISTRY[model_name]))
        self.model.eval()

    @torch.inference_mode()
    def infer(
        self,
        images: torch.Tensor,
        *,
        ref_view_strategy: str,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        assert images.ndim == 4 and images.shape[1] == 3
        device = next(self.parameters()).device
        batch = images.to(device, non_blocking=True)[None].float()
        autocast_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        with torch.autocast(device_type=device.type, dtype=autocast_dtype):
            output = self.model(batch, None, None, [], False, False, ref_view_strategy)

        depth = output["depth"].squeeze(0).squeeze(-1).cpu().numpy()
        confidence = output["depth_conf"]
        if confidence is not None:
            confidence = confidence.squeeze(0).cpu().numpy()
        return depth, confidence
