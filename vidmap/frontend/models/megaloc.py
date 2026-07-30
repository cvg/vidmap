"""
Code to use MegaLoc as a global descriptor model.

MegaLoc paper: https://arxiv.org/abs/2502.17237
"""

import importlib.util

import torch
import torchvision.transforms as tvf
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

from vidmap.frontend.cache import file_fingerprint
from vidmap.model_sources import model_package_root

MEGALOC_SOURCE_REVISION = "1af071c68fc3ab6c6018c5c868391763516e50f7"
MEGALOC_MODEL_REVISION = "7cb9f7970d366fdf059963d04d372e503e8e9df9"
MEGALOC_MODEL_SHA256 = "d4f9f2bcb60018f91eb6a8e061ed054fd55654e10c2569cf13841ea986ffb4f8"
MEGALOC_SOURCE = model_package_root("MegaLoc", "third_party/MegaLoc")
MEGALOC_SOURCE_FILE = MEGALOC_SOURCE / "megaloc_model.py"


def _create_megaloc_model():
    """Construct MegaLoc from VidMap's pinned packaged source."""
    spec = importlib.util.spec_from_file_location("vidmap_pinned_megaloc", MEGALOC_SOURCE_FILE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import pinned MegaLoc source from {MEGALOC_SOURCE_FILE}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.MegaLoc()


class MegaLocDescriptorModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        pinned_weights = hf_hub_download(
            repo_id="gberton/MegaLoc",
            filename="model.safetensors",
            revision=MEGALOC_MODEL_REVISION,
        )
        actual = file_fingerprint(pinned_weights)
        if actual != MEGALOC_MODEL_SHA256:
            raise RuntimeError(
                f"MegaLoc weights at {pinned_weights} have sha256 {actual}, expected {MEGALOC_MODEL_SHA256}"
            )
        self.net = _create_megaloc_model()
        self.net.load_state_dict(load_file(pinned_weights))
        self.net.eval()
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        self.norm_rgb = tvf.Normalize(mean=mean, std=std)
        for parameter in self.parameters():
            parameter.requires_grad = False

    def forward(self, data):
        assert "image" in data, "Missing key image in data"
        image = self.norm_rgb(data["image"])
        desc = self.net(image)
        return {
            "global_descriptor": desc,
        }


def megaloc_cache_identity():
    return {
        "source_revision": MEGALOC_SOURCE_REVISION,
        "source_sha256": file_fingerprint(MEGALOC_SOURCE_FILE),
        "model_revision": MEGALOC_MODEL_REVISION,
        "model_sha256": MEGALOC_MODEL_SHA256,
    }
