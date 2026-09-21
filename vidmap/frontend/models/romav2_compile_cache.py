"""RoMa saved-program identity and image/feature input contract."""

import sys
from collections.abc import Callable
from dataclasses import is_dataclass
from pathlib import Path
from types import SimpleNamespace

import torch

from vidmap.frontend.models.compiled_graph import CachedGraph


class CachedRoMaGraph(CachedGraph):
    """Lazily capture or load a tensor-only RoMa component."""

    bidirectional = False
    threshold = None

    def __init__(
        self, factory: Callable[[], torch.nn.Module], model_identity: dict, source_root: Path, *, component: str
    ):
        sources = [
            *sorted(source_root.rglob("*.py")),
            Path(__file__),
            Path(__file__).with_name("romav2.py"),
            Path(__file__).with_name("romav2_inference.py"),
        ]
        super().__init__(
            factory=factory, namespace="romav2", component=component, model_identity=model_identity, sources=sources
        )

    def forward(self, *inputs: torch.Tensor) -> dict:
        if torch.is_autocast_enabled("cuda"):
            raise ValueError("Saved RoMa graphs require outer autocast to be disabled")
        if not inputs or any(
            value.device != torch.device("cuda", self.runtime["device"])
            or value.dtype not in (torch.float32, torch.bfloat16)
            or not value.is_contiguous()
            or not value.is_inference()
            for value in inputs
        ):
            raise ValueError("RoMa components require contiguous float32 or bfloat16 CUDA inference tensors")
        if self.identity["component"] == "whole_model" and (
            len(inputs) not in (2, 4)
            or any(value.dtype != torch.float32 or value.ndim != 4 or value.shape[1] != 3 for value in inputs)
        ):
            raise ValueError("RoMa requires two or four float32 image tensors")
        return super().forward(*inputs)

    def load_module(self, path):
        descriptor_source = (
            Path(torch.hub.get_dir()) / "facebookresearch_dinov3_adc254450203739c8149213a7a69d8d905b4fcfa"
        )
        if not descriptor_source.is_dir():
            raise FileNotFoundError(f"Cached RoMa requires its pinned DINOv3 source: {descriptor_source}")
        sys.path.insert(0, str(descriptor_source))
        return super().load_module(path)

    def prepare_module(self, net):
        # PyTorch's portable type guards require globally importable config types.
        for module in net.modules():
            if "cfg" in module.__dict__ and is_dataclass(module.cfg):
                module.cfg = SimpleNamespace(**vars(module.cfg))
        return net
