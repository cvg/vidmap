"""DA3 saved-program identity and backbone/head input contract."""

import sys
from importlib.metadata import version
from pathlib import Path

import torch
from torch.utils._pytree import tree_leaves

from vidmap.frontend.models.compiled_graph import CachedGraph


class CachedDa3Graph(CachedGraph):
    """Save/load one DA3 backbone or head, retaining the ordinary model owner."""

    def __init__(self, net, *, component, model_identity, source_root):
        self.component = component
        if component.endswith("head"):
            net.plain_output = True
        sources = [*sorted(source_root.rglob("*.py")), *sorted(Path(__file__).parent.glob("da3_*.py"))]
        super().__init__(
            net=net,
            namespace="da3",
            component=component,
            model_identity=model_identity,
            sources=sources,
            extra_identity={
                "inference_tensors": [value.is_inference() for value in [*net.parameters(), *net.buffers()]],
                "dependencies": {
                    "einops": version("einops"),
                    "xformers": version("xformers") if "xformers" in sys.modules else None,
                },
                "module_types": sorted(
                    {
                        "|".join(f"{base.__module__}.{base.__qualname__}" for base in type(module).__mro__)
                        for module in net.modules()
                    }
                ),
            },
        )

    def forward(self, *args, **kwargs):
        allowed = (
            {"cam_token", "export_feat_layers", "ref_view_strategy"}
            if self.component.endswith("backbone")
            else {"patch_start_idx"}
        )
        assert not set(kwargs) - allowed, set(kwargs) - allowed
        inputs = [value for value in tree_leaves((args, kwargs)) if isinstance(value, torch.Tensor)]
        assert inputs and all(value.device == torch.device("cuda", self.runtime["device"]) for value in inputs)
        result = super().forward(*args, **kwargs)
        if self.component.endswith("head"):
            from addict import Dict

            return Dict(result)
        return result
