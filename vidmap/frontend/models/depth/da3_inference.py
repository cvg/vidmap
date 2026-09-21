"""Inference-only Depth Anything 3 interface.

Depth expressions derive from DA3, Copyright (c) 2025 ByteDance Ltd. and/or its
affiliates, Apache-2.0. See third_party/Depth-Anything-3/LICENSE.
"""

from __future__ import annotations

import torch
from accelerate import init_empty_weights
from huggingface_hub import PyTorchModelHubMixin
from safetensors.torch import load_file


class Da3Inference(torch.nn.Module, PyTorchModelHubMixin):
    """DA3 model construction, checkpoint loading, and inference used by VidMap."""

    def __init__(self, model_name: str = "da3-large"):
        super().__init__()
        from depth_anything_3.cfg import create_object, load_config
        from depth_anything_3.registry import MODEL_REGISTRY

        # Buffers keep their upstream initialization; checkpoint parameters do
        # not need disposable random values before strict assignment.
        with init_empty_weights(include_buffers=False):
            self.model = create_object(load_config(MODEL_REGISTRY[model_name]))
        self.model.eval()
        head = self.model.da3.head
        assert self.model.da3.cam_dec is not None
        assert head.head_main == "depth" and head.head_aux == "ray"
        head.compute_rays = False
        self.encoders = None

    @classmethod
    def _load_as_safetensor(cls, model, model_file, map_location, strict):
        """Assign verified checkpoint tensors, including upstream shared aliases."""
        assert map_location == "cpu"
        state = load_file(model_file, device="cpu")
        aliases = {}
        for name, parameter in model.named_parameters(remove_duplicate=False):
            aliases.setdefault(id(parameter), []).append(name)
        for names in aliases.values():
            present = [name for name in names if name in state]
            if not present:
                raise ValueError(f"DA3 checkpoint is missing parameters: {names}")
            if any(not torch.equal(state[name], state[present[0]]) for name in present[1:]):
                raise ValueError(f"DA3 checkpoint has conflicting shared parameters: {present}")
            for name in names:
                if name not in state:
                    state[name] = state[present[0]]
        model.load_state_dict(state, strict=True, assign=True)
        # assign=True creates Parameter objects. Restore shared identity before
        # upload so aliases remain tied and do not allocate duplicate weights.
        for names in aliases.values():
            parameter = model.get_parameter(names[0])
            for name in names[1:]:
                parent, _, leaf = name.rpartition(".")
                model.get_submodule(parent).register_parameter(leaf, parameter)
        assert not any(value.is_meta for value in (*model.parameters(), *model.buffers()))
        return model

    def configure_runtime(self, *, compile: bool) -> None:
        assert self.encoders is None
        self.encoders = []
        if compile:
            from types import MethodType

            from depth_anything_3.model.dinov2.layers.swiglu_ffn import SwiGLUFFN, SwiGLUFFNFused

            from vidmap.frontend.models.depth.da3_compile_cache import CachedDa3Graph
            from vidmap.frontend.models.depth.da3_video import (
                DA3_MODEL_CHECKPOINT_SHA256,
                DA3_MODEL_CONFIG_SHA256,
                DA3_PACKAGE_ROOT,
                DA3_SOURCE_REVISION,
            )

            identity = {
                "checkpoint": DA3_MODEL_CHECKPOINT_SHA256,
                "config": DA3_MODEL_CONFIG_SHA256,
                "source": DA3_SOURCE_REVISION,
            }
            for name, branch in (("anyview", self.model.da3), ("metric", self.model.da3_metric)):
                encoder = branch.backbone.pretrained
                for module in encoder.modules():
                    if type(module) is SwiGLUFFNFused:
                        module.forward = MethodType(SwiGLUFFN.forward, module)
                if encoder.rope is not None:
                    self.encoders.append(encoder)
                branch.backbone = CachedDa3Graph(
                    branch.backbone,
                    component=f"{name}.backbone",
                    model_identity=identity,
                    source_root=DA3_PACKAGE_ROOT,
                )
                branch.head = CachedDa3Graph(
                    branch.head, component=f"{name}.head", model_identity=identity, source_root=DA3_PACKAGE_ROOT
                )

    @torch.inference_mode()
    def infer_windows(self, windows, *, batch_size: int, ref_view_strategy: str):
        assert self.encoders is not None and 0 < len(windows) <= batch_size
        assert all(image.shape == windows[0].shape for image in windows)
        device = next(self.parameters()).device
        batch = torch.stack(windows).to(device, non_blocking=True).float()
        if len(windows) < batch_size:
            batch = torch.cat((batch, batch.new_zeros((batch_size - len(windows), *batch.shape[1:]))))
        outputs = self._infer_batch(batch, count=len(windows), ref_view_strategy=ref_view_strategy)
        return [
            (
                output["depth"].squeeze(0).squeeze(-1).cpu().numpy(),
                output["depth_conf"].squeeze(0).cpu().numpy(),
                output["intrinsics"].squeeze(0).float().cpu().numpy(),
            )
            for output in outputs
        ]

    def _infer_batch(self, images, *, count, ref_view_strategy):
        assert images.ndim == 5 and images.shape[2] == 3 and 0 < count <= images.shape[0]
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        # Several independent shapes/branches share upstream Python code objects.
        # Keep specialization explicit instead of hitting Dynamo's default eager fallback.
        with (
            torch._dynamo.config.patch(recompile_limit=64, fail_on_recompile_limit_hit=True),
            torch.autocast("cuda", dtype=dtype),
        ):
            # The upstream outer product is autocast-sensitive, even when the
            # requested rotary table dtype is float32. Preserve its context.
            for encoder in self.encoders:
                height, width = (size // encoder.patch_size for size in images.shape[-2:])
                encoder.position_getter(1, height, width, images.device)
                rope = encoder.rope
                rope.max_position = max(height, width) + int(encoder.patch_start_idx > 0)
                dimension = encoder.embed_dim // encoder.num_heads // 2
                for table_dtype in (dtype, torch.float32):
                    frequencies = rope._compute_frequency_components(
                        dimension, rope.max_position, images.device, table_dtype
                    )
                    for name, value in zip(("cos", "sin"), frequencies):
                        rope.register_buffer(
                            f"{name}_{str(table_dtype).removeprefix('torch.')}", value, persistent=False
                        )
            anyview = self._neural(self.model.da3, images, ref_view_strategy)
            metric = self._neural(self.model.da3_metric, images, "saddle_balanced")
            outputs = []
            for index in range(count):
                output = self.model.da3._process_mono_sky_estimation(self._window(anyview, index))
                metric_output = self.model.da3_metric._process_mono_sky_estimation(self._window(metric, index))
                output = self.model._apply_metric_scaling(output, metric_output)
                output = self.model._apply_depth_alignment(output, metric_output)
                output = self.model._handle_sky_regions(output, metric_output)
                outputs.append(output)
        return outputs

    def _neural(self, branch, images, strategy):
        features, auxiliary = branch.backbone(
            images, cam_token=None, export_feat_layers=[], ref_view_strategy=strategy
        )
        height, width = images.shape[-2:]
        with torch.autocast(device_type=images.device.type, enabled=False):
            output = branch._process_depth_head(features, height, width)
            output = branch._process_camera_estimation(features, height, width, output)
        assert not auxiliary
        return output

    def _window(self, outputs, index):
        from addict import Dict

        assert all(isinstance(value, torch.Tensor) for value in outputs.values())
        return Dict({key: value[index : index + 1].clone() for key, value in outputs.items()})
