from dataclasses import asdict
from functools import cache
from importlib.metadata import version
from pathlib import Path

import lightglue.aliked
import torch
from pydantic import ConfigDict

from vidmap.configuration.validators import dataclass
from vidmap.frontend.cache import file_fingerprint, fingerprint

ALIKED_CHECKPOINT_SHA256 = "5be8704840ed662d9d8c561bf7279c222092674e7eb05fd0feab94899e9d82f2"


@cache
def aliked_cache_identity() -> dict[str, str]:
    """Describe the installed model implementation for cache provenance."""
    actual_version = version("lightglue")
    package_root = Path(lightglue.__file__).resolve().parent
    source_files = [
        [path.relative_to(package_root).as_posix(), file_fingerprint(path)]
        for path in sorted(package_root.rglob("*.py"))
    ]
    return {
        "name": "aliked-n16",
        "checkpoint_sha256": ALIKED_CHECKPOINT_SHA256,
        "package_version": actual_version,
        "source_sha256": fingerprint(source_files),
    }


@cache
def _verify_aliked_checkpoint() -> None:
    path = Path(torch.hub.get_dir()) / "checkpoints/aliked-n16.pth"
    if not path.is_file():
        raise RuntimeError(f"ALIKED checkpoint is unavailable: {path}")
    actual = file_fingerprint(path)
    if actual != ALIKED_CHECKPOINT_SHA256:
        raise RuntimeError(f"ALIKED checkpoint has sha256 {actual}, expected {ALIKED_CHECKPOINT_SHA256}")


@dataclass(frozen=True, config=ConfigDict(extra="forbid"))
class ALIKEDOptions:
    """Typed config for :class:`ALIKED`."""

    model_name: str = "aliked-n16"
    max_num_keypoints: int = -1
    detection_threshold: float = 0.2
    nms_radius: int = 2
    sub_pixel: bool = False


class _VariableLengthALIKED(lightglue.aliked.ALIKED):
    def forward(self, data: dict) -> dict:
        image = data["image"]
        if image.shape[1] == 1:
            image = lightglue.aliked.grayscale_to_rgb(image)
        feature_map, score_map = self.extract_dense_map(image)

        sub_pixel = data.get("sub_pixel", False)
        keypoints, kptscores, scoredispersitys = self.dkd(
            score_map, image_size=data.get("image_size"), sub_pixel=sub_pixel
        )
        descriptors, offsets = self.desc_head(feature_map, keypoints)

        _, _, h, w = image.shape
        wh = torch.tensor([w - 1, h - 1], device=image.device)

        # no padding required
        # we can set detection_threshold=-1 and conf.max_num_keypoints > 0
        # Return lists instead of stacking to handle variable-length keypoints per image

        return {
            "keypoints": [wh * (kps + 1) / 2.0 for kps in keypoints],  # List of (N_i, 2) tensors
            "descriptors": descriptors,  # Already a list of (N_i, D) tensors
            "keypoint_scores": kptscores,  # Already a list of (N_i,) tensors
        }


class ALIKED(torch.nn.Module):
    def __init__(self, conf: ALIKEDOptions):
        super().__init__()
        assert isinstance(conf, ALIKEDOptions), f"Expected ALIKEDOptions, got {type(conf).__name__}"
        self.conf = conf
        self.sub_pixel = conf.sub_pixel
        kwargs = asdict(conf)
        kwargs.pop("sub_pixel")
        # LightGlue downloads the selected ALIKED checkpoint on first use.
        self.model = _VariableLengthALIKED(**kwargs)
        _verify_aliked_checkpoint()
        for parameter in self.parameters():
            parameter.requires_grad = False

    def forward(self, data):
        assert "image" in data, "Missing key image in data"
        if not data["image"].is_cuda:
            data["image"] = data["image"].cuda()
        data["sub_pixel"] = self.sub_pixel
        features = self.model(data)

        return {
            "keypoints": [f for f in features["keypoints"]],
            "keypoint_scores": [f for f in features["keypoint_scores"]],
            "descriptors": [f.t() for f in features["descriptors"]],
        }
