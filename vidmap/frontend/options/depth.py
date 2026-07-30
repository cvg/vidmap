"""Schemas owned by frontend depth estimation."""

from dataclasses import field as dc_field
from typing import Literal

from pydantic import ConfigDict

from vidmap.configuration.validators import dataclass
from vidmap.frontend.options.matching import PreprocessingOptions


@dataclass(frozen=True, config=ConfigDict(extra="forbid"))
class Da3VideoOptions:
    """Typed config for the Depth Anything 3 video backend."""

    type: Literal["da3_video"] = "da3_video"
    window_size: int = 1
    ref_view_strategy: Literal["middle", "first", "saddle_balanced"] = "middle"
    process_res: int = 504


@dataclass(frozen=True, config=ConfigDict(extra="forbid"))
class DepthEstimationOptions:
    backend: Da3VideoOptions = dc_field(default_factory=Da3VideoOptions)
    cache_map: bool = True
    preprocessing: PreprocessingOptions = dc_field(
        default_factory=lambda: PreprocessingOptions(resize_max=None, resize_force=False)
    )
    batch_size: int = 1
    num_workers: int = 4

    @property
    def depth_model(self) -> str:
        """Stable cache token consumed by the current pipeline setup boundary."""
        return f"da3_video_w{self.backend.window_size}"
