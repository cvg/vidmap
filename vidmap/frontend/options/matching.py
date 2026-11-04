"""Schemas shared by frontend image matching stages."""

from dataclasses import field as dc_field
from typing import Annotated, Literal, Optional, Tuple

from pydantic import ConfigDict, Field, model_validator

from vidmap.configuration.defaults import CANONICAL_POINT_BUDGET
from vidmap.configuration.validators import dataclass


@dataclass(frozen=True, config=ConfigDict(extra="forbid"))
class RoMaV2Options:
    """RoMaV2 matching options."""

    name: Literal["romav2"] = "romav2"
    compile: bool = True
    lc_feature_capacity: int = Field(default=8, ge=2)


@dataclass(frozen=True, config=ConfigDict(extra="forbid"))
class PreprocessingOptions:
    """Image preprocessing shared by salient-feature and depth loaders."""

    grayscale: bool = False
    resize_max: Optional[int] = None
    resize_force: bool = False
    interpolation: str = "cv2_area"


@dataclass(frozen=True, config=ConfigDict(extra="forbid"))
class LowresMatchOptions:
    coarse_only: bool = False
    batch_size: int = 8
    num_workers: int = 4
    # None uses the selected matcher's default low-resolution input size.
    resize_to_shape: Optional[Tuple[int, int]] = None
    interpolation: str = "torch_bicubic"

    @model_validator(mode="after")
    def _validate_square_resolution(self):
        if self.resize_to_shape is None:
            return self
        width, height = self.resize_to_shape
        if width != height:
            raise ValueError("Low-resolution matching requires a square resize_to_shape")
        if width < 1:
            raise ValueError("Low-resolution matching resolution must be positive")
        return self

    @property
    def resolution(self) -> int:
        assert self.resize_to_shape is not None, "Resolve the matcher input size before reading its resolution"
        return int(self.resize_to_shape[0])


@dataclass(frozen=True, config=ConfigDict(extra="forbid"))
class RoMaImageOptions:
    """Image resolutions shared by tracking and loop-closure RoMa inference."""

    grayscale: bool = False
    resize_max: Optional[int] = 1200
    resize_force: bool = True
    interpolation: str = "cv2_area"


@dataclass(frozen=True, config=ConfigDict(extra="forbid"))
class ExtendedMatchOptions:
    retrieval_min_score: float = 0.1
    nquery: int = 10
    # Match count at CANONICAL_POINT_BUDGET; resolved for the active budget.
    tcorr_min_matches: int = Field(default=200, ge=0, strict=True)
    lc_pair_nms: bool = True
    lc_pair_nms_radius: int = 2
    lc_match_thresh: float = 0.05
    max_pairs_per_frame_pair: Annotated[int, Field(ge=1, strict=True)] | None = dc_field(
        default=None, metadata={"omit_default": True}
    )

    def effective_tcorr_min_matches(self, point_budget: int) -> int:
        return self.tcorr_min_matches * point_budget // CANONICAL_POINT_BUDGET


@dataclass(frozen=True, config=ConfigDict(extra="forbid"))
class FlexGlueOptions:
    """Sparse keyframing and RoMa-refined tracking/retrieval."""

    name: Literal["flexglue"] = "flexglue"
    keyframing_layers: Annotated[int, Field(ge=1, le=9)] = 6
    tracking_layers: Annotated[int, Field(ge=1, le=9)] = 6
    retrieval_layers: Annotated[int, Field(ge=1, le=9)] = 9
    compile: bool = True
    lc_feature_capacity: Annotated[int, Field(ge=2)] = 8
    pair_policy: Literal["off", "reject"] = "reject"
    pair_threshold: Annotated[float, Field(gt=0, lt=1)] = 0.8
    early_stop_decay: Annotated[float, Field(ge=0, lt=1)] | None = 0.2
