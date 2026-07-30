"""Schemas shared by frontend image matching stages."""

from typing import Optional, Tuple

from pydantic import ConfigDict, model_validator

from vidmap.configuration.validators import dataclass


@dataclass(frozen=True, config=ConfigDict(extra="forbid"))
class RoMaV2Options:
    """Options for the sole supported RoMaV2 frontend configuration."""

    compile: bool = True


@dataclass(frozen=True, config=ConfigDict(extra="forbid"))
class PreprocessingOptions:
    """Image preprocessing shared by salient-feature and depth loaders."""

    grayscale: bool = False
    resize_max: Optional[int] = None
    resize_force: bool = False
    interpolation: str = "cv2_area"


@dataclass(frozen=True, config=ConfigDict(extra="forbid"))
class LowresMatchOptions:
    batch_size: int = 8
    num_workers: int = 16
    resize_to_shape: Tuple[int, int] = (560, 560)
    interpolation: str = "torch_bicubic"

    @model_validator(mode="after")
    def _validate_square_resolution(self):
        width, height = self.resize_to_shape
        if width != height:
            raise ValueError("RoMa low-resolution matching requires a square resize_to_shape")
        if width < 1:
            raise ValueError("RoMa low-resolution matching resolution must be positive")
        return self

    @property
    def resolution(self) -> int:
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
    tcorr_min_matches: int = 200
    lc_pair_nms: bool = True
    lc_pair_nms_radius: int = 2
    lc_match_thresh: float = 0.05
