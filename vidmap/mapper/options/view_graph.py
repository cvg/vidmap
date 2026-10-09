"""Options for view-graph filtering, calibration, and relative orientation."""

from dataclasses import field as dc_field
from typing import Annotated, Literal, Optional

from pydantic import ConfigDict, Field, model_validator

from vidmap.configuration.validators import dataclass as pydantic_dataclass
from vidmap.configuration.validators import instantiate_nested_options


@pydantic_dataclass(frozen=True, config=ConfigDict(extra="forbid", strict=True))
class VGCFilterOptions:
    enabled: bool = True
    num_threads: Optional[int] = None
    min_matches: Annotated[int, Field(gt=0)] = 10
    min_cheirality_points: Annotated[int, Field(gt=0)] = 5
    subsample_size: Annotated[int, Field(gt=0)] = 50
    strong_pair_match_count: Annotated[int, Field(gt=0)] = 50
    min_median_triangulation_angle_deg: float = 16.0
    max_abs_forward_translation_ratio: float = 0.95
    min_strong_kept_pairs: int = 50


@pydantic_dataclass(frozen=True, config=ConfigDict(extra="forbid", strict=True))
class VGCCalibrationOptions:
    unlock_focal: bool = True
    # Outer prior coefficient before multiplication by eligible pairs / images.
    focal_prior_weight: Annotated[float, Field(gt=0, allow_inf_nan=False)] = 1.0e-5
    normalize_weight_by_pair_count: bool = True
    relative_focal_weight: Annotated[float, Field(ge=0, allow_inf_nan=False)] = 1.0e-1
    relative_focal_loss_scale: Annotated[float, Field(gt=0, allow_inf_nan=False)] = 0.05


@pydantic_dataclass(frozen=True, config=ConfigDict(extra="forbid", strict=True))
class VGCOptions:
    filter: VGCFilterOptions = dc_field(default_factory=VGCFilterOptions)
    calibration: VGCCalibrationOptions = dc_field(default_factory=VGCCalibrationOptions)

    @model_validator(mode="before")
    @classmethod
    def instantiate_leaf_options(cls, raw):
        return instantiate_nested_options(
            raw,
            {
                "filter": VGCFilterOptions,
                "calibration": VGCCalibrationOptions,
            },
        )


@pydantic_dataclass(frozen=True, config=ConfigDict(extra="forbid", strict=True))
class MDRPOptions:
    max_workers: Annotated[Optional[int], Field(gt=0)] = None
    ransac_max_iterations: Annotated[int, Field(gt=0)] = 50000
    ransac_max_epipolar_error: Annotated[float, Field(gt=0, allow_inf_nan=False)] = 4.0
    depth_stddev_multiplier: Annotated[float, Field(gt=0, allow_inf_nan=False)] = 1.0


@pydantic_dataclass(frozen=True, config=ConfigDict(extra="forbid", strict=True))
class RAOptions:
    max_iterations: Annotated[int, Field(gt=0)] = 100
    filter_risky_loop_closure_pairs: bool = False
    filter_unregistered_images: bool = True
    # Disable post-RA edge filtering to retain the current pose mask.
    max_rotation_error_deg: Annotated[float, Field(ge=0, allow_inf_nan=False)] = 0.0
    video_tracking_huber_scale: Annotated[float, Field(gt=0, allow_inf_nan=False)] = 0.1
    video_lc_cauchy_scale: Annotated[float, Field(gt=0, allow_inf_nan=False)] = 0.05
    # None means one thread; -1 means every core.
    num_threads: Annotated[int, Field(gt=0)] | Literal[-1] | None = None


@pydantic_dataclass(frozen=True, config=ConfigDict(extra="forbid", strict=True))
class InlierThresholdOptions:
    max_epipolar_error_E: float = 8.0
    max_epipolar_error_F: float = 8.0
    max_epipolar_error_H: float = 8.0
    min_angle_from_epipole: float = 0.001
    max_angle_error: Annotated[float, Field(ge=0, allow_inf_nan=False)] = 1.0
    min_triangulation_angle: Annotated[float, Field(ge=0, allow_inf_nan=False)] = 0.001
    min_inlier_num: float = 5.0
    min_inlier_ratio: float = 0.0
