"""Options for view-graph filtering, calibration, and relative orientation."""

from dataclasses import field as dc_field
from typing import Annotated, Optional

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
    compute_reproj_error_outliers: bool = True
    reproj_outlier_threshold: Annotated[float, Field(gt=0, allow_inf_nan=False)] = 5.0
    ransac_max_iterations: Annotated[int, Field(gt=0)] = 50000
    ransac_max_epipolar_error: Annotated[float, Field(gt=0, allow_inf_nan=False)] = 4.0
    depth_stddev_multiplier: Annotated[float, Field(gt=0, allow_inf_nan=False)] = 1.0


@pydantic_dataclass(frozen=True, config=ConfigDict(extra="forbid", strict=True))
class RAOptions:
    filter_risky_loop_closure_pairs: bool = False
    filter_unregistered_images: bool = True
    # Pin native random initialization for byte-identical rotation averaging.
    random_seed: int = 1
    # Disable native post-RA edge filtering to avoid silent de-registration.
    max_rotation_error_deg: float = 0.0
    video_tracking_huber_scale: float = 0.1
    video_lc_cauchy_scale: float = 0.05
    # One thread keeps the solve byte-identical; None means one, -1 means every core.
    num_threads: Optional[int] = None


@pydantic_dataclass(frozen=True, config=ConfigDict(extra="forbid", strict=True))
class InlierThresholdOptions:
    max_epipolar_error_E: float = 8.0
    max_epipolar_error_F: float = 8.0
    max_epipolar_error_H: float = 8.0
    min_angle_from_epipole: float = 0.001
    max_angle_error: float = 1.0
    min_triangulation_angle: float = 0.001
    min_inlier_num: float = 5.0
    min_inlier_ratio: float = 0.0
