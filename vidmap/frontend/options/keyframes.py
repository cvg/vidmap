"""Schemas owned by keyframe selection and feature extraction."""

from dataclasses import field as dc_field
from typing import Literal

from pydantic import ConfigDict

from vidmap.configuration.validators import dataclass
from vidmap.frontend.options.matching import LowresMatchOptions, PreprocessingOptions


@dataclass(frozen=True, config=ConfigDict(extra="forbid"))
class DetectKeyframesOptions:
    max_normalized_keypoint_drift: float = 0.11
    target_frac: float = 0.4
    certainty_threshold: float = 0.01
    force_gt_keyframes: bool = False
    intrinsics_source: Literal["geocalib", "ground_truth"] = "geocalib"


@dataclass(frozen=True, config=ConfigDict(extra="forbid"))
class SalientFeatureOptions:
    preprocessing: PreprocessingOptions = dc_field(
        default_factory=lambda: PreprocessingOptions(resize_max=1200, resize_force=False)
    )
    nms_radius: int = 5
    max_num_keypoints: int = 2000
    sub_pixel: bool = False


@dataclass(frozen=True, config=ConfigDict(extra="forbid"))
class KeyframeOptions:
    """Keyframe selection and per-keyframe feature frontend."""

    matching: LowresMatchOptions = dc_field(default_factory=LowresMatchOptions)
    selection: DetectKeyframesOptions = dc_field(default_factory=DetectKeyframesOptions)
    features: SalientFeatureOptions = dc_field(default_factory=SalientFeatureOptions)
