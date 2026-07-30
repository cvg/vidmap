"""BAOptions — nested bundle-adjustment config."""

from dataclasses import field as dc_field
from typing import Annotated, Literal, Optional

from pydantic import ConfigDict, Field, model_validator

from vidmap.configuration.validators import dataclass as pydantic_dataclass
from vidmap.configuration.validators import instantiate_nested_options

ReprojectionLossName = Literal["trivial", "soft_l1", "cauchy", "huber"]
DepthLossName = Literal["trivial", "cauchy", "soft_l1"]
ScaleLossName = Literal["trivial", "huber", "cauchy", "soft_l1"]


@pydantic_dataclass(frozen=True, config=ConfigDict(extra="forbid", strict=True))
class BATriangulationOptions:
    min_angle: float = 0.001
    ignore_two_view_tracks: bool = False
    create_max_angle_error: float = 3.0
    re_max_angle_error: float = 3.0
    re_max_trials: int = 1_000_000
    re_min_ratio: float = 1.0
    relaxed_max_transitivity: Annotated[int, Field(gt=0)] = 1000


@pydantic_dataclass(frozen=True, config=ConfigDict(extra="forbid", strict=True))
class BADepthOptions:
    param_multiplier: float = 1.0
    magnitude_multiplier: float = 1.0
    cutoff_cauchy_scales: int = 0
    reg_loss_name: DepthLossName = "cauchy"
    scale_std: float = 1.0
    target_stddev_multiplier: float = 1.0
    scale_reg_loss_name: ScaleLossName = "soft_l1"
    risky_track_length_threshold: Annotated[int, Field(gt=0)] = 3
    risky_triangulation_angle_deg: Annotated[float, Field(ge=0, allow_inf_nan=False)] = 1.0


@pydantic_dataclass(frozen=True, config=ConfigDict(extra="forbid", strict=True))
class BANormalOptions:
    iterations: Annotated[int, Field(gt=0)] = 3
    kp_stddev: float = 2.0
    reproj_loss_name: ReprojectionLossName = "soft_l1"
    robust_risky_depth: bool = False
    observation_filter_multiplier: float = 4.0
    convergence_filtered_point_ratio: float = 0.001


@pydantic_dataclass(frozen=True, config=ConfigDict(extra="forbid", strict=True))
class BAAnnealingOptions:
    iterations: Annotated[int, Field(gt=0)] = 3
    kp_std: float = 0.5
    depth_magnitude_multiplier: float = 0.25
    reproj_loss_name: ReprojectionLossName = "soft_l1"
    fix_rotations: bool = False
    robust_risky_depth: bool = True
    depth_cutoff_cauchy_scales: int = 2
    depth_param_multiplier: float = 0.25
    prior_std_factor: Annotated[Optional[float], Field(gt=0, allow_inf_nan=False)] = 0.1
    mad_sigma_floor: float = 0.1
    observation_filter_multiplier: float = 4.0
    convergence_filtered_point_ratio: float = 0.001
    final_depth_param_multiplier: float = 0.125


@pydantic_dataclass(frozen=True, config=ConfigDict(extra="forbid", strict=True))
class BAIntrinsicsOptions:
    use_prior: bool = True
    prior_std_factor: Annotated[float, Field(gt=0, allow_inf_nan=False)] = 0.01
    refine_principal_point: bool = False


@pydantic_dataclass(frozen=True, config=ConfigDict(extra="forbid", strict=True))
class BAOptions:
    normal: BANormalOptions = dc_field(default_factory=BANormalOptions)
    annealing: BAAnnealingOptions = dc_field(default_factory=BAAnnealingOptions)
    triangulation: BATriangulationOptions = dc_field(default_factory=BATriangulationOptions)
    depth: BADepthOptions = dc_field(default_factory=BADepthOptions)
    intrinsics: BAIntrinsicsOptions = dc_field(default_factory=BAIntrinsicsOptions)

    retriangulation_reproj_multiplier: float = 8.0
    first_iteration_error_multiplier: float = 2.0
    multiply_errors: float = 2.0
    reproj_loss_scale: float = 1.5
    num_threads: Optional[int] = None
    variable_point_track_length_threshold: Annotated[int, Field(gt=0)] = 15
    post_annealing_point_refinement: bool = True

    @property
    def resolved_annealing_prior_std_factor(self) -> float:
        prior_std_factor = self.annealing.prior_std_factor
        if prior_std_factor is None:
            return self.intrinsics.prior_std_factor
        return prior_std_factor

    @model_validator(mode="before")
    @classmethod
    def instantiate_nested_options(cls, raw):
        return instantiate_nested_options(
            raw,
            {
                "normal": BANormalOptions,
                "annealing": BAAnnealingOptions,
                "depth": BADepthOptions,
                "intrinsics": BAIntrinsicsOptions,
                "triangulation": BATriangulationOptions,
            },
        )
