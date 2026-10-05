"""Depth, track, and global-positioning options."""

from dataclasses import field as dc_field
from typing import Annotated, Literal, Optional

from pydantic import ConfigDict, Field, model_validator

from vidmap.configuration.validators import dataclass as pydantic_dataclass
from vidmap.configuration.validators import instantiate_nested_options
from vidmap.mapper.options.solver import SolverBackendOptions

ScaleLossName = Literal["trivial", "huber", "cauchy", "soft_l1"]
NativeLossName = Literal["trivial", "soft_l1", "cauchy", "huber"]


@pydantic_dataclass(frozen=True, kw_only=True, config=ConfigDict(extra="forbid", strict=True))
class LossConfig:
    """Typed keyword-only loss specification."""

    name: Literal["trivial", "huber", "cauchy", "soft_l1"]
    scale: Annotated[float, Field(gt=0, allow_inf_nan=False)] = 1.0
    weight: Annotated[float, Field(ge=0, allow_inf_nan=False)] = 1.0


def instantiate_losses(raw, names):
    return instantiate_nested_options(raw, {name: LossConfig for name in names})


@pydantic_dataclass(frozen=True, kw_only=True, config=ConfigDict(extra="forbid", strict=True))
class GPCommonOptions:
    use_lc_observations: bool = True
    use_metric_depth_constraint: bool = True
    bearing_kp_stddev: Annotated[float, Field(gt=0, allow_inf_nan=False)] = 8.0
    optimize_depth_map_scales: bool = True
    use_log_scale_for_depth_map_scales: bool = False
    smooth_log_linear_transition: bool = False
    log_linear_threshold: Annotated[float, Field(gt=0, allow_inf_nan=False)] = 0.1
    random_seed: Annotated[int, Field(ge=-1)] = 1
    loss_function_type: NativeLossName = "huber"
    loss_function_weight: Annotated[float, Field(ge=0, allow_inf_nan=False)] = 1.0
    loss_function_scale: Annotated[float, Field(gt=0, allow_inf_nan=False)] = 0.1
    apply_uncalibrated_loss_downweight: bool = False
    num_threads: Optional[int] = None
    zero_residuals_behind_camera: bool = False

    @model_validator(mode="after")
    def validate_threads(self):
        if self.num_threads == 0:
            raise ValueError("num_threads must be nonzero")
        return self


@pydantic_dataclass(frozen=True, kw_only=True, config=ConfigDict(extra="forbid", strict=True))
class GPFirstPassOptions:
    scale_prior_stddev: Annotated[float, Field(gt=0, allow_inf_nan=False)] = 0.05
    max_iterations: Annotated[int, Field(gt=0)] = 100
    function_tolerance_when_second: Optional[Annotated[float, Field(ge=0, allow_inf_nan=False)]] = 1.0e-4
    gradient_tolerance_when_second: Optional[Annotated[float, Field(ge=0, allow_inf_nan=False)]] = 1.0e-8
    parameter_tolerance_when_second: Optional[Annotated[float, Field(ge=0, allow_inf_nan=False)]] = 1.0e-6
    scale_reg_loss_name: ScaleLossName = "trivial"
    scale_reg_weight: Annotated[float, Field(ge=0, allow_inf_nan=False)] = 1.0
    initialize_warm_start_scales: bool = True
    sequential_support_warmup_rounds: Annotated[int, Field(ge=0)] = 16
    sequential_support_observations_per_track: Annotated[int, Field(ge=0)] = 16
    sequential_support_loss: Optional[LossConfig] = LossConfig(name="trivial", weight=1.0)
    loss_lc_geometry: LossConfig = LossConfig(name="cauchy", scale=2.0, weight=0.4)
    loss_lc_depth: LossConfig = LossConfig(name="cauchy", scale=2.0, weight=0.4)
    loss_normal_geometry: LossConfig = LossConfig(name="huber")
    loss_normal_depth: LossConfig = LossConfig(name="trivial")

    @model_validator(mode="before")
    @classmethod
    def instantiate_loss_configs(cls, raw):
        return instantiate_losses(
            raw,
            (
                "loss_lc_geometry",
                "loss_lc_depth",
                "loss_normal_geometry",
                "loss_normal_depth",
                "sequential_support_loss",
            ),
        )

    @model_validator(mode="after")
    def validate_sequential_support(self):
        enabled = self.sequential_support_warmup_rounds > 0
        if enabled != (self.sequential_support_observations_per_track > 0) or enabled != (
            self.sequential_support_loss is not None
        ):
            raise ValueError("sequential support requires warm-up rounds, observations per track, and a loss")
        return self


@pydantic_dataclass(frozen=True, kw_only=True, config=ConfigDict(extra="forbid", strict=True))
class GPSecondPassOptions:
    enabled: bool = True
    scale_prior_stddev: Annotated[float, Field(gt=0, allow_inf_nan=False)] = 0.05
    max_iterations: Annotated[int, Field(gt=0)] = 100
    use_log_depth_residual: bool = True
    relax_angular_stddevs: Annotated[float, Field(gt=0, allow_inf_nan=False)] = 1.0
    scale_reg_loss_name: ScaleLossName = "huber"
    scale_reg_weight: Annotated[float, Field(ge=0, allow_inf_nan=False)] = 1.0
    loss_lc_geometry: LossConfig = LossConfig(name="cauchy", scale=4.0, weight=1.0)
    loss_lc_depth: LossConfig = LossConfig(name="cauchy", scale=4.0, weight=1.0)
    loss_normal_geometry: LossConfig = LossConfig(name="huber")
    loss_normal_depth: LossConfig = LossConfig(name="huber")

    @model_validator(mode="before")
    @classmethod
    def instantiate_loss_configs(cls, raw):
        return instantiate_losses(
            raw,
            (
                "loss_lc_geometry",
                "loss_lc_depth",
                "loss_normal_geometry",
                "loss_normal_depth",
            ),
        )


@pydantic_dataclass(frozen=True, kw_only=True, config=ConfigDict(extra="forbid", strict=True))
class GPTrackFilterOptions:
    min_num_views_per_track: Annotated[int, Field(gt=0)] = 2
    loss_normal_depth_outlier: LossConfig = LossConfig(name="cauchy", scale=3.0, weight=1.0)
    depth_prior_outlier_max_depth: Optional[Annotated[float, Field(gt=0, allow_inf_nan=False)]] = None
    depth_prior_outlier_stages: Literal["gp1", "gp1_gp2"] = "gp1"

    @model_validator(mode="before")
    @classmethod
    def instantiate_loss_configs(cls, raw):
        return instantiate_losses(raw, ("loss_normal_depth_outlier",))


@pydantic_dataclass(frozen=True, kw_only=True, config=ConfigDict(extra="forbid", strict=True))
class GPTemporalAccelerationOptions:
    """Optional smoothness prior over adjacent video triplets.

    ``timestamp`` uses physical elapsed seconds, so acceleration quantities are
    in world-units/s^2 (m/s^2 with a metric-scale anchor). ``keyframe_index``
    uses unit spacing between selected keyframes and penalizes geometric
    displacement curvature in world-units/keyframe^2.
    """

    coordinate: Literal["timestamp", "keyframe_index"] = "timestamp"
    stddev: Annotated[float, Field(gt=0, allow_inf_nan=False)] = 1.0
    first_pass_weight: Annotated[float, Field(ge=0, allow_inf_nan=False)] = 0.0
    second_pass_weight: Annotated[float, Field(ge=0, allow_inf_nan=False)] = 0.0
    first_pass_dead_zone: Annotated[float, Field(ge=0, allow_inf_nan=False)] = 0.0
    first_pass_huber_width: Annotated[float, Field(gt=0, allow_inf_nan=False)] = 1.0
    second_pass_dead_zone: Annotated[float, Field(ge=0, allow_inf_nan=False)] = 0.0
    second_pass_huber_width: Annotated[float, Field(gt=0, allow_inf_nan=False)] = 1.0


@pydantic_dataclass(frozen=True, kw_only=True, config=ConfigDict(extra="forbid", strict=True))
class GPOptions:
    common: GPCommonOptions = dc_field(default_factory=GPCommonOptions)
    solver_backend: SolverBackendOptions = dc_field(default_factory=SolverBackendOptions)
    first_pass: GPFirstPassOptions = dc_field(default_factory=GPFirstPassOptions)
    second_pass: GPSecondPassOptions = dc_field(default_factory=GPSecondPassOptions)
    track_filter: GPTrackFilterOptions = dc_field(default_factory=GPTrackFilterOptions)
    temporal_acceleration: GPTemporalAccelerationOptions = dc_field(default_factory=GPTemporalAccelerationOptions)

    @model_validator(mode="before")
    @classmethod
    def instantiate_nested_options(cls, raw):
        return instantiate_nested_options(
            raw,
            {
                "common": GPCommonOptions,
                "solver_backend": SolverBackendOptions,
                "first_pass": GPFirstPassOptions,
                "second_pass": GPSecondPassOptions,
                "track_filter": GPTrackFilterOptions,
                "temporal_acceleration": GPTemporalAccelerationOptions,
            },
        )


@pydantic_dataclass(frozen=True, config=ConfigDict(extra="forbid", strict=True))
class DepthConsistencyOptions:
    enabled: bool = True
    depth_outlier_propagation: Literal["boundary", "forward"] = "boundary"
    mark_boundary_depth_outliers: bool = True
    depth_ratio_threshold: float = 1.3


@pydantic_dataclass(frozen=True, config=ConfigDict(extra="forbid", strict=True))
class MapperTrackOptions:
    include_loop_closure_observations: bool = True
    min_num_views_per_track: Annotated[int, Field(gt=0)] = 2
    max_num_views_per_track: int = 100
    two_view_depth_gate: bool = True
    loop_closure_second_pass: bool = True
