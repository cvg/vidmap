"""Options for absolute location priors (e.g. from visual localization) in RA, GP, and BA."""

from typing import Annotated, Literal, Optional

from pydantic import ConfigDict, Field

from vidmap.configuration.validators import dataclass as pydantic_dataclass


@pydantic_dataclass(frozen=True, config=ConfigDict(extra="forbid", strict=True))
class LocationPriorOptions:
    """Options for absolute rotation and 2D-3D correspondence constraints."""

    enabled: bool = False
    path: Optional[str] = None
    min_confidence: Annotated[float, Field(ge=0.0, allow_inf_nan=False)] = 0.05
    min_inliers: Annotated[int, Field(ge=1)] = 4

    # Stage 1: Rotation Averaging
    use_in_rotation_averaging: bool = True
    ra_weight: Annotated[float, Field(ge=0.0, allow_inf_nan=False)] = 20.0
    ra_cauchy_scale: Annotated[float, Field(gt=0.0, allow_inf_nan=False)] = 3.0
    ra_ref_sigma_deg: Annotated[float, Field(gt=0.0, allow_inf_nan=False)] = 1.0
    ra_min_sigma_deg: Annotated[float, Field(gt=0.0, allow_inf_nan=False)] = 0.5

    # Stage 2: Global Positioning
    use_in_global_positioning: bool = True
    use_in_gp1: bool = True
    gp_weight: Annotated[float, Field(ge=0.0, allow_inf_nan=False)] = 1.0
    # The prior bearing residual is whitened by the keypoint stddev before the Cauchy loss, so the
    # loss inflection in pixels is (cauchy_scale * kp_stddev_px). GP1 uses a stiff pull to initialize
    # positions from the priors; GP2 is softer so that coherent blocks of prior outliers can be overruled.
    gp1_kp_stddev_px: Annotated[float, Field(gt=0.0, allow_inf_nan=False)] = 0.5
    gp1_cauchy_scale: Annotated[float, Field(gt=0.0, allow_inf_nan=False)] = 48.0
    gp_kp_stddev_px: Annotated[float, Field(gt=0.0, allow_inf_nan=False)] = 1.0
    gp_cauchy_scale: Annotated[float, Field(gt=0.0, allow_inf_nan=False)] = 12.0
    gp_relaxed_scale_prior_stddev: Annotated[float, Field(gt=0.0, allow_inf_nan=False)] = 5.0
    gp_ransac_inlier_threshold_m: Annotated[float, Field(gt=0.0, allow_inf_nan=False)] = 3.0
    # Below this extent of the anchors, the GP1 scale is kept and only a translation is estimated.
    gp_alignment_min_extent_for_scale_m: Annotated[float, Field(ge=0.0, allow_inf_nan=False)] = 2.0
    # GP1 is aligned to the priors separately for each part of the reconstruction whose images share fewer than
    # gp_alignment_min_shared_points 3D points with the rest (e.g. across video cuts), counting only the points whose
    # GP1 angular reprojection error is below gp_alignment_max_angle_error_deg.
    gp_alignment_min_shared_points: Annotated[int, Field(ge=1)] = 20
    gp_alignment_max_angle_error_deg: Annotated[float, Field(gt=0.0, allow_inf_nan=False)] = 2.0

    # Stage 3: Bundle Adjustment
    use_in_bundle_adjustment: bool = True
    ba_weight: Annotated[float, Field(ge=0.0, allow_inf_nan=False)] = 1.0
    ba_kp_stddev_px: Annotated[float, Field(gt=0.0, allow_inf_nan=False)] = 1.0
    ba_loss_name: Literal["trivial", "soft_l1", "cauchy", "huber"] = "cauchy"
    ba_loss_scale_px: Annotated[float, Field(gt=0.0, allow_inf_nan=False)] = 8.0
    ba_relaxed_scale_prior_stddev: Annotated[float, Field(gt=0.0, allow_inf_nan=False)] = 5.0
