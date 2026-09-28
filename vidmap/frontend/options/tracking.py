"""Schemas owned by sparse-track propagation."""

import math
from dataclasses import field as dc_field
from numbers import Integral
from typing import Iterable, Optional

from pydantic import ConfigDict, Field, field_validator, model_validator

from vidmap.configuration.defaults import CANONICAL_POINT_BUDGET
from vidmap.configuration.validators import dataclass
from vidmap.frontend.options.matching import RoMaImageOptions


def normalize_multiflow_hops(multiflow_hops: Iterable[int]) -> tuple[int, ...]:
    """Validate and normalize the explicit sparse hop schedule."""
    if multiflow_hops is None:
        raise ValueError("multiflow_hops must be explicit")
    raw_hops = tuple(multiflow_hops)
    if len(raw_hops) == 0:
        raise ValueError("multiflow_hops must not be empty")
    if any(isinstance(hop, bool) or not isinstance(hop, Integral) for hop in raw_hops):
        raise ValueError("multiflow_hops must contain integer hops, not booleans")
    hops = tuple(int(hop) for hop in raw_hops)
    if any(hop <= 0 for hop in hops):
        raise ValueError("multiflow_hops must be positive")
    if len(set(hops)) != len(hops):
        raise ValueError("multiflow_hops must be unique")
    if 1 not in hops:
        raise ValueError("multiflow_hops must include hop 1")
    return tuple(sorted(hops, reverse=True))


@dataclass(frozen=True, config=ConfigDict(extra="forbid"))
class SparseTrackOptions:
    """Typed config for streaming sparse-track propagation."""

    num_workers: int = 4
    nms_radius: int = 3
    max_kps: int = Field(default=CANONICAL_POINT_BUDGET, gt=0, strict=True)
    min_conf: float = 0.05
    max_sequential_track_sigma_roma_px: Optional[float] = 8.0
    density_thin_k: float = 0.05
    # Bandwidths below are reference values at CANONICAL_POINT_BUDGET.
    density_thin_std: float = Field(default=0.013, gt=0, allow_inf_nan=False)
    sampling_density_std: float = Field(default=0.1, gt=0, allow_inf_nan=False)
    tvg_max_epipolar_error: float = 4.0
    tvg_max_iterations: int = 50000
    tvg_min_iterations: Optional[int] = 50
    tvg_min_inliers: int = 50
    salient_density_ref_kps: int = 2000
    salient_density_std: float = 0.1
    salient_density_max_scale: float = 3.0
    salient_density_power: float = 1.0
    lt_cov_scale: float = 4.0
    multiflow_hops: tuple[int, ...] = (8, 6, 4, 2, 1)

    @property
    def effective_density_thin_std(self) -> float:
        return self.density_thin_std * math.sqrt(CANONICAL_POINT_BUDGET / self.max_kps)

    @property
    def effective_sampling_density_std(self) -> float:
        return self.sampling_density_std * math.sqrt(CANONICAL_POINT_BUDGET / self.max_kps)

    @field_validator("multiflow_hops", mode="before")
    @classmethod
    def _normalize_multiflow_hops(cls, value):
        return normalize_multiflow_hops(value)

    @field_validator("max_sequential_track_sigma_roma_px", mode="before")
    @classmethod
    def _reject_boolean_sequential_sigma_threshold(cls, value):
        if isinstance(value, bool):
            raise ValueError("max_sequential_track_sigma_roma_px must be a number, not a boolean")
        return value

    @model_validator(mode="after")
    def _validate_sequential_sigma_threshold(self):
        sequential_threshold = self.max_sequential_track_sigma_roma_px
        if sequential_threshold is not None and (not math.isfinite(sequential_threshold) or sequential_threshold <= 0):
            raise ValueError("max_sequential_track_sigma_roma_px must be finite and positive")
        return self


@dataclass(frozen=True, config=ConfigDict(extra="forbid"))
class FrontendTrackOptions:
    """High-resolution matching and sparse temporal propagation."""

    images: RoMaImageOptions = dc_field(default_factory=RoMaImageOptions)
    propagation: SparseTrackOptions = dc_field(default_factory=SparseTrackOptions)
