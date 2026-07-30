"""Schemas owned by mapper-input preparation."""

from dataclasses import field as dc_field

from pydantic import ConfigDict, model_validator
from pydantic_core import ArgsKwargs

from vidmap.configuration.validators import dataclass


@dataclass(frozen=True, config=ConfigDict(extra="forbid"))
class CameraPriorEstimationOptions:
    max_images: int = 30


@dataclass(frozen=True, config=ConfigDict(extra="forbid", strict=True))
class GeomVerifOptions:
    max_H_inlier_ratio: float = 0.8
    min_num_inliers: int = 15
    ransac_max_num_trials: int = 20_000
    ransac_min_inlier_ratio: float = 0.25
    ransac_max_error: float = 4.0


@dataclass(frozen=True, config=ConfigDict(extra="forbid", strict=True))
class LCOptions:
    exclude_lc_matches: bool = False
    exclude_window_lc: bool = False
    min_lc_score: float = 0.0


@dataclass(frozen=True, kw_only=True, config=ConfigDict(extra="forbid", strict=True))
class PreparationOptions:
    """Frontend-owned policy for producing finalized mapper inputs."""

    lc: LCOptions = dc_field(default_factory=LCOptions)
    geom_verif: GeomVerifOptions = dc_field(default_factory=GeomVerifOptions)

    @model_validator(mode="before")
    @classmethod
    def _instantiate_nested_option_groups(cls, raw):
        if isinstance(raw, ArgsKwargs):
            if raw.args:
                raise TypeError("PreparationOptions accepts keyword arguments only")
            raw = dict(raw.kwargs or {})
        if not isinstance(raw, dict):
            return raw
        raw = dict(raw)
        for name, option_type in {
            "lc": LCOptions,
            "geom_verif": GeomVerifOptions,
        }.items():
            value = raw.get(name)
            if value is None:
                raw[name] = option_type()
            elif isinstance(value, dict):
                raw[name] = option_type(**value)
        return raw
