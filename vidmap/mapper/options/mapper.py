from dataclasses import field as dc_field
from typing import Literal, Optional

from pydantic import ConfigDict, model_validator
from pydantic_core import ArgsKwargs

from vidmap.configuration.validators import dataclass as pydantic_dataclass
from vidmap.configuration.validators import instantiate_nested_options

from .positioning import DepthConsistencyOptions, GPOptions, MapperTrackOptions
from .refinement import BAOptions
from .view_graph import InlierThresholdOptions, MDRPOptions, RAOptions, VGCOptions


@pydantic_dataclass(frozen=True, config=ConfigDict(extra="forbid", strict=True))
class SetupOptions:
    """Finalized mapping-problem loading options."""

    depth_uncertainty_scale: float = 0.05


@pydantic_dataclass(frozen=True, config=ConfigDict(extra="forbid", strict=True))
class ReplayCacheOptions:
    """Canonical database-boundary input and stage-capture controls."""

    mode: Literal["off", "byte_check"] = "off"
    root: Optional[str] = None
    write_stage: Optional[str] = None


def coerce_to_dict(raw):
    """Copy keyword input without accepting positional construction."""
    if isinstance(raw, dict):
        return dict(raw)
    if isinstance(raw, ArgsKwargs):
        if raw.args:
            raise TypeError("MapperOptions accepts keyword arguments only")
        return dict(raw.kwargs or {})
    return raw


@pydantic_dataclass(frozen=True, config=ConfigDict(extra="forbid", strict=True))
class MapperOptions:
    """Typed options for the mapping stages."""

    setup: SetupOptions = dc_field(default_factory=SetupOptions)
    depth_consistency: DepthConsistencyOptions = dc_field(default_factory=DepthConsistencyOptions)
    ba: BAOptions = dc_field(default_factory=BAOptions)
    vgc: VGCOptions = dc_field(default_factory=VGCOptions)
    inlier_thresholds: InlierThresholdOptions = dc_field(default_factory=InlierThresholdOptions)
    mdrp: MDRPOptions = dc_field(default_factory=MDRPOptions)
    gp: GPOptions = dc_field(default_factory=GPOptions)
    ra: RAOptions = dc_field(default_factory=RAOptions)
    tracks: MapperTrackOptions = dc_field(default_factory=MapperTrackOptions)

    replay_cache: ReplayCacheOptions = dc_field(default_factory=ReplayCacheOptions)

    @model_validator(mode="before")
    @classmethod
    def instantiate_nested_option_groups(cls, raw):
        """Instantiate nested option groups before strict field validation."""
        option_groups = {
            "setup": SetupOptions,
            "depth_consistency": DepthConsistencyOptions,
            "ba": BAOptions,
            "vgc": VGCOptions,
            "inlier_thresholds": InlierThresholdOptions,
            "mdrp": MDRPOptions,
            "gp": GPOptions,
            "ra": RAOptions,
            "tracks": MapperTrackOptions,
            "replay_cache": ReplayCacheOptions,
        }
        raw = instantiate_nested_options(coerce_to_dict(raw), option_groups)
        if not isinstance(raw, dict):
            return raw

        for option_name, option_type in option_groups.items():
            if raw.get(option_name) is None:
                raw[option_name] = option_type()
        return raw

    @model_validator(mode="after")
    def reject_replay_controls_when_disabled(self):
        """Keep replay/determinism controls out of raw production configs."""
        if self.replay_cache.mode != "off":
            return self

        replay_only = []
        for field in (
            "roundtrip_before_ba",
            "canonical_checkpoint",
        ):
            if getattr(self.gp.common, field):
                replay_only.append(f"gp.common.{field}")
        for field in ("skip_zero_observation_points", "update_point3d_errors"):
            if getattr(self.gp.track_filter, field):
                replay_only.append(f"gp.track_filter.{field}")

        if self.gp.common.num_threads is not None:
            replay_only.append("gp.common.num_threads")
        if self.ba.num_threads is not None:
            replay_only.append("ba.num_threads")
        if self.gp.second_pass.center_init_mode != GPOptions().second_pass.center_init_mode:
            replay_only.append("gp.second_pass.center_init_mode")
        if self.gp.common.parameter_ordering_strategy != GPOptions().common.parameter_ordering_strategy:
            replay_only.append("gp.common.parameter_ordering_strategy")
        if self.gp.common.camera_center_strategy != GPOptions().common.camera_center_strategy:
            replay_only.append("gp.common.camera_center_strategy")
        if replay_only:
            controls = ", ".join(sorted(replay_only))
            raise ValueError(
                "Replay/determinism controls require replay_cache.mode != 'off'; " f"got raw config with {controls}"
            )
        return self
