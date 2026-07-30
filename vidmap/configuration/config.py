from dataclasses import dataclass
from dataclasses import field as dc_field

from pydantic import ConfigDict, model_validator
from pydantic_core import ArgsKwargs

from vidmap.configuration.defaults import (
    EvaluationOptions,
    FrontendRunOptions,
    MappingRunOptions,
    TargetSelectionOptions,
)
from vidmap.configuration.validators import dataclass as pydantic_dataclass
from vidmap.frontend.options import (
    CameraPriorEstimationOptions,
    DepthEstimationOptions,
    ExtendedMatchOptions,
    FrontendTrackOptions,
    KeyframeOptions,
    PreparationOptions,
    RoMaV2Options,
)
from vidmap.mapper.options import MapperOptions


def _keyword_values(values, *, owner: str):
    if isinstance(values, ArgsKwargs):
        if values.args:
            raise TypeError(f"{owner} accepts keyword arguments only")
        return {} if values.kwargs is None else dict(values.kwargs)
    return values


@pydantic_dataclass(frozen=True, config=ConfigDict(extra="forbid", strict=True))
class FrontendConfig:
    roma: RoMaV2Options = dc_field(default_factory=RoMaV2Options)
    keyframes: KeyframeOptions = dc_field(default_factory=KeyframeOptions)
    tracks: FrontendTrackOptions = dc_field(default_factory=FrontendTrackOptions)
    loop_closure: ExtendedMatchOptions = dc_field(default_factory=ExtendedMatchOptions)
    depth: DepthEstimationOptions = dc_field(default_factory=DepthEstimationOptions)
    camera_priors: CameraPriorEstimationOptions = dc_field(default_factory=CameraPriorEstimationOptions)
    preparation: PreparationOptions = dc_field(default_factory=PreparationOptions)
    use_geocalib: bool = True
    view_graph_calibration: bool = True

    @model_validator(mode="before")
    @classmethod
    def _normalize_input(cls, values):
        values = _keyword_values(values, owner="FrontendConfig")
        if not isinstance(values, dict):
            return values
        values = dict(values)
        for name, option_type in {
            "roma": RoMaV2Options,
            "keyframes": KeyframeOptions,
            "tracks": FrontendTrackOptions,
            "loop_closure": ExtendedMatchOptions,
            "depth": DepthEstimationOptions,
            "camera_priors": CameraPriorEstimationOptions,
            "preparation": PreparationOptions,
        }.items():
            value = values.get(name)
            if value is None:
                values[name] = option_type()
            elif isinstance(value, dict):
                values[name] = option_type(**value)
        return values


@pydantic_dataclass(frozen=True, config=ConfigDict(extra="forbid", strict=True))
class MappingConfig:
    mapper: MapperOptions = dc_field(default_factory=MapperOptions)

    @model_validator(mode="before")
    @classmethod
    def _normalize_input(cls, values):
        values = _keyword_values(values, owner="MappingConfig")
        if not isinstance(values, dict):
            return values
        values = dict(values)
        value = values.get("mapper")
        if value is None:
            values["mapper"] = MapperOptions()
        elif isinstance(value, dict):
            values["mapper"] = MapperOptions(**value)
        return values


@dataclass(frozen=True)
class FrontendRunSpec:
    """Resolved frontend recipe plus non-algorithm execution ownership."""

    pipeline: FrontendConfig
    name: str
    colmap_runtime: str
    selection: TargetSelectionOptions
    run: FrontendRunOptions


@dataclass(frozen=True)
class MappingRunSpec:
    """Resolved mapping recipe plus selection, storage, and evaluation ownership."""

    pipeline: MappingConfig
    name: str
    colmap_runtime: str
    selection: TargetSelectionOptions
    run: MappingRunOptions
    evaluation: EvaluationOptions
