"""Explicit focal-prior settings; predictor selection belongs to the frontend."""

from typing import Annotated

from pydantic import ConfigDict, Field, model_validator

from vidmap.configuration.validators import dataclass


@dataclass(frozen=True, config=ConfigDict(extra="forbid", strict=True))
class CalibrationOptions:
    da3_log_focal_stddev: Annotated[float, Field(gt=0, allow_inf_nan=False)] = 0.24
    vgc_enabled: bool = True
    vgc_focal_prior: bool = False
    optimize_intrinsics: bool = True

    @model_validator(mode="after")
    def validate_camera_locking(self):
        if not self.optimize_intrinsics and self.vgc_enabled:
            raise ValueError("Locked intrinsics require vgc_enabled=false; VGC can change camera parameters")
        return self
