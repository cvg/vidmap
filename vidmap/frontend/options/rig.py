"""Numeric camera matching policy for synchronized calibrated rigs."""

from typing import Annotated

from pydantic import ConfigDict, Field, model_validator

from vidmap.configuration.validators import dataclass as options_dataclass

CameraId = Annotated[int, Field(strict=True, ge=0, lt=2**32 - 1)]
CameraPair = tuple[CameraId, CameraId]
FiniteFloat = Annotated[float, Field(strict=True, allow_inf_nan=False)]


@options_dataclass(frozen=True, config=ConfigDict(extra="forbid"))
class RigCameraOptions:
    camera_model_name: str
    camera_params: tuple[FiniteFloat, ...]
    cam_from_rig_rotation: tuple[FiniteFloat, FiniteFloat, FiniteFloat, FiniteFloat] | None = None
    cam_from_rig_translation: tuple[FiniteFloat, FiniteFloat, FiniteFloat] | None = None

    @model_validator(mode="after")
    def _validate_pose(self):
        if self.cam_from_rig_rotation is not None:
            if abs(sum(value * value for value in self.cam_from_rig_rotation) - 1) > 1e-6:
                raise ValueError("cam_from_rig_rotation must be a unit wxyz quaternion")
        return self


@options_dataclass(frozen=True, config=ConfigDict(extra="forbid"))
class RigOptions:
    """Input calibration and matching policy for numeric camera IDs."""

    cameras: tuple[RigCameraOptions, ...] = ()
    frames: str | None = None

    keyframe_camera: CameraId | None = None
    match_same_timestamp: tuple[CameraPair, ...] = ()
    flag_same_timestamp_as_sequential: tuple[CameraPair, ...] = ()

    @model_validator(mode="after")
    def _validate(self):
        for camera_id, camera in enumerate(self.cameras):
            if camera_id == 0:
                if camera.cam_from_rig_rotation is not None or camera.cam_from_rig_translation is not None:
                    raise ValueError("Camera 0 is the reference and must not specify cam_from_rig")
            elif camera.cam_from_rig_rotation is None or camera.cam_from_rig_translation is None:
                raise ValueError("Every non-reference camera requires cam_from_rig rotation and translation")
        matching = set()
        sequential = set()
        for pairs, destination in (
            (self.match_same_timestamp, matching),
            (self.flag_same_timestamp_as_sequential, sequential),
        ):
            for first, second in pairs:
                key = frozenset((first, second))
                if len(key) != 2:
                    raise ValueError("Camera pairs must contain two distinct camera IDs")
                if key in destination:
                    raise ValueError("Duplicate or reversed duplicate camera pair")
                destination.add(key)
        if not sequential <= matching:
            raise ValueError("Sequential same-timestamp pairs must be a subset of matching pairs")
        return self
