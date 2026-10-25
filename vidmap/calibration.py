"""Original prediction associations shared by extraction and mapping."""

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class CalibrationObservation:
    camera_id: int
    key: str
    image_name: str | None


@dataclass(frozen=True)
class CalibrationEvidence:
    estimator: str
    inference: str
    model: str
    pixel_domain: str
    observations: tuple[CalibrationObservation, ...]

    def __post_init__(self):
        if self.estimator not in {"geocalib", "da3"}:
            raise ValueError("Calibration evidence requires an enabled predictor")
        if self.inference not in {"selected_batch", "per_view"}:
            raise ValueError("Unknown calibration inference")
        if self.model not in {"PINHOLE", "SIMPLE_DIVISIONAL"} or self.pixel_domain != "working_images":
            raise ValueError("Unsupported calibration prediction domain")
        if self.estimator == "da3" and (self.inference != "per_view" or self.model != "PINHOLE"):
            raise ValueError("DA3 calibration requires per-view pinhole predictions")
        if self.model == "SIMPLE_DIVISIONAL" and self.inference != "selected_batch":
            raise ValueError("Divisional calibration requires selected-batch GeoCalib predictions")
        if not self.observations:
            raise ValueError("Calibration results require camera associations")
        if len(set(self.observations)) != len(self.observations):
            raise ValueError("Repeated calibration observation")
        shared = self.inference == "selected_batch"
        if any((observation.image_name is None) != shared for observation in self.observations):
            raise ValueError("Calibration observations disagree with inference mode")
        if not shared and len({observation.image_name for observation in self.observations}) != len(self.observations):
            raise ValueError("Per-view calibration requires unique image associations")

    def as_dict(self) -> dict:
        return {
            "estimator": self.estimator,
            "inference": self.inference,
            "model": self.model,
            "pixel_domain": self.pixel_domain,
            "observations": [asdict(observation) for observation in self.observations],
        }

    @classmethod
    def from_dict(cls, value: dict) -> "CalibrationEvidence":
        return cls(
            estimator=value["estimator"],
            inference=value["inference"],
            model=value["model"],
            pixel_domain=value["pixel_domain"],
            observations=tuple(CalibrationObservation(**item) for item in value["observations"]),
        )
