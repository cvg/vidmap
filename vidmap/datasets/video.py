"""Explicit public-sequence inventories for video benchmark datasets."""

from dataclasses import dataclass
from typing import Literal

GroundTruthQuality = Literal["sensor", "reference", "none"]


@dataclass(frozen=True)
class VideoSequence:
    """One independently runnable public video sequence."""

    name: str
    split: str
    gt_quality: GroundTruthQuality
    condition_tags: tuple[str, ...] = ()
    total_size_gb: float | None = None
    asset_count: int | None = None

    @property
    def gt_evaluable(self) -> bool:
        """Whether public poses make this sequence locally evaluable."""
        return self.gt_quality != "none"


@dataclass(frozen=True)
class VideoDatasetManifest:
    """Validated, ordered inventory of all public sequences in a dataset."""

    name: str
    sequences: tuple[VideoSequence, ...]

    def __post_init__(self) -> None:
        names = tuple(sequence.name for sequence in self.sequences)
        if len(names) != len(set(names)):
            raise ValueError(f"Duplicate sequence in {self.name} manifest")

    @property
    def scenes(self) -> tuple[str, ...]:
        return tuple(sequence.name for sequence in self.sequences)

    def get(self, scene: str) -> VideoSequence:
        for sequence in self.sequences:
            if sequence.name == scene:
                return sequence
        raise ValueError(f"Unknown {self.name} sequence {scene!r}")

    @property
    def gt_evaluable(self) -> tuple[VideoSequence, ...]:
        return tuple(sequence for sequence in self.sequences if sequence.gt_evaluable)

    @property
    def reconstruction_only(self) -> tuple[VideoSequence, ...]:
        return tuple(sequence for sequence in self.sequences if not sequence.gt_evaluable)
