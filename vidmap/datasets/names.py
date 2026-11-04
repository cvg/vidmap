"""Lightweight ordered catalog for benchmark datasets."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Literal

from vidmap.datasets.resources import dataset_asset

EvaluationMetric = Literal["pose_auc", "wate_auc"]


@dataclass(frozen=True)
class DatasetDefinition:
    """Static metadata needed before a concrete dataset layout is requested."""

    name: str
    default_mode: str
    default_evaluation_metric: EvaluationMetric
    preparer_module: str | None
    scenes: tuple[str, ...] = ()
    manifest_reference: str | None = None

    def __post_init__(self) -> None:
        if bool(self.scenes) == bool(self.manifest_reference):
            raise ValueError(f"Dataset {self.name!r} must define scenes or one manifest reference")
        if self.default_evaluation_metric not in {"pose_auc", "wate_auc"}:
            raise ValueError(f"Unsupported default evaluation metric {self.default_evaluation_metric!r}")
        if self.manifest_reference is not None and self.manifest_reference.count(":") != 1:
            raise ValueError(f"Invalid manifest reference for dataset {self.name!r}: {self.manifest_reference!r}")


_CAPTURE_SUITES = json.loads(dataset_asset("benchmark_suites.json").read_text())

DATASET_DEFINITIONS = (
    DatasetDefinition(
        name="crocodl",
        scenes=("ios-ARCHE_D2", "ios-ARCHE_GRANDE", "ios-ARCHE_B3", "ios-ARCHE_B5"),
        default_mode="complete",
        default_evaluation_metric="wate_auc",
        preparer_module="vidmap.datasets.prepare.crocodl",
    ),
    DatasetDefinition(
        name="eth3d_slam",
        manifest_reference="vidmap.datasets.eth3d_slam:ETH3D_SLAM_MANIFEST",
        default_mode="mono",
        default_evaluation_metric="pose_auc",
        preparer_module="vidmap.datasets.prepare.eth3d_slam",
    ),
    DatasetDefinition(
        name="euroc",
        scenes=("machine_hall", "vicon_room1", "vicon_room2"),
        default_mode="mono",
        default_evaluation_metric="pose_auc",
        preparer_module="vidmap.datasets.prepare.euroc",
    ),
    DatasetDefinition(
        name="hilti",
        scenes=tuple(_CAPTURE_SUITES["hilti"]["scenes"]),
        default_mode="panorama",
        default_evaluation_metric="wate_auc",
        preparer_module="vidmap.datasets.prepare.hilti",
    ),
    DatasetDefinition(
        name="lamar",
        scenes=("CAB", "HGE", "LIN"),
        default_mode="complete",
        default_evaluation_metric="wate_auc",
        preparer_module="vidmap.datasets.prepare.lamar",
    ),
    DatasetDefinition(
        name="spires",
        scenes=tuple(_CAPTURE_SUITES["spires"]["scenes"]),
        default_mode="undistorted-mono",
        default_evaluation_metric="wate_auc",
        preparer_module="vidmap.datasets.prepare.spires",
    ),
)

SUPPORTED_DATASETS = tuple(definition.name for definition in DATASET_DEFINITIONS)
_DEFINITIONS_BY_NAME = {definition.name: definition for definition in DATASET_DEFINITIONS}

if len(_DEFINITIONS_BY_NAME) != len(DATASET_DEFINITIONS):
    raise RuntimeError("Dataset catalog names must be unique")
if SUPPORTED_DATASETS != tuple(sorted(SUPPORTED_DATASETS)):
    raise RuntimeError("Dataset catalog must remain alphabetically ordered")


def get_dataset_definition(name: str) -> DatasetDefinition:
    """Return one exact catalog entry."""
    if name not in _DEFINITIONS_BY_NAME:
        supported = ", ".join(SUPPORTED_DATASETS)
        raise ValueError(f"Unsupported dataset {name!r}; expected one of: {supported}")
    return _DEFINITIONS_BY_NAME[name]


def supported_preparers() -> tuple[str, ...]:
    """Return catalog datasets with a preparation command."""
    return tuple(definition.name for definition in DATASET_DEFINITIONS if definition.preparer_module is not None)
