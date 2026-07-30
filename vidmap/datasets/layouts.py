"""Lazy concrete filesystem layouts for benchmark datasets."""

from __future__ import annotations

import importlib
from functools import cache
from typing import TYPE_CHECKING

from vidmap.datasets.names import DatasetDefinition, get_dataset_definition

if TYPE_CHECKING:
    from vidmap.datasets.base import DatasetLayout
    from vidmap.datasets.video import VideoDatasetManifest


def _load_manifest(definition: DatasetDefinition) -> VideoDatasetManifest | None:
    if definition.manifest_reference is None:
        return None
    module_name, attribute = definition.manifest_reference.split(":")
    module = importlib.import_module(module_name)
    manifest = getattr(module, attribute)
    if manifest.name != definition.name:
        raise ValueError(f"Dataset {definition.name!r} manifest declares name {manifest.name!r}")
    return manifest


@cache
def get_dataset_layout(name: str) -> DatasetLayout:
    """Construct and cache one requested dataset layout."""
    from vidmap.datasets.base import DatasetLayout
    from vidmap.paths import dataset_paths

    definition = get_dataset_definition(name)
    paths = dataset_paths(name)
    manifest = _load_manifest(definition)
    scenes = definition.scenes if manifest is None else manifest.scenes
    return DatasetLayout(
        name=name,
        data_dir=paths.data,
        default_exp_dir=paths.experiments,
        default_cache_dir=paths.cache,
        testsets=paths.testsets,
        scenes=scenes,
        manifest=manifest,
    )
