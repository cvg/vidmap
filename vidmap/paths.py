"""Resolve user-owned dataset paths independently of the installed package."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

from vidmap.datasets.names import SUPPORTED_DATASETS

PATHS_FILENAME = "vidmap-paths.toml"
_PATH_FIELDS = ("data", "cache", "experiments", "testsets")
_ENV_SUFFIXES = {"data": "DATA", "cache": "CACHE", "experiments": "EXP", "testsets": "TESTSETS"}


@dataclass(frozen=True)
class DatasetPaths:
    """Resolved storage paths for one benchmark dataset."""

    data: Path
    cache: Path
    experiments: Path
    testsets: Path


def _default_paths(dataset: str, working_directory: Path) -> DatasetPaths:
    local = working_directory / "local"
    return DatasetPaths(
        data=local / "datasets" / dataset,
        cache=local / "cache" / dataset,
        experiments=local / "experiments" / dataset,
        testsets=local / "testsets" / dataset,
    )


def _configured_paths(config_file: Path) -> dict[str, dict[str, Path]]:
    if not config_file.is_file():
        return {}
    with config_file.open("rb") as file:
        raw = tomllib.load(file)
    unknown_datasets = sorted(set(raw) - set(SUPPORTED_DATASETS))
    if unknown_datasets:
        raise ValueError(f"Unknown datasets in {config_file}: {unknown_datasets}")
    configured = {}
    for dataset, values in raw.items():
        if not isinstance(values, Mapping):
            raise ValueError(f"{config_file}: [{dataset}] must be a table")
        unknown_fields = sorted(set(values) - set(_PATH_FIELDS))
        if unknown_fields:
            raise ValueError(f"{config_file}: [{dataset}] has unknown fields {unknown_fields}")
        paths = {}
        for field, value in values.items():
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{config_file}: [{dataset}].{field} must be a non-empty path string")
            path = Path(value).expanduser()
            paths[field] = path if path.is_absolute() else (config_file.parent / path).resolve()
        configured[dataset] = paths
    return configured


def dataset_paths(
    dataset: str,
    *,
    working_directory: Path | None = None,
    config_file: Path | None = None,
    environ: Mapping[str, str] | None = None,
) -> DatasetPaths:
    """Resolve environment, working-directory TOML, and local defaults."""
    if dataset not in SUPPORTED_DATASETS:
        raise ValueError(f"Unsupported dataset {dataset!r}")
    working_directory = Path.cwd() if working_directory is None else working_directory
    working_directory = working_directory.expanduser().resolve()
    config_file = working_directory / PATHS_FILENAME if config_file is None else config_file
    config_file = config_file.expanduser()
    if not config_file.is_absolute():
        config_file = working_directory / config_file
    config_file = config_file.resolve()
    environ = os.environ if environ is None else environ
    defaults = _default_paths(dataset, working_directory)
    configured = _configured_paths(config_file).get(dataset, {})
    variable = dataset.upper()

    def resolve(field: str, default: Path) -> Path:
        environment_name = f"VIDMAP_{variable}_{_ENV_SUFFIXES[field]}_DIR"
        value = environ.get(environment_name, configured.get(field, default))
        path = Path(value).expanduser()
        return (working_directory / path).resolve() if not path.is_absolute() else path

    return DatasetPaths(
        data=resolve("data", defaults.data),
        cache=resolve("cache", defaults.cache),
        experiments=resolve("experiments", defaults.experiments),
        testsets=resolve("testsets", defaults.testsets),
    )
