from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from vidmap.datasets.layouts import get_dataset_layout
from vidmap.datasets.names import DATASET_DEFINITIONS


def _canonical_layouts():
    return {definition.name: get_dataset_layout(definition.name) for definition in DATASET_DEFINITIONS}


def select_dataset_root(
    config: dict[str, Any],
    run: Path | None,
    valid: Callable[[Path], bool],
    *,
    purpose: str,
) -> tuple[Path | None, tuple[str, ...]]:
    """Select one configured local dataset root, rejecting ambiguous fallbacks."""
    layouts = _canonical_layouts()
    roots = {name: layout.data_dir for name, layout in layouts.items()}
    configured = configured_dataset_names(config, run, layouts)
    candidates = tuple((name, roots[name]) for name in configured) if configured else tuple(roots.items())
    matches = tuple((name, root) for name, root in candidates if valid(root))
    if len(matches) > 1:
        choices = ", ".join(f"{name}={root}" for name, root in matches)
        raise ValueError(f"{purpose} dataset root is ambiguous: {choices}")
    return (matches[0][1] if matches else None), configured


def configured_dataset_names(
    config: dict[str, Any],
    run: Path | None,
    layouts: dict[str, Any],
) -> tuple[str, ...]:
    obsolete = sorted({"benchmark_dataset", "dataset_name", "dataset"} & config.keys())
    if obsolete:
        raise ValueError(
            "Playback configs do not accept dataset identity fields "
            f"{obsolete}; locate the run under the dataset experiment root"
        )
    requested = []
    locations = []
    if "output_root" in config and config["output_root"]:
        locations.append(Path(config["output_root"]).expanduser())
    if run is not None:
        locations.append(Path(run).expanduser())
    for name, layout in layouts.items():
        if any(_is_relative_to(location, layout.default_exp_dir) for location in locations):
            requested.append(name)
    unique = tuple(dict.fromkeys(requested))
    if len(unique) > 1:
        raise ValueError("Run identifies multiple datasets: " + ", ".join(unique))
    return unique


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True
