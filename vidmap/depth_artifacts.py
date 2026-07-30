"""Names, references, and scale metadata for persisted depth artifacts."""

from __future__ import annotations

import json
import os
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from vidmap.mapper.inputs import MapperInputs

FULL_DEPTH_MAPS_NAME = "full_depth_maps.h5"
REFERENCE_NAME = "depth_lift.json"
REFERENCE_SCHEMA_VERSION = 1
DEPTH_SCALES_NAME = "depth_lift_scales.json"
DEPTH_SCALES_SCHEMA_VERSION = 1


def write_reference(run: Path, mapper_inputs: MapperInputs) -> Path | None:
    """Record the full- and sampled-depth artifacts associated with a run."""
    full_path = mapper_inputs.full_depth_maps_path
    if full_path is None:
        return None
    return write_reference_paths(run, full_path, mapper_inputs.depth_maps_path)


def write_reference_paths(run: Path, full_path: Path, sampled_path: Path) -> Path:
    """Atomically attach explicit full- and sampled-depth paths to a run."""
    run = Path(run).resolve()
    run.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": REFERENCE_SCHEMA_VERSION,
        "full_depth_maps": os.path.relpath(Path(full_path).resolve(), run),
        "sampled_depth_maps": os.path.relpath(Path(sampled_path).resolve(), run),
    }
    return _write_json_atomic(run / REFERENCE_NAME, payload)


def write_depth_scales(run: Path, values: Any, *, logarithmic: bool = False) -> Path:
    """Atomically retain final optimized per-image depth scales."""
    scales = {int(key): float(value) for key, value in dict(values).items()}
    if logarithmic:
        scales = {key: float(np.exp(value)) for key, value in scales.items()}
    payload = {
        "schema_version": DEPTH_SCALES_SCHEMA_VERSION,
        "scale_by_image_id": {str(key): value for key, value in sorted(scales.items())},
    }
    return _write_json_atomic(Path(run) / DEPTH_SCALES_NAME, payload)


def load_depth_scales(run: Path) -> dict[int, float]:
    """Load and validate optimized per-image depth scales, if present."""
    path = Path(run) / DEPTH_SCALES_NAME
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or set(payload) != {
        "schema_version",
        "scale_by_image_id",
    }:
        raise ValueError(f"Invalid depth-lift scales: {path}")
    if payload["schema_version"] != DEPTH_SCALES_SCHEMA_VERSION or not isinstance(payload["scale_by_image_id"], dict):
        raise ValueError(f"Invalid depth-lift scales: {path}")
    result = {int(key): float(value) for key, value in payload["scale_by_image_id"].items()}
    if any(not np.isfinite(value) or value <= 0.0 for value in result.values()):
        raise ValueError(f"Invalid depth-lift scales: {path}")
    return result


def resolve_depth_paths(run: Path, explicit: Path | None = None) -> tuple[Path, Path | None]:
    """Resolve the full maps and optional sampled priors associated with a run."""
    if explicit is not None:
        full_path = Path(explicit).expanduser().resolve()
        if not full_path.is_file():
            raise FileNotFoundError(f"Full depth-map artifact is missing: {full_path}")
        sampled = full_path.with_name("depth_maps.h5")
        return full_path, sampled if sampled.is_file() else None

    run = Path(run)
    reference_path = run / REFERENCE_NAME
    if not reference_path.is_file():
        local_full = run / FULL_DEPTH_MAPS_NAME
        local_sampled = run / "mapper_inputs" / "depth_maps.h5"
        if local_full.is_file():
            return (
                local_full.resolve(),
                local_sampled.resolve() if local_sampled.is_file() else None,
            )
        raise FileNotFoundError(
            f"Depth lift needs {reference_path}; rerun frontend with --cache-depth-maps "
            "and map from the resulting mapper inputs, or pass --depth-maps."
        )
    payload = json.loads(reference_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or set(payload) != {
        "schema_version",
        "full_depth_maps",
        "sampled_depth_maps",
    }:
        raise ValueError(f"Invalid depth-lift reference fields: {reference_path}")
    if payload["schema_version"] != REFERENCE_SCHEMA_VERSION:
        raise ValueError(f"Unsupported depth-lift reference schema: {reference_path}")
    full_path = _referenced_path(reference_path, payload["full_depth_maps"])
    sampled_path = _referenced_path(reference_path, payload["sampled_depth_maps"])
    if not full_path.is_file():
        raise FileNotFoundError(f"Referenced full depth-map artifact is missing: {full_path}")
    return full_path, sampled_path if sampled_path.is_file() else None


def _referenced_path(reference: Path, value: Any) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"Invalid depth-lift path in {reference}")
    path = Path(value).expanduser()
    return path if path.is_absolute() else (reference.parent / path).resolve()


def _write_json_atomic(output: Path, payload: dict[str, Any]) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    return output
