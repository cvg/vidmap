"""Write canonical mapper stage evidence for byte checks."""

from __future__ import annotations

import json
import logging
import pickle
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..options import ReplayCacheOptions
from .evidence.scene import track_records_summary
from .evidence.stages import ba_start_summary

logger = logging.getLogger(__name__)

_STAGES = {
    "all",
    "database",
    "db_to_glomap",
    "relative_pose",
    "ra",
    "tracks",
    "gp1",
    "gp2",
    "ba_start",
}


def _stage_selected(stage_spec: str, stage: str) -> bool:
    selected = {part.strip() for part in stage_spec.split(",") if part.strip()}
    if not selected:
        raise ValueError("replay_cache.write_stage must select at least one stage")
    unknown = selected - _STAGES
    if unknown:
        raise ValueError(f"Unknown replay_cache.write_stage values: {sorted(unknown)}")
    return "all" in selected or stage in selected


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if is_dataclass(value) and not isinstance(value, type):
        return {field.name: _jsonable(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


class ReplayCache:
    """Own the output paths and writes for replay evidence."""

    def __init__(self, options: ReplayCacheOptions, sfm_outputs_dir: Path) -> None:
        self.options = options
        self.mode = options.mode
        self._ba_start_tracks_summary: dict | None = None
        if options.root is None:
            self.root = Path(sfm_outputs_dir) / "replay_cache"
        else:
            self.root = Path(options.root).expanduser()

    def write_enabled(self, stage: str) -> bool:
        if self.mode != "byte_check":
            return False
        stage_spec = "all" if self.options.write_stage is None else self.options.write_stage
        return _stage_selected(stage_spec, stage)

    def _path(self, stage: str, name: str) -> Path:
        return self.root / stage / name

    def write_pickle(self, stage: str, name: str, value: Any) -> None:
        path = self._path(stage, name)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as file:
            pickle.dump(value, file)
        logger.info("[REPLAY:%s] wrote %s", stage, path)

    def write_json(self, stage: str, name: str, value: dict) -> None:
        path = self._path(stage, name)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(_jsonable(value), indent=2, sort_keys=True) + "\n")
        logger.info("[REPLAY:%s] wrote %s", stage, path)

    def capture_ba_start_tracks(self, tracks: dict) -> None:
        """Retain the native track summary lost by COLMAP serialization."""
        if self.write_enabled("ba_start"):
            self._ba_start_tracks_summary = track_records_summary(tracks)

    def write_ba_start_summary(self, reconstruction: Any, solve_state: Any) -> None:
        if not self.write_enabled("ba_start"):
            return
        if self._ba_start_tracks_summary is None:
            raise RuntimeError("BA-start replay requires the post-GP track summary")
        self.write_json(
            "ba_start",
            "summary.json",
            ba_start_summary(
                reconstruction,
                solve_state,
                track_summary=self._ba_start_tracks_summary,
            ),
        )
