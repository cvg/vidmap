"""Validated, non-executable storage for loop-closure match masks."""

from __future__ import annotations

import json
import logging
import tempfile
from collections.abc import Mapping
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1


def _validated_pair(pair: object) -> tuple[str, str]:
    if (
        not isinstance(pair, tuple)
        or len(pair) != 2
        or any(not isinstance(name, str) or not name for name in pair)
        or pair[0] == pair[1]
    ):
        raise ValueError(f"Invalid loop-closure mask image pair: {pair!r}")
    return pair


def _validated_mask(mask: object, *, pair: tuple[str, str]) -> np.ndarray:
    array = np.asarray(mask)
    if array.dtype != np.bool_ or array.ndim != 1:
        raise ValueError(f"Loop-closure mask for {pair!r} must be a one-dimensional boolean array")
    return array


def write_loop_closure_masks(
    loop_closure_masks: Mapping[tuple[str, str], np.ndarray],
    path: str | Path,
) -> None:
    """Atomically store dense in-memory masks as sparse match indices."""
    if not isinstance(loop_closure_masks, Mapping):
        raise TypeError("loop_closure_masks must be a mapping")

    pairs = []
    seen_pairs: set[frozenset[str]] = set()
    for raw_pair, raw_mask in loop_closure_masks.items():
        pair = _validated_pair(raw_pair)
        undirected = frozenset(pair)
        if undirected in seen_pairs:
            raise ValueError(f"Duplicate undirected loop-closure mask pair: {pair!r}")
        seen_pairs.add(undirected)
        mask = _validated_mask(raw_mask, pair=pair)
        pairs.append(
            {
                "first": pair[0],
                "second": pair[1],
                "matchCount": len(mask),
                "loopClosureMatchIndices": np.flatnonzero(mask).astype(int).tolist(),
            }
        )

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            json.dump(
                {"schemaVersion": SCHEMA_VERSION, "pairs": pairs},
                stream,
                separators=(",", ":"),
            )
            stream.write("\n")
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    logger.info("Wrote %d loop-closure match masks to %s", len(pairs), path)


def _validated_nonnegative_integer(value: object, *, label: str, path: Path) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"Invalid loop-closure mask {label} in {path}")
    return value


def read_loop_closure_masks(path: str | Path) -> dict[tuple[str, str], np.ndarray]:
    """Read sparse match indices and reconstruct their dense Boolean masks."""
    path = Path(path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"Invalid loop-closure mask file: {path}") from error
    if (
        not isinstance(payload, dict)
        or set(payload) != {"schemaVersion", "pairs"}
        or payload["schemaVersion"] != SCHEMA_VERSION
        or isinstance(payload["schemaVersion"], bool)
        or not isinstance(payload["pairs"], list)
    ):
        raise ValueError(f"Invalid loop-closure mask schema: {path}")

    masks: dict[tuple[str, str], np.ndarray] = {}
    seen_pairs: set[frozenset[str]] = set()
    for entry in payload["pairs"]:
        if not isinstance(entry, dict) or set(entry) != {
            "first",
            "second",
            "matchCount",
            "loopClosureMatchIndices",
        }:
            raise ValueError(f"Invalid loop-closure mask entry in {path}")
        pair = _validated_pair((entry["first"], entry["second"]))
        undirected = frozenset(pair)
        if undirected in seen_pairs:
            raise ValueError(f"Duplicate undirected loop-closure mask pair {pair!r} in {path}")
        seen_pairs.add(undirected)
        match_count = _validated_nonnegative_integer(
            entry["matchCount"],
            label="count",
            path=path,
        )
        indices = entry["loopClosureMatchIndices"]
        if not isinstance(indices, list):
            raise ValueError(f"Invalid loop-closure mask indices in {path}")
        validated_indices = [_validated_nonnegative_integer(index, label="index", path=path) for index in indices]
        if any(index >= match_count for index in validated_indices) or validated_indices != sorted(
            set(validated_indices)
        ):
            raise ValueError(f"Invalid loop-closure mask indices in {path}")
        mask = np.zeros(match_count, dtype=np.bool_)
        mask[validated_indices] = True
        masks[pair] = mask

    logger.info("Read %d loop-closure masks from %s", len(masks), path)
    return masks
