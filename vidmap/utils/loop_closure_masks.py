"""Validated, non-executable storage for loop-closure match masks."""

from __future__ import annotations

import logging
import uuid
from collections.abc import Mapping
from pathlib import Path

import h5py
import numpy as np

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1
_INDEX_WIDTH = 8


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
    """Atomically write ordered image pairs and boolean masks to HDF5."""

    if not isinstance(loop_closure_masks, Mapping):
        raise TypeError("loop_closure_masks must be a mapping")
    path = Path(path)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    seen_pairs: set[frozenset[str]] = set()
    try:
        with h5py.File(temporary, "w") as hfile:
            hfile.attrs["schema_version"] = SCHEMA_VERSION
            pairs = hfile.create_group("pairs", track_order=True)
            for index, (raw_pair, raw_mask) in enumerate(loop_closure_masks.items()):
                pair = _validated_pair(raw_pair)
                undirected = frozenset(pair)
                if undirected in seen_pairs:
                    raise ValueError(f"Duplicate undirected loop-closure mask pair: {pair!r}")
                seen_pairs.add(undirected)
                mask = _validated_mask(raw_mask, pair=pair)
                entry = pairs.create_group(f"{index:0{_INDEX_WIDTH}d}")
                entry.attrs["first"] = pair[0]
                entry.attrs["second"] = pair[1]
                entry.create_dataset("mask", data=mask, dtype=np.bool_, track_times=False)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    logger.info("Wrote %d loop-closure masks to %s", len(loop_closure_masks), path)


def _read_name(entry: h5py.Group, name: str, *, path: Path) -> str:
    value = entry.attrs.get(name)
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if not isinstance(value, str) or not value:
        raise ValueError(f"Invalid loop-closure mask {name!r} in {path}")
    return value


def _require_hard_link(group: h5py.Group | h5py.File, name: str, *, path: Path) -> None:
    if not isinstance(group.get(name, getlink=True), h5py.HardLink):
        raise ValueError(f"Loop-closure mask schema contains a non-local link {name!r}: {path}")


def read_loop_closure_masks(path: str | Path) -> dict[tuple[str, str], np.ndarray]:
    """Read and validate ordered loop-closure masks without executable decoding."""

    path = Path(path)
    masks: dict[tuple[str, str], np.ndarray] = {}
    seen_pairs: set[frozenset[str]] = set()
    try:
        with h5py.File(path, "r") as hfile:
            schema_version = hfile.attrs.get("schema_version")
            if (
                set(hfile.attrs) != {"schema_version"}
                or isinstance(schema_version, (bool, np.bool_))
                or not isinstance(schema_version, (int, np.integer))
                or int(schema_version) != SCHEMA_VERSION
            ):
                raise ValueError(f"Unsupported loop-closure mask schema: {path}")
            if set(hfile) != {"pairs"}:
                raise ValueError(f"Invalid loop-closure mask root: {path}")
            _require_hard_link(hfile, "pairs", path=path)
            if not isinstance(hfile["pairs"], h5py.Group):
                raise ValueError(f"Invalid loop-closure mask root: {path}")
            pairs = hfile["pairs"]
            if set(pairs.attrs):
                raise ValueError(f"Invalid loop-closure mask pair metadata: {path}")
            expected_names = tuple(f"{index:0{_INDEX_WIDTH}d}" for index in range(len(pairs)))
            if tuple(pairs) != expected_names:
                raise ValueError(f"Loop-closure mask entries are not contiguous and ordered: {path}")
            for entry_name in expected_names:
                _require_hard_link(pairs, entry_name, path=path)
                entry = pairs[entry_name]
                if (
                    not isinstance(entry, h5py.Group)
                    or set(entry.attrs) != {"first", "second"}
                    or set(entry) != {"mask"}
                ):
                    raise ValueError(f"Invalid loop-closure mask entry {entry_name!r}: {path}")
                _require_hard_link(entry, "mask", path=path)
                if not isinstance(entry["mask"], h5py.Dataset):
                    raise ValueError(f"Invalid loop-closure mask entry {entry_name!r}: {path}")
                pair = _validated_pair(
                    (
                        _read_name(entry, "first", path=path),
                        _read_name(entry, "second", path=path),
                    )
                )
                undirected = frozenset(pair)
                if undirected in seen_pairs:
                    raise ValueError(f"Duplicate undirected loop-closure mask pair {pair!r}: {path}")
                seen_pairs.add(undirected)
                dataset = entry["mask"]
                if (
                    set(dataset.attrs)
                    or dataset.dtype != np.dtype(np.bool_)
                    or dataset.ndim != 1
                    or dataset.is_virtual
                    or dataset.external is not None
                ):
                    raise ValueError(f"Invalid loop-closure mask array for {pair!r}: {path}")
                masks[pair] = dataset[:]
    except OSError as error:
        raise ValueError(f"Invalid loop-closure mask file: {path}") from error
    logger.info("Read %d loop-closure masks from %s", len(masks), path)
    return masks
