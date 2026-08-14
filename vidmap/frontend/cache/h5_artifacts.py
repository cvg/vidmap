"""Read, write, and resume frontend H5 caches."""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from .metadata import CACHE_METADATA_ATTR, PAYLOAD_DATASET, CacheMetadataMismatch, canonical_json, fingerprint
from .validation import (
    array_payload_fingerprint,
    dataset_parent_names,
    decode_pair_array,
    h5_payload_fingerprint,
    identity_mismatch,
    read_cache_metadata,
    validate_cache_metadata,
    validate_incremental_item,
    validate_incremental_items,
)

logger = logging.getLogger(__name__)


def prepare_incremental_cache(path: Path, expected: Mapping[str, Any], *, overwrite=False) -> bool:
    """Prepare a resumable H5 cache and return whether its identity matched."""
    path = Path(path)
    matched = False
    mismatch = None
    actual = None
    if path.exists() and not overwrite:
        try:
            actual = read_cache_metadata(path)
            mismatch = identity_mismatch(path, actual, expected)
            matched = mismatch is None
            if matched and actual.get("complete", False):
                try:
                    validate_cache_metadata(path, expected)
                except (
                    CacheMetadataMismatch,
                    OSError,
                    KeyError,
                    TypeError,
                    ValueError,
                ) as exc:
                    mismatch = str(exc)
                    matched = False
        except (CacheMetadataMismatch, OSError, KeyError, TypeError, ValueError) as exc:
            mismatch = str(exc)
            matched = False
    if path.exists() and (overwrite or not matched):
        if mismatch:
            logger.info("Ignoring stale cache: %s", mismatch)
        path.unlink()
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        with h5py.File(path, "w") as hfile:
            metadata = dict(expected)
            metadata["complete"] = False
            hfile.attrs[CACHE_METADATA_ATTR] = canonical_json(metadata)
    return matched


def attach_incremental_cache_identity(path: Path, expected: Mapping[str, Any]) -> None:
    """Attach cache identity metadata after a producer closes its H5 file."""
    metadata = dict(expected)
    metadata["complete"] = False
    try:
        with h5py.File(path, "a") as hfile:
            hfile.attrs[CACHE_METADATA_ATTR] = canonical_json(metadata)
    except (OSError, TypeError, ValueError) as exc:
        raise CacheMetadataMismatch(f"{path}: cannot attach incremental cache identity") from exc


def inspect_incremental_items(
    path: Path,
    requested_items: Sequence[str],
    expected: Mapping[str, Any] | None = None,
    *,
    repair_malformed: bool = False,
) -> tuple[list[str], list[str]]:
    """Partition requested H5 paths without accepting malformed files."""
    requested = list(dict.fromkeys(requested_items))
    try:
        with h5py.File(path, "a" if repair_malformed else "r") as hfile:
            present = []
            for name in requested:
                if name not in hfile:
                    continue
                try:
                    if expected is not None:
                        validate_incremental_item(hfile, name, expected["stage"])
                except CacheMetadataMismatch:
                    if not repair_malformed:
                        raise
                    del hfile[name]
                    continue
                present.append(name)
    except CacheMetadataMismatch as exc:
        raise CacheMetadataMismatch(f"{path}: malformed incremental cache item: {exc}") from exc
    except (OSError, KeyError, TypeError, ValueError) as exc:
        raise CacheMetadataMismatch(f"{path}: cannot inspect incremental cache items") from exc
    present_set = set(present)
    return present, [name for name in requested if name not in present_set]


def prune_incremental_items(path: Path, expected_items: Sequence[str]) -> list[str]:
    """Remove stale leaf groups that are outside the deterministic item plan."""
    expected = set(expected_items)
    with h5py.File(path, "a") as hfile:
        extras = sorted(set(dataset_parent_names(hfile)) - expected)
        for name in extras:
            if name in hfile:
                del hfile[name]
            parent = name.rpartition("/")[0]
            while parent and parent in hfile and isinstance(hfile[parent], h5py.Group) and not hfile[parent]:
                next_parent = parent.rpartition("/")[0]
                del hfile[parent]
                parent = next_parent
    return extras


def mark_incremental_cache_complete(path: Path, expected: Mapping[str, Any], expected_items: Sequence[str]) -> None:
    try:
        with h5py.File(path, "r") as hfile:
            missing = [name for name in expected_items if name not in hfile]
            present = [name for name in expected_items if name in hfile]
            validate_incremental_items(hfile, present, expected["stage"])
            extras = sorted(set(dataset_parent_names(hfile)) - set(expected_items))
    except CacheMetadataMismatch:
        raise
    except (OSError, KeyError, IndexError, RuntimeError, TypeError, ValueError) as exc:
        raise CacheMetadataMismatch(f"{path}: cannot validate incremental cache completion") from exc
    if missing:
        raise CacheMetadataMismatch(f"{path}: cannot mark complete; missing {len(missing)} expected items")
    if extras:
        raise CacheMetadataMismatch(f"{path}: cannot mark complete; found {len(extras)} unplanned items")
    metadata = dict(expected)
    metadata["complete"] = True
    metadata["expected_items"] = list(expected_items)
    metadata["expected_items_fingerprint"] = fingerprint(list(expected_items))
    metadata["payload_fingerprint"] = h5_payload_fingerprint(path)
    metadata["artifact_fingerprint"] = fingerprint(
        {
            "identity": metadata["identity_fingerprint"],
            "payload": metadata["payload_fingerprint"],
        }
    )
    with h5py.File(path, "a") as hfile:
        hfile.attrs[CACHE_METADATA_ATTR] = canonical_json(metadata)


def write_single_dataset(
    path: Path,
    data: Any,
    metadata: Mapping[str, Any],
    *,
    dtype=None,
) -> None:
    """Write a complete single-purpose H5 artifact."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    array = np.asarray(data, dtype=object if dtype is not None else None)
    final_metadata = dict(metadata)
    final_metadata["payload_fingerprint"] = array_payload_fingerprint(array)
    final_metadata["artifact_fingerprint"] = fingerprint(
        {
            "identity": final_metadata["identity_fingerprint"],
            "payload": final_metadata["payload_fingerprint"],
        }
    )
    final_metadata["complete"] = True

    tmp_path = path.with_name(f".{path.name}.tmp")
    tmp_path.unlink(missing_ok=True)
    try:
        with h5py.File(tmp_path, "w") as hfile:
            hfile.create_dataset(PAYLOAD_DATASET, data=array, dtype=dtype)
            hfile.attrs[CACHE_METADATA_ATTR] = canonical_json(final_metadata)
        tmp_path.replace(path)
    finally:
        tmp_path.unlink(missing_ok=True)


def read_single_dataset(path: Path, expected: Mapping[str, Any]) -> np.ndarray:
    validate_cache_metadata(path, expected)
    try:
        with h5py.File(path, "r") as hfile:
            if set(hfile.keys()) != {PAYLOAD_DATASET}:
                raise CacheMetadataMismatch(f"{path}: expected only the '{PAYLOAD_DATASET}' payload dataset")
            data = hfile[PAYLOAD_DATASET][:]
    except (OSError, KeyError, TypeError, ValueError) as exc:
        raise CacheMetadataMismatch(f"{path}: unreadable payload") from exc
    actual = read_cache_metadata(path)
    if actual.get("payload_fingerprint") != array_payload_fingerprint(data):
        raise CacheMetadataMismatch(f"{path}: payload fingerprint mismatch")
    return data


def write_pair_artifact(path: Path, pairs: Sequence[Sequence[str]], metadata: Mapping[str, Any]) -> None:
    normalized_pairs = []
    for index, pair in enumerate(pairs):
        if isinstance(pair, (str, bytes)):
            raise ValueError(f"Pair {index} must contain exactly two image names")
        pair = tuple(pair)
        if len(pair) != 2:
            raise ValueError(f"Pair {index} must contain exactly two image names, got {len(pair)}")
        if not all(isinstance(name, str) for name in pair):
            raise TypeError(f"Pair {index} image names must be strings")
        normalized_pairs.append(pair)

    string_dtype = h5py.string_dtype(encoding="utf-8")
    pair_array = np.empty((len(normalized_pairs), 2), dtype=object)
    if normalized_pairs:
        pair_array[:] = normalized_pairs
    write_single_dataset(path, pair_array, metadata, dtype=string_dtype)


def read_pair_artifact(path: Path, expected: Mapping[str, Any]) -> list[tuple[str, str]]:
    try:
        data = read_single_dataset(path, expected)
        if data.ndim != 2 or data.shape[1] != 2:
            raise CacheMetadataMismatch(f"{path}: pair payload must have shape (N, 2), got {data.shape}")
        if data.dtype.kind not in {"O", "S", "U"}:
            raise CacheMetadataMismatch(f"{path}: pair payload must use a string dtype, got {data.dtype}")
        return decode_pair_array(path, data)
    except CacheMetadataMismatch:
        raise
    except (UnicodeDecodeError, TypeError, ValueError) as exc:
        raise CacheMetadataMismatch(f"{path}: malformed pair payload") from exc
