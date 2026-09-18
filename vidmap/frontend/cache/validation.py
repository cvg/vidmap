"""Validation of cache metadata and completed artifact payloads."""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from vidmap.repro.hashing import canonical_h5_hash

from .metadata import (
    CACHE_METADATA_ATTR,
    CACHE_SCHEMA_VERSION,
    PAYLOAD_DATASET,
    CacheMetadataMismatch,
    canonical_json,
    canonical_value,
    fingerprint,
)

logger = logging.getLogger(__name__)


def _reject_json_constant(value: str):
    raise ValueError(f"invalid JSON constant {value}")


def read_cache_metadata(path: Path) -> dict[str, Any]:
    if not Path(path).is_file():
        raise CacheMetadataMismatch(f"{path}: cache artifact does not exist")
    try:
        with h5py.File(path, "r") as hfile:
            raw = hfile.attrs[CACHE_METADATA_ATTR]
    except (OSError, KeyError, TypeError, ValueError) as exc:
        raise CacheMetadataMismatch(f"{path}: missing or unreadable cache metadata") from exc

    try:
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        metadata = json.loads(raw, parse_constant=_reject_json_constant)
    except (TypeError, UnicodeDecodeError, ValueError) as exc:
        raise CacheMetadataMismatch(f"{path}: malformed cache metadata") from exc
    if not isinstance(metadata, dict):
        raise CacheMetadataMismatch(f"{path}: cache metadata is not an object")
    try:
        if metadata.get("complete"):
            payload = metadata.get("payload_fingerprint")
            expected_artifact = fingerprint(
                {
                    "identity": metadata.get("identity_fingerprint"),
                    "payload": payload,
                }
            )
            if payload is None or metadata.get("artifact_fingerprint") != expected_artifact:
                raise CacheMetadataMismatch(f"{path}: artifact fingerprint is missing or malformed")
    except (TypeError, ValueError) as exc:
        raise CacheMetadataMismatch(f"{path}: malformed cache metadata") from exc
    return metadata


def validate_cache_metadata(path: Path, expected: Mapping[str, Any]) -> dict[str, Any]:
    if not path.exists():
        raise CacheMetadataMismatch(f"{path}: cache artifact does not exist")
    actual = read_cache_metadata(path)
    if actual.get("schema_version") != CACHE_SCHEMA_VERSION:
        raise CacheMetadataMismatch(
            f"{path}: schema version {actual.get('schema_version')!r} does not match {CACHE_SCHEMA_VERSION}"
        )
    if not actual.get("complete", False):
        raise CacheMetadataMismatch(f"{path}: cache artifact is incomplete")
    for key, value in expected.items():
        if canonical_value(actual.get(key)) != canonical_value(value):
            raise CacheMetadataMismatch(f"{path}: {key} mismatch (cached {actual.get(key)!r}, expected {value!r})")
    validate_complete_payload(path, actual)
    return actual


def cache_is_valid(path: Path, expected: Mapping[str, Any]) -> bool:
    try:
        validate_cache_metadata(path, expected)
    except CacheMetadataMismatch as exc:
        logger.info("Ignoring stale cache: %s", exc)
        return False
    return True


def identity_mismatch(path: Path, actual: Mapping[str, Any], expected: Mapping[str, Any]) -> str | None:
    for key, value in expected.items():
        if key == "complete":
            continue
        if canonical_value(actual.get(key)) != canonical_value(value):
            return f"{path}: {key} mismatch (cached {actual.get(key)!r}, expected {value!r})"
    return None


def dataset_parent_names(hfile) -> list[str]:
    names = set()

    def collect(_, obj):
        if isinstance(obj, h5py.Dataset):
            names.add(obj.parent.name.strip("/"))

    hfile.visititems(collect)
    return sorted(names)


def _numeric(dataset) -> bool:
    return isinstance(dataset, h5py.Dataset) and dataset.dtype.kind in {
        "b",
        "f",
        "i",
        "u",
    }


def validate_incremental_item(hfile, name: str, metadata: Mapping[str, Any]) -> None:
    stage = metadata["stage"]
    if name not in hfile or not isinstance(hfile[name], h5py.Group):
        raise CacheMetadataMismatch(f"missing required group {name!r}")
    group = hfile[name]

    def require(dataset_name, *, numeric=True, ndim=None, last_dim=None):
        if dataset_name not in group or not isinstance(group[dataset_name], h5py.Dataset):
            raise CacheMetadataMismatch(f"{name!r}: missing dataset {dataset_name!r}")
        dataset = group[dataset_name]
        if numeric and not _numeric(dataset):
            raise CacheMetadataMismatch(f"{name!r}/{dataset_name}: expected numeric dtype")
        if ndim is not None and dataset.ndim != ndim:
            raise CacheMetadataMismatch(f"{name!r}/{dataset_name}: expected {ndim} dimensions")
        if last_dim is not None and (dataset.ndim == 0 or dataset.shape[-1] != last_dim):
            raise CacheMetadataMismatch(f"{name!r}/{dataset_name}: expected trailing dimension {last_dim}")
        return dataset

    if stage in {"salient_features", "sparse_features"}:
        require("keypoints", ndim=2, last_dim=2)
    elif stage in {"sparse_tracks", "extended_matches"}:
        if set(group) != {"matches0", "matching_scores0"}:
            raise CacheMetadataMismatch(f"{name!r}: unexpected match datasets {sorted(group)}")
        matches = require("matches0", numeric=False, ndim=1)
        if matches.dtype.kind not in {"i", "u"}:
            raise CacheMetadataMismatch(f"{name!r}/matches0: expected integer dtype")
        match_values = matches[...]
        if matches.dtype.kind == "u" and np.any(match_values > np.iinfo(np.int64).max):
            raise CacheMetadataMismatch(f"{name!r}/matches0: value exceeds int64 range")
        if np.any(match_values.astype(np.int64, copy=False) < -1):
            raise CacheMetadataMismatch(f"{name!r}/matches0: expected match indices or -1")
        scores = require("matching_scores0", ndim=1)
        if scores.dtype.kind != "f" or not np.isfinite(scores[...]).all():
            raise CacheMetadataMismatch(f"{name!r}/matching_scores0: expected finite floating-point values")
        if len(matches) != len(scores):
            raise CacheMetadataMismatch(f"{name!r}: match and score lengths differ")
    elif stage == "retrieval_features":
        descriptor = require("global_descriptor", ndim=1)
        if descriptor.size == 0 or not np.isfinite(descriptor[...]).all():
            raise CacheMetadataMismatch(f"{name!r}/global_descriptor: expected finite nonempty values")
    elif stage == "depth":
        depth = require("depth", ndim=1)
        valid = require("valid", ndim=1)
        if len(depth) != len(valid):
            raise CacheMetadataMismatch(f"{name!r}: depth and validity lengths differ")
        if valid.dtype.kind != "b":
            raise CacheMetadataMismatch(f"{name!r}/valid: expected boolean dtype")
        if not np.isfinite(depth[...]).all():
            raise CacheMetadataMismatch(f"{name!r}/depth: expected finite values")
        if "conf" in group:
            confidence = require("conf", ndim=1)
            if len(confidence) != len(depth) or not np.isfinite(confidence[...]).all():
                raise CacheMetadataMismatch(f"{name!r}/conf: expected finite values matching depth")
    elif stage == "full_depth":
        depth = require("depth", ndim=2)
        valid = require("valid", ndim=2)
        if depth.shape != valid.shape:
            raise CacheMetadataMismatch(f"{name!r}: full depth and validity shapes differ")
        if valid.dtype.kind != "b":
            raise CacheMetadataMismatch(f"{name!r}/valid: expected boolean dtype")
        if not np.isfinite(depth[...]).all():
            raise CacheMetadataMismatch(f"{name!r}/depth: expected finite values")
        if "conf" in group:
            confidence = require("conf", ndim=2)
            if confidence.shape != depth.shape or not np.isfinite(confidence[...]).all():
                raise CacheMetadataMismatch(f"{name!r}/conf: expected finite values matching full depth")
        for attribute in ("original_width", "original_height"):
            value = group.attrs[attribute]
            if not isinstance(value, (int, np.integer)) or isinstance(value, (bool, np.bool_)) or int(value) < 1:
                raise CacheMetadataMismatch(f"{name!r}: expected positive integer attribute {attribute!r}")

    if stage in {"geocalib_per_image", "geocalib_batch"} or (
        stage == "depth" and metadata["payload_format"] == "per-image-keypoint-depth-calibration"
    ):
        K = require("K", ndim=2)[...]
        if K.shape != (3, 3) or not np.isfinite(K).all() or min(K[0, 0], K[1, 1]) <= 0:
            raise CacheMetadataMismatch("Calibration requires finite positive PINHOLE intrinsics")
        if not np.array_equal(K[[0, 1, 2, 2, 2], [1, 0, 0, 1, 2]], [0, 0, 0, 0, 1]):
            raise CacheMetadataMismatch("PINHOLE K requires zero skew and a canonical homogeneous row")
        size = require("image_size")[...]
        if (
            size.shape != (2,)
            or size.dtype.kind == "b"
            or not np.isfinite(size).all()
            or np.any(size <= 0)
            or np.any(size != np.floor(size))
        ):
            raise CacheMetadataMismatch("Calibration requires positive integral original image dimensions")
        if stage.startswith("geocalib") or "focal_std_px" in group:
            std = require("focal_std_px", ndim=1)[...]
            if std.dtype.kind == "b" or not std.size or not np.isfinite(std).all() or np.any(std <= 0):
                raise CacheMetadataMismatch(
                    "Reported focal standard deviations must be positive finite numeric source-pixel values"
                )
            if stage == "geocalib_per_image" and len(std) != 1:
                raise CacheMetadataMismatch("Per-view GeoCalib requires one focal standard deviation")
            if stage == "geocalib_batch":
                try:
                    names = tuple(require("topk_images", numeric=False, ndim=1).asstr()[:])
                except (TypeError, UnicodeError) as error:
                    raise CacheMetadataMismatch("Shared GeoCalib requires valid encoded image names") from error
                if any(not name for name in names) or len(set(names)) != len(names) or len(names) != len(std):
                    raise CacheMetadataMismatch("Shared GeoCalib diagnostics must follow the selected image inventory")


def validate_incremental_items(hfile, expected_items: Sequence[str], metadata: Mapping[str, Any]) -> None:
    for name in expected_items:
        validate_incremental_item(hfile, name, metadata)
    if metadata["stage"] == "retrieval_features" and expected_items:
        widths = {hfile[name]["global_descriptor"].shape[0] for name in expected_items}
        if len(widths) != 1:
            raise CacheMetadataMismatch("retrieval descriptors have inconsistent widths")


def incremental_cache_is_complete(
    path: Path,
    expected: Mapping[str, Any],
    expected_items: Sequence[str],
) -> bool:
    try:
        validate_incremental_cache(path, expected, expected_items)
    except (
        CacheMetadataMismatch,
        OSError,
        KeyError,
        IndexError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as exc:
        logger.info("Ignoring stale cache: %s", exc)
        return False
    return True


def validate_incremental_cache(
    path: Path,
    expected: Mapping[str, Any],
    expected_items: Sequence[str],
) -> dict[str, Any]:
    """Validate one cache-only incremental artifact and its exact item plan."""
    actual = validate_cache_metadata(path, expected)
    if actual.get("expected_items_fingerprint") != fingerprint(list(expected_items)):
        raise CacheMetadataMismatch(f"{path}: expected-item set or ordering changed")
    return actual


def array_payload_fingerprint(data: np.ndarray) -> str:
    array = np.asarray(data)
    if array.dtype.kind in {"O", "S", "U"}:
        payload = {
            "dtype": str(array.dtype),
            "shape": list(array.shape),
            "values": array.tolist(),
        }
        return fingerprint(payload)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(canonical_json(list(array.shape)).encode("ascii"))
    digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def h5_payload_fingerprint(path: Path) -> str:
    """Hash H5 payload structure and values, excluding Vidmap cache metadata."""
    digest, _ = canonical_h5_hash(
        path,
        excluded_root_attrs=frozenset({CACHE_METADATA_ATTR}),
        include_storage_layout=False,
    )
    return digest


def validate_complete_payload(path: Path, metadata: Mapping[str, Any]) -> None:
    try:
        with h5py.File(path, "r") as hfile:
            if set(hfile.keys()) == {PAYLOAD_DATASET} and isinstance(hfile[PAYLOAD_DATASET], h5py.Dataset):
                data = hfile[PAYLOAD_DATASET][:]
                stage = metadata.get("stage")
                if stage in {"track_pairs", "retrieval_pairs"}:
                    if data.ndim != 2 or data.shape[1] != 2 or data.dtype.kind not in {"O", "S", "U"}:
                        raise CacheMetadataMismatch(f"{path}: malformed pair payload")
                    decode_pair_array(path, data)
                payload_fingerprint = array_payload_fingerprint(data)
            else:
                expected_items = metadata.get("expected_items")
                if not isinstance(expected_items, list):
                    raise CacheMetadataMismatch(f"{path}: expected-item declaration is missing")
                if metadata.get("expected_items_fingerprint") != fingerprint(expected_items):
                    raise CacheMetadataMismatch(f"{path}: expected-item declaration is malformed")
                missing = [name for name in expected_items if name not in hfile]
                if missing:
                    raise CacheMetadataMismatch(f"{path}: missing {len(missing)} expected items")
                validate_incremental_items(hfile, expected_items, metadata)
                extras = sorted(set(dataset_parent_names(hfile)) - set(expected_items))
                if extras:
                    raise CacheMetadataMismatch(f"{path}: found {len(extras)} unplanned items")
                payload_fingerprint = h5_payload_fingerprint(path)
    except CacheMetadataMismatch:
        raise
    except (OSError, KeyError, IndexError, RuntimeError, TypeError, ValueError) as exc:
        raise CacheMetadataMismatch(f"{path}: malformed or unreadable cache payload") from exc
    if metadata.get("payload_fingerprint") != payload_fingerprint:
        raise CacheMetadataMismatch(f"{path}: payload fingerprint mismatch")


def decode_pair_array(path: Path, data: np.ndarray) -> list[tuple[str, str]]:
    try:
        pairs = []
        for pair in data:
            decoded = []
            for item in pair:
                if isinstance(item, bytes):
                    decoded.append(item.decode("utf-8"))
                elif isinstance(item, str):
                    decoded.append(item)
                else:
                    raise TypeError(f"unexpected pair item type {type(item).__name__}")
            pairs.append(tuple(decoded))
        return pairs
    except (UnicodeDecodeError, TypeError, ValueError) as exc:
        raise CacheMetadataMismatch(f"{path}: malformed pair payload") from exc
