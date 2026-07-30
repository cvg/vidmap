"""Semantic cache identities and deterministic fingerprints."""

from __future__ import annotations

import dataclasses
import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np

CACHE_METADATA_ATTR = "vidmap_cache_metadata"
CACHE_SCHEMA_VERSION = 1
PAYLOAD_DATASET = "data"


class CacheMetadataMismatch(RuntimeError):
    """Raised when cache-only loading encounters a stale cache artifact."""


def _immutable_snapshot(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({key: _immutable_snapshot(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_immutable_snapshot(item) for item in value)
    if isinstance(value, set):
        return frozenset(_immutable_snapshot(item) for item in value)
    return value


def canonical_value(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        value = dataclasses.asdict(value)
    elif hasattr(value, "model_dump"):
        value = value.model_dump(mode="python")
    elif hasattr(value, "to_container"):
        value = value.to_container(resolve=True)

    if isinstance(value, Mapping):
        if all(isinstance(key, str) for key in value):
            return {key: canonical_value(value[key]) for key in sorted(value)}
        entries = [(canonical_value(key), canonical_value(item)) for key, item in value.items()]
        entries.sort(key=lambda item: canonical_json(item[0]))
        return {"__mapping__": [[key, item] for key, item in entries]}
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.ndarray):
        return [canonical_value(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, tuple):
        return [canonical_value(item) for item in value]
    if isinstance(value, list):
        return [canonical_value(item) for item in value]
    if isinstance(value, set):
        return [canonical_value(item) for item in sorted(value, key=repr)]
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise TypeError(f"Unsupported cache fingerprint value: {type(value).__name__}")


def canonical_json(value: Any) -> str:
    """Serialize semantic values deterministically while preserving list order."""
    return json.dumps(
        canonical_value(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def fingerprint(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def file_fingerprint(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ordered_files_fingerprint(root: Path, names: Sequence[str]) -> str:
    return fingerprint([[name, file_fingerprint(Path(root) / name)] for name in names])


def semantic_config(value: Any, *, exclude_fields: frozenset[str] = frozenset()) -> Any:
    """Project a config into semantic identity with explicit stage-local exclusions."""
    canonical = canonical_value(value)
    if isinstance(canonical, dict):
        return {
            key: semantic_config(item, exclude_fields=exclude_fields)
            for key, item in canonical.items()
            if key not in exclude_fields
        }
    if isinstance(canonical, list):
        return [semantic_config(item, exclude_fields=exclude_fields) for item in canonical]
    return canonical


def cache_metadata(
    *,
    stage: str,
    config: Any,
    ordered_inputs: Any,
    upstream: Mapping[str, str] | None = None,
    payload_format: str,
    payload_version: int = 1,
    complete: bool = True,
    nonsemantic_config_fields: frozenset[str] = frozenset(),
) -> dict[str, Any]:
    upstream = {} if upstream is None else dict(upstream)
    identity = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "stage": stage,
        "config_fingerprint": fingerprint(semantic_config(config, exclude_fields=nonsemantic_config_fields)),
        "ordered_input_fingerprint": fingerprint(ordered_inputs),
        "upstream_fingerprints": upstream,
        "payload_format": payload_format,
        "payload_version": payload_version,
    }
    return {
        **identity,
        "identity_fingerprint": fingerprint(identity),
        "complete": complete,
    }


@dataclass(frozen=True)
class CompleteArtifactContract:
    """Validated metadata for one complete cache artifact."""

    metadata: Mapping[str, Any]

    def __post_init__(self):
        _validate_completed_contract_metadata(self.metadata)
        object.__setattr__(self, "metadata", _immutable_snapshot(self.metadata))

    @property
    def fingerprint(self) -> str:
        return artifact_fingerprint(self.metadata)


@dataclass(frozen=True)
class IncrementalArtifactContract:
    """Validated metadata and item plan for one incremental H5 cache."""

    metadata: Mapping[str, Any]
    expected_items: tuple[str, ...]

    def __post_init__(self):
        expected_items = tuple(self.expected_items)
        _validate_completed_contract_metadata(self.metadata)
        if list(expected_items) != self.metadata.get("expected_items"):
            raise CacheMetadataMismatch("incremental cache metadata does not match its expected items")
        if self.metadata.get("expected_items_fingerprint") != fingerprint(list(expected_items)):
            raise CacheMetadataMismatch("incremental cache has an invalid expected-item fingerprint")
        object.__setattr__(self, "metadata", _immutable_snapshot(self.metadata))
        object.__setattr__(self, "expected_items", expected_items)

    @property
    def fingerprint(self) -> str:
        return artifact_fingerprint(self.metadata)


def _validate_completed_contract_metadata(metadata: Mapping[str, Any]) -> None:
    if metadata.get("complete") is not True:
        raise CacheMetadataMismatch("completed artifact contract requires complete cache metadata")
    identity = metadata.get("identity_fingerprint")
    payload = metadata.get("payload_fingerprint")
    artifact = metadata.get("artifact_fingerprint")
    if not all(isinstance(value, str) and value for value in (identity, payload, artifact)):
        raise CacheMetadataMismatch("completed artifact contract requires identity and payload fingerprints")
    if artifact != fingerprint({"identity": identity, "payload": payload}):
        raise CacheMetadataMismatch("cache metadata has an invalid artifact fingerprint")


def artifact_fingerprint(metadata: Mapping[str, Any]) -> str:
    value = metadata.get("artifact_fingerprint")
    if not isinstance(value, str):
        raise CacheMetadataMismatch("cache metadata has no artifact fingerprint")
    return value


def certify_complete_artifact(
    path: Path,
    expected: Mapping[str, Any],
) -> CompleteArtifactContract:
    from .validation import validate_cache_metadata

    return CompleteArtifactContract(validate_cache_metadata(path, expected))


def certify_incremental_artifact(
    path: Path,
    expected: Mapping[str, Any],
    expected_items: Sequence[str],
) -> IncrementalArtifactContract:
    from .validation import validate_incremental_cache

    items = tuple(expected_items)
    return IncrementalArtifactContract(
        validate_incremental_cache(path, expected, items),
        items,
    )
