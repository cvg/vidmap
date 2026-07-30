from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hash_jsonable(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()


def _hash_h5_value(value: Any) -> bytes:
    array = np.asarray(value)
    if array.dtype.kind == "O":
        return _hash_jsonable(array.tolist())
    return np.ascontiguousarray(array).tobytes()


def canonical_h5_hash(
    path: Path,
    *,
    excluded_root_attrs: frozenset[str] = frozenset(),
    include_storage_layout: bool = True,
) -> tuple[str, list[dict[str, str]]]:
    import h5py

    digest = hashlib.sha256()
    entries: list[dict[str, str]] = []

    def update(token: str | bytes) -> None:
        if isinstance(token, str):
            token = token.encode()
        digest.update(len(token).to_bytes(8, "big"))
        digest.update(token)

    def hash_attrs(obj: Any, name: str) -> None:
        for key in sorted(obj.attrs.keys()):
            if name == "" and key in excluded_root_attrs:
                continue
            update(f"attr:{key}")
            update(_hash_h5_value(obj.attrs[key]))

    def visit(obj: Any, name: str) -> None:
        if isinstance(obj, h5py.Dataset):
            meta = {
                "kind": "dataset",
                "path": name,
                "shape": obj.shape,
                "dtype": obj.dtype.str,
            }
            if include_storage_layout:
                meta.update(
                    chunks=obj.chunks,
                    compression=obj.compression,
                    compression_opts=obj.compression_opts,
                    shuffle=obj.shuffle,
                    fletcher32=obj.fletcher32,
                )
            data = _hash_h5_value(obj[()])
            entry_hash = hashlib.sha256(_hash_jsonable(meta) + data).hexdigest()
            entries.append({"path": name, "kind": "dataset", "sha256": entry_hash})
            update(_hash_jsonable(meta))
            update(data)
            hash_attrs(obj, name)
        elif isinstance(obj, h5py.Group):
            update(_hash_jsonable({"kind": "group", "path": name}))
            entries.append(
                {
                    "path": name,
                    "kind": "group",
                    "sha256": hashlib.sha256(name.encode()).hexdigest(),
                }
            )
            hash_attrs(obj, name)
            for child in sorted(obj.keys()):
                child_name = f"{name}/{child}" if name else child
                visit(obj[child], child_name)

    with h5py.File(path, "r") as hfile:
        visit(hfile, "")
    return digest.hexdigest(), entries


def file_bytes_item(path: Path, relpath: Path, *, stage: str) -> dict[str, Any]:
    return {
        "stage": stage,
        "relpath": relpath.as_posix(),
        "kind": "file-bytes",
        "size": path.stat().st_size,
        "sha256": sha256_file(path),
    }
