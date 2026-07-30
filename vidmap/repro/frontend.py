from __future__ import annotations

import hashlib
import json
import os
import random
import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Any

import numpy as np
import torch

from vidmap.mapper.inputs import require_finalized_sqlite
from vidmap.repro.frontend_bootstrap import deterministic_env


def apply_deterministic_runtime(seed: int = 0) -> None:
    os.environ.update(deterministic_env(seed))
    random.seed(seed)

    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()


def _stable_array_summary(value: Any) -> dict[str, Any]:
    arr = np.asarray(value)
    if arr.dtype.kind == "O":
        payload = _json_bytes(arr.tolist())
        dtype = "object"
    else:
        arr = np.ascontiguousarray(arr)
        payload = arr.tobytes()
        dtype = arr.dtype.str
    return {
        "shape": list(arr.shape),
        "dtype": dtype,
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def _stable_pair(pair: Any) -> list[str]:
    return [str(pair[0]), str(pair[1])]


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n")


def write_pair_order_artifact(path: Path, pairs: Any, *, label: str) -> None:
    items = [_stable_pair(pair) for pair in pairs]
    digest = hashlib.sha256(_json_bytes(items)).hexdigest()
    _write_json(path, {"kind": label, "count": len(items), "sha256": digest, "pairs": items})


def write_sequence_artifact(path: Path, sequence: Any, *, label: str) -> None:
    items = [str(item) for item in sequence]
    digest = hashlib.sha256(_json_bytes(items)).hexdigest()
    _write_json(path, {"kind": label, "count": len(items), "sha256": digest, "items": items})


def tcorr_lc_summary(tcorr: dict | None, lc_masks: dict | None = None) -> dict[str, Any]:
    if tcorr is None:
        tcorr = {}
    if lc_masks is None:
        lc_masks = {}
    entries = []
    for pair in sorted(tcorr, key=lambda p: (str(p[0]), str(p[1]))):
        matches = tcorr[pair]
        entry = {
            "pair": _stable_pair(pair),
            "matches": _stable_array_summary(matches),
            "num_matches": int(len(matches)),
        }
        if pair in lc_masks:
            mask = lc_masks[pair]
            entry["lc_mask"] = _stable_array_summary(mask)
            entry["num_lc"] = int(mask.sum())
        entries.append(entry)
    digest = hashlib.sha256(_json_bytes(entries)).hexdigest()
    return {
        "kind": "tcorr_lc_canonical",
        "pair_count": len(entries),
        "match_count": sum(entry["num_matches"] for entry in entries),
        "sha256": digest,
        "entries": entries,
    }


def write_tcorr_artifact(path: Path, tcorr: dict | None, lc_masks: dict | None = None, *, label: str) -> None:
    summary = tcorr_lc_summary(tcorr, lc_masks)
    summary["kind"] = label
    _write_json(path, summary)


def _sqlite_tables(conn: sqlite3.Connection) -> set[str]:
    rows = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    return {str(row[0]) for row in rows}


def _sqlite_columns(conn: sqlite3.Connection, table: str) -> list[str]:
    return [str(row[1]) for row in conn.execute(f'PRAGMA table_info("{table}")').fetchall()]


def _sqlite_semantic_columns(conn: sqlite3.Connection, table: str, columns: list[str]) -> list[str]:
    optional_null_columns = {
        "two_view_geometries": {"camera1", "camera2"},
    }
    optional = optional_null_columns[table] if table in optional_null_columns else set()
    semantic = []
    for column in columns:
        if column in optional:
            populated = conn.execute(
                f'SELECT EXISTS(SELECT 1 FROM "{table}" WHERE "{column}" IS NOT NULL)'
            ).fetchone()[0]
            if not populated:
                continue
        semantic.append(column)
    return semantic


def _sqlite_order_columns(table: str, columns: list[str]) -> list[str]:
    preferred_by_table = {
        "cameras": ["camera_id"],
        "images": ["image_id"],
        "keypoints": ["image_id"],
        "matches": ["pair_id"],
        "two_view_geometries": ["pair_id"],
    }
    preferred = preferred_by_table[table] if table in preferred_by_table else []
    order = [col for col in preferred if col in columns]
    return order or columns


def _sqlite_value(value: Any) -> Any:
    if isinstance(value, bytes):
        return {
            "blob_len": len(value),
            "blob_sha256": hashlib.sha256(value).hexdigest(),
        }
    return value


def canonical_sqlite_summary(path: Path) -> dict[str, Any]:
    require_finalized_sqlite(path)
    tables_to_hash = [
        "cameras",
        "images",
        "keypoints",
        "matches",
        "two_view_geometries",
    ]
    summary: dict[str, Any] = {
        "kind": "sqlite-canonical",
        "path_name": path.name,
        "tables": {},
    }
    with closing(sqlite3.connect(f"file:{path}?mode=ro&immutable=1", uri=True)) as conn:
        existing = _sqlite_tables(conn)
        for table in tables_to_hash:
            if table not in existing:
                summary["tables"][table] = {
                    "present": False,
                    "row_count": 0,
                    "sha256": None,
                }
                continue
            columns = _sqlite_semantic_columns(conn, table, _sqlite_columns(conn, table))
            order_cols = _sqlite_order_columns(table, columns)
            order_by = ", ".join(f'"{col}"' for col in order_cols)
            selected = ", ".join(f'"{col}"' for col in columns)
            rows = []
            for row in conn.execute(f'SELECT {selected} FROM "{table}" ORDER BY {order_by}').fetchall():
                rows.append({col: _sqlite_value(value) for col, value in zip(columns, row)})
            table_payload = {"columns": columns, "rows": rows}
            summary["tables"][table] = {
                "present": True,
                "row_count": len(rows),
                "sha256": hashlib.sha256(_json_bytes(table_payload)).hexdigest(),
                "columns": columns,
            }
    summary["sha256"] = hashlib.sha256(_json_bytes(summary["tables"])).hexdigest()
    return summary


def write_sqlite_summary_artifact(path: Path, database_path: Path, *, label: str) -> None:
    summary = canonical_sqlite_summary(database_path)
    summary["kind"] = label
    _write_json(path, summary)
