"""Package one run inside the browser reconstruction viewer."""

from __future__ import annotations

import base64
import gzip
import io
import sqlite3
import tempfile
from collections.abc import Mapping
from pathlib import Path

import numpy as np
from PIL import Image

from vidmap.mapper.inputs import LC_MASKS_NAME
from vidmap.utils.loop_closure_masks import read_loop_closure_masks

from . import scene
from .exporter import write_html

_PAIR_ID_BASE = 2147483647
_QUERY_CHUNK_SIZE = 500
_VALID_TWO_VIEW_CONFIGS = frozenset({2, 3, 4, 5, 6, 9})


def _embed_image_previews(
    images_dir: Path,
    image_names: tuple[str, ...],
    *,
    maximum_size: int = 640,
) -> tuple[str, ...]:
    """Encode bounded JPEG previews for an embedded keyframe timeline."""
    if maximum_size < 1:
        raise ValueError("preview maximum size must be at least one")
    resolved_images_dir = images_dir.resolve()
    previews = []
    for name in image_names:
        normalized_name = name.replace("\\", "/")
        if normalized_name.startswith("/") or ".." in normalized_name.split("/"):
            raise ValueError(f"Reconstruction image name is not a safe relative path: {name}")
        image_path = (resolved_images_dir / normalized_name).resolve()
        if not image_path.is_relative_to(resolved_images_dir):
            raise ValueError(f"Reconstruction image name escapes the image directory: {name}")
        if not image_path.is_file():
            raise FileNotFoundError(f"Timeline image does not exist: {image_path}")
        with Image.open(image_path) as source:
            preview = source.convert("RGB")
            preview.thumbnail((maximum_size, maximum_size), Image.Resampling.LANCZOS)
            encoded = io.BytesIO()
            preview.save(encoded, format="JPEG", quality=80, optimize=True)
        previews.append(f"data:image/jpeg;base64,{base64.b64encode(encoded.getvalue()).decode('ascii')}")
    return tuple(previews)


def _required_file(path: Path, label: str) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"Embedded viewer requires {label}: {path}")
    return path


def _encoded_bytes(data: bytes) -> dict[str, object]:
    compressed = gzip.compress(data, compresslevel=6, mtime=0)
    return {
        "encoding": "gzip-base64",
        "size": len(data),
        "base64": base64.b64encode(compressed).decode("ascii"),
    }


def _encoded_file(path: Path) -> dict[str, object]:
    return _encoded_bytes(path.read_bytes())


def _pair_id(first: int, second: int) -> int:
    return min(first, second) * _PAIR_ID_BASE + max(first, second)


def _rows_for_pair_ids(connection, table: str, columns: str, pair_ids: tuple[int, ...]):
    for start in range(0, len(pair_ids), _QUERY_CHUNK_SIZE):
        chunk = pair_ids[start : start + _QUERY_CHUNK_SIZE]
        placeholders = ",".join("?" for _ in chunk)
        yield from connection.execute(
            f"SELECT {columns} FROM {table} WHERE pair_id IN ({placeholders})",  # noqa: S608
            chunk,
        )


def _compact_loop_closure_database(
    database: Path,
    loop_closure_masks: Mapping[tuple[str, str], np.ndarray],
) -> bytes:
    """Retain only the SQLite rows queried by the embedded LC viewer."""
    uri = f"{database.resolve().as_uri()}?mode=ro&immutable=1"
    with sqlite3.connect(uri, uri=True) as source:
        image_ids = {
            str(name): int(image_id) for image_id, name in source.execute("SELECT image_id, name FROM images")
        }
        selected_names: set[str] = set()
        pair_ids = []
        for (first, second), mask in loop_closure_masks.items():
            if not mask.any():
                continue
            if first not in image_ids or second not in image_ids:
                continue
            selected_names.update((first, second))
            pair_ids.append(_pair_id(image_ids[first], image_ids[second]))
        ordered_pair_ids = tuple(sorted(set(pair_ids)))
        geometry_rows = tuple(
            row
            for row in _rows_for_pair_ids(
                source,
                "two_view_geometries",
                "pair_id, rows, config",
                ordered_pair_ids,
            )
            if int(row[1]) > 0 and row[2] is not None and int(row[2]) in _VALID_TWO_VIEW_CONFIGS
        )
        valid_pair_ids = tuple(sorted(int(row[0]) for row in geometry_rows))
        match_rows = tuple(_rows_for_pair_ids(source, "matches", "pair_id, rows, cols, data", valid_pair_ids))
        selected_images = tuple(sorted((image_ids[name], name) for name in selected_names))

    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as stream:
            temporary = Path(stream.name)
        with sqlite3.connect(temporary) as target:
            target.executescript(
                """
                PRAGMA journal_mode=OFF;
                PRAGMA synchronous=OFF;
                CREATE TABLE images(image_id INTEGER PRIMARY KEY, name TEXT NOT NULL);
                CREATE TABLE two_view_geometries(pair_id INTEGER PRIMARY KEY, rows INTEGER, config INTEGER);
                CREATE TABLE matches(pair_id INTEGER PRIMARY KEY, rows INTEGER, cols INTEGER, data BLOB);
                """
            )
            target.executemany("INSERT INTO images VALUES (?, ?)", selected_images)
            target.executemany("INSERT INTO two_view_geometries VALUES (?, ?, ?)", geometry_rows)
            target.executemany("INSERT INTO matches VALUES (?, ?, ?, ?)", match_rows)
        return temporary.read_bytes()
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _image_previews(run: Path, images_dir: Path | None) -> dict[str, dict[str, object]]:
    reconstruction = run / "rec"
    if images_dir is None:
        from vidmap.reconstruction import local_run_image_dir

        images_dir = local_run_image_dir(run)
        if images_dir is None:
            return {}
    images_dir = images_dir.expanduser().resolve(strict=True)

    import pycolmap

    model = pycolmap.Reconstruction(reconstruction)
    image_by_name = {str(image.name): image for image in model.images.values() if image.has_pose}
    names = tuple(sorted(image_by_name))
    previews = _embed_image_previews(images_dir, names)
    return {
        name: {
            "url": preview,
            "width": int(model.cameras[image_by_name[name].camera_id].width),
            "height": int(model.cameras[image_by_name[name].camera_id].height),
        }
        for name, preview in zip(names, previews, strict=True)
    }


def _embedded_run_payload(run_dir: str | Path, *, images_dir: str | Path | None = None) -> dict[str, object]:
    """Package one normalized run for the browser's ordinary load path."""
    run = Path(run_dir).expanduser().resolve(strict=True)
    reconstruction = run / "rec"
    mapper_inputs = run / "mapper_inputs"
    source_files = {
        f"rec/{name}": _required_file(reconstruction / name, f"rec/{name}")
        for name in ("cameras.bin", "images.bin", "points3D.bin")
    }
    files = {name: _encoded_file(path) for name, path in source_files.items()}
    database = mapper_inputs / "database_complete.db"
    loop_closure_masks_path = mapper_inputs / LC_MASKS_NAME
    if database.is_file() != loop_closure_masks_path.is_file():
        raise FileNotFoundError(
            "Embedded loop-closure inspection requires both " f"{database} and {loop_closure_masks_path}"
        )
    if database.is_file():
        loop_closure_masks = read_loop_closure_masks(loop_closure_masks_path)
        files["mapper_inputs/database_complete.db"] = _encoded_bytes(
            _compact_loop_closure_database(database, loop_closure_masks)
        )
        files[f"mapper_inputs/{LC_MASKS_NAME}"] = _encoded_file(loop_closure_masks_path)
    covariance = reconstruction / "visualization_cache" / "point_covariance_rank_v2.bin"
    if covariance.is_file():
        files["rec/visualization_cache/point_covariance_rank_v2.bin"] = _encoded_file(covariance)
    return {
        "files": files,
        "imagePreviews": _image_previews(
            run,
            None if images_dir is None else Path(images_dir),
        ),
    }


def write_embedded_viewer_html(
    run_dir: str | Path,
    output: str | Path,
    *,
    images_dir: str | Path | None = None,
) -> Path:
    """Write an embedded viewer that automatically loads one run."""
    payload = _embedded_run_payload(run_dir, images_dir=images_dir)
    return write_html(output, scene.render_viewer_html(embedded_run=payload))
