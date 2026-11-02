"""Small filesystem helpers shared by dataset preparation scripts."""

from __future__ import annotations

import json
import shutil
import subprocess
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory

from vidmap.repro.hashing import sha256_file

IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png"})
_MODEL_FILES = ("cameras.bin", "images.bin", "points3D.bin")
Command = Callable[[Path], Sequence[str]]


def complete_colmap_model(path: Path) -> bool:
    return path.is_dir() and all((path / name).is_file() and (path / name).stat().st_size > 0 for name in _MODEL_FILES)


def run_tool(dataset: str, command: Sequence[str]) -> None:
    if shutil.which(command[0]) is None:
        raise RuntimeError(f"{dataset} preparation requires {command[0]!r} on PATH")
    subprocess.run(command, check=True)


def download(url: str, destination: Path, dataset: str) -> Path:
    if destination.is_file() and destination.stat().st_size:
        return destination
    if destination.exists():
        raise RuntimeError(f"{dataset} download destination is not a file: {destination}")

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.download")
    temporary.unlink(missing_ok=True)
    try:
        run_tool(dataset, ("wget", "-O", str(temporary), url))
        if not temporary.is_file() or temporary.stat().st_size == 0:
            raise RuntimeError(f"{dataset} download produced an empty file: {temporary}")
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def extract_archive(
    archive: Path,
    destination: Path,
    dataset: str,
    command: Command,
    child: str | None = None,
) -> Path:
    if destination.is_dir():
        return destination
    if not archive.is_file() or archive.stat().st_size == 0:
        raise RuntimeError(f"{dataset} archive is missing or empty: {archive}")

    temporary = destination.with_name(f".{destination.name}.extracting")
    shutil.rmtree(temporary, ignore_errors=True)
    temporary.mkdir(parents=True)
    try:
        run_tool(dataset, command(temporary))
        extracted = temporary / child if child else temporary
        if not extracted.is_dir():
            raise RuntimeError(f"{dataset} archive did not contain {child or 'a directory'}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        extracted.rename(destination)
    finally:
        shutil.rmtree(temporary, ignore_errors=True)
    return destination


@contextmanager
def scene_staging(destination: Path) -> Iterator[Path]:
    temporary = destination.with_name(f".{destination.name}.preparing")
    shutil.rmtree(temporary, ignore_errors=True)
    if destination.exists():
        raise RuntimeError(f"Prepared scene already exists: {destination}")
    temporary.mkdir(parents=True)
    try:
        yield temporary
    finally:
        shutil.rmtree(temporary, ignore_errors=True)


def _prepared_image_hashes(root: Path, names: Sequence[str]) -> dict[str, str]:
    paths = {name: root / "images" / name for name in names}
    if any(not path.is_file() for path in paths.values()):
        raise ValueError("Missing prepared image")
    return {name: sha256_file(path) for name, path in paths.items()}


def prepare_image_directory(
    output: Path, identity: dict, names: Sequence[str], write_images: Callable[[Path], None]
) -> dict[str, str]:
    """Write or verify a prepared image directory and return its current hashes."""
    identity = json.loads(json.dumps(identity, sort_keys=True))
    if output.exists() or output.is_symlink():
        manifest = json.loads((output / "manifest.json").read_text())
        if manifest["identity"] != identity:
            raise ValueError("Prepared output identity mismatch; refusing overwrite")
        hashes = _prepared_image_hashes(output, names)
        if manifest != dict(identity=identity, images=hashes):
            raise ValueError("Prepared output manifest mismatch; images or fields changed")
        return hashes
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=f".{output.name}-", dir=output.parent) as temporary:
        staging = Path(temporary) / "capture"
        staging.mkdir()
        write_images(staging / "images")
        hashes = _prepared_image_hashes(staging, names)
        (staging / "manifest.json").write_text(
            json.dumps(dict(identity=identity, images=hashes), sort_keys=True, indent=2) + "\n"
        )
        if output.exists() or output.is_symlink():
            raise FileExistsError(f"Output appeared while preparing: {output}")
        staging.rename(output)
    return hashes
