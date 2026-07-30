"""Shared, conservative dataset download helpers."""

from __future__ import annotations

import hashlib
import zipfile
from collections.abc import Callable, Iterable
from pathlib import Path

import requests

DOWNLOAD_USER_AGENT = "VidMap/0.1 (+https://github.com/cvg/vidmap)"


def checksum(path: Path, algorithm: str) -> str:
    """Return a streaming checksum without loading a large archive in memory."""
    digest = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_download(
    path: Path,
    *,
    expected_size: int | None,
    expected_digest: str | None,
    algorithm: str,
) -> None:
    if expected_size is not None and path.stat().st_size != expected_size:
        raise ValueError(f"Size mismatch for {path}: expected {expected_size}, got {path.stat().st_size}")
    if expected_digest is not None:
        actual = checksum(path, algorithm)
        if actual.lower() != expected_digest.lower():
            raise ValueError(f"{algorithm.upper()} mismatch for {path}: expected {expected_digest}, got {actual}")


def resumable_http_download(
    url: str,
    destination: Path,
    *,
    expected_size: int | None = None,
    expected_digest: str | None = None,
    digest_algorithm: str = "sha256",
    session: requests.Session | None = None,
    response_validator: Callable[[requests.Response], None] | None = None,
) -> Path:
    """Download to ``.part`` with Range resume, atomic promotion, and validation."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        _validate_download(
            destination,
            expected_size=expected_size,
            expected_digest=expected_digest,
            algorithm=digest_algorithm,
        )
        return destination

    partial = destination.with_suffix(destination.suffix + ".part")
    if partial.exists() and (expected_size is not None or expected_digest is not None):
        try:
            _validate_download(
                partial,
                expected_size=expected_size,
                expected_digest=expected_digest,
                algorithm=digest_algorithm,
            )
        except ValueError:
            pass
        else:
            partial.replace(destination)
            return destination
    offset = partial.stat().st_size if partial.exists() else 0
    if expected_size is not None and offset >= expected_size:
        offset = 0
    headers = {"User-Agent": DOWNLOAD_USER_AGENT}
    if offset:
        headers["Range"] = f"bytes={offset}-"
    client = requests.Session() if session is None else session
    with client.get(url, headers=headers, stream=True, timeout=(30, 300)) as response:
        response.raise_for_status()
        if response_validator is not None:
            response_validator(response)
        if offset and response.status_code != 206:
            offset = 0
        mode = "ab" if offset else "wb"
        with partial.open(mode) as handle:
            for block in response.iter_content(chunk_size=8 * 1024 * 1024):
                if block:
                    handle.write(block)
    _validate_download(
        partial,
        expected_size=expected_size,
        expected_digest=expected_digest,
        algorithm=digest_algorithm,
    )
    partial.replace(destination)
    return destination


def safe_extract_zip(archive: Path, destination: Path, *, members: Iterable[str] | None = None) -> None:
    """Idempotently extract selected ZIP members while rejecting path traversal."""
    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()
    with zipfile.ZipFile(archive) as bundle:
        selected = bundle.namelist() if members is None else members
        for member in selected:
            target = (destination / member).resolve()
            if root not in target.parents and target != root:
                raise ValueError(f"Unsafe ZIP member {member!r}")
            if not target.exists():
                bundle.extract(member, destination)
