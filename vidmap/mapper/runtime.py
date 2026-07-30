"""Lazy imports for mapping packages."""

from __future__ import annotations

import importlib
import re
import warnings
from functools import cache
from types import ModuleType

TESTED_PYCOLMAP_VERSION = (4, 1)


def _colmap_major_minor(version: str) -> tuple[int, int] | None:
    match = re.match(r"^(\d+)\.(\d+)(?:\.|$)", version)
    return None if match is None else (int(match.group(1)), int(match.group(2)))


def load_pycolmap_runtime() -> ModuleType:
    """Return PyCOLMAP, warning when its version differs from the tested release."""
    pycolmap = importlib.import_module("pycolmap")
    version = str(pycolmap.__version__)
    if "+" in version or _colmap_major_minor(version) != TESTED_PYCOLMAP_VERSION:
        expected = ".".join(str(value) for value in TESTED_PYCOLMAP_VERSION)
        warnings.warn(
            f"VidMap was tested with pycolmap {expected}; continuing with {version!r}.",
            RuntimeWarning,
            stacklevel=2,
        )
    return pycolmap


@cache
def load_mapping_runtime() -> ModuleType:
    """Return VidMap's native extension after loading PyCOLMAP."""

    load_pycolmap_runtime()
    return importlib.import_module("vidmap_native._core")
