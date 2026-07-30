"""Named, immutable COLMAP runtime variants."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ColmapRuntimeVariant:
    name: str
    revision: str
    pycolmap_major_minor: tuple[int, int]
    canonical_environment: str


COLMAP_RUNTIME_VARIANTS = {
    "stock": ColmapRuntimeVariant(
        name="stock",
        revision="fa8e3b3ff591552855f8ad2806723c80f963f69c",
        pycolmap_major_minor=(4, 1),
        canonical_environment="VIDMAP_STOCK_COLMAP",
    ),
}


def colmap_runtime_variant(name: str) -> ColmapRuntimeVariant:
    if name not in COLMAP_RUNTIME_VARIANTS:
        choices = ", ".join(sorted(COLMAP_RUNTIME_VARIANTS))
        raise ValueError(f"Unknown COLMAP runtime {name!r}; expected one of: {choices}")
    return COLMAP_RUNTIME_VARIANTS[name]


def canonical_runtime_python(name: str, home: Path) -> Path:
    variant = colmap_runtime_variant(name)
    return home.resolve() / "venvs" / variant.canonical_environment / "bin/python"
