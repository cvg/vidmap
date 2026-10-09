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
        revision="9e653da3baaaada926bda6c5286568413fb3e675",
        pycolmap_major_minor=(4, 3),
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
