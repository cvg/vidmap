"""Access read-only resources bundled with the installed VidMap package."""

from importlib.resources import files
from pathlib import Path


def dataset_asset(*parts: str) -> Path:
    """Return one filesystem resource owned by the installed package."""
    resource = files("vidmap.datasets.assets").joinpath(*parts)
    path = Path(str(resource))
    if not path.is_file():
        raise FileNotFoundError(f"Bundled dataset asset does not exist: {'/'.join(parts)}")
    return path
