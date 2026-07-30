"""Image-name ordering shared by datasets, reconstruction, and evaluation."""

from pathlib import Path


def timestamp_from_image_name(name: str) -> float:
    """Parse a numeric timestamp from supported image-name conventions."""
    stem = Path(name).stem
    try:
        return float(stem)
    except ValueError:
        pass
    if "-" in stem:
        try:
            return float(stem.split("-")[0])
        except ValueError:
            pass
    return 0.0
