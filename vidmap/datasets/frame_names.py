"""Image-name ordering shared by datasets, reconstruction, and evaluation."""

from decimal import Decimal
from pathlib import Path


def _timestamp_text(name: str) -> str | None:
    stem = Path(name).stem
    try:
        float(stem)
        return stem
    except ValueError:
        pass
    if "-" in stem:
        try:
            prefix = stem.split("-")[0]
            float(prefix)
            return prefix
        except ValueError:
            pass
    return None


def timestamp_from_image_name(name: str) -> float:
    """Parse a numeric timestamp from supported image-name conventions."""
    text = _timestamp_text(name)
    return 0.0 if text is None else float(text)


def timestamp_ns_from_image_name(name: str) -> int:
    """Resolve a timestamp in seconds without losing nanosecond precision."""
    text = _timestamp_text(name)
    if text is None or not Decimal(text).is_finite():
        raise ValueError(f"Cannot resolve timestamp in {name!r}; set rig.frames to a frames.csv")
    timestamp = Decimal(text) * 1_000_000_000
    if timestamp != timestamp.to_integral_value():
        raise ValueError(f"Timestamp in {name!r} is finer than nanoseconds; set rig.frames to a frames.csv")
    return int(timestamp)
