from collections.abc import Sequence
from pathlib import Path

from natsort import natsorted

IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png"})


def select_images(rgb_dir: Path, imnames: Sequence[str] | None, *, camera_dir: str | None = None) -> list[str]:
    directory = rgb_dir if camera_dir is None else rgb_dir / camera_dir
    if imnames is None:
        candidates = natsorted(
            entry.relative_to(rgb_dir).as_posix()
            for entry in directory.iterdir()
            if entry.is_file() and entry.suffix.lower() in IMAGE_SUFFIXES
        )
    else:
        if isinstance(imnames, (str, bytes)):
            raise TypeError("imnames must be a sequence of image names, not a string")
        candidates = list(imnames)

    validated = []
    for name in candidates:
        if not isinstance(name, str) or not name:
            raise ValueError(f"Image names must be non-empty strings, got {name!r}")
        parts = name.split("/")
        if camera_dir is not None and parts[0] != camera_dir:
            raise ValueError(f"Images must belong to {camera_dir}/: {name!r}")
        if "\\" in name or Path(name).is_absolute() or any(part in {"", ".", ".."} for part in parts):
            raise ValueError(f"Image names must be safe paths relative to {rgb_dir}, got {name!r}")
        if Path(name).suffix.lower() not in IMAGE_SUFFIXES:
            raise ValueError(f"Unsupported image suffix for {name!r}; expected one of {sorted(IMAGE_SUFFIXES)}")
        path = rgb_dir.joinpath(*parts)
        if not path.is_file():
            raise ValueError(f"Image {name!r} is not a file under {rgb_dir}")
        validated.append(name)
    if len(set(validated)) != len(validated):
        raise ValueError("Input image names must be unique")
    if not validated:
        raise ValueError(f"No supported images were selected from {rgb_dir}")
    return validated
