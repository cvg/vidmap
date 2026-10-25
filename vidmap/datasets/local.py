from collections.abc import Sequence
from pathlib import Path

import pycolmap
from natsort import natsorted
from PIL import Image

from .base import PreparedSceneParser

IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png"})


def select_images(rgb_dir: Path, imnames: Sequence[str] | None) -> list[str]:
    if imnames is None:
        candidates = natsorted(
            entry.name for entry in rgb_dir.iterdir() if entry.is_file() and entry.suffix.lower() in IMAGE_SUFFIXES
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


class LocalImageParser(PreparedSceneParser):
    """Adapt an uncalibrated local image sequence to native COLMAP frames."""

    scene = "<custom>"

    def __init__(
        self,
        *,
        image_dir: str | Path,
        imnames: Sequence[str] | None = None,
    ) -> None:
        self.rgb_dir = Path(image_dir).expanduser()
        if not self.rgb_dir.is_dir():
            raise FileNotFoundError(f"image_dir is not a directory: {self.rgb_dir}")
        self.imnames = select_images(self.rgb_dir, imnames)

        dimensions = set()
        for name in self.imnames:
            with Image.open(self.rgb_dir / name) as pixels:
                dimensions.add(pixels.size)
        if len(dimensions) != 1:
            raise ValueError("All images in a monocular sequence must have the same shape")
        width, height = dimensions.pop()
        self.rec = pycolmap.Reconstruction()
        self.reconstruction_dir = None
        camera = pycolmap.Camera.create_from_model_name(1, "PINHOLE", 1000.0, width, height)
        camera.params = [1000.0, 1000.0, 1000.0, 1000.0]
        camera.has_prior_focal_length = True
        self.rec.add_camera_with_trivial_rig(camera)

        for image_id, image_name in enumerate(self.imnames, start=1):
            image = pycolmap.Image(
                name=image_name,
                camera_id=camera.camera_id,
                image_id=image_id,
            )
            self.rec.add_image_with_trivial_frame(image)
