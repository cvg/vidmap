from collections.abc import Mapping, Sequence
from numbers import Real
from pathlib import Path

import numpy as np
import pycolmap
import yaml
from natsort import natsorted
from PIL import Image

from .base import PreparedSceneParser

IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png"})


def _select_images(rgb_dir: Path, imnames: Sequence[str] | None) -> list[str]:
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


def _add_cameras(
    rec: pycolmap.Reconstruction,
    intrinsics: Mapping,
    names: tuple[str, ...],
    root: Path,
):
    input_names = set(names)
    source_keys: dict[int, object] = {}
    assigned: dict[str, tuple[object, int]] = {}

    for source_id, camera_data in intrinsics.items():
        if isinstance(source_id, bool) or not isinstance(source_id, (int, str)):
            raise ValueError(f"Camera ID must be an integer or decimal string, got {source_id!r}")
        digits = source_id[1:] if isinstance(source_id, str) and source_id[:1] in {"+", "-"} else source_id
        if isinstance(digits, str) and not digits.isdecimal():
            raise ValueError(f"Camera ID must be an integer or decimal string, got {source_id!r}")
        camera_id = int(source_id)
        if not 0 <= camera_id < pycolmap.INVALID_CAMERA_ID:
            raise ValueError(f"Camera ID must be in [0, {pycolmap.INVALID_CAMERA_ID}), got {source_id!r}")
        if camera_id in source_keys:
            raise ValueError(f"Camera IDs {source_keys[camera_id]!r} and {source_id!r} both normalize to {camera_id}")
        if not isinstance(camera_data, Mapping) or "params" not in camera_data or "images" not in camera_data:
            raise ValueError(f"Camera {source_id!r} must define params and images")

        raw_params = camera_data["params"]
        if (
            isinstance(raw_params, (str, bytes))
            or not isinstance(raw_params, Sequence)
            or any(isinstance(item, bool) or not isinstance(item, Real) for item in raw_params)
        ):
            raise ValueError(f"Camera {camera_id} PINHOLE params must contain four finite numbers")
        params = tuple(float(item) for item in raw_params)
        if len(params) != 4 or not np.isfinite(params).all() or params[0] <= 0 or params[1] <= 0:
            raise ValueError(
                f"Camera {camera_id} PINHOLE params must be [fx, fy, cx, cy] with positive finite focal lengths"
            )

        selected = camera_data["images"]
        if selected == "all":
            camera_images = names
        elif isinstance(selected, list):
            camera_images = tuple(selected)
        else:
            raise ValueError(f"Camera {source_id!r} images must be 'all' or a list")
        if not camera_images:
            raise ValueError(f"Camera {source_id!r} must select at least one image")
        for image_name in camera_images:
            if not isinstance(image_name, str) or image_name not in input_names:
                raise ValueError(f"Camera {source_id!r} selects unknown input image {image_name!r}")
            if image_name in assigned:
                raise ValueError(
                    f"Input image {image_name!r} is assigned to cameras {assigned[image_name][0]!r} and {source_id!r}"
                )
            assigned[image_name] = (source_id, camera_id)

        dimensions = []
        for image_name in camera_images:
            path = root / image_name
            try:
                with Image.open(path) as image:
                    dimensions.append(image.size)
            except OSError as error:
                raise ValueError(f"Could not read image dimensions from {path}: {error}") from error
        if len(set(dimensions)) != 1:
            raise ValueError(f"All images assigned to camera {camera_id} must have the same shape")
        width, height = dimensions[0]
        if width <= 0 or height <= 0:
            raise ValueError(f"Camera {camera_id} images must have positive dimensions")
        camera = pycolmap.Camera.create_from_model_name(camera_id, "PINHOLE", params[0], width, height)
        camera.params = params
        camera.has_prior_focal_length = True
        rec.add_camera_with_trivial_rig(camera)
        source_keys[camera_id] = source_id

    if missing := [image_name for image_name in names if image_name not in assigned]:
        raise ValueError(f"Intrinsics do not assign every input image; missing: {missing}")
    return {image_name: source[1] for image_name, source in assigned.items()}


class LocalImageParser(PreparedSceneParser):
    scene = "<custom>"

    def __init__(
        self,
        *,
        image_dir: str | Path,
        imnames: Sequence[str] | None = None,
        intrinsics_path: str | Path | None = None,
        use_geocalib: bool = False,
    ) -> None:
        self.rgb_dir = Path(image_dir).expanduser()
        if not self.rgb_dir.is_dir():
            raise FileNotFoundError(f"image_dir is not a directory: {self.rgb_dir}")
        self.imnames = _select_images(self.rgb_dir, imnames)

        if use_geocalib:
            if intrinsics_path is not None:
                raise ValueError("intrinsics_path cannot be combined with use_geocalib=True")
            intrinsics: Mapping = {1: {"params": [1000.0, 1000.0, 1000.0, 1000.0], "images": "all"}}
        else:
            if intrinsics_path is None:
                raise ValueError("intrinsics_path is required when use_geocalib=False")
            path = Path(intrinsics_path).expanduser()
            try:
                with path.open(encoding="utf-8") as file:
                    intrinsics = yaml.safe_load(file)
            except (UnicodeDecodeError, yaml.YAMLError) as error:
                raise ValueError(f"Invalid intrinsics file {path}: {error}") from error
            if not isinstance(intrinsics, Mapping) or not intrinsics:
                raise ValueError(f"Intrinsics file {path} must define at least one camera mapping")

        self.rec = pycolmap.Reconstruction()
        self.reconstruction_dir = None
        image_camera_ids = _add_cameras(self.rec, intrinsics, tuple(self.imnames), self.rgb_dir)

        for image_id, image_name in enumerate(self.imnames, start=1):
            image = pycolmap.Image(
                name=image_name,
                camera_id=image_camera_ids[image_name],
                image_id=image_id,
            )
            self.rec.add_image_with_trivial_frame(image)
