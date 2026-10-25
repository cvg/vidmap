# Strict integer checks intentionally reject bool.
# ruff: noqa: E721
import collections.abc as collections
import glob
from dataclasses import dataclass
from dataclasses import field as dc_field
from functools import cached_property
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import PIL.Image
import torch
from pydantic import ConfigDict
from torch.nn import functional as F

from vidmap.configuration.validators import dataclass as pydantic_dataclass
from vidmap.frontend.options.rig import RigOptions
from vidmap.frontend.rig import camera_streams, resolve_rig_options
from vidmap.utils.io import read_image
from vidmap.utils.parsers import parse_image_lists


@dataclass(frozen=True)
class FrameSequence:
    """Camera-local execution streams, with the admission camera first."""

    streams: tuple[tuple[str, ...], ...]
    same_frame_stream_pairs: tuple[tuple[int, int, bool], ...] = ()
    frame_ids: tuple[int, ...] = dc_field(kw_only=True)
    stream_camera_ids: tuple[int, ...] = dc_field(kw_only=True)

    @cached_property
    def names(self) -> tuple[str, ...]:
        return tuple(name for stream in self.streams for name in stream)

    def select(self, indices):
        """Apply admission-camera indices to every synchronized stream."""
        indices = tuple(indices)
        return tuple(tuple(stream[i] for i in indices) for stream in self.streams)

    def indices(self, frame_ids):
        """Resolve an ordered native frame selection against this timeline."""
        frame_ids = tuple(frame_ids)
        if any(type(frame_id) is not int for frame_id in frame_ids):
            raise ValueError("Selected native frame IDs must be integers")
        positions = {frame_id: i for i, frame_id in enumerate(self.frame_ids)}
        if any(frame_id not in positions for frame_id in frame_ids):
            raise ValueError("Selected frame is absent from the capture timeline")
        indices = tuple(positions[frame_id] for frame_id in frame_ids)
        if any(b <= a for a, b in zip(indices, indices[1:])):
            raise ValueError("Selected frames must be unique and follow the capture timeline")
        return indices

    @classmethod
    def from_scene(cls, scene_parser, *, rig_options: RigOptions | None = None):
        options = RigOptions() if rig_options is None else rig_options
        reference, policy = resolve_rig_options(scene_parser, options)
        streams = camera_streams(scene_parser, reference_camera_id=reference)
        same_frame_stream_pairs = ()
        if policy:
            positions = {scene_parser.image(stream[0]).camera_id: index for index, stream in enumerate(streams)}
            same_frame_stream_pairs = tuple(
                (positions[first], positions[second], sequential) for first, second, sequential in policy
            )
        return cls(
            streams,
            same_frame_stream_pairs,
            frame_ids=tuple(scene_parser.image(name).frame_id for name in streams[0]),
            stream_camera_ids=tuple(scene_parser.image(stream[0]).camera_id for stream in streams),
        )


def get_image_size(scene_parser, image_name):
    """Read original image dimensions without decoding full image data."""
    with PIL.Image.open(scene_parser.rgb_dir / image_name) as image:
        return image.size


def resize_image(image, size, interp):
    if interp.startswith("torch_"):
        # 1. Parse mode (e.g., "torch_bicubic")
        mode = interp[len("torch_") :]

        # 2. Prepare Tensor: (H, W, C) -> (1, C, H, W)
        # Ensure we are working with float32 to avoid quantization errors
        if len(image.shape) == 2:  # Grayscale
            t_img = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).float()
        else:  # RGB
            t_img = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float()

        # 3. Handle Dimensions
        # 'size' comes in as (W, H), but F.interpolate expects (H, W)
        target_size = (size[1], size[0])

        # 4. Interpolate matches BatchResize defaults (align_corners=False for bicubic)
        align = False if mode in ["bicubic", "bilinear"] else None

        resized_t = F.interpolate(t_img, size=target_size, mode=mode, align_corners=align)

        # 5. Back to Numpy: (1, C, H, W) -> (H, W, C)
        resized = resized_t.squeeze(0).permute(1, 2, 0).numpy()

        # Handle grayscale squeeze if necessary
        if len(image.shape) == 2:
            resized = resized.squeeze(-1)
        return resized
    elif interp.startswith("cv2_"):
        interp = getattr(cv2, "INTER_" + interp[len("cv2_") :].upper())
        h, w = image.shape[:2]
        if interp == cv2.INTER_AREA and (w < size[0] or h < size[1]):
            interp = cv2.INTER_LINEAR
        resized = cv2.resize(image, size, interpolation=interp)
    elif interp.startswith("pil_"):
        interp = getattr(PIL.Image, interp[len("pil_") :].upper())
        resized = PIL.Image.fromarray(image.astype(np.uint8))
        resized = resized.resize(size, resample=interp)
        resized = np.asarray(resized, dtype=image.dtype)
    else:
        raise ValueError(f"Unknown interpolation {interp}.")
    return resized


@pydantic_dataclass(frozen=True, config=ConfigDict(extra="forbid", arbitrary_types_allowed=True))
class ImageDatasetOptions:
    """Typed config for ``ImageDataset`` and its subclasses."""

    globs: list[str] = dc_field(default_factory=lambda: ["*.jpg", "*.png", "*.jpeg", "*.JPG", "*.PNG"])
    grayscale: bool = False
    resize_max: Optional[int] = None
    resize_min: Optional[int] = None
    resize_force: bool = False
    interpolation: str = "cv2_area"  # pil_linear is more accurate but slower
    resize_to_shape: Optional[tuple] = None


class ImageDataset(torch.utils.data.Dataset):
    _options_cls = ImageDatasetOptions

    def __init__(self, root, conf: ImageDatasetOptions, paths=None):
        assert isinstance(conf, ImageDatasetOptions), f"Expected ImageDatasetOptions, got {type(conf).__name__}"
        self.conf = conf
        self.root = root

        if paths is None:
            paths = []
            for g in self.conf.globs:
                paths += glob.glob((Path(root) / "**" / g).as_posix(), recursive=True)
            if len(paths) == 0:
                raise ValueError(f"Could not find any image in root: {root}.")
            paths = sorted(set(paths))
            self.names = [Path(p).relative_to(root).as_posix() for p in paths]
        else:
            if isinstance(paths, (Path, str)):
                self.names = parse_image_lists(paths)
            elif isinstance(paths, collections.Iterable):
                self.names = [p.as_posix() if isinstance(p, Path) else p for p in paths]
            else:
                raise ValueError(f"Unknown format for path argument {paths}.")

            for name in self.names:
                if not (root / name).exists():
                    raise ValueError(f"Image {name} does not exists in root: {root}.")

    def __getitem__(self, idx):
        name = self.names[idx]
        image = read_image(self.root / name, self.conf.grayscale)
        image = image.astype(np.float32)
        size = image.shape[:2][::-1]
        max_dim = max(size)

        # Resize logic: scale down if too large, scale up if too small, leave as is if in between
        should_resize = False
        if self.conf.resize_max and (self.conf.resize_force or max_dim > self.conf.resize_max):
            # Scale down if larger than resize_max
            scale = self.conf.resize_max / max_dim
            should_resize = True
        elif self.conf.resize_min and max_dim < self.conf.resize_min:
            # Scale up if smaller than resize_min
            scale = self.conf.resize_min / max_dim
            should_resize = True
        if should_resize:
            size_new = tuple(int(round(x * scale)) for x in size)
            image = resize_image(image, size_new, self.conf.interpolation)

        image = image[None] if self.conf.grayscale else image.transpose((2, 0, 1))  # HxWxC to CxHxW
        image = image / 255.0

        data = {
            "image": image,
            "original_size": np.array(size),
        }
        return data

    def __len__(self):
        return len(self.names)
