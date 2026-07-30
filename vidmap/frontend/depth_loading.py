"""Worker-side image loading for depth inference."""

from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from vidmap.frontend.image_dataset import ImageDataset, ImageDatasetOptions

_NORMALIZE = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
_PATCH_SIZE = 14


def _nearest_patch_multiple(value: int) -> int:
    lower = (value // _PATCH_SIZE) * _PATCH_SIZE
    upper = lower + _PATCH_SIZE
    return upper if abs(upper - value) <= abs(value - lower) else lower


def _prepare_da3_image(image: np.ndarray, process_res: int) -> torch.Tensor:
    assert image.dtype == np.uint8
    pil_image = Image.fromarray(image).convert("RGB")
    width, height = pil_image.size
    longest = max(width, height)
    if longest != process_res:
        scale = process_res / float(longest)
        width = max(1, int(round(width * scale)))
        height = max(1, int(round(height * scale)))
        interpolation = cv2.INTER_CUBIC if scale > 1.0 else cv2.INTER_AREA
        pil_image = Image.fromarray(cv2.resize(np.asarray(pil_image), (width, height), interpolation=interpolation))

    resized_width = max(1, _nearest_patch_multiple(width))
    resized_height = max(1, _nearest_patch_multiple(height))
    if (resized_width, resized_height) != (width, height):
        upscale = resized_width > width or resized_height > height
        interpolation = cv2.INTER_CUBIC if upscale else cv2.INTER_AREA
        pil_image = Image.fromarray(
            cv2.resize(
                np.asarray(pil_image),
                (resized_width, resized_height),
                interpolation=interpolation,
            )
        )
    return _NORMALIZE(transforms.ToTensor()(pil_image))


def stack_window(images: list[torch.Tensor]) -> torch.Tensor:
    height = min(image.shape[1] for image in images)
    width = min(image.shape[2] for image in images)
    if any(image.shape[1:] != (height, width) for image in images):
        crop = transforms.CenterCrop((height, width))
        images = [crop(image) for image in images]
    return torch.stack(images)


class _Da3ImageDataset(ImageDataset):
    """Load one image and apply DA3 preprocessing in a DataLoader worker."""

    def __init__(self, root: Path, image_names, process_res: int):
        super().__init__(root, ImageDatasetOptions(), paths=image_names)
        self.process_res = process_res

    def __getitem__(self, index: int):
        name = self.names[index]
        with Image.open(self.root / name) as opened:
            image = np.asarray(opened.convert("RGB"))
        height, width = image.shape[:2]
        return {
            "image": _prepare_da3_image(image, self.process_res),
            "original_size": np.array([width, height]),
            "name": name,
        }


class WindowDataset(Dataset):
    """Load one explicit image window for each pending center image."""

    def __init__(self, image_dataset: ImageDataset, pending_names, window_size: int):
        self.image_dataset = image_dataset
        self.pending_names = tuple(pending_names)
        self.window_size = window_size
        self.image_to_index = {name: index for index, name in enumerate(self.image_dataset.names)}

    def __len__(self) -> int:
        return len(self.pending_names)

    def __getitem__(self, item: int):
        name = self.pending_names[item]
        index = self.image_to_index[name]
        half_window = self.window_size // 2
        start = max(0, index - half_window)
        end = min(len(self.image_dataset), index + half_window + 1)
        if end - start < self.window_size:
            if start == 0:
                end = min(len(self.image_dataset), self.window_size)
            else:
                start = max(0, len(self.image_dataset) - self.window_size)

        items = [self.image_dataset[image_index] for image_index in range(start, end)]
        center_index = index - start
        return {
            "name": name,
            "center_index": center_index,
            "images": stack_window([item["image"] for item in items]),
            "original_size": items[center_index]["original_size"],
        }


def _collate_window(batch):
    """Pass through one worker-prepared window."""
    return batch[0]


def create_da3_window_loader(
    rgb_dir: Path,
    image_names,
    pending_names,
    *,
    window_size: int,
    process_res: int,
    num_workers: int,
) -> DataLoader:
    """Create the ordered worker pool that prepares DA3 input windows."""
    image_dataset = _Da3ImageDataset(rgb_dir, image_names, process_res)
    dataset = WindowDataset(image_dataset, pending_names, window_size)
    return DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        prefetch_factor=None,
        collate_fn=_collate_window,
        pin_memory=True,
    )
