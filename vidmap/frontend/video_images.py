import numpy as np
import torch

from vidmap.frontend.image_dataset import ImageDataset, resize_image
from vidmap.utils.io import read_image


class RomaVideoImageDataset(ImageDataset):
    def __init__(self, root, conf, paths=None):
        super().__init__(root, conf, paths=paths)

        # Pre-create mean/std once per dataset instance (so per worker)
        self._mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self._std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        self.normalize = True

    def __getitem__(self, idx):
        name = self.names[idx]
        image = read_image(self.root / name, self.conf.grayscale)
        return self.prepare_image(image, idx)

    def prepare_image(self, image, idx):
        """Apply this dataset's exact preprocessing to an already decoded image."""
        name = self.names[idx]
        image = image.astype(np.float32)

        size = image.shape[:2][::-1]  # (W, H)
        max_dim = max(size)

        if self.conf.resize_to_shape is not None:
            target_size = self.conf.resize_to_shape
            image = resize_image(image, target_size, self.conf.interpolation)
        else:
            should_resize = False
            if self.conf.resize_max and (self.conf.resize_force or max_dim > self.conf.resize_max):
                scale = self.conf.resize_max / max_dim
                should_resize = True
            elif self.conf.resize_min and max_dim < self.conf.resize_min:
                scale = self.conf.resize_min / max_dim
                should_resize = True

            if should_resize:
                size_new = tuple(int(round(x * scale)) for x in size)
                image = resize_image(image, size_new, self.conf.interpolation)

        if self.conf.grayscale:
            image = image[None]  # (1, H, W)
        else:
            image = image.transpose((2, 0, 1))  # (C, H, W)

        # Torch + in-place scaling
        image = torch.from_numpy(image).float()
        image.div_(255.0)

        if not self.conf.grayscale and image.shape[0] >= 3 and self.normalize:
            # Ensure mean/std on same dtype/device (CPU)
            mean = self._mean.to(dtype=image.dtype, device=image.device)
            std = self._std.to(dtype=image.dtype, device=image.device)
            image[:3].sub_(mean).div_(std)

        return {
            "image": image,
            "original_size": np.array(size),
            "name": name,
        }


def load_roma_resolution_pair(high_resolution_dataset, low_resolution_dataset, frame_index):
    """Decode one frame once and apply the established high/low RoMa preprocessing."""
    if high_resolution_dataset.root != low_resolution_dataset.root:
        raise ValueError("Paired RoMa datasets must share an image root")
    if high_resolution_dataset.names[frame_index] != low_resolution_dataset.names[frame_index]:
        raise ValueError("Paired RoMa datasets must use the same ordered image names")
    if high_resolution_dataset.conf.grayscale != low_resolution_dataset.conf.grayscale:
        return high_resolution_dataset[frame_index], low_resolution_dataset[frame_index]
    image = read_image(
        high_resolution_dataset.root / high_resolution_dataset.names[frame_index],
        high_resolution_dataset.conf.grayscale,
    )
    return high_resolution_dataset.prepare_image(image, frame_index), low_resolution_dataset.prepare_image(
        image, frame_index
    )
