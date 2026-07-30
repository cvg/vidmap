"""Streaming iteration-window dataset and helpers for track frontend."""

from collections import OrderedDict

from torch.utils.data import Dataset

from vidmap.frontend.video_images import load_roma_resolution_pair


class LRUCache:
    """Simple LRU cache with a fixed max size."""

    def __init__(self, maxsize=2):
        self._cache = OrderedDict()
        self._maxsize = maxsize

    def __contains__(self, key):
        return key in self._cache

    def __getitem__(self, key):
        self._cache.move_to_end(key)
        return self._cache[key]

    def __setitem__(self, key, value):
        if key in self._cache:
            self._cache.move_to_end(key)
        self._cache[key] = value
        while len(self._cache) > self._maxsize:
            self._cache.popitem(last=False)

    def clear(self):
        self._cache.clear()


class IterationWindowDataset(Dataset):
    """Dataset where __getitem__(kf_id) returns all images needed for that iteration's window."""

    def __init__(
        self,
        highres_dataset,
        lowres_dataset,
        iteration_plan,
    ):
        """
        Args:
            highres_dataset: RomaVideoImageDataset for high-res images
            lowres_dataset: RomaVideoImageDataset for low-res images
            iteration_plan: Per-target ordered multiflow records
        """
        self.highres_dataset = highres_dataset
        self.lowres_dataset = lowres_dataset
        self.iteration_plan = iteration_plan

    def __len__(self):
        return len(self.iteration_plan)

    def __getitem__(self, kf_id):
        """Load all images needed for iteration kf_id.

        Returns dict with:
            - kf_id: the iteration index
            - images: dict mapping image_idx -> {"highres": tensor, "lowres": tensor}
        """
        target_idx = kf_id + 1
        records = self.iteration_plan[kf_id]
        needed_indices = [target_idx - record.hop for record in records]
        needed_indices.append(target_idx)

        images = {}
        for idx in needed_indices:
            highres, lowres = load_roma_resolution_pair(self.highres_dataset, self.lowres_dataset, idx)
            images[idx] = {
                "highres": highres["image"],
                "lowres": lowres["image"],
            }

        return {"kf_id": kf_id, "images": images}


def collate_iteration_window(batch):
    """Collate function - just pass through since batch_size=1."""
    return batch[0]
