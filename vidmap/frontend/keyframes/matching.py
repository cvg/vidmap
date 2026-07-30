"""Ordered low-resolution matching for keyframe selection."""

import torch
from torch.utils.data import DataLoader

from vidmap.frontend.image_dataset import ImageDatasetOptions
from vidmap.frontend.video_images import RomaVideoImageDataset


class ImagePairDataset(torch.utils.data.Dataset):
    """Load an explicit ordered image-pair plan for keyframe matching."""

    def __init__(self, image_dataset, pair_indices):
        self.image_dataset = image_dataset
        self.pair_indices = pair_indices

    def __len__(self):
        return len(self.pair_indices)

    def __getitem__(self, index):
        first, second = self.pair_indices[index]
        image_a = self.image_dataset[first]
        image_b = self.image_dataset[second]
        return image_a["image"], image_b["image"], image_a["name"], image_b["name"]


def collate_image_pairs(batch):
    """Collate keyframe-selection pairs without moving tensors to CUDA."""
    image_a, image_b, names_a, names_b = zip(*batch)
    return {
        "im_A_batch": torch.stack(image_a),
        "im_B_batch": torch.stack(image_b),
        "names_A": list(names_a),
        "names_B": list(names_b),
    }


def _worker_init(_worker_id):
    import ctypes

    libc = ctypes.CDLL("libc.so.6")
    libc.mallopt(ctypes.c_int(-1), ctypes.c_int(0))
    libc.mallopt(ctypes.c_int(-3), ctypes.c_int(65536))


def build_pair_loader(scene_parser, sequence, lowres_options):
    image_dataset = RomaVideoImageDataset(
        scene_parser.rgb_dir,
        ImageDatasetOptions(
            resize_to_shape=tuple(lowres_options.resize_to_shape),
            interpolation=lowres_options.interpolation,
        ),
        sequence,
    )
    original_width, original_height = image_dataset[0]["original_size"]
    pair_indices = [(i, i + 1) for i in range(len(sequence) - 1)]
    pair_dataset = ImagePairDataset(image_dataset=image_dataset, pair_indices=pair_indices)
    loader = DataLoader(
        pair_dataset,
        batch_size=lowres_options.batch_size,
        num_workers=lowres_options.num_workers,
        shuffle=False,
        collate_fn=collate_image_pairs,
        worker_init_fn=_worker_init,
        pin_memory=False,
    )
    return loader, len(pair_indices), original_width, original_height


def match_lowres_batch(tracker_model, batch, original_width, original_height, first_batch):
    from vidmap.utils.profiling import record_timing, sync_time

    names_A = batch["names_A"]
    names_B = batch["names_B"]
    batch_indices = list(range(len(names_A)))
    batch_start = sync_time()
    im_A = batch["im_A_batch"][batch_indices].cuda(non_blocking=True)
    im_B = batch["im_B_batch"][batch_indices].cuda(non_blocking=True)
    output = tracker_model.match_lowres_batch(
        im_A,
        im_B,
        names_a=names_A,
        names_b=names_B,
        output_size=(original_width, original_height),
    )
    if first_batch:
        record_timing("keyframing_first_batch", sync_time() - batch_start, first=True)

    return output.matches, output.certainty
