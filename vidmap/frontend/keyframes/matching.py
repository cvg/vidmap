"""Ordered low-resolution matching for keyframe selection."""

from contextlib import contextmanager

import torch
from torch.utils.data import DataLoader

from vidmap.frontend.image_dataset import ImageDatasetOptions
from vidmap.frontend.video_images import RomaVideoImageDataset


class ImagePairDataset(torch.utils.data.Dataset):
    """Load an explicit ordered image-pair plan for keyframe matching."""

    def __init__(self, image_dataset, pair_indices):
        self.image_dataset = image_dataset
        self.pair_indices = pair_indices
        self.previous = None

    def __len__(self):
        return len(self.pair_indices)

    def __getitem__(self, index):
        first, second = self.pair_indices[index]
        image_a = (
            self.previous[1] if self.previous is not None and self.previous[0] == first else self.image_dataset[first]
        )
        image_b = self.image_dataset[second]
        self.previous = (second, image_b)
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


@contextmanager
def pipelined_matches(tracker_model, loader, original_width, original_height, *, batch_size):
    """Overlap one ordered inference batch with CPU selection, draining on exit."""
    from vidmap.utils.profiling import record_timing, sync_time

    source = iter(loader)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    first_batch = True

    def launch(batch):
        nonlocal first_batch
        with torch.cuda.stream(stream):
            batch_start = sync_time()
            output = tracker_model.match_lowres_batch(
                batch["im_A_batch"],
                batch["im_B_batch"],
                names_a=batch["names_A"],
                names_b=batch["names_B"],
                output_size=(original_width, original_height),
                batch_size=batch_size,
            )
            if first_batch:
                record_timing("keyframing_first_batch", sync_time() - batch_start, first=True)
            ready = torch.cuda.Event()
            ready.record()
        first_batch = False
        return (output.matches, output.certainty), ready

    def iterate():
        first = next(source, None)
        pending = None if first is None else launch(first)
        while pending is not None:
            (matches, certainties), ready = pending
            consumer = torch.cuda.current_stream()
            consumer.wait_event(ready)
            matches.record_stream(consumer)
            certainties.record_stream(consumer)
            following = next(source, None)
            pending = None if following is None else launch(following)
            yield matches, certainties

    batches = iterate()
    try:
        yield batches
    finally:
        stream.synchronize()
        batches.close()
