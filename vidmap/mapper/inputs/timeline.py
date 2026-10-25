# Strict integer checks intentionally reject bool.
# ruff: noqa: E721
"""One explicit temporal order over native COLMAP frames."""

from collections.abc import Mapping
from dataclasses import dataclass


@dataclass
class MappingTimeline:
    """Validated frame order and its single image-index projection."""

    frame_ids: tuple[int, ...]
    timestamps_ns: tuple[int, ...] | None
    image_indices: Mapping[int, int]
    consecutive_pair_ids: tuple[int, ...]


def validate_frame_sequence(sequence):
    if not isinstance(sequence, dict) or set(sequence) != {"frame_ids", "timestamps_ns"}:
        raise ValueError("Frame sequence requires frame_ids and timestamps_ns")
    ids, times = sequence["frame_ids"], sequence["timestamps_ns"]
    if (
        not isinstance(ids, list)
        or not ids
        or any(type(fid) is not int or not 0 <= fid < 2**32 - 1 for fid in ids)
        or len(set(ids)) != len(ids)
    ):
        raise ValueError("Frame sequence requires unique native frame IDs")
    if times is not None and (
        not isinstance(times, list)
        or len(times) != len(ids)
        or any(type(time) is not int for time in times)
        or any(b <= a for a, b in zip(times, times[1:]))
    ):
        raise ValueError("Frame sequence timestamps must increase and cover every frame")


def mapping_timeline(sequence, reconstruction):
    """Validate temporal chains and project frame order to images for every input."""
    import pycolmap

    validate_frame_sequence(sequence)
    ids, times = sequence["frame_ids"], sequence["timestamps_ns"]
    if set(ids) != set(reconstruction.frames) or not reconstruction.is_valid():
        raise ValueError("Frame sequence must cover exactly the valid native reconstruction")
    previous = None
    consecutive, indices = [], {}
    for index, fid in enumerate(ids):
        frame = reconstruction.frame(fid)
        members = sorted(frame.image_ids, key=lambda data: data.sensor_id.id)
        sensors = set(reconstruction.rig(frame.rig_id).sensor_ids())
        if not members or {data.sensor_id for data in members} != sensors or len(members) != len(sensors):
            raise ValueError("Frame sequence requires complete camera frames")
        images = [reconstruction.image(data.id) for data in members]
        if any(image.frame_id != fid for image in images):
            raise ValueError("Frame sequence and native image membership disagree")
        if previous is not None:
            # A singleton stream may have independent calibration at each time.
            if len(images) != len(previous) or (
                len(images) > 1 and [image.camera_id for image in images] != [image.camera_id for image in previous]
            ):
                raise ValueError("Frame sequence must preserve synchronized camera streams")
            consecutive.extend(
                pycolmap.image_pair_to_pair_id(a.image_id, b.image_id) for a, b in zip(previous, images, strict=True)
            )
        for image in images:
            indices[image.image_id] = index
        previous = images
    return MappingTimeline(
        tuple(ids),
        None if times is None else tuple(times),
        indices,
        tuple(consecutive),
    )
