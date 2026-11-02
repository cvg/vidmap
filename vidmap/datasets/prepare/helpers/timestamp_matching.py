import math
from bisect import bisect_left
from collections.abc import Iterable, Sequence


def _finite_timestamps(values: Iterable[float], label: str) -> tuple[float, ...]:
    try:
        timestamps = tuple(float(value) for value in values)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must contain numeric timestamps") from error
    if any(not math.isfinite(timestamp) for timestamp in timestamps):
        raise ValueError(f"{label} must contain only finite timestamps")
    return timestamps


def assign_images_to_timestamp_ids(images: Iterable[float], timestamps: Sequence[float]) -> list[int]:
    image_times = _finite_timestamps(images, "images")
    reference_times = _finite_timestamps(timestamps, "timestamps")
    if not image_times:
        return []
    if not reference_times:
        raise ValueError("timestamps must not be empty when images are provided")
    if any(after < before for before, after in zip(reference_times, reference_times[1:], strict=False)):
        raise ValueError("timestamps must be sorted in nondecreasing order")

    assigned = []
    for img_time in image_times:
        i = bisect_left(reference_times, img_time)
        if i == 0:
            closest = 0
        elif i == len(reference_times):
            closest = len(reference_times) - 1
        else:
            before = reference_times[i - 1]
            after = reference_times[i]
            closest = i - 1 if abs(img_time - before) <= abs(img_time - after) else i
        assigned.append(closest)
    return assigned
