# Strict integer checks intentionally reject bool.
# ruff: noqa: E721
"""Lightweight validation of panoramic viewer media, independent of scene geometry."""

import math


def validate_panorama_metadata(metadata):
    """Validate the optional viewer descriptor, never a second scene graph."""
    if not isinstance(metadata, dict) or set(metadata) != {"panorama_size", "rig_from_panorama_rotation", "frames"}:
        raise ValueError("Panorama media requires dimensions, alignment and source frames")
    size = metadata["panorama_size"]
    if (
        not isinstance(size, list)
        or len(size) != 2
        or any(type(v) is not int or v <= 0 for v in size)
        or size[0] != 2 * size[1]
    ):
        raise ValueError("Panorama media requires positive 2:1 dimensions")
    rotation = metadata["rig_from_panorama_rotation"]
    if (
        not isinstance(rotation, list)
        or len(rotation) != 3
        or any(
            not isinstance(row, list)
            or len(row) != 3
            or any(type(v) not in (float, int) or not math.isfinite(v) for v in row)
            for row in rotation
        )
    ):
        raise ValueError("Panorama alignment must be a finite 3x3 rotation")
    for i in range(3):
        for j in range(3):
            if abs(sum(rotation[i][k] * rotation[j][k] for k in range(3)) - int(i == j)) > 1e-6:
                raise ValueError("Panorama alignment must be orthonormal")
    a, b, c = rotation
    determinant = (
        a[0] * (b[1] * c[2] - b[2] * c[1]) - a[1] * (b[0] * c[2] - b[2] * c[0]) + a[2] * (b[0] * c[1] - b[1] * c[0])
    )
    if abs(determinant - 1) > 1e-6:
        raise ValueError("Panorama alignment must be a proper rotation")
    frames = metadata["frames"]
    if not isinstance(frames, list) or not frames:
        raise ValueError("Panorama media requires source frames")
    frame_ids, names = set(), set()
    for frame in frames:
        if not isinstance(frame, dict) or set(frame) != {"frame_id", "image_name", "sha256"}:
            raise ValueError("Panorama source frame requires a native ID, image name and digest")
        fid, name = frame["frame_id"], frame["image_name"]
        if type(fid) is not int or not 0 <= fid < 2**32 - 1:
            raise ValueError("Panorama frame ID must be a native integer")
        if (
            not isinstance(name, str)
            or not name
            or "\\" in name
            or "\x00" in name
            or any(part in ("", ".", "..") for part in name.split("/"))
        ):
            raise ValueError("Panorama source names must be safe relative POSIX paths")
        if fid in frame_ids or name in names:
            raise ValueError("Panorama source frames and image names must be unique")
        digest = frame["sha256"]
        if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError("Panorama source image requires a SHA256 digest")
        frame_ids.add(fid)
        names.add(name)
