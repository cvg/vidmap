"""Stitch Hilti fisheye pairs into equirectangular images using the official 10 m sphere."""

from __future__ import annotations

import json
import sqlite3
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from pathlib import Path
from tempfile import TemporaryDirectory

import cv2
import numpy as np
import yaml

TOPICS = tuple(f"/cam{i}/image_raw/compressed" for i in (0, 1))
_WRITE_WORKERS = 2


def _write_jpeg(path, image):
    if not cv2.imwrite(str(path), image, [cv2.IMWRITE_JPEG_QUALITY, 95]):
        raise OSError(path)


def compressed_stream(connection, topic):
    """Yield exact header nanoseconds and original compressed image bytes."""
    from rosbags.typesys import Stores, get_typestore

    rows = connection.execute("SELECT id, type, serialization_format FROM topics WHERE name = ?", (topic,)).fetchall()
    if len(rows) != 1 or rows[0][1:] != ("sensor_msgs/msg/CompressedImage", "cdr"):
        raise ValueError(f"Expected one CDR CompressedImage topic: {topic}")
    typestore = get_typestore(Stores.ROS2_HUMBLE)
    previous = -1
    cursor = connection.execute("SELECT data FROM messages WHERE topic_id = ? ORDER BY timestamp, id", (rows[0][0],))
    for (data,) in cursor:
        message = typestore.deserialize_cdr(data, "sensor_msgs/msg/CompressedImage")
        stamp = message.header.stamp
        if not 0 <= stamp.nanosec < 1_000_000_000:
            raise ValueError("Invalid image header nanoseconds")
        timestamp = int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)
        if timestamp <= previous:
            raise ValueError(f"Duplicate or non-increasing header timestamp in {topic}")
        previous = timestamp
        yield timestamp, bytes(message.data)


def synchronized_pairs(stream0, stream1, counts):
    """Merge exact timestamp intersections, counting discarded unpaired images."""
    streams = [iter(stream0), iter(stream1)]
    current = [next(stream, None) for stream in streams]
    while all(item is not None for item in current):
        if current[0][0] == current[1][0]:
            yield current[0][0], (current[0][1], current[1][1])
            current = [next(stream, None) for stream in streams]
        else:
            index = int(current[1][0] < current[0][0])
            counts[index] += 1
            current[index] = next(streams[index], None)
    for index, item in enumerate(current):
        if item is not None:
            counts[index] += 1 + sum(1 for _ in streams[index])


PANORAMA_FROM_CAM0 = np.diag([-1.0, -1.0, 1.0])  # Original physical images are upside down.


def read_eucm_calibration(path):
    """Use the original EUCM parameters embedded in the official calibration file."""
    text = Path(path).read_text()
    marker = "########### EUCM ############"
    if marker not in text:
        raise ValueError("Expected original EUCM calibration block in official Hilti YAML")
    lines = text.split(marker, 1)[1].splitlines()
    source = "\n".join(line[2:] for line in lines if line.startswith("# "))
    result = yaml.safe_load(source)
    for name in ("cam0", "cam1"):
        camera = result[name]
        if camera["camera_model"] != "eucm" or camera["distortion_model"] != "none":
            raise ValueError("Expected EUCM cameras without additional distortion")
        a, b, fx, fy, cx, cy = camera["intrinsics"]
        if not np.isfinite([a, b, fx, fy, cx, cy]).all() or not (0 < a < 1 and min(b, fx, fy) > 0):
            raise ValueError("Invalid EUCM calibration")
    return result


def lens_map(points, calibration, mask):
    """EUCM projection with original OpenCV integer pixel centres and mask polarity."""
    a, b, fx, fy, cx, cy = calibration["intrinsics"]
    x, y, z = np.moveaxis(points, -1, 0)
    d = np.sqrt(b * (x * x + y * y) + z * z)
    denominator = a * d + (1 - a) * z
    # EUCM's invertible domain (Usenko et al.): z > -w*d. Merely
    # requiring a positive projection denominator admits the folded branch.
    domain_weight = a / (1 - a) if a <= 0.5 else (1 - a) / a
    valid = (denominator > 1e-12) & (z > -domain_weight * d)
    u = fx * np.divide(x, denominator, out=np.zeros_like(x), where=valid) + cx
    v = fy * np.divide(y, denominator, out=np.zeros_like(y), where=valid) + cy
    w, h = calibration["resolution"]
    valid &= (u >= 0) & (v >= 0) & (u < w - 1) & (v < h - 1)
    # Conservatively require all four bilinear taps to be valid.
    i = np.clip(np.floor(u).astype(int), 0, w - 2)
    j = np.clip(np.floor(v).astype(int), 0, h - 2)
    valid &= (mask[j, i] == 0) & (mask[j + 1, i] == 0) & (mask[j, i + 1] == 0) & (mask[j + 1, i + 1] == 0)
    score = np.where(valid, z / np.linalg.norm(points, axis=-1), -np.inf)
    return np.stack([u, v], -1).astype(np.float32), score


def render_maps(calibration, masks, width, height, sphere_radius_m):
    T0 = np.asarray(calibration["cam0"]["T_cam_imu"])
    T1 = np.asarray(calibration["cam1"]["T_cam_imu"])
    relative = T1 @ np.linalg.inv(T0)
    yy, xx = np.indices((height, width))
    longitude = (xx + 0.5) * (2 * np.pi / width) - np.pi
    latitude = (yy + 0.5) * (np.pi / height) - np.pi / 2
    rays = np.stack([np.cos(latitude) * np.sin(longitude), np.sin(latitude), np.cos(latitude) * np.cos(longitude)], -1)
    points0 = sphere_radius_m * (rays @ PANORAMA_FROM_CAM0)
    points1 = points0 @ relative[:3, :3].T + relative[:3, 3]
    projected = [lens_map(p, calibration[f"cam{i}"], masks[i]) for i, p in enumerate((points0, points1))]
    scores = np.stack([item[1] for item in projected])
    winner = np.argmax(scores, axis=0)
    valid = np.isfinite(scores.max(axis=0))
    xy = np.where((winner == 0)[..., None], projected[0][0], projected[1][0])
    xy[~valid] = -1
    # Quantize before offsetting cam1 so joining the images preserves interpolation.
    xy, weights = cv2.convertMaps(xy, None, cv2.CV_16SC2)
    xy[..., 0][winner == 1] += calibration["cam0"]["resolution"][0]
    return xy, weights, valid, relative


def prepare(bag, calibration, masks, output):
    sphere_radius_m = 10.0
    bag, calibration, output = Path(bag).resolve(), Path(calibration).resolve(), Path(output).absolute()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    if len(masks) != 2:
        raise ValueError("Provide two official masks")
    calibration_data = read_eucm_calibration(calibration)
    width = max(1024, 2 * min(camera["resolution"][0] for camera in calibration_data.values()))
    height = width // 2
    original_masks = []
    for i, path in enumerate(masks):
        mask = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        w, h = calibration_data[f"cam{i}"]["resolution"]
        if mask is None or mask.shape != (h, w) or mask.dtype != np.uint8 or not set(np.unique(mask)) <= {0, 255}:
            raise ValueError("Expected binary original-size OpenVINS masks (zero valid)")
        original_masks.append(mask)
    map_xy, map_weights, valid, relative = render_maps(
        calibration_data, original_masks, width, height, sphere_radius_m
    )
    resolutions = [calibration_data[f"cam{i}"]["resolution"] for i in range(2)]
    source = np.zeros((max(h for w, h in resolutions), sum(w for w, h in resolutions), 3), dtype=np.uint8)
    manifest = dict(
        schema_version=1,
        variants={"panorama": dict(rig=None, force_gt_keyframes=False)},
        preparation="hilti-equirectangular-panorama",
        source_camera_model="EUCM",
        image_size=[width, height],
        sphere_radius_m=sphere_radius_m,
        physical_baseline_m=float(np.linalg.norm(relative[:3, 3])),
        geometry="central-panorama-approximation",
        panorama_from_cam0=PANORAMA_FROM_CAM0.tolist(),
        timestamp_unit="nanoseconds",
        unpaired_timestamp_counts=[0, 0],
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=f".{output.name}-", dir=output.parent) as temporary:
        staged = Path(temporary) / "prepared"
        (staged / "panorama" / "images" / "cam0").mkdir(parents=True)
        count = 0
        pending = deque()
        with (
            closing(sqlite3.connect(bag.as_uri() + "?mode=ro", uri=True)) as connection,
            ThreadPoolExecutor(max_workers=_WRITE_WORKERS) as writer,
        ):
            connection.execute("PRAGMA temp_store=MEMORY")
            streams = [compressed_stream(connection, topic) for topic in TOPICS]
            for timestamp, payloads in synchronized_pairs(*streams, manifest["unpaired_timestamp_counts"]):
                if len(pending) == _WRITE_WORKERS:
                    pending.popleft().result()
                count += 1
                offset = 0
                for payload, (w, h) in zip(payloads, resolutions, strict=True):
                    image = cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_COLOR)
                    if image is None or image.shape != (h, w, 3):
                        raise ValueError("Unexpected physical image shape")
                    source[:h, offset : offset + w] = image
                    offset += w
                image = cv2.remap(source, map_xy, map_weights, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
                name = f"panorama/images/cam0/{timestamp // 10**9}.{timestamp % 10**9:09d}.jpg"
                pending.append(writer.submit(_write_jpeg, staged / name, image))
            for future in pending:
                future.result()
        if count < 2:
            raise ValueError("At least two synchronized timestamps are required")
        manifest["timestamp_count"] = count
        manifest["valid_fraction"] = float(valid.mean())
        (staged / "preparation.json").write_text(json.dumps(manifest, indent=2) + "\n")
        if output.exists():
            raise FileExistsError(output)
        staged.rename(output)
    return output / "panorama" / "images"
