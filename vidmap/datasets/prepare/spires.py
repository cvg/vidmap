"""Prepare native and rectified Spires camera streams."""

import io
import json
import os
import shutil
from decimal import Decimal
from pathlib import Path
from tempfile import TemporaryDirectory

import cv2
import numpy as np
import yaml
from PIL import Image
from scipy.spatial.transform import Rotation

from vidmap.datasets.prepare.helpers.rig_capture import write_frame_table


def prepare(bags, calibration_dir, output, *, mode="undistorted-mono", previous=None):
    """Publish the requested camera streams atomically."""
    output = Path(output).absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=f".{output.name}-", dir=output.parent) as temporary:
        staged = Path(temporary) / "prepared"
        result = _prepare(bags, calibration_dir, staged, mode=mode, previous=previous)
        if output.exists():
            raise FileExistsError(output)
        staged.rename(output)
    return result


def _prepare(bags, calibration_dir, output, *, mode, previous):
    from rosbags.highlevel import AnyReader

    from vidmap.datasets.prepare.helpers.ros_benchmarks import modes

    bags = [Path(path).resolve(strict=True) for path in bags]
    calroot = Path(calibration_dir).resolve(strict=True)
    output = Path(output).absolute()
    if not bags or len(set(bags)) != len(bags):
        raise ValueError("Provide distinct raw bags in recording order")
    definitions = modes("spires")
    selections = definitions.values() if mode == "all" else (definitions[mode],)
    streams = {}
    for selection in selections:
        variant = selection["variant"]
        streams[variant] = max(streams[variant] if variant in streams else 0, selection["num_cameras"])
    if previous is not None:
        previous = Path(previous)
        metadata = json.loads((previous / "preparation.json").read_text())
        for variant, spec in metadata["variants"].items():
            streams[variant] = max(streams[variant] if variant in streams else 0, len(spec["rig"]["cameras"]))
    streams = dict(sorted(streams.items()))
    output.mkdir(parents=True, exist_ok=False)
    pending = {variant: set() for variant in streams}
    for variant, count in streams.items():
        for i in range(count):
            relative = Path(variant) / "images" / f"cam{i}"
            if previous is not None and (previous / relative).is_dir():
                # Link immutable images into staging; models and metadata are rebuilt separately.
                shutil.copytree(previous / relative, output / relative, copy_function=os.link)
            else:
                (output / relative).mkdir(parents=True)
                pending[variant].add(i)
    camera_ids = range(max(streams.values()))
    maps = {}
    sizes = {}
    intrinsics = {}
    native_cameras = []
    for i in camera_ids:
        c = yaml.safe_load((calroot / f"cam{i}.yaml").read_text())
        K = np.array(c["camera_matrix"]["data"]).reshape(3, 3)
        D = np.array(c["distortion_coefficients"]["data"])
        size = (c["image_width"], c["image_height"])
        newK = K.copy()
        newK[0, 2], newK[1, 2] = size[0] / 2.0, size[1] / 2.0
        # OpenCV uses integer pixel centres; COLMAP uses half-integer centres.
        intrinsics[i] = [float(newK[0, 0]), float(newK[1, 1]), float(newK[0, 2] + 0.5), float(newK[1, 2] + 0.5)]
        if "rectified" in pending and i in pending["rectified"]:
            maps[i] = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), newK, size, cv2.CV_16SC2)
        sizes[i] = size
        native_cameras.append(
            dict(
                camera_model_name="OPENCV_FISHEYE",
                camera_params=[
                    float(K[0, 0]),
                    float(K[1, 1]),
                    float(K[0, 2] + 0.5),
                    float(K[1, 2] + 0.5),
                    *D.tolist(),
                ],
            )
        )
    extrinsics_path = calroot / "cam-lidar.yaml"
    extrinsics = yaml.safe_load(extrinsics_path.read_text())
    transforms = [np.asarray(extrinsics[f"cam{i}"]["T_cam_lidar"], dtype=float) for i in camera_ids]
    for transform in transforms:
        if (
            transform.shape != (4, 4)
            or not np.isfinite(transform).all()
            or not np.allclose(transform[3], [0, 0, 0, 1])
            or not np.allclose(transform[:3, :3].T @ transform[:3, :3], np.eye(3), atol=1e-6)
            or not np.isclose(np.linalg.det(transform[:3, :3]), 1)
        ):
            raise ValueError("Invalid camera-to-LiDAR rigid calibration")
    cameras = []
    for i in camera_ids:
        camera = dict(camera_model_name="PINHOLE", camera_params=intrinsics[i])
        if i:
            relative = transforms[i] @ np.linalg.inv(transforms[0])
            x, y, z, w = Rotation.from_matrix(relative[:3, :3]).as_quat()
            camera.update(
                cam_from_rig_rotation=[float(w), float(x), float(y), float(z)],
                cam_from_rig_translation=relative[:3, 3].tolist(),
            )
            native_cameras[i].update(
                cam_from_rig_rotation=camera["cam_from_rig_rotation"],
                cam_from_rig_translation=camera["cam_from_rig_translation"],
            )
        cameras.append(camera)
    needed = set().union(*pending.values())
    topics = {f"/alphasense_driver_ros/cam{i}/debayered/image/compressed": i for i in needed}
    seen = {i: set() for i in needed}
    for bag in bags if needed else ():
        with AnyReader([bag]) as reader:
            connections = [c for c in reader.connections if c.topic in topics]
            if set(c.topic for c in connections) != set(topics):
                raise ValueError("Missing camera topic")
            for connection, _, raw in reader.messages(connections=connections):
                i = topics[connection.topic]
                message = reader.deserialize(raw, connection.msgtype)
                t = int(message.header.stamp.sec) * 10**9 + int(message.header.stamp.nanosec)
                if t in seen[i]:
                    raise ValueError("Duplicate image header timestamp")
                seen[i].add(t)
                name = f"{t//10**9}.{t%10**9:09d}.jpg"
                payload = bytes(message.data)
                if not payload.startswith(b"\xff\xd8"):
                    raise ValueError("Expected original JPEG bytes")
                if "native" in pending and i in pending["native"]:
                    (output / "native/images" / f"cam{i}" / name).write_bytes(payload)
                if "rectified" in pending and i in pending["rectified"]:
                    image = cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_COLOR)
                    if image is None or image.shape[:2] != sizes[i][::-1]:
                        raise ValueError("Camera dimensions changed")
                    image = cv2.remap(image, *maps[i], interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
                    if not cv2.imwrite(str(output / "rectified/images" / f"cam{i}" / name), image):
                        raise OSError("Image encoding failed")
                else:
                    with Image.open(io.BytesIO(payload)) as image:
                        if image.size != sizes[i]:
                            raise ValueError("Camera dimensions changed")
    seen = {}
    variants = {}
    for variant, count in streams.items():
        timestamps = {
            i: {int(Decimal(path.stem) * 10**9) for path in (output / variant / "images" / f"cam{i}").glob("*.jpg")}
            for i in range(count)
        }
        for i, values in timestamps.items():
            if i in seen and seen[i] != values:
                raise ValueError(f"Native and rectified cam{i} timestamps differ")
            seen[i] = values
        stamps = sorted(set.intersection(*timestamps.values()))
        if len(stamps) < 2:
            raise ValueError("Insufficient synchronized timestamps")
        frames = None
        if count > 1:
            frames = "frames.csv"
            write_frame_table(
                output / variant / "images",
                [{t: f"cam{i}/{t//10**9}.{t%10**9:09d}.jpg" for t in values} for i, values in timestamps.items()],
            )
        calibration = native_cameras if variant == "native" else cameras
        variants[variant] = dict(rig=dict(cameras=calibration[:count], frames=frames), force_gt_keyframes=False)
    stamps = set.intersection(*seen.values())
    result = dict(
        variants=variants,
        per_camera_frames={str(i): len(s) for i, s in seen.items()},
        rig_frames=len(stamps),
        unpaired_frames={str(i): len(s) - len(stamps) for i, s in seen.items()},
        mono_frames=len(seen[0]),
    )
    (output / "preparation.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main(scenes=None, *, mode="undistorted-mono", download=True, delete_files=False, workers=1):
    from vidmap.datasets.prepare.helpers.ros_benchmarks import main as prepare_dataset

    prepare_dataset("spires", scenes, mode=mode, download=download, delete_files=delete_files, workers=workers)


if __name__ == "__main__":
    from vidmap.datasets.prepare.helpers.ros_benchmarks import run_cli

    run_cli("spires")
