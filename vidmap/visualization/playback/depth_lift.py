"""Full-depth artifact discovery and bounded per-frame world-space lifting."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import h5py
import numpy as np

from vidmap.utils.io import read_image


class DepthLiftReader:
    """Read and lift at most one current full depth map at a time."""

    def __init__(
        self,
        full_path: Path,
        sampled_path: Path | None,
        model: Any,
        rgb_dir: Path,
        *,
        scale_by_image_id: dict[int, float] | None = None,
    ) -> None:
        self.full_path = Path(full_path)
        self.sampled_path = None if sampled_path is None else Path(sampled_path)
        self.model = model
        self.rgb_dir = Path(rgb_dir)
        self.scale_by_image_id = {} if scale_by_image_id is None else dict(scale_by_image_id)
        self._cached_key = None
        self._cached_value = None

    def lift(
        self,
        image: Any,
        *,
        alignment: Any | None,
        stride: int,
        max_depth: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        key = (str(image.name), int(stride), float(max_depth))
        if key != self._cached_key:
            self._cached_key = key
            self._cached_value = self._lift_unaligned(image, stride=stride, max_depth=max_depth)
        points, colors = self._cached_value
        if alignment is not None and len(points):
            points = np.asarray(alignment * points)
        return points, colors

    def _lift_unaligned(self, image: Any, *, stride: int, max_depth: float) -> tuple[np.ndarray, np.ndarray]:
        with h5py.File(self.full_path, "r") as hfile:
            name = str(image.name)
            if name not in hfile:
                return _empty_cloud()
            group = hfile[name]
            depth = np.asarray(group["depth"], dtype=np.float64)
            valid = np.asarray(group["valid"], dtype=bool)
            original_width = int(group.attrs["original_width"])
            original_height = int(group.attrs["original_height"])
        if depth.ndim != 2 or valid.shape != depth.shape:
            raise ValueError(f"Malformed full depth map for {image.name!r} in {self.full_path}")
        ys, xs = np.mgrid[0 : depth.shape[0] : stride, 0 : depth.shape[1] : stride]
        values = depth[ys, xs].reshape(-1)
        keep = valid[ys, xs].reshape(-1) & np.isfinite(values) & (values > 0.0)
        if not np.any(keep):
            return _empty_cloud()
        values = values[keep] * self._reconstruction_scale(image)
        xys = np.column_stack(
            [
                xs.reshape(-1)[keep] * original_width / depth.shape[1],
                ys.reshape(-1)[keep] * original_height / depth.shape[0],
            ]
        )
        keep_depth = np.isfinite(values) & (values > 0.0) & (values <= max_depth)
        values = values[keep_depth]
        xys = xys[keep_depth]
        if not len(values):
            return _empty_cloud()

        camera = self.model.cameras[image.camera_id]
        normalized = np.asarray(camera.cam_from_img(xys), dtype=np.float64).reshape((-1, 2))
        points_camera = np.column_stack([normalized, np.ones(len(normalized))]) * values[:, None]
        cam_from_world = image.cam_from_world()
        rotation = np.asarray(cam_from_world.rotation.matrix(), dtype=np.float64).reshape((3, 3))
        translation = np.asarray(cam_from_world.translation, dtype=np.float64).reshape(3)
        points_world = (points_camera - translation) @ rotation

        rgb = read_image(self.rgb_dir / str(image.name))
        pixels = np.rint(xys).astype(np.int64)
        pixels[:, 0] = np.clip(pixels[:, 0], 0, rgb.shape[1] - 1)
        pixels[:, 1] = np.clip(pixels[:, 1], 0, rgb.shape[0] - 1)
        return points_world, np.asarray(rgb[pixels[:, 1], pixels[:, 0]], dtype=np.uint8)

    def _reconstruction_scale(self, image: Any) -> float:
        """Robustly align raw depth to the selected final reconstruction."""
        image_id = int(image.image_id)
        if image_id in self.scale_by_image_id:
            return self.scale_by_image_id[image_id]
        if self.sampled_path is None:
            raise FileNotFoundError(
                f"Cannot align full depth for {image.name!r}: no optimized scale or sampled-depth artifact"
            )
        with h5py.File(self.sampled_path, "r") as hfile:
            group = hfile[str(image.name)]
            depth = np.asarray(group["depth"], dtype=np.float64).reshape(-1)
            valid = np.asarray(group["valid"], dtype=bool).reshape(-1)
        cam_from_world = image.cam_from_world()
        ratios = []
        for index, point2d in enumerate(image.points2D):
            if index >= len(depth) or not valid[index] or not point2d.has_point3D() or depth[index] <= 0.0:
                continue
            point = self.model.point3D(int(point2d.point3D_id)).xyz
            camera_point = np.asarray(cam_from_world * point, dtype=np.float64).reshape(3)
            if np.isfinite(camera_point[2]) and camera_point[2] > 0.0:
                ratios.append(float(camera_point[2] / depth[index]))
        if not ratios:
            raise ValueError(f"Cannot align full depth for {image.name!r}: no valid 3D depth correspondences")
        scale = float(np.median(np.asarray(ratios, dtype=np.float64)))
        if not np.isfinite(scale) or scale <= 0.0:
            raise ValueError(f"Cannot align full depth for {image.name!r}: invalid fitted scale {scale}")
        return scale


def _empty_cloud() -> tuple[np.ndarray, np.ndarray]:
    return np.empty((0, 3), dtype=np.float64), np.empty((0, 3), dtype=np.uint8)
