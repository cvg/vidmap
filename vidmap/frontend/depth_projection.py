"""Frozen forward-view geometry and depth transfer to working image pixels."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

import cv2
import h5py
import numpy as np
import pycolmap

from vidmap.frontend.cache import canonical_json
from vidmap.utils.camera_geometry import forward_depth_camera, image_remap


@dataclass(frozen=True)
class ForwardDepthProjection:
    """Owned native camera snapshots; both views share the same optical axis."""

    _source: pycolmap.Camera
    _auxiliary: pycolmap.Camera

    def __post_init__(self):
        object.__setattr__(self, "_source", pycolmap.Camera(self._source.todict()))
        object.__setattr__(self, "_auxiliary", pycolmap.Camera(self._auxiliary.todict()))

    @classmethod
    def from_camera(cls, camera: pycolmap.Camera) -> ForwardDepthProjection:
        return cls(camera, forward_depth_camera(camera))

    def image_remap(self) -> np.ndarray:
        return image_remap(self._source, self._auxiliary)

    def depth_remap(self, shape: tuple[int, int]) -> np.ndarray:
        """Map predicted axial depth back without changing its optical-axis convention."""
        forward = pycolmap.Camera(self._auxiliary.todict())
        forward.rescale(shape[1], shape[0])
        remap = image_remap(forward, self._source)
        support = (
            np.isfinite(remap).all(axis=2)
            & (remap[..., 0] >= 0)
            & (remap[..., 0] < shape[1] - 1)
            & (remap[..., 1] >= 0)
            & (remap[..., 1] < shape[0] - 1)
        )
        remap[~support] = -1
        return remap

    def metadata(self) -> dict:
        return {
            name: {
                "model": camera.model_name,
                "width": camera.width,
                "height": camera.height,
                "params": camera.params.tolist(),
            }
            for name, camera in (("source", self._source), ("auxiliary", self._auxiliary))
        }


def resolve_depth_projections(
    reconstruction: pycolmap.Reconstruction, camera_ids: Iterable[int]
) -> dict[int, ForwardDepthProjection]:
    """Resolve only the auxiliary views selected by the preparation plan."""
    return {cid: ForwardDepthProjection.from_camera(reconstruction.camera(cid)) for cid in camera_ids}


def depth_projection_identity(projections: Mapping[int, ForwardDepthProjection] | None) -> dict | None:
    if not projections:
        return None
    return {
        "convention": "forward-axial-z",
        "cameras": {str(cid): projection.metadata() for cid, projection in sorted(projections.items())},
    }


def write_depth_projection(path: Path, identity: dict | None) -> None:
    """Include frozen projection geometry in the certified H5 payload."""
    if identity is not None:
        with h5py.File(path, "a") as hfile:
            hfile.attrs["depth_projection"] = canonical_json(identity)


def load_depth_projections(path: Path) -> dict[int, ForwardDepthProjection]:
    """Recover the original inference projection without consulting optimized cameras."""
    with h5py.File(path, "r") as hfile:
        if "depth_projection" not in hfile.attrs:
            return {}
        identity = json.loads(hfile.attrs["depth_projection"])
    if identity["convention"] != "forward-axial-z":
        raise ValueError("Unsupported depth projection convention")
    return {
        int(cid): ForwardDepthProjection(pycolmap.Camera(value["source"]), pycolmap.Camera(value["auxiliary"]))
        for cid, value in identity["cameras"].items()
    }


def restore_projected_depth(remap, depth, valid, confidence):
    """Keep observations outside the auxiliary view, but give them no depth."""
    # Require all four interpolation neighbors to have valid depth.
    valid = cv2.remap(valid.astype(np.float32), remap, None, cv2.INTER_LINEAR) >= 1.0
    depth = cv2.remap(depth, remap, None, cv2.INTER_LINEAR)
    confidence = cv2.remap(confidence, remap, None, cv2.INTER_LINEAR)
    valid &= np.isfinite(depth) & (depth > 0)
    depth[~valid] = 0
    confidence[~valid] = 0
    return depth, valid, confidence
