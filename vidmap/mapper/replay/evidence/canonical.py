"""Canonical hashing and rotation serialization for mapper byte evidence."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pycolmap

EVIDENCE_SCHEMA_VERSION = 1
RELATIVE_POSE_STATE_SCHEMA_VERSION = 2
ROTATION_CANONICAL_DECIMALS = 13


@dataclass(frozen=True)
class ReplayImageSnapshot:
    image_id: int
    camera_id: int
    frame_id: int
    name: str
    has_pose: bool
    cam_from_world: pycolmap.Rigid3d | None
    features: np.ndarray
    features_undist: np.ndarray
    depth_priors: np.ndarray
    depth_prior_stddevs: np.ndarray
    depth_prior_validity: np.ndarray
    is_depth_outlier: np.ndarray


@dataclass(frozen=True)
class ReplayTrackSnapshot:
    xyz: np.ndarray
    observations: np.ndarray
    loop_closure_observations: np.ndarray


def snapshot_images(state, *, reconstruction=None) -> dict[int, ReplayImageSnapshot]:
    from vidmap.mapper.native.extension import native

    rec = state.reconstruction if reconstruction is None else reconstruction
    snapshots = {}
    for image_id in state.image_order:
        if not rec.exists_image(image_id):
            continue
        image = rec.image(image_id)
        data = state.image_data(image_id)
        snapshots[image_id] = ReplayImageSnapshot(
            image_id=image_id,
            camera_id=image.camera_id,
            frame_id=image.frame_id,
            name=image.name,
            has_pose=image.has_pose,
            cam_from_world=deepcopy(image.cam_from_world()) if image.has_pose else None,
            features=native.point2D_coords(image),
            features_undist=np.asarray(data.bearings).copy(),
            depth_priors=np.asarray(data.depth_values).copy(),
            depth_prior_stddevs=np.asarray(data.depth_stddevs).copy(),
            depth_prior_validity=np.asarray(data.depth_validity, dtype=bool),
            is_depth_outlier=np.asarray(data.is_depth_outlier, dtype=bool),
        )
    return snapshots


def snapshot_tracks(state, *, reconstruction=None) -> dict[int, ReplayTrackSnapshot]:
    rec = state.reconstruction if reconstruction is None else reconstruction
    sidecar_ids = set(state.sidecars.track_ids)
    snapshots = {}
    for point_id, point in rec.points3D.items():
        data = state.sidecars.track(point_id) if point_id in sidecar_ids else None
        snapshots[point_id] = ReplayTrackSnapshot(
            xyz=point.xyz.copy(),
            observations=np.asarray(
                [(el.image_id, el.point2D_idx) for el in point.track.elements], dtype=np.uint32
            ).reshape((-1, 2)),
            loop_closure_observations=(
                np.asarray(data.loop_closure_observations).copy()
                if data is not None
                else np.empty((0, 2), dtype=np.uint32)
            ),
        )
    return snapshots


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def array_hash(value) -> str:
    return hashlib.sha1(np.ascontiguousarray(np.asarray(value)).tobytes()).hexdigest()[:16]


def hash_array_like(value: Any) -> str:
    if value is None:
        return ""
    return hashlib.sha1(np.ascontiguousarray(np.asarray(value)).tobytes()).hexdigest()[:12]


def hash_pose_translation(value: Any) -> str:
    if value is None:
        return ""
    arr = np.round(np.asarray(value, dtype=np.float64), 9)
    arr[arr == 0.0] = 0.0
    return hash_array_like(arr)


def inlier_array(pair: Any) -> np.ndarray:
    values = np.asarray(pair.inlier_indices)
    if values.size == 0:
        return np.empty(0, dtype=np.float64)
    return values.astype(np.int64, copy=False)


def mapping_array_hash(mapping: dict | None) -> str:
    h = hashlib.sha1()
    items = () if mapping is None else mapping.items()
    for key, value in sorted(items, key=lambda item: repr(item[0])):
        h.update(repr(key).encode())
        h.update(np.ascontiguousarray(np.asarray(value)).tobytes())
    return h.hexdigest()[:16]


def stable_tuple_json_hash(rows: list[tuple]) -> str:
    return hashlib.sha1(json.dumps(rows, separators=(",", ":")).encode()).hexdigest()[:16]


def update_intish_hash(h: Any, value: Any) -> None:
    h.update(str(int(value)).encode())
    h.update(b"\x00")


def canonical_rotation_array(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    arr = np.asarray(value, dtype=np.float64)
    arr = np.round(arr, ROTATION_CANONICAL_DECIMALS)
    arr[arr == 0.0] = 0.0
    return arr


def count_content_hash(*values: Any) -> str:
    return hashlib.sha1(json.dumps(values, separators=(",", ":")).encode()).hexdigest()[:16]


def pose_rotation(pose: pycolmap.Rigid3d | None):
    return None if pose is None else pose.rotation


def rotation_matrix(rotation) -> np.ndarray | None:
    return None if rotation is None else np.asarray(rotation.matrix(), dtype=np.float64)


def rotation_quaternion(rotation) -> np.ndarray | None:
    return None if rotation is None else np.asarray(rotation.quat, dtype=np.float64)


def canonical_rotation_quaternion(quat: np.ndarray | None) -> np.ndarray | None:
    if quat is None:
        return None
    norm = np.linalg.norm(quat)
    if not np.isfinite(norm) or norm == 0.0:
        return None
    q = np.asarray(quat, dtype=np.float64) / norm
    if q[3] < 0.0:
        q = -q
    elif q[3] == 0.0:
        for value in q[:3]:
            if value == 0.0:
                continue
            if value < 0.0:
                q = -q
            break
    return q


def json_safe_float(value: Any) -> float | None:
    value = float(value)
    if not np.isfinite(value):
        return None
    value = round(value, ROTATION_CANONICAL_DECIMALS)
    return 0.0 if value == 0.0 else value


def json_safe_vector(value: np.ndarray | None) -> list[float | None] | None:
    if value is None:
        return None
    return [json_safe_float(x) for x in np.asarray(value).reshape(-1)]


def json_safe_matrix(value: np.ndarray | None) -> list[list[float | None]] | None:
    if value is None:
        return None
    arr = np.asarray(value, dtype=np.float64)
    if arr.shape != (3, 3):
        return None
    return [[json_safe_float(x) for x in row] for row in arr]


def image_pose(image: ReplayImageSnapshot):
    return image.cam_from_world if image.has_pose else None


def rotation_artifact(images: dict) -> dict:
    rows = []
    canonical_h = hashlib.sha1()
    raw_h = hashlib.sha1()
    for image_id, image in sorted(images.items(), key=lambda item: int(item[0])):
        pose = image_pose(image)
        rotation = pose_rotation(pose)
        raw_quat = rotation_quaternion(rotation)
        canonical_quat = canonical_rotation_array(canonical_rotation_quaternion(raw_quat))
        canonical_matrix = json_safe_matrix(canonical_rotation_array(rotation_matrix(rotation)))
        row = {
            "image_id": int(image_id),
            "name": str(image.name),
            "has_pose": image.has_pose,
            "raw_quaternion_xyzw": json_safe_vector(raw_quat),
            "canonical_quaternion_xyzw": json_safe_vector(canonical_quat),
            "canonical_rotation_matrix": canonical_matrix,
        }
        rows.append(row)
        canonical_h.update(
            json.dumps(
                {
                    "image_id": int(image_id),
                    "canonical_rotation_matrix": canonical_matrix,
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        )
        raw_h.update(
            json.dumps(
                {
                    "image_id": int(image_id),
                    "raw_quaternion_xyzw": row["raw_quaternion_xyzw"],
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        )
    return {
        "schema": "videosfm.native_rotation_artifact",
        "schema_version": EVIDENCE_SCHEMA_VERSION,
        "count": len(rows),
        "content_hash": canonical_h.hexdigest()[:16],
        "canonical_content_hash": canonical_h.hexdigest()[:16],
        "raw_content_hash": raw_h.hexdigest()[:16],
        "images": rows,
    }


def rotation_artifact_summary(images: dict) -> dict:
    artifact = rotation_artifact(images)
    return {
        "count": artifact["count"],
        "content_hash": artifact["content_hash"],
        "canonical_content_hash": artifact["canonical_content_hash"],
        "raw_content_hash": artifact["raw_content_hash"],
    }
