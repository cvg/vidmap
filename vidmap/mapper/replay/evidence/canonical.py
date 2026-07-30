"""Canonical hashing and rotation serialization for mapper byte evidence."""

from __future__ import annotations

import hashlib
import json
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
    angular_stddevs: np.ndarray
    is_inlier: np.ndarray
    is_track_anchor: np.ndarray
    is_depth_outlier: np.ndarray


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


def int_sort_key(value: Any) -> tuple[int, Any]:
    return (0, int(value))


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


def pair_cam2_from_cam1(pair: Any) -> Any:
    pose = pair.geometry.cam2_from_cam1
    return pose if pose.has_pose else None


def image_value(image: Any, output_name: str) -> Any:
    if isinstance(image, ReplayImageSnapshot):
        values = {
            "features": image.features,
            "features_undist": image.features_undist,
            "angular_stddevs": image.angular_stddevs,
            "depth_priors": image.depth_priors,
            "depth_prior_stddevs": image.depth_prior_stddevs,
            "depth_prior_validity": image.depth_prior_validity,
            "is_inlier": image.is_inlier,
            "is_depth_outlier": image.is_depth_outlier,
            "is_track_anchor": image.is_track_anchor,
        }
    else:
        values = {
            "features": image.keypoints,
            "features_undist": image.bearings,
            "angular_stddevs": image.angular_stddevs,
            "depth_priors": image.depth_values,
            "depth_prior_stddevs": image.depth_stddevs,
            "depth_prior_validity": image.depth_validity,
            "is_inlier": image.is_inlier,
            "is_depth_outlier": image.is_depth_outlier,
            "is_track_anchor": image.is_track_anchor,
        }
    value = values[output_name]
    if output_name in {
        "depth_prior_validity",
        "is_inlier",
        "is_depth_outlier",
        "is_track_anchor",
    }:
        return np.asarray(value, dtype=bool)
    return value


def image_has_pose(image: Any) -> bool:
    if isinstance(image, ReplayImageSnapshot):
        return image.has_pose
    return bool(image.pose.has_pose)


def pose_rotation(pose: Any) -> Any:
    if pose is None:
        return None
    if isinstance(pose, pycolmap.Rigid3d):
        return pose.rotation
    return pycolmap.Rotation3d(np.asarray(pose.rotation_xyzw, dtype=np.float64))


def rotation_matrix(rotation: Any) -> np.ndarray | None:
    if rotation is None:
        return None
    if isinstance(rotation, pycolmap.Rotation3d):
        return np.asarray(rotation.matrix(), dtype=np.float64)
    q = np.asarray(rotation.rotation_xyzw, dtype=np.float64).reshape(-1)
    if q.size != 4:
        return None
    norm = np.linalg.norm(q)
    if not np.isfinite(norm) or norm == 0.0:
        return None
    x, y, z, w = q / norm
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return np.array(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=np.float64,
    )


def rotation_quaternion(rotation: Any) -> np.ndarray | None:
    if rotation is None:
        return None
    quaternion = rotation.quat if isinstance(rotation, pycolmap.Rotation3d) else rotation.rotation_xyzw
    q = np.asarray(quaternion, dtype=np.float64).reshape(-1)
    return q if q.size == 4 else None


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


def image_pose(image):
    if isinstance(image, ReplayImageSnapshot):
        return image.cam_from_world if image.has_pose else None
    return image.pose if image.pose.has_pose else None


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
            "has_pose": image_has_pose(image),
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
