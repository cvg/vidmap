"""Value-only conversion between stock pycolmap and native records."""

from __future__ import annotations

import numpy as np
import pycolmap

from .extension import native


def camera_record_from_pycolmap(camera: pycolmap.Camera) -> native.CameraRecord:
    record = native.CameraRecord()
    record.camera_id = int(camera.camera_id)
    record.model_id = int(camera.model)
    record.width = int(camera.width)
    record.height = int(camera.height)
    record.params = np.asarray(camera.params, dtype=np.float64)
    record.has_prior_focal_length = bool(camera.has_prior_focal_length)
    return record


def pose_record_from_pycolmap(pose: pycolmap.Rigid3d | None) -> native.PoseRecord:
    record = native.PoseRecord()
    if pose is not None:
        record.has_pose = True
        record.rotation_xyzw = np.asarray(pose.rotation.quat, dtype=np.float64)
        record.translation = np.asarray(pose.translation, dtype=np.float64)
    return record


def pose_record_to_pycolmap(record: native.PoseRecord) -> pycolmap.Rigid3d | None:
    if not record.has_pose:
        return None
    return pycolmap.Rigid3d(
        rotation=np.asarray(record.rotation_xyzw, dtype=np.float64),
        translation=np.asarray(record.translation, dtype=np.float64),
    )


def image_record_from_pycolmap(
    image: pycolmap.Image,
    keypoints: np.ndarray | None = None,
) -> native.ImageRecord:
    record = native.ImageRecord()
    record.image_id = int(image.image_id)
    record.camera_id = int(image.camera_id)
    record.frame_id = int(image.frame_id)
    record.name = str(image.name)
    record.pose = pose_record_from_pycolmap(image.cam_from_world() if image.has_pose else None)
    if keypoints is None:
        keypoints = np.asarray([point.xy for point in image.points2D], dtype=np.float64)
    record.keypoints = np.asarray(keypoints, dtype=np.float64).reshape((-1, 2))
    return record


def geometry_record_from_pycolmap(
    geometry: pycolmap.TwoViewGeometry,
) -> native.TwoViewGeometryRecord:
    record = native.TwoViewGeometryRecord()
    record.configuration = int(geometry.config)
    if geometry.E is not None:
        record.has_essential = True
        record.essential = np.asarray(geometry.E, dtype=np.float64)
    if geometry.F is not None:
        record.has_fundamental = True
        record.fundamental = np.asarray(geometry.F, dtype=np.float64)
    if geometry.H is not None:
        record.has_homography = True
        record.homography = np.asarray(geometry.H, dtype=np.float64)
    record.cam2_from_cam1 = pose_record_from_pycolmap(geometry.cam2_from_cam1)
    return record


def pair_record_from_pycolmap(
    pair_id: int,
    image_id1: int,
    image_id2: int,
    geometry: pycolmap.TwoViewGeometry,
    matches: np.ndarray,
) -> native.PairRecord:
    record = native.PairRecord()
    record.pair_id = int(pair_id)
    record.image_id1 = int(image_id1)
    record.image_id2 = int(image_id2)
    record.geometry = geometry_record_from_pycolmap(geometry)
    record.all_matches = np.asarray(matches, dtype=np.uint32).reshape((-1, 2))
    record.inlier_indices = np.empty(0, dtype=np.int32)
    return record
