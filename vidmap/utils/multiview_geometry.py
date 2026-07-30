"""Utility functions for 3D geometry operations."""

import numpy as np


def project_point_depths(reconstruction, image_id, point3D_ids):
    """Project selected reconstruction points and return their camera-frame depths."""
    image = reconstruction.images[image_id]
    world_points = np.asarray(
        [reconstruction.point3D(int(point3D_id)).xyz for point3D_id in point3D_ids],
        dtype=np.float64,
    ).reshape((-1, 3))
    camera = reconstruction.cameras[image.camera_id]
    return project_world_points_with_colmap_camera(image, camera, world_points)[1]


def project_world_points_with_colmap_camera(image, camera, world_points):
    """Project world points into an image using a COLMAP camera."""
    camera_from_world = getattr(image, "cam_from_world")
    if callable(camera_from_world):
        camera_from_world = camera_from_world()
    camera_from_world = np.concatenate([camera_from_world.matrix(), np.array([[0, 0, 0, 1]])], axis=0)
    calibration_matrix = camera.calibration_matrix()
    return project_world_points(world_points, camera_from_world, calibration_matrix)


def project_world_points(world_points, camera_from_world, calibration_matrix):
    """Project world points using a homogeneous camera-from-world transform and calibration matrix."""
    points3D_h = np.hstack([world_points, np.ones((world_points.shape[0], 1))])
    points_cam = (camera_from_world @ points3D_h.T)[:3, :].T
    depth = points_cam[:, 2].copy()
    pts = ((calibration_matrix @ (points_cam / depth[:, None]).T).T)[:, :2]
    return pts, depth


def point_has_positive_depth(camera_from_world, world_point):
    """Check if a 3D point has positive depth in the camera coordinate system."""
    point3D_homogeneous = np.append(world_point, 1)
    third_row = camera_from_world[2, :]
    depth = np.dot(third_row, point3D_homogeneous)
    return depth >= np.finfo(float).eps
