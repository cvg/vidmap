"""Camera-model-aware geometry for native distorted image coordinates."""

import numpy as np
import pycolmap

DISTORTED_CAMERA_MODELS = frozenset(
    {"SIMPLE_RADIAL", "RADIAL", "OPENCV", "FULL_OPENCV", "SIMPLE_DIVISION", "DIVISION", "OPENCV_FISHEYE"}
)


def fit_distorted_camera(camera, K, division):
    """Fit the requested COLMAP model to GeoCalib's division-model rays."""
    if camera.model_name not in DISTORTED_CAMERA_MODELS:
        raise ValueError(f"Calibration prediction does not support {camera.model_name}")
    if camera.model_name == "DIVISION":
        camera.params = [K[0, 0], K[1, 1], K[0, 2], K[1, 2], division]
    elif camera.model_name == "SIMPLE_DIVISION" and K[0, 0] == K[1, 1]:
        camera.params = [K[0, 0], K[0, 2], K[1, 2], division]
    elif camera.model_name == "OPENCV_FISHEYE":
        camera.params = _fit_fisheye_params(K, division, camera.width, camera.height)
    else:
        from scipy.optimize import least_squares

        source = pycolmap.Camera(
            model="DIVISION",
            width=camera.width,
            height=camera.height,
            params=[K[0, 0], K[1, 1], K[0, 2], K[1, 2], division],
        )
        validate_distorted_camera(source)
        pixels = np.stack(
            np.meshgrid(np.linspace(0.5, camera.width - 0.5, 53), np.linspace(0.5, camera.height - 0.5, 41)), -1
        ).reshape(-1, 2)
        rays = source.cam_ray_from_img(pixels)
        params = np.zeros_like(camera.params)
        focal = list(camera.focal_length_idxs())
        params[focal] = np.mean([K[0, 0], K[1, 1]]) if len(focal) == 1 else [K[0, 0], K[1, 1]]
        params[list(camera.principal_point_idxs())] = K[:2, 2]
        indices = focal + list(camera.extra_params_idxs())

        def residual(values):
            params[indices] = values
            camera.params = params
            return (camera.img_from_cam(rays) - pixels).ravel()

        lower = np.r_[np.full(len(focal), 1e-6), np.full(len(indices) - len(focal), -np.inf)]
        result = least_squares(residual, params[indices], bounds=(lower, np.inf), max_nfev=300)
        if not result.success:
            raise RuntimeError(f"Calibration conversion failed for {camera.model_name}: {result.message}")
        params[indices] = result.x
        camera.params = params
    validate_distorted_camera(camera)


def validate_distorted_camera(camera):
    """Check the calibration and forward projection over the sensor."""
    if (
        not camera.verify_params()
        or not np.isfinite(camera.params).all()
        or any(camera.params[i] <= 0 for i in camera.focal_length_idxs())
    ):
        raise ValueError("Calibration requires finite parameters and positive focal lengths")
    if camera.model_name == "OPENCV_FISHEYE":
        _validate_fisheye_camera(camera)
        return
    pixels = np.stack(np.meshgrid(np.linspace(0, camera.width, 17), np.linspace(0, camera.height, 13)), -1).reshape(
        -1, 2
    )
    rays = camera.cam_ray_from_img(pixels)
    if (
        not np.isfinite(rays).all()
        or np.any(rays[:, 2] <= 0)
        or not np.allclose(camera.img_from_cam(rays), pixels, atol=1e-4, rtol=0)
    ):
        raise ValueError("Calibration exceeds its invertible forward-hemisphere domain")


def _fit_fisheye_params(K, division, width, height):
    """Fit OpenCV fisheye coefficients to GeoCalib's division-model rays."""
    focal, center = K[[0, 1], [0, 1]], K[:2, 2]
    if not np.isfinite(K).all() or np.any(focal <= 0) or not np.isfinite(division):
        raise ValueError("Fisheye calibration requires finite parameters and positive focal lengths")
    pixels = np.stack(np.meshgrid(np.linspace(0.5, width - 0.5, 53), np.linspace(0.5, height - 0.5, 41)), -1)
    radius = np.linalg.norm((pixels.reshape(-1, 2) - center) / focal, axis=1)
    if np.any(1 - abs(division) * radius**2 <= 0):
        raise ValueError("GeoCalib calibration exceeds the forward-hemisphere domain")
    theta = np.arctan2(radius, 1 + division * radius**2)
    design = theta[:, None] ** np.array([3, 5, 7, 9])
    coefficients = np.linalg.lstsq(design, radius - theta, rcond=None)[0]
    return np.r_[focal, center, coefficients]


def _validate_fisheye_camera(camera):
    """Require an invertible OpenCV fisheye over the forward-facing sensor."""
    coefficients = camera.params[4:]
    roots = np.polynomial.polynomial.polyroots(np.r_[1, [3, 5, 7, 9] * coefficients])
    limit = min([np.pi / 2, *[np.sqrt(r.real) for r in roots if abs(r.imag) < 1e-9 and r.real > 0]])
    max_radius = limit * np.polynomial.polynomial.polyval(limit**2, np.r_[1, coefficients])
    corners = np.array([[0, 0], [camera.width, 0], [0, camera.height], [camera.width, camera.height]])
    if np.any(np.linalg.norm((corners - camera.params[2:4]) / camera.params[:2], axis=1) >= max_radius):
        raise ValueError("Fisheye calibration exceeds its invertible forward-hemisphere domain")


def undistorted_pixels(camera, points):
    """Map observations to a virtual pinhole with the same intrinsic matrix."""
    xy = camera.cam_from_img(np.asarray(points, dtype=np.float64))
    K = camera.calibration_matrix()
    return xy * K[[0, 1], [0, 1]] + K[:2, 2]


def poselib_camera(camera, *, undistorted=False):
    """Build a PoseLib camera for original or virtual pinhole pixels."""
    import poselib

    if undistorted:
        K = camera.calibration_matrix()
        return poselib.Camera("PINHOLE", [K[0, 0], K[1, 1], K[0, 2], K[1, 2]], camera.width, camera.height)
    return poselib.Camera(camera.model_name, camera.params.tolist(), camera.width, camera.height)


def estimate_distorted_two_view(camera1, points1, camera2, points2, matches, options):
    """Estimate geometry in pixel units without fitting a raw-pixel F or H."""
    geometry = pycolmap.TwoViewGeometry()
    if len(matches) < options.min_num_inliers:
        return geometry
    result = pycolmap.estimate_relative_pose(
        camera1, points1[matches[:, 0]], camera2, points2[matches[:, 1]], options.ransac
    )
    if result is None or result["num_inliers"] < options.min_num_inliers:
        return geometry
    geometry.cam2_from_cam1 = result["cam2_from_cam1"]
    geometry.E = pycolmap.essential_matrix_from_pose(geometry.cam2_from_cam1)
    geometry.inlier_matches = matches[np.asarray(result["inlier_mask"], dtype=bool)]
    geometry.config = pycolmap.TwoViewGeometryConfiguration.CALIBRATED
    return geometry


def forward_depth_camera(camera):
    """Take a blank-free forward crop, capped at 90 degrees, solely for depth."""
    target = pycolmap.undistort_camera(
        pycolmap.UndistortCameraOptions(max_image_size=min(camera.width, camera.height)), camera
    )
    fx, fy, cx, cy = target.params
    size = int(2 * min(fx, fy, cx, cy, target.width - cx, target.height - cy))
    target.width = target.height = size
    target.params = [fx, fy, size / 2, size / 2]
    return target


def image_remap(source, target):
    """OpenCV sampling coordinates for target pixels in a source camera."""
    yy, xx = np.mgrid[: target.height, : target.width]
    pixels = np.column_stack([xx.ravel(), yy.ravel()]) + 0.5
    rays = target.cam_ray_from_img(pixels)
    source_pixels = source.img_from_cam(rays) - 0.5
    return source_pixels.reshape(target.height, target.width, 2).astype(np.float32)


def projection_jacobian(camera, point):
    """Derivative of distorted pixels with respect to a camera-space point."""
    if camera.model_name != "OPENCV_FISHEYE":
        point = np.asarray(point, dtype=np.float64)
        step = 1e-6 * max(1.0, np.linalg.norm(point))
        offsets = step * np.eye(3)
        return ((camera.img_from_cam(point + offsets) - camera.img_from_cam(point - offsets)) / (2 * step)).T
    import cv2

    _, jacobian = cv2.fisheye.projectPoints(
        np.asarray(point, dtype=np.float64).reshape(1, 1, 3),
        np.zeros(3),
        np.zeros(3),
        camera.calibration_matrix(),
        camera.params[4:],
    )
    return jacobian[:, 11:14]
