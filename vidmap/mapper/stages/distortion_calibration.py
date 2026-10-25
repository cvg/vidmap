"""Shared focal/distortion calibration from independent two-view problems."""

import logging

import cv2
import numpy as np
import pycolmap

from vidmap.mapper.native.extension import native
from vidmap.utils.camera_geometry import validate_distorted_camera

logger = logging.getLogger(__name__)


def calibrate_distorted_view_graph(state, options, exclusions):
    rec = state.reconstruction
    for cid, camera in rec.cameras.items():
        pairs = [
            pid
            for pid in state.pair_order
            if pid not in exclusions
            and state.pose_graph.is_valid(pid)
            and all(rec.image(i).camera_id == cid for i in pycolmap.pair_id_to_image_pair(pid))
        ]
        indices = np.linspace(0, len(pairs) - 1, min(len(pairs), options.max_distortion_pairs), dtype=int)
        observations = []
        for index in indices:
            pid = pairs[index]
            first, second = pycolmap.pair_id_to_image_pair(pid)
            matches = np.asarray(state.pair_data(pid).all_matches)
            subset = np.linspace(
                0, len(matches) - 1, min(len(matches), options.max_distortion_matches_per_pair), dtype=int
            )
            matches = matches[subset]
            observations.append(
                (
                    native.point2D_coords(rec.image(first))[matches[:, 0]],
                    native.point2D_coords(rec.image(second))[matches[:, 1]],
                )
            )
        camera.params = calibrate_distorted_camera(camera, observations)


def calibrate_distorted_camera(camera, pairs):
    """Export only calibration; each pair has its own nuisance pose and points."""
    validate_distorted_camera(camera)
    rec = pycolmap.Reconstruction()
    calibration = pycolmap.Camera(camera.todict())
    calibration.camera_id = 1
    rec.add_camera_with_trivial_rig(calibration)
    initialized = []
    source_pixels = []
    ransac = pycolmap.RANSACOptions(max_error=4.0, max_num_trials=20000)
    for source, target in pairs:
        if len(source) < 15:
            continue
        result = pycolmap.estimate_relative_pose(camera, source, camera, target, ransac)
        if result is None or result["num_inliers"] < 15:
            continue
        source = source[result["inlier_mask"]]
        target = target[result["inlier_mask"]]
        pose = result["cam2_from_cam1"]
        points = cv2.triangulatePoints(
            np.column_stack([np.eye(3), np.zeros(3)]),
            pose.matrix(),
            camera.cam_from_img(source).T,
            camera.cam_from_img(target).T,
        )
        points = (points[:3] / points[3]).T
        keep = np.isfinite(points).all(axis=1) & (points[:, 2] > 0) & ((pose * points)[:, 2] > 0)
        if keep.sum() < 15:
            continue
        initialized.append((source[keep], target[keep], points[keep], pose))
        source_pixels.extend(source[keep])
    if not initialized:
        raise RuntimeError("Distortion VGC has no usable two-view reconstructions")
    rec.add_image_with_trivial_frame(
        pycolmap.Image(
            image_id=1,
            camera_id=1,
            name="origin",
            points2D=pycolmap.Point2DList([pycolmap.Point2D(xy) for xy in source_pixels]),
        ),
        pycolmap.Rigid3d(),
    )
    offset = 0
    for iid, (source, target, points, pose) in enumerate(initialized, start=2):
        rec.add_image_with_trivial_frame(
            pycolmap.Image(
                image_id=iid,
                camera_id=1,
                name=str(iid),
                points2D=pycolmap.Point2DList([pycolmap.Point2D(xy) for xy in target]),
            ),
            pose,
        )
        for index, xyz in enumerate(points):
            rec.add_point3D(
                xyz, pycolmap.Track([pycolmap.TrackElement(1, offset + index), pycolmap.TrackElement(iid, index)])
            )
        offset += len(source)
    config = pycolmap.BundleAdjustmentConfig()
    for iid in rec.images:
        config.add_image(iid)
    config.set_constant_rig_from_world_pose(1)
    config.fix_gauge(pycolmap.BundleAdjustmentGauge.UNSPECIFIED)
    ba = pycolmap.BundleAdjustmentOptions(
        refine_focal_length=True,
        refine_principal_point=False,
        refine_extra_params=True,
        refine_sensor_from_rig=False,
        print_summary=False,
    )
    ba.ceres.loss_function_type = pycolmap.LossFunctionType.CAUCHY
    ba.ceres.loss_function_scale = 2.0
    ba.ceres.solver_options.num_threads = 1
    ba.ceres.solver_options.function_tolerance = 1e-8
    owner = pycolmap.create_default_bundle_adjuster(ba, config, rec)
    summary = owner.solve()
    if not summary.is_solution_usable():
        raise RuntimeError(f"Distortion VGC failed: {summary.brief_report()}")
    validate_distorted_camera(rec.camera(1))
    logger.info(f"Distortion VGC: {len(initialized)} pairs, {offset} matches; {summary.brief_report()}")
    return rec.camera(1).params.copy()
