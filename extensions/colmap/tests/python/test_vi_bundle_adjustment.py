# SPDX-License-Identifier: BSD-3-Clause

"""Synthetic verification tests for Visual-Inertial Bundle Adjustment."""

from __future__ import annotations

import numpy as np
import pycolmap
import pycolmap.inertial
import vidmap_native._core as native


def _skew(v: np.ndarray) -> np.ndarray:
    return np.array(
        [
            [0.0, -v[2], v[1]],
            [v[2], 0.0, -v[0]],
            [-v[1], v[0], 0.0],
        ],
        dtype=np.float64,
    )


def _so3_exp(w: np.ndarray) -> np.ndarray:
    theta = float(np.linalg.norm(w))
    if theta < 1e-12:
        return np.eye(3) + _skew(w)
    axis = w / theta
    K = _skew(axis)
    return np.eye(3) + np.sin(theta) * K + (1.0 - np.cos(theta)) * (K @ K)


def _quat_xyzw_from_rotmat(R: np.ndarray) -> np.ndarray:
    return pycolmap.Rotation3d(R).quat


def _rotmat_from_quat_xyzw(q_xyzw: np.ndarray) -> np.ndarray:
    return pycolmap.Rotation3d(np.asarray(q_xyzw, dtype=np.float64)).matrix()


def _angle_deg(u: np.ndarray, v: np.ndarray) -> float:
    u_n = u / np.linalg.norm(u)
    v_n = v / np.linalg.norm(v)
    cos_val = float(np.clip(np.dot(u_n, v_n), -1.0, 1.0))
    return float(np.degrees(np.arccos(cos_val)))


def _rot_error_deg(R1: np.ndarray, R2: np.ndarray) -> float:
    dR = R1.T @ R2
    cos_val = float(np.clip(0.5 * (np.trace(dR) - 1.0), -1.0, 1.0))
    return float(np.degrees(np.arccos(cos_val)))


def _build_synthetic_vi_scene(
    *,
    true_scale: float = 3.0,
    use_stabilization: bool = True,
    pose_trans_noise_std: float = 0.0,
    pose_rot_noise_deg: float = 0.0,
    point_noise_std: float = 0.0,
    pixel_noise_std: float = 0.0,
    seed: int = 42,
):
    """Generate a synthetic VI trajectory with 3x scale and rotated world."""
    rng = np.random.default_rng(seed)

    num_frames = 21
    dt_frame = 0.2  # 5 Hz keyframes over 4.0 seconds
    imu_rate = 200.0
    dt_imu = 1.0 / imu_rate
    steps_per_interval = int(round(dt_frame / dt_imu))

    gravity_mag = 9.81
    g_canon = np.array([0.0, 0.0, -gravity_mag], dtype=np.float64)

    # Arbitrary 3D world rotation R_world (115 deg around non-trivial axis).
    R_world = _so3_exp(np.array([0.6, -0.7, 0.38]) / np.linalg.norm([0.6, -0.7, 0.38]) * np.radians(115.0))
    g_world_true = R_world @ g_canon
    g_dir_true = g_world_true / gravity_mag

    # True constant biases.
    bg_true = np.array([0.025, -0.018, 0.012], dtype=np.float64)
    ba_true = np.array([-0.08, 0.06, -0.04], dtype=np.float64)

    # Non-identity IMU-from-camera extrinsics in metric units.
    R_IC_true = _so3_exp(np.array([0.15, -0.08, 0.12]))
    t_IC_true = np.array([0.03, -0.05, 0.02], dtype=np.float64)
    R_CI_true = R_IC_true.T
    t_CI_true = -(R_CI_true @ t_IC_true)

    def body_kinematics_in_world(t: float):
        # Smooth analytical 3D position, velocity, and acceleration in canonical
        # frame, then rotated into the arbitrary world frame by R_world.
        p_c = np.array(
            [
                1.8 * np.sin(1.4 * t),
                1.2 * (1.0 - np.cos(1.1 * t)),
                0.6 * np.sin(2.1 * t) + 0.15 * t * t,
            ]
        )
        v_c = np.array(
            [
                1.8 * 1.4 * np.cos(1.4 * t),
                1.2 * 1.1 * np.sin(1.1 * t),
                0.6 * 2.1 * np.cos(2.1 * t) + 0.30 * t,
            ]
        )
        a_c = np.array(
            [
                -1.8 * 1.4 * 1.4 * np.sin(1.4 * t),
                1.2 * 1.1 * 1.1 * np.cos(1.1 * t),
                -0.6 * 2.1 * 2.1 * np.sin(2.1 * t) + 0.30,
            ]
        )
        p_w = R_world @ p_c
        v_w = R_world @ v_c
        a_w = R_world @ a_c

        # Orientation R_WB(t).
        aa_c = np.array(
            [
                0.35 * np.sin(1.3 * t),
                0.40 * np.cos(0.9 * t),
                0.50 * np.sin(0.7 * t),
            ]
        )
        R_WB = R_world @ _so3_exp(aa_c)
        return p_w, v_w, a_w, R_WB

    def body_gyro(t: float) -> np.ndarray:
        eps = 1e-6
        _, _, _, R_minus = body_kinematics_in_world(t - eps)
        _, _, _, R_plus = body_kinematics_in_world(t + eps)
        dR = R_minus.T @ R_plus
        rot_vec = np.array([dR[2, 1] - dR[1, 2], dR[0, 2] - dR[2, 0], dR[1, 0] - dR[0, 1]]) * 0.5
        return rot_vec / (2.0 * eps)

    # IMU calibration and preintegration options.
    imu_options = pycolmap.inertial.ImuPreintegrationOptions()
    imu_options.method = pycolmap.inertial.ImuIntegrationMethod.RK4
    imu_calib = pycolmap.ImuCalibration()
    imu_calib.gravity_magnitude = gravity_mag
    imu_calib.gyro_noise_density = 1e-3
    imu_calib.accel_noise_density = 1e-2
    imu_calib.bias_gyro_random_walk_sigma = 1e-4
    imu_calib.bias_accel_random_walk_sigma = 1e-3
    imu_calib.imu_rate = imu_rate

    # Keyframe timestamps and stabilization rotations.
    frame_times = [i * dt_frame for i in range(num_frames)]
    q_iori_list = []
    for i in range(num_frames):
        if use_stabilization:
            R_iori = _so3_exp(
                np.array(
                    [
                        0.08 * np.sin(0.5 * i),
                        -0.06 * np.cos(0.4 * i),
                        0.05 * np.sin(0.3 * i),
                    ]
                )
            )
        else:
            R_iori = np.eye(3)
        q_iori_list.append(_quat_xyzw_from_rotmat(R_iori))

    # Pre-integrate IMU measurements across each consecutive keyframe interval.
    integrators = []
    imu_edges = []
    for i in range(num_frames - 1):
        t0 = frame_times[i]
        t1 = frame_times[i + 1]
        t0_ns = pycolmap.timestamp_from_seconds(t0)
        t1_ns = pycolmap.timestamp_from_seconds(t1)
        integrator = pycolmap.inertial.ImuPreintegrator(imu_options, imu_calib, t0_ns, t1_ns)

        ms = pycolmap.ImuMeasurements()
        for step in range(steps_per_interval + 1):
            t = t0 + step * dt_imu
            t_ns = pycolmap.timestamp_from_seconds(t)
            _, _, a_w, R_WB = body_kinematics_in_world(t)
            w_body = body_gyro(t)
            f_body = R_WB.T @ (a_w - g_world_true)
            gyro_meas = w_body + bg_true
            accel_meas = f_body + ba_true
            ms.insert(pycolmap.ImuMeasurement(timestamp=t_ns, accel=accel_meas, gyro=gyro_meas))
        integrator.integrate(ms)
        data = integrator.extract()

        edge = native.ImuEdgeRecord()
        edge.image_id1 = i + 1
        edge.image_id2 = i + 2
        edge.data = data
        edge.set_integrator(integrator)
        edge.q_iori_1_xyzw = q_iori_list[i]
        edge.q_iori_2_xyzw = q_iori_list[i + 1]
        integrators.append(integrator)
        imu_edges.append(edge)

    # Build camera, 3D landmarks, and keyframe poses.
    problem = native.MappingProblem()
    cam = native.CameraRecord()
    cam.camera_id = 1
    cam.model_id = int(pycolmap.CameraModelId.SIMPLE_PINHOLE)
    cam.width = 640
    cam.height = 480
    cam.params = np.array([500.0, 320.0, 240.0], dtype=np.float64)
    cam.has_prior_focal_length = True
    problem.add_camera(cam)

    # Generate 60 3D landmarks in front of the trajectory in the canonical
    # frame, then rotate into the world frame and unscale by true_scale.
    num_points = 60
    pts_canon = np.column_stack(
        [
            rng.uniform(-2.5, 3.5, size=num_points),
            rng.uniform(-2.0, 3.0, size=num_points),
            rng.uniform(7.5, 12.0, size=num_points),
        ]
    )
    pts_world_metric = (R_world @ pts_canon.T).T
    pts_world_unscaled_gt = pts_world_metric / true_scale

    true_cam_R_cw = []
    true_cam_c_unscaled = []
    true_v_metric = []

    for i in range(num_frames):
        p_w, v_w, _, R_WB = body_kinematics_in_world(frame_times[i])
        R_iori = _rotmat_from_quat_xyzw(q_iori_list[i])
        # R_WB = R_WC * R_iori * R_CI => R_CW = R_iori * R_CI * R_BW
        R_CW = R_iori @ R_CI_true @ R_WB.T
        R_WC_phys = R_CW.T @ R_iori
        # p_w = c_w_metric + R_WC_phys * t_CI_true
        c_w_metric = p_w - R_WC_phys @ t_CI_true
        c_w_unscaled = c_w_metric / true_scale

        true_cam_R_cw.append(R_CW)
        true_cam_c_unscaled.append(c_w_unscaled)
        true_v_metric.append(v_w)

    # Project 3D landmarks into every image using the true noiseless geometry.
    keypoints_per_image = []
    for i in range(num_frames):
        R_CW = true_cam_R_cw[i]
        t_CW = -(R_CW @ true_cam_c_unscaled[i])
        pts_cam = (R_CW @ pts_world_unscaled_gt.T).T + t_CW
        uv = np.column_stack(
            [
                500.0 * (pts_cam[:, 0] / pts_cam[:, 2]) + 320.0,
                500.0 * (pts_cam[:, 1] / pts_cam[:, 2]) + 240.0,
            ]
        )
        if pixel_noise_std > 0.0:
            uv = uv + rng.normal(0.0, pixel_noise_std, size=uv.shape)
        keypoints_per_image.append(uv)

    # Add images (with optional pose noise on frames i >= 1).
    for i in range(num_frames):
        R_CW = true_cam_R_cw[i].copy()
        c_w = true_cam_c_unscaled[i].copy()
        if i > 0:
            if pose_rot_noise_deg > 0.0:
                d_rot = rng.normal(0.0, np.radians(pose_rot_noise_deg), size=3)
                R_CW = _so3_exp(d_rot) @ R_CW
            if pose_trans_noise_std > 0.0:
                c_w = c_w + rng.normal(0.0, pose_trans_noise_std, size=3)
        t_CW = -(R_CW @ c_w)

        img = native.ImageRecord()
        img.image_id = i + 1
        img.camera_id = 1
        img.frame_id = i + 1
        img.name = f"frame_{i + 1:03d}.png"
        img.pose.has_pose = True
        img.pose.rotation_xyzw = _quat_xyzw_from_rotmat(R_CW)
        img.pose.translation = t_CW
        img.keypoints = keypoints_per_image[i]
        problem.add_image(img)

    # Add 3D tracks.
    point_ids = []
    for pt_idx in range(num_points):
        point3D_id = pt_idx + 1
        point_ids.append(point3D_id)
        track = native.TrackRecord()
        track.point3D_id = point3D_id
        xyz = pts_world_unscaled_gt[pt_idx].copy()
        if point_noise_std > 0.0:
            xyz = xyz + rng.normal(0.0, point_noise_std, size=3)
        track.xyz = xyz
        track.color = np.array([128, 128, 128], dtype=np.uint8)
        track.error = 0.0
        obs = np.column_stack(
            [
                np.arange(1, num_frames + 1, dtype=np.uint32),
                np.full(num_frames, pt_idx, dtype=np.uint32),
            ]
        )
        track.observations = obs
        problem.add_track(track)

    imu_from_cam_rec = native.PoseRecord()
    imu_from_cam_rec.has_pose = True
    imu_from_cam_rec.rotation_xyzw = _quat_xyzw_from_rotmat(R_IC_true)
    imu_from_cam_rec.translation = t_IC_true

    return {
        "problem": problem,
        "imu_edges": imu_edges,
        "integrators": integrators,
        "point_ids": point_ids,
        "imu_from_cam": imu_from_cam_rec,
        "R_IC_true": R_IC_true,
        "t_IC_true": t_IC_true,
        "true_scale": true_scale,
        "g_dir_true": g_dir_true,
        "g_world_true": g_world_true,
        "bg_true": bg_true,
        "ba_true": ba_true,
        "true_v_metric": np.asarray(true_v_metric),
        "true_cam_R_cw": true_cam_R_cw,
        "true_cam_c_unscaled": np.asarray(true_cam_c_unscaled),
    }


def test_synthetic_viba_mode_a_fixed_pose_alignment():
    """Mode A: Fixed poses, 3x scale perturbation, rotated world, R_iori."""
    scene = _build_synthetic_vi_scene(
        true_scale=3.0,
        use_stabilization=True,
        pose_trans_noise_std=0.0,
        pose_rot_noise_deg=0.0,
    )
    problem = scene["problem"]

    options = native.BundleAdjustmentOptions()
    options.image_order = list(problem.image_ids)
    options.constant_camera_ids = [1]
    options.refine_focal_length = False
    options.refine_principal_point = False
    options.refine_extra_params = False
    options.refine_points3D = False
    options.fix_all_poses = True
    options.use_imu = True
    options.use_analytical_imu_cost = True
    options.imu_from_cam = scene["imu_from_cam"]
    options.initial_log_scale = 0.0  # Unscaled guess s = 1.0 (true s = 3.0)
    options.initial_gravity_direction = np.array([0.0, 0.0, -1.0])
    options.max_num_iterations = 100

    result = native.run_bundle_adjustment(
        options,
        [],
        [],
        [],
        problem,
        imu_edges=scene["imu_edges"],
    )

    assert result.success
    assert result.diagnostics.num_imu_residuals == 20

    scale_rel_err = abs(result.scale - scene["true_scale"]) / scene["true_scale"]
    grav_err_deg = _angle_deg(result.gravity_direction, scene["g_dir_true"])

    est_bg = np.array([result.imu_states[i].bias_gyro for i in problem.image_ids])
    est_ba = np.array([result.imu_states[i].bias_accel for i in problem.image_ids])
    est_v_metric = np.array([result.imu_states[i].metric_velocity for i in problem.image_ids])

    bg_rmse = float(np.sqrt(np.mean(np.sum((est_bg - scene["bg_true"]) ** 2, axis=1))))
    ba_rmse = float(np.sqrt(np.mean(np.sum((est_ba - scene["ba_true"]) ** 2, axis=1))))
    v_rmse = float(np.sqrt(np.mean(np.sum((est_v_metric - scene["true_v_metric"]) ** 2, axis=1))))

    assert scale_rel_err < 0.005, f"Mode A scale error {scale_rel_err * 100:.4f}% >= 0.5%"
    assert grav_err_deg < 0.1, f"Mode A gravity error {grav_err_deg:.4f} deg >= 0.1 deg"
    assert bg_rmse < 1e-3, f"Mode A gyro bias RMSE {bg_rmse:.6f} rad/s"
    assert ba_rmse < 2e-2, f"Mode A accel bias RMSE {ba_rmse:.6f} m/s^2"
    assert v_rmse < 2e-2, f"Mode A velocity RMSE {v_rmse:.6f} m/s"


def test_synthetic_viba_mode_b_joint_optimization_and_extrinsics():
    """Mode B: Jointly optimize noisy poses, 3D tracks, scale, gravity, R_BC."""
    scene = _build_synthetic_vi_scene(
        true_scale=3.0,
        use_stabilization=True,
        pose_trans_noise_std=0.008,
        pose_rot_noise_deg=0.25,
        point_noise_std=0.015,
        pixel_noise_std=0.05,
    )
    problem = scene["problem"]

    # Perturb initial R_BC by 0.35 deg to test R_BC refinement with 0.5 deg
    # prior.
    R_IC_init = _so3_exp(np.radians([0.20, -0.22, 0.18])) @ scene["R_IC_true"]
    init_rot_err_deg = _rot_error_deg(R_IC_init, scene["R_IC_true"])
    assert 0.3 < init_rot_err_deg < 0.45

    imu_from_cam_perturbed = native.PoseRecord()
    imu_from_cam_perturbed.has_pose = True
    imu_from_cam_perturbed.rotation_xyzw = _quat_xyzw_from_rotmat(R_IC_init)
    imu_from_cam_perturbed.translation = scene["t_IC_true"]

    options = native.BundleAdjustmentOptions()
    options.image_order = list(problem.image_ids)
    options.constant_camera_ids = [1]
    options.variable_point3D_ids = scene["point_ids"]
    options.refine_focal_length = False
    options.refine_principal_point = False
    options.refine_extra_params = False
    options.refine_points3D = True
    options.fix_first_pose = True
    options.fix_all_poses = False
    options.use_imu = True
    options.use_analytical_imu_cost = True
    options.refine_imu_from_cam_rotation = True
    options.refine_imu_from_cam_translation = False
    options.imu_from_cam_rotation_prior_stddev_deg = 0.5
    options.imu_from_cam = imu_from_cam_perturbed
    options.initial_log_scale = 0.0
    options.initial_gravity_direction = np.array([0.0, 0.0, -1.0])
    options.max_num_iterations = 100

    result = native.run_bundle_adjustment(
        options,
        [],
        [],
        [],
        problem,
        imu_edges=scene["imu_edges"],
    )

    assert result.success
    assert result.diagnostics.num_imu_residuals == 20
    assert result.diagnostics.num_imu_extrinsics_prior_residuals == 1

    scale_rel_err = abs(result.scale - scene["true_scale"]) / scene["true_scale"]
    grav_err_deg = _angle_deg(result.gravity_direction, scene["g_dir_true"])
    R_IC_final = _rotmat_from_quat_xyzw(result.imu_from_cam.rotation_xyzw)
    final_rot_err_deg = _rot_error_deg(R_IC_final, scene["R_IC_true"])

    est_bg = np.array([result.imu_states[i].bias_gyro for i in problem.image_ids])
    est_ba = np.array([result.imu_states[i].bias_accel for i in problem.image_ids])
    bg_rmse = float(np.sqrt(np.mean(np.sum((est_bg - scene["bg_true"]) ** 2, axis=1))))
    ba_rmse = float(np.sqrt(np.mean(np.sum((est_ba - scene["ba_true"]) ** 2, axis=1))))

    assert scale_rel_err < 0.005, f"Mode B scale error {scale_rel_err * 100:.4f}% >= 0.5%"
    assert grav_err_deg < 0.1, f"Mode B gravity error {grav_err_deg:.4f} deg >= 0.1 deg"
    assert bg_rmse < 1e-3, f"Mode B gyro bias RMSE {bg_rmse:.6f} rad/s"
    assert ba_rmse < 2e-2, f"Mode B accel bias RMSE {ba_rmse:.6f} m/s^2"
    assert final_rot_err_deg < init_rot_err_deg, (
        f"Expected R_BC error to decrease " f"({init_rot_err_deg:.4f} -> {final_rot_err_deg:.4f} deg)"
    )


def test_synthetic_viba_apply_imu_alignment_to_problem():
    """Verify apply_imu_alignment_to_problem transforms to metric frame."""
    scene = _build_synthetic_vi_scene(
        true_scale=3.0,
        use_stabilization=True,
    )
    problem = scene["problem"]

    options = native.BundleAdjustmentOptions()
    options.image_order = list(problem.image_ids)
    options.constant_camera_ids = [1]
    options.refine_focal_length = False
    options.refine_principal_point = False
    options.refine_extra_params = False
    options.refine_points3D = False
    options.fix_all_poses = True
    options.use_imu = True
    options.apply_imu_alignment_to_problem = True
    options.imu_from_cam = scene["imu_from_cam"]

    result = native.run_bundle_adjustment(
        options,
        [],
        [],
        [],
        problem,
        imu_edges=scene["imu_edges"],
    )

    assert result.success
    # Check that camera centers in problem were scaled by ~3.0x to metric units.
    img1 = problem.image(1)
    img21 = problem.image(21)
    c1 = -(_rotmat_from_quat_xyzw(img1.pose.rotation_xyzw).T @ img1.pose.translation)
    c21 = -(_rotmat_from_quat_xyzw(img21.pose.rotation_xyzw).T @ img21.pose.translation)
    dist_metric_est = float(np.linalg.norm(c21 - c1))
    dist_metric_gt = float(
        np.linalg.norm(scene["true_cam_c_unscaled"][-1] - scene["true_cam_c_unscaled"][0]) * scene["true_scale"]
    )
    assert abs(dist_metric_est - dist_metric_gt) / dist_metric_gt < 0.005
