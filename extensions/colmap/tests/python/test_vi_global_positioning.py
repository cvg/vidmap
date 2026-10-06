# SPDX-License-Identifier: BSD-3-Clause

"""Synthetic verification tests for Joint Inertial Global Positioning (I-GP)."""

from __future__ import annotations

import numpy as np
import pytest
import vidmap_native._core as native
from scipy.spatial.transform import Rotation

import pycolmap


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


def _build_synthetic_igp_scene(
    *,
    num_frames: int = 100,
    dt_frame: float = 0.1,
    true_scale: float = 2.5,
    blackout_frame_range: tuple[int, int] | None = (35, 65),
    outlier_fraction: float = 0.25,
    constant_velocity_degenerate: bool = False,
    preintegrate_at_zero_bias: bool = False,
    seed: int = 42,
):
    """Build a synthetic VI Global Positioning problem with blackout/outliers.

    Args:
        num_frames: Number of frames (100 frames at 10 Hz = 10.0 s).
        dt_frame: Inter-frame interval in seconds.
        true_scale: Ratio of metric world scale to unscaled visual SfM world.
        blackout_frame_range: Half-open 0-based index range [start, end) where
            ALL visual observations are removed (e.g., (35, 65) = 30 frames).
        outlier_fraction: Fraction of remaining visual observations corrupted
            with large directional outliers.
        constant_velocity_degenerate: If True, simulate a zero-acceleration,
            constant-orientation trajectory to trigger the low-acceleration
            observability safeguard.
        preintegrate_at_zero_bias: If True, preintegrate IMU with zero nominal
            bias and pass true_bg via ImuStateRecord to test reintegration.
        seed: Random seed.
    """
    rng = np.random.default_rng(seed)

    imu_rate = 200.0
    dt_imu = 1.0 / imu_rate
    steps_per_interval = int(round(dt_frame / dt_imu))

    gravity_mag = 9.81
    g_canon = np.array([0.0, 0.0, -gravity_mag], dtype=np.float64)

    # Arbitrary 3D world rotation R_world (110 deg around a general 3D axis).
    axis_w = np.array([0.55, -0.72, 0.42], dtype=np.float64)
    axis_w /= np.linalg.norm(axis_w)
    R_world = _so3_exp(axis_w * np.radians(110.0))
    g_world_true = R_world @ g_canon
    g_dir_true = g_world_true / gravity_mag

    bg_true = np.array([0.025, -0.018, 0.012], dtype=np.float64)
    ba_true = np.array([-0.08, 0.06, -0.04], dtype=np.float64)

    # Metric IMU-from-camera extrinsics.
    R_IC_true = _so3_exp(np.array([0.12, -0.09, 0.14], dtype=np.float64))
    t_IC_true = np.array([0.04, -0.03, 0.02], dtype=np.float64)
    R_CI_true = R_IC_true.T
    t_CI_true = -(R_CI_true @ t_IC_true)

    def body_kinematics_in_world(t: float):
        if constant_velocity_degenerate:
            p_c = np.array([0.8 * t, -0.4 * t, 0.2 * t], dtype=np.float64)
            v_c = np.array([0.8, -0.4, 0.2], dtype=np.float64)
            a_c = np.zeros(3, dtype=np.float64)
            R_WB = R_world.copy()
            return R_world @ p_c, R_world @ v_c, R_world @ a_c, R_WB

        p_c = np.array(
            [
                2.2 * np.cos(0.8 * t) + 0.35 * t,
                1.6 * np.sin(1.1 * t),
                0.7 * np.sin(1.5 * t) + 0.05 * t * t,
            ],
            dtype=np.float64,
        )
        v_c = np.array(
            [
                -2.2 * 0.8 * np.sin(0.8 * t) + 0.35,
                1.6 * 1.1 * np.cos(1.1 * t),
                0.7 * 1.5 * np.cos(1.5 * t) + 0.10 * t,
            ],
            dtype=np.float64,
        )
        a_c = np.array(
            [
                -2.2 * 0.8 * 0.8 * np.cos(0.8 * t),
                -1.6 * 1.1 * 1.1 * np.sin(1.1 * t),
                -0.7 * 1.5 * 1.5 * np.sin(1.5 * t) + 0.10,
            ],
            dtype=np.float64,
        )
        aa_c = np.array(
            [
                0.35 * np.sin(0.9 * t),
                0.30 * np.cos(0.7 * t),
                0.40 * t + 0.20 * np.sin(0.5 * t),
            ],
            dtype=np.float64,
        )
        R_WB = R_world @ _so3_exp(aa_c)
        return R_world @ p_c, R_world @ v_c, R_world @ a_c, R_WB

    def body_gyro(t: float) -> np.ndarray:
        if constant_velocity_degenerate:
            return np.zeros(3, dtype=np.float64)
        eps = 1e-6
        _, _, _, R_minus = body_kinematics_in_world(t - eps)
        _, _, _, R_plus = body_kinematics_in_world(t + eps)
        dR = R_minus.T @ R_plus
        return Rotation.from_matrix(dR).as_rotvec() / (2.0 * eps)

    imu_options = pycolmap.ImuPreintegrationOptions()
    imu_options.method = pycolmap.ImuIntegrationMethod.RK4
    init_biases = np.zeros(6, dtype=np.float64)
    if not preintegrate_at_zero_bias:
        init_biases[:3] = bg_true
    imu_calib = pycolmap.ImuCalibration()
    imu_calib.gravity_magnitude = gravity_mag
    imu_calib.gyro_noise_density = 1e-4
    imu_calib.accel_noise_density = 1e-3
    imu_calib.bias_gyro_random_walk_sigma = 1e-5
    imu_calib.bias_accel_random_walk_sigma = 1e-4
    imu_calib.imu_rate = imu_rate

    frame_times = [i * dt_frame for i in range(num_frames)]
    q_iori_list = []
    for i in range(num_frames):
        if constant_velocity_degenerate:
            R_iori = np.eye(3)
        else:
            R_iori = _so3_exp(
                np.array(
                    [
                        0.05 * np.sin(0.4 * i),
                        -0.04 * np.cos(0.3 * i),
                        0.03 * np.sin(0.5 * i),
                    ],
                    dtype=np.float64,
                )
            )
        q_iori_list.append(_quat_xyzw_from_rotmat(R_iori))

    integrators = []
    imu_edges = []
    for i in range(num_frames - 1):
        t0 = frame_times[i]
        t1 = frame_times[i + 1]
        t0_ns = pycolmap.timestamp_from_seconds(t0)
        t1_ns = pycolmap.timestamp_from_seconds(t1)
        integrator = pycolmap.ImuPreintegrator(
            imu_options, imu_calib, t0_ns, t1_ns
        )

        ms = pycolmap.ImuMeasurements()
        for step in range(steps_per_interval + 1):
            t = t0 + step * dt_imu
            t_ns = pycolmap.timestamp_from_seconds(t)
            _, _, a_w, R_WB = body_kinematics_in_world(t)
            w_body = body_gyro(t)
            f_body = R_WB.T @ (a_w - g_world_true)
            ms.insert(
                pycolmap.ImuMeasurement(
                    timestamp=t_ns,
                    accel=f_body + ba_true,
                    gyro=w_body + bg_true,
                )
            )
        integrator.integrate(ms)
        if not preintegrate_at_zero_bias:
            integrator.reintegrate(init_biases)
        edge = native.ImuEdgeRecord()
        edge.image_id1 = i + 1
        edge.image_id2 = i + 2
        edge.data = integrator.extract()
        edge.set_integrator(integrator)
        edge.q_iori_1_xyzw = q_iori_list[i]
        edge.q_iori_2_xyzw = q_iori_list[i + 1]
        integrators.append(integrator)
        imu_edges.append(edge)

    imu_states = []
    for i in range(num_frames):
        st = native.ImuStateRecord()
        st.image_id = i + 1
        st.velocity = np.zeros(3, dtype=np.float64)
        st.bias_gyro = bg_true.copy()
        st.bias_accel = np.zeros(3, dtype=np.float64)
        imu_states.append(st)

    problem = native.MappingProblem()
    cam = native.CameraRecord()
    cam.camera_id = 1
    cam.model_id = int(pycolmap.CameraModelId.SIMPLE_PINHOLE)
    cam.width = 640
    cam.height = 480
    cam.params = np.array([500.0, 320.0, 240.0], dtype=np.float64)
    cam.has_prior_focal_length = True
    problem.add_camera(cam)

    # 80 3D landmarks surrounding the trajectory.
    num_points = 80
    pts_canon = np.column_stack(
        [
            rng.uniform(-4.0, 6.0, size=num_points),
            rng.uniform(-4.0, 4.0, size=num_points),
            rng.uniform(8.0, 15.0, size=num_points),
        ]
    )
    pts_world_metric = (R_world @ pts_canon.T).T
    pts_world_unscaled = pts_world_metric / true_scale

    true_cam_R_cw = []
    true_cam_c_metric = []
    true_cam_c_unscaled = []
    true_v_metric = []

    for i in range(num_frames):
        p_w, v_w, _, R_WB = body_kinematics_in_world(frame_times[i])
        R_iori = _rotmat_from_quat_xyzw(q_iori_list[i])
        R_CW = R_iori @ R_CI_true @ R_WB.T
        R_WC_phys = R_CW.T @ R_iori
        c_w_metric = p_w - R_WC_phys @ t_CI_true
        c_w_unscaled = c_w_metric / true_scale

        true_cam_R_cw.append(R_CW)
        true_cam_c_metric.append(c_w_metric)
        true_cam_c_unscaled.append(c_w_unscaled)
        true_v_metric.append(v_w)

    blackout_set = set()
    if blackout_frame_range is not None:
        blackout_set = set(
            range(blackout_frame_range[0], blackout_frame_range[1])
        )

    # Build per-image keypoints & bearings, corrupting outlier_fraction.
    for i in range(num_frames):
        R_CW = true_cam_R_cw[i]
        t_CW = -(R_CW @ true_cam_c_unscaled[i])
        pts_cam = (R_CW @ pts_world_unscaled.T).T + t_CW
        bearings = pts_cam / np.linalg.norm(pts_cam, axis=1, keepdims=True)
        uv = np.column_stack(
            [
                500.0 * (pts_cam[:, 0] / pts_cam[:, 2]) + 320.0,
                500.0 * (pts_cam[:, 1] / pts_cam[:, 2]) + 240.0,
            ]
        )

        img = native.ImageRecord()
        img.image_id = i + 1
        img.camera_id = 1
        img.frame_id = i + 1
        img.name = f"frame_{i + 1:04d}.png"
        img.pose.has_pose = True
        img.pose.rotation_xyzw = _quat_xyzw_from_rotmat(R_CW)
        img.pose.translation = t_CW
        img.keypoints = uv
        img.bearings = bearings
        problem.add_image(img)

    # Create tracks, omitting all observations in blackout_set and corrupting
    # outlier_fraction of the remaining observations.
    visible_frame_indices = [
        i for i in range(num_frames) if i not in blackout_set
    ]
    total_corrupted = 0
    total_obs = 0

    for pt_idx in range(num_points):
        point3D_id = pt_idx + 1
        obs_list = []
        for i in visible_frame_indices:
            image_id = i + 1
            obs_list.append([image_id, pt_idx])
            total_obs += 1
            if outlier_fraction > 0.0 and rng.random() < outlier_fraction:
                # Corrupt bearing direction by 25-60 degrees.
                img = problem.image(image_id)
                b = img.bearings.copy()
                pert = rng.normal(0.0, 1.0, size=3)
                pert /= np.linalg.norm(pert)
                angle = rng.uniform(np.radians(25.0), np.radians(60.0))
                b[pt_idx] = _so3_exp(pert * angle) @ b[pt_idx]
                b[pt_idx] /= np.linalg.norm(b[pt_idx])
                img.bearings = b
                problem.update_image(img)
                total_corrupted += 1

        track = native.TrackRecord()
        track.point3D_id = point3D_id
        track.xyz = pts_world_unscaled[pt_idx].copy()
        track.color = np.array([128, 128, 128], dtype=np.uint8)
        track.error = 0.0
        track.observations = np.asarray(obs_list, dtype=np.uint32)
        problem.add_track(track)

    imu_from_cam_rec = native.PoseRecord()
    imu_from_cam_rec.has_pose = True
    imu_from_cam_rec.rotation_xyzw = _quat_xyzw_from_rotmat(R_IC_true)
    imu_from_cam_rec.translation = t_IC_true

    return {
        "problem": problem,
        "imu_edges": imu_edges,
        "imu_states": imu_states,
        "integrators": integrators,
        "imu_from_cam": imu_from_cam_rec,
        "true_scale": true_scale,
        "g_dir_true": g_dir_true,
        "g_world_true": g_world_true,
        "bg_true": bg_true,
        "ba_true": ba_true,
        "true_cam_R_cw": np.asarray(true_cam_R_cw),
        "true_cam_c_metric": np.asarray(true_cam_c_metric),
        "true_cam_c_unscaled": np.asarray(true_cam_c_unscaled),
        "true_v_metric": np.asarray(true_v_metric),
        "blackout_set": blackout_set,
        "total_corrupted": total_corrupted,
        "total_obs": total_obs,
    }


def _extract_estimated_centers(
    problem: native.MappingProblem, num_frames: int
) -> np.ndarray:
    centers = []
    for i in range(num_frames):
        img = problem.image(i + 1)
        R_cw = _rotmat_from_quat_xyzw(img.pose.rotation_xyzw)
        c_w = -(R_cw.T @ img.pose.translation)
        centers.append(c_w)
    return np.asarray(centers)


def test_vi_global_positioning_bridges_3s_blackout_and_25pct_outliers():
    """Verify I-GP bridges a 3s (30-frame) blackout with 25% visual outliers."""
    scene = _build_synthetic_igp_scene(
        num_frames=100,
        dt_frame=0.1,
        true_scale=2.5,
        blackout_frame_range=(35, 65),
        outlier_fraction=0.25,
        seed=42,
    )
    problem = scene["problem"]

    options = native.GlobalPositioningOptions()
    options.use_imu = True
    options.use_linear_gravity_warm_start = True
    options.imu_from_cam = scene["imu_from_cam"]
    options.gravity_magnitude = 9.81
    options.generate_random_positions = True
    options.generate_random_points = True
    options.use_initial_positions = False
    options.random_seed = 42
    options.max_num_iterations = 100
    options.num_threads = 4

    result = native.run_global_positioning(
        options,
        problem,
        imu_edges=scene["imu_edges"],
        imu_states=scene["imu_states"],
    )

    assert result.success
    assert result.diagnostics.num_camera_centers == 100
    assert result.diagnostics.num_imu_residuals == 99

    # Gravity direction accuracy (< 0.5 deg).
    grav_err_deg = _angle_deg(result.gravity_direction, scene["g_dir_true"])
    assert grav_err_deg < 0.5, (
        f"Gravity direction error too large: {grav_err_deg:.3f} deg"
    )

    # Camera centers are defined up to a global 3D translation offset in GP.
    # Align translation gauge by mean center offset and check metric position
    # RMSE across all 100 frames and inside the 30-frame blackout window.
    est_centers = _extract_estimated_centers(problem, 100)
    gt_centers = scene["true_cam_c_metric"]
    translation_offset = np.mean(gt_centers - est_centers, axis=0)
    aligned_centers = est_centers + translation_offset

    all_pos_rmse = float(
        np.sqrt(np.mean(np.sum((aligned_centers - gt_centers) ** 2, axis=1)))
    )
    blackout_idx = sorted(scene["blackout_set"])
    blackout_pos_rmse = float(
        np.sqrt(
            np.mean(
                np.sum(
                    (aligned_centers[blackout_idx] - gt_centers[blackout_idx])
                    ** 2,
                    axis=1,
                )
            )
        )
    )
    assert all_pos_rmse < 0.05, (
        f"Overall metric position RMSE too large: {all_pos_rmse:.4f} m"
    )
    assert blackout_pos_rmse < 0.05, (
        f"Blackout metric position RMSE too large: {blackout_pos_rmse:.4f} m"
    )

    # Check metric scale accuracy by comparing pairwise chord lengths vs GT.
    chord_gt = np.linalg.norm(gt_centers[70:] - gt_centers[:30], axis=1)
    chord_est = np.linalg.norm(
        aligned_centers[70:] - aligned_centers[:30], axis=1
    )
    scale_ratio = float(np.median(chord_est / chord_gt))
    assert abs(scale_ratio - 1.0) < 0.015, (
        f"Metric scale ratio error too large: {scale_ratio:.4f}"
    )

    # Check recovered per-frame metric velocities and accelerometer biases.
    est_vel = np.array([result.imu_states[i + 1].velocity for i in range(100)])
    est_ba = np.array([result.imu_states[i + 1].bias_accel for i in range(100)])
    vel_rmse = float(
        np.sqrt(
            np.mean(np.sum((est_vel - scene["true_v_metric"]) ** 2, axis=1))
        )
    )
    ba_rmse = float(
        np.sqrt(np.mean(np.sum((est_ba - scene["ba_true"]) ** 2, axis=1)))
    )
    assert vel_rmse < 0.05, f"Velocity RMSE too large: {vel_rmse:.4f} m/s"
    assert ba_rmse < 0.03, (
        f"Accelerometer bias RMSE too large: {ba_rmse:.4f} m/s^2"
    )


@pytest.mark.parametrize("gravity_angle_deg", [0.0, 90.0, 180.0])
@pytest.mark.parametrize("scale_factor", [0.1, 1.0, 10.0])
def test_vi_global_positioning_option_gp_b_adversarial_initializations(
    gravity_angle_deg: float,
    scale_factor: float,
):
    """Verify Option GP-B invariance to 90/180 deg gravity & 0.1x/10x scale."""
    scene = _build_synthetic_igp_scene(
        num_frames=60,
        dt_frame=0.1,
        true_scale=2.5,
        blackout_frame_range=(20, 35),
        outlier_fraction=0.10,
        seed=7,
    )
    problem = scene["problem"]
    g_true = scene["g_dir_true"]

    # Construct an adversarial initial gravity direction rotated by angle.
    if gravity_angle_deg == 0.0:
        g_init = g_true.copy()
    elif gravity_angle_deg == 180.0:
        g_init = -g_true.copy()
    else:
        ortho = np.cross(g_true, np.array([1.0, 0.0, 0.0]))
        if np.linalg.norm(ortho) < 1e-3:
            ortho = np.cross(g_true, np.array([0.0, 1.0, 0.0]))
        ortho /= np.linalg.norm(ortho)
        g_init = _so3_exp(ortho * np.radians(gravity_angle_deg)) @ g_true

    options = native.GlobalPositioningOptions()
    options.use_imu = True
    options.use_linear_gravity_warm_start = True
    options.imu_from_cam = scene["imu_from_cam"]
    options.gravity_magnitude = 9.81
    options.initial_gravity_direction = g_init
    options.initial_scale = scene["true_scale"] * scale_factor
    options.generate_random_positions = False
    options.generate_random_points = False
    options.use_initial_positions = True
    options.max_num_iterations = 80
    options.num_threads = 4

    result = native.run_global_positioning(
        options,
        problem,
        imu_edges=scene["imu_edges"],
        imu_states=scene["imu_states"],
    )

    assert result.success
    grav_err_deg = _angle_deg(result.gravity_direction, g_true)
    scale_rel_err = (
        abs(result.scale - scene["true_scale"]) / scene["true_scale"]
    )
    assert grav_err_deg < 0.5, (
        f"GP-B failed for angle={gravity_angle_deg}, "
        f"scale_factor={scale_factor}: grav_err={grav_err_deg:.3f} deg"
    )
    assert scale_rel_err < 0.015, (
        f"GP-B failed for angle={gravity_angle_deg}, "
        f"scale_factor={scale_factor}: scale_rel_err={scale_rel_err:.4f}"
    )


def test_vi_global_positioning_gp_b_outperforms_gp_a_on_inverted_gravity():
    """Verify Option GP-B succeeds where Option GP-A fails on 180-deg init."""
    scene_a = _build_synthetic_igp_scene(
        num_frames=60,
        dt_frame=0.1,
        true_scale=2.5,
        blackout_frame_range=(20, 35),
        outlier_fraction=0.10,
        seed=7,
    )
    scene_b = _build_synthetic_igp_scene(
        num_frames=60,
        dt_frame=0.1,
        true_scale=2.5,
        blackout_frame_range=(20, 35),
        outlier_fraction=0.10,
        seed=7,
    )
    g_true = scene_a["g_dir_true"]

    def make_options(use_warm: bool) -> native.GlobalPositioningOptions:
        opts = native.GlobalPositioningOptions()
        opts.use_imu = True
        opts.use_linear_gravity_warm_start = use_warm
        opts.imu_from_cam = scene_a["imu_from_cam"]
        opts.gravity_magnitude = 9.81
        opts.initial_gravity_direction = -g_true
        opts.initial_scale = scene_a["true_scale"] * 10.0
        opts.generate_random_positions = False
        opts.generate_random_points = False
        opts.use_initial_positions = True
        opts.max_num_iterations = 80
        opts.num_threads = 4
        return opts

    res_a = native.run_global_positioning(
        make_options(False),
        scene_a["problem"],
        imu_edges=scene_a["imu_edges"],
        imu_states=scene_a["imu_states"],
    )
    res_b = native.run_global_positioning(
        make_options(True),
        scene_b["problem"],
        imu_edges=scene_b["imu_edges"],
        imu_states=scene_b["imu_states"],
    )

    grav_err_a = _angle_deg(res_a.gravity_direction, g_true)
    grav_err_b = _angle_deg(res_b.gravity_direction, g_true)
    assert grav_err_b < 0.5
    assert grav_err_a > 30.0


def test_vi_global_positioning_constant_velocity_trajectory():
    """Verify constant-velocity motion converges without safeguard priors."""
    scene_degen = _build_synthetic_igp_scene(
        num_frames=40,
        dt_frame=0.1,
        true_scale=2.0,
        blackout_frame_range=None,
        outlier_fraction=0.0,
        constant_velocity_degenerate=True,
        seed=11,
    )
    options = native.GlobalPositioningOptions()
    options.use_imu = True
    options.use_linear_gravity_warm_start = True
    options.imu_from_cam = scene_degen["imu_from_cam"]
    options.generate_random_positions = False
    options.generate_random_points = False
    options.use_initial_positions = True
    options.max_num_iterations = 50

    res_degen = native.run_global_positioning(
        options,
        scene_degen["problem"],
        imu_edges=scene_degen["imu_edges"],
        imu_states=scene_degen["imu_states"],
    )

    assert res_degen.success
    est_ba = np.array(
        [res_degen.imu_states[i + 1].bias_accel for i in range(40)]
    )
    assert np.all(np.isfinite(est_ba))


def test_vi_global_positioning_replaces_temporal_accel_and_reintegrates():
    """Verify temporal accel prior is replaced by IMU and reintegrates."""
    scene = _build_synthetic_igp_scene(
        num_frames=30,
        dt_frame=0.1,
        true_scale=2.0,
        blackout_frame_range=None,
        outlier_fraction=0.0,
        preintegrate_at_zero_bias=True,
        seed=19,
    )
    problem = scene["problem"]

    priors = []
    for i in range(1, 29):
        p = native.TemporalAccelerationPrior()
        p.prev_image_id = i
        p.image_id = i + 1
        p.next_image_id = i + 2
        p.dt_prev = 0.1
        p.dt_next = 0.1
        p.sqrt_observation_count = 5.0
        priors.append(p)

    options = native.GlobalPositioningOptions()
    options.use_imu = True
    options.replace_temporal_acceleration_with_imu = True
    options.use_temporal_acceleration_prior = True
    options.temporal_acceleration_priors = priors
    options.temporal_acceleration_prior_stddev = 1.0
    options.temporal_acceleration_prior_weight = 1.0
    options.imu_from_cam = scene["imu_from_cam"]
    options.generate_random_positions = False
    options.generate_random_points = False
    options.use_initial_positions = True
    options.max_num_iterations = 60

    result = native.run_global_positioning(
        options,
        problem,
        imu_edges=scene["imu_edges"],
        imu_states=scene["imu_states"],
    )

    assert result.success
    assert result.diagnostics.num_temporal_acceleration_residuals == 0
    assert result.diagnostics.num_imu_residuals == 29
    assert _angle_deg(result.gravity_direction, scene["g_dir_true"]) < 0.2
    assert abs(result.scale - scene["true_scale"]) / scene["true_scale"] < 0.01
