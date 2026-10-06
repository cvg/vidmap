# SPDX-License-Identifier: BSD-3-Clause

"""End-to-end verification tests for the chained Visual-Inertial Mapper pipeline."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pycolmap
import vidmap_native._core as native

from vidmap.mapper.inputs.loader import MappingStageInputs
from vidmap.mapper.mapper import Mapper
from vidmap.mapper.native.records import (
    camera_record_from_pycolmap,
    image_record_from_pycolmap,
    pair_record_from_pycolmap,
    pose_record_from_pycolmap,
    pose_record_to_pycolmap,
)
from vidmap.mapper.native.state import SolveState
from vidmap.mapper.options import MapperOptions


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


def _angle_deg(u: np.ndarray, v: np.ndarray) -> float:
    u_n = u / np.linalg.norm(u)
    v_n = v / np.linalg.norm(v)
    cos_val = float(np.clip(np.dot(u_n, v_n), -1.0, 1.0))
    return float(np.degrees(np.arccos(cos_val)))


def _se3_ate_rmse(centers_est: np.ndarray, centers_gt: np.ndarray) -> float:
    """Compute rigid SE(3)-aligned (unit scale) ATE RMSE in meters."""
    mu_e = centers_est.mean(axis=0)
    mu_g = centers_gt.mean(axis=0)
    X = centers_est - mu_e
    Y = centers_gt - mu_g
    H = X.T @ Y
    U, _, Vt = np.linalg.svd(H)
    S = np.eye(3)
    if np.linalg.det(Vt.T @ U.T) < 0:
        S[2, 2] = -1.0
    R = Vt.T @ S @ U.T
    t = mu_g - R @ mu_e
    aligned = (R @ centers_est.T).T + t
    return float(np.sqrt(np.mean(np.sum((aligned - centers_gt) ** 2, axis=1))))


def _sim3_ate_rmse_and_scale(centers_est: np.ndarray, centers_gt: np.ndarray) -> tuple[float, float]:
    """Compute Sim(3)-aligned ATE RMSE and best-fit scale."""
    mu_e = centers_est.mean(axis=0)
    mu_g = centers_gt.mean(axis=0)
    X = centers_est - mu_e
    Y = centers_gt - mu_g
    var_x = float(np.sum(X**2))
    H = X.T @ Y
    U, D, Vt = np.linalg.svd(H)
    S = np.eye(3)
    if np.linalg.det(Vt.T @ U.T) < 0:
        S[2, 2] = -1.0
    R = Vt.T @ S @ U.T
    scale = float(np.sum(D * np.diag(S)) / max(var_x, 1e-12))
    t = mu_g - scale * (R @ mu_e)
    aligned = scale * (R @ centers_est.T).T + t
    rmse = float(np.sqrt(np.mean(np.sum((aligned - centers_gt) ** 2, axis=1))))
    return rmse, scale


def _build_synthetic_vi_stage_inputs(
    *,
    num_frames: int = 20,
    dt_frame: float = 0.1,
    include_unscaled_depths: bool = False,
    raw_depth_scale: float = 2.5,
    outlier_consecutive_pair_idx: int | None = 8,
    seed: int = 7,
):
    """Build in-memory MappingStageInputs + IMU edges for end-to-end Mapper testing."""
    rng = np.random.default_rng(seed)
    imu_rate = 200.0
    dt_imu = 1.0 / imu_rate
    steps_per_interval = int(round(dt_frame / dt_imu))
    gravity_mag = 9.81

    axis_w = np.array([0.45, -0.65, 0.61], dtype=np.float64)
    axis_w /= np.linalg.norm(axis_w)
    R_world = _so3_exp(axis_w * np.radians(75.0))
    g_world_true = R_world @ np.array([0.0, 0.0, -gravity_mag], dtype=np.float64)
    g_dir_true = g_world_true / gravity_mag

    bg_true = np.array([0.022, -0.015, 0.011], dtype=np.float64)
    ba_true = np.array([-0.06, 0.05, -0.03], dtype=np.float64)

    R_IC_true = _so3_exp(np.array([0.08, -0.06, 0.10], dtype=np.float64))
    t_IC_true = np.array([0.03, -0.02, 0.01], dtype=np.float64)
    R_CI_true = R_IC_true.T
    t_CI_true = -(R_CI_true @ t_IC_true)
    imu_from_cam_true = pycolmap.Rigid3d(pycolmap.Rotation3d(R_IC_true), t_IC_true)

    def body_kinematics(t: float):
        p_c = np.array(
            [
                1.8 * np.cos(0.9 * t) + 0.3 * t,
                1.4 * np.sin(1.2 * t),
                0.6 * np.sin(1.6 * t) + 0.04 * t * t,
            ],
            dtype=np.float64,
        )
        v_c = np.array(
            [
                -1.8 * 0.9 * np.sin(0.9 * t) + 0.3,
                1.4 * 1.2 * np.cos(1.2 * t),
                0.6 * 1.6 * np.cos(1.6 * t) + 0.08 * t,
            ],
            dtype=np.float64,
        )
        a_c = np.array(
            [
                -1.8 * 0.81 * np.cos(0.9 * t),
                -1.4 * 1.44 * np.sin(1.2 * t),
                -0.6 * 2.56 * np.sin(1.6 * t) + 0.08,
            ],
            dtype=np.float64,
        )
        psi = np.array(
            [0.30 * np.sin(1.1 * t), 0.25 * np.cos(0.8 * t), 0.45 * t],
            dtype=np.float64,
        )
        R_WB = R_world @ _so3_exp(psi)
        return R_world @ p_c, R_world @ v_c, R_world @ a_c, R_WB

    def body_omega(t: float) -> np.ndarray:
        from scipy.spatial.transform import Rotation

        eps = 1e-6
        _, _, _, R_m = body_kinematics(t - eps)
        _, _, _, R_p = body_kinematics(t + eps)
        dR = R_m.T @ R_p
        return Rotation.from_matrix(dR).as_rotvec() / (2.0 * eps)

    width, height = 640, 480
    fx, fy, cx, cy = 520.0, 520.0, 320.0, 240.0
    K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
    cam = pycolmap.Camera(
        model="PINHOLE",
        width=width,
        height=height,
        params=[fx, fy, cx, cy],
        camera_id=1,
    )
    cam.has_prior_focal_length = True

    # Generate 3D points surrounding the trajectory so every frame sees >= 35 points.
    t_samples = np.linspace(0.0, (num_frames - 1) * dt_frame, num_frames)
    centers_gt = np.zeros((num_frames, 3), dtype=np.float64)
    vels_gt = np.zeros((num_frames, 3), dtype=np.float64)
    R_CW_gt = []
    q_iori_list = []
    for idx, t in enumerate(t_samples):
        p_B, v_B, _, R_WB = body_kinematics(float(t))
        R_CW_phys = R_CI_true @ R_WB.T
        c_cam = p_B + R_WB @ t_CI_true
        R_iori = _so3_exp(rng.normal(0.0, 0.03, size=3))
        R_CW_stab = R_iori @ R_CW_phys
        centers_gt[idx] = c_cam
        vels_gt[idx] = v_B
        R_CW_gt.append(R_CW_stab)
        q_iori_list.append(pycolmap.Rotation3d(R_iori).quat)

    # Create 80 3D points in front of the cameras.
    mean_center = centers_gt.mean(axis=0)
    mean_forward = np.mean([R[2, :] for R in R_CW_gt], axis=0)
    mean_forward /= np.linalg.norm(mean_forward)
    points3D_gt = (
        mean_center[None, :]
        + mean_forward[None, :] * rng.uniform(5.0, 9.0, size=(80, 1))
        + rng.uniform(-2.0, 2.0, size=(80, 3))
    )

    rec = pycolmap.Reconstruction()
    problem = native.MappingProblem()
    rec.add_camera_with_trivial_rig(cam)
    problem.add_camera(camera_record_from_pycolmap(cam))

    image_keypoints = {}
    image_point_depths = {}
    for idx in range(num_frames):
        image_id = idx + 1
        R_cw = R_CW_gt[idx]
        t_cw = -(R_cw @ centers_gt[idx])
        pts_cam = (R_cw @ points3D_gt.T).T + t_cw[None, :]
        uv = (K @ pts_cam.T).T
        uv = uv[:, :2] / uv[:, 2:3]
        uv += rng.normal(0.0, 0.15, size=uv.shape)
        image_keypoints[image_id] = uv
        image_point_depths[image_id] = pts_cam[:, 2].copy()

        gimg = pycolmap.Image(
            name=f"frame_{image_id:04d}.png",
            camera_id=1,
            image_id=image_id,
            keypoints=uv,
        )
        rec.add_image_with_trivial_frame(gimg, pycolmap.Rigid3d())
        img_rec = image_record_from_pycolmap(rec.image(image_id), uv)
        if include_unscaled_depths:
            raw_d = np.clip(pts_cam[:, 2] / raw_depth_scale, 0.1, None)
            img_rec.depth_values = raw_d.astype(np.float64)
            img_rec.depth_stddevs = (0.02 * raw_d).astype(np.float64)
            img_rec.depth_validity = np.ones(len(raw_d), dtype=np.uint8)
        problem.add_image(img_rec)

    consecutive_pair_ids = []
    num_pts = len(points3D_gt)
    all_matches = np.column_stack([np.arange(num_pts, dtype=np.uint32)] * 2)
    for idx1 in range(num_frames):
        for step in (1, 2, 3):
            idx2 = idx1 + step
            if idx2 >= num_frames:
                continue
            id1, id2 = idx1 + 1, idx2 + 1
            pid = int(pycolmap.image_pair_to_pair_id(id1, id2))
            if step == 1:
                consecutive_pair_ids.append(pid)

            R_21 = R_CW_gt[idx2] @ R_CW_gt[idx1].T
            c1_in_2 = R_CW_gt[idx2] @ (centers_gt[idx1] - centers_gt[idx2])
            t_21 = -c1_in_2
            if np.linalg.norm(t_21) > 1e-9:
                t_21_dir = t_21 / np.linalg.norm(t_21)
            else:
                t_21_dir = np.array([1.0, 0.0, 0.0], dtype=np.float64)

            if step == 1 and outlier_consecutive_pair_idx is not None and idx1 == outlier_consecutive_pair_idx:
                # Corrupt one consecutive edge rotation by 40 deg to test I-RA outlier pruning/salvaging.
                R_21 = _so3_exp(np.array([0.0, np.radians(40.0), 0.0])) @ R_21

            tvg = pycolmap.TwoViewGeometry()
            tvg.config = pycolmap.TwoViewGeometryConfiguration.CALIBRATED
            tvg.cam2_from_cam1 = pycolmap.Rigid3d(pycolmap.Rotation3d(R_21), t_21_dir)
            tvg.inlier_matches = all_matches

            pair_rec = pair_record_from_pycolmap(pid, id1, id2, tvg, all_matches)
            pair_rec.inlier_indices = np.arange(num_pts, dtype=np.int32)
            pair_rec.are_loop_closure = np.zeros(num_pts, dtype=np.uint8)
            pair_rec.geometry.configuration = int(pycolmap.TwoViewGeometryConfiguration.CALIBRATED)
            pair_rec.geometry.cam2_from_cam1 = pose_record_from_pycolmap(tvg.cam2_from_cam1)
            problem.add_pair(pair_rec)

    # Pre-integrate IMU measurements between consecutive frames at ZERO nominal bias.
    imu_options = pycolmap.ImuPreintegrationOptions()
    imu_options.method = pycolmap.ImuIntegrationMethod.RK4
    imu_calib = pycolmap.ImuCalibration()
    imu_calib.gravity_magnitude = gravity_mag
    imu_calib.gyro_noise_density = 1e-4
    imu_calib.accel_noise_density = 1e-3
    imu_calib.bias_gyro_random_walk_sigma = 1e-5
    imu_calib.bias_accel_random_walk_sigma = 1e-4
    imu_calib.imu_rate = imu_rate

    integrators = []
    imu_edges = []
    for idx in range(num_frames - 1):
        id1, id2 = idx + 1, idx + 2
        t0 = float(t_samples[idx])
        t1 = float(t_samples[idx + 1])
        t0_ns = pycolmap.timestamp_from_seconds(t0)
        t1_ns = pycolmap.timestamp_from_seconds(t1)
        integrator = pycolmap.ImuPreintegrator(imu_options, imu_calib, t0_ns, t1_ns)
        ms = pycolmap.ImuMeasurements()
        for step_i in range(steps_per_interval + 1):
            t_cur = t0 + step_i * dt_imu
            t_ns = pycolmap.timestamp_from_seconds(t_cur)
            _, _, a_W, R_WB = body_kinematics(t_cur)
            omega_B = body_omega(t_cur) + bg_true
            accel_B = R_WB.T @ (a_W - g_world_true) + ba_true
            ms.insert(
                pycolmap.ImuMeasurement(
                    timestamp=t_ns,
                    accel=accel_B,
                    gyro=omega_B,
                )
            )
        integrator.integrate(ms)
        integrators.append(integrator)

        edge = native.ImuEdgeRecord()
        edge.image_id1 = id1
        edge.image_id2 = id2
        edge.data = integrator.extract()
        edge.set_integrator(integrator)
        edge.q_iori_1_xyzw = q_iori_list[idx]
        edge.q_iori_2_xyzw = q_iori_list[idx + 1]
        imu_edges.append(edge)

    state = SolveState(rec, problem)
    state.set_imu_data(imu_edges, imu_from_cam=imu_from_cam_true)
    stage_inputs = MappingStageInputs(
        solve_state=state,
        consecutive_pair_ids=consecutive_pair_ids,
        sequence_id_to_index={idx + 1: idx for idx in range(num_frames)},
        vgc_exclusion_ids=set(),
    )
    return {
        "stage_inputs": stage_inputs,
        "integrators": integrators,
        "centers_gt": centers_gt,
        "vels_gt": vels_gt,
        "R_CW_gt": R_CW_gt,
        "g_dir_true": g_dir_true,
        "bg_true": bg_true,
        "ba_true": ba_true,
        "imu_from_cam_true": imu_from_cam_true,
    }


def test_imu_edge_reintegrate_binding():
    scene = _build_synthetic_vi_stage_inputs(num_frames=6, outlier_consecutive_pair_idx=None)
    state = scene["stage_inputs"].solve_state
    edge0 = state.imu_edges[0]
    np.testing.assert_allclose(edge0.data.biases, np.zeros(6), atol=1e-12)

    new_biases = np.r_[scene["bg_true"], scene["ba_true"]]
    edge0.reintegrate(new_biases)
    np.testing.assert_allclose(edge0.data.biases, new_biases, atol=1e-12)


def test_end_to_end_pipeline_monocular_stage_ablation(tmp_path: Path):
    """Verify monotonic improvement across Vision-Only -> +I-RA -> +I-RA+I-GP -> +I-RA+I-GP+VI-BA."""
    results = {}
    for mode in ("vision_only", "ira_only", "ira_igp", "ira_igp_viba"):
        scene = _build_synthetic_vi_stage_inputs(
            num_frames=20,
            include_unscaled_depths=False,
            outlier_consecutive_pair_idx=8,
            seed=11,
        )
        use_ra_imu = mode in ("ira_only", "ira_igp", "ira_igp_viba")
        use_gp_imu = mode in ("ira_igp", "ira_igp_viba")
        use_ba_imu = mode == "ira_igp_viba"

        conf = MapperOptions(
            calibration={"vgc_enabled": False, "optimize_intrinsics": False, "vgc_focal_prior": False},
            mdrp={"enabled": False},
            ra={
                "use_imu": use_ra_imu,
                "refine_gyro_bias": True,
                "invalidate_outlier_pairs": True,
                "salvage_outlier_translations": False,
                "imu_max_rotation_error_deg": 5.0,
            },
            depth_consistency={"enabled": False},
            gp={
                "common": {
                    "use_imu": use_gp_imu,
                    "use_linear_gravity_warm_start": True,
                },
                "first_pass": {"max_iterations": 40},
                "second_pass": {"enabled": True, "max_iterations": 40},
            },
            ba={
                "normal": {"iterations": 1},
                "annealing": {"iterations": 1},
                "focal_prior": {"enabled": False},
                "post_annealing_point_refinement": False,
                "imu": {
                    "use_imu": use_ba_imu,
                    "refine_imu_from_cam_rotation": False,
                },
            },
        )

        stage_metrics = {}

        def on_stage_complete(stage_name: str, solve_state: SolveState):
            if stage_name in ("gp", "ba"):
                rec = solve_state.reconstruction
                reg_ids = sorted(rec.reg_image_ids())
                centers = np.array([rec.image(iid).projection_center() for iid in reg_ids])
                gt_sub = scene["centers_gt"][[iid - 1 for iid in reg_ids]]
                se3_rmse = _se3_ate_rmse(centers, gt_sub)
                sim3_rmse, sim3_scale = _sim3_ate_rmse_and_scale(centers, gt_sub)
                stage_metrics[stage_name] = {
                    "num_reg": len(reg_ids),
                    "se3_rmse": se3_rmse,
                    "sim3_rmse": sim3_rmse,
                    "sim3_scale": sim3_scale,
                }

        mapper = Mapper(
            conf=conf,
            mapper_inputs=None,
            sfm_outputs_dir=tmp_path / mode,
        )
        mapper.solve_stage_inputs(scene["stage_inputs"], on_stage_complete=on_stage_complete)
        results[mode] = (stage_metrics, mapper.last_solve_state, scene)

    # 1. Verify monotonic improvement in Sim(3) and SE(3) ATE across the modes:
    m_vis = results["vision_only"][0]["ba"]
    m_ira = results["ira_only"][0]["ba"]
    m_igp_gp = results["ira_igp"][0]["gp"]
    m_viba = results["ira_igp_viba"][0]["ba"]

    assert m_ira["sim3_rmse"] < m_vis["sim3_rmse"]
    assert m_igp_gp["se3_rmse"] < m_ira["se3_rmse"]
    assert m_viba["se3_rmse"] <= m_igp_gp["se3_rmse"] + 1e-3
    assert m_viba["se3_rmse"] < 0.03
    assert abs(m_viba["sim3_scale"] - 1.0) < 0.015

    # 2. Verify physical parameters (gravity, gyro bias, accel bias) in full VI pipeline:
    final_state = results["ira_igp_viba"][1]
    scene_ref = results["ira_igp_viba"][2]
    R_CW_est_0 = final_state.reconstruction.image(1).cam_from_world().rotation.matrix()
    g_cam0_est = R_CW_est_0 @ final_state.gravity_direction
    g_cam0_gt = scene_ref["R_CW_gt"][0] @ scene_ref["g_dir_true"]
    grav_err_deg = _angle_deg(g_cam0_est, g_cam0_gt)
    assert grav_err_deg < 0.5

    bg_est = np.mean([s.bias_gyro for s in final_state.imu_states.values()], axis=0)
    ba_est = np.mean([s.bias_accel for s in final_state.imu_states.values()], axis=0)
    assert float(np.linalg.norm(bg_est - scene_ref["bg_true"])) < 0.005
    assert float(np.linalg.norm(ba_est - scene_ref["ba_true"])) < 0.05


def test_end_to_end_pipeline_with_unscaled_monocular_depths(tmp_path: Path):
    """Verify that unscaled monocular depth priors do not pull metric VI-BA back to raw depth scale."""
    raw_depth_scale = 2.5
    scene = _build_synthetic_vi_stage_inputs(
        num_frames=16,
        include_unscaled_depths=True,
        raw_depth_scale=raw_depth_scale,
        outlier_consecutive_pair_idx=None,
        seed=19,
    )
    # Perturb initial R_BC by 0.3 deg to test R_BC refinement in final BA.
    R_IC_true = scene["imu_from_cam_true"].rotation.matrix()
    R_IC_pert = _so3_exp(np.radians([0.2, -0.15, 0.15])) @ R_IC_true
    imu_from_cam_pert = pycolmap.Rigid3d(
        pycolmap.Rotation3d(R_IC_pert),
        scene["imu_from_cam_true"].translation,
    )
    scene["stage_inputs"].solve_state.set_imu_data(
        scene["stage_inputs"].solve_state.imu_edges,
        imu_from_cam=imu_from_cam_pert,
    )

    conf = MapperOptions(
        calibration={"vgc_enabled": False, "optimize_intrinsics": True, "vgc_focal_prior": False},
        mdrp={"enabled": False},
        ra={"use_imu": True, "refine_gyro_bias": True},
        depth_consistency={"enabled": False},
        gp={
            "common": {"use_imu": True, "use_linear_gravity_warm_start": True},
            "first_pass": {"max_iterations": 40},
            "second_pass": {"enabled": True, "max_iterations": 40},
        },
        ba={
            "normal": {"iterations": 1},
            "annealing": {"iterations": 1},
            "focal_prior": {"enabled": False},
            "post_annealing_point_refinement": True,
            "imu": {
                "use_imu": True,
                "refine_imu_from_cam_rotation": True,
            },
        },
    )

    mapper = Mapper(
        conf=conf,
        mapper_inputs=None,
        sfm_outputs_dir=tmp_path / "with_depths",
    )
    rec = mapper.solve_stage_inputs(scene["stage_inputs"])
    reg_ids = sorted(rec.reg_image_ids())
    assert len(reg_ids) == 16

    centers = np.array([rec.image(iid).projection_center() for iid in reg_ids])
    se3_rmse = _se3_ate_rmse(centers, scene["centers_gt"])
    _, sim3_scale = _sim3_ate_rmse_and_scale(centers, scene["centers_gt"])

    # Reconstruction must stay in metric meters (sim3_scale ≈ 1.0, NOT 1 / 2.5 = 0.4)!
    assert abs(sim3_scale - 1.0) < 0.02
    assert se3_rmse < 0.03

    # Verify R_BC extrinsics moved closer to ground truth during final BA.
    R_IC_opt = pose_record_to_pycolmap(mapper.last_solve_state.imu_from_cam).rotation.matrix()
    err_init_deg = float(np.degrees(pycolmap.Rotation3d(R_IC_pert.T @ R_IC_true).angle()))
    err_opt_deg = float(np.degrees(pycolmap.Rotation3d(R_IC_opt.T @ R_IC_true).angle()))
    assert err_opt_deg < err_init_deg
