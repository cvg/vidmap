import numpy as np
import pycolmap
import vidmap_native._core as native
from scipy.spatial.transform import Rotation


def _pair_id(image_id1: int, image_id2: int) -> int:
    return min(image_id1, image_id2) * 2147483647 + max(image_id1, image_id2)


def _geodesic_deg(R1: np.ndarray, R2: np.ndarray) -> float:
    return float(np.degrees(Rotation.from_matrix(R1.T @ R2).magnitude()))


def _simulate_sequence_with_contiguous_outliers(
    num_frames: int = 100,
    corrupted_edge_range: tuple[int, int] = (30, 60),
    outlier_drift_deg_per_frame: float = 10.0,
):
    dt_frame = 0.1
    imu_rate = 200
    dt_imu = 1.0 / imu_rate

    true_bg = np.array([0.025, -0.018, 0.012], dtype=np.float64)
    true_ba = np.array([-0.08, 0.06, -0.04], dtype=np.float64)
    g_W = np.array([0.0, 0.0, -9.81], dtype=np.float64)

    R_BC = Rotation.from_euler("xyz", [8.0, -5.0, 12.0], degrees=True).as_matrix()
    p_BC = np.array([0.03, -0.02, 0.01], dtype=np.float64)

    def body_pose(t: float):
        pos = np.array(
            [
                2.0 * np.cos(0.7 * t) + 0.3 * t,
                1.5 * np.sin(1.1 * t),
                0.5 * np.sin(1.5 * t),
            ],
            dtype=np.float64,
        )
        vel = np.array(
            [
                -1.4 * np.sin(0.7 * t) + 0.3,
                1.65 * np.cos(1.1 * t),
                0.75 * np.cos(1.5 * t),
            ],
            dtype=np.float64,
        )
        acc = np.array(
            [
                -0.98 * np.cos(0.7 * t),
                -1.815 * np.sin(1.1 * t),
                -1.125 * np.sin(1.5 * t),
            ],
            dtype=np.float64,
        )
        rpy = np.array(
            [
                0.25 * np.sin(0.9 * t),
                0.20 * np.cos(0.6 * t),
                0.45 * t + 0.15 * np.sin(0.5 * t),
            ],
            dtype=np.float64,
        )
        R_WB = Rotation.from_euler("xyz", rpy).as_matrix()
        return pos, vel, acc, R_WB

    imu_options = pycolmap.ImuPreintegrationOptions()
    imu_options.method = pycolmap.ImuIntegrationMethod.RK4
    imu_calib = pycolmap.ImuCalibration()
    imu_calib.gyro_noise_density = 1e-4
    imu_calib.accel_noise_density = 1e-3
    imu_calib.bias_gyro_random_walk_sigma = 1e-5
    imu_calib.bias_accel_random_walk_sigma = 1e-4
    imu_calib.gravity_magnitude = 9.81
    imu_calib.imu_rate = float(imu_rate)

    times = [i * dt_frame for i in range(num_frames)]
    R_iori_list = []
    gt_R_cw_list = []
    gt_c_w_list = []

    for i, t in enumerate(times):
        pos_B, _, _, R_WB = body_pose(t)
        R_WC_phys = R_WB @ R_BC
        c_W = pos_B + R_WB @ p_BC
        R_iori = Rotation.from_euler(
            "xyz",
            [1.5 * np.sin(0.4 * i), -1.2 * np.cos(0.3 * i), 0.8 * np.sin(0.5 * i)],
            degrees=True,
        ).as_matrix()
        R_CW = R_iori @ R_WC_phys.T
        R_iori_list.append(R_iori)
        gt_R_cw_list.append(R_CW)
        gt_c_w_list.append(c_W)

    imu_edges = []
    integrators = []
    for i in range(num_frames - 1):
        t0, t1 = times[i], times[i + 1]
        t0_ns = pycolmap.timestamp_from_seconds(t0)
        t1_ns = pycolmap.timestamp_from_seconds(t1)
        integ = pycolmap.ImuPreintegrator(imu_options, imu_calib, t0_ns, t1_ns)
        ms = pycolmap.ImuMeasurements()
        num_steps = int(round((t1 - t0) / dt_imu))
        for step in range(num_steps + 1):
            ts = t0 + step * dt_imu
            ts_ns = pycolmap.timestamp_from_seconds(ts)
            eps = 1e-6
            _, _, acc_W, R_WB = body_pose(ts)
            _, _, _, R_WB_minus = body_pose(ts - eps)
            _, _, _, R_WB_plus = body_pose(ts + eps)
            dR = R_WB_minus.T @ R_WB_plus
            omega_B = Rotation.from_matrix(dR).as_rotvec() / (2.0 * eps)
            specific_force_B = R_WB.T @ (acc_W - g_W)
            ms.insert(
                pycolmap.ImuMeasurement(
                    timestamp=ts_ns,
                    accel=specific_force_B + true_ba,
                    gyro=omega_B + true_bg,
                )
            )
        integ.integrate(ms)
        edge = native.ImuEdgeRecord()
        edge.image_id1 = i + 1
        edge.image_id2 = i + 2
        edge.data = integ.extract()
        edge.set_integrator(integ)
        edge.q_iori_1_xyzw = Rotation.from_matrix(R_iori_list[i]).as_quat()
        edge.q_iori_2_xyzw = Rotation.from_matrix(R_iori_list[i + 1]).as_quat()
        integrators.append(integ)
        imu_edges.append(edge)

    problem = native.MappingProblem()
    camera = native.CameraRecord()
    camera.camera_id = 1
    camera.model_id = int(pycolmap.CameraModelId.PINHOLE)
    camera.width = 640
    camera.height = 480
    camera.params = np.array([500.0, 500.0, 320.0, 240.0], dtype=np.float64)
    problem.add_camera(camera)

    image_ids = list(range(1, num_frames + 1))
    for i, image_id in enumerate(image_ids):
        image = native.ImageRecord()
        image.image_id = image_id
        image.frame_id = image_id
        image.camera_id = 1
        image.name = f"frame_{image_id:04d}.jpg"
        image.keypoints = np.zeros((20, 2), dtype=np.float64)
        image.pose.has_pose = True
        if i == 0:
            image.pose.rotation_xyzw = Rotation.from_matrix(gt_R_cw_list[0]).as_quat()
        else:
            image.pose.rotation_xyzw = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
        image.pose.translation = -(gt_R_cw_list[i] @ gt_c_w_list[i])
        problem.add_image(image)

    pair_ids = []
    corrupted_pair_ids = set()
    drift_rot = Rotation.from_euler(
        "xyz",
        [
            outlier_drift_deg_per_frame * 0.6,
            -outlier_drift_deg_per_frame * 0.8,
            outlier_drift_deg_per_frame * 0.2,
        ],
        degrees=True,
    ).as_matrix()

    for i in range(num_frames - 1):
        id1 = i + 1
        id2 = i + 2
        pid = _pair_id(id1, id2)
        pair_ids.append(pid)
        R_2_from_1 = gt_R_cw_list[i + 1] @ gt_R_cw_list[i].T
        if corrupted_edge_range[0] <= i < corrupted_edge_range[1]:
            R_2_from_1 = drift_rot @ R_2_from_1
            corrupted_pair_ids.add(pid)

        pair = native.PairRecord()
        pair.pair_id = pid
        pair.image_id1 = id1
        pair.image_id2 = id2
        pair.is_valid = True
        pair.all_matches = np.column_stack((np.arange(20), np.arange(20))).astype(np.uint32)
        pair.inlier_indices = np.arange(20, dtype=np.int32)
        pair.are_loop_closure = np.zeros(20, dtype=np.uint8)
        pair.geometry.cam2_from_cam1.has_pose = True
        pair.geometry.cam2_from_cam1.rotation_xyzw = Rotation.from_matrix(R_2_from_1).as_quat()
        problem.add_pair(pair)

    imu_from_cam = native.PoseRecord()
    imu_from_cam.has_pose = True
    imu_from_cam.rotation_xyzw = Rotation.from_matrix(R_BC).as_quat()
    imu_from_cam.translation = p_BC

    return (
        problem,
        image_ids,
        pair_ids,
        corrupted_pair_ids,
        imu_edges,
        integrators,
        imu_from_cam,
        gt_R_cw_list,
        true_bg,
    )


def test_vi_rotation_averaging_contiguous_outliers_and_bias_recovery():
    (
        problem,
        image_ids,
        pair_ids,
        corrupted_pair_ids,
        imu_edges,
        _integrators,
        imu_from_cam,
        gt_R_cw_list,
        true_bg,
    ) = _simulate_sequence_with_contiguous_outliers(
        num_frames=100,
        corrupted_edge_range=(30, 60),
        outlier_drift_deg_per_frame=10.0,
    )

    options = native.RotationAveragingOptions()
    options.use_imu = True
    options.imu_from_cam = imu_from_cam
    options.max_rotation_error_deg = 3.0
    options.invalidate_outlier_pairs = True

    result = native.run_video_rotation_averaging(
        options,
        image_ids,
        pair_ids,
        problem,
        imu_edges=imu_edges,
    )

    assert result.success
    assert len(result.registered_image_ids) == 100
    assert set(result.outlier_pair_ids) == corrupted_pair_ids

    for pid in pair_ids:
        if pid in corrupted_pair_ids:
            assert not problem.pair(pid).is_valid
        else:
            assert problem.pair(pid).is_valid

    est_R_cw_0 = Rotation.from_quat(problem.image(image_ids[0]).pose.rotation_xyzw).as_matrix()
    R_align = est_R_cw_0.T @ gt_R_cw_list[0]

    rot_errors_deg = []
    bg_errors = []
    for i, image_id in enumerate(image_ids):
        est_q_xyzw = problem.image(image_id).pose.rotation_xyzw
        est_R_cw = Rotation.from_quat(est_q_xyzw).as_matrix() @ R_align
        rot_errors_deg.append(_geodesic_deg(est_R_cw, gt_R_cw_list[i]))
        est_bg = result.imu_states[image_id].bias_gyro
        bg_errors.append(np.linalg.norm(est_bg - true_bg))

    assert max(rot_errors_deg) < 0.1, f"Max rotation error {max(rot_errors_deg):.4f} deg >= 0.1 deg"
    assert max(bg_errors) < 1e-4, f"Max gyro bias error {max(bg_errors):.6f} rad/s >= 1e-4 rad/s"

    g_dir_true = np.array([0.0, 0.0, -1.0])
    est_g_dir_aligned = R_align.T @ result.initial_gravity_direction
    grav_angle_deg = np.degrees(np.arccos(np.clip(np.dot(est_g_dir_aligned, g_dir_true), -1.0, 1.0)))
    assert grav_angle_deg < 3.0


def test_vi_rotation_averaging_bridges_complete_visual_gap():
    (
        problem,
        image_ids,
        pair_ids,
        _corrupted_pair_ids,
        imu_edges,
        _integrators,
        imu_from_cam,
        gt_R_cw_list,
        true_bg,
    ) = _simulate_sequence_with_contiguous_outliers(
        num_frames=50,
        corrupted_edge_range=(0, 0),
        outlier_drift_deg_per_frame=0.0,
    )

    # Mark visual edges 15..25 invalid (simulating complete visual tracking loss).
    for idx in range(15, 25):
        pair = problem.pair(pair_ids[idx])
        pair.is_valid = False
        problem.update_pair(pair)

    options = native.RotationAveragingOptions()
    options.use_imu = True
    options.imu_from_cam = imu_from_cam
    options.max_rotation_error_deg = 3.0

    result = native.run_video_rotation_averaging(
        options,
        image_ids,
        pair_ids,
        problem,
        imu_edges=imu_edges,
    )

    assert result.success
    assert len(result.registered_image_ids) == 50
    assert len(result.outlier_pair_ids) == 0

    est_R_cw_0 = Rotation.from_quat(problem.image(image_ids[0]).pose.rotation_xyzw).as_matrix()
    R_align = est_R_cw_0.T @ gt_R_cw_list[0]

    for i, image_id in enumerate(image_ids):
        est_q_xyzw = problem.image(image_id).pose.rotation_xyzw
        est_R_cw = Rotation.from_quat(est_q_xyzw).as_matrix() @ R_align
        assert _geodesic_deg(est_R_cw, gt_R_cw_list[i]) < 0.1
        assert np.linalg.norm(result.imu_states[image_id].bias_gyro - true_bg) < 1e-4


def test_vi_rotation_averaging_salvage_outlier_translations():
    (
        problem,
        image_ids,
        pair_ids,
        corrupted_pair_ids,
        imu_edges,
        _integrators,
        imu_from_cam,
        gt_R_cw_list,
        _true_bg,
    ) = _simulate_sequence_with_contiguous_outliers(
        num_frames=30,
        corrupted_edge_range=(10, 15),
        outlier_drift_deg_per_frame=10.0,
    )

    rng = np.random.default_rng(123)
    pts_w = rng.uniform(-3.0, 3.0, size=(80, 3))
    pts_w[:, 2] += 10.0

    # Populate valid bearings on frames 10..16 so pair (11, 12) has 60 static
    # background matches and 20 moving-object matches, while pair (12, 13) has
    # only 10 static matches (< min_inlier_ratio).
    for idx in range(10, 16):
        iid = image_ids[idx]
        im = problem.image(iid)
        R_cw = gt_R_cw_list[idx]
        t_cw = np.asarray(im.pose.translation)
        pts_c = (R_cw @ pts_w.T).T + t_cw
        bearings = pts_c / np.linalg.norm(pts_c, axis=1, keepdims=True)
        if idx == 11:
            # Corrupt 20 of 80 bearings on image 12 (pair 11-12 has 60/80 = 75% inliers)
            rand_vecs = rng.normal(size=(20, 3))
            rand_vecs /= np.linalg.norm(rand_vecs, axis=1, keepdims=True)
            bearings[60:] = rand_vecs
        elif idx == 12:
            # Corrupt 70 of 80 bearings on image 13 (pair 12-13 has 10/80 = 12.5% inliers)
            rand_vecs = rng.normal(size=(70, 3))
            rand_vecs /= np.linalg.norm(rand_vecs, axis=1, keepdims=True)
            bearings[10:] = rand_vecs
        im.keypoints = np.zeros((80, 2), dtype=np.float64)
        im.bearings = bearings
        problem.update_image(im)

    for idx in (10, 11):
        pid = pair_ids[idx]
        pair = problem.pair(pid)
        pair.all_matches = np.column_stack((np.arange(80), np.arange(80))).astype(np.uint32)
        pair.inlier_indices = np.arange(80, dtype=np.int32)
        pair.are_loop_closure = np.zeros(80, dtype=np.uint8)
        problem.update_pair(pair)

    options = native.RotationAveragingOptions()
    options.use_imu = True
    options.imu_from_cam = imu_from_cam
    options.max_rotation_error_deg = 3.0
    options.invalidate_outlier_pairs = True
    options.salvage_outlier_translations = True
    options.salvage_min_inliers = 40
    options.salvage_min_inlier_ratio = 0.50

    result = native.run_video_rotation_averaging(
        options,
        image_ids,
        pair_ids,
        problem,
        imu_edges=imu_edges,
    )

    assert result.success
    assert set(result.outlier_pair_ids) == corrupted_pair_ids
    salvaged_pid = pair_ids[10]
    unsalvaged_pid = pair_ids[11]
    assert salvaged_pid in set(result.salvaged_pair_ids)
    assert unsalvaged_pid not in set(result.salvaged_pair_ids)

    salvaged_pair = problem.pair(salvaged_pid)
    assert salvaged_pair.is_valid
    assert set(range(60)).issubset(set(salvaged_pair.inlier_indices))
    assert len(salvaged_pair.inlier_indices) <= 65
    assert not problem.pair(unsalvaged_pid).is_valid

    # Verify the salvaged relative translation direction matches ground truth
    im1 = problem.image(salvaged_pair.image_id1)
    im2 = problem.image(salvaged_pair.image_id2)
    R2_gt = gt_R_cw_list[11]
    c1_gt = -(gt_R_cw_list[10].T @ np.asarray(im1.pose.translation))
    c2_gt = -(gt_R_cw_list[11].T @ np.asarray(im2.pose.translation))
    t_21_gt = R2_gt @ (c1_gt - c2_gt)
    t_21_gt /= np.linalg.norm(t_21_gt)
    t_21_est = np.asarray(salvaged_pair.geometry.cam2_from_cam1.translation)
    cos_angle = float(np.clip(np.dot(t_21_est, t_21_gt), -1.0, 1.0))
    assert np.degrees(np.arccos(cos_angle)) < 0.5
