"""Tests for loading absolute location priors from the generic .npz interface."""

from types import SimpleNamespace

import numpy as np
import pyceres
import pycolmap
from scipy.spatial.transform import Rotation

from vidmap.mapper.location_priors import LocationAnchorPrior, LocationPriorSet, load_location_priors
from vidmap.mapper.native.extension import native
from vidmap.mapper.native.state import SolveState
from vidmap.mapper.options.location_priors import LocationPriorOptions
from vidmap.mapper.options.view_graph import RAOptions
from vidmap.mapper.stages.rotation_averaging import RotationAverager


def _write_priors(path, image_names, confidence, num_matches):
    count = len(image_names)
    indptr = np.concatenate([[0], np.cumsum(num_matches)]).astype(np.int32)
    num_points = int(indptr[-1])
    np.savez(
        path,
        image_names=np.array(image_names),
        confidence=np.asarray(confidence, dtype=np.float32),
        R_cam_from_world=np.tile(np.eye(3), (count, 1, 1)),
        cov_cam_from_world=np.tile(np.eye(6) * 1e-4, (count, 1, 1)),
        points3D=np.arange(num_points * 3, dtype=np.float64).reshape(num_points, 3),
        match_indptr=indptr,
        match_point_indices=np.arange(num_points, dtype=np.int32),
        match_uv_norm=np.full((num_points, 2), 0.5, dtype=np.float32),
    )


def _solve_state(names):
    images = {i: SimpleNamespace(name=name) for i, name in enumerate(names, start=1)}
    return SimpleNamespace(reconstruction=SimpleNamespace(images=images))


def test_load_location_priors_matching_and_filtering(tmp_path):
    path = tmp_path / "priors.npz"
    _write_priors(
        path,
        # exact name, timestamp within 1 ms, low confidence, too few matches, unknown frame
        [
            "0000000001.000000000.jpg",
            "0000000002.000400000.jpg",
            "0000000003.000000000.jpg",
            "0000000004.000000000.jpg",
            "0000000099.000000000.jpg",
        ],
        confidence=[0.9, 0.8, 0.01, 0.9, 0.9],
        num_matches=[5, 6, 5, 2, 5],
    )
    state = _solve_state(
        [
            "0000000001.000000000.jpg",
            "0000000002.000000000.jpg",
            "0000000003.000000000.jpg",
            "0000000004.000000000.jpg",
        ]
    )
    priors = load_location_priors(LocationPriorOptions(enabled=True, path=str(path)), state)

    assert sorted(priors.anchors) == [1, 2]
    assert priors.anchors[2].image_name == "0000000002.000400000.jpg"
    assert len(priors.anchors[2].prior_point_ids) == 6
    np.testing.assert_array_equal(priors.anchors[2].points3D_xyz, priors.mapped_points_xyz[5:11])
    assert priors.anchors[1].cov_rot_cam_from_world.shape == (3, 3)


def test_load_location_priors_disabled(tmp_path):
    assert load_location_priors(LocationPriorOptions(enabled=False, path=None), _solve_state([])) is None


def _anchor(image_id, R_cam_from_world, points3D=np.zeros((0, 3)), uv_norm=np.zeros((0, 2))):
    return LocationAnchorPrior(
        image_id=image_id,
        image_name=f"{image_id}.jpg",
        confidence=1.0,
        R_cam_from_world=R_cam_from_world,
        cov_rot_cam_from_world=np.eye(3) * np.deg2rad(1.0) ** 2,
        prior_point_ids=np.arange(len(points3D), dtype=np.uint32),
        points3D_xyz=np.asarray(points3D, dtype=np.float64),
        uv_norm=np.asarray(uv_norm, dtype=np.float64),
    )


def test_rotation_averaging_aligns_to_location_priors():
    # Ground-truth rotations in the prior world frame; the relative rotations alone leave the world frame free.
    rotations = Rotation.from_euler("xyz", [[30, 5, 0], [32, 10, 1], [35, 14, 2], [36, 20, 2]], degrees=True)
    reconstruction = pycolmap.Reconstruction()
    reconstruction.add_camera_with_trivial_rig(pycolmap.Camera.create_from_model_name(1, "PINHOLE", 500.0, 640, 480))
    graph = pycolmap.PoseGraph()
    sidecars = native.MappingSidecars()
    for image_id in range(1, 5):
        image = pycolmap.Image(image_id=image_id, camera_id=1, name=f"{image_id}.jpg", keypoints=np.zeros((1, 2)))
        reconstruction.add_image_with_trivial_frame(image, pycolmap.Rigid3d())
        sidecars.add_image(image_id, native.ImageData())
    for first, second in ((1, 2), (2, 3), (3, 4)):
        pair = native.PairData()
        pair.all_matches = np.zeros((1, 2), dtype=np.uint32)
        pair.inlier_indices = np.zeros(1, dtype=np.int32)
        pair.are_loop_closure = np.zeros(1, dtype=np.uint8)
        pair.has_relative_pose = True
        pair_id = pycolmap.image_pair_to_pair_id(first, second)
        sidecars.add_pair(pair_id, pair)
        edge = pycolmap.PoseGraphEdge()
        edge.valid = True
        relative = rotations[second - 1] * rotations[first - 1].inv()
        edge.cam2_from_cam1 = pycolmap.Rigid3d(rotation=pycolmap.Rotation3d(relative.as_matrix()))
        graph.add_edge(first, second, edge)

    priors = LocationPriorSet(
        options=LocationPriorOptions(enabled=True),
        anchors={i: _anchor(i, rotations[i - 1].as_matrix()) for i in (2, 4)},
        mapped_points_xyz=np.zeros((0, 3)),
    )
    averager = RotationAverager(
        solve_state=SolveState(reconstruction, graph, sidecars),
        options=RAOptions(),
        sequence_id_to_index={i: i - 1 for i in range(1, 5)},
        filtered_consecutive_pair_ids=set(),
        replay=None,
        location_priors=priors,
    )
    assert averager.run_pass()
    for image_id in range(1, 5):
        estimate = reconstruction.image(image_id).cam_from_world().rotation.matrix()
        error = Rotation.from_matrix(estimate @ rotations[image_id - 1].as_matrix().T).magnitude()
        assert np.rad2deg(error) < 0.01


def test_bearing_observations_position_a_camera():
    from vidmap_native import global_positioning as gp_costs

    R_cw = Rotation.from_euler("xyz", [10, -20, 5], degrees=True).as_matrix()
    true_center = np.array([1.0, -2.0, 0.5])
    points = np.array([[5.0, 1.0, 20.0], [-4.0, 2.0, 15.0], [0.5, -3.0, 25.0], [2.0, 4.0, 18.0]]) @ R_cw + true_center
    bearings = (points - true_center) @ R_cw.T
    bearings /= np.linalg.norm(bearings, axis=1, keepdims=True)
    center = np.zeros(3)
    problem = pyceres.Problem()
    storage, count = gp_costs.append_bearing_observations(
        problem, center, R_cw, points, bearings, np.full((len(points), 3), 1e-3), pyceres.TrivialLoss()
    )
    assert count == len(points)
    summary = pyceres.SolverSummary()
    pyceres.solve(pyceres.SolverOptions(), problem, summary)
    assert summary.IsSolutionUsable()
    np.testing.assert_allclose(center, true_center, atol=1e-4)


def test_constant_point_reprojections_position_a_camera():
    from vidmap_native import bundle_adjustment as ba_costs

    reconstruction = pycolmap.Reconstruction()
    camera = pycolmap.Camera.create_from_model_name(1, "PINHOLE", 500.0, 640, 480)
    reconstruction.add_camera_with_trivial_rig(camera)
    true_pose = pycolmap.Rigid3d(pycolmap.Rotation3d(np.array([0.1, -0.2, 0.05])), np.array([0.3, -0.1, 0.4]))
    image = pycolmap.Image(image_id=1, camera_id=1, name="1.jpg")
    reconstruction.add_image_with_trivial_frame(image, pycolmap.Rigid3d(true_pose.rotation, np.zeros(3)))
    points3D = true_pose.inverse() * np.array([[1.0, 0.5, 8.0], [-1.0, 0.2, 6.0], [0.3, -1.0, 9.0], [0.8, 0.9, 7.0]])
    points2D = camera.img_from_cam(true_pose * points3D)

    pose = reconstruction.frame(reconstruction.image(1).frame_id).rig_from_world.params
    params = reconstruction.camera(1).params
    problem = pyceres.Problem()
    problem.add_parameter_block(pose, 7)
    problem.set_manifold(pose, pyceres.SubsetManifold(7, [0, 1, 2, 3]))
    problem.add_parameter_block(params, 4)
    problem.set_parameter_block_constant(params)
    loss, count = ba_costs.append_constant_point_reprojections(
        problem, reconstruction, 1, points2D, points3D, pycolmap.LossFunctionType.TRIVIAL, 1.0, 1.0
    )
    assert count == len(points3D)
    summary = pyceres.SolverSummary()
    pyceres.solve(pyceres.SolverOptions(), problem, summary)
    assert summary.IsSolutionUsable()
    np.testing.assert_allclose(reconstruction.image(1).cam_from_world().translation, true_pose.translation, atol=1e-6)


def test_gp1_alignment_is_per_covisibility_component(monkeypatch):
    # Two parts without shared points (e.g. across a cut), each off by a different similarity in GP1.
    reconstruction = pycolmap.Reconstruction()
    reconstruction.add_camera_with_trivial_rig(pycolmap.Camera.create_from_model_name(1, "PINHOLE", 500.0, 640, 480))
    centers = {1: [0.0, 0, 0], 2: [1.0, 0, 0], 3: [5.0, 0, 0], 4: [6.0, 0, 0], 5: [7.0, 0, 0]}
    for image_id, center in centers.items():
        image = pycolmap.Image(image_id=image_id, camera_id=1, name=f"{image_id}.jpg", keypoints=np.zeros((30, 2)))
        reconstruction.add_image_with_trivial_frame(image, pycolmap.Rigid3d(pycolmap.Rotation3d(), -np.array(center)))
    for component in ((1, 2), (3, 4, 5)):
        for k in range(25):
            track = pycolmap.Track()
            for image_id in component:
                track.add_element(image_id, k)
            reconstruction.add_point3D(np.array([centers[component[0]][0], 0.0, 10.0 + k]), track)
    transforms = {
        1: (1.0, np.array([10.0, 0, 0])),
        2: (1.0, np.array([10.0, 0, 0])),
        3: (2.0, np.array([0, -50.0, 0])),
        4: (2.0, np.array([0, -50.0, 0])),
        5: (2.0, np.array([0, -50.0, 0])),
    }
    expected = {i: transforms[i][0] * np.array(c) + transforms[i][1] for i, c in centers.items()}
    monkeypatch.setattr(
        LocationPriorSet,
        "_resect_camera_center_from_rays",
        staticmethod(lambda rec, anchor: expected[anchor.image_id]),
    )
    priors = LocationPriorSet(
        options=LocationPriorOptions(enabled=True, use_in_gp1=False, gp_alignment_max_angle_error_deg=180.0),
        anchors={i: _anchor(i, np.eye(3)) for i in centers},
        mapped_points_xyz=np.zeros((0, 3)),
    )
    scales = priors.align_gp1_to_location_priors_4dof(reconstruction, {i: 1.0 for i in centers})
    for image_id, center in expected.items():
        np.testing.assert_allclose(reconstruction.image(image_id).projection_center(), center, atol=1e-9)
    assert scales == {1: 1.0, 2: 1.0, 3: 2.0, 4: 2.0, 5: 2.0}
    point = next(p for p in reconstruction.points3D.values() if p.track.elements[0].image_id == 3)
    np.testing.assert_allclose(point.xyz[1], -50.0)
