import numpy as np
import pycolmap
import pytest

from vidmap.mapper.native.state import SolveState
from vidmap.mapper.options.view_graph import RAOptions
from vidmap.mapper.stages.rotation_averaging import RotationAverager


def rotation_problem(image_ids, posed_image_ids, edges):
    from test_records import add_pair, scene

    reconstruction, graph, sidecars = owners = scene(max(image_ids), 10)
    for image_id in image_ids:
        image = reconstruction.image(image_id)
        if image_id in posed_image_ids:
            image.frame.rig_from_world = pycolmap.Rigid3d(
                translation=[0.25 * image_id, -0.1 * image_id, 0.05 * image_id]
            )
        else:
            reconstruction.deregister_frame(image.frame_id)
    for first, second, angle_axis in edges:
        pid = add_pair(owners, first, second, list(zip(range(10), range(10))))
        sidecars.pair(pid).has_relative_pose = True
        graph.edges[pid].cam2_from_cam1 = pycolmap.Rigid3d(
            rotation=pycolmap.Rotation3d(np.asarray(angle_axis, dtype=np.float64))
        )
    return owners


def solve(problem, sequence_ids=None, **option_values):
    if sequence_ids is None:
        sequence_ids = sorted(problem[0].images)
    averager = RotationAverager(
        solve_state=SolveState(*problem),
        options=RAOptions(**option_values),
        sequence_id_to_index={image_id: index for index, image_id in enumerate(sequence_ids)},
        filtered_consecutive_pair_ids=set(),
        replay=None,
    )
    assert averager.run_pass()
    return set(problem[0].reg_image_ids())


def test_disconnected_graph_solves_only_the_largest_component():
    image_ids = [1, 2, 3, 4, 5]
    edges = [
        (1, 2, [0.0, 0.04, 0.0]),
        (2, 3, [0.0, 0.05, 0.0]),
        (4, 5, [0.0, -0.03, 0.0]),
    ]
    problem = rotation_problem(image_ids, set(image_ids), edges)

    before = {i: problem[0].image(i).cam_from_world().matrix().copy() for i in (4, 5)}
    solve(problem)
    for i, pose in before.items():
        np.testing.assert_array_equal(problem[0].image(i).cam_from_world().matrix(), pose)
    for a, b, angle_axis in edges[:2]:
        relative = (
            problem[0].image(b).cam_from_world().rotation * problem[0].image(a).cam_from_world().rotation.inverse()
        )
        np.testing.assert_allclose(relative.matrix(), pycolmap.Rotation3d(np.array(angle_axis)).matrix(), atol=1e-8)


def test_unregistered_image_filter_controls_component_membership():
    image_ids = [1, 2, 3]
    edges = [
        (1, 2, [0.01, 0.03, 0.0]),
        (2, 3, [0.0, 0.02, 0.01]),
    ]

    filtered = rotation_problem(image_ids, {1, 2}, edges)
    assert solve(filtered, filter_unregistered_images=True) == {1, 2}

    initialized = rotation_problem(image_ids, {1, 2}, edges)
    assert solve(initialized, filter_unregistered_images=False) == {1, 2}


@pytest.mark.parametrize("threshold", [0.0, 20.0])
def test_rotation_initialization_preserves_pose_mask(threshold):
    reconstruction, graph, sidecars = rotation_problem([1, 2, 3], {1, 2}, [(1, 2, [0, 0.04, 0]), (2, 3, [0, 0.05, 0])])
    averager = RotationAverager(
        solve_state=SolveState(reconstruction, graph, sidecars),
        options=RAOptions(filter_unregistered_images=False, max_rotation_error_deg=threshold),
        sequence_id_to_index={1: 0, 2: 1, 3: 2},
        filtered_consecutive_pair_ids=set(),
        replay=None,
    )
    assert averager.run_pass()
    assert not reconstruction.image(3).has_pose
    assert set(reconstruction.reg_image_ids()) == {1, 2}
    for image_id in (1, 2):
        assert np.isnan(reconstruction.image(image_id).cam_from_world().translation).all()


def test_outlier_filter_does_not_reintroduce_a_discarded_component():
    image_ids = list(range(1, 11))
    edges = [
        (1, 2, [0.0, 0.0, 0.0]),
        (1, 3, [0.0, 0.0, 0.0]),
        (2, 3, [0.0, 0.0, 0.0]),
        (4, 5, [0.0, 0.0, 0.0]),
        (4, 6, [0.0, 0.0, 0.0]),
        (5, 6, [0.0, 0.0, 0.0]),
        (1, 4, [0.8, 0.0, 0.0]),
        (3, 6, [-0.8, 0.0, 0.0]),
        (7, 8, [0.0, 0.0, 0.0]),
        (8, 9, [0.0, 0.0, 0.0]),
        (9, 10, [0.0, 0.0, 0.0]),
    ]
    problem = rotation_problem(image_ids, set(image_ids), edges)

    assert solve(
        problem,
        max_rotation_error_deg=20.0,
        video_tracking_huber_scale=10.0,
    ) in ({1, 2, 3}, {4, 5, 6})


@pytest.mark.parametrize("skip_loops", [False, True])
def test_vidmap_losses_and_tracking_initialization(monkeypatch, skip_loops):
    problem = rotation_problem([1, 2, 3], {1, 2, 3}, [(1, 2, [0, 0, 0]), (2, 3, [0, 0, 0]), (1, 3, [0, 0, 1])])
    rec, graph, sidecars = problem
    loop_id = pycolmap.image_pair_to_pair_id(1, 3)
    sidecars.pair(loop_id).are_loop_closure = np.ones(10, dtype=np.uint8)
    create = pycolmap.create_default_ceres_rotation_averager
    losses = []

    def capture(options, local_graph, reconstruction):
        losses.append(options.ceres.loss_function_type)
        assert set(local_graph.edges) == set(graph.edges) - {loop_id}
        rotations = {i: image.cam_from_world().rotation.matrix().copy() for i, image in reconstruction.images.items()}
        owner = create(options, local_graph, reconstruction)
        assert owner.problem.num_residual_blocks() == 2
        for image_id, image in reconstruction.images.items():
            expected = rotations[image_id] if options.skip_initialization else np.eye(3)
            np.testing.assert_allclose(image.cam_from_world().rotation.matrix(), expected, atol=1e-12)

        class Solve:
            problem = owner.problem
            add_relative_rotation_residual = owner.add_relative_rotation_residual

            def solve(self):
                assert self.problem.num_residual_blocks() == (2 if skip_loops else 3)
                for image in reconstruction.images.values():
                    image.frame.rig_from_world.rotation.quat[:] = pycolmap.Rotation3d().quat
                reconstruction.image(2).frame.rig_from_world.rotation.quat[:] = pycolmap.Rotation3d(
                    np.array([0.4, 0, 0])
                ).quat
                summary = owner.solve()
                expected = 2 * (
                    2 * 0.1 * 0.4 - 0.1**2
                    if options.ceres.loss_function_type == pycolmap.LossFunctionType.HUBER
                    else 0.1**2 * np.log1p(0.4**2 / 0.1**2)
                )
                if not skip_loops:
                    expected += 0.05**2 * np.log1p(1 / 0.05**2)
                assert summary.initial_cost == pytest.approx(0.5 * expected)
                return summary

        return Solve()

    monkeypatch.setattr(pycolmap, "create_default_ceres_rotation_averager", capture)
    solve(problem, filter_risky_loop_closure_pairs=skip_loops)
    assert losses == [pycolmap.LossFunctionType.HUBER, pycolmap.LossFunctionType.CAUCHY]
    assert graph.edges[loop_id].num_matches == 0  # Initialization weights stay local to RA.


@pytest.mark.parametrize("skip_loops", [False, True])
@pytest.mark.parametrize("all_loops", [False, True])
def test_loop_closures_bridge_nearest_temporal_gaps(monkeypatch, skip_loops, all_loops):
    edges = [(1, 2, [0, 0, 0]), (2, 3, [0, 0, 0]), (1, 4, [0, 0, 0]), (3, 4, [0, 0, 1]), (5, 6, [0, 0, 0])]
    problem = rotation_problem(range(1, 7), set(range(1, 7)), edges)
    rec, graph, sidecars = problem
    for first, second, count in [(1, 2, 10), (2, 3, 4), (1, 4, 2), (3, 4, 9), (5, 6, 10)]:
        pair = sidecars.pair(pycolmap.image_pair_to_pair_id(first, second))
        pair.inlier_indices = np.arange(count, dtype=np.int32)
        if all_loops or (first, second) != (1, 2):
            pair.are_loop_closure = np.ones(10, dtype=np.uint8)
    create = pycolmap.create_default_ceres_rotation_averager

    def capture(options, local_graph, reconstruction):
        assert {pid for pid, edge in local_graph.edges.items() if edge.valid} == {
            pycolmap.image_pair_to_pair_id(a, b) for a, b in [(1, 2), (2, 3), (1, 4)]
        }
        rotation = reconstruction.image(4).cam_from_world().rotation.matrix().copy()
        owner = create(options, local_graph, reconstruction)
        assert owner.problem.num_residual_blocks() == 3
        expected = rotation if options.skip_initialization else np.eye(3)
        np.testing.assert_allclose(reconstruction.image(4).cam_from_world().rotation.matrix(), expected, atol=1e-12)

        class Solve:
            add_relative_rotation_residual = owner.add_relative_rotation_residual

            def solve(self):
                assert owner.problem.num_residual_blocks() == (3 if skip_loops else 4)
                for image in reconstruction.images.values():
                    image.frame.rig_from_world.rotation.quat[:] = pycolmap.Rotation3d().quat
                reconstruction.image(4).frame.rig_from_world.rotation.quat[:] = pycolmap.Rotation3d(
                    np.array([0, 0, 0.4])
                ).quat
                summary = owner.solve()
                expected = (
                    2 * 0.1 * 0.4 - 0.1**2
                    if options.ceres.loss_function_type == pycolmap.LossFunctionType.HUBER
                    else 0.1**2 * np.log1p(0.4**2 / 0.1**2)
                )
                if not skip_loops:
                    expected += 0.05**2 * np.log1p(0.6**2 / 0.05**2)
                assert summary.initial_cost == pytest.approx(0.5 * expected)
                return summary

        return Solve()

    monkeypatch.setattr(pycolmap, "create_default_ceres_rotation_averager", capture)
    # Pair (1, 4) is closer in time than (3, 4), despite fewer inliers and a larger ID gap.
    solve(problem, sequence_ids=[3, 2, 1, 4, 5, 6], filter_risky_loop_closure_pairs=skip_loops)
    assert all(edge.num_matches == 0 for edge in graph.edges.values())


def test_filtering_every_edge_fails():
    problem = rotation_problem([1, 2, 3], {1, 2, 3}, [(1, 2, [0, 0, 0]), (2, 3, [0, 0, 0]), (1, 3, [0, 0.3, 0])])
    averager = RotationAverager(
        solve_state=SolveState(*problem),
        options=RAOptions(max_rotation_error_deg=1e-6),
        sequence_id_to_index={1: 0, 2: 1, 3: 2},
        filtered_consecutive_pair_ids=set(),
        replay=None,
    )
    assert not averager.run_pass()
