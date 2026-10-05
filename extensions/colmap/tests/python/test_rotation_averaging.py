import numpy as np
import pycolmap
import pytest
import vidmap_native._core as native


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


def solve(problem, **option_values):
    options = native.RotationAveragingOptions()
    for name, value in option_values.items():
        setattr(options, name, value)
    result = native.run_video_rotation_averaging(options, *problem)
    assert result.success
    return set(result.registered_image_ids)


def test_disconnected_graph_solves_only_the_largest_component():
    image_ids = [1, 2, 3, 4, 5]
    edges = [
        (1, 2, [0.0, 0.04, 0.0]),
        (2, 3, [0.0, 0.05, 0.0]),
        (4, 5, [0.0, -0.03, 0.0]),
    ]
    problem = rotation_problem(image_ids, set(image_ids), edges)

    assert solve(problem) == {1, 2, 3}


def test_unregistered_image_filter_controls_component_membership():
    image_ids = [1, 2, 3]
    edges = [
        (1, 2, [0.01, 0.03, 0.0]),
        (2, 3, [0.0, 0.02, 0.01]),
    ]

    filtered = rotation_problem(image_ids, {1, 2}, edges)
    assert solve(filtered, filter_unregistered_images=True) == {1, 2}

    initialized = rotation_problem(image_ids, {1, 2}, edges)
    assert solve(initialized, filter_unregistered_images=False) == {1, 2, 3}


@pytest.mark.parametrize("threshold", [0.0, 20.0])
def test_mapper_preserves_unposed_images_and_translations_after_rotation_initialization(threshold):
    from vidmap.mapper.native.state import SolveState
    from vidmap.mapper.options.view_graph import RAOptions
    from vidmap.mapper.stages.rotation_averaging import RotationAverager

    reconstruction, graph, sidecars = rotation_problem([1, 2, 3], {1, 2}, [(1, 2, [0, 0.04, 0]), (2, 3, [0, 0.05, 0])])
    translations = {i: reconstruction.image(i).cam_from_world().translation.copy() for i in (1, 2)}
    averager = RotationAverager(
        solve_state=SolveState(reconstruction, graph, sidecars),
        options=RAOptions(),
        consecutive_pair_ids=[],
        filtered_consecutive_pair_ids=set(),
        replay=None,
    )
    options = native.RotationAveragingOptions()
    options.filter_unregistered_images = False
    options.max_rotation_error_deg = threshold
    assert averager.run_pass(options).success
    assert not reconstruction.image(3).has_pose
    assert set(reconstruction.reg_image_ids()) == {1, 2}
    for image_id, translation in translations.items():
        np.testing.assert_array_equal(reconstruction.image(image_id).cam_from_world().translation, translation)


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
        tracking_huber_scale=10.0,
    ) in ({1, 2, 3}, {4, 5, 6})
