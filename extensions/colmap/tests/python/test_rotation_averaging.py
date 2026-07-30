import numpy as np
import vidmap_native._core as native


def pair_id(image_id1, image_id2):
    return min(image_id1, image_id2) * 2147483647 + max(image_id1, image_id2)


def quaternion_from_angle_axis(value):
    value = np.asarray(value, dtype=np.float64)
    angle = np.linalg.norm(value)
    if angle == 0.0:
        return np.array([0.0, 0.0, 0.0, 1.0])
    return np.r_[value / angle * np.sin(angle / 2.0), np.cos(angle / 2.0)]


def rotation_problem(image_ids, posed_image_ids, edges):
    problem = native.MappingProblem()
    camera = native.CameraRecord()
    camera.camera_id = 1
    camera.model_id = 1
    camera.width = 640
    camera.height = 480
    camera.params = np.array([500.0, 500.0, 320.0, 240.0])
    problem.add_camera(camera)

    for image_id in image_ids:
        image = native.ImageRecord()
        image.image_id = image_id
        image.frame_id = image_id
        image.camera_id = 1
        image.name = f"image-{image_id}.jpg"
        image.keypoints = np.zeros((10, 2))
        image.pose.has_pose = image_id in posed_image_ids
        image.pose.translation = np.array([0.25 * image_id, -0.1 * image_id, 0.05 * image_id])
        problem.add_image(image)

    for image_id1, image_id2, angle_axis in edges:
        pair = native.PairRecord()
        pair.image_id1 = image_id1
        pair.image_id2 = image_id2
        pair.pair_id = pair_id(image_id1, image_id2)
        pair.all_matches = np.column_stack((np.arange(10), np.arange(10))).astype(np.uint32)
        pair.inlier_indices = np.arange(10, dtype=np.int32)
        pair.are_loop_closure = np.zeros(10, dtype=np.uint8)
        pair.geometry.cam2_from_cam1.has_pose = True
        pair.geometry.cam2_from_cam1.rotation_xyzw = quaternion_from_angle_axis(angle_axis)
        problem.add_pair(pair)
    return problem


def solve(problem, image_ids, edges, **option_values):
    options = native.RotationAveragingOptions()
    for name, value in option_values.items():
        setattr(options, name, value)
    result = native.run_video_rotation_averaging(
        options,
        image_ids,
        [pair_id(image_id1, image_id2) for image_id1, image_id2, _ in edges],
        problem,
    )
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

    assert solve(problem, image_ids, edges) == {1, 2, 3}


def test_unregistered_image_filter_controls_component_membership():
    image_ids = [1, 2, 3]
    edges = [
        (1, 2, [0.01, 0.03, 0.0]),
        (2, 3, [0.0, 0.02, 0.01]),
    ]

    filtered = rotation_problem(image_ids, {1, 2}, edges)
    assert solve(filtered, image_ids, edges, filter_unregistered_images=True) == {1, 2}

    initialized = rotation_problem(image_ids, {1, 2}, edges)
    assert solve(initialized, image_ids, edges, filter_unregistered_images=False) == {1, 2, 3}


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
        image_ids,
        edges,
        max_rotation_error_deg=20.0,
        tracking_huber_scale=10.0,
    ) == {1, 2, 3}
