import numpy as np
import vidmap_native._core as native
from test_records import camera_record, image_record, pair_record


def test_prepare_image_bearings_uses_locked_colmap_camera_model():
    problem = native.MappingProblem()
    problem.add_camera(camera_record())
    image = image_record(1, num_features=2)
    image.keypoints = np.array([[320.0, 240.0], [820.0, 240.0]])
    problem.add_image(image)

    native.prepare_image_bearings(problem)

    expected = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 1.0]])
    expected /= np.linalg.norm(expected, axis=1, keepdims=True)
    np.testing.assert_array_equal(problem.image(1).bearings, expected)


def calibrated_two_view(points):
    problem = native.MappingProblem()
    camera = camera_record()
    camera.model_id = 0  # SIMPLE_PINHOLE
    camera.params = np.array([1.0, 50.0, 50.0])
    camera.has_prior_focal_length = True
    problem.add_camera(camera)

    translation = np.array([1.0, 0.0, 0.0])
    for image_id in (1, 2):
        image = image_record(image_id, num_features=len(points))
        image.camera_id = 1
        keypoints = np.empty((len(points), 2))
        bearings = np.empty((len(points), 3))
        for index, point1 in enumerate(points):
            point = point1 if image_id == 1 else point1 + translation
            bearings[index] = point / np.linalg.norm(point)
            keypoints[index] = point[:2] / point[2]
        image.bearings = bearings
        image.keypoints = keypoints
        problem.add_image(image)

    pair = pair_record()
    pair.all_matches = np.column_stack([np.arange(len(points), dtype=np.uint32)] * 2)
    pair.are_loop_closure = np.zeros(len(points), dtype=np.uint8)
    pair.inlier_indices = np.empty(0, dtype=np.int32)
    pair.geometry.configuration = 2
    pair.geometry.cam2_from_cam1.has_pose = True
    pair.geometry.cam2_from_cam1.rotation_xyzw = np.array([0.0, 0.0, 0.0, 1.0])
    pair.geometry.cam2_from_cam1.translation = translation
    problem.add_pair(pair)
    problem.validate()
    return problem, pair.pair_id


def test_essential_inlier_scoring_keeps_all_exact_inliers():
    points = np.array(
        [
            [0.0, 0.0, 5.0],
            [0.0, 0.5, 5.0],
            [0.0, -0.5, 5.0],
            [-0.3, 0.4, 5.0],
            [-0.4, -0.3, 5.0],
        ]
    )
    problem, pair_id = calibrated_two_view(points)
    options = native.InlierThresholdOptions()
    options.max_epipolar_error_essential = 1e-2
    native.score_image_pair_inliers(options, True, problem)
    np.testing.assert_array_equal(problem.pair(pair_id).inlier_indices, np.arange(5))


def test_pair_filters_preserve_empty_pair_ratio_behavior():
    problem, pair_id = calibrated_two_view(np.array([[0.0, 0.0, 5.0]]))
    assert native.filter_pairs_by_inlier_ratio(0.5, problem) == 1
    assert not problem.pair(pair_id).is_valid


def test_inlier_count_filter_reports_only_newly_invalid_pairs():
    problem, pair_id = calibrated_two_view(np.array([[0.0, 0.0, 5.0]]))
    assert native.filter_pairs_by_inlier_count(1, problem) == 1
    assert native.filter_pairs_by_inlier_count(1, problem) == 0
    assert not problem.pair(pair_id).is_valid
