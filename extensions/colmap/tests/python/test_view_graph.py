import numpy as np
import pytest
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


def focal_prior(rows, weight=1.0, loss="huber", camera_id=1):
    record = native.LogFocalPriorRecord()
    record.camera_id = camera_id
    record.observations = np.asarray(rows)
    record.loss.type = getattr(native.LossFunctionType, loss.upper())
    record.loss.weight = weight
    return record


@pytest.mark.parametrize("weight", [-0.1, np.nan, np.inf])
def test_focal_observations_reject_invalid_loss_weight(weight):
    options = native.FocalCalibrationOptions()
    options.focal_priors = [focal_prior([[600.0, 0.1]], weight=weight)]
    with pytest.raises(ValueError):
        options.validate()


def focal_calibration_problem():
    problem = native.MappingProblem()
    problem.add_camera(camera_record())
    problem.add_image(image_record(1))
    problem.add_image(image_record(2))
    angle = 0.2
    rotation = np.array([[np.cos(angle), 0, np.sin(angle)], [0, 1, 0], [-np.sin(angle), 0, np.cos(angle)]])
    translation_skew = np.array([[0, -0.4, 0.3], [0.4, 0, -0.2], [-0.3, 0.2, 0]])
    intrinsics = np.array([[500.0, 0, 320.0], [0, 500.0, 240.0], [0, 0, 1]])
    inverse = np.linalg.inv(intrinsics)
    pair = pair_record()
    pair.geometry.configuration = 3  # UNCALIBRATED
    pair.geometry.has_fundamental = True
    pair.geometry.fundamental = inverse.T @ translation_skew @ rotation @ inverse
    problem.add_pair(pair)
    return problem


@pytest.mark.parametrize("camera_ids,error", [([1, 1], ValueError), ([2], IndexError)])
def test_focal_observation_camera_inventory(camera_ids, error):
    options = native.FocalCalibrationOptions()
    options.focal_priors = [focal_prior([[600.0, 0.1]], camera_id=cid) for cid in camera_ids]
    with pytest.raises(error):
        native.calibrate_focal_lengths(options, focal_calibration_problem())


def test_empty_or_zero_weight_focal_observations_preserve_raw_calibration():
    calibrated = []
    for priors in (None, [], [focal_prior([[600.0, 0.24]], weight=0.0)]):
        problem = focal_calibration_problem()
        options = native.FocalCalibrationOptions()
        options.num_threads = 1
        if priors is not None:
            options.focal_priors = priors
        result = native.calibrate_focal_lengths(options, problem)
        assert result.success
        native.apply_focal_calibration(options, result, problem)
        calibrated.append(problem.camera(1).params)
    np.testing.assert_array_equal(calibrated[1:], [calibrated[0]] * 2)
    np.testing.assert_allclose(calibrated[0][:2], [500.0, 500.0], atol=1e-8)


@pytest.mark.parametrize("locked", [False, True])
@pytest.mark.parametrize("with_prior", [False, True])
def test_focal_prior_respects_camera_lock(locked, with_prior):
    problem = focal_calibration_problem()
    camera = problem.camera(1)
    camera.params = np.array([800.0, 800.0, 320.0, 240.0])
    camera.has_prior_focal_length = locked
    problem.update_camera(camera)
    options = native.FocalCalibrationOptions()
    options.num_threads = 1
    if with_prior:
        options.focal_priors = [focal_prior([[600.0, 0.24]], weight=1e-5, loss="cauchy")]
    result = native.calibrate_focal_lengths(options, problem)
    assert result.success
    expected = 800.0 if locked else 500.0
    # Inspect the solved focal independently of the apply-time lock.
    camera.has_prior_focal_length = False
    problem.update_camera(camera)
    native.apply_focal_calibration(options, result, problem)
    np.testing.assert_allclose(problem.camera(1).params[:2], expected, atol=0.01, rtol=0)


@pytest.mark.parametrize("loss", ["huber", "cauchy"])
def test_positive_focal_prior_and_ratio_rollback(loss):
    optimized = []
    for maximum_ratio in (10.0, 1.0):
        problem = focal_calibration_problem()
        options = native.FocalCalibrationOptions()
        options.num_threads = 1
        options.max_focal_length_ratio = maximum_ratio
        options.focal_priors = [focal_prior([[600.0, 0.1]], weight=0.1 * options.loss_function_scale**2, loss=loss)]
        result = native.calibrate_focal_lengths(options, problem)
        assert result.success
        native.apply_focal_calibration(options, result, problem)
        optimized.append(problem.camera(1).params[0])
    assert optimized[0] > 500.0
    assert optimized[1] == 500.0


@pytest.mark.parametrize("num_pairs,num_observations", [(0, 2), (1, 2), (4, 3), (4, 7)])
@pytest.mark.parametrize("loss", ["huber", "cauchy"])
def test_native_pair_normalization_matches_explicit_weights(num_pairs, num_observations, loss):
    rows = np.column_stack([np.linspace(600, 700, num_observations), np.full(num_observations, 0.24)])
    calibrated = []
    for normalize in (False, True):
        problem = focal_calibration_problem()
        base = problem.pair(problem.pair_ids[0])
        base.is_valid = num_pairs > 0
        problem.update_pair(base)
        for index in range(1, num_pairs + 2):
            problem.add_image(image_record(index + 2))
            pair = pair_record(1, index + 2)
            pair.geometry = base.geometry
            pair.geometry.configuration = 2 if index % 2 else 3
            pair.is_valid = index != num_pairs
            if index == num_pairs + 1:
                pair.geometry.configuration = 4  # Excluded/planar.
            problem.add_pair(pair)
        options = native.FocalCalibrationOptions()
        options.num_threads = 1
        options.normalize_weight_by_pair_count = normalize
        weight = 0.00001 if normalize else 0.00001 * num_pairs / num_observations
        options.focal_priors = [focal_prior(rows, weight=weight, loss=loss)]
        result = native.calibrate_focal_lengths(options, problem)
        assert result.success
        native.apply_focal_calibration(options, result, problem)
        assert options.focal_priors[0].loss.weight == weight
        calibrated.append(problem.camera(1).params)
    np.testing.assert_array_equal(*calibrated)


@pytest.mark.parametrize("normalize", [False, True])
def test_selected_vgc_recipe_matches_explicit_native_control(normalize):
    from types import SimpleNamespace

    from vidmap.mapper.options.view_graph import VGCCalibrationOptions
    from vidmap.mapper.stages.view_graph_calibration import ViewGraphCalibrator

    rows = np.array([[600.0, 0.24], [650.0, 0.24]])
    control = focal_calibration_problem()
    options = native.FocalCalibrationOptions()
    options.min_focal_length_ratio = np.finfo(float).tiny
    options.max_focal_length_ratio = np.finfo(float).max
    weight = 0.00001 * 1 / 2 if normalize else 0.00001
    options.focal_priors = [focal_prior(rows, weight=weight, loss="cauchy")]
    result = native.calibrate_focal_lengths(options, control)
    assert result.success
    native.apply_focal_calibration(options, result, control)

    problem = focal_calibration_problem()
    state = SimpleNamespace(
        reconstruction=SimpleNamespace(cameras={1: SimpleNamespace(has_prior_focal_length=False)}),
        native_problem=problem,
        import_cameras=lambda: None,
        export_cameras=lambda: None,
        pair_records=lambda: {pid: problem.pair(pid) for pid in problem.pair_ids},
    )
    prior = {1: tuple(map(tuple, rows))}
    ViewGraphCalibrator(
        solve_state=state,
        options=VGCCalibrationOptions(normalize_weight_by_pair_count=normalize),
        enabled=True,
        consecutive_pair_ids=[],
        exclusion_ids=set(),
        focal_prior=prior,
    ).calibrate()
    np.testing.assert_array_equal(problem.camera(1).params, control.camera(1).params)
    np.testing.assert_array_equal(prior[1], rows)
