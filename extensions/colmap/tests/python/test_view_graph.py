import numpy as np
import pyceres
import pycolmap
import pytest
import vidmap_native._core as native
from test_records import add_pair, scene


def test_prepare_image_bearings_uses_locked_colmap_camera_model():
    rec, _, data = scene(1, 2)
    rec.image(1).point2D(0).xy = [320.0, 240.0]
    rec.image(1).point2D(1).xy = [820.0, 240.0]
    native.prepare_image_bearings(rec, data)
    expected = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 1.0]])
    expected /= np.linalg.norm(expected, axis=1, keepdims=True)
    np.testing.assert_array_equal(data.image(1).bearings, expected)


@pytest.mark.parametrize(
    "has_prior,valid,configuration,expected",
    [
        (True, True, 4, 2),
        (False, True, 4, 4),
        (True, False, 4, 4),
        (True, True, 3, 3),
    ],
)
def test_planar_pair_reclassification_requires_valid_calibrated_cameras(has_prior, valid, configuration, expected):
    rec, graph, data = owners = scene()
    pid = add_pair(owners)
    rec.camera(1).has_prior_focal_length = has_prior
    graph.edges[pid].valid = valid
    data.pair(pid).geometry.config = configuration
    native.reclassify_calibrated_planar_pairs(*owners)
    assert data.pair(pid).geometry.config == expected


def calibrated_two_view(points):
    rec, graph, data = owners = scene(2, len(points))
    rec.camera(1).params = [1, 1, 50, 50]
    rec.camera(1).has_prior_focal_length = True
    translation = np.array([1.0, 0.0, 0.0])
    for image_id in (1, 2):
        xyz = points if image_id == 1 else points + translation
        data.image(image_id).bearings = xyz / np.linalg.norm(xyz, axis=1)[:, None]
        for i, xy in enumerate(xyz[:, :2] / xyz[:, 2:]):
            rec.image(image_id).point2D(i).xy = xy
    pid = add_pair(owners, matches=list(zip(range(len(points)), range(len(points)))))
    pair = data.pair(pid)
    pair.geometry.config = 2
    pair.has_relative_pose = True
    pair.inlier_indices = np.empty(0, dtype=np.int32)
    graph.edges[pid].cam2_from_cam1 = pycolmap.Rigid3d(translation=translation)
    data.validate(rec)
    return owners, pid


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
    native.score_image_pair_inliers(options, *problem)
    np.testing.assert_array_equal(problem[2].pair(pair_id).inlier_indices, np.arange(5))


def test_pair_filters_preserve_empty_pair_ratio_behavior():
    problem, pair_id = calibrated_two_view(np.array([[0.0, 0.0, 5.0]]))
    assert native.filter_pairs_by_inlier_ratio(0.5, *problem[1:]) == 1
    assert not problem[1].is_valid(pair_id)


def test_inlier_count_filter_reports_only_newly_invalid_pairs():
    problem, pair_id = calibrated_two_view(np.array([[0.0, 0.0, 5.0]]))
    assert native.filter_pairs_by_inlier_count(1, *problem[1:]) == 1
    assert native.filter_pairs_by_inlier_count(1, *problem[1:]) == 0
    assert not problem[1].is_valid(pair_id)


def focal_prior(rows, weight=1.0, loss="huber", camera_id=1):
    record = native.LogFocalPriorRecord()
    record.camera_id = camera_id
    record.observations = np.asarray(rows)
    record.loss = pyceres.LossFunction(dict(name=loss, params=[1.0], magnitude=weight))
    return record


def focal_calibration_problem():
    problem = scene()
    angle = 0.2
    rotation = np.array(
        [
            [np.cos(angle), 0, np.sin(angle)],
            [0, 1, 0],
            [-np.sin(angle), 0, np.cos(angle)],
        ]
    )
    translation_skew = np.array([[0, -0.4, 0.3], [0.4, 0, -0.2], [-0.3, 0.2, 0]])
    intrinsics = np.array([[500.0, 0, 320.0], [0, 500.0, 240.0], [0, 0, 1]])
    inverse = np.linalg.inv(intrinsics)
    pid = add_pair(problem)
    pair = problem[2].pair(pid)
    pair.geometry.config = 3  # UNCALIBRATED
    pair.geometry.F = inverse.T @ translation_skew @ rotation @ inverse
    return problem


@pytest.mark.parametrize("camera_ids,error", [([1, 1], ValueError), ([2], IndexError)])
def test_focal_observation_camera_inventory(camera_ids, error):
    options = pycolmap.ViewGraphCalibrationOptions()
    focal_priors = [focal_prior([[600.0, 0.1]], camera_id=cid) for cid in camera_ids]
    with pytest.raises(error):
        native.calibrate_focal_lengths(options, *focal_calibration_problem(), focal_priors=focal_priors)


def test_empty_or_zero_weight_focal_observations_preserve_raw_calibration():
    calibrated = []
    for focal_priors in ([], [focal_prior([[600.0, 0.24]], weight=0.0)]):
        problem = focal_calibration_problem()
        options = pycolmap.ViewGraphCalibrationOptions()
        native.calibrate_focal_lengths(
            options,
            *problem,
            focal_priors=focal_priors,
        )
        calibrated.append(problem[0].camera(1).params)
    np.testing.assert_allclose(*calibrated, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(calibrated[0][:2], [500.0, 500.0], atol=1e-8)


@pytest.mark.parametrize("locked", [False, True])
@pytest.mark.parametrize("with_prior", [False, True])
def test_focal_prior_respects_camera_lock(locked, with_prior):
    problem = focal_calibration_problem()
    camera = problem[0].camera(1)
    camera.params = np.array([800.0, 800.0, 320.0, 240.0])
    camera.has_prior_focal_length = locked
    options = pycolmap.ViewGraphCalibrationOptions()
    focal_priors = []
    if with_prior:
        focal_priors = [focal_prior([[600.0, 0.24]], weight=1e-5, loss="cauchy")]
    native.calibrate_focal_lengths(
        options,
        *problem,
        focal_priors=focal_priors,
    )
    expected = 800.0 if locked else 500.0
    np.testing.assert_allclose(problem[0].camera(1).params[:2], expected, atol=0.01, rtol=0)


def test_positive_focal_prior_and_ratio_rollback():
    optimized = []
    for maximum_ratio in (10.0, 1.0):
        problem = focal_calibration_problem()
        options = pycolmap.ViewGraphCalibrationOptions()
        options.max_focal_length_ratio = maximum_ratio
        focal_priors = [focal_prior([[600.0, 0.1]], weight=0.1 * options.loss_function_scale**2, loss="cauchy")]
        native.calibrate_focal_lengths(
            options,
            *problem,
            focal_priors=focal_priors,
        )
        optimized.append(problem[0].camera(1).params[0])
    assert optimized[0] > 500.0
    assert optimized[1] == 500.0


@pytest.mark.parametrize("normalize", [False, True])
def test_selected_vgc_recipe_matches_explicit_native_control(normalize):
    from types import SimpleNamespace

    from vidmap.mapper.options.view_graph import VGCCalibrationOptions
    from vidmap.mapper.stages.view_graph_calibration import ViewGraphCalibrator

    rows = np.array([[600.0, 0.24], [650.0, 0.24]])
    control = focal_calibration_problem()
    options = pycolmap.ViewGraphCalibrationOptions()
    options.min_focal_length_ratio = np.finfo(float).tiny
    options.max_focal_length_ratio = np.finfo(float).max
    weight = 0.00001 * 1 / 2 if normalize else 0.00001
    focal_priors = [focal_prior(rows, weight=weight, loss="cauchy")]
    native.calibrate_focal_lengths(options, *control, focal_priors=focal_priors)

    problem = focal_calibration_problem()
    state = SimpleNamespace(
        reconstruction=problem[0],
        pose_graph=problem[1],
        sidecars=problem[2],
        pair_order=list(problem[1].edges),
        pair_data=problem[2].pair,
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
    np.testing.assert_allclose(problem[0].camera(1).params, control[0].camera(1).params, rtol=1e-12, atol=1e-12)
    np.testing.assert_array_equal(prior[1], rows)
