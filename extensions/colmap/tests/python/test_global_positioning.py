import numpy as np
import pytest
import vidmap_native._core as native


def test_global_positioning_options_accept_explicit_frame_centers():
    options = native.GlobalPositioningOptions()
    options.initial_frame_centers = {7: np.array([1.0, 2.0, 3.0])}

    np.testing.assert_array_equal(options.initial_frame_centers[7], [1.0, 2.0, 3.0])


def _playback_problem():
    problem = native.MappingProblem()
    camera = native.CameraRecord()
    camera.camera_id = 1
    camera.model_id = 1
    camera.width = 640
    camera.height = 480
    camera.params = np.asarray([500.0, 500.0, 320.0, 240.0])
    camera.has_prior_focal_length = True
    problem.add_camera(camera)

    point = np.asarray([0.0, 0.0, 5.0])
    centers = (np.zeros(3), np.asarray([1.0, 0.0, 0.0]), np.asarray([0.0, 1.0, 0.0]))
    for image_id, center in enumerate(centers, start=1):
        camera_point = point - center
        image = native.ImageRecord()
        image.image_id = image_id
        image.camera_id = 1
        image.frame_id = image_id
        image.name = f"image-{image_id}.jpg"
        image.keypoints = np.asarray(
            [
                [
                    500.0 * camera_point[0] / camera_point[2] + 320.0,
                    500.0 * camera_point[1] / camera_point[2] + 240.0,
                ]
            ]
        )
        image.bearings = np.asarray([camera_point / np.linalg.norm(camera_point)])
        pose = native.PoseRecord()
        pose.has_pose = True
        pose.translation = -center
        image.pose = pose
        problem.add_image(image)

    track = native.TrackRecord()
    track.point3D_id = 7
    track.xyz = point
    track.observations = np.asarray([[1, 0], [2, 0]], dtype=np.uint32)
    track.loop_closure_observations = np.asarray([[3, 0]], dtype=np.uint32)
    track.loop_closure_anchors = np.asarray([[1, 0]], dtype=np.uint32)
    problem.add_track(track)
    return problem


def _sequential_support_problem(storage_order):
    problem = _playback_problem()
    track = problem.track(7)
    track.observations = np.asarray([[image_id, 0] for image_id in storage_order], dtype=np.uint32)
    track.loop_closure_observations = np.empty((0, 2), dtype=np.uint32)
    track.loop_closure_anchors = np.empty((0, 2), dtype=np.uint32)
    problem.update_track(track)
    return problem


def _sequential_support_options(iterations):
    options = native.GlobalPositioningOptions()
    options.generate_random_positions = False
    options.generate_random_points = False
    options.use_initial_positions = True
    options.min_num_views_per_track = 2
    options.num_threads = 1
    options.max_num_iterations = 1
    options.sequential_support_warmup_rounds = iterations
    options.sequential_support_max_trust_region_radius = 1e4
    options.sequential_support_observations_per_track = 2
    options.sequential_support_image_timeline = [1, 2, 3]
    return options


def test_continuous_support_preserves_capped_reference_solution():
    problem = _sequential_support_problem([3, 1, 2])
    options = _sequential_support_options(2)

    result = native.run_global_positioning(options, problem)

    assert result.success
    np.testing.assert_allclose(
        problem.track(7).xyz,
        [0.14776050274253688, 0.004862738467740524, 3.497669352556072],
        rtol=0.0,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        [result.final_bata_scales[key] for key in ("7:1:0:0", "7:2:0:0", "7:3:0:0")],
        [0.5868746726263283, 0.5785674026757762, 1.0],
        rtol=0.0,
        atol=1e-12,
    )


def test_sequential_support_requires_complete_unique_timeline():
    options = _sequential_support_options(2)
    options.sequential_support_image_timeline = [1, 2]

    with np.testing.assert_raises(ValueError):
        native.run_global_positioning(options, _sequential_support_problem([1, 2, 3]))


def test_sequential_support_uses_explicit_chronology():
    def solve(timeline):
        problem = _sequential_support_problem([3, 1, 2])
        for image_id in (1, 2):
            image = problem.image(image_id)
            image.is_inlier = np.ones(1, dtype=np.uint8)
            problem.update_image(image)
        options = _sequential_support_options(2)
        options.sequential_support_image_timeline = timeline
        native.run_global_positioning(options, problem)
        return problem.track(7).xyz

    np.testing.assert_allclose(
        solve([1, 2, 3]),
        [0.09679540707997863, 0.13290493297426395, 3.353711273003999],
        rtol=0.0,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        solve([3, 2, 1]),
        [0.011432501623407216, 0.3830162700655329, 2.9690467648852747],
        rtol=0.0,
        atol=1e-12,
    )


def test_sequential_support_is_disabled_by_default():
    result = native.run_global_positioning(
        native.GlobalPositioningOptions(),
        _sequential_support_problem([1, 2, 3]),
    )

    assert result.success


def test_sequential_support_playback_starts_before_warmup():
    captures = []
    options = _sequential_support_options(4)
    options.playback.snapshot_every_n_iterations = 3
    options.playback.callback = captures.append

    result = native.run_global_positioning(options, _sequential_support_problem([3, 1, 2]))

    assert captures[0]["phase"] == "initial"
    assert captures[0]["iteration"] == -1
    np.testing.assert_array_equal(captures[0]["points_xyz"][0], result.initial_point3D_xyz[7])
    assert [(capture["phase"], capture["iteration"]) for capture in captures[1:-1]] == [
        ("iteration", 0),
        ("iteration", 3),
    ]
    assert captures[-1]["phase"] == "final"


def test_global_positioning_playback_emits_owned_loop_closure_state():
    problem = _playback_problem()
    captures = []
    options = native.GlobalPositioningOptions()
    options.generate_random_positions = False
    options.generate_random_points = False
    options.use_initial_positions = True
    options.use_loop_closure_observations = True
    options.min_num_views_per_track = 2
    options.num_threads = 1
    options.max_num_iterations = 3
    options.playback.callback = captures.append

    result = native.run_global_positioning(options, problem)

    assert result.success
    assert captures[0]["phase"] == "initial"
    assert captures[-1]["phase"] == "final"
    np.testing.assert_array_equal(captures[0]["image_ids"], [1, 2, 3])
    np.testing.assert_array_equal(captures[0]["point_ids"], [7])
    np.testing.assert_array_equal(captures[0]["lc_pairs"], [[1, 3]])
    np.testing.assert_array_equal(captures[0]["lc_support_count"], [1])
    assert np.isfinite(captures[0]["lc_raw_score"]).all()


def test_global_positioning_temporal_acceleration_is_opt_in():
    off_options = native.GlobalPositioningOptions()
    off_options.generate_random_positions = False
    off_options.generate_random_points = False
    off_options.use_initial_positions = True
    off_options.use_loop_closure_observations = True
    off_options.min_num_views_per_track = 2
    off_options.num_threads = 1
    off_options.max_num_iterations = 3

    off_result = native.run_global_positioning(off_options, _playback_problem())

    assert off_result.success
    assert off_result.diagnostics.num_temporal_acceleration_residuals == 0

    prior = native.TemporalAccelerationPrior()
    prior.prev_image_id = 1
    prior.image_id = 2
    prior.next_image_id = 3
    prior.dt_prev = 1.0
    prior.dt_next = 1.0
    prior.sqrt_observation_count = 1.0
    on_options = native.GlobalPositioningOptions()
    on_options.generate_random_positions = False
    on_options.generate_random_points = False
    on_options.use_initial_positions = True
    on_options.use_loop_closure_observations = True
    on_options.min_num_views_per_track = 2
    on_options.num_threads = 1
    on_options.max_num_iterations = 3
    on_options.use_temporal_acceleration_prior = True
    on_options.temporal_acceleration_priors = [prior]
    on_options.temporal_acceleration_prior_stddev = 1.0
    on_options.temporal_acceleration_prior_weight = 1.0
    on_options.temporal_acceleration_prior_loss_dead_zone = 0.0
    on_options.temporal_acceleration_prior_loss_huber_width = 1.0

    on_result = native.run_global_positioning(on_options, _playback_problem())

    assert on_result.success
    assert on_result.diagnostics.num_temporal_acceleration_residuals == 1


def test_bundle_adjustment_playback_honors_explicit_value_selection():
    problem = _playback_problem()
    captures = []
    options = native.BundleAdjustmentOptions()
    options.image_order = [1, 2]
    options.refine_points3D = False
    options.fix_rotations = True
    options.num_threads = 1
    options.max_num_iterations = 3
    options.playback.image_ids = [3, 1, 2]
    options.playback.point3D_ids = [7]
    options.playback.callback = captures.append

    result = native.run_bundle_adjustment(options, [], [], [], problem)

    assert result.success
    assert captures[0]["phase"] == "initial"
    assert captures[-1]["phase"] == "final"
    np.testing.assert_array_equal(captures[0]["image_ids"], [3, 1, 2])
    np.testing.assert_array_equal(captures[0]["point_ids"], [7])
    assert captures[0]["centers"].shape == (3, 3)
    assert captures[0]["points_xyz"].shape == (1, 3)


def test_bundle_adjustment_resolves_automatic_thread_count():
    problem = _playback_problem()
    options = native.BundleAdjustmentOptions()
    options.image_order = [1, 2]
    options.refine_points3D = False
    options.fix_rotations = True
    options.num_threads = -1
    options.max_num_iterations = 3

    result = native.run_bundle_adjustment(options, [], [], [], problem)

    assert result.success


def test_bundle_adjustment_applies_robust_log_mean_focal_prior():
    problem = _playback_problem()
    options = native.BundleAdjustmentOptions()
    options.image_order = [1, 2]
    options.refine_points3D = False
    options.fix_rotations = True
    options.num_threads = 1
    options.max_num_iterations = 50

    prior = native.LogFocalPriorRecord()
    prior.camera_id = 1
    prior.observations = np.array([[800.0, 0.05]])
    prior.loss.type = native.LossFunctionType.HUBER
    prior.loss.scale = 1.0
    prior.loss.weight = 1e6

    result = native.run_bundle_adjustment(options, [], [], [prior], problem)

    assert result.success
    assert result.diagnostics.num_intrinsics_prior_residuals == 1
    np.testing.assert_allclose(problem.camera(1).params[:2].mean(), 800.0, rtol=1e-3)


@pytest.mark.parametrize("weight,scales", [(0.025, [0.05, 0.2]), (0.0002, [0.24, 0.24])])
def test_bundle_adjustment_log_focal_cost_matches_vgc_equation(weight, scales):
    """Check the native cost, not merely convergence toward a strong prior."""
    options = native.BundleAdjustmentOptions()
    options.image_order = [1, 2]
    options.refine_points3D = False
    options.fix_all_poses = True
    options.refine_focal_length = False
    options.refine_principal_point = False
    options.refine_extra_params = False
    options.num_threads = 1
    baseline = native.run_bundle_adjustment(options, [], [], [], _playback_problem())
    observations = np.column_stack(([800.0, 400.0], scales))
    for loss_type in (
        native.LossFunctionType.TRIVIAL,
        native.LossFunctionType.HUBER,
        native.LossFunctionType.CAUCHY,
    ):
        problem = _playback_problem()
        focal = problem.camera(1).params[:2].mean()
        prior = native.LogFocalPriorRecord()
        prior.camera_id = 1
        prior.observations = observations
        prior.loss.type = loss_type
        prior.loss.scale = 1.0
        prior.loss.weight = weight
        squared = (np.log(focal / observations[:, 0]) / observations[:, 1]) ** 2
        if loss_type == native.LossFunctionType.HUBER:
            rho = np.where(squared <= 1, squared, 2 * np.sqrt(squared) - 1)
        elif loss_type == native.LossFunctionType.CAUCHY:
            rho = np.log1p(squared)
        else:
            rho = squared
        result = native.run_bundle_adjustment(options, [], [], [prior], problem)
        np.testing.assert_allclose(
            result.diagnostics.initial_cost - baseline.diagnostics.initial_cost,
            0.5 * prior.loss.weight * rho.sum(),
            rtol=1e-9,
            atol=1e-9,
        )


def test_no_focal_prior_leaves_enabled_intrinsics_free():
    problem = _sequential_support_problem([1, 2, 3])
    camera = problem.camera(1)
    camera.params = np.array([650.0, 650.0, 320.0, 240.0])
    problem.update_camera(camera)
    options = native.BundleAdjustmentOptions()
    options.image_order = [1, 2, 3]
    options.fix_all_poses = True
    options.refine_points3D = False
    options.refine_focal_length = True
    options.refine_principal_point = False
    options.refine_extra_params = False
    options.num_threads = 1
    result = native.run_bundle_adjustment(options, [], [], [], problem)
    assert result.success
    assert result.diagnostics.num_intrinsics_prior_residuals == 0
    np.testing.assert_allclose(problem.camera(1).params, [500.0, 500.0, 320.0, 240.0], atol=1e-5, rtol=0)


def test_duplicate_log_focal_camera_records_are_rejected():
    problem = _playback_problem()
    options = native.BundleAdjustmentOptions()
    options.image_order = [1, 2]
    prior = native.LogFocalPriorRecord()
    prior.camera_id = 1
    prior.observations = np.array([[500.0, 0.02]])
    with np.testing.assert_raises_regex(ValueError, "duplicate BA log-focal prior camera"):
        native.run_bundle_adjustment(options, [], [], [prior, prior], problem)


def test_bundle_adjustment_preserves_camera_without_observations():
    problem = _playback_problem()
    camera = problem.camera(1)
    camera.camera_id = 2
    problem.add_camera(camera)
    image = problem.image(3)
    image.camera_id = 2
    problem.update_image(image)
    track = problem.track(7)
    track.loop_closure_observations = np.empty((0, 2), dtype=np.uint32)
    track.loop_closure_anchors = np.empty((0, 2), dtype=np.uint32)
    problem.update_track(track)
    priors = []
    for camera_id in (1, 2):
        prior = native.LogFocalPriorRecord()
        prior.camera_id = camera_id
        prior.observations = np.array([[800.0, 0.24]])
        priors.append(prior)
    options = native.BundleAdjustmentOptions()
    options.image_order = [1, 2, 3]
    options.refine_points3D = False
    options.fix_all_poses = True
    options.num_threads = 1
    result = native.run_bundle_adjustment(options, [], [], priors, problem)
    assert result.success
    assert result.diagnostics.num_intrinsics_prior_residuals == 1
    np.testing.assert_array_equal(problem.camera(2).params, camera.params)


@pytest.mark.parametrize(("rounds", "function_tolerance", "expected_updates"), [(5, 0.0, 5), (16, 1.0, 0)])
def test_continuous_warmup_respects_budget_and_early_convergence(rounds, function_tolerance, expected_updates):
    captures = []
    options = _sequential_support_options(rounds)
    options.function_tolerance = function_tolerance
    options.gradient_tolerance = 0.0
    options.parameter_tolerance = 0.0
    options.playback.callback = captures.append

    result = native.run_global_positioning(options, _sequential_support_problem([3, 1, 2]))

    assert result.success
    warmup = [c for c in captures if c["phase"] == "iteration" and c["iteration"] < rounds]
    assert [c["iteration"] for c in warmup] == list(range(expected_updates))
    assert captures[0]["phase"] == "initial"
    assert captures[-1]["phase"] == "final"
    assert captures[-1]["iteration"] >= rounds


def test_python_radius_cap_preserves_initial_step_and_limits_later_steps():
    def warmup_updates(cap):
        captures = []
        options = _sequential_support_options(2)
        options.parameter_ordering = native.GlobalPositioningOrdering.SINGLETON
        options.function_tolerance = options.gradient_tolerance = options.parameter_tolerance = 0.0
        options.sequential_support_max_trust_region_radius = cap
        options.playback.callback = captures.append
        result = native.run_global_positioning(options, _sequential_support_problem([3, 1, 2]))
        assert result.success
        updates = [c["points_xyz"] for c in captures if c["phase"] == "iteration" and c["iteration"] < 2]
        assert len(updates) == 2
        return updates

    capped, uncapped = warmup_updates(1e4), warmup_updates(1e16)
    np.testing.assert_array_equal(capped[0], uncapped[0])
    assert np.linalg.norm(capped[1] - uncapped[1]) > 1e-8


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("warmup_rounds", -1),
        ("max_trust_region_radius", 1.0),
        ("max_trust_region_radius", 0.0),
        ("max_trust_region_radius", float("nan")),
    ],
)
def test_invalid_warmup_solver_options_are_rejected_before_mutation(field, value):
    problem = _sequential_support_problem([3, 1, 2])
    before = problem.track(7).xyz.copy()
    options = _sequential_support_options(16)
    setattr(options, f"sequential_support_{field}", value)
    with pytest.raises(ValueError, match="invalid sequential support solver options"):
        native.run_global_positioning(options, problem)
    np.testing.assert_array_equal(problem.track(7).xyz, before)
