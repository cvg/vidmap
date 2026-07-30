import numpy as np
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
