import numpy as np
import pytest
import vidmap_native._core as native


def camera_record(camera_id=1, *, has_prior=True):
    camera = native.CameraRecord()
    camera.camera_id = camera_id
    camera.model_id = 1  # PINHOLE
    camera.width = 640
    camera.height = 480
    camera.params = np.array([500.0, 500.0, 320.0, 240.0])
    camera.has_prior_focal_length = has_prior
    return camera


def image_record(image_id, num_features, *, camera_id=1, valid_depth=True):
    image = native.ImageRecord()
    image.image_id = image_id
    image.camera_id = camera_id
    image.frame_id = image_id
    image.name = f"image-{image_id}.jpg"
    values = np.arange(num_features, dtype=float) * 100.0
    image.keypoints = np.column_stack([values, values])
    image.bearings = np.tile(np.array([[0.0, 0.0, 1.0]]), (num_features, 1))
    image.depth_values = np.ones(num_features) if valid_depth else np.zeros(num_features)
    image.depth_validity = np.full(num_features, int(valid_depth), dtype=np.uint8)
    return image


def pair_record(image_id1, image_id2, matches, *, loop_closure_rows=()):
    pair = native.PairRecord()
    pair.image_id1 = image_id1
    pair.image_id2 = image_id2
    pair.pair_id = min(image_id1, image_id2) * 2147483647 + max(image_id1, image_id2)
    pair.all_matches = np.asarray(matches, dtype=np.uint32).reshape(-1, 2)
    pair.inlier_indices = np.arange(len(matches), dtype=np.int32)
    loop_closure_mask = np.zeros(len(matches), dtype=np.uint8)
    loop_closure_mask[list(loop_closure_rows)] = 1
    pair.are_loop_closure = loop_closure_mask
    return pair


def mapping_problem(num_images, num_features, *, valid_depth=True):
    problem = native.MappingProblem()
    problem.add_camera(camera_record())
    for image_id in range(1, num_images + 1):
        problem.add_image(image_record(image_id, num_features, valid_depth=valid_depth))
    return problem


def observation_set(track, *, loop_closure=False):
    values = track.loop_closure_observations if loop_closure else track.observations
    return {tuple(map(int, row)) for row in values}


def track_by_id(tracks):
    return {int(track.point3D_id): track for track in tracks}


def test_establishes_triangle_tracks_with_observation_root_ids():
    problem = mapping_problem(3, 5)
    matches = [(index, index) for index in range(5)]
    pairs = [
        pair_record(1, 2, matches),
        pair_record(1, 3, matches),
        pair_record(2, 3, matches),
    ]
    for pair in pairs:
        problem.add_pair(pair)

    options = native.TrackEstablishmentOptions()
    tracks = native.establish_full_tracks(
        problem,
        [1, 2, 3],
        [pair.pair_id for pair in pairs],
        options,
    )

    assert len(tracks) == 5
    by_id = track_by_id(tracks)
    for feature_id in range(5):
        track_id = (1 << 32) | feature_id
        assert observation_set(by_id[track_id]) == {
            (1, feature_id),
            (2, feature_id),
            (3, feature_id),
        }


def test_intra_image_inconsistency_drops_fused_track():
    problem = mapping_problem(3, 2)
    pairs = [
        pair_record(1, 2, [(0, 0)]),
        pair_record(2, 3, [(0, 0)]),
        pair_record(1, 3, [(1, 0)]),
    ]
    for pair in pairs:
        problem.add_pair(pair)
    options = native.TrackEstablishmentOptions()
    options.min_num_views_per_track = 2
    options.intra_image_consistency_threshold = 10.0

    tracks = native.establish_full_tracks(
        problem,
        [1, 2, 3],
        [pair.pair_id for pair in pairs],
        options,
    )
    assert tracks == []


def test_greedy_track_quota_retains_inclusive_comparison():
    problem = mapping_problem(3, 2)
    matches = [(0, 0), (1, 1)]
    pairs = [
        pair_record(1, 2, matches),
        pair_record(1, 3, matches),
        pair_record(2, 3, matches),
    ]
    for pair in pairs:
        problem.add_pair(pair)
    options = native.TrackEstablishmentOptions()
    options.required_tracks_per_view = 0

    tracks = native.establish_full_tracks(
        problem,
        [1, 2, 3],
        [pair.pair_id for pair in pairs],
        options,
    )
    assert [track.point3D_id for track in tracks] == [(1 << 32) | 1]


def test_loop_closure_second_pass_keeps_exact_shared_endpoint_match_out():
    problem = mapping_problem(3, 1)
    loop_pair = pair_record(1, 2, [(0, 0)], loop_closure_rows=(0,))
    regular_pair = pair_record(1, 3, [(0, 0)])
    problem.add_pair(loop_pair)
    problem.add_pair(regular_pair)
    options = native.TrackEstablishmentOptions()
    options.min_num_views_per_track = 1

    tracks = native.establish_full_tracks(
        problem,
        [1, 2, 3],
        [loop_pair.pair_id, regular_pair.pair_id],
        options,
        True,
    )

    assert len(tracks) == 1
    assert observation_set(tracks[0]) == {(1, 0), (3, 0)}
    assert observation_set(tracks[0], loop_closure=True) == {(2, 0)}
    np.testing.assert_array_equal(tracks[0].loop_closure_anchors, [[1, 0]])


def test_loop_closure_orphans_create_reciprocal_tracks():
    problem = mapping_problem(2, 1)
    pair = pair_record(1, 2, [(0, 0)], loop_closure_rows=(0,))
    problem.add_pair(pair)

    options = native.TrackEstablishmentOptions()
    options.min_num_views_per_track = 1
    tracks = native.establish_full_tracks(problem, [1, 2], [pair.pair_id], options, True)
    by_id = track_by_id(tracks)
    id1 = 1 << 32
    id2 = 2 << 32
    assert observation_set(by_id[id1]) == {(1, 0)}
    assert observation_set(by_id[id1], loop_closure=True) == {(2, 0)}
    np.testing.assert_array_equal(by_id[id1].loop_closure_anchors, [[1, 0]])
    assert observation_set(by_id[id2]) == {(2, 0)}
    assert observation_set(by_id[id2], loop_closure=True) == {(1, 0)}
    np.testing.assert_array_equal(by_id[id2].loop_closure_anchors, [[2, 0]])


def test_loop_closure_collisions_preserve_regular_components_and_anchors():
    problem = mapping_problem(10, 1)
    pair_specs = [
        (1, 2, False),
        (2, 3, False),
        (4, 5, False),
        (6, 7, False),
        (3, 8, True),
        (5, 6, True),
        (9, 10, True),
        (1, 3, True),
    ]
    pairs = [
        pair_record(image_id1, image_id2, [(0, 0)], loop_closure_rows=((0,) if is_lc else ()))
        for image_id1, image_id2, is_lc in pair_specs
    ]
    for pair in pairs:
        problem.add_pair(pair)

    options = native.TrackEstablishmentOptions()
    options.min_num_views_per_track = 2
    tracks = native.establish_full_tracks(
        problem,
        list(range(1, 11)),
        [pair.pair_id for pair in pairs],
        options,
        True,
    )
    by_observations = {frozenset(observation_set(track)): track for track in tracks}

    assert set(by_observations) == {
        frozenset({(1, 0), (2, 0), (3, 0)}),
        frozenset({(4, 0), (5, 0)}),
        frozenset({(6, 0), (7, 0)}),
        frozenset({(9, 0)}),
        frozenset({(10, 0)}),
    }
    assert observation_set(by_observations[frozenset({(1, 0), (2, 0), (3, 0)})], loop_closure=True) == {(8, 0)}
    assert observation_set(by_observations[frozenset({(4, 0), (5, 0)})], loop_closure=True) == {(6, 0)}
    assert observation_set(by_observations[frozenset({(6, 0), (7, 0)})], loop_closure=True) == {(5, 0)}


def make_track(track_id, observations, *, loop_closure_observations=()):
    track = native.TrackRecord()
    track.point3D_id = track_id
    track.observations = np.asarray(observations, dtype=np.uint32).reshape(-1, 2)
    track.loop_closure_observations = np.asarray(loop_closure_observations, dtype=np.uint32).reshape(-1, 2)
    return track


def test_problem_filter_applies_two_view_depth_gate():
    problem = mapping_problem(2, 1, valid_depth=True)
    image = image_record(1, 1, valid_depth=False)
    problem.update_image(image)
    track = make_track(0, [(1, 0), (2, 0)])
    options = native.TrackProblemFilterOptions()
    options.min_num_views_per_track = 2
    options.two_view_depth_gate = True

    assert native.filter_tracks_for_problem(problem, [1, 2], [track], options) == []


def test_problem_filter_does_not_count_loop_closure_as_regular_view():
    problem = mapping_problem(2, 1)
    track = make_track(0, [(1, 0)], loop_closure_observations=[(2, 0)])
    options = native.TrackProblemFilterOptions()
    options.min_num_views_per_track = 2

    assert native.filter_tracks_for_problem(problem, [1, 2], [track], options) == []


def posed_problem():
    problem = mapping_problem(2, 1)
    for image_id, translation in ((1, [0.0, 0.0, 0.0]), (2, [-1.0, 0.0, 0.0])):
        image = problem.image(image_id)
        pose = native.PoseRecord()
        pose.has_pose = True
        pose.translation = np.asarray(translation)
        image.pose = pose
        problem.update_image(image)
    return problem


def test_angle_filter_removes_only_inconsistent_regular_observations():
    problem = posed_problem()
    image = problem.image(2)
    image.bearings = np.array([[1.0, 0.0, 0.0]])
    problem.update_image(image)
    track = make_track(7, [(1, 0), (2, 0)], loop_closure_observations=[(2, 0)])
    track.xyz = np.array([0.0, 0.0, 5.0])

    result = native.filter_tracks_by_angle(problem, [track], 1.0)
    assert result.counter == 1
    assert observation_set(result.tracks[0]) == {(1, 0)}
    assert observation_set(result.tracks[0], loop_closure=True) == {(2, 0)}


def test_triangulation_filter_clears_regular_observations_only():
    problem = posed_problem()
    track = make_track(7, [(1, 0), (2, 0)], loop_closure_observations=[(2, 0)])
    track.xyz = np.array([0.0, 0.0, 5.0])

    result = native.filter_tracks_by_triangulation_angle(problem, [track], 20.0)
    assert result.counter == 1
    assert result.tracks[0].observations.shape == (0, 2)
    assert observation_set(result.tracks[0], loop_closure=True) == {(2, 0)}


def test_loop_closure_second_pass_requires_aligned_metadata():
    problem = mapping_problem(2, 1)
    pair = pair_record(1, 2, [(0, 0)])
    pair.are_loop_closure = np.empty(0, dtype=np.uint8)
    problem.add_pair(pair)

    with pytest.raises(ValueError, match="loop-closure mask"):
        native.establish_full_tracks(
            problem,
            [1, 2],
            [pair.pair_id],
            native.TrackEstablishmentOptions(),
            True,
        )
