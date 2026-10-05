import numpy as np
import pytest
import vidmap_native._core as native
from test_records import add_pair, scene


def establish(owners, pair_ids, *, minimum=3, lc=False, depth_gate=False, capture=False):
    options = native.TrackEstablishmentOptions()
    options.min_num_views_per_track = minimum
    options.two_view_depth_gate = depth_gate
    return native.establish_tracks(*owners, list(owners[0].images), pair_ids, options, lc, True, capture)


def observations(track):
    return {(element.image_id, element.point2D_idx) for element in track.elements}


def loop_observations(data):
    return {tuple(map(int, row)) for row in data.loop_closure_observations}


def test_establishes_triangle_tracks():
    owners = scene(3, 5)
    pairs = [add_pair(owners, a, b, list(zip(range(5), range(5)))) for a, b in [(1, 2), (1, 3), (2, 3)]]
    result = establish(owners, pairs)
    assert result.num_tracks == 5
    assert result.full_tracks == {}
    assert {frozenset(observations(point.track)) for point in owners[0].points3D.values()} == {
        frozenset((image_id, feature) for image_id in (1, 2, 3)) for feature in range(5)
    }
    assert all(point.track.length() == 3 for point in owners[0].points3D.values())


def test_intra_image_inconsistency_drops_fused_track():
    owners = scene(3, 2)
    owners[0].image(1).point2D(0).xy = [0, 0]
    owners[0].image(1).point2D(1).xy = [100, 100]
    pairs = [add_pair(owners, 1, 2, [(0, 0)]), add_pair(owners, 2, 3, [(0, 0)]), add_pair(owners, 1, 3, [(1, 0)])]
    assert establish(owners, pairs, minimum=2).num_tracks == 0


def test_loop_closure_second_pass_keeps_exact_shared_endpoint_match_out():
    owners = scene(3, 1)
    pairs = [add_pair(owners, 1, 2, [(0, 0)], [0]), add_pair(owners, 1, 3, [(0, 0)])]
    establish(owners, pairs, minimum=1, lc=True)
    assert owners[0].num_points3D() == 1
    point_id = next(iter(owners[0].points3D))
    assert observations(owners[0].point3D(point_id).track) == {(1, 0), (3, 0)}
    assert loop_observations(owners[2].track(point_id)) == {(2, 0)}
    np.testing.assert_array_equal(owners[2].track(point_id).loop_closure_anchors, [[1, 0]])


def test_loop_closure_orphans_create_reciprocal_tracks():
    owners = scene(2, 1)
    establish(owners, [add_pair(owners, 1, 2, [(0, 0)], [0])], minimum=1, lc=True)
    by_observations = {
        frozenset(observations(point.track)): point_id for point_id, point in owners[0].points3D.items()
    }
    assert set(by_observations) == {frozenset({(1, 0)}), frozenset({(2, 0)})}
    for own, other in [(1, 2), (2, 1)]:
        point_id = by_observations[frozenset({(own, 0)})]
        assert loop_observations(owners[2].track(point_id)) == {(other, 0)}
        np.testing.assert_array_equal(owners[2].track(point_id).loop_closure_anchors, [[own, 0]])


def test_loop_closure_collisions_preserve_regular_components_and_anchors():
    owners = scene(10, 1)
    specs = [
        (1, 2, False),
        (2, 3, False),
        (4, 5, False),
        (6, 7, False),
        (3, 8, True),
        (5, 6, True),
        (9, 10, True),
        (1, 3, True),
    ]
    pairs = [add_pair(owners, a, b, [(0, 0)], [0] if lc else []) for a, b, lc in specs]
    result = establish(owners, pairs, minimum=2, lc=True, capture=True)
    by_observations = {frozenset(observations(track)): pid for pid, track in result.full_tracks.items()}
    assert set(by_observations) == {
        frozenset({(1, 0), (2, 0), (3, 0)}),
        frozenset({(4, 0), (5, 0)}),
        frozenset({(6, 0), (7, 0)}),
        frozenset({(9, 0)}),
        frozenset({(10, 0)}),
    }
    for regular, expected in [
        ({(1, 0), (2, 0), (3, 0)}, {(8, 0)}),
        ({(4, 0), (5, 0)}, {(6, 0)}),
        ({(6, 0), (7, 0)}, {(5, 0)}),
    ]:
        assert loop_observations(result.full_track_data[by_observations[frozenset(regular)]]) == expected
    assert result.num_tracks == 3


def test_problem_filter_applies_two_view_depth_gate():
    owners = scene(2, 1)
    owners[2].image(1).depth_validity = np.zeros(1, dtype=np.uint8)
    result = establish(owners, [add_pair(owners, 1, 2, [(0, 0)])], minimum=2, depth_gate=True)
    assert result.num_full_tracks == 1
    assert result.num_tracks == 0


def test_problem_filter_does_not_count_loop_closure_as_regular_view():
    owners = scene(2, 1)
    result = establish(owners, [add_pair(owners, 1, 2, [(0, 0)], [0])], minimum=2, lc=True)
    assert result.num_full_tracks == 2
    assert result.num_tracks == 0


def test_loop_closure_second_pass_requires_aligned_metadata():
    owners = scene(2, 1)
    pid = add_pair(owners, 1, 2, [(0, 0)])
    owners[2].pair(pid).are_loop_closure = np.empty(0, dtype=np.uint8)
    with pytest.raises(ValueError, match="loop-closure mask"):
        establish(owners, [pid], minimum=1, lc=True)


def test_ba_graph_keeps_inliers_from_invalid_pairs():
    owners = scene(2, 1)
    pid = add_pair(owners, 1, 2, [(0, 0)])
    owners[1].edges[pid].valid = False
    graph = native.create_correspondence_graph(owners[0], owners[2], [pid])
    assert graph.num_matches_between_images(1, 2) == 1
