import numpy as np
import pycolmap
import pytest
import vidmap_native._core as native


@pytest.mark.parametrize("observations", [[], [(9, 2), (3, 0)]])
def test_point_constructor_preserves_bits_order_ids_and_owns_copies(observations):
    point = pycolmap.Point3D(
        xyz=[1.25, -0.0, np.nextafter(1.0, 2.0)],
        color=[17, 23, 31],
        error=np.nextafter(0.125, 1.0),
        track=pycolmap.Track([pycolmap.TrackElement(*element) for element in observations]),
    )
    record = native.TrackRecord(2**63 + 1, point)
    assert record.point3D_id == 2**63 + 1
    for field in ["xyz", "color", "error"]:
        expected = np.asarray(getattr(point, field))
        actual = np.asarray(getattr(record, field))
        assert actual.dtype == expected.dtype
        assert actual.tobytes() == expected.tobytes()
    expected_observations = np.array(observations, dtype=np.uint32).reshape((-1, 2))
    np.testing.assert_array_equal(record.observations, expected_observations)
    assert record.observations.dtype == expected_observations.dtype
    assert record.loop_closure_observations.shape == (0, 2)
    assert record.loop_closure_anchors.shape == (0, 2)

    point.xyz = [9.0, 8.0, 7.0]
    if observations:
        point.track.delete_element(0)
    np.testing.assert_array_equal(record.xyz, [1.25, -0.0, np.nextafter(1.0, 2.0)])
    np.testing.assert_array_equal(record.observations, expected_observations)


def test_point_constructor_retains_validation_at_insertion():
    point = pycolmap.Point3D(xyz=[np.nan, 0.0, 1.0])
    record = native.TrackRecord(5, point)
    assert np.isnan(record.xyz[0])
    with pytest.raises(ValueError):
        native.MappingProblem().add_track(record)
