import numpy as np
import pycolmap
import vidmap_native._core as native
from test_records import scene


def test_bulk_geometry_preserves_ids_bits_and_observation_order():
    rec, _, _ = scene(2, 3)
    point = pycolmap.Point3D(
        xyz=[1.25, -0.0, np.nextafter(1.0, 2.0)],
        track=pycolmap.Track([pycolmap.TrackElement(2, 2), pycolmap.TrackElement(1, 0)]),
    )
    rec.add_point3D_with_id(2**63 + 1, point)
    ids, xyz, lengths = native.point3D_table(rec)
    assert ids.tolist() == [2**63 + 1]
    assert xyz[0].tobytes() == point.xyz.tobytes()
    assert lengths.tolist() == [2]
    assert native.image_point3D_ids(rec.image(2))[2] == 2**63 + 1
    np.testing.assert_array_equal(native.point2D_coords(rec.image(1)), [[0, 1], [2, 3], [4, 5]])


def test_publish_geometry_updates_canonical_values_without_replacing_tracks():
    rec, _, _ = scene()
    rec.add_point3D([1, 2, 3], pycolmap.Track([pycolmap.TrackElement(1, 0), pycolmap.TrackElement(2, 0)]))
    source = pycolmap.Reconstruction(rec)
    point_id = next(iter(rec.points3D))
    source.point3D(point_id).xyz = [4, 5, 6]
    source.camera(1).params = [600, 600, 320, 240]
    source.image(1).frame.rig_from_world = pycolmap.Rigid3d(translation=[1, 2, 3])
    native.publish_geometry(source, rec)
    np.testing.assert_array_equal(rec.point3D(point_id).xyz, [4, 5, 6])
    assert rec.point3D(point_id).track.length() == 2
    np.testing.assert_array_equal(rec.camera(1).params, [600, 600, 320, 240])
    np.testing.assert_array_equal(rec.image(1).cam_from_world().translation, [1, 2, 3])
