import numpy as np
import pytest
import vidmap_native._core as native


def test_native_api_version_is_explicit():
    assert native.__api_version__ == 5


def test_solver_backend_capabilities_are_explicit():
    assert native.__ceres_version__.count(".") == 2
    assert isinstance(native.__cuda_dense_solver_available__, bool)
    assert isinstance(native.__cuda_sparse_solver_available__, bool)
    assert set(native.LinearSolverType.__members__) == {
        "DENSE_SCHUR",
        "SPARSE_SCHUR",
        "ITERATIVE_SCHUR",
    }


def test_global_positioning_ordering_names_are_current():
    assert set(native.GlobalPositioningOrdering.__members__) == {"GROUPED", "SINGLETON"}


def camera_record(camera_id=1):
    camera = native.CameraRecord()
    camera.camera_id = camera_id
    camera.model_id = 1  # PINHOLE
    camera.width = 640
    camera.height = 480
    camera.params = np.array([500.0, 500.0, 320.0, 240.0])
    return camera


def image_record(image_id, camera_id=1, num_features=3):
    image = native.ImageRecord()
    image.image_id = image_id
    image.camera_id = camera_id
    image.frame_id = image_id
    image.name = f"image-{image_id}.jpg"
    image.keypoints = np.arange(num_features * 2, dtype=float).reshape(-1, 2)
    image.depth_values = np.ones(num_features)
    image.depth_stddevs = np.full(num_features, 0.1)
    image.depth_validity = np.ones(num_features, dtype=np.uint8)
    return image


def pair_record(image_id1=1, image_id2=2):
    pair = native.PairRecord()
    pair.image_id1 = image_id1
    pair.image_id2 = image_id2
    pair.pair_id = min(image_id1, image_id2) * 2147483647 + max(image_id1, image_id2)
    pair.all_matches = np.array([[0, 1], [1, 2]], dtype=np.uint32)
    pair.inlier_indices = np.array([0], dtype=np.int32)
    pair.are_loop_closure = np.array([0, 1], dtype=np.uint8)
    return pair


def test_mapping_problem_owns_records_and_orders_ids():
    problem = native.MappingProblem()
    problem.add_camera(camera_record())
    problem.add_image(image_record(2))
    problem.add_image(image_record(1))
    pair = pair_record()
    problem.add_pair(pair)

    pair.all_matches = np.array([[2, 1], [1, 2]], dtype=np.uint32)
    assert problem.image_ids == [1, 2]
    assert problem.pair(pair.pair_id).all_matches[0, 0] == 0
    problem.validate()


def test_mapping_problem_clears_tracks():
    problem = native.MappingProblem()
    problem.add_camera(camera_record())
    problem.add_image(image_record(1))
    track = native.TrackRecord()
    track.point3D_id = 7
    track.observations = np.array([[1, 0]], dtype=np.uint32)
    problem.add_track(track)

    problem.clear_tracks()

    assert problem.num_tracks == 0
    assert problem.point3D_ids == []


def test_track_error_uses_colmap_default():
    assert native.TrackRecord().error == -1.0


def test_pair_record_rejects_misaligned_loop_closure_mask():
    pair = pair_record()
    pair.are_loop_closure = np.ones(1, dtype=np.uint8)
    with pytest.raises(ValueError, match="loop-closure mask"):
        pair.validate()


def test_problem_rejects_feature_index_outside_image():
    problem = native.MappingProblem()
    problem.add_camera(camera_record())
    problem.add_image(image_record(1, num_features=1))
    problem.add_image(image_record(2, num_features=1))
    pair = pair_record()
    problem.add_pair(pair)
    with pytest.raises(ValueError, match="missing feature"):
        problem.validate()


def test_problem_rejects_duplicate_ids():
    problem = native.MappingProblem()
    problem.add_camera(camera_record())
    with pytest.raises(ValueError, match="already exists"):
        problem.add_camera(camera_record())


def test_native_classes_are_not_colmap_types():
    classes = (
        native.CameraRecord,
        native.ImageRecord,
        native.PairRecord,
        native.TrackRecord,
        native.MappingProblem,
    )
    assert all(cls.__module__ == "vidmap_native._core" for cls in classes)
    assert all("colmap" not in cls.__name__.lower() for cls in classes)
