import numpy as np
import pycolmap
import pytest
import vidmap_native._core as native


def camera(camera_id=1, has_prior=False):
    result = pycolmap.Camera(
        camera_id=camera_id, model="PINHOLE", width=640, height=480, params=[500.0, 500.0, 320.0, 240.0]
    )
    result.has_prior_focal_length = has_prior
    return result


def scene(num_images=2, num_features=3):
    reconstruction = pycolmap.Reconstruction()
    reconstruction.add_camera_with_trivial_rig(camera())
    sidecars = native.MappingSidecars()
    graph = pycolmap.PoseGraph()
    for image_id in range(1, num_images + 1):
        image = pycolmap.Image(
            image_id=image_id,
            camera_id=1,
            name=f"image-{image_id}.jpg",
            keypoints=np.arange(num_features * 2, dtype=float).reshape(-1, 2),
        )
        reconstruction.add_image_with_trivial_frame(image, pycolmap.Rigid3d())
        data = native.ImageData()
        data.depth_values = np.ones(num_features)
        data.depth_stddevs = np.full(num_features, 0.1)
        data.depth_validity = np.ones(num_features, dtype=np.uint8)
        sidecars.add_image(image_id, data)
    return reconstruction, graph, sidecars


def add_pair(owners, first=1, second=2, matches=((0, 1), (1, 2)), loop_rows=()):
    _, graph, sidecars = owners
    pair = native.PairData()
    pair.all_matches = np.asarray(matches, dtype=np.uint32).reshape(-1, 2)
    pair.inlier_indices = np.arange(len(matches), dtype=np.int32)
    mask = np.zeros(len(matches), dtype=np.uint8)
    mask[list(loop_rows)] = 1
    pair.are_loop_closure = mask
    pair_id = pycolmap.image_pair_to_pair_id(first, second)
    sidecars.add_pair(pair_id, pair)
    graph.add_edge(first, second, pycolmap.PoseGraphEdge())
    return pair_id


def test_metadata_mutates_without_duplicating_scene_geometry():
    rec, _, data = scene()
    data.image(1).depth_values = np.full(3, 2.0)
    np.testing.assert_array_equal(data.image(1).depth_values, 2.0)
    data.validate(rec)


def test_metadata_checks_feature_alignment():
    rec, _, data = scene()
    data.image(1).depth_stddevs = np.ones(2)
    with pytest.raises(ValueError, match="feature-aligned"):
        data.validate(rec)


def test_pair_rejects_misaligned_loop_closure_mask():
    owners = scene()
    pair_id = add_pair(owners)
    pair = owners[2].pair(pair_id)
    pair.are_loop_closure = np.ones(1, dtype=np.uint8)
    with pytest.raises(ValueError, match="loop-closure mask"):
        pair.validate()


def test_metadata_rejects_feature_index_outside_image():
    owners = scene(num_features=1)
    add_pair(owners)
    with pytest.raises(ValueError, match="unknown feature"):
        owners[2].validate(owners[0])


def test_release_workspace_retains_image_metadata():
    owners = scene()
    pair_id = add_pair(owners)
    owners[2].clear_pairs()
    with pytest.raises(IndexError):
        owners[2].pair(pair_id)
    assert len(owners[2].image(1).depth_values) == 3
