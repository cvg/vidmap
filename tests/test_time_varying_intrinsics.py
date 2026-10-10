"""Tests for time-varying camera intrinsics modeling and temporal focal smoothing."""

from __future__ import annotations

import tempfile
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pycolmap
import pytest
import vidmap_native._core as native
from PIL import Image

from vidmap.datasets.local import LocalImageParser
from vidmap.frontend.options.preparation import CameraPriorEstimationOptions
from vidmap.frontend.preparation.camera_priors import apply_camera_priors
from vidmap.mapper.focal_prior import load_focal_prior
from vidmap.utils.camera_smoothing import smooth_temporal_focals


def test_smooth_temporal_focals_outlier_rejection():
    # Constant focal sequence with an extreme isolated spike.
    raw = [600.0, 600.0, 600.0, 1800.0, 600.0, 600.0, 600.0]
    smoothed = smooth_temporal_focals(raw, window_size=5, gaussian_sigma=1.0)
    assert len(smoothed) == len(raw)
    # The 1800 spike should be completely eliminated.
    assert np.all(smoothed < 650.0)
    assert np.all(smoothed > 550.0)
    assert np.isclose(smoothed[3], 600.0, atol=10.0)


def test_smooth_temporal_focals_zooming_trajectory():
    # Linear zoom from 500 to 1000 over 15 frames with 2 random outliers.
    t = np.linspace(0, 1, 15)
    gt_zoom = 500.0 + 500.0 * t
    noisy_zoom = gt_zoom.copy()
    noisy_zoom[4] = 1600.0  # outlier spike
    noisy_zoom[10] = 300.0  # outlier dip

    smoothed = smooth_temporal_focals(noisy_zoom, window_size=5, gaussian_sigma=1.0)
    # Outliers should be suppressed and smoothed should stay close to ground truth.
    max_err = np.max(np.abs(smoothed - gt_zoom))
    raw_max_err = np.max(np.abs(noisy_zoom - gt_zoom))
    assert max_err < 0.25 * raw_max_err
    assert smoothed[4] < 1000.0
    assert smoothed[10] > 600.0


def test_smooth_temporal_focals_edge_cases():
    # Empty or short sequences.
    assert len(smooth_temporal_focals([])) == 0
    single = np.array([500.0])
    np.testing.assert_array_equal(smooth_temporal_focals(single), single)
    pair = np.array([500.0, 600.0])
    np.testing.assert_array_equal(smooth_temporal_focals(pair), pair)

    # Invalid values.
    with pytest.raises(ValueError, match="positive and finite"):
        smooth_temporal_focals([500.0, -100.0, 500.0])
    with pytest.raises(ValueError, match="positive and finite"):
        smooth_temporal_focals([500.0, np.nan, 500.0])
    with pytest.raises(ValueError, match="1D sequence"):
        smooth_temporal_focals(np.ones((3, 3)))


def test_camera_prior_options_validation():
    # Valid time-varying config.
    opts = CameraPriorEstimationOptions(
        estimator="da3", inference="per_view", time_varying=True
    )
    assert opts.time_varying is True

    # time_varying requires per_view inference.
    with pytest.raises(
        ValueError, match="Time-varying camera priors require per_view inference"
    ):
        CameraPriorEstimationOptions(
            estimator="geocalib", inference="selected_batch", time_varying=True
        )

    # time_varying requires an estimator.
    with pytest.raises(
        ValueError, match="Time-varying camera priors require an estimator"
    ):
        CameraPriorEstimationOptions(
            estimator="none",
            initialization="supplied",
            inference="per_view",
            time_varying=True,
        )


def test_local_image_parser_time_varying_cameras():
    with tempfile.TemporaryDirectory() as tmpdir:
        image_dir = Path(tmpdir)
        imnames = ["frame_001.jpg", "frame_002.jpg", "frame_003.jpg"]
        for name in imnames:
            img = Image.new("RGB", (640, 480), color=(100, 100, 100))
            img.save(image_dir / name)

        # Standard uncalibrated parser (shared camera).
        parser_shared = LocalImageParser(
            image_dir=image_dir,
            imnames=imnames,
            estimate_intrinsics=True,
            time_varying_intrinsics=False,
        )
        assert len(parser_shared.rec.cameras) == 1
        for img in parser_shared.rec.images.values():
            assert img.camera_id == 1

        # Time-varying uncalibrated parser (per-frame cameras).
        parser_varying = LocalImageParser(
            image_dir=image_dir,
            imnames=imnames,
            estimate_intrinsics=True,
            time_varying_intrinsics=True,
        )
        assert len(parser_varying.rec.cameras) == 3
        camera_ids = {img.camera_id for img in parser_varying.rec.images.values()}
        assert camera_ids == {1, 2, 3}


def test_apply_camera_priors_time_varying():
    rec = pycolmap.Reconstruction()
    imnames = ["001.jpg", "002.jpg", "003.jpg"]
    for i, name in enumerate(imnames, start=1):
        cam = pycolmap.Camera.create_from_model_name(i, "PINHOLE", 1000.0, 640, 480)
        rec.add_camera_with_trivial_rig(cam)
        rec.add_image_with_trivial_frame(
            pycolmap.Image(name=name, camera_id=i, image_id=i)
        )

    # Raw results: 500, 1500 (outlier), 520
    results = [
        {
            "K": np.array([[500.0, 0, 320.0], [0, 500.0, 240.0], [0, 0, 1.0]]),
            "image_size": (640, 480),
        },
        {
            "K": np.array([[1500.0, 0, 320.0], [0, 1500.0, 240.0], [0, 0, 1.0]]),
            "image_size": (640, 480),
        },
        {
            "K": np.array([[520.0, 0, 320.0], [0, 520.0, 240.0], [0, 0, 1.0]]),
            "image_size": (640, 480),
        },
    ]

    apply_camera_priors(
        results=results,
        shared=False,
        reconstruction=rec,
        time_varying=True,
        names=imnames,
    )

    focal_cam1 = rec.cameras[1].focal_length_x
    focal_cam2 = rec.cameras[2].focal_length_x
    focal_cam3 = rec.cameras[3].focal_length_x

    # Frame 2 outlier (1500) should be smoothed down.
    assert focal_cam2 < 600.0
    assert np.isclose(focal_cam1, 500.0, atol=25.0)
    assert np.isclose(focal_cam3, 520.0, atol=25.0)


def test_load_focal_prior_time_varying(tmp_path):
    h5_path = tmp_path / "depth_maps.h5"
    imnames = ["001.jpg", "002.jpg", "003.jpg"]
    with h5py.File(h5_path, "w") as hfile:
        for name, f in zip(imnames, [500.0, 1500.0, 520.0], strict=True):
            grp = hfile.create_group(name)
            grp.create_dataset("image_size", data=np.array([640, 480]))
            grp.create_dataset(
                "K",
                data=np.array(
                    [[f, 0, 320.0], [0, f, 240.0], [0, 0, 1.0]], dtype=np.float32
                ),
            )
            grp.create_dataset("focal_std_px", data=np.array([10.0], dtype=np.float32))

    rec = pycolmap.Reconstruction()
    for i, name in enumerate(imnames, start=1):
        cam = pycolmap.Camera.create_from_model_name(i, "PINHOLE", 1000.0, 640, 480)
        rec.add_camera_with_trivial_rig(cam)
        rec.add_image_with_trivial_frame(
            pycolmap.Image(name=name, camera_id=i, image_id=i)
        )

    state = SimpleNamespace(reconstruction=rec, image_order=[1, 2, 3])
    # One raw (unsmoothed) observation per frame camera.
    prior = load_focal_prior(h5_path, state)
    assert set(prior.keys()) == {1, 2, 3}
    assert prior[2][0][0] == 1500.0


def test_bundle_adjustment_relative_focal_prior_cost():
    # The BA relative log-focal cost pulls an outlier focal toward its temporal neighbors.
    import pyceres
    from vidmap_native import bundle_adjustment as ba_costs

    cameras = [pycolmap.Camera.create_from_model_name(i, "PINHOLE", f, 640, 480) for i, f in [(1, 500.0), (2, 600.0), (3, 520.0)]]
    params = [np.array(camera.params) for camera in cameras]
    problem = pyceres.Problem()
    for camera, block, sigma in zip(cameras, params, (0.01, 0.5, 0.01)):
        focal = float(camera.mean_focal_length())
        problem.add_residual_block(ba_costs.focal_prior_cost(camera, focal, sigma), None, [block])
    for i, j in ((0, 1), (1, 2)):
        problem.add_residual_block(
            ba_costs.relative_focal_prior_cost(cameras[i], cameras[j], 0.0, 0.05), None, [params[i], params[j]]
        )
    for block in params:
        problem.set_manifold(block, pyceres.SubsetManifold(4, [2, 3]))
    summary = pyceres.SolverSummary()
    pyceres.solve(pyceres.SolverOptions(), problem, summary)
    assert summary.IsSolutionUsable()
    assert params[1][0] < 540.0 and abs(params[0][0] - 500.0) < 5.0 and abs(params[2][0] - 520.0) < 5.0
    assert params[1][0] == pytest.approx(params[1][1])  # fx and fy move together


def test_build_colmap_database_per_image_camera_policy(tmp_path):
    from vidmap.frontend.colmap_database import build_colmap_database

    features_path = tmp_path / "features.h5"
    with h5py.File(features_path, "w") as hfile:
        for name in ("001.jpg", "002.jpg"):
            grp = hfile.create_group(name)
            grp.create_dataset("keypoints", data=np.zeros((5, 2), dtype=np.float32))

    rec = pycolmap.Reconstruction()
    cam1 = pycolmap.Camera.create_from_model_name(1, "PINHOLE", 500.0, 640, 480)
    cam2 = pycolmap.Camera.create_from_model_name(2, "PINHOLE", 800.0, 640, 480)
    rec.add_camera_with_trivial_rig(cam1)
    rec.add_camera_with_trivial_rig(cam2)
    rec.add_image_with_trivial_frame(
        pycolmap.Image(name="001.jpg", camera_id=1, image_id=1)
    )
    rec.add_image_with_trivial_frame(
        pycolmap.Image(name="002.jpg", camera_id=2, image_id=2)
    )

    db_path = tmp_path / "test.db"
    build_colmap_database(
        db_path,
        rec,
        ["001.jpg", "002.jpg"],
        features_path,
        [("001.jpg", "002.jpg")],
        camera_policy="per_image",
        prior_focal_length=False,
        matches={("001.jpg", "002.jpg"): np.zeros((0, 2), dtype=np.uint32)},
    )

    db = pycolmap.Database.open(db_path)
    assert db.num_cameras() == 2
    assert db.read_camera(0).params[0] == 500.0
    assert db.read_camera(1).params[0] == 800.0
    assert db.read_image(1).camera_id == 0
    assert db.read_image(2).camera_id == 1
    db.close()


def test_native_relative_focal_priors_vgc():
    from vidmap.mapper.focal_prior import native_focal_priors, native_relative_focal_priors

    reconstruction = pycolmap.Reconstruction()
    sidecars = native.MappingSidecars()
    graph = pycolmap.PoseGraph()
    for cid, f in [(1, 500.0), (2, 600.0), (3, 520.0)]:
        reconstruction.add_camera_with_trivial_rig(pycolmap.Camera.create_from_model_name(cid, "PINHOLE", f, 640, 480))
        image = pycolmap.Image(image_id=cid, camera_id=cid, name=f"00{cid}.jpg", keypoints=np.zeros((2, 2)))
        reconstruction.add_image_with_trivial_frame(image, pycolmap.Rigid3d())
        data = native.ImageData()
        data.depth_values = np.ones(2)
        data.depth_stddevs = np.full(2, 0.1)
        data.depth_validity = np.ones(2, dtype=np.uint8)
        sidecars.add_image(cid, data)

    angle = 0.2
    rotation = np.array([[np.cos(angle), 0, np.sin(angle)], [0, 1, 0], [-np.sin(angle), 0, np.cos(angle)]])
    translation_skew = np.array([[0, -0.4, 0.3], [0.4, 0, -0.2], [-0.3, 0.2, 0]])
    inv_K = np.linalg.inv(np.array([[500.0, 0, 320.0], [0, 500.0, 240.0], [0, 0, 1]]))
    for id1, id2 in [(1, 2), (2, 3)]:
        pair = native.PairData()
        pair.all_matches = np.zeros((0, 2), dtype=np.uint32)
        pair.inlier_indices = np.zeros(0, dtype=np.int32)
        pair.are_loop_closure = np.zeros(0, dtype=np.uint8)
        pair.geometry.config = 3  # UNCALIBRATED
        pair.geometry.F = inv_K.T @ translation_skew @ rotation @ inv_K
        sidecars.add_pair(pycolmap.image_pair_to_pair_id(id1, id2), pair)
        graph.add_edge(id1, id2, pycolmap.PoseGraphEdge())

    options = pycolmap.ViewGraphCalibrationOptions()
    options.min_focal_length_ratio = 0.01
    options.max_focal_length_ratio = 100.0
    unary = native_focal_priors({1: ((500.0, 0.1),), 2: ((600.0, 0.5),), 3: ((520.0, 0.1),)}, camera_ids=[1, 2, 3], loss="cauchy")
    relative = native_relative_focal_priors([(1, 2), (2, 3)], loss="cauchy", scale=0.05, weight=10.0)
    native.calibrate_focal_lengths(options, reconstruction, graph, sidecars, unary, relative)
    focals = {cid: reconstruction.camera(cid).mean_focal_length() for cid in (1, 2, 3)}
    # Camera 2 is pulled toward cameras 1 and 3 by the relative constraints.
    assert focals[2] < 540.0
    assert focals[1] > 490.0
    assert focals[3] < 530.0
