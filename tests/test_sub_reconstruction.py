"""Unit tests for covisibility components, sub-reconstruction extraction, and export."""

import tempfile
from pathlib import Path

import numpy as np
import pycolmap
import pytest

from vidmap.mapper.sub_reconstruction import (
    build_covisibility_graph,
    decompose_reconstruction,
    detect_covisibility_components,
    export_sub_reconstructions,
    extract_sub_reconstruction,
)


def _create_synthetic_reconstruction(
    num_images: int = 20,
    cut_after_image: int | None = None,
    points_per_pair: int = 15,
) -> pycolmap.Reconstruction:
    rec = pycolmap.Reconstruction()
    cam = pycolmap.Camera()
    cam.camera_id = 1
    cam.model = pycolmap.CameraModelId.PINHOLE
    cam.width = 640
    cam.height = 480
    cam.params = [500.0, 500.0, 320.0, 240.0]
    rec.add_camera_with_trivial_rig(cam)

    max_pts = num_images * points_per_pair * 2
    for i in range(1, num_images + 1):
        img = pycolmap.Image(image_id=i, name=f"frame_{i:04d}.jpg", camera_id=1)
        img.points2D = [
            pycolmap.Point2D(xy=np.array([100.0, 100.0])) for _ in range(max_pts)
        ]
        rec.add_image_with_trivial_frame(img)
        rec.frames[i].rig_from_world = pycolmap.Rigid3d()
        rec.register_frame(i)

    next_p2d_idx = {i: 0 for i in range(1, num_images + 1)}
    pid = 1

    for i in range(1, num_images):
        if cut_after_image is not None and i == cut_after_image:
            continue  # Sharp cut boundary: no shared points across this pair
        for _ in range(points_per_pair):
            track = pycolmap.Track()
            track.add_element(i, next_p2d_idx[i])
            track.add_element(i + 1, next_p2d_idx[i + 1])
            next_p2d_idx[i] += 1
            next_p2d_idx[i + 1] += 1
            pt = pycolmap.Point3D(xyz=np.array([float(pid), 0.0, 5.0]), track=track)
            rec.add_point3D_with_id(pid, pt)
            pid += 1

    return rec


@pytest.fixture
def base_reconstruction() -> pycolmap.Reconstruction:
    """A continuous synthetic sequence of 20 images without cuts."""
    return _create_synthetic_reconstruction(num_images=20, cut_after_image=None)


@pytest.fixture
def two_shot_reconstruction() -> pycolmap.Reconstruction:
    """A synthetic sequence of 20 images with an artificial cut between frame 10 and 11."""
    return _create_synthetic_reconstruction(num_images=20, cut_after_image=10)


def test_build_covisibility_graph(two_shot_reconstruction):
    rec = two_shot_reconstruction
    edge_weights, adjacency = build_covisibility_graph(rec)

    reg_ids = sorted([iid for iid, img in rec.images.items() if img.has_pose])
    id_a = reg_ids[0]
    id_a_next = reg_ids[1]
    id_cut_left = reg_ids[9]
    id_cut_right = reg_ids[10]

    # Internal edges should have positive shared points
    assert edge_weights.get((min(id_a, id_a_next), max(id_a, id_a_next)), 0) > 0
    # Cross-cut edge must be 0
    assert (
        edge_weights.get(
            (min(id_cut_left, id_cut_right), max(id_cut_left, id_cut_right)), 0
        )
        == 0
    )


def test_detect_covisibility_components(two_shot_reconstruction):
    rec = two_shot_reconstruction
    components = detect_covisibility_components(
        rec,
        min_shared_points=10,
        min_component_size=5,
    )
    assert len(components) == 2
    assert len(components[0]) == 10
    assert len(components[1]) == 10


def test_extract_sub_reconstruction(two_shot_reconstruction):
    rec = two_shot_reconstruction
    reg_ids = sorted([iid for iid, img in rec.images.items() if img.has_pose])
    shot_a_ids = reg_ids[:10]

    sub_a = extract_sub_reconstruction(rec, shot_a_ids)

    assert sub_a.num_images() == 10
    assert sub_a.num_reg_images() == 10
    assert sub_a.num_points3D() > 0

    # Ensure all track elements only reference images in Shot A
    for pt in sub_a.points3D.values():
        for el in pt.track.elements:
            assert el.image_id in shot_a_ids

    # Test serialization to disk and reload
    with tempfile.TemporaryDirectory() as tmpdir:
        sub_a.write(tmpdir)
        reloaded = pycolmap.Reconstruction(tmpdir)
        assert reloaded.num_reg_images() == 10
        assert reloaded.num_points3D() == sub_a.num_points3D()


def test_decompose_reconstruction(two_shot_reconstruction):
    rec = two_shot_reconstruction
    sub_models = decompose_reconstruction(
        rec,
        min_shared_points=10,
        min_model_size=5,
    )
    assert len(sub_models) == 2
    assert sub_models[0].num_reg_images() == 10
    assert sub_models[1].num_reg_images() == 10


def test_export_sub_reconstructions(two_shot_reconstruction):
    rec = two_shot_reconstruction
    sub_models = decompose_reconstruction(rec, min_shared_points=10, min_model_size=5)

    with tempfile.TemporaryDirectory() as tmpdir:
        paths = export_sub_reconstructions(sub_models, tmpdir)
        assert len(paths) == 2
        assert paths[0] == Path(tmpdir) / "rec" / "0"
        assert paths[1] == Path(tmpdir) / "rec" / "1"

        # Verify both sub-models reload cleanly
        rec0 = pycolmap.Reconstruction(paths[0])
        rec1 = pycolmap.Reconstruction(paths[1])
        assert rec0.num_reg_images() == 10
        assert rec1.num_reg_images() == 10


def test_run_options_cut_splitting():
    from vidmap.run_options import RunOptions

    opts = RunOptions()
    assert not opts.split_cuts
    assert opts.min_covisibility_points == 20
    assert opts.min_model_size == 5

    custom_opts = RunOptions(
        split_cuts=True,
        min_covisibility_points=30,
        min_model_size=8,
    )
    assert custom_opts.split_cuts
    assert custom_opts.min_covisibility_points == 30
    assert custom_opts.min_model_size == 8


def test_run_parser_cut_splitting_arguments():
    from vidmap.run import build_parser

    parser = build_parser()
    args = parser.parse_args(
        [
            "--input_data",
            "test.mp4",
            "--output",
            "test_run",
            "--split-cuts",
            "--min-covisibility-points",
            "25",
            "--min-model-size",
            "12",
            "--html",
        ]
    )
    assert args.split_cuts is True
    assert args.min_covisibility_points == 25
    assert args.min_model_size == 12
    assert args.html is True


def test_viewer_sub_reconstruction_export(two_shot_reconstruction):
    from vidmap.visualization.html.cli import main as html_main
    from vidmap.visualization.html.embedded import write_all_embedded_viewers

    sub_models = decompose_reconstruction(
        two_shot_reconstruction, min_shared_points=10, min_model_size=5
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_run = Path(tmpdir)
        export_sub_reconstructions(sub_models, tmp_run)

        # Test write_all_embedded_viewers default
        written = write_all_embedded_viewers(tmp_run)
        assert len(written) == 2
        assert written[0].name == "vidmap-viewer-embedded-sub0.html"
        assert written[1].name == "vidmap-viewer-embedded-sub1.html"
        for w in written:
            assert w.is_file()
            content = w.read_text(encoding="utf-8")
            assert "VidMap reconstruction" in content

        # Test HTML CLI with custom output name
        custom_out = tmp_run / "custom_output.html"
        ret = html_main(["--run-dir", str(tmp_run), "-o", str(custom_out)])
        assert ret == 0
        out_sub0 = tmp_run / "custom_output_sub0.html"
        out_sub1 = tmp_run / "custom_output_sub1.html"
        assert out_sub0.is_file()
        assert out_sub1.is_file()


def test_single_model_export_and_viewer(base_reconstruction):
    from vidmap.visualization.html.cli import main as html_main
    from vidmap.visualization.html.embedded import write_all_embedded_viewers

    sub_models = decompose_reconstruction(
        base_reconstruction, min_shared_points=10, min_model_size=5
    )
    assert len(sub_models) == 1  # Base reconstruction has no cuts

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_run = Path(tmpdir)
        paths = export_sub_reconstructions(sub_models, tmp_run)
        assert len(paths) == 1
        assert paths[0] == tmp_run / "rec"

        written = write_all_embedded_viewers(tmp_run)
        assert len(written) == 1
        assert written[0].name == "vidmap-viewer-embedded.html"
        assert written[0].is_file()

        # HTML CLI test
        custom_out = tmp_run / "single.html"
        ret = html_main(["--run-dir", str(tmp_run), "-o", str(custom_out)])
        assert ret == 0
        assert custom_out.is_file()


def test_extract_sub_reconstruction_time_varying_intrinsics():
    """Verify that reconstructions with per-image cameras and rigs decompose cleanly."""
    rec = pycolmap.Reconstruction()
    num_images = 10
    for i in range(1, num_images + 1):
        cam = pycolmap.Camera()
        cam.camera_id = i
        cam.model = pycolmap.CameraModelId.PINHOLE
        cam.width = 640
        cam.height = 480
        cam.params = [500.0 + i * 10, 500.0 + i * 10, 320.0, 240.0]
        rec.add_camera_with_trivial_rig(cam)

        img = pycolmap.Image(image_id=i, name=f"frame_{i:04d}.jpg", camera_id=i)
        img.points2D = [
            pycolmap.Point2D(xy=np.array([100.0, 100.0])) for _ in range(50)
        ]
        rec.add_image_with_trivial_frame(img)
        rec.frames[i].rig_from_world = pycolmap.Rigid3d()
        rec.register_frame(i)

    next_p2d_idx = {i: 0 for i in range(1, num_images + 1)}
    pid = 1
    for i in range(1, num_images):
        if i == 5:
            continue  # Cut between 5 and 6
        for _ in range(15):
            track = pycolmap.Track()
            track.add_element(i, next_p2d_idx[i])
            track.add_element(i + 1, next_p2d_idx[i + 1])
            next_p2d_idx[i] += 1
            next_p2d_idx[i + 1] += 1
            pt = pycolmap.Point3D(xyz=np.array([float(pid), 0.0, 5.0]), track=track)
            rec.add_point3D_with_id(pid, pt)
            pid += 1

    sub_models = decompose_reconstruction(rec, min_shared_points=10, min_model_size=3)
    assert len(sub_models) == 2
    assert sub_models[0].num_reg_images() == 5
    assert sub_models[1].num_reg_images() == 5


def test_detect_covisibility_components_pan_continuous(base_reconstruction):
    """Verify that continuous panning motions with lower overlap stay unified in one model."""
    components = detect_covisibility_components(
        base_reconstruction,
        min_shared_points=10,
        min_component_size=5,
    )
    assert len(components) == 1
    assert len(components[0]) == 20
