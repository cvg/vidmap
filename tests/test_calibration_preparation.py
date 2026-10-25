"""Calibration ordering, camera ownership, and finalized-input reuse."""

import json
import sqlite3
from types import SimpleNamespace

import numpy as np
import pycolmap
import pytest

from vidmap.calibration import CalibrationEvidence, CalibrationObservation
from vidmap.configuration.dump import config_to_dict
from vidmap.frontend.options.preparation import PreparationOptions
from vidmap.frontend.pipeline import Frontend
from vidmap.mapper.inputs import MapperInputs, write_mapper_inputs_manifest
from vidmap.mapper.mapper import Mapper
from vidmap.mapper.options import MapperOptions


@pytest.mark.parametrize("distorted,diagnostic", [(False, False), (True, False), (False, True), (True, True)])
def test_stage_order(monkeypatch, distorted, diagnostic):
    from vidmap.frontend import depth_projection
    from vidmap.mapper import runtime

    events = []
    tracking, depth = object(), object()
    verification = SimpleNamespace(database_path="database")
    refined = SimpleNamespace(cameras={})
    projections = object()
    frontend = object.__new__(Frontend)
    frontend.preparation_plan = SimpleNamespace(bootstrap_required=distorted)
    frontend.preparation = PreparationOptions()
    frontend.replay = SimpleNamespace(write_enabled=lambda stage: False)
    frontend.run_tracking = lambda: (events.append("tracking"), tracking)[1]
    frontend._verify = lambda state, **kw: (events.append("verification"), verification)[1]
    frontend._complete_camera_prediction = lambda state, result: (events.append("prediction"), tracking)[1]

    def estimate(state, **kwargs):
        assert state is tracking
        assert kwargs == ({"projections": projections} if distorted else {})
        events.append("depth")
        return depth

    def project(rec, cameras):
        assert rec is refined
        events.append("projection")
        return projections

    frontend._estimate_depth = estimate
    monkeypatch.setattr(runtime, "load_mapping_runtime", lambda: events.append("runtime"))
    monkeypatch.setattr(frontend, "_calibrate_distortion", lambda *args: (events.append("vgc"), refined)[1])
    monkeypatch.setattr(depth_projection, "resolve_depth_projections", project)
    result = frontend._execute_verified_boundary(pre_geom_db_stop=diagnostic)
    assert result[:2] == (tracking, verification)
    if distorted:
        expected = (
            ["tracking", "verification"]
            if diagnostic
            else ["runtime", "tracking", "verification", "vgc", "projection", "depth"]
        )
        assert result[2] is (None if diagnostic else depth)
    else:
        expected = ["tracking", "depth", "prediction", "verification"]
    assert events == expected


@pytest.mark.parametrize("distorted", [False, True])
def test_depth_uses_original_capture_or_refined_projection(monkeypatch, distorted):
    from vidmap.frontend import depth
    from vidmap.frontend.depth_projection import ForwardDepthProjection

    source, working = object(), object()
    camera = pycolmap.Camera(
        camera_id=1,
        model="OPENCV_FISHEYE",
        width=640,
        height=480,
        params=[400, 400, 320, 240, 0, 0, 0, 0],
    )
    projections = {1: ForwardDepthProjection.from_camera(camera)} if distorted else None
    keyframes = SimpleNamespace(names=("first.jpg", "second.jpg"), streams=(("first.jpg", "second.jpg"),))
    tracking = SimpleNamespace(capture=working, paths=object(), keyframes=keyframes, keyframe_sequence=keyframes.names)
    frontend = object.__new__(Frontend)
    frontend.preparation_plan = SimpleNamespace(
        bootstrap_required=distorted, camera_priors=SimpleNamespace(estimator="geocalib" if distorted else "da3")
    )
    frontend.scene_parser = source
    frontend.options = SimpleNamespace(depth=object())
    frontend.force_recompute = False
    frontend.cache_full_depth_maps = True
    frontend.device = "cpu"
    frontend.input_snapshot = SimpleNamespace(image_content_fingerprint=lambda names: "image-content")
    result = object()

    class DepthRecorder:
        def __init__(self, **kwargs):
            assert kwargs["scene_parser"] is (working if distorted else source)
            assert kwargs["keyframes"] is keyframes
            assert kwargs["depth_projections"] is projections
            assert kwargs["calibration_enabled"] is (not distorted)
            if distorted:
                assert kwargs["depth_projections"][1].metadata()["source"]["params"][:2] == [400, 400]

        def estimate(self, *, cache_full_depth_maps):
            assert cache_full_depth_maps
            return result

    monkeypatch.setattr(depth, "DepthEstimator", DepthRecorder)
    assert frontend._estimate_depth(tracking, projections=projections) is result


@pytest.mark.parametrize(
    "supplied,enabled,refine,valid",
    [
        (False, False, False, False),
        (False, False, True, False),
        (False, True, False, True),
        (False, True, True, True),
        (True, False, False, True),
        (True, True, False, False),
        (True, True, True, True),
    ],
)
def test_vgc_policy(supplied, enabled, refine, valid):
    mapper = object.__new__(Mapper)
    mapper.conf = MapperOptions(calibration={"vgc_enabled": enabled, "optimize_intrinsics": refine})
    mapper.mapper_inputs = SimpleNamespace(distortion_calibrated=lambda: False)
    rec = SimpleNamespace(cameras={1: SimpleNamespace(has_prior_focal_length=supplied)})
    if valid:
        mapper._validate_calibration_policy(rec)
    else:
        with pytest.raises(ValueError):
            mapper._validate_calibration_policy(rec)


def test_distorted_mapping_overrides_are_rejected():
    mapper = object.__new__(Mapper)
    mapper.conf = MapperOptions(vgc={"filter": {"enabled": False}})
    mapper.mapper_inputs = SimpleNamespace(distortion_calibrated=lambda: True)
    with pytest.raises(ValueError, match="frontend preparation.distortion_calibration"):
        mapper._validate_calibration_policy(SimpleNamespace(cameras={}))


def test_preparation_options_preserve_unaffected_default_identity():
    assert set(config_to_dict(PreparationOptions())) == {"lc", "geom_verif"}
    options = PreparationOptions(distortion_calibration={"max_distortion_pairs": 20})
    assert config_to_dict(options)["distortion_calibration"]["max_distortion_pairs"] == 20
    with pytest.raises(ValueError):
        MapperOptions(vgc={"calibration": {"max_distortion_pairs": 20}})


@pytest.mark.parametrize("distorted", [False, True])
def test_finalized_input_marker(tmp_path, distorted):
    for name, payload in [("database_complete.db", b""), ("depth_maps.h5", b""), ("lc_masks.json", b"{}")]:
        (tmp_path / name).write_bytes(payload)
    evidence = None
    if distorted:
        (tmp_path / "geocalib_batch.h5").write_bytes(b"")
        evidence = CalibrationEvidence(
            "geocalib",
            "selected_batch",
            "SIMPLE_DIVISIONAL",
            "working_images",
            (CalibrationObservation(1, "batch_calibration", None),),
        )
    identity = dict(
        schema_version=6,
        tag="test",
        config_name="test",
        config_fingerprint="a" * 64,
        colmap_runtime="stock",
        dataset="test",
        scene="test",
        mode="test",
        testset_id="test",
        reference_image_ids=[1, 2],
        source_fingerprint="b" * 64,
    )
    write_mapper_inputs_manifest(
        tmp_path,
        frontend_identity=identity,
        frame_sequence={"frame_ids": [1, 2], "timestamps_ns": None},
        calibration=evidence,
        distortion_calibrated=distorted,
    )
    inputs = MapperInputs.from_directory(tmp_path)
    inputs.validate()
    assert inputs.distortion_calibrated() is distorted
    if distorted:
        path = tmp_path / "manifest.json"
        manifest = json.loads(path.read_text())
        del manifest["distortion_calibrated"]
        path.write_text(json.dumps(manifest))
        with pytest.raises(ValueError, match="calibration before depth"):
            MapperInputs.from_directory(tmp_path).validate()


def test_calibration_updates_only_database_cameras(tmp_path, monkeypatch):
    from vidmap.mapper.stages import distortion_calibration

    camera = pycolmap.Camera(
        camera_id=1,
        model="OPENCV_FISHEYE",
        width=640,
        height=480,
        params=[350, 350, 320, 240, 0, 0, 0, 0],
        has_prior_focal_length=False,
    )
    rec = pycolmap.Reconstruction()
    rec.add_camera_with_trivial_rig(camera)
    for iid in (1, 2):
        rec.add_image_with_trivial_frame(pycolmap.Image(image_id=iid, camera_id=1, name=str(iid)))
    path = tmp_path / "geometry.db"
    with pycolmap.Database.open(path) as db:
        db.write_camera(camera, use_camera_id=True)
        db.write_rig(rec.rig(1), use_rig_id=True)
        for frame in rec.frames.values():
            db.write_frame(frame, use_frame_id=True)
        for image in rec.images.values():
            db.write_image(image, use_image_id=True)
            db.write_keypoints(image.image_id, np.array([[100, 100], [200, 200]], dtype=np.float32))
        matches = np.array([[0, 0], [1, 1]], dtype=np.uint32)
        db.write_matches(1, 2, matches)
        db.write_two_view_geometry(
            1,
            2,
            pycolmap.TwoViewGeometry(
                config=pycolmap.TwoViewGeometryConfiguration.CALIBRATED,
                E=np.eye(3),
                inlier_matches=matches,
            ),
        )
    with sqlite3.connect(path) as db:
        before = [line for line in db.iterdump() if not line.startswith('INSERT INTO "cameras"')]

    def refine(state, options, exclusions):
        params = state.reconstruction.camera(1).params.copy()
        params[:2] = 400
        state.reconstruction.camera(1).params = params

    monkeypatch.setattr(distortion_calibration, "calibrate_distorted_view_graph", refine)
    frontend = object.__new__(Frontend)
    frontend.preparation = PreparationOptions(distortion_calibration={"filter": {"enabled": False}})
    refined = frontend._calibrate_distortion(path)
    with sqlite3.connect(path) as db:
        after = [line for line in db.iterdump() if not line.startswith('INSERT INTO "cameras"')]
    assert before == after
    with pycolmap.Database.open(path) as db:
        np.testing.assert_array_equal(db.read_camera(1).params, refined.camera(1).params)
        assert not db.read_camera(1).has_prior_focal_length
    assert camera.focal_length_x == 350
    assert refined.camera(1).focal_length_x == 400


def test_mapper_does_not_repeat_prepared_calibration(monkeypatch):
    from vidmap.mapper import mapper as mapping

    mapper = object.__new__(Mapper)
    mapper.conf = MapperOptions()
    mapper.mapper_inputs = SimpleNamespace(distortion_calibrated=lambda: True)

    def repeated(*args, **kwargs):
        pytest.fail("Prepared distorted inputs must not repeat mapping VGC")

    class ReachedPoseEstimation(Exception):
        pass

    def stop_at_bearings(state):
        raise ReachedPoseEstimation

    monkeypatch.setattr(mapping, "ViewGraphFilter", repeated)
    monkeypatch.setattr(mapping, "ViewGraphCalibrator", repeated)
    monkeypatch.setattr(mapping.GlobalPositioner, "prepare_bearings", stop_at_bearings)
    with pytest.raises(ReachedPoseEstimation):
        mapper._solve(SimpleNamespace(solve_state=object()), None, None, {})
