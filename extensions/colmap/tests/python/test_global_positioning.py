from dataclasses import replace
from unittest.mock import patch

import numpy as np
import pycolmap
import pytest
import vidmap_native._core as native

from tests.mapper.test_bundle_adjustment_problem import _options as _ba_options
from vidmap.mapper.native.state import SolveState
from vidmap.mapper.options.positioning import GPOptions
from vidmap.mapper.playback_trace import SolverPlaybackOptions
from vidmap.mapper.stages.bundle_adjustment.problem import run_bundle_adjustment
from vidmap.mapper.stages.global_positioning import problem as gp
from vidmap.mapper.stages.global_positioning.problem import TemporalAccelerationPrior


def _gp_options(rounds=0):
    return GPOptions(
        common=dict(
            use_metric_depth_constraint=False, num_threads=1, loss_function_type="huber", loss_function_scale=0.1
        ),
        first_pass=dict(
            max_iterations=1,
            sequential_support_warmup_rounds=rounds,
            sequential_support_observations_per_track=2 if rounds else 0,
            sequential_support_loss=dict(name="trivial") if rounds else None,
            loss_lc_geometry=dict(name="trivial"),
        ),
    )


def _run_gp(options, state, callback=None, *, timeline=None, **kwargs):
    prepared = gp.prepare_observations(
        state.reconstruction, state.sidecars, options, state.image_order if timeline is None else timeline
    )
    create = pycolmap.create_default_global_positioner

    def initialize(upstream, *args):
        upstream.generate_random_positions = False
        upstream.generate_random_points = False
        return create(upstream, *args)

    with patch.object(pycolmap, "create_default_global_positioner", initialize):
        return gp.run_global_positioning(
            options,
            state.reconstruction,
            state.sidecars,
            prepared,
            None,
            playback_callback=callback,
            capture_state=True,
            **kwargs,
        )


def _playback_problem(storage_order=None, *, extra_camera=False):
    reconstruction = pycolmap.Reconstruction()
    for camera_id in (1, 2) if extra_camera else (1,):
        reconstruction.add_camera_with_trivial_rig(
            pycolmap.Camera(
                camera_id=camera_id,
                model="PINHOLE",
                width=640,
                height=480,
                params=[500.0, 500.0, 320.0, 240.0],
                has_prior_focal_length=True,
            )
        )
    sidecars = native.MappingSidecars()
    point = np.asarray([0.0, 0.0, 5.0])
    centers = (np.zeros(3), np.asarray([1.0, 0.0, 0.0]), np.asarray([0.0, 1.0, 0.0]))
    for image_id, center in enumerate(centers, start=1):
        camera_point = point - center
        reconstruction.add_image_with_trivial_frame(
            pycolmap.Image(
                image_id=image_id,
                camera_id=2 if extra_camera and image_id == 3 else 1,
                name=f"image-{image_id}.jpg",
                keypoints=np.asarray(
                    [
                        [
                            500.0 * camera_point[0] / camera_point[2] + 320.0,
                            500.0 * camera_point[1] / camera_point[2] + 240.0,
                        ]
                    ]
                ),
            ),
            pycolmap.Rigid3d(translation=-center),
        )
        data = native.ImageData()
        data.bearings = np.asarray([camera_point / np.linalg.norm(camera_point)])
        sidecars.add_image(image_id, data)
    track = pycolmap.Track([pycolmap.TrackElement(i, 0) for i in (storage_order or [1, 2])])
    point_id = reconstruction.add_point3D(point, track)
    assert point_id == 1
    data = native.TrackData()
    if storage_order is None:
        data.loop_closure_observations = np.asarray([[3, 0]], dtype=np.uint32)
        data.loop_closure_anchors = np.asarray([[1, 0]], dtype=np.uint32)
    sidecars.set_track(point_id, data)
    return SolveState(reconstruction, pycolmap.PoseGraph(), sidecars)


def test_continuous_support_reduces_cost():
    state = _playback_problem([3, 1, 2])
    result = _run_gp(_gp_options(2), state)
    assert result.success
    assert result.diagnostics.final_cost < result.diagnostics.initial_cost


def test_sequential_support_requires_complete_unique_timeline():
    with pytest.raises(ValueError, match="unique image timeline"):
        _run_gp(_gp_options(2), _playback_problem([1, 2, 3]), timeline=[1, 2])


def test_sequential_support_uses_explicit_chronology():
    def solve(timeline):
        state = _playback_problem([3, 1, 2])
        _run_gp(_gp_options(2), state, timeline=timeline)
        return state.reconstruction.point3D(1).xyz

    assert np.linalg.norm(solve([1, 2, 3]) - solve([3, 2, 1])) > 1e-5


def test_sequential_support_playback_starts_before_warmup(monkeypatch):
    callbacks_per_solve = []
    solve = gp.pyceres.solve

    def record_callbacks(options, problem, summary):
        callbacks_per_solve.append(list(options.callbacks))
        return solve(options, problem, summary)

    monkeypatch.setattr(gp.pyceres, "solve", record_callbacks)
    captures = []
    result = _run_gp(
        _gp_options(4),
        _playback_problem([3, 1, 2]),
        captures.append,
        playback_options=SolverPlaybackOptions(snapshot_every_n_iterations=3),
    )
    assert captures[0]["phase"] == "initial"
    assert captures[0]["iteration"] == -1
    np.testing.assert_array_equal(captures[0]["points_xyz"][0], result.initial_point3D_xyz[1])
    assert [(c["phase"], c["iteration"]) for c in captures[1:-1]] == [("iteration", 0), ("iteration", 3)]
    assert captures[-1]["phase"] == "final"
    assert [len(callbacks) for callbacks in callbacks_per_solve] == [1, 1]
    assert callbacks_per_solve[0][0] is callbacks_per_solve[1][0]


def test_global_positioning_playback_emits_owned_loop_closure_state():
    captures = []
    result = _run_gp(_gp_options(), _playback_problem(), captures.append)
    assert result.success
    assert captures[0]["phase"] == "initial"
    assert captures[-1]["phase"] == "final"
    np.testing.assert_array_equal(captures[0]["image_ids"], [1, 2, 3])
    np.testing.assert_array_equal(captures[0]["point_ids"], [1])
    np.testing.assert_array_equal(captures[0]["lc_pairs"], [[1, 3]])
    np.testing.assert_array_equal(captures[0]["lc_support_count"], [1])
    assert np.isfinite(captures[0]["lc_raw_score"]).all()


@pytest.mark.parametrize("second_pass", [False, True])
def test_temporal_acceleration_uses_pass_specific_whitened_loss(second_pass, monkeypatch):
    options = _gp_options()
    options = replace(
        options,
        temporal_acceleration=replace(
            options.temporal_acceleration,
            stddev=4.0,
            first_pass_weight=1.0,
            second_pass_weight=2.0,
            first_pass_dead_zone=2.0,
            second_pass_dead_zone=8.0,
            first_pass_huber_width=5.0,
            second_pass_huber_width=12.0,
        ),
    )
    loss = gp.gp_costs.dead_zone_huber_loss
    calls = []

    def capture(dead_zone, width):
        calls.append((dead_zone, width))
        return loss(dead_zone, width)

    monkeypatch.setattr(gp.gp_costs, "dead_zone_huber_loss", capture)
    result = _run_gp(
        options,
        _playback_problem(),
        second_pass=second_pass,
        temporal_priors=[TemporalAccelerationPrior(1, 2, 3, 0.1, 1.0, 2.0)],
    )
    assert result.diagnostics.num_temporal_acceleration_residuals == 1
    assert calls == [(2.0, 3.0) if second_pass else (0.5, 1.25)]
    assert _run_gp(_gp_options(), _playback_problem()).diagnostics.num_temporal_acceleration_residuals == 0


@pytest.mark.parametrize("rounds,tolerance,updates", [(5, 0.0, 5), (16, 1.0, 0)])
def test_continuous_warmup_respects_budget_and_early_convergence(rounds, tolerance, updates, monkeypatch):
    options = _gp_options(rounds)
    options = replace(
        options,
        first_pass=replace(
            options.first_pass,
            function_tolerance_when_second=tolerance,
            gradient_tolerance_when_second=0.0,
            parameter_tolerance_when_second=0.0,
        ),
    )
    solves = []
    solve = gp._solve

    def capture(problem, solver, callback, offset):
        solves.append((solver.max_num_iterations, solver.max_trust_region_radius))
        return solve(problem, solver, callback, offset)

    monkeypatch.setattr(gp, "_solve", capture)
    captures = []
    result = _run_gp(options, _playback_problem([3, 1, 2]), captures.append)
    assert result.success
    assert solves == [(rounds, 1e4), (1, 1e16)]
    assert [c["iteration"] for c in captures if c["phase"] == "iteration" and c["iteration"] < rounds] == list(
        range(updates)
    )
    assert captures[-1]["phase"] == "final"


def test_bundle_adjustment_playback_honors_explicit_value_selection():
    problem = _playback_problem()
    captures = []
    options, config = _ba_options(problem)
    config.remove_image(3)
    options.refine_points3D = False
    options.constant_rig_from_world_rotation = True
    options.ceres.solver_options.num_threads = 1
    options.ceres.solver_options.max_num_iterations = 3
    playback = SolverPlaybackOptions(image_ids=[3, 1, 2], point3D_ids=[1])
    result = run_bundle_adjustment(
        options, config, [], [], [], problem, playback_callback=captures.append, playback_options=playback
    )

    assert result.success
    assert captures[0]["phase"] == "initial"
    assert captures[-1]["phase"] == "final"
    np.testing.assert_array_equal(captures[0]["image_ids"], [3, 1, 2])
    np.testing.assert_array_equal(captures[0]["point_ids"], [1])
    assert captures[0]["centers"].shape == (3, 3)
    assert captures[0]["points_xyz"].shape == (1, 3)


def test_bundle_adjustment_applies_robust_log_mean_focal_prior():
    problem = _playback_problem()
    options, config = _ba_options(problem)
    config.remove_image(3)
    options.refine_points3D = False
    options.constant_rig_from_world_rotation = True
    options.ceres.solver_options.num_threads = 1
    options.ceres.solver_options.max_num_iterations = 50

    prior = native.LogFocalPriorRecord()
    prior.camera_id = 1
    prior.observations = np.array([[800.0, 0.05]])
    prior.loss = pycolmap.create_ceres_loss_function(pycolmap.LossFunctionType.HUBER, 1.0, 1e6)

    result = run_bundle_adjustment(options, config, [], [], [prior], problem)

    assert result.success
    assert result.diagnostics.num_intrinsics_prior_residuals == 1
    np.testing.assert_allclose(problem.reconstruction.camera(1).params[:2].mean(), 800.0, rtol=1e-3)


def test_bundle_adjustment_log_focal_cost_matches_vgc_equation():
    """Check the native cost, not merely convergence toward a strong prior."""
    options, config = _ba_options(_playback_problem())
    config.remove_image(3)
    options.refine_points3D = False
    options.refine_rig_from_world = False
    options.refine_focal_length = False
    options.refine_principal_point = False
    options.refine_extra_params = False
    options.ceres.solver_options.num_threads = 1
    baseline = run_bundle_adjustment(options, config, [], [], [], _playback_problem())
    observations = np.array([[800.0, 0.05], [400.0, 0.2]])
    weight = 0.025
    problem = _playback_problem()
    focal = problem.reconstruction.camera(1).params[:2].mean()
    prior = native.LogFocalPriorRecord()
    prior.camera_id = 1
    prior.observations = observations
    prior.loss = pycolmap.create_ceres_loss_function(pycolmap.LossFunctionType.CAUCHY, 1.0, weight)
    squared = (np.log(focal / observations[:, 0]) / observations[:, 1]) ** 2
    result = run_bundle_adjustment(options, config, [], [], [prior], problem)
    np.testing.assert_allclose(
        result.diagnostics.initial_cost - baseline.diagnostics.initial_cost,
        0.5 * weight * np.log1p(squared).sum(),
        rtol=1e-9,
        atol=1e-9,
    )


def test_no_focal_prior_leaves_enabled_intrinsics_free():
    problem = _playback_problem([1, 2, 3])
    camera = problem.reconstruction.camera(1)
    camera.params = np.array([650.0, 650.0, 320.0, 240.0])
    options, config = _ba_options(problem)
    options.refine_rig_from_world = False
    options.refine_points3D = False
    options.refine_focal_length = True
    options.refine_principal_point = False
    options.refine_extra_params = False
    options.ceres.solver_options.num_threads = 1
    result = run_bundle_adjustment(options, config, [], [], [], problem)
    assert result.success
    assert result.diagnostics.num_intrinsics_prior_residuals == 0
    np.testing.assert_allclose(
        problem.reconstruction.camera(1).params,
        [500.0, 500.0, 320.0, 240.0],
        atol=1e-5,
        rtol=0,
    )


def test_duplicate_log_focal_camera_records_are_rejected():
    problem = _playback_problem()
    options, config = _ba_options(problem)
    config.remove_image(3)
    prior = native.LogFocalPriorRecord()
    prior.camera_id = 1
    prior.observations = np.array([[500.0, 0.02]])
    with np.testing.assert_raises_regex(ValueError, "duplicate BA log-focal prior camera"):
        run_bundle_adjustment(options, config, [], [], [prior, prior], problem)


def test_bundle_adjustment_preserves_camera_without_observations():
    problem = _playback_problem(extra_camera=True)
    camera_params = problem.reconstruction.camera(2).params.copy()
    data = problem.sidecars.track(1)
    data.loop_closure_observations = np.empty((0, 2), dtype=np.uint32)
    data.loop_closure_anchors = np.empty((0, 2), dtype=np.uint32)
    priors = []
    for camera_id in (1, 2):
        prior = native.LogFocalPriorRecord()
        prior.camera_id = camera_id
        prior.observations = np.array([[800.0, 0.24]])
        priors.append(prior)
    options, config = _ba_options(problem)
    options.refine_points3D = False
    options.refine_rig_from_world = False
    options.ceres.solver_options.num_threads = 1
    result = run_bundle_adjustment(options, config, [], [], priors, problem)
    assert result.success
    assert result.diagnostics.num_intrinsics_prior_residuals == 1
    np.testing.assert_array_equal(problem.reconstruction.camera(2).params, camera_params)
