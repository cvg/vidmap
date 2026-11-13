"""Non-finite MDRP poses must not enter the view graph as valid edges."""

from types import SimpleNamespace

import numpy as np
import pytest

from vidmap.mapper.stages.relative_pose import pair_estimation

NUM_MATCHES = 8
IDENTITY_WXYZ = [1.0, 0.0, 0.0, 0.0]


def _estimate(monkeypatch, *, q, t, scale):
    """Run the pinhole MDRP path against a stubbed poselib result."""

    def fake_mdrp(points1, *_args):
        geometry = SimpleNamespace(pose=SimpleNamespace(q=q, t=t), scale=scale)
        return geometry, {"inliers": np.ones(len(points1), dtype=bool)}

    monkeypatch.setattr(pair_estimation.poselib, "estimate_monodepth_relative_pose", fake_mdrp)
    rng = np.random.default_rng(0)
    features = {1: rng.uniform(0, 100, (NUM_MATCHES, 2)), 2: rng.uniform(0, 100, (NUM_MATCHES, 2))}
    matches = np.stack([np.arange(NUM_MATCHES)] * 2, axis=1)
    _, result = pair_estimation.estimate_relative_pose_for_pair(
        (7, 1, 2, matches),
        images={1: SimpleNamespace(name="a", camera_id=1), 2: SimpleNamespace(name="b", camera_id=1)},
        ransac_options={},
        bundle_options={},
        camera_poselib_cache={1: None},
        image_feature_cache=features,
        image_depth_cache={1: np.ones(NUM_MATCHES), 2: np.ones(NUM_MATCHES)},
        image_valid_cache={1: np.ones(NUM_MATCHES, dtype=bool), 2: np.ones(NUM_MATCHES, dtype=bool)},
    )
    return result


@pytest.mark.parametrize(
    ("q", "t", "scale"),
    [
        ([np.nan, 0.0, 0.0, 0.0], [np.nan, 0.0, 0.0], np.nan),  # observed: everything NaN
        ([np.nan, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0], 1.0),
        (IDENTITY_WXYZ, [np.nan, 0.0, 0.0], 1.0),
        (IDENTITY_WXYZ, [np.inf, 0.0, 0.0], 1.0),
        (IDENTITY_WXYZ, [1.0, 0.0, 0.0], np.nan),
        ([2.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0], 1.0),  # non-unit quaternion
    ],
)
def test_non_finite_mdrp_pose_is_invalid(monkeypatch, q, t, scale):
    result = _estimate(monkeypatch, q=q, t=t, scale=scale)
    assert result == {"is_valid": False, "invalid_reason": pair_estimation.NON_FINITE_POSE_REASON}


@pytest.mark.parametrize("t", [[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])  # zero translation: pure rotation stays valid
def test_finite_mdrp_pose_is_valid(monkeypatch, t):
    result = _estimate(monkeypatch, q=IDENTITY_WXYZ, t=t, scale=1.0)
    assert result["is_valid"] is True
    np.testing.assert_allclose(result["cam2_from_cam1"].translation, t)
    np.testing.assert_array_equal(result["inliers"], np.arange(NUM_MATCHES))
