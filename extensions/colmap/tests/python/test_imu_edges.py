"""Tests for building IMU preintegration edges between keyframes."""

from __future__ import annotations

import numpy as np
import pycolmap
import pytest

from vidmap.mapper.inputs.imu import build_imu_edge_records


def _constant_rate_measurements(omega: np.ndarray, t_end: float, rate: float = 200.0) -> pycolmap.ImuMeasurements:
    ms = pycolmap.ImuMeasurements()
    for t in np.arange(0.0, t_end + 1e-9, 1.0 / rate):
        ms.insert(
            pycolmap.ImuMeasurement(
                timestamp=pycolmap.timestamp_from_seconds(float(t)),
                accel=np.array([0.0, 0.0, 9.81]),
                gyro=omega,
            )
        )
    return ms


def _calibration() -> pycolmap.ImuCalibration:
    calib = pycolmap.ImuCalibration()
    calib.gravity_magnitude = 9.81
    calib.gyro_noise_density = 1e-4
    calib.accel_noise_density = 1e-3
    calib.bias_gyro_random_walk_sigma = 1e-5
    calib.bias_accel_random_walk_sigma = 1e-4
    calib.imu_rate = 200.0
    return calib


def test_build_imu_edge_records_constant_rotation():
    omega = np.array([0.0, 0.0, 0.5])
    ms = _constant_rate_measurements(omega, t_end=2.0)
    image_ids = [3, 7, 9]
    timestamps = np.array([0.2, 0.7, 1.4])
    q_iori = np.tile([0.0, 0.0, 0.0, 1.0], (3, 1))
    edges = build_imu_edge_records(ms, image_ids, timestamps, _calibration(), time_offset_s=0.1, q_iori_xyzw=q_iori)

    assert [(e.image_id1, e.image_id2) for e in edges] == [(3, 7), (7, 9)]
    for e, dt in zip(edges, np.diff(timestamps), strict=True):
        assert e.data.delta_t == pytest.approx(dt, abs=1e-6)
        angle = e.data.delta_R.angle()
        assert angle == pytest.approx(omega[2] * dt, abs=1e-4)
        np.testing.assert_allclose(e.q_iori_1_xyzw, [0.0, 0.0, 0.0, 1.0])
        # The attached integrator supports reintegration with new biases.
        e.reintegrate(np.zeros(6))


def test_build_imu_edge_records_rejects_bad_inputs():
    ms = _constant_rate_measurements(np.zeros(3), t_end=1.0)
    with pytest.raises(ValueError):
        build_imu_edge_records(ms, [1, 2], [0.5, 0.2], _calibration())
    with pytest.raises(ValueError):
        build_imu_edge_records(ms, [1, 2, 3], [0.1, 0.2], _calibration())
