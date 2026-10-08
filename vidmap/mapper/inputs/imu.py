# SPDX-License-Identifier: BSD-3-Clause
"""Build IMU preintegration edges between consecutive keyframes."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pycolmap
import pycolmap.inertial

from vidmap.mapper.native.extension import native


def build_imu_edge_records(
    measurements: pycolmap.ImuMeasurements,
    image_ids: Sequence[int],
    timestamps_s: Sequence[float] | np.ndarray,
    calibration: pycolmap.ImuCalibration,
    *,
    time_offset_s: float = 0.0,
    q_iori_xyzw: np.ndarray | None = None,
    preintegration_options: pycolmap.inertial.ImuPreintegrationOptions | None = None,
) -> list[native.ImuEdgeRecord]:
    """Preintegrate the IMU between consecutive keyframes.

    Args:
        measurements: IMU samples on the IMU clock.
        image_ids: Keyframe image IDs in temporal order.
        timestamps_s: Keyframe timestamps on the video clock, in seconds.
        calibration: IMU noise model.
        time_offset_s: Clock offset such that t_imu = t_video + time_offset_s.
        q_iori_xyzw: Optional (N, 4) per-keyframe stabilization rotations
            (virtual camera from physical camera), e.g. GoPro HyperSmooth IORI.
        preintegration_options: Defaults to RK4 integration.
    """
    image_ids = [int(value) for value in image_ids]
    timestamps = np.asarray(timestamps_s, dtype=np.float64)
    if len(image_ids) != len(timestamps):
        raise ValueError("image_ids and timestamps_s must have the same length")
    if np.any(np.diff(timestamps) <= 0):
        raise ValueError("Keyframe timestamps must be strictly increasing")
    if q_iori_xyzw is not None and len(q_iori_xyzw) != len(image_ids):
        raise ValueError("q_iori_xyzw must have one rotation per keyframe")
    if preintegration_options is None:
        preintegration_options = pycolmap.inertial.ImuPreintegrationOptions()
        preintegration_options.method = pycolmap.inertial.ImuIntegrationMethod.RK4

    edges: list[native.ImuEdgeRecord] = []
    for k in range(len(image_ids) - 1):
        t1 = pycolmap.timestamp_from_seconds(float(timestamps[k] + time_offset_s))
        t2 = pycolmap.timestamp_from_seconds(float(timestamps[k + 1] + time_offset_s))
        integrator = pycolmap.inertial.ImuPreintegrator(preintegration_options, calibration, t1, t2)
        integrator.integrate(measurements.extract_measurements_in_range(t1, t2))
        edge = native.ImuEdgeRecord()
        edge.image_id1 = image_ids[k]
        edge.image_id2 = image_ids[k + 1]
        edge.data = integrator.extract()
        edge.set_integrator(integrator)
        if q_iori_xyzw is not None:
            edge.q_iori_1_xyzw = np.asarray(q_iori_xyzw[k], dtype=np.float64)
            edge.q_iori_2_xyzw = np.asarray(q_iori_xyzw[k + 1], dtype=np.float64)
        edges.append(edge)
    return edges


def build_gopro_imu_edge_records(
    telemetry,
    image_ids: Sequence[int],
    timestamps_s: Sequence[float] | np.ndarray,
    *,
    time_offset_s: float = 0.0,
    calibration: pycolmap.ImuCalibration | None = None,
    use_iori: bool = True,
) -> tuple[list[native.ImuEdgeRecord], np.ndarray | None]:
    """Build IMU edges from GoPro telemetry (see vidmap.datasets.gopro_telemetry).

    Returns the edges and the per-keyframe IORI rotations (xyzw) if used.
    """
    from vidmap.datasets.gopro_telemetry import gopro_imu_calibration

    timestamps = np.asarray(timestamps_s, dtype=np.float64)
    if calibration is None:
        calibration = gopro_imu_calibration()
    q_iori = telemetry.interpolate_q_iori(timestamps)[:, [1, 2, 3, 0]] if use_iori else None
    measurements = telemetry.to_imu_measurements(
        t_start_s=float(timestamps[0] + time_offset_s - 0.5),
        t_end_s=float(timestamps[-1] + time_offset_s + 0.5),
    )
    edges = build_imu_edge_records(
        measurements,
        image_ids,
        timestamps,
        calibration,
        time_offset_s=time_offset_s,
        q_iori_xyzw=q_iori,
    )
    return edges, q_iori
