# SPDX-License-Identifier: BSD-3-Clause

"""GoPro HERO11/12 GPMF telemetry and IMU loader for VidMap and COLMAP.

Parses binary GPMF streams and .npz telemetry dumps (ACCL, GYRO, CORI, IORI,
GRAV, TMPC, STMP), transforms sensor coordinates into COLMAP camera optical
frame (RDF: right-down-forward), and provides pycolmap.ImuMeasurements,
pycolmap.ImuCalibration, and HyperSmooth R_iori interpolation.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import gpmf.io
import gpmf.parse
import numpy as np
import pycolmap
from scipy.spatial.transform import Rotation, Slerp

# Transformation matrix from GoPro accelerometer/gyroscope XYZ coordinates
# to COLMAP camera optical frame (X right, Y down, Z forward):
# P_cam_from_xyz = diag(-1, -1, -1) @ [[1, 0, 0], [0, 0, 1], [0, 1, 0]]
# det(P_cam_from_xyz) = +1
P_CAM_FROM_XYZ = np.array([[-1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, -1.0, 0.0]], dtype=np.float64)

# Transformation matrix from native GPMF ORIN="ZXY" channels [ch0, ch1, ch2]
# to COLMAP camera optical frame:
# v_x^cam = -ch1, v_y^cam = -ch0, v_z^cam = -ch2
# det(P_cam_from_raw) = +1
P_CAM_FROM_RAW = np.array([[0.0, -1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, -1.0]], dtype=np.float64)


def transform_xyz_to_cam(vectors_xyz: np.ndarray) -> np.ndarray:
    """Transform 3D vectors from GoPro sensor XYZ to camera optical frame."""
    arr = np.asarray(vectors_xyz, dtype=np.float64)
    if arr.ndim == 1:
        if arr.shape[0] != 3:
            raise ValueError(f"Expected 3-element vector, got {arr.shape}")
        return np.array([-arr[0], -arr[2], -arr[1]], dtype=np.float64)
    if arr.ndim == 2:
        if arr.shape[1] != 3:
            raise ValueError(f"Expected (N, 3) matrix, got {arr.shape}")
        return np.column_stack([-arr[:, 0], -arr[:, 2], -arr[:, 1]])
    raise ValueError(f"Expected 1D or 2D array, got ndim={arr.ndim}")


def transform_raw_to_cam(channels_zxy: np.ndarray) -> np.ndarray:
    """Transform 3D channels from native GPMF ZXY to camera optical frame."""
    arr = np.asarray(channels_zxy, dtype=np.float64)
    if arr.ndim == 1:
        if arr.shape[0] != 3:
            raise ValueError(f"Expected 3-element vector, got {arr.shape}")
        return np.array([-arr[1], -arr[0], -arr[2]], dtype=np.float64)
    if arr.ndim == 2:
        if arr.shape[1] != 3:
            raise ValueError(f"Expected (N, 3) matrix, got {arr.shape}")
        return np.column_stack([-arr[:, 1], -arr[:, 0], -arr[:, 2]])
    raise ValueError(f"Expected 1D or 2D array, got ndim={arr.ndim}")


def gopro_imu_calibration() -> pycolmap.ImuCalibration:
    """Return recommended VIO calibration for Bosch BMI260/270 in GoPro."""
    calib = pycolmap.ImuCalibration()
    calib.imu_rate = 200.0  # nominal ~197.33 Hz
    # Recommended VIO priors:
    calib.gyro_noise_density = 5.0e-4  # rad / (s * sqrt(Hz))
    calib.accel_noise_density = 8.0e-3  # m / (s^2 * sqrt(Hz))
    calib.bias_gyro_random_walk_sigma = 5.0e-5  # rad / (s^2 * sqrt(Hz))
    calib.bias_accel_random_walk_sigma = 1.0e-3  # m / (s^3 * sqrt(Hz))
    calib.gravity_magnitude = 9.81
    return calib


def interpolate_quaternions(
    query_timestamps_s: Sequence[float] | np.ndarray,
    timestamps_s: np.ndarray,
    quats_wxyz: np.ndarray,
) -> np.ndarray:
    """Interpolate unit quaternions [w, x, y, z] using SLERP.

    Returns interpolated quaternions in Hamilton [w, x, y, z] convention.
    """
    queries = np.asarray(query_timestamps_s, dtype=np.float64)
    if len(timestamps_s) < 2:
        raise ValueError("Need at least 2 keyframes to interpolate quaternions")

    # Scipy Rotation expects scalar-last [x, y, z, w]:
    quats_xyzw = quats_wxyz[:, [1, 2, 3, 0]]
    rotations = Rotation.from_quat(quats_xyzw)
    slerp = Slerp(timestamps_s, rotations)

    # Clamp queries to valid range to prevent out-of-bounds error
    clamped_queries = np.clip(queries, timestamps_s[0], timestamps_s[-1])
    interp_rots = slerp(clamped_queries)
    interp_xyzw = interp_rots.as_quat()
    if interp_xyzw.ndim == 1:
        return interp_xyzw[[3, 0, 1, 2]]
    return interp_xyzw[:, [3, 0, 1, 2]]


@dataclass
class GoProTelemetry:
    """Synchronized telemetry streams from a GoPro camera."""

    device_name: str
    duration_s: float
    video_fps: float
    accl_timestamps_s: np.ndarray
    accl_cam: np.ndarray  # (N, 3) in m/s^2, camera optical frame
    gyro_timestamps_s: np.ndarray
    gyro_cam: np.ndarray  # (N, 3) in rad/s, camera optical frame
    cori_timestamps_s: np.ndarray
    cori_quats_wxyz: np.ndarray  # (N, 4) physical cam_from_world
    iori_timestamps_s: np.ndarray
    iori_quats_wxyz: np.ndarray  # (N, 4) virtual_cam_from_physical_cam
    grav_timestamps_s: np.ndarray
    grav_cam: np.ndarray  # (N, 3) unit downward gravity in optical frame
    temperature_timestamps_s: np.ndarray | None = None
    temperatures_c: np.ndarray | None = None

    def to_imu_measurements(
        self,
        t_start_s: float | None = None,
        t_end_s: float | None = None,
        time_offset_s: float = 0.0,
    ) -> pycolmap.ImuMeasurements:
        """Convert synchronized gyro and accel into pycolmap.ImuMeasurements.

        Parameters
        ----------
        t_start_s : float, optional
            Start time in seconds.
        t_end_s : float, optional
            End time in seconds.
        time_offset_s : float
            Timestamp shift: t_imu = t_video + time_offset_s.

        Returns
        -------
        pycolmap.ImuMeasurements
        """
        # Interpolate accel onto gyro timestamps so each measurement has both
        t_gyro = self.gyro_timestamps_s
        t_accl = self.accl_timestamps_s

        if len(t_gyro) == len(t_accl) and np.allclose(t_gyro, t_accl, atol=1e-5):
            common_t = t_gyro
            interp_accel = self.accl_cam
            gyro_data = self.gyro_cam
        else:
            # Linear interpolation of accelerometer onto gyro timestamps
            common_t = t_gyro
            gyro_data = self.gyro_cam
            interp_accel = np.zeros_like(gyro_data)
            for ch in range(3):
                interp_accel[:, ch] = np.interp(common_t, t_accl, self.accl_cam[:, ch])

        mask = np.ones(len(common_t), dtype=bool)
        if t_start_s is not None:
            mask &= common_t >= t_start_s
        if t_end_s is not None:
            mask &= common_t <= t_end_s

        filtered_t = common_t[mask]
        filtered_gyro = gyro_data[mask]
        filtered_accel = interp_accel[mask]

        measurements = pycolmap.ImuMeasurements()
        # Convert seconds to integer nanoseconds
        prev_ts = None
        for t, g, a in zip(filtered_t, filtered_gyro, filtered_accel, strict=True):
            ts = int(round((t + time_offset_s) * 1e9))
            if prev_ts is not None and ts <= prev_ts:
                ts = prev_ts + 1
            prev_ts = ts
            measurements.insert(pycolmap.ImuMeasurement(ts, g, a))

        return measurements

    def interpolate_q_iori(self, query_timestamps_s: Sequence[float] | np.ndarray) -> np.ndarray:
        """Interpolate HyperSmooth R_iori [w, x, y, z] at query timestamps."""
        return interpolate_quaternions(query_timestamps_s, self.iori_timestamps_s, self.iori_quats_wxyz)


def load_gopro_telemetry_npz(npz_path: Path | str) -> GoProTelemetry:
    """Load GoPro telemetry from a pre-parsed NumPy .npz archive."""
    data = np.load(str(npz_path))
    device_name = str(data.get("device_name", "GoPro HERO"))
    duration_s = float(data.get("duration_s", 0.0))

    accl_timestamps = np.asarray(data["accl_timestamps_s"], dtype=np.float64)
    gyro_timestamps = np.asarray(data["gyro_timestamps_s"], dtype=np.float64)

    if "accl_xyz" in data:
        accl_cam = transform_xyz_to_cam(data["accl_xyz"])
    elif "accl_raw" in data:
        accl_cam = transform_raw_to_cam(data["accl_raw"])
    else:
        raise KeyError("Archive missing 'accl_xyz' or 'accl_raw'")

    if "gyro_xyz" in data:
        gyro_cam = transform_xyz_to_cam(data["gyro_xyz"])
    elif "gyro_raw" in data:
        gyro_cam = transform_raw_to_cam(data["gyro_raw"])
    else:
        raise KeyError("Archive missing 'gyro_xyz' or 'gyro_raw'")

    cori_timestamps = np.asarray(data["cori_timestamps_s"], dtype=np.float64)
    cori_quats = np.asarray(data["cori"], dtype=np.float64)

    iori_timestamps = np.asarray(data["iori_timestamps_s"], dtype=np.float64)
    iori_quats = np.asarray(data["iori"], dtype=np.float64)

    grav_timestamps = np.asarray(data["grav_timestamps_s"], dtype=np.float64)
    grav_cam = np.asarray(data["grav"], dtype=np.float64)

    if "video_fps" in data:
        video_fps = float(data["video_fps"])
    elif len(cori_timestamps) >= 2:
        video_fps = 1.0 / float(np.median(np.diff(cori_timestamps)))
    else:
        raise KeyError("Archive missing 'video_fps' and has insufficient 'cori_timestamps_s' to infer it")

    return GoProTelemetry(
        device_name=device_name,
        duration_s=duration_s,
        video_fps=video_fps,
        accl_timestamps_s=accl_timestamps,
        accl_cam=accl_cam,
        gyro_timestamps_s=gyro_timestamps,
        gyro_cam=gyro_cam,
        cori_timestamps_s=cori_timestamps,
        cori_quats_wxyz=cori_quats,
        iori_timestamps_s=iori_timestamps,
        iori_quats_wxyz=iori_quats,
        grav_timestamps_s=grav_timestamps,
        grav_cam=grav_cam,
    )


_STREAM_DATA_KEYS = frozenset(("ACCL", "GYRO", "CORI", "IORI", "GRAV"))


def _first_scalar(val: object) -> float | None:
    arr = np.asarray(val).reshape(-1)
    return float(arr[0]) if arr.size > 0 else None


def _parse_strm(strm_items: Sequence) -> tuple[str | None, np.ndarray | None, int | None, float | None]:
    """Extract (stream_key, scaled_data, stmp_us, temperature_c) from a GPMF STRM block."""
    scale = 1.0
    stmp_us: int | None = None
    tmpc_c: float | None = None
    stream_key: str | None = None
    raw_data = None

    for sub in strm_items:
        if sub.key == "SCAL":
            s = _first_scalar(sub.value)
            if s is not None and s != 0.0:
                scale = s
        elif sub.key == "STMP":
            stmp = _first_scalar(sub.value)
            if stmp is not None:
                stmp_us = int(stmp)
        elif sub.key == "TMPC":
            tmpc_c = _first_scalar(sub.value)
        elif sub.key in _STREAM_DATA_KEYS:
            stream_key = sub.key
            raw_data = sub.value

    if stream_key is None or raw_data is None:
        return None, None, stmp_us, tmpc_c
    return stream_key, np.asarray(raw_data, dtype=np.float64) / scale, stmp_us, tmpc_c


def _infer_timing(
    blocks: dict[str, list[np.ndarray]],
    stmps: dict[str, list[int]],
    num_payloads: int,
) -> tuple[float, float]:
    """Infer (payload_duration_s, video_fps) from GPMF STMP timestamps and frame counts."""
    payload_duration = 1.0
    for key in ("CORI", "IORI", "GRAV", "ACCL", "GYRO"):
        key_stmps = stmps.get(key, [])
        if len(key_stmps) >= 2 and key_stmps[-1] > key_stmps[0]:
            payload_duration = (key_stmps[-1] - key_stmps[0]) * 1e-6 / (len(key_stmps) - 1)
            break

    video_fps = 60.0
    for key in ("CORI", "IORI", "GRAV"):
        key_blocks = blocks.get(key, [])
        if not key_blocks:
            continue
        key_stmps = stmps.get(key, [])
        if len(key_blocks) >= 2 and len(key_stmps) >= 2 and key_stmps[-1] > key_stmps[0]:
            dt_span_s = (key_stmps[-1] - key_stmps[0]) * 1e-6
            frames_span = sum(len(b) for b in key_blocks[:-1])
            video_fps = frames_span / dt_span_s
        else:
            total_frames = sum(len(b) for b in key_blocks)
            video_fps = total_frames / (max(num_payloads, 1) * payload_duration)
        break

    return payload_duration, video_fps


def _build_payload_timestamps(blocks: list[np.ndarray], payload_duration: float, t0_offset: float = 0.0) -> np.ndarray:
    """Build uniformly spaced sample timestamps across GPMF payloads."""
    if not blocks:
        return np.zeros(0, dtype=np.float64)
    return np.concatenate(
        [
            t0_offset
            + p_idx * payload_duration
            + np.arange(len(block), dtype=np.float64) * (payload_duration / len(block))
            for p_idx, block in enumerate(blocks)
        ]
    )


def _build_stmp_anchored_timestamps(blocks: list[np.ndarray], stmps_s: Sequence[float]) -> np.ndarray:
    """Build sample timestamps by anchoring each GPMF payload at its own STMP.

    STMP is the camera-clock time of the first sample of each payload. The
    sensor oscillator drifts slowly against the camera clock, so samples are
    spaced uniformly only within a payload, at the rate implied by the next
    payload's STMP (the last payload reuses the previous rate).
    """
    if not blocks:
        return np.zeros(0, dtype=np.float64)
    stmps = np.asarray(stmps_s, dtype=np.float64)
    if len(stmps) != len(blocks):
        raise ValueError("Need exactly one STMP per payload")
    if len(blocks) == 1:
        raise ValueError("Need at least two payloads to infer the sample rate")
    counts = np.array([len(block) for block in blocks], dtype=np.float64)
    periods = np.diff(stmps) / counts[:-1]
    if np.any(periods <= 0):
        raise ValueError("STMP timestamps must be strictly increasing")
    periods = np.append(periods, periods[-1])
    return np.concatenate(
        [
            stmp + np.arange(len(block), dtype=np.float64) * period
            for stmp, period, block in zip(stmps, periods, blocks)
        ]
    )


def load_gopro_telemetry_bin(bin_path: Path | str, imu_timing: str = "stmp") -> GoProTelemetry:
    """Parse binary GPMF telemetry track (or video file) and extract IMU/optical streams.

    ``imu_timing`` selects how ACCL/GYRO sample times are reconstructed:
    ``"stmp"`` (default) anchors every payload at its own STMP, which follows the
    camera clock and absorbs the drift of the IMU oscillator, while ``"uniform"``
    spreads samples over the average payload duration (legacy behavior, off by a
    few milliseconds when the IMU clock drifts).
    """
    p = Path(bin_path)
    buf = gpmf.io.extract_gpmf_stream(str(p)) if p.suffix.lower() in (".mp4", ".mov") else p.read_bytes()

    device_name = "GoPro HERO"
    blocks: dict[str, list[np.ndarray]] = {k: [] for k in _STREAM_DATA_KEYS}
    stmps: dict[str, list[int]] = {k: [] for k in _STREAM_DATA_KEYS}
    tmpc_values: list[float] = []
    tmpc_payload_indices: list[float] = []

    devc_packets = [d for d in gpmf.parse.iter_klv(buf) if d.key == "DEVC"]
    for payload_idx, devc in enumerate(devc_packets):
        for item in devc.value:
            if item.key == "DVNM" and item.value:
                device_name = str(item.value)
                continue
            if item.key != "STRM":
                continue
            key, data, stmp_us, tmpc_c = _parse_strm(list(item.value))
            if tmpc_c is not None:
                tmpc_values.append(tmpc_c)
                tmpc_payload_indices.append(float(payload_idx))
            if key is not None and data is not None:
                blocks[key].append(data)
                if stmp_us is not None:
                    stmps[key].append(stmp_us)

    if not blocks["ACCL"] or not blocks["GYRO"]:
        raise ValueError(f"No ACCL or GYRO streams found in GPMF file {bin_path}")

    all_accl = np.concatenate(blocks["ACCL"], axis=0)
    all_gyro = np.concatenate(blocks["GYRO"], axis=0)
    all_cori = np.concatenate(blocks["CORI"], axis=0) if blocks["CORI"] else np.zeros((0, 4))
    all_iori = np.concatenate(blocks["IORI"], axis=0) if blocks["IORI"] else np.zeros((0, 4))
    all_grav = np.concatenate(blocks["GRAV"], axis=0) if blocks["GRAV"] else np.zeros((0, 3))

    num_payloads = max(len(devc_packets), 1)
    payload_duration, video_fps = _infer_timing(blocks, stmps, num_payloads)
    duration_s = float(num_payloads) * payload_duration

    # Camera-to-IMU clock offset from first-payload GPMF microsecond timestamps (STMP)
    t0_imu = 0.0
    if stmps["ACCL"] and stmps["CORI"]:
        t0_imu = (stmps["ACCL"][0] - stmps["CORI"][0]) * 1e-6

    if imu_timing == "stmp":
        if not stmps["CORI"]:
            raise ValueError("STMP-anchored IMU timing requires CORI STMPs as the video time origin")
        video_origin_us = stmps["CORI"][0]

        def stmp_timestamps(key: str) -> np.ndarray:
            if len(stmps[key]) != len(blocks[key]):
                raise ValueError(f"Missing STMP for some {key} payloads")
            return _build_stmp_anchored_timestamps(
                blocks[key], [(value - video_origin_us) * 1e-6 for value in stmps[key]]
            )

        accl_timestamps = stmp_timestamps("ACCL")
        gyro_timestamps = stmp_timestamps("GYRO")
    elif imu_timing == "uniform":
        accl_timestamps = _build_payload_timestamps(blocks["ACCL"], payload_duration, t0_imu)
        gyro_timestamps = _build_payload_timestamps(blocks["GYRO"], payload_duration, t0_imu)
    else:
        raise ValueError(f"Unknown imu_timing '{imu_timing}'")
    cori_timestamps = np.arange(len(all_cori), dtype=np.float64) / video_fps
    iori_timestamps = np.arange(len(all_iori), dtype=np.float64) / video_fps
    grav_timestamps = np.arange(len(all_grav), dtype=np.float64) / video_fps

    return GoProTelemetry(
        device_name=device_name,
        duration_s=duration_s,
        video_fps=video_fps,
        accl_timestamps_s=accl_timestamps,
        accl_cam=transform_raw_to_cam(all_accl),
        gyro_timestamps_s=gyro_timestamps,
        gyro_cam=transform_raw_to_cam(all_gyro),
        cori_timestamps_s=cori_timestamps,
        cori_quats_wxyz=all_cori,
        iori_timestamps_s=iori_timestamps,
        iori_quats_wxyz=all_iori,
        grav_timestamps_s=grav_timestamps,
        grav_cam=all_grav,
        temperature_timestamps_s=(
            np.array(tmpc_payload_indices, dtype=np.float64) * payload_duration if tmpc_payload_indices else None
        ),
        temperatures_c=np.array(tmpc_values, dtype=np.float64) if tmpc_values else None,
    )


def load_gopro_telemetry(path: Path | str, imu_timing: str = "stmp") -> GoProTelemetry:
    """Load GoPro telemetry from either .npz or binary GPMF (.bin, .gpmf, .mp4, .mov).

    ``imu_timing`` only applies to binary GPMF inputs (see ``load_gopro_telemetry_bin``);
    .npz files store precomputed sample timestamps.
    """
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix == ".npz":
        return load_gopro_telemetry_npz(p)
    if suffix in (".bin", ".gpmf", ".mp4", ".mov"):
        return load_gopro_telemetry_bin(p, imu_timing=imu_timing)
    raise ValueError(f"Unsupported GoPro telemetry file suffix '{p.suffix}' for {p}")
