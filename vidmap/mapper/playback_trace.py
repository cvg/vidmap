from __future__ import annotations

import os
import re
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

from vidmap.mapper.playback_trace_storage import PlaybackSnapshot, PlaybackTopology, PlaybackTraceWriter

_PAIR_ID_BASE = 2147483647
_MIN_LC_RANK_GAP = 10
_NATIVE_POINT_LIMIT = 200_000


@dataclass(frozen=True, kw_only=True)
class PlaybackTraceOptions:
    """Process-owned playback sampling controls."""

    iteration_stride: int = 3
    point_cap: int | None = None

    def __post_init__(self) -> None:
        if (
            not isinstance(self.iteration_stride, int)
            or isinstance(self.iteration_stride, bool)
            or self.iteration_stride < 1
        ):
            raise ValueError("playback trace iteration stride must be a positive integer")
        if self.point_cap is not None and (
            not isinstance(self.point_cap, int) or isinstance(self.point_cap, bool) or self.point_cap < 1
        ):
            raise ValueError("playback trace point cap must be a positive integer")
        if self.point_cap is not None and self.point_cap > _NATIVE_POINT_LIMIT:
            raise ValueError(f"playback trace point cap cannot exceed {_NATIVE_POINT_LIMIT}")


@dataclass(kw_only=True)
class PlaybackTraceRecorder:
    """Capture only the solver state needed by Rerun playback."""

    writer: PlaybackTraceWriter
    database: Path
    options: PlaybackTraceOptions
    _started_at: float
    _callbacks: list[Any] = field(default_factory=list)
    _stage_subsolves: dict[str, int] = field(default_factory=dict)

    @classmethod
    def preflight(cls, output_dir: Path, *, replace: bool = False) -> None:
        _validate_global_positioning_playback_capabilities()
        _validate_bundle_adjustment_playback_capabilities()
        destination = Path(output_dir) / "playback_trace"
        if os.path.lexists(destination):
            if replace and destination.is_dir() and not destination.is_symlink():
                return
            raise FileExistsError(f"Playback trace already exists: {destination}")

    @classmethod
    def start(
        cls,
        output_dir: Path,
        database: Path,
        *,
        options: PlaybackTraceOptions | None = None,
        replace: bool = False,
    ) -> PlaybackTraceRecorder:
        cls.preflight(output_dir, replace=replace)
        database = Path(database)
        if not database.is_file():
            raise FileNotFoundError(f"Finalized playback database is missing: {database}")
        options = PlaybackTraceOptions() if options is None else options
        if not isinstance(options, PlaybackTraceOptions):
            raise TypeError(f"Expected PlaybackTraceOptions, got {type(options).__name__}")
        return cls(
            writer=PlaybackTraceWriter.create(
                Path(output_dir) / "playback_trace",
                metadata={
                    "sampling": {
                        "iteration_stride": options.iteration_stride,
                        "point_cap": options.point_cap,
                    }
                },
                replace=replace,
            ),
            database=database,
            options=options,
            _started_at=time.monotonic(),
        )

    def attach_global_positioning(self, options: Any, stage: str) -> None:
        subsolve = self._next_subsolve(stage)
        sink = _GlobalPositioningTraceSink(
            self.writer,
            self.database,
            stage,
            subsolve,
            self.options,
            self._elapsed_seconds,
        )
        options.playback.snapshot_every_n_iterations = self.options.iteration_stride
        options.playback.callback = sink
        self._callbacks.append(sink)

    def attach_bundle_adjustment(self, options: Any, reconstruction: Any) -> BundleAdjustmentTraceSink:
        subsolve = self._next_subsolve("ba1")
        sink = BundleAdjustmentTraceSink(
            self.writer,
            reconstruction,
            subsolve,
            self.options,
            self._elapsed_seconds,
        )
        options.playback.snapshot_every_n_iterations = self.options.iteration_stride
        options.playback.image_ids = [int(value) for value in sink.image_ids]
        options.playback.point3D_ids = [int(value) for value in sink.point_ids]
        options.playback.callback = sink
        self._callbacks.append(sink)
        return sink

    def finish(self, reconstruction: Any) -> None:
        unfinished = [f"{sink.stage}/{sink.subsolve}" for sink in self._callbacks if not sink.finished]
        if unfinished:
            raise RuntimeError(f"playback_trace solver captures did not finish: {', '.join(unfinished)}")
        trace = self.writer.snapshot()
        if not trace.stages:
            raise RuntimeError("playback_trace contains no GP captures")
        stage = trace.stages[-1]
        point_ids = _selected_point_ids(reconstruction, self.options.point_cap)
        topology, scores = _terminal_topology(trace, reconstruction, stage)
        topology_id = self.writer.add_topology(topology)
        arrays, observed_ids = _reconstruction_frame(reconstruction, point_ids)
        if not np.array_equal(observed_ids, topology.image_ids):
            raise ValueError("Terminal playback topology changed during capture")
        self.writer.append_frame(
            PlaybackSnapshot(
                stage,
                topology_id,
                arrays["centers"],
                arrays["points_xyz"],
                lc_raw_score=scores if stage != "ba1" else None,
                terminal=True,
                phase="terminal",
                iteration=None,
                subsolve=self._stage_subsolves[stage] - 1,
                elapsed_seconds=self._elapsed_seconds(),
            )
        )
        self.writer.complete()

    def abort(self) -> None:
        self.writer.discard()

    def _next_subsolve(self, stage: str) -> int:
        subsolve = self._stage_subsolves[stage] if stage in self._stage_subsolves else 0
        self._stage_subsolves[stage] = subsolve + 1
        return subsolve

    def _elapsed_seconds(self) -> float:
        return time.monotonic() - self._started_at


class _GlobalPositioningTraceSink:
    def __init__(
        self,
        writer: PlaybackTraceWriter,
        database: Path,
        stage: str,
        subsolve: int,
        options: PlaybackTraceOptions,
        elapsed_seconds: Callable[[], float],
    ) -> None:
        self.writer = writer
        self.database = database
        self.stage = stage
        self.subsolve = subsolve
        self.options = options
        self.elapsed_seconds = elapsed_seconds
        self.finished = False
        self.topology: int | None = None
        self.image_ids: np.ndarray | None = None
        self.point_ids: np.ndarray | None = None
        self.point_indices: np.ndarray | None = None
        self.native_pairs: np.ndarray | None = None
        self.native_support: np.ndarray | None = None
        self.native_pair_count = 0

    def __call__(self, capture: dict[str, Any]) -> None:
        phase, iteration = _capture_identity(capture)
        if self.finished:
            raise RuntimeError(f"Playback stage {self.stage}/{self.subsolve} emitted a capture after final")
        if phase == "initial":
            if self.topology is not None:
                raise RuntimeError(f"Duplicate initial playback capture for {self.stage}")
            self._start(capture)
        else:
            if self.topology is None:
                raise RuntimeError(f"Playback stage {self.stage} emitted {phase} before initial")
            self._check(capture)
        if phase == "final":
            self.finished = True
        if self.topology is None:
            raise RuntimeError(f"Playback stage {self.stage} emitted a frame before its topology")
        if not _retain_capture(phase, iteration, self.options.iteration_stride):
            return

        scores = np.asarray(capture["lc_raw_score"], dtype=np.float32)
        if len(scores) != self.native_pair_count:
            raise ValueError(f"Playback stage {self.stage} changed LC score count")
        self.writer.append_frame_if_changed(
            PlaybackSnapshot(
                self.stage,
                self.topology,
                np.asarray(capture["centers"]),
                _selected_capture_points(capture, self.point_indices, self.point_ids, self.stage),
                lc_raw_score=scores,
                phase=phase,
                iteration=iteration,
                subsolve=self.subsolve,
                elapsed_seconds=self.elapsed_seconds(),
            ),
        )

    def _start(self, capture: dict[str, Any]) -> None:
        image_ids = np.asarray(capture["image_ids"], dtype=np.int64)
        point_ids = np.asarray(capture["point_ids"], dtype=np.int64)
        pairs = _normalized_pairs(capture["lc_pairs"])
        support = np.asarray(capture["lc_support_count"], dtype=np.uint64)
        if len(support) != len(pairs) or len({tuple(pair) for pair in pairs}) != len(pairs):
            raise ValueError(f"Playback stage {self.stage} emitted inconsistent LC topology")

        native = {tuple(int(value) for value in pair) for pair in pairs}
        rejected = sorted(_database_loop_closure_pairs(self.database, image_ids) - native)
        rejected_pairs = np.asarray(rejected, dtype=np.int64).reshape((-1, 2))
        all_pairs = np.concatenate((pairs, rejected_pairs)) if len(rejected_pairs) else pairs
        self.topology = self.writer.add_topology(
            PlaybackTopology(
                self.stage,
                image_ids,
                tuple(_image_names(self.database, image_ids)),
                lc_pairs=all_pairs,
                lc_support_count=np.concatenate((support, np.zeros(len(rejected_pairs), dtype=np.uint64))),
                native_pair_count=len(pairs),
            )
        )
        self.image_ids = image_ids.copy()
        self.point_ids = point_ids.copy()
        self.point_indices = _selected_point_indices(point_ids, self.options.point_cap)
        self.native_pairs = pairs.copy()
        self.native_support = support.copy()
        self.native_pair_count = len(pairs)

    def _check(self, capture: dict[str, Any]) -> None:
        checks = (
            (
                "image IDs",
                self.image_ids,
                np.asarray(capture["image_ids"], dtype=np.int64),
            ),
            (
                "point IDs",
                self.point_ids,
                np.asarray(capture["point_ids"], dtype=np.int64),
            ),
            ("LC pairs", self.native_pairs, _normalized_pairs(capture["lc_pairs"])),
            (
                "LC support",
                self.native_support,
                np.asarray(capture["lc_support_count"], dtype=np.uint64),
            ),
        )
        for label, expected, actual in checks:
            if expected is None or not np.array_equal(expected, actual):
                raise ValueError(f"Playback stage {self.stage} changed {label} during solve")


class BundleAdjustmentTraceSink:
    stage = "ba1"

    def __init__(
        self,
        writer: PlaybackTraceWriter,
        reconstruction: Any,
        subsolve: int,
        options: PlaybackTraceOptions,
        elapsed_seconds: Callable[[], float],
    ) -> None:
        self.writer = writer
        self.reconstruction = reconstruction
        self.subsolve = subsolve
        self.options = options
        self.elapsed_seconds = elapsed_seconds
        self.point_ids = _selected_point_ids(reconstruction, options.point_cap)
        topology, _scores = _terminal_topology(writer.snapshot(), reconstruction, "ba1")
        self.image_ids = topology.image_ids.copy()
        self.topology = writer.add_topology(topology)
        self.started = False
        self.finished = False

    def __call__(self, capture: dict[str, Any]) -> None:
        image_ids = np.asarray(capture["image_ids"], dtype=np.int64)
        if not np.array_equal(image_ids, self.image_ids):
            raise ValueError("BA playback camera topology changed during one solve")
        point_ids = np.asarray(capture["point_ids"], dtype=np.int64)
        if not np.array_equal(point_ids, self.point_ids):
            raise ValueError("BA playback point topology changed during one solve")
        phase, iteration = _capture_identity(capture)
        if self.finished:
            raise RuntimeError(f"BA playback subsolve {self.subsolve} emitted a capture after final")
        if not self.started and phase != "initial":
            raise ValueError("BA playback must start with an initial capture")
        if self.started and phase == "initial":
            raise ValueError("BA playback emitted a duplicate initial capture")
        self.started = True
        if phase == "final":
            self.finished = True
        if not _retain_capture(phase, iteration, self.options.iteration_stride):
            return
        self.writer.append_frame_if_changed(
            PlaybackSnapshot(
                "ba1",
                self.topology,
                np.asarray(capture["centers"]),
                np.asarray(capture["points_xyz"]),
                phase=phase,
                iteration=iteration,
                subsolve=self.subsolve,
                elapsed_seconds=self.elapsed_seconds(),
            ),
        )

    def finish_recovered(self, reconstruction: Any, *, final_iteration: int) -> None:
        """Record the restored pre-solve state after an unusable native solve."""
        _phase, final_iteration = _capture_identity({"phase": "final", "iteration": final_iteration})
        if not self.started:
            raise RuntimeError(f"BA playback subsolve {self.subsolve} cannot recover before its initial capture")
        if self.finished:
            raise RuntimeError(f"BA playback subsolve {self.subsolve} already emitted a final capture")
        arrays, image_ids = _reconstruction_frame(reconstruction, self.point_ids)
        if not np.array_equal(image_ids, self.image_ids):
            raise ValueError("Recovered BA playback camera topology changed during one solve")
        self.writer.append_frame_if_changed(
            PlaybackSnapshot(
                "ba1",
                self.topology,
                arrays["centers"],
                arrays["points_xyz"],
                phase="final",
                iteration=final_iteration,
                subsolve=self.subsolve,
                elapsed_seconds=self.elapsed_seconds(),
            ),
        )
        self.finished = True


def _terminal_topology(
    trace: Any,
    reconstruction: Any,
    stage: str,
) -> tuple[PlaybackTopology, np.ndarray | None]:
    cameras = _registered_camera_poses(reconstruction)
    image_ids = np.asarray([item[0] for item in cameras], dtype=np.int64)
    valid = set(int(value) for value in image_ids)
    gp_frames = tuple(frame for frame in trace.frames() if frame.stage in {"gp1", "gp2"})
    if not gp_frames:
        raise RuntimeError("BA playback requires an earlier GP capture")
    source = gp_frames[-1]
    source_pairs = np.empty((0, 2), dtype=np.int64) if source.lc_pairs is None else np.asarray(source.lc_pairs)
    keep = np.asarray(
        [int(left) in valid and int(right) in valid for left, right in source_pairs],
        dtype=bool,
    )
    pairs = source_pairs[keep]
    native_pair_count = int(np.count_nonzero(keep[: source.native_pair_count]))
    support = None if source.lc_support_count is None else np.asarray(source.lc_support_count)[keep]
    topology = PlaybackTopology(
        stage=stage,
        image_ids=image_ids,
        names=tuple(item[1] for item in cameras),
        lc_pairs=pairs,
        lc_support_count=support,
        native_pair_count=native_pair_count,
    )
    if stage == "ba1":
        return topology, None
    scores = None if source.lc_raw_score is None else np.asarray(source.lc_raw_score)[keep[: source.native_pair_count]]
    return topology, scores


def _reconstruction_frame(reconstruction: Any, point_ids: np.ndarray) -> tuple[dict[str, np.ndarray], np.ndarray]:
    cameras = _registered_camera_poses(reconstruction)
    points = np.asarray(
        [reconstruction.point3D(int(point_id)).xyz for point_id in point_ids],
        dtype=np.float64,
    ).reshape((-1, 3))
    return {
        "centers": np.asarray([item[2] for item in cameras]).reshape((-1, 3)),
        "points_xyz": points,
    }, np.asarray([item[0] for item in cameras], dtype=np.int64)


def _registered_camera_poses(
    reconstruction: Any,
) -> list[tuple[int, str, np.ndarray]]:
    values = []
    for image_id, image in sorted(
        reconstruction.images.items(),
        key=lambda item: (_natural_key(str(item[1].name)), int(item[0])),
    ):
        if not image.has_pose:
            continue
        cam_from_world = image.cam_from_world()
        rotation = np.asarray(cam_from_world.rotation.matrix(), dtype=np.float64)
        values.append(
            (
                int(image_id),
                str(image.name),
                -rotation.T @ np.asarray(cam_from_world.translation),
            )
        )
    return values


def _selected_point_ids(reconstruction: Any, point_cap: int | None) -> np.ndarray:
    point_ids = sorted(int(value) for value in reconstruction.point3D_ids())
    limit = _NATIVE_POINT_LIMIT if point_cap is None else point_cap
    if len(point_ids) > limit:
        point_ids = sorted(point_ids, key=lambda value: (_splitmix64(value), value))[:limit]
        point_ids.sort()
    return np.asarray(point_ids, dtype=np.int64)


def _selected_point_indices(point_ids: np.ndarray, point_cap: int | None) -> np.ndarray:
    if point_ids.ndim != 1 or len(np.unique(point_ids)) != len(point_ids):
        raise ValueError("Playback point IDs must be a unique one-dimensional array")
    if point_cap is None or len(point_ids) <= point_cap:
        return np.arange(len(point_ids), dtype=np.int64)
    selected = sorted(
        range(len(point_ids)),
        key=lambda index: (_splitmix64(int(point_ids[index])), int(point_ids[index])),
    )[:point_cap]
    selected.sort(key=lambda index: int(point_ids[index]))
    return np.asarray(selected, dtype=np.int64)


def _selected_capture_points(
    capture: dict[str, Any],
    point_indices: np.ndarray | None,
    point_ids: np.ndarray | None,
    stage: str,
) -> np.ndarray:
    if point_indices is None or point_ids is None:
        raise RuntimeError(f"Playback stage {stage} has no point selection")
    points = np.asarray(capture["points_xyz"])
    if points.ndim != 2 or points.shape != (len(point_ids), 3):
        raise ValueError(f"Playback stage {stage} changed point coordinate count")
    return points[point_indices]


def _capture_identity(capture: dict[str, Any]) -> tuple[str, int]:
    phase = str(capture["phase"])
    if phase not in {"initial", "iteration", "final"}:
        raise ValueError(f"Unsupported solver playback phase: {phase!r}")
    iteration = capture["iteration"]
    if not isinstance(iteration, (int, np.integer)) or isinstance(iteration, (bool, np.bool_)):
        raise ValueError("Solver playback iteration must be an integer")
    iteration = int(iteration)
    if phase == "initial" and iteration != -1:
        raise ValueError("Initial solver playback iteration must be -1")
    if phase == "iteration" and iteration < 0:
        raise ValueError("Solver playback iteration capture must be nonnegative")
    if phase == "final" and iteration < -1:
        raise ValueError("Final solver playback iteration must be at least -1")
    return phase, iteration


def _retain_capture(phase: str, iteration: int, stride: int) -> bool:
    return phase != "iteration" or iteration % stride == 0


def _splitmix64(value: int) -> int:
    value = (value + 0x9E3779B97F4A7C15) & ((1 << 64) - 1)
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & ((1 << 64) - 1)
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & ((1 << 64) - 1)
    return value ^ (value >> 31)


def _database_loop_closure_pairs(database: Path, image_ids: np.ndarray) -> set[tuple[int, int]]:
    valid = set(int(value) for value in image_ids)
    with sqlite3.connect(str(database)) as connection:
        rows = tuple(connection.execute("SELECT image_id, name FROM images"))
        pair_ids = tuple(
            int(row[0]) for row in connection.execute("SELECT pair_id FROM two_view_geometries WHERE rows > 0")
        )
    rank = {
        int(image_id): index
        for index, (image_id, _name) in enumerate(sorted(rows, key=lambda row: (_natural_key(str(row[1])), row[0])))
    }
    pairs = {
        tuple(
            sorted(
                (
                    (pair_id - pair_id % _PAIR_ID_BASE) // _PAIR_ID_BASE,
                    pair_id % _PAIR_ID_BASE,
                )
            )
        )
        for pair_id in pair_ids
    }
    return {
        pair
        for pair in pairs
        if pair[0] in valid
        and pair[1] in valid
        and pair[0] != pair[1]
        and abs(rank[pair[1]] - rank[pair[0]]) > _MIN_LC_RANK_GAP
    }


def _image_names(database: Path, image_ids: np.ndarray) -> list[str]:
    with sqlite3.connect(str(database)) as connection:
        names = {
            int(image_id): str(name) for image_id, name in connection.execute("SELECT image_id, name FROM images")
        }
    missing = [int(image_id) for image_id in image_ids if int(image_id) not in names]
    if missing:
        raise ValueError(f"Playback image IDs are missing from the finalized database: {missing[:5]}")
    return [names[int(image_id)] for image_id in image_ids]


def _normalized_pairs(value: Any) -> np.ndarray:
    pairs = np.asarray(value, dtype=np.int64).reshape((-1, 2)).copy()
    if len(pairs):
        pairs.sort(axis=1)
    return pairs


def _natural_key(value: str) -> tuple[tuple[int, object], ...]:
    return tuple((0, int(part)) if part.isdigit() else (1, part) for part in re.split(r"(\d+)", value) if part)


def _validate_global_positioning_playback_capabilities() -> None:
    from vidmap.mapper.native.extension import native as native_extension

    options = native_extension.GlobalPositioningOptions()
    if "playback" not in dir(options) or "callback" not in dir(options.playback):
        raise RuntimeError("playback_trace requires vidmap_native global positioning callbacks")


def _validate_bundle_adjustment_playback_capabilities() -> None:
    from vidmap.mapper.native.extension import native as native_extension

    options = native_extension.BundleAdjustmentOptions()
    if "playback" not in dir(options) or "callback" not in dir(options.playback):
        raise RuntimeError("playback_trace requires vidmap_native bundle-adjustment callbacks")
