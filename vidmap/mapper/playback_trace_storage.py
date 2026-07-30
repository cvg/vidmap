"""Storage for mapper solver playback traces."""

from __future__ import annotations

import json
import math
import os
import shutil
import tempfile
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Mapping

import numpy as np

__all__ = [
    "PlaybackFrame",
    "PlaybackSnapshot",
    "PlaybackTopology",
    "PlaybackTrace",
    "PlaybackTraceWriter",
]

FORMAT = "vidmap_playback_trace"
VERSION = 9
STAGES = ("gp1", "gp2", "ba1")
VALID_STAGE_ORDERS = {
    ("gp1",),
    ("gp1", "gp2"),
    ("gp1", "ba1"),
    ("gp1", "gp2", "ba1"),
}

_FLOAT_FIELDS = {"centers", "lc_raw_score", "points_xyz"}
_INTEGER_DTYPES = {
    "image_ids": np.dtype("<i8"),
    "lc_pairs": np.dtype("<i8"),
    "lc_support_count": np.dtype("<u8"),
}


@dataclass(frozen=True)
class PlaybackTopology:
    stage: str
    image_ids: np.ndarray
    names: tuple[str, ...] = ()
    lc_pairs: np.ndarray | None = None
    lc_support_count: np.ndarray | None = None
    native_pair_count: int = 0

    def _stored_arrays(self) -> dict[str, np.ndarray]:
        return {
            name: _canonical_array(name, value)
            for name, value in (
                ("image_ids", self.image_ids),
                ("lc_pairs", self.lc_pairs),
                ("lc_support_count", self.lc_support_count),
            )
            if value is not None
        }


@dataclass(frozen=True)
class PlaybackSnapshot:
    stage: str
    topology: int
    centers: np.ndarray
    points_xyz: np.ndarray
    lc_raw_score: np.ndarray | None = None
    terminal: bool = False
    phase: str | None = None
    iteration: int | None = None
    subsolve: int = 0
    elapsed_seconds: float = 0.0

    def _stored_arrays(self) -> dict[str, np.ndarray]:
        return {
            name: _canonical_array(name, value)
            for name, value in (
                ("centers", self.centers),
                ("points_xyz", self.points_xyz),
                ("lc_raw_score", self.lc_raw_score),
            )
            if value is not None
        }


@dataclass(frozen=True)
class PlaybackFrame:
    ordinal: int
    stage: str
    topology: int
    terminal: bool
    phase: str
    iteration: int | None
    subsolve: int
    elapsed_seconds: float | None
    _trace: PlaybackTrace = field(repr=False, compare=False)

    @property
    def image_ids(self) -> np.ndarray:
        return self._topology_array("image_ids")

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self._trace._topologies[self.topology]["names"])

    @property
    def centers(self) -> np.ndarray:
        return self._frame_array("centers")

    @property
    def points_xyz(self) -> np.ndarray:
        return self._frame_array("points_xyz")

    @property
    def lc_pairs(self) -> np.ndarray | None:
        return self._optional_topology_array("lc_pairs")

    @property
    def native_pair_count(self) -> int:
        return int(self._trace._topologies[self.topology]["native_pair_count"])

    @property
    def lc_support_count(self) -> np.ndarray | None:
        return self._optional_topology_array("lc_support_count")

    @property
    def lc_raw_score(self) -> np.ndarray | None:
        return self._optional_frame_array("lc_raw_score")

    def _frame_array(self, name: str) -> np.ndarray:
        return self._trace._array(f"frames/f{self.ordinal:06d}/{name}.npy")

    def _optional_frame_array(self, name: str) -> np.ndarray | None:
        return self._trace._optional_array(f"frames/f{self.ordinal:06d}/{name}.npy")

    def _topology_array(self, name: str) -> np.ndarray:
        return self._trace._array(f"topologies/t{self.topology:06d}/{name}.npy")

    def _optional_topology_array(self, name: str) -> np.ndarray | None:
        return self._trace._optional_array(f"topologies/t{self.topology:06d}/{name}.npy")


class PlaybackTrace:
    def __init__(self, path: Path, manifest: Mapping[str, Any]) -> None:
        self.path = path
        self._manifest = dict(manifest)
        self._frames = tuple(dict(frame) for frame in manifest["frames"])
        self._topologies = tuple(dict(topology) for topology in manifest["topologies"])

    @classmethod
    def load(cls, path: str | Path) -> PlaybackTrace:
        root = Path(path).expanduser().resolve()
        manifest_path = root / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Playback trace manifest is missing: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        _validate_manifest(root, manifest)
        trace = cls(root, manifest)
        trace._validate(complete=True)
        return trace

    @property
    def version(self) -> int:
        return int(self._manifest["version"])

    @property
    def metadata(self) -> dict[str, Any]:
        return deepcopy(self._manifest["metadata"])

    @property
    def stages(self) -> tuple[str, ...]:
        return _stage_runs(self._frames)

    def frame(self, ordinal: int) -> PlaybackFrame:
        if not isinstance(ordinal, int) or isinstance(ordinal, bool) or ordinal < 0 or ordinal >= len(self._frames):
            raise IndexError(f"Playback frame ordinal out of range: {ordinal}")
        entry = self._frames[ordinal]
        return PlaybackFrame(
            ordinal=ordinal,
            stage=entry["stage"],
            topology=entry["topology"],
            terminal=entry["terminal"],
            phase=entry["phase"],
            iteration=entry["iteration"],
            subsolve=entry["subsolve"],
            elapsed_seconds=entry["elapsed_seconds"],
            _trace=self,
        )

    def frames(self, stage: str | None = None) -> Iterator[PlaybackFrame]:
        if stage is not None and stage not in self.stages:
            available = ", ".join(self.stages)
            raise ValueError(f"Playback trace has no stage {stage!r}; available stages: {available}")
        for ordinal in range(len(self._frames)):
            frame = self.frame(ordinal)
            if stage is None or frame.stage == stage:
                yield frame

    def _array(self, relative: str) -> np.ndarray:
        return _load(self.path, relative)

    def _optional_array(self, relative: str) -> np.ndarray | None:
        return _load_optional(self.path, relative)

    def _validate(self, *, complete: bool) -> None:
        _validate_manifest(self.path, self._manifest)
        for index, topology in enumerate(self._topologies):
            _validate_topology(self.path, index, topology)
        for index, frame in enumerate(self._frames):
            _validate_frame(self, index, frame)
        if complete:
            _validate_complete(self)


class PlaybackTraceWriter:
    def __init__(self, final: Path, temporary: Path, metadata: Mapping[str, Any], *, replace: bool) -> None:
        self.final = final
        self.path = temporary
        self.metadata = deepcopy(dict(metadata))
        self.replace = replace
        self.topologies: list[dict[str, Any]] = []
        self.frames: list[dict[str, Any]] = []
        self._state = "open"
        (temporary / "frames").mkdir()
        (temporary / "topologies").mkdir()

    @classmethod
    def create(
        cls,
        path: str | Path,
        *,
        metadata: Mapping[str, Any] | None = None,
        replace: bool = False,
    ) -> PlaybackTraceWriter:
        final = Path(path).expanduser().absolute()
        metadata = (
            {"sampling": {"iteration_stride": 1, "point_cap": None}} if metadata is None else deepcopy(dict(metadata))
        )
        _validate_metadata(final, metadata)
        final.parent.mkdir(parents=True, exist_ok=True)
        destination_exists = os.path.lexists(final)
        if destination_exists and not replace:
            raise FileExistsError(f"Playback trace already exists: {final}")
        if destination_exists and (not final.is_dir() or final.is_symlink()):
            raise FileExistsError(f"Playback trace destination is not a replaceable directory: {final}")
        temporary = Path(tempfile.mkdtemp(prefix=f".{final.name}.", suffix=".tmp", dir=final.parent))
        try:
            return cls(final, temporary, metadata, replace=replace)
        except BaseException:
            shutil.rmtree(temporary, ignore_errors=True)
            raise

    def __enter__(self) -> PlaybackTraceWriter:
        self._require_open()
        return self

    def __exit__(self, _exc_type: object, _exc: object, _traceback: object) -> None:
        if self._state == "open":
            self.discard()

    def add_topology(self, topology: PlaybackTopology) -> int:
        self._require_open()
        index = len(self.topologies)
        directory = self.path / "topologies" / f"t{index:06d}"
        entry = {
            "stage": topology.stage,
            "names": list(topology.names),
            "native_pair_count": topology.native_pair_count,
        }
        try:
            self._write_arrays(directory, topology._stored_arrays())
            _validate_topology(self.path, index, entry)
        except Exception:
            shutil.rmtree(directory, ignore_errors=True)
            raise
        self.topologies.append(entry)
        return index

    def append_frame(self, snapshot: PlaybackSnapshot) -> int:
        self._require_open()
        return self._append_stored_frame(snapshot, snapshot._stored_arrays())

    def append_frame_if_changed(self, snapshot: PlaybackSnapshot) -> int | None:
        """Append a frame unless it is byte-identical to the latest stored state."""
        self._require_open()
        arrays = snapshot._stored_arrays()
        phase = snapshot.phase
        if phase is None:
            phase = "terminal" if snapshot.terminal else "initial"
        iteration = snapshot.iteration
        if phase == "initial" and iteration is None:
            iteration = -1
        if self.frames:
            previous = self.snapshot().frame(len(self.frames) - 1)
            previous_score = previous.lc_raw_score
            score = None if snapshot.lc_raw_score is None else arrays["lc_raw_score"]
            same = (
                previous.stage == snapshot.stage
                and previous.topology == snapshot.topology
                and previous.terminal == snapshot.terminal
                and previous.phase == phase
                and previous.iteration == iteration
                and previous.subsolve == snapshot.subsolve
                and _stored_array_equal(previous.centers, arrays["centers"])
                and _stored_array_equal(previous.points_xyz, arrays["points_xyz"])
                and (previous_score is None) == (score is None)
            )
            if score is not None and previous_score is not None:
                same = same and _stored_array_equal(previous_score, score)
            if same:
                return None
        return self._append_stored_frame(snapshot, arrays)

    def _append_stored_frame(
        self,
        snapshot: PlaybackSnapshot,
        arrays: Mapping[str, np.ndarray],
    ) -> int:
        self._require_open()
        topology = snapshot.topology
        if (
            not isinstance(topology, int)
            or isinstance(topology, bool)
            or topology < 0
            or topology >= len(self.topologies)
        ):
            raise ValueError(f"Unknown playback topology index: {topology!r}")
        ordinal = len(self.frames)
        directory = self.path / "frames" / f"f{ordinal:06d}"
        phase = snapshot.phase
        if phase is None:
            phase = "terminal" if snapshot.terminal else "initial"
        iteration = snapshot.iteration
        if phase == "initial" and iteration is None:
            iteration = -1
        entry = {
            "stage": snapshot.stage,
            "topology": topology,
            "terminal": snapshot.terminal,
            "phase": phase,
            "iteration": iteration,
            "subsolve": snapshot.subsolve,
            "elapsed_seconds": snapshot.elapsed_seconds,
        }
        try:
            self._write_arrays(directory, arrays)
            trace = PlaybackTrace(self.path, self._manifest(frames=[*self.frames, entry]))
            _validate_frame(trace, ordinal, entry)
        except Exception:
            shutil.rmtree(directory, ignore_errors=True)
            raise
        self.frames.append(entry)
        return ordinal

    def snapshot(self) -> PlaybackTrace:
        if self._state == "discarded":
            raise RuntimeError("Playback trace writer has been discarded")
        return PlaybackTrace(self.path, self._manifest())

    def complete(self) -> None:
        self._require_open()
        trace = PlaybackTrace(self.path, self._manifest())
        trace._validate(complete=True)
        manifest_json = json.dumps(trace._manifest, allow_nan=False, separators=(",", ":"), sort_keys=True) + "\n"
        (self.path / "manifest.json").write_text(manifest_json, encoding="utf-8")
        if os.path.lexists(self.final):
            if not self.replace:
                raise FileExistsError(f"Playback trace destination appeared before publish: {self.final}")
            if not self.final.is_dir() or self.final.is_symlink():
                raise FileExistsError(f"Playback trace destination is not a replaceable directory: {self.final}")
            self._replace_published_trace()
        else:
            self.path.rename(self.final)
        self.path = self.final
        self._state = "published"

    def _replace_published_trace(self) -> None:
        backup_root = Path(tempfile.mkdtemp(prefix=f".{self.final.name}.", suffix=".backup", dir=self.final.parent))
        backup = backup_root / self.final.name
        try:
            self.final.rename(backup)
        except BaseException:
            shutil.rmtree(backup_root, ignore_errors=True)
            raise
        try:
            self.path.rename(self.final)
        except BaseException:
            backup.rename(self.final)
            shutil.rmtree(backup_root, ignore_errors=True)
            raise
        shutil.rmtree(backup_root, ignore_errors=True)

    def discard(self) -> None:
        if self._state == "open":
            shutil.rmtree(self.path, ignore_errors=True)
            self._state = "discarded"

    def _manifest(self, *, frames: list[dict[str, Any]] | None = None) -> dict[str, Any]:
        return {
            "format": FORMAT,
            "version": VERSION,
            "metadata": self.metadata,
            "topologies": self.topologies,
            "frames": self.frames if frames is None else frames,
        }

    def _require_open(self) -> None:
        if self._state != "open":
            raise RuntimeError(f"Playback trace writer is {self._state}")

    @staticmethod
    def _write_arrays(directory: Path, arrays: Mapping[str, np.ndarray]) -> None:
        directory.mkdir()
        for name, value in arrays.items():
            np.save(
                directory / f"{name}.npy",
                value,
                allow_pickle=False,
            )


def _validate_manifest(root: Path, manifest: Any) -> None:
    if not isinstance(manifest, dict):
        raise ValueError(f"{root}: playback trace manifest must be a JSON object")
    version = manifest.get("version")
    expected_fields = {"format", "version", "metadata", "topologies", "frames"}
    if set(manifest) != expected_fields:
        raise ValueError(f"{root}: unsupported playback trace manifest fields")
    if (
        manifest.get("format") != FORMAT
        or not isinstance(version, int)
        or isinstance(version, bool)
        or version != VERSION
    ):
        raise ValueError(f"{root}: unsupported playback trace schema")
    _validate_metadata(root, manifest["metadata"])
    if not isinstance(manifest.get("topologies"), list) or not isinstance(manifest.get("frames"), list):
        raise ValueError(f"{root}: playback trace manifest requires topology and frame lists")
    for topology in manifest["topologies"]:
        if not isinstance(topology, dict) or set(topology) != {
            "stage",
            "names",
            "native_pair_count",
        }:
            raise ValueError(f"{root}: invalid topology descriptor")
    for frame in manifest["frames"]:
        frame_fields = {
            "stage",
            "topology",
            "terminal",
            "phase",
            "iteration",
            "subsolve",
            "elapsed_seconds",
        }
        if not isinstance(frame, dict) or set(frame) != frame_fields:
            raise ValueError(f"{root}: invalid frame descriptor")


def _validate_metadata(root: Path, metadata: Any) -> None:
    if not isinstance(metadata, dict) or set(metadata) != {"sampling"}:
        raise ValueError(f"{root}: playback trace metadata requires exactly one sampling object")
    sampling = metadata["sampling"]
    if not isinstance(sampling, dict) or set(sampling) != {
        "iteration_stride",
        "point_cap",
    }:
        raise ValueError(f"{root}: invalid playback sampling metadata")
    stride = sampling["iteration_stride"]
    if not isinstance(stride, int) or isinstance(stride, bool) or stride < 1:
        raise ValueError(f"{root}: playback iteration_stride must be a positive integer")
    point_cap = sampling["point_cap"]
    if point_cap is not None and (not isinstance(point_cap, int) or isinstance(point_cap, bool) or point_cap < 1):
        raise ValueError(f"{root}: playback point_cap must be null or a positive integer")
    if point_cap is not None and point_cap > 200000:
        raise ValueError(f"{root}: playback point_cap cannot exceed 200000")


def _validate_topology(root: Path, index: int, topology: Mapping[str, Any]) -> None:
    stage = topology.get("stage")
    if stage not in STAGES:
        raise ValueError(f"{root}: unsupported topology stage {stage!r}")
    names = topology.get("names")
    if not isinstance(names, list) or any(not isinstance(name, str) for name in names):
        raise ValueError(f"{root}: topology names must be a list of strings")
    native_pair_count = topology.get("native_pair_count")
    if not isinstance(native_pair_count, int) or isinstance(native_pair_count, bool) or native_pair_count < 0:
        raise ValueError(f"{root}: native_pair_count must be a nonnegative integer")

    image_ids = _load(root, f"topologies/t{index:06d}/image_ids.npy")
    if image_ids.ndim != 1:
        raise ValueError(f"image_ids must have shape (N,), got {tuple(image_ids.shape)}")
    _dtype(image_ids, _INTEGER_DTYPES["image_ids"], "image_ids")
    if len(names) != len(image_ids):
        raise ValueError(f"{root}: topology names length does not match image_ids")
    if len(np.unique(image_ids)) != len(image_ids):
        raise ValueError(f"{root}: image_ids contains duplicates")

    pairs = _load_optional(root, f"topologies/t{index:06d}/lc_pairs.npy")
    support = _load_optional(root, f"topologies/t{index:06d}/lc_support_count.npy")
    if pairs is None:
        if support is not None:
            raise ValueError(f"{root}: lc_support_count requires lc_pairs")
        if native_pair_count:
            raise ValueError(f"{root}: native_pair_count requires lc_pairs")
        pair_count = 0
    else:
        if pairs.ndim != 2 or pairs.shape[1:] != (2,):
            raise ValueError(f"lc_pairs must have shape (N, 2), got {tuple(pairs.shape)}")
        _dtype(pairs, _INTEGER_DTYPES["lc_pairs"], "lc_pairs")
        valid_ids = set(int(value) for value in image_ids)
        if any(int(value) not in valid_ids for value in pairs.reshape(-1)):
            raise ValueError(f"{root}: lc_pairs references an unknown image id")
        pair_count = len(pairs)
        if native_pair_count > pair_count:
            raise ValueError(f"{root}: native_pair_count exceeds lc_pairs length")
        if support is not None:
            _shape(support, (pair_count,), "lc_support_count")
            _dtype(support, _INTEGER_DTYPES["lc_support_count"], "lc_support_count")


def _validate_frame(trace: PlaybackTrace, index: int, frame: Mapping[str, Any]) -> None:
    stage = frame.get("stage")
    topology = frame.get("topology")
    if stage not in STAGES:
        raise ValueError(f"{trace.path}: unsupported frame stage {stage!r}")
    if (
        not isinstance(topology, int)
        or isinstance(topology, bool)
        or topology < 0
        or topology >= len(trace._topologies)
    ):
        raise ValueError(f"{trace.path}: frame topology index is out of range")
    if not isinstance(frame.get("terminal"), bool):
        raise ValueError(f"{trace.path}: frame terminal flag must be boolean")
    if trace._topologies[topology]["stage"] != stage:
        raise ValueError(f"{trace.path}: frame stage does not match topology stage")
    _validate_frame_metadata(trace.path, frame)

    image_ids = _load(trace.path, f"topologies/t{topology:06d}/image_ids.npy")
    centers = _load(trace.path, f"frames/f{index:06d}/centers.npy")
    _shape(centers, (len(image_ids), 3), "centers")
    _float32(centers, "centers")
    _finite(centers, "centers")
    points = _load(trace.path, f"frames/f{index:06d}/points_xyz.npy")
    if points.ndim != 2 or points.shape[1:] != (3,):
        raise ValueError(f"points_xyz must have shape (N, 3), got {tuple(points.shape)}")
    _float32(points, "points_xyz")
    _finite(points, "points_xyz")

    pairs = _load_optional(trace.path, f"topologies/t{topology:06d}/lc_pairs.npy")
    scores = _load_optional(trace.path, f"frames/f{index:06d}/lc_raw_score.npy")
    native_pair_count = int(trace._topologies[topology]["native_pair_count"])
    if stage in {"gp1", "gp2"} and pairs is not None and scores is None:
        raise ValueError(f"{trace.path}: GP lc_pairs require lc_raw_score")
    if stage == "ba1" and scores is not None:
        raise ValueError(f"{trace.path}: BA frames cannot contain lc_raw_score")
    if scores is not None:
        if pairs is None:
            raise ValueError(f"{trace.path}: lc_raw_score requires lc_pairs")
        _shape(scores, (native_pair_count,), "lc_raw_score")
        _float32(scores, "lc_raw_score")
        _finite(scores, "lc_raw_score")


def _validate_complete(trace: PlaybackTrace) -> None:
    if not trace._frames:
        raise ValueError("Complete playback trace has no frames")
    stages = _stage_runs(trace._frames)
    if stages not in VALID_STAGE_ORDERS:
        raise ValueError(f"Complete playback trace has unsupported stage order: {stages}")
    terminals = [index for index, frame in enumerate(trace._frames) if frame["terminal"]]
    if terminals != [len(trace._frames) - 1]:
        raise ValueError("Complete playback trace requires exactly one final terminal frame")
    if len(trace._frames) == 1:
        raise ValueError("Complete playback trace requires a nonterminal solver frame")
    _validate_solver_lifecycle(trace)
    elapsed = [float(frame["elapsed_seconds"]) for frame in trace._frames]
    if any(following < previous for previous, following in zip(elapsed, elapsed[1:])):
        raise ValueError("Complete playback trace requires nondecreasing elapsed_seconds")


def _validate_solver_lifecycle(trace: PlaybackTrace) -> None:
    solver_frames = trace._frames[:-1]
    terminal = trace._frames[-1]
    current: tuple[str, int] | None = None
    seen: set[tuple[str, int]] = set()
    finished = False
    last_iteration = -1
    for frame in solver_frames:
        identity = (frame["stage"], frame["subsolve"])
        phase = frame["phase"]
        iteration = int(frame["iteration"])
        if identity != current:
            if current is not None and not finished:
                raise ValueError(f"Complete playback trace is missing final capture for {current[0]}/{current[1]}")
            if identity in seen:
                raise ValueError(f"Complete playback trace repeats noncontiguous subsolve {identity[0]}/{identity[1]}")
            if phase != "initial":
                raise ValueError(
                    f"Complete playback trace subsolve {identity[0]}/{identity[1]} must start with initial"
                )
            seen.add(identity)
            current = identity
            finished = False
            last_iteration = -1
        elif finished:
            raise ValueError(f"Complete playback trace has a capture after final for {identity[0]}/{identity[1]}")
        elif phase == "initial":
            raise ValueError(f"Complete playback trace has duplicate initial capture for {identity[0]}/{identity[1]}")

        if phase == "iteration":
            if iteration <= last_iteration:
                raise ValueError(
                    f"Complete playback trace has non-increasing iteration for {identity[0]}/{identity[1]}"
                )
            last_iteration = iteration
        elif phase == "final":
            if iteration < last_iteration:
                raise ValueError(f"Complete playback trace final iteration regressed for {identity[0]}/{identity[1]}")
            finished = True

    if current is None or not finished:
        label = "unknown" if current is None else f"{current[0]}/{current[1]}"
        raise ValueError(f"Complete playback trace is missing final capture for {label}")
    terminal_identity = (terminal["stage"], terminal["subsolve"])
    if terminal_identity != current:
        raise ValueError("Complete playback trace terminal identity must match the final solver subsolve")


def _validate_frame_metadata(root: Path, frame: Mapping[str, Any]) -> None:
    phase = frame.get("phase")
    if phase not in {"initial", "iteration", "final", "terminal"}:
        raise ValueError(f"{root}: unsupported playback phase {phase!r}")
    terminal = frame["terminal"]
    if terminal != (phase == "terminal"):
        raise ValueError(f"{root}: terminal playback phase and flag must agree")
    iteration = frame.get("iteration")
    valid_iteration = isinstance(iteration, int) and not isinstance(iteration, bool)
    if phase == "initial" and iteration != -1:
        raise ValueError(f"{root}: initial playback iteration must be -1")
    if phase == "iteration" and (not valid_iteration or iteration < 0):
        raise ValueError(f"{root}: iteration playback phase requires a nonnegative iteration")
    if phase == "final" and (not valid_iteration or iteration < -1):
        raise ValueError(f"{root}: final playback phase requires an iteration of at least -1")
    if phase == "terminal" and iteration is not None:
        raise ValueError(f"{root}: terminal playback iteration must be null")
    subsolve = frame.get("subsolve")
    if not isinstance(subsolve, int) or isinstance(subsolve, bool) or subsolve < 0:
        raise ValueError(f"{root}: playback subsolve must be a nonnegative integer")
    elapsed = frame.get("elapsed_seconds")
    if not isinstance(elapsed, (int, float)) or isinstance(elapsed, bool) or not math.isfinite(elapsed) or elapsed < 0:
        raise ValueError(f"{root}: playback elapsed_seconds must be finite and nonnegative")


def _stage_runs(frames: tuple[dict[str, Any], ...]) -> tuple[str, ...]:
    stages: list[str] = []
    for frame in frames:
        if not stages or stages[-1] != frame["stage"]:
            stages.append(frame["stage"])
    return tuple(stages)


def _stored_array_equal(left: np.ndarray, right: np.ndarray) -> bool:
    return (
        left.shape == right.shape
        and left.dtype == right.dtype
        and np.array_equal(left.view(np.uint8), right.view(np.uint8))
    )


def _canonical_array(name: str, value: np.ndarray) -> np.ndarray:
    array = np.asarray(value)
    if name in _FLOAT_FIELDS:
        if array.dtype.kind not in {"f", "i", "u"}:
            raise ValueError(f"{name} must be numeric, got {array.dtype}")
        with np.errstate(over="ignore", invalid="ignore"):
            return np.array(array, dtype="<f4", order="C", copy=True)
    if name not in _INTEGER_DTYPES or array.dtype.kind not in {"i", "u"}:
        raise ValueError(f"{name} must be an integer array, got {array.dtype}")
    dtype = _INTEGER_DTYPES[name]
    limits = np.iinfo(dtype)
    if array.size and (np.any(array < limits.min) or np.any(array > limits.max)):
        raise ValueError(f"{name} contains a value outside {dtype.str}")
    return np.array(array, dtype=dtype, order="C", copy=True)


def _load(root: Path, relative: str) -> np.ndarray:
    path = root / relative
    if not path.is_file():
        raise FileNotFoundError(f"Playback trace array is missing: {path}")
    array = np.load(path, mmap_mode="r", allow_pickle=False)
    array.flags.writeable = False
    return array


def _load_optional(root: Path, relative: str) -> np.ndarray | None:
    path = root / relative
    return _load(root, relative) if path.is_file() else None


def _shape(array: np.ndarray, expected: tuple[int, ...], name: str) -> None:
    if tuple(array.shape) != expected:
        raise ValueError(f"{name} must have shape {expected}, got {tuple(array.shape)}")


def _dtype(array: np.ndarray, expected: np.dtype[Any], name: str) -> None:
    if array.dtype != expected:
        raise ValueError(f"{name} must have dtype {expected.str}, got {array.dtype.str}")


def _float32(array: np.ndarray, name: str) -> None:
    _dtype(array, np.dtype("<f4"), name)


def _finite(array: np.ndarray, name: str) -> None:
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
