from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from vidmap.mapper.playback_trace_storage import PlaybackFrame, PlaybackTrace

from .alignment import (
    Similarity,
    estimate_model_to_ground_truth_alignment,
    estimate_stage_similarities,
    estimate_trace_to_model_alignment,
    resolve_ground_truth_model_path,
    similarity_from_pycolmap,
)
from .api import Playback
from .scene import (
    Frame,
    Palette,
    Resolution,
    Scene,
    Sequence,
    compute_scene_framing,
    estimate_path_up_direction,
    theme_palette,
)

MAX_EDGES = 5000
FORCE_REFERENCE = 100.0


@dataclass(frozen=True)
class _PlaybackPlan:
    global_positioning_stages: tuple[tuple[tuple[PlaybackFrame, int], ...], ...]
    bundle_adjustment_frames: tuple[PlaybackFrame, ...]
    bundle_adjustment_frame_indices: tuple[int, ...]
    global_positioning_duration: float
    duration: float


def build_solver_playback_sequence(playback: Playback, resolution: Resolution) -> Sequence:
    colors = theme_palette(playback.theme)
    trace = PlaybackTrace.load(_trace_path(Path(playback.source)))
    recorded_gp = tuple(frame for frame in trace.frames() if frame.stage in {"gp1", "gp2"})
    if not recorded_gp:
        raise FileNotFoundError(f"Compact trace has no GP frames: {trace.path}")
    recorded_ba = tuple(trace.frames("ba1")) if "ba1" in trace.stages and not playback.gp_only else ()
    if not playback.gp_only and "ba1" not in trace.stages:
        warnings.warn(
            f"BA1 is unavailable in {trace.path}; replaying GP only.",
            RuntimeWarning,
            stacklevel=2,
        )

    final_gp = next(frame for frame in reversed(recorded_gp) if not frame.terminal)
    gp_pairs, gp_colors = _global_positioning_overlay(final_gp, colors)
    final_recorded = (
        recorded_ba[-1]
        if recorded_ba
        else next((frame for frame in reversed(recorded_gp) if frame.terminal), final_gp)
    )
    recorded_stages = tuple(
        tuple(trace.frames(stage))
        for stage in trace.stages
        if stage in {"gp1", "gp2"} or (stage == "ba1" and recorded_ba)
    )
    alignments, gt = _stage_world_alignments(
        Path(playback.source),
        recorded_stages,
        align_to_ground_truth=playback.align_to_ground_truth,
    )
    color_lookup = _loop_closure_color_lookup(gp_pairs, gp_colors)
    final_overlay = (
        _bundle_adjustment_overlay(final_recorded, color_lookup)
        if final_recorded.stage == "ba1"
        else _global_positioning_overlay(final_recorded, colors)
    )
    final = _frame_from_solver_snapshot(
        final_recorded,
        resolution,
        alignments[final_recorded.stage],
        *final_overlay,
    )
    reference = gt if len(gt) else final.centers
    up = np.asarray([0.0, 0.0, 1.0]) if len(gt) else estimate_path_up_direction(reference)
    up, scale = compute_scene_framing(reference, up, resolution, playback.view)
    scene = Scene(np.empty((0, 3)), np.empty((0, 3)), gt, reference, up, scale)

    if playback.mode == "final":
        frames = (replace(final, elapsed=0.0),)
        duration = 1.0
    else:
        plan = _build_playback_plan(recorded_gp, recorded_ba)
        frames = _iter_playback_timeline(
            plan,
            resolution,
            alignments,
            color_lookup,
            final,
            colors,
        )
        duration = plan.duration
    return Sequence(scene, frames, playback.view, resolution, colors, duration)


def _build_playback_plan(
    global_positioning_frames: tuple[PlaybackFrame, ...],
    bundle_adjustment_frames: tuple[PlaybackFrame, ...],
) -> _PlaybackPlan:
    global_positioning_stages = tuple(
        _build_global_positioning_stage_plan(
            tuple(frame for frame in global_positioning_frames if frame.stage == stage)
        )
        for stage in dict.fromkeys(frame.stage for frame in global_positioning_frames)
    )
    global_positioning_duration = float(sum(max(0, len(stage) - 1) for stage in global_positioning_stages))
    global_positioning_selection_units = sum(
        sum(1 + interpolation_count for _frame, interpolation_count in stage[1:])
        for stage in global_positioning_stages
    )
    bundle_adjustment_frame_indices = (
        _select_bundle_adjustment_frames(len(bundle_adjustment_frames), global_positioning_selection_units)
        if bundle_adjustment_frames
        else ()
    )
    playback_duration = global_positioning_duration
    if bundle_adjustment_frame_indices:
        playback_duration += 0.5 * global_positioning_duration
    return _PlaybackPlan(
        global_positioning_stages,
        bundle_adjustment_frames,
        bundle_adjustment_frame_indices,
        global_positioning_duration,
        max(playback_duration, 1.0),
    )


def _build_global_positioning_stage_plan(
    recorded: tuple[PlaybackFrame, ...],
) -> tuple[tuple[PlaybackFrame, int], ...]:
    selected = [(recorded[0], 0)]
    current = recorded[0]
    for index, following in enumerate(recorded[1:], start=1):
        compatible = _recorded_compatible(current, following)
        ratio = _recorded_motion_ratio(current, following) if compatible else None
        if compatible and index < len(recorded) - 1 and ratio is not None and ratio <= 0.1:
            continue
        selected.append((following, _interpolation_count(ratio)))
        current = following
    return tuple(selected)


def _iter_playback_timeline(
    plan: _PlaybackPlan,
    resolution: Resolution,
    alignments: dict[str, Similarity],
    color_lookup: dict[tuple[int, int], np.ndarray],
    final: Frame,
    palette: Palette,
):
    global_positioning_elapsed = 0.0
    for stage_index, stage in enumerate(plan.global_positioning_stages):
        emitted = _iter_global_positioning_frames(
            stage,
            resolution,
            alignments[stage[0][0].stage],
            palette,
            global_positioning_elapsed,
        )
        pending = next(emitted)
        for frame in emitted:
            yield pending
            pending = frame
        if (
            stage_index == len(plan.global_positioning_stages) - 1
            and not plan.bundle_adjustment_frames
            and stage[-1][0].terminal
        ):
            pending = replace(final, elapsed=pending.elapsed)
        yield pending
        global_positioning_elapsed += max(0, len(stage) - 1)
    if not plan.bundle_adjustment_frame_indices:
        return
    bundle_adjustment_duration = plan.global_positioning_duration * 0.5
    interval = bundle_adjustment_duration / max(1, len(plan.bundle_adjustment_frame_indices) - 1)
    for position, index in enumerate(plan.bundle_adjustment_frame_indices):
        elapsed = global_positioning_elapsed + position * interval
        if index == len(plan.bundle_adjustment_frames) - 1:
            yield replace(final, elapsed=elapsed)
        else:
            pairs, colors = _bundle_adjustment_overlay(plan.bundle_adjustment_frames[index], color_lookup)
            yield replace(
                _frame_from_solver_snapshot(
                    plan.bundle_adjustment_frames[index],
                    resolution,
                    alignments[plan.bundle_adjustment_frames[index].stage],
                    pairs,
                    colors,
                ),
                elapsed=elapsed,
            )


def _iter_global_positioning_frames(
    plan: tuple[tuple[PlaybackFrame, int], ...],
    resolution: Resolution,
    alignment: Similarity,
    palette: Palette,
    start: float = 0.0,
):
    current = replace(
        _frame_from_solver_snapshot(
            plan[0][0],
            resolution,
            alignment,
            *_global_positioning_overlay(plan[0][0], palette),
        ),
        elapsed=start,
    )
    yield current
    for item, interpolation_count in plan[1:]:
        following = _frame_from_solver_snapshot(
            item,
            resolution,
            alignment,
            *_global_positioning_overlay(item, palette),
        )
        following_elapsed = current.elapsed + 1.0
        for step in range(1, interpolation_count + 1):
            alpha = step / (interpolation_count + 1)
            yield replace(
                current,
                centers=current.centers + alpha * (following.centers - current.centers),
                points=current.points + alpha * (following.points - current.points),
                elapsed=current.elapsed + alpha,
            )
        following = replace(following, elapsed=following_elapsed)
        yield following
        current = following


def _frame_from_solver_snapshot(
    recorded: PlaybackFrame,
    resolution: Resolution,
    alignment: Similarity,
    pairs: np.ndarray | None,
    colors: np.ndarray | None,
) -> Frame:
    centers = _apply_similarity(alignment, recorded.centers)
    points = _apply_similarity(alignment, recorded.points_xyz)
    return Frame(
        recorded.stage,
        tuple(int(value) for value in recorded.image_ids),
        centers,
        points,
        pairs,
        colors,
    )


def _global_positioning_overlay(
    recorded: PlaybackFrame, palette: Palette
) -> tuple[np.ndarray | None, np.ndarray | None]:
    if recorded.lc_pairs is None or recorded.lc_raw_score is None:
        return None, None
    pairs = np.asarray(recorded.lc_pairs)
    support = (
        np.zeros(len(pairs), dtype=np.uint64)
        if recorded.lc_support_count is None
        else np.asarray(recorded.lc_support_count)
    )
    native_count = recorded.native_pair_count
    optimized = sorted(
        range(native_count),
        key=lambda i: (-int(support[i]), int(pairs[i, 0]), int(pairs[i, 1])),
    )
    rejected = sorted(range(native_count, len(pairs)), key=lambda i: tuple(pairs[i]))
    selected = np.asarray(optimized[:MAX_EDGES] + rejected, dtype=np.int64)
    if not len(selected):
        return None, None
    scores = np.zeros(len(selected), dtype=np.float32)
    native = selected < native_count
    scores[native] = np.asarray(recorded.lc_raw_score)[selected[native]]
    ratio = np.clip(np.nan_to_num(scores, nan=0.0) / FORCE_REFERENCE, 0.0, 1.0)[:, None]
    rejected_color = np.asarray(palette.rejected)
    accepted_color = np.asarray(palette.accepted)
    colors = np.rint(rejected_color + ratio * (accepted_color - rejected_color)).astype(np.uint8)
    return np.asarray(pairs[selected], dtype=np.int64), colors


def _loop_closure_color_lookup(
    pairs: np.ndarray | None, colors: np.ndarray | None
) -> dict[tuple[int, int], np.ndarray]:
    if pairs is None or colors is None:
        return {}
    return {
        tuple(sorted((int(left), int(right)))): np.asarray(color, dtype=np.uint8)
        for (left, right), color in zip(pairs, colors, strict=True)
    }


def _bundle_adjustment_overlay(
    recorded: PlaybackFrame, lookup: dict[tuple[int, int], np.ndarray]
) -> tuple[np.ndarray | None, np.ndarray | None]:
    if recorded.lc_pairs is None or not lookup:
        return None, None
    pairs, colors = [], []
    for pair in np.asarray(recorded.lc_pairs):
        key = tuple(sorted((int(pair[0]), int(pair[1]))))
        color = lookup.get(key)
        if color is not None:
            pairs.append(pair)
            colors.append(color)
    if not pairs:
        return None, None
    return np.asarray(pairs, dtype=np.int64), np.asarray(colors, dtype=np.uint8)


def _recorded_compatible(left: PlaybackFrame, right: PlaybackFrame) -> bool:
    return (
        left.topology == right.topology
        and np.array_equal(left.image_ids, right.image_ids)
        and left.centers.shape == right.centers.shape
        and left.points_xyz.shape == right.points_xyz.shape
    )


def _recorded_motion_ratio(left: PlaybackFrame, right: PlaybackFrame) -> float | None:
    left_centers = np.asarray(left.centers)
    right_centers = np.asarray(right.centers)
    motion = (
        float(np.percentile(np.linalg.norm(right_centers - left_centers, axis=1), 90)) if len(left_centers) else 0.0
    )
    if len(left_centers) < 2:
        return None
    order = np.argsort(np.asarray(left.image_ids))
    distances = np.linalg.norm(np.diff(left_centers[order], axis=0), axis=1)
    distances = distances[distances > 0.0]
    return None if not len(distances) else motion / float(np.median(distances))


def _interpolation_count(ratio: float | None) -> int:
    return 0 if ratio is None or ratio <= 0.02 else min(10, max(1, math.ceil(10 * min(1.0, (ratio - 0.02) / 0.98))))


def _select_bundle_adjustment_frames(frame_count: int, global_positioning_selection_units: int) -> tuple[int, ...]:
    transition_count = frame_count - 1
    target_transition_count = max(1, int(round(global_positioning_selection_units * 0.5)))
    if transition_count <= 0 or target_transition_count >= transition_count:
        return tuple(range(frame_count))
    return tuple(sorted({int(round(value)) for value in np.linspace(0, frame_count - 1, target_transition_count + 1)}))


def _stage_world_alignments(
    source: Path,
    stages: tuple[tuple[PlaybackFrame, ...], ...],
    *,
    align_to_ground_truth: bool = True,
) -> tuple[dict[str, Similarity], np.ndarray]:
    run = source.parent if source.name == "playback_trace" else source
    gt_path = resolve_ground_truth_model_path(run) if align_to_ground_truth else None
    final_to_world = Similarity.identity()
    gt = np.empty((0, 3), dtype=np.float64)
    if gt_path is not None:
        model_path = next(
            (run / name for name in ("rec", "rec-ba", "rec-gp") if (run / name).is_dir()),
            None,
        )
        if model_path is None:
            raise FileNotFoundError(f"Cannot align playback to GT without a saved model in {run}")
        import pycolmap

        saved_model = pycolmap.Reconstruction(model_path)
        trace_to_model = estimate_trace_to_model_alignment(stages[-1][-1], saved_model)
        model_to_gt, gt = estimate_model_to_ground_truth_alignment(saved_model, gt_path, run=run)
        if model_to_gt is None:
            raise ValueError(f"Could not align saved model {model_path} to GT {gt_path}")
        final_to_world = trace_to_model.followed_by(similarity_from_pycolmap(model_to_gt))
    return estimate_stage_similarities(stages, final_to_world), gt


def _apply_similarity(alignment: Similarity | None, values: np.ndarray) -> np.ndarray:
    values = np.asarray(values).reshape((-1, 3))
    return values if alignment is None else alignment.transform(values)


def _trace_path(source: Path) -> Path:
    if (source / "manifest.json").is_file():
        return source
    path = source / "playback_trace"
    if (path / "manifest.json").is_file():
        return path
    if (source / "gp_trace").exists() or (source / "ba_trace").exists():
        raise ValueError(f"{source} contains an unsupported trace layout; regenerate playback_trace")
    raise FileNotFoundError(f"Compact playback trace is missing: {path / 'manifest.json'}")
