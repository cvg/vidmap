from __future__ import annotations

import bisect
import re
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import yaml
from natsort import natsorted
from scipy.spatial.transform import Rotation, RotationSpline, Slerp

from vidmap.depth_artifacts import load_depth_scales, resolve_depth_paths
from vidmap.utils.io import read_image
from vidmap.visualization.point_quality import lowest_covariance_point_ids

from .alignment import estimate_model_to_ground_truth_alignment, resolve_ground_truth_model_path
from .api import Playback, Resolution
from .dataset_roots import select_dataset_root
from .depth_lift import DepthLiftReader
from .scene import (
    DEPTH_LIFT_RADIUS_UI_POINTS,
    Camera,
    Frame,
    Palette,
    PointBatch,
    Pose,
    Scene,
    Sequence,
    camera_rotation,
    compute_scene_framing,
    estimate_path_up_direction,
    normalize_vector,
    theme_palette,
)

CHASE_DISTANCE = 6.0
CHASE_SMOOTHING_SECONDS = 0.6
KEYFRAME_HISTORY_LENGTH = 9
_TIMESTAMP = re.compile(r"^(\d+(?:\.\d+)?)(?:-|$)")


def build_model_flythrough_sequence(playback: Playback, resolution: Resolution) -> Sequence:
    colors = theme_palette(playback.theme)
    run_root, _config_path, model_path, model, image_names, rgb_dir, gt_path = _resolve_flythrough_inputs(playback)
    from vidmap.reconstruction import decoded_video_frame_timestamps

    frame_timestamps = decoded_video_frame_timestamps(rgb_dir)
    alignment, gt_centers = estimate_model_to_ground_truth_alignment(
        model,
        gt_path if playback.align_to_ground_truth else None,
        run=run_root,
    )
    poses, point_ids, points, observations = _load_sparse_model_scene(
        model,
        rgb_dir,
        alignment,
        colors.neutral,
        covariance_percentile=playback.sparse_point_covariance_percentile,
    )
    if not poses:
        raise ValueError(f"Selected model has no finite registered camera poses: {model_path}")

    images = _image_paths(rgb_dir, tuple(poses))
    path_names, _elapsed = _elapsed_timestamps(
        tuple(name for name in poses if name in images),
        timestamps=frame_timestamps,
    )
    if len(path_names) < 2:
        raise ValueError(f"Could not match model image names to RGB frames under {rgb_dir}")
    support = _support_by_current(observations, path_names)
    depth_lift_reader = None
    model_images = {str(image.name): image for image in model.images.values() if image.has_pose}
    if playback.depth_lift:
        full_depth_path, sampled_depth_path = resolve_depth_paths(run_root, playback.depth_maps)
        depth_lift_reader = DepthLiftReader(
            full_depth_path,
            sampled_depth_path,
            model,
            rgb_dir,
            scale_by_image_id={} if playback.depth_lift_refit_scale else load_depth_scales(run_root),
        )

    reference = gt_centers if len(gt_centers) else np.asarray([poses[name].center for name in path_names])
    world_unit = 1.0 if len(gt_centers) else _estimate_path_unit(reference)
    up = np.asarray([0.0, 0.0, 1.0]) if len(gt_centers) else estimate_path_up_direction(reference)
    up, scale = compute_scene_framing(reference, up, resolution, playback.view)
    source = _source_frames(
        path_names,
        images,
        poses,
        follow=playback.view == "follow",
        chase_distance=CHASE_DISTANCE * world_unit,
        timestamps=frame_timestamps,
    )
    segment_offset = 0.0
    segment_duration = float(source[-1][0])
    source = _duration_fraction(source, playback.duration_fraction)
    if playback.duration_fraction != 1.0:
        segment_duration = float(source[-1][0])
    scene_points, batches = _sparse_point_geometry(
        playback.sparse_point_mode,
        point_ids,
        points,
        source,
        observations,
    )
    scene = Scene(
        scene_points,
        np.asarray([poses[name].center for name in path_names]),
        gt_centers,
        reference,
        up,
        scale,
        batches,
        world_unit,
    )
    frames = _iter_cumulative_frames(
        source,
        path_names,
        poses,
        support,
        images,
        observations,
        batches,
        colors,
        depth_lift_reader=depth_lift_reader,
        model_images=model_images,
        alignment=alignment,
        depth_lift_stride=playback.depth_lift_stride,
        depth_lift_max=playback.depth_lift_max,
        depth_lift_keep=playback.depth_lift_keep,
        depth_lift_point_radius=playback.depth_lift_point_radius,
    )
    return Sequence(
        scene,
        frames,
        playback.view,
        resolution,
        colors,
        duration=segment_duration,
        time_offset=segment_offset,
    )


def flythrough_keyframe_count(playback: Playback) -> int:
    """Return the registered reconstruction keyframe count without loading depth maps."""
    run_root = _resolve_playback_run_dir(playback.source)
    model_path = _model_path(run_root, playback.reconstruction)
    import pycolmap

    model = pycolmap.Reconstruction(model_path)
    return sum(1 for image in model.images.values() if image.has_pose)


def flythrough_input_paths(playback: Playback) -> tuple[Path, ...]:
    """Return every file whose state can change a batched flythrough."""
    run, config, model_path, _model, image_names, rgb_dir, gt_path = _resolve_flythrough_inputs(playback)
    paths = [config, *_files_under(model_path), *_image_paths(rgb_dir, image_names).values()]

    from vidmap.depth_artifacts import DEPTH_SCALES_NAME, REFERENCE_NAME
    from vidmap.reconstruction import LOCAL_INPUT_MANIFEST_NAME

    paths.extend(
        (
            run / "mapper_inputs" / LOCAL_INPUT_MANIFEST_NAME,
            rgb_dir / "video_frames.json",
        )
    )
    if playback.align_to_ground_truth and gt_path is not None:
        paths.extend(_files_under(gt_path))
    if playback.depth_lift:
        full_depth, sampled_depth = resolve_depth_paths(run, playback.depth_maps)
        paths.extend((run / REFERENCE_NAME, run / DEPTH_SCALES_NAME, full_depth))
        if sampled_depth is not None:
            paths.append(sampled_depth)
    return tuple(dict.fromkeys(Path(path).expanduser().resolve() for path in paths))


def _resolve_flythrough_inputs(
    playback: Playback,
) -> tuple[Path, Path, Path, Any, tuple[str, ...], Path, Path | None]:
    run = _resolve_playback_run_dir(playback.source)
    from vidmap.configuration.dump import mapping_config_path

    config_path = mapping_config_path(run)
    if not config_path.is_file():
        raise FileNotFoundError(f"Flythrough needs the run config to infer RGB and GT data: {config_path}")
    config = yaml.safe_load(config_path.read_text()) or {}
    model_path = _model_path(run, playback.reconstruction)
    import pycolmap

    model = pycolmap.Reconstruction(model_path)
    image_names = tuple(str(image.name) for image in model.images.values())
    if playback.image_dir is None:
        from vidmap.reconstruction import local_run_image_dir

        local_images = local_run_image_dir(run)
        if local_images is None:
            rgb_dir, gt_path = _resolve_dataset_paths(config, image_names, run=run)
        else:
            rgb_dir, gt_path = _validate_explicit_image_dir(local_images, image_names), None
    else:
        rgb_dir, gt_path = _validate_explicit_image_dir(playback.image_dir, image_names), None
    if playback.ground_truth is not None:
        gt_path = playback.ground_truth
    elif playback.align_to_ground_truth and gt_path is None:
        gt_path = resolve_ground_truth_model_path(run)
    return run, config_path, model_path, model, image_names, rgb_dir, gt_path


def _files_under(directory: Path) -> tuple[Path, ...]:
    return tuple(sorted((path for path in Path(directory).rglob("*") if path.is_file()), key=str))


def _duration_fraction(
    source: tuple[tuple[float, str, Pose, str, Pose | None], ...],
    fraction: float,
) -> tuple[tuple[float, str, Pose, str, Pose | None], ...]:
    """Keep an initial flythrough segment without loading later frame payloads."""
    if not source or fraction >= 1.0:
        return source
    cutoff = float(source[-1][0]) * fraction
    elapsed = np.asarray([frame[0] for frame in source], dtype=np.float64)
    count = max(1, int(np.searchsorted(elapsed, cutoff, side="right")))
    return source[:count]


def _sparse_point_geometry(
    mode: str,
    point_ids: np.ndarray,
    points: np.ndarray,
    source: tuple[tuple[float, str, Pose, str, Pose | None], ...],
    observations: dict[str, tuple[np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, tuple[PointBatch, ...]]:
    """Choose one immutable sparse entity or progressive reveal batches."""
    if mode == "static":
        return np.asarray(points, dtype=np.float64).reshape((-1, 3)), ()
    if mode == "progressive":
        return np.empty((0, 3)), _point_batches(point_ids, points, source, observations)
    raise ValueError(f"unknown sparse point mode: {mode!r}")


def _resolve_playback_run_dir(source: Path) -> Path:
    path = Path(source).expanduser()
    if path.name == "playback_trace":
        return path.parent
    if path.is_dir():
        return path
    raise FileNotFoundError(f"Flythrough run directory is missing: {source}")


def _model_path(run: Path, selection: str | Path | None) -> Path:
    if selection is None:
        raise ValueError("Flythrough requires an explicit saved-model selection")
    names = {"final": "rec", "ba": "rec-ba", "gp": "rec-gp"}
    path = run / names[selection] if isinstance(selection, str) and selection in names else Path(selection)
    path = path.expanduser()
    if not path.is_dir():
        raise FileNotFoundError(f"Selected flythrough model is missing: {path}")
    return path


def _resolve_dataset_paths(
    config: dict[str, Any], names: tuple[str, ...], *, run: Path | None = None
) -> tuple[Path, Path | None]:
    scene = _require_single_config_value(config.get("scene"), "scene")
    parents = {Path(name).parent for name in names}
    root, configured = select_dataset_root(
        config,
        run,
        lambda candidate: _contains_sequences(candidate, scene, parents),
        purpose="Flythrough",
    )
    if root is not None:
        dataset = root / scene
        gt = dataset / "rec"
        return dataset / "images", gt if gt.is_dir() else None
    sequences = ", ".join(sorted(str(parent) for parent in parents))
    requested = ", ".join(configured) if configured else "<not configured>"
    raise FileNotFoundError(
        f"Could not resolve RGB data for configured dataset {requested}, "
        f"scene {scene!r}, and model sequences: {sequences}"
    )


def _validate_explicit_image_dir(image_dir: Path, names: tuple[str, ...]) -> Path:
    root = Path(image_dir).expanduser()
    if not root.is_dir():
        raise FileNotFoundError(f"Explicit flythrough image directory is missing: {root}")
    for name in names:
        image = root / name
        if not image.is_file():
            raise FileNotFoundError(f"Explicit flythrough image directory is missing model image: {image}")
    return root


def _contains_sequences(root: Path, scene: str, parents: set[Path]) -> bool:
    images = root / scene / "images"
    return images.is_dir() and all((images / parent).is_dir() for parent in parents)


def _require_single_config_value(value: Any, label: str) -> str:
    if isinstance(value, (list, tuple)) and value:
        return str(value[0])
    if isinstance(value, str) and value:
        return value
    raise ValueError(f"Run config must contain {label} to infer flythrough inputs")


def _load_sparse_model_scene(
    model: Any,
    rgb_dir: Path,
    alignment: Any | None,
    neutral: tuple[int, int, int],
    *,
    covariance_percentile: float | None = None,
) -> tuple[
    dict[str, Pose],
    np.ndarray,
    np.ndarray,
    dict[str, tuple[np.ndarray, np.ndarray]],
]:
    poses: dict[str, Pose] = {}
    observations: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    point_ids = np.asarray(sorted(int(value) for value in model.point3D_ids()), dtype=np.int64)
    if covariance_percentile is not None:
        point_ids = lowest_covariance_point_ids(model, point_ids, covariance_percentile)
    points = np.asarray(
        [model.point3D(int(point_id)).xyz for point_id in point_ids],
        dtype=np.float64,
    ).reshape((-1, 3))
    if alignment is not None:
        points = np.asarray(alignment * points)

    for image in model.images.values():
        if not image.has_pose:
            continue
        cam_from_world = image.cam_from_world()
        center = np.asarray(cam_from_world.inverse().translation, dtype=np.float64).reshape(3)
        world_from_camera = np.asarray(cam_from_world.rotation.matrix(), dtype=np.float64).reshape(3, 3).T
        if alignment is not None:
            rotation = np.asarray(alignment.rotation.matrix(), dtype=np.float64)
            center = np.asarray(alignment * center.reshape(1, 3)).reshape(3)
            world_from_camera = rotation @ world_from_camera
        name = str(image.name)
        poses[name] = Pose(center, world_from_camera, _camera_rays(model.cameras[image.camera_id]))
        observed_ids, colors = _observed_points(image, rgb_dir, neutral)
        if len(observed_ids):
            observations[name] = observed_ids, colors
    return poses, point_ids, points, observations


def _observed_points(
    image: Any,
    rgb_dir: Path,
    neutral: tuple[int, int, int],
) -> tuple[np.ndarray, np.ndarray]:
    point_ids: list[int] = []
    pixels: list[np.ndarray] = []
    for point2d in image.points2D:
        if not point2d.has_point3D():
            continue
        point_ids.append(int(point2d.point3D_id))
        pixels.append(np.asarray(point2d.xy, dtype=np.float64))
    if not point_ids:
        return (
            np.empty(0, dtype=np.int64),
            np.empty((0, 3), dtype=np.uint8),
        )
    image_path = rgb_dir / str(image.name)
    colors = np.full((len(point_ids), 3), neutral, dtype=np.uint8)
    if image_path.is_file():
        rgb = read_image(image_path)
        height, width = rgb.shape[:2]
        xy = np.rint(np.asarray(pixels)).astype(np.int64)
        xy[:, 0] = np.clip(xy[:, 0], 0, width - 1)
        xy[:, 1] = np.clip(xy[:, 1], 0, height - 1)
        colors = rgb[xy[:, 1], xy[:, 0]].astype(np.uint8)
    return (
        np.asarray(point_ids, dtype=np.int64),
        colors,
    )


def _support_by_current(
    observations: dict[str, tuple[np.ndarray, np.ndarray]],
    sequence: tuple[str, ...],
) -> dict[str, tuple[tuple[str, float], ...]]:
    observed = {name: np.unique(value[0]) for name, value in observations.items()}
    result: dict[str, tuple[tuple[str, float], ...]] = {}
    for current_index, current_name in enumerate(sequence):
        current = observed.get(current_name, np.empty(0, dtype=np.int64))
        if current_index == 0 or not len(current):
            continue
        history = sequence[max(0, current_index - KEYFRAME_HISTORY_LENGTH) : current_index]
        counts = [
            len(
                np.intersect1d(
                    current,
                    observed.get(name, np.empty(0, dtype=np.int64)),
                    assume_unique=True,
                )
            )
            for name in history
        ]
        maximum = max(counts, default=0)
        if not maximum:
            continue
        result[current_name] = tuple(
            (name, float(count) / float(maximum)) for name, count in zip(history, counts, strict=True)
        )
    return result


def _image_paths(rgb_dir: Path, model_names: tuple[str, ...]) -> dict[str, Path]:
    suffixes = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    paths = []
    for parent in sorted({Path(name).parent for name in model_names}, key=str):
        directory = rgb_dir / parent
        if directory.is_dir():
            paths.extend(path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in suffixes)
    return {str(path.relative_to(rgb_dir)): path for path in natsorted(paths, key=lambda item: item.name)}


def _elapsed_timestamps(
    names: tuple[str, ...], *, timestamps: dict[str, float] | None = None
) -> tuple[tuple[str, ...], np.ndarray]:
    if timestamps is not None:
        missing = [name for name in names if name not in timestamps]
        if missing:
            raise ValueError(f"Decoded-video timestamps are missing flythrough image: {missing[0]}")
        parsed_values = np.asarray([timestamps[name] for name in names], dtype=np.float64)
        order = np.argsort(parsed_values, kind="stable")
        values = parsed_values[order]
        ordered = tuple(names[index] for index in order)
        if np.any(np.diff(values) <= 0.0):
            raise ValueError("Flythrough image timestamps must be unique and strictly increasing")
        return ordered, values - values[0]
    parsed = []
    decimal = False
    for name in names:
        match = _TIMESTAMP.match(Path(name).stem)
        if match is None:
            raise ValueError(
                f"Flythrough image filename must start with a numeric timestamp or use a numeric stem: {name!r}"
            )
        token = match.group(1)
        decimal = decimal or "." in token
        parsed.append((name, token))
    if not parsed:
        return (), np.empty(0, dtype=np.float64)
    if decimal and any("." not in token for _name, token in parsed):
        raise ValueError("Flythrough image timestamps mix integer and decimal conventions")

    values = np.asarray([float(token) for _name, token in parsed], dtype=np.float64)
    order = np.argsort(values, kind="stable")
    values = values[order]
    ordered = tuple(parsed[index][0] for index in order)
    deltas = np.diff(values)
    if np.any(deltas <= 0.0):
        raise ValueError("Flythrough image timestamps must be unique and strictly increasing")
    if decimal:
        scale = 1.0
    else:
        median = float(np.median(deltas)) if len(deltas) else 0.0
        candidates = [scale for scale in (1.0, 1e6, 1e9) if 1.0 / 240.0 <= median / scale <= 2.0]
        if len(candidates) != 1:
            raise ValueError(
                "Could not infer whether integer flythrough timestamps use seconds, microseconds, or nanoseconds"
            )
        scale = candidates[0]
    return ordered, (values - values[0]) / scale


def _iter_cumulative_frames(
    source: tuple[tuple[float, str, Pose, str, Pose | None], ...],
    path_names: tuple[str, ...],
    poses: dict[str, Pose],
    support: dict[str, tuple[tuple[str, float], ...]],
    images: dict[str, Path],
    observations: dict[str, tuple[np.ndarray, np.ndarray]],
    batches: tuple[PointBatch, ...],
    palette: Palette,
    *,
    depth_lift_reader: DepthLiftReader | None = None,
    model_images: dict[str, Any] | None = None,
    alignment: Any | None = None,
    depth_lift_stride: int = 1,
    depth_lift_max: float = 20.0,
    depth_lift_keep: bool = False,
    depth_lift_point_radius: float = DEPTH_LIFT_RADIUS_UI_POINTS,
) -> Iterator[Frame]:
    order = {name: index for index, name in enumerate(path_names)}
    colored = tuple(batch for batch in batches if batch.colors is not None)
    if colored:
        lookup_ids = np.concatenate([batch.point_ids for batch in colored])
        lookup_points = np.concatenate([batch.points for batch in colored])
        lookup_colors = np.concatenate([batch.colors for batch in colored])
        lookup_order = np.argsort(lookup_ids, kind="stable")
        lookup_ids = lookup_ids[lookup_order]
        lookup_points = lookup_points[lookup_order]
        lookup_colors = lookup_colors[lookup_order]
    else:
        lookup_ids = np.empty(0, dtype=np.int64)
        lookup_points = np.empty((0, 3))
        lookup_colors = np.empty((0, 3), dtype=np.uint8)
    empty = np.empty((0, 3))
    for elapsed, image_name, pose, scene_name, render_pose in source:
        current_index = order[scene_name]
        density = dict(support.get(scene_name, ()))
        history_start = max(0, current_index - KEYFRAME_HISTORY_LENGTH)
        historical = tuple(
            Camera(
                f"{index:06d}",
                poses[name],
                _density_color(density.get(name, 0.0), palette),
            )
            for index, name in enumerate(
                path_names[history_start:current_index],
                start=history_start,
            )
        )
        observed_ids = observations.get(scene_name, (np.empty(0, dtype=np.int64), empty))[0]
        visible = np.unique(observed_ids)
        positions = np.searchsorted(lookup_ids, visible)
        valid = positions < len(lookup_ids)
        valid[valid] &= lookup_ids[positions[valid]] == visible[valid]
        positions = positions[valid]
        highlighted_points = lookup_points[positions]
        highlighted_colors = lookup_colors[positions]
        depth_lift_points = None
        depth_lift_colors = None
        if depth_lift_reader is not None:
            assert model_images is not None
            depth_lift_points, depth_lift_colors = depth_lift_reader.lift(
                model_images[scene_name],
                alignment=alignment,
                stride=depth_lift_stride,
                max_depth=depth_lift_max,
            )
            highlighted_points = None
            highlighted_colors = None
        yield Frame(
            "tracking",
            (),
            empty,
            empty,
            None,
            None,
            historical,
            Camera("current", pose, palette.estimated),
            image=images[image_name],
            render_pose=render_pose,
            highlighted_points=highlighted_points,
            highlighted_colors=highlighted_colors,
            depth_lift_points=depth_lift_points,
            depth_lift_colors=depth_lift_colors,
            depth_lift_keep=depth_lift_keep,
            depth_lift_point_radius=depth_lift_point_radius,
            elapsed=elapsed,
        )


def _point_batches(
    point_ids: np.ndarray,
    points: np.ndarray,
    source: tuple[tuple[float, str, Pose, str, Pose | None], ...],
    observations: dict[str, tuple[np.ndarray, np.ndarray]],
) -> tuple[PointBatch, ...]:
    point_ids = np.asarray(point_ids, dtype=np.int64).reshape(-1)
    points = np.asarray(points, dtype=np.float64).reshape((-1, 3))
    if len(point_ids) != len(points):
        raise ValueError("Flythrough point IDs and positions must have equal length")
    order = np.argsort(point_ids, kind="stable")
    point_ids = point_ids[order]
    points = points[order]
    if len(point_ids) > 1 and np.any(np.diff(point_ids) == 0):
        raise ValueError("Flythrough model point IDs must be unique")
    unseen = np.ones(len(point_ids), dtype=bool)
    batches: list[PointBatch] = []
    for frame_index, (
        _elapsed,
        _image_name,
        _pose,
        scene_name,
        _render_pose,
    ) in enumerate(source):
        observed_ids, colors = observations.get(
            scene_name,
            (
                np.empty(0, dtype=np.int64),
                np.empty((0, 3), dtype=np.uint8),
            ),
        )
        positions = np.searchsorted(point_ids, observed_ids)
        valid = positions < len(point_ids)
        valid[valid] &= point_ids[positions[valid]] == observed_ids[valid]
        observation_indices = np.flatnonzero(valid)
        positions = positions[valid]
        if len(positions):
            _values, first = np.unique(positions, return_index=True)
            first = np.sort(first)
            positions = positions[first]
            observation_indices = observation_indices[first]
            fresh = unseen[positions]
            positions = positions[fresh]
            observation_indices = observation_indices[fresh]
        if not len(positions):
            continue
        unseen[positions] = False
        batches.append(
            PointBatch(
                f"{frame_index:06d}",
                point_ids[positions],
                points[positions],
                np.asarray(colors[observation_indices], dtype=np.uint8),
                frame_index,
            )
        )
    if np.any(unseen):
        batches.append(
            PointBatch(
                "unobserved",
                point_ids[unseen],
                points[unseen],
                None,
                None,
            )
        )
    return tuple(batches)


def _source_frames(
    path_names: tuple[str, ...],
    images: dict[str, Path],
    poses: dict[str, Pose],
    *,
    follow: bool,
    chase_distance: float = CHASE_DISTANCE,
    timestamps: dict[str, float] | None = None,
) -> tuple[tuple[float, str, Pose, str, Pose | None], ...]:
    source_names, source_times = _elapsed_timestamps(tuple(images), timestamps=timestamps)
    time_by_name = dict(zip(source_names, source_times))
    keyframes = sorted((time_by_name[name], name) for name in path_names if name in time_by_name)
    if len(keyframes) < 2:
        return tuple((float(time_by_name[name]), name, poses[name], name, None) for name in path_names)

    start, end = keyframes[0][0], keyframes[-1][0]
    selected = (source_times >= start) & (source_times <= end)
    source_names = tuple(name for name, keep in zip(source_names, selected) if keep)
    query = source_times[selected] - start
    keyframes = [(timestamp - start, name) for timestamp, name in keyframes]
    subject = _interpolate_poses(keyframes, poses, query)
    chase = _chase_poses(keyframes, poses, query, chase_distance) if follow else (None,) * len(query)
    positions = [position for position, _name in keyframes]
    frames = []
    for index, timestamp in enumerate(query):
        scene_index = max(0, bisect.bisect_right(positions, float(timestamp)) - 1)
        scene_name = keyframes[scene_index][1]
        pose = subject[index]
        frames.append(
            (
                float(timestamp),
                source_names[index],
                Pose(pose.center, pose.world_from_camera, poses[scene_name].rays),
                scene_name,
                chase[index],
            )
        )
    return tuple(frames)


def _interpolate_poses(
    keyframes: list[tuple[float, str]],
    poses: dict[str, Pose],
    query: np.ndarray,
) -> tuple[Pose, ...]:
    positions = np.asarray([position for position, _name in keyframes], dtype=np.float64)
    query = np.clip(np.asarray(query, dtype=np.float64), positions[0], positions[-1])
    centers = np.asarray([poses[name].center for _position, name in keyframes], dtype=np.float64)
    rotations = np.asarray(
        [poses[name].world_from_camera for _position, name in keyframes],
        dtype=np.float64,
    )
    if len(positions) <= 2:
        interpolated_centers = np.column_stack(
            [np.interp(query, positions, centers[:, dimension]) for dimension in range(3)]
        )
    else:
        from scipy.interpolate import CubicSpline

        interpolated_centers = np.column_stack(
            [CubicSpline(positions, centers[:, dimension], bc_type="natural")(query) for dimension in range(3)]
        )
    values = Rotation.from_matrix(rotations)
    interpolator = RotationSpline(positions, values) if len(positions) >= 3 else Slerp(positions, values)
    interpolated_rotations = interpolator(query).as_matrix()
    return tuple(Pose(center, rotation) for center, rotation in zip(interpolated_centers, interpolated_rotations))


def _chase_poses(
    keyframes: list[tuple[float, str]],
    poses: dict[str, Pose],
    output_query: np.ndarray,
    distance: float = CHASE_DISTANCE,
) -> tuple[Pose, ...]:
    source = _interpolate_poses(keyframes, poses, output_query)
    source_centers = np.asarray([pose.center for pose in source])
    source_world_from_camera = np.asarray([pose.world_from_camera for pose in source])
    up = normalize_vector(np.median(-source_world_from_camera[:, :, 1], axis=0))
    directions = source_world_from_camera[:, :, 2]
    directions -= np.sum(directions * up[None], axis=1, keepdims=True) * up[None]
    forward = np.asarray([normalize_vector(direction) for direction in directions])
    height = distance / 3.0
    lookahead = distance * 8.0 / 3.0
    eyes = _smooth_centers(source_centers - distance * forward + height * up, output_query)
    targets = _smooth_centers(source_centers + lookahead * source_world_from_camera[:, :, 2], output_query)
    result = []
    previous_forward = None
    for eye, target in zip(eyes, targets):
        camera_forward = normalize_vector(target - eye)
        if previous_forward is not None and float(np.dot(camera_forward, previous_forward)) < 0.0:
            camera_forward = -camera_forward
        world_from_camera = camera_rotation(eye, eye + camera_forward, up)
        result.append(Pose(eye, world_from_camera))
        previous_forward = camera_forward
    return tuple(result)


def _estimate_path_unit(centers: np.ndarray) -> float:
    centers = np.asarray(centers, dtype=np.float64).reshape((-1, 3))
    distances = np.linalg.norm(np.diff(centers, axis=0), axis=1)
    distances = distances[np.isfinite(distances) & (distances > 1e-9)]
    if len(distances):
        return float(np.median(distances))
    span = float(np.max(np.ptp(centers, axis=0))) if len(centers) else 0.0
    return span / max(len(centers) - 1, 1) if span > 1e-9 else 1.0


def _smooth_centers(centers: np.ndarray, timestamps: np.ndarray) -> np.ndarray:
    if len(centers) <= 2 or CHASE_SMOOTHING_SECONDS <= 0.0:
        return centers
    forward = np.asarray(centers, dtype=np.float64).copy()
    for index in range(1, len(forward)):
        alpha = 1.0 - np.exp(-(timestamps[index] - timestamps[index - 1]) / CHASE_SMOOTHING_SECONDS)
        forward[index] = forward[index - 1] + alpha * (centers[index] - forward[index - 1])
    backward = np.asarray(centers, dtype=np.float64).copy()
    for index in range(len(backward) - 2, -1, -1):
        alpha = 1.0 - np.exp(-(timestamps[index + 1] - timestamps[index]) / CHASE_SMOOTHING_SECONDS)
        backward[index] = backward[index + 1] + alpha * (centers[index] - backward[index + 1])
    return 0.5 * (forward + backward)


def _camera_rays(camera: Any) -> np.ndarray:
    corners = np.asarray(
        [
            [0.0, 0.0],
            [camera.width, 0.0],
            [camera.width, camera.height],
            [0.0, camera.height],
        ]
    )
    normalized = np.asarray(camera.cam_from_img(corners), dtype=np.float64).reshape((4, 2))
    return np.column_stack((normalized, np.ones(4)))


def _density_color(density: float, palette: Palette) -> tuple[int, int, int]:
    value = float(np.clip(density, 0.0, 1.0))
    low = np.asarray(palette.support_low)
    high = np.asarray(palette.support_high)
    return tuple(int(round(channel)) for channel in low + (high - low) * value)
