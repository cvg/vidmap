from __future__ import annotations

import itertools
from pathlib import Path
from typing import Any

import numpy as np

from .scene import (
    CAMERA_RADIUS_UI_POINTS,
    CANONICAL_UI_HEIGHT,
    ESTIMATED_PATH_RADIUS_UI_POINTS,
    GT_PATH_RADIUS_UI_POINTS,
    LC_RADIUS_UI_POINTS,
    PLAYBACK_VIEW_ID,
    POINT_COLOR_RGBA,
    POINT_RADIUS_UI_POINTS,
    RENDER_CAMERA_PATH,
    STROKE_UI_POINTS,
    TRACKING_IMAGE_TO_SPATIAL_RATIO,
    Camera,
    Frame,
    Pose,
    Sequence,
    fixed_view_axes,
    fixed_view_camera_pose,
)

APP_NAME = "vidmap-trace-playback"
PREFIX = "world/playback"
FIXED_FRUSTUM_PIXELS = 16.0
FOLLOW_FRUSTUM_DEPTH = 0.56
GT_PATH_DRAW_ORDER = 10_000.0
ESTIMATED_PATH_DRAW_ORDER = 10_001.0


def write(sequence: Sequence, output: Path) -> None:
    import rerun as rr

    output.parent.mkdir(parents=True, exist_ok=True)
    stream = rr.RecordingStream(APP_NAME)
    try:
        rr.save(output, recording=stream)
        frames = iter(sequence.frames)
        first = next(frames, None)
        if first is None:
            raise FileNotFoundError("No playback frames were emitted")
        _blueprint(rr, stream, sequence, first, has_images=first.image is not None)
        scene = sequence.scene
        if len(scene.points):
            rr.log(
                f"{PREFIX}/points3D",
                rr.Points3D(
                    _layer(scene.points, sequence, 1),
                    colors=POINT_COLOR_RGBA,
                    radii=rr.Radius.ui_points(_ui_radius(sequence, POINT_RADIUS_UI_POINTS)),
                ),
                static=True,
                recording=stream,
            )
        _path(
            rr,
            stream,
            sequence,
            "camera_path",
            scene.estimated_path,
            sequence.palette.estimated,
            static=True,
        )
        _path(
            rr,
            stream,
            sequence,
            "gt_camera_path",
            scene.gt_path,
            sequence.palette.ground_truth,
            static=True,
        )
        historical_colors: dict[str, tuple[int, int, int]] = {}
        for index, frame in enumerate(itertools.chain((first,), frames)):
            rr.set_time("frame", sequence=index, recording=stream)
            rr.set_time("elapsed", duration=frame.elapsed, recording=stream)
            if index == 0:
                rr.log(
                    f"{PREFIX}/output_width",
                    rr.Scalars(sequence.resolution[0]),
                    recording=stream,
                )
                rr.log(
                    f"{PREFIX}/output_height",
                    rr.Scalars(sequence.resolution[1]),
                    recording=stream,
                )
                rr.log(
                    f"{PREFIX}/time_offset_seconds",
                    rr.Scalars(sequence.time_offset),
                    recording=stream,
                )
            rr.log(f"{PREFIX}/elapsed_seconds", rr.Scalars(frame.elapsed), recording=stream)
            rr.log(
                f"{PREFIX}/duration_seconds",
                rr.Scalars(sequence.duration),
                recording=stream,
            )
            if index == 0:
                for batch in scene.point_batches:
                    _point_batch(rr, stream, sequence, batch, POINT_COLOR_RGBA)
            for batch in scene.point_batches:
                if batch.reveal_frame == index:
                    _point_batch(rr, stream, sequence, batch, batch.colors)
            _record_solver_frame(rr, stream, sequence, frame, historical_colors, frame_index=index)
    finally:
        stream.flush()
        stream.disconnect()


def _record_solver_frame(
    rr: Any,
    stream: Any,
    sequence: Sequence,
    frame: Frame,
    historical_colors: dict[str, tuple[int, int, int]],
    frame_index: int | None = None,
) -> None:
    points_path = f"{PREFIX}/points3D"
    if len(frame.points):
        rr.log(
            points_path,
            rr.Points3D(
                _layer(frame.points, sequence, 1),
                colors=POINT_COLOR_RGBA,
                radii=rr.Radius.ui_points(_ui_radius(sequence, POINT_RADIUS_UI_POINTS)),
            ),
            recording=stream,
        )
    elif not len(sequence.scene.points) and not sequence.scene.point_batches:
        rr.log(points_path, rr.Clear(recursive=True), recording=stream)
    if len(frame.centers):
        _path(
            rr,
            stream,
            sequence,
            "camera_path",
            _ordered_centers(frame),
            sequence.palette.estimated,
        )
        _lc(rr, stream, sequence, frame, frame.centers)
    elif not len(sequence.scene.estimated_path):
        _path(rr, stream, sequence, "camera_path", frame.centers, sequence.palette.estimated)
        _lc(rr, stream, sequence, frame, frame.centers)
    _record_camera_poses(rr, stream, sequence, frame, historical_colors)
    observed_path = f"{PREFIX}/current_observed_points"
    rr.log(
        observed_path,
        (
            rr.Points3D(
                _layer(frame.highlighted_points, sequence, 1),
                colors=frame.highlighted_colors,
                radii=rr.Radius.ui_points(_ui_radius(sequence, 2.0 * POINT_RADIUS_UI_POINTS)),
            )
            if frame.highlighted_points is not None and len(frame.highlighted_points)
            else rr.Clear(recursive=True)
        ),
        recording=stream,
    )
    depth_points = (
        rr.Points3D(
            _layer(frame.depth_lift_points, sequence, 1),
            colors=frame.depth_lift_colors,
            radii=rr.Radius.ui_points(_ui_radius(sequence, frame.depth_lift_point_radius)),
        )
        if frame.depth_lift_points is not None and len(frame.depth_lift_points)
        else None
    )
    if frame.depth_lift_keep:
        if depth_points is not None:
            if frame_index is None:
                raise ValueError("Persistent depth lift requires a frame index")
            rr.log(
                f"{PREFIX}/lifted_depth_history/{frame_index:06d}",
                depth_points,
                recording=stream,
            )
    else:
        rr.log(
            f"{PREFIX}/current_lifted_depth",
            depth_points if depth_points is not None else rr.Clear(recursive=True),
            recording=stream,
        )
    rr.log(
        "tracking/current_image",
        (rr.EncodedImage(path=frame.image) if frame.image is not None else rr.Clear(recursive=True)),
        recording=stream,
    )
    _render_camera(rr, stream, sequence, frame.render_pose)


def _point_batch(rr: Any, stream: Any, sequence: Sequence, batch: Any, colors: Any) -> None:
    rr.log(
        f"{PREFIX}/point_batches/{batch.key}",
        rr.Points3D(
            _layer(batch.points, sequence, 1),
            colors=colors,
            radii=rr.Radius.ui_points(_ui_radius(sequence, POINT_RADIUS_UI_POINTS)),
        ),
        recording=stream,
    )


def _path(
    rr: Any,
    stream: Any,
    sequence: Sequence,
    name: str,
    points: np.ndarray,
    color: tuple[int, int, int],
    *,
    static: bool = False,
) -> None:
    if len(points) >= 2:
        estimated = name == "camera_path"
        radius = ESTIMATED_PATH_RADIUS_UI_POINTS if estimated else GT_PATH_RADIUS_UI_POINTS
        draw_order = ESTIMATED_PATH_DRAW_ORDER if estimated else GT_PATH_DRAW_ORDER
        rr.log(
            f"{PREFIX}/{name}",
            rr.LineStrips3D(
                [points],
                colors=color,
                radii=rr.Radius.ui_points(_ui_radius(sequence, radius)),
            ),
            rr.AnyValues(**{"rerun.components.DrawOrder": [draw_order]}),
            static=static,
            recording=stream,
        )
    elif not static:
        rr.log(f"{PREFIX}/{name}", rr.Clear(recursive=True), recording=stream)


def _ordered_centers(frame: Frame) -> np.ndarray:
    centers = np.asarray(frame.centers, dtype=np.float64).reshape((-1, 3))
    if len(frame.image_ids) != len(centers):
        return centers
    return centers[np.argsort(np.asarray(frame.image_ids), kind="stable")]


def _lc(rr: Any, stream: Any, sequence: Sequence, frame: Frame, centers: np.ndarray) -> None:
    path = f"{PREFIX}/lc_edges/force"
    if frame.lc_pairs is None or frame.lc_colors is None:
        rr.log(path, rr.Clear(recursive=True), recording=stream)
        return
    by_id = dict(zip(frame.image_ids, _layer(centers, sequence, 0)))
    edges, colors = [], []
    for index, (left, right) in enumerate(frame.lc_pairs):
        if int(left) in by_id and int(right) in by_id:
            edges.append([by_id[int(left)], by_id[int(right)]])
            colors.append(frame.lc_colors[index])
    rr.log(
        path,
        (
            rr.LineStrips3D(
                edges,
                colors=colors,
                radii=rr.Radius.ui_points(_ui_radius(sequence, LC_RADIUS_UI_POINTS)),
            )
            if edges
            else rr.Clear(recursive=True)
        ),
        recording=stream,
    )


def _record_camera_poses(
    rr: Any,
    stream: Any,
    sequence: Sequence,
    frame: Frame,
    historical_colors: dict[str, tuple[int, int, int]],
) -> None:
    depth = (
        FOLLOW_FRUSTUM_DEPTH * sequence.scene.world_unit
        if sequence.view == "follow"
        else FIXED_FRUSTUM_PIXELS * sequence.scene.scale / sequence.resolution[1]
    )
    active_keys = {camera.key for camera in frame.historical_cameras}
    for key in tuple(historical_colors):
        if key not in active_keys:
            rr.log(
                f"{PREFIX}/keyframes/{key}",
                rr.Clear(recursive=True),
                recording=stream,
            )
            del historical_colors[key]
    for camera in frame.historical_cameras:
        if historical_colors.get(camera.key) != camera.color:
            _camera(rr, stream, sequence, f"{PREFIX}/keyframes/{camera.key}", camera, depth)
            historical_colors[camera.key] = camera.color
    if frame.current_camera is not None:
        _camera(rr, stream, sequence, f"{PREFIX}/keyframes/current", frame.current_camera, depth)


def _camera(
    rr: Any,
    stream: Any,
    sequence: Sequence,
    path: str,
    camera: Camera,
    depth: float,
) -> None:
    rr.log(
        f"{path}/frustum",
        rr.LineStrips3D(
            _frustum(camera.pose, depth),
            colors=camera.color,
            radii=rr.Radius.ui_points(_ui_radius(sequence, CAMERA_RADIUS_UI_POINTS)),
        ),
        recording=stream,
    )
    rr.log(
        f"{path}/center",
        rr.Points3D(
            [camera.pose.center],
            colors=camera.color,
            radii=rr.Radius.ui_points(_ui_radius(sequence, CAMERA_RADIUS_UI_POINTS)),
        ),
        recording=stream,
    )


def _ui_radius(sequence: Sequence, canonical_radius: float) -> float:
    """Scale screen-space radii from the canonical 960-pixel output height."""
    return float(canonical_radius) * float(sequence.resolution[1]) / CANONICAL_UI_HEIGHT


def _render_camera(rr: Any, stream: Any, sequence: Sequence, pose: Pose | None) -> None:
    pose = pose or fixed_view_camera_pose(sequence.scene, sequence.view)
    width, height = sequence.resolution
    intrinsics = np.asarray([[height, 0.0, width / 2.0], [0.0, height, height / 2.0], [0.0, 0.0, 1.0]])
    rr.log(
        RENDER_CAMERA_PATH,
        rr.Transform3D(translation=pose.center, mat3x3=pose.world_from_camera),
        recording=stream,
    )
    rr.log(
        RENDER_CAMERA_PATH,
        rr.Pinhole(
            image_from_camera=intrinsics,
            resolution=[width, height],
            camera_xyz=rr.ViewCoordinates.RDF,
            image_plane_distance=1e-6,
            color=[0, 0, 0, 0],
            line_width=0.0,
        ),
        recording=stream,
    )


def _layer(values: np.ndarray, sequence: Sequence, level: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64).reshape((-1, 3))
    if sequence.view == "follow" or not len(values):
        return values
    direction, _right, _screen_up = fixed_view_axes(sequence.scene.reference, sequence.scene.up, sequence.view)
    spacing = STROKE_UI_POINTS * sequence.scene.scale / sequence.resolution[1]
    return values + level * spacing * direction


def _frustum(pose: Pose, depth: float) -> list[np.ndarray]:
    rays = pose.rays
    if rays is None:
        rays = np.asarray([[-1.0, -1.0, 1.4], [1.0, -1.0, 1.4], [1.0, 1.0, 1.4], [-1.0, 1.0, 1.4]]) / 1.4
    rays = np.asarray(rays).reshape((4, 3))
    corners = pose.center + depth * ((rays / rays[:, 2, None]) @ pose.world_from_camera.T)
    return [np.asarray([pose.center, corner]) for corner in corners] + [
        np.asarray([corners[index], corners[(index + 1) % 4]]) for index in range(4)
    ]


def _follow_eye_controls_kwargs(first: Frame) -> dict[str, Any]:
    pose = first.render_pose
    if pose is None:
        raise ValueError("Follow playback requires a render camera pose")
    return {
        "position": pose.center,
        "look_target": pose.center + pose.world_from_camera[:, 2],
        "eye_up": -pose.world_from_camera[:, 1],
        "tracking_entity": RENDER_CAMERA_PATH,
    }


def _blueprint(rr: Any, stream: Any, sequence: Sequence, first: Frame, *, has_images: bool) -> None:
    import rerun.blueprint as rrb
    from rerun.blueprint.archetypes.line_grid3d import LineGrid3D

    if sequence.view == "follow":
        eye_controls = rrb.EyeControls3D(**_follow_eye_controls_kwargs(first))
    else:
        pose = fixed_view_camera_pose(sequence.scene, sequence.view)
        eye_controls = rrb.EyeControls3D(
            position=pose.center,
            look_target=pose.center + pose.world_from_camera[:, 2],
            eye_up=-pose.world_from_camera[:, 1],
        )
    spatial = rrb.Spatial3DView(
        name="Playback",
        origin=PREFIX,
        contents=[
            f"+ {PREFIX}/**",
            f"+ {RENDER_CAMERA_PATH}",
            f"- {RENDER_CAMERA_PATH}/**",
        ],
        eye_controls=eye_controls,
        background=rrb.Background(color=sequence.palette.background, kind=rrb.BackgroundKind.SolidColor),
        line_grid=LineGrid3D(visible=False),
    )
    spatial.id = PLAYBACK_VIEW_ID
    content = spatial
    if has_images:
        content = rrb.Horizontal(
            spatial,
            rrb.Spatial2DView(
                origin="tracking/current_image",
                contents="tracking/current_image",
                name="Tracking Frame",
            ),
            column_shares=[1.0, TRACKING_IMAGE_TO_SPATIAL_RATIO],
        )
    rr.send_blueprint(
        rrb.Blueprint(
            content,
            rrb.BlueprintPanel(state="collapsed"),
            rrb.SelectionPanel(state="collapsed"),
            rrb.TimePanel(state="collapsed"),
            auto_views=False,
        ),
        make_active=True,
        make_default=True,
        recording=stream,
    )
