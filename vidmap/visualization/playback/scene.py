from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Literal
from uuid import UUID

import numpy as np

Resolution = tuple[int, int]
View = Literal["topdown", "side", "isometric", "follow"]

STROKE_UI_POINTS = 1.0
CANONICAL_UI_HEIGHT = 960
LC_RADIUS_UI_POINTS = STROKE_UI_POINTS
ESTIMATED_PATH_RADIUS_UI_POINTS = 3.0 * STROKE_UI_POINTS
GT_PATH_RADIUS_UI_POINTS = 3.0 * STROKE_UI_POINTS
CAMERA_RADIUS_UI_POINTS = 3.0 * STROKE_UI_POINTS
POINT_RADIUS_UI_POINTS = 0.26 * STROKE_UI_POINTS
DEPTH_LIFT_RADIUS_UI_POINTS = 1.45 * STROKE_UI_POINTS
POINT_COLOR_RGBA = (255, 255, 255, 153)
FRAMING_MARGIN = 1.08
TRACKING_IMAGE_TO_SPATIAL_RATIO = 2.0 / 3.0
RENDER_CAMERA_PATH = "/__render/render_camera"
PLAYBACK_VIEW_ID = UUID("305fe839-7f79-47af-8f9f-c0d5e87ea1d2")


@dataclass(frozen=True)
class Palette:
    background: tuple[int, int, int, int]
    estimated: tuple[int, int, int]
    ground_truth: tuple[int, int, int]
    neutral: tuple[int, int, int]
    support_low: tuple[int, int, int]
    support_high: tuple[int, int, int]
    rejected: tuple[int, int, int]
    accepted: tuple[int, int, int]


def theme_palette(theme: str) -> Palette:
    if theme == "dark":
        return Palette(
            (0, 0, 0, 255),
            (0, 100, 255),
            (255, 220, 0),
            (190, 190, 190),
            (255, 255, 255),
            (255, 0, 0),
            (240, 70, 70),
            (0, 210, 130),
        )
    if theme == "light":
        return Palette(
            (255, 255, 255, 255),
            (0, 100, 255),
            (255, 220, 0),
            (0, 0, 0),
            (120, 120, 120),
            (255, 0, 0),
            (240, 70, 70),
            (0, 210, 130),
        )
    if theme == "neon":
        return Palette(
            (5, 8, 12, 255),
            (0, 229, 255),
            (255, 232, 0),
            (150, 163, 177),
            (248, 248, 255),
            (255, 0, 153),
            (255, 55, 95),
            (0, 255, 170),
        )
    raise ValueError(f"unknown playback theme: {theme!r}")


@dataclass(frozen=True)
class Pose:
    center: np.ndarray
    world_from_camera: np.ndarray
    rays: np.ndarray | None = None


@dataclass(frozen=True)
class Camera:
    key: str
    pose: Pose
    color: tuple[int, int, int]


@dataclass(frozen=True)
class PointBatch:
    key: str
    point_ids: np.ndarray
    points: np.ndarray
    colors: np.ndarray | None
    reveal_frame: int | None


@dataclass(frozen=True)
class Scene:
    points: np.ndarray
    estimated_path: np.ndarray
    gt_path: np.ndarray
    reference: np.ndarray
    up: np.ndarray
    scale: float
    point_batches: tuple[PointBatch, ...] = ()
    world_unit: float = 1.0


@dataclass(frozen=True)
class Frame:
    stage: str
    image_ids: tuple[int, ...]
    centers: np.ndarray
    points: np.ndarray
    lc_pairs: np.ndarray | None
    lc_colors: np.ndarray | None
    historical_cameras: tuple[Camera, ...] = ()
    current_camera: Camera | None = None
    image: Path | None = None
    render_pose: Pose | None = None
    highlighted_points: np.ndarray | None = None
    highlighted_colors: np.ndarray | None = None
    depth_lift_points: np.ndarray | None = None
    depth_lift_colors: np.ndarray | None = None
    depth_lift_keep: bool = False
    depth_lift_point_radius: float = DEPTH_LIFT_RADIUS_UI_POINTS
    elapsed: float = 0.0


@dataclass(frozen=True)
class Sequence:
    scene: Scene
    frames: Iterable[Frame]
    view: View
    resolution: Resolution
    palette: Palette = field(default_factory=lambda: theme_palette("dark"))
    duration: float = 0.0
    time_offset: float = 0.0


def compute_scene_framing(
    reference: np.ndarray, up: np.ndarray, resolution: Resolution, view: View
) -> tuple[np.ndarray, float]:
    reference = np.asarray(reference, dtype=np.float64).reshape((-1, 3))
    up = normalize_vector(up)
    if not len(reference) or view == "follow":
        return up, 0.0
    _direction, right, screen_up = fixed_view_axes(reference, up, view)
    width = float(np.ptp(reference @ right))
    height = float(np.ptp(reference @ screen_up))
    scale = max(height, width / (resolution[0] / resolution[1]), np.finfo(np.float64).eps) * FRAMING_MARGIN
    return up, scale


def estimate_path_up_direction(centers: np.ndarray) -> np.ndarray:
    values = np.asarray(centers, dtype=np.float64).reshape((-1, 3))
    if len(values) < 3:
        return np.asarray([0.0, 0.0, 1.0])
    _u, _s, axes = np.linalg.svd(values - values.mean(0), full_matrices=True)
    up = axes[-1]
    if up[int(np.argmax(np.abs(up)))] < 0:
        up = -up
    return normalize_vector(up)


def fixed_view_axes(centers: np.ndarray, up: np.ndarray, view: View) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    right, screen_up = screen_plane_axes(centers, up)
    if view == "topdown":
        return up, right, screen_up
    side = normalize_vector(np.cross(right, up))
    direction = side if view == "side" else normalize_vector(-right - side + up)
    rotation = camera_rotation(direction, np.zeros(3), up)
    return direction, rotation[:, 0], -rotation[:, 1]


def fixed_view_camera_pose(scene: Scene, view: View) -> Pose:
    direction, right, camera_up = fixed_view_axes(scene.reference, scene.up, view)
    horizontal = scene.reference - (scene.reference @ direction)[:, None] * direction
    target = (
        0.5 * (float(np.min(horizontal @ right)) + float(np.max(horizontal @ right))) * right
        + 0.5 * (float(np.min(horizontal @ camera_up)) + float(np.max(horizontal @ camera_up))) * camera_up
        + float(np.median(scene.reference @ direction)) * direction
    )
    geometry = [scene.reference, scene.points, scene.estimated_path, scene.gt_path]
    geometry.extend(batch.points for batch in scene.point_batches)
    visible = np.concatenate([value for value in geometry if len(value)])
    eye_depth = float(np.max(visible @ direction)) + scene.scale
    eye = target + (eye_depth - float(target @ direction)) * direction
    return Pose(eye, camera_rotation(eye, target, camera_up))


def screen_plane_axes(centers: np.ndarray, up: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    horizontal = centers - (centers @ up)[:, None] * up
    centered = horizontal - horizontal.mean(0) if len(horizontal) else horizontal
    if len(centered) >= 2 and float(np.linalg.norm(centered)) > 1e-12:
        _u, _s, axes = np.linalg.svd(centered, full_matrices=False)
        right = axes[0]
        if float(right @ (horizontal[-1] - horizontal[0])) < 0:
            right = -right
    else:
        right = np.asarray([1.0, 0.0, 0.0])
    right -= float(right @ up) * up
    if np.linalg.norm(right) < 1e-12:
        right = np.cross(np.asarray([1.0, 0.0, 0.0]), up)
    if np.linalg.norm(right) < 1e-12:
        right = np.cross(np.asarray([0.0, 1.0, 0.0]), up)
    right = normalize_vector(right)
    return right, normalize_vector(np.cross(up, right))


def camera_rotation(eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
    forward = normalize_vector(target - eye)
    down = normalize_vector(-up)
    right = normalize_vector(np.cross(down, forward))
    return np.column_stack((right, normalize_vector(np.cross(forward, right)), forward))


def normalize_vector(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=np.float64).reshape(3)
    norm = float(np.linalg.norm(value))
    return value / norm if norm > 1e-12 else np.asarray([1.0, 0.0, 0.0])
