from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from natsort import natsorted

from vidmap.datasets.base import image_has_public_pose

from .dataset_roots import select_dataset_root


@dataclass(frozen=True)
class Similarity:
    """A fixed Sim3 mapping one solver stage into the shared world frame."""

    scale: float
    rotation: np.ndarray
    translation: np.ndarray

    @classmethod
    def identity(cls) -> Similarity:
        return cls(1.0, np.eye(3, dtype=np.float64), np.zeros(3, dtype=np.float64))

    def transform(self, xyz: np.ndarray) -> np.ndarray:
        values = np.asarray(xyz, dtype=np.float64).reshape((-1, 3))
        return self.scale * (values @ self.rotation.T) + self.translation

    def followed_by(self, following: Similarity) -> Similarity:
        """Compose this source-to-middle transform with middle-to-target."""
        return Similarity(
            following.scale * self.scale,
            following.rotation @ self.rotation,
            following.scale * (following.rotation @ self.translation) + following.translation,
        )


def estimate_stage_similarities(
    stages: tuple[tuple[Any, ...], ...],
    final_to_world: Similarity,
) -> dict[str, Similarity]:
    """Compose fixed stage gauges backwards from the final solver stage."""
    if not stages or any(not stage for stage in stages):
        raise ValueError("Cannot align an empty solver stage")
    result = {str(stages[-1][-1].stage): final_to_world}
    for previous, following in zip(reversed(stages[:-1]), reversed(stages[1:])):
        previous_name = str(previous[-1].stage)
        following_name = str(following[0].stage)
        bridge = align_trace_frames(
            previous[-1],
            following[0],
            context=f"{previous_name}->{following_name}",
        )
        result[previous_name] = bridge.followed_by(result[following_name])
    return result


def align_trace_frames(source: Any, target: Any, *, context: str) -> Similarity:
    source_xyz, target_xyz = matching_camera_centers(source, target)
    return estimate_sim3_alignment(source_xyz, target_xyz, context=context)


def estimate_trace_to_model_alignment(frame: Any, model: Any) -> Similarity:
    """Estimate the terminal trace state's gauge relative to a saved model."""
    by_name = {str(image.name): image for image in model.images.values() if image.has_pose}
    by_id = {int(image.image_id): image for image in model.images.values() if image.has_pose}
    source, target = [], []
    for index, center in enumerate(frame.centers):
        name = frame.names[index] if index < len(frame.names) else ""
        image_id = int(frame.image_ids[index]) if index < len(frame.image_ids) else -1
        image = by_name.get(name) or by_id.get(image_id)
        if image is not None:
            source.append(center)
            target.append(_camera_center(image))
    return estimate_sim3_alignment(np.asarray(source), np.asarray(target), context="terminal trace->saved model")


def matching_camera_centers(source: Any, target: Any) -> tuple[np.ndarray, np.ndarray]:
    target_names = {name: index for index, name in enumerate(target.names) if str(name)}
    target_ids = {int(image_id): index for index, image_id in enumerate(target.image_ids)}
    source_xyz, target_xyz = [], []
    for index, center in enumerate(source.centers):
        name = source.names[index] if index < len(source.names) else ""
        image_id = int(source.image_ids[index]) if index < len(source.image_ids) else -1
        target_index = target_names.get(name) if name else None
        if target_index is None:
            target_index = target_ids.get(image_id)
        if target_index is not None:
            source_xyz.append(center)
            target_xyz.append(target.centers[target_index])
    return (
        np.asarray(source_xyz, dtype=np.float64).reshape((-1, 3)),
        np.asarray(target_xyz, dtype=np.float64).reshape((-1, 3)),
    )


def estimate_sim3_alignment(source: np.ndarray, target: np.ndarray, *, context: str) -> Similarity:
    source = np.asarray(source, dtype=np.float64).reshape((-1, 3))
    target = np.asarray(target, dtype=np.float64).reshape((-1, 3))
    if len(source) < 3:
        raise ValueError(f"Need at least three matching camera poses for {context}, got {len(source)}")
    source_mean = source.mean(axis=0)
    target_mean = target.mean(axis=0)
    source_zero = source - source_mean
    target_zero = target - target_mean
    variance = float(np.sum(source_zero * source_zero) / len(source))
    if not np.isfinite(variance) or variance <= np.finfo(np.float64).eps:
        raise ValueError(f"Camera poses are degenerate for {context}")
    covariance = target_zero.T @ source_zero / len(source)
    left, singular_values, right = np.linalg.svd(covariance)
    correction = np.eye(3)
    if np.linalg.det(left) * np.linalg.det(right) < 0:
        correction[-1, -1] = -1.0
    rotation = left @ correction @ right
    scale = float(np.sum(singular_values * np.diag(correction)) / variance)
    translation = target_mean - scale * (rotation @ source_mean)
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError(f"Could not estimate a positive Sim3 scale for {context}")
    return Similarity(scale, rotation, translation)


def similarity_from_pycolmap(transform: Any) -> Similarity:
    return Similarity(
        float(transform.scale),
        np.asarray(transform.rotation.matrix(), dtype=np.float64),
        np.asarray(transform.translation, dtype=np.float64).reshape(3),
    )


def resolve_ground_truth_model_path(run: Path) -> Path | None:
    """Resolve the scene's saved ground-truth model."""
    from vidmap.configuration.dump import mapping_config_path

    config_path = mapping_config_path(run)
    if not config_path.is_file():
        return None
    config = yaml.safe_load(config_path.read_text()) or {}
    scene = _single_config_value(config.get("scene"))
    if scene is None:
        return None
    root, _configured = select_dataset_root(
        config,
        run,
        lambda candidate: (candidate / scene / "rec").is_dir(),
        purpose="Solver playback GT",
    )
    return None if root is None else root / scene / "rec"


def estimate_model_to_ground_truth_alignment(
    model: Any,
    gt_path: Path | None,
    *,
    run: Path | None = None,
) -> tuple[Any | None, np.ndarray]:
    """Estimate one saved-model-to-GT transform and return the GT path."""
    if gt_path is None:
        return None, np.empty((0, 3), dtype=np.float64)

    import pycolmap

    gt = pycolmap.Reconstruction(gt_path)
    parents = {Path(image.name).parent.as_posix() for image in model.images.values()}
    gt_images = {
        str(image.name): image
        for image in gt.images.values()
        if image_has_public_pose(image) and Path(image.name).parent.as_posix() in parents
    }
    source, target = [], []
    for image in model.images.values():
        reference = gt_images.get(str(image.name))
        if image.has_pose and reference is not None:
            source.append(_camera_center(image))
            target.append(_camera_center(reference))
    if len(source) < 3:
        return None, np.empty((0, 3), dtype=np.float64)
    transform = pycolmap.estimate_sim3d(np.asarray(source), np.asarray(target))
    if transform is None:
        raise ValueError("Could not align the selected saved model to GT")
    selected = _testset_image_names(run, gt_path, gt)
    path_names = gt_images if selected is None else set(gt_images) & selected
    path = np.asarray(
        [_camera_center(gt_images[name]) for name in natsorted(path_names)],
        dtype=np.float64,
    )
    return transform, path.reshape((-1, 3))


def _testset_image_names(run: Path | None, gt_path: Path, gt: Any) -> set[str] | None:
    from vidmap.configuration.dump import mapping_config_path

    if run is None or not (config_path := mapping_config_path(run)).is_file():
        return None
    config = yaml.safe_load(config_path.read_text()) or {}
    mode = config.get("mode")
    scene = _single_config_value(config.get("scene"))
    testset = _single_config_value(config.get("testset_id"))
    if not isinstance(mode, str) or scene is None or testset is None:
        return None

    from vidmap.datasets.names import SUPPORTED_DATASETS
    from vidmap.paths import dataset_paths

    candidates = (
        dataset_paths(part).testsets / scene / f"{mode}.yaml" for part in gt_path.parts if part in SUPPORTED_DATASETS
    )
    testset_path = next(
        (candidate for candidate in candidates if candidate.is_file()),
        None,
    )
    if testset_path is None:
        return None
    payload = yaml.safe_load(testset_path.read_text()) or {}
    image_ids = payload.get(testset)
    if not isinstance(image_ids, list):
        return None
    return {str(gt.images[int(image_id)].name) for image_id in image_ids if int(image_id) in gt.images}


def _single_config_value(value: Any) -> str | None:
    if isinstance(value, str) and value:
        return value
    if isinstance(value, (list, tuple)) and len(value) == 1:
        return str(value[0])
    return None


def _camera_center(image: Any) -> np.ndarray:
    return np.asarray(image.cam_from_world().inverse().translation, dtype=np.float64).reshape(3)
