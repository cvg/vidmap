"""Trajectory conversion and timestamp-based pose remapping."""

from __future__ import annotations

import bisect
import logging
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pycolmap
from evo.core import metrics, sync

from vidmap.datasets.frame_names import timestamp_from_image_name

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _TimedImage:
    image_id: int
    timestamp: float


@dataclass(frozen=True)
class TimestampMatch:
    """Source samples bracketing one target timestamp."""

    before: int | None
    after: int | None
    fraction: float | None

    @property
    def is_exact(self) -> bool:
        return self.before is not None and self.before == self.after


@dataclass(frozen=True)
class PoseRemap:
    """Source poses expressed on the target reconstruction's image timeline."""

    reconstruction: pycolmap.Reconstruction
    target_subset: pycolmap.Reconstruction | None = None

    @property
    def used_exact_keyframes(self) -> bool:
        return self.target_subset is not None


def _posed_timeline(reconstruction, image_ids: Iterable[int] | None = None) -> tuple[_TimedImage, ...]:
    selected_ids = reconstruction.images.keys() if image_ids is None else image_ids
    timeline = []
    for image_id in selected_ids:
        image = reconstruction.images[image_id]
        if image.has_pose:
            timeline.append(_TimedImage(int(image_id), timestamp_from_image_name(image.name)))
    timeline.sort(key=lambda item: (item.timestamp, item.image_id))
    timestamps = [item.timestamp for item in timeline]
    if len(set(timestamps)) != len(timestamps):
        # Some datasets use non-timestamp image names. Their image IDs provide
        # the shared monotonic timeline across GT and reconstruction.
        return tuple(
            _TimedImage(item.image_id, float(item.image_id)) for item in sorted(timeline, key=lambda x: x.image_id)
        )
    return tuple(timeline)


def match_timestamps(target: Iterable[float], source: Iterable[float]) -> tuple[TimestampMatch, ...]:
    """Find exact or interpolating source samples for each target timestamp."""
    target = tuple(target)
    source = tuple(source)
    if tuple(sorted(source)) != source:
        raise ValueError("Source timestamps must be sorted")
    if not source and target:
        raise ValueError("Cannot match a non-empty target timeline without source timestamps")

    matches = []
    for timestamp in target:
        index = bisect.bisect_left(source, timestamp)
        if index < len(source) and source[index] == timestamp:
            matches.append(TimestampMatch(index, index, 0.0))
        elif index == 0:
            matches.append(TimestampMatch(None, 0, None))
        elif index == len(source):
            matches.append(TimestampMatch(len(source) - 1, None, None))
        else:
            previous, following = source[index - 1], source[index]
            matches.append(TimestampMatch(index - 1, index, (timestamp - previous) / (following - previous)))
    return tuple(matches)


def _empty_pose_reconstruction(template) -> pycolmap.Reconstruction:
    reconstruction = pycolmap.Reconstruction()
    for camera in template.cameras.values():
        reconstruction.add_camera_with_trivial_rig(camera)
    return reconstruction


def _add_target_image(reconstruction, target_image, pose) -> None:
    image = pycolmap.Image(
        image_id=target_image.image_id,
        camera_id=target_image.camera_id,
        name=target_image.name,
    )
    reconstruction.add_image_with_trivial_frame(image, pose)


def _exact_keyframe_remap(target, source, target_timeline, source_timeline) -> PoseRemap | None:
    target_matches = match_timestamps(
        (item.timestamp for item in source_timeline),
        (item.timestamp for item in target_timeline),
    )
    if not all(match.is_exact for match in target_matches):
        return None

    target_indices = [match.before for match in target_matches]
    if len(set(target_indices)) != len(target_indices):
        return None

    remapped = _empty_pose_reconstruction(target)
    target_subset = _empty_pose_reconstruction(target)
    for source_item, match in zip(source_timeline, target_matches, strict=True):
        target_item = target_timeline[match.before]
        target_image = target.images[target_item.image_id]
        _add_target_image(remapped, target_image, source.images[source_item.image_id].cam_from_world())
        _add_target_image(target_subset, target_image, target_image.cam_from_world())
    return PoseRemap(remapped, target_subset)


def remap_poses_to_timeline(
    target,
    source,
    *,
    prefer_exact_keyframes: bool = False,
) -> PoseRemap:
    """Map source poses to target image IDs by timestamp.

    Exact source keyframes can be retained without interpolation when every
    source timestamp exists on the target timeline. Otherwise, source poses are
    interpolated at every target timestamp inside the source time range.
    """
    target_timeline = _posed_timeline(target)
    source_timeline = _posed_timeline(source)
    if not target_timeline:
        raise ValueError("Target reconstruction has no posed images")
    if not source_timeline:
        raise ValueError("Source reconstruction has no posed images")

    if prefer_exact_keyframes:
        exact = _exact_keyframe_remap(target, source, target_timeline, source_timeline)
        if exact is not None:
            logger.info(
                "Pose remap retained %d exact source keyframes on a %d-frame target timeline",
                len(source_timeline),
                len(target_timeline),
            )
            return exact

    matches = match_timestamps(
        (item.timestamp for item in target_timeline),
        (item.timestamp for item in source_timeline),
    )
    remapped = _empty_pose_reconstruction(target)
    for target_item, match in zip(target_timeline, matches, strict=True):
        if match.fraction is None:
            continue
        target_image = target.images[target_item.image_id]
        if match.is_exact:
            source_item = source_timeline[match.before]
            pose = source.images[source_item.image_id].cam_from_world()
        else:
            before = source.images[source_timeline[match.before].image_id].cam_from_world().inverse()
            after = source.images[source_timeline[match.after].image_id].cam_from_world().inverse()
            pose = pycolmap.Rigid3d.interpolate(before, after, match.fraction).inverse()
        _add_target_image(remapped, target_image, pose)

    logger.info(
        "Pose remap interpolated %d target poses from %d source poses",
        remapped.num_reg_images(),
        len(source_timeline),
    )
    return PoseRemap(remapped)


def _select_subreconstruction(image_ids, reconstruction):
    """Copy selected posed images without mutating the source reconstruction."""
    image_ids = set(image_ids)
    camera_ids = {reconstruction.images[image_id].camera_id for image_id in image_ids}
    selected = pycolmap.Reconstruction()
    for camera_id in camera_ids:
        selected.add_camera_with_trivial_rig(reconstruction.cameras[camera_id])
    for image_id in image_ids:
        source = reconstruction.images[image_id]
        image = pycolmap.Image(image_id=source.image_id, camera_id=source.camera_id, name=source.name)
        selected.add_image_with_trivial_frame(image, source.cam_from_world())
    return selected


def align_reconstruction_to_reference_sequence(
    reconstruction: pycolmap.Reconstruction,
    reference: pycolmap.Reconstruction,
    max_error: float = 1.0,
    min_common: int = 3,
    colmap_bin: str = "colmap",
) -> tuple[pycolmap.Reconstruction, pycolmap.Sim3d]:
    """Align a reconstruction to the posed timeline in a reference model."""
    posed_image_ids = [image_id for image_id, image in reference.images.items() if image.has_pose]
    mapped_reference = remap_poses_to_timeline(
        reconstruction,
        _select_subreconstruction(posed_image_ids, reference),
    ).reconstruction

    with tempfile.TemporaryDirectory() as tmpdir:
        work_dir = Path(tmpdir)
        source_dir = work_dir / "source"
        reference_dir = work_dir / "reference"
        aligned_dir = work_dir / "aligned"
        transform_path = work_dir / "sim3.txt"
        source_dir.mkdir()
        reference_dir.mkdir()
        aligned_dir.mkdir()
        reconstruction.write(source_dir)
        mapped_reference.write(reference_dir)

        cmd = [
            colmap_bin,
            "model_aligner",
            "--input_path",
            str(source_dir),
            "--output_path",
            str(aligned_dir),
            "--ref_model_path",
            str(reference_dir),
            "--alignment_type",
            "custom",
            "--min_common_images",
            str(min_common),
            "--alignment_max_error",
            str(max_error),
            "--transform_path",
            str(transform_path),
        ]
        completed_process = subprocess.run(cmd, capture_output=True)
        if completed_process.returncode != 0 or not (aligned_dir / "cameras.bin").exists():
            used_translation_fallback = True
            shutil.copytree(source_dir, aligned_dir, dirs_exist_ok=True)

            source_centers = [image.projection_center() for image in reconstruction.images.values() if image.has_pose]
            reference_centers = [
                image.projection_center() for image in mapped_reference.images.values() if image.has_pose
            ]
            if source_centers and reference_centers:
                translation = np.stack(reference_centers).mean(0) - np.stack(source_centers).mean(0)
            else:
                translation = np.zeros(3)
            np.savetxt(transform_path, np.r_[1, 0, 0, 0, 0, translation])
        else:
            used_translation_fallback = False

        transform = np.loadtxt(transform_path)
        scale = transform[0]
        qw, qx, qy, qz = transform[1:5]
        rotation = pycolmap.Rotation3d([qx, qy, qz, qw])
        translation = transform[5:8]
        similarity = pycolmap.Sim3d(scale=scale, rotation=rotation, translation=translation)

        if used_translation_fallback:
            aligned = pycolmap.Reconstruction(source_dir)
            aligned.transform(similarity)
        else:
            aligned = pycolmap.Reconstruction(aligned_dir)

    return aligned, similarity


def reconstruction_to_evo_trajectory(reconstruction):
    """Convert posed reconstruction images to an evo trajectory."""
    timeline = _posed_timeline(reconstruction)
    if not timeline:
        raise ValueError("Cannot build a trajectory from a reconstruction without posed images")
    timestamps = np.asarray([item.timestamp for item in timeline], dtype=np.float64)
    if np.any(np.diff(timestamps) <= 0):
        raise ValueError("Posed image timestamps must be unique and strictly increasing")

    poses = [reconstruction.images[item.image_id].cam_from_world().inverse() for item in timeline]
    pose_rows = np.vstack([np.concatenate([pose.translation, pose.rotation.quat]) for pose in poses])

    from evo.core.trajectory import PoseTrajectory3D

    return PoseTrajectory3D(
        positions_xyz=pose_rows[:, :3],
        orientations_quat_wxyz=pose_rows[:, [6, 3, 4, 5]],
        timestamps=timestamps,
    )


def compute_ate_statistics(ground_truth_trajectory, estimated_trajectory):
    """Compute translation APE statistics after timestamp association."""
    ground_truth, estimated = sync.associate_trajectories(
        ground_truth_trajectory,
        estimated_trajectory,
        max_diff=1e7,
    )
    if len(ground_truth.timestamps) == 0:
        return None

    metric = metrics.APE(metrics.PoseRelation.translation_part)
    metric.process_data((ground_truth, estimated))
    return metric.get_all_statistics()
