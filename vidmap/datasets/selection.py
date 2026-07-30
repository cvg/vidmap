"""Prepared dataset target selection and parser construction."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING

import pycolmap

from vidmap.datasets.base import image_has_public_pose
from vidmap.datasets.registry import get_dataset_spec

if TYPE_CHECKING:
    from vidmap.configuration.defaults import TargetSelectionOptions
    from vidmap.datasets.base import DatasetParser, DatasetSpec


def validate_profile_selection(selection: PreparedTargetSelection, *, enabled: bool) -> None:
    """Reject ambiguous profiles without preparing or executing targets."""
    if not enabled:
        return
    target_count = sum(len(targets) for _, targets in selection.iter_scene_targets())
    if target_count != 1:
        raise ValueError(f"--profile requires exactly one selected target; resolved {target_count}")


@dataclass(frozen=True)
class PreparedTarget:
    """One selected prepared-dataset reconstruction target."""

    scene: str
    testset_type: str
    testset_desc: str
    ref_imids: tuple[int, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "ref_imids", tuple(self.ref_imids))


class PreparedTargetSelection:
    """Select prepared-dataset targets and construct their scene parsers."""

    def __init__(self, options: TargetSelectionOptions, dataset: str | DatasetSpec) -> None:
        self.options = options
        self.dataset_spec = get_dataset_spec(dataset) if isinstance(dataset, str) else dataset
        self.dataset_layout = self.dataset_spec.layout

    def iter_scene_targets(self) -> Iterator[tuple[str, tuple[PreparedTarget, ...]]]:
        scenes = self.dataset_layout.scenes if self.options.scene is None else self.options.scene
        for scene in scenes:
            if not self.scene_has_public_ground_truth(scene) and self.options.mode != "all":
                raise ValueError(
                    f"{self.dataset_spec.name}/{scene} is reconstruction-only and supports mode=all; "
                    "pose-selected testsets require public ground truth"
                )
            testsets = self.dataset_spec.select_testsets(scene, self.options.mode, self.options.testset_id)
            targets = tuple(
                PreparedTarget(
                    scene=scene,
                    testset_type=self.options.mode,
                    testset_desc=str(testset_id),
                    ref_imids=tuple(ref_imids),
                )
                for testset_id, ref_imids in testsets.items()
            )
            yield scene, targets

    def _configure_testset_reconstructions(self, scene_parser, reference_image_ids, source_reconstruction):
        if self.options.mode == "all":
            reference_image_names = [scene_parser.rec.images[image_id].name for image_id in reference_image_ids]
            scene_parser.kf_names = reference_image_names
            return reference_image_names

        scene_parser.rec = pycolmap.Reconstruction()
        for image_id in reference_image_ids:
            source_image = source_reconstruction.images[image_id]
            camera = source_image.camera
            has_pose = image_has_public_pose(source_image)
            pose = source_image.cam_from_world() if has_pose else pycolmap.Rigid3d()
            if camera.camera_id not in scene_parser.rec.cameras:
                scene_parser.rec.add_camera_with_trivial_rig(camera)
            testset_image = pycolmap.Image(
                image_id=source_image.image_id,
                camera_id=source_image.camera_id,
                name=source_image.name,
            )
            scene_parser.rec.add_image_with_trivial_frame(testset_image, pose)
            if not has_pose:
                scene_parser.rec.deregister_frame(image_id)

        reference_image_names = [scene_parser.rec.images[image_id].name for image_id in reference_image_ids]
        scene_parser.kf_names = reference_image_names
        return reference_image_names

    def scene_has_public_ground_truth(self, scene: str) -> bool:
        """Return manifest GT availability, defaulting manifest-less datasets to evaluable."""
        if self.dataset_layout.manifest is None:
            return True
        return self.dataset_layout.manifest.get(scene).gt_evaluable

    def prepared_targets(self) -> Iterator[tuple[PreparedTarget, DatasetParser, list[str]]]:
        """Yield selected targets with their target-specific parser and image names."""
        for scene, scene_targets in self.iter_scene_targets():
            descriptions = (target.testset_desc for target in scene_targets)
            parsers = self.dataset_spec.parsers_for_cases(scene, descriptions)
            source_parser = None
            for target, scene_parser in zip(scene_targets, parsers, strict=True):
                if scene_parser is not source_parser:
                    source_parser = scene_parser
                    source_reconstruction = scene_parser.rec
                reference_image_names = self._configure_testset_reconstructions(
                    scene_parser,
                    target.ref_imids,
                    source_reconstruction,
                )
                yield target, scene_parser, reference_image_names
