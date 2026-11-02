from __future__ import annotations

import importlib
import json
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

import numpy as np
import pycolmap
import yaml
from PIL import Image

if TYPE_CHECKING:
    from vidmap.datasets.capture import PreparedCapture
    from vidmap.datasets.video import VideoDatasetManifest

from vidmap.datasets.names import EvaluationMetric

CANONICAL_UNPOSED_IMAGES_FILENAME = "unposed_images.json"
_UNPOSED_SENTINEL_QUATERNION = np.array([0.0, 0.0, 0.0, -1.0])
_UNPOSED_SENTINEL_TRANSLATION = np.zeros(3)


def image_has_public_pose(image: pycolmap.Image) -> bool:
    """Return whether an image has a pose other than the exact unavailable-GT sentinel."""
    if not image.has_pose:
        return False
    pose = image.cam_from_world()
    return not (
        np.array_equal(pose.rotation.quat, _UNPOSED_SENTINEL_QUATERNION)
        and np.array_equal(pose.translation, _UNPOSED_SENTINEL_TRANSLATION)
    )


def validate_path_component(value: object, label: str) -> str:
    """Validate one lexical relative path component before joining it to a root."""
    if not isinstance(value, str):
        raise TypeError(f"{label} must be a string, got {type(value).__name__}")
    invalid = (
        not value
        or value != value.strip()
        or value in {".", ".."}
        or Path(value).is_absolute()
        or any(character in value for character in "/\\\0")
        or any(ord(character) < 32 or ord(character) == 127 for character in value)
    )
    if invalid:
        raise ValueError(f"{label} must be one unambiguous relative path component, got {value!r}")
    return value


@dataclass(frozen=True)
class PreparedTestset:
    """A benchmark selection with an explicit prepared input, within its scene."""

    recording: str
    variant: str
    image_ids: tuple[int, ...]

    def __post_init__(self) -> None:
        validate_path_component(self.recording, "Recording")
        validate_path_component(self.variant, "Image variant")
        if any(
            type(image_id) is not int or not 0 <= image_id < pycolmap.INVALID_IMAGE_ID for image_id in self.image_ids
        ):
            raise ValueError("Testsets must contain valid integer image IDs")
        object.__setattr__(self, "image_ids", tuple(self.image_ids))

    def as_dict(self) -> dict:
        return dict(recording=self.recording, variant=self.variant, image_ids=list(self.image_ids))


@dataclass(frozen=True)
class DatasetLayout:
    name: str
    data_dir: Path
    default_exp_dir: Path
    default_cache_dir: Path
    testsets: Path
    scenes: tuple[str, ...]
    manifest: VideoDatasetManifest | None = None

    def scene_dir(self, scene: str) -> Path:
        return self.data_dir / validate_path_component(scene, "Dataset scene")

    def recording_dir(self, scene: str, recording: str) -> Path:
        return self.scene_dir(scene) / validate_path_component(recording, "Recording")

    def input_dir(self, scene: str, recording: str, variant: str) -> Path:
        return self.recording_dir(scene, recording) / validate_path_component(variant, "Image variant")

    def images_dir(self, scene: str, recording: str, variant: str) -> Path:
        return self.input_dir(scene, recording, variant) / "images"

    def reconstruction_dir(self, scene: str, recording: str, variant: str) -> Path:
        return self.input_dir(scene, recording, variant) / "rec"

    def testset_path(self, scene: str, mode: str) -> Path:
        scene = validate_path_component(scene, "Dataset scene")
        mode = validate_path_component(mode, "Dataset testset mode")
        return self.testsets / scene / f"{mode}.yaml"

    def select_scenes(self, scenes: Iterable[str] | None) -> tuple[str, ...]:
        if scenes is None:
            return self.scenes
        if isinstance(scenes, (str, bytes)):
            raise TypeError("Dataset scenes must be an iterable of scene names, not a string")
        selected = tuple(scenes)
        if any(not isinstance(scene, str) or not scene for scene in selected):
            raise ValueError(f"Dataset scenes must be non-empty strings, got {selected!r}")
        selected_set = set(selected)
        if len(selected_set) != len(selected):
            raise ValueError(f"Dataset scenes must be unique, got {selected!r}")
        unknown = sorted(selected_set - set(self.scenes))
        if unknown:
            raise ValueError(f"Dataset {self.name!r} does not define scenes: {unknown}")
        return tuple(scene for scene in self.scenes if scene in selected_set)

    def read_testsets(self, scene: str, mode: str) -> dict[str, PreparedTestset] | None:
        path = self.testset_path(scene, mode)
        if not path.exists():
            return None
        try:
            with path.open(encoding="utf-8") as file:
                testsets = yaml.safe_load(file)
        except (UnicodeDecodeError, yaml.YAMLError) as error:
            raise ValueError(f"Invalid testset file {path}: {error}") from error
        if not isinstance(testsets, Mapping):
            raise ValueError(f"Invalid testset file {path}: expected a YAML mapping")

        normalized = {}
        for key, selection in testsets.items():
            if isinstance(key, bool) or not isinstance(key, (str, int)):
                raise ValueError(f"Invalid testset file {path}: testset IDs must be strings or integers, got {key!r}")
            testset_id = str(key)
            try:
                validate_path_component(testset_id, "Testset ID")
            except (TypeError, ValueError) as error:
                raise ValueError(f"Invalid testset file {path}: unsafe testset ID {key!r}") from error
            if testset_id in normalized:
                raise ValueError(f"Invalid testset file {path}: duplicate testset ID {testset_id!r}")
            if not isinstance(selection, Mapping) or set(selection) != {"recording", "variant", "image_ids"}:
                raise ValueError(
                    f"Invalid testset file {path}: {testset_id!r} must identify its recording and variant"
                )
            if not isinstance(selection["image_ids"], list):
                raise ValueError(f"Invalid testset file {path}: {testset_id!r} image_ids must be a list")
            normalized[testset_id] = PreparedTestset(**selection)
        return normalized


class DatasetParser(Protocol):
    scene: str
    rgb_dir: Path
    reconstruction_dir: Path | None
    rec: pycolmap.Reconstruction

    def frontend_capture(self) -> PreparedCapture: ...


@dataclass(frozen=True)
class DatasetCapabilities:
    allows_missing_testsets: bool = False
    ignores_unknown_testsets: bool = False


@dataclass(frozen=True)
class DatasetPreparation:
    module: str
    profiles: tuple[tuple[str, str], ...] = ()

    def profile(self, mode: str) -> str:
        profiles = dict(self.profiles)
        if mode not in profiles:
            raise ValueError(f"Unsupported preparation mode {mode!r}; expected one of {tuple(profiles)}")
        return profiles[mode]

    def __call__(self, scenes: tuple[str, ...] | None = None, *, mode: str) -> None:
        module = importlib.import_module(self.module)
        if self.profiles:
            module.main(scenes=scenes, profile=self.profile(mode))
        else:
            module.main(scenes=scenes)


def _missing_standard_scene_artifacts(layout: DatasetLayout, scenes: tuple[str, ...], mode: str) -> tuple[Path, ...]:
    missing = []
    for scene in scenes:
        testsets = layout.read_testsets(scene, mode)
        if testsets is None:
            missing.append(layout.testset_path(scene, mode))
            continue
        for recording, variant in dict.fromkeys((case.recording, case.variant) for case in testsets.values()):
            images = layout.images_dir(scene, recording, variant)
            reconstruction = layout.reconstruction_dir(scene, recording, variant)
            receipt = layout.recording_dir(scene, recording) / "preparation.json"
            if not receipt.is_file():
                missing.append(receipt)
            if not (images / "cam0").is_dir() or not any(path.is_file() for path in (images / "cam0").rglob("*")):
                missing.append(images)
            if not reconstruction.is_dir() or not any(reconstruction.iterdir()):
                missing.append(reconstruction)
            if layout.manifest is not None:
                sidecar = layout.input_dir(scene, recording, variant) / CANONICAL_UNPOSED_IMAGES_FILENAME
                if not sidecar.is_file():
                    missing.append(sidecar)
    return tuple(missing)


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    layout: DatasetLayout
    preparer: DatasetPreparation | Callable[[], None] | None = None
    capabilities: DatasetCapabilities = DatasetCapabilities()
    missing_artifacts: Callable[[DatasetLayout, tuple[str, ...], str], tuple[Path, ...]] | None = (
        _missing_standard_scene_artifacts
    )
    default_mode: str = "minimal"
    default_evaluation_metric: EvaluationMetric = "pose_auc"

    def __post_init__(self) -> None:
        if self.name != self.layout.name:
            raise ValueError(f"Dataset spec {self.name!r} does not match layout {self.layout.name!r}")
        validate_path_component(self.default_mode, "Default testset mode")
        if self.default_evaluation_metric not in {"pose_auc", "wate_auc"}:
            raise ValueError(f"Unsupported default evaluation metric {self.default_evaluation_metric!r}")

    def _missing_preparation_artifacts(self, scenes: tuple[str, ...], mode: str) -> tuple[Path, ...]:
        missing = self.missing_artifacts(self.layout, scenes, mode)
        if not self.capabilities.allows_missing_testsets:
            missing += tuple(
                self.layout.testset_path(scene, mode)
                for scene in scenes
                if not self.layout.testset_path(scene, mode).is_file()
            )
        return missing

    def prepare(self, scenes: Iterable[str] | None = None, *, mode: str | None = None) -> None:
        if self.missing_artifacts is None:
            return
        selected_scenes = self.layout.select_scenes(scenes)
        if not selected_scenes:
            return
        mode = self.default_mode if mode is None else mode
        missing = self._missing_preparation_artifacts(selected_scenes, mode)
        if not missing:
            return
        if self.preparer is None:
            expected = ", ".join(str(path) for path in missing)
            raise FileNotFoundError(
                f"Dataset {self.name!r} is not prepared and does not support automatic preparation; "
                f"provide non-empty prepared artifacts at: {expected}"
            )
        if isinstance(self.preparer, DatasetPreparation):
            self.preparer(selected_scenes, mode=mode)
        else:
            self.preparer()
        missing = self._missing_preparation_artifacts(selected_scenes, mode)
        if missing:
            expected = ", ".join(str(path) for path in missing)
            raise RuntimeError(
                f"Dataset preparer for {self.name!r} did not satisfy its postcondition; "
                f"missing non-empty prepared artifacts: {expected}"
            )

    def select_testsets(
        self, scene: str, mode: str, testset_ids: Iterable[str] | None = None
    ) -> dict[str, PreparedTestset]:
        testsets = self.layout.read_testsets(scene, mode)
        if testsets is None:
            if self.capabilities.allows_missing_testsets:
                return {}
            raise FileNotFoundError(self.layout.testset_path(scene, mode))
        if testset_ids is None:
            return testsets
        if self.capabilities.ignores_unknown_testsets:
            return {testset_id: testsets[testset_id] for testset_id in testset_ids if testset_id in testsets}
        return {testset_id: testsets[testset_id] for testset_id in testset_ids}

    def parsers_for_cases(self, scene: str, selections: Iterable[PreparedTestset]) -> Iterator[DatasetParser]:
        parser = None
        current = None
        for selection in selections:
            key = (selection.recording, selection.variant)
            if key != current:
                parser = PreparedSceneParser(self.layout, scene, selection)
                current = key
            yield parser


class PreparedSceneParser:
    def __init__(self, layout: DatasetLayout, scene: str, selection: PreparedTestset) -> None:
        self.layout = layout
        self.scene = scene
        self.recording = selection.recording
        self.variant = selection.variant
        self.input_dir = layout.input_dir(scene, self.recording, self.variant)
        self.rgb_dir = self.input_dir / "images"
        self.reconstruction_dir = self.input_dir / "rec"
        preparation = json.loads((layout.recording_dir(scene, self.recording) / "preparation.json").read_text())
        self.input_metadata = preparation["variants"][self.variant]
        self.rec = pycolmap.Reconstruction(self.reconstruction_dir)
        if self.rec.num_points3D() or any(image.num_points2D() for image in self.rec.images.values()):
            raise ValueError(
                f"Prepared reference models must contain only camera poses and calibration: {self.reconstruction_dir}"
            )
        images_by_name = {image.name: image for image in self.rec.images.values()}
        unposed_names = {
            name for name, image in images_by_name.items() if image.has_pose and not image_has_public_pose(image)
        }
        if layout.manifest is not None:
            layout.manifest.get(scene)
            sidecar = self.input_dir / CANONICAL_UNPOSED_IMAGES_FILENAME
            if not sidecar.is_file():
                raise FileNotFoundError(f"Canonical scene is missing unposed-image sidecar: {sidecar}")
            try:
                payload = json.loads(sidecar.read_text(encoding="utf-8"))
                sidecar_names = payload["image_names"]
            except (json.JSONDecodeError, KeyError, TypeError) as error:
                raise ValueError(f"Malformed canonical unposed-image sidecar: {sidecar}") from error
            if (
                not isinstance(sidecar_names, list)
                or not all(isinstance(name, str) for name in sidecar_names)
                or len(sidecar_names) != len(set(sidecar_names))
            ):
                raise ValueError(f"Canonical unposed-image sidecar must contain unique image names: {sidecar}")
            unknown = set(sidecar_names) - set(images_by_name)
            if unknown:
                raise ValueError(f"Canonical unposed-image sidecar references unknown images: {sorted(unknown)}")
            unposed_names.update(sidecar_names)
        for frame_id in sorted(
            {images_by_name[name].frame_id for name in unposed_names if images_by_name[name].has_pose}
        ):
            self.rec.deregister_frame(frame_id)

    def frontend_capture(self) -> PreparedCapture:
        """Copy input geometry without exposing ground-truth poses to the frontend."""
        from vidmap.datasets.capture import PreparedCapture, copy_unposed_reconstruction

        if self.input_metadata["rig"] is not None:
            from vidmap.datasets.prepare.helpers.rig_capture import load_capture
            from vidmap.frontend.options.rig import RigOptions

            options = RigOptions(**self.input_metadata["rig"])
            streams = {Path(image.name).parts[0] for image in self.rec.images.values()}
            if streams == {"cam0"}:
                options = replace(options, cameras=options.cameras[:1], frames=None)
            elif streams != {f"cam{i}" for i in range(len(options.cameras))}:
                raise ValueError("Prepared rig selection must contain cam0 or all declared streams")
            capture = load_capture(self.rgb_dir, options)
            selected = copy_unposed_reconstruction(capture.rec, [image.name for image in self.rec.images.values()])
            return replace(
                capture,
                rec=selected,
                frame_timestamps_ns={fid: capture.frame_timestamps_ns[fid] for fid in selected.frames},
            )

        return PreparedCapture(
            rec=copy_unposed_reconstruction(self.rec),
            rgb_dir=self.rgb_dir,
            frame_timestamps_ns=None,
            provenance={},
            preserve_calibration=False,
            forced_keyframe_names=(
                frozenset(image.name for image in self.rec.images.values() if image.has_pose)
                if self.input_metadata["force_gt_keyframes"]
                else frozenset()
            ),
        )

    def camera(self, imid: int) -> pycolmap.Camera:
        return self.rec.cameras[self.rec.images[imid].camera_id]

    def pose(self, imid: int) -> pycolmap.Rigid3d:
        return self.rec.images[imid].cam_from_world()

    def image_name(self, imid: int) -> str:
        return self.rec.images[imid].name

    def rgb(self, imid: int) -> np.ndarray:
        with Image.open(self.rgb_dir / self.image_name(imid)) as image:
            return np.array(image)
