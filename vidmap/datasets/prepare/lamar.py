"""Download and exactly reproduce the established iOS-only LaMAR benchmark."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import logging
import re
import shutil
import tarfile
import zipfile
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pycolmap
import yaml
from PIL import Image

from vidmap.datasets.layouts import get_dataset_layout
from vidmap.datasets.prepare import filesystem as prep_fs
from vidmap.datasets.prepare.video_common import resumable_http_download
from vidmap.datasets.resources import dataset_asset

logger = logging.getLogger(__name__)
LAMAR_LAYOUT = get_dataset_layout("lamar")

_GROUND_TRUTH_MODELS_FILENAME = "lamar_ground_truth_models.tar.gz"
_GROUND_TRUTH_MODELS_SIZE = 11_689_090
_GROUND_TRUTH_MODELS_SHA256 = "60770ead917d936fcef7b0115a8d67cd8044e7dc6655321e7e837923b3736428"
_GROUND_TRUTH_MODELS_URL = (
    f"https://github.com/cvg/vidmap/releases/download/dataset-assets-v2/{_GROUND_TRUTH_MODELS_FILENAME}"
)
_PREPARATION_PROFILE = "sample-300"
_SAMPLE_SHA256 = {
    "CAB": "71f3741af6f372a3db7fd2ab10c89b7111c0ff537c16a13baa68fff29462a08a",
    "HGE": "e496c0b26920e02139d2d183d6466747074e17f679c4b20a4e4721247dc1a890",
    "LIN": "273ea2532e036097179d5b96e6e486e4913b44472f4af81d01cba7f1b1821114",
}
_EXPECTED_GT_IMAGE_COUNTS = {"CAB": 206_161, "HGE": 130_953, "LIN": 117_970}
_EXPECTED_SAMPLE_IMAGE_COUNTS = {"CAB": 36_334, "HGE": 27_289, "LIN": 30_866}
_MODEL_FILES = ("cameras.bin", "images.bin", "points3D.bin")
_SAMPLE_TARGET = re.compile(r"^(ios_.+)_\d{3}-\d+$")


@dataclass(frozen=True)
class SourceArchive:
    scene: str
    filename: str
    size: int
    sha256: str
    url: str


@dataclass(frozen=True)
class RawCamera:
    width: int
    height: int
    params: tuple[float, float, float, float]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _ground_truth_models_archive(downloads: Path, *, download: bool) -> Path:
    path = downloads / _GROUND_TRUTH_MODELS_FILENAME
    if not download and not path.is_file():
        raise FileNotFoundError(f"Missing LaMAR ground-truth archive: {path}")
    return resumable_http_download(
        _GROUND_TRUTH_MODELS_URL,
        path,
        expected_size=_GROUND_TRUTH_MODELS_SIZE,
        expected_digest=_GROUND_TRUTH_MODELS_SHA256,
    )


def source_inventory() -> dict[str, tuple[SourceArchive, ...]]:
    """Load the checksum-pinned official archives needed by sample-300."""
    payload = json.loads(dataset_asset("lamar", "source_archives.json").read_text(encoding="utf-8"))
    base_url = payload["base_url"]
    inventory = {}
    for scene in LAMAR_LAYOUT.scenes:
        required = _sample_sessions(scene)
        entries = []
        for raw in payload["scenes"][scene]["archives"]:
            if Path(raw["filename"]).stem in required:
                entries.append(
                    SourceArchive(
                        scene=scene,
                        filename=raw["filename"],
                        size=int(raw["size"]),
                        sha256=raw["sha256"],
                        url=base_url.format(scene=scene) + raw["filename"],
                    )
                )
        selected = {Path(entry.filename).stem for entry in entries}
        if selected != required:
            raise ValueError(
                f"LaMAR {scene} source inventory does not cover sample-300: {sorted(required - selected)}"
            )
        inventory[scene] = tuple(entries)
    return inventory


def parse_sensors(text: str) -> dict[int, RawCamera]:
    """Parse LaMAR iOS per-frame PINHOLE calibration rows."""
    cameras = {}
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line or line.startswith("#"):
            continue
        fields = [value.strip() for value in line.split(",")]
        if len(fields) >= 3 and fields[2] != "camera":
            continue
        if len(fields) != 10 or fields[2:4] != ["camera", "PINHOLE"] or not fields[0].startswith("cam_phone_"):
            raise ValueError(f"Invalid LaMAR sensor row {line_number}: {line!r}")
        timestamp = int(fields[0].removeprefix("cam_phone_"))
        if timestamp in cameras:
            raise ValueError(f"Duplicate LaMAR camera timestamp {timestamp}")
        camera = RawCamera(
            int(fields[4]),
            int(fields[5]),
            tuple(float(value) for value in fields[6:10]),
        )
        if camera.width <= 0 or camera.height <= 0 or not np.isfinite(camera.params).all():
            raise ValueError(f"Invalid LaMAR camera at timestamp {timestamp}")
        cameras[timestamp] = camera
    if not cameras:
        raise ValueError("LaMAR sensors file contains no phone cameras")
    return cameras


def parse_trajectories(text: str) -> dict[int, pycolmap.Rigid3d]:
    """Parse raw rig-to-world poses and return camera-from-world poses."""
    poses = {}
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line or line.startswith("#"):
            continue
        fields = [value.strip() for value in line.split(",")]
        if len(fields) != 9:
            raise ValueError(f"Invalid LaMAR trajectory row {line_number}: {line!r}")
        timestamp = int(fields[0])
        if timestamp in poses:
            raise ValueError(f"Duplicate LaMAR trajectory timestamp {timestamp}")
        values = np.asarray([float(value) for value in fields[2:]], dtype=np.float64)
        if not np.isfinite(values).all() or np.linalg.norm(values[:4]) <= np.finfo(np.float64).eps:
            raise ValueError(f"Invalid LaMAR pose at timestamp {timestamp}")
        qw, qx, qy, qz, tx, ty, tz = values
        world_from_camera = pycolmap.Rigid3d(pycolmap.Rotation3d([qx, qy, qz, qw]), [tx, ty, tz])
        poses[timestamp] = world_from_camera.inverse()
    if not poses:
        raise ValueError("LaMAR trajectories file contains no poses")
    return poses


def rotate_camera(camera: RawCamera, rotations: int) -> RawCamera:
    """Apply the LaMAR image-rotation convention to a raw camera calibration."""
    rotations = (-rotations) % 4
    width, height = camera.width, camera.height
    fx, fy, cx, cy = camera.params
    if rotations == 0:
        return camera
    if rotations == 1:
        cx, cy = cy, width - cx
    elif rotations == 2:
        cx, cy = width - cx, height - cy
    else:
        cx, cy = height - cy, cx
    if rotations % 2:
        return RawCamera(height, width, (fy, fx, cx, cy))
    return RawCamera(width, height, (fx, fy, cx, cy))


def infer_rotation(camera: RawCamera, target: pycolmap.Camera) -> int:
    """Infer the unique image rotation that maps raw calibration to the GT camera."""
    matches = []
    for rotations in range(4):
        candidate = rotate_camera(camera, rotations)
        if (candidate.width, candidate.height) == (
            target.width,
            target.height,
        ) and np.allclose(candidate.params, target.params, rtol=0.0, atol=1e-9):
            matches.append(rotations)
    if len(matches) != 1:
        raise ValueError(f"Could not uniquely match LaMAR camera {camera} to {target}; rotations={matches}")
    return matches[0]


def generate_testsets(scene: str, image_ids: set[int]) -> dict[str, dict[str, list[int]]]:
    """Reproduce the public sample-300 benchmark set."""
    with gzip.open(dataset_asset("lamar", "ground_truth_subsequences.json.gz"), "rt", encoding="utf-8") as file:
        payload = json.load(file)[scene]
    result: dict[str, dict[str, list[int]]] = {"sample": {}}
    for subsequence, anchors in payload.items():
        if not anchors:
            raise ValueError(f"Empty LaMAR GT subsequence {scene}/{subsequence}")
        chunks = [anchors[index : index + 300] for index in range(0, len(anchors), 300)]
        chunks = [chunk for chunk in chunks if len(chunk) == 300][:20]
        for index, chunk in enumerate(chunks):
            result["sample"][f"{subsequence}-{index}"] = [
                image_id for image_id in range(chunk[0], chunk[-1] + 1) if image_id in image_ids
            ]
    return result


def canonical_testsets(scene: str) -> dict[str, bytes]:
    """Read the exact canonical sample-300 YAML bytes for one scene."""
    return {"sample": dataset_asset("lamar", "testsets", scene, "sample-300.yaml").read_bytes()}


def _sample_mapping(scene: str) -> dict[str, list[int]]:
    mapping = yaml.safe_load(canonical_testsets(scene)["sample"])
    if not isinstance(mapping, dict):
        raise ValueError(f"Bundled LaMAR sample-300 testset is malformed for {scene}")
    return mapping


def _sample_sessions(scene: str) -> set[str]:
    sessions = set()
    for target in _sample_mapping(scene):
        match = _SAMPLE_TARGET.fullmatch(target)
        if match is None:
            raise ValueError(f"Invalid LaMAR sample-300 target: {scene}/{target}")
        sessions.add(match.group(1))
    return sessions


def _sample_image_names(scene: str, ground_truth: pycolmap.Reconstruction) -> set[str]:
    image_ids = {image_id for values in _sample_mapping(scene).values() for image_id in values}
    by_id = {int(image.image_id): str(image.name) for image in ground_truth.images.values()}
    missing = image_ids - by_id.keys()
    if missing:
        raise ValueError(f"LaMAR {scene} sample-300 references missing GT image IDs: {sorted(missing)[:3]}")
    names = {by_id[image_id] for image_id in image_ids}
    expected = _EXPECTED_SAMPLE_IMAGE_COUNTS[scene]
    if len(names) != expected:
        raise ValueError(f"LaMAR {scene} sample-300 image count mismatch: expected {expected}, got {len(names)}")
    sessions = {Path(name).parent.as_posix() for name in names}
    if sessions != _sample_sessions(scene):
        raise ValueError(f"LaMAR {scene} sample-300 session mapping is inconsistent")
    return names


def _ensure_testsets(scenes: Iterable[str]) -> None:
    """Install exact canonical YAML bytes, refusing to replace any mismatch."""
    for scene in scenes:
        for mode, contents in canonical_testsets(scene).items():
            destination = LAMAR_LAYOUT.testsets / scene / f"{mode}.yaml"
            if destination.exists() or destination.is_symlink():
                if not destination.is_file() or destination.read_bytes() != contents:
                    raise RuntimeError(f"Existing LaMAR testset differs from the canonical asset: {destination}")
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_suffix(destination.suffix + ".part")
            if temporary.exists():
                raise RuntimeError(f"Unexpected LaMAR testset staging artifact: {temporary}")
            temporary.write_bytes(contents)
            temporary.rename(destination)


def _extract_ground_truth(scene: str, destination: Path, ground_truth_models: Path) -> None:
    destination.mkdir(parents=True)
    with tarfile.open(ground_truth_models) as archive:
        for filename in _MODEL_FILES:
            member = archive.getmember(f"{scene}/rec/{filename}")
            source = archive.extractfile(member)
            if source is None:
                raise ValueError(f"LaMAR ground-truth release asset is missing {scene}/rec/{filename}")
            with source, (destination / filename).open("xb") as target:
                shutil.copyfileobj(source, target, length=8 * 1024 * 1024)


def _copy_image(bundle: zipfile.ZipFile, member: str, destination: Path, rotations: int) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    if temporary.exists():
        raise RuntimeError(f"Unexpected LaMAR staging artifact: {temporary}")
    with bundle.open(member) as source:
        if rotations:
            operations = (
                Image.Transpose.ROTATE_270,
                Image.Transpose.ROTATE_180,
                Image.Transpose.ROTATE_90,
            )
            with Image.open(source) as image:
                image.transpose(operations[rotations - 1]).save(temporary, format="JPEG")
        else:
            with temporary.open("xb") as target:
                shutil.copyfileobj(source, target, length=8 * 1024 * 1024)
    temporary.rename(destination)


def _convert_scene(
    scene: str,
    archives: tuple[Path, ...],
    destination: Path,
    ground_truth_models: Path,
) -> None:
    rec_dir = destination / "rec"
    _extract_ground_truth(scene, rec_dir, ground_truth_models)
    ground_truth = pycolmap.Reconstruction(rec_dir)
    if ground_truth.num_images() != _EXPECTED_GT_IMAGE_COUNTS[scene]:
        raise ValueError(f"LaMAR {scene} GT image count mismatch")
    by_name = {image.name: image for image in ground_truth.images.values()}
    required_names = _sample_image_names(scene, ground_truth)
    archive_sessions = {path.stem for path in archives}
    if archive_sessions != _sample_sessions(scene):
        raise ValueError(f"LaMAR {scene} archives do not exactly match the sample-300 sessions")

    seen_names = set()
    rotated = 0
    for archive_path in archives:
        sequence = archive_path.stem
        logger.info(f"Preparing LaMAR {scene}/{sequence}")
        with zipfile.ZipFile(archive_path) as bundle:
            prefix = f"{sequence}/"
            sensors = parse_sensors(bundle.read(prefix + "sensors.txt").decode("utf-8"))
            poses = parse_trajectories(bundle.read(prefix + "trajectories.txt").decode("utf-8"))
            if set(sensors) != set(poses):
                raise ValueError(f"LaMAR {scene}/{sequence} sensor/trajectory timestamp mismatch")
            members = set(bundle.namelist())
            for timestamp in poses:
                name = f"{sequence}/{timestamp}.jpg"
                if name not in required_names:
                    continue
                if name in seen_names or name not in by_name:
                    raise ValueError(f"Unexpected or duplicate LaMAR image {scene}/{name}")
                gt_image = by_name[name]
                rotations = infer_rotation(sensors[timestamp], gt_image.camera)
                member = prefix + f"raw_data/images/{timestamp}.jpg"
                if member not in members:
                    raise ValueError(f"LaMAR archive is missing {member}")
                _copy_image(bundle, member, destination / "images" / name, rotations)
                rotated += bool(rotations)
                seen_names.add(name)

    if seen_names != required_names:
        missing = sorted(required_names - seen_names)
        raise ValueError(
            f"LaMAR {scene} source inventory omitted {len(missing)} sample-300 images, including {missing[:3]}"
        )
    generated = generate_testsets(scene, set(ground_truth.images))
    for mode, expected in generated.items():
        testset = LAMAR_LAYOUT.testset_path(scene, mode)
        if yaml.safe_load(testset.read_text(encoding="utf-8")) != expected:
            raise ValueError(f"Canonical LaMAR {mode} testset content mismatch: {testset}")
    with (destination / "preparation.json").open("x", encoding="utf-8") as file:
        json.dump(
            {
                "dataset": "LaMAR",
                "data_version": "2.2",
                "device": "iOS",
                "ground_truth_asset_sha256": _GROUND_TRUTH_MODELS_SHA256,
                "images": len(seen_names),
                "profile": _PREPARATION_PROFILE,
                "rotated_images": rotated,
                "scene": scene,
                "source_archives": [path.name for path in archives],
            },
            file,
            indent=2,
            sort_keys=True,
        )
        file.write("\n")


def lamar_scene_is_prepared(scene_dir: Path) -> bool:
    """Return whether an existing scene has every canonical LaMAR artifact."""
    scene = scene_dir.name
    if scene not in _EXPECTED_GT_IMAGE_COUNTS:
        return False
    if not prep_fs.complete_colmap_model(scene_dir / "rec"):
        return False
    manifest_path = scene_dir / "preparation.json"
    if not manifest_path.is_file():
        return False
    ground_truth = pycolmap.Reconstruction(scene_dir / "rec")
    if ground_truth.num_images() != _EXPECTED_GT_IMAGE_COUNTS[scene]:
        return False
    expected_names = _sample_image_names(scene, ground_truth)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        return False
    manifest_keys = {
        "dataset",
        "data_version",
        "device",
        "ground_truth_asset_sha256",
        "images",
        "profile",
        "rotated_images",
        "scene",
        "source_archives",
    }
    if set(manifest) != manifest_keys:
        return False
    expected_archives = [source.filename for source in source_inventory()[scene]]
    if (
        manifest["dataset"] != "LaMAR"
        or manifest["data_version"] != "2.2"
        or manifest["device"] != "iOS"
        or manifest["ground_truth_asset_sha256"] != _GROUND_TRUTH_MODELS_SHA256
        or manifest["images"] != len(expected_names)
        or manifest["profile"] != _PREPARATION_PROFILE
        or manifest["scene"] != scene
        or manifest["source_archives"] != expected_archives
    ):
        return False
    images = scene_dir / "images"
    if not images.is_dir():
        return False
    actual_names = {path.relative_to(images).as_posix() for path in images.rglob("*.jpg") if path.is_file()}
    if actual_names != expected_names:
        return False
    sample = LAMAR_LAYOUT.testset_path(scene, "sample")
    return sample.is_file() and _sha256(sample) == _SAMPLE_SHA256[scene]


def _pending_scenes(scenes: tuple[str, ...]) -> tuple[str, ...]:
    pending = []
    for scene in scenes:
        destination = LAMAR_LAYOUT.scene_dir(scene)
        if destination.exists():
            if not lamar_scene_is_prepared(destination):
                raise RuntimeError(f"LaMAR scene destination is partial or incompatible: {destination}")
            continue
        pending.append(scene)
    return tuple(pending)


def main(
    scenes: Iterable[str] | None = None,
    *,
    download: bool = True,
    delete_files: bool = False,
) -> None:
    """Download official iOS sessions and prepare selected or all LaMAR scenes."""
    selected = LAMAR_LAYOUT.select_scenes(scenes)
    _ensure_testsets(selected)
    pending = _pending_scenes(selected)
    if not pending:
        return
    inventory = source_inventory()
    downloads = LAMAR_LAYOUT.data_dir / "downloads"
    downloads.mkdir(parents=True, exist_ok=True)
    ground_truth_models = _ground_truth_models_archive(downloads, download=download)
    for scene in pending:
        archive_paths = []
        for source in inventory[scene]:
            path = downloads / scene / source.filename
            if download:
                resumable_http_download(
                    source.url,
                    path,
                    expected_size=source.size,
                    expected_digest=source.sha256,
                )
            elif not path.is_file() or path.stat().st_size != source.size or _sha256(path) != source.sha256:
                raise FileNotFoundError(f"Missing checksum-valid LaMAR archive: {path}")
            archive_paths.append(path)
        destination = LAMAR_LAYOUT.scene_dir(scene)
        with prep_fs.scene_staging(destination) as staging:
            _convert_scene(scene, tuple(archive_paths), staging, ground_truth_models)
            if destination.exists():
                raise RuntimeError(f"LaMAR scene destination appeared during preparation: {destination}")
            staging.rename(destination)
        if not lamar_scene_is_prepared(destination):
            raise RuntimeError(f"LaMAR preparation did not produce a complete scene: {scene}")
        if delete_files:
            for path in archive_paths:
                path.unlink()
            scene_downloads = downloads / scene
            if not any(scene_downloads.iterdir()):
                scene_downloads.rmdir()
    if delete_files:
        ground_truth_models.unlink()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenes", nargs="+", choices=LAMAR_LAYOUT.scenes)
    parser.add_argument(
        "--no-download",
        action="store_true",
        help="Use checksum-valid archives already in downloads/",
    )
    parser.add_argument(
        "--delete-downloads",
        action="store_true",
        help="Delete validated source archives after successful preparation",
    )
    arguments = parser.parse_args()
    main(
        arguments.scenes,
        download=not arguments.no_download,
        delete_files=arguments.delete_downloads,
    )
