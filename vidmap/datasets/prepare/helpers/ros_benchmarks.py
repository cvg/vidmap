"""Shared download and preparation flow for Spires and Hilti."""

import argparse
import bisect
import json
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from decimal import Decimal
from functools import partial
from pathlib import Path

import numpy as np
import pycolmap
import yaml

from vidmap.datasets.layouts import get_dataset_layout
from vidmap.datasets.names import get_dataset_definition
from vidmap.datasets.prepare.helpers.filesystem import scene_staging
from vidmap.datasets.prepare.helpers.video_common import checksum, resumable_http_download
from vidmap.datasets.resources import dataset_asset

PREPARATION_VERSIONS = {"spires": 1, "hilti": 1}


def source_inventory(dataset):
    return json.loads(dataset_asset(dataset, "sources.json").read_text())


def modes(dataset):
    """Map benchmark names to an image variant and the number of streams starting at cam0."""
    suite = json.loads(dataset_asset("benchmark_suites.json").read_text())[dataset]
    return suite["modes"]


def timestamp_ns(value):
    nanoseconds = Decimal(value) * 1_000_000_000
    if nanoseconds != nanoseconds.to_integral_value():
        raise ValueError(f"Timestamp is not an integer number of nanoseconds: {value}")
    return int(nanoseconds)


def read_poses(path):
    """Read TUM world-from-sensor poses."""
    timestamps, poses = [], []
    for line in Path(path).read_text().splitlines():
        fields = line.split()
        if not fields or fields[0].startswith("#"):
            continue
        if len(fields) != 8:
            raise ValueError(f"Expected timestamp xyz quaternion in {path}")
        stamp = timestamp_ns(fields[0])
        values = np.asarray(fields[1:], dtype=float)
        if not np.isfinite(values).all() or not np.isclose(np.linalg.norm(values[3:]), 1, atol=1e-3):
            raise ValueError(f"Invalid GT pose in {path}")
        if timestamps and stamp <= timestamps[-1]:
            raise ValueError(f"GT timestamps must strictly increase: {path}")
        timestamps.append(stamp)
        poses.append(pycolmap.Rigid3d(pycolmap.Rotation3d(values[3:]), values[:3]))
    if len(timestamps) < 3:
        raise ValueError(f"Expected at least three GT poses: {path}")
    return timestamps, poses


def nearest_pose(timestamps, poses, stamp, max_offset_ns):
    index = bisect.bisect_left(timestamps, stamp)
    candidates = range(max(0, index - 1), min(len(timestamps), index + 1))
    nearest = min(candidates, key=lambda i: (abs(timestamps[i] - stamp), timestamps[i]))
    return poses[nearest] if abs(timestamps[nearest] - stamp) <= max_offset_ns else None


def write_ground_truth(
    root, dataset, recording, variants, gt_path, camera_from_source, max_offset_ns, min_timestamp_ns=0
):
    """Write calibrated models per variant, preserving benchmark image IDs."""
    from vidmap.datasets.local import select_images
    from vidmap.datasets.prepare.helpers.rig_capture import load_capture
    from vidmap.frontend.options.rig import RigOptions

    # Panorama intrinsics remain placeholders; the runtime frontend calibrates its rendered views.
    times, poses = read_poses(gt_path)
    captures = {
        name: load_capture(root / name / "images", None if spec["rig"] is None else RigOptions(**spec["rig"]))
        for name, spec in variants.items()
    }
    image_ids, testsets = {}, {}
    # Assign the existing benchmark IDs before adding other prepared stream images.
    for mode, selection in modes(dataset).items():
        variant, num_cameras = selection["variant"], selection["num_cameras"]
        if variant not in captures or len(captures[variant].rec.cameras) < num_cameras:
            continue
        capture = captures[variant]
        names = (
            [image.name for _, image in sorted(capture.rec.images.items()) if image.camera_id < num_cameras]
            if num_cameras > 1
            else select_images(capture.rgb_dir, None, camera_dir="cam0")
        )
        for name in names:
            if name not in image_ids:
                image_ids[name] = len(image_ids) + 1
        testsets[mode] = dict(recording=recording, variant=variant, image_ids=[image_ids[name] for name in names])

    for variant, capture in captures.items():
        reconstruction, posed_ids = pycolmap.Reconstruction(), set()
        for camera_id, camera in sorted(capture.rec.cameras.items()):
            reconstruction.add_camera_with_trivial_rig(camera)
            for name in select_images(capture.rgb_dir, None, camera_dir=f"cam{camera_id}"):
                if name not in image_ids:
                    image_ids[name] = len(image_ids) + 1
                image_id = image_ids[name]
                stamp = timestamp_ns(Path(name).stem)
                source_pose = (
                    nearest_pose(times, poses, stamp, max_offset_ns)
                    if camera_id == 0 and stamp >= min_timestamp_ns
                    else None
                )
                pose = (
                    camera_from_source * source_pose.inverse()
                    if source_pose is not None
                    else pycolmap.Rigid3d(pycolmap.Rotation3d([0.0, 0.0, 0.0, -1.0]), np.zeros(3))
                )
                image = pycolmap.Image(image_id=image_id, name=name, camera_id=camera_id)
                reconstruction.add_image_with_trivial_frame(image, pose)
                if source_pose is not None:
                    posed_ids.add(image_id)
        for mode, selection in testsets.items():
            if selection["variant"] == variant and len(posed_ids.intersection(selection["image_ids"])) < 3:
                raise ValueError(f"Fewer than three reference-camera images have supported GT: {mode}")
        destination = root / variant / "rec"
        destination.mkdir()
        reconstruction.write(destination)
    return testsets


def _download(rows, root, download):
    paths = []
    for row in rows:
        path = root / row["path"]
        if not download and not path.is_file():
            raise FileNotFoundError(f"Missing source file: {path}")
        paths.append(
            resumable_http_download(row["url"], path, expected_size=row["size"], expected_digest=row["sha256"])
        )
    return paths


def _install_testsets(layout, scene, destination):
    testsets = json.loads((destination / "preparation.json").read_text())["testsets"]
    for mode in testsets:
        contents = yaml.safe_dump({mode: testsets[mode]}).encode()
        path = layout.testset_path(scene, mode)
        if path.exists() and path.read_bytes() != contents:
            raise ValueError(f"Existing testset differs from prepared scene: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".yaml.part")
        temporary.write_bytes(contents)
        temporary.replace(path)


def scene_is_prepared(layout, scene, *, mode=None):
    from vidmap.frontend.options.rig import RigOptions

    definitions = modes(layout.name)
    if mode is not None and mode not in definitions:
        raise ValueError(f"Unknown {layout.name} input mode: {mode}")
    destination = layout.recording_dir(scene, "0")
    receipt = destination / "preparation.json"
    if not receipt.is_file():
        return False
    manifest = json.loads(receipt.read_text())
    if manifest["sources"] != source_inventory(layout.name)["shared"] + source_inventory(layout.name)["scenes"][scene]:
        return False
    if manifest["schema_version"] != PREPARATION_VERSIONS[layout.name]:
        return False
    files = manifest["files"]
    if mode is not None:
        if mode not in manifest["testsets"]:
            return False
        definition = definitions[mode]
        variant = definition["variant"]
        if manifest["testsets"][mode]["variant"] != variant:
            return False
        image_root = Path(variant) / "images"
        source = manifest["variants"][variant]
        required = set()
        directories = tuple(f"cam{cid}" for cid in range(definition["num_cameras"]))
        if definition["num_cameras"] > 1:
            options = RigOptions(**source["rig"])
            if options.frames is not None:
                required.add((image_root / options.frames).as_posix())
        prefixes = (f"{variant}/rec/", *(f"{image_root}/{directory}/" for directory in directories))
        files = {name: size for name, size in files.items() if name in required or name.startswith(prefixes)}
    for name, expected in files.items():
        path = destination / name
        if not path.is_file() or path.stat().st_size != expected:
            return False
    for name, digest in manifest["metadata_sha256"].items():
        if name not in files:
            continue
        path = destination / name
        if not path.is_file() or checksum(path, "sha256") != digest:
            return False
    return True


def _delete_downloads(rows, root):
    for row in rows:
        (root / row["path"]).unlink(missing_ok=True)


def _prepare_scene(scene, *, layout, inventory, shared, mode, download, delete_files):
    dataset = layout.name
    downloads = layout.data_dir / "downloads"
    destination = layout.recording_dir(scene, "0")
    sources = _download(inventory["scenes"][scene], downloads, download)
    with scene_staging(destination, replace=True) as staging:
        output = staging / "scene"
        if dataset == "spires":
            from vidmap.datasets.prepare.spires import prepare

            calibration = next(path for path in shared if path.name == "cam-lidar.yaml")
            prepare(
                [path for path in sources if path.suffix == ".bag"],
                calibration.parent,
                output,
                mode=mode,
                previous=destination if destination.exists() else None,
            )
            transform = np.asarray(yaml.safe_load(calibration.read_text())["cam0"]["T_cam_lidar"])
            # Spires GT reports base_link poses; calibration is relative to the LiDAR.
            base_from_lidar = np.diag([-1.0, -1.0, 1.0, 1.0])
            base_from_lidar[:3, 3] = [0, 0, 0.124]
            camera_from_source = pycolmap.Rigid3d((transform @ np.linalg.inv(base_from_lidar))[:3])
            # Upstream Spires COLMAP evaluation uses evo --t_max_diff 0.01.
            max_offset_ns = 10_000_000
            min_timestamp_ns = 0
        else:
            from vidmap.datasets.prepare.helpers.hilti_stitching import PANORAMA_FROM_CAM0, prepare

            bag = next(path for path in sources if path.suffix == ".db3")
            prepare(
                bag,
                next(path for path in shared if path.suffix == ".yaml"),
                sorted(path for path in shared if path.suffix == ".png"),
                output,
            )
            camera_from_source = pycolmap.Rigid3d(pycolmap.Rotation3d(PANORAMA_FROM_CAM0), np.zeros(3))
            max_offset_ns = 20_000_000
            min_timestamp_ns = 10_005_000_000_000
        gt_path = next(path for path in sources if path.suffix == ".txt")
        preparation = json.loads((output / "preparation.json").read_text())
        testsets = write_ground_truth(
            output,
            dataset,
            "0",
            preparation["variants"],
            gt_path,
            camera_from_source,
            max_offset_ns,
            min_timestamp_ns,
        )
        files = {
            path.relative_to(output).as_posix(): path.stat().st_size
            for path in output.rglob("*")
            if path.is_file() and path != output / "preparation.json"
        }
        hashes = {
            name: checksum(output / name, "sha256") for name in files if Path(name).suffix not in {".jpg", ".png"}
        }
        preparation.update(
            schema_version=PREPARATION_VERSIONS[dataset],
            sources=inventory["shared"] + inventory["scenes"][scene],
            testsets=testsets,
            max_gt_offset_ns=max_offset_ns,
            min_gt_timestamp_ns=min_timestamp_ns,
            files=files,
            metadata_sha256=hashes,
        )
        (output / "preparation.json").write_text(json.dumps(preparation, indent=2) + "\n")
        if destination.exists():
            destination.rename(staging / "previous")
        output.rename(destination)
    _install_testsets(layout, scene, destination)
    if not scene_is_prepared(layout, scene):
        raise RuntimeError(f"Preparation failed validation: {scene}")
    if delete_files:
        _delete_downloads(inventory["scenes"][scene], downloads)


def _initialize_worker():
    import cv2

    cv2.setNumThreads(1)


def main(dataset, scenes=None, *, mode=None, download=True, delete_files=False, workers=1):
    if workers < 1:
        raise ValueError("workers must be positive")
    mode = get_dataset_definition(dataset).default_mode if mode is None else mode
    if mode != "all" and mode not in modes(dataset):
        raise ValueError(f"Unknown {dataset} input mode: {mode}")
    requested = modes(dataset) if mode == "all" else (mode,)
    layout = get_dataset_layout(dataset)
    inventory = source_inventory(dataset)
    selected = layout.select_scenes(scenes)
    downloads = layout.data_dir / "downloads"
    pending = []
    for scene in selected:
        destination = layout.recording_dir(scene, "0")
        prepared = ()
        if destination.exists():
            if not scene_is_prepared(layout, scene):
                raise RuntimeError(f"Partial or incompatible preparation: {destination}")
            prepared = json.loads((destination / "preparation.json").read_text())["testsets"]
        if all(item in prepared for item in requested):
            _install_testsets(layout, scene, destination)
            if delete_files:
                _delete_downloads(inventory["scenes"][scene], downloads)
        else:
            pending.append(scene)
    if pending:
        shared = _download(inventory["shared"], downloads, download)
        prepare_scene = partial(
            _prepare_scene,
            layout=layout,
            inventory=inventory,
            shared=shared,
            mode=mode,
            download=download,
            delete_files=delete_files,
        )
        if workers == 1 or len(pending) == 1:
            for scene in pending:
                prepare_scene(scene)
        else:
            with ProcessPoolExecutor(
                max_workers=min(workers, len(pending)),
                mp_context=multiprocessing.get_context("spawn"),
                initializer=_initialize_worker,
            ) as executor:
                futures = [executor.submit(prepare_scene, scene) for scene in pending]
                try:
                    for future in as_completed(futures):
                        future.result()
                finally:
                    for future in futures:
                        future.cancel()
    if delete_files and selected:
        _delete_downloads(inventory["shared"], downloads)


def run_cli(dataset):
    parser = argparse.ArgumentParser(description=f"Download and prepare the {dataset} benchmark.")
    parser.add_argument("--scenes", nargs="+", choices=get_dataset_layout(dataset).scenes)
    parser.add_argument(
        "--mode", choices=(*modes(dataset), "all"), default=get_dataset_definition(dataset).default_mode
    )
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--delete-downloads", action="store_true")
    parser.add_argument(
        "--workers", type=int, default=1, help="Number of recordings to prepare concurrently (default: 1)"
    )
    args = parser.parse_args()
    main(
        dataset,
        args.scenes,
        mode=args.mode,
        download=not args.no_download,
        delete_files=args.delete_downloads,
        workers=args.workers,
    )
