from __future__ import annotations

import logging
import multiprocessing
import signal
import sys
import tempfile
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager, redirect_stdout
from io import StringIO
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pycolmap
from tqdm import tqdm

from vidmap.datasets.capture import copy_unposed_reconstruction
from vidmap.frontend.cache import validate_incremental_cache
from vidmap.frontend.correspondences import ImagePair
from vidmap.mapper.inputs.snapshot import require_finalized_sqlite
from vidmap.utils.io import get_keypoints, get_matches
from vidmap.utils.logging import progress_bars_enabled

if TYPE_CHECKING:
    from vidmap.frontend.pipeline import TrackingFrontendResult

logger = logging.getLogger(__name__)


def _import_matches(
    image_ids: dict[str, int],
    database_path: Path,
    pairs: Sequence[ImagePair],
    *,
    matches_path: Path | None,
    matches: Mapping[ImagePair, object] | None,
    seed_two_view_geometry: bool,
) -> None:
    """Import the selected frontend-owned matches into one COLMAP database."""
    logger.info("Importing matches into %s", database_path)
    database = pycolmap.Database.open(database_path)
    try:
        with pycolmap.DatabaseTransaction(database):
            imported = set()
            count = 0
            for name0, name1 in tqdm(pairs, disable=not progress_bars_enabled()):
                if name0 not in image_ids or name1 not in image_ids:
                    continue
                id0, id1 = image_ids[name0], image_ids[name1]
                if (id0, id1) in imported or (id1, id0) in imported:
                    continue

                if matches_path is not None:
                    matches_data, _scores = get_matches(matches_path, name0, name1)
                elif matches is not None:
                    pair = (name0, name1)
                    reverse = (name1, name0)
                    if pair in matches:
                        matches_data = matches[pair]
                    elif reverse in matches:
                        matches_data = matches[reverse][:, [1, 0]]
                    else:
                        continue
                else:
                    raise ValueError("Exactly one match source is required")

                matches_array = np.asarray(matches_data, dtype=np.uint32)
                database.write_matches(id0, id1, matches_array)
                imported.update({(id0, id1), (id1, id0)})
                if seed_two_view_geometry:
                    database.write_two_view_geometry(
                        id0,
                        id1,
                        pycolmap.TwoViewGeometry(
                            config=pycolmap.TwoViewGeometryConfiguration.CALIBRATED,
                            E=None,
                            F=None,
                            H=None,
                            cam2_from_cam1=None,
                            inlier_matches=matches_array,
                        ),
                    )
                count += 1
    finally:
        database.close()
    logger.info("Added %d matches to %s", count, database_path)


def import_features(image_ids: Mapping[str, int], database_path: Path, features_path: Path) -> None:
    """Import cached keypoints while honoring VidMap's progress policy."""
    logger.info("Importing features into %s", database_path)
    db = pycolmap.Database.open(database_path)
    try:
        with pycolmap.DatabaseTransaction(db):
            for image_name, image_id in tqdm(image_ids.items(), disable=not progress_bars_enabled()):
                keypoints = get_keypoints(features_path, image_name)
                db.write_keypoints(image_id, np.asarray(keypoints + 0.5, dtype=np.float32))
    finally:
        db.close()


def _transitive_reconstruction(reconstruction, image_names):
    """Give each diagnostic image a fresh singleton camera, in name order."""
    names = set(image_names)
    images = sorted(
        (image for image in reconstruction.images.values() if image.name in names), key=lambda image: image.name
    )
    if {image.name for image in images} != names:
        raise ValueError("Database reconstruction is missing requested images")
    selected = pycolmap.Reconstruction()
    for camera_id, image in enumerate(images, start=1):
        camera = pycolmap.Camera(reconstruction.camera(image.camera_id).todict())
        camera.camera_id = camera_id
        selected.add_camera_with_trivial_rig(camera)
        selected.add_image_with_trivial_frame(
            pycolmap.Image(name=image.name, camera_id=camera_id, image_id=image.image_id)
        )
    return selected


def build_colmap_database(
    database_path: Path,
    reconstruction: pycolmap.Reconstruction,
    image_names: Sequence[str],
    sparse_features_path: Path,
    pairs: Sequence[ImagePair],
    *,
    prior_focal_length: bool,
    diagnostic_transitive: bool = False,
    sparse_matches_path: Path | None = None,
    matches: Mapping[ImagePair, object] | None = None,
    seed_two_view_geometry: bool = False,
) -> pycolmap.Reconstruction:
    """Serialize prepared topology, or explicitly flatten a diagnostic database."""
    if (sparse_matches_path is None) == (matches is None):
        raise ValueError("Exactly one match source is required")
    image_names = tuple(image_names)
    if any(not isinstance(name, str) or not name for name in image_names):
        raise ValueError("Database image names must be non-empty strings")
    if len(set(image_names)) != len(image_names):
        raise ValueError("Database image names must be unique")

    require_finalized_sqlite(database_path)
    build_path = database_path.with_name(f".{database_path.name}.building")
    build_path.unlink(missing_ok=True)
    logger.info("Building COLMAP database at %s", build_path)
    try:
        with pycolmap.Database.open(build_path) as db:
            with pycolmap.DatabaseTransaction(db):
                if diagnostic_transitive:
                    selected = _transitive_reconstruction(reconstruction, image_names)
                else:
                    selected = copy_unposed_reconstruction(reconstruction, image_names)
                for camera in selected.cameras.values():
                    camera.has_prior_focal_length = prior_focal_length
                images = selected.images.values()
                if diagnostic_transitive:
                    images = sorted(images, key=lambda image: image.name)
                image_ids = {image.name: image.image_id for image in images}
                camera_ids = dict.fromkeys(selected.image(image_id).camera_id for image_id in image_ids.values())
                for camera_id in camera_ids:
                    db.write_camera(selected.camera(camera_id), use_camera_id=True)
                for rig in selected.rigs.values():
                    db.write_rig(rig, use_rig_id=True)
                for frame in selected.frames.values():
                    db.write_frame(frame, use_frame_id=True)
                for image_id in image_ids.values():
                    db.write_image(selected.image(image_id), use_image_id=True)
        import_features(image_ids, build_path, sparse_features_path)
        _import_matches(
            image_ids,
            build_path,
            pairs,
            matches_path=sparse_matches_path,
            matches=matches,
            seed_two_view_geometry=seed_two_view_geometry,
        )
        build_path.replace(database_path)
    finally:
        build_path.unlink(missing_ok=True)
    return selected


def create_database_from_frontend(
    frontend_result: TrackingFrontendResult,
    initial_reconstruction: pycolmap.Reconstruction,
    database_path: Path,
    *,
    estimate_intrinsics: bool,
    matches: Mapping[ImagePair, object],
) -> pycolmap.Reconstruction:
    """Build a COLMAP database from a validated frontend result."""
    paths = frontend_result.paths
    artifacts = frontend_result.artifacts
    validate_incremental_cache(
        paths.sparse_features_path,
        artifacts.sparse_features.metadata,
        artifacts.sparse_features.expected_items,
    )
    validate_incremental_cache(
        paths.sparse_matches_path,
        artifacts.sparse_matches.metadata,
        artifacts.sparse_matches.expected_items,
    )
    return build_colmap_database(
        database_path,
        initial_reconstruction,
        frontend_result.keyframe_sequence,
        paths.sparse_features_path,
        tuple(sorted(matches, key=lambda pair: (str(pair[0]), str(pair[1])))),
        prior_focal_length=not estimate_intrinsics,
        matches=matches,
    )


@contextmanager
def _verification_output(verbose: bool):
    output = StringIO()
    try:
        if verbose:
            yield
        else:
            with redirect_stdout(output):
                yield
    finally:
        if not verbose and sys.exc_info()[0] is not None:
            logger.error("Geometric verification failed with output:\n%s", output.getvalue())
        sys.stdout.flush()


def _verify_matches_process(database_path: Path, pairs_path: Path, verbose: bool, options: dict) -> None:
    signal.pthread_sigmask(signal.SIG_UNBLOCK, {signal.SIGINT})
    logger.info("Geometric verification options: %s", options)
    with _verification_output(verbose):
        with pycolmap.ostream():
            pycolmap.verify_matches(database_path, pairs_path, options=options)


def _write_pair_list(pairs: Sequence[ImagePair], path: Path) -> None:
    path.write_text("\n".join(f"{name0} {name1}" for name0, name1 in pairs))


def run_geometric_verification(
    database_path: Path,
    pairs: list[ImagePair],
    verbose: bool,
    options: dict,
) -> None:
    """Run one isolated geometric-verification attempt and fail contextually."""
    logger.info("Geometric verification of %s with %d pairs", database_path, len(pairs))
    with pycolmap.Database.open(database_path) as db:
        cameras = {camera.camera_id: camera for camera in db.read_all_cameras()}
        if any(camera.model_name not in ("PINHOLE", "SIMPLE_PINHOLE") for camera in cameras.values()):
            from vidmap.utils.camera_geometry import estimate_distorted_two_view

            images = {image.name: image for image in db.read_all_images()}
            geometry_options = pycolmap.TwoViewGeometryOptions(options)
            geometry_options.compute_relative_pose = True
            with ThreadPoolExecutor() as executor, pycolmap.DatabaseTransaction(db):
                for start in tqdm(range(0, len(pairs), 32), disable=not verbose):
                    pending = []
                    for name1, name2 in pairs[start : start + 32]:
                        image1, image2 = images[name1], images[name2]
                        future = executor.submit(
                            estimate_distorted_two_view,
                            cameras[image1.camera_id],
                            db.read_keypoints(image1.image_id)[:, :2],
                            cameras[image2.camera_id],
                            db.read_keypoints(image2.image_id)[:, :2],
                            db.read_matches(image1.image_id, image2.image_id),
                            geometry_options,
                        )
                        pending.append((image1.image_id, image2.image_id, future))
                    for first, second, future in pending:
                        db.write_two_view_geometry(first, second, future.result())
            return
    with tempfile.TemporaryDirectory() as tmpdir:
        pairs_path = Path(tmpdir) / "pairs.txt"
        _write_pair_list(pairs, pairs_path)
        process = multiprocessing.Process(
            target=_verify_matches_process,
            args=(database_path, pairs_path, verbose, options),
        )
        try:
            previous_signals = signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGINT})
            try:
                process.start()
            finally:
                signal.pthread_sigmask(signal.SIG_SETMASK, previous_signals)
            process.join()
        except BaseException:
            if process.pid is not None and process.is_alive():
                process.terminate()
                process.join(timeout=10)
            if process.pid is not None and process.is_alive():
                process.kill()
                process.join()
            raise
        if process.exitcode != 0:
            raise RuntimeError(
                "Geometric verification failed: "
                f"database={database_path}, pairs={len(pairs)}, attempt=1, exit_code={process.exitcode}"
            )
