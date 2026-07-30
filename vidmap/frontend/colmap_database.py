from __future__ import annotations

import logging
import multiprocessing
import signal
import sys
import tempfile
from collections.abc import Mapping, Sequence
from contextlib import contextmanager, redirect_stdout
from io import StringIO
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np
import pycolmap
from tqdm import tqdm

from vidmap.frontend.cache import CacheMetadataMismatch, read_pair_artifact, validate_incremental_cache
from vidmap.frontend.correspondences import ImagePair
from vidmap.utils.io import get_keypoints, get_matches
from vidmap.utils.logging import progress_bars_enabled

if TYPE_CHECKING:
    from vidmap.frontend.pipeline import TrackingFrontendResult

CameraPolicy = Literal["transitive", "per_image", "shared"]
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


def build_colmap_database(
    database_path: Path,
    reconstruction,
    image_names: Sequence[str],
    sparse_features_path: Path,
    pairs: Sequence[ImagePair],
    *,
    camera_policy: CameraPolicy,
    prior_focal_length: bool,
    sparse_matches_path: Path | None = None,
    matches: Mapping[ImagePair, object] | None = None,
    seed_two_view_geometry: bool = False,
) -> dict[str, int]:
    """Create and populate one frontend-owned COLMAP database."""
    if camera_policy not in ("transitive", "per_image", "shared"):
        raise ValueError(f"Unsupported camera policy: {camera_policy}")
    if (sparse_matches_path is None) == (matches is None):
        raise ValueError("Exactly one match source is required")
    image_names = tuple(image_names)
    if any(not isinstance(name, str) or not name for name in image_names):
        raise ValueError("Database image names must be non-empty strings")
    if len(set(image_names)) != len(image_names):
        raise ValueError("Database image names must be unique")

    database_path.unlink(missing_ok=True)
    build_path = database_path.with_name(f".{database_path.name}.building")
    build_path.unlink(missing_ok=True)
    db = pycolmap.Database.open(build_path)
    base_succeeded = False
    try:
        with pycolmap.DatabaseTransaction(db):
            selected_names = set(image_names)
            image_ids: dict[str, int] = {}
            shared_camera_added = False
            shared_camera_signature = None
            transitive_camera_id = 1

            for reconstruction_index, (image_id, image) in enumerate(
                sorted(reconstruction.images.items(), key=lambda item: item[1].name)
            ):
                imname = image.name
                if imname not in selected_names:
                    continue
                camera = reconstruction.cameras[image.camera_id]

                image_ids[imname] = image_id

                write_camera = True
                if camera_policy == "shared":
                    database_camera_id = 0
                    signature = (
                        camera.model,
                        camera.width,
                        camera.height,
                        tuple(np.asarray(camera.params).tolist()),
                    )
                    if shared_camera_signature is None:
                        shared_camera_signature = signature
                    elif signature != shared_camera_signature:
                        raise ValueError(
                            "Shared-camera database policy requires identical camera models, dimensions, and parameters"
                        )
                    if shared_camera_added:
                        write_camera = False
                    else:
                        shared_camera_added = True
                else:
                    if camera_policy == "transitive":
                        database_camera_id = transitive_camera_id
                        transitive_camera_id += 1
                    else:
                        database_camera_id = reconstruction_index

                if write_camera:
                    db.write_camera(
                        pycolmap.Camera(
                            camera_id=database_camera_id,
                            model=camera.model,
                            width=camera.width,
                            height=camera.height,
                            params=camera.params,
                            has_prior_focal_length=prior_focal_length,
                        ),
                        use_camera_id=True,
                    )
                db.write_image(
                    pycolmap.Image(name=imname, camera_id=database_camera_id, image_id=image_id),
                    use_image_id=True,
                )

            missing_names = sorted(selected_names - set(image_ids))
            if missing_names:
                raise ValueError(f"Database reconstruction is missing requested image {missing_names[0]!r}")
        base_succeeded = True
    finally:
        try:
            db.close()
        finally:
            if not base_succeeded:
                build_path.unlink(missing_ok=True)

    imports_succeeded = False
    try:
        import_features(image_ids, build_path, sparse_features_path)
        _import_matches(
            image_ids,
            build_path,
            pairs,
            matches_path=sparse_matches_path,
            matches=matches,
            seed_two_view_geometry=seed_two_view_geometry,
        )
        imports_succeeded = True
    finally:
        if not imports_succeeded:
            build_path.unlink(missing_ok=True)
    build_path.replace(database_path)
    return image_ids


def create_database_from_frontend(
    frontend_result: TrackingFrontendResult,
    initial_reconstruction: pycolmap.Reconstruction,
    database_path: Path,
    *,
    view_graph_calibration: bool,
    matches: dict[ImagePair, object] | None = None,
) -> dict[str, int]:
    """Build a COLMAP database from a validated frontend result."""
    paths = frontend_result.paths
    artifacts = frontend_result.artifacts
    pairs = read_pair_artifact(paths.track_pairs_path, artifacts.track_pairs.metadata)
    if tuple(pairs) != tuple(frontend_result.track_pairs):
        raise CacheMetadataMismatch("Track-pair cache changed after frontend")
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

    import_pairs = (
        tuple(frontend_result.track_pairs)
        if matches is None
        else tuple(sorted(matches, key=lambda pair: (str(pair[0]), str(pair[1]))))
    )
    return build_colmap_database(
        database_path,
        initial_reconstruction,
        frontend_result.keyframe_sequence,
        paths.sparse_features_path,
        import_pairs,
        camera_policy="shared" if view_graph_calibration else "per_image",
        prior_focal_length=not view_graph_calibration,
        sparse_matches_path=paths.sparse_matches_path if matches is None else None,
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
