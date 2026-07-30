"""Byte-preserving copy and decoding of the finalized mapper database."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

import numpy as np
import pycolmap

from vidmap.mapper.native.extension import native
from vidmap.mapper.native.records import (
    camera_record_from_pycolmap,
    image_record_from_pycolmap,
    pair_record_from_pycolmap,
)
from vidmap.mapper.native.state import SolveState

logger = logging.getLogger(__name__)


def _copy_optional_fundamental(target: pycolmap.TwoViewGeometry, value: np.ndarray | None) -> None:
    if value is not None:
        target.F = np.asarray(value, dtype=float)


def remove_database_sidecars(database_path: Path) -> None:
    for suffix in ("-wal", "-shm", "-journal"):
        Path(f"{database_path}{suffix}").unlink(missing_ok=True)


def copy_finalized_database(source: Path, destination: Path) -> None:
    """Copy database bytes after clearing stale SQLite sidecars."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    remove_database_sidecars(destination)
    shutil.copy2(source, destination)
    remove_database_sidecars(destination)


def load_finalized_database(
    database_path: Path,
) -> SolveState:
    """Load cameras, images, keypoints, and verified matches from a COLMAP database."""
    if not database_path.is_file():
        raise FileNotFoundError(f"Finalized mapper database not found: {database_path}")
    database = pycolmap.Database.open(str(database_path))
    try:
        correspondence_graph = pycolmap.CorrespondenceGraph()
        rec = pycolmap.Reconstruction()
        problem = native.MappingProblem()

        for cam in database.read_all_cameras():
            rec.add_camera_with_trivial_rig(cam)
            problem.add_camera(camera_record_from_pycolmap(cam))

        images_colmap = database.read_all_images()
        logger.info("Loading %d images from the finalized database", len(images_colmap))
        for img in images_colmap:
            image_id = img.image_id
            kps = database.read_keypoints(image_id)
            features = kps if len(kps) > 0 else np.empty((0, 2), dtype=float)
            gimg = pycolmap.Image(
                name=img.name,
                camera_id=img.camera_id,
                image_id=image_id,
                keypoints=features,
            )
            rec.add_image_with_trivial_frame(gimg, pycolmap.Rigid3d())
            correspondence_graph.add_image(image_id, len(features))
            problem.add_image(image_record_from_pycolmap(rec.image(image_id), features))
        pair_ids, matches = database.read_all_matches()
        total_pairs = len(pair_ids)
        invalid_count = 0
        for pair_id, feat_matches in zip(pair_ids, matches):
            img1_id, img2_id = pycolmap.pair_id_to_image_pair(pair_id)
            two_view = database.read_two_view_geometry(img1_id, img2_id)
            cfg = two_view.config
            if cfg in (
                pycolmap.TwoViewGeometryConfiguration.UNDEFINED,
                pycolmap.TwoViewGeometryConfiguration.DEGENERATE,
                pycolmap.TwoViewGeometryConfiguration.WATERMARK,
                pycolmap.TwoViewGeometryConfiguration.MULTIPLE,
            ):
                invalid_count += 1
                continue

            tvg = pycolmap.TwoViewGeometry()
            tvg.config = cfg
            if cfg in (
                pycolmap.TwoViewGeometryConfiguration.UNCALIBRATED,
                pycolmap.TwoViewGeometryConfiguration.CALIBRATED,
            ):
                tvg.F = np.asarray(two_view.F, dtype=float)
            elif cfg in (
                pycolmap.TwoViewGeometryConfiguration.PLANAR,
                pycolmap.TwoViewGeometryConfiguration.PANORAMIC,
                pycolmap.TwoViewGeometryConfiguration.PLANAR_OR_PANORAMIC,
            ):
                tvg.H = np.asarray(two_view.H, dtype=float)
                # Homography-only rows have no fundamental matrix. Some
                # pycolmap versions expose that absence as None and reject the
                # scalar NaN that np.asarray(None) would pass to the 3x3 setter.
                _copy_optional_fundamental(tvg, two_view.F)

            correspondence_graph.add_two_view_geometry(img1_id, img2_id, tvg)
            problem.add_pair(
                pair_record_from_pycolmap(
                    int(pair_id),
                    img1_id,
                    img2_id,
                    tvg,
                    feat_matches if len(feat_matches) > 0 else np.empty((0, 2), dtype=np.uint32),
                )
            )

        logger.info("Loaded %d image pairs; %d are invalid", total_pairs, invalid_count)
        return SolveState(
            rec,
            problem,
            image_order=list(rec.images.keys()),
            pair_order=correspondence_graph.image_pairs(),
        )
    finally:
        database.close()
