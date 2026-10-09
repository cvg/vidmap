"""Byte-preserving copy and decoding of the finalized mapper database."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

import numpy as np
import pycolmap

from vidmap.mapper.native.extension import native
from vidmap.mapper.native.state import SolveState

logger = logging.getLogger(__name__)


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
        rec = pycolmap.Reconstruction()
        sidecars = native.MappingSidecars()
        pose_graph = pycolmap.PoseGraph()

        for cam in database.read_all_cameras():
            rec.add_camera_with_trivial_rig(cam)

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
            rec.add_image_with_trivial_frame(gimg, pycolmap.Rigid3d(translation=np.full(3, np.nan)))
            sidecars.add_image(image_id, native.ImageData())
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

            if img1_id == img2_id:
                raise ValueError("An image pair must contain two distinct images")
            rec.image(img1_id)
            rec.image(img2_id)
            data = native.PairData()
            data.geometry = pycolmap.TwoViewGeometry(config=cfg, F=two_view.F, H=two_view.H)
            data.all_matches = np.asarray(feat_matches, dtype=np.uint32).reshape((-1, 2))
            sidecars.add_pair(int(pair_id), data)
            edge = pycolmap.PoseGraphEdge()
            edge.num_matches = len(feat_matches)
            pose_graph.add_edge(img1_id, img2_id, edge)

        logger.info("Loaded %d image pairs; %d are invalid", total_pairs, invalid_count)
        return SolveState(
            rec,
            pose_graph,
            sidecars,
            image_order=list(rec.images.keys()),
            pair_order=sorted(pose_graph.edges),
        )
    finally:
        database.close()
