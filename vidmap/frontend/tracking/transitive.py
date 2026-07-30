"""Build tracking-owned transitive correspondences from sparse tracks."""

import logging
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass as result_dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pycolmap
from pycolmap import Database, DatabaseCache

from vidmap.datasets.base import DatasetParser
from vidmap.frontend.colmap_database import build_colmap_database
from vidmap.frontend.correspondences import ImagePair, immutable_array_mapping
from vidmap.frontend.initial_reconstruction import build_initial_reconstruction
from vidmap.frontend.keyframes.processing import KeyframePlan
from vidmap.frontend.paths import FrontendPaths
from vidmap.frontend.tracking.sparse_tracks import SparseTrackResult
from vidmap.repro.frontend import write_sequence_artifact, write_sqlite_summary_artifact, write_tcorr_artifact

logger = logging.getLogger(__name__)

_TRANSITIVE_CORRESPONDENCE_DEPTH = 10_000_000


def build_transitive_correspondences(graph, names_to_ids, sequence, transitivity_depth, database_path):
    """Build transitive rows with the established set-deduplicated byte order."""
    correspondences = defaultdict(list)
    with pycolmap.Database.open(str(database_path)) as database:
        for name in sequence:
            image_id = names_to_ids[name]
            if not graph.exists_image(image_id):
                continue
            for point2d_id in range(database.read_keypoints(image_id).shape[0]):
                # This is pycolmap's public method name, not VidMap frontend
                # terminology, so retain the upstream API verb.
                for match in graph.extract_transitive_correspondences(image_id, point2d_id, transitivity_depth):
                    if (match.image_id, image_id) in correspondences:
                        continue
                    correspondences[image_id, match.image_id].append((point2d_id, match.point2D_idx))
    return {pair: np.array(list(set(matches))) for pair, matches in correspondences.items() if matches}


@result_dataclass(frozen=True)
class TransitiveCorrespondenceResult:
    """Transitive correspondence graph built from certified sparse tracks."""

    sequence: tuple[str, ...]
    tcorr: Mapping[ImagePair, Any]

    def __post_init__(self):
        object.__setattr__(self, "sequence", tuple(self.sequence))
        object.__setattr__(self, "tcorr", immutable_array_mapping(self.tcorr))


def load_correspondence_graph(database_path, vidmap_rec):
    """Load the COLMAP correspondence graph and populate reconstruction keypoints."""
    database = Database.open(str(database_path))
    cache = DatabaseCache.create(database, pycolmap.DatabaseCacheOptions())
    graph = cache.correspondence_graph
    for image in database.read_all_images():
        keypoints = database.read_keypoints(image.image_id)
        vidmap_rec.images[image.image_id].points2D = [
            pycolmap.Point2D(xy=keypoint) for keypoint in keypoints.astype(np.float16)
        ]
    return graph


class TransitiveCorrespondenceBuilder:
    """Build the transitive database and its immutable correspondence result."""

    def __init__(
        self,
        *,
        scene_parser: DatasetParser,
        paths: FrontendPaths,
        repro_dir: Path | None,
        keyframes: KeyframePlan,
        tracks: SparseTrackResult,
    ):
        self.scene_parser = scene_parser
        self.paths = paths
        self.repro_dir = repro_dir
        self.keyframes = keyframes
        self.tracks = tracks

    def build(self) -> TransitiveCorrespondenceResult:
        logger.info("Building transitive correspondences")
        pairs = self.tracks.track_pairs
        keyframe_names = tuple(self.keyframes.names)
        keyframes = set(keyframe_names)
        if tuple(pairs) != self.keyframes.track_pairs:
            raise ValueError("Sparse track pairs do not match the certified adjacent keyframe plan")
        source_reconstruction = self.scene_parser.rec
        if source_reconstruction is None:
            raise ValueError("scene_parser.rec is required for transitive correspondences frontend")
        source_names = {image.name for image in source_reconstruction.images.values()}
        missing_names = sorted(keyframes - source_names)
        if missing_names:
            raise ValueError(f"Image {missing_names[0]} not found in scene_parser.rec.images")

        database_path = self.paths.database_transitive_path
        database_path.parent.mkdir(parents=True, exist_ok=True)
        build_colmap_database(
            database_path,
            source_reconstruction,
            keyframes,
            self.paths.sparse_features_path,
            pairs,
            camera_policy="transitive",
            prior_focal_length=True,
            sparse_matches_path=self.paths.sparse_matches_path,
            seed_two_view_geometry=True,
        )

        reconstruction = build_initial_reconstruction(
            scene_parser=self.scene_parser,
            reference_image_names=list(keyframe_names),
        )
        correspondence_graph = load_correspondence_graph(database_path, vidmap_rec=reconstruction)
        sequence = keyframe_names
        names_to_ids = {image.name: image_id for image_id, image in reconstruction.images.items()}
        tcorr = build_transitive_correspondences(
            correspondence_graph,
            names_to_ids,
            sequence,
            _TRANSITIVE_CORRESPONDENCE_DEPTH,
            database_path,
        )
        tcorr = {
            (
                reconstruction.images[pair[0]].name,
                reconstruction.images[pair[1]].name,
            ): matches
            for pair, matches in tcorr.items()
        }
        logger.info("Built %d transitive correspondence pairs", len(tcorr))

        if self.repro_dir is not None:
            write_sequence_artifact(
                self.repro_dir / "stage1_transitive_sequence_order.json",
                sequence,
                label="transitive_sequence",
            )
            write_tcorr_artifact(self.repro_dir / "stage1_raw_tcorr.json", tcorr, label="raw_tcorr")
            write_sqlite_summary_artifact(
                self.repro_dir / "stage1_database_transitive_summary.json",
                database_path,
                label="database_transitive",
            )
        return TransitiveCorrespondenceResult(tuple(sequence), tcorr)
