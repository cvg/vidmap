"""Build tracking-owned transitive correspondences from sparse tracks."""

import logging
from collections.abc import Mapping
from dataclasses import dataclass as result_dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pycolmap

from vidmap.datasets.capture import PreparedCapture
from vidmap.frontend.colmap_database import build_colmap_database
from vidmap.frontend.correspondences import ImagePair, immutable_array_mapping
from vidmap.frontend.keyframes.processing import KeyframePlan
from vidmap.frontend.paths import FrontendPaths
from vidmap.frontend.tracking.sparse_tracks import SparseTrackResult
from vidmap.repro.frontend import write_sequence_artifact, write_sqlite_summary_artifact, write_tcorr_artifact
from vidmap.utils.io import get_matches_from_h5

logger = logging.getLogger(__name__)

_TRANSITIVE_CORRESPONDENCE_DEPTH = 10_000_000


def build_correspondence_graph(source_reconstruction, keyframes, pairs, paths):
    """Build COLMAP's graph directly, preserving SQLite's canonical pair traversal."""
    names_to_ids = {image.name: image_id for image_id, image in source_reconstruction.images.items()}
    graph = pycolmap.CorrespondenceGraph()
    counts = {}
    with h5py.File(paths.sparse_features_path, "r", libver="latest") as features:
        for name in keyframes:
            image_id = names_to_ids[name]
            counts[image_id] = features[name]["keypoints"].shape[0]
            graph.add_image(image_id, counts[image_id])
    ordered_pairs = {}
    for name0, name1 in pairs:
        pair_id = pycolmap.image_pair_to_pair_id(names_to_ids[name0], names_to_ids[name1])
        if pair_id not in ordered_pairs:
            ordered_pairs[pair_id] = (name0, name1)
    with h5py.File(paths.sparse_matches_path, "r", libver="latest") as matches_file:
        for pair_id, (name0, name1) in sorted(ordered_pairs.items()):
            matches, _ = get_matches_from_h5(matches_file, name0, name1)
            if not len(matches):
                continue
            if names_to_ids[name0] > names_to_ids[name1]:
                matches = matches[:, ::-1]
            graph.add_two_view_geometry(
                *pycolmap.pair_id_to_image_pair(pair_id),
                pycolmap.TwoViewGeometry(
                    config=pycolmap.TwoViewGeometryConfiguration.CALIBRATED,
                    inlier_matches=np.asarray(matches, dtype=np.uint32),
                ),
            )
    graph.finalize()
    return graph, names_to_ids, counts


def build_transitive_correspondences(graph, names_to_ids, sequence, transitivity_depth, counts):
    """Build ordered native rows and translate image IDs to frontend names."""
    from vidmap_native import _core

    ids = [names_to_ids[name] for name in sequence]
    rows = _core.build_transitive_correspondences(graph, ids, [counts[image] for image in ids], transitivity_depth)
    names = {image_id: name for name, image_id in names_to_ids.items()}
    return {(names[a], names[b]): matches for (a, b), matches in rows.items()}


@result_dataclass(frozen=True)
class TransitiveCorrespondenceResult:
    """Transitive correspondence graph built from certified sparse tracks."""

    tcorr: Mapping[ImagePair, Any]

    def __post_init__(self):
        object.__setattr__(self, "tcorr", immutable_array_mapping(self.tcorr))


class TransitiveCorrespondenceBuilder:
    """Build the native graph and its immutable correspondence result."""

    def __init__(
        self,
        *,
        scene_parser: PreparedCapture,
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

        correspondence_graph, names_to_ids, counts = build_correspondence_graph(
            source_reconstruction, keyframes, pairs, self.paths
        )
        sequence = keyframe_names
        tcorr = build_transitive_correspondences(
            correspondence_graph, names_to_ids, sequence, _TRANSITIVE_CORRESPONDENCE_DEPTH, counts
        )
        logger.info("Built %d transitive correspondence pairs", len(tcorr))

        if self.repro_dir is not None:
            database_path = self.paths.database_transitive_path
            database_path.parent.mkdir(parents=True, exist_ok=True)
            build_colmap_database(
                database_path,
                source_reconstruction,
                keyframes,
                self.paths.sparse_features_path,
                pairs,
                diagnostic_transitive=True,
                prior_focal_length=True,
                sparse_matches_path=self.paths.sparse_matches_path,
                seed_two_view_geometry=True,
            )

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
        return TransitiveCorrespondenceResult(tcorr)
