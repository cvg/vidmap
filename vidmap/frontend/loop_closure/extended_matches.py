"""Retrieval cache reuse and loop-closure match finalization."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass as result_dataclass
from pathlib import Path
from typing import Any

import numpy as np

from vidmap.datasets.base import DatasetParser
from vidmap.frontend.cache import (
    CompleteArtifactContract,
    IncrementalArtifactContract,
    artifact_fingerprint,
    cache_is_valid,
    cache_metadata,
    certify_complete_artifact,
    incremental_cache_is_complete,
    read_cache_metadata,
    read_pair_artifact,
    write_pair_artifact,
)
from vidmap.frontend.correspondences import (
    ImagePair,
    combine_and_deduplicate_pairs,
    immutable_array_mapping,
    validate_correspondence_alignment,
    validate_pair_name_plan,
)
from vidmap.frontend.loop_closure.matching import merge_matches
from vidmap.frontend.loop_closure.retrieval import (
    RETRIEVAL_PAIR_SELECTION_POLICY_VERSION,
    compute_retrieval_features,
    generate_retrieval_pairs,
    retrieval_cache_identity,
)
from vidmap.frontend.models.romav2 import LazyRoMaV2Tracker
from vidmap.frontend.options.matching import ExtendedMatchOptions, RoMaImageOptions
from vidmap.frontend.paths import FrontendPaths
from vidmap.frontend.tracking.sparse_tracks import SparseTrackResult
from vidmap.frontend.tracking.transitive import TransitiveCorrespondenceResult
from vidmap.repro.frontend import write_pair_order_artifact, write_tcorr_artifact

_RUNTIME_CONFIG_FIELDS = frozenset({"num_workers"})

logger = logging.getLogger(__name__)


@result_dataclass(frozen=True)
class ExtendedMatchResult:
    """Final extended matches and merged transitive correspondences."""

    retrieval_pairs: tuple[ImagePair, ...]
    retrieval_pairs_artifact: CompleteArtifactContract
    tcorr: Mapping[ImagePair, Any]
    lc_masks: Mapping[ImagePair, Any]
    artifact: IncrementalArtifactContract

    def __post_init__(self):
        validate_correspondence_alignment(self.tcorr, self.lc_masks)
        object.__setattr__(self, "retrieval_pairs", tuple(tuple(pair) for pair in self.retrieval_pairs))
        object.__setattr__(self, "tcorr", immutable_array_mapping(self.tcorr))
        object.__setattr__(self, "lc_masks", immutable_array_mapping(self.lc_masks))


def retrieval_pairs_cache_metadata(
    extended_options, sequence, sequential_pairs, tcorr, retrieval_features_fingerprint
):
    conf = extended_options
    return cache_metadata(
        stage="retrieval_pairs",
        config={
            "retrieval_model": "megaloc",
            "selection_policy_version": RETRIEVAL_PAIR_SELECTION_POLICY_VERSION,
            "nquery": conf.nquery,
            "retrieval_min_score": conf.retrieval_min_score,
            "tcorr_min_matches": conf.tcorr_min_matches,
            "lc_pair_nms": conf.lc_pair_nms,
            "lc_pair_nms_radius": conf.lc_pair_nms_radius,
        },
        ordered_inputs={
            "sequence": sequence,
            "sequential_pairs": sequential_pairs,
            "tcorr": tcorr,
        },
        upstream={"retrieval_features": retrieval_features_fingerprint},
        payload_format="ordered-image-pairs",
        nonsemantic_config_fields=_RUNTIME_CONFIG_FIELDS,
    )


class ExtendedMatchBuilder:
    """Own retrieval pairing and final certification of extended matches."""

    def __init__(
        self,
        *,
        scene_parser: DatasetParser,
        paths: FrontendPaths,
        force_recompute: bool,
        repro_dir: Path | None,
        tracker: LazyRoMaV2Tracker,
        tracks: SparseTrackResult,
        transitive: TransitiveCorrespondenceResult,
        highres_options: RoMaImageOptions,
        lowres_match_resolution: int,
        extended_options: ExtendedMatchOptions,
        image_content_fingerprint: str,
    ):
        self.scene_parser = scene_parser
        self.paths = paths
        self.force_recompute = force_recompute
        self.repro_dir = repro_dir
        self.tracker = tracker
        self.tracks = tracks
        self.transitive = transitive
        self.highres_options = highres_options
        self.lowres_match_resolution = lowres_match_resolution
        self.extended_options = extended_options
        self.image_content_fingerprint = image_content_fingerprint

    def _build_retrieval_pairs(self):
        """Compute retrieval features and loop-closure matches."""
        paths = self.paths
        sequence = self.transitive.sequence
        sequential_pairs = self.tracks.sequential_pairs
        tcorr = self.transitive.tcorr
        from vidmap.utils.profiling import log_memory, record_timing, sync_time

        retrieval_pairs = []
        ext_conf = self.extended_options
        retrieval_features_metadata = retrieval_cache_identity(
            sequence,
            self.image_content_fingerprint,
        )
        retrieval_features_valid = incremental_cache_is_complete(
            paths.retrieval_features_path,
            retrieval_features_metadata,
            sequence,
        )
        retrieval_features_fingerprint = None
        if retrieval_features_valid:
            retrieval_features_fingerprint = artifact_fingerprint(read_cache_metadata(paths.retrieval_features_path))

        retrieval_pairs_metadata = None
        if retrieval_features_fingerprint is not None:
            retrieval_pairs_metadata = retrieval_pairs_cache_metadata(
                ext_conf,
                sequence,
                sequential_pairs,
                tcorr,
                retrieval_features_fingerprint,
            )

        if (
            not self.force_recompute
            and retrieval_pairs_metadata is not None
            and cache_is_valid(paths.retrieval_pairs_path, retrieval_pairs_metadata)
        ):
            retrieval_pairs = read_pair_artifact(paths.retrieval_pairs_path, retrieval_pairs_metadata)
            logger.info("Loaded %d cached retrieval pairs", len(retrieval_pairs))
        else:
            logger.info("Computing retrieval features for %d images", len(sequence))
            started = sync_time()
            compute_retrieval_features(
                scene_parser=self.scene_parser,
                retrieval_features_path=paths.retrieval_features_path,
                image_list=sequence,
                overwrite=self.force_recompute,
                cache_identity=retrieval_features_metadata,
            )
            record_timing("retrieval", sync_time() - started)
            log_memory("retrieval")
            retrieval_features_fingerprint = artifact_fingerprint(read_cache_metadata(paths.retrieval_features_path))
            retrieval_pairs_metadata = retrieval_pairs_cache_metadata(
                ext_conf,
                sequence,
                sequential_pairs,
                tcorr,
                retrieval_features_fingerprint,
            )
            retrieval_pairs = generate_retrieval_pairs(
                sequence=sequence,
                tcorr=tcorr,
                sequential_pairs=sequential_pairs,
                retrieval_path=paths.retrieval_features_path,
                tcorr_min_matches=ext_conf.tcorr_min_matches,
                retrieval_min_score=ext_conf.retrieval_min_score,
                nquery=ext_conf.nquery,
                lc_pair_nms=ext_conf.lc_pair_nms,
                lc_pair_nms_radius=ext_conf.lc_pair_nms_radius,
            )
            validate_pair_name_plan((*sequential_pairs, *retrieval_pairs))
            write_pair_artifact(paths.retrieval_pairs_path, retrieval_pairs, retrieval_pairs_metadata)
            logger.info("Cached %d retrieval pairs", len(retrieval_pairs))
        validate_pair_name_plan((*sequential_pairs, *retrieval_pairs))
        if self.repro_dir is not None:
            write_pair_order_artifact(
                self.repro_dir / "stage1_retrieval_pair_order.json",
                retrieval_pairs,
                label="retrieval_pairs",
            )

        started = sync_time()
        _present_count, missing_count = self.tracks.extended_matches.repair(
            retrieval_pairs,
            label="retrieval",
            tracker=self.tracker,
            scene_parser=self.scene_parser,
            image_options=self.highres_options,
            lowres_match_resolution=self.lowres_match_resolution,
            match_threshold=ext_conf.lc_match_thresh,
        )
        if missing_count:
            record_timing("lc_streaming", sync_time() - started)
            log_memory("lc_streaming")

        retrieval_pairs_artifact = certify_complete_artifact(paths.retrieval_pairs_path, retrieval_pairs_metadata)
        return retrieval_pairs, retrieval_pairs_artifact

    def _finalize_extended_matches(self, retrieval_pairs):
        """Merge persisted extended matches and certify their complete pair plan."""
        tcorr = dict(self.transitive.tcorr)
        extended_pairs = combine_and_deduplicate_pairs(self.tracks.sequential_pairs, retrieval_pairs)
        if self.repro_dir is not None:
            write_pair_order_artifact(
                self.repro_dir / "stage1_extended_pairs.json",
                extended_pairs,
                label="extended_pairs",
            )

        if extended_pairs:
            matches = self.tracks.extended_matches.collect(
                extended_pairs,
                match_threshold=self.extended_options.lc_match_thresh,
            )
            merged_tcorr, lc_masks = merge_matches(tcorr, matches)
            logger.info("Merged %d extended match pairs into transitive correspondences", len(matches))
        else:
            merged_tcorr = tcorr
            lc_masks = {pair: np.zeros(len(matches), dtype=bool) for pair, matches in tcorr.items()}

        if self.repro_dir is not None:
            write_tcorr_artifact(
                self.repro_dir / "stage1_merged_tcorr_lc_masks.json",
                merged_tcorr,
                lc_masks,
                label="merged_tcorr_lc_masks",
            )

        return merged_tcorr, lc_masks

    def build(self) -> ExtendedMatchResult:
        retrieval_pairs, retrieval_pairs_artifact = self._build_retrieval_pairs()
        tcorr, lc_masks = self._finalize_extended_matches(retrieval_pairs)
        extended_pairs = combine_and_deduplicate_pairs(self.tracks.sequential_pairs, retrieval_pairs)
        return ExtendedMatchResult(
            retrieval_pairs=tuple(retrieval_pairs),
            retrieval_pairs_artifact=retrieval_pairs_artifact,
            tcorr=tcorr,
            lc_masks=lc_masks,
            artifact=self.tracks.extended_matches.complete(extended_pairs),
        )
