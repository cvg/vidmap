"""Sparse tracking and cache reuse."""

from __future__ import annotations

import logging
import tempfile
from dataclasses import dataclass as internal_dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping

from vidmap.datasets.base import DatasetParser
from vidmap.frontend.cache import (
    CompleteArtifactContract,
    IncrementalArtifactContract,
    artifact_fingerprint,
    cache_metadata,
    certify_complete_artifact,
    certify_incremental_artifact,
    incremental_cache_is_complete,
    mark_incremental_cache_complete,
    read_cache_metadata,
    semantic_config,
)
from vidmap.frontend.correspondences import validate_pair_name_plan
from vidmap.frontend.keyframes.processing import KeyframePlan
from vidmap.frontend.loop_closure.cache import ExtendedMatchCache
from vidmap.frontend.models.romav2 import LazyRoMaV2Tracker, romav2_cache_identity
from vidmap.frontend.options.matching import RoMaImageOptions, RoMaV2Options
from vidmap.frontend.options.tracking import SparseTrackOptions
from vidmap.frontend.paths import FrontendPaths
from vidmap.frontend.tracking.multiflow import generate_multiflow_overlap_pairs
from vidmap.frontend.tracking.propagation import StreamingTrackPropagator
from vidmap.repro.frontend import write_pair_order_artifact
from vidmap.utils.parsers import names_to_pair

_RUNTIME_CONFIG_FIELDS = frozenset({"num_workers"})

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from vidmap.frontend.options.matching import ExtendedMatchOptions


@internal_dataclass(frozen=True)
class _TrackOutputMetadata:
    sparse_features: Mapping[str, Any]
    sparse_matches: Mapping[str, Any]


@internal_dataclass(frozen=True)
class _TrackCachePlan:
    keyframe_sequence: tuple[str, ...]
    keyframe_pairs: tuple[tuple[str, str], ...]
    sequential_pairs: tuple[tuple[str, str], ...]
    track_pairs: Mapping[str, Any]
    outputs: _TrackOutputMetadata


@internal_dataclass(frozen=True)
class SparseTrackResult:
    """Certified sparse tracking outputs and the sequential LC cache identity."""

    track_pairs: tuple[tuple[str, str], ...]
    sequential_pairs: tuple[tuple[str, str], ...]
    track_pairs_artifact: CompleteArtifactContract
    sparse_features: IncrementalArtifactContract
    sparse_matches: IncrementalArtifactContract
    extended_matches: ExtendedMatchCache

    def __post_init__(self):
        object.__setattr__(self, "track_pairs", tuple(tuple(pair) for pair in self.track_pairs))
        object.__setattr__(
            self,
            "sequential_pairs",
            tuple(tuple(pair) for pair in self.sequential_pairs),
        )


def overlap_pair_plan(keyframe_sequence, *, track_options):
    return generate_multiflow_overlap_pairs(
        sequence=keyframe_sequence,
        multiflow_hops=track_options.multiflow_hops,
    )


def _trackprop_cache_config(track_options, lc_match_thresh):
    """Return the sparse-track options that affect propagation outputs."""
    config = semantic_config(track_options, exclude_fields=_RUNTIME_CONFIG_FIELDS)
    config["lc_match_thresh"] = lc_match_thresh
    return config


def _highres_cache_config(highres_options, lowres_match_resolution):
    """Return image and low-resolution matching settings used by tracking."""
    config = semantic_config(highres_options)
    config["lowres_match_resolution"] = lowres_match_resolution
    return config


def track_output_cache_metadata(
    track_options,
    highres_options,
    lowres_match_resolution,
    extended_options,
    tracker_options,
    keyframe_sequence,
    keyframe_pairs,
    track_pairs_metadata,
    salient_features_fingerprint,
):
    common_config = {
        "tracker": romav2_cache_identity(tracker_options),
        "trackprop": _trackprop_cache_config(track_options, extended_options.lc_match_thresh),
        "highres": _highres_cache_config(highres_options, lowres_match_resolution),
    }
    ordered_inputs = {
        "keyframe_sequence": keyframe_sequence,
        "keyframe_pairs": keyframe_pairs,
    }
    upstream = {
        "track_pairs": artifact_fingerprint(track_pairs_metadata),
        "salient_features": salient_features_fingerprint,
    }
    return _TrackOutputMetadata(
        sparse_features=cache_metadata(
            stage="sparse_features",
            config=common_config,
            ordered_inputs=ordered_inputs,
            upstream=upstream,
            payload_format="per-image-local-features",
            nonsemantic_config_fields=_RUNTIME_CONFIG_FIELDS,
        ),
        sparse_matches=cache_metadata(
            stage="sparse_tracks",
            config=common_config,
            ordered_inputs=ordered_inputs,
            upstream=upstream,
            payload_format="per-pair-indexed-matches",
            nonsemantic_config_fields=_RUNTIME_CONFIG_FIELDS,
        ),
    )


def extended_matches_cache_metadata(
    track_options,
    highres_options,
    lowres_match_resolution,
    extended_options,
    tracker_options,
    keyframe_sequence,
    sequential_pairs,
    track_pairs_metadata,
    sparse_features_metadata,
):
    return cache_metadata(
        stage="extended_matches",
        config={
            "tracker": romav2_cache_identity(tracker_options),
            "highres": _highres_cache_config(highres_options, lowres_match_resolution),
            "trackprop": _trackprop_cache_config(track_options, extended_options.lc_match_thresh),
            "retrieval": {},
            "extended": extended_options,
        },
        ordered_inputs={
            "keyframe_sequence": keyframe_sequence,
            "sequential_pairs": sequential_pairs,
        },
        upstream={
            "track_pairs": artifact_fingerprint(track_pairs_metadata),
            "sparse_features": artifact_fingerprint(sparse_features_metadata),
        },
        payload_format="per-pair-extended-matches",
        nonsemantic_config_fields=_RUNTIME_CONFIG_FIELDS,
    )


class SparseTrackBuilder:
    """Own sparse tracking caches and propagation."""

    def __init__(
        self,
        *,
        scene_parser: DatasetParser,
        paths: FrontendPaths,
        force_recompute: bool,
        repro_dir: Path | None,
        tracker: LazyRoMaV2Tracker,
        tracker_options: RoMaV2Options,
        keyframes: KeyframePlan | None,
        track_options: SparseTrackOptions,
        highres_options: RoMaImageOptions,
        lowres_match_resolution: int,
        extended_options: ExtendedMatchOptions,
    ):
        self.scene_parser = scene_parser
        self.paths = paths
        self.force_recompute = force_recompute
        self.repro_dir = repro_dir
        self.tracker = tracker
        self.tracker_options = tracker_options
        self.keyframes = keyframes
        self.track_options = track_options
        self.highres_options = highres_options
        self.lowres_match_resolution = lowres_match_resolution
        self.extended_options = extended_options

    def _cache_plan(self) -> _TrackCachePlan:
        paths = self.paths
        keyframe_sequence = tuple(self.keyframes.names)
        keyframe_pairs = tuple(self.keyframes.track_pairs)
        sequential_pairs = tuple(
            overlap_pair_plan(
                keyframe_sequence,
                track_options=self.track_options,
            )
        )
        validate_pair_name_plan(sequential_pairs)
        track_metadata = self.keyframes.track_pairs_artifact.metadata
        salient_features_fingerprint = artifact_fingerprint(read_cache_metadata(paths.salient_features_path))
        outputs = track_output_cache_metadata(
            self.track_options,
            self.highres_options,
            self.lowres_match_resolution,
            self.extended_options,
            self.tracker_options,
            keyframe_sequence,
            keyframe_pairs,
            track_metadata,
            salient_features_fingerprint,
        )
        return _TrackCachePlan(
            keyframe_sequence,
            keyframe_pairs,
            sequential_pairs,
            track_metadata,
            outputs,
        )

    def _outputs_complete(self, plan: _TrackCachePlan) -> bool:
        paths = self.paths
        return incremental_cache_is_complete(
            paths.sparse_features_path,
            plan.outputs.sparse_features,
            plan.keyframe_sequence,
        ) and incremental_cache_is_complete(
            paths.sparse_matches_path,
            plan.outputs.sparse_matches,
            [names_to_pair(*pair) for pair in plan.keyframe_pairs],
        )

    def _extended_metadata(self, plan: _TrackCachePlan, sparse_features_metadata):
        return extended_matches_cache_metadata(
            self.track_options,
            self.highres_options,
            self.lowres_match_resolution,
            self.extended_options,
            self.tracker_options,
            plan.keyframe_sequence,
            plan.sequential_pairs,
            self.keyframes.track_pairs_artifact.metadata,
            sparse_features_metadata,
        )

    def _load(self, plan: _TrackCachePlan) -> ExtendedMatchCache:
        paths = self.paths
        extended_metadata = self._extended_metadata(
            plan,
            read_cache_metadata(paths.sparse_features_path),
        )
        extended_cache = ExtendedMatchCache(
            paths,
            extended_metadata,
            force_recompute=self.force_recompute,
        )
        extended_cache.prepare()
        return extended_cache

    def _publish(self, plan, staging_path):
        paths = self.paths
        mark_incremental_cache_complete(
            paths.sparse_features_path,
            plan.outputs.sparse_features,
            plan.keyframe_sequence,
        )
        mark_incremental_cache_complete(
            paths.sparse_matches_path,
            plan.outputs.sparse_matches,
            [names_to_pair(*pair) for pair in plan.keyframe_pairs],
        )
        extended_cache = ExtendedMatchCache(
            paths,
            self._extended_metadata(plan, read_cache_metadata(paths.sparse_features_path)),
            force_recompute=self.force_recompute,
        )
        extended_cache.publish_staging(staging_path)
        return extended_cache

    def admit_and_build_tracks(self, candidates, keyframe_options, finalize_keyframes):
        """Admit candidate keyframes while building their sparse tracks."""
        from vidmap.utils.profiling import log_memory, record_timing, sync_time

        paths = self.paths
        paths.sparse_features_path.unlink(missing_ok=True)
        paths.sparse_matches_path.unlink(missing_ok=True)
        with tempfile.TemporaryDirectory(
            dir=paths.extended_matches_path.parent,
            prefix=".extended-matches-",
        ) as staging_dir:
            staging_path = Path(staging_dir) / paths.extended_matches_path.name
            started = sync_time()
            accepted_ids = StreamingTrackPropagator(
                conf=self.track_options,
                scene_parser=self.scene_parser,
                paths=paths,
                tracker_model=self.tracker.get(),
                keyframe_sequence=candidates.names,
                candidate_indices=candidates.indices,
                forced_candidate_indices=candidates.forced_indices,
                calibrations=candidates.calibrations,
                keyframe_options=keyframe_options,
                conf_highres=self.highres_options,
                lowres_match_resolution=self.lowres_match_resolution,
                extended_matches_path=staging_path,
                lc_match_thresh=self.extended_options.lc_match_thresh,
            ).run()
            record_timing("track_propagation", sync_time() - started)
            log_memory("track_propagation")

            self.keyframes = finalize_keyframes(accepted_ids)
            plan = self._cache_plan()
            extended_cache = self._publish(plan, staging_path)

        self._repair_sequential_matches(plan, extended_cache, loaded=False)
        return self.keyframes, self._result(plan, extended_cache)

    def _repair_sequential_matches(self, plan: _TrackCachePlan, extended_cache: ExtendedMatchCache, *, loaded):
        present_count, _ = extended_cache.repair(
            plan.sequential_pairs,
            label="sequential",
            tracker=self.tracker,
            scene_parser=self.scene_parser,
            image_options=self.highres_options,
            lowres_match_resolution=self.lowres_match_resolution,
            match_threshold=self.extended_options.lc_match_thresh,
        )
        if loaded:
            logger.info("Loaded %d cached track-overlap pairs", present_count)

    def _result(self, plan: _TrackCachePlan, extended_cache: ExtendedMatchCache) -> SparseTrackResult:
        sparse_match_items = tuple(names_to_pair(*pair) for pair in plan.keyframe_pairs)
        result = SparseTrackResult(
            track_pairs=plan.keyframe_pairs,
            sequential_pairs=plan.sequential_pairs,
            track_pairs_artifact=certify_complete_artifact(self.paths.track_pairs_path, plan.track_pairs),
            sparse_features=certify_incremental_artifact(
                self.paths.sparse_features_path,
                plan.outputs.sparse_features,
                plan.keyframe_sequence,
            ),
            sparse_matches=certify_incremental_artifact(
                self.paths.sparse_matches_path,
                plan.outputs.sparse_matches,
                sparse_match_items,
            ),
            extended_matches=extended_cache,
        )
        if self.repro_dir is not None:
            write_pair_order_artifact(
                self.repro_dir / "stage1_track_pair_order.json",
                plan.keyframe_pairs,
                label="track_pairs",
            )
        return result

    def load_complete(self) -> SparseTrackResult | None:
        """Load only a fully published core tracking section."""
        if self.force_recompute or self.keyframes is None:
            return None
        plan = self._cache_plan()
        if not self._outputs_complete(plan):
            return None
        extended_cache = self._load(plan)
        self._repair_sequential_matches(plan, extended_cache, loaded=True)
        return self._result(plan, extended_cache)
