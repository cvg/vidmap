"""Sparse tracking and cache reuse."""

from __future__ import annotations

import logging
import tempfile
from dataclasses import dataclass as internal_dataclass
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping

from vidmap.datasets.capture import PreparedCapture
from vidmap.frontend.cache import (
    IncrementalArtifactContract,
    artifact_fingerprint,
    cache_metadata,
    certify_incremental_artifact,
    incremental_cache_is_complete,
    mark_incremental_cache_complete,
    read_cache_metadata,
    semantic_config,
)
from vidmap.frontend.correspondences import combine_and_deduplicate_pairs, validate_pair_name_plan
from vidmap.frontend.keyframes.processing import KeyframePlan
from vidmap.frontend.loop_closure.cache import ExtendedMatchCache
from vidmap.frontend.loop_closure.matching import EXTENDED_MATCH_DIRECTION_POLICY
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
    import torch

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
    outputs: _TrackOutputMetadata


@internal_dataclass(frozen=True)
class SparseTrackResult:
    """Certified sparse tracking outputs and the sequential LC cache identity."""

    track_pairs: tuple[tuple[str, str], ...]
    sequential_pairs: tuple[tuple[str, str], ...]
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


def _highres_cache_config(highres_options, lowres_match_resolution):
    """Return image and low-resolution matching settings used by tracking."""
    config = semantic_config(highres_options)
    config["lowres_match_resolution"] = lowres_match_resolution
    return config


def track_output_cache_metadata(
    track_options,
    highres_options,
    lowres_match_resolution,
    tracker_options,
    keyframe_sequence,
    keyframe_pairs,
    admission_fingerprint,
    salient_features_fingerprint,
):
    common_config = {
        "tracker": romav2_cache_identity(tracker_options),
        "trackprop": semantic_config(track_options, exclude_fields=_RUNTIME_CONFIG_FIELDS),
        "highres": _highres_cache_config(highres_options, lowres_match_resolution),
    }
    ordered_inputs = {
        "keyframe_sequence": keyframe_sequence,
        "keyframe_pairs": keyframe_pairs,
    }
    upstream = {
        "admission": admission_fingerprint,
        "salient_features": salient_features_fingerprint,
    }
    return _TrackOutputMetadata(
        sparse_features=cache_metadata(
            stage="sparse_features",
            config=common_config,
            ordered_inputs=ordered_inputs,
            upstream=upstream,
            payload_format="per-image-local-features",
            payload_version=2,
            nonsemantic_config_fields=_RUNTIME_CONFIG_FIELDS,
        ),
        sparse_matches=cache_metadata(
            stage="sparse_tracks",
            config=common_config,
            ordered_inputs=ordered_inputs,
            upstream=upstream,
            payload_format="per-pair-indexed-matches",
            payload_version=2,
            nonsemantic_config_fields=_RUNTIME_CONFIG_FIELDS,
        ),
    )


def extended_matches_cache_metadata(
    highres_options,
    lowres_match_resolution,
    extended_options,
    tracker_options,
    sparse_features_metadata,
):
    """Bind finalized inference to its sparse index domain and sampling settings."""
    return cache_metadata(
        stage="extended_matches",
        config={
            "tracker": romav2_cache_identity(tracker_options),
            "highres": _highres_cache_config(highres_options, lowres_match_resolution),
            "inference_direction": EXTENDED_MATCH_DIRECTION_POLICY,
            "retrieval_direction": "temporal" if extended_options.lc_pair_nms else "canonical",
            "lc_match_thresh": extended_options.lc_match_thresh,
        },
        ordered_inputs={},
        upstream={"sparse_features": artifact_fingerprint(sparse_features_metadata)},
        payload_format="per-pair-extended-matches",
        nonsemantic_config_fields=_RUNTIME_CONFIG_FIELDS,
    )


class SparseTrackBuilder:
    """Own sparse tracking caches and propagation."""

    def __init__(
        self,
        *,
        scene_parser: PreparedCapture,
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
        device: torch.device,
    ):
        self.device = device
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
            pair
            for stream in self.keyframes.streams
            for pair in overlap_pair_plan(stream, track_options=self.track_options)
        )
        directed = {frozenset(pair): pair for pair in sequential_pairs}
        sequential_pairs = tuple(directed[frozenset(pair)] for pair in combine_and_deduplicate_pairs(sequential_pairs))
        validate_pair_name_plan(sequential_pairs)
        salient_features_fingerprint = artifact_fingerprint(read_cache_metadata(paths.salient_features_path))
        outputs = track_output_cache_metadata(
            self.track_options,
            self.highres_options,
            self.lowres_match_resolution,
            self.tracker_options,
            keyframe_sequence,
            keyframe_pairs,
            self.keyframes.admission_artifact.fingerprint,
            salient_features_fingerprint,
        )
        return _TrackCachePlan(keyframe_sequence, keyframe_pairs, sequential_pairs, outputs)

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

    def _extended_metadata(self, sparse_features_metadata):
        return extended_matches_cache_metadata(
            self.highres_options,
            self.lowres_match_resolution,
            self.extended_options,
            self.tracker_options,
            sparse_features_metadata,
        )

    def _load(self) -> ExtendedMatchCache:
        paths = self.paths
        extended_metadata = self._extended_metadata(
            read_cache_metadata(paths.sparse_features_path),
        )
        extended_cache = ExtendedMatchCache(paths, force_recompute=self.force_recompute)
        extended_cache.prepare(extended_metadata)
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
        extended_cache = ExtendedMatchCache(paths, force_recompute=self.force_recompute)
        extended_cache.publish_staging(
            staging_path, self._extended_metadata(read_cache_metadata(paths.sparse_features_path))
        )
        return extended_cache

    def admit_and_build_tracks(self, candidates, keyframe_options, finalize_keyframes, *, complete_salients):
        """Admit synchronized frames, then rebuild every camera's sparse tracks."""
        from vidmap.utils.profiling import log_memory, record_timing, sync_time

        paths = self.paths
        paths.sparse_features_path.unlink(missing_ok=True)
        paths.sparse_matches_path.unlink(missing_ok=True)
        with tempfile.TemporaryDirectory(
            dir=paths.extended_matches_path.parent,
            prefix=".extended-matches-",
        ) as staging_dir:
            staging_path = Path(staging_dir) / paths.extended_matches_path.name
            propagator = partial(
                StreamingTrackPropagator,
                conf=self.track_options,
                scene_parser=self.scene_parser,
                paths=self.paths,
                tracker_model=self.tracker.get(),
                conf_highres=self.highres_options,
                lowres_match_resolution=self.lowres_match_resolution,
                extended_matches_path=staging_path,
                lc_match_thresh=self.extended_options.lc_match_thresh,
                device=self.device,
            )
            started = sync_time()
            accepted_ids = propagator(
                keyframe_sequence=candidates.names,
                candidate_indices=candidates.indices,
                forced_candidate_indices=candidates.forced_indices,
                calibrations=candidates.calibrations,
                keyframe_options=keyframe_options,
            ).run()
            propagation_time = sync_time() - started
            self.keyframes = finalize_keyframes(accepted_ids)
            complete_salients(self.keyframes)
            for camera_index, stream in enumerate(self.keyframes.streams[1:], start=1):
                logger.info(
                    "Tracking camera stream %d/%d: %d admitted frames",
                    camera_index + 1,
                    len(self.keyframes.streams),
                    len(stream),
                )
                started = sync_time()
                propagator(keyframe_sequence=stream).run()
                propagation_time += sync_time() - started
            record_timing("track_propagation", propagation_time)
            log_memory("track_propagation")
            plan = self._cache_plan()
            extended_cache = self._publish(plan, staging_path)
        return self.keyframes, self._result(plan, extended_cache)

    def _result(self, plan: _TrackCachePlan, extended_cache: ExtendedMatchCache) -> SparseTrackResult:
        sparse_match_items = tuple(names_to_pair(*pair) for pair in plan.keyframe_pairs)
        result = SparseTrackResult(
            track_pairs=plan.keyframe_pairs,
            sequential_pairs=plan.sequential_pairs,
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
        extended_cache = self._load()
        return self._result(plan, extended_cache)
