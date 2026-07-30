"""Orchestrate streaming keyframe selection and its certified output plan."""

import logging
from dataclasses import dataclass as result_dataclass
from pathlib import Path

import torch
from tqdm import tqdm

from vidmap.datasets.base import DatasetParser
from vidmap.frontend.cache import (
    CacheMetadataMismatch,
    CompleteArtifactContract,
    cache_is_valid,
    certify_complete_artifact,
    read_cache_metadata,
)
from vidmap.frontend.correspondences import validate_pair_name_plan
from vidmap.frontend.geocalib import (
    estimate_keyframe_bootstrap_intrinsics,
    validate_keyframe_bootstrap_frame_dimensions,
)
from vidmap.frontend.image_dataset import FrameSequence
from vidmap.frontend.keyframes import cache as keyframe_cache
from vidmap.frontend.keyframes import matching as keyframe_matching
from vidmap.frontend.keyframes import selector as keyframe_selector
from vidmap.frontend.models.romav2 import LazyRoMaV2Tracker
from vidmap.frontend.options.keyframes import DetectKeyframesOptions, SalientFeatureOptions
from vidmap.frontend.options.matching import LowresMatchOptions, RoMaV2Options
from vidmap.frontend.paths import FrontendPaths
from vidmap.repro.frontend import write_pair_order_artifact, write_sequence_artifact
from vidmap.utils.logging import progress_bars_enabled

logger = logging.getLogger(__name__)


@result_dataclass(frozen=True)
class KeyframePlan:
    """Certified keyframe selection and its consecutive tracking plan."""

    names: tuple[str, ...]
    track_pairs: tuple[tuple[str, str], ...]
    keyframes: CompleteArtifactContract

    def __post_init__(self):
        names = tuple(self.names)
        track_pairs = tuple(tuple(pair) for pair in self.track_pairs)
        if len(names) < 2 or len(set(names)) != len(names):
            raise ValueError("Keyframe plan requires at least two unique images")
        expected_pairs = tuple(zip(names, names[1:]))
        if track_pairs != expected_pairs:
            raise ValueError("Keyframe track pairs must be the exact adjacent image chain")
        validate_pair_name_plan(track_pairs)
        object.__setattr__(self, "names", names)
        object.__setattr__(self, "track_pairs", track_pairs)


class KeyframeProcessor:
    """Own keyframe selection and its paired salient-feature cache."""

    def __init__(
        self,
        *,
        scene_parser: DatasetParser,
        frames: FrameSequence,
        paths: FrontendPaths,
        force_recompute: bool,
        repro_dir: Path | None,
        tracker: LazyRoMaV2Tracker,
        tracker_options: RoMaV2Options,
        lowres_options: LowresMatchOptions,
        keyframe_options: DetectKeyframesOptions,
        salient_options: SalientFeatureOptions,
    ):
        self.scene_parser = scene_parser
        self.frames = frames
        self.paths = paths
        self.force_recompute = force_recompute
        self.repro_dir = repro_dir
        self.tracker = tracker
        self.tracker_options = tracker_options
        self.lowres_options = lowres_options
        self.keyframe_options = keyframe_options
        self.salient_options = salient_options

    def process(self) -> KeyframePlan:
        """Select keyframes and return their certified ordered plans."""
        from vidmap.utils.profiling import log_memory, record_timing, sync_time

        sequence = self.frames.names
        timestamps = self.frames.timestamps
        keyframes_metadata = keyframe_cache.keyframe_cache_metadata(
            scene_parser=self.scene_parser,
            sequence=sequence,
            timestamps=timestamps,
            tracker_options=self.tracker_options,
            lowres_options=self.lowres_options,
            keyframe_options=self.keyframe_options,
            salient_options=self.salient_options,
        )
        keyframe_dependency_metadata = keyframes_metadata
        if cache_is_valid(self.paths.keyframes_path, keyframes_metadata):
            keyframe_dependency_metadata = read_cache_metadata(self.paths.keyframes_path)
        salient_metadata = keyframe_cache.salient_feature_cache_metadata(
            salient_options=self.salient_options,
            sequence=sequence,
            timestamps=timestamps,
            keyframe_metadata=keyframe_dependency_metadata,
        )

        started = sync_time()
        keyframe_ids = self._select_keyframe_ids(keyframes_metadata, salient_metadata)
        record_timing("keyframing", sync_time() - started)
        log_memory("keyframing")

        keyframe_sequence = tuple(sequence[index] for index in keyframe_ids)
        keyframe_pairs = tuple(zip(keyframe_sequence, keyframe_sequence[1:]))
        if self.repro_dir is not None:
            write_sequence_artifact(
                self.repro_dir / "stage1_keyframe_order.json",
                keyframe_sequence,
                label="keyframe_order",
            )
            write_pair_order_artifact(
                self.repro_dir / "stage1_keyframe_pair_order.json",
                keyframe_pairs,
                label="keyframe_pairs",
            )
        if len(keyframe_sequence) < 2 or not keyframe_pairs:
            raise CacheMetadataMismatch("Frontend requires at least two keyframes and one track pair")
        return KeyframePlan(
            names=keyframe_sequence,
            track_pairs=keyframe_pairs,
            keyframes=certify_complete_artifact(self.paths.keyframes_path, keyframes_metadata),
        )

    def _select_keyframe_ids(self, keyframes_metadata, salient_metadata):
        sequence = list(self.frames.names)
        keyframe_options = self.keyframe_options
        scene_parser = self.scene_parser
        if keyframe_options.intrinsics_source == "geocalib":
            validate_keyframe_bootstrap_frame_dimensions(scene_parser.rgb_dir, sequence)
        cached_keyframes = keyframe_cache.load_or_repair_cached_ids(
            scene_parser=scene_parser,
            sequence=self.frames.names,
            keyframes_path=self.paths.keyframes_path,
            salient_features_path=self.paths.salient_features_path,
            force_recompute=self.force_recompute,
            keyframe_options=keyframe_options,
            salient_options=self.salient_options,
            keyframes_metadata=keyframes_metadata,
            salient_metadata=salient_metadata,
        )
        if cached_keyframes is not None:
            return cached_keyframes

        bootstrap_intrinsics = None
        if keyframe_options.intrinsics_source == "geocalib":
            bootstrap_intrinsics = estimate_keyframe_bootstrap_intrinsics(
                scene_parser.rgb_dir,
                sequence,
            )
        tracker_model = self.tracker.get()
        logger.info("Starting streaming keyframe detection and salient-feature extraction")
        keyframe_cache.prepare_salient_feature_cache(
            self.paths.salient_features_path,
            salient_metadata,
            overwrite=self.force_recompute,
        )

        loader, total_pairs, original_width, original_height = keyframe_matching.build_pair_loader(
            scene_parser,
            sequence,
            self.lowres_options,
        )

        aliked_model = None
        selector = None
        try:
            aliked_model = keyframe_selector.create_aliked_model(self.salient_options)
            selector = keyframe_selector.KeyframeSelector(
                scene_parser,
                self.paths.salient_features_path,
                sequence,
                keyframe_options,
                self.salient_options,
                aliked_model,
                original_width,
                original_height,
                bootstrap_intrinsics,
            )

            first_batch = True
            with tqdm(
                total=total_pairs,
                desc="Streaming keyframe detection",
                disable=not progress_bars_enabled(),
            ) as progress:
                for batch in loader:
                    matches, certainties = keyframe_matching.match_lowres_batch(
                        tracker_model,
                        batch,
                        original_width,
                        original_height,
                        first_batch,
                    )
                    first_batch = False
                    for pair_match_lr, pair_cert_lr in zip(matches, certainties):
                        selector.process_pair(pair_match_lr, pair_cert_lr)
                        progress.update(1)

            keyframe_ids = selector.finish()
            keyframe_cache.commit_keyframes(
                sequence=self.frames.names,
                timestamps=self.frames.timestamps,
                keyframe_ids=keyframe_ids,
                gt_frame_indices=selector.gt_frame_indices,
                keyframes_path=self.paths.keyframes_path,
                salient_features_path=self.paths.salient_features_path,
                keyframe_options=keyframe_options,
                salient_options=self.salient_options,
                keyframes_metadata=keyframes_metadata,
                salient_metadata=salient_metadata,
            )
            return keyframe_ids
        finally:
            del selector
            del aliked_model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
