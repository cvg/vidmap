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
    certify_complete_artifact,
    prepare_incremental_cache,
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
from vidmap.frontend.options.matching import LowresMatchOptions
from vidmap.frontend.paths import FrontendPaths
from vidmap.repro.frontend import write_pair_order_artifact, write_sequence_artifact
from vidmap.utils.logging import progress_bars_enabled

logger = logging.getLogger(__name__)


@result_dataclass(frozen=True)
class KeyframePlan:
    """Certified keyframe selection and its consecutive tracking plan."""

    names: tuple[str, ...]
    track_pairs: tuple[tuple[str, str], ...]
    track_pairs_artifact: CompleteArtifactContract

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


@result_dataclass(frozen=True)
class KeyframeCandidates:
    """Candidate keyframes and inputs required by lookahead admission."""

    indices: tuple[int, ...]
    names: tuple[str, ...]
    forced_indices: frozenset[int]
    calibrations: dict


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
        self.lowres_options = lowres_options
        self.keyframe_options = keyframe_options
        self.salient_options = salient_options

    def load_cached(self, track_pairs_metadata) -> KeyframePlan | None:
        """Load the admitted plan from its adjacent track-pair chain."""
        if self.keyframe_options.intrinsics_source == "geocalib":
            validate_keyframe_bootstrap_frame_dimensions(
                self.scene_parser.rgb_dir,
                self.frames.names,
            )
        keyframe_names = keyframe_cache.load_cached_names(
            scene_parser=self.scene_parser,
            sequence=self.frames.names,
            timestamps=self.frames.timestamps,
            track_pairs_path=self.paths.track_pairs_path,
            salient_features_path=self.paths.salient_features_path,
            force_recompute=self.force_recompute,
            keyframe_options=self.keyframe_options,
            salient_options=self.salient_options,
            track_pairs_metadata=track_pairs_metadata,
        )
        if keyframe_names is None:
            return None
        plan = self._certified_plan(keyframe_names, track_pairs_metadata)
        self._write_repro_plan(plan)
        return plan

    def _certified_plan(self, keyframe_sequence, track_pairs_metadata) -> KeyframePlan:
        keyframe_sequence = tuple(keyframe_sequence)
        keyframe_pairs = tuple(zip(keyframe_sequence, keyframe_sequence[1:]))
        if len(keyframe_sequence) < 2 or not keyframe_pairs:
            raise CacheMetadataMismatch("Frontend requires at least two keyframes and one track pair")
        return KeyframePlan(
            names=keyframe_sequence,
            track_pairs=keyframe_pairs,
            track_pairs_artifact=certify_complete_artifact(self.paths.track_pairs_path, track_pairs_metadata),
        )

    def _write_repro_plan(self, plan: KeyframePlan) -> None:
        if self.repro_dir is None:
            return
        write_sequence_artifact(
            self.repro_dir / "stage1_keyframe_order.json",
            plan.names,
            label="keyframe_order",
        )
        write_pair_order_artifact(
            self.repro_dir / "stage1_keyframe_pair_order.json",
            plan.track_pairs,
            label="keyframe_pairs",
        )

    def select_candidates(self, salient_metadata) -> KeyframeCandidates:
        """Select low-resolution candidates without publishing final keyframes."""
        sequence = list(self.frames.names)
        keyframe_options = self.keyframe_options
        scene_parser = self.scene_parser
        bootstrap_intrinsics = None
        if keyframe_options.intrinsics_source == "geocalib":
            bootstrap_intrinsics = estimate_keyframe_bootstrap_intrinsics(
                scene_parser.rgb_dir,
                sequence,
            )
        tracker_model = self.tracker.get()
        logger.info("Starting streaming keyframe detection and salient-feature extraction")
        prepare_incremental_cache(
            self.paths.salient_features_path,
            salient_metadata,
            overwrite=True,
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
            if keyframe_options.intrinsics_source == "geocalib":
                calibrations = {index: bootstrap_intrinsics for index in keyframe_ids}
            else:
                calibration_plan = keyframe_selector.ground_truth_intrinsics_plan(scene_parser, sequence)
                calibrations = {index: calibration_plan[index][1] for index in keyframe_ids}
            return KeyframeCandidates(
                indices=tuple(keyframe_ids),
                names=tuple(self.frames.names[index] for index in keyframe_ids),
                forced_indices=frozenset(selector.gt_frame_indices),
                calibrations=calibrations,
            )
        finally:
            del selector
            del aliked_model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def commit_keyframes(self, admitted_ids, forced_ids, track_pairs_metadata) -> KeyframePlan:
        """Publish the candidates retained by lookahead admission."""
        keyframe_cache.commit_keyframes(
            sequence=self.frames.names,
            timestamps=self.frames.timestamps,
            keyframe_ids=admitted_ids,
            gt_frame_indices=forced_ids,
            track_pairs_path=self.paths.track_pairs_path,
            salient_features_path=self.paths.salient_features_path,
            keyframe_options=self.keyframe_options,
            salient_options=self.salient_options,
            track_pairs_metadata=track_pairs_metadata,
        )
        names = tuple(self.frames.names[index] for index in admitted_ids)
        plan = self._certified_plan(names, track_pairs_metadata)
        self._write_repro_plan(plan)
        return plan
