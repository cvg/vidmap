"""Orchestrate streaming keyframe selection and its certified output plan."""

from __future__ import annotations

import logging
from dataclasses import dataclass as result_dataclass
from dataclasses import field
from itertools import chain
from pathlib import Path
from typing import TYPE_CHECKING

import h5py
from tqdm import tqdm

if TYPE_CHECKING:
    import torch

from vidmap.datasets.capture import CaptureSnapshot
from vidmap.frontend.cache import (
    CompleteArtifactContract,
    attach_incremental_cache_identity,
    certify_complete_artifact,
    inspect_incremental_items,
    mark_incremental_cache_complete,
    prepare_incremental_cache,
    prune_incremental_items,
    read_cache_metadata,
)
from vidmap.frontend.correspondences import combine_and_deduplicate_pairs
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
from vidmap.utils.device import empty_device_cache
from vidmap.utils.logging import progress_bars_enabled

logger = logging.getLogger(__name__)


@result_dataclass
class KeyframePlan:
    """Certified keyframe selection and its consecutive tracking plan."""

    frame_ids: tuple[int, ...]
    frames: FrameSequence
    admission_artifact: CompleteArtifactContract
    streams: tuple[tuple[str, ...], ...] = field(init=False)
    same_frame_flags: dict = field(init=False)
    names: tuple[str, ...] = field(init=False)
    track_pairs: tuple[tuple[str, str], ...] = field(init=False)
    frame_camera_indices: dict = field(init=False)

    def __post_init__(self):
        self.frame_ids = tuple(self.frame_ids)
        self.streams = self.frames.select(self.frames.indices(self.frame_ids))
        if len(self.frame_ids) < 2:
            raise ValueError("Keyframe selection must contain at least two native frames")
        self.names = tuple(chain.from_iterable(self.streams))
        self.track_pairs = tuple(pair for stream in self.streams for pair in zip(stream, stream[1:]))
        self.same_frame_flags = {
            (frame[first], frame[second]): flag
            for frame in zip(*self.streams, strict=True)
            for first, second, flag in self.frames.same_frame_stream_pairs
        }
        self.frame_camera_indices = {
            name: (index, camera_id)
            for stream, camera_id in zip(self.streams, self.frames.stream_camera_ids, strict=True)
            for index, name in enumerate(stream)
        }

    @property
    def same_frame_pairs(self):
        return tuple(combine_and_deduplicate_pairs(tuple(self.same_frame_flags)))


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
        input_snapshot: CaptureSnapshot,
        frames: FrameSequence,
        paths: FrontendPaths,
        force_recompute: bool,
        repro_dir: Path | None,
        tracker: LazyRoMaV2Tracker,
        lowres_options: LowresMatchOptions,
        keyframe_options: DetectKeyframesOptions,
        salient_options: SalientFeatureOptions,
        device: torch.device,
    ):
        self.device = device
        self.scene_parser = input_snapshot.capture
        self.input_snapshot = input_snapshot
        self.frames = frames
        self.paths = paths
        self.force_recompute = force_recompute
        self.repro_dir = repro_dir
        self.tracker = tracker
        self.lowres_options = lowres_options
        self.keyframe_options = keyframe_options
        self.salient_options = salient_options

    def load_cached(self, admission_metadata) -> KeyframePlan | None:
        """Load the frame selection and project it onto current camera streams."""
        if self.keyframe_options.intrinsics_source == "geocalib":
            validate_keyframe_bootstrap_frame_dimensions(
                self.scene_parser.rgb_dir,
                self.frames.streams[0],
            )
        frame_ids = keyframe_cache.load_cached_frame_ids(
            frames=self.frames,
            scene_parser=self.scene_parser,
            keyframe_options=self.keyframe_options,
            admitted_frames_path=self.paths.admitted_frames_path,
            force_recompute=self.force_recompute,
            admission_metadata=admission_metadata,
        )
        if frame_ids is None:
            return None
        plan = self._certified_plan(frame_ids, admission_metadata)
        self._write_repro_plan(plan)
        return plan

    def _certified_plan(self, frame_ids, admission_metadata) -> KeyframePlan:
        return KeyframePlan(
            frame_ids=frame_ids,
            frames=self.frames,
            admission_artifact=certify_complete_artifact(self.paths.admitted_frames_path, admission_metadata),
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
        sequence = list(self.frames.streams[0])
        keyframe_options = self.keyframe_options
        scene_parser = self.scene_parser
        bootstrap_intrinsics = None
        if keyframe_options.intrinsics_source == "geocalib":
            bootstrap_intrinsics = estimate_keyframe_bootstrap_intrinsics(
                scene_parser.rgb_dir,
                sequence,
                device=self.device,
            )
        tracker_model = self.tracker.get()
        logger.info("Starting streaming keyframe detection and salient-feature extraction")
        prepare_incremental_cache(
            self.paths.salient_features_path,
            salient_metadata,
            overwrite=self.force_recompute,
        )

        attach_incremental_cache_identity(self.paths.salient_features_path, salient_metadata)
        with h5py.File(self.paths.salient_features_path, "a") as hfile:
            for name in sequence:
                if name in hfile:
                    del hfile[name]

        loader, total_pairs, original_width, original_height = keyframe_matching.build_pair_loader(
            scene_parser,
            sequence,
            self.lowres_options,
        )

        aliked_model = None
        selector = None
        try:
            aliked_model = keyframe_selector.create_aliked_model(self.salient_options, device=self.device)
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

            with (
                keyframe_matching.pipelined_matches(
                    tracker_model,
                    loader,
                    original_width,
                    original_height,
                    batch_size=self.lowres_options.batch_size,
                    device=self.device,
                    coarse_only=self.lowres_options.coarse_only,
                ) as batches,
                tqdm(
                    total=total_pairs,
                    desc="Streaming keyframe detection",
                    disable=not progress_bars_enabled(),
                ) as progress,
            ):
                for matches, certainties in batches:
                    for pair_match_lr, pair_cert_lr in zip(matches, certainties):
                        selector.process_pair(pair_match_lr, pair_cert_lr)
                        progress.update(1)

            keyframe_ids = selector.finish()
            with h5py.File(self.paths.salient_features_path, "a") as hfile:
                for name in sequence:
                    if name in hfile:
                        hfile[name].attrs["input_fingerprint"] = self.input_snapshot.image_sha256[name]
            if keyframe_options.intrinsics_source == "geocalib":
                calibrations = {index: bootstrap_intrinsics for index in keyframe_ids}
            else:
                calibration_plan = keyframe_selector.ground_truth_intrinsics_plan(scene_parser, sequence)
                calibrations = {index: calibration_plan[index][1] for index in keyframe_ids}
            return KeyframeCandidates(
                indices=tuple(keyframe_ids),
                names=tuple(sequence[index] for index in keyframe_ids),
                forced_indices=frozenset(selector.gt_frame_indices),
                calibrations=calibrations,
            )
        finally:
            del selector
            del aliked_model
            empty_device_cache()

    def commit_keyframes(self, admitted_ids, forced_ids, admission_metadata) -> KeyframePlan:
        """Publish the candidates retained by lookahead admission."""
        keyframe_cache.commit_keyframes(
            frames=self.frames,
            keyframe_ids=admitted_ids,
            gt_frame_indices=forced_ids,
            admitted_frames_path=self.paths.admitted_frames_path,
            keyframe_options=self.keyframe_options,
            admission_metadata=admission_metadata,
        )
        frame_ids = tuple(self.frames.frame_ids[index] for index in admitted_ids)
        plan = self._certified_plan(frame_ids, admission_metadata)
        self._write_repro_plan(plan)
        return plan

    def complete_salient_features(self, plan):
        """Complete only missing or changed selected-image features."""
        metadata = keyframe_cache.salient_feature_cache_metadata(salient_options=self.salient_options)
        path = self.paths.salient_features_path
        prepare_incremental_cache(path, metadata)
        cached = read_cache_metadata(path)
        complete = "complete" in cached and cached["complete"]
        sources = {name: self.input_snapshot.image_sha256[name] for name in plan.names}
        with h5py.File(path, "r") as hfile:
            stale = [
                name
                for name in plan.names
                if name in hfile
                and (
                    "input_fingerprint" not in hfile[name].attrs
                    or hfile[name].attrs["input_fingerprint"] != sources[name]
                )
            ]
        if not stale and complete and cached["expected_items"] == list(plan.names):
            return
        attach_incremental_cache_identity(path, metadata)
        with h5py.File(path, "a") as hfile:
            for name in stale:
                del hfile[name]
        prune_incremental_items(path, plan.names)
        _, missing = inspect_incremental_items(path, plan.names, metadata, repair_malformed=True)
        self._extract_other_views(missing)
        with h5py.File(path, "a") as hfile:
            for name in missing:
                hfile[name].attrs["input_fingerprint"] = sources[name]
        mark_incremental_cache_complete(path, metadata, plan.names)

    def _extract_other_views(self, names):
        """Extract side-camera salient features only after whole-frame admission."""
        from vidmap.frontend.keyframes.selection import compute_aliked_features_for_frame

        if not names:
            return
        model = keyframe_selector.create_aliked_model(self.salient_options, device=self.device)
        try:
            for name in names:
                compute_aliked_features_for_frame(
                    self.scene_parser, self.paths.salient_features_path, name, model, self.salient_options
                )
        finally:
            del model
            empty_device_cache(self.device)
