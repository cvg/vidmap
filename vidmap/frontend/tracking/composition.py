"""Concrete composition of frontend's tracking-owned stages."""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

from vidmap.frontend.cache import fingerprint
from vidmap.frontend.depth import DepthEstimator
from vidmap.frontend.geocalib import CameraPriorEstimator
from vidmap.frontend.image_dataset import FrameSequence
from vidmap.frontend.keyframes import cache as keyframe_cache
from vidmap.frontend.keyframes.processing import KeyframeProcessor
from vidmap.frontend.loop_closure.extended_matches import ExtendedMatchBuilder
from vidmap.frontend.models.romav2 import create_lazy_romav2_tracker, romav2_cache_identity
from vidmap.frontend.paths import build_frontend_paths
from vidmap.frontend.preparation.camera_priors import CalibrationResult
from vidmap.frontend.preparation.cameras import initialize_selected_camera_geometry
from vidmap.frontend.tracking.sparse_tracks import SparseTrackBuilder
from vidmap.frontend.tracking.transitive import TransitiveCorrespondenceBuilder

logger = logging.getLogger(__name__)


class TrackingPipeline:
    """Run the visible keyframe-to-certified-tracking stage sequence."""

    def __init__(
        self,
        *,
        options,
        sample_name,
        cache_dir,
        cache_namespace,
        input_snapshot,
        preparation_plan,
        prepared_capture,
        force_recompute,
        deterministic,
        repro_dir,
        cache_full_depth_maps=False,
        device: torch.device,
    ):
        self.device = device
        self.options = options
        self.sample_name = sample_name
        self.cache_dir = Path(cache_dir)
        self.cache_namespace = None if cache_namespace is None else str(cache_namespace)
        self.input_snapshot = input_snapshot
        self.preparation_plan = preparation_plan
        self.prepared_capture = prepared_capture
        self.force_recompute = force_recompute
        self.cache_full_depth_maps = bool(cache_full_depth_maps)
        self.deterministic = deterministic
        self.repro_dir = repro_dir

    def run(self):
        """Run model stages, then depth and optional camera calibration."""
        with ThreadPoolExecutor(max_workers=1, thread_name_prefix="da3-verify") as preparation:
            return self._run(preparation)

    def _run(self, preparation):
        from vidmap.frontend.pipeline import FrontendArtifacts, TrackingFrontendResult, validate_tracking_result

        source = self.input_snapshot.capture
        working_capture = self.prepared_capture
        options = self.options
        tracker_options = options.roma
        prior_options = self.preparation_plan.camera_priors
        use_geocalib = prior_options.estimator == "geocalib"

        # Resolve every cache path from the frontend identity so all stages
        # operate on one immutable artifact plan.
        paths = build_frontend_paths(
            cache_dir=self.cache_dir,
            sample_name=self.sample_name,
            depth_model_name=options.depth.depth_model,
            use_geocalib=use_geocalib,
            frontend_tag=None,
            config_name=self.cache_namespace,
            cache_variant="cache-" + fingerprint({"tracker": romav2_cache_identity(tracker_options)})[:16],
        )
        paths.admitted_frames_path.parent.mkdir(exist_ok=True, parents=True)
        logger.info("Input sparse features: %s", paths.sparse_features_path)
        logger.info("Depth maps: %s", paths.depth_maps_path)
        if use_geocalib:
            logger.info("Geo-calibration (per-image): %s", paths.geocalib_per_image_path)
            logger.info("Geo-calibration (batch): %s", paths.geocalib_batch_path)
        frames = FrameSequence.from_scene(source, rig_options=options.rig)
        geocalib = None
        calibration = None
        verification = None
        with create_lazy_romav2_tracker(tracker_options, device=self.device) as tracker:
            keyframe_processor = KeyframeProcessor(
                input_snapshot=self.input_snapshot,
                frames=frames,
                paths=paths,
                force_recompute=self.force_recompute,
                repro_dir=self.repro_dir,
                tracker=tracker,
                lowres_options=options.keyframes.matching,
                keyframe_options=options.keyframes.selection,
                salient_options=options.keyframes.features,
                device=self.device,
            )
            admission_metadata = keyframe_cache.admitted_frames_cache_metadata(
                input_snapshot=self.input_snapshot,
                sequence=frames.streams[0],
                frame_ids=frames.frame_ids,
                tracker_options=tracker_options,
                lowres_options=options.keyframes.matching,
                highres_options=options.tracks.images,
                keyframe_options=options.keyframes.selection,
                salient_options=options.keyframes.features,
                track_options=options.tracks.propagation,
            )
            keyframes = keyframe_processor.load_cached(admission_metadata)
            sparse_track_builder = SparseTrackBuilder(
                scene_parser=source,
                paths=paths,
                force_recompute=self.force_recompute,
                repro_dir=self.repro_dir,
                tracker=tracker,
                tracker_options=tracker_options,
                keyframes=keyframes,
                track_options=options.tracks.propagation,
                highres_options=options.tracks.images,
                lowres_match_resolution=options.keyframes.matching.resolution,
                extended_options=options.loop_closure,
                device=self.device,
            )
            if keyframes is not None:
                keyframe_processor.complete_salient_features(keyframes)
            tracks = sparse_track_builder.load_complete()
            if tracks is None:
                provisional_salient_metadata = keyframe_cache.salient_feature_cache_metadata(
                    salient_options=options.keyframes.features,
                )
                candidates = keyframe_processor.select_candidates(provisional_salient_metadata)
                from vidmap.frontend.models.depth.da3_video import verify_da3_model_snapshot

                verification = preparation.submit(verify_da3_model_snapshot)
                keyframes, tracks = sparse_track_builder.admit_and_build_tracks(
                    candidates,
                    options.keyframes.selection if options.keyframes.selection.lookahead_pruning else None,
                    lambda admitted_ids: keyframe_processor.commit_keyframes(
                        admitted_ids,
                        candidates.forced_indices,
                        admission_metadata,
                    ),
                    complete_salients=keyframe_processor.complete_salient_features,
                )

            transitive_builder = TransitiveCorrespondenceBuilder(
                scene_parser=source,
                paths=paths,
                repro_dir=self.repro_dir,
                keyframes=keyframes,
                tracks=tracks,
            )
            transitive = transitive_builder.build()

            image_content = self.input_snapshot.image_content_fingerprint(keyframes.names)
            extended_match_builder = ExtendedMatchBuilder(
                keyframes=keyframes,
                scene_parser=source,
                paths=paths,
                force_recompute=self.force_recompute,
                repro_dir=self.repro_dir,
                tracker=tracker,
                tracks=tracks,
                transitive=transitive,
                highres_options=options.tracks.images,
                lowres_match_resolution=options.keyframes.matching.resolution,
                extended_options=options.loop_closure,
                point_budget=options.tracks.propagation.max_kps,
                image_content_fingerprint=image_content,
                device=self.device,
            )
            extended = extended_match_builder.build()

        if verification is not None:
            verification.result()
        depth_estimator = DepthEstimator(
            scene_parser=source,
            paths=paths,
            force_recompute=self.force_recompute,
            keyframes=keyframes,
            options=options.depth,
            calibration_enabled=prior_options.estimator == "da3",
            image_content_fingerprint=image_content,
            device=self.device,
        )
        depth = (
            depth_estimator.estimate(cache_full_depth_maps=True)
            if self.cache_full_depth_maps
            else depth_estimator.estimate()
        )
        if use_geocalib:
            geocalib = self._camera_prior_estimator(paths, keyframes.names, image_content).estimate()

        initialize_selected_camera_geometry(working_capture.rec, source.rec, keyframes.names, self.preparation_plan)
        if prior_options.estimator != "none":
            if prior_options.estimator == "da3":
                calibration_path, calibration_artifact = paths.depth_maps_path, depth.sampled
            else:
                calibration_path = (
                    paths.geocalib_batch_path
                    if prior_options.inference == "selected_batch"
                    else paths.geocalib_per_image_path
                )
                calibration_artifact = geocalib
            calibration = CalibrationResult.from_artifact(
                path=calibration_path,
                reconstruction=working_capture.rec,
                image_names=keyframes.names,
                options=prior_options,
                artifact=calibration_artifact,
            )

        logger.info("Features located in %s", paths.sparse_features_path)
        result = TrackingFrontendResult(
            paths=paths,
            artifacts=FrontendArtifacts(
                retrieval_pairs=extended.retrieval_pairs_artifact,
                sparse_features=tracks.sparse_features,
                sparse_matches=tracks.sparse_matches,
                extended_matches=extended.artifact,
                depth=depth.sampled,
                full_depth=depth.full,
            ),
            keyframe_sequence=keyframes.names,
            retrieval_pairs=extended.retrieval_pairs,
            tcorr=extended.tcorr,
            lc_masks=extended.lc_masks,
            capture=working_capture,
            calibration=calibration,
        )
        validate_tracking_result(result, geocalib_inference=prior_options.inference if use_geocalib else None)
        return result

    def _camera_prior_estimator(self, paths, keyframe_names, image_content):
        return CameraPriorEstimator(
            rgb_dir=self.input_snapshot.capture.rgb_dir,
            per_image_path=paths.geocalib_per_image_path,
            batch_path=paths.geocalib_batch_path,
            force_recompute=self.force_recompute,
            keyframe_names=keyframe_names,
            options=self.preparation_plan.camera_priors,
            image_content_fingerprint=image_content,
            device=self.device,
        )
