"""Concrete composition of frontend's tracking-owned stages."""

import logging
from pathlib import Path

from vidmap.frontend.cache import fingerprint, ordered_files_fingerprint
from vidmap.frontend.depth import DepthEstimator
from vidmap.frontend.geocalib import CameraPriorEstimator
from vidmap.frontend.image_dataset import FrameSequence
from vidmap.frontend.keyframes import cache as keyframe_cache
from vidmap.frontend.keyframes.processing import KeyframeProcessor
from vidmap.frontend.loop_closure.extended_matches import ExtendedMatchBuilder
from vidmap.frontend.models.romav2 import create_lazy_romav2_tracker, romav2_cache_identity
from vidmap.frontend.paths import build_frontend_paths
from vidmap.frontend.tracking.sparse_tracks import SparseTrackBuilder
from vidmap.frontend.tracking.transitive import TransitiveCorrespondenceBuilder

logger = logging.getLogger(__name__)


class TrackingPipeline:
    """Run the visible keyframe-to-certified-tracking stage sequence."""

    def __init__(
        self,
        *,
        options,
        use_geocalib,
        sample_name,
        cache_dir,
        cache_namespace,
        scene_parser,
        force_recompute,
        deterministic,
        repro_dir,
        cache_full_depth_maps=False,
    ):
        self.options = options
        self.use_geocalib = use_geocalib
        self.sample_name = sample_name
        self.cache_dir = Path(cache_dir)
        self.cache_namespace = None if cache_namespace is None else str(cache_namespace)
        self.scene_parser = scene_parser
        self.force_recompute = force_recompute
        self.cache_full_depth_maps = bool(cache_full_depth_maps)
        self.deterministic = deterministic
        self.repro_dir = repro_dir

    def run(self):
        """Run model stages, then depth and optional camera calibration."""
        from vidmap.frontend.pipeline import FrontendArtifacts, TrackingFrontendResult, validate_tracking_result

        options = self.options
        tracker_options = options.roma

        # Resolve every cache path from the frontend identity so all stages
        # operate on one immutable artifact plan.
        paths = build_frontend_paths(
            cache_dir=self.cache_dir,
            sample_name=self.sample_name,
            depth_model_name=options.depth.depth_model,
            use_geocalib=self.use_geocalib,
            frontend_tag=None,
            config_name=self.cache_namespace,
            cache_variant="cache-" + fingerprint({"tracker": romav2_cache_identity(tracker_options)})[:16],
        )
        paths.track_pairs_path.parent.mkdir(exist_ok=True, parents=True)
        logger.info("Input sparse features: %s", paths.sparse_features_path)
        logger.info("Depth maps: %s", paths.depth_maps_path)
        if self.use_geocalib:
            logger.info("Geo-calibration (per-image): %s", paths.geocalib_per_image_path)
            logger.info("Geo-calibration (batch): %s", paths.geocalib_batch_path)
        frames = FrameSequence.from_scene(self.scene_parser)

        with create_lazy_romav2_tracker(tracker_options) as tracker:
            keyframe_processor = KeyframeProcessor(
                scene_parser=self.scene_parser,
                frames=frames,
                paths=paths,
                force_recompute=self.force_recompute,
                repro_dir=self.repro_dir,
                tracker=tracker,
                lowres_options=options.keyframes.matching,
                keyframe_options=options.keyframes.selection,
                salient_options=options.keyframes.features,
            )
            track_pairs_metadata = keyframe_cache.admitted_track_pairs_cache_metadata(
                scene_parser=self.scene_parser,
                sequence=frames.names,
                timestamps=frames.timestamps,
                tracker_options=tracker_options,
                lowres_options=options.keyframes.matching,
                highres_options=options.tracks.images,
                keyframe_options=options.keyframes.selection,
                salient_options=options.keyframes.features,
            )
            keyframes = keyframe_processor.load_cached(track_pairs_metadata)
            sparse_track_builder = SparseTrackBuilder(
                scene_parser=self.scene_parser,
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
            )
            tracks = sparse_track_builder.load_complete()
            if tracks is None:
                provisional_salient_metadata = keyframe_cache.salient_feature_cache_metadata(
                    salient_options=options.keyframes.features,
                    sequence=frames.names,
                    timestamps=frames.timestamps,
                    track_pairs_metadata=track_pairs_metadata,
                )
                candidates = keyframe_processor.select_candidates(provisional_salient_metadata)
                keyframes, tracks = sparse_track_builder.admit_and_build_tracks(
                    candidates,
                    options.keyframes.selection if options.keyframes.selection.lookahead_pruning else None,
                    lambda admitted_ids: keyframe_processor.commit_keyframes(
                        admitted_ids,
                        candidates.forced_indices,
                        track_pairs_metadata,
                    ),
                )

            transitive_builder = TransitiveCorrespondenceBuilder(
                scene_parser=self.scene_parser,
                paths=paths,
                repro_dir=self.repro_dir,
                keyframes=keyframes,
                tracks=tracks,
            )
            transitive = transitive_builder.build()

            image_content = ordered_files_fingerprint(self.scene_parser.rgb_dir, keyframes.names)
            extended_match_builder = ExtendedMatchBuilder(
                scene_parser=self.scene_parser,
                paths=paths,
                force_recompute=self.force_recompute,
                repro_dir=self.repro_dir,
                tracker=tracker,
                tracks=tracks,
                transitive=transitive,
                highres_options=options.tracks.images,
                lowres_match_resolution=options.keyframes.matching.resolution,
                extended_options=options.loop_closure,
                image_content_fingerprint=image_content,
            )
            extended = extended_match_builder.build()

        depth_estimator = DepthEstimator(
            scene_parser=self.scene_parser,
            paths=paths,
            force_recompute=self.force_recompute,
            keyframes=keyframes,
            options=options.depth,
            image_content_fingerprint=image_content,
        )
        depth = (
            depth_estimator.estimate(cache_full_depth_maps=True)
            if self.cache_full_depth_maps
            else depth_estimator.estimate()
        )
        geocalib = self._estimate_camera_priors(paths, keyframes.names, image_content)

        logger.info("Features located in %s", paths.sparse_features_path)
        result = TrackingFrontendResult(
            paths=paths,
            artifacts=FrontendArtifacts(
                track_pairs=tracks.track_pairs_artifact,
                retrieval_pairs=extended.retrieval_pairs_artifact,
                sparse_features=tracks.sparse_features,
                sparse_matches=tracks.sparse_matches,
                extended_matches=extended.artifact,
                depth=depth.sampled,
                full_depth=depth.full,
                geocalib_batch=geocalib,
            ),
            keyframe_sequence=keyframes.names,
            track_pairs=tracks.track_pairs,
            retrieval_pairs=extended.retrieval_pairs,
            tcorr=extended.tcorr,
            lc_masks=extended.lc_masks,
        )
        validate_tracking_result(result, geocalib_enabled=self.use_geocalib)
        return result

    def _estimate_camera_priors(self, paths, keyframe_names, image_content):
        if not self.use_geocalib:
            return None
        if paths.geocalib_per_image_path is None or paths.geocalib_batch_path is None:
            raise ValueError("GeoCalib frontend paths are required when GeoCalib is enabled")
        return CameraPriorEstimator(
            rgb_dir=self.scene_parser.rgb_dir,
            per_image_path=paths.geocalib_per_image_path,
            batch_path=paths.geocalib_batch_path,
            force_recompute=self.force_recompute,
            keyframe_names=keyframe_names,
            options=self.options.camera_priors,
            image_content_fingerprint=image_content,
        ).estimate()
