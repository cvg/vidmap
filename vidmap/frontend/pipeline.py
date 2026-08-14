from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass as result_dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from vidmap.datasets.base import DatasetParser
from vidmap.frontend.cache import CacheMetadataMismatch, CompleteArtifactContract, IncrementalArtifactContract
from vidmap.frontend.correspondences import ImagePair, immutable_array_mapping, validate_correspondence_alignment
from vidmap.frontend.depth_output import publish_full_depth_output
from vidmap.frontend.initial_reconstruction import build_initial_reconstruction
from vidmap.frontend.paths import FrontendPaths
from vidmap.frontend.preparation.camera_priors import apply_camera_priors
from vidmap.frontend.preparation.geometric_verification import GeometricVerifier
from vidmap.frontend.preparation.mapper_inputs import write_verified_mapper_inputs
from vidmap.frontend.preparation.tcorr_filtering import CorrespondenceFilter
from vidmap.mapper.inputs import MapperInputs
from vidmap.mapper.options import ReplayCacheOptions
from vidmap.mapper.replay.cache import ReplayCache
from vidmap.utils.loop_closure_masks import write_loop_closure_masks

if TYPE_CHECKING:
    from vidmap.configuration.config import FrontendConfig

__all__ = [
    "Frontend",
    "FrontendArtifacts",
    "TrackingFrontendResult",
    "validate_tracking_result",
]


@result_dataclass(frozen=True)
class FrontendArtifacts:
    """Cache identities and expected item plans exposed by tracking frontend."""

    track_pairs: CompleteArtifactContract
    retrieval_pairs: CompleteArtifactContract
    sparse_features: IncrementalArtifactContract
    sparse_matches: IncrementalArtifactContract
    extended_matches: IncrementalArtifactContract
    depth: IncrementalArtifactContract
    full_depth: IncrementalArtifactContract | None = None
    geocalib_batch: IncrementalArtifactContract | None = None


@result_dataclass(frozen=True)
class TrackingFrontendResult:
    """Certified in-memory frontend state used to prepare mapper inputs."""

    paths: FrontendPaths
    artifacts: FrontendArtifacts
    keyframe_sequence: tuple[str, ...]
    track_pairs: tuple[ImagePair, ...]
    retrieval_pairs: tuple[ImagePair, ...]
    tcorr: Mapping[ImagePair, Any]
    lc_masks: Mapping[ImagePair, Any]

    def __post_init__(self):
        if not isinstance(self.tcorr, Mapping) or not isinstance(self.lc_masks, Mapping):
            raise TypeError("tcorr and lc_masks must be mappings")
        validate_correspondence_alignment(self.tcorr, self.lc_masks)
        object.__setattr__(self, "keyframe_sequence", tuple(self.keyframe_sequence))
        object.__setattr__(self, "track_pairs", tuple(tuple(pair) for pair in self.track_pairs))
        object.__setattr__(self, "retrieval_pairs", tuple(tuple(pair) for pair in self.retrieval_pairs))
        object.__setattr__(self, "tcorr", immutable_array_mapping(self.tcorr))
        object.__setattr__(self, "lc_masks", immutable_array_mapping(self.lc_masks))


def validate_tracking_result(result: TrackingFrontendResult, *, geocalib_enabled: bool) -> None:
    """Reject inconsistent optional artifacts at the tracking boundary."""
    if len(result.keyframe_sequence) < 2 or not result.track_pairs:
        raise CacheMetadataMismatch("Frontend requires at least two keyframes and one track pair")
    if result.artifacts.depth is None:
        raise CacheMetadataMismatch("Depth frontend is enabled but its certified artifact is unavailable")
    geocalib_values = (
        result.paths.geocalib_per_image_path,
        result.paths.geocalib_batch_path,
        result.artifacts.geocalib_batch,
    )
    if geocalib_enabled and any(value is None for value in geocalib_values):
        raise CacheMetadataMismatch("GeoCalib is enabled but its certified artifacts are unavailable")
    if not geocalib_enabled and any(value is not None for value in geocalib_values):
        raise CacheMetadataMismatch("GeoCalib is disabled but geocalib artifacts were exposed")


class Frontend:
    """Build tracking artifacts and publish the complete mapper boundary."""

    def __init__(
        self,
        *,
        conf: FrontendConfig,
        replay_cache: ReplayCacheOptions,
        sample_name: str,
        cache_dir: Path,
        sfm_outputs_dir: Path,
        scene_parser: DatasetParser,
        reference_image_names: Sequence[str],
        deterministic_frontend: bool = False,
        pre_geom_repro_dir: str | Path | None = None,
        force_recompute: bool = False,
        cache_full_depth_maps: bool = False,
        frontend_tag: str,
        namespace_cache_by_config: bool = True,
        mapper_inputs_dir: Path | None = None,
    ):
        from vidmap.configuration.config import FrontendConfig

        if not isinstance(conf, FrontendConfig):
            raise TypeError(f"Expected FrontendConfig, got {type(conf).__name__}")
        if not sample_name:
            raise ValueError("sample_name is required for frontend")
        self.options = conf
        self.preparation = conf.preparation
        self.use_geocalib = conf.use_geocalib
        self.view_graph_calibration = conf.view_graph_calibration
        self.sample_name = str(sample_name)
        self.cache_dir = Path(cache_dir)
        self.outputs_dir = Path(sfm_outputs_dir)
        if not frontend_tag:
            raise ValueError("frontend_tag must be the identity derived from the frontend config name")
        self.frontend_tag = str(frontend_tag)
        self.namespace_cache_by_config = bool(namespace_cache_by_config)
        self.mapper_inputs_dir = (
            self.outputs_dir / "mapper_inputs" if mapper_inputs_dir is None else Path(mapper_inputs_dir)
        )
        self.scene_parser = scene_parser
        self.reference_image_names = tuple(reference_image_names)
        self.deterministic = bool(deterministic_frontend)
        self.repro_dir = Path(pre_geom_repro_dir).expanduser() if pre_geom_repro_dir is not None else None
        self.force_recompute = bool(force_recompute)
        self.cache_full_depth_maps = bool(cache_full_depth_maps)
        self.replay = ReplayCache(replay_cache, self.outputs_dir)

    def run(self, *, frontend_identity: Mapping[str, object]) -> MapperInputs:
        """Run the complete frontend and mapper-input publication workflow."""
        tracking, filtered, verification = self._execute_verified_boundary(pre_geom_db_stop=False)
        mapper_inputs = write_verified_mapper_inputs(
            self.mapper_inputs_dir,
            tracking,
            filtered,
            verification,
            frontend_identity=frontend_identity,
        )
        publish_full_depth_output(
            mapper_inputs.directory.parent,
            tracking.paths.full_depth_maps_path,
            tracking.artifacts.full_depth,
        )
        verification.database_path.unlink()
        return mapper_inputs

    def run_pre_geom(self) -> Path:
        """Run frontend through deterministic pre-GV database construction."""
        _, _, verification = self._execute_verified_boundary(pre_geom_db_stop=True)
        return verification.database_path

    def _execute_verified_boundary(self, *, pre_geom_db_stop: bool):
        """Execute the shared tracking-to-database boundary in its canonical order."""
        # Tracking owns all learned predictions and correspondence artifacts.
        tracking = self.run_tracking()

        # Geometry preparation starts from the source cameras and applies any
        # calibrated or learned priors before correspondence filtering and GV.
        reconstruction = build_initial_reconstruction(
            self.scene_parser, reference_image_names=list(self.reference_image_names)
        )
        apply_camera_priors(
            use_geocalib=self.use_geocalib,
            view_graph_calibration=self.view_graph_calibration,
            geocalib_batch_path=tracking.paths.geocalib_batch_path,
            geocalib_batch_artifact=tracking.artifacts.geocalib_batch,
            source_reconstruction=self.scene_parser.rec,
            reconstruction=reconstruction,
        )

        correspondence_filter = CorrespondenceFilter(
            options=self.preparation.lc,
            repro_dir=self.repro_dir,
        )
        filtered = correspondence_filter.filter(
            tcorr=tracking.tcorr,
            lc_masks=tracking.lc_masks,
            retrieval_pairs=tracking.retrieval_pairs,
            extended_matches_path=tracking.paths.extended_matches_path,
            extended_matches_artifact=tracking.artifacts.extended_matches,
        )

        # Persist the masks aligned with the filtered matches before geometric
        # verification publishes the finalized mapper database.
        sparse_matches_path = tracking.paths.sparse_matches_path
        lc_masks_path = sparse_matches_path.with_name(f"lc_masks-{sparse_matches_path.stem}.h5")
        write_loop_closure_masks(filtered.lc_masks, lc_masks_path)

        geometric_verifier = GeometricVerifier(
            options=self.preparation.geom_verif,
            database_path=self.outputs_dir / ("database_pre_geom.db" if pre_geom_db_stop else "database_complete.db"),
            replay=self.replay,
            repro_dir=self.repro_dir,
            view_graph_calibration=self.view_graph_calibration,
            pre_geom_db_stop=pre_geom_db_stop,
        )
        verification = geometric_verifier.verify(tracking, reconstruction, filtered.tcorr)
        return tracking, filtered, verification

    def run_tracking(self) -> TrackingFrontendResult:
        """Run tracking, depth, and optional GeoCalib stages."""
        from vidmap.frontend.tracking.composition import TrackingPipeline

        tracking_pipeline = TrackingPipeline(
            options=self.options,
            use_geocalib=self.use_geocalib,
            sample_name=self.sample_name,
            cache_dir=self.cache_dir,
            cache_namespace=self.frontend_tag if self.namespace_cache_by_config else None,
            scene_parser=self.scene_parser,
            force_recompute=self.force_recompute,
            cache_full_depth_maps=self.cache_full_depth_maps,
            deterministic=self.deterministic,
            repro_dir=self.repro_dir,
        )
        return tracking_pipeline.run()
