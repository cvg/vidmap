from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass as result_dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from vidmap.datasets.capture import CaptureSnapshot, PreparedCapture, copy_unposed_reconstruction
from vidmap.frontend.cache import CacheMetadataMismatch, CompleteArtifactContract, IncrementalArtifactContract
from vidmap.frontend.correspondences import ImagePair, immutable_array_mapping, validate_correspondence_alignment
from vidmap.frontend.depth_output import publish_full_depth_output
from vidmap.frontend.paths import FrontendPaths
from vidmap.frontend.preparation.camera_priors import CalibrationResult, apply_calibration
from vidmap.frontend.preparation.cameras import CameraPreparationPlan, prepare_capture, resolve_camera_preparation
from vidmap.frontend.preparation.geometric_verification import GeometricVerifier
from vidmap.frontend.preparation.mapper_inputs import write_verified_mapper_inputs
from vidmap.frontend.preparation.tcorr_filtering import CorrespondenceFilter
from vidmap.mapper.inputs import MapperInputs
from vidmap.mapper.options import ReplayCacheOptions
from vidmap.mapper.replay.cache import ReplayCache
from vidmap.utils.device import resolve_device
from vidmap.utils.loop_closure_masks import write_loop_closure_masks

if TYPE_CHECKING:
    import torch

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

    retrieval_pairs: CompleteArtifactContract
    sparse_features: IncrementalArtifactContract
    sparse_matches: IncrementalArtifactContract
    extended_matches: IncrementalArtifactContract
    depth: IncrementalArtifactContract
    full_depth: IncrementalArtifactContract | None = None


@result_dataclass(frozen=True)
class TrackingFrontendResult:
    """Certified in-memory frontend state used to prepare mapper inputs."""

    paths: FrontendPaths
    artifacts: FrontendArtifacts
    keyframe_sequence: tuple[str, ...]
    retrieval_pairs: tuple[ImagePair, ...]
    tcorr: Mapping[ImagePair, Any]
    lc_masks: Mapping[ImagePair, Any]
    capture: PreparedCapture
    calibration: CalibrationResult | None = None

    def __post_init__(self):
        if not isinstance(self.tcorr, Mapping) or not isinstance(self.lc_masks, Mapping):
            raise TypeError("tcorr and lc_masks must be mappings")
        validate_correspondence_alignment(self.tcorr, self.lc_masks)
        object.__setattr__(self, "keyframe_sequence", tuple(self.keyframe_sequence))
        object.__setattr__(self, "retrieval_pairs", tuple(tuple(pair) for pair in self.retrieval_pairs))
        object.__setattr__(self, "tcorr", immutable_array_mapping(self.tcorr))
        object.__setattr__(self, "lc_masks", immutable_array_mapping(self.lc_masks))


def validate_tracking_result(result: TrackingFrontendResult, *, geocalib_inference: str | None) -> None:
    """Reject inconsistent optional artifacts at the tracking boundary."""
    if len(result.keyframe_sequence) < 2 or not result.artifacts.sparse_matches.expected_items:
        raise CacheMetadataMismatch("Frontend requires at least two keyframes and one track pair")
    if result.artifacts.depth is None:
        raise CacheMetadataMismatch("Depth frontend is enabled but its certified artifact is unavailable")
    calibration = result.calibration
    if geocalib_inference is not None:
        if calibration is None or calibration.evidence.estimator != "geocalib" or calibration.artifact is None:
            raise CacheMetadataMismatch("GeoCalib is enabled but its certified artifact is unavailable")
        if calibration.evidence.inference != geocalib_inference:
            raise CacheMetadataMismatch("GeoCalib inference disagrees with its calibration result")
    elif calibration is not None and calibration.evidence.estimator == "geocalib":
        raise CacheMetadataMismatch("GeoCalib is disabled but a calibration result was exposed")


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
        reference_image_names: Sequence[str],
        input_snapshot: CaptureSnapshot,
        preparation_plan: CameraPreparationPlan | None = None,
        deterministic_frontend: bool = False,
        pre_geom_repro_dir: str | Path | None = None,
        force_recompute: bool = False,
        cache_full_depth_maps: bool = False,
        frontend_tag: str,
        namespace_cache_by_config: bool = True,
        mapper_inputs_dir: Path | None = None,
        device: str | torch.device | None = None,
    ):
        from vidmap.configuration.config import FrontendConfig

        if not isinstance(conf, FrontendConfig):
            raise TypeError(f"Expected FrontendConfig, got {type(conf).__name__}")
        if not sample_name:
            raise ValueError("sample_name is required for frontend")
        self.device = resolve_device(device)
        self.options = conf
        self.preparation = conf.preparation
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
        self.scene_parser = input_snapshot.capture
        self.preparation_plan = (
            resolve_camera_preparation(self.scene_parser, conf) if preparation_plan is None else preparation_plan
        )
        self.prepared_capture = prepare_capture(self.scene_parser, self.preparation_plan)
        self.estimate_intrinsics = self.preparation_plan.camera_priors.initialization == "predicted"
        self.input_snapshot = input_snapshot
        self.reference_image_names = tuple(reference_image_names)
        self.deterministic = bool(deterministic_frontend)
        self.repro_dir = Path(pre_geom_repro_dir).expanduser() if pre_geom_repro_dir is not None else None
        self.force_recompute = bool(force_recompute)
        self.cache_full_depth_maps = bool(cache_full_depth_maps)
        self.replay = ReplayCache(replay_cache, self.outputs_dir)

    def run(self, *, frontend_identity: Mapping[str, object]) -> MapperInputs:
        """Run the complete frontend and mapper-input publication workflow."""
        tracking, verification = self._execute_verified_boundary(pre_geom_db_stop=False)
        mapper_inputs = write_verified_mapper_inputs(
            self.mapper_inputs_dir,
            tracking,
            verification,
            frontend_identity=frontend_identity,
            panorama=self.scene_parser.panorama_metadata(),
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
        _, verification = self._execute_verified_boundary(pre_geom_db_stop=True)
        return verification.database_path

    def _execute_verified_boundary(self, *, pre_geom_db_stop: bool):
        """Execute the shared tracking-to-database boundary in its canonical order."""
        # Tracking owns all learned predictions and correspondence artifacts.
        tracking = self.run_tracking()

        reconstruction = tracking.capture.rec
        if self.estimate_intrinsics:
            reconstruction = copy_unposed_reconstruction(reconstruction, tracking.keyframe_sequence)
            if tracking.calibration is None:
                raise ValueError("Predicted initialization requires calibration results")
            apply_calibration(tracking.calibration, reconstruction)

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
        write_loop_closure_masks(filtered.lc_masks, tracking.paths.lc_masks_path)

        geometric_verifier = GeometricVerifier(
            capture=tracking.capture,
            options=self.preparation.geom_verif,
            database_path=self.outputs_dir / ("database_pre_geom.db" if pre_geom_db_stop else "database_complete.db"),
            replay=self.replay,
            repro_dir=self.repro_dir,
            estimate_intrinsics=self.estimate_intrinsics,
            pre_geom_db_stop=pre_geom_db_stop,
        )
        verification = geometric_verifier.verify(tracking, reconstruction, filtered.tcorr)
        return tracking, verification

    def run_tracking(self) -> TrackingFrontendResult:
        """Run tracking, depth, and optional GeoCalib stages."""
        from vidmap.frontend.tracking.composition import TrackingPipeline

        tracking_pipeline = TrackingPipeline(
            options=self.options,
            sample_name=self.sample_name,
            cache_dir=self.cache_dir,
            cache_namespace=self.frontend_tag if self.namespace_cache_by_config else None,
            input_snapshot=self.input_snapshot,
            preparation_plan=self.preparation_plan,
            prepared_capture=self.prepared_capture,
            force_recompute=self.force_recompute,
            cache_full_depth_maps=self.cache_full_depth_maps,
            deterministic=self.deterministic,
            repro_dir=self.repro_dir,
            device=self.device,
        )
        return tracking_pipeline.run()
