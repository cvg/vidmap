"""Validate frontend artifacts and write the mapper-input directory."""

from __future__ import annotations

import json
import shutil
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, AbstractSet

from vidmap.calibration import CalibrationEvidence
from vidmap.frontend.cache import CacheMetadataMismatch, validate_incremental_cache
from vidmap.mapper.inputs import (
    CANONICAL_NAMES,
    DATABASE_NAME,
    DEPTH_MAPS_NAME,
    LC_MASKS_NAME,
    MANIFEST_NAME,
    REQUIRED_PAYLOAD_NAMES,
    VGC_FILTERED_PAIRS_NAME,
    MapperInputs,
    require_finalized_sqlite,
    resolve_mapper_inputs_directory,
    write_mapper_inputs_manifest,
)
from vidmap.utils.loop_closure_masks import write_loop_closure_masks

if TYPE_CHECKING:
    from vidmap.frontend.pipeline import TrackingFrontendResult
    from vidmap.frontend.preparation.geometric_verification import GeometricVerificationResult


def write_mapper_inputs(
    directory: Path,
    sources: Mapping[str, Path],
    lc_masks: Mapping[tuple[str, str], object] | None,
    vgc_filtered_pairs: AbstractSet[tuple[str, str]] | None = None,
    *,
    frontend_identity: Mapping[str, object],
    frame_sequence: dict,
    calibration: CalibrationEvidence | None = None,
) -> MapperInputs:
    """Replace the directory with one self-contained mapper-input generation."""
    directory = resolve_mapper_inputs_directory(directory)
    source_names = set(sources)
    generated_names = {VGC_FILTERED_PAIRS_NAME, MANIFEST_NAME}
    unexpected = source_names - (CANONICAL_NAMES - generated_names)
    if unexpected:
        raise ValueError(f"Unexpected mapper-input names: {sorted(unexpected)}")

    provided_names = source_names | ({LC_MASKS_NAME} if lc_masks is not None else set())
    missing = REQUIRED_PAYLOAD_NAMES - provided_names
    if missing:
        raise FileNotFoundError(f"Missing required mapper inputs: {sorted(missing)}")

    source_paths = {name: Path(source) for name, source in sources.items()}
    for name, source in source_paths.items():
        if not source.is_file():
            raise FileNotFoundError(f"Mapper input source not found: {source}")
        if name == DATABASE_NAME:
            require_finalized_sqlite(source)

    if lc_masks is not None and LC_MASKS_NAME in source_paths:
        raise ValueError("Loop-closure masks must have exactly one source")
    if directory.exists():
        shutil.rmtree(directory)
    directory.mkdir(parents=True)
    for name, source in source_paths.items():
        shutil.copy2(source, directory / name)
    if lc_masks is not None:
        write_loop_closure_masks(lc_masks, directory / LC_MASKS_NAME)
    if vgc_filtered_pairs is not None:
        (directory / VGC_FILTERED_PAIRS_NAME).write_text(
            json.dumps(
                {"pairs": [list(pair) for pair in sorted(vgc_filtered_pairs)]},
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
    write_mapper_inputs_manifest(
        directory,
        frontend_identity=frontend_identity,
        frame_sequence=frame_sequence,
        calibration=calibration,
    )
    result = MapperInputs.from_directory(directory, expected_identity=frontend_identity)
    result.validate()
    return result


def write_verified_mapper_inputs(
    directory: Path,
    state: TrackingFrontendResult,
    verification: GeometricVerificationResult,
    *,
    frontend_identity: Mapping[str, object],
) -> MapperInputs:
    """Validate the current frontend generation, then write its mapper inputs."""
    paths = state.paths
    depth_contract = state.artifacts.depth
    if depth_contract is None:
        raise CacheMetadataMismatch("Depth mapper input has no frontend cache contract")

    validate_incremental_cache(
        paths.depth_maps_path,
        depth_contract.metadata,
        depth_contract.expected_items,
    )

    sources = {
        DATABASE_NAME: verification.database_path,
        LC_MASKS_NAME: paths.lc_masks_path,
        DEPTH_MAPS_NAME: paths.depth_maps_path,
    }
    calibration = state.calibration
    if calibration is not None and calibration.evidence.estimator == "geocalib":
        from vidmap.mapper.inputs.snapshot import calibration_artifact_name

        contract = calibration.artifact
        validate_incremental_cache(calibration.path, contract.metadata, contract.expected_items)
        filename = calibration_artifact_name(calibration.evidence.estimator, calibration.evidence.inference)
        sources[filename] = calibration.path

    return write_mapper_inputs(
        directory,
        sources,
        None,
        frontend_identity=frontend_identity,
        frame_sequence=verification.frame_sequence,
        calibration=None if calibration is None else calibration.evidence,
    )


def prepare_frame_sequence(capture, allocated):
    """Carry allocated frame IDs forward in capture order."""

    frames = allocated.frames
    ids = [fid for fid in capture.frame_ids if fid in frames]
    if set(ids) != set(frames):
        raise ValueError("Allocated frame inventory disagrees with selected capture")
    times = None if capture.frame_timestamps_ns is None else [capture.frame_timestamps_ns[fid] for fid in ids]
    return {"frame_ids": ids, "timestamps_ns": times}
