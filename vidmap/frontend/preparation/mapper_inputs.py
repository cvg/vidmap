"""Validate frontend artifacts and write the mapper-input directory."""

from __future__ import annotations

import json
import shutil
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, AbstractSet

from vidmap.frontend.cache import CacheMetadataMismatch, read_pair_artifact, validate_incremental_cache
from vidmap.mapper.inputs import (
    CANONICAL_NAMES,
    DATABASE_NAME,
    DEPTH_MAPS_NAME,
    GEOCALIB_BATCH_NAME,
    LC_MASKS_NAME,
    MANIFEST_NAME,
    REQUIRED_PAYLOAD_NAMES,
    TRACK_PAIRS_NAME,
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
    from vidmap.frontend.preparation.tcorr_filtering import FilteredCorrespondences


def write_mapper_inputs(
    directory: Path,
    sources: Mapping[str, Path],
    lc_masks: Mapping[tuple[str, str], object] | None,
    vgc_filtered_pairs: AbstractSet[tuple[str, str]] | None,
    *,
    frontend_identity: Mapping[str, object],
) -> MapperInputs:
    """Replace the directory with one self-contained mapper-input generation."""
    directory = resolve_mapper_inputs_directory(directory)
    source_names = set(sources)
    generated_names = {LC_MASKS_NAME, VGC_FILTERED_PAIRS_NAME, MANIFEST_NAME}
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
    write_mapper_inputs_manifest(directory, frontend_identity=frontend_identity)
    return MapperInputs.from_directory(directory, expected_identity=frontend_identity)


def write_verified_mapper_inputs(
    directory: Path,
    state: TrackingFrontendResult,
    filtered: FilteredCorrespondences,
    verification: GeometricVerificationResult,
    *,
    frontend_identity: Mapping[str, object],
) -> MapperInputs:
    """Validate the current frontend generation, then write its mapper inputs."""
    paths = state.paths
    depth_contract = state.artifacts.depth
    if depth_contract is None:
        raise CacheMetadataMismatch("Depth mapper input has no frontend cache contract")

    verification.database_provenance.validate(verification.database_path)
    track_pairs = read_pair_artifact(
        paths.track_pairs_path,
        state.artifacts.track_pairs.metadata,
    )
    if tuple(track_pairs) != state.track_pairs:
        raise CacheMetadataMismatch("Track-pair cache changed before mapper-input publication")
    validate_incremental_cache(
        paths.depth_maps_path,
        depth_contract.metadata,
        depth_contract.expected_items,
    )

    sources = {
        DATABASE_NAME: verification.database_path,
        TRACK_PAIRS_NAME: paths.track_pairs_path,
        DEPTH_MAPS_NAME: paths.depth_maps_path,
    }
    if paths.geocalib_batch_path is not None:
        geocalib_contract = state.artifacts.geocalib_batch
        if geocalib_contract is None:
            raise CacheMetadataMismatch("GeoCalib mapper input has no frontend cache contract")
        validate_incremental_cache(
            paths.geocalib_batch_path,
            geocalib_contract.metadata,
            geocalib_contract.expected_items,
        )
        sources[GEOCALIB_BATCH_NAME] = paths.geocalib_batch_path

    return write_mapper_inputs(
        directory,
        sources,
        filtered.lc_masks,
        verification.vgc_filtered_pairs,
        frontend_identity=frontend_identity,
    )
