"""Publication of optional full-depth frontend output."""

from __future__ import annotations

import os
import shutil
import uuid
from pathlib import Path

from vidmap.depth_artifacts import FULL_DEPTH_MAPS_NAME
from vidmap.frontend.cache import CacheMetadataMismatch, IncrementalArtifactContract, validate_incremental_cache


def publish_full_depth_output(
    output_dir: Path,
    source: Path | None,
    artifact: IncrementalArtifactContract | None,
) -> Path | None:
    """Atomically publish the validated full-depth output for one frontend generation."""
    output_dir = Path(output_dir)
    destination = output_dir / FULL_DEPTH_MAPS_NAME
    if artifact is None:
        destination.unlink(missing_ok=True)
        return None
    if source is None:
        raise CacheMetadataMismatch("Full depth output has an frontend cache contract but no source")
    validate_incremental_cache(source, artifact.metadata, artifact.expected_items)
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    try:
        shutil.copy2(source, temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination
