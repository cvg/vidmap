"""Two-stage VGC geometric verification.

Creates the per-run database (``database_complete.db``), runs geometric
verification either in a single pass or in two passes (strict for VGC and
relaxed for the rest of the pipeline).

"""

from __future__ import annotations

import logging
import os
import sqlite3
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pycolmap

from vidmap.frontend.colmap_database import create_database_from_frontend, run_geometric_verification
from vidmap.frontend.correspondences import ImagePair
from vidmap.frontend.options.preparation import GeomVerifOptions
from vidmap.mapper.inputs import FileProvenance, require_finalized_sqlite, sqlite_sidecar_paths
from vidmap.mapper.replay.evidence.stages import database_file_summary
from vidmap.repro.frontend import write_sqlite_summary_artifact
from vidmap.utils.logging import progress_bars_enabled

if TYPE_CHECKING:
    from vidmap.frontend.pipeline import TrackingFrontendResult
    from vidmap.mapper.replay.cache import ReplayCache


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GeometricVerificationResult:
    """Exact finalized database path and optional strict-VGC exclusions."""

    database_path: Path
    database_provenance: FileProvenance
    vgc_filtered_pairs: frozenset[ImagePair] | None


def _fsync_file(path: Path) -> None:
    with path.open("rb") as stream:
        os.fsync(stream.fileno())


def _fsync_directory(directory: Path) -> None:
    descriptor = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _finalize_sqlite_generation(database_path: Path) -> None:
    """Checkpoint a complete WAL generation and retire its sidecars durably."""
    database_path = Path(database_path)
    journal_path, _, wal_path = sqlite_sidecar_paths(database_path)
    if journal_path.exists() and journal_path.stat().st_size:
        raise RuntimeError(f"Cannot finalize SQLite database with a nonempty rollback journal: {journal_path}")
    if wal_path.exists() and wal_path.stat().st_size:
        with sqlite3.connect(database_path) as connection:
            result = connection.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
        if result is None or result[0] != 0 or result[1] != result[2]:
            raise RuntimeError(f"SQLite WAL checkpoint did not complete for {database_path}: {result}")
    _fsync_file(database_path)
    for sidecar in sqlite_sidecar_paths(database_path):
        sidecar.unlink(missing_ok=True)
    _fsync_directory(database_path.parent)
    require_finalized_sqlite(database_path)


def _prepare_sqlite_destination(database_path: Path) -> None:
    """Finalize the current SQLite generation before publishing a replacement."""
    database_path = Path(database_path)
    if database_path.exists():
        _finalize_sqlite_generation(database_path)
        return

    journal_path, shm_path, wal_path = sqlite_sidecar_paths(database_path)
    unsafe_orphans = [path for path in (journal_path, wal_path) if path.exists() and path.stat().st_size]
    if unsafe_orphans:
        formatted = ", ".join(str(path) for path in unsafe_orphans)
        raise RuntimeError(f"Cannot replace missing SQLite database with nonempty orphan sidecars: {formatted}")
    for sidecar in (journal_path, shm_path, wal_path):
        sidecar.unlink(missing_ok=True)
    _fsync_directory(database_path.parent)


def _verification_options(options, max_H_inlier_ratio):
    return {
        "max_H_inlier_ratio": max_H_inlier_ratio,
        "min_num_inliers": options.min_num_inliers,
        "ransac": {
            "max_num_trials": options.ransac_max_num_trials,
            "min_inlier_ratio": options.ransac_min_inlier_ratio,
            "max_error": options.ransac_max_error,
        },
    }


class GeometricVerifier:
    """Own database construction, replay, and geometric verification."""

    def __init__(
        self,
        *,
        options: GeomVerifOptions,
        database_path: Path,
        replay: ReplayCache,
        repro_dir: Path | None,
        view_graph_calibration: bool,
        pre_geom_db_stop: bool = False,
    ):
        self.options = options
        self.database_path = Path(database_path)
        self.replay = replay
        self.repro_dir = repro_dir
        self.view_graph_calibration = view_graph_calibration
        self.pre_geom_db_stop = pre_geom_db_stop

    def verify(
        self,
        state: TrackingFrontendResult,
        initial_reconstruction: pycolmap.Reconstruction,
        tcorr: Mapping[ImagePair, Any],
    ) -> GeometricVerificationResult:
        """Create and verify the database, returning its exact output."""
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{self.database_path.name}-",
            suffix=".tmp",
            dir=self.database_path.parent,
        )
        os.close(descriptor)
        temporary_path = Path(temporary_name)
        temporary_path.unlink()
        try:
            vgc_filtered_pairs = _verify_database(
                self.options,
                state,
                initial_reconstruction,
                tcorr,
                view_graph_calibration=self.view_graph_calibration,
                database_path=temporary_path,
                pre_geom_db_stop=self.pre_geom_db_stop,
            )
            _finalize_sqlite_generation(temporary_path)
            _prepare_sqlite_destination(self.database_path)
            os.replace(temporary_path, self.database_path)
            _fsync_file(self.database_path)
            _fsync_directory(self.database_path.parent)
            require_finalized_sqlite(self.database_path)
        finally:
            for path in (temporary_path, *sqlite_sidecar_paths(temporary_path)):
                path.unlink(missing_ok=True)
        if self.repro_dir is not None and self.pre_geom_db_stop:
            write_sqlite_summary_artifact(
                self.repro_dir / "stage1_database_pre_geom_summary.json",
                self.database_path,
                label="database_pre_geom",
            )
        if not self.pre_geom_db_stop and self.replay.write_enabled("database"):
            self.replay.write_json(
                "database",
                "summary.json",
                database_file_summary(self.database_path),
            )
        provenance = FileProvenance.from_path(self.database_path)
        return GeometricVerificationResult(self.database_path, provenance, vgc_filtered_pairs)


def _verify_database(
    options: GeomVerifOptions,
    state: TrackingFrontendResult,
    initial_reconstruction: pycolmap.Reconstruction,
    tcorr: Mapping[ImagePair, Any],
    *,
    view_graph_calibration: bool,
    database_path: Path,
    pre_geom_db_stop: bool,
) -> frozenset[ImagePair] | None:
    if pre_geom_db_stop:
        create_database_from_frontend(
            state,
            initial_reconstruction,
            database_path,
            view_graph_calibration=view_graph_calibration,
            matches=tcorr,
        )

        return None

    create_database_from_frontend(
        state,
        initial_reconstruction,
        database_path,
        view_graph_calibration=view_graph_calibration,
        matches=tcorr,
    )

    geom_verif_options = _verification_options(options, options.max_H_inlier_ratio)
    run_geometric_verification(
        database_path,
        list(tcorr),
        progress_bars_enabled(),
        geom_verif_options,
    )
    return None
