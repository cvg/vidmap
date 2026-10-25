"""Canonical frontend-to-mapper input paths and validation."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from vidmap.calibration import CalibrationEvidence
from vidmap.configuration.names import config_name_to_output_slug
from vidmap.depth_artifacts import FULL_DEPTH_MAPS_NAME
from vidmap.mapper.inputs.timeline import validate_frame_sequence

DATABASE_NAME = "database_complete.db"
DEPTH_MAPS_NAME = "depth_maps.h5"
LC_MASKS_NAME = "lc_masks.json"
GEOCALIB_PER_IMAGE_NAME = "geocalib_per_image.h5"
GEOCALIB_BATCH_NAME = "geocalib_batch.h5"
VGC_FILTERED_PAIRS_NAME = "vgc_filtered_pairs.json"
MANIFEST_NAME = "manifest.json"
MANIFEST_SCHEMA_VERSION = 14
_FRONTEND_IDENTITY_SCHEMA_VERSION = 6
SQLITE_SIDECAR_SUFFIXES = ("-journal", "-shm", "-wal")

REQUIRED_PAYLOAD_NAMES = frozenset({DATABASE_NAME, DEPTH_MAPS_NAME, LC_MASKS_NAME})
REQUIRED_NAMES = REQUIRED_PAYLOAD_NAMES | {MANIFEST_NAME}
OPTIONAL_NAMES = frozenset({GEOCALIB_PER_IMAGE_NAME, GEOCALIB_BATCH_NAME, VGC_FILTERED_PAIRS_NAME})
CANONICAL_NAMES = REQUIRED_NAMES | OPTIONAL_NAMES
FRONTEND_IDENTITY_KEYS = frozenset(
    {
        "schema_version",
        "tag",
        "config_name",
        "config_fingerprint",
        "colmap_runtime",
        "dataset",
        "scene",
        "mode",
        "testset_id",
        "reference_image_ids",
        "source_fingerprint",
    }
)


def calibration_artifact_name(estimator: str, inference: str) -> str | None:
    """Select the existing calibration payload from validated boundary settings."""
    if estimator == "none":
        return None
    if estimator == "da3":
        return DEPTH_MAPS_NAME
    if estimator == "geocalib":
        return GEOCALIB_BATCH_NAME if inference == "selected_batch" else GEOCALIB_PER_IMAGE_NAME
    raise ValueError(f"Unknown calibration estimator: {estimator}")


def sqlite_sidecar_paths(database_path: Path) -> tuple[Path, ...]:
    """Return every SQLite sidecar path for one database generation."""
    database_path = Path(database_path)
    return tuple(Path(f"{database_path}{suffix}") for suffix in SQLITE_SIDECAR_SUFFIXES)


def require_finalized_sqlite(database_path: Path) -> None:
    """Reject a database whose payload still depends on SQLite sidecars."""
    sidecars = [path for path in sqlite_sidecar_paths(database_path) if path.exists()]
    if sidecars:
        raise ValueError(f"Finalized SQLite database has sidecars: {sidecars}")


@dataclass(frozen=True)
class FileProvenance:
    """Stable byte identity for one finalized file."""

    size: int
    sha256: str

    @classmethod
    def from_path(cls, path: Path) -> "FileProvenance":
        digest = hashlib.sha256()
        size = 0
        with Path(path).open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                size += len(chunk)
                digest.update(chunk)
        return cls(size=size, sha256=digest.hexdigest())

    def validate(self, path: Path) -> None:
        actual = self.from_path(path)
        if actual != self:
            raise ValueError(
                f"Mapper-input provenance mismatch for {path}: "
                f"expected size={self.size} sha256={self.sha256}, "
                f"got size={actual.size} sha256={actual.sha256}"
            )


def mapper_inputs_manifest(
    directory: Path,
    *,
    frontend_identity: Mapping[str, object],
    frame_sequence: dict,
    panorama: dict | None = None,
    calibration: CalibrationEvidence | None = None,
) -> dict:
    """Build deterministic provenance for canonical mapper-input payloads."""
    directory = resolve_mapper_inputs_directory(directory)
    names = {name for name in CANONICAL_NAMES - {MANIFEST_NAME} if (directory / name).is_file()}
    artifacts = {
        name: dict(proof)
        for name, proof in _verified_artifacts(directory, _artifact_signature(directory, names)).items()
    }
    return _tagged_manifest(artifacts, frontend_identity, frame_sequence, panorama, calibration)


def _artifact_signature(
    directory: Path,
    artifact_names: set[str],
) -> tuple[tuple[str, int, int, int, int, int], ...]:
    signature = []
    for name in sorted(artifact_names):
        path = Path(directory) / name
        stat = path.stat()
        signature.append((path.name, stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns))
    return tuple(signature)


@lru_cache(maxsize=64)
def _verified_artifacts(
    directory: Path,
    signature: tuple[tuple[str, int, int, int, int, int], ...],
) -> dict[str, dict[str, int | str]]:
    artifacts = {}
    for name, *_stat in signature:
        proof = FileProvenance.from_path(directory / name)
        artifacts[name] = {"size": proof.size, "sha256": proof.sha256}
    return artifacts


def _tagged_manifest(
    artifacts: Mapping[str, object],
    frontend_identity: Mapping[str, object],
    frame_sequence: dict,
    panorama: dict | None = None,
    calibration: CalibrationEvidence | None = None,
) -> dict[str, object]:
    """Build the current tagged mapper-input manifest."""
    validate_frame_sequence(frame_sequence)
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "frontend_identity": dict(frontend_identity),
        "artifacts": dict(artifacts),
        "frame_sequence": frame_sequence,
        "calibration": None if calibration is None else calibration.as_dict(),
    }
    if panorama is not None:
        manifest["panorama"] = panorama
    return manifest


def _tagged_identity(manifest: Mapping[str, object], *, manifest_path: Path) -> dict[str, object]:
    """Read the identity from current tagged mapper inputs."""
    schema_version = manifest.get("schema_version")
    if schema_version != MANIFEST_SCHEMA_VERSION:
        raise ValueError(f"Mapper inputs require manifest schema {MANIFEST_SCHEMA_VERSION}: {manifest_path}")
    fields = {"schema_version", "frontend_identity", "artifacts", "frame_sequence", "calibration"}
    if set(manifest) not in (fields, fields | {"panorama"}):
        raise ValueError(f"Invalid mapper-input manifest fields: {manifest_path}")
    validate_frame_sequence(manifest["frame_sequence"])
    if "panorama" in manifest:
        from vidmap.datasets.panorama_metadata import validate_panorama_metadata

        validate_panorama_metadata(manifest["panorama"])
    identity = manifest.get("frontend_identity")
    _validate_frontend_identity(identity, manifest_path=manifest_path)
    return identity


def write_mapper_inputs_manifest(
    directory: Path,
    *,
    frontend_identity: Mapping[str, object],
    frame_sequence: dict,
    panorama: dict | None = None,
    calibration: CalibrationEvidence | None = None,
) -> None:
    payload = mapper_inputs_manifest(
        directory,
        frontend_identity=frontend_identity,
        frame_sequence=frame_sequence,
        panorama=panorama,
        calibration=calibration,
    )
    (Path(directory) / MANIFEST_NAME).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def validate_mapper_inputs_identity(directory: Path, expected_identity: Mapping[str, object]) -> None:
    """Validate tagged-boundary metadata without hashing its payload files."""

    manifest_path = Path(directory) / MANIFEST_NAME
    try:
        manifest = json.loads(manifest_path.read_text())
    except FileNotFoundError:
        raise
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Invalid mapper-input manifest: {manifest_path}") from error
    actual_identity = _tagged_identity(manifest, manifest_path=manifest_path)
    expected = dict(expected_identity)
    if not isinstance(actual_identity, dict) or any(
        actual_identity.get(key) != value for key, value in expected.items()
    ):
        raise ValueError(
            f"Mapper-input frontend identity mismatch for {manifest_path}: "
            f"expected at least {expected!r}, got {actual_identity!r}"
        )


def _validate_frontend_identity(identity: object, *, manifest_path: Path) -> None:
    if not isinstance(identity, dict) or set(identity) != FRONTEND_IDENTITY_KEYS:
        raise ValueError(f"Invalid mapper-input frontend identity fields: {manifest_path}")
    if identity["schema_version"] != _FRONTEND_IDENTITY_SCHEMA_VERSION or isinstance(identity["schema_version"], bool):
        raise ValueError(f"Invalid mapper-input frontend identity schema: {manifest_path}")
    for key in (
        "tag",
        "config_name",
        "colmap_runtime",
        "dataset",
        "scene",
        "mode",
        "testset_id",
    ):
        if not isinstance(identity[key], str) or not identity[key]:
            raise ValueError(f"Invalid mapper-input frontend identity {key}: {manifest_path}")
    tag = identity["tag"]
    if tag in {".", ".."} or Path(tag).name != tag or "\\" in tag:
        raise ValueError(f"Invalid mapper-input frontend identity tag: {manifest_path}")
    config_name = identity["config_name"]
    if (
        config_name.startswith("/")
        or "\\" in config_name
        or any(part in {"", ".", ".."} for part in config_name.split("/"))
        or config_name.endswith((".yaml", ".yml"))
        or config_name_to_output_slug(config_name) != tag
    ):
        raise ValueError(f"Mapper-input frontend tag does not match config_name: {manifest_path}")
    for key in ("config_fingerprint", "source_fingerprint"):
        fingerprint = identity[key]
        if not isinstance(fingerprint, str) or re.fullmatch(r"[0-9a-f]{64}", fingerprint) is None:
            raise ValueError(f"Invalid mapper-input frontend {key}: {manifest_path}")
    reference_image_ids = identity["reference_image_ids"]
    if not isinstance(reference_image_ids, list) or any(
        not isinstance(image_id, int) or isinstance(image_id, bool) for image_id in reference_image_ids
    ):
        raise ValueError(f"Invalid mapper-input frontend reference_image_ids: {manifest_path}")


def resolve_mapper_inputs_directory(directory: Path) -> Path:
    """Resolve a mapper-input directory, including user-provided symlink paths."""
    return Path(directory).expanduser().resolve()


@dataclass(frozen=True)
class MapperInputs:
    """Validated inputs in one self-contained mapper directory."""

    directory: Path
    generation: FileProvenance | None = None
    expected_identity: Mapping[str, object] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "directory", resolve_mapper_inputs_directory(self.directory))

    @classmethod
    def from_directory(
        cls,
        directory: Path,
        *,
        expected_identity: Mapping[str, object] | None = None,
    ) -> "MapperInputs":
        directory = resolve_mapper_inputs_directory(directory)
        manifest_path = directory / MANIFEST_NAME
        generation = FileProvenance.from_path(manifest_path) if manifest_path.is_file() else None
        return cls(directory, generation, expected_identity)

    def validate(self) -> None:
        if not self.directory.is_dir():
            raise FileNotFoundError(f"Mapper-input directory not found: {self.directory}")

        names = {path.name for path in self.directory.iterdir()}
        missing = REQUIRED_NAMES - names
        if missing:
            raise FileNotFoundError(f"Missing required mapper-input files in {self.directory}: {sorted(missing)}")

        canonical_names = names & CANONICAL_NAMES
        for name in canonical_names:
            path = self.directory / name
            if path.is_symlink():
                raise ValueError(f"Mapper inputs must not contain symlinks: {path}")
            if not path.is_file():
                raise ValueError(f"Mapper input must be a regular file: {path}")

        require_finalized_sqlite(self.database_path)
        if VGC_FILTERED_PAIRS_NAME in names:
            self.read_vgc_filtered_pairs()

        manifest_path = self.directory / MANIFEST_NAME
        if self.generation is None:
            if manifest_path.is_file():
                raise ValueError(f"Mapper-input generation changed before consumption: {manifest_path}")
        else:
            try:
                self.generation.validate(manifest_path)
            except (OSError, ValueError) as error:
                raise ValueError(f"Mapper-input generation changed before consumption: {manifest_path}") from error
        try:
            manifest = json.loads(manifest_path.read_text())
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"Invalid mapper-input manifest: {manifest_path}") from error
        schema_version = manifest.get("schema_version")
        if schema_version != MANIFEST_SCHEMA_VERSION:
            raise ValueError(f"Unsupported mapper-input manifest schema in {manifest_path}: {schema_version!r}")
        frontend_identity = _tagged_identity(manifest, manifest_path=manifest_path)
        evidence = None if manifest["calibration"] is None else CalibrationEvidence.from_dict(manifest["calibration"])
        selected_artifact = (
            None if evidence is None else calibration_artifact_name(evidence.estimator, evidence.inference)
        )
        expected_geocalib = {selected_artifact} & {GEOCALIB_PER_IMAGE_NAME, GEOCALIB_BATCH_NAME}
        if names & {GEOCALIB_PER_IMAGE_NAME, GEOCALIB_BATCH_NAME} != expected_geocalib:
            raise ValueError("GeoCalib artifacts disagree with the extraction's predictor settings")
        if self.expected_identity is not None and (
            not isinstance(frontend_identity, dict)
            or any(frontend_identity.get(key) != value for key, value in dict(self.expected_identity).items())
        ):
            raise ValueError(
                f"Mapper-input frontend identity mismatch for {manifest_path}: "
                f"expected {dict(self.expected_identity)!r}, got {frontend_identity!r}"
            )
        expected_manifest = _tagged_manifest(
            _verified_artifacts(
                self.directory,
                _artifact_signature(self.directory, canonical_names - {MANIFEST_NAME}),
            ),
            frontend_identity,
            manifest["frame_sequence"],
            manifest["panorama"] if "panorama" in manifest else None,
            evidence,
        )
        if manifest != expected_manifest:
            raise ValueError(f"Mapper-input manifest does not match written payloads: {manifest_path}")

    def frontend_identity(self) -> dict[str, object]:
        """Return the persisted frontend identity for a finalized tagged boundary."""
        manifest_path = self.directory / MANIFEST_NAME
        try:
            manifest = json.loads(manifest_path.read_text())
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"Invalid mapper-input manifest: {manifest_path}") from error
        return _tagged_identity(manifest, manifest_path=manifest_path)

    def calibration(self) -> CalibrationEvidence | None:
        """Read original prediction associations without inferring camera membership."""
        manifest = json.loads((self.directory / MANIFEST_NAME).read_text())
        _tagged_identity(manifest, manifest_path=self.directory / MANIFEST_NAME)
        value = manifest["calibration"]
        return None if value is None else CalibrationEvidence.from_dict(value)

    def frame_sequence(self) -> dict:
        """Read the frontend's explicit order over finalized native frames."""
        manifest = json.loads((self.directory / MANIFEST_NAME).read_text())
        _tagged_identity(manifest, manifest_path=self.directory / MANIFEST_NAME)
        return manifest["frame_sequence"]

    @property
    def full_depth_maps_path(self) -> Path | None:
        # Full grids are a visualization sidecar, not a mapper input. Keeping
        # them outside the validated snapshot avoids hashing a large file on
        # every reconstruction while retaining it with the frontend result.
        path = self.directory.parent / FULL_DEPTH_MAPS_NAME
        return path if path.is_file() else None

    def read_vgc_filtered_pairs(self) -> set[tuple[str, str]] | None:
        path = self.vgc_filtered_pairs_path
        if path is None:
            return None
        try:
            payload = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"Invalid strict-VGC mapper input: {path}") from error
        if not isinstance(payload, dict) or set(payload) != {"pairs"} or not isinstance(payload["pairs"], list):
            raise ValueError(f"Invalid strict-VGC mapper input schema: {path}")
        pairs = payload["pairs"]
        if any(
            not isinstance(pair, list) or len(pair) != 2 or not all(isinstance(name, str) for name in pair)
            for pair in pairs
        ):
            raise ValueError(f"Invalid strict-VGC pair in mapper input: {path}")
        return {(pair[0], pair[1]) for pair in pairs}

    @property
    def database_path(self) -> Path:
        return self.directory / DATABASE_NAME

    @property
    def depth_maps_path(self) -> Path:
        return self.directory / DEPTH_MAPS_NAME

    @property
    def lc_masks_path(self) -> Path:
        return self.directory / LC_MASKS_NAME

    @property
    def vgc_filtered_pairs_path(self) -> Path | None:
        path = self.directory / VGC_FILTERED_PAIRS_NAME
        return path if path.exists() else None
