"""Frontend artifact paths and their pure construction."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class FrontendPaths:
    """Filesystem artifacts produced or consumed by one frontend run."""

    keyframes_path: Path
    track_pairs_path: Path
    retrieval_pairs_path: Path
    sparse_features_path: Path
    sparse_matches_path: Path
    database_transitive_path: Path
    retrieval_features_path: Path
    extended_matches_path: Path
    depth_maps_path: Path
    salient_features_path: Path
    full_depth_maps_path: Path | None = None
    geocalib_per_image_path: Path | None = None
    geocalib_batch_path: Path | None = None


def _path_component(value, label):
    value = str(value)
    if not value or value in {".", ".."} or Path(value).name != value:
        raise ValueError(f"{label} must be one non-empty path component, got {value!r}")
    return value


def build_frontend_paths(
    cache_dir,
    sample_name,
    depth_model_name,
    use_geocalib=False,
    frontend_tag=None,
    config_name=None,
    cache_variant=None,
):
    """Return the canonical paths for one frontend run without filesystem I/O."""
    sample_name = _path_component(sample_name, "sample_name")
    trajectory_frontend_dir = Path(cache_dir)
    if frontend_tag:
        trajectory_frontend_dir /= _path_component(frontend_tag, "frontend_tag")
    if config_name:
        trajectory_frontend_dir /= _path_component(config_name, "config_name")
    trajectory_frontend_dir /= sample_name
    if cache_variant:
        trajectory_frontend_dir /= _path_component(cache_variant, "cache_variant")

    keyframes_path = trajectory_frontend_dir / "keyframes.h5"
    track_pairs_path = trajectory_frontend_dir / "track_pairs.h5"
    retrieval_pairs_path = trajectory_frontend_dir / "retrieval_pairs.h5"
    salient_features_path = trajectory_frontend_dir / "salient_features.h5"
    sparse_features_path = trajectory_frontend_dir / "sparse_features.h5"
    sparse_matches_path = trajectory_frontend_dir / "sparse_matches.h5"
    depth_maps_path = trajectory_frontend_dir / f"depth_maps-{depth_model_name}.h5"
    full_depth_maps_path = trajectory_frontend_dir / f"full_depth_maps-{depth_model_name}.h5"
    geocalib_per_image_path = None
    geocalib_batch_path = None
    if use_geocalib:
        geocalib_per_image_path = trajectory_frontend_dir / "geocalib_per_image.h5"
        geocalib_batch_path = trajectory_frontend_dir / "geocalib_batch.h5"

    return FrontendPaths(
        keyframes_path=keyframes_path,
        track_pairs_path=track_pairs_path,
        retrieval_pairs_path=retrieval_pairs_path,
        sparse_features_path=sparse_features_path,
        sparse_matches_path=sparse_matches_path,
        database_transitive_path=trajectory_frontend_dir / "database_transitive.db",
        retrieval_features_path=trajectory_frontend_dir / "retrieval_features.h5",
        extended_matches_path=trajectory_frontend_dir / "extended_matches.h5",
        depth_maps_path=depth_maps_path,
        full_depth_maps_path=full_depth_maps_path,
        geocalib_per_image_path=geocalib_per_image_path,
        geocalib_batch_path=geocalib_batch_path,
        salient_features_path=salient_features_path,
    )
