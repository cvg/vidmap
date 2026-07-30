"""Lifecycle owner for the one extended-match H5 artifact."""

import os

import h5py

from vidmap.frontend.cache import (
    CacheMetadataMismatch,
    IncrementalArtifactContract,
    attach_incremental_cache_identity,
    cache_is_valid,
    certify_incremental_artifact,
    inspect_incremental_items,
    mark_incremental_cache_complete,
    prepare_incremental_cache,
    prune_incremental_items,
)
from vidmap.frontend.correspondences import canonical_pair_names
from vidmap.frontend.loop_closure.matching import collect_cached_extended_matches, repair_extended_match_pairs


class ExtendedMatchCache:
    """Create, publish, repair, consume, and certify extended matches."""

    def __init__(self, paths, metadata, *, force_recompute):
        self.paths = paths
        self.path = paths.extended_matches_path
        self.metadata = metadata
        self.force_recompute = force_recompute

    def prepare(self):
        prepare_incremental_cache(
            self.path,
            self.metadata,
            overwrite=self.force_recompute,
        )

    def publish_staging(self, staging_path):
        """Atomically publish staged sequential matches while retaining valid cached pairs."""
        if not self.force_recompute and cache_is_valid(self.path, self.metadata):
            with h5py.File(self.path, "r") as source, h5py.File(staging_path, "a") as destination:
                for name in source:
                    if name not in destination:
                        source.copy(name, destination)
        attach_incremental_cache_identity(staging_path, self.metadata)
        os.replace(staging_path, self.path)

    def repair(
        self,
        pairs,
        *,
        label,
        tracker,
        scene_parser,
        image_options,
        lowres_match_resolution,
        match_threshold,
    ):
        return repair_extended_match_pairs(
            pairs,
            label=label,
            paths=self.paths,
            metadata=self.metadata,
            tracker=tracker,
            scene_parser=scene_parser,
            conf_highres=image_options,
            lowres_match_resolution=lowres_match_resolution,
            lc_match_thresh=match_threshold,
        )

    def collect(self, pairs, *, match_threshold):
        return collect_cached_extended_matches(
            extended_pairs=pairs,
            extended_matches_path=self.path,
            lc_match_thresh=match_threshold,
        )

    def complete(self, pairs) -> IncrementalArtifactContract:
        expected_items = tuple(canonical_pair_names(pairs))
        _, missing_items = inspect_incremental_items(
            self.path,
            expected_items,
            self.metadata,
            repair_malformed=True,
        )
        if missing_items:
            raise CacheMetadataMismatch(f"{self.path}: missing {len(missing_items)} planned extended-match pairs")
        prune_incremental_items(self.path, expected_items)
        mark_incremental_cache_complete(self.path, self.metadata, expected_items)
        return certify_incremental_artifact(self.path, self.metadata, expected_items)
