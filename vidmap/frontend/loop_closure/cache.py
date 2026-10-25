"""Lifecycle owner for the one extended-match H5 artifact."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import h5py

if TYPE_CHECKING:
    import torch

from vidmap.frontend.cache import (
    CacheMetadataMismatch,
    IncrementalArtifactContract,
    attach_incremental_cache_identity,
    cache_is_valid,
    certify_incremental_artifact,
    inspect_incremental_items,
    mark_incremental_cache_complete,
    prepare_incremental_cache,
)
from vidmap.frontend.cache.validation import dataset_parent_names
from vidmap.frontend.correspondences import canonical_pair_names
from vidmap.frontend.loop_closure.matching import collect_cached_extended_matches, repair_extended_match_pairs


class ExtendedMatchCache:
    """Create, publish, repair, consume, and certify extended matches."""

    def __init__(self, paths, *, force_recompute):
        self.paths = paths
        self.path = paths.extended_matches_path
        self.force_recompute = force_recompute

    def prepare(self, metadata):
        self.metadata = metadata
        prepare_incremental_cache(
            self.path,
            self.metadata,
            overwrite=self.force_recompute,
        )

    def publish_staging(self, staging_path, metadata):
        """Bind private samples to completed sparse indices, retaining compatible pairs."""
        self.metadata = metadata
        if not self.force_recompute and cache_is_valid(self.path, metadata):
            with h5py.File(self.path, "r") as source, h5py.File(staging_path, "a") as destination:
                for pair in dataset_parent_names(source):
                    if pair not in destination:
                        parent, name = pair.rsplit("/", 1)
                        source.copy(pair, destination.require_group(parent), name=name)
        attach_incremental_cache_identity(staging_path, metadata)
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
        device: torch.device,
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
            device=device,
        )

    def collect(self, pairs, *, match_threshold):
        return collect_cached_extended_matches(
            extended_pairs=pairs,
            extended_matches_path=self.path,
            lc_match_thresh=match_threshold,
        )

    def complete(self, pairs) -> IncrementalArtifactContract:
        requested_items = set(canonical_pair_names(pairs))
        with h5py.File(self.path, "r") as hfile:
            stored_items = set(dataset_parent_names(hfile))
        present, missing = inspect_incremental_items(
            self.path,
            sorted(stored_items | requested_items),
            self.metadata,
            repair_malformed=True,
        )
        missing_requested = requested_items.intersection(missing)
        if missing_requested:
            raise CacheMetadataMismatch(f"{self.path}: missing {len(missing_requested)} planned extended-match pairs")
        # Selection owns the requested plan; inference retains valid unselected
        # pairs so a later plan expansion does not repeat matching.
        expected_items = tuple(present)
        mark_incremental_cache_complete(self.path, self.metadata, expected_items)
        return certify_incremental_artifact(self.path, self.metadata, expected_items)
