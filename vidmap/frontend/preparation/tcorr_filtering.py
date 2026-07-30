"""LC / window-LC / score-threshold filtering of transitive correspondences.

Copies ``tcorr`` and ``lc_masks`` from the frontend result and returns
filtered immutable snapshots according to the preparation options.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from vidmap.frontend.cache import CacheMetadataMismatch, IncrementalArtifactContract, validate_incremental_cache
from vidmap.frontend.correspondences import ImagePair, immutable_array_mapping, validate_correspondence_alignment
from vidmap.frontend.options.preparation import LCOptions
from vidmap.repro.frontend import write_tcorr_artifact

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FilteredCorrespondences:
    """Immutable transitive correspondences and their aligned LC masks."""

    tcorr: Mapping[ImagePair, Any]
    lc_masks: Mapping[ImagePair, Any]

    def __post_init__(self) -> None:
        validate_correspondence_alignment(self.tcorr, self.lc_masks)
        object.__setattr__(self, "tcorr", immutable_array_mapping(self.tcorr))
        object.__setattr__(self, "lc_masks", immutable_array_mapping(self.lc_masks))


def _remove_lc_matches(tcorr, lc_masks, *, protected_pairs=frozenset()):
    removed = 0
    for pair in list(tcorr):
        lc_mask = lc_masks.get(pair)
        if lc_mask is None or pair in protected_pairs:
            continue
        keep = ~np.asarray(lc_mask, dtype=bool)
        removed += (~keep).sum()
        if keep.any():
            tcorr[pair], lc_masks[pair] = tcorr[pair][keep], lc_mask[keep]
        else:
            del tcorr[pair]
            if pair in lc_masks:
                del lc_masks[pair]
    return removed


def _filter_lc_scores(path, contract, tcorr, lc_masks, threshold):
    from vidmap.utils.io import get_matches_from_h5

    removed = 0
    validate_incremental_cache(path, contract.metadata, contract.expected_items)
    try:
        with h5py.File(str(path), "r", libver="latest") as hfile:
            for pair in list(tcorr.keys()):
                lc_mask = lc_masks[pair]
                if not lc_mask.any():
                    continue
                h5_matches, h5_scores = get_matches_from_h5(hfile, pair[0], pair[1])
                scores = dict(zip(map(tuple, h5_matches), h5_scores))
                keep = np.ones(len(tcorr[pair]), dtype=bool)
                for index, (match, is_lc) in enumerate(zip(tcorr[pair], lc_mask, strict=True)):
                    if not is_lc:
                        continue
                    key = tuple(match)
                    if key not in scores:
                        raise CacheMetadataMismatch(
                            f"Missing LC score for pair {pair}, row {index}, match {key} in {path}"
                        )
                    keep[index] = scores[key] >= threshold
                removed += (~keep).sum()
                if keep.any():
                    tcorr[pair], lc_masks[pair] = tcorr[pair][keep], lc_mask[keep]
                else:
                    del tcorr[pair], lc_masks[pair]
    except CacheMetadataMismatch:
        raise
    except (OSError, KeyError, TypeError, ValueError) as error:
        raise CacheMetadataMismatch(f"Failed to read LC scores from {path}") from error
    return removed


class CorrespondenceFilter:
    """Own LC filtering and its canonical frontend sidecar."""

    def __init__(self, *, options: LCOptions, repro_dir: Path | None):
        self.options = options
        self.repro_dir = repro_dir

    def filter(
        self,
        *,
        tcorr: Mapping[ImagePair, Any],
        lc_masks: Mapping[ImagePair, Any],
        retrieval_pairs: tuple[ImagePair, ...],
        extended_matches_path: Path,
        extended_matches_artifact: IncrementalArtifactContract,
    ) -> FilteredCorrespondences:
        """Filter frontend-owned correspondences into an immutable result."""
        tcorr = dict(tcorr)
        lc_masks = dict(lc_masks)

        options = self.options
        if options.exclude_lc_matches:
            logger.info("Excluding all loop-closure matches")
            n_removed = _remove_lc_matches(tcorr, lc_masks)
            logger.info("Removed %d loop-closure matches", n_removed)

        if options.exclude_window_lc:
            retrieval_set = {frozenset(pair) for pair in retrieval_pairs}
            protected_pairs = {pair for pair in tcorr if frozenset(pair) in retrieval_set}
            logger.info("Excluding window loop closures; keeping %d retrieval pairs", len(retrieval_set))
            n_removed = _remove_lc_matches(tcorr, lc_masks, protected_pairs=protected_pairs)
            logger.info("Removed %d window loop-closure matches", n_removed)

        if options.min_lc_score > 0:
            if not extended_matches_path.exists():
                raise FileNotFoundError(f"LC score filtering requires match scores at {extended_matches_path}")
            logger.info("Filtering loop-closure matches with score >= %s", options.min_lc_score)
            n_removed = _filter_lc_scores(
                extended_matches_path,
                extended_matches_artifact,
                tcorr,
                lc_masks,
                options.min_lc_score,
            )
            logger.info("Removed %d low-score loop-closure matches", n_removed)

        if self.repro_dir is not None:
            write_tcorr_artifact(
                self.repro_dir / "stage1_filtered_tcorr_lc_masks.json",
                tcorr,
                lc_masks,
                label="filtered_tcorr_lc_masks",
            )

        return FilteredCorrespondences(tcorr=tcorr, lc_masks=lc_masks)
