"""Correspondence pair planning, validation, assignment, and immutable results."""

from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import Any

import numpy as np
from natsort import natsorted
from scipy.spatial import KDTree

from vidmap.utils.parsers import names_to_pair

ImagePair = tuple[str, str]


def validate_pair_name_plan(pairs) -> None:
    """Reject logical pairs that would alias in the established H5 namespace."""
    logical_by_name = {}
    for pair in pairs:
        if (
            isinstance(pair, (str, bytes))
            or not isinstance(pair, Sequence)
            or len(pair) != 2
            or not all(
                isinstance(name, str)
                and name
                and name not in {".", ".."}
                and not any(character.isspace() for character in name)
                for name in pair
            )
        ):
            raise ValueError(f"Invalid image pair: {pair!r}")
        pair = tuple(pair)
        if pair[0] == pair[1]:
            raise ValueError(f"Image pair cannot reference the same image twice: {pair!r}")
        encoded = names_to_pair(*pair)
        previous = logical_by_name.setdefault(encoded, pair)
        if previous != pair:
            raise ValueError(f"Image pairs {previous!r} and {pair!r} collide at H5 name {encoded!r}")


def immutable_array_mapping(values: Mapping[ImagePair, Any]) -> Mapping[ImagePair, Any]:
    """Snapshot arrays once into intrinsically read-only storage and share stage-owned snapshots."""
    snapshot = {}
    for key, value in values.items():
        array = np.asarray(value)
        base = array
        while isinstance(base.base, np.ndarray):
            base = base.base
        if array.flags.writeable or not isinstance(base.base, bytes):
            contiguous = np.ascontiguousarray(array)
            array = np.frombuffer(contiguous.tobytes(), dtype=contiguous.dtype).reshape(contiguous.shape)
        snapshot[key] = array
    return MappingProxyType(snapshot)


def validate_correspondence_alignment(tcorr: Mapping[ImagePair, Any], lc_masks: Mapping[ImagePair, Any]) -> None:
    """Require one boolean LC-mask row for every correspondence row and pair."""
    if set(tcorr) != set(lc_masks):
        missing = sorted(set(tcorr) - set(lc_masks))
        extra = sorted(set(lc_masks) - set(tcorr))
        raise ValueError(f"Correspondence/LC-mask pair mismatch (missing={missing}, extra={extra})")
    validate_pair_name_plan(tcorr)
    seen = set()
    for pair, correspondences in tcorr.items():
        undirected = frozenset(pair)
        if undirected in seen:
            raise ValueError(f"Duplicate undirected correspondence pair: {pair!r}")
        seen.add(undirected)
        correspondences = np.asarray(correspondences)
        masks = np.asarray(lc_masks[pair])
        if correspondences.ndim != 2 or correspondences.shape[1] != 2:
            raise ValueError(f"Correspondences for {pair!r} must have shape (N, 2)")
        if correspondences.dtype.kind not in {"i", "u"} or np.any(correspondences < 0):
            raise ValueError(f"Correspondences for {pair!r} must contain nonnegative integer indices")
        if masks.ndim != 1 or masks.dtype.kind != "b":
            raise ValueError(f"LC mask for {pair!r} must be a one-dimensional boolean array")
        if len(correspondences) != len(masks):
            raise ValueError(f"Correspondence/LC-mask row mismatch for {pair!r}")


def assign_keypoints(keypoints, candidates, max_error):
    """Assign each keypoint to its nearest candidate within ``max_error``."""
    if len(candidates) == 0 or len(keypoints) == 0:
        return np.full(len(keypoints), -1, dtype=np.int64)
    distances, candidate_ids = KDTree(np.asarray(candidates)).query(keypoints, distance_upper_bound=max_error)
    candidate_ids[distances > max_error] = -1
    return candidate_ids


def canonical_pair_names(pairs):
    """Return sorted canonical H5 names for undirected image pairs."""
    validate_pair_name_plan(pairs)
    return sorted({names_to_pair(name0, name1) for name0, name1 in pairs})


def combine_and_deduplicate_pairs(*pair_lists):
    """Combine undirected pair lists into deterministic natural order."""
    all_pairs = []
    for pair_list in pair_lists:
        all_pairs.extend(pair_list)
    validate_pair_name_plan(all_pairs)
    return natsorted(tuple(natsorted(el)) for el in {frozenset(el) for el in all_pairs})
