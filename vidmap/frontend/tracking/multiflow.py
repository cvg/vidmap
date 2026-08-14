"""Helpers for sparse source-to-target multiflow schedules."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

from natsort import natsorted

from vidmap.frontend.options.tracking import normalize_multiflow_hops


@dataclass(frozen=True)
class MultiflowWindowPair:
    pair: tuple[str, str]
    hop: int
    slot: int


def build_multiflow_window(
    accepted_history: Sequence[str],
    target_name: str,
    multiflow_hops: Iterable[int],
) -> tuple[MultiflowWindowPair, ...]:
    """Build scheduled source fields for one newly accepted target."""
    records = []
    for hop in multiflow_hops:
        source_idx = len(accepted_history) - hop
        if source_idx < 0:
            continue
        records.append(
            MultiflowWindowPair(
                pair=(accepted_history[source_idx], target_name),
                hop=hop,
                slot=-hop,
            )
        )
    return tuple(records)


def generate_multiflow_overlap_pairs(
    sequence: Sequence[str],
    multiflow_hops: Iterable[int],
) -> list[tuple[str, str]]:
    """Generate all overlap/direct pairs implied by a multiflow schedule."""
    pairs = []
    hops = normalize_multiflow_hops(multiflow_hops)
    for target_idx in range(1, len(sequence)):
        records = build_multiflow_window(sequence[:target_idx], sequence[target_idx], hops)
        for record in records:
            pairs.append(record.pair)
    directed_pairs = {}
    for pair in pairs:
        undirected = frozenset(pair)
        if len(undirected) > 1:
            directed_pairs.setdefault(undirected, pair)
    return natsorted(directed_pairs.values())
