"""Helpers for sparse source-to-target multiflow schedules."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

from natsort import natsorted

from vidmap.frontend.options.tracking import normalize_multiflow_hops


@dataclass(frozen=True)
class MultiflowWindowPair:
    pair: tuple[str, str]
    hop: int
    slot: int


@dataclass(frozen=True)
class MultiflowIterationPlan:
    hops: tuple[int, ...]
    windows: tuple[tuple[MultiflowWindowPair, ...], ...]


def dense_multiflow_hops(window: int) -> tuple[int, ...]:
    if window < 1:
        raise ValueError("window must be positive")
    return tuple(range(window, 0, -1))


def effective_multiflow_hops(multiflow_hops: Optional[Iterable[int]], window: int) -> tuple[int, ...]:
    normalized = normalize_multiflow_hops(multiflow_hops, window)
    return dense_multiflow_hops(window) if normalized is None else normalized


def _window_pairs_from_hops(keyframe_sequence, target_idx, scheduled_hops):
    records = []
    for hop in scheduled_hops:
        source_idx = target_idx - hop
        if source_idx < 0:
            continue
        records.append(
            MultiflowWindowPair(
                pair=(keyframe_sequence[source_idx], keyframe_sequence[target_idx]),
                hop=hop,
                slot=-hop,
            )
        )
    return records


def build_multiflow_iteration_plan(
    keyframe_sequence: Sequence[str],
    window: int,
    multiflow_hops: Optional[Iterable[int]],
) -> MultiflowIterationPlan:
    """Build the complete ordered streaming plan from one normalized schedule."""
    scheduled_hops = effective_multiflow_hops(multiflow_hops, window)
    return MultiflowIterationPlan(
        hops=scheduled_hops,
        windows=tuple(
            tuple(_window_pairs_from_hops(keyframe_sequence, target_idx, scheduled_hops))
            for target_idx in range(1, len(keyframe_sequence))
        ),
    )


def generate_multiflow_overlap_pairs(
    sequence: Sequence[str],
    window: int,
    multiflow_hops: Optional[Iterable[int]],
) -> list[tuple[str, str]]:
    """Generate all overlap/direct pairs implied by a multiflow schedule."""
    pairs = []
    plan = build_multiflow_iteration_plan(sequence, window, multiflow_hops)
    for records in plan.windows:
        for record in records:
            pairs.append(record.pair)
    return natsorted(
        tuple(natsorted(pair)) for pair in {frozenset(pair) for pair in pairs if len(frozenset(pair)) > 1}
    )
