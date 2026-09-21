"""Retain nearby future LC views without reordering or reversing match requests."""

from collections import defaultdict, deque


def retained_feature_names(pairs, capacity):
    if capacity < 2:
        raise ValueError("Loop-closure feature capacity must be at least two")
    uses = defaultdict(deque)
    for index, pair in enumerate(pairs):
        for name in set(pair):
            uses[name].append(index)
    retained = set()
    for pair in pairs:
        required = set(pair)
        assert 1 <= len(required) <= 2
        for name in required:
            uses[name].popleft()
        optional = sorted(
            (name for name in retained - required if uses[name]),
            key=lambda name: (uses[name][0], name),
        )
        retained = required | set(optional[: capacity - len(required)])
        yield tuple(sorted(retained))
