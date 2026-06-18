"""Purged Combinatorial Cross-Validation (CPCV) index math — spec §4 (PHASE 2).

Why CPCV here: the walk-forward calendar folds built by
``pooled_validation._run_fold_loop`` OVERLAP heavily (IS fraction 10% with a 5%
step gives ~50% IS overlap between adjacent folds), so fold scores are
correlated and the block-bootstrap LCB is optimistic. This is an OVERLAP
problem, NOT a leak: per-slice label recomputation (``add_pivot_labels`` on
each reset slice) already prevents any IS->OOS look-ahead — the "one-sided
embargo leak" claim is debunked in plan/gpu-refactor-plan.md §2.

CPCV partitions the active timeline into ``n_groups`` contiguous groups and
tests every ``C(n_groups, k_test)`` combination of ``k_test`` groups, which
reconstructs ``k_test * C(N, k) / N`` full out-of-sample paths — the extra OOS
coverage the event-rich LOW side needs.

Purge + embargo: a train bar at position ``p`` carries a centered structural
label window ``[p - purge, p + purge]`` with ``purge = max(STRUCTURAL_NEST) =
200``, so every bar whose window touches a test group is PURGED from training;
an additional ``embargo_bars`` AFTER each test group guards serial correlation.

Default OFF: nothing in the default pipeline calls this module. The opt-in
entry point is ``pooled_validation.build_cpcv_folds``.
"""
from __future__ import annotations

import itertools
import logging
from dataclasses import dataclass
from math import comb

from .scoring import STRUCTURAL_NEST

logger = logging.getLogger(__name__)

#: Centered structural-label half-width — bars this close to a test group have
#: a label window that overlaps it and must be purged from training.
PURGE_BARS: int = max(STRUCTURAL_NEST)  # 200

BarRange = tuple[int, int]  # [start, end) offsets into the reference index


@dataclass(frozen=True)
class CPCVConfig:
    """Knobs for one CPCV partition (spec §4: N groups 6-8, k=2)."""

    n_groups: int = 6
    k_test: int = 2
    purge_bars: int = PURGE_BARS
    embargo_bars: int = PURGE_BARS

    def __post_init__(self) -> None:
        if self.n_groups < 3:
            raise ValueError(f"n_groups must be >= 3, got {self.n_groups}")
        if not 1 <= self.k_test < self.n_groups:
            raise ValueError(
                f"k_test must be in [1, n_groups), got {self.k_test}")
        if self.purge_bars < 0 or self.embargo_bars < 0:
            raise ValueError("purge_bars/embargo_bars must be >= 0")
        if not 6 <= self.n_groups <= 8:
            logger.debug("n_groups %d outside the spec range [6, 8]",
                         self.n_groups)


@dataclass(frozen=True)
class CPCVSplit:
    """One train/test combination: k test groups + purged train ranges."""

    split_id: int
    test_groups: tuple[int, ...]
    test_ranges: tuple[BarRange, ...]
    train_ranges: tuple[BarRange, ...]


def group_bounds(n_bars: int, n_groups: int) -> list[BarRange]:
    """Contiguous near-equal group ranges covering ``[0, n_bars)`` exactly."""
    if n_bars < n_groups:
        raise ValueError(f"n_bars={n_bars} < n_groups={n_groups}")
    edges = [round(i * n_bars / n_groups) for i in range(n_groups + 1)]
    return [(edges[i], edges[i + 1]) for i in range(n_groups)]


def _train_complement(
    test_ranges: tuple[BarRange, ...], n_bars: int, purge: int, embargo: int,
) -> tuple[BarRange, ...]:
    """Maximal train ranges left after blocking each test range +- guards.

    Blocked interval per test range ``[s, e)``: ``[s - purge, e + purge +
    embargo)`` — purge on BOTH sides (the centered label window is symmetric)
    plus the one-sided embargo after the test group.
    """
    blocked = sorted(
        (max(0, s - purge), min(n_bars, e + purge + embargo))
        for s, e in test_ranges
    )
    merged: list[list[int]] = []
    for lo, hi in blocked:
        if merged and lo <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], hi)
        else:
            merged.append([lo, hi])
    train: list[BarRange] = []
    cursor = 0
    for lo, hi in merged:
        if cursor < lo:
            train.append((cursor, lo))
        cursor = max(cursor, hi)
    if cursor < n_bars:
        train.append((cursor, n_bars))
    return tuple(train)


def build_cpcv_splits(
    n_bars: int, config: CPCVConfig | None = None,
) -> list[CPCVSplit]:
    """All ``C(n_groups, k_test)`` purged splits over ``[0, n_bars)``."""
    cfg = config or CPCVConfig()
    bounds = group_bounds(n_bars, cfg.n_groups)
    splits: list[CPCVSplit] = []
    for sid, combo in enumerate(
            itertools.combinations(range(cfg.n_groups), cfg.k_test)):
        test_ranges = tuple(bounds[g] for g in combo)
        train_ranges = _train_complement(
            test_ranges, n_bars, cfg.purge_bars, cfg.embargo_bars)
        splits.append(CPCVSplit(split_id=sid, test_groups=tuple(combo),
                                test_ranges=test_ranges,
                                train_ranges=train_ranges))
    logger.info("build_cpcv_splits: %d splits (n_bars=%d, N=%d, k=%d, "
                "purge=%d, embargo=%d)", len(splits), n_bars, cfg.n_groups,
                cfg.k_test, cfg.purge_bars, cfg.embargo_bars)
    return splits


def validate_purge(
    splits: list[CPCVSplit], purge_bars: int = PURGE_BARS,
) -> None:
    """Raise ``ValueError`` if any train bar's label window touches a test
    group (spec §4 purge-correctness assertion)."""
    for sp in splits:
        for a, b in sp.train_ranges:
            for s, e in sp.test_ranges:
                if not (b - 1 + purge_bars < s or a - purge_bars >= e):
                    raise ValueError(
                        f"split {sp.split_id}: train range [{a},{b}) label "
                        f"window (+-{purge_bars}) overlaps test [{s},{e})")


def n_paths(n_groups: int, k_test: int) -> int:
    """Number of reconstructable full OOS paths: ``k*C(N,k)/N = C(N-1,k-1)``."""
    return comb(n_groups - 1, k_test - 1)


def reconstruct_paths(splits: list[CPCVSplit]) -> list[list[tuple[int, int]]]:
    """Assign every (split, test-group) occurrence to an OOS path.

    Canonical CPCV path reconstruction: for each group, its i-th appearance as
    a test group (in split order) belongs to path i. Each returned path is a
    list of ``(split_id, group_id)`` cells sorted by group, covering every
    group exactly once — one full out-of-sample pass over the timeline.
    """
    occurrence: dict[int, int] = {}
    paths: dict[int, list[tuple[int, int]]] = {}
    for sp in sorted(splits, key=lambda s: s.split_id):
        for g in sp.test_groups:
            pid = occurrence.get(g, 0)
            occurrence[g] = pid + 1
            paths.setdefault(pid, []).append((sp.split_id, g))
    out = [sorted(paths[pid], key=lambda cell: cell[1])
           for pid in sorted(paths)]
    groups = sorted(occurrence)
    for pid, path in enumerate(out):
        if [g for _, g in path] != groups:
            raise ValueError(
                f"path {pid} does not cover every group exactly once: {path}")
    return out


__all__ = [
    "PURGE_BARS", "CPCVConfig", "CPCVSplit", "group_bounds",
    "build_cpcv_splits", "validate_purge", "n_paths", "reconstruct_paths",
]
