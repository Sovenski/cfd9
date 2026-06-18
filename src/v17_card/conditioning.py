"""Signal Card §3.2/§3.4 — live conditioning and the span clock (spec FIXED v3).

Implements THE LIVE UPDATE RULE (F1-corrected, R1/R2-pinned):

    P(N*_eff >= x | L, survived k) = 0                        if x > L
                                   = S_R(max(x, k)) / S_R(k)  otherwise

with the off-grid step-function-floor lookup rule (re-review R4): table
arguments evaluate at the LARGEST grid value <= the argument, and
``S_R(x) = 1`` for ``x`` below the noise floor (20). Engine and Pine must
share this exact rule (§4.2 parity assertion).
"""
from __future__ import annotations

import logging
from typing import Sequence

import numpy as np

from ..scoring_v5 import SPAN_GRID

logger = logging.getLogger(__name__)

#: Pivots with N* below this floor do not exist as events (spec §1.2).
NOISE_FLOOR: int = SPAN_GRID[0]

#: E[hold] cap, consistent with the "500+" censor bucket (spec §3.4, F10).
HOLD_CAP: int = SPAN_GRID[-1]


def survival_lookup(
    table: np.ndarray, x: float, grid: Sequence[int] = tuple(SPAN_GRID),
) -> float:
    """Off-grid step-function-floor lookup of ``S_R`` (spec §3.2, R4).

    Evaluates the table at the largest grid value <= ``x``;
    ``S_R(x) = 1`` below the noise floor. Arguments above the cap floor to
    the cap.
    """
    grid_arr = np.asarray(grid)
    if x < grid_arr[0]:
        return 1.0
    idx = int(np.searchsorted(grid_arr, x, side="right")) - 1
    return float(table[idx])


def conditional_survival(
    table: np.ndarray,
    x: float,
    k: float,
    left_span: float,
    grid: Sequence[int] = tuple(SPAN_GRID),
) -> float:
    """``P(N*_eff >= x | L, survived k)`` — the live update rule (F1).

    The survival table is fitted on the right-span R only, so the left side
    enters EXCLUSIVELY through the L-clamp (no double-counting): ``x > L``
    is impossible regardless of right-side survival.

    Args:
        table: ``S_R`` at the grid points (engine- or Pine-exported).
        x: Queried span threshold (off-grid allowed, R4 floor rule).
        k: Bars survived, counted from the candidate pivot bar ``i`` (R2).
        left_span: Proven left-span ``L`` at ``i`` (causal at fire).

    Returns:
        Probability in [0, 1]; exactly 0.0 above ``L``.
    """
    if x > left_span:
        return 0.0
    s_k = survival_lookup(table, k, grid)
    if s_k <= 0.0:
        return 0.0
    return min(1.0, survival_lookup(table, max(x, k), grid) / s_k)


def expected_hold(
    table: np.ndarray,
    k: float = 0.0,
    left_span: float = float(HOLD_CAP),
    grid: Sequence[int] = tuple(SPAN_GRID),
) -> float:
    """Discrete-survival span clock ``E[hold]`` (spec §3.4).

    ``E[hold] = sum_j P(N*_eff >= x_j | L, k) * dx_j`` over the span grid,
    capped at 500 (consistent with the "500+" bucket). Terms with
    ``x_j > L`` contribute zero (the F1 clamp), and all lookups use the R4
    floor rule.
    """
    total = 0.0
    prev = 0.0
    for x_j in grid:
        total += conditional_survival(table, float(x_j), k, left_span, grid) \
            * (float(x_j) - prev)
        prev = float(x_j)
    return min(total, float(HOLD_CAP))


__all__ = [
    "NOISE_FLOOR", "HOLD_CAP",
    "survival_lookup", "conditional_survival", "expected_hold",
]
