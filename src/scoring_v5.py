"""Scorer v5 — span-weighted ±1 direct-hit scoring core.

Implements ``plan/scorer-v5-signal-card-spec.md`` (FIXED v3) §1 (continuous
pivot span N*) and §2 (weighted matching / precision / recall / N_eff):

* §1.2 span labeling on ``SPAN_GRID`` by REUSING the frozen-validated
  ``label_pivots`` verbatim per grid scale (O(grid), monotone by
  construction) — detection semantics are never reimplemented here.
* §1.2 right-edge censoring (proven lower bound ``N*_lb`` + flag) and the
  "500+" top bucket (recorded 500, always censored).
* §1.3 span weights ``w(N*) = N*/100`` (REFERENCE_N normalization).
* §2.1 ±1-bar direct-hit Hungarian matching (no lead bias; larger-span
  tie-break at equal distance).
* §2.2 weighted precision / recall, the ``W_FP`` false-positive exchange
  rate (F3) and the ``REFERENCE_MASS`` / N_eff recall-target basis (F4).
* §3.2 causal left-span ``L`` (F1) for the Signal Card phases.

This module is scorer-side only (spec §0 non-goals): it is imported
exclusively by editable files and re-exported through ``src.scoring`` (lazy
PEP 562 forwarding) so the public import surface stays ``src.scoring``.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Union

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from .scoring import (
    MIN_RATE,
    PRECISION_EXPONENT,
    RECALL_TARGET,
    REFERENCE_N,
    _HUNGARIAN_INF,
    _HUNGARIAN_INF_THRESHOLD,
    label_pivots,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants (spec §1.2, §2.1-§2.2)
# ---------------------------------------------------------------------------

#: Log-spaced span grid (spec §1.2). N* is recorded as the LARGEST grid value
#: passed (grid-floor semantics, F10 — conservative, downward-biased by at
#: most one grid step). Pivots with N* < 20 do not exist as events (the
#: Scorer-v3 minor-dip lesson; the floor is deliberate and load-bearing).
SPAN_GRID: list[int] = [20, 30, 40, 50, 70, 100, 140, 200, 300, 500]

#: SCORER_VERSION — the objective era marker. A change to the pricing below
#: (W_FP) changes the objective, so LCBs from different eras are comparable
#: only BY LABEL: every run records this string in its trace and top-level
#: meta (project_scorer_v5_era). v5.0 -> v5.1 bumped W_FP 0.2 -> 0.5 and the
#: firing-cap default 2.0 -> 1.0 (2026-06-11 anti-spray re-pricing).
SCORER_VERSION: str = "v5.1"

#: W_FP — the price of a false signal in pivot-mass units (spec §2.2, F3).
#: Exchange-rate semantics: at the v5.1 default 0.5 one false signal costs
#: 2.5 floor-grade (N*=20) events, i.e. four false signals cost one T1
#: (N*=200) hit — a 2.5x higher spray price than the v5.0 default of 0.2.
#: This single constant controls the optimizer's spray incentive; its
#: default is justified by the mandatory sensitivity test
#: (``tests/test_scorer_v5.py::test_w_fp_sensitivity_firing_gate_catches_spray``).
W_FP: float = 0.5

#: REFERENCE_MASS — the v4-equivalent recall-target basis (spec §2.2, F4):
#: 27 SPX T1 highs at w(200) = 2.0 each => 54.0. Pinned ONCE here; the
#: v4<->v5 anchor is asserted by a unit test so it can never drift silently.
REFERENCE_MASS: float = 27 * 2.0

#: ±1-bar direct-hit window (spec §2.1). Symmetric; no lead bias.
MATCH_WINDOW: int = 1

#: Secondary-cost epsilon for the larger-span tie-break at equal distance
#: (spec §2.1). Must satisfy ``TIEBREAK_EPS < 1 / (2 * w(max(SPAN_GRID)))``
#: = 0.1 so the perturbation can never flip a distance ranking (asserted in
#: tests).
TIEBREAK_EPS: float = 0.01


# ---------------------------------------------------------------------------
# §1.3 — span weights
# ---------------------------------------------------------------------------


def span_weight(n_star: float) -> float:
    """Credit mass of a pivot of span ``n_star``: ``w(N*) = N*/REFERENCE_N``.

    Empirical pivot frequency falls ~1/N, so ``w ∝ N*`` auto-balances total
    credit mass per log-band — the optimizer cannot win by farming small
    swings (spec §1.3). Censored pivots pass their proven bound ``N*_lb``.
    """
    return float(n_star) / float(REFERENCE_N)


# ---------------------------------------------------------------------------
# §1.2 — span labeling on the grid (reusing the frozen labeler)
# ---------------------------------------------------------------------------


def _censor_flags(spans: np.ndarray) -> np.ndarray:
    """Right-censoring flags for recorded spans (spec §1.2).

    A recorded span ``s`` is only a proven LOWER BOUND when the next grid
    value's centered window does not fit inside the data (series edge), or
    when ``s`` is the 500 cap ("500+" bucket — never exact).
    """
    n = spans.shape[0]
    last = n - 1
    out = np.zeros(n, dtype=bool)
    next_grid = {g: SPAN_GRID[i + 1] for i, g in enumerate(SPAN_GRID[:-1])}
    cap = SPAN_GRID[-1]
    for i in np.flatnonzero(spans > 0):
        s = int(spans[i])
        if s >= cap:
            out[i] = True
        else:
            ng = next_grid[s]
            if i + ng > last or i - ng < 0:
                out[i] = True
    return out


def label_pivot_spans(df: pd.DataFrame) -> pd.DataFrame:
    """Per-bar pivot spans on ``SPAN_GRID`` for both sides (spec §1.1-§1.2).

    Calls the EXISTING frozen-validated ``label_pivots`` once per grid scale
    and records the per-bar LARGEST grid value passed (monotone by
    construction; plateau collapse and ambiguous-both-sides exclusion are
    inherited verbatim from the labeler).

    Args:
        df: OHLCV frame with ``high`` and ``low`` columns.

    Returns:
        DataFrame (same index as ``df``) with columns ``pivot_span_high``,
        ``pivot_span_low`` (0 = no event, else grid N*) and
        ``pivot_span_censored_{high,low}`` (bool — recorded value is a
        proven lower bound ``N*_lb``).
    """
    n = len(df)
    span_high = np.zeros(n, dtype=np.int32)
    span_low = np.zeros(n, dtype=np.int32)
    for grid_n in SPAN_GRID:
        if n < 2 * grid_n + 1:
            break
        lbl = label_pivots(df, grid_n).to_numpy()
        span_high[lbl == 1] = grid_n
        span_low[lbl == -1] = grid_n
    return pd.DataFrame(
        {
            "pivot_span_high": span_high,
            "pivot_span_low": span_low,
            "pivot_span_censored_high": _censor_flags(span_high),
            "pivot_span_censored_low": _censor_flags(span_low),
        },
        index=df.index,
    )


def compute_left_span(arr: np.ndarray, i: int, is_high: bool = False) -> int:
    """Proven left-span ``L`` at bar ``i`` (spec §3.2, F1 — causal at fire).

    ``L = max{ l : low[i] = min(low[i-l .. i]) }`` for the LOW side (HIGH
    side mirrors with highs/max). Clamped at the data's left edge and at the
    grid cap (500). Pine-portable as a backward scan with early exit.
    """
    cap = SPAN_GRID[-1]
    v = arr[i]
    left = 0
    j = i - 1
    while j >= 0 and left < cap:
        if (arr[j] > v) if is_high else (arr[j] < v):
            break
        left += 1
        j -= 1
    return left


# ---------------------------------------------------------------------------
# §2.1 — ±1 weighted Hungarian matching
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WeightedStats:
    """Span-mass match statistics of one scored slice (spec §2.2).

    Fields are floats so pooled (cluster-weighted) aggregates can reuse the
    same dataclass and scoring math.
    """

    tp_mass: float
    total_mass: float
    n_signals: float
    n_unmatched: float
    n_bars: float


def _to_bool_array(signals: Union[pd.Series, np.ndarray]) -> np.ndarray:
    if isinstance(signals, pd.Series):
        return signals.fillna(False).astype(bool).to_numpy()
    return np.asarray(signals, dtype=bool)


def match_signals_weighted(
    signals: Union[pd.Series, np.ndarray],
    spans: Union[pd.Series, np.ndarray],
) -> WeightedStats:
    """±1 direct-hit Hungarian matching against span-weighted pivots.

    Window: a pivot within ``[t-1, t+1]`` of signal bar ``t`` (spec §2.1) —
    symmetric, no lead bias. Hungarian 1:1 assignment on ``|distance|`` with
    out-of-window cost ``_HUNGARIAN_INF``; at equal distance the secondary
    cost ``-TIEBREAK_EPS * w(N*)`` prefers the larger-span pivot without
    ever flipping a distance ranking. A signal matching no pivot in ±1 is a
    false positive.

    Args:
        signals: Boolean per-bar firing mask.
        spans: Per-bar pivot span column (0 = no pivot, else grid N* /
            proven bound N*_lb — censored pivots match at their bound).

    Returns:
        ``WeightedStats`` with matched mass, total slice mass, signal count,
        unmatched-signal count and bar count.
    """
    sig_arr = _to_bool_array(signals)
    span_arr = spans.to_numpy() if isinstance(spans, pd.Series) else np.asarray(spans)
    if sig_arr.shape[0] != span_arr.shape[0]:
        raise ValueError(
            f"signals length {sig_arr.shape[0]} != spans length {span_arr.shape[0]}")
    n_bars = int(sig_arr.shape[0])
    sig_pos = np.flatnonzero(sig_arr)
    piv_pos = np.flatnonzero(span_arr > 0)
    weights = span_arr[piv_pos].astype(float) / float(REFERENCE_N)
    total_mass = float(weights.sum())
    n_signals = int(sig_pos.size)

    if n_signals == 0 or piv_pos.size == 0:
        return WeightedStats(
            tp_mass=0.0, total_mass=total_mass,
            n_signals=float(n_signals), n_unmatched=float(n_signals),
            n_bars=float(n_bars),
        )

    diffs = piv_pos[None, :] - sig_pos[:, None]
    cost = np.abs(diffs).astype(float) - TIEBREAK_EPS * weights[None, :]
    cost[np.abs(diffs) > MATCH_WINDOW] = _HUNGARIAN_INF
    sig_idx, piv_idx = linear_sum_assignment(cost)
    valid = cost[sig_idx, piv_idx] < _HUNGARIAN_INF_THRESHOLD
    tp_mass = float(weights[piv_idx[valid]].sum())
    n_matched = int(valid.sum())
    return WeightedStats(
        tp_mass=tp_mass, total_mass=total_mass,
        n_signals=float(n_signals), n_unmatched=float(n_signals - n_matched),
        n_bars=float(n_bars),
    )


# ---------------------------------------------------------------------------
# §2.2 — weighted precision / recall / composite side score
# ---------------------------------------------------------------------------


def weighted_precision(stats: WeightedStats, w_fp: float = W_FP) -> float:
    """``precision_w = tp_mass / (tp_mass + n_unmatched_signals * w_FP)``."""
    denom = stats.tp_mass + stats.n_unmatched * float(w_fp)
    return float(stats.tp_mass / denom) if denom > 0 else 0.0


def weighted_recall(stats: WeightedStats) -> float:
    """``recall_w = tp_mass / total_mass`` (0 when the slice has no mass)."""
    if stats.total_mass <= 0:
        return 0.0
    return float(stats.tp_mass / stats.total_mass)


def recall_target_eff(n_eff: float) -> float:
    """Mass-based recall target (spec §2.2, F4).

    ``recall_target_eff = RECALL_TARGET * sqrt(REFERENCE_MASS / N_eff)`` with
    ``N_eff = total_mass`` — mass is normalized to REFERENCE_N units by
    ``w = N*/100``, so N_eff reads as the equivalent number of N=100 pivots.
    """
    if n_eff <= 0:
        return float(RECALL_TARGET)
    return float(RECALL_TARGET * np.sqrt(REFERENCE_MASS / float(n_eff)))


def compute_side_score_v5(
    stats: WeightedStats,
    w_fp: float = W_FP,
    return_components: bool = False,
) -> Union[float, tuple[float, dict[str, float]]]:
    """Composite v5 side score on weighted quantities (spec §2.2).

    Same FORM as the v4 single-scale composite (PRECISION_EXPONENT, smooth
    recall saturation, MIN_RATE frequency floor, two-sided excess penalty)
    operating on span-mass precision/recall, with the recall target rescaled
    on ``N_eff = total_mass`` (F4). The two-sided excess penalty compares the
    signal COUNT against ``N_eff`` (pivot mass in REFERENCE_N units — the v4
    count analog).
    """
    precision = weighted_precision(stats, w_fp)
    recall = weighted_recall(stats)
    n_eff = float(stats.total_mass)

    if stats.n_signals <= 0 or stats.n_bars <= 0:
        empty = {
            "precision": 0.0, "recall": 0.0, "recall_saturated": 0.0,
            "frequency_factor": 0.0, "excess_penalty": 0.0,
            "tp_mass": float(stats.tp_mass), "total_mass": n_eff,
            "n_eff": n_eff, "n_signals": float(stats.n_signals),
            "n_unmatched": float(stats.n_unmatched),
        }
        return (0.0, empty) if return_components else 0.0

    target = recall_target_eff(n_eff)
    recall_sat = float(1.0 - np.exp(-recall / max(target, 1e-9))) if precision > 0 else 0.0
    scale_score = (precision ** PRECISION_EXPONENT) * recall_sat

    signal_rate = stats.n_signals / stats.n_bars
    frequency_factor = min(1.0, signal_rate / MIN_RATE)

    n_e = max(float(stats.n_signals), 1.0)
    t_e = max(n_eff, 1.0)
    excess_penalty = 2.0 * n_e * t_e / (n_e * n_e + t_e * t_e)

    final = float(scale_score * frequency_factor * excess_penalty)
    logger.debug(
        "compute_side_score_v5 n_sig=%.1f tp_mass=%.3f total_mass=%.3f "
        "p=%.4f r=%.4f ff=%.3f ep=%.3f final=%.4f",
        stats.n_signals, stats.tp_mass, stats.total_mass,
        precision, recall, frequency_factor, excess_penalty, final,
    )
    components = {
        "precision": float(precision), "recall": float(recall),
        "recall_saturated": float(recall_sat),
        "frequency_factor": float(frequency_factor),
        "excess_penalty": float(excess_penalty),
        "tp_mass": float(stats.tp_mass), "total_mass": n_eff,
        "n_eff": n_eff, "n_signals": float(stats.n_signals),
        "n_unmatched": float(stats.n_unmatched),
    }
    return (final, components) if return_components else final


__all__ = [
    "SCORER_VERSION",
    "SPAN_GRID", "W_FP", "REFERENCE_MASS", "MATCH_WINDOW", "TIEBREAK_EPS",
    "WeightedStats", "span_weight", "label_pivot_spans", "compute_left_span",
    "match_signals_weighted", "weighted_precision", "weighted_recall",
    "recall_target_eff", "compute_side_score_v5",
]
