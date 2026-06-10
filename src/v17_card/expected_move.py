"""Signal Card §3.3/§3.6 — expected move (c_side) and conviction (spec FIXED v3).

What is displayed must be what was fitted (F2): ``c_side`` is regressed on
the LIVE-COMPUTABLE regressor ``x = sigma_HAR(t) * sqrt(E[hold at fire])``,
never on the true span. A chronological two-fold split de-biases the fit
(the survival table feeding E[hold] is estimated from the same signals);
the FINAL exported table is still fitted on all signals.

Censored-span responses are EXCLUDED (re-review R3 — no observable
|move to span-end|; downward length bias, documented in the calibration
report), and the response exists only for MATCHED signals (the displayed
expected move is conditional on match).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from ..scoring_v5 import SPAN_GRID
from .conditioning import expected_hold
from .survival import fit_km, km_table

logger = logging.getLogger(__name__)

#: Below this fit R^2 the card shows the pooled median move (spec §3.3).
R2_FLOOR: float = 0.1

#: Minimum pooled out-of-fold pairs for a meaningful through-origin fit.
MIN_FIT_PAIRS: int = 4


@dataclass(frozen=True)
class SignalRecord:
    """One historical signal's calibration row (spec §3.1-§3.3).

    ``right_span``/``right_censored`` are the R1 survival inputs (EVERY
    signal enters the table); ``realized_abs_move`` is the |move to
    span-end| response (defined only for matched, non-span-censored
    signals).
    """

    fire_bar: int
    sigma_har: float
    left_span: float
    right_span: float
    right_censored: bool
    matched: bool
    span_censored: bool
    realized_abs_move: float


@dataclass(frozen=True)
class CSideFit:
    """Two-fold de-biased through-origin fit of ``c_side`` (F2).

    ``r_squared`` is the conventional centered R^2 of the through-origin
    predictions (1 - SS_res/SS_tot against the mean baseline); when it is
    below ``R2_FLOOR`` the card falls back to ``fallback_median``.
    """

    c_side: float
    r_squared: float
    n_fit: int
    use_fallback: bool
    fallback_median: float


def _eligible(rec: SignalRecord) -> bool:
    """R3 response filter: matched, span observable, finite, sigma > 0."""
    return (rec.matched and not rec.span_censored
            and np.isfinite(rec.realized_abs_move) and rec.sigma_har > 0.0)


def fit_c_side(
    records: Sequence[SignalRecord],
    grid: Sequence[int] = tuple(SPAN_GRID),
    r2_floor: float = R2_FLOOR,
    min_fit: int = MIN_FIT_PAIRS,
) -> CSideFit:
    """Fit one ``c_side`` scalar per side via the two-fold split (F2).

    Splits signals chronologically into halves A and B; fits the survival
    table on A to compute ``E[hold at fire]`` regressors (k=0, with the F1
    L-clamp) for B and vice versa; pools the out-of-fold pairs and fits one
    least-squares-through-origin slope.

    Args:
        records: All historical signals of the side (matched or not —
            every signal enters the per-fold survival tables, R1).

    Returns:
        ``CSideFit`` with the slope, centered R^2, pooled-median fallback
        and the R3-eligible pair count.
    """
    recs = sorted(records, key=lambda r: r.fire_bar)
    half = len(recs) // 2
    fold_a, fold_b = recs[:half], recs[half:]

    xs: list[float] = []
    ys: list[float] = []
    for fit_fold, eval_fold in ((fold_a, fold_b), (fold_b, fold_a)):
        if not fit_fold:
            continue
        curve = fit_km(
            np.array([r.right_span for r in fit_fold], dtype=float),
            np.array([r.right_censored for r in fit_fold], dtype=bool),
        )
        table = km_table(curve, grid)
        for rec in eval_fold:
            if not _eligible(rec):
                continue
            e_hold = expected_hold(table, k=0.0, left_span=rec.left_span,
                                   grid=grid)
            xs.append(rec.sigma_har * float(np.sqrt(max(e_hold, 0.0))))
            ys.append(rec.realized_abs_move)

    moves = np.array([r.realized_abs_move for r in recs if _eligible(r)],
                     dtype=float)
    fallback_median = float(np.median(moves)) if moves.size else float("nan")

    x_arr = np.asarray(xs, dtype=float)
    y_arr = np.asarray(ys, dtype=float)
    sxx = float(np.dot(x_arr, x_arr))
    if x_arr.size < min_fit or sxx <= 0.0:
        logger.warning("fit_c_side: insufficient pairs (n=%d) — fallback",
                       x_arr.size)
        return CSideFit(float("nan"), float("nan"), int(x_arr.size), True,
                        fallback_median)

    c_side = float(np.dot(x_arr, y_arr) / sxx)
    ss_res = float(np.sum((y_arr - c_side * x_arr) ** 2))
    ss_tot = float(np.sum((y_arr - y_arr.mean()) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else 0.0
    use_fallback = r_squared < r2_floor
    logger.info("fit_c_side: c=%.4f R^2=%.3f n_fit=%d fallback=%s",
                c_side, r_squared, x_arr.size, use_fallback)
    return CSideFit(c_side, r_squared, int(x_arr.size), use_fallback,
                    fallback_median)


def expected_move(fit: CSideFit, sigma_har: float, e_hold: float) -> float:
    """Live expected move ``c_side * sigma_HAR * sqrt(E[hold])`` (spec §3.3).

    Falls back to the pooled median move when the fit failed the R^2 floor.
    The value is conditional on match ("expected move if this is a real
    turn", R3) — the caller must label it as such.
    """
    if fit.use_fallback:
        return fit.fallback_median
    return fit.c_side * sigma_har * float(np.sqrt(max(e_hold, 0.0)))


def conviction_percentile(
    value: float, historical: Sequence[float],
) -> float:
    """Mid-rank percentile of ``value`` among historical signals (spec §3.6).

    Pure display ranking on ``P(N*_eff >= 50 | features, k=0) *
    expected_move``; 0-100. Empty history -> 50.0 (uninformative).
    """
    hist = np.asarray(historical, dtype=float)
    if hist.size == 0:
        return 50.0
    less = float(np.count_nonzero(hist < value))
    equal = float(np.count_nonzero(hist == value))
    return 100.0 * (less + 0.5 * equal) / hist.size


__all__ = [
    "R2_FLOOR", "MIN_FIT_PAIRS", "SignalRecord", "CSideFit",
    "fit_c_side", "expected_move", "conviction_percentile",
]
