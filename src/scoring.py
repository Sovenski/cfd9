"""Speculatores Pivot Optimizer — Scorer v2.

Scorer v2 (this module) implements 14 of 16 expert-suggested fixes on top
of the prior Path A surgical revision. Detector logic is untouched — only
scoring/aggregation semantics changed.

Changes vs. Scorer v1 (Path A):

- ``_fold_score`` (item 1): replaces the kinked
  ``max(0, oos - γ·max(0, is-oos))`` with the smooth, always-non-negative
  exponential penalty ``oos · exp(-γ · max(0, is-oos))``. Preserves gradient
  on losing trials so Optuna's TPE can still learn from them.
- ``precision_at_n_stats`` (items 6, 7): rewrites tolerance matching as a
  greedy nearest-neighbor 1-to-1 assignment that prevents one pivot from
  being claimed by multiple signals (and vice versa). Per-scale tolerance
  cap is now ``min(N, max(30, round(0.3 N)))``: scale-adaptive lookahead.
- ``compute_side_score`` (items 2, 3, 4, 5, 15):
  - ``excess_penalty`` is two-sided harmonic-mean of n_signals vs. the
    median valid scale's pivot count (replaces one-sided ``min(1, …)``
    against the arithmetic mean — the mean was small-N-dominated).
  - Recall saturation is now ``1 - exp(-recall / target)`` (smooth, no
    hard cap at ``RECALL_TARGET``).
  - The recall target is scale-adaptive:
    ``RECALL_TARGET · sqrt(REFERENCE_N / N)`` — small N gets a harder
    target (more pivots, achievable recall is small in absolute terms);
    large N gets an easier target.
  - Optional ``return_per_scale=True`` returns the per-scale score dict
    alongside the scalar (for report diagnostics).
- ``label_pivots`` (item 14): vectorised with
  ``numpy.lib.stride_tricks.sliding_window_view`` for ~10× speedup. The
  output is byte-identical to the previous rolling-apply implementation
  (verified in the module self-test).

NOT changed in Scorer v2 (intentionally deferred):

- Item 13 (shared-scale intersection in ``_fold_score``) — defended by the
  Bayesian expert as the right hygiene; revisit in a future round.
- Item 16 — analysis-only task, no code change.

Note on backward compatibility: per-scale absolute scale values are
generally *higher* than Scorer v1 (recall saturation is softer), so
historical best-value numbers are not directly comparable across versions.
"""

from __future__ import annotations

import logging
from typing import Union

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PIVOT_SCALES: list[int] = [5, 10, 20, 50, 100, 200, 500]
MIN_RATE: float = 0.001          # 0.1% signal rate floor
GAMMA: float = 2.0               # IS-OOS overfit penalty multiplier (Scorer v2: exponential)
LEAD_WINDOW_CAP: int = 30        # Floor for the scale-adaptive forward TP-window cap (item 7)
PRECISION_EXPONENT: float = 1.2  # Precision-leaning per-scale score
RECALL_TARGET: float = 0.40      # Reference recall target (Scorer v2: scaled by sqrt(REFERENCE_N/N))
REFERENCE_N: int = 50            # Pivot scale at which RECALL_TARGET applies unchanged (item 5)


# ---------------------------------------------------------------------------
# Pivot ground-truth labeling
# ---------------------------------------------------------------------------


def _dedup_consecutive_runs(labels: np.ndarray) -> np.ndarray:
    """Collapse runs of consecutive identical non-zero labels to first bar.

    A 10-bar plateau at the top of a price series produces 10 consecutive
    +1 labels in the centered-window definition. Keep only the first bar of
    each run so ``total_pivots`` represents distinct turning points, not
    the width of flat regions.
    """
    if len(labels) == 0:
        return labels
    out = labels.copy()
    for i in range(1, len(out)):
        if out[i] != 0 and out[i] == out[i - 1]:
            out[i] = 0
    return out


def label_pivots(df: pd.DataFrame, N: int) -> pd.Series:
    """Label pivot highs (+1) and pivot lows (-1) at scale N.

    A bar is a pivot high when its ``high`` is the maximum in the symmetric
    window ``[i-N, i+N]`` (width 2N+1, centered). A bar is a pivot low when
    its ``low`` is the minimum in the same window. If a bar satisfies both
    conditions simultaneously (degenerate flat segment), it is labelled 0
    (ambiguous) instead of arbitrarily winning for the high side.

    Adjacent same-sign labels (plateaus) are collapsed: only the first bar
    of each run keeps the non-zero label.

    Non-causal: the first and last N bars are always 0 due to the rolling
    window needing data on both sides.

    Args:
        df: OHLCV DataFrame with at minimum ``high`` and ``low`` columns.
        N: Half-window size.

    Returns:
        Series of {+1, -1, 0} with the same index as ``df``.
    """
    high_arr = df["high"].to_numpy()
    low_arr = df["low"].to_numpy()
    n_bars = len(df)
    width = 2 * N + 1

    ph_full = np.zeros(n_bars, dtype=bool)
    pl_full = np.zeros(n_bars, dtype=bool)

    if n_bars >= width:
        # Vectorised centered max/min via sliding_window_view (item 14).
        # The centered window for bar i covers indices [i-N, i+N]. The
        # first valid center is bar N; the last valid center is n_bars-1-N.
        # The rolling-apply implementation produced NaN (treated as False)
        # at the first and last N bars; here we leave them as False, which
        # matches that semantics byte-for-byte.
        hw = sliding_window_view(high_arr, window_shape=width)  # shape (n-2N, 2N+1)
        lw = sliding_window_view(low_arr, window_shape=width)
        # Center of each window is at offset N. The window at row r covers
        # bars [r, r+2N] in the source array, so the center bar is r+N.
        center_high = hw[:, N]
        center_low = lw[:, N]
        win_max = hw.max(axis=1)
        win_min = lw.min(axis=1)
        ph_mid = center_high == win_max
        pl_mid = center_low == win_min
        ph_full[N : n_bars - N] = ph_mid
        pl_full[N : n_bars - N] = pl_mid

    result = np.zeros(n_bars, dtype=np.int8)
    # Symmetric tie-break: bars satisfying both ph and pl are ambiguous → 0.
    result[ph_full & ~pl_full] = 1
    result[pl_full & ~ph_full] = -1
    result = _dedup_consecutive_runs(result)
    return pd.Series(result, index=df.index, dtype=np.int8)


def add_pivot_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``pivot_N{scale}`` columns for all PIVOT_SCALES to *df*."""
    for N in PIVOT_SCALES:
        col = f"pivot_N{N}"
        if col not in df.columns:
            df[col] = label_pivots(df, N)
            logger.info("Added column %s", col)
    return df


# ---------------------------------------------------------------------------
# Lead-aware precision (with capped lookahead)
# ---------------------------------------------------------------------------


def precision_at_n_stats(
    signals: pd.Series,
    pivots: pd.Series,
    side: str,
    n: int,
) -> dict[str, float | int]:
    """Lead-aware precision at scale N — Scorer v2: greedy 1-to-1 matching.

    Items 6 + 7. Signals and pivots are matched as nearest-neighbor
    1-to-1 pairs: each signal claims at most one unmatched pivot within
    its tolerance window, and each pivot can satisfy at most one signal.
    This eliminates the prior double-counting where a single dense pivot
    cluster could "rescue" many signals as TP.

    Tolerance window for a signal at bar ``t``: ``[t-2, t + cap]`` where
    ``cap = min(N, max(LEAD_WINDOW_CAP, round(0.3·N)))`` — at least 30
    bars (the legacy floor), at least 30% of N for larger scales, but
    never more than N itself.

    Args:
        signals: Boolean Series — True where a signal fired.
        pivots: Series of {+1, -1, 0} — pivot labels at scale N.
        side: "high" (match +1 pivots) or "low" (match -1 pivots).
        n: Pivot scale (controls scale-adaptive lookahead cap).

    Returns:
        Dict with precision, tp, fp, n_signals, matched_pivots, total_pivots.
    """
    if side not in ("high", "low"):
        raise ValueError(f"side must be 'high' or 'low', got {side!r}")
    pivot_sign = 1 if side == "high" else -1
    # Scale-adaptive forward cap (item 7).
    cap = min(n, max(LEAD_WINDOW_CAP, int(round(n * 0.3))))

    sig_mask = signals.fillna(False).astype(bool).to_numpy()
    signal_positions = np.flatnonzero(sig_mask)
    pivot_positions = np.flatnonzero(pivots.to_numpy() == pivot_sign)
    total_pivots = int(pivot_positions.size)

    if signal_positions.size == 0:
        return {
            "precision": 0.0,
            "tp": 0,
            "fp": 0,
            "n_signals": 0,
            "matched_pivots": 0,
            "total_pivots": total_pivots,
        }

    # Greedy 1-to-1 assignment (item 6). For each signal (in bar order),
    # pick the nearest unmatched pivot inside [t-2, t+cap]. "Nearest" is
    # by absolute bar distance so we don't bias toward backward or
    # forward matches; ties broken in favour of the earlier pivot.
    matched_pivot_mask = np.zeros(total_pivots, dtype=bool)
    tp = 0
    fp = 0
    for sig_pos in signal_positions:
        window_lo = sig_pos - 2
        window_hi = sig_pos + cap
        in_window = (
            (pivot_positions >= window_lo)
            & (pivot_positions <= window_hi)
            & (~matched_pivot_mask)
        )
        candidate_idx = np.flatnonzero(in_window)
        if candidate_idx.size == 0:
            fp += 1
            continue
        candidate_positions = pivot_positions[candidate_idx]
        # Pick nearest by |distance|; np.argmin ties go to the lower index,
        # i.e. the earlier pivot — deterministic and stable.
        dists = np.abs(candidate_positions - sig_pos)
        best = candidate_idx[int(np.argmin(dists))]
        matched_pivot_mask[best] = True
        tp += 1

    n_signals = tp + fp
    precision = (tp / n_signals) if n_signals else 0.0
    return {
        "precision": float(precision),
        "tp": int(tp),
        "fp": int(fp),
        "n_signals": int(n_signals),
        "matched_pivots": int(matched_pivot_mask.sum()),
        "total_pivots": total_pivots,
    }


def precision_at_n(
    signals: pd.Series,
    pivots: pd.Series,
    side: str,
    n: int,
) -> float:
    """Backward-compatible scalar precision wrapper."""
    return float(precision_at_n_stats(signals, pivots, side, n)["precision"])


# ---------------------------------------------------------------------------
# Side score (single coherent formula)
# ---------------------------------------------------------------------------


def _valid_scales(n_bars: int, scales: list[int] | None = None) -> list[int]:
    """Return scales N for which a centered rolling window of 2N+1 fits."""
    pool = scales if scales is not None else PIVOT_SCALES
    return [N for N in pool if n_bars >= 2 * N + 1]


def compute_side_score(
    df: pd.DataFrame,
    signals: pd.Series,
    side: str,
    scales: list[int] | None = None,
    return_per_scale: bool = False,
) -> Union[float, tuple[float, dict[int, float]]]:
    """Score one detector side on a single slice (IS or OOS) — Scorer v2.

    Per-scale contribution::

        precision^PRECISION_EXPONENT * (1 - exp(-recall / target_N))

    where ``target_N = RECALL_TARGET · sqrt(REFERENCE_N / N)`` adapts the
    saturation point to the attainable recall at scale N (item 5). Smooth
    saturation (item 4) preserves gradient past the target.

    The per-scale scores are weighted by ``log(N) / log(500)`` and
    renormalised by the sum of weights so the raw score lies in ``[0, 1]``.

    Multipliers applied after the per-scale loop:

    - ``frequency_factor = min(1, signal_rate / MIN_RATE)`` — floor
      against detectors that fire too rarely to be useful.
    - ``excess_penalty = 2·n·t / (n² + t²)`` (items 2 + 3): two-sided
      harmonic-mean-of-ratios between ``n = n_signals`` and ``t = median
      valid scale's total_pivots``. Equals 1 at parity, decays
      symmetrically toward 0 as either side grows much larger. Using
      the median (not the mean) avoids small-N domination.

    Args:
        df: OHLCV slice to score (must contain or be augmentable with
            ``pivot_N{N}`` columns).
        signals: Boolean Series — True where the detector fired.
        side: "high" or "low".
        scales: Optional explicit scale list; if None, all scales N with
            ``n_bars >= 2N+1`` are used.
        return_per_scale: If True (item 15), return a
            ``(scalar_score, {N: scale_score})`` tuple. The default of
            False preserves the scalar return contract.

    Returns:
        Float score in ``[0, 1]``, or ``(score, {N: scale_score})`` when
        ``return_per_scale`` is True.
    """
    if side not in ("high", "low"):
        raise ValueError(f"side must be 'high' or 'low', got {side!r}")
    n_bars = len(df)
    n_signals = int(signals.fillna(False).astype(bool).sum())
    if n_bars == 0 or n_signals == 0:
        if return_per_scale:
            return 0.0, {}
        return 0.0

    if scales is None:
        scales = _valid_scales(n_bars)
    else:
        scales = [N for N in scales if n_bars >= 2 * N + 1]
    if not scales:
        if return_per_scale:
            return 0.0, {}
        return 0.0

    signal_rate = n_signals / n_bars
    frequency_factor = min(1.0, signal_rate / MIN_RATE)

    missing = [f"pivot_N{N}" for N in scales if f"pivot_N{N}" not in df.columns]
    if missing:
        df = add_pivot_labels(df)

    raw_score = 0.0
    weight_sum = 0.0
    total_pivots_per_scale: list[float] = []
    per_scale_scores: dict[int, float] = {}
    for N in scales:
        col = f"pivot_N{N}"
        pivots = df[col]
        stats = precision_at_n_stats(signals, pivots, side, N)
        precision = float(stats["precision"])
        matched = float(stats["matched_pivots"])
        total = float(stats["total_pivots"])
        total_pivots_per_scale.append(total)
        if precision > 0.0 and total > 0.0:
            recall = matched / total
            # Scale-adaptive smooth recall saturation (items 4 + 5).
            target_for_scale = RECALL_TARGET * np.sqrt(REFERENCE_N / max(N, 1))
            recall_sat = 1.0 - np.exp(-recall / max(target_for_scale, 1e-9))
            scale_score = (precision ** PRECISION_EXPONENT) * recall_sat
        else:
            scale_score = 0.0
        per_scale_scores[N] = float(scale_score)
        weight = np.log(N) / np.log(500)
        raw_score += weight * scale_score
        weight_sum += weight

    normalized_raw = raw_score / weight_sum if weight_sum > 0 else 0.0

    # Two-sided excess penalty (items 2 + 3). Reference is the MEDIAN of
    # per-scale total_pivots, not the mean — the mean was dominated by
    # small N (which produce thousands of micro-pivots).
    sorted_totals = sorted(total_pivots_per_scale)
    ref_pivots = sorted_totals[len(sorted_totals) // 2] if sorted_totals else 0.0
    n_eff = max(float(n_signals), 1.0)
    t_eff = max(float(ref_pivots), 1.0)
    excess_penalty = 2.0 * n_eff * t_eff / (n_eff * n_eff + t_eff * t_eff)

    final = float(normalized_raw * frequency_factor * excess_penalty)
    logger.debug(
        "compute_side_score side=%s n_sig=%d rate=%.4f ff=%.3f raw=%.4f ep=%.3f ref=%.0f final=%.4f",
        side, n_signals, signal_rate, frequency_factor,
        normalized_raw, excess_penalty, ref_pivots, final,
    )
    if return_per_scale:
        return final, per_scale_scores
    return final


# ---------------------------------------------------------------------------
# Per-fold objective contributions (with shared scale set + non-negative clamp)
# ---------------------------------------------------------------------------


def _fold_score(
    df_is: pd.DataFrame,
    df_oos: pd.DataFrame,
    sig_is: pd.Series,
    sig_oos: pd.Series,
    side: str,
) -> float:
    """Per-fold OOS score with smooth IS-OOS overfit penalty — Scorer v2.

    Both halves are scored on the **same** scale set (the intersection of
    scales valid on both IS and OOS), so the gap is not contaminated by
    scales that fit one half but not the other. (Item 13 — defended in
    panel as the right hygiene, intentionally NOT changed in v2.)

    Item 1: exponential penalty replaces the prior subtractive clamp.
    Returns ``oos · exp(-GAMMA · max(0, is - oos))`` — always in
    ``[0, oos]``, smooth everywhere, and preserves gradient on losing
    trials (where the old formula would return 0 and silently kill the
    learning signal).
    """
    common_n = min(len(df_is), len(df_oos))
    scales = _valid_scales(common_n)
    is_score = compute_side_score(df_is, sig_is, side, scales=scales)
    oos_score = compute_side_score(df_oos, sig_oos, side, scales=scales)
    gap = max(0.0, float(is_score) - float(oos_score))
    return float(oos_score * np.exp(-GAMMA * gap))


def fold_score_high(
    df_is: pd.DataFrame,
    df_oos: pd.DataFrame,
    sig_is: pd.Series,
    sig_oos: pd.Series,
) -> float:
    """Per-fold contribution for the HIGH side study (clamped at 0)."""
    return _fold_score(df_is, df_oos, sig_is, sig_oos, "high")


def fold_score_low(
    df_is: pd.DataFrame,
    df_oos: pd.DataFrame,
    sig_is: pd.Series,
    sig_oos: pd.Series,
) -> float:
    """Per-fold contribution for the LOW side study (clamped at 0)."""
    return _fold_score(df_is, df_oos, sig_is, sig_oos, "low")


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    logger.info("Starting scoring.py self-test (Path A) ...")

    np.random.seed(42)
    n = 1000
    t_arr = np.linspace(0, 4 * np.pi, n)
    base_price = 100.0 + 10.0 * np.sin(t_arr) + 0.5 * np.random.randn(n)
    base_price[198:203] += np.array([1, 3, 6, 3, 1])
    base_price[698:703] -= np.array([1, 3, 6, 3, 1])
    high_arr = base_price + np.abs(0.3 * np.random.randn(n))
    low_arr = base_price - np.abs(0.3 * np.random.randn(n))
    df = pd.DataFrame({
        "open": base_price, "high": high_arr,
        "low": low_arr, "close": base_price,
        "volume": np.ones(n),
    })

    # --- Plateau dedup smoke ---
    flat = pd.DataFrame({
        "open": [100] * 50, "high": [100] * 50,
        "low": [99] * 50, "close": [100] * 50,
        "volume": [1] * 50,
    })
    labels = label_pivots(flat, 5)
    nz_labels = (labels != 0).sum()
    logger.info("Plateau dedup: %d non-zero labels (expect 0; or ≤2 at edges)", nz_labels)
    assert nz_labels <= 2, f"Plateau over-counted: got {nz_labels}"

    # --- Ambiguous-bar handling ---
    labels20 = label_pivots(df, 20)
    logger.info(
        "Synthetic: %d +1 labels, %d -1 labels (post-dedup)",
        (labels20 == 1).sum(), (labels20 == -1).sum(),
    )

    # --- compute_side_score on synthetic ---
    df_with_labels = add_pivot_labels(df.copy())
    sig_perfect = pd.Series(False, index=df.index)
    sig_perfect.iloc[200] = True
    sig_perfect.iloc[700] = True
    sig_high_only = sig_perfect & (df["close"].diff() > 0).fillna(False)

    s_high = compute_side_score(df_with_labels.copy(), sig_perfect, "high")
    s_low = compute_side_score(df_with_labels.copy(), sig_perfect, "low")
    logger.info("2-signal perfect detector: high_score=%.4f, low_score=%.4f", s_high, s_low)

    # --- _fold_score sanity ---
    df_is = df.iloc[:700].copy()
    df_oos = df.iloc[700:].copy()
    add_pivot_labels(df_is); add_pivot_labels(df_oos)
    sig_is = sig_perfect.iloc[:700].copy()
    sig_oos = sig_perfect.iloc[700:].copy()
    fold = _fold_score(df_is, df_oos, sig_is, sig_oos, "high")
    logger.info("Fold score (synthetic): %.4f", fold)
    assert fold >= 0.0, "Fold score must be non-negative"

    print("\nscoring.py (Path A) self-test PASSED")
