"""Speculatores Pivot Optimizer — scoring module (Path A surgical revision).

Changes vs. pre-PathA:

- ``label_pivots``: ambiguous bars (both ph and pl in the same window) are
  labelled 0 instead of +1; consecutive same-sign labels (plateaus) are
  collapsed to the first bar of each run so ``total_pivots`` no longer
  over-counts flat regions.
- ``precision_at_n_stats``: forward lookahead is capped at
  ``LEAD_WINDOW_CAP`` bars; this prevents large scales from accepting any
  signal as a TP simply because the lookahead window is huge.
- ``compute_side_score``: a single coherent formula replaces the previous
  stack of four multiplicative low-count clamps. Per-scale contribution is
  ``precision^PRECISION_EXPONENT * min(1, recall / RECALL_TARGET)`` weighted
  by ``log(N)/log(500)`` and renormalised by the sum of weights so the raw
  score lies in ``[0, 1]``. After the loop the score is multiplied by
  ``frequency_factor`` (floor) and ``excess_penalty`` (anti-spam cap).
- ``_fold_score``: returns ``max(0, oos - GAMMA * max(0, is - oos))`` —
  clamped at 0 so the optimiser never sees an arbitrarily negative score,
  and IS/OOS use the **same** scale set so the gap is not contaminated by
  scales that fit IS but not OOS.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PIVOT_SCALES: list[int] = [5, 10, 20, 50, 100, 200, 500]
MIN_RATE: float = 0.001          # 0.1% signal rate floor
GAMMA: float = 2.0               # IS-OOS overfit penalty multiplier
LEAD_WINDOW_CAP: int = 30        # Cap forward TP-window at 30 bars
PRECISION_EXPONENT: float = 1.2  # Precision-leaning per-scale score
RECALL_TARGET: float = 0.40      # Recall saturates here


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
    high = df["high"]
    low = df["low"]

    ph = (
        high.rolling(2 * N + 1, center=True)
        .apply(lambda w: w[N] == w.max(), raw=True)
        .fillna(0)
        .astype(bool)
    )
    pl = (
        low.rolling(2 * N + 1, center=True)
        .apply(lambda w: w[N] == w.min(), raw=True)
        .fillna(0)
        .astype(bool)
    )

    result = np.zeros(len(df), dtype=np.int8)
    # Symmetric tie-break: bars satisfying both ph and pl are ambiguous → 0.
    result[(ph & ~pl).values] = 1
    result[(pl & ~ph).values] = -1
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
    """Lead-aware precision at scale N for one side, with capped lookahead.

    A signal at bar ``t`` is a true positive if a pivot of the correct sign
    exists anywhere in ``[t-2, t + min(n, LEAD_WINDOW_CAP)]``. The
    backward allowance accounts for short detector lag; the forward cap
    prevents large scales from accepting any signal as TP merely because
    the lookahead window is huge.

    Args:
        signals: Boolean Series — True where a signal fired.
        pivots: Series of {+1, -1, 0} — pivot labels at scale N.
        side: "high" (match +1 pivots) or "low" (match -1 pivots).
        n: Lookahead window size (capped at LEAD_WINDOW_CAP).

    Returns:
        Dict with precision, tp, fp, n_signals, matched_pivots, total_pivots.
    """
    if side not in ("high", "low"):
        raise ValueError(f"side must be 'high' or 'low', got {side!r}")
    pivot_sign = 1 if side == "high" else -1
    signal_bars = signals[signals].index
    pivot_positions = np.flatnonzero((pivots.values == pivot_sign))
    forward_lookahead = min(n, LEAD_WINDOW_CAP)

    tp = 0
    fp = 0
    matched_pivots: set[int] = set()
    for t in signal_bars:
        pos = signals.index.get_loc(t)
        window_start = max(0, pos - 2)
        window_end = min(len(pivots) - 1, pos + forward_lookahead)
        window = pivots.iloc[window_start : window_end + 1]
        if (window == pivot_sign).any():
            tp += 1
            window_positions = pivot_positions[
                (pivot_positions >= window_start) & (pivot_positions <= window_end)
            ]
            matched_pivots.update(int(idx) for idx in window_positions.tolist())
        else:
            fp += 1

    n_signals = tp + fp
    precision = (tp / n_signals) if n_signals else 0.0
    return {
        "precision": precision,
        "tp": tp,
        "fp": fp,
        "n_signals": n_signals,
        "matched_pivots": len(matched_pivots),
        "total_pivots": int(len(pivot_positions)),
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
) -> float:
    """Score one detector side on a single slice (IS or OOS).

    Per-scale contribution::

        precision^PRECISION_EXPONENT * min(1, recall / RECALL_TARGET)

    weighted by ``log(N) / log(500)`` and renormalised by the sum of weights
    so the raw score lies in ``[0, 1]``. Multiplied by:

    - ``frequency_factor = min(1, signal_rate / MIN_RATE)`` — floor against
      detectors that fire too rarely to be useful.
    - ``excess_penalty = min(1, avg_total_pivots / max(n_signals, 1))`` —
      anti-spam cap: firing far more often than the average pivot count
      across the scoring scales is penalised proportionally.

    Args:
        df: OHLCV slice to score (must contain or be augmentable with
            ``pivot_N{N}`` columns).
        signals: Boolean Series — True where the detector fired.
        side: "high" or "low".
        scales: Optional explicit scale list; if None, all scales N with
            ``n_bars >= 2N+1`` are used.

    Returns:
        Float score in ``[0, 1]``.
    """
    if side not in ("high", "low"):
        raise ValueError(f"side must be 'high' or 'low', got {side!r}")
    n_bars = len(df)
    n_signals = int(signals.fillna(False).astype(bool).sum())
    if n_bars == 0 or n_signals == 0:
        return 0.0

    if scales is None:
        scales = _valid_scales(n_bars)
    else:
        scales = [N for N in scales if n_bars >= 2 * N + 1]
    if not scales:
        return 0.0

    signal_rate = n_signals / n_bars
    frequency_factor = min(1.0, signal_rate / MIN_RATE)

    missing = [f"pivot_N{N}" for N in scales if f"pivot_N{N}" not in df.columns]
    if missing:
        df = add_pivot_labels(df)

    raw_score = 0.0
    weight_sum = 0.0
    total_pivots_per_scale: list[float] = []
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
            recall_sat = min(1.0, recall / RECALL_TARGET)
            scale_score = (precision ** PRECISION_EXPONENT) * recall_sat
        else:
            scale_score = 0.0
        weight = np.log(N) / np.log(500)
        raw_score += weight * scale_score
        weight_sum += weight

    normalized_raw = raw_score / weight_sum if weight_sum > 0 else 0.0

    avg_total_pivots = (
        sum(total_pivots_per_scale) / len(total_pivots_per_scale)
        if total_pivots_per_scale
        else 0.0
    )
    excess_penalty = min(1.0, avg_total_pivots / max(n_signals, 1))

    final = normalized_raw * frequency_factor * excess_penalty
    logger.debug(
        "compute_side_score side=%s n_sig=%d rate=%.4f ff=%.3f raw=%.4f ep=%.3f final=%.4f",
        side, n_signals, signal_rate, frequency_factor,
        normalized_raw, excess_penalty, final,
    )
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
    """Per-fold OOS score with IS-OOS overfit penalty, clamped at 0.

    Both halves are scored on the **same** scale set (the intersection of
    scales valid on both IS and OOS), so the gap is not contaminated by
    scales that fit one half but not the other.

    Returns ``max(0, oos - GAMMA * max(0, is - oos))``.
    """
    common_n = min(len(df_is), len(df_oos))
    scales = _valid_scales(common_n)
    is_score = compute_side_score(df_is, sig_is, side, scales=scales)
    oos_score = compute_side_score(df_oos, sig_oos, side, scales=scales)
    penalised = oos_score - GAMMA * max(0.0, is_score - oos_score)
    return max(0.0, penalised)


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
