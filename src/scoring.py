"""Speculatores Pivot Optimizer — scoring module.

Provides pivot ground-truth labeling at multiple time scales, lead-aware
precision computation, and composite scoring with frequency floor and
IS/OOS overfitting penalty used by the Optuna objective.
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
MIN_RATE: float = 0.003   # Minimum signal frequency (smooth floor denominator)
GAMMA: float = 2.0        # IS-OOS overfitting penalty multiplier
SCALE_SIGNAL_FLOOR: int = 4
MATCHED_PIVOT_FLOOR: int = 4
POSTERIOR_ALPHA: float = 1.0
POSTERIOR_BETA: float = 4.0


# ---------------------------------------------------------------------------
# Pivot ground-truth labeling
# ---------------------------------------------------------------------------


def label_pivots(df: pd.DataFrame, N: int) -> pd.Series:
    """Label pivot highs (+1) and pivot lows (-1) at scale N.

    A bar is a pivot high when its ``high`` is the maximum in the symmetric
    window [i-N, i+N] (width 2N+1, centered).  A bar is a pivot low when
    its ``low`` is the minimum in the same window.

    Non-causal: the last N bars will always be 0 due to the rolling window
    needing future data.

    Args:
        df: OHLCV DataFrame with at minimum ``high`` and ``low`` columns.
        N: Half-window size (number of bars on each side of center).

    Returns:
        Series with values +1 (pivot high), -1 (pivot low), or 0.
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

    result = pd.Series(0, index=df.index, dtype=np.int8)
    result[ph] = 1
    result[pl & ~ph] = -1
    return result


def add_pivot_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``pivot_N{scale}`` columns for all PIVOT_SCALES to *df*.

    Modifies *df* in-place and also returns it for chaining.

    Args:
        df: OHLCV DataFrame (must have ``high`` and ``low`` columns).

    Returns:
        df with added pivot label columns.

    Precondition:
        ``df`` must be an owned DataFrame, not a view or slice.
        Callers should pass ``df.copy()`` if working with a slice.
    """
    for N in PIVOT_SCALES:
        col = f"pivot_N{N}"
        if col not in df.columns:
            df[col] = label_pivots(df, N)
            logger.info("Added column %s", col)
    return df


# ---------------------------------------------------------------------------
# Lead-aware precision
# ---------------------------------------------------------------------------


def precision_at_n_stats(
    signals: pd.Series,
    pivots: pd.Series,
    side: str,
    n: int,
) -> dict[str, float | int]:
    """Lead-aware precision at scale N for one side.

    A signal at bar ``t`` is a true positive if a pivot of the correct sign
    exists anywhere in the window ``[t-2, t+n]``.  The backward allowance of
    2 bars accounts for the n-bar confirmation lag of ``ta.pivothigh/pivotlow``.

    Args:
        signals: Boolean Series — True where a signal fired.
        pivots: Series of {+1, -1, 0} — pivot labels at scale N.
        side: "high" (match +1 pivots) or "low" (match -1 pivots).
        n: Lookahead window size (same as pivot scale N).

    Returns:
        Dict with precision, tp, fp, n_signals, matched_pivots, total_pivots.
    """
    if side not in ("high", "low"):
        raise ValueError(f"side must be 'high' or 'low', got {side!r}")
    pivot_sign = 1 if side == "high" else -1
    signal_bars = signals[signals].index
    pivot_positions = np.flatnonzero((pivots.values == pivot_sign))

    tp = 0
    fp = 0
    matched_pivots: set[int] = set()
    for t in signal_bars:
        pos = signals.index.get_loc(t)
        window_start = max(0, pos - 2)
        window_end = min(len(pivots) - 1, pos + n)
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
# Side score
# ---------------------------------------------------------------------------


def compute_side_score(
    df: pd.DataFrame,
    signals: pd.Series,
    side: str,
) -> float:
    """Precision-weighted score for one detector side.

    Computes a weighted sum of precision_at_n over all PIVOT_SCALES, then
    applies a smooth frequency floor factor (``min(1, rate / MIN_RATE)``).

    Args:
        df: OHLCV DataFrame (pivot label columns are added lazily if missing).
        signals: Boolean Series aligned to df's index — True where signal fired.
        side: "high" or "low".

    Returns:
        Raw score in [0, ~7] * frequency_factor ∈ [0, 1].
        In practice bounded by the sum of log-weights ≤ 1.0 per scale.
    """
    if side not in ("high", "low"):
        raise ValueError(f"side must be 'high' or 'low', got {side!r}")
    n_bars = len(df)
    n_signals = int(signals.fillna(False).astype(bool).sum())
    if n_bars == 0:
        return 0.0
    signal_rate = n_signals / n_bars
    frequency_factor = min(1.0, signal_rate / MIN_RATE)

    # Lazy pivot labeling
    missing = [f"pivot_N{N}" for N in PIVOT_SCALES if f"pivot_N{N}" not in df.columns]
    if missing:
        df = add_pivot_labels(df)

    raw_score = 0.0
    for N in PIVOT_SCALES:
        col = f"pivot_N{N}"
        pivots = df[col]
        stats = precision_at_n_stats(signals, pivots, side, N)
        posterior_precision = (
            (float(stats["tp"]) + POSTERIOR_ALPHA)
            / (float(stats["n_signals"]) + POSTERIOR_ALPHA + POSTERIOR_BETA)
            if int(stats["n_signals"]) > 0
            else 0.0
        )
        signal_support = min(1.0, float(stats["n_signals"]) / SCALE_SIGNAL_FLOOR)
        pivot_support = min(1.0, float(stats["matched_pivots"]) / MATCHED_PIVOT_FLOOR)
        support = signal_support * pivot_support
        weight = np.log(N) / np.log(500)
        raw_score += weight * posterior_precision * support

    logger.debug(
        "compute_side_score side=%s n_signals=%d rate=%.4f ff=%.3f raw=%.4f",
        side, n_signals, signal_rate, frequency_factor, raw_score,
    )
    return raw_score * frequency_factor


# ---------------------------------------------------------------------------
# Per-fold objective contributions
# ---------------------------------------------------------------------------


def _fold_score(
    df_is: pd.DataFrame,
    df_oos: pd.DataFrame,
    sig_is: pd.Series,
    sig_oos: pd.Series,
    side: str,
) -> float:
    """Private helper: per-fold OOS score with IS-OOS overfitting penalty.

    Returns OOS score penalized for IS-OOS gap:
        oos_score - GAMMA * max(0, is_score - oos_score)

    Args:
        df_is: In-sample OHLCV DataFrame.
        df_oos: Out-of-sample OHLCV DataFrame.
        sig_is: Boolean signals on in-sample period.
        sig_oos: Boolean signals on out-of-sample period.
        side: "high" or "low".

    Returns:
        Penalized OOS score (float).
    """
    is_score = compute_side_score(df_is, sig_is, side)
    oos_score = compute_side_score(df_oos, sig_oos, side)
    return oos_score - GAMMA * max(0.0, is_score - oos_score)


def fold_score_high(
    df_is: pd.DataFrame,
    df_oos: pd.DataFrame,
    sig_is: pd.Series,
    sig_oos: pd.Series,
) -> float:
    """Per-fold contribution for the HIGH side study.

    Returns OOS score penalized for IS-OOS gap:
        oos_score - GAMMA * max(0, is_score - oos_score)

    Args:
        df_is: In-sample OHLCV DataFrame.
        df_oos: Out-of-sample OHLCV DataFrame.
        sig_is: Boolean signals on in-sample period.
        sig_oos: Boolean signals on out-of-sample period.

    Returns:
        Penalized OOS score (float).
    """
    return _fold_score(df_is, df_oos, sig_is, sig_oos, "high")


def fold_score_low(
    df_is: pd.DataFrame,
    df_oos: pd.DataFrame,
    sig_is: pd.Series,
    sig_oos: pd.Series,
) -> float:
    """Per-fold contribution for the LOW side study.

    Returns OOS score penalized for IS-OOS gap:
        oos_score - GAMMA * max(0, is_score - oos_score)

    Args:
        df_is: In-sample OHLCV DataFrame.
        df_oos: Out-of-sample OHLCV DataFrame.
        sig_is: Boolean signals on in-sample period.
        sig_oos: Boolean signals on out-of-sample period.

    Returns:
        Penalized OOS score (float).
    """
    return _fold_score(df_is, df_oos, sig_is, sig_oos, "low")


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    logger.info("Starting scoring.py self-test...")

    # --- Build synthetic DataFrame with known pivots ---
    np.random.seed(42)
    n = 1000

    # Construct a price series with a prominent peak at bar 200 and a trough at bar 700
    t_arr = np.linspace(0, 4 * np.pi, n)
    base_price = 100.0 + 10.0 * np.sin(t_arr) + 0.5 * np.random.randn(n)

    # Inject a sharp peak at bar 200
    base_price[198:203] += np.array([1, 3, 6, 3, 1])
    # Inject a sharp trough at bar 700
    base_price[698:703] -= np.array([1, 3, 6, 3, 1])

    high_arr = base_price + np.abs(0.3 * np.random.randn(n))
    low_arr = base_price - np.abs(0.3 * np.random.randn(n))

    # Ensure peak high is prominent at bar 200
    high_arr[200] = base_price[200] + 10.0
    # Ensure trough low is prominent at bar 700
    low_arr[700] = base_price[700] - 10.0

    df = pd.DataFrame(
        {
            "open": base_price,
            "high": high_arr,
            "low": low_arr,
            "close": base_price,
            "volume": np.ones(n) * 1000,
        }
    )

    # --- Test 1: label_pivots returns +1 at known peak ---
    pivots_5 = label_pivots(df, N=5)
    assert pivots_5.iloc[200] == 1, (
        f"Expected +1 at bar 200, got {pivots_5.iloc[200]}"
    )
    logger.info("Test 1 PASSED: label_pivots returns +1 at known peak (bar 200)")

    # --- Test 2: label_pivots returns -1 at known trough ---
    assert pivots_5.iloc[700] == -1, (
        f"Expected -1 at bar 700, got {pivots_5.iloc[700]}"
    )
    logger.info("Test 2 PASSED: label_pivots returns -1 at known trough (bar 700)")

    # --- Test 3: add_pivot_labels adds all columns ---
    df = add_pivot_labels(df)
    for N in PIVOT_SCALES:
        col = f"pivot_N{N}"
        assert col in df.columns, f"Missing column {col}"
    logger.info("Test 3 PASSED: add_pivot_labels added all %d pivot columns", len(PIVOT_SCALES))

    # --- Test 4: precision_at_n_stats returns 1.0 when signals align with pivots at N=5 ---
    pivot_high_bars = df.index[df["pivot_N5"] == 1]
    if len(pivot_high_bars) == 0:
        logger.warning("No pivot highs found at N=5; skipping precision test")
    else:
        # Signals exactly at pivot high bars → precision should be 1.0
        perfect_signals = pd.Series(False, index=df.index)
        perfect_signals.iloc[pivot_high_bars[:5]] = True
        stats = precision_at_n_stats(perfect_signals, df["pivot_N5"], "high", 5)
        assert stats["precision"] == 1.0, f"Expected precision 1.0, got {stats['precision']:.4f}"
        assert stats["matched_pivots"] >= 1
        logger.info("Test 4 PASSED: precision_at_n_stats returns 1.0 for perfectly aligned signals")

    # --- Test 5: one lucky perfect pivot is penalized vs. several matched pivots ---
    one_signal = pd.Series(False, index=df.index)
    many_signals = pd.Series(False, index=df.index)
    if len(pivot_high_bars) >= 4:
        one_signal.iloc[pivot_high_bars[:1]] = True
        many_signals.iloc[pivot_high_bars[:4]] = True
        score_one = compute_side_score(df, one_signal, "high")
        score_many = compute_side_score(df, many_signals, "high")
        assert score_many > score_one, (
            f"Expected multi-pivot score {score_many:.4f} > one-pivot score {score_one:.4f}"
        )
        logger.info(
            "Test 5 PASSED: distinct-pivot support works (one=%.4f, many=%.4f)",
            score_one,
            score_many,
        )

    # --- Test 6: compute_side_score returns value in [0, theoretical max] ---
    # Use a moderate signal rate (~1% of bars)
    signal_idx = np.random.choice(df.index, size=20, replace=False)
    signals_high = pd.Series(False, index=df.index)
    signals_high.loc[signal_idx] = True

    score_high = compute_side_score(df, signals_high, "high")
    # The theoretical max raw_score equals sum of weights ≈ 4.46 (unnormalized),
    # so after * frequency_factor the result won't exceed ~4.5.
    # We just verify it is non-negative and not absurdly large.
    assert score_high >= 0.0, f"score_high must be >= 0, got {score_high}"
    max_weight_sum = sum(np.log(N) / np.log(500) for N in PIVOT_SCALES)
    assert score_high <= max_weight_sum + 1e-9, (
        f"score_high={score_high:.4f} exceeds theoretical max {max_weight_sum:.4f}"
    )
    logger.info("Test 6 PASSED: compute_side_score = %.4f (in valid range [0, %.4f])",
                score_high, max_weight_sum)

    # --- Test 7: fold_score_high / fold_score_low run without error ---
    mid = len(df) // 2
    df_is = df.iloc[:mid].copy()
    df_oos = df.iloc[mid:].copy()
    sig_is = signals_high.iloc[:mid]
    sig_oos = signals_high.iloc[mid:]

    fs_high = fold_score_high(df_is, df_oos, sig_is, sig_oos)
    fs_low = fold_score_low(df_is, df_oos, sig_is, sig_oos)
    assert isinstance(fs_high, float)
    assert isinstance(fs_low, float)
    logger.info("Test 7 PASSED: fold_score_high=%.4f  fold_score_low=%.4f", fs_high, fs_low)

    logger.info("scoring.py self-test PASSED")
    print("\nscoring.py self-test PASSED")
    sys.exit(0)
