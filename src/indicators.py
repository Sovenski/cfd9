"""Speculatores Pivot Optimizer — indicator primitives and precomputation.

Faithful translation of Pine Script indicator functions to pandas/numpy.
Includes the frozen Params dataclass, all primitive functions, and the
SMA + PIR matrix precomputation used to accelerate the Optuna loop.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------


def nz(val: pd.Series, default: float = 0.0) -> pd.Series:
    """Pine `nz` — replace NaN with *default*."""
    return val.fillna(default)


# ---------------------------------------------------------------------------
# Params dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Params:
    """All detector parameters for both HIGH and LOW sides.

    Defaults correspond to the Gold 1D Current preset.
    """

    # --- HIGH side ---
    S_detect_high: int = 12
    scale_start_high: int = 3
    scale_end_high: int = 270
    scale_step_high: int = 3
    min_duration_high: int = 13
    cooldown_bars_high: int = 9
    price_gate_lb_high: int = 23
    vola_range_len_high: int = 120
    er_period_high: int = 28
    pct_extreme_high: float = 0.96
    min_agreement_high: float = 0.75
    dur_extreme_pct_high: float = 0.83
    confirm_count_high: int = 5
    vol_surge_thresh_high: float = 1.5
    scale_div_thresh_high: float = 0.35
    slope_thresh_high: float = 0.22
    vola_high_pct_high: float = 0.92
    pivot_drift_lookback_high: int = 5
    pivot_drift_thresh_high: float = 0.008
    pivot_drift_gate_mult_high: float = 8.0
    pivot_drift_confirm_bias_high: int = 1
    momentum_velocity_thresh_high: float = 0.014
    er_directional_high: bool = False
    use_trend_high: bool = False
    use_volume_high: bool = False
    use_momentum_high: bool = False
    use_momentum_velocity_high: bool = True
    use_volatility_high: bool = False
    use_er_gate_high: bool = False
    use_gjr_asym_high: bool = False
    use_har_vol_high: bool = False
    gjr_vote_thresh_high: float = 0.15
    har_vote_thresh_high: float = 0.15
    vola_method_high: str = "ATR"
    momentum_velocity_mode_high: str = "Reversal"
    # V15 — edge-triggered voting (HIGH side)
    use_edge_voting_high: bool = False
    edge_window_high: int = 5

    # --- LOW side ---
    S_detect_low: int = 26
    scale_start_low: int = 19
    scale_end_low: int = 250
    scale_step_low: int = 15
    min_duration_low: int = 2
    cooldown_bars_low: int = 7
    price_gate_lb_low: int = 58
    vola_range_len_low: int = 20
    er_period_low: int = 47
    pct_extreme_low: float = 0.85
    min_agreement_low: float = 0.30
    dur_extreme_pct_low: float = 0.72
    confirm_count_low: int = 3
    vol_surge_thresh_low: float = 2.2
    scale_div_thresh_low: float = 0.39
    slope_thresh_low: float = 0.15
    vola_high_pct_low: float = 0.78
    pivot_drift_lookback_low: int = 10
    pivot_drift_thresh_low: float = 0.005
    pivot_drift_gate_mult_low: float = 4.0
    pivot_drift_confirm_bias_low: int = 0
    momentum_velocity_thresh_low: float = 0.007
    er_directional_low: bool = False
    use_trend_low: bool = False
    use_volume_low: bool = False
    use_momentum_low: bool = False
    use_momentum_velocity_low: bool = True
    use_volatility_low: bool = True
    use_er_gate_low: bool = False
    use_gjr_asym_low: bool = False
    use_har_vol_low: bool = False
    gjr_vote_thresh_low: float = 0.15
    har_vote_thresh_low: float = 0.15
    vola_method_low: str = "ATR"
    momentum_velocity_mode_low: str = "Reversal"
    # V15 — edge-triggered voting (LOW side)
    use_edge_voting_low: bool = False
    edge_window_low: int = 5

    # --- Fixed constants (not per-side, not optimized) ---
    baseline_lb: int = 20


# ---------------------------------------------------------------------------
# Pine Script primitive functions
# ---------------------------------------------------------------------------


def sma(series: pd.Series, n: int) -> pd.Series:
    """Simple moving average."""
    return series.rolling(n).mean()


def atr(df: pd.DataFrame, n: int) -> pd.Series:
    """Pine-exact ``ta.atr(n)``: Wilder RMA of true range (NOT an SMA).

    ``ta.atr = ta.rma(ta.tr(true), n)`` — the RMA seeds with the SMA of the
    first *n* TRs, then recurses ``rma[t] = (rma[t-1]*(n-1) + tr[t]) / n``.
    The 2026-06-10 TV-export audit caught the old SMA-of-TR implementation
    flipping the volatility vote (8 LOW signal flips over 155yr of SPX).
    """
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    tr_vals = tr.values.astype(np.float64)
    out = np.full(len(tr_vals), np.nan, dtype=np.float64)
    if len(tr_vals) >= n:
        out[n - 1] = np.mean(tr_vals[:n])
        for t in range(n, len(tr_vals)):
            out[t] = (out[t - 1] * (n - 1) + tr_vals[t]) / n
    return pd.Series(out, index=df.index)


def stdev(series: pd.Series, n: int) -> pd.Series:
    """Pine-exact ``ta.stdev(n)``: POPULATION std (biased, ddof=0).

    Pine's default is ``biased=true`` (divide by n); the old ddof=1 sample
    std diverged from Pine for the "StdDev" volatility method.
    """
    return series.rolling(n).std(ddof=0)


def pir_of(val: pd.Series, lb: int) -> pd.Series:
    """Percent-in-range: normalize *val* to [0, 1] over a rolling window."""
    lo = val.rolling(lb).min()
    hi = val.rolling(lb).max()
    span = (hi - lo).clip(lower=1e-10)
    result = (val - lo) / span
    # When hi == lo the span is effectively zero — return 0.5
    result = result.where(hi != lo, 0.5)
    return result


def pivot_high(series: pd.Series, n: int) -> pd.Series:
    """Non-causal pivot high detector for ground truth labeling ONLY.

    Returns True at bar ``i`` when ``series[i]`` is the maximum value in the
    symmetric window ``[i-n, i+n]`` (width ``2n+1``).

    Non-causal: the result at bar ``i`` depends on ``n`` future bars, so the
    last ``n`` bars of the returned series are always False/NaN.  Bar ``i``
    "fires" only after bar ``i+n`` has been observed.

    **ONLY for ground truth labeling** (e.g. scoring.py).
    NEVER use for real-time signal evaluation.
    """
    window = 2 * n + 1
    rolled = series.rolling(window, center=True).apply(
        lambda w: 1.0 if w[n] == w.max() else 0.0, raw=True
    )
    return rolled.fillna(0).astype(bool)


def pivot_low(series: pd.Series, n: int) -> pd.Series:
    """Non-causal pivot low detector for ground truth labeling ONLY.

    Returns True at bar ``i`` when ``series[i]`` is the minimum value in the
    symmetric window ``[i-n, i+n]`` (width ``2n+1``).

    Non-causal: the result at bar ``i`` depends on ``n`` future bars, so the
    last ``n`` bars of the returned series are always False/NaN.  Bar ``i``
    "fires" only after bar ``i+n`` has been observed.

    **ONLY for ground truth labeling** (e.g. scoring.py).
    NEVER use for real-time signal evaluation.
    """
    window = 2 * n + 1
    rolled = series.rolling(window, center=True).apply(
        lambda w: 1.0 if w[n] == w.min() else 0.0, raw=True
    )
    return rolled.fillna(0).astype(bool)


def pivot_high_pine(series: pd.Series, n: int) -> pd.Series:
    """Pine-exact ``ta.pivothigh(n, n)``: ``>=`` on the LEFT, strict ``>`` on
    the RIGHT — for tied maxima the LATER twin is the pivot.

    Differs from :func:`pivot_high` (``w[n] == w.max()``, ties count BOTH
    twins), which is kept for ground-truth LABELING only. Rule proven
    empirically against the 2026-06-10 TV export (25,249 SPX bars, 346
    pivots, 0 disagreements; the loose rule produced 15 phantom tie-pivots
    and strict-both rejected 13 real ones). Detection-path baseline pivots
    MUST use this variant — they feed the shared confirmed-pivots drift
    stack (parity).
    """
    window = 2 * n + 1
    rolled = series.rolling(window, center=True).apply(
        lambda w: 1.0 if (w[n] >= w[:n].max() and w[n] > w[n + 1:].max()) else 0.0,
        raw=True,
    )
    return rolled.fillna(0).astype(bool)


def pivot_low_pine(series: pd.Series, n: int) -> pd.Series:
    """Pine-exact ``ta.pivotlow(n, n)``: ``<=`` on the LEFT, strict ``<`` on
    the RIGHT — for tied minima the LATER twin is the pivot.

    See :func:`pivot_high_pine`. Detection-path baseline pivots MUST use this.
    """
    window = 2 * n + 1
    rolled = series.rolling(window, center=True).apply(
        lambda w: 1.0 if (w[n] <= w[:n].min() and w[n] < w[n + 1:].min()) else 0.0,
        raw=True,
    )
    return rolled.fillna(0).astype(bool)


def linreg_value(series: pd.Series, length: int) -> pd.Series:
    """Rolling OLS endpoint value (Pine ``ta.linreg(series, length, 0)``)."""

    def _linreg_val(w: np.ndarray) -> float:
        x = np.arange(len(w))
        coeffs = np.polyfit(x, w, 1)
        return coeffs[0] * (len(w) - 1) + coeffs[1]

    return series.rolling(length).apply(_linreg_val, raw=True)


def linreg_slope_step(series: pd.Series, length: int) -> pd.Series:
    """Pine ``ta.linreg(series, length, 0) - ta.linreg(series, length, 1)``.

    This is the fitted one-step slope from the current rolling regression
    window, not the difference between endpoint values from consecutive
    windows.
    """

    def _linreg_step(w: np.ndarray) -> float:
        x = np.arange(len(w))
        coeffs = np.polyfit(x, w, 1)
        return coeffs[0]

    return series.rolling(length).apply(_linreg_step, raw=True)


# ---------------------------------------------------------------------------
# Agreement calculation (slow path — without precomputed matrices)
# ---------------------------------------------------------------------------


def calc_agreement(
    df: pd.DataFrame,
    scale_start: int,
    scale_end: int,
    scale_step: int,
    pct_extreme: float,
    pir_matrix: Optional[np.ndarray] = None,
    scales_list: Optional[list[int]] = None,
) -> tuple[pd.Series, pd.Series, int, pd.Series, pd.Series]:
    """Count extreme PIR fractions across SMA scales.

    Returns:
        (scales_high, scales_low, n_scales,
         agreement_high_fraction, agreement_low_fraction)
    """
    close = df["close"]
    n_bars = len(close)
    scales_high = np.zeros(n_bars, dtype=np.float64)
    scales_low = np.zeros(n_bars, dtype=np.float64)
    n_scales = 0

    for s in range(scale_start, scale_end + 1, scale_step):
        if pir_matrix is not None and scales_list is not None:
            idx = s - scales_list[0]
            if 0 <= idx < pir_matrix.shape[0]:
                pir_vals = pir_matrix[idx]
            else:
                continue
        else:
            sma_vals = close.rolling(s).mean()
            ratio = close / sma_vals
            pir_vals = pir_of(ratio, max(s, 20)).values

        scales_high += pir_vals > pct_extreme
        scales_low += pir_vals < (1.0 - pct_extreme)
        n_scales += 1

    if n_scales == 0:
        idx_range = close.index
        zeros = pd.Series(0.0, index=idx_range)
        return zeros.copy(), zeros.copy(), 0, zeros.copy(), zeros.copy()

    agree_high = pd.Series(scales_high / n_scales, index=close.index)
    agree_low = pd.Series(scales_low / n_scales, index=close.index)
    return (
        pd.Series(scales_high, index=close.index),
        pd.Series(scales_low, index=close.index),
        n_scales,
        agree_high,
        agree_low,
    )


# ---------------------------------------------------------------------------
# Pivot drift (per-bar scalar — used inside stateful detector loop)
# ---------------------------------------------------------------------------


def calc_pivot_drift(
    pivots: list[float], lookback: int
) -> Optional[float]:
    """Slope of the last *lookback* confirmed pivots."""
    min_pivots = max(lookback, 2)
    sz = len(pivots)
    if sz < min_pivots:
        return None
    start_val = pivots[sz - min_pivots]
    end_val = pivots[sz - 1]
    pivot_count = min_pivots - 1
    return ((end_val - start_val) / max(abs(start_val), 1e-9)) / pivot_count


# ---------------------------------------------------------------------------
# GJR-GARCH asymmetry
# ---------------------------------------------------------------------------

_GJR_ALPHA: float = 0.03
_GJR_BETA: float = 0.90
_GJR_GAMMA: float = 0.08


def calc_gjr_asym(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """GJR-GARCH asymmetry ratio and normalized score.

    Returns:
        (gjr_asym_norm, gjr_asym_ratio) — both as pd.Series.
    """
    close = df["close"].values.astype(np.float64)
    n = len(close)

    # Log returns — first bar uses close[0]/close[0] = 1 → log_ret = 0
    prev_close = np.empty(n, dtype=np.float64)
    prev_close[0] = close[0]
    prev_close[1:] = close[:-1]
    log_ret = np.log(close / prev_close)
    r2 = log_ret ** 2

    # Long-run variance (rolling 252-bar mean of r2)
    lr_var_series = pd.Series(r2).rolling(252).mean()
    lr_var = np.clip(lr_var_series.values, 1e-12, None)

    # Omega
    omega = np.clip(
        lr_var * (1.0 - _GJR_ALPHA - _GJR_BETA - _GJR_GAMMA / 2.0),
        1e-12,
        None,
    )

    # Stateful loop
    gjr_var = np.full(n, np.nan)
    sym_var = np.full(n, np.nan)

    # Seed with r2[0] (the squared return at bar 0); lr_var[0] is NaN because
    # it requires 252 bars.  Fall back to 1e-12 only if r2[0] is also NaN.
    init_val = r2[0] if not np.isnan(r2[0]) else 1e-12
    init_val = max(init_val, 1e-12)
    gjr_var[0] = init_val
    sym_var[0] = init_val

    for t in range(1, n):
        # omega[t] is NaN for bars 0–251 (before lr_var becomes valid).
        # Fall back to max(r2[t], 1e-12) to mirror the Pine Script behaviour.
        om = omega[t] if not np.isnan(omega[t]) else max(r2[t], 1e-12)
        om = max(om, 1e-12)
        leverage = 1.0 if log_ret[t - 1] < 0 else 0.0
        gjr_var[t] = max(
            om + (_GJR_ALPHA + _GJR_GAMMA * leverage) * r2[t - 1]
            + _GJR_BETA * gjr_var[t - 1],
            1e-12,
        )
        sym_var[t] = max(
            om + (_GJR_ALPHA + _GJR_GAMMA * 0.5) * r2[t - 1]
            + _GJR_BETA * sym_var[t - 1],
            1e-12,
        )

    gjr_asym_ratio = gjr_var / sym_var
    gjr_asym_norm = np.clip((gjr_asym_ratio - 1.0) / 0.1, -1.0, 1.0)

    idx = df.index
    return pd.Series(gjr_asym_norm, index=idx), pd.Series(gjr_asym_ratio, index=idx)


# ---------------------------------------------------------------------------
# HAR forecast (Garman-Klass)
# ---------------------------------------------------------------------------


def calc_har_vol(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """HAR volatility forecast ratio and normalized score.

    Returns:
        (har_vol_norm, har_vol_ratio) — both as pd.Series.
    """
    high = df["high"].values.astype(np.float64)
    low = df["low"].values.astype(np.float64)
    close_arr = df["close"].values.astype(np.float64)
    open_arr = df["open"].values.astype(np.float64)

    log_hl = np.clip(np.log(high / low), 1e-10, None)
    log_co = np.clip(np.log(close_arr / open_arr), -1e10, 1e10)

    gk_var = np.clip(
        0.5 * log_hl ** 2 - (2.0 * np.log(2.0) - 1.0) * log_co ** 2,
        1e-10,
        None,
    )

    gk_series = pd.Series(gk_var, index=df.index)
    gk_weekly = gk_series.rolling(5).mean().fillna(gk_series)
    gk_monthly = gk_series.rolling(22).mean().fillna(gk_series)

    har_forecast = np.clip(
        0.36 * gk_series.values + 0.28 * gk_weekly.values + 0.28 * gk_monthly.values,
        1e-10,
        None,
    )

    har_vol_ratio = np.clip(
        np.sqrt(har_forecast) / np.sqrt(gk_var), 1e-10, None
    )
    har_vol_norm = np.clip((har_vol_ratio - 1.0) / 0.5, -1.0, 1.0)

    idx = df.index
    return pd.Series(har_vol_norm, index=idx), pd.Series(har_vol_ratio, index=idx)


# ---------------------------------------------------------------------------
# SMA + PIR matrix precomputation
# ---------------------------------------------------------------------------


def precompute_matrices(
    close: pd.Series,
    scale_min: int = 2,
    scale_max: int = 500,
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Precompute SMA and PIR matrices for all integer scales.

    Args:
        close: Close price series.
        scale_min: Minimum SMA scale (inclusive).
        scale_max: Maximum SMA scale (inclusive).

    PINE-FAITHFUL (2026-06-10 TV-export audit): the Pine indicator's parity
    shim (``pir_for_scale``) computes SMA as a ``ta.cum`` cumulative-sum
    difference — first valid at bar ``s`` (it needs ``csum[t-s]``), one bar
    LATER than a pandas rolling mean — and scans pir min/max over the
    AVAILABLE (non-na) ratio bars, i.e. partial windows during warmup, with
    ``hi != lo ? (v-lo)/(hi-lo) : 0.5``. Both matrices are float64 because
    Pine compares float64 pir against the thresholds; a float32 matrix flips
    edge comparisons (pir ~ 0 vs extreme ``pct_extreme``).

    Returns:
        (sma_matrix, pir_matrix, scales_list)
        sma_matrix: float64, shape (n_scales, n_bars)
        pir_matrix: float64, shape (n_scales, n_bars)
        scales_list: list of integer scales [scale_min .. scale_max]
    """
    scales_list = list(range(scale_min, scale_max + 1))
    n_scales = len(scales_list)
    n_bars = len(close)
    close_vals = close.values.astype(np.float64)

    sma_matrix = np.full((n_scales, n_bars), np.nan, dtype=np.float64)
    pir_matrix = np.full((n_scales, n_bars), np.nan, dtype=np.float64)

    logger.info(
        "Precomputing SMA/PIR matrices: %d scales x %d bars ...",
        n_scales, n_bars,
    )

    # ta.cum(close): strictly sequential float64 running sum (np.cumsum is
    # sequential for accumulate — byte-identical to Pine's bar-by-bar cum).
    csum = np.cumsum(close_vals)

    for i, s in enumerate(scales_list):
        # SMA via cumsum difference: (cum[t] - cum[t-s]) / s, valid t >= s
        # (Pine's sma_at(s, 0) needs the csum value BEFORE the window).
        sma_vals = np.full(n_bars, np.nan, dtype=np.float64)
        if s < n_bars:
            sma_vals[s:] = (csum[s:] - csum[:-s]) / s
        sma_matrix[i] = sma_vals

        # Ratio: Pine's `sma_now > 0 ? close / sma_now : 1.0` (na stays na).
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(sma_vals > 0, close_vals / sma_vals, 1.0)
        ratio[np.isnan(sma_vals)] = np.nan

        # PIR over the last `lb` bars, PARTIAL windows allowed (Pine skips na
        # bars in its scan), hi == lo -> 0.5; na ratio -> na (counts as
        # neither extreme, same as Pine's 0.5 fallback).
        pir_lb = max(s, 20)
        r = pd.Series(ratio)
        lo = r.rolling(pir_lb, min_periods=1).min().values
        hi = r.rolling(pir_lb, min_periods=1).max().values
        with np.errstate(divide="ignore", invalid="ignore"):
            pir_vals = np.where(hi != lo, (ratio - lo) / (hi - lo), 0.5)
        pir_vals[np.isnan(ratio)] = np.nan
        pir_matrix[i] = pir_vals

    logger.info("Precomputation complete.")
    return sma_matrix, pir_matrix, scales_list


# ---------------------------------------------------------------------------
# Fast agreement from precomputed PIR matrix
# ---------------------------------------------------------------------------


def calc_agreement_fast(
    pir_matrix: np.ndarray,
    scales_list: list[int],
    scale_start: int,
    scale_end: int,
    scale_step: int,
    pct_extreme: float,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Slice precomputed PIR matrix and count extremes.

    Returns:
        (agreement_high, agreement_low, n_scales) — numpy arrays + int.
    """
    base = scales_list[0]
    scales_idx = [
        s - base
        for s in range(scale_start, scale_end + 1, scale_step)
        if base <= s <= scales_list[-1]
    ]
    if not scales_idx:
        n_bars = pir_matrix.shape[1]
        return np.zeros(n_bars), np.zeros(n_bars), 0

    sliced = pir_matrix[scales_idx, :]  # (n_selected, n_bars)
    n_scales = len(scales_idx)

    high_counts = (sliced > pct_extreme).sum(axis=0)
    low_counts = (sliced < (1.0 - pct_extreme)).sum(axis=0)

    return high_counts / n_scales, low_counts / n_scales, n_scales


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    from pathlib import Path

    logging.basicConfig(level=logging.INFO)

    data_path = Path(__file__).resolve().parent.parent / "data" / "raw" / "SPX_1D_18710201_20260318.csv"
    if not data_path.exists():
        print(f"Data file not found: {data_path}", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(data_path)
    close = df["close"]
    logger.info("Loaded %d bars from %s", len(df), data_path.name)

    # 1. Params default construction
    p = Params()
    assert p.S_detect_high == 12
    assert p.S_detect_low == 26
    assert p.baseline_lb == 20
    logger.info("Params default construction OK")

    # 2. pir_of
    pir_vals = pir_of(close, 20)
    # After warmup (first 19 bars are NaN), all values should be in [0, 1]
    valid = pir_vals.dropna()
    assert valid.min() >= 0.0, f"pir min = {valid.min()}"
    assert valid.max() <= 1.0, f"pir max = {valid.max()}"
    logger.info("pir_of OK: %d valid values in [0, 1]", len(valid))

    # 3. precompute_matrices (small range for speed)
    sma_mat, pir_mat, scales = precompute_matrices(close, 2, 50)
    assert sma_mat.shape == (49, len(close)), f"sma shape {sma_mat.shape}"
    assert pir_mat.shape == (49, len(close)), f"pir shape {pir_mat.shape}"
    assert len(scales) == 49
    assert sma_mat.dtype == np.float32
    assert pir_mat.dtype == np.float32
    logger.info("precompute_matrices OK: shape %s, dtype %s", sma_mat.shape, sma_mat.dtype)

    # 4. calc_agreement_fast
    ah, al, ns = calc_agreement_fast(pir_mat, scales, 3, 45, 3, 0.96)
    assert ns > 0, "No scales selected"
    logger.info("calc_agreement_fast OK: %d scales, agreement_high range [%.4f, %.4f]",
                ns, np.nanmin(ah), np.nanmax(ah))

    print("\nindicators.py self-test PASSED")
