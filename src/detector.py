"""Speculatores Pivot Optimizer — pivot signal detector.

Translates the Speculatores V12 Pine Script pivot detector to Python.
Phase 1 vectorizes all stateless indicators; Phase 2 runs a stateful
bar-by-bar loop for duration counters, pivot drift, baseline pivots,
and cooldown logic.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from .indicators import (
    Params,
    atr,
    calc_agreement_fast,
    calc_gjr_asym,
    calc_har_vol,
    calc_pivot_drift,
    linreg_value,
    nz,
    pir_of,
    pivot_high,
    pivot_low,
    precompute_matrices,
    sma,
    stdev,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level constants (NOT in Params — fixed True)
# ---------------------------------------------------------------------------

_USE_PRICE_GATE: bool = True
_USE_PIVOT_DRIFT: bool = True


@dataclass(frozen=True)
class DetectorArtifacts:
    """Fold-invariant detector precomputations reused across trials."""

    pir_matrix: np.ndarray
    scales_list: list[int]
    gjr_asym_norm: pd.Series
    har_vol_norm: pd.Series


def build_detector_artifacts(df: pd.DataFrame) -> DetectorArtifacts:
    """Precompute expensive detector inputs that do not depend on Params."""
    close = df["close"]
    _, pir_matrix, scales_list = precompute_matrices(close, 2, 500)
    gjr_asym_norm, _ = calc_gjr_asym(df)
    har_vol_norm, _ = calc_har_vol(df)
    return DetectorArtifacts(
        pir_matrix=pir_matrix,
        scales_list=scales_list,
        gjr_asym_norm=gjr_asym_norm,
        har_vol_norm=har_vol_norm,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _to_bool(arr: np.ndarray) -> np.ndarray:
    """Convert an array that may contain NaN to a clean boolean array."""
    result = np.zeros(len(arr), dtype=bool)
    float_arr = arr.astype(float)
    mask = ~np.isnan(float_arr)
    result[mask] = float_arr[mask].astype(bool)
    return result


# ---------------------------------------------------------------------------
# SpeculatorDetector
# ---------------------------------------------------------------------------


class SpeculatorDetector:
    """Detects pivot signals using the Speculatores multi-scale framework.

    Args:
        df: OHLCV DataFrame with columns: open, high, low, close, volume.
        params: Frozen Params dataclass with all detector parameters.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        params: Params,
        artifacts: Optional[DetectorArtifacts] = None,
        include_debug_columns: bool = False,
    ) -> None:
        self._df = df.reset_index(drop=True)
        self._params = params
        self._artifacts = artifacts
        self._include_debug_columns = include_debug_columns
        self._result: Optional[pd.DataFrame] = None

    def run(self) -> pd.DataFrame:
        """Run detection. Returns result DataFrame. Caches result."""
        if self._result is not None:
            return self._result
        self._result = self._detect()
        return self._result

    def _detect(self) -> pd.DataFrame:
        """Execute Phase 1 (vectorized) then Phase 2 (stateful loop)."""
        p = self._params
        df = self._df
        close = df["close"]
        high = df["high"]
        low = df["low"]
        volume = df["volume"]
        n = len(df)

        logger.info("Starting detection on %d bars ...", n)

        # ---------------------------------------------------------------
        # Phase 1: Vectorized precomputation
        # ---------------------------------------------------------------

        # --- SMA/PIR matrices (covers all scales used by both sides) ---
        artifacts = self._artifacts
        if artifacts is None:
            artifacts = build_detector_artifacts(df)
        pir_matrix = artifacts.pir_matrix
        scales_list = artifacts.scales_list

        # --- Per-side agreement fractions (numpy arrays, shape (n,)) ---
        agr_h_high, agr_l_high, _ = calc_agreement_fast(
            pir_matrix, scales_list,
            p.scale_start_high, p.scale_end_high, p.scale_step_high,
            p.pct_extreme_high,
        )
        agr_h_low, agr_l_low, _ = calc_agreement_fast(
            pir_matrix, scales_list,
            p.scale_start_low, p.scale_end_low, p.scale_step_low,
            p.pct_extreme_low,
        )

        # --- Scale divergence ---
        sma_detect_high = sma(close, p.S_detect_high)
        pir_detect_high = pir_of(
            close / sma_detect_high.clip(lower=1e-9),
            max(p.S_detect_high, 20),
        )
        scale_div_high = pir_detect_high - pd.Series(
            agr_h_high, index=close.index
        )
        scale_div_high_flag = scale_div_high.abs() > p.scale_div_thresh_high

        sma_detect_low = sma(close, p.S_detect_low)
        pir_detect_low = pir_of(
            close / sma_detect_low.clip(lower=1e-9),
            max(p.S_detect_low, 20),
        )
        scale_div_low = (1.0 - pir_detect_low) - pd.Series(
            agr_l_low, index=close.index
        )
        scale_div_low_flag = scale_div_low.abs() > p.scale_div_thresh_low

        # --- Trend — HIGH side ---
        slope_delta_high = max(round(p.S_detect_high / 4), 2)
        sma_val_high = sma_detect_high
        slope_val_high = (
            nz(sma_val_high - sma_val_high.shift(slope_delta_high), 0.0)
            / (slope_delta_high * sma_val_high.clip(lower=1e-9))
            * 1000
        )
        linreg_val_high = linreg_value(close, p.S_detect_high)
        linreg_norm_high = (
            linreg_val_high.diff() / sma_val_high.clip(lower=1e-9) * 1000
        )
        trend_up_high = (slope_val_high > p.slope_thresh_high) & (
            linreg_norm_high > p.slope_thresh_high
        )

        # --- Trend — LOW side ---
        slope_delta_low = max(round(p.S_detect_low / 4), 2)
        sma_val_low = sma_detect_low
        slope_val_low = (
            nz(sma_val_low - sma_val_low.shift(slope_delta_low), 0.0)
            / (slope_delta_low * sma_val_low.clip(lower=1e-9))
            * 1000
        )
        linreg_val_low = linreg_value(close, p.S_detect_low)
        linreg_norm_low = (
            linreg_val_low.diff() / sma_val_low.clip(lower=1e-9) * 1000
        )
        trend_down_low = (slope_val_low < -p.slope_thresh_low) & (
            linreg_norm_low < -p.slope_thresh_low
        )

        # --- Volume surge ---
        vol_fast_len_high = max(round(p.S_detect_high / 4), 2)
        vol_surge_high = (
            sma(volume, vol_fast_len_high)
            / sma(volume, p.S_detect_high).clip(lower=1e-10)
        )
        vol_fast_len_low = max(round(p.S_detect_low / 4), 2)
        vol_surge_low = (
            sma(volume, vol_fast_len_low)
            / sma(volume, p.S_detect_low).clip(lower=1e-10)
        )

        # --- Momentum ---
        price_ret_high = (
            (close - close.shift(p.S_detect_high))
            / close.shift(p.S_detect_high).clip(lower=1e-9)
        )
        vol_ret_high = (
            (volume - volume.shift(p.S_detect_high))
            / volume.shift(p.S_detect_high).clip(lower=1)
        )
        mom_diverge_high = price_ret_high * vol_ret_high

        price_ret_low = (
            (close - close.shift(p.S_detect_low))
            / close.shift(p.S_detect_low).clip(lower=1e-9)
        )
        vol_ret_low = (
            (volume - volume.shift(p.S_detect_low))
            / volume.shift(p.S_detect_low).clip(lower=1)
        )
        mom_diverge_low = price_ret_low * vol_ret_low

        # --- Momentum velocity ---
        mom_vel_high = price_ret_high - price_ret_high.shift(1)
        thresh_h = abs(p.momentum_velocity_thresh_high)
        if p.momentum_velocity_mode_high == "Reversal":
            mom_vel_high_ok = mom_vel_high <= -thresh_h
        else:  # Trend
            mom_vel_high_ok = mom_vel_high >= thresh_h

        mom_vel_low = price_ret_low - price_ret_low.shift(1)
        thresh_l = abs(p.momentum_velocity_thresh_low)
        if p.momentum_velocity_mode_low == "Reversal":
            mom_vel_low_ok = mom_vel_low >= thresh_l
        else:  # Trend
            mom_vel_low_ok = mom_vel_low <= -thresh_l

        # --- Volatility ---
        vola_elevated_high = self._calc_vola_elevated(
            df, close, p.S_detect_high,
            p.vola_method_high, p.vola_range_len_high, p.vola_high_pct_high,
        )
        vola_elevated_low = self._calc_vola_elevated(
            df, close, p.S_detect_low,
            p.vola_method_low, p.vola_range_len_low, p.vola_high_pct_low,
        )

        # --- ER (Efficiency Ratio) ---
        er_path_high = close.diff().abs().rolling(p.er_period_high).sum()
        if p.er_directional_high:
            er_net_high = close - close.shift(p.er_period_high)
        else:
            er_net_high = (close - close.shift(p.er_period_high)).abs()
        er_val_high = er_net_high / er_path_high.clip(lower=1e-10)
        if not p.use_er_gate_high:
            er_gate_ok_high = pd.Series(True, index=close.index)
        else:
            er_gate_ok_high = (
                er_val_high < er_val_high.shift(1)
            ).fillna(True)

        er_path_low = close.diff().abs().rolling(p.er_period_low).sum()
        if p.er_directional_low:
            er_net_low = close - close.shift(p.er_period_low)
        else:
            er_net_low = (close - close.shift(p.er_period_low)).abs()
        er_val_low = er_net_low / er_path_low.clip(lower=1e-10)
        if not p.use_er_gate_low:
            er_gate_ok_low = pd.Series(True, index=close.index)
        else:
            er_gate_ok_low = (
                er_val_low < er_val_low.shift(1)
            ).fillna(True)

        # --- GJR + HAR votes ---
        gjr_asym_norm = artifacts.gjr_asym_norm
        gjr_high_vote = gjr_asym_norm <= -p.gjr_vote_thresh_high
        gjr_low_vote = gjr_asym_norm >= p.gjr_vote_thresh_low

        har_vol_norm = artifacts.har_vol_norm
        har_high_vote = har_vol_norm >= p.har_vote_thresh_high
        har_low_vote = har_vol_norm >= p.har_vote_thresh_low

        # --- Price gate ---
        price_high_ok = high >= (
            high.rolling(p.price_gate_lb_high)
            .max()
            .shift(1)
            .fillna(-np.inf)
        )
        price_low_ok = low <= (
            low.rolling(p.price_gate_lb_low)
            .min()
            .shift(1)
            .fillna(np.inf)
        )

        # --- Baseline pivot series (non-causal, used with offset) ---
        ph_series = pivot_high(high, p.baseline_lb)
        pl_series = pivot_low(low, p.baseline_lb)

        # ---------------------------------------------------------------
        # Convert all to numpy arrays before loop (performance)
        # ---------------------------------------------------------------
        ms_agree_high_arr = _to_bool(agr_h_high >= p.min_agreement_high)
        ms_agree_low_arr = _to_bool(agr_l_low >= p.min_agreement_low)
        scale_div_high_arr = _to_bool(scale_div_high_flag.values)
        scale_div_low_arr = _to_bool(scale_div_low_flag.values)
        trend_up_high_arr = _to_bool(trend_up_high.values)
        trend_down_low_arr = _to_bool(trend_down_low.values)
        vol_surge_high_arr = vol_surge_high.values.astype(float)
        vol_surge_low_arr = vol_surge_low.values.astype(float)
        mom_diverge_high_arr = mom_diverge_high.values.astype(float)
        mom_diverge_low_arr = mom_diverge_low.values.astype(float)
        mom_vel_high_ok_arr = _to_bool(mom_vel_high_ok.values)
        mom_vel_low_ok_arr = _to_bool(mom_vel_low_ok.values)
        vola_elevated_high_arr = _to_bool(vola_elevated_high)
        vola_elevated_low_arr = _to_bool(vola_elevated_low)
        er_gate_ok_high_arr = _to_bool(er_gate_ok_high.values)
        er_gate_ok_low_arr = _to_bool(er_gate_ok_low.values)
        gjr_high_vote_arr = _to_bool(gjr_high_vote.values)
        gjr_low_vote_arr = _to_bool(gjr_low_vote.values)
        har_high_vote_arr = _to_bool(har_high_vote.values)
        har_low_vote_arr = _to_bool(har_low_vote.values)
        price_high_ok_arr = _to_bool(price_high_ok.values)
        price_low_ok_arr = _to_bool(price_low_ok.values)
        ph_arr = ph_series.values
        pl_arr = pl_series.values
        high_arr = high.values.astype(float)
        low_arr = low.values.astype(float)

        # ---------------------------------------------------------------
        # Phase 2: Stateful bar-by-bar loop
        # ---------------------------------------------------------------

        # Precompute max_votes (constant per Params)
        max_votes_high = int(sum([
            p.use_trend_high, p.use_volume_high, p.use_momentum_high,
            p.use_momentum_velocity_high, p.use_volatility_high,
            p.use_gjr_asym_high, p.use_har_vol_high,
        ]))
        max_votes_low = int(sum([
            p.use_trend_low, p.use_volume_low, p.use_momentum_low,
            p.use_momentum_velocity_low, p.use_volatility_low,
            p.use_gjr_asym_low, p.use_har_vol_low,
        ]))

        # State
        dur_at_high = 0
        dur_miss_high = 0
        dur_at_low = 0
        dur_miss_low = 0
        bars_since_high = 999
        bars_since_low = 999
        confirmed_pivots: list[float] = []

        # Output arrays
        out_signal_high = np.zeros(n, dtype=bool)
        out_signal_low = np.zeros(n, dtype=bool)
        out_baseline_ph = np.full(n, np.nan)
        out_baseline_pl = np.full(n, np.nan)
        out_pivot_drift_high = np.zeros(n, dtype=float)
        out_pivot_drift_low = np.zeros(n, dtype=float)
        out_ph_confirms = np.zeros(n, dtype=float)
        out_pl_confirms = np.zeros(n, dtype=float)

        for t in range(n):
            bars_since_high += 1
            bars_since_low += 1

            # --- Baseline pivots (causal: bar t-baseline_lb is now known) ---
            pivot_bar = t - p.baseline_lb
            if pivot_bar >= p.baseline_lb:  # t >= 2*baseline_lb
                if ph_arr[pivot_bar]:
                    out_baseline_ph[t] = high_arr[pivot_bar]
                    confirmed_pivots.append(high_arr[pivot_bar])
                if pl_arr[pivot_bar]:
                    out_baseline_pl[t] = low_arr[pivot_bar]
                    confirmed_pivots.append(low_arr[pivot_bar])

            # --- Pivot drift ---
            drift_high = calc_pivot_drift(
                confirmed_pivots, p.pivot_drift_lookback_high
            )
            drift_high = drift_high if drift_high is not None else 0.0
            drift_low = calc_pivot_drift(
                confirmed_pivots, p.pivot_drift_lookback_low
            )
            drift_low = drift_low if drift_low is not None else 0.0

            drift_down_high = drift_high < -p.pivot_drift_thresh_high
            drift_up_high = drift_high > p.pivot_drift_thresh_high
            drift_gate_up_high = drift_high > (
                p.pivot_drift_thresh_high * p.pivot_drift_gate_mult_high
            )
            drift_up_low = drift_low > p.pivot_drift_thresh_low
            out_pivot_drift_high[t] = drift_high
            out_pivot_drift_low[t] = drift_low

            # --- Duration at extreme ---
            if agr_h_high[t] > p.dur_extreme_pct_high:
                dur_at_high += 1
                dur_miss_high = 0
            else:
                dur_miss_high += 1
                if dur_miss_high > 1:
                    dur_at_high = 0
            dur_high_flag = dur_at_high >= p.min_duration_high

            if agr_l_low[t] > p.dur_extreme_pct_low:
                dur_at_low += 1
                dur_miss_low = 0
            else:
                dur_miss_low += 1
                if dur_miss_low > 1:
                    dur_at_low = 0
            dur_low_flag = dur_at_low >= p.min_duration_low

            # --- Gate pass ---
            gate_h = (
                (not _USE_PRICE_GATE or bool(price_high_ok_arr[t]))
                and bool(ms_agree_high_arr[t])
                and dur_high_flag
                and not bool(scale_div_high_arr[t])
                and bool(er_gate_ok_high_arr[t])
                and not (_USE_PIVOT_DRIFT and drift_gate_up_high)
            )
            gate_l = (
                (not _USE_PRICE_GATE or bool(price_low_ok_arr[t]))
                and bool(ms_agree_low_arr[t])
                and dur_low_flag
                and not bool(scale_div_low_arr[t])
                and bool(er_gate_ok_low_arr[t])
            )

            # --- Vote counts ---
            ph_c = int(sum([
                p.use_trend_high and bool(trend_up_high_arr[t]),
                _USE_PIVOT_DRIFT and drift_down_high,
                p.use_volume_high and (
                    vol_surge_high_arr[t] < (1.0 / p.vol_surge_thresh_high)
                ),
                p.use_momentum_high and (mom_diverge_high_arr[t] < 0),
                p.use_momentum_velocity_high and bool(
                    mom_vel_high_ok_arr[t]
                ),
                p.use_volatility_high and bool(vola_elevated_high_arr[t]),
                p.use_gjr_asym_high and bool(gjr_high_vote_arr[t]),
                p.use_har_vol_high and bool(har_high_vote_arr[t]),
            ]))
            pl_c = int(sum([
                p.use_trend_low and bool(trend_down_low_arr[t]),
                _USE_PIVOT_DRIFT and drift_up_low,
                p.use_volume_low and (
                    vol_surge_low_arr[t] > p.vol_surge_thresh_low
                ),
                p.use_momentum_low and (mom_diverge_low_arr[t] < 0),
                p.use_momentum_velocity_low and bool(
                    mom_vel_low_ok_arr[t]
                ),
                p.use_volatility_low and bool(vola_elevated_low_arr[t]),
                p.use_gjr_asym_low and bool(gjr_low_vote_arr[t]),
                p.use_har_vol_low and bool(har_low_vote_arr[t]),
            ]))
            out_ph_confirms[t] = ph_c
            out_pl_confirms[t] = pl_c

            # --- Required votes (clamped) ---
            mv_h = max(max_votes_high, 1)
            high_req = max(1, min(
                mv_h,
                p.confirm_count_high + int(
                    _USE_PIVOT_DRIFT
                    and drift_up_high
                    and p.pivot_drift_confirm_bias_high
                ),
            ))
            mv_l = max(max_votes_low, 1)
            low_req = max(1, min(
                mv_l,
                p.confirm_count_low - int(
                    _USE_PIVOT_DRIFT
                    and drift_up_low
                    and p.pivot_drift_confirm_bias_low
                ),
            ))

            # --- Signal generation ---
            sig_h = (
                gate_h
                and ph_c >= high_req
                and bars_since_high > p.cooldown_bars_high
            )
            sig_l = (
                gate_l
                and pl_c >= low_req
                and bars_since_low > p.cooldown_bars_low
            )

            if sig_h:
                bars_since_high = 0
            if sig_l:
                bars_since_low = 0

            out_signal_high[t] = sig_h
            out_signal_low[t] = sig_l

        logger.info(
            "Detection complete: %d high signals, %d low signals",
            out_signal_high.sum(),
            out_signal_low.sum(),
        )

        result = {
            "signal_high": out_signal_high,
            "signal_low": out_signal_low,
            "baseline_ph": out_baseline_ph,
            "baseline_pl": out_baseline_pl,
        }
        if self._include_debug_columns:
            result.update(
                {
                    "agreement_high_side": agr_h_high.astype(float),
                    "agreement_low_side": agr_l_low.astype(float),
                    "momentum_velocity_high": mom_vel_high.values.astype(float),
                    "momentum_velocity_low": mom_vel_low.values.astype(float),
                    "pivot_drift_high": out_pivot_drift_high,
                    "pivot_drift_low": out_pivot_drift_low,
                    "ph_confirms": out_ph_confirms,
                    "pl_confirms": out_pl_confirms,
                    "price_gate_high": price_high_ok_arr.astype(float),
                    "price_gate_low": price_low_ok_arr.astype(float),
                    "er_val_high": er_val_high.values.astype(float),
                    "er_val_low": er_val_low.values.astype(float),
                    "baseline_pivot_high": out_baseline_ph,
                    "baseline_pivot_low": out_baseline_pl,
                }
            )

        return pd.DataFrame(result, index=df.index)

    # -------------------------------------------------------------------
    # Helper methods
    # -------------------------------------------------------------------

    @staticmethod
    def _calc_vola_elevated(
        df: pd.DataFrame,
        close: pd.Series,
        calc_len: int,
        method: str,
        range_len: int,
        threshold: float,
    ) -> np.ndarray:
        """Compute volatility elevated flag for one side."""
        if method == "ATR":
            raw = atr(df, calc_len)
        elif method == "Intraday":
            raw = (
                (df["high"] - df["low"]) / df["close"].clip(lower=1e-9)
            ).rolling(calc_len).mean()
        else:  # StdDev
            raw = stdev(close, calc_len)
        pos = pir_of(raw, range_len)
        return (pos > threshold).values


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    from pathlib import Path

    logging.basicConfig(level=logging.INFO)

    data_path = (
        Path(__file__).resolve().parent.parent
        / "data"
        / "raw"
        / "SPX_1D_18710201_20260318.csv"
    )
    if not data_path.exists():
        print(f"Data file not found: {data_path}", file=sys.stderr)
        sys.exit(1)

    raw = pd.read_csv(data_path)
    # Normalize column names to lowercase
    raw.columns = raw.columns.str.lower()
    logger.info("Loaded %d bars from %s", len(raw), data_path.name)

    params = Params()
    result = SpeculatorDetector(raw, params).run()

    n_high = result["signal_high"].sum()
    n_low = result["signal_low"].sum()
    n_total = len(result)

    logger.info("n_high_signals = %d", n_high)
    logger.info("n_low_signals  = %d", n_low)
    logger.info("total bars     = %d", n_total)

    # Gold preset tuned for Gold 1D — HIGH side may produce 0 signals on SPX
    assert n_low > 0, f"Expected LOW signals, got 0"
    assert n_low < len(result) * 0.05, f"Too many LOW signals: {n_low}"
    logger.info("HIGH signals: %d (may be 0 — Gold preset tuned for Gold, not SPX)", n_high)

    print("\ndetector.py self-test PASSED")
