"""v17 precompute scorer — a faithful, fast re-score of the pivot detector.

``FastDetector`` precomputes every per-bar array of ``SpeculatorDetector._detect``
that depends only on the FIXED *shape*/architecture params, then ``signals(params)``
applies the (varying) *threshold* comparisons + the identical stateful Phase-2
loop. This skips the expensive Phase-1 recompute (multi-scale agreement, rolling
SMA/ATR/linreg/GJR/HAR) on every evaluation during coordinate-ascent.

It reuses the detector's own helpers (``_edge_or_state``, ``_to_bool``, the gate
flags) and ``src.indicators`` so the output is byte-identical — verified by
``tests/test_v17_fastdetector.py``. Only fields in ``active_threshold_fields`` may
differ from the base; everything else must match (asserted). Final/holdout scoring
still uses the real ``SpeculatorDetector``.
"""

from __future__ import annotations

import dataclasses
import logging
from typing import Optional

import numpy as np
import pandas as pd

from .detector import (
    DetectorArtifacts,
    _USE_PIVOT_DRIFT,
    _USE_PRICE_GATE,
    _edge_or_state,
    _to_bool,
    build_detector_artifacts,
)
from .indicators import (
    Params,
    atr,
    calc_agreement_fast,
    calc_pivot_drift,
    linreg_slope_step,
    nz,
    pir_of,
    pivot_high,
    pivot_low,
    sma,
    stdev,
)
from .v17_optimize import active_threshold_fields

logger = logging.getLogger(__name__)


def _vola_pos(df: pd.DataFrame, close: pd.Series, calc_len: int, method: str,
              range_len: int) -> np.ndarray:
    """The pir position array behind the volatility vote (without the threshold)."""
    if method == "ATR":
        raw = atr(df, calc_len)
    elif method == "Intraday":
        raw = ((df["high"] - df["low"]) / df["close"].clip(lower=1e-9)).rolling(calc_len).mean()
    else:  # StdDev
        raw = stdev(close, calc_len)
    return pir_of(raw, range_len).values


class FastDetector:
    """Precompute shape-dependent arrays once; re-score thresholds cheaply."""

    def __init__(self, df: pd.DataFrame, base_params: Params,
                 artifacts: Optional[DetectorArtifacts] = None) -> None:
        self.df = df.reset_index(drop=True)
        self.base = base_params
        self.art = artifacts or build_detector_artifacts(self.df)
        self._varied = set(active_threshold_fields(base_params, "high")
                           + active_threshold_fields(base_params, "low"))
        self._agr_cache: dict = {}
        self._precompute()

    # ------------------------------------------------------------------
    def _precompute(self) -> None:
        p = self.base
        df = self.df
        close, high, low, volume = df["close"], df["high"], df["low"], df["volume"]
        n = len(df)
        self.n = n
        pir_matrix = self.art.pir_matrix
        scales_list = self.art.scales_list

        # --- scale-divergence base (pir_detect_*; agr added per-pct in cache) ---
        sma_detect_high = sma(close, p.S_detect_high)
        self._pir_detect_high = pir_of(close / sma_detect_high.clip(lower=1e-9),
                                       max(p.S_detect_high, 20))
        sma_detect_low = sma(close, p.S_detect_low)
        self._pir_detect_low = pir_of(close / sma_detect_low.clip(lower=1e-9),
                                      max(p.S_detect_low, 20))

        # --- trend (store raw slope arrays; threshold applied in signals) ---
        slope_delta_high = max(round(p.S_detect_high / 4), 2)
        self._slope_val_high = (nz(sma_detect_high - sma_detect_high.shift(slope_delta_high), 0.0)
                                / (slope_delta_high * sma_detect_high.clip(lower=1e-9)) * 1000)
        self._linreg_norm_high = (linreg_slope_step(close, p.S_detect_high)
                                  / sma_detect_high.clip(lower=1e-9) * 1000)
        slope_delta_low = max(round(p.S_detect_low / 4), 2)
        self._slope_val_low = (nz(sma_detect_low - sma_detect_low.shift(slope_delta_low), 0.0)
                               / (slope_delta_low * sma_detect_low.clip(lower=1e-9)) * 1000)
        self._linreg_norm_low = (linreg_slope_step(close, p.S_detect_low)
                                 / sma_detect_low.clip(lower=1e-9) * 1000)

        # --- volume surge (raw ratio; threshold applied in signals) ---
        vfh = max(round(p.S_detect_high / 4), 2)
        vsh = sma(volume, p.S_detect_high)
        self._vol_surge_high = (sma(volume, vfh) / vsh.where(vsh != 0)).values.astype(float)
        vfl = max(round(p.S_detect_low / 4), 2)
        vsl = sma(volume, p.S_detect_low)
        self._vol_surge_low = (sma(volume, vfl) / vsl.where(vsl != 0)).values.astype(float)

        # --- momentum divergence vote (mom_div<0 — no threshold => edge now) ---
        pr_h = (close - close.shift(p.S_detect_high)) / close.shift(p.S_detect_high).clip(lower=1e-9)
        vr_h = (volume - volume.shift(p.S_detect_high)) / volume.shift(p.S_detect_high).clip(lower=1)
        mom_div_h = (pr_h * vr_h)
        pr_l = (close - close.shift(p.S_detect_low)) / close.shift(p.S_detect_low).clip(lower=1e-9)
        vr_l = (volume - volume.shift(p.S_detect_low)) / volume.shift(p.S_detect_low).clip(lower=1)
        mom_div_l = (pr_l * vr_l)
        self._eff_md_h = _edge_or_state(_to_bool((mom_div_h < 0).values),
                                        p.edge_window_high, p.use_edge_voting_high)
        self._eff_md_l = _edge_or_state(_to_bool((mom_div_l < 0).values),
                                        p.edge_window_low, p.use_edge_voting_low)

        # --- momentum velocity (raw; threshold+mode applied in signals) ---
        self._mom_vel_high = (pr_h - pr_h.shift(1)).values.astype(float)
        self._mom_vel_low = (pr_l - pr_l.shift(1)).values.astype(float)

        # --- volatility position (raw pir; threshold applied in signals) ---
        self._vola_pos_high = _vola_pos(df, close, p.S_detect_high, p.vola_method_high,
                                        p.vola_range_len_high)
        self._vola_pos_low = _vola_pos(df, close, p.S_detect_low, p.vola_method_low,
                                       p.vola_range_len_low)

        # --- GJR / HAR raw norms (threshold applied in signals) ---
        self._gjr_norm = self.art.gjr_asym_norm.values.astype(float)
        self._har_norm = self.art.har_vol_norm.values.astype(float)

        # --- ER gate (shape only: period/directional/use_er_gate) ---
        self._er_gate_ok_high = _to_bool(self._er_gate(close, p.er_period_high,
                                         p.er_directional_high, p.use_er_gate_high).values)
        self._er_gate_ok_low = _to_bool(self._er_gate(close, p.er_period_low,
                                        p.er_directional_low, p.use_er_gate_low).values)

        # --- price gate (shape: lookback) ---
        self._price_high_ok = _to_bool((high >= high.rolling(p.price_gate_lb_high).max().shift(1)).values)
        self._price_low_ok = _to_bool((low <= low.rolling(p.price_gate_lb_low).min().shift(1)).values)

        # --- baseline pivot series (shape: baseline_lb) ---
        self._ph_arr = pivot_high(high, p.baseline_lb).values
        self._pl_arr = pivot_low(low, p.baseline_lb).values
        self._high_arr = high.values.astype(float)
        self._low_arr = low.values.astype(float)

        # --- max votes (architecture, fixed) ---
        self._max_votes_high = int(sum([p.use_trend_high, p.use_volume_high, p.use_momentum_high,
                                        p.use_momentum_velocity_high, p.use_volatility_high,
                                        p.use_gjr_asym_high, p.use_har_vol_high]))
        self._max_votes_low = int(sum([p.use_trend_low, p.use_volume_low, p.use_momentum_low,
                                       p.use_momentum_velocity_low, p.use_volatility_low,
                                       p.use_gjr_asym_low, p.use_har_vol_low]))

    @staticmethod
    def _er_gate(close: pd.Series, period: int, directional: bool, use_gate: bool) -> pd.Series:
        path = close.diff().abs().rolling(period).sum()
        net = (close - close.shift(period)) if directional else (close - close.shift(period)).abs()
        val = net / path.clip(lower=1e-10)
        if not use_gate:
            return pd.Series(True, index=close.index)
        return (val < val.shift(1)).fillna(True)

    def _agr(self, side: str, pct: float):
        """(agr_array, scale_div_raw_array) for one side at a given pct_extreme — memoized."""
        key = (side, round(float(pct), 12))
        if key in self._agr_cache:
            return self._agr_cache[key]
        p = self.base
        if side == "high":
            agr_h, _agr_l, _ = calc_agreement_fast(
                self.art.pir_matrix, self.art.scales_list,
                p.scale_start_high, p.scale_end_high, p.scale_step_high, pct)
            agr = agr_h
            scale_div_raw = (self._pir_detect_high - pd.Series(agr_h, index=self.df.index)).values
        else:
            _agr_h, agr_l, _ = calc_agreement_fast(
                self.art.pir_matrix, self.art.scales_list,
                p.scale_start_low, p.scale_end_low, p.scale_step_low, pct)
            agr = agr_l
            scale_div_raw = ((1.0 - self._pir_detect_low)
                             - pd.Series(agr_l, index=self.df.index)).values
        self._agr_cache[key] = (np.asarray(agr, dtype=float), np.asarray(scale_div_raw, dtype=float))
        return self._agr_cache[key]

    # ------------------------------------------------------------------
    def signals(self, params: Params) -> dict:
        """Re-score with `params` (must share the base's shape/architecture)."""
        p = params
        # Guard: only threshold fields may differ from the base.
        for f in vars(self.base):
            if f in self._varied:
                continue
            if getattr(p, f) != getattr(self.base, f):
                raise ValueError(f"FastDetector: non-threshold field {f!r} changed; "
                                 "rebuild FastDetector for a new shape/architecture.")

        agr_h_high, scale_div_high_raw = self._agr("high", p.pct_extreme_high)
        agr_l_low, scale_div_low_raw = self._agr("low", p.pct_extreme_low)

        ew_h, uev_h = p.edge_window_high, p.use_edge_voting_high
        ew_l, uev_l = p.edge_window_low, p.use_edge_voting_low

        # --- threshold comparisons -> edge-effective vote arrays (mirror detector) ---
        ms_agree_high = _to_bool(agr_h_high >= p.min_agreement_high)
        ms_agree_low = _to_bool(agr_l_low >= p.min_agreement_low)
        scale_div_high = _to_bool(np.abs(scale_div_high_raw) > p.scale_div_thresh_high)
        scale_div_low = _to_bool(np.abs(scale_div_low_raw) > p.scale_div_thresh_low)

        trend_up_h = _to_bool((self._slope_val_high > p.slope_thresh_high).values
                              & (self._linreg_norm_high > p.slope_thresh_high).values)
        trend_dn_l = _to_bool((self._slope_val_low < -p.slope_thresh_low).values
                              & (self._linreg_norm_low < -p.slope_thresh_low).values)

        vol_slow_h = (self._vol_surge_high < (1.0 / p.vol_surge_thresh_high))
        vol_surge_l = (self._vol_surge_low > p.vol_surge_thresh_low)

        if p.momentum_velocity_mode_high == "Reversal":
            mv_h_ok = self._mom_vel_high <= -abs(p.momentum_velocity_thresh_high)
        else:
            mv_h_ok = self._mom_vel_high >= abs(p.momentum_velocity_thresh_high)
        if p.momentum_velocity_mode_low == "Reversal":
            mv_l_ok = self._mom_vel_low >= abs(p.momentum_velocity_thresh_low)
        else:
            mv_l_ok = self._mom_vel_low <= -abs(p.momentum_velocity_thresh_low)

        vola_h = self._vola_pos_high > p.vola_high_pct_high
        vola_l = self._vola_pos_low > p.vola_high_pct_low
        gjr_h = self._gjr_norm <= -p.gjr_vote_thresh_high
        gjr_l = self._gjr_norm >= p.gjr_vote_thresh_low
        har_h = self._har_norm >= p.har_vote_thresh_high
        har_l = self._har_norm >= p.har_vote_thresh_low

        eff_t_up_h = _edge_or_state(trend_up_h, ew_h, uev_h)
        eff_vs_h = _edge_or_state(_to_bool(vol_slow_h), ew_h, uev_h)
        eff_mv_h = _edge_or_state(_to_bool(mv_h_ok), ew_h, uev_h)
        eff_va_h = _edge_or_state(_to_bool(vola_h), ew_h, uev_h)
        eff_g_h = _edge_or_state(_to_bool(gjr_h), ew_h, uev_h)
        eff_h_h = _edge_or_state(_to_bool(har_h), ew_h, uev_h)
        eff_md_h = self._eff_md_h
        eff_t_dn_l = _edge_or_state(trend_dn_l, ew_l, uev_l)
        eff_vs_l = _edge_or_state(_to_bool(vol_surge_l), ew_l, uev_l)
        eff_mv_l = _edge_or_state(_to_bool(mv_l_ok), ew_l, uev_l)
        eff_va_l = _edge_or_state(_to_bool(vola_l), ew_l, uev_l)
        eff_g_l = _edge_or_state(_to_bool(gjr_l), ew_l, uev_l)
        eff_h_l = _edge_or_state(_to_bool(har_l), ew_l, uev_l)
        eff_md_l = self._eff_md_l

        ms_agree_high_arr, ms_agree_low_arr = ms_agree_high, ms_agree_low
        scale_div_high_arr, scale_div_low_arr = scale_div_high, scale_div_low
        er_h, er_l = self._er_gate_ok_high, self._er_gate_ok_low
        price_h, price_l = self._price_high_ok, self._price_low_ok
        ph_arr, pl_arr = self._ph_arr, self._pl_arr
        high_arr, low_arr = self._high_arr, self._low_arr
        mvh, mvl = max(self._max_votes_high, 1), max(self._max_votes_low, 1)
        n = self.n

        # --- Phase 2: stateful loop (verbatim from SpeculatorDetector._detect) ---
        dur_at_high = dur_miss_high = dur_at_low = dur_miss_low = 0
        bars_since_high = bars_since_low = 999
        confirmed_pivots: list[float] = []
        out_h = np.zeros(n, dtype=bool)
        out_l = np.zeros(n, dtype=bool)
        state_pd_dn_h = np.zeros(n, dtype=bool)
        state_pd_up_l = np.zeros(n, dtype=bool)

        for t in range(n):
            bars_since_high += 1
            bars_since_low += 1

            current_baseline_ph = np.nan
            current_baseline_pl = np.nan
            pivot_bar = t - p.baseline_lb
            if pivot_bar >= 0:
                if ph_arr[pivot_bar]:
                    current_baseline_ph = high_arr[pivot_bar]
                if pl_arr[pivot_bar]:
                    current_baseline_pl = low_arr[pivot_bar]

            drift_high = calc_pivot_drift(confirmed_pivots, p.pivot_drift_lookback_high)
            drift_high = drift_high if drift_high is not None else 0.0
            drift_low = calc_pivot_drift(confirmed_pivots, p.pivot_drift_lookback_low)
            drift_low = drift_low if drift_low is not None else 0.0

            drift_down_high = drift_high < -p.pivot_drift_thresh_high
            drift_up_high = drift_high > p.pivot_drift_thresh_high
            drift_gate_up_high = drift_high > (p.pivot_drift_thresh_high * p.pivot_drift_gate_mult_high)
            drift_up_low = drift_low > p.pivot_drift_thresh_low

            state_pd_dn_h[t] = drift_down_high
            state_pd_up_l[t] = drift_up_low
            if (not uev_h) or (t < ew_h):
                eff_pd_dn_h_t = drift_down_high
            else:
                eff_pd_dn_h_t = drift_down_high and not bool(state_pd_dn_h[t - ew_h])
            if (not uev_l) or (t < ew_l):
                eff_pd_up_l_t = drift_up_low
            else:
                eff_pd_up_l_t = drift_up_low and not bool(state_pd_up_l[t - ew_l])

            if not np.isnan(current_baseline_ph):
                confirmed_pivots.append(current_baseline_ph)
            if not np.isnan(current_baseline_pl):
                confirmed_pivots.append(current_baseline_pl)

            if agr_h_high[t] > p.dur_extreme_pct_high:
                dur_at_high += 1; dur_miss_high = 0
            else:
                dur_miss_high += 1
                if dur_miss_high > 1:
                    dur_at_high = 0
            dur_high_flag = dur_at_high >= p.min_duration_high

            if agr_l_low[t] > p.dur_extreme_pct_low:
                dur_at_low += 1; dur_miss_low = 0
            else:
                dur_miss_low += 1
                if dur_miss_low > 1:
                    dur_at_low = 0
            dur_low_flag = dur_at_low >= p.min_duration_low

            gate_h = (
                (not _USE_PRICE_GATE or bool(price_h[t]))
                and bool(ms_agree_high_arr[t])
                and dur_high_flag
                and not bool(scale_div_high_arr[t])
                and bool(er_h[t])
                and not (_USE_PIVOT_DRIFT and drift_gate_up_high)
            )
            gate_l = (
                (not _USE_PRICE_GATE or bool(price_l[t]))
                and bool(ms_agree_low_arr[t])
                and dur_low_flag
                and not bool(scale_div_low_arr[t])
                and bool(er_l[t])
            )

            ph_c = int(sum([
                p.use_trend_high and bool(eff_t_up_h[t]),
                _USE_PIVOT_DRIFT and eff_pd_dn_h_t,
                p.use_volume_high and bool(eff_vs_h[t]),
                p.use_momentum_high and bool(eff_md_h[t]),
                p.use_momentum_velocity_high and bool(eff_mv_h[t]),
                p.use_volatility_high and bool(eff_va_h[t]),
                p.use_gjr_asym_high and bool(eff_g_h[t]),
                p.use_har_vol_high and bool(eff_h_h[t]),
            ]))
            pl_c = int(sum([
                p.use_trend_low and bool(eff_t_dn_l[t]),
                _USE_PIVOT_DRIFT and eff_pd_up_l_t,
                p.use_volume_low and bool(eff_vs_l[t]),
                p.use_momentum_low and bool(eff_md_l[t]),
                p.use_momentum_velocity_low and bool(eff_mv_l[t]),
                p.use_volatility_low and bool(eff_va_l[t]),
                p.use_gjr_asym_low and bool(eff_g_l[t]),
                p.use_har_vol_low and bool(eff_h_l[t]),
            ]))

            high_req = max(1, min(mvh, p.confirm_count_high + int(
                _USE_PIVOT_DRIFT and drift_up_high and p.pivot_drift_confirm_bias_high)))
            low_req = max(1, min(mvl, p.confirm_count_low - int(
                _USE_PIVOT_DRIFT and drift_up_low and p.pivot_drift_confirm_bias_low)))

            sig_h = gate_h and ph_c >= high_req and bars_since_high > p.cooldown_bars_high
            sig_l = gate_l and pl_c >= low_req and bars_since_low > p.cooldown_bars_low
            if sig_h:
                bars_since_high = 0
            if sig_l:
                bars_since_low = 0
            out_h[t] = sig_h
            out_l[t] = sig_l

        return {"signal_high": out_h, "signal_low": out_l}


class FastPooledScorer:
    """Drop-in for ``PooledScorer`` that scores via ``FastDetector`` (precomputed
    once per slice) instead of re-running the detector each eval. Returns the
    identical pooled block-bootstrap LCB (verified in test_v17_fastscorer)."""

    def __init__(self, folds, streams, side: str, base_params: Params,
                 n_boot: int = 1000, alpha: float = 0.10, block_len: int = 2) -> None:
        from .pooled_validation import _fold_is_informative, cluster_weights
        from .scoring import REFERENCE_N
        if side not in ("high", "low"):
            raise ValueError(f"side must be 'high'|'low', got {side!r}")
        self.side = side
        self.base = base_params
        self.n_boot, self.alpha, self.block_len = n_boot, alpha, block_len
        self._weights = cluster_weights(streams)
        self._sig_key = "signal_high" if side == "high" else "signal_low"
        self._is_informative = _fold_is_informative
        lbl = 1 if side == "high" else -1
        col = f"pivot_N{REFERENCE_N}"
        # Keep only label-informative folds; precompute a FastDetector per slice.
        self._fast: list = []
        for fold in folds:
            if sum(int((sl.df_oos[col] == lbl).sum()) for sl in fold) <= 0:
                continue
            entries = [(FastDetector(sl.df_is, base_params, sl.artifacts_is),
                        FastDetector(sl.df_oos, base_params, sl.artifacts_oos), sl)
                       for sl in fold]
            self._fast.append(entries)
        logger.info("FastPooledScorer[%s]: %d informative folds precomputed", side, len(self._fast))

    def score(self, params: Params) -> float:
        from .pooled_scoring import pooled_fold_score
        from .pooled_validation import _stream_stat
        from .validation import fold_scores_bootstrap_ci
        key = self._sig_key
        fold_scores: list[float] = []
        for entries in self._fast:
            is_stats, oos_stats = [], []
            for fd_is, fd_oos, sl in entries:
                w = self._weights.get(sl.stream.stream_id, 1.0)
                sig_is = pd.Series(fd_is.signals(params)[key], index=sl.df_is.index)
                sig_oos = pd.Series(fd_oos.signals(params)[key], index=sl.df_oos.index)
                is_stats.append(_stream_stat(sl.df_is, sig_is, self.side, w))
                oos_stats.append(_stream_stat(sl.df_oos, sig_oos, self.side, w))
            s, comp = pooled_fold_score(is_stats, oos_stats, self.side)
            if self._is_informative(comp):
                fold_scores.append(s)
        if not fold_scores:
            return 0.0
        return float(fold_scores_bootstrap_ci(
            fold_scores, n_boot=self.n_boot, alpha=self.alpha, block_len=self.block_len)[0])


__all__ = ["FastDetector", "FastPooledScorer"]
