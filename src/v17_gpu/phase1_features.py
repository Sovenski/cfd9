"""Shape-only Phase-1 feature precompute (build-spec §5.3 — CPU mirror part).

Verbatim mirror of ``FastDetector._precompute`` using the oracle's OWN
pandas/numpy helpers, so every array is byte-identical by construction:

- SMA/ATR/stdev rolling means use pandas' Kahan-compensated kernels and
  ``linreg_slope_step`` uses ``np.polyfit`` (P8 default (a), generalized) —
  a vectorized torch reduction is NOT bitwise-equal at float64. These arrays
  depend only on the FROZEN shape params and are computed ONCE per slice,
  off the hot scoring path.
- GJR/HAR come from the per-slice ``DetectorArtifacts`` (P2 slice-local
  seeding); the momentum-divergence product is kept RAW (v18 P2.3 — its
  vote threshold ``momentum_diverge_thresh`` is per-candidate, applied in
  the torch threshold layer together with the P6 edge).
- P1 dtypes: ``pir_matrix`` float32; all other float features float64.

The torch threshold layer (``eval_torch.TorchPhase1``) uploads these as
tensors and applies the per-candidate comparisons.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.detector import DetectorArtifacts, _to_bool
from src.indicators import (
    Params,
    atr,
    linreg_slope_step,
    nz,
    pir_of,
    pivot_high_pine,
    pivot_low_pine,
    sma,
    stdev,
)

logger = logging.getLogger(__name__)

__all__ = ["Phase1Features", "compute_phase1_features"]


@dataclass(frozen=True)
class Phase1Features:
    """Every Phase-1 array of ``FastDetector._precompute`` for ONE slice.

    P1 dtypes: ``pir_matrix`` float32; all other float features float64;
    gates/edge votes bool.
    """

    pir_matrix: np.ndarray
    scales_list: list[int]
    pir_detect_high: np.ndarray
    pir_detect_low: np.ndarray
    slope_val_high: np.ndarray
    slope_val_low: np.ndarray
    linreg_norm_high: np.ndarray
    linreg_norm_low: np.ndarray
    vol_surge_high: np.ndarray
    vol_surge_low: np.ndarray
    mom_div_high: np.ndarray
    mom_div_low: np.ndarray
    mom_vel_high: np.ndarray
    mom_vel_low: np.ndarray
    vola_pos_high: np.ndarray
    vola_pos_low: np.ndarray
    gjr_norm: np.ndarray
    har_norm: np.ndarray
    er_gate_ok_high: np.ndarray
    er_gate_ok_low: np.ndarray
    price_high_ok: np.ndarray
    price_low_ok: np.ndarray
    ph_arr: np.ndarray
    pl_arr: np.ndarray
    high_arr: np.ndarray
    low_arr: np.ndarray
    max_votes_high: int
    max_votes_low: int
    n: int


def _vola_pos(df: pd.DataFrame, close: pd.Series, calc_len: int, method: str,
              range_len: int) -> np.ndarray:
    """Verbatim ``v17_fastdetector._vola_pos`` (pir position behind the vote)."""
    if method == "ATR":
        raw = atr(df, calc_len)
    elif method == "Intraday":
        raw = ((df["high"] - df["low"]) / df["close"].clip(lower=1e-9)).rolling(calc_len).mean()
    else:  # StdDev
        raw = stdev(close, calc_len)
    return pir_of(raw, range_len).values


def _er_gate(close: pd.Series, period: int, directional: bool, use_gate: bool) -> pd.Series:
    """Verbatim ``FastDetector._er_gate``."""
    path = close.diff().abs().rolling(period).sum()
    net = (close - close.shift(period)) if directional else (close - close.shift(period)).abs()
    val = net / path.clip(lower=1e-10)
    if not use_gate:
        return pd.Series(True, index=close.index)
    return (val < val.shift(1)).fillna(True)


def compute_phase1_features(
    df: pd.DataFrame, p: Params, art: DetectorArtifacts, pir_matrix: np.ndarray
) -> Phase1Features:
    """Verbatim mirror of ``FastDetector._precompute`` for one slice (P2).

    Args:
        df: slice OHLCV frame (already ``reset_index(drop=True)``).
        p: FROZEN base/shape params.
        art: per-slice artifacts (GJR/HAR/PIR — P2 slice-local seeding).
        pir_matrix: float32 PIR matrix (oracle's or ``build_pir_matrix_torch``).
    """
    close, high, low, volume = df["close"], df["high"], df["low"], df["volume"]

    sma_detect_high = sma(close, p.S_detect_high)
    pir_detect_high = pir_of(close / sma_detect_high.clip(lower=1e-9),
                             max(p.S_detect_high, 20))
    sma_detect_low = sma(close, p.S_detect_low)
    pir_detect_low = pir_of(close / sma_detect_low.clip(lower=1e-9),
                            max(p.S_detect_low, 20))

    slope_delta_high = max(round(p.S_detect_high / 4), 2)
    slope_val_high = (nz(sma_detect_high - sma_detect_high.shift(slope_delta_high), 0.0)
                      / (slope_delta_high * sma_detect_high.clip(lower=1e-9)) * 1000)
    linreg_norm_high = (linreg_slope_step(close, p.S_detect_high)        # P8 (a)
                        / sma_detect_high.clip(lower=1e-9) * 1000)
    slope_delta_low = max(round(p.S_detect_low / 4), 2)
    slope_val_low = (nz(sma_detect_low - sma_detect_low.shift(slope_delta_low), 0.0)
                     / (slope_delta_low * sma_detect_low.clip(lower=1e-9)) * 1000)
    linreg_norm_low = (linreg_slope_step(close, p.S_detect_low)          # P8 (a)
                       / sma_detect_low.clip(lower=1e-9) * 1000)

    vfh = max(round(p.S_detect_high / 4), 2)
    vsh = sma(volume, p.S_detect_high)
    vol_surge_high = (sma(volume, vfh) / vsh.where(vsh != 0)).values.astype(float)
    vfl = max(round(p.S_detect_low / 4), 2)
    vsl = sma(volume, p.S_detect_low)
    vol_surge_low = (sma(volume, vfl) / vsl.where(vsl != 0)).values.astype(float)

    # v18 P2.3: raw momentum-divergence product; the vote threshold
    # (`mom_div < -momentum_diverge_thresh`) is per-candidate, so the
    # comparison + edge happen in the torch threshold layer (votes_t).
    pr_h = (close - close.shift(p.S_detect_high)) / close.shift(p.S_detect_high).clip(lower=1e-9)
    vr_h = (volume - volume.shift(p.S_detect_high)) / volume.shift(p.S_detect_high).clip(lower=1)
    mom_div_h = (pr_h * vr_h)
    pr_l = (close - close.shift(p.S_detect_low)) / close.shift(p.S_detect_low).clip(lower=1e-9)
    vr_l = (volume - volume.shift(p.S_detect_low)) / volume.shift(p.S_detect_low).clip(lower=1)
    mom_div_l = (pr_l * vr_l)

    mom_vel_high = (pr_h - pr_h.shift(1)).values.astype(float)
    mom_vel_low = (pr_l - pr_l.shift(1)).values.astype(float)

    return Phase1Features(
        pir_matrix=pir_matrix,
        scales_list=list(art.scales_list),
        pir_detect_high=pir_detect_high.values,
        pir_detect_low=pir_detect_low.values,
        slope_val_high=slope_val_high.values,
        slope_val_low=slope_val_low.values,
        linreg_norm_high=linreg_norm_high.values,
        linreg_norm_low=linreg_norm_low.values,
        vol_surge_high=vol_surge_high,
        vol_surge_low=vol_surge_low,
        mom_div_high=mom_div_h.values.astype(float),
        mom_div_low=mom_div_l.values.astype(float),
        mom_vel_high=mom_vel_high,
        mom_vel_low=mom_vel_low,
        vola_pos_high=_vola_pos(df, close, p.S_detect_high, p.vola_method_high,
                                p.vola_range_len_high),
        vola_pos_low=_vola_pos(df, close, p.S_detect_low, p.vola_method_low,
                               p.vola_range_len_low),
        gjr_norm=art.gjr_asym_norm.values.astype(float),                 # P2
        har_norm=art.har_vol_norm.values.astype(float),                  # P2
        er_gate_ok_high=_to_bool(_er_gate(close, p.er_period_high,
                                          p.er_directional_high, p.use_er_gate_high).values),
        er_gate_ok_low=_to_bool(_er_gate(close, p.er_period_low,
                                         p.er_directional_low, p.use_er_gate_low).values),
        price_high_ok=_to_bool((high >= high.rolling(p.price_gate_lb_high).max().shift(1)).values),
        price_low_ok=_to_bool((low <= low.rolling(p.price_gate_lb_low).min().shift(1)).values),
        ph_arr=pivot_high_pine(high, p.baseline_lb).values,
        pl_arr=pivot_low_pine(low, p.baseline_lb).values,
        high_arr=high.values.astype(float),
        low_arr=low.values.astype(float),
        max_votes_high=int(sum([p.use_trend_high, p.use_volume_high, p.use_momentum_high,
                                p.use_momentum_velocity_high, p.use_volatility_high,
                                p.use_gjr_asym_high, p.use_har_vol_high,
                                p.count_drift_vote_high])),       # v18 P2.4
        max_votes_low=int(sum([p.use_trend_low, p.use_volume_low, p.use_momentum_low,
                               p.use_momentum_velocity_low, p.use_volatility_low,
                               p.use_gjr_asym_low, p.use_har_vol_low,
                               p.count_drift_vote_low])),         # v18 P2.4
        n=len(df),
    )
