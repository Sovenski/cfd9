"""Vote bounds audit — can each vote's searched threshold box EVER be selective?

Hypothesis: some votes are structural rubber stamps (or structurally dead)
because the search-space threshold bounds do not bracket the feature's real
distribution. For each vote's underlying CONTINUOUS feature (exact detector
formulas, parity-anchored in temp/vote_contribution.py), pooled over 16
streams (bars >= 300), report distribution quantiles and the vote pass-rate
at threshold = box-lo / box-mid / box-hi from src/search_space.py
(HIGH_SPACE for the HIGH side, LOW_SPACE for the LOW side).

Comparison geometry per vote (src/detector.py refs as in vote_contribution):
  trend    HIGH: min(slope_val, linreg_norm) > T      LOW: max(...) < -T
  volume   HIGH: vol_surge < 1/T                      LOW: vol_surge > T
  momentum sign test mom_diverge < 0 — NO searched threshold
  mom_vel  HIGH: mom_vel <= -T (Reversal)             LOW: mom_vel >= T
  vola     pir_of(atr(S_detect), range_len) > T       (both sides)
  gjr      HIGH: gjr_asym_norm <= -T                  LOW: >= T
  har      har_vol_norm >= T                          (both sides)
  drift    HIGH: exported pivot_drift < -T            LOW: > T
For every vote, box-lo is the LEAST restrictive edge and box-hi the MOST.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import logging

logging.disable(logging.INFO)

import numpy as np
import pandas as pd

from src.detector import SpeculatorDetector, build_detector_artifacts
from src.indicators import Params, atr, linreg_slope_step, nz, pir_of, sma
from src.pooled_validation import load_stream_frame
from src.search_space import HIGH_SPACE, LOW_SPACE
from src.universe import resolve_streams

from vote_contribution import HIGH_P, LOW_P  # same dir; winners v17gpu_20260611_131405

WARMUP = 300
QS = [1, 5, 25, 50, 75, 95, 99]
VOTES = ("trend", "volume", "momentum", "mom_vel", "vola", "gjr", "har", "drift")
BOUND_KEY = dict(trend="slope_thresh", volume="vol_surge_thresh",
                 mom_vel="momentum_velocity_thresh", vola="vola_high_pct",
                 gjr="gjr_vote_thresh", har="har_vote_thresh",
                 drift="pivot_drift_thresh")


def features_for_side(df: pd.DataFrame, p: Params, side: str, det: pd.DataFrame,
                      art) -> dict:
    """Continuous per-bar feature behind each vote (exact detector quantities)."""
    close, volume = df["close"], df["volume"]
    g = lambda k: getattr(p, f"{k}_{side}")
    S = g("S_detect")

    slope_delta = max(round(S / 4), 2)
    sma_d = sma(close, S)
    slope_val = (nz(sma_d - sma_d.shift(slope_delta), 0.0)
                 / (slope_delta * sma_d.clip(lower=1e-9)) * 1000)
    linreg_norm = linreg_slope_step(close, S) / sma_d.clip(lower=1e-9) * 1000
    # vote passes iff BOTH beat thresh -> binding quantity is min (HIGH) / max (LOW)
    if side == "high":
        trend_f = np.minimum(slope_val.values, linreg_norm.values)
    else:
        trend_f = np.maximum(slope_val.values, linreg_norm.values)

    vol_fast_len = max(round(S / 4), 2)
    vol_slow = sma(volume, S)
    vol_surge = (sma(volume, vol_fast_len) / vol_slow.where(vol_slow != 0)).values.astype(float)

    price_ret = (close - close.shift(S)) / close.shift(S).clip(lower=1e-9)
    vol_ret = (volume - volume.shift(S)) / volume.shift(S).clip(lower=1)
    mom_f = (price_ret * vol_ret).values.astype(float)

    mv_f = (price_ret - price_ret.shift(1)).values.astype(float)

    vola_f = pir_of(atr(df, S), g("vola_range_len")).values.astype(float)

    return dict(trend=trend_f, volume=vol_surge, momentum=mom_f, mom_vel=mv_f,
                vola=vola_f, gjr=art.gjr_asym_norm.values.astype(float),
                har=art.har_vol_norm.values.astype(float),
                drift=det[f"pivot_drift_{side}"].to_numpy(float))


def pass_rate(vote: str, side: str, f: np.ndarray, thr: float) -> float:
    """Vote pass-rate at threshold value thr (NaN compares False, as detector)."""
    with np.errstate(invalid="ignore"):
        if vote == "trend":
            ok = (f > thr) if side == "high" else (f < -thr)
        elif vote == "volume":
            ok = (f < 1.0 / thr) if side == "high" else (f > thr)
        elif vote == "momentum":
            ok = f < 0  # sign test, thr ignored
        elif vote == "mom_vel":
            ok = (f <= -abs(thr)) if side == "high" else (f >= abs(thr))
        elif vote == "vola":
            ok = f > thr
        elif vote == "gjr":
            ok = (f <= -thr) if side == "high" else (f >= thr)
        elif vote == "har":
            ok = f >= thr
        else:  # drift
            ok = (f < -thr) if side == "high" else (f > thr)
    return float(np.nanmean(np.where(np.isnan(f), False, ok)))


def verdict(p_lo: float, p_hi: float) -> str:
    """box-lo = least restrictive edge, box-hi = most restrictive edge."""
    if p_hi > 0.35:
        return "STRUCTURAL STAMP"
    if p_lo < 0.02:
        return "STRUCTURAL DEAD"
    lo, hi = sorted((p_lo, p_hi))
    return f"SELECTIVE-CAPABLE ({lo:.1%}..{hi:.1%})"


def main() -> None:
    streams = resolve_streams(["INDICES", "COMMODITIES", "FX", "WORLD_ETF"],
                              ["1D"], str(ROOT / "data" / "raw_v16"))
    sides = {"high": (Params(**HIGH_P), HIGH_SPACE), "low": (Params(**LOW_P), LOW_SPACE)}
    pooled = {s: {v: [] for v in VOTES} for s in sides}

    for st in streams:
        df = load_stream_frame(st.path)
        n = len(df)
        art = build_detector_artifacts(df)
        mask = np.arange(n) >= WARMUP
        for side, (p, _) in sides.items():
            det = SpeculatorDetector(df, p, artifacts=art, include_debug_columns=True).run()
            feats = features_for_side(df, p, side, det, art)
            for v in VOTES:
                pooled[side][v].append(feats[v][mask])
        print(f"{st.stream_id} done", flush=True)

    for side in ("high", "low"):
        _, space = sides[side]
        print(f"\n=== {side.upper()} side — feature distributions vs "
              f"{'HIGH_SPACE' if side == 'high' else 'LOW_SPACE'} threshold boxes "
              f"(pooled, bars>={WARMUP}) ===")
        qhdr = "".join(f"q{q:<8}" for q in QS)
        hdr = (f"{'vote':<9}{qhdr}{'box[lo,hi]':>18}"
               f"{'p@lo':>8}{'p@mid':>8}{'p@hi':>8}  verdict")
        print(hdr)
        print("-" * len(hdr))
        for v in VOTES:
            f = np.concatenate(pooled[side][v])
            qs = np.nanquantile(f, [q / 100 for q in QS])
            qstr = "".join(f"{x:<9.4g}" for x in qs)
            if v == "momentum":
                neg = float(np.nanmean(np.where(np.isnan(f), False, f < 0)))
                zer = float(np.nanmean(np.where(np.isnan(f), False, f == 0)))
                pos = float(np.nanmean(np.where(np.isnan(f), False, f > 0)))
                nan = float(np.isnan(f).mean())
                vd = ("STRUCTURAL STAMP" if neg > 0.35 else
                      "STRUCTURAL DEAD" if neg < 0.02 else
                      f"FIXED sign test ({neg:.1%})")
                print(f"{v:<9}{qstr}{'(sign test)':>18}"
                      f"{neg:>8.3f}{neg:>8.3f}{neg:>8.3f}  {vd}"
                      f"  [sign: {neg:.1%}<0, {zer:.1%}=0, {pos:.1%}>0, {nan:.1%}NaN]")
                continue
            lo, hi = space.float_bounds[BOUND_KEY[v]]
            mid = (lo + hi) / 2.0
            p_lo, p_mid, p_hi = (pass_rate(v, side, f, t) for t in (lo, mid, hi))
            print(f"{v:<9}{qstr}{f'[{lo:g},{hi:g}]':>18}"
                  f"{p_lo:>8.3f}{p_mid:>8.3f}{p_hi:>8.3f}  {verdict(p_lo, p_hi)}")
        print("  p@lo/mid/hi = vote pass-rate with threshold at box edge/midpoint;"
              " lo = least restrictive edge, hi = most restrictive")


if __name__ == "__main__":
    main()
