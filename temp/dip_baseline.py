"""D1/D2 — LOW detector vs naive dip-buyer null (pre-registered addendum,
plan/habituation-wedge-tests.md).

Naive rule: fire when low <= rolling N-bar min of low, N calibrated PER STREAM
so naive fire count matches the LOW preset's fire count (+/-10%), cooldown 7.
"""
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import logging

logging.disable(logging.INFO)
import numpy as np
import pandas as pd

from src.detector import SpeculatorDetector
from src.indicators import Params
from src.pooled_validation import load_stream_frame
from src.universe import resolve_streams

SWING_W, DROP_ATR, TOL, COOLDOWN = 20, 2.0, 2, 7
run = json.loads(Path(r"C:\Users\kuben\Downloads\v17gpu_20260611_090553_v17gpu.json").read_text())
p_low = Params(**{k: v for k, v in run["sides"]["low"]["best_params"].items() if k != "baseline_lb"})


def wilder_atr(df, n=14):
    h, l, c = (df[k].to_numpy(float) for k in ("high", "low", "close"))
    tr = np.maximum(h[1:] - l[1:], np.maximum(np.abs(h[1:] - c[:-1]), np.abs(l[1:] - c[:-1])))
    a = pd.Series(np.concatenate([[h[0] - l[0]], tr])).ewm(alpha=1 / n, adjust=False).mean().to_numpy()
    a[a == 0] = np.nan
    return a


def real_bottoms(df, atr):
    h, l = df["high"].to_numpy(float), df["low"].to_numpy(float)
    n = len(df)
    out = np.zeros(n, bool)
    for t in range(SWING_W, n - SWING_W):
        if l[t] == l[t - SWING_W:t + SWING_W + 1].min() and \
           (h[t + 1:t + SWING_W + 1].max() - l[t]) >= DROP_ATR * atr[t]:
            out[t] = True
    return out


def naive_fires(df, N, lo_bound, hi_bound):
    """low touches rolling N-bar min, cooldown applied; only bars in [lo,hi)."""
    low = df["low"].to_numpy(float)
    rmin = pd.Series(low).rolling(N, min_periods=N).min().to_numpy()
    raw = low <= rmin + 1e-12
    fires = []
    last = -10**9
    for t in range(lo_bound, hi_bound):
        if raw[t] and (t - last) > COOLDOWN:
            fires.append(t)
            last = t
    return np.array(fires, dtype=int)


streams = resolve_streams(["INDICES", "COMMODITIES", "FX", "WORLD_ETF"], ["1D"], "data/raw_v16")
rows = []
for st in streams:
    df = load_stream_frame(st.path)
    atr = wilder_atr(df)
    n = len(df)
    logc = np.log(df["close"].to_numpy(float))
    lo_b, hi_b = 300, n - SWING_W
    det = SpeculatorDetector(df, p_low).run()
    det_fires = np.where(det["signal_low"].to_numpy())[0]
    det_fires = det_fires[(det_fires >= lo_b) & (det_fires < hi_b)]
    target = len(det_fires)
    if target < 5:
        continue
    # calibrate N (5..200) to match fire count within +/-10%, else closest
    best_N, best_gap, best_f = None, 10**9, None
    for N in range(5, 201, 5):
        f = naive_fires(df, N, lo_b, hi_b)
        gap = abs(len(f) - target)
        if gap < best_gap:
            best_N, best_gap, best_f = N, gap, f
        if gap <= 0.10 * target:
            break
    turn = real_bottoms(df, atr)
    fwd20 = np.full(n, np.nan); fwd40 = np.full(n, np.nan)
    fwd20[:-20] = logc[20:] - logc[:-20]
    fwd40[:-40] = logc[40:] - logc[:-40]

    def add(kind, idxs):
        for t in idxs:
            rows.append(dict(stream=st.stream_id, kind=kind,
                             hit=bool(turn[max(0, t - TOL):t + TOL + 1].any()),
                             fwd20=float(fwd20[t]) if not np.isnan(fwd20[t]) else np.nan,
                             fwd40=float(fwd40[t]) if not np.isnan(fwd40[t]) else np.nan))
    add("det", det_fires)
    add("naive", best_f)
    print(f"{st.stream_id}: det={target} naive={len(best_f)} (N={best_N})", flush=True)

d = pd.DataFrame(rows)
det, nai = d[d["kind"] == "det"], d[d["kind"] == "naive"]
p_det, p_nai = det["hit"].mean(), nai["hit"].mean()
print(f"\nPooled fires: detector={len(det)}  naive={len(nai)}")
print(f"D1 location: detector precision {p_det:.3f} vs naive {p_nai:.3f} "
      f"(diff {p_det - p_nai:+.3f})")
rng = np.random.default_rng(0)
sids = d["stream"].unique()
boots = []
for _ in range(2000):
    pick = rng.choice(sids, len(sids), replace=True)
    bb = pd.concat([d[d["stream"] == s] for s in pick])
    a = bb[bb["kind"] == "det"]["hit"]; b = bb[bb["kind"] == "naive"]["hit"]
    if len(a) and len(b):
        boots.append(a.mean() - b.mean())
boots = np.asarray(boots)
pval = float((boots <= 0).mean())
d1 = (p_det - p_nai) >= 0.03 and pval < 0.05
print(f"cluster bootstrap p={pval:.3f} -> D1 {'PASS' if d1 else 'FAIL'}")
print(f"D2 payoff: fwd20 det {det['fwd20'].median():+.4f} naive {nai['fwd20'].median():+.4f}"
      f" | fwd40 det {det['fwd40'].median():+.4f} naive {nai['fwd40'].median():+.4f}")
