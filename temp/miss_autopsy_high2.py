"""HIGH residual attribution: split the 39.8% unexplained misses into
dur_flag-blocked vs drift_gate-blocked (formulas reconstructed exactly from
detector.py L509-546 using the EXPORTED agreement + pivot_drift columns)."""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import logging; logging.disable(logging.INFO)
import numpy as np, pandas as pd
from src.detector import SpeculatorDetector
from src.indicators import Params
from src.pooled_validation import load_stream_frame
from src.universe import resolve_streams

SWING_W, DROP_ATR, TOL = 20, 2.0, 2
run = json.loads(Path(r"C:\Users\kuben\Downloads\v17gpu_20260611_090553_v17gpu.json").read_text())
p = Params(**{k: v for k, v in run["sides"]["high"]["best_params"].items() if k != "baseline_lb"})

def wilder_atr(df, n=14):
    h,l,c = (df[k].to_numpy(float) for k in ("high","low","close"))
    tr = np.maximum(h[1:]-l[1:], np.maximum(np.abs(h[1:]-c[:-1]), np.abs(l[1:]-c[:-1])))
    a = pd.Series(np.concatenate([[h[0]-l[0]], tr])).ewm(alpha=1/n, adjust=False).mean().to_numpy()
    a[a==0]=np.nan; return a

def real_tops(df, atr):
    h,l = df["high"].to_numpy(float), df["low"].to_numpy(float); n=len(df)
    out = np.zeros(n, bool)
    for t in range(SWING_W, n-SWING_W):
        if h[t]==h[t-SWING_W:t+SWING_W+1].max() and \
           (h[t]-l[t+1:t+SWING_W+1].min())>=DROP_ATR*atr[t]: out[t]=True
    return out

rows = []
for st in resolve_streams(["INDICES","COMMODITIES","FX","WORLD_ETF"], ["1D"], "data/raw_v16"):
    df = load_stream_frame(st.path); atr = wilder_atr(df); n = len(df)
    det = SpeculatorDetector(df, p, include_debug_columns=True).run()
    sig = det["signal_high"].to_numpy().astype(bool)
    agr = det["agreement_high_side"].to_numpy(float)
    drift = det["pivot_drift_high"].to_numpy(float)
    # dur flag reconstruction (detector.py L539-546)
    dur_flag = np.zeros(n, bool); dur_at = 0; dur_miss = 0
    for t in range(n):
        if agr[t] > p.dur_extreme_pct_high: dur_at += 1; dur_miss = 0
        else:
            dur_miss += 1
            if dur_miss > 1: dur_at = 0
        dur_flag[t] = dur_at >= p.min_duration_high
    # drift gate blocks when TRUE (detector.py L509-510, L564)
    gate_blk = drift > (p.pivot_drift_thresh_high * p.pivot_drift_gate_mult_high)
    turn = real_tops(df, atr)
    for t in np.where(turn)[0]:
        if t < 300 or t >= n - SWING_W: continue
        w = slice(t-TOL, t+TOL+1)
        rows.append(dict(detected=bool(sig[w].any()),
                         blk_dur=bool((~dur_flag[w]).all()),
                         blk_drift=bool(gate_blk[w].all())))
d = pd.DataFrame(rows); miss = d[~d["detected"]]
print(f"HIGH real tops={len(d)}  missed={len(miss)}")
print(f"  dur_flag blocks   : {miss['blk_dur'].mean():.1%} of misses")
print(f"  drift_gate blocks : {miss['blk_drift'].mean():.1%} of misses")
print(f"  either            : {(miss['blk_dur']|miss['blk_drift']).mean():.1%}")
