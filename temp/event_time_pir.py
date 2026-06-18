"""T3 — event-time PIR vs bar-time PIR as hit/spam separators on real fires.

pir_evt[t] = position of close within [min,max] over bars since the EVT_K-th
most recent CONFIRMED structural pivot (pivot_high_pine/low_pine n=20,
confirmed 20 bars later). Window expires on structure, not calendar.
"""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import logging; logging.disable(logging.INFO)
import numpy as np, pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from src.detector import SpeculatorDetector
from src.indicators import Params, pir_of, pivot_high_pine, pivot_low_pine, sma
from src.pooled_validation import load_stream_frame
from src.universe import resolve_streams

SWING_W, DROP_ATR, TOL, EVT_K = 20, 2.0, 2, 8
run = json.loads(Path(r"C:\Users\kuben\Downloads\v17gpu_20260611_090553_v17gpu.json").read_text())
P = {s: Params(**{k: v for k, v in run["sides"][s]["best_params"].items() if k != "baseline_lb"})
     for s in ("high", "low")}

def wilder_atr(df, n=14):
    h,l,c = (df[k].to_numpy(float) for k in ("high","low","close"))
    tr = np.maximum(h[1:]-l[1:], np.maximum(np.abs(h[1:]-c[:-1]), np.abs(l[1:]-c[:-1])))
    a = pd.Series(np.concatenate([[h[0]-l[0]], tr])).ewm(alpha=1/n, adjust=False).mean().to_numpy()
    a[a==0]=np.nan; return a

def real_turns(df, kind, atr):
    h,l = df["high"].to_numpy(float), df["low"].to_numpy(float); n=len(df)
    out = np.zeros(n, bool)
    for t in range(SWING_W, n-SWING_W):
        if kind=="top" and h[t]==h[t-SWING_W:t+SWING_W+1].max() and \
           (h[t]-l[t+1:t+SWING_W+1].min())>=DROP_ATR*atr[t]: out[t]=True
        elif kind=="bot" and l[t]==l[t-SWING_W:t+SWING_W+1].min() and \
           (h[t+1:t+SWING_W+1].max()-l[t])>=DROP_ATR*atr[t]: out[t]=True
    return out

def event_time_pir(df) -> np.ndarray:
    """Anchor = bar index of the EVT_K-th most recent confirmed pivot (any side)."""
    n = len(df); c = df["close"].to_numpy(float)
    ph, pl = pivot_high_pine(df["high"], 20), pivot_low_pine(df["low"], 20)
    conf = sorted([(i+20, i) for i in np.where(ph.notna())[0]] +
                  [(i+20, i) for i in np.where(pl.notna())[0]])  # (confirm_bar, pivot_bar)
    out = np.full(n, np.nan); anchors = []
    j = 0
    for t in range(n):
        while j < len(conf) and conf[j][0] <= t:
            anchors.append(conf[j][1]); j += 1
        if len(anchors) >= EVT_K:
            a = anchors[-EVT_K]
            w = c[a:t+1]; lo, hi = w.min(), w.max()
            out[t] = (c[t]-lo)/(hi-lo) if hi != lo else 0.5
    return out

streams = resolve_streams(["INDICES","COMMODITIES","FX","WORLD_ETF"], ["1D"], "data/raw_v16")
rows = {"high": [], "low": []}
for st in streams:
    df = load_stream_frame(st.path); atr = wilder_atr(df); n = len(df)
    evt = event_time_pir(df)
    bars = {S: pir_of(df["close"]/sma(df["close"], S).clip(lower=1e-9), max(S,20)).to_numpy()
            for S in (26, 50)}
    for side, sigcol, kind in (("high","signal_high","top"), ("low","signal_low","bot")):
        det = SpeculatorDetector(df, P[side]).run()
        fired = np.where(det[sigcol].to_numpy())[0]
        fired = fired[(fired >= 300) & (fired < n - SWING_W)]
        turn = real_turns(df, kind, atr)
        for t in fired:
            rows[side].append(dict(stream=st.stream_id,
                hit=bool(turn[max(0,t-TOL):t+TOL+1].any()),
                pir_evt=float(evt[t]), pir26=float(bars[26][t]), pir50=float(bars[50][t])))
    print(f"{st.stream_id} done", flush=True)

def loso_auc(d, col):
    X = np.nan_to_num(d[[col]].to_numpy(float)); y = d["hit"].to_numpy(int)
    pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced"))
    pr = cross_val_predict(pipe, X, y, cv=GroupKFold(5), groups=d["stream"], method="predict_proba")[:,1]
    u,_ = stats.mannwhitneyu(pr[y==1], pr[y==0], alternative="two-sided")
    return u/((y==1).sum()*(y==0).sum())

for side in ("high","low"):
    d = pd.DataFrame(rows[side])
    a_evt = loso_auc(d, "pir_evt"); a26 = loso_auc(d, "pir26"); a50 = loso_auc(d, "pir50")
    best_bar = max(a26, a50)
    print(f"{side.upper()}: AUC evt={a_evt:.3f}  bar26={a26:.3f}  bar50={a50:.3f}  "
          f"delta={a_evt-best_bar:+.3f}")
print("T3 verdict: PASS iff delta>=+0.03 both sides, or >=+0.05 one side")
