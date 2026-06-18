"""Measurement-validity diagnostic for T1/T2 of the pooled 090553 fires.

NOT a re-test: prints distributions of sync and graded frame intensity
(frame_frac) at fire bars. No verdicts, no threshold changes.
"""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import logging; logging.disable(logging.INFO)
import numpy as np, pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from src.detector import SpeculatorDetector
from src.indicators import Params, precompute_matrices
from src.pooled_validation import load_stream_frame
from src.universe import resolve_streams

SWING_W, DROP_ATR, TOL, FRAME_X, SYNC_K = 20, 2.0, 2, 5, 3
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

def expiry_events(x: np.ndarray, W: int, mode: str) -> np.ndarray:
    """True at t where the rolling-W extreme holder rotated WITHOUT renewal."""
    n = len(x); ev = np.zeros(n, bool)
    if n < W + 1: return ev
    xx = np.where(np.isnan(x), -np.inf if mode == "max" else np.inf, x)  # NaN guard
    sw = sliding_window_view(xx, W)                    # rows end at t = W-1 .. n-1
    arg = (np.argmax(sw,1) if mode=="max" else np.argmin(sw,1))
    holder = arg + np.arange(n - W + 1)                # absolute holder index
    t_idx = np.arange(W, n)                            # compare t vs t-1 (full windows)
    h_now, h_prev = holder[1:], holder[:-1]
    ev[t_idx] = (h_now != h_prev) & (h_now != t_idx)   # changed, not set by current bar
    valid = np.flatnonzero(~np.isnan(x))               # NaNs occur only at the head; mask warmup
    ev[:min(n, (valid[0] if len(valid) else n) + W)] = False
    return ev

def frame_and_sync(df, p: Params, side: str):
    """Same as frame_attribution.frame_and_sync, plus graded frame_frac
    (fraction of scale windows with an expiry event within the last FRAME_X bars)."""
    close = df["close"]; n = len(df)
    pirM, _, scales = precompute_matrices(close)
    g = lambda k: getattr(p, f"{k}_{side}")
    sl = [s for s in range(g("scale_start"), g("scale_end")+1, g("scale_step"))
          if scales[0] <= s <= scales[-1]]
    idx = [s - scales[0] for s in sl]
    pct = g("pct_extreme")
    csum = np.cumsum(close.to_numpy(np.float64))
    frame_any = np.zeros(n, bool); ext = np.zeros((len(sl), n), bool); fresh = np.zeros((len(sl), n), bool)
    recent = np.zeros((len(sl), n), bool)              # per-scale: expiry event within last FRAME_X bars
    for j, s in enumerate(sl):
        sma = np.full(n, np.nan); sma[s:] = (csum[s:]-csum[:-s])/s
        ratio = np.where(sma>0, close.to_numpy(np.float64)/sma, 1.0); ratio[np.isnan(sma)] = np.nan
        W = max(s, 20)
        ev = expiry_events(ratio, W, "max" if side=="high" else "min")
        recent[j] = pd.Series(ev).rolling(FRAME_X, min_periods=1).max().to_numpy().astype(bool)
        frame_any |= recent[j]
        pir = pirM[idx[j]]
        e = (pir > pct) if side=="high" else (pir < 1.0-pct)
        ext[j] = np.nan_to_num(e)
        crossed = ext[j] & ~np.roll(ext[j], 1); crossed[0] = ext[j][0]
        fresh[j] = pd.Series(crossed).rolling(SYNC_K, min_periods=1).max().to_numpy().astype(bool) & ext[j]
    # price-gate window counts as one extra reference frame (binary mask only, as in main script)
    px = (df["high"] if side=="high" else df["low"]).to_numpy(float)
    evg = expiry_events(px, g("price_gate_lb"), "max" if side=="high" else "min")
    frame_any |= pd.Series(evg).rolling(FRAME_X, min_periods=1).max().to_numpy().astype(bool)
    n_ext = ext.sum(0); n_fresh = fresh.sum(0)
    sync = np.where(n_ext > 0, n_fresh / np.maximum(n_ext, 1), 0.0)
    frame_frac = recent.sum(0) / max(len(sl), 1)       # scales only, not the price gate
    return frame_any, sync, frame_frac

streams = resolve_streams(["INDICES","COMMODITIES","FX","WORLD_ETF"], ["1D"], "data/raw_v16")
rows = {"high": [], "low": []}
for st in streams:
    df = load_stream_frame(st.path); atr = wilder_atr(df); n = len(df)
    for side, sigcol, kind in (("high","signal_high","top"), ("low","signal_low","bot")):
        det = SpeculatorDetector(df, P[side]).run()
        fired = np.where(det[sigcol].to_numpy())[0]
        fired = fired[(fired >= 300) & (fired < n - SWING_W)]
        if not len(fired): continue
        turn = real_turns(df, kind, atr)
        frame_any, sync, frame_frac = frame_and_sync(df, P[side], side)
        for t in fired:
            rows[side].append(dict(stream=st.stream_id, t=int(t),
                hit=bool(turn[max(0,t-TOL):t+TOL+1].any()),
                frame=bool(frame_any[t]), sync=float(sync[t]),
                frame_frac=float(frame_frac[t])))
    print(f"{st.stream_id} done", flush=True)

rng = np.random.default_rng(0)
for side in ("high","low"):
    d = pd.DataFrame(rows[side]); y = d["hit"].to_numpy(int)
    print(f"\n=== {side.upper()}  fires={len(d)}  precision={y.mean():.3f} ===")
    # --- 1. sync distribution at fires ---
    sv = d["sync"].to_numpy()
    print(f"sync: distinct={len(np.unique(sv))}  min={sv.min():.4f}  "
          f"median={np.median(sv):.4f}  max={sv.max():.4f}  "
          f"frac(sync==1.0)={(sv==1.0).mean():.3f}  frac(sync==0.0)={(sv==0.0).mean():.3f}")
    # --- 2. frame_frac distribution + tercile precision ---
    ff = d["frame_frac"].to_numpy()
    print(f"frame_frac: distinct={len(np.unique(ff))}  min={ff.min():.4f}  "
          f"median={np.median(ff):.4f}  p90={np.percentile(ff,90):.4f}  max={ff.max():.4f}")
    qs = d["frame_frac"].quantile([1/3, 2/3]).to_numpy()
    lo = ff <= qs[0]; hi = ff >= qs[1]; mid = ~lo & ~hi
    print(f"frame_frac tercile cuts: q33={qs[0]:.4f} q67={qs[1]:.4f} | "
          f"n lo/mid/hi = {lo.sum()}/{mid.sum()}/{hi.sum()}")
    p_lo = y[lo].mean(); p_mid = y[mid].mean() if mid.sum() else float("nan"); p_hi = y[hi].mean()
    print(f"hit precision by frame_frac tercile: low {p_lo:.3f}  mid {p_mid:.3f}  high {p_hi:.3f}  "
          f"(low-high diff {p_lo-p_hi:+.3f})")
    sids = d["stream"].unique(); boots = []
    for _ in range(2000):
        pick = rng.choice(sids, len(sids), replace=True)
        bb = pd.concat([d[d["stream"]==s] for s in pick])
        q = bb["frame_frac"].quantile([1/3, 2/3]).to_numpy()
        a = bb[bb["frame_frac"]<=q[0]]["hit"]; b = bb[bb["frame_frac"]>=q[1]]["hit"]
        if len(a) and len(b): boots.append(a.mean() - b.mean())
    boots = np.array(boots)
    print(f"cluster bootstrap (low - high precision): p={(boots<=0).mean():.3f}  "
          f"(boot mean {boots.mean():+.3f}, n={len(boots)})")
