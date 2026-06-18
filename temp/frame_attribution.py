"""T1 frame-attribution + T2 synchrony of the pooled 090553 fires.

Frame event (per scale s, side high): the holder (argmax) of the rolling
max(s,20)-bar window over ratio=close/SMA_s rotates because the old extreme
EXPIRED (new holder != current bar). Side low: same on the window min.
Also tracked: the price-gate rolling max(price_gate_lb) window.
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
    close = df["close"]; n = len(df)
    pirM, _, scales = precompute_matrices(close)
    g = lambda k: getattr(p, f"{k}_{side}")
    sl = [s for s in range(g("scale_start"), g("scale_end")+1, g("scale_step"))
          if scales[0] <= s <= scales[-1]]
    idx = [s - scales[0] for s in sl]
    pct = g("pct_extreme")
    # ratio rows for expiry tracking (recompute: cheap vs matrix memory games)
    csum = np.cumsum(close.to_numpy(np.float64))
    frame_any = np.zeros(n, bool); ext = np.zeros((len(sl), n), bool); fresh = np.zeros((len(sl), n), bool)
    for j, s in enumerate(sl):
        sma = np.full(n, np.nan); sma[s:] = (csum[s:]-csum[:-s])/s
        ratio = np.where(sma>0, close.to_numpy(np.float64)/sma, 1.0); ratio[np.isnan(sma)] = np.nan
        W = max(s, 20)
        ev = expiry_events(ratio, W, "max" if side=="high" else "min")
        frame_any |= pd.Series(ev).rolling(FRAME_X, min_periods=1).max().to_numpy().astype(bool)
        pir = pirM[idx[j]]
        e = (pir > pct) if side=="high" else (pir < 1.0-pct)
        ext[j] = np.nan_to_num(e)
        crossed = ext[j] & ~np.roll(ext[j], 1); crossed[0] = ext[j][0]
        fresh[j] = pd.Series(crossed).rolling(SYNC_K, min_periods=1).max().to_numpy().astype(bool) & ext[j]
    # price-gate window counts as one extra reference frame
    px = (df["high"] if side=="high" else df["low"]).to_numpy(float)
    evg = expiry_events(px, g("price_gate_lb"), "max" if side=="high" else "min")
    frame_any |= pd.Series(evg).rolling(FRAME_X, min_periods=1).max().to_numpy().astype(bool)
    n_ext = ext.sum(0); n_fresh = fresh.sum(0)
    sync = np.where(n_ext > 0, n_fresh / np.maximum(n_ext, 1), 0.0)
    return frame_any, sync

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
        frame_any, sync = frame_and_sync(df, P[side], side)
        for t in fired:
            rows[side].append(dict(stream=st.stream_id, t=int(t),
                hit=bool(turn[max(0,t-TOL):t+TOL+1].any()),
                frame=bool(frame_any[t]), sync=float(sync[t])))
    print(f"{st.stream_id} done", flush=True)

rng = np.random.default_rng(0)
for side in ("high","low"):
    d = pd.DataFrame(rows[side]); y = d["hit"].to_numpy(int)
    print(f"\n=== {side.upper()}  fires={len(d)}  precision={y.mean():.3f} ===")
    # --- T1 ---
    f = d["frame"].to_numpy(bool)
    pc, pf = y[~f].mean(), y[f].mean()
    sids = d["stream"].unique(); boots = []
    for _ in range(2000):
        pick = rng.choice(sids, len(sids), replace=True)
        bb = pd.concat([d[d["stream"]==s] for s in pick])
        if bb["frame"].sum() and (~bb["frame"]).sum():
            boots.append(bb[~bb["frame"]]["hit"].mean() - bb[bb["frame"]]["hit"].mean())
    boots = np.array(boots); pval = float((boots <= 0).mean())
    per = [(s, grp[~grp["frame"]]["hit"].mean() > grp[grp["frame"]]["hit"].mean())
           for s, grp in d.groupby("stream")
           if len(grp) >= 8 and grp["frame"].sum() and (~grp["frame"]).sum()]
    wins = sum(w for _, w in per)
    print(f"T1: frame-coincident {f.mean():.1%} of fires | precision clean {pc:.3f} vs "
          f"frame {pf:.3f} (diff {pc-pf:+.3f}, boot p={pval:.3f}) | streams {wins}/{len(per)}")
    print(f"T1 verdict: {'PASS' if (pc-pf)>=0.03 and pval<0.05 and wins>=10 else 'FAIL'}")
    # --- T2 ---
    qs = d["sync"].quantile([1/3, 2/3]).to_numpy()
    lo_m, hi_m = d["sync"]<=qs[0], d["sync"]>=qs[1]
    pt, pb = y[hi_m.to_numpy()].mean(), y[lo_m.to_numpy()].mean()
    boots2 = []
    for _ in range(2000):
        pick = rng.choice(sids, len(sids), replace=True)
        bb = pd.concat([d[d["stream"]==s] for s in pick])
        q = bb["sync"].quantile([1/3, 2/3]).to_numpy()
        a, b = bb[bb["sync"]>=q[1]]["hit"], bb[bb["sync"]<=q[0]]["hit"]
        if len(a) and len(b): boots2.append(a.mean()-b.mean())
    boots2 = np.array(boots2); pval2 = float((boots2 <= 0).mean())
    per2 = [(s, grp[grp["sync"]>=grp["sync"].median()]["hit"].mean()
                > grp[grp["sync"]<grp["sync"].median()]["hit"].mean())
            for s, grp in d.groupby("stream") if len(grp) >= 8]
    wins2 = sum(w for _, w in per2)
    print(f"T2: sync terciles precision top {pt:.3f} vs bottom {pb:.3f} "
          f"(diff {pt-pb:+.3f}, boot p={pval2:.3f}) | streams {wins2}/{len(per2)}")
    print(f"T2 verdict: {'PASS' if (pt-pb)>=0.05 and pval2<0.05 and wins2>=10 else 'FAIL'}")
