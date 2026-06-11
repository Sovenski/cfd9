"""Miss autopsy — WHY does the detector miss real turns? (habituation diagnosis)

T1-T3 tested the fires (precision side). Habituation's damage is in the
MISSES (recall side): real turns where the detector stayed silent. For every
pooled real turn not matched by a fire (±TOL), attribute the block using the
REAL detector's debug columns (include_debug_columns=True) + known thresholds:

  agreement   agreement < min_agreement at every bar of the ±TOL window
              (the habituation signature — Zalando ATH: 0.30 vs 0.544)
  price_gate  rolling-extreme touch failed
  votes       ph/pl_confirms < required (req=1 for both winners: single vote)
  cooldown    a fire occurred within cooldown_bars before the window
  other       scale_div/dur/drift-gate residual (not directly exported)

A condition is 'blocking' if it fails on ALL bars of the ±TOL window (any
passing bar would have allowed a fire on that bar for that condition).
Also: agreement shortfall distribution at misses vs detected turns.
"""
import sys, json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import logging

logging.disable(logging.INFO)
import numpy as np
import pandas as pd

from src.detector import SpeculatorDetector
from src.indicators import Params, pir_of, sma
from src.pooled_validation import load_stream_frame
from src.universe import resolve_streams

SWING_W, DROP_ATR, TOL = 20, 2.0, 2
run = json.loads(Path(r"C:\Users\kuben\Downloads\v17gpu_20260611_090553_v17gpu.json").read_text())
P = {s: Params(**{k: v for k, v in run["sides"][s]["best_params"].items() if k != "baseline_lb"})
     for s in ("high", "low")}

def wilder_atr(df, n=14):
    h, l, c = (df[k].to_numpy(float) for k in ("high", "low", "close"))
    tr = np.maximum(h[1:] - l[1:], np.maximum(np.abs(h[1:] - c[:-1]), np.abs(l[1:] - c[:-1])))
    a = pd.Series(np.concatenate([[h[0] - l[0]], tr])).ewm(alpha=1 / n, adjust=False).mean().to_numpy()
    a[a == 0] = np.nan
    return a

def real_turns(df, kind, atr):
    h, l = df["high"].to_numpy(float), df["low"].to_numpy(float)
    n = len(df)
    out = np.zeros(n, bool)
    for t in range(SWING_W, n - SWING_W):
        if kind == "top" and h[t] == h[t - SWING_W:t + SWING_W + 1].max() and \
           (h[t] - l[t + 1:t + SWING_W + 1].min()) >= DROP_ATR * atr[t]:
            out[t] = True
        elif kind == "bot" and l[t] == l[t - SWING_W:t + SWING_W + 1].min() and \
             (h[t + 1:t + SWING_W + 1].max() - l[t]) >= DROP_ATR * atr[t]:
            out[t] = True
    return out

def scale_div_flag(df, p, side):
    """|pir_detect - agreement| > thresh  (gate blocks when TRUE) — formula copy."""
    close = df["close"]
    g = lambda k: getattr(p, f"{k}_{side}")
    pird = pir_of(close / sma(close, g("S_detect")).clip(lower=1e-9), max(g("S_detect"), 20))
    return pird  # combined with exported agreement downstream

streams = resolve_streams(["INDICES", "COMMODITIES", "FX", "WORLD_ETF"], ["1D"], "data/raw_v16")
rows = {"high": [], "low": []}
for st in streams:
    df = load_stream_frame(st.path)
    atr = wilder_atr(df)
    n = len(df)
    for side, sigcol, kind in (("high", "signal_high", "top"), ("low", "signal_low", "bot")):
        p = P[side]
        g = lambda k: getattr(p, f"{k}_{side}")
        det = SpeculatorDetector(df, p, include_debug_columns=True).run()
        sig = det[sigcol].to_numpy().astype(bool)
        agr = det[f"agreement_{side}_side"].to_numpy(float)
        pg = det[f"price_gate_{side}"].to_numpy(float) > 0
        conf = det["ph_confirms" if side == "high" else "pl_confirms"].to_numpy(float)
        pird = scale_div_flag(df, p, side).to_numpy(float)
        sd = np.abs((pird - agr) if side == "high" else ((1.0 - pird) - agr)) > g("scale_div_thresh")
        turn = real_turns(df, kind, atr)
        fire_idx = np.where(sig)[0]
        cd = g("cooldown_bars")
        for t in np.where(turn)[0]:
            if t < 300 or t >= n - SWING_W:
                continue
            wlo, whi = t - TOL, t + TOL + 1
            detected = sig[wlo:whi].any()
            win = slice(wlo, whi)
            # blocking = condition fails on EVERY bar of the window
            blk_agr = bool((agr[win] < g("min_agreement")).all())
            blk_pg = bool((~pg[win]).all())
            blk_vote = bool((conf[win] < 1).all())
            blk_sd = bool(sd[win].all())
            prior = fire_idx[(fire_idx < wlo)]
            blk_cool = bool(len(prior) and (wlo - prior[-1]) <= cd)
            rows[side].append(dict(
                stream=st.stream_id, t=int(t), detected=bool(detected),
                blk_agr=blk_agr, blk_pg=blk_pg, blk_vote=blk_vote,
                blk_sd=blk_sd, blk_cool=blk_cool,
                agr_max=float(np.nanmax(agr[win])), thr=float(g("min_agreement")),
            ))
    print(f"{st.stream_id} done", flush=True)

for side in ("high", "low"):
    d = pd.DataFrame(rows[side])
    miss = d[~d["detected"]]
    print(f"\n=== {side.upper()}  real turns={len(d)}  detected={d['detected'].mean():.1%}"
          f"  missed={len(miss)} ===")
    for c, name in (("blk_agr", "agreement"), ("blk_pg", "price_gate"),
                    ("blk_vote", "votes"), ("blk_sd", "scale_div"),
                    ("blk_cool", "cooldown")):
        print(f"  blocked by {name:<11s}: {miss[c].mean():.1%} of misses")
    only_agr = miss["blk_agr"] & ~miss["blk_pg"] & ~miss["blk_vote"] & ~miss["blk_sd"] & ~miss["blk_cool"]
    none_known = ~(miss["blk_agr"] | miss["blk_pg"] | miss["blk_vote"] | miss["blk_sd"] | miss["blk_cool"])
    print(f"  agreement is the SOLE identified blocker: {only_agr.mean():.1%}")
    print(f"  no identified blocker (dur/drift/er residual): {none_known.mean():.1%}")
    # habituation depth: how far below threshold is agreement at agreement-blocked misses?
    ab = miss[miss["blk_agr"]]
    if len(ab):
        short = (ab["thr"] - ab["agr_max"])
        print(f"  agreement shortfall at agr-blocked misses: median {short.median():.3f}"
              f"  p25 {short.quantile(.25):.3f}  p75 {short.quantile(.75):.3f}"
              f"  (threshold {ab['thr'].iloc[0]:.3f})")
    det_t = d[d["detected"]]
    print(f"  agreement at detected turns: median {det_t['agr_max'].median():.3f}"
          f"  | at missed: median {miss['agr_max'].median():.3f}")
