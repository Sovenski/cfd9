"""Confirmatory leg: D3 on the HOLDOUT TAIL ONLY (bars after 2015-06-09 embargo
start) — the selection-untouched period — for the new LOW (131405 minus_momentum)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import logging; logging.disable(logging.INFO)
import numpy as np, pandas as pd
import importlib.util
spec = importlib.util.spec_from_file_location("base", "temp/dip_baseline_newlow.py")
# reuse param dict + helpers by re-defining inline (avoid running base script)
exec(Path("temp/dip_baseline_newlow.py").read_text().split("rows = []")[0].replace(
    'print(st.stream_id', '#print(st.stream_id'))

TAIL_TS = pd.Timestamp("2015-06-09").value // 10**9
rows = []
for st in resolve_streams(["INDICES","COMMODITIES","FX","WORLD_ETF"], ["1D"], "data/raw_v16"):
    df = load_stream_frame(st.path); atr = wilder_atr(df); n = len(df)
    ts = df["time"].to_numpy(float) if "time" in df else None
    logc = np.log(df["close"].to_numpy(float))
    lo_b, hi_b = 300, n - max(SWING_W, 40)
    tail_lo = int(np.searchsorted(ts, TAIL_TS)) if ts is not None else lo_b
    lo_eff = max(lo_b, tail_lo + 200)   # embargo
    if hi_b - lo_eff < 100: continue
    det = SpeculatorDetector(df, p_low).run()
    dF = np.where(det["signal_low"].to_numpy())[0]
    dF = dF[(dF >= lo_eff) & (dF < hi_b)]
    if len(dF) < 3: continue
    best = None
    for N in range(5, 201, 5):
        f = naive_fires(df, N, lo_eff, hi_b)
        gap = abs(len(f)-len(dF))
        if best is None or gap < best[0]: best = (gap, f, N)
        if gap <= 0.10*len(dF): break
    turn = real_bottoms(df, atr)
    for kind, idxs in (("det", dF), ("naive", best[1])):
        for t in idxs:
            rows.append(dict(stream=st.stream_id, kind=kind,
                hit=bool(turn[max(0,t-TOL):t+TOL+1].any()),
                fwd20=logc[t+20]-logc[t], fwd40=logc[t+40]-logc[t]))
d = pd.DataFrame(rows)
det, nai = d[d.kind=="det"], d[d.kind=="naive"]
print(f"HOLDOUT-TAIL fires: det={len(det)} naive={len(nai)}")
print(f"D1 tail: det {det['hit'].mean():.3f} vs naive {nai['hit'].mean():.3f}")
rng = np.random.default_rng(0); sids = d["stream"].unique()
def boot(metric):
    bs = []
    for _ in range(2000):
        pick = rng.choice(sids, len(sids), replace=True)
        bb = pd.concat([d[d.stream==s] for s in pick])
        a, b = bb[bb.kind=="det"], bb[bb.kind=="naive"]
        if len(a) and len(b): bs.append(metric(a)-metric(b))
    return float((np.asarray(bs) <= 0).mean())
for h in ("fwd20","fwd40"):
    p = boot(lambda x, h=h: x[h].mean())
    print(f"D3 tail {h}: det {det[h].mean():+.5f} naive {nai[h].mean():+.5f} "
          f"delta {det[h].mean()-nai[h].mean():+.5f} p={p:.3f}")
wins = sum((g[g.kind=='det']['fwd20'].median() > g[g.kind=='naive']['fwd20'].median()) or
           (g[g.kind=='det']['fwd40'].median() > g[g.kind=='naive']['fwd40'].median())
           for s, g in d.groupby("stream")
           if len(g[g.kind=='det'])>=3 and len(g[g.kind=='naive'])>=3)
print(f"consistency: {wins}/{d['stream'].nunique()} streams favor detector")
