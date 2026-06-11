"""D3 — registered dip-selection payoff test (rules in plan addendum)."""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import logging; logging.disable(logging.INFO)
import numpy as np, pandas as pd
from src.detector import SpeculatorDetector
from src.indicators import Params
from src.pooled_validation import load_stream_frame
from src.universe import resolve_streams

SWING_W, COOLDOWN = 20, 7
run = json.loads(Path(r"C:\Users\kuben\Downloads\v17gpu_20260611_090553_v17gpu.json").read_text())
p_low = Params(**{k: v for k, v in run["sides"]["low"]["best_params"].items() if k != "baseline_lb"})

def naive_fires(df, N, lo_b, hi_b):
    low = df["low"].to_numpy(float)
    rmin = pd.Series(low).rolling(N, min_periods=N).min().to_numpy()
    raw = low <= rmin + 1e-12
    out, last = [], -10**9
    for t in range(lo_b, hi_b):
        if raw[t] and (t - last) > COOLDOWN:
            out.append(t); last = t
    return np.array(out, int)

rows = []
for st in resolve_streams(["INDICES","COMMODITIES","FX","WORLD_ETF"], ["1D"], "data/raw_v16"):
    df = load_stream_frame(st.path); n = len(df)
    logc = np.log(df["close"].to_numpy(float))
    lo_b, hi_b = 300, n - max(SWING_W, 40)
    det = SpeculatorDetector(df, p_low).run()
    dF = np.where(det["signal_low"].to_numpy())[0]
    dF = dF[(dF >= lo_b) & (dF < hi_b)]
    if len(dF) < 5: continue
    best = None
    for N in range(5, 201, 5):
        f = naive_fires(df, N, lo_b, hi_b)
        gap = abs(len(f) - len(dF))
        if best is None or gap < best[0]: best = (gap, f, N)
        if gap <= 0.10 * len(dF): break
    nF = best[1]
    f20 = lambda t: logc[t+20] - logc[t]
    f40 = lambda t: logc[t+40] - logc[t]
    for kind, idxs in (("det", dF), ("naive", nF)):
        for t in idxs:
            rows.append(dict(stream=st.stream_id, kind=kind, fwd20=f20(t), fwd40=f40(t)))
    print(st.stream_id, "done", flush=True)

d = pd.DataFrame(rows)
det, nai = d[d.kind=="det"], d[d.kind=="naive"]
print(f"\nfires: det={len(det)} naive={len(nai)}")
for h in ("fwd20","fwd40"):
    print(f"{h}: det mean {det[h].mean():+.5f} median {det[h].median():+.5f} | "
          f"naive mean {nai[h].mean():+.5f} median {nai[h].median():+.5f} | "
          f"delta(mean) {det[h].mean()-nai[h].mean():+.5f}")
# D3a pooled bootstrap on MEAN delta, both horizons
rng = np.random.default_rng(0); sids = d["stream"].unique()
pv = {}
for h in ("fwd20","fwd40"):
    boots = []
    for _ in range(2000):
        pick = rng.choice(sids, len(sids), replace=True)
        bb = pd.concat([d[d.stream==s] for s in pick])
        a, b = bb[bb.kind=="det"][h], bb[bb.kind=="naive"][h]
        if len(a) and len(b): boots.append(a.mean()-b.mean())
    pv[h] = float((np.asarray(boots) <= 0).mean())
print(f"D3a: p(fwd20)={pv['fwd20']:.3f} p(fwd40)={pv['fwd40']:.3f} -> "
      f"{'PASS' if pv['fwd20']<0.05 and pv['fwd40']<0.05 else 'FAIL'}")
# D3b per-stream consistency
wins = 0; tot = 0
for s, grp in d.groupby("stream"):
    a, b = grp[grp.kind=="det"], grp[grp.kind=="naive"]
    if len(a) < 5 or len(b) < 5: continue
    tot += 1
    if (a["fwd20"].median() > b["fwd20"].median()) or (a["fwd40"].median() > b["fwd40"].median()):
        wins += 1
print(f"D3b: detector favored in {wins}/{tot} streams -> {'PASS' if wins>=10 else 'FAIL'}")
# D3c effect floor
dm = det["fwd20"].median() - nai["fwd20"].median()
print(f"D3c: median fwd20 delta {dm:+.5f} -> {'PASS' if dm>=0.0010 else 'FAIL'}")
