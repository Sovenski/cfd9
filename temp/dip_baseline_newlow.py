"""D1/D3 null tests on the NEW LOW winner (ablation run 131405, minus_momentum,
deflated 0.0678). Same registered rules as plan addendum (D1: +3pts location,
p<0.05; D3a/b/c payoff)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import logging; logging.disable(logging.INFO)
import numpy as np, pandas as pd
from src.detector import SpeculatorDetector
from src.indicators import Params
from src.pooled_validation import load_stream_frame
from src.universe import resolve_streams

SWING_W, DROP_ATR, TOL, COOLDOWN = 20, 2.0, 2, 7
LOW_PARAMS = dict(  # FULL best_params (low) from v17gpu_20260611_131405 report
    S_detect_high=12, scale_start_high=3, scale_end_high=270, scale_step_high=3,
    min_duration_high=13, cooldown_bars_high=9, price_gate_lb_high=23,
    vola_range_len_high=120, er_period_high=28, pct_extreme_high=0.96,
    min_agreement_high=0.75, dur_extreme_pct_high=0.83, confirm_count_high=5,
    vol_surge_thresh_high=1.5, scale_div_thresh_high=0.35, slope_thresh_high=0.22,
    vola_high_pct_high=0.92, pivot_drift_lookback_high=5, pivot_drift_thresh_high=0.008,
    pivot_drift_gate_mult_high=8.0, pivot_drift_confirm_bias_high=1,
    momentum_velocity_thresh_high=0.014, er_directional_high=False,
    use_trend_high=True, use_volume_high=True, use_momentum_high=False,
    use_momentum_velocity_high=True, use_volatility_high=True, use_er_gate_high=False,
    use_gjr_asym_high=True, use_har_vol_high=True, gjr_vote_thresh_high=0.15,
    har_vote_thresh_high=0.15, vola_method_high="ATR",
    momentum_velocity_mode_high="Reversal", use_edge_voting_high=False, edge_window_high=5,
    S_detect_low=26, scale_start_low=19, scale_end_low=250, scale_step_low=15,
    min_duration_low=2, cooldown_bars_low=7, price_gate_lb_low=58,
    vola_range_len_low=20, er_period_low=47, pct_extreme_low=0.49268731212549655,
    min_agreement_low=0.02065350717621217, dur_extreme_pct_low=0.7096091393768227,
    confirm_count_low=3, vol_surge_thresh_low=2.2382428216185266,
    scale_div_thresh_low=0.7358944084387257, slope_thresh_low=0.4025458749698267,
    vola_high_pct_low=0.5571651880232253, pivot_drift_lookback_low=10,
    pivot_drift_thresh_low=0.030211982920262805, pivot_drift_gate_mult_low=4.0,
    pivot_drift_confirm_bias_low=0, momentum_velocity_thresh_low=0.003987965341605985,
    er_directional_low=False, use_trend_low=True, use_volume_low=True,
    use_momentum_low=False, use_momentum_velocity_low=True, use_volatility_low=True,
    use_er_gate_low=False, use_gjr_asym_low=True, use_har_vol_low=True,
    gjr_vote_thresh_low=0.3614954474374879, har_vote_thresh_low=0.11076327581817122,
    vola_method_low="ATR", momentum_velocity_mode_low="Reversal",
    use_edge_voting_low=False, edge_window_low=5,
)
p_low = Params(**LOW_PARAMS)

def wilder_atr(df, n=14):
    h,l,c = (df[k].to_numpy(float) for k in ("high","low","close"))
    tr = np.maximum(h[1:]-l[1:], np.maximum(np.abs(h[1:]-c[:-1]), np.abs(l[1:]-c[:-1])))
    a = pd.Series(np.concatenate([[h[0]-l[0]], tr])).ewm(alpha=1/n, adjust=False).mean().to_numpy()
    a[a==0]=np.nan; return a

def real_bottoms(df, atr):
    h,l = df["high"].to_numpy(float), df["low"].to_numpy(float); n=len(df)
    out = np.zeros(n, bool)
    for t in range(SWING_W, n-SWING_W):
        if l[t]==l[t-SWING_W:t+SWING_W+1].min() and \
           (h[t+1:t+SWING_W+1].max()-l[t])>=DROP_ATR*atr[t]: out[t]=True
    return out

def naive_fires(df, N, lo_b, hi_b):
    low = df["low"].to_numpy(float)
    rmin = pd.Series(low).rolling(N, min_periods=N).min().to_numpy()
    raw = low <= rmin + 1e-12
    out, last = [], -10**9
    for t in range(lo_b, hi_b):
        if raw[t] and (t-last) > COOLDOWN: out.append(t); last = t
    return np.array(out, int)

rows = []
for st in resolve_streams(["INDICES","COMMODITIES","FX","WORLD_ETF"], ["1D"], "data/raw_v16"):
    df = load_stream_frame(st.path); atr = wilder_atr(df); n = len(df)
    logc = np.log(df["close"].to_numpy(float))
    lo_b, hi_b = 300, n - max(SWING_W, 40)
    det = SpeculatorDetector(df, p_low).run()
    dF = np.where(det["signal_low"].to_numpy())[0]
    dF = dF[(dF >= lo_b) & (dF < hi_b)]
    if len(dF) < 5: continue
    best = None
    for N in range(5, 201, 5):
        f = naive_fires(df, N, lo_b, hi_b)
        gap = abs(len(f)-len(dF))
        if best is None or gap < best[0]: best = (gap, f, N)
        if gap <= 0.10*len(dF): break
    turn = real_bottoms(df, atr)
    for kind, idxs in (("det", dF), ("naive", best[1])):
        for t in idxs:
            rows.append(dict(stream=st.stream_id, kind=kind,
                hit=bool(turn[max(0,t-TOL):t+TOL+1].any()),
                fwd20=logc[t+20]-logc[t], fwd40=logc[t+40]-logc[t]))
    print(st.stream_id, "det", len(dF), "naive", len(best[1]), flush=True)

d = pd.DataFrame(rows)
det, nai = d[d.kind=="det"], d[d.kind=="naive"]
p_d, p_n = det["hit"].mean(), nai["hit"].mean()
print(f"\nfires: det={len(det)} naive={len(nai)}")
print(f"D1: det precision {p_d:.3f} vs naive {p_n:.3f} (diff {p_d-p_n:+.3f})")
rng = np.random.default_rng(0); sids = d["stream"].unique()
def boot(metric):
    bs = []
    for _ in range(2000):
        pick = rng.choice(sids, len(sids), replace=True)
        bb = pd.concat([d[d.stream==s] for s in pick])
        a, b = bb[bb.kind=="det"], bb[bb.kind=="naive"]
        if len(a) and len(b): bs.append(metric(a)-metric(b))
    return float((np.asarray(bs) <= 0).mean())
p1 = boot(lambda x: x["hit"].mean())
print(f"  cluster p={p1:.3f} -> D1 {'PASS' if (p_d-p_n)>=0.03 and p1<0.05 else 'FAIL'}")
for h in ("fwd20","fwd40"):
    ph = boot(lambda x, h=h: x[h].mean())
    print(f"D3 {h}: det mean {det[h].mean():+.5f} naive {nai[h].mean():+.5f} "
          f"delta {det[h].mean()-nai[h].mean():+.5f} p={ph:.3f}")
wins = sum((grp[grp.kind=='det']['fwd20'].median() > grp[grp.kind=='naive']['fwd20'].median()) or
           (grp[grp.kind=='det']['fwd40'].median() > grp[grp.kind=='naive']['fwd40'].median())
           for s, grp in d.groupby("stream")
           if len(grp[grp.kind=='det'])>=5 and len(grp[grp.kind=='naive'])>=5)
print(f"D3b consistency: {wins}/16 streams favor detector")
