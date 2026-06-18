"""Habituation-wedge tests P1/P2/P3/P4b (pre-registered: plan/habituation-wedge-tests.md).

Usage: python temp/habituation_wedge.py [--anchor2]   (--anchor2 = P5 robustness:
anchor at the 2nd-last opposite confirmed pivot)
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
from src.indicators import Params, pivot_high_pine, pivot_low_pine
from src.pooled_validation import load_stream_frame
from src.universe import resolve_streams

SWING_W, DROP_ATR, TOL = 20, 2.0, 2
MIN_ANCHOR_AGE = 5
ANCHOR_BACK = 2 if "--anchor2" in sys.argv else 1   # P5: 2nd-last opposite pivot
FIRE_PREC = {"high": 0.240, "low": 0.286}           # pre-registered baselines

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


def anchors_for(df, side: str) -> np.ndarray:
    """anchor[t] = bar index of the ANCHOR_BACK-th last confirmed OPPOSITE pivot."""
    n = len(df)
    if side == "high":
        pv = pivot_low_pine(df["low"], 20)
    else:
        pv = pivot_high_pine(df["high"], 20)
    conf = [(i + 20, i) for i in np.where(pv.notna())[0]]   # (confirm_bar, pivot_bar)
    out = np.full(n, -1, dtype=int)
    seq: list[int] = []
    j = 0
    for t in range(n):
        while j < len(conf) and conf[j][0] <= t:
            seq.append(conf[j][1])
            j += 1
        if len(seq) >= ANCHOR_BACK:
            out[t] = seq[-ANCHOR_BACK]
    return out


def side_pirs(df, p: Params, side: str):
    """bar-PIR and anchored-PIR per slice scale -> (agr_bar, agr_anch, wedge)."""
    close = df["close"].to_numpy(np.float64)
    n = len(close)
    g = lambda k: getattr(p, f"{k}_{side}")
    sl = list(range(g("scale_start"), g("scale_end") + 1, g("scale_step")))
    pct = g("pct_extreme")
    csum = np.cumsum(close)
    anchor = anchors_for(df, side)
    bar_sum = np.zeros(n); anch_sum = np.zeros(n)
    bar_ext = np.zeros(n); anch_ext = np.zeros(n)
    valid = np.zeros(n)
    for s in sl:
        sma = np.full(n, np.nan)
        if s < n:
            sma[s:] = (csum[s:] - csum[:-s]) / s
        ratio = np.where(sma > 0, close / sma, 1.0)
        ratio[np.isnan(sma)] = np.nan
        # bar-PIR (status quo, max(s,20) rolling window, partial allowed)
        W = max(s, 20)
        r = pd.Series(ratio)
        lo = r.rolling(W, min_periods=1).min().to_numpy()
        hi = r.rolling(W, min_periods=1).max().to_numpy()
        with np.errstate(divide="ignore", invalid="ignore"):
            pir_b = np.where(hi != lo, (ratio - lo) / (hi - lo), 0.5)
        # anchored-PIR: running min/max since anchor (anchor non-decreasing)
        pir_a = np.full(n, np.nan)
        cur_a = -2; rlo = np.inf; rhi = -np.inf; astart = -1
        for t in range(n):
            a = anchor[t]
            if a < 0 or np.isnan(ratio[t]):
                cur_a = -2
                continue
            if a != cur_a:                      # anchor advanced -> recompute window
                cur_a = a; astart = a
                w = ratio[a:t + 1]
                w = w[~np.isnan(w)]
                if len(w) == 0:
                    rlo, rhi = np.inf, -np.inf
                else:
                    rlo, rhi = w.min(), w.max()
            else:
                rlo = min(rlo, ratio[t]); rhi = max(rhi, ratio[t])
            if t - astart >= MIN_ANCHOR_AGE and rhi != rlo:
                pir_a[t] = (ratio[t] - rlo) / (rhi - rlo)
        ok = ~np.isnan(pir_b) & ~np.isnan(pir_a)
        if side == "high":
            b_pos, a_pos = pir_b, pir_a
            b_ex, a_ex = pir_b > pct, pir_a > pct
        else:
            b_pos, a_pos = 1.0 - pir_b, 1.0 - pir_a
            b_ex, a_ex = pir_b < 1.0 - pct, pir_a < 1.0 - pct
        bar_sum[ok] += b_pos[ok]; anch_sum[ok] += a_pos[ok]
        bar_ext[ok] += b_ex[ok]; anch_ext[ok] += a_ex[ok]
        valid[ok] += 1
    v = np.maximum(valid, 1)
    agr_bar = bar_ext / v; agr_anch = anch_ext / v
    wedge = (anch_sum - bar_sum) / v
    wedge[valid < len(sl) * 0.5] = np.nan       # need most scales valid
    agr_anch[valid < len(sl) * 0.5] = np.nan
    return agr_bar, agr_anch, wedge


def cluster_boot(d, col_a_mask, col_b_mask, value_col, n_boot=2000, seed=0):
    """bootstrap p for median(value|A) - median(value|B) <= 0, clustered by stream."""
    rng = np.random.default_rng(seed)
    sids = d["stream"].unique(); diffs = []
    for _ in range(n_boot):
        pick = rng.choice(sids, len(sids), replace=True)
        bb = pd.concat([d[d["stream"] == s] for s in pick])
        a = bb[bb["_a"]][value_col]; b = bb[bb["_b"]][value_col]
        if len(a) and len(b):
            diffs.append(a.median() - b.median())
    diffs = np.asarray(diffs)
    return float((diffs <= 0).mean()), float(np.mean(diffs))


streams = resolve_streams(["INDICES", "COMMODITIES", "FX", "WORLD_ETF"], ["1D"], "data/raw_v16")
bars_rows = {"high": [], "low": []}
turn_rows = {"high": [], "low": []}
for st in streams:
    df = load_stream_frame(st.path)
    atr = wilder_atr(df)
    n = len(df)
    logc = np.log(df["close"].to_numpy(float))
    for side, sigcol, kind in (("high", "signal_high", "top"), ("low", "signal_low", "bot")):
        p = P[side]
        g = lambda k: getattr(p, f"{k}_{side}")
        det = SpeculatorDetector(df, p, include_debug_columns=True).run()
        sig = det[sigcol].to_numpy().astype(bool)
        agr_det = det[f"agreement_{side}_side"].to_numpy(float)
        pg = det[f"price_gate_{side}"].to_numpy(float) > 0
        agr_bar, agr_anch, wedge = side_pirs(df, p, side)
        turn = real_turns(df, kind, atr)
        near_turn5 = pd.Series(turn).rolling(11, center=True, min_periods=1).max().to_numpy().astype(bool)
        thr = g("min_agreement")
        # per-bar table (for P1 recovered set + P2 control + P3)
        el = np.arange(300, n - SWING_W)
        sgn = -1 if side == "high" else 1
        fwd20 = np.full(n, np.nan); fwd40 = np.full(n, np.nan)
        fwd20[:-20] = sgn * (logc[20:] - logc[:-20])
        fwd40[:-40] = sgn * (logc[40:] - logc[:-40])
        for t in el:
            if np.isnan(agr_anch[t]):
                continue
            bars_rows[side].append(dict(
                stream=st.stream_id, t=int(t),
                turn=bool(turn[max(0, t - TOL):t + TOL + 1].any()),
                near5=bool(near_turn5[t]),
                weak=bool(agr_det[t] < thr),
                rec=bool(agr_anch[t] >= thr and agr_det[t] < thr),
                wedge=float(wedge[t]) if not np.isnan(wedge[t]) else np.nan,
                fwd20=float(fwd20[t]) if not np.isnan(fwd20[t]) else np.nan,
                fwd40=float(fwd40[t]) if not np.isnan(fwd40[t]) else np.nan,
            ))
        # per-turn table (P1 recovery of blocked misses, P4b)
        dur_pct = g("dur_extreme_pct")
        dur_flag = np.zeros(n, bool); da = 0; dm = 0
        for t in range(n):
            if agr_det[t] > dur_pct:
                da += 1; dm = 0
            else:
                dm += 1
                if dm > 1:
                    da = 0
            dur_flag[t] = da >= g("min_duration")
        for t in np.where(turn)[0]:
            if t < 300 or t >= n - SWING_W or np.isnan(agr_anch[t]):
                continue
            w = slice(t - TOL, t + TOL + 1)
            turn_rows[side].append(dict(
                stream=st.stream_id, t=int(t),
                detected=bool(sig[w].any()),
                blk_agr=bool((agr_det[w] < thr).all()),
                blk_dur=bool((~dur_flag[w]).all()),
                blk_pg=bool((~pg[w]).all()),
                rec=bool((agr_anch[w] >= thr).any()),
            ))
    print(f"{st.stream_id} done", flush=True)

tag = f"(anchor={ANCHOR_BACK})"
for side in ("high", "low"):
    B = pd.DataFrame(bars_rows[side])
    T = pd.DataFrame(turn_rows[side])
    miss = T[~T["detected"]]
    blocked = miss[miss["blk_agr"] | miss["blk_dur"]]
    print(f"\n=== {side.upper()} {tag}  bars={len(B)}  turns={len(T)} "
          f"(missed {len(miss)}, agr/dur-blocked {len(blocked)}) ===")
    # ---- P2: wedge at blocked misses vs weak-agreement nothing-bars ----
    wb = B.merge(blocked[["stream", "t"]].assign(_blk=True), on=["stream", "t"], how="left")
    wb["_a"] = wb["_blk"].fillna(False).astype(bool)
    wb["_b"] = wb["weak"] & ~wb["near5"]
    a_med = wb[wb["_a"]]["wedge"].median(); b_med = wb[wb["_b"]]["wedge"].median()
    pval, boot_mean = cluster_boot(wb.dropna(subset=["wedge"]), "_a", "_b", "wedge")
    p2 = (a_med - b_med) >= 0.10 and pval < 0.05
    print(f"P2 wedge: blocked-miss median {a_med:+.3f} vs nothing-bars {b_med:+.3f} "
          f"(diff {a_med - b_med:+.3f}, boot p={pval:.3f}) -> {'PASS' if p2 else 'FAIL'}")
    # ---- P1: recovered-set precision + recovery rate ----
    R = B[B["rec"]]
    prec_R = R["turn"].mean() if len(R) else float("nan")
    rec_rate = blocked["rec"].mean() if len(blocked) else float("nan")
    rng = np.random.default_rng(1)
    sids = B["stream"].unique(); boots = []
    for _ in range(2000):
        pick = rng.choice(sids, len(sids), replace=True)
        bb = pd.concat([B[B["stream"] == s] for s in pick])
        rr = bb[bb["rec"]]
        if len(rr):
            boots.append(rr["turn"].mean())
    boots = np.asarray(boots)
    p_prec = float((boots < FIRE_PREC[side]).mean())
    p1 = (prec_R >= FIRE_PREC[side]) and (p_prec < 0.05) and (rec_rate >= 0.30)
    print(f"P1 economics: recovered-only bars n={len(R)} ({len(R)/max(len(B),1):.2%} of bars) "
          f"precision {prec_R:.3f} vs fire-precision {FIRE_PREC[side]:.3f} "
          f"(boot p={p_prec:.3f}) | recovery of blocked misses {rec_rate:.1%} "
          f"-> {'PASS' if p1 else 'FAIL'}")
    # ---- P3: payoff (report-only) ----
    rnd20, rnd40 = B["fwd20"].median(), B["fwd40"].median()
    print(f"P3 payoff: recovered fwd20 {R['fwd20'].median():+.4f} / fwd40 "
          f"{R['fwd40'].median():+.4f}  vs all-bars {rnd20:+.4f} / {rnd40:+.4f}")
    # ---- P4b (LOW only): price-gate-blocked misses ----
    if side == "low":
        pgb = miss[miss["blk_pg"]]
        print(f"P4b: price-gate-blocked misses {len(pgb)}; anchored-extreme among them: "
              f"{pgb['rec'].mean():.1%} (rec flag = anchored>=thr while bar-agr<thr at any window bar)")
