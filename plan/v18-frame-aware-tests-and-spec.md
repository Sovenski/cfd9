# v18 Frame-Aware Detection — Falsification Tests + Conditional Spec

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Empirically test the reference-frame thesis (fires caused by our own sliding windows, not by market events), then spec v18 from the pre-committed decision matrix — only the branches the tests validate get built.

**Architecture:** Part A = three local CPU falsification tests on the pooled 090553 fires with pre-registered decision rules (no garden of forking paths). Part B = conditional v18 detector spec: every branch is an additive, default-off, Pine-portable mechanism with paired Pine+Python build, TV-export re-audit, and golden byte-identity with switches off. Scorer stays v5.1 (detector version ≠ objective era).

**Tech Stack:** numpy/pandas/sklearn locally (real `SpeculatorDetector` fires, no surrogates in the verdict path); existing `shape_variants` machinery for the eventual GPU head-to-head.

**The thesis being tested (from re-anchoring + 2026-06-11 discussion):** every feature is windowed; a threshold crossing has two possible causes — price moved (market event) or the window's contents changed (frame event: an old extreme expired, or the window habituated to trend). Frame events are deterministic functions of our own data and are market-noise. Prediction: frame-coincident fires have lower hit-precision.

---

## Part A — Falsification tests (local, no GPU, ~minutes runtime each)

### Pre-registered constants (locked before any result is seen)

```python
SWING_W, DROP_ATR, TOL = 20, 2.0, 2        # ground truth — same as all prior analyses
FRAME_X = 5          # a fire is frame-coincident if >=1 contributing window had an
                     # expiry rotation within the last X bars
SYNC_K = 3           # a scale's extremeness is "fresh" if it crossed within K bars
EVT_K = 8            # event-time anchor = the 8th most recent confirmed pivot
RUN_JSON = r"C:\Users\kuben\Downloads\v17gpu_20260611_090553_v17gpu.json"
DATA_DIR = "data/raw_v16"; GROUPS = ["INDICES","COMMODITIES","FX","WORLD_ETF"]
```

**Decision rules (pre-committed, no peeking adjustments):**

| Test | PASS iff |
|---|---|
| **T1** frame mask | precision(frame-clean) − precision(frame-coincident) ≥ **3pts** pooled, cluster-bootstrap p < 0.05 (2000 resamples by stream), direction holds in ≥ **10/16** streams with ≥8 fires |
| **T2** synchrony | precision(top sync tercile) − precision(bottom tercile) ≥ **5pts** pooled, same bootstrap + ≥10/16 streams |
| **T3** event-time | leave-streams-out single-feature AUC(pir_evt) ≥ best bar-time pir AUC + **0.03** on both sides (or +0.05 on one) |

Also reported (no decision weight): fraction of fires that are frame-coincident (the size of the prize), and precision at matched firing rates (a mask shrinks n_signals; honesty requires same-selectivity comparison).

### Task A1: Frame-event tracker + T1/T2 script

**Files:**
- Create: `temp/frame_attribution.py`

- [ ] **Step 1: Write the script** (full code; reuses the proven `pooled_selector.py` plumbing for fires/labels)

```python
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
    sw = sliding_window_view(x, W)                     # rows end at t = W-1 .. n-1
    arg = (np.nanargmax(sw,1) if mode=="max" else np.nanargmin(sw,1))
    holder = arg + np.arange(n - W + 1)                # absolute holder index
    t_idx = np.arange(W, n)                            # compare t vs t-1 (full windows)
    h_now, h_prev = holder[1:], holder[:-1]
    ev[t_idx] = (h_now != h_prev) & (h_now != t_idx)   # changed, not set by current bar
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
```

- [ ] **Step 2: Run it**

Run: `python temp/frame_attribution.py`
Expected: per-stream progress lines, then per side: T1 line (frame share, clean-vs-frame precision, p, stream count) + verdict, T2 line + verdict. Runtime ~3–10 min (16 streams × 2 sides × ~90-scale expiry tracking).

- [ ] **Step 3: Record outcomes in this plan** — fill the Decision Matrix table in Part B with PASS/FAIL per side and the measured numbers. Commit the script + updated plan: `git add temp/frame_attribution.py plan/v18-frame-aware-tests-and-spec.md && git commit -m "test(frame): T1/T2 frame-attribution results"`.

### Task A2: T3 event-time PIR script

**Files:**
- Create: `temp/event_time_pir.py`

- [ ] **Step 1: Write the script**

```python
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
```

- [ ] **Step 2: Run it**

Run: `python temp/event_time_pir.py`
Expected: per-side AUC comparison lines + verdict rule. Runtime ~5 min.

- [ ] **Step 3: Record T3 outcome in the Decision Matrix; commit** (`git add temp/event_time_pir.py plan/... && git commit -m "test(frame): T3 event-time PIR result"`).

### Task A3: Matched-selectivity honesty check (only if T1 or T2 PASS)

- [ ] **Step 1:** Re-run the passing test's split, reporting precision **at matched firing rate** (drop the same number of fires by the mask/sync rank vs by RANDOM drop, 200 draws — the lift must beat random pruning's 95th percentile). Append the result to the plan. This guards against "any pruning looks good because the base mix improves."

---

## Part B — v18 conditional spec (decision matrix → branch)

### Decision Matrix (filled 2026-06-11; scripts: temp/frame_attribution.py + temp/frame_diag.py)

| Test | HIGH | LOW | Measured numbers |
|---|---|---|---|
| T1 frame mask | **FAIL** | **FAIL** | HIGH: clean 0.315 vs frame 0.235 (+0.080) but p=0.068, 7/14 streams, mask saturated (94% of fires frame-coincident). LOW: ±0.001, p=0.56. **Dose-response check (frame_frac terciles): flat, wrong direction (−0.008, p≈0.64) both sides** → the HIGH binary trend is a small-remnant/price-gate artifact, not a graded frame effect. |
| T2 synchrony | **FAIL (non-measurement)** | **FAIL (non-measurement)** | sync == 0.0 at ~100% of fire bars (HIGH max 0.013; LOW 99.9% exactly 0). The detector fires long after extremeness onset (confirm/cooldown latency), so ≤3-bar freshness cannot discriminate at fires. Terciles compared identical populations — hypothesis untested as operationalized, not refuted. |
| T3 event-time | ☐ | ☐ | pending (Task A2) |

**Diagnostic finding worth keeping:** at fire bars, NO contributing scale crossed into extremeness within the last 3 bars — fires are universally *stale*-extremeness events. The fire bar sits well downstream of extremeness onset; any onset-timing information is upstream of where the detector samples it.

### Branch v18.A — Frame-Shift Mask *(build iff T1 PASS on ≥1 side)*

**Mechanism:** suppress fires while any contributing reference window recently rotated its extreme by expiry. Pure bookkeeping of our own windows — zero market parameters, not outcome-fitted.

- New Params (per side, additive): `use_frame_mask_{side}: bool = False`, `frame_mask_bars_{side}: int = 5` (searchable 2–10).
- Python: in the detector's signal condition, add `and not frame_recent[t]` when enabled, where `frame_recent` = OR over the agreement slice scales + price-gate window of `expiry_events` (exact semantics of Task A1) rolled over `frame_mask_bars`.
- Pine: per scale in the existing parity-shim loop, expiry rotation = `prev_holder_age == W-1 and not new_extreme` tracked with two `var` arrays (holder value + age) — same state-machine class as the scale cache block; price-gate window via `ta.highest`/`ta.barssince`.
- Export/audit: new dbg columns `dbg_frame_recent_{side}` (bool, exact match required), masked-signal parity = signals exact bool.

### Branch v18.B — Freshness-weighted agreement *(build iff T2 PASS on ≥1 side)*

**Mechanism:** count only freshly-extreme scales toward agreement: scale s contributes iff extreme now AND first crossed within `sync_k` bars. Habituated/staggered extremeness stops counting.

- New Params: `use_fresh_agreement_{side}: bool = False`, `sync_k_{side}: int = 3` (searchable 2–8).
- Python: `agreement_fresh = fresh_count / n_scales` (Task A1 `fresh` semantics) replaces `agreement` in the `ms_agree` comparison when enabled.
- Pine: per-scale crossing age = `var int age` updated in the shim loop; fresh iff `age <= sync_k`.
- Audit: `dbg_agreement_fresh_{side}` numeric 1e-6 + signals exact.

### Branch v18.C — Event-time PIR for scale_div *(build iff T3 PASS strongly AND A/B both FAIL — else defer to v19)*

**Mechanism:** `pir_detect` window anchored at the `evt_k`-th last confirmed structural pivot instead of `max(S_detect,20)` bars. Largest parity surface (pivot-state-dependent window), hence last resort / v19 candidate.

- New Params: `use_event_pir_{side}: bool = False`, `evt_k_{side}: int = 8` (searchable 4–16).

### ALL FAIL → no v18 detector change

The frame thesis is mechanically real but not what costs precision. Redirect per re-anchoring: Route A (Sobol in-space precision frontier — already planned), Route D (LOW productization). Record the negative result in the re-anchoring doc's evidence ledger as [SOLID].

### Invariants for ANY built branch (non-negotiable)

1. **Additive + default-off, but wired into the search from day one** (`v17_search` dims + workbook) — the "built but unwired" lesson is standing policy.
2. **Freeze protocol:** v18 re-opens `src/detector.py`/`src/indicators.py` under paired Pine+Python build. Trust is re-established by (a) golden byte-identity with all v18 switches OFF (additive proof, test pinned), (b) TV-export audit with switches ON (signals exact, dbg numerics 1e-6) on an SPX export BEFORE any optimizer result is trusted. Then the v18 surface re-freezes.
3. **Scorer unchanged (v5.1)** — detector version and objective era are orthogonal; run JSONs carry `detector: "v18"` alongside `scorer: "v5.1"`.
4. **Validation run design:** one GPU run, `shape_variants = {"v17_base": {}, "v18_<branch>": {use_…: True}, …}` both sides, winner by era_pass-first-then-deflated; HIGH remains MONITOR-ONLY unless its v18 variant clears break-even precision on the run's own TV export (heatmap method).
5. **Implementation plan for the surviving branch is written only after Part A outcomes** (separate doc, full TDD task breakdown per writing-plans).

---

## Self-review

- Spec coverage: thesis → T1 (amnesia), T2 (synchrony/habituation), T3 (event-time) — all three levers from the discussion have a test; every branch maps to exactly one PASS condition; ALL-FAIL handled.
- No placeholders: both scripts complete and runnable; decision rules numeric; v18 branches specify exact Params, semantics, Pine mechanism, audit columns. (Branch impl details deliberately deferred to post-outcome plan per YAGNI — stated explicitly, not hidden.)
- Type consistency: `expiry_events(x, W, mode)` and `fresh` semantics defined once in A1 and referenced by name in v18.A/B.
