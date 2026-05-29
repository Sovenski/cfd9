# v17 — Calibrated Coordinate-Ascent (CCA): a non-search optimizer for the pivot detector

**Status:** design → implementation (branch `feature/v17-calibrated-coordinate-ascent`)
**Date:** 2026-05-29
**Parity:** `src/detector.py` and `src/indicators.py` stay **byte-identical**. v17 only *produces* `Params` constants and *scores* them with the existing pipeline. Nothing v17 emits is non-Pine-portable.

---

## 1. Why v17 exists (the problem with v16)

v16 = black-box derivative-free search (Optuna TPE/CMA-ES/GP) over ~35 mixed params, scored by a noisy block-bootstrap LCB on ~100 validatable events. It is slow (13–54 s/trial × 500), noisy (CMA-ES stuck at 0), and information-limited. A multi-agent research sweep (run `wf_c35e82d2-79f`, 42 agents) that **explicitly disqualified "use a different sampler"** converged on one family:

> **Precompute features once → seed each vote threshold by an exact ROC/PR breakpoint scan against the labels (using all ~25k bars as negatives) → refine with cyclic coordinate-ascent (exact 1-D line search per threshold) against the real objective → enumerate the discrete architecture. Select cutpoints under purged CV for robustness.**

Top verified winners (all the same paradigm): *exact cutpoint coordinate-ascent over sorted breakpoints*; *per-vote exact threshold scan + greedy vote-count calibration via PR/ROC convex hull, decomposed by the AND-gate*; *Neyman–Pearson cutpoint calibration*; *STreeD optimal sparse tree (pystreed)* and *subgroup discovery (pysubgroup)* as cross-checks; *Learn-then-Test / Pareto Testing* for risk-controlled final selection.

**Why it's fundamentally different from v16:** it is not sampling a black box. It *reads off* thresholds from the label geometry (exact, global per axis) and uses the **huge negative class** (~25k bars) to position each cut — sidestepping the ~100-positive noise wall that caps the LCB. The only thing "searched" is the small discrete architecture, which is *enumerated*.

---

## 2. Detector facts v17 exploits (confirmed from `src/detector.py`)

The detector is a **per-bar boolean function**: Phase 1 builds per-bar feature arrays; each *vote* is `feature (cmp) threshold`; Phase 2 assembles them via an AND-gate + `vote_count ≥ confirm_count` + cooldown (stateful).

Params split into three roles:

| Role | Params | v17 treatment |
|---|---|---|
| **Shape / integer** (reshape feature arrays) | `S_detect`, `scale_start/end/step`, `pct_extreme`*, `vola_range_len`, `er_period`, `price_gate_lb`, `pivot_drift_lookback`, `baseline_lb`, `min_duration`, `cooldown_bars`, `edge_window` | **Enumerate** a small grid (seeded by v16 best) |
| **Continuous threshold** (pure final comparator) | `min_agreement`, `scale_div_thresh`, `slope_thresh`, `vol_surge_thresh`, `vola_high_pct`, `gjr_vote_thresh`, `har_vote_thresh`, `momentum_velocity_thresh` | **Calibrate** by exact label-aware breakpoint scan, then coordinate-ascent |
| **Stateful threshold** (feeds duration/drift state) | `dur_extreme_pct`, `pivot_drift_thresh`, `pivot_drift_gate_mult` | **Coordinate-ascent only** (no closed form — depends on Phase-2 state) |
| **Architecture / categorical** | `use_trend/volume/momentum/momentum_velocity/volatility/gjr_asym/har_vol`, `confirm_count`, `vola_method`, `momentum_velocity_mode`, `er_directional`, `use_er_gate`, `use_edge_voting`, `pivot_drift_confirm_bias` | **Enumerate** (small, sensible set incl. v16 best) |

\* `pct_extreme` feeds `calc_agreement_fast`, so it is a *shape* param (changes the agreement array), not a pure comparator.

**Per-vote comparator directions** (for the calibration seed; `+side` = HIGH):
- `trend`: HIGH `slope_val>θ & linreg_norm>θ`; LOW `<-θ & <-θ`
- `volume`: HIGH `vol_surge < 1/θ`; LOW `vol_surge > θ`
- `momentum` (free-of-θ boolean `mom_div<0`) — not threshold-calibrated
- `momentum_velocity`: HIGH `mom_vel ≤ -θ` (Reversal) / `≥ θ` (Trend); LOW mirror
- `volatility`: `pir(raw) > θ`
- `gjr`: HIGH `gjr_norm ≤ -θ`; LOW `gjr_norm ≥ θ`
- `har`: `har_norm ≥ θ` (both)
- gate `min_agreement`: `agr ≥ θ`; `scale_div`: `|scale_div| > θ` (gate wants NOT-flag)

**Labels:** `add_pivot_labels(df)` writes `pivot_N100 ∈ {+1 HIGH, −1 LOW, 0}`. Scoring: `precision_at_n_stats(signals, pivots, side, n=100)` → `tp/matched_pivots/total_pivots/precision`. Pooled objective unchanged (`src/pooled_validation.py`, `src/pooled_scoring.py`).

---

## 3. Algorithm

```
INPUT: stream pool (asset×tf frames with pivot labels), side ∈ {high, low}
for each ARCHITECTURE a in enumerated set (seeded by v16 best):
  for each SHAPE s in enumerated grid (seeded by v16 best):
     # 1. PRECOMPUTE (once per (stream, s)) — reuse detector Phase-1 feature arrays
     feats = compute_vote_features(stream, s)            # cached by build_detector_artifacts
     # 2. SEED thresholds: exact label-aware breakpoint scan per active vote
     theta0 = {v: best_cutpoint(feats[v], labels, direction[v], obj=Fbeta|Youden)  # purged-CV
               for v in active_votes(a)}
     # 3. REFINE: cyclic coordinate-ascent on the REAL objective
     params = assemble_params(a, s, theta0, seed=V16_BEST)
     repeat until no improvement (or budget):
        for θ in continuous+stateful thresholds:
           candidates = sorted_unique_breakpoints(feats or grid for θ)   # exact 1-D
           θ* = argmax over candidates of POOLED_SCORE(params with θ)     # existing scorer
           params[θ] = θ*
     score = pooled LCB(params)            # existing build_pooled_optuna_objective scorer
  keep best (a, s, params, score)
OUTPUT: best Params per side + pooled LCB + provenance
```

- **Step 2** is the novel core: O(#breakpoints) exact scan, label-driven, all-bars-as-negatives. Pure numpy.
- **Step 3** reuses the **real `SpeculatorDetector` + pooled scorer** as the evaluator → zero parity risk (we never reimplement the detector). Speed comes from (a) coordinate-ascent converging in ~dozens of 1-D steps vs 500 blind trials, (b) the ROC seed starting near-optimal, (c) `build_detector_artifacts` caching the heavy Phase-1 precompute (already fold-invariant).
- **Robustness:** cutpoint chosen to maximize the **purged-CV** objective, not in-sample; optional bootstrap-stability average. Directly attacks the ~100-event overfit.

---

## 4. New modules (parity-safe; detector untouched)

| File | Responsibility | ~LOC |
|---|---|---|
| `src/v17_features.py` | Given `(df, Params)`, return per-vote feature arrays + comparator directions, by reusing `src/indicators.py` (same math as detector Phase 1). | 250 |
| `src/v17_calibrate.py` | **Novel core.** Exact label-aware breakpoint cutpoint scan (Youden-J / F-β), per-vote direction, purged-CV selection. Pure numpy, no pipeline deps. | 220 |
| `src/v17_optimize.py` | Cyclic coordinate-ascent over continuous+stateful thresholds using the existing `SpeculatorDetector`+pooled scorer; architecture/shape enumeration. | 320 |
| `src/v17_runner.py` | Orchestrate on a stream pool, emit best `Params` per side + pooled LCB + provenance JSON. CLI + importable. | 200 |
| `tests/test_v17_calibrate.py` | Unit-test the cutpoint math on synthetic separable data + CV. | 120 |
| `tests/test_v17_features.py` | Assert v17 feature arrays equal the detector's internal Phase-1 arrays (parity of the precompute). | 100 |

Notebook: add a **`OPTIMIZER = optuna | v17`** knob to the v16 launch cell (later) — out of scope for the MVP commit.

---

## 5. Validation & smoke test

1. **Feature parity test:** v17 feature arrays == detector Phase-1 arrays (via `include_debug_columns`) on a real slice. Guarantees calibration sees exactly what the detector votes on.
2. **Calibration unit test:** on synthetic data with a known separating threshold, `best_cutpoint` recovers it; CV variant doesn't overfit injected noise.
3. **Smoke (real):** run v17 on a tiny pool (e.g. SPX+NDX+DAX 1D, common-era) for LOW side; assert it emits a valid `Params`, the detector runs, and the pooled LCB is **≥ the v16/v16.1 LOW baseline** (or at least non-degenerate). Wall-clock target: minutes, not hours.
4. **Head-to-head:** score v17's chosen params with the *exact* `build_pooled_optuna_objective` used by 16/16.1 → directly comparable LCB.

---

## 6. Parity argument

v17 never edits `detector.py`/`indicators.py`. It (a) recomputes the same features via the same `indicators` functions purely to *calibrate* thresholds, and (b) evaluates every candidate by calling the real `SpeculatorDetector`. Output is a `Params` of plain numbers + categorical choices — identical in kind to a v16 preset — so the Pine port is unchanged. The feature-parity test (5.1) is the guardrail.

---

## 7. Risks & fallbacks

- **Vote interaction / sequential state** (cooldown, duration, drift): per-vote seeds are *approximate*; coordinate-ascent on the true objective corrects the interactions. Stateful thresholds (`dur_extreme_pct`, `pivot_drift_*`) skip the closed form and are handled only by ascent.
- **~100-event overfitting:** mitigated by purged-CV cutpoint selection + scoring on the same held-out folds as v16; report the true LCB, never the in-sample cut quality.
- **Coordinate-ascent local optima:** multi-start from {v16 best, ROC seed, robust-quantile seed}; keep best.
- **Information ceiling unchanged:** v17 extracts more of the signal that exists and converges far faster/cheaper; it does **not** manufacture HIGH-side signal beyond the event limit. Honest expectation: faster + more trustworthy LOW, and a fair, cheap shot at HIGH — not a miracle.
- **Account rate-limit (today):** build is incremental & committed per-module so partial progress is never lost.
