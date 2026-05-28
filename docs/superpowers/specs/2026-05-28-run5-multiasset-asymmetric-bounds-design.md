# Run 5 — Multi-Asset Pooled Optimization + Asymmetric Search Bounds

**Date:** 2026-05-28
**Status:** Design approved (sections 1–4), pending spec review → implementation plan
**Author:** kuben + Claude

## 1. Motivation

A 3-agent diagnostic study (`results/diag/SYNTHESIS.md`) established that the HIGH
(market-top) side of the Speculatores pivot detector fails for an **information /
validation reason, not a feature void**:

- The top signal *exists* (multivariate in-sample CV-AUC ≈ 0.81 on real V15 features)
  but cannot be *validated*: only **27 structural HIGH pivots in 155 years**, with
  **10/14 walk-forward OOS folds containing ≤1 HIGH pivot** (2 contain zero) and only
  **4 HIGH pivots in the holdout**. The objective Optuna maximizes is therefore a stack
  of near-binary coin-flips with no stable gradient → it wanders to the noise floor.
  More Optuna trials cannot help (confirmed: Run 4's 1000 trials still hit noise floor).
- Secondary, real asymmetry: tops are a slow *process* (dim, diffuse multi-scale
  agreement, median 0.56), bottoms a sharp *event* (agreement 0.93, capitulation volume,
  V-shaped recovery). Volume adds nothing to the HIGH side. `pivot_drift` is the only
  top-favorable feature. Era/volume confound is symmetric and NOT the driver.

Two levers follow, and this spec covers both:

1. **Raise the validatable event count** via multi-asset (and multi-timeframe) pooling —
   the highest-leverage move, because the bottleneck is events-per-fold, not search.
2. **Let HIGH express its own character** by decoupling the HIGH/LOW search bounds and
   removing two clamps — without adding any new features.

## 2. Goals / Non-Goals

**Goals**
- Side-specific search spaces (`HIGH_SPACE` / `LOW_SPACE`); HIGH gets two relaxed floors.
- A configurable multi-asset, multi-timeframe data universe selectable from the notebook.
- A time-aligned, pooled-fold objective that multiplies events-per-fold while preserving
  walk-forward hygiene.
- Re-optimize **both** sides on the pooled substrate (Run 5).

**Non-Goals (explicitly deferred)**
- No new detector features.
- No change to `detector.py` logic → **Pine ↔ Python parity contract is untouched.**
  Multi-asset lives entirely in data loading + objective/scoring.
- No true multi-timeframe *confluence* detector (one detector reading HTF+LTF together).
  That is a separate future milestone with a Pine-rewrite cost (`request.security`).
- LOW search bounds are NOT widened (its ranges are frozen as today).

## 3. Objective 1 — Decouple + extend the search bounds

**Current state:** `params_from_trial(trial, side)` (`src/speculatores145.py:382`) uses
identical literal ranges regardless of `side`; the `{side}_` prefix only namespaces the
sampled value. Symmetry is hard-coded into the search.

**Design:**
- New frozen `SearchSpace` dataclass: one entry per tunable parameter, holding either a
  numeric `(low, high)` range (with int/float type) or a categorical option list.
- Two instances: `HIGH_SPACE`, `LOW_SPACE`.
- `params_from_trial(trial, side)` selects `HIGH_SPACE if side == "high" else LOW_SPACE`
  and drives every `trial.suggest_*` call from it. **Parameter names are unchanged**
  (`{side}_dur_extreme_pct`, …) so existing study/journal key schemas still parse.
- `LOW_SPACE` reproduces today's literals **exactly** (regression-tested; the working
  bottom detector and its reproducibility are preserved).
- `HIGH_SPACE` = today's literals **except**:
  - `dur_extreme_pct`: floor `0.50 → 0.30` (duration counter can accumulate on dim top
    agreement instead of being structurally unsatisfiable).
  - `pct_extreme`: floor `0.70 → 0.55` (optimizer can densify top agreement).
- No other HIGH bound changes: `min_duration` floor is already 1, `use_*` votes are
  already free booleans, scales are already wide. HIGH can already reach a top-shaped
  config; we only remove the two clamps and rely on the (now measurable) objective to
  prefer it.

**Rationale for the dataclass:** keeps the two spaces side-by-side and diff-able, makes
"what is different about HIGH" explicit and testable, minimal blast radius (one function).

## 4. Objective 2 — Multi-asset / multi-timeframe pooled optimization

### 4.1 Data & universe layer
- TV CSV exports (clean volume) dropped into `data/raw/`, named
  `{TICKER}_{TF}_{start}_{end}.csv` (existing convention). Export list provided once spec
  is locked.
- **Universe registry** (config/module): named groups → ticker lists
  (e.g. `INDICES_US`, `INDICES_GLOBAL`, `STOCKS_MEGACAP`, `COMMODITIES`) plus a
  `TIMEFRAMES` list. The pool unit is an **`(asset, timeframe)` stream**.
- **Notebook selectors (dropdowns):** choose asset group(s) AND timeframe(s) — single /
  some / all / mixed. The run pools every selected `(asset, tf)` stream. Mixing timeframes
  in one run is supported (scale-invariance bet) and so is restricting to one.
- **Loader:** reads each selected stream via existing `load_data`, applies
  `add_pivot_labels` (v4 oracle, **nest stays [50,100,200] *bars* on every timeframe** —
  this IS the scale-invariance assumption), and caches `build_detector_artifacts`
  once per stream (artifacts are param-invariant → reused across all trials; cached to the
  Drive-mounted repo on Colab).
- **Volume:** used as-is (clean from TV); volume-dependent votes behave normally.

### 4.2 Time-aligned pooled folds (calendar-based)
- Folds defined on a **master calendar** (date ranges), mirroring today's fractions but in
  time: holdout = last 20% of the master span; IS ≈ 10%, OOS ≈ 3%, step ≈ 5%,
  embargo = a fixed **calendar** gap (~4 weeks ≈ 20 trading days) applied to the date
  window so it is unit-consistent across mixed timeframes, ≈ 14 folds.
- For each fold's date window, **slice every selected stream** to that window. Streams that
  don't exist yet (e.g. NDX pre-1985) contribute nothing to that fold — no padding, no
  leakage.

### 4.3 Pooled per-fold scoring
1. Per `(asset, tf)` stream in the fold: run the per-asset detector → `(signals,
   structural_pivots)`.
2. **Hungarian match per stream** (no cross-asset matching) → per-stream TP/FP/FN.
3. **Pool counts across all streams** in the fold → pooled precision/recall/side-score.
   A fold now rests on (sum over streams) events instead of ~1.
4. Aggregate folds with the existing **bootstrap-CI lower-bound** objective (unchanged
   contract, richer folds). Optuna still sees a single scalar per side.

**Code touch points:** `validation.py` gains calendar-fold construction + a pooled-scoring
loop over streams; `scoring.py` per-series matching reused per stream with a thin
aggregation wrapper; `detector.py` untouched.

## 5. Statistical caveats, reproducibility, testing

### 5.1 Correlated streams (must handle honestly)
SPX-1D vs SPX-1W (same underlying), and SPX/NDX/DJIA (same economy), are NOT independent.
Naive pooling inflates apparent events without proportional information, and the bootstrap
assumes some independence.
- **Block/cluster bootstrap by underlying:** resample at the asset-cluster level, not the
  individual-event level, so the LCB does not over-credit correlated duplicates.
- **Report an effective-event count** per fold (raw events ÷ rough correlation factor)
  next to the raw count, so we never mistake 4 correlated indices for 4× information.
- Implication: independence is what we are really buying → global indices + commodities
  beat US-only pools for the stated purpose.

### 5.2 Reproducibility
- New seeded Optuna study (e.g. `spec_v15_run5_multiasset_<side>`); a *new* study, not
  merged with single-asset journals.
- Run report records: selected universe + timeframes, per-stream date ranges, data file
  hashes, and the resolved `HIGH_SPACE`/`LOW_SPACE` bounds.

### 5.3 Testing (TDD where it counts)
1. `SearchSpace` refactor: assert `LOW_SPACE` reproduces today's exact ranges; `HIGH_SPACE`
   differs *only* in the two floors.
2. Calendar-fold construction: synthetic 2-stream test with different start dates →
   non-overlapping IS/OOS windows, embargo respected, last fold ends before holdout.
3. Pooled scoring: two toy streams with known TP/FP/FN aggregate to correct pooled
   precision/recall.
4. End-to-end smoke run on SPX-1D + DAX-1D (data already on disk) before the full universe
   export.

## 6. Open items for implementation planning
- Exact calendar durations for IS/OOS/step (derive from current fractions; confirm in plan).
- `SearchSpace` representation details (dataclass field types, categorical handling).
- Universe registry format (Python dict vs YAML) — follow existing config conventions.
- Artifact cache key (stream id + data hash) and Colab cache location.
- Exact TV export list (tickers × timeframes) — delivered to user once plan starts.
