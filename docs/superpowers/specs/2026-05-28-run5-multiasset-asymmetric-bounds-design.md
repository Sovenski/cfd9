# Run 5 — Multi-Asset Pooled Optimization + Asymmetric Search Bounds

**Date:** 2026-05-28
**Status:** Design v2 — revised after 3-agent review (stats / engineering / consistency).
Pending final user read → implementation plan.
**Author:** kuben + Claude
**Build phasing (decided):** one combined effort toward a single Run 5.

## 1. Motivation

A 3-agent diagnostic study (`results/diag/SYNTHESIS.md`) established that the HIGH
(market-top) side fails for an **information / validation reason, not a feature void**:

- The top signal *exists* (multivariate in-sample CV-AUC ≈ 0.81 on real V15 features) but
  cannot be *validated*: only **27 structural HIGH pivots in 155 years**, **10/14
  walk-forward OOS folds with ≤1 HIGH pivot** (2 contain zero), only **4 HIGH pivots in
  holdout**. The objective is a stack of near-binary coin-flips → no stable gradient →
  noise floor. Run 4's 1000 trials confirmed more search cannot help.
- Secondary real asymmetry: tops are a slow *process* (dim agreement, median 0.56),
  bottoms a sharp *event* (0.93, capitulation volume). Volume adds nothing to HIGH.
  `pivot_drift` is the only top-favorable feature. Era/volume confound is symmetric.

Two levers, both covered here: (1) raise the validatable event count via multi-asset /
multi-timeframe pooling; (2) let HIGH express its own character by decoupling the search
bounds and removing two clamps — no new features.

## 2. Goals / Non-Goals

**Goals**
- Side-specific search spaces (`HIGH_SPACE` / `LOW_SPACE`); HIGH gets two relaxed floors.
- Configurable multi-asset, multi-timeframe universe selectable from the notebook
  (timeframe selection symmetric for both sides — see §4.1).
- Time-aligned, **correlation-aware** pooled-fold objective that multiplies
  events-per-fold while preserving walk-forward hygiene.
- Re-optimize **both** sides on the pooled substrate (Run 5).

**Non-Goals (deferred)**
- No new detector features.
- **No change to `detector.py` logic → Pine ↔ Python parity is untouched.** Multi-asset
  lives in data loading + objective/scoring only.
- No true multi-timeframe *confluence* detector (HTF+LTF in one detector) — separate
  future milestone (Pine `request.security` rewrite).
- LOW search **bounds** are not widened (frozen as today). (LOW *data* changes via pooling.)
- Stream-level artifact caching is out of scope (would touch the detector interface →
  parity risk); we keep per-fold-slice artifact builds (§4.4).

## 3. Objective 1 — Decouple + extend the search bounds

**Current state:** `params_from_trial(trial, side)` (`src/speculatores145.py:382`) uses
identical literal ranges regardless of `side`; symmetry is hard-coded into the search.

**Design:**
- New module **`src/search_space.py`** (≤200 lines): a frozen `SearchSpace` dataclass —
  one entry per tunable parameter holding a numeric `(low, high)` + type, or a categorical
  option list. Two instances: `HIGH_SPACE`, `LOW_SPACE`.
- `params_from_trial(trial, side)` selects `HIGH_SPACE if side=="high" else LOW_SPACE`
  and drives every `trial.suggest_*` from it.
- **Naming contract (review fix, MAJOR):** Optuna trial keys are UNCHANGED, including the
  existing short-form keys that differ from `Params` field names — notably
  `{s}_pivot_drift_lb` (trial key) ↔ `pivot_drift_lookback_{side}` (Params field).
  `SearchSpace` records both the trial-key and the Params-field name per parameter so
  journals/TPE warm-start still parse. A regression test asserts the resolved trial keys
  are byte-identical to today's.
- **Categoricals frozen (review fix):** `HIGH_SPACE` categorical option lists are identical
  (same order) to `LOW_SPACE`; only numeric ranges may differ — else TPE misreads journals.
- `LOW_SPACE` reproduces today's literals exactly (regression-tested).
- `HIGH_SPACE` = today's literals **except** two floors:
  - `dur_extreme_pct`: `0.50 → 0.30` (duration counter can accumulate on dim top agreement)
  - `pct_extreme`: `0.70 → 0.55` (optimizer can densify top agreement)
- **Stability-probe consistency (review fix):** `_mutate_local_param` and
  `_sample_global_params` must consult `HIGH_SPACE`/`LOW_SPACE` per side, not the shared
  `INT_BOUNDS`/`FLOAT_BOUNDS` — otherwise HIGH restarts ignore the relaxed floors.

## 4. Objective 2 — Multi-asset / multi-timeframe pooled optimization

### 4.1 Data & universe layer
- TV CSV exports (clean volume) in `data/raw/`, named `{TICKER}_{TF}_{start}_{end}.csv`.
  Export list delivered once spec is locked.
- **Universe registry** in new **`src/universe.py`** (plain Python dict, mirroring the
  existing `INSTRUMENTS` dict — no YAML dependency): named groups → ticker lists
  (`INDICES_US`, `INDICES_GLOBAL`, `STOCKS_MEGACAP`, `COMMODITIES`, …) + a `TIMEFRAMES`
  list. Pool unit = an **`(asset, timeframe)` stream**.
- **Notebook selectors (dropdowns):** choose asset group(s) AND timeframe(s) — single /
  some / all / mixed. **Timeframe selection is symmetric for both sides, decided per run,
  with no built-in default** (user choice; HIGH and LOW each get the selected streams).
- **Loader:** reads each stream via `load_data`, applies `add_pivot_labels` (v4 oracle,
  **nest stays [50,100,200] bars on every timeframe** — the scale-invariance assumption,
  gated by §5.3), computes labels on the FULL stream then date-slices (avoids edge
  artifacts), and builds detector artifacts per fold-slice (§4.4).
- **Volume:** used as-is (clean from TV).

### 4.2 Time-aligned pooled folds (calendar-based) — core interface change
**This replaces the row-index fold interface; it is not additive (review BLOCKER).** The
single-asset path must keep working (thin adapter wrapping one stream).
- New entry point: `build_pooled_optuna_objective(streams, params_from_trial, side)
  -> Callable[[Trial], float]`, where `streams` is a list of `(stream_id, df, cluster_id)`.
  Touch points that change shape: `walk_forward_folds`, `prepare_walk_forward_folds`,
  `build_optuna_objective`, `_evaluate_with_components`.
- Folds defined on a **master calendar** (date ranges), mirroring today's fractions in
  time: holdout = last 20% of master span; IS ≈ 10%, OOS ≈ 3%, step ≈ 5%; **≈ 14 folds.**
- **Embargo (review fix, MAJOR):** ≥ `max(nest) = 200` bars per stream (NOT 20), expressed
  as a calendar gap per timeframe, so the non-causal label window `[i−200, i+200]` cannot
  leak future prices across the IS/OOS boundary.
- For each fold window, **slice every selected stream** to that window.
- **Minimum-bars gate (review BLOCKER):** a stream contributes to a fold only if its slice
  has ≥ `2·max(nest)+1 = 401` bars; otherwise it is dropped from that fold with a logged
  warning (prevents 1W-in-short-OOS from silently contributing 0 pivots and biasing the
  excess penalty). Streams that don't exist yet contribute nothing (no padding).

### 4.3 Pooled, correlation-aware per-fold scoring
1. Per stream in the fold: run the per-asset detector → `(signals, structural_pivots)` at
   the single tolerance scale `REFERENCE_N=100` (v4 uses one scale, so the cross-stream
   scale-set mismatch reviewers feared is avoided; assert single-scale).
2. **Hungarian match per stream** → per-stream TP/FP/FN.
3. **Correlation-aware pooling (review BLOCKER):** weight each stream's TP/FP/FN by
   `w_k = 1 / cluster_size_k` (cluster = same underlying/economy; e.g. SPX-1D, SPX-1W,
   SPX-1H share a cluster; SPX/NDX/DJIA may share a US-equity cluster — cluster map lives
   in `universe.py`) **before** summing into the pooled fold precision/recall. This makes
   the correction actually enter the objective, not just a report.
4. Aggregate folds with the bootstrap-CI lower bound, using **cluster-block resampling**
   (resample at the cluster level, not the individual fold-contribution level);
   `block_len` re-derived for pooled folds (documented, not the inherited `2`).
- Report an **effective-event count** per fold (raw ÷ correlation factor) alongside raw.

### 4.4 Compute budget (review MAJOR)
- `build_detector_artifacts` builds a PIR matrix to scale 500 (~150 MB for a 38k-bar
  stream) + GJR/HAR, per fold-slice. We keep **per-fold-slice** builds (parity-safe; no
  detector change) and **bound the pool size per run** so peak memory stays within Colab
  (~12 GB). The notebook surfaces an estimated stream-count × fold-count cost before launch
  and warns past a threshold. Stream-level artifact reuse is explicitly out of scope.

## 5. Statistical integrity, reproducibility, testing

### 5.1 Correlated streams — enters the objective
See §4.3: per-stream `1/cluster_size` weighting into the pooled fold score + cluster-block
bootstrap. The effective-event count is additionally reported. This is the mechanism that
prevents the LCB from over-crediting correlated duplicates (the review's top concern).

### 5.2 Reproducibility + holdout governance (decided: pre-register + pre/post)
- New seeded Optuna study `spec_v15_run5_multiasset_<side>` (a *new* study, not merged with
  single-asset journals); storage via `monitor145.make_storage`.
- Run report records: selected universe + timeframes, per-stream date ranges + data-file
  hashes, cluster map, resolved `HIGH_SPACE`/`LOW_SPACE` bounds.
- **Pre-registration (review MAJOR):** before the run, write the expected direction of each
  relaxed HIGH floor (`dur_extreme_pct↓`, `pct_extreme↓`) and the hypotheses, into the run
  report. **Report exact holdout HIGH/LOW hit counts before vs after** (not "improvement"),
  acknowledging the post-2000 famous tops are public and were visually inspected in Runs 1–4.

### 5.3 Scale-invariance falsification gate (review MAJOR) — required before any mixed-TF run
Optimize HIGH on **SPX-1D only**, then apply the resulting params to **SPX-1W**. If
precision drops > 0.15 absolute, scale-invariance is rejected and timeframes must NOT be
mixed in one run (fall back to single-timeframe pools). This gate runs before trusting any
mixed-timeframe result.

### 5.4 Testing (TDD where it counts)
1. `SearchSpace`: `LOW_SPACE` reproduces today's exact ranges AND resolved trial keys are
   byte-identical to today's (incl. `pivot_drift_lb`); `HIGH_SPACE` differs only in the two
   floors; categorical option lists identical to LOW.
2. Calendar folds: synthetic 2-stream test (different start dates, different timeframes) →
   non-overlapping IS/OOS date windows, ≥200-bar embargo respected, min-bars gate drops a
   too-short stream, last fold ends before holdout. Single-asset adapter reproduces the
   current row-index folds within rounding.
3. Pooled scoring: toy streams with known TP/FP/FN and a known cluster map aggregate to the
   correct `1/cluster_size`-weighted pooled precision/recall.
4. **Sprint-0 acceptance test (first deliverable, not a precondition):** end-to-end pooled
   run on SPX-1D + DAX-1D (data already on disk) proving the pooled objective works on two
   real streams before the full universe export.

### 5.5 Parity coverage for new assets (review MINOR)
Each new asset stream gets parity coverage only if a matching TV enriched export exists in
`data/enriched/` (`find_matching_enriched_export`). Adding a `data/enriched/{TICKER}_TV_*`
export per new stream is a delivery requirement, tracked in the plan.

## 6. Open items for implementation planning
- Exact calendar durations for IS/OOS/step + the pooled `block_len` value (derive in plan).
- `SearchSpace` field schema (trial-key, Params-field, type, range/options).
- Cluster map contents in `universe.py` (which assets share a cluster).
- Notebook cost-estimate threshold for the pre-launch memory warning.
- Exact TV export list (tickers × timeframes × naming) — delivered to user when plan starts.

## Appendix A — Review resolution log (2026-05-28, 3-agent review)
- BLOCKER (stats/eng/consistency): cluster correction must enter objective → §4.3 weighting + cluster-block bootstrap.
- BLOCKER (eng): calendar-fold is a core interface replacement → §4.2 explicit new signature, single-asset adapter, multi-day effort acknowledged.
- BLOCKER (stats): short folds × 401-bar nest → §4.2 minimum-bars gate.
- MAJOR (stats): non-causal label leakage → §4.2 embargo ≥ 200 bars + full-stream labelling.
- MAJOR (stats): scale-invariance untested → §5.3 falsification gate.
- MAJOR (stats): holdout contamination → §5.2 pre-registration + pre/post reporting.
- MAJOR (eng): Colab compute/memory → §4.4 budget + bounded pool + per-fold-slice builds.
- MAJOR (consistency): `pivot_drift_lb` naming + frozen categoricals + stability-probe bounds → §3.
- MINOR: module placement (`src/search_space.py`, `src/universe.py`), parity fixtures per asset, smoke = sprint-0 → §3/§4.1/§5.4/§5.5.
- Decisions: 1W-in-LOW = per-run dropdown, no default (§4.1); phasing = one combined effort; governance = pre-register + pre/post (§5.2).
