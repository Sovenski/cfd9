# Speculatores v16 — Multi-Asset Pooled Optimizer + Asymmetric Search Bounds

**Branch:** `feature/run5-multiasset-asymmetric-bounds`
**Status:** stable / ready to run. Tests 31/31 green; Pine↔Python parity preserved (`src/detector.py` byte-unchanged); single-asset path untouched.

## Why this release exists

A 3-agent diagnostic (`results/diag/SYNTHESIS.md`) showed the HIGH (market-top) side
fails for an **information limit**, not a feature void: only 27 structural HIGH pivots in
155 years, 10/14 walk-forward OOS folds with ≤1 HIGH event → the objective is a stack of
coin-flips with no gradient. v16 attacks that two ways, with **no new detector features**
and **no detector changes** (so the Pine indicator stays in parity):

1. **Decoupled search bounds** — HIGH and LOW get independent Optuna ranges; two HIGH
   floors are relaxed (`dur_extreme_pct` 0.50→0.30, `pct_extreme` 0.70→0.55) so the
   optimizer can express a dim-agreement "top" detector. (`src/search_space.py`)
2. **Multi-asset / multi-timeframe pooling** — pool `(asset, timeframe)` streams into
   time-aligned, bar-sized walk-forward folds, scoring each fold on cluster-weighted
   pooled match counts. Pooling multiplies validatable events per fold (e.g. SPX+DAX
   roughly doubles structural pivots: 27→47 HIGH, 35→58 LOW).

## How to run (Colab)

1. Export TV daily CSVs into `data/raw/` as `{TICKER}_{TF}_{start}_{end}.csv`
   (e.g. `SPX_1D_*.csv`, `NDX_1D_*.csv`, `DAX_1D_*.csv`, …). The loader auto-lowercases
   columns and parses the Unix `time` column.
2. Open `optimize.ipynb`. The Run 5 cells default to:
   - `SELECTED_GROUPS = ["INDICES_US", "INDICES_GLOBAL"]`  (SPX, NDX, DJI, RUT, DAX, NI225, UKX)
   - `SELECTED_TIMEFRAMES = ["1D"]`
   - `VOLUME_POLICY = "price_only"`  (robust default; volume adds nothing to HIGH, and raw
     index daily volume is unreliable — use `volume_required` only for LOW-side volume
     experiments)
   - `N_TRIALS = 250`  (per side)
3. **Run the pool-sufficiency diagnostic cell FIRST.** It prints resolved streams, fold
   count, and structural-pivots-per-OOS-fold per side, with a rule of thumb (aim for
   ≥~5 informative folds, mean ≥~3 pivots/fold). If it warns the pool is thin, add more
   index exports before launching — a thin pool wastes the run.
4. Run the launch cell (250 trials/side, seeded from the heuristic-structural config).
5. Run the reproducibility-report cell to record universe, volume policy + per-stream
   quality tags, date ranges, data hashes, and resolved bounds.

## Design guarantees & honest caveats

- **Bar-sized folds.** Fold windows are sized in bars off the longest (reference) stream
  and mapped to dates, then every stream is sliced by date — so pools that aren't anchored
  on the 155-year SPX series still form folds (a calendar-day-sizing bug that crashed such
  pools was fixed pre-release).
- **No label leakage.** A 200-bar embargo (= `max(STRUCTURAL_NEST)`) separates IS/OOS, and
  pivot labels are computed per-slice; a 401-bar (`2·max(nest)+1`) minimum gate drops
  slices too short to label any structural pivot.
- **Correlation-aware.** Streams sharing a cluster (e.g. SPX/NDX/DJI = `US_EQ`,
  SPX-1D/SPX-1W) are weighted `1/cluster_size` so correlated duplicates don't over-credit
  the bootstrap-LCB objective.
- **Informative folds only.** Folds whose OOS contains zero structural pivots of the side
  are excluded (they'd be forced zeros that only dilute the LCB).
- **Objective magnitudes.** Per-side LCBs in the ~0.05–0.20 range are normal for this
  scorer (Run 4 LOW was 0.097); they are precision×recall-saturation×penalty products on
  rare events, not accuracies.
- **Mixed timeframes** are supported but are a *scale-invariance bet* (the nest is in bars,
  so a 1W pivot is a multi-year event vs a 1D macro top). `src/scale_invariance.py`
  provides a falsification gate (optimize on 1D, apply to 1W; >0.15 precision drop ⇒ don't
  mix). Default is 1D-only.
- **Universe vs data.** `src/universe.py` lists the intended index set; `resolve_streams`
  silently skips tickers/timeframes with no file on disk (logged). With only SPX_1D and
  DAX_1D present, the "indices 1D" default resolves to a 2-stream pool — the diagnostic
  cell makes this explicit.

## What's NOT in scope (deferred)

- True multi-timeframe *confluence* (one detector reading HTF+LTF together) — a separate
  milestone with a Pine `request.security` rewrite.
- New detector features for tops (the asymmetry lever) — revisit once multi-asset event
  supply makes top-specific configs validatable.

## Spec & plans
- Spec: `docs/superpowers/specs/2026-05-28-run5-multiasset-asymmetric-bounds-design.md`
- Plans: `docs/superpowers/plans/2026-05-28-run5-part1-searchspace-decoupling.md`,
  `…-part2-multiasset.md`
- Diagnostic that motivated it: `results/diag/SYNTHESIS.md`
