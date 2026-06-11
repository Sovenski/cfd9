# Holdout→era_pass + Pruning Variants — Build Spec (v17.5 addendum)

**Status:** approved for implementation ("Do it", 2026-06-11)
**Scope:** runner/report-side only. NO search-objective change, NO parity
surface, NO frozen-file edits (src/detector.py, src/v17_fastdetector.py,
src/v17_gpu/**, src/parity.py, src/validation.py, src/universe.py,
src/search_space.py, src/speculatores145.py, src/indicators.py existing
functions, pine/*). EDITABLE: src/v17_runner.py, src/pooled_validation.py,
src/v17_acceptance.py, src/v17_card/**, src/scoring_v5.py (additive),
temp/ builders, tests/, h100_v17_gpu.ipynb via temp/build_h100_notebook.py.
GOLDEN: signal arrays byte-identical (nothing here touches detection);
fold scores unchanged (holdout is ADDITIVE reporting).

---

## A. Holdout evaluation → era_pass (true selection-untouched OOS)

### A1. Holdout region
`build_calendar_folds` reserves the final `holdout_fraction` (0.20) of the
common-era reference index; folds never touch it. Define per-stream holdout
slices:

- `holdout_start_ts` = the reference-index timestamp at `active` (the first
  reserved bar), **plus an EMBARGO of 200 bars** (EMBARGO_NEST_BARS) on the
  reference index — prevents label bleed from the last fold's OOS into the
  holdout (centered span labels reach ±200).
- Per stream: `df_holdout = df[ts >= holdout_embargo_ts]`, reset_index,
  `add_pivot_labels` per slice (standalone warm-up inside the slice, exactly
  like every fold slice — consistency rule). Drop streams with
  `len < MIN_STREAM_BARS` (401).
- NEW helper in `src/pooled_validation.py`:
  `build_holdout_slices(stream_datas, era_kw...) -> list[PreparedSlice-like]`
  reusing the existing slice construction (no copy-paste drift; factor or
  reuse `_run_fold_loop`'s slice-prep code path).

### A2. Holdout score
For winner Params per side: run the EXACT CPU `SpeculatorDetector` on each
holdout slice; compute the v5 weighted stream stats (`_stream_stat`) with
cluster weights; pool with the SAME composite used for a fold's OOS leg
(reuse `pooled_fold_score`'s OOS-side computation — implement as
`pooled_holdout_score(stats, side) -> (score, components)` next to it, NOT by
modifying fold scoring). Components must include `precision_w`, `recall_w`,
`n_signals`, `tp_mass`, `total_mass`.

### A3. era_pass rule (pre-committed, constants in code + tested)
```
era_pass = (holdout_score > 0.0)
           and (holdout_score >= ERA_PASS_MIN_RATIO * mean(raw winner fold scores))
ERA_PASS_MIN_RATIO = 0.5
```
Mean over the winner's RAW (unpenalized) informative fold scores — already
available via `raw_fold_scores`. Document: the 0.5 ratio is a generalization
floor (holdout may be one regime; demanding parity with the search mean would
be stricter than the fold-to-fold variance justifies).

### A4. Wiring + output
- `run_v17_gpu(..., holdout: bool = True)`: after the winner is chosen,
  evaluate holdout, set `era_pass` (overriding the None default UNLESS the
  caller passed an explicit era_pass), feed `summarize_acceptance`.
  PASS becomes reachable: no pins + bootstrap stable + holdout pass.
- `out["sides"][side]["holdout"] = {score, components, per_stream (top lines),
  era_pass, n_slices, holdout_start, embargo_bars, min_ratio}`.
- Cell-6 report: a HOLDOUT block per side (score vs fold-mean, ratio, pass,
  n_signals, per-stream tail).
- NEW `temp/holdout_posthoc.py <run_json> <data_dir>`: loads a saved run
  JSON's winner `best_params` + pool config, rebuilds the pool, evaluates the
  holdout for both sides, prints the same block — so the CURRENT big run
  (already in flight on the pre-change code) gets its holdout verdict
  post-hoc without a rerun.

### A5. Tests (TDD)
- Holdout slices start ≥ embargo after the last fold bar (no overlap with any
  fold slice; assert on a synthetic 2-stream pool).
- era_pass truth table: score 0 → False; ratio just-below/above 0.5 → F/T.
- `pooled_holdout_score` equals the OOS leg of `pooled_fold_score` on
  identical stats (pin by construction).
- Golden regression: fold scores and signal arrays unchanged with
  holdout=True (additive-only proof).

## B. Pruning / shape-variant outer loop

### B1. API
`run_v17_gpu(..., shape_variants: Optional[dict[str, dict]] = None)`.
Each entry: `name -> {Params field overrides}` applied to the seed via
`dataclasses.replace` BEFORE scorer construction. `None` → current behavior
(single unnamed variant "baseline"). For each variant, run the FULL per-side
pipeline (fresh GpuPooledScorer precompute, search, CPU re-score, gates,
holdout) sequentially. **Free GPU memory between variants** (drop scorer
refs + `torch.cuda.empty_cache()` — post-leak discipline; assert via the
memory estimator log per variant).

### B2. Output + selection
- `out["variants"][name]["sides"][side]` = the existing per-side dict.
- Top-level `out["sides"]` = the per-side BEST variant by: era_pass first,
  then deflated LCB (document the rule); `out["winner_variant"] = {side: name}`.
- Cell-6: a variant comparison table (per variant × side: raw/deflated LCB,
  verdict, holdout pass, n_signals) ABOVE the winner detail.

### B3. The data-driven pruning configs (workbook defaults)
Cell 5 gains `RUN_PRUNED = True` (checkbox). When set, passes:
```
shape_variants = {
  "baseline": {},
  "pruned":   {"use_momentum_velocity_high": False,
               "use_momentum_velocity_low":  False},
}
```
Rationale string in the form comment: mom-vel neutralized (thresh→~0) in two
independent v5 runs and never load-bearing in any era; this is the three-era
on/off question settled head-to-head. (Deeper prunes stay manual overrides.)
NOTE: with mom-vel off, HIGH has max_votes_high=0 → req clamps to ≥1 with
only the drift vote available — verify the engine handles max_votes=0
gracefully (existing clamp `max(max_votes,1)`); add a unit test for a side
with zero use_* votes (drift still votes; signals still possible).

### B4. Tests
- Two-variant run on the tiny synthetic pool: both variants complete, JSON
  shape correct, winner_variant rule respected, GPU memory freed between
  variants (estimator re-logs).
- Zero-active-votes side: detector + FastDetector + GPU path all agree
  (byte-identity on the tiny pool) and don't crash.
- Workbook regenerated + nbformat-valid.

## Acceptance
Full suite green; freeze audit clean; golden untouched; report
`plan/IMPLEMENTATION_REPORT_holdout_pruning.md` with the post-hoc usage line
for the in-flight big run.
