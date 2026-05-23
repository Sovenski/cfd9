# Scorer v3 Plan — Address round-2 refinement findings

**Goal:** Address the 9 refinement-level concerns surfaced in the second unprimed-panel review of Scorer v2. Constraint unchanged: keep the centered-window pivot oracle. Python-only.

**Scope discipline:** these are *refinements* of v2's fixes, not replacements. v2's core architecture (smooth IS-OOS, two-sided excess penalty, smooth recall saturation, 1-to-1 matching) stays; v3 tightens each.

## Items implementing

### Tier A — 3-expert consensus

**A1. Use bootstrap-CI lower bound as the optimizer objective (not `mean − 0.5·std`)**
Where: `src/validation.py::build_optuna_objective` and `evaluate_params_on_prepared_folds`
Why: the `0.5` is a magic number, not calibrated to fold-count or fold variance. The bootstrap CI lower bound is already computed via `fold_scores_bootstrap_ci`; route it into the objective.
How: compute `lcb_bootstrap = fold_scores_bootstrap_ci(scores, n_boot=1000, alpha=0.10)[0]` and return that instead of `mean - 0.5·std`. Same `trial.report` change.
Cost: ~5 lines.

**A2. Increase bootstrap block length to respect IS overlap**
Where: `src/validation.py::fold_scores_bootstrap_ci`
Why: folds overlap heavily on IS (step=5%, IS=10% → 50% overlap). Block-1 sampling overstates effective N and produces too-tight CIs.
How: parameterize `block_len` (default `2`), implement stationary block bootstrap. With block_len=2, sample blocks of 2 consecutive fold-indices, wrap modulo, mean each bootstrap resample.
Cost: ~10 lines.

### Tier B — 2-expert consensus

**B1. Hungarian matching for TP assignment**
Where: `src/scoring.py::precision_at_n_stats`
Why: greedy bar-order assignment is order-dependent — an early signal can claim a backward pivot that a later, closer signal would have matched better. Hungarian (linear-sum assignment) is invariant.
How: build a cost matrix where `C[i,j] = |signal_i - pivot_j|` if pivot j is in signal i's window, else `inf`. Use `scipy.optimize.linear_sum_assignment` for O(k³) solution. k is small (signals × pivots in window).
Cost: ~30 lines. Bigger lift but real semantic improvement.

**B2. Lead-biased tiebreak in matching**
Where: same function
Why: the detector's stated value is anticipation; equal `|d|` should prefer forward (lead) over backward (lag). Bake this into the cost matrix.
How: `C[i,j] = abs(d) + (0.0 if d >= 0 else lead_bias_penalty)` where `lead_bias_penalty = 0.5` (so a 1-bar lead matches before a 1-bar lag). When Hungarian solves, leads will be preferred at equal absolute distance.
Cost: 2 lines on top of B1.

**B3. Expose component metrics via `trial.set_user_attr`**
Where: `src/validation.py::build_optuna_objective`
Why: scalar objective hides multi-objective tradeoffs. Logging components (precision, recall_saturated, frequency_factor, excess_penalty, IS-OOS gap) per-trial enables post-hoc Pareto inspection without changing the scalar contract.
How: in the per-fold compute, also store `per_scale_raw`, `frequency_factor`, `excess_penalty`, `is_score`, `oos_score`. Aggregate across folds (means). Call `trial.set_user_attr("component_X", ...)` at end of objective.
Cost: ~20 lines. Requires `compute_side_score` to expose more diagnostics (use `return_per_scale=True` and add new fields).

**B4. Stabilize the median reference scale**
Where: `src/scoring.py::compute_side_score`
Why: the median valid scale shifts when slice length crosses a scale-validity threshold (an N drops out of `valid_scales` → median changes discontinuously). Multi-objective expert called this "discontinuity hiding inside a smooth objective."
How: use a FIXED reference scale `REFERENCE_N=50` (the same one used for recall_target). Compute `excess_penalty` against `total_pivots_at_N=REFERENCE_N` if available, fall back to nearest valid scale.
Cost: ~5 lines.

### Tier C — Single-expert, concrete

**C1. Edge-zone correction in `total_pivots` denominator**
Where: `src/scoring.py::precision_at_n_stats`
Why: bars in `[n-N, n)` cannot be labeled (centered window needs forward bars). Counting them in `total_pivots` biases recall downward in a length-dependent way.
How: `effective_total = total_pivots_in_slice` (already excludes edge zones because labels there are 0); but the denominator we use IS that count. Verify behavior. If `label_pivots` does indeed return 0 for those bars, then `np.flatnonzero(pivots == sign)` naturally excludes them — this might already be correct. Add a unit-test that confirms `total_pivots` only counts labeled bars, not all bars where a label *could* have been.
Cost: 3 lines + a test.

### Tier D — Reporting only (lowest priority)

**D1. Deflated-Sharpe-style report on best_value**
Where: `src/speculatores145.py` report writer
Why: 500 trials × 14 folds × 7 scales = many implicit comparisons; best_value is biased upward.
How: report a "deflated best" using the simple Bonferroni-equivalent shrink: `best_value − std(top_k_values)·sqrt(2·log(n_trials)/n_trials)`. Pure reporting, not optimization.
Cost: ~10 lines.

## Items NOT implementing (already addressed or out of scope)

- Round-1 items (all resolved in v2)
- Replace centered-window oracle (off-limits)
- Multi-objective NSGAII reframing (separate larger project)
- Stability verdict threshold calibration against a null (round-2 finding but it's a tweak not a fix)

## Implementation order

1. Tier C first (verify what's already correct, doesn't change behavior, low risk)
2. Tier B4 (stabilize reference scale — needed before component logging is meaningful)
3. Tier B3 (component metrics — separate from algorithm changes)
4. Tier A1 + A2 (bootstrap-CI LCB + block length — same file, related)
5. Tier B1 + B2 (Hungarian matching + lead-bias tiebreak — most invasive, do last)
6. Tier D1 (reporting, optional)

## Validation

1. Smoke tests still pass (no regression on v2 semantics)
2. New smoke test: compare v2 vs v3 on the same Trial #178 vs Trial #249 — v3 should produce the same ordering (sparse > dense) with possibly different absolute scores
3. Component-metric logging visible in 5-trial end-to-end run

## Debug audit

Spawn `code-reviewer` agent to verify the v3 changes don't regress v2 fixes and don't introduce new bugs.

## Out of scope

- Pine changes (scorer is Python-only)
- Detector logic changes
- Re-running the optimizer (separate user decision)
