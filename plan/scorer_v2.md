# Scorer v2 Plan — Implement all 16 expert-suggested changes

**Goal:** Address all 16 issues raised by the unprimed 5-expert panel on the Speculatores scorer/optimizer. Constraint: keep the centered-window pivot oracle. Python-only changes — Pine indicator unaffected because detector output semantics don't change, only the scorer.

## Implementation order

Files touched: `src/scoring.py`, `src/validation.py`, `src/speculatores145.py`.

### Items implementing (14 of 16)

| # | Where | Concrete change | Notes |
|---|---|---|---|
| 1 | `scoring.py::_fold_score` | Replace `max(0.0, oos − γ·max(0, is−oos))` with `oos · exp(−γ · max(0, is−oos))`. Always positive, always differentiable, decays smoothly. | Set γ=2.0 unchanged. |
| 2 | `scoring.py::compute_side_score` | Make `excess_penalty` two-sided. Replace `min(1, total/max(n_sigs,1))` with `2·n·t/(n²+t²+ε)` (harmonic-mean of ratios; 1 at parity, decays both directions). | Penalize both spam AND extreme sparsity |
| 3 | `scoring.py::compute_side_score` | Compute the reference pivot count for `excess_penalty` as `total_pivots_per_scale[len(scales)//2]` (median valid scale), not arithmetic mean | Removes small-N domination of the average |
| 4 | `scoring.py::compute_side_score` | Replace `min(1, recall/RECALL_TARGET)` with `1 − exp(−recall/RECALL_TARGET)`. Smooth saturation, gradient survives past target. | Different absolute scale than before — recalibration documented in commit |
| 5 | `scoring.py::compute_side_score` | Make `RECALL_TARGET` scale-adaptive: per-scale target = `RECALL_TARGET · √(REFERENCE_N / N)`. Small N gets harder target, large N gets easier. | `REFERENCE_N = 50`. Bayesian-weighted by attainable recall. |
| 6 | `scoring.py::precision_at_n_stats` | Replace many-to-one tolerance-window match with greedy nearest-neighbor 1-to-1 assignment. Sort signals by time; each signal claims at most one unmatched pivot within its window. | Eliminates double-counting in TP and recall |
| 7 | `scoring.py::precision_at_n_stats` | Per-scale tolerance cap = `min(N, max(30, int(round(N · 0.3))))` — 30% of N or 30, whichever larger, capped at N | Scale-adaptive replacement for flat LEAD_WINDOW_CAP=30 |
| 8 | `validation.py::evaluate_params_on_prepared_folds` | Replace `np.mean(fold_scores)` with `np.mean(scores) - 0.5·np.std(scores)`. Bandit LCB. Penalizes high-variance configurations. | Robust + uncertainty-aware in one step |
| 9 | (folded into #8) | LCB serves as the objective — no separate item | — |
| 10 | `validation.py` reporting | Add `_fold_score_ci_block_bootstrap(scores, block_len)` helper that returns a 90% CI. Log alongside the trial value. | Block length = 1 (since folds are already coarse units). Pure reporting. |
| 11 | `speculatores145.py::summarize_stability` | Add 20 non-local random restarts (uniform in `INT_BOUNDS` + `FLOAT_BOUNDS`) alongside the 50 local perturbations. Compute `local_vs_global_gap = local_mean − global_mean`. | Extra diagnostic, doesn't change verdict yet |
| 12 | `speculatores145.py::summarize_stability` | Move hard-coded thresholds `0.2` and `0.6` to module constants `STABILITY_ROBUST_THRESHOLD = 0.2` and `STABILITY_OUTLIER_THRESHOLD = 0.6`. Document them. | No behavior change, just configurability |
| 14 | `scoring.py::label_pivots` | Vectorize the centered rolling-max/min with numpy strided views instead of `rolling().apply(lambda)`. ~10x speedup expected. | Defer if it perturbs labels by 1 bar — must produce identical output |
| 15 | `scoring.py::compute_side_score` | Add optional `return_per_scale=True` argument. When set, returns the per-scale score dict alongside the scalar so the report writer can log it. | Backward compatible default |
| 16 | (analysis task) | After re-run, dump per-scale contributions of the new winning trial as a sanity check. | Diagnostic, no code change |

### Items SKIPPING (2 of 16)

| # | Where | Why skip |
|---|---|---|
| 13 | `scoring.py::_fold_score` shared-scale intersection | Multi-objective expert wanted to remove it. Bayesian expert defended it as the right hygiene. Dissent — leave as-is, revisit in a future round. |

## Validation steps after implementation

1. **Imports clean**: `python -m src.scoring`, `python -m src.validation`, `python -c "from src.speculatores145 import *"`
2. **Existing smoke tests pass**: `temp/smoke_test_v15_equiv.py`, `temp/smoke_test_v15_edge_subset.py`, `temp/smoke_test_v15_end_to_end.py`
3. **Manual scoring comparison**: synthesize two signal sets — one sparse (Trial #178 style, 64 signals) and one spammy (Trial #249 style, 357 signals). Both on the SAME parameters and data. Verify Scorer v2 ranks the sparse one HIGHER (or at least closer) where Scorer v1 ranked them similarly.
4. **5-trial end-to-end optuna**: confirm pipeline survives and produces a report.

## Debug round

Spawn a code-reviewer subagent to audit the diff for:
- Numerical stability (exp overflow, division-by-zero)
- Backward compatibility (default behavior of `return_per_scale=True`)
- Off-by-one in scale-tolerance computation
- LCB sign (we want `mean - 0.5*std`, not `mean + 0.5*std`)
- Two-sided excess_penalty edge case at n_signals=0

## Expert re-review

Spawn 5 fresh unbiased subagents (same personas, same minimal prompts, same constraint about not replacing the oracle). Have them assess the NEW code without telling them what changed.

Compare their critiques against the previous round to see if the changes addressed the concerns.

## Out of scope

- Pine-side changes (the scorer is Python-only; detector parity is preserved)
- Replacing the oracle with forward-economic outcomes (explicitly off the table)
- Multi-objective NSGAII reframing (separate roadmap item)
- Conditional sampling of gated dimensions (was Tier 2 in primary panel, didn't survive unprimed round)
