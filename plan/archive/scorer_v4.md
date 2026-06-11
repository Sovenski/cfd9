# Scorer v4 — Nested-scale Structural Oracle

## Diagnosis (why v3 wasn't enough)

Visual inspection of V15 Run 2 on TradingView (the score-winning preset from a 250-trial v3 optimization) **missed 6 of 7 famous SPX structural lows**:

| | Result |
|---|---|
| 2002-10 dotcom bottom | MISS |
| 2009-03 GFC bottom | MISS |
| 2011-10 Eurozone | MISS |
| 2016-02 oil crash | MISS |
| 2018-12 selloff | HIT |
| 2020-03 COVID bottom | MISS |
| 2022-10 inflation bottom | MISS |

Component-attr analysis revealed that v3's `frequency_factor` saturated at 1.0 for all top-20 trials on both sides, and the per-scale aggregation `mean over scales [5, 10, 20, 50, 100, 200, 500]` was dominated by the small-scale contributions because small-scale "pivots" are far more numerous. **The optimizer was finding configs that catch every 50-bar dip in a trending market with high recall — exactly what the score asked for. The score wasn't asking for structural detection.**

The labelling oracle treats any centered-window local extremum at scale N as a "pivot," so scale-5 pivots flood the count and dominate optimization pressure even though we want once-a-decade events.

## Constraint reaffirmed

The centered-window pivot oracle stays. Forward-economic outcomes (drawdowns, returns) remain off-limits.

## Fix — multi-scale persistence

A bar is a **structural pivot** if and only if it is the centered-window extremum at ALL scales in a nested set:

```
STRUCTURAL_NEST = [50, 100, 200]
```

Reading:
- Extremum over [-50, +50] AND
- Extremum over [-100, +100] AND
- Extremum over [-200, +200]

Picking [50, 100, 200] (not [100, 200, 500]) so the smallest OOS slice (756 bars) can still validate the 200-scale (needs 401 bars). With these scales:
- 50-bar window = ~quarter
- 100-bar window = ~5 months
- 200-bar window = ~10 months

A bar that's the lowest over a centered ±10-month window is plausibly a structural pivot — anything between a multi-quarter bottom and a multi-year one.

## Implementation

### Items implementing

| # | Where | Change |
|---|---|---|
| 1 | `src/scoring.py` constants | Add `STRUCTURAL_NEST: list[int] = [50, 100, 200]`. Replace `PIVOT_SCALES = [5,10,20,50,100,200,500]` with `PIVOT_SCALES = [100]` (single tolerance scale). Set `REFERENCE_N = 100`. |
| 2 | `src/scoring.py::add_pivot_labels` | Write a single column `pivot_N100` that contains the **AND-of-nest** result: +1 if pivot HIGH at all 3 nest scales, −1 if pivot LOW at all 3, else 0. The `pivot_N100` column carries the structural label; consumers (precision_at_n_stats, compute_side_score) are unchanged. |
| 3 | `src/scoring.py::label_structural_pivots` | New helper called by `add_pivot_labels`. Wraps `label_pivots` three times and computes the AND. Drops consecutive-run dedup (already strict). |
| 4 | `src/scoring.py::compute_side_score` | No code change — the existing loop now runs ONCE (single-scale), giving a single precision*recall_sat product. All v3 mechanics (Hungarian, lead-bias, smooth saturation, bootstrap LCB, two-sided excess penalty against REFERENCE_N=100) survive intact. |
| 5 | `temp/smoke_test_scorer_v4.py` | New smoke test: assert structural pivots are far less common than v3 per-scale pivots (e.g., on a 5000-bar synthetic random walk, ≤ 1% of bars are structural HIGH/LOW). |
| 6 | `temp/smoke_test_scorer_v4_oracle.py` | New: compute structural labels on the SPX 1D dataset and assert at least 5 of the 7 famous lows from §1 fall within a 30-bar window of a labelled structural low. |

### Items NOT implementing

- Removing `frequency_factor`: leave the multiplier in place. With scale=[100] it's still computed against MIN_RATE; if it stays saturated we'll see and can deflate in v5.
- Asymmetric HIGH/LOW scoring: the symmetric oracle handles both sides cleanly.
- Forward-economic outcomes: still off the table.
- Touching the detector or Pine: scorer-only change.

## Validation

1. **All 5 existing smoke tests still pass** (no regression to v3 semantics on the parts not changed).
2. **New `smoke_test_scorer_v4.py`**: nested-scale oracle returns ≤ 1% of bars as structural pivots on synthetic data.
3. **New `smoke_test_scorer_v4_oracle.py`**: assert ≥ 5 of 7 famous SPX lows fall near a labelled structural pivot. This is the "did we actually fix the bug" test.
4. **End-to-end 5-trial optuna run on a small SPX slice**: pipeline survives the API change.

## Debug audit

`code-reviewer` agent reviews the diff for:
- Slice-length edge cases (what if OOS < 401 bars → scale 200 can't be computed?). Fallback: write zeros into the `pivot_N100` column for those slices.
- Backward compatibility: existing tests reading `pivot_N5`, `pivot_N50`, etc. — those columns are no longer written. Audit which tests/code paths assumed multi-scale columns.
- The single-scale aggregation in `compute_side_score`: `weight_sum = log(100)/log(500)` ≈ 0.74, but normalizing `raw_score / weight_sum` still gives `precision^1.2 * recall_sat`. No issue.
- `REFERENCE_N` lookup: with `PIVOT_SCALES = [100]` and `REFERENCE_N = 100`, the lookup succeeds trivially.

## Run 3 plan

After v4 lands and tests pass:
- Update `optimize.ipynb` title to "Run 3 (Scorer v4 — nested-scale structural oracle)".
- Commit on feature branch `feature/scorer-v4-structural-oracle`.
- Open PR.
- User runs the notebook on Colab.
- Inspect: do the winners actually fire near 2009-03, 2020-03, 2022-10, etc.?

## Out of scope

- Re-running the optimizer (separate user decision after merge).
- Re-doing expert panel reviews — v4 is one targeted oracle fix, not a multi-component overhaul. If the structural test (`smoke_test_scorer_v4_oracle.py`) passes, the scorer is doing what we asked.
