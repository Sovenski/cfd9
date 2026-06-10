# Implementation Report — Scorer v5 + Signal Card (v17.5)

**Date:** 2026-06-10
**Branch:** `feature/v17-antigaming` (uncommitted working tree on top of `3e3a8f0`)
**Spec:** `plan/scorer-v5-signal-card-spec.md` (FIXED v3 — all panel findings
F1–F11 + re-review R1–R6 applied)
**Plan:** `plan/scorer-v5-impl-plan.md` (16 tasks, all executed TDD-first)

---

## 1. Final gate result

- **Full suite:** `python -m pytest -q` → **269 passed, 0 failed, 0 skipped,
  0 errors** (13 warnings: 12 pre-existing NaN-in-log RuntimeWarnings from
  synthetic fixtures in `test_pooled_validation.py`, 1 nbformat
  MissingIDFieldWarning from the upstream Colab notebook format). Exit code 0.
- **Freeze audit:** PASS — every frozen file byte-identical to HEAD
  (`git diff HEAD --quiet` clean for `src/detector.py`,
  `src/v17_fastdetector.py`, `src/v17_gpu/**`, `src/parity.py`,
  `src/universe.py`, `src/search_space.py`, `src/speculatores145.py`,
  `src/validation.py`, `src/indicators.py`, all pre-existing `pine/*.pine`).
  `src/indicators.py` has zero diff (additive-only rule trivially satisfied —
  nothing was needed there).
- **Golden invariant:** PASS — all 12 signal-array sha256 digests in
  `results/diag/golden/golden_baseline.json` (6 per asset × SPX/DAX) are
  **identical** to the pre-v5 backup `temp/v5_backup/golden_baseline.pre_v5.json`.
  The recapture (`temp/capture_baseline.py`) changed exactly 6 JSON lines:
  fold scores + LCBs (the expected v5 re-pin per spec §2.5). Detection
  output is untouched.

## 2. What changed (by deliverable)

### 2.1 Spec fix pass
`plan/scorer-v5-signal-card-spec.md` rewritten in place to FIXED v3: F1
left-span L-clamp conditioning, F2 live-regressor c_side fit, F3 W_FP
exchange-rate + sensitivity test, F4 REFERENCE_mass=54.0 basis, F5 Pine
active-signal-only live scope, F6 cluster-bootstrap bands, F7 in-sample
disclaimer, F8 stop truth table, F9 deterministic chunk formula, F10
grid-floor/"500+" censoring semantics, F11 migration checklist; plus R1
(survival event = right-span R), R2 (k-origin = candidate pivot bar incl.
t+1 shift), R3 (censored exclusion + conditional-on-match label), R4
(off-grid step-function-floor lookup), R5 (truth-table row 6 re-fire),
R6 (banded Pine probabilities, no exemption).

### 2.2 Scorer v5 core
- **NEW `src/scoring_v5.py`** (350 lines): `SPAN_GRID = [20…500]`,
  `span_weight w = N*/100`, `W_FP = 0.2` (documented exchange rate),
  `REFERENCE_MASS = 27 · w(200) = 54.0`, `label_pivot_spans` (reuses the
  frozen `label_pivots` verbatim per grid scale; censoring incl. the
  "500+" cap), `compute_left_span`, ±1 weighted Hungarian
  `match_signals_weighted` (larger-span tie-break), `weighted_precision` /
  `weighted_recall`, `recall_target_eff` (N_eff = total_mass),
  `compute_side_score_v5` (same composite FORM as v4 on weighted masses).
- **`src/scoring.py`** (+36/−6): `add_pivot_labels` additionally writes
  `pivot_span_{high,low}` + `pivot_span_censored_{high,low}`
  (`pivot_N100` retained per §1.2); PEP 562 lazy re-export of the v5 surface.
  v4 functions (incl. `LEAD_BIAS`) are RETAINED in the module **only**
  because frozen `src/speculatores145.py` imports `compute_side_score`;
  no v17 scorer-path code calls them anymore.
- **`src/pooled_scoring.py`** (rewritten in place, 100 changed lines):
  pools `tp_mass`/`total_mass`/`n_unmatched` cluster-weighted, delegates to
  `compute_side_score_v5`.
- **`src/pooled_validation.py`** (+39 lines): mass plumbing through
  `_stream_stat`; informative-fold filter is now `pooled_total_mass_oos > 0`
  (spec §2.4 — fold counts will differ vs v4 runs, see §5 below).
- **`src/v17_acceptance.py`** (+11): `firing_excess` consumes weighted
  recall/precision.

### 2.3 Signal Card package — NEW `src/v17_card/`
| Module | Lines | Content |
|---|---|---|
| `survival.py` | 228 | Kaplan–Meier on right-span R (R1), grid/edge right-censoring (F10), cluster bootstrap S_lo/S_hi bands (F6) |
| `conditioning.py` | 102 | F1 L-clamp live rule `P(N*_eff≥x|L,k)`, R4 off-grid floor lookup, E[hold] span clock (§3.4) |
| `expected_move.py` | 175 | c_side two-fold chronological fit on the LIVE regressor (F2), R3 censored exclusion, R²<0.1 median fallback |
| `stop_rule.py` | 184 | Six-row F8 truth table (R2 k-origin/t+1 shift, R5 row-6 re-fire), engine `stop_series` with NaN==na semantics |
| `grading.py` | 218 | Retrospective grades, realized span badge, R-multiple backtest (§3.7) |
| `calibration.py` | 399 | Per-run orchestration → calibration object + signal_cards + r_multiple_backtest |
| `gpu_memory.py` | 177 | `gpu_memory_estimate` (HARD warn > 14 GiB) + F9 deterministic `chunk_size` formula, `BYTES_PER_CANDIDATE_BAR` probe-asserted |

### 2.4 Engine emission
`src/v17_runner.py` (+80): output gains `scorer: "v5"`, `calibration`,
`signal_cards`, `r_multiple_backtest`, `trace.scorer_version` +
`trace.calibration_block_hash`; F9 chunking applied at the RUNNER level
(`chunked_score_pop`, byte-identical to unchunked by test) — `src/v17_gpu/`
itself untouched; memory estimate + chunk formula inputs logged at run start.

### 2.5 Pine v17.5 + parity
- **NEW `temp/build_pine_v17_5.py`** (564 lines) → generates
  **`pine/speculatores_v17_5_signalcard.pine`** (952 lines): detection
  region byte-identical to the parity-PASS v17 file (anchor-tested),
  `// === CALIBRATION BLOCK (generated) ===` with `S_R` + `S_lo`/`S_hi`
  band tables per side, c_side, conviction breakpoints, stop constants;
  F5 single active-signal `var` state block per side; banded card label
  (R6); F7 disclaimer + plain-language T1/T2/T3 tooltips; version "17.5";
  §4.2 data-window parity export plots (band edges, move, hold, stop).
- **NEW `temp/parity_v17_5.py`** (293 lines): signals exact bool equality +
  count diff == 0, card numerics atol=1e-6, stop bar-for-bar exact
  (NaN==na, row-6 switch bars), off-grid survival-lookup assertions.

### 2.6 Workbook v5
`temp/build_h100_notebook.py` (+86) → `h100_v17_gpu.ipynb` now 8 cells:
Cells 1–5 interface unchanged (Cell 5 defaults GROUPS=indices, POPSIZE=128,
GENERATIONS=10, SOBOL_N=128; memory estimator prints the budget), Cell 6
extended engine report (calibration summary, card legend, capture ratio,
last-10-signal grade tail), **NEW Cell 7** writes the freshly calibrated
v17.5 pine file to Drive.

## 3. Test inventory (new/changed)

| File | Tests | Covers |
|---|---|---|
| `tests/test_scorer_v5.py` (NEW) | 27 | §7.2 truth tables, span weights, censoring, tie-break, F3 w_FP sensitivity (spray caught at 0.2), F4 constant |
| `tests/test_v17_card.py` (NEW) | 35 | KM vs closed form, L-clamp identity (off-grid args), k-origin/t+1 shift, F8 six-row table, `test_stop_series_na_after_*`, grading |
| `tests/test_v17_card_calibration.py` (NEW) | 11 | bootstrap-band seed reproducibility, c_side two-fold recovery, censored exclusion, card JSON schema |
| `tests/test_gpu_memory.py` (NEW) | 14 | F9 formula, ALL-1D pool peak < 18 GB @ popsize 128, chunked == unchunked byte-identity |
| `tests/test_pine_v17_5_builder.py` (NEW) | 24 | detection byte-identity anchors, calibration block, tooltips/disclaimer, version, export plots |
| `tests/test_workbook_v5.py` (NEW) | 6 | cell count +1, Cell 6/7 content, Cells 1–5 byte-identity |
| `tests/test_pooled_scoring.py` | 3 | re-pinned to mass-based composition |
| `tests/test_pooled_validation.py` | 14 | informative filter re-pinned to `total_mass` |
| `tests/test_v17_gpu_integration.py` | 6 | end-to-end v5 output fields + golden snapshot intact |
| `tests/test_cpcv_purge.py` | 18 | fixtures migrated to span columns |

**Migration checklist (spec §9) outcome:**
- RE-PINNED: golden fold scores/LCBs (via `temp/capture_baseline.py`),
  pooled-score composition, informative-fold filter, cpcv fixtures.
- UNCHANGED (hard gate): golden signal sha256 assertions — passed
  byte-identical.
- DELETED→REPLACED: count-based informative-filter tests → mass filter
  tests; v4 count-ratio precision test → mass-ratio test. `LEAD_BIAS` and
  lead-window matching are gone from the v17 scorer path (retained only
  inside frozen-consumer legacy v4 functions, see §2.2).

## 4. Calibration sample summary (builder smoke)

The checked-in `pine/speculatores_v17_5_signalcard.pine` carries a SMOKE
calibration from the 2-stream test pool (SPX+DAX 6000-bar tails):
`n_signals = 18` per side over `n_streams = 2`, band method = contiguous
signal-**block** bootstrap (stream count < 5), `n_boot = 50`, `seed = 0`,
`c_r2 = None` → pooled-median expected-move fallback active (R² rule).
This is a plumbing artifact, NOT a tradable calibration — the real
calibration block is regenerated by Colab Cell 7 from the optimizer run
JSON. Grid-floor bias (F10), block-band method (F6), c_side length bias
(R3) and the in-sample disclaimer (F7) are printed in the Cell 6 report.

## 5. Notes for the first v5 run

- All v4 LCBs are incomparable (objective era change); run slugs carry
  `scorer: "v5"`.
- Informative-fold counts will RISE vs v4 (mass filter admits T2/T3-only
  folds) — expected, called out in the run report.
- HIGH side stays monitor-grade (event ceiling, MEMORY note).

## 6. USER TODO (exact, in order)

a. Assistant commits + pushes `feature/v17-antigaming`; user flips the
   repo public if it is private (Colab clone needs it).
b. Fresh Colab session (L4 or H100), open workbook v5
   (`h100_v17_gpu.ipynb`), run **Cells 1–5 SMALL TEST**:
   `GROUPS=INDICES`, `GENERATIONS=10`, `POPSIZE=128` (L4-safe; the memory
   estimator prints the budget + F9 chunk at run start).
c. Run **Cell 6** and review the report with the assistant → then run
   **Cell 7**, which writes the calibrated v17.5 pine file to Drive.
d. Paste the v17.5 file into TradingView, apply the preset, **export the
   chart CSV** (same flow as the 2026-06-10 audit).
e. Hand the CSV to the assistant → `temp/parity_v17_5.py` audit (signals
   exact + card numerics 1e-6 + stops bar-for-bar). **On PASS:** the very
   big run — `GROUPS=INDICES,COMMODITIES,FX,WORLD_ETF`, `GENERATIONS=50`,
   `SOBOL_N=256`.

## 7. Outstanding / known limitations

- Censored-aware c_side fit is a permitted later upgrade (R3); current fit
  excludes censored spans (downward length bias, documented in report).
- Survival bands are coarse by design (~10² events, 10 grid points).
- R-multiple backtest ignores costs/slippage (documented).
- `BYTES_PER_CANDIDATE_BAR` is probe-asserted within 2× of the analytic
  estimate on CPU; first GPU run should confirm the logged estimate vs
  `nvidia-smi` (no action unless the 14 GiB warn fires).
