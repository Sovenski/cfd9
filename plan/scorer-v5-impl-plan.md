# Scorer v5 + Signal Card — Implementation Plan

**Source spec:** `plan/scorer-v5-signal-card-spec.md` (FIXED v3, F1–F11 + R1–R6 applied)
**Panel findings:** `plan/scorer-v5-spec-review.md`
**Date:** 2026-06-10
**Discipline:** strict TDD — every task lists its test FIRST; write it, run
`python -m pytest <file> -q`, watch it fail, implement, loop to green. Never
weaken an assertion. NO git commits.

---

## 0. Hard constraints (restated, binding for every task)

- **FROZEN (never modify):** `src/detector.py`, `src/v17_fastdetector.py`,
  `src/v17_gpu/**`, `src/parity.py`, `src/universe.py`, `src/search_space.py`,
  `src/speculatores145.py`, `src/validation.py`, existing `pine/*.pine` files,
  and ALL existing functions in `src/indicators.py` (additive-only there).
- **EDITABLE:** `src/scoring.py`, `src/pooled_scoring.py`,
  `src/pooled_validation.py`, `src/v17_acceptance.py`, `src/v17_runner.py`,
  NEW `src/v17_card/` package, `tests/`, `temp/` builders, NEW pine v17.5 file,
  `temp/build_h100_notebook.py` + `h100_v17_gpu.ipynb`.
- **GOLDEN INVARIANT:** `results/diag/golden/golden_{SPX,DAX}.npz` signal
  arrays stay BYTE-IDENTICAL (sha256s in `golden_baseline.json` unchanged).
  Fold scores/LCB are recaptured under v5 (Task 8). Any signal-byte change is
  a CRITICAL bug — stop, diagnose, do not re-pin.
- Detection math is sacred: v5 changes scorer/calibration/report side only.
  Both scorer paths (CPU exact + GPU search) score CPU-side; the GPU layer
  only produces signals (spec §2.3) — it is untouched.
- Repo style: module logger, full type hints, ≤ ~400-line files, named module
  constants, docstrings with spec cross-references.

### File-size note (declared up front)
`src/scoring.py` is 704 lines. Tasks 1–2 ADD code. To respect the size rule
without touching frozen files, v5 additions go into a NEW scorer-side module
`src/scoring_v5.py` (span labeling, weighted matching, v5 constants),
re-exported from `src/scoring.py` (`from .scoring_v5 import ...`) so the
public import surface stays `src.scoring`. This is the only new `src/` module
outside `src/v17_card/`; it is scorer-side by definition and within the
spirit of the EDITABLE list (it is imported ONLY by editable files). If the
owner objects, the fallback is in-file sections in `scoring.py` (accepting
~1,100 lines) — flagged as a decision point, default = new module.

---

## 1. Task graph (order + dependencies)

```
T1 span labeling ──► T2 weighted matching ──► T3 pooled plumbing ──► T4 acceptance inputs
                                                    │                       │
                                                    ▼                       ▼
                                              T5 w_FP sensitivity ──► T8 GOLDEN RECAPTURE (gate)
T1 ──► T6 card pkg skeleton + KM survival ──► T7 stop truth table
T6,T7 ──► T9 c_side / E[hold] / conviction ──► T10 grading + R-backtest
T10,T8 ──► T11 calibration orchestration + run_v17_gpu fields
(any time after T3) T12 memory estimator + chunk formula
T11 ──► T13 Pine v17.5 builder + parity script
T11,T13 ──► T14 workbook v5 (Cell 6 + Cell 7)
T8..T14 ──► T15 migration checklist execution ──► T16 final acceptance
```

16 tasks. T8 is a hard gate: nothing in T11+ that consumes fold scores may
land before the golden snapshot is recaptured and `tests/test_parity_golden.py`
is green with UNCHANGED signal sha256s.

---

## 2. Tasks

### T1 — Span labeling: SPAN_GRID, N\*, censoring, left-span L, columns
**Implements:** spec §1.1–§1.4 (incl. F10 grid-floor + "500+" censor bucket),
§3.2 left-span L definition (F1).
**Test FIRST:** `tests/test_scoring_v5_spans.py`
- `SPAN_GRID == [20, 30, 40, 50, 70, 100, 140, 200, 300, 500]` pinned.
- Synthetic series with a known global minimum: `pivot_span_low` records the
  LARGEST grid value passed (a true span-180 pivot → 140; F10 truncation).
- Noise floor: bar that is only a ±19 minimum → span 0 (no event).
- Monotonicity: span-N pivot is also a pivot at every smaller grid N
  (cross-check against existing `label_pivots(df, N)` for each grid N —
  byte-equality of the implied boolean masks, the strongest possible anchor
  to the frozen-semantics labeler).
- Plateau runs collapse to FIRST bar; ambiguous-both-sides bars excluded
  (same fixtures style as existing `label_pivots` tests).
- Right-edge censoring: bar 60 bars from the end with a true ≥50 left+right
  window → recorded `N*_lb = 50`, `pivot_span_censored_low == True`.
- Top bucket: a pivot passing 500 → recorded 500 AND censored flag True
  ("500+" semantics, never exact).
- `pivot_N100` column still produced by `add_pivot_labels` (retention, §1.2).
- Span weights: `w(N*) = N*/100` exact for every grid value; censored pivot
  weight uses `w(N*_lb)`.
- Left-span: `compute_left_span(low, i)` on a fixture where bar `i` is a
  30-bar-left / 200-bar-right minimum → `L == 30`; clamped at left edge and
  at 500.
**Files touched:** NEW `src/scoring_v5.py` (constants `SPAN_GRID`,
`REFERENCE_N` re-use, `span_weight()`, `label_pivot_spans(df) ->
pivot_span_{high,low} + pivot_span_censored_{high,low}`,
`compute_left_span(arr, i, is_high)`), `src/scoring.py` (re-export +
`add_pivot_labels` extension to add the four new columns; `pivot_N100`
retained), `tests/test_scoring_v5_spans.py`.
**Implementation note:** O(grid) construction — call the EXISTING
`label_pivots` per grid N and take per-bar max passed N (monotone by
construction); censoring derives from window-fit at the edge (`i + N > last`
→ that N is unproven → record bound + flag). Do NOT reimplement extremum
logic; reuse the frozen-validated labeler verbatim.
**Depends on:** nothing.

### T2 — ±1 weighted Hungarian matching + weighted precision/recall + N_eff
**Implements:** spec §2.1, §2.2 (F3 constant exposure, F4 N_eff/REFERENCE_mass).
**Test FIRST:** `tests/test_scoring_v5_matching.py`
- ±1 truth table (spec §7.2): signal at `t`, pivot at `t−1`/`t`/`t+1` → match;
  pivot at `t±2` → no match (FP + missed mass). Symmetric — assert no lead
  bias by mirroring fixtures.
- Hungarian 1:1: two signals, one pivot → exactly one TP; two pivots in
  window at EQUAL distance → tie-break prefers larger span.
- `W_FP == 0.2 == span_weight(20)` pinned, with the exchange-rate docstring
  asserted present at the definition site (string check keeps F3
  documentation load-bearing).
- `precision_w = tp_mass / (tp_mass + n_unmatched · W_FP)` and
  `recall_w = tp_mass / total_mass` exact on hand-computed fixtures,
  censored pivots contributing `w(N*_lb)`.
- `REFERENCE_MASS == 54.0` pinned (27 · w(200)); `recall_target_eff =
  RECALL_TARGET · sqrt(REFERENCE_MASS / N_eff)` with `N_eff = total_mass`
  asserted on two fixture masses (the F4 anchor test — v4↔v5 basis can never
  drift silently).
- `compute_side_score_v5` composite: same FORM as v4 (PRECISION_EXPONENT,
  recall saturation, MIN_RATE frequency factor, GAMMA fold penalty) operating
  on weighted quantities — assert against a hand-computed scalar.
**Files touched:** `src/scoring_v5.py` (`W_FP`, `REFERENCE_MASS`,
`match_signals_weighted()`, `weighted_stats()` → dataclass
`WeightedStats(tp_mass, total_mass, n_signals, n_unmatched, n_bars)`,
`compute_side_score_v5()`), `src/scoring.py` (re-exports),
`tests/test_scoring_v5_matching.py`.
**Implementation note:** keep the existing Hungarian implementation
(`scipy.optimize.linear_sum_assignment` path in `scoring.py`); costs =
`|distance|`, out-of-window = `_HUNGARIAN_INF`; tie-break via a secondary
cost epsilon `−ε·w(N*)` with ε small enough to never flip a distance ranking
(assert ε < 1/(2·max_w) in a test). `LEAD_BIAS` is NOT used in the v5 path
(deletion happens in T15 once v4 tests are migrated).
**Depends on:** T1.

### T3 — Pooled plumbing: mass through StreamStat, informative-fold filter
**Implements:** spec §2.3, §2.4.
**Test FIRST:** `tests/test_pooled_scoring_v5.py`
- `StreamStat` (or a v5 sibling `StreamStatV5`) carries `tp_mass`,
  `total_mass`, `n_unmatched_signals`; pooled aggregation sums mass with
  cluster weights exactly as counts were summed in v4 (fixture: two streams,
  weights 1.0/0.5, hand-computed pooled precision_w/recall_w).
- `pooled_fold_score` composition unchanged in FORM: assert
  `oos · exp(−GAMMA·max(0, is−oos))` on weighted fixture scores.
- `_fold_is_informative`: fold with ONLY a span-30 OOS pivot (no N100) →
  informative under v5 (`total_mass > 0`), NOT informative under the old
  `pivot_N100` count check — the test encodes the intended §2.4 behavior
  change explicitly.
- Zero-mass fold → score 0.0, excluded, no crash (mirrors existing
  `test_pooled_validation.py` degenerate-fold tests).
**Files touched:** `src/pooled_scoring.py` (`pooled_side_score` /
`pooled_fold_score` consume mass fields; `N_eff`-based
`recall_target_eff`), `src/pooled_validation.py` (`_stream_stat` computes
v5 weighted stats via `match_signals_weighted`; `_fold_is_informative` →
`pooled_total_mass_oos > 0.0`; components dict gains
`tp_mass_{is,oos}`, `total_mass_{is,oos}`, `n_eff_oos`),
`tests/test_pooled_scoring_v5.py`.
**Implementation note:** `evaluate_pooled_fold` signature unchanged
(callers in `v17_optimize.py`/`v17_search.py` are NOT in the editable list’s
spirit to rewrite — verify they only consume the returned `(score,
components)` tuple; if `v17_optimize.py` needs edits beyond imports, stop
and confirm — it is not on the EDITABLE list).
**Depends on:** T2.

### T4 — firing_excess on weighted inputs + acceptance-side wiring
**Implements:** spec §2.2 last bullet ("firing_excess now uses weighted
recall/precision"), §9 text audit deferred to T15.
**Test FIRST:** `tests/test_v17_acceptance_v5.py`
- `firing_excess(precision_w, recall_w, cap)` numerics on weighted fixtures
  (function FORM unchanged — assert v4 fixture values still hold, since the
  function is input-agnostic; what changes is WHICH numbers are fed in).
- `rank_by_penalized_lcb` unchanged behavior (regression fixtures).
- The components-dict path (`v17_acceptance.py:147`) reads
  `precision_oos`/`recall_oos` — assert those keys now carry the WEIGHTED
  values from T3’s components (integration fixture through
  `evaluate_pooled_fold` on a tiny synthetic stream).
**Files touched:** `src/v17_acceptance.py` (docstring/inputs note only —
function math unchanged), `src/pooled_validation.py` (ensure components
keys `precision_oos`/`recall_oos` are the weighted quantities),
`tests/test_v17_acceptance_v5.py`.
**Depends on:** T3.

### T5 — w_FP sensitivity test (F3, mandatory before default is trusted)
**Implements:** spec §2.2 F3(b)(c), §7.2.
**Test FIRST (the test IS the deliverable):** `tests/test_w_fp_sensitivity.py`
- Build a toy landscape: synthetic pivot set (mixed spans on the grid) + two
  fixed candidate signal series — SPRAY (fires ~every 5 bars, hits
  everything ±1 eventually) and SELECTIVE (fires near true pivots only).
- For `w_FP ∈ {0.2, 0.5, 1.0}`: compute weighted fold scores + `firing_excess`
  for both candidates; assert the acceptance gate (`firing_excess` folded
  into penalized LCB via `rank_by_penalized_lcb`) ranks SELECTIVE above
  SPRAY at the LOOSEST setting (0.2).
- If this assertion cannot be made to pass at 0.2 (honest outcome), the
  default `W_FP` moves UP until it passes and BOTH the constant and spec
  §2.2 are updated — record the decision in `plan/IMPLEMENTATION_REPORT_v5.md`.
**Files touched:** `tests/test_w_fp_sensitivity.py`; possibly `W_FP` value
in `src/scoring_v5.py` + spec edit (only per the F3(c) rule above).
**Depends on:** T4.

### T6 — `src/v17_card/` package: KM survival on right-span R + bands + conditioning
**Implements:** spec §3.1, §3.2 (R1 event variable, F1 L-clamp, R4 off-grid
rule, F6 cluster bootstrap), §3.4 E[hold].
**Test FIRST:** `tests/test_v17_card_survival.py`
- **KM vs closed form:** synthetic exponential right-spans with independent
  censoring → KM at grid points within tolerance of the analytic survivor;
  includes "500+" cap censoring (events at cap treated as censored, F10/R1).
- **Right-span R definition:** fixture lows where the first strictly lower
  low after candidate bar `i` is at `i+7` → `R == 7`; no lower low before
  series end → censored at edge; HIGH side mirrors with strictly higher high.
- **Population rule:** matched, unmatched, and near-ambiguous signals ALL
  enter the table (count assertion on a mixed fixture) — R is defined from
  the candidate pivot bar, never a matched pivot.
- **Conditioning identity (F1):**
  `P(N*_eff ≥ x | L, k) == 0 if x > L else S_R(max(x,k))/S_R(k)` asserted
  for x, k ON and OFF the grid; off-grid args floor to the largest grid
  value ≤ arg; `S_R(x) = 1 for x < 20` (R4). Same function is the ONE used
  by the live card, the calibration export, and the Pine block generator.
- **k-origin (R2):** bars-survived counted from candidate pivot bar `i`
  (incl. the t+1 shift fixture: pivot shifts to t+1 at widening → L
  recomputed at t+1, k restarts) — shared fixture with T7.
- **Cluster bootstrap (F6):** fixed seed → byte-identical `S_lo`/`S_hi`
  envelopes across two runs; stream-level resampling when ≥5 streams,
  contiguous signal-blocks otherwise (assert the switch); ≥200 resamples;
  band contains the point KM.
- **E[hold] (§3.4):** discrete expectation over the grid with Δx steps,
  L-clamp zeroing terms x_j > L, cap at 500 — hand-computed fixture.
- Optional covariate split: gate refuses split when a branch has n < 60.
**Files touched (NEW package, each file ≤ ~300 lines):**
- `src/v17_card/__init__.py` — public API + `__all__`
  (`calibrate_run`, `SurvivalTable`, `SignalCard`, `StopState`, ...).
- `src/v17_card/survival.py` — right-span extraction, KM fit
  (`fit_km_survival`), `cluster_bootstrap_bands`, `condition_survival`
  (THE F1/R4 rule), `expected_hold`.
- `src/v17_card/snapshot.py` — §3.1 fire-time feature snapshot dataclass
  (`@dataclass(frozen=True)`), built from detector outputs + `calc_har_vol`
  (read-only use of frozen `src/indicators.py`).
- `tests/test_v17_card_survival.py`.
**Depends on:** T1 (SPAN_GRID, left-span).

### T7 — Stop / H0 truth table + candidate pivot bar + live state machine
**Implements:** spec §3.5 (F8 exact widening, R2 candidate bar/k-origin,
R5 row-6 re-fire), §7.3.
**Test FIRST:** `tests/test_v17_card_stops.py` — one test per truth-table row,
LOW side + HIGH mirror:
1. Fire at t-close: `stop == min(low[t−1], low[t])`, `i == argmin` (tie →
   earlier bar), card active.
2. t+1 closes ≥ fire-time stop with lower wick BELOW it: NOT invalidated
   (wick is hypothesis-consistent); stop widens to include `low[t+1]`; if
   `low[t+1]` is the new STRICT min → `i = t+1`, L recomputed, k restarts;
   tie → `i` keeps earlier bar; stop now FINAL.
3. `close[t+1] < fire-time stop`: INVALIDATED at t+1, NO widening (assert
   ordering: breach check against PRE-widening stop), grade "stopped".
4. u > t+1, intrabar `low[u] < stop`: INVALIDATED at u, k freezes at u−i.
5. u > t+1, `low[u] ≥ stop`: k = u−i, conditioning per §3.2 (calls T6's
   `condition_survival` — integration assertion).
6. Same-side re-fire at u: old card FREEZES at fire-time values (grade
   pending), state block RESETS per row 1, plotted stop series switches on
   the re-fire bar — assert the engine stop SERIES bar-for-bar on an
   overlapping-signals fixture (this exact series is what `temp/parity_v17_5.py`
   later compares against Pine).
- Disaster stop: q95 of matched-hit MAE in σ-units computed under v5
  matching on a fixture (and a guard test that the old 15.2σ literal appears
  NOWHERE in `src/v17_card/`).
- Risk display: `entry_close − stop`.
**Files touched:** `src/v17_card/stops.py` (`StopState` state machine —
pure function `step(state, bar) -> state`, no I/O; `stop_series(df, fires)`
producing the per-bar engine stop column), `tests/test_v17_card_stops.py`.
**Depends on:** T6 (conditioning call), T1 (left-span recompute).

### T8 — GOLDEN RECAPTURE (gate) — signal bytes identical, scores re-pinned
**Implements:** spec §2.5, §7.1, §9 re-pin items.
**Test FIRST:** extend `tests/test_parity_golden.py` BEFORE recapture with an
explicit split:
- (a) signal-array sha256 assertions read the CURRENT
  `golden_baseline.json` `array_sha256` map — copy those ten hex digests as
  LITERALS into the test (hard CI gate, immune to a re-capture overwriting
  the JSON); any mismatch is CRITICAL, never re-pin.
- (b) fold-score/LCB assertions read from the JSON (re-pinnable by design).
**Procedure:**
1. Add the literal-sha256 test, run pytest → green against current snapshot.
2. Land T1–T5 (already done by ordering), run
   `python temp/capture_baseline.py` ONCE (it overwrites fold scores + LCB
   in the JSON; arrays are regenerated but must hash identically).
3. Run full `python -m pytest tests/ -q`:
   - literal sha256s green (signal bytes unchanged — detection untouched);
   - fold-score tests green against the new JSON;
   - any OLD hard-coded v4 score literals in other tests surface here →
     re-pin per the T15 checklist (RE-PIN list only — never the sha256s).
4. Record old→new LCBs per asset/side in `plan/IMPLEMENTATION_REPORT_v5.md`.
**Files touched:** `tests/test_parity_golden.py`,
`results/diag/golden/golden_baseline.json` (regenerated by the script —
arrays byte-identical), possibly `temp/capture_baseline.py` (ONLY if its
informative-fold comment block needs the §2.4 wording update; logic
untouched).
**Depends on:** T1–T5 (the v5 scorer must be live in the pooled path).

### T9 — Expected move (c_side two-fold), E[hold]-at-fire, conviction
**Implements:** spec §3.3 (F2, R3), §3.4 (k=0 case), §3.6.
**Test FIRST:** `tests/test_v17_card_expected_move.py`
- **F2 recovery:** synthetic signals where true `|move| = c·σ·sqrt(E[hold])
  + noise` → two-fold chronological split (fit table on A → regressors for
  B, vice versa; pool; least squares through origin) recovers `c` within
  tolerance; assert the regressor is the LIVE-computable quantity (the test
  computes E[hold] only from fire-time info).
- **R3 exclusion:** censored-span matched signals are EXCLUDED from the fit
  (count assertion) and the fit diagnostics dict carries
  `censored_excluded_n` + the documented downward-bias note string.
- **R3 conditional label:** calibration output marks expected_move as
  `conditional_on_match: true` (schema assertion; the Pine/tooltip text
  consumes this in T13).
- Fallback: synthetic R² < 0.1 → card uses pooled median move (flag set).
- IQR band from empirical residuals — fixture assertion.
- FINAL exported table fitted on ALL signals (split only de-biases c_side):
  assert exported `S_R` equals full-sample fit.
- Conviction: percentile rank of `P(N*_eff≥50|·, k=0) · expected_move` among
  historical same-side signals, 0–100, monotone fixture.
**Files touched:** `src/v17_card/expected_move.py` (`fit_c_side_twofold`,
`expected_move_at_fire`), `src/v17_card/conviction.py` (or fold into
`calibration.py` if < 80 lines), `tests/test_v17_card_expected_move.py`.
**Depends on:** T6.

### T10 — Retrospective grading + R-multiple backtest
**Implements:** spec §3.7.
**Test FIRST:** `tests/test_v17_card_grading.py`
- Per-signal grade rows on a fixture: fire bar/side, matched ±1 y/n,
  realized span badge (grid value or `"≥N (censored)"` bound), realized move
  to span end, MAE/MFE, realized R-multiple computed under the card's OWN
  entry/stop/clock rules (entry = fire close, stop per T7 state machine,
  exit by clock at E[hold] if neither stopped nor span-resolved) — one
  hand-computed end-to-end row.
- Row-6 overlap: a signal frozen by re-fire still gets graded when its span
  resolves (grade pending → resolved).
- Aggregates: capture ratio, expectancy, win rate on the fixture; costs/
  slippage ignored — assert the `"costs_ignored": true` documentation flag.
**Files touched:** `src/v17_card/grading.py` (`grade_signals`,
`r_multiple_backtest`), `tests/test_v17_card_grading.py`.
**Depends on:** T7 (stop rules), T9 (E[hold] clock).

### T11 — Calibration orchestration + run_v17_gpu output fields + card JSON schema
**Implements:** spec §5 first bullet, §3.1 storage, §9 trace fields.
**Test FIRST:** `tests/test_v17_card_calibration.py`
- `calibrate_run(folds/streams, signals, side) -> CalibrationResult` produces
  the full JSON-serializable payload; schema test pins required keys:
  `scorer: "v5"`, `scorer_version: "v5"` in trace, `calibration_block_hash`
  (sha256 of the canonical JSON of survival+bands+c_side+stop constants),
  `calibration: {S_R, S_lo, S_hi, grid, c_side, conviction_breakpoints,
  stop_rule, fit_diagnostics{r2, censored_excluded_n, band_method,
  grid_floor_bias_note, in_sample_disclaimer}}`, `signal_cards: [...]`
  (§3.7 rows), `r_multiple_backtest: {...}`.
- Round-trip: `json.dumps` → `json.loads` → equality (no numpy leakage).
- `run_v17_gpu` smoke (tiny synthetic, monkeypatched GPU scorer as in
  existing `tests/test_v17_search.py` style): output dict gains the fields
  above; the DEFAULT v17 keys are unchanged (regression assertion against
  the existing smoke fixture).
**Files touched:** `src/v17_card/calibration.py` (orchestrator),
`src/v17_card/__init__.py`, `src/v17_runner.py` (`run_v17_gpu` calls
`calibrate_run` post-acceptance and attaches `scorer`, `calibration`,
`signal_cards`, `r_multiple_backtest`; run-slug metadata gains
`scorer: "v5"`), `tests/test_v17_card_calibration.py`.
**Depends on:** T8 (scores stable), T9, T10.

### T12 — L4 memory estimator + deterministic chunk formula (F9)
**Implements:** spec §6 (all four requirements).
**Test FIRST:** `tests/test_gpu_memory.py`
- `gpu_memory_estimate(n_streams, n_folds, n_slices, bars_per_slice,
  n_scales=499, popsize, bytes_per_candidate)` returns bytes; INDICES pool
  shape (3×5×2, ~2850 bars) ≈ 0.4 GB and ALL-1D (~15 streams) ≈ 1.8 GB
  within 20% (pins the §6 budget arithmetic).
- `chunk_size(budget_bytes, n_lanes, max_bars, bytes_per_candidate)` ==
  `floor(budget / (n_lanes·max_bars·bpc))`, clamped to `[1, popsize]`;
  `BUDGET_BYTES == 14 GiB` pinned.
- ALL-1D pool at popsize 128: estimated peak < 18 GB AND chunk ≥ 1 (§6.3).
- `bytes_per_candidate` analytic estimate (≈16 B/bar) vs the probe
  measurement hook: test injects a fake probe measurement and asserts the
  2x-agreement assertion fires correctly both ways (§6.2).
- Chunked-vs-unchunked LCB byte-identity (§6.4): CPU-side synthetic
  15-stream pool through the POOLED scorer with the chunk boundaries forced
  in the candidate loop of the runner — byte-identical LCBs (this tests the
  chunk PARTITIONING math, not the frozen GPU kernels).
**Files touched:** NEW `src/v17_card/gpu_memory.py` (pure functions — no
GPU import; lives in the card package because `src/v17_gpu/**` is frozen
and `v17_runner.py` must stay ≤400 lines), `src/v17_runner.py` (log the
estimate + formula inputs + resulting chunk at `run_v17_gpu` start; HARD
warn via `logger.warning` above 14 GB; pass the precomputed chunk to the
GPU layer ONLY through its existing popsize-chunk parameter — if the frozen
API exposes none, the value is logged as advisory and the limitation
recorded in the report; NO adaptive OOM-retry logic anywhere),
`tests/test_gpu_memory.py`.
**Depends on:** none (parallel-safe); wire-up into runner after T11 lands.

### T13 — Pine v17.5 builder + parity script
**Implements:** spec §4.1, §4.2, §3.2 banded display (R6), F5 scope, F7
disclaimer, §1.4 tooltips.
**Test FIRST:** `tests/test_pine_v17_5_builder.py`
- Builder is a pure function `build_pine_v17_5(v17_text, calibration) -> str`
  (imported from `temp/build_pine_v17_5.py` via path injection, mirroring
  how `temp/build_h100_notebook.py` self-validates):
  - **Detection byte-identity:** every line of
    `pine/speculatores_v17_presets_gold.pine` between the detection anchors
    appears VERBATIM and in order in the output (anchor-region equality —
    the frozen-parity property); additions are append/insert-only outside
    the detection region.
  - Contains `// === CALIBRATION BLOCK (generated) ===` with `S_R`, `S_lo`,
    `S_hi` arrays per side, `c_side`, conviction breakpoints, stop-rule
    constants — values match the input calibration JSON exactly (parse-back
    assertion).
  - Version string `"17.5"` in the indicator title.
  - Tooltip text anchors: T1/T2/T3 §1.4 sentences, conditional-on-match
    label (R3), H0 explanation, live-vs-frozen scope note (F5), the F7
    in-sample disclaimer EXACT sentence.
  - Band rendering: the card label format string carries `a–b%` band
    placeholders for P(T2+) and P(T1) (R6 — never a point value).
  - Live `var` state block per side present: fire bar, candidate pivot bar
    `i`, stop, L, k counter, intact flag; row-6 reset code path present
    (text anchors for the state variables).
  - Off-grid floor-lookup helper emitted (the SAME rule as
    `condition_survival` — generated from one shared table so engine and
    Pine cannot diverge, R4).
  - Debug/export plots: `plot()` lines for P(T2+) lo/hi, P(T1) lo/hi,
    expected move, expected hold, stop level (data-window export for §4.2).
  - Determinism: same inputs → byte-identical output (run twice).
- `temp/parity_v17_5.py` audit logic unit-tested on a synthetic TV-style
  CSV: `signal_high/low` exact bool equality + count diff == 0 (binding
  gate); card numerics `atol=1e-6`; stop column EXACT bar-for-bar incl. a
  row-2/row-3 t+1 ordering fixture and a row-6 re-fire switch fixture
  (reuses T7's engine `stop_series`); survival lookups exact at non-grid
  arguments (R4 assertion).
**Files touched:** NEW `temp/build_pine_v17_5.py`, NEW
`temp/parity_v17_5.py`, NEW generated `pine/speculatores_v17_5_signalcard.pine`
(output artifact — existing pine files untouched),
`tests/test_pine_v17_5_builder.py`.
**Depends on:** T11 (calibration JSON), T7 (engine stop series for parity
fixtures).

### T14 — Workbook v5: extended Cell 6 report + new Cell 7 (Pine → Drive)
**Implements:** spec §5 (workbook bullets), F7 (Cell-6 card legend
disclaimer).
**Test FIRST:** `tests/test_workbook_v5.py`
- Run `temp/build_h100_notebook.py` (subprocess or import); load
  `h100_v17_gpu.ipynb` with nbformat:
  - Cell count == previous + 1; Cells 1–5 sources byte-identical to the
    pre-change builder output for those cells (interface unchanged, §5).
  - Cell 6 source contains: calibration summary block, card legend WITH the
    F7 disclaimer sentence, capture-ratio line, last-10-signals table tail,
    fold-count-change + v4-LCB-incomparability callouts (§2.4/§2.5
    report wording).
  - NEW Cell 7: writes `pine/speculatores_v17_5_signalcard.pine` (built via
    `temp/build_pine_v17_5.py` from the run JSON) to the mounted Drive path;
    marked optional in its markdown header.
  - All pure-python code cells compile (the builder's existing self-check
    stays in place and is asserted by the test run’s exit code 0).
**Files touched:** `temp/build_h100_notebook.py` (CELL_INSPECT extension +
new `CELL_WRITE_PINE`), regenerated `h100_v17_gpu.ipynb`,
`tests/test_workbook_v5.py`.
**Depends on:** T11 (run-JSON fields Cell 6 reads), T13 (builder Cell 7
invokes).

### T15 — Migration checklist execution (spec §9, F11)
**Implements:** spec §9 verbatim — each checkbox is a sub-step; tick them in
the spec file copy inside `plan/IMPLEMENTATION_REPORT_v5.md`.
**Test FIRST:** the checklist's own test moves —
- DELETE (with their v5 replacements ALREADY green from T2/T3): lead-window
  matching tests, `LEAD_BIAS` tests, count-based informative-fold tests —
  grep `tests/` for `LEAD_BIAS|lead_window|pivot_N100 == ` to enumerate;
  each deletion is recorded next to its §7.2 replacement in the report.
- RE-PIN sweep: any remaining hard-coded v4 fold-score/LCB/firing literal
  (already handled at T8; this is the verification grep).
- NEW guard test `tests/test_v5_migration.py`:
  - `LEAD_BIAS` no longer referenced by the v5 scorer path (grep-style
    assertion over `src/scoring_v5.py`, `src/pooled_validation.py`,
    `src/pooled_scoring.py` sources);
  - no v4 vocabulary ("lead window", "LEAD_BIAS", tier-ratio wording) in
    `src/v17_acceptance.py` / `src/pooled_validation.py` docstrings and log
    strings ("pivot_N100" allowed ONLY at the retained column definition and
    the §1.2 retention comment);
  - run-JSON trace carries `scorer_version: "v5"` + calibration block hash.
**Steps:**
1. Text audit + update of editable files' docstrings/log messages to v5
   vocabulary (span mass, ±1 direct hit). Detection-side files: FORBIDDEN —
   the guard test scope deliberately excludes them.
2. Delete `LEAD_BIAS` constant and the `pivot_N100 == lbl` informative check
   from the SCORER path (`pivot_N100` COLUMN stays per §1.2).
3. Run-report wording: fold-count change (§2.4) + v4-LCB incomparability
   (§2.5) — already asserted in T14's Cell-6 test; verify engine-side report
   strings too.
**Files touched:** `src/scoring.py`, `src/scoring_v5.py`,
`src/pooled_validation.py`, `src/pooled_scoring.py`, `src/v17_acceptance.py`
(text/cleanup only), deleted/added test files,
`tests/test_v5_migration.py`.
**Depends on:** T8–T14 (everything green first; deletions last so coverage
never gaps).

### T16 — Final acceptance + implementation report
**Implements:** spec §7 (definition of done), §9 report items.
**Steps (verification, no new code):**
1. `python -m pytest tests/ -q` — full green; count and record.
2. Confirm golden signal sha256 literals green (§7.1) and that
   `results/diag/golden/*.npz` files are byte-identical to pre-v5 (hash the
   files themselves, record in report).
3. Regenerate `pine/speculatores_v17_5_signalcard.pine` and
   `h100_v17_gpu.ipynb` from the latest available run JSON (or the T11 smoke
   fixture if no fresh run exists — stated explicitly in the report).
4. Write `plan/IMPLEMENTATION_REPORT_v5.md`: what changed, pytest counts,
   re-pinned vs deleted test ledger (filled §9 checklist), old→new golden
   LCBs, w_FP sensitivity outcome, known limitations (§8 restated incl.
   c_side downward bias R3, grid-floor bias F10, costs-ignored backtest),
   and the EXACT user TODO: paste v17.5 into TV → export chart CSV → hand
   to assistant (`temp/parity_v17_5.py` run) → small optimizer test
   (INDICES, gens 10) → big run (ALL-1D, gens 50).
**Files touched:** `plan/IMPLEMENTATION_REPORT_v5.md`.
**Depends on:** T15.

---

## 3. Test inventory (all written FIRST, per task)

| # | Test file | Task | Spec § |
|---|-----------|------|--------|
| 1 | `tests/test_scoring_v5_spans.py` | T1 | §1.1–1.3, F10 |
| 2 | `tests/test_scoring_v5_matching.py` | T2 | §2.1–2.2, F3/F4 |
| 3 | `tests/test_pooled_scoring_v5.py` | T3 | §2.3–2.4 |
| 4 | `tests/test_v17_acceptance_v5.py` | T4 | §2.2 |
| 5 | `tests/test_w_fp_sensitivity.py` | T5 | F3, §7.2 |
| 6 | `tests/test_v17_card_survival.py` | T6 | §3.2/3.4, R1/R4/F1/F6 |
| 7 | `tests/test_v17_card_stops.py` | T7 | §3.5, F8/R2/R5 |
| 8 | `tests/test_parity_golden.py` (extended) | T8 | §2.5, §7.1 |
| 9 | `tests/test_v17_card_expected_move.py` | T9 | §3.3/3.6, F2/R3 |
| 10 | `tests/test_v17_card_grading.py` | T10 | §3.7 |
| 11 | `tests/test_v17_card_calibration.py` | T11 | §5, §9 trace |
| 12 | `tests/test_gpu_memory.py` | T12 | §6, F9 |
| 13 | `tests/test_pine_v17_5_builder.py` | T13 | §4.1–4.2, F5/F7/R6 |
| 14 | `tests/test_workbook_v5.py` | T14 | §5 |
| 15 | `tests/test_v5_migration.py` | T15 | §9, F11 |

Deleted under T15: lead-window/`LEAD_BIAS`/count-filter tests (each with its
green v5 replacement already in rows 1–3).

---

## 4. Risks / decision points

1. **`src/scoring_v5.py` is a new `src/` module not on the EDITABLE list** —
   chosen to honor the ≤400-line rule; imported only by editable files.
   Fallback: in-file sections in `scoring.py`. Confirm with owner at T1.
2. **`src/v17_optimize.py` / `src/v17_search.py` are not on the EDITABLE
   list** but sit between the pooled scorer and acceptance. The plan keeps
   their call signatures untouched (T3 note); if any edit beyond an import
   turns out to be required, STOP and confirm before touching.
3. **F9 chunk wiring:** `src/v17_gpu/**` is frozen — if the GPU API exposes
   no chunk parameter, the deterministic chunk is computed + logged but
   advisory (documented limitation in the report). The formula/tests are
   unconditional either way.
4. **w_FP default may move** (F3 honest outcome) — spec edit + report note
   are the sanctioned path, pre-authorized by §2.2(c).
5. **Golden recapture ordering:** literal-sha256 test MUST land before
   `temp/capture_baseline.py` is re-run (T8 step 1), otherwise a silent
   signal-byte regression could be re-pinned by accident.
6. **Pine object/runtime limits:** F5 scope (active-signal-only live state)
   is designed in, but TV compile is only verifiable at the user's manual
   §4.2 step — builder tests cover text anchors, not Pine compilation.
