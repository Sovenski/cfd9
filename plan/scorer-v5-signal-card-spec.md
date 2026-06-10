# Scorer v5 + Signal Card — Design Spec (v17.5)

**Status:** FIXED v3, post-panel + re-review (all findings F1–F11 from
`plan/scorer-v5-spec-review.md` applied, plus the re-review fixes R1–R6:
survival event variable pinned to right-span R, k-origin pinned to the
candidate pivot bar, c_side censoring handling, off-grid lookup rule,
re-fire truth-table row 6, banded Pine probabilities; ready for
implementation)
**Date:** 2026-06-10
**Owner decisions baked in:** continuous pivot span N\* (not discrete tiers); ±1-bar
matching ("direct hit"); HAR as expected-move core (no VIX/options data); pivot-low
H0 stop; tier names T1/T2 retained ONLY as display vocabulary; plain-language
tooltips in Pine; deliverable = Pine v17.5 + engine + workbook, L4-safe.

---

## 0. Goals / non-goals

**Goals**
1. Replace Scorer v4's `[t−2, t+30]` lead-window matching with **±1-bar direct-hit
   matching** against **span-weighted pivots** (continuous N\*).
2. Add a **per-signal Signal Card** to BOTH the engine and Pine v17.5:
   live P(span ≥ X) survival estimates, HAR-based expected move, span-clock
   expected hold, pivot-low invalidation (H0), conviction score. All card
   constants are **calibrated from history per run** and **exported to Pine**
   like presets — never hand-set.
3. Retrospective per-signal **quality grades** (realized span badge, realized
   R-multiple) in engine reports.
4. Preserve the existing trust chain: detection math untouched, GPU search →
   exact CPU re-score → acceptance gates → TV parity audit.

**Non-goals**
- No PnL-based optimization objective (economics live in the calibration layer,
  never in the search objective — anti-overfit separation of concerns).
- No options/IV data. No VIX (HAR core only; VIX may be a later overlay).
- No change to ANY detector computation (`detector.py`, `indicators.py` detection
  functions, GPU evaluator) — the 2026-06-10 parity PASS state is frozen.

---

## 1. Pivot span N\* (the continuous quality scale)

### 1.1 Definition
For a bar `i` with `low[i]` (resp. `high[i]`), the **span** `N*(i)` is the largest
integer `N` such that `low[i]` is the strict-tie-collapsed minimum of the centered
window `[i−N, i+N]` (same construction as `label_pivots`, ambiguous-both-sides
bars excluded, plateau runs collapsed to first bar).

Monotonicity makes this well-defined: extremum of ±N ⟹ extremum of ±M for all
M < N. The old "structural nest 50∧100∧200" ≡ `N* ≥ 200` (verified empirically:
nest count == single-scale N=200 count on SPX and DAX).

### 1.2 Labeling algorithm
- Compute `label_pivots(df, N)` on the grid `N ∈ SPAN_GRID = [20, 30, 40, 50, 70,
  100, 140, 200, 300, 500]` (log-spaced; exact N\* between grid points is not
  needed — `N*` is recorded as the largest grid value passed; weights use the
  grid value). Rationale: O(grid) cost, monotone by construction, and the card
  only ever reports survival at grid thresholds.
- **Grid-floor semantics (F10):** recording "largest grid value passed"
  TRUNCATES — a span-180 pivot counts as 140. Consequence: span weights w(N\*)
  and E[hold] are **grid-floor biased (conservative, downward)**; this is
  accepted and must be stated in the calibration report (the bias is bounded
  by one grid step, ≤ ~43% at the coarsest gap, typically far less in mass
  terms). The **top bucket 500 is "500+"**: a pivot recorded at 500 means
  `N* ≥ 500`, and KM must treat it as **right-censored at the cap**, never as
  an exact span of 500 (same machinery as edge censoring below; the E[hold]
  cap at 500 in §3.4 is consistent with this).
- **Noise floor:** pivots with `N* < 20` do not exist as events (the Scorer-v3
  minor-dip lesson — this floor is deliberate and load-bearing).
- **Right-edge censoring:** `N*` at bar `i` requires `i + N` bars of future data.
  Near the series end, record the **proven lower bound** `N*_lb` and a censored
  flag. Censored pivots: usable for matching at their proven bound, EXCLUDED
  from span grading above their bound (standard right-censoring; the same
  censoring machinery right-censors the §3.2 signal right-span R at the
  edge/cap). The old "dead margin" is thereby
  softened: a bar 60 bars from the end can still be a proven N\*≥50 event.
- Output columns per side: `pivot_span_high`, `pivot_span_low` (0 = no pivot,
  else grid N\*), `pivot_span_censored_{high,low}` (bool).
- `pivot_N100` is RETAINED for backward compatibility (informative-fold filter
  migration path, §2.4).

### 1.3 Span weights
Credit per pivot: `w(N*) = N* / 100` (REFERENCE_N normalization). Empirical
pivot frequency falls ≈ 1/N (measured SPX: 353/138/58/27 at N=20/50/100/200),
so `w ∝ N*` auto-balances total credit mass per log-band — the optimizer cannot
win by farming small swings, structurally. No hand-tuned tier ratios.

### 1.4 Display vocabulary (NOT engine concepts)
- **T1** = `N* ≥ 200` — "a major turning point: the highest/lowest price of
  roughly the surrounding year or more."
- **T2** = `50 ≤ N* < 200` — "a meaningful swing turn: extremum of the
  surrounding quarter, not strong enough to be T1."
- **T3** = `20 ≤ N* < 50` — "a minor swing turn (weeks-scale)."
Used in Pine tooltips, chart badges, and report headings only.

---

## 2. Scorer v5 (the optimization objective)

### 2.1 Matching
- Window: pivot within **`[t−1, t+1]`** of signal bar `t`. No lead bias (delete
  `LEAD_BIAS`); symmetric.
- Hungarian 1:1 assignment (keep the existing implementation; costs = |distance|,
  out-of-window = INF). At equal distance prefer the larger-span pivot.
- A signal matching NO pivot in ±1 is a false positive.

### 2.2 Weighted precision / recall
- `tp_mass = Σ w(N*) over matched pivots`; `total_mass = Σ w(N*) over all pivots`
  in the scored slice (censored pivots contribute `w(N*_lb)`).
- `precision_w = tp_mass / (tp_mass + n_unmatched_signals · w_FP)` with
  `w_FP = w(20) = 0.2`.
- **w_FP exchange-rate semantics (F3):** `w_FP` is the price of a false signal
  in pivot-mass units. At the default 0.2, one false signal costs one
  floor-grade (N\*=20) event, i.e. **ten false signals cost one T1 (N\*=200)
  hit**. This single constant controls the optimizer's spray incentive —
  exactly the gaming axis the firing penalty exists for — so it is NOT a
  hidden constant:
  (a) it is exposed as a named scorer constant `W_FP` in `src/scoring.py`
  with this exchange-rate meaning documented at the definition site;
  (b) the implementation MUST include a **sensitivity test**: optimize a toy
  landscape (synthetic pivots + a spray candidate vs. a selective candidate)
  at `w_FP ∈ {0.2, 0.5, 1.0}` and assert that the acceptance gates
  (`firing_excess`) catch the spray strategy at the LOOSEST setting (0.2);
  (c) the default 0.2 is justified by that test — if the gate fails at 0.2,
  the default moves up until it passes, and the spec value is updated.
- `recall_w = tp_mass / total_mass`.
- Composite, IS/OOS structure, MIN_RATE, GAMMA overfit penalty, PRECISION_EXPONENT,
  RECALL_TARGET sqrt-rescaling: all transfer unchanged in FORM, operating on the
  weighted quantities.
- **Weighted RECALL_TARGET basis (F4):** the sqrt rescale needs an "N" when
  events are weighted mass. Define `N_eff = total_mass` — mass is already
  normalized to REFERENCE_N units by `w = N*/100`, so `N_eff` reads as the
  "equivalent number of N=100 pivots" in the slice. Then:

  ```
  recall_target_eff = RECALL_TARGET · sqrt(REFERENCE_mass / N_eff)
  ```

  with `REFERENCE_mass` pinned ONCE, in code, to the v4-equivalent basis:
  27 SPX T1 highs ⇒ `REFERENCE_mass = 27 · w(200) = 27 · 2.0 = 54.0`. A unit
  test asserts this constant and the rescale formula (so the v4↔v5 anchor can
  never drift silently).
- Everything downstream is unchanged: pooled_fold_score composition,
  cluster weights, block-bootstrap LCB, firing penalty (`firing_excess` now uses
  weighted recall/precision), boundary-pin and stability gates, deflation, PBO.

### 2.3 Scoring-side files touched
`src/scoring.py` (labeling + matching + fold scores), `src/pooled_scoring.py`
(`_stream_stat` mass plumbing), `src/pooled_validation.py` (informative-fold
filter, §2.4), `src/v17_acceptance.py` (firing_excess inputs). The GPU layer is
NOT touched: it produces signals; all scoring is CPU-side in both scorer paths
(verified property of the current architecture).

### 2.4 Informative-fold filter
A fold is label-informative when its OOS slices contain **any pivot mass**
(`total_mass > 0`) for the side — replaces the `pivot_N100 == lbl` count check.
Effect: more folds become informative (T2/T3 events exist where N=100 events
don't). This is intended and must be called out in the run report (fold counts
will change vs v4 runs).

### 2.5 Objective-era bookkeeping
- All v4 LCBs are incomparable. Run slugs gain `scorer: "v5"` metadata.
- Golden baseline: RECAPTURED under v5 (fold scores + LCB change; signal arrays
  do NOT change — detection is untouched; the golden signal sha256s must remain
  IDENTICAL, a strong regression test in itself).

---

## 3. Signal Card (engine + Pine, live and causal)

### 3.1 Feature snapshot at fire time (causal inputs)
Per signal: side, vote count vs required, agreement value & margin over
`min_agreement`, duration counter, drift value, vola pir position, HAR σ
(`calc_har_vol` forecast, annualization-free per-bar σ), bars-since-last-signal.
Stored per historical signal by the calibration pass.

### 3.2 Survival calibration (the quality/probability core)
- For all historical signals of a side (pooled across the run's streams), grade
  retrospectively: matched? span N\* (grid), censored? (this grading feeds
  §3.3 and §3.7; it is NOT the survival-table event variable — see next).
- **Survival event variable = right-span R (re-review R1):** the table is
  fitted on the SAME quantity the live counter tracks. For each historical
  signal, take its candidate pivot bar `i` (§3.5, including the t+1 shift)
  and define `R` = number of bars after bar `i` until a **strictly lower
  low** occurs (HIGH side: strictly higher high); `R` is **right-censored**
  at the series edge and at the grid cap (500), exactly like the live
  counter. EVERY signal enters the table — matched, unmatched, and signals
  near ambiguous-excluded bars alike (`R` is defined per signal from its
  candidate pivot bar, never from a matched pivot), so the table population
  is unambiguous. Fitting on N\* instead is FORBIDDEN: it would
  (a) double-count the left cap (the table would already contain historical
  L-truncation and the live rule would then clamp on L again) and
  (b) mismatch the "survived k" event the live counter observes.
- Fit **Kaplan-Meier survival** `S_R(x) = P(R ≥ x | signal)` on the span grid,
  with right-censored events handled natively. Coarse by design: the grid has
  10 points.
- **Left-span L (F1 — causal, computed at fire):** `N*(i) ≥ k` requires
  `low[i]` to be the minimum of the SYMMETRIC window `[i−k, i+k]`. Right-side
  survival only tracks the right half; the left half is fully observable at
  fire but NOT guaranteed — a signal can fire on a bar that is only the 30-bar
  left minimum, capping achievable N\* at 30 regardless of right-side
  survival. Therefore at fire compute the **proven left-span**
  `L = max{ l : low[i] = min(low[i−l .. i]) }`
  (causal, cheap; Pine-portable as a backward scan with early exit, or via
  `ta.barssince` logic). For the candidate pivot bar of a LOW signal, `i` is
  the bar of the candidate pivot low (§3.5); HIGH side mirrors with `high`/max.
  `L` is clamped at the data's left edge and at the grid cap (500).
- **Conditioning (THE LIVE UPDATE RULE, F1-corrected, R1/R2-pinned):** the
  card quantity is `N*_eff = min(L, R)`. The survival table is fitted on the
  right-span R only, so the left side enters EXCLUSIVELY through the live
  L-clamp (no double-counting):

  ```
  P(N*_eff ≥ x | L, survived k) = 0                        if x > L
                                = S_R(max(x, k)) / S_R(k)  otherwise
  ```

  where `k` = bars survived **counted from the candidate pivot bar `i`**
  (§3.5) — NOT from the fire bar `t`. `i` may shift to `t+1` at widening
  (§3.5 row 2), in which case `L` is recomputed at the new `i` (causal) and
  `k` restarts from the new `i`; calibration (the R fit above) and the live
  counter use the SAME k-origin, eliminating the up-to-2-bar skew of a
  fire-bar count. Implementable in Pine as a lookup into the exported
  S_R(grid) table, one division, and one comparison against L.
- **Off-grid lookup rule (re-review R4):** `S_R(k)` and `S_R(max(x, k))`
  evaluate the table at the **largest grid value ≤ the argument**
  (step-function floor), with `S_R(x) = 1` for `x < 20` (noise floor).
  Engine and Pine MUST share this exact rule; it is part of the §4.2
  "survival lookups exact" parity assertion and is exercised by the §7.3
  L-clamp identity test at non-grid arguments.
- **L as covariate:** L is the single most informative causal feature for span
  and costs nothing — it is the FIRST covariate candidate for the optional
  split below (ahead of vote-margin).
- Optional single covariate split (only if sample permits, gate at n≥60 per
  branch): L above/below median (preferred) or vote-margin above/below median
  → two survival tables. NO richer models (~10²-event regime; the spec
  hard-caps model complexity).
- **Confidence bands (F6):** signals cluster by stream/era (overlapping
  swings, pooled assets), so Greenwood/binomial CIs on the pooled KM are too
  tight — false precision. CIs are computed by **cluster bootstrap**: resample
  whole streams with replacement (or contiguous signal-blocks per stream when
  stream count < 5), refit KM per resample, take the 5th/95th percentile
  envelope over ≥200 resamples. Cheap at n≈10² events. The card displays the
  **bootstrap BAND**, never a point percentage — in the engine/report AND in
  Pine: the band tables `S_lo`/`S_hi` per side are exported in the Pine
  calibration block alongside `S_R` (§4.1) so the Pine label renders banded
  probabilities too. The band method is stated in the calibration report and
  the Pine tooltip.

### 3.3 Expected move
- `expected_move = c_side · σ_HAR(t) · sqrt(E[hold at fire])`. **What is
  displayed must be what was fitted (F2):** the regression is fitted on the
  LIVE-COMPUTABLE regressor, NOT on the true span. Fitting on `σ·sqrt(N*)`
  (future info) and then plugging in `sqrt(E[hold])` live would introduce
  systematic bias (Jensen + estimation error, direction unknown).
- **Fit procedure (F2):** regress realized |move to span-end| of historical
  matched signals on `x = σ_HAR(t) · sqrt(E[hold at fire])`, where
  `E[hold at fire]` is produced by the SAME survival table used live (§3.4,
  k=0, with the L-clamp of §3.2). To avoid self-fit (the survival table is
  estimated from the same signals), use a **two-fold split**: split signals
  chronologically into halves A and B; fit the survival table on A to compute
  `E[hold]` regressors for B and vice versa; pool the two out-of-fold
  regressor/response sets and fit one `c_side` by least squares through the
  origin. One scalar per side. The FINAL exported survival table is still
  fitted on all signals (the split exists only to de-bias the c_side fit).
- **Censored-span signals in the c_side fit (re-review R3):** matched
  signals whose realized span is censored (series edge or "500+" cap) have
  no observable |move to span-end| and are **EXCLUDED** from the c_side
  regression. This truncates the longest swings and biases `c_side`
  **downward** (a length bias); the bias MUST be documented in the
  calibration report next to the R² line. A censored-aware fit is a
  permitted later upgrade, not required for v5.
- **Conditional-on-match labeling (re-review R3):** the regression response
  exists only for MATCHED signals, so the displayed expected move is
  `E[|move| | signal matched a real pivot]` — NOT unconditional. The card
  legend and the Pine tooltip MUST label it as conditional on match (e.g.
  "expected move if this is a real turn"); unmatched signals are NOT mixed
  into the response.
- Report fit R² honestly in the calibration report (if R² < 0.1 the card
  shows the pooled median move instead — fallback rule).
- Displayed at the survival-expected horizon (§3.4), with IQR band from the
  empirical residuals.

### 3.4 Expected hold (the span clock)
- `E[hold] = Σ_grid P(N*_eff ≥ x_j | L, survived k) · Δx_j` (discrete survival
  expectation over the §3.2 rule, S_R-based), capped at 500 (consistent with
  the "500+" censor bucket, §1.2/F10). Updated live with k like §3.2 (k
  counted from the candidate pivot bar `i`), INCLUDING the L-clamp (terms with
  `x_j > L` contribute zero) and the §3.2 off-grid floor-lookup rule.

### 3.5 H0 / invalidation (owner-corrected design)
- **Hard stop = candidate pivot low** `= min(low[t−1], low[t])` at fire,
  widened once to include `low[t+1]` when bar t+1 closes (the ±1 wiggle bar).
  Within the pivot's (eventual) window a breach DEFINITIONALLY refutes the
  matched-pivot hypothesis — α≈0, not a statistical stop.
- **Exact widening rule (F8 — the decided semantics):** breach at t+1 is
  evaluated against the FIRE-time (pre-widening) stop and is **close-based**
  at t+1 only; widening applies only if t+1 survived on close. Rationale: a
  lower LOW at bar t+1 is hypothesis-CONSISTENT (within the ±1 matching
  window the candidate pivot may simply be at t+1 — that is what the wiggle
  bar is for), so an intrabar wick below the fire-time stop at t+1 must not
  invalidate; only a close-through does. From t+2 onward the stop is FINAL
  and any intrabar tick below it definitionally refutes the matched-pivot
  hypothesis (§3.5 first bullet).
- **Candidate pivot bar & k-origin (re-review R2):** at fire the candidate
  pivot bar is `i = argmin(low)` over `{t−1, t}` (tie → the EARLIER bar,
  consistent with §1.1 plateau-collapse-to-first); if widening (row 2 below)
  makes `low[t+1]` the new minimum, the candidate pivot bar SHIFTS to
  `i = t+1` and `L` is **recomputed at the new `i`** (a causal backward scan
  from t+1). **Bars-survived is counted from `i`, never from the fire bar
  `t`** — the §3.2 calibration (right-span R) and the live counter share
  this k-origin exactly (otherwise up to 2 bars of skew). Truth table (LOW
  side; HIGH mirrors with highs and `>`):

  | # | Given (state)                  | When (event at bar close)            | Then |
  |---|--------------------------------|--------------------------------------|------|
  | 1 | bar t closes, signal fires     | —                                    | `stop = min(low[t−1], low[t])` (fire-time stop); `i = argmin(low)` over `{t−1, t}`; card active |
  | 2 | active, bar t+1 closes         | `close[t+1] ≥ fire-time stop`        | survived: widen once, `stop = min(stop, low[t+1])`; if `low[t+1]` is the new STRICT minimum, `i = t+1` and `L` is recomputed at the new `i` (tie → `i` keeps the earlier bar); stop is now FINAL |
  | 3 | active, bar t+1 closes         | `close[t+1] < fire-time stop`        | INVALIDATED at t+1 (close-confirmed breach of the PRE-widening stop); no widening; card frozen, grade "stopped" |
  | 4 | active, bar u > t+1 closes     | `low[u] < stop` (final, intrabar)    | INVALIDATED at u; bars-survived freezes at u−i |
  | 5 | active, bar u > t+1 closes     | `low[u] ≥ stop`                      | bars-survived k = u−i; live conditioning per §3.2 |
  | 6 | active, bar u > t closes       | same-side signal RE-FIRES at u       | active card FREEZES at its fire-time values, grade pending (§3.7 grades it when its span resolves); var-block state (stop, L, k counter, intact flag) RESETS to the new signal per row 1; the plotted stop series switches to the new signal's stop on its fire bar |

  The engine implements the IDENTICAL row-6 replacement rule, so the §4.2
  bar-for-bar exact stop parity holds across overlapping same-side signals.
  The parity script (§4.2) must assert the Pine stop column matches the
  engine **bar-for-bar, exactly** — including the row-2/row-3 ordering at
  t+1 (breach check against the PRE-widening stop, then widen) and the
  row-6 switch bar.
- **Plotted stop after invalidation (DECIDED semantic, parity-pinned):**
  the plotted stop series is Pine's natural `plot(intact ? stop : na)`
  evaluated AT BAR CLOSE — after a row-3/row-4 invalidation it is **na/NaN
  on the invalidation bar and every bar after**, until a same-side re-fire
  (row 6) resumes plotting with the new card's stop. The engine's
  `stop_series` implements exactly this (NaN where Pine plots na); the
  Pine v17.5 builder MUST emit exactly this idiom. Pinned by TDD tests
  `test_stop_series_na_after_*` in `tests/test_v17_card.py` so engine and
  Pine cannot diverge at the §4.2 audit.
- **Guarantee expiry:** the stop's protective logic holds for the span the
  pivot eventually proves. Operationally the card displays: "stop valid ~E[hold]
  bars; exit by clock/target beyond that."
- **Disaster stop** (account insurance only, documented as non-informative):
  q95 of matched-hit MAE in σ-units, recomputed under v5 matching (the old
  15.2σ number is superseded and must NOT be reused).
- Risk per trade shown on card: `entry_close − stop` (the wick gap).

### 3.6 Conviction
`conviction = percentile rank of [P(N*_eff≥50|features,k=0) · expected_move(E[hold])]`
among all historical signals of that side. Pure display ranking, 0–100.

### 3.7 Retrospective grading (engine reports only)
Per signal table: fire date/bar, side, matched (±1) y/n, realized span badge
(N\* grid / censored-bound), realized move to span end, realized R-multiple
under the card's own entry/stop/clock rules, MAE/MFE. Plus aggregate: capture
ratio, expectancy, win rate — the R-multiple backtest that grades the card
against reality each run.

---

## 4. Pine v17.5

### 4.1 New file `pine/speculatores_v17_5_signalcard.pine`
Generated by a builder script from the v17 file. Detection engine BYTE-IDENTICAL
(the parity-PASS state). Additions only:
- **Signal Card display:** on each signal, a label/table showing:
  `P(T2+)= a–b% · P(T1)= c–d% | move ±z% | hold ~h bars | stop S` —
  probabilities are rendered as the F6 cluster-bootstrap **band**, never a
  point value, in Pine exactly as in the engine/report (re-review R6: no
  Pine exemption). Driven entirely by an auto-generated
  `// === CALIBRATION BLOCK (generated) ===` of constants (survival table
  `S_R` **plus band tables `S_lo`/`S_hi`** per side, c_side, conviction
  percentile breakpoints, stop rule). The live band applies the §3.2
  conditioning/L-clamp and off-grid floor-lookup rule to `S_lo` and `S_hi`
  separately (a pragmatic envelope, stated in the tooltip).
- **Live-updating scope (F5 — Pine feasibility):** live survival conditioning
  per §3.2 runs ONLY for the **most recent (active) signal per side**. One
  `var` state block per side holds: fire bar index, candidate pivot bar `i`
  (§3.5/R2, incl. the t+1 shift), candidate stop, proven left-span L,
  bars-survived counter (k-origin = `i`), intact flag; on a same-side
  re-fire the whole block resets per §3.5 row 6. Per-bar work is O(1); no
  loops over history, no per-bar updates of past drawing objects (Pine
  label/table object and runtime limits forbid updating every historical
  signal's card). **Historical signals get a FROZEN card** — the values
  computed at fire time, baked into their label by the calibration block —
  plus a **final-grade badge** once their span resolves (matched/missed,
  realized N\* tier). This scope split is stated in the card tooltip.
- **Tooltips** (`tooltip=` on inputs): plain-language T1/T2/T3 definitions
  (§1.4 text), card legend (incl. the R3 conditional-on-match label on the
  expected move), H0 explanation, the F5 live-vs-frozen scope note,
  and the **in-sample honesty disclaimer (F7)**: "probabilities are calibrated
  on this instrument's history; card values shown on historical bars are
  in-sample; only forward bars (after the calibration run date) are
  out-of-sample." The same disclaimer appears in the engine's calibration
  report and the Cell-6 card legend.
- **Version string "17.5"** in the indicator title.
- **Debug/export plots for parity:** card numerics (P(T2+) and P(T1) band
  edges — lo and hi per R6 — expected move, expected hold, stop level)
  exported as data-window plots so the TV CSV export carries them for the
  audit.

### 4.2 Parity protocol for v17.5 (the user's only manual step)
1. Engine generates `pine/speculatores_v17_5_signalcard.pine` with the
   calibration block from the latest run.
2. User pastes into TradingView, applies preset, **exports chart CSV** (same
   flow as the 2026-06-10 audit).
3. `temp/parity_v17_5.py` audits: `signal_high/low` **exact bool equality &
   count diff == 0** (unchanged binding gate); card numeric columns within
   `atol=1e-6`; stop levels exact (incl. across §3.5 row-6 same-side
   re-fire overlaps, treating engine-NaN == Pine-na on post-invalidation
   bars per the §3.5 decided plotting semantic); survival lookups exact
   (table-driven, asserting the §3.2 off-grid step-function-floor rule at
   non-grid arguments).
PASS rule identical in spirit to §3.3 of the v17 plan.

---

## 5. Engine & workbook integration

- `run_v17_gpu` output gains: `scorer: "v5"`, `calibration` (survival tables,
  c_side, stop constants, fit diagnostics), `signal_cards` (per-signal table,
  §3.7), `r_multiple_backtest` summary.
- Cell-6 report extended with: calibration summary, card legend, capture-ratio
  line, and a per-signal table tail (last 10 signals with grades).
- A new builder regenerates the Pine calibration block + v17.5 file from the
  run JSON (`temp/build_pine_v17_5.py`), so optimizer → Pine is one command.
- Workbook: Cell 5 unchanged interface; Cell 6 swaps in the extended report;
  NEW Cell 7 (optional): writes the freshly calibrated v17.5 pine file to Drive
  for download.

---

## 6. L4 memory safety (22.5 GB)

Budget (f64 PIR era):
- PIR per slice ≈ 499 × bars × 8 B (e.g. 2,850-bar IS slice ≈ 11.4 MB).
- Pool sizing: `INDICES` (3 streams × 5 folds × 2 slices) ≈ 0.4 GB. ALL-1D pool
  (~15 streams) ≈ 1.8 GB. Vote/agreement per-candidate tensors: popsize-chunked.
**Requirements:**
1. A `gpu_memory_estimate(folds, popsize)` utility (slices × scales × 8B + per-
   candidate working set) logged at run start; HARD warn > 14 GB.
2. **Deterministic chunk-size formula (F9 — no OOM-retry loops):** the
   candidate chunk size is computed up front, not discovered by failing:

   ```
   chunk = floor(budget_bytes / (n_lanes · max_bars · bytes_per_candidate))
   ```

   with `budget_bytes = 14 GiB` FIXED (leaves ~8.5 GB headroom on the
   22.5 GB L4 for the PIR pool + framework overhead), `n_lanes` = streams ×
   folds × slices in the pool, `max_bars` = longest slice length, and
   `bytes_per_candidate` MEASURED once on a probe batch (expected ≈ 16 B/bar:
   8 vote bools + workspace) and asserted by a unit test against the analytic
   estimate within 2x. `chunk` is clamped to `[1, popsize]`. The formula, its
   inputs, and the resulting chunk are logged at run start. Deterministic ⇒
   testable ⇒ no adaptive OOM-retry logic anywhere.
3. Agreement scale-tiling fallback verified by a test that simulates the ALL-1D
   pool shape and asserts estimated peak < 18 GB at popsize 128 (and that the
   F9 formula yields chunk ≥ 1 there).
4. CI test: `score_pop` on synthetic 15-stream pool with chunking forced — must
   produce byte-identical LCBs to unchunked on a small case.

---

## 7. Acceptance criteria (definition of done)

1. Full pytest green; golden **signal arrays byte-identical** (detection
   untouched); golden fold scores recaptured under v5.
2. Scorer v5 unit tests: ±1 matching truth table, span weights, censoring,
   Hungarian tie-break, weighted firing_excess, informative-fold filter.
3. Calibration tests: KM survival on synthetic censored data vs closed form
   (incl. "500+" cap censoring, F10, fitted on right-span R per §3.2/R1);
   L-clamped live-conditioning identity
   (`= 0 if x > L, else S_R(max(x,k))/S_R(k)`, F1) including the off-grid
   step-function-floor lookup rule at non-grid k and x (R4);
   k-origin test: bars-survived counted from the candidate pivot bar `i`
   incl. the t+1 shift (R2); cluster-bootstrap band
   reproducibility under a fixed seed (F6); c_side two-fold-split regression
   recovery on synthetic data (F2, censored-span exclusion per R3); F3 w_FP
   sensitivity; F4 REFERENCE_mass constant; F8 stop truth table (incl.
   row-6 same-side re-fire replacement, R5); F9 chunk formula; card JSON
   schema.
4. Pine v17.5 file generated, nbformat-style sanity (compiles as text anchors),
   contains calibration block + tooltips + version 17.5; parity script ready.
5. Workbook v5 regenerated and validated; L4 memory tests pass.
6. `plan/IMPLEMENTATION_REPORT_v5.md` with: what changed, pytest counts, the
   exact user TODO (paste v17.5 → export CSV → hand to assistant → small
   optimizer test (INDICES, gens 10) → big run (ALL-1D, gens 50)).

## 8. Known limitations (stated honestly, not hidden)

- ~150–600 weighted events per side pooled: survival tables are coarse (10 grid
  points, wide cluster-bootstrap bands per F6) — the card shows BANDS, not
  precise percentages.
- c_side is a single scalar; expected-move error bars are wide (IQR shown).
- HIGH side remains monitor-grade regardless of scorer (event ceiling).
- Censoring near the live edge means recent signals' grades finalize slowly.
- The R-multiple backtest ignores costs/slippage (documented).

---

## 9. Migration checklist v4 → v5 (F11)

Work through in order; each item is a discrete, verifiable step.

**Code & metadata**
- [ ] `src/scoring.py`: v5 labeling/matching/weights behind `scorer: "v5"`;
      `W_FP`, `REFERENCE_mass`, `SPAN_GRID` as named module constants.
- [ ] Run-JSON `trace` fields gain `scorer_version: "v5"` (and the
      calibration block hash) so every artifact is era-attributable.
- [ ] Run slugs gain `scorer: "v5"` metadata (§2.5).
- [ ] `boundary_pinned` / acceptance-gate TEXT: audit all v4-language
      references ("pivot_N100", "lead window", "LEAD_BIAS", tier-ratio
      wording) in `src/v17_acceptance.py`, `src/pooled_validation.py`
      docstrings/log messages and report templates; update to v5 vocabulary
      (span mass, ±1 direct hit). Detection-side files are FROZEN — text
      updates there are forbidden; only scorer/report-side files change.
- [ ] Delete `LEAD_BIAS` and the `pivot_N100 == lbl` informative-fold check
      from the SCORER path; `pivot_N100` column itself is retained (§1.2)
      until the workbook migration completes.

**Tests: re-pin vs delete**
- [ ] RE-PIN (same test, new expected values): golden fold scores, golden
      LCBs, pooled-score composition tests, firing_excess thresholds —
      recapture via `temp/capture_baseline.py` AFTER the v5 scorer lands.
- [ ] UNCHANGED (must pass byte-identical, hard CI gate): golden **signal
      array sha256 assertions** in `results/diag/` — any signal-byte change
      is a CRITICAL bug, not a re-pin.
- [ ] DELETE (concept gone in v5): lead-window matching tests, `LEAD_BIAS`
      tests, count-based informative-fold-filter tests. Each deletion is
      replaced by its v5 counterpart from §7.2 (±1 truth table, mass filter).
- [ ] NEW: F3 w_FP sensitivity test, F4 REFERENCE_mass constant test, F8
      stop truth-table test (incl. row-6 re-fire, R5), F9 chunk-formula test,
      KM/censoring tests on right-span R (§7.3, R1), R2 k-origin test, R4
      off-grid floor-lookup test.

**Reports & docs**
- [ ] Run report calls out the fold-count change vs v4 (§2.4) and the v4-LCB
      incomparability (§2.5).
- [ ] Calibration report includes: bootstrap-band method (F6), grid-floor
      bias note (F10), c_side two-fold fit + R² (F2), in-sample disclaimer
      (F7).
- [ ] `plan/IMPLEMENTATION_REPORT_v5.md` records which tests were re-pinned
      vs deleted (this checklist, filled in).
