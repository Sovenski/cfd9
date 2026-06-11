# Spec Panel Review — Scorer v5 + Signal Card (plan/scorer-v5-signal-card-spec.md)

Mode: critique · Panel: Wiegers (requirements), Nygard (failure modes/ops),
Fowler (design), Crispin (testing), Adzic (examples) + domain lenses
(statistics, causality, Pine portability, GPU memory).
Verdict: **sound core, 6 must-fix findings, 5 should-fix** — spec is
implementable after fixes. No fatal flaws.

---

## CRITICAL (must fix before implementation)

### F1 — Left-span conflation in survival conditioning (statistics/causality) — Nygard/Wiegers
§3.2 defines live updating as "pivot low unbroken k bars ⇒ condition on N\*≥k".
WRONG AS STATED: `N*(i) ≥ k` requires `low[i]` to be the minimum of the
SYMMETRIC window `[i−k, i+k]`. The right side is what survival tracks; the
LEFT side is fully observable at fire but NOT guaranteed — a signal can fire
on a bar that is only the 30-bar left minimum, capping achievable N\* at 30
regardless of right-side survival.
**Fix:** at fire compute the proven left-span
`L = max{l : low[i] = min(low[i−l..i])}` (causal, cheap, Pine-portable as a
backward scan with early exit / or via barssince logic). Define the card
quantity as `N*_eff = min(L, right-survival)`; the survival table is fitted on
N\* but conditioning must clamp: `P(N* ≥ x | L, survived k) = 0 for x > L`,
else `S(max(x,k))/S(k)`. L is also a calibration covariate candidate (it is
the single most informative causal feature for span and costs nothing).

### F2 — Calibration/live regressor mismatch in expected move (statistics) — Wiegers
§3.3 fits `|move|` on `σ·sqrt(N*)` (true span, future info — fine offline) but
the live card plugs in `sqrt(E[hold])` (estimated). Jensen + estimation error
⇒ systematic bias, direction unknown.
**Fix:** fit the regression on the LIVE-COMPUTABLE quantity: regress realized
|move| on `σ_HAR(t) · sqrt(E[hold at fire])` where E[hold at fire] is produced
by the same survival table used live (leave-one-out or simple two-fold split
to avoid self-fit). What is displayed must be what was fitted.

### F3 — FP exchange rate is an unexamined free parameter (objective design) — Fowler
§2.2 sets `w_FP = w(20) = 0.2`: ten false signals cost one T1 hit. This single
constant controls the optimizer's spray incentive — exactly the gaming axis the
firing penalty exists for.
**Fix:** (a) document the exchange-rate meaning explicitly; (b) add a
sensitivity test to the implementation: optimize a toy landscape at
w_FP ∈ {0.2, 0.5, 1.0} and assert the acceptance gates (firing_excess) catch
the spray strategy at the loosest setting; (c) expose w_FP as a scorer
constant with the chosen default justified by that test.

### F4 — Weighted RECALL_TARGET basis under-specified (internal consistency) — Wiegers
§2.2 says the sqrt rescale "transfers in form" but never defines N when events
are weighted mass.
**Fix:** define `N_eff = total_mass` (mass already normalized to REFERENCE_N
units by w = N*/100, so N_eff is "equivalent number of N=100 pivots");
`recall_target_eff = RECALL_TARGET · sqrt(REFERENCE_mass / N_eff)` with
REFERENCE_mass pinned to the v4-equivalent (27 SPX T1 highs ⇒ mass 54…/define
once, in code, with a test).

### F5 — Pine live-updating scope is infeasible as written (Pine portability) — Adzic
§4.1 implies every historical signal's card updates live. Pine labels/tables
for ALL past signals with per-bar updates = unbounded drawing objects and
per-bar loops over history — will hit Pine object/runtime limits.
**Fix:** live conditioning ONLY for the most recent (active) signal per side
(one `var` state block: fire bar, candidate stop, L, bars-survived).
Historical signals get a FROZEN card (values at fire) in their label +
final-grade badge once span resolves. State this in the spec and the tooltip.

### F6 — KM confidence intervals invalid under clustering (statistics) — Crispin
Signals cluster by stream/era (overlapping swings, pooled assets); Greenwood
CIs on pooled KM will be too tight, and the card would display false precision.
**Fix:** CIs via cluster bootstrap (resample streams, or signal-blocks per
stream); display the bootstrap band. Cheap (n≈10² events, 200 resamples).
Spec must state the card shows BANDS, and the band method.

## MAJOR (should fix)

### F7 — In-sample card display honesty — Nygard
Constants calibrated on the same history the chart displays ⇒ historical card
values are in-sample. Add to spec + Pine tooltip: "probabilities are
calibrated on this instrument's history; values on historical bars are
in-sample; only forward bars are out-of-sample."

### F8 — Stop-widening rule needs an exact truth table — Adzic
§3.5 "widened once to include low[t+1]" — specify exactly: at fire (bar t
close): stop = min(low[t−1], low[t]). At t+1 close: stop = min(stop, low[t+1])
IFF no breach occurred intrabar at t+1 (breach check uses the PRE-widening
stop? or post?). Decide: breach at t+1 evaluated against the FIRE-time stop;
widening applies only if t+1 survived. Write the Given/When/Then table into
the spec; parity script must assert the Pine stop column matches the engine
bar-for-bar.

### F9 — Chunk-size formula, not just warnings (L4 memory) — Nygard
§6 warns at 14 GB and "auto-reduces" at 18 GB but gives no formula.
Fix: `chunk = floor(budget_bytes / (n_lanes · max_bars · bytes_per_candidate))`
with bytes_per_candidate measured once (8 vote bools + workspace ≈ 16 B/bar);
budget = 14 GB fixed. Deterministic, testable, no OOM-retry loops.

### F10 — Survival grid edge semantics — Crispin
N\* recorded as "largest grid value passed" truncates (a span-180 pivot counts
as 140). Acceptable, but then E[hold] and weights are grid-floor biased ⇒
document; and the top bucket "500" must be treated as "500+" (censored at cap)
in KM, not as an exact value.

### F11 — Migration/trace completeness — Fowler
Spec omits: updating `boundary_pinned`/acceptance text references to v4
language, run-JSON `trace` fields gaining scorer version, and which existing
tests get re-pinned vs deleted. Add a migration checklist section.

## CONSENSUS
- The N\*/survival core is the right design; F1 is the only finding that
  touches its correctness, and it has a clean fix (left-span clamp).
- Detection-path freeze + golden-signal-bytes-identical invariant is the
  strongest safety property in the spec — keep it as a hard CI gate.
- Sample-size honesty (§8) is adequate once F6's bands are in.

## PRIORITY ORDER FOR THE FIX PASS
F1, F2, F4 (correctness) → F5, F8 (Pine feasibility/exactness) → F3, F6
(robustness) → F9 (ops) → F7, F10, F11 (documentation/completeness).
