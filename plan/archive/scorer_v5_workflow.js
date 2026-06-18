export const meta = {
  name: 'scorer-v5-signal-card',
  description: 'Build Scorer v5 (continuous pivot-span objective) + Signal Card (survival calibration, HAR expected move, pivot-low H0) + Pine v17.5 + parity harness + workbook v5. Spec-fix -> re-review -> plan -> TDD implement -> verify -> report.',
  phases: [
    { title: 'Spec-Fix', detail: 'apply the 11 panel findings to the spec, then two-lens re-review gate' },
    { title: 'Plan', detail: 'implementation plan + plan check' },
    { title: 'Scorer', detail: 'span labeling, ±1 weighted matching, weighted objective (TDD)' },
    { title: 'Card', detail: 'KM survival + clamps, expected move/hold, stop truth table, grading (TDD)' },
    { title: 'Engine', detail: 'run_v17_gpu integration, memory guard, golden recapture' },
    { title: 'Pine+Workbook', detail: 'v17.5 builder + parity script + workbook v5' },
    { title: 'Verify', detail: 'full suite + adversarial lenses + repair' },
    { title: 'Report', detail: 'freeze checks + IMPLEMENTATION_REPORT_v5.md + user TODO' },
  ],
}

const RULES = `Repo root is the working dir; use relative paths. READ FIRST:
- plan/scorer-v5-signal-card-spec.md   (the spec — being fixed in phase 1; later phases read the FIXED version)
- plan/scorer-v5-spec-review.md        (panel findings F1-F11)
HARD RULES:
- FROZEN (parity-validated 2026-06-10 — NEVER modify): src/detector.py, src/v17_fastdetector.py,
  src/v17_gpu/** , src/parity.py, src/universe.py, src/search_space.py, src/speculatores145.py,
  src/validation.py, existing pine/*.pine files, and ALL existing functions in src/indicators.py
  (additive-only there). Detection math is sacred; v5 is scorer/calibration/report-side only.
- EDITABLE: src/scoring.py, src/pooled_scoring.py, src/pooled_validation.py, src/v17_acceptance.py,
  src/v17_runner.py, NEW src/v17_card/ package, tests/, temp/ builders, NEW pine v17.5 file,
  temp/build_h100_notebook.py + h100_v17_gpu.ipynb.
- GOLDEN INVARIANT: results/diag/golden signal arrays must stay BYTE-IDENTICAL (detection untouched);
  fold scores/LCB are recaptured under v5 (run temp/capture_baseline.py once scorer lands; the signal
  sha256 assertions must pass UNCHANGED — treat any signal-byte change as a CRITICAL bug).
- TDD: write the test first, run \`python -m pytest <file> -q\`, loop to green. Never weaken an assertion.
- NO git commits. Match repo style (logging, type hints, <=~400-line files).`

const REPORT = {
  type: 'object',
  properties: {
    task: { type: 'string' },
    files_written: { type: 'array', items: { type: 'string' } },
    files_modified: { type: 'array', items: { type: 'string' } },
    pytest_status: { type: 'string', enum: ['green', 'red', 'not-run', 'blocked'] },
    pytest_summary: { type: 'string' },
    notes: { type: 'string' },
    blockers: { type: 'array', items: { type: 'string' } },
  },
  required: ['task', 'pytest_status', 'pytest_summary'],
  additionalProperties: false,
}
const VERDICT = {
  type: 'object',
  properties: {
    lens: { type: 'string' },
    verdict: { type: 'string', enum: ['sound', 'needs-fix', 'fatal'] },
    findings: { type: 'array', items: { type: 'string' } },
    must_fix: { type: 'array', items: { type: 'string' } },
  },
  required: ['lens', 'verdict'],
  additionalProperties: false,
}
const PLAN_SCHEMA = {
  type: 'object',
  properties: {
    plan_path: { type: 'string' },
    task_count: { type: 'integer' },
    risks: { type: 'array', items: { type: 'string' } },
    summary: { type: 'string' },
  },
  required: ['plan_path', 'summary'],
  additionalProperties: false,
}
const FINAL = {
  type: 'object',
  properties: {
    full_pytest_status: { type: 'string', enum: ['green', 'red', 'partial'] },
    full_pytest_summary: { type: 'string' },
    frozen_untouched: { type: 'boolean' },
    golden_signals_identical: { type: 'boolean' },
    report_path: { type: 'string' },
    user_todo: { type: 'string' },
    outstanding: { type: 'array', items: { type: 'string' } },
    executive_summary: { type: 'string' },
  },
  required: ['full_pytest_status', 'frozen_untouched', 'golden_signals_identical', 'executive_summary'],
  additionalProperties: false,
}
const brief = (r) => (r ? { task: r.task, pytest: r.pytest_status, blockers: r.blockers || [] } : null)

// ---------------------------------------------------------------------------
phase('Spec-Fix')
const specFix = await agent(`${RULES}

TASK: Apply ALL 11 findings from plan/scorer-v5-spec-review.md to
plan/scorer-v5-signal-card-spec.md (edit in place; bump Status to "FIXED v2,
post-panel"). Priority order F1,F2,F4 -> F5,F8 -> F3,F6 -> F9 -> F7,F10,F11.
Each fix must be CONCRETE in the spec: F1 left-span L definition + clamp
formula; F2 live-regressor fit w/ two-fold split; F3 w_FP rationale + the
sensitivity-test requirement; F4 N_eff/REFERENCE_mass definition; F5 active-
signal-only live updating + frozen historical labels; F6 cluster-bootstrap
bands; F8 the stop Given/When/Then truth table; F9 the deterministic chunk
formula; F10 grid-floor + 500+ censor semantics; F11 a migration checklist
section. Return the report (files_modified must include the spec).`,
  { label: 'spec:fix', phase: 'Spec-Fix', schema: REPORT })

const reReview = (await parallel([
  () => agent(`${RULES}

TASK: RE-REVIEW plan/scorer-v5-signal-card-spec.md (post-fix) through the
STATISTICS + CAUSALITY lens. Verify F1 (left-span clamp is mathematically
correct: P(N*>=x | L, survived k) = 0 for x>L else S(max(x,k))/S(k), with the
left side fully observable at fire), F2 (fit-what-you-display), F4 (N_eff
basis), F6 (cluster bands), F10 (censor-at-cap). Hunt NEW lookahead leaks in
every live formula. verdict sound|needs-fix|fatal + must_fix.`,
    { label: 'rereview:stats', phase: 'Spec-Fix', schema: VERDICT }),
  () => agent(`${RULES}

TASK: RE-REVIEW plan/scorer-v5-signal-card-spec.md (post-fix) through the
PINE-PORTABILITY + OPS lens. Verify F5 (active-signal-only updating is
implementable with one var-block; frozen labels bounded), F8 (stop truth table
exact + parity-testable), F9 (chunk formula deterministic, no OOM-retry), and
that EVERY live card formula is expressible in Pine v6 (table lookups, sqrt,
division, barssince — no loops over history except the bounded left-span scan,
which must state its max depth). verdict sound|needs-fix|fatal + must_fix.`,
    { label: 'rereview:pine-ops', phase: 'Spec-Fix', schema: VERDICT }),
])).filter(Boolean)

const mustFix = reReview.flatMap(r => (r.verdict !== 'sound' ? (r.must_fix || []) : []))
if (mustFix.length) {
  log(`re-review demands ${mustFix.length} fixes — one repair round`)
  await agent(`${RULES}

TASK: Repair plan/scorer-v5-signal-card-spec.md per these re-review findings,
precisely and minimally: ${JSON.stringify(mustFix)}`,
    { label: 'spec:repair', phase: 'Spec-Fix', schema: REPORT })
}
log(`Spec-Fix done: re-review verdicts = ${reReview.map(r => r.verdict).join(', ')}`)

// ---------------------------------------------------------------------------
phase('Plan')
const plan = await agent(`${RULES}

TASK: Write the implementation plan to plan/scorer-v5-impl-plan.md from the
FIXED spec. Break into ordered tasks with: files touched, the TEST written
first for each, dependencies, and which spec section each implements. Cover:
(1) span labeling (SPAN_GRID, censoring, left-span L, pivot_span_* columns,
keep pivot_N100), (2) ±1 weighted Hungarian + weighted precision/recall +
w_FP + firing_excess inputs + informative-fold filter + N_eff basis,
(3) src/v17_card/ package: KM survival (censored, cluster-bootstrap bands),
conditioning w/ L-clamp, c_side two-fold fit, E[hold], conviction, stop truth
table, retrospective grading + R-multiple backtest, (4) run_v17_gpu output
fields + memory estimator/chunk formula util + golden recapture step,
(5) temp/build_pine_v17_5.py generating pine/speculatores_v17_5_signalcard.pine
from the v17 file + calibration JSON (card display per F5, tooltips, version
string, parity export plots) + temp/parity_v17_5.py, (6) workbook v5 (cell 6
extended report, new cell 7 writes the calibrated pine to Drive) via
temp/build_h100_notebook.py, (7) test migration checklist execution. Return
plan_path + summary.`,
  { label: 'plan:write', phase: 'Plan', schema: PLAN_SCHEMA })

const planCheck = await agent(`${RULES}

TASK: Goal-backward check of plan/scorer-v5-impl-plan.md against the FIXED
spec's acceptance criteria (§7). Does every criterion map to a concrete task
with a test? Are FROZEN files kept out of every task? Is golden-signal
byte-identity asserted? verdict sound|needs-fix|fatal + must_fix.`,
  { label: 'plan:check', phase: 'Plan', schema: VERDICT })
if (planCheck && planCheck.verdict !== 'sound' && (planCheck.must_fix || []).length) {
  await agent(`${RULES}

TASK: Repair plan/scorer-v5-impl-plan.md per: ${JSON.stringify(planCheck.must_fix)}`,
    { label: 'plan:repair', phase: 'Plan', schema: REPORT })
}

// ---------------------------------------------------------------------------
phase('Scorer')
const scorer = await agent(`${RULES}

TASK: Implement plan/scorer-v5-impl-plan.md items (1)+(2) — the Scorer v5 core
in src/scoring.py + src/pooled_scoring.py + src/pooled_validation.py +
src/v17_acceptance.py per the FIXED spec §1-§2. TDD: write
tests/test_scorer_v5.py FIRST (matching truth table incl. ±1 edges and
larger-span tie-break, span weights + grid-floor, censoring lb mass, weighted
precision/recall ranges, w_FP sensitivity test w/ firing-gate catch, N_eff
recall basis, informative-fold filter on mass). Keep pivot_N100 emitted.
Run the affected existing tests too (test_pooled_scoring, test_pooled_validation,
test_v17_acceptance*) and re-pin what the migration checklist says to re-pin —
NEVER re-pin parity/golden signal assertions. Loop to green. Report.`,
  { label: 'build:scorer', phase: 'Scorer', schema: REPORT })

const scorerVerify = await agent(`${RULES}

TASK: Adversarially verify the Scorer v5 build (prior: ${JSON.stringify(brief(scorer))}).
Re-run its tests + the full scoring-related suite. Attack: degenerate slices
(no pivots, all-censored), plateau dedup interaction with span labeling,
Hungarian 1:1 under multiple signals near one pivot, mass conservation
(tp_mass <= total_mass always), w_FP spray exploit actually caught by
firing_excess. verdict + must_fix.`,
  { label: 'verify:scorer', phase: 'Scorer', agentType: 'code-reviewer', schema: VERDICT })
if (scorerVerify && scorerVerify.verdict !== 'sound' && (scorerVerify.must_fix || []).length) {
  await agent(`${RULES}

TASK: Repair Scorer v5 per: ${JSON.stringify(scorerVerify.must_fix)}. Re-run to green.`,
    { label: 'repair:scorer', phase: 'Scorer', schema: REPORT })
}

// ---------------------------------------------------------------------------
phase('Card')
const card = await agent(`${RULES}

TASK: Implement plan item (3): NEW src/v17_card/ package per FIXED spec §3
(KM survival w/ right-censoring + cluster-bootstrap bands; conditioning with
the F1 left-span clamp; c_side two-fold live-regressor fit w/ R^2 fallback;
discrete-survival E[hold]; conviction percentile; the F8 stop truth table as
a pure function; retrospective grading + R-multiple backtest). TDD FIRST:
tests/test_v17_card.py — KM vs closed-form on synthetic censored data;
conditioning identity + L-clamp (P=0 above L); c_side recovery on synthetic;
stop truth-table cases verbatim from the spec; grading on a constructed
series; backtest accounting identity (sum of R-multiples consistent with
equity delta). Loop to green. Report.`,
  { label: 'build:card', phase: 'Card', schema: REPORT })

const cardVerify = await agent(`${RULES}

TASK: Adversarially verify src/v17_card (prior: ${JSON.stringify(brief(card))}).
Attack lenses: LOOKAHEAD (does any live-card function read data after the
fire bar except the explicitly-modeled survival counter?), numerical edges
(S(k)=0 division, empty signal sets, all-censored), Pine-expressibility (is
every live formula a table lookup / sqrt / division / bounded scan?). Re-run
tests. verdict + must_fix.`,
  { label: 'verify:card', phase: 'Card', agentType: 'code-reviewer', schema: VERDICT })
if (cardVerify && cardVerify.verdict !== 'sound' && (cardVerify.must_fix || []).length) {
  await agent(`${RULES}

TASK: Repair src/v17_card per: ${JSON.stringify(cardVerify.must_fix)}. Re-run to green.`,
    { label: 'repair:card', phase: 'Card', schema: REPORT })
}

// ---------------------------------------------------------------------------
phase('Engine')
const engine = await agent(`${RULES}

TASK: Implement plan item (4): src/v17_runner.py emits scorer:"v5",
calibration object, signal_cards table, r_multiple_backtest summary; NEW
memory util (estimate + the F9 deterministic chunk formula) wired into the GPU
scorer call path WITHOUT modifying src/v17_gpu internals (pass chunk size in
via existing knobs or wrap at the runner level; if a knob is missing, add it
to the RUNNER side only and document); L4 tests per spec §6 (ALL-1D-shape
estimate < 18GB at popsize 128; forced-chunk equivalence on a small case).
Then RECAPTURE golden via temp/capture_baseline.py and run
tests/test_parity_golden.py — golden SIGNAL sha256s must be UNCHANGED vs git
HEAD (verify with git diff on the npz files: if signal arrays changed, STOP
and report CRITICAL). Loop to green. Report.`,
  { label: 'build:engine', phase: 'Engine', schema: REPORT })

// ---------------------------------------------------------------------------
phase('Pine+Workbook')
const pine = await agent(`${RULES}

TASK: Implement plan items (5)+(6): temp/build_pine_v17_5.py generates
pine/speculatores_v17_5_signalcard.pine FROM pine/speculatores_v17_presets_gold.pine
(detection lines byte-identical — assert the diff touches only: title w/
"V17.5", the appended CALIBRATION BLOCK constants, the card display block
(active-signal live per F5, frozen historical labels), input tooltips w/ the
plain-language T1/T2/T3 text from spec §1.4, and parity export plots for card
numerics incl. stop level). Builder consumes a calibration JSON (write a
sample one from synthetic/test calibration so the file builds TODAY; the real
one comes from the user's next run). Also temp/parity_v17_5.py (extends the
2026-06-10 audit: signals exact + count diff 0; card numeric columns atol
1e-6; stop level exact). Then workbook: update temp/build_h100_notebook.py
(cell 6 extended report incl. calibration + card tail + capture ratio; NEW
cell 7 writing the freshly built v17.5 pine to RESULTS_DIR on Drive),
regenerate h100_v17_gpu.ipynb, nbformat-validate. Tests: a build test that
asserts the pine diff-surface claim programmatically. Loop to green. Report.`,
  { label: 'build:pine+wb', phase: 'Pine+Workbook', schema: REPORT })

// ---------------------------------------------------------------------------
phase('Verify')
const fullVerify = (await parallel([
  () => agent(`${RULES}

TASK: Run the FULL test suite (python -m pytest -q, capture counts). Then
attack END-TO-END: tiny run on local data (SPX+DAX raw CSVs, era_kw small)
through run_v17_gpu device="cpu" with the v5 scorer — assert it completes,
emits calibration + cards + backtest, memory estimator logs, and the winner
re-score path still enforces |gpu-cpu|<tol. Report verdict + must_fix.`,
    { label: 'verify:e2e', phase: 'Verify', schema: VERDICT }),
  () => agent(`${RULES}

TASK: FREEZE AUDIT: git diff --name-only vs HEAD — assert NO frozen file
changed (src/detector.py, src/v17_fastdetector.py, src/v17_gpu/**, src/parity.py,
src/universe.py, src/search_space.py, src/speculatores145.py, src/validation.py,
existing pine files); indicators.py diff must be ADDITIVE-only (no existing
function bodies changed — verify by diff inspection). Golden npz signal arrays
unchanged vs HEAD. Report verdict + must_fix with exact offending files.`,
    { label: 'verify:freeze', phase: 'Verify', schema: VERDICT }),
])).filter(Boolean)

const vFix = fullVerify.flatMap(v => (v.verdict !== 'sound' ? (v.must_fix || []) : []))
if (vFix.length) {
  await agent(`${RULES}

TASK: Repair per final-verify findings, re-run affected tests + freeze audit to
green: ${JSON.stringify(vFix)}`,
    { label: 'repair:final', phase: 'Verify', schema: REPORT })
}

// ---------------------------------------------------------------------------
phase('Report')
const rollup = JSON.stringify({
  specFix: brief(specFix), plan: plan && plan.summary, scorer: brief(scorer),
  card: brief(card), engine: brief(engine), pine: brief(pine),
})
const final = await agent(`${RULES}

TASK: Final gate + handoff. (1) python -m pytest -q full suite, exact counts.
(2) Freeze audit one more time (frozen files + additive-only indicators.py +
golden signal bytes). (3) Write plan/IMPLEMENTATION_REPORT_v5.md: changes,
counts, calibration sample summary, the EXACT USER TODO list:
  a. (assistant commits/pushes; user flips repo public if private)
  b. fresh Colab session, workbook v5, run Cells 1-5 SMALL TEST: GROUPS=INDICES,
     GENERATIONS=10, POPSIZE=128 (L4-safe; memory estimator prints budget)
  c. Cell 6 report review with assistant -> Cell 7 writes calibrated v17.5 pine
  d. paste v17.5 into TradingView, apply preset, EXPORT chart CSV
  e. hand CSV to assistant -> temp/parity_v17_5.py audit (signals exact + card
     numerics 1e-6) -> on PASS: the very big run (GROUPS=INDICES,COMMODITIES,FX,WORLD_ETF,
     GENERATIONS=50, SOBOL_N=256)
(4) Return user_todo as a compact string + executive_summary.
Prior phase rollup: ${rollup}`,
  { label: 'final:report', phase: 'Report', schema: FINAL })

return {
  pytest: final ? final.full_pytest_status : 'unknown',
  frozen_untouched: final ? final.frozen_untouched : null,
  golden_signals_identical: final ? final.golden_signals_identical : null,
  report: final ? final.report_path : 'plan/IMPLEMENTATION_REPORT_v5.md',
  user_todo: final ? final.user_todo : null,
  outstanding: final ? final.outstanding : null,
  summary: final ? final.executive_summary : null,
}
