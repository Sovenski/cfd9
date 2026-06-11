export const meta = {
  name: 'gpu-refactor-implementation',
  description: 'Build the full GPU/H100 refactor in one swoop: TDD-built, parity-test-gated, adversarially verified. CPU-builds + CPU-verifies everything; emits a Colab H100 confirmation script. Reads plan/gpu-refactor-build-spec.md.',
  phases: [
    { title: 'Setup', detail: 'env check + install torch/scipy/cma + confirm baseline pytest green' },
    { title: 'Baseline', detail: 'Phase 0 golden capture + parity harness' },
    { title: 'PIR-Spike', detail: 'Phase 0.5 CPU spike — selects trust-kernel vs noisy-ranker branch' },
    { title: 'Optimizer', detail: 'Phase 1 Sobol+sep-CMA-ES on existing scorer — prerequisite that produces the wide candidate batch the GPU scores (coordinate ascent cannot feed a GPU); no new parity surface' },
    { title: 'Validation', detail: 'Phase 2 CPCV + dead-margin fix + PBO/deflation (default-off)' },
    { title: 'GPU-Evaluator', detail: 'Phase 3 torch score_pop, 4 gated submodules + adversarial parity loop' },
    { title: 'Integration', detail: 'Phase 4 run_v17_gpu pipeline + Colab H100 validation script' },
    { title: 'Report', detail: 'final pytest + oracle-untouched check + IMPLEMENTATION_REPORT.md' },
  ],
}

// ---------------------------------------------------------------------------
// Shared context + schemas
// ---------------------------------------------------------------------------
const SPEC = `Working dir is the repo root; use RELATIVE paths. Before coding, READ:
- plan/gpu-refactor-build-spec.md  (the executable spec — esp §0 GROUND RULES + the §N for your task)
- plan/gpu-refactor-plan.md        (design rationale, as needed)
HARD RULES (§0): never modify the trust-root/oracle files (src/detector.py, src/indicators.py,
src/v17_fastdetector.py, src/parity.py, src/scoring.py, src/pooled_scoring.py, src/validation.py,
src/search_space.py, src/speculatores145.py, src/universe.py, pine/*). Only src/v17_runner.py,
src/pooled_validation.py, src/v17_acceptance.py may be edited (additively, no behavior change off-by-default).
TDD: write the test FIRST, run \`python -m pytest <file> -q\`, loop until GREEN. Never relax a parity
assertion to pass. No git commits. Match repo style (logging, type hints, dataclasses, files <=~400 lines).`

const REPORT_SCHEMA = {
  type: 'object',
  properties: {
    task: { type: 'string' },
    files_written: { type: 'array', items: { type: 'string' } },
    files_modified: { type: 'array', items: { type: 'string' } },
    pytest_status: { type: 'string', enum: ['green', 'red', 'not-run', 'blocked'] },
    pytest_summary: { type: 'string' },
    parity_assertions: { type: 'string', description: 'which §0.6 invariants this task asserts, and the result' },
    blockers: { type: 'array', items: { type: 'string' } },
    notes: { type: 'string' },
  },
  required: ['task', 'pytest_status', 'pytest_summary'],
  additionalProperties: false,
}

const SETUP_SCHEMA = {
  type: 'object',
  properties: {
    env_ready: { type: 'boolean' },
    python_cmd: { type: 'string', description: 'the working python invocation (e.g. python / py / python3)' },
    deps_installed: { type: 'array', items: { type: 'string' } },
    torch_available: { type: 'boolean' },
    baseline_pytest_status: { type: 'string', enum: ['green', 'red', 'not-run'] },
    baseline_pytest_summary: { type: 'string' },
    blockers: { type: 'array', items: { type: 'string' } },
  },
  required: ['env_ready', 'python_cmd', 'torch_available', 'baseline_pytest_status'],
  additionalProperties: false,
}

const SPIKE_SCHEMA = {
  type: 'object',
  properties: {
    exact_match: { type: 'boolean' },
    winning_method: { type: 'string', description: 'cumsum-difference | pandas-faithful | none' },
    max_abs_drift: { type: 'number' },
    signal_flip_rate: { type: 'number' },
    recommended_branch: { type: 'string', enum: ['trust-kernel', 'noisy-ranker'] },
    suggested_topK: { type: 'integer' },
    details: { type: 'string' },
  },
  required: ['exact_match', 'recommended_branch', 'details'],
  additionalProperties: false,
}

const VERIFY_SCHEMA = {
  type: 'object',
  properties: {
    task: { type: 'string' },
    passed: { type: 'boolean' },
    ran_pytest: { type: 'boolean' },
    issues: { type: 'array', items: { type: 'string' } },
    must_fix: { type: 'array', items: { type: 'string' } },
    verdict: { type: 'string', enum: ['sound', 'needs-fix', 'fatal'] },
  },
  required: ['task', 'passed', 'verdict'],
  additionalProperties: false,
}

const SKEPTIC_SCHEMA = {
  type: 'object',
  properties: {
    lens: { type: 'string' },
    broke: { type: 'boolean', description: 'did you find a CONCRETE case where GPU signals differ from the CPU oracle?' },
    failing_case: { type: 'string', description: 'exact params/slice/bar that diverges, reproducible' },
    evidence: { type: 'string' },
  },
  required: ['lens', 'broke'],
  additionalProperties: false,
}

const FINAL_SCHEMA = {
  type: 'object',
  properties: {
    full_pytest_status: { type: 'string', enum: ['green', 'red', 'partial'] },
    full_pytest_summary: { type: 'string' },
    oracle_untouched: { type: 'boolean', description: 'git diff touched ONLY new files + the 3 allowed surgical files' },
    golden_intact: { type: 'boolean' },
    report_path: { type: 'string' },
    pir_branch_taken: { type: 'string' },
    colab_steps: { type: 'string', description: 'exact steps the user runs on a Colab H100' },
    outstanding: { type: 'array', items: { type: 'string' } },
    executive_summary: { type: 'string' },
  },
  required: ['full_pytest_status', 'oracle_untouched', 'golden_intact', 'executive_summary'],
  additionalProperties: false,
}

const verifyOf = (r) => (r ? { task: r.task, pytest_status: r.pytest_status, blockers: r.blockers || [] } : null)

// ===========================================================================
// PHASE: Setup
// ===========================================================================
phase('Setup')
const setup = await agent(`${SPEC}

TASK: Prepare the build environment (build-spec §0.3). Determine the working python command. Confirm
numpy/pandas/optuna present. Install (uv pip install, else pip install): torch (CPU build is fine),
scipy, cma. Confirm \`import torch\` works and report torch version + that CUDA is absent (expected — we
test on CPU). Then run the EXISTING test suite \`python -m pytest -q\` to capture the GREEN starting
baseline (so later regressions are attributable). Report env_ready, the python_cmd, deps installed,
torch availability, and the baseline pytest status/summary. Do NOT modify any source files.`,
  { label: 'setup:env', phase: 'Setup', schema: SETUP_SCHEMA })
log(`Setup: env_ready=${setup && setup.env_ready}, torch=${setup && setup.torch_available}, baseline pytest=${setup && setup.baseline_pytest_status}`)

// ===========================================================================
// PHASE: Baseline (Phase 0)
// ===========================================================================
phase('Baseline')
const baseline = await agent(`${SPEC}

TASK: Build-spec §1 (PHASE 0 — parity harness + golden baseline). Write temp/capture_baseline.py and
tests/test_parity_golden.py exactly as specified; capture golden signal arrays + per-fold scores + final
LCB for SPX and DAX base-gold Params into results/diag/golden/ (seed=42). TDD: write the test first, run
pytest, loop to green. Also record (in the test) the existing FastDetector==SpeculatorDetector
np.array_equal contract and FastPooledScorer≈PooledScorer (abs<=1e-12) contract as the reference.
Return the report.`,
  { label: 'phase0:golden', phase: 'Baseline', schema: REPORT_SCHEMA })
log(`Baseline: pytest=${baseline && baseline.pytest_status}`)

// ===========================================================================
// PHASE: PIR-Spike (Phase 0.5) — selects the GPU branch
// ===========================================================================
phase('PIR-Spike')
const spike = await agent(`${SPEC}

TASK: Build-spec §2 (PHASE 0.5 — PIR byte-identity spike, CPU). Write temp/pir_parity_spike.py. On a real
SPX IS slice (~5.7k bars) across all 499 scales (2..500), build the torch float32 pir_matrix two ways
(cumsum-difference AND pandas-faithful unfold/convolution) on torch.device("cpu"), compare each to
indicators.precompute_matrices(...) float32 via np.array_equal, and measure max_abs_drift + the
signal-flip rate through calc_agreement_fast. RUN it. Decide recommended_branch: "trust-kernel" iff some
method is byte-identical (or flip rate exactly 0), else "noisy-ranker" (and suggest top-K from the flip
rate). Return the structured verdict. This is a MEASUREMENT (no pass/fail gate) that selects the Phase-3
strategy.`,
  { label: 'phase0.5:spike', phase: 'PIR-Spike', schema: SPIKE_SCHEMA })

const BRANCH = (spike && spike.recommended_branch) || 'noisy-ranker'
const branchNote = BRANCH === 'trust-kernel'
  ? `PIR SPIKE RESULT: trust-kernel. The torch PIR is byte-identical to pandas, so Phase-3 parity tests must
be strict np.array_equal on signals (build-spec §0.7). The kernel's signals are authoritative for ranking.`
  : `PIR SPIKE RESULT: noisy-ranker (flip_rate≈${spike ? spike.signal_flip_rate : 'unknown'}). The torch PIR is
NOT byte-identical, so the GPU kernel is a RANKER ONLY. Phase-3 parity tests assert signal_flip_rate <=
the measured threshold (NOT strict equality on PIR-derived votes), and the EXACT CPU SpeculatorDetector
produces every reported/exported number. Size top-K to the flip rate (suggested ${spike ? spike.suggested_topK : 'TBD'}).
Still enforce STRICT np.array_equal for every NON-PIR-derived invariant (P2–P8).`
log(`PIR-Spike: branch=${BRANCH}`)

// ===========================================================================
// PHASE: Optimizer (Phase 1) — build then verify
// ===========================================================================
phase('Optimizer')
const optBuild = await agent(`${SPEC}

TASK: Build-spec §3 (PHASE 1 — batch-hungry optimizer on the EXISTING scorer). Create src/v17_search.py
(BatchOptimizer: Sobol seeding via scipy.stats.qmc.Sobol over active threshold bounds, separable CMA-ES
via the cma lib warm-started at the gold seed, population 256–1024, returns full population + top-K by
penalized LCB, shape params FROZEN). Add a search="cma" route to src/v17_runner.py (keep "ascent" as the
unchanged default). Wire firing_excess into the batched ranking via src/v17_acceptance.py. Write
tests/test_v17_search.py with ALL §3 assertions (Sobol within bounds; shape frozen; winner re-scored by
the REAL PooledScorer matches to <1e-9; determinism; search="ascent" byte-identical to today). TDD to
green. Return the report.`,
  { label: 'phase1:build', phase: 'Optimizer', schema: REPORT_SCHEMA })

const optVerify = await agent(`${SPEC}

TASK: Adversarially VERIFY Phase 1 (build-spec §3). Prior builder report: ${JSON.stringify(verifyOf(optBuild))}.
Re-run \`python -m pytest tests/test_v17_search.py -q\` AND the full suite. Independently confirm: (a) the
search winner's reported LCB equals the EXACT PooledScorer LCB to <1e-9 (this is the no-new-parity-surface
guarantee); (b) run_v17(search="ascent") output is byte-identical to the pre-change behavior (regression);
(c) shape/architecture params never vary within a population. List any must_fix. verdict sound|needs-fix|fatal.`,
  { label: 'phase1:verify', phase: 'Optimizer', agentType: 'code-reviewer', schema: VERIFY_SCHEMA })

if (optVerify && optVerify.passed === false && (optVerify.must_fix || []).length) {
  await agent(`${SPEC}

TASK: Repair Phase 1 per these verifier findings, then re-run tests to green:
${JSON.stringify(optVerify.must_fix)}`,
    { label: 'phase1:repair', phase: 'Optimizer', schema: REPORT_SCHEMA })
}
log(`Optimizer: build=${optBuild && optBuild.pytest_status}, verify=${optVerify && optVerify.verdict}`)

// ===========================================================================
// PHASE: Validation (Phase 2) — build then verify (regression is the key gate)
// ===========================================================================
phase('Validation')
const valBuild = await agent(`${SPEC}

TASK: Build-spec §4 (PHASE 2 — validation fixes + selection-bias guards, parity-orthogonal, default-OFF).
Create src/cpcv.py (purged combinatorial CV, justified on OVERLAP not a leak) and src/overfit_guard.py
(PBO via CSCV + deflation scaled to λ×generations). Add _run_cpcv_loop ALONGSIDE _run_fold_loop in
src/pooled_validation.py (default OFF); mask the dead 200-bar label margins in _stream_stat; add a
per-asset-then-aggregate HIGH diagnostic. Wire deflation into the reported number in src/v17_runner.py
when batched search is used. Write tests/test_cpcv_purge.py with ALL §4 assertions, INCLUDING the
critical regression: with CPCV OFF (default), PooledScorer/run_v17 results are UNCHANGED vs the Phase-0
golden. TDD to green. Return the report.`,
  { label: 'phase2:build', phase: 'Validation', schema: REPORT_SCHEMA })

const valVerify = await agent(`${SPEC}

TASK: Adversarially VERIFY Phase 2 (build-spec §4). Builder report: ${JSON.stringify(verifyOf(valBuild))}.
Re-run tests/test_cpcv_purge.py + tests/test_parity_golden.py + full suite. Confirm: (a) CPCV-OFF default
reproduces the Phase-0 golden bit-for-bit (no silent behavior change); (b) the purge actually prevents
any IS label window from overlapping a test group; (c) HIGH PBO/percentile are ADVISORY only, primary
HIGH summary is event-count + wide interval. must_fix + verdict.`,
  { label: 'phase2:verify', phase: 'Validation', agentType: 'code-reviewer', schema: VERIFY_SCHEMA })

if (valVerify && valVerify.passed === false && (valVerify.must_fix || []).length) {
  await agent(`${SPEC}

TASK: Repair Phase 2 per these findings, re-run to green (esp. the CPCV-OFF golden regression):
${JSON.stringify(valVerify.must_fix)}`,
    { label: 'phase2:repair', phase: 'Validation', schema: REPORT_SCHEMA })
}
log(`Validation: build=${valBuild && valBuild.pytest_status}, verify=${valVerify && valVerify.verdict}`)

// ===========================================================================
// PHASE: GPU-Evaluator (Phase 3) — 4 gated submodules, then adversarial parity loop
// ===========================================================================
phase('GPU-Evaluator')

// §5.1 drift precompute  ->  §5.2 upload/packer  ->  §5.3 features+threshold  ->  §5.4 batched scan
const gpuSubmodules = [
  { id: '5.1', label: 'gpu:drift', desc: 'src/v17_gpu/drift_precompute.py (build-spec §5.1): precomputed per-asset drift array replacing the growing stack. Reproduce P5 exactly (Kmax>=20, None->0.0, min_pivots=max(lookback,2), /(min_pivots-1), max(|start|,1e-9), HIGH-before-LOW, read-before-append). Test np.array_equal of per-bar drift vs the CPU _detect loop, incl. a both-pivots-same-bar case.' },
  { id: '5.2', label: 'gpu:upload', desc: 'src/v17_gpu/upload.py (build-spec §5.2): padded [n_assets,max_bars,...] packer + valid_mask (P7) + length-bucketing + PIR scale-tiling hook (P1 memory). Test pack/unpack round-trip exactness and valid_mask correctness.' },
  { id: '5.3', label: 'gpu:features', desc: 'src/v17_gpu/eval_torch.py Phase-1 features + threshold layer (build-spec §5.3): every Phase-1 array of FastDetector._precompute + the threshold comparisons of FastDetector.signals, obeying P1–P4,P6,P8. Test each intermediate array np.array_equal vs the matching FastDetector attribute on >=3 slices.' },
  { id: '5.4', label: 'gpu:scan', desc: 'src/v17_gpu/eval_torch.py batched Phase-2 scan + score_pop (build-spec §5.4): the stateful loop as a batched bar-loop over (candidate x asset) lanes (carry per P5/P7), returning signals + pooled LCB reusing pooled_fold_score/bootstrap for numeric identity. Write tests/test_v17_gpu_parity.py with the §0.7 GREEN criteria + all edge-case tests. This is the load-bearing gate — HARD FAIL on any mismatch.' },
]

let lastGpu = null
for (const m of gpuSubmodules) {
  lastGpu = await agent(`${SPEC}

${branchNote}

TASK: Build ${m.desc}
Build in the context of the prior submodules already on disk. TDD to green; the parity test for THIS
submodule must pass before you return. Report which §0.6 invariants (P1–P8) you assert and their result.`,
    { label: m.label, phase: 'GPU-Evaluator', schema: REPORT_SCHEMA })
  log(`GPU §${m.id}: pytest=${lastGpu && lastGpu.pytest_status}`)
}

// Adversarial parity loop — bounded rounds; skeptics try to BREAK byte-identity, repair agent fixes.
const LENSES = [
  'read-before-append ordering AND HIGH-pivot-before-LOW-pivot same-bar push (P5)',
  'warm-up NaN region: first ~300 bars where GJR/lr_var/linreg/HAR are NaN -> _to_bool NaN->False (P2/P4)',
  'per-asset segmented reset: cooldown (bars_since) and the pivot ring must NOT leak across asset boundaries; pad bars freeze all carry (P7)',
  'pivot ring truncation at Kmax and the strict comparison operators verbatim (P3/P5)',
]
let parityRound = 0
let unresolved = []
while (parityRound < 3) {
  const skeptics = (await parallel(LENSES.map((L, i) => () =>
    agent(`${SPEC}

${branchNote}

TASK: ADVERSARIAL PARITY ATTACK on the GPU evaluator (src/v17_gpu/), lens: ${L}.
Try HARD to find a CONCRETE (params, slice, bar) where src/v17_gpu score_pop / signals differ from the
exact CPU SpeculatorDetector (run both and diff signal_high/signal_low). Construct adversarial inputs that
stress this lens. If you find a divergence, report broke=true with the exact reproducible failing_case and
evidence. If after genuine effort you cannot break it, broke=false. Do NOT edit code — only investigate/run.`,
      { label: `break:${i}:r${parityRound}`, phase: 'GPU-Evaluator', schema: SKEPTIC_SCHEMA })
  ))).filter(Boolean)

  const breaks = skeptics.filter(s => s.broke)
  if (breaks.length === 0) { unresolved = []; log(`GPU parity: clean after round ${parityRound}`); break }
  unresolved = breaks
  log(`GPU parity round ${parityRound}: ${breaks.length} break(s) found -> repairing`)
  await agent(`${SPEC}

${branchNote}

TASK: REPAIR the GPU evaluator parity defects below, then re-run tests/test_v17_gpu_parity.py to green and
confirm each failing_case now matches the CPU oracle. Fix root causes (ordering, NaN handling, segmented
reset, ring size, operator strictness) — never weaken a parity assertion.
DEFECTS: ${JSON.stringify(breaks)}`,
    { label: `gpu:repair:r${parityRound}`, phase: 'GPU-Evaluator', schema: REPORT_SCHEMA })
  parityRound++
}
if (unresolved.length) log(`GPU parity: ${unresolved.length} unresolved break(s) after ${parityRound} rounds — will surface in final report`)

// ===========================================================================
// PHASE: Integration (Phase 4) — pipeline + Colab handoff
// ===========================================================================
phase('Integration')
const integ = await agent(`${SPEC}

${branchNote}

TASK: Build-spec §6 (PHASE 4 — integration + Colab handoff). Add run_v17_gpu(...) to src/v17_runner.py:
GPU batched search (score_pop) -> CPU finalist re-score with the EXACT SpeculatorDetector+PooledScorer
(top-K sized to the spike flip rate) -> v17_acceptance gates -> per-asset TradingView-export audit hook;
generalize the existing |fast-real|>1e-9 warning into a HARD finalist filter. Write
temp/colab_h100_validation.py: the ~5-min script the USER runs on a Colab H100 (install torch-cuda, run
the PIR spike ON GPU, run tests/test_v17_gpu_parity.py ON GPU, run one tiny end-to-end run_v17_gpu, print
a single PASS/FAIL + the real-hardware signal-flip rate). Add the §6 end-to-end integration test
(2-asset CPU pool; top finalist reported LCB == exact CPU PooledScorer LCB to <1e-9; golden intact). TDD
to green. Return the report.`,
  { label: 'phase4:integration', phase: 'Integration', schema: REPORT_SCHEMA })
log(`Integration: pytest=${integ && integ.pytest_status}`)

// ===========================================================================
// PHASE: Report (Phase 7 — final verification + handoff doc)
// ===========================================================================
phase('Report')
const allReports = {
  setup, baseline, spike: { branch: BRANCH, ...(spike || {}) },
  optimizer: { build: verifyOf(optBuild), verify: optVerify },
  validation: { build: verifyOf(valBuild), verify: valVerify },
  gpu: { submodules: gpuSubmodules.map(m => m.id), parity_rounds: parityRound, unresolved },
  integration: verifyOf(integ),
}

const final = await agent(`${SPEC}

TASK: Build-spec §7 (DEFINITION OF DONE — final verification + handoff). Do ALL:
1. Run the FULL suite \`python -m pytest -q\`; report exact pass/fail counts (never hide a red test).
2. Run \`git diff --name-only\`; confirm ONLY new files + the 3 allowed surgical files changed
   (src/v17_runner.py, src/pooled_validation.py, src/v17_acceptance.py). If any oracle file changed,
   set oracle_untouched=false and list it.
3. Confirm the Phase-0 golden contract still reproduces bit-for-bit (golden_intact).
4. Completeness critic: list any un-ported feature, un-asserted P1–P8 invariant, skipped edge case, or
   leftover TODO.
5. Write plan/IMPLEMENTATION_REPORT.md: files created/modified, full pytest result, the PIR-spike branch
   taken (${BRANCH}), the EXACT Colab steps the user runs (from temp/colab_h100_validation.py), and any
   blockers/outstanding items.
Per-phase rollup for your context: ${JSON.stringify(allReports)}`,
  { label: 'final:verify+report', phase: 'Report', schema: FINAL_SCHEMA })

return {
  pir_branch: BRANCH,
  full_pytest: final ? final.full_pytest_status : 'unknown',
  oracle_untouched: final ? final.oracle_untouched : null,
  golden_intact: final ? final.golden_intact : null,
  gpu_parity_rounds: parityRound,
  gpu_parity_unresolved: unresolved.length,
  report_path: final ? final.report_path : 'plan/IMPLEMENTATION_REPORT.md',
  summary: final ? final.executive_summary : null,
  outstanding: final ? final.outstanding : null,
}
