# GPU Refactor — Executable Build Spec

**Consumed by:** `plan/gpu_implementation_workflow.js` (the implementation workflow).
**Companion design doc:** `plan/gpu-refactor-plan.md` (the *why*; this file is the *what/how*).
**Audience:** autonomous build/verify agents. Every task below is test-gated: write the test FIRST, implement, run `pytest`, loop until green.

---

## §0. GROUND RULES (every agent obeys these)

### 0.1 The trust root — DO NOT MODIFY (these files ARE the parity oracle)
`src/detector.py`, `src/indicators.py`, `src/v17_fastdetector.py`, `src/parity.py`,
`src/scoring.py`, `src/pooled_scoring.py`, `src/validation.py`, `src/search_space.py`,
`src/speculatores145.py`, `src/universe.py`, all `pine/*`.

Touching any of these = parity violation. The GPU code is validated by being **byte-identical to the
output of these files**. If a change seems needed here, STOP and record it as a blocker in your
return value — do not edit.

**Surgical edits allowed** (additive only, must not change existing default behavior):
`src/v17_runner.py`, `src/pooled_validation.py`, `src/v17_acceptance.py`.
After editing, run the existing test suite to prove no regression.

### 0.2 TDD protocol (non-negotiable)
1. Write the test file first with the exact assertions in this spec.
2. Run `python -m pytest <testfile> -q` → expect RED.
3. Implement the minimum to pass.
4. Run again → GREEN. If RED, fix and rerun. Loop until green or a hard blocker.
5. Return a structured report: files written, final pytest status (pass/fail + counts), blockers.

Never report "done" on a RED test. Never weaken an assertion to make it pass — if a parity assertion
cannot pass, that is a finding, not a test to relax.

### 0.3 Dependencies
Allowed new deps: `torch` (CPU build is fine — we test on CPU), `scipy` (Sobol via
`scipy.stats.qmc.Sobol`), `cma` (CMA-ES). Prefer `uv pip install <pkg>`; fall back to `pip install`.
`optuna`, `numpy`, `pandas` already present. Do NOT add heavyweight frameworks (no JAX, no EvoTorch)
unless this spec names them.

### 0.4 No git commits
Leave all work in the working tree. Do not commit, branch, or push. The user reviews the diff.

### 0.5 Logging / style
Follow the repo conventions (module `logging.getLogger(__name__)`, type hints, dataclasses for config,
no `print` in library code, files ≤ ~400 lines — split if larger). Match surrounding code idiom.

### 0.6 PARITY INVARIANTS CHECKLIST (the GPU evaluator MUST reproduce ALL of these)
Any GPU/torch reimplementation of detector math must reproduce these exactly. These are the assertions
the Phase-3 adversarial verifiers will try to break.

- **P1 — dtype split:** `pir_matrix` stays **float32** (matches `indicators.precompute_matrices`,
  which is float32). ALL other boundary-feeding features (GJR, HAR, trend/linreg, ER, momentum,
  vol_surge, volatility pos, drift, scale_div, agreement) are **float64**. Disable TF32 / low-precision
  reductions on any threshold-feeding op.
- **P2 — per-slice artifacts with slice-local seeds:** GJR/HAR/PIR are built PER SLICE (never once per
  asset across slices). Reproduce `calc_gjr_asym` seeding exactly: `gjr_var[0]=sym_var[0]=max(r2[0],1e-12)`;
  in-slice 252-bar `lr_var` rolling mean; for bars < 252 the omega fallback `om=max(r2[t],1e-12)`;
  per-step `max(., 1e-12)` clamp.
- **P3 — strict comparison operators verbatim:** reproduce the exact operator from `detector.py` /
  `v17_fastdetector.py` for every gate/vote. Examples: `agr >= min_agreement`, `|scale_div| > thresh`,
  `agr > dur_extreme_pct`, `dur_at >= min_duration`, `bars_since > cooldown`, `gjr <= -thr`,
  `har >= thr`, `vola_pos > vola_high_pct`, `vol_surge_high < 1/thresh`, `vol_surge_low > thresh`.
  Never normalize `>` to `>=`.
- **P4 — NaN→False:** in the warm-up region (first ~252 bars where GJR/lr_var/linreg/HAR are NaN),
  bool votes resolve via `detector._to_bool`'s explicit NaN→False rule. Unit-test the first ~300 bars.
- **P5 — pivot drift exactness:** `calc_pivot_drift` semantics — `None → 0.0`;
  `min_pivots = max(lookback, 2)`; divide by `(min_pivots - 1)`; `max(|start_val|, 1e-9)`.
  Same-bar push order is **HIGH pivot pushed before LOW** (see `_detect` L533-536). Drift is read
  from the **PRE-update** ring, THEN this bar's pivots are appended (read-BEFORE-append).
  The `confirmed_pivots` stack is **ONE per-asset buffer SHARED by both sides**, capacity
  `Kmax = max over {high,low} of pivot_drift_lookback`, which is **≤ 20** (bound from
  `search_space.INT_BOUNDS["pivot_drift_lookback"] = (2,20)`); size it ≥ 20 from the FROZEN base Params.
- **P6 — edge voting as fixed-lag arrays:** `state_pd_dn_h`/`state_pd_up_l` and all other edge votes use
  `detector._edge_or_state` (absolute-bar lag `edge_window`, not pivot-count). Precompute these as arrays;
  do NOT carry them in the scan state.
- **P7 — segmented scan:** when assets are packed into one padded batch, `valid_mask` must FREEZE
  `bars_since`/`dur_at`/`dur_miss` increments AND pivot appends on pad bars, and RESET all carry at
  every asset boundary. A flat concat must not leak cooldown or pivots across instruments.
- **P8 — linreg:** the CPU uses `np.polyfit` (`indicators.linreg_slope_step`). A closed-form torch OLS
  will NOT be bitwise-equal even at float64. Either (a) keep `np.polyfit` semantics for the feature
  precompute (acceptable — it is shape-precompute, off the hot scoring path), or (b) treat linreg as a
  search-only surrogate and DOCUMENT it, sizing top-K to absorb the rank error. Default: (a).

### 0.7 Definition of GPU parity "GREEN"
For a sampled set of Params (≥ the base gold Params + ≥5 random in-bounds threshold draws), on EVERY
test slice (≥ SPX IS, SPX OOS, DAX IS, DAX OOS, one short stream):
`np.array_equal(gpu["signal_high"], SpeculatorDetector(...).run()["signal_high"])` AND same for `_low`,
AND `abs(FastPooledScorer_or_gpu_LCB − PooledScorer_LCB) < 1e-9`.
Any mismatch = HARD FAIL (raise in the test). This mirrors the existing
`tests/test_v17_fastdetector.py` / `tests/test_v17_fastscorer.py` contracts.

---

## §1. PHASE 0 — Parity harness + golden baseline (PREREQUISITE)

**Goal:** a reproducible golden snapshot so every later change is diffable.

**Files:**
- NEW `temp/capture_baseline.py` — loads SPX (`data/raw/SPX_1D_18710201_20260318.csv`) and DAX
  (`data/raw/DAX_1D_19700102_20260324.csv`); for the base gold `Params()`, runs `SpeculatorDetector`
  on a few representative slices; writes golden arrays (`signal_high`, `signal_low`) + per-fold scores +
  final pooled LCB to `results/diag/golden/` as `.npz`/`.json`, seed=42.
- NEW `tests/test_parity_golden.py` — re-runs the capture logic and asserts bit-for-bit reproduction
  across two runs; records the existing `FastDetector` vs `SpeculatorDetector` `np.array_equal` pass and
  `FastPooledScorer` vs `PooledScorer` `abs ≤ 1e-12` pass as the reference contract.

**Done:** `pytest tests/test_parity_golden.py -q` green; golden files exist.

---

## §2. PHASE 0.5 — PIR byte-identity spike (CPU, GATING the GPU branch)

**Goal:** decide whether a torch float32 rolling-mean/min-max can match `pandas.rolling().mean()` /
`indicators.pir_of` to `np.array_equal`. This is run **on CPU** (`torch.device("cpu")`, float32) — CPU
can DISPROVE the cumsum approach (if it drifts on CPU-float32 it is dead) and strongly indicate the
GPU outcome. The residual "does real H100 reduction order match" question is deferred to the Colab
confirmation in Phase 4.

**Files:** NEW `temp/pir_parity_spike.py`.

**Procedure:** on a real SPX IS slice (~5.7k bars), for all 499 scales (2..500):
1. Build `pir_matrix` two ways in torch float32: (a) **cumsum-difference** rolling mean, (b)
   **pandas-faithful** per-window mean (unfold/convolution matching pandas summation order).
2. Compare each to the CPU `indicators.precompute_matrices(...)` float32 matrix via `np.array_equal`,
   and record `max_abs_drift` and the **signal-flip rate** when fed through `calc_agreement_fast` at a
   representative `pct_extreme`.

**Return (structured):** `exact_match` (bool — did ANY method give `np.array_equal`), the winning method,
`max_abs_drift`, `signal_flip_rate`, and `recommended_branch`:
- `"trust-kernel"` if a method is byte-identical (or flip rate is exactly 0),
- `"noisy-ranker"` otherwise (GPU ranks; CPU re-scores finalists; size top-K from `signal_flip_rate`).

**Done:** spike script runs and prints the verdict JSON. The workflow reads `recommended_branch` and
adapts Phase 3. No pass/fail gate — this is a measurement that selects a path.

---

## §3. PHASE 1 — Batch-hungry optimizer on the EXISTING scorer (HIGHEST ROI, CPU, no new parity surface)

**Goal:** replace coordinate ascent with Sobol-seeded separable-CMA-ES emitting 256–1024 candidates/round,
scored via the EXISTING `FastPooledScorer` (parallelized across candidates). `score(params)→float` is
unchanged, so parity surface is zero.

**Why this is mandatory (NOT a CPU substitute for the GPU):** coordinate ascent emits candidates one at a
time (each depends on the prior argmax) — it has nothing to parallelize and CANNOT fill a GPU. This phase
is the PREREQUISITE that produces the wide candidate batch the Phase-3 GPU evaluator scores in parallel.
The CPU run here measures optimizer *quality* (hardware-independent); on capped-CPU Colab the *throughput*
speedup still comes from the GPU. Do not frame Phase 1 as a reason to skip Phase 3.

**Files:**
- NEW `src/v17_search.py` — `class BatchOptimizer`: Sobol seeding (`scipy.stats.qmc.Sobol` over the
  active continuous threshold bounds from `search_space.space_for(side).float_bounds`, restricted to
  `v17_optimize.active_threshold_fields(seed, side)`), separable CMA-ES (`cma` lib with diagonal
  covariance, warm-started at the v16/gold seed), population λ configurable [256,1024], returns the
  full evaluated population + **top-K** finalists by penalized LCB. Discrete SHAPE params stay FROZEN
  (reuse the `FastDetector` shape guard). Parallelize candidate scoring with `multiprocessing` (or a
  thread pool if the scorer releases the GIL via numpy) — keep it optional/configurable.
- MODIFY `src/v17_runner.py` — add a `search="cma"` route in `run_v17` that uses `BatchOptimizer`
  against `FastPooledScorer`; KEEP coordinate ascent as `search="ascent"` (default unchanged).
- MODIFY `src/v17_acceptance.py` — expose the firing penalty so the batched objective can apply
  `firing_excess` per candidate BEFORE argmax (the scorer already supports `firing_penalty`; wire a
  helper that folds it into the population ranking).

**Tests:** NEW `tests/test_v17_search.py`:
- Sobol seeding stays within `float_bounds`; shape params never change across the population.
- The reported winner re-scored by the REAL `PooledScorer` matches the search-reported LCB to `< 1e-9`.
- Determinism: same seed → same population (use a fixed CMA seed; vary only by explicit seed arg).
- `run_v17(..., search="ascent")` output is byte-identical to today (regression).

**Done:** `pytest tests/test_v17_search.py -q` green; existing suite still green.

---

## §4. PHASE 2 — Validation fixes + selection-bias guards (CPU, parity-orthogonal)

**Goal:** fix fold overlap + dead OOS margins; add PBO/deflation scaled to the batched search size.
Default behavior stays IDENTICAL unless explicitly enabled.

**Files:**
- NEW `src/cpcv.py` — purged Combinatorial CV (N groups 6–8, k=2), justified on OVERLAP removal (NOT a
  leak). Asserts no IS group's label window overlaps any test group (purge + 200-bar embargo). Reconstructs
  OOS paths for the LOW (event-rich) side.
- NEW `src/overfit_guard.py` — PBO via CSCV; deflated-best haircut scaled to `λ × generations`.
- MODIFY `src/pooled_validation.py` — add `_run_cpcv_loop` ALONGSIDE `_run_fold_loop` (default OFF);
  mask the dead 200-bar label margins in `_stream_stat` (do not score bars where the centered label
  window is undefined); add a per-asset-then-aggregate HIGH diagnostic next to the pooled LCB.
- MODIFY `src/v17_runner.py` — wire deflation into the REPORTED number when batched search is used;
  emit per-asset HIGH diagnostics.

**Tests:** NEW `tests/test_cpcv_purge.py`:
- No IS group label window overlaps any test group (purge correctness).
- LOW-side reconstructed path count ≥ current fold count.
- For HIGH: PBO/percentile are returned as **advisory** fields; the primary HIGH summary is event count
  + a wide Wilson/credible interval.
- **Regression:** with CPCV OFF (default), `PooledScorer` / `run_v17` results are unchanged vs Phase-0 golden.

**Done:** `pytest tests/test_cpcv_purge.py -q` green; golden regression intact.

---

## §5. PHASE 3 — Batched GPU evaluator (`score_pop`) — the parity-fragile core

**Goal:** a PyTorch (float64; PIR float32) batched evaluator byte-identical to `FastDetector`/
`SpeculatorDetector`, tested on CPU-torch. Decomposed into 4 gated sub-modules; build in order, each
gated by its own `np.array_equal` test against the CPU oracle BEFORE proceeding.

**Branch (from §2):** if `recommended_branch == "trust-kernel"`, the kernel's signals are authoritative
for ranking and the parity tests must be `np.array_equal`. If `"noisy-ranker"`, the kernel is a ranker
only: parity tests assert `signal_flip_rate ≤ measured_threshold` and top-K is sized to the flip rate;
the EXACT CPU detector still produces every reported/exported number either way.

**Package:** NEW `src/v17_gpu/` (`__init__.py` with `__all__`).

### §5.1 `src/v17_gpu/drift_precompute.py`
Replace the growing `confirmed_pivots` stack with a precomputed per-asset drift array. Implements P5:
fixed-capacity ring (`Kmax ≥ 20`), `None→0.0`, `min_pivots=max(lookback,2)`, `/(min_pivots-1)`,
`max(|start|,1e-9)`, HIGH-before-LOW push, read-BEFORE-append. Output `drift[n_assets, max_bars, 2]`
(high, low) float64.
**Test:** for a sample of params, `np.array_equal` of the per-bar drift values vs the values the CPU
`_detect` loop computes (instrument `detector` or recompute the reference inline). Edge cases: a bar
that confirms BOTH a high and low pivot; the first bar where `min_pivots` is first satisfied.

### §5.2 `src/v17_gpu/upload.py`
Slice → device padded packer: builds `[n_assets, max_bars, ...]` tensors + `valid_mask` (P7) +
length-bucketing + scale-tiling hooks for the PIR matrix (P1 memory discipline). Pure data marshaling.
**Test:** round-trip — packing then per-asset unpacking reproduces the source arrays exactly;
`valid_mask` marks exactly the real bars.

### §5.3 `src/v17_gpu/eval_torch.py` — Phase-1 features + threshold layer
Reproduce, in torch, every Phase-1 array of `FastDetector._precompute` + the threshold comparisons of
`FastDetector.signals` (P1–P4, P6, P8): PIR float32 (method chosen by §2), agreement, scale_div, trend
(slope_val + linreg_norm via `np.polyfit` semantics — P8 default (a)), vol_surge, momentum-divergence
edge, momentum-velocity, volatility pos, GJR/HAR norms (P2 per-slice seeding), ER gate, price gate,
baseline pivots, then all `_edge_or_state` vote arrays.
**Test:** each intermediate array `np.array_equal` (float32 for PIR; exact for the rest, or `atol=0` at
float64 where the op is a pure comparison) vs the corresponding `FastDetector` attribute on ≥3 slices.

### §5.4 `src/v17_gpu/eval_torch.py` — batched Phase-2 scan + `score_pop`
The stateful loop (`FastDetector.signals` Phase-2) as a hand-written batched bar-loop over
`(candidate × asset)` lanes: carry = `{dur_at,dur_miss,bars_since}` per side + `pivot_ring[Kmax]` +
`pivot_count` (P5, P7). One wide vector op per bar-step. `score_pop(params_batch)` returns
`signal_high/low [n_candidates, n_assets, max_bars]` → pooled reduction → `lcb[n_candidates]` reusing
the existing `pooled_fold_score` / bootstrap logic for numeric identity.
**Test (`tests/test_v17_gpu_parity.py`, the load-bearing gate):** §0.7 GREEN criteria. Explicit edge-case
tests for read-before-append, HIGH-before-LOW same-bar, warm-up NaN (first ~300 bars), per-asset reset
(no cooldown/pivot leak across the asset boundary), ring truncation at `Kmax`. HARD FAIL on any mismatch.

**Adversarial verification (workflow-driven):** after §5.4 is green, independent skeptic agents each try
to BREAK parity on one lens (read-before-append / HIGH-before-LOW; warm-up NaN; per-asset reset/cooldown;
ring truncation). Any concrete failing case → a repair agent fixes and re-runs `tests/test_v17_gpu_parity.py`.
Loop until no skeptic can break it (bounded rounds).

---

## §6. PHASE 4 — Integration + Colab handoff

**Goal:** full pipeline + the human's H100 confirmation step.

**Files:**
- MODIFY `src/v17_runner.py` — `run_v17_gpu(...)`: GPU batched search (`score_pop`) → CPU finalist
  re-score with the EXACT `SpeculatorDetector`+`PooledScorer` (top-K sized to §2's flip rate) →
  `v17_acceptance` gates → per-asset TradingView-export audit hook. Generalize the existing
  `|fast-real|>1e-9` warning in `run_v17` into a HARD finalist filter (drop any finalist whose GPU LCB
  disagrees with the CPU LCB beyond tolerance).
- NEW `temp/colab_h100_validation.py` (+ optional `.ipynb`) — the ~5-minute script the USER runs on a
  Colab H100: installs torch-cuda, runs `temp/pir_parity_spike.py` ON GPU, runs
  `tests/test_v17_gpu_parity.py` ON GPU, runs one tiny end-to-end `run_v17_gpu`, and prints a single
  PASS/FAIL verdict + the measured GPU-vs-CPU signal-flip rate (to confirm or revise the §2 branch on
  real hardware).

**Tests:** end-to-end `run_v17_gpu` on a 2-asset CPU pool produces a leaderboard whose top finalist's
reported LCB equals the exact CPU `PooledScorer` LCB to `< 1e-9`; golden Phase-0 contract intact.

**Done:** the integration test green; the Colab script exists and is self-contained.

---

## §7. DEFINITION OF DONE (final verification agent)
1. `python -m pytest -q` — full suite green (report exact pass/fail counts; never hide a red test).
2. `git diff --name-only` touches ONLY new files + the three allowed surgical files (§0.1). If any oracle
   file changed → FAIL and report.
3. Golden Phase-0 contract still reproduces bit-for-bit.
4. A completeness critic lists anything unfinished: unported feature, un-asserted invariant (P1–P8),
   skipped edge case, or a TODO left in code.
5. Final report `plan/IMPLEMENTATION_REPORT.md`: files created/modified, full pytest result, the §2
   PIR-spike branch taken, the EXACT Colab steps the user must run, and any blockers.
