# GPU/H100 Refactor + Multi-Asset Optimizer + Validation Redesign — Decisive Plan

**Author:** Lead architect synthesis
**Date:** 2026-06-09
**Branch context:** `feature/v17-antigaming`
**Status:** Implementation-ready, phased, parity-gated

---

## 0. TL;DR (read this first)

**Recommended architecture: a phased HYBRID, "C-lite first, B-core later" — NOT a big-bang rewrite.**

> **WHY A GPU IS REQUIRED ON COLAB — capped CPU (read first; this supersedes any earlier "CPU might suffice" wording).** The goal is a *vast* wall-clock speedup and the hardware is fixed: Colab gives ~8–12 CPU cores but a powerful GPU. There are **two distinct speedups and they MULTIPLY**: **(A) a smarter search** (coordinate ascent → batched CMA-ES) explores more per unit of compute, and **(B) parallel throughput** scores 256–1024 candidates *at once*. (A) is hardware-independent and partly CPU-measurable; **(B) is exactly what your capped CPU cannot provide and a GPU can.** Critically, **(A) is NOT a substitute for the GPU — it is the prerequisite that makes the GPU usable:** coordinate ascent emits candidates one at a time (each depends on the prior argmax), so it has *nothing to parallelize* and cannot fill a GPU. Switching to a population optimizer is what gives the GPU a wide batch to score. **You need BOTH, in sequence — the optimizer swap THEN the GPU evaluator — not one or the other.** Any sentence elsewhere implying the GPU "may not pay for itself" or that "a multi-core CPU gets most of the win" applies only to machines with *many* CPU cores, NOT to capped-CPU Colab, and is **retracted.**

The four candidate proposals (A faithful port, B fully-batched population optimizer, C hybrid GPU+CPCV) all earned a "risky" verdict for the same root reason: each bundles a *correct* idea with an *over-scoped or mis-premised* idea. The synthesis unbundles them:

1. **The binding constraint is the OPTIMIZER, not the kernel.** Coordinate ascent (`src/v17_optimize.py`) is deterministic, issues ~163–217 evals/side strictly sequentially, and leaves any GPU ~99% idle. Swapping it for a batch-hungry producer (Sobol → sep-CMA-ES) against the **existing** `FastPooledScorer` needs **zero new parity surface** and is the **prerequisite that gives the GPU a wide batch to score** — so it comes FIRST. It is plumbing for the GPU, *not* a CPU substitute for it (see the callout above).
2. **The GPU port is worth doing, but only the per-candidate `score()` path, and only after the optimizer can fill it.** Port Phase-1 + threshold layer + Phase-2 scan to a batched evaluator. Keep the real `SpeculatorDetector` (`src/detector.py`) as the untouched float64 judge.
3. **The fold system is NOT "bullshit," but it has two real defects and one false alarm.** Fix the real ones (fold overlap optimism via purging; dead OOS margins). Do NOT chase the "one-sided embargo leak" — that claim is **wrong** (labels are slice-local). Add selection-bias guards (PBO + deflation) that scale with batch size. CPCV is the right *target* but is a follow-on, not a blocker.
4. **Parity is preserved BY CONSTRUCTION:** the GPU only RANKS candidates; every reported/exported number is re-derived on the exact float64 CPU `SpeculatorDetector` and gated by a per-asset TradingView-export audit. A buggy GPU kernel mis-ranks finalists; it can never corrupt a signal pasted into Pine.

**Honest expectation:** end-to-end search speedup of **~10–40×** on a multi-asset pool (NOT 50–200×, NOT 100×). The H100 buys *throughput* (wider search, more pooled assets), never *statistical power* on the ~27-event HIGH side.

---

## 1. Recommended architecture

### 1.1 The decision

**Adopt Candidate C's "propose-on-GPU / judge-on-CPU" wall and its validation-fix instinct, but with Candidate B's batched-scan engineering for the evaluator, and reject the over-scoped pieces of both.** Concretely:

| Layer | Decision | Source | Rejected alternative |
|---|---|---|---|
| Optimizer | Sobol seed → separable-CMA-ES (warm-started from v16 seed), population λ∈[256,1024] | B/C | Keep coordinate ascent (A) — leaves GPU idle |
| Evaluator | Batched `score_pop` over existing precompute; Phase-2 as one segmented scan batched across (candidate × stream × fold) | B | Per-asset-only batch (A) — mis-identifies the parallel axis |
| Framework | **PyTorch** for v1 (eager + `torch.func.vmap` + a hand-written batched sequential scan), JAX `lax.scan` as a *considered* v2 | A | All-in JAX (B/C) — Colab toolchain fragility + brutal byte-identity-vs-pandas debugging slog |
| Judge | Unchanged float64 `SpeculatorDetector` + `PooledScorer`; finalists re-scored on CPU | A/B/C | Trust the GPU number (none proposed this, but enforce it) |
| Validation | Fix overlap (purge) + dead margins + add PBO/deflation; CPCV as opt-in follow-on | C (corrected) | Big-bang CPCV rewrite justified by a non-existent leak (C as-stated) |

### 1.2 Why PyTorch, not JAX, for v1

The research correctly notes JAX's `lax.scan` is the cleaner primitive for the sequential carry. But two facts dominate for *this* codebase:

- **Byte-identity to pandas is the gating risk, and it is framework-agnostic.** The hardest task is making a GPU rolling-mean / rolling-min-max reproduce `pandas.rolling().mean()` summation order so the float32 `pir_matrix` is bit-identical (`np.array_equal`). A `cumsum`-difference SMA accumulates float32 rounding *differently* than pandas and drifts ~1e-5..1e-6 over 57k bars — flipping the strict `agr > pct_extreme` count by `1/n_scales`. This is true in JAX *and* PyTorch. **Therefore the parity work is identical; pick the framework with the lower operational risk.**
- PyTorch eager is debuggable step-by-step, runs fine on a single H100 with no NCCL/EFA multi-node failure mode, and `torch.func.vmap` covers the candidate-batch axis. The Phase-2 scan is a hand-written batched bar-loop (one wide vector op per bar-step) — we do NOT rely on `torch.compile` unrolling 5.7k iterations.

If v1 profiling shows the Python-level bar-loop dispatch is the bottleneck (likely on very wide batches), v2 can wrap the scan body in a CUDA graph (static shapes — fine here) or port the single scan kernel to JAX `lax.scan`. That is an isolated, later optimization behind the same `score_pop` interface.

### 1.3 Addressing every adversarial `must_fix`

**Candidate A (faithful port) — must-fixes folded in:**
- *Batch axis mis-identified ("assets only").* CORRECTED: the parallel lanes are `(candidates × informative_folds × streams × {IS,OOS} × sides)`. We batch the candidate axis (the optimizer's job) and the `(stream × fold)` axis together. Confirmed in `FastPooledScorer._fold_scores`.
- *57k "latency floor" straw man.* CORRECTED: `signals()` sees IS slices (`is_bars = max(401, 0.10·n)` ≈ 5.7k for SPX) and OOS slices (`oos_bars = max(401, 0.03·n)` ≈ 1.7k), never the full 57k. Scan depth per lane is ~5.7k, and there are many short lanes. Memory/speedup re-estimated against slice lengths.
- *"Bench the cheaper numba/precompute-drift alternative first."* ADOPTED as Phase 1 — but the even cheaper win is the optimizer swap, which we sequence before any kernel work.

**Candidate B (fully batched) — must-fixes folded in:**
- *Pivot ring buffer `Kmax≈8` is WRONG.* CONFIRMED against `search_space.py`: `pivot_drift_lookback ∈ (2,20)` and `calc_pivot_drift` reads `min_pivots = max(lookback,2)` entries → **`Kmax = max over both sides of max(pivot_drift_lookback, 2)`, i.e. ≥20**, sized from the FROZEN Params, holding the shared HIGH+LOW interleaved stack. This is a `must_fix`, not a "con."
- *Drift-edge ring unbounded.* RESOLVED by precomputing `state_pd_dn_h / state_pd_up_l` as fixed-lag arrays (`edge_or_state`) and dropping them from the scan carry entirely (the lag is absolute-bar, not pivot-count).
- *Read-before-append ordering.* PINNED: scan step computes drift from the PRE-update ring → emits votes/gates → THEN appends this bar's HIGH-then-LOW pivots for `t+1`. Explicit unit test on pivot-confirm bars.
- *"Parity is provable" overclaim.* CORRECTED: the GPU-vs-FastDetector byte-identity test certifies GPU==CPU only. CPU==Pine remains the empirical per-asset TV-export audit (`compare_python_to_tv`, count diff == 0). Internal harness is necessary, not sufficient.
- *Saturation-vs-overfitting contradiction.* RESOLVED: deflation (haircut scaled to λ×generations) + PBO + firing-penalty-in-objective. We report the *deflation-adjusted effective population* and re-derive speedup from realized concurrent lanes, not raw λ.
- *PyTorch-eager fallback negates the lax.scan premise.* AVOIDED by choosing PyTorch + a hand-written batched scan from the start (not `torch.compile`).

**Candidate C (hybrid + CPCV) — must-fixes folded in:**
- *"One-sided embargo leak" is FALSE.* CONFIRMED against `pooled_validation._run_fold_loop` L118–125: each slice is `reset_index(drop=True)` then `add_pivot_labels` is called **separately per slice**, so a centered label window at the OOS edge clips at the slice boundary and CANNOT see the next fold's IS. **There is no within-fold look-ahead leak.** The real defect is ~50% fold OVERLAP (IS=10%/step=5%) → optimism/non-independence, which purging (CPCV) fixes by construction. We re-justify validation changes on *overlap + dead margins*, never on a leak.
- *9-decimal parity gate is unimplemented and contradicts the 1e-6 audit.* DROPPED. We gate on the ACTUAL `parity.py` audit (exact-bool signals + `atol=1e-6` numeric) plus a margin-to-boundary *diagnostic* (advisory, not a separate gate).
- *GARCH "once per asset" breaks slice-local seeding.* CONFIRMED: `build_detector_artifacts` runs **per slice**, and `calc_gjr_asym` seeds `gjr_var[0]=max(r2[0],1e-12)` from the slice's first bar with an in-slice 252-bar `lr_var`. **The GPU must build artifacts PER SLICE with slice-local seeds.** GARCH is amortized across *candidates* (already off the per-candidate hot path), NOT across slices. We keep GARCH as a plain sequential per-slice kernel (it is not the bottleneck) and drop the associative-scan optimization as premature.
- *PIR cumsum reformulation drifts from pandas.* RESOLVED by a gating spike (Phase 0.5): prove a float32 GPU PIR matches `pandas.rolling().mean()` to `np.array_equal` on a real SPX slice BEFORE committing to the kernel; if impossible, compute SMA per-scale the pandas way or accept the GPU as a noisy ranker and size top-K to the measured rank error.
- *Memory infeasibility understated.* CONFIRMED: PIR is **499 scales** (`scale 2..500`), so one full IS slice PIR ≈ `499 × 57k × 4B ≈ 114 GB` for ONE asset — over 80 GB. Scale-tiling + candidate-chunking are MANDATORY; speedup re-estimated with chunking baked in.
- *CPCV worsens HIGH-side thinness; PBO too thin to gate.* ADOPTED: for HIGH, PBO and the per-path percentile are **advisory diagnostics**, not export gates. Primary HIGH summary = raw event count + a wide Wilson/credible interval. CPCV is opt-in and **not** the default for HIGH.
- *Top-K sized to measured rank error.* ADOPTED: K calibrated empirically from GPU-vs-CPU LCB rank correlation on a held-out batch, not picked a priori.

---

## 2. Fold / pooled-validation verdict

**Verdict: KEEP the leak-free skeleton, FIX two real defects, ADD selection-bias guards. Do NOT replace wholesale. The user's instinct ("maybe bad design") is half-right: the *objective* is statistically thin, but the *mechanics* are not leaky.**

### What is CORRECT and stays (non-negotiable)
- **Per-slice label recomputation** (`add_pivot_labels` on each `reset_index`'d slice) — this is a strong purge; there is **no IS→OOS look-ahead**. The "one-sided embargo leak" claim is **debunked**.
- 1/cluster_size weighting (`cluster_weights`), the informative-fold filter (`_fold_is_informative`), and the final re-score of the winner with the real detector.

### Real defect #1 — fold OVERLAP (fix)
`IS_FRACTION=0.10`, `STEP_FRACTION=0.05` ⇒ adjacent folds reuse ~50% of bars. The block bootstrap (`block_len=2`) only partly accounts for this; the LCB 10th-percentile over single-digit informative HIGH folds is a discreteness artifact, not a calibrated bound. **Fix:** purged Combinatorial CV (CPCV, N=6–8 groups, k=2) eliminates overlap by construction and yields ~7 reconstructed OOS *paths* instead of 5–8 overlapping folds — for the **LOW (event-rich) side**. For HIGH, CPCV repartitions the same ~27 events and is **not** mandated.

### Real defect #2 — dead OOS margins (fix)
The centered label window (half-width `max(STRUCTURAL_NEST)=200`) forces the first/last 200 bars of every slice to zero labels. A 401-bar `MIN_STREAM_BARS` OOS slice is almost entirely unscorable. **Fix:** explicitly mask (don't score) the 200-bar dead margins in `_stream_stat`, and/or enlarge OOS (`OOS_FRACTION`) so the dead margin is a small fraction. This is a ~30-line change, parity-orthogonal.

### Missing guard — selection bias (add)
A batch-hungry optimizer evaluates *far* more configs than coordinate ascent's 163–217, inflating multiple-comparison risk on the ~27-event HIGH set. **Add:**
- **Fold the firing penalty (`firing_excess`) INTO the batched objective** so gaming candidates (loosen gates → fire more → recall/precision ratio up) are down-weighted BEFORE argmax, not just filtered after.
- **Deflated-best haircut** scaled to `λ × generations` (thousands of trials), applied to the *reported* number.
- **PBO via CSCV** — advisory go/no-go for LOW; advisory-only (with caveats) for HIGH.

### The honest ceiling
With ~27 validatable HIGH tops in 155 yr, **NO CV design manufactures power the data lacks.** Pooling across assets is the only genuine lever for HIGH events — keep it, but report per-asset-then-aggregate HIGH diagnostics alongside the pooled LCB; if pooled and median-per-asset disagree sharply, flag the pooled number as a heterogeneity artifact. **For HIGH, report event counts + wide intervals and treat it as monitor-only, low-confidence.** The H100 does not change this.

---

## 3. Parity-preservation strategy

### 3.1 Two-tier discipline (the whole game)
- **TIER 1 — SEARCH (GPU, approximate, fast):** ranks candidates. May use float32 ONLY where a value does not feed a strict comparison. Returns **top-K (K≥20, calibrated), never top-1.**
- **TIER 2 — JUDGE/EXPORT (CPU, exact float64, MANDATORY):** every finalist re-scored with the real `SpeculatorDetector` + `PooledScorer`. The reported LCB, all `v17_acceptance` gates, and the params pasted into Pine come ONLY from this CPU number. (`run_v17` already warns on `|fast-real|>1e-9`; generalize to a hard finalist filter.)

### 3.2 Float-precision rules enforced in the GPU evaluator
- **`pir_matrix` stays float32** to MATCH the already-float32 CPU path (`indicators.py` L452-453). Do NOT promote it (would diverge from today's validated CPU baseline). Validate the GPU PIR **bar-by-bar** against the CPU float32 matrix.
- **ALL boundary-feeding features stay float64:** GJR/HAR (β=0.90 compounds error), trend/linreg, ER, momentum, vol_surge, volatility, drift. (H100 FP64 ≈ 34 TFLOPS, half of FP32 — fine, the workload is carry/latency-bound.) PyTorch: disable TF32 (`torch.backends.cuda.matmul.allow_tf32=False`) and fp16/bf16 reductions on threshold-feeding ops.
- **Build artifacts PER SLICE** with slice-local seeds: reproduce `gjr_var[0]=max(r2[0],1e-12)`, the in-slice 252-bar `lr_var`, the NaN-omega fallback `om=max(r2[t],1e-12)` for bars 0–251, and the in-loop `max(.,1e-12)` clamp every step. Prefer a sequential GARCH kernel over a parallel scan (it is amortized once per slice).
- **Reproduce exact strictness verbatim:** `agr>=min_agreement`, `|scale_div|>thresh`, `agr>dur_extreme_pct`, `dur_at>=min_duration`, `bars_since>cooldown`, `gjr<=-thr`, `har>=thr`. Never normalize `>` to `>=`.
- **NaN→bool via `_to_bool`'s explicit NaN→False rule** in the warm-up region (first ~252 bars where GJR/lr_var/linreg/HAR are NaN). Unit-test the first ~300 bars of each stream.
- **Pivot-drift precompute reproduces:** `None→0.0`, `min_pivots=max(lookback,2)`, `/(min_pivots-1)`, `max(|start|,1e-9)`, **HIGH-pivot-before-LOW** same-bar push order, and the **read-BEFORE-append off-by-one**. The `confirmed_pivots` stack is ONE per-asset buffer SHARED by both sides, capacity `Kmax ≥ 20`.
- **Pre-existing divergences reproduced, never "fixed":** the GARCH seed (do not switch to Pine's `lr_var` seed) and `np.polyfit` linreg (a JAX/torch closed-form OLS will not be bitwise-equal to `np.polyfit` even at float64; keep `np.polyfit` semantics or accept it as a search-only surrogate and absorb in top-K).
- **Segmented scan:** `valid_mask` must freeze `bars_since` increments, dur counters, AND pivot appends on pad bars, and RESET all carry at asset boundaries. A flattened concat over streams leaks cooldown across instruments.

### 3.3 The "is it safe to paste into Pine?" decision rule
A param set is safe to export **iff ALL hold**:
1. **Internal byte-identity:** the GPU `score_pop` (and `FastDetector`) vs the real `SpeculatorDetector` produce `np.array_equal(signal_high)` and `np.array_equal(signal_low)` on EVERY slice for that exact Params (extend `tests/test_v17_fastdetector.py`), and `|FastPooledScorer − PooledScorer| < 1e-9` (extend `tests/test_v17_fastscorer.py`). Treat any mismatch as a **hard CI failure**, not a warning.
2. **CPU re-score is the reported number:** the finalist's LCB and all `v17_acceptance` gates come from the float64 `PooledScorer`, not the GPU.
3. **Per-asset TradingView-export audit passes for EACH instrument:** `parity.compare_python_to_tv` → `signal_high`/`signal_low` exact-bool-equal, **count diff == 0** (the binding gate; count diff > 0 = FAIL), ≤1-bar position shift tolerated, numeric debug cols within `atol=1e-6`.
4. **Margin-to-boundary diagnostic (advisory):** flag any bar with `|feature − threshold| < 1e-7`; if such a bar flips a signal between GPU and CPU, investigate (it will already fail gate 1/3). Float diffs that do not flip a signal are acceptable (matches current policy).

If all four hold → paste. Otherwise → REJECT and do not export.

---

## 4. Multi-asset batched-eval: data layout + optimizer

### 4.1 Data layout (resident on device)
Per CPCV/calendar slice, computed once (Params-independent), length-bucketed to bound padding waste:

| Tensor | Shape | dtype | Notes |
|---|---|---|---|
| `pir` | `[n_assets, n_scales, max_bars]` | **float32** | matches CPU; **tile over scales** (499 × 57k × 4B ≈ 114 GB/asset uncut) |
| `feat64` | `[n_assets, max_bars, n_feat]` | float64 | slope, linreg_norm, vol_surge, mom_vel, vola_pos, gjr_norm, har_norm, er_gate, price_gate, pir_detect |
| `drift` | `[n_assets, max_bars, 2]` | float64 | precomputed pivot_drift_high/low (removes the growing stack) |
| `ph/pl_arr` + prices | `[n_assets, max_bars]` | bool/float64 | baseline pivot push schedule + pivot prices |
| `valid_mask` | `[n_assets, max_bars]` | bool | real-vs-pad; resets per-stream scan carry |
| `weights` | `[n_assets]` | float64 | 1/cluster_size |

Per search round: `params_batch` pytree, leaves `[n_candidates]` (256–1024) over the 6–12 active float thresholds. `vmap(score_one)` over the candidate axis; the Phase-2 scan carry per `(candidate, asset)` lane = `{dur_at, dur_miss, bars_since}` (int scalars, per side) + a fixed-capacity `pivot_ring[Kmax≥20]` (shared HIGH+LOW) + `pivot_count`. Output signals `[n_candidates, n_assets, max_bars]` bool → pooled reduction → `path_scores[n_candidates, n_paths]` → bootstrap/percentile → `lcb[n_candidates]`.

**Memory discipline (mandatory, not optional):** tile PIR over scales; chunk the candidate batch (`scan over param-chunks`) so the live vote tensor never exceeds HBM. The 1024 × 8 × 57k × 1B ≈ 467 GB single-batch vote tensor is impossible — chunking is required and partially serializes the candidate axis (factored into the speedup estimate).

**Parallelize the GRID `(candidates × streams × folds)`, NOT the bars within a stream.** Each bar-step is a wide elementwise op over all lanes (the Mamba "sequential along L, parallel across batch" pattern). The cooldown feedback (`bars_since` reset on fire) is irreducibly sequential per lane but is an O(1) scalar update — free when batched.

### 4.2 Optimizer choice
**Separable (diagonal) CMA-ES, warm-started by Sobol/LHS coverage, via EvoTorch (PyTorch).**
- **Continuous thresholds** (the 6–12 active fields in `_ALWAYS_FIELDS + _DRIFT_FIELDS + active _VOTE_FIELDS`) are CMA-ES's native decision vector. Population **λ∈[256,1024]** is the GPU-saturation knob AND damps rank-noise in the event-scarce regime. Many warm-started restarts (v16 seed + perturbations + Sobol) run CONCURRENTLY as one batch; IPOP/BIPOP population-doubling for multimodality.
- **Why not coordinate ascent:** deterministic (re-running 10× reproduces the identical answer — "restarts" are pointless), strictly sequential (each axis depends on the prior argmax), emits ~6 candidates/axis → leaves the GPU ~99% idle, and cannot exploit threshold interactions (e.g. `min_agreement × dur_extreme_pct` couple through the same agreement array).
- **Discrete SHAPE params** (`S_detect`, `scale_*`, `use_*` switches, categoricals) stay FROZEN per FastDetector build (changing them requires a fresh precompute; the guard at `v17_fastdetector.py` L200-205 stands).
- **Int thresholds** (`confirm_count`, `min_duration`, `cooldown_bars`, `pivot_drift_confirm_bias`) live INSIDE the Phase-2 scan and need NO precompute rebuild. They are a **later extension** (Differential Evolution with integer rounding, or an outer enumeration) — they widen the gaming surface and must pass the same acceptance gates.
- **Fallbacks:** Optuna batched ask/tell (reuses v16 plumbing — quickest) or Ax/BoTorch qNEI (noise-aware, sample-efficient late-stage refiner).

---

## 5. Phased roadmap

> Each phase is independently shippable. Parity is validated against the CPU reference at every gate. **Phase 0 captures the baseline BEFORE any port.**
>
> **Execution model (IMPORTANT — read before judging the estimates below).** The "~days / ~weeks" figures are *human-developer* estimates and are **obsolete** for how this will actually be built: a single parallel-agent **implementation workflow** authors and CPU-verifies all phases in **hours of wall-clock**, not weeks. The real cost variable is **not calendar time** — it is the number of **parity-verification rounds** the fragile Phase-3 core needs for its byte-identity tests to go green. So read the per-phase "Effort" as **verification difficulty**, not staffing time. The whole thing runs here (no GPU); the only step that touches an actual H100 is a ~5-minute Colab confirmation script the workflow hands you. Execution is **test-gated**: every phase writes its parity/regression test FIRST and is not "done" until that test passes under `pytest`. See `plan/gpu-refactor-build-spec.md` for the executable task breakdown and `plan/gpu_implementation_workflow.js` for the workflow that runs it.

### Phase 0 — Parity harness + baseline capture (PREREQUISITE) — ~2–3 days
- **Goal:** a golden, reproducible baseline of detector outputs and scores so every later change is diffed against it. No ports yet.
- **Files:** NEW `tests/test_parity_golden.py`, NEW `temp/capture_baseline.py` (writes golden `signal_high/low` arrays + per-fold scores + final LCB to `results/diag/golden/` for SPX + DAX + ≥1 short stream, seed=42).
- **Validation:** assert the golden capture reproduces bit-for-bit across two runs; record the existing `FastDetector` vs `SpeculatorDetector` `np.array_equal` pass and `FastPooledScorer` vs `PooledScorer` `abs<=1e-12` pass as the reference contract.
- **Effort:** Low. **Risk:** none. Ships a regression net.

### Phase 0.5 — PIR byte-identity spike (GATING) — ~2–3 days
- **Goal:** PROVE a GPU/torch float32 rolling-mean + rolling-min/max can match `pandas.rolling().mean()` / `pir_of` to `np.array_equal` on a real SPX IS slice. This de-risks the single hardest parity task before committing kernel weeks.
- **Files:** NEW `temp/pir_parity_spike.py`.
- **Validation:** `np.array_equal(gpu_pir_f32, cpu_pir_f32)` on a 5.7k-bar slice across all 499 scales. If it fails: fall back to per-scale pandas-order summation OR accept GPU-as-noisy-ranker and record the measured drift to size top-K.
- **Effort:** Low. **Risk:** this is the make-or-break finding for the whole GPU port — surface it early.

### Phase 1 — Batch-hungry optimizer on the EXISTING scorer (HIGHEST ROI) — ~3–5 days
- **Goal:** replace coordinate ascent with Sobol → sep-CMA-ES that emits 256–1024 candidates/round, scored via the **existing** `FastPooledScorer` parallelized across candidates with `multiprocessing` (or numba). **Zero new parity surface** — `score(params)→float` is unchanged.
- **Files:** NEW `src/v17_search.py` (EvoTorch sep-CMA-ES driver + Sobol seeding + top-K extraction); MODIFY `src/v17_runner.py` (route SEARCH through the new producer; keep coordinate ascent as a reference fallback). Fold `firing_excess` into the batched objective in `src/v17_acceptance.py`.
- **Validation:** the winner re-scored by the real `PooledScorer` must match the reported LCB to `<1e-9`; per-asset TV audit unchanged. Golden Phase-0 contract still passes.
- **Effort:** Low-Medium. **Risk:** low (no kernel change). This delivers the optimizer **quality** win (escaping coordinate ascent's greedy basin, exploiting threshold coupling) AND is the **prerequisite that lets the GPU parallelize** — coordinate ascent emits one candidate at a time and cannot fill a GPU. Benchmark it to confirm the *quality* gain is real (hardware-independent); the *throughput / wall-clock* win still requires the GPU evaluator on capped-CPU Colab. This is groundwork for the GPU, not a replacement for it.

### Phase 2 — Validation fixes + selection-bias guards (CPU, parity-orthogonal) — ~1–1.5 wk
- **Goal:** fix fold overlap + dead margins; add PBO/deflation scaled to the new batch size.
- **Files:** NEW `src/cpcv.py` (CPCV group partition + purge; **justified on overlap, not a leak**); NEW `src/overfit_guard.py` (PBO via CSCV + trial-count deflation); MODIFY `src/pooled_validation.py` (add `_run_cpcv_loop` alongside `_run_fold_loop`; mask dead 200-bar margins in `_stream_stat`; add per-asset-aggregate HIGH diagnostic); MODIFY `src/v17_runner.py` (wire deflation into the reported number).
- **Validation:** NEW `tests/test_cpcv_purge.py` asserts no IS group's label window overlaps any test group; assert LOW-side path count ≥ fold count; for HIGH, assert PBO/percentile are reported as **advisory** (event count + Wilson interval is the primary summary). Existing `PooledScorer` results unchanged when CPCV is off (default).
- **Effort:** Medium. **Risk:** low (CPU-only). Ships the validation fix even if GPU work slips.

### Phase 3 — Batched GPU evaluator (`score_pop`) — ~2–3 wk
- **Goal:** PyTorch float64 (PIR float32) batched evaluator: Phase-1 precompute PER SLICE + threshold layer + segmented batched Phase-2 scan + drift pre-pass. Byte-identical to `FastDetector`.
- **Files:** NEW `src/v17_gpu/eval_torch.py` (precompute + threshold + batched scan over candidate axis via `torch.func.vmap`); NEW `src/v17_gpu/drift_precompute.py` (cumulative-count + last-K gather replacing the growing stack; reproduces `None→0.0`, `Kmax≥20`, HIGH-before-LOW, read-before-append); NEW `src/v17_gpu/upload.py` (slice→device padded packer + valid-mask + length bucketing + scale-tiling); NEW `tests/test_v17_gpu_parity.py`.
- **Validation (the load-bearing gate):** `np.array_equal(gpu.signal_high, SpeculatorDetector.signal_high)` and `_low` on EVERY slice for a sample of params; `abs<1e-9` on LCB. Explicit edge-case tests: ring truncation, read-before-append, HIGH-before-LOW, warm-up-NaN first 300 bars, segmented per-stream reset (no cooldown leak across assets). FAIL the run on any mismatch.
- **Effort:** High (the parity-fragile core). **Risk:** high — concentrated here. Gate it behind Phase 0.5's spike result.

### Phase 4 — End-to-end integration + tuning — ~0.5–1 wk
- **Goal:** GPU search → CPU finalist re-score (top-K calibrated to measured GPU-vs-CPU rank error) → acceptance gates → per-asset TV-export audit on a real multi-asset pool. Tune λ, CPCV N/k, scale-tiling.
- **Files:** MODIFY `src/v17_runner.py` (full pipeline wiring; generalize the `|fast-real|>1e-9` warn into a hard finalist filter).
- **Validation:** full per-asset TV-export parity audit (count diff == 0 per instrument) on SPX + DAX + the pool. Golden Phase-0 contract intact.
- **Effort:** Medium. **Risk:** medium (HBM pressure → chunking tuning).

### Optional Phase 5 — JAX/CUDA-graph scan kernel (only if profiled-needed) — ~1 wk
- **Goal:** if v1's Python bar-loop dispatch dominates on wide batches, wrap the scan body in a CUDA graph (static shapes) or port the single scan kernel to JAX `lax.scan` behind the same `score_pop` interface.
- **Validation:** re-run `tests/test_v17_gpu_parity.py` unchanged.
- **Effort:** Medium. **Risk:** isolated.

**Total to a working, faster multi-asset optimizer with fixed validation:** Phases 0→2 ≈ **2.5–3 wk** (ships the optimizer + validation win, no GPU kernel). Add Phases 3→4 for the full GPU port ≈ **+3–4 wk**.

---

## 6. Top risks + mitigations

| # | Risk | Likelihood | Mitigation |
|---|---|---|---|
| 1 | **GPU PIR not byte-identical to pandas** (cumsum drift flips agreement by 1/n_scales) | High | **Phase 0.5 gating spike** before kernel work; per-scale pandas-order fallback; or GPU-as-noisy-ranker with top-K sized to measured drift |
| 2 | **Phase-2 scan parity bugs** (ring `Kmax<20`, read-before-append, HIGH-before-LOW, warm-up NaN, segmented reset) | High | `Kmax≥20` from frozen Params; pinned carry-update order; explicit edge-case unit tests; FAIL run on any `np.array_equal` mismatch |
| 3 | **H100 sits idle** if optimizer stays sequential | Certain (if not fixed) | Phase 1 swaps to batch-hungry CMA-ES FIRST; GPU port only after the producer can fill it |
| 4 | **Overfitting the ~27-event HIGH side** with a wider search | High | Firing penalty in batched objective; deflation scaled to λ×generations; PBO; HIGH = monitor-only with event-count + wide interval |
| 5 | **HBM blowout** (114 GB/asset PIR, 467 GB vote tensor) | Certain at full batch | Mandatory scale-tiling + candidate-chunking + length-bucketing; re-estimate speedup with chunking |
| 6 | **GARCH built per-asset not per-slice** → warm-up divergence at every slice boundary | Medium | Build artifacts PER SLICE with slice-local seeds; sequential GARCH kernel (amortized, not a bottleneck) |
| 7 | **Two backends drift** (torch GPU vs pandas CPU) as detector evolves | Medium | Golden Phase-0 contract + `test_v17_gpu_parity` as hard CI gates on every detector change |
| 8 | **JAX-on-Colab toolchain fragility** | Medium | Choose PyTorch for v1; JAX is opt-in Phase 5 only |

### What to do FIRST (cheapest de-risk)
1. **Phase 0** golden baseline (1 day of value, permanent regression net).
2. **Phase 0.5 PIR spike** (2 days) — the single finding that determines whether the GPU port is byte-feasible at all.
3. **Phase 1 optimizer swap** — zero new parity surface, and the **prerequisite** that lets the GPU batch (coordinate ascent can't feed a GPU). Benchmark it to confirm the optimizer-**quality** gain is real before investing in the kernel.

The Phase-1 CPU benchmark answers *"is the new search better?"* (hardware-independent) — **not** *"is it fast enough?"*. On capped-CPU Colab the throughput / wall-clock target requires the GPU evaluator; Phase 1 is the groundwork that makes that GPU step usable, **not an alternative to it**.

---

## 7. Where the H100 WILL and WILL NOT help (honest expectations)

**WILL help:**
- Collapsing a whole CMA-ES generation (256–1024 candidates) into ~one slice-scan-depth of wall-clock once the optimizer emits a wide batch.
- Pooling MORE assets at near-zero marginal wall-clock — the only genuine lever on the HIGH-side event count.
- Removing per-candidate Python `for t` loop + pandas overhead across the `(candidate × stream × fold)` grid.

**Realistic speedup:** **~10–40× end-to-end search** on a multi-asset pool, bounded below by (a) the longest per-lane scan depth (~5.7k IS bars, not 57k), (b) FP64 at half throughput, (c) mandatory candidate-chunking under HBM limits, (d) the CPCV ~3–5× slice multiplier, and (e) the serial CPU re-score tail of top-K finalists. **NOT 100×, NOT 50–200×.**

**Will NOT help:**
- **Statistical power.** ~27 validatable HIGH tops in 155 yr is a *data* ceiling; an H100 evaluates more configs but cannot create informative events. More candidates *increases* multiple-comparison risk — the deflation/PBO guards exist precisely to claw that back.
- **Parity.** The H100 changes how candidates are *proposed*, never how they are *judged*. The float64 CPU detector + per-asset TV audit remain the sole export gate.
- **An idle device is worse than no device.** Without the Phase-1 optimizer swap, the H100 is the "expensive idle device" the engineer warned about — coordinate ascent would keep it ~99% idle. The optimizer change is what makes the hardware earn its cost.

---

## 8. File-change summary

**NEW:** `src/v17_search.py`, `src/cpcv.py`, `src/overfit_guard.py`, `src/v17_gpu/eval_torch.py`, `src/v17_gpu/drift_precompute.py`, `src/v17_gpu/upload.py`, `tests/test_parity_golden.py`, `tests/test_cpcv_purge.py`, `tests/test_v17_gpu_parity.py`, `temp/capture_baseline.py`, `temp/pir_parity_spike.py`.

**MODIFIED (surgical):** `src/v17_runner.py` (optimizer routing, deflation wiring, hard finalist filter), `src/pooled_validation.py` (`_run_cpcv_loop`, dead-margin masking, per-asset HIGH diagnostic), `src/v17_acceptance.py` (firing penalty into batched objective, deflation hook).

**UNCHANGED (the trust root — DO NOT TOUCH the math):** `src/detector.py`, `src/indicators.py`, `src/v17_fastdetector.py` (kept as the CPU byte-identity oracle), `src/parity.py`, `src/scoring.py`, `src/pooled_scoring.py`, `src/validation.py`, all Pine files. Coordinate ascent (`src/v17_optimize.py`) stays as a reference fallback.
