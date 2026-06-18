# GPU Refactor — Implementation Report (spec §7 Definition of Done)

**Date:** 2026-06-10
**Branch:** `feature/v17-antigaming` (working tree only — no commits, per spec §0.4)
**Spec:** `plan/gpu-refactor-build-spec.md` · **Design:** `plan/gpu-refactor-plan.md`

---

## 1. Final verification results

| Gate | Result |
|---|---|
| Full suite `python -m pytest -q` | **GREEN — 146 passed, 0 failed** (exit 0; 13 new helper tests on top of the 133-test post-build suite) |
| `git diff --name-only` (tracked) | **`src/pooled_validation.py`, `src/v17_runner.py` ONLY** — both on the §0.1 allowed surgical list; **zero oracle files modified** |
| Phase-0 golden contract | **Intact** — `tests/test_parity_golden.py` (9 tests: C1 capture determinism, C2 golden-on-disk == fresh capture, C3 FastDetector `np.array_equal`, C4 FastPooledScorer ≤1e-12) green against the untouched `results/diag/golden/` snapshot |
| §2 PIR-spike branch | **trust-kernel** (pandas-faithful float64 window sums, cast float32 → byte-identical on CPU-torch; flip rate exactly 0) |

Only warnings: the known `RuntimeWarning: invalid value encountered in log`
from `src/indicators.py` (lines 331/396) during `tests/test_pooled_validation.py`
— expected warm-up NaN behavior, present since before this refactor.

## 2. §0.1 compliance repair performed at this gate

Earlier phases had left **additive but rule-prohibited** edits inside three
trust-root files (`src/v17_fastdetector.py`, `src/v17_optimize.py`,
`src/validation.py`): `firing_penalty`/`firing_cap` per-fold regularization, a
`fold_scores()` raw-scores accessor, and a `seed` parameter on
`fold_scores_bootstrap_ci`. Both phase verifiers flagged this as must-fix.
Resolution (this session, TDD):

- **Reverted** all three oracle files to their committed bytes
  (`git checkout -- src/v17_fastdetector.py src/v17_optimize.py src/validation.py`).
- **Re-homed the functionality additively** in `src/v17_acceptance.py`
  (allowed file):
  - `raw_fold_scores(scorer, params)` — raw per-fold scores for
    `PooledScorer` / `FastPooledScorer` without touching the oracle classes;
  - `PenalizedScorer` (frozen dataclass) — `score()` wrapper applying the
    per-fold firing penalty for the batched search only;
  - `_bootstrap_ci_seeded(...)` — local seed-parameterized copy of
    `validation.fold_scores_bootstrap_ci` used by `bootstrap_stability`;
    asserted **exactly equal** to the oracle at `seed=42`.
- **Consumers updated:** `src/v17_runner.py` (acceptance gate now calls
  `raw_fold_scores(real, wp)`), `tests/test_cpcv_purge.py`,
  `tests/test_v17_gpu_parity.py`, `temp/capture_baseline.py`,
  `temp/adv_p7_pooled_attack{,2}.py`, `src/v17_search.py` (doc comments).
- **New gate tests:** `tests/test_v17_acceptance_helpers.py` (13 tests, H1–H4):
  seeded-CI == oracle at seed 42 (incl. empty/single/block-clamp edges), seed
  determinism + sensitivity, penalty math on stubs, and real-data SPX identity
  (`raw_fold_scores` real-vs-fast ≤1e-12; bootstrap of raw scores reproduces
  `score()` bit-for-bit; `PenalizedScorer` with penalty off == `score()`).

The golden snapshot did **not** need recapture: `raw_fold_scores` reproduces
the removed `fold_scores()` numbers exactly (asserted by the golden regression
in `test_cpcv_purge.py` and the C1/C2 capture tests).

## 3. Files created / modified

**Modified (tracked — the only two, both §0.1-allowed):**
- `src/v17_runner.py` — `search="cma"` route (§3), deflation + per-asset HIGH
  diagnostic wiring (§4), `run_v17_gpu` with hard finalist filter (§6),
  `raw_fold_scores` usage (this gate). Default `search="ascent"` byte-identical.
- `src/pooled_validation.py` — `_run_cpcv_loop` alongside `_run_fold_loop`
  (default OFF), dead-margin masking in `_stream_stat`, per-asset HIGH
  diagnostic (§4). 535 lines (above the ~400 guideline; kept in place because
  §4 names this exact file for the surgical additions).

**New source:**
- `src/v17_acceptance.py` — firing penalty, penalized ranking, boundary-pin,
  bootstrap stability, acceptance verdict + the §0.1-compliance helpers above.
- `src/v17_search.py` — `BatchOptimizer`: Sobol seeding + separable CMA-ES,
  shape params frozen, top-K finalists (§3).
- `src/cpcv.py` — purged Combinatorial CV with 200-bar embargo (§4).
- `src/overfit_guard.py` — PBO via CSCV + deflated-best haircut (§4).
- `src/v17_finalists.py` — finalist filtering, top-K sizing, TV-export audit (§6).
- `src/v17_gpu/` — `__init__.py`, `drift_precompute.py` (P5),
  `upload.py` (P7 packing), `eval_torch.py` + `phase1_features.py` (P1–P4, P6,
  P8 feature/threshold layer), `phase2_scan.py` (batched scan + `score_pop`,
  `GpuPooledScorer`) (§5).

**New tests:** `test_parity_golden.py`, `test_v17_search.py`,
`test_cpcv_purge.py`, `test_v17_acceptance.py`, `test_v17_acceptance_helpers.py`,
`test_v17_gpu_drift.py`, `test_v17_gpu_upload.py`, `test_v17_gpu_eval_phase1.py`,
`test_v17_gpu_parity.py` (the §0.7 load-bearing gate), `test_v17_gpu_integration.py`.

**New scripts / artifacts:** `temp/capture_baseline.py`,
`temp/pir_parity_spike.py`, `temp/colab_h100_validation.py`,
`results/diag/golden/{golden_SPX.npz, golden_DAX.npz, golden_baseline.json}`.

## 4. §2 PIR spike — branch taken

**`trust-kernel`.** The pandas-faithful per-window float64 sum (torch unfold),
cast to float32, is `np.array_equal` to `indicators.precompute_matrices` on the
real 5.7k-bar SPX IS slice across all 499 scales: 0 differing cells, 0 NaN-mask
mismatches, value- and vote-flip rates exactly 0. The cumsum-difference f32
variant drifts (2.6e-2) and is dead, as the spec hypothesized. Consequently all
§5 parity tests assert strict `np.array_equal`; `topk_for_flip_rate(0.0)` keeps
top-K as a sanity floor only. Residual risk (H100 CUDA reduction order, TF32
off) is exactly what the Colab script re-measures on hardware.

## 5. Colab H100 steps (the user runs these)

The self-contained ~5-minute confirmation is `temp/colab_h100_validation.py`.
On a Colab **H100 GPU runtime**:

```bash
# 1. Get the working tree onto the runtime (zip/drive/git — any way), then:
%cd /content/cfd9        # repo root: src/, tests/, temp/, data/raw/ must exist

# 2. Make sure both daily CSVs are present:
#    data/raw/SPX_1D_18710201_20260318.csv
#    data/raw/DAX_1D_19700102_20260324.csv

# 3. Run the validation (installs torch-cuda itself if missing):
!python temp/colab_h100_validation.py
```

It executes, in order: (1) torch+CUDA check, (2) the §2 PIR spike ON the GPU,
(3) `tests/test_v17_gpu_parity.py` ON the GPU via a device-redefault pytest
plugin, (4) an end-to-end GPU-vs-`SpeculatorDetector` signal-flip measurement
(SPX+DAX tails × 4 param draws), (5) one tiny `run_v17_gpu` end-to-end run.

**Read the last line:** `H100 VALIDATION: PASS` requires parity tests green,
flip rate exactly 0.0 and 0 dropped finalists → trust-kernel confirmed on
hardware. If the flip rate is nonzero the script prints the revised
`noisy-ranker` instruction with the `top_k` to pass to `run_v17_gpu`
(`topk_for_flip_rate`). Exit code 0 = PASS, 1 = FAIL.

## 6. Completeness critic — outstanding items

1. **H100 hardware confirmation is pending by design** — byte-identity is
   proven on CPU-torch only; CUDA reduction order + TF32-off (P1) must be
   confirmed by the Colab run above (spec defers this to Phase 4).
2. **P8 linreg** uses spec option (a): `np.polyfit` semantics on the CPU
   precompute path; `TorchPhase1` consumes the same precomputed arrays rather
   than a torch OLS. Documented, not a gap — but a future fully-on-GPU
   linreg would need the noisy-ranker treatment.
3. **`PenalizedScorer` is available but not wired into the default `run_v17`
   cma route** (the route constructs the raw scorer; penalization is opt-in by
   wrapping, exactly like the previous in-scorer `firing_penalty=0.0` default).
   Reported numbers always come from the RAW objective, per the acceptance
   contract.
4. **CPCV (`_run_cpcv_loop`) keeps only the longest contiguous purged train
   range as IS** (the detector needs contiguous bars) — fewer IS bars than
   theoretically available; documented in §4 verification as non-blocking.
5. **`src/pooled_validation.py` is 535 lines** (> the ~400 style guideline);
   splitting it would mean restructuring a surgical file beyond the additive
   mandate, so it was left in place.
6. **Golden LCB is genuinely 0.0 for base `Params()`** on both assets/sides
   (true 90% block-bootstrap lower bound over mostly-zero fold scores); the
   high-resolution per-fold scores and signal arrays are the sensitive
   regression fingerprint (documented in `temp/capture_baseline.py`).
7. No TODO/FIXME/XXX markers remain in `src/` or `tests/`.

## 7. Blockers

None. The previously recorded blocker (trust-root edits from Phase 1) is
resolved by the §0.1 compliance repair in section 2.
