# v18 Repair Spec — un-painting the vote knobs

**Approved:** "do both phases now" (2026-06-11). **Origin:** parity-anchored
vote audits (temp/vote_contribution.py, temp/vote_bounds_audit.py): 3.5 of 7
searched vote dims were structurally incapable of selectivity.

**v18 = the feature space v17 claimed to have.** No new mechanisms; repairs only.
Scorer stays v5.1. Runner stamps `detector: "v18"`.

## Phase 1 — bounds repair (search_space.py only; zero parity surface)

| dim | old box | new box | basis (measured pass-rates) |
|---|---|---|---|
| slope_thresh | [0.01, 0.5] | **[0.1, 6.0]** | value is ×1000-scaled, dist −12..+9; new box spans ~46%→~5% |
| vol_surge_thresh | [1.0, 3.0] | **[1.0, 1.5]** | ratio q99=1.78; old upper half dead; new spans 19%→~3% |
| gjr_vote_thresh | [0.05, 0.5] | **[0.5, 3.0]** | paired with P2.2 unclip; unclipped dist spans 40%→2% |
| har_vote_thresh | [0.05, 0.5] | unchanged | working range 19–41%, borderline OK |
| NEW: momentum_diverge_thresh | — | **[0.0, 0.02]** | sign-test 18% at 0.0 → ~2–4% at 0.02; 0.0 = legacy |

tests/test_search_space.py re-pinned with rationale comments.

## Phase 2 — feature repair (parity cycle; frozen files re-opened under v18)

- **P2.1 pir_of partial-window:** `rolling(lb)` → `rolling(lb, min_periods=1)`
  in `src/indicators.py::pir_of` — matches Pine partial-window scan (proven by
  the e830d audit flips) and precompute_matrices' documented semantics.
  Propagates to detector/Fast/GPU via shared use.
- **P2.2 gjr unclip:** `gjr_asym_norm = (ratio−1)/0.1` WITHOUT `clip(±1)`
  (`calc_gjr_asym`). Artifact-propagated to Fast/GPU automatically. Pine v18
  gjr block edited identically. Defaults off → golden untouched by this.
- **P2.3 momentum threshold:** new Params `momentum_diverge_thresh_{side}: float = 0.0`.
  Vote becomes `mom_diverge < -thresh` (exact current direction per side
  verified from detector source; 0.0 reproduces the legacy sign test
  BIT-EXACTLY — pinned by test). Mirrored in detector.py, v17_fastdetector.py,
  v17_gpu/eval_torch.py (+ phase1 plumbing if needed), Pine.
- **P2.4 drift requirable:** new Params `count_drift_vote_{side}: bool = False`.
  When True, max_votes += 1 (drift joins the requirable pool) in all three
  engines + Pine. False = legacy bit-exact (pinned).
- **P2.5 search wiring (built = wired):** momentum_diverge_thresh in float dims,
  count_drift_vote in bool dims, both sides, v17_search + notebook untouched
  defaults.
- **P2.6 Pine v18:** `pine/speculatores_v18_signalcard.pine` from the v17.5
  lineage (presets carried, incl. v18-lean): gjr unclip, two new inputs per
  side (momentum_diverge_thresh default 0.0, count_drift_vote default false)
  with per-preset ternaries preserving every shipped preset's behavior.
- **P2.7 runner stamp:** `out["detector"] = "v18"`.
- **Workbook for the 30-gen run:** GENERATIONS=30, RUN_ABLATION=False,
  RUN_PRUNED=False, fresh seed 211; cell-pin re-pinned.

## Acceptance gate (all must pass before push — "local parity test first")

1. Full pytest green; NEW byte-identity tests: `Params()` defaults and every
   shipped preset produce IDENTICAL signal arrays pre/post-v18 on the golden
   slices, EXCEPT divergences attributable to P2.1 warm-up bars (must be
   enumerated explicitly if any).
2. **TV-export regression (the local Pine parity test):**
   `SP_SPX, 1D_e830d.csv` (v18-lean preset): HIGH flips 5 → **0** (all bars);
   LOW stays 0. `SP_SPX, 1D_e7bb6.csv` (v17.5 audit PASS): **still PASS**.
3. Golden: if P2.1 changes any golden array, deliberate re-pin with a byte-diff
   report (which arrays, which bars, why warm-up).
4. Trust kernel: FastDetector & GPU byte-identity on the tiny pool with the
   NEW params exercised (thresh>0, count_drift=True).
5. Freeze ledger updated: v18 re-freeze list = v17 list + the v18 diffs.
