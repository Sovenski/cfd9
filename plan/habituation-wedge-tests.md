# Habituation-Wedge Tests — pre-registration (locked before results)

**Thesis (2026-06-11):** weak agreement has two causes — (a) nothing is
happening, (b) the ruler stretched during the trend it measures. They are
distinguishable, and the (b)-bars near real turns are recoverable at
acceptable FP cost. Motivated by the miss autopsy: HIGH misses are ~50%
weak/brief-extremeness-blocked; LOW misses are price-gate-dominated (61%).

## The instrument: anchored re-normalization (the "wedge")

For each scale s in the side's preset slice, `ratio_s = close / SMA_s` (the
exact existing feature). Two normalizations of the SAME ratio:

- **bar-PIR** (status quo): position of ratio_s in its rolling `max(s,20)`-bar
  min/max — the habituating ruler.
- **anchored-PIR**: position of ratio_s in its min/max over `[anchor, t]`,
  where **anchor = bar index of the last CONFIRMED opposite structural pivot**
  (HIGH: last confirmed pivot-low; LOW: last confirmed pivot-high;
  `pivot_low_pine/high_pine(n=20)`, confirmed 20 bars later — fully causal).
  NaN if the anchor window is < 5 bars old.

**wedge[t] = mean_s(anchored_PIR) − mean_s(bar_PIR)** (HIGH; LOW mirrored on
the 1−PIR side). Positive wedge = the bar ruler under-reads vs the swing ruler
= habituation. `agr_anch[t]` = fraction of slice scales whose anchored-PIR is
extreme at the side's own `pct_extreme`.

## Pre-registered tests and rules (decided before running)

| # | Question | PASS rule |
|---|---|---|
| **P2 (thesis core)** | Is the wedge LARGE at agreement/dur-blocked missed turns and SMALL at weak-agreement nothing-bars (no turn within ±5)? | median wedge difference ≥ **0.10**, cluster bootstrap (2000, by stream) p < 0.05, per side |
| **P1 (economics)** | Recovered-only bars `R = {agr_anch ≥ min_agreement AND agr_bar < min_agreement}`: is precision(R) ≥ current fire precision (HIGH 0.240 / LOW 0.286)? And does R recover ≥ **30%** of the agreement/dur-blocked misses? | both conditions, bootstrap p < 0.05 on the precision comparison |
| **P3 (payoff, report-only)** | Median forward trade-direction return of R at 20/40-bar horizons vs random bars | no gate — informational |
| **P4b (LOW-specific)** | Among LOW's price-gate-blocked misses (61%), what fraction is `agr_anch`-extreme, and what is the FP price of an anchored OR-trigger there? | same form as P1, LOW numbers |
| **P5 (robustness)** | Re-run P1/P2 with anchor = 2nd-last opposite pivot, and report P1 at pct_extreme ±0.05 | direction must hold (no numeric gate) |

**Build rule:** a v18 branch for a side requires **P2 AND P1 both PASS** on
that side. P3/P5 inform design, never gate. Everything runs pooled on the 16
streams, real `SpeculatorDetector` (debug columns), ground truth and blocking
attribution identical to `temp/miss_autopsy.py`.

**Anti-fooling notes:** anchors use confirmed pivots only (20-bar lag, causal);
recovered-set precision is measured on ALL eligible bars (not conditioned on
fires); the nothing-bars control is matched on weak agreement so P2 cannot
pass by "weak bars differ from strong bars" alone.

---

## RESULTS (2026-06-11, temp/habituation_wedge.py, anchor=1)

| Test | HIGH | LOW |
|---|---|---|
| **P2 thesis** | **PASS** — wedge at blocked misses **+0.367** vs nothing-bars +0.008 (p=0.000) | **PASS** — **+0.399** vs +0.000 (p=0.000) |
| **P1 economics** | **FAIL** — recovered set = **17.6% of ALL bars**, precision 0.106 vs 0.240 required (p=1.0); recovery 98% | **FAIL** — 3.45% of bars, precision **0.016** vs 0.286 (p=1.0); recovery 99% |
| P3 payoff | recovered fwd20 −0.0018 vs all-bars −0.0029 (still negative for shorts) | +0.0033 vs +0.0029 (≈ random) |
| P4b | — | 99.8% of price-gate-blocked misses anchored-extreme (same saturation) |

**Verdict (per pre-registered build rule P2∧P1): NO v18 BUILD.**

**Interpretation:** the thesis's cause-separation claim is TRUE — "weak because
the ruler stretched" is cleanly distinguishable from "weak because nothing is
happening" (a genuine mechanistic finding). But the anchored ruler is extreme
on essentially the WHOLE of every trend: the recovered set is ~39× larger than
its targets, and shrinking it to acceptable precision requires exactly the
turn-vs-continuation discriminator that every prior test failed to find.
**Habituation explains WHY the model is silent at slow extremes; un-silencing
it is the original unsolved discrimination problem in a new coat.** P5
(anchor=2) skipped as moot: P1's failure mode (whole-trend admission) is
structural, not anchor-parameter-sensitive.

---

## Addendum: dip-buyer null baseline (pre-registered before running)

**Question:** does the LOW detector beat a one-line naive dip rule, or is it
repackaged "buy the dip + drift"?

**Naive rule:** fire when `low <= rolling N-bar min of low`, N calibrated PER
STREAM so the naive fire count matches the LOW preset's fire count (±10%),
same cooldown (7 bars) applied.

**Pre-registered rules:**
- **D1 (location):** precision(LOW fires) − precision(naive fires) ≥ **3pts**
  pooled, cluster bootstrap p < 0.05 → the detector's location skill adds
  value over the dummy.
- **D2 (payoff, report-only):** median fwd20/fwd40 long returns, LOW vs naive
  vs all-bars.
- If D1 fails → LOW's documented edge is naive-dip-replicable; Route D
  productization should use the simpler rule (or be re-examined).

### D1/D2 RESULTS (2026-06-11, temp/dip_baseline.py)

- **D1 location: FAIL.** Detector precision 0.286 vs naive dip-buyer 0.277 at
  matched firing rates (diff **+0.008**, p=0.022 — statistically real but far
  below the 3-point bar). The LOW detector locates bottoms essentially no
  better than `low <= rolling N-bar min` with a cooldown.
- **D2 payoff (report-only):** detector fires earn ~2x the naive forward
  return (fwd20 +0.28% vs +0.15%; fwd40 +0.40% vs +0.20%). Location is
  naive-replicable; the residual detector contribution, if any, is in WHICH
  dips it selects (payoff quality), not WHERE it fires. Unregistered metric —
  exploratory only.
- **Consequence (per pre-registration):** LOW's edge claim is downgraded —
  largely naive-dip + drift. Route D, if pursued, must either use the simpler
  rule or justify the detector by the payoff delta under a registered test.

### D3 — registered dip-SELECTION payoff test (confirmatory-on-same-data; hypothesis from D2)

Does the detector's dip selection beat the naive rule's dips on forward return?
- **D3a** pooled: cluster-bootstrap (2000, by stream) p < 0.05 on mean(det fwd) − mean(naive fwd) at BOTH 20 and 40 bars
- **D3b** consistency: per-stream median fwd20 OR fwd40 favors detector in ≥ 10/16 streams
- **D3c** effect floor: median delta ≥ +0.0010 (10bps) at fwd20
- PASS requires all three. Caveat recorded: same data that generated the hypothesis.

### D3 RESULTS (2026-06-11, temp/dip_selection_d3.py)

- D3a pooled significance: **FAIL** (p=0.157 fwd20 / 0.166 fwd40)
- D3b consistency: PASS (12/16 streams favor detector)
- D3c effect floor: PASS (median fwd20 delta +13bps)
- **Overall: FAIL** (all three required). The dip-selection payoff tilt is
  directionally consistent but statistically inseparable from zero under
  stream-cluster bootstrap. LOW's final evidence state: location =
  naive-replicable (D1), payoff tilt = unproven (D3). LOW = disciplined
  buy-the-dip; no demonstrated proprietary alpha beyond the risk/calibration
  layer.

### Ablation run (131405) + new-LOW null tests (2026-06-11)

**Ablation noise map (15-gen LOO, holdout deltas):** LOW — gjr_asym −0.003
(noise/harmful), har_vol +0.007 (noise band), volume +0.023 top (SUSPECT:
volume vote is structurally dead on 9/16 price_only/FX streams — audit flag),
rest +0.013..0.016. HIGH — all deltas positive (+0.011..+0.042) BUT the all_on
winner pinned 8 thresholds at maximally-PERMISSIVE floors (votes neutralized)
and raw LCB fell 0.093→0.034 vs the lean 090553 winner: forcing votes on hurt;
the positive deltas are largely vote-count arithmetic, not information.

**New LOW (minus_momentum, deflated 0.0678 — best ever, holdout 0.95):**
- D1 location: FAIL again — 0.290 vs naive 0.282 (+0.8pts, p=0.129). THIRD
  preset producing the same +0.8pts: location is naive-replicable, robustly.
- D3 payoff full-period: +29/+33bps, p=0.002, 14/16 — but TRUE holdout-tail
  only (post-2015, n=468): +19bps p=0.128 / +15bps p=0.349, 11/16. The
  full-period significance is fit-period inflated. Dip-selection edge remains
  directionally positive, OOS-unproven.
