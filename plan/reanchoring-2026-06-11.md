# Re-anchoring — what this system is, what it provably does, and the path to an edge

**Date:** 2026-06-11 · **Status:** strategy anchor (written while the feature-ablation run is in flight)
**Rule for this document:** every claim carries its evidence grade.
Grades: **[SOLID]** = pooled/OOS/counting, would bet on it · **[GOOD]** = clean test, one dataset or mild caveats · **[HYP]** = surrogate/in-sample, hypothesis only.

---

## 1. What the system is DESIGNED to do

Three layers, one contract:

1. **Detector (Pine + exact Python twin).** A multi-scale geometric *turn locator*: fire when price sits at an extreme of its position-in-range across many SMA scales simultaneously (agreement), confirmed by auxiliary votes (trend, volatility, momentum-velocity, drift, gjr/har), gated by structural-pivot proximity and debounced by cooldown. Designed promise per fire: *"this bar is likely a pivot high/low of meaningful span."*
2. **Optimizer (GPU search + CPU exact re-score).** Find per-side parameterizations whose fires match real labeled pivots — scorer v5.1: span-weighted ±1 matching, false positives priced at W_FP=0.5, recall targeted against reference mass — validated by walk-forward folds, embargoed holdout (era_pass), bootstrap stability, boundary-pin auto-reject, deflation, firing cap.
3. **Signal card.** Per-fire calibrated stats (KM survival bands, expected move, span-clock hold, pivot stop) so a human can judge each signal.

**The identity constraint:** Pine↔Python parity. Every number reported must be reproducible on a TradingView chart. This rules out: black-box models in the signal path, cross-asset inputs, anything not expressible in Pine.

**The data constraints:** 1D bars; ~27 validatable T1 HIGH events in 155y of SPX (the information limit); volume usable only on equities/ETFs (FX has none); no options/breadth data.

## 2. What it REALLY does — the evidence ledger

**It locates turns far better than chance.** Pooled, 16 streams: HIGH precision 0.24 vs 0.07 random base (2.3×); LOW 0.29 vs 0.08 (4.3×). The signal-vs-random event study shows a sharp V into the fire bar vs the smooth drift line. **[SOLID]**

**That skill currently converts to ~zero directional edge.** Forward path of fired signals ≈ forward path of random bars (HIGH exactly on the drift line; LOW +0.3%/20 bars above it). Both sides sit almost exactly at their break-even precision. **[SOLID]**

**Edge is ~linear in precision; precision is the only lever.** Blended hit/miss returns vs random: each +10pts precision ≈ +0.4–0.5% per trade at the 20–40-bar horizon. Break-even ≈ 15–20% HIGH / 25–35% LOW (label-optimistic bounds). **[GOOD]** (SPX heatmap; pattern robust, absolute levels flattered by ±2 labeling)

**The two sides are economically different animals.**
- LOW: bottoms have ~2× the base rate, resolve UP with the drift, pay at SHORT holds (~40 bars). LOW-pruned: holdout 1.73× fold-mean (era_pass TRUE), backtest +0.72R/trade before costs. **[SOLID]** — **DOWNGRADED 2026-06-11 (D1 null test):** location precision is naive-dip-replicable (0.286 vs 0.277 for `low<=N-bar min` at matched firing rate; +0.8pts, below the 3pt bar). The residual detector contribution is dip *selection* (fires earn ~2× naive forward return — exploratory, unregistered), not location. LOW ≈ disciplined buy-the-dip + drift, plus possibly better dip-quality picking + the stops/calibration layer.
- HIGH: fights the drift, real tops decay SLOWLY (edge, if precision were high, peaks at ~250–350-bar holds), backtest −31R. Currently a candidate-locator without a resolver. **[SOLID]**

**At the fire bar, hits and misses look nearly identical in everything tested so far.** Approach price paths overlap almost exactly (the separation is all aftermath); hit-vs-miss discrimination AUCs: price geometry ~0.5 pooled, vol-structure 0.45–0.54, detector's own continuous features 0.52–0.53 (only LOW top-decile +10pts survives leave-streams-out, p=0.018). **[GOOD]** — with the standing caveat that these used *surrogate* feature sets, not the full Pine machinery.

**The model generalizes in time.** Both 090553 winners pass the embargoed never-searched holdout (HIGH 0.66×, LOW 1.21× fold-mean — and the prior big run 0.80×/1.73×). The scoring edge is not a fold artifact. **[SOLID]**

**The validation machinery works.** Pins correctly flagged box-edge sitters; deflation killed spray regimes; v5.1 re-pricing + widened bounds moved the winners materially; cross-seed replication confirmed real basins. **[SOLID]** (process claim)

**Reference-frame thesis: tested and not supported (2026-06-11, three pre-registered tests, plan/v18-frame-aware-tests-and-spec.md).** Frame events (window-expiry rotations) are mechanically real but carry no precision signal at fire bars (dose-response flat, p≈0.64); cross-scale freshness is degenerate at fires (sync==0 at ~100% — fires are stale-extremeness events, onset info lives upstream of the fire bar); event-time PIR beats bar-time only on HIGH and below the registered bar (Δ+0.034). **No v18 detector change.** **[SOLID]** (negative)

**Habituation: mechanism confirmed, unexploitable (2026-06-11, pre-registered
wedge tests, plan/habituation-wedge-tests.md).** "Weak agreement because the
ruler stretched" is cleanly separable from "weak because nothing happening"
(wedge +0.37/+0.40 vs ~0.00, p=0.000 both sides) and explains ~half of HIGH's
missed tops — but every recovery trigger admits whole trends (recovered set
17.6% of all bars at 0.106 precision vs 0.240 required). Un-silencing the
model at slow extremes requires the same turn-vs-continuation discriminator
that does not exist in current information. **[SOLID]** (mechanism positive,
economics negative) **No v18 build.**

**Open / hypothesis-grade:**
- The true precision frontier of the *real Pine-expressible space* is unmeasured. Surrogate ceilings (v1: no HIGH pocket; v2 with added families: pockets both sides) proved only that ceiling estimates are hostage to feature-list choices. **[HYP]**
- Duration/long-memory ("bars since extreme", from-ATH) and continuous short-slope looked like the strongest *new* discriminator candidates in surrogate space. **[HYP]** — not in the Pine script today; adopting them is a design decision, not a finding.
- Volume on equities: untested, genuinely new information. **[HYP]**

## 3. The gap, stated exactly

The system is a **calibrated candidate-extreme locator operating at break-even precision**. An edge requires pushing hit-precision a handful of points past break-even *at a horizon the side actually pays on* — OR accepting the side that is already marginally past it and engineering around the gap.

Three honest routes (+1 default):

- **Route A — squeeze the existing space.** Maybe the optimizer's objective (recall-targeted) parks it away from sparser/higher-precision pockets that exist in-space. Instruments: the in-flight ablation (which votes matter, in-architecture), then a local Sobol *precision-frontier* sweep over the real FastDetector (no surrogates: real gates, real votes), then — if a pocket shows — a precision-tilted objective variant for one GPU run. Cheapest route, fully inside the parity contract.
- **Route B — select among fires.** Card-side conviction tilt. Only one effect survived pooled leave-streams-out: LOW top-decile +10pts. Usable as a sizing tilt, not a system-maker. Display-side, zero parity risk.
- **Route C — add one new information family.** Only justified AFTER Route A's frontier says the existing space is exhausted, and then as a deliberate spec'd extension (the gjr/har pattern: new optional vote, parity-built, TV-audited). Current candidates ranked by evidence: duration/long-memory, continuous trend-slope, volume (equities-only). **Not before the baseline exists.**
- **Route D — specialize in what already works.** LOW-only, short holds, drift-up assets, costs modeled, conviction tilt for size. This is the minimum viable product of the whole project and it is *already* holdout-validated. HIGH stays a monitor/context tool until a discriminator is proven.

## 4. The decision sequence from here (no drift)

1. **Ablation lands** → read which votes carry in-architecture weight per side. (Running.)
2. **In-space precision frontier** (local, CPU, real detector, Sobol ~20k presets, score = raw hit precision at matched firing rates) → answers "does OUR space contain a pocket?" once and for all, no surrogates.
3. Frontier says pocket → **one targeted GPU run** with a precision-tilted objective. Frontier says no pocket → Route C becomes legitimate (pick ONE family, spec, build, audit).
4. **In parallel and regardless:** Route D productization of LOW-pruned (cost model, sizing, conviction tilt) — because it's the only thing currently past break-even and it should not wait on research.
5. HIGH: no more search/widening spend until step 2/3 produce a reason.

## 5. Standing lessons (so we stop re-learning them)

- Local CPU falsifies hypotheses for free; the GPU validates surviving hypotheses expensively. Never invert the order again.
- Surrogate analyses generate hypotheses, never verdicts about the Pine space. Label them as such the moment they're produced.
- Precision is the scoreboard. Every proposed change must answer: *which side, how many precision points, at what firing rate, at what horizon?*
- One ruler: scorer v5.1. Era-marked. Cross-era comparisons by label only.
- The parity contract is the project. Anything that can't ship to a TradingView chart is out of scope by definition.

**Vote-layer audit: the user's "painted knobs" hypothesis CONFIRMED (2026-06-11,
temp/vote_contribution.py + temp/vote_bounds_audit.py, parity-anchored).**
Of 7 searched vote dimensions: trend = STRUCTURAL STAMP (comparison value spans
−12..+9 per-mille, box [0.01,0.5] is a sliver near zero — reachable pass-rate
only 36–46%, selectivity impossible at ANY allowed threshold); gjr = STRUCTURAL
STAMP (calc_gjr_asym's /0.1 normalization clips ~50% of mass to the ±1 rails —
bimodal, box slices between rails); volume box 50–100% dead space (ratio q99
1.78 vs box [1,3]); momentum has NO searchable threshold (fixed sign test, 64%
of bars exactly 0 from FX volume absence). Only mom_vel, vola, drift, har were
ever genuinely tunable — and the contribution census found drift+vola to be the
only real selectors among those. The optimizer demonstrably rides the max-pass
corners of stampable boxes to satisfy confirm_count. **Consequence: all prior
searches were valid tests of 4 features, not 8; whether properly-scaled trend /
unclipped gjr / thresholded momentum carry discrimination is UNKNOWN. The
freeze/kill decision is deferred until one run on a repaired space.** **[SOLID]**
