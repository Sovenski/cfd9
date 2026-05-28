# Run 5 Pre-Registration (filed before the run)

**Hypotheses (HIGH side, relaxed floors):**
- `dur_extreme_pct` floor 0.50→0.30 → expected to make the duration gate
  satisfiable for dim-agreement tops; HIGH fire-rate should rise from ~noise.
- `pct_extreme` floor 0.70→0.55 → expected to densify top agreement; HIGH
  in-sample precision-at-REFERENCE_N should rise vs Run 4.

**Primary metric:** pooled OOS bootstrap-LCB per side (Optuna objective).
**Holdout reporting rule:** report EXACT holdout HIGH/LOW hit counts (TP, FP,
n_signals, total_pivots) BEFORE vs AFTER, never only "improvement". The famous
post-2000 tops are public and were visually inspected in Runs 1–4 — treat any
holdout gain with that contamination in mind.

**Decision rule:** ship a new HIGH preset only if pooled OOS-LCB > Run 4 HIGH
(0.0656) AND holdout HIGH TP > 0 with FP not inflated.
