"""Scorer v2 smoke test — Trial #178 (sparse LOW) vs Trial #249 (dense LOW).

Goal: confirm Scorer v2 ranks the V15 "Run 1 Selective" LOW (trial #178,
64 lifetime signals) closer to the V15 "Run 1" LOW (trial #249, 357
signals) than the old scorer did — and crucially that the two scores are
*meaningfully different* (so the optimizer has a learning signal to
prefer one over the other).

Inputs are extracted from `temp/parity_audit_v15_run1.py` (trial #249)
and `temp/add_pine_v15_run1_selective.py` (trial #178). The HIGH-side
preset is shared between both runs and is held constant.

This test is intentionally tolerant: it only asserts that
  1. Both scores are positive.
  2. The two scores are not identical (i.e. Scorer v2 still
     discriminates between the two regimes).
  3. The excess_penalty machinery has not collapsed: Scorer v2 should
     not drive the dense trial #249 score above the sparse trial #178
     score by more than 2x (otherwise the two-sided penalty isn't
     working).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.detector import SpeculatorDetector, build_detector_artifacts  # noqa: E402
from src.indicators import Params  # noqa: E402
from src.scoring import add_pivot_labels, compute_side_score  # noqa: E402

DATA_PATH = ROOT / "data" / "raw" / "SPX_1D_18710201_20260318.csv"

# HIGH preset — identical across Run 1 and Run 1 Selective (trial #64).
HIGH_KWARGS = dict(
    S_detect_high=5, scale_start_high=22, scale_end_high=182, scale_step_high=8,
    min_duration_high=1, cooldown_bars_high=6, price_gate_lb_high=40,
    vola_range_len_high=99, er_period_high=19, confirm_count_high=2,
    pivot_drift_lookback_high=16, pivot_drift_confirm_bias_high=2,
    pct_extreme_high=0.7839, min_agreement_high=0.8208,
    dur_extreme_pct_high=0.8019, vol_surge_thresh_high=2.3810,
    scale_div_thresh_high=0.2225, slope_thresh_high=0.2340,
    vola_high_pct_high=0.7133, pivot_drift_thresh_high=0.0352,
    pivot_drift_gate_mult_high=9.0627, momentum_velocity_thresh_high=0.0172,
    gjr_vote_thresh_high=0.2770, har_vote_thresh_high=0.3205,
    er_directional_high=True, use_trend_high=True, use_volume_high=False,
    use_momentum_high=False, use_momentum_velocity_high=False,
    use_volatility_high=False, use_er_gate_high=False,
    use_gjr_asym_high=False, use_har_vol_high=False,
    vola_method_high="Intraday", momentum_velocity_mode_high="Reversal",
    use_edge_voting_high=True, edge_window_high=21,
)

# LOW trial #249 — Run 1 score winner (dense, 357 signals).
LOW_KWARGS_DENSE = dict(
    S_detect_low=8, scale_start_low=4, scale_end_low=209, scale_step_low=20,
    min_duration_low=2, cooldown_bars_low=3, price_gate_lb_low=66,
    vola_range_len_low=38, er_period_low=53, confirm_count_low=1,
    pivot_drift_lookback_low=12, pivot_drift_confirm_bias_low=0,
    pct_extreme_low=0.7041, min_agreement_low=0.1284,
    dur_extreme_pct_low=0.5221, vol_surge_thresh_low=2.4163,
    scale_div_thresh_low=0.3256, slope_thresh_low=0.3399,
    vola_high_pct_low=0.7211, pivot_drift_thresh_low=0.0131,
    pivot_drift_gate_mult_low=6.5114, momentum_velocity_thresh_low=0.0416,
    gjr_vote_thresh_low=0.3949, har_vote_thresh_low=0.2100,
    er_directional_low=True, use_trend_low=False, use_volume_low=False,
    use_momentum_low=False, use_momentum_velocity_low=False,
    use_volatility_low=False, use_er_gate_low=False,
    use_gjr_asym_low=True, use_har_vol_low=False,
    vola_method_low="StdDev", momentum_velocity_mode_low="Reversal",
    use_edge_voting_low=False, edge_window_low=29,
)

# LOW trial #178 — Run 1 Selective sparse alt (64 signals).
LOW_KWARGS_SPARSE = dict(
    S_detect_low=16, scale_start_low=11, scale_end_low=206, scale_step_low=17,
    min_duration_low=3, cooldown_bars_low=11, price_gate_lb_low=41,
    vola_range_len_low=138, er_period_low=32, confirm_count_low=1,
    pivot_drift_lookback_low=11, pivot_drift_confirm_bias_low=2,
    pct_extreme_low=0.7536, min_agreement_low=0.7931,
    dur_extreme_pct_low=0.5199, vol_surge_thresh_low=2.6458,
    scale_div_thresh_low=0.5935, slope_thresh_low=0.3980,
    vola_high_pct_low=0.5747, pivot_drift_thresh_low=0.0259,
    pivot_drift_gate_mult_low=8.8631, momentum_velocity_thresh_low=0.0466,
    gjr_vote_thresh_low=0.3772, har_vote_thresh_low=0.2266,
    er_directional_low=False, use_trend_low=False, use_volume_low=False,
    use_momentum_low=False, use_momentum_velocity_low=False,
    use_volatility_low=False, use_er_gate_low=True,
    use_gjr_asym_low=True, use_har_vol_low=False,
    vola_method_low="StdDev", momentum_velocity_mode_low="Reversal",
    use_edge_voting_low=True, edge_window_low=21,
)


def _score_one(df: pd.DataFrame, params: Params, label: str) -> dict:
    artifacts = build_detector_artifacts(df)
    det = SpeculatorDetector(df, params, artifacts=artifacts).run()
    n_low = int(det["signal_low"].sum())

    # Full LOW score (scalar + per-scale dict via item 15).
    result = compute_side_score(df, det["signal_low"], "low", return_per_scale=True)
    assert isinstance(result, tuple), "return_per_scale=True must yield a tuple"
    scalar, per_scale = result
    print(f"[scorer_v2] {label}: LOW signals = {n_low}")
    print(f"[scorer_v2] {label}: LOW Scorer v2 = {scalar:.6f}")
    print(f"[scorer_v2] {label}: per-scale contributions =")
    for N, v in sorted(per_scale.items()):
        print(f"               N={N:>3}: {v:.4f}")
    return {"signals": n_low, "score": float(scalar), "per_scale": per_scale}


def main() -> int:
    print(f"[scorer_v2] loading {DATA_PATH.name}")
    raw = pd.read_csv(DATA_PATH)
    raw.columns = raw.columns.str.lower()
    print(f"[scorer_v2] {len(raw)} bars loaded")

    # Precompute pivot labels once, reuse across both runs.
    add_pivot_labels(raw)

    params_dense = Params(**{**HIGH_KWARGS, **LOW_KWARGS_DENSE})
    params_sparse = Params(**{**HIGH_KWARGS, **LOW_KWARGS_SPARSE})

    dense = _score_one(raw.copy(), params_dense, "Trial #249 (DENSE LOW)")
    print()
    sparse = _score_one(raw.copy(), params_sparse, "Trial #178 (SPARSE LOW)")

    print()
    print("=== Scorer v2 comparison ===")
    print(f"  Dense  (n={dense['signals']:>3}): {dense['score']:.6f}")
    print(f"  Sparse (n={sparse['signals']:>3}): {sparse['score']:.6f}")
    delta = dense["score"] - sparse["score"]
    print(f"  Delta (dense - sparse): {delta:+.6f}")

    # Sanity assertions.
    if dense["score"] <= 0.0 or sparse["score"] <= 0.0:
        print("[scorer_v2] FAIL — at least one score is non-positive.")
        return 1
    if abs(delta) < 1e-9:
        print("[scorer_v2] FAIL — scores are identical; Scorer v2 lost discrimination.")
        return 1

    # Two-sided excess_penalty sanity: the dense trial should not dominate
    # the sparse trial by more than 2x. The whole point of the two-sided
    # harmonic-mean penalty is that gross over-firing is punished.
    ratio = dense["score"] / sparse["score"]
    if ratio > 2.0:
        print(
            f"[scorer_v2] FAIL — dense/sparse score ratio {ratio:.2f}x suggests "
            "the two-sided excess_penalty failed to engage."
        )
        return 1
    print(
        f"[scorer_v2] PASS — both scores positive, distinct, and within 2x "
        f"(ratio={ratio:.2f}x; sparse-side penalty is in effect)."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
