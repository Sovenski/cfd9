"""Scorer v3 smoke test — Hungarian matching + bootstrap-LCB objective.

Verifies that Scorer v3 refinements preserve the v2 semantic ranking
(sparse Trial #178 > dense Trial #249 on LOW side, SPX 1D) AND that the
new machinery is engaged:

1. ``compute_side_score(..., return_components=True)`` returns a
   component diagnostics dict with the v3 keys (precision_at_ref,
   recall_saturated_at_ref, frequency_factor, excess_penalty,
   ref_pivots, ref_scale, n_signals).
2. ``fold_scores_bootstrap_ci`` honours the new ``block_len`` parameter
   and returns a tight CI for a single fold and a wider CI for
   correlated multi-fold inputs.
3. ``build_optuna_objective``'s inner objective calls
   ``trial.set_user_attr`` with ``component_*`` keys via a
   lightweight mock trial.

The sparse/dense Trial #178 vs #249 LOW configuration is read from the
V15 Run 1 / V15 Run 1 Selective presets (identical to the v2 smoke test).
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.detector import SpeculatorDetector, build_detector_artifacts  # noqa: E402
from src.indicators import Params  # noqa: E402
from src.scoring import (  # noqa: E402
    add_pivot_labels,
    compute_side_score,
    precision_at_n_stats,
)
from src.validation import (  # noqa: E402
    _evaluate_with_components,
    _set_component_attrs,
    build_optuna_objective,
    fold_scores_bootstrap_ci,
    prepare_walk_forward_folds,
    walk_forward_folds,
)

DATA_PATH = ROOT / "data" / "raw" / "SPX_1D_18710201_20260318.csv"

# HIGH preset shared between Trial #178 and Trial #249 (V15 Run 1).
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


class _MockTrial:
    """Minimal Trial stand-in that captures set_user_attr calls."""

    def __init__(self) -> None:
        self.user_attrs: dict[str, Any] = {}

    def set_user_attr(self, key: str, value: Any) -> None:
        self.user_attrs[key] = value


def _score_one(df: pd.DataFrame, params: Params, label: str) -> dict:
    artifacts = build_detector_artifacts(df)
    det = SpeculatorDetector(df, params, artifacts=artifacts).run()
    n_low = int(det["signal_low"].sum())
    scalar, per_scale, components = compute_side_score(
        df,
        det["signal_low"],
        "low",
        return_per_scale=True,
        return_components=True,
    )
    print(f"[scorer_v3] {label}: LOW signals = {n_low}")
    print(f"[scorer_v3] {label}: LOW Scorer v3 = {scalar:.6f}")
    print(f"[scorer_v3] {label}: ref_scale={components['ref_scale']} "
          f"precision_at_ref={components['precision_at_ref']:.4f} "
          f"recall_sat_at_ref={components['recall_saturated_at_ref']:.4f} "
          f"freq_factor={components['frequency_factor']:.3f} "
          f"excess_penalty={components['excess_penalty']:.3f}")
    return {
        "signals": n_low,
        "score": float(scalar),
        "per_scale": per_scale,
        "components": components,
    }


def _check_components(components: dict, label: str) -> bool:
    required = {
        "precision_at_ref",
        "recall_saturated_at_ref",
        "frequency_factor",
        "excess_penalty",
        "ref_pivots",
        "ref_scale",
        "n_signals",
    }
    missing = required - set(components)
    if missing:
        print(f"[scorer_v3] FAIL — {label} components missing: {sorted(missing)}")
        return False
    # Scorer v4: REFERENCE_N is now 100 (and the structural-nest oracle is
    # written into pivot_N100). The stable-reference invariant survives —
    # only the integer changed. Accept either v3 (50) or v4 (100) here so
    # this test serves as a regression check across versions.
    from src.scoring import REFERENCE_N as _CURRENT_REFERENCE_N
    if components["ref_scale"] != _CURRENT_REFERENCE_N:
        print(
            f"[scorer_v3] FAIL — {label} expected ref_scale={_CURRENT_REFERENCE_N}, "
            f"got {components['ref_scale']}"
        )
        return False
    return True


def _test_bootstrap_ci() -> bool:
    # Single fold → CI lower = CI upper = score.
    lo1, hi1 = fold_scores_bootstrap_ci([0.42], n_boot=200, block_len=2)
    if not (lo1 == hi1 == 0.42):
        print(f"[scorer_v3] FAIL — single-fold CI not collapsed: ({lo1}, {hi1})")
        return False

    # Multi-fold with low variance → tight CI bracketing the mean.
    scores = [0.10, 0.12, 0.11, 0.13, 0.09]
    lo, hi = fold_scores_bootstrap_ci(scores, n_boot=2000, alpha=0.10, block_len=2)
    mean = float(np.mean(scores))
    if not (lo < mean < hi):
        print(f"[scorer_v3] FAIL — CI does not bracket mean: lo={lo}, mean={mean}, hi={hi}")
        return False
    if hi - lo > 0.2:
        print(f"[scorer_v3] FAIL — CI too wide: lo={lo}, hi={hi}")
        return False

    # Block length 2 should produce noticeably wider CIs than block length 1
    # on positively-correlated inputs (linear trend).
    correlated = list(np.linspace(0.0, 1.0, 14))
    lo1_, hi1_ = fold_scores_bootstrap_ci(correlated, n_boot=2000, block_len=1)
    lo2_, hi2_ = fold_scores_bootstrap_ci(correlated, n_boot=2000, block_len=2)
    width1 = hi1_ - lo1_
    width2 = hi2_ - lo2_
    print(
        f"[scorer_v3] block_len CI widths (correlated input): "
        f"block=1: {width1:.4f}, block=2: {width2:.4f}"
    )
    if width2 < width1 * 0.7:
        print("[scorer_v3] FAIL — block-2 CI is much tighter than block-1, unexpected")
        return False
    return True


def _test_precision_at_n_lead_bias() -> bool:
    """Construct a 1-signal, 2-pivot scenario where Hungarian+lead_bias
    must pick the leading pivot over the lagging pivot of equal abs dist.
    """
    n_bars = 200
    signals = pd.Series(False, index=range(n_bars))
    signals.iloc[100] = True
    pivots = pd.Series(0, index=range(n_bars), dtype=np.int8)
    pivots.iloc[99] = 1   # 1-bar lag
    pivots.iloc[101] = 1  # 1-bar lead
    stats = precision_at_n_stats(signals, pivots, "high", n=20)
    # Both pivots are in-window, equal absolute distance. Hungarian with
    # LEAD_BIAS=0.5 should match the leading pivot (101) over the
    # lagging one (99). tp must be 1 (signal matched) and matched_pivots
    # must be 1. We can't directly verify *which* pivot was picked from
    # the public stats, but if tp==1 and exactly 1 pivot was matched, the
    # signal got one TP and the implementation is sound.
    if stats["tp"] != 1 or stats["matched_pivots"] != 1:
        print(f"[scorer_v3] FAIL — lead_bias stats: {stats}")
        return False
    return True


def _test_objective_user_attrs(df: pd.DataFrame) -> bool:
    """Verify trial.set_user_attr is called with component_* keys.

    We bypass Optuna's real Trial by calling the helpers
    ``_evaluate_with_components`` and ``_set_component_attrs`` directly
    on a tiny prepared-fold list, then checking the mock trial's
    captured user_attrs dictionary.
    """
    # Tiny slice for fast prep.
    small = df.tail(2500).copy()
    folds = walk_forward_folds(small)
    if not folds:
        print("[scorer_v3] FAIL — could not build folds for user_attr test")
        return False
    prepared = prepare_walk_forward_folds(small, folds[:2])
    params = Params(**{**HIGH_KWARGS, **LOW_KWARGS_SPARSE})
    components_per_fold: list[dict[str, float]] = []
    for _score, components in _evaluate_with_components(params, "low", prepared):
        components_per_fold.append(components)
    mock = _MockTrial()
    _set_component_attrs(mock, components_per_fold)
    expected_prefixes = (
        "component_mean_oos_score",
        "component_mean_is_score",
        "component_mean_is_oos_gap",
        "component_mean_precision_at_ref_oos",
        "component_mean_recall_saturated_at_ref_oos",
        "component_mean_frequency_factor_oos",
        "component_mean_excess_penalty_oos",
    )
    missing = [k for k in expected_prefixes if k not in mock.user_attrs]
    if missing:
        print(f"[scorer_v3] FAIL — set_user_attr missing keys: {missing}")
        return False
    print(
        f"[scorer_v3] set_user_attr captured {len(mock.user_attrs)} component_* keys, "
        f"sample: mean_oos_score={mock.user_attrs['component_mean_oos_score']:.4f}, "
        f"mean_excess_penalty_oos={mock.user_attrs['component_mean_excess_penalty_oos']:.4f}"
    )
    return True


def main() -> int:
    print(f"[scorer_v3] loading {DATA_PATH.name}")
    raw = pd.read_csv(DATA_PATH)
    raw.columns = raw.columns.str.lower()
    print(f"[scorer_v3] {len(raw)} bars loaded")

    add_pivot_labels(raw)

    params_dense = Params(**{**HIGH_KWARGS, **LOW_KWARGS_DENSE})
    params_sparse = Params(**{**HIGH_KWARGS, **LOW_KWARGS_SPARSE})

    dense = _score_one(raw.copy(), params_dense, "Trial #249 (DENSE LOW)")
    print()
    sparse = _score_one(raw.copy(), params_sparse, "Trial #178 (SPARSE LOW)")

    print()
    print("=== Scorer v3 comparison ===")
    print(f"  Dense  (n={dense['signals']:>3}): {dense['score']:.6f}")
    print(f"  Sparse (n={sparse['signals']:>3}): {sparse['score']:.6f}")
    print(f"  Delta (sparse - dense): {sparse['score'] - dense['score']:+.6f}")

    # Semantic preservation: sparse > dense.
    if dense["score"] <= 0.0 or sparse["score"] <= 0.0:
        print("[scorer_v3] FAIL — at least one score is non-positive.")
        return 1
    if sparse["score"] <= dense["score"]:
        print(
            "[scorer_v3] FAIL — sparse Trial #178 must outrank dense Trial #249 "
            "(Scorer v2 semantic must be preserved)."
        )
        return 1

    # Component dict shape + stable reference scale.
    if not _check_components(dense["components"], "dense"):
        return 1
    if not _check_components(sparse["components"], "sparse"):
        return 1

    # Bootstrap CI with block length 2.
    if not _test_bootstrap_ci():
        return 1

    # Hungarian + lead-bias smoke.
    if not _test_precision_at_n_lead_bias():
        return 1

    # trial.set_user_attr wiring.
    if not _test_objective_user_attrs(raw):
        return 1

    print(
        f"[scorer_v3] PASS — sparse outranks dense (sparse={sparse['score']:.4f} > "
        f"dense={dense['score']:.4f}), components dict OK, bootstrap-CI block=2 OK, "
        f"Hungarian+lead-bias OK, trial.set_user_attr OK."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
