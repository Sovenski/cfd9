"""Leave-one-out feature ablation via shape_variants (no parity change).

The 7 vote features combine ADDITIVELY in the detector (detector.py max_votes
sum), so leaving one out and re-searching gives a clean, holdout-validated
marginal contribution per feature — the 'what does it want' probe.
"""
from __future__ import annotations

import pytest

from src.v17_ablation import (
    ABLATION_VOTE_FEATURES,
    ablation_deltas,
    build_ablation_variants,
)


def test_ablation_features_are_the_seven_vote_features():
    # Exactly the use_X votes summed into max_votes (detector.py:446-455).
    assert ABLATION_VOTE_FEATURES == (
        "trend", "volume", "momentum", "momentum_velocity",
        "volatility", "gjr_asym", "har_vol",
    )


def test_build_ablation_variants_has_all_on_plus_one_minus_per_feature():
    v = build_ablation_variants()
    assert set(v) == {"all_on"} | {f"minus_{f}" for f in ABLATION_VOTE_FEATURES}
    assert len(v) == 1 + len(ABLATION_VOTE_FEATURES)  # 8


def test_all_on_turns_every_vote_on_both_sides():
    allon = build_ablation_variants()["all_on"]
    for f in ABLATION_VOTE_FEATURES:
        for s in ("high", "low"):
            assert allon[f"use_{f}_{s}"] is True
    assert len(allon) == 2 * len(ABLATION_VOTE_FEATURES)


def test_minus_variant_drops_exactly_one_feature_both_sides():
    v = build_ablation_variants()
    allon, m = v["all_on"], v["minus_gjr_asym"]
    assert m["use_gjr_asym_high"] is False and m["use_gjr_asym_low"] is False
    # only that feature differs from all_on; everything else stays on
    diffs = {k for k in allon if allon[k] != m[k]}
    assert diffs == {"use_gjr_asym_high", "use_gjr_asym_low"}
    assert m["use_trend_high"] is True and m["use_har_vol_low"] is True


def test_ablation_deltas_signed_leave_one_out_importance():
    # delta(feature) = holdout(all_on) - holdout(minus_feature):
    # >0 the feature HELPS the reserved tail; <0 it HURTS (overfit noise).
    out = {"variants": {
        "all_on":      {"sides": {"high": {"holdout": {"score": 0.20}}}},
        "minus_trend": {"sides": {"high": {"holdout": {"score": 0.20}}}},  # redundant
        "minus_gjr_asym": {"sides": {"high": {"holdout": {"score": 0.05}}}},  # load-bearing
        "minus_har_vol":  {"sides": {"high": {"holdout": {"score": 0.26}}}},  # HURTS
    }}
    d = ablation_deltas(out, "high")
    assert d["gjr_asym"] == pytest.approx(0.15)
    assert d["trend"] == pytest.approx(0.0)
    assert d["har_vol"] == pytest.approx(-0.06)
    # sorted most-helpful first
    assert list(d)[0] == "gjr_asym"
    assert list(d)[-1] == "har_vol"


def test_ablation_deltas_requires_all_on_baseline():
    with pytest.raises(KeyError):
        ablation_deltas({"variants": {"minus_trend": {}}}, "high")


def test_ablation_deltas_skips_minus_variant_with_no_holdout_score():
    # A degenerate variant (holdout None / missing) must not crash the report;
    # it is simply omitted from the importance table.
    out = {"variants": {
        "all_on":         {"sides": {"low": {"holdout": {"score": 0.10}}}},
        "minus_trend":    {"sides": {"low": {"holdout": None}}},
        "minus_volume":   {"sides": {"low": {}}},
        "minus_har_vol":  {"sides": {"low": {"holdout": {"score": 0.08}}}},
    }}
    d = ablation_deltas(out, "low")
    assert set(d) == {"har_vol"}
    assert d["har_vol"] == pytest.approx(0.02)
