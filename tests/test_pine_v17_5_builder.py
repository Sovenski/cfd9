"""T13 — Pine v17.5 builder + parity-script tests (spec §4.1/§4.2, F5/F7/R4/R6).

The builder is a pure function ``build_pine_v17_5(v17_text, calibration)``
imported from ``temp/build_pine_v17_5.py`` via path injection (mirroring how
``temp/build_h100_notebook.py`` self-validates). The HARD claim under test:
the generated file's diff against ``pine/speculatores_v17_presets_gold.pine``
touches ONLY the indicator-title line (version "17.5") — every detection line
is byte-identical and in order — plus an APPENDED V17.5 block (calibration
constants, card display, tooltips, parity export plots).
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.scoring_v5 import SPAN_GRID
from src.v17_card.calibration import IN_SAMPLE_DISCLAIMER
from src.v17_card.conditioning import survival_lookup
from src.v17_card.stop_rule import stop_series

_REPO = Path(__file__).resolve().parents[1]
_V17_PINE = _REPO / "pine" / "speculatores_v17_presets_gold.pine"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, _REPO / "temp" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def builder():
    return _load("build_pine_v17_5")


@pytest.fixture(scope="module")
def parity():
    return _load("parity_v17_5")


# ---------------------------------------------------------------------------
# fixture calibration — hand-written §5 payload (both sides)
# ---------------------------------------------------------------------------


def _toy_side(scale: float) -> dict:
    s_r = [round(1.0 - 0.09 * j * scale, 6) for j in range(10)]
    return {
        "calibration": {
            "grid": list(SPAN_GRID),
            "S_R": s_r,
            "S_lo": [round(v - 0.05, 6) for v in s_r],
            "S_hi": [round(min(v + 0.05, 1.0), 6) for v in s_r],
            "band_method": "stream", "n_boot": 200, "seed": 0,
            "c_side": {"c": 1.234567 * scale, "r_squared": 0.42, "n_fit": 12,
                       "use_fallback": False, "fallback_median": 0.0517},
            "expected_hold_at_fire": 117.25, "clock_bars": 117,
            "conviction_breakpoints": [round(0.001 * j * scale, 8)
                                       for j in range(11)],
            "stop_rule": {"match_window": 1},
            "n_signals": 14, "n_streams": 2, "degenerate": False,
            "fit_diagnostics": {"in_sample_disclaimer": IN_SAMPLE_DISCLAIMER},
        },
        "calibration_block_hash": "deadbeef" * 8,
    }


@pytest.fixture(scope="module")
def calibration() -> dict:
    return {"high": _toy_side(1.0), "low": _toy_side(0.9)}


@pytest.fixture(scope="module")
def built(builder, calibration) -> str:
    return builder.build_pine_v17_5(
        _V17_PINE.read_text(encoding="utf-8"), calibration)


# ---------------------------------------------------------------------------
# THE diff-surface claim (detection byte-identity, programmatic)
# ---------------------------------------------------------------------------


def test_detection_lines_byte_identical_and_in_order(built):
    orig = _V17_PINE.read_text(encoding="utf-8").splitlines()
    gen = built.splitlines()
    assert len(gen) > len(orig), "v17.5 must APPEND content"
    head = gen[:len(orig)]
    diffs = [i for i, (a, b) in enumerate(zip(head, orig)) if a != b]
    # the ONLY modified original line is the indicator title
    assert len(diffs) == 1, f"unexpected detection-line edits at {diffs}"
    assert "indicator(" in orig[diffs[0]]
    assert "V17.5" in head[diffs[0]]


def test_appended_block_only_after_original_tail(built):
    orig = _V17_PINE.read_text(encoding="utf-8").splitlines()
    appended = built.splitlines()[len(orig):]
    assert appended, "appended V17.5 block missing"
    assert appended[0].startswith("//"), "appended block must open with a comment banner"
    text = "\n".join(appended)
    assert "// === CALIBRATION BLOCK (generated) ===" in text


def test_version_string_in_title(built):
    title_line = next(l for l in built.splitlines() if l.startswith("indicator("))
    assert "17.5" in title_line


def test_determinism_byte_identical_across_builds(builder, calibration):
    t = _V17_PINE.read_text(encoding="utf-8")
    assert builder.build_pine_v17_5(t, calibration) == \
        builder.build_pine_v17_5(t, calibration)


# ---------------------------------------------------------------------------
# calibration block — parse-back exactness
# ---------------------------------------------------------------------------


def test_calibration_block_values_round_trip(builder, built, calibration):
    parsed = builder.parse_calibration_block(built)
    for side in ("high", "low"):
        cal = calibration[side]["calibration"]
        assert parsed[side]["S_R"] == cal["S_R"]
        assert parsed[side]["S_lo"] == cal["S_lo"]
        assert parsed[side]["S_hi"] == cal["S_hi"]
        assert parsed[side]["conviction_breakpoints"] == \
            cal["conviction_breakpoints"]
        assert parsed[side]["c"] == cal["c_side"]["c"]
        assert parsed[side]["use_fallback"] == cal["c_side"]["use_fallback"]
        assert parsed[side]["fallback_median"] == \
            cal["c_side"]["fallback_median"]
        assert parsed[side]["clock_bars"] == cal["clock_bars"]
    # grid generated from the ONE shared table (R4 — engine/Pine cannot diverge)
    assert parsed["grid"] == list(SPAN_GRID)
    # block hash traced
    assert calibration["high"]["calibration_block_hash"] in built
    assert calibration["low"]["calibration_block_hash"] in built


# ---------------------------------------------------------------------------
# tooltips — §1.4 plain language, R3/F5/F7 anchors, H0 explanation
# ---------------------------------------------------------------------------


def test_tooltip_tier_definitions_verbatim(built):
    # spec §1.4 — display vocabulary, plain-language
    assert ("a major turning point: the highest/lowest price of roughly "
            "the surrounding year or more") in built
    assert ("a meaningful swing turn: extremum of the surrounding quarter, "
            "not strong enough to be T1") in built
    assert "a minor swing turn (weeks-scale)" in built


def test_tooltip_conditional_on_match_label(built):
    assert "expected move if this is a real turn" in built  # R3


def test_tooltip_h0_explanation(built):
    assert "candidate pivot low" in built
    assert "definitionally refutes the matched-pivot hypothesis" in built


def test_tooltip_live_vs_frozen_scope_note(built):
    assert "most recent (active) signal per side" in built  # F5
    assert "FROZEN card" in built


def test_tooltip_in_sample_disclaimer_exact_sentence(built):
    assert IN_SAMPLE_DISCLAIMER in built  # F7, verbatim


def test_band_method_stated(built):
    assert "cluster-bootstrap band" in built  # F6/R6
    assert "S_lo and S_hi separately" in built  # pragmatic-envelope note


# ---------------------------------------------------------------------------
# card display block — banded probabilities, live var state, R4 helper
# ---------------------------------------------------------------------------


def test_band_rendering_never_point_values(built):
    # the label/table format carries a-b% band placeholders (R6)
    for side in ("high", "low"):
        for p in ("t2", "t1"):
            assert f"card_p_{p}_lo_{side}" in built
            assert f"card_p_{p}_hi_{side}" in built
    assert "P(T2+)= " in built and "P(T1)= " in built


def test_live_var_state_block_per_side(built):
    for side in ("high", "low"):
        for var in ("card_fire", "card_i", "card_stop", "card_fire_stop",
                    "card_L", "card_intact", "card_final"):
            assert f"var" in built and f"{var}_{side}" in built, \
                f"missing live state var {var}_{side}"
    # row-6 reset path: fire handler doubles as the same-side re-fire reset
    assert "row-6" in built


def test_off_grid_floor_lookup_helper_emitted(built):
    assert "card_lookup(" in built
    # R4 rule documented at the helper
    assert "largest grid value <=" in built
    # F1 conditioning + §3.4 clock helpers present
    assert "card_cond(" in built
    assert "card_ehold(" in built


def test_post_invalidation_na_plot_idiom(built):
    # §3.5 DECIDED plotting semantic — exact Pine idiom, parity-pinned
    assert "card_intact_high ? card_stop_high : na" in built
    assert "card_intact_low ? card_stop_low : na" in built


def test_parity_export_plots_for_card_numerics(built):
    names = [f"card_{q}_{side}"
             for side in ("high", "low")
             for q in ("p_t2_lo", "p_t2_hi", "p_t1_lo", "p_t1_hi",
                       "move", "ehold", "stop")]
    for name in names:
        assert f'"{name}"' in built, f"missing parity export plot {name}"
        assert f'plot(' in built


def test_degenerate_side_rejected(builder, calibration):
    bad = {"high": calibration["high"],
           "low": {"calibration": {"degenerate": True, "S_R": None},
                   "calibration_block_hash": "00" * 32}}
    with pytest.raises(ValueError, match="degenerate"):
        builder.build_pine_v17_5(
            _V17_PINE.read_text(encoding="utf-8"), bad)


# ---------------------------------------------------------------------------
# sample calibration — the file must build TODAY
# ---------------------------------------------------------------------------


def test_sample_calibration_builds_pine(builder):
    sample = builder.make_sample_calibration()
    for side in ("high", "low"):
        assert sample[side]["calibration"]["degenerate"] is False
        assert sample[side]["calibration"]["n_signals"] > 0
    out = builder.build_pine_v17_5(
        _V17_PINE.read_text(encoding="utf-8"), sample)
    assert "// === CALIBRATION BLOCK (generated) ===" in out
    json.dumps(sample)  # JSON-serializable (round-trips to disk)


def test_generated_artifact_exists_and_matches_sample(builder):
    """The repo artifact pine/speculatores_v17_5_signalcard.pine is built."""
    out_path = _REPO / "pine" / "speculatores_v17_5_signalcard.pine"
    sample_path = _REPO / "temp" / "sample_calibration_v17_5.json"
    assert out_path.exists(), "run temp/build_pine_v17_5.py to generate it"
    assert sample_path.exists()
    sample = json.loads(sample_path.read_text(encoding="utf-8"))
    rebuilt = builder.build_pine_v17_5(
        _V17_PINE.read_text(encoding="utf-8"), sample)
    assert out_path.read_text(encoding="utf-8") == rebuilt


# ---------------------------------------------------------------------------
# temp/parity_v17_5.py — audit logic on synthetic TV-style data (§4.2)
# ---------------------------------------------------------------------------


def _zigzag_df(n: int = 60) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    base = 100.0 + np.cumsum(rng.normal(0, 0.5, n))
    low = base - 0.5
    high = base + 0.5
    return pd.DataFrame({"open": base, "high": high, "low": low,
                         "close": base, "volume": np.ones(n)})


def test_audit_signals_exact_and_count(parity):
    a = np.array([False, True, False, True])
    ok, _ = parity.audit_signals(a, a.copy(), "low")
    assert ok
    b = a.copy()
    b[0] = True  # one flip — count diff AND exactness must both fail
    ok, msg = parity.audit_signals(a, b, "low")
    assert not ok and "low" in msg


def test_audit_numeric_atol_1e_6(parity):
    py = np.array([0.5, np.nan, 0.25])
    ok, _ = parity.audit_numeric(py + 5e-7, py, "card_move_low")
    assert ok
    ok, _ = parity.audit_numeric(py + 5e-6, py, "card_move_low")
    assert not ok
    # NaN mask must match exactly (engine-NaN == Pine-na)
    bad = py.copy()
    bad[1] = 0.5
    ok, _ = parity.audit_numeric(bad, py, "card_move_low")
    assert not ok


def test_audit_stop_exact_with_na_semantics(parity):
    df = _zigzag_df()
    fires = [10, 30]
    py = stop_series(df["low"].to_numpy(), df["close"].to_numpy(), fires)
    ok, _ = parity.audit_stop(py.copy(), py, "low")
    assert ok
    tv = py.copy()
    idx = int(np.flatnonzero(np.isfinite(tv))[0])
    tv[idx] += 1e-9  # stop is EXACT — any deviation fails
    ok, _ = parity.audit_stop(tv, py, "low")
    assert not ok


def test_engine_card_series_stop_matches_stop_series(parity, builder):
    """Row-2/3 ordering + row-6 re-fire: the engine card stop column IS
    stop_series (the §4.2 reference), bar for bar, NaN==NaN."""
    df = _zigzag_df(80)
    sig = np.zeros(80, dtype=bool)
    sig[[20, 24, 50]] = True  # 20->24 is a same-side re-fire overlap (row 6)
    sample = builder.make_sample_calibration()
    cols = parity.engine_card_series(df, sig, "low",
                                     sample["low"]["calibration"])
    ref = stop_series(df["low"].to_numpy(), df["close"].to_numpy(),
                      [20, 24, 50])
    np.testing.assert_array_equal(cols["card_stop_low"].to_numpy(), ref)
    # card numerics share the stop's NaN mask (na unless an intact card)
    for c in cols.columns:
        assert np.array_equal(np.isnan(cols[c].to_numpy()), np.isnan(ref)), c


def test_survival_lookup_audit_off_grid_floor_rule(parity, builder):
    """R4: engine lookup == the Pine helper transcription, ON and OFF grid."""
    sample = builder.make_sample_calibration()
    ok, msg = parity.audit_survival_lookup(sample)
    assert ok, msg
    table = np.asarray(sample["low"]["calibration"]["S_R"], dtype=float)
    for x in (5.0, 19.999, 20.0, 25.0, 47.0, 70.5, 199.0, 200.0, 450.0,
              500.0, 700.0):
        assert parity.pine_lookup(table, x) == survival_lookup(table, x)
