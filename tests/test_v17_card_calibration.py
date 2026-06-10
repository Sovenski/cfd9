"""Calibration orchestration — TDD spec (impl plan T11; spec §5, §9 trace).

Pins the run-output contract of ``src/v17_card/calibration.py``:

* ``calibrate_run`` produces the full JSON-serializable payload with the
  required keys: ``scorer: "v5"``, ``calibration_block_hash`` (sha256 of the
  canonical calibration JSON), ``calibration`` (S_R/S_lo/S_hi/grid, c_side,
  conviction breakpoints, stop rule, fit diagnostics incl. the F6 band
  method, the F10 grid-floor bias note, the R3 conditional-on-match flag +
  downward-bias note, and the F7 in-sample disclaimer), ``signal_cards``
  (§3.7 rows) and ``r_multiple_backtest`` (costs ignored, spec §8).
* Round-trip: ``json.dumps`` -> ``json.loads`` -> equality (no numpy
  leakage; non-finite floats serialized as null).
* Determinism under a fixed seed (F6 bands are bootstrap-based).
* Degenerate zero-signal pools do not crash and are flagged.
"""
from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from src.scoring_v5 import SPAN_GRID
from src.v17_card.calibration import calibrate_run

#: F7 sentence — must appear VERBATIM in engine report, Pine tooltip, Cell 6.
IN_SAMPLE_DISCLAIMER = (
    "probabilities are calibrated on this instrument's history; card values "
    "shown on historical bars are in-sample; only forward bars (after the "
    "calibration run date) are out-of-sample"
)


def _v_stream(n: int = 400, pivot: int = 150, seed: int = 0,
              fires: tuple[int, ...] = (151, 300)) -> tuple[pd.DataFrame, np.ndarray]:
    """V-bottom stream (grading-fixture style) with planted LOW signals."""
    rng = np.random.default_rng(seed)
    b = np.arange(n, dtype=float)
    low = np.where(b <= pivot, 50.0 - 0.2 * b,
                   (50.0 - 0.2 * pivot) + 0.1 * (b - pivot))
    low = low + rng.uniform(0.0, 0.01, size=n)  # de-tie plateaus
    low[20] = 5.0
    df = pd.DataFrame({
        "open": low + 0.2, "high": low + 0.5, "low": low,
        "close": low + 0.3, "volume": np.ones(n),
    })
    sig = np.zeros(n, dtype=bool)
    for f in fires:
        sig[f] = True
    return df, sig


@pytest.fixture(scope="module")
def payload():
    df_a, sig_a = _v_stream(seed=1)
    df_b, sig_b = _v_stream(seed=2, fires=(149, 152, 280))
    return calibrate_run(
        {"A_1D": df_a, "B_1D": df_b},
        {"A_1D": sig_a, "B_1D": sig_b},
        side="low", seed=0, n_boot=50,
    )


# ---------------------------------------------------------------------------
# schema — the §5 / §9 run-output contract
# ---------------------------------------------------------------------------


def test_payload_required_keys_and_scorer_v5(payload):
    assert payload["scorer"] == "v5"
    assert payload["side"] == "low"
    assert set(payload) >= {"scorer", "side", "calibration",
                            "calibration_block_hash", "signal_cards",
                            "r_multiple_backtest"}


def test_calibration_block_schema(payload):
    cal = payload["calibration"]
    assert cal["grid"] == SPAN_GRID
    for key in ("S_R", "S_lo", "S_hi"):
        tbl = cal[key]
        assert len(tbl) == len(SPAN_GRID)
        assert all(0.0 <= v <= 1.0 for v in tbl)
        assert all(a >= b - 1e-12 for a, b in zip(tbl, tbl[1:])), \
            f"{key} not non-increasing"
    # band contains the point table (F6)
    for lo, pt, hi in zip(cal["S_lo"], cal["S_R"], cal["S_hi"]):
        assert lo - 1e-12 <= pt <= hi + 1e-12
    assert set(cal["c_side"]) >= {"c", "r_squared", "n_fit", "use_fallback",
                                  "fallback_median"}
    assert len(cal["conviction_breakpoints"]) == 11      # deciles 0..100
    bps = cal["conviction_breakpoints"]
    assert all(a <= b + 1e-12 for a, b in zip(bps, bps[1:]))
    assert cal["clock_bars"] >= 1
    assert cal["stop_rule"]["match_window"] == 1
    assert cal["stop_rule"]["t_plus_1_breach"] == "close_vs_fire_stop"
    assert cal["stop_rule"]["final_from"] == "t_plus_2_intrabar"
    assert cal["degenerate"] is False


def test_fit_diagnostics_honesty_block(payload):
    diag = payload["calibration"]["fit_diagnostics"]
    assert set(diag) >= {"r2", "censored_excluded_n", "band_method",
                         "grid_floor_bias_note", "c_side_bias_note",
                         "in_sample_disclaimer", "conditional_on_match",
                         "expected_move_units"}
    assert diag["conditional_on_match"] is True          # R3
    assert diag["band_method"] in ("stream", "block")    # F6
    assert diag["in_sample_disclaimer"] == IN_SAMPLE_DISCLAIMER  # F7
    assert "grid-floor" in diag["grid_floor_bias_note"]  # F10
    assert "downward" in diag["c_side_bias_note"]        # R3 length bias
    assert isinstance(diag["censored_excluded_n"], int)


def test_signal_cards_rows(payload):
    cards = payload["signal_cards"]
    assert len(cards) == 5                                # 2 + 3 signals
    required = {"stream", "fire_bar", "side", "matched", "pivot_bar",
                "realized_span", "span_censored", "tier", "realized_move",
                "entry", "stop", "risk", "exit_bar", "exit_price",
                "r_multiple", "mae_r", "mfe_r", "outcome", "conviction"}
    for row in cards:
        assert set(row) >= required
        assert row["side"] == "low"
        assert row["tier"] in {"T1", "T2", "T3", "miss"}
        assert row["conviction"] is None or 0.0 <= row["conviction"] <= 100.0
    # at least one direct hit exists in the fixture (fire 151 vs pivot ~150)
    assert any(row["matched"] for row in cards)


def test_r_multiple_backtest_summary(payload):
    bt = payload["r_multiple_backtest"]
    assert set(bt) >= {"n_signals", "n_trades", "win_rate", "expectancy",
                       "total_r", "equity_delta", "capture_ratio",
                       "costs_ignored"}
    assert bt["costs_ignored"] is True                    # spec §8
    assert bt["n_signals"] == 5.0


# ---------------------------------------------------------------------------
# JSON round-trip + hash + determinism
# ---------------------------------------------------------------------------


def test_json_round_trip_no_numpy_leakage(payload):
    dumped = json.dumps(payload, sort_keys=True, allow_nan=False)
    assert json.loads(dumped) == payload


def test_calibration_block_hash_is_sha256_of_canonical_json(payload):
    blob = json.dumps(payload["calibration"], sort_keys=True,
                      allow_nan=False).encode("utf-8")
    assert payload["calibration_block_hash"] == hashlib.sha256(blob).hexdigest()
    assert len(payload["calibration_block_hash"]) == 64


def test_deterministic_under_fixed_seed(payload):
    df_a, sig_a = _v_stream(seed=1)
    df_b, sig_b = _v_stream(seed=2, fires=(149, 152, 280))
    again = calibrate_run({"A_1D": df_a, "B_1D": df_b},
                          {"A_1D": sig_a, "B_1D": sig_b},
                          side="low", seed=0, n_boot=50)
    assert again == payload


# ---------------------------------------------------------------------------
# degenerate pools
# ---------------------------------------------------------------------------


def test_zero_signal_pool_is_degenerate_not_a_crash():
    df, _ = _v_stream(seed=3)
    out = calibrate_run({"A_1D": df}, {"A_1D": np.zeros(len(df), dtype=bool)},
                        side="low", seed=0, n_boot=50)
    assert out["scorer"] == "v5"
    assert out["calibration"]["degenerate"] is True
    assert out["signal_cards"] == []
    assert out["r_multiple_backtest"]["n_signals"] == 0.0
    json.loads(json.dumps(out, allow_nan=False))          # still serializable


def test_side_validated():
    df, sig = _v_stream()
    with pytest.raises(ValueError, match="side"):
        calibrate_run({"A_1D": df}, {"A_1D": sig}, side="up")


def test_stream_key_mismatch_rejected():
    df, sig = _v_stream()
    with pytest.raises(ValueError, match="stream"):
        calibrate_run({"A_1D": df}, {"B_1D": sig}, side="low")
