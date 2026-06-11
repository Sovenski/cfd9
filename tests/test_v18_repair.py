"""v18 repair tests (plan/v18-repair-spec.md) — Stages A and B.

A2 (REVISED by B1, 2026-06-11) — ``pir_of`` warm-up: Pine's ``pir_of`` uses
``ta.lowest``/``ta.highest`` which return na until ``lb`` bars of CHART
HISTORY exist but SKIP na values inside the window; the ``hi != lo ?``
ternary maps na to **0.5**. So: bars ``t < lb-1`` -> 0.5; afterwards the
min/max scan ignores a na head (e.g. ATR warm-up). Proven bar-exact against
e830d's ``dbg_high_vote_volatility`` (25,250 bars, 0 flips; Stage A's
min_periods=1 partial-window guess flipped 65 vola votes). Post-warm-up must
stay bit-identical to the historical full-window result.

B1 (P2.1-bis) — agreement-matrix warm-up parity: Pine's ``pir_for_scale``
substitutes ratio = 1.0 on every bar where the cumsum SMA is na (the ternary
``sma_b > 0 ? c_b / sma_b : 1.0`` takes the FALSE branch on na), including
virtual pre-history bars, and always scans exactly ``lb`` bars. So the PIR
matrix must be NaN-free with 1.0-filled warm-up ratios (proven by the e830d
bars 133/227/234/238/244 HIGH flips: Pine dbg_agreement_high_high 0.856 vs
the old Python 0.344 at bar 133 — fully-warm-up scales vote pir = 0.5).

B2 (P2.2) — gjr unclip; B3 (P2.3) — momentum_diverge_thresh; B4 (P2.4) —
count_drift_vote. New knobs at neutral (0.0 / False) must be bit-exact legacy.
"""
from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.indicators import (
    Params,
    calc_agreement_fast,
    calc_gjr_asym,
    pir_of,
    precompute_matrices,
)


def test_pir_of_history_warmup_is_half():
    rng = np.random.default_rng(42)
    val = pd.Series(rng.normal(100.0, 5.0, 300))
    lb = 20

    out = pir_of(val, lb)

    # 1. No NaN anywhere — Pine's na ternary yields 0.5, never na.
    assert not out.isna().any(), "pir_of must be defined from bar 0"

    # 2. Every value in [0, 1].
    assert ((out >= 0.0) & (out <= 1.0)).all()

    # 3. Bars t < lb-1: ta.lowest/highest are na (insufficient chart
    #    history) -> Pine's ternary takes the 0.5 branch.
    assert (out.iloc[: lb - 1] == 0.5).all()

    # 4. Post-warm-up (bar >= lb-1): EXACT equality with the historical
    #    full-window computation (min_periods == window).
    lo = val.rolling(lb).min()
    hi = val.rolling(lb).max()
    span = (hi - lo).clip(lower=1e-10)
    full = ((val - lo) / span).where(hi != lo, 0.5)
    np.testing.assert_array_equal(
        out.iloc[lb - 1:].to_numpy(), full.iloc[lb - 1:].to_numpy(),
        err_msg="pir_of post-warm-up region must be bit-identical")


def test_pir_of_skips_na_head_inside_window():
    # ta.lowest/ta.highest SKIP na values: with a 5-bar na head and lb=10,
    # bar 9 scans only bars 5..9 (the e830d ATR-warm-up mechanism: the
    # vola vote goes live at bar lb-1, NOT at na-head + lb).
    val = pd.Series([np.nan] * 5 + list(np.arange(1.0, 26.0)))
    out = pir_of(val, 10)
    assert (out.iloc[:9] == 0.5).all()   # insufficient history -> 0.5
    assert out.iloc[9] == 1.0            # rising series pins to the top
    assert (out.iloc[9:] == 1.0).all()


def test_pir_of_all_na_window_is_half():
    val = pd.Series([np.nan] * 30)
    out = pir_of(val, 10)
    assert (out == 0.5).all()            # na lo/hi -> Pine 0.5 branch


# ---------------------------------------------------------------------------
# B1 (P2.1-bis) — agreement-matrix warm-up parity vs a Pine brute-force oracle
# ---------------------------------------------------------------------------


def _pine_pir_for_scale(close: np.ndarray, s: int) -> np.ndarray:
    """Literal transcription of the Pine parity shim (pine/..._signalcard.pine
    L112-138): ``sma_at`` via ta.cum difference (na before bar ``s + back``),
    ratio ``sma_b > 0 ? c_b / sma_b : 1.0`` (na condition -> FALSE -> 1.0),
    scan over exactly ``lb`` bars incl. virtual pre-history (ratio 1.0)."""
    n = len(close)
    csum = np.cumsum(close)
    lb = max(s, 20)

    def ratio_at(t: int, back: int) -> float:
        a, b = t - back, t - back - s
        if a < 0 or b < 0:                      # na csum -> na sma -> 1.0
            return 1.0
        sma_b = (csum[a] - csum[b]) / s
        return close[a] / sma_b if sma_b > 0 else 1.0

    out = np.empty(n, dtype=np.float64)
    for t in range(n):
        val_now = ratio_at(t, 0)
        lo = hi = val_now
        for back in range(1, lb):
            r = ratio_at(t, back)
            lo = min(lo, r)
            hi = max(hi, r)
        out[t] = (val_now - lo) / (hi - lo) if hi != lo else 0.5
    return out


def test_pir_matrix_matches_pine_brute_force():
    rng = np.random.default_rng(7)
    close = pd.Series(100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, 120))))
    sma_mat, pir_mat, scales = precompute_matrices(close, 2, 130)
    for s in (2, 5, 20, 37, 119, 120, 130):      # incl. s >= n_bars edge
        row = pir_mat[s - scales[0]]
        ref = _pine_pir_for_scale(close.to_numpy(), s)
        np.testing.assert_array_equal(
            row, ref, err_msg=f"scale {s} diverges from the Pine shim")


def test_pir_matrix_nan_free_and_05_during_full_warmup():
    rng = np.random.default_rng(8)
    close = pd.Series(100.0 + np.cumsum(rng.normal(0, 1.0, 80)))
    _, pir_mat, scales = precompute_matrices(close, 2, 100)
    assert not np.isnan(pir_mat).any(), "Pine warm-up never yields na pir"
    # A scale larger than the slice: sma is na everywhere -> ratio 1.0
    # everywhere -> hi == lo -> the 0.5 degenerate rule on EVERY bar.
    assert (pir_mat[100 - scales[0]] == 0.5).all()


def test_warmup_scales_vote_at_low_pct_extreme():
    """The e830d mechanism: with pct_extreme < 0.5 a fully-warm-up scale's
    pir == 0.5 counts as a HIGH extreme (0.5 > pct), raising agreement."""
    rng = np.random.default_rng(9)
    close = pd.Series(100.0 + np.cumsum(rng.normal(0, 1.0, 60)))
    _, pir_mat, scales = precompute_matrices(close, 2, 100)
    agr_h, _, n_scales = calc_agreement_fast(pir_mat, scales, 3, 99, 3, 0.40)
    assert n_scales == 33
    assert agr_h[0] > 0.0, "warm-up scales must vote 0.5 > 0.40 at bar 0"


# ---------------------------------------------------------------------------
# B2-B5 — gjr unclip, momentum_diverge_thresh, count_drift_vote
# ---------------------------------------------------------------------------

_DAX = Path("data/raw/DAX_1D_19700102_20260324.csv")


@pytest.fixture(scope="module")
def dax_slice() -> pd.DataFrame:
    if not _DAX.exists():
        pytest.skip(f"missing {_DAX}")
    from src.pooled_validation import load_stream_frame
    return load_stream_frame(str(_DAX)).iloc[-3000:].reset_index(drop=True)


@pytest.fixture(scope="module")
def dax_art(dax_slice):
    from src.detector import build_detector_artifacts
    return build_detector_artifacts(dax_slice)


def test_new_knob_defaults_are_neutral():
    p = Params()
    assert p.momentum_diverge_thresh_high == 0.0
    assert p.momentum_diverge_thresh_low == 0.0
    assert p.count_drift_vote_high is False
    assert p.count_drift_vote_low is False


def test_gjr_unclipped_range(dax_slice):
    """B2: the norm must escape the old clip(+-1) rails on real data."""
    norm, ratio = calc_gjr_asym(dax_slice)
    assert (norm.abs() > 1.0).any(), "unclipped gjr norm never left [-1, 1]"
    # the norm is still the exact affine map of the ratio (no clip residue)
    np.testing.assert_array_equal(norm.to_numpy(),
                                  ((ratio - 1.0) / 0.1).to_numpy())


def test_momentum_thresh_zero_is_legacy_and_neg_zero_safe(dax_slice, dax_art):
    """B3: 0.0 must reproduce the legacy sign test (`< -0.0` == `< 0`)."""
    from src.detector import SpeculatorDetector
    base = dataclasses.replace(Params(), use_momentum_high=True,
                               use_momentum_low=True)
    r0 = SpeculatorDetector(dax_slice, base, dax_art).run()
    rneg0 = SpeculatorDetector(
        dax_slice,
        dataclasses.replace(base, momentum_diverge_thresh_high=-0.0,
                            momentum_diverge_thresh_low=-0.0),
        dax_art).run()
    np.testing.assert_array_equal(r0["signal_high"].to_numpy(),
                                  rneg0["signal_high"].to_numpy())
    np.testing.assert_array_equal(r0["signal_low"].to_numpy(),
                                  rneg0["signal_low"].to_numpy())


def test_momentum_thresh_active_reduces_votes(dax_slice, dax_art):
    """B3 teeth: thresh > 0 yields STRICTLY fewer momentum votes/confirms."""
    from src.detector import SpeculatorDetector
    base = dataclasses.replace(Params(), use_momentum_high=True,
                               use_momentum_low=True)
    hot = dataclasses.replace(base, momentum_diverge_thresh_high=0.02,
                              momentum_diverge_thresh_low=0.02)
    r0 = SpeculatorDetector(dax_slice, base, dax_art,
                            include_debug_columns=True).run()
    r2 = SpeculatorDetector(dax_slice, hot, dax_art,
                            include_debug_columns=True).run()
    assert r2["ph_confirms"].sum() < r0["ph_confirms"].sum()
    assert r2["pl_confirms"].sum() < r0["pl_confirms"].sum()
    # measured on this slice: confirms 3103->1621 (high) / 5031->3811 (low)
    assert int(r2["signal_low"].sum()) <= int(r0["signal_low"].sum())


def test_count_drift_vote_raises_required(dax_slice, dax_art):
    """B4: with count_drift_vote the drift vote joins the requirable pool —
    LOW (mv+vola votes, confirm_count=3): legacy required = min(2, 3) = 2;
    counted-drift required = min(3, 3) = 3, so every fire needs the FULL
    drift+vote conjunction (pl_confirms >= 3)."""
    from src.detector import SpeculatorDetector
    base = Params()
    counted = dataclasses.replace(base, count_drift_vote_low=True)
    r0 = SpeculatorDetector(dax_slice, base, dax_art,
                            include_debug_columns=True).run()
    rc = SpeculatorDetector(dax_slice, counted, dax_art,
                            include_debug_columns=True).run()
    f0 = np.flatnonzero(r0["signal_low"].to_numpy())
    fc = np.flatnonzero(rc["signal_low"].to_numpy())
    assert len(f0) > 0 and len(fc) > 0           # teeth on this slice (24/6)
    assert (r0["pl_confirms"].to_numpy()[f0] < 3).any(), \
        "legacy must contain sub-conjunction fires for the knob to matter"
    assert (rc["pl_confirms"].to_numpy()[fc] >= 3).all(), \
        "counted-drift fires must satisfy the full conjunction"
    assert len(fc) < len(f0)


def test_count_drift_vote_in_max_votes_all_engines(dax_slice, dax_art):
    """B4: max_votes mirrors across FastDetector and the GPU features."""
    from src.v17_fastdetector import FastDetector
    counted = dataclasses.replace(Params(), count_drift_vote_high=True,
                                  count_drift_vote_low=True)
    fd = FastDetector(dax_slice, counted, dax_art)
    assert fd._max_votes_high == 2          # mv vote + counted drift
    assert fd._max_votes_low == 3           # mv + vola + counted drift
    torch = pytest.importorskip("torch")
    from src.v17_gpu.eval_torch import TorchPhase1
    tp = TorchPhase1(dax_slice, counted, dax_art)
    assert tp.feat.max_votes_high == fd._max_votes_high
    assert tp.feat.max_votes_low == fd._max_votes_low


def test_new_knobs_active_cpu_fast_gpu_byte_identical(dax_slice, dax_art):
    """B5: engine/Fast/GPU byte-identity with BOTH new knobs exercised."""
    from src.detector import SpeculatorDetector
    from src.v17_fastdetector import FastDetector
    p = dataclasses.replace(
        Params(),
        use_momentum_high=True, use_momentum_low=True,
        momentum_diverge_thresh_high=0.01, momentum_diverge_thresh_low=0.005,
        count_drift_vote_high=True, count_drift_vote_low=True,
    )
    ref = SpeculatorDetector(dax_slice, p, dax_art).run()
    ref_h = ref["signal_high"].to_numpy()
    ref_l = ref["signal_low"].to_numpy()
    fast = FastDetector(dax_slice, p, dax_art).signals(p)
    np.testing.assert_array_equal(fast["signal_high"], ref_h)
    np.testing.assert_array_equal(fast["signal_low"], ref_l)
    torch = pytest.importorskip("torch")
    from src.v17_gpu.eval_torch import TorchPhase1
    from src.v17_gpu.phase2_scan import signals_torch
    got = signals_torch(TorchPhase1(dax_slice, p, dax_art), p)
    np.testing.assert_array_equal(got["signal_high"], ref_h)
    np.testing.assert_array_equal(got["signal_low"], ref_l)


def test_momentum_thresh_is_a_searchable_threshold_field():
    """B3 wiring: in FLOAT_BOUNDS and active iff use_momentum is on; the
    FastDetector guard must accept varying it without a rebuild."""
    from src.search_space import FLOAT_BOUNDS, BOOL_FIELDS, space_for
    from src.v17_optimize import active_threshold_fields
    assert FLOAT_BOUNDS["momentum_diverge_thresh"] == (0.0, 0.02)
    assert "count_drift_vote" in BOOL_FIELDS
    for side in ("high", "low"):
        assert space_for(side).float_bounds["momentum_diverge_thresh"] == (0.0, 0.02)
        off = active_threshold_fields(Params(), side)
        assert f"momentum_diverge_thresh_{side}" not in off
        on = active_threshold_fields(
            dataclasses.replace(Params(), use_momentum_high=True,
                                use_momentum_low=True), side)
        assert f"momentum_diverge_thresh_{side}" in on


def test_fastdetector_varies_momentum_thresh_without_rebuild(dax_slice, dax_art):
    from src.detector import SpeculatorDetector
    from src.v17_fastdetector import FastDetector
    base = dataclasses.replace(Params(), use_momentum_high=True,
                               use_momentum_low=True)
    fd = FastDetector(dax_slice, base, dax_art)
    varied = dataclasses.replace(base, momentum_diverge_thresh_high=0.015,
                                 momentum_diverge_thresh_low=0.015)
    got = fd.signals(varied)                  # guard must NOT raise
    ref = SpeculatorDetector(dax_slice, varied, dax_art).run()
    np.testing.assert_array_equal(got["signal_high"],
                                  ref["signal_high"].to_numpy())
    np.testing.assert_array_equal(got["signal_low"],
                                  ref["signal_low"].to_numpy())
