"""Signal Card package — TDD spec (impl plan item 3; spec FIXED v3 §3).

Covers, per the task contract:

* KM survival on right-span R with right-censoring vs closed form /
  hand-computed product-limit values, incl. the "500+" cap bucket (§3.2,
  §1.2/F10) and the edge-censored ``right_span`` helper (R1).
* Cluster-bootstrap confidence bands: fixed-seed reproducibility and the
  block fallback for < 5 streams (F6).
* Live conditioning identity ``= 0 if x > L else S_R(max(x,k))/S_R(k)``
  (F1) incl. the off-grid step-function-floor lookup rule at non-grid
  arguments (R4) and the noise floor ``S(x<20) = 1``.
* Discrete-survival E[hold] with the 500 cap and the L-clamp (§3.4).
* c_side two-fold live-regressor recovery on synthetic data, the R² < 0.1
  pooled-median fallback, and the R3 censored/unmatched exclusion (F2).
* Conviction percentile (§3.6).
* The F8 stop truth table rows 1-6 verbatim (LOW side + HIGH mirror),
  incl. the R2 k-origin (bars survived counted from the candidate pivot
  bar ``i`` with the t+1 shift) and the row-6 re-fire stop-series switch.
* Retrospective grading on a constructed series and the R-multiple
  backtest accounting identity (sum of R-multiples == equity delta) (§3.7).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.scoring_v5 import SPAN_GRID
from src.v17_card import (
    CSideFit,
    KMSurvival,
    SignalGrade,
    SignalRecord,
    StopState,
    SurvivalBands,
    backtest_summary,
    cluster_bootstrap_bands,
    conditional_survival,
    conviction_percentile,
    expected_hold,
    expected_move,
    fit_c_side,
    fit_km,
    grade_signals,
    init_stop_state,
    km_at,
    km_table,
    right_span,
    stop_series,
    survival_lookup,
    track_signal,
    update_stop_state,
)

GRID_DELTAS = [20, 10, 10, 10, 20, 30, 40, 60, 100, 200]  # Δx_j of SPAN_GRID


# ---------------------------------------------------------------------------
# §3.2 / R1 — right-span R
# ---------------------------------------------------------------------------


def test_right_span_low_side_strictly_lower_breaks():
    arr = np.array([5.0, 4.0, 3.0, 4.0, 3.0, 2.0, 9.0])
    # i=2 (val 3): equal lows at 4 do NOT break; first strictly lower at j=5.
    r, cens = right_span(arr, 2)
    assert (r, cens) == (3, False)


def test_right_span_high_side_mirrors():
    arr = np.array([1.0, 2.0, 3.0, 3.0, 2.5, 4.0])
    r, cens = right_span(arr, 2, is_high=True)
    assert (r, cens) == (3, False)  # strictly higher high at j=5


def test_right_span_edge_censored_lower_bound():
    arr = np.array([5.0, 3.0, 4.0, 4.0])
    r, cens = right_span(arr, 1)
    assert (r, cens) == (2, True)  # no breach to series end: R >= 2, censored


def test_right_span_cap_censored_500_plus():
    arr = np.full(700, 1.0)
    arr[0] = 0.5
    r, cens = right_span(arr, 0)
    assert (r, cens) == (SPAN_GRID[-1], True)  # "500+" bucket, never exact
    # breach beyond the cap must still record the censored cap
    arr2 = np.full(700, 1.0)
    arr2[0] = 0.5
    arr2[600] = 0.1
    assert right_span(arr2, 0) == (SPAN_GRID[-1], True)


# ---------------------------------------------------------------------------
# §3.2 — Kaplan-Meier vs closed form / hand-computed values
# ---------------------------------------------------------------------------


def test_km_uncensored_equals_empirical_survival():
    values = np.array([25, 25, 45, 80, 80, 250], dtype=float)
    curve = fit_km(values, np.zeros(6, dtype=bool))
    for x in range(1, 301, 7):
        assert km_at(curve, x) == pytest.approx(float(np.mean(values >= x)))
    assert km_at(curve, 30) == pytest.approx(4 / 6)
    assert km_at(curve, 50) == pytest.approx(0.5)
    assert km_at(curve, 100) == pytest.approx(1 / 6)
    assert km_at(curve, 300) == 0.0


def test_km_censored_hand_computed_product_limit():
    # values 30,50,50c,70,90c — events at 30 (n=5), 50 (n=4), 70 (n=2)
    values = np.array([30, 50, 50, 70, 90], dtype=float)
    cens = np.array([False, False, True, False, True])
    curve = fit_km(values, cens)
    assert km_at(curve, 30) == pytest.approx(1.0)
    assert km_at(curve, 31) == pytest.approx(0.8)
    assert km_at(curve, 50) == pytest.approx(0.8)
    assert km_at(curve, 51) == pytest.approx(0.8 * 0.75)
    assert km_at(curve, 70) == pytest.approx(0.6)
    assert km_at(curve, 71) == pytest.approx(0.6 * 0.5)
    assert km_at(curve, 200) == pytest.approx(0.3)  # censored tail never hits 0


def test_km_closed_form_geometric_with_random_censoring():
    rng = np.random.default_rng(7)
    n = 20000
    p, q = 0.15, 0.85
    t_true = rng.geometric(p, n).astype(float)
    c_cens = rng.geometric(0.08, n).astype(float)
    obs = np.minimum(t_true, c_cens)
    cens = c_cens < t_true  # tie -> event observed (standard convention)
    curve = fit_km(obs, cens)
    for x in (2, 5, 10, 15):
        assert km_at(curve, x) == pytest.approx(q ** (x - 1), abs=0.02)


def test_km_500_cap_right_censored_not_exact():
    values = np.array([100, 500, 500], dtype=float)
    cens = np.array([False, True, True])
    curve = fit_km(values, cens)
    table = km_table(curve)
    assert km_at(curve, 500) == pytest.approx(2 / 3)
    assert km_at(curve, 501) == pytest.approx(2 / 3)  # cap is a bound, not an event
    assert table[-1] == pytest.approx(2 / 3)
    assert table[0] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# F6 — cluster-bootstrap bands
# ---------------------------------------------------------------------------


def _synthetic_clusters(n_clusters: int, seed: int = 3):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_clusters):
        vals = 20.0 + 10.0 * rng.geometric(0.3, 40)
        cens = rng.random(40) < 0.15
        out.append((vals, cens))
    return out


def test_cluster_bootstrap_bands_reproducible_under_fixed_seed():
    clusters = _synthetic_clusters(6)
    b1 = cluster_bootstrap_bands(clusters, n_boot=200, seed=11)
    b2 = cluster_bootstrap_bands(clusters, n_boot=200, seed=11)
    assert np.array_equal(b1.s_lo, b2.s_lo)
    assert np.array_equal(b1.s_hi, b2.s_hi)
    assert b1.method == "stream"
    assert np.all(b1.s_lo <= b1.s_hi + 1e-12)
    # monotone non-increasing envelopes; point estimate inside the band
    assert np.all(np.diff(b1.s_lo) <= 1e-12)
    assert np.all(np.diff(b1.s_hi) <= 1e-12)
    assert np.all(b1.s_lo - 0.05 <= b1.s_point)
    assert np.all(b1.s_point <= b1.s_hi + 0.05)


def test_cluster_bootstrap_block_fallback_below_five_streams():
    clusters = _synthetic_clusters(2)
    b1 = cluster_bootstrap_bands(clusters, n_boot=100, seed=5)
    b2 = cluster_bootstrap_bands(clusters, n_boot=100, seed=5)
    assert b1.method == "block"  # contiguous signal-blocks per stream (F6)
    assert np.array_equal(b1.s_lo, b2.s_lo)
    assert np.array_equal(b1.s_hi, b2.s_hi)


# ---------------------------------------------------------------------------
# F1 / R4 — conditioning identity, L-clamp, off-grid floor lookup
# ---------------------------------------------------------------------------

TABLE = np.array([0.95, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1])


def test_survival_lookup_floor_rule_off_grid():
    assert survival_lookup(TABLE, 19.999) == 1.0  # noise floor (R4)
    assert survival_lookup(TABLE, 0) == 1.0
    assert survival_lookup(TABLE, 20) == pytest.approx(0.95)
    assert survival_lookup(TABLE, 29.9) == pytest.approx(0.95)
    assert survival_lookup(TABLE, 30) == pytest.approx(0.9)
    assert survival_lookup(TABLE, 180) == pytest.approx(0.4)  # floor to 140
    assert survival_lookup(TABLE, 500) == pytest.approx(0.1)
    assert survival_lookup(TABLE, 1e9) == pytest.approx(0.1)


def test_conditioning_identity_at_non_grid_arguments():
    # P = S(max(x,k))/S(k) with floor lookups: k=37 -> S@30, x=95 -> S@70
    p = conditional_survival(TABLE, x=95, k=37, left_span=600)
    assert p == pytest.approx(0.6 / 0.9)
    # x <= k -> certainty
    assert conditional_survival(TABLE, x=25, k=37, left_span=600) == 1.0
    # unconditional at k=0 (S(0)=1 below the noise floor)
    assert conditional_survival(TABLE, x=20, k=0, left_span=600) == pytest.approx(0.95)


def test_conditioning_left_span_clamp_zero_above_L():
    assert conditional_survival(TABLE, x=95, k=37, left_span=80) == 0.0
    assert conditional_survival(TABLE, x=80.001, k=0, left_span=80) == 0.0
    # at x == L the clamp does NOT bite
    assert conditional_survival(TABLE, x=80, k=37, left_span=80) == pytest.approx(0.6 / 0.9)


def test_conditioning_zero_survival_guard():
    dead = TABLE.copy()
    dead[-1] = 0.0
    assert conditional_survival(dead, x=500, k=500, left_span=600) == 0.0


def test_expected_hold_discrete_expectation_and_cap():
    ones = np.ones(10)
    assert expected_hold(ones, k=0, left_span=600) == pytest.approx(500.0)  # cap
    assert expected_hold(ones, k=0, left_span=50) == pytest.approx(50.0)  # L-clamp
    # terms with x_j > L contribute zero: L=85 keeps grid points <= 70
    assert expected_hold(ones, k=0, left_span=85) == pytest.approx(70.0)
    half = np.array([1, 1, 1, 1, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
    assert expected_hold(half, k=0, left_span=600) == pytest.approx(50 + 0.5 * 450)
    # flat-after-70 table conditioned past 70: no remaining hazard -> cap
    assert expected_hold(half, k=75, left_span=600) == pytest.approx(500.0)


# ---------------------------------------------------------------------------
# F2 / R3 — c_side two-fold fit, fallback, exclusions
# ---------------------------------------------------------------------------


def _make_records(rng, n: int, c_true: float, noise: float = 0.02):
    spans = rng.integers(25, 400, n).astype(float)
    curve = fit_km(spans, np.zeros(n, dtype=bool))
    table = km_table(curve)
    e_hold = expected_hold(table, k=0, left_span=SPAN_GRID[-1])
    recs = []
    for j in range(n):
        sigma = float(rng.uniform(0.5, 2.0))
        x = sigma * np.sqrt(e_hold)
        y = c_true * x * (1.0 + noise * float(rng.standard_normal()))
        recs.append(SignalRecord(
            fire_bar=j, sigma_har=sigma, left_span=SPAN_GRID[-1],
            right_span=float(spans[j]), right_censored=False,
            matched=True, span_censored=False, realized_abs_move=y,
        ))
    return recs


def test_c_side_two_fold_recovery_on_synthetic():
    rng = np.random.default_rng(42)
    fit = fit_c_side(_make_records(rng, 600, c_true=2.5))
    assert fit.c_side == pytest.approx(2.5, rel=0.1)
    assert fit.r_squared > 0.9
    assert not fit.use_fallback
    assert fit.n_fit == 600
    # expected_move uses the fitted scalar
    assert expected_move(fit, sigma_har=1.0, e_hold=100.0) == pytest.approx(
        fit.c_side * 10.0)


def test_c_side_r2_fallback_to_pooled_median():
    rng = np.random.default_rng(1)
    n = 300
    spans = rng.integers(25, 400, n).astype(float)
    moves = rng.uniform(4.0, 6.0, n)
    recs = [SignalRecord(
        fire_bar=j, sigma_har=float(rng.uniform(0.2, 3.0)),
        left_span=SPAN_GRID[-1], right_span=float(spans[j]),
        right_censored=False, matched=True, span_censored=False,
        realized_abs_move=float(moves[j]),
    ) for j in range(n)]
    fit = fit_c_side(recs)
    assert fit.r_squared < 0.1
    assert fit.use_fallback
    assert fit.fallback_median == pytest.approx(float(np.median(moves)))
    # fallback rule: the card shows the pooled median move (spec §3.3)
    assert expected_move(fit, sigma_har=9.9, e_hold=400.0) == pytest.approx(
        fit.fallback_median)


def test_c_side_excludes_censored_and_unmatched_responses():
    rng = np.random.default_rng(9)
    base = _make_records(rng, 400, c_true=2.0, noise=0.0)
    junk = []
    for j in range(40):
        junk.append(SignalRecord(
            fire_bar=1000 + j, sigma_har=1.0, left_span=SPAN_GRID[-1],
            right_span=float(rng.integers(25, 400)),
            right_censored=False, matched=True,
            span_censored=True, realized_abs_move=1e6,  # R3: no observable move
        ))
        junk.append(SignalRecord(
            fire_bar=2000 + j, sigma_har=1.0, left_span=SPAN_GRID[-1],
            right_span=float(rng.integers(25, 400)),
            right_censored=False, matched=False,  # unmatched: not in response
            span_censored=False, realized_abs_move=1e6,
        ))
    fit = fit_c_side(base + junk)
    assert fit.n_fit == 400  # only eligible responses enter the regression
    assert fit.c_side == pytest.approx(2.0, rel=0.15)  # 1e6 moves never leak in
    assert not fit.use_fallback


# ---------------------------------------------------------------------------
# §3.6 — conviction percentile
# ---------------------------------------------------------------------------


def test_conviction_percentile_midrank():
    hist = [1.0, 2.0, 3.0, 4.0]
    assert conviction_percentile(5.0, hist) == pytest.approx(100.0)
    assert conviction_percentile(2.5, hist) == pytest.approx(50.0)
    assert conviction_percentile(3.0, hist) == pytest.approx(62.5)  # midrank tie
    assert conviction_percentile(0.0, hist) == pytest.approx(0.0)
    assert conviction_percentile(1.0, []) == pytest.approx(50.0)


# ---------------------------------------------------------------------------
# F8 / R2 — stop truth table, verbatim rows (LOW side; HIGH mirror)
# ---------------------------------------------------------------------------


def test_stop_row1_fire_time_stop_and_candidate_pivot_bar():
    lows = np.array([5.0, 4.0, 3.0, 3.5, 3.6])
    st = init_stop_state(lows, t=2)
    assert st.stop == 3.0 and st.fire_stop == 3.0  # min(low[t-1], low[t])
    assert st.pivot_bar == 2  # low[2] < low[1] -> i = t
    assert st.bars_survived == 0  # k counted from i (R2)
    assert st.left_span == 2  # lows[1], lows[0] >= 3.0 -> scan to left edge
    assert st.intact and not st.final and st.invalidated_at is None


def test_stop_row1_tie_prefers_earlier_bar():
    lows = np.array([5.0, 3.0, 3.0, 3.5])
    st = init_stop_state(lows, t=2)
    assert st.pivot_bar == 1  # tie -> EARLIER bar (plateau-collapse-to-first)
    assert st.bars_survived == 1  # k = t - i


def test_stop_row2_survive_widen_shift_and_k_restart():
    lows = np.array([5.0, 4.0, 3.0, 2.0, 2.5])
    closes = np.array([5.2, 4.2, 3.2, 3.1, 2.9])
    st = init_stop_state(lows, t=2)
    st = update_stop_state(st, lows, closes, 3)  # close[3] >= fire stop
    assert st.intact and st.final  # stop now FINAL
    assert st.stop == 2.0  # widened once: min(stop, low[t+1])
    assert st.pivot_bar == 3  # strict new minimum -> i = t+1 (R2 shift)
    assert st.left_span == 3  # L RECOMPUTED at the new i (causal backward scan)
    assert st.bars_survived == 0  # k restarts from the new i
    st = update_stop_state(st, lows, closes, 4)
    assert st.bars_survived == 1  # row 5: k = u - i


def test_stop_row2_tie_keeps_earlier_pivot_bar():
    lows = np.array([5.0, 4.0, 3.0, 3.0, 3.5])
    closes = lows + 0.4
    st = init_stop_state(lows, t=2)
    st = update_stop_state(st, lows, closes, 3)
    assert st.stop == 3.0 and st.pivot_bar == 2  # tie -> i keeps the earlier bar
    assert st.bars_survived == 1


def test_stop_row2_intrabar_wick_at_t_plus_1_must_not_invalidate():
    # a lower LOW at t+1 is hypothesis-CONSISTENT (the wiggle bar);
    # only a CLOSE through the fire-time stop invalidates at t+1 (F8).
    lows = np.array([5.0, 4.0, 3.0, 1.0, 2.5])
    closes = np.array([5.2, 4.2, 3.2, 3.4, 2.9])
    st = init_stop_state(lows, t=2)
    st = update_stop_state(st, lows, closes, 3)
    assert st.intact
    assert st.stop == 1.0 and st.pivot_bar == 3


def test_stop_row3_close_breach_at_t_plus_1_no_widening():
    lows = np.array([5.0, 4.0, 3.0, 2.0, 2.5])
    closes = np.array([5.2, 4.2, 3.2, 2.9, 2.9])  # close[3] < fire stop 3.0
    st = init_stop_state(lows, t=2)
    st = update_stop_state(st, lows, closes, 3)
    assert not st.intact and st.invalidated_at == 3
    assert st.stop == 3.0  # breach checked against the PRE-widening stop; no widening
    assert st.bars_survived == 1  # frozen at u - i
    # frozen thereafter
    st2 = update_stop_state(st, lows, closes, 4)
    assert not st2.intact and st2.stop == 3.0 and st2.invalidated_at == 3


def test_stop_row4_intrabar_breach_after_t_plus_1_final_stop():
    lows = np.array([5.0, 4.0, 3.0, 3.2, 3.1, 2.9, 3.5])
    closes = lows + 0.4
    st = init_stop_state(lows, t=2)
    st = update_stop_state(st, lows, closes, 3)
    st = update_stop_state(st, lows, closes, 4)
    assert st.intact and st.bars_survived == 2  # row 5
    st = update_stop_state(st, lows, closes, 5)  # low[5]=2.9 < stop 3.0 intrabar
    assert not st.intact and st.invalidated_at == 5
    assert st.bars_survived == 3  # freezes at u - i


def test_stop_row5_touch_at_stop_survives():
    lows = np.array([5.0, 4.0, 3.0, 3.2, 3.0, 3.4])
    closes = lows + 0.4
    states = track_signal(lows, closes, t=2, end=5)
    assert states[-1].intact  # low[u] == stop -> survives (>= rule)
    assert states[-1].bars_survived == 3


def test_stop_row6_refire_resets_state_and_switches_plotted_stop():
    lows = np.array([5.0, 4.0, 3.0, 3.5, 3.6, 3.7, 3.2, 3.3, 3.4, 3.5])
    closes = lows + 0.5
    out = stop_series(lows, closes, fire_bars=[2, 6])
    assert np.all(np.isnan(out[:2]))
    assert np.allclose(out[2:6], 3.0)  # first card's stop
    # the plotted stop switches to the NEW signal's stop on its fire bar
    assert np.allclose(out[6:], min(lows[5], lows[6]))


def test_stop_series_na_after_row4_intrabar_invalidation():
    # DECIDED SEMANTIC (F8 / §4.2 parity pin): the plotted stop is Pine's
    # natural ``plot(intact ? stop : na)`` evaluated at bar close — NaN ON
    # the invalidation bar and every bar after, until a same-side re-fire.
    # Engine and Pine v17.5 builder MUST both implement this; the §4.2
    # bar-for-bar stop parity treats engine-NaN == Pine-na.
    lows = np.array([5.0, 4.0, 3.0, 3.5, 2.9, 3.1, 3.2])
    closes = lows + 0.5
    out = stop_series(lows, closes, fire_bars=[2])
    assert np.allclose(out[2:4], 3.0)  # intact bars plot the stop
    # row 4: low[4] = 2.9 < final stop 3.0 -> invalidated at bar 4
    assert np.all(np.isnan(out[4:]))  # na from the invalidation bar onward


def test_stop_series_na_after_row3_close_breach_at_t_plus_1():
    lows = np.array([5.0, 4.0, 3.0, 2.0, 2.5, 2.6])
    closes = np.array([5.2, 4.2, 3.2, 2.9, 2.9, 3.0])  # close[3] < fire stop
    out = stop_series(lows, closes, fire_bars=[2])
    assert out[2] == pytest.approx(3.0)
    assert np.all(np.isnan(out[3:]))  # row 3 invalidation at t+1 -> na


def test_stop_series_refire_after_invalidation_resumes_plotting():
    lows = np.array([5.0, 4.0, 3.0, 3.5, 2.9, 3.1, 3.0, 3.2, 3.3])
    closes = lows + 0.5
    out = stop_series(lows, closes, fire_bars=[2, 6])
    assert np.allclose(out[2:4], 3.0)
    assert np.all(np.isnan(out[4:6]))  # stopped out at 4; na until re-fire
    # row 6 re-fire at 6: plotting resumes with the NEW card's stop
    assert np.allclose(out[6:], min(lows[5], lows[6]))


def test_stop_series_na_after_invalidation_high_side_mirror():
    highs = np.array([5.0, 6.0, 7.0, 6.5, 7.1, 6.9])
    closes = highs - 0.5
    out = stop_series(highs, closes, fire_bars=[2], is_high=True)
    assert np.allclose(out[2:4], 7.0)
    # row 4 mirror: high[4] = 7.1 > final stop 7.0 -> invalidated at bar 4
    assert np.all(np.isnan(out[4:]))


def test_stop_high_side_mirror():
    highs = np.array([5.0, 6.0, 7.0, 8.0, 6.5])
    closes = np.array([4.9, 5.9, 6.9, 6.9, 6.4])
    st = init_stop_state(highs, t=2, is_high=True)
    assert st.stop == 7.0 and st.pivot_bar == 2
    st = update_stop_state(st, highs, closes, 3)  # close 6.9 <= 7.0 -> survive
    assert st.intact and st.stop == 8.0 and st.pivot_bar == 3  # widened + shifted
    st = update_stop_state(st, highs, closes, 4)
    assert st.intact and st.bars_survived == 1
    # close-based mirror breach at t+1
    st2 = init_stop_state(highs, t=2, is_high=True)
    closes_breach = np.array([4.9, 5.9, 6.9, 7.5, 6.4])
    st2 = update_stop_state(st2, highs, closes_breach, 3)
    assert not st2.intact and st2.stop == 7.0


# ---------------------------------------------------------------------------
# §3.7 — retrospective grading + R-multiple backtest
# ---------------------------------------------------------------------------


def _grading_fixture():
    """V-bottom at bar 150 (span 100, uncensored), deep low at bar 20
    truncating the span; signal at 151 (matched, clock exit) and at 300
    (unmatched, intrabar stop-out at bar 303)."""
    n = 400
    b = np.arange(n, dtype=float)
    low = np.where(b <= 150, 50.0 - 0.2 * b, 20.0 + 0.1 * (b - 150.0))
    low[20] = 10.0  # caps N*(150) at grid 100 (window +-140 reaches bar 20)
    low[303] = 33.9  # intrabar dip below signal-2 stop (low[299] ~ 34.9)
    df = pd.DataFrame({
        "open": low + 0.2, "high": low + 0.5, "low": low,
        "close": low + 0.3, "volume": np.ones(n),
    })
    sig = np.zeros(n, dtype=bool)
    sig[151] = True
    sig[300] = True
    return df, sig


def test_grading_on_constructed_series():
    df, sig = _grading_fixture()
    low = df["low"].to_numpy()
    close = df["close"].to_numpy()
    grades = grade_signals(df, sig, side="low", clock_bars=60)
    assert len(grades) == 2
    g1, g2 = sorted(grades, key=lambda g: g.fire_bar)

    # signal 1: direct hit on the bar-150 pivot (distance 1)
    assert g1.fire_bar == 151 and g1.matched and g1.pivot_bar == 150
    assert g1.realized_span == 100 and not g1.span_censored
    assert g1.tier == "T2"  # 50 <= N* < 200 (spec §1.4)
    assert g1.entry == pytest.approx(close[151])
    assert g1.stop == pytest.approx(low[150])  # fire-time stop = min(low[150:152])
    risk1 = close[151] - low[150]
    assert g1.risk == pytest.approx(risk1)
    assert g1.outcome == "clock" and g1.exit_bar == 210  # i=150 + 60 bars
    r1 = (close[210] - close[151]) / risk1
    assert g1.r_multiple == pytest.approx(r1)
    assert g1.realized_move == pytest.approx(abs(close[250] - low[150]))
    assert g1.mae_r == pytest.approx((close[151] - low[151:211].min()) / risk1)
    assert g1.mfe_r == pytest.approx(
        (df["high"].to_numpy()[151:211].max() - close[151]) / risk1)

    # signal 2: no pivot in +-1, intrabar stop-out at 303, R = -1 exactly
    assert g2.fire_bar == 300 and not g2.matched and g2.tier == "miss"
    assert g2.outcome == "stopped_intrabar" and g2.exit_bar == 303
    assert g2.exit_price == pytest.approx(low[299])
    assert g2.r_multiple == pytest.approx(-1.0)
    assert np.isnan(g2.realized_move)


def test_backtest_accounting_identity():
    df, sig = _grading_fixture()
    grades = grade_signals(df, sig, side="low", clock_bars=60)
    summary = backtest_summary(grades)
    assert summary["n_signals"] == 2 and summary["n_trades"] == 2
    total_r = sum(g.r_multiple for g in grades)
    # accounting identity: sum of R-multiples == equity delta of 1-risk-unit trades
    assert summary["total_r"] == pytest.approx(total_r, abs=1e-12)
    assert summary["equity_delta"] == pytest.approx(total_r, abs=1e-9)
    # independent recomputation from raw prices
    close = df["close"].to_numpy()
    low = df["low"].to_numpy()
    r1 = (close[210] - close[151]) / (close[151] - low[150])
    expected_equity = r1 + (-1.0)
    assert summary["equity_delta"] == pytest.approx(expected_equity)
    assert summary["win_rate"] == pytest.approx(0.5)
    assert summary["expectancy"] == pytest.approx(total_r / 2.0)
    # capture ratio: realized capture over available move of matched signals
    realized = close[210] - close[151]
    available = abs(close[250] - low[150])
    assert summary["capture_ratio"] == pytest.approx(realized / available)
