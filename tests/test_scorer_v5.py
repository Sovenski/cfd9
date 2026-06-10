"""Scorer v5 core — TDD spec (impl plan T1-T5; spec FIXED v3 §1-§2).

Covers, per the task contract:

* ``SPAN_GRID`` / span weights / grid-floor truncation (§1.2-§1.3, F10)
* right-edge + "500+" censoring with lower-bound mass (§1.2)
* ±1 direct-hit matching truth table incl. edges, Hungarian 1:1 and the
  larger-span tie-break (§2.1, §7.2)
* weighted precision / recall: exact hand-computed values + [0, 1] ranges
  (§2.2)
* ``W_FP`` exchange-rate constant + sensitivity test with firing-gate catch
  (F3) via ``firing_excess`` folded into ``rank_by_penalized_lcb``
* ``REFERENCE_MASS`` / N_eff recall-target basis (F4)
* informative-fold filter on pivot MASS (§2.4) incl. the span-30-only fold
  that the old ``pivot_N100`` count check would have discarded
* ``pivot_N100`` column retention in ``add_pivot_labels`` (§1.2)
"""
from __future__ import annotations

import inspect

import numpy as np
import pandas as pd
import pytest

import src.scoring_v5 as scoring_v5
from src.pooled_scoring import StreamStat, pooled_fold_score, pooled_side_score
from src.pooled_validation import _fold_is_informative, _stream_stat
from src.scoring import RECALL_TARGET, REFERENCE_N, add_pivot_labels, label_pivots
from src.scoring_v5 import (
    MATCH_WINDOW,
    REFERENCE_MASS,
    SPAN_GRID,
    TIEBREAK_EPS,
    W_FP,
    WeightedStats,
    compute_left_span,
    compute_side_score_v5,
    label_pivot_spans,
    match_signals_weighted,
    recall_target_eff,
    span_weight,
    weighted_precision,
    weighted_recall,
)
from src.v17_acceptance import firing_excess, rank_by_penalized_lcb
from src.validation import fold_scores_bootstrap_ci


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _v_df(n: int, c: int, extra_lows: dict[int, float] | None = None) -> pd.DataFrame:
    """V-shaped series with its unique minimum at bar ``c``.

    On a strict V, the ONLY interior pivot low is ``c`` (every other bar's
    centered window contains a lower bar nearer ``c``) and there are no
    interior pivot highs. ``extra_lows`` plants deeper minima to truncate
    the span of ``c`` at a known grid value.
    """
    low = np.abs(np.arange(n) - c).astype(float)
    if extra_lows:
        for pos, val in extra_lows.items():
            low[pos] = val
    high = low + 1.0
    close = low + 0.5
    return pd.DataFrame({
        "open": close, "high": high, "low": low, "close": close,
        "volume": np.ones(n),
    })


def _match(signal_pos: list[int], span_map: dict[int, int], n: int = 200) -> WeightedStats:
    sig = np.zeros(n, dtype=bool)
    sig[signal_pos] = True
    spans = np.zeros(n, dtype=np.int32)
    for p, s in span_map.items():
        spans[p] = s
    return match_signals_weighted(sig, spans)


# ---------------------------------------------------------------------------
# §1.2 / §1.3 — grid, weights, grid-floor, noise floor
# ---------------------------------------------------------------------------


def test_span_grid_pinned():
    assert SPAN_GRID == [20, 30, 40, 50, 70, 100, 140, 200, 300, 500]


def test_span_weights_exact_for_every_grid_value():
    for g in SPAN_GRID:
        assert span_weight(g) == pytest.approx(g / REFERENCE_N, abs=0.0)
    assert W_FP == pytest.approx(0.2) == pytest.approx(span_weight(20))
    assert REFERENCE_MASS == pytest.approx(54.0) == pytest.approx(27 * span_weight(200))


def test_w_fp_exchange_rate_documented_at_definition_site():
    """F3(a): the exchange-rate meaning is load-bearing documentation."""
    src = inspect.getsource(scoring_v5)
    assert "W_FP" in src
    assert "ten false signals cost one T1" in src


def test_grid_floor_truncates_span_to_largest_grid_value_passed():
    """F10: a true span-180 pivot records 140 (uncensored — the 200-window
    fits and genuinely fails)."""
    df = _v_df(1000, 500, extra_lows={681: -5.0})
    spans = label_pivot_spans(df)
    assert int(spans["pivot_span_low"].iloc[500]) == 140
    assert bool(spans["pivot_span_censored_low"].iloc[500]) is False
    # the planted deeper low is itself a pivot, censored at its 300 bound
    # (the 500-window does not fit on the right).
    assert int(spans["pivot_span_low"].iloc[681]) == 300
    assert bool(spans["pivot_span_censored_low"].iloc[681]) is True


def test_noise_floor_19_bar_minimum_is_no_event():
    """§1.2: pivots with N* < 20 do not exist as events."""
    df = _v_df(1000, 500, extra_lows={520: -5.0})
    spans = label_pivot_spans(df)
    assert int(spans["pivot_span_low"].iloc[500]) == 0


def test_span_masks_match_frozen_labeler_per_grid_scale():
    """Anchor to the frozen-semantics labeler: ``span >= N`` is exactly the
    ``label_pivots(df, N) == -1`` mask at every fitting grid scale."""
    df = _v_df(1000, 500, extra_lows={681: -5.0})
    span_low = label_pivot_spans(df)["pivot_span_low"].to_numpy()
    for N in SPAN_GRID:
        if len(df) < 2 * N + 1:
            continue
        lbl = label_pivots(df, N).to_numpy()
        assert np.array_equal(span_low >= N, lbl == -1), f"mask mismatch at N={N}"


# ---------------------------------------------------------------------------
# §1.2 — censoring (right edge + "500+" cap) and lower-bound mass
# ---------------------------------------------------------------------------


def test_right_edge_censoring_records_proven_lower_bound():
    """A bar 60 bars from the end with a true >=50 window: N*_lb = 50,
    censored flag set (the 70-window cannot fit on the right)."""
    n = 1000
    c = n - 1 - 60
    df = _v_df(n, c)
    spans = label_pivot_spans(df)
    assert int(spans["pivot_span_low"].iloc[c]) == 50
    assert bool(spans["pivot_span_censored_low"].iloc[c]) is True


def test_top_bucket_500_is_always_censored():
    """'500+' semantics: a pivot passing 500 is recorded 500 AND censored."""
    df = _v_df(1100, 550)
    spans = label_pivot_spans(df)
    assert int(spans["pivot_span_low"].iloc[550]) == 500
    assert bool(spans["pivot_span_censored_low"].iloc[550]) is True


def test_censored_pivot_contributes_lower_bound_mass():
    """§2.2: censored pivots contribute w(N*_lb) to tp/total mass."""
    df = _v_df(1000, 500, extra_lows={681: -5.0})
    df = add_pivot_labels(df)
    sig = pd.Series(False, index=df.index)
    sig.iloc[681] = True                       # signal on the censored pivot
    stats = match_signals_weighted(sig, df["pivot_span_low"])
    assert stats.tp_mass == pytest.approx(span_weight(300))   # w(N*_lb) = 3.0
    assert stats.total_mass == pytest.approx(span_weight(140) + span_weight(300))


# ---------------------------------------------------------------------------
# §2.1 — ±1 matching truth table, Hungarian 1:1, larger-span tie-break
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("pivot_at, matched", [
    (9, True), (10, True), (11, True),   # t-1, t, t+1 → direct hit
    (8, False), (12, False),             # t-2, t+2 → FP + missed mass
])
def test_pm1_matching_truth_table(pivot_at: int, matched: bool):
    stats = _match([10], {pivot_at: 100})
    if matched:
        assert stats.tp_mass == pytest.approx(1.0)
        assert stats.n_unmatched == 0
    else:
        assert stats.tp_mass == 0.0
        assert stats.n_unmatched == 1            # false positive
        assert stats.total_mass == pytest.approx(1.0)  # missed mass stays


def test_matching_is_symmetric_no_lead_bias():
    lead = _match([10], {11: 100})
    lag = _match([10], {9: 100})
    assert lead == lag


def test_hungarian_one_to_one():
    stats = _match([9, 11], {10: 100})
    assert stats.tp_mass == pytest.approx(1.0)   # exactly one TP
    assert stats.n_unmatched == 1


def test_equal_distance_tie_break_prefers_larger_span():
    stats = _match([10], {9: 50, 11: 200})
    assert stats.tp_mass == pytest.approx(span_weight(200))
    assert stats.n_unmatched == 0
    assert stats.total_mass == pytest.approx(2.5)


def test_tie_break_epsilon_never_flips_a_distance_ranking():
    assert TIEBREAK_EPS < 1.0 / (2.0 * span_weight(SPAN_GRID[-1]))
    # a distance-0 floor pivot must beat a distance-1 cap pivot
    stats = _match([10], {10: 20, 11: 500})
    assert stats.tp_mass == pytest.approx(span_weight(20))


# ---------------------------------------------------------------------------
# §2.2 — weighted precision / recall
# ---------------------------------------------------------------------------


def test_weighted_precision_recall_exact_on_hand_fixture():
    stats = _match([10, 50], {10: 50, 30: 200})
    assert stats.tp_mass == pytest.approx(0.5)
    assert stats.total_mass == pytest.approx(2.5)
    assert stats.n_unmatched == 1
    assert weighted_precision(stats) == pytest.approx(0.5 / (0.5 + 1 * W_FP))
    assert weighted_recall(stats) == pytest.approx(0.5 / 2.5)


def test_weighted_precision_recall_ranges():
    rng = np.random.default_rng(7)
    sig = rng.random(500) < 0.05
    spans = np.where(rng.random(500) < 0.05,
                     rng.choice(SPAN_GRID, size=500), 0)
    stats = match_signals_weighted(sig, spans)
    assert 0.0 <= weighted_precision(stats) <= 1.0
    assert 0.0 <= weighted_recall(stats) <= 1.0


# ---------------------------------------------------------------------------
# F4 — REFERENCE_MASS / N_eff recall-target basis
# ---------------------------------------------------------------------------


def test_reference_mass_anchor_and_recall_target_eff():
    """The v4<->v5 basis anchor can never drift silently."""
    assert REFERENCE_MASS == 54.0
    assert recall_target_eff(54.0) == pytest.approx(RECALL_TARGET)
    assert recall_target_eff(13.5) == pytest.approx(RECALL_TARGET * 2.0)
    assert recall_target_eff(216.0) == pytest.approx(RECALL_TARGET / 2.0)


def test_compute_side_score_v5_hand_computed():
    stats = WeightedStats(tp_mass=2.0, total_mass=4.0, n_signals=3.0,
                          n_unmatched=1.0, n_bars=1000.0)
    precision = 2.0 / (2.0 + 1.0 * W_FP)
    recall = 0.5
    target = RECALL_TARGET * np.sqrt(REFERENCE_MASS / 4.0)
    recall_sat = 1.0 - np.exp(-recall / target)
    expected = (precision ** 1.2) * recall_sat * 1.0 * (2 * 3 * 4 / (9 + 16))
    score = compute_side_score_v5(stats)
    assert score == pytest.approx(expected)
    assert score == pytest.approx(0.2469, abs=5e-4)   # independent hand pin
    # w_FP sensitivity direction: a higher FP price lowers the score
    assert compute_side_score_v5(stats, w_fp=1.0) < compute_side_score_v5(stats, w_fp=0.2)


# ---------------------------------------------------------------------------
# F3 — w_FP sensitivity: SPRAY vs SELECTIVE through the firing gate
# ---------------------------------------------------------------------------


def _toy_landscape() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Synthetic pivots every 200 bars (spans cycling the grid's tiers) plus
    a SPRAY candidate (fires every 5 bars) and a SELECTIVE candidate (fires
    exactly on the true pivot bars)."""
    n = 4000
    spans = np.zeros(n, dtype=np.int32)
    cycle = [20, 50, 100, 200]
    for k, pos in enumerate(range(200, n, 200)):
        spans[pos] = cycle[k % 4]
    spray = np.zeros(n, dtype=bool)
    spray[::5] = True
    selective = np.zeros(n, dtype=bool)
    selective[spans > 0] = True
    return spans, spray, selective


def _candidate_metrics(sig: np.ndarray, spans: np.ndarray, w_fp: float,
                       n_folds: int = 4) -> tuple[float, float]:
    """(bootstrap LCB over fold scores, aggregate firing_excess)."""
    seg = len(sig) // n_folds
    fold_scores: list[float] = []
    excesses: list[float] = []
    for k in range(n_folds):
        sl = slice(k * seg, (k + 1) * seg)
        stats = match_signals_weighted(sig[sl], spans[sl])
        fold_scores.append(compute_side_score_v5(stats, w_fp=w_fp))
        excesses.append(firing_excess(weighted_precision(stats, w_fp=w_fp),
                                      weighted_recall(stats)))
    lcb = float(fold_scores_bootstrap_ci(fold_scores)[0])
    return lcb, float(np.mean(excesses))


@pytest.mark.parametrize("w_fp", [0.2, 0.5, 1.0])
def test_w_fp_sensitivity_firing_gate_catches_spray(w_fp: float):
    """F3(b)(c): the acceptance gate (firing_excess folded into the penalized
    LCB ranking) must rank SELECTIVE above SPRAY at the LOOSEST setting 0.2
    (parametrized over the whole sweep)."""
    spans, spray, selective = _toy_landscape()
    lcb_spray, exc_spray = _candidate_metrics(spray, spans, w_fp)
    lcb_sel, exc_sel = _candidate_metrics(selective, spans, w_fp)
    assert exc_spray > 0.0, "spray must trip the firing gate"
    assert exc_sel == pytest.approx(0.0), "selective must not trip the gate"
    order = rank_by_penalized_lcb([lcb_spray, lcb_sel],
                                  firing_excesses=[exc_spray, exc_sel],
                                  penalty=0.1)
    assert int(order[0]) == 1, "SELECTIVE must outrank SPRAY through the gate"


def test_w_fp_increases_spray_fp_price():
    spans, spray, _ = _toy_landscape()
    stats = match_signals_weighted(spray, spans)
    assert weighted_precision(stats, w_fp=1.0) < weighted_precision(stats, w_fp=0.5) \
        < weighted_precision(stats, w_fp=0.2)


# ---------------------------------------------------------------------------
# §2.3 — pooled plumbing (mass through StreamStat with cluster weights)
# ---------------------------------------------------------------------------


def _mass_stat(n_signals: int, tp_mass: float, total_mass: float,
               n_unmatched: int, n_bars: int, weight: float) -> StreamStat:
    tp = n_signals - n_unmatched
    return StreamStat(
        n_signals=n_signals, tp=tp, matched_pivots=tp, total_pivots=0,
        n_bars=n_bars, weight=weight, tp_mass=tp_mass, total_mass=total_mass,
        n_unmatched=n_unmatched,
    )


def test_pooled_mass_aggregation_with_cluster_weights():
    a = _mass_stat(n_signals=2, tp_mass=1.0, total_mass=2.0,
                   n_unmatched=1, n_bars=1000, weight=1.0)
    b = _mass_stat(n_signals=4, tp_mass=2.0, total_mass=4.0,
                   n_unmatched=2, n_bars=1000, weight=0.5)
    score, comp = pooled_side_score([a, b], "low")
    # pooled tp = 1*1 + 0.5*2 = 2 ; total = 1*2 + 0.5*4 = 4 ; unm = 1 + 1 = 2
    assert comp["precision"] == pytest.approx(2.0 / (2.0 + 2 * W_FP))
    assert comp["recall"] == pytest.approx(0.5)
    assert comp["n_eff"] == pytest.approx(4.0)
    assert 0.0 <= score <= 1.0


def test_pooled_fold_score_form_and_v5_component_keys():
    is_stats = [_mass_stat(4, 4.0, 4.0, 0, 2000, 1.0)]    # perfect IS
    oos_stats = [_mass_stat(4, 1.0, 4.0, 3, 2000, 1.0)]   # weaker OOS
    fold, comp = pooled_fold_score(is_stats, oos_stats, "low")
    from src.scoring import GAMMA
    gap = max(0.0, comp["is_score"] - comp["oos_score"])
    assert fold == pytest.approx(comp["oos_score"] * np.exp(-GAMMA * gap))
    for key in ("tp_mass_is", "tp_mass_oos", "total_mass_is", "total_mass_oos",
                "n_eff_oos", "pooled_total_mass_oos"):
        assert key in comp, f"missing v5 component key {key}"
    # T4: the acceptance path reads precision_oos / recall_oos — these must
    # carry the WEIGHTED quantities.
    assert comp["precision_oos"] == pytest.approx(1.0 / (1.0 + 3 * W_FP))
    assert comp["recall_oos"] == pytest.approx(0.25)


def test_zero_mass_fold_scores_zero_without_crash():
    z = _mass_stat(0, 0.0, 0.0, 0, 0, 1.0)
    fold, comp = pooled_fold_score([z], [z], "low")
    assert fold == 0.0
    assert _fold_is_informative(comp) is False


# ---------------------------------------------------------------------------
# §2.4 — informative-fold filter on MASS
# ---------------------------------------------------------------------------


def test_fold_is_informative_on_mass():
    assert _fold_is_informative({"pooled_total_mass_oos": 0.3}) is True
    assert _fold_is_informative({"pooled_total_mass_oos": 0.0}) is False
    assert _fold_is_informative({}) is False


def test_span30_only_fold_is_informative_under_v5():
    """§2.4 behavior change, encoded explicitly: a fold whose OOS has only
    sub-100 span pivots (zero pivot_N100 labels) IS informative under v5."""
    df = _v_df(300, 150, extra_lows={181: -5.0})
    df = add_pivot_labels(df)
    # the old count check would have discarded this fold:
    assert int((df[f"pivot_N{REFERENCE_N}"] != 0).sum()) == 0
    assert int(df["pivot_span_low"].iloc[150]) == 30
    sig = pd.Series(False, index=df.index)
    sig.iloc[150] = True
    st = _stream_stat(df, sig, "low", weight=1.0)
    assert st.total_mass > 0.0
    fold, comp = pooled_fold_score([st], [st], "low")
    assert _fold_is_informative(comp) is True


# ---------------------------------------------------------------------------
# §1.2 retention + §3.2 left-span (T1 deliverables)
# ---------------------------------------------------------------------------


def test_add_pivot_labels_emits_pivot_n100_and_span_columns():
    df = _v_df(1000, 500)
    df = add_pivot_labels(df)
    assert f"pivot_N{REFERENCE_N}" in df.columns      # retained (§1.2)
    for col in ("pivot_span_high", "pivot_span_low",
                "pivot_span_censored_high", "pivot_span_censored_low"):
        assert col in df.columns, f"missing v5 column {col}"


def test_compute_left_span_basic_clamps():
    arr = np.full(100, 10.0)
    arr[60] = 1.0          # candidate bar
    arr[29] = 0.5          # strictly lower low 31 bars back
    assert compute_left_span(arr, 60, is_high=False) == 30
    # left-edge clamp
    assert compute_left_span(np.array([5.0, 6.0, 7.0, 1.0]), 3, is_high=False) == 3
    # grid-cap clamp at 500
    long_arr = np.full(700, 10.0)
    long_arr[650] = 1.0
    assert compute_left_span(long_arr, 650, is_high=False) == 500
    # HIGH side mirrors with strictly higher highs
    h = np.full(100, 10.0)
    h[60] = 20.0
    h[39] = 25.0
    assert compute_left_span(h, 60, is_high=True) == 20
