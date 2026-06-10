"""Pooled-score composition tests — re-pinned to Scorer v5 span-mass inputs
(spec §9 migration: "RE-PIN (same test, new expected values)")."""
from src.pooled_scoring import StreamStat, pooled_side_score, pooled_fold_score
from src.scoring_v5 import W_FP


def _stat(n_signals, tp_mass, total_mass, n_unmatched, n_bars, weight):
    tp = n_signals - n_unmatched
    return StreamStat(
        n_signals=n_signals, tp=tp, matched_pivots=tp, total_pivots=0,
        n_bars=n_bars, weight=weight,
        tp_mass=tp_mass, total_mass=total_mass, n_unmatched=n_unmatched,
    )


def test_pooled_precision_is_weighted_mass_ratio():
    # Two streams, weight 0.5 each (same cluster of size 2).
    a = _stat(n_signals=10, tp_mass=8.0, total_mass=10.0, n_unmatched=2,
              n_bars=2000, weight=0.5)
    b = _stat(n_signals=10, tp_mass=2.0, total_mass=10.0, n_unmatched=8,
              n_bars=2000, weight=0.5)
    score, comp = pooled_side_score([a, b], "high")
    # Pooled tp_mass = 0.5*8 + 0.5*2 = 5 ; pooled unmatched = 0.5*2 + 0.5*8 = 5
    # precision_w = 5 / (5 + 5*W_FP) = 5 / 6
    assert abs(comp["precision"] - 5.0 / (5.0 + 5.0 * W_FP)) < 1e-9
    # recall_w = 5 / (0.5*10 + 0.5*10) = 0.5
    assert abs(comp["recall"] - 0.5) < 1e-9
    assert 0.0 <= score <= 1.0


def test_cluster_weight_halves_a_duplicate_streams_contribution():
    # One unique stream (weight 1) vs the same numbers duplicated at weight 0.5
    # twice must give identical pooled precision (correlation is neutralised).
    single = pooled_side_score(
        [_stat(10, 6.0, 12.0, 4, 3000, 1.0)], "high")[1]["precision"]
    dup = pooled_side_score(
        [_stat(10, 6.0, 12.0, 4, 3000, 0.5), _stat(10, 6.0, 12.0, 4, 3000, 0.5)],
        "high")[1]["precision"]
    assert abs(single - dup) < 1e-9


def test_pooled_fold_applies_is_oos_exponential_penalty():
    is_stats = [_stat(10, 10.0, 10.0, 0, 2000, 1.0)]   # perfect IS
    oos_stats = [_stat(10, 5.0, 10.0, 5, 2000, 1.0)]   # weaker OOS
    fold, comp = pooled_fold_score(is_stats, oos_stats, "high")
    assert comp["oos_score"] <= comp["is_score"]
    # fold = oos * exp(-GAMMA * max(0, is-oos)) <= oos_score
    assert fold <= comp["oos_score"] + 1e-9
    assert fold >= 0.0
