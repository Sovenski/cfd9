"""Cluster-weighted pooled scoring for multi-asset folds.

Pools the per-stream match counts from ``precision_at_n_stats`` (at the single
v4 tolerance scale ``REFERENCE_N``) weighted by ``1/cluster_size``, then runs
the SAME side-score math as ``compute_side_score`` on the pooled counts. The
weighting is what prevents correlated streams (e.g. SPX-1D + SPX-1W, or
SPX+NDX) from over-crediting the objective.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .scoring import (
    GAMMA, MIN_RATE, PRECISION_EXPONENT, RECALL_TARGET, REFERENCE_N,
)


@dataclass(frozen=True)
class StreamStat:
    """Per-stream match counts at REFERENCE_N, plus the stream's pool weight."""
    n_signals: int
    tp: int
    matched_pivots: int
    total_pivots: int
    n_bars: int
    weight: float


def pooled_side_score(
    stats: list[StreamStat], side: str,
) -> tuple[float, dict[str, float]]:
    """Weighted-pooled single-scale side score in [0, 1].

    Mirrors ``compute_side_score`` for the v4 single-scale (REFERENCE_N) case:
    ``precision**PRECISION_EXPONENT * recall_sat * frequency_factor *
    excess_penalty``.
    """
    if side not in ("high", "low"):
        raise ValueError(f"side must be 'high' or 'low', got {side!r}")

    w_nsig = sum(s.weight * s.n_signals for s in stats)
    w_tp = sum(s.weight * s.tp for s in stats)
    w_matched = sum(s.weight * s.matched_pivots for s in stats)
    w_total = sum(s.weight * s.total_pivots for s in stats)
    w_bars = sum(s.weight * s.n_bars for s in stats)

    empty = {
        "precision": 0.0, "recall": 0.0, "recall_saturated": 0.0,
        "frequency_factor": 0.0, "excess_penalty": 0.0,
        "pooled_n_signals": float(w_nsig), "pooled_total_pivots": float(w_total),
    }
    if w_nsig <= 0 or w_bars <= 0:
        return 0.0, empty

    precision = w_tp / w_nsig
    recall = (w_matched / w_total) if w_total > 0 else 0.0
    # Single scale → target_for_scale = RECALL_TARGET * sqrt(REFERENCE_N/REFERENCE_N).
    recall_sat = 1.0 - np.exp(-recall / max(RECALL_TARGET, 1e-9)) if precision > 0 else 0.0
    scale_score = (precision ** PRECISION_EXPONENT) * recall_sat

    signal_rate = w_nsig / w_bars
    frequency_factor = min(1.0, signal_rate / MIN_RATE)

    n_eff = max(float(w_nsig), 1.0)
    t_eff = max(float(w_total), 1.0)
    excess_penalty = 2.0 * n_eff * t_eff / (n_eff * n_eff + t_eff * t_eff)

    final = float(scale_score * frequency_factor * excess_penalty)
    comp = {
        "precision": float(precision), "recall": float(recall),
        "recall_saturated": float(recall_sat),
        "frequency_factor": float(frequency_factor),
        "excess_penalty": float(excess_penalty),
        "pooled_n_signals": float(w_nsig), "pooled_total_pivots": float(w_total),
    }
    return final, comp


def pooled_fold_score(
    is_stats: list[StreamStat],
    oos_stats: list[StreamStat],
    side: str,
) -> tuple[float, dict[str, float]]:
    """Pooled per-fold score with the smooth IS-OOS overfit penalty.

    ``fold = oos_score * exp(-GAMMA * max(0, is_score - oos_score))`` — same
    contract as ``scoring._fold_score`` but on pooled, cluster-weighted counts.
    """
    is_score, is_comp = pooled_side_score(is_stats, side)
    oos_score, oos_comp = pooled_side_score(oos_stats, side)
    gap = max(0.0, float(is_score) - float(oos_score))
    fold = float(oos_score * np.exp(-GAMMA * gap))
    components = {
        "is_score": float(is_score),
        "oos_score": float(oos_score),
        "is_oos_gap": float(gap),
        "fold_score": fold,
        "precision_oos": oos_comp["precision"],
        "recall_oos": oos_comp["recall"],
        "excess_penalty_oos": oos_comp["excess_penalty"],
        "frequency_factor_oos": oos_comp["frequency_factor"],
        "pooled_total_pivots_oos": oos_comp["pooled_total_pivots"],
    }
    return fold, components
