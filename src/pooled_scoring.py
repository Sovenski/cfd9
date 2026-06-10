"""Cluster-weighted pooled scoring for multi-asset folds — Scorer v5.

Pools per-stream SPAN-MASS match statistics (``match_signals_weighted``,
spec §2.1-§2.2: ±1 direct-hit Hungarian matching against span-weighted
pivots) weighted by ``1/cluster_size``, then runs the SAME composite side-
score math as the per-slice scorer (``scoring_v5.compute_side_score_v5``)
on the pooled masses. The cluster weighting is what prevents correlated
streams (e.g. SPX-1D + SPX-1W, or SPX+NDX) from over-crediting the
objective; the mass weighting is what prevents the optimizer from farming
small swings (spec §1.3).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .scoring import GAMMA
from .scoring_v5 import WeightedStats, compute_side_score_v5


@dataclass(frozen=True)
class StreamStat:
    """Per-stream match statistics plus the stream's pool weight.

    Scorer v5 fields (spec §2.3): ``tp_mass`` / ``total_mass`` /
    ``n_unmatched`` carry the span-mass quantities the pooled score is
    computed from. The v4 count fields (``tp``, ``matched_pivots``,
    ``total_pivots``) are retained as diagnostics (1:1 matching makes
    ``tp == matched_pivots``; ``total_pivots`` counts grid-span events).
    """

    n_signals: int
    tp: int
    matched_pivots: int
    total_pivots: int
    n_bars: int
    weight: float
    tp_mass: float = 0.0
    total_mass: float = 0.0
    n_unmatched: int = 0


def pooled_side_score(
    stats: list[StreamStat], side: str,
) -> tuple[float, dict[str, float]]:
    """Weighted-pooled v5 side score in [0, 1].

    Sums span mass with cluster weights exactly as v4 summed counts, then
    delegates to ``compute_side_score_v5`` so the pooled path and the
    per-slice path share one composite formula (spec §2.2-§2.3).
    """
    if side not in ("high", "low"):
        raise ValueError(f"side must be 'high' or 'low', got {side!r}")

    pooled = WeightedStats(
        tp_mass=float(sum(s.weight * s.tp_mass for s in stats)),
        total_mass=float(sum(s.weight * s.total_mass for s in stats)),
        n_signals=float(sum(s.weight * s.n_signals for s in stats)),
        n_unmatched=float(sum(s.weight * s.n_unmatched for s in stats)),
        n_bars=float(sum(s.weight * s.n_bars for s in stats)),
    )
    w_total_pivots = float(sum(s.weight * s.total_pivots for s in stats))

    score, comp = compute_side_score_v5(pooled, return_components=True)
    comp["pooled_n_signals"] = pooled.n_signals
    comp["pooled_total_pivots"] = w_total_pivots   # diagnostic count
    return score, comp


def pooled_fold_score(
    is_stats: list[StreamStat],
    oos_stats: list[StreamStat],
    side: str,
) -> tuple[float, dict[str, float]]:
    """Pooled per-fold score with the smooth IS-OOS overfit penalty.

    ``fold = oos_score * exp(-GAMMA * max(0, is_score - oos_score))`` — same
    contract as ``scoring._fold_score`` but on pooled, cluster-weighted span
    masses. ``precision_oos`` / ``recall_oos`` carry the WEIGHTED v5
    quantities (consumed by ``v17_acceptance.firing_excess``, spec §2.2).
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
        "tp_mass_is": is_comp["tp_mass"],
        "tp_mass_oos": oos_comp["tp_mass"],
        "total_mass_is": is_comp["total_mass"],
        "total_mass_oos": oos_comp["total_mass"],
        "n_eff_oos": oos_comp["n_eff"],
        "pooled_total_mass_oos": oos_comp["total_mass"],
        "pooled_total_pivots_oos": oos_comp["pooled_total_pivots"],
    }
    return fold, components
