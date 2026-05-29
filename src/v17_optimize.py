"""v17 — cyclic coordinate-ascent over detector thresholds (real-objective).

This is the refinement stage of the Calibrated Coordinate-Ascent design
(plan/v17-design.md). It optimizes the continuous threshold params one axis at
a time by an EXACT 1-D line search over a candidate grid, scoring every
candidate with the SAME pooled objective v16 uses (block-bootstrap LCB over
informative folds). The detector is called unchanged -> parity preserved.

Seeding: the ``seed`` Params may come from the v16 best, the gold default, or
(preferably) the label-calibrated cutpoints from ``v17_calibrate``. Coordinate
ascent then corrects vote interactions and sequential-state effects that the
per-vote calibration cannot see.
"""

from __future__ import annotations

import dataclasses
import logging
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from .indicators import Params
from .pooled_validation import (
    Fold,
    _fold_is_informative,
    cluster_weights,
    evaluate_pooled_fold,
)
from .scoring import REFERENCE_N  # noqa: F401  (kept for parity of imports)
from .search_space import space_for
from .universe import Stream
from .validation import fold_scores_bootstrap_ci

logger = logging.getLogger(__name__)

# Continuous threshold params that are pure gate/shape (always relevant).
_ALWAYS_FIELDS = ["min_agreement", "dur_extreme_pct", "scale_div_thresh", "pct_extreme"]
# Continuous threshold params gated by a vote switch.
_VOTE_FIELDS = {
    "slope_thresh": "use_trend",
    "vol_surge_thresh": "use_volume",
    "vola_high_pct": "use_volatility",
    "gjr_vote_thresh": "use_gjr_asym",
    "har_vote_thresh": "use_har_vol",
    "momentum_velocity_thresh": "use_momentum_velocity",
}
# Pivot-drift vote is always active in the detector (_USE_PIVOT_DRIFT=True).
_DRIFT_FIELDS = ["pivot_drift_thresh", "pivot_drift_gate_mult"]


@dataclass
class PooledScorer:
    """Score an arbitrary ``Params`` with v16's pooled block-bootstrap LCB.

    Mirrors ``build_pooled_optuna_objective`` exactly (informative-fold filter +
    block bootstrap), but takes a concrete ``Params`` instead of an Optuna trial.
    """

    folds: list[Fold]
    streams: list[Stream]
    side: str
    n_boot: int = 1000
    alpha: float = 0.10
    block_len: int = 2
    _weights: dict = field(init=False)

    def __post_init__(self) -> None:
        if self.side not in ("high", "low"):
            raise ValueError(f"side must be 'high'|'low', got {self.side!r}")
        self._weights = cluster_weights(self.streams)

    def score(self, params: Params) -> float:
        fold_scores: list[float] = []
        for fold in self.folds:
            s, comp = evaluate_pooled_fold(params, self.side, fold, self._weights)
            if _fold_is_informative(comp):
                fold_scores.append(s)
        if not fold_scores:
            return 0.0
        return float(fold_scores_bootstrap_ci(
            fold_scores, n_boot=self.n_boot, alpha=self.alpha, block_len=self.block_len
        )[0])


def active_threshold_fields(params: Params, side: str) -> list[str]:
    """Suffixed field names worth line-searching for this params/side."""
    fields = list(_ALWAYS_FIELDS) + list(_DRIFT_FIELDS)
    for base, use in _VOTE_FIELDS.items():
        if bool(getattr(params, f"{use}_{side}")):
            fields.append(base)
    return [f"{b}_{side}" for b in fields]


def _candidate_grid(base: str, side: str, current: float, grid_n: int) -> np.ndarray:
    """Candidate values for one float field: bounds-spanning grid + current."""
    space = space_for(side)
    lo, hi = space.float_bounds[base]
    grid = np.linspace(float(lo), float(hi), max(2, grid_n))
    return np.unique(np.append(grid, float(current)))


@dataclass
class AscentResult:
    params: Params
    score: float
    seed_score: float
    n_evals: int
    history: list[tuple[str, float, float]]  # (field, chosen_value, score_after)


def coordinate_ascent(
    seed: Params,
    scorer: PooledScorer,
    side: str,
    grid_n: int = 7,
    max_sweeps: int = 3,
    eps: float = 1e-6,
    progress: Optional[Callable[[str], None]] = None,
) -> AscentResult:
    """Optimize continuous thresholds by exact per-axis line search.

    Each sweep line-searches every active threshold over its candidate grid,
    keeping the value that maximizes the pooled LCB. Stops when a full sweep
    yields no improvement or ``max_sweeps`` is reached. Never regresses
    (current value is always in the grid).
    """
    log = progress or (lambda m: logger.info(m))
    cur = seed
    cur_score = scorer.score(cur)
    seed_score = cur_score
    n_evals = 1
    history: list[tuple[str, float, float]] = []
    fields = active_threshold_fields(cur, side)
    log(f"[v17:{side}] seed LCB={cur_score:.5f}; ascending {len(fields)} thresholds")

    for sweep in range(max_sweeps):
        improved = False
        for fld in fields:
            base = fld[: -(len(side) + 1)]  # strip "_high"/"_low"
            cands = _candidate_grid(base, side, getattr(cur, fld), grid_n)
            best_val = getattr(cur, fld)
            best_sc = cur_score
            for c in cands:
                if float(c) == float(getattr(cur, fld)):
                    continue
                trial_params = dataclasses.replace(cur, **{fld: float(c)})
                sc = scorer.score(trial_params)
                n_evals += 1
                if sc > best_sc + eps:
                    best_sc, best_val = sc, float(c)
            if best_val != getattr(cur, fld):
                cur = dataclasses.replace(cur, **{fld: best_val})
                cur_score = best_sc
                improved = True
                history.append((fld, best_val, cur_score))
                log(f"[v17:{side}] sweep {sweep} {fld} -> {best_val:.5g}  LCB={cur_score:.5f}")
        if not improved:
            log(f"[v17:{side}] converged after sweep {sweep} ({n_evals} evals)")
            break
    return AscentResult(cur, cur_score, seed_score, n_evals, history)


__all__ = [
    "PooledScorer", "AscentResult", "coordinate_ascent",
    "active_threshold_fields",
]
