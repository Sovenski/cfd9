"""Speculatores Pivot Optimizer — public API."""

from .detector import SpeculatorDetector
from .scoring import (
    PIVOT_SCALES,
    label_pivots,
    add_pivot_labels,
    precision_at_n,
    compute_side_score,
    fold_score_high,
    fold_score_low,
)
from .validation import (
    temporal_split,
    walk_forward_folds,
    load_data,
    load_cross_asset,
    build_optuna_objective,
    FOLD_DEFINITIONS,
    EMBARGO_BARS,
)

__all__ = [
    "SpeculatorDetector",
    "PIVOT_SCALES",
    "label_pivots",
    "add_pivot_labels",
    "precision_at_n",
    "compute_side_score",
    "fold_score_high",
    "fold_score_low",
    "temporal_split",
    "walk_forward_folds",
    "load_data",
    "load_cross_asset",
    "build_optuna_objective",
    "FOLD_DEFINITIONS",
    "EMBARGO_BARS",
]
