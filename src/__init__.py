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

__all__ = [
    "SpeculatorDetector",
    "PIVOT_SCALES",
    "label_pivots",
    "add_pivot_labels",
    "precision_at_n",
    "compute_side_score",
    "fold_score_high",
    "fold_score_low",
]
