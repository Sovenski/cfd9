"""Speculatores Pivot Optimizer — public API."""

from .indicators import Params
from .detector import DetectorArtifacts, SpeculatorDetector, build_detector_artifacts
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
    ValidationScheme,
    infer_validation_scheme,
    describe_validation_scheme,
    EMBARGO_BARS,
    MIN_SIGNALS_PER_FOLD,
)
from .speculatores145 import RunConfig, params_from_trial, run_full_pipeline

__all__ = [
    "Params",
    "DetectorArtifacts",
    "SpeculatorDetector",
    "build_detector_artifacts",
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
    "ValidationScheme",
    "infer_validation_scheme",
    "describe_validation_scheme",
    "EMBARGO_BARS",
    "MIN_SIGNALS_PER_FOLD",
    "RunConfig",
    "params_from_trial",
    "run_full_pipeline",
]
