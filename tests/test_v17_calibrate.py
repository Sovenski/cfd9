"""Unit tests for the v17 exact cutpoint calibration core."""

from __future__ import annotations

import numpy as np

from src.v17_calibrate import (
    Cutpoint,
    best_cutpoint,
    cv_select_cutpoint,
    purged_time_folds,
)


def _separable(n=2000, cut=1.5, noise=0.0, seed=0):
    rng = np.random.default_rng(seed)
    feature = rng.normal(0.0, 1.0, n)
    pos = feature >= cut
    if noise:  # flip a fraction of labels
        flip = rng.random(n) < noise
        pos = np.where(flip, ~pos, pos)
    return feature, pos


def test_geq_recovers_known_threshold():
    feature, pos = _separable(cut=1.5)
    cp = best_cutpoint(feature, pos, "geq", objective="fbeta", beta=1.0)
    assert isinstance(cp, Cutpoint)
    assert abs(cp.threshold - 1.5) < 0.15          # near the true cut
    assert cp.precision > 0.95 and cp.recall > 0.95  # clean separation
    assert cp.n_predicted > 0


def test_leq_direction_mirrors():
    # positives are the LOW tail: feature <= -1.0
    rng = np.random.default_rng(1)
    feature = rng.normal(0, 1, 3000)
    pos = feature <= -1.0
    cp = best_cutpoint(feature, pos, "leq", objective="fbeta", beta=1.0)
    assert abs(cp.threshold - (-1.0)) < 0.15
    assert cp.precision > 0.9 and cp.recall > 0.9


def test_nan_features_are_never_predicted_positive():
    feature, pos = _separable(cut=1.0)
    feature[:100] = np.nan
    pos[:100] = True            # NaN positives must hurt recall, not become tp
    cp = best_cutpoint(feature, pos, "geq", objective="fbeta", beta=1.0)
    assert np.isfinite(cp.threshold)
    assert cp.recall < 1.0      # the 100 NaN positives can't be recovered


def test_no_positives_returns_nan():
    feature = np.linspace(0, 1, 100)
    pos = np.zeros(100, dtype=bool)
    cp = best_cutpoint(feature, pos, "geq")
    assert np.isnan(cp.threshold) and cp.score == 0.0


def test_min_predicted_filters_degenerate():
    feature, pos = _separable(cut=2.5)   # very few positives
    cp = best_cutpoint(feature, pos, "geq", min_predicted=50)
    # threshold must fire on >=50 bars
    assert (feature >= cp.threshold).sum() >= 50


def test_purged_folds_shapes_and_embargo():
    folds = purged_time_folds(1000, k=5, embargo=10)
    assert len(folds) == 5
    for train, test in folds:
        assert test.size > 0 and train.size > 0
        # embargo gap: no train index within `embargo` of the test block
        assert train.min() < test.min() or train.max() > test.max()
        gap = np.intersect1d(train, np.arange(test.min() - 10, test.max() + 10 + 1))
        assert gap.size == 0


def test_cv_select_is_robust_to_label_noise():
    # clean cut at 1.0 but 10% label noise; CV cut should stay near 1.0
    feature, pos = _separable(cut=1.0, noise=0.10, seed=3)
    cp = cv_select_cutpoint(feature, pos, "geq", objective="fbeta", beta=1.0,
                            k=5, embargo=5, n_grid=128)
    assert np.isfinite(cp.threshold)
    assert abs(cp.threshold - 1.0) < 0.4
    assert cp.n_predicted > 0
