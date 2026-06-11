"""Unit tests for the SearchSpace decoupling (Run 5 Part 1)."""
from __future__ import annotations

from src.search_space import (
    INT_BOUNDS,
    FLOAT_BOUNDS,
    BOOL_FIELDS,
    CATEGORY_FIELDS,
    HIGH_SPACE,
    LOW_SPACE,
    SearchSpace,
)


#: 2026-06-11 aggressive widen — the big-run (v17gpu_20260610_225518) boundary
#: pins, each extended in its pinned direction (v5.1 spray pricing is the
#: counterweight). LOW relaxes four; HIGH relaxes its two floors further.
_LOW_WIDENED = {
    "pct_extreme": (0.40, 0.99),        # pin lo 0.70 -> 0.40
    "min_agreement": (0.02, 0.90),      # pin lo 0.10 -> 0.02
    "scale_div_thresh": (0.10, 0.95),   # pin hi 0.60 -> 0.95
    "pivot_drift_thresh": (0.0001, 0.050),  # pin lo 0.001 -> 0.0001
}
_HIGH_WIDENED = {
    "dur_extreme_pct": (0.10, 0.99),    # pin lo 0.30 -> 0.10
    "pct_extreme": (0.30, 0.99),        # pin lo 0.55 -> 0.30
}


def test_low_space_widens_only_the_four_big_run_pins():
    assert LOW_SPACE.int_bounds == INT_BOUNDS
    assert LOW_SPACE.bool_fields == tuple(BOOL_FIELDS)
    assert LOW_SPACE.category_fields == {
        k: tuple(v) for k, v in CATEGORY_FIELDS.items()
    }
    for k, bound in _LOW_WIDENED.items():
        assert LOW_SPACE.float_bounds[k] == bound, k
    # every other LOW float bound is the historical base, untouched
    for k in FLOAT_BOUNDS:
        if k not in _LOW_WIDENED:
            assert LOW_SPACE.float_bounds[k] == FLOAT_BOUNDS[k], k


def test_high_space_widens_only_its_two_floors():
    assert HIGH_SPACE.int_bounds == LOW_SPACE.int_bounds
    for k, bound in _HIGH_WIDENED.items():
        assert HIGH_SPACE.float_bounds[k] == bound, k
    # every other HIGH float bound is the historical base (HIGH did not pin them)
    for k in FLOAT_BOUNDS:
        if k not in _HIGH_WIDENED:
            assert HIGH_SPACE.float_bounds[k] == FLOAT_BOUNDS[k], k


def test_categoricals_frozen_identical_between_sides():
    assert HIGH_SPACE.category_fields == LOW_SPACE.category_fields
    assert HIGH_SPACE.bool_fields == LOW_SPACE.bool_fields


def test_pivot_drift_trial_key_override_preserved():
    # Optuna journal key must stay 'pivot_drift_lb', not the Params field name.
    assert HIGH_SPACE.trial_key("pivot_drift_lookback") == "pivot_drift_lb"
    assert LOW_SPACE.trial_key("pivot_drift_lookback") == "pivot_drift_lb"
    # Non-overridden fields map to themselves.
    assert HIGH_SPACE.trial_key("S_detect") == "S_detect"


def test_search_space_is_frozen():
    import dataclasses
    assert dataclasses.is_dataclass(SearchSpace)
    space = SearchSpace(int_bounds={}, float_bounds={}, bool_fields=(),
                        category_fields={}, trial_key_overrides={})
    try:
        space.int_bounds = {"x": (1, 2)}  # type: ignore[misc]
        raised = False
    except dataclasses.FrozenInstanceError:
        raised = True
    assert raised


import optuna


def _trial_distributions(side: str):
    """Run one no-op trial and return the recorded Optuna distributions."""
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    from src.speculatores145 import params_from_trial

    def _objective(trial):
        params_from_trial(trial, side)
        return 0.0

    study = optuna.create_study()
    study.optimize(_objective, n_trials=1)
    return study.trials[0].distributions


def test_high_floor_relaxed_in_actual_sampling():
    dists = _trial_distributions("high")
    assert dists["high_dur_extreme_pct"].low == 0.10
    assert dists["high_pct_extreme"].low == 0.30


def test_low_floor_widened_in_actual_sampling():
    dists = _trial_distributions("low")
    assert dists["low_dur_extreme_pct"].low == 0.50      # NOT pinned -> unchanged
    assert dists["low_pct_extreme"].low == 0.40
    assert dists["low_min_agreement"].low == 0.02
    assert dists["low_scale_div_thresh"].high == 0.95
    assert dists["low_pivot_drift_thresh"].low == 0.0001


def test_trial_keys_unchanged_for_journal_compatibility():
    dists = _trial_distributions("high")
    # The historical short key must still be emitted (TPE warm-start).
    assert "high_pivot_drift_lb" in dists
    assert "high_pivot_drift_lookback" not in dists
    # Spot-check a few other historical keys still exist.
    for key in ("high_S_detect", "high_min_agreement", "high_use_volume",
                "high_vola_method", "high_edge_window"):
        assert key in dists


import random


def test_global_sampler_respects_high_relaxed_floor():
    from src.speculatores145 import _sample_global_params

    rng = random.Random(0)
    saw_below_050 = False
    for _ in range(400):
        p = _sample_global_params("high", rng)
        if p.dur_extreme_pct_high < 0.50:
            saw_below_050 = True
            break
    # HIGH floor is 0.30, so values below 0.50 must be reachable.
    assert saw_below_050


def test_global_sampler_keeps_low_floor():
    from src.speculatores145 import _sample_global_params

    rng = random.Random(0)
    for _ in range(400):
        p = _sample_global_params("low", rng)
        # LOW floor stays 0.50 — never below.
        assert p.dur_extreme_pct_low >= 0.50
