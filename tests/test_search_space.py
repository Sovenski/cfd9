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


def test_low_space_reproduces_today_bounds_exactly():
    # LOW must be byte-identical to the historical (symmetric) bounds.
    assert LOW_SPACE.int_bounds == INT_BOUNDS
    assert LOW_SPACE.float_bounds == FLOAT_BOUNDS
    assert LOW_SPACE.bool_fields == tuple(BOOL_FIELDS)
    assert LOW_SPACE.category_fields == {
        k: tuple(v) for k, v in CATEGORY_FIELDS.items()
    }


def test_high_space_differs_only_in_two_floors():
    # Ints identical.
    assert HIGH_SPACE.int_bounds == LOW_SPACE.int_bounds
    # Exactly the two relaxed floors differ; everything else identical.
    differing = {
        k for k in FLOAT_BOUNDS
        if HIGH_SPACE.float_bounds[k] != LOW_SPACE.float_bounds[k]
    }
    assert differing == {"dur_extreme_pct", "pct_extreme"}
    assert HIGH_SPACE.float_bounds["dur_extreme_pct"] == (0.30, 0.99)
    assert HIGH_SPACE.float_bounds["pct_extreme"] == (0.55, 0.99)


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
