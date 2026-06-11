"""Per-side Optuna search bounds (Run 5 Part 1 — bounds decoupling).

Canonical home for the optimizer's search ranges, previously scattered as
module-level literals in ``speculatores145.py``. ``LOW_SPACE`` reproduces the
historical (side-symmetric) bounds exactly; ``HIGH_SPACE`` relaxes two floors so
the optimizer can express a market-top ("dim, diffuse agreement") detector.

Optuna trial-key NAMES are preserved exactly via ``trial_key_overrides`` so
existing journals and TPE warm-start continue to parse — notably the historical
``{side}_pivot_drift_lb`` key for the ``pivot_drift_lookback`` Params field.
"""
from __future__ import annotations

import types
from dataclasses import dataclass, field
from typing import Mapping

# --- Base (== LOW == historical) bounds -----------------------------------
INT_BOUNDS: dict[str, tuple[int, int]] = {
    "S_detect": (5, 60),
    "scale_start": (2, 30),
    "scale_end": (100, 500),
    "scale_step": (2, 20),
    "min_duration": (1, 20),
    "cooldown_bars": (1, 20),
    "price_gate_lb": (5, 100),
    "vola_range_len": (20, 200),
    "er_period": (5, 60),
    "confirm_count": (1, 5),
    "pivot_drift_lookback": (2, 20),
    "pivot_drift_confirm_bias": (0, 2),
    "edge_window": (3, 60),
}

FLOAT_BOUNDS: dict[str, tuple[float, float]] = {
    "pct_extreme": (0.70, 0.99),
    "min_agreement": (0.10, 0.90),
    "dur_extreme_pct": (0.50, 0.99),
    "vol_surge_thresh": (1.0, 3.0),
    "scale_div_thresh": (0.10, 0.60),
    "slope_thresh": (0.01, 0.50),
    "vola_high_pct": (0.50, 0.99),
    "pivot_drift_thresh": (0.001, 0.050),
    "pivot_drift_gate_mult": (1.0, 10.0),
    "momentum_velocity_thresh": (0.0, 0.05),
    "gjr_vote_thresh": (0.05, 0.50),
    "har_vote_thresh": (0.05, 0.50),
}

BOOL_FIELDS: list[str] = [
    "er_directional",
    "use_trend",
    "use_volume",
    "use_momentum",
    "use_momentum_velocity",
    "use_volatility",
    "use_er_gate",
    "use_gjr_asym",
    "use_har_vol",
    "use_edge_voting",
]

CATEGORY_FIELDS: dict[str, list[str]] = {
    "vola_method": ["ATR", "StdDev", "Intraday"],
    "momentum_velocity_mode": ["Trend", "Reversal"],
}

# Historical Optuna trial-key stems that differ from the field name.
_TRIAL_KEY_OVERRIDES: dict[str, str] = {
    "pivot_drift_lookback": "pivot_drift_lb",
}


@dataclass(frozen=True)
class SearchSpace:
    """Immutable per-side search bounds."""

    int_bounds: Mapping[str, tuple[int, int]]
    float_bounds: Mapping[str, tuple[float, float]]
    bool_fields: tuple[str, ...]
    category_fields: Mapping[str, tuple[str, ...]]
    trial_key_overrides: Mapping[str, str] = field(default_factory=dict)

    def trial_key(self, field_name: str) -> str:
        """Map a Params field stem to its historical Optuna trial-key stem."""
        return self.trial_key_overrides.get(field_name, field_name)


def _freeze_categories(cats: dict[str, list[str]]) -> dict[str, tuple[str, ...]]:
    return {k: tuple(v) for k, v in cats.items()}


# LOW: widen the four big-run boundary pins (v17gpu_20260610_225518), each in
# its pinned direction (2026-06-11 aggressive widen). The v5.1 spray pricing
# (W_FP 0.5, firing cap 1.0) is the counterweight that lets the optimizer
# explore the looser corner without being rewarded for overfiring.
# FLOAT_BOUNDS stays the historical reference; every non-pinned bound is reused.
_LOW_FLOAT_BOUNDS = dict(FLOAT_BOUNDS)
_LOW_FLOAT_BOUNDS["pct_extreme"] = (0.40, 0.99)          # pin lo 0.70 -> 0.40
_LOW_FLOAT_BOUNDS["min_agreement"] = (0.02, 0.90)        # pin lo 0.10 -> 0.02
_LOW_FLOAT_BOUNDS["scale_div_thresh"] = (0.10, 0.95)     # pin hi 0.60 -> 0.95
_LOW_FLOAT_BOUNDS["pivot_drift_thresh"] = (0.0001, 0.050)  # pin lo 0.001 -> 0.0001

LOW_SPACE = SearchSpace(
    int_bounds=types.MappingProxyType(dict(INT_BOUNDS)),
    float_bounds=types.MappingProxyType(_LOW_FLOAT_BOUNDS),
    bool_fields=tuple(BOOL_FIELDS),
    category_fields=types.MappingProxyType(_freeze_categories(CATEGORY_FIELDS)),
    trial_key_overrides=types.MappingProxyType(dict(_TRIAL_KEY_OVERRIDES)),
)

# HIGH: relax its two pinned floors further off the historical base (the
# big-run HIGH winner sat exactly on both). Non-pinned bounds reuse the base.
_HIGH_FLOAT_BOUNDS = dict(FLOAT_BOUNDS)
_HIGH_FLOAT_BOUNDS["dur_extreme_pct"] = (0.10, 0.99)     # pin lo 0.30 -> 0.10
_HIGH_FLOAT_BOUNDS["pct_extreme"] = (0.30, 0.99)         # pin lo 0.55 -> 0.30

HIGH_SPACE = SearchSpace(
    int_bounds=types.MappingProxyType(dict(INT_BOUNDS)),
    float_bounds=types.MappingProxyType(_HIGH_FLOAT_BOUNDS),
    bool_fields=tuple(BOOL_FIELDS),
    category_fields=types.MappingProxyType(_freeze_categories(CATEGORY_FIELDS)),
    trial_key_overrides=types.MappingProxyType(dict(_TRIAL_KEY_OVERRIDES)),
)


def space_for(side: str) -> SearchSpace:
    """Return the SearchSpace for a side ('high' | 'low')."""
    if side == "high":
        return HIGH_SPACE
    if side == "low":
        return LOW_SPACE
    raise ValueError(f"side must be 'high' or 'low', got {side!r}")
