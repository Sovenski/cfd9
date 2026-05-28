# Run 5 Part 1 — SearchSpace Decoupling Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give the HIGH and LOW optimizer sides independent search bounds, and relax two HIGH-side floors (`dur_extreme_pct` 0.50→0.30, `pct_extreme` 0.70→0.55) so the optimizer can express a top-shaped detector — without adding features or changing `detector.py`.

**Architecture:** Introduce a frozen `SearchSpace` dataclass (new `src/search_space.py`) holding per-side bounds. `params_from_trial` and the stability-probe samplers read bounds from `HIGH_SPACE`/`LOW_SPACE` instead of shared literals. Optuna trial-key NAMES are preserved byte-for-byte (including the `pivot_drift_lb` quirk) so existing journals/TPE warm-start still parse. `LOW_SPACE` reproduces today's bounds exactly; only two HIGH floats change.

**Tech Stack:** Python 3, Optuna, dataclasses, pytest.

**Reference spec:** `docs/superpowers/specs/2026-05-28-run5-multiasset-asymmetric-bounds-design.md` §3.

---

## File Structure

- **Create** `src/search_space.py` (~120 lines) — canonical bounds constants + `SearchSpace` dataclass + `HIGH_SPACE`/`LOW_SPACE`. New single home for what is currently scattered in `speculatores145.py`.
- **Modify** `src/speculatores145.py` — import bounds/fields from `search_space` (remove local duplicates); refactor `params_from_trial` int/float sampling to be space-driven; refactor `_mutate_local_param`, `_sample_local_params`, `_sample_global_params` to take a `SearchSpace`.
- **Create** `tests/__init__.py` (empty) and `tests/test_search_space.py` — unit tests for the dataclass + the refactor's behavioral guarantees.
- **Create** `pytest.ini` — minimal pytest config so `python -m pytest` discovers `tests/`.

Constants currently in `speculatores145.py` to MOVE into `search_space.py`: `INT_BOUNDS` (lines 89–103), `FLOAT_BOUNDS` (105–118), `BOOL_FIELDS` (227–238), `CATEGORY_FIELDS` (240–243).

---

## Task 1: Create `src/search_space.py` with the SearchSpace dataclass

**Files:**
- Create: `src/search_space.py`
- Create: `tests/__init__.py`
- Create: `tests/test_search_space.py`
- Create: `pytest.ini`

- [ ] **Step 1: Create pytest config and tests package**

Create `pytest.ini`:

```ini
[pytest]
testpaths = tests
python_files = test_*.py
addopts = -q
```

Create `tests/__init__.py` (empty file, zero bytes).

- [ ] **Step 2: Write the failing test**

Create `tests/test_search_space.py`:

```python
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
```

- [ ] **Step 3: Run test to verify it fails**

Run: `python -m pytest tests/test_search_space.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.search_space'`.

- [ ] **Step 4: Write minimal implementation**

Create `src/search_space.py`:

```python
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


LOW_SPACE = SearchSpace(
    int_bounds=dict(INT_BOUNDS),
    float_bounds=dict(FLOAT_BOUNDS),
    bool_fields=tuple(BOOL_FIELDS),
    category_fields=_freeze_categories(CATEGORY_FIELDS),
    trial_key_overrides=dict(_TRIAL_KEY_OVERRIDES),
)

# HIGH: identical to LOW except two relaxed floors (spec §3).
_HIGH_FLOAT_BOUNDS = dict(FLOAT_BOUNDS)
_HIGH_FLOAT_BOUNDS["dur_extreme_pct"] = (0.30, 0.99)
_HIGH_FLOAT_BOUNDS["pct_extreme"] = (0.55, 0.99)

HIGH_SPACE = SearchSpace(
    int_bounds=dict(INT_BOUNDS),
    float_bounds=_HIGH_FLOAT_BOUNDS,
    bool_fields=tuple(BOOL_FIELDS),
    category_fields=_freeze_categories(CATEGORY_FIELDS),
    trial_key_overrides=dict(_TRIAL_KEY_OVERRIDES),
)


def space_for(side: str) -> SearchSpace:
    """Return the SearchSpace for a side ('high' | 'low')."""
    if side == "high":
        return HIGH_SPACE
    if side == "low":
        return LOW_SPACE
    raise ValueError(f"side must be 'high' or 'low', got {side!r}")
```

- [ ] **Step 5: Run test to verify it passes**

Run: `python -m pytest tests/test_search_space.py -q`
Expected: PASS (5 passed).

- [ ] **Step 6: Commit**

```bash
git add src/search_space.py tests/__init__.py tests/test_search_space.py pytest.ini
git commit -m "feat(search-space): SearchSpace dataclass with decoupled HIGH/LOW bounds"
```

---

## Task 2: Point `speculatores145.py` constants at `search_space` (no behavior change yet)

**Files:**
- Modify: `src/speculatores145.py:89-103` (INT_BOUNDS), `:105-118` (FLOAT_BOUNDS), `:227-238` (BOOL_FIELDS), `:240-243` (CATEGORY_FIELDS)

- [ ] **Step 1: Replace the four local constant blocks with an import**

Delete the literal definitions of `INT_BOUNDS`, `FLOAT_BOUNDS`, `BOOL_FIELDS`, `CATEGORY_FIELDS` in `src/speculatores145.py` and instead import them near the top of the module's local imports (after the existing `from .parity import ...` line):

```python
from .search_space import (
    INT_BOUNDS,
    FLOAT_BOUNDS,
    BOOL_FIELDS,
    CATEGORY_FIELDS,
    SearchSpace,
    space_for,
)
```

Leave every existing *use* of these names unchanged for now (they still refer to the same objects, so `_mutate_local_param`, `_sample_global_params`, etc. keep working identically).

- [ ] **Step 2: Run the existing search-space tests + import check**

Run: `python -m pytest tests/test_search_space.py -q && python -c "import src.speculatores145"`
Expected: tests PASS and the module imports with no error.

- [ ] **Step 3: Verify the constants are now the imported objects**

Run: `python -c "import src.speculatores145 as m, src.search_space as s; assert m.INT_BOUNDS is s.INT_BOUNDS; assert m.FLOAT_BOUNDS is s.FLOAT_BOUNDS; print('OK')"`
Expected: prints `OK`.

- [ ] **Step 4: Commit**

```bash
git add src/speculatores145.py
git commit -m "refactor(search-space): source bounds constants from src.search_space"
```

---

## Task 3: Make `params_from_trial` space-driven (the actual decoupling)

**Files:**
- Modify: `src/speculatores145.py:382-437` (the int/float `suggest_*` block inside `params_from_trial`)
- Modify: `tests/test_search_space.py` (add behavioral tests)

- [ ] **Step 1: Write the failing behavioral test**

Append to `tests/test_search_space.py`:

```python
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
    assert dists["high_dur_extreme_pct"].low == 0.30
    assert dists["high_pct_extreme"].low == 0.55


def test_low_floor_unchanged_in_actual_sampling():
    dists = _trial_distributions("low")
    assert dists["low_dur_extreme_pct"].low == 0.50
    assert dists["low_pct_extreme"].low == 0.70


def test_trial_keys_unchanged_for_journal_compatibility():
    dists = _trial_distributions("high")
    # The historical short key must still be emitted (TPE warm-start).
    assert "high_pivot_drift_lb" in dists
    assert "high_pivot_drift_lookback" not in dists
    # Spot-check a few other historical keys still exist.
    for key in ("high_S_detect", "high_min_agreement", "high_use_volume",
                "high_vola_method", "high_edge_window"):
        assert key in dists
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_search_space.py::test_high_floor_relaxed_in_actual_sampling -q`
Expected: FAIL — `assert 0.50 == 0.30` (current code samples HIGH with the old 0.50 floor).

- [ ] **Step 3: Refactor the int/float sampling block to be space-driven**

In `src/speculatores145.py`, inside `params_from_trial`, replace the inline-literal int/float `suggest_*` calls (currently lines 388–412) with space-driven helpers. Insert immediately after `s = side` (line 386):

```python
    space = space_for(side)

    def _gi(name: str) -> int:
        low, high = space.int_bounds[name]
        return trial.suggest_int(f"{s}_{space.trial_key(name)}", low, high)

    def _gf(name: str) -> float:
        low, high = space.float_bounds[name]
        return trial.suggest_float(f"{s}_{space.trial_key(name)}", low, high)

    S_detect = _gi("S_detect")
    scale_start = _gi("scale_start")
    scale_end = _gi("scale_end")
    scale_step = _gi("scale_step")
    min_duration = _gi("min_duration")
    cooldown_bars = _gi("cooldown_bars")
    price_gate_lb = _gi("price_gate_lb")
    vola_range_len = _gi("vola_range_len")
    er_period = _gi("er_period")
    confirm_count = _gi("confirm_count")
    pivot_drift_lb = _gi("pivot_drift_lookback")
    pivot_drift_confirm_bias = _gi("pivot_drift_confirm_bias")

    pct_extreme = _gf("pct_extreme")
    min_agreement = _gf("min_agreement")
    dur_extreme_pct = _gf("dur_extreme_pct")
    vol_surge_thresh = _gf("vol_surge_thresh")
    scale_div_thresh = _gf("scale_div_thresh")
    slope_thresh = _gf("slope_thresh")
    vola_high_pct = _gf("vola_high_pct")
    pivot_drift_thresh = _gf("pivot_drift_thresh")
    pivot_drift_gate_mult = _gf("pivot_drift_gate_mult")
    momentum_velocity_thresh = _gf("momentum_velocity_thresh")
    gjr_vote_thresh = _gf("gjr_vote_thresh")
    har_vote_thresh = _gf("har_vote_thresh")
```

Leave the categorical/bool `suggest_categorical` block (lines 414–437) and the `kwargs_high`/`kwargs_low` assembly UNCHANGED — categoricals stay inline-literal, which guarantees they are frozen identical across sides (spec §3).

Note: `pivot_drift_lb` keeps its variable name and `_gi("pivot_drift_lookback")` resolves the trial key to `{s}_pivot_drift_lb` via the override, so the journal key is byte-identical to today.

- [ ] **Step 4: Run the behavioral tests to verify they pass**

Run: `python -m pytest tests/test_search_space.py -q`
Expected: PASS (8 passed).

- [ ] **Step 5: Commit**

```bash
git add src/speculatores145.py tests/test_search_space.py
git commit -m "feat(search-space): drive params_from_trial int/float bounds per side"
```

---

## Task 4: Make the stability-probe samplers side-aware

**Files:**
- Modify: `src/speculatores145.py:733-758` (`_mutate_local_param`), `:761-772` (`_sample_local_params`), `:775-796` (`_sample_global_params`), and their call sites
- Modify: `tests/test_search_space.py` (add probe tests)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_search_space.py`:

```python
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
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_search_space.py::test_global_sampler_respects_high_relaxed_floor -q`
Expected: FAIL — current `_sample_global_params` uses the shared `FLOAT_BOUNDS` (floor 0.50), so it never samples below 0.50 for HIGH.

- [ ] **Step 3: Refactor the three probe functions to use the per-side space**

Replace `_mutate_local_param` (lines 733–758) with a space-aware version:

```python
def _mutate_local_param(
    rng: random.Random,
    value: Any,
    field_name: str,
    space: SearchSpace,
) -> Any:
    if field_name in space.int_bounds:
        low, high = space.int_bounds[field_name]
        radius = max(1, int(round((high - low) * 0.1)))
        local_low = max(low, int(value) - radius)
        local_high = min(high, int(value) + radius)
        return rng.randint(local_low, local_high)
    if field_name in space.float_bounds:
        low, high = space.float_bounds[field_name]
        radius = (high - low) * 0.1
        local_low = max(low, float(value) - radius)
        local_high = min(high, float(value) + radius)
        return rng.uniform(local_low, local_high)
    if field_name in space.bool_fields:
        return value if rng.random() < 0.85 else (not bool(value))
    if field_name in space.category_fields:
        choices = space.category_fields[field_name]
        if rng.random() < 0.85:
            return value
        alternatives = [choice for choice in choices if choice != value]
        return rng.choice(alternatives) if alternatives else value
    return value
```

Replace `_sample_local_params` (lines 761–772):

```python
def _sample_local_params(
    params: Params,
    side: str,
    rng: random.Random,
) -> Params:
    space = space_for(side)
    base_dict = _params_fields(Params())
    current = _params_fields(params)
    overrides: dict[str, Any] = {}
    field_names = (
        list(space.int_bounds) + list(space.float_bounds)
        + list(space.bool_fields) + list(space.category_fields)
    )
    for field_name in field_names:
        side_field = f"{field_name}_{side}"
        overrides[side_field] = _mutate_local_param(
            rng, current[side_field], field_name, space
        )
    return Params(**{**base_dict, **overrides})
```

Replace `_sample_global_params` (lines 775–796):

```python
def _sample_global_params(
    side: str,
    rng: random.Random,
) -> Params:
    """Uniform global restart sampler — Scorer v2 item 11.

    Draws each int/float field uniformly from its per-side bounds, each bool
    50/50, each categorical uniformly. Used by ``summarize_stability`` to
    compare the local basin around a winner against the wider search space.
    """
    space = space_for(side)
    base_dict = _params_fields(Params())
    overrides: dict[str, Any] = {}
    for field_name, (low, high) in space.int_bounds.items():
        overrides[f"{field_name}_{side}"] = rng.randint(low, high)
    for field_name, (low, high) in space.float_bounds.items():
        overrides[f"{field_name}_{side}"] = rng.uniform(low, high)
    for field_name in space.bool_fields:
        overrides[f"{field_name}_{side}"] = bool(rng.random() < 0.5)
    for field_name, choices in space.category_fields.items():
        overrides[f"{field_name}_{side}"] = rng.choice(list(choices))
    return Params(**{**base_dict, **overrides})
```

- [ ] **Step 4: Run to verify the probe tests pass**

Run: `python -m pytest tests/test_search_space.py -q`
Expected: PASS (10 passed).

- [ ] **Step 5: Commit**

```bash
git add src/speculatores145.py tests/test_search_space.py
git commit -m "feat(search-space): make stability-probe samplers per-side"
```

---

## Task 5: Regression — full search-space test suite + end-to-end smoke

**Files:**
- Reference (no edit unless broken): `temp/smoke_test_v15_end_to_end.py`

- [ ] **Step 1: Run the full unit suite**

Run: `python -m pytest tests/ -q`
Expected: PASS (10 passed).

- [ ] **Step 2: Run the existing end-to-end pipeline smoke**

Run: `python temp/smoke_test_v15_end_to_end.py`
Expected: completes and writes its markdown report with no exception (a 5-trial/side run; confirms the decoupled-bounds wiring doesn't break any downstream path).

- [ ] **Step 3: Sanity-check journal-key continuity against an existing study**

Run: `python -c "import optuna, src.speculatores145 as m; optuna.logging.set_verbosity(optuna.logging.WARNING); st=optuna.create_study(); st.optimize(lambda t:(m.params_from_trial(t,'high'),0.0)[1], n_trials=1); keys=set(st.trials[0].distributions); assert 'high_pivot_drift_lb' in keys and 'high_dur_extreme_pct' in keys; print('keys OK:', len(keys), 'params')"`
Expected: prints `keys OK: 34 params` (or the historical param count) — confirms trial keys unchanged.

- [ ] **Step 4: Commit any fixups (only if Step 2 required a change)**

```bash
git add -A
git commit -m "test(search-space): regression pass for decoupled bounds"
```

---

## Self-Review (completed by plan author)

- **Spec coverage (§3):** SearchSpace dataclass (Task 1) ✓; decouple HIGH/LOW (Task 3) ✓; two relaxed HIGH floors (Task 1 data + Task 3 wiring, asserted) ✓; trial-key naming contract incl. `pivot_drift_lb` (Tasks 1,3) ✓; categoricals frozen identical (Task 1 test + left inline in Task 3) ✓; stability-probe `_mutate_local_param`/`_sample_global_params` consult per-side space (Task 4) ✓; module location `src/search_space.py` ≤200 lines ✓.
- **Placeholder scan:** none — every code step contains complete code.
- **Type consistency:** `SearchSpace` fields (`int_bounds`, `float_bounds`, `bool_fields`, `category_fields`, `trial_key_overrides`) and `space_for(side)` are used identically across Tasks 1–4; `trial_key()` used consistently; `_mutate_local_param(rng, value, field_name, space)` signature matches its call in `_sample_local_params`.
- **Out of scope (Part 2):** universe registry, volume profiler, calendar folds, pooled scoring, notebook — covered by the Part 2 plan.
