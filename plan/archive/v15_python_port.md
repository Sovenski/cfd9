# V15 Python Port Plan — Edge-Triggered Voting

**Goal:** Port Pine V15's edge-triggered voting rule to the Python detector so the optimizer can search for `use_edge_voting_*` and `edge_window_*` as new dimensions. Maintain bit-exact parity with V14 when `use_edge_voting=False`.

**Naming:** This is V15 across the board — Pine indicator name, notebook title, study slug, results dir. V14 study/results stay untouched as a reference.

**Constraints (non-negotiable):**
- **Zero gap with Pine.** Per the parity mandate: V14 preset runs on Python must produce identical signals to V14 Pine; V15 Edge preset runs on Python must produce identical signals to V15 Pine with `use_edge_voting=true`.
- **Backward compatible.** Default `use_edge_voting=False` for all existing Params. Detector behavior unchanged when those defaults are used.
- **No detector compute regression.** Edge wrapping must remain O(n) per side; no extra loops over scales or history.

---

## Pine V15 spec being ported (the source of truth)

```pine
edge_or_state(bool v, int win, bool use_edge) =>
    bool prev = bar_index >= win ? v[win] : false
    use_edge ? (v and not prev) : v
```

Each of the 8 vote sources per side is wrapped:
```
t_up_h_eff = edge_or_state(trend_up_high, edge_window_high, use_edge_voting_high)
pd_dn_h_eff = edge_or_state(pivot_drift_down_high, edge_window_high, use_edge_voting_high)
vs_h_eff = edge_or_state(_vol_slow_high, edge_window_high, use_edge_voting_high)
md_h_eff = edge_or_state(_mom_div_neg_high, edge_window_high, use_edge_voting_high)
mv_h_eff = edge_or_state(mom_vel_high_ok, edge_window_high, use_edge_voting_high)
va_h_eff = edge_or_state(vola_elevated_high, edge_window_high, use_edge_voting_high)
g_h_eff = edge_or_state(gjr_high_vote, edge_window_high, use_edge_voting_high)
h_h_eff = edge_or_state(har_high_vote, edge_window_high, use_edge_voting_high)
```
Same for LOW side. `ph_confirms` / `pl_confirms` sum the `_eff` bools (still gated by `use_*` flags).

**Inline-predicate hoisting** required for parity:
- `_vol_slow_high = vol_surge_high < (1.0 / vol_surge_thresh_high)`
- `_vol_surge_low = vol_surge_low > vol_surge_thresh_low`
- `_mom_div_neg_high = mom_diverge_high < 0`
- `_mom_div_neg_low = mom_diverge_low < 0`

---

## Implementation tasks (order matters)

### 1. `src/indicators.py` — Add 4 Params fields

```python
use_edge_voting_high: bool = False
edge_window_high: int = 5
use_edge_voting_low: bool = False
edge_window_low: int = 5
```

Defaults reproduce V14 behavior (passthrough). Keep `frozen=True`.

### 2. `src/detector.py` — Add `_edge_or_state` helper + wire into `_detect()`

**2a. Module-level helper:**
```python
def _edge_or_state(state: np.ndarray, win: int, use_edge: bool) -> np.ndarray:
    """Vectorized Pine V15 edge_or_state.
    When use_edge=False, return state unchanged. When True, return
    state AND NOT state-shifted-back-by-win, with the shifted region
    before bar `win` zero-filled (mirrors Pine's `bar_index >= win ? v[win] : false`).
    """
    if not use_edge or win <= 0:
        return state.astype(bool, copy=True)
    out = state.astype(bool, copy=True)
    if win < len(out):
        shifted = np.zeros_like(out)
        shifted[win:] = out[:-win]
        out = out & ~shifted
    return out
```

**2b. Phase 1 additions** (after existing per-side vote arrays are built, before phase 2 loop): build the 14 `eff_*` arrays for the 7 vectorizable predicates × 2 sides.

```python
# HIGH side hoisted predicates
_vol_slow_high  = (vol_surge_high_arr < (1.0 / p.vol_surge_thresh_high))
_mom_div_neg_h  = (mom_diverge_high_arr < 0)
eff_t_up_h      = _edge_or_state(trend_up_high_arr, p.edge_window_high, p.use_edge_voting_high)
eff_vs_h        = _edge_or_state(_vol_slow_high,    p.edge_window_high, p.use_edge_voting_high)
eff_md_h        = _edge_or_state(_mom_div_neg_h,    p.edge_window_high, p.use_edge_voting_high)
eff_mv_h        = _edge_or_state(mom_vel_high_ok_arr, p.edge_window_high, p.use_edge_voting_high)
eff_va_h        = _edge_or_state(vola_elevated_high_arr, p.edge_window_high, p.use_edge_voting_high)
eff_g_h         = _edge_or_state(gjr_high_vote_arr, p.edge_window_high, p.use_edge_voting_high)
eff_h_h         = _edge_or_state(har_high_vote_arr, p.edge_window_high, p.use_edge_voting_high)

# LOW side (analogous)
_vol_surge_low_ = (vol_surge_low_arr > p.vol_surge_thresh_low)
_mom_div_neg_l  = (mom_diverge_low_arr < 0)
eff_t_dn_l      = _edge_or_state(trend_down_low_arr, p.edge_window_low, p.use_edge_voting_low)
eff_vs_l        = _edge_or_state(_vol_surge_low_,    p.edge_window_low, p.use_edge_voting_low)
eff_md_l        = _edge_or_state(_mom_div_neg_l,     p.edge_window_low, p.use_edge_voting_low)
eff_mv_l        = _edge_or_state(mom_vel_low_ok_arr, p.edge_window_low, p.use_edge_voting_low)
eff_va_l        = _edge_or_state(vola_elevated_low_arr, p.edge_window_low, p.use_edge_voting_low)
eff_g_l         = _edge_or_state(gjr_low_vote_arr,  p.edge_window_low, p.use_edge_voting_low)
eff_h_l         = _edge_or_state(har_low_vote_arr,  p.edge_window_low, p.use_edge_voting_low)
```

**2c. Phase 2 loop**: maintain bar-by-bar history of `pivot_drift_down_high` and `pivot_drift_up_low` (only votes that aren't vectorized in phase 1 because they depend on the running `confirmed_pivots` array). Compute their `eff_*` value inline.

```python
# Pre-loop allocations
state_pd_dn_h = np.zeros(n, dtype=bool)
state_pd_up_l = np.zeros(n, dtype=bool)

# Inside the loop, AFTER computing drift_down_high / drift_up_low:
state_pd_dn_h[t] = drift_down_high
state_pd_up_l[t] = drift_up_low

# Per-bar effective drift vote
eff_pd_dn_h_t = (
    drift_down_high if (not p.use_edge_voting_high or t < p.edge_window_high)
    else (drift_down_high and not state_pd_dn_h[t - p.edge_window_high])
)
eff_pd_up_l_t = (
    drift_up_low if (not p.use_edge_voting_low or t < p.edge_window_low)
    else (drift_up_low and not state_pd_up_l[t - p.edge_window_low])
)
```

**2d. Vote aggregation** — replace existing `ph_c = int(sum([...]))` and `pl_c = int(sum([...]))` lists with the `eff_*` versions:

```python
ph_c = int(sum([
    p.use_trend_high and bool(eff_t_up_h[t]),
    _USE_PIVOT_DRIFT and eff_pd_dn_h_t,
    p.use_volume_high and bool(eff_vs_h[t]),
    p.use_momentum_high and bool(eff_md_h[t]),
    p.use_momentum_velocity_high and bool(eff_mv_h[t]),
    p.use_volatility_high and bool(eff_va_h[t]),
    p.use_gjr_asym_high and bool(eff_g_h[t]),
    p.use_har_vol_high and bool(eff_h_h[t]),
]))
```
Same for `pl_c`.

### 3. `src/speculatores145.py` — Add 4 trial dimensions + Edge seed

**3a. `params_from_trial`** add per side:
```python
use_edge_voting = trial.suggest_categorical(f"{s}_use_edge_voting", [True, False])
edge_window = trial.suggest_int(f"{s}_edge_window", 3, 60)
```
Wide integer range (not categorical) so TPE can find side-asymmetric K
freely — LOW likely wants K=3-5 (sharp capitulation), HIGH likely wants
K=20-50 if edge is preferred at all. Add to `kwargs_high` / `kwargs_low`.

**3b. Add Heuristic Structural Edge seed dict** (mirrors `SEED_HEURISTIC_STRUCTURAL_HIGH/LOW` but with `use_edge_voting=True, edge_window=5`).

**3c. Update version string** in module header: `VERSION = "Speculatores 15"`. Keep `VERSION_SLUG` updated so the journal lands at a new path.

### 4. `optimize.ipynb` — Rename to V15

- Title cell (markdown): "Speculatores 15 Colab Runner"
- Any references to "14.5" / "14" → "15"
- Default config: keep TRIALS_PER_SIDE=5000, WORKERS_PER_SIDE=4
- STUDY_PREFIX template uses VERSION_SLUG so naturally inherits new name

### 5. `temp/smoke_test_v15_parity.py` — Three regression checks

1. **V14-style on new code:** Run Path A 5k preset (use_edge=False) through V15 detector. Compare HIGH/LOW signal arrays byte-for-byte against V14 reference. Must match exactly.
2. **Edge variant produces fewer signals:** Same preset but `use_edge_voting_*=True, edge_window_*=5`. Confirm signal count is strictly less than V14 reference, and check no signal is at a bar where state-version didn't also fire.
3. **End-to-end no-error run:** Run 5-trial Optuna with new search space, verify journal writes and report generates without exceptions.

### 6. (Deferred to after-run) `pine/speculatores_v15_presets_gold.pine` — Add winning Edge preset

If the optimizer finds a configuration better than Path A 5k Edge, add it as a new preset in Pine V15 using the existing `add_pine_preset_*.py` pattern. Out of scope for this port.

---

## Files touched (summary)

| File | Action | Lines (estimate) |
|---|---|---|
| `src/indicators.py` | Add 4 Params fields | +6 |
| `src/detector.py` | Add helper + phase 1 eff arrays + phase 2 inline drift edges + modified vote sums | +60 / ~16 modified |
| `src/speculatores145.py` | Trial dimensions + Edge seed + VERSION bump | +80 (mostly seed dicts) |
| `optimize.ipynb` | Title / version refs | ~5 lines via JSON patch |
| `temp/smoke_test_v15_parity.py` | New regression script | ~150 lines |

Total: ~250 lines net change in repo. No file restructuring, no architectural changes.

---

## Risks & mitigations

| Risk | Mitigation |
|---|---|
| Parity drift between Pine and Python for `_eff` boundary at bar < win | Match Pine exactly: when `t < win`, edge is unchanged (= state). Test #1 covers this. |
| Optimizer collapses to all-zero scores when use_edge=True (votes too scarce) | TPE will explore both branches (`use_edge=True/False`). If edge-on regions are all dead, they get pruned. State branch still wins. We can also seed Path A 5k Edge with `confirm_count=1` to force at least some firing. |
| `use_edge_voting` and `edge_window` increase search dimensionality from 35 to 39 — slower convergence | Run 5k trials (already standard); also seed both Path A 5k (state) AND Path A 5k Edge (edge) so optimizer starts with two anchors. |
| Confusion between V14 / V15 results in `results/` directory | New VERSION_SLUG forces new journal path. Old results stay in V14 dirs. |

---

## Execution order

1. Implement task 1 (Params) — verify import OK
2. Implement task 2 (detector) — verify imports OK, run a single-trial detector call with use_edge=False, confirm it produces signals
3. Implement task 5 (smoke test #1, V14-equivalence) — must pass before continuing
4. Implement task 3 (speculatores145) — verify imports OK
5. Implement task 5 (smoke test #2, edge produces fewer signals)
6. Implement task 4 (notebook rename)
7. Run task 5 (smoke test #3, 5-trial end-to-end)
8. Commit + push, then user can do a real run

Estimated time: 90-120 minutes for implementation + smoke tests.

---

## Out of scope (explicitly)

- Pine V15 file edits — already done in commits `744b450`, `5bfa978`, `659a930`
- Adding new winning presets to Pine — wait for optimizer results
- Cross-asset parity audit — happens after a real run produces a winning preset
- Score function changes — V15 uses the same Path A scorer
- Fold scheme changes — V15 uses the same bar-based 14-fold scheme
