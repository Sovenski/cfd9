"""Parity spec for the GPU pivot-drift precompute (build-spec §5.1, invariant P5).

The GPU evaluator replaces the growing ``confirmed_pivots`` stack of
``SpeculatorDetector._detect`` with a precomputed per-asset drift array.
This is ONLY legitimate if every per-bar drift value is byte-identical to
what the CPU loop computes:

- ``calc_pivot_drift`` semantics: ``None -> 0.0``, ``min_pivots = max(lookback, 2)``,
  ``/(min_pivots - 1)``, ``max(|start_val|, 1e-9)``.
- Same-bar push order is HIGH pivot before LOW pivot (detector.py L533-536).
- Drift is read from the PRE-update stack, THEN this bar's pivots are appended
  (read-BEFORE-append).
- A fixed ring of capacity ``Kmax >= 20`` is sufficient
  (``search_space.INT_BOUNDS["pivot_drift_lookback"] = (2, 20)``).
"""
from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.detector import SpeculatorDetector, build_detector_artifacts
from src.indicators import Params, calc_pivot_drift, pivot_high, pivot_low
from src.pooled_validation import load_stream_frame

gpumod = pytest.importorskip("src.v17_gpu.drift_precompute")  # RED until module exists
DriftSpec = gpumod.DriftSpec
confirmed_pivot_events = gpumod.confirmed_pivot_events
drift_per_bar = gpumod.drift_per_bar
precompute_drift = gpumod.precompute_drift
precompute_drift_batch = gpumod.precompute_drift_batch

_CSV = Path("data/raw/SPX_1D_20170428_20260318.csv")


def _df() -> pd.DataFrame:
    if not _CSV.exists():
        pytest.skip(f"missing {_CSV}")
    return load_stream_frame(str(_CSV))


def _reference_drift_loop(
    df: pd.DataFrame,
    baseline_lb: int,
    lookback_high: int,
    lookback_low: int,
    low_before_high: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Inline replica of the CPU ``_detect`` drift portion (detector.py L481-536)."""
    df = df.reset_index(drop=True)
    high = df["high"]
    low = df["low"]
    n = len(df)
    ph_arr = pivot_high(high, baseline_lb).values
    pl_arr = pivot_low(low, baseline_lb).values
    high_arr = high.values.astype(float)
    low_arr = low.values.astype(float)

    confirmed_pivots: list[float] = []
    out_high = np.zeros(n, dtype=float)
    out_low = np.zeros(n, dtype=float)
    for t in range(n):
        current_baseline_ph = np.nan
        current_baseline_pl = np.nan
        pivot_bar = t - baseline_lb
        if pivot_bar >= 0:
            if ph_arr[pivot_bar]:
                current_baseline_ph = high_arr[pivot_bar]
            if pl_arr[pivot_bar]:
                current_baseline_pl = low_arr[pivot_bar]

        drift_high = calc_pivot_drift(confirmed_pivots, lookback_high)
        out_high[t] = drift_high if drift_high is not None else 0.0
        drift_low = calc_pivot_drift(confirmed_pivots, lookback_low)
        out_low[t] = drift_low if drift_low is not None else 0.0

        # Pine updates the confirmed pivot stack AFTER evaluating drift.
        if low_before_high:  # deliberately wrong order — used to prove order matters
            if not np.isnan(current_baseline_pl):
                confirmed_pivots.append(current_baseline_pl)
            if not np.isnan(current_baseline_ph):
                confirmed_pivots.append(current_baseline_ph)
        else:  # oracle order: HIGH before LOW (detector.py L533-536)
            if not np.isnan(current_baseline_ph):
                confirmed_pivots.append(current_baseline_ph)
            if not np.isnan(current_baseline_pl):
                confirmed_pivots.append(current_baseline_pl)
    return out_high, out_low


def _synthetic_df() -> pd.DataFrame:
    """40-bar OHLC frame where bar 10 confirms BOTH a high and a low pivot."""
    n = 40
    rng = np.random.default_rng(7)
    base = 100.0 + np.cumsum(rng.normal(0.0, 0.3, n))
    high = base + 1.0
    low = base - 1.0
    # bar 10: outside bar — window max on the high side AND window min on the low
    high[10] = base.max() + 25.0
    low[10] = base.min() - 25.0
    # extra clear pivots so min_pivots is reached during the frame
    high[5] += 6.0
    low[7] -= 6.0
    high[20] += 7.0
    low[24] -= 7.0
    high[30] += 8.0
    low[33] -= 8.0
    return pd.DataFrame(
        {
            "open": base,
            "high": high,
            "low": low,
            "close": base,
            "volume": np.full(n, 1000.0),
        }
    )


# ---------------------------------------------------------------------------
# P5 against the REAL CPU _detect loop (debug columns are the oracle output)
# ---------------------------------------------------------------------------


def test_drift_matches_detector_debug_columns_on_base_params() -> None:
    df = _df()
    p = Params()
    art = build_detector_artifacts(df)
    ref = SpeculatorDetector(df, p, art, include_debug_columns=True).run()
    drift = precompute_drift(df, DriftSpec.from_params(p))
    assert drift.dtype == np.float64
    assert drift.shape == (len(df), 2)
    assert np.array_equal(drift[:, 0], ref["pivot_drift_high"].values)
    assert np.array_equal(drift[:, 1], ref["pivot_drift_low"].values)


def test_drift_matches_inline_reference_under_lookback_draws() -> None:
    df = _df()
    p = Params()
    rng = np.random.default_rng(42)
    for _ in range(5):
        lb_h = int(rng.integers(2, 21))
        lb_l = int(rng.integers(2, 21))
        spec = DriftSpec.from_params(
            dataclasses.replace(
                p, pivot_drift_lookback_high=lb_h, pivot_drift_lookback_low=lb_l
            )
        )
        ref_h, ref_l = _reference_drift_loop(df, p.baseline_lb, lb_h, lb_l)
        drift = precompute_drift(df, spec)
        assert np.array_equal(drift[:, 0], ref_h), (lb_h, lb_l)
        assert np.array_equal(drift[:, 1], ref_l), (lb_h, lb_l)


# ---------------------------------------------------------------------------
# Edge cases: same-bar HIGH+LOW pivot, first satisfied bar, read-before-append
# ---------------------------------------------------------------------------


def test_same_bar_high_and_low_pivot_high_pushed_first() -> None:
    df = _synthetic_df()
    baseline_lb = 2
    high = df["high"]
    low = df["low"]
    ph = pivot_high(high, baseline_lb).values
    pl = pivot_low(low, baseline_lb).values
    assert bool(ph[10]) and bool(pl[10]), "fixture must confirm BOTH pivots at bar 10"

    spec = DriftSpec(baseline_lb=baseline_lb, lookback_high=2, lookback_low=3)
    drift = precompute_drift(df, spec)
    ref_h, ref_l = _reference_drift_loop(df, baseline_lb, 2, 3)
    assert np.array_equal(drift[:, 0], ref_h)
    assert np.array_equal(drift[:, 1], ref_l)

    # The test has teeth: pushing LOW before HIGH must change the drift.
    bad_h, bad_l = _reference_drift_loop(df, baseline_lb, 2, 3, low_before_high=True)
    assert not (np.array_equal(drift[:, 0], bad_h) and np.array_equal(drift[:, 1], bad_l))

    # The HIGH value precedes the LOW value in the event sequence at bar 10.
    ev_bars, ev_vals = confirmed_pivot_events(
        high.values.astype(float), low.values.astype(float), ph, pl, baseline_lb
    )
    idx = np.flatnonzero(ev_bars == 12)  # confirmed at bar 10 + baseline_lb
    assert len(idx) == 2
    assert ev_vals[idx[0]] == high.values[10]
    assert ev_vals[idx[1]] == low.values[10]


def test_first_bar_min_pivots_satisfied_and_read_before_append() -> None:
    df = _synthetic_df()
    baseline_lb = 2
    lookback = 3  # min_pivots = 3
    spec = DriftSpec(baseline_lb=baseline_lb, lookback_high=lookback, lookback_low=lookback)
    drift = precompute_drift(df, spec)

    high = df["high"]
    low = df["low"]
    ph = pivot_high(high, baseline_lb).values
    pl = pivot_low(low, baseline_lb).values
    ev_bars, _ = confirmed_pivot_events(
        high.values.astype(float), low.values.astype(float), ph, pl, baseline_lb
    )
    assert len(ev_bars) >= 3, "fixture must confirm at least min_pivots pivots"
    t_third = int(ev_bars[2])  # bar that APPENDS the 3rd pivot

    # Read-BEFORE-append: at t_third the stack still has < 3 entries -> drift 0.0.
    assert np.all(drift[: t_third + 1, 0] == 0.0)
    # First bar where min_pivots is satisfied is the NEXT bar; drift is non-zero.
    assert drift[t_third + 1, 0] != 0.0
    ref_h, _ = _reference_drift_loop(df, baseline_lb, lookback, lookback)
    assert drift[t_third + 1, 0] == ref_h[t_third + 1]


def test_kmax_ring_truncation_is_lossless_at_boundary() -> None:
    """A fixed Kmax=20 ring must reproduce the growing stack for lookback=20."""
    df = _df().iloc[:1500].reset_index(drop=True)
    baseline_lb = Params().baseline_lb
    lookback = 20  # min_pivots == Kmax floor — the truncation boundary
    spec = DriftSpec(baseline_lb=baseline_lb, lookback_high=lookback, lookback_low=lookback)
    assert spec.kmax >= 20

    high = df["high"]
    low = df["low"]
    ph = pivot_high(high, baseline_lb).values
    pl = pivot_low(low, baseline_lb).values
    ev_bars, ev_vals = confirmed_pivot_events(
        high.values.astype(float), low.values.astype(float), ph, pl, baseline_lb
    )
    assert len(ev_bars) > spec.kmax, "fixture must overflow the ring"

    # Literal fixed-capacity ring simulation (what the §5.4 scan will carry).
    ring = np.zeros(spec.kmax, dtype=np.float64)
    count = 0
    out = np.zeros(len(df), dtype=np.float64)
    ev_i = 0
    min_pivots = max(lookback, 2)
    for t in range(len(df)):
        if count >= min_pivots:
            start = ring[(count - min_pivots) % spec.kmax]
            end = ring[(count - 1) % spec.kmax]
            out[t] = ((end - start) / max(abs(start), 1e-9)) / (min_pivots - 1)
        while ev_i < len(ev_bars) and ev_bars[ev_i] == t:
            ring[count % spec.kmax] = ev_vals[ev_i]
            count += 1
            ev_i += 1

    drift = precompute_drift(df, spec)
    assert np.array_equal(drift[:, 0], out)
    assert np.array_equal(drift[:, 1], out)


# ---------------------------------------------------------------------------
# Batch packing + spec validation
# ---------------------------------------------------------------------------


def test_batch_pads_with_zero_and_keeps_assets_isolated() -> None:
    df_a = _synthetic_df()
    df_b = _df().iloc[:300].reset_index(drop=True)
    spec = DriftSpec(baseline_lb=2, lookback_high=2, lookback_low=4)
    batch = precompute_drift_batch([df_a, df_b], spec)
    assert batch.dtype == np.float64
    assert batch.shape == (2, len(df_b), 2)
    assert np.array_equal(batch[0, : len(df_a)], precompute_drift(df_a, spec))
    assert np.all(batch[0, len(df_a):] == 0.0)  # pad region
    assert np.array_equal(batch[1], precompute_drift(df_b, spec))


def test_spec_validates_kmax_and_lookbacks() -> None:
    with pytest.raises(ValueError):
        DriftSpec(baseline_lb=20, lookback_high=5, lookback_low=5, kmax=10)
    with pytest.raises(ValueError):
        DriftSpec(baseline_lb=20, lookback_high=25, lookback_low=5, kmax=20)
    spec = DriftSpec.from_params(Params())
    assert spec.kmax >= 20
