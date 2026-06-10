"""Precomputed per-asset pivot drift (build-spec §5.1, parity invariant P5).

Replaces the growing ``confirmed_pivots`` stack of ``SpeculatorDetector._detect``
(src/detector.py L464, L497-536) with a per-bar drift array the batched GPU
evaluator can read without carrying a Python list in the scan state.

P5 semantics reproduced EXACTLY (oracle: ``indicators.calc_pivot_drift`` +
the ``_detect`` loop):

- ``None -> 0.0`` when fewer than ``min_pivots`` pivots are confirmed.
- ``min_pivots = max(lookback, 2)``.
- ``drift = ((end - start) / max(|start|, 1e-9)) / (min_pivots - 1)``.
- Same-bar push order is HIGH pivot before LOW pivot (detector.py L533-536).
- Drift at bar ``t`` is read from the PRE-update stack — only pivots appended
  at bars ``< t`` are visible (read-BEFORE-append).
- A fixed ring of capacity ``Kmax >= 20`` is lossless because the read window
  ``[count - min_pivots, count - 1]`` always lies within the last ``Kmax``
  entries (``search_space.INT_BOUNDS["pivot_drift_lookback"] = (2, 20)``).

All drift values are float64 (parity invariant P1).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

from src.indicators import Params, pivot_high, pivot_low

logger = logging.getLogger(__name__)

#: Lower bound on the shared pivot-ring capacity. From the FROZEN search space:
#: ``search_space.INT_BOUNDS["pivot_drift_lookback"] = (2, 20)`` -> Kmax >= 20.
KMAX_FLOOR: int = 20

#: Side index into the last axis of the drift arrays produced here.
HIGH: int = 0
LOW: int = 1


def _min_pivots(lookback: int) -> int:
    """``calc_pivot_drift``'s ``min_pivots = max(lookback, 2)``."""
    return max(int(lookback), 2)


@dataclass(frozen=True)
class DriftSpec:
    """Shape/lookback inputs of the drift precompute.

    Args:
        baseline_lb: ``Params.baseline_lb`` — pivot confirmation lag.
        lookback_high: ``Params.pivot_drift_lookback_high``.
        lookback_low: ``Params.pivot_drift_lookback_low``.
        kmax: capacity of the (shared, per-asset) pivot ring; must be >= 20
            and >= both sides' ``min_pivots`` (P5).
    """

    baseline_lb: int
    lookback_high: int
    lookback_low: int
    kmax: int = KMAX_FLOOR

    def __post_init__(self) -> None:
        if self.kmax < KMAX_FLOOR:
            raise ValueError(
                f"kmax={self.kmax} below the P5 floor {KMAX_FLOOR} "
                "(search_space pivot_drift_lookback upper bound)"
            )
        needed = max(_min_pivots(self.lookback_high), _min_pivots(self.lookback_low))
        if needed > self.kmax:
            raise ValueError(
                f"ring capacity kmax={self.kmax} < required min_pivots={needed}; "
                "drift would be lossy"
            )

    @property
    def min_pivots_high(self) -> int:
        return _min_pivots(self.lookback_high)

    @property
    def min_pivots_low(self) -> int:
        return _min_pivots(self.lookback_low)

    @classmethod
    def from_params(cls, p: Params) -> "DriftSpec":
        """Build the spec for one ``Params`` draw (ring sized >= 20, P5)."""
        return cls(
            baseline_lb=int(p.baseline_lb),
            lookback_high=int(p.pivot_drift_lookback_high),
            lookback_low=int(p.pivot_drift_lookback_low),
            kmax=max(
                KMAX_FLOOR,
                _min_pivots(p.pivot_drift_lookback_high),
                _min_pivots(p.pivot_drift_lookback_low),
            ),
        )


def confirmed_pivot_events(
    high_arr: np.ndarray,
    low_arr: np.ndarray,
    ph_flags: np.ndarray,
    pl_flags: np.ndarray,
    baseline_lb: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Ordered confirmed-pivot event stream for ONE asset.

    Mirrors the ``_detect`` loop: at bar ``t`` the pivot formed at
    ``pivot_bar = t - baseline_lb`` becomes known; if both a high and a low
    pivot confirm on the same bar, the HIGH value is pushed first (P5).

    Args:
        high_arr: float64 highs (``high.values.astype(float)``).
        low_arr: float64 lows.
        ph_flags: ``pivot_high(high, baseline_lb).values`` (bool).
        pl_flags: ``pivot_low(low, baseline_lb).values`` (bool).
        baseline_lb: pivot confirmation lag.

    Returns:
        ``(event_bars, event_values)`` — append bar index (int64, ascending,
        HIGH-before-LOW within a bar) and the pivot price (float64).
    """
    n = len(high_arr)
    bars = np.arange(n, dtype=np.int64)
    pivot_bar = bars - int(baseline_lb)
    valid = pivot_bar >= 0
    ph_at = np.zeros(n, dtype=bool)
    pl_at = np.zeros(n, dtype=bool)
    ph_at[valid] = ph_flags[pivot_bar[valid]].astype(bool)
    pl_at[valid] = pl_flags[pivot_bar[valid]].astype(bool)

    hb = bars[ph_at]
    lb = bars[pl_at]
    ev_bars = np.concatenate([hb, lb])
    ev_vals = np.concatenate(
        [
            high_arr[hb - baseline_lb].astype(np.float64),
            low_arr[lb - baseline_lb].astype(np.float64),
        ]
    )
    # 0 = HIGH, 1 = LOW: lexsort keys are (secondary, primary) -> within one
    # bar the HIGH event sorts before the LOW event (P5 push order).
    side = np.concatenate(
        [np.zeros(len(hb), dtype=np.int8), np.ones(len(lb), dtype=np.int8)]
    )
    order = np.lexsort((side, ev_bars))
    return ev_bars[order], ev_vals[order]


def drift_per_bar(
    event_bars: np.ndarray,
    event_values: np.ndarray,
    lookback: int,
    n_bars: int,
) -> np.ndarray:
    """Per-bar drift for one side, byte-identical to ``calc_pivot_drift``.

    Read-BEFORE-append: the count visible at bar ``t`` covers events appended
    at bars strictly before ``t`` (``searchsorted(..., side="left")``).

    Args:
        event_bars: ascending append-bar indices from ``confirmed_pivot_events``.
        event_values: matching pivot prices (float64).
        lookback: ``pivot_drift_lookback`` for this side.
        n_bars: slice length.

    Returns:
        float64 array of length ``n_bars``; 0.0 where the CPU returns ``None``.
    """
    min_pivots = _min_pivots(lookback)
    drift = np.zeros(n_bars, dtype=np.float64)
    counts = np.searchsorted(event_bars, np.arange(n_bars, dtype=np.int64), side="left")
    ok = counts >= min_pivots
    if not ok.any():
        return drift
    c = counts[ok]
    start = event_values[c - min_pivots]
    end = event_values[c - 1]
    # ((end - start) / max(|start|, 1e-9)) / (min_pivots - 1) — verbatim P5.
    drift[ok] = ((end - start) / np.maximum(np.abs(start), 1e-9)) / (min_pivots - 1)
    return drift


def precompute_drift(df: pd.DataFrame, spec: DriftSpec) -> np.ndarray:
    """Drift array ``[n_bars, 2]`` (HIGH, LOW) float64 for one asset slice.

    Args:
        df: OHLC frame with ``high``/``low`` columns (one slice, one asset).
        spec: shape/lookback spec (typically ``DriftSpec.from_params``).

    Returns:
        float64 array ``[len(df), 2]``; column 0 = high side, 1 = low side.
    """
    df = df.reset_index(drop=True)
    high = df["high"]
    low = df["low"]
    ph_flags = pivot_high(high, spec.baseline_lb).values
    pl_flags = pivot_low(low, spec.baseline_lb).values
    ev_bars, ev_vals = confirmed_pivot_events(
        high.values.astype(float),
        low.values.astype(float),
        ph_flags,
        pl_flags,
        spec.baseline_lb,
    )
    n = len(df)
    out = np.zeros((n, 2), dtype=np.float64)
    out[:, HIGH] = drift_per_bar(ev_bars, ev_vals, spec.lookback_high, n)
    out[:, LOW] = drift_per_bar(ev_bars, ev_vals, spec.lookback_low, n)
    return out


def precompute_drift_batch(
    dfs: Sequence[pd.DataFrame], spec: DriftSpec
) -> np.ndarray:
    """Padded drift batch ``[n_assets, max_bars, 2]`` float64.

    Each asset is computed independently (no pivot/cooldown leak across
    instruments — P7); pad bars carry 0.0, matching the CPU loop's zero
    initialisation of the drift output arrays.

    Args:
        dfs: one OHLC frame per asset slice.
        spec: shared shape/lookback spec.

    Returns:
        float64 array ``[len(dfs), max(len(df)), 2]``.
    """
    if not dfs:
        return np.zeros((0, 0, 2), dtype=np.float64)
    max_bars = max(len(df) for df in dfs)
    out = np.zeros((len(dfs), max_bars, 2), dtype=np.float64)
    for i, df in enumerate(dfs):
        out[i, : len(df)] = precompute_drift(df, spec)
    logger.debug(
        "precomputed drift batch: %d assets, max_bars=%d, kmax=%d",
        len(dfs),
        max_bars,
        spec.kmax,
    )
    return out
