"""Signal Card §3.2 — right-span survival core (spec FIXED v3).

Kaplan-Meier survival with native right-censoring, fitted on the right-span
``R`` (re-review R1: the SAME quantity the live counter tracks — never on
N*), plus the F6 cluster-bootstrap confidence bands (Greenwood/binomial CIs
are forbidden: signals cluster by stream/era, so pooled CIs would be too
tight).

Scorer/calibration-side only — detection math is frozen (spec §0).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Sequence, Union

import numpy as np

from ..scoring_v5 import SPAN_GRID

logger = logging.getLogger(__name__)

#: One stream's (or block's) survival inputs: (values, censored-flags).
ClusterData = tuple[np.ndarray, np.ndarray]


# ---------------------------------------------------------------------------
# Right-span R (re-review R1)
# ---------------------------------------------------------------------------


def right_span(
    prices: Union[Sequence[float], np.ndarray],
    i: int,
    is_high: bool = False,
    cap: int = SPAN_GRID[-1],
) -> tuple[int, bool]:
    """Right-span ``R`` of the candidate pivot bar ``i`` (spec §3.2, R1).

    ``R`` = number of bars after bar ``i`` until a STRICTLY lower low occurs
    (HIGH side: strictly higher high). Right-censored at the series edge and
    at the grid cap (the "500+" bucket, §1.2/F10) — exactly like the live
    bars-survived counter.

    Returns:
        ``(R, censored)`` — when censored, ``R`` is the proven lower bound.
    """
    arr = np.asarray(prices, dtype=float)
    n = arr.shape[0]
    if not 0 <= i < n:
        raise IndexError(f"bar index {i} out of range for {n} bars")
    v = arr[i]
    last = min(i + cap, n - 1)
    for j in range(i + 1, last + 1):
        breached = (arr[j] > v) if is_high else (arr[j] < v)
        if breached:
            return j - i, False
    if i + cap <= n - 1:
        return cap, True  # "500+": R >= cap, never exact
    return n - 1 - i, True  # series edge: proven lower bound


# ---------------------------------------------------------------------------
# Kaplan-Meier with right-censoring
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class KMSurvival:
    """Product-limit estimate of ``S(x) = P(R >= x)``.

    ``survival[j]`` is the survival just ABOVE ``event_times[j]``, i.e.
    ``P(R > event_times[j])``; censored observations at value ``c`` remain
    at risk through ``c`` (standard event-before-censoring tie convention).
    """

    event_times: np.ndarray
    survival: np.ndarray
    n_events: int
    n_censored: int


def fit_km(
    values: Union[Sequence[float], np.ndarray],
    censored: Union[Sequence[bool], np.ndarray, None] = None,
) -> KMSurvival:
    """Fit the KM survival curve on (possibly right-censored) spans.

    Args:
        values: Observed right-spans (exact when uncensored, proven lower
            bounds when censored — incl. edge and "500+" cap entries).
        censored: Right-censoring flags (default: all exact).

    Returns:
        ``KMSurvival`` step function (``P(R >= x)`` via :func:`km_at`).
    """
    vals = np.asarray(values, dtype=float)
    cens = (np.zeros(vals.shape[0], dtype=bool) if censored is None
            else np.asarray(censored, dtype=bool))
    if vals.shape != cens.shape:
        raise ValueError(
            f"values shape {vals.shape} != censored shape {cens.shape}")
    event_times = np.unique(vals[~cens])
    surv = np.empty(event_times.shape[0], dtype=float)
    s = 1.0
    for j, r in enumerate(event_times):
        n_at_risk = int(np.count_nonzero(vals >= r))
        d_events = int(np.count_nonzero((vals == r) & ~cens))
        s *= 1.0 - d_events / n_at_risk
        surv[j] = s
    curve = KMSurvival(
        event_times=event_times, survival=surv,
        n_events=int(np.count_nonzero(~cens)),
        n_censored=int(np.count_nonzero(cens)),
    )
    logger.debug("fit_km: %d events, %d censored, %d distinct event times",
                 curve.n_events, curve.n_censored, event_times.shape[0])
    return curve


def km_at(curve: KMSurvival, x: float) -> float:
    """Evaluate ``S(x) = P(R >= x)`` (product over event times < x)."""
    idx = int(np.searchsorted(curve.event_times, x, side="left"))
    return 1.0 if idx == 0 else float(curve.survival[idx - 1])


def km_table(
    curve: KMSurvival, grid: Sequence[int] = tuple(SPAN_GRID),
) -> np.ndarray:
    """Survival table ``S_R`` at the span-grid points (the exported table)."""
    return np.array([km_at(curve, g) for g in grid], dtype=float)


# ---------------------------------------------------------------------------
# F6 — cluster-bootstrap confidence bands
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SurvivalBands:
    """Bootstrap envelope around the pooled KM table (spec §3.2, F6).

    The card displays the BAND ``[s_lo, s_hi]``, never a point percentage;
    ``method`` records whether whole streams or contiguous signal-blocks
    were resampled (block fallback when stream count < 5).
    """

    grid: tuple[int, ...]
    s_lo: np.ndarray
    s_hi: np.ndarray
    s_point: np.ndarray
    n_boot: int
    method: str


def _block_units(units: list[ClusterData], target_blocks: int) -> list[ClusterData]:
    """Split each stream into contiguous signal-blocks (F6 fallback)."""
    per_stream = max(2, int(np.ceil(target_blocks / max(len(units), 1))))
    blocks: list[ClusterData] = []
    for vals, cens in units:
        for v_blk, c_blk in zip(np.array_split(vals, per_stream),
                                np.array_split(cens, per_stream)):
            if v_blk.size:
                blocks.append((v_blk, c_blk))
    return blocks


def cluster_bootstrap_bands(
    clusters: Sequence[ClusterData],
    n_boot: int = 200,
    seed: int = 0,
    grid: Sequence[int] = tuple(SPAN_GRID),
    min_clusters: int = 5,
    target_blocks: int = 10,
    quantiles: tuple[float, float] = (5.0, 95.0),
) -> SurvivalBands:
    """Cluster-bootstrap survival bands (spec §3.2, F6).

    Resamples whole streams with replacement (or contiguous signal-blocks
    per stream when stream count < ``min_clusters``), refits KM per
    resample, and takes the ``quantiles`` percentile envelope at the grid.

    Args:
        clusters: Per-stream ``(values, censored)`` survival inputs.
        n_boot: Number of resamples (>= 200 per spec).
        seed: RNG seed — bands are reproducible under a fixed seed.

    Returns:
        ``SurvivalBands`` with the envelope and the pooled point table.
    """
    units: list[ClusterData] = [
        (np.asarray(v, dtype=float), np.asarray(c, dtype=bool))
        for v, c in clusters
    ]
    method = "stream"
    if len(units) < min_clusters:
        method = "block"
        units = _block_units(units, target_blocks)
    if not units:
        raise ValueError("cluster_bootstrap_bands: no non-empty clusters")

    all_vals = np.concatenate([u[0] for u in units])
    all_cens = np.concatenate([u[1] for u in units])
    s_point = km_table(fit_km(all_vals, all_cens), grid)

    rng = np.random.default_rng(seed)
    m = len(units)
    boot = np.empty((n_boot, len(grid)), dtype=float)
    for b in range(n_boot):
        pick = rng.integers(0, m, size=m)
        vals = np.concatenate([units[j][0] for j in pick])
        cens = np.concatenate([units[j][1] for j in pick])
        boot[b] = km_table(fit_km(vals, cens), grid)
    s_lo = np.percentile(boot, quantiles[0], axis=0)
    s_hi = np.percentile(boot, quantiles[1], axis=0)
    logger.info(
        "cluster_bootstrap_bands: method=%s units=%d n_boot=%d seed=%d",
        method, m, n_boot, seed)
    return SurvivalBands(
        grid=tuple(int(g) for g in grid), s_lo=s_lo, s_hi=s_hi,
        s_point=s_point, n_boot=n_boot, method=method,
    )


__all__ = [
    "ClusterData", "KMSurvival", "SurvivalBands",
    "right_span", "fit_km", "km_at", "km_table", "cluster_bootstrap_bands",
]
