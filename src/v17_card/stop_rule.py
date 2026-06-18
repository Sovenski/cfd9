"""Signal Card §3.5 — the F8 stop truth table as pure functions (spec FIXED v3).

LOW side shown; HIGH mirrors with highs and ``>``:

| # | Given                       | When (bar close)              | Then |
|---|-----------------------------|-------------------------------|------|
| 1 | bar t closes, signal fires  | —                             | stop = min(low[t-1], low[t]); i = argmin (tie -> earlier) |
| 2 | active, bar t+1 closes      | close[t+1] >= fire-time stop  | widen once; strict new min -> i = t+1, L recomputed, k restarts; stop FINAL |
| 3 | active, bar t+1 closes      | close[t+1] <  fire-time stop  | INVALIDATED (close-confirmed vs PRE-widening stop); no widening |
| 4 | active, bar u > t+1 closes  | low[u] < stop (intrabar)      | INVALIDATED at u; k freezes at u-i |
| 5 | active, bar u > t+1 closes  | low[u] >= stop                | k = u-i |
| 6 | active, same-side re-fire   | —                             | old card freezes; state resets per row 1; plotted stop switches on the fire bar |

The k-origin is the candidate pivot bar ``i`` (re-review R2), NEVER the
fire bar — calibration (right-span R) and the live counter share it
exactly. All functions are pure: states are frozen dataclasses.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from typing import Optional, Sequence, Union

import numpy as np

from ..scoring_v5 import compute_left_span

logger = logging.getLogger(__name__)

ArrayLike = Union[Sequence[float], np.ndarray]


@dataclass(frozen=True)
class StopState:
    """Per-signal stop/card state (the Pine ``var`` block mirror, F5)."""

    fire_bar: int
    pivot_bar: int  # candidate pivot bar i — the k-origin (R2)
    stop: float
    fire_stop: float  # PRE-widening stop (the row-3 breach reference)
    left_span: int  # proven left-span L at the current i (F1, causal)
    bars_survived: int  # k = last surviving bar - i (frozen on invalidation)
    last_bar: int
    intact: bool
    final: bool  # stop FINAL once bar t+1 is processed
    invalidated_at: Optional[int]
    is_high: bool


def init_stop_state(prices: ArrayLike, t: int, is_high: bool = False) -> StopState:
    """Row 1 — fire-time stop and candidate pivot bar.

    Args:
        prices: Per-bar lows (LOW side) or highs (HIGH side).
        t: Fire bar (must have a preceding bar).

    Returns:
        Active ``StopState`` with ``stop = min(low[t-1], low[t])`` (HIGH:
        max of highs), ``i = argmin`` over ``{t-1, t}`` (tie -> the EARLIER
        bar, consistent with §1.1 plateau-collapse-to-first) and ``L``
        computed causally at ``i``.
    """
    arr = np.asarray(prices, dtype=float)
    if t < 1 or t >= arr.shape[0]:
        raise ValueError(f"fire bar t={t} needs a preceding bar in range")
    prev_v, fire_v = arr[t - 1], arr[t]
    if is_high:
        stop = max(prev_v, fire_v)
        pivot_bar = t - 1 if prev_v >= fire_v else t
    else:
        stop = min(prev_v, fire_v)
        pivot_bar = t - 1 if prev_v <= fire_v else t
    left = compute_left_span(arr, pivot_bar, is_high)
    return StopState(
        fire_bar=t, pivot_bar=pivot_bar, stop=float(stop),
        fire_stop=float(stop), left_span=left, bars_survived=t - pivot_bar,
        last_bar=t, intact=True, final=False, invalidated_at=None,
        is_high=is_high,
    )


def update_stop_state(
    state: StopState, prices: ArrayLike, closes: ArrayLike, u: int,
) -> StopState:
    """Rows 2-5 — advance the stop state by exactly one bar close.

    Row ordering at ``t+1`` is load-bearing (F8): the breach check runs
    against the PRE-widening fire-time stop FIRST and is close-based; only
    a surviving close widens (an intrabar wick below the stop at ``t+1`` is
    hypothesis-consistent — the wiggle bar). From ``t+2`` onward the stop
    is FINAL and any intrabar tick through it invalidates.
    """
    if u != state.last_bar + 1:
        raise ValueError(
            f"bars must be processed sequentially: got {u}, "
            f"expected {state.last_bar + 1}")
    if not state.intact:
        return replace(state, last_bar=u)  # frozen card (rows 3/4 aftermath)

    arr = np.asarray(prices, dtype=float)
    cls = np.asarray(closes, dtype=float)
    high = state.is_high

    if u == state.fire_bar + 1:
        breached = (cls[u] > state.fire_stop) if high else (cls[u] < state.fire_stop)
        if breached:  # row 3 — close-confirmed vs the PRE-widening stop
            return replace(
                state, intact=False, final=True, invalidated_at=u,
                bars_survived=u - state.pivot_bar, last_bar=u)
        # row 2 — survived: widen once; strict new extreme shifts i (R2)
        p = arr[u]
        new_stop = max(state.stop, p) if high else min(state.stop, p)
        is_new_extreme = (p > arr[state.pivot_bar]) if high \
            else (p < arr[state.pivot_bar])
        if is_new_extreme:
            pivot_bar = u
            left = compute_left_span(arr, u, high)  # L recomputed, causal
        else:
            pivot_bar = state.pivot_bar  # tie -> i keeps the earlier bar
            left = state.left_span
        return replace(
            state, stop=float(new_stop), pivot_bar=pivot_bar,
            left_span=left, bars_survived=u - pivot_bar, final=True,
            last_bar=u)

    # rows 4/5 — final stop, intrabar breach test
    breached = (arr[u] > state.stop) if high else (arr[u] < state.stop)
    if breached:  # row 4
        return replace(state, intact=False, invalidated_at=u,
                       bars_survived=u - state.pivot_bar, last_bar=u)
    return replace(state, bars_survived=u - state.pivot_bar, last_bar=u)  # row 5


def track_signal(
    prices: ArrayLike, closes: ArrayLike, t: int, end: int,
    is_high: bool = False,
) -> list[StopState]:
    """States for bars ``t..end`` of one signal (rows 1-5; no re-fire)."""
    states = [init_stop_state(prices, t, is_high)]
    for u in range(t + 1, end + 1):
        states.append(update_stop_state(states[-1], prices, closes, u))
    return states


def stop_series(
    prices: ArrayLike, closes: ArrayLike, fire_bars: Sequence[int],
    is_high: bool = False,
) -> np.ndarray:
    """Per-bar plotted stop with the row-6 same-side re-fire rule.

    On a re-fire the active card freezes (grade pending, §3.7) and the
    whole state block resets per row 1; the plotted stop switches to the
    new signal's stop ON ITS FIRE BAR — the engine implements the
    IDENTICAL replacement rule so §4.2 bar-for-bar stop parity holds
    across overlapping same-side signals.

    DECIDED post-invalidation semantic (F8 / §4.2 parity pin): the plotted
    stop is Pine's natural ``plot(intact ? stop : na)`` evaluated AT BAR
    CLOSE — after a row-3/row-4 invalidation the series is NaN on the
    invalidation bar and every bar after, until a same-side re-fire (row 6)
    resumes plotting. The Pine v17.5 builder MUST emit exactly this idiom;
    the §4.2 bar-for-bar stop parity treats engine-NaN == Pine-na.
    """
    arr = np.asarray(prices, dtype=float)
    out = np.full(arr.shape[0], np.nan)
    fires = set(int(b) for b in fire_bars)
    state: Optional[StopState] = None
    for u in range(arr.shape[0]):
        if u in fires:
            if state is not None:
                logger.debug("stop_series: row-6 re-fire at bar %d "
                             "(old card frozen)", u)
            state = init_stop_state(arr, u, is_high)  # row 6 -> row 1 reset
        elif state is not None and u > state.last_bar:
            state = update_stop_state(state, arr, closes, u)
        if state is not None and state.intact:
            out[u] = state.stop  # plot(intact ? stop : na) — see docstring
    return out


__all__ = [
    "StopState", "init_stop_state", "update_stop_state",
    "track_signal", "stop_series",
]
