"""Signal Card §3.7 — retrospective grading + R-multiple backtest (spec FIXED v3).

Per signal: fire bar, ±1 matched y/n (Hungarian 1:1, same cost surface as
Scorer v5), realized span badge (grid N* / censored bound), realized move
to span end, realized R-multiple under the card's own entry/stop/clock
rules, MAE/MFE. Aggregates: capture ratio, expectancy, win rate — the
backtest that grades the card against reality each run.

Engine-report side only. Overlapping same-side signals are graded
INDEPENDENTLY here (retrospective — no Pine object limits); the row-6
freeze of §3.5 is the live display rule, implemented in ``stop_rule``.
Costs/slippage are ignored and intrabar stop-outs fill AT the stop
(documented limitation, spec §8).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Union

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from ..scoring import _HUNGARIAN_INF, _HUNGARIAN_INF_THRESHOLD, REFERENCE_N
from ..scoring_v5 import MATCH_WINDOW, TIEBREAK_EPS, label_pivot_spans
from .stop_rule import init_stop_state, update_stop_state

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SignalGrade:
    """One signal's retrospective grade row (spec §3.7)."""

    fire_bar: int
    side: str
    matched: bool
    pivot_bar: Optional[int]
    realized_span: int  # grid N* (proven bound when censored); 0 = miss
    span_censored: bool
    tier: str  # display vocabulary only (spec §1.4)
    realized_move: float  # |move to span-end|; NaN when unobservable (R3)
    entry: float
    stop: float  # fire-time stop (risk basis: the wick gap, §3.5)
    risk: float
    exit_bar: int
    exit_price: float
    r_multiple: float
    mae_r: float
    mfe_r: float
    outcome: str  # stopped_close | stopped_intrabar | clock | open


def _tier(span: int) -> str:
    """Display badge per spec §1.4 (NOT an engine concept)."""
    if span >= 200:
        return "T1"
    if span >= 50:
        return "T2"
    if span >= 20:
        return "T3"
    return "miss"


def _assign_matches(
    sig_pos: np.ndarray, piv_pos: np.ndarray, piv_spans: np.ndarray,
) -> dict[int, int]:
    """±1 Hungarian 1:1 signal->pivot assignment (same costs as Scorer v5)."""
    if sig_pos.size == 0 or piv_pos.size == 0:
        return {}
    weights = piv_spans.astype(float) / float(REFERENCE_N)
    diffs = piv_pos[None, :] - sig_pos[:, None]
    cost = np.abs(diffs).astype(float) - TIEBREAK_EPS * weights[None, :]
    cost[np.abs(diffs) > MATCH_WINDOW] = _HUNGARIAN_INF
    rows, cols = linear_sum_assignment(cost)
    return {
        int(sig_pos[r]): int(piv_pos[c])
        for r, c in zip(rows, cols)
        if cost[r, c] < _HUNGARIAN_INF_THRESHOLD
    }


def _simulate_trade(
    prices: np.ndarray, closes: np.ndarray, t: int, clock_bars: int,
    is_high: bool,
) -> tuple[int, float, str, "object"]:
    """Run rows 1-5 from fire to exit; return (exit_bar, exit_price, outcome, state)."""
    n = prices.shape[0]
    state = init_stop_state(prices, t, is_high)
    for u in range(t + 1, n):
        state = update_stop_state(state, prices, closes, u)
        if not state.intact:
            if u == t + 1:  # row 3: close-confirmed -> exit at the close
                return u, float(closes[u]), "stopped_close", state
            return u, float(state.stop), "stopped_intrabar", state  # row 4
        if u >= state.pivot_bar + clock_bars:  # span clock expiry (§3.5)
            return u, float(closes[u]), "clock", state
    last = n - 1
    return last, float(closes[last]), "open", state  # mark-to-market


def grade_signals(
    df: pd.DataFrame,
    signals: Union[pd.Series, np.ndarray],
    side: str = "low",
    clock_bars: int = 100,
) -> list[SignalGrade]:
    """Grade every signal retrospectively (spec §3.7).

    Args:
        df: OHLCV frame (``open/high/low/close`` required).
        signals: Boolean per-bar firing mask for ``side``.
        side: ``"low"`` (long) or ``"high"`` (short, mirrored).
        clock_bars: Clock exit horizon in bars FROM the candidate pivot bar
            ``i`` (the caller supplies E[hold] from the calibrated table).

    Returns:
        One ``SignalGrade`` per signal (chronological).
    """
    if side not in ("low", "high"):
        raise ValueError(f"side must be 'low' or 'high', got {side!r}")
    is_high = side == "high"
    direction = -1.0 if is_high else 1.0
    lows = df["low"].to_numpy(dtype=float)
    highs = df["high"].to_numpy(dtype=float)
    closes = df["close"].to_numpy(dtype=float)
    prices = highs if is_high else lows
    n = prices.shape[0]

    sig_arr = (signals.fillna(False).astype(bool).to_numpy()
               if isinstance(signals, pd.Series)
               else np.asarray(signals, dtype=bool))
    sig_pos = np.flatnonzero(sig_arr)
    if sig_pos.size and sig_pos[0] == 0:
        logger.warning("grade_signals: dropping bar-0 signal (no t-1 bar)")
        sig_pos = sig_pos[1:]

    spans_df = label_pivot_spans(df)
    span_col = spans_df[f"pivot_span_{side}"].to_numpy()
    cens_col = spans_df[f"pivot_span_censored_{side}"].to_numpy()
    piv_pos = np.flatnonzero(span_col > 0)
    matches = _assign_matches(sig_pos, piv_pos, span_col[piv_pos])

    grades: list[SignalGrade] = []
    for t in sig_pos:
        t = int(t)
        entry = float(closes[t])
        exit_bar, exit_price, outcome, state = _simulate_trade(
            prices, closes, t, clock_bars, is_high)
        risk = direction * (entry - state.fire_stop)
        r_mult = (direction * (exit_price - entry) / risk
                  if risk > 0 else float("nan"))
        window_low = lows[t:exit_bar + 1]
        window_high = highs[t:exit_bar + 1]
        if risk > 0:
            mae = ((window_high.max() - entry) if is_high
                   else (entry - window_low.min())) / risk
            mfe = ((entry - window_low.min()) if is_high
                   else (window_high.max() - entry)) / risk
        else:
            mae = mfe = float("nan")

        piv = matches.get(t)
        matched = piv is not None
        span = int(span_col[piv]) if matched else 0
        span_cens = bool(cens_col[piv]) if matched else False
        realized_move = float("nan")
        if matched and not span_cens and piv + span < n:
            realized_move = abs(float(closes[piv + span]) - float(prices[piv]))

        grades.append(SignalGrade(
            fire_bar=t, side=side, matched=matched, pivot_bar=piv,
            realized_span=span, span_censored=span_cens, tier=_tier(span),
            realized_move=realized_move, entry=entry,
            stop=float(state.fire_stop), risk=float(risk),
            exit_bar=exit_bar, exit_price=exit_price, r_multiple=r_mult,
            mae_r=float(mae), mfe_r=float(mfe), outcome=outcome,
        ))
    logger.info("grade_signals: side=%s n=%d matched=%d",
                side, len(grades), sum(g.matched for g in grades))
    return grades


def backtest_summary(grades: list[SignalGrade]) -> dict[str, float]:
    """Aggregate R-multiple backtest of the card rules (spec §3.7).

    ``equity_delta`` is recomputed from the raw entry/exit/risk fields with
    1-risk-unit sizing, so ``total_r == equity_delta`` is the accounting
    identity guarded by the unit test. ``capture_ratio`` = signed realized
    capture over the available |move to span-end| of matched signals.
    """
    traded = [g for g in grades if np.isfinite(g.r_multiple)]
    r_vals = np.array([g.r_multiple for g in traded], dtype=float)
    equity_delta = 0.0
    for g in traded:
        direction = -1.0 if g.side == "high" else 1.0
        equity_delta += direction * (g.exit_price - g.entry) / g.risk

    cap_num = cap_den = 0.0
    for g in traded:
        if g.matched and np.isfinite(g.realized_move):
            direction = -1.0 if g.side == "high" else 1.0
            cap_num += direction * (g.exit_price - g.entry)
            cap_den += g.realized_move

    return {
        "n_signals": float(len(grades)),
        "n_trades": float(len(traded)),
        "win_rate": float(np.mean(r_vals > 0)) if r_vals.size else float("nan"),
        "expectancy": float(r_vals.mean()) if r_vals.size else float("nan"),
        "total_r": float(r_vals.sum()),
        "equity_delta": float(equity_delta),
        "capture_ratio": cap_num / cap_den if cap_den > 0 else float("nan"),
    }


__all__ = ["SignalGrade", "grade_signals", "backtest_summary"]
