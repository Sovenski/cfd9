"""Calibration orchestration — the §5 run-output payload (spec FIXED v3).

Pools all historical signals of a side across the run's streams (§3.2),
fits the right-span KM survival table + F6 cluster-bootstrap bands, the F2
two-fold ``c_side``, the §3.4 span clock, §3.6 conviction breakpoints, and
the §3.7 retrospective grades / R-multiple backtest — and packs everything
into ONE JSON-serializable payload for ``run_v17_gpu`` (``scorer: "v5"``,
``calibration``, ``signal_cards``, ``r_multiple_backtest``,
``calibration_block_hash``).

Implementation decisions (documented, load-bearing):

* ``sigma_HAR`` is the annualization-free per-bar sigma
  ``sqrt(har_forecast)`` — the same Garman-Klass HAR forecast that the
  FROZEN ``calc_har_vol`` computes internally (mirrored read-only here
  because the frozen API only exports the normalized ratio/score).
* The c_side regression response is the RELATIVE |move to span-end|
  (fraction of the pivot price), so one scalar per side pools across
  price scales; the card's expected move is therefore a FRACTION
  (``expected_move_units`` in the fit diagnostics; Pine renders "±z%").
* Non-finite floats are serialized as ``null`` (strict JSON; the §5 run
  JSON must round-trip).

Detection math is FROZEN (spec §0): the detector is only EXECUTED here,
never modified; golden signal arrays stay byte-identical.
"""
from __future__ import annotations

import hashlib
import json
import logging
from typing import Optional, Sequence, Union

import numpy as np
import pandas as pd

from ..scoring_v5 import SPAN_GRID, MATCH_WINDOW
from .conditioning import conditional_survival, expected_hold
from .expected_move import CSideFit, SignalRecord, expected_move, fit_c_side
from .grading import SignalGrade, _assign_matches, backtest_summary, grade_signals
from .stop_rule import init_stop_state, update_stop_state
from .survival import ClusterData, cluster_bootstrap_bands, right_span

logger = logging.getLogger(__name__)

SignalMask = Union[pd.Series, np.ndarray]

#: F7 — the exact in-sample honesty sentence (engine report, Pine tooltip,
#: Cell-6 card legend all carry it VERBATIM).
IN_SAMPLE_DISCLAIMER: str = (
    "probabilities are calibrated on this instrument's history; card values "
    "shown on historical bars are in-sample; only forward bars (after the "
    "calibration run date) are out-of-sample"
)

#: F10 — grid-floor truncation bias, stated in every calibration report.
GRID_FLOOR_BIAS_NOTE: str = (
    "span weights and E[hold] are grid-floor biased (conservative, downward):"
    " N* records the largest grid value passed, truncating by at most one"
    " grid step; the top bucket 500 is '500+' (right-censored at the cap)"
)

#: R3 — censored-span exclusion from the c_side fit truncates the longest
#: swings (length bias).
C_SIDE_BIAS_NOTE: str = (
    "censored-span matched signals are excluded from the c_side regression;"
    " this truncates the longest swings and biases c_side downward"
)

#: §3.5 stop-rule constants exported to Pine alongside the tables.
STOP_RULE: dict[str, object] = {
    "match_window": MATCH_WINDOW,
    "fire_stop": "min(low[t-1], low[t])",          # HIGH mirrors with max/high
    "widen_once_at_t_plus_1": True,
    "t_plus_1_breach": "close_vs_fire_stop",       # F8 row-3 ordering
    "final_from": "t_plus_2_intrabar",             # rows 4/5
    "k_origin": "candidate_pivot_bar",             # R2
}


def _jsonify(obj: object) -> object:
    """Pure-python JSON payload: numpy scalars unboxed, non-finite -> None."""
    if isinstance(obj, dict):
        return {str(k): _jsonify(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonify(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [_jsonify(v) for v in obj.tolist()]
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        f = float(obj)
        return f if np.isfinite(f) else None
    return obj


def _har_sigma(df: pd.DataFrame) -> np.ndarray:
    """Per-bar HAR sigma ``sqrt(har_forecast)`` (spec §3.1).

    Read-only mirror of the Garman-Klass HAR forecast inside the frozen
    ``calc_har_vol`` (which only exports the normalized ratio/score).
    Units: per-bar log-return sigma — annualization-free.
    """
    high = df["high"].to_numpy(dtype=np.float64)
    low = df["low"].to_numpy(dtype=np.float64)
    close = df["close"].to_numpy(dtype=np.float64)
    open_ = df["open"].to_numpy(dtype=np.float64)
    log_hl = np.clip(np.log(high / low), 1e-10, None)
    log_co = np.clip(np.log(close / open_), -1e10, 1e10)
    gk_var = np.clip(
        0.5 * log_hl ** 2 - (2.0 * np.log(2.0) - 1.0) * log_co ** 2,
        1e-10, None)
    gk = pd.Series(gk_var, index=df.index)
    weekly = gk.rolling(5).mean().fillna(gk)
    monthly = gk.rolling(22).mean().fillna(gk)
    forecast = np.clip(
        0.36 * gk.to_numpy() + 0.28 * weekly.to_numpy()
        + 0.28 * monthly.to_numpy(),
        1e-10, None)
    return np.sqrt(forecast)


def _to_mask(signals: SignalMask) -> np.ndarray:
    if isinstance(signals, pd.Series):
        return signals.fillna(False).astype(bool).to_numpy()
    return np.asarray(signals, dtype=bool)


def build_signal_records(
    df: pd.DataFrame, signals: SignalMask, side: str,
) -> list[SignalRecord]:
    """§3.1/§3.2 calibration rows for one stream's signals.

    EVERY signal enters (matched, unmatched, near-ambiguous alike, R1);
    the right-span ``R`` and proven left-span ``L`` are taken at the
    candidate pivot bar ``i`` AFTER the t+1 widening resolution (R2 — the
    same k-origin the live counter uses). The c_side response is the
    RELATIVE |move to span-end| (fraction of the pivot price).
    """
    from ..scoring_v5 import label_pivot_spans

    is_high = side == "high"
    prices = (df["high"] if is_high else df["low"]).to_numpy(dtype=float)
    closes = df["close"].to_numpy(dtype=float)
    n = prices.shape[0]
    sigma = _har_sigma(df)

    sig_pos = np.flatnonzero(_to_mask(signals))
    if sig_pos.size and sig_pos[0] == 0:
        sig_pos = sig_pos[1:]  # mirrors grade_signals' bar-0 drop

    spans_df = label_pivot_spans(df)
    span_col = spans_df[f"pivot_span_{side}"].to_numpy()
    cens_col = spans_df[f"pivot_span_censored_{side}"].to_numpy()
    piv_pos = np.flatnonzero(span_col > 0)
    matches = _assign_matches(sig_pos, piv_pos, span_col[piv_pos])

    records: list[SignalRecord] = []
    for t in sig_pos:
        t = int(t)
        state = init_stop_state(prices, t, is_high)
        if t + 1 < n:  # resolve the t+1 candidate-pivot shift (R2)
            state = update_stop_state(state, prices, closes, t + 1)
        i = state.pivot_bar
        r_span, r_cens = right_span(prices, i, is_high)

        piv = matches.get(t)
        matched = piv is not None
        span = int(span_col[piv]) if matched else 0
        span_cens = bool(cens_col[piv]) if matched else False
        rel_move = float("nan")
        if matched and not span_cens and piv + span < n:
            pivot_price = float(prices[piv])
            abs_move = abs(float(closes[piv + span]) - pivot_price)
            if pivot_price > 0:
                rel_move = abs_move / pivot_price
        records.append(SignalRecord(
            fire_bar=t, sigma_har=float(sigma[t]),
            left_span=float(state.left_span), right_span=float(r_span),
            right_censored=bool(r_cens), matched=matched,
            span_censored=span_cens, realized_abs_move=rel_move,
        ))
    return records


def _conviction_values(
    records: Sequence[SignalRecord], table: np.ndarray, c_fit: CSideFit,
) -> list[float]:
    """§3.6 raw conviction products ``P(N*_eff>=50|L,k=0) * expected_move``."""
    out: list[float] = []
    for rec in records:
        e_hold = expected_hold(table, k=0.0, left_span=rec.left_span)
        em = expected_move(c_fit, rec.sigma_har, e_hold)
        p50 = conditional_survival(table, 50.0, 0.0, rec.left_span)
        out.append(p50 * em if np.isfinite(em) else float("nan"))
    return out


def _degenerate_payload(side: str, n_streams: int) -> dict:
    """Zero-signal pool: honest empty payload, never a crash."""
    logger.warning("calibrate_run[%s]: 0 signals across %d streams — "
                   "degenerate calibration", side, n_streams)
    calibration = {
        "grid": list(SPAN_GRID), "S_R": None, "S_lo": None, "S_hi": None,
        "band_method": None, "n_boot": 0, "seed": 0,
        "c_side": {"c": None, "r_squared": None, "n_fit": 0,
                   "use_fallback": True, "fallback_median": None},
        "expected_hold_at_fire": None, "clock_bars": 1,
        "conviction_breakpoints": [], "stop_rule": dict(STOP_RULE),
        "n_signals": 0, "n_streams": n_streams, "degenerate": True,
        "fit_diagnostics": _fit_diagnostics(None, 0, None),
    }
    calibration = _jsonify(calibration)
    return {
        "scorer": "v5", "side": side, "calibration": calibration,
        "calibration_block_hash": _hash_block(calibration),
        "signal_cards": [],
        "r_multiple_backtest": _jsonify({**backtest_summary([]),
                                         "costs_ignored": True}),
    }


def _fit_diagnostics(r2: Optional[float], censored_excluded_n: int,
                     band_method: Optional[str]) -> dict:
    """The honesty block: F6 band method, F10/R3 bias notes, F7 disclaimer."""
    return {
        "r2": r2,
        "censored_excluded_n": int(censored_excluded_n),
        "band_method": band_method,
        "grid_floor_bias_note": GRID_FLOOR_BIAS_NOTE,
        "c_side_bias_note": C_SIDE_BIAS_NOTE,
        "in_sample_disclaimer": IN_SAMPLE_DISCLAIMER,
        "conditional_on_match": True,          # R3 — card legend label
        "expected_move_units": "fraction_of_pivot_price",
    }


def _hash_block(calibration_jsonified: dict) -> str:
    """sha256 of the canonical calibration JSON (§9 trace field)."""
    blob = json.dumps(calibration_jsonified, sort_keys=True,
                      allow_nan=False).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def calibrate_run(
    frames: dict[str, pd.DataFrame],
    signals: dict[str, SignalMask],
    side: str,
    seed: int = 0,
    n_boot: int = 200,
) -> dict:
    """Full §5 calibration payload for one side of a run.

    Args:
        frames: Stream-id -> OHLCV frame (the run's full stream histories).
        signals: Stream-id -> per-bar boolean firing mask for ``side``
            (produced by the FROZEN detector with the winner Params).
        side: ``"high"`` or ``"low"``.
        seed: F6 cluster-bootstrap seed (bands reproducible under it).
        n_boot: Bootstrap resamples (>= 200 per spec for real runs).

    Returns:
        JSON-serializable dict: ``scorer``, ``side``, ``calibration``,
        ``calibration_block_hash``, ``signal_cards``,
        ``r_multiple_backtest``.
    """
    if side not in ("high", "low"):
        raise ValueError(f"side must be 'high'|'low', got {side!r}")
    if set(frames) != set(signals):
        raise ValueError(
            f"stream keys mismatch: frames={sorted(frames)} "
            f"signals={sorted(signals)}")

    stream_ids = sorted(frames)
    records_by_stream = {
        sid: build_signal_records(frames[sid], signals[sid], side)
        for sid in stream_ids
    }
    all_records = [r for sid in stream_ids for r in records_by_stream[sid]]
    if not all_records:
        return _degenerate_payload(side, len(stream_ids))

    # §3.2 — pooled KM on right-span R + F6 cluster-bootstrap bands.
    clusters: list[ClusterData] = [
        (np.array([r.right_span for r in recs], dtype=float),
         np.array([r.right_censored for r in recs], dtype=bool))
        for sid in stream_ids
        if (recs := records_by_stream[sid])
    ]
    bands = cluster_bootstrap_bands(clusters, n_boot=n_boot, seed=seed)
    table = bands.s_point

    # §3.3 — F2 two-fold c_side (R3 exclusions counted for the report).
    c_fit = fit_c_side(all_records)
    censored_excluded = sum(
        1 for r in all_records
        if r.matched and (r.span_censored or not np.isfinite(r.realized_abs_move)))

    # §3.4 — the span clock at fire (k=0, L unclamped -> generic horizon).
    e_hold0 = expected_hold(table, k=0.0)
    clock_bars = max(1, int(round(e_hold0)))

    # §3.6 — conviction breakpoints (deciles of the historical products).
    conv_values = _conviction_values(all_records, table, c_fit)
    finite_conv = np.asarray([v for v in conv_values if np.isfinite(v)])
    breakpoints = (np.percentile(finite_conv, np.arange(0, 101, 10)).tolist()
                   if finite_conv.size else [])
    conv_pct: list[Optional[float]] = []
    for v in conv_values:
        if np.isfinite(v) and finite_conv.size:
            less = float(np.count_nonzero(finite_conv < v))
            eq = float(np.count_nonzero(finite_conv == v))
            conv_pct.append(100.0 * (less + 0.5 * eq) / finite_conv.size)
        else:
            conv_pct.append(None)

    # §3.7 — grades under the card's own entry/stop/clock rules.
    cards: list[dict] = []
    grades: list[SignalGrade] = []
    rec_iter = iter(zip(conv_pct, all_records))
    for sid in stream_ids:
        stream_grades = grade_signals(frames[sid], signals[sid], side=side,
                                      clock_bars=clock_bars)
        grades.extend(stream_grades)
        for g in stream_grades:
            conviction, rec = next(rec_iter)
            assert rec.fire_bar == g.fire_bar, "record/grade misalignment"
            cards.append({**g.__dict__, "stream": sid,
                          "conviction": conviction})

    calibration = _jsonify({
        "grid": list(SPAN_GRID),
        "S_R": table, "S_lo": bands.s_lo, "S_hi": bands.s_hi,
        "band_method": bands.method, "n_boot": bands.n_boot, "seed": seed,
        "c_side": {"c": c_fit.c_side, "r_squared": c_fit.r_squared,
                   "n_fit": c_fit.n_fit, "use_fallback": c_fit.use_fallback,
                   "fallback_median": c_fit.fallback_median},
        "expected_hold_at_fire": e_hold0, "clock_bars": clock_bars,
        "conviction_breakpoints": breakpoints,
        "stop_rule": dict(STOP_RULE),
        "n_signals": len(all_records), "n_streams": len(stream_ids),
        "degenerate": False,
        "fit_diagnostics": _fit_diagnostics(
            c_fit.r_squared, censored_excluded, bands.method),
    })
    payload = {
        "scorer": "v5", "side": side, "calibration": calibration,
        "calibration_block_hash": _hash_block(calibration),
        "signal_cards": _jsonify(cards),
        "r_multiple_backtest": _jsonify({**backtest_summary(grades),
                                         "costs_ignored": True}),
    }
    logger.info(
        "calibrate_run[%s]: %d signals / %d streams, band=%s, c_side=%s, "
        "clock=%d bars, hash=%s", side, len(all_records), len(stream_ids),
        bands.method, f"{c_fit.c_side:.4f}" if np.isfinite(c_fit.c_side)
        else "fallback", clock_bars, payload["calibration_block_hash"][:12])
    return payload


def calibrate_for_run(
    stream_datas: Sequence[object],
    params: object,
    side: str,
    seed: int = 0,
    n_boot: int = 200,
    artifacts_cache: Optional[dict[str, object]] = None,
) -> dict:
    """Runner glue: FROZEN detector signals for the winner -> calibration.

    Executes ``SpeculatorDetector`` (read-only use of the frozen detection
    path) with the winner ``params`` over each stream's FULL history, then
    calibrates. ``artifacts_cache`` (stream-id keyed) lets ``run_v17_gpu``
    reuse the Params-independent artifacts across both sides.
    """
    from ..detector import SpeculatorDetector, build_detector_artifacts

    frames: dict[str, pd.DataFrame] = {}
    masks: dict[str, np.ndarray] = {}
    for sd in stream_datas:
        sid = sd.stream.stream_id
        if artifacts_cache is not None and sid in artifacts_cache:
            art = artifacts_cache[sid]
        else:
            art = build_detector_artifacts(sd.df)
            if artifacts_cache is not None:
                artifacts_cache[sid] = art
        res = SpeculatorDetector(sd.df, params, art).run()
        frames[sid] = sd.df
        masks[sid] = res[f"signal_{side}"].to_numpy()
    return calibrate_run(frames, masks, side, seed=seed, n_boot=n_boot)


__all__ = [
    "IN_SAMPLE_DISCLAIMER", "GRID_FLOOR_BIAS_NOTE", "C_SIDE_BIAS_NOTE",
    "STOP_RULE", "build_signal_records", "calibrate_run", "calibrate_for_run",
]
