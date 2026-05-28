"""Calendar-based pooled folds + pooled Optuna objective (multi-asset).

Keeps ``src/validation.py`` (single-asset, row-index folds) untouched. This
module adds the time-aligned, cluster-weighted pooled path used by Run 5.
"""
from __future__ import annotations

import dataclasses
import logging
from collections.abc import Callable

import numpy as np
import optuna
import pandas as pd

from .detector import SpeculatorDetector, build_detector_artifacts
from .indicators import Params
from .pooled_scoring import StreamStat, pooled_fold_score
from .scoring import (
    REFERENCE_N, STRUCTURAL_NEST, add_pivot_labels, precision_at_n_stats,
)
from .universe import Stream
from .validation import fold_scores_bootstrap_ci, load_data
from .volume_quality import REAL_FLOOR, VolumeQuality, profile_volume

logger = logging.getLogger(__name__)

EMBARGO_NEST_BARS: int = max(STRUCTURAL_NEST)    # 200 — label look-ahead guard
MIN_STREAM_BARS: int = 2 * max(STRUCTURAL_NEST) + 1   # 401 — nest must fit
HOLDOUT_FRACTION: float = 0.20
IS_FRACTION: float = 0.10
OOS_FRACTION: float = 0.03
STEP_FRACTION: float = 0.05


def load_stream_frame(path: str) -> pd.DataFrame:
    """Load a stream CSV via the canonical loader (lowercases, indexes time)."""
    return load_data(path)


def apply_volume_policy(
    df: pd.DataFrame, policy: str,
) -> tuple[pd.DataFrame, bool]:
    """Apply the run-level volume policy to a single stream.

    Returns ``(possibly_trimmed_df, keep)``:
    - ``price_only`` / ``mixed``: keep all bars (volume votes handled by params).
    - ``volume_required``: trim to ``[volume_real_from, end]``; drop the stream
      entirely (``keep=False``) if its quality is ``none``.
    """
    if policy in ("price_only", "mixed"):
        return df, True
    if policy == "volume_required":
        vq: VolumeQuality = profile_volume(df)
        if vq.quality == "none" or vq.real_from is None:
            return df, False
        # real_from marks the first window that is reliably real; advance to
        # the first bar where volume is actually above the real floor so the
        # slice never starts on a straggler placeholder bar.
        candidate = df.loc[df.index >= vq.real_from]
        real_mask = candidate["volume"] >= REAL_FLOOR
        if not real_mask.any():
            return df, False
        first_real = candidate.index[real_mask.to_numpy().argmax()]
        return df.loc[df.index >= first_real].copy(), True
    raise ValueError(f"unknown volume policy {policy!r}")


@dataclasses.dataclass
class StreamData:
    """A loaded, labelled stream plus its bar spacing (seconds)."""
    stream: Stream
    df: pd.DataFrame          # full, pivot-labelled, DatetimeIndex
    bar_seconds: float


@dataclasses.dataclass
class PreparedSlice:
    stream: Stream
    df_is: pd.DataFrame
    df_oos: pd.DataFrame
    artifacts_is: object
    artifacts_oos: object


# One fold = the list of per-stream prepared slices for that calendar window.
Fold = list[PreparedSlice]


def _master_span(stream_datas: list[StreamData]) -> tuple[pd.Timestamp, pd.Timestamp]:
    starts = [sd.df.index[0] for sd in stream_datas if len(sd.df)]
    ends = [sd.df.index[-1] for sd in stream_datas if len(sd.df)]
    return min(starts), max(ends)


def build_calendar_folds(
    stream_datas: list[StreamData],
    is_fraction: float = IS_FRACTION,
    oos_fraction: float = OOS_FRACTION,
    step_fraction: float = STEP_FRACTION,
    holdout_fraction: float = HOLDOUT_FRACTION,
) -> list[Fold]:
    """Time-aligned calendar folds; each stream sliced by date per fold.

    Embargo between IS and OOS is the calendar span of ``EMBARGO_NEST_BARS``
    bars at the COARSEST selected timeframe (guards the non-causal label
    window for every stream). Streams whose slice has < ``MIN_STREAM_BARS``
    bars are dropped from that fold (logged).
    """
    if not stream_datas:
        return []
    start, end = _master_span(stream_datas)
    total_days = max((end - start).days, 1)
    active_days = total_days * (1.0 - holdout_fraction)
    is_days = total_days * is_fraction
    oos_days = total_days * oos_fraction
    step_days = max(total_days * step_fraction, 1.0)

    coarsest_bar_seconds = max(sd.bar_seconds for sd in stream_datas)
    embargo_days = EMBARGO_NEST_BARS * coarsest_bar_seconds / 86400.0

    folds: list[Fold] = []
    is_start_off = 0.0
    while is_start_off + is_days + embargo_days + oos_days <= active_days:
        is_start = start + pd.Timedelta(days=is_start_off)
        is_end = is_start + pd.Timedelta(days=is_days)
        oos_start = is_end + pd.Timedelta(days=embargo_days)
        oos_end = oos_start + pd.Timedelta(days=oos_days)

        fold: Fold = []
        for sd in stream_datas:
            df_is = sd.df.loc[(sd.df.index >= is_start) & (sd.df.index < is_end)].copy()
            df_oos = sd.df.loc[(sd.df.index >= oos_start) & (sd.df.index < oos_end)].copy()
            if len(df_is) < MIN_STREAM_BARS or len(df_oos) < MIN_STREAM_BARS:
                continue   # nest cannot fit / too few bars → drop from this fold
            add_pivot_labels(df_is)
            add_pivot_labels(df_oos)
            fold.append(PreparedSlice(
                stream=sd.stream, df_is=df_is, df_oos=df_oos,
                artifacts_is=build_detector_artifacts(df_is),
                artifacts_oos=build_detector_artifacts(df_oos),
            ))
        if fold:
            folds.append(fold)
        is_start_off += step_days

    logger.info("build_calendar_folds: %d folds over %d master days",
                len(folds), total_days)
    return folds


def cluster_weights(streams: list[Stream]) -> dict[str, float]:
    """Map stream_id -> 1 / (number of streams sharing its cluster)."""
    sizes: dict[str, int] = {}
    for s in streams:
        sizes[s.cluster_id] = sizes.get(s.cluster_id, 0) + 1
    return {s.stream_id: 1.0 / sizes[s.cluster_id] for s in streams}


def _stream_stat(
    df: pd.DataFrame, signals: pd.Series, side: str, weight: float,
) -> StreamStat:
    stats = precision_at_n_stats(signals, df[f"pivot_N{REFERENCE_N}"], side, REFERENCE_N)
    return StreamStat(
        n_signals=int(stats["n_signals"]),
        tp=int(stats["tp"]),
        matched_pivots=int(stats["matched_pivots"]),
        total_pivots=int(stats["total_pivots"]),
        n_bars=len(df),
        weight=weight,
    )


def evaluate_pooled_fold(
    params: Params, side: str, fold: Fold, weights: dict[str, float],
) -> tuple[float, dict[str, float]]:
    """Run the detector per stream in a fold and return the pooled fold score."""
    sig_key = "signal_high" if side == "high" else "signal_low"
    is_stats: list[StreamStat] = []
    oos_stats: list[StreamStat] = []
    for sl in fold:
        w = weights.get(sl.stream.stream_id, 1.0)
        det_is = SpeculatorDetector(sl.df_is, params, sl.artifacts_is).run()
        det_oos = SpeculatorDetector(sl.df_oos, params, sl.artifacts_oos).run()
        is_stats.append(_stream_stat(sl.df_is, det_is[sig_key], side, w))
        oos_stats.append(_stream_stat(sl.df_oos, det_oos[sig_key], side, w))
    return pooled_fold_score(is_stats, oos_stats, side)


def build_pooled_optuna_objective(
    folds: list[Fold],
    streams: list[Stream],
    params_from_trial: Callable[[optuna.Trial, str], Params],
    side: str,
) -> Callable[[optuna.Trial], float]:
    """Pooled multi-asset objective: bootstrap-LCB over pooled fold scores.

    Within-fold correlated streams are neutralised by 1/cluster_size weighting
    in ``pooled_side_score``; temporal overlap between adjacent folds is handled
    by the block bootstrap (block_len=2), same as the single-asset objective.
    """
    if side not in ("high", "low"):
        raise ValueError(f"side must be 'high' or 'low', got {side!r}")
    if not folds:
        raise ValueError("No calendar folds constructed.")
    weights = cluster_weights(streams)

    def objective(trial: optuna.Trial) -> float:
        params = params_from_trial(trial, side)
        fold_scores: list[float] = []
        for fold_idx, fold in enumerate(folds):
            score, _components = evaluate_pooled_fold(params, side, fold, weights)
            fold_scores.append(score)
            running_lcb = fold_scores_bootstrap_ci(
                fold_scores, n_boot=1000, alpha=0.10, block_len=2
            )[0]
            trial.report(running_lcb, fold_idx)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
        return float(fold_scores_bootstrap_ci(
            fold_scores, n_boot=1000, alpha=0.10, block_len=2
        )[0])

    return objective
