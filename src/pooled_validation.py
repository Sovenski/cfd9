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


def build_calendar_folds(
    stream_datas: list[StreamData],
    is_fraction: float = IS_FRACTION,
    oos_fraction: float = OOS_FRACTION,
    step_fraction: float = STEP_FRACTION,
    holdout_fraction: float = HOLDOUT_FRACTION,
) -> list[Fold]:
    """Time-aligned calendar folds sized in BARS on the reference stream.

    The **reference stream** is the one with the most bars; all fold windows
    are expressed as integer bar offsets on that stream and then converted to
    date boundaries used to slice every other stream.  This avoids the
    calendar-day sizing bug where ``OOS_FRACTION * calendar_days`` fails to
    reach ``MIN_STREAM_BARS`` for any pool not anchored on a 155-year series.

    The embargo is exactly ``EMBARGO_NEST_BARS`` reference-stream bars (fixes
    the prior under-shooting caused by scaling ``coarsest_bar_seconds / 86400``).

    Streams whose IS or OOS date-slice has < ``MIN_STREAM_BARS`` bars are
    dropped from that fold (their data is sparser than the reference).
    """
    if not stream_datas:
        return []

    # Reference stream: the one with the most bars.
    ref = max(stream_datas, key=lambda sd: len(sd.df))
    ref_idx = ref.df.index  # sorted DatetimeIndex
    n = len(ref_idx)

    # Bar-based sizes (mirrors src/validation.py single-asset scheme).
    holdout_bars = int(round(n * holdout_fraction))
    active = n - holdout_bars
    is_bars = max(MIN_STREAM_BARS, int(round(n * is_fraction)))
    oos_bars = max(MIN_STREAM_BARS, int(round(n * oos_fraction)))
    step_bars = max(1, int(round(n * step_fraction)))
    embargo_bars = EMBARGO_NEST_BARS  # true 200-bar gap

    folds: list[Fold] = []
    is_start = 0
    while is_start + is_bars + embargo_bars + oos_bars <= active:
        is_s = ref_idx[is_start]
        is_e = ref_idx[is_start + is_bars]
        oos_s = ref_idx[is_start + is_bars + embargo_bars]
        oos_e_off = is_start + is_bars + embargo_bars + oos_bars
        oos_e = ref_idx[oos_e_off]  # oos_e_off <= active <= n-1 by loop guard

        fold: Fold = []
        for sd in stream_datas:
            df_is = sd.df.loc[(sd.df.index >= is_s) & (sd.df.index < is_e)]
            df_oos = sd.df.loc[(sd.df.index >= oos_s) & (sd.df.index < oos_e)]
            if len(df_is) < MIN_STREAM_BARS or len(df_oos) < MIN_STREAM_BARS:
                continue  # nest cannot fit / too few bars → drop from this fold
            df_is_r = df_is.reset_index(drop=True)
            df_oos_r = df_oos.reset_index(drop=True)
            add_pivot_labels(df_is_r)
            add_pivot_labels(df_oos_r)
            fold.append(PreparedSlice(
                stream=sd.stream, df_is=df_is_r, df_oos=df_oos_r,
                artifacts_is=build_detector_artifacts(df_is_r),
                artifacts_oos=build_detector_artifacts(df_oos_r),
            ))
        if fold:
            folds.append(fold)
        is_start += step_bars

    logger.info(
        "build_calendar_folds: %d folds (ref=%s, is=%d oos=%d step=%d embargo=%d bars over %d active)",
        len(folds), ref.stream.stream_id, is_bars, oos_bars, step_bars, embargo_bars, active,
    )
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


def _fold_is_informative(components: dict) -> bool:
    """A fold informs the side objective only if its pooled OOS contains
    at least one structural pivot of that side (else its score is a forced
    zero that only dilutes the bootstrap LCB)."""
    return float(components.get("pooled_total_pivots_oos", 0.0)) > 0.0


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

    Folds whose pooled OOS contains zero structural pivots of the requested
    side are excluded: their score is a forced zero regardless of params and
    would only pin the bootstrap-LCB near zero without providing signal.
    If no fold is informative, the trial returns 0.0 without crashing.
    """
    if side not in ("high", "low"):
        raise ValueError(f"side must be 'high' or 'low', got {side!r}")
    if not folds:
        raise ValueError("No calendar folds constructed.")
    weights = cluster_weights(streams)

    def objective(trial: optuna.Trial) -> float:
        params = params_from_trial(trial, side)
        fold_scores: list[float] = []
        report_step: int = 0
        for fold in folds:
            score, components = evaluate_pooled_fold(params, side, fold, weights)
            if not _fold_is_informative(components):
                continue
            fold_scores.append(score)
            running_lcb = fold_scores_bootstrap_ci(
                fold_scores, n_boot=1000, alpha=0.10, block_len=2
            )[0]
            trial.report(running_lcb, report_step)
            report_step += 1
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
        if not fold_scores:
            return 0.0
        return float(fold_scores_bootstrap_ci(
            fold_scores, n_boot=1000, alpha=0.10, block_len=2
        )[0])

    return objective
