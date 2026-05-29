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


def _run_fold_loop(
    stream_datas: list[StreamData],
    ref_idx: pd.DatetimeIndex,
    is_fraction: float,
    oos_fraction: float,
    step_fraction: float,
    holdout_fraction: float,
) -> list[Fold]:
    """Inner fold-building loop over a (possibly era-restricted) ref_idx."""
    n = len(ref_idx)
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
    return folds


def build_calendar_folds(
    stream_datas: list[StreamData],
    is_fraction: float = IS_FRACTION,
    oos_fraction: float = OOS_FRACTION,
    step_fraction: float = STEP_FRACTION,
    holdout_fraction: float = HOLDOUT_FRACTION,
    start: str | None = None,
    min_streams: int | None = None,
    coverage: float = 0.5,
) -> list[Fold]:
    """Time-aligned calendar folds sized in BARS on the reference stream.

    By default, folds are concentrated in the **common era** — the period
    where at least ``eff_min`` streams coexist — so that nearly every fold is
    multi-asset.  This eliminates the SPX-only-era dilution that occurs when
    the longest stream pre-dates all others by decades.

    Parameters
    ----------
    stream_datas:
        Pool of loaded streams.
    is_fraction, oos_fraction, step_fraction, holdout_fraction:
        Existing bar-fraction parameters (unchanged semantics).
    start:
        If given, force ``era_start`` to this date string (e.g. ``"2013-01-01"``),
        bypassing the automatic common-era detection.
    min_streams:
        Minimum number of streams that must be live before the era starts.
        Defaults to ``max(2, round(coverage * len(stream_datas)))``.
    coverage:
        Fraction of streams that must coexist to define the common era.
        Used only when ``min_streams`` is ``None``.  Default 0.5.

    Notes
    -----
    Auto-fallback (regression guard): if the era restriction yields 0 folds
    and ``start`` was not explicitly supplied, the function retries with the
    full history (``era_start`` set to the earliest stream start) to guarantee
    it never returns fewer folds than the old full-history behaviour.

    The embargo, MIN_STREAM_BARS gate, add_pivot_labels, and build_detector_artifacts
    calls are unchanged.
    """
    if not stream_datas:
        return []

    # ------------------------------------------------------------------
    # Step 1 — determine era_start
    # ------------------------------------------------------------------
    starts_all = sorted(
        sd.df.index[0] for sd in stream_datas if len(sd.df) > 0
    )
    if not starts_all:
        return []

    start_explicit = start is not None
    eff_min_label: str  # for logging
    era_end: pd.Timestamp | None  # None means no upper clip
    if start_explicit:
        era_start = pd.Timestamp(start)
        era_end = None
        eff_min_label = "override"
    else:
        eff_min = min_streams if min_streams is not None else max(
            2, round(coverage * len(stream_datas))
        )
        eff_min = min(eff_min, len(stream_datas))
        eff_min_label = str(eff_min)
        # era_start = the date the eff_min-th stream begins (0-indexed: [eff_min-1])
        era_start = starts_all[eff_min - 1]
        # era_end = last date when at least eff_min streams are still alive.
        # Sort all stream ends descending; the (eff_min-1)-th entry is the latest
        # date at which at least eff_min streams have not yet ended.
        ends_all = sorted(
            (sd.df.index[-1] for sd in stream_datas if len(sd.df) > 0),
            reverse=True,
        )
        era_end = ends_all[eff_min - 1] if len(ends_all) >= eff_min else None

    # ------------------------------------------------------------------
    # Step 2 — choose reference as the stream with most bars in common era
    # ------------------------------------------------------------------
    def _era_bar_count(sd: StreamData) -> int:
        mask = sd.df.index >= era_start
        if era_end is not None:
            mask = mask & (sd.df.index <= era_end)
        return int(mask.sum())

    ref = max(stream_datas, key=_era_bar_count)
    mask_ref = ref.df.index >= era_start
    if era_end is not None:
        mask_ref = mask_ref & (ref.df.index <= era_end)
    ref_idx = ref.df.index[mask_ref]
    n = len(ref_idx)

    # ------------------------------------------------------------------
    # Step 3 — run fold loop on era-restricted ref_idx
    # ------------------------------------------------------------------
    folds = _run_fold_loop(
        stream_datas, ref_idx,
        is_fraction, oos_fraction, step_fraction, holdout_fraction,
    )

    # ------------------------------------------------------------------
    # Step 4 — auto-fallback: if era restriction produced 0 folds and
    # start was NOT explicitly overridden, retry with full history.
    # ------------------------------------------------------------------
    if len(folds) == 0 and not start_explicit and era_start > starts_all[0]:
        logger.info(
            "build_calendar_folds: era_start=%s produced 0 folds; "
            "falling back to full history (era_start=%s)",
            era_start, starts_all[0],
        )
        era_start = starts_all[0]
        ref_full = max(stream_datas, key=lambda sd: len(sd.df))
        ref_idx_full = ref_full.df.index
        folds = _run_fold_loop(
            stream_datas, ref_idx_full,
            is_fraction, oos_fraction, step_fraction, holdout_fraction,
        )

    logger.info(
        "build_calendar_folds: %d folds "
        "(era_start=%s, eff_min=%s, ref=%s, n_ref_bars=%d)",
        len(folds), era_start, eff_min_label, ref.stream.stream_id, n,
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
