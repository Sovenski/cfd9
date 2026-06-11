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

from .cpcv import CPCVConfig, build_cpcv_splits, reconstruct_paths
from .detector import SpeculatorDetector, build_detector_artifacts
from .indicators import Params
from .pooled_scoring import StreamStat, pooled_fold_score, pooled_side_score
from .scoring import (
    REFERENCE_N, STRUCTURAL_NEST, add_pivot_labels, precision_at_n_stats,
)
from .scoring_v5 import match_signals_weighted
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

#: Spec §A3 (pre-committed): the holdout score must reach at least this
#: fraction of the winner's MEAN raw informative fold score to pass.
#: 0.5 is a generalization floor — the holdout may be one regime; demanding
#: parity with the search mean would be stricter than the fold-to-fold
#: variance justifies.
ERA_PASS_MIN_RATIO: float = 0.5


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


def _prep_slice_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, object] | None:
    """Shared per-stream slice prep (fold AND holdout path, spec §A1).

    MIN_STREAM_BARS gate -> ``reset_index`` -> per-slice ``add_pivot_labels``
    (standalone warm-up inside the slice — the existing leak guard) ->
    ``build_detector_artifacts``. Factored verbatim out of the fold loop so
    ``build_holdout_slices`` cannot drift from it; returns ``None`` when the
    nest cannot fit.
    """
    if len(df) < MIN_STREAM_BARS:
        return None
    df_r = df.reset_index(drop=True)
    add_pivot_labels(df_r)
    return df_r, build_detector_artifacts(df_r)


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
            prep_is = _prep_slice_frame(df_is)
            prep_oos = _prep_slice_frame(df_oos)
            if prep_is is None or prep_oos is None:  # pragma: no cover - gated above
                continue
            df_is_r, artifacts_is = prep_is
            df_oos_r, artifacts_oos = prep_oos
            fold.append(PreparedSlice(
                stream=sd.stream, df_is=df_is_r, df_oos=df_oos_r,
                artifacts_is=artifacts_is,
                artifacts_oos=artifacts_oos,
            ))
        if fold:
            folds.append(fold)
        is_start += step_bars
    return folds


@dataclasses.dataclass(frozen=True)
class CPCVFoldMeta:
    """Provenance of one CPCV fold: which split/test group/OOS path it is."""

    split_id: int
    test_group: int
    path_id: int
    is_range: tuple[int, int]    # [start, end) offsets into the ref index
    test_range: tuple[int, int]


def _slice_by_range(
    df: pd.DataFrame, ref_idx: pd.DatetimeIndex, rng: tuple[int, int],
) -> pd.DataFrame:
    """Calendar slice of ``df`` covering ref-index offsets ``[start, end)``."""
    a, b = rng
    return df.loc[(df.index >= ref_idx[a]) & (df.index <= ref_idx[b - 1])]


def _run_cpcv_loop(
    stream_datas: list[StreamData],
    ref_idx: pd.DatetimeIndex,
    config: CPCVConfig | None = None,
    holdout_fraction: float = HOLDOUT_FRACTION,
) -> tuple[list[Fold], list[CPCVFoldMeta]]:
    """CPCV fold-building loop ALONGSIDE ``_run_fold_loop`` (spec §4).

    DEFAULT OFF: nothing in the default pipeline calls this — calendar
    walk-forward folds remain the default everywhere.

    One fold is materialized per (split, test group). Per stream, ``df_is``
    is the LONGEST contiguous purged train range (the detector requires
    contiguous bars) and ``df_oos`` is the test group. Slice preparation
    (MIN_STREAM_BARS gate, reset_index, ``add_pivot_labels``,
    ``build_detector_artifacts``) mirrors ``_run_fold_loop`` exactly, so
    per-slice label recomputation — the existing leak guard — is unchanged.
    """
    config = config or CPCVConfig()
    n = len(ref_idx)
    holdout_bars = int(round(n * holdout_fraction))
    active = n - holdout_bars
    if active < config.n_groups * MIN_STREAM_BARS:
        logger.info("_run_cpcv_loop: %d active bars too short for %d groups",
                    active, config.n_groups)
        return [], []
    splits = build_cpcv_splits(active, config)
    paths = reconstruct_paths(splits)
    path_of = {cell: pid for pid, path in enumerate(paths) for cell in path}

    folds: list[Fold] = []
    metas: list[CPCVFoldMeta] = []
    for sp in splits:
        if not sp.train_ranges:
            continue
        is_range = max(sp.train_ranges, key=lambda r: r[1] - r[0])
        for g, test_range in zip(sp.test_groups, sp.test_ranges):
            fold: Fold = []
            for sd in stream_datas:
                df_is = _slice_by_range(sd.df, ref_idx, is_range)
                df_oos = _slice_by_range(sd.df, ref_idx, test_range)
                if len(df_is) < MIN_STREAM_BARS or len(df_oos) < MIN_STREAM_BARS:
                    continue
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
                metas.append(CPCVFoldMeta(
                    split_id=sp.split_id, test_group=g,
                    path_id=path_of[(sp.split_id, g)],
                    is_range=is_range, test_range=test_range))
    logger.info("_run_cpcv_loop: %d folds from %d splits (%d OOS paths)",
                len(folds), len(splits), len(paths))
    return folds, metas


def build_cpcv_folds(
    stream_datas: list[StreamData],
    config: CPCVConfig | None = None,
    holdout_fraction: float = HOLDOUT_FRACTION,
) -> tuple[list[Fold], list[CPCVFoldMeta]]:
    """Opt-in purged-CPCV entry point (spec §4; default pipeline unaffected).

    The reference index is the stream with the most bars, mirroring the
    full-history fallback of ``build_calendar_folds``.
    """
    if not stream_datas:
        return [], []
    ref = max(stream_datas, key=lambda sd: len(sd.df))
    return _run_cpcv_loop(stream_datas, ref.df.index, config, holdout_fraction)


@dataclasses.dataclass(frozen=True)
class EraRef:
    """Resolved common-era reference index (shared by folds + holdout)."""

    ref_idx: pd.DatetimeIndex
    era_start: pd.Timestamp
    eff_min_label: str
    ref_stream_id: str
    starts_all: tuple


def _resolve_era_ref_index(
    stream_datas: list[StreamData],
    start: str | None = None,
    min_streams: int | None = None,
    coverage: float = 0.5,
) -> EraRef | None:
    """Steps 1+2 of ``build_calendar_folds``, factored out (no behavior change).

    Determines ``era_start``/``era_end`` and the era-restricted reference
    index. Reused by ``build_holdout_slices`` (spec §A1) so the holdout
    boundary sits on the SAME reference index the folds were built on.
    """
    starts_all = sorted(
        sd.df.index[0] for sd in stream_datas if len(sd.df) > 0
    )
    if not starts_all:
        return None

    era_end: pd.Timestamp | None  # None means no upper clip
    if start is not None:
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

    def _era_bar_count(sd: StreamData) -> int:
        mask = sd.df.index >= era_start
        if era_end is not None:
            mask = mask & (sd.df.index <= era_end)
        return int(mask.sum())

    ref = max(stream_datas, key=_era_bar_count)
    mask_ref = ref.df.index >= era_start
    if era_end is not None:
        mask_ref = mask_ref & (ref.df.index <= era_end)
    return EraRef(
        ref_idx=ref.df.index[mask_ref], era_start=era_start,
        eff_min_label=eff_min_label, ref_stream_id=ref.stream.stream_id,
        starts_all=tuple(starts_all),
    )


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
    # Steps 1+2 — era_start + era-restricted reference index (factored into
    # _resolve_era_ref_index, shared with build_holdout_slices — spec §A1)
    # ------------------------------------------------------------------
    era = _resolve_era_ref_index(stream_datas, start, min_streams, coverage)
    if era is None:
        return []
    start_explicit = start is not None
    era_start = era.era_start
    starts_all = era.starts_all
    ref_idx = era.ref_idx
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
        len(folds), era_start, era.eff_min_label, era.ref_stream_id, n,
    )
    return folds


# ---------------------------------------------------------------------------
# Spec §A — holdout slices (selection-untouched OOS) + era_pass rule
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class HoldoutSlice:
    """One stream's reserved-tail holdout slice (PreparedSlice analog).

    OOS-only by construction: the slice starts after the active boundary
    plus the embargo; ``start_ts``/``end_ts`` carry the original timestamps
    (the frame itself is reset_index'd like every fold slice).
    """

    stream: Stream
    df: pd.DataFrame          # reset_index'd, per-slice pivot-labelled
    artifacts: object
    start_ts: pd.Timestamp
    end_ts: pd.Timestamp


def build_holdout_slices(
    stream_datas: list[StreamData],
    is_fraction: float = IS_FRACTION,
    oos_fraction: float = OOS_FRACTION,
    step_fraction: float = STEP_FRACTION,
    holdout_fraction: float = HOLDOUT_FRACTION,
    start: str | None = None,
    min_streams: int | None = None,
    coverage: float = 0.5,
) -> tuple[list[HoldoutSlice], dict]:
    """Per-stream slices of the reserved holdout tail (spec §A1).

    ``build_calendar_folds`` reserves the final ``holdout_fraction`` of the
    common-era reference index; folds never touch it. The holdout starts at
    the reference-index timestamp at ``active`` (the first reserved bar)
    PLUS an ``EMBARGO_NEST_BARS`` (200) embargo on the reference index —
    preventing label bleed from the last fold's OOS into the holdout
    (centered span labels reach ±200). Slice prep is the EXACT fold path
    (``_prep_slice_frame``): MIN_STREAM_BARS gate, ``reset_index``,
    per-slice ``add_pivot_labels`` (standalone warm-up — consistency rule),
    fresh detector artifacts.

    ``is_fraction``/``oos_fraction``/``step_fraction`` are accepted so the
    run's ``era_kw`` can be passed through verbatim, but they do not affect
    the boundary — only ``holdout_fraction`` and the era arguments do.

    Returns:
        ``(slices, meta)`` where meta carries the boundary provenance:
        ``holdout_start`` (first reserved reference bar), ``embargo_ts``
        (first scoreable holdout bar), ``embargo_bars``, ``n_ref_bars``,
        ``n_slices`` and ``dropped_streams``.
    """
    del is_fraction, oos_fraction, step_fraction  # boundary-irrelevant (doc)
    meta: dict = {
        "holdout_start": None, "embargo_ts": None,
        "embargo_bars": EMBARGO_NEST_BARS,
        "holdout_fraction": float(holdout_fraction),
        "n_ref_bars": 0, "n_slices": 0, "dropped_streams": [],
    }
    if not stream_datas:
        return [], meta
    era = _resolve_era_ref_index(stream_datas, start, min_streams, coverage)
    if era is None:
        return [], meta
    ref_idx = era.ref_idx
    n = len(ref_idx)
    meta["n_ref_bars"] = n
    holdout_bars = int(round(n * holdout_fraction))
    active = n - holdout_bars
    if holdout_bars <= 0 or active < 0 or active + EMBARGO_NEST_BARS >= n:
        logger.info("build_holdout_slices: no usable holdout "
                    "(n_ref_bars=%d, active=%d, embargo=%d)",
                    n, active, EMBARGO_NEST_BARS)
        return [], meta
    holdout_start = ref_idx[active]
    embargo_ts = ref_idx[active + EMBARGO_NEST_BARS]
    meta["holdout_start"] = holdout_start
    meta["embargo_ts"] = embargo_ts

    slices: list[HoldoutSlice] = []
    for sd in stream_datas:
        df_h = sd.df.loc[sd.df.index >= embargo_ts]
        prep = _prep_slice_frame(df_h)
        if prep is None:
            meta["dropped_streams"].append(sd.stream.stream_id)
            continue
        df_r, artifacts = prep
        slices.append(HoldoutSlice(
            stream=sd.stream, df=df_r, artifacts=artifacts,
            start_ts=df_h.index[0], end_ts=df_h.index[-1]))
    meta["n_slices"] = len(slices)
    logger.info(
        "build_holdout_slices: %d slices (holdout_start=%s, embargo_ts=%s, "
        "embargo=%d bars, ref=%s, dropped=%s)",
        len(slices), holdout_start, embargo_ts, EMBARGO_NEST_BARS,
        era.ref_stream_id, meta["dropped_streams"],
    )
    return slices, meta


def pooled_holdout_score(
    stats: list[StreamStat], side: str,
) -> tuple[float, dict[str, float]]:
    """The OOS leg of ``pooled_fold_score`` on holdout stats (spec §A2).

    Implemented NEXT TO fold scoring without modifying it: delegates to the
    SAME ``pooled_side_score`` composite the OOS leg of ``pooled_fold_score``
    uses (pinned by ``tests/test_holdout.py``), then maps the weighted
    components onto the holdout reporting contract (``precision_w`` /
    ``recall_w`` / ``n_signals`` / ``tp_mass`` / ``total_mass``).
    """
    score, comp = pooled_side_score(stats, side)
    components = {
        "score": float(score),
        "precision_w": float(comp["precision"]),
        "recall_w": float(comp["recall"]),
        "n_signals": float(comp["n_signals"]),
        "tp_mass": float(comp["tp_mass"]),
        "total_mass": float(comp["total_mass"]),
        "n_eff": float(comp["n_eff"]),
        "n_unmatched": float(comp["n_unmatched"]),
        "excess_penalty": float(comp["excess_penalty"]),
        "frequency_factor": float(comp["frequency_factor"]),
        "pooled_total_pivots": float(comp["pooled_total_pivots"]),
    }
    return float(score), components


def holdout_era_pass(
    holdout_score: float,
    fold_scores: list[float],
    min_ratio: float = ERA_PASS_MIN_RATIO,
) -> bool:
    """Pre-committed era_pass rule (spec §A3).

    ``era_pass = holdout_score > 0 AND holdout_score >= min_ratio * mean(raw
    winner fold scores)``. The mean is over the winner's RAW (unpenalized)
    informative fold scores (``v17_acceptance.raw_fold_scores``). With an
    empty basis the mean is taken as 0.0, so any positive holdout score
    passes (the LCB is 0 in that case anyway — the verdict carries no risk).
    """
    score = float(holdout_score)
    if not score > 0.0:
        return False
    fold_mean = float(np.mean(fold_scores)) if fold_scores else 0.0
    return bool(score >= float(min_ratio) * fold_mean)


def evaluate_holdout(
    params: Params,
    side: str,
    slices: list[HoldoutSlice],
    weights: dict[str, float],
) -> tuple[float, dict[str, float], list[dict]]:
    """Run the EXACT CPU detector on the holdout slices and pool (spec §A2).

    Per slice: ``SpeculatorDetector`` -> ``_stream_stat`` (v5 weighted span
    masses, cluster weights) -> ``pooled_holdout_score``. Returns
    ``(score, components, per_stream)`` with per-stream lines sorted by
    weighted matched mass (for the report's holdout tail).
    """
    if side not in ("high", "low"):
        raise ValueError(f"side must be 'high' or 'low', got {side!r}")
    sig_key = "signal_high" if side == "high" else "signal_low"
    stats: list[StreamStat] = []
    per_stream: list[dict] = []
    for sl in slices:
        w = weights.get(sl.stream.stream_id, 1.0)
        det = SpeculatorDetector(sl.df, params, sl.artifacts).run()
        st = _stream_stat(sl.df, det[sig_key], side, w)
        stats.append(st)
        per_stream.append({
            "stream_id": sl.stream.stream_id,
            "n_signals": int(st.n_signals),
            "tp": int(st.tp),
            "tp_mass": float(st.tp_mass),
            "total_mass": float(st.total_mass),
            "n_bars": int(st.n_bars),
            "weight": float(w),
            "start": str(sl.start_ts),
            "end": str(sl.end_ts),
        })
    score, components = pooled_holdout_score(stats, side)
    per_stream.sort(key=lambda e: (-(e["weight"] * e["tp_mass"]),
                                   -e["n_signals"], e["stream_id"]))
    return score, components, per_stream


def cluster_weights(streams: list[Stream]) -> dict[str, float]:
    """Map stream_id -> 1 / (number of streams sharing its cluster)."""
    sizes: dict[str, int] = {}
    for s in streams:
        sizes[s.cluster_id] = sizes.get(s.cluster_id, 0) + 1
    return {s.stream_id: 1.0 / sizes[s.cluster_id] for s in streams}


def _stream_stat(
    df: pd.DataFrame, signals: pd.Series, side: str, weight: float,
    mask_dead_margins: bool = False,
) -> StreamStat:
    """Per-stream Scorer-v5 span-mass match statistics (spec §2.1-§2.3).

    Matches the stream's signals against the side's ``pivot_span_{side}``
    column (±1 direct hit, larger-span tie-break) and carries the weighted
    masses into ``StreamStat`` for pooled aggregation. Count fields are kept
    as diagnostics (1:1 matching ⇒ ``tp == matched_pivots``).
    """
    if mask_dead_margins:
        # Spec §4: the centered structural label window (half-width
        # EMBARGO_NEST_BARS = 200 = max(STRUCTURAL_NEST)) is undefined on the
        # first/last 200 bars of a slice, so those margins can never contain
        # a labelled pivot. Masking removes the dead bars from n_bars and
        # dead-margin signals from the precision denominator. OPT-IN ONLY:
        # the default (False) is byte-identical to the legacy behavior.
        m = EMBARGO_NEST_BARS
        if len(df) <= 2 * m:
            return StreamStat(n_signals=0, tp=0, matched_pivots=0,
                              total_pivots=0, n_bars=0, weight=weight)
        df = df.iloc[m:-m]
        signals = signals.iloc[m:-m]
    spans = df[f"pivot_span_{side}"]
    ws = match_signals_weighted(signals, spans)
    n_matched = int(ws.n_signals - ws.n_unmatched)
    return StreamStat(
        n_signals=int(ws.n_signals),
        tp=n_matched,
        matched_pivots=n_matched,
        total_pivots=int((spans.to_numpy() > 0).sum()),
        n_bars=len(df),
        weight=weight,
        tp_mass=float(ws.tp_mass),
        total_mass=float(ws.total_mass),
        n_unmatched=int(ws.n_unmatched),
    )


def _fold_is_informative(components: dict) -> bool:
    """A fold informs the side objective only if its pooled OOS contains
    any pivot MASS of that side (Scorer v5, spec §2.4 — replaces the v4
    structural-pivot count check; a zero-mass fold's score is a forced zero
    that only dilutes the bootstrap LCB)."""
    return float(components.get("pooled_total_mass_oos", 0.0)) > 0.0


def evaluate_pooled_fold(
    params: Params, side: str, fold: Fold, weights: dict[str, float],
    mask_dead_margins: bool = False,
) -> tuple[float, dict[str, float]]:
    """Run the detector per stream in a fold and return the pooled fold score.

    ``mask_dead_margins`` (spec §4, OPT-IN, default False = legacy behavior)
    excludes the 200-bar dead label margins of each slice from scoring.
    """
    sig_key = "signal_high" if side == "high" else "signal_low"
    is_stats: list[StreamStat] = []
    oos_stats: list[StreamStat] = []
    for sl in fold:
        w = weights.get(sl.stream.stream_id, 1.0)
        det_is = SpeculatorDetector(sl.df_is, params, sl.artifacts_is).run()
        det_oos = SpeculatorDetector(sl.df_oos, params, sl.artifacts_oos).run()
        is_stats.append(_stream_stat(sl.df_is, det_is[sig_key], side, w,
                                     mask_dead_margins=mask_dead_margins))
        oos_stats.append(_stream_stat(sl.df_oos, det_oos[sig_key], side, w,
                                      mask_dead_margins=mask_dead_margins))
    return pooled_fold_score(is_stats, oos_stats, side)


def per_asset_high_diagnostic(
    folds: list[Fold], params: Params, side: str = "high", alpha: float = 0.05,
) -> dict:
    """Per-asset-then-aggregate HIGH diagnostic (spec §4, advisory).

    Runs the EXACT ``SpeculatorDetector`` on every fold's OOS slice and sums
    raw (unweighted) match counts per asset, then across assets. The PRIMARY
    summary for the thin (~27-event) HIGH side is the OOS event count plus a
    wide Wilson interval on OOS precision; PBO and the selection percentile
    are ADVISORY fields only (filled by the batched-search route, never used
    as a gate). ``heterogeneity_flag`` marks the pooled number as a likely
    pooling artifact when pooled precision and the median per-asset precision
    disagree sharply. Counts are summed over fold OOS slices — walk-forward
    OOS windows may overlap; CPCV test groups partition the timeline.
    """
    from .overfit_guard import wilson_interval
    if side not in ("high", "low"):
        raise ValueError(f"side must be 'high' or 'low', got {side!r}")
    sig_key = f"signal_{side}"
    col = f"pivot_N{REFERENCE_N}"
    count_keys = ("oos_events", "n_signals", "tp", "matched_pivots")
    counts: dict[str, dict[str, int]] = {}
    for fold in folds:
        for sl in fold:
            det = SpeculatorDetector(sl.df_oos, params, sl.artifacts_oos).run()
            st = precision_at_n_stats(det[sig_key], sl.df_oos[col], side,
                                      REFERENCE_N)
            acc = counts.setdefault(sl.stream.stream_id,
                                    dict.fromkeys(count_keys, 0))
            acc["oos_events"] += int(st["total_pivots"])
            acc["n_signals"] += int(st["n_signals"])
            acc["tp"] += int(st["tp"])
            acc["matched_pivots"] += int(st["matched_pivots"])

    def _entry(c: dict[str, int]) -> dict:
        precision = c["tp"] / c["n_signals"] if c["n_signals"] > 0 else 0.0
        lo, hi = wilson_interval(c["tp"], c["n_signals"], alpha)
        return {**c, "precision": float(precision),
                "wilson_low": float(lo), "wilson_high": float(hi)}

    per_asset = {sid: _entry(c) for sid, c in sorted(counts.items())}
    totals = {k: sum(c[k] for c in counts.values()) for k in count_keys} \
        if counts else dict.fromkeys(count_keys, 0)
    aggregate = _entry(totals)
    precisions = [e["precision"] for e in per_asset.values()]
    median_prec = float(np.median(precisions)) if precisions else 0.0
    aggregate["median_asset_precision"] = median_prec
    aggregate["heterogeneity_flag"] = bool(
        abs(aggregate["precision"] - median_prec) > 0.25)
    logger.info("per_asset_high_diagnostic[%s]: %d assets, %d OOS events, "
                "pooled precision %.3f [%.3f, %.3f], median-asset %.3f",
                side, len(per_asset), aggregate["oos_events"],
                aggregate["precision"], aggregate["wilson_low"],
                aggregate["wilson_high"], median_prec)
    return {"side": side, "per_asset": per_asset, "aggregate": aggregate,
            "primary": "event_count+wilson_interval",
            "advisory": {"pbo": None, "selection_percentile": None}}


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

    Folds whose pooled OOS contains zero pivot mass of the requested side
    (Scorer v5, spec §2.4) are excluded: their score is a forced zero
    regardless of params and would only pin the bootstrap-LCB near zero
    without providing signal. If no fold is informative, the trial returns
    0.0 without crashing.
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
