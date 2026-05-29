"""v17 — exact label-aware threshold calibration (the non-search core).

Each detector vote is ``feature (>=|<=) threshold``. Given the structural-pivot
labels, the optimal single cutpoint is found EXACTLY by sweeping the sorted
breakpoints of the feature (no sampling). We score every candidate with a
label objective that exploits the huge negative class (~25k bars) so the cut is
far better determined than the ~100-event LCB would suggest.

Pure numpy; no detector/pipeline dependency, so it is unit-testable in isolation.
The chosen cutpoints SEED the coordinate-ascent in ``v17_optimize`` and remain
plain numbers that plug into the (unchanged) detector — parity preserved.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np

logger = logging.getLogger(__name__)

Direction = Literal["geq", "leq"]
Objective = Literal["fbeta", "youden"]


@dataclass(frozen=True)
class Cutpoint:
    """Result of a single-vote threshold calibration."""

    threshold: float
    score: float
    precision: float
    recall: float
    n_predicted: int


def _score(tp: np.ndarray, cnt: np.ndarray, n_pos: int, n_neg: int,
           objective: Objective, beta: float) -> np.ndarray:
    """Vectorized label objective for arrays of (tp, predicted-count)."""
    tp = tp.astype(np.float64)
    cnt = cnt.astype(np.float64)
    precision = np.divide(tp, cnt, out=np.zeros_like(tp), where=cnt > 0)
    recall = tp / n_pos if n_pos > 0 else np.zeros_like(tp)
    if objective == "youden":
        fp = cnt - tp
        fpr = fp / n_neg if n_neg > 0 else np.zeros_like(tp)
        return recall - fpr
    b2 = beta * beta
    denom = b2 * precision + recall
    return np.divide((1.0 + b2) * precision * recall, denom,
                     out=np.zeros_like(tp), where=denom > 0)


def _scan_geq(feature: np.ndarray, pos: np.ndarray,
              objective: Objective, beta: float,
              min_predicted: int) -> Cutpoint:
    """Exact sweep for the rule ``feature >= threshold``.

    NaN-feature bars can never satisfy ``>= threshold`` (the detector treats
    them as a False vote), so they are excluded from candidate thresholds and
    counted as predicted-negative in recall/precision via the full ``pos`` totals.
    """
    feature = np.asarray(feature, dtype=np.float64)
    pos = np.asarray(pos, dtype=bool)
    n_pos = int(pos.sum())
    n_neg = int((~pos).sum())
    if n_pos == 0:
        return Cutpoint(float("nan"), 0.0, 0.0, 0.0, 0)

    valid = ~np.isnan(feature)
    f = feature[valid]
    y = pos[valid].astype(np.int64)
    if f.size == 0:
        return Cutpoint(float("nan"), 0.0, 0.0, 0.0, 0)

    order = np.argsort(f, kind="mergesort")
    fs = f[order]
    ys = y[order]

    # First index of each unique value: threshold = fs[i] predicts bars i..end.
    first = np.concatenate(([0], np.flatnonzero(np.diff(fs)) + 1))
    total = fs.size
    pos_suffix = np.concatenate((np.cumsum(ys[::-1])[::-1], [0]))  # pos_suffix[i]=sum ys[i:]
    tp = pos_suffix[first]
    cnt = (total - first).astype(np.int64)

    scores = _score(tp, cnt, n_pos, n_neg, objective, beta)
    scores = np.where(cnt >= min_predicted, scores, -np.inf)
    k = int(np.argmax(scores))
    if not np.isfinite(scores[k]):
        return Cutpoint(float("nan"), 0.0, 0.0, 0.0, 0)
    thr = float(fs[first[k]])
    prec = float(tp[k] / cnt[k]) if cnt[k] else 0.0
    rec = float(tp[k] / n_pos) if n_pos else 0.0
    return Cutpoint(thr, float(scores[k]), prec, rec, int(cnt[k]))


def best_cutpoint(feature: np.ndarray, positives: np.ndarray, direction: Direction,
                  objective: Objective = "fbeta", beta: float = 0.5,
                  min_predicted: int = 1) -> Cutpoint:
    """Exact label-optimal single threshold for one vote.

    Args:
        feature: per-bar feature values (NaN allowed).
        positives: boolean mask, True at the side's structural-pivot bars.
        direction: ``"geq"`` if the vote fires when ``feature >= threshold``,
            ``"leq"`` if it fires when ``feature <= threshold``.
        objective: ``"fbeta"`` (precision-leaning by default, beta<1) or
            ``"youden"`` (TPR-FPR).
        beta: F-beta beta; <1 favours precision (apt for rare events).
        min_predicted: ignore thresholds firing on fewer than this many bars.

    Returns:
        Cutpoint with the threshold expressed in the ORIGINAL feature units.
    """
    if direction == "geq":
        return _scan_geq(feature, positives, objective, beta, min_predicted)
    # feature <= t  <=>  -feature >= -t
    cp = _scan_geq(-np.asarray(feature, dtype=np.float64), positives,
                   objective, beta, min_predicted)
    if not np.isfinite(cp.threshold):
        return cp
    return Cutpoint(-cp.threshold, cp.score, cp.precision, cp.recall, cp.n_predicted)


def purged_time_folds(n: int, k: int = 5, embargo: int = 0) -> list[tuple[np.ndarray, np.ndarray]]:
    """Contiguous time-ordered K-fold splits with a symmetric embargo gap.

    Returns list of (train_idx, test_idx). Embargo bars on each side of the test
    block are dropped from train to avoid leakage across the centered-window labels.
    """
    folds: list[tuple[np.ndarray, np.ndarray]] = []
    bounds = np.linspace(0, n, k + 1, dtype=int)
    for i in range(k):
        lo, hi = bounds[i], bounds[i + 1]
        test = np.arange(lo, hi)
        left = np.arange(0, max(0, lo - embargo))
        right = np.arange(min(n, hi + embargo), n)
        train = np.concatenate((left, right))
        if test.size and train.size:
            folds.append((train, test))
    return folds


def cv_select_cutpoint(feature: np.ndarray, positives: np.ndarray, direction: Direction,
                       objective: Objective = "fbeta", beta: float = 0.5,
                       n_grid: int = 256, k: int = 5, embargo: int = 0,
                       min_predicted: int = 1) -> Cutpoint:
    """Robust cutpoint: pick the threshold maximizing MEAN out-of-fold objective.

    Candidate thresholds are a quantile grid of the feature (bounded to ``n_grid``)
    so cost is O(n_grid * n). This is the overfit-guard for the ~100-event regime:
    the cut is chosen by held-out score, not in-sample.
    """
    feature = np.asarray(feature, dtype=np.float64)
    positives = np.asarray(positives, dtype=bool)
    n = feature.size
    if int(positives.sum()) == 0:
        return Cutpoint(float("nan"), 0.0, 0.0, 0.0, 0)

    finite = feature[~np.isnan(feature)]
    if finite.size == 0:
        return Cutpoint(float("nan"), 0.0, 0.0, 0.0, 0)
    qs = np.linspace(0.0, 1.0, min(n_grid, finite.size))
    cands = np.unique(np.quantile(finite, qs))

    folds = purged_time_folds(n, k=k, embargo=embargo)
    if not folds:
        return best_cutpoint(feature, positives, direction, objective, beta, min_predicted)

    fire = (feature[:, None] >= cands[None, :]) if direction == "geq" \
        else (feature[:, None] <= cands[None, :])
    fire &= ~np.isnan(feature)[:, None]

    mean_obj = np.zeros(cands.size, dtype=np.float64)
    used = 0
    for train, test in folds:
        yte = positives[test]
        n_pos = int(yte.sum()); n_neg = int((~yte).sum())
        if n_pos == 0:
            continue
        used += 1
        fte = fire[test]                       # (|test|, n_cands)
        cnt = fte.sum(axis=0)
        tp = (fte & yte[:, None]).sum(axis=0)
        mean_obj += _score(tp.astype(float), cnt.astype(float), n_pos, n_neg, objective, beta)
    if used == 0:
        return best_cutpoint(feature, positives, direction, objective, beta, min_predicted)
    mean_obj /= used

    # tie-break toward the in-sample-best feasible threshold
    full_fire = (feature[:, None] >= cands[None, :]) if direction == "geq" \
        else (feature[:, None] <= cands[None, :])
    full_fire &= ~np.isnan(feature)[:, None]
    full_cnt = full_fire.sum(axis=0)
    mean_obj = np.where(full_cnt >= min_predicted, mean_obj, -np.inf)
    if not np.isfinite(mean_obj.max()):
        return best_cutpoint(feature, positives, direction, objective, beta, min_predicted)

    k_best = int(np.argmax(mean_obj))
    thr = float(cands[k_best])
    pred = full_fire[:, k_best]
    tp = int((pred & positives).sum()); cnt = int(pred.sum())
    prec = tp / cnt if cnt else 0.0
    rec = tp / int(positives.sum()) if positives.sum() else 0.0
    return Cutpoint(thr, float(mean_obj[k_best]), prec, rec, cnt)


__all__ = [
    "Cutpoint", "Direction", "Objective",
    "best_cutpoint", "cv_select_cutpoint", "purged_time_folds",
]
