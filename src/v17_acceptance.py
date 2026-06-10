"""v17 anti-gaming: firing-rate penalty (#2) + acceptance gates (#3).

The deep analysis showed v17 could raise the bootstrap LCB by *loosening gates to
fire more signals* on ~100 noisy events (precision fell while the composite rose),
with thresholds pinning to box bounds and gains failing seed/era robustness.

This module supplies:
- ``firing_excess`` — a per-fold penalty on firing >cap x the pivot count
  (recall/precision = n_signals/total_pivots). Used by the search scorer so
  "fire more" cannot raise the *search* objective.
- ``boundary_pinned`` — flags changed thresholds sitting on a search-space bound
  (auto-reject: a boundary optimum means the metric is monotone in looseness).
- ``bootstrap_stability`` — re-evaluates the LCB across bootstrap seeds (stop
  trusting a single seed=42 draw).
- ``summarize_acceptance`` — combine gates into PASS / FRAGILE / REJECT.

The FINAL reported LCB and these gates always use the RAW objective (no penalty)
scored by the real detector — the penalty only steers the search.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterable, Iterator, Optional, Sequence

import numpy as np

from .search_space import space_for
from .validation import fold_scores_bootstrap_ci

if TYPE_CHECKING:  # pragma: no cover - typing only, avoids import cycles
    from .indicators import Params


def firing_excess(precision: float, recall: float, cap: float = 2.0) -> float:
    """Per-fold firing penalty: max(0, (recall/precision) - cap).

    recall/precision == n_signals/total_pivots, so this penalizes folds that fire
    more than ``cap`` signals per structural pivot (the gate-loosening mechanism).
    precision==0 with any recall => pure false-positive spray => large penalty.
    """
    ratio = float(recall) / max(float(precision), 1e-9)
    return max(0.0, ratio - float(cap))


def rank_by_penalized_lcb(
    lcbs: Sequence[float],
    firing_excesses: Optional[Sequence[float]] = None,
    penalty: float = 0.0,
) -> np.ndarray:
    """Population ranking for the batched search (spec §3): penalized-LCB argmax.

    Returns the candidate indices sorted DESCENDING by ``lcb - penalty * excess``
    (stable, so ties keep submission order). This folds ``firing_excess`` into
    the population ranking BEFORE the argmax: a candidate that buys its LCB by
    over-firing is demoted, exactly like the per-fold ``firing_penalty`` the
    scorers already apply inside ``score`` — pass ``firing_excesses`` only when
    the supplied ``lcbs`` are RAW (un-penalized).

    Args:
        lcbs: Per-candidate pooled LCBs (already penalized, or raw).
        firing_excesses: Optional per-candidate aggregate ``firing_excess``.
        penalty: Penalty weight lambda; ignored unless ``> 0`` and excesses given.

    Returns:
        ``np.ndarray`` of indices into ``lcbs``, best candidate first.
    """
    arr = np.asarray(list(lcbs), dtype=float)
    if firing_excesses is not None and float(penalty) > 0.0:
        exc = np.asarray(list(firing_excesses), dtype=float)
        if exc.shape != arr.shape:
            raise ValueError(f"firing_excesses shape {exc.shape} != lcbs shape {arr.shape}")
        arr = arr - float(penalty) * exc
    return np.argsort(-arr, kind="stable")


def _iter_fold_scores(scorer, params: "Params") -> Iterator[tuple[float, dict]]:
    """Yield ``(raw_fold_score, components)`` per informative fold of a scorer.

    Reproduces the per-fold loop of ``v17_optimize.PooledScorer.score`` /
    ``v17_fastdetector.FastPooledScorer.score`` WITHOUT modifying those
    trust-root classes (spec §0.1): the loop bodies are duplicated here, off
    the oracle files, and pinned to them by tests
    (``tests/test_v17_acceptance_helpers.py``, golden regressions in
    ``tests/test_cpcv_purge.py`` / ``tests/test_v17_gpu_parity.py``).
    """
    from .pooled_validation import _fold_is_informative

    if hasattr(scorer, "_fast"):  # FastPooledScorer
        import pandas as pd

        from .pooled_scoring import pooled_fold_score
        from .pooled_validation import _stream_stat
        key = scorer._sig_key
        for entries in scorer._fast:
            is_stats, oos_stats = [], []
            for fd_is, fd_oos, sl in entries:
                w = scorer._weights.get(sl.stream.stream_id, 1.0)
                sig_is = pd.Series(fd_is.signals(params)[key], index=sl.df_is.index)
                sig_oos = pd.Series(fd_oos.signals(params)[key], index=sl.df_oos.index)
                is_stats.append(_stream_stat(sl.df_is, sig_is, scorer.side, w))
                oos_stats.append(_stream_stat(sl.df_oos, sig_oos, scorer.side, w))
            s, comp = pooled_fold_score(is_stats, oos_stats, scorer.side)
            if _fold_is_informative(comp):
                yield float(s), comp
    elif hasattr(scorer, "_eval_folds"):  # PooledScorer
        from .pooled_validation import evaluate_pooled_fold
        for fold in scorer._eval_folds:
            s, comp = evaluate_pooled_fold(params, scorer.side, fold, scorer._weights)
            if _fold_is_informative(comp):
                yield float(s), comp
    else:
        raise TypeError(
            f"unsupported scorer type {type(scorer).__name__!r}: expected "
            "PooledScorer (._eval_folds) or FastPooledScorer (._fast)")


def raw_fold_scores(scorer, params: "Params") -> list[float]:
    """RAW (unpenalized) per-fold scores of a pooled scorer — for the gates.

    ``fold_scores_bootstrap_ci(raw_fold_scores(scorer, p))[0]`` reproduces
    ``scorer.score(p)`` bit-for-bit (asserted in the helper tests).
    """
    return [s for s, _ in _iter_fold_scores(scorer, params)]


@dataclass(frozen=True)
class PenalizedScorer:
    """``score()`` wrapper applying the per-fold firing penalty (search only).

    Drop-in for the batched search: ``score(params)`` equals the wrapped
    scorer's score when ``firing_penalty == 0``; with ``firing_penalty > 0``
    each informative fold score is reduced by
    ``firing_penalty * firing_excess(precision_oos, recall_oos, firing_cap)``
    BEFORE the pooled block bootstrap, so "fire more" cannot raise the search
    objective. The FINAL reported LCB must always come from the RAW objective.
    """

    scorer: object
    firing_penalty: float = 0.0
    firing_cap: float = 2.0

    def score(self, params: "Params") -> float:
        fs: list[float] = []
        for s, comp in _iter_fold_scores(self.scorer, params):
            if self.firing_penalty > 0:
                pen = firing_excess(comp.get("precision_oos", 0.0),
                                    comp.get("recall_oos", 0.0), self.firing_cap)
                s = s - self.firing_penalty * pen
            fs.append(s)
        if not fs:
            return 0.0
        return float(fold_scores_bootstrap_ci(
            fs, n_boot=self.scorer.n_boot, alpha=self.scorer.alpha,
            block_len=self.scorer.block_len)[0])


def _bootstrap_ci_seeded(
    scores: list[float],
    seed: int,
    n_boot: int = 1000,
    alpha: float = 0.10,
    block_len: int = 2,
) -> tuple[float, float]:
    """``validation.fold_scores_bootstrap_ci`` with a configurable RNG seed.

    Local copy (spec §0.1: ``src/validation.py`` is trust-root and stays
    byte-identical) — the ONLY difference is ``np.random.default_rng(seed)``
    instead of the hardcoded 42. ``seed=42`` is asserted equal to the oracle
    in ``tests/test_v17_acceptance_helpers.py``.
    """
    if not scores:
        return (0.0, 0.0)
    arr = np.asarray(list(scores), dtype=float)
    n = len(arr)
    if n == 1:
        return (float(arr[0]), float(arr[0]))
    block_len = max(1, min(int(block_len), n))
    n_blocks = int(np.ceil(n / block_len))
    rng = np.random.default_rng(seed)
    boot_means = np.empty(n_boot, dtype=float)
    block_offsets = np.arange(block_len)
    for b in range(n_boot):
        starts = rng.integers(0, n, size=n_blocks)
        # Stationary block bootstrap with wrap-around.
        idx = (starts[:, None] + block_offsets[None, :]) % n  # (n_blocks, block_len)
        sample = arr[idx.reshape(-1)][:n]
        boot_means[b] = sample.mean()
    lo = float(np.percentile(boot_means, 100.0 * alpha / 2.0))
    hi = float(np.percentile(boot_means, 100.0 * (1.0 - alpha / 2.0)))
    return (lo, hi)


def boundary_pinned(changed: list[tuple[str, float]], side: str,
                    tol: float = 0.02) -> list[tuple[str, str]]:
    """Which changed threshold fields sit on a search-space bound.

    Returns ``[(field, 'lo'|'hi'), ...]``. A boundary optimum is an automatic
    reject signal: the objective is monotone in that knob's looseness.
    """
    fb = space_for(side).float_bounds
    suffix = f"_{side}"
    pinned: list[tuple[str, str]] = []
    for field, value in changed:
        # Dead knob: the drift GATE exists only on the HIGH side (detector.py
        # L509/L564 — gate_l never reads it), so a "pin" on the LOW gate-mult
        # is random drift of a no-op parameter, not a looseness signal.
        if field == "pivot_drift_gate_mult_low":
            continue
        base = field[: -len(suffix)] if field.endswith(suffix) else field
        if base not in fb:
            continue
        lo, hi = fb[base]
        rng = max(hi - lo, 1e-12)
        if float(value) <= lo + tol * rng:
            pinned.append((field, "lo"))
        elif float(value) >= hi - tol * rng:
            pinned.append((field, "hi"))
    return pinned


def bootstrap_stability(fold_scores: list[float], seeds: Iterable[int] = range(10),
                        n_boot: int = 1000, alpha: float = 0.10, block_len: int = 2,
                        rel_std_max: float = 0.5) -> dict:
    """Re-bootstrap the pooled LCB across seeds; report mean/std/min + a pass flag.

    Pass = min LCB across seeds > 0 AND std <= rel_std_max * mean (i.e. the result
    is not a lucky single-seed draw).
    """
    seeds = list(seeds)
    lcbs = [float(_bootstrap_ci_seeded(fold_scores, seed=s, n_boot=n_boot, alpha=alpha,
                                       block_len=block_len)[0]) for s in seeds]
    mean = float(np.mean(lcbs)) if lcbs else 0.0
    std = float(np.std(lcbs)) if lcbs else 0.0
    mn = float(np.min(lcbs)) if lcbs else 0.0
    passed = bool(mn > 0.0 and std <= rel_std_max * max(mean, 1e-9))
    return {"mean": mean, "std": std, "min": mn,
            "seeds": [int(s) for s in seeds], "lcbs": lcbs, "pass": passed}


def summarize_acceptance(bootstrap_pass: bool, era_pass: bool,
                         pinned: list[tuple[str, str]]) -> str:
    """Combine the gates: REJECT if pinned or unstable; PASS if all clear; else FRAGILE."""
    if pinned or not bootstrap_pass:
        return "REJECT"
    if not era_pass:
        return "FRAGILE"
    return "PASS"


__all__ = ["firing_excess", "rank_by_penalized_lcb", "boundary_pinned",
           "bootstrap_stability", "summarize_acceptance",
           "raw_fold_scores", "PenalizedScorer"]
