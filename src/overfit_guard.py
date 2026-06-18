"""Selection-bias guards for the batched search — spec §4 (PHASE 2).

A batch-hungry optimizer (Sobol + separable CMA-ES) evaluates
``lambda x generations + sobol_n + 1`` configurations per run — orders of
magnitude more than coordinate ascent's ~200 — inflating the
multiple-comparison risk, worst on the thin (~27-event) HIGH side. This module
supplies the guards the plan (plan/gpu-refactor-plan.md §2) requires:

- ``deflated_best`` — a deflation haircut on the best reported LCB scaled to
  the trial count via the expected maximum of n iid standard normals
  (Bailey & Lopez de Prado's deflated-best construction). Wired into the
  REPORTED number on the batched-search route in ``v17_runner``.
- ``pbo_cscv`` — Probability of Backtest Overfitting via Combinatorially
  Symmetric Cross-Validation (CSCV). ADVISORY-only: a go/no-go hint for LOW,
  advisory-with-caveats for HIGH (too few events to gate on).
- ``wilson_interval`` — the wide interval used as the PRIMARY HIGH summary
  (event count + interval), per the "honest ceiling": no CV design
  manufactures power the data lacks.

Default OFF: nothing in the default (ascent) pipeline calls this module.
"""
from __future__ import annotations

import itertools
import logging
import math
from typing import Optional, Sequence

import numpy as np
from scipy.stats import norm

logger = logging.getLogger(__name__)

_EULER_GAMMA: float = 0.5772156649015329


def expected_max_gaussian(n_trials: int) -> float:
    """E[max of n iid N(0,1)] — the deflation null for n independent trials.

    Uses the standard asymptotic approximation
    ``(1-gamma) * z(1 - 1/n) + gamma * z(1 - 1/(n*e))`` with the
    Euler-Mascheroni constant; 0.0 for ``n <= 1`` (a single trial has no
    selection effect).
    """
    n = int(n_trials)
    if n <= 1:
        return 0.0
    return float((1.0 - _EULER_GAMMA) * norm.ppf(1.0 - 1.0 / n)
                 + _EULER_GAMMA * norm.ppf(1.0 - 1.0 / (n * math.e)))


def deflated_best(
    best: float,
    population_scores: Sequence[float],
    n_trials: Optional[int] = None,
) -> dict:
    """Deflate the best score for having been selected from many trials.

    Haircut = (cross-trial score dispersion) x E[max of n_trials N(0,1)] —
    the expected spurious advantage of the argmax under the null that all
    candidates share one true mean. ``n_trials`` should be the FULL evaluated
    count (``lambda x generations + sobol_n + 1``), not the finalist count.

    Returns a dict with ``raw``, ``deflated`` (clamped at 0), ``haircut``,
    ``sigma_trials``, ``e_max_z`` and ``n_trials``.
    """
    scores = np.asarray(list(population_scores), dtype=float)
    if scores.size == 0:
        raise ValueError("population_scores must be non-empty")
    n = int(n_trials) if n_trials is not None else int(scores.size)
    if n < 1:
        raise ValueError(f"n_trials must be >= 1, got {n}")
    sigma = float(np.std(scores))
    e_max = expected_max_gaussian(n)
    haircut = sigma * e_max
    deflated = float(max(0.0, float(best) - haircut))
    logger.info("deflated_best: raw=%.6f haircut=%.6f (sigma=%.6f, "
                "E[maxZ_%d]=%.3f) -> deflated=%.6f",
                float(best), haircut, sigma, n, e_max, deflated)
    return {"raw": float(best), "deflated": deflated, "haircut": float(haircut),
            "sigma_trials": sigma, "e_max_z": float(e_max), "n_trials": n}


def pbo_cscv(score_matrix: np.ndarray, n_partitions: int = 8) -> dict:
    """Probability of Backtest Overfitting via CSCV (Bailey et al. 2017).

    ``score_matrix`` is ``[n_observations, n_candidates]`` (rows = folds or
    time blocks, columns = configurations). Rows are split into
    ``n_partitions`` contiguous blocks; for every half/half combination the
    in-sample argmax candidate is ranked on the held-out half. PBO is the
    fraction of combinations where that candidate falls below the median
    out-of-sample (relative-rank logit <= 0).

    ADVISORY ONLY (never an export gate): on the HIGH side the event count is
    far too small for PBO to be decisive — the primary HIGH summary stays
    event count + Wilson interval.
    """
    m = np.asarray(score_matrix, dtype=float)
    if m.ndim != 2:
        raise ValueError(f"score_matrix must be 2-D, got shape {m.shape}")
    t, k = m.shape
    if k < 2:
        raise ValueError("need >= 2 candidates to measure selection bias")
    s = min(int(n_partitions), t)
    s -= s % 2  # CSCV needs an even number of row blocks
    if s < 2:
        raise ValueError(f"need >= 2 row blocks, got {s} (t={t})")
    blocks = np.array_split(np.arange(t), s)
    logits: list[float] = []
    for combo in itertools.combinations(range(s), s // 2):
        train_rows = np.concatenate([blocks[i] for i in combo])
        test_rows = np.concatenate(
            [blocks[i] for i in range(s) if i not in combo])
        mu_train = m[train_rows].mean(axis=0)
        mu_test = m[test_rows].mean(axis=0)
        star = int(np.argmax(mu_train))
        rank = 1.0 + float(np.sum(mu_test < mu_test[star]))
        omega = rank / (k + 1.0)
        logits.append(math.log(omega / (1.0 - omega)))
    pbo = float(np.mean([lam <= 0.0 for lam in logits]))
    logger.info("pbo_cscv: PBO=%.3f over %d combinations (S=%d, T=%d, K=%d)",
                pbo, len(logits), s, t, k)
    return {"pbo": pbo, "n_combinations": len(logits), "n_partitions": s,
            "logits": logits, "advisory": True}


def wilson_interval(
    successes: int, n: int, alpha: float = 0.05,
) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion.

    The PRIMARY summary for the thin HIGH side is the raw event count plus
    this (wide) interval. ``n == 0`` returns the maximally wide ``(0, 1)`` —
    zero evidence constrains nothing.
    """
    if n < 0 or successes < 0 or successes > max(n, 0):
        raise ValueError(f"invalid counts: successes={successes}, n={n}")
    if n == 0:
        return (0.0, 1.0)
    z = float(norm.ppf(1.0 - alpha / 2.0))
    p = successes / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denom
    half = z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


__all__ = ["expected_max_gaussian", "deflated_best", "pbo_cscv",
           "wilson_interval"]
