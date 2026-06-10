"""v17 batch search — Sobol-seeded separable CMA-ES over detector thresholds.

PHASE 1 of the GPU refactor (plan/gpu-refactor-build-spec.md §3): replace the
one-candidate-at-a-time coordinate ascent with a batch-hungry optimizer that
emits hundreds of candidates per generation, so a wide evaluator (the CPU
``FastPooledScorer`` today, the GPU ``score_pop`` later) can score them in
parallel.

Parity surface is ZERO: every candidate is a plain ``Params`` scored by the
EXISTING scorer (``FastPooledScorer`` / ``PooledScorer``). Only the continuous
threshold fields from ``active_threshold_fields(seed, side)`` are varied,
inside ``search_space.space_for(side).float_bounds``; every discrete SHAPE /
architecture param stays FROZEN at the seed value (``FastDetector.signals``
re-asserts this at eval time). Ranking goes through
``v17_acceptance.rank_by_penalized_lcb`` so ``firing_excess`` is folded in
BEFORE the argmax — wrap the scorer in ``v17_acceptance.PenalizedScorer``
(``firing_penalty > 0``) to make the search objective itself penalized.
"""

from __future__ import annotations

import dataclasses
import logging
from dataclasses import dataclass, field
from typing import Optional, Protocol

import numpy as np
from scipy.stats import qmc

from .indicators import Params
from .search_space import space_for
from .v17_acceptance import rank_by_penalized_lcb
from .v17_optimize import active_threshold_fields

logger = logging.getLogger(__name__)


class Scorer(Protocol):
    """Anything with the ``PooledScorer`` scoring contract."""

    def score(self, params: Params) -> float: ...


@dataclass(frozen=True)
class BatchSearchConfig:
    """Knobs for one ``BatchOptimizer.run()``.

    ``popsize`` (CMA lambda) is intended to be 256–1024 in production (spec §3)
    so the batched evaluator stays full; smaller values are allowed for tests
    and smoke runs. ``sobol_n`` should be a power of two (Sobol balance).
    """

    popsize: int = 256          # CMA-ES lambda (production range [256, 1024])
    sobol_n: int = 256          # Sobol seeding draws (0 disables seeding)
    generations: int = 4        # CMA ask/tell rounds
    top_k: int = 16             # finalists returned by penalized-LCB rank
    sigma0: float = 0.25        # initial step size in normalized [0,1] coords
    rng_seed: int = 42          # drives BOTH the Sobol scramble and CMA-ES
    n_jobs: int = 1             # >1 => thread-pool candidate scoring

    def __post_init__(self) -> None:
        if self.popsize < 2:
            raise ValueError(f"popsize must be >= 2, got {self.popsize}")
        if self.sobol_n < 0 or self.generations < 0 or self.top_k < 1:
            raise ValueError("sobol_n/generations must be >= 0, top_k >= 1")
        if not 256 <= self.popsize <= 1024:
            logger.debug("popsize %d outside the production range [256, 1024]",
                         self.popsize)


@dataclass(frozen=True)
class Candidate:
    """One evaluated configuration (full Params; shape == seed shape)."""

    params: Params
    score: float
    stage: str  # "seed" | "sobol" | "cma"


@dataclass
class BatchSearchResult:
    """Mirror of ``AscentResult`` plus the full population and the finalists."""

    params: Params                                  # winner (penalized-LCB argmax)
    score: float                                    # winner's search LCB
    seed_score: float
    n_evals: int
    history: list[tuple[str, float, float]]         # (field, winner_value, score)
    coords: list[str] = field(default_factory=list)  # searched threshold fields
    trace: list = field(default_factory=list)        # every evaluated config + score
    population: list[Candidate] = field(default_factory=list)
    top_k: list[Candidate] = field(default_factory=list)


class BatchOptimizer:
    """Sobol seeding + separable CMA-ES over the active threshold box.

    Seeding: ``scipy.stats.qmc.Sobol`` (scrambled, seeded) over the active
    continuous threshold bounds. Refinement: ``cma`` with diagonal covariance
    (separable CMA-ES), warm-started at the v16/gold ``seed`` Params, run in
    normalized [0,1] coordinates with box bounds. Returns the full evaluated
    population plus the top-K finalists by penalized LCB.
    """

    def __init__(self, seed: Params, scorer: Scorer, side: str,
                 config: Optional[BatchSearchConfig] = None) -> None:
        if side not in ("high", "low"):
            raise ValueError(f"side must be 'high'|'low', got {side!r}")
        self.seed = seed
        self.scorer = scorer
        self.side = side
        self.config = config or BatchSearchConfig()
        # The drift GATE exists only on the HIGH side (detector.py L509/L564
        # — gate_l never reads it), so pivot_drift_gate_mult_low is a dead
        # knob: searching it wastes a dimension and its boundary "pins" are
        # spurious reject triggers.
        self.fields: list[str] = [
            f for f in active_threshold_fields(seed, side)
            if f != "pivot_drift_gate_mult_low"
        ]
        fb = space_for(side).float_bounds
        suffix_len = len(side) + 1
        self._lo = np.array([fb[f[:-suffix_len]][0] for f in self.fields], dtype=float)
        self._hi = np.array([fb[f[:-suffix_len]][1] for f in self.fields], dtype=float)

    # ------------------------------------------------------------------
    def _to_params(self, x: np.ndarray) -> Params:
        """Clip a real-coordinate vector into bounds and bind it to Params."""
        x = np.clip(np.asarray(x, dtype=float), self._lo, self._hi)
        return dataclasses.replace(
            self.seed, **{f: float(v) for f, v in zip(self.fields, x)})

    def _denorm(self, u: np.ndarray) -> np.ndarray:
        return self._lo + np.asarray(u, dtype=float) * (self._hi - self._lo)

    def _norm_seed(self) -> np.ndarray:
        x = np.array([float(getattr(self.seed, f)) for f in self.fields])
        return np.clip((x - self._lo) / (self._hi - self._lo), 0.0, 1.0)

    def _score_batch(self, candidates: list[Params]) -> list[float]:
        """Score a batch (the GPU ``score_pop`` slots in here in Phase 3)."""
        if self.config.n_jobs > 1:
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=self.config.n_jobs) as ex:
                return [float(s) for s in ex.map(self.scorer.score, candidates)]
        return [float(self.scorer.score(p)) for p in candidates]

    # ------------------------------------------------------------------
    def run(self) -> BatchSearchResult:
        """Seed eval -> Sobol batch -> CMA generations -> penalized-LCB top-K."""
        cfg = self.config
        population: list[Candidate] = []

        def _record(cands: list[Params], scores: list[float], stage: str) -> None:
            population.extend(Candidate(params=p, score=s, stage=stage)
                              for p, s in zip(cands, scores))

        seed_score = self._score_batch([self.seed])[0]
        _record([self.seed], [seed_score], "seed")
        logger.info("[v17_search:%s] seed LCB=%.5f; %d threshold dims, "
                    "sobol_n=%d popsize=%d generations=%d",
                    self.side, seed_score, len(self.fields),
                    cfg.sobol_n, cfg.popsize, cfg.generations)

        # --- Sobol seeding over the active threshold box ---
        if cfg.sobol_n > 0:
            sampler = qmc.Sobol(d=len(self.fields), scramble=True, seed=cfg.rng_seed)
            pts = qmc.scale(sampler.random(cfg.sobol_n), self._lo, self._hi)
            cands = [self._to_params(x) for x in pts]
            _record(cands, self._score_batch(cands), "sobol")

        # --- separable CMA-ES, warm-started at the seed Params ---
        if cfg.generations > 0:
            import cma
            opts = {
                "popsize": cfg.popsize,
                "bounds": [0.0, 1.0],
                "seed": max(1, int(cfg.rng_seed)),  # cma treats 0 as time-based
                "verbose": -9,
                "verb_log": 0,
                "CMA_diagonal": True,               # separable / diagonal covariance
            }
            es = cma.CMAEvolutionStrategy(list(self._norm_seed()), cfg.sigma0, opts)
            for gen in range(cfg.generations):
                xs = es.ask()
                cands = [self._to_params(self._denorm(np.asarray(x))) for x in xs]
                scores = self._score_batch(cands)
                es.tell(xs, [-s for s in scores])   # cma minimizes
                _record(cands, scores, "cma")
                logger.info("[v17_search:%s] gen %d/%d best=%.5f",
                            self.side, gen + 1, cfg.generations, max(scores))

        # --- penalized-LCB ranking (firing_excess folded in BEFORE argmax; a
        # PenalizedScorer wrapper already penalizes each score per fold) ---
        order = rank_by_penalized_lcb([c.score for c in population])
        finalists = [population[i] for i in order[: cfg.top_k]]
        winner = finalists[0]
        history = [(f, float(getattr(winner.params, f)), winner.score)
                   for f in self.fields
                   if float(getattr(winner.params, f)) != float(getattr(self.seed, f))]
        trace = [{**{f: float(getattr(c.params, f)) for f in self.fields},
                  "score": float(c.score), "varied": c.stage}
                 for c in population]
        logger.info("[v17_search:%s] %d evals; winner LCB=%.5f (seed %.5f)",
                    self.side, len(population), winner.score, seed_score)
        return BatchSearchResult(
            params=winner.params, score=winner.score, seed_score=seed_score,
            n_evals=len(population), history=history, coords=list(self.fields),
            trace=trace, population=population, top_k=finalists)


__all__ = ["BatchOptimizer", "BatchSearchConfig", "BatchSearchResult", "Candidate"]
