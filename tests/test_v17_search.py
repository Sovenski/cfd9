"""Spec §3 (PHASE 1) — Sobol-seeded separable CMA-ES batch optimizer.

Assertions (plan/gpu-refactor-build-spec.md §3):
- Sobol seeding stays within ``space_for(side).float_bounds``; shape params
  never change across the population.
- The reported winner re-scored by the REAL ``PooledScorer`` matches the
  search-reported LCB to ``< 1e-9``.
- Determinism: same seed -> same population.
- ``run_v17(..., search="ascent")`` output is byte-identical to today.
- ``firing_excess`` is folded into the population ranking via the
  ``v17_acceptance`` helper.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.indicators import Params
from src.search_space import space_for
from src.v17_acceptance import rank_by_penalized_lcb
from src.v17_optimize import PooledScorer, active_threshold_fields
from src.v17_search import BatchOptimizer, BatchSearchConfig

_CSV = Path("data/raw/DAX_1D_19700102_20260324.csv")


class _QuadScorer:
    """Cheap deterministic scorer (no market data): peak at mid-bounds."""

    def __init__(self, seed: Params, side: str) -> None:
        self.fields = active_threshold_fields(seed, side)
        fb = space_for(side).float_bounds
        self.bounds = {f: fb[f[: -(len(side) + 1)]] for f in self.fields}
        self.n_calls = 0

    def score(self, params: Params) -> float:
        self.n_calls += 1
        sc = 0.0
        for f, (lo, hi) in self.bounds.items():
            x = (float(getattr(params, f)) - lo) / (hi - lo)
            sc -= (x - 0.5) ** 2
        return sc


def _population_vectors(res, fields: list[str]) -> list[list[float]]:
    return [[float(getattr(c.params, f)) for f in fields] + [float(c.score), c.stage]
            for c in res.population]


# ---------------------------------------------------------------------------
# Sobol within bounds + shape params frozen
# ---------------------------------------------------------------------------
def test_sobol_within_bounds_and_shape_frozen():
    seed = Params()
    side = "low"
    scorer = _QuadScorer(seed, side)
    cfg = BatchSearchConfig(popsize=8, sobol_n=16, generations=2, top_k=5, rng_seed=7)
    res = BatchOptimizer(seed=seed, scorer=scorer, side=side, config=cfg).run()

    fields = active_threshold_fields(seed, side)
    fb = space_for(side).float_bounds
    threshold_fields = set(fields)
    for cand in res.population:
        # 1) every searched threshold stays inside its float bounds
        for f in fields:
            lo, hi = fb[f[: -(len(side) + 1)]]
            v = float(getattr(cand.params, f))
            assert lo <= v <= hi, f"{cand.stage} candidate {f}={v} outside [{lo},{hi}]"
        # 2) SHAPE / non-active params are FROZEN at the seed values
        for f in vars(seed):
            if f in threshold_fields:
                continue
            assert getattr(cand.params, f) == getattr(seed, f), \
                f"shape field {f!r} changed in {cand.stage} candidate"

    stages = [c.stage for c in res.population]
    assert stages.count("seed") == 1
    assert stages.count("sobol") == cfg.sobol_n
    assert stages.count("cma") == cfg.generations * cfg.popsize
    assert res.n_evals == len(res.population) == 1 + cfg.sobol_n + cfg.generations * cfg.popsize
    assert scorer.n_calls == res.n_evals
    assert res.coords == fields
    assert len(res.trace) == len(res.population)


# ---------------------------------------------------------------------------
# Determinism: same explicit seed -> identical population (values AND scores)
# ---------------------------------------------------------------------------
def test_determinism_same_seed_same_population():
    seed = Params()
    side = "low"
    fields = active_threshold_fields(seed, side)

    def _run(rng_seed: int):
        cfg = BatchSearchConfig(popsize=8, sobol_n=8, generations=2, top_k=4,
                                rng_seed=rng_seed)
        return BatchOptimizer(seed=seed, scorer=_QuadScorer(seed, side), side=side,
                              config=cfg).run()

    a, b = _run(42), _run(42)
    assert _population_vectors(a, fields) == _population_vectors(b, fields)
    assert a.score == b.score and a.seed_score == b.seed_score
    assert all(getattr(a.params, f) == getattr(b.params, f) for f in vars(a.params))
    # a different seed must actually change the sampled population
    c = _run(43)
    assert _population_vectors(a, fields) != _population_vectors(c, fields)


# ---------------------------------------------------------------------------
# firing_excess wiring: ranking goes through v17_acceptance.rank_by_penalized_lcb
# ---------------------------------------------------------------------------
def test_rank_by_penalized_lcb_helper():
    assert list(rank_by_penalized_lcb([1.0, 0.9, 0.95])) == [0, 2, 1]
    # firing excess demotes the high-firing candidate BEFORE argmax
    assert list(rank_by_penalized_lcb([1.0, 0.9], firing_excesses=[5.0, 0.0],
                                      penalty=0.1)) == [1, 0]
    # penalty=0 (or no excesses) -> pure LCB order, stable on ties
    assert list(rank_by_penalized_lcb([0.5, 0.5], firing_excesses=[1.0, 0.0],
                                      penalty=0.0)) == [0, 1]


def test_top_k_matches_penalized_ranking():
    seed = Params()
    side = "low"
    cfg = BatchSearchConfig(popsize=8, sobol_n=8, generations=1, top_k=5, rng_seed=3)
    res = BatchOptimizer(seed=seed, scorer=_QuadScorer(seed, side), side=side,
                         config=cfg).run()
    scores = [c.score for c in res.population]
    expected = rank_by_penalized_lcb(scores)[: cfg.top_k]
    assert [c.score for c in res.top_k] == [scores[i] for i in expected]
    # the winner IS top_k[0], and the seed is in the population so the
    # batch search can never regress below the seed score
    assert res.score == res.top_k[0].score
    assert res.params == res.top_k[0].params
    assert res.score >= res.seed_score


# ---------------------------------------------------------------------------
# Winner re-scored by the REAL PooledScorer matches the reported LCB < 1e-9
# ---------------------------------------------------------------------------
def _folds():
    if not _CSV.exists():
        pytest.skip(f"missing {_CSV}")
    from src.pooled_validation import StreamData, build_calendar_folds, load_stream_frame
    from src.scoring import add_pivot_labels
    from src.universe import Stream
    df = load_stream_frame(str(_CSV)).iloc[-6000:].copy()
    add_pivot_labels(df)
    s = Stream(ticker="DAX", timeframe="1D", path=str(_CSV), cluster_id="EU_EQ")
    sd = StreamData(stream=s, df=df, bar_seconds=86400.0)
    return build_calendar_folds([sd])[:3], [s]


def test_winner_rescored_by_real_pooled_scorer():
    from src.v17_fastdetector import FastPooledScorer
    folds, streams = _folds()
    base = Params()
    fast = FastPooledScorer(folds=folds, streams=streams, side="low", base_params=base)
    cfg = BatchSearchConfig(popsize=4, sobol_n=4, generations=1, top_k=3, rng_seed=11)
    res = BatchOptimizer(seed=base, scorer=fast, side="low", config=cfg).run()
    real = PooledScorer(folds=folds, streams=streams, side="low")
    assert abs(real.score(res.params) - res.score) < 1e-9


# ---------------------------------------------------------------------------
# run_v17(search="ascent") byte-identical to today's default path
# ---------------------------------------------------------------------------
def _mini_data_dir(tmp_path: Path) -> str:
    lines = _CSV.read_text().splitlines()
    keep = [lines[0]] + lines[1:][-6000:]
    (tmp_path / "DAX_1D_00000000_00000000.csv").write_text("\n".join(keep) + "\n")
    return str(tmp_path)


def test_run_v17_ascent_route_byte_identical(tmp_path):
    if not _CSV.exists():
        pytest.skip(f"missing {_CSV}")
    from src.pooled_validation import (
        StreamData, apply_volume_policy, build_calendar_folds, load_stream_frame,
    )
    from src.scoring import add_pivot_labels
    from src.universe import resolve_streams
    from src.v17_fastdetector import FastPooledScorer
    from src.v17_optimize import coordinate_ascent
    from src.v17_runner import run_v17

    data_dir = _mini_data_dir(tmp_path)
    kw = dict(groups=["INDICES"], timeframes=["1D"], data_dir=data_dir,
              sides=("low",), era_kw={"step_fraction": 0.25}, grid_n=2,
              max_sweeps=1, run_slug="t_reg", progress=False)
    out_default = run_v17(**kw)
    out_ascent = run_v17(**kw, search="ascent")

    # the explicit "ascent" route is byte-identical to the default route
    assert (json.dumps(out_default, sort_keys=True, default=str)
            == json.dumps(out_ascent, sort_keys=True, default=str))
    # no new keys leak into the default output (regression vs today)
    assert set(out_default.keys()) == {
        "run_slug", "groups", "timeframes", "volume_policy", "n_folds",
        "streams", "grid_n", "max_sweeps", "sides",
    }
    assert set(out_default["sides"]["low"].keys()) == {
        "seed_lcb", "final_lcb", "n_evals", "changed", "best_params",
        "coords", "trace",
    }

    # ...and equal to the unrouted pipeline (what run_v17 computes today)
    streams = resolve_streams(["INDICES"], ["1D"], data_dir=data_dir)
    sds = []
    for s in streams:
        df = load_stream_frame(s.path)
        df, keepit = apply_volume_policy(df, policy="price_only")
        assert keepit
        add_pivot_labels(df)
        sds.append(StreamData(stream=s, df=df, bar_seconds=86400.0))
    folds = build_calendar_folds(sds, step_fraction=0.25)
    kept = [sd.stream for sd in sds]
    seed = Params()
    scorer = FastPooledScorer(folds=folds, streams=kept, side="low", base_params=seed)
    res = coordinate_ascent(seed, scorer, side="low", grid_n=2, max_sweeps=1,
                            seed_score=scorer.score(seed))
    real_final = PooledScorer(folds=folds, streams=kept, side="low").score(res.params)

    side = out_default["sides"]["low"]
    assert side["seed_lcb"] == res.seed_score
    assert side["final_lcb"] == real_final
    assert side["n_evals"] == res.n_evals
    assert side["best_params"] == {k: getattr(res.params, k) for k in vars(res.params)}
    assert side["coords"] == res.coords
    assert side["trace"] == res.trace


# ---------------------------------------------------------------------------
# run_v17(search="cma") routes through BatchOptimizer and reports the real LCB
# ---------------------------------------------------------------------------
def test_run_v17_cma_route(tmp_path):
    if not _CSV.exists():
        pytest.skip(f"missing {_CSV}")
    from src.v17_runner import run_v17
    data_dir = _mini_data_dir(tmp_path)
    out = run_v17(groups=["INDICES"], timeframes=["1D"], data_dir=data_dir,
                  sides=("low",), era_kw={"step_fraction": 0.25},
                  run_slug="t_cma", progress=False, search="cma",
                  search_kw={"popsize": 4, "sobol_n": 4, "generations": 1,
                             "top_k": 3, "rng_seed": 11})
    side = out["sides"]["low"]
    assert side["search"] == "cma"
    assert len(side["top_k"]) == 3
    # reported number is the REAL detector LCB and matches the search LCB <1e-9
    assert abs(side["final_lcb"] - side["top_k"][0]["score"]) < 1e-9
