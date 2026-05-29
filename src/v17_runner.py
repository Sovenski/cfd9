"""v17 runner — orchestrate Calibrated Coordinate-Ascent on a stream pool.

Reuses the v16 ingest/fold machinery (resolve_streams, common-era calendar
folds) and the existing pooled objective, then refines per-side thresholds with
``coordinate_ascent``. Emits the chosen Params + seed/final LCB + provenance.

Seeding:
- default: the gold ``Params()`` preset, OR
- WARM-START from a v16 best-params trial dict (recommended), mapped through the
  exact ``params_from_trial`` so the seed is a real v16 configuration.

The label-calibrated seed (v17_calibrate via v17_features) is the next increment;
this runner already accepts any seed Params, so wiring it is a drop-in.
"""

from __future__ import annotations

import json
import logging
from dataclasses import replace
from pathlib import Path
from typing import Optional

import optuna

from .indicators import Params
from .pooled_validation import (
    StreamData,
    apply_volume_policy,
    build_calendar_folds,
    load_stream_frame,
)
from .scoring import add_pivot_labels
from .speculatores145 import params_from_trial
from .universe import resolve_streams
from .v17_optimize import PooledScorer, coordinate_ascent

logger = logging.getLogger(__name__)

_TF_SECONDS = {"1D": 86400.0, "1W": 604800.0, "60": 3600.0, "240": 14400.0, "1m": 60.0}


def seed_from_trial_dict(trial_params: dict, side: str) -> Params:
    """Build a Params seed from a v16 best-trial dict via the exact mapping."""
    return params_from_trial(optuna.trial.FixedTrial(trial_params), side)


def run_v17(
    groups: list[str],
    timeframes: list[str],
    data_dir: str,
    sides: tuple[str, ...] = ("high", "low"),
    seed_params: Optional[dict] = None,        # {"high": {...}, "low": {...}} v16 best
    volume_policy: str = "price_only",
    era_kw: Optional[dict] = None,
    grid_n: int = 7,
    max_sweeps: int = 3,
    results_dir: Optional[str] = None,
    run_slug: str = "v17_local",
) -> dict:
    """Resolve a pool, build folds, and run coordinate-ascent per side."""
    era_kw = era_kw or {}
    streams = resolve_streams(groups, timeframes, data_dir=data_dir)
    if not streams:
        raise RuntimeError(f"0 streams resolved from {data_dir} for {groups} {timeframes}")

    stream_datas: list[StreamData] = []
    for s in streams:
        df = load_stream_frame(s.path)
        df, keep = apply_volume_policy(df, policy=volume_policy)
        if not keep:
            logger.info("drop %s: volume policy", s.stream_id)
            continue
        add_pivot_labels(df)
        stream_datas.append(StreamData(stream=s, df=df, bar_seconds=_TF_SECONDS[s.timeframe]))

    folds = build_calendar_folds(stream_datas, **era_kw)
    kept = [sd.stream for sd in stream_datas]
    logger.info("%d folds; streams/fold=%s", len(folds), [len(f) for f in folds])
    if not folds:
        raise RuntimeError("No folds — pool too small/short.")

    out: dict = {
        "run_slug": run_slug,
        "groups": groups, "timeframes": timeframes,
        "volume_policy": volume_policy, "n_folds": len(folds),
        "streams": [s.stream_id for s in kept],
        "grid_n": grid_n, "max_sweeps": max_sweeps,
        "sides": {},
    }
    for side in sides:
        seed = (seed_from_trial_dict(seed_params[side], side)
                if seed_params and side in seed_params else Params())
        scorer = PooledScorer(folds=folds, streams=kept, side=side)
        res = coordinate_ascent(seed, scorer, side=side, grid_n=grid_n,
                                max_sweeps=max_sweeps, progress=logger.info)
        out["sides"][side] = {
            "seed_lcb": res.seed_score,
            "final_lcb": res.score,
            "n_evals": res.n_evals,
            "changed": [(f, v) for f, v, _ in res.history],
            "best_params": {k: getattr(res.params, k) for k in vars(res.params)},
        }
        logger.info("[v17:%s] seed=%.5f -> final=%.5f (%d evals)",
                    side, res.seed_score, res.score, res.n_evals)

    if results_dir:
        Path(results_dir).mkdir(parents=True, exist_ok=True)
        p = Path(results_dir) / f"{run_slug}_v17.json"
        p.write_text(json.dumps(out, indent=2, default=str))
        out["_written"] = str(p)
        logger.info("wrote %s", p)
    return out


__all__ = ["run_v17", "seed_from_trial_dict"]
