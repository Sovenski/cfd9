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
from .v17_fastdetector import FastPooledScorer
from .v17_optimize import PooledScorer, coordinate_ascent, planned_evals

logger = logging.getLogger(__name__)


def _ascend_with_progress(seed, scorer, side, grid_n, max_sweeps, show):
    """Run coordinate_ascent with an up-front ETA + live progress bar.

    Times one seed eval, estimates total evals, prints an ETA, then shows a
    tqdm bar (falls back to plain prints if tqdm is unavailable).
    """
    import time

    planned = planned_evals(seed, side, grid_n, max_sweeps)
    t0 = time.time()
    seed_score = scorer.score(seed)        # one real eval -> timing basis
    sec = time.time() - t0
    eta_min = sec * planned / 60.0
    _nfolds = len(getattr(scorer, "_eval_folds", getattr(scorer, "_fast", [])))
    print(f"[v17:{side}] ~{planned} evals x {sec:.1f}s/eval  ->  ETA <= {eta_min:.1f} min "
          f"({_nfolds} informative folds; seed LCB {seed_score:.5f})")

    bar = None
    if show:
        try:
            from tqdm.auto import tqdm
            bar = tqdm(total=planned, desc=f"v17:{side}", unit="eval")
            bar.update(1)  # account for the seed eval already done
        except Exception:
            bar = None

    last = [1]
    def _on_eval(done: int) -> None:
        if bar is not None:
            bar.update(done - last[0]); last[0] = done
        elif done % 10 == 0:
            print(f"[v17:{side}] {done}/{planned} evals "
                  f"({(time.time()-t0)/60:.1f} min)")

    res = coordinate_ascent(seed, scorer, side=side, grid_n=grid_n,
                            max_sweeps=max_sweeps, on_eval=_on_eval,
                            seed_score=seed_score)
    if bar is not None:
        bar.close()
    return res

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
    progress: bool = True,
    fast: bool = True,
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
        if fast:
            scorer = FastPooledScorer(folds=folds, streams=kept, side=side, base_params=seed)
        else:
            scorer = PooledScorer(folds=folds, streams=kept, side=side)
        res = _ascend_with_progress(seed, scorer, side, grid_n, max_sweeps, progress)

        # Science contract: report the FINAL LCB scored by the REAL detector,
        # never the fast surrogate (which is byte-identical, but verify at runtime).
        final_lcb = res.score
        if fast:
            real_final = PooledScorer(folds=folds, streams=kept, side=side).score(res.params)
            if abs(real_final - res.score) > 1e-9:
                logger.warning("v17 %s: fast LCB %.6f != real %.6f — reporting real",
                               side, res.score, real_final)
            final_lcb = real_final

        out["sides"][side] = {
            "seed_lcb": res.seed_score,
            "final_lcb": final_lcb,
            "n_evals": res.n_evals,
            "changed": [(f, v) for f, v, _ in res.history],
            "best_params": {k: getattr(res.params, k) for k in vars(res.params)},
            "coords": res.coords,      # explored threshold field names
            "trace": res.trace,        # every evaluated config + 'score' (for the map)
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
