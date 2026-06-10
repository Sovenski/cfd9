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
    per_asset_high_diagnostic,
)
from .scoring import add_pivot_labels
from .speculatores145 import params_from_trial
from .universe import resolve_streams
from .v17_acceptance import (
    boundary_pinned,
    bootstrap_stability,
    raw_fold_scores,
    summarize_acceptance,
)
from .v17_fastdetector import FastPooledScorer
from .v17_finalists import filter_finalists, topk_for_flip_rate, tv_export_audit
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
    search: str = "ascent",
    search_kw: Optional[dict] = None,
) -> dict:
    """Resolve a pool, build folds, and run the per-side threshold search.

    ``search="ascent"`` (default, unchanged) runs coordinate ascent;
    ``search="cma"`` runs the Sobol + separable CMA-ES ``BatchOptimizer``
    (spec §3) against the same scorer. ``search_kw`` feeds
    ``BatchSearchConfig`` (e.g. popsize/sobol_n/generations/top_k/rng_seed).
    """
    if search not in ("ascent", "cma"):
        raise ValueError(f"search must be 'ascent'|'cma', got {search!r}")
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
        if search == "cma":
            from .v17_search import BatchOptimizer, BatchSearchConfig
            cfg = BatchSearchConfig(**(search_kw or {}))
            res = BatchOptimizer(seed=seed, scorer=scorer, side=side, config=cfg).run()
        else:
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
        if search == "cma":
            # Extra keys ONLY on the cma route — default output stays byte-identical.
            out["sides"][side]["search"] = "cma"
            out["sides"][side]["top_k"] = [
                {**{f: float(getattr(c.params, f)) for f in res.coords},
                 "score": float(c.score), "stage": c.stage}
                for c in res.top_k
            ]
            # Spec §4: deflate the REPORTED number for the batched trial
            # count (lambda x generations + sobol_n + seed = res.n_evals).
            # final_lcb stays the raw real-detector LCB; the deflated value
            # is the headline to quote for a batched search.
            from .overfit_guard import deflated_best
            pop_scores = [float(c.score) for c in res.population]
            defl = deflated_best(final_lcb, pop_scores, n_trials=res.n_evals)
            out["sides"][side]["final_lcb_deflated"] = defl["deflated"]
            out["sides"][side]["deflation"] = defl
            logger.info("[v17:%s] deflated LCB %.5f (raw %.5f, haircut %.5f "
                        "over %d trials)", side, defl["deflated"], final_lcb,
                        defl["haircut"], defl["n_trials"])
            if side == "high":
                # Spec §4: per-asset-then-aggregate HIGH diagnostic next to
                # the pooled LCB; PBO/percentile stay ADVISORY (never a gate).
                diag = per_asset_high_diagnostic(folds, res.params, side="high")
                diag["advisory"]["selection_percentile"] = float(
                    sum(s < res.score for s in pop_scores)
                ) / max(len(pop_scores), 1)
                out["sides"]["high"]["per_asset_diagnostic"] = diag
        logger.info("[v17:%s] seed=%.5f -> final=%.5f (%d evals)",
                    side, res.seed_score, res.score, res.n_evals)

    if results_dir:
        Path(results_dir).mkdir(parents=True, exist_ok=True)
        p = Path(results_dir) / f"{run_slug}_v17.json"
        p.write_text(json.dumps(out, indent=2, default=str))
        out["_written"] = str(p)
        logger.info("wrote %s", p)
    return out


def run_v17_gpu(
    groups: list[str],
    timeframes: list[str],
    data_dir: str,
    sides: tuple[str, ...] = ("high", "low"),
    seed_params: Optional[dict] = None,
    volume_policy: str = "price_only",
    era_kw: Optional[dict] = None,
    results_dir: Optional[str] = None,
    run_slug: str = "v17_gpu",
    search_kw: Optional[dict] = None,
    flip_rate: float = 0.0,          # §2 spike: trust-kernel measured 0.0
    finalist_tol: float = 1e-9,
    device: str = "cpu",
    era_pass: Optional[bool] = None,  # external era-robustness check, if any
    tv_audit: bool = True,
) -> dict:
    """GPU batched search -> EXACT CPU finalist re-score -> gates -> audit (§6).

    Pipeline per side: ``GpuPooledScorer.score_pop`` evaluates the whole
    Sobol+CMA population in ONE segmented scan; the top-K finalists (K sized
    to the §2 flip rate unless ``search_kw['top_k']`` overrides) are re-scored
    by the EXACT ``SpeculatorDetector``+``PooledScorer``; ``filter_finalists``
    HARD-drops any finalist whose GPU LCB disagrees beyond ``finalist_tol``;
    ``v17_acceptance`` gates run on the winner; ``tv_export_audit`` records
    per-asset Pine parity. Every reported number is the CPU detector's.
    """
    from .overfit_guard import deflated_best
    from .v17_gpu.phase2_scan import GpuPooledScorer   # torch stays optional
    from .v17_search import BatchOptimizer, BatchSearchConfig

    if any(side not in ("high", "low") for side in sides):
        raise ValueError(f"sides must be 'high'|'low', got {sides!r}")

    class _PopOptimizer(BatchOptimizer):
        """BatchOptimizer whose batch eval is ONE ``score_pop`` call."""

        def _score_batch(self, candidates: list[Params]) -> list[float]:
            return [float(s) for s in self.scorer.score_pop(candidates)]

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
    if not folds:
        raise RuntimeError("No folds — pool too small/short.")
    logger.info("run_v17_gpu: %d folds; streams=%s; device=%s",
                len(folds), [s.stream_id for s in kept], device)

    kw = dict(search_kw or {})
    kw.setdefault("top_k", topk_for_flip_rate(flip_rate))
    cfg = BatchSearchConfig(**kw)
    out: dict = {
        "run_slug": run_slug, "groups": groups, "timeframes": timeframes,
        "volume_policy": volume_policy, "n_folds": len(folds),
        "streams": [s.stream_id for s in kept],
        "search": "cma-gpu", "device": device,
        "flip_rate": flip_rate, "top_k": cfg.top_k,
        "finalist_tol": finalist_tol, "sides": {},
    }
    for side in sides:
        seed = (seed_from_trial_dict(seed_params[side], side)
                if seed_params and side in seed_params else Params())
        gpu = GpuPooledScorer(folds=folds, streams=kept, side=side,
                              base_params=seed, device=device)
        res = _PopOptimizer(seed=seed, scorer=gpu, side=side, config=cfg).run()

        # Science contract (§6): finalists re-scored by the EXACT CPU scorer;
        # parity violations are HARD-dropped, never warned away.
        real = PooledScorer(folds=folds, streams=kept, side=side)
        survivors, dropped = filter_finalists(res.top_k, real, tol=finalist_tol)
        if not survivors:
            raise RuntimeError(
                f"run_v17_gpu[{side}]: ALL {len(dropped)} finalists failed the "
                f"|gpu-cpu| <= {finalist_tol:g} filter — GPU parity is broken; "
                "re-run the §2 spike / tests/test_v17_gpu_parity.py.")
        winner = survivors[0]
        wp: Params = winner["candidate"].params
        final_lcb = float(winner["cpu_lcb"])

        # Acceptance gates (v17_acceptance) on the RAW real-detector objective.
        changed = [(f, float(getattr(wp, f))) for f in res.coords
                   if float(getattr(wp, f)) != float(getattr(seed, f))]
        pinned = boundary_pinned(changed, side)
        stab = bootstrap_stability(raw_fold_scores(real, wp))
        verdict = summarize_acceptance(stab["pass"], bool(era_pass), pinned)

        pop_scores = [float(c.score) for c in res.population]
        defl = deflated_best(final_lcb, pop_scores, n_trials=res.n_evals)
        out["sides"][side] = {
            "seed_lcb": res.seed_score,
            "final_lcb": final_lcb,
            "final_lcb_deflated": defl["deflated"],
            "deflation": defl,
            "n_evals": res.n_evals,
            "coords": res.coords,
            "changed": changed,
            "best_params": {k: getattr(wp, k) for k in vars(wp)},
            "leaderboard": [
                {**{f: float(getattr(e["candidate"].params, f)) for f in res.coords},
                 "gpu_lcb": e["gpu_lcb"], "cpu_lcb": e["cpu_lcb"],
                 "abs_diff": e["abs_diff"], "stage": e["candidate"].stage}
                for e in survivors],
            "n_finalists": len(survivors),
            "n_dropped_finalists": len(dropped),
            "dropped": [{"gpu_lcb": e["gpu_lcb"], "cpu_lcb": e["cpu_lcb"],
                         "abs_diff": e["abs_diff"]} for e in dropped],
            "acceptance": {"verdict": verdict, "pinned": pinned,
                           "bootstrap": stab, "era_pass": era_pass},
        }
        if side == "high":
            diag = per_asset_high_diagnostic(folds, wp, side="high")
            diag["advisory"]["selection_percentile"] = float(
                sum(sc < winner["gpu_lcb"] for sc in pop_scores)
            ) / max(len(pop_scores), 1)
            out["sides"]["high"]["per_asset_diagnostic"] = diag
        logger.info("[v17_gpu:%s] seed=%.5f -> final=%.5f (%d evals, "
                    "%d finalists kept, %d dropped, verdict=%s)",
                    side, res.seed_score, final_lcb, res.n_evals,
                    len(survivors), len(dropped), verdict)

    out["tv_audit"] = tv_export_audit(kept) if tv_audit else None
    if results_dir:
        Path(results_dir).mkdir(parents=True, exist_ok=True)
        p = Path(results_dir) / f"{run_slug}_v17gpu.json"
        p.write_text(json.dumps(out, indent=2, default=str))
        out["_written"] = str(p)
        logger.info("wrote %s", p)
    return out


__all__ = ["run_v17", "run_v17_gpu", "seed_from_trial_dict",
           "filter_finalists", "topk_for_flip_rate", "tv_export_audit"]
