"""Post-hoc holdout verdict for a saved ``run_v17_gpu`` JSON (spec §A4).

Lets the CURRENT big run (already in flight on the pre-change code) get its
holdout -> era_pass verdict WITHOUT a rerun: it loads the saved winner
``best_params`` + pool config from the run JSON, rebuilds the pool exactly
like ``run_v17_gpu`` does (resolve_streams -> volume policy ->
add_pivot_labels -> calendar folds), evaluates the embargoed holdout slices
for every side present, applies the pre-committed ``holdout_era_pass`` rule
(ERA_PASS_MIN_RATIO = 0.5) and prints the same HOLDOUT block Cell 6 prints.

Usage (from the repo root):

    python temp/holdout_posthoc.py <run_json> <data_dir> \
        [--is-fraction 0.20] [--oos-fraction 0.15] [--step-fraction 0.10] \
        [--holdout-fraction 0.20] [--start YYYY-MM-DD] [--min-streams N] \
        [--coverage 0.5]

The fold-fraction defaults match the H100 workbook Cell 5 defaults (the
in-flight big run). They affect ONLY the fold-mean denominator of the
era_pass ratio — the holdout boundary itself depends only on
``--holdout-fraction`` and the era arguments.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from src.indicators import Params  # noqa: E402
from src.pooled_validation import (  # noqa: E402
    ERA_PASS_MIN_RATIO,
    StreamData,
    apply_volume_policy,
    build_calendar_folds,
    build_holdout_slices,
    cluster_weights,
    evaluate_holdout,
    holdout_era_pass,
    load_stream_frame,
)
from src.scoring import add_pivot_labels  # noqa: E402
from src.universe import resolve_streams  # noqa: E402
from src.v17_acceptance import raw_fold_scores  # noqa: E402
from src.v17_optimize import PooledScorer  # noqa: E402

logger = logging.getLogger(__name__)

_TF_SECONDS = {"1D": 86400.0, "1W": 604800.0, "60": 3600.0,
               "240": 14400.0, "1m": 60.0}


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_json", help="saved run_v17_gpu results JSON")
    p.add_argument("data_dir", help="directory with the run's CSV pool")
    p.add_argument("--is-fraction", type=float, default=0.20)
    p.add_argument("--oos-fraction", type=float, default=0.15)
    p.add_argument("--step-fraction", type=float, default=0.10)
    p.add_argument("--holdout-fraction", type=float, default=0.20)
    p.add_argument("--start", type=str, default=None)
    p.add_argument("--min-streams", type=int, default=None)
    p.add_argument("--coverage", type=float, default=0.5)
    return p.parse_args(argv)


def _f(x: object, n: int = 5) -> str:
    return f"{x:.{n}f}" if isinstance(x, (int, float)) and x is not None else str(x)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    run = json.loads(Path(args.run_json).read_text())
    era_kw = {"is_fraction": args.is_fraction,
              "oos_fraction": args.oos_fraction,
              "step_fraction": args.step_fraction,
              "holdout_fraction": args.holdout_fraction,
              "start": args.start, "min_streams": args.min_streams,
              "coverage": args.coverage}

    # --- rebuild the pool exactly like run_v17_gpu --------------------------
    groups = run["groups"]
    timeframes = run["timeframes"]
    volume_policy = run.get("volume_policy", "price_only")
    streams = resolve_streams(groups, timeframes, data_dir=args.data_dir)
    if not streams:
        raise RuntimeError(
            f"0 streams resolved from {args.data_dir} for {groups} {timeframes}")
    stream_datas: list[StreamData] = []
    for s in streams:
        df = load_stream_frame(s.path)
        df, keep = apply_volume_policy(df, policy=volume_policy)
        if not keep:
            logger.info("drop %s: volume policy", s.stream_id)
            continue
        add_pivot_labels(df)
        stream_datas.append(StreamData(
            stream=s, df=df, bar_seconds=_TF_SECONDS[s.timeframe]))
    folds = build_calendar_folds(stream_datas, **era_kw)
    kept = [sd.stream for sd in stream_datas]
    if not folds:
        raise RuntimeError("No folds — check the fold-fraction arguments.")
    weights = cluster_weights(kept)
    slices, meta = build_holdout_slices(stream_datas, **era_kw)
    if not slices:
        raise RuntimeError("No holdout slices — pool tail too short for the "
                           "embargoed holdout.")

    print("=" * 78)
    print(f"POST-HOC HOLDOUT VERDICT — {run.get('run_slug')}")
    print("=" * 78)
    print(f"pool:    {sorted(s.stream_id for s in kept)}  ({len(folds)} folds)")
    print(f"holdout: {meta['n_slices']} slices  start={meta['holdout_start']}"
          f"  embargo={meta['embargo_bars']} bars"
          f"  (fraction={meta['holdout_fraction']})")

    for side, d in run.get("sides", {}).items():
        wp = Params(**d["best_params"])
        rfs = [float(s) for s in raw_fold_scores(
            PooledScorer(folds=folds, streams=kept, side=side), wp)]
        fold_mean = float(np.mean(rfs)) if rfs else 0.0
        h_score, h_comp, h_per_stream = evaluate_holdout(wp, side, slices,
                                                         weights)
        h_pass = holdout_era_pass(h_score, rfs)
        ratio = (h_score / fold_mean) if fold_mean > 0 else None

        print()
        print("-" * 78)
        print(f"{side.upper()} SIDE — winner from {Path(args.run_json).name}")
        print("-" * 78)
        print("HOLDOUT (selection-untouched OOS, embargoed reserved tail):")
        print(f"  score {_f(h_score)} vs fold-mean {_f(fold_mean)}"
              f"   ratio {_f(ratio, 3)} (min {ERA_PASS_MIN_RATIO})"
              f"   -> era_pass={h_pass}")
        print(f"  n_slices={meta['n_slices']}  start={meta['holdout_start']}"
              f"  embargo={meta['embargo_bars']} bars"
              f"  n_signals={_f(h_comp.get('n_signals'), 1)}")
        print(f"  precision_w={_f(h_comp.get('precision_w'), 4)}"
              f"  recall_w={_f(h_comp.get('recall_w'), 4)}"
              f"  tp_mass={_f(h_comp.get('tp_mass'), 3)}"
              f"  total_mass={_f(h_comp.get('total_mass'), 3)}")
        print(f"  fold basis: {len(rfs)} raw winner fold scores"
              f"  (saved final_lcb={_f(d.get('final_lcb'))})")
        for ps in h_per_stream[:8]:
            print(f"    {str(ps.get('stream_id')):<14s}"
                  f" n_sig={ps.get('n_signals'):>4}"
                  f"  tp_mass={_f(ps.get('tp_mass'), 2)}"
                  f" / total {_f(ps.get('total_mass'), 2)}"
                  f"  (w={_f(ps.get('weight'), 2)})")
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    raise SystemExit(main())
