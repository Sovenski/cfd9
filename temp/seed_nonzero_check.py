"""De-risk check (T7): verify a non-zero pooled LCB is achievable on real data.

Loads SPX-1D + DAX-1D as StreamData (price_only, add_pivot_labels), builds
calendar folds, builds the pooled LOW objective, creates a study, enqueues
SEED_HEURISTIC_STRUCTURAL_LOW, runs ~12 trials, and prints the best LCB.

GOAL: confirm a NON-zero best LCB is achievable (proves the signal→pivot
matching path works end-to-end on real pooled data).
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import optuna

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.scoring import add_pivot_labels  # noqa: E402
from src.universe import Stream  # noqa: E402
from src.pooled_validation import (  # noqa: E402
    StreamData, build_calendar_folds, build_pooled_optuna_objective,
    load_stream_frame, apply_volume_policy,
)
from src.speculatores145 import (  # noqa: E402
    params_from_trial,
    SEED_HEURISTIC_STRUCTURAL_LOW,
)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")
optuna.logging.set_verbosity(optuna.logging.WARNING)

SPECS = [
    ("SPX", "data/raw/SPX_1D_18710201_20260318.csv", "US_EQ", 86400.0),
    ("DAX", "data/raw/DAX_1D_19700102_20260324.csv", "EU_EQ", 86400.0),
]


def main() -> None:
    stream_datas = []
    for ticker, rel_path, cluster, bar_s in SPECS:
        path = str(ROOT / rel_path)
        df = load_stream_frame(path)
        df, keep = apply_volume_policy(df, policy="price_only")
        assert keep, f"{ticker}: price_only policy rejected stream"
        add_pivot_labels(df)
        stream_datas.append(StreamData(
            stream=Stream(ticker, "1D", path, cluster),
            df=df,
            bar_seconds=bar_s,
        ))
        n_high = (df["pivot_N100"] == 1).sum()
        n_low = (df["pivot_N100"] == -1).sum()
        print(f"{ticker}: {len(df)} bars  HIGH={n_high}  LOW={n_low}")

    folds = build_calendar_folds(stream_datas)
    print(f"folds={len(folds)}  streams/fold={[len(f) for f in folds]}")
    assert folds, "ERROR: no folds built — cannot run check"

    streams = [sd.stream for sd in stream_datas]
    side = "low"
    objective = build_pooled_optuna_objective(folds, streams, params_from_trial, side)
    study = optuna.create_study(direction="maximize")
    study.enqueue_trial(SEED_HEURISTIC_STRUCTURAL_LOW)

    print(f"\nRunning 12 trials (side={side}, seed trial enqueued) ...")
    study.optimize(objective, n_trials=12, show_progress_bar=False)

    print("\n--- Per-trial results ---")
    for t in study.trials:
        state = t.state.name
        val = f"{t.value:.6f}" if t.value is not None else "None"
        print(f"  trial {t.number:2d}  state={state:9s}  value={val}")

    best = study.best_value
    print(f"\nbest pooled LCB (low, 12 trials) = {best:.6f}")
    if best > 0.0:
        print("RESULT: NON-ZERO — signal→pivot matching path verified on pooled data.")
    else:
        print("RESULT: ZERO — investigate per-trial output above.")


if __name__ == "__main__":
    main()
