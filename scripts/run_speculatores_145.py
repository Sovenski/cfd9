from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Set thread caps before importing numpy/pandas-heavy modules in child processes.
for _var in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ.setdefault(_var, "1")

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.speculatores145 import (
    DEFAULT_N_TRIALS,
    DEFAULT_RESULTS_DIR,
    DEFAULT_STARTUP_TRIALS,
    DEFAULT_STABILITY_TRIALS,
    DEFAULT_STORAGE_FILE,
    DEFAULT_WORKERS_PER_SIDE,
    RunConfig,
    run_full_pipeline,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Speculatores 14.5 optimizer.")
    parser.add_argument(
        "--dataset",
        required=True,
        help="Path to the primary dataset CSV, e.g. data/raw/SPX_1D_18710201_20260318.csv",
    )
    parser.add_argument(
        "--storage",
        default=str(DEFAULT_STORAGE_FILE),
        help="Path to the Optuna journal storage file.",
    )
    parser.add_argument(
        "--results-dir",
        default=str(DEFAULT_RESULTS_DIR),
        help="Directory for the timestamped Markdown run report.",
    )
    parser.add_argument(
        "--trials-per-side",
        type=int,
        default=DEFAULT_N_TRIALS,
        help="Target completed/pruned/failed trials per side.",
    )
    parser.add_argument(
        "--workers-per-side",
        type=int,
        default=DEFAULT_WORKERS_PER_SIDE,
        help="Parallel worker processes per side.",
    )
    parser.add_argument(
        "--startup-trials",
        type=int,
        default=DEFAULT_STARTUP_TRIALS,
        help="TPE startup trials before the model becomes fully adaptive.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base random seed for samplers.",
    )
    parser.add_argument(
        "--stability-trials",
        type=int,
        default=DEFAULT_STABILITY_TRIALS,
        help="Local neighborhood trials per side to test best-param stability.",
    )
    parser.add_argument(
        "--study-prefix",
        default="speculatores_14_5",
        help="Study-name prefix used in Optuna storage.",
    )
    parser.add_argument(
        "--skip-cross-asset",
        action="store_true",
        help="Skip cross-asset evaluation in the final report.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = RunConfig(
        dataset_path=Path(args.dataset),
        storage_path=Path(args.storage),
        results_dir=Path(args.results_dir),
        trials_per_side=args.trials_per_side,
        workers_per_side=args.workers_per_side,
        startup_trials=args.startup_trials,
        seed=args.seed,
        study_prefix=args.study_prefix,
        cross_asset=not args.skip_cross_asset,
        stability_trials=args.stability_trials,
    )
    report_path = run_full_pipeline(config)
    print(f"Speculatores 14.5 report written to: {report_path}")


if __name__ == "__main__":
    main()
