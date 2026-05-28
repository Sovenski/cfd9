"""Sprint-0 acceptance: pooled objective end-to-end on SPX-1D + DAX-1D.

Proves the pooled path runs on two REAL streams with a tiny Optuna budget,
before exporting the full universe. Not a unit test — an integration gate.
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
from src.speculatores145 import params_from_trial  # noqa: E402

logging.basicConfig(level=logging.INFO)
optuna.logging.set_verbosity(optuna.logging.WARNING)

SPECS = [
    ("SPX", "data/raw/SPX_1D_18710201_20260318.csv", "US_EQ", 86400.0),
    ("DAX", "data/raw/DAX_1D_19700102_20260324.csv", "EU_EQ", 86400.0),
]


def main() -> None:
    stream_datas = []
    for ticker, path, cluster, bar_s in SPECS:
        df = load_stream_frame(str(ROOT / path))
        df, keep = apply_volume_policy(df, policy="price_only")
        assert keep
        add_pivot_labels(df)
        stream_datas.append(StreamData(
            stream=Stream(ticker, "1D", path, cluster), df=df, bar_seconds=bar_s,
        ))
        print(f"{ticker}: {len(df)} bars "
              f"HIGH={(df['pivot_N100']==1).sum()} LOW={(df['pivot_N100']==-1).sum()}")

    folds = build_calendar_folds(stream_datas)
    print(f"folds={len(folds)}; "
          f"streams/fold={[len(f) for f in folds]}")
    assert folds, "no folds built"

    streams = [sd.stream for sd in stream_datas]
    for side in ("high", "low"):
        obj = build_pooled_optuna_objective(folds, streams, params_from_trial, side)
        study = optuna.create_study(direction="maximize")
        study.optimize(obj, n_trials=5)
        print(f"[{side}] best pooled LCB over 5 trials = {study.best_value:.5f}")
    print("\nsprint0 pooled smoke PASSED")


if __name__ == "__main__":
    main()
