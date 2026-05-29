"""Fast end-to-end smoke for v17 coordinate-ascent on REAL data.

Single stream (DAX_1D), capped + few folds, so it runs in ~1-2 min. Proves:
- folds build, PooledScorer evaluates the real detector,
- coordinate_ascent runs, emits a valid Params, finite LCB, and never regresses.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path

import pandas as pd

from src.indicators import Params
from src.pooled_validation import StreamData, build_calendar_folds, load_stream_frame
from src.scoring import add_pivot_labels
from src.universe import Stream
from src.v17_optimize import PooledScorer, coordinate_ascent

logging.basicConfig(level=logging.WARNING)

CSV = Path("data/raw/DAX_1D_19700102_20260324.csv")
SIDE = "low"

df = load_stream_frame(str(CSV))
df = df.iloc[-8000:].copy()          # cap for a fast smoke
add_pivot_labels(df)
stream = Stream(ticker="DAX", timeframe="1D", path=str(CSV), cluster_id="EU_EQ")
sd = StreamData(stream=stream, df=df, bar_seconds=86400.0)

folds = build_calendar_folds([sd])
print(f"built {len(folds)} folds; streams/fold = {[len(f) for f in folds]}")
folds = folds[:6]                    # bound smoke cost

scorer = PooledScorer(folds=folds, streams=[stream], side=SIDE)
seed = Params()

t0 = time.time()
res = coordinate_ascent(seed, scorer, side=SIDE, grid_n=3, max_sweeps=1,
                        progress=print)
dt = time.time() - t0

print("\n--- v17 smoke result ---")
print(f"side             : {SIDE}")
print(f"seed LCB         : {res.seed_score:.6f}")
print(f"final LCB        : {res.score:.6f}")
print(f"evals            : {res.n_evals}")
print(f"changed fields   : {[(f, round(v,5)) for f,v,_ in res.history]}")
print(f"wall-clock       : {dt:.1f}s")

assert res.score >= res.seed_score - 1e-9, "ascent regressed below seed"
assert res.score == res.score, "NaN score"  # not NaN
assert isinstance(res.params, Params)
print("\nv17 smoke PASSED")
