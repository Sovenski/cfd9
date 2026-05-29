"""Functional test: real ascent -> trace -> the notebook's viz cell renders a PNG.

Validates (a) coordinate_ascent emits a usable trace+coords, and (b) the exact
viz-cell source from optimize_v17.ipynb executes headless and produces a figure.
"""
from __future__ import annotations
import os
os.environ["MPLBACKEND"] = "Agg"
import json
import numpy as np
from pathlib import Path

from src.indicators import Params
from src.pooled_validation import StreamData, build_calendar_folds, load_stream_frame
from src.scoring import add_pivot_labels
from src.universe import Stream
from src.v17_optimize import PooledScorer, coordinate_ascent

# --- real ascent on a small DAX pool -> populate trace ---
df = load_stream_frame("data/raw/DAX_1D_19700102_20260324.csv").iloc[-5000:].copy()
add_pivot_labels(df)
stream = Stream(ticker="DAX", timeframe="1D", path="x", cluster_id="EU_EQ")
sd = StreamData(stream=stream, df=df, bar_seconds=86400.0)
folds = build_calendar_folds([sd])[:4]
scorer = PooledScorer(folds=folds, streams=[stream], side="low")
res = coordinate_ascent(Params(), scorer, side="low", grid_n=3, max_sweeps=1)
assert res.trace and len(res.trace) > 1, "trace not populated"
assert res.coords and all(c in res.trace[0] for c in res.coords), "coords/trace mismatch"
print(f"real ascent: {len(res.trace)} trace points, {len(res.coords)} coords, "
      f"n_evals={res.n_evals}")

# --- build a v17_out; add a synthetic HIGH with varied scores to exercise color ---
rng = np.random.default_rng(0)
coords = res.coords
hi_trace = []
for i in range(40):
    pt = {c: float(rng.normal(0.3, 0.05)) for c in coords}
    pt["score"] = float(0.05 + 0.02 * rng.standard_normal() + 0.001 * i)
    pt["varied"] = "seed" if i == 0 else coords[i % len(coords)]
    hi_trace.append(pt)
v17_out = {"sides": {
    "high": {"coords": coords, "trace": hi_trace, "seed_lcb": hi_trace[0]["score"],
             "final_lcb": max(t["score"] for t in hi_trace)},
    "low":  {"coords": res.coords, "trace": res.trace, "seed_lcb": res.seed_score,
             "final_lcb": res.score},
}}

# --- exec the EXACT viz-cell source from the notebook ---
nb = json.loads(Path("optimize_v17.ipynb").read_text(encoding="utf-8"))
viz_src = next("".join(c["source"]) for c in nb["cells"]
               if "explored-space map" in "".join(c["source"]))
ns = {"v17_out": v17_out}
exec(viz_src, ns)

out_png = Path("temp/v17_map_test.png")
ns["fig"].savefig(out_png, dpi=80)
assert out_png.exists() and out_png.stat().st_size > 5000, "PNG not rendered"
print(f"viz cell executed; wrote {out_png} ({out_png.stat().st_size} bytes)")
print("V17 VIZ TEST PASSED")
