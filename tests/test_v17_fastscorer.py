"""Equivalence spec: FastPooledScorer must return the SAME pooled LCB as the
real-detector PooledScorer (because FastDetector signals are byte-identical).
"""
from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pytest

from src.indicators import Params
from src.pooled_validation import StreamData, build_calendar_folds, load_stream_frame
from src.scoring import add_pivot_labels
from src.universe import Stream
from src.v17_optimize import PooledScorer, active_threshold_fields

fastmod = pytest.importorskip("src.v17_fastdetector")
FastPooledScorer = getattr(fastmod, "FastPooledScorer", None)

_CSV = Path("data/raw/DAX_1D_19700102_20260324.csv")


def _folds():
    if not _CSV.exists():
        pytest.skip(f"missing {_CSV}")
    df = load_stream_frame(str(_CSV)).iloc[-6000:].copy()
    add_pivot_labels(df)
    s = Stream(ticker="DAX", timeframe="1D", path=str(_CSV), cluster_id="EU_EQ")
    sd = StreamData(stream=s, df=df, bar_seconds=86400.0)
    return build_calendar_folds([sd])[:4], [s]


@pytest.mark.skipif(FastPooledScorer is None, reason="FastPooledScorer not implemented yet")
def test_fast_scorer_equals_real_scorer():
    folds, streams = _folds()
    base = Params()
    real = PooledScorer(folds=folds, streams=streams, side="low")
    fast = FastPooledScorer(folds=folds, streams=streams, side="low", base_params=base)
    rng = np.random.default_rng(1)
    fields = active_threshold_fields(base, "low")
    for _ in range(4):
        over = {}
        for f in fields:
            v = float(getattr(base, f))
            nv = v * float(rng.uniform(0.8, 1.2))
            over[f] = min(0.99, max(0.01, nv)) if 0 < v <= 1 else nv
        p = dataclasses.replace(base, **over)
        assert fast.score(p) == pytest.approx(real.score(p), abs=1e-12)
