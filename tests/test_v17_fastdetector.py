"""Parity spec for the v17 precompute scorer.

The FastDetector precomputes shape-dependent arrays once, then re-scores cheaply
when only threshold params change. It is ONLY legitimate if it reproduces the
real SpeculatorDetector byte-for-byte. This test is that contract: for params
that differ from a fixed base only in the coordinate-ascent threshold fields,
FastDetector.signals(params) must equal SpeculatorDetector(...).run() exactly.
"""
from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pytest

from src.detector import SpeculatorDetector, build_detector_artifacts
from src.indicators import Params
from src.pooled_validation import load_stream_frame
from src.v17_optimize import active_threshold_fields

fastmod = pytest.importorskip("src.v17_fastdetector")  # RED until module exists
FastDetector = fastmod.FastDetector

_CSV = Path("data/raw/SPX_1D_20170428_20260318.csv")


def _df():
    if not _CSV.exists():
        pytest.skip(f"missing {_CSV}")
    return load_stream_frame(str(_CSV))


def _perturb(base: Params, fields: set[str], rng) -> Params:
    over = {}
    for f in fields:
        v = float(getattr(base, f))
        nv = v * float(rng.uniform(0.7, 1.3))
        if 0.0 < v <= 1.0:           # keep pct/agreement-like in (0,1)
            nv = min(0.99, max(0.01, nv))
        over[f] = nv
    return dataclasses.replace(base, **over)


def test_fastdetector_equals_detector_on_base_params():
    df = _df()
    art = build_detector_artifacts(df)
    base = Params()
    ref = SpeculatorDetector(df, base, art).run()
    fast = FastDetector(df, base, art).signals(base)
    assert np.array_equal(ref["signal_high"].values, fast["signal_high"])
    assert np.array_equal(ref["signal_low"].values, fast["signal_low"])


def test_fastdetector_matches_under_threshold_perturbations():
    df = _df()
    art = build_detector_artifacts(df)
    base = Params()
    fd = FastDetector(df, base, art)                 # precompute once
    fields = set(active_threshold_fields(base, "high") + active_threshold_fields(base, "low"))
    rng = np.random.default_rng(0)
    for trial in range(10):
        p = _perturb(base, fields, rng)
        ref = SpeculatorDetector(df, p, art).run()
        fast = fd.signals(p)                          # cheap re-score
        assert np.array_equal(ref["signal_high"].values, fast["signal_high"]), \
            f"HIGH mismatch trial {trial}"
        assert np.array_equal(ref["signal_low"].values, fast["signal_low"]), \
            f"LOW mismatch trial {trial}"
