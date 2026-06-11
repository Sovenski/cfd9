"""v18 repair tests (plan/v18-repair-spec.md) — Stage A.

A2 — ``pir_of`` partial-window warm-up (spec P2.1): ``rolling(lb)`` must use
``min_periods=1`` so warm-up bars get a value from the AVAILABLE bars, matching
Pine's parity-shim scan (proven by the e830d TV-export audit flips). The
post-warm-up region must be bit-identical to the historical full-window result.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.indicators import pir_of


def test_pir_of_partial_window_warmup():
    rng = np.random.default_rng(42)
    val = pd.Series(rng.normal(100.0, 5.0, 300))
    lb = 20

    out = pir_of(val, lb)

    # 1. No NaN anywhere — warm-up bars use the available (partial) window.
    assert not out.isna().any(), "pir_of must be defined from bar 0"

    # 2. Every value in [0, 1].
    assert ((out >= 0.0) & (out <= 1.0)).all()

    # 3. Bar 0: window of one bar -> hi == lo -> the 0.5 degenerate rule.
    assert out.iloc[0] == 0.5

    # 4. Post-warm-up (bar >= lb-1): EXACT equality with the historical
    #    full-window computation (min_periods == window).
    lo = val.rolling(lb).min()
    hi = val.rolling(lb).max()
    span = (hi - lo).clip(lower=1e-10)
    full = ((val - lo) / span).where(hi != lo, 0.5)
    np.testing.assert_array_equal(
        out.iloc[lb - 1:].to_numpy(), full.iloc[lb - 1:].to_numpy(),
        err_msg="pir_of post-warm-up region must be bit-identical")


def test_pir_of_warmup_partial_window_values():
    # Monotonic series: at bar t < lb-1 the partial window is val[0..t], so
    # the current value is the max -> pir == 1.0 for t >= 1.
    val = pd.Series(np.arange(1.0, 51.0))
    out = pir_of(val, 10)
    assert out.iloc[0] == 0.5            # single-bar window, hi == lo
    assert (out.iloc[1:] == 1.0).all()   # rising series pins to the top
