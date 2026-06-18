"""Spec for v17 anti-gaming pieces: firing-rate penalty, boundary-pin detection,
and bootstrap-seed stability gate."""
from __future__ import annotations

import numpy as np
import pytest

accept = pytest.importorskip("src.v17_acceptance")  # RED until module exists
firing_excess = accept.firing_excess
boundary_pinned = accept.boundary_pinned
bootstrap_stability = accept.bootstrap_stability


def test_firing_excess_penalizes_spray():
    # HIGH gamed fold: precision 0.10, recall 1.2x => ratio 12 => excess 10 over cap 2
    assert firing_excess(0.10, 1.20, cap=2.0) == pytest.approx(10.0, abs=1e-6)
    # healthy: ratio recall/precision = 0.58 < cap => no penalty
    assert firing_excess(0.364, 0.211, cap=2.0) == 0.0
    # pure false-positive spray (precision 0) => large penalty
    assert firing_excess(0.0, 0.5, cap=2.0) > 100.0
    # no signals/catches => no penalty
    assert firing_excess(0.0, 0.0, cap=2.0) == 0.0


def test_boundary_pinned_flags_bound_hits():
    # the actual run: HIGH min_agreement at lower bound 0.10, LOW scale_div at upper 0.60
    hi = boundary_pinned([("min_agreement_high", 0.10), ("scale_div_thresh_high", 0.35)], "high")
    assert ("min_agreement_high", "lo") in hi
    assert all(f != "scale_div_thresh_high" for f, _ in hi)  # 0.35 is interior
    # LOW scale_div hi widened to 0.95 (2026-06-11): the hi-pin example tracks
    # the current bound; min_agreement lo widened to 0.02.
    lo = boundary_pinned([("scale_div_thresh_low", 0.95), ("min_agreement_low", 0.02)], "low")
    assert ("scale_div_thresh_low", "hi") in lo
    assert ("min_agreement_low", "lo") in lo


def test_bootstrap_stability_passes_stable_fails_fragile():
    rng = np.random.default_rng(0)
    stable = list(0.12 + 0.005 * rng.standard_normal(12))   # tight, all positive
    s = bootstrap_stability(stable, seeds=range(10), n_boot=300)
    assert s["pass"] is True
    assert s["min"] > 0 and s["std"] <= 0.5 * s["mean"]

    fragile = [0.45, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]           # one nonzero fold
    f = bootstrap_stability(fragile, seeds=range(10), n_boot=300)
    assert f["pass"] is False                                # some seeds bootstrap to ~0
