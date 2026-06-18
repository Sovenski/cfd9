"""L4 memory safety — spec §6 (all four requirements) + F9 chunk formula.

TDD spec for ``src/v17_card/gpu_memory.py`` (impl plan T12):

1. ``gpu_memory_estimate`` pins the §6 budget arithmetic: INDICES pool
   (3 streams x 5 folds x 2 slices, ~2850 bars) ~= 0.4 GB and ALL-1D
   (~15 streams) ~= 1.8 GB, each within 20%.
2. ``chunk_size`` == ``floor(budget / (n_lanes * max_bars * bpc))`` clamped
   to ``[1, popsize]``; ``BUDGET_BYTES == 14 GiB`` pinned (F9 — the chunk is
   computed up front, never discovered by failing; no OOM-retry loops).
3. ALL-1D pool at popsize 128: estimated peak < 18 GB AND chunk >= 1 (§6.3).
4. Chunked-vs-unchunked LCB byte-identity (§6.4): the chunk PARTITIONING
   math through the runner-side ``chunked_score_pop`` wrapper, on a
   synthetic 15-stream pool with the REAL block-bootstrap reduction, and
   on a real ``GpuPooledScorer`` small case (frozen kernels untouched —
   the wrapper lives OUTSIDE ``src/v17_gpu``).

The probe-agreement check (§6.2): ``bytes_per_candidate`` measured on a
probe batch must agree with the analytic ~16 B/bar within 2x — the test
injects fake probe measurements and asserts the gate fires both ways.
"""
from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pytest

from src.v17_card.gpu_memory import (
    BUDGET_BYTES,
    BYTES_PER_CANDIDATE_BAR,
    N_SCALES,
    PIR_BYTES,
    assert_probe_agreement,
    chunk_size,
    chunked_score_pop,
    estimate_from_folds,
    gpu_memory_estimate,
)

GIB = 1024 ** 3
GB = 1_000_000_000


# ---------------------------------------------------------------------------
# §6.1 — gpu_memory_estimate pins the budget arithmetic
# ---------------------------------------------------------------------------


def test_constants_pinned():
    assert BUDGET_BYTES == 14 * GIB          # F9: fixed, never adaptive
    assert N_SCALES == 499                   # PIR scale count (spec §6)
    assert PIR_BYTES == 8                    # f64 PIR era
    assert BYTES_PER_CANDIDATE_BAR == 16     # 8 vote bools + workspace


def test_indices_pool_estimate_within_20pct():
    est = gpu_memory_estimate(n_streams=3, n_folds=5, n_slices=2,
                              bars_per_slice=2850, popsize=0)
    assert est["pool_bytes"] == 3 * 5 * 2 * 2850 * 499 * 8
    assert abs(est["pool_bytes"] - 0.4 * GB) / (0.4 * GB) < 0.20


def test_all_1d_pool_estimate_within_20pct():
    est = gpu_memory_estimate(n_streams=15, n_folds=5, n_slices=2,
                              bars_per_slice=2850, popsize=0)
    assert abs(est["pool_bytes"] - 1.8 * GB) / (1.8 * GB) < 0.20


def test_estimate_total_includes_candidate_working_set():
    est = gpu_memory_estimate(n_streams=3, n_folds=5, n_slices=2,
                              bars_per_slice=2850, popsize=128)
    n_lanes = 3 * 5 * 2
    work = 128 * n_lanes * 2850 * BYTES_PER_CANDIDATE_BAR
    assert est["n_lanes"] == n_lanes
    assert est["work_bytes"] == work
    assert est["total_bytes"] == est["pool_bytes"] + work


# ---------------------------------------------------------------------------
# §6.2 — F9 deterministic chunk-size formula
# ---------------------------------------------------------------------------


def test_chunk_size_is_the_f9_floor_formula():
    n_lanes, max_bars, bpc = 150, 2850, BYTES_PER_CANDIDATE_BAR
    expected = int(BUDGET_BYTES // (n_lanes * max_bars * bpc))
    got = chunk_size(BUDGET_BYTES, n_lanes, max_bars, bpc, popsize=10_000)
    assert got == expected


def test_chunk_size_clamped_to_1_and_popsize():
    # tiny budget -> floor would be 0 -> clamp to 1
    assert chunk_size(1, 150, 2850, 16, popsize=128) == 1
    # huge budget -> clamp to popsize
    assert chunk_size(BUDGET_BYTES, 2, 100, 16, popsize=128) == 128


def test_chunk_size_rejects_degenerate_inputs():
    with pytest.raises(ValueError):
        chunk_size(BUDGET_BYTES, 0, 2850, 16, popsize=128)
    with pytest.raises(ValueError):
        chunk_size(BUDGET_BYTES, 150, 2850, 16, popsize=0)


def test_probe_agreement_2x_gate_fires_both_ways():
    assert_probe_agreement(16.0)                  # exact -> fine
    assert_probe_agreement(31.9)                  # < 2x -> fine
    assert_probe_agreement(8.1)                   # > 0.5x -> fine
    with pytest.raises(ValueError, match="probe"):
        assert_probe_agreement(33.0)              # > 2x analytic
    with pytest.raises(ValueError, match="probe"):
        assert_probe_agreement(7.9)               # < analytic / 2


# ---------------------------------------------------------------------------
# §6.3 — ALL-1D shape at popsize 128: peak < 18 GB and chunk >= 1
# ---------------------------------------------------------------------------


def test_all_1d_popsize_128_estimate_below_18gb_and_chunk_geq_1():
    est = gpu_memory_estimate(n_streams=15, n_folds=5, n_slices=2,
                              bars_per_slice=2850, popsize=128)
    assert est["total_bytes"] < 18 * GB, \
        f"ALL-1D popsize-128 estimate {est['total_bytes'] / GB:.2f} GB >= 18 GB"
    chunk = chunk_size(BUDGET_BYTES, est["n_lanes"], 2850,
                       BYTES_PER_CANDIDATE_BAR, popsize=128)
    assert chunk >= 1


# ---------------------------------------------------------------------------
# §6.4 — chunked vs unchunked: byte-identical LCBs
# ---------------------------------------------------------------------------


class _SyntheticPoolScorer:
    """Deterministic 15-stream-pool stand-in using the REAL LCB reduction.

    Per-candidate fold scores are a pure function of the candidate index;
    the LCB is the real ``fold_scores_bootstrap_ci`` (its internal seed is
    fixed per call), so chunk partitioning must be byte-identical.
    """

    N_STREAMS = 15
    N_FOLDS = 6

    def __init__(self) -> None:
        self.call_sizes: list[int] = []

    def score_pop(self, params_list) -> np.ndarray:
        from src.validation import fold_scores_bootstrap_ci
        self.call_sizes.append(len(params_list))
        lcbs = []
        for p in params_list:
            key = float(p)  # candidates are plain floats in this test
            fs = [abs(np.sin(key * (f + 1) * self.N_STREAMS)) * 0.3
                  for f in range(self.N_FOLDS)]
            lcbs.append(float(fold_scores_bootstrap_ci(fs)[0]))
        return np.asarray(lcbs, dtype=np.float64)


def test_chunked_score_pop_byte_identical_on_synthetic_15_stream_pool():
    cands = [0.1 * i + 0.05 for i in range(13)]
    unchunked = _SyntheticPoolScorer().score_pop(cands)
    for chunk in (1, 2, 5, 13, 64):
        sc = _SyntheticPoolScorer()
        got = chunked_score_pop(sc.score_pop, cands, chunk)
        assert got.dtype == np.float64
        assert np.array_equal(got, unchunked), f"chunk={chunk} drifted"
        assert all(s <= chunk for s in sc.call_sizes)
        assert sum(sc.call_sizes) == len(cands)


def test_chunked_score_pop_rejects_bad_chunk():
    sc = _SyntheticPoolScorer()
    with pytest.raises(ValueError):
        chunked_score_pop(sc.score_pop, [0.1], 0)


def test_chunked_score_pop_empty_population():
    sc = _SyntheticPoolScorer()
    out = chunked_score_pop(sc.score_pop, [], 4)
    assert out.shape == (0,) and out.dtype == np.float64


# --- real GpuPooledScorer small case (frozen kernels; wrapper outside) -----

_DAX = Path("data/raw/DAX_1D_19700102_20260324.csv")


def test_forced_chunk_equivalence_real_gpu_scorer_small_case():
    torch = pytest.importorskip("torch")  # noqa: F841
    if not _DAX.exists():
        pytest.skip(f"missing {_DAX}")
    from src.indicators import Params
    from src.pooled_validation import (
        StreamData, build_calendar_folds, load_stream_frame)
    from src.scoring import add_pivot_labels
    from src.universe import Stream
    from src.v17_gpu.phase2_scan import GpuPooledScorer

    df = load_stream_frame(str(_DAX)).iloc[-3000:].copy()
    add_pivot_labels(df)
    stream = Stream(ticker="DAX", timeframe="1D", path=str(_DAX),
                    cluster_id="EU_EQ")
    sd = StreamData(stream=stream, df=df, bar_seconds=86400.0)
    folds = build_calendar_folds([sd], oos_fraction=0.15)[:2]
    assert folds, "small-case pool produced no folds"

    base = Params()
    gpu = GpuPooledScorer(folds=folds, streams=[stream], side="low",
                          base_params=base, device="cpu")
    cands = [base] + [
        dataclasses.replace(base, min_agreement_low=float(v))
        for v in (0.55, 0.60, 0.65, 0.70)
    ]
    unchunked = gpu.score_pop(cands)
    chunked = chunked_score_pop(gpu.score_pop, cands, chunk=2)
    assert np.array_equal(chunked, unchunked), \
        "forced chunk=2 LCBs != unchunked (must be byte-identical)"


def test_estimate_from_folds_matches_slice_geometry():
    """Lane-level estimate: sums actual IS/OOS slice lengths."""

    class _Slice:
        def __init__(self, n_is: int, n_oos: int) -> None:
            self.df_is = np.zeros(n_is)
            self.df_oos = np.zeros(n_oos)

    folds = [[_Slice(1000, 200), _Slice(900, 200)], [_Slice(1100, 300)]]
    est = estimate_from_folds(folds, popsize=8)
    assert est["n_lanes"] == 6
    assert est["max_bars"] == 1100
    pool = (1000 + 200 + 900 + 200 + 1100 + 300) * N_SCALES * PIR_BYTES
    assert est["pool_bytes"] == pool
    assert est["work_bytes"] == 8 * 6 * 1100 * BYTES_PER_CANDIDATE_BAR
    assert est["total_bytes"] == est["pool_bytes"] + est["work_bytes"]
