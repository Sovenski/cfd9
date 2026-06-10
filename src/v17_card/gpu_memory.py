"""L4 memory safety — estimate + F9 deterministic chunk formula (spec §6).

Pure functions, NO GPU import: ``src/v17_gpu/**`` is frozen, so the chunk is
computed HERE (runner side) and applied by partitioning the candidate list
around the unchanged ``GpuPooledScorer.score_pop`` — never inside the
kernels. Deterministic => testable => no adaptive OOM-retry logic anywhere
(F9): the chunk size is computed up front, not discovered by failing.

Budget model (f64 PIR era, spec §6):

* pool: every IS/OOS slice holds a ``N_SCALES x bars`` f64 PIR matrix
  (e.g. a 2,850-bar slice ~= 11.4 MB; INDICES pool ~= 0.4 GB, ALL-1D
  ~= 1.8 GB).
* per-candidate working set: vote/agreement tensors, ~16 B per (candidate,
  lane, bar) — 8 vote bools + workspace (``BYTES_PER_CANDIDATE_BAR``,
  analytic; a probe measurement must agree within 2x, §6.2).

The F9 chunk formula::

    chunk = floor(budget_bytes / (n_lanes * max_bars * bytes_per_candidate))

with ``budget_bytes = 14 GiB`` FIXED (leaves ~8.5 GB headroom on the
22.5 GB L4 for the PIR pool + framework overhead), clamped to
``[1, popsize]``. The formula, its inputs, and the resulting chunk are
logged at run start by ``run_v17_gpu``.
"""
from __future__ import annotations

import logging
from typing import Callable, Sequence

import numpy as np

logger = logging.getLogger(__name__)

#: F9 fixed candidate-chunk budget — 14 GiB, NEVER adaptive (spec §6.2).
BUDGET_BYTES: int = 14 * 1024 ** 3

#: PIR scale count per slice (the 499-scale f64 PIR era, spec §6).
N_SCALES: int = 499

#: Bytes per PIR matrix element (f64).
PIR_BYTES: int = 8

#: Analytic per-(candidate, lane, bar) working-set bytes: 8 vote bools +
#: workspace (spec §6.2). A probe measurement must agree within 2x —
#: asserted via :func:`assert_probe_agreement`.
BYTES_PER_CANDIDATE_BAR: int = 16


def gpu_memory_estimate(
    n_streams: int,
    n_folds: int,
    n_slices: int,
    bars_per_slice: int,
    popsize: int,
    n_scales: int = N_SCALES,
    bytes_per_candidate: int = BYTES_PER_CANDIDATE_BAR,
) -> dict[str, int]:
    """Estimated peak GPU bytes for a uniform pool shape (spec §6.1).

    Args:
        n_streams: Streams in the pool.
        n_folds: Calendar folds.
        n_slices: Slices per (stream, fold) — IS + OOS = 2.
        bars_per_slice: Bars per slice (uniform-shape approximation).
        popsize: Candidate batch size (0 = pool only).

    Returns:
        Dict with ``n_lanes``, ``pool_bytes`` (PIR matrices),
        ``work_bytes`` (per-candidate tensors) and ``total_bytes``.
    """
    n_lanes = n_streams * n_folds * n_slices
    pool = n_lanes * bars_per_slice * n_scales * PIR_BYTES
    work = popsize * n_lanes * bars_per_slice * bytes_per_candidate
    return {"n_lanes": n_lanes, "pool_bytes": int(pool),
            "work_bytes": int(work), "total_bytes": int(pool + work)}


def estimate_from_folds(folds: Sequence[Sequence[object]],
                        popsize: int) -> dict[str, int]:
    """Lane-level estimate from REAL fold geometry (run-start logging).

    Sums the actual IS/OOS slice lengths instead of the uniform-shape
    approximation; ``max_bars`` is the longest slice (the F9 formula input).
    """
    lengths: list[int] = []
    for fold in folds:
        for sl in fold:
            lengths.append(len(sl.df_is))
            lengths.append(len(sl.df_oos))
    if not lengths:
        raise ValueError("estimate_from_folds: empty fold list")
    n_lanes = len(lengths)
    max_bars = max(lengths)
    pool = sum(lengths) * N_SCALES * PIR_BYTES
    work = popsize * n_lanes * max_bars * BYTES_PER_CANDIDATE_BAR
    return {"n_lanes": n_lanes, "max_bars": int(max_bars),
            "pool_bytes": int(pool), "work_bytes": int(work),
            "total_bytes": int(pool + work)}


def chunk_size(
    budget_bytes: int,
    n_lanes: int,
    max_bars: int,
    bytes_per_candidate: int,
    popsize: int,
) -> int:
    """THE F9 deterministic candidate-chunk formula (spec §6.2).

    ``chunk = floor(budget_bytes / (n_lanes * max_bars *
    bytes_per_candidate))``, clamped to ``[1, popsize]``. No OOM-retry
    loops anywhere — the chunk is computed up front.
    """
    if n_lanes <= 0 or max_bars <= 0 or bytes_per_candidate <= 0:
        raise ValueError(
            f"chunk_size needs positive n_lanes/max_bars/bytes_per_candidate,"
            f" got {n_lanes}/{max_bars}/{bytes_per_candidate}")
    if popsize <= 0:
        raise ValueError(f"popsize must be >= 1, got {popsize}")
    raw = int(budget_bytes // (n_lanes * max_bars * bytes_per_candidate))
    return max(1, min(raw, popsize))


def assert_probe_agreement(
    measured_bytes_per_bar: float,
    analytic: float = float(BYTES_PER_CANDIDATE_BAR),
) -> None:
    """§6.2 probe gate: measured bytes/candidate-bar within 2x of analytic.

    Raises:
        ValueError: When the probe measurement disagrees with the analytic
            ~16 B/bar estimate by more than 2x in either direction — the
            budget model is then wrong and must be re-derived, not patched.
    """
    if measured_bytes_per_bar <= 0:
        raise ValueError(
            f"probe measurement must be positive, got {measured_bytes_per_bar}")
    ratio = measured_bytes_per_bar / analytic
    if ratio > 2.0 or ratio < 0.5:
        raise ValueError(
            f"probe bytes/candidate-bar {measured_bytes_per_bar:.1f} disagrees "
            f"with analytic {analytic:.1f} beyond 2x (ratio {ratio:.2f}) — "
            "re-derive the §6 budget model")


def chunked_score_pop(
    score_pop: Callable[[Sequence[object]], np.ndarray],
    params_list: Sequence[object],
    chunk: int,
) -> np.ndarray:
    """Runner-side candidate partitioning around an UNCHANGED ``score_pop``.

    ``score_pop`` evaluates candidates independently (per-candidate fold
    scores -> per-candidate LCB with a fixed internal bootstrap seed), so
    partitioning the candidate list is byte-identical to one call —
    asserted by ``tests/test_gpu_memory.py`` (§6.4). The frozen
    ``src/v17_gpu`` internals are never modified; this wrapper IS the F9
    wiring.
    """
    if chunk < 1:
        raise ValueError(f"chunk must be >= 1, got {chunk}")
    if not params_list:
        return np.empty(0, dtype=np.float64)
    parts = [
        np.asarray(score_pop(params_list[i:i + chunk]), dtype=np.float64)
        for i in range(0, len(params_list), chunk)
    ]
    return np.concatenate(parts)


__all__ = [
    "BUDGET_BYTES", "N_SCALES", "PIR_BYTES", "BYTES_PER_CANDIDATE_BAR",
    "gpu_memory_estimate", "estimate_from_folds", "chunk_size",
    "assert_probe_agreement", "chunked_score_pop",
]
