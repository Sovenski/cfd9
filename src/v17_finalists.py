"""§6 finalist plumbing for ``run_v17_gpu`` (build-spec PHASE 4, torch-free).

Split out of ``v17_runner`` for the <=~400-line file rule. Three pieces:

- ``topk_for_flip_rate`` — size the CPU-re-score finalist set from the §2
  PIR-spike's measured GPU signal-flip rate (trust-kernel => 0.0 => 16).
- ``filter_finalists`` — the HARD ``|gpu - cpu| > tol`` finalist filter that
  generalizes the old ``run_v17`` warning into a gate.
- ``tv_export_audit`` — the per-asset TradingView-export Pine-parity audit
  hook recorded next to every ``run_v17_gpu`` leaderboard.
"""

from __future__ import annotations

import logging
from typing import Optional, Protocol, Sequence

from .indicators import Params
from .universe import Stream

logger = logging.getLogger(__name__)

#: Finalist-set sizing tiers (spec §6): mirror of the §2 spike's
#: ``suggest_topk`` — top-K CPU re-scores sized to the measured GPU
#: signal-flip rate (trust-kernel => flip rate 0.0 => 16).
_FLIP_TOPK_TIERS: tuple[tuple[float, int], ...] = (
    (0.0, 16), (1e-6, 32), (1e-5, 48), (1e-4, 64),
    (1e-3, 128), (1e-2, 256), (1e-1, 512),
)


class _Finalist(Protocol):
    """``v17_search.Candidate`` shape: a scored Params configuration."""

    params: Params
    score: float


class _Scorer(Protocol):
    def score(self, params: Params) -> float: ...


def topk_for_flip_rate(flip_rate: float) -> int:
    """Size the CPU-re-score finalist set from the measured GPU flip rate."""
    for threshold, k in _FLIP_TOPK_TIERS:
        if flip_rate <= threshold:
            return k
    return 1024


def filter_finalists(finalists: Sequence[_Finalist], real_scorer: _Scorer,
                     tol: float = 1e-9) -> tuple[list[dict], list[dict]]:
    """HARD finalist filter (spec §6) — the old warning, made a gate.

    ``run_v17`` only WARNED when ``|fast - real| > 1e-9``; here any finalist
    whose GPU LCB disagrees with the EXACT CPU ``PooledScorer`` LCB beyond
    ``tol`` is DROPPED from the leaderboard (a parity violation must never be
    reported, let alone exported). Survivors are re-ranked by the CPU LCB
    (descending, stable), which is the only number ever reported.

    Args:
        finalists: ``v17_search.Candidate``-like objects (``params``/``score``).
        real_scorer: the EXACT CPU scorer (``PooledScorer`` contract).
        tol: max allowed ``|gpu_lcb - cpu_lcb|``.

    Returns:
        ``(survivors, dropped)`` — dict entries with keys ``candidate``,
        ``gpu_lcb``, ``cpu_lcb``, ``abs_diff``; survivors CPU-LCB-sorted.
    """
    survivors: list[dict] = []
    dropped: list[dict] = []
    for cand in finalists:
        cpu_lcb = float(real_scorer.score(cand.params))
        entry = {"candidate": cand, "gpu_lcb": float(cand.score),
                 "cpu_lcb": cpu_lcb,
                 "abs_diff": abs(cpu_lcb - float(cand.score))}
        if entry["abs_diff"] <= tol:
            survivors.append(entry)
        else:
            logger.warning("finalist DROPPED (parity): gpu=%.12f cpu=%.12f "
                           "diff=%.3e > %g", entry["gpu_lcb"], cpu_lcb,
                           entry["abs_diff"], tol)
            dropped.append(entry)
    survivors.sort(key=lambda e: -e["cpu_lcb"])  # stable: ties keep GPU order
    return survivors, dropped


def tv_export_audit(streams: list[Stream],
                    params: Optional[Params] = None) -> dict:
    """Per-asset TradingView-export audit hook (spec §6).

    For every stream in the pool, look up its enriched TV export and run the
    Pine-vs-Python parity comparison. Parity is defined at the gold
    ``Params()`` (the preset the enriched exports were generated with), so the
    audit defaults to that — NOT the search winner. Audit failures are
    recorded, never raised (the hook must not kill a finished run).
    """
    from .parity import compare_python_to_tv, find_matching_enriched_export
    out: dict = {}
    for s in streams:
        export = find_matching_enriched_export(s.path)
        if export is None:
            out[s.stream_id] = {"export": None, "metrics": None}
            continue
        try:
            summary, _ = compare_python_to_tv(s.path, export,
                                              params=params or Params())
            out[s.stream_id] = {"export": str(export),
                                "metrics": summary.to_dict(orient="records")}
        except (ValueError, KeyError, OSError) as exc:
            logger.warning("tv_export_audit %s failed: %s", s.stream_id, exc)
            out[s.stream_id] = {"export": str(export), "error": str(exc)}
    return out


__all__ = ["topk_for_flip_rate", "filter_finalists", "tv_export_audit"]
