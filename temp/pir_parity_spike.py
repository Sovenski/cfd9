"""Phase 0.5 — PIR byte-identity spike, CPU (gpu-refactor-build-spec §2).

Measures whether a torch float32 ``pir_matrix`` can be byte-identical to the
CPU oracle ``indicators.precompute_matrices`` on a real SPX IS slice
(~5.7k bars), across all 499 scales (2..500), on ``torch.device("cpu")``.

Two methods (each at two internal precisions, f32 / f64 — output is ALWAYS
cast to float32 to honour parity invariant P1):

  (a) cumsum-difference rolling mean  — the GPU-fast prefix-sum kernel.
  (b) pandas-faithful per-window mean — unfold + per-window summation,
      mirroring the oracle's "fresh sum per window" semantics as closely
      as torch allows (pandas itself uses a Kahan-compensated running sum,
      so even f64 may differ in the last ulp).

For each candidate matrix we record:
  * ``np.array_equal(candidate, oracle, equal_nan=True)`` (byte identity),
  * ``max_abs_drift`` over bars where both are finite + NaN-mask mismatches,
  * the SIGNAL-FLIP RATE through ``calc_agreement_fast`` at the gold
    representative ``pct_extreme`` (both HIGH and LOW side configs):
      - value flips: per-bar agreement fraction differs at all,
      - vote flips:  per-bar ``agr >= min_agreement`` gate decision differs.

Verdict (printed as JSON, consumed by the workflow):
  recommended_branch = "trust-kernel" iff some method is byte-identical
  (or its vote-flip rate is exactly 0), else "noisy-ranker" with a top-K
  suggestion sized from the measured flip rate.

This is a MEASUREMENT, not a pass/fail gate. Run from the repo root:

    python temp/pir_parity_spike.py
"""
from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path
from typing import Callable

import numpy as np
import torch

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from src.indicators import (  # noqa: E402
    Params,
    calc_agreement_fast,
    precompute_matrices,
)
from src.pooled_validation import load_stream_frame  # noqa: E402

logger = logging.getLogger(__name__)

SPX_CSV: Path = _REPO / "data" / "raw" / "SPX_1D_18710201_20260318.csv"
TAIL_BARS: int = 6000      # same deterministic tail as temp/capture_baseline.py
IS_BARS: int = 5700        # IS slice = tail minus a 300-bar OOS/embargo stub
SCALE_MIN: int = 2
SCALE_MAX: int = 500
DEVICE: torch.device = torch.device("cpu")


# ---------------------------------------------------------------------------
# Torch building blocks
# ---------------------------------------------------------------------------


def _rolling_mean_cumsum(x: torch.Tensor, window: int) -> torch.Tensor:
    """Prefix-sum rolling mean: (csum[t] - csum[t-w]) / w. NaN warm-up."""
    n = x.shape[0]
    out = torch.full((n,), float("nan"), dtype=x.dtype, device=DEVICE)
    csum = torch.cat(
        [torch.zeros(1, dtype=x.dtype, device=DEVICE), torch.cumsum(x, dim=0)]
    )
    out[window - 1:] = (csum[window:] - csum[:-window]) / window
    return out


def _rolling_mean_unfold(x: torch.Tensor, window: int) -> torch.Tensor:
    """Per-window (fresh) sum rolling mean via unfold. NaN warm-up."""
    n = x.shape[0]
    out = torch.full((n,), float("nan"), dtype=x.dtype, device=DEVICE)
    out[window - 1:] = x.unfold(0, window, 1).sum(dim=1) / window
    return out


def _pir_from_ratio(ratio: torch.Tensor, lb: int) -> torch.Tensor:
    """Torch mirror of ``indicators.pir_of`` (rolling min/max + span clamp).

    NaN propagation through amin/amax reproduces pandas' min_periods=window
    warm-up exactly: any window touching the SMA warm-up NaNs yields NaN.
    """
    n = ratio.shape[0]
    out = torch.full((n,), float("nan"), dtype=ratio.dtype, device=DEVICE)
    windows = ratio.unfold(0, lb, 1)
    lo = windows.amin(dim=1)
    hi = windows.amax(dim=1)
    val = ratio[lb - 1:]
    span = (hi - lo).clamp(min=1e-10)
    res = (val - lo) / span
    # pandas: result.where(hi != lo, 0.5) — NaN != NaN is True → NaN kept.
    out[lb - 1:] = torch.where(hi != lo, res, torch.full_like(res, 0.5))
    return out


def build_pir_torch(
    close64: np.ndarray,
    mean_fn: Callable[[torch.Tensor, int], torch.Tensor],
    compute_dtype: torch.dtype,
) -> np.ndarray:
    """Build the full float32 pir_matrix in torch for one method/precision."""
    n_bars = len(close64)
    scales = list(range(SCALE_MIN, SCALE_MAX + 1))
    out = np.full((len(scales), n_bars), np.nan, dtype=np.float32)
    x = torch.tensor(close64, dtype=compute_dtype, device=DEVICE)
    for i, s in enumerate(scales):
        sma = mean_fn(x, s)
        ratio = x / sma
        pir = _pir_from_ratio(ratio, max(s, 20))
        out[i] = pir.to(torch.float32).numpy()
    return out


# ---------------------------------------------------------------------------
# Comparison + flip-rate measurement
# ---------------------------------------------------------------------------


def compare_matrices(candidate: np.ndarray, oracle: np.ndarray) -> dict:
    """Byte-identity + drift stats between candidate and oracle (both f32)."""
    mask_c = np.isnan(candidate)
    mask_o = np.isnan(oracle)
    both_finite = ~mask_c & ~mask_o
    diffs = np.abs(
        candidate[both_finite].astype(np.float64)
        - oracle[both_finite].astype(np.float64)
    )
    return {
        "exact_match": bool(np.array_equal(candidate, oracle, equal_nan=True)),
        "max_abs_drift": float(diffs.max()) if diffs.size else 0.0,
        "n_value_diffs": int((diffs > 0).sum()),
        "n_finite_cells": int(both_finite.sum()),
        "n_nan_mask_mismatch": int((mask_c != mask_o).sum()),
    }


def flip_stats(
    candidate: np.ndarray,
    oracle: np.ndarray,
    scales_list: list[int],
    params: Params,
) -> dict:
    """Signal-flip rates through calc_agreement_fast at the gold thresholds."""
    sides = {
        "high": (
            params.scale_start_high, params.scale_end_high,
            params.scale_step_high, params.pct_extreme_high,
            params.min_agreement_high,
        ),
        "low": (
            params.scale_start_low, params.scale_end_low,
            params.scale_step_low, params.pct_extreme_low,
            params.min_agreement_low,
        ),
    }
    out: dict = {}
    for side, (s0, s1, step, pct, min_agr) in sides.items():
        ah_o, al_o, n_scales = calc_agreement_fast(
            oracle, scales_list, s0, s1, step, pct)
        ah_c, al_c, _ = calc_agreement_fast(
            candidate, scales_list, s0, s1, step, pct)
        n_bars = len(ah_o)
        value_flips = int(((ah_c != ah_o) | (al_c != al_o)).sum())
        # Detector gate operator is `agr >= min_agreement` (P3, verbatim).
        vote_flips = int((
            ((ah_c >= min_agr) != (ah_o >= min_agr))
            | ((al_c >= min_agr) != (al_o >= min_agr))
        ).sum())
        out[side] = {
            "n_scales": n_scales,
            "pct_extreme": pct,
            "min_agreement": min_agr,
            "value_flip_rate": value_flips / n_bars,
            "vote_flip_rate": vote_flips / n_bars,
        }
    out["max_value_flip_rate"] = max(
        out["high"]["value_flip_rate"], out["low"]["value_flip_rate"])
    out["max_vote_flip_rate"] = max(
        out["high"]["vote_flip_rate"], out["low"]["vote_flip_rate"])
    return out


def suggest_topk(flip_rate: float) -> int:
    """Size the CPU re-score finalist set from the measured flip rate."""
    tiers = [
        (0.0, 16), (1e-6, 32), (1e-5, 48), (1e-4, 64),
        (1e-3, 128), (1e-2, 256), (1e-1, 512),
    ]
    for threshold, k in tiers:
        if flip_rate <= threshold:
            return k
    return 1024


# ---------------------------------------------------------------------------
# Main spike
# ---------------------------------------------------------------------------


def main() -> dict:
    """Run the spike and return (and print) the verdict dict."""
    df = load_stream_frame(str(SPX_CSV)).iloc[-TAIL_BARS:].copy()
    df_is = df.iloc[:IS_BARS]
    close = df_is["close"]
    close64 = close.values.astype(np.float64)
    logger.info("SPX IS slice: %d bars (%s .. %s)", len(df_is),
                df_is.index[0], df_is.index[-1])

    t0 = time.time()
    _, pir_oracle, scales_list = precompute_matrices(close, SCALE_MIN, SCALE_MAX)
    logger.info("oracle precompute_matrices: %.1fs, shape=%s, dtype=%s",
                time.time() - t0, pir_oracle.shape, pir_oracle.dtype)

    params = Params()

    # PRODUCTION variant (decides the branch): the real torch builder, which
    # mirrors the PINE-FAITHFUL oracle construction (2026-06-10 parity fix —
    # cumsum SMA valid from bar s, partial-window pir, f64). The legacy
    # variants below predate that fix and are kept as INFORMATIONAL baselines
    # only; their warm-up semantics intentionally differ from the new oracle.
    from src.v17_gpu.eval_torch import build_pir_matrix_torch
    t0 = time.time()
    prod = build_pir_matrix_torch(close64, SCALE_MIN, SCALE_MAX, device=str(DEVICE))
    prod_stats = compare_matrices(prod, pir_oracle)
    prod_stats["flips"] = flip_stats(prod, pir_oracle, scales_list, params)
    prod_stats["build_seconds"] = round(time.time() - t0, 1)
    logger.info(
        "%-24s exact=%s drift=%.3e value_diffs=%d/%d nan_mismatch=%d "
        "vote_flip=%.2e value_flip=%.2e",
        "production-pine/f64", prod_stats["exact_match"],
        prod_stats["max_abs_drift"], prod_stats["n_value_diffs"],
        prod_stats["n_finite_cells"], prod_stats["n_nan_mask_mismatch"],
        prod_stats["flips"]["max_vote_flip_rate"],
        prod_stats["flips"]["max_value_flip_rate"],
    )

    variants: dict[str, tuple[Callable, torch.dtype]] = {
        "cumsum-difference/f32": (_rolling_mean_cumsum, torch.float32),
        "cumsum-difference/f64": (_rolling_mean_cumsum, torch.float64),
        "pandas-faithful/f32": (_rolling_mean_unfold, torch.float32),
        "pandas-faithful/f64": (_rolling_mean_unfold, torch.float64),
    }

    results: dict[str, dict] = {}
    for name, (mean_fn, dtype) in variants.items():
        t0 = time.time()
        candidate = build_pir_torch(close64, mean_fn, dtype)
        stats = compare_matrices(candidate, pir_oracle)
        stats["flips"] = flip_stats(candidate, pir_oracle, scales_list, params)
        stats["build_seconds"] = round(time.time() - t0, 1)
        results[name] = stats
        logger.info(
            "%-24s exact=%s drift=%.3e value_diffs=%d/%d nan_mismatch=%d "
            "vote_flip=%.2e value_flip=%.2e",
            name, stats["exact_match"], stats["max_abs_drift"],
            stats["n_value_diffs"], stats["n_finite_cells"],
            stats["n_nan_mask_mismatch"],
            stats["flips"]["max_vote_flip_rate"],
            stats["flips"]["max_value_flip_rate"],
        )

    # Method-level verdict: best variant per spec-named method.
    def best_variant(method: str) -> tuple[str, dict]:
        names = [n for n in results if n.startswith(method)]
        key = lambda n: (  # noqa: E731
            not results[n]["exact_match"],
            results[n]["flips"]["max_vote_flip_rate"],
            results[n]["max_abs_drift"],
        )
        best = min(names, key=key)
        return best, results[best]

    best_cumsum_name, best_cumsum = best_variant("cumsum-difference")
    best_pandas_name, best_pandas = best_variant("pandas-faithful")

    results["production-pine/f64"] = prod_stats
    candidates = [
        ("production-pine-faithful", "production-pine/f64", prod_stats),
        ("cumsum-difference", best_cumsum_name, best_cumsum),
        ("pandas-faithful", best_pandas_name, best_pandas),
    ]
    exact = [c for c in candidates if c[2]["exact_match"]]
    zero_flip = [
        c for c in candidates
        if c[2]["flips"]["max_vote_flip_rate"] == 0.0
    ]
    if exact:
        winner = exact[0]
        recommended = "trust-kernel"
    elif zero_flip:
        winner = zero_flip[0]
        recommended = "trust-kernel"
    else:
        winner = min(
            candidates,
            key=lambda c: (c[2]["flips"]["max_vote_flip_rate"],
                           c[2]["max_abs_drift"]),
        )
        recommended = "noisy-ranker"

    method, variant_name, stats = winner
    flip_rate = stats["flips"]["max_vote_flip_rate"]
    verdict = {
        "exact_match": bool(exact),
        "winning_method": method if (exact or zero_flip) else "none",
        "best_method_overall": method,
        "best_variant": variant_name,
        "max_abs_drift": stats["max_abs_drift"],
        "signal_flip_rate": flip_rate,
        "value_flip_rate": stats["flips"]["max_value_flip_rate"],
        "recommended_branch": recommended,
        "suggested_topK": suggest_topk(
            max(flip_rate, stats["flips"]["max_value_flip_rate"])),
        "slice": {
            "csv": str(SPX_CSV.relative_to(_REPO)),
            "n_bars": len(df_is),
            "scales": [SCALE_MIN, SCALE_MAX],
        },
        "per_variant": results,
    }
    print(json.dumps(verdict, indent=2, default=str))
    return verdict


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    main()
