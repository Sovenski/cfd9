"""Scale-invariance falsification gate (spec §5.3).

Optimize/choose HIGH params on one timeframe (e.g. SPX-1D), then measure the
precision drop when the SAME params are applied to another timeframe (SPX-1W).
A drop > 0.15 absolute rejects scale-invariance → do not mix timeframes.
"""
from __future__ import annotations

import pandas as pd

from .detector import SpeculatorDetector
from .indicators import Params
from .scoring import REFERENCE_N, add_pivot_labels, precision_at_n_stats

DROP_THRESHOLD: float = 0.15


def _precision(df: pd.DataFrame, params: Params, side: str) -> float:
    work = df.reset_index(drop=True).copy()
    add_pivot_labels(work)
    det = SpeculatorDetector(work, params).run()
    sig = det["signal_high" if side == "high" else "signal_low"]
    stats = precision_at_n_stats(sig, work[f"pivot_N{REFERENCE_N}"], side, REFERENCE_N)
    return float(stats["precision"])


def precision_transfer(
    params: Params, side: str, base_df: pd.DataFrame, other_df: pd.DataFrame,
) -> dict[str, float]:
    """Return precision on base timeframe, on other timeframe, and the drop."""
    p_base = _precision(base_df, params, side)
    p_other = _precision(other_df, params, side)
    return {
        "precision_base": p_base,
        "precision_other": p_other,
        "drop": p_base - p_other,
    }


def scale_invariance_holds(transfer: dict[str, float]) -> bool:
    """True if the precision drop is within the acceptance threshold."""
    return transfer["drop"] <= DROP_THRESHOLD
