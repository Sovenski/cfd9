"""Calendar-based pooled folds + pooled Optuna objective (multi-asset).

Keeps ``src/validation.py`` (single-asset, row-index folds) untouched. This
module adds the time-aligned, cluster-weighted pooled path used by Run 5.
"""
from __future__ import annotations

import dataclasses
import logging
from collections.abc import Callable

import numpy as np
import optuna
import pandas as pd

from .detector import SpeculatorDetector, build_detector_artifacts
from .indicators import Params
from .pooled_scoring import StreamStat, pooled_fold_score
from .scoring import (
    REFERENCE_N, STRUCTURAL_NEST, add_pivot_labels, precision_at_n_stats,
)
from .universe import Stream
from .validation import fold_scores_bootstrap_ci, load_data
from .volume_quality import VolumeQuality, profile_volume

logger = logging.getLogger(__name__)

EMBARGO_NEST_BARS: int = max(STRUCTURAL_NEST)    # 200 — label look-ahead guard
MIN_STREAM_BARS: int = 2 * max(STRUCTURAL_NEST) + 1   # 401 — nest must fit
HOLDOUT_FRACTION: float = 0.20
IS_FRACTION: float = 0.10
OOS_FRACTION: float = 0.03
STEP_FRACTION: float = 0.05


def load_stream_frame(path: str) -> pd.DataFrame:
    """Load a stream CSV via the canonical loader (lowercases, indexes time)."""
    return load_data(path)


def apply_volume_policy(
    df: pd.DataFrame, policy: str,
) -> tuple[pd.DataFrame, bool]:
    """Apply the run-level volume policy to a single stream.

    Returns ``(possibly_trimmed_df, keep)``:
    - ``price_only`` / ``mixed``: keep all bars (volume votes handled by params).
    - ``volume_required``: trim to ``[volume_real_from, end]``; drop the stream
      entirely (``keep=False``) if its quality is ``none``.
    """
    if policy in ("price_only", "mixed"):
        return df, True
    if policy == "volume_required":
        vq: VolumeQuality = profile_volume(df)
        if vq.quality == "none" or vq.real_from is None:
            return df, False
        # real_from marks the first window that is reliably real; advance to
        # the first bar where volume is actually above the real floor so the
        # slice never starts on a straggler placeholder bar.
        candidate = df.loc[df.index >= vq.real_from]
        real_mask = candidate["volume"] >= 100_000
        if not real_mask.any():
            return df, False
        first_real = candidate.index[real_mask.to_numpy().argmax()]
        return df.loc[df.index >= first_real].copy(), True
    raise ValueError(f"unknown volume policy {policy!r}")
