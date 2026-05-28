"""Per-stream volume-quality profiler.

TV intraday volume for indices is often backfilled with placeholder values
(constant bar-seconds like 3600, or zeros) early in history and becomes real
later. We FLAG quality (full/partial/none) and the date real volume begins,
so runs can SEPARATE volume-bearing data from price-only (never deleting it).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Bar-seconds placeholders TV emits for volume-less intraday history.
_PLACEHOLDER_VALUES = {0, 2400, 3596, 3597, 3598, 3599, 3600, 3601}
REAL_FLOOR = 100_000           # below this is treated as non-real for an index/stock
_REAL_FLOOR = REAL_FLOOR       # backward-compat alias
_REAL_SHARE_THRESHOLD = 0.05   # a window is "real" when <5% of bars are placeholder


@dataclass(frozen=True)
class VolumeQuality:
    quality: str                       # "full" | "partial" | "none"
    real_from: pd.Timestamp | None     # first date where volume is reliably real
    placeholder_share: float           # overall fraction of placeholder bars


def _placeholder_mask(volume: pd.Series) -> np.ndarray:
    v = volume.fillna(0)
    return (v.isin(_PLACEHOLDER_VALUES) | (v < _REAL_FLOOR)).to_numpy()


def profile_volume(df: pd.DataFrame, window: int = 60) -> VolumeQuality:
    """Classify a stream's volume quality and find where real volume begins.

    ``window`` is the rolling bar count over which the placeholder share is
    evaluated to locate the real-volume onset.
    """
    if "volume" not in df.columns or len(df) == 0:
        return VolumeQuality("none", None, 1.0)

    ph = _placeholder_mask(df["volume"])
    overall_share = float(ph.mean())

    if overall_share < _REAL_SHARE_THRESHOLD:
        return VolumeQuality("full", _index_ts(df, 0), overall_share)
    if overall_share > (1.0 - _REAL_SHARE_THRESHOLD):
        return VolumeQuality("none", None, overall_share)

    # Partial: find first index where the forward rolling placeholder share
    # stays below threshold.
    ph_series = pd.Series(ph.astype(float), index=df.index)
    fwd_share = ph_series[::-1].rolling(window, min_periods=1).mean()[::-1]
    real_idx = np.flatnonzero((fwd_share < _REAL_SHARE_THRESHOLD).to_numpy())
    real_from = _index_ts(df, int(real_idx[0])) if real_idx.size else None
    return VolumeQuality("partial", real_from, overall_share)


def _index_ts(df: pd.DataFrame, pos: int) -> pd.Timestamp:
    if isinstance(df.index, pd.DatetimeIndex):
        return df.index[pos]
    return pd.to_datetime(df["time"].iloc[pos], unit="s")
