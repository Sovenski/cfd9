"""Speculatores Pivot Optimizer — validation framework.

Provides temporal train/test splits, walk-forward fold generation, data
loading utilities, and the Optuna objective builder used by the optimiser.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import optuna

from .indicators import Params
from .detector import (
    DetectorArtifacts,
    SpeculatorDetector,
    build_detector_artifacts,
)
from .scoring import add_pivot_labels, fold_score_high, fold_score_low

logger = logging.getLogger(__name__)

MIN_SIGNALS_PER_FOLD: int = 6
PreparedFold = tuple[pd.DataFrame, pd.DataFrame, DetectorArtifacts, DetectorArtifacts]

# ---------------------------------------------------------------------------
# Walk-forward fold definitions
# ---------------------------------------------------------------------------

FOLD_DEFINITIONS: list[tuple[str, str, str, str]] = [
    # (is_start, is_end, oos_start, oos_end)
    ("1983-01-01", "1993-01-01", "1993-01-01", "1996-01-01"),
    ("1986-01-01", "1996-01-01", "1996-01-01", "1999-01-01"),
    ("1989-01-01", "1999-01-01", "1999-01-01", "2002-01-01"),
    ("1992-01-01", "2002-01-01", "2002-01-01", "2005-01-01"),
    ("1995-01-01", "2005-01-01", "2005-01-01", "2008-01-01"),
]

EMBARGO_BARS: int = 20


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _get_time_series(df: pd.DataFrame) -> pd.Series:
    """Get datetime series from df (handles DatetimeIndex or 'time' column)."""
    if isinstance(df.index, pd.DatetimeIndex):
        return pd.Series(df.index, index=df.index)
    return pd.to_datetime(df["time"], unit="s")


# ---------------------------------------------------------------------------
# Layer 1 — Temporal holdout split
# ---------------------------------------------------------------------------


def temporal_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Layer 1 holdout split.

    IS: data before 2010-01-01
    OOS: data from 2010-01-01 onwards

    Args:
        df: OHLCV DataFrame with DatetimeIndex or 'time' column.

    Returns:
        (df_is, df_oos) tuple.
    """
    cutoff = pd.Timestamp("2010-01-01")
    if isinstance(df.index, pd.DatetimeIndex):
        is_mask = df.index < cutoff
    else:
        time_col = pd.to_datetime(df["time"], unit="s")
        is_mask = time_col < cutoff
    df_is = df[is_mask].copy()
    df_oos = df[~is_mask].copy()
    logger.info(
        "temporal_split: IS=%d bars (ends %s), OOS=%d bars (starts %s)",
        len(df_is),
        df_is.index[-1] if len(df_is) else "N/A",
        len(df_oos),
        df_oos.index[0] if len(df_oos) else "N/A",
    )
    return df_is, df_oos


# ---------------------------------------------------------------------------
# Layer 2 — Walk-forward folds
# ---------------------------------------------------------------------------


def walk_forward_folds(
    df: pd.DataFrame,
) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    """Generate walk-forward folds with 20-bar embargo.

    Returns 5 (df_is, df_oos) pairs per FOLD_DEFINITIONS.  The last
    EMBARGO_BARS rows are dropped from each IS slice before returning.

    Args:
        df: OHLCV DataFrame with DatetimeIndex or 'time' column.

    Returns:
        List of (df_is, df_oos) tuples, one per valid fold.
    """
    folds: list[tuple[pd.DataFrame, pd.DataFrame]] = []
    time_series = _get_time_series(df)

    for is_start, is_end, oos_start, oos_end in FOLD_DEFINITIONS:
        is_start_ts = pd.Timestamp(is_start)
        is_end_ts = pd.Timestamp(is_end)
        oos_start_ts = pd.Timestamp(oos_start)
        oos_end_ts = pd.Timestamp(oos_end)

        is_mask = (time_series >= is_start_ts) & (time_series < is_end_ts)
        oos_mask = (time_series >= oos_start_ts) & (time_series < oos_end_ts)

        df_is_raw = df[is_mask].copy()
        df_oos = df[oos_mask].copy()

        # Apply embargo: drop last EMBARGO_BARS rows from IS
        if len(df_is_raw) <= EMBARGO_BARS:
            logger.warning("Fold IS slice too small for embargo (%d bars), skipping", len(df_is_raw))
            continue
        df_is = df_is_raw.iloc[:-EMBARGO_BARS].copy()

        if len(df_is) > 0 and len(df_oos) > 0:
            folds.append((df_is, df_oos))
            logger.debug(
                "Fold %s→%s | IS %d bars, OOS %d bars",
                oos_start, oos_end, len(df_is), len(df_oos),
            )

    logger.info("walk_forward_folds: %d valid folds generated", len(folds))
    return folds


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_data(path: str | Path) -> pd.DataFrame:
    """Load OHLCV CSV with Unix-timestamp 'time' column.

    Expected columns: time, open, high, low, close, volume.
    'time' is Unix timestamp (seconds) converted to DatetimeIndex.

    Args:
        path: Path to CSV file.

    Returns:
        DataFrame with DatetimeIndex, sorted ascending, lowercase column names.
    """
    df = pd.read_csv(path)
    df.columns = [c.lower() for c in df.columns]
    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"], unit="s")
        df = df.set_index("time")
    df = df.sort_index()
    logger.info("load_data: %d bars loaded from %s", len(df), Path(path).name)
    return df


def load_cross_asset(
    path: str | Path,
    resample_to_1d: bool = False,
) -> pd.DataFrame:
    """Load cross-asset data with optional 1M→1D resampling.

    Resampling aggregation: Open=first, High=max, Low=min, Close=last,
    Volume=sum.  Empty days (no trading) and days with zero volume are
    dropped.

    Args:
        path: Path to CSV file.
        resample_to_1d: If True, resample minute bars to daily bars.

    Returns:
        DataFrame with DatetimeIndex and OHLCV columns.
    """
    df = load_data(path)
    if not resample_to_1d:
        return df

    df_daily = df.resample("1d").agg(
        {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }
    ).dropna(subset=["close"])

    # Drop days with no volume (incomplete sessions)
    df_daily = df_daily[df_daily["volume"] > 0]

    logger.info(
        "load_cross_asset: resampled %d minute bars → %d daily bars from %s",
        len(df), len(df_daily), Path(path).name,
    )
    return df_daily


# ---------------------------------------------------------------------------
# Optuna objective builder
# ---------------------------------------------------------------------------


def build_optuna_objective(
    df_full: pd.DataFrame,
    params_from_trial: Callable[[optuna.Trial, str], Params],
    side: str,
) -> Callable[[optuna.Trial], float]:
    """Build an Optuna objective function for one study side.

    The returned objective:
    1. Samples Params from trial via params_from_trial(trial, side).
    2. Runs SpeculatorDetector on each walk-forward fold's IS and OOS slices.
    3. Computes fold_score_high or fold_score_low per fold.
    4. Reports intermediate cumulative mean score after each fold for pruning.
    5. Returns the mean score across all folds.

    Args:
        df_full: Full OHLCV DataFrame (walk-forward folds derived from this).
        params_from_trial: Callable that builds a Params instance from an
            optuna.Trial and a side string ("high" or "low").
        side: "high" or "low" — which detector side to optimise.

    Returns:
        Optuna objective callable.
    """
    if side not in ("high", "low"):
        raise ValueError(f"side must be 'high' or 'low', got {side!r}")

    folds = walk_forward_folds(df_full)
    if not folds:
        raise ValueError("No walk-forward folds could be constructed from df_full.")
    if len(folds) != len(FOLD_DEFINITIONS):
        raise ValueError(
            f"Expected {len(FOLD_DEFINITIONS)} folds, got {len(folds)}. "
            "Ensure df_full covers 1983-2008."
        )

    prepared_folds = prepare_walk_forward_folds(df_full, folds)

    def objective(trial: optuna.Trial) -> float:
        params = params_from_trial(trial, side)

        fold_scores: list[float] = []
        for fold_idx, score in enumerate(evaluate_params_on_prepared_folds(params, side, prepared_folds)):
            fold_scores.append(score)
            trial.report(float(np.mean(fold_scores)), fold_idx)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        return float(np.mean(fold_scores))

    return objective


def prepare_walk_forward_folds(
    df_full: pd.DataFrame,
    folds: list[tuple[pd.DataFrame, pd.DataFrame]] | None = None,
) -> list[PreparedFold]:
    folds = folds or walk_forward_folds(df_full)
    prepared_folds: list[PreparedFold] = []
    for fold_idx, (df_is, df_oos) in enumerate(folds):
        df_is_r = df_is.reset_index(drop=True)
        df_oos_r = df_oos.reset_index(drop=True)
        add_pivot_labels(df_is_r)
        add_pivot_labels(df_oos_r)
        logger.info(
            "Precomputing fold %d artifacts: IS=%d bars, OOS=%d bars",
            fold_idx + 1,
            len(df_is_r),
            len(df_oos_r),
        )
        artifacts_is = build_detector_artifacts(df_is_r)
        artifacts_oos = build_detector_artifacts(df_oos_r)
        prepared_folds.append((df_is_r, df_oos_r, artifacts_is, artifacts_oos))
    return prepared_folds


def evaluate_params_on_prepared_folds(
    params: Params,
    side: str,
    prepared_folds: list[PreparedFold],
) -> list[float]:
    fold_scores: list[float] = []
    for df_is_r, df_oos_r, artifacts_is, artifacts_oos in prepared_folds:
        det_is = SpeculatorDetector(df_is_r, params, artifacts_is).run()
        det_oos = SpeculatorDetector(df_oos_r, params, artifacts_oos).run()

        sig_high_is = det_is["signal_high"]
        sig_low_is = det_is["signal_low"]
        sig_high_oos = det_oos["signal_high"]
        sig_low_oos = det_oos["signal_low"]

        if side == "high":
            score = fold_score_high(df_is_r, df_oos_r, sig_high_is, sig_high_oos)
            n_is_signals = int(sig_high_is.sum())
            n_oos_signals = int(sig_high_oos.sum())
        else:
            score = fold_score_low(df_is_r, df_oos_r, sig_low_is, sig_low_oos)
            n_is_signals = int(sig_low_is.sum())
            n_oos_signals = int(sig_low_oos.sum())

        support_factor = min(1.0, n_is_signals / MIN_SIGNALS_PER_FOLD) * min(
            1.0, n_oos_signals / MIN_SIGNALS_PER_FOLD
        )
        fold_scores.append(score * support_factor)
    return fold_scores


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )

    data_dir = Path(__file__).resolve().parent.parent / "data" / "raw"

    # 1. Load SPX daily data
    spx_path = data_dir / "SPX_1D_18710201_20260318.csv"
    if not spx_path.exists():
        print(f"Data file not found: {spx_path}", file=sys.stderr)
        sys.exit(1)

    df = load_data(spx_path)
    logger.info("Loaded SPX: %d bars, range %s → %s", len(df), df.index[0], df.index[-1])

    # 2. temporal_split
    df_is, df_oos = temporal_split(df)
    assert len(df_is) > 0, "IS split is empty"
    assert len(df_oos) > 0, "OOS split is empty"
    assert df_is.index[-1] < pd.Timestamp("2010-01-01"), (
        f"IS ends at {df_is.index[-1]}, expected < 2010-01-01"
    )
    assert df_oos.index[0] >= pd.Timestamp("2010-01-01"), (
        f"OOS starts at {df_oos.index[0]}, expected >= 2010-01-01"
    )
    logger.info(
        "temporal_split OK: IS %d bars (last %s), OOS %d bars (first %s)",
        len(df_is), df_is.index[-1].date(),
        len(df_oos), df_oos.index[0].date(),
    )

    # 3. walk_forward_folds
    folds = walk_forward_folds(df)
    assert len(folds) == 5, f"Expected 5 folds, got {len(folds)}"
    for i, (fold_is, fold_oos) in enumerate(folds):
        assert len(fold_is) > 0, f"Fold {i} IS is empty"
        assert len(fold_oos) > 0, f"Fold {i} OOS is empty"
        logger.info(
            "  Fold %d: IS=%d bars (%s→%s), OOS=%d bars (%s→%s)",
            i + 1,
            len(fold_is), fold_is.index[0].date(), fold_is.index[-1].date(),
            len(fold_oos), fold_oos.index[0].date(), fold_oos.index[-1].date(),
        )

    # 4. load_cross_asset with DAX 1D (no resampling)
    dax_path = data_dir / "DAX_1D_19700102_20260324.csv"
    if dax_path.exists():
        df_dax = load_cross_asset(dax_path, resample_to_1d=False)
        assert len(df_dax) > 0, "DAX daily load returned empty DataFrame"
        logger.info(
            "load_cross_asset (no resample) OK: %d bars from %s",
            len(df_dax), dax_path.name,
        )
    else:
        logger.warning("DAX 1D file not found, skipping cross-asset test: %s", dax_path)

    logger.info("validation.py self-test PASSED")
    print("\nvalidation.py self-test PASSED")
