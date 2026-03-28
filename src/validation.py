"""Speculatores Pivot Optimizer — validation framework.

Provides temporal train/test splits, walk-forward fold generation, data
loading utilities, and the Optuna objective builder used by the optimiser.
The validation layer supports both the original long-history date-based
scheme and a generic rolling-bar scheme for arbitrary OHLCV bar datasets.
"""

from __future__ import annotations

import dataclasses
import logging
from collections.abc import Callable
from pathlib import Path

import numpy as np
import optuna
import pandas as pd

from .detector import DetectorArtifacts, SpeculatorDetector, build_detector_artifacts
from .indicators import Params
from .scoring import add_pivot_labels, fold_score_high, fold_score_low

logger = logging.getLogger(__name__)

MIN_SIGNALS_PER_FOLD: int = 6
DEFAULT_HOLDOUT_FRACTION: float = 0.2
DEFAULT_N_FOLDS: int = 5
MIN_TRAIN_BARS: int = 252
MIN_TEST_BARS: int = 63
EMBARGO_BARS: int = 20

PreparedFold = tuple[pd.DataFrame, pd.DataFrame, DetectorArtifacts, DetectorArtifacts]

# Preserved for backwards compatibility and legacy long-history validation.
LEGACY_FOLD_DEFINITIONS: list[tuple[str, str, str, str]] = [
    ("1983-01-01", "1993-01-01", "1993-01-01", "1996-01-01"),
    ("1986-01-01", "1996-01-01", "1996-01-01", "1999-01-01"),
    ("1989-01-01", "1999-01-01", "1999-01-01", "2002-01-01"),
    ("1992-01-01", "2002-01-01", "2002-01-01", "2005-01-01"),
    ("1995-01-01", "2005-01-01", "2005-01-01", "2008-01-01"),
]
FOLD_DEFINITIONS = LEGACY_FOLD_DEFINITIONS


@dataclasses.dataclass(frozen=True)
class ValidationScheme:
    mode: str
    holdout_mode: str
    holdout_cutoff: pd.Timestamp | None
    holdout_fraction: float
    n_folds: int
    embargo_bars: int
    min_train_bars: int
    min_test_bars: int


def _get_time_series(df: pd.DataFrame) -> pd.Series:
    """Get datetime series from df (handles DatetimeIndex or 'time' column)."""
    if isinstance(df.index, pd.DatetimeIndex):
        return pd.Series(df.index, index=df.index)
    return pd.to_datetime(df["time"], unit="s")


def infer_validation_scheme(df: pd.DataFrame) -> ValidationScheme:
    """Choose a validation scheme that fits the dataset span and size."""
    if len(df) <= (MIN_TRAIN_BARS + MIN_TEST_BARS + EMBARGO_BARS):
        raise ValueError(
            f"Dataset has {len(df)} bars, which is too small for validation "
            f"(needs more than {MIN_TRAIN_BARS + MIN_TEST_BARS + EMBARGO_BARS})."
        )

    time_series = _get_time_series(df)
    first_ts = pd.Timestamp(time_series.iloc[0])
    last_ts = pd.Timestamp(time_series.iloc[-1])
    has_legacy_span = (
        first_ts <= pd.Timestamp("1983-01-01")
        and last_ts >= pd.Timestamp("2008-01-01")
    )
    has_legacy_holdout = (
        first_ts < pd.Timestamp("2010-01-01")
        and last_ts >= pd.Timestamp("2010-01-01")
    )

    if has_legacy_span:
        return ValidationScheme(
            mode="legacy_dates",
            holdout_mode="date_cutoff" if has_legacy_holdout else "last_fraction",
            holdout_cutoff=pd.Timestamp("2010-01-01") if has_legacy_holdout else None,
            holdout_fraction=DEFAULT_HOLDOUT_FRACTION,
            n_folds=len(LEGACY_FOLD_DEFINITIONS),
            embargo_bars=EMBARGO_BARS,
            min_train_bars=MIN_TRAIN_BARS,
            min_test_bars=MIN_TEST_BARS,
        )

    return ValidationScheme(
        mode="rolling_bars",
        holdout_mode="last_fraction",
        holdout_cutoff=None,
        holdout_fraction=DEFAULT_HOLDOUT_FRACTION,
        n_folds=DEFAULT_N_FOLDS,
        embargo_bars=EMBARGO_BARS,
        min_train_bars=MIN_TRAIN_BARS,
        min_test_bars=MIN_TEST_BARS,
    )


def describe_validation_scheme(df: pd.DataFrame) -> dict[str, object]:
    scheme = infer_validation_scheme(df)
    folds = walk_forward_folds(df, scheme=scheme)
    return {
        "mode": scheme.mode,
        "holdout_mode": scheme.holdout_mode,
        "holdout_cutoff": str(scheme.holdout_cutoff) if scheme.holdout_cutoff is not None else None,
        "holdout_fraction": scheme.holdout_fraction,
        "n_folds": len(folds),
        "embargo_bars": scheme.embargo_bars,
        "min_train_bars": scheme.min_train_bars,
        "min_test_bars": scheme.min_test_bars,
    }


def temporal_split(
    df: pd.DataFrame,
    scheme: ValidationScheme | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Layer 1 holdout split."""
    scheme = scheme or infer_validation_scheme(df)
    if scheme.holdout_mode == "date_cutoff" and scheme.holdout_cutoff is not None:
        cutoff = scheme.holdout_cutoff
        if isinstance(df.index, pd.DatetimeIndex):
            is_mask = df.index < cutoff
        else:
            time_col = pd.to_datetime(df["time"], unit="s")
            is_mask = time_col < cutoff
    else:
        split_idx = max(scheme.min_train_bars, int(len(df) * (1.0 - scheme.holdout_fraction)))
        split_idx = min(split_idx, len(df) - scheme.min_test_bars)
        if split_idx <= 0 or split_idx >= len(df):
            raise ValueError(
                f"Dataset has {len(df)} bars, which is too small for holdout split "
                f"(min_train={scheme.min_train_bars}, min_test={scheme.min_test_bars})."
            )
        is_mask = pd.Series(False, index=df.index)
        is_mask.iloc[:split_idx] = True
    df_is = df[is_mask].copy()
    df_oos = df[~is_mask].copy()
    logger.info(
        "temporal_split[%s]: IS=%d bars (ends %s), OOS=%d bars (starts %s)",
        scheme.holdout_mode,
        len(df_is),
        df_is.index[-1] if len(df_is) else "N/A",
        len(df_oos),
        df_oos.index[0] if len(df_oos) else "N/A",
    )
    return df_is, df_oos


def walk_forward_folds(
    df: pd.DataFrame,
    scheme: ValidationScheme | None = None,
) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    """Generate walk-forward folds that fit the dataset."""
    scheme = scheme or infer_validation_scheme(df)
    folds: list[tuple[pd.DataFrame, pd.DataFrame]] = []
    time_series = _get_time_series(df)

    if scheme.mode == "legacy_dates":
        for is_start, is_end, oos_start, oos_end in LEGACY_FOLD_DEFINITIONS:
            is_start_ts = pd.Timestamp(is_start)
            is_end_ts = pd.Timestamp(is_end)
            oos_start_ts = pd.Timestamp(oos_start)
            oos_end_ts = pd.Timestamp(oos_end)

            is_mask = (time_series >= is_start_ts) & (time_series < is_end_ts)
            oos_mask = (time_series >= oos_start_ts) & (time_series < oos_end_ts)

            df_is_raw = df[is_mask].copy()
            df_oos = df[oos_mask].copy()
            if len(df_is_raw) <= scheme.embargo_bars:
                logger.warning("Fold IS slice too small for embargo (%d bars), skipping", len(df_is_raw))
                continue
            df_is = df_is_raw.iloc[:-scheme.embargo_bars].copy()
            if len(df_is) > 0 and len(df_oos) > 0:
                folds.append((df_is, df_oos))
        logger.info("walk_forward_folds[%s]: %d valid folds generated", scheme.mode, len(folds))
        return folds

    n = len(df)
    required = scheme.min_train_bars + scheme.min_test_bars + scheme.embargo_bars
    if n <= required:
        raise ValueError(
            f"Dataset has {n} bars, which is too small for rolling folds (needs more than {required})."
        )

    test_size = max(scheme.min_test_bars, int(round(n * 0.1)))
    max_test_size = max(
        scheme.min_test_bars,
        (n - scheme.min_train_bars - scheme.embargo_bars) // max(scheme.n_folds, 1),
    )
    test_size = min(test_size, max_test_size)
    initial_train = max(scheme.min_train_bars, n - scheme.n_folds * test_size)
    if initial_train + scheme.embargo_bars + test_size > n:
        initial_train = max(
            scheme.min_train_bars,
            n - (scheme.embargo_bars + scheme.n_folds * test_size),
        )

    for fold_idx in range(scheme.n_folds):
        oos_start_idx = initial_train + fold_idx * test_size
        oos_end_idx = min(oos_start_idx + test_size, n)
        is_end_idx = oos_start_idx - scheme.embargo_bars
        if is_end_idx < scheme.min_train_bars:
            continue
        if (oos_end_idx - oos_start_idx) < scheme.min_test_bars:
            continue
        df_is = df.iloc[:is_end_idx].copy()
        df_oos = df.iloc[oos_start_idx:oos_end_idx].copy()
        if len(df_is) > 0 and len(df_oos) > 0:
            folds.append((df_is, df_oos))

    logger.info("walk_forward_folds[%s]: %d valid folds generated", scheme.mode, len(folds))
    return folds


def load_data(path: str | Path) -> pd.DataFrame:
    """Load OHLCV CSV with Unix-timestamp 'time' column."""
    df = pd.read_csv(path)
    df.columns = [c.lower() for c in df.columns]
    required = {"open", "high", "low", "close", "volume"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing OHLCV columns in {path}: {sorted(missing)}")
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
    """Load cross-asset data with optional 1M→1D resampling."""
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
    df_daily = df_daily[df_daily["volume"] > 0]

    logger.info(
        "load_cross_asset: resampled %d minute bars → %d daily bars from %s",
        len(df), len(df_daily), Path(path).name,
    )
    return df_daily


def build_optuna_objective(
    df_full: pd.DataFrame,
    params_from_trial: Callable[[optuna.Trial, str], Params],
    side: str,
) -> Callable[[optuna.Trial], float]:
    """Build an Optuna objective function for one study side."""
    if side not in ("high", "low"):
        raise ValueError(f"side must be 'high' or 'low', got {side!r}")

    folds = walk_forward_folds(df_full)
    if not folds:
        raise ValueError("No walk-forward folds could be constructed from df_full.")

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


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    data_dir = Path(__file__).resolve().parent.parent / "data" / "raw"
    spx_path = data_dir / "SPX_1D_18710201_20260318.csv"
    if not spx_path.exists():
        print(f"Data file not found: {spx_path}", file=sys.stderr)
        sys.exit(1)

    df = load_data(spx_path)
    legacy_scheme = infer_validation_scheme(df)
    assert legacy_scheme.mode == "legacy_dates"
    df_is, df_oos = temporal_split(df, legacy_scheme)
    assert len(df_is) > 0 and len(df_oos) > 0
    folds = walk_forward_folds(df, legacy_scheme)
    assert len(folds) == 5, f"Expected 5 legacy folds, got {len(folds)}"

    modern_path = data_dir / "DAX_1M_20250113_20260227.csv"
    if modern_path.exists():
        modern_df = load_data(modern_path)
        modern_scheme = infer_validation_scheme(modern_df)
        assert modern_scheme.mode == "rolling_bars"
        modern_is, modern_oos = temporal_split(modern_df, modern_scheme)
        assert len(modern_is) > 0 and len(modern_oos) > 0
        modern_folds = walk_forward_folds(modern_df, modern_scheme)
        assert len(modern_folds) >= 3, f"Expected at least 3 rolling folds, got {len(modern_folds)}"
        logger.info(
            "Modern rolling validation OK: folds=%d, IS=%d, OOS=%d",
            len(modern_folds),
            len(modern_is),
            len(modern_oos),
        )

    logger.info("validation.py self-test PASSED")
    print("\nvalidation.py self-test PASSED")
