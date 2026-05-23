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

MIN_SIGNALS_PER_FOLD: int = 1
MIN_TRAIN_BARS: int = 252
MIN_TEST_BARS: int = 63
EMBARGO_BARS: int = 20

# Bar-based fold scheme (timeframe-agnostic).
# All sizes are expressed as fractions of the full dataset length so the same
# scheme works on daily, hourly, or minute data without retuning.
DEFAULT_HOLDOUT_FRACTION: float = 0.20  # Last 20% of data reserved for holdout
DEFAULT_IS_FRACTION: float = 0.10       # In-sample slice per fold (10% of full)
DEFAULT_OOS_FRACTION: float = 0.03      # Out-of-sample slice per fold (3% of full)
DEFAULT_STEP_FRACTION: float = 0.05     # Step between fold starts (5% of full)

PreparedFold = tuple[pd.DataFrame, pd.DataFrame, DetectorArtifacts, DetectorArtifacts]


@dataclasses.dataclass(frozen=True)
class ValidationScheme:
    is_bars: int
    oos_bars: int
    step_bars: int
    embargo_bars: int
    holdout_bars: int
    n_bars_active: int  # = total bars - holdout_bars
    # Retained for backward compatibility with consumers that inspect these.
    min_train_bars: int = MIN_TRAIN_BARS
    min_test_bars: int = MIN_TEST_BARS


def _get_time_series(df: pd.DataFrame) -> pd.Series:
    """Get datetime series from df (handles DatetimeIndex or 'time' column)."""
    if isinstance(df.index, pd.DatetimeIndex):
        return pd.Series(df.index, index=df.index)
    return pd.to_datetime(df["time"], unit="s")


def infer_validation_scheme(df: pd.DataFrame) -> ValidationScheme:
    """Compute bar-based fold sizes from dataset length.

    All sizes scale with dataset length so the same scheme works on daily,
    hourly, or minute data. The last `DEFAULT_HOLDOUT_FRACTION` bars are
    reserved for the holdout split; folds are generated from the remaining
    "active" range using sliding IS/OOS windows.
    """
    n = len(df)
    if n <= (MIN_TRAIN_BARS + MIN_TEST_BARS + EMBARGO_BARS):
        raise ValueError(
            f"Dataset has {n} bars, which is too small for validation "
            f"(needs more than {MIN_TRAIN_BARS + MIN_TEST_BARS + EMBARGO_BARS})."
        )

    holdout_bars = max(MIN_TEST_BARS, int(round(n * DEFAULT_HOLDOUT_FRACTION)))
    n_active = n - holdout_bars
    is_bars = max(MIN_TRAIN_BARS, int(round(n * DEFAULT_IS_FRACTION)))
    oos_bars = max(MIN_TEST_BARS, int(round(n * DEFAULT_OOS_FRACTION)))
    step_bars = max(MIN_TEST_BARS, int(round(n * DEFAULT_STEP_FRACTION)))

    required = is_bars + oos_bars + EMBARGO_BARS
    if required > n_active:
        raise ValueError(
            f"Dataset has {n} bars; active range {n_active} cannot hold one fold "
            f"(needs IS={is_bars} + OOS={oos_bars} + embargo={EMBARGO_BARS})."
        )

    return ValidationScheme(
        is_bars=is_bars,
        oos_bars=oos_bars,
        step_bars=step_bars,
        embargo_bars=EMBARGO_BARS,
        holdout_bars=holdout_bars,
        n_bars_active=n_active,
    )


def describe_validation_scheme(df: pd.DataFrame) -> dict[str, object]:
    scheme = infer_validation_scheme(df)
    folds = walk_forward_folds(df, scheme=scheme)
    return {
        "mode": "rolling_bars_sliding",
        "is_bars": scheme.is_bars,
        "oos_bars": scheme.oos_bars,
        "step_bars": scheme.step_bars,
        "embargo_bars": scheme.embargo_bars,
        "holdout_bars": scheme.holdout_bars,
        "n_bars_active": scheme.n_bars_active,
        "n_folds": len(folds),
    }


def temporal_split(
    df: pd.DataFrame,
    scheme: ValidationScheme | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Layer 1 holdout split: last `holdout_bars` bars are OOS, rest is IS."""
    scheme = scheme or infer_validation_scheme(df)
    n = len(df)
    split_idx = n - scheme.holdout_bars
    if split_idx <= 0 or split_idx >= n:
        raise ValueError(
            f"Dataset has {n} bars, which is too small for holdout split "
            f"(holdout_bars={scheme.holdout_bars})."
        )
    df_is = df.iloc[:split_idx].copy()
    df_oos = df.iloc[split_idx:].copy()
    logger.info(
        "temporal_split: IS=%d bars (ends %s), OOS=%d bars (starts %s)",
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
    """Sliding bar-based walk-forward folds over the active range.

    Each fold is (IS_slice, OOS_slice) where:
        IS  = df.iloc[is_start : is_start + is_bars]
        OOS = df.iloc[is_end + embargo : is_end + embargo + oos_bars]
    `is_start` advances by `step_bars` between folds. All slices live
    within the active range (first `n_bars_active` bars).
    """
    scheme = scheme or infer_validation_scheme(df)
    folds: list[tuple[pd.DataFrame, pd.DataFrame]] = []
    active = scheme.n_bars_active

    is_start = 0
    while is_start + scheme.is_bars + scheme.embargo_bars + scheme.oos_bars <= active:
        is_end = is_start + scheme.is_bars
        oos_start = is_end + scheme.embargo_bars
        oos_end = oos_start + scheme.oos_bars
        df_is = df.iloc[is_start:is_end].copy()
        df_oos = df.iloc[oos_start:oos_end].copy()
        if len(df_is) >= scheme.min_train_bars and len(df_oos) >= scheme.min_test_bars:
            folds.append((df_is, df_oos))
        is_start += scheme.step_bars

    logger.info(
        "walk_forward_folds[sliding bar]: %d folds (IS=%d, OOS=%d, step=%d) over %d active bars",
        len(folds), scheme.is_bars, scheme.oos_bars, scheme.step_bars, active,
    )
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
    """Build an Optuna objective function for one study side.

    Scorer v3 changes (items A1, A2, B3):
    - The objective value is the **bootstrap-CI lower bound** on the
      per-fold mean instead of the v2 ``mean - 0.5·std`` heuristic. The
      ``0.5`` was a magic number unrelated to the fold count or fold
      variance; the bootstrap LCB calibrates itself to both.
    - The bootstrap uses block length 2 (item A2) to respect the heavy
      IS overlap between adjacent folds (IS=10%, step=5% → 50% overlap).
    - Per-fold component metrics (item B3) are aggregated and exposed
      via ``trial.set_user_attr`` so post-hoc Pareto inspection can be
      done without rerunning trials. The scalar objective contract is
      unchanged — components are diagnostic only.
    """
    if side not in ("high", "low"):
        raise ValueError(f"side must be 'high' or 'low', got {side!r}")

    folds = walk_forward_folds(df_full)
    if not folds:
        raise ValueError("No walk-forward folds could be constructed from df_full.")

    prepared_folds = prepare_walk_forward_folds(df_full, folds)

    def objective(trial: optuna.Trial) -> float:
        params = params_from_trial(trial, side)
        fold_scores: list[float] = []
        per_fold_components: list[dict[str, float]] = []
        for fold_idx, (score, components) in enumerate(
            _evaluate_with_components(params, side, prepared_folds)
        ):
            fold_scores.append(score)
            per_fold_components.append(components)
            # Scorer v3 (A1 + A2): running LCB is the bootstrap-CI lower
            # bound with block length 2. Falls back gracefully on single
            # folds (CI lower == CI upper == mean).
            running_lcb = fold_scores_bootstrap_ci(
                fold_scores, n_boot=1000, alpha=0.10, block_len=2
            )[0]
            trial.report(running_lcb, fold_idx)
            if trial.should_prune():
                _set_component_attrs(trial, per_fold_components)
                raise optuna.exceptions.TrialPruned()
        # Final objective: bootstrap LCB over all folds (A1).
        final_lcb = fold_scores_bootstrap_ci(
            fold_scores, n_boot=1000, alpha=0.10, block_len=2
        )[0]
        _set_component_attrs(trial, per_fold_components)
        return float(final_lcb)

    return objective


def _evaluate_with_components(
    params: Params,
    side: str,
    prepared_folds: list[PreparedFold],
):
    """Generator yielding ``(fold_score, components)`` for each prepared fold.

    Scorer v3 item B3 helper. Mirrors
    ``evaluate_params_on_prepared_folds`` but threads the component
    diagnostics back out alongside the scalar.
    """
    from .scoring import fold_score_high, fold_score_low

    for df_is_r, df_oos_r, artifacts_is, artifacts_oos in prepared_folds:
        det_is = SpeculatorDetector(df_is_r, params, artifacts_is).run()
        det_oos = SpeculatorDetector(df_oos_r, params, artifacts_oos).run()
        if side == "high":
            score, components = fold_score_high(
                df_is_r,
                df_oos_r,
                det_is["signal_high"],
                det_oos["signal_high"],
                return_components=True,
            )
        else:
            score, components = fold_score_low(
                df_is_r,
                df_oos_r,
                det_is["signal_low"],
                det_oos["signal_low"],
                return_components=True,
            )
        yield float(score), components


def _set_component_attrs(
    trial: optuna.Trial,
    per_fold_components: list[dict[str, float]],
) -> None:
    """Aggregate per-fold component metrics and attach to the trial.

    Scorer v3 item B3. Means are computed across the (possibly partial)
    list of folds completed before pruning so the user attrs are
    available even on pruned trials.
    """
    if not per_fold_components:
        return
    keys = per_fold_components[0].keys()
    for key in keys:
        values = [float(c.get(key, 0.0)) for c in per_fold_components]
        try:
            trial.set_user_attr(f"component_mean_{key}", float(np.mean(values)))
        except Exception:  # pragma: no cover - best-effort logging
            logger.debug("set_user_attr failed for component_mean_%s", key)


def fold_scores_bootstrap_ci(
    scores: list[float],
    n_boot: int = 1000,
    alpha: float = 0.10,
    block_len: int = 2,
) -> tuple[float, float]:
    """Stationary block-bootstrap CI for the mean of fold scores.

    Scorer v3 (item A2): block length defaults to 2 to respect the
    heavy IS overlap between adjacent walk-forward folds (the validation
    scheme uses IS=10% with step=5%, giving 50% overlap). With block=1
    the bootstrap treats folds as independent and reports an overly
    tight CI.

    For each bootstrap iteration we sample ``ceil(n / block_len)``
    random start indices, take ``block_len`` consecutive scores per
    start (mod n_folds, stationary block bootstrap), concatenate, and
    truncate to length n_folds. The mean of that sample is one bootstrap
    draw.

    Single-fold inputs return ``(score, score)`` so callers can use the
    lower bound as a graceful fallback.

    Args:
        scores: Per-fold OOS scores (any sequence convertible to float).
        n_boot: Number of bootstrap resamples (default 1000).
        alpha: Two-sided significance level (default 0.10 → 90% CI).
        block_len: Block length for stationary bootstrap (default 2).

    Returns:
        ``(lower, upper)`` percentiles of the bootstrap mean distribution.
        Returns ``(0.0, 0.0)`` for an empty input list.
    """
    if not scores:
        return (0.0, 0.0)
    arr = np.asarray(list(scores), dtype=float)
    n = len(arr)
    if n == 1:
        return (float(arr[0]), float(arr[0]))
    block_len = max(1, min(int(block_len), n))
    n_blocks = int(np.ceil(n / block_len))
    rng = np.random.default_rng(42)
    boot_means = np.empty(n_boot, dtype=float)
    block_offsets = np.arange(block_len)
    for b in range(n_boot):
        starts = rng.integers(0, n, size=n_blocks)
        # Stationary block bootstrap with wrap-around.
        idx = (starts[:, None] + block_offsets[None, :]) % n  # (n_blocks, block_len)
        sample = arr[idx.reshape(-1)][:n]
        boot_means[b] = sample.mean()
    lo = float(np.percentile(boot_means, 100.0 * alpha / 2.0))
    hi = float(np.percentile(boot_means, 100.0 * (1.0 - alpha / 2.0)))
    return (lo, hi)


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
        else:
            score = fold_score_low(df_is_r, df_oos_r, sig_low_is, sig_low_oos)
        fold_scores.append(score)
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
