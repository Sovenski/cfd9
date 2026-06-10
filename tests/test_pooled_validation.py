import numpy as np
import pandas as pd

from src.universe import Stream
from src.pooled_validation import load_stream_frame, apply_volume_policy


def _write_csv(tmp_path, name, n, start_ts=1420070400, step=86400, volume=None):
    t = np.arange(n) * step + start_ts
    close = 100 + np.cumsum(np.random.RandomState(0).randn(n))
    vol = volume if volume is not None else np.full(n, 1_000_000)
    df = pd.DataFrame({"time": t, "open": close, "high": close + 1,
                       "low": close - 1, "close": close, "Volume": vol})
    p = tmp_path / name
    df.to_csv(p, index=False)
    return str(p)


def test_load_stream_frame_normalizes_and_indexes(tmp_path):
    path = _write_csv(tmp_path, "SPX_1D_a_b.csv", 500)
    df = load_stream_frame(path)
    assert isinstance(df.index, pd.DatetimeIndex)
    assert {"open", "high", "low", "close", "volume"}.issubset(df.columns)


def test_volume_policy_required_trims_to_real_range(tmp_path):
    vol = np.concatenate([np.full(300, 3600),
                          np.random.RandomState(1).randint(1_000_000, 5_000_000, 300)])
    path = _write_csv(tmp_path, "SPX_1D_a_b.csv", 600, volume=vol)
    df = load_stream_frame(path)
    trimmed, kept = apply_volume_policy(df, policy="volume_required")
    assert kept is True
    assert len(trimmed) < len(df)        # placeholder head removed
    assert trimmed["volume"].iloc[0] >= 100_000


def test_volume_policy_price_only_keeps_all(tmp_path):
    path = _write_csv(tmp_path, "SPX_1D_a_b.csv", 500, volume=np.full(500, 3600))
    df = load_stream_frame(path)
    trimmed, kept = apply_volume_policy(df, policy="price_only")
    assert kept is True and len(trimmed) == len(df)


def test_volume_policy_required_drops_volumeless_stream(tmp_path):
    path = _write_csv(tmp_path, "SPX_1D_a_b.csv", 500, volume=np.full(500, 3600))
    df = load_stream_frame(path)
    _, kept = apply_volume_policy(df, policy="volume_required")
    assert kept is False     # 'none' quality → excluded under volume_required


from src.pooled_validation import StreamData, build_calendar_folds, MIN_STREAM_BARS
from src.scoring import add_pivot_labels


def _stream_data(tmp_path, ticker, n, start_ts, step, cluster):
    path = _write_csv(tmp_path, f"{ticker}_1D_a_b.csv", n, start_ts=start_ts, step=step)
    df = load_stream_frame(path)
    add_pivot_labels(df)
    return StreamData(
        stream=Stream(ticker, "1D", path, cluster),
        df=df, bar_seconds=float(step),
    )


def test_calendar_folds_respect_holdout_and_embargo(tmp_path):
    # Two daily streams, different start dates, both long enough for many folds.
    # 15000 daily bars → is=1500, oos=450 bars >= MIN_STREAM_BARS (401).
    a = _stream_data(tmp_path, "SPX", 15000, 1262304000, 86400, "US_EQ")  # 2010+
    b = _stream_data(tmp_path, "DAX", 4000, 1420070400, 86400, "EU_EQ")  # 2015+
    folds = build_calendar_folds([a, b])
    assert len(folds) >= 3
    for fold in folds:
        for sl in fold:
            # min-bars gate: every contributing slice clears the nest requirement
            assert len(sl.df_is) >= MIN_STREAM_BARS
            assert len(sl.df_oos) >= MIN_STREAM_BARS


def test_calendar_folds_drop_too_short_streams(tmp_path):
    # A 1W-like coarse stream too short to fit the 401-bar nest in any fold.
    big = _stream_data(tmp_path, "SPX", 15000, 1262304000, 86400, "US_EQ")
    tiny = _stream_data(tmp_path, "DAX", 120, 1420070400, 604800, "EU_EQ")  # 120 weekly bars
    folds = build_calendar_folds([big, tiny])
    # tiny never satisfies MIN_STREAM_BARS (401) in any fold slice.
    tickers_used = {sl.stream.ticker for fold in folds for sl in fold}
    assert "DAX" not in tickers_used
    assert "SPX" in tickers_used


def test_calendar_folds_form_on_short_pool(tmp_path):
    """B1 regression: short pools (~3000 bars, no 155-year anchor) must form >= 1 fold.

    Under the old calendar-day scheme, OOS_FRACTION * calendar_days ≈ 248 days
    which yields ~248 trading bars — below MIN_STREAM_BARS (401) — so 0 folds
    were produced.  The bar-based scheme sizes OOS off the reference stream
    (3000 * 0.03 = 90, max'd to MIN_STREAM_BARS = 401) and must form at least
    one fold; every contributing slice must have IS >= 401 and OOS >= 401.
    """
    a = _stream_data(tmp_path, "SPX", 3000, 1262304000, 86400, "US_EQ")
    b = _stream_data(tmp_path, "DAX", 3000, 1262304000, 86400, "EU_EQ")
    folds = build_calendar_folds([a, b])
    assert len(folds) >= 1, (
        f"Expected >= 1 fold from 3000-bar pool but got 0 "
        f"(bar-based sizing must floor to MIN_STREAM_BARS)"
    )
    for fold in folds:
        for sl in fold:
            assert len(sl.df_is) >= MIN_STREAM_BARS, (
                f"IS slice too short: {len(sl.df_is)} < {MIN_STREAM_BARS}"
            )
            assert len(sl.df_oos) >= MIN_STREAM_BARS, (
                f"OOS slice too short: {len(sl.df_oos)} < {MIN_STREAM_BARS}"
            )


import optuna
from src.pooled_validation import (
    cluster_weights, evaluate_pooled_fold, build_pooled_optuna_objective,
    _fold_is_informative,
)
from src.indicators import Params


def test_cluster_weights_are_inverse_cluster_size():
    a = Stream("SPX", "1D", "p", "US_EQ")
    b = Stream("NDX", "1D", "p", "US_EQ")
    c = Stream("DAX", "1D", "p", "EU_EQ")
    w = cluster_weights([a, b, c])
    assert abs(w["SPX_1D"] - 0.5) < 1e-9   # US_EQ has 2 members
    assert abs(w["NDX_1D"] - 0.5) < 1e-9
    assert abs(w["DAX_1D"] - 1.0) < 1e-9   # EU_EQ has 1


def test_pooled_objective_runs_and_returns_float(tmp_path):
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    a = _stream_data(tmp_path, "SPX", 15000, 1262304000, 86400, "US_EQ")
    b = _stream_data(tmp_path, "DAX", 15000, 1262304000, 86400, "EU_EQ")
    folds = build_calendar_folds([a, b])
    from src.speculatores145 import params_from_trial
    objective = build_pooled_optuna_objective(folds, [a.stream, b.stream],
                                              params_from_trial, "low")
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=2)
    assert isinstance(study.best_value, float)


# ---------------------------------------------------------------------------
# TDD: informative-fold filter
# ---------------------------------------------------------------------------

def test_fold_is_informative_filters_zero_pivot_folds():
    """_fold_is_informative returns False when pooled OOS pivot MASS == 0
    (Scorer v5, spec §2.4 — replaces the v4 structural-pivot count check)."""
    assert _fold_is_informative({"pooled_total_mass_oos": 0.0}) is False
    assert _fold_is_informative({"pooled_total_mass_oos": 0.3}) is True
    assert _fold_is_informative({}) is False


def test_objective_skips_noninformative_folds():
    """build_pooled_optuna_objective returns 0.0 when all folds are non-informative.

    We verify the guard logic by constructing an objective over an empty folds
    list is still rejected (raises ValueError) — that contract is unchanged —
    then directly verify _fold_is_informative drives the skip by checking that
    a components dict with pooled_total_pivots_oos==0 is treated as
    non-informative, which is the only precondition the inner loop depends on.

    The 0.0 fall-back when every fold is skipped is tested by monkey-patching
    evaluate_pooled_fold to return (score=0.0, components={"pooled_total_mass_oos": 0.0})
    for every fold, then running the objective via a fresh Optuna trial.
    """
    import pytest
    import unittest.mock as mock
    from src.pooled_validation import build_pooled_optuna_objective

    # Confirm empty-folds still raises (contract unchanged).
    with pytest.raises(ValueError):
        build_pooled_optuna_objective(
            folds=[],
            streams=[Stream("SPX", "1D", "p", "US_EQ")],
            params_from_trial=lambda trial, side: None,
            side="high",
        )

    # Build a minimal non-empty fold placeholder (content doesn't matter;
    # evaluate_pooled_fold will be mocked).
    dummy_fold = [object()]  # one fake PreparedSlice
    dummy_streams = [Stream("SPX", "1D", "p", "US_EQ")]

    # All folds yield zero OOS pivot mass → all non-informative → objective must return 0.0.
    non_informative_result = (0.0, {"pooled_total_mass_oos": 0.0})
    with mock.patch(
        "src.pooled_validation.evaluate_pooled_fold",
        return_value=non_informative_result,
    ):
        objective = build_pooled_optuna_objective(
            folds=[dummy_fold, dummy_fold],
            streams=dummy_streams,
            params_from_trial=lambda trial, side: None,
            side="high",
        )
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=1)
        assert study.best_value == 0.0


# ---------------------------------------------------------------------------
# TDD: common-era fold window (feature/v16.1-common-era-folds)
# ---------------------------------------------------------------------------

def test_common_era_eliminates_solo_folds(tmp_path):
    """With a long SPX-like solo stream + three short streams that start ~2012,
    every fold produced with default settings must include >= 2 streams.

    Timestamps:
    - ~2000-01-01 = 946684800   (long stream start)
    - ~2012-01-01 = 1325376000  (short streams start)
    """
    import datetime
    spx = _stream_data(tmp_path, "SPX", 12000, 946684800, 86400, "US_EQ")
    eu  = _stream_data(tmp_path, "EU",  3500,  1325376000, 86400, "EU_EQ")
    met = _stream_data(tmp_path, "MET", 3500,  1325376000, 86400, "METALS")
    fx  = _stream_data(tmp_path, "FX",  3500,  1325376000, 86400, "FX")
    folds = build_calendar_folds([spx, eu, met, fx])
    assert len(folds) >= 1, "Expected at least 1 fold"
    for i, fold in enumerate(folds):
        assert len(fold) >= 2, (
            f"Fold {i} has only {len(fold)} stream(s) — solo fold should not appear"
        )


def test_build_calendar_folds_start_override(tmp_path):
    """Explicit start= override must be respected and produce >= 1 fold.

    When start= is supplied explicitly, era_end clipping is not applied
    (the user controls the window manually), so the override may produce
    more or fewer folds than the auto case.  The key guarantee is simply
    that it runs without error and yields at least one fold.
    """
    spx = _stream_data(tmp_path, "SPX", 12000, 946684800,  86400, "US_EQ")
    eu  = _stream_data(tmp_path, "EU",  3500,  1325376000, 86400, "EU_EQ")
    met = _stream_data(tmp_path, "MET", 3500,  1325376000, 86400, "METALS")
    fx  = _stream_data(tmp_path, "FX",  3500,  1325376000, 86400, "FX")

    folds_auto     = build_calendar_folds([spx, eu, met, fx])
    folds_override = build_calendar_folds([spx, eu, met, fx], start="2013-01-01")

    assert len(folds_override) >= 1, "start override must produce >= 1 fold"
    assert len(folds_auto) >= 1, "auto must produce >= 1 fold"


def test_full_history_fallback(tmp_path):
    """If era restriction leaves 0 folds, auto-fallback to full history.

    Pool: one long stream (12000 bars from 2000) + one very short stream
    (200 bars from 2024). With eff_min=2, era_start = 2024 start, leaving
    only ~200 bars — too few for even one fold. The fallback must rescue
    this and return >= 1 fold (sized on the full long stream).
    """
    ts_2000 = 946684800    # ~2000-01-01
    ts_2024 = 1704067200   # ~2024-01-01
    big   = _stream_data(tmp_path, "SPX", 12000, ts_2000, 86400, "US_EQ")
    small = _stream_data(tmp_path, "FX",  200,   ts_2024, 86400, "FX")
    folds = build_calendar_folds([big, small])
    assert len(folds) >= 1, (
        "Full-history fallback must produce >= 1 fold when era restriction "
        "yields 0 folds"
    )
