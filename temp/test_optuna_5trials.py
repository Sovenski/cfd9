"""Local test: 5 Optuna trials per side with local SQLite storage."""
import dataclasses
import logging
import sys
from pathlib import Path

import numpy as np
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

# Suppress Optuna noise
optuna.logging.set_verbosity(optuna.logging.WARNING)
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent.parent))

from src import (
    Params,
    SpeculatorDetector,
    build_optuna_objective,
    fold_score_high,
    fold_score_low,
    load_data,
)

# ── Config ────────────────────────────────────────────────────────────────────
N_TRIALS = 5
STORAGE = "sqlite:///temp/test_optuna.db"
SPX_PATH = "data/raw/SPX_1D_18710201_20260318.csv"


# ── params_from_trial (same as notebook) ──────────────────────────────────────
def _params_fields(p: Params) -> dict:
    return {f.name: getattr(p, f.name) for f in dataclasses.fields(p)}


def params_from_trial(trial: optuna.Trial, side: str) -> Params:
    if side not in ("high", "low"):
        raise ValueError(f"side must be 'high' or 'low', got {side!r}")
    s = side

    S_detect = trial.suggest_int(f"{s}_S_detect", 5, 60)
    scale_start = trial.suggest_int(f"{s}_scale_start", 2, 30)
    scale_end = trial.suggest_int(f"{s}_scale_end", 100, 500)
    scale_step = trial.suggest_int(f"{s}_scale_step", 2, 20)
    min_duration = trial.suggest_int(f"{s}_min_duration", 1, 20)
    cooldown_bars = trial.suggest_int(f"{s}_cooldown_bars", 1, 20)
    price_gate_lb = trial.suggest_int(f"{s}_price_gate_lb", 5, 100)
    vola_range_len = trial.suggest_int(f"{s}_vola_range_len", 20, 200)
    er_period = trial.suggest_int(f"{s}_er_period", 5, 60)
    confirm_count = trial.suggest_int(f"{s}_confirm_count", 1, 5)
    pivot_drift_lb = trial.suggest_int(f"{s}_pivot_drift_lb", 2, 20)
    pivot_drift_confirm_bias = trial.suggest_int(f"{s}_pivot_drift_confirm_bias", 0, 2)
    pct_extreme = trial.suggest_float(f"{s}_pct_extreme", 0.70, 0.99)
    min_agreement = trial.suggest_float(f"{s}_min_agreement", 0.10, 0.90)
    dur_extreme_pct = trial.suggest_float(f"{s}_dur_extreme_pct", 0.50, 0.99)
    vol_surge_thresh = trial.suggest_float(f"{s}_vol_surge_thresh", 1.0, 3.0)
    scale_div_thresh = trial.suggest_float(f"{s}_scale_div_thresh", 0.10, 0.60)
    slope_thresh = trial.suggest_float(f"{s}_slope_thresh", 0.01, 0.50)
    vola_high_pct = trial.suggest_float(f"{s}_vola_high_pct", 0.50, 0.99)
    pivot_drift_thresh = trial.suggest_float(f"{s}_pivot_drift_thresh", 0.001, 0.050)
    pivot_drift_gate_mult = trial.suggest_float(f"{s}_pivot_drift_gate_mult", 1.0, 10.0)
    momentum_velocity_thresh = trial.suggest_float(f"{s}_momentum_velocity_thresh", 0.0, 0.05)
    gjr_vote_thresh = trial.suggest_float(f"{s}_gjr_vote_thresh", 0.05, 0.50)
    har_vote_thresh = trial.suggest_float(f"{s}_har_vote_thresh", 0.05, 0.50)
    er_directional = trial.suggest_categorical(f"{s}_er_directional", [True, False])
    use_trend = trial.suggest_categorical(f"{s}_use_trend", [True, False])
    use_volume = trial.suggest_categorical(f"{s}_use_volume", [True, False])
    use_momentum = trial.suggest_categorical(f"{s}_use_momentum", [True, False])
    use_momentum_velocity = trial.suggest_categorical(f"{s}_use_momentum_velocity", [True, False])
    use_volatility = trial.suggest_categorical(f"{s}_use_volatility", [True, False])
    use_er_gate = trial.suggest_categorical(f"{s}_use_er_gate", [True, False])
    use_gjr_asym = trial.suggest_categorical(f"{s}_use_gjr_asym", [True, False])
    use_har_vol = trial.suggest_categorical(f"{s}_use_har_vol", [True, False])
    vola_method = trial.suggest_categorical(f"{s}_vola_method", ["ATR", "StdDev", "Intraday"])
    momentum_velocity_mode = trial.suggest_categorical(
        f"{s}_momentum_velocity_mode", ["Trend", "Reversal"]
    )

    kwargs_high = dict(
        S_detect_high=S_detect, scale_start_high=scale_start, scale_end_high=scale_end,
        scale_step_high=scale_step, min_duration_high=min_duration, cooldown_bars_high=cooldown_bars,
        price_gate_lb_high=price_gate_lb, vola_range_len_high=vola_range_len, er_period_high=er_period,
        pct_extreme_high=pct_extreme, min_agreement_high=min_agreement, dur_extreme_pct_high=dur_extreme_pct,
        confirm_count_high=confirm_count, vol_surge_thresh_high=vol_surge_thresh,
        scale_div_thresh_high=scale_div_thresh, slope_thresh_high=slope_thresh,
        vola_high_pct_high=vola_high_pct, pivot_drift_lookback_high=pivot_drift_lb,
        pivot_drift_thresh_high=pivot_drift_thresh, pivot_drift_gate_mult_high=pivot_drift_gate_mult,
        pivot_drift_confirm_bias_high=pivot_drift_confirm_bias,
        momentum_velocity_thresh_high=momentum_velocity_thresh,
        er_directional_high=er_directional, use_trend_high=use_trend, use_volume_high=use_volume,
        use_momentum_high=use_momentum, use_momentum_velocity_high=use_momentum_velocity,
        use_volatility_high=use_volatility, use_er_gate_high=use_er_gate,
        use_gjr_asym_high=use_gjr_asym, use_har_vol_high=use_har_vol,
        gjr_vote_thresh_high=gjr_vote_thresh, har_vote_thresh_high=har_vote_thresh,
        vola_method_high=vola_method, momentum_velocity_mode_high=momentum_velocity_mode,
    )
    kwargs_low = dict(
        S_detect_low=S_detect, scale_start_low=scale_start, scale_end_low=scale_end,
        scale_step_low=scale_step, min_duration_low=min_duration, cooldown_bars_low=cooldown_bars,
        price_gate_lb_low=price_gate_lb, vola_range_len_low=vola_range_len, er_period_low=er_period,
        pct_extreme_low=pct_extreme, min_agreement_low=min_agreement, dur_extreme_pct_low=dur_extreme_pct,
        confirm_count_low=confirm_count, vol_surge_thresh_low=vol_surge_thresh,
        scale_div_thresh_low=scale_div_thresh, slope_thresh_low=slope_thresh,
        vola_high_pct_low=vola_high_pct, pivot_drift_lookback_low=pivot_drift_lb,
        pivot_drift_thresh_low=pivot_drift_thresh, pivot_drift_gate_mult_low=pivot_drift_gate_mult,
        pivot_drift_confirm_bias_low=pivot_drift_confirm_bias,
        momentum_velocity_thresh_low=momentum_velocity_thresh,
        er_directional_low=er_directional, use_trend_low=use_trend, use_volume_low=use_volume,
        use_momentum_low=use_momentum, use_momentum_velocity_low=use_momentum_velocity,
        use_volatility_low=use_volatility, use_er_gate_low=use_er_gate,
        use_gjr_asym_low=use_gjr_asym, use_har_vol_low=use_har_vol,
        gjr_vote_thresh_low=gjr_vote_thresh, har_vote_thresh_low=har_vote_thresh,
        vola_method_low=vola_method, momentum_velocity_mode_low=momentum_velocity_mode,
    )
    base_dict = _params_fields(Params())
    overrides = kwargs_high if side == "high" else kwargs_low
    return Params(**{**base_dict, **overrides})


# ── Run ───────────────────────────────────────────────────────────────────────
def main():
    logger.info("Loading SPX 1D data...")
    df = load_data(SPX_PATH)
    logger.info("  %d bars, %s – %s", len(df), df.index[0].date(), df.index[-1].date())

    Path("temp").mkdir(exist_ok=True)

    for side in ("high", "low"):
        logger.info("\n── Running study_%s (%d trials) ──", side, N_TRIALS)
        study = optuna.create_study(
            study_name=f"test_speculatores_{side}",
            storage=STORAGE,
            direction="maximize",
            sampler=TPESampler(n_startup_trials=50, seed=42),
            pruner=MedianPruner(n_startup_trials=20, n_warmup_steps=2),
            load_if_exists=True,
        )
        objective = build_optuna_objective(df, params_from_trial, side)
        study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)

        completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        pruned = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
        logger.info(
            "  Completed: %d | Pruned: %d | Best score: %.4f (trial #%d)",
            len(completed), len(pruned),
            study.best_value, study.best_trial.number,
        )
        logger.info("  Best params (side=%s):", side)
        for k, v in list(study.best_params.items())[:6]:
            logger.info("    %s = %s", k.replace(f"{side}_", ""), v)
        logger.info("    ... (%d total params)", len(study.best_params))

    logger.info("\n✓ 5-trial local test PASSED — pipeline is end-to-end functional.")


if __name__ == "__main__":
    main()
