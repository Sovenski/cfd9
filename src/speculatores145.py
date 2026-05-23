"""Speculatores 15 optimization pipeline.

Script-friendly orchestration for Optuna studies, Colab-safe multiprocessing,
and one-file Markdown exports that capture the equivalent of notebook outputs.

V15 adds two new search dimensions per side — ``use_edge_voting`` and
``edge_window`` — to match the Pine V15 edge-triggered voting semantics.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import multiprocessing as mp
import os
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import optuna
import pandas as pd
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend, JournalFileOpenLock
from optuna.study import MaxTrialsCallback
from optuna.trial import TrialState

from .detector import SpeculatorDetector, build_detector_artifacts
from .indicators import Params
from .parity import compare_python_to_tv, find_matching_enriched_export
from .scoring import compute_side_score
from .validation import (
    build_optuna_objective,
    describe_validation_scheme,
    evaluate_params_on_prepared_folds,
    load_cross_asset,
    load_data,
    prepare_walk_forward_folds,
    temporal_split,
    walk_forward_folds,
)

logger = logging.getLogger(__name__)

VERSION = "Speculatores 15 (Path A scoring, edge-triggered voting)"
VERSION_SLUG = "speculatores_15_pathA"
DEFAULT_RESULTS_DIR = Path("results")
DEFAULT_STORAGE_FILE = Path("temp") / f"{VERSION_SLUG}.journal"
DEFAULT_N_TRIALS = 500
DEFAULT_STARTUP_TRIALS = 80
DEFAULT_WORKERS_PER_SIDE = 2
DEFAULT_SEED = 42
DEFAULT_STABILITY_TRIALS = 50

TRIAL_STATES_DONE = (
    TrialState.COMPLETE,
    TrialState.PRUNED,
    TrialState.FAIL,
)

INSTRUMENTS: dict[str, dict[str, Any]] = {
    "DAX": {"path": "data/raw/DAX_1D_19700102_20260324.csv", "resample": False},
    "NDX": {"path": "data/raw/NDX_1M_20250203_20260226.csv", "resample": True},
    "GC1": {"path": "data/raw/GC1_1M_20260215_20260312.csv", "resample": True},
    "SI1": {"path": "data/raw/SI1_1M_20260215_20260312.csv", "resample": True},
    "WTI": {"path": "data/raw/WTI_1M_20260215_20260312.csv", "resample": True},
    "EURUSD": {"path": "data/raw/EURUSD_1M_20260215_20260312.csv", "resample": True},
    "VIX": {"path": "data/raw/VIX_1M_20241104_20260313.csv", "resample": True},
}

INT_BOUNDS: dict[str, tuple[int, int]] = {
    "S_detect": (5, 60),
    "scale_start": (2, 30),
    "scale_end": (100, 500),
    "scale_step": (2, 20),
    "min_duration": (1, 20),
    "cooldown_bars": (1, 20),
    "price_gate_lb": (5, 100),
    "vola_range_len": (20, 200),
    "er_period": (5, 60),
    "confirm_count": (1, 5),
    "pivot_drift_lookback": (2, 20),
    "pivot_drift_confirm_bias": (0, 2),
    "edge_window": (3, 60),
}

FLOAT_BOUNDS: dict[str, tuple[float, float]] = {
    "pct_extreme": (0.70, 0.99),
    "min_agreement": (0.10, 0.90),
    "dur_extreme_pct": (0.50, 0.99),
    "vol_surge_thresh": (1.0, 3.0),
    "scale_div_thresh": (0.10, 0.60),
    "slope_thresh": (0.01, 0.50),
    "vola_high_pct": (0.50, 0.99),
    "pivot_drift_thresh": (0.001, 0.050),
    "pivot_drift_gate_mult": (1.0, 10.0),
    "momentum_velocity_thresh": (0.0, 0.05),
    "gjr_vote_thresh": (0.05, 0.50),
    "har_vote_thresh": (0.05, 0.50),
}

# ---------------------------------------------------------------------------
# Seed: Heuristic Structural preset from
# pine/speculatores_v14_presets_gold.pine ("SPX 1D 2026-04-05 Heuristic Structural")
# Keys match the names used in params_from_trial (i.e. "{side}_{name}").
# enqueue_trial primes worker 0's study with this point so trial #0 starts
# at the visually-validated structural-turn configuration.
# ---------------------------------------------------------------------------

SEED_HEURISTIC_STRUCTURAL_HIGH: dict[str, Any] = {
    "high_S_detect": 14,
    "high_scale_start": 24,
    "high_scale_end": 158,
    "high_scale_step": 3,
    "high_min_duration": 18,
    "high_cooldown_bars": 4,
    "high_price_gate_lb": 18,
    "high_vola_range_len": 152,
    "high_er_period": 38,
    "high_confirm_count": 3,
    "high_pivot_drift_lb": 11,
    "high_pivot_drift_confirm_bias": 1,
    "high_pct_extreme": 0.7400,
    "high_min_agreement": 0.6600,
    "high_dur_extreme_pct": 0.6800,
    "high_vol_surge_thresh": 1.2500,
    "high_scale_div_thresh": 0.3500,
    "high_slope_thresh": 0.1600,
    "high_vola_high_pct": 0.6600,
    "high_pivot_drift_thresh": 0.0090,
    "high_pivot_drift_gate_mult": 3.4942,
    "high_momentum_velocity_thresh": 0.0471,
    "high_gjr_vote_thresh": 0.3150,
    "high_har_vote_thresh": 0.4205,
    "high_er_directional": True,
    "high_use_trend": True,
    "high_use_volume": True,
    "high_use_momentum": False,
    "high_use_momentum_velocity": False,
    "high_use_volatility": True,
    "high_use_er_gate": True,
    "high_use_gjr_asym": False,
    "high_use_har_vol": False,
    "high_vola_method": "StdDev",
    "high_momentum_velocity_mode": "Reversal",
    # V15 — passthrough (V14-equivalent)
    "high_use_edge_voting": False,
    "high_edge_window": 5,
}

SEED_HEURISTIC_STRUCTURAL_LOW: dict[str, Any] = {
    "low_S_detect": 20,
    "low_scale_start": 17,
    "low_scale_end": 194,
    "low_scale_step": 7,
    "low_min_duration": 14,
    "low_cooldown_bars": 6,
    "low_price_gate_lb": 72,
    "low_vola_range_len": 35,
    "low_er_period": 52,
    "low_confirm_count": 2,
    "low_pivot_drift_lb": 6,
    "low_pivot_drift_confirm_bias": 0,
    "low_pct_extreme": 0.8200,
    "low_min_agreement": 0.8800,
    "low_dur_extreme_pct": 0.6000,
    "low_vol_surge_thresh": 1.7413,
    "low_scale_div_thresh": 0.4000,
    "low_slope_thresh": 0.4282,
    "low_vola_high_pct": 0.8900,
    "low_pivot_drift_thresh": 0.0500,
    "low_pivot_drift_gate_mult": 3.3731,
    "low_momentum_velocity_thresh": 0.0064,
    "low_gjr_vote_thresh": 0.1911,
    "low_har_vote_thresh": 0.1461,
    "low_er_directional": True,
    "low_use_trend": True,
    "low_use_volume": True,
    "low_use_momentum": False,
    "low_use_momentum_velocity": True,
    "low_use_volatility": True,
    "low_use_er_gate": True,
    "low_use_gjr_asym": False,
    "low_use_har_vol": False,
    "low_vola_method": "ATR",
    "low_momentum_velocity_mode": "Trend",
    # V15 — passthrough (V14-equivalent)
    "low_use_edge_voting": False,
    "low_edge_window": 5,
}

# V15 — Heuristic Structural seed extended with edge-triggered voting.
# Same hyperparameters as SEED_HEURISTIC_STRUCTURAL_{HIGH,LOW} but with
# use_edge_voting=True and edge_window=5 (matching Pine V15 Path A 5k Edge
# preset's default window).
SEED_HEURISTIC_STRUCTURAL_EDGE_HIGH: dict[str, Any] = {
    **SEED_HEURISTIC_STRUCTURAL_HIGH,
    "high_use_edge_voting": True,
    "high_edge_window": 5,
}

SEED_HEURISTIC_STRUCTURAL_EDGE_LOW: dict[str, Any] = {
    **SEED_HEURISTIC_STRUCTURAL_LOW,
    "low_use_edge_voting": True,
    "low_edge_window": 5,
}


BOOL_FIELDS = [
    "er_directional",
    "use_trend",
    "use_volume",
    "use_momentum",
    "use_momentum_velocity",
    "use_volatility",
    "use_er_gate",
    "use_gjr_asym",
    "use_har_vol",
    "use_edge_voting",
]

CATEGORY_FIELDS: dict[str, list[str]] = {
    "vola_method": ["ATR", "StdDev", "Intraday"],
    "momentum_velocity_mode": ["Trend", "Reversal"],
}

PINE_HIGH_PARAMS = [
    ("S_detect_high", "S_detect_high"),
    ("scale_start_high", "scale_start_high"),
    ("scale_end_high", "scale_end_high"),
    ("scale_step_high", "scale_step_high"),
    ("min_duration_high", "min_duration_high"),
    ("cooldown_bars_high", "cooldown_bars_high"),
    ("price_gate_lb_high", "price_gate_lb_high"),
    ("vola_range_len_high", "vola_range_len_high"),
    ("er_period_high", "er_period_high"),
    ("pct_extreme_high", "pct_extreme_high"),
    ("min_agreement_high", "min_agreement_high"),
    ("dur_extreme_pct_high", "dur_extreme_pct_high"),
    ("confirm_count_high", "confirm_count_high"),
    ("vol_surge_thresh_high", "vol_surge_thresh_high"),
    ("scale_div_thresh_high", "scale_div_thresh_high"),
    ("slope_thresh_high", "slope_thresh_high"),
    ("vola_high_pct_high", "vola_high_pct_high"),
    ("pivot_drift_lookback_high", "pivot_drift_lookback_high"),
    ("pivot_drift_thresh_high", "pivot_drift_thresh_high"),
    ("pivot_drift_gate_mult_high", "pivot_drift_gate_mult_high"),
    ("pivot_drift_confirm_bias_high", "pivot_drift_confirm_bias_high"),
    ("momentum_velocity_thresh_high", "momentum_velocity_thresh_high"),
    ("er_directional_high", "er_directional_high"),
    ("use_trend_high", "use_trend_high"),
    ("use_volume_high", "use_volume_high"),
    ("use_momentum_high", "use_momentum_high"),
    ("use_momentum_velocity_high", "use_momentum_velocity_high"),
    ("use_volatility_high", "use_volatility_high"),
    ("use_er_gate_high", "use_er_gate_high"),
    ("momentum_velocity_mode_high", "momentum_velocity_mode_high"),
    ("vola_method_high", "vola_method_high"),
    ("use_edge_voting_high", "use_edge_voting_high"),
    ("edge_window_high", "edge_window_high"),
]

PINE_LOW_PARAMS = [
    ("S_detect_low", "S_detect_low"),
    ("scale_start_low", "scale_start_low"),
    ("scale_end_low", "scale_end_low"),
    ("scale_step_low", "scale_step_low"),
    ("min_duration_low", "min_duration_low"),
    ("cooldown_bars_low", "cooldown_bars_low"),
    ("price_gate_lb_low", "price_gate_lb_low"),
    ("vola_range_len_low", "vola_range_len_low"),
    ("er_period_low", "er_period_low"),
    ("pct_extreme_low", "pct_extreme_low"),
    ("min_agreement_low", "min_agreement_low"),
    ("dur_extreme_pct_low", "dur_extreme_pct_low"),
    ("confirm_count_low", "confirm_count_low"),
    ("vol_surge_thresh_low", "vol_surge_thresh_low"),
    ("scale_div_thresh_low", "scale_div_thresh_low"),
    ("slope_thresh_low", "slope_thresh_low"),
    ("vola_high_pct_low", "vola_high_pct_low"),
    ("pivot_drift_lookback_low", "pivot_drift_lookback_low"),
    ("pivot_drift_thresh_low", "pivot_drift_thresh_low"),
    ("pivot_drift_gate_mult_low", "pivot_drift_gate_mult_low"),
    ("pivot_drift_confirm_bias_low", "pivot_drift_confirm_bias_low"),
    ("momentum_velocity_thresh_low", "momentum_velocity_thresh_low"),
    ("er_directional_low", "er_directional_low"),
    ("use_trend_low", "use_trend_low"),
    ("use_volume_low", "use_volume_low"),
    ("use_momentum_low", "use_momentum_low"),
    ("use_momentum_velocity_low", "use_momentum_velocity_low"),
    ("use_volatility_low", "use_volatility_low"),
    ("use_er_gate_low", "use_er_gate_low"),
    ("momentum_velocity_mode_low", "momentum_velocity_mode_low"),
    ("vola_method_low", "vola_method_low"),
    ("use_edge_voting_low", "use_edge_voting_low"),
    ("edge_window_low", "edge_window_low"),
]


@dataclass(frozen=True)
class RunConfig:
    dataset_path: Path
    storage_path: Path
    results_dir: Path
    trials_per_side: int = DEFAULT_N_TRIALS
    workers_per_side: int = DEFAULT_WORKERS_PER_SIDE
    startup_trials: int = DEFAULT_STARTUP_TRIALS
    seed: int = DEFAULT_SEED
    study_prefix: str = VERSION_SLUG
    cross_asset: bool = True
    stability_trials: int = DEFAULT_STABILITY_TRIALS


def configure_process_environment() -> None:
    """Avoid oversubscribing BLAS/OpenMP threads per worker."""
    for var in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        os.environ.setdefault(var, "1")


def params_to_dict(params: Params) -> dict[str, Any]:
    return {field.name: getattr(params, field.name) for field in dataclasses.fields(params)}


def _pretty_value(value: Any, digits: int = 3) -> Any:
    if isinstance(value, float):
        return round(value, digits)
    return value


def _quantile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    w = pos - lo
    return ordered[lo] * (1.0 - w) + ordered[hi] * w


def _mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _std(values: list[float]) -> float:
    if not values:
        return 0.0
    m = _mean(values)
    return float((sum((v - m) ** 2 for v in values) / len(values)) ** 0.5)


def _params_fields(p: Params) -> dict[str, Any]:
    return {field.name: getattr(p, field.name) for field in dataclasses.fields(p)}


def params_from_trial(trial: optuna.Trial, side: str) -> Params:
    """Sample the full search space for one side without shrinking it."""
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
    use_momentum_velocity = trial.suggest_categorical(
        f"{s}_use_momentum_velocity", [True, False]
    )
    use_volatility = trial.suggest_categorical(f"{s}_use_volatility", [True, False])
    use_er_gate = trial.suggest_categorical(f"{s}_use_er_gate", [True, False])
    use_gjr_asym = trial.suggest_categorical(f"{s}_use_gjr_asym", [True, False])
    use_har_vol = trial.suggest_categorical(f"{s}_use_har_vol", [True, False])

    vola_method = trial.suggest_categorical(
        f"{s}_vola_method", ["ATR", "StdDev", "Intraday"]
    )
    momentum_velocity_mode = trial.suggest_categorical(
        f"{s}_momentum_velocity_mode", ["Trend", "Reversal"]
    )

    # V15 — edge-triggered voting search dimensions.
    use_edge_voting = trial.suggest_categorical(
        f"{s}_use_edge_voting", [True, False]
    )
    edge_window = trial.suggest_int(f"{s}_edge_window", 3, 60)

    kwargs_high = dict(
        S_detect_high=S_detect,
        scale_start_high=scale_start,
        scale_end_high=scale_end,
        scale_step_high=scale_step,
        min_duration_high=min_duration,
        cooldown_bars_high=cooldown_bars,
        price_gate_lb_high=price_gate_lb,
        vola_range_len_high=vola_range_len,
        er_period_high=er_period,
        pct_extreme_high=pct_extreme,
        min_agreement_high=min_agreement,
        dur_extreme_pct_high=dur_extreme_pct,
        confirm_count_high=confirm_count,
        vol_surge_thresh_high=vol_surge_thresh,
        scale_div_thresh_high=scale_div_thresh,
        slope_thresh_high=slope_thresh,
        vola_high_pct_high=vola_high_pct,
        pivot_drift_lookback_high=pivot_drift_lb,
        pivot_drift_thresh_high=pivot_drift_thresh,
        pivot_drift_gate_mult_high=pivot_drift_gate_mult,
        pivot_drift_confirm_bias_high=pivot_drift_confirm_bias,
        momentum_velocity_thresh_high=momentum_velocity_thresh,
        er_directional_high=er_directional,
        use_trend_high=use_trend,
        use_volume_high=use_volume,
        use_momentum_high=use_momentum,
        use_momentum_velocity_high=use_momentum_velocity,
        use_volatility_high=use_volatility,
        use_er_gate_high=use_er_gate,
        use_gjr_asym_high=use_gjr_asym,
        use_har_vol_high=use_har_vol,
        gjr_vote_thresh_high=gjr_vote_thresh,
        har_vote_thresh_high=har_vote_thresh,
        vola_method_high=vola_method,
        momentum_velocity_mode_high=momentum_velocity_mode,
        use_edge_voting_high=use_edge_voting,
        edge_window_high=edge_window,
    )
    kwargs_low = dict(
        S_detect_low=S_detect,
        scale_start_low=scale_start,
        scale_end_low=scale_end,
        scale_step_low=scale_step,
        min_duration_low=min_duration,
        cooldown_bars_low=cooldown_bars,
        price_gate_lb_low=price_gate_lb,
        vola_range_len_low=vola_range_len,
        er_period_low=er_period,
        pct_extreme_low=pct_extreme,
        min_agreement_low=min_agreement,
        dur_extreme_pct_low=dur_extreme_pct,
        confirm_count_low=confirm_count,
        vol_surge_thresh_low=vol_surge_thresh,
        scale_div_thresh_low=scale_div_thresh,
        slope_thresh_low=slope_thresh,
        vola_high_pct_low=vola_high_pct,
        pivot_drift_lookback_low=pivot_drift_lb,
        pivot_drift_thresh_low=pivot_drift_thresh,
        pivot_drift_gate_mult_low=pivot_drift_gate_mult,
        pivot_drift_confirm_bias_low=pivot_drift_confirm_bias,
        momentum_velocity_thresh_low=momentum_velocity_thresh,
        er_directional_low=er_directional,
        use_trend_low=use_trend,
        use_volume_low=use_volume,
        use_momentum_low=use_momentum,
        use_momentum_velocity_low=use_momentum_velocity,
        use_volatility_low=use_volatility,
        use_er_gate_low=use_er_gate,
        use_gjr_asym_low=use_gjr_asym,
        use_har_vol_low=use_har_vol,
        gjr_vote_thresh_low=gjr_vote_thresh,
        har_vote_thresh_low=har_vote_thresh,
        vola_method_low=vola_method,
        momentum_velocity_mode_low=momentum_velocity_mode,
        use_edge_voting_low=use_edge_voting,
        edge_window_low=edge_window,
    )

    base_dict = _params_fields(Params())
    overrides = kwargs_high if side == "high" else kwargs_low
    return Params(**{**base_dict, **overrides})


def make_storage(storage_path: Path) -> JournalStorage:
    storage_path.parent.mkdir(parents=True, exist_ok=True)
    # Google Drive mounts in Colab do not support Optuna's default symlink lock.
    # Use the file-open lock everywhere so the same runner works on Windows,
    # Linux, and Drive-backed filesystems.
    lock_obj = JournalFileOpenLock(str(storage_path))
    return JournalStorage(JournalFileBackend(str(storage_path), lock_obj=lock_obj))


def make_sampler(seed: int) -> TPESampler:
    return TPESampler(
        seed=seed,
        n_startup_trials=DEFAULT_STARTUP_TRIALS,
        multivariate=True,
        group=True,
        constant_liar=True,
    )


def make_study(
    storage: JournalStorage,
    study_name: str,
    seed: int,
    startup_trials: int,
) -> optuna.Study:
    sampler = TPESampler(
        seed=seed,
        n_startup_trials=startup_trials,
        multivariate=True,
        group=True,
        constant_liar=True,
    )
    return optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction="maximize",
        sampler=sampler,
        pruner=MedianPruner(n_startup_trials=20, n_warmup_steps=2),
        load_if_exists=True,
    )


def dataset_version_from_path(path: Path) -> str:
    return path.stem


def _worker_entry(
    side: str,
    config: RunConfig,
    worker_index: int,
) -> None:
    configure_process_environment()
    logging.basicConfig(
        level=logging.INFO,
        format=f"%(asctime)s [{side}-w{worker_index}] %(levelname)s %(name)s: %(message)s",
    )
    df = load_data(config.dataset_path)
    storage = make_storage(config.storage_path)
    study_name = f"{config.study_prefix}_{dataset_version_from_path(config.dataset_path)}_{side}"
    study = make_study(
        storage=storage,
        study_name=study_name,
        seed=config.seed + worker_index,
        startup_trials=config.startup_trials,
    )
    objective = build_optuna_objective(df, params_from_trial, side)
    done_before = len([t for t in study.trials if t.state in TRIAL_STATES_DONE])
    logger.info("Worker %d sees %d finished trials for %s", worker_index, done_before, side)
    if worker_index == 0 and done_before == 0:
        seed_params = (
            SEED_HEURISTIC_STRUCTURAL_HIGH if side == "high"
            else SEED_HEURISTIC_STRUCTURAL_LOW
        )
        study.enqueue_trial(seed_params)
        logger.info(
            "Worker 0 seeded %s study with Heuristic Structural preset", side,
        )
        # V15 — also seed the Edge variant so TPE has both anchors.
        edge_seed = (
            SEED_HEURISTIC_STRUCTURAL_EDGE_HIGH if side == "high"
            else SEED_HEURISTIC_STRUCTURAL_EDGE_LOW
        )
        study.enqueue_trial(edge_seed)
        logger.info(
            "Worker 0 seeded %s study with Heuristic Structural Edge preset",
            side,
        )
    stop_cb = MaxTrialsCallback(config.trials_per_side, states=TRIAL_STATES_DONE)
    study.optimize(
        objective,
        n_trials=config.trials_per_side,
        callbacks=[stop_cb],
        gc_after_trial=True,
        show_progress_bar=False,
    )


def run_parallel_studies(config: RunConfig) -> dict[str, optuna.Study]:
    """Run high and low studies with separate worker pools."""
    configure_process_environment()
    ctx = mp.get_context("spawn")
    processes: list[mp.Process] = []
    for side in ("high", "low"):
        for worker_idx in range(config.workers_per_side):
            proc = ctx.Process(
                target=_worker_entry,
                args=(side, config, worker_idx),
                name=f"{side}-worker-{worker_idx}",
            )
            proc.start()
            processes.append(proc)
    failures: list[str] = []
    for proc in processes:
        proc.join()
        if proc.exitcode != 0:
            failures.append(f"{proc.name} exited with code {proc.exitcode}")
    if failures:
        raise RuntimeError("; ".join(failures))

    storage = make_storage(config.storage_path)
    studies: dict[str, optuna.Study] = {}
    dataset_slug = dataset_version_from_path(config.dataset_path)
    for side in ("high", "low"):
        study_name = f"{config.study_prefix}_{dataset_slug}_{side}"
        studies[side] = optuna.load_study(study_name=study_name, storage=storage)
    return studies


def run_default_detector(df: pd.DataFrame) -> dict[str, Any]:
    params = Params()
    det = SpeculatorDetector(df.reset_index(drop=True), params).run()
    return {
        "high_signals": int(det["signal_high"].sum()),
        "low_signals": int(det["signal_low"].sum()),
        "bars": len(df),
    }


def evaluate_folds_for_params(
    df: pd.DataFrame,
    params: Params,
    side: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for idx, (df_is, df_oos) in enumerate(walk_forward_folds(df)):
        df_is_r = df_is.reset_index(drop=True)
        df_oos_r = df_oos.reset_index(drop=True)
        det_is = SpeculatorDetector(
            df_is_r, params, build_detector_artifacts(df_is_r)
        ).run()
        det_oos = SpeculatorDetector(
            df_oos_r, params, build_detector_artifacts(df_oos_r)
        ).run()
        sig_key = f"signal_{side}"
        is_score = compute_side_score(df_is_r, det_is[sig_key], side)
        oos_score = compute_side_score(df_oos_r, det_oos[sig_key], side)
        rows.append(
            {
                "fold": idx + 1,
                "oos_period": f"{df_oos.index[0].strftime('%Y-%m-%d')} – {df_oos.index[-1].strftime('%Y-%m-%d')}",
                "is_bars": len(df_is_r),
                "oos_bars": len(df_oos_r),
                "is_signals": int(det_is[sig_key].sum()),
                "oos_signals": int(det_oos[sig_key].sum()),
                "is_score": round(float(is_score), 4),
                "oos_score": round(float(oos_score), 4),
                "gap": round(float(is_score - oos_score), 4),
            }
        )
    return pd.DataFrame(rows)


def evaluate_holdout_for_params(
    df: pd.DataFrame,
    params: Params,
    side: str,
) -> dict[str, Any]:
    df_is, df_oos = temporal_split(df)
    df_is_r = df_is.reset_index(drop=True)
    df_oos_r = df_oos.reset_index(drop=True)
    det_is = SpeculatorDetector(df_is_r, params, build_detector_artifacts(df_is_r)).run()
    det_oos = SpeculatorDetector(df_oos_r, params, build_detector_artifacts(df_oos_r)).run()
    sig_key = f"signal_{side}"
    is_score = compute_side_score(df_is_r, det_is[sig_key], side)
    oos_score = compute_side_score(df_oos_r, det_oos[sig_key], side)
    return {
        "is_bars": len(df_is_r),
        "oos_bars": len(df_oos_r),
        "is_signals": int(det_is[sig_key].sum()),
        "oos_signals": int(det_oos[sig_key].sum()),
        "is_score": round(float(is_score), 4),
        "oos_score": round(float(oos_score), 4),
        "gap": round(float(is_score - oos_score), 4),
    }


def _mutate_local_param(
    rng: random.Random,
    value: Any,
    field_name: str,
) -> Any:
    if field_name in INT_BOUNDS:
        low, high = INT_BOUNDS[field_name]
        radius = max(1, int(round((high - low) * 0.1)))
        local_low = max(low, int(value) - radius)
        local_high = min(high, int(value) + radius)
        return rng.randint(local_low, local_high)
    if field_name in FLOAT_BOUNDS:
        low, high = FLOAT_BOUNDS[field_name]
        radius = (high - low) * 0.1
        local_low = max(low, float(value) - radius)
        local_high = min(high, float(value) + radius)
        return rng.uniform(local_low, local_high)
    if field_name in BOOL_FIELDS:
        return value if rng.random() < 0.85 else (not bool(value))
    if field_name in CATEGORY_FIELDS:
        choices = CATEGORY_FIELDS[field_name]
        if rng.random() < 0.85:
            return value
        alternatives = [choice for choice in choices if choice != value]
        return rng.choice(alternatives) if alternatives else value
    return value


def _sample_local_params(
    params: Params,
    side: str,
    rng: random.Random,
) -> Params:
    base_dict = _params_fields(Params())
    current = _params_fields(params)
    overrides: dict[str, Any] = {}
    for field_name in list(INT_BOUNDS) + list(FLOAT_BOUNDS) + BOOL_FIELDS + list(CATEGORY_FIELDS):
        side_field = f"{field_name}_{side}"
        overrides[side_field] = _mutate_local_param(rng, current[side_field], field_name)
    return Params(**{**base_dict, **overrides})


def summarize_stability(
    params: Params,
    side: str,
    best_value: float,
    prepared_folds: list[Any],
    trials: int,
    seed: int,
) -> tuple[dict[str, Any], pd.DataFrame]:
    rng = random.Random(seed)
    rows: list[dict[str, Any]] = []
    scores: list[float] = []
    for idx in range(trials):
        sampled = _sample_local_params(params, side, rng)
        fold_scores = evaluate_params_on_prepared_folds(sampled, side, prepared_folds)
        score = float(_mean(fold_scores))
        scores.append(score)
        rows.append(
            {
                "trial": idx + 1,
                "score": score,
                "delta_vs_best": score - best_value,
                "pct_of_best": (score / best_value) if best_value > 0 else 0.0,
            }
        )

    mean_score = _mean(scores)
    if best_value <= 0:
        share_ge_90 = None
        mean_ratio = None
        verdict = "no positive winner"
    else:
        share_ge_90 = sum(score >= (0.9 * best_value) for score in scores) / len(scores)
        mean_ratio = mean_score / best_value
        if share_ge_90 >= 0.2 and mean_ratio >= 0.6:
            verdict = "stable basin"
        elif share_ge_90 >= 0.08 and mean_ratio >= 0.35:
            verdict = "moderately stable"
        else:
            verdict = "likely outlier"

    summary = {
        "trials": trials,
        "best_score": round(float(best_value), 6),
        "local_mean": round(mean_score, 6),
        "local_p75": round(_quantile(scores, 0.75) or 0.0, 6),
        "local_std": round(_std(scores), 6),
        "share_positive": round(sum(score > 0 for score in scores) / len(scores), 3) if scores else 0.0,
        "share_ge_90pct_best": round(share_ge_90, 3) if share_ge_90 is not None else None,
        "mean_vs_best": round(mean_ratio, 3) if mean_ratio is not None else None,
        "max_local": round(max(scores), 6) if scores else 0.0,
        "verdict": verdict,
    }
    return summary, pd.DataFrame(rows)


def score_cross_assets(
    params_high: Params,
    params_low: Params,
    instruments: dict[str, dict[str, Any]] | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for name, info in (instruments or INSTRUMENTS).items():
        path = Path(info["path"])
        if not path.exists():
            rows.append({"instrument": name, "error": f"missing file: {path}"})
            continue
        try:
            df = load_cross_asset(path, resample_to_1d=bool(info["resample"]))
            df_r = df.reset_index(drop=True)
            det_h = SpeculatorDetector(
                df_r, params_high, build_detector_artifacts(df_r)
            ).run()
            det_l = SpeculatorDetector(
                df_r, params_low, build_detector_artifacts(df_r)
            ).run()
            rows.append(
                {
                    "instrument": name,
                    "bars": len(df_r),
                    "note": "resampled 1M→1D" if info["resample"] else "native 1D",
                    "high_score": round(
                        float(compute_side_score(df_r, det_h["signal_high"], "high")), 4
                    ),
                    "low_score": round(
                        float(compute_side_score(df_r, det_l["signal_low"], "low")), 4
                    ),
                }
            )
        except Exception as exc:  # pragma: no cover - export should continue
            rows.append({"instrument": name, "error": str(exc)})
    return pd.DataFrame(rows)


def format_pine_value(value: Any) -> str:
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, float):
        return f"{value:.4f}"
    if isinstance(value, str):
        return f"\"{value}\""
    return str(value)


def build_pine_block(
    study_high: optuna.Study,
    study_low: optuna.Study,
    params_high: Params,
    params_low: Params,
) -> str:
    best_dict_h = params_to_dict(params_high)
    best_dict_l = params_to_dict(params_low)
    lines = [
        "// ============================================================",
        "// Optimizer output — paste into is_gold_next branch",
        f"// HIGH: trial {study_high.best_trial.number}, score {study_high.best_trial.value:.6f}",
        f"// LOW:  trial {study_low.best_trial.number}, score {study_low.best_trial.value:.6f}",
        "// ============================================================",
        "",
        "// HIGH side",
    ]
    for py_name, pine_name in PINE_HIGH_PARAMS:
        lines.append(f"{pine_name} = {format_pine_value(best_dict_h.get(py_name, '???'))}")
    lines += ["", "// LOW side"]
    for py_name, pine_name in PINE_LOW_PARAMS:
        lines.append(f"{pine_name} = {format_pine_value(best_dict_l.get(py_name, '???'))}")
    lines += [
        "",
        "// ── GJR/HAR parameters (NOT yet in speculatores_v14_presets_gold.pine) ──",
        "// To activate: add GJR/HAR vote computation and update max_votes in Pine.",
    ]
    gjr_har_fields = [
        ("use_gjr_asym_high", best_dict_h),
        ("gjr_vote_thresh_high", best_dict_h),
        ("use_har_vol_high", best_dict_h),
        ("har_vote_thresh_high", best_dict_h),
        ("use_gjr_asym_low", best_dict_l),
        ("gjr_vote_thresh_low", best_dict_l),
        ("use_har_vol_low", best_dict_l),
        ("har_vote_thresh_low", best_dict_l),
    ]
    for field, source in gjr_har_fields:
        lines.append(f"// {field} = {format_pine_value(source.get(field, '???'))}")
    if any(
        best_dict_h.get(flag, False) or best_dict_l.get(flag, False)
        for flag in ("use_gjr_asym_high", "use_gjr_asym_low", "use_har_vol_high", "use_har_vol_low")
    ):
        lines.append("// ⚠️  At least one GJR/HAR flag is active — Pine code additions required.")
    return "\n".join(lines)


def study_summary(study: optuna.Study) -> dict[str, Any]:
    states = {state.name.lower(): 0 for state in TrialState}
    for trial in study.trials:
        states[trial.state.name.lower()] += 1
    return {
        "study_name": study.study_name,
        "best_trial": study.best_trial.number,
        "best_value": round(float(study.best_trial.value), 6),
        "n_trials": len(study.trials),
        "states": states,
    }


def _df_to_markdown(df: pd.DataFrame) -> str:
    if df.empty:
        return "_empty_"
    render_df = df.copy()
    render_df = render_df.map(_pretty_value)
    render_df = render_df.fillna("").astype(str)
    headers = list(render_df.columns)
    rows = render_df.values.tolist()
    widths = [len(str(header)) for header in headers]
    for row in rows:
        for idx, cell in enumerate(row):
            widths[idx] = max(widths[idx], len(cell))
    header_line = "| " + " | ".join(
        header.ljust(widths[idx]) for idx, header in enumerate(headers)
    ) + " |"
    separator_line = "| " + " | ".join("-" * widths[idx] for idx in range(len(headers))) + " |"
    row_lines = [
        "| " + " | ".join(cell.ljust(widths[idx]) for idx, cell in enumerate(row)) + " |"
        for row in rows
    ]
    return "\n".join([header_line, separator_line, *row_lines])


def render_report(
    config: RunConfig,
    study_high: optuna.Study,
    study_low: optuna.Study,
    df_primary: pd.DataFrame,
    validation_info: dict[str, Any],
    default_run: dict[str, Any],
    high_folds: pd.DataFrame,
    low_folds: pd.DataFrame,
    high_holdout: dict[str, Any],
    low_holdout: dict[str, Any],
    high_stability: dict[str, Any],
    low_stability: dict[str, Any],
    cross_asset_df: pd.DataFrame,
    pine_block: str,
    parity_df: pd.DataFrame,
    parity_path: Path | None,
    report_timestamp: datetime,
) -> str:
    params_high = params_from_trial(study_high.best_trial, "high")
    params_low = params_from_trial(study_low.best_trial, "low")
    best_high_df = pd.DataFrame.from_dict(
        study_high.best_trial.params, orient="index", columns=["value"]
    ).reset_index(names="param")
    best_high_df["param"] = best_high_df["param"].str.replace("^high_", "", regex=True)
    best_high_df["value"] = best_high_df["value"].map(_pretty_value)
    best_low_df = pd.DataFrame.from_dict(
        study_low.best_trial.params, orient="index", columns=["value"]
    ).reset_index(names="param")
    best_low_df["param"] = best_low_df["param"].str.replace("^low_", "", regex=True)
    best_low_df["value"] = best_low_df["value"].map(_pretty_value)
    dataset_version = dataset_version_from_path(config.dataset_path)
    lines = [
        f"# {VERSION} Run Report",
        "",
        f"- Dataset: `{dataset_version}`",
        f"- Generated: `{report_timestamp.strftime('%Y-%m-%d %H:%M:%S')}`",
        f"- Storage: `{config.storage_path}`",
        f"- Trials per side target: `{config.trials_per_side}`",
        f"- Workers per side: `{config.workers_per_side}`",
        f"- Stability probe trials per side: `{config.stability_trials}`",
        f"- Bars in primary dataset: `{len(df_primary)}`",
        "",
        "## Cell 2 — Data",
        "",
        f"- Source path: `{config.dataset_path}`",
        f"- Index range: `{df_primary.index[0]}` → `{df_primary.index[-1]}`",
        f"- Validation mode: `{validation_info['mode']}`",
        f"- Walk-forward folds: `{validation_info['n_folds']}`",
        f"- IS bars per fold: `{validation_info['is_bars']}`",
        f"- OOS bars per fold: `{validation_info['oos_bars']}`",
        f"- Step bars between folds: `{validation_info['step_bars']}`",
        f"- Holdout bars: `{validation_info['holdout_bars']}`",
        f"- Embargo bars: `{validation_info['embargo_bars']}`",
        "",
        "## Cell 3 — Verify Default Params",
        "",
        f"- HIGH signals: `{default_run['high_signals']}`",
        f"- LOW signals: `{default_run['low_signals']}`",
        f"- Bars: `{default_run['bars']}`",
        "",
        "## Cell 3.3 — Pine/Python Parity",
        "",
        (
            f"- TradingView export used: `{parity_path}`"
            if parity_path is not None
            else "- TradingView export used: `_none found_`"
        ),
        "",
        _df_to_markdown(parity_df),
        "",
        "## Cell 4 — Optimization Summary",
        "",
        "### HIGH Study",
        "",
        "```json",
        json.dumps(study_summary(study_high), indent=2),
        "```",
        "",
        "### LOW Study",
        "",
        "```json",
        json.dumps(study_summary(study_low), indent=2),
        "```",
        "",
        "## Cell 5.1 — Best Params",
        "",
        "### HIGH",
        "",
        _df_to_markdown(best_high_df),
        "",
        "### LOW",
        "",
        _df_to_markdown(best_low_df),
        "",
        "## Cell 5.2 — Walk-Forward Fold Breakdown",
        "",
        "### HIGH",
        "",
        _df_to_markdown(high_folds),
        "",
        "### LOW",
        "",
        _df_to_markdown(low_folds),
        "",
        "## Cell 5.3 — Final Holdout (Temporal Split)",
        "",
        "### HIGH",
        "",
        "```json",
        json.dumps(high_holdout, indent=2),
        "```",
        "",
        "### LOW",
        "",
        "```json",
        json.dumps(low_holdout, indent=2),
        "```",
        "",
        "## Cell 5.4 — Local Stability Probe",
        "",
        "Neighborhood probe around each side's best params using the same walk-forward objective.",
        "",
        "### HIGH",
        "",
        "```json",
        json.dumps(high_stability, indent=2),
        "```",
        "",
        "### LOW",
        "",
        "```json",
        json.dumps(low_stability, indent=2),
        "```",
        "",
        "## Cell 6 — Cross-Asset Generalization",
        "",
        _df_to_markdown(cross_asset_df),
        "",
        "## Cell 7.1 — Best Params JSON",
        "",
        "```json",
        json.dumps(
            {
                "high": params_to_dict(params_high),
                "low": params_to_dict(params_low),
            },
            indent=2,
        ),
        "```",
        "",
        "## Cell 7.2 — Pine Export Block",
        "",
        "```text",
        pine_block,
        "```",
        "",
    ]
    return "\n".join(lines)


def export_report(config: RunConfig, report_text: str, report_timestamp: datetime) -> Path:
    config.results_dir.mkdir(parents=True, exist_ok=True)
    dataset_version = dataset_version_from_path(config.dataset_path)
    timestamp_slug = report_timestamp.strftime("%Y%m%d_%H%M%S")
    output_path = config.results_dir / f"{dataset_version}__{timestamp_slug}__{VERSION_SLUG}.md"
    output_path.write_text(report_text, encoding="utf-8")
    return output_path


def run_full_pipeline(config: RunConfig) -> Path:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    df_primary = load_data(config.dataset_path)
    validation_info = describe_validation_scheme(df_primary)
    default_run = run_default_detector(df_primary)
    studies = run_parallel_studies(config)
    study_high = studies["high"]
    study_low = studies["low"]
    params_high = params_from_trial(study_high.best_trial, "high")
    params_low = params_from_trial(study_low.best_trial, "low")
    prepared_folds = prepare_walk_forward_folds(df_primary)
    high_folds = evaluate_folds_for_params(df_primary, params_high, "high")
    low_folds = evaluate_folds_for_params(df_primary, params_low, "low")
    high_holdout = evaluate_holdout_for_params(df_primary, params_high, "high")
    low_holdout = evaluate_holdout_for_params(df_primary, params_low, "low")
    high_stability, _ = summarize_stability(
        params_high,
        "high",
        float(study_high.best_trial.value),
        prepared_folds,
        config.stability_trials,
        config.seed + 10_001,
    )
    low_stability, _ = summarize_stability(
        params_low,
        "low",
        float(study_low.best_trial.value),
        prepared_folds,
        config.stability_trials,
        config.seed + 20_001,
    )
    cross_asset_df = (
        score_cross_assets(params_high, params_low) if config.cross_asset else pd.DataFrame()
    )
    pine_block = build_pine_block(study_high, study_low, params_high, params_low)
    parity_path = find_matching_enriched_export(config.dataset_path)
    if parity_path is not None:
        parity_df, _ = compare_python_to_tv(config.dataset_path, parity_path, Params())
    else:
        parity_df = pd.DataFrame([{"note": "No matching enriched TradingView export found"}])
    now = datetime.now()
    report_text = render_report(
        config=config,
        study_high=study_high,
        study_low=study_low,
        df_primary=df_primary,
        validation_info=validation_info,
        default_run=default_run,
        high_folds=high_folds,
        low_folds=low_folds,
        high_holdout=high_holdout,
        low_holdout=low_holdout,
        high_stability=high_stability,
        low_stability=low_stability,
        cross_asset_df=cross_asset_df,
        pine_block=pine_block,
        parity_df=parity_df,
        parity_path=parity_path,
        report_timestamp=now,
    )
    return export_report(config, report_text, now)
