"""Speculatores 14.5 optimization pipeline.

Script-friendly orchestration for Optuna studies, Colab-safe multiprocessing,
and one-file Markdown exports that capture the equivalent of notebook outputs.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import multiprocessing as mp
import os
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
    FOLD_DEFINITIONS,
    build_optuna_objective,
    load_cross_asset,
    load_data,
    temporal_split,
    walk_forward_folds,
)

logger = logging.getLogger(__name__)

VERSION = "Speculatores 14.5"
VERSION_SLUG = "speculatores_14_5"
DEFAULT_RESULTS_DIR = Path("results")
DEFAULT_STORAGE_FILE = Path("temp") / f"{VERSION_SLUG}.journal"
DEFAULT_N_TRIALS = 500
DEFAULT_STARTUP_TRIALS = 80
DEFAULT_WORKERS_PER_SIDE = 2
DEFAULT_SEED = 42

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
    )

    base_dict = _params_fields(Params())
    overrides = kwargs_high if side == "high" else kwargs_low
    return Params(**{**base_dict, **overrides})


def make_storage(storage_path: Path) -> JournalStorage:
    storage_path.parent.mkdir(parents=True, exist_ok=True)
    lock_obj = JournalFileOpenLock(str(storage_path)) if os.name == "nt" else None
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
                "oos_period": f"{FOLD_DEFINITIONS[idx][2][:7]} – {FOLD_DEFINITIONS[idx][3][:7]}",
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
    render_df = df.fillna("").astype(str)
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
    default_run: dict[str, Any],
    high_folds: pd.DataFrame,
    low_folds: pd.DataFrame,
    high_holdout: dict[str, Any],
    low_holdout: dict[str, Any],
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
    best_low_df = pd.DataFrame.from_dict(
        study_low.best_trial.params, orient="index", columns=["value"]
    ).reset_index(names="param")
    best_low_df["param"] = best_low_df["param"].str.replace("^low_", "", regex=True)
    dataset_version = dataset_version_from_path(config.dataset_path)
    lines = [
        f"# {VERSION} Run Report",
        "",
        f"- Dataset: `{dataset_version}`",
        f"- Generated: `{report_timestamp.strftime('%Y-%m-%d %H:%M:%S')}`",
        f"- Storage: `{config.storage_path}`",
        f"- Trials per side target: `{config.trials_per_side}`",
        f"- Workers per side: `{config.workers_per_side}`",
        f"- Bars in primary dataset: `{len(df_primary)}`",
        "",
        "## Cell 2 — Data",
        "",
        f"- Source path: `{config.dataset_path}`",
        f"- Index range: `{df_primary.index[0]}` → `{df_primary.index[-1]}`",
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
    default_run = run_default_detector(df_primary)
    studies = run_parallel_studies(config)
    study_high = studies["high"]
    study_low = studies["low"]
    params_high = params_from_trial(study_high.best_trial, "high")
    params_low = params_from_trial(study_low.best_trial, "low")
    high_folds = evaluate_folds_for_params(df_primary, params_high, "high")
    low_folds = evaluate_folds_for_params(df_primary, params_low, "low")
    high_holdout = evaluate_holdout_for_params(df_primary, params_high, "high")
    low_holdout = evaluate_holdout_for_params(df_primary, params_low, "low")
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
        default_run=default_run,
        high_folds=high_folds,
        low_folds=low_folds,
        high_holdout=high_holdout,
        low_holdout=low_holdout,
        cross_asset_df=cross_asset_df,
        pine_block=pine_block,
        parity_df=parity_df,
        parity_path=parity_path,
        report_timestamp=now,
    )
    return export_report(config, report_text, now)
