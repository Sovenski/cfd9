"""Execute the v16 notebook's OWN cell logic locally (resolve + launch cells),
against the local repo data/raw, N_TRIALS=3, to prove the notebook runs end-to-end
(make_storage persistence + seed enqueue + resume + pooled objective).
"""
from __future__ import annotations
import sys, tempfile
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# --- settings cell (DATA_DIR -> real raw_v16; RUNS_DIR -> temp) ---
DATA_DIR = ROOT / "data" / "raw_v16"
RUNS_DIR = Path(tempfile.mkdtemp(prefix="v16runs_"))
SELECTED_GROUPS = ["INDICES", "COMMODITIES", "WORLD_ETF", "FX"]
SELECTED_TIMEFRAMES = ["1D"]
VOLUME_POLICY = "price_only"
N_TRIALS = 2
SEED = 42
RUN_SLUG = f"v16val_{datetime.now():%Y%m%d_%H%M%S}"

# --- resolve cell ---
from src.universe import resolve_streams
STREAMS = resolve_streams(SELECTED_GROUPS, SELECTED_TIMEFRAMES, data_dir=str(DATA_DIR))
print(f"resolved {len(STREAMS)} streams:", [s.stream_id for s in STREAMS])
assert STREAMS, "no streams resolved locally"

# --- launch cell (verbatim logic) ---
import optuna
from src.scoring import add_pivot_labels
from src.monitor145 import make_storage
from src.pooled_validation import (
    StreamData, build_calendar_folds, build_pooled_optuna_objective,
    load_stream_frame, apply_volume_policy,
)
from src.speculatores145 import (
    params_from_trial, SEED_HEURISTIC_STRUCTURAL_HIGH, SEED_HEURISTIC_STRUCTURAL_LOW,
)
optuna.logging.set_verbosity(optuna.logging.WARNING)
_TF_SECONDS = {"1D": 86400.0, "1W": 604800.0, "60": 3600.0, "240": 14400.0}

stream_datas = []
for s in STREAMS:
    df = load_stream_frame(s.path)
    df, keep = apply_volume_policy(df, policy=VOLUME_POLICY)
    if not keep:
        print(f"drop {s.stream_id}"); continue
    add_pivot_labels(df)
    stream_datas.append(StreamData(stream=s, df=df, bar_seconds=_TF_SECONDS[s.timeframe]))

folds = build_calendar_folds(stream_datas)
streams = [sd.stream for sd in stream_datas]
print(f"{len(folds)} folds; streams/fold = {[len(f) for f in folds]}")
assert folds

_SEEDS = {"high": SEED_HEURISTIC_STRUCTURAL_HIGH, "low": SEED_HEURISTIC_STRUCTURAL_LOW}
for side in ("high", "low"):
    storage = make_storage(RUNS_DIR / f"{RUN_SLUG}_{side}.journal")
    study = optuna.create_study(
        study_name=f"spec_v16_{side}", direction="maximize",
        storage=storage, load_if_exists=True,
        sampler=optuna.samplers.TPESampler(multivariate=True, group=True, seed=SEED),
        pruner=optuna.pruners.MedianPruner(),
    )
    if not study.trials:
        study.enqueue_trial(_SEEDS[side])
    done = len([t for t in study.trials if t.state.is_finished()])
    remaining = max(0, N_TRIALS - done)
    print(f"[{side}] resuming at {done}/{N_TRIALS}; running {remaining} more ...")
    study.optimize(build_pooled_optuna_objective(folds, streams, params_from_trial, side),
                   n_trials=remaining)
    print(f"[{side}] best LCB = {study.best_value:.5f}  ({len(study.trials)} trials in journal)")

# --- resume check: re-open the LOW study, confirm trials persisted ---
storage2 = make_storage(RUNS_DIR / f"{RUN_SLUG}_low.journal")
study2 = optuna.create_study(study_name="spec_v16_low", direction="maximize",
                             storage=storage2, load_if_exists=True)
print(f"RESUME CHECK: reopened LOW study has {len(study2.trials)} trials persisted")
assert len(study2.trials) >= N_TRIALS, "persistence/resume failed"
print("\nv16 notebook-cell validation PASSED")
