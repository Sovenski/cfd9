"""Append Run 5 pooled-launch cells to optimize.ipynb.

Mirrors the pattern in temp/patch_notebook_run3.py:
  1. Load notebook JSON.
  2. Append 3 new code cells (Task 9 Step 1, Task 9 Step 2, Task 10 Step 2).
  3. Write back.

Run from the repo root:
    python temp/patch_notebook_run5.py
"""
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
NB_PATH = REPO_ROOT / "optimize.ipynb"

# ---------------------------------------------------------------------------
# Cell A — Task 9 Step 1: universe/timeframe/volume-policy selector
# ---------------------------------------------------------------------------
CELL_A_SOURCE = """\
# === Run 5 — multi-asset pool selection ===
from src.universe import UNIVERSE, TIMEFRAMES, resolve_streams

# Choose any subset of these groups + timeframes (edit before launching):
SELECTED_GROUPS = ["INDICES_GLOBAL", "COMMODITIES"]   # see UNIVERSE.keys()
SELECTED_TIMEFRAMES = ["1D"]                            # subset of TIMEFRAMES
VOLUME_POLICY = "price_only"                            # price_only | volume_required | mixed

print("available groups:", list(UNIVERSE))
print("available timeframes:", TIMEFRAMES)
STREAMS = resolve_streams(SELECTED_GROUPS, SELECTED_TIMEFRAMES, data_dir="data/raw")
print(f"resolved {len(STREAMS)} streams:",
      [s.stream_id for s in STREAMS])
"""

# ---------------------------------------------------------------------------
# Cell B — Task 9 Step 2: pooled folds build + seeded launch (both sides)
# Enhancement: enqueue heuristic seed before study.optimize()
# ---------------------------------------------------------------------------
CELL_B_SOURCE = """\
# === Run 5 — build pooled folds + launch (both sides) ===
import optuna
from src.scoring import add_pivot_labels
from src.pooled_validation import (
    StreamData, build_calendar_folds, build_pooled_optuna_objective,
    load_stream_frame, apply_volume_policy,
)
from src.speculatores145 import (
    params_from_trial,
    SEED_HEURISTIC_STRUCTURAL_HIGH,
    SEED_HEURISTIC_STRUCTURAL_LOW,
)

_TF_SECONDS = {"1D": 86400.0, "1W": 604800.0, "60": 3600.0, "240": 14400.0}

stream_datas = []
for s in STREAMS:
    df = load_stream_frame(s.path)
    df, keep = apply_volume_policy(df, policy=VOLUME_POLICY)
    if not keep:
        print(f"drop {s.stream_id}: volume policy excluded it"); continue
    add_pivot_labels(df)
    stream_datas.append(StreamData(stream=s, df=df,
                                   bar_seconds=_TF_SECONDS[s.timeframe]))

folds = build_calendar_folds(stream_datas)
print(f"{len(folds)} calendar folds; streams/fold = {[len(f) for f in folds]}")
streams = [sd.stream for sd in stream_datas]

N_TRIALS = 1000  # per side
for side in ("high", "low"):
    objective = build_pooled_optuna_objective(folds, streams, params_from_trial, side)
    study = optuna.create_study(
        study_name=f"spec_v15_run5_multiasset_{side}",
        direction="maximize",
        sampler=optuna.samplers.TPESampler(multivariate=True, group=True, seed=42),
    )
    seed = SEED_HEURISTIC_STRUCTURAL_HIGH if side == "high" else SEED_HEURISTIC_STRUCTURAL_LOW
    study.enqueue_trial(seed)
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)
    print(f"[{side}] best pooled LCB = {study.best_value:.5f}")
"""

# ---------------------------------------------------------------------------
# Cell C — Task 10 Step 2: reproducibility report (spec §5.2)
# ---------------------------------------------------------------------------
CELL_C_SOURCE = """\
# === Run 5 — record provenance for reproducibility (spec §5.2) ===
import json, hashlib
from src.volume_quality import profile_volume

def _hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()[:12]

report = {
    "selected_groups": SELECTED_GROUPS,
    "selected_timeframes": SELECTED_TIMEFRAMES,
    "volume_policy": VOLUME_POLICY,
    "streams": [
        {
            "stream_id": sd.stream.stream_id,
            "cluster": sd.stream.cluster_id,
            "bars": len(sd.df),
            "date_range": [str(sd.df.index[0]), str(sd.df.index[-1])],
            "data_hash": _hash(sd.stream.path),
            "volume_quality": profile_volume(sd.df).quality,
        }
        for sd in stream_datas
    ],
    "n_folds": len(folds),
}
print(json.dumps(report, indent=2))
"""


def _make_code_cell(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


def main() -> None:
    nb = json.load(open(NB_PATH, encoding="utf-8"))
    nb["cells"].append(_make_code_cell(CELL_A_SOURCE))
    nb["cells"].append(_make_code_cell(CELL_B_SOURCE))
    nb["cells"].append(_make_code_cell(CELL_C_SOURCE))
    json.dump(nb, open(NB_PATH, "w", encoding="utf-8"), indent=1)
    print(f"Appended 3 cells to {NB_PATH}")
    print(f"Total cells: {len(nb['cells'])}")


if __name__ == "__main__":
    main()
