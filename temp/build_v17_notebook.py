"""Build a clean, single-purpose v17 notebook: optimize_v17.ipynb.

8 cells, top-to-bottom, no v15/Optuna zombie parts. Reuses the proven v16
resolve + diagnostic code and the existing run_v17 runner. Detector untouched.
"""
from __future__ import annotations
import json
from pathlib import Path

BRANCH = "feature/v17-calibrated-coordinate-ascent"

MD_INTRO = f"""# Speculatores v17 — Calibrated Coordinate-Ascent

Clean, self-contained workbook for the **v17** optimizer. Run **top to bottom**.

v17 sets the detector's thresholds by **deterministic coordinate-ascent**: it
warm-starts from the committed v16 best config and sweeps one threshold at a
time over a small grid, keeping each move that improves the pooled
lower-confidence-bound (LCB) score on the same time-aligned multi-asset folds
the v16 path used. It is cheap (tens of evals), reproducible, and auditable.

**Parity:** the detector and the Pine indicator are **unchanged** — v17 only
chooses *which constants to use*. Outputs are Pine-portable numbers.

**Honest ceiling:** with only ~100 validatable macro pivots, no optimizer
manufactures signal the events lack. Treat v17 as a local refinement /
reproducibility check on v16, not a way past the event-count floor.

Flow: mount Drive → clone repo → set run settings → resolve pool → diagnostic →
**run v17** → inspect best params."""

CELL_MOUNT = """# Cell 1 - Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')"""

CELL_CLONE = f"""# Cell 2 - Clone/update repo (v17 branch) and install deps
import os
from pathlib import Path

REPO_DIR = Path('/content/cfd9')
REPO_URL = 'https://github.com/Sovenski/cfd9.git'
BRANCH = '{BRANCH}'

if not REPO_DIR.exists():
    !git clone {{REPO_URL}} {{REPO_DIR}}
%cd {{REPO_DIR}}
!git fetch origin -q && git checkout {{BRANCH}} && git pull --ff-only -q
!pip install -q -r requirements.txt
!git rev-parse --abbrev-ref HEAD && git rev-parse --short HEAD"""

CELL_SETTINGS = '''#@title v17 Run Settings  { display-mode: "form" }
#@markdown ## Speculatores v17 - run settings
#@markdown Fill the form, run this cell, then run the cells below in order.
#@markdown TradingView exports go in Google Drive at **`MyDrive/cfd9/data/raw_v16/`**
#@markdown (TV-native filenames are fine - no renaming).
from pathlib import Path
from datetime import datetime

DRIVE_ROOT = Path('/content/drive/MyDrive/cfd9')
DATA_DIR = DRIVE_ROOT / 'data' / 'raw_v16'
RESULTS_DIR = DRIVE_ROOT / 'results'
for _d in (DATA_DIR, RESULTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

#@markdown ---
#@markdown ### 1) Asset groups  (tick any combination to pool together)
USE_INDICES     = True   #@param {type:"boolean"}
USE_STOCKS      = False  #@param {type:"boolean"}
USE_COMMODITIES = True   #@param {type:"boolean"}
USE_WORLD_ETF   = False  #@param {type:"boolean"}
USE_FX          = False  #@param {type:"boolean"}
#@markdown Each v17 eval re-runs the detector on every stream*fold, so widen the
#@markdown pool OR raise the v17 knobs (section 5) - not both. INDICES+COMMODITIES
#@markdown is a sane first run.

#@markdown ---
#@markdown ### 2) Timeframe(s)
TF_1D  = True   #@param {type:"boolean"}
TF_240 = False  #@param {type:"boolean"}
TF_60  = False  #@param {type:"boolean"}

#@markdown ---
#@markdown ### 3) History window (common era)
COMMON_ERA = "auto"  #@param ["auto", "all", "2000-01-01", "2010-01-01", "2015-01-01"]
#@markdown **auto** concentrates folds where >=~half the streams coexist (recommended);
#@markdown **all** = full history; a date = start folds there.

#@markdown ---
#@markdown ### 4) Volume policy
VOLUME_POLICY = "price_only"  #@param ["price_only", "volume_required", "mixed"]

#@markdown ---
#@markdown ### 5) v17 optimizer knobs
V17_GRID_N     = 5     #@param {type:"integer"}
V17_MAX_SWEEPS = 2     #@param {type:"integer"}
V17_WARM_START = True  #@param {type:"boolean"}
#@markdown - **V17_GRID_N** - candidates per threshold per sweep (finer = slower).
#@markdown - **V17_MAX_SWEEPS** - full passes of coordinate ascent.
#@markdown - **V17_WARM_START** - seed from `results/v16_best_params.json` (recommended);
#@markdown   off = gold-default `Params()` seed.
SEED = 42  #@param {type:"integer"}

# ---------------------------------------------------------------------------
# Assemble selections (do not edit below)
# ---------------------------------------------------------------------------
SELECTED_GROUPS = [g for g, on in {
    "INDICES": USE_INDICES, "STOCKS": USE_STOCKS, "COMMODITIES": USE_COMMODITIES,
    "WORLD_ETF": USE_WORLD_ETF, "FX": USE_FX,
}.items() if on]
SELECTED_TIMEFRAMES = [tf for tf, on in {
    "1D": TF_1D, "240": TF_240, "60": TF_60,
}.items() if on]
_ERA_KW = {} if COMMON_ERA == "auto" else ({"start": "1871-01-01"} if COMMON_ERA == "all" else {"start": COMMON_ERA})
RUN_SLUG = f"v17_{'-'.join(SELECTED_TIMEFRAMES) or 'none'}_{datetime.now():%Y%m%d_%H%M%S}"

assert SELECTED_GROUPS, "Tick at least one asset group (section 1)."
assert SELECTED_TIMEFRAMES, "Tick at least one timeframe (section 2)."
print("data dir   :", DATA_DIR)
print("CSV files  :", len(list(DATA_DIR.glob("*.csv"))), "present in", DATA_DIR.name)
print("groups     :", SELECTED_GROUPS)
print("timeframes :", SELECTED_TIMEFRAMES, "| era:", COMMON_ERA, "| volume:", VOLUME_POLICY)
print("v17 knobs  : grid_n", V17_GRID_N, "| max_sweeps", V17_MAX_SWEEPS, "| warm_start", V17_WARM_START)
print("run_slug   :", RUN_SLUG)'''

CELL_RESOLVE = """# === resolve the stream pool ===
from src.universe import UNIVERSE, TIMEFRAMES, resolve_streams
print('available groups     :', list(UNIVERSE))
print('available timeframes :', TIMEFRAMES)
STREAMS = resolve_streams(SELECTED_GROUPS, SELECTED_TIMEFRAMES, data_dir=str(DATA_DIR))
print(f'resolved {len(STREAMS)} streams:', [s.stream_id for s in STREAMS])
if not STREAMS:
    print('\\nWARNING: 0 streams. Upload TV exports to', DATA_DIR,
          'as {TICKER}_{TF}_*.csv and re-run this cell.')"""

CELL_DIAG = '''#@title Pool sufficiency diagnostic (run before v17) { display-mode: "form" }
# === pre-run pool sufficiency diagnostic ===
from src.scoring import add_pivot_labels, REFERENCE_N
from src.pooled_validation import (
    StreamData, build_calendar_folds, load_stream_frame, apply_volume_policy,
)
import numpy as np
_TF_SECONDS = {"1D": 86400.0, "1W": 604800.0, "60": 3600.0, "240": 14400.0}

if not STREAMS:
    print('WARNING: 0 streams resolved - see the resolve cell above.')
else:
    _sd = []
    for s in STREAMS:
        df = load_stream_frame(s.path)
        df, keep = apply_volume_policy(df, policy=VOLUME_POLICY)
        if not keep:
            print(f'  drop {s.stream_id}: volume policy excluded it'); continue
        add_pivot_labels(df)
        _sd.append(StreamData(stream=s, df=df, bar_seconds=_TF_SECONDS[s.timeframe]))
    _folds = build_calendar_folds(_sd, **_ERA_KW)
    print(f'calendar folds   = {len(_folds)}; streams/fold = {[len(f) for f in _folds]}')
    for side, lbl in (("HIGH", 1), ("LOW", -1)):
        per_fold = [sum(int((sl.df_oos[f"pivot_N{REFERENCE_N}"] == lbl).sum()) for sl in f)
                    for f in _folds]
        n_inf = sum(1 for x in per_fold if x > 0)
        print(f'  {side}: OOS pivots/fold = {per_fold} | informative = {n_inf}/{len(_folds)}'
              f' | mean = {np.mean(per_fold) if per_fold else 0:.1f}')
    print('\\nRule of thumb: aim for >= ~5 informative folds, mean >= ~3 pivots/fold per side.')'''

CELL_RUN = """# === run v17 (Calibrated Coordinate-Ascent), both sides ===
import json
from src.v17_runner import run_v17

v17_seed = None
if V17_WARM_START:
    try:
        v17_seed = json.load(open('results/v16_best_params.json'))
        print('warm-start: results/v16_best_params.json (sides:', list(v17_seed), ')')
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print('warm-start unavailable (', type(e).__name__, ') -> gold-default seed')
        v17_seed = None
else:
    print('warm-start disabled -> gold-default Params() seed')

v17_out = run_v17(
    groups=SELECTED_GROUPS,
    timeframes=SELECTED_TIMEFRAMES,
    data_dir=str(DATA_DIR),
    sides=('high', 'low'),
    seed_params=v17_seed,
    volume_policy=VOLUME_POLICY,
    era_kw=_ERA_KW,
    grid_n=V17_GRID_N,
    max_sweeps=V17_MAX_SWEEPS,
    results_dir=str(RESULTS_DIR),
    run_slug=RUN_SLUG,
)

print(f"\\nv17 '{RUN_SLUG}'  pool={v17_out['streams']}  folds={v17_out['n_folds']}")
for side in ('high', 'low'):
    s = v17_out['sides'][side]
    print(f"[{side}] seed LCB {s['seed_lcb']:.5f}  ->  final LCB {s['final_lcb']:.5f}"
          f"   ({s['n_evals']} evals, changed {len(s['changed'])} thresholds)")
print('provenance:', v17_out.get('_written'))"""

CELL_RESULTS = """# === inspect best params per side (Pine-portable constants) ===
import json
for side in ('high', 'low'):
    s = v17_out['sides'][side]
    print(f"\\n===== {side.upper()}  (final LCB {s['final_lcb']:.5f}) =====")
    print('changed thresholds:', s['changed'])
    print(json.dumps(s['best_params'], indent=2, default=str))"""

CELLS = [
    ("markdown", MD_INTRO),
    ("code", CELL_MOUNT),
    ("code", CELL_CLONE),
    ("code", CELL_SETTINGS),
    ("code", CELL_RESOLVE),
    ("code", CELL_DIAG),
    ("code", CELL_RUN),
    ("code", CELL_RESULTS),
]


def _cell(i, ctype, src):
    lines = src.splitlines(keepends=True)
    base = {"cell_type": ctype, "id": f"v17c{i}", "metadata": {}, "source": lines}
    if ctype == "code":
        base.update({"execution_count": None, "outputs": []})
    return base


nb = {
    "cells": [_cell(i, t, s) for i, (t, s) in enumerate(CELLS)],
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python"},
        "colab": {"provenance": []},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
Path("optimize_v17.ipynb").write_text(json.dumps(nb, indent=1), encoding="utf-8")
print("wrote optimize_v17.ipynb with", len(nb["cells"]), "cells")

# --- validate: nbformat + compile pure-python code cells ---
import nbformat
loaded = nbformat.read("optimize_v17.ipynb", as_version=4)
nbformat.validate(loaded)
compiled = 0
for c in loaded.cells:
    if c.cell_type != "code":
        continue
    src = c.source
    if any(tok in src for tok in ("!git", "!pip", "%cd", "get_ipython", "google.colab")):
        continue  # Colab magics / runtime-only imports
    compile(src, "<cell>", "exec")
    compiled += 1
print(f"nbformat valid; compiled {compiled} pure-python code cells OK")
