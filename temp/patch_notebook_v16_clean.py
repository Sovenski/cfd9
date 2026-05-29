"""Rebuild optimize.ipynb into a clean, top-to-bottom v16 multi-asset flow.

Fixes: title still said v15; old single-asset cells ran first; v16 read repo-
relative (untracked, empty-on-Colab) data/raw instead of Drive; no resumable
storage for a 250-trial run. New flow:
  0 md intro | 1 mount | 2 clone+install | 3 v16 settings (Drive data dir) |
  4 resolve pool | 5 diagnostic | 6 launch BOTH sides (persistent+seeded+resumable) |
  7 provenance | 8 md legacy divider | 9..13 legacy v15 single-asset (optional)
Legacy setup/run cells are preserved (reused verbatim) below the divider.
"""
from __future__ import annotations
import json
from pathlib import Path

NB = Path("optimize.ipynb")
nb = json.loads(NB.read_text(encoding="utf-8"))
cells = nb["cells"]


def _src(i: int) -> list[str]:
    s = cells[i]["source"]
    return s if isinstance(s, list) else [s]


# Reuse existing setup cells verbatim: 1 (mount drive), 2 (clone+install).
mount_src = _src(1)
clone_src = _src(2)
# Legacy single-asset cells = current 3,4,5,6,7 (config/launch/monitor/report/parity).
legacy = [_src(3), _src(4), _src(5), _src(6), _src(7)]


def code(text: str) -> dict:
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": text.splitlines(keepends=True)}


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {},
            "source": text.splitlines(keepends=True)}


intro = md(
    "# Speculatores v16 — Multi-Asset Pooled Optimizer\n"
    "\n"
    "Run **top to bottom**. Optimizes both pivot sides on a *pool* of "
    "`(asset, timeframe)` streams (default: indices, 1D), **250 trials/side**, with "
    "persistent **resumable** storage on Drive.\n"
    "\n"
    "**Before running:** put your TradingView daily exports in Google Drive at "
    "`MyDrive/cfd9/data/raw/`, named `{TICKER}_{TF}_*.csv` "
    "(e.g. `SPX_1D_*.csv`, `NDX_1D_*.csv`, `DAX_1D_*.csv`). The detector and Pine "
    "indicator are unchanged (parity preserved).\n"
    "\n"
    "Flow: mount Drive → clone repo → set run settings → resolve pool → **run the "
    "diagnostic** (confirms the pool has enough structural events per fold) → launch "
    "(both sides) → provenance report.\n"
    "\n"
    "The legacy **v15 single-asset** runner is preserved at the bottom (optional).\n"
)

settings = code(
    "# === v16 — run settings ===\n"
    "from pathlib import Path\n"
    "from datetime import datetime\n"
    "\n"
    "DRIVE_ROOT = Path('/content/drive/MyDrive/cfd9')\n"
    "DATA_DIR = DRIVE_ROOT / 'data' / 'raw'      # <-- put TV exports here: {TICKER}_{TF}_*.csv\n"
    "RESULTS_DIR = DRIVE_ROOT / 'results'\n"
    "RUNS_DIR = DRIVE_ROOT / 'runs'\n"
    "for _d in (DATA_DIR, RESULTS_DIR, RUNS_DIR):\n"
    "    _d.mkdir(parents=True, exist_ok=True)\n"
    "\n"
    "# --- pool selection (edit as desired) ---\n"
    "SELECTED_GROUPS = [\"INDICES_US\", \"INDICES_GLOBAL\"]   # see src.universe.UNIVERSE.keys()\n"
    "SELECTED_TIMEFRAMES = [\"1D\"]\n"
    "VOLUME_POLICY = \"price_only\"                          # price_only | volume_required | mixed\n"
    "# --- optimizer budget ---\n"
    "N_TRIALS = 250                                         # per side\n"
    "SEED = 42\n"
    "RUN_SLUG = f\"v16_{'-'.join(SELECTED_TIMEFRAMES)}_{datetime.now():%Y%m%d_%H%M%S}\"\n"
    "\n"
    "print('data dir   :', DATA_DIR)\n"
    "print('CSV present:', sorted(p.name for p in DATA_DIR.glob('*.csv')))\n"
    "print('settings   :', dict(groups=SELECTED_GROUPS, timeframes=SELECTED_TIMEFRAMES,\n"
    "                          volume_policy=VOLUME_POLICY, n_trials=N_TRIALS, run_slug=RUN_SLUG))\n"
)

resolve = code(
    "# === v16 — resolve the pool ===\n"
    "from src.universe import UNIVERSE, TIMEFRAMES, resolve_streams\n"
    "print('available groups    :', list(UNIVERSE))\n"
    "print('available timeframes :', TIMEFRAMES)\n"
    "STREAMS = resolve_streams(SELECTED_GROUPS, SELECTED_TIMEFRAMES, data_dir=str(DATA_DIR))\n"
    "print(f'resolved {len(STREAMS)} streams:', [s.stream_id for s in STREAMS])\n"
    "if not STREAMS:\n"
    "    print('\\nWARNING: 0 streams resolved. Upload TV exports to', DATA_DIR,\n"
    "          'as {TICKER}_{TF}_*.csv and re-run this cell.')\n"
)

diagnostic = code(
    "# === v16 — pre-run pool sufficiency diagnostic (run BEFORE launch) ===\n"
    "from src.scoring import add_pivot_labels, REFERENCE_N\n"
    "from src.pooled_validation import (\n"
    "    StreamData, build_calendar_folds, load_stream_frame, apply_volume_policy,\n"
    ")\n"
    "import numpy as np\n"
    "_TF_SECONDS = {\"1D\": 86400.0, \"1W\": 604800.0, \"60\": 3600.0, \"240\": 14400.0}\n"
    "\n"
    "if not STREAMS:\n"
    "    print('WARNING: 0 streams resolved — see the resolve cell above.')\n"
    "else:\n"
    "    _sd = []\n"
    "    for s in STREAMS:\n"
    "        df = load_stream_frame(s.path)\n"
    "        df, keep = apply_volume_policy(df, policy=VOLUME_POLICY)\n"
    "        if not keep:\n"
    "            print(f'  drop {s.stream_id}: volume policy excluded it'); continue\n"
    "        add_pivot_labels(df)\n"
    "        _sd.append(StreamData(stream=s, df=df, bar_seconds=_TF_SECONDS[s.timeframe]))\n"
    "    _folds = build_calendar_folds(_sd)\n"
    "    print(f'resolved streams = {[s.stream_id for s in STREAMS]}')\n"
    "    print(f'calendar folds   = {len(_folds)}; streams/fold = {[len(f) for f in _folds]}')\n"
    "    for side, lbl in ((\"HIGH\", 1), (\"LOW\", -1)):\n"
    "        per_fold = [sum(int((sl.df_oos[f'pivot_N{REFERENCE_N}'] == lbl).sum()) for sl in f)\n"
    "                    for f in _folds]\n"
    "        n_inf = sum(1 for x in per_fold if x > 0)\n"
    "        print(f'  {side}: OOS pivots/fold = {per_fold} | informative = {n_inf}/{len(_folds)}'\n"
    "              f' | mean = {np.mean(per_fold) if per_fold else 0:.1f}')\n"
    "    print('\\nRule of thumb: aim for >= ~5 informative folds, mean >= ~3 pivots/fold per '\n"
    "          'side. If thin, add more index exports before launching.')\n"
)

launch = code(
    "# === v16 — build pooled folds + launch BOTH sides (persistent, resumable) ===\n"
    "import optuna\n"
    "from src.scoring import add_pivot_labels\n"
    "from src.monitor145 import make_storage\n"
    "from src.pooled_validation import (\n"
    "    StreamData, build_calendar_folds, build_pooled_optuna_objective,\n"
    "    load_stream_frame, apply_volume_policy,\n"
    ")\n"
    "from src.speculatores145 import (\n"
    "    params_from_trial, SEED_HEURISTIC_STRUCTURAL_HIGH, SEED_HEURISTIC_STRUCTURAL_LOW,\n"
    ")\n"
    "optuna.logging.set_verbosity(optuna.logging.WARNING)\n"
    "_TF_SECONDS = {\"1D\": 86400.0, \"1W\": 604800.0, \"60\": 3600.0, \"240\": 14400.0}\n"
    "\n"
    "stream_datas = []\n"
    "for s in STREAMS:\n"
    "    df = load_stream_frame(s.path)\n"
    "    df, keep = apply_volume_policy(df, policy=VOLUME_POLICY)\n"
    "    if not keep:\n"
    "        print(f'drop {s.stream_id}: volume policy excluded it'); continue\n"
    "    add_pivot_labels(df)\n"
    "    stream_datas.append(StreamData(stream=s, df=df, bar_seconds=_TF_SECONDS[s.timeframe]))\n"
    "\n"
    "folds = build_calendar_folds(stream_datas)\n"
    "streams = [sd.stream for sd in stream_datas]\n"
    "print(f'{len(folds)} folds; streams/fold = {[len(f) for f in folds]}')\n"
    "assert folds, 'No folds — pool too small/short. Add more/longer exports (run the diagnostic).'\n"
    "\n"
    "_SEEDS = {\"high\": SEED_HEURISTIC_STRUCTURAL_HIGH, \"low\": SEED_HEURISTIC_STRUCTURAL_LOW}\n"
    "for side in (\"high\", \"low\"):\n"
    "    storage = make_storage(RUNS_DIR / f'{RUN_SLUG}_{side}.journal')\n"
    "    study = optuna.create_study(\n"
    "        study_name=f'spec_v16_{side}', direction='maximize',\n"
    "        storage=storage, load_if_exists=True,\n"
    "        sampler=optuna.samplers.TPESampler(multivariate=True, group=True, seed=SEED),\n"
    "        pruner=optuna.pruners.MedianPruner(),\n"
    "    )\n"
    "    if not study.trials:\n"
    "        study.enqueue_trial(_SEEDS[side])\n"
    "    done = len([t for t in study.trials if t.state.is_finished()])\n"
    "    remaining = max(0, N_TRIALS - done)\n"
    "    print(f'[{side}] resuming at {done}/{N_TRIALS}; running {remaining} more ...')\n"
    "    study.optimize(\n"
    "        build_pooled_optuna_objective(folds, streams, params_from_trial, side),\n"
    "        n_trials=remaining, show_progress_bar=True,\n"
    "    )\n"
    "    print(f'[{side}] best LCB = {study.best_value:.5f}')\n"
    "    print(f'[{side}] best params = {study.best_params}')\n"
)

provenance = code(
    "# === v16 — provenance report (reproducibility, spec 5.2) ===\n"
    "import json, hashlib\n"
    "from src.volume_quality import profile_volume\n"
    "\n"
    "def _hash(path):\n"
    "    h = hashlib.sha256()\n"
    "    with open(path, 'rb') as f:\n"
    "        for chunk in iter(lambda: f.read(8192), b''):\n"
    "            h.update(chunk)\n"
    "    return h.hexdigest()[:12]\n"
    "\n"
    "report = {\n"
    "    'run_slug': RUN_SLUG,\n"
    "    'selected_groups': SELECTED_GROUPS,\n"
    "    'selected_timeframes': SELECTED_TIMEFRAMES,\n"
    "    'volume_policy': VOLUME_POLICY,\n"
    "    'n_trials_per_side': N_TRIALS,\n"
    "    'n_folds': len(folds),\n"
    "    'streams': [\n"
    "        {'stream_id': sd.stream.stream_id, 'cluster': sd.stream.cluster_id,\n"
    "         'bars': len(sd.df),\n"
    "         'date_range': [str(sd.df.index[0]), str(sd.df.index[-1])],\n"
    "         'data_hash': _hash(sd.stream.path),\n"
    "         'volume_quality': profile_volume(sd.df).quality}\n"
    "        for sd in stream_datas\n"
    "    ],\n"
    "}\n"
    "out = RESULTS_DIR / f'{RUN_SLUG}_provenance.json'\n"
    "out.write_text(json.dumps(report, indent=2))\n"
    "print('wrote', out)\n"
    "print(json.dumps(report, indent=2))\n"
)

legacy_divider = md(
    "---\n"
    "## Legacy: v15 single-asset runner (optional — NOT part of v16)\n"
    "\n"
    "The cells below are the previous single-asset, script-backed pipeline "
    "(`run_speculatores_145.py`). They are kept for reference / one-off single-asset "
    "runs. **Do not run them as part of the v16 flow above** — they define their own "
    "config and launch a different optimizer.\n"
)

new_cells = [intro,
             code("".join(mount_src)),
             code("".join(clone_src)),
             settings, resolve, diagnostic, launch, provenance,
             legacy_divider]
for legacy_src in legacy:
    new_cells.append(code("".join(legacy_src)))

nb["cells"] = new_cells
NB.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print(f"Wrote {NB}: {len(new_cells)} cells "
      f"(v16 flow = cells 0-7, legacy divider = 8, legacy = 9-{len(new_cells)-1})")
