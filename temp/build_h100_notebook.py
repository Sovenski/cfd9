"""Build the H100 validation + GPU-optimizer workbook: h100_v17_gpu.ipynb.

Follows the optimize_v17.ipynb convention: run top-to-bottom on a Colab H100.
Mount Drive -> clone repo (feature/v17-antigaming) -> stage the two daily CSVs
from Drive -> run the ~5-min H100 validation -> (optional) full GPU run.
"""
from __future__ import annotations

import json
from pathlib import Path

BRANCH = "feature/v17-antigaming"

MD_INTRO = f"""# Speculatores v17-GPU — H100 validation + batched optimizer

Run **top to bottom** on a **Colab H100 runtime** (Runtime > Change runtime type > H100 GPU).

Two jobs, in order:

1. **H100 validation (~5 min)** — confirms on real hardware what is already
   proven on CPU: the torch evaluator is **byte-identical** to the exact
   `SpeculatorDetector` (trust-kernel branch, CPU-measured flip rate 0.0).
   Prints one line: `H100 VALIDATION: PASS|FAIL`.
2. **(Optional) full GPU optimizer run** — Sobol + sep-CMA-ES population scored
   by the batched `score_pop`; top-K finalists re-scored by the **exact CPU
   detector**; acceptance gates + per-asset TV parity audit. Every reported
   number comes from the CPU detector — GPU only ranks.

**Parity:** detector + Pine indicator unchanged. **Honest ceiling:** the GPU
buys throughput, not statistical power on the ~27-event HIGH side.

Branch: `{BRANCH}` (make the repo public before Cell 2, private again after)."""

CELL_MOUNT = """# Cell 1 - Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')"""

CELL_CLONE = f"""# Cell 2 - Clone/update repo ({BRANCH}) and install deps
import os
from pathlib import Path

REPO_DIR = Path('/content/cfd9')
REPO_URL = 'https://github.com/Sovenski/cfd9.git'
BRANCH = '{BRANCH}'

if not REPO_DIR.exists():
    !git clone --branch {{BRANCH}} {{REPO_URL}} {{REPO_DIR}}
%cd {{REPO_DIR}}
!git fetch -q origin {{BRANCH}}
!git checkout -q {{BRANCH}}
!git pull --ff-only -q
!pip install -q -r requirements.txt pytest
import torch
import subprocess
sha = subprocess.run(['git', 'rev-parse', '--short', 'HEAD'],
                     capture_output=True, text=True).stdout.strip()
print('repo @', sha)
print('torch', torch.__version__, '| cuda:', torch.cuda.is_available(),
      '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO GPU - fix runtime type!')"""

CELL_SETTINGS = '''#@title Cell 3 - Data settings + stage CSVs  { display-mode: "form" }
#@markdown The validation needs the two long daily histories copied into the
#@markdown cloned repo. Point these at your Drive copies (defaults assume
#@markdown **`MyDrive/cfd9/data/raw_v16/`**). The optional optimizer run uses
#@markdown the whole `DATA_DIR` pool.
from pathlib import Path
import shutil

DRIVE_ROOT = Path('/content/drive/MyDrive/cfd9')
DATA_DIR = '/content/drive/MyDrive/cfd9/data/raw_v16'  #@param {type:"string"}
RESULTS_DIR = '/content/drive/MyDrive/cfd9/results'    #@param {type:"string"}
SPX_1D_CSV = '/content/drive/MyDrive/cfd9/data/raw_v16/SP_SPX, 1D_a20e0.csv'  #@param {type:"string"}
DAX_1D_CSV = '/content/drive/MyDrive/cfd9/data/raw_v16/XETR_DLY_DAX, 1D_0ab60.csv'  #@param {type:"string"}

REPO_RAW = Path('/content/cfd9/data/raw')
REPO_RAW.mkdir(parents=True, exist_ok=True)
_targets = {
    SPX_1D_CSV: REPO_RAW / 'SPX_1D_18710201_20260318.csv',
    DAX_1D_CSV: REPO_RAW / 'DAX_1D_19700102_20260324.csv',
}
missing = [s for s in _targets if not Path(s).exists()]
if missing:
    print('MISSING source CSV(s):')
    for s in missing:
        print('  -', s)
    print('\\nCandidates found on Drive:')
    for p in sorted(Path(DATA_DIR).glob('*.csv')):
        print('  *', p.name)
    raise FileNotFoundError('fix SPX_1D_CSV / DAX_1D_CSV form fields above')
for src, dst in _targets.items():
    if not dst.exists():
        shutil.copy(src, dst)
    print('staged', dst.name, f'({dst.stat().st_size/1e6:.1f} MB)')
Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)
print('results ->', RESULTS_DIR)'''

CELL_VALIDATE = """# Cell 4 - H100 validation (~5 min) - THE gate. Needs: PASS
# Re-runs the PIR byte-identity spike ON THE GPU, the full GPU parity test
# suite ON THE GPU, an end-to-end signal-flip measurement vs the exact CPU
# detector, and one tiny run_v17_gpu. PASS = trust-kernel confirmed on H100.
!python temp/colab_h100_validation.py"""

CELL_RUN = '''#@title Cell 5 (optional) - Full GPU optimizer run  { display-mode: "form" }
#@markdown Only run after Cell 4 printed **PASS**. Scores a Sobol+CMA-ES
#@markdown population on the H100; finalists re-scored by the exact CPU
#@markdown detector; acceptance gates + TV parity audit. Results JSON -> Drive.
from datetime import datetime

GROUPS = "indices"          #@param {type:"string"}
TIMEFRAMES = "1D"           #@param {type:"string"}
SIDES = "high,low"          #@param {type:"string"}
#@markdown **Sizing (measured on L4):** ~14s/generation at POPSIZE 128 — the GPU is
#@markdown not the bottleneck; the CPU finalist re-score tail is. L4 is sufficient.
POPSIZE = 128               #@param {type:"integer"}
GENERATIONS = 10            #@param {type:"integer"}
SOBOL_N = 128               #@param {type:"integer"}
TOP_K = 16                  #@param {type:"integer"}
RNG_SEED = 42               #@param {type:"integer"}
#@markdown **Fold geometry.** The centered 200-bar label window kills the first/last
#@markdown 200 bars of every slice, so SMALL OOS slices have almost no scorable bars
#@markdown (the default 0.03 leaves ~28 live bars -> all LCBs 0.0). Large slices:
#@markdown IS~2850 / OOS~2140 bars (~1740 live) -> ~5 real folds on the indices pool.
IS_FRACTION = 0.20          #@param {type:"number"}
OOS_FRACTION = 0.15         #@param {type:"number"}
STEP_FRACTION = 0.10        #@param {type:"number"}

import logging, json
# force=True: Colab's kernel pre-installs a root handler, which makes a plain
# basicConfig a silent no-op — without this you see NO progress at all.
logging.basicConfig(level=logging.INFO, force=True,
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")
from src.v17_runner import run_v17_gpu

print('starting run_v17_gpu — fold/artifact build takes a few minutes before '
      'the first generation logs appear...', flush=True)

run_slug = f"v17gpu_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
out = run_v17_gpu(
    groups=[g.strip().upper() for g in GROUPS.split(',') if g.strip()],
    timeframes=[t.strip() for t in TIMEFRAMES.split(',') if t.strip()],
    data_dir=DATA_DIR,
    sides=tuple(s.strip() for s in SIDES.split(',') if s.strip()),
    era_kw={"is_fraction": IS_FRACTION, "oos_fraction": OOS_FRACTION,
            "step_fraction": STEP_FRACTION},
    results_dir=RESULTS_DIR,
    run_slug=run_slug,
    search_kw={"popsize": POPSIZE, "generations": GENERATIONS,
               "sobol_n": SOBOL_N, "top_k": TOP_K, "rng_seed": RNG_SEED},
    device="cuda",
)
print(json.dumps({s: {k: v for k, v in d.items() if k != 'trace'}
                  for s, d in out.get('sides', {}).items()}, indent=2, default=str)[:4000])
print('written ->', out.get('_written'))'''

CELL_INSPECT = '''# Cell 6 - FULL ENGINE REPORT (copy-paste friendly)
import json as _json

def _f(x, n=5):
    return f"{x:.{n}f}" if isinstance(x, (int, float)) and x is not None else str(x)

L = []
A = L.append
A("=" * 78)
A(f"SPECULATORES v17-GPU RUN REPORT — {out.get('run_slug')}")
A("=" * 78)
A(f"pool:      {out.get('streams')}  ({out.get('n_folds')} folds, device={out.get('device')})")
A(f"search:    {out.get('search')}  top_k={out.get('top_k')}  flip_rate={out.get('flip_rate')}"
  f"  finalist_tol={out.get('finalist_tol')}")
A(f"volume:    {out.get('volume_policy')}   groups={out.get('groups')} tf={out.get('timeframes')}")
A(f"results:   {out.get('_written', '(not written)')}")
for side, d in out.get("sides", {}).items():
    acc = d.get("acceptance", {})
    defl = d.get("deflation", {})
    boot = acc.get("bootstrap", {})
    A("")
    A("-" * 78)
    A(f"{side.upper()} SIDE")
    A("-" * 78)
    A(f"LCB:        seed {_f(d.get('seed_lcb'))} -> final {_f(d.get('final_lcb'))}"
      f"   DEFLATED {_f(d.get('final_lcb_deflated'))}")
    A(f"deflation:  haircut {_f(defl.get('haircut'))} (sigma {_f(defl.get('sigma_trials'))},"
      f" E[maxZ_{defl.get('n_trials')}] {_f(defl.get('e_max_z'), 3)})")
    A(f"verdict:    {acc.get('verdict', 'n/a')}   boundary-pinned: {acc.get('pinned') or 'none'}")
    A(f"bootstrap:  mean {_f(boot.get('mean'))} std {_f(boot.get('std'))} min {_f(boot.get('min'))}"
      f"  pass={boot.get('pass')}   era_pass={acc.get('era_pass')}")
    A(f"evals:      {d.get('n_evals')}   finalists kept/dropped: "
      f"{d.get('n_finalists')}/{d.get('n_dropped_finalists')}")
    A("")
    A("changed thresholds vs seed:")
    changed = d.get("changed", [])
    if not changed:
        A("  (none — landscape flat or seed optimal)")
    for fld, v in changed:
        A(f"  {fld:40s} -> {v!r}")
    A("")
    A("finalist leaderboard (CPU-exact LCB; abs_diff = |gpu-cpu| parity):")
    coords = d.get("coords", [])
    for k, e in enumerate(d.get("leaderboard", [])[:10]):
        vals = " ".join(f"{c.split('_')[0][:6]}={e.get(c):.4g}" for c in coords if c in e)
        A(f"  #{k+1:<2d} cpu_lcb={_f(e.get('cpu_lcb'))} gpu={_f(e.get('gpu_lcb'))}"
          f" d={e.get('abs_diff'):.1e} [{e.get('stage')}]  {vals}")
    if side == "high" and "per_asset_diagnostic" in d:
        A("")
        A(f"HIGH per-asset diagnostic (advisory): "
          f"{_json.dumps(d['per_asset_diagnostic'], default=str)[:400]}")
    A("")
    A(f"FULL best_params ({side}) — Pine-ready:")
    A(_json.dumps(d.get("best_params", {}), indent=2, default=str))
if out.get("tv_audit") is not None:
    A("")
    A(f"tv_audit hook: {_json.dumps(out['tv_audit'], default=str)[:400]}")
report = "\\n".join(L)
print(report)'''


def _code(src: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": src}


def _md(src: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": src}


nb = {
    "cells": [
        _md(MD_INTRO),
        _code(CELL_MOUNT),
        _code(CELL_CLONE),
        _code(CELL_SETTINGS),
        _code(CELL_VALIDATE),
        _code(CELL_RUN),
        _code(CELL_INSPECT),
    ],
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python"},
        "colab": {"provenance": [], "gpuType": "A100"},
        "accelerator": "GPU",
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
Path("h100_v17_gpu.ipynb").write_text(json.dumps(nb, indent=1), encoding="utf-8")
print("wrote h100_v17_gpu.ipynb with", len(nb["cells"]), "cells")

# --- validate: nbformat + compile pure-python code cells ---
import nbformat

loaded = nbformat.read("h100_v17_gpu.ipynb", as_version=4)
nbformat.validate(loaded)
compiled = 0
for c in loaded.cells:
    if c.cell_type != "code":
        continue
    src = c.source
    if any(tok in src for tok in ("!git", "!pip", "!python", "%cd", "get_ipython", "google.colab")):
        continue  # Colab magics / runtime-only imports
    compile(src, "<cell>", "exec")
    compiled += 1
print(f"nbformat valid; compiled {compiled} pure-python code cells OK")
