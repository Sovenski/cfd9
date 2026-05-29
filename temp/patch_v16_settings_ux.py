"""Replace the v16 settings cell with a Colab form (interactive dropdowns/checkboxes)
and add friendly #@title headers to the diagnostic + launch cells. #@param / #@title /
#@markdown are Colab form magics and are inert comments in plain Python, so the cell
still runs normally outside Colab.
"""
from __future__ import annotations
import json
from pathlib import Path

NB = Path("optimize.ipynb")
nb = json.loads(NB.read_text(encoding="utf-8"))

SETTINGS = r'''#@title v16 Run Settings  { display-mode: "form" }
#@markdown ## Speculatores v16 - run settings
#@markdown Fill the form, run this cell, then run the cells below in order.
#@markdown TradingView exports go in Google Drive at **`MyDrive/cfd9/data/raw_v16/`**
#@markdown (TV-native filenames are fine - no renaming).
from pathlib import Path
from datetime import datetime

DRIVE_ROOT = Path('/content/drive/MyDrive/cfd9')
DATA_DIR = DRIVE_ROOT / 'data' / 'raw_v16'
RESULTS_DIR = DRIVE_ROOT / 'results'
RUNS_DIR = DRIVE_ROOT / 'runs'
for _d in (DATA_DIR, RESULTS_DIR, RUNS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

#@markdown ---
#@markdown ### 1) Asset groups  (tick any combination to pool together)
USE_INDICES     = True   #@param {type:"boolean"}
USE_STOCKS      = False  #@param {type:"boolean"}
USE_COMMODITIES = True   #@param {type:"boolean"}
USE_WORLD_ETF   = True   #@param {type:"boolean"}
USE_FX          = True   #@param {type:"boolean"}
#@markdown - **INDICES** - SPX, NDX, DAX
#@markdown - **STOCKS** - 30 US large-caps (many events, but all one US-equity cluster)
#@markdown - **COMMODITIES** - gold / silver / platinum / palladium (oil only at 240/60)
#@markdown - **WORLD_ETF** - VT, VWCE
#@markdown - **FX** - 7 major pairs (EURUSD, GBPUSD, ...)
#@markdown Recommended default for daily: INDICES + COMMODITIES + WORLD_ETF + FX
#@markdown (~16 streams across 5 independent clusters).

#@markdown ---
#@markdown ### 2) Timeframe(s)  (tick one, or several to mix)
TF_1D  = True   #@param {type:"boolean"}
TF_240 = False  #@param {type:"boolean"}
TF_60  = False  #@param {type:"boolean"}
#@markdown - **1D** = daily (recommended; matches the macro-pivot goal)
#@markdown - **240 / 60** = 4h / 1h intraday (more independent assets, but mixing horizons
#@markdown   is a *scale-invariance bet* - the structural nest is in bars, so a 1h pivot is
#@markdown   an intraday swing, not a macro top). Validate before trusting a mixed run.

#@markdown ---
#@markdown ### 3) Volume policy
VOLUME_POLICY = "price_only"  #@param ["price_only", "volume_required", "mixed"]
#@markdown - **price_only** - ignore volume (robust default; raw index daily volume is patchy)
#@markdown - **volume_required** - use only each stream's real-volume date range (best for
#@markdown   LOW-side volume experiments); streams with no real volume are dropped
#@markdown - **mixed** - use volume where present, inert where missing

#@markdown ---
#@markdown ### 4) Optimizer budget
N_TRIALS = 250  #@param {type:"integer"}
SEED = 42       #@param {type:"integer"}
#@markdown `N_TRIALS` is per side (HIGH and LOW each get this many). The run is resumable -
#@markdown if Colab disconnects, just re-run the launch cell and it continues.

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
RUN_SLUG = f"v16_{'-'.join(SELECTED_TIMEFRAMES) or 'none'}_{datetime.now():%Y%m%d_%H%M%S}"

assert SELECTED_GROUPS, "Tick at least one asset group (section 1)."
assert SELECTED_TIMEFRAMES, "Tick at least one timeframe (section 2)."
print("data dir   :", DATA_DIR)
print("CSV files  :", len(list(DATA_DIR.glob("*.csv"))), "present in", DATA_DIR.name)
print("groups     :", SELECTED_GROUPS)
print("timeframes :", SELECTED_TIMEFRAMES)
print("volume     :", VOLUME_POLICY, "| trials/side:", N_TRIALS, "| seed:", SEED)
print("run_slug   :", RUN_SLUG)
'''

changed = []
for c in nb["cells"]:
    if c.get("cell_type") != "code":
        continue
    src = "".join(c["source"]) if isinstance(c["source"], list) else c["source"]
    # settings cell: identify by its defining vars
    if "SELECTED_GROUPS" in src and "DATA_DIR = DRIVE_ROOT" in src and "resolve_streams" not in src:
        c["source"] = SETTINGS.splitlines(keepends=True)
        changed.append("settings")
    elif "pool sufficiency diagnostic" in src and "#@title" not in src:
        c["source"] = ['#@title Step A - Pool sufficiency diagnostic (run BEFORE launch) '
                       '{ display-mode: "form" }\n'] + (c["source"] if isinstance(c["source"], list)
                                                        else [c["source"]])
        changed.append("diagnostic")
    elif "launch BOTH sides" in src and "#@title" not in src:
        c["source"] = ['#@title Step B - Launch optimization (both sides, resumable) '
                       '{ display-mode: "form" }\n'] + (c["source"] if isinstance(c["source"], list)
                                                        else [c["source"]])
        changed.append("launch")

assert "settings" in changed, f"settings cell not found (changed={changed})"
NB.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print("patched:", changed)
