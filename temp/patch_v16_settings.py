"""Targeted patch: update the v16 settings cell's DATA_DIR + SELECTED_GROUPS to the
real raw_v16 export dir and the broad multi-cluster 1D default. ASCII-robust matching."""
from __future__ import annotations
import json
from pathlib import Path

NB = Path("optimize.ipynb")
nb = json.loads(NB.read_text(encoding="utf-8"))

OLD_DATA = "DATA_DIR = DRIVE_ROOT / 'data' / 'raw'"
NEW_DATA = "DATA_DIR = DRIVE_ROOT / 'data' / 'raw_v16'  # <-- TV exports here (TV-native filenames OK)"
OLD_GROUPS = 'SELECTED_GROUPS = ["INDICES_US", "INDICES_GLOBAL"]   # see src.universe.UNIVERSE.keys()'
NEW_GROUPS = 'SELECTED_GROUPS = ["INDICES", "COMMODITIES", "WORLD_ETF", "FX"]   # groups: INDICES, STOCKS, COMMODITIES, WORLD_ETF, FX'

changed = 0
for c in nb["cells"]:
    src = "".join(c["source"]) if isinstance(c["source"], list) else c["source"]
    new = src
    if OLD_DATA in new and "raw_v16" not in new:
        new = new.replace(OLD_DATA + "      # <-- put TV exports here: {TICKER}_{TF}_*.csv", NEW_DATA)
        if new == src:  # comment text differed; fall back to line-prefix replace
            new = new.replace(OLD_DATA, NEW_DATA)
    if OLD_GROUPS in new:
        new = new.replace(OLD_GROUPS, NEW_GROUPS)
    if new != src:
        c["source"] = new.splitlines(keepends=True)
        changed += 1

assert changed, "no cell patched — check anchors"
NB.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print(f"patched {changed} cell(s): DATA_DIR=data/raw_v16, broad multi-cluster default")
