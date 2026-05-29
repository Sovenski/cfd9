"""Wire a COMMON_ERA form knob into the v16 notebook settings cell and pass it
to build_calendar_folds in the diagnostic + launch cells."""
from __future__ import annotations
import json
from pathlib import Path

NB = Path("optimize.ipynb")
nb = json.loads(NB.read_text(encoding="utf-8"))

ERA_PARAM = (
    '#@markdown ---\n'
    '#@markdown ### 2b) History window (common era)\n'
    'COMMON_ERA = "auto"  #@param ["auto", "all", "2000-01-01", "2010-01-01", "2015-01-01"]\n'
    '#@markdown - **auto** (recommended) - concentrate folds where >=~half the streams\n'
    '#@markdown   coexist, so every fold is multi-asset (avoids thin single-asset folds\n'
    '#@markdown   from a long stream like SPX-1871). - **all** = use full history (old behaviour).\n'
    '#@markdown   - a date = start folds at that date.\n'
)
ERA_KW = (
    "_ERA_KW = {} if COMMON_ERA == \"auto\" else "
    "({\"start\": \"1871-01-01\"} if COMMON_ERA == \"all\" else {\"start\": COMMON_ERA})\n"
)

changed = []
for c in nb["cells"]:
    if c.get("cell_type") != "code":
        continue
    src = "".join(c["source"]) if isinstance(c["source"], list) else c["source"]
    new = src
    # settings cell
    if "SELECTED_GROUPS = [g for g, on in {" in new and "COMMON_ERA" not in new:
        new = new.replace('#@markdown ### 3) Volume policy', ERA_PARAM + '#@markdown ### 3) Volume policy', 1)
        # insert _ERA_KW right after the SELECTED_TIMEFRAMES assignment block
        anchor = '}.items() if on]\nRUN_SLUG ='
        assert anchor in new, "timeframes/RUN_SLUG anchor not found"
        new = new.replace(anchor, '}.items() if on]\n' + ERA_KW + 'RUN_SLUG =', 1)
        changed.append("settings")
    elif "_folds = build_calendar_folds(_sd)" in new:
        new = new.replace("_folds = build_calendar_folds(_sd)",
                          "_folds = build_calendar_folds(_sd, **_ERA_KW)")
        changed.append("diagnostic")
    elif "folds = build_calendar_folds(stream_datas)" in new:
        new = new.replace("folds = build_calendar_folds(stream_datas)",
                          "folds = build_calendar_folds(stream_datas, **_ERA_KW)")
        changed.append("launch")
    if new != src:
        c["source"] = new.splitlines(keepends=True)

assert set(changed) >= {"settings", "diagnostic", "launch"}, f"missing: {changed}"
NB.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print("patched:", changed)
