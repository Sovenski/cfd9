"""Apply the v17 notebook PatchPlan (from the workflow) to optimize.ipynb.

Deterministic: load plan -> apply settings string-replacements to whichever cell
uniquely contains each old_string -> insert the new cells (in order) right after
the provenance cell -> validate JSON + compile the new code cell + assert order.
"""
from __future__ import annotations
import html
import json
from pathlib import Path

PLAN_FILE = Path(
    r"C:\Users\kuben\AppData\Local\Temp\claude\C--Users-kuben-Desktop-Projekte-cfd9"
    r"\ee34217a-20a8-407f-ab16-978826afdd2c\tasks\wxt4kxlwa.output"
)
NB = Path("optimize.ipynb")

plan = json.loads(PLAN_FILE.read_text(encoding="utf-8"))["result"]
nb = json.loads(NB.read_text(encoding="utf-8"))


def _src(cell) -> str:
    s = cell["source"]
    return "".join(s) if isinstance(s, list) else s


def _set(cell, text: str) -> None:
    cell["source"] = text.splitlines(keepends=True)


# --- 1. settings / markdown string replacements (each must hit exactly one cell)
for rep in plan["settings_replacements"]:
    old = html.unescape(rep["old_string"])
    new = html.unescape(rep["new_string"])
    hits = [c for c in nb["cells"] if old in _src(c)]
    assert len(hits) == 1, f"replacement matched {len(hits)} cells, expected 1:\n{old[:80]!r}"
    _set(hits[0], _src(hits[0]).replace(old, new, 1))
    print("applied replacement -> 1 cell")

# --- 2. insert new cells (preserve order) right after the anchor cell
def _new_cell(spec: dict) -> dict:
    source = html.unescape(spec["source"]).splitlines(keepends=True)
    if spec["cell_type"] == "code":
        return {"cell_type": "code", "metadata": {}, "execution_count": None,
                "outputs": [], "source": source}
    return {"cell_type": "markdown", "metadata": {}, "source": source}


new_cells = plan["new_cells"]
anchor_sub = new_cells[0]["insert_after_substr"]
idx = next(i for i, c in enumerate(nb["cells"]) if anchor_sub in _src(c))
print(f"anchor cell index = {idx} (contains {anchor_sub!r})")
for off, spec in enumerate(new_cells, start=1):
    nb["cells"].insert(idx + off, _new_cell(spec))
print(f"inserted {len(new_cells)} cells after anchor")

NB.write_text(json.dumps(nb, indent=1), encoding="utf-8")

# --- 3. validate
reload = json.loads(NB.read_text(encoding="utf-8"))
order = [c["cell_type"] for c in reload["cells"]]
ai = next(i for i, c in enumerate(reload["cells"]) if anchor_sub in _src(c))
after = [(_src(c)[:48].replace("\n", " ")) for c in reload["cells"][ai:ai + 4]]
print("\nJSON reload OK; cells after anchor:")
for a in after:
    print("  ·", a)

# compile the inserted v17 code cell
v17_code = next(_src(c) for c in reload["cells"]
                if c["cell_type"] == "code" and "run_v17(" in _src(c))
compile(v17_code, "<v17_cell>", "exec")
print("\nv17 code cell compiles OK")

# legacy v15 divider must come AFTER the new cells
legacy_i = next(i for i, c in enumerate(reload["cells"]) if "Legacy: v15" in _src(c))
v17md_i = next(i for i, c in enumerate(reload["cells"]) if "Calibrated Coordinate-Ascent (alternative" in _src(c))
assert ai < v17md_i < legacy_i, f"order wrong: anchor={ai} v17md={v17md_i} legacy={legacy_i}"
print(f"order OK: provenance({ai}) < v17({v17md_i}) < legacy-v15({legacy_i})")
print("\nPATCH APPLIED + VALIDATED")
