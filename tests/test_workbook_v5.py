"""T14 — workbook v5 tests (spec §5 workbook bullets, F7 Cell-6 legend).

Runs ``temp/build_h100_notebook.py`` as a subprocess (its self-check must
exit 0), loads the regenerated ``h100_v17_gpu.ipynb`` with nbformat and
asserts:

* cell count grew by exactly one (the new Cell 7);
* Cells 1-5 sources are BYTE-IDENTICAL to the pre-change builder output
  (sha256 literals pinned below — the Cell-5 interface is unchanged, §5);
* Cell 6 carries the extended scorer-v5 report (calibration summary, card
  legend WITH the F7 disclaimer, capture-ratio line, last-10-signals tail,
  §2.4 fold-count + §2.5 v4-LCB-incomparability callouts);
* the NEW Cell 7 writes the freshly built v17.5 pine file to Drive.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import nbformat
import pytest

from src.v17_card.calibration import IN_SAMPLE_DISCLAIMER

_REPO = Path(__file__).resolve().parents[1]
_NB = _REPO / "h100_v17_gpu.ipynb"

#: sha256 of cells 0-5 sources of the PRE-change builder output
#: (captured 2026-06-10 before the T14 edit — the unchanged interface, §5).
#: Cell 5 RE-PINNED 2026-06-11: commit d220871 ("chore(notebook): big-run
#: defaults — all-1D groups, 30 gens, sobol 256, seed 69") intentionally
#: changed the Cell-5 form defaults without updating this pin; the new hash
#: is the sha256 of THAT committed cell (verified byte-identical to
#: ``git show d220871:h100_v17_gpu.ipynb`` cells[5]).
#: Cell 5 RE-PINNED AGAIN 2026-06-11: holdout-and-pruning spec §B3
#: intentionally adds the RUN_PRUNED checkbox + the baseline/pruned
#: ``shape_variants`` (mom-vel off both sides) to Cell 5; the new content is
#: asserted in tests/test_shape_variants.py (test_cell5_run_pruned_*).
#: Cell 5 RE-PINNED AGAIN 2026-06-11: v5.1 anti-spray re-pricing lowers the
#: Cell-5 ``FIRING_CAP`` default 2.0 -> 1.0 (matches the run_v17_gpu default);
#: the new value is asserted below (test_cell5_firing_cap_is_v5_1_default).
#: Cell 5 RE-PINNED AGAIN 2026-06-11: feature-ablation adds the RUN_ABLATION
#: checkbox + build_ablation_variants wiring (leave-one-out, all votes on);
#: asserted in tests/test_shape_variants.py::test_cell5_run_ablation_*.
#: Cell 5 RE-PINNED AGAIN 2026-06-11: v18 production run defaults — gen30
#: seed211, ablation/pruned off (plan/v18-repair-spec.md Stage C, C2).
#: Cell 5 RE-PINNED 2026-06-12: RUN_V18_VARIANTS checkbox (baseline vs
#: repaired: gjr on/unclipped, drift requirable, mom-vel off LOW) — the seed-211
#: run never engaged the v18 repairs (use-flags frozen at seed); this is the fix.
_PRE_CHANGE_CELL_SHA256 = [
    "694cc61d0520b2b596488663a133a241b2acfcdaeb42eac1c22ecb1707796409",  # 0 md intro
    "021625f0c9aec826339563fccf713c8bdcdc33c049a82272f60a3633a753b2e6",  # 1 mount
    "d4de24c91407fa5f670fa35206227e933be1e782158e630cc7aae3d14389b0b6",  # 2 clone
    "79f036c65d62bbb6feb18f85f06d9dc4f47fccf954a329b74df22664b98ff17b",  # 3 settings
    "4ec943aa666a08fb9070aec11f2fdaaea233f70b9bff5608b8d8dfc8c7eadb83",  # 4 validate
    "afd86b0a1759c796924d1cc32e747a7f6c5d08d021b24a99059ecda4b58ef353",  # 5 run (v18 production: gen30, seed211, ablation/pruned off)
]


def test_cell5_firing_cap_is_v5_1_default(notebook):
    """v5.1 re-pricing: the Cell-5 form default matches run_v17_gpu (1.0)."""
    src = _src(notebook.cells[5])
    assert "FIRING_CAP = 1.0" in src


def _src(cell) -> str:
    s = cell["source"]
    return s if isinstance(s, str) else "".join(s)


@pytest.fixture(scope="module")
def notebook():
    res = subprocess.run(
        [sys.executable, str(_REPO / "temp" / "build_h100_notebook.py")],
        cwd=_REPO, capture_output=True, text=True)
    assert res.returncode == 0, f"builder self-check failed:\n{res.stderr}"
    nb = nbformat.read(_NB, as_version=4)
    nbformat.validate(nb)
    return nb


def test_cell_count_grew_by_exactly_one(notebook):
    assert len(notebook.cells) == 8  # was 7 (md + cells 1-6)


def test_cells_1_to_5_byte_identical_to_pre_change(notebook):
    for i, expected in enumerate(_PRE_CHANGE_CELL_SHA256):
        got = hashlib.sha256(_src(notebook.cells[i]).encode()).hexdigest()
        assert got == expected, f"cell {i} source changed (interface frozen)"


def test_cell6_extended_v5_report(notebook):
    src = _src(notebook.cells[6])
    assert "SCORER v5" in src and "CALIBRATION" in src
    assert "capture_ratio" in src
    assert "signal_cards" in src and "[-10:]" in src  # last-10 table tail
    assert IN_SAMPLE_DISCLAIMER in src  # F7, verbatim in the card legend
    assert "conditional on match" in src  # R3 label
    # §2.4 / §2.5 report callouts
    assert "fold counts" in src and "v4" in src
    assert "INCOMPARABLE" in src


def test_cell7_writes_v17_5_pine_to_drive(notebook):
    src = _src(notebook.cells[7])
    assert "optional" in src.lower()  # marked optional in its header
    assert "build_pine_v17_5" in src
    assert "speculatores_v17_5_signalcard.pine" in src
    assert "RESULTS_DIR" in src  # Drive destination
    assert "out['_written']" in src  # built from THIS run's JSON


def test_all_pure_python_cells_compile(notebook):
    compiled = 0
    for c in notebook.cells:
        if c.cell_type != "code":
            continue
        src = _src(c)
        if any(tok in src for tok in ("!git", "!pip", "!python", "%cd",
                                      "get_ipython", "google.colab")):
            continue
        compile(src, "<cell>", "exec")
        compiled += 1
    assert compiled >= 4  # cells 5, 6, 7 + validate cell? at least the v5 ones


def test_notebook_json_loads_cleanly():
    json.loads(_NB.read_text(encoding="utf-8"))
