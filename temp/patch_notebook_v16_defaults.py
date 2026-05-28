"""
patch_notebook_v16_defaults.py
-------------------------------
Applies v16 defaults to the Run 5 cells in optimize.ipynb:

1. Selector cell (cell containing '=== Run 5 ... multi-asset pool selection ==='):
   - SELECTED_GROUPS  -> ["INDICES_US", "INDICES_GLOBAL"]
   - SELECTED_TIMEFRAMES -> ["1D"]
   - VOLUME_POLICY    -> "price_only"  (already default, kept for explicitness)

2. Launch cell (cell containing '=== Run 5 ... build pooled folds + launch ==='):
   - N_TRIALS -> 250  # per side

3. Insert a new diagnostic cell immediately AFTER the selector cell and BEFORE
   the launch cell.

Usage:
    python temp/patch_notebook_v16_defaults.py
"""

import json
import re
import sys
from pathlib import Path

NOTEBOOK_PATH = Path("optimize.ipynb")

# ---------------------------------------------------------------------------
# Diagnostic cell source (verbatim as specified)
# ---------------------------------------------------------------------------
DIAGNOSTIC_SOURCE = """\
# === Run 5 / v16 — pre-run pool sufficiency diagnostic ===
# Run this BEFORE the launch cell. It tells you whether your selected pool has
# enough structural events per OOS fold for a meaningful optimization.
from src.scoring import add_pivot_labels, REFERENCE_N
from src.pooled_validation import (
    StreamData, build_calendar_folds, load_stream_frame, apply_volume_policy,
)
_TF_SECONDS = {"1D": 86400.0, "1W": 604800.0, "60": 3600.0, "240": 14400.0}

if not STREAMS:
    print("WARNING: 0 streams resolved — export the index CSVs into data/raw/ "
          "named {TICKER}_{TF}_*.csv (e.g. SPX_1D_*.csv, NDX_1D_*.csv).")
else:
    _sd = []
    for s in STREAMS:
        df = load_stream_frame(s.path)
        df, keep = apply_volume_policy(df, policy=VOLUME_POLICY)
        if not keep:
            print(f"  drop {s.stream_id}: volume policy excluded it"); continue
        add_pivot_labels(df)
        _sd.append(StreamData(stream=s, df=df, bar_seconds=_TF_SECONDS[s.timeframe]))
    _folds = build_calendar_folds(_sd)
    print(f"resolved streams = {[s.stream_id for s in STREAMS]}")
    print(f"calendar folds   = {len(_folds)}; streams/fold = {[len(f) for f in _folds]}")
    import numpy as np
    for side, lbl in (("HIGH", 1), ("LOW", -1)):
        per_fold = []
        for f in _folds:
            tot = sum(int((sl.df_oos[f'pivot_N{REFERENCE_N}'] == lbl).sum()) for sl in f)
            per_fold.append(tot)
        n_inf = sum(1 for x in per_fold if x > 0)
        print(f"  {side}: OOS structural pivots/fold = {per_fold} | "
              f"informative folds = {n_inf}/{len(_folds)} | "
              f"mean = {np.mean(per_fold):.1f}")
    print("\\nRule of thumb: aim for >= ~5 informative folds with mean >= ~3 pivots/fold "
          "per side. If thin, add more index exports (more streams) before launching.")\
"""


def make_code_cell(source: str) -> dict:
    """Create a minimal Jupyter code cell dict."""
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source,
    }


def patch_selector_cell(source: str) -> str:
    """Replace SELECTED_GROUPS, SELECTED_TIMEFRAMES, VOLUME_POLICY lines."""
    # SELECTED_GROUPS
    source = re.sub(
        r'SELECTED_GROUPS\s*=\s*\[.*?\]',
        'SELECTED_GROUPS = ["INDICES_US", "INDICES_GLOBAL"]',
        source,
    )
    # SELECTED_TIMEFRAMES
    source = re.sub(
        r'SELECTED_TIMEFRAMES\s*=\s*\[.*?\]',
        'SELECTED_TIMEFRAMES = ["1D"]',
        source,
    )
    # VOLUME_POLICY (keep trailing comment if present)
    source = re.sub(
        r'VOLUME_POLICY\s*=\s*"[^"]*"',
        'VOLUME_POLICY = "price_only"',
        source,
    )
    return source


def patch_launch_cell(source: str) -> str:
    """Replace N_TRIALS assignment."""
    source = re.sub(
        r'N_TRIALS\s*=\s*\d+',
        'N_TRIALS = 250',
        source,
    )
    return source


def main() -> None:
    if not NOTEBOOK_PATH.exists():
        print(f"ERROR: {NOTEBOOK_PATH} not found. Run from repo root.", file=sys.stderr)
        sys.exit(1)

    with open(NOTEBOOK_PATH, encoding="utf-8") as fh:
        nb = json.load(fh)

    cells = nb["cells"]

    # ------------------------------------------------------------------
    # Find the three Run 5 cells by matching unique substrings in source
    # ------------------------------------------------------------------
    SELECTOR_MARKER = "multi-asset pool selection"
    LAUNCH_MARKER = "build pooled folds + launch"
    REPRO_MARKER = "record provenance for reproducibility"

    selector_idx = launch_idx = repro_idx = None

    for i, cell in enumerate(cells):
        src = "".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]
        if SELECTOR_MARKER in src:
            selector_idx = i
        elif LAUNCH_MARKER in src:
            launch_idx = i
        elif REPRO_MARKER in src:
            repro_idx = i

    # Validate discovery
    missing = []
    if selector_idx is None:
        missing.append(f"selector cell (marker: '{SELECTOR_MARKER}')")
    if launch_idx is None:
        missing.append(f"launch cell (marker: '{LAUNCH_MARKER}')")
    if repro_idx is None:
        missing.append(f"repro cell (marker: '{REPRO_MARKER}')")

    if missing:
        print("ERROR: Could not locate cells by marker text:", file=sys.stderr)
        for m in missing:
            print(f"  - {m}", file=sys.stderr)
        print("\nActual cell sources:", file=sys.stderr)
        for i, c in enumerate(cells):
            s = ("".join(c["source"]) if isinstance(c["source"], list) else c["source"])[:80]
            print(f"  Cell {i}: {s!r}", file=sys.stderr)
        sys.exit(2)

    print(f"Found selector cell at index {selector_idx}")
    print(f"Found launch cell at index   {launch_idx}")
    print(f"Found repro cell at index    {repro_idx}")

    # Sanity: selector should immediately precede launch (before diagnostic insert)
    if launch_idx != selector_idx + 1:
        print(
            f"WARNING: launch cell ({launch_idx}) is not immediately after "
            f"selector cell ({selector_idx}). Proceeding anyway.",
        )

    # ------------------------------------------------------------------
    # 1. Patch selector cell
    # ------------------------------------------------------------------
    sel_cell = cells[selector_idx]
    sel_src = "".join(sel_cell["source"]) if isinstance(sel_cell["source"], list) else sel_cell["source"]
    sel_src_patched = patch_selector_cell(sel_src)
    cells[selector_idx]["source"] = sel_src_patched
    print("Patched selector cell defaults.")

    # ------------------------------------------------------------------
    # 2. Patch launch cell (N_TRIALS)
    # ------------------------------------------------------------------
    # After inserting diagnostic below, launch_idx will shift by 1 —
    # patch it now before the insert.
    launch_cell = cells[launch_idx]
    launch_src = "".join(launch_cell["source"]) if isinstance(launch_cell["source"], list) else launch_cell["source"]
    launch_src_patched = patch_launch_cell(launch_src)
    cells[launch_idx]["source"] = launch_src_patched
    print("Patched launch cell N_TRIALS = 250.")

    # ------------------------------------------------------------------
    # 3. Check whether diagnostic cell already exists
    # ------------------------------------------------------------------
    DIAG_MARKER = "pre-run pool sufficiency diagnostic"
    diag_already_present = any(
        DIAG_MARKER in ("".join(c["source"]) if isinstance(c["source"], list) else c["source"])
        for c in cells
    )

    if diag_already_present:
        print("Diagnostic cell already present — skipping insertion.")
    else:
        # Insert AFTER selector cell (which is at selector_idx)
        insert_at = selector_idx + 1
        diag_cell = make_code_cell(DIAGNOSTIC_SOURCE)
        cells.insert(insert_at, diag_cell)
        print(f"Inserted diagnostic cell at index {insert_at} "
              f"(between selector and launch cells).")

    # ------------------------------------------------------------------
    # 4. Write back
    # ------------------------------------------------------------------
    with open(NOTEBOOK_PATH, "w", encoding="utf-8") as fh:
        json.dump(nb, fh, ensure_ascii=False, indent=1)
    print(f"\nNotebook written: {NOTEBOOK_PATH}")


if __name__ == "__main__":
    main()
