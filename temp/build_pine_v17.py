"""Build pine/speculatores_v17_presets_gold.pine from the v15 file.

V17 changed the OPTIMIZER (GPU CMA-ES batched search), never the detector
math — so every computation line is copied byte-identical (parity contract).
This is a REBRAND + UI DECLUTTER only:

  1. Title -> Speculatores V17.
  2. Default preset -> the v17-GPU Run 1 winner.
  3. Dropdown pruned to the curated set (Gold / v17-GPU Run 1 / Legacy V11).
     The pruned presets' is_* flags and ternary chains stay in the source —
     they are simply unreachable from the UI (always-false flags), which keeps
     the diff to the v15 engine reviewable and the parity surface ZERO.
  4. Header rewritten for v17 provenance.
"""
from pathlib import Path

SRC = Path("pine/speculatores_v15_presets_gold.pine")
DST = Path("pine/speculatores_v17_presets_gold.pine")

V17_PRESET = "INDICES 1D 2026-06-10 v17-GPU Run 1"

text = SRC.read_text(encoding="utf-8")

# 1. Title
old_title = ('indicator("Speculatores V15 Presets - Per-Side Regimes (edge voting)", '
             'overlay=true, max_bars_back=5000)')
new_title = ('indicator("Speculatores V17 Presets - Per-Side Regimes (GPU CMA-ES)", '
             'overlay=true, max_bars_back=5000)')
assert old_title in text
text = text.replace(old_title, new_title, 1)

# 2+3. Default preset + pruned dropdown. The full options list is replaced by
# the curated set; all is_* flags/ternaries for removed presets remain in code
# (unreachable, always false) so no computation chain is edited.
old_input = ('string preset = input.string("Gold 1D Current", "Config Preset", '
             'options=["Gold 1D Current", "DAX 1M Latest", '
             '"SPX 1D 2026-03-29 18:35 Trial 1225", '
             '"SPX 1D 2026-04-05 Parity High+Low", '
             '"SPX 1D 2026-04-05 Heuristic Structural", '
             '"SPX 1D 2026-04-05 20:59 Salvaged Best", '
             '"SPX 1D 2026-05-18 21:51 Path A", "SPX 1D 2026-05-23 Path A 5k", '
             '"SPX 1D 2026-05-23 Path A 5k Edge", "SPX 1D 2026-05-23 V15 Run 1", '
             '"SPX 1D 2026-05-23 V15 Run 1 Selective", "SPX 1D 2026-05-23 V15 Run 2", '
             '"SPX 1D 2026-05-24 V15 Run 3", '
             f'"{V17_PRESET}", "Legacy V11 Gold"], group=GRP_PRESET)')
new_input = (f'string preset = input.string("{V17_PRESET}", "Config Preset", '
             f'options=["{V17_PRESET}", "Gold 1D Current", "Legacy V11 Gold"], '
             'group=GRP_PRESET)')
assert old_input in text, "dropdown anchor drifted — check the v15 options list"
text = text.replace(old_input, new_input, 1)

# 4. Header block: insert a v17 provenance banner directly under the title.
banner = """
// ============================================================================
// SPECULATORES V17 — same detection engine as V15 (byte-identical math; the
// v17 release changed the OPTIMIZER, not the detector), rebranded + decluttered.
// Curated presets:
//   - INDICES 1D 2026-06-10 v17-GPU Run 1 (DEFAULT): GPU CMA-ES batched search
//     on the SPX+NDX+DAX 1D pool. LOW = stable cross-seed plateau (deflated
//     LCB 0.027-0.032), gates REJECT pending box-edge bound widening —
//     INSPECTION-grade. HIGH = hypothesis-grade only (~27-event ceiling),
//     MONITOR-only.
//   - Gold 1D Current: the long-standing gold reference preset.
//   - Legacy V11 Gold: historical compatibility.
// Legacy v15-era presets remain in source (unreachable) for diff-parity with
// pine/speculatores_v15_presets_gold.pine.
// ============================================================================
"""
text = text.replace(new_title, new_title + "\n" + banner.strip("\n"), 1)

DST.write_text(text, encoding="utf-8")

# Sanity: engine lines identical except the documented edits.
import difflib
src_lines = SRC.read_text(encoding="utf-8").splitlines()
dst_lines = DST.read_text(encoding="utf-8").splitlines()
changed = [d for d in difflib.unified_diff(src_lines, dst_lines, lineterm="", n=0)
           if d.startswith(("+", "-")) and not d.startswith(("+++", "---"))]
removed = [d for d in changed if d.startswith("-")]
print(f"wrote {DST} ({len(dst_lines)} lines)")
print(f"diff vs v15: {len(removed)} lines changed, {len(changed) - len(removed)} added")
for d in removed:
    print("  CHANGED:", d[1:90])
