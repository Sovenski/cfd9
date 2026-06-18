"""Add 'INDICES 1D 2026-06-10 v17-GPU Run 1' preset to pine/speculatores_v15_presets_gold.pine.

Source: results/v17gpu_20260610_153006_v17gpu.json (seed 7, 50 generations,
6529 evals/side, SPX+NDX+DAX 1D pool, large-slice folds IS=0.20/OOS=0.15).
Values are FULL precision (repr round-trip), not 4-decimal rounding.
"""
from pathlib import Path
import re

PINE = Path("pine/speculatores_v15_presets_gold.pine")
PRESET_NAME = "INDICES 1D 2026-06-10 v17-GPU Run 1"
PRESET_FLAG = "is_indices_20260610_v17gpu_run1"

HIGH = dict(
    S_detect=12, scale_start=3, scale_end=270, scale_step=3,
    min_duration=13, cooldown_bars=9, price_gate_lb=23, vola_range_len=120,
    er_period=28, confirm_count=5, pivot_drift_lookback=5,
    pivot_drift_confirm_bias=1,
    pct_extreme=0.554436424188316, min_agreement=0.47057701572775845,
    dur_extreme_pct=0.6388478276878595, vol_surge_thresh=1.5,
    scale_div_thresh=0.23202978018671275, slope_thresh=0.22,
    vola_high_pct=0.92, pivot_drift_thresh=0.02970966456271708,
    pivot_drift_gate_mult=4.846518321894109,
    momentum_velocity_thresh=0.0109414285980165,
    gjr_vote_thresh=0.15, har_vote_thresh=0.15,
    er_directional="false", use_trend="false", use_volume="false",
    use_momentum="false", use_momentum_velocity="true", use_volatility="false",
    use_er_gate="false", use_gjr_asym="false", use_har_vol="false",
    vola_method='"ATR"', momentum_velocity_mode='"Reversal"',
)
LOW = dict(
    S_detect=26, scale_start=19, scale_end=250, scale_step=15,
    min_duration=2, cooldown_bars=7, price_gate_lb=58, vola_range_len=20,
    er_period=47, confirm_count=3, pivot_drift_lookback=10,
    pivot_drift_confirm_bias=0,
    pct_extreme=0.9898309362374363, min_agreement=0.10161243349676918,
    dur_extreme_pct=0.8481916445277483, vol_surge_thresh=2.2,
    scale_div_thresh=0.24889258429931752, slope_thresh=0.15,
    vola_high_pct=0.7099270056620911, pivot_drift_thresh=0.013284897982313939,
    pivot_drift_gate_mult=4.0, momentum_velocity_thresh=0.007715178292267143,
    gjr_vote_thresh=0.15, har_vote_thresh=0.15,
    er_directional="false", use_trend="false", use_volume="false",
    use_momentum="false", use_momentum_velocity="true", use_volatility="true",
    use_er_gate="false", use_gjr_asym="false", use_har_vol="false",
    vola_method='"ATR"', momentum_velocity_mode='"Reversal"',
)

text = PINE.read_text(encoding="utf-8")

# 1. Header comment block — insert above the V15 Run 3 entry
old_header = "// SPX 1D 2026-05-24 V15 Run 3 (Scorer v4 winners):"
new_header_lines = [
    "// INDICES 1D 2026-06-10 v17-GPU Run 1 (CMA-ES batched GPU search, SPX+NDX+DAX 1D pool):",
    "//   LOW:  raw LCB 0.0978, DEFLATED 0.0271 (6529 trials) — STABLE PLATEAU: two",
    "//         independent seeds (42/10gen, 7/50gen) converge on the same region",
    "//         ('few scales agree, but at near-total extremes'). Gates: REJECT due to",
    "//         BOX-EDGE pins (min_agreement at lo bound 0.10, pct_extreme at hi bound",
    "//         0.99 — the TIGHT end, i.e. the search box is too small, not gaming).",
    "//         Pending bound-widening rerun. Use for visual inspection, not live.",
    "//   HIGH: deflated LCB 0.0 — hypothesis-grade ONLY (~27 validatable events;",
    "//         winner at 99.98th pctile of own population = selection effect).",
    "//         MONITOR-ONLY by design.",
    "",
    "// SPX 1D 2026-05-24 V15 Run 3 (Scorer v4 winners):",
]
assert old_header in text
text = text.replace(old_header, "\n".join(new_header_lines), 1)

# 2. Dropdown option
old_dropdown = '"SPX 1D 2026-05-24 V15 Run 3", "Legacy V11 Gold"'
assert old_dropdown in text
new_dropdown = f'"SPX 1D 2026-05-24 V15 Run 3", "{PRESET_NAME}", "Legacy V11 Gold"'
text = text.replace(old_dropdown, new_dropdown, 1)

# 3. is_* flag
old_flag = 'bool is_spx_20260524_v15_run3 = preset == "SPX 1D 2026-05-24 V15 Run 3"'
assert old_flag in text
text = text.replace(
    old_flag,
    old_flag + f'\nbool {PRESET_FLAG} = preset == "{PRESET_NAME}"',
    1,
)


# 4. Append values to all 70 per-side ternaries (full-precision floats)
def _val(side_dict, key):
    v = side_dict[key]
    if isinstance(v, float):
        return repr(v)  # shortest round-trip — parses to the exact double
    return str(v)


high_keys = [
    ("S_detect_high", "S_detect"), ("scale_start_high", "scale_start"),
    ("scale_end_high", "scale_end"), ("scale_step_high", "scale_step"),
    ("min_duration_high", "min_duration"), ("cooldown_bars_high", "cooldown_bars"),
    ("price_gate_lb_high", "price_gate_lb"), ("vola_range_len_high", "vola_range_len"),
    ("er_period_high", "er_period"), ("pct_extreme_high", "pct_extreme"),
    ("min_agreement_high", "min_agreement"), ("dur_extreme_pct_high", "dur_extreme_pct"),
    ("confirm_count_high", "confirm_count"), ("vol_surge_thresh_high", "vol_surge_thresh"),
    ("scale_div_thresh_high", "scale_div_thresh"), ("slope_thresh_high", "slope_thresh"),
    ("vola_high_pct_high", "vola_high_pct"),
    ("pivot_drift_lookback_high", "pivot_drift_lookback"),
    ("pivot_drift_thresh_high", "pivot_drift_thresh"),
    ("pivot_drift_gate_mult_high", "pivot_drift_gate_mult"),
    ("pivot_drift_confirm_bias_high", "pivot_drift_confirm_bias"),
    ("use_er_gate_high", "use_er_gate"), ("er_directional_high", "er_directional"),
    ("use_trend_high", "use_trend"), ("use_volume_high", "use_volume"),
    ("use_momentum_high", "use_momentum"),
    ("use_momentum_velocity_high", "use_momentum_velocity"),
    ("use_volatility_high", "use_volatility"),
    ("use_gjr_asym_high", "use_gjr_asym"), ("use_har_vol_high", "use_har_vol"),
    ("gjr_vote_thresh_high", "gjr_vote_thresh"),
    ("har_vote_thresh_high", "har_vote_thresh"),
    ("momentum_velocity_mode_high", "momentum_velocity_mode"),
    ("momentum_velocity_thresh_high", "momentum_velocity_thresh"),
    ("vola_method_high", "vola_method"),
]
# Suffix-only replace: "vola_high_pct_high" must become "vola_high_pct_low",
# NOT "vola_low_pct_low" (a full replace() mangled it — 2026-06-11 audit bug).
low_keys = [(h[: -len("_high")] + "_low", key) for h, key in high_keys]

out_lines = []
hits = 0
for line in text.splitlines(keepends=True):
    stripped = line.rstrip("\n").rstrip("\r")
    matched = False
    for var_name, key in high_keys + low_keys:
        if re.match(rf"^\s*(int|float|bool|string)\s+{re.escape(var_name)}\s*=", stripped):
            value = _val(HIGH if var_name.endswith("_high") else LOW, key)
            last_colon = stripped.rfind(" : ")
            if last_colon < 0:
                break
            out_lines.append(
                stripped[:last_colon]
                + f" : {PRESET_FLAG} ? {value}"
                + stripped[last_colon:]
                + "\n"
            )
            matched = True
            hits += 1
            break
    if not matched:
        out_lines.append(line)
text = "".join(out_lines)
assert hits == len(high_keys) + len(low_keys), f"patched {hits} of {len(high_keys) + len(low_keys)}"

# 5. vola_low_pct groups (Pine-only knob, not in Params): HIGH joins the 0.05
#    or-chain; LOW takes the gold default 0.07 (the seed this run derives from).
old_vlph = "is_spx_20260524_v15_run3 ? 0.05"
assert old_vlph in text
text = text.replace(old_vlph, f"is_spx_20260524_v15_run3 or {PRESET_FLAG} ? 0.05", 1)
old_vlpl = "is_spx_20260405_2059_salvaged_best ? 0.07"
assert old_vlpl in text
text = text.replace(old_vlpl, f"is_spx_20260405_2059_salvaged_best or {PRESET_FLAG} ? 0.07", 1)

# 6. Edge voting: this preset uses use_edge_voting=false / edge_window=5 on BOTH
#    sides — identical to the fall-through defaults, so NO ternary edit needed.

PINE.write_text(text, encoding="utf-8")
print(f"Patched {hits} per-side ternary lines (+dropdown, flag, header, 2 vola_low_pct chains)")
print(f"Wrote {PINE} ({len(text.splitlines())} lines)")
