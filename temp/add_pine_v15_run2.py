"""Add 'SPX 1D 2026-05-23 V15 Run 2' preset (Scorer v3 winners).

HIGH = trial 229 (score 0.0679, deflated 0.0667)
       use_edge_voting=False, K=54 (irrelevant since edge is off),
       ER_directional + trend-only vote, no other gates.
LOW  = trial 130 (score 0.1441, deflated 0.1415)
       use_edge_voting=True, K=48,
       ER_dir + trend + momentum + momentum_velocity + volatility,
       GJR/HAR OFF.

Notes:
- HIGH side of this run showed holdout degradation (IS 0.065 → OOS 0.032).
  Worth visual inspection before declaring it the new HIGH-side preset.
- LOW side generalizes cleanly to holdout (IS 0.189 ≈ OOS 0.194) and is the
  clear keeper from Run 2.
"""
from pathlib import Path
import re

PINE = Path("pine/speculatores_v15_presets_gold.pine")
PRESET_NAME = "SPX 1D 2026-05-23 V15 Run 2"
PRESET_FLAG = "is_spx_20260523_v15_run2"

HIGH = dict(
    S_detect=24, scale_start=25, scale_end=125, scale_step=14,
    min_duration=1, cooldown_bars=14, price_gate_lb=34, vola_range_len=166,
    er_period=8, confirm_count=4, pivot_drift_lookback=2,
    pivot_drift_confirm_bias=0, pct_extreme=0.7868, min_agreement=0.3164,
    dur_extreme_pct=0.5269, vol_surge_thresh=1.6376, scale_div_thresh=0.1073,
    slope_thresh=0.0160, vola_high_pct=0.9335, pivot_drift_thresh=0.0272,
    pivot_drift_gate_mult=8.1146, momentum_velocity_thresh=0.0118,
    gjr_vote_thresh=0.4001, har_vote_thresh=0.0637,
    er_directional="true", use_trend="true", use_volume="false",
    use_momentum="false", use_momentum_velocity="false", use_volatility="false",
    use_er_gate="false", use_gjr_asym="false", use_har_vol="false",
    vola_method='"StdDev"', momentum_velocity_mode='"Reversal"',
)
LOW = dict(
    S_detect=11, scale_start=12, scale_end=105, scale_step=17,
    min_duration=5, cooldown_bars=15, price_gate_lb=43, vola_range_len=108,
    er_period=9, confirm_count=1, pivot_drift_lookback=9,
    pivot_drift_confirm_bias=0, pct_extreme=0.7628, min_agreement=0.1752,
    dur_extreme_pct=0.5550, vol_surge_thresh=1.8823, scale_div_thresh=0.2181,
    slope_thresh=0.4762, vola_high_pct=0.8497, pivot_drift_thresh=0.0199,
    pivot_drift_gate_mult=7.1884, momentum_velocity_thresh=0.0400,
    gjr_vote_thresh=0.2633, har_vote_thresh=0.3027,
    er_directional="true", use_trend="true", use_volume="false",
    use_momentum="true", use_momentum_velocity="true", use_volatility="true",
    use_er_gate="false", use_gjr_asym="false", use_har_vol="false",
    vola_method='"Intraday"', momentum_velocity_mode='"Trend"',
)

text = PINE.read_text(encoding="utf-8")

# 1. Header comment — anchored on the GRP_PRESET declaration so we insert above it
anchor = 'string GRP_PRESET = "Preset"'
new_header_lines = [
    "// SPX 1D 2026-05-23 V15 Run 2 (Scorer v3 winners):",
    "//   HIGH: trial 229, score 0.0679 (deflated 0.0667). use_edge_voting=False.",
    "//         er_directional + trend-only vote. No other gates.",
    "//         CAUTION: holdout IS=0.065 -> OOS=0.032 (~50% degradation on 2004-2026).",
    "//   LOW:  trial 130, score 0.1441 (deflated 0.1415). use_edge_voting=True, K=48.",
    "//         er_dir + trend + mom + mom_vel + vola consensus. GJR/HAR off.",
    "//         Holdout IS=0.189 ~= OOS=0.194 -- generalizes cleanly to modern regime.",
    "//   Scorer v3 = Hungarian matching + bootstrap-CI LCB + stable reference scale N=50.",
    "",
    anchor,
]
assert anchor in text, "GRP_PRESET anchor not found"
text = text.replace(anchor, "\n".join(new_header_lines), 1)

# 2. Dropdown — insert Run 2 before Run 1 Selective
old_dropdown = '"SPX 1D 2026-05-23 V15 Run 1", "SPX 1D 2026-05-23 V15 Run 1 Selective", "Legacy V11 Gold"'
new_dropdown = (
    '"SPX 1D 2026-05-23 V15 Run 1", '
    '"SPX 1D 2026-05-23 V15 Run 1 Selective", '
    f'"{PRESET_NAME}", '
    '"Legacy V11 Gold"'
)
assert old_dropdown in text, "dropdown anchor not found"
text = text.replace(old_dropdown, new_dropdown, 1)

# 3. is_* flag
old_flag = 'bool is_spx_20260523_v15_run1_sel = preset == "SPX 1D 2026-05-23 V15 Run 1 Selective"'
new_flag = old_flag + f"\nbool {PRESET_FLAG} = preset == \"{PRESET_NAME}\""
assert old_flag in text, "flag anchor not found"
text = text.replace(old_flag, new_flag, 1)


# 4. Per-side ternary lines — same approach as the V15 Run 1 Selective patcher
def _val(side_dict, key):
    v = side_dict[key]
    if isinstance(v, float):
        return f"{v:.4f}"
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
low_keys = [(h.replace("_high", "_low"), key) for h, key in high_keys]

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
            new_line = (
                stripped[:last_colon]
                + f" : {PRESET_FLAG} ? {value}"
                + stripped[last_colon:]
                + "\n"
            )
            out_lines.append(new_line)
            matched = True
            hits += 1
            break
    if not matched:
        out_lines.append(line)
text = "".join(out_lines)
assert hits == len(high_keys) + len(low_keys), f"patched {hits} of {len(high_keys) + len(low_keys)}"

# 5. vola_low_pct group — keep 0.05 default
text = text.replace(
    "is_spx_20260523_v15_run1_sel ? 0.05",
    f"is_spx_20260523_v15_run1_sel or {PRESET_FLAG} ? 0.05",
)

# 6. Edge voting wiring: HIGH=false (K irrelevant), LOW=true K=48
edge_high_old = (
    'bool use_edge_voting_high = is_spx_20260523_pathA_5k_edge ? true '
    ': is_spx_20260523_v15_run1 ? true '
    ': is_spx_20260523_v15_run1_sel ? true : false'
)
edge_high_new = (
    'bool use_edge_voting_high = is_spx_20260523_pathA_5k_edge ? true '
    ': is_spx_20260523_v15_run1 ? true '
    ': is_spx_20260523_v15_run1_sel ? true '
    f': {PRESET_FLAG} ? false : false'
)
assert edge_high_old in text, "edge_high anchor not found"
text = text.replace(edge_high_old, edge_high_new, 1)

edge_low_old = (
    'bool use_edge_voting_low = is_spx_20260523_pathA_5k_edge ? true '
    ': is_spx_20260523_v15_run1 ? false '
    ': is_spx_20260523_v15_run1_sel ? true : false'
)
edge_low_new = (
    'bool use_edge_voting_low = is_spx_20260523_pathA_5k_edge ? true '
    ': is_spx_20260523_v15_run1 ? false '
    ': is_spx_20260523_v15_run1_sel ? true '
    f': {PRESET_FLAG} ? true : false'
)
assert edge_low_old in text, "edge_low anchor not found"
text = text.replace(edge_low_old, edge_low_new, 1)

ew_high_old = (
    'int edge_window_high = is_spx_20260523_pathA_5k_edge ? 5 '
    ': is_spx_20260523_v15_run1 ? 21 '
    ': is_spx_20260523_v15_run1_sel ? 21 : 5'
)
ew_high_new = (
    'int edge_window_high = is_spx_20260523_pathA_5k_edge ? 5 '
    ': is_spx_20260523_v15_run1 ? 21 '
    ': is_spx_20260523_v15_run1_sel ? 21 '
    f': {PRESET_FLAG} ? 54 : 5'
)
assert ew_high_old in text, "edge_window_high anchor not found"
text = text.replace(ew_high_old, ew_high_new, 1)

ew_low_old = (
    'int edge_window_low = is_spx_20260523_pathA_5k_edge ? 5 '
    ': is_spx_20260523_v15_run1 ? 29 '
    ': is_spx_20260523_v15_run1_sel ? 21 : 5'
)
ew_low_new = (
    'int edge_window_low = is_spx_20260523_pathA_5k_edge ? 5 '
    ': is_spx_20260523_v15_run1 ? 29 '
    ': is_spx_20260523_v15_run1_sel ? 21 '
    f': {PRESET_FLAG} ? 48 : 5'
)
assert ew_low_old in text, "edge_window_low anchor not found"
text = text.replace(ew_low_old, ew_low_new, 1)

PINE.write_text(text, encoding="utf-8")
print(f"Patched {hits} per-side ternary lines + 4 edge-voting ternaries")
print(f"Wrote {PINE} ({len(text.splitlines())} lines)")
