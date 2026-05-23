"""Add 'SPX 1D 2026-05-24 V15 Run 3' preset (Scorer v4 winners).

HIGH = trial 167 (score 0.0656, deflated 0.0635)
       use_edge_voting=True, K=28
       use_trend + use_volume + use_momentum_velocity + use_volatility
       vola_method=StdDev, momentum_velocity_mode=Reversal
       CAUTION: holdout IS=0.025 -> OOS=0.000 (HIGH is the noise floor on SPX 1D).

LOW  = trial 169 (score 0.0708, deflated 0.0667)
       use_edge_voting=False
       use_trend + use_volume + use_momentum_velocity + use_er_gate
       vola_method=ATR, momentum_velocity_mode=Trend
       Holdout IS=0.029 -> OOS=0.089 (OOS 3x better than IS). Stability mean/best=0.997.
       This is the real keeper from v4.

Notes:
- v4 used the nested-scale structural oracle [50, 100, 200] which catches
  7/7 famous SPX lows (2002, 2009, 2011, 2016, 2018, 2020, 2022).
- HIGH stays in the preset for chart inspection / comparison, but expect
  it to fire rarely and possibly off-target. The 2 holdout HIGH signals
  hit zero structural pivots.
"""
from pathlib import Path
import re

PINE = Path("pine/speculatores_v15_presets_gold.pine")
PRESET_NAME = "SPX 1D 2026-05-24 V15 Run 3"
PRESET_FLAG = "is_spx_20260524_v15_run3"

HIGH = dict(
    S_detect=44, scale_start=10, scale_end=179, scale_step=20,
    min_duration=20, cooldown_bars=14, price_gate_lb=97, vola_range_len=120,
    er_period=58, confirm_count=2, pivot_drift_lookback=14,
    pivot_drift_confirm_bias=0, pct_extreme=0.7440, min_agreement=0.3703,
    dur_extreme_pct=0.6365, vol_surge_thresh=1.5240, scale_div_thresh=0.1563,
    slope_thresh=0.2371, vola_high_pct=0.5283, pivot_drift_thresh=0.0478,
    pivot_drift_gate_mult=1.0751, momentum_velocity_thresh=0.0295,
    gjr_vote_thresh=0.1227, har_vote_thresh=0.3990,
    er_directional="false", use_trend="true", use_volume="true",
    use_momentum="false", use_momentum_velocity="true", use_volatility="true",
    use_er_gate="false", use_gjr_asym="false", use_har_vol="false",
    vola_method='"StdDev"', momentum_velocity_mode='"Reversal"',
)
LOW = dict(
    S_detect=52, scale_start=13, scale_end=215, scale_step=5,
    min_duration=4, cooldown_bars=3, price_gate_lb=60, vola_range_len=184,
    er_period=36, confirm_count=2, pivot_drift_lookback=4,
    pivot_drift_confirm_bias=0, pct_extreme=0.7297, min_agreement=0.5595,
    dur_extreme_pct=0.6241, vol_surge_thresh=1.8537, scale_div_thresh=0.3094,
    slope_thresh=0.3338, vola_high_pct=0.9639, pivot_drift_thresh=0.0364,
    pivot_drift_gate_mult=5.4354, momentum_velocity_thresh=0.0127,
    gjr_vote_thresh=0.3361, har_vote_thresh=0.4080,
    er_directional="true", use_trend="true", use_volume="true",
    use_momentum="false", use_momentum_velocity="true", use_volatility="false",
    use_er_gate="true", use_gjr_asym="false", use_har_vol="false",
    vola_method='"ATR"', momentum_velocity_mode='"Trend"',
)

text = PINE.read_text(encoding="utf-8")

# 1. Header comment — anchored on GRP_PRESET declaration
anchor = 'string GRP_PRESET = "Preset"'
new_header_lines = [
    "// SPX 1D 2026-05-24 V15 Run 3 (Scorer v4 winners):",
    "//   HIGH: trial 167, score 0.0656 (deflated 0.0635). use_edge_voting=True, K=28.",
    "//         trend + volume + mom_vel + vola consensus.",
    "//         CAUTION: holdout IS=0.025 -> OOS=0.000. HIGH is the noise floor on SPX 1D",
    "//         in this feature space. Keep on chart for comparison, not signal.",
    "//   LOW:  trial 169, score 0.0708 (deflated 0.0667). use_edge_voting=False.",
    "//         trend + volume + mom_vel + er_gate consensus. ATR vola, Trend mom_vel mode.",
    "//         Holdout IS=0.029 -> OOS=0.089 (OOS 3x better than IS).",
    "//         Stability local_mean/best = 99.7%. This is the v4 keeper.",
    "//   Scorer v4 oracle = structural-nest [50, 100, 200], catches 7/7 famous SPX lows.",
    "",
    anchor,
]
assert anchor in text, "GRP_PRESET anchor not found"
text = text.replace(anchor, "\n".join(new_header_lines), 1)

# 2. Dropdown — insert Run 3 between Run 2 and Legacy V11 Gold
old_dropdown = '"SPX 1D 2026-05-23 V15 Run 2", "Legacy V11 Gold"'
new_dropdown = (
    '"SPX 1D 2026-05-23 V15 Run 2", '
    f'"{PRESET_NAME}", '
    '"Legacy V11 Gold"'
)
assert old_dropdown in text, "dropdown anchor not found"
text = text.replace(old_dropdown, new_dropdown, 1)

# 3. is_* flag
old_flag = 'bool is_spx_20260523_v15_run2 = preset == "SPX 1D 2026-05-23 V15 Run 2"'
new_flag = old_flag + f"\nbool {PRESET_FLAG} = preset == \"{PRESET_NAME}\""
assert old_flag in text, "flag anchor not found"
text = text.replace(old_flag, new_flag, 1)


# 4. Per-side ternary lines
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
    "is_spx_20260523_v15_run2 ? 0.05",
    f"is_spx_20260523_v15_run2 or {PRESET_FLAG} ? 0.05",
)

# 6. Edge voting wiring: HIGH edge=true K=28, LOW edge=false K=11
edge_high_old = (
    'bool use_edge_voting_high = is_spx_20260523_pathA_5k_edge ? true '
    ': is_spx_20260523_v15_run1 ? true '
    ': is_spx_20260523_v15_run1_sel ? true '
    ': is_spx_20260523_v15_run2 ? false : false'
)
edge_high_new = (
    'bool use_edge_voting_high = is_spx_20260523_pathA_5k_edge ? true '
    ': is_spx_20260523_v15_run1 ? true '
    ': is_spx_20260523_v15_run1_sel ? true '
    ': is_spx_20260523_v15_run2 ? false '
    f': {PRESET_FLAG} ? true : false'
)
assert edge_high_old in text, "edge_high anchor not found"
text = text.replace(edge_high_old, edge_high_new, 1)

edge_low_old = (
    'bool use_edge_voting_low = is_spx_20260523_pathA_5k_edge ? true '
    ': is_spx_20260523_v15_run1 ? false '
    ': is_spx_20260523_v15_run1_sel ? true '
    ': is_spx_20260523_v15_run2 ? true : false'
)
edge_low_new = (
    'bool use_edge_voting_low = is_spx_20260523_pathA_5k_edge ? true '
    ': is_spx_20260523_v15_run1 ? false '
    ': is_spx_20260523_v15_run1_sel ? true '
    ': is_spx_20260523_v15_run2 ? true '
    f': {PRESET_FLAG} ? false : false'
)
assert edge_low_old in text, "edge_low anchor not found"
text = text.replace(edge_low_old, edge_low_new, 1)

ew_high_old = (
    'int edge_window_high = is_spx_20260523_pathA_5k_edge ? 5 '
    ': is_spx_20260523_v15_run1 ? 21 '
    ': is_spx_20260523_v15_run1_sel ? 21 '
    ': is_spx_20260523_v15_run2 ? 54 : 5'
)
ew_high_new = (
    'int edge_window_high = is_spx_20260523_pathA_5k_edge ? 5 '
    ': is_spx_20260523_v15_run1 ? 21 '
    ': is_spx_20260523_v15_run1_sel ? 21 '
    ': is_spx_20260523_v15_run2 ? 54 '
    f': {PRESET_FLAG} ? 28 : 5'
)
assert ew_high_old in text, "edge_window_high anchor not found"
text = text.replace(ew_high_old, ew_high_new, 1)

ew_low_old = (
    'int edge_window_low = is_spx_20260523_pathA_5k_edge ? 5 '
    ': is_spx_20260523_v15_run1 ? 29 '
    ': is_spx_20260523_v15_run1_sel ? 21 '
    ': is_spx_20260523_v15_run2 ? 48 : 5'
)
ew_low_new = (
    'int edge_window_low = is_spx_20260523_pathA_5k_edge ? 5 '
    ': is_spx_20260523_v15_run1 ? 29 '
    ': is_spx_20260523_v15_run1_sel ? 21 '
    ': is_spx_20260523_v15_run2 ? 48 '
    f': {PRESET_FLAG} ? 11 : 5'
)
assert ew_low_old in text, "edge_window_low anchor not found"
text = text.replace(ew_low_old, ew_low_new, 1)

PINE.write_text(text, encoding="utf-8")
print(f"Patched {hits} per-side ternary lines + 4 edge-voting ternaries")
print(f"Wrote {PINE} ({len(text.splitlines())} lines)")
