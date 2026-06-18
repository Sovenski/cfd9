"""Add the v5 run winner preset to the calibrated v17.5 pine (Downloads copy).

Source run: v17gpu_20260610_212938 (INDICES 1D, scorer v5, gens 10, A100).
HIGH winner from sides.high.best_params; LOW winner from sides.low.best_params
(per-side winners combined into one preset). Set as the dropdown DEFAULT so the
TV export pairs the v5 winner's signals with the v5 calibration block.
"""
from pathlib import Path
import re
import sys

PINE = Path(sys.argv[1] if len(sys.argv) > 1
            else r"C:\Users\kuben\Downloads\speculatores_v17_5_signalcard.pine")
OUT = Path(r"C:\Users\kuben\Desktop\Projekte\cfd9\temp\speculatores_v17_5_signalcard_v5preset.pine")
PRESET_NAME = "INDICES 1D 2026-06-10 v5 Run"
PRESET_FLAG = "is_indices_20260610_v5_run"

HIGH = dict(
    S_detect=12, scale_start=3, scale_end=270, scale_step=3,
    min_duration=13, cooldown_bars=9, price_gate_lb=23, vola_range_len=120,
    er_period=28, confirm_count=5, pivot_drift_lookback=5,
    pivot_drift_confirm_bias=1,
    pct_extreme=0.6449214841548585, min_agreement=0.10235877088821599,
    dur_extreme_pct=0.3067583081621106, vol_surge_thresh=1.5,
    scale_div_thresh=0.5648265369823214, slope_thresh=0.22,
    vola_high_pct=0.92, pivot_drift_thresh=0.033380017630768956,
    pivot_drift_gate_mult=8.126706097454868,
    momentum_velocity_thresh=0.000406850442654248,
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
    pct_extreme=0.7089677498942575, min_agreement=0.31741163082602114,
    dur_extreme_pct=0.5054146427804495, vol_surge_thresh=2.2,
    scale_div_thresh=0.20622631913042141, slope_thresh=0.15,
    vola_high_pct=0.7609491744714391, pivot_drift_thresh=0.015082698670489024,
    pivot_drift_gate_mult=4.0, momentum_velocity_thresh=0.0005828514910812214,
    gjr_vote_thresh=0.15, har_vote_thresh=0.15,
    er_directional="false", use_trend="false", use_volume="false",
    use_momentum="false", use_momentum_velocity="true", use_volatility="true",
    use_er_gate="false", use_gjr_asym="false", use_har_vol="false",
    vola_method='"ATR"', momentum_velocity_mode='"Reversal"',
)

text = PINE.read_text(encoding="utf-8")

# 1. Dropdown: add the preset and make it the DEFAULT.
old_input_default = 'input.string("INDICES 1D 2026-06-10 v17-GPU Run 1", "Config Preset"'
assert old_input_default in text
text = text.replace(old_input_default,
                    f'input.string("{PRESET_NAME}", "Config Preset"', 1)
old_opts = 'options=["INDICES 1D 2026-06-10 v17-GPU Run 1", "Gold 1D Current", "Legacy V11 Gold"]'
assert old_opts in text
text = text.replace(
    old_opts,
    f'options=["{PRESET_NAME}", "INDICES 1D 2026-06-10 v17-GPU Run 1", '
    '"Gold 1D Current", "Legacy V11 Gold"]', 1)

# 2. is_* flag after the v17gpu_run1 flag.
old_flag = 'bool is_indices_20260610_v17gpu_run1 = preset == "INDICES 1D 2026-06-10 v17-GPU Run 1"'
assert old_flag in text
text = text.replace(
    old_flag, old_flag + f'\nbool {PRESET_FLAG} = preset == "{PRESET_NAME}"', 1)


def _val(side_dict, key):
    v = side_dict[key]
    return repr(v) if isinstance(v, float) else str(v)


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
            out_lines.append(stripped[:last_colon]
                             + f" : {PRESET_FLAG} ? {value}"
                             + stripped[last_colon:] + "\n")
            matched = True
            hits += 1
            break
    if not matched:
        out_lines.append(line)
text = "".join(out_lines)
assert hits == 70, f"patched {hits} of 70"

# 3. vola_low_pct chains (Pine-only knob): high joins 0.05; low takes gold 0.07.
o1 = "is_spx_20260524_v15_run3 or is_indices_20260610_v17gpu_run1 ? 0.05"
assert o1 in text
text = text.replace(o1, o1.replace("? 0.05", f"or {PRESET_FLAG} ? 0.05"), 1)
o2 = "is_spx_20260405_2059_salvaged_best or is_indices_20260610_v17gpu_run1 ? 0.07"
assert o2 in text
text = text.replace(o2, o2.replace("? 0.07", f"or {PRESET_FLAG} ? 0.07"), 1)

# Edge voting: winner uses false/5 both sides == fall-through defaults; no edit.

OUT.write_text(text, encoding="utf-8")
print(f"patched 70 ternaries + dropdown(default) + flag + 2 vola chains")
print(f"wrote {OUT} ({len(text.splitlines())} lines)")
