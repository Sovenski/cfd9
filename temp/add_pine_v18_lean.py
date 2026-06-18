"""Add the 'v18 Lean Discriminators' preset to the two-preset v17.5 pine.

Votes: ONLY use_volatility on (pivot-drift is structurally always-on);
trend/volume/momentum/mom_vel/gjr/har OFF both sides — per the parity-anchored
vote-contribution census (drift + volatility are the only real discriminators;
gjr/trend were rubber stamps, volume dead, momentum/har inverted).
Thresholds/gates: v17gpu_20260611_131405 winners (HIGH=all_on, LOW=minus_momentum).
Effective rule (required clamps to 1): gate AND (vola OR drift) AND cooldown.
Set as dropdown DEFAULT. Card block remains calibrated to 090553 (stale for
this preset — signals exact, card numbers indicative only).
Suffix-safe LOW keys (feedback_pine_preset_patcher_pitfall).
"""
from pathlib import Path
import re

PINE = Path(r"temp/speculatores_v17_5_two_presets.pine")
OUT = Path(r"temp/speculatores_v17_5_v18lean.pine")
PRESET_NAME = "v18 Lean Discriminators 2026-06-11"
PRESET_FLAG = "is_v18_lean_20260611"

HIGH = dict(
    S_detect=12, scale_start=3, scale_end=270, scale_step=3,
    min_duration=13, cooldown_bars=9, price_gate_lb=23, vola_range_len=120,
    er_period=28, confirm_count=5, pivot_drift_lookback=5,
    pivot_drift_confirm_bias=1,
    pct_extreme=0.3989736829331265, min_agreement=0.5902329742597521,
    dur_extreme_pct=0.10034295109907004, vol_surge_thresh=1.002017461803299,
    scale_div_thresh=0.599107129205612, slope_thresh=0.014515054272148754,
    vola_high_pct=0.5027540174875171,
    pivot_drift_thresh=0.033318432240576665,
    pivot_drift_gate_mult=5.9550694191377165,
    momentum_velocity_thresh=1.3548394099613192e-08,
    gjr_vote_thresh=0.05168051407941195, har_vote_thresh=0.05605184243891868,
    er_directional="false", use_trend="false", use_volume="false",
    use_momentum="false", use_momentum_velocity="false", use_volatility="true",
    use_er_gate="false", use_gjr_asym="false", use_har_vol="false",
    vola_method='"ATR"', momentum_velocity_mode='"Reversal"',
)
LOW = dict(
    S_detect=26, scale_start=19, scale_end=250, scale_step=15,
    min_duration=2, cooldown_bars=7, price_gate_lb=58, vola_range_len=20,
    er_period=47, confirm_count=3, pivot_drift_lookback=10,
    pivot_drift_confirm_bias=0,
    pct_extreme=0.49268731212549655, min_agreement=0.02065350717621217,
    dur_extreme_pct=0.7096091393768227, vol_surge_thresh=2.2382428216185266,
    scale_div_thresh=0.7358944084387257, slope_thresh=0.4025458749698267,
    vola_high_pct=0.5571651880232253,
    pivot_drift_thresh=0.030211982920262805, pivot_drift_gate_mult=4.0,
    momentum_velocity_thresh=0.003987965341605985,
    gjr_vote_thresh=0.3614954474374879, har_vote_thresh=0.11076327581817122,
    er_directional="false", use_trend="false", use_volume="false",
    use_momentum="false", use_momentum_velocity="false", use_volatility="true",
    use_er_gate="false", use_gjr_asym="false", use_har_vol="false",
    vola_method='"ATR"', momentum_velocity_mode='"Reversal"',
)

text = PINE.read_text(encoding="utf-8")

# 1. Dropdown: new preset becomes the DEFAULT; keep all existing options.
old_default = 'input.string("INDICES 1D 2026-06-11 v5.1 Widened", "Config Preset"'
assert old_default in text, "default anchor not found"
text = text.replace(old_default,
                    f'input.string("{PRESET_NAME}", "Config Preset"', 1)
old_opts = ('options=["INDICES 1D 2026-06-11 v5.1 Widened", '
            '"INDICES 1D 2026-06-10 v5 Big Run", '
            '"INDICES 1D 2026-06-10 v17-GPU Run 1", "Gold 1D Current", '
            '"Legacy V11 Gold"]')
assert old_opts in text, "options anchor not found"
text = text.replace(
    old_opts,
    f'options=["{PRESET_NAME}", "INDICES 1D 2026-06-11 v5.1 Widened", '
    '"INDICES 1D 2026-06-10 v5 Big Run", '
    '"INDICES 1D 2026-06-10 v17-GPU Run 1", "Gold 1D Current", '
    '"Legacy V11 Gold"]', 1)

# 2. Flag after the v5big flag.
old_flag = ('bool is_indices_20260610_v5big = preset == '
            '"INDICES 1D 2026-06-10 v5 Big Run"')
assert old_flag in text, "flag anchor not found"
text = text.replace(
    old_flag, old_flag + f'\nbool {PRESET_FLAG} = preset == "{PRESET_NAME}"', 1)


def _val(side, key):
    v = side[key]
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
low_keys = [(h[: -len("_high")] + "_low", key) for h, key in high_keys]

out_lines, hits = [], 0
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

# 3. vola_low_pct chains: join 0.05 (high) / 0.07 (low) like prior presets.
o1 = "or is_indices_20260610_v5big ? 0.05"
assert o1 in text
text = text.replace(o1, f"or is_indices_20260610_v5big or {PRESET_FLAG} ? 0.05", 1)
o2 = "or is_indices_20260610_v5big ? 0.07"
assert o2 in text
text = text.replace(o2, f"or is_indices_20260610_v5big or {PRESET_FLAG} ? 0.07", 1)

OUT.write_text(text, encoding="utf-8")
print(f"added DEFAULT preset '{PRESET_NAME}' (70 ternaries + options + flag + 2 vola)")
print(f"wrote {OUT} ({len(text.splitlines())} lines)")
