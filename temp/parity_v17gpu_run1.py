"""Parity audit: v17-GPU Run 1 preset — Pine (TV export) vs Python detector.

The TV export carries its own OHLCV, so the Python detector runs on the EXACT
bars Pine saw (no cross-file time-join drift). Signal columns in this export
are 'Pivot High'/'Pivot Low' -> normalized to signal_high/signal_low.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

from src.indicators import Params
from src.parity import compare_python_to_tv

EXPORT = Path(r"C:\Users\kuben\Downloads\SP_SPX, 1D_d2243.csv")

# The v17-GPU Run 1 preset — full precision, identical to the Pine ternaries.
PARAMS = Params(
    pct_extreme_high=0.554436424188316,
    min_agreement_high=0.47057701572775845,
    dur_extreme_pct_high=0.6388478276878595,
    scale_div_thresh_high=0.23202978018671275,
    pivot_drift_thresh_high=0.02970966456271708,
    pivot_drift_gate_mult_high=4.846518321894109,
    momentum_velocity_thresh_high=0.0109414285980165,
    pct_extreme_low=0.9898309362374363,
    min_agreement_low=0.10161243349676918,
    dur_extreme_pct_low=0.8481916445277483,
    scale_div_thresh_low=0.24889258429931752,
    pivot_drift_thresh_low=0.013284897982313939,
    momentum_velocity_thresh_low=0.007715178292267143,
    vola_high_pct_low=0.7099270056620911,
    # Everything else: gold defaults — unchanged by the run (incl. the dead
    # pivot_drift_gate_mult_low=4.0 and all use_* switches).
)

# Normalize the export's signal plot names for src.parity.
tv = pd.read_csv(EXPORT)
tv.columns = [str(c).strip() for c in tv.columns]
tv["signal_high"] = tv["Pivot High"].fillna(0).astype(float) > 0
tv["signal_low"] = tv["Pivot Low"].fillna(0).astype(float) > 0
norm_path = _REPO / "temp" / "_tv_export_v17gpu_run1_norm.csv"
tv.to_csv(norm_path, index=False)

merged, metrics = compare_python_to_tv(
    raw_path=EXPORT,         # detector runs on the exact bars Pine saw
    tv_export_path=norm_path,
    params=PARAMS,
)

n_high_py = int(merged.get("signal_high_py", merged.get("signal_high")).sum())
n_low_py = int(merged.get("signal_low_py", merged.get("signal_low")).sum())
n_high_tv = int(tv["signal_high"].sum())
n_low_tv = int(tv["signal_low"].sum())
print(f"\nbars compared: {len(merged)}")
print(f"signal counts  PY: high={n_high_py} low={n_low_py}   "
      f"TV: high={n_high_tv} low={n_low_tv}")
print(f"{'metric':28s} {'rows':>6s} {'mismatch':>9s} {'rate':>9s} {'max|diff|':>12s}")
fails = 0
for m in metrics:
    flag = ""
    if m.name.startswith("signal") and m.mismatch_rows > 0:
        flag = "  << FAIL"
        fails += 1
    mad = f"{m.max_abs_diff:.3g}" if m.max_abs_diff is not None else "-"
    print(f"{m.name:28s} {m.compared_rows:>6d} {m.mismatch_rows:>9d} "
          f"{m.mismatch_rate:>9.5f} {mad:>12s}{flag}")

count_ok = (n_high_py == n_high_tv) and (n_low_py == n_low_tv)
print(f"\ncount diff == 0 gate: {'PASS' if count_ok else 'FAIL'} "
      f"(high {n_high_py - n_high_tv:+d}, low {n_low_py - n_low_tv:+d})")
print("PARITY:", "PASS" if (fails == 0 and count_ok) else "FAIL")
