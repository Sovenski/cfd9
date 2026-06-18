"""Stage A acceptance — e830d TV-export signal audit (v18-lean preset).

Loads the TradingView export ``SP_SPX, 1D_e830d.csv`` (v18-lean preset on the
chart), runs ``SpeculatorDetector`` on the export's own OHLC (volume=1.0) with
the v18-lean Params (mirrored from temp/add_pine_v18_lean.py), and compares
``signal_high``/``signal_low`` against the export's 'Pivot High'/'Pivot Low'.

Expected AFTER the pir_of partial-window fix: HIGH flips 0 (was 5), LOW 0.

Run from the repo root:  python temp/audit_e830d_v18lean.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

from src.detector import SpeculatorDetector, build_detector_artifacts  # noqa: E402
from src.indicators import Params  # noqa: E402

EXPORT = Path(r"C:\Users\kuben\Downloads\SP_SPX, 1D_e830d.csv")

# v18-lean winners (v17gpu_20260611_131405; temp/add_pine_v18_lean.py HIGH/LOW)
_HIGH = dict(
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
    er_directional=False, use_trend=False, use_volume=False,
    use_momentum=False, use_momentum_velocity=False, use_volatility=True,
    use_er_gate=False, use_gjr_asym=False, use_har_vol=False,
    vola_method="ATR", momentum_velocity_mode="Reversal",
)
_LOW = dict(
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
    er_directional=False, use_trend=False, use_volume=False,
    use_momentum=False, use_momentum_velocity=False, use_volatility=True,
    use_er_gate=False, use_gjr_asym=False, use_har_vol=False,
    vola_method="ATR", momentum_velocity_mode="Reversal",
)


def v18_lean_params() -> Params:
    over = {f"{k}_high": v for k, v in _HIGH.items()}
    over.update({f"{k}_low": v for k, v in _LOW.items()})
    return Params(**over)


def main() -> int:
    tv = pd.read_csv(EXPORT)
    tv.columns = [str(c).strip() for c in tv.columns]
    ren = {c: c.lower() for c in tv.columns
           if c.lower() in ("time", "open", "high", "low", "close")}
    tv = tv.rename(columns=ren)
    df = tv[["open", "high", "low", "close"]].copy()
    df["volume"] = 1.0

    res = SpeculatorDetector(df, v18_lean_params(),
                             build_detector_artifacts(df)).run()

    fails = 0
    for side, col in (("high", "Pivot High"), ("low", "Pivot Low")):
        tv_sig = tv[col].fillna(0).astype(float).to_numpy() > 0
        py_sig = res[f"signal_{side}"].to_numpy().astype(bool)
        flip_idx = np.flatnonzero(tv_sig != py_sig)
        print(f"signal_{side}: tv_count={int(tv_sig.sum())} "
              f"py_count={int(py_sig.sum())} flips={len(flip_idx)} "
              f"bars={flip_idx.tolist()[:20]}")
        fails += int(len(flip_idx) > 0)
    print(f"\ne830d v18-lean AUDIT: {'PASS' if fails == 0 else 'FAIL'}")
    return 0 if fails == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
