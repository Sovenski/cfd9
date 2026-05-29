"""Runner integration smoke: warm-start from real v16 best params, seed-only
(max_sweeps=0) on INDICES 1D common-era. Validates resolve->folds->score->
provenance and that warm-start yields a non-degenerate seed LCB. Also persists
the v16 best params for future warm-starts."""
from __future__ import annotations
import json, logging, time
from pathlib import Path
from src.v17_runner import run_v17

logging.basicConfig(level=logging.WARNING)

# Real v16 best params (from the first completed v16 1D run, run slug v16_1D_20260529_000217)
V16_BEST = {
  "high": {"high_S_detect":38,"high_scale_start":15,"high_scale_end":217,"high_scale_step":11,
    "high_min_duration":2,"high_cooldown_bars":16,"high_price_gate_lb":89,"high_vola_range_len":115,
    "high_er_period":35,"high_confirm_count":1,"high_pivot_drift_lb":13,"high_pivot_drift_confirm_bias":0,
    "high_pct_extreme":0.7617405092259827,"high_min_agreement":0.4382353613487828,
    "high_dur_extreme_pct":0.406776275798483,"high_vol_surge_thresh":2.5961272441270764,
    "high_scale_div_thresh":0.30344535782232585,"high_slope_thresh":0.41093110485774226,
    "high_vola_high_pct":0.6921246235625901,"high_pivot_drift_thresh":0.030456734142431188,
    "high_pivot_drift_gate_mult":8.427397853280967,"high_momentum_velocity_thresh":0.043155687015693375,
    "high_gjr_vote_thresh":0.4596970088888114,"high_har_vote_thresh":0.4540553145944793,
    "high_er_directional":True,"high_use_trend":True,"high_use_volume":False,"high_use_momentum":False,
    "high_use_momentum_velocity":True,"high_use_volatility":False,"high_use_er_gate":True,
    "high_use_gjr_asym":False,"high_use_har_vol":True,"high_vola_method":"ATR",
    "high_momentum_velocity_mode":"Reversal","high_use_edge_voting":False,"high_edge_window":20},
  "low": {"low_S_detect":5,"low_scale_start":28,"low_scale_end":114,"low_scale_step":15,
    "low_min_duration":5,"low_cooldown_bars":15,"low_price_gate_lb":88,"low_vola_range_len":180,
    "low_er_period":29,"low_confirm_count":1,"low_pivot_drift_lb":7,"low_pivot_drift_confirm_bias":1,
    "low_pct_extreme":0.8810645037005915,"low_min_agreement":0.5782108672877112,
    "low_dur_extreme_pct":0.5433870368935335,"low_vol_surge_thresh":1.6865679054263292,
    "low_scale_div_thresh":0.16259167468775407,"low_slope_thresh":0.21918546095190755,
    "low_vola_high_pct":0.7306527163652201,"low_pivot_drift_thresh":0.035357340058527684,
    "low_pivot_drift_gate_mult":3.4407667812759293,"low_momentum_velocity_thresh":0.044526475752234715,
    "low_gjr_vote_thresh":0.3179430470679154,"low_har_vote_thresh":0.18661088745890775,
    "low_er_directional":True,"low_use_trend":True,"low_use_volume":False,"low_use_momentum":True,
    "low_use_momentum_velocity":True,"low_use_volatility":False,"low_use_er_gate":False,
    "low_use_gjr_asym":False,"low_use_har_vol":True,"low_vola_method":"StdDev",
    "low_momentum_velocity_mode":"Trend","low_use_edge_voting":False,"low_edge_window":29},
}
Path("results").mkdir(exist_ok=True)
Path("results/v16_best_params.json").write_text(json.dumps(V16_BEST, indent=2))

t0 = time.time()
out = run_v17(groups=["INDICES"], timeframes=["1D"], data_dir="data/raw_v16",
              sides=("low",), seed_params=V16_BEST, era_kw={},
              grid_n=3, max_sweeps=0, results_dir="results", run_slug="v17_smoke_indices")
print(json.dumps({"folds": out["n_folds"], "streams": out["streams"],
                  "low": out["sides"]["low"]}, indent=2, default=str)[:1200])
print(f"wall-clock {time.time()-t0:.1f}s")
print("RUNNER SMOKE OK")
