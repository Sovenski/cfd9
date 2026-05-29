"""Confirm the common-era fold fix eliminates solo folds on the REAL default pool."""
import sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.scoring import add_pivot_labels, REFERENCE_N
from src.universe import resolve_streams
from src.pooled_validation import StreamData, build_calendar_folds, load_stream_frame, apply_volume_policy

_TF = {"1D": 86400.0, "240": 14400.0, "60": 3600.0}
streams = resolve_streams(["INDICES", "COMMODITIES", "WORLD_ETF", "FX"], ["1D"], data_dir=str(ROOT/"data"/"raw_v16"))
sd = []
for s in streams:
    df = load_stream_frame(s.path); df, keep = apply_volume_policy(df, "price_only")
    if keep:
        add_pivot_labels(df); sd.append(StreamData(stream=s, df=df, bar_seconds=_TF[s.timeframe]))

for label, kw in [("OLD (full history)", dict(start="1871-01-01")), ("NEW (auto common-era)", dict())]:
    folds = build_calendar_folds(sd, **kw)
    spf = [len(f) for f in folds]
    solo = sum(1 for x in spf if x <= 1)
    low = [sum(int((sl.df_oos[f"pivot_N{REFERENCE_N}"] == -1).sum()) for sl in f) for f in folds]
    high = [sum(int((sl.df_oos[f"pivot_N{REFERENCE_N}"] == 1).sum()) for sl in f) for f in folds]
    print(f"\n{label}: {len(folds)} folds, solo(<=1 stream)={solo}")
    print(f"  streams/fold = {spf}")
    print(f"  LOW pivots/fold  = {low}  (mean {np.mean(low):.1f})")
    print(f"  HIGH pivots/fold = {high}  (mean {np.mean(high):.1f})")
