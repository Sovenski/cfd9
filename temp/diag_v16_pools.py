"""Report fold count + structural-pivots-per-OOS-fold per side for candidate v16
1D pools on the real data/raw_v16 exports. Informs the notebook default."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.scoring import add_pivot_labels, REFERENCE_N
from src.universe import resolve_streams
from src.pooled_validation import (
    StreamData, build_calendar_folds, load_stream_frame, apply_volume_policy,
)

DATA_DIR = str(ROOT / "data" / "raw_v16")
_TF = {"1D": 86400.0, "240": 14400.0, "60": 3600.0}

CANDIDATES = {
    "INDICES @1D": (["INDICES"], ["1D"]),
    "INDICES+COMMODITIES+WORLD_ETF+FX @1D": (
        ["INDICES", "COMMODITIES", "WORLD_ETF", "FX"], ["1D"]),
    "ALL @1D (incl. stocks)": (
        ["INDICES", "STOCKS", "COMMODITIES", "WORLD_ETF", "FX"], ["1D"]),
}


def run(groups, tfs):
    streams = resolve_streams(groups, tfs, data_dir=DATA_DIR)
    sd = []
    for s in streams:
        df = load_stream_frame(s.path)
        df, keep = apply_volume_policy(df, "price_only")
        if not keep:
            continue
        add_pivot_labels(df)
        sd.append(StreamData(stream=s, df=df, bar_seconds=_TF[s.timeframe]))
    folds = build_calendar_folds(sd)
    out = {"streams": len(sd), "clusters": len({x.stream.cluster_id for x in sd}),
           "folds": len(folds)}
    for side, lbl in (("HIGH", 1), ("LOW", -1)):
        per = [sum(int((sl.df_oos[f"pivot_N{REFERENCE_N}"] == lbl).sum()) for sl in f)
               for f in folds]
        out[side] = (sum(1 for x in per if x > 0), round(float(np.mean(per)), 1) if per else 0, per)
    return out


for name, (g, t) in CANDIDATES.items():
    r = run(g, t)
    print(f"\n### {name}")
    print(f"  streams={r['streams']} clusters={r['clusters']} folds={r['folds']}")
    for side in ("HIGH", "LOW"):
        inf, mean, per = r[side]
        print(f"  {side}: informative={inf}/{r['folds']} mean_pivots/fold={mean}  per_fold={per}")
