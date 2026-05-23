"""Oracle-quality test — Scorer v4 catches famous SPX structural lows.

Loads the TradingView SPX 1D export (must be available in
``data/raw/SPX_1D_18710201_20260318.csv``) and verifies that the new
nested-scale oracle labels at least 5 of 7 famous SPX structural lows
within a +/-30-bar window of an actual labelled structural pivot.

This is the "did v4 actually fix the bug" test. If it fails, the
nest scales need adjustment (e.g., [25, 50, 100] instead of
[50, 100, 200]).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.scoring import STRUCTURAL_NEST, label_structural_pivots

FAMOUS_LOWS = {
    "2002-10-09": "dotcom bottom",
    "2009-03-09": "GFC bottom",
    "2011-10-04": "Eurozone crisis low",
    "2016-02-11": "oil crash low",
    "2018-12-24": "late-2018 selloff",
    "2020-03-23": "COVID bottom",
    "2022-10-13": "2022 inflation bottom",
}

FAMOUS_HIGHS = {
    "2000-03-24": "dotcom top",
    "2007-10-09": "pre-GFC top",
    "2020-02-19": "pre-COVID top",
    "2022-01-04": "2022 top",
}

WINDOW_DAYS = 30


def _load_spx_1d() -> pd.DataFrame:
    csv_path = Path("data/raw/SPX_1D_18710201_20260318.csv")
    if not csv_path.exists():
        # CI/local fallback: skip cleanly if the dataset isn't on disk.
        print(f"SKIP: dataset not available at {csv_path}")
        raise SystemExit(0)
    df = pd.read_csv(csv_path)
    time_col = "time" if "time" in df.columns else df.columns[0]
    df[time_col] = pd.to_datetime(df[time_col], unit="s", errors="coerce")
    df = df.dropna(subset=[time_col]).set_index(time_col).sort_index()
    return df


def main() -> None:
    df = _load_spx_1d()
    print(f"loaded {len(df)} bars, range {df.index.min()} -> {df.index.max()}")
    print(f"using STRUCTURAL_NEST = {STRUCTURAL_NEST}")

    labels = label_structural_pivots(df, STRUCTURAL_NEST)
    structural_low_dates = labels.index[labels == -1]
    structural_high_dates = labels.index[labels == 1]
    print(
        f"structural LOWs: {len(structural_low_dates)} "
        f"({100.0 * len(structural_low_dates) / len(df):.3f}%)"
    )
    print(
        f"structural HIGHs: {len(structural_high_dates)} "
        f"({100.0 * len(structural_high_dates) / len(df):.3f}%)"
    )

    print()
    print("=== LOW hit-test ===")
    low_hits = 0
    for date_str, label in FAMOUS_LOWS.items():
        target = pd.Timestamp(date_str)
        near = [
            d for d in structural_low_dates
            if abs((d - target).days) <= WINDOW_DAYS
        ]
        if near:
            low_hits += 1
            offset = (near[0] - target).days
            print(f"  HIT  {date_str} ({label}): nearest at {near[0].date()} ({offset:+d}d)")
        else:
            print(f"  MISS {date_str} ({label})")

    print()
    print("=== HIGH hit-test ===")
    high_hits = 0
    for date_str, label in FAMOUS_HIGHS.items():
        target = pd.Timestamp(date_str)
        near = [
            d for d in structural_high_dates
            if abs((d - target).days) <= WINDOW_DAYS
        ]
        if near:
            high_hits += 1
            offset = (near[0] - target).days
            print(f"  HIT  {date_str} ({label}): nearest at {near[0].date()} ({offset:+d}d)")
        else:
            print(f"  MISS {date_str} ({label})")

    print()
    print(f"LOW hits: {low_hits}/{len(FAMOUS_LOWS)} (require >= 5)")
    print(f"HIGH hits: {high_hits}/{len(FAMOUS_HIGHS)} (require >= 3)")
    assert low_hits >= 5, f"v4 oracle missed too many famous lows: {low_hits}/7"
    assert high_hits >= 3, f"v4 oracle missed too many famous highs: {high_hits}/4"

    print()
    print("ALL OK")


if __name__ == "__main__":
    main()
