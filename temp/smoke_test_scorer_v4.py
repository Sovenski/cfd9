"""Scorer v4 smoke test — nested-scale oracle is strictly more selective.

Synthesizes a 5000-bar random walk and verifies that:
1. ``label_structural_pivots`` returns ≤ 1% of bars as structural pivots.
2. The same series, labelled with single-scale ``label_pivots(N=50)``,
   returns far more pivots — confirming the nest is strictly stricter.
3. ``add_pivot_labels`` writes the structural label into ``pivot_N100``
   (no other ``pivot_N*`` columns are populated by default).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.scoring import (
    PIVOT_SCALES,
    STRUCTURAL_NEST,
    add_pivot_labels,
    label_pivots,
    label_structural_pivots,
)


def _make_random_walk(n_bars: int = 5000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    steps = rng.normal(0.0, 1.0, size=n_bars)
    close = 100.0 + np.cumsum(steps)
    high = close + rng.uniform(0.0, 0.5, size=n_bars)
    low = close - rng.uniform(0.0, 0.5, size=n_bars)
    idx = pd.date_range("2000-01-01", periods=n_bars, freq="D")
    return pd.DataFrame(
        {"open": close, "high": high, "low": low, "close": close, "volume": 1.0},
        index=idx,
    )


def main() -> None:
    df = _make_random_walk()

    structural = label_structural_pivots(df, STRUCTURAL_NEST)
    n_structural = int((structural != 0).sum())
    pct_structural = 100.0 * n_structural / len(df)
    print(
        f"structural pivots: {n_structural} / {len(df)} "
        f"({pct_structural:.3f}%)"
    )
    assert pct_structural <= 1.0, (
        f"structural pivots should be ≤ 1% of bars, got {pct_structural:.2f}%"
    )

    single_50 = label_pivots(df, 50)
    n_single = int((single_50 != 0).sum())
    print(f"single-scale (N=50) pivots: {n_single} / {len(df)}")
    assert n_single > n_structural, (
        f"single-scale should yield more pivots than nest, got "
        f"{n_single} vs {n_structural}"
    )
    ratio = n_single / max(n_structural, 1)
    assert ratio >= 3.0, (
        f"single-scale should yield at least 3× more pivots, got {ratio:.2f}×"
    )

    df_with = add_pivot_labels(df.copy())
    structural_col = f"pivot_N{PIVOT_SCALES[0]}"
    assert structural_col in df_with.columns
    assert (df_with[structural_col] == structural).all(), (
        "add_pivot_labels disagreement with label_structural_pivots"
    )
    for old_n in [5, 10, 20, 50, 200, 500]:
        assert f"pivot_N{old_n}" not in df_with.columns, (
            f"v3 column pivot_N{old_n} should NOT be written in v4"
        )

    print("ALL OK")


if __name__ == "__main__":
    main()
