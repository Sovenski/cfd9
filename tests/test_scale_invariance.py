import numpy as np
import pandas as pd

from src.scale_invariance import precision_transfer
from src.indicators import Params


def _labelled(n, seed):
    rng = np.random.RandomState(seed)
    close = 100 + np.cumsum(rng.randn(n))
    idx = pd.date_range("2000-01-01", periods=n, freq="D")
    df = pd.DataFrame({"open": close, "high": close + 1, "low": close - 1,
                       "close": close, "volume": np.full(n, 1_000_000)}, index=idx)
    return df


def test_precision_transfer_returns_two_precisions_and_drop():
    base = _labelled(3000, 0)
    other = _labelled(1500, 1)
    res = precision_transfer(Params(), "high", base, other)
    assert set(res) == {"precision_base", "precision_other", "drop"}
    assert res["drop"] == res["precision_base"] - res["precision_other"]
