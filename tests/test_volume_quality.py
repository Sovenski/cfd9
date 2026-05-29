import numpy as np
import pandas as pd

from src.volume_quality import profile_volume, VolumeQuality


def _df(volume, start="2015-01-01"):
    n = len(volume)
    idx = pd.date_range(start, periods=n, freq="D")
    return pd.DataFrame({
        "open": np.ones(n), "high": np.ones(n), "low": np.ones(n),
        "close": np.ones(n), "volume": volume,
    }, index=idx)


def test_full_quality_when_volume_all_real():
    vq = profile_volume(_df(np.random.randint(1_000_000, 5_000_000, 500)))
    assert vq.quality == "full"
    assert vq.placeholder_share < 0.05


def test_none_quality_when_all_placeholder():
    vq = profile_volume(_df(np.full(500, 3600)))
    assert vq.quality == "none"


def test_partial_quality_detects_transition():
    # 300 placeholder days then 300 real-volume days.
    vol = np.concatenate([np.full(300, 3600),
                          np.random.randint(1_000_000, 5_000_000, 300)])
    vq = profile_volume(_df(vol, start="2015-01-01"))
    assert vq.quality == "partial"
    assert vq.real_from is not None
    # Real volume begins around day ~300 (year 2015 + ~300 days → 2015-10/11).
    assert vq.real_from.year == 2015 and vq.real_from.month >= 10
