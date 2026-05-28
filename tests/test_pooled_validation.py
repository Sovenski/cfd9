import numpy as np
import pandas as pd

from src.universe import Stream
from src.pooled_validation import load_stream_frame, apply_volume_policy


def _write_csv(tmp_path, name, n, start_ts=1420070400, step=86400, volume=None):
    t = np.arange(n) * step + start_ts
    close = 100 + np.cumsum(np.random.RandomState(0).randn(n))
    vol = volume if volume is not None else np.full(n, 1_000_000)
    df = pd.DataFrame({"time": t, "open": close, "high": close + 1,
                       "low": close - 1, "close": close, "Volume": vol})
    p = tmp_path / name
    df.to_csv(p, index=False)
    return str(p)


def test_load_stream_frame_normalizes_and_indexes(tmp_path):
    path = _write_csv(tmp_path, "SPX_1D_a_b.csv", 500)
    df = load_stream_frame(path)
    assert isinstance(df.index, pd.DatetimeIndex)
    assert {"open", "high", "low", "close", "volume"}.issubset(df.columns)


def test_volume_policy_required_trims_to_real_range(tmp_path):
    vol = np.concatenate([np.full(300, 3600),
                          np.random.RandomState(1).randint(1_000_000, 5_000_000, 300)])
    path = _write_csv(tmp_path, "SPX_1D_a_b.csv", 600, volume=vol)
    df = load_stream_frame(path)
    trimmed, kept = apply_volume_policy(df, policy="volume_required")
    assert kept is True
    assert len(trimmed) < len(df)        # placeholder head removed
    assert trimmed["volume"].iloc[0] >= 100_000


def test_volume_policy_price_only_keeps_all(tmp_path):
    path = _write_csv(tmp_path, "SPX_1D_a_b.csv", 500, volume=np.full(500, 3600))
    df = load_stream_frame(path)
    trimmed, kept = apply_volume_policy(df, policy="price_only")
    assert kept is True and len(trimmed) == len(df)


def test_volume_policy_required_drops_volumeless_stream(tmp_path):
    path = _write_csv(tmp_path, "SPX_1D_a_b.csv", 500, volume=np.full(500, 3600))
    df = load_stream_frame(path)
    _, kept = apply_volume_policy(df, policy="volume_required")
    assert kept is False     # 'none' quality → excluded under volume_required
