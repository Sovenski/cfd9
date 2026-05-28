from pathlib import Path

from src.universe import (
    Stream, UNIVERSE, TIMEFRAMES, CLUSTER_MAP, resolve_streams,
)


def test_groups_and_clusters_are_consistent():
    # Every ticker in every group has a cluster assignment.
    all_tickers = {t for tickers in UNIVERSE.values() for t in tickers}
    missing = [t for t in all_tickers if t not in CLUSTER_MAP]
    assert missing == [], f"tickers without cluster: {missing}"
    assert "1D" in TIMEFRAMES


def test_resolve_streams_finds_existing_files(tmp_path):
    # Two fake exports following the canonical naming convention.
    (tmp_path / "SPX_1D_18710201_20260318.csv").write_text("time,open,high,low,close,volume\n")
    (tmp_path / "DAX_1D_19700102_20260324.csv").write_text("time,open,high,low,close,volume\n")
    streams = resolve_streams(
        groups=["TEST"], timeframes=["1D"], data_dir=str(tmp_path),
        universe={"TEST": ["SPX", "DAX"]}, cluster_map={"SPX": "US", "DAX": "EU"},
    )
    ids = sorted((s.ticker, s.timeframe) for s in streams)
    assert ids == [("DAX", "1D"), ("SPX", "1D")]
    assert {s.cluster_id for s in streams} == {"US", "EU"}
    assert all(Path(s.path).exists() for s in streams)


def test_resolve_streams_skips_missing(tmp_path, caplog):
    (tmp_path / "SPX_1D_x_y.csv").write_text("time,open,high,low,close,volume\n")
    streams = resolve_streams(
        groups=["TEST"], timeframes=["1D", "1W"], data_dir=str(tmp_path),
        universe={"TEST": ["SPX"]}, cluster_map={"SPX": "US"},
    )
    # Only SPX_1D exists; SPX_1W is absent and silently skipped.
    assert [(s.ticker, s.timeframe) for s in streams] == [("SPX", "1D")]
