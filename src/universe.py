"""Asset / timeframe / cluster registry for multi-asset pooled optimization.

A pool unit is an ``(asset, timeframe)`` stream. ``CLUSTER_MAP`` groups assets
that share an underlying/economy so the pooled scorer can down-weight
correlated duplicates (see ``pooled_scoring``).
"""
from __future__ import annotations

import glob
import logging
import os
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Named asset groups (TV tickers, bare symbol — source prefix dropped).
UNIVERSE: dict[str, list[str]] = {
    "INDICES_US": ["SPX", "NDX", "DJI", "RUT"],
    "INDICES_GLOBAL": ["SPX", "NDX", "DAX", "NI225", "UKX"],
    "ETF_PROXIES": ["SPY", "QQQ", "DIA", "IWM"],
    "STOCKS_MEGACAP": ["AAPL", "MSFT", "AMZN", "GOOGL", "NVDA"],
    "COMMODITIES": ["GC1", "CL1", "SI1"],
}

TIMEFRAMES: list[str] = ["1D", "1W", "60", "240"]

# Assets sharing a cluster are correlated; pooled scoring weights each
# stream by 1 / (streams in its cluster present in the fold).
CLUSTER_MAP: dict[str, str] = {
    "SPX": "US_EQ", "NDX": "US_EQ", "DJI": "US_EQ", "RUT": "US_EQ",
    "SPY": "US_EQ", "QQQ": "US_EQ", "DIA": "US_EQ", "IWM": "US_EQ",
    "AAPL": "US_EQ", "MSFT": "US_EQ", "AMZN": "US_EQ",
    "GOOGL": "US_EQ", "NVDA": "US_EQ",
    "DAX": "EU_EQ", "UKX": "EU_EQ",
    "NI225": "JP_EQ",
    "GC1": "METALS", "SI1": "METALS",
    "CL1": "ENERGY",
}


@dataclass(frozen=True)
class Stream:
    ticker: str
    timeframe: str
    path: str
    cluster_id: str

    @property
    def stream_id(self) -> str:
        return f"{self.ticker}_{self.timeframe}"


def resolve_streams(
    groups: list[str],
    timeframes: list[str],
    data_dir: str = "data/raw",
    universe: dict[str, list[str]] | None = None,
    cluster_map: dict[str, str] | None = None,
) -> list[Stream]:
    """Resolve selected groups × timeframes to existing data files.

    Each stream's file is matched as ``{TICKER}_{TF}_*.csv`` under
    ``data_dir``. Missing (ticker, timeframe) combinations are skipped with a
    warning. De-duplicates tickers that appear in multiple selected groups.
    """
    universe = universe or UNIVERSE
    cluster_map = cluster_map or CLUSTER_MAP
    tickers: list[str] = []
    for g in groups:
        for t in universe.get(g, []):
            if t not in tickers:
                tickers.append(t)

    streams: list[Stream] = []
    for ticker in tickers:
        for tf in timeframes:
            pattern = os.path.join(data_dir, f"{ticker}_{tf}_*.csv")
            matches = sorted(glob.glob(pattern))
            if not matches:
                logger.warning("resolve_streams: no file for %s_%s (%s)",
                               ticker, tf, pattern)
                continue
            streams.append(Stream(
                ticker=ticker, timeframe=tf, path=matches[0],
                cluster_id=cluster_map.get(ticker, ticker),
            ))
    logger.info("resolve_streams: %d streams resolved", len(streams))
    return streams
