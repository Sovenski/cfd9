"""Asset / timeframe / cluster registry for multi-asset pooled optimization.

A pool unit is an ``(asset, timeframe)`` stream. ``CLUSTER_MAP`` groups assets
that share an underlying/economy so the pooled scorer can down-weight
correlated duplicates (see ``pooled_scoring``).
"""
from __future__ import annotations

import glob
import logging
import os
import re
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Real v16 UNIVERSE, TIMEFRAMES, CLUSTER_MAP
# ---------------------------------------------------------------------------

# 1m exists in the exports but is excluded from the default TIMEFRAMES list —
# too fine-grained for the structural nesting approach.
TIMEFRAMES: list[str] = ["1D", "240", "60"]

UNIVERSE: dict[str, list[str]] = {
    # SPX has 1D/240/60; NDX and DAX have 240/60 only
    "INDICES": ["SPX", "NDX", "DAX"],
    "STOCKS": [
        "AAPL", "ABBV", "AMD", "AMZN", "AVGO", "BAC", "BRKB", "COST", "CSCO", "CVX",
        "GE", "GOOG", "GOOGL", "HD", "JNJ", "JPM", "KO", "LLY", "MA", "MCD", "META",
        "MSFT", "NFLX", "ORCL", "PG", "PLTR", "TMUS", "TSLA", "V", "XOM",
    ],
    # EURUSD also has 240/60; others are 1D only
    "FX": ["EURGBP", "EURJPY", "EURUSD", "GBPUSD", "USDCHF", "USDRUB", "USDSEK"],
    "COMMODITIES": ["GC1", "SI1", "PL1", "PA1", "WTI"],
    "WORLD_ETF": ["VT", "VWCE"],
}

CLUSTER_MAP: dict[str, str] = {
    # US equity index + single names — one correlated cluster
    "SPX": "US_EQ", "NDX": "US_EQ",
    "AAPL": "US_EQ", "ABBV": "US_EQ", "AMD": "US_EQ", "AMZN": "US_EQ",
    "AVGO": "US_EQ", "BAC": "US_EQ", "BRKB": "US_EQ", "COST": "US_EQ",
    "CSCO": "US_EQ", "CVX": "US_EQ", "GE": "US_EQ", "GOOG": "US_EQ",
    "GOOGL": "US_EQ", "HD": "US_EQ", "JNJ": "US_EQ", "JPM": "US_EQ",
    "KO": "US_EQ", "LLY": "US_EQ", "MA": "US_EQ", "MCD": "US_EQ",
    "META": "US_EQ", "MSFT": "US_EQ", "NFLX": "US_EQ", "ORCL": "US_EQ",
    "PG": "US_EQ", "PLTR": "US_EQ", "TMUS": "US_EQ", "TSLA": "US_EQ",
    "V": "US_EQ", "XOM": "US_EQ",
    "DAX": "EU_EQ",
    "VT": "WORLD_EQ", "VWCE": "WORLD_EQ",
    "GC1": "METALS", "SI1": "METALS", "PL1": "METALS", "PA1": "METALS",
    "WTI": "ENERGY",
    "EURGBP": "FX", "EURJPY": "FX", "EURUSD": "FX", "GBPUSD": "FX",
    "USDCHF": "FX", "USDRUB": "FX", "USDSEK": "FX",
}

# Regex for TV-native export filenames: "{SOURCE_PREFIX}, {TF}_{hash}.csv"
# Hash is nominally hex but we accept any word chars to handle synthetic test
# filenames like "_x" used in unit tests.
_TV_RE = re.compile(
    r"^(?P<prefix>.+),\s*(?P<tf>1D|1W|240|60|1)_(?P<hash>\w+)\.csv$"
)

_TF_CANON: dict[str, str] = {
    "1D": "1D",
    "1W": "1W",
    "240": "240",
    "60": "60",
    "1": "1m",
}

# Canonical-style regex: {TICKER}_{TF}_{anything}.csv
# TF must be one of the known values (no underscore in ticker assumed).
_CANON_TF_SET = {"1D", "1W", "240", "60", "1m"}


def parse_tv_filename(name: str) -> tuple[str, str] | None:
    """Parse a TradingView-exported CSV filename into ``(ticker, timeframe)``.

    TV exports follow the pattern::

        {EXCHANGE_PREFIX}, {TF}_{hash}.csv

    e.g. ``BATS_AAPL, 1D_9f4ab.csv`` → ``("AAPL", "1D")``.

    Args:
        name: Bare filename (no directory part).

    Returns:
        ``(ticker, tf_canon)`` on success, or ``None`` if the filename does
        not match the TV pattern.
    """
    m = _TV_RE.match(name)
    if m is None:
        return None
    prefix: str = m.group("prefix")
    tf_raw: str = m.group("tf")
    # Last segment after the last "_" is the bare symbol; strip exchange layers.
    ticker = prefix.rsplit("_", 1)[-1]
    # Normalise special chars (futures root "GC1!" → "GC1", "BRK.B" → "BRKB").
    ticker = ticker.replace("!", "").replace(".", "")
    tf_canon = _TF_CANON[tf_raw]
    return ticker, tf_canon


def _parse_canonical(name: str) -> tuple[str, str] | None:
    """Try to parse a canonical ``{TICKER}_{TF}_{rest}.csv`` filename.

    Returns ``(ticker, tf)`` or ``None``.
    """
    if not name.endswith(".csv"):
        return None
    stem = name[:-4]  # strip .csv
    parts = stem.split("_")
    # Need at least TICKER + TF + one more part.
    if len(parts) < 3:
        return None
    # TF is the second token; anything after is the date-range / hash suffix.
    tf = parts[1]
    if tf not in _CANON_TF_SET:
        return None
    ticker = parts[0]
    return ticker, tf


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

    Understands two CSV naming conventions:

    * **Canonical**: ``{TICKER}_{TF}_{suffix}.csv``
      (e.g. ``SPX_1D_18710201_20260318.csv``)
    * **TV-native**: ``{EXCHANGE_PREFIX}, {TF}_{hash}.csv``
      (e.g. ``BATS_AAPL, 1D_9f4ab.csv``)

    For each requested ``(ticker, timeframe)`` pair the first matching file
    found in ``data_dir`` is used. When the same ``(ticker, tf)`` appears in
    multiple files only the first is kept and a debug message is logged.

    Missing ``(ticker, timeframe)`` combinations are skipped with a warning.

    Args:
        groups: Keys into ``universe`` whose tickers should be included.
        timeframes: Timeframe strings to look up (e.g. ``["1D", "240"]``).
        data_dir: Directory to scan for CSV files.
        universe: Override for the module-level ``UNIVERSE`` dict.
        cluster_map: Override for the module-level ``CLUSTER_MAP`` dict.

    Returns:
        List of :class:`Stream` objects, one per resolved ``(ticker, tf)``.
    """
    universe = universe or UNIVERSE
    cluster_map = cluster_map or CLUSTER_MAP

    # Build the ordered, de-duplicated ticker list from requested groups.
    tickers: list[str] = []
    for g in groups:
        for t in universe.get(g, []):
            if t not in tickers:
                tickers.append(t)

    # Scan data_dir once and build available[(ticker, tf)] = path.
    available: dict[tuple[str, str], str] = {}
    for fname in os.listdir(data_dir):
        fpath = os.path.join(data_dir, fname)
        if not os.path.isfile(fpath):
            continue
        # Try canonical parse first, then TV-native.
        parsed = _parse_canonical(fname) or parse_tv_filename(fname)
        if parsed is None:
            continue
        ticker, tf = parsed
        key = (ticker, tf)
        if key in available:
            logger.debug(
                "resolve_streams: duplicate file for %s_%s — keeping %s, ignoring %s",
                ticker, tf, available[key], fpath,
            )
        else:
            available[key] = fpath

    # Emit one Stream per requested (ticker, tf) that exists on disk.
    streams: list[Stream] = []
    for ticker in tickers:
        for tf in timeframes:
            path = available.get((ticker, tf))
            if path is None:
                logger.warning(
                    "resolve_streams: no file for %s_%s in %s",
                    ticker, tf, data_dir,
                )
                continue
            streams.append(Stream(
                ticker=ticker,
                timeframe=tf,
                path=path,
                cluster_id=cluster_map.get(ticker, ticker),
            ))

    logger.info("resolve_streams: %d streams resolved", len(streams))
    return streams
