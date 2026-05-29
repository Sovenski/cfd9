"""Catalog data/raw_v16 TV exports: parse {SOURCE..._TICKER}, {TF}_{hash}.csv into
(ticker, timeframe) with bar count + date range, so we can design real pools."""
from __future__ import annotations
import re
from pathlib import Path
import pandas as pd

SRC = Path("data/raw_v16")
TF_CANON = {"1D": "1D", "1W": "1W", "240": "240", "60": "60", "1": "1m"}

pat = re.compile(r"^(?P<prefix>.+),\s*(?P<tf>1D|1W|240|60|1)_(?P<hash>[0-9a-fA-F]+)\.csv$")


def norm_ticker(prefix: str) -> str:
    t = prefix.rsplit("_", 1)[-1]
    return t.replace("!", "").replace(".", "")


rows = []
for f in sorted(SRC.glob("*.csv")):
    m = pat.match(f.name)
    if not m:
        print("UNPARSED:", f.name); continue
    ticker = norm_ticker(m.group("prefix"))
    tf = TF_CANON[m.group("tf")]
    try:
        df = pd.read_csv(f)
        df.columns = [c.lower() for c in df.columns]
        d0 = pd.to_datetime(df["time"], unit="s").min().date()
        d1 = pd.to_datetime(df["time"], unit="s").max().date()
        rows.append((ticker, tf, len(df), str(d0), str(d1), f.name))
    except Exception as e:
        print("ERR", f.name, e)

cat = pd.DataFrame(rows, columns=["ticker", "tf", "bars", "start", "end", "file"])
print("\n=== by (ticker, tf) ===")
print(cat[["ticker", "tf", "bars", "start", "end"]].to_string(index=False))

print("\n=== tickers available per timeframe ===")
for tf in ["1D", "240", "60", "1m"]:
    ts = sorted(cat.loc[cat.tf == tf, "ticker"].tolist())
    print(f"  {tf}: {len(ts)} -> {ts}")

print("\n=== duplicate (ticker,tf) (need de-dup) ===")
dups = cat.groupby(["ticker", "tf"]).size()
print(dups[dups > 1].to_string() or "(none)")
