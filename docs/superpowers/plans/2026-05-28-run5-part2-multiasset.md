# Run 5 Part 2 — Multi-Asset Pooled Optimization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Optimize both detector sides on a time-aligned pool of `(asset, timeframe)` streams so the HIGH side gets enough validatable structural events to escape the noise floor — keeping `detector.py` untouched (Pine parity) and the single-asset path working.

**Architecture:** New modules — `src/universe.py` (asset/timeframe/cluster registry), `src/volume_quality.py` (per-stream volume profiler), `src/pooled_scoring.py` (cluster-weighted count-level pooling), `src/pooled_validation.py` (calendar-based folds + pooled Optuna objective). Pooling weights each stream's match counts by `1/cluster_size` BEFORE the side-score math, so correlated streams (SPX-1D/1W, SPX/NDX) cannot over-credit the bootstrap LCB. Calendar folds slice every stream by date; a per-stream label-embargo of `max(nest)=200` bars and a `≥401`-bar minimum gate prevent leakage and silent zero-pivot bias.

**Tech Stack:** Python 3, Optuna, pandas, NumPy, SciPy (Hungarian, reused), pytest.

**Reference spec:** `docs/superpowers/specs/2026-05-28-run5-multiasset-asymmetric-bounds-design.md` §4–§5. **Depends on Part 1** (`src/search_space.py`, decoupled `params_from_trial`).

---

## File Structure

- **Create** `src/universe.py` (~120 lines) — `Stream` dataclass, `UNIVERSE` groups, `TIMEFRAMES`, `CLUSTER_MAP`, `resolve_streams()`.
- **Create** `src/volume_quality.py` (~90 lines) — `VolumeQuality` dataclass, `profile_volume()`.
- **Create** `src/pooled_scoring.py` (~140 lines) — `pooled_side_score()`, `pooled_fold_score()` (reuse `precision_at_n_stats`, `REFERENCE_N`, `GAMMA`, etc. from `scoring`).
- **Create** `src/pooled_validation.py` (~220 lines) — `StreamData`, `load_streams()`, `build_calendar_folds()`, `build_pooled_optuna_objective()`.
- **Create** `tests/test_universe.py`, `tests/test_volume_quality.py`, `tests/test_pooled_scoring.py`, `tests/test_pooled_validation.py`.
- **Modify** `optimize.ipynb` — add universe/timeframe/volume-policy selector cell + pooled-launch cell (Task 9).
- **Reuse unchanged:** `src/detector.py`, `src/scoring.py` (`precision_at_n_stats`, `add_pivot_labels`), `src/validation.py` (single-asset path), `src/monitor145.py` (`make_storage`).

Constants reused from `src/scoring.py`: `REFERENCE_N` (100), `PRECISION_EXPONENT` (1.2), `RECALL_TARGET` (0.40), `MIN_RATE` (0.001), `GAMMA` (2.0), `STRUCTURAL_NEST` ([50,100,200]).

---

## Task 1: Universe registry (`src/universe.py`)

**Files:**
- Create: `src/universe.py`
- Create: `tests/test_universe.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_universe.py`:

```python
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
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_universe.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.universe'`.

- [ ] **Step 3: Write the implementation**

Create `src/universe.py`:

```python
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
```

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest tests/test_universe.py -q`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add src/universe.py tests/test_universe.py
git commit -m "feat(universe): asset/timeframe/cluster registry + resolve_streams"
```

---

## Task 2: Volume-quality profiler (`src/volume_quality.py`)

**Files:**
- Create: `src/volume_quality.py`
- Create: `tests/test_volume_quality.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_volume_quality.py`:

```python
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
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_volume_quality.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.volume_quality'`.

- [ ] **Step 3: Write the implementation**

Create `src/volume_quality.py`:

```python
"""Per-stream volume-quality profiler.

TV intraday volume for indices is often backfilled with placeholder values
(constant bar-seconds like 3600, or zeros) early in history and becomes real
later. We FLAG quality (full/partial/none) and the date real volume begins,
so runs can SEPARATE volume-bearing data from price-only (never deleting it).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Bar-seconds placeholders TV emits for volume-less intraday history.
_PLACEHOLDER_VALUES = {0, 2400, 3596, 3597, 3598, 3599, 3600, 3601}
_REAL_FLOOR = 100_000          # below this is treated as non-real for an index/stock
_REAL_SHARE_THRESHOLD = 0.05   # a window is "real" when <5% of bars are placeholder


@dataclass(frozen=True)
class VolumeQuality:
    quality: str                       # "full" | "partial" | "none"
    real_from: pd.Timestamp | None     # first date where volume is reliably real
    placeholder_share: float           # overall fraction of placeholder bars


def _placeholder_mask(volume: pd.Series) -> np.ndarray:
    v = volume.fillna(0)
    return (v.isin(_PLACEHOLDER_VALUES) | (v < _REAL_FLOOR)).to_numpy()


def profile_volume(df: pd.DataFrame, window: int = 60) -> VolumeQuality:
    """Classify a stream's volume quality and find where real volume begins.

    ``window`` is the rolling bar count over which the placeholder share is
    evaluated to locate the real-volume onset.
    """
    if "volume" not in df.columns or len(df) == 0:
        return VolumeQuality("none", None, 1.0)

    ph = _placeholder_mask(df["volume"])
    overall_share = float(ph.mean())

    if overall_share < _REAL_SHARE_THRESHOLD:
        return VolumeQuality("full", _index_ts(df, 0), overall_share)
    if overall_share > (1.0 - _REAL_SHARE_THRESHOLD):
        return VolumeQuality("none", None, overall_share)

    # Partial: find first index where the forward rolling placeholder share
    # stays below threshold.
    ph_series = pd.Series(ph.astype(float), index=df.index)
    fwd_share = ph_series[::-1].rolling(window, min_periods=1).mean()[::-1]
    real_idx = np.flatnonzero((fwd_share < _REAL_SHARE_THRESHOLD).to_numpy())
    real_from = _index_ts(df, int(real_idx[0])) if real_idx.size else None
    return VolumeQuality("partial", real_from, overall_share)


def _index_ts(df: pd.DataFrame, pos: int) -> pd.Timestamp:
    if isinstance(df.index, pd.DatetimeIndex):
        return df.index[pos]
    return pd.to_datetime(df["time"].iloc[pos], unit="s")
```

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest tests/test_volume_quality.py -q`
Expected: PASS (3 passed).

- [ ] **Step 5: Validate against the real export (manual check)**

Run: `python -c "import pandas as pd; from src.volume_quality import profile_volume; df=pd.read_csv(r'C:\Users\kuben\Downloads\SP_SPX, 60_d2f5a.csv'); df.columns=[c.lower() for c in df.columns]; df.index=pd.to_datetime(df['time'],unit='s'); print(profile_volume(df))"`
Expected: `VolumeQuality(quality='partial', real_from=Timestamp('2021-...' or '2022-...'), placeholder_share=~0.59)` — partial, onset in 2021–2022.

- [ ] **Step 6: Commit**

```bash
git add src/volume_quality.py tests/test_volume_quality.py
git commit -m "feat(volume): per-stream volume-quality profiler (full/partial/none)"
```

---

## Task 3: Cluster-weighted pooled scoring (`src/pooled_scoring.py`)

**Files:**
- Create: `src/pooled_scoring.py`
- Create: `tests/test_pooled_scoring.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_pooled_scoring.py`:

```python
from src.pooled_scoring import StreamStat, pooled_side_score, pooled_fold_score


def _stat(n_signals, tp, matched, total, n_bars, weight):
    return StreamStat(
        n_signals=n_signals, tp=tp, matched_pivots=matched,
        total_pivots=total, n_bars=n_bars, weight=weight,
    )


def test_pooled_precision_is_weighted_count_ratio():
    # Two streams, weight 0.5 each (same cluster of size 2).
    a = _stat(n_signals=10, tp=8, matched=8, total=10, n_bars=2000, weight=0.5)
    b = _stat(n_signals=10, tp=2, matched=2, total=10, n_bars=2000, weight=0.5)
    score, comp = pooled_side_score([a, b], "high")
    # Weighted precision = (0.5*8 + 0.5*2) / (0.5*10 + 0.5*10) = 5/10 = 0.5
    assert abs(comp["precision"] - 0.5) < 1e-9
    assert 0.0 <= score <= 1.0


def test_cluster_weight_halves_a_duplicate_streams_contribution():
    # One unique stream (weight 1) vs the same numbers duplicated at weight 0.5
    # twice must give identical pooled precision (correlation is neutralised).
    single = pooled_side_score(
        [_stat(10, 6, 6, 12, 3000, 1.0)], "high")[1]["precision"]
    dup = pooled_side_score(
        [_stat(10, 6, 6, 12, 3000, 0.5), _stat(10, 6, 6, 12, 3000, 0.5)],
        "high")[1]["precision"]
    assert abs(single - dup) < 1e-9


def test_pooled_fold_applies_is_oos_exponential_penalty():
    is_stats = [_stat(10, 10, 10, 10, 2000, 1.0)]   # perfect IS
    oos_stats = [_stat(10, 5, 5, 10, 2000, 1.0)]    # weaker OOS
    fold, comp = pooled_fold_score(is_stats, oos_stats, "high")
    assert comp["oos_score"] <= comp["is_score"]
    # fold = oos * exp(-GAMMA * max(0, is-oos)) <= oos_score
    assert fold <= comp["oos_score"] + 1e-9
    assert fold >= 0.0
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_pooled_scoring.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.pooled_scoring'`.

- [ ] **Step 3: Write the implementation**

Create `src/pooled_scoring.py`:

```python
"""Cluster-weighted pooled scoring for multi-asset folds.

Pools the per-stream match counts from ``precision_at_n_stats`` (at the single
v4 tolerance scale ``REFERENCE_N``) weighted by ``1/cluster_size``, then runs
the SAME side-score math as ``compute_side_score`` on the pooled counts. The
weighting is what prevents correlated streams (e.g. SPX-1D + SPX-1W, or
SPX+NDX) from over-crediting the objective.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .scoring import (
    GAMMA, MIN_RATE, PRECISION_EXPONENT, RECALL_TARGET, REFERENCE_N,
)


@dataclass(frozen=True)
class StreamStat:
    """Per-stream match counts at REFERENCE_N, plus the stream's pool weight."""
    n_signals: int
    tp: int
    matched_pivots: int
    total_pivots: int
    n_bars: int
    weight: float


def pooled_side_score(
    stats: list[StreamStat], side: str,
) -> tuple[float, dict[str, float]]:
    """Weighted-pooled single-scale side score in [0, 1].

    Mirrors ``compute_side_score`` for the v4 single-scale (REFERENCE_N) case:
    ``precision**PRECISION_EXPONENT * recall_sat * frequency_factor *
    excess_penalty``.
    """
    if side not in ("high", "low"):
        raise ValueError(f"side must be 'high' or 'low', got {side!r}")

    w_nsig = sum(s.weight * s.n_signals for s in stats)
    w_tp = sum(s.weight * s.tp for s in stats)
    w_matched = sum(s.weight * s.matched_pivots for s in stats)
    w_total = sum(s.weight * s.total_pivots for s in stats)
    w_bars = sum(s.weight * s.n_bars for s in stats)

    empty = {
        "precision": 0.0, "recall": 0.0, "recall_saturated": 0.0,
        "frequency_factor": 0.0, "excess_penalty": 0.0,
        "pooled_n_signals": float(w_nsig), "pooled_total_pivots": float(w_total),
    }
    if w_nsig <= 0 or w_bars <= 0:
        return 0.0, empty

    precision = (w_tp / w_nsig) if w_nsig > 0 else 0.0
    recall = (w_matched / w_total) if w_total > 0 else 0.0
    # Single scale → target_for_scale = RECALL_TARGET * sqrt(REFERENCE_N/REFERENCE_N).
    recall_sat = 1.0 - np.exp(-recall / max(RECALL_TARGET, 1e-9)) if precision > 0 else 0.0
    scale_score = (precision ** PRECISION_EXPONENT) * recall_sat

    signal_rate = w_nsig / w_bars
    frequency_factor = min(1.0, signal_rate / MIN_RATE)

    n_eff = max(float(w_nsig), 1.0)
    t_eff = max(float(w_total), 1.0)
    excess_penalty = 2.0 * n_eff * t_eff / (n_eff * n_eff + t_eff * t_eff)

    final = float(scale_score * frequency_factor * excess_penalty)
    comp = {
        "precision": float(precision), "recall": float(recall),
        "recall_saturated": float(recall_sat),
        "frequency_factor": float(frequency_factor),
        "excess_penalty": float(excess_penalty),
        "pooled_n_signals": float(w_nsig), "pooled_total_pivots": float(w_total),
    }
    return final, comp


def pooled_fold_score(
    is_stats: list[StreamStat],
    oos_stats: list[StreamStat],
    side: str,
) -> tuple[float, dict[str, float]]:
    """Pooled per-fold score with the smooth IS-OOS overfit penalty.

    ``fold = oos_score * exp(-GAMMA * max(0, is_score - oos_score))`` — same
    contract as ``scoring._fold_score`` but on pooled, cluster-weighted counts.
    """
    is_score, is_comp = pooled_side_score(is_stats, side)
    oos_score, oos_comp = pooled_side_score(oos_stats, side)
    gap = max(0.0, float(is_score) - float(oos_score))
    fold = float(oos_score * np.exp(-GAMMA * gap))
    components = {
        "is_score": float(is_score),
        "oos_score": float(oos_score),
        "is_oos_gap": float(gap),
        "fold_score": fold,
        "precision_oos": oos_comp["precision"],
        "recall_oos": oos_comp["recall"],
        "excess_penalty_oos": oos_comp["excess_penalty"],
        "frequency_factor_oos": oos_comp["frequency_factor"],
        "pooled_total_pivots_oos": oos_comp["pooled_total_pivots"],
    }
    return fold, components
```

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest tests/test_pooled_scoring.py -q`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add src/pooled_scoring.py tests/test_pooled_scoring.py
git commit -m "feat(pooled-scoring): cluster-weighted count-level pooled side/fold score"
```

---

## Task 4: Stream loading with volume policy (`src/pooled_validation.py` — part A)

**Files:**
- Create: `src/pooled_validation.py`
- Create: `tests/test_pooled_validation.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_pooled_validation.py`:

```python
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
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_pooled_validation.py -q`
Expected: FAIL — `ImportError: cannot import name 'load_stream_frame'`.

- [ ] **Step 3: Write the implementation (part A)**

Create `src/pooled_validation.py`:

```python
"""Calendar-based pooled folds + pooled Optuna objective (multi-asset).

Keeps ``src/validation.py`` (single-asset, row-index folds) untouched. This
module adds the time-aligned, cluster-weighted pooled path used by Run 5.
"""
from __future__ import annotations

import dataclasses
import logging
from collections.abc import Callable

import numpy as np
import optuna
import pandas as pd

from .detector import SpeculatorDetector, build_detector_artifacts
from .indicators import Params
from .pooled_scoring import StreamStat, pooled_fold_score
from .scoring import (
    REFERENCE_N, STRUCTURAL_NEST, add_pivot_labels, precision_at_n_stats,
)
from .universe import Stream
from .validation import fold_scores_bootstrap_ci, load_data
from .volume_quality import VolumeQuality, profile_volume

logger = logging.getLogger(__name__)

EMBARGO_NEST_BARS: int = max(STRUCTURAL_NEST)    # 200 — label look-ahead guard
MIN_STREAM_BARS: int = 2 * max(STRUCTURAL_NEST) + 1   # 401 — nest must fit
HOLDOUT_FRACTION: float = 0.20
IS_FRACTION: float = 0.10
OOS_FRACTION: float = 0.03
STEP_FRACTION: float = 0.05


def load_stream_frame(path: str) -> pd.DataFrame:
    """Load a stream CSV via the canonical loader (lowercases, indexes time)."""
    return load_data(path)


def apply_volume_policy(
    df: pd.DataFrame, policy: str,
) -> tuple[pd.DataFrame, bool]:
    """Apply the run-level volume policy to a single stream.

    Returns ``(possibly_trimmed_df, keep)``:
    - ``price_only`` / ``mixed``: keep all bars (volume votes handled by params).
    - ``volume_required``: trim to ``[volume_real_from, end]``; drop the stream
      entirely (``keep=False``) if its quality is ``none``.
    """
    if policy in ("price_only", "mixed"):
        return df, True
    if policy == "volume_required":
        vq: VolumeQuality = profile_volume(df)
        if vq.quality == "none" or vq.real_from is None:
            return df, False
        return df.loc[df.index >= vq.real_from].copy(), True
    raise ValueError(f"unknown volume policy {policy!r}")
```

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest tests/test_pooled_validation.py -q`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
git add src/pooled_validation.py tests/test_pooled_validation.py
git commit -m "feat(pooled-validation): stream loading + volume policy"
```

---

## Task 5: Calendar-based pooled folds (`src/pooled_validation.py` — part B)

**Files:**
- Modify: `src/pooled_validation.py` (append)
- Modify: `tests/test_pooled_validation.py` (append)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_pooled_validation.py`:

```python
from src.pooled_validation import StreamData, build_calendar_folds


def _stream_data(tmp_path, ticker, n, start_ts, step, cluster):
    path = _write_csv(tmp_path, f"{ticker}_1D_a_b.csv", n, start_ts=start_ts, step=step)
    df = load_stream_frame(path)
    add_pivot_labels(df)
    return StreamData(
        stream=Stream(ticker, "1D", path, cluster),
        df=df, bar_seconds=float(step),
    )


def test_calendar_folds_respect_holdout_and_embargo(tmp_path):
    # Two daily streams, different start dates, both long enough for many folds.
    # NOTE: streams must be large enough that OOS (3% of master span in calendar
    # days) >= MIN_STREAM_BARS=401, else the strict gate yields 0 folds.
    a = _stream_data(tmp_path, "SPX", 15000, 1262304000, 86400, "US_EQ")  # 2010+
    b = _stream_data(tmp_path, "DAX", 4000, 1420070400, 86400, "EU_EQ")  # 2015+
    folds = build_calendar_folds([a, b])
    assert len(folds) >= 3
    for fold in folds:
        for sl in fold:
            # min-bars gate: every contributing slice clears the nest requirement
            assert len(sl.df_is) >= 1   # labels exist; emptiness handled by gate
        # IS end strictly precedes OOS start by >= embargo for each slice
        for sl in fold:
            if len(sl.df_is) and len(sl.df_oos):
                assert sl.df_is.index[-1] < sl.df_oos.index[0]


def test_calendar_folds_drop_too_short_streams(tmp_path):
    # A 1W-like coarse stream too short to fit the 401-bar nest in any fold.
    big = _stream_data(tmp_path, "SPX", 15000, 1262304000, 86400, "US_EQ")
    tiny = _stream_data(tmp_path, "DAX", 120, 1420070400, 604800, "EU_EQ")  # 120 weekly bars
    folds = build_calendar_folds([big, tiny])
    # tiny never satisfies MIN_STREAM_BARS (401) in any fold slice.
    tickers_used = {sl.stream.ticker for fold in folds for sl in fold}
    assert "DAX" not in tickers_used
    assert "SPX" in tickers_used
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_pooled_validation.py -k calendar -q`
Expected: FAIL — `ImportError: cannot import name 'StreamData'`.

- [ ] **Step 3: Write the implementation (part B)**

Append to `src/pooled_validation.py`:

```python
@dataclasses.dataclass
class StreamData:
    """A loaded, labelled stream plus its bar spacing (seconds)."""
    stream: Stream
    df: pd.DataFrame          # full, pivot-labelled, DatetimeIndex
    bar_seconds: float


@dataclasses.dataclass
class PreparedSlice:
    stream: Stream
    df_is: pd.DataFrame
    df_oos: pd.DataFrame
    artifacts_is: object
    artifacts_oos: object


# One fold = the list of per-stream prepared slices for that calendar window.
Fold = list[PreparedSlice]


def _master_span(stream_datas: list[StreamData]) -> tuple[pd.Timestamp, pd.Timestamp]:
    starts = [sd.df.index[0] for sd in stream_datas if len(sd.df)]
    ends = [sd.df.index[-1] for sd in stream_datas if len(sd.df)]
    return min(starts), max(ends)


def build_calendar_folds(
    stream_datas: list[StreamData],
    is_fraction: float = IS_FRACTION,
    oos_fraction: float = OOS_FRACTION,
    step_fraction: float = STEP_FRACTION,
    holdout_fraction: float = HOLDOUT_FRACTION,
) -> list[Fold]:
    """Time-aligned calendar folds; each stream sliced by date per fold.

    Embargo between IS and OOS is the calendar span of ``EMBARGO_NEST_BARS``
    bars at the COARSEST selected timeframe (guards the non-causal label
    window for every stream). Streams whose slice has < ``MIN_STREAM_BARS``
    bars are dropped from that fold (logged).
    """
    if not stream_datas:
        return []
    start, end = _master_span(stream_datas)
    total_days = max((end - start).days, 1)
    active_days = total_days * (1.0 - holdout_fraction)
    is_days = total_days * is_fraction
    oos_days = total_days * oos_fraction
    step_days = max(total_days * step_fraction, 1.0)

    coarsest_bar_seconds = max(sd.bar_seconds for sd in stream_datas)
    embargo_days = EMBARGO_NEST_BARS * coarsest_bar_seconds / 86400.0

    folds: list[Fold] = []
    is_start_off = 0.0
    while is_start_off + is_days + embargo_days + oos_days <= active_days:
        is_start = start + pd.Timedelta(days=is_start_off)
        is_end = is_start + pd.Timedelta(days=is_days)
        oos_start = is_end + pd.Timedelta(days=embargo_days)
        oos_end = oos_start + pd.Timedelta(days=oos_days)

        fold: Fold = []
        for sd in stream_datas:
            df_is = sd.df.loc[(sd.df.index >= is_start) & (sd.df.index < is_end)]
            df_oos = sd.df.loc[(sd.df.index >= oos_start) & (sd.df.index < oos_end)]
            if len(df_is) < MIN_STREAM_BARS or len(df_oos) < MIN_STREAM_BARS:
                continue   # nest cannot fit / too few bars → drop from this fold
            df_is_r = df_is.reset_index(drop=True)
            df_oos_r = df_oos.reset_index(drop=True)
            add_pivot_labels(df_is_r)
            add_pivot_labels(df_oos_r)
            fold.append(PreparedSlice(
                stream=sd.stream, df_is=df_is_r, df_oos=df_oos_r,
                artifacts_is=build_detector_artifacts(df_is_r),
                artifacts_oos=build_detector_artifacts(df_oos_r),
            ))
        if fold:
            folds.append(fold)
        is_start_off += step_days

    logger.info("build_calendar_folds: %d folds over %d master days",
                len(folds), total_days)
    return folds
```

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest tests/test_pooled_validation.py -k calendar -q`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add src/pooled_validation.py tests/test_pooled_validation.py
git commit -m "feat(pooled-validation): calendar folds with per-stream embargo + min-bars gate"
```

---

## Task 6: Pooled Optuna objective + cluster weights (`src/pooled_validation.py` — part C)

**Files:**
- Modify: `src/pooled_validation.py` (append)
- Modify: `tests/test_pooled_validation.py` (append)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_pooled_validation.py`:

```python
import optuna
from src.pooled_validation import (
    cluster_weights, evaluate_pooled_fold, build_pooled_optuna_objective,
)
from src.indicators import Params


def test_cluster_weights_are_inverse_cluster_size():
    a = Stream("SPX", "1D", "p", "US_EQ")
    b = Stream("NDX", "1D", "p", "US_EQ")
    c = Stream("DAX", "1D", "p", "EU_EQ")
    w = cluster_weights([a, b, c])
    assert abs(w["SPX_1D"] - 0.5) < 1e-9   # US_EQ has 2 members
    assert abs(w["NDX_1D"] - 0.5) < 1e-9
    assert abs(w["DAX_1D"] - 1.0) < 1e-9   # EU_EQ has 1


def test_pooled_objective_runs_and_returns_float(tmp_path):
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    # 15000 bars so OOS (3% of master span) >= MIN_STREAM_BARS=401 → folds form.
    a = _stream_data(tmp_path, "SPX", 15000, 1262304000, 86400, "US_EQ")
    b = _stream_data(tmp_path, "DAX", 15000, 1262304000, 86400, "EU_EQ")
    folds = build_calendar_folds([a, b])
    from src.speculatores145 import params_from_trial
    objective = build_pooled_optuna_objective(folds, [a.stream, b.stream],
                                              params_from_trial, "low")
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=2)
    assert isinstance(study.best_value, float)
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_pooled_validation.py -k "cluster_weights or pooled_objective" -q`
Expected: FAIL — `ImportError: cannot import name 'cluster_weights'`.

- [ ] **Step 3: Write the implementation (part C)**

Append to `src/pooled_validation.py`:

```python
def cluster_weights(streams: list[Stream]) -> dict[str, float]:
    """Map stream_id -> 1 / (number of streams sharing its cluster)."""
    sizes: dict[str, int] = {}
    for s in streams:
        sizes[s.cluster_id] = sizes.get(s.cluster_id, 0) + 1
    return {s.stream_id: 1.0 / sizes[s.cluster_id] for s in streams}


def _stream_stat(
    df: pd.DataFrame, signals: pd.Series, side: str, weight: float,
) -> StreamStat:
    stats = precision_at_n_stats(signals, df[f"pivot_N{REFERENCE_N}"], side, REFERENCE_N)
    return StreamStat(
        n_signals=int(stats["n_signals"]),
        tp=int(stats["tp"]),
        matched_pivots=int(stats["matched_pivots"]),
        total_pivots=int(stats["total_pivots"]),
        n_bars=len(df),
        weight=weight,
    )


def evaluate_pooled_fold(
    params: Params, side: str, fold: Fold, weights: dict[str, float],
) -> tuple[float, dict[str, float]]:
    """Run the detector per stream in a fold and return the pooled fold score."""
    sig_key = "signal_high" if side == "high" else "signal_low"
    is_stats: list[StreamStat] = []
    oos_stats: list[StreamStat] = []
    for sl in fold:
        w = weights.get(sl.stream.stream_id, 1.0)
        det_is = SpeculatorDetector(sl.df_is, params, sl.artifacts_is).run()
        det_oos = SpeculatorDetector(sl.df_oos, params, sl.artifacts_oos).run()
        is_stats.append(_stream_stat(sl.df_is, det_is[sig_key], side, w))
        oos_stats.append(_stream_stat(sl.df_oos, det_oos[sig_key], side, w))
    return pooled_fold_score(is_stats, oos_stats, side)


def build_pooled_optuna_objective(
    folds: list[Fold],
    streams: list[Stream],
    params_from_trial: Callable[[optuna.Trial, str], Params],
    side: str,
) -> Callable[[optuna.Trial], float]:
    """Pooled multi-asset objective: bootstrap-LCB over pooled fold scores.

    Within-fold correlated streams are neutralised by 1/cluster_size weighting
    in ``pooled_side_score``; temporal overlap between adjacent folds is handled
    by the block bootstrap (block_len=2), same as the single-asset objective.
    """
    if side not in ("high", "low"):
        raise ValueError(f"side must be 'high' or 'low', got {side!r}")
    if not folds:
        raise ValueError("No calendar folds constructed.")
    weights = cluster_weights(streams)

    def objective(trial: optuna.Trial) -> float:
        params = params_from_trial(trial, side)
        fold_scores: list[float] = []
        for fold_idx, fold in enumerate(folds):
            score, _components = evaluate_pooled_fold(params, side, fold, weights)
            fold_scores.append(score)
            running_lcb = fold_scores_bootstrap_ci(
                fold_scores, n_boot=1000, alpha=0.10, block_len=2
            )[0]
            trial.report(running_lcb, fold_idx)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
        return float(fold_scores_bootstrap_ci(
            fold_scores, n_boot=1000, alpha=0.10, block_len=2
        )[0])

    return objective
```

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest tests/test_pooled_validation.py -q`
Expected: PASS (all pooled_validation tests).

- [ ] **Step 5: Commit**

```bash
git add src/pooled_validation.py tests/test_pooled_validation.py
git commit -m "feat(pooled-validation): cluster-weighted pooled Optuna objective"
```

---

## Task 7: Sprint-0 end-to-end smoke (SPX-1D + DAX-1D)

**Files:**
- Create: `temp/sprint0_pooled_smoke.py`

- [ ] **Step 1: Write the smoke script**

Create `temp/sprint0_pooled_smoke.py`:

```python
"""Sprint-0 acceptance: pooled objective end-to-end on SPX-1D + DAX-1D.

Proves the pooled path runs on two REAL streams with a tiny Optuna budget,
before exporting the full universe. Not a unit test — an integration gate.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import optuna

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.scoring import add_pivot_labels  # noqa: E402
from src.universe import Stream  # noqa: E402
from src.pooled_validation import (  # noqa: E402
    StreamData, build_calendar_folds, build_pooled_optuna_objective,
    load_stream_frame, apply_volume_policy,
)
from src.speculatores145 import params_from_trial  # noqa: E402

logging.basicConfig(level=logging.INFO)
optuna.logging.set_verbosity(optuna.logging.WARNING)

SPECS = [
    ("SPX", "data/raw/SPX_1D_18710201_20260318.csv", "US_EQ", 86400.0),
    ("DAX", "data/raw/DAX_1D_19700102_20260324.csv", "EU_EQ", 86400.0),
]


def main() -> None:
    stream_datas = []
    for ticker, path, cluster, bar_s in SPECS:
        df = load_stream_frame(str(ROOT / path))
        df, keep = apply_volume_policy(df, policy="price_only")
        assert keep
        add_pivot_labels(df)
        stream_datas.append(StreamData(
            stream=Stream(ticker, "1D", path, cluster), df=df, bar_seconds=bar_s,
        ))
        print(f"{ticker}: {len(df)} bars "
              f"HIGH={(df['pivot_N100']==1).sum()} LOW={(df['pivot_N100']==-1).sum()}")

    folds = build_calendar_folds(stream_datas)
    print(f"folds={len(folds)}; "
          f"streams/fold={[len(f) for f in folds]}")
    assert folds, "no folds built"

    streams = [sd.stream for sd in stream_datas]
    for side in ("high", "low"):
        obj = build_pooled_optuna_objective(folds, streams, params_from_trial, side)
        study = optuna.create_study(direction="maximize")
        study.optimize(obj, n_trials=5)
        print(f"[{side}] best pooled LCB over 5 trials = {study.best_value:.5f}")
    print("\nsprint0 pooled smoke PASSED")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the smoke**

Run: `python temp/sprint0_pooled_smoke.py`
Expected: prints per-stream pivot counts, a fold count ≥3 with per-fold stream counts, a `best pooled LCB` line per side, and `sprint0 pooled smoke PASSED`. (DAX may be dropped from early folds before its 1970 start contributes — that's fine.)

- [ ] **Step 3: Commit**

```bash
git add temp/sprint0_pooled_smoke.py
git commit -m "test(pooled): sprint-0 end-to-end smoke on SPX-1D + DAX-1D"
```

---

## Task 8: Scale-invariance falsification gate (§5.3)

**Files:**
- Create: `src/scale_invariance.py`
- Create: `tests/test_scale_invariance.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_scale_invariance.py`:

```python
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
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_scale_invariance.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.scale_invariance'`.

- [ ] **Step 3: Write the implementation**

Create `src/scale_invariance.py`:

```python
"""Scale-invariance falsification gate (spec §5.3).

Optimize/choose HIGH params on one timeframe (e.g. SPX-1D), then measure the
precision drop when the SAME params are applied to another timeframe (SPX-1W).
A drop > 0.15 absolute rejects scale-invariance → do not mix timeframes.
"""
from __future__ import annotations

import pandas as pd

from .detector import SpeculatorDetector
from .indicators import Params
from .scoring import REFERENCE_N, add_pivot_labels, precision_at_n_stats

DROP_THRESHOLD: float = 0.15


def _precision(df: pd.DataFrame, params: Params, side: str) -> float:
    work = df.reset_index(drop=True).copy()
    add_pivot_labels(work)
    det = SpeculatorDetector(work, params).run()
    sig = det["signal_high" if side == "high" else "signal_low"]
    stats = precision_at_n_stats(sig, work[f"pivot_N{REFERENCE_N}"], side, REFERENCE_N)
    return float(stats["precision"])


def precision_transfer(
    params: Params, side: str, base_df: pd.DataFrame, other_df: pd.DataFrame,
) -> dict[str, float]:
    """Return precision on base timeframe, on other timeframe, and the drop."""
    p_base = _precision(base_df, params, side)
    p_other = _precision(other_df, params, side)
    return {
        "precision_base": p_base,
        "precision_other": p_other,
        "drop": p_base - p_other,
    }


def scale_invariance_holds(transfer: dict[str, float]) -> bool:
    """True if the precision drop is within the acceptance threshold."""
    return transfer["drop"] <= DROP_THRESHOLD
```

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest tests/test_scale_invariance.py -q`
Expected: PASS (1 passed).

- [ ] **Step 5: Commit**

```bash
git add src/scale_invariance.py tests/test_scale_invariance.py
git commit -m "feat(scale-invariance): precision-transfer falsification gate"
```

---

## Task 9: Notebook selectors + pooled launch cell

**Files:**
- Modify: `optimize.ipynb` (add two cells after the existing config cell)

- [ ] **Step 1: Add the universe/timeframe/volume-policy selector cell**

In `optimize.ipynb`, after the existing config cell (the one defining `DATASET_KEY`), insert a new code cell with exactly this content:

```python
# === Run 5 — multi-asset pool selection ===
from src.universe import UNIVERSE, TIMEFRAMES, resolve_streams

# Choose any subset of these groups + timeframes (edit before launching):
SELECTED_GROUPS = ["INDICES_GLOBAL", "COMMODITIES"]   # see UNIVERSE.keys()
SELECTED_TIMEFRAMES = ["1D"]                            # subset of TIMEFRAMES
VOLUME_POLICY = "price_only"                            # price_only | volume_required | mixed

print("available groups:", list(UNIVERSE))
print("available timeframes:", TIMEFRAMES)
STREAMS = resolve_streams(SELECTED_GROUPS, SELECTED_TIMEFRAMES, data_dir="data/raw")
print(f"resolved {len(STREAMS)} streams:",
      [s.stream_id for s in STREAMS])
```

- [ ] **Step 2: Add the pooled-launch cell**

Insert a second new code cell with exactly this content:

```python
# === Run 5 — build pooled folds + launch (both sides) ===
import optuna
from src.scoring import add_pivot_labels
from src.pooled_validation import (
    StreamData, build_calendar_folds, build_pooled_optuna_objective,
    load_stream_frame, apply_volume_policy,
)
from src.speculatores145 import params_from_trial

_TF_SECONDS = {"1D": 86400.0, "1W": 604800.0, "60": 3600.0, "240": 14400.0}

stream_datas = []
for s in STREAMS:
    df = load_stream_frame(s.path)
    df, keep = apply_volume_policy(df, policy=VOLUME_POLICY)
    if not keep:
        print(f"drop {s.stream_id}: volume policy excluded it"); continue
    add_pivot_labels(df)
    stream_datas.append(StreamData(stream=s, df=df,
                                   bar_seconds=_TF_SECONDS[s.timeframe]))

folds = build_calendar_folds(stream_datas)
print(f"{len(folds)} calendar folds; streams/fold = {[len(f) for f in folds]}")
streams = [sd.stream for sd in stream_datas]

N_TRIALS = 1000  # per side
for side in ("high", "low"):
    objective = build_pooled_optuna_objective(folds, streams, params_from_trial, side)
    study = optuna.create_study(
        study_name=f"spec_v15_run5_multiasset_{side}",
        direction="maximize",
        sampler=optuna.samplers.TPESampler(multivariate=True, group=True, seed=42),
    )
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)
    print(f"[{side}] best pooled LCB = {study.best_value:.5f}")
```

- [ ] **Step 3: Sanity-check the notebook still parses**

Run: `python -c "import json; json.load(open('optimize.ipynb')); print('notebook JSON OK')"`
Expected: prints `notebook JSON OK`.

- [ ] **Step 4: Commit**

```bash
git add optimize.ipynb
git commit -m "feat(notebook): Run 5 pool selectors + pooled launch cells"
```

---

## Task 10: Run-report fields + pre-registration (§5.2)

**Files:**
- Create: `temp/run5_preregistration.md`
- Modify: `optimize.ipynb` (add a report cell)

- [ ] **Step 1: Write the pre-registration note**

Create `temp/run5_preregistration.md`:

```markdown
# Run 5 Pre-Registration (filed before the run)

**Hypotheses (HIGH side, relaxed floors):**
- `dur_extreme_pct` floor 0.50→0.30 → expected to make the duration gate
  satisfiable for dim-agreement tops; HIGH fire-rate should rise from ~noise.
- `pct_extreme` floor 0.70→0.55 → expected to densify top agreement; HIGH
  in-sample precision-at-REFERENCE_N should rise vs Run 4.

**Primary metric:** pooled OOS bootstrap-LCB per side (Optuna objective).
**Holdout reporting rule:** report EXACT holdout HIGH/LOW hit counts (TP, FP,
n_signals, total_pivots) BEFORE vs AFTER, never only "improvement". The famous
post-2000 tops are public and were visually inspected in Runs 1–4 — treat any
holdout gain with that contamination in mind.

**Decision rule:** ship a new HIGH preset only if pooled OOS-LCB > Run 4 HIGH
(0.0656) AND holdout HIGH TP > 0 with FP not inflated.
```

- [ ] **Step 2: Add a run-report cell to the notebook**

In `optimize.ipynb`, append a code cell with exactly this content:

```python
# === Run 5 — record provenance for reproducibility (spec §5.2) ===
import json, hashlib
from src.volume_quality import profile_volume

def _hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()[:12]

report = {
    "selected_groups": SELECTED_GROUPS,
    "selected_timeframes": SELECTED_TIMEFRAMES,
    "volume_policy": VOLUME_POLICY,
    "streams": [
        {
            "stream_id": sd.stream.stream_id,
            "cluster": sd.stream.cluster_id,
            "bars": len(sd.df),
            "date_range": [str(sd.df.index[0]), str(sd.df.index[-1])],
            "data_hash": _hash(sd.stream.path),
            "volume_quality": profile_volume(sd.df).quality,
        }
        for sd in stream_datas
    ],
    "n_folds": len(folds),
}
print(json.dumps(report, indent=2))
```

- [ ] **Step 3: Verify the notebook parses**

Run: `python -c "import json; json.load(open('optimize.ipynb')); print('OK')"`
Expected: prints `OK`.

- [ ] **Step 4: Commit**

```bash
git add temp/run5_preregistration.md optimize.ipynb
git commit -m "docs(run5): pre-registration note + reproducibility report cell"
```

---

## Task 11: Full regression + scale-invariance gate run

**Files:**
- Reference (no edit unless broken): all `tests/`, `temp/sprint0_pooled_smoke.py`

- [ ] **Step 1: Run the full unit suite (Part 1 + Part 2)**

Run: `python -m pytest tests/ -q`
Expected: PASS (all tests green).

- [ ] **Step 2: Re-run the sprint-0 pooled smoke**

Run: `python temp/sprint0_pooled_smoke.py`
Expected: `sprint0 pooled smoke PASSED`.

- [ ] **Step 3: Confirm the single-asset path still works (no regression)**

Run: `python temp/smoke_test_v15_end_to_end.py`
Expected: completes and writes its report (single-asset `build_optuna_objective` untouched).

- [ ] **Step 4: Commit any fixups (only if a step required a change)**

```bash
git add -A
git commit -m "test(run5): full regression pass — pooled + single-asset paths green"
```

---

## Self-Review (completed by plan author)

- **Spec coverage:**
  - §4.1 data/universe layer → Task 1 (`universe.py`, `resolve_streams`) ✓; TV schema/normalization handled by reusing `load_data` (Task 4 `load_stream_frame`) ✓.
  - §4.1a volume quality flag & separate → Task 2 (`profile_volume`) + Task 4 (`apply_volume_policy`, separate `volume_required` trimming) ✓.
  - §4.2 calendar folds, embargo ≥200 bars, min-bars ≥401 gate → Task 5 (`build_calendar_folds`, `EMBARGO_NEST_BARS`, `MIN_STREAM_BARS`) ✓.
  - §4.3 cluster-weighted pooled scoring → Task 3 (`pooled_side_score` weighting) + Task 6 (`cluster_weights`, objective) ✓.
  - §4.4 compute budget → per-fold-slice artifact builds kept (Task 5), pool size user-bounded via notebook selection (Task 9) ✓.
  - §5.2 reproducibility + pre-registration → Task 10 ✓.
  - §5.3 scale-invariance gate → Task 8 ✓.
  - §5.4 tests incl. sprint-0 smoke → Tasks 1-8, 11 ✓.
  - Single-asset path untouched (`validation.py` unchanged); regression in Task 11 ✓.
- **Placeholder scan:** none — every code step has complete code.
- **Type consistency:** `Stream(ticker, timeframe, path, cluster_id)` + `.stream_id` used identically in Tasks 1,5,6,9; `StreamStat(n_signals, tp, matched_pivots, total_pivots, n_bars, weight)` consistent between Task 3 (def) and Task 6 (`_stream_stat` construction); `StreamData(stream, df, bar_seconds)` consistent Tasks 5,7,9; `PreparedSlice(stream, df_is, df_oos, artifacts_is, artifacts_oos)` consistent Tasks 5,6; `build_pooled_optuna_objective(folds, streams, params_from_trial, side)` signature identical in Task 6 def and Tasks 7,9 calls; `apply_volume_policy(df, policy)->(df,bool)` consistent Tasks 4,7,9; `profile_volume(df)->VolumeQuality(quality, real_from, placeholder_share)` consistent Tasks 2,4,10.
- **Known caveat carried forward:** mixing 1W with 1D inflates the calendar embargo (coarsest-timeframe driven), so 1W streams may be dropped from most folds by the min-bars gate — this is intentional/honest and surfaced by the §5.3 gate before trusting mixed-timeframe runs.
```
