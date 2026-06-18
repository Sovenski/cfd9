"""Spec §A — holdout evaluation -> era_pass (plan/holdout-and-pruning-spec.md §A5).

Assertions (never weaken):

- A5.1 holdout slices start >= EMBARGO_NEST_BARS after the active boundary
  and never overlap any fold slice (synthetic 2-stream pool).
- A5.2 era_pass truth table: score 0 -> False; ratio just-below/above
  ERA_PASS_MIN_RATIO -> False/True.
- A5.3 ``pooled_holdout_score`` equals the OOS leg of ``pooled_fold_score``
  on identical stats (pinned by construction).
- A5.4 golden additive-only regression: fold scores and signal arrays are
  unchanged after the holdout path runs (synthetic pool always; on-disk
  golden snapshot when data is present).
- §A4 wiring: ``run_v17_gpu(holdout=True)`` emits the holdout block and the
  era_pass override semantics hold (explicit caller value always wins).
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.detector import SpeculatorDetector
from src.indicators import Params
from src.pooled_scoring import StreamStat, pooled_fold_score
from src.pooled_validation import (
    EMBARGO_NEST_BARS,
    ERA_PASS_MIN_RATIO,
    HOLDOUT_FRACTION,
    MIN_STREAM_BARS,
    StreamData,
    build_calendar_folds,
    build_holdout_slices,
    cluster_weights,
    evaluate_holdout,
    holdout_era_pass,
    load_stream_frame,
    pooled_holdout_score,
)
from src.scoring import add_pivot_labels
from src.universe import Stream
from src.v17_acceptance import raw_fold_scores
from src.v17_optimize import PooledScorer

_REPO = Path(__file__).resolve().parents[1]
_GOLDEN = _REPO / "results" / "diag" / "golden"

#: Fold geometry for the synthetic pool: 900-bar OOS slices, 2 folds
#: (same rationale as temp/capture_baseline.py OOS_FRACTION_GOLDEN).
ERA_KW = {"oos_fraction": 0.15, "step_fraction": 0.5}


# ---------------------------------------------------------------------------
# synthetic 2-stream pool helpers (pattern of tests/test_pooled_validation.py)
# ---------------------------------------------------------------------------


def _write_csv(tmp_path, name, n, start_ts=1262304000, step=86400):
    t = np.arange(n) * step + start_ts
    close = 100 + np.cumsum(np.random.RandomState(7).randn(n))
    df = pd.DataFrame({"time": t, "open": close, "high": close + 1,
                       "low": close - 1, "close": close,
                       "Volume": np.full(n, 1_000_000)})
    p = tmp_path / name
    df.to_csv(p, index=False)
    return str(p)


def _stream_data(tmp_path, ticker, n, start_ts, step, cluster):
    path = _write_csv(tmp_path, f"{ticker}_1D_a_b.csv", n,
                      start_ts=start_ts, step=step)
    df = load_stream_frame(path)
    add_pivot_labels(df)
    return StreamData(stream=Stream(ticker, "1D", path, cluster),
                      df=df, bar_seconds=float(step))


def _pool(tmp_path):
    """Two identical-range 6000-bar daily streams -> era = full range, ref = a."""
    a = _stream_data(tmp_path, "SPX", 6000, 1262304000, 86400, "US_EQ")
    b = _stream_data(tmp_path, "DAX", 6000, 1262304000, 86400, "EU_EQ")
    return a, b


# ---------------------------------------------------------------------------
# A5.1 — holdout boundary: embargo after the active region, no fold overlap
# ---------------------------------------------------------------------------


def test_holdout_slices_start_after_embargo_and_never_overlap_folds(tmp_path):
    a, b = _pool(tmp_path)
    folds = build_calendar_folds([a, b], **ERA_KW)
    slices, meta = build_holdout_slices([a, b], **ERA_KW)

    ref_idx = a.df.index            # identical ranges -> ref index = stream a
    n = len(ref_idx)
    assert n == 6000
    holdout_bars = int(round(n * HOLDOUT_FRACTION))
    active = n - holdout_bars       # 4800: first reserved reference bar
    assert meta["embargo_bars"] == EMBARGO_NEST_BARS == 200
    assert meta["holdout_start"] == ref_idx[active]
    assert meta["embargo_ts"] == ref_idx[active + EMBARGO_NEST_BARS]
    assert meta["n_slices"] == len(slices) == 2

    for sd, sl in zip((a, b), slices):
        assert sl.start_ts >= meta["embargo_ts"]
        expect = int((sd.df.index >= meta["embargo_ts"]).sum())
        assert len(sl.df) == expect >= MIN_STREAM_BARS
        # the slice's data really starts at the embargoed reference bar
        assert sl.df["close"].iloc[0] == \
            sd.df["close"].iloc[active + EMBARGO_NEST_BARS]
        # per-slice standalone labels exist (consistency rule, spec §A1)
        assert "pivot_span_high" in sl.df.columns
        assert "pivot_span_low" in sl.df.columns

    # No fold slice reaches the holdout region: locate every fold OOS slice's
    # last bar in the original stream by its (unique random-walk) close value
    # and require >= the full embargo between it and the first holdout bar.
    assert len(folds) == 2          # pins the fold-loop arithmetic for n=6000
    by_ticker = {"SPX": a, "DAX": b}
    first_holdout_off = active + EMBARGO_NEST_BARS
    for fold in folds:
        for sl in fold:
            close = by_ticker[sl.stream.ticker].df["close"].to_numpy()
            pos = int(np.flatnonzero(close == sl.df_oos["close"].iloc[-1])[0])
            assert pos < active, "fold OOS bar inside the reserved holdout"
            assert first_holdout_off - pos >= EMBARGO_NEST_BARS, (
                f"gap {first_holdout_off - pos} < embargo {EMBARGO_NEST_BARS}")


def test_build_holdout_slices_empty_pool_and_too_short_streams(tmp_path):
    slices, meta = build_holdout_slices([])
    assert slices == [] and meta["n_slices"] == 0
    # a pool whose holdout tail is shorter than MIN_STREAM_BARS drops streams
    tiny = _stream_data(tmp_path, "SPX", 1500, 1262304000, 86400, "US_EQ")
    slices, meta = build_holdout_slices([tiny])
    # 1500 bars -> holdout 300, active 1200, embargo end 1400 -> 100-bar tail
    assert slices == []
    assert meta["dropped_streams"] == ["SPX_1D"]


# ---------------------------------------------------------------------------
# A5.2 — era_pass truth table (pre-committed constants)
# ---------------------------------------------------------------------------


def test_era_pass_min_ratio_constant():
    assert ERA_PASS_MIN_RATIO == 0.5


def test_era_pass_truth_table():
    fold_scores = [0.4, 0.6]                 # mean 0.5 -> threshold 0.25
    assert holdout_era_pass(0.0, fold_scores) is False        # score 0
    assert holdout_era_pass(-0.1, fold_scores) is False
    assert holdout_era_pass(0.2499999, fold_scores) is False  # just below
    assert holdout_era_pass(0.25, fold_scores) is True        # at the ratio
    assert holdout_era_pass(0.2500001, fold_scores) is True   # just above
    # degenerate basis: no informative folds -> mean 0 -> any positive passes
    assert holdout_era_pass(0.01, []) is True
    assert holdout_era_pass(0.0, []) is False


# ---------------------------------------------------------------------------
# A5.3 — pooled_holdout_score == the OOS leg of pooled_fold_score
# ---------------------------------------------------------------------------


def _stats():
    oos = [
        StreamStat(n_signals=5, tp=3, matched_pivots=3, total_pivots=4,
                   n_bars=900, weight=1.0, tp_mass=2.4, total_mass=3.2,
                   n_unmatched=2),
        StreamStat(n_signals=2, tp=1, matched_pivots=1, total_pivots=3,
                   n_bars=900, weight=0.5, tp_mass=0.6, total_mass=2.0,
                   n_unmatched=1),
    ]
    is_ = [
        StreamStat(n_signals=7, tp=4, matched_pivots=4, total_pivots=6,
                   n_bars=600, weight=1.0, tp_mass=3.0, total_mass=4.4,
                   n_unmatched=3),
    ]
    return is_, oos


@pytest.mark.parametrize("side", ["high", "low"])
def test_pooled_holdout_score_equals_oos_leg(side):
    is_stats, oos_stats = _stats()
    _, comp = pooled_fold_score(is_stats, oos_stats, side)
    score, hcomp = pooled_holdout_score(oos_stats, side)
    assert score == comp["oos_score"]
    # spec §A2: required component fields
    assert {"precision_w", "recall_w", "n_signals",
            "tp_mass", "total_mass"} <= set(hcomp)
    assert hcomp["precision_w"] == comp["precision_oos"]
    assert hcomp["recall_w"] == comp["recall_oos"]
    assert hcomp["tp_mass"] == comp["tp_mass_oos"]
    assert hcomp["total_mass"] == comp["total_mass_oos"]
    # pin by construction: an empty IS leg makes the fold score the OOS leg
    fold0, _ = pooled_fold_score([], oos_stats, side)
    assert fold0 == score


def test_pooled_holdout_score_empty_stats_is_zero():
    score, comp = pooled_holdout_score([], "high")
    assert score == 0.0
    assert comp["n_signals"] == 0.0 and comp["total_mass"] == 0.0


# ---------------------------------------------------------------------------
# A5.4 — golden additive-only regression
# ---------------------------------------------------------------------------


def test_holdout_is_additive_fold_scores_and_signals_unchanged(tmp_path):
    """Running the FULL holdout path leaves fold scores + signal arrays
    byte-identical (holdout is ADDITIVE reporting, spec GOLDEN clause)."""
    a, b = _pool(tmp_path)
    streams = [a.stream, b.stream]
    folds = build_calendar_folds([a, b], **ERA_KW)
    base = Params()

    def _fold_scores(fs):
        return {side: [float(x) for x in raw_fold_scores(
            PooledScorer(folds=fs, streams=streams, side=side), base)]
            for side in ("high", "low")}

    sl0 = folds[0][0]
    det = SpeculatorDetector(sl0.df_oos, base, sl0.artifacts_oos).run()
    sig_before = {k: det[k].to_numpy().copy()
                  for k in ("signal_high", "signal_low")}
    before = _fold_scores(folds)

    # the full holdout leg: slices + exact detector + pooled score + rule
    slices, meta = build_holdout_slices([a, b], **ERA_KW)
    assert slices, "holdout slices must form on this pool"
    weights = cluster_weights(streams)
    for side in ("high", "low"):
        score, comp, per_stream = evaluate_holdout(base, side, slices, weights)
        assert {"precision_w", "recall_w", "n_signals",
                "tp_mass", "total_mass"} <= set(comp)
        assert len(per_stream) == len(slices)
        holdout_era_pass(score, before[side])

    # 1) same fold objects: scores bit-identical
    assert _fold_scores(folds) == before
    # 2) signal arrays bit-identical
    det2 = SpeculatorDetector(sl0.df_oos, base, sl0.artifacts_oos).run()
    for k, arr in sig_before.items():
        assert np.array_equal(det2[k].to_numpy(), arr), f"{k} drifted"
    # 3) stream frames untouched: rebuilding the folds reproduces the scores
    folds2 = build_calendar_folds([a, b], **ERA_KW)
    assert _fold_scores(folds2) == before


def test_golden_disk_snapshot_reproduced_after_holdout_path():
    """The on-disk golden fold scores, LCBs and signal arrays reproduce
    bit-for-bit AFTER the holdout path runs on the same pool (A5.4)."""
    cap = _REPO / "temp" / "capture_baseline.py"
    payload_path = _GOLDEN / "golden_baseline.json"
    if not payload_path.exists():
        pytest.skip("golden snapshot not captured yet")
    spec = importlib.util.spec_from_file_location("capture_baseline_holdout", cap)
    assert spec is not None and spec.loader is not None
    cb = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cb)

    asset = "SPX"
    csv = _REPO / cb.ASSETS[asset]["path"]
    if not csv.exists():
        pytest.skip(f"missing data CSV for {asset}")
    df = load_stream_frame(str(csv)).iloc[-cb.N_BARS:].copy()
    stream = Stream(ticker=asset, timeframe="1D", path=str(csv),
                    cluster_id=cb.ASSETS[asset]["cluster_id"])
    sd = StreamData(stream=stream, df=df, bar_seconds=86400.0)
    folds = build_calendar_folds(
        [sd], oos_fraction=cb.OOS_FRACTION_GOLDEN)[:cb.N_FOLDS]
    base = Params()

    # holdout leg FIRST (the potential contaminator), golden re-check AFTER
    slices, _ = build_holdout_slices([sd], oos_fraction=cb.OOS_FRACTION_GOLDEN)
    weights = cluster_weights([stream])
    for side in ("high", "low"):
        evaluate_holdout(base, side, slices, weights)

    golden = json.loads(payload_path.read_text())["assets"][asset]
    for side in ("high", "low"):
        scorer = PooledScorer(folds=folds, streams=[stream], side=side)
        fresh = [float(s) for s in raw_fold_scores(scorer, base)]
        assert fresh == golden["scores"][side]["fold_scores"], \
            f"{asset}/{side} fold scores drifted with holdout in the loop"
        assert float(scorer.score(base)) == golden["scores"][side]["lcb"]
    arrays = {}
    for tag, sdf, art in cb.representative_slices(folds):
        res = SpeculatorDetector(sdf, base, art).run()
        arrays[f"{tag}_signal_high"] = res["signal_high"].to_numpy()
        arrays[f"{tag}_signal_low"] = res["signal_low"].to_numpy()
    with np.load(_GOLDEN / f"golden_{asset}.npz") as disk:
        for key, arr in arrays.items():
            assert np.array_equal(disk[key], arr), f"golden {key} drifted"


# ---------------------------------------------------------------------------
# §A4 wiring — run_v17_gpu holdout block + era_pass override semantics
# ---------------------------------------------------------------------------


def test_effective_era_pass_override_semantics():
    from src.v17_runner import _effective_era_pass
    assert _effective_era_pass(None, True) is True
    assert _effective_era_pass(None, False) is False
    assert _effective_era_pass(True, False) is True    # explicit caller wins
    assert _effective_era_pass(False, True) is False   # explicit caller wins
    assert _effective_era_pass(None, None) is None


_SPX = _REPO / "data" / "raw" / "SPX_1D_18710201_20260318.csv"
_DAX = _REPO / "data" / "raw" / "DAX_1D_19700102_20260324.csv"


def _two_asset_dir(tmp_path: Path) -> str:
    """SPX + DAX 6000-bar tails (pattern of test_v17_gpu_integration)."""
    if not _SPX.exists() or not _DAX.exists():
        pytest.skip(f"missing {_SPX} or {_DAX}")
    for csv, name in ((_SPX, "SPX_1D_00000000_00000000.csv"),
                      (_DAX, "DAX_1D_00000000_00000000.csv")):
        lines = csv.read_text().splitlines()
        keep = [lines[0]] + lines[1:][-6000:]
        (tmp_path / name).write_text("\n".join(keep) + "\n")
    return str(tmp_path)


def test_run_v17_gpu_emits_holdout_block_and_sets_era_pass(tmp_path):
    pytest.importorskip("torch")
    from src.v17_runner import run_v17_gpu

    out = run_v17_gpu(
        groups=["INDICES"], timeframes=["1D"],
        data_dir=_two_asset_dir(tmp_path),
        sides=("low",), era_kw=ERA_KW,
        search_kw={"popsize": 4, "sobol_n": 4, "generations": 1,
                   "top_k": 3, "rng_seed": 11},
        run_slug="t_holdout_e2e", device="cpu",
        results_dir=str(tmp_path / "results"),
    )

    # v5.1 era marker: the objective bump is labelled so v5 and v5.1 LCBs are
    # never silently compared, and the pricing that defines the era is recorded.
    assert out["scorer"] == "v5.1"
    assert out["firing_cap"] == 1.0
    assert out["pricing"] == {"w_fp": 0.5, "firing_penalty": out["firing_penalty"],
                              "firing_cap": 1.0}

    side = out["sides"]["low"]
    assert side["trace"]["scorer_version"] == "v5.1"
    hd = side["holdout"]
    assert hd is not None
    # spec §A4 field contract
    assert {"score", "components", "per_stream", "era_pass", "n_slices",
            "holdout_start", "embargo_bars", "min_ratio"} <= set(hd)
    assert hd["embargo_bars"] == EMBARGO_NEST_BARS
    assert hd["min_ratio"] == ERA_PASS_MIN_RATIO
    assert hd["n_slices"] == 2
    assert isinstance(hd["era_pass"], bool)
    assert {"precision_w", "recall_w", "n_signals",
            "tp_mass", "total_mass"} <= set(hd["components"])
    assert len(hd["per_stream"]) >= 1
    # era_pass override semantics: None default -> holdout verdict feeds gates
    assert side["acceptance"]["era_pass"] == hd["era_pass"]
    # §B1: shape_variants=None keeps the single-run output shape unchanged
    assert "variants" not in out and "winner_variant" not in out
    # the run JSON round-trips with the holdout block inside
    written = json.loads(Path(out["_written"]).read_text())
    assert written["sides"]["low"]["holdout"]["era_pass"] == hd["era_pass"]
