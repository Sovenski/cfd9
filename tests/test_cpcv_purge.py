"""Spec §4 (PHASE 2) — purged CPCV + selection-bias guards (default-OFF).

Assertions (plan/gpu-refactor-build-spec.md §4):

- Purge correctness: no IS (train) bar's centered structural label window
  (half-width ``max(STRUCTURAL_NEST)=200``) overlaps any test group, and the
  200-bar embargo after each test group holds.
- LOW-side reconstructed OOS paths: the CPCV folds (OOS group segments across
  all reconstructed paths) number at least the current walk-forward fold count.
- HIGH: PBO / selection percentile are returned as ADVISORY fields only; the
  primary HIGH summary is the OOS event count + a wide Wilson interval.
- Deflation (``overfit_guard.deflated_best``) scales with the trial count
  (lambda x generations + sobol + seed) and is wired into the cma-route report.
- Dead-margin masking in ``_stream_stat`` is opt-in (default OFF).
- CRITICAL REGRESSION: with CPCV OFF (default), ``PooledScorer`` / ``run_v17``
  results are UNCHANGED vs the Phase-0 golden snapshot.

Never weaken these assertions: a mismatch is a finding, not a test bug.
"""
from __future__ import annotations

import dataclasses
import importlib.util
import inspect
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.indicators import Params
from src.scoring import REFERENCE_N, add_pivot_labels
from src.universe import Stream
from src.v17_acceptance import raw_fold_scores
from src.v17_optimize import PooledScorer

_REPO = Path(__file__).resolve().parents[1]
_CSV = _REPO / "data" / "raw" / "DAX_1D_19700102_20260324.csv"
_CAPTURE = _REPO / "temp" / "capture_baseline.py"

N_BARS_SYNTH = 6000
PURGE = 200
EMBARGO = 200


# ---------------------------------------------------------------------------
# CPCV index math (synthetic — no market data needed)
# ---------------------------------------------------------------------------
def _splits(n_bars: int = N_BARS_SYNTH, n_groups: int = 6, k_test: int = 2):
    from src.cpcv import CPCVConfig, build_cpcv_splits
    return build_cpcv_splits(n_bars, CPCVConfig(n_groups=n_groups, k_test=k_test))


def test_group_bounds_partition_timeline():
    from src.cpcv import group_bounds
    bounds = group_bounds(N_BARS_SYNTH, 6)
    assert bounds[0][0] == 0 and bounds[-1][1] == N_BARS_SYNTH
    for (a, b), (c, _) in zip(bounds, bounds[1:]):
        assert b == c, "groups must be contiguous"
    sizes = [b - a for a, b in bounds]
    assert max(sizes) - min(sizes) <= 1, "groups must be near-equal"


def test_split_count_is_n_choose_k():
    splits = _splits()
    assert len(splits) == math.comb(6, 2) == 15
    assert [sp.split_id for sp in splits] == list(range(15))
    for sp in splits:
        assert len(sp.test_groups) == len(sp.test_ranges) == 2
        assert sp.train_ranges, "every split must keep some train data"


def test_purge_no_label_window_overlap():
    """No train bar's label window [p-200, p+200] may touch a test group."""
    from src.cpcv import validate_purge
    splits = _splits()
    for sp in splits:
        for a, b in sp.train_ranges:
            assert a < b
            for s, e in sp.test_ranges:
                # the closest train bars are a and b-1; both label windows
                # must clear the test range entirely
                assert (b - 1 + PURGE < s) or (a - PURGE >= e), (
                    f"split {sp.split_id}: train [{a},{b}) label window "
                    f"overlaps test [{s},{e})")
    validate_purge(splits, purge_bars=PURGE)  # must not raise


def test_embargo_after_test_groups():
    """Train data resuming AFTER a test group must skip purge+embargo bars."""
    for sp in _splits():
        for s, e in sp.test_ranges:
            for a, b in sp.train_ranges:
                if a >= e:  # train range that starts after this test group
                    assert a >= e + PURGE + EMBARGO, (
                        f"split {sp.split_id}: train starts {a} < "
                        f"{e + PURGE + EMBARGO} (test end {e} + purge + embargo)")


def test_validate_purge_detects_violation():
    from src.cpcv import CPCVSplit, validate_purge
    bad = CPCVSplit(split_id=0, test_groups=(0,),
                    test_ranges=((1000, 2000),),
                    train_ranges=((0, 950),))  # 949+200 >= 1000 -> overlap
    with pytest.raises(ValueError):
        validate_purge([bad], purge_bars=PURGE)


def test_path_reconstruction_covers_every_group_once():
    from src.cpcv import n_paths, reconstruct_paths
    splits = _splits()
    paths = reconstruct_paths(splits)
    assert len(paths) == n_paths(6, 2) == math.comb(5, 1) == 5
    seen: set[tuple[int, int]] = set()
    for path in paths:
        assert sorted(g for _, g in path) == list(range(6)), \
            "each path must cover every group exactly once"
        for cell in path:
            assert cell not in seen, "a (split, group) cell was reused"
            seen.add(cell)
    # every test-group occurrence across all splits is consumed exactly once
    assert len(seen) == 15 * 2


# ---------------------------------------------------------------------------
# CPCV fold materialization on real data (opt-in loop, default OFF)
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def dax_stream():
    if not _CSV.exists():
        pytest.skip(f"missing {_CSV}")
    from src.pooled_validation import StreamData, load_stream_frame
    df = load_stream_frame(str(_CSV)).iloc[-6000:].copy()
    add_pivot_labels(df)
    s = Stream(ticker="DAX", timeframe="1D", path=str(_CSV), cluster_id="EU_EQ")
    return StreamData(stream=s, df=df, bar_seconds=86400.0)


@pytest.fixture(scope="module")
def calendar_folds(dax_stream):
    from src.pooled_validation import build_calendar_folds
    return build_calendar_folds([dax_stream])


@pytest.fixture(scope="module")
def cpcv_folds(dax_stream):
    from src.pooled_validation import build_cpcv_folds
    return build_cpcv_folds([dax_stream])


def test_cpcv_low_side_path_count_vs_current_folds(cpcv_folds, calendar_folds):
    """LOW-side reconstructed OOS coverage >= current walk-forward fold count.

    CPCV with N=6, k=2 reconstructs ``k*C(N,k)/N = 5`` full OOS paths of 6
    group segments each — 30 OOS evaluations versus the single walk-forward
    path's fold count.
    """
    folds, metas = cpcv_folds
    assert len(folds) == len(metas) == math.comb(6, 2) * 2 == 30
    assert len(calendar_folds) >= 1
    assert len(folds) >= len(calendar_folds), \
        "CPCV must give the LOW side at least as many OOS segments as folds"
    # the 30 segments reconstruct exactly 5 paths, each covering all 6 groups
    path_ids = sorted({m.path_id for m in metas})
    assert path_ids == list(range(5))
    for pid in path_ids:
        groups = sorted(m.test_group for m in metas if m.path_id == pid)
        assert groups == list(range(6))


def test_cpcv_slices_sized_and_purged(cpcv_folds):
    from src.pooled_validation import MIN_STREAM_BARS
    folds, metas = cpcv_folds
    col = f"pivot_N{REFERENCE_N}"
    for fold, meta in zip(folds, metas):
        ia, ib = meta.is_range
        s, e = meta.test_range
        # the materialized IS range obeys the purge gap to its own test group
        assert (ib - 1 + PURGE < s) or (ia - PURGE >= e), \
            f"fold(split={meta.split_id},g={meta.test_group}) IS overlaps test"
        for sl in fold:
            assert len(sl.df_is) >= MIN_STREAM_BARS
            assert len(sl.df_oos) >= MIN_STREAM_BARS
            assert col in sl.df_is.columns and col in sl.df_oos.columns
            assert sl.artifacts_is is not None and sl.artifacts_oos is not None


# ---------------------------------------------------------------------------
# Dead-margin masking in _stream_stat (opt-in, default OFF)
# ---------------------------------------------------------------------------
def _toy_df(n: int = 1000, pivots: tuple[int, ...] = (500,)) -> pd.DataFrame:
    df = pd.DataFrame({
        "open": np.ones(n), "high": np.ones(n), "low": np.ones(n),
        "close": np.ones(n), "volume": np.ones(n),
    })
    lbl = np.zeros(n, dtype=np.int8)
    lbl[list(pivots)] = 1
    df[f"pivot_N{REFERENCE_N}"] = lbl
    return df


def test_stream_stat_dead_margin_mask_opt_in():
    from src.pooled_validation import EMBARGO_NEST_BARS, _stream_stat
    assert EMBARGO_NEST_BARS == PURGE == 200
    df = _toy_df(1000, pivots=(500,))
    sig = pd.Series(False, index=df.index)
    sig.iloc[[10, 500, 995]] = True  # dead-margin, live, dead-margin

    legacy = _stream_stat(df, sig, "high", 1.0)
    assert legacy.n_signals == 3 and legacy.n_bars == 1000
    assert legacy.tp == 1 and legacy.total_pivots == 1

    masked = _stream_stat(df, sig, "high", 1.0, mask_dead_margins=True)
    assert masked.n_bars == 1000 - 2 * 200, "dead margins must leave n_bars"
    assert masked.n_signals == 1, "dead-margin signals must not be scored"
    assert masked.tp == 1 and masked.total_pivots == 1
    assert masked.weight == legacy.weight == 1.0


def test_stream_stat_fully_dead_slice_masks_to_zero():
    from src.pooled_validation import _stream_stat
    df = _toy_df(400, pivots=())
    sig = pd.Series(False, index=df.index)
    sig.iloc[200] = True
    st = _stream_stat(df, sig, "high", 0.5, mask_dead_margins=True)
    assert (st.n_signals, st.tp, st.matched_pivots, st.total_pivots,
            st.n_bars) == (0, 0, 0, 0, 0)
    assert st.weight == 0.5


def test_masking_and_cpcv_are_default_off():
    """The §4 fixes must be opt-in: every new knob defaults to OFF."""
    from src.pooled_validation import _stream_stat, evaluate_pooled_fold
    assert inspect.signature(_stream_stat).parameters[
        "mask_dead_margins"].default is False
    assert inspect.signature(evaluate_pooled_fold).parameters[
        "mask_dead_margins"].default is False


# ---------------------------------------------------------------------------
# overfit_guard: Wilson interval, PBO via CSCV, deflated-best haircut
# ---------------------------------------------------------------------------
def test_wilson_interval_wide_for_thin_counts():
    from src.overfit_guard import wilson_interval
    assert wilson_interval(0, 0) == (0.0, 1.0)  # no information -> max width
    lo, hi = wilson_interval(3, 27)
    assert 0.0 <= lo < 3 / 27 < hi <= 1.0
    assert hi - lo > 0.15, "thin-count interval must be WIDE"
    lo2, hi2 = wilson_interval(30, 270)
    assert hi2 - lo2 < hi - lo, "more evidence must narrow the interval"


def test_pbo_cscv_dominant_vs_noise():
    from src.overfit_guard import pbo_cscv
    rng = np.random.default_rng(42)
    noise = rng.normal(size=(32, 20))
    dominant = noise.copy()
    dominant[:, 7] += 5.0  # genuinely best in every fold
    res = pbo_cscv(dominant, n_partitions=8)
    assert res["pbo"] == 0.0
    assert res["n_combinations"] == math.comb(8, 4) == 70
    assert res["advisory"] is True
    res_noise = pbo_cscv(noise, n_partitions=8)
    assert 0.05 <= res_noise["pbo"] <= 0.95, \
        "pure-luck selection must show substantial PBO"
    assert res_noise["pbo"] > res["pbo"]
    assert all(np.isfinite(res_noise["logits"]))


def test_deflated_best_scales_with_trial_count():
    from src.overfit_guard import deflated_best
    rng = np.random.default_rng(0)
    scores = list(rng.normal(0.1, 0.02, size=64))
    small = deflated_best(0.2, scores, n_trials=10)
    big = deflated_best(0.2, scores, n_trials=4096)
    assert big["haircut"] > small["haircut"] > 0.0, \
        "haircut must grow with lambda x generations"
    for d in (small, big):
        assert d["deflated"] <= d["raw"] and d["deflated"] >= 0.0
        assert d["deflated"] == max(0.0, d["raw"] - d["haircut"])
    assert deflated_best(0.2, scores, n_trials=1)["haircut"] == 0.0
    assert deflated_best(0.2, [0.1] * 5)["haircut"] == 0.0  # zero dispersion


# ---------------------------------------------------------------------------
# Per-asset-then-aggregate HIGH diagnostic (advisory PBO/percentile)
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def wide_oos_folds(dax_stream):
    from src.pooled_validation import build_calendar_folds
    return build_calendar_folds([dax_stream], oos_fraction=0.15,
                                step_fraction=0.25)


def test_high_diagnostic_primary_is_events_plus_wilson(wide_oos_folds):
    from src.pooled_validation import per_asset_high_diagnostic
    diag = per_asset_high_diagnostic(wide_oos_folds, Params(), side="high")
    assert diag["side"] == "high"
    assert diag["primary"] == "event_count+wilson_interval"
    # PBO / percentile exist ONLY as advisory fields (never a gate)
    assert set(diag["advisory"]) == {"pbo", "selection_percentile"}
    assert diag["advisory"]["pbo"] is None
    assert diag["advisory"]["selection_percentile"] is None

    assert "DAX_1D" in diag["per_asset"]
    for entry in diag["per_asset"].values():
        for key in ("oos_events", "n_signals", "tp", "precision",
                    "wilson_low", "wilson_high"):
            assert key in entry
        assert 0.0 <= entry["wilson_low"] <= entry["wilson_high"] <= 1.0

    agg = diag["aggregate"]
    assert agg["oos_events"] == sum(
        e["oos_events"] for e in diag["per_asset"].values())
    assert agg["n_signals"] == sum(
        e["n_signals"] for e in diag["per_asset"].values())
    assert 0.0 <= agg["wilson_low"] <= agg["wilson_high"] <= 1.0
    assert "median_asset_precision" in agg
    assert isinstance(agg["heterogeneity_flag"], bool)


# ---------------------------------------------------------------------------
# Runner wiring: deflation + HIGH diagnostic on the cma route ONLY
# ---------------------------------------------------------------------------
def _mini_data_dir(tmp_path: Path) -> str:
    lines = _CSV.read_text().splitlines()
    keep = [lines[0]] + lines[1:][-6000:]
    (tmp_path / "DAX_1D_00000000_00000000.csv").write_text("\n".join(keep) + "\n")
    return str(tmp_path)


def test_run_v17_default_route_has_no_new_keys(tmp_path):
    """Default (ascent) run_v17 output keys are EXACTLY the legacy set."""
    if not _CSV.exists():
        pytest.skip(f"missing {_CSV}")
    from src.v17_runner import run_v17
    out = run_v17(groups=["INDICES"], timeframes=["1D"],
                  data_dir=_mini_data_dir(tmp_path), sides=("low",),
                  era_kw={"step_fraction": 0.25}, grid_n=2, max_sweeps=1,
                  run_slug="t_cpcv_reg", progress=False)
    assert set(out.keys()) == {
        "run_slug", "groups", "timeframes", "volume_policy", "n_folds",
        "streams", "grid_n", "max_sweeps", "sides",
    }
    assert set(out["sides"]["low"].keys()) == {
        "seed_lcb", "final_lcb", "n_evals", "changed", "best_params",
        "coords", "trace",
    }, "no §4 key may leak into the default route"


def test_run_v17_cma_route_deflation_and_high_diagnostic(tmp_path):
    if not _CSV.exists():
        pytest.skip(f"missing {_CSV}")
    from src.v17_runner import run_v17
    out = run_v17(groups=["INDICES"], timeframes=["1D"],
                  data_dir=_mini_data_dir(tmp_path), sides=("high",),
                  era_kw={"step_fraction": 0.25, "oos_fraction": 0.15},
                  run_slug="t_cpcv_cma", progress=False, search="cma",
                  search_kw={"popsize": 4, "sobol_n": 4, "generations": 1,
                             "top_k": 3, "rng_seed": 11})
    side = out["sides"]["high"]
    assert side["search"] == "cma"
    # deflation wired into the reported number, scaled to the trial count
    defl = side["deflation"]
    assert defl["n_trials"] == side["n_evals"]  # lambda*gens + sobol + seed
    assert side["final_lcb_deflated"] == defl["deflated"]
    assert side["final_lcb_deflated"] <= side["final_lcb"] + 1e-12
    assert defl["haircut"] >= 0.0
    # raw final_lcb stays the REAL detector LCB (existing cma contract)
    assert abs(side["final_lcb"] - side["top_k"][0]["score"]) < 1e-9
    # per-asset HIGH diagnostic emitted next to the pooled LCB
    diag = side["per_asset_diagnostic"]
    assert diag["primary"] == "event_count+wilson_interval"
    assert diag["advisory"]["pbo"] is None  # advisory, not computed as a gate
    pct = diag["advisory"]["selection_percentile"]
    assert pct is not None and 0.0 <= pct <= 1.0
    assert "oos_events" in diag["aggregate"]


# ---------------------------------------------------------------------------
# CRITICAL REGRESSION — CPCV OFF (default): unchanged vs Phase-0 golden
# ---------------------------------------------------------------------------
def _load_capture_module():
    if "capture_baseline" in sys.modules:
        return sys.modules["capture_baseline"]
    if not _CAPTURE.exists():
        pytest.fail(f"missing capture script {_CAPTURE}")
    spec = importlib.util.spec_from_file_location("capture_baseline", _CAPTURE)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["capture_baseline"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize("asset", ("SPX", "DAX"))
def test_regression_golden_pooledscorer_unchanged(asset):
    """With CPCV OFF (default), PooledScorer reproduces the Phase-0 golden
    fold scores and pooled LCB EXACTLY (bit-for-bit float equality)."""
    cb = _load_capture_module()
    if not (_REPO / cb.ASSETS[asset]["path"]).exists():
        pytest.skip(f"missing data CSV for {asset}")
    golden_json = cb.GOLDEN_DIR / "golden_baseline.json"
    if not golden_json.exists():
        cb.main()
    payload = json.loads(golden_json.read_text())

    folds, streams = cb.build_folds(asset)
    base = Params()
    for side in ("high", "low"):
        gold = payload["assets"][asset]["scores"][side]
        scorer = PooledScorer(folds=folds, streams=streams, side=side)
        assert [float(s) for s in raw_fold_scores(scorer, base)] == \
            gold["fold_scores"], f"{asset}/{side} fold scores drifted vs golden"
        assert float(scorer.score(base)) == gold["lcb"], \
            f"{asset}/{side} pooled LCB drifted vs golden"
