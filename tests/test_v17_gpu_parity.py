"""THE load-bearing gate for the batched GPU evaluator (build-spec §5.4, §0.7).

PIR-spike verdict (§2) was ``trust-kernel``, so every signal assertion here is
strict ``np.array_equal`` against the real ``SpeculatorDetector`` — never a
tolerance, never a flip-rate budget. Any mismatch is a HARD FAIL.

§0.7 GREEN criteria asserted:

- For the base gold ``Params()`` + a permissive draw + 5 random in-bounds
  threshold draws, on EVERY test slice (SPX IS, SPX OOS, DAX IS, DAX OOS, one
  short stream): ``np.array_equal(gpu_sig, SpeculatorDetector(...).run())``
  for BOTH sides.
- ``abs(gpu_LCB - PooledScorer_LCB) < 1e-9`` for every candidate in the batch.

Edge cases (build-spec §5.4) / invariants (§0.6):

- P4 — warm-up NaN -> False on the first ~300 bars (short stream).
- P5 — read-BEFORE-append, HIGH-before-LOW same-bar push, ring truncation at
  ``Kmax`` (synthetic double-pivot stream + ring simulation, with teeth).
- P6 — drift edge votes as fixed-lag arrays (edge-voting architecture draw).
- P7 — segmented scan: per-asset carry reset in ONE packed batch (no
  cooldown/pivot leak across instruments), pad bars frozen and signal-free.
"""
from __future__ import annotations

import collections
import dataclasses
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.detector import SpeculatorDetector, build_detector_artifacts
from src.indicators import Params, calc_pivot_drift, pivot_high, pivot_low
from src.pooled_validation import StreamData, build_calendar_folds, load_stream_frame
from src.scoring import add_pivot_labels
from src.search_space import space_for
from src.universe import Stream
from src.v17_gpu.drift_precompute import (
    DriftSpec,
    confirmed_pivot_events,
    drift_per_bar,
    precompute_drift,
)
from src.v17_acceptance import raw_fold_scores
from src.v17_optimize import PooledScorer, active_threshold_fields

torch = pytest.importorskip("torch")
evalmod = pytest.importorskip("src.v17_gpu.eval_torch")
scanmod = pytest.importorskip("src.v17_gpu.phase2_scan")  # RED until §5.4 exists
TorchPhase1 = evalmod.TorchPhase1
GpuPooledScorer = scanmod.GpuPooledScorer
batched_signals = scanmod.batched_signals
signals_torch = scanmod.signals_torch

_SPX = Path("data/raw/SPX_1D_20170428_20260318.csv")
_DAX = Path("data/raw/DAX_1D_19700102_20260324.csv")

#: Architecture with EVERY vote switch on + edge voting (P6 teeth on drift).
ALLVOTES_EDGE = dataclasses.replace(
    Params(),
    use_trend_high=True, use_volume_high=True, use_momentum_high=True,
    use_volatility_high=True, use_gjr_asym_high=True, use_har_vol_high=True,
    use_trend_low=True, use_volume_low=True, use_momentum_low=True,
    use_gjr_asym_low=True, use_har_vol_low=True,
    use_edge_voting_high=True, edge_window_high=5,
    use_edge_voting_low=True, edge_window_low=7,
)


def _in_bounds_draws(base: Params, n: int, seed: int) -> list[Params]:
    """Random draws strictly inside the search-space float bounds (§0.7)."""
    rng = np.random.default_rng(seed)
    out: list[Params] = []
    for _ in range(n):
        over: dict[str, float] = {}
        for side in ("high", "low"):
            bounds = space_for(side).float_bounds
            for f in active_threshold_fields(base, side):
                stem = f[: f.rfind("_")]
                lo, hi = bounds[stem]
                over[f] = float(rng.uniform(lo, hi))
        out.append(dataclasses.replace(base, **over))
    return out


def _permissive(base: Params) -> Params:
    """A loose draw at the firing-friendly bound ends, so the suite has teeth
    (signals actually fire and the stateful loop is exercised)."""
    return dataclasses.replace(
        base,
        min_agreement_high=0.10, min_agreement_low=0.10,
        dur_extreme_pct_high=0.30, dur_extreme_pct_low=0.50,
        scale_div_thresh_high=0.60, scale_div_thresh_low=0.60,
        pct_extreme_high=0.55, pct_extreme_low=0.70,
        momentum_velocity_thresh_high=0.0, momentum_velocity_thresh_low=0.0,
        vola_high_pct_low=0.50,
        pivot_drift_thresh_high=0.001, pivot_drift_thresh_low=0.001,
        pivot_drift_gate_mult_high=10.0, pivot_drift_gate_mult_low=10.0,
    )


def _synthetic_df(n: int = 700, seed: int = 5) -> pd.DataFrame:
    """Random walk + wide-range spike bars that pivot HIGH and LOW on the
    SAME bar (P5 same-bar push-order stress)."""
    rng = np.random.default_rng(seed)
    close = 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.01, n)))
    high = close * (1.0 + np.abs(rng.normal(0.0, 0.004, n)))
    low = close * (1.0 - np.abs(rng.normal(0.0, 0.004, n)))
    for i in range(80, n - 80, 97):  # double-sided spike bars
        high[i] = close[i] * 1.08
        low[i] = close[i] * 0.92
    return pd.DataFrame(
        {"open": close, "high": high, "low": low, "close": close,
         "volume": rng.integers(1000, 5000, n).astype(float)},
        index=pd.date_range("2000-01-03", periods=n, freq="B"),
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def param_set() -> list[Params]:
    """Base gold + permissive + 5 random in-bounds draws (§0.7 sample)."""
    base = Params()
    return [base, _permissive(base)] + _in_bounds_draws(base, 5, seed=7)


@pytest.fixture(scope="module")
def slices() -> dict[str, pd.DataFrame]:
    if not _SPX.exists() or not _DAX.exists():
        pytest.skip(f"missing {_SPX} or {_DAX}")
    spx = load_stream_frame(str(_SPX))
    dax = load_stream_frame(str(_DAX))
    return {
        "spx_is": spx.iloc[:1500].reset_index(drop=True),
        "spx_oos": spx.iloc[1500:1900].reset_index(drop=True),
        "dax_is": dax.iloc[-2000:-800].reset_index(drop=True),
        "dax_oos": dax.iloc[-800:].reset_index(drop=True),
        "short": spx.iloc[1900:2220].reset_index(drop=True),  # short stream
    }


@pytest.fixture(scope="module")
def built(slices):
    """(df, artifacts, TorchPhase1, drift tensor) per slice, base Params."""
    base = Params()
    spec = DriftSpec.from_params(base)
    out = {}
    for name, df in slices.items():
        art = build_detector_artifacts(df)
        tp = TorchPhase1(df, base, art)
        drift = torch.from_numpy(precompute_drift(df, spec))
        out[name] = (df, art, tp, drift)
    return out


@pytest.fixture(scope="module")
def oracle_sigs(built, param_set):
    """SpeculatorDetector reference signals per (slice, param index)."""
    refs = {}
    for name, (df, art, _, _) in built.items():
        for pi, p in enumerate(param_set):
            res = SpeculatorDetector(df, p, art).run()
            refs[(name, pi)] = (np.asarray(res["signal_high"].values, dtype=bool),
                                np.asarray(res["signal_low"].values, dtype=bool))
    return refs


# ---------------------------------------------------------------------------
# §0.7 — signals byte-identical on EVERY slice for EVERY sampled Params
# ---------------------------------------------------------------------------


def test_signals_match_detector_on_all_slices_and_draws(built, param_set, oracle_sigs):
    fired_high = fired_low = 0
    for name, (_, _, tp, drift) in built.items():
        for pi, p in enumerate(param_set):
            got = signals_torch(tp, p, drift)
            ref_h, ref_l = oracle_sigs[(name, pi)]
            assert got["signal_high"].dtype == np.bool_
            assert got["signal_low"].dtype == np.bool_
            if not np.array_equal(got["signal_high"], ref_h):
                raise AssertionError(
                    f"HARD FAIL §0.7: signal_high mismatch on {name}, params #{pi}")
            if not np.array_equal(got["signal_low"], ref_l):
                raise AssertionError(
                    f"HARD FAIL §0.7: signal_low mismatch on {name}, params #{pi}")
            fired_high += int(ref_h.sum())
            fired_low += int(ref_l.sum())
    # Teeth: the sampled set must actually exercise the stateful loop.
    assert fired_low > 0, "no LOW signal fired anywhere — parity test is vacuous"
    assert fired_high + fired_low > 0


def test_signals_match_with_edge_voting_and_all_votes(built):
    """P6: drift edge votes are fixed-lag arrays — exercised with every vote
    switch ON and edge voting enabled on both sides."""
    for name in ("spx_is", "dax_oos"):
        df, art, _, _ = built[name]
        tp = TorchPhase1(df, ALLVOTES_EDGE, art)
        drift = torch.from_numpy(
            precompute_drift(df, DriftSpec.from_params(ALLVOTES_EDGE)))
        draws = [ALLVOTES_EDGE, _permissive(ALLVOTES_EDGE)] \
            + _in_bounds_draws(ALLVOTES_EDGE, 2, seed=13)
        for pi, p in enumerate(draws):
            ref = SpeculatorDetector(df, p, art).run()
            got = signals_torch(tp, p, drift)
            assert np.array_equal(got["signal_high"], ref["signal_high"].values), \
                (name, pi)
            assert np.array_equal(got["signal_low"], ref["signal_low"].values), \
                (name, pi)


# ---------------------------------------------------------------------------
# P4 — warm-up NaN -> False (first ~300 bars, short stream)
# ---------------------------------------------------------------------------


def test_warmup_nan_region_first_300_bars(built, param_set, oracle_sigs):
    _, _, tp, drift = built["short"]
    # Precondition: the warm-up region really is NaN-laden (GJR is seeded at
    # bar 0 — P2 — so the NaN carriers are the rolling-window features).
    assert np.isnan(tp.feat.linreg_norm_high[:50]).any()
    assert np.isnan(tp.feat.linreg_norm_low[:50]).any()
    assert np.isnan(tp.feat.vol_surge_low[:50]).any()
    assert np.isnan(tp.feat.mom_vel_high[:50]).any()
    assert np.isnan(tp.feat.vola_pos_low[:300]).any()
    for pi, p in enumerate(param_set):
        got = signals_torch(tp, p, drift)
        ref_h, ref_l = oracle_sigs[("short", pi)]
        assert np.array_equal(got["signal_high"][:300], ref_h[:300]), pi
        assert np.array_equal(got["signal_low"][:300], ref_l[:300]), pi


# ---------------------------------------------------------------------------
# P7 — ONE packed multi-asset batch: per-asset carry reset, frozen pad bars
# ---------------------------------------------------------------------------


def test_batched_scan_resets_carry_per_asset_no_leak(built, param_set, oracle_sigs):
    names = list(built.keys())
    lanes = [(built[n][2], built[n][3]) for n in names]
    params_list = [param_set[0], param_set[1]]  # base + permissive (fires)
    sig_h, sig_l, valid = batched_signals(lanes, params_list)
    assert sig_h.shape == (2, len(lanes), max(built[n][2].feat.n for n in names))
    sig_h_np, sig_l_np = sig_h.cpu().numpy(), sig_l.cpu().numpy()
    valid_np = valid.cpu().numpy()
    for li, name in enumerate(names):
        n = built[name][2].feat.n
        for ci in range(len(params_list)):
            ref_h, ref_l = oracle_sigs[(name, ci)]
            # The flat packed batch must equal each per-asset standalone run:
            # any cooldown / dur-counter / pivot leak across the asset
            # boundary (or unfrozen pad bars) breaks this equality.
            assert np.array_equal(sig_h_np[ci, li, :n], ref_h), (name, ci)
            assert np.array_equal(sig_l_np[ci, li, :n], ref_l), (name, ci)
            # Pad bars are frozen AND signal-free (P7).
            assert not sig_h_np[ci, li, n:].any(), (name, ci)
            assert not sig_l_np[ci, li, n:].any(), (name, ci)
        assert valid_np[li, :n].all() and not valid_np[li, n:].any(), name


# ---------------------------------------------------------------------------
# P5 — read-BEFORE-append + HIGH-before-LOW same-bar push (synthetic stress)
# ---------------------------------------------------------------------------


def test_drift_read_before_append_and_high_before_low():
    base = dataclasses.replace(Params(), pivot_drift_lookback_high=2,
                               pivot_drift_lookback_low=2)
    df = _synthetic_df()
    n = len(df)
    spec = DriftSpec.from_params(base)
    ph = pivot_high(df["high"], base.baseline_lb).values
    pl = pivot_low(df["low"], base.baseline_lb).values
    high_arr = df["high"].values.astype(float)
    low_arr = df["low"].values.astype(float)
    ev_bars, ev_vals = confirmed_pivot_events(high_arr, low_arr, ph, pl,
                                              base.baseline_lb)

    # Precondition: a bar exists that confirms BOTH a high and a low pivot.
    dup = ev_bars[:-1][ev_bars[1:] == ev_bars[:-1]]
    assert dup.size > 0, "synthetic stream produced no same-bar double pivot"

    # Teeth (HIGH-before-LOW): swapping the within-bar order must change drift.
    swapped = ev_vals.copy()
    for b in np.unique(dup):
        idx = np.flatnonzero(ev_bars == b)
        swapped[idx] = swapped[idx[::-1]]
    d_ok = drift_per_bar(ev_bars, ev_vals, base.pivot_drift_lookback_high, n)
    d_sw = drift_per_bar(ev_bars, swapped, base.pivot_drift_lookback_high, n)
    assert not np.array_equal(d_ok, d_sw), "push order has no effect — vacuous"

    # Oracle replica of the _detect stack loop: read drift, THEN append
    # (HIGH first). The precomputed drift must match it bar-for-bar.
    confirmed: list[float] = []
    ref = np.zeros((n, 2), dtype=np.float64)
    for t in range(n):
        for j, lb in enumerate((base.pivot_drift_lookback_high,
                                base.pivot_drift_lookback_low)):
            d = calc_pivot_drift(confirmed, lb)
            ref[t, j] = 0.0 if d is None else d
        pb = t - base.baseline_lb
        if pb >= 0:
            if ph[pb]:
                confirmed.append(high_arr[pb])
            if pl[pb]:
                confirmed.append(low_arr[pb])
    drift = precompute_drift(df, spec)
    assert np.array_equal(drift, ref)

    # Teeth (read-BEFORE-append): some confirm bar changes the next bar's read.
    changes = [int(b) for b in np.unique(ev_bars)
               if b + 1 < n and drift[int(b), 0] != drift[int(b) + 1, 0]]
    assert changes, "no confirm bar moved the drift — read-order untestable"

    # End-to-end through the scan on the same stressed stream.
    art = build_detector_artifacts(df)
    tp = TorchPhase1(df, base, art)
    drift_t = torch.from_numpy(drift)
    for p in [base, _permissive(base)] + _in_bounds_draws(base, 2, seed=21):
        ref_sig = SpeculatorDetector(df, p, art).run()
        got = signals_torch(tp, p, drift_t)
        assert np.array_equal(got["signal_high"], ref_sig["signal_high"].values)
        assert np.array_equal(got["signal_low"], ref_sig["signal_low"].values)


def test_ring_truncation_at_kmax_is_lossless(built):
    """P5: a fixed ring of capacity Kmax sees the SAME drift as the growing
    stack, because the read window always lies in the last Kmax entries."""
    base = Params()
    spec = DriftSpec.from_params(base)
    df, _, tp, _ = built["dax_is"]
    n = tp.feat.n
    ph = pivot_high(df["high"], base.baseline_lb).values
    pl = pivot_low(df["low"], base.baseline_lb).values
    ev_bars, ev_vals = confirmed_pivot_events(
        df["high"].values.astype(float), df["low"].values.astype(float),
        ph, pl, base.baseline_lb)
    assert len(ev_vals) > spec.kmax, "slice has too few pivots to truncate"

    for lb in (base.pivot_drift_lookback_high, base.pivot_drift_lookback_low, 20):
        ring: collections.deque = collections.deque(maxlen=spec.kmax)
        ref = np.zeros(n, dtype=np.float64)
        ptr = 0
        for t in range(n):
            d = calc_pivot_drift(list(ring), lb)  # truncated view
            ref[t] = 0.0 if d is None else d
            while ptr < len(ev_bars) and ev_bars[ptr] == t:
                ring.append(ev_vals[ptr])
                ptr += 1
        assert np.array_equal(drift_per_bar(ev_bars, ev_vals, lb, n), ref), lb


# ---------------------------------------------------------------------------
# §0.7 — pooled LCB numerically identical to the real PooledScorer
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def dax_folds():
    if not _DAX.exists():
        pytest.skip(f"missing {_DAX}")
    df = load_stream_frame(str(_DAX)).iloc[-6000:].copy()
    add_pivot_labels(df)
    s = Stream(ticker="DAX", timeframe="1D", path=str(_DAX), cluster_id="EU_EQ")
    sd = StreamData(stream=s, df=df, bar_seconds=86400.0)
    # All 12 calendar folds: 7 are LOW-informative and the permissive draw
    # scores nonzero on them, so the LCB identity check has teeth.
    return build_calendar_folds([sd]), [s]


def test_pooled_lcb_matches_real_pooled_scorer(dax_folds, param_set):
    folds, streams = dax_folds
    base = Params()
    real = PooledScorer(folds=folds, streams=streams, side="low")
    gpu = GpuPooledScorer(folds=folds, streams=streams, side="low",
                          base_params=base)
    lcbs = gpu.score_pop(param_set)
    assert lcbs.shape == (len(param_set),)
    assert lcbs.dtype == np.float64
    for pi, p in enumerate(param_set):
        ref = real.score(p)
        diff = abs(float(lcbs[pi]) - ref)
        if not diff < 1e-9:
            raise AssertionError(
                f"HARD FAIL §0.7: LCB mismatch params #{pi}: "
                f"gpu={float(lcbs[pi])!r} real={ref!r} diff={diff:g}")
    # Single-candidate API agrees with the batched one.
    assert float(gpu.score(param_set[0])) == float(lcbs[0])
    # Teeth: at least one candidate scores nonzero.
    assert np.abs(lcbs).max() > 0.0


def test_pooled_raw_fold_scores_match_real(dax_folds, param_set):
    folds, streams = dax_folds
    real = PooledScorer(folds=folds, streams=streams, side="low")
    gpu = GpuPooledScorer(folds=folds, streams=streams, side="low",
                          base_params=Params())
    p = param_set[1]  # permissive: fires
    ref = raw_fold_scores(real, p)
    got = gpu.fold_scores(p)
    assert len(got) == len(ref)
    assert np.allclose(got, ref, rtol=0.0, atol=1e-12)


# ---------------------------------------------------------------------------
# Guards
# ---------------------------------------------------------------------------


def test_scan_path_rejects_shape_change(built):
    _, _, tp, drift = built["short"]
    bad = dataclasses.replace(Params(), min_duration_low=3)  # scan-state shape
    with pytest.raises(ValueError, match="non-threshold"):
        signals_torch(tp, bad, drift)
