"""Parity spec for the torch Phase-1 features + threshold layer (build-spec §5.3).

The GPU evaluator may only replace ``FastDetector``'s Phase-1 precompute and
the threshold comparisons of ``FastDetector.signals`` if every intermediate
array is byte-identical to the CPU oracle. PIR-spike verdict (§2) was
``trust-kernel`` (pandas-faithful/f64 -> float32 cast is byte-identical), so
every assertion here is strict ``np.array_equal`` — never a tolerance.

Invariants asserted (build-spec §0.6):

- P1 — dtype split: pir_matrix float32; every other threshold-feeding feature
  float64; TF32/low-precision reductions disabled.
- P2 — per-slice artifacts: GJR/HAR/PIR match the per-slice ``FastDetector``
  on every slice independently (incl. a short stream with its own seed bar).
- P3 — strict comparison operators verbatim (boundary draws give the >= vs >
  distinction teeth).
- P4 — NaN -> False in the warm-up region (first ~300 bars).
- P6 — edge votes are fixed-lag arrays (``_edge_or_state`` semantics).
- P8 — linreg via ``np.polyfit`` semantics (default (a): the oracle's own
  ``indicators.linreg_slope_step`` feeds the torch threshold layer).
"""
from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.detector import _edge_or_state, _to_bool, build_detector_artifacts
from src.indicators import Params
from src.pooled_validation import load_stream_frame
from src.v17_fastdetector import FastDetector
from src.v17_optimize import active_threshold_fields

torch = pytest.importorskip("torch")
evalmod = pytest.importorskip("src.v17_gpu.eval_torch")  # RED until module exists
TorchPhase1 = evalmod.TorchPhase1
build_pir_matrix_torch = evalmod.build_pir_matrix_torch
edge_or_state_torch = evalmod.edge_or_state_torch
to_bool_torch = evalmod.to_bool_torch

_SPX = Path("data/raw/SPX_1D_20170428_20260318.csv")
_DAX = Path("data/raw/DAX_1D_19700102_20260324.csv")

#: Architecture with EVERY vote switch on, so every threshold field is active.
ALLVOTES = dataclasses.replace(
    Params(),
    use_trend_high=True, use_volume_high=True, use_momentum_high=True,
    use_volatility_high=True, use_gjr_asym_high=True, use_har_vol_high=True,
    use_trend_low=True, use_volume_low=True, use_momentum_low=True,
    use_gjr_asym_low=True, use_har_vol_low=True,
)

_FEATURE_ATTRS = [
    # (FastDetector attr, Phase1Features attr, expected dtype kind)
    ("_pir_detect_high", "pir_detect_high", "f8"),
    ("_pir_detect_low", "pir_detect_low", "f8"),
    ("_slope_val_high", "slope_val_high", "f8"),
    ("_slope_val_low", "slope_val_low", "f8"),
    ("_linreg_norm_high", "linreg_norm_high", "f8"),   # P8 (np.polyfit)
    ("_linreg_norm_low", "linreg_norm_low", "f8"),     # P8
    ("_vol_surge_high", "vol_surge_high", "f8"),
    ("_vol_surge_low", "vol_surge_low", "f8"),
    ("_mom_div_high", "mom_div_high", "f8"),           # v18 P2.3 raw product
    ("_mom_div_low", "mom_div_low", "f8"),
    ("_mom_vel_high", "mom_vel_high", "f8"),
    ("_mom_vel_low", "mom_vel_low", "f8"),
    ("_vola_pos_high", "vola_pos_high", "f8"),
    ("_vola_pos_low", "vola_pos_low", "f8"),
    ("_gjr_norm", "gjr_norm", "f8"),                   # P2 per-slice seeding
    ("_har_norm", "har_norm", "f8"),                   # P2
    ("_er_gate_ok_high", "er_gate_ok_high", "b1"),
    ("_er_gate_ok_low", "er_gate_ok_low", "b1"),
    ("_price_high_ok", "price_high_ok", "b1"),
    ("_price_low_ok", "price_low_ok", "b1"),
    ("_ph_arr", "ph_arr", "b1"),
    ("_pl_arr", "pl_arr", "b1"),
    ("_high_arr", "high_arr", "f8"),
    ("_low_arr", "low_arr", "f8"),
]

_VOTE_KEYS = [
    "ms_agree_high", "ms_agree_low", "scale_div_high", "scale_div_low",
    "eff_t_up_h", "eff_vs_h", "eff_mv_h", "eff_va_h", "eff_g_h", "eff_h_h",
    "eff_md_h",
    "eff_t_dn_l", "eff_vs_l", "eff_mv_l", "eff_va_l", "eff_g_l", "eff_h_l",
    "eff_md_l",
]


# ---------------------------------------------------------------------------
# Fixtures: >=3 slices (SPX IS-like, SPX short OOS-like, DAX IS-like)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def slices() -> dict[str, pd.DataFrame]:
    if not _SPX.exists() or not _DAX.exists():
        pytest.skip(f"missing {_SPX} or {_DAX}")
    spx = load_stream_frame(str(_SPX))
    dax = load_stream_frame(str(_DAX))
    return {
        "spx_is": spx.iloc[:1500].reset_index(drop=True),
        "spx_oos": spx.iloc[1500:1900].reset_index(drop=True),  # short stream
        "dax_is": dax.iloc[:1200].reset_index(drop=True),
    }


@pytest.fixture(scope="module")
def built(slices):
    """(df, artifacts, FastDetector, TorchPhase1) per slice, base Params."""
    base = Params()
    out = {}
    for name, df in slices.items():
        art = build_detector_artifacts(df)
        out[name] = (df, art, FastDetector(df, base, art), TorchPhase1(df, base, art))
    return out


@pytest.fixture(scope="module")
def built_allvotes(built):
    """ALLVOTES architecture on the SPX IS slice (every threshold active)."""
    df, art, _, _ = built["spx_is"]
    return df, art, FastDetector(df, ALLVOTES, art), TorchPhase1(df, ALLVOTES, art)


def _perturb(base: Params, fields: set[str], rng) -> Params:
    over = {}
    for f in fields:
        v = float(getattr(base, f))
        nv = v * float(rng.uniform(0.7, 1.3))
        if 0.0 < v <= 1.0:
            nv = min(0.99, max(0.01, nv))
        over[f] = nv
    return dataclasses.replace(base, **over)


def _reference_votes(fd: FastDetector, p: Params) -> dict[str, np.ndarray]:
    """Inline replica of the oracle threshold block (v17_fastdetector L207-256)."""
    agr_h_high, scale_div_high_raw = fd._agr("high", p.pct_extreme_high)
    agr_l_low, scale_div_low_raw = fd._agr("low", p.pct_extreme_low)
    ew_h, uev_h = p.edge_window_high, p.use_edge_voting_high
    ew_l, uev_l = p.edge_window_low, p.use_edge_voting_low

    ms_agree_high = _to_bool(agr_h_high >= p.min_agreement_high)
    ms_agree_low = _to_bool(agr_l_low >= p.min_agreement_low)
    scale_div_high = _to_bool(np.abs(scale_div_high_raw) > p.scale_div_thresh_high)
    scale_div_low = _to_bool(np.abs(scale_div_low_raw) > p.scale_div_thresh_low)

    trend_up_h = _to_bool((fd._slope_val_high > p.slope_thresh_high).values
                          & (fd._linreg_norm_high > p.slope_thresh_high).values)
    trend_dn_l = _to_bool((fd._slope_val_low < -p.slope_thresh_low).values
                          & (fd._linreg_norm_low < -p.slope_thresh_low).values)
    vol_slow_h = (fd._vol_surge_high < (1.0 / p.vol_surge_thresh_high))
    vol_surge_l = (fd._vol_surge_low > p.vol_surge_thresh_low)
    if p.momentum_velocity_mode_high == "Reversal":
        mv_h_ok = fd._mom_vel_high <= -abs(p.momentum_velocity_thresh_high)
    else:
        mv_h_ok = fd._mom_vel_high >= abs(p.momentum_velocity_thresh_high)
    if p.momentum_velocity_mode_low == "Reversal":
        mv_l_ok = fd._mom_vel_low >= abs(p.momentum_velocity_thresh_low)
    else:
        mv_l_ok = fd._mom_vel_low <= -abs(p.momentum_velocity_thresh_low)
    vola_h = fd._vola_pos_high > p.vola_high_pct_high
    vola_l = fd._vola_pos_low > p.vola_high_pct_low
    gjr_h = fd._gjr_norm <= -p.gjr_vote_thresh_high
    gjr_l = fd._gjr_norm >= p.gjr_vote_thresh_low
    har_h = fd._har_norm >= p.har_vote_thresh_high
    har_l = fd._har_norm >= p.har_vote_thresh_low
    # v18 P2.3: momentum vote per candidate (0.0 == legacy `< 0`)
    md_h_ok = fd._mom_div_high < -p.momentum_diverge_thresh_high
    md_l_ok = fd._mom_div_low < -p.momentum_diverge_thresh_low

    return {
        "agr_h_high": agr_h_high, "agr_l_low": agr_l_low,
        "scale_div_high_raw": scale_div_high_raw,
        "scale_div_low_raw": scale_div_low_raw,
        "ms_agree_high": ms_agree_high, "ms_agree_low": ms_agree_low,
        "scale_div_high": scale_div_high, "scale_div_low": scale_div_low,
        "eff_t_up_h": _edge_or_state(trend_up_h, ew_h, uev_h),
        "eff_vs_h": _edge_or_state(_to_bool(vol_slow_h), ew_h, uev_h),
        "eff_mv_h": _edge_or_state(_to_bool(mv_h_ok), ew_h, uev_h),
        "eff_va_h": _edge_or_state(_to_bool(vola_h), ew_h, uev_h),
        "eff_g_h": _edge_or_state(_to_bool(gjr_h), ew_h, uev_h),
        "eff_h_h": _edge_or_state(_to_bool(har_h), ew_h, uev_h),
        "eff_md_h": _edge_or_state(_to_bool(md_h_ok), ew_h, uev_h),
        "eff_t_dn_l": _edge_or_state(trend_dn_l, ew_l, uev_l),
        "eff_vs_l": _edge_or_state(_to_bool(vol_surge_l), ew_l, uev_l),
        "eff_mv_l": _edge_or_state(_to_bool(mv_l_ok), ew_l, uev_l),
        "eff_va_l": _edge_or_state(_to_bool(vola_l), ew_l, uev_l),
        "eff_g_l": _edge_or_state(_to_bool(gjr_l), ew_l, uev_l),
        "eff_h_l": _edge_or_state(_to_bool(har_l), ew_l, uev_l),
        "eff_md_l": _edge_or_state(_to_bool(md_l_ok), ew_l, uev_l),
    }


def _assert_votes_equal(got: dict, ref: dict, ctx: str) -> None:
    assert np.array_equal(got["agr_h_high"], ref["agr_h_high"]), ctx
    assert np.array_equal(got["agr_l_low"], ref["agr_l_low"]), ctx
    for k in _VOTE_KEYS:
        assert got[k].dtype == np.bool_, (ctx, k)
        assert np.array_equal(got[k], ref[k]), (ctx, k)


# ---------------------------------------------------------------------------
# P1 — dtype split + no low-precision reductions
# ---------------------------------------------------------------------------


def test_tf32_and_low_precision_reductions_disabled() -> None:
    assert torch.backends.cuda.matmul.allow_tf32 is False
    assert torch.backends.cudnn.allow_tf32 is False
    assert torch.get_float32_matmul_precision() == "highest"


def test_feature_dtype_split(built) -> None:
    for name, (_, _, _, tp) in built.items():
        assert tp.feat.pir_matrix.dtype == np.float64, name      # P1: PIR f64 (Pine-faithful)
        for _, feat_attr, kind in _FEATURE_ATTRS:
            arr = getattr(tp.feat, feat_attr)
            assert arr.dtype == np.dtype(kind), (name, feat_attr)
        agr, sd_raw = tp.agreement("high", Params().pct_extreme_high)
        assert agr.dtype == np.float64 and sd_raw.dtype == np.float64, name


# ---------------------------------------------------------------------------
# §2 winning method — torch PIR byte-identical (P1)
# ---------------------------------------------------------------------------


def test_pir_matrix_torch_byte_identical_on_three_slices(built) -> None:
    for name, (df, art, _, tp) in built.items():
        pir = build_pir_matrix_torch(
            df["close"], art.scales_list[0], art.scales_list[-1])
        assert pir.dtype == np.float64, name
        assert np.array_equal(pir, art.pir_matrix, equal_nan=True), name
        # the evaluator's resident matrix is byte-identical too
        assert np.array_equal(tp.feat.pir_matrix, art.pir_matrix, equal_nan=True), name


def test_torch_pir_source_is_byte_identical_in_evaluator(built) -> None:
    df, art, fd, _ = built["spx_oos"]  # short slice: windows > n_bars edge case
    tp = TorchPhase1(df, Params(), art, use_torch_pir=True)
    assert np.array_equal(tp.feat.pir_matrix, art.pir_matrix, equal_nan=True)
    ref = _reference_votes(fd, Params())
    _assert_votes_equal(tp.votes(Params()), ref, "torch-pir source")


# ---------------------------------------------------------------------------
# Phase-1 feature arrays — np.array_equal vs FastDetector attrs (P1, P2, P8)
# ---------------------------------------------------------------------------


def test_phase1_features_match_fastdetector_on_three_slices(built) -> None:
    for name, (_, _, fd, tp) in built.items():
        for fd_attr, feat_attr, _ in _FEATURE_ATTRS:
            ref = getattr(fd, fd_attr)
            ref = ref.values if isinstance(ref, pd.Series) else ref
            got = getattr(tp.feat, feat_attr)
            assert np.array_equal(got, np.asarray(ref), equal_nan=True), \
                (name, feat_attr)
        assert tp.feat.max_votes_high == fd._max_votes_high, name
        assert tp.feat.max_votes_low == fd._max_votes_low, name
        assert tp.feat.n == fd.n, name


# ---------------------------------------------------------------------------
# Agreement + scale_div (the per-candidate hot op) — exact, incl. boundary pct
# ---------------------------------------------------------------------------


def test_agreement_and_scale_div_match(built) -> None:
    rng = np.random.default_rng(11)
    for name, (_, art, fd, tp) in built.items():
        # a pct sitting EXACTLY on a stored float32 PIR cell of a SELECTED
        # scale row (scale 3 = row 1 is inside the high-side selection)
        row = art.pir_matrix[3 - art.scales_list[0]]
        finite = row[np.isfinite(row)]
        inrange = finite[(finite > 0.05) & (finite < 0.95)]
        boundary = float(inrange[0]) if inrange.size else 0.5
        # one ulp64 below a stored f32 cell: a float64 comparison would flip
        # the count (cell > pct), the oracle's f32 weak-scalar one must not.
        below = float(np.nextafter(boundary, 0.0))
        pcts = [Params().pct_extreme_high, Params().pct_extreme_low,
                0.5, boundary, below] + [float(rng.uniform(0.05, 0.99)) for _ in range(4)]
        for pct in pcts:
            for side in ("high", "low"):
                ref_agr, ref_sd = fd._agr(side, pct)
                got_agr, got_sd = tp.agreement(side, pct)
                assert np.array_equal(got_agr, ref_agr), (name, side, pct)
                assert np.array_equal(got_sd, ref_sd, equal_nan=True), (name, side, pct)


def test_agreement_kernel_exact_at_f32_scalar_boundary(built) -> None:
    """Cache-bypassing kernel check: the pct scalar must downcast to float32
    exactly like numpy's weak-scalar promotion in ``calc_agreement_fast``
    (``fd._agr``/``tp.agreement`` memoize on ``round(pct, 12)``, so one-ulp64
    neighbours share a cache slot there and never reach the kernel)."""
    from src.indicators import calc_agreement_fast

    _, art, _, tp = built["spx_is"]
    p = Params()
    row = art.pir_matrix[3 - art.scales_list[0]]  # scale 3: in the high selection
    finite = row[np.isfinite(row)]
    v = float(finite[(finite > 0.05) & (finite < 0.95)][0])
    for pct in (v, float(np.nextafter(v, 0.0)), float(np.nextafter(v, 1.0))):
        ref_h, ref_l, ref_n = calc_agreement_fast(
            art.pir_matrix, art.scales_list,
            p.scale_start_high, p.scale_end_high, p.scale_step_high, pct)
        got_h, got_l, got_n = tp._agreement_torch(
            p.scale_start_high, p.scale_end_high, p.scale_step_high, pct)
        assert got_n == ref_n
        assert np.array_equal(got_h.cpu().numpy(), ref_h), pct
        assert np.array_equal(got_l.cpu().numpy(), ref_l), pct


# ---------------------------------------------------------------------------
# Threshold layer — every vote array equal under random draws (P3, P4, P6)
# ---------------------------------------------------------------------------


def test_votes_match_reference_on_three_slices_base_params(built) -> None:
    base = Params()
    for name, (_, _, fd, tp) in built.items():
        _assert_votes_equal(tp.votes(base), _reference_votes(fd, base), name)


def test_votes_match_reference_under_threshold_draws(built_allvotes) -> None:
    _, _, fd, tp = built_allvotes
    fields = set(active_threshold_fields(ALLVOTES, "high")
                 + active_threshold_fields(ALLVOTES, "low"))
    rng = np.random.default_rng(0)
    for trial in range(8):
        p = _perturb(ALLVOTES, fields, rng)
        _assert_votes_equal(tp.votes(p), _reference_votes(fd, p), f"trial {trial}")


def test_votes_match_for_non_reversal_momentum_mode(built) -> None:
    df, art, _, _ = built["spx_oos"]
    base = dataclasses.replace(Params(),
                               momentum_velocity_mode_high="Continuation",
                               momentum_velocity_mode_low="Continuation")
    fd = FastDetector(df, base, art)
    tp = TorchPhase1(df, base, art)
    _assert_votes_equal(tp.votes(base), _reference_votes(fd, base), "mv mode")


# ---------------------------------------------------------------------------
# P3 — strict operators verbatim: boundary draws where >= vs > flips the vote
# ---------------------------------------------------------------------------


def test_operator_boundaries_exact(built_allvotes) -> None:
    _, _, fd, tp = built_allvotes
    p0 = ALLVOTES

    agr_h, sd_raw = fd._agr("high", p0.pct_extreme_high)
    t = int(np.argmax(agr_h))
    assert agr_h[t] > 0
    v = tp.votes(dataclasses.replace(p0, min_agreement_high=float(agr_h[t])))
    assert bool(v["ms_agree_high"][t]) is True          # agr >= min_agreement

    finite = np.flatnonzero(np.isfinite(sd_raw) & (np.abs(sd_raw) > 0))
    t = int(finite[len(finite) // 2])
    v = tp.votes(dataclasses.replace(
        p0, scale_div_thresh_high=float(np.abs(sd_raw[t]))))
    assert bool(v["scale_div_high"][t]) is False        # |scale_div| > thresh

    har = fd._har_norm
    t = int(np.flatnonzero(np.isfinite(har) & (har > 0))[0])
    v = tp.votes(dataclasses.replace(p0, har_vote_thresh_high=float(har[t])))
    assert bool(v["eff_h_h"][t]) is True                # har >= thr

    gjr = fd._gjr_norm
    neg = np.flatnonzero(np.isfinite(gjr) & (gjr < 0))
    t = int(neg[0])
    v = tp.votes(dataclasses.replace(p0, gjr_vote_thresh_high=float(-gjr[t])))
    assert bool(v["eff_g_h"][t]) is True                # gjr <= -thr

    vola = fd._vola_pos_high
    t = int(np.flatnonzero(np.isfinite(vola) & (vola > 0))[0])
    v = tp.votes(dataclasses.replace(p0, vola_high_pct_high=float(vola[t])))
    assert bool(v["eff_va_h"][t]) is False              # vola_pos > pct (strict)

    vs_l = fd._vol_surge_low
    t = int(np.flatnonzero(np.isfinite(vs_l) & (vs_l > 0))[0])
    v = tp.votes(dataclasses.replace(p0, vol_surge_thresh_low=float(vs_l[t])))
    assert bool(v["eff_vs_l"][t]) is False              # vol_surge > thresh (strict)

    mv = fd._mom_vel_high
    neg = np.flatnonzero(np.isfinite(mv) & (mv < 0))
    t = int(neg[0])
    v = tp.votes(dataclasses.replace(
        p0, momentum_velocity_thresh_high=float(-mv[t])))
    assert bool(v["eff_mv_h"][t]) is True               # mv <= -abs(thr) (Reversal)


# ---------------------------------------------------------------------------
# P4 — NaN -> False in the warm-up region (first ~300 bars)
# ---------------------------------------------------------------------------


def test_warmup_nan_votes_resolve_false(built) -> None:
    base = Params()
    _, _, fd, tp = built["spx_is"]
    v = tp.votes(base)
    w = 300
    pairs = [
        (fd._linreg_norm_high.values, "eff_t_up_h"),
        (fd._linreg_norm_low.values, "eff_t_dn_l"),
        (fd._vol_surge_high, "eff_vs_h"),
        (fd._vol_surge_low, "eff_vs_l"),
        (fd._mom_vel_high, "eff_mv_h"),
        (fd._mom_vel_low, "eff_mv_l"),
        (fd._mom_div_high, "eff_md_h"),   # v18 P2.3 per-candidate vote
        (fd._mom_div_low, "eff_md_l"),
    ]
    for feat, key in pairs:
        nan_mask = np.isnan(feat[:w])
        assert nan_mask.any(), key  # the warm-up region really contains NaNs
        assert not v[key][:w][nan_mask].any(), key
    # v18 B1: vola_pos (pir_of) is NaN-free — Pine na -> 0.5 warm-up values.
    for feat in (fd._vola_pos_high, fd._vola_pos_low):
        assert not np.isnan(feat).any()
    ref = _reference_votes(fd, base)
    for k in _VOTE_KEYS:
        assert np.array_equal(v[k][:w], ref[k][:w]), k


def test_to_bool_torch_nan_to_false() -> None:
    x = torch.tensor([np.nan, 1.0, 0.0, -2.0, np.nan], dtype=torch.float64)
    got = to_bool_torch(x).numpy()
    ref = _to_bool(x.numpy())
    assert got.dtype == np.bool_
    assert np.array_equal(got, ref)
    b = torch.tensor([True, False, True])
    assert np.array_equal(to_bool_torch(b).numpy(), np.array([True, False, True]))


# ---------------------------------------------------------------------------
# P6 — edge voting as fixed-lag arrays (absolute-bar lag, not pivot count)
# ---------------------------------------------------------------------------


def test_edge_voting_fixed_lag_arrays(built) -> None:
    df, art, _, _ = built["spx_is"]
    base = dataclasses.replace(ALLVOTES,
                               use_edge_voting_high=True, edge_window_high=5,
                               use_edge_voting_low=True, edge_window_low=7)
    fd = FastDetector(df, base, art)
    tp = TorchPhase1(df, base, art)
    v = tp.votes(base)
    _assert_votes_equal(v, _reference_votes(fd, base), "edge base")

    # Explicit fixed-lag construction: eff = state & ~state[t - win]
    state = _to_bool(fd._vola_pos_high > base.vola_high_pct_high)
    win = base.edge_window_high
    shifted = np.zeros_like(state)
    shifted[win:] = state[:-win]
    assert np.array_equal(v["eff_va_h"], state & ~shifted)
    assert v["eff_va_h"].sum() < state.sum()  # the edge filter has teeth here


def test_edge_or_state_torch_matches_oracle_helper() -> None:
    rng = np.random.default_rng(3)
    state = rng.random(200) < 0.3
    for win, use_edge in [(5, True), (7, True), (0, True), (5, False), (250, True)]:
        ref = _edge_or_state(state, win, use_edge)
        got = edge_or_state_torch(torch.from_numpy(state), win, use_edge).numpy()
        assert np.array_equal(got, ref), (win, use_edge)


# ---------------------------------------------------------------------------
# Shape guard — same contract as FastDetector
# ---------------------------------------------------------------------------


def test_guard_rejects_non_threshold_field_change(built) -> None:
    _, _, _, tp = built["spx_oos"]
    bad = dataclasses.replace(Params(), S_detect_high=14)
    with pytest.raises(ValueError, match="non-threshold"):
        tp.votes(bad)
