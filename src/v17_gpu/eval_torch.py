"""Torch Phase-1 features + threshold layer (build-spec §5.3, P1-P4, P6, P8).

Mirrors ``FastDetector._precompute`` (every Phase-1 array — see
``phase1_features``) and the threshold comparisons of
``FastDetector.signals``; the §5.4 batched Phase-2 scan consumes the vote
tensors produced here. PIR-spike verdict (§2): ``trust-kernel`` — the
pandas-faithful per-window mean at internal float64, cast to float32, is
byte-identical to ``indicators.precompute_matrices``; that method is
``build_pir_matrix_torch``.

Parity layout (what runs where, and why it is byte-identical):

- **PIR matrix (P1, float32):** built in torch by the §2 winning method, or
  reused from the per-slice ``DetectorArtifacts`` — both byte-identical.
- **Per-candidate hot path — torch:** agreement at any ``pct_extreme``
  (``calc_agreement_fast`` semantics: float32 comparisons + integer counts),
  ``scale_div``, every threshold comparison with the VERBATIM operators of
  ``v17_fastdetector.signals`` (P3), NaN -> False (P4 — IEEE comparisons with
  NaN are False in torch exactly as in numpy), and ``_edge_or_state`` as
  fixed-lag tensor ops (P6). These are pure comparisons / exact float64
  element-wise arithmetic, so they are bitwise-reproducible on any device.
- **Shape-only per-slice precompute** — the oracle's own pandas/numpy
  helpers in ``phase1_features`` (P2 slice-local seeding; P8 default (a)),
  uploaded once per slice as float64 tensors (P1).
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd
import torch

from src.detector import DetectorArtifacts, build_detector_artifacts
from src.indicators import Params
from src.v17_gpu.phase1_features import Phase1Features, compute_phase1_features
from src.v17_optimize import active_threshold_fields

logger = logging.getLogger(__name__)

__all__ = [
    "Phase1Features",
    "TorchPhase1",
    "build_pir_matrix_torch",
    "compute_phase1_features",
    "edge_or_state_torch",
    "to_bool_torch",
    # §5.4 names (implemented in phase2_scan, lazily re-exported below)
    "LaneInputs",
    "candidate_lane_inputs",
    "scan_phase2",
    "signals_torch",
    "batched_signals",
    "GpuPooledScorer",
]

#: §5.4 lives in ``phase2_scan`` (split for the <=~400-line file rule);
#: PEP 562 lazy re-export keeps ``eval_torch`` the spec's single import
#: surface without a circular import.
_PHASE2_NAMES = frozenset({
    "LaneInputs", "candidate_lane_inputs", "scan_phase2",
    "signals_torch", "batched_signals", "GpuPooledScorer",
})


def __getattr__(name: str):
    if name in _PHASE2_NAMES:
        from src.v17_gpu import phase2_scan
        return getattr(phase2_scan, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def _disable_low_precision() -> None:
    """P1: no TF32 / low-precision reductions on threshold-feeding ops."""
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")


_disable_low_precision()


# ---------------------------------------------------------------------------
# Torch primitives (exact mirrors of the oracle helpers)
# ---------------------------------------------------------------------------


def to_bool_torch(x: torch.Tensor) -> torch.Tensor:
    """``detector._to_bool`` in torch: NaN -> False, else truthiness (P4)."""
    if x.dtype == torch.bool:
        return x.clone()
    return (x != 0) & ~torch.isnan(x)


def edge_or_state_torch(state: torch.Tensor, win: int, use_edge: bool) -> torch.Tensor:
    """``detector._edge_or_state`` in torch (P6 — absolute-bar fixed lag).

    Args:
        state: bool vote-state tensor; the bar axis is the LAST axis.
        win: edge window in bars.
        use_edge: when False, returns ``state`` unchanged.

    Returns:
        Bool tensor of effective per-bar votes, same shape as ``state``.
    """
    if not use_edge or win <= 0:
        return state.clone()
    out = state.clone()
    if win < out.shape[-1]:
        shifted = torch.zeros_like(out)
        shifted[..., win:] = out[..., :-win]
        out = out & ~shifted
    return out


def _rolling_mean_unfold(x: torch.Tensor, window: int) -> torch.Tensor:
    """Per-window (fresh-sum) rolling mean — the §2 'pandas-faithful' method."""
    n = x.shape[0]
    out = torch.full((n,), float("nan"), dtype=x.dtype, device=x.device)
    if 0 < window <= n:
        out[window - 1:] = x.unfold(0, window, 1).sum(dim=1) / window
    return out


def _pir_torch(val: torch.Tensor, lb: int) -> torch.Tensor:
    """Torch mirror of ``indicators.pir_of`` (v18 B1 Pine-true semantics).

    Bars ``t < lb-1`` -> 0.5 (ta.lowest/highest na on insufficient chart
    history); na values INSIDE a window are skipped (nan-ignoring min/max);
    an all-na window -> 0.5 (Pine's na ternary takes the 0.5 branch).
    """
    n = val.shape[0]
    out = torch.full((n,), 0.5, dtype=val.dtype, device=val.device)
    if 0 < lb <= n:
        windows = val.unfold(0, lb, 1)
        posinf = torch.tensor(float("inf"), dtype=val.dtype, device=val.device)
        lo = torch.where(torch.isnan(windows), posinf, windows).amin(dim=1)
        hi = torch.where(torch.isnan(windows), -posinf, windows).amax(dim=1)
        v = val[lb - 1:]
        span = (hi - lo).clamp(min=1e-10)
        res = (v - lo) / span
        res = torch.where(hi != lo, res, torch.full_like(res, 0.5))
        res = torch.where(torch.isinf(lo) | torch.isinf(hi),
                          torch.full_like(res, 0.5), res)
        out[lb - 1:] = res
    return out


def build_pir_matrix_torch(
    close: "pd.Series | np.ndarray",
    scale_min: int = 2,
    scale_max: int = 500,
    device: str = "cpu",
) -> np.ndarray:
    """Float64 PIR matrix mirroring the PINE-FAITHFUL ``precompute_matrices``.

    2026-06-10 TV-export audit (warm-up re-audited 2026-06-11, v18 B1):
    Pine's parity shim computes SMA as a ``ta.cum`` cumulative-sum difference
    (first valid at bar ``s``); the ratio ternary ``sma > 0 ? c/sma : 1.0``
    takes the FALSE branch on na, so warm-up ratio is **1.0, never na**, and
    the pir min/max scan covers exactly ``lb`` bars including virtual
    pre-history bars (also 1.0), with ``hi != lo ? (v-lo)/(hi-lo) : 0.5`` in
    float64. This builder mirrors that construction in torch; the production
    path uses the numpy oracle matrix from ``DetectorArtifacts`` directly, so
    this exists for on-device validation (the Colab script measures its
    real-hardware flip rate, where a parallel-scan cumsum may differ from
    numpy's sequential one).

    Args:
        close: close prices for ONE slice (per-slice artifact — P2).
        scale_min: smallest SMA scale (inclusive).
        scale_max: largest SMA scale (inclusive).
        device: torch device for the build.

    Returns:
        float64 array ``[n_scales, n_bars]`` (NaN where the SMA is invalid).
    """
    close64 = np.asarray(
        close.values if isinstance(close, pd.Series) else close, dtype=np.float64
    )
    scales = list(range(scale_min, scale_max + 1))
    n_bars = len(close64)
    out = np.full((len(scales), n_bars), np.nan, dtype=np.float64)
    x = torch.tensor(close64, dtype=torch.float64, device=device)
    csum = torch.cumsum(x, dim=0)
    for i, s in enumerate(scales):
        # SMA via cumsum difference, valid t >= s (Pine sma_at semantics).
        sma_t = torch.full((n_bars,), float("nan"), dtype=torch.float64, device=device)
        if s < n_bars:
            sma_t[s:] = (csum[s:] - csum[:-s]) / s
        # Pine ternary `sma > 0 ? c/sma : 1.0`: na condition -> FALSE branch,
        # so warm-up ratio is 1.0, never NaN (v18 B1).
        ratio = torch.where(sma_t > 0, x / sma_t, torch.ones_like(x))
        lb = max(s, 20)
        # Pine scans exactly lb bars; virtual pre-history bars are ratio 1.0.
        pad = lb - 1
        one = torch.ones(pad, dtype=torch.float64, device=device)
        v = torch.cat([one, ratio])
        lo = v.unfold(0, lb, 1).min(dim=1).values
        hi = v.unfold(0, lb, 1).max(dim=1).values
        res = torch.where(hi != lo, (ratio - lo) / (hi - lo),
                          torch.full_like(ratio, 0.5))
        out[i] = res.cpu().numpy()
    logger.debug("torch PIR matrix built: %d scales x %d bars", len(scales), n_bars)
    return out


# ---------------------------------------------------------------------------
# TorchPhase1 — uploaded features + the per-candidate threshold layer
# ---------------------------------------------------------------------------

_TENSOR_F64 = [
    "pir_detect_high", "pir_detect_low", "slope_val_high", "slope_val_low",
    "linreg_norm_high", "linreg_norm_low", "vol_surge_high", "vol_surge_low",
    "mom_div_high", "mom_div_low",                        # v18 P2.3 (raw)
    "mom_vel_high", "mom_vel_low", "vola_pos_high", "vola_pos_low",
    "gjr_norm", "har_norm",
]
_TENSOR_BOOL = [
    "er_gate_ok_high", "er_gate_ok_low",
    "price_high_ok", "price_low_ok", "ph_arr", "pl_arr",
]


class TorchPhase1:
    """Precompute shape-dependent tensors once; torch threshold layer per eval.

    The drop-in torch counterpart of ``FastDetector`` up to (and including)
    the threshold comparisons; the §5.4 batched Phase-2 scan consumes
    ``votes_t``. Same contract: only fields in ``active_threshold_fields``
    may differ from the base (guarded).
    """

    def __init__(self, df: pd.DataFrame, base_params: Params,
                 artifacts: Optional[DetectorArtifacts] = None,
                 device: str = "cpu", use_torch_pir: bool = False) -> None:
        self.df = df.reset_index(drop=True)
        self.base = base_params
        self.device = torch.device(device)
        self.art = artifacts or build_detector_artifacts(self.df)
        if use_torch_pir:
            pir = build_pir_matrix_torch(self.df["close"], self.art.scales_list[0],
                                         self.art.scales_list[-1], device)
        else:
            pir = self.art.pir_matrix
        self.feat = compute_phase1_features(self.df, base_params, self.art, pir)
        self._varied = set(active_threshold_fields(base_params, "high")
                           + active_threshold_fields(base_params, "low"))
        self._agr_cache: dict = {}
        self._t: dict[str, torch.Tensor] = {
            "pir": torch.from_numpy(np.ascontiguousarray(self.feat.pir_matrix)).to(self.device),
        }
        for name in _TENSOR_F64:
            arr = np.ascontiguousarray(getattr(self.feat, name), dtype=np.float64)
            self._t[name] = torch.from_numpy(arr).to(self.device)
        for name in _TENSOR_BOOL:
            arr = np.ascontiguousarray(getattr(self.feat, name))
            self._t[name] = torch.from_numpy(arr).to(self.device)
        logger.debug("TorchPhase1 ready: n=%d, device=%s, torch_pir=%s",
                     self.feat.n, self.device, use_torch_pir)

    # -- agreement (calc_agreement_fast semantics, per-candidate hot op) ----
    def _agreement_torch(self, scale_start: int, scale_end: int, scale_step: int,
                         pct: float) -> tuple[torch.Tensor, torch.Tensor, int]:
        scales_list = self.feat.scales_list
        base = scales_list[0]
        idx = [s - base for s in range(scale_start, scale_end + 1, scale_step)
               if base <= s <= scales_list[-1]]
        n_bars = self.feat.n
        if not idx:
            z = torch.zeros(n_bars, dtype=torch.float64, device=self.device)
            return z, z.clone(), 0
        sliced = self._t["pir"][idx]                     # float32 (P1)
        n_scales = len(idx)
        # float32 comparisons + integer counts, then ONE float64 division —
        # bit-for-bit `calc_agreement_fast` (the scalar pct downcasts to f32
        # under both numpy and torch weak-scalar promotion).
        high = (sliced > pct).sum(dim=0).to(torch.float64) / n_scales
        low = (sliced < (1.0 - pct)).sum(dim=0).to(torch.float64) / n_scales
        return high, low, n_scales

    def _agr_t(self, side: str, pct: float) -> tuple[torch.Tensor, torch.Tensor]:
        """(agr, scale_div_raw) tensors for one side — mirrors FastDetector._agr."""
        key = (side, round(float(pct), 12))
        if key in self._agr_cache:
            return self._agr_cache[key]
        # Bound the memo: CMA-ES proposes a FRESH pct per candidate, so an
        # unbounded cache retains ~0.7 GB of dead tensors per generation
        # (observed OOM trajectory on the 16-stream pool, 2026-06-10). The
        # cache only ever pays off for repeated pcts (finalist re-scores,
        # coordinate ascent); clearing it is value-neutral (pure memo).
        if len(self._agr_cache) >= 64:
            self._agr_cache.clear()
        p = self.base
        if side == "high":
            agr, _, _ = self._agreement_torch(
                p.scale_start_high, p.scale_end_high, p.scale_step_high, pct)
            scale_div_raw = self._t["pir_detect_high"] - agr
        else:
            _, agr, _ = self._agreement_torch(
                p.scale_start_low, p.scale_end_low, p.scale_step_low, pct)
            scale_div_raw = (1.0 - self._t["pir_detect_low"]) - agr
        self._agr_cache[key] = (agr, scale_div_raw)
        return self._agr_cache[key]

    def agreement(self, side: str, pct: float) -> tuple[np.ndarray, np.ndarray]:
        """(agr, scale_div_raw) numpy float64 — parity surface for the tests."""
        agr, sd = self._agr_t(side, pct)
        return agr.cpu().numpy(), sd.cpu().numpy()

    # -- threshold layer (verbatim operators — P3; NaN -> False — P4) -------
    def _guard(self, p: Params) -> None:
        for f in vars(self.base):
            if f in self._varied:
                continue
            if getattr(p, f) != getattr(self.base, f):
                raise ValueError(f"TorchPhase1: non-threshold field {f!r} changed; "
                                 "rebuild TorchPhase1 for a new shape/architecture.")

    def votes_t(self, params: Params) -> dict[str, torch.Tensor]:
        """All vote/gate tensors for one candidate (§5.4 scan inputs)."""
        p = params
        self._guard(p)
        t = self._t
        agr_h, sd_h_raw = self._agr_t("high", p.pct_extreme_high)
        agr_l, sd_l_raw = self._agr_t("low", p.pct_extreme_low)
        ew_h, uev_h = p.edge_window_high, p.use_edge_voting_high
        ew_l, uev_l = p.edge_window_low, p.use_edge_voting_low

        # P3: operators verbatim from v17_fastdetector.signals (never normalized)
        ms_agree_high = agr_h >= p.min_agreement_high
        ms_agree_low = agr_l >= p.min_agreement_low
        scale_div_high = sd_h_raw.abs() > p.scale_div_thresh_high
        scale_div_low = sd_l_raw.abs() > p.scale_div_thresh_low
        trend_up_h = ((t["slope_val_high"] > p.slope_thresh_high)
                      & (t["linreg_norm_high"] > p.slope_thresh_high))
        trend_dn_l = ((t["slope_val_low"] < -p.slope_thresh_low)
                      & (t["linreg_norm_low"] < -p.slope_thresh_low))
        vol_slow_h = t["vol_surge_high"] < (1.0 / p.vol_surge_thresh_high)
        vol_surge_l = t["vol_surge_low"] > p.vol_surge_thresh_low
        if p.momentum_velocity_mode_high == "Reversal":
            mv_h_ok = t["mom_vel_high"] <= -abs(p.momentum_velocity_thresh_high)
        else:
            mv_h_ok = t["mom_vel_high"] >= abs(p.momentum_velocity_thresh_high)
        if p.momentum_velocity_mode_low == "Reversal":
            mv_l_ok = t["mom_vel_low"] >= abs(p.momentum_velocity_thresh_low)
        else:
            mv_l_ok = t["mom_vel_low"] <= -abs(p.momentum_velocity_thresh_low)
        # v18 P2.3: momentum vote per candidate (0.0 == legacy `< 0`)
        md_h_ok = t["mom_div_high"] < -p.momentum_diverge_thresh_high
        md_l_ok = t["mom_div_low"] < -p.momentum_diverge_thresh_low
        vola_h = t["vola_pos_high"] > p.vola_high_pct_high
        vola_l = t["vola_pos_low"] > p.vola_high_pct_low
        gjr_h = t["gjr_norm"] <= -p.gjr_vote_thresh_high
        gjr_l = t["gjr_norm"] >= p.gjr_vote_thresh_low
        har_h = t["har_norm"] >= p.har_vote_thresh_high
        har_l = t["har_norm"] >= p.har_vote_thresh_low

        return {
            "agr_h_high": agr_h, "agr_l_low": agr_l,
            "ms_agree_high": ms_agree_high, "ms_agree_low": ms_agree_low,
            "scale_div_high": scale_div_high, "scale_div_low": scale_div_low,
            "eff_t_up_h": edge_or_state_torch(trend_up_h, ew_h, uev_h),
            "eff_vs_h": edge_or_state_torch(vol_slow_h, ew_h, uev_h),
            "eff_mv_h": edge_or_state_torch(mv_h_ok, ew_h, uev_h),
            "eff_va_h": edge_or_state_torch(vola_h, ew_h, uev_h),
            "eff_g_h": edge_or_state_torch(gjr_h, ew_h, uev_h),
            "eff_h_h": edge_or_state_torch(har_h, ew_h, uev_h),
            "eff_md_h": edge_or_state_torch(md_h_ok, ew_h, uev_h),
            "eff_t_dn_l": edge_or_state_torch(trend_dn_l, ew_l, uev_l),
            "eff_vs_l": edge_or_state_torch(vol_surge_l, ew_l, uev_l),
            "eff_mv_l": edge_or_state_torch(mv_l_ok, ew_l, uev_l),
            "eff_va_l": edge_or_state_torch(vola_l, ew_l, uev_l),
            "eff_g_l": edge_or_state_torch(gjr_l, ew_l, uev_l),
            "eff_h_l": edge_or_state_torch(har_l, ew_l, uev_l),
            "eff_md_l": edge_or_state_torch(md_l_ok, ew_l, uev_l),
            "er_high": t["er_gate_ok_high"], "er_low": t["er_gate_ok_low"],
            "price_high": t["price_high_ok"], "price_low": t["price_low_ok"],
        }

    def votes(self, params: Params) -> dict[str, np.ndarray]:
        """Numpy view of ``votes_t`` (the parity surface for the tests)."""
        return {k: v.cpu().numpy() for k, v in self.votes_t(params).items()}
