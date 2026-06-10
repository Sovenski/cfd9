"""Batched Phase-2 scan + ``score_pop`` (build-spec §5.4 — P5, P7).

The §5.4 half of ``eval_torch`` (split out for the <=~400-line file rule;
``eval_torch`` re-exports these names). Reproduces the stateful Phase-2 loop
of ``FastDetector.signals`` / ``SpeculatorDetector._detect`` as a hand-written
batched bar-loop over ``(candidate x asset)`` lanes:

- Carry per lane = ``{dur_at, dur_miss, bars_since}`` per side, int64. The
  pivot ring is NOT carried: §5.1's ``precompute_drift`` already materialises
  the read-BEFORE-append / HIGH-before-LOW ring semantics (P5) as a per-asset
  float64 drift array, so the drift votes become fixed-lag arrays (P6) built
  OUTSIDE the loop.
- One wide vector op per bar-step; ``valid_mask`` FREEZES every counter on
  pad bars and each asset is its own lane with fresh carry, so a packed batch
  can never leak cooldown or pivots across instruments (P7).
- ``GpuPooledScorer.score_pop`` maps the batched signals through the EXACT
  CPU pooled reduction (``_stream_stat`` -> ``pooled_fold_score`` ->
  ``fold_scores_bootstrap_ci``) for numeric identity with ``PooledScorer``.

Byte-identity with the CPU oracle is the contract
(``tests/test_v17_gpu_parity.py``, §0.7 — strict ``np.array_equal``).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import pandas as pd
import torch

from src.detector import _USE_PIVOT_DRIFT, _USE_PRICE_GATE
from src.indicators import Params
from src.v17_gpu.drift_precompute import HIGH, LOW, DriftSpec, precompute_drift
from src.v17_gpu.eval_torch import TorchPhase1, edge_or_state_torch
from src.v17_gpu.upload import valid_mask_from_lengths

logger = logging.getLogger(__name__)

__all__ = [
    "LaneInputs",
    "candidate_lane_inputs",
    "scan_phase2",
    "signals_torch",
    "batched_signals",
    "GpuPooledScorer",
]

#: High-side vote terms: (use-flag field, vote-tensor key) — order mirrors
#: the ``ph_c`` sum in ``v17_fastdetector.signals`` (the sum is commutative,
#: the order is kept for auditability).
_HIGH_VOTES = (
    ("use_trend_high", "eff_t_up_h"),
    ("use_volume_high", "eff_vs_h"),
    ("use_momentum_high", "eff_md_h"),
    ("use_momentum_velocity_high", "eff_mv_h"),
    ("use_volatility_high", "eff_va_h"),
    ("use_gjr_asym_high", "eff_g_h"),
    ("use_har_vol_high", "eff_h_h"),
)
_LOW_VOTES = (
    ("use_trend_low", "eff_t_dn_l"),
    ("use_volume_low", "eff_vs_l"),
    ("use_momentum_low", "eff_md_l"),
    ("use_momentum_velocity_low", "eff_mv_l"),
    ("use_volatility_low", "eff_va_l"),
    ("use_gjr_asym_low", "eff_g_l"),
    ("use_har_vol_low", "eff_h_l"),
)


@dataclass(frozen=True)
class LaneInputs:
    """Loop-free per-bar inputs of the Phase-2 scan for ONE (candidate, asset).

    ``pre_*`` folds every bar-local condition of the oracle loop (gates, vote
    count vs required count, drift gate) — only the carried counters remain
    for the scan. ``dur_hit_*`` is the per-bar ``agr > dur_extreme_pct`` hit.
    """

    pre_high: torch.Tensor      # bool [n_bars]
    pre_low: torch.Tensor       # bool [n_bars]
    dur_hit_high: torch.Tensor  # bool [n_bars]
    dur_hit_low: torch.Tensor   # bool [n_bars]


def candidate_lane_inputs(
    tp: TorchPhase1, params: Params, drift: torch.Tensor
) -> LaneInputs:
    """Build the scan inputs for one candidate on one slice.

    Args:
        tp: precomputed torch Phase-1 evaluator for the slice (§5.3).
        params: candidate Params (thresholds may differ from ``tp.base``;
            shape/architecture is guarded inside ``votes_t``).
        drift: float64 ``[n_bars, 2]`` per-asset drift from §5.1
            (``precompute_drift`` — P5 ring semantics already applied).

    Returns:
        ``LaneInputs`` with bool tensors of length ``tp.feat.n``.
    """
    p = params
    v = tp.votes_t(p)  # shape guard + verbatim threshold operators (P3/P4)

    # --- drift votes (P5 values; P6 fixed-lag edge arrays) -----------------
    drift_h = drift[:, HIGH]
    drift_l = drift[:, LOW]
    drift_down_high = drift_h < -p.pivot_drift_thresh_high
    drift_up_high = drift_h > p.pivot_drift_thresh_high
    drift_gate_up_high = drift_h > (p.pivot_drift_thresh_high
                                    * p.pivot_drift_gate_mult_high)
    drift_up_low = drift_l > p.pivot_drift_thresh_low
    eff_pd_dn_h = edge_or_state_torch(drift_down_high, p.edge_window_high,
                                      p.use_edge_voting_high)
    eff_pd_up_l = edge_or_state_torch(drift_up_low, p.edge_window_low,
                                      p.use_edge_voting_low)

    # --- vote counts (verbatim ph_c / pl_c sums) ---------------------------
    i64 = torch.int64
    ph_c = eff_pd_dn_h.to(i64) if _USE_PIVOT_DRIFT else \
        torch.zeros_like(drift_h, dtype=i64)
    for flag, key in _HIGH_VOTES:
        if getattr(p, flag):
            ph_c = ph_c + v[key].to(i64)
    pl_c = eff_pd_up_l.to(i64) if _USE_PIVOT_DRIFT else \
        torch.zeros_like(drift_l, dtype=i64)
    for flag, key in _LOW_VOTES:
        if getattr(p, flag):
            pl_c = pl_c + v[key].to(i64)

    # --- required counts: max(1, min(max_votes, cc +/- drift bias)) --------
    # int(_USE_PIVOT_DRIFT and drift_up and bias) == drift_up * bias when the
    # drift vote is enabled (Python `and` returns the int bias verbatim).
    mvh = max(tp.feat.max_votes_high, 1)
    mvl = max(tp.feat.max_votes_low, 1)
    bias_h = int(p.pivot_drift_confirm_bias_high) if _USE_PIVOT_DRIFT else 0
    bias_l = int(p.pivot_drift_confirm_bias_low) if _USE_PIVOT_DRIFT else 0
    high_req = torch.clamp(p.confirm_count_high
                           + drift_up_high.to(i64) * bias_h, 1, mvh)
    low_req = torch.clamp(p.confirm_count_low
                          - drift_up_low.to(i64) * bias_l, 1, mvl)

    # --- gates (everything except the carried dur/cooldown state) ----------
    gate_h = v["ms_agree_high"] & ~v["scale_div_high"] & v["er_high"]
    gate_l = v["ms_agree_low"] & ~v["scale_div_low"] & v["er_low"]
    if _USE_PRICE_GATE:
        gate_h = gate_h & v["price_high"]
        gate_l = gate_l & v["price_low"]
    if _USE_PIVOT_DRIFT:
        gate_h = gate_h & ~drift_gate_up_high

    return LaneInputs(
        pre_high=gate_h & (ph_c >= high_req),
        pre_low=gate_l & (pl_c >= low_req),
        dur_hit_high=v["agr_h_high"] > p.dur_extreme_pct_high,
        dur_hit_low=v["agr_l_low"] > p.dur_extreme_pct_low,
    )


def scan_phase2(
    pre_high: torch.Tensor,
    pre_low: torch.Tensor,
    dur_hit_high: torch.Tensor,
    dur_hit_low: torch.Tensor,
    valid_mask: torch.Tensor,
    params: Params,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The stateful Phase-2 loop, batched over leading lane dims (P7).

    Carry per lane: ``dur_at``/``dur_miss``/``bars_since`` per side (int64).
    One wide vector op per bar-step; ``valid_mask`` freezes all carry on pad
    bars, and a fresh carry per lane means no cross-asset leak.

    Args:
        pre_high, pre_low, dur_hit_high, dur_hit_low: bool ``[..., n_bars]``.
        valid_mask: bool, broadcastable to ``[..., n_bars]`` (P7).
        params: supplies the FROZEN scan scalars (``min_duration_*``,
            ``cooldown_bars_*`` — never per-candidate threshold fields).

    Returns:
        ``(signal_high, signal_low)`` bool tensors shaped like ``pre_high``.
    """
    p = params
    lanes = pre_high.shape[:-1]
    n = pre_high.shape[-1]
    device = pre_high.device
    zeros = torch.zeros(lanes, dtype=torch.int64, device=device)
    dur_at_h, dur_miss_h = zeros.clone(), zeros.clone()
    dur_at_l, dur_miss_l = zeros.clone(), zeros.clone()
    bars_h = torch.full(lanes, 999, dtype=torch.int64, device=device)
    bars_l = torch.full(lanes, 999, dtype=torch.int64, device=device)
    out_h = torch.zeros(pre_high.shape, dtype=torch.bool, device=device)
    out_l = torch.zeros(pre_high.shape, dtype=torch.bool, device=device)

    for t in range(n):
        valid = valid_mask[..., t]
        inc = valid.to(torch.int64)
        bars_h = bars_h + inc
        bars_l = bars_l + inc

        # duration counters (verbatim miss-then-reset semantics), frozen on
        # pad bars (P7)
        hit_h = dur_hit_high[..., t]
        miss_h = torch.where(hit_h, zeros, dur_miss_h + 1)
        at_h = torch.where(hit_h, dur_at_h + 1,
                           torch.where(miss_h > 1, zeros, dur_at_h))
        dur_miss_h = torch.where(valid, miss_h, dur_miss_h)
        dur_at_h = torch.where(valid, at_h, dur_at_h)

        hit_l = dur_hit_low[..., t]
        miss_l = torch.where(hit_l, zeros, dur_miss_l + 1)
        at_l = torch.where(hit_l, dur_at_l + 1,
                           torch.where(miss_l > 1, zeros, dur_at_l))
        dur_miss_l = torch.where(valid, miss_l, dur_miss_l)
        dur_at_l = torch.where(valid, at_l, dur_at_l)

        sig_h = (pre_high[..., t] & (dur_at_h >= p.min_duration_high)
                 & (bars_h > p.cooldown_bars_high) & valid)
        sig_l = (pre_low[..., t] & (dur_at_l >= p.min_duration_low)
                 & (bars_l > p.cooldown_bars_low) & valid)
        bars_h = torch.where(sig_h, zeros, bars_h)
        bars_l = torch.where(sig_l, zeros, bars_l)
        out_h[..., t] = sig_h
        out_l[..., t] = sig_l

    return out_h, out_l


def signals_torch(
    tp: TorchPhase1, params: Params, drift: Optional[torch.Tensor] = None
) -> dict[str, np.ndarray]:
    """Single-slice signals through the batched scan — the parity surface.

    Drop-in for ``FastDetector.signals``: returns numpy bool arrays that are
    byte-identical to ``SpeculatorDetector(...).run()`` (§0.7).
    """
    if drift is None:
        drift = torch.from_numpy(
            precompute_drift(tp.df, DriftSpec.from_params(tp.base))
        ).to(tp.device)
    lane = candidate_lane_inputs(tp, params, drift)
    valid = torch.ones((1, tp.feat.n), dtype=torch.bool, device=tp.device)
    sig_h, sig_l = scan_phase2(
        lane.pre_high[None], lane.pre_low[None],
        lane.dur_hit_high[None], lane.dur_hit_low[None], valid, params)
    return {"signal_high": sig_h[0].cpu().numpy(),
            "signal_low": sig_l[0].cpu().numpy()}


def batched_signals(
    lanes: Sequence[tuple[TorchPhase1, torch.Tensor]],
    params_list: Sequence[Params],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """ONE segmented scan over every ``(candidate x asset)`` lane (P7).

    Args:
        lanes: per asset-slice ``(TorchPhase1, drift_tensor)``; all built
            from the same FROZEN base Params.
        params_list: the candidate batch (threshold fields only may differ).

    Returns:
        ``(signal_high, signal_low, valid_mask)`` — signals bool
        ``[n_candidates, n_lanes, max_bars]`` (pad bars are always False),
        valid_mask bool ``[n_lanes, max_bars]``.
    """
    if not lanes:
        raise ValueError("batched_signals needs at least one lane")
    if not params_list:
        raise ValueError("batched_signals needs at least one candidate")
    device = lanes[0][0].device
    lengths = np.asarray([tp.feat.n for tp, _ in lanes], dtype=np.int64)
    max_bars = int(lengths.max())
    shape = (len(params_list), len(lanes), max_bars)
    pre_h = torch.zeros(shape, dtype=torch.bool, device=device)
    pre_l = torch.zeros_like(pre_h)
    hit_h = torch.zeros_like(pre_h)
    hit_l = torch.zeros_like(pre_h)
    valid = torch.from_numpy(valid_mask_from_lengths(lengths, max_bars)).to(device)
    for ci, p in enumerate(params_list):
        for li, (tp, drift) in enumerate(lanes):
            lane = candidate_lane_inputs(tp, p, drift)
            nb = tp.feat.n
            pre_h[ci, li, :nb] = lane.pre_high
            pre_l[ci, li, :nb] = lane.pre_low
            hit_h[ci, li, :nb] = lane.dur_hit_high
            hit_l[ci, li, :nb] = lane.dur_hit_low
    sig_h, sig_l = scan_phase2(pre_h, pre_l, hit_h, hit_l, valid,
                               params_list[0])
    logger.debug("batched scan: %d candidates x %d lanes x %d bars",
                 *shape)
    return sig_h, sig_l, valid


class GpuPooledScorer:
    """Population scorer: batched torch scan + the EXACT CPU pooled reduction.

    Drop-in for ``PooledScorer``/``FastPooledScorer`` with a vectorised
    ``score_pop``: signals come from ONE segmented scan over every
    ``(candidate x slice)`` lane; the per-fold stats, informative-fold filter,
    firing penalty and block-bootstrap LCB reuse the oracle's own functions
    for numeric identity (§0.7: ``abs(gpu_LCB - PooledScorer_LCB) < 1e-9``).
    """

    def __init__(self, folds, streams, side: str, base_params: Params,
                 n_boot: int = 1000, alpha: float = 0.10, block_len: int = 2,
                 firing_penalty: float = 0.0, firing_cap: float = 2.0,
                 device: str = "cpu") -> None:
        from src.pooled_validation import _fold_is_informative, cluster_weights
        from src.scoring import REFERENCE_N
        if side not in ("high", "low"):
            raise ValueError(f"side must be 'high'|'low', got {side!r}")
        self.side = side
        self.base = base_params
        self.device = device
        self.n_boot, self.alpha, self.block_len = n_boot, alpha, block_len
        self.firing_penalty, self.firing_cap = firing_penalty, firing_cap
        self._weights = cluster_weights(streams)
        self._is_informative = _fold_is_informative
        spec = DriftSpec.from_params(base_params)
        lbl = 1 if side == "high" else -1
        col = f"pivot_N{REFERENCE_N}"
        # Label-informative folds only (same filter as PooledScorer); one
        # (TorchPhase1, drift) lane per IS/OOS slice, precomputed once.
        self._lanes: list[tuple[TorchPhase1, torch.Tensor]] = []
        self._folds_meta: list[list[tuple[int, int, object]]] = []
        for fold in folds:
            if sum(int((sl.df_oos[col] == lbl).sum()) for sl in fold) <= 0:
                continue
            metas: list[tuple[int, int, object]] = []
            for sl in fold:
                li_is = len(self._lanes)
                self._lanes.append((
                    TorchPhase1(sl.df_is, base_params, sl.artifacts_is,
                                device=device),
                    torch.from_numpy(precompute_drift(sl.df_is, spec)).to(device),
                ))
                li_oos = len(self._lanes)
                self._lanes.append((
                    TorchPhase1(sl.df_oos, base_params, sl.artifacts_oos,
                                device=device),
                    torch.from_numpy(precompute_drift(sl.df_oos, spec)).to(device),
                ))
                metas.append((li_is, li_oos, sl))
            self._folds_meta.append(metas)
        logger.info("GpuPooledScorer[%s]: %d informative folds, %d lanes",
                    side, len(self._folds_meta), len(self._lanes))

    def _fold_scores_pop(self, params_list: Sequence[Params],
                         regularize: bool) -> list[list[float]]:
        from src.pooled_scoring import pooled_fold_score
        from src.pooled_validation import _stream_stat
        from src.v17_acceptance import firing_excess
        sig_h, sig_l, _ = batched_signals(self._lanes, params_list)
        sig = (sig_h if self.side == "high" else sig_l).cpu().numpy()
        all_scores: list[list[float]] = []
        for ci in range(len(params_list)):
            out: list[float] = []
            for metas in self._folds_meta:
                is_stats, oos_stats = [], []
                for li_is, li_oos, sl in metas:
                    w = self._weights.get(sl.stream.stream_id, 1.0)
                    s_is = pd.Series(sig[ci, li_is, :len(sl.df_is)],
                                     index=sl.df_is.index)
                    s_oos = pd.Series(sig[ci, li_oos, :len(sl.df_oos)],
                                      index=sl.df_oos.index)
                    is_stats.append(_stream_stat(sl.df_is, s_is, self.side, w))
                    oos_stats.append(_stream_stat(sl.df_oos, s_oos, self.side, w))
                s, comp = pooled_fold_score(is_stats, oos_stats, self.side)
                if not self._is_informative(comp):
                    continue
                if regularize and self.firing_penalty > 0:
                    pen = firing_excess(comp.get("precision_oos", 0.0),
                                        comp.get("recall_oos", 0.0),
                                        self.firing_cap)
                    s = s - self.firing_penalty * pen
                out.append(s)
            all_scores.append(out)
        return all_scores

    def score_pop(self, params_list: Sequence[Params]) -> np.ndarray:
        """Pooled block-bootstrap LCB per candidate — float64 ``[n_candidates]``."""
        from src.validation import fold_scores_bootstrap_ci
        lcbs = []
        for fs in self._fold_scores_pop(params_list, regularize=True):
            if not fs:
                lcbs.append(0.0)
                continue
            lcbs.append(float(fold_scores_bootstrap_ci(
                fs, n_boot=self.n_boot, alpha=self.alpha,
                block_len=self.block_len)[0]))
        return np.asarray(lcbs, dtype=np.float64)

    def score(self, params: Params) -> float:
        """Single-candidate LCB (PooledScorer-compatible API)."""
        return float(self.score_pop([params])[0])

    def fold_scores(self, params: Params) -> list[float]:
        """RAW (unpenalized) per-fold scores — for the acceptance gates."""
        return self._fold_scores_pop([params], regularize=False)[0]
