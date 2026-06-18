"""Vote contribution analysis — what does each VOTE contribute at fire bars?

Reconstructs each individual vote bit per bar for both sides by copying the
EXACT formulas from src/detector.py (Phase 1 vectorized predicates, lines
204-330, 386-417; drift vote L507-518; vote count L575-594), anchored by a
bar-exact parity assert against the exported ph_confirms / pl_confirms.

Vote sources (src/detector.py line refs):
  trend     L204-234  slope_val & linreg_norm both beyond +/-slope_thresh
  drift     L507/512  HIGH: drift < -thresh (drift_down_high); LOW: drift > thresh
  volume    L386-388  HIGH: vol_surge < 1/thresh (slow);  L414-416 LOW: > thresh
  momentum  L250-269, L389/417  mom_diverge < 0 (price_ret * vol_ret)
  mom_vel   L271-284  Reversal: HIGH <= -|thresh|, LOW >= |thresh|
  vola      L677-695  pir_of(atr(df, S_detect), range_len) > vola_high_pct
  gjr       L324-326  HIGH: gjr_asym_norm <= -thresh; LOW: >= thresh
  har       L328-330  har_vol_norm >= thresh (both sides)
Edge voting is OFF in both winners -> effective votes == plain votes.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import logging

logging.disable(logging.INFO)

import numpy as np
import pandas as pd

from src.detector import SpeculatorDetector, build_detector_artifacts
from src.indicators import Params, atr, linreg_slope_step, nz, pir_of, sma
from src.pooled_validation import load_stream_frame
from src.universe import resolve_streams

SWING_W, DROP_ATR, TOL = 20, 2.0, 2
WARMUP = 300
FX_TICKERS = ("EURGBP", "EURJPY", "EURUSD", "GBPUSD", "USDCHF", "USDRUB", "USDSEK")
VOTES = ("trend", "drift", "volume", "momentum", "mom_vel", "vola", "gjr", "har")

HIGH_P = dict(S_detect_high=12, scale_start_high=3, scale_end_high=270, scale_step_high=3,
 min_duration_high=13, cooldown_bars_high=9, price_gate_lb_high=23, vola_range_len_high=120,
 er_period_high=28, pct_extreme_high=0.3989736829331265, min_agreement_high=0.5902329742597521,
 dur_extreme_pct_high=0.10034295109907004, confirm_count_high=5, vol_surge_thresh_high=1.002017461803299,
 scale_div_thresh_high=0.599107129205612, slope_thresh_high=0.014515054272148754,
 vola_high_pct_high=0.5027540174875171, pivot_drift_lookback_high=5,
 pivot_drift_thresh_high=0.033318432240576665, pivot_drift_gate_mult_high=5.9550694191377165,
 pivot_drift_confirm_bias_high=1, momentum_velocity_thresh_high=1.3548394099613192e-08,
 er_directional_high=False, use_trend_high=True, use_volume_high=True, use_momentum_high=True,
 use_momentum_velocity_high=True, use_volatility_high=True, use_er_gate_high=False,
 use_gjr_asym_high=True, use_har_vol_high=True, gjr_vote_thresh_high=0.05168051407941195,
 har_vote_thresh_high=0.05605184243891868, vola_method_high="ATR",
 momentum_velocity_mode_high="Reversal", use_edge_voting_high=False, edge_window_high=5,
 S_detect_low=26, scale_start_low=19, scale_end_low=250, scale_step_low=15, min_duration_low=2,
 cooldown_bars_low=7, price_gate_lb_low=58, vola_range_len_low=20, er_period_low=47,
 pct_extreme_low=0.85, min_agreement_low=0.3, dur_extreme_pct_low=0.72, confirm_count_low=3,
 vol_surge_thresh_low=2.2, scale_div_thresh_low=0.39, slope_thresh_low=0.15, vola_high_pct_low=0.78,
 pivot_drift_lookback_low=10, pivot_drift_thresh_low=0.005, pivot_drift_gate_mult_low=4.0,
 pivot_drift_confirm_bias_low=0, momentum_velocity_thresh_low=0.007, er_directional_low=False,
 use_trend_low=True, use_volume_low=True, use_momentum_low=False, use_momentum_velocity_low=True,
 use_volatility_low=True, use_er_gate_low=False, use_gjr_asym_low=True, use_har_vol_low=True,
 gjr_vote_thresh_low=0.15, har_vote_thresh_low=0.15, vola_method_low="ATR",
 momentum_velocity_mode_low="Reversal", use_edge_voting_low=False, edge_window_low=5)
LOW_P = dict(HIGH_P)
LOW_P.update(dict(pct_extreme_high=0.96, min_agreement_high=0.75, dur_extreme_pct_high=0.83,
 scale_div_thresh_high=0.35, slope_thresh_high=0.22, vola_high_pct_high=0.92,
 pivot_drift_thresh_high=0.008, pivot_drift_gate_mult_high=8.0, momentum_velocity_thresh_high=0.014,
 vol_surge_thresh_high=1.5, use_momentum_high=False, gjr_vote_thresh_high=0.15, har_vote_thresh_high=0.15,
 pct_extreme_low=0.49268731212549655, min_agreement_low=0.02065350717621217,
 dur_extreme_pct_low=0.7096091393768227, scale_div_thresh_low=0.7358944084387257,
 slope_thresh_low=0.4025458749698267, vol_surge_thresh_low=2.2382428216185266,
 vola_high_pct_low=0.5571651880232253, pivot_drift_thresh_low=0.030211982920262805,
 momentum_velocity_thresh_low=0.003987965341605985, gjr_vote_thresh_low=0.3614954474374879,
 har_vote_thresh_low=0.11076327581817122))


def wilder_atr(df: pd.DataFrame, n: int = 14) -> np.ndarray:
    h, l, c = (df[k].to_numpy(float) for k in ("high", "low", "close"))
    tr = np.maximum(h[1:] - l[1:], np.maximum(np.abs(h[1:] - c[:-1]), np.abs(l[1:] - c[:-1])))
    a = pd.Series(np.concatenate([[h[0] - l[0]], tr])).ewm(alpha=1 / n, adjust=False).mean().to_numpy()
    a[a == 0] = np.nan
    return a


def real_turns(df: pd.DataFrame, kind: str, atr_arr: np.ndarray) -> np.ndarray:
    h, l = df["high"].to_numpy(float), df["low"].to_numpy(float)
    n = len(df)
    out = np.zeros(n, bool)
    for t in range(SWING_W, n - SWING_W):
        if kind == "top" and h[t] == h[t - SWING_W:t + SWING_W + 1].max() and \
           (h[t] - l[t + 1:t + SWING_W + 1].min()) >= DROP_ATR * atr_arr[t]:
            out[t] = True
        elif kind == "bot" and l[t] == l[t - SWING_W:t + SWING_W + 1].min() and \
             (h[t + 1:t + SWING_W + 1].max() - l[t]) >= DROP_ATR * atr_arr[t]:
            out[t] = True
    return out


def reconstruct_votes(df: pd.DataFrame, p: Params, side: str, det: pd.DataFrame,
                      art) -> dict:
    """Reconstruct the 8 raw vote bits per bar — exact detector.py formulas."""
    close, volume = df["close"], df["volume"]
    g = lambda k: getattr(p, f"{k}_{side}")
    S = g("S_detect")

    # --- trend (detector.py L204-234) ---
    slope_delta = max(round(S / 4), 2)
    sma_d = sma(close, S)
    slope_val = (nz(sma_d - sma_d.shift(slope_delta), 0.0)
                 / (slope_delta * sma_d.clip(lower=1e-9)) * 1000)
    linreg_norm = linreg_slope_step(close, S) / sma_d.clip(lower=1e-9) * 1000
    thr = g("slope_thresh")
    if side == "high":
        trend = ((slope_val > thr) & (linreg_norm > thr)).to_numpy()
    else:
        trend = ((slope_val < -thr) & (linreg_norm < -thr)).to_numpy()

    # --- volume surge (L236-248, L386-388 / L414-416) ---
    vol_fast_len = max(round(S / 4), 2)
    vol_slow = sma(volume, S)
    vol_surge = (sma(volume, vol_fast_len) / vol_slow.where(vol_slow != 0)).values.astype(float)
    if side == "high":
        vol_bit = vol_surge < (1.0 / g("vol_surge_thresh"))
    else:
        vol_bit = vol_surge > g("vol_surge_thresh")

    # --- momentum (L250-269, mom_diverge < 0) ---
    price_ret = (close - close.shift(S)) / close.shift(S).clip(lower=1e-9)
    vol_ret = (volume - volume.shift(S)) / volume.shift(S).clip(lower=1)
    mom_bit = (price_ret * vol_ret).values.astype(float) < 0

    # --- momentum velocity (L271-284, mode=Reversal both sides) ---
    mom_vel = price_ret - price_ret.shift(1)
    mvthr = abs(g("momentum_velocity_thresh"))
    if side == "high":
        mv_bit = (mom_vel <= -mvthr).to_numpy()
    else:
        mv_bit = (mom_vel >= mvthr).to_numpy()

    # --- volatility (L677-695, method=ATR) ---
    raw = atr(df, S)
    vola_bit = (pir_of(raw, g("vola_range_len")) > g("vola_high_pct")).to_numpy()

    # --- gjr / har (L324-330, artifacts norms) ---
    gjr_norm = art.gjr_asym_norm.values.astype(float)
    har_norm = art.har_vol_norm.values.astype(float)
    if side == "high":
        gjr_bit = gjr_norm <= -g("gjr_vote_thresh")
    else:
        gjr_bit = gjr_norm >= g("gjr_vote_thresh")
    har_bit = har_norm >= g("har_vote_thresh")

    # --- pivot drift (L507/512: HIGH drift_down = drift < -thresh; LOW > thresh)
    drift = det[f"pivot_drift_{side}"].to_numpy(float)
    if side == "high":
        drift_bit = drift < -g("pivot_drift_thresh")
    else:
        drift_bit = drift > g("pivot_drift_thresh")

    return dict(trend=trend, drift=drift_bit, volume=vol_bit, momentum=mom_bit,
                mom_vel=mv_bit, vola=vola_bit, gjr=gjr_bit, har=har_bit)


def use_flags(p: Params, side: str) -> dict:
    g = lambda k: getattr(p, f"{k}_{side}")
    return dict(trend=g("use_trend"), drift=True,  # _USE_PIVOT_DRIFT fixed True
                volume=g("use_volume"), momentum=g("use_momentum"),
                mom_vel=g("use_momentum_velocity"), vola=g("use_volatility"),
                gjr=g("use_gjr_asym"), har=g("use_har_vol"))


def required_votes(p: Params, side: str, drift: np.ndarray) -> np.ndarray:
    """Per-bar required vote count (detector.py L598-616)."""
    g = lambda k: getattr(p, f"{k}_{side}")
    max_votes = int(sum([g("use_trend"), g("use_volume"), g("use_momentum"),
                         g("use_momentum_velocity"), g("use_volatility"),
                         g("use_gjr_asym"), g("use_har_vol")]))
    mv = max(max_votes, 1)
    drift_up = drift > g("pivot_drift_thresh")
    if side == "high":
        bias = (drift_up & bool(g("pivot_drift_confirm_bias"))).astype(int)
        req = g("confirm_count") + bias
    else:
        bias = (drift_up & bool(g("pivot_drift_confirm_bias"))).astype(int)
        req = g("confirm_count") - bias
    return np.maximum(1, np.minimum(mv, req))


def main() -> None:
    streams = resolve_streams(["INDICES", "COMMODITIES", "FX", "WORLD_ETF"],
                              ["1D"], str(ROOT / "data" / "raw_v16"))
    sides = {"high": Params(**HIGH_P), "low": Params(**LOW_P)}
    # pooled accumulators: per side, per vote -> masks collected across streams
    pool = {s: {v: dict(bits=[], fire=[], lb=[], hit=[], fx=[]) for v in VOTES}
            for s in sides}
    meta = {s: dict(n_bars=0, n_fires=0, n_marginal=0, fire_hits=[]) for s in sides}

    for st in streams:
        df = load_stream_frame(st.path)
        n = len(df)
        art = build_detector_artifacts(df)
        atr14 = wilder_atr(df)
        is_fx = any(t in st.stream_id for t in FX_TICKERS)
        for side, kind in (("high", "top"), ("low", "bot")):
            p = sides[side]
            det = SpeculatorDetector(df, p, artifacts=art, include_debug_columns=True).run()
            bits = reconstruct_votes(df, p, side, det, art)
            uf = use_flags(p, side)
            conf = det["ph_confirms" if side == "high" else "pl_confirms"].to_numpy(float)

            # ---- CORRECTNESS ANCHOR: bar-exact parity vs exported confirms ----
            recon = np.zeros(n, dtype=int)
            for v in VOTES:
                if uf[v]:
                    recon += bits[v].astype(int)
            mask = np.arange(n) >= WARMUP
            if not np.array_equal(recon[mask], conf[mask].astype(int)):
                bad = np.where(mask & (recon != conf.astype(int)))[0]
                t0 = bad[0]
                detail = {v: bool(bits[v][t0]) for v in VOTES}
                raise AssertionError(
                    f"PARITY FAIL {st.stream_id} {side}: bar {t0} recon={recon[t0]} "
                    f"exported={conf[t0]:.0f} bits={detail} ({len(bad)} mismatching bars)")

            sig = det[f"signal_{side}"].to_numpy().astype(bool)
            drift = det[f"pivot_drift_{side}"].to_numpy(float)
            req = required_votes(p, side, drift)
            turn = real_turns(df, kind, atr14)
            # hit per bar: real turn within +/-TOL
            turn_near = np.zeros(n, bool)
            for i in np.where(turn)[0]:
                turn_near[max(0, i - TOL):i + TOL + 1] = True
            valid = mask & (np.arange(n) < n - SWING_W)
            fire = sig & valid
            marginal = fire & (conf.astype(int) == req)

            meta[side]["n_bars"] += int(valid.sum())
            meta[side]["n_fires"] += int(fire.sum())
            meta[side]["n_marginal"] += int(marginal.sum())
            meta[side]["fire_hits"].append(turn_near[fire])
            for v in VOTES:
                b = bits[v]
                pool[side][v]["bits"].append(b[valid])
                pool[side][v]["fx"].append(np.full(int(valid.sum()), is_fx))
                pool[side][v]["fire"].append(b[fire])
                pool[side][v]["lb"].append((b & marginal)[fire])
                pool[side][v]["hit"].append(np.column_stack([b[fire], turn_near[fire]]))
        print(f"{st.stream_id} parity OK (both sides)", flush=True)

    print(f"\nPARITY ANCHOR PASSED: {len(streams)} streams x 2 sides, bar-exact "
          f"(bars >= {WARMUP})")

    for side in ("high", "low"):
        p = sides[side]
        uf = use_flags(p, side)
        m = meta[side]
        fh = np.concatenate(m["fire_hits"])
        print(f"\n=== {side.upper()} side  (Params: {'HIGH_P/all_on' if side == 'high' else 'LOW_P/minus_momentum'})"
              f"  bars={m['n_bars']}  fires={m['n_fires']}"
              f"  marginal(conf==req)={m['n_marginal']} ({m['n_marginal']/max(m['n_fires'],1):.1%})"
              f"  fire hit-rate={fh.mean():.1%} ===")
        hdr = (f"{'vote':<9}{'on':<4}{'base':>7}{'baseFX':>8}{'baseOth':>9}"
               f"{'fire':>7}{'loadbear':>9}{'hit|T':>8}{'hit|F':>8}{'nT':>7}{'nF':>7}")
        print(hdr)
        print("-" * len(hdr))
        for v in VOTES:
            d = pool[side][v]
            bits = np.concatenate(d["bits"])
            fx = np.concatenate(d["fx"])
            fire_b = np.concatenate(d["fire"])
            lb = np.concatenate(d["lb"])
            ht = np.vstack(d["hit"])  # cols: vote_bit, hit
            vt, hh = ht[:, 0].astype(bool), ht[:, 1].astype(bool)
            n_t, n_f = int(vt.sum()), int((~vt).sum())
            hit_t = hh[vt].mean() if n_t else float("nan")
            hit_f = hh[~vt].mean() if n_f else float("nan")
            print(f"{v:<9}{('Y' if uf[v] else 'n'):<4}"
                  f"{bits.mean():>7.3f}"
                  f"{(bits[fx].mean() if fx.any() else float('nan')):>8.3f}"
                  f"{(bits[~fx].mean() if (~fx).any() else float('nan')):>9.3f}"
                  f"{(fire_b.mean() if len(fire_b) else float('nan')):>7.3f}"
                  f"{(lb.mean() if len(lb) else float('nan')):>9.3f}"
                  f"{hit_t:>8.3f}{hit_f:>8.3f}{n_t:>7}{n_f:>7}")
        print("  base=P(vote|all bars)  fire=P(vote|fire)  loadbear=P(vote & conf==req|fire)"
              "\n  hit|T / hit|F = fire hit-rate (turn within +/-2) given vote true/false")


if __name__ == "__main__":
    main()
