"""Vol-onset top-confirmation test (PRE-REGISTERED).

Founding mechanism: the top of a swing is marked by the TRANSITION from low
volatility to high volatility (a z-score crossing event), not a vol STATE.
Framed as a top-CONFIRMATION trigger: vol wakes up AFTER the top prints, and
tops pay at LONG holds (~250-350 bars), so a late confirmation can still
capture the decline.

Registered rules (fixed before results; all cells reported):
  z[t] = (rv20[t] - mean250(rv20)) / std250(rv20), rv20 = trailing 20-bar std
  of daily log-returns. All trailing/causal.
  ONSET: z crosses above 0 from below AND median(z[t-60:t]) < -0.25,
  20-bar refractory.
  Bottom mirror (reported only): z[t] > +1 AND z[t] < max(z[t-10:t]) - 1.0,
  20-bar refractory (vol-exhaustion rollover).
  Ground-truth tops: high == max over +/-20 AND drop >= 2*ATR14(Wilder)
  within 20 bars (miss_autopsy pattern).
  Metrics: (a) precision of a real top within last k bars (k=5,10,20) vs base
  rate; (b) forward SHORT log-return at h=20/40/100/250 vs all-bars baseline,
  cluster bootstrap by stream (2000); (c) events per stream.
  PASS: precision(k=10) >= 2x base rate (p<0.05) AND short payoff > baseline
  at h=100 or h=250 (p<0.05). PARTIAL: exactly one. GBM null: same pipeline.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import logging

logging.disable(logging.INFO)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.pooled_validation import load_stream_frame
from src.universe import resolve_streams

SWING_W, DROP_ATR = 20, 2.0
KS = (5, 10, 20)
HS = (20, 40, 100, 250)
N_BOOT = 2000
RNG_BOOT = np.random.default_rng(42)


# ---------------------------------------------------------------- primitives
def wilder_atr(df: pd.DataFrame, n: int = 14) -> np.ndarray:
    h, l, c = (df[k].to_numpy(float) for k in ("high", "low", "close"))
    tr = np.maximum(h[1:] - l[1:],
                    np.maximum(np.abs(h[1:] - c[:-1]), np.abs(l[1:] - c[:-1])))
    a = pd.Series(np.concatenate([[h[0] - l[0]], tr])).ewm(
        alpha=1 / n, adjust=False).mean().to_numpy()
    a[a == 0] = np.nan
    return a


def real_turns(df: pd.DataFrame, kind: str, atr: np.ndarray) -> np.ndarray:
    h, l = df["high"].to_numpy(float), df["low"].to_numpy(float)
    n = len(df)
    out = np.zeros(n, bool)
    for t in range(SWING_W, n - SWING_W):
        if kind == "top" and h[t] == h[t - SWING_W:t + SWING_W + 1].max() and \
           (h[t] - l[t + 1:t + SWING_W + 1].min()) >= DROP_ATR * atr[t]:
            out[t] = True
        elif kind == "bot" and l[t] == l[t - SWING_W:t + SWING_W + 1].min() and \
             (h[t + 1:t + SWING_W + 1].max() - l[t]) >= DROP_ATR * atr[t]:
            out[t] = True
    return out


def vol_z(close: np.ndarray) -> np.ndarray:
    """Causal vol z-score: trailing rv20 standardized by trailing 250-bar stats."""
    r = pd.Series(np.log(close)).diff()
    rv20 = r.rolling(20).std()
    mu = rv20.rolling(250).mean()
    sd = rv20.rolling(250).std()
    z = (rv20 - mu) / sd.replace(0.0, np.nan)
    return z.to_numpy(float)


def onset_events(z: np.ndarray) -> np.ndarray:
    """ONSET: z crosses above 0 from below, prior-60 median < -0.25, 20-bar refractory."""
    n = len(z)
    med60 = pd.Series(z).rolling(60).median().to_numpy(float)  # window t-59..t
    out = []
    last = -10**9
    for t in range(61, n):
        if not (np.isfinite(z[t]) and np.isfinite(z[t - 1])):
            continue
        if z[t - 1] < 0.0 <= z[t] and np.isfinite(med60[t - 1]) \
                and med60[t - 1] < -0.25 and t - last > 20:
            out.append(t)
            last = t
    return np.asarray(out, int)


def bottom_events(z: np.ndarray) -> np.ndarray:
    """Mirror (reported only): vol-exhaustion rollover, 20-bar refractory."""
    n = len(z)
    out = []
    last = -10**9
    for t in range(11, n):
        w = z[t - 10:t]
        if not (np.isfinite(z[t]) and np.all(np.isfinite(w))):
            continue
        if z[t] > 1.0 and z[t] < w.max() - 1.0 and t - last > 20:
            out.append(t)
            last = t
    return np.asarray(out, int)


def eligible_mask(z: np.ndarray) -> np.ndarray:
    """Bars where the onset rule is evaluable: z finite at t,t-1 and prior-60 median finite."""
    n = len(z)
    med60 = pd.Series(z).rolling(60).median().to_numpy(float)
    m = np.zeros(n, bool)
    fin = np.isfinite(z)
    m[61:] = fin[61:] & fin[60:-1] & np.isfinite(med60[60:-1])
    return m


def top_in_last_k(top: np.ndarray, k: int) -> np.ndarray:
    """Bool per bar t: any real top at u in [t-k, t]."""
    return pd.Series(top.astype(float)).rolling(k + 1, min_periods=1) \
        .max().to_numpy() > 0


def gbm_frame(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Vol-matched GBM null: same length, same unconditional daily mu/sigma,
    no vol clustering. high/low = max/min(open, close) (close-path geometry)."""
    c = df["close"].to_numpy(float)
    r = np.diff(np.log(c))
    rng = np.random.default_rng(seed)
    sim_r = rng.normal(np.nanmean(r), np.nanstd(r), size=len(c) - 1)
    sc = np.exp(np.concatenate([[np.log(c[0])], np.log(c[0]) + np.cumsum(sim_r)]))
    op = np.concatenate([[sc[0]], sc[:-1]])
    return pd.DataFrame({"open": op, "close": sc,
                         "high": np.maximum(op, sc), "low": np.minimum(op, sc)})


# ------------------------------------------------------------ per-stream pass
def analyze_stream(df: pd.DataFrame, stream_id: str) -> dict:
    n = len(df)
    close = df["close"].to_numpy(float)
    logc = np.log(close)
    atr = wilder_atr(df)
    top = real_turns(df, "top", atr)
    bot = real_turns(df, "bot", atr)
    z = vol_z(close)
    ev = onset_events(z)
    bev = bottom_events(z)
    elig = eligible_mask(z)
    label_ok = np.zeros(n, bool)
    label_ok[:n - SWING_W] = True          # top labels valid for u < n-SWING_W

    s = {"stream": stream_id, "n": n, "n_events": len(ev), "n_bot_events": len(bev)}

    # (a) precision sums (events / eligible bars with valid labels)
    for k in KS:
        lk = top_in_last_k(top, k)
        ev_ok = ev[label_ok[ev]]
        bars = elig & label_ok
        s[f"ev_n_k{k}"] = len(ev_ok)
        s[f"ev_hit_k{k}"] = int(lk[ev_ok].sum())
        s[f"base_n_k{k}"] = int(bars.sum())
        s[f"base_hit_k{k}"] = int(lk[bars].sum())
        # bottoms mirror (long side): real bottom within last k bars
        lkb = top_in_last_k(bot, k)
        bev_ok = bev[label_ok[bev]]
        s[f"bev_n_k{k}"] = len(bev_ok)
        s[f"bev_hit_k{k}"] = int(lkb[bev_ok].sum())
        s[f"bbase_hit_k{k}"] = int(lkb[bars].sum())

    # (b) forward short payoff sums
    for h in HS:
        fwd = np.full(n, np.nan)
        fwd[:n - h] = -(logc[h:] - logc[:-h])       # short log-return
        ev_ok = ev[ev < n - h]
        s[f"ev_pay_n_h{h}"] = len(ev_ok)
        s[f"ev_pay_sum_h{h}"] = float(np.nansum(fwd[ev_ok]))
        bars = elig.copy()
        bars[n - h:] = False
        s[f"base_pay_n_h{h}"] = int(bars.sum())
        s[f"base_pay_sum_h{h}"] = float(np.nansum(fwd[bars]))
        bev_ok = bev[bev < n - h]
        s[f"bev_pay_n_h{h}"] = len(bev_ok)
        s[f"bev_pay_sum_h{h}"] = float(-np.nansum(fwd[bev_ok]))  # long payoff

    # event-study paths (short cum log-ret, -20..+250 around onset)
    H = 250
    plen = 20 + H + 1                              # t-20 .. t+250
    paths = []
    for t in ev:
        p = np.full(plen, np.nan)
        lo, hi = max(0, t - 20), min(n, t + H + 1)
        p[lo - (t - 20):hi - (t - 20)] = -(logc[lo:hi] - logc[t])
        paths.append(p)
    s["paths"] = np.array(paths) if paths else np.empty((0, plen))
    # baseline paths: strided eligible bars (~500/stream)
    idx = np.where(elig)[0]
    idx = idx[::max(1, len(idx) // 500)]
    bpaths = []
    for t in idx:
        p = np.full(plen, np.nan)
        lo, hi = max(0, t - 20), min(n, t + H + 1)
        p[lo - (t - 20):hi - (t - 20)] = -(logc[lo:hi] - logc[t])
        bpaths.append(p)
    s["bpaths"] = np.array(bpaths)
    return s


# ------------------------------------------------------------- pooled stats
def pooled_precision(stats: list[dict], k: int, prefix: str = "ev",
                     base_prefix: str = "base") -> tuple[float, float, float, float]:
    """Returns (precision, base_rate, p_vs_base, p_vs_2x) via cluster bootstrap."""
    ev_n = np.array([s[f"{prefix}_n_k{k}"] for s in stats], float)
    ev_h = np.array([s[f"{prefix}_hit_k{k}"] for s in stats], float)
    b_n = np.array([s[f"base_n_k{k}"] for s in stats], float)
    b_h = np.array([s[f"{base_prefix}_hit_k{k}"] for s in stats], float)
    prec = ev_h.sum() / max(ev_n.sum(), 1)
    base = b_h.sum() / max(b_n.sum(), 1)
    m = len(stats)
    picks = RNG_BOOT.integers(0, m, size=(N_BOOT, m))
    pe_n, pe_h = ev_n[picks].sum(1), ev_h[picks].sum(1)
    pb_n, pb_h = b_n[picks].sum(1), b_h[picks].sum(1)
    ok = pe_n > 0
    pr, br = pe_h[ok] / pe_n[ok], pb_h[ok] / np.maximum(pb_n[ok], 1)
    p_vs_base = float(np.mean(pr <= br))
    p_vs_2x = float(np.mean(pr < 2 * br))
    return prec, base, p_vs_base, p_vs_2x


def pooled_payoff(stats: list[dict], h: int, prefix: str = "ev"
                  ) -> tuple[float, float, float]:
    """Returns (event_mean, base_mean, p one-sided event>base) cluster bootstrap."""
    ev_n = np.array([s[f"{prefix}_pay_n_h{h}"] for s in stats], float)
    ev_s = np.array([s[f"{prefix}_pay_sum_h{h}"] for s in stats], float)
    b_n = np.array([s[f"base_pay_n_h{h}"] for s in stats], float)
    b_s = np.array([s[f"base_pay_sum_h{h}"] for s in stats], float)
    if prefix == "bev":  # bottoms are long; compare vs LONG baseline (=-short)
        b_s = -b_s
    em = ev_s.sum() / max(ev_n.sum(), 1)
    bm = b_s.sum() / max(b_n.sum(), 1)
    m = len(stats)
    picks = RNG_BOOT.integers(0, m, size=(N_BOOT, m))
    pe_n, pe_s = ev_n[picks].sum(1), ev_s[picks].sum(1)
    pb_n, pb_s = b_n[picks].sum(1), b_s[picks].sum(1)
    ok = pe_n > 0
    diff = pe_s[ok] / pe_n[ok] - pb_s[ok] / np.maximum(pb_n[ok], 1)
    return em, bm, float(np.mean(diff <= 0))


# -------------------------------------------------------------------- main
def main() -> None:
    streams = resolve_streams(["INDICES", "COMMODITIES", "FX", "WORLD_ETF"],
                              ["1D"], "data/raw_v16")
    real_stats, gbm_stats = [], []
    for i, st in enumerate(streams):
        df = load_stream_frame(st.path)
        real_stats.append(analyze_stream(df, st.stream_id))
        gbm_stats.append(analyze_stream(gbm_frame(df, seed=1000 + i),
                                        st.stream_id + "_GBM"))
        print(f"{st.stream_id}: n={len(df)} onsets={real_stats[-1]['n_events']} "
              f"gbm_onsets={gbm_stats[-1]['n_events']}", flush=True)

    print("\n=== (c) EVENT COUNTS PER STREAM (real | GBM null) ===")
    for rs, gs in zip(real_stats, gbm_stats):
        print(f"  {rs['stream']:<28s} bars={rs['n']:>6d}  onsets={rs['n_events']:>3d}"
              f"  bottoms={rs['n_bot_events']:>3d}  | gbm onsets={gs['n_events']:>3d}")
    tot_ev = sum(s["n_events"] for s in real_stats)
    print(f"  TOTAL onsets={tot_ev}  bottoms={sum(s['n_bot_events'] for s in real_stats)}"
          f"  gbm onsets={sum(s['n_events'] for s in gbm_stats)}")

    print("\n=== (a) RECENT-TOP CONFIRMATION PRECISION (top within last k bars) ===")
    print(f"{'k':>4s} {'set':>6s} {'precision':>10s} {'base_rate':>10s} "
          f"{'ratio':>7s} {'p>base':>8s} {'p>=2x':>8s}")
    prec_rows = {}
    for k in KS:
        for name, ss in (("real", real_stats), ("GBM", gbm_stats)):
            pr, br, p1, p2 = pooled_precision(ss, k)
            prec_rows[(name, k)] = (pr, br, p1, p2)
            print(f"{k:>4d} {name:>6s} {pr:>10.4f} {br:>10.4f} "
                  f"{pr / max(br, 1e-12):>7.2f} {p1:>8.4f} {p2:>8.4f}")

    print("\n=== (b) FORWARD SHORT PAYOFF (log-return, onset vs all-bars) ===")
    print(f"{'h':>4s} {'set':>6s} {'event_mean':>11s} {'base_mean':>11s} "
          f"{'diff':>9s} {'p(one-sided)':>13s}")
    pay_rows = {}
    for h in HS:
        for name, ss in (("real", real_stats), ("GBM", gbm_stats)):
            em, bm, p = pooled_payoff(ss, h)
            pay_rows[(name, h)] = (em, bm, p)
            print(f"{h:>4d} {name:>6s} {em:>11.5f} {bm:>11.5f} "
                  f"{em - bm:>9.5f} {p:>13.4f}")

    print("\n=== BOTTOM-TRIGGER MIRROR (reported only, long side) ===")
    for k in KS:
        pr, br, p1, p2 = pooled_precision(real_stats, k, "bev", "bbase")
        print(f"  k={k:<3d} precision={pr:.4f} base={br:.4f} "
              f"ratio={pr / max(br, 1e-12):.2f} p>base={p1:.4f}")
    for h in HS:
        em, bm, p = pooled_payoff(real_stats, h, "bev")
        print(f"  h={h:<4d} long event_mean={em:+.5f} base={bm:+.5f} p={p:.4f}")

    # registered verdict
    _, _, _, p2x_10 = prec_rows[("real", 10)]
    pr10, br10 = prec_rows[("real", 10)][:2]
    cond_i = pr10 >= 2 * br10 and p2x_10 < 0.05
    cond_ii = any(pay_rows[("real", h)][0] > pay_rows[("real", h)][1]
                  and pay_rows[("real", h)][2] < 0.05 for h in (100, 250))
    verdict = "PASS" if (cond_i and cond_ii) else \
              "PARTIAL" if (cond_i or cond_ii) else "FAIL"
    print(f"\n=== REGISTERED VERDICT: {verdict} ===")
    print(f"  (i)  precision(k=10) >= 2x base w/ p<0.05: {cond_i} "
          f"(prec={pr10:.4f}, 2x base={2 * br10:.4f}, p={p2x_10:.4f})")
    print(f"  (ii) short payoff > baseline at h=100 or 250 w/ p<0.05: {cond_ii} "
          + " ".join(f"[h={h}: diff={pay_rows[('real', h)][0] - pay_rows[('real', h)][1]:+.5f}"
                     f" p={pay_rows[('real', h)][2]:.4f}]" for h in (100, 250)))

    # ------------------------------------------------------------- figure
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    x = np.arange(-20, 251)
    ep = np.vstack([s["paths"] for s in real_stats if len(s["paths"])])
    bp = np.vstack([s["bpaths"] for s in real_stats])
    gp = np.vstack([s["paths"] for s in gbm_stats if len(s["paths"])])
    ax = axes[0]
    ax.plot(x, np.nanmedian(ep, 0), color="crimson", lw=2,
            label=f"onset events (n={len(ep)})")
    ax.plot(x, np.nanmedian(bp, 0), color="gray", lw=1.5, ls="--",
            label="all-bars baseline")
    if len(gp):
        ax.plot(x, np.nanmedian(gp, 0), color="steelblue", lw=1.2, ls=":",
                label=f"GBM null onsets (n={len(gp)})")
    ax.axvline(0, color="k", lw=0.6)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xlabel("bars relative to onset")
    ax.set_ylabel("median SHORT cum log-return")
    ax.set_title("Event study: vol-onset short path")
    ax.legend(fontsize=8)
    ax = axes[1]
    w = 0.25
    xs = np.arange(len(KS))
    ax.bar(xs - w, [prec_rows[("real", k)][0] for k in KS], w,
           color="crimson", label="real precision")
    ax.bar(xs, [prec_rows[("real", k)][1] for k in KS], w,
           color="gray", label="base rate")
    ax.bar(xs + w, [prec_rows[("GBM", k)][0] for k in KS], w,
           color="steelblue", label="GBM null precision")
    ax.set_xticks(xs, [f"k={k}" for k in KS])
    ax.set_ylabel("P(real top within last k bars)")
    ax.set_title(f"Top-confirmation precision — verdict: {verdict}")
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = Path(__file__).with_suffix(".png")
    fig.savefig(out, dpi=130)
    print(f"\nfigure -> {out}")


if __name__ == "__main__":
    main()
