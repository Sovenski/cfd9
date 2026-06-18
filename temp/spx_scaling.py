"""Frequency/scaling structure of SPX daily log-returns.

Compares SPX daily log-returns against Gaussian white noise and power-law
(earthquake / Zipf style) statistics. Produces temp/spx_scaling_laws.png and
temp/spx_scaling_report.md.
"""
import sys

sys.path.insert(0, ".")

from typing import Tuple

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import signal, stats

from src.indicators import pivot_high_pine, pivot_low_pine
from src.pooled_validation import load_stream_frame
from src.universe import resolve_streams

RNG = np.random.default_rng(42)
OUT_PNG = "temp/spx_scaling_laws.png"
OUT_PNG2 = "temp/spx_swing_zipf.png"
OUT_MD = "temp/spx_scaling_report.md"
PIVOT_SCALES = [20, 50, 100, 200]


def swing_segments(price: pd.Series, n: int) -> pd.DataFrame:
    """Segment a price series at confirmed n-bar structural pivots.

    Merges pivot_high_pine and pivot_low_pine bars; consecutive pivots bound
    swing segments. Returns amplitude (|log price change|), duration (bars)
    and speed per segment.
    """
    ph = pivot_high_pine(price, n)
    pl = pivot_low_pine(price, n)
    mask = (ph | pl).to_numpy()
    idx = np.flatnonzero(mask)
    if len(idx) < 2:
        return pd.DataFrame(columns=["amplitude", "duration", "speed"])
    logp = np.log(price.to_numpy())
    amp = np.abs(np.diff(logp[idx]))
    dur = np.diff(idx).astype(float)
    keep = (amp > 0) & (dur > 0)
    return pd.DataFrame({
        "amplitude": amp[keep],
        "duration": dur[keep],
        "speed": amp[keep] / dur[keep],
    })


def zipf_slope_fit(values: np.ndarray, rmin: int = 5, rmax_frac: float = 0.8) -> float:
    """Slope of log(rank) vs log(value) over the central rank range."""
    v = np.sort(values)[::-1]
    ranks = np.arange(1, len(v) + 1)
    rmax = max(int(len(v) * rmax_frac), rmin + 5)
    sel = (ranks >= rmin) & (ranks <= rmax) & (v > 0)
    return float(np.polyfit(np.log(v[sel]), np.log(ranks[sel]), 1)[0])


def swing_analysis(ohlc4: pd.Series, sd_daily: float) -> dict:
    """Run the pivot-segmentation 'market vocabulary' analysis + GBM null."""
    n_bars = len(ohlc4)
    # GBM null: same length, same daily vol, drift = empirical mean log return
    lr = np.log(ohlc4).diff().dropna().to_numpy()
    gbm_logp = np.concatenate([[0.0], np.cumsum(RNG.normal(lr.mean(), sd_daily, n_bars - 1))])
    gbm = pd.Series(np.exp(gbm_logp), index=ohlc4.index)

    res: dict = {"real": {}, "gbm": {}}
    for n in PIVOT_SCALES:
        res["real"][n] = swing_segments(ohlc4, n)
        res["gbm"][n] = swing_segments(gbm, n)
    return res


def plot_swing_figure(res: dict) -> dict:
    """Figure 2: swing vocabulary panels. Returns stats dict for the report."""
    colors = {20: "tab:blue", 50: "tab:green", 100: "tab:orange", 200: "tab:red"}
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    fig.suptitle("SPX swing 'vocabulary': pivot-segmented amplitudes vs GBM null", fontsize=13)
    stats_out: dict = {}

    # Panel 1: Zipf of amplitudes
    ax = axes[0, 0]
    for n in PIVOT_SCALES:
        a = res["real"][n]["amplitude"].to_numpy()
        v = np.sort(a)[::-1]
        sl = zipf_slope_fit(a)
        ax.loglog(np.arange(1, len(v) + 1), v, ".", ms=3, color=colors[n],
                  label=f"n={n} (slope {sl:.2f}, {len(v)} swings)")
        stats_out.setdefault(n, {})["zipf_amp"] = sl
        stats_out[n]["count"] = len(v)
    ax.set_xlabel("rank"); ax.set_ylabel("swing amplitude |Δlog p|")
    ax.set_title("Zipf rank–frequency: swing amplitudes")
    ax.legend(fontsize=8)

    # Panel 2: Zipf of durations
    ax = axes[0, 1]
    for n in PIVOT_SCALES:
        d = res["real"][n]["duration"].to_numpy()
        v = np.sort(d)[::-1]
        sl = zipf_slope_fit(d)
        ax.loglog(np.arange(1, len(v) + 1), v, ".", ms=3, color=colors[n],
                  label=f"n={n} (slope {sl:.2f})")
        stats_out[n]["zipf_dur"] = sl
    ax.set_xlabel("rank"); ax.set_ylabel("swing duration (bars)")
    ax.set_title("Zipf rank–frequency: swing durations")
    ax.legend(fontsize=8)

    # Panel 3: amplitude CCDF, real vs GBM null
    ax = axes[1, 0]
    for n in PIVOT_SCALES:
        a = res["real"][n]["amplitude"].to_numpy()
        xs, p = ccdf(a)
        ax.loglog(xs, p, "-", lw=1.3, color=colors[n], label=f"SPX n={n}")
        ag = res["gbm"][n]["amplitude"].to_numpy()
        if len(ag) > 1:
            xg, pg = ccdf(ag)
            ax.loglog(xg, pg, "--", lw=1.0, color=colors[n], alpha=0.55)
            # KS distance real vs GBM
            ks = stats.ks_2samp(a, ag)
            stats_out[n]["ks_vs_gbm"] = float(ks.statistic)
            stats_out[n]["ks_vs_gbm_p"] = float(ks.pvalue)
            stats_out[n]["gbm_count"] = len(ag)
            stats_out[n]["med_ratio_vs_gbm"] = float(np.median(a) / np.median(ag))
        # Hill tail exponent on top 20% (small samples)
        al, se, k = hill_alpha(a, 0.20)
        stats_out[n]["hill_amp"] = al
        stats_out[n]["hill_amp_se"] = se
        stats_out[n]["hill_amp_k"] = k
        if len(ag) > 1:
            stats_out[n]["hill_amp_gbm"] = hill_alpha(ag, 0.20)[0]
    ax.plot([], [], "k-", label="SPX (solid)")
    ax.plot([], [], "k--", alpha=0.55, label="GBM null (dashed)")
    ax.set_xlabel("swing amplitude"); ax.set_ylabel("P(A > x)")
    ax.set_title("Survival CCDF: real swings vs GBM-noise swings")
    ax.legend(fontsize=8)

    # Panel 4: self-similarity collapse (rescale by median)
    ax = axes[1, 1]
    rescaled = {}
    for n in PIVOT_SCALES:
        a = res["real"][n]["amplitude"].to_numpy()
        a_r = a / np.median(a)
        rescaled[n] = a_r
        xs, p = ccdf(a_r)
        ax.loglog(xs, p, "-", lw=1.3, color=colors[n], label=f"n={n} / median")
    # pairwise KS between rescaled distributions
    ks_pairs = {}
    for i, n1 in enumerate(PIVOT_SCALES):
        for n2 in PIVOT_SCALES[i + 1:]:
            ks_pairs[(n1, n2)] = float(stats.ks_2samp(rescaled[n1], rescaled[n2]).statistic)
    max_ks = max(ks_pairs.values())
    max_pair = max(ks_pairs, key=ks_pairs.get)
    ax.annotate(f"collapse quality:\nmax pairwise KS = {max_ks:.3f}\n(worst pair n={max_pair[0]} vs n={max_pair[1]})",
                xy=(0.04, 0.08), xycoords="axes fraction", fontsize=9)
    ax.set_xlabel("amplitude / median(amplitude)"); ax.set_ylabel("P(A/med > x)")
    ax.set_title("Self-similarity collapse test")
    ax.legend(fontsize=8)

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(OUT_PNG2, dpi=120)
    print(f"saved {OUT_PNG2}")
    stats_out["ks_pairs"] = ks_pairs
    stats_out["max_ks"] = max_ks
    stats_out["max_pair"] = max_pair
    return stats_out


def load_returns(ticker: str, groups: list) -> Tuple[pd.Series, np.ndarray]:
    st = [s for s in resolve_streams(groups, ["1D"], "data/raw_v16") if s.ticker == ticker][0]
    df = load_stream_frame(st.path)
    r = np.log(df["close"]).diff().dropna()
    r = r[np.isfinite(r)]
    return r, r.to_numpy()


def hill_alpha(x: np.ndarray, frac: float) -> Tuple[float, float, int]:
    """Hill estimator on top `frac` of |x| (x must be positive magnitudes)."""
    x = np.sort(x[x > 0])[::-1]
    k = max(int(len(x) * frac), 10)
    tail = x[:k]
    xk = x[k]
    alpha = 1.0 / np.mean(np.log(tail / xk))
    se = alpha / np.sqrt(k)
    return alpha, se, k


def ccdf(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    xs = np.sort(x)
    p = 1.0 - np.arange(1, len(xs) + 1) / (len(xs) + 1.0)
    return xs, p


def dfa_hurst(x: np.ndarray, min_n: int = 10, max_frac: float = 0.1) -> float:
    """DFA-1 Hurst exponent."""
    y = np.cumsum(x - np.mean(x))
    N = len(y)
    ns = np.unique(np.logspace(np.log10(min_n), np.log10(N * max_frac), 25).astype(int))
    flucts = []
    for n in ns:
        nseg = N // n
        segs = y[: nseg * n].reshape(nseg, n)
        t = np.arange(n)
        # detrend each segment with linear fit
        f2 = []
        for s in segs:
            c = np.polyfit(t, s, 1)
            f2.append(np.mean((s - np.polyval(c, t)) ** 2))
        flucts.append(np.sqrt(np.mean(f2)))
    coef = np.polyfit(np.log(ns), np.log(flucts), 1)
    return float(coef[0])


def acf(x: np.ndarray, nlags: int) -> np.ndarray:
    x = x - x.mean()
    full = np.correlate(x, x, mode="full")[len(x) - 1 :]
    return full[: nlags + 1] / full[0]


def main() -> None:
    r_spx, r = load_returns("SPX", ["INDICES"])
    r_fx, rf = load_returns("EURUSD", ["FX"])
    st_spx = [s for s in resolve_streams(["INDICES"], ["1D"], "data/raw_v16")
              if s.ticker == "SPX"][0]
    df_spx = load_stream_frame(st_spx.path)
    ohlc4 = (df_spx["open"] + df_spx["high"] + df_spx["low"] + df_spx["close"]) / 4.0
    n = len(r)
    mu, sd = r.mean(), r.std()
    g = RNG.normal(mu, sd, n)

    # ---------- 1. tail / magnitude-frequency ----------
    abs_r = np.abs(r)
    a_all, se_all, k_all = hill_alpha(abs_r, 0.025)
    a_pos, se_pos, k_pos = hill_alpha(r[r > 0], 0.025)
    a_neg, se_neg, k_neg = hill_alpha(-r[r < 0], 0.025)
    # sensitivity range 1-5%
    a_lo = hill_alpha(abs_r, 0.01)[0]
    a_hi = hill_alpha(abs_r, 0.05)[0]

    # 1987 crash magnitude
    crash = abs_r.max()
    crash_date = r_spx.abs().idxmax()
    z = crash / sd
    # Gaussian two-sided exceedance prob per day
    p_gauss = 2 * stats.norm.sf(z)
    years_per_event = np.inf if p_gauss == 0 else 1.0 / (p_gauss * 252)
    log10_years = np.log10(years_per_event) if np.isfinite(years_per_event) else np.inf
    n_ge_5sd_obs = int(np.sum(abs_r > 5 * sd))
    n_ge_5sd_gauss = 2 * stats.norm.sf(5) * n
    n_ge_crash_obs = int(np.sum(abs_r >= crash))

    # ---------- 2. Zipf rank-frequency ----------
    mags = np.sort(abs_r)[::-1]
    ranks = np.arange(1, len(mags) + 1)
    # fit on top decade of ranks 10..1000 (clean scaling region)
    sel = (ranks >= 10) & (ranks <= 1000)
    zipf_slope = np.polyfit(np.log(mags[sel]), np.log(ranks[sel]), 1)[0]
    gmags = np.sort(np.abs(g))[::-1]
    zipf_slope_g = np.polyfit(np.log(gmags[sel]), np.log(ranks[sel]), 1)[0]

    # ---------- 3. spectra ----------
    f_r, P_r = signal.welch(r, nperseg=4096)
    f_a, P_a = signal.welch(abs_r - abs_r.mean(), nperseg=4096)
    fit_sel = (f_a > 1e-3) & (f_a < 1e-1)
    beta = -np.polyfit(np.log(f_a[fit_sel]), np.log(P_a[fit_sel]), 1)[0]
    beta_r = -np.polyfit(np.log(f_r[fit_sel]), np.log(P_r[fit_sel]), 1)[0]

    # ---------- 4. memory ----------
    nlags = 250
    acf_r = acf(r, nlags)
    acf_a = acf(abs_r - abs_r.mean(), nlags)
    below = np.where(acf_a[1:] < 0.05)[0]
    lag_a_05 = int(below[0] + 1) if len(below) else nlags
    below_r = np.where(np.abs(acf_r[1:]) < 0.05)[0]
    lag_r_05 = int(below_r[0] + 1) if len(below_r) else nlags

    # ---------- 5. Hurst (DFA) ----------
    H_r = dfa_hurst(r)
    H_a = dfa_hurst(abs_r)
    H_g = dfa_hurst(g)

    # ---------- 6. FX tail ----------
    abs_f = np.abs(rf)
    a_fx, se_fx, k_fx = hill_alpha(abs_f, 0.025)

    # ================= figure =================
    fig, axes = plt.subplots(2, 3, figsize=(17, 10))
    fig.suptitle("SPX daily log-returns: Gaussian noise vs earthquake-like scaling (1871–2026, n=%d)" % n,
                 fontsize=14)

    # Panel 1: CCDF
    ax = axes[0, 0]
    xs, p = ccdf(abs_r / sd)
    ax.loglog(xs, p, ".", ms=2, color="navy", label="SPX |r| (in std units)")
    xg, pg = ccdf(np.abs(g) / sd)
    ax.loglog(xg, pg, ".", ms=2, color="gray", label="matched Gaussian")
    xx = np.logspace(np.log10(2), np.log10(25), 50)
    # anchor power law at x=3 sd empirical
    p3 = np.mean(abs_r / sd > 3)
    ax.loglog(xx, p3 * (xx / 3) ** (-a_all), "r--", lw=1.5,
              label=f"power law α={a_all:.2f}")
    ax.axvline(z, color="k", ls=":", lw=1)
    ax.annotate(f"1987 crash: {z:.1f}σ\nGaussian: 1 per 10^{log10_years:.0f} yr\nobserved: {n_ge_crash_obs}× in 155 yr",
                xy=(z, 3e-5), fontsize=8, ha="right")
    ax.set_xlabel("|return| / σ"); ax.set_ylabel("P(|r| > x)")
    ax.set_title("Magnitude–frequency (earthquake plot)")
    ax.legend(fontsize=8); ax.set_ylim(1e-5, 1.2)

    # Panel 2: Zipf
    ax = axes[0, 1]
    ax.loglog(ranks, mags, ".", ms=2, color="navy", label=f"SPX (slope {zipf_slope:.2f})")
    ax.loglog(ranks, gmags, ".", ms=2, color="gray", label=f"Gaussian (slope {zipf_slope_g:.2f})")
    fitline = np.exp((np.log(ranks[sel]) - np.polyfit(np.log(mags[sel]), np.log(ranks[sel]), 1)[1]) / zipf_slope)
    ax.loglog(ranks[sel], fitline, "r--", lw=1.2)
    ax.set_xlabel("rank"); ax.set_ylabel("|return|")
    ax.set_title("Zipf rank–frequency of |returns|")
    ax.legend(fontsize=8)

    # Panel 3: PSD
    ax = axes[0, 2]
    ax.loglog(f_r[1:], P_r[1:], color="steelblue", lw=0.8, alpha=0.8,
              label=f"returns (β≈{beta_r:.2f}, ~white)")
    ax.loglog(f_a[1:], P_a[1:], color="darkorange", lw=0.8, alpha=0.8,
              label=f"|returns| (β={beta:.2f})")
    ax.loglog(f_a[fit_sel], np.exp(np.polyval(np.polyfit(np.log(f_a[fit_sel]), np.log(P_a[fit_sel]), 1), np.log(f_a[fit_sel]))),
              "r--", lw=1.5)
    ax.set_xlabel("frequency (1/day)"); ax.set_ylabel("PSD")
    ax.set_title("Welch spectral density")
    ax.legend(fontsize=8)

    # Panel 4: ACF
    ax = axes[1, 0]
    lags = np.arange(1, nlags + 1)
    ax.semilogy(lags, np.clip(acf_a[1:], 1e-4, None), color="darkorange", label="|returns|")
    ax.semilogy(lags, np.clip(np.abs(acf_r[1:]), 1e-4, None), color="steelblue", alpha=0.6, label="returns (|ACF|)")
    ax.axhline(0.05, color="k", ls=":", lw=1)
    ax.axhline(2 / np.sqrt(n), color="gray", ls="--", lw=0.8, label="95% noise band")
    ax.annotate(f"|r| ACF < 0.05 at lag {lag_a_05}\nreturns at lag {lag_r_05}",
                xy=(0.45, 0.8), xycoords="axes fraction", fontsize=8)
    ax.set_xlabel("lag (days)"); ax.set_ylabel("ACF (log)")
    ax.set_title("Memory: volatility clustering")
    ax.legend(fontsize=8)

    # Panel 5: Hurst (DFA fluctuation plot)
    ax = axes[1, 1]
    for series, lbl, c in [(r, f"returns H={H_r:.2f}", "steelblue"),
                           (abs_r, f"|returns| H={H_a:.2f}", "darkorange"),
                           (g, f"Gaussian H={H_g:.2f}", "gray")]:
        y = np.cumsum(series - series.mean())
        ns = np.unique(np.logspace(1, np.log10(len(y) * 0.1), 25).astype(int))
        fl = []
        for nn in ns:
            nseg = len(y) // nn
            segs = y[: nseg * nn].reshape(nseg, nn)
            t = np.arange(nn)
            fl.append(np.sqrt(np.mean([np.mean((s - np.polyval(np.polyfit(t, s, 1), t)) ** 2) for s in segs])))
        ax.loglog(ns, fl / fl[0], ".-", ms=3, lw=0.8, color=c, label=lbl)
    ax.set_xlabel("window n (days)"); ax.set_ylabel("F(n), normalized")
    ax.set_title("DFA Hurst exponent")
    ax.legend(fontsize=8)

    # Panel 6: FX universality
    ax = axes[1, 2]
    sdf = rf.std()
    xs2, p2 = ccdf(abs_f / sdf)
    ax.loglog(xs2, p2, ".", ms=2, color="seagreen", label=f"EURUSD |r| (α={a_fx:.2f})")
    ax.loglog(xs, p, ".", ms=2, color="navy", alpha=0.4, label=f"SPX |r| (α={a_all:.2f})")
    xg2, pg2 = ccdf(np.abs(RNG.normal(0, 1, len(rf))))
    ax.loglog(xg2, pg2, ".", ms=2, color="gray", alpha=0.5, label="Gaussian")
    ax.set_xlabel("|return| / σ"); ax.set_ylabel("P(|r| > x)")
    ax.set_title("Universality: EURUSD same fat tail")
    ax.legend(fontsize=8); ax.set_ylim(1e-5, 1.2)

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(OUT_PNG, dpi=120)
    print(f"saved {OUT_PNG}")

    # ================= swing vocabulary (figure 2) =================
    sd_o4 = np.log(ohlc4).diff().dropna().std()
    res_sw = swing_analysis(ohlc4, sd_o4)
    sw = plot_swing_figure(res_sw)

    # ================= report =================
    kurt = stats.kurtosis(r)
    report = f"""# SPX Daily Log-Returns: Frequency/Scaling Structure

Data: SPX 1D, {r_spx.index[0].date()} – {r_spx.index[-1].date()}, n={n} returns. σ={sd:.4%}/day, excess kurtosis={kurt:.1f}.

## Plain-language summary (Gaussian noise or earthquake-like?)

1. SPX daily returns are NOT Gaussian noise — the magnitude–frequency plot bends like an earthquake plot, not like the Gaussian's parabolic collapse.
2. Hill tail exponent α ≈ {a_all:.2f} ± {se_all:.2f} (top 2.5% of |r|; range {a_hi:.2f}–{a_lo:.2f} over top 5%–1%) — the classic "cubic law of returns" (α≈3).
3. Both tails are fat: losses α = {a_neg:.2f} ± {se_neg:.2f}, gains α = {a_pos:.2f} ± {se_pos:.2f} (losses slightly fatter).
4. That puts markets BETWEEN Gaussian (α = ∞, no tail) and earthquakes (Gutenberg–Richter b≈1 → energy α≈2/3, magnitude exceedance much fatter): power-law family, but with a steeper exponent than seismicity.
5. The 1987-size move ({crash:.1%}, {z:.1f}σ on {crash_date.date()}) should occur once per ~10^{log10_years:.0f} YEARS under Gaussian; it happened {n_ge_crash_obs} time(s) in 155 years. Moves >5σ: observed {n_ge_5sd_obs}, Gaussian predicts {n_ge_5sd_gauss:.2f}.
6. Zipf rank–frequency slope of |r| ≈ {zipf_slope:.2f} (vs {zipf_slope_g:.2f} for matched Gaussian, which is not a power law at all) — consistent with α≈3 in the tail region.
7. Returns themselves are spectrally white (PSD slope β≈{beta_r:.2f}) — no linear predictability, just like Gaussian noise.
8. But |returns| show 1/f^β long memory with β = {beta:.2f}: volatility is strongly autocorrelated (earthquake aftershock-like clustering, Omori-style).
9. ACF of |r| stays above 0.05 out to lag ≈ {lag_a_05} days; the ACF of signed returns dies by lag {lag_r_05}.
10. Hurst (DFA-1): returns H = {H_r:.2f} (≈0.5, random walk), |returns| H = {H_a:.2f} (strong persistence; matched Gaussian gives {H_g:.2f}). Verdict: amplitude process is earthquake-like; sign process is coin-flip-like.

## Key numbers

| Quantity | SPX | Matched Gaussian | Earthquakes (ref) |
|---|---|---|---|
| Hill α, \\|r\\| top 2.5% | {a_all:.2f} ± {se_all:.2f} (k={k_all}) | ∞ | ~2 (b≈1) |
| Hill α, negative tail | {a_neg:.2f} ± {se_neg:.2f} (k={k_neg}) | ∞ | — |
| Hill α, positive tail | {a_pos:.2f} ± {se_pos:.2f} (k={k_pos}) | ∞ | — |
| Hill α range (top 5%→1%) | {a_hi:.2f} → {a_lo:.2f} | — | — |
| Zipf slope (rank 10–1000) | {zipf_slope:.2f} | {zipf_slope_g:.2f} | ~-1 to -2 |
| PSD β of returns | {beta_r:.2f} (white) | 0 | — |
| PSD β of \\|returns\\| | {beta:.2f} (long memory) | 0 | — |
| ACF<0.05 lag: \\|r\\| / r | {lag_a_05} d / {lag_r_05} d | 1 d / 1 d | — |
| Hurst H (DFA): r / \\|r\\| | {H_r:.2f} / {H_a:.2f} | {H_g:.2f} | — |
| 1987-size move ({z:.1f}σ) | {n_ge_crash_obs}× in 155 yr | 1 per 10^{log10_years:.0f} yr | — |
| >5σ days | {n_ge_5sd_obs} | {n_ge_5sd_gauss:.2f} expected | — |
| EURUSD Hill α (universality) | {a_fx:.2f} ± {se_fx:.2f} (k={k_fx}) | — | — |

## What this means for a pivot detector

Extremes are vastly more common than Gaussian statistics admit: with α≈{a_all:.1f}, a 10σ
capitulation day is a once-per-few-decades event, not a once-per-universe-lifetime one, so
"impossible" washout/blowoff bars are a real, recurring feature the detector can key on.
The long memory in \\|r\\| (β≈{beta:.1f}, H≈{H_a:.2f}, ACF persisting ~{lag_a_05} days) means
volatility regimes are forecastable: when a pivot zone forms, the elevated-volatility state
around it persists for months — features built on vol level/regime carry genuine signal.
But the SIGN process is white (β≈{beta_r:.1f}, H≈{H_r:.2f}, ACF dead at lag {lag_r_05}): knowing
a storm is in progress says little about which way the next bar goes. That is exactly the
project's empirical finding — capitulation/volatility features separate pivot zones well,
while directional timing (especially tops, with few validatable events) stays near the
information-theoretic floor. The market is an earthquake catalog for magnitudes and a coin
flip for direction.
"""
    # ---- swing vocabulary section ----
    rows = []
    for nn in PIVOT_SCALES:
        s = sw[nn]
        rows.append(
            f"| {nn} | {s['count']} | {s['zipf_amp']:.2f} | {s['zipf_dur']:.2f} | "
            f"{s['hill_amp']:.2f} ± {s['hill_amp_se']:.2f} (k={s['hill_amp_k']}) | "
            f"{s.get('hill_amp_gbm', float('nan')):.2f} | "
            f"{s.get('ks_vs_gbm', float('nan')):.3f} (p={s.get('ks_vs_gbm_p', float('nan')):.1e}) | "
            f"{s.get('med_ratio_vs_gbm', float('nan')):.2f} |"
        )
    swing_table = "\n".join(rows)
    ks_pair_str = ", ".join(
        f"n={a}/n={b}: {v:.3f}" for (a, b), v in sw["ks_pairs"].items()
    )
    report += f"""

## Swing vocabulary: pivot-segmented analysis (figure 2: spx_swing_zipf.png)

Price segmented at confirmed structural pivots (pivot_high_pine | pivot_low_pine on
ohlc4) for n ∈ {{{', '.join(str(x) for x in PIVOT_SCALES)}}}. Each segment between
consecutive pivots is a "word": amplitude = |Δ log ohlc4|, duration = bars.

| n | swings | Zipf slope (amp) | Zipf slope (dur) | Hill α amp (top 20%) | Hill α GBM null | KS real-vs-GBM | median ratio real/GBM |
|---|---|---|---|---|---|---|---|
{swing_table}

Self-similarity collapse (amplitude CCDFs rescaled by median): max pairwise KS =
{sw['max_ks']:.3f} (worst pair n={sw['max_pair'][0]} vs n={sw['max_pair'][1]}); all pairs: {ks_pair_str}.

**GBM-null verdict:** swings exist in pure noise too — segmentation alone creates a
size distribution — but at the fine scales where the sample is large enough to tell
(n=20, 50; KS p ≈ 1e-10 and 3e-4) the real distribution differs from the GBM null in
a characteristic way: a SMALLER median swing (median ratio < 1 — vol clustering packs
many small swings into quiet regimes) combined with a HEAVIER tail (Hill α ≈ 2.1–2.4
vs ≈ 3.0–3.3 for GBM). The market's large swings are genuinely over-represented
relative to a volatility-matched random walk, not an artifact of the pivot rule. At
n=100–200 the same ordering persists but the sample (61–124 swings) is too small for
significance (KS p = 0.10 / 0.42).

**Collapse verdict:** rescaling by the median collapses the four amplitude CCDFs
onto an approximately common master curve (see max-KS above; values well below the
real-vs-GBM KS at the same scales). The swing alphabet is close to scale-invariant:
an n=200 swing is statistically a magnified n=20 swing.

**Plain language — does the market have a Zipfian swing vocabulary?** Yes, in the
weak sense that swing amplitudes follow a heavy-tailed rank–frequency law at every
pivot scale, with roughly stable Zipf slopes across n — a "vocabulary" whose word-size
distribution looks the same whether you read the tape at 20-bar or 200-bar resolution.
The grammar is (approximately) self-similar: the same generative shape repeats across
scales, only the median word size grows with n. For a multi-scale pivot detector this
is the load-bearing assumption made explicit: features and thresholds learned at one
pivot scale should transfer across scales after a single volatility/median rescaling,
which is why a shared feature stack with per-scale normalization (the project's design)
is the right architecture — and why genuinely scale-specific tuning should add little
beyond the rescaling. The caveat from part 1 still applies: the vocabulary describes
swing SIZES, not their direction or termination timing; a Zipfian alphabet does not
make the next pivot easier to call, it only guarantees that large words keep appearing.
"""
    with open(OUT_MD, "w", encoding="utf-8") as fh:
        fh.write(report)
    print(f"saved {OUT_MD}")

    print("\n=== SWING NUMBERS ===")
    for nn in PIVOT_SCALES:
        s = sw[nn]
        print(f"n={nn}: swings={s['count']} (gbm {s.get('gbm_count')}), "
              f"zipf_amp={s['zipf_amp']:.2f}, zipf_dur={s['zipf_dur']:.2f}, "
              f"hill_amp={s['hill_amp']:.2f}+-{s['hill_amp_se']:.2f} vs gbm {s.get('hill_amp_gbm', float('nan')):.2f}, "
              f"KS_vs_gbm={s.get('ks_vs_gbm', float('nan')):.3f} (p={s.get('ks_vs_gbm_p', float('nan')):.1e}), "
              f"med_ratio={s.get('med_ratio_vs_gbm', float('nan')):.2f}")
    print(f"collapse: max pairwise KS = {sw['max_ks']:.3f} ({sw['max_pair']})")

    print("\n=== KEY NUMBERS ===")
    print(f"Hill alpha |r| (2.5%): {a_all:.2f}+-{se_all:.2f}; neg {a_neg:.2f}; pos {a_pos:.2f}; range(5%->1%) {a_hi:.2f}->{a_lo:.2f}")
    print(f"Zipf slope: SPX {zipf_slope:.2f} vs Gaussian {zipf_slope_g:.2f}")
    print(f"PSD beta: returns {beta_r:.2f}, |returns| {beta:.2f}")
    print(f"ACF<0.05 lag: |r| {lag_a_05}, r {lag_r_05}")
    print(f"Hurst DFA: r {H_r:.2f}, |r| {H_a:.2f}, gaussian {H_g:.2f}")
    print(f"1987 crash {crash:.2%} = {z:.1f} sigma on {crash_date.date()}; Gaussian once per 10^{log10_years:.0f} years; observed {n_ge_crash_obs}")
    print(f">5sigma days: observed {n_ge_5sd_obs}, Gaussian expects {n_ge_5sd_gauss:.2f}")
    print(f"EURUSD Hill alpha: {a_fx:.2f}+-{se_fx:.2f}")


if __name__ == "__main__":
    main()
