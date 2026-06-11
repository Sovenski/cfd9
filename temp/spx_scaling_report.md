# SPX Daily Log-Returns: Frequency/Scaling Structure

Data: SPX 1D, 1871-03-01 – 2026-05-28, n=25239 returns. σ=1.3067%/day, excess kurtosis=17.6.

## Plain-language summary (Gaussian noise or earthquake-like?)

1. SPX daily returns are NOT Gaussian noise — the magnitude–frequency plot bends like an earthquake plot, not like the Gaussian's parabolic collapse.
2. Hill tail exponent α ≈ 2.55 ± 0.10 (top 2.5% of |r|; range 2.36–3.23 over top 5%–1%) — the classic "cubic law of returns" (α≈3).
3. Both tails are fat: losses α = 2.53 ± 0.15, gains α = 2.53 ± 0.14 (losses slightly fatter).
4. That puts markets BETWEEN Gaussian (α = ∞, no tail) and earthquakes (Gutenberg–Richter b≈1 → energy α≈2/3, magnitude exceedance much fatter): power-law family, but with a steeper exponent than seismicity.
5. The 1987-size move (22.9%, 17.5σ on 1987-10-19) should occur once per ~10^66 YEARS under Gaussian; it happened 1 time(s) in 155 years. Moves >5σ: observed 135, Gaussian predicts 0.01.
6. Zipf rank–frequency slope of |r| ≈ -2.64 (vs -7.46 for matched Gaussian, which is not a power law at all) — consistent with α≈3 in the tail region.
7. Returns themselves are spectrally white (PSD slope β≈0.01) — no linear predictability, just like Gaussian noise.
8. But |returns| show 1/f^β long memory with β = 0.62: volatility is strongly autocorrelated (earthquake aftershock-like clustering, Omori-style).
9. ACF of |r| stays above 0.05 out to lag ≈ 250 days; the ACF of signed returns dies by lag 1.
10. Hurst (DFA-1): returns H = 0.53 (≈0.5, random walk), |returns| H = 0.90 (strong persistence; matched Gaussian gives 0.52). Verdict: amplitude process is earthquake-like; sign process is coin-flip-like.

## Key numbers

| Quantity | SPX | Matched Gaussian | Earthquakes (ref) |
|---|---|---|---|
| Hill α, \|r\| top 2.5% | 2.55 ± 0.10 (k=622) | ∞ | ~2 (b≈1) |
| Hill α, negative tail | 2.53 ± 0.15 (k=291) | ∞ | — |
| Hill α, positive tail | 2.53 ± 0.14 (k=330) | ∞ | — |
| Hill α range (top 5%→1%) | 2.36 → 3.23 | — | — |
| Zipf slope (rank 10–1000) | -2.64 | -7.46 | ~-1 to -2 |
| PSD β of returns | 0.01 (white) | 0 | — |
| PSD β of \|returns\| | 0.62 (long memory) | 0 | — |
| ACF<0.05 lag: \|r\| / r | 250 d / 1 d | 1 d / 1 d | — |
| Hurst H (DFA): r / \|r\| | 0.53 / 0.90 | 0.52 | — |
| 1987-size move (17.5σ) | 1× in 155 yr | 1 per 10^66 yr | — |
| >5σ days | 135 | 0.01 expected | — |
| EURUSD Hill α (universality) | 4.13 ± 0.22 (k=352) | — | — |

## What this means for a pivot detector

Extremes are vastly more common than Gaussian statistics admit: with α≈2.6, a 10σ
capitulation day is a once-per-few-decades event, not a once-per-universe-lifetime one, so
"impossible" washout/blowoff bars are a real, recurring feature the detector can key on.
The long memory in \|r\| (β≈0.6, H≈0.90, ACF persisting ~250 days) means
volatility regimes are forecastable: when a pivot zone forms, the elevated-volatility state
around it persists for months — features built on vol level/regime carry genuine signal.
But the SIGN process is white (β≈0.0, H≈0.53, ACF dead at lag 1): knowing
a storm is in progress says little about which way the next bar goes. That is exactly the
project's empirical finding — capitulation/volatility features separate pivot zones well,
while directional timing (especially tops, with few validatable events) stays near the
information-theoretic floor. The market is an earthquake catalog for magnitudes and a coin
flip for direction.


## Swing vocabulary: pivot-segmented analysis (figure 2: spx_swing_zipf.png)

Price segmented at confirmed structural pivots (pivot_high_pine | pivot_low_pine on
ohlc4) for n ∈ {20, 50, 100, 200}. Each segment between
consecutive pivots is a "word": amplitude = |Δ log ohlc4|, duration = bars.

| n | swings | Zipf slope (amp) | Zipf slope (dur) | Hill α amp (top 20%) | Hill α GBM null | KS real-vs-GBM | median ratio real/GBM |
|---|---|---|---|---|---|---|---|
| 20 | 737 | -1.50 | -1.80 | 2.06 ± 0.17 (k=147) | 2.97 | 0.177 (p=8.1e-11) | 0.81 |
| 50 | 288 | -1.46 | -1.57 | 2.39 ± 0.32 (k=57) | 3.28 | 0.173 (p=2.6e-04) | 0.78 |
| 100 | 124 | -1.43 | -1.49 | 2.42 ± 0.49 (k=24) | 2.92 | 0.147 (p=9.7e-02) | 0.91 |
| 200 | 61 | -1.35 | -1.29 | 3.89 ± 1.12 (k=12) | 4.10 | 0.143 (p=4.2e-01) | 1.09 |

Self-similarity collapse (amplitude CCDFs rescaled by median): max pairwise KS =
0.090 (worst pair n=50 vs n=200); all pairs: n=20/n=50: 0.057, n=20/n=100: 0.050, n=20/n=200: 0.081, n=50/n=100: 0.064, n=50/n=200: 0.090, n=100/n=200: 0.064.

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
