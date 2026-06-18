# The 2×2 model & what we learned about markets — 2026-06-11

*One day of pre-registered tests on 155 years of SPX + a 16-stream daily pool.
Every claim below survived (or was killed by) a null model.*

## The 2×2 state model, in one paragraph

Every bar lives in one of four regimes, defined by two axes: **price direction**
(rising/falling) and **volatility direction** (rising/falling). The leverage
effect (falling prices pump volatility) makes two states the attractors:
**calm-bull** (price↑ vol↓) and **panic** (price↓ vol↑). In this language a
market bottom is the exit from panic, a top is the exit from calm-bull, and
pivot probability is the hazard of leaving the current state — ideally caught
not at the transition itself but as an *extremeness event inside the right
state*.

## What the tests said about that model

- **The state map is real — and free.** Tops cluster in up-states (×1.7),
  bottoms in panic (×2.0), exactly as drawn. But a volatility-matched random
  walk reproduces the same map almost number-for-number: the map is mostly
  *label geometry* (a swing top is near a local high by definition; the drop
  that defines it inflates vol mechanically). The structure exists; it just
  isn't information.
- **Regimes don't age.** Dwell times are exponential — memoryless. How long a
  state has lasted says nothing about when it ends. "The rally is old, a top
  is due" is quantitatively false; transition hazard is flat.
- **Of the two axes, only volatility carries real information — and only at
  bottoms.** Conditioning a simple price-extreme trigger on the panic state
  adds +3.5 points of precision (p<0.001) and beats a stricter unconditional
  trigger — genuine state information, not just selectivity. The price axis is
  redundant (a price extreme already implies its state). Nothing conditions
  tops.

## The five market facts the day established

1. **Markets are earthquake-family, not Gaussian-family.** Return magnitudes
   follow a power law (tail exponent ≈ 2.5, the "cubic law"). 135 five-sigma
   days occurred where a Gaussian world predicts 0.01; the 1987 crash is a
   once-per-10⁶⁶-years event under normality. Extremes are not anomalies —
   they are the distribution.
2. **The market has a scale-invariant swing vocabulary.** Slicing price at its
   pivots yields swing sizes that are Zipf-distributed at every scale, and the
   distributions at n=20 and n=200 collapse onto one master curve after a
   single rescaling. A big swing is a magnified small swing — the grammar is
   self-similar (the deep justification for a multi-scale detector, and a hint
   that per-scale tuning is partly redundant).
3. **Two processes wear one price chart.** Magnitude is deeply structured:
   volatility clusters with aftershock-style memory (Hurst 0.90, correlations
   alive after a year). Direction is structureless: signed returns are
   spectrally white (Hurst 0.53), a fair coin even mid-storm. *The market
   remembers how hard it has been moving and forgets which way.*
4. **The leverage effect is one-sided in a way that decides everything.**
   Panic is an *event*: at bottoms, the vol spike and the price extreme land
   on the same bars — so bottoms are detectable, and vol-peak *exhaustion*
   times them (long-horizon payoff p<0.0001). Complacency is a *state*: at
   tops, vol decays slowly into a plateau with no event to catch — vol
   *awakening* after tops is late, noisy, artifact-prone, and carries zero
   short payoff at any horizon. Bottoms are earthquakes; tops are droughts.
5. **Because of 3+4, a price-only detector can locate turns but not time
   direction.** Turn location at 2–4× chance is achievable (it harvests
   magnitude structure). Direction-at-the-bar is ~unpredictable; long entries
   at located bottoms work largely because the secular drift resolves them.
   Hence: bottoms pay at short holds, tops would only pay at very long holds
   and never cleared an honest null.

## Implications for the detector (and the day's engineering echo)

The engine's entire measured information reduces to the model's two axes —
the drift vote (price regime) and the volatility vote (vol regime); everything
else was rubber-stamp, dead, or inverted (much of it because search boxes
never bracketed the features' real distributions — repaired in v18). A
133-line distillation keeping only agreement + drift + vola matched the full
engine on its own benchmark. The honest product shape this implies: a
**bottom-detector conditioned on panic-state volatility dynamics, exit by
drift/time**, with tops treated as a located-but-untimed monitor.

## Meta-lesson (worth more than any single finding)

Twice today an exciting structure was fully reproduced by a random-walk null —
our own labels reflected back at us. The GBM null + pre-registered thresholds
are now permanent fixtures: in this domain, *everything* looks like signal
until it survives noise.
