"""Leave-one-out feature ablation as ``shape_variants`` (no parity change).

The 7 vote features combine ADDITIVELY in the detector — a signal fires when
``sum(use_X & vote_X) >= confirm_count`` (``detector.py`` max_votes sum /
vote-count gate). So the clean way to gauge "what features the model wants" is
leave-one-out: search an all-features-on baseline, then all-on-minus-one for
each feature, and read the HOLDOUT-score change as that feature's marginal
contribution.

This rides the existing ``run_v17_gpu(shape_variants=...)`` machinery — every
variant is independently searched, re-scored on the exact CPU detector, gated,
and holdout-evaluated. Nothing here touches the parity-frozen detection path.
"""
from __future__ import annotations

#: The 7 use_X votes summed into ``max_votes`` (``detector.py``). pivot_drift is
#: structurally always-on (not a use_X toggle); er_gate is a directional gate,
#: not an additive vote — so neither is part of the leave-one-out set.
ABLATION_VOTE_FEATURES: tuple[str, ...] = (
    "trend", "volume", "momentum", "momentum_velocity",
    "volatility", "gjr_asym", "har_vol",
)


def build_ablation_variants(
    features: tuple[str, ...] = ABLATION_VOTE_FEATURES,
    sides: tuple[str, ...] = ("high", "low"),
) -> dict[str, dict[str, bool]]:
    """Leave-one-out ``shape_variants`` for ``run_v17_gpu``.

    Returns ``{"all_on": {...}, "minus_<feature>": {...}, ...}`` — each value a
    dict of ``use_<feature>_<side>`` overrides. ``all_on`` enables every vote on
    both sides; each ``minus_<feature>`` is ``all_on`` with that one feature
    disabled on both sides (so the delta isolates a single feature).
    """
    all_on: dict[str, bool] = {
        f"use_{f}_{s}": True for f in features for s in sides
    }
    variants: dict[str, dict[str, bool]] = {"all_on": dict(all_on)}
    for f in features:
        variant = dict(all_on)
        for s in sides:
            variant[f"use_{f}_{s}"] = False
        variants[f"minus_{f}"] = variant
    return variants


def ablation_deltas(out: dict, side: str) -> dict[str, float]:
    """Signed leave-one-out importance per feature, most-helpful first.

    ``delta(feature) = holdout(all_on) - holdout(minus_feature)`` on the given
    side. ``> 0`` the feature HELPS the selection-untouched tail; ``< 0`` it
    HURTS it (the optimizer generalizes better without it — overfit noise).
    Variants are independently searched, so a negative delta is meaningful, not
    a bug. Returns an insertion-ordered dict sorted by descending delta.

    Raises:
        KeyError: if the ``all_on`` baseline variant is absent.
    """
    variants = out["variants"]
    base = variants["all_on"]["sides"][side]["holdout"]["score"]
    deltas: dict[str, float] = {}
    for name, v in variants.items():
        if not name.startswith("minus_"):
            continue
        feature = name[len("minus_"):]
        try:                                  # a degenerate variant (None/missing
            score = v["sides"][side]["holdout"]["score"]   # holdout) is skipped,
        except (KeyError, TypeError):                       # not crashed on.
            continue
        deltas[feature] = float(base) - float(score)
    return dict(sorted(deltas.items(), key=lambda kv: kv[1], reverse=True))


__all__ = ["ABLATION_VOTE_FEATURES", "build_ablation_variants", "ablation_deltas"]
