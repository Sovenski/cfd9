"""Scorer-helper contracts for ``v17_acceptance`` (spec §0.1 compliance).

These helpers replace the ``firing_penalty`` / ``fold_scores`` / seeded
bootstrap additions that previously lived INSIDE trust-root files
(``src/v17_optimize.py``, ``src/v17_fastdetector.py``, ``src/validation.py``).
The oracle files return to their committed bytes; the helpers must be
numerically IDENTICAL to what they replace:

  H1 — ``_bootstrap_ci_seeded(scores, seed=42)`` equals
       ``validation.fold_scores_bootstrap_ci(scores)`` exactly (same RNG
       stream, same percentile math, same empty/single-fold edge cases).
  H2 — seeded CI is deterministic per seed and seed-sensitive.
  H3 — ``raw_fold_scores`` / ``PenalizedScorer``: penalty inactive reproduces
       the raw bootstrap LCB; penalty>0 subtracts ``firing_excess`` per fold
       BEFORE the bootstrap (the original in-scorer semantics).
  H4 — real-data identity: on golden SPX folds, ``raw_fold_scores`` on
       ``PooledScorer`` and ``FastPooledScorer`` agree (<=1e-12, the C4
       contract) and bootstrapping the raw scores reproduces ``score()``.

Never weaken these assertions (spec §0.2).
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import src.v17_acceptance as accept
from src.indicators import Params
from src.v17_acceptance import (
    PenalizedScorer,
    _bootstrap_ci_seeded,
    firing_excess,
    raw_fold_scores,
)
from src.v17_fastdetector import FastPooledScorer
from src.v17_optimize import PooledScorer
from src.validation import fold_scores_bootstrap_ci

_REPO = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------- H1
@pytest.mark.parametrize("n", [2, 3, 5, 8, 12])
def test_seeded_ci_matches_oracle_at_default_seed(n: int) -> None:
    rng = np.random.default_rng(n)
    scores = [float(x) for x in rng.normal(0.3, 0.2, size=n)]
    assert _bootstrap_ci_seeded(scores, seed=42) == fold_scores_bootstrap_ci(scores)


def test_seeded_ci_edge_cases_match_oracle() -> None:
    assert _bootstrap_ci_seeded([], seed=42) == fold_scores_bootstrap_ci([]) == (0.0, 0.0)
    assert _bootstrap_ci_seeded([0.7], seed=42) == fold_scores_bootstrap_ci([0.7]) == (0.7, 0.7)
    # block_len clamping path (block_len > n)
    s2 = [0.1, 0.9]
    assert _bootstrap_ci_seeded(s2, seed=42, block_len=5) == \
        fold_scores_bootstrap_ci(s2, block_len=5)


# ---------------------------------------------------------------- H2
def test_seeded_ci_deterministic_and_seed_sensitive() -> None:
    rng = np.random.default_rng(123)
    scores = [float(x) for x in rng.normal(0.3, 0.2, size=9)]
    a = _bootstrap_ci_seeded(scores, seed=7)
    b = _bootstrap_ci_seeded(scores, seed=7)
    assert a == b
    lows = {_bootstrap_ci_seeded(scores, seed=s)[0] for s in range(6)}
    assert len(lows) > 1, "seed must actually steer the bootstrap RNG"


def test_bootstrap_stability_still_seed_spread() -> None:
    """bootstrap_stability must keep producing one LCB per seed (not 10 copies)."""
    rng = np.random.default_rng(0)
    scores = [float(x) for x in rng.normal(0.3, 0.15, size=8)]
    out = accept.bootstrap_stability(scores, seeds=range(10), n_boot=300)
    assert len(out["lcbs"]) == 10
    assert len(set(out["lcbs"])) > 1


# ---------------------------------------------------------------- H3
def _stub_folds() -> list[tuple[float, dict]]:
    return [
        (0.50, {"precision_oos": 0.10, "recall_oos": 0.40}),  # excess vs cap=2: 2.0
        (0.30, {"precision_oos": 0.20, "recall_oos": 0.20}),  # ratio 1.0 -> 0 excess
        (0.40, {"precision_oos": 0.05, "recall_oos": 0.30}),  # ratio 6.0 -> 4.0
    ]


def test_raw_and_penalized_on_stub(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(accept, "_iter_fold_scores",
                        lambda scorer, params: iter(_stub_folds()))
    scorer = SimpleNamespace(n_boot=400, alpha=0.10, block_len=2)
    raw = raw_fold_scores(scorer, params=None)
    assert raw == [0.50, 0.30, 0.40]

    # penalty inactive => exactly the raw bootstrap LCB (original score() math)
    plain = PenalizedScorer(scorer=scorer)
    assert plain.score(None) == float(fold_scores_bootstrap_ci(
        raw, n_boot=400, alpha=0.10, block_len=2)[0])

    # penalty>0 => per-fold "s - lambda*firing_excess(...)" BEFORE the bootstrap
    lam, cap = 0.05, 2.0
    pen = PenalizedScorer(scorer=scorer, firing_penalty=lam, firing_cap=cap)
    expected = [s - lam * firing_excess(c["precision_oos"], c["recall_oos"], cap)
                for s, c in _stub_folds()]
    assert pen.score(None) == float(fold_scores_bootstrap_ci(
        expected, n_boot=400, alpha=0.10, block_len=2)[0])


def test_penalized_empty_folds_returns_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(accept, "_iter_fold_scores", lambda scorer, params: iter(()))
    scorer = SimpleNamespace(n_boot=400, alpha=0.10, block_len=2)
    assert raw_fold_scores(scorer, params=None) == []
    assert PenalizedScorer(scorer=scorer, firing_penalty=0.1).score(None) == 0.0


def test_iter_rejects_unknown_scorer() -> None:
    with pytest.raises(TypeError):
        raw_fold_scores(object(), params=None)


# ---------------------------------------------------------------- H4
@pytest.fixture(scope="module")
def spx_folds():
    cap = _REPO / "temp" / "capture_baseline.py"
    spec = importlib.util.spec_from_file_location("capture_baseline_h", cap)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["capture_baseline_h"] = mod
    spec.loader.exec_module(mod)
    if not (_REPO / mod.ASSETS["SPX"]["path"]).exists():
        pytest.skip("missing SPX data CSV")
    return mod.build_folds("SPX")


@pytest.mark.parametrize("side", ("high", "low"))
def test_raw_fold_scores_real_vs_fast_and_score_identity(spx_folds, side: str) -> None:
    folds, streams = spx_folds
    base = Params()
    real = PooledScorer(folds=folds, streams=streams, side=side)
    fast = FastPooledScorer(folds=folds, streams=streams, side=side, base_params=base)
    rf = raw_fold_scores(real, base)
    ff = raw_fold_scores(fast, base)
    assert len(rf) == len(ff)
    assert np.allclose(ff, rf, rtol=0.0, atol=1e-12), \
        f"SPX/{side} fast vs real raw fold scores drift"
    # bootstrapping the raw fold scores reproduces score() bit-for-bit
    expect = (0.0, 0.0) if not rf else fold_scores_bootstrap_ci(rf)
    assert float(real.score(base)) == (float(expect[0]) if rf else 0.0)
    # PenalizedScorer with the penalty off is score()-identical on real data
    assert PenalizedScorer(scorer=real).score(base) == float(real.score(base))
    assert PenalizedScorer(scorer=fast).score(base) == float(fast.score(base))
