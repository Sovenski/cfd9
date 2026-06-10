"""Phase 0 — golden-baseline parity spec (gpu-refactor-build-spec §1).

Reference contracts recorded here (every later GPU/optimizer change must keep
these green; see spec §0.6/§0.7):

  C1 — capture determinism: re-running the capture logic twice in-process
       reproduces the golden signal arrays, per-fold scores and pooled LCB
       bit-for-bit (``np.array_equal`` / exact float equality, seed=42).
  C2 — golden files on disk (``results/diag/golden/``) match a fresh capture
       bit-for-bit.
  C3 — FastDetector vs SpeculatorDetector: ``np.array_equal`` on
       ``signal_high`` and ``signal_low`` (existing test_v17_fastdetector
       contract, re-asserted on the golden slices).
  C4 — FastPooledScorer vs PooledScorer: ``abs(diff) <= 1e-12`` on the pooled
       block-bootstrap LCB (existing test_v17_fastscorer contract).

Never weaken these assertions: a mismatch is a parity finding, not a test bug.
"""
from __future__ import annotations

import dataclasses
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from src.detector import SpeculatorDetector
from src.indicators import Params
from src.v17_fastdetector import FastDetector, FastPooledScorer
from src.v17_optimize import PooledScorer, active_threshold_fields

_REPO = Path(__file__).resolve().parents[1]
_CAPTURE = _REPO / "temp" / "capture_baseline.py"

ASSETS = ("SPX", "DAX")


def _load_capture_module():
    """Import temp/capture_baseline.py by path (temp/ is not a package)."""
    if not _CAPTURE.exists():
        pytest.fail(f"missing capture script {_CAPTURE}")
    spec = importlib.util.spec_from_file_location("capture_baseline", _CAPTURE)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["capture_baseline"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def cb():
    return _load_capture_module()


@pytest.fixture(scope="module")
def folds_cache(cb):
    """build_folds is artifact-heavy; build once per asset for C3/C4/sanity."""
    return {asset: cb.build_folds(asset) for asset in ASSETS}


@pytest.fixture(scope="module")
def captures(cb):
    """One fresh in-process capture per asset (shared across tests)."""
    out = {}
    for asset in ASSETS:
        if not (_REPO / cb.ASSETS[asset]["path"]).exists():
            pytest.skip(f"missing data CSV for {asset}")
        out[asset] = cb.capture_asset(asset)
    return out


@pytest.fixture(scope="module")
def golden_on_disk(cb):
    """Ensure golden files exist (capture once if missing), return the dir."""
    if not all(
        (cb.GOLDEN_DIR / f"golden_{a}.npz").exists()
        and (cb.GOLDEN_DIR / "golden_baseline.json").exists()
        for a in ASSETS
    ):
        cb.main()
    return cb.GOLDEN_DIR


# ---------------------------------------------------------------- C1
@pytest.mark.parametrize("asset", ASSETS)
def test_capture_bit_for_bit_reproducible(cb, captures, asset):
    arrays_a, meta_a = captures[asset]
    arrays_b, meta_b = cb.capture_asset(asset)
    assert sorted(arrays_a) == sorted(arrays_b)
    for key in arrays_a:
        assert np.array_equal(arrays_a[key], arrays_b[key]), \
            f"{asset}:{key} not bit-for-bit reproducible"
    # per-fold scores + final LCB: exact float equality (same code, seed=42)
    assert meta_a["scores"] == meta_b["scores"], f"{asset} scores drifted"
    assert meta_a == meta_b


# ---------------------------------------------------------------- C2
@pytest.mark.parametrize("asset", ASSETS)
def test_golden_files_match_fresh_capture(cb, captures, golden_on_disk, asset):
    arrays, meta = captures[asset]
    npz_path = golden_on_disk / f"golden_{asset}.npz"
    assert npz_path.exists(), f"golden npz missing for {asset}"
    with np.load(npz_path) as disk:
        assert sorted(disk.files) == sorted(arrays)
        for key in arrays:
            assert np.array_equal(disk[key], arrays[key]), \
                f"{asset}:{key} on disk != fresh capture"
    payload = json.loads((golden_on_disk / "golden_baseline.json").read_text())
    assert payload["assets"][asset]["scores"] == meta["scores"], \
        f"{asset} golden scores on disk != fresh capture"
    assert payload["seed"] == cb.SEED == 42


# ---------------------------------------------------------------- C3
@pytest.mark.parametrize("asset", ASSETS)
def test_fastdetector_equals_detector_on_golden_slices(cb, folds_cache, asset):
    base = Params()
    for tag, df, art in cb.representative_slices(folds_cache[asset][0]):
        ref = SpeculatorDetector(df, base, art).run()
        fast = FastDetector(df, base, art).signals(base)
        assert np.array_equal(ref["signal_high"].values, fast["signal_high"]), \
            f"{asset}:{tag} FastDetector HIGH mismatch"
        assert np.array_equal(ref["signal_low"].values, fast["signal_low"]), \
            f"{asset}:{tag} FastDetector LOW mismatch"


# ---------------------------------------------------------------- C4
@pytest.mark.parametrize("asset", ASSETS)
def test_fastscorer_equals_pooledscorer(cb, folds_cache, asset):
    folds, streams = folds_cache[asset]
    base = Params()
    for side in ("high", "low"):
        real = PooledScorer(folds=folds, streams=streams, side=side)
        fast = FastPooledScorer(folds=folds, streams=streams, side=side,
                                base_params=base)
        assert abs(fast.score(base) - real.score(base)) <= 1e-12, \
            f"{asset}/{side} FastPooledScorer LCB drift on base params"
        # also under threshold perturbations (the contract the optimizer uses)
        rng = np.random.default_rng(cb.SEED)
        fields = active_threshold_fields(base, side)
        over = {}
        for f in fields:
            v = float(getattr(base, f))
            nv = v * float(rng.uniform(0.8, 1.2))
            over[f] = min(0.99, max(0.01, nv)) if 0 < v <= 1 else nv
        p = dataclasses.replace(base, **over)
        assert abs(fast.score(p) - real.score(p)) <= 1e-12, \
            f"{asset}/{side} FastPooledScorer LCB drift on perturbed params"


# ---------------------------------------------------------------- LCB sanity
def test_golden_lcb_matches_scorer_recompute(cb, captures, folds_cache):
    """The stored final LCB equals PooledScorer.score(Params()) re-run now."""
    for asset in ASSETS:
        folds, streams = folds_cache[asset]
        _, meta = captures[asset]
        for side in ("high", "low"):
            lcb = PooledScorer(folds=folds, streams=streams, side=side).score(Params())
            assert lcb == meta["scores"][side]["lcb"], \
                f"{asset}/{side} golden LCB != fresh PooledScorer LCB"
