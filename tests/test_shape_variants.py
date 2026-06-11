"""Spec §B — pruning / shape-variant outer loop (plan/holdout-and-pruning-spec.md).

Assertions (never weaken):

- §B2 winner rule (``select_winner_variant``): era_pass FIRST, then deflated
  LCB; ``era_pass=None`` counts as False; ties keep the first declared variant.
- §B1 API: ``run_v17_gpu(shape_variants=...)`` defaults to None (current
  behavior); an empty dict is rejected.
- §B3 zero-active-votes safety: with mom-vel off on BOTH sides the HIGH side
  has ZERO active use_* votes (max_votes_high == 0) — the engine, the
  FastDetector and the GPU scan must all agree BYTE-IDENTICALLY on the tiny
  pool, must not crash, and the drift vote + ``max(max_votes, 1)`` clamp must
  keep signals possible (teeth: the zero-vote side fires somewhere).
- §B4 two-variant e2e on the tiny 2-asset pool: both variants complete, JSON
  shape correct, winner_variant rule respected, GPU memory freed between
  variants (the memory estimator RE-LOGS per variant).
- §B3 workbook: Cell 5 gains the RUN_PRUNED checkbox passing the
  baseline/pruned shape_variants (mom-vel off both sides) + rationale; Cell 6
  gains the variant comparison table ABOVE the winner detail; nbformat-valid.
"""
from __future__ import annotations

import dataclasses
import inspect
import json
import logging
import subprocess
import sys
from pathlib import Path

import nbformat
import numpy as np
import pytest

from src.detector import SpeculatorDetector, build_detector_artifacts
from src.indicators import Params
from src.pooled_validation import (
    StreamData,
    build_calendar_folds,
    load_stream_frame,
)
from src.scoring import add_pivot_labels
from src.search_space import space_for
from src.universe import Stream
from src.v17_optimize import PooledScorer, active_threshold_fields
from src.v17_runner import run_v17_gpu, select_winner_variant

_REPO = Path(__file__).resolve().parents[1]
_NB = _REPO / "h100_v17_gpu.ipynb"
_SPX = _REPO / "data" / "raw" / "SPX_1D_18710201_20260318.csv"
_DAX = _REPO / "data" / "raw" / "DAX_1D_19700102_20260324.csv"

#: Fold geometry for the tiny pool (same rationale as tests/test_holdout.py).
ERA_KW = {"oos_fraction": 0.15, "step_fraction": 0.5}
SEARCH_KW = {"popsize": 4, "sobol_n": 4, "generations": 1, "top_k": 3,
             "rng_seed": 11}

#: Spec §B3 — the workbook's data-driven pruning config (mom-vel off BOTH sides).
PRUNED_OVERRIDES = {"use_momentum_velocity_high": False,
                    "use_momentum_velocity_low": False}

#: Every HIGH-side use_* vote switch summed into max_votes_high (detector.py).
_HIGH_VOTE_FLAGS = ("use_trend_high", "use_volume_high", "use_momentum_high",
                    "use_momentum_velocity_high", "use_volatility_high",
                    "use_gjr_asym_high", "use_har_vol_high")


# ---------------------------------------------------------------------------
# §B2 — winner selection rule: era_pass first, then deflated LCB
# ---------------------------------------------------------------------------


def _vd(era_pass, deflated: float) -> dict:
    return {"sides": {"low": {"acceptance": {"era_pass": era_pass},
                              "final_lcb_deflated": deflated}}}


def test_winner_rule_era_pass_beats_higher_deflated():
    v = {"a": _vd(False, 0.9), "b": _vd(True, 0.1)}
    assert select_winner_variant(v, "low") == "b"


def test_winner_rule_deflated_decides_within_same_pass_class():
    assert select_winner_variant(
        {"a": _vd(True, 0.2), "b": _vd(True, 0.3)}, "low") == "b"
    assert select_winner_variant(
        {"a": _vd(False, 0.2), "b": _vd(False, 0.3)}, "low") == "b"


def test_winner_rule_none_era_pass_counts_as_false_and_tie_keeps_first():
    # None == False class; exact tie on both keys -> first declared wins
    assert select_winner_variant(
        {"a": _vd(None, 0.5), "b": _vd(False, 0.5)}, "low") == "a"
    # ...and a True era_pass still beats both
    assert select_winner_variant(
        {"a": _vd(None, 0.5), "b": _vd(True, 0.0)}, "low") == "b"


def test_winner_rule_single_variant_and_empty_dict():
    assert select_winner_variant({"only": _vd(False, -1.0)}, "low") == "only"
    with pytest.raises(ValueError):
        select_winner_variant({}, "low")


# ---------------------------------------------------------------------------
# §B1 — API surface
# ---------------------------------------------------------------------------


def test_run_v17_gpu_signature_has_shape_variants_default_none():
    sig = inspect.signature(run_v17_gpu)
    assert "shape_variants" in sig.parameters
    assert sig.parameters["shape_variants"].default is None


def test_run_v17_gpu_rejects_empty_shape_variants(tmp_path):
    pytest.importorskip("torch")
    with pytest.raises(ValueError, match="shape_variants"):
        run_v17_gpu(groups=["INDICES"], timeframes=["1D"],
                    data_dir=str(tmp_path), shape_variants={})


# ---------------------------------------------------------------------------
# §B3 — zero-active-votes side: engine + FastDetector + GPU byte-identity
# ---------------------------------------------------------------------------


def _zero_votes_params() -> Params:
    """The §B3 pruned shape on the gold seed: mom-vel off BOTH sides leaves
    the HIGH side with ZERO active use_* votes (max_votes_high == 0)."""
    z = dataclasses.replace(Params(), **PRUNED_OVERRIDES)
    assert sum(bool(getattr(z, f)) for f in _HIGH_VOTE_FLAGS) == 0
    return z


def _permissive(base: Params) -> Params:
    """Threshold-only loosening so the stateful loop actually fires (teeth).
    momentum_velocity_thresh_* stays untouched — that vote is OFF here."""
    return dataclasses.replace(
        base,
        min_agreement_high=0.10, min_agreement_low=0.10,
        dur_extreme_pct_high=0.30, dur_extreme_pct_low=0.50,
        scale_div_thresh_high=0.60, scale_div_thresh_low=0.60,
        pct_extreme_high=0.55, pct_extreme_low=0.70,
        vola_high_pct_low=0.50,
        pivot_drift_thresh_high=0.001, pivot_drift_thresh_low=0.001,
        pivot_drift_gate_mult_high=10.0, pivot_drift_gate_mult_low=10.0,
    )


def _in_bounds_draws(base: Params, n: int, seed: int) -> list[Params]:
    """Random draws strictly inside the search-space float bounds. With the
    mom-vel vote OFF, ``active_threshold_fields`` excludes its threshold."""
    rng = np.random.default_rng(seed)
    out: list[Params] = []
    for _ in range(n):
        over: dict[str, float] = {}
        for side in ("high", "low"):
            bounds = space_for(side).float_bounds
            for f in active_threshold_fields(base, side):
                stem = f[: f.rfind("_")]
                lo, hi = bounds[stem]
                over[f] = float(rng.uniform(lo, hi))
        out.append(dataclasses.replace(base, **over))
    return out


def _zero_vote_slices() -> dict:
    if not _SPX.exists() or not _DAX.exists():
        pytest.skip(f"missing {_SPX} or {_DAX}")
    return {
        "spx_tail": load_stream_frame(str(_SPX)).iloc[-2200:].reset_index(drop=True),
        "dax_tail": load_stream_frame(str(_DAX)).iloc[-2200:].reset_index(drop=True),
    }


def test_zero_active_votes_engine_fast_gpu_byte_identical():
    torch = pytest.importorskip("torch")
    from src.v17_fastdetector import FastDetector
    from src.v17_gpu.drift_precompute import DriftSpec, precompute_drift
    from src.v17_gpu.eval_torch import TorchPhase1
    from src.v17_gpu.phase2_scan import signals_torch

    z = _zero_votes_params()
    draws = [z, _permissive(z)] + _in_bounds_draws(z, 2, seed=13)
    fired_high = fired_low = 0
    for name, df in _zero_vote_slices().items():
        art = build_detector_artifacts(df)
        fd = FastDetector(df, z, art)
        tp = TorchPhase1(df, z, art)
        drift = torch.from_numpy(precompute_drift(df, DriftSpec.from_params(z)))
        for pi, p in enumerate(draws):
            ref = SpeculatorDetector(df, p, art).run()       # must not crash
            ref_h = np.asarray(ref["signal_high"].to_numpy(), dtype=bool)
            ref_l = np.asarray(ref["signal_low"].to_numpy(), dtype=bool)
            fast = fd.signals(p)
            got = signals_torch(tp, p, drift)
            assert np.array_equal(fast["signal_high"], ref_h), (name, pi)
            assert np.array_equal(fast["signal_low"], ref_l), (name, pi)
            assert np.array_equal(got["signal_high"], ref_h), (name, pi)
            assert np.array_equal(got["signal_low"], ref_l), (name, pi)
            fired_high += int(ref_h.sum())
            fired_low += int(ref_l.sum())
    # §B3 teeth: drift still votes, req clamps to >= 1 -> signals POSSIBLE
    assert fired_high > 0, ("zero-vote HIGH side never fired anywhere — the "
                            "drift-vote / max(max_votes,1) clamp is untested")
    assert fired_low > 0


def test_zero_active_votes_pooled_scorers_agree():
    pytest.importorskip("torch")
    from src.v17_fastdetector import FastPooledScorer
    from src.v17_gpu.phase2_scan import GpuPooledScorer

    if not _DAX.exists():
        pytest.skip(f"missing {_DAX}")
    df = load_stream_frame(str(_DAX)).iloc[-6000:].copy()
    add_pivot_labels(df)
    s = Stream(ticker="DAX", timeframe="1D", path=str(_DAX), cluster_id="EU_EQ")
    folds = build_calendar_folds(
        [StreamData(stream=s, df=df, bar_seconds=86400.0)], **ERA_KW)
    z = _zero_votes_params()
    real = PooledScorer(folds=folds, streams=[s], side="high")
    fast = FastPooledScorer(folds=folds, streams=[s], side="high",
                            base_params=z)
    gpu = GpuPooledScorer(folds=folds, streams=[s], side="high",
                          base_params=z)
    for p in (z, _permissive(z)):
        ref = float(real.score(p))
        assert abs(float(fast.score(p)) - ref) < 1e-12
        assert abs(float(gpu.score(p)) - ref) < 1e-9


# ---------------------------------------------------------------------------
# §B4 — two-variant e2e on the tiny 2-asset pool
# ---------------------------------------------------------------------------


def _two_asset_dir(tmp_path: Path) -> str:
    """SPX + DAX 6000-bar tails (pattern of tests/test_v17_gpu_integration)."""
    if not _SPX.exists() or not _DAX.exists():
        pytest.skip(f"missing {_SPX} or {_DAX}")
    for csv, name in ((_SPX, "SPX_1D_00000000_00000000.csv"),
                      (_DAX, "DAX_1D_00000000_00000000.csv")):
        lines = csv.read_text().splitlines()
        keep = [lines[0]] + lines[1:][-6000:]
        (tmp_path / name).write_text("\n".join(keep) + "\n")
    return str(tmp_path)


def test_run_v17_gpu_two_variant_outer_loop(tmp_path, caplog):
    pytest.importorskip("torch")
    caplog.set_level(logging.INFO)

    out = run_v17_gpu(
        groups=["INDICES"], timeframes=["1D"],
        data_dir=_two_asset_dir(tmp_path),
        sides=("low",), era_kw=ERA_KW, search_kw=SEARCH_KW,
        run_slug="t_variants_e2e", device="cpu",
        results_dir=str(tmp_path / "results"),
        shape_variants={"baseline": {}, "pruned": dict(PRUNED_OVERRIDES)},
    )

    # --- §B2 output structure ---------------------------------------------
    assert set(out["variants"]) == {"baseline", "pruned"}
    assert out["variants"]["baseline"]["overrides"] == {}
    assert out["variants"]["pruned"]["overrides"] == PRUNED_OVERRIDES
    for name in ("baseline", "pruned"):
        d = out["variants"][name]["sides"]["low"]
        # the existing per-side dict, complete per variant
        assert {"seed_lcb", "final_lcb", "final_lcb_deflated", "deflation",
                "best_params", "leaderboard", "acceptance", "holdout",
                "calibration", "signal_cards", "trace"} <= set(d)
        assert d["holdout"] is not None          # holdout ran per variant

    # --- §B1 the override really reached the seed (dataclasses.replace) ----
    bp_pruned = out["variants"]["pruned"]["sides"]["low"]["best_params"]
    assert bp_pruned["use_momentum_velocity_high"] is False
    assert bp_pruned["use_momentum_velocity_low"] is False
    bp_base = out["variants"]["baseline"]["sides"]["low"]["best_params"]
    assert bp_base["use_momentum_velocity_high"] is True
    assert bp_base["use_momentum_velocity_low"] is True

    # --- §B2 winner selection rule respected --------------------------------
    win = out["winner_variant"]["low"]
    assert win in out["variants"]
    assert win == select_winner_variant(out["variants"], "low")
    cand = {n: v["sides"]["low"] for n, v in out["variants"].items()}
    passing = {n for n, d in cand.items()
               if bool(d["acceptance"]["era_pass"])}
    pool = passing or set(cand)
    assert win in pool                       # era_pass class always wins
    best_defl = max(cand[n]["final_lcb_deflated"] for n in pool)
    assert cand[win]["final_lcb_deflated"] == best_defl
    # top-level sides == the winning variant's per-side dict
    assert out["sides"]["low"] == cand[win]

    # --- §B1 GPU memory freed between variants: estimator RE-LOGS per variant
    msgs = [r.getMessage() for r in caplog.records
            if "gpu memory estimate" in r.getMessage()]
    for name in ("baseline", "pruned"):
        assert any(f"variant={name}" in m for m in msgs), \
            f"memory estimator did not re-log for variant {name!r}"

    # --- run JSON round-trips with the full variant structure ---------------
    written = json.loads(Path(out["_written"]).read_text())
    assert set(written["variants"]) == {"baseline", "pruned"}
    assert written["winner_variant"]["low"] == win
    assert written["sides"]["low"]["final_lcb"] == \
        written["variants"][win]["sides"]["low"]["final_lcb"]


# ---------------------------------------------------------------------------
# §B3 workbook — RUN_PRUNED checkbox + Cell-6 variant comparison table
# ---------------------------------------------------------------------------


def _src(cell) -> str:
    s = cell["source"]
    return s if isinstance(s, str) else "".join(s)


@pytest.fixture(scope="module")
def notebook():
    res = subprocess.run(
        [sys.executable, str(_REPO / "temp" / "build_h100_notebook.py")],
        cwd=_REPO, capture_output=True, text=True)
    assert res.returncode == 0, f"builder self-check failed:\n{res.stderr}"
    nb = nbformat.read(_NB, as_version=4)
    nbformat.validate(nb)
    return nb


def test_cell5_run_pruned_checkbox_passes_variant_configs(notebook):
    src = _src(notebook.cells[5])
    norm = " ".join(src.split())
    assert 'RUN_PRUNED = True #@param {type:"boolean"}' in norm
    assert '"baseline": {}' in norm
    assert '"use_momentum_velocity_high": False' in norm
    assert '"use_momentum_velocity_low": False' in norm
    assert "if RUN_PRUNED else None" in norm
    assert "shape_variants=shape_variants" in norm
    assert "neutralized" in src              # §B3 rationale in the form comment
    compile(src, "<cell5>", "exec")


def test_cell5_run_ablation_passes_leave_one_out_variants(notebook):
    src = _src(notebook.cells[5])
    norm = " ".join(src.split())
    assert 'RUN_ABLATION = True #@param {type:"boolean"}' in norm
    assert "build_ablation_variants" in norm
    # ablation takes precedence over the pruned/baseline dict
    assert "if RUN_ABLATION else" in norm
    assert "shape_variants=shape_variants" in norm
    compile(src, "<cell5>", "exec")


def test_cell6_ablation_delta_table_when_all_on_present(notebook):
    src = _src(notebook.cells[6])
    assert "ablation_deltas" in src
    assert "ABLATION" in src               # the LOO importance section header
    assert "all_on" in src                 # presence gate for the ablation table
    compile(src, "<cell6>", "exec")


def test_cell6_variant_comparison_table_above_winner_detail(notebook):
    src = _src(notebook.cells[6])
    assert "VARIANT COMPARISON" in src
    assert "winner_variant" in src
    # spec §B2 table columns: raw/deflated LCB, verdict, holdout pass, n_signals
    for tok in ("final_lcb", "final_lcb_deflated", "verdict", "era_pass",
                "n_signals"):
        assert tok in src, tok
    # the table sits ABOVE the per-side winner detail
    assert src.index("VARIANT COMPARISON") < src.index("changed thresholds")
    compile(src, "<cell6>", "exec")


def test_notebook_nbformat_valid_and_cell_count_unchanged(notebook):
    assert len(notebook.cells) == 8          # no new cells, §B3 edits in-place
