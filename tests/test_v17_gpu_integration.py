"""Spec §6 (PHASE 4) — ``run_v17_gpu`` integration + Colab handoff support.

Assertions (plan/gpu-refactor-build-spec.md §6):

- End-to-end ``run_v17_gpu`` on a 2-asset CPU pool produces a leaderboard
  whose top finalist's reported LCB equals the exact CPU ``PooledScorer``
  LCB to ``< 1e-9`` (the kernel ranks; the EXACT detector reports).
- The old ``|fast-real| > 1e-9`` warning is generalized into a HARD finalist
  filter: any finalist whose GPU LCB disagrees with the CPU LCB beyond
  tolerance is DROPPED (``filter_finalists``).
- ``top_k`` defaults are sized from the §2 PIR-spike signal-flip rate
  (``topk_for_flip_rate`` — trust-kernel flip rate 0.0 -> 16).
- Acceptance gates (``v17_acceptance``) and the per-asset TradingView-export
  audit hook are wired into the output.
- Golden Phase-0 snapshot intact (sha256 self-consistency here; the full
  behavioral contract stays in ``tests/test_parity_golden.py``).
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

import numpy as np  # noqa: E402

from src.indicators import Params  # noqa: E402
from src.pooled_validation import (  # noqa: E402
    StreamData,
    apply_volume_policy,
    build_calendar_folds,
    load_stream_frame,
)
from src.scoring import add_pivot_labels  # noqa: E402
from src.universe import resolve_streams  # noqa: E402
from src.v17_optimize import PooledScorer  # noqa: E402
from src.v17_runner import filter_finalists, run_v17_gpu, topk_for_flip_rate  # noqa: E402
from src.v17_search import Candidate  # noqa: E402

_SPX = Path("data/raw/SPX_1D_18710201_20260318.csv")
_DAX = Path("data/raw/DAX_1D_19700102_20260324.csv")
_GOLDEN = Path("results/diag/golden")

#: Fold geometry for the tiny pool: 6000-bar tails, 900-bar OOS slices so the
#: LOW side has informative folds (same rationale as temp/capture_baseline.py).
ERA_KW = {"oos_fraction": 0.15, "step_fraction": 0.5}
SEARCH_KW = {"popsize": 4, "sobol_n": 4, "generations": 1, "top_k": 3,
             "rng_seed": 11}


# ---------------------------------------------------------------------------
# topk_for_flip_rate — finalist set sized to the §2 spike flip rate
# ---------------------------------------------------------------------------


def test_topk_for_flip_rate_tiers():
    assert topk_for_flip_rate(0.0) == 16        # trust-kernel (measured §2)
    assert topk_for_flip_rate(1e-7) == 32
    assert topk_for_flip_rate(1e-6) == 32
    assert topk_for_flip_rate(1e-5) == 48
    assert topk_for_flip_rate(1e-4) == 64
    assert topk_for_flip_rate(5e-4) == 128
    assert topk_for_flip_rate(1e-2) == 256
    assert topk_for_flip_rate(1e-1) == 512
    assert topk_for_flip_rate(0.5) == 1024      # ranker is near-noise


# ---------------------------------------------------------------------------
# filter_finalists — the HARD |gpu-cpu| > tol finalist filter
# ---------------------------------------------------------------------------


def _cand(gpu_score: float, marker: float) -> Candidate:
    """Candidate distinguishable via its min_agreement_low value."""
    p = dataclasses.replace(Params(), min_agreement_low=marker)
    return Candidate(params=p, score=gpu_score, stage="cma")


class _TableScorer:
    """Fake EXACT CPU scorer: looks up the candidate by its marker field."""

    def __init__(self, table: dict[float, float]) -> None:
        self.table = table
        self.n_calls = 0

    def score(self, params: Params) -> float:
        self.n_calls += 1
        return self.table[round(float(params.min_agreement_low), 6)]


def test_filter_finalists_drops_disagreeing_and_reranks_by_cpu_lcb():
    a = _cand(0.5, 0.11)            # exact agreement -> kept
    b = _cand(0.6, 0.12)            # GPU says 0.6, CPU says 0.3 -> DROPPED
    c = _cand(0.4, 0.13)            # within 1e-9 -> kept
    real = _TableScorer({0.11: 0.5, 0.12: 0.3, 0.13: 0.4 + 5e-10})

    survivors, dropped = filter_finalists([b, a, c], real, tol=1e-9)

    assert real.n_calls == 3
    # b violated parity beyond tolerance: HARD-dropped, never reported
    assert [e["candidate"] for e in dropped] == [b]
    assert dropped[0]["abs_diff"] == pytest.approx(0.3)
    # survivors re-ranked by the EXACT CPU LCB (descending)
    assert [e["candidate"] for e in survivors] == [a, c]
    assert survivors[0]["cpu_lcb"] == 0.5
    assert survivors[1]["cpu_lcb"] == 0.4 + 5e-10
    for e in survivors:
        assert e["abs_diff"] <= 1e-9


def test_filter_finalists_all_dropped_returns_empty_survivors():
    b = _cand(0.6, 0.12)
    survivors, dropped = filter_finalists([b], _TableScorer({0.12: 0.0}),
                                          tol=1e-9)
    assert survivors == []
    assert len(dropped) == 1


# ---------------------------------------------------------------------------
# Golden Phase-0 snapshot intact (file-level; behavior re-asserted by
# tests/test_parity_golden.py which stays in the suite)
# ---------------------------------------------------------------------------


def test_golden_snapshot_files_intact():
    payload_path = _GOLDEN / "golden_baseline.json"
    if not payload_path.exists():
        pytest.skip("golden snapshot not captured yet (run temp/capture_baseline.py)")
    payload = json.loads(payload_path.read_text())
    assert payload["seed"] == 42
    for asset in ("SPX", "DAX"):
        npz_path = _GOLDEN / f"golden_{asset}.npz"
        assert npz_path.exists(), f"golden npz missing for {asset}"
        sha = payload["assets"][asset]["array_sha256"]
        with np.load(npz_path) as npz:
            assert sorted(npz.files) == sorted(sha)
            for key in npz.files:
                assert hashlib.sha256(npz[key].tobytes()).hexdigest() == sha[key], \
                    f"golden {asset}:{key} drifted on disk"


# ---------------------------------------------------------------------------
# §6 end-to-end: 2-asset CPU pool through run_v17_gpu
# ---------------------------------------------------------------------------


def _two_asset_dir(tmp_path: Path) -> str:
    """SPX + DAX 6000-bar tails as a canonical-name 2-asset pool."""
    if not _SPX.exists() or not _DAX.exists():
        pytest.skip(f"missing {_SPX} or {_DAX}")
    for csv, name in ((_SPX, "SPX_1D_00000000_00000000.csv"),
                      (_DAX, "DAX_1D_00000000_00000000.csv")):
        lines = csv.read_text().splitlines()
        keep = [lines[0]] + lines[1:][-6000:]
        (tmp_path / name).write_text("\n".join(keep) + "\n")
    return str(tmp_path)


def _rebuild_folds(data_dir: str):
    """The EXACT fold pipeline run_v17_gpu uses (mirrors run_v17)."""
    streams = resolve_streams(["INDICES"], ["1D"], data_dir=data_dir)
    sds = []
    for s in streams:
        df = load_stream_frame(s.path)
        df, keep = apply_volume_policy(df, policy="price_only")
        assert keep
        add_pivot_labels(df)
        sds.append(StreamData(stream=s, df=df, bar_seconds=86400.0))
    folds = build_calendar_folds(sds, **ERA_KW)
    return folds, [sd.stream for sd in sds]


def test_run_v17_gpu_end_to_end_two_asset_pool(tmp_path):
    data_dir = _two_asset_dir(tmp_path)
    out = run_v17_gpu(
        groups=["INDICES"], timeframes=["1D"], data_dir=data_dir,
        sides=("low",), era_kw=ERA_KW, search_kw=SEARCH_KW,
        run_slug="t_gpu_e2e", device="cpu",
        results_dir=str(tmp_path / "results"),
    )

    assert sorted(out["streams"]) == ["DAX_1D", "SPX_1D"]
    assert out["n_folds"] >= 2          # genuinely multi-fold, multi-asset
    assert out["search"] == "cma-gpu"
    side = out["sides"]["low"]

    # --- leaderboard: HARD finalist filter + CPU re-rank ------------------
    lb = side["leaderboard"]
    assert 1 <= len(lb) <= SEARCH_KW["top_k"]
    assert side["n_dropped_finalists"] == 0      # trust-kernel: no flips
    cpu = [e["cpu_lcb"] for e in lb]
    assert cpu == sorted(cpu, reverse=True)
    for e in lb:
        assert e["abs_diff"] <= 1e-9             # every survivor passed the filter
    assert side["final_lcb"] == lb[0]["cpu_lcb"]

    # --- the load-bearing §6 assertion: reported LCB == EXACT CPU LCB -----
    folds, kept = _rebuild_folds(data_dir)
    real = PooledScorer(folds=folds, streams=kept, side="low")
    best = Params(**side["best_params"])
    diff = abs(side["final_lcb"] - real.score(best))
    if not diff < 1e-9:
        raise AssertionError(
            f"HARD FAIL §6: reported LCB != exact CPU PooledScorer LCB "
            f"(diff={diff:g})")
    assert abs(side["seed_lcb"] - real.score(Params())) < 1e-9
    assert side["final_lcb"] > 0.0               # teeth: the pool actually scores

    # --- acceptance gates + deflation + audit hook -------------------------
    acc = side["acceptance"]
    assert acc["verdict"] in {"PASS", "FRAGILE", "REJECT"}
    assert set(acc) >= {"verdict", "pinned", "bootstrap", "era_pass"}
    assert acc["bootstrap"]["lcbs"], "bootstrap stability never evaluated"
    assert "final_lcb_deflated" in side and "deflation" in side
    assert side["final_lcb_deflated"] <= side["final_lcb"]

    audit = out["tv_audit"]
    assert audit is not None and sorted(audit) == sorted(out["streams"])
    for entry in audit.values():
        assert entry["export"] is None           # no enriched export in tmp pool

    # --- provenance ---------------------------------------------------------
    assert side["n_evals"] == 1 + SEARCH_KW["sobol_n"] \
        + SEARCH_KW["generations"] * SEARCH_KW["popsize"]
    assert Path(out["_written"]).exists()


def test_run_v17_gpu_rejects_bad_args(tmp_path):
    with pytest.raises(ValueError, match="sides"):
        run_v17_gpu(groups=["INDICES"], timeframes=["1D"],
                    data_dir=str(tmp_path), sides=("sideways",))
