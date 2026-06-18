"""Colab H100 validation — the ~5-minute real-hardware confirmation (spec §6).

Run this ON A COLAB H100 from the repo root (with ``data/raw/*.csv`` present):

    !python temp/colab_h100_validation.py

It performs, in order:

  1. torch-cuda install check (installs torch if missing; H100 needs CUDA).
  2. §2 PIR byte-identity spike ON THE GPU (``temp/pir_parity_spike.py`` with
     ``DEVICE=cuda``) — confirms or revises the CPU "trust-kernel" verdict on
     real H100 reduction order.
  3. ``tests/test_v17_gpu_parity.py`` ON THE GPU (a pytest plugin re-defaults
     every TorchPhase1/GpuPooledScorer to ``cuda`` and moves drift tensors).
  4. END-TO-END signal-flip measurement: ``signals_torch`` on cuda vs the
     EXACT ``SpeculatorDetector`` on representative slices x params draws.
  5. One tiny end-to-end ``run_v17_gpu`` on a 2-asset pool, ``device="cuda"``.

Prints ONE final verdict line:  ``H100 VALIDATION: PASS|FAIL`` plus the
measured real-hardware signal-flip rate. PASS requires: parity tests green,
flip rate exactly 0.0 (trust-kernel confirmed), and 0 dropped finalists in
the e2e run. A nonzero flip rate prints the revised branch ("noisy-ranker")
and the top-K the §6 pipeline should use instead.
"""
from __future__ import annotations

import dataclasses
import importlib.util
import json
import logging
import subprocess
import sys
import tempfile
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

logger = logging.getLogger("colab_h100_validation")

SPX_CSV = _REPO / "data" / "raw" / "SPX_1D_18710201_20260318.csv"
DAX_CSV = _REPO / "data" / "raw" / "DAX_1D_19700102_20260324.csv"
SEARCH_KW = {"popsize": 4, "sobol_n": 4, "generations": 1, "top_k": 3,
             "rng_seed": 11}
ERA_KW = {"oos_fraction": 0.15, "step_fraction": 0.5}
#: Validation device. "cuda" on the H100; a CPU dry-run (mechanics check,
#: NOT the real-hardware confirmation) is possible with DEVICE = "cpu".
DEVICE = "cuda"


# ---------------------------------------------------------------------------
# Step 1 — torch + CUDA
# ---------------------------------------------------------------------------


def ensure_torch_cuda() -> "object | None":
    """Import torch (pip-installing it if absent); require a CUDA device."""
    try:
        import torch
    except ImportError:
        logger.info("torch missing — installing (Colab wheel includes CUDA)...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "torch"],
                       check=True)
        import torch
    if DEVICE == "cuda" and not torch.cuda.is_available():
        logger.error("torch %s has NO CUDA device — run on a GPU runtime",
                     torch.__version__)
        return None
    logger.info("torch %s on %s", torch.__version__,
                torch.cuda.get_device_name(0) if DEVICE == "cuda" else DEVICE)
    return torch


# ---------------------------------------------------------------------------
# Step 2 — §2 PIR spike on the GPU
# ---------------------------------------------------------------------------


def run_pir_spike_on_gpu(torch) -> dict:
    """Run temp/pir_parity_spike.py with DEVICE=cuda (host-transfer patched)."""
    import numpy as np
    spec = importlib.util.spec_from_file_location(
        "pir_parity_spike", _REPO / "temp" / "pir_parity_spike.py")
    spike = importlib.util.module_from_spec(spec)
    sys.modules["pir_parity_spike"] = spike
    spec.loader.exec_module(spike)
    spike.DEVICE = torch.device(DEVICE)

    def _build_pir_gpu(close64, mean_fn, compute_dtype):
        """spike.build_pir_torch with the result moved back to the host."""
        scales = list(range(spike.SCALE_MIN, spike.SCALE_MAX + 1))
        out = np.full((len(scales), len(close64)), np.nan, dtype=np.float32)
        x = torch.tensor(close64, dtype=compute_dtype, device=spike.DEVICE)
        for i, s in enumerate(scales):
            ratio = x / mean_fn(x, s)
            pir = spike._pir_from_ratio(ratio, max(s, 20))
            out[i] = pir.to(torch.float32).cpu().numpy()
        return out

    spike.build_pir_torch = _build_pir_gpu
    return spike.main()


# ---------------------------------------------------------------------------
# Step 3 — parity tests on the GPU (device-redefault pytest plugin)
# ---------------------------------------------------------------------------


class _CudaDefaultPlugin:
    """Re-default every torch evaluator in the suite to ``cuda``."""

    def pytest_sessionstart(self, session) -> None:  # noqa: ARG002
        from src.v17_gpu import eval_torch, phase2_scan
        tp_init = eval_torch.TorchPhase1.__init__
        gp_init = phase2_scan.GpuPooledScorer.__init__
        cli = phase2_scan.candidate_lane_inputs

        def tp_cuda(self, df, base_params, artifacts=None, device=DEVICE,
                    use_torch_pir=False):
            tp_init(self, df, base_params, artifacts, device, use_torch_pir)

        def gp_cuda(self, folds, streams, side, base_params, **kw):
            kw.setdefault("device", DEVICE)
            gp_init(self, folds, streams, side, base_params, **kw)

        def cli_cuda(tp, params, drift):
            return cli(tp, params, drift.to(tp.device))  # CPU-built fixtures

        eval_torch.TorchPhase1.__init__ = tp_cuda
        phase2_scan.GpuPooledScorer.__init__ = gp_cuda
        phase2_scan.candidate_lane_inputs = cli_cuda


def run_parity_tests_on_gpu() -> bool:
    import pytest
    rc = pytest.main(["tests/test_v17_gpu_parity.py", "-q", "--no-header",
                      "-p", "no:cacheprovider"],
                     plugins=[_CudaDefaultPlugin()])
    return rc == 0


# ---------------------------------------------------------------------------
# Step 4 — real-hardware end-to-end signal-flip rate
# ---------------------------------------------------------------------------


def measure_signal_flip_rate(torch) -> dict:
    """signals_torch on CUDA vs the EXACT SpeculatorDetector, per-bar flips."""
    import numpy as np
    from src.detector import SpeculatorDetector, build_detector_artifacts
    from src.indicators import Params
    from src.pooled_validation import load_stream_frame
    from src.search_space import space_for
    from src.v17_gpu.drift_precompute import DriftSpec, precompute_drift
    from src.v17_gpu.eval_torch import TorchPhase1
    from src.v17_gpu.phase2_scan import signals_torch
    from src.v17_optimize import active_threshold_fields

    base = Params()
    rng = np.random.default_rng(7)
    draws = [base]
    for _ in range(3):  # in-bounds threshold draws (same recipe as the tests)
        over = {}
        for side in ("high", "low"):
            fb = space_for(side).float_bounds
            for f in active_threshold_fields(base, side):
                lo, hi = fb[f[: f.rfind("_")]]
                over[f] = float(rng.uniform(lo, hi))
        draws.append(dataclasses.replace(base, **over))

    slices = {
        "spx": load_stream_frame(str(SPX_CSV)).iloc[-2000:].reset_index(drop=True),
        "dax": load_stream_frame(str(DAX_CSV)).iloc[-2000:].reset_index(drop=True),
    }
    spec = DriftSpec.from_params(base)
    flips = bars = 0
    for name, df in slices.items():
        art = build_detector_artifacts(df)
        tp = TorchPhase1(df, base, art, device=DEVICE)
        drift = torch.from_numpy(precompute_drift(df, spec)).to(DEVICE)
        for p in draws:
            ref = SpeculatorDetector(df, p, art).run()
            got = signals_torch(tp, p, drift)
            flips += int((got["signal_high"] != ref["signal_high"].values).sum())
            flips += int((got["signal_low"] != ref["signal_low"].values).sum())
            bars += 2 * len(df)
        logger.info("flip check %s: cumulative %d/%d", name, flips, bars)
    return {"flips": flips, "bars": bars,
            "flip_rate": flips / bars if bars else 0.0}


# ---------------------------------------------------------------------------
# Step 5 — tiny end-to-end run_v17_gpu on CUDA
# ---------------------------------------------------------------------------


def run_tiny_e2e(tmp_dir: str) -> dict:
    from src.v17_runner import run_v17_gpu
    pool = Path(tmp_dir) / "pool"
    pool.mkdir()
    for csv, name in ((SPX_CSV, "SPX_1D_00000000_00000000.csv"),
                      (DAX_CSV, "DAX_1D_00000000_00000000.csv")):
        lines = csv.read_text().splitlines()
        (pool / name).write_text("\n".join([lines[0]] + lines[1:][-6000:]) + "\n")
    return run_v17_gpu(groups=["INDICES"], timeframes=["1D"],
                       data_dir=str(pool), sides=("low",), era_kw=ERA_KW,
                       search_kw=SEARCH_KW, run_slug="colab_h100",
                       device=DEVICE, tv_audit=False)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    t0 = time.time()
    report: dict = {"stages": {}}

    torch = ensure_torch_cuda()
    report["stages"]["torch_cuda"] = bool(torch)
    if torch is None:
        print("H100 VALIDATION: FAIL  (no CUDA device)")
        return 1
    if not SPX_CSV.exists() or not DAX_CSV.exists():
        print(f"H100 VALIDATION: FAIL  (missing {SPX_CSV} / {DAX_CSV} — "
              "upload data/raw before running)")
        return 1

    logger.info("=== [2/5] PIR spike on GPU ===")
    verdict = run_pir_spike_on_gpu(torch)
    spike_ok = verdict["recommended_branch"] == "trust-kernel"
    report["stages"]["pir_spike"] = {
        "recommended_branch": verdict["recommended_branch"],
        "exact_match": verdict["exact_match"],
        "signal_flip_rate": verdict["signal_flip_rate"],
        "suggested_topK": verdict["suggested_topK"],
    }

    logger.info("=== [3/5] tests/test_v17_gpu_parity.py on GPU ===")
    tests_ok = run_parity_tests_on_gpu()
    report["stages"]["gpu_parity_tests"] = tests_ok

    logger.info("=== [4/5] end-to-end signal-flip rate on GPU ===")
    flip = measure_signal_flip_rate(torch)
    report["stages"]["signal_flip"] = flip

    logger.info("=== [5/5] tiny run_v17_gpu on the device ===")
    with tempfile.TemporaryDirectory() as tmp:
        out = run_tiny_e2e(tmp)
    side = out["sides"]["low"]
    e2e_ok = (side["n_dropped_finalists"] == 0
              and side["final_lcb"] == side["leaderboard"][0]["cpu_lcb"])
    report["stages"]["e2e_run_v17_gpu"] = {
        "ok": e2e_ok, "final_lcb": side["final_lcb"],
        "n_finalists": side["n_finalists"],
        "n_dropped_finalists": side["n_dropped_finalists"],
        "max_abs_diff": max((e["abs_diff"] for e in side["leaderboard"]),
                            default=0.0),
    }

    passed = spike_ok and tests_ok and flip["flip_rate"] == 0.0 and e2e_ok
    report["verdict"] = "PASS" if passed else "FAIL"
    report["minutes"] = round((time.time() - t0) / 60.0, 1)
    print(json.dumps(report, indent=2, default=str))
    print(f"H100 VALIDATION: {report['verdict']}  "
          f"(GPU-vs-CPU signal-flip rate = {flip['flip_rate']:.3e}, "
          f"{flip['flips']}/{flip['bars']} bars)")
    if flip["flip_rate"] > 0.0:
        from src.v17_runner import topk_for_flip_rate
        print(f"REVISE §2 BRANCH -> noisy-ranker: pass "
              f"flip_rate={flip['flip_rate']:.3e} to run_v17_gpu "
              f"(top_k={topk_for_flip_rate(flip['flip_rate'])}).")
    return 0 if passed else 1


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    sys.exit(main())
