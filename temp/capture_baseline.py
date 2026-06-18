"""Phase 0 — golden baseline capture (gpu-refactor-build-spec §1).

Loads SPX and DAX daily data, builds the standard calendar folds on the last
``N_BARS`` bars, runs the EXACT ``SpeculatorDetector`` with the base gold
``Params()`` on representative slices, and snapshots:

  * golden signal arrays (``signal_high``/``signal_low``) per slice → ``.npz``
  * per-fold pooled scores + the final block-bootstrap LCB per side → ``.json``

into ``results/diag/golden/`` (seed=42 throughout — the bootstrap LCB uses
``validation.fold_scores_bootstrap_ci``'s default ``seed=42``).

Every later optimizer/GPU change must reproduce this snapshot bit-for-bit
(asserted by ``tests/test_parity_golden.py``).

Run from the repo root:  python temp/capture_baseline.py
"""
from __future__ import annotations

import hashlib
import json
import logging
import sys
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from src.detector import SpeculatorDetector  # noqa: E402
from src.indicators import Params  # noqa: E402
from src.pooled_validation import (  # noqa: E402
    Fold,
    StreamData,
    build_calendar_folds,
    load_stream_frame,
)
from src.universe import Stream  # noqa: E402
from src.v17_acceptance import raw_fold_scores  # noqa: E402
from src.v17_optimize import PooledScorer  # noqa: E402

logger = logging.getLogger(__name__)

SEED: int = 42
N_BARS: int = 6000          # deterministic tail window per asset
N_FOLDS: int = 6            # calendar folds kept per asset
# Golden-specific OOS widening: with the default oos_fraction (0.03) a
# 6000-bar tail yields 401-bar OOS slices (MIN_STREAM_BARS floor) that contain
# ZERO structural pivot_N100 labels -> every fold is filtered as
# non-informative and the pooled LCB degenerates to 0.0.  0.15 gives 900-bar
# OOS slices with informative folds on BOTH sides for BOTH assets, so the
# golden snapshot actually exercises the detector + pooled scorer.
OOS_FRACTION_GOLDEN: float = 0.15
GOLDEN_DIR: Path = _REPO / "results" / "diag" / "golden"

ASSETS: dict[str, dict[str, str]] = {
    "SPX": {"path": "data/raw/SPX_1D_18710201_20260318.csv",
            "timeframe": "1D", "cluster_id": "US_EQ"},
    "DAX": {"path": "data/raw/DAX_1D_19700102_20260324.csv",
            "timeframe": "1D", "cluster_id": "EU_EQ"},
}


def build_folds(asset: str) -> tuple[list[Fold], list[Stream]]:
    """Deterministic calendar folds for one asset (last ``N_BARS`` bars)."""
    spec = ASSETS[asset]
    csv = _REPO / spec["path"]
    df = load_stream_frame(str(csv)).iloc[-N_BARS:].copy()
    stream = Stream(ticker=asset, timeframe=spec["timeframe"],
                    path=str(csv), cluster_id=spec["cluster_id"])
    sd = StreamData(stream=stream, df=df, bar_seconds=86400.0)
    folds = build_calendar_folds([sd], oos_fraction=OOS_FRACTION_GOLDEN)[:N_FOLDS]
    return folds, [stream]


def representative_slices(
    folds: list[Fold],
) -> Iterator[tuple[str, pd.DataFrame, object]]:
    """Representative (tag, df, artifacts) slices: fold0 IS/OOS + last OOS."""
    first, last = folds[0][0], folds[-1][0]
    yield "fold0_is", first.df_is, first.artifacts_is
    yield "fold0_oos", first.df_oos, first.artifacts_oos
    yield f"fold{len(folds) - 1}_oos", last.df_oos, last.artifacts_oos


def capture_asset(asset: str) -> tuple[dict[str, np.ndarray], dict]:
    """Golden signal arrays + per-fold scores + final pooled LCB for one asset."""
    np.random.seed(SEED)
    folds, streams = build_folds(asset)
    base = Params()

    arrays: dict[str, np.ndarray] = {}
    for tag, df, art in representative_slices(folds):
        res = SpeculatorDetector(df, base, art).run()
        arrays[f"{tag}_signal_high"] = res["signal_high"].to_numpy()
        arrays[f"{tag}_signal_low"] = res["signal_low"].to_numpy()

    scores: dict[str, dict] = {}
    for side in ("high", "low"):
        scorer = PooledScorer(folds=folds, streams=streams, side=side)
        fold_scores = [float(s) for s in raw_fold_scores(scorer, base)]
        scores[side] = {"fold_scores": fold_scores,
                        "lcb": float(scorer.score(base))}

    meta = {
        "asset": asset,
        "csv": ASSETS[asset]["path"],
        "n_bars": N_BARS,
        "n_folds": len(folds),
        "oos_fraction": OOS_FRACTION_GOLDEN,
        "scores": scores,
        "array_sha256": {k: hashlib.sha256(v.tobytes()).hexdigest()
                         for k, v in sorted(arrays.items())},
    }
    return arrays, meta


def main() -> None:
    """Capture both assets and write the golden snapshot to GOLDEN_DIR."""
    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    payload: dict = {"seed": SEED, "params": "base-gold Params()", "assets": {}}
    for asset in ASSETS:
        logger.info("capturing golden baseline for %s ...", asset)
        arrays, meta = capture_asset(asset)
        np.savez(GOLDEN_DIR / f"golden_{asset}.npz", **arrays)
        payload["assets"][asset] = meta
        for side in ("high", "low"):
            logger.info("%s/%s: %d fold scores, LCB=%.10f", asset, side,
                        len(meta["scores"][side]["fold_scores"]),
                        meta["scores"][side]["lcb"])
    out = GOLDEN_DIR / "golden_baseline.json"
    out.write_text(json.dumps(payload, indent=2, sort_keys=True))
    logger.info("golden baseline written to %s", GOLDEN_DIR)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    main()
