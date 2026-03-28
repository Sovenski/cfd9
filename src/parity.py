"""Pine vs Python parity checks for Speculatores outputs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .detector import SpeculatorDetector
from .indicators import Params


@dataclass(frozen=True)
class ParityMetric:
    name: str
    tv_column: str
    compared_rows: int
    mismatch_rows: int
    mismatch_rate: float
    max_abs_diff: float | None


def find_matching_enriched_export(dataset_path: str | Path) -> Path | None:
    dataset = Path(dataset_path)
    parts = dataset.stem.split("_")
    if len(parts) < 2:
        return None
    prefix = f"{parts[0]}_{parts[1]}_"
    enriched_dir = dataset.parent.parent / "enriched"
    candidates = sorted(enriched_dir.glob(f"{prefix}*_TV_*.csv"))
    return candidates[0] if candidates else None


def _normalize_tv_columns(tv: pd.DataFrame) -> pd.DataFrame:
    out = tv.copy()
    out.columns = [str(col).strip() for col in out.columns]
    if "Strong High" in out.columns or "Regular High" in out.columns:
        out["signal_high"] = (
            out.get("Strong High", 0).fillna(0).astype(float)
            + out.get("Regular High", 0).fillna(0).astype(float)
        ) > 0
    if "Strong Low" in out.columns or "Regular Low" in out.columns:
        out["signal_low"] = (
            out.get("Strong Low", 0).fillna(0).astype(float)
            + out.get("Regular Low", 0).fillna(0).astype(float)
        ) > 0
    return out


def compare_python_to_tv(
    raw_path: str | Path,
    tv_export_path: str | Path,
    params: Params | None = None,
    tolerance: float = 1e-6,
) -> tuple[pd.DataFrame, list[ParityMetric]]:
    raw_df = pd.read_csv(raw_path)
    raw_df.columns = [str(col).strip().lower() for col in raw_df.columns]
    if "time" not in raw_df.columns:
        raise ValueError("raw dataset must include a 'time' column")

    tv_df = _normalize_tv_columns(pd.read_csv(tv_export_path))
    if "time" not in tv_df.columns:
        raise ValueError("TradingView export must include a 'time' column")

    py_result = SpeculatorDetector(
        raw_df.copy(),
        params or Params(),
        include_debug_columns=True,
    ).run()
    py_result = py_result.copy()
    py_result["time"] = raw_df["time"].values

    merged = py_result.merge(tv_df, on="time", how="inner", suffixes=("_py", "_tv"))
    metrics: list[ParityMetric] = []

    candidates = {
        "signal_high": ["signal_high"],
        "signal_low": ["signal_low"],
        "agreement_high_side": ["agreement_high_side", "agreement_high"],
        "agreement_low_side": ["agreement_low_side", "agreement_low"],
        "momentum_velocity_high": ["momentum_velocity_high"],
        "momentum_velocity_low": ["momentum_velocity_low"],
        "pivot_drift_high": ["pivot_drift_high"],
        "pivot_drift_low": ["pivot_drift_low"],
        "ph_confirms": ["ph_confirms"],
        "pl_confirms": ["pl_confirms"],
        "price_gate_high": ["price_gate_high"],
        "price_gate_low": ["price_gate_low"],
        "er_val_high": ["er_val_high", "er_val"],
        "er_val_low": ["er_val_low"],
        "baseline_pivot_high": ["baseline_pivot_high"],
        "baseline_pivot_low": ["baseline_pivot_low"],
    }

    for py_col, tv_candidates in candidates.items():
        py_merged_col = py_col if py_col in merged.columns else f"{py_col}_py"
        if py_merged_col not in merged.columns:
            continue
        tv_col = next(
            (
                col
                for cand in tv_candidates
                for col in (cand, f"{cand}_tv")
                if col in merged.columns
            ),
            None,
        )
        if tv_col is None:
            continue

        py_vals = merged[py_merged_col]
        tv_vals = merged[tv_col]
        valid = ~(py_vals.isna() | tv_vals.isna())
        compared_rows = int(valid.sum())
        if compared_rows == 0:
            metrics.append(
                ParityMetric(py_col, tv_col, 0, 0, 0.0, None)
            )
            continue

        py_valid = py_vals[valid]
        tv_valid = tv_vals[valid]
        if py_valid.dtype == bool or tv_valid.dtype == bool:
            mismatches = (py_valid.astype(bool) != tv_valid.astype(bool))
            max_abs_diff = None
        else:
            diffs = (py_valid.astype(float) - tv_valid.astype(float)).abs()
            mismatches = diffs > tolerance
            max_abs_diff = float(diffs.max()) if len(diffs) else None
        mismatch_rows = int(mismatches.sum())
        metrics.append(
            ParityMetric(
                name=py_col,
                tv_column=tv_col,
                compared_rows=compared_rows,
                mismatch_rows=mismatch_rows,
                mismatch_rate=(mismatch_rows / compared_rows) if compared_rows else 0.0,
                max_abs_diff=max_abs_diff,
            )
        )

    summary_df = pd.DataFrame(
        [
            {
                "python_metric": metric.name,
                "tv_metric": metric.tv_column,
                "compared_rows": metric.compared_rows,
                "mismatch_rows": metric.mismatch_rows,
                "mismatch_rate": round(metric.mismatch_rate, 6),
                "max_abs_diff": None if metric.max_abs_diff is None else round(metric.max_abs_diff, 8),
            }
            for metric in metrics
        ]
    )
    return summary_df, metrics
