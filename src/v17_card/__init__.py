"""Signal Card package (spec FIXED v3 §3 — scorer/calibration/report side).

Modules:
    survival       — right-span R, KM with right-censoring, F6 bootstrap bands
    conditioning   — F1 L-clamped live conditioning + §3.4 E[hold] (R4 lookup)
    expected_move  — F2 two-fold c_side fit, R² fallback, §3.6 conviction
    stop_rule      — F8 stop truth table as pure functions (R2 k-origin)
    grading        — §3.7 retrospective grading + R-multiple backtest
    calibration    — §5 run-payload orchestration (scorer "v5" fields)
    gpu_memory     — §6 L4 estimate + F9 deterministic chunk formula

Detection math is FROZEN (spec §0): nothing here touches signal
generation; golden signal arrays stay byte-identical.
"""
from .calibration import (
    C_SIDE_BIAS_NOTE,
    GRID_FLOOR_BIAS_NOTE,
    IN_SAMPLE_DISCLAIMER,
    STOP_RULE,
    build_signal_records,
    calibrate_for_run,
    calibrate_run,
)
from .conditioning import (
    HOLD_CAP,
    NOISE_FLOOR,
    conditional_survival,
    expected_hold,
    survival_lookup,
)
from .expected_move import (
    MIN_FIT_PAIRS,
    R2_FLOOR,
    CSideFit,
    SignalRecord,
    conviction_percentile,
    expected_move,
    fit_c_side,
)
from .gpu_memory import (
    BUDGET_BYTES,
    BYTES_PER_CANDIDATE_BAR,
    assert_probe_agreement,
    chunk_size,
    chunked_score_pop,
    estimate_from_folds,
    gpu_memory_estimate,
)
from .grading import SignalGrade, backtest_summary, grade_signals
from .stop_rule import (
    StopState,
    init_stop_state,
    stop_series,
    track_signal,
    update_stop_state,
)
from .survival import (
    ClusterData,
    KMSurvival,
    SurvivalBands,
    cluster_bootstrap_bands,
    fit_km,
    km_at,
    km_table,
    right_span,
)

__all__ = [
    # calibration orchestration
    "IN_SAMPLE_DISCLAIMER", "GRID_FLOOR_BIAS_NOTE", "C_SIDE_BIAS_NOTE",
    "STOP_RULE", "build_signal_records", "calibrate_run", "calibrate_for_run",
    # gpu memory (§6 / F9)
    "BUDGET_BYTES", "BYTES_PER_CANDIDATE_BAR", "gpu_memory_estimate",
    "estimate_from_folds", "chunk_size", "assert_probe_agreement",
    "chunked_score_pop",
    # survival
    "ClusterData", "KMSurvival", "SurvivalBands",
    "right_span", "fit_km", "km_at", "km_table", "cluster_bootstrap_bands",
    # conditioning
    "NOISE_FLOOR", "HOLD_CAP",
    "survival_lookup", "conditional_survival", "expected_hold",
    # expected move / conviction
    "R2_FLOOR", "MIN_FIT_PAIRS", "SignalRecord", "CSideFit",
    "fit_c_side", "expected_move", "conviction_percentile",
    # stop rule
    "StopState", "init_stop_state", "update_stop_state",
    "track_signal", "stop_series",
    # grading / backtest
    "SignalGrade", "grade_signals", "backtest_summary",
]
