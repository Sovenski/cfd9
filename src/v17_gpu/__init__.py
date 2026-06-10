"""GPU batched evaluator package (build-spec §5).

Sub-modules are built in gated order; each is parity-tested against the CPU
oracle (``SpeculatorDetector`` / ``FastDetector``) before the next is added.
"""
from .drift_precompute import (
    DriftSpec,
    confirmed_pivot_events,
    drift_per_bar,
    precompute_drift,
    precompute_drift_batch,
)

try:  # torch-dependent evaluator; numpy-only callers can still import the package
    from .eval_torch import (
        Phase1Features,
        TorchPhase1,
        build_pir_matrix_torch,
        compute_phase1_features,
        edge_or_state_torch,
        to_bool_torch,
    )
    from .phase2_scan import (
        GpuPooledScorer,
        LaneInputs,
        batched_signals,
        candidate_lane_inputs,
        scan_phase2,
        signals_torch,
    )
except ImportError:  # pragma: no cover - torch missing
    pass
from .upload import (
    LengthBucket,
    PackedBatch,
    bucket_by_length,
    iter_pir_tiles,
    pack_bars,
    pack_pir_tile,
    to_torch,
    unpack_bars,
    valid_mask_from_lengths,
)

__all__ = [
    "Phase1Features",
    "TorchPhase1",
    "GpuPooledScorer",
    "LaneInputs",
    "batched_signals",
    "candidate_lane_inputs",
    "scan_phase2",
    "signals_torch",
    "build_pir_matrix_torch",
    "compute_phase1_features",
    "edge_or_state_torch",
    "to_bool_torch",
    "DriftSpec",
    "confirmed_pivot_events",
    "drift_per_bar",
    "precompute_drift",
    "precompute_drift_batch",
    "LengthBucket",
    "PackedBatch",
    "bucket_by_length",
    "iter_pir_tiles",
    "pack_bars",
    "pack_pir_tile",
    "to_torch",
    "unpack_bars",
    "valid_mask_from_lengths",
]
