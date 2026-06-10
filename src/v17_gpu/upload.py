"""Slice -> device padded packer (build-spec §5.2, invariants P1/P7).

Pure data marshaling for the batched GPU evaluator:

- ``pack_bars`` / ``unpack_bars``: per-asset bar-major arrays into one padded
  ``[n_assets, max_bars, ...]`` batch and back, BITWISE exact. The packer
  never casts — PIR stays float32, every other feature stays float64 (P1).
- ``valid_mask`` ``[n_assets, max_bars]`` marks exactly the real bars; the
  §5.4 segmented scan uses it to FREEZE counters/pivot appends on pad bars
  and RESET carry at asset boundaries (P7).
- ``bucket_by_length``: length-bucketing so padding waste per batch is
  bounded instead of paying ``max_bars`` for every short OOS slice.
- ``pack_pir_tile`` / ``iter_pir_tiles``: scale-tiling hook for the PIR
  matrix (499 scales x ~57k bars x float32 per asset is HBM-infeasible in
  one piece — P1 memory discipline). Tiles are float32, padded with NaN to
  match ``indicators.precompute_matrices``'s own undefined-bar convention.
- ``to_torch``: upload hook (``torch.from_numpy`` — zero-copy on CPU),
  preserving dtype 1:1 (float32 -> torch.float32, float64 -> torch.float64).
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any, Iterator, Sequence

import numpy as np

logger = logging.getLogger(__name__)

__all__ = [
    "PackedBatch",
    "LengthBucket",
    "valid_mask_from_lengths",
    "pack_bars",
    "unpack_bars",
    "bucket_by_length",
    "pack_pir_tile",
    "iter_pir_tiles",
    "to_torch",
]


@dataclass(frozen=True)
class PackedBatch:
    """One padded batch of per-asset arrays.

    Args:
        data: padded values; bar axis is axis 1 for bar-major packs
            (``[n_assets, max_bars, ...]``) and axis 2 for PIR tiles
            (``[n_assets, n_tile_scales, max_bars]``).
        valid_mask: bool ``[n_assets, max_bars]`` — True exactly on real bars.
        lengths: int64 ``[n_assets]`` true bar counts.
    """

    data: np.ndarray
    valid_mask: np.ndarray
    lengths: np.ndarray

    @property
    def n_assets(self) -> int:
        return int(self.lengths.shape[0])

    @property
    def max_bars(self) -> int:
        return int(self.valid_mask.shape[1]) if self.valid_mask.ndim == 2 else 0


@dataclass(frozen=True)
class LengthBucket:
    """One length bucket: asset indices sharing a padded width."""

    indices: np.ndarray  # int64 positions into the original asset order
    max_bars: int


def valid_mask_from_lengths(lengths: np.ndarray, max_bars: int) -> np.ndarray:
    """Bool ``[n_assets, max_bars]`` mask, True exactly on the real bars (P7)."""
    lengths = np.asarray(lengths, dtype=np.int64)
    return np.arange(max_bars, dtype=np.int64)[None, :] < lengths[:, None]


def _check_uniform(arrays: Sequence[np.ndarray]) -> None:
    """All arrays must share dtype and trailing (non-bar) shape."""
    first = arrays[0]
    for a in arrays[1:]:
        if a.dtype != first.dtype:
            raise ValueError(
                f"mixed dtypes in pack: {a.dtype} vs {first.dtype} — the packer "
                "never casts (P1); pack float32 PIR and float64 features separately"
            )
        if a.shape[1:] != first.shape[1:]:
            raise ValueError(
                f"mixed trailing shapes in pack: {a.shape[1:]} vs {first.shape[1:]}"
            )


def _check_pad_value(dtype: np.dtype, pad_value: float) -> None:
    if isinstance(pad_value, float) and math.isnan(pad_value):
        if not np.issubdtype(dtype, np.floating):
            raise ValueError(f"NaN pad_value is invalid for dtype {dtype}")


def pack_bars(
    arrays: Sequence[np.ndarray], pad_value: float = 0.0
) -> PackedBatch:
    """Pack per-asset bar-major arrays into one padded batch.

    Args:
        arrays: one array per asset, shape ``[n_bars]`` or ``[n_bars, ...]``;
            all must share dtype and trailing shape (no silent casts — P1).
        pad_value: fill for pad bars (NaN allowed for float dtypes only).

    Returns:
        ``PackedBatch`` with ``data [n_assets, max_bars, ...]`` in the
        common dtype, plus ``valid_mask`` and ``lengths``.
    """
    if not arrays:
        return PackedBatch(
            data=np.zeros((0, 0), dtype=np.float64),
            valid_mask=np.zeros((0, 0), dtype=bool),
            lengths=np.zeros(0, dtype=np.int64),
        )
    _check_uniform(arrays)
    dtype = arrays[0].dtype
    _check_pad_value(dtype, pad_value)
    lengths = np.asarray([a.shape[0] for a in arrays], dtype=np.int64)
    max_bars = int(lengths.max())
    data = np.full((len(arrays), max_bars, *arrays[0].shape[1:]), pad_value, dtype=dtype)
    for i, a in enumerate(arrays):
        data[i, : a.shape[0]] = a
    logger.debug(
        "packed %d assets, max_bars=%d, dtype=%s, pad=%r",
        len(arrays), max_bars, dtype, pad_value,
    )
    return PackedBatch(
        data=data,
        valid_mask=valid_mask_from_lengths(lengths, max_bars),
        lengths=lengths,
    )


def unpack_bars(batch: PackedBatch) -> list[np.ndarray]:
    """Invert ``pack_bars``: per-asset arrays, bitwise equal to the sources."""
    return [
        np.ascontiguousarray(batch.data[i, : int(n)])
        for i, n in enumerate(batch.lengths)
    ]


def bucket_by_length(
    lengths: np.ndarray, max_pad_ratio: float = 0.25
) -> list[LengthBucket]:
    """Partition assets into buckets with bounded padding waste.

    Greedy over assets sorted by descending length: a bucket is anchored at
    its longest member; an asset joins while its pad fraction
    ``1 - n / bucket_max`` stays ``<= max_pad_ratio``.

    Args:
        lengths: int bar counts per asset (original order defines indices).
        max_pad_ratio: max tolerated pad fraction per asset within a bucket
            (0.0 -> one bucket per distinct length).

    Returns:
        Buckets covering every index exactly once.
    """
    lengths = np.asarray(lengths, dtype=np.int64)
    if lengths.size == 0:
        return []
    order = np.argsort(lengths)[::-1].astype(np.int64)  # descending
    buckets: list[LengthBucket] = []
    current: list[int] = []
    bucket_max = int(lengths[order[0]])
    for idx in order:
        n = int(lengths[idx])
        if current and (1.0 - n / bucket_max) > max_pad_ratio:
            buckets.append(
                LengthBucket(indices=np.asarray(current, dtype=np.int64), max_bars=bucket_max)
            )
            current = []
            bucket_max = n
        if not current:
            bucket_max = n
        current.append(int(idx))
    buckets.append(
        LengthBucket(indices=np.asarray(current, dtype=np.int64), max_bars=bucket_max)
    )
    logger.debug(
        "bucketed %d assets into %d buckets (max_pad_ratio=%.2f)",
        lengths.size, len(buckets), max_pad_ratio,
    )
    return buckets


def pack_pir_tile(
    pir_list: Sequence[np.ndarray], scale_lo: int, scale_hi: int
) -> PackedBatch:
    """Pack ONE scale tile of the per-asset PIR matrices (P1 memory hook).

    Args:
        pir_list: per-asset PIR matrices, float32 ``[n_scales, n_bars]``
            from ``indicators.precompute_matrices`` (n_scales identical
            across assets; n_bars may differ).
        scale_lo: tile start (row index into the scale axis, inclusive).
        scale_hi: tile end (exclusive).

    Returns:
        ``PackedBatch`` with float32 ``data [n_assets, scale_hi-scale_lo,
        max_bars]``; pad bars are NaN (the matrix's own undefined-bar value).
    """
    if not pir_list:
        raise ValueError("pack_pir_tile needs at least one asset")
    n_scales = pir_list[0].shape[0]
    for p in pir_list[1:]:
        if p.shape[0] != n_scales:
            raise ValueError(
                f"inconsistent scale counts across assets: {p.shape[0]} vs {n_scales}"
            )
    if not (0 <= scale_lo < scale_hi <= n_scales):
        raise ValueError(
            f"bad tile [{scale_lo}, {scale_hi}) for n_scales={n_scales}"
        )
    dtype = pir_list[0].dtype
    lengths = np.asarray([p.shape[1] for p in pir_list], dtype=np.int64)
    max_bars = int(lengths.max())
    data = np.full(
        (len(pir_list), scale_hi - scale_lo, max_bars), np.nan, dtype=dtype
    )
    for i, p in enumerate(pir_list):
        data[i, :, : p.shape[1]] = p[scale_lo:scale_hi]
    return PackedBatch(
        data=data,
        valid_mask=valid_mask_from_lengths(lengths, max_bars),
        lengths=lengths,
    )


def iter_pir_tiles(
    pir_list: Sequence[np.ndarray], tile_scales: int
) -> Iterator[tuple[tuple[int, int], PackedBatch]]:
    """Yield ``((scale_lo, scale_hi), tile)`` covering the full scale axis.

    Never materialises the full ``[n_assets, n_scales, max_bars]`` tensor —
    the caller consumes one tile at a time (scale-tiling, P1 memory).
    """
    if tile_scales < 1:
        raise ValueError(f"tile_scales must be >= 1, got {tile_scales}")
    if not pir_list:
        return
    n_scales = pir_list[0].shape[0]
    for lo in range(0, n_scales, tile_scales):
        hi = min(lo + tile_scales, n_scales)
        yield (lo, hi), pack_pir_tile(pir_list, lo, hi)


def to_torch(batch: PackedBatch, device: str = "cpu") -> tuple[Any, Any]:
    """Upload a packed batch: ``(data, valid_mask)`` torch tensors.

    dtype maps 1:1 (float32 -> torch.float32, float64 -> torch.float64,
    bool -> torch.bool) — no silent promotion (P1). ``torch.from_numpy`` is
    zero-copy; ``.to(device)`` copies only when leaving the CPU.
    """
    import torch  # local import: numpy-only callers never need torch

    data = torch.from_numpy(batch.data).to(device)
    mask = torch.from_numpy(batch.valid_mask).to(device)
    return data, mask
