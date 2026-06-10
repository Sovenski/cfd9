"""Parity spec for the GPU padded packer (build-spec §5.2, invariants P1/P7).

``src/v17_gpu/upload.py`` is pure data marshaling: slice arrays are packed
into padded ``[n_assets, max_bars, ...]`` batches plus a ``valid_mask``.
Legitimacy conditions:

- Round-trip exactness: packing then per-asset unpacking reproduces the
  source arrays BITWISE (``np.array_equal``) — marshaling must never touch
  a value the threshold layer will compare (P1).
- ``valid_mask`` marks exactly the real bars (P7's freeze/reset signal for
  the segmented scan).
- dtype discipline: PIR stays float32, every other feature stays float64;
  the packer never casts silently (P1).
- Length-bucketing partitions the assets with bounded padding waste.
- PIR scale-tiling reconstructs the full per-asset matrix exactly without
  ever materialising ``[n_assets, 499, max_bars]`` in one piece (P1 memory).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

gpumod = pytest.importorskip("src.v17_gpu.upload")  # RED until module exists
PackedBatch = gpumod.PackedBatch
LengthBucket = gpumod.LengthBucket
valid_mask_from_lengths = gpumod.valid_mask_from_lengths
pack_bars = gpumod.pack_bars
unpack_bars = gpumod.unpack_bars
bucket_by_length = gpumod.bucket_by_length
pack_pir_tile = gpumod.pack_pir_tile
iter_pir_tiles = gpumod.iter_pir_tiles
to_torch = gpumod.to_torch

_CSV = Path("data/raw/SPX_1D_20170428_20260318.csv")


def _rand_streams(
    lengths: list[int], trailing: tuple[int, ...] = (), dtype: type = np.float64
) -> list[np.ndarray]:
    rng = np.random.default_rng(123)
    return [
        rng.normal(0.0, 1.0, size=(n, *trailing)).astype(dtype) for n in lengths
    ]


# ---------------------------------------------------------------------------
# Round-trip exactness + valid_mask (P7)
# ---------------------------------------------------------------------------


def test_pack_unpack_roundtrip_1d_is_bitwise_exact() -> None:
    arrays = _rand_streams([37, 512, 5, 300])
    batch = pack_bars(arrays)
    assert batch.data.shape == (4, 512)
    assert batch.data.dtype == np.float64
    out = unpack_bars(batch)
    assert len(out) == len(arrays)
    for src, dst in zip(arrays, out):
        assert dst.dtype == src.dtype
        assert np.array_equal(dst, src)


def test_pack_unpack_roundtrip_trailing_feature_axis() -> None:
    arrays = _rand_streams([40, 7, 19], trailing=(3,))
    batch = pack_bars(arrays)
    assert batch.data.shape == (3, 40, 3)
    for src, dst in zip(arrays, unpack_bars(batch)):
        assert np.array_equal(dst, src)


def test_valid_mask_marks_exactly_the_real_bars() -> None:
    lengths = [37, 512, 5, 300]
    batch = pack_bars(_rand_streams(lengths))
    assert batch.valid_mask.dtype == np.bool_
    assert batch.valid_mask.shape == (4, 512)
    for i, n in enumerate(lengths):
        assert batch.valid_mask[i, :n].all()
        assert not batch.valid_mask[i, n:].any()
    assert np.array_equal(batch.valid_mask.sum(axis=1), np.asarray(lengths))
    assert np.array_equal(batch.lengths, np.asarray(lengths, dtype=np.int64))
    # standalone helper agrees
    assert np.array_equal(
        valid_mask_from_lengths(np.asarray(lengths, dtype=np.int64), 512),
        batch.valid_mask,
    )


def test_pad_region_carries_pad_value_and_does_not_alias_real_bars() -> None:
    arrays = _rand_streams([10, 25])
    batch = pack_bars(arrays, pad_value=0.0)
    assert np.all(batch.data[0, 10:] == 0.0)
    nan_batch = pack_bars(arrays, pad_value=np.nan)
    assert np.isnan(nan_batch.data[0, 10:]).all()
    # real bars are untouched by the pad choice
    assert np.array_equal(nan_batch.data[0, :10], arrays[0])


# ---------------------------------------------------------------------------
# dtype discipline (P1)
# ---------------------------------------------------------------------------


def test_dtype_is_preserved_never_cast() -> None:
    f32 = pack_bars(_rand_streams([20, 30], dtype=np.float32))
    assert f32.data.dtype == np.float32  # PIR path stays float32 (P1)
    f64 = pack_bars(_rand_streams([20, 30], dtype=np.float64))
    assert f64.data.dtype == np.float64  # feature path stays float64 (P1)
    flags = pack_bars([np.ones(8, dtype=bool), np.zeros(3, dtype=bool)])
    assert flags.data.dtype == np.bool_
    assert not flags.data[1, 3:].any()  # bool pad is False


def test_pack_rejects_mixed_dtypes_shapes_and_bad_pads() -> None:
    with pytest.raises(ValueError):
        pack_bars([np.zeros(5, dtype=np.float32), np.zeros(5, dtype=np.float64)])
    with pytest.raises(ValueError):
        pack_bars([np.zeros((5, 2)), np.zeros((5, 3))])
    with pytest.raises(ValueError):
        pack_bars([np.zeros(5, dtype=bool)], pad_value=np.nan)  # NaN into bool
    assert pack_bars([]).data.shape == (0, 0)


# ---------------------------------------------------------------------------
# Length-bucketing
# ---------------------------------------------------------------------------


def test_bucket_by_length_is_an_exact_partition() -> None:
    lengths = np.asarray([5700, 401, 5650, 420, 2400, 410, 5500], dtype=np.int64)
    buckets = bucket_by_length(lengths, max_pad_ratio=0.25)
    seen = np.concatenate([b.indices for b in buckets])
    assert sorted(seen.tolist()) == list(range(len(lengths)))
    assert len(seen) == len(set(seen.tolist()))
    for b in buckets:
        assert b.max_bars == int(lengths[b.indices].max())


def test_bucket_padding_waste_is_bounded() -> None:
    rng = np.random.default_rng(42)
    lengths = rng.integers(401, 5701, size=64).astype(np.int64)
    ratio = 0.25
    buckets = bucket_by_length(lengths, max_pad_ratio=ratio)
    for b in buckets:
        per_asset_waste = 1.0 - lengths[b.indices] / float(b.max_bars)
        assert float(per_asset_waste.max()) <= ratio + 1e-12
    # degenerate ratio -> one bucket per distinct length, still a partition
    tight = bucket_by_length(lengths, max_pad_ratio=0.0)
    assert sum(len(b.indices) for b in tight) == len(lengths)
    for b in tight:
        assert np.all(lengths[b.indices] == b.max_bars)


# ---------------------------------------------------------------------------
# PIR scale-tiling (P1 memory discipline; float32 preserved bitwise)
# ---------------------------------------------------------------------------


def _real_pir_matrices() -> list[np.ndarray]:
    if not _CSV.exists():
        pytest.skip(f"missing {_CSV}")
    from src.indicators import precompute_matrices
    from src.pooled_validation import load_stream_frame

    df = load_stream_frame(str(_CSV))
    out: list[np.ndarray] = []
    for n in (300, 420):
        close = df["close"].iloc[:n].reset_index(drop=True)
        _, pir, _ = precompute_matrices(close, 2, 60)
        out.append(pir)
    return out


def test_pir_tiles_reconstruct_real_matrices_bitwise() -> None:
    pir_list = _real_pir_matrices()
    n_scales = pir_list[0].shape[0]
    tile_scales = 16
    rebuilt = [
        np.empty_like(p) for p in pir_list
    ]  # reassemble from tiles, then compare bitwise
    covered: list[int] = []
    for (lo, hi), tile in iter_pir_tiles(pir_list, tile_scales=tile_scales):
        assert hi - lo <= tile_scales
        assert tile.data.dtype == np.float32  # P1: PIR stays float32
        assert tile.data.shape == (len(pir_list), hi - lo, 420)
        covered.extend(range(lo, hi))
        for i, p in enumerate(pir_list):
            n = p.shape[1]
            rebuilt[i][lo:hi] = tile.data[i, :, :n]
            # pad region beyond the asset's real bars is NaN, mask is False
            assert np.isnan(tile.data[i, :, n:]).all()
            assert not tile.valid_mask[i, n:].any()
            assert tile.valid_mask[i, :n].all()
    assert covered == list(range(n_scales))
    for src, dst in zip(pir_list, rebuilt):
        assert np.array_equal(dst, src, equal_nan=True)


def test_pack_pir_tile_single_call_matches_iterator() -> None:
    pir_list = _real_pir_matrices()
    tile = pack_pir_tile(pir_list, 3, 11)
    it_tile = None
    for (lo, hi), t in iter_pir_tiles(pir_list, tile_scales=8):
        if lo == 0:
            continue
        if (lo, hi) == (8, 16):
            it_tile = t
    assert it_tile is not None
    assert np.array_equal(tile.data[:, 5:8], it_tile.data[:, 0:3], equal_nan=True)
    with pytest.raises(ValueError):
        pack_pir_tile(pir_list, 5, 5)  # empty tile
    with pytest.raises(ValueError):
        pack_pir_tile(
            [pir_list[0], pir_list[1][:-1]], 0, 4
        )  # inconsistent scale count


# ---------------------------------------------------------------------------
# Torch upload hook (zero-copy view on CPU; dtypes map 1:1)
# ---------------------------------------------------------------------------


def test_to_torch_preserves_dtype_and_values() -> None:
    torch = pytest.importorskip("torch")
    arrays = _rand_streams([50, 33])
    batch = pack_bars(arrays)
    data_t, mask_t = to_torch(batch, device="cpu")
    assert data_t.dtype == torch.float64
    assert mask_t.dtype == torch.bool
    assert np.array_equal(data_t.numpy(), batch.data)
    assert np.array_equal(mask_t.numpy(), batch.valid_mask)

    pir = pack_bars(_rand_streams([50, 33], dtype=np.float32))
    pir_t, _ = to_torch(pir, device="cpu")
    assert pir_t.dtype == torch.float32  # P1: no silent fp32->fp64 promotion
    assert np.array_equal(pir_t.numpy(), pir.data)
