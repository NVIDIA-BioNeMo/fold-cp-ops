# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

"""Tests for ``fold_cp_ops._internal.compile_time.ln_prologue_layout``.

Everything here runs with no GPU. That is the point of the compile-time/runtime split: the K-tile
choice, the byte accounting and the divisibility rules that decide whether the prologue is CORRECT
are pure arithmetic, so they can be ENUMERATED rather than sampled. The tiled copy needs an MLIR
environment (it emits layout algebra) but still no device.

The strongest gate here is :func:`test_auto_blk_k_is_the_widest_divisor_over_the_whole_range`: it
checks every K from 16 to 4096 against a brute-force answer, which is a different implementation of
the same rule rather than a restatement of this one.
"""

import math

import cutlass
import pytest

from fold_cp_ops._internal.compile_time.ln_prologue_layout import (
    ALLOWED_BLK_K,
    MMA_INST_K_16BIT,
    STATS_THREADS_PER_ROW,
    auto_blk_k,
    ln_affine_bytes,
    mma_inst_tile_k,
    stats_scratch_bytes,
    stats_tiled_copy,
)


# ------------------------------------------------------------------ the K tile


def test_auto_blk_k_is_the_widest_divisor_over_the_whole_range():
    """Enumerated, not sampled: every legal K up to 4096 gets the widest tile that divides it.

    The oracle is a brute-force scan of the same pool, which is a genuinely different computation --
    a bug in the implementation's loop order or early exit would have to be reproduced here to hide.
    """
    for k in range(MMA_INST_K_16BIT, 4097, MMA_INST_K_16BIT):
        expected = max(b for b in ALLOWED_BLK_K if k % b == 0)
        assert auto_blk_k(k) == expected, f"K={k}"


def test_auto_blk_k_returns_the_plain_gemm_default_on_every_trimul_feature_width():
    """128/256/384/512 all pick 64 -- so the fusion tiles exactly like the unfused GEMM.

    This is why the K tile can be a parameter without fragmenting the artifact cache in practice:
    on every width the workflow runs, the default is the one the plain kernel already uses.
    """
    for d in (128, 256, 384, 512):
        assert auto_blk_k(d) == 64, f"D={d}"


@pytest.mark.parametrize("bad", [0, -64, 8, 24, 130], ids=["zero", "negative", "k8", "k24", "k130"])
def test_auto_blk_k_refuses_a_K_no_tile_divides(bad):
    """A K the pool cannot tile raises HERE rather than producing an out-of-range gain index.

    ``8`` is the value worth noting: it satisfies the package's 16-byte alignment floor at 16 bits,
    so it reaches this function looking legitimate. Without the check the front door would build a
    kernel whose last K-tile is partial, and the in-place normalize would read the staged ``(K,)``
    gain past its end -- a wrong answer with no fault.
    """
    with pytest.raises(ValueError, match=r"positive multiple of 16"):
        auto_blk_k(bad)


@pytest.mark.parametrize("blk_k", ALLOWED_BLK_K)
def test_mma_inst_tile_k_reconstructs_the_tile_from_the_atom(blk_k):
    """``atom_K * mma_inst_tile_k(blk_k) == blk_k`` -- the identity the functor's property relies on."""
    assert MMA_INST_K_16BIT * mma_inst_tile_k(blk_k) == blk_k


@pytest.mark.parametrize(
    "bad", [8, 48, 128, 0], ids=["half_atom", "not_in_pool", "too_wide", "zero"]
)
def test_mma_inst_tile_k_refuses_a_tile_outside_the_declared_pool(bad):
    """48 divides the atom's K but is not in the pool, and is refused rather than rounded.

    The distinction matters: rounding would silently hand the caller a different tile from the one
    it asked for, and the tile is a NUMERICAL parameter -- a different tile is a different bit
    pattern, not a different speed.
    """
    with pytest.raises(ValueError, match=r"must be one of"):
        mma_inst_tile_k(bad)


# ------------------------------------------------------------------ the byte accounting


@pytest.mark.parametrize("tile_m", [64, 128, 192, 256])
def test_the_statistics_scratch_is_two_fp32_columns_per_row(tile_m):
    """One ``(mu, rstd)`` pair per row of the CTA tile, in fp32."""
    assert stats_scratch_bytes(tile_m) == tile_m * 8


@pytest.mark.parametrize("gemm_k", [128, 256, 384, 512])
def test_the_affine_reservation_counts_the_bias_whether_or_not_one_is_supplied(gemm_k):
    """Reserving the bias unconditionally is what keeps the two configurations comparable.

    The reservation feeds the pipeline-depth calculation, so a conditional one would make the
    emitted SCHEDULE depend on whether the caller passed a LayerNorm bias -- turning "with bias"
    and "without bias" into two kernels that cannot be compared to each other. The default counts
    both; the explicit ``has_bias=False`` form exists for a caller that does not want that.
    """
    assert ln_affine_bytes(gemm_k) == gemm_k * 8
    assert ln_affine_bytes(gemm_k, has_bias=False) == gemm_k * 4


# ------------------------------------------------------------------ the statistics partition


@pytest.mark.parametrize(
    "tile_m,tile_k,num_threads",
    [(128, 64, 256), (128, 32, 256), (128, 16, 256), (64, 64, 128), (256, 64, 256)],
    ids=["blk64", "blk32", "blk16", "one_wg", "tall_tile"],
)
def test_the_statistics_partition_covers_the_tile_exactly_once(
    mlir_ctx, tile_m, tile_k, num_threads
):
    """The copy's tiler divides the staged tile exactly in BOTH modes, so nothing is left unowned.

    A ``TiledCopy`` covers one TILER per application and repeats it over the tensor, so the property
    that matters is not "the TV layout is as big as the tile" -- it never is -- but that the tile is
    a whole number of tilers. A ragged remainder is the silent failure: the leftover elements are
    dropped from ``Sum x`` (a wrong mean) and never rescaled by the normalize (a raw value
    multiplied against a normalized weight). Both look like plausible numbers.

    The expected tiler is recomputed here from the declared rule rather than read off the object,
    so the test states the invariant instead of echoing the implementation.
    """
    import cutlass.cute as cute

    tc = stats_tiled_copy(cutlass.BFloat16, (tile_m, tile_k), num_threads)
    _, value_shape = tc.layout_tv_tiled.shape
    vec = cute.size(value_shape)
    tiler_m = num_threads // STATS_THREADS_PER_ROW
    tiler_k = STATS_THREADS_PER_ROW * vec
    assert tile_m % tiler_m == 0, f"tile_M={tile_m} is not a whole number of {tiler_m}-row groups"
    assert tile_k % tiler_k == 0, f"tile_K={tile_k} is not a whole number of {tiler_k}-wide tilers"
    assert (tile_m // tiler_m) * (tile_k // tiler_k) * tiler_m * tiler_k == tile_m * tile_k


@pytest.mark.parametrize("tile_k", [64, 32, 16])
def test_the_vector_width_is_the_widest_that_fits_a_threads_column_span(mlir_ctx, tile_k):
    """The width is ``gcd(tile_K / lanes_per_row, 128 // dtype.width)`` -- derived, not guessed.

    Checked against an independent recomputation rather than a frozen literal, so the test states
    the RULE. A width that exceeded a thread's span would read another thread's columns; one that
    exceeded the atom's legal set would be refused by ``tiled_copy_2d`` with a message about bits.
    """
    import cutlass.cute as cute

    tc = stats_tiled_copy(cutlass.BFloat16, (128, tile_k), 256)
    expected = math.gcd(tile_k // STATS_THREADS_PER_ROW, 128 // cutlass.BFloat16.width)
    _, value_shape = tc.layout_tv_tiled.shape
    assert cute.size(value_shape) == expected


@pytest.mark.parametrize(
    "tile_m,tile_k,num_threads,tpr,match",
    [
        (128, 64, 255, 2, r"multiple of threads_per_row"),
        (128, 64, 256, 64, r"must divide a warp"),
        (100, 64, 256, 2, r"multiple of the row-group count"),
        (128, 33, 256, 2, r"multiple of threads_per_row"),
    ],
    ids=["ragged_threads", "non_warp_divisor", "ragged_rows", "ragged_columns"],
)
def test_the_partition_refuses_a_geometry_that_would_leave_elements_unowned(
    mlir_ctx, tile_m, tile_k, num_threads, tpr, match
):
    """Each refused geometry is a silent-wrong-answer condition, so each is checked rather than trusted.

    ``threads_per_row=64`` is the interesting one: it divides the thread count and the tile cleanly,
    so every element still has an owner -- but it does NOT divide a warp, and the per-row combine is
    an intra-warp butterfly that cannot reach a lane in another warp. The reduction would then
    combine a subset of the lanes and produce a mean over part of the row.
    """
    with pytest.raises(AssertionError, match=match):
        stats_tiled_copy(cutlass.BFloat16, (tile_m, tile_k), num_threads, tpr)


def test_two_lanes_per_row_keeps_the_butterfly_inside_a_warp():
    """The declared default divides 32, which is the condition the shuffle-based combine needs."""
    assert 32 % STATS_THREADS_PER_ROW == 0
    assert STATS_THREADS_PER_ROW >= 2, (
        "one lane per row removes the butterfly but makes a single lane read the whole row, "
        "serialising the load it was meant to spread"
    )
