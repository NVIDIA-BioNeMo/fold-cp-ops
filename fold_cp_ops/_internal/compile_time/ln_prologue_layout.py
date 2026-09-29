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

"""Compile-time geometry of the fused-LayerNorm prologue: the K tile, the partition, the bytes.

Everything here is either pure Python arithmetic (byte counts, the K-tile choice) or layout algebra
that the compiler folds away (the statistics tiled copy) -- nothing survives as an instruction. It
lives apart from :mod:`fold_cp_ops._internal.ln_prologue`, which is the half that DOES emit
instructions (the reductions, the in-place normalize), so a reader can tell at a glance which half a
change lands in.

**Why the K tile is a compile-time decision and not a shape.** The mainloop's k-loop trip count is
``K / BLK_K``, so BLK_K sets how the per-row ``Sum x`` / ``Sum x^2`` partials and the WGMMA
accumulator are ASSOCIATED. Two runs with different BLK_K compute the same value in exact arithmetic
and different bit patterns in floating point. That makes BLK_K a tile-geometry parameter of the
functor -- keyed like ``tile_M``/``tile_N`` -- rather than something derived from the runtime K
extent. The kernel stays dynamic in K; only the tiling is fixed. :func:`auto_blk_k` is the front
door's default policy, matching what the upstream physical-fusion kernel baked in, so a caller who
does not think about BLK_K reproduces the upstream bit pattern at every K it supports.
"""

import math
from typing import Tuple, Type

import cutlass
import cutlass.cute as cute

from fold_cp_ops._internal.compile_time.copy_descriptors import max_vec_elems, tiled_copy_2d

#: K tiles the mainloop may use, widest first. Each is a multiple of the 16-bit WGMMA atom's K
#: extent (16), so a tile is always a whole number of MMA K-instructions; anything else would leave
#: the atom half-fed. 64 is the plain GEMM's default, which is why it is tried first -- picking it
#: whenever it divides K is what makes the fused kernel's tiling identical to the unfused one's.
ALLOWED_BLK_K = (64, 32, 16)

#: The WGMMA K extent, in elements, for a 16-bit operand. Not configuration: it is the atom's own
#: shape, and every entry of :data:`ALLOWED_BLK_K` is a multiple of it.
MMA_INST_K_16BIT = 16

#: Lanes that cooperate on ONE row of the per-row statistics reduction. Two, so the butterfly that
#: combines them is a single ``shfl.bfly`` inside a warp: at 2 lanes the exchange never crosses a
#: warp boundary, which is what lets the reduction avoid shared memory entirely. Raising it widens
#: the butterfly (more shuffles per row) and narrows the per-thread column span; lowering it to 1
#: removes the butterfly but makes one lane read the whole row, serialising the load.
STATS_THREADS_PER_ROW = 2


def auto_blk_k(gemm_k: int) -> int:
    """The mainloop K tile for a contraction extent, largest first among :data:`ALLOWED_BLK_K`.

    Purpose
        The default BLK_K policy of the fused-LayerNorm prologue, and the reason a caller who never
        thinks about tiling still gets the upstream kernel's exact arithmetic. It is a pure function
        of ``gemm_k`` so a test can enumerate it with no GPU.

    Semantics
        Returns the widest allowed tile that divides ``gemm_k`` EXACTLY. Exact division is the
        point: with a partial last K-tile the in-place normalize would index the staged ``(K,)``
        gain past its end, reading whatever the allocator left there -- a wrong answer, not a fault.
        The caller is responsible for having padded ``gemm_k`` to a multiple of 16 before asking
        (the front door does, by zero-extending the operands); this function refuses rather than
        silently returning a tile that does not divide.

        For every ``gemm_k`` that is a multiple of 64 -- which is every feature width the TriMul
        workflow runs (128, 256, 384, 512) -- the answer is 64, the plain GEMM's default. So the
        narrow tiles are a correctness fallback for unusual K, not a routine choice.

    Args:
        gemm_k: The contraction extent the mainloop will tile, in elements. Must be a positive
            multiple of 16; anything else has no legal tile in the pool and raises here rather than
            producing an out-of-range weight index inside the kernel.

    Returns:
        One of :data:`ALLOWED_BLK_K`, dividing ``gemm_k`` exactly.

    Raises:
        ValueError: If ``gemm_k`` is not a positive multiple of 16.
    """
    if gemm_k <= 0 or gemm_k % MMA_INST_K_16BIT != 0:
        raise ValueError(
            f"the fused-LayerNorm mainloop needs a contraction extent that is a positive multiple "
            f"of {MMA_INST_K_16BIT} (the 16-bit WGMMA atom's K extent); got gemm_k={gemm_k}. Pad "
            f"the operands to ceil(K/{MMA_INST_K_16BIT})*{MMA_INST_K_16BIT} at the front door -- "
            f"a partial last K-tile would index the staged (K,) gain past its end."
        )
    for blk_k in ALLOWED_BLK_K:
        if gemm_k % blk_k == 0:
            return blk_k
    # Unreachable: 16 is in the pool and divides every multiple of 16.
    raise AssertionError(f"no tile in {ALLOWED_BLK_K} divides gemm_k={gemm_k}")


def mma_inst_tile_k(blk_k: int) -> int:
    """WGMMA K-instructions per mainloop K tile, for a 16-bit operand.

    Purpose
        ``GemmSm90`` derives ``cta_tile_k`` as ``atom_K * _MMA_INST_TILE_K``, so this is how a
        requested BLK_K is expressed in the units the base class configures. Keeping the conversion
        here means the functor's property is one line and the arithmetic has one test.

    Args:
        blk_k: The wanted mainloop K tile, in elements. Must be in :data:`ALLOWED_BLK_K`; a value
            outside it either does not divide the atom's K extent (leaving the MMA half-fed) or was
            never measured, so it is refused rather than rounded.

    Returns:
        ``blk_k // 16`` -- 4, 2 or 1.

    Raises:
        ValueError: If ``blk_k`` is not in :data:`ALLOWED_BLK_K`.
    """
    if blk_k not in ALLOWED_BLK_K:
        raise ValueError(
            f"blk_k must be one of {ALLOWED_BLK_K} (each a multiple of the 16-bit WGMMA atom's K "
            f"extent {MMA_INST_K_16BIT}); got {blk_k}."
        )
    return blk_k // MMA_INST_K_16BIT


def stats_scratch_bytes(tile_m: int) -> int:
    """Bytes of the per-CTA ``(tile_M, 2)`` fp32 statistics scratch.

    Purpose
        The prologue publishes one ``(mu, rstd)`` pair per row of the CTA tile through shared
        memory, because the thread that COMPUTES a row's statistics is not the set of threads that
        normalize it. This is that buffer's size, reserved out of the SMEM budget before the
        mainloop pipeline is staged.

    Args:
        tile_m: The CTA tile's M extent, in rows. Must be the tile the kernel is actually built
            with: under-reserving here does not fail at compile, it overruns the dynamic SMEM cap
            at launch as ``cudaErrorInvalidValue``.

    Returns:
        ``tile_m * 2 * 4`` bytes -- two fp32 columns, mean then reciprocal standard deviation.
    """
    return tile_m * 2 * 4


def ln_affine_bytes(gemm_k: int, has_bias: bool = True) -> int:
    """Bytes of the staged ``(K,)`` fp32 LayerNorm gain and bias.

    Purpose
        The gain and bias are read once per K-tile by every normalizing thread, so they are staged
        in shared memory rather than re-read from global. This is the reservation.

    Semantics
        Counts BOTH vectors unconditionally by default, matching the upstream kernel: it allocates
        the bias slot whether or not a bias was supplied, so the SMEM budget -- and therefore the
        chosen pipeline depth, and therefore the emitted schedule -- does not depend on whether the
        caller passed one. Pass ``has_bias=False`` only where that budget equality is not wanted.

    Args:
        gemm_k: The PADDED contraction extent, in elements. It must be the padded value, not the
            true feature width: the normalize indexes the staged vectors by the GLOBAL k of its
            tile, which runs to the padded extent.
        has_bias: Whether to count the bias vector. Defaults True -- see Semantics.

    Returns:
        ``gemm_k * 4 * (2 if has_bias else 1)`` bytes.
    """
    return gemm_k * 4 * (2 if has_bias else 1)


def stats_tiled_copy(
    dtype: Type[cutlass.Numeric],
    tile_shape_mk: Tuple[int, int],
    num_threads: int,
    threads_per_row: int = STATS_THREADS_PER_ROW,
) -> cute.TiledCopy:
    """Build the tiled copy that partitions a staged A tile for the per-row statistics.

    Purpose
        ONE partition drives both passes of the prologue -- the ``Sum x`` / ``Sum x^2`` reduction
        and the in-place normalize -- so that a thread reduces exactly the elements it later
        rescales. Deriving it twice would let the two drift, and the failure mode is a row
        normalized with another row's statistics: a wrong answer with no diagnostic.

    Semantics
        ``threads_per_row`` lanes cooperate on one row, so each thread owns
        ``tile_K / threads_per_row`` columns of ``tile_M / (num_threads / threads_per_row)`` rows.
        The vector width is ``gcd(tile_K / threads_per_row, max_vec_elems(dtype))`` -- the widest
        access that both fits one thread's column span and is a legal atom width.

        Pure layout algebra: it emits ``!cute.tiled_copy`` and nothing that produces a runtime
        value, so the descriptor is folded away and only its effect on addressing survives.

    Args:
        dtype: The staged tile's element type -- the ACTIVATION's type (16-bit here), not fp32.
            The reduction promotes to fp32 after the load; sizing the copy for fp32 would halve the
            access width for nothing.
        tile_shape_mk: ``(tile_M, tile_K)`` of the staged A tile. ``tile_K`` must be divisible by
            ``threads_per_row`` and ``tile_M`` by ``num_threads // threads_per_row``, or the
            partition leaves elements unowned -- silently dropping them from the sum.
        num_threads: Threads cooperating, across ALL the MMA warpgroups that will normalize. Must
            be a multiple of ``threads_per_row``.
        threads_per_row: Lanes per row; defaults to :data:`STATS_THREADS_PER_ROW`. Must divide 32,
            or the butterfly that combines the lanes would cross a warp boundary, where
            ``shfl.bfly`` cannot reach.

    Returns:
        A ``cute.TiledCopy`` over the ``(tile_M, tile_K)`` staged tile.

    Raises:
        AssertionError: For any of the divisibility conditions above. Each is a silent-wrong-answer
            condition rather than a crash, which is why they are checked here rather than trusted.

    Note:
        Emits MLIR, so a caller outside ``@cute.jit`` must supply a Context, a Module and an
        InsertionPoint -- see ``tests/_internal/compile_time/conftest.py``.
    """
    tile_m, tile_k = tile_shape_mk
    assert num_threads % threads_per_row == 0, (
        f"num_threads ({num_threads}) must be a multiple of threads_per_row ({threads_per_row}); "
        f"otherwise the thread layout leaves a partial row group whose rows nobody reduces."
    )
    assert 32 % threads_per_row == 0, (
        f"threads_per_row ({threads_per_row}) must divide a warp (32): the per-row combine is an "
        f"intra-warp butterfly shuffle, which cannot reach a lane in another warp."
    )
    assert tile_m % (num_threads // threads_per_row) == 0, (
        f"tile_M ({tile_m}) must be a multiple of the row-group count "
        f"({num_threads // threads_per_row}), or the last group owns fewer rows than the reduction "
        f"loop iterates over."
    )
    assert tile_k % threads_per_row == 0, (
        f"tile_K ({tile_k}) must be a multiple of threads_per_row ({threads_per_row}), or the "
        f"lanes cooperating on a row do not cover it exactly."
    )
    vec = math.gcd(tile_k // threads_per_row, max_vec_elems(dtype))
    return tiled_copy_2d(dtype, threads_per_row, num_threads, vec)
