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

"""A minimal LayerNorm-prologue kernel: the GPU harness for ``_internal/ln_prologue``'s unit tests.

``out[m, :] = (x[m, :] - mu_m) * rstd_m * gain + bias`` computed EXACTLY the way the fused
dual-gated GEMM computes it -- same partition, same two passes, same shared-memory scratch -- with
the WGMMA, the epilogue and the pipeline removed.

**Why a separate kernel rather than testing through the fused GEMM.** The prologue's four helpers
are the fused GEMM's dependencies, so testing them *through* it would make the unit test circular:
a helper bug and a mainloop bug would be indistinguishable, and the tolerance needed to compare a
gated bf16 product would swallow a statistics error outright. Here the oracle is
``torch.nn.functional.layer_norm``, so a failure localizes to the helper and can be gated tightly.

**The two-pass structure is reproduced deliberately, not simplified.** Pass 1 walks every k-tile
accumulating ``Sum x`` / ``Sum x^2``; pass 2 walks them again and rewrites each in place. That is
what exercises the thing most likely to be wrong -- that a thread rescales the rows whose statistics
it helped compute -- which a single-tile harness could not reach.

**This harness compiles WITHOUT tvm-ffi, deliberately.** The package's benchmark path uses
``cute.compile(..., options="--enable-tvm-ffi")`` because the raw ``from_dlpack``-per-call path adds
a fixed host launch tax that would mask kernel speed. Nothing here is ever timed -- every test in
``test_ln_prologue.py`` compares VALUES against a torch oracle -- so the tax buys simplicity at no
cost to any measurement. The fused kernel this harness stands in for does go through the tvm-ffi
entry, and its perf gate is ``tests/perf/``.

Input requirements are on :func:`run_ln_prologue`; the kernel itself is not called directly.
"""

from typing import Optional

import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import torch
from cutlass import Float32, Int32, const_expr
from cutlass.cute.runtime import from_dlpack

from fold_cp_ops._internal.compile_time.ln_prologue_layout import (
    STATS_THREADS_PER_ROW,
    stats_tiled_copy,
)
from fold_cp_ops._internal.compile_time.cute_dsl_utils import torch2cute_dtype_map
from fold_cp_ops._internal.ln_prologue import (
    accumulate_row_stats,
    finalize_row_stats,
    make_row_stat_partials,
    normalize_tile,
    stage_ln_affine,
)

#: The named barrier this harness rendezvous on. Any id but 0 will do -- 0 is ``__syncthreads``'s.
_BARRIER_ID = 1


@cute.kernel
def _ln_prologue_kernel(
    mX: cute.Tensor,  # (M, K)
    mOut: cute.Tensor,  # (M, K)
    mWeight: cute.Tensor,  # (K,) fp32
    mBias: Optional[cute.Tensor],  # (K,) fp32 or None
    mMu: Optional[cute.Tensor],  # (M,) fp32 or None -- the published mean, for inspection
    mRstd: Optional[cute.Tensor],  # (M,) fp32 or None
    eps: Float32,
    tile_m: cutlass.Constexpr[int],
    tile_k: cutlass.Constexpr[int],
    gemm_k: cutlass.Constexpr[int],
    k_real: cutlass.Constexpr[int],
    num_threads: cutlass.Constexpr[int],
    x_dtype: cutlass.Constexpr,
):
    """One CTA per ``tile_m``-row block: reduce over the row, then normalize it in place.

    Semantics
        Allocates one ``(tile_m, tile_k)`` staging tile -- ONE, not a pipeline -- plus the
        ``(tile_m, 2)`` statistics scratch and the staged ``(gemm_k,)`` gain/bias, then runs the
        production sequence: stage the affine terms, walk the k-tiles accumulating, finalize, walk
        them again normalizing, and copy each normalized tile out.

        The barriers around the staging copy are the harness's own, not the helpers': in production
        the mainloop's pipeline provides that ordering, and a test that omitted it would read a
        half-written tile.

    Args:
        mX: The ``(M, K)`` activation. ``M`` must be a multiple of ``tile_m`` and ``K`` equal to
            ``k_real``; the harness does not predicate, so a ragged shape reads out of bounds.
        mOut: The ``(M, K)`` destination, written in full.
        mWeight: The ``(k_real,)`` fp32 gain.
        mBias: The ``(k_real,)`` fp32 bias, or None.
        mMu: Optional ``(M,)`` fp32 destination for the published means, so a test can gate the
            statistics directly rather than only through their effect.
        mRstd: Optional ``(M,)`` destination for the reciprocal standard deviations.
        eps: The variance floor.
        tile_m: Rows per CTA.
        tile_k: Columns per k-tile. Must divide ``gemm_k``.
        gemm_k: The padded contraction extent -- the staged gain's length.
        k_real: The true feature width -- the LayerNorm normalizer.
        num_threads: Threads per CTA.
        x_dtype: The activation's element type.

    Returns:
        None. Its effects are ``mOut`` and, when given, ``mMu``/``mRstd``.
    """
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    barrier = pipeline.NamedBarrier(barrier_id=_BARRIER_ID, num_threads=num_threads)

    smem = cutlass.utils.SmemAllocator()
    sX = smem.allocate_tensor(x_dtype, cute.make_layout((tile_m, tile_k)), byte_alignment=16)
    s_stats = smem.allocate_tensor(Float32, cute.make_layout((tile_m, 2)), byte_alignment=16)
    s_weight = smem.allocate_tensor(Float32, cute.make_layout((gemm_k,)), byte_alignment=16)
    s_bias = (
        smem.allocate_tensor(Float32, cute.make_layout((gemm_k,)), byte_alignment=16)
        if const_expr(mBias is not None)
        else None
    )

    stage_ln_affine(s_weight, s_bias, mWeight, mBias, gemm_k, k_real, num_threads, tidx, barrier)

    tiled_copy = stats_tiled_copy(x_dtype, (tile_m, tile_k), num_threads, STATS_THREADS_PER_ROW)
    thr_copy = tiled_copy.get_slice(tidx)
    k_tiles = const_expr(gemm_k // tile_k)
    row0 = bidx * tile_m

    row_sum, row_sqsum = make_row_stat_partials(thr_copy, sX)
    for kt in cutlass.range_constexpr(k_tiles):
        _stage_tile(mX, sX, row0, kt, tile_m, tile_k, k_real, num_threads, tidx, barrier)
        row_sum, row_sqsum = accumulate_row_stats(thr_copy, sX, row_sum, row_sqsum)
        barrier.arrive_and_wait()
    finalize_row_stats(
        s_stats,
        tidx,
        row_sum,
        row_sqsum,
        Int32(k_real),
        eps,
        STATS_THREADS_PER_ROW,
        num_threads,
        barrier,
    )
    if const_expr(mMu is not None):
        if tidx < tile_m:
            mMu[row0 + tidx] = s_stats[tidx, 0]
            mRstd[row0 + tidx] = s_stats[tidx, 1]

    for kt in cutlass.range_constexpr(k_tiles):
        _stage_tile(mX, sX, row0, kt, tile_m, tile_k, k_real, num_threads, tidx, barrier)
        normalize_tile(
            sX, s_stats, s_weight, s_bias, thr_copy, Int32(kt), (tile_m, tile_k), x_dtype
        )
        barrier.arrive_and_wait()
        _unstage_tile(mOut, sX, row0, kt, tile_m, tile_k, k_real, num_threads, tidx)
        barrier.arrive_and_wait()


@cute.jit
def _stage_tile(mX, sX, row0, kt, tile_m, tile_k, k_real, num_threads, tidx, barrier):
    """Copy one ``(tile_m, tile_k)`` block of the activation into shared memory, zero-filling the tail.

    A deliberately plain element-per-thread copy: the harness is testing the prologue, not the
    loader, and a TMA here would drag the pipeline machinery in with it. The zero fill for columns
    past ``k_real`` reproduces what the TMA does in production, which is what makes the padded-K
    case testable at all.

    Args:
        mX: The ``(M, K)`` source.
        sX: The ``(tile_m, tile_k)`` destination, overwritten.
        row0: This CTA's first row.
        kt: The k-tile index.
        tile_m: Rows.
        tile_k: Columns per tile.
        k_real: The true feature width; columns at or past it are zeroed.
        num_threads: Threads cooperating.
        tidx: This thread's index.
        barrier: The rendezvous, arrived at once at the end so the tile is complete before use.

    Returns:
        None.
    """
    n_elems = const_expr(tile_m * tile_k)
    for i in cutlass.range_constexpr((n_elems + num_threads - 1) // num_threads):
        idx = i * num_threads + tidx
        if idx < n_elems:
            m = idx // tile_k
            kk = idx % tile_k
            g_k = kt * tile_k + kk
            if g_k < k_real:
                sX[m, kk] = mX[row0 + m, g_k]
            else:
                sX[m, kk] = sX.element_type(0.0)
    barrier.arrive_and_wait()


@cute.jit
def _unstage_tile(mOut, sX, row0, kt, tile_m, tile_k, k_real, num_threads, tidx):
    """Copy one normalized shared-memory tile back out, dropping the padded tail.

    Args:
        mOut: The ``(M, K)`` destination.
        sX: The normalized ``(tile_m, tile_k)`` tile.
        row0: This CTA's first row.
        kt: The k-tile index.
        tile_m: Rows.
        tile_k: Columns per tile.
        k_real: The true feature width; columns at or past it are not written.
        num_threads: Threads cooperating.
        tidx: This thread's index.

    Returns:
        None.
    """
    n_elems = const_expr(tile_m * tile_k)
    for i in cutlass.range_constexpr((n_elems + num_threads - 1) // num_threads):
        idx = i * num_threads + tidx
        if idx < n_elems:
            m = idx // tile_k
            kk = idx % tile_k
            g_k = kt * tile_k + kk
            if g_k < k_real:
                mOut[row0 + m, g_k] = sX[m, kk]


@cute.jit
def _ln_prologue_launch(
    mX,
    mOut,
    mWeight,
    mBias,
    mMu,
    mRstd,
    eps,
    grid_m,
    tile_m: cutlass.Constexpr[int],
    tile_k: cutlass.Constexpr[int],
    gemm_k: cutlass.Constexpr[int],
    k_real: cutlass.Constexpr[int],
    num_threads: cutlass.Constexpr[int],
    x_dtype: cutlass.Constexpr,
    stream,
):
    """Launch the harness: one CTA per row block, ``num_threads`` threads each.

    ``grid_m`` is a RUNTIME argument rather than ``M // tile_m`` derived from the tensor, and that
    is load-bearing: the harness compiles against a concrete ``(tile_m, K)`` fake tensor, so the
    tensor's M extent is BAKED at trace time. Deriving the grid from it launched one CTA for every
    shape and left every row past the first block untouched -- which surfaced as NaN from the
    uninitialised destination, not as a wrong number. The row indexing is unaffected: it uses the
    strides, which are the same for any M.

    Args:
        See :func:`_ln_prologue_kernel`. ``grid_m`` is the CTA count, ``M // tile_m``; ``stream`` is
        the CUDA stream to launch on.

    Returns:
        None.
    """
    _ln_prologue_kernel(
        mX,
        mOut,
        mWeight,
        mBias,
        mMu,
        mRstd,
        eps,
        tile_m,
        tile_k,
        gemm_k,
        k_real,
        num_threads,
        x_dtype,
    ).launch(grid=[grid_m, 1, 1], block=[num_threads, 1, 1], stream=stream)


def run_ln_prologue(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    *,
    tile_m: int = 128,
    tile_k: int = 64,
    gemm_k: Optional[int] = None,
    num_threads: int = 256,
    eps: float = 1e-5,
    want_stats: bool = False,
):
    """Run the prologue over ``x`` and return the normalized result (and optionally its statistics).

    Purpose
        The one entry the tests call. It compiles per distinct configuration and caches, so a
        parametrized sweep pays for each geometry once.

    Args:
        x: ``(M, K)`` activation on CUDA, 16-bit. ``M`` must be a multiple of `tile_m` -- the
            harness does not predicate the row axis, and a ragged M would read past the tensor.
        weight: ``(K,)`` fp32 gain, on the same device.
        bias: ``(K,)`` fp32 bias, or None. None compiles the add out.
        tile_m: Rows per CTA. Must divide ``M`` and ``num_threads // 2``.
        tile_k: Columns per k-tile. Must divide `gemm_k`.
        gemm_k: The padded contraction extent. None means ``ceil(K / tile_k) * tile_k``, which is
            the production rule; pass it explicitly only to test the padded path at a K that is
            already tile-aligned.
        num_threads: Threads per CTA. Must be a multiple of the statistics lanes-per-row.
        eps: The variance floor.
        want_stats: Also return the published ``(mu, rstd)`` per row.

    Returns:
        The ``(M, K)`` normalized tensor in `x`'s dtype, or ``(out, mu, rstd)`` when `want_stats`.

    Raises:
        AssertionError: If ``M`` is not a multiple of `tile_m`, or the partition geometry is
            rejected by ``stats_tiled_copy``.
    """
    M, K = x.shape
    assert M % tile_m == 0, f"the harness does not predicate M; got M={M}, tile_m={tile_m}"
    if gemm_k is None:
        gemm_k = (K + tile_k - 1) // tile_k * tile_k
    out = torch.empty_like(x)
    mu = torch.empty(M, device=x.device, dtype=torch.float32) if want_stats else None
    rstd = torch.empty(M, device=x.device, dtype=torch.float32) if want_stats else None
    fn = _compile(
        torch2cute_dtype_map[x.dtype],
        bias is not None,
        want_stats,
        tile_m,
        tile_k,
        gemm_k,
        K,
        num_threads,
    )
    fn(
        from_dlpack(x, assumed_align=16),
        from_dlpack(out, assumed_align=16),
        from_dlpack(weight, assumed_align=4),
        from_dlpack(bias, assumed_align=4) if bias is not None else None,
        from_dlpack(mu, assumed_align=4) if want_stats else None,
        from_dlpack(rstd, assumed_align=4) if want_stats else None,
        Float32(eps),
        Int32(M // tile_m),
        cutlass.cuda.default_stream(),
    )
    torch.cuda.synchronize()
    return (out, mu, rstd) if want_stats else out


_CACHE = {}


def _compile(x_dtype, has_bias, want_stats, tile_m, tile_k, gemm_k, k_real, num_threads):
    """Compile one harness configuration, memoized on every argument.

    Args:
        x_dtype: Cutlass element type of the activation.
        has_bias: Whether a LayerNorm bias is present. Part of the key because its ABSENCE is
            compiled out rather than added as zero.
        want_stats: Whether the statistics outputs exist. Likewise compiled out when False.
        tile_m: Rows per CTA.
        tile_k: Columns per k-tile.
        gemm_k: The padded contraction extent.
        k_real: The true feature width.
        num_threads: Threads per CTA.

    Returns:
        The compiled launcher.
    """
    key = (x_dtype, has_bias, want_stats, tile_m, tile_k, gemm_k, k_real, num_threads)
    if key not in _CACHE:
        dev = torch.device("cuda")
        tdt = {v: k for k, v in torch2cute_dtype_map.items()}[x_dtype]
        fx = torch.zeros(tile_m, k_real, device=dev, dtype=tdt)
        fw = torch.zeros(k_real, device=dev, dtype=torch.float32)
        fm = torch.zeros(tile_m, device=dev, dtype=torch.float32)
        _CACHE[key] = cute.compile(
            _ln_prologue_launch,
            from_dlpack(fx, assumed_align=16),
            from_dlpack(torch.zeros_like(fx), assumed_align=16),
            from_dlpack(fw, assumed_align=4),
            from_dlpack(fw.clone(), assumed_align=4) if has_bias else None,
            from_dlpack(fm, assumed_align=4) if want_stats else None,
            from_dlpack(fm.clone(), assumed_align=4) if want_stats else None,
            Float32(1e-5),
            Int32(1),
            tile_m,
            tile_k,
            gemm_k,
            k_real,
            num_threads,
            x_dtype,
            cutlass.cuda.default_stream(),
        )
    return _CACHE[key]
