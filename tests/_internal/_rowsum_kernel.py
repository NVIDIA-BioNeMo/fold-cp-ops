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

# Copyright (c) 2025-2026, Tri Dao.
# Copyright (c) 2025, Wentao Guo, Ted Zadouri, Tri Dao.
# This file has been modified by NVIDIA CORPORATION & AFFILIATES.

"""Minimal row-sum kernel: the shared GPU harness for the ``_internal`` unit tests.

``out[m] = sum(x[m, :])`` -- the simplest kernel that still drives every ``_internal`` helper
LayerNorm depends on:

    ReductionBase._get_tiled_copy   -> copy_utils.tiled_copy_2d -> copy_utils.get_copy_atom
    copy_utils.predicate_k          (engaged only when N is not tile-aligned)
    copy_utils.copy
    copy_utils.fill_oob             (only when `oob_fill` is set; see below)
    layout_utils.expand             (gives the (M,) output a column mode so it partitions)
    reduce.row_reduce               -> block_or_cluster_reduce
                                       -> block_reduce      (cluster_n == 1)
                                       -> cluster_reduce    (cluster_n > 1)
                                          -> utils.store_shared_remote -> utils.set_block_rank
                                          -> utils.elem_pointer

**Why a separate kernel rather than reusing LayerNorm.** These helpers are LayerNorm's
dependencies, so testing them *through* LayerNorm would make the unit test circular -- a helper
bug and a kernel bug are indistinguishable, and the "oracle" would be the very numerics under
test. A row sum has a one-line, obviously-correct torch oracle (``x.sum(-1)``), so a failure here
localizes to the helper.

**Sum, not max, deliberately.** A predicated ``cp.async`` zero-fills the masked tail of SMEM
rather than leaving it undefined, so the trailing lanes of a non-tile-aligned row contribute
exact zeros. That is the identity for ADD (harmless) but *not* for MAX, which would read those
zeros as real data on an all-negative row. Keep this harness on ADD.

**That same asymmetry is why the harness carries an ``oob_fill`` knob.** ADD's immunity to the
zero-fill is exactly what makes a plain row sum blind to :func:`copy_utils.fill_oob`, so the sum is
made *deliberately* sensitive to the masked tail: filling it with ``v`` moves the row sum by
``n_pad * v``, an exact integer offset a test can pin. Filling unconditionally (``fill_all``)
overwrites the whole tile, so the sum reports ``tile_cols * v`` -- which hands the test the tile
width it would otherwise have to recompute from the ladders, i.e. duplicate.

This is a test-local helper, not shipped code.
"""

import math
from functools import partial
from typing import Optional, Type

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
from cutlass import Float32, const_expr

import torch

import fold_cp_ops._internal.copy_utils as copy_utils
import fold_cp_ops._internal.compile_time.layout_utils as layout_utils
from fold_cp_ops._internal.cache_utils import jit_cache
from fold_cp_ops._internal.compile_time.compile_utils import make_fake_tensor as fake_tensor
from fold_cp_ops._internal.compile_time.cute_dsl_utils import torch2cute_dtype_map
from fold_cp_ops._internal.reduce import row_reduce
from fold_cp_ops._internal.reduction_base import ReductionBase, ReductionParams


class RowSumParams(ReductionParams):
    """The base reduction pack plus the two knobs that make the sum see the masked tail.

    Attributes:
        oob_fill: Value written into the predicate-masked tail before the reduction, or None to
            skip the fill entirely (the default -- what a plain row sum does). A float or int;
            anything else is a runtime value and is rejected at construction.
        fill_all: If True the fill runs with **no predicate**, overwriting the whole tile rather
            than just its tail, and the row sum becomes ``tile_cols * oob_fill``. Ignored when
            ``oob_fill`` is None. Meaningless -- and silently destructive of the real data -- for
            any use other than measuring the tile width.
    """

    oob_fill: Optional[float] = None
    fill_all: bool = False


class RowSum(ReductionBase):
    """``out[m] = sum(x[m, :])`` over a 2-D row-major input.

    Args:
        dtype: Element type of the input tensor.
        N: Feature (reduced) extent. Need not be a power of two or a tile multiple.
        cluster_n: Forced cluster width. 1 takes the ``block_reduce`` path; >1 takes
            ``cluster_reduce`` and therefore the distributed-shared-memory primitives in
            ``_internal/utils.py``. Forced rather than derived so a test can select the path.
        oob_fill: See :class:`RowSumParams`.
        fill_all: See :class:`RowSumParams`.
    """

    Params = RowSumParams

    def __init__(
        self,
        dtype: Type[cutlass.Numeric],
        N: int,
        cluster_n: int = 1,
        oob_fill: Optional[float] = None,
        fill_all: bool = False,
    ):
        """Bind the harness's compile-time parameters.

        Args:
            dtype: Element type of the input.
            N: Feature extent.
            cluster_n: Forced cluster width -- passed straight through as a declared parameter
                rather than derived, which is what lets a test select the block-vs-cluster path
                independently of the shape.
            oob_fill: Masked-tail fill value, or None for no fill. See :class:`RowSumParams`.
            fill_all: Fill the whole tile rather than its tail. See :class:`RowSumParams`.
        """
        self._bind_params(
            dtype=dtype,
            N=N,
            stage=1,
            cluster_n=cluster_n,
            oob_fill=oob_fill,
            fill_all=fill_all,
        )

    def _threads_per_row(self):
        """Same ladder LayerNorm uses, so the harness exercises the same tile geometries."""
        for limit, threads in [(64, 8), (128, 16), (3072, 32), (6144, 64), (16384, 128)]:
            if self.N <= limit:
                return threads
        return 256

    @cute.jit
    def __call__(self, mX: cute.Tensor, mO: cute.Tensor, stream: cuda.CUstream):
        vecsize = math.gcd(self.N, 128 // mX.element_type.width)
        tiled_copy, tiler_mn, threads_per_row = self._get_tiled_copy(vecsize=vecsize)
        num_threads = tiled_copy.size
        # (M,) -> (M, N) broadcast view so the output partitions like the data tensor.
        mO = layout_utils.expand(mO, dim=1, size=self.N)
        self.kernel(mX, mO, tiler_mn, tiled_copy, threads_per_row).launch(
            grid=[cute.ceil_div(mX.shape[0], tiler_mn[0]), self.cluster_n, 1],
            block=[num_threads, 1, 1],
            cluster=[1, self.cluster_n, 1] if const_expr(self.cluster_n > 1) else None,
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mX: cute.Tensor,
        mO: cute.Tensor,
        tiler_mn: cute.Shape,
        tiled_copy: cute.TiledCopy,
        threads_per_row: cutlass.Constexpr[int],
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        cluster_y = const_expr(0) if const_expr(self.cluster_n == 1) else cute.arch.block_idx()[1]
        tv_layout = tiled_copy.layout_tv_tiled

        smem = cutlass.utils.SmemAllocator()
        sX = smem.allocate_tensor(
            mX.element_type, cute.make_ordered_layout(tiler_mn, order=(1, 0)), byte_alignment=16
        )
        reduction_buffer, mbar_ptr = self._allocate_reduction_buffer_and_mbar(smem, tv_layout)

        shape = mX.shape
        idX = cute.make_identity_tensor(shape)
        gX, gO, cX = [cute.local_tile(mT, tiler_mn, (bidx, cluster_y)) for mT in (mX, mO, idX)]
        thr_copy_X = tiled_copy.get_slice(tidx)
        tXgX = thr_copy_X.partition_S(gX)
        tXsX = thr_copy_X.partition_D(sX)
        tXrO = thr_copy_X.partition_D(gO)
        tXcX = thr_copy_X.partition_S(cX)[(0, None), None, None]
        tXrX = cute.make_rmem_tensor_like(tXgX)

        num_warps = cute.size(tiled_copy) // cute.arch.WARP_SIZE
        self._initialize_cluster(tidx, mbar_ptr, num_warps)

        is_even_N = const_expr(shape[1] == tiler_mn[1] * self.cluster_n)
        tXpX = (
            copy_utils.predicate_k(thr_copy_X.partition_S(cX), limit=shape[1])
            if not is_even_N
            else None
        )
        copy = partial(copy_utils.copy, pred=tXpX)

        row = tXcX[0][0]
        if row < shape[0]:
            copy(tXgX, tXsX, is_async=True)
        cute.arch.cp_async_commit_group()
        cute.arch.cp_async_wait_group(0)
        cute.autovec_copy(tXsX, tXrX)
        # `tXpX` is None when the extent tiles evenly -- and `fill_oob(.., None, ..)` means "fill
        # EVERYTHING", so the tail-fill must be skipped there rather than handed a None predicate.
        if const_expr(self.oob_fill is not None and (self.fill_all or not is_even_N)):
            copy_utils.fill_oob(
                tXrX,
                None if const_expr(self.fill_all) else tXpX,
                mX.element_type(self.oob_fill),
            )
        x = tXrX.load().to(cute.Float32)

        total = row_reduce(
            x,
            cute.ReductionOp.ADD,
            threads_per_row,
            reduction_buffer[None, None, 0],
            mbar_ptr,
            init_val=0.0,
            hook_fn=cute.arch.cluster_wait if const_expr(self.cluster_n > 1) else None,
        )
        if (
            tXcX[0][1] == 0
            and row < shape[0]
            and (self.cluster_n == 1 or cute.arch.block_idx_in_cluster() == 0)
        ):
            tXrO[0] = total


@jit_cache
def _compile_row_sum(dtype, N: int, cluster_n: int, oob_fill: Optional[float], fill_all: bool):
    """Compile :class:`RowSum` on the tvm-ffi launch path, keyed on every compile-time parameter.

    Compiles with ``--enable-tvm-ffi`` and a fake env stream, matching how the shipped kernels
    compile. The alternative -- raw ``cute.compile`` plus a per-call ``from_dlpack`` and an
    explicit ``CUstream`` -- carries a fixed tens-of-microseconds host launch tax, so a harness
    built that way would not be representative of the code it is testing.

    Args:
        dtype: A ``cutlass`` numeric type, i.e. a *value* of ``torch2cute_dtype_map`` (not a
            ``torch.dtype``). Its ``.width`` must divide 128, which every entry of that map
            satisfies. Together with ``N`` it fixes the vector width, so it is part of the key.
        N: Feature extent, i.e. the reduced dimension. Must be positive. Need not be a power of
            two nor a multiple of the tile width -- a non-tile-aligned N engages ``predicate_k``
            and is the interesting case. Baked into the compiled artifact, so a new N recompiles;
            M is symbolic and does not.
        cluster_n: Forced cluster width, >= 1. Must be a power of two, and ``> 1`` requires SM90
            (thread-block clusters) and an N large enough to split across the cluster. Part of
            the key because it selects the block-vs-cluster reduction branch at compile time.
        oob_fill: Masked-tail fill value, or None. **Must be part of the key**: it is folded into
            the kernel as a constant, so sharing a cache entry across two values would hand one
            call the other's fill -- silently, and across processes via the on-disk cache.
        fill_all: Whether the fill ignores the predicate. Same reasoning: it selects a different
            compiled branch.

    Returns:
        A compiled callable taking ``(x, out)`` as torch tensors; the stream comes from the
        tvm-ffi environment rather than being passed per call.
    """
    batch_sym = cute.sym_int()
    div = math.gcd(N, 128 // dtype.width)
    x_cute = fake_tensor(dtype, (batch_sym, N), div)
    o_cute = fake_tensor(Float32, (batch_sym,))
    return cute.compile(
        RowSum(dtype, N, cluster_n, oob_fill, fill_all),
        x_cute,
        o_cute,
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi",
    )


def row_sum(
    x: torch.Tensor,
    cluster_n: int = 1,
    oob_fill: Optional[float] = None,
    fill_all: bool = False,
) -> torch.Tensor:
    """Run the harness kernel: ``out[m] = sum(x[m, :])``, accumulated in float32.

    Args:
        x: The input. Requirements: exactly 2-D; resident on CUDA; **row-major / contiguous**
            (the kernel's TMA-free vectorized loads assume the feature axis is stride-1, and a
            transposed view would be read as if it were contiguous, silently giving wrong sums);
            dtype one of float16 / bfloat16 / float32 (the keys of ``torch2cute_dtype_map`` this
            harness is exercised with). Its feature extent ``N = x.shape[1]`` must satisfy the
            repo's 16-byte alignment floor. ``M = x.shape[0]`` is unconstrained -- it is a
            symbolic dim, so varying it does not recompile. Not modified.
        cluster_n: Forced cluster width; see :class:`RowSum`. 1 (default) takes the
            ``block_reduce`` path; a power of two ``> 1`` takes ``cluster_reduce`` and so
            exercises the distributed-shared-memory primitives. Forcing it -- rather than
            deriving it from N as the shipped kernel does -- is what lets a test select the path
            independently of the shape.
        oob_fill: Value written into the predicate-masked tail before reducing, or None (default)
            for the plain sum. Must be exactly representable in ``x.dtype`` for the returned sum
            to be an exact integer -- pass a small whole number, not 0.1. With an off-grid ``N``
            the result becomes ``sum(x) + n_pad * oob_fill``; with a tile-aligned ``N`` there is
            no tail, so it has no effect at all.
        fill_all: Fill the whole tile instead of only its tail, which **discards every real
            element**: the result is ``tile_cols * oob_fill``, i.e. a way to read back the tile
            width. Ignored when ``oob_fill`` is None.

    Returns:
        A new ``(M,)`` float32 tensor of row sums. Compare against ``x.float().sum(-1)``;
        reduction order differs from torch's, so compare with a tolerance rather than exactly.

    Raises:
        AssertionError: If ``x`` is not a 2-D CUDA tensor.
    """
    assert x.dim() == 2 and x.is_cuda, "harness takes a 2-D CUDA tensor"
    M, N = x.shape
    out = torch.empty(M, device=x.device, dtype=torch.float32)
    _compile_row_sum(torch2cute_dtype_map[x.dtype], N, cluster_n, oob_fill, fill_all)(x, out)
    return out
