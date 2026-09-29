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

# Copyright (c) 2025, Wentao Guo, Ted Zadouri, Tri Dao.
# This file has been modified by NVIDIA CORPORATION & AFFILIATES.

"""LayerNorm forward kernel (CuTe-DSL), in two load variants behind one front door.

Two-stage row reduction: the per-row mean, then the variance about that mean, normalizing via
``x_hat = (x - mean) * rstd`` with optional affine ``weight`` / ``bias``. Optionally writes back
the row ``mean`` and ``rstd``, and fuses a residual add (``out = LN(x + residual)``, with
``residual_out = x + residual``).

**Forward only.** There is no backward pass here; nothing in this workflow trains through it.

Two variants, selected by ``layernorm_fwd(..., transpose=...)``:

* ``transpose=False`` (:class:`LayerNorm`) -- ``x`` is ``(M, N)`` **row-major**, staged tile by tile
  through ``cp.async``. This is the default and the shape everything but the TriMul back half uses.
* ``transpose=True`` (:class:`LayerNormTransposeSm90`) -- ``x`` is ``(M, N)`` **LayoutLeft** (stride
  ``(1, M)``), i.e. the reduce axis is the STRIDED one. The tile is TMA-bulk-loaded out of the
  ``(N, M)`` row-major view and transposed in shared memory, so the reduction still sees a whole
  contiguous row and the output comes out row-major for free.

The reduction, the affine and the front door are shared; **the load is not**. That is the whole of
the difference and it is why the transposing variant is a subclass with its own ``__call__`` and
``kernel`` rather than a boolean threaded through one kernel body.
"""

import math
from typing import Optional, Tuple, Type
from functools import cached_property, partial

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
from cutlass import Float32, const_expr
from cutlass.cute.nvgpu import cpasync
from cutlass.utils import LayoutEnum

import torch
from torch import Tensor

import fold_cp_ops._internal.copy_utils as copy_utils
import fold_cp_ops._internal.sm90_utils as sm90_utils
import fold_cp_ops._internal.compile_time.layout_utils as layout_utils
from fold_cp_ops._internal.autotune import AutotuneConfig, autotune
from fold_cp_ops._internal.compile_time.compile_utils import make_fake_tensor as fake_tensor
from fold_cp_ops._internal.compile_time.copy_descriptors import tiled_copy_2d
from fold_cp_ops._internal.heuristic_arch import (
    TUNED_ARCH,
    heuristic_arch,
    warn_arch_suboptimal_once,
)
from fold_cp_ops._internal.reduce import row_reduce
from fold_cp_ops._internal.tensor_contract import check_tensor
from fold_cp_ops._internal.compile_time.template_params import TemplateParams
from fold_cp_ops._internal.reduction_base import ReductionBase, ReductionParams
from fold_cp_ops._internal.cache_utils import jit_cache
from fold_cp_ops._internal.compile_time.cute_dsl_utils import torch2cute_dtype_map


# Pure-torch reference implementations: the oracle the LayerNorm tests assert against. They must
# not be "improved" -- a reference that shares an optimization with the kernel it checks stops
# being independent evidence, which is the one property an oracle has to keep.
def layernorm_ref(x: Tensor, w: Tensor, eps: float = 1e-6) -> Tensor:
    """Reference implementation for LayerNorm."""
    x_f32 = x.float()
    return torch.nn.functional.layer_norm(x_f32, w.shape, w, None, eps).to(x.dtype)


def layernorm_rstd_ref(x: torch.Tensor, eps: float = 1e-6):
    x_f32 = x.float()
    mean = x_f32.mean(dim=-1, keepdim=True)
    var = ((x_f32 - mean) ** 2).mean(dim=-1)
    return 1.0 / torch.sqrt(var + eps)


def layernorm_mean_ref(x: torch.Tensor) -> torch.Tensor:
    return x.float().mean(dim=-1)


__all__ = [
    "LayerNorm",
    "LayerNormTransposeSm90",
    "layernorm_fwd",
    "layernorm",
    "layernorm_ref",
    "layernorm_rstd_ref",
    "layernorm_mean_ref",
    "layernorm_transpose_freeze",
    "rstd_ref",
    "mean_ref",
]

# Aliases matching the names used by benchmarks/benchmark_layernorm.py
rstd_ref = layernorm_rstd_ref
mean_ref = layernorm_mean_ref


#: Estimated-live-register thresholds above which the weight/bias load is DELAYED past the two
#: reductions. Keyed by ``(element width in bits, narrow-and-aligned tile)`` -- see
#: :meth:`LayerNorm._resolve_delay_w_load` for what the second element means. Every entry is a
#: MEASURED boundary on H100 (sm_90) at M=4096, bracketed below by the last shape that is healthy
#: with the early load and above by the first shape that is not.
#:
#: **What the table tracks is a ptxas CLIFF, not a register count.** Past the boundary the allocator
#: does not degrade gracefully -- it gives up, drops to 32 registers/thread and spills the whole
#: live set to local memory. NCU at N=2816 fp32 with the early load: **9,895,936 local-load sectors,
#: 32 registers/thread, a 1664-byte stack frame, 630 MB of DRAM traffic against an ideal 92 MB
#: (6.9x amplification), 266 us**. The same shape with the load delayed: **0 spill sectors, 163
#: registers, 70.7 MB, 36.7 us** -- 7.3x. The estimate below is only a proxy for where that cliff
#: sits, which is why the boundaries are tabulated rather than derived.
#:
#:   key           threshold   last healthy (early)          first collapsed (early)
#:   (16, *)       255         N=2688 est 252, 1.14x         N=2816 est 264, 11.1x
#:   (32, True)    344         N=2688 est 336, 1.04x         N=2816 est 352, 6.4x
#:   (32, False)   448         N=14336 est 448, 0.87x        N=16384 est 512, 8.6x
#:
#: **The three are not interchangeable and none may be folded into another.** Each substitution was
#: measured, and each costs more than it saves:
#:   * (16,*) used for fp32 at tpr>=64 fires at N=8200 (est 272) and costs 13%.
#:   * (32,True) used for fp32 at tpr>=64 fires at N=6136 (est 384) costing 19%, and at N=14336
#:     (est 448) costing 15%; used on a PADDED fp32 tile it fires at N=2752 (est 352) costing 11%.
#:   * (32,False) used on an aligned fp32 tile at tpr<=32 leaves N in (2688, 3072] spilling at
#:     ~6x -- the exact defect this table exists to remove, which it missed on that one rung until
#:     the ladder was swept densely rather than at powers of two.
#:
#: PERF-ONLY. `delay_w_load` moves *when* the affine weight/bias are fetched, never what is computed
#: (`y = x_hat * w + b` is untouched), so both settings are bit-identical in output -- pinned by
#: tests/kernels/test_layernorm.py::test_delay_w_load_is_bit_identical.
_DELAY_W_EST_REG_THRESHOLD = {
    (16, True): 255,
    (16, False): 255,
    (32, True): 344,
    (32, False): 448,
}


def _cluster_n_for(dtype: Type[cutlass.Numeric], N: int) -> int:
    """Thread-block cluster width for a LayerNorm of this dtype and feature extent.

    A pure function of ``(dtype, N)`` so it can be evaluated BEFORE the parameter pack is bound —
    which is what lets ``cluster_n`` be a declared, immutable parameter instead of an attribute
    assigned by a later ``_set_cluster_n()`` call. That ordering was the bug: a freshly built
    functor had no ``cluster_n`` and ``_get_tiled_copy()`` raised ``AttributeError``.

    Args:
        dtype: Element type. Only ``.width`` is used; 16-bit types cross each rung at half the N of
            32-bit ones, because the rungs are really about bytes in flight.
        N: Feature extent. Must be positive.

    Returns:
        A power of two in 1..16. Values > 1 require SM90 (thread-block clusters) and select the
        distributed-shared-memory reduction path.
    """
    # cluster_n = 4 is faster and cluster_n = 2 for N=64k for some reason
    # Similarly cluster_n = 8 is faster for N=128k
    if dtype.width == 16:
        thresholds = [(16 * 1024, 1), (32 * 1024, 2), (64 * 1024, 4), (128 * 1024, 8)]
    else:
        thresholds = [(32 * 1024, 1), (64 * 1024, 2), (128 * 1024, 4), (256 * 1024, 8)]
    for limit, cluster in thresholds:
        if N <= limit:
            return cluster
    return 16


class LayerNormParams(ReductionParams):
    """LayerNorm's compile-time parameters: the base pack plus where ``x`` is re-read from.

    Attributes:
        reload_from: Where ``x`` is re-read between the mean and variance passes — None to keep it
            live in registers, ``"smem"`` to re-read the staged tile, ``"gmem"`` to re-read global.
            None is faster when the fragments fit; past N=16384 they do not, so the tile is re-read
            instead. Not a perf knob to tune freely: ``"gmem"`` re-reads through the copy predicate
            and must stay consistent with how the tile was staged.
    """

    reload_from: Optional[str] = None


class LayerNorm(ReductionBase):
    """LayerNorm forward: mean-then-variance reduction (2 stages).

    Computes the per-row mean first, then the variance about that mean, and normalizes via
    ``x_hat = (x - mean) * rstd`` with optional affine ``weight`` / ``bias``.

    All compile-time configuration is declared in :class:`LayerNormParams` and bound once in
    ``__init__``; ``delay_w_load`` is deliberately **not** among them — it depends on the operand
    set, so it varies per call and is passed as an explicit kernel argument instead.
    """

    Params = LayerNormParams

    def __init__(self, dtype: Type[cutlass.Numeric], N: int):
        """Bind LayerNorm's compile-time parameters.

        Args:
            dtype: Element type of ``x``. Must be a cutlass numeric type; fp16/bf16/fp32 are the
                dtypes the public entry admits.
            N: Feature extent. Positive; may be off-grid.

        Raises:
            TypeError: If either argument is a runtime value rather than a compile-time constant.
            ValueError: If ``(dtype, N)`` needs a copy narrower than 32 bits -- see below.
        """
        # Validate BEFORE deriving cluster_n from dtype/N -- otherwise a runtime value fails with
        # an opaque AttributeError from inside _cluster_n_for instead of a message naming it.
        self._require_compile_time(dtype=dtype, N=N)
        # FRONT DOOR for the one unsupported region. The tiled copy moves gcd(N, 128 // width)
        # elements per instruction, so the access is that many times `width` bits. Below 32 bits it
        # is a 16-bit copy atom, which fails IR verification inside cutlass-dsl -- an ICE several
        # frames down naming neither N nor the dtype. Refuse it here, where (dtype, N) is first
        # known, so EVERY path in is covered: layernorm_fwd, the registered op, and a direct
        # LayerNorm(...) construction alike.
        #
        # A `raise`, not an `assert`: asserts are stripped under `python -O`, and a guard that
        # disappears under optimization is not a guard.
        #
        # This is NOT the 16-byte alignment floor. MEASURED (M=8, rstd vs the fp32 reference):
        # float32 computes every N from 1 up -- N=1, 3, 5 are 4, 12 and 20 bytes -- and bfloat16
        # computes every EVEN N, including N=2 (4 bytes). The constraint is the atom width alone,
        # and nothing outside it returns a wrong answer.
        copy_bits = math.gcd(N, 128 // dtype.width) * dtype.width
        if copy_bits < 32:
            raise ValueError(
                f"unsupported N={N} for {dtype}: the vectorized copy would be {copy_bits} bits "
                f"(gcd({N}, {128 // dtype.width}) x {dtype.width}), and anything under 32 needs a "
                f"16-bit copy atom that fails IR verification in cutlass-dsl. "
                + (
                    f"Use an EVEN N at {dtype}."
                    if dtype.width == 16
                    else f"Use an N with gcd(N, {128 // dtype.width}) >= {32 // dtype.width}."
                )
            )
        # stage=2: one reduction buffer for the mean, one for the variance.
        self._bind_params(
            dtype=dtype,
            N=N,
            stage=2,
            cluster_n=_cluster_n_for(dtype, N),
            reload_from=None if N <= 16384 else "smem",
        )

    def _threads_per_row(self):
        N = self.N
        for limit, threads in [(64, 8), (128, 16), (3072, 32), (6144, 64), (16384, 128)]:
            if N <= limit:
                return threads
        return 256

    def _resolve_delay_w_load(
        self,
        elems_per_thread: int,
        threads_per_row: int,
        is_even_N: bool,
        has_w: bool,
        has_b: bool,
    ) -> bool:
        """Decide whether to fetch the affine weight/bias AFTER the reductions rather than before.

        Both orders compute the same values; this only moves the loads in program order, trading
        latency hiding (load early, overlap with the reduction) against register pressure (the
        fragments stay live across both reduction passes).

        Why it matters, measured: each thread holds ``elems_per_thread`` elements of x and out
        (``dtype``) plus w and b (**always fp32**, so 2 registers per element at bf16). Past a point
        ptxas stops fitting the live set in the 255-entry register file and spills it to local
        memory -- and it does so as a cliff, collapsing to 32 registers/thread rather than degrading
        gradually. NCU at M=4096, N=16384, bf16 measured 69.1M local-load + 69.0M local-store
        sectors and 3.89 GB of DRAM traffic against an ideal 268 MB -- 14.5x amplification, 151 GB/s
        vs a ~2400 GB/s neighbour. Delaying the load removes w/b from the live set across the
        reductions and restores full bandwidth.

        Args:
            elems_per_thread: Elements of the row each thread owns, i.e. ``tiler_n // threads_per_row``.
                Must be positive. This is the quantity the register demand scales with.
            threads_per_row: Lanes cooperating on one row, i.e. what ``_threads_per_row()`` resolved
                to for this shape. Together with ``is_even_N`` it selects the threshold.
            is_even_N: Whether the feature extent fills the tile exactly, i.e.
                ``N == tiler_n * cluster_n``. **Must match what the kernel computes**, or the gate
                is fitted to a shape the kernel is not compiling; nothing raises if it does not.
            has_w: Whether an affine weight is present. A fragment that does not exist cannot be
                delayed, so it contributes nothing to the estimate.
            has_b: Whether an affine bias is present. Same reasoning.

        Returns:
            True to defer the weight/bias fetch. False -- the default at low pressure -- keeps the
            early load, because delaying it costs 8-19% where there is no collapse to avoid
            (measured: N=16392/20480 bf16 ~+9%, N=2752 fp32 +11%, N=14336 fp32 +15%,
            N=6136 fp32 +19%).

        **Why the threshold depends on the rung AND on the padding, which looks arbitrary and is
        not.** At float32 the collapse needs a *narrow* rung (``threads_per_row <= 32``, i.e. a CTA
        tile several rows tall) AND an *exactly filled* tile. Measured at M=4096, pairing each
        aligned N with a padded N of the SAME tile width and elements-per-thread:

            elems  aligned N   early->late     padded N   early->late
              84   2688        1.04x           2624       1.07x
              88   2816        **0.156x**      2752       1.11x
              92   2944        **0.164x**      2880       1.03x
              96   3072        **0.167x**      3000       1.06x

        The aligned column collapses and the padded column does not, at identical register demand.
        The padded tile compiles its copies through ``predicate_k``, and that is apparently enough
        to keep ptxas from giving up -- an allocator artefact, not something the estimate can
        predict, which is why it is measured and tabulated. **At bf16 the same sweep collapses in
        BOTH columns** (0.089x aligned / 0.082x padded at elems 88), so the 16-bit entries do not
        carry the distinction and must not be "simplified" to match the 32-bit ones.
        """
        # 32-bit register slots for the fragments live across the reductions: x and out at the
        # element dtype, w and b always fp32. Only count affine fragments that actually exist.
        per_elem = 2 * self.dtype.width // 32 + (1 if has_w else 0) + (1 if has_b else 0)
        est_live_regs = elems_per_thread * per_elem
        narrow_aligned_tile = threads_per_row <= 32 and is_even_N
        return est_live_regs > _DELAY_W_EST_REG_THRESHOLD[(self.dtype.width, narrow_aligned_tile)]

    @cute.jit
    def __call__(
        self,
        mX: cute.Tensor,
        mW: Optional[cute.Tensor],
        mB: Optional[cute.Tensor],
        mRes: Optional[cute.Tensor],
        mO: cute.Tensor,
        mResO: Optional[cute.Tensor],
        mRstd: Optional[cute.Tensor],
        mMean: Optional[cute.Tensor],
        eps: Float32,
        stream: cuda.CUstream,
    ):
        assert mX.element_type == self.dtype
        largest_dtype_width = const_expr(
            max(*(t.element_type.width for t in [mX, mRes, mW, mB, mO, mResO] if t is not None))
        )
        vecsize = math.gcd(self.N, 128 // largest_dtype_width)
        tiled_copy, tiler_mn, threads_per_row = self._get_tiled_copy(vecsize=vecsize)
        num_threads = tiled_copy.size
        # Resolve the w/b load position from the ACTUAL tile geometry, not from N: the register
        # demand depends on elements-per-thread, which folds in vecsize, cluster_n and the ladders.
        # A LOCAL, passed explicitly to the kernel below -- it varies per call (it depends on which
        # operands are present), so it is an argument, not configuration. See ReductionBase.
        delay_w_load = const_expr(
            self._resolve_delay_w_load(
                tiler_mn[1] // threads_per_row,
                threads_per_row,
                # Same predicate the kernel derives below; kept in step by construction.
                self.N == tiler_mn[1] * self.cluster_n,
                mW is not None,
                mB is not None,
            )
        )
        mW, mB = [
            layout_utils.expand(mT, dim=0, size=tiler_mn[0]) if const_expr(mT is not None) else None
            for mT in (mW, mB)
        ]
        mRstd, mMean = [
            layout_utils.expand(mT, dim=1, size=self.N) if const_expr(mT is not None) else None
            for mT in (mRstd, mMean)
        ]
        self.kernel(
            mX,
            mW,
            mB,
            mRes,
            mO,
            mResO,
            mRstd,
            mMean,
            eps,
            tiler_mn,
            tiled_copy,
            threads_per_row,
            delay_w_load,
        ).launch(
            grid=[cute.ceil_div(mX.shape[0], tiler_mn[0]), self.cluster_n, 1],
            block=[num_threads, 1, 1],
            cluster=[1, self.cluster_n, 1] if const_expr(self.cluster_n > 1) else None,
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mX: cute.Tensor,
        mW: Optional[cute.Tensor],
        mB: Optional[cute.Tensor],
        mRes: Optional[cute.Tensor],
        mO: cute.Tensor,
        mResO: Optional[cute.Tensor],
        mRstd: Optional[cute.Tensor],
        mMean: Optional[cute.Tensor],
        eps: Float32,
        tiler_mn: cute.Shape,
        tiled_copy: cute.TiledCopy,
        threads_per_row: cutlass.Constexpr[int],
        delay_w_load: cutlass.Constexpr[bool],
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        cluster_y = const_expr(0) if const_expr(self.cluster_n == 1) else cute.arch.block_idx()[1]
        tv_layout = tiled_copy.layout_tv_tiled

        smem = cutlass.utils.SmemAllocator()
        sX = smem.allocate_tensor(
            mX.element_type, cute.make_ordered_layout(tiler_mn, order=(1, 0)), byte_alignment=16
        )
        if const_expr(mRes is not None):
            sRes = smem.allocate_tensor(
                mRes.element_type,
                cute.make_ordered_layout(tiler_mn, order=(1, 0)),
                byte_alignment=16,
            )
        reduction_buffer, mbar_ptr = self._allocate_reduction_buffer_and_mbar(smem, tv_layout)

        shape = mX.shape
        idX = cute.make_identity_tensor(shape)
        # slice for CTAs
        gX, gRes, gO, gResO, gRstd, gMean, cX = [
            cute.local_tile(mT, tiler_mn, (bidx, cluster_y)) if mT is not None else None
            for mT in (mX, mRes, mO, mResO, mRstd, mMean, idX)
        ]
        gW, gB = [
            cute.local_tile(mT, tiler_mn, (0, cluster_y)) if const_expr(mT is not None) else None
            for mT in (mW, mB)
        ]

        thr_copy_X = tiled_copy.get_slice(tidx)

        tXgW = thr_copy_X.partition_S(gW) if const_expr(mW is not None) else None
        tXgB = thr_copy_X.partition_S(gB) if const_expr(mB is not None) else None
        tXgX = thr_copy_X.partition_S(gX)
        tXsX = thr_copy_X.partition_D(sX)
        if const_expr(mRes is not None):
            tXgRes = thr_copy_X.partition_S(gRes)
            tXsRes = thr_copy_X.partition_D(sRes)
        tXgO = thr_copy_X.partition_D(gO)
        if const_expr(mResO is not None):
            tXgResO = thr_copy_X.partition_D(gResO)
        tXrRstd = thr_copy_X.partition_D(gRstd) if const_expr(mRstd is not None) else None
        tXrMean = thr_copy_X.partition_D(gMean) if const_expr(mMean is not None) else None
        tXcX = thr_copy_X.partition_S(cX)[(0, None), None, None]

        # allocate fragments for gmem->rmem
        tXrW = cute.make_rmem_tensor_like(tXgW) if const_expr(mW is not None) else None
        tXrB = cute.make_rmem_tensor_like(tXgB) if const_expr(mB is not None) else None
        tXrX, tXrO = [cute.make_rmem_tensor_like(t) for t in (tXgX, tXgO)]
        if const_expr(mRes is not None):
            tXrRes = cute.make_rmem_tensor_like(tXgRes)

        num_warps = cute.size(tiled_copy) // cute.arch.WARP_SIZE
        self._initialize_cluster(tidx, mbar_ptr, num_warps)

        is_even_N = const_expr(shape[1] == tiler_mn[1] * self.cluster_n)
        tXpX = (
            copy_utils.predicate_k(thr_copy_X.partition_S(cX), limit=shape[1])
            if not is_even_N
            else None
        )
        # Each copy will use the same predicate
        copy = partial(copy_utils.copy, pred=tXpX)

        row = tXcX[0][0]
        if row < shape[0]:
            copy(tXgX, tXsX, is_async=True)
            if const_expr(mRes is not None):
                copy(tXgRes, tXsRes, is_async=True)
        cute.arch.cp_async_commit_group()

        if const_expr(not delay_w_load):
            if const_expr(mW is not None):
                copy(tXgW, tXrW)
            if const_expr(mB is not None):
                copy(tXgB, tXrB)

        cute.arch.cp_async_wait_group(0)
        cute.autovec_copy(tXsX, tXrX)
        x = tXrX.load().to(cute.Float32)
        if const_expr(mRes is not None):
            cute.autovec_copy(tXsRes, tXrRes)
            x += tXrRes.load().to(cute.Float32)
        if const_expr(mResO is not None):
            tXrResO = cute.make_rmem_tensor_like(tXgResO)
            tXrResO.store(x.to(tXrResO.element_type))
            if row < shape[0]:
                copy(tXrResO, tXgResO)

        # LayerNorm: compute mean first, then variance about the mean.
        sum_x = row_reduce(
            x,
            cute.ReductionOp.ADD,
            threads_per_row,
            reduction_buffer[None, None, 0],
            mbar_ptr + 0 if const_expr(self.cluster_n > 1) else None,
            init_val=0.0,
            hook_fn=cute.arch.cluster_wait if const_expr(self.cluster_n > 1) else None,
        )
        mean = sum_x / shape[1]
        if const_expr(mMean is not None):
            # Only the thread corresponding to column 0 writes out the mean to gmem
            if (
                tXcX[0][1] == 0
                and row < shape[0]
                and (self.cluster_n == 1 or cute.arch.block_idx_in_cluster() == 0)
            ):
                tXrMean[0] = mean
        if const_expr(self.reload_from == "smem"):
            cute.autovec_copy(tXsX, tXrX)
            x = tXrX.load().to(cute.Float32)
            if const_expr(mRes is not None):
                cute.autovec_copy(tXsRes, tXrRes)
                x += tXrRes.load().to(cute.Float32)
        elif const_expr(self.reload_from == "gmem"):
            copy(tXgX, tXrX)
            x = tXrX.load().to(cute.Float32)
            if const_expr(mRes is not None):
                copy(tXgRes, tXrRes)
                x += tXrRes.load().to(cute.Float32)
        # The masked tail of the tile is ZERO-FILLED by the predicated cp.async. Zero is the
        # identity for the mean's ADD above, so pass 1 needs no mask -- but this pass reduces
        # (x - mean)^2 over the WHOLE tile, where each masked lane contributes (0 - mean)^2 =
        # mean^2 rather than 0, inflating the variance by (tile - N) * mean^2 / N. Mask the
        # CENTRED values rather than the raw ones: filling x with the mean instead would have to
        # round the fp32 mean into the tile's element type. Wholly const_expr-pruned when the
        # feature extent tiles evenly, which is every aligned shape including the TriMul front.
        x_sub_mean = x - mean
        if const_expr(not is_even_N):
            tXrXsm = cute.make_rmem_tensor(tXrX.shape, Float32)
            tXrXsm.store(x_sub_mean)
            copy_utils.fill_oob(tXrXsm, tXpX, Float32(0.0))
            x_sub_mean = tXrXsm.load()
        sum_sq_x_sub_mean = row_reduce(
            x_sub_mean * x_sub_mean,
            cute.ReductionOp.ADD,
            threads_per_row,
            reduction_buffer[None, None, 1],
            mbar_ptr + 1 if const_expr(self.cluster_n > 1) else None,
            init_val=0.0,
        )
        rstd = cute.math.rsqrt(sum_sq_x_sub_mean / shape[1] + eps, fastmath=True)
        if const_expr(mRstd is not None):
            # Only the thread corresponding to column 0 writes out the rstd to gmem
            if (
                tXcX[0][1] == 0
                and row < shape[0]
                and (self.cluster_n == 1 or cute.arch.block_idx_in_cluster() == 0)
            ):
                tXrRstd[0] = rstd
        if const_expr(delay_w_load):
            if const_expr(mW is not None):
                copy(tXgW, tXrW)
            if const_expr(mB is not None):
                copy(tXgB, tXrB)
        if const_expr(self.reload_from == "smem" or self.reload_from == "gmem"):
            if const_expr(self.reload_from == "smem"):
                cute.autovec_copy(tXsX, tXrX)
                if const_expr(mRes is not None):
                    cute.autovec_copy(tXsRes, tXrRes)
            else:
                copy(tXgX, tXrX)
                if const_expr(mRes is not None):
                    copy(tXgRes, tXrRes)
            x = tXrX.load().to(cute.Float32)
            if const_expr(mRes is not None):
                x += tXrRes.load().to(cute.Float32)
        x_hat = (x - mean) * rstd
        y = x_hat
        if const_expr(mW is not None):
            y *= tXrW.load().to(cute.Float32)
        if const_expr(mB is not None):
            y += tXrB.load().to(cute.Float32)
        tXrO.store(y.to(tXrO.element_type))
        if row < shape[0]:
            copy(tXrO, tXgO)


@torch.library.custom_op(
    "fold_cp_ops::_layernorm_fwd",
    mutates_args=("out", "rstd", "mean", "residual_out"),
    device_types="cuda",
    # We need to specify the schema manually since we're mutating an optional tensor
    schema="(Tensor x, Tensor? weight, Tensor(a2!) out, Tensor? bias, Tensor(a4!)? rstd, Tensor(a5!)? mean, Tensor? residual, Tensor(a7!)? residual_out, float eps=1e-6) -> ()",
)
def _layernorm_fwd(
    x: Tensor,
    weight: Optional[Tensor],
    out: Tensor,
    bias: Optional[Tensor] = None,
    rstd: Optional[Tensor] = None,
    mean: Optional[Tensor] = None,
    residual: Optional[Tensor] = None,
    residual_out: Optional[Tensor] = None,
    eps: float = 1e-6,
) -> None:
    """LayerNorm forward pass.
    Args:
        x: Input tensor of shape (M, N)
        weight: Optional weight tensor of shape (N,)
        eps: Small value for numerical stability
    Returns:
        Normalized output tensor of same shape as x
    """
    # Don't need to check is_cuda since torch.library ensures that
    supported_types = {torch.float16, torch.bfloat16, torch.float32}
    # `raise`, not `assert` -- see `layernorm_fwd`. Each name is the one in this signature, so a
    # caller is told which argument to change rather than which internal tensor disagreed.
    for nm, t in (
        ("x", x),
        ("weight", weight),
        ("residual", residual),
        ("bias", bias),
        ("out", out),
        ("residual_out", residual_out),
    ):
        check_tensor(nm, t)
        if t is not None and t.dtype not in supported_types:
            raise ValueError(
                f"{nm} must be float16, bfloat16 or float32 for the LayerNorm kernel; got {t.dtype}."
            )

    _, N = x.shape
    dtype, out_dtype, weight_dtype, bias_dtype, res_dtype, res_out_dtype = [
        torch2cute_dtype_map[t.dtype] if t is not None else None
        for t in [x, out, weight, bias, residual, residual_out]
    ]
    _compile_layernorm_fwd(
        dtype,
        out_dtype,
        res_dtype,
        weight_dtype,
        bias_dtype,
        res_out_dtype,
        N,
        rstd is not None,
        mean is not None,
    )(x, weight, bias, residual, out, residual_out, rstd, mean, eps)


@_layernorm_fwd.register_fake
def _layernorm_fwd_fake(
    x: Tensor,
    weight: Optional[Tensor],
    out: Tensor,
    bias: Optional[Tensor] = None,
    rstd: Optional[Tensor] = None,
    mean: Optional[Tensor] = None,
    residual: Optional[Tensor] = None,
    residual_out: Optional[Tensor] = None,
    eps: float = 1e-6,
) -> None:
    # See softmax.py _softmax_fwd_fake for why register_fake is needed.
    from fold_cp_ops._internal.cache_utils import COMPILE_ONLY

    if COMPILE_ONLY and not isinstance(x.size(1), torch.SymInt):
        N = x.size(1)
        dtype, out_dtype, weight_dtype, bias_dtype, res_dtype, res_out_dtype = [
            torch2cute_dtype_map[t.dtype] if t is not None else None
            for t in [x, out, weight, bias, residual, residual_out]
        ]
        _compile_layernorm_fwd(
            dtype,
            out_dtype,
            res_dtype,
            weight_dtype,
            bias_dtype,
            res_out_dtype,
            N,
            rstd is not None,
            mean is not None,
        )


@jit_cache
def _compile_layernorm_fwd(
    dtype,
    out_dtype,
    res_dtype,
    weight_dtype,
    bias_dtype,
    res_out_dtype,
    N,
    has_rstd,
    has_mean,
):
    batch_sym = cute.sym_int()
    all_dtypes = [dtype, out_dtype, res_dtype, weight_dtype, bias_dtype, res_out_dtype]
    div = math.gcd(N, *(128 // dt.width for dt in all_dtypes if dt is not None))
    x_cute, out_cute, res_cute, res_out_cute = [
        fake_tensor(dt, (batch_sym, N), div) for dt in [dtype, out_dtype, res_dtype, res_out_dtype]
    ]
    weight_cute, bias_cute = [fake_tensor(dt, (N,), div) for dt in [weight_dtype, bias_dtype]]
    rstd_cute = fake_tensor(Float32, (batch_sym,)) if has_rstd else None
    mean_cute = fake_tensor(Float32, (batch_sym,)) if has_mean else None
    return cute.compile(
        LayerNorm(dtype, N),
        x_cute,
        weight_cute,
        bias_cute,
        res_cute,
        out_cute,
        res_out_cute,
        rstd_cute,
        mean_cute,
        Float32(0),  # eps, just for compilation
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi",
    )


def layernorm_fwd(
    x: Tensor,
    weight: Tensor,
    bias: Optional[Tensor] = None,
    eps: float = 1e-6,
    return_rstd: bool = False,
    return_mean: bool = False,
    transpose: bool = False,
    blk_m: Optional[int] = None,
    subt: Optional[int] = None,
    threads_per_row: Optional[int] = None,
    do_transpose: str = "auto",
    select: str = "heuristic",
    _config: Optional[AutotuneConfig] = None,
):
    """LayerNorm forward pass using the LayerNorm-specialized kernel.

    Args:
        x: Input tensor of shape (M, N). Row-major when ``transpose`` is False; **LayoutLeft**
            (stride ``(1, M)``) when it is True -- see below.
        weight: Weight tensor of shape (N,). Must be float32.
        bias: Optional bias tensor of shape (N,). Must be float32.
        eps: Small value for numerical stability
        return_rstd: Whether to return the reciprocal standard deviation. Not available with
            ``transpose=True`` -- that kernel does not write the row statistics out.
        return_mean: Whether to return the mean. Same restriction as ``return_rstd``.
        transpose: Select the **transposing-load** variant (:class:`LayerNormTransposeSm90`), for an
            ``x`` whose reduce axis N is the strided one. False (default) is the ``cp.async``
            row-major path and is byte-for-byte the kernel this entry has always compiled.
        blk_m: ``transpose=True`` only -- output rows per CTA. None auto-picks (16).
        subt: ``transpose=True`` only -- N-edge of a transpose sub-tile. None auto-picks the largest
            TILEABLE divisor of N under the per-N bucket base; see :func:`_subt_is_tileable`.
        threads_per_row: ``transpose=True`` only -- lanes cooperating on one row's N-reduction.
            None auto-picks the largest power-of-two divisor of N under the bucket base.
        do_transpose: ``transpose=True`` only -- ``"SMEM"`` / ``"shuffle"`` / ``"ldmatrix"`` /
            ``"auto"``. See :func:`_transpose_fwd`.
        select: ``transpose=True`` only -- ``"heuristic"`` (default, size->knob formula),
            ``"autotune"`` (measure this shape) or ``"default"`` (run exactly what was passed).
        _config: ``transpose=True`` only -- a frozen :class:`AutotuneConfig` from
            :func:`layernorm_transpose_freeze`, which skips the per-call knob resolution.

    Returns:
        Normalized output tensor of same shape as x. Row-major in **both** variants: with
        ``transpose=True`` the LayoutLeft input becomes a row-major output, which is the layout flip
        the variant exists for.
        If return_rstd is True, also returns rstd tensor of shape (M,)
        If return_mean is True, also returns mean tensor of shape (M,)

    Raises:
        ValueError: If a tensor argument violates its dtype/shape contract; if a transpose-only knob
            is passed with ``transpose=False`` (which would otherwise be silently ignored -- the
            caller would get the row-major kernel and no signal); or, on the transposing path, if a
            size gate is violated (see :func:`_transpose_fwd`).
        NotImplementedError: ``transpose=True, do_transpose="ldmatrix"`` -- a documented blocker,
            see :meth:`LayerNormTransposeSm90._ldmatrix_fragment`.
    """
    if transpose:
        return _transpose_fwd(
            x,
            weight,
            bias,
            eps=eps,
            return_rstd=return_rstd,
            return_mean=return_mean,
            blk_m=blk_m,
            subt=subt,
            threads_per_row=threads_per_row,
            do_transpose=do_transpose,
            select=select,
            _config=_config,
        )
    # A transpose-only knob with transpose=False is a caller mistake with no symptom: the row-major
    # kernel runs, ignores the knob, and returns a correct answer for the WRONG input layout. Refuse
    # it here rather than let the caller conclude the knob had no effect.
    transpose_only = {
        "blk_m": blk_m,
        "subt": subt,
        "threads_per_row": threads_per_row,
        "_config": _config,
    }
    passed = [k for k, v in transpose_only.items() if v is not None]
    if do_transpose != "auto":
        passed.append("do_transpose")
    if select != "heuristic":
        passed.append("select")
    if passed:
        raise ValueError(
            f"{sorted(passed)} are knobs of the TRANSPOSING variant and are ignored with "
            f"transpose=False. Pass transpose=True (and a LayoutLeft x), or drop them."
        )
    # `raise`, not `assert`: `python -O` strips asserts, so under -O the four checks below simply
    # vanish and an unsupported weight reaches the kernel. Measured before this change, a float64
    # weight here produced an AssertionError -- the one failure class the kernel-matrix rule refuses
    # to accept as a refusal, because it cannot be distinguished from a stripped check.
    check_tensor("x", x, expect_shape=(None, None))
    check_tensor("weight", weight, expect_dtype=torch.float32, expect_shape=(x.shape[-1],))
    check_tensor("bias", bias, expect_dtype=torch.float32, expect_shape=(x.shape[-1],))
    if x.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError(
            f"x must be float16, bfloat16 or float32 for the LayerNorm kernel; got {x.dtype}."
        )

    M, N = x.shape
    device = x.device
    out = torch.empty_like(x)
    rstd = torch.empty(M, device=device, dtype=torch.float32) if return_rstd else None
    mean = torch.empty(M, device=device, dtype=torch.float32) if return_mean else None

    _layernorm_fwd(x, weight, out, bias, rstd, mean, None, None, eps)

    if return_rstd and return_mean:
        return out, rstd, mean
    elif return_rstd:
        return out, rstd
    elif return_mean:
        return out, mean
    return out


# Alias used by benchmarks/benchmark_layernorm.py
layernorm = layernorm_fwd


# ═══════════════════════════════════════════════════════════════════════════════════════════════ #
#  The TRANSPOSING-LOAD variant: LayoutLeft in, row-major out, one CTA per BLK_M output rows.
#
#  `x` is (M, N) LayoutLeft (stride (1, M)): the LN reduce axis N is STRIDED by M and the batch
#  axis M is contiguous. In memory that is exactly an (N, M) row-major array, so we view it as one
#  (`mXt`). Each CTA owns BLK_M consecutive M-columns (== output rows) and the FULL N, staged as:
#
#    * TMA bulk G2S of a (SUBT, BLK_M) sub-block of the (N, M) input into a 128-B-swizzled SMEM
#      tile `sX` -- BLK_M, the contiguous M mode, is the swizzled mode;
#    * a transpose into the wide (BLK_M, N) tile `sT` at the right N offset, so `sT[m, n]` is
#      N-innermost -- the layout the normalize + store already wants.
#
#  The sub-tile loop covers all of N (N <= 1024 fits), so the WHOLE row lands in SMEM per BLK_M
#  block and :func:`~fold_cp_ops._internal.reduce.row_reduce` runs unchanged on it. The transpose is
#  then "free" at the store: the output is written N-innermost, i.e. row-major.
#
#  KEY TENSION (resolved): the reduction prefers wide-N / thin-M tiles, while a TMA+swizzle
#  transpose is efficient only on squarish ones. Transposing squarish (SUBT, BLK_M) sub-blocks and
#  ASSEMBLING them into one wide (BLK_M, N) tile gets both -- a coalesced load and a full row under
#  the reduction. BLK_M and SUBT are tunable.
# ═══════════════════════════════════════════════════════════════════════════════════════════════ #


def _subt_is_tileable(subt: int) -> bool:
    """Can ``make_smem_layout`` build an EXACT ``(subt, blk_m)`` tile for this ``subt``?

    MEASURED, not assumed -- the layouts were printed from a live MLIR trace for ``subt`` in
    8..128. ``make_smem_layout`` -> ``cute.tile_to_shape`` tiles a whole swizzle atom of row-extent
    8 across the ``subt`` mode, producing shape ``((8, k), (16, 1), (1, n_sub_n))`` with
    ``k = ceil(subt/8)``. ``tile_to_shape`` itself ALWAYS succeeds -- verified for every ``subt`` in
    8..128. The two failures land on the NEXT two calls, and both are silent at the call site:

    * ``cute.slice_(ld_layout, (None, None, 0))`` -- which drops the stage mode -- **ABORTS the
      process** (SIGABRT, no Python traceback, dies inside the swizzled composed_layout) when ``k``
      is odd and ``k >= 3``; ``k=1`` is fine. Hit by ``subt`` = 20, 24, 35, 40, 50, 54, 55, 56, 70,
      72.
    * when ``8k != subt`` the tiling QUANTISES, e.g. ``subt=63`` -> 64 rows. ``tile_to_shape`` and
      ``slice_`` both accept it; the TMA atom is built from the true ``subt``, so
      ``make_tiled_tma_atom`` rejects the pair with "expected top-level shape equivalence between
      the SMEM layout and the CTA V-map" (a catchable ``ValueError``, not an abort).

    N=560, every divisor 8..128, one fresh process each: ``slice_`` OK at ``subt`` {8, 16, 80, 112},
    SIGABRT at {20, 35, 40, 56, 70}, ``ValueError`` at {10, 14, 28}. The OK set is EXACTLY
    ``{8} u {multiples of 16 dividing 560}`` -- the predicate below, with no slack either way.

    Legal <=> ``subt % 8 == 0`` (no quantisation) AND ``k`` even or 1 (no abort) <=> ``subt == 8``
    or ``subt % 16 == 0``.

    Note the constraint comes from tiling the atom ACROSS the ``subt`` mode, **not** from choosing
    it: the atom IS keyed on ``blk_m``. An earlier version reasoned from the latter to "any positive
    divisor of N is valid", which is what let the auto-pick return e.g. 56 and abort the process.

    Args:
        subt: A candidate sub-tile N-edge. Any positive int; values outside 8..128 are answered by
            the same arithmetic and are not rejected here (the caller bounds the range).

    Returns:
        True if ``make_smem_layout`` + ``slice_`` + ``make_tiled_tma_atom`` accept this edge. False
        means "do not use it" -- and note the consequence of ignoring that is a SIGABRT with no
        traceback, not an exception.
    """
    return subt == 8 or subt % 16 == 0


def _auto_subt(N: int, base_subt: int) -> int:
    """Largest TILEABLE divisor of ``N`` that is ``<= base_subt`` (see :func:`_subt_is_tileable`).

    This does NOT narrow the supported shapes: the repo's 16-byte floor already forces
    ``N % 8 == 0``, so 8 always divides N and 8 is always tileable -- a legal ``subt`` exists for
    every supported N. Nor does it perturb any shape that works today: the ``subt`` values in use
    (32, 64) are both tileable, so they are still chosen and their tile / layout / perf are
    unchanged. Only N whose largest divisor ``<= base_subt`` was NON-tileable move, and those
    aborted the process before.

    Args:
        N: Feature extent. Positive; must be a multiple of 8, which the front door's swizzle-atom
            floor already guarantees. A non-multiple falls through to the ``return 8`` below and
            yields a ``subt`` that does not divide N, which the functor then rejects.
        base_subt: The per-N-bucket ceiling from :func:`_transpose_default_config`. Positive.

    Returns:
        A tileable divisor of ``N``, at most ``base_subt``.
    """
    for candidate in range(base_subt, 0, -1):
        if N % candidate == 0 and _subt_is_tileable(candidate):
            return candidate
    return 8  # always reachable: the 16-byte floor guarantees N % 8 == 0 and 8 is tileable


def _auto_tpr(N: int, base_tpr: int, blk_m: int) -> int:
    """Largest power-of-two divisor of ``N`` that is ``<= base_tpr``.

    ``threads_per_row`` MUST be a power of two: ``row_reduce`` finishes with a butterfly
    ``shuffle_sync_bfly``, which requires the group size to be one. A non-power-of-two would not
    raise -- it would reduce over the wrong lane set and return a wrong mean.

    Args:
        N: Feature extent. Positive and even (the front door's floor guarantees far more).
        base_tpr: The per-N-bucket ceiling from :func:`_transpose_default_config`. Positive.
        blk_m: Output rows per CTA. Only used to document the warp-alignment consequence: at the
            shipped ``blk_m = 16`` any ``tpr >= 2`` gives ``num_threads = tpr * 16 >= 32``, so the
            functor's warp-multiple check is satisfied by construction.

    Returns:
        A power-of-two divisor of ``N``, at most ``base_tpr``, and at least 2.
    """
    p = base_tpr
    while p >= 2:
        if N % p == 0 and (p & (p - 1)) == 0:  # power-of-2 divisor of N check
            return p
        p //= 2
    return 2  # fallback (N must be even; see the swizzle-atom floor in _transpose_fwd)


def _transpose_default_config(N: int, dtype: Type[cutlass.Numeric]) -> Tuple[int, int, int]:
    """Return ``(blk_m, subt, threads_per_row)`` defaults for the transposing variant.

    * ``blk_m`` -- output rows per CTA. Must be a multiple of ``subt``'s partner edge so the
      transpose sub-tiles tile it; fixed at 16, which is the swizzle atom's row extent.
    * ``subt`` -- transpose sub-tile N-edge (``blk_m`` is the M-edge). Must divide N **and** be
      tileable; auto-picked as the largest such divisor under the per-N-bucket base. At the default
      power-of-two N values the result is identical to the hand-written table this replaced.
    * ``threads_per_row`` -- lanes cooperating on one row's N-reduction. Must be a power of two
      (butterfly shuffle) and divide N; auto-picked as the largest such divisor under the base.

    The bucket bases were autotuned at power-of-two N. Thin-N (<= 256) prefers a whole-row or
    half-row sub-tile with few lanes per token, which wins at large M.

    Args:
        N: Feature extent. Must already have cleared the front door's swizzle-atom floor
            (``N % 16 == 0`` at 16-bit, ``N % 32 == 0`` at fp32) and ``N <= 1024``; this function
            does not re-check, and an N below the floor yields a config the functor then rejects.
        dtype: Element type. Accepted for symmetry with the per-dtype floors documented at the front
            door; the buckets themselves are keyed on N alone, which is why it is unused here.

    Returns:
        ``(blk_m, subt, threads_per_row)``, all positive, with ``subt`` dividing N and
        ``threads_per_row`` a power-of-two divisor of N.
    """
    blk_m = 16
    # Thin-N (<=256): WHOLE-row or half-row sub-tile + few lanes/token wins large-M.
    if N <= 128:
        base_subt, base_tpr = 128, 16  # subt=N one-shot, 256 threads
    elif N <= 256:
        base_subt, base_tpr = 128, 16  # subt=128 (n_sub_n=2), 256 threads
    elif N <= 512:
        base_subt, base_tpr = 32, 32  # 512 threads
    else:  # N <= 1024
        base_subt, base_tpr = 64, 64  # 1024 threads, 16 elems/thread
    return (blk_m, _auto_subt(N, base_subt), _auto_tpr(N, base_tpr, blk_m))


class LayerNormTransposeParams(ReductionParams):
    """Compile-time parameters of the transposing-load LayerNorm.

    Extends :class:`~fold_cp_ops._internal.reduction_base.ReductionParams` rather than
    :class:`LayerNormParams`: ``reload_from`` is a knob of the ``cp.async`` mainloop, which this
    variant does not have -- the whole row is already resident in SMEM, so there is nothing to
    re-read.

    Attributes:
        M: Token (row) extent of the output, i.e. the CONTIGUOUS extent of the ``(N, M)`` view the
            TMA reads. It is a compile-time parameter here and not a symbolic dim (unlike the
            row-major variant's batch axis) because the grid is ``ceil_div(M, blk_m)`` and the
            kernel's own row guard compares against it; a new M therefore recompiles.
        blk_m: Output rows per CTA, and the M-edge of a transpose sub-tile. Must divide the swizzle
            atom's expectations -- 16 is the only value the shipped config emits.
        subt: N-edge of a transpose sub-tile. Must divide ``N`` and satisfy
            :func:`_subt_is_tileable`; a value failing the latter **aborts the process** inside the
            DSL rather than raising.
        threads_per_row: Lanes cooperating on one row's reduction. Must be a power of two (butterfly
            shuffle) and divide ``N``; ``threads_per_row * blk_m`` must be a warp multiple.
        do_transpose: ``"SMEM"`` | ``"shuffle"`` | ``"ldmatrix"`` -- how the per-thread reduction
            fragment is built from the staged tile. Bit-identical output between ``"SMEM"`` and
            ``"shuffle"``; ``"ldmatrix"`` is a documented blocker (see
            :meth:`LayerNormTransposeSm90._ldmatrix_fragment`).
        buffer_align_bytes: Alignment of the two SMEM tiles, in bytes. 1024 keeps the swizzled tiles
            on a 128-B boundary with room for the TMA's own alignment; lowering it does not raise,
            it produces a TMA descriptor the hardware rejects at launch.
    """

    M: int
    blk_m: int
    subt: int
    threads_per_row: int
    do_transpose: str
    buffer_align_bytes: int = 1024


class LayerNormTransposeCallParams(TemplateParams):
    """The operand facts the transposing variant reads off its tensors, bound at ``__call__``.

    Only the two MAJORS. Everything else the functor needs from an operand -- every SMEM layout,
    the TMA transaction size, the shared-storage struct -- is a property of these plus the
    construction parameters, which is why they are the whole call-phase surface.

    ``LayoutEnum`` is a plain ``Enum`` and therefore a compile-time value: it is read at trace time
    and folded in, carrying no MLIR value. That is what makes an operand's major declarable at all.

    Attributes:
        xt_major: Major mode of the ``(N, M)`` view of the LayoutLeft input. Decides the swizzle
            atom for the staged load tile; the wrong one silently transposes it.
        o_major: Major mode of the ``(M, N)`` row-major output. Decides the ``sT`` swizzle, and is
            read only on the ``"SMEM"`` path.
    """

    xt_major: LayoutEnum
    o_major: LayoutEnum


class LayerNormTransposeSm90(LayerNorm):
    """LayerNorm over the STRIDED axis: TMA+swizzle transposing load, row-major store.

    Computes exactly what :class:`LayerNorm` computes -- ``x_hat = (x - mean) * rstd``, then the
    affine -- and shares its reduction (:func:`~fold_cp_ops._internal.reduce.row_reduce`) and its
    reduction-buffer apparatus. **What differs is the load, and only the load.** The base stages an
    ``(M, N)`` row-major tile through ``cp.async``; this variant TMA-bulk-loads ``(subt, blk_m)``
    sub-blocks of the ``(N, M)`` view into a 128-B-swizzled tile and transposes them into one wide
    ``(blk_m, N)`` fragment. So ``__call__`` and ``kernel`` are overridden wholesale, and nothing of
    :class:`LayerNorm`'s own body runs -- the inheritance says "same operation, different load",
    it does not share code.

    **SM90 only**, by construction: the load is a ``cpasync.CopyBulkTensorTileG2SOp`` TMA and the
    SMEM layout comes from the SM90 WGMMA swizzle atoms.

    Input requirements (all enforced at the front door, :func:`_transpose_fwd`, not here):
    ``x`` LayoutLeft ``(M, N)`` stride ``(1, M)``; ``N`` a multiple of the swizzle atom
    (16 elements at 16-bit, 32 at fp32); ``N <= 1024``; ``M`` 16-byte aligned. Constructing this
    functor directly with a violating ``(N, subt, threads_per_row)`` raises ``ValueError`` below --
    except for a non-tileable ``subt``, which cannot be caught: it SIGABRTs inside the DSL.
    """

    #: Post-construction ``const_expr`` gate not declared in either parameter pack. ``reload_from``
    #: selects which staged copy the transposed reload reads, is written after construction, and is
    #: read inside a ``const_expr``, so it decides emitted code while ``param_dict()`` cannot see it.
    #: Pinned against an AST scan by ``tests/_internal/compile_time/test_template_params.py``.
    COMPILE_GATED_ATTRS = ("reload_from",)

    Params = LayerNormTransposeParams
    CallParams = LayerNormTransposeCallParams

    #: Transpose-into-registers strategies, ``const_expr``-selected in the kernel body. Order is not
    #: meaningful; membership is what ``__init__`` validates against.
    DO_TRANSPOSE = ("SMEM", "ldmatrix", "shuffle")

    def __init__(
        self,
        dtype: Type[cutlass.Numeric],
        M: int,
        N: int,
        blk_m: int,
        subt: int,
        threads_per_row: int,
        do_transpose: str = "SMEM",
    ):
        """Bind the transposing variant's compile-time parameters.

        Args:
            dtype: Element type of ``x`` and of the output. ``"ldmatrix"`` additionally requires a
                16-bit type -- ``LDSM.trans`` has no 32-bit form.
            M: Token extent. Positive. Recompiles per value (see :class:`LayerNormTransposeParams`).
            N: Feature extent, the reduced axis. Must be a multiple of ``subt`` and of
                ``threads_per_row``.
            blk_m: Output rows per CTA. Positive; ``threads_per_row * blk_m`` must be a warp
                multiple.
            subt: Sub-tile N-edge. Must divide ``N``. Must ALSO satisfy :func:`_subt_is_tileable`,
                which is **not checked here** because it cannot be: a non-tileable value aborts the
                process inside ``cute.slice_`` with no catchable exception. Use
                :func:`_transpose_default_config`.
            threads_per_row: Lanes per row. Must be a power of two and divide ``N``.
            do_transpose: One of :data:`DO_TRANSPOSE`.

        Raises:
            TypeError: If any argument is a runtime value rather than a compile-time constant.
            ValueError: If ``do_transpose`` is unknown, if ``"ldmatrix"`` is asked of a 32-bit
                dtype, if ``subt`` or ``threads_per_row`` does not divide ``N``, if
                ``threads_per_row`` is not a power of two, or if ``threads_per_row * blk_m`` is not
                a multiple of the warp size.
        """
        self._require_compile_time(
            dtype=dtype,
            M=M,
            N=N,
            blk_m=blk_m,
            subt=subt,
            threads_per_row=threads_per_row,
            do_transpose=do_transpose,
        )
        # `raise`, not `assert`, for every one of these: `python -O` strips asserts, and a functor
        # that silently accepts a tile it cannot build is the failure this repo refuses to ship.
        if do_transpose not in self.DO_TRANSPOSE:
            raise ValueError(
                f"do_transpose={do_transpose!r} is not one of {list(self.DO_TRANSPOSE)}."
            )
        if do_transpose == "ldmatrix" and dtype.width != 16:
            raise ValueError(
                f"do_transpose='ldmatrix' is 16-bit (bfloat16/float16) only; got {dtype} "
                f"({dtype.width}-bit). LDSM.trans has no 32-bit form."
            )
        if N % subt != 0:
            raise ValueError(
                f"N ({N}) must be a multiple of subt ({subt}); use _transpose_default_config(N)."
            )
        if threads_per_row & (threads_per_row - 1):
            raise ValueError(
                f"threads_per_row ({threads_per_row}) must be a power of 2 -- row_reduce finishes "
                f"with a butterfly shuffle_sync_bfly, which reduces over the wrong lane set (a "
                f"wrong mean, not an error) for any other group size."
            )
        if N % threads_per_row != 0:
            raise ValueError(f"N ({N}) must be divisible by threads_per_row ({threads_per_row}).")
        if (threads_per_row * blk_m) % cute.arch.WARP_SIZE != 0:
            raise ValueError(
                f"threads_per_row * blk_m ({threads_per_row} * {blk_m} = "
                f"{threads_per_row * blk_m}) must be a multiple of the warp size "
                f"({cute.arch.WARP_SIZE})."
            )
        # stage=2: one reduction-buffer slot for the mean, one for the variance.
        # cluster_n=1: no cluster is needed -- the whole row fits one CTA's SMEM.
        self._bind_params(
            dtype=dtype,
            N=N,
            stage=2,
            cluster_n=1,
            M=M,
            blk_m=blk_m,
            subt=subt,
            threads_per_row=threads_per_row,
            do_transpose=do_transpose,
        )

    @property
    def n_sub_n(self) -> int:
        """Number of transpose sub-tiles along N.

        Derived rather than stored so it cannot desynchronize from ``N`` and ``subt``.

        Returns:
            ``N // subt``, exact because ``__init__`` rejects a ``subt`` that does not divide N.
        """
        return self.N // self.subt

    @property
    def num_threads(self) -> int:
        """Threads per CTA.

        Returns:
            ``threads_per_row * blk_m`` -- one lane group per output row, ``blk_m`` rows per CTA.
            A multiple of the warp size, checked in ``__init__``.
        """
        return self.threads_per_row * self.blk_m

    def _threads_per_row(self) -> int:
        """Lanes cooperating on one row. Overrides :class:`LayerNorm`'s N-keyed ladder.

        Returns:
            The bound ``threads_per_row`` parameter. The base class derives this from N; here it is
            a declared parameter because it is co-chosen with ``subt`` against the same divisibility
            constraints, so deriving it independently would let the two disagree.
        """
        return self.threads_per_row

    def _num_threads(self) -> int:
        """Threads per block. Overrides :class:`ReductionBase`'s N-keyed 128/256 rule.

        Returns:
            :attr:`num_threads`, i.e. ``threads_per_row * blk_m``.
        """
        return self.num_threads

    @cached_property
    def ld_layout(self) -> cute.ComposedLayout:
        """Staged input tile: ``(subt, blk_m, n_sub_n)``, swizzled like the ``(N, M)`` view.

        A ``cached_property`` rather than a local of ``__call__``, because ``tma_copy_bytes`` and
        ``shared_storage`` derive from it and both are read inside the TRACED ``kernel``. A value
        assigned mid-trace is mutable state that can desynchronize from an already-compiled kernel;
        a property of the two parameter packs cannot. Same rule ``GemmSm90.a_smem_layout_staged``
        follows.

        Returns:
            The swizzled layout, ``blk_m`` (the contiguous M mode) being the swizzled one.
        """
        return sm90_utils.make_smem_layout(
            self.dtype, self.xt_major, (self.subt, self.blk_m), self.n_sub_n
        )

    @cached_property
    def ld_layout_one(self) -> cute.ComposedLayout:
        """:attr:`ld_layout` with the stage mode dropped -- the TMA atom's box.

        Returns:
            The rank-2 slice. **Aborts the process** (SIGABRT, no traceback) rather than raising if
            ``subt`` is not tileable; see :func:`_subt_is_tileable`.
        """
        return cute.slice_(self.ld_layout, (None, None, 0))

    @cached_property
    def st_layout(self) -> cute.Layout:
        """Assembled tile ``sT``: ``(blk_m, N)`` swizzled like the row-major output.

        Only the ``"SMEM"`` path materializes it; the register-gather paths get a rank-3 dummy so
        the name is in scope either way (the DSL's scope rule forbids defining it inside the
        ``const_expr`` branch). Building the real one unconditionally would trip an INTER-swizzle
        crash when ``N * dtype_bits % 256 != 0`` -- bf16 N=40/72 -- on paths that never use it.

        Returns:
            The swizzled ``(blk_m, N)`` layout, or a ``(blk_m, 1, 1)`` placeholder.
        """
        if self.do_transpose != "SMEM":
            return cute.make_layout((self.blk_m, 1, 1))
        return sm90_utils.make_smem_layout(self.dtype, self.o_major, (self.blk_m, self.N), 1)

    @cached_property
    def tma_copy_bytes(self) -> int:
        """Bytes one ``(subt, blk_m)`` TMA sub-block deposits, for the mbarrier's expect-tx.

        Returns:
            ``size_in_bytes(dtype, ld_layout_one)``. The kernel multiplies it by ``n_sub_n`` to get
            the whole block's transaction count; an under-count leaves the barrier waiting forever.
        """
        return cute.size_in_bytes(self.dtype, self.ld_layout_one)

    @cached_property
    def shared_storage(self):
        """The SMEM struct: the load mbarrier, the reduction buffer, ``sT`` and ``sX``.

        ``sT`` collapses to a size-1 placeholder off the ``"SMEM"`` path -- the register-gather
        paths build the reduction fragment straight from ``sX``, so the round-trip buffer is pure
        waste there and dropping it buys occupancy. The field stays present either way so the
        struct's shape does not depend on the path.

        Returns:
            A ``@cute.struct`` class, to be passed to ``SmemAllocator.allocate``.
        """
        st_cosize = const_expr(cute.cosize(self.st_layout) if self.do_transpose == "SMEM" else 1)
        red_layout, ld_layout = self.red_layout, self.ld_layout

        @cute.struct
        class SharedStorage:
            load_mbar: cute.struct.MemRange[cutlass.Int64, 1]
            reduction: cute.struct.Align[
                cute.struct.MemRange[self.reduction_dtype, cute.cosize(red_layout)], 8
            ]
            sT: cute.struct.Align[
                cute.struct.MemRange[self.dtype, st_cosize], self.buffer_align_bytes
            ]
            sX: cute.struct.Align[
                cute.struct.MemRange[self.dtype, cute.cosize(ld_layout)], self.buffer_align_bytes
            ]

        return SharedStorage

    @cached_property
    def red_layout(self) -> cute.Layout:
        """:meth:`_reduction_layout` as a property, so a traced method may read it.

        Returns:
            The reduction staging layout. Identical to the method, which is kept because
            :class:`ReductionBase` names it.
        """
        return self._reduction_layout()

    def _reduction_layout(self) -> cute.Layout:
        """Shape the reduction staging buffer for this CTA geometry.

        Purpose
            The base's :meth:`ReductionBase._get_reduction_buffer_layout` reads the geometry off a
            tiled copy's TV layout. This variant has no staging tiled copy -- the load is a TMA --
            so the same shape is built directly from ``num_threads`` and ``threads_per_row``.

        Returns:
            A layout of shape ``(num_warps // warps_per_row, (warps_per_row, cluster_n), stage)``
            with ``order=(1, 0, 2)``, i.e. warps-per-row fastest, which is the order
            ``row_reduce`` indexes it in. ``cluster_n`` is 1 here, so the second mode is
            ``(warps_per_row, 1)``.
        """
        num_warps = self.num_threads // cute.arch.WARP_SIZE
        warps_per_row = max(self.threads_per_row // cute.arch.WARP_SIZE, 1)
        return cute.make_ordered_layout(
            (num_warps // warps_per_row, (warps_per_row, self.cluster_n), self.stage),
            order=(1, 0, 2),
        )

    def _ldmatrix_fragment(self, sX, xfrag, ldmatrix_copy, tidx):
        """LDSM.T hardware transpose of ``sX`` into the reduction fragment. **NOT IMPLEMENTED.**

        BLOCKER (diagnosed, not reconciled): the ``ldmatrix.x4.trans`` native fragment delivers, per
        lane ``l`` / register ``r``, ``sX_tile[row, col]`` with the EMPIRICALLY-VALIDATED map
        ``row(feature) = (l>>2) + 8*((r>>1)&1)``, ``col(token) = (l&3)*2 + (r&1) + 8*(r>>2)``.
        A single lane's 8 registers therefore span 4 DISTINCT tokens ``{c0, c0+1, c0+8, c0+9}`` with
        ``c0 = (l&3)*2``, each with 2 feature-rows. That is a fundamentally different fragment from
        the reduction's "lane ``(row_in_blk, col_lane)`` owns ONE token's ``tpr``-strided features"
        layout that :func:`~fold_cp_ops._internal.reduce.row_reduce` and the
        ``(num_warps, warps_per_row, stage)`` reduction buffer are built around. Reconciling them
        needs a bespoke two-stage warp reduction (in-register sum over each lane's 4 owned tokens,
        then a strided shuffle-xor reduction over the 8 lanes sharing ``l&3``, repeated over N/16
        tiles) AND a matching normalize+store mapping -- a from-scratch reduction.

        The ``"shuffle"`` path already removes the ``sT`` round-trip this one was for, and validates
        **bit-identical** to ``"SMEM"``, so the blocker costs no capability.

        Args:
            sX: The staged ``(subt, blk_m, n_sub_n)`` swizzled input tile.
            xfrag: The register fragment to fill, ``(N // threads_per_row,)`` fp32.
            ldmatrix_copy: ``(tiled_copy, tiled_mma)`` built in ``__call__``.
            tidx: This thread's index within the block.

        Raises:
            NotImplementedError: Always. The front door refuses ``do_transpose="ldmatrix"`` before
                a functor is ever built, so this fires only for a direct construction.
        """
        raise NotImplementedError(
            "do_transpose='ldmatrix' is not implemented: the native LDSM.x4.trans fragment cannot "
            "reuse row_reduce (see _ldmatrix_fragment's docstring); it needs a bespoke warp "
            "reduction. Use 'shuffle', which removes the same SMEM round-trip and is bit-identical "
            "to 'SMEM'."
        )

    @cute.jit
    def __call__(
        self,
        mXt: cute.Tensor,  # (N, M) row-major view of the LayoutLeft input (M contiguous)
        mW: cute.Tensor,  # (N,) fp32
        mB: cute.Tensor,  # (N,) fp32
        mO: cute.Tensor,  # (M, N) row-major output (N contiguous)
        eps: Float32,
        stream: cuda.CUstream,
    ):
        """Build the TMA atom, the SMEM layouts and the reduction layout, then launch.

        Everything emitted here is layout/type algebra plus one ``.launch()``; the arithmetic is all
        in :meth:`kernel`.

        Args:
            mXt: The ``(N, M)`` **row-major** view of the LayoutLeft input -- i.e. ``x.mT``, with M
                contiguous. Element type must equal the bound ``dtype``; a mismatch is a wrong-typed
                TMA descriptor, not an error.
            mW: ``(N,)`` fp32 affine scale. Read with a broadcast load, so its layout is free.
            mB: ``(N,)`` fp32 affine shift.
            mO: ``(M, N)`` **row-major** output, N contiguous. Written with a direct indexed store;
                a non-contiguous N mode would be written as if it were contiguous.
            eps: Added to the variance before the ``rsqrt``. A runtime value.
            stream: The launch stream.

        Returns:
            None. Launches ``ceil_div(M, blk_m)`` CTAs of ``num_threads`` threads as a side effect,
            and sets ``self.tma_copy_bytes`` / ``self.shared_storage`` for :meth:`kernel` to read.
        """
        assert mXt.element_type == self.dtype
        subt, blk_m = self.subt, self.blk_m
        # The two operand majors are the only CALL-phase facts this functor needs; every layout
        # below is a property of them plus the construction parameters. Binding them here is what
        # lets `tma_copy_bytes` and `shared_storage` be derived properties instead of attributes
        # assigned mid-trace -- see the class docstring.
        self._bind_call_params(
            xt_major=LayoutEnum.from_tensor(mXt),  # (N, M): M is contiguous (major) mode
            o_major=LayoutEnum.from_tensor(mO),  # (M, N): N contiguous
        )
        ld_layout, st_layout = self.ld_layout, self.st_layout

        # G2S load atom over the (N, M) input with a (subt, blk_m) box.
        tma_atom_load, tma_tensor_x = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(),
            mXt,
            self.ld_layout_one,
            (subt, blk_m),
            num_multicast=1,
        )

        # Coalesced row-major store descriptor for the (blk_m, N) output tile. `tiled_copy_O` and
        # `store_vec` are threaded to the kernel and NOT read there: the store is a direct indexed
        # write (see the epilogue of `kernel`). Both are pure layout algebra and emit no
        # instructions, and they are kept because this is a port -- dropping a traced argument
        # changes the compiled entry's signature, which is the one thing a bring-back may not do.
        store_vec = const_expr(math.gcd(self.N, max(1, 128 // mO.element_type.width)))
        tiled_copy_O = tiled_copy_2d(
            mO.element_type, self.threads_per_row, self.num_threads, store_vec
        )

        red_layout = self.red_layout

        # ldmatrix path: a m16n8k16 BF16 TiledMMA + its operand-A LDSM.x4.trans TiledCopy, used to
        # hardware-transpose 16x16 sub-tiles of sX (feature x token) into registers.
        ldmatrix_copy = None
        if const_expr(self.do_transpose == "ldmatrix"):
            from cutlass.cute.nvgpu import warp as _warp

            mma_op = _warp.MmaF16BF16Op(self.dtype, Float32, (16, 8, 16))
            tiled_mma = cute.make_tiled_mma(mma_op, cute.make_layout((1, 1, 1)))
            ld_atom = cute.make_copy_atom(
                _warp.LdMatrix8x8x16bOp(transpose=True, num_matrices=4), self.dtype
            )
            ldmatrix_copy = (cute.make_tiled_copy_A(ld_atom, tiled_mma), tiled_mma)

        # MATERIALIZE both derived values HERE, in `__call__`'s region, before the launch. They are
        # `cached_property`, so a first touch inside the traced `kernel` would build them there --
        # and `shared_storage` calls `cute.cosize` on layouts created in THIS region, which the
        # verifier rejects: "'cute.cosize' op using value defined outside the region ... required by
        # region isolation constraints". Being lazy is what makes them safe against desynchronizing;
        # being forced here is what keeps every op they emit in the region that owns their inputs.
        _ = self.shared_storage, self.tma_copy_bytes

        grid = (cute.ceil_div(self.M, blk_m), 1, 1)
        self.kernel(
            tma_atom_load,
            tma_tensor_x,
            mW,
            mB,
            mO,
            eps,
            ld_layout,
            st_layout,
            red_layout,
            tiled_copy_O,
            store_vec,
            ldmatrix_copy,
        ).launch(grid=grid, block=[self.num_threads, 1, 1], stream=stream)

    @cute.kernel
    def kernel(
        self,
        tma_atom_load: cute.CopyAtom,
        mXt: cute.Tensor,  # (N, M)
        mW: cute.Tensor,
        mB: cute.Tensor,
        mO: cute.Tensor,  # (M, N) row-major
        eps: Float32,
        ld_layout,
        st_layout,
        red_layout,
        tiled_copy_O: cute.TiledCopy,
        store_vec: cutlass.Constexpr[int],
        ldmatrix_copy,
    ):
        """One CTA: TMA-load and transpose ``blk_m`` output rows, reduce, normalize, store.

        Args:
            tma_atom_load: The bulk-tensor G2S atom over the ``(N, M)`` input.
            mXt: The TMA tensor for the ``(N, M)`` view.
            mW: ``(N,)`` fp32 affine scale.
            mB: ``(N,)`` fp32 affine shift.
            mO: ``(M, N)`` row-major output.
            eps: Variance epsilon.
            ld_layout: The staged ``(subt, blk_m, n_sub_n)`` swizzled SMEM layout.
            st_layout: The assembled ``(blk_m, N)`` swizzled SMEM layout, or the rank-3 dummy when
                ``do_transpose != "SMEM"`` (in which case ``sT`` is never materialized).
            red_layout: The reduction staging layout from :meth:`_reduction_layout`.
            tiled_copy_O: Unused -- see the note in :meth:`__call__`.
            store_vec: Unused -- see the note in :meth:`__call__`.
            ldmatrix_copy: ``(tiled_copy, tiled_mma)`` for the ``"ldmatrix"`` path, else None.

        Returns:
            None. Writes ``mO`` and nothing else -- **no rstd / mean are emitted**, which is why the
            front door refuses ``return_rstd`` / ``return_mean`` on this path.
        """
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        subt, blk_m = self.subt, self.blk_m
        N = self.N
        tpr = self.threads_per_row

        if warp_idx == 0:
            cpasync.prefetch_descriptor(tma_atom_load)

        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)
        mbar_ptr = storage.load_mbar.data_ptr()

        # sT : (blk_m, N) swizzled row-major tile (N contiguous), 1 stage. Only the "SMEM" path uses
        # it; the register-transpose paths build the reduction fragment straight from sX (no sT).
        if const_expr(self.do_transpose == "SMEM"):
            sT = storage.sT.get_tensor(st_layout.outer, swizzle=st_layout.inner)[(None, None, 0)]
        # sX : (subt, blk_m, n_sub_n) swizzled input sub-tiles, all in flight at once.
        sX = storage.sX.get_tensor(ld_layout.outer, swizzle=ld_layout.inner)
        reduction_buffer = cute.make_tensor(storage.reduction.data_ptr(), red_layout)

        # ---- init the TMA-load mbarrier ----
        if tidx == 0:
            cute.arch.mbarrier_init(mbar_ptr, 1)
        cute.arch.mbarrier_init_fence()
        cute.arch.barrier()

        # This CTA owns output rows [bidx*blk_m : +blk_m], all N. Input (N, M): sub-block sn covers
        # N-rows [sn*subt:+subt] and M-cols [bidx*blk_m:+blk_m].
        total_bytes = const_expr(self.tma_copy_bytes * self.n_sub_n)
        if warp_idx == 0:
            with cute.arch.elect_one():
                cute.arch.mbarrier_arrive_and_expect_tx(mbar_ptr, total_bytes)
            # Build the (M-block) gmem tensor over the full N once: gXt_full (subt, blk_m, n_sub_n),
            # the n_sub_n tiling of the N axis. TMA-partition both sides keeping the n_sub_n mode.
            gXt_blk = cute.local_tile(mXt, (self.N, blk_m), (0, bidx))  # (N, blk_m)
            gXt_full = cute.flat_divide(gXt_blk, (subt, blk_m))  # (subt, blk_m, n_sub_n, 1)
            gXt_full = gXt_full[(None, None, None, 0)]  # (subt, blk_m, n_sub_n)
            tXsX_all, tXgX_all = cpasync.tma_partition(
                tma_atom_load,
                0,
                cute.make_layout(1),
                cute.group_modes(sX, 0, 2),  # ((subt*blk_m), n_sub_n)
                cute.group_modes(gXt_full, 0, 2),  # ((subt*blk_m), n_sub_n)
            )
            for sn in cutlass.range_constexpr(self.n_sub_n):
                cute.copy(
                    tma_atom_load, tXgX_all[(None, sn)], tXsX_all[(None, sn)], tma_bar_ptr=mbar_ptr
                )
        cute.arch.mbarrier_wait(mbar_ptr, phase=0)

        row_in_blk = tidx // tpr
        col_lane = tidx % tpr
        elems_per_thread = const_expr(N // tpr)
        xfrag = cute.make_rmem_tensor(elems_per_thread, Float32)

        if const_expr(self.do_transpose == "SMEM"):
            # ---- threaded SMEM->SMEM transpose: sT[m, sn*subt + n] = sX[n, m, sn] ----
            n_elems_sub = const_expr(subt * blk_m)
            for sn in cutlass.range_constexpr(self.n_sub_n):
                sX_sn = sX[(None, None, sn)]  # (subt, blk_m)
                for e in cutlass.range(tidx, n_elems_sub, self.num_threads, unroll=8):
                    nn = e // blk_m  # N-within-sub-tile (0..subt-1)
                    mm = e % blk_m  # M / output row (0..blk_m-1)
                    sT[mm, sn * subt + nn] = sX_sn[nn, mm]
            cute.arch.barrier()
            # ---- LayerNorm fragment from sT, per output row ----
            for j in cutlass.range_constexpr(elems_per_thread):
                nf = col_lane + j * tpr
                xfrag[j] = sT[row_in_blk, nf].to(Float32)
        elif const_expr(self.do_transpose == "shuffle"):
            # ---- direct register gather from sX: NO sT round-trip ----
            # The reduction needs xfrag[j] = x[token=row_in_blk, feature = col_lane + j*tpr], i.e.
            # the transpose of sX. sX[n, m, sn] holds feature (sn*subt + n), token m. So we read
            # sX[feature%subt, row_in_blk, feature//subt] directly: each thread gathers ITS OWN N
            # features for ITS OWN token straight out of the swizzled SMEM input tile. Validated
            # bit-identical to the sT fragment. NO cross-lane exchange is required -- despite the
            # name, the transpose here is a per-thread index permutation, not a data movement, so a
            # shuffle instruction on this path would be a misreading.
            for j in cutlass.range_constexpr(elems_per_thread):
                feature = col_lane + j * tpr
                sn = feature // subt
                nn = feature % subt
                xfrag[j] = sX[nn, row_in_blk, sn].to(Float32)
        else:  # "ldmatrix"
            self._ldmatrix_fragment(sX, xfrag, ldmatrix_copy, tidx)

        local_sum = Float32(0.0)
        for j in cutlass.range_constexpr(elems_per_thread):
            local_sum += xfrag[j]
        sum_x = row_reduce(
            local_sum,
            cute.ReductionOp.ADD,
            tpr,
            reduction_buffer[None, None, 0],
            None,
            init_val=0.0,
        )
        mean = sum_x / Float32(N)

        local_var = Float32(0.0)
        for j in cutlass.range_constexpr(elems_per_thread):
            d = xfrag[j] - mean
            local_var += d * d
        sum_sq = row_reduce(
            local_var,
            cute.ReductionOp.ADD,
            tpr,
            reduction_buffer[None, None, 1],
            None,
            init_val=0.0,
        )
        rstd = cute.math.rsqrt(sum_sq / Float32(N) + eps, fastmath=True)

        # normalize + affine, store coalesced to the (M, N) row-major output.
        row_global = bidx * blk_m + row_in_blk
        gO = cute.local_tile(mO, (blk_m, N), (bidx, 0))  # (blk_m, N)
        if row_global < self.M:
            for j in cutlass.range_constexpr(elems_per_thread):
                n = col_lane + j * tpr
                xh = (xfrag[j] - mean) * rstd
                y = xh * mW[n].to(Float32) + mB[n].to(Float32)
                gO[row_in_blk, n] = y.to(mO.element_type)


# ───────────────────────────── transposing variant: the launch path ─────────────────────────────
# A lean tvm-ffi host launch (no per-call from_dlpack, no stream wrap; torch tensors pass through
# the FFI and the env stream is used), mirroring `_layernorm_fwd` above so the launch overhead of
# the two variants is comparable.

#: ``do_transpose`` as an int, because the ``torch.library`` schema takes ints and string arguments
#: are brittle across the FFI boundary. The DSL constructor takes the string, so this pair is the
#: single source of truth for the mapping.
_DO_TRANSPOSE_ENUM = {"SMEM": 0, "ldmatrix": 1, "shuffle": 2}
_DO_TRANSPOSE_NAME = {v: k for k, v in _DO_TRANSPOSE_ENUM.items()}


@torch.library.custom_op(
    "fold_cp_ops::_layernorm_transpose_fwd",
    mutates_args=("out",),
    device_types="cuda",
    schema=(
        "(Tensor xt, Tensor weight, Tensor bias, Tensor(a3!) out, float eps, int blk_m, int subt, "
        "int threads_per_row, int do_transpose) -> ()"
    ),
)
def _layernorm_transpose_fwd_op(
    xt: Tensor,
    weight: Tensor,
    bias: Tensor,
    out: Tensor,
    eps: float,
    blk_m: int,
    subt: int,
    threads_per_row: int,
    do_transpose: int,
) -> None:
    """Compile-and-launch the transposing LayerNorm. Registered so it is traceable and mutates ``out``.

    Args:
        xt: The ``(N, M)`` row-major view of the LayoutLeft input. Its shape is read as ``(N, M)``,
            so passing the ``(M, N)`` tensor here transposes the meaning of every gate silently.
        weight: ``(N,)`` fp32 affine scale.
        bias: ``(N,)`` fp32 affine shift.
        out: ``(M, N)`` row-major output, **written in place**.
        eps: Variance epsilon.
        blk_m: Output rows per CTA.
        subt: Sub-tile N-edge. Must be tileable (:func:`_subt_is_tileable`).
        threads_per_row: Lanes per row.
        do_transpose: The int form; see :data:`_DO_TRANSPOSE_ENUM`.

    Returns:
        None. Every gate lives in :func:`_transpose_fwd`; this entry trusts its caller.
    """
    N, M = xt.shape  # xt is the (N, M) row-major view of the (M, N) LayoutLeft input
    dtype = torch2cute_dtype_map[xt.dtype]
    _compile_layernorm_transpose_fwd(dtype, M, N, blk_m, subt, threads_per_row, do_transpose)(
        xt, weight, bias, out, eps
    )


@_layernorm_transpose_fwd_op.register_fake
def _layernorm_transpose_fwd_fake(
    xt, weight, bias, out, eps, blk_m, subt, threads_per_row, do_transpose
) -> None:
    """Meta kernel: under ``--compile-only`` it compiles, otherwise it does nothing.

    See ``_layernorm_fwd_fake`` for why ``register_fake`` is needed at all.

    Args:
        xt: Meta tensor for the ``(N, M)`` view. A symbolic M means the shape is not known, so the
            compile is skipped rather than keyed on a ``SymInt``.
        weight: Unused here.
        bias: Unused here.
        out: Unused here.
        eps: Unused here.
        blk_m: Part of the compile key.
        subt: Part of the compile key.
        threads_per_row: Part of the compile key.
        do_transpose: Part of the compile key.

    Returns:
        None.
    """
    from fold_cp_ops._internal.cache_utils import COMPILE_ONLY

    if COMPILE_ONLY and not isinstance(xt.size(1), torch.SymInt):
        N, M = xt.shape
        _compile_layernorm_transpose_fwd(
            torch2cute_dtype_map[xt.dtype], M, N, blk_m, subt, threads_per_row, do_transpose
        )


@jit_cache
def _compile_layernorm_transpose_fwd(dtype, M, N, blk_m, subt, threads_per_row, do_transpose):
    """Compile one transposing-LayerNorm configuration; cached on disk by the argument tuple.

    Note that **M is part of the key**, unlike the row-major variant's batch axis: the grid is
    ``ceil_div(M, blk_m)`` and the kernel's row guard compares against it, so M is a compile-time
    parameter of the functor rather than a symbolic dim.

    Args:
        dtype: cutlass element type of ``x`` and the output.
        M: Token extent. Positive.
        N: Feature extent. Positive; must satisfy the front door's floors.
        blk_m: Output rows per CTA.
        subt: Sub-tile N-edge; must divide N and be tileable.
        threads_per_row: Lanes per row; power of two dividing N.
        do_transpose: The INT form (see :data:`_DO_TRANSPOSE_ENUM`) -- ints hash stably into the
            disk cache key, and the constructor is handed the name.

    Returns:
        The compiled, tvm-ffi-enabled callable, invoked as ``(xt, weight, bias, out, eps)``.
    """
    w = dtype.width
    div_m, div_n = math.gcd(M, 128 // w), math.gcd(N, 128 // w)
    xt_cute = fake_tensor(dtype, (N, M), div_m)  # (N, M) row-major, M contiguous
    o_cute = fake_tensor(dtype, (M, N), div_n)  # (M, N) row-major, N contiguous
    w_cute = fake_tensor(Float32, (N,), math.gcd(N, 4))
    b_cute = fake_tensor(Float32, (N,), math.gcd(N, 4))
    return cute.compile(
        LayerNormTransposeSm90(
            dtype, M, N, blk_m, subt, threads_per_row, _DO_TRANSPOSE_NAME[do_transpose]
        ),
        xt_cute,
        w_cute,
        b_cute,
        o_cute,
        Float32(0),  # eps, just for compilation
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi",
    )


def _transpose_fwd(
    x: Tensor,
    weight: Tensor,
    bias: Optional[Tensor] = None,
    eps: float = 1e-6,
    return_rstd: bool = False,
    return_mean: bool = False,
    blk_m: Optional[int] = None,
    subt: Optional[int] = None,
    threads_per_row: Optional[int] = None,
    do_transpose: str = "auto",
    select: str = "heuristic",
    _config: Optional[AutotuneConfig] = None,
) -> Tensor:
    """LayerNorm over N of a LayoutLeft ``(M, N)`` input, written row-major. The ``transpose=True`` body.

    Reached only through ``layernorm_fwd(..., transpose=True)``; every gate below is a front-door
    refusal, so no unsupported shape ever reaches the DSL.

    Args:
        x: ``(M, N)`` **LayoutLeft** -- stride ``(1, M)``, so the LN reduce axis N is strided by M.
            A row-major ``x`` is refused: it would be read as if it were LayoutLeft and produce a
            wrong answer with no error.
        weight: ``(N,)`` fp32 affine scale. A 16-bit one is refused -- the gain is applied in fp32.
        bias: ``(N,)`` fp32 affine shift. Required on this path (the kernel always adds it); passing
            None is refused rather than silently zero-filled.
        eps: Added to the variance before the ``rsqrt``.
        return_rstd: Must be False -- this kernel does not write the row statistics out.
        return_mean: Must be False, same reason.
        blk_m: Output rows per CTA, or None to auto-pick.
        subt: Sub-tile N-edge, or None to auto-pick the largest TILEABLE divisor of N under the
            bucket base. **Supplying a non-tileable value aborts the process** -- see
            :func:`_subt_is_tileable`.
        threads_per_row: Lanes per row, or None to auto-pick the largest power-of-two divisor of N
            under the bucket base.
        do_transpose: How the reduction fragment is built.

            * ``"SMEM"`` -- threaded SMEM->SMEM scatter through an ``sT`` round-trip tile.
            * ``"shuffle"`` -- direct per-thread register gather from ``sX``, no round-trip. Output
              is **bit-identical** to ``"SMEM"`` and ~7-15% faster at N<=256 (less L1TEX traffic),
              tying at N>=512 (DRAM-bound, and the gather adds bank conflicts).
            * ``"ldmatrix"`` -- refused; a documented blocker.
            * ``"auto"`` -- ``"shuffle"`` for N<=256 else ``"SMEM"``, the measured per-N optimum.
        select: How the ``do_transpose`` perf knob is resolved.

            * ``"heuristic"`` (default) -- :func:`_transpose_heuristic_config`, i.e. the ``"auto"``
              rule as an EXPLICIT config, so the kernel skips the per-call ``auto`` branch. An
              explicit ``do_transpose`` overrides it.
            * ``"autotune"`` -- sweep ``{"shuffle", "SMEM"}`` for THIS shape.
            * ``"default"`` -- run exactly what was passed (``"auto"`` still resolves per N).
        _config: A frozen config from :func:`layernorm_transpose_freeze`; skips knob resolution.

    Returns:
        ``(M, N)`` **row-major** (N contiguous) -- i.e. LayerNorm-then-transpose in one pass.

    Raises:
        ValueError: If ``x`` is not 2-D, not LayoutLeft, or of an unsupported dtype; if
            ``return_rstd`` / ``return_mean`` is asked for; if ``weight`` / ``bias`` is missing, not
            fp32, or not ``(N,)``; if N is below the swizzle-atom floor (``N % 16`` at 16-bit,
            ``N % 32`` at fp32 -- below it the CUTLASS DSL backend corrupts the heap rather than
            raising); if ``N > 1024`` (the whole row is staged in SMEM); or if M is not 16-byte
            aligned (M is the contiguous dim of the ``(N, M)`` TMA view).
        NotImplementedError: If ``do_transpose="ldmatrix"``.
    """
    # --- knob resolution: freeze / autotune / heuristic. The body below is one fixed path. ---
    if select not in ("heuristic", "autotune", "default"):
        raise ValueError(
            f"select={select!r} is not one of 'heuristic' | 'autotune' | 'default'. An unrecognized "
            f"value would otherwise fall through to the fixed path, i.e. silently mean 'default'."
        )
    if _config is not None:  # frozen perf-config: run the fixed path with the resolved knobs
        return _transpose_fwd(x, weight, bias, eps=eps, select="default", **_config.all_kwargs())
    if select == "autotune":  # sweep the perf knobs for THIS shape (keyed on shape/dtype)
        return _layernorm_transpose_fwd_tuned(x, weight, bias, eps=eps)

    check_tensor("x", x, expect_shape=(None, None))
    if x.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError(
            f"x must be float16, bfloat16 or float32 for the LayerNorm kernel; got {x.dtype}."
        )
    if return_rstd or return_mean:
        raise ValueError(
            "return_rstd / return_mean are not available with transpose=True: the transposing "
            "kernel writes only the normalized output. Run the row-major variant (transpose=False) "
            "if the row statistics are needed."
        )
    M, N = x.shape
    # Size->knob heuristic: resolve do_transpose from N, FORMALIZING the `auto` rule as an explicit
    # config so the kernel skips its per-call `auto` branch. An explicit do_transpose overrides it.
    if select == "heuristic" and do_transpose == "auto":
        do_transpose = _transpose_heuristic_config(N, device=x.device).get("do_transpose")
    if do_transpose == "auto":
        do_transpose = "shuffle" if N <= 256 else "SMEM"
    if do_transpose == "ldmatrix":
        raise NotImplementedError(
            "do_transpose='ldmatrix' is not implemented: the LDSM.x4.trans fragment cannot reuse "
            "row_reduce (see LayerNormTransposeSm90._ldmatrix_fragment). Use 'shuffle', which "
            "removes the same SMEM round-trip and is bit-identical to 'SMEM'."
        )
    if do_transpose not in _DO_TRANSPOSE_ENUM:
        raise ValueError(
            f"do_transpose={do_transpose!r} is not one of {sorted(_DO_TRANSPOSE_ENUM) + ['auto']}."
        )
    if x.stride() != (1, M):
        raise ValueError(
            f"x must be LayoutLeft (stride (1, M)) for transpose=True; got stride {x.stride()}. A "
            f"row-major x would be READ as if it were LayoutLeft -- a wrong answer, not an error."
        )
    # The N floor is the SMEM transpose swizzle atom (K_SW32 = 16 elements at 2 bytes, K_SW64 = 32
    # at 4 bytes). An N below it (e.g. bf16 N%8-but-not-%16, or fp32 N=48/80) does not raise inside
    # the CUTLASS DSL -- it corrupts the heap during MLIR lowering -- so reject it cleanly here.
    _swz = 8 * x.element_size()  # 16 for bf16/fp16, 32 for fp32
    if N % _swz != 0:
        raise ValueError(
            f"N ({N}) must be a multiple of {_swz} (the transpose swizzle atom for {x.dtype}). "
            f"Below the floor the CUTLASS DSL backend corrupts its heap during lowering rather "
            f"than raising, which is why this is refused here."
        )
    if N > 1024:
        raise ValueError(
            f"N ({N}) > 1024; the transposing load stages a whole row in SMEM, so there is no "
            f"config for a wider feature axis. Use transpose=False."
        )
    # M is the contiguous mode of the (N, M) TMA view, so the 16-byte TMA row floor lands on M.
    if M % (16 // x.element_size()) != 0:
        raise ValueError(
            f"M ({M}) must be 16-byte aligned for the (N, M) TMA view: M is that view's contiguous "
            f"mode, so it needs M % {16 // x.element_size()} == 0 at {x.dtype}."
        )
    check_tensor("weight", weight, expect_dtype=torch.float32, expect_shape=(N,))
    check_tensor("bias", bias, expect_dtype=torch.float32, expect_shape=(N,))
    if weight is None or bias is None:
        raise ValueError(
            "weight and bias are both REQUIRED with transpose=True: the kernel always applies the "
            "affine, so an absent one would be read from unallocated memory rather than skipped."
        )

    dtype = torch2cute_dtype_map[x.dtype]
    d_blk_m, d_subt, d_tpr = _transpose_default_config(N, dtype)
    blk_m = blk_m or d_blk_m
    subt = subt or d_subt
    threads_per_row = threads_per_row or d_tpr

    # The memory of x IS an (N, M) row-major array (M contiguous) -- view it as such for the TMA.
    xt = x.mT  # (N, M), stride (M, 1) -> row-major, M contiguous
    out = torch.empty(M, N, device=x.device, dtype=x.dtype)  # (M, N) row-major
    _layernorm_transpose_fwd_op(
        xt, weight, bias, out, eps, blk_m, subt, threads_per_row, _DO_TRANSPOSE_ENUM[do_transpose]
    )
    return out


# ───────────────────────── transposing variant: autotune-free size heuristic ─────────────────────
# A pure size->knob formula for the one big perf lever (`do_transpose`), used on the DEFAULT path so
# there is no timing and no per-call autotuner key-build. It FORMALIZES the long-shipped
# `do_transpose="auto"` rule as an explicit config:
#   * N <= 256 : "shuffle" -- the direct per-thread register gather (no sT round-trip) cuts L1TEX
#                traffic; 0.89-0.99x SMEM, the win growing with M (the round-trip is pure overhead
#                at thin N).
#   * N >= 512 : "SMEM"    -- the kernel is DRAM-bound there and shuffle's transposed gather adds a
#                bank-conflict surge, so SMEM is faster-or-tied (shuffle 1.00-1.02x SMEM).
# The built-in autotune sweep reproduces this pick in 14/16 measured cells; the 2 mismatches are
# sub-0.5% launch-bound ties at the smallest M, i.e. the heuristic is correct wherever the gap is
# real (>=4%). `do_transpose` is purely a transpose-strategy choice with NO size constraint --
# "SMEM" is valid for EVERY supported N (the universal fallback) and "shuffle" has no hard N upper
# bound (merely suboptimal at N>=512) -- so there is no validity gate here, only a perf threshold.
#
#: The crossover, per arch. Bisected on ``heuristic_arch.TUNED_ARCH``; a non-tuned arch warns once
#: and falls back to that set. Both transpose paths are bit-identical in OUTPUT, so there is no
#: arch-independent validity layer underneath this -- it is a pure perf pick.
_TRANSPOSE_PERF = {
    # "shuffle" (register-gather) wins thin-N (N <= SHUFFLE_MAX_N); "SMEM" (DRAM-bound) ties/wins up.
    "H200_SXM5": {"SHUFFLE_MAX_N": 256},
}


def _transpose_heuristic_config(N: int, device=None) -> AutotuneConfig:
    """Resolve ``do_transpose`` from the feature width N alone -- no timing, no memory query.

    ``"shuffle"`` if ``N <= SHUFFLE_MAX_N`` (the register gather wins thin-N), else ``"SMEM"``
    (DRAM-bound; SMEM ties or wins at N>=512). Both paths are bit-identical in output, so this is
    purely a perf pick and never a validity gate: ``"SMEM"`` is valid for every supported N and is
    the universal fallback. The hard floors (swizzle atom, M alignment, N<=1024) are enforced in
    :func:`_transpose_fwd` and apply to both values identically.

    Returned as an :class:`AutotuneConfig` so the entry's ``_config`` / ``all_kwargs()`` plumbing
    and the freeze fast-path consume it uniformly.

    Args:
        N: Feature extent. Any positive int; the threshold comparison is total, so an N the front
            door would reject still yields a config rather than raising here.
        device: Selects the arch for the threshold lookup. None means "do not query a device" --
            it uses ``TUNED_ARCH`` and warns about nothing, which is what a host-side unit test
            wants. A real device on a non-tuned arch warns ONCE and falls back to the tuned set.

    Returns:
        An :class:`AutotuneConfig` carrying exactly ``do_transpose``.
    """
    arch = heuristic_arch(device) if device is not None else TUNED_ARCH
    if arch != TUNED_ARCH:
        warn_arch_suboptimal_once(arch, "layernorm_fwd(transpose=True)")
    perf = _TRANSPOSE_PERF.get(arch, _TRANSPOSE_PERF[TUNED_ARCH])
    return AutotuneConfig(do_transpose="shuffle" if N <= perf["SHUFFLE_MAX_N"] else "SMEM")


# ───────────────────────── transposing variant: built-in autotune + freeze ──────────────────────
#: The candidate pool. ONLY ``do_transpose`` is swept: ``blk_m`` / ``subt`` / ``threads_per_row``
#: are left None so the kernel auto-picks them from valid divisors of N, and sweeping them would
#: need shape-aware candidates (a fixed list would be invalid at most N). Each shape/dtype tunes
#: independently -- the autotuner appends every tensor argument's shape/stride/dtype to the cache
#: key automatically, so ``key=()`` suffices.
_TRANSPOSE_TUNING_CONFIGS = (
    AutotuneConfig(do_transpose="shuffle"),
    AutotuneConfig(do_transpose="SMEM"),
)


def _transpose_config_is_valid(config, request) -> bool:
    """Whether one ``do_transpose`` candidate CAN RUN for this request. Rejection only.

    Purpose
        Both candidates are valid for every shape the front door admits, so this rejects nothing in
        practice. It exists to state that -- and to keep the one global cap (whole-row SMEM
        staging) expressed where the pool is filtered rather than only in the wrapper.

    Semantics
        A PURE function of the config and the request, as the tuner requires: an impure one makes
        ranks measure different pools.

    Args:
        config: The candidate, carrying ``do_transpose``.
        request: The bound call arguments by name; ``x`` gives the shape.

    Returns:
        False for ``"shuffle"`` above the whole-row SMEM cap (N > 1024), which the wrapper also
        refuses outright. True otherwise -- notably including ``"shuffle"`` at N>=512, which is
        SUBOPTIMAL but not invalid, and is left in so the sweep can confirm the heuristic.
    """
    _, N = request["x"].shape
    return not (config.get("do_transpose") == "shuffle" and N > 1024)


@autotune(
    configs=_TRANSPOSE_TUNING_CONFIGS,
    key=(),
    validity=_transpose_config_is_valid,
)
def _layernorm_transpose_fwd_tuned(
    x, weight, bias, eps=1e-6, blk_m=None, subt=None, threads_per_row=None, do_transpose="auto"
):
    """Measure ``do_transpose`` for THIS shape, then dispatch. The ``select="autotune"`` target.

    Args:
        x: ``(M, N)`` LayoutLeft input.
        weight: ``(N,)`` fp32 affine scale.
        bias: ``(N,)`` fp32 affine shift.
        eps: Variance epsilon.
        blk_m: Passed through; left None so the kernel auto-picks.
        subt: Passed through; left None so the kernel auto-picks.
        threads_per_row: Passed through; left None so the kernel auto-picks.
        do_transpose: **Injected by the tuner** from the winning config.

    Returns:
        The ``(M, N)`` row-major output, from the fixed path (``select="default"``) so the tuner's
        pick is not re-resolved by the heuristic.
    """
    return _transpose_fwd(
        x,
        weight,
        bias,
        eps=eps,
        blk_m=blk_m,
        subt=subt,
        threads_per_row=threads_per_row,
        do_transpose=do_transpose,
        select="default",
    )


def layernorm_transpose_freeze(x: Tensor, weight: Tensor, bias: Tensor, eps: float = 1e-6):
    """Resolve the winning ``do_transpose`` for THIS shape/dtype ONCE and bind it.

    Purpose
        ``select="autotune"`` costs a cache-key build on every call. Freezing pays the sweep once
        and returns a callable that skips it.

    Args:
        x: ``(M, N)`` LayoutLeft input -- the sweep is keyed on its shape/stride/dtype, so the
            returned callable is only valid for tensors matching them.
        weight: ``(N,)`` fp32 affine scale.
        bias: ``(N,)`` fp32 affine shift.
        eps: Variance epsilon. Becomes the returned callable's default.

    Returns:
        A callable ``(x, weight, bias, eps=eps) -> (M, N) row-major``, with ``.config`` exposing the
        :class:`AutotuneConfig` that won. It does NOT re-check the shape against the one tuned on;
        calling it with a different shape runs a config chosen for another.
    """
    layernorm_fwd(x, weight, bias, eps=eps, transpose=True, select="autotune")
    # `.autotuner.best_config`, not `.best_config`: this tree's `@autotune` puts the winner on the
    # Autotuner instance and hangs it off the wrapper as `.autotuner`. Reading the wrapper attribute
    # directly gives an AttributeError, which is at least loud -- but the trap next door is silent,
    # so it is worth naming: `autotune.freeze(config)` returns the knobs as kwargs for a NON-tuned
    # entry, and passing those to the tuned wrapper is refused. `_config=` is the pin.
    frozen = _layernorm_transpose_fwd_tuned.autotuner.best_config

    def frozen_call(x, weight, bias, eps=eps):
        return layernorm_fwd(x, weight, bias, eps=eps, transpose=True, _config=frozen)

    frozen_call.config = frozen
    return frozen_call
