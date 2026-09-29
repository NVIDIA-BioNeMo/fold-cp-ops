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

# Based on the cute-dsl example:
# https://github.com/NVIDIA/cutlass/blob/main/examples/python/CuTeDSL/hopper/dense_gemm.py
# This file has been modified by NVIDIA CORPORATION & AFFILIATES.

"""The token-pair einsum, its LayerNorm, the output projection and the output gate, in ONE call.

Ops 7, 8, 9 and 12 of the reference TriMul numbering (``docs/cp1_kernel_bringback_complete.md`` §2)::

     7  tri  = einsum('bikd,bjkd->bijd', a, b)          outgoing   (B, N, N, D)
        tri  = einsum('bkid,bkjd->bijd', a, b)          incoming
     8  trin = LayerNorm(tri; norm_w, norm_b)  over D   (B, N, N, D)
     9  proj = trin @ proj_w^T + proj_b                 (B, N, N, D)
    12  out  = proj * gate3                             (B, N, N, D)   gate3 optional

Three kernels, launched back to back by :class:`GemmLayerNormGemmSm90`:

1. :class:`TokenPairGemmStatsSm90` — a CTA owns one ``(BLK_i, BLK_j)`` token-pair tile and loops
   ``BLK_D`` features, running one full WGMMA per feature over the length-``N`` token contraction.
   Each accumulator is staged into ``s_tri[:, :, dl]`` and the whole ``(BLK_i, BLK_j, BLK_D)`` block
   is written **d-contiguous (stride 1)**. That store order is the entire reason this kernel exists:
   it is the layout op 8 wants, so the transposing pass the alternative chain needs never happens.
   The same epilogue accumulates each token's partial ``sum(x)`` / ``sum(x^2)`` over this CTA's
   features into a ``(D/BLK_D, B*N^2, 2)`` workspace.
2. :class:`RowStatsReduceSm90` — one thread per token, summing those partials to ``(mean, rstd)``.
3. :class:`FeatureGemmPrologLnSm90` — contracts the FEATURE axis with a software-pipelined
   normalize-on-load prologue, then multiplies by ``gate3``. The prologue IS ``prolog_ln``.

:class:`TokenPairGemmStatsContPipeSm90` is a scheduling variant of kernel 1 (one continuous
``(d, k)`` pipeline instead of a per-feature one) selected by the ``cont_pipe`` perf knob; it
computes exactly the same numbers.

Nothing here subclasses ``GemmSm90`` and nothing here shares the epilogue machinery — this is a
self-contained chain, which is why it is the whole file.

Naming, versus the upstream this was ported from (``main:fold_cp_ops/kernels/gemm_ln_gemm.py``):
``TriGemm1StatsSm90`` -> ``TokenPairGemmStatsSm90``, ``TriGemm1StatsContPipeSm90`` ->
``TokenPairGemmStatsContPipeSm90``, ``FinalizeStatsSm90`` -> ``RowStatsReduceSm90``,
``TriGemm2NormPipeSm90`` -> ``FeatureGemmPrologLnSm90``, ``out1`` -> ``tri``, ``out2`` -> ``out``,
``W2``/``b2`` -> ``proj_w``/``proj_b``, ``blk_k2`` -> ``blk_kf``, ``k1_cont_pipe`` -> ``cont_pipe``.

Hardware: SM90 (H100/H200) only.
"""

from typing import Optional, Tuple, Type

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
from cutlass import Float32, Int32, const_expr
from cutlass.cute.nvgpu import cpasync, warpgroup
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait
import cutlass.utils.hopper_helpers as sm90_helpers
from cutlass.utils import LayoutEnum

import torch
import torch.nn.functional as F
from torch import Tensor

import fold_cp_ops._internal.sm90_utils as fold_cp_ops_sm90
from fold_cp_ops._internal.autotune import AxisSpace, TuneAxis, autotune
from fold_cp_ops._internal.compile_time.compile_utils import make_fake_tensor as fake_tensor
from fold_cp_ops._internal.compile_time.copy_descriptors import tiled_copy_2d
from fold_cp_ops._internal.compile_time.cute_dsl_utils import torch2cute_dtype_map
from fold_cp_ops._internal.heuristic_arch import (
    TUNED_ARCH,
    heuristic_arch,
    warn_arch_suboptimal_once,
)

# TVM-FFI launch: compile each kernel with fake operands (symbolic strides matching each runtime
# view's contiguity) plus an env-stream placeholder, then call the compiled callable with the real
# torch tensors and NO explicit stream -- tvm-ffi injects the ambient CUDA stream.  This mirrors the
# stock `gemm` launch path so the fused kernels carry the same lean per-launch host overhead as the
# unfused baselines, which is what makes a comparison against them fair.
_TVM_FFI = dict(options="--enable-tvm-ffi")


def _fake_stream():
    """A placeholder stream for compilation, standing in for the ambient one at call time.

    Purpose:
        ``cute.compile`` needs a stream argument, but the compiled callable is invoked WITHOUT one
        so that tvm-ffi injects the caller's ambient CUDA stream. This supplies the compile-time
        stand-in that makes that possible.

    Returns:
        A fake ``cuda.CUstream`` bound to the tvm-ffi environment stream. Passing a REAL stream here
        instead would bake that stream into the artifact, so every later call would run on it no
        matter which stream the caller is on -- silently serialising against the caller's work.
    """
    return cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)


# In-process compile cache.  `cute.compile` does NOT cache across calls, so without this every call
# recompiles all three kernels (~4 s) and a benchmark loop would time compilation, not the kernel.
# A compiled callable is bound only to tensor METADATA (shape / stride / dtype) plus the tile
# parameters, so it is safe to reuse across calls with fresh tensors of the same layout.  Keyed by
# (kernel id, dtype, shapes, tiles) -- everything that changes the generated code.
_COMPILE_CACHE = {}


class TokenPairGemmStatsSm90:
    """Op 7 (the token-pair einsum) plus op 8's per-token partial statistics, feature-grouped.

    Purpose:
        Produce ``tri[b, i, j, d]`` with **d contiguous** -- the layout the LayerNorm over the
        feature axis wants -- while accumulating, for free, each token's partial ``sum(x)`` and
        ``sum(x^2)`` over the features this CTA owns.

    Functionality and semantics:
        A CTA owns ONE ``(blk_i, blk_j)`` token-pair tile and loops over ``blk_d`` consecutive
        features. For each local feature it runs a full WGMMA over the length-``N`` token
        contraction::

            A_d = a[d, i_tile, :]   (M=blk_i, K=N)
            B_d = b[d, j_tile, :]   (N=blk_j, K=N)
            acc = A_d @ B_d^T       (blk_i, blk_j) fp32

        Each accumulator is cast to ``dtype`` and staged into ``s_tri[:, :, dl]``; after the feature
        loop the whole ``(blk_i, blk_j, blk_d)`` block is written to global memory with ``d`` at
        stride 1. Grid is ``(ceil(N/blk_i), ceil(N/blk_j), B*(D/blk_d))``; one warpgroup, warp 0 the
        TMA producer and all four warps the WGMMA consumer, over a ``ab_stage``-deep k-pipeline,
        with fp32 accumulation.

        The statistics need **no atomics and no barrier**: each accumulator element maps to a FIXED
        ``(i, j)`` across the whole feature loop (the same ``partition_C`` thread-value layout every
        iteration), so the per-token partial is a plain elementwise accumulation into two register
        fragments shaped like ``acc``; each ``(feature chunk, token)`` slot is then written by
        exactly one CTA and each ``(i, j)`` by exactly one thread.

        ``N`` is PREDICATED throughout -- ceil-div grid, TMA out-of-bounds-masked loads, and a
        compile-time-gated bounds check on the two stores -- so a partial M/N/K tile is zero-filled
        by the TMA and its invalid rows and columns are simply not stored. When ``N`` divides both
        tile extents the predicate is pruned at compile time and the store costs zero runtime ALU;
        both arms are kept deliberately, because collapsing them onto the predicated one is a
        measurable regression on the common path.

    Args:
        dtype: Element type of ``a`` / ``b`` and of the staged output. Must be a 16-bit type the
            SM90 WGMMA atom accepts (bfloat16 / float16); a 32-bit one has no matching atom and
            fails at MMA construction.
        N: The token-axis extent (the triangle side). Must satisfy ``N % 8 == 0`` -- the 16-byte TMA
            alignment floor for a 16-bit dtype. It need NOT be a multiple of any tile: a non-multiple
            takes the predicated arm. A misaligned ``N`` faults inside the TMA descriptor build.
        D: The feature-axis extent. Must satisfy ``D % blk_d == 0``, so a feature chunk never
            straddles a batch boundary; otherwise the grid-z decomposition maps a chunk onto two
            batches and reads the wrong operand slice.
        blk_i: Token-tile extent along ``i``, i.e. the WGMMA M. Must be a multiple of 64, which is
            what fixes the warpgroup count; anything else builds an MMA whose M the hardware cannot
            issue.
        blk_j: Token-tile extent along ``j``, i.e. the WGMMA N. Must be a multiple of 8.
        blk_d: Features per CTA. Sets the shared-memory staging buffer at
            ``blk_i * blk_j * blk_d`` elements, which is what caps occupancy -- raising it trades
            occupancy for a longer coalesced store.
        blk_k: Contraction tile over the token axis. Must be a multiple of 16. A partial last tile
            is fine (the TMA zero-fills it and a zero-padded k contributes nothing to the sum).
        B: Batch. The operands are viewed ``(M=N, K=N, L=B*D)`` by flattening ``(B, D)`` onto the
            L axis, so ``B == 1`` is byte-identical to an unbatched build.

    Raises:
        AssertionError: On any of the tile / extent constraints above. These are the kernel's last
            line of defence; the public front door raises a ``ValueError`` before reaching here.
    """

    def __init__(
        self,
        dtype: Type[cutlass.Numeric],
        N: int,
        D: int,
        blk_i: int,
        blk_j: int,
        blk_d: int,
        blk_k: int,
        B: int = 1,
    ):
        self.dtype = dtype
        self.acc_dtype = Float32
        self.N = N
        self.D = D
        # Batch B>1: the token axis is M = B·N²; a, b are (B,D,N,N) -> the operands are viewed
        # (M=N, K=N, L=B·D) (flatten (B,D)->L, stride N²), so grid-z = B·(D/blk_d) and a CTA derives
        # b = zgroup//(D/blk_d), feature-chunk c = zgroup%(D/blk_d).  B==1 is byte-identical (b≡0,
        # c≡zgroup, L-index ≡ the plain feature index).  tri is (B,N,N,D); the statistics workspace
        # token axis is B·N².
        self.B = B
        self.blk_i = blk_i
        self.blk_j = blk_j
        self.blk_d = blk_d
        self.blk_k = blk_k
        assert N % 8 == 0, "N must be 16-byte aligned (N % 8 == 0 for bf16/fp16)"
        assert D % blk_d == 0, "D must be a multiple of blk_d (the feature group)"
        assert blk_i % 64 == 0, "WGMMA M must be a multiple of 64"
        assert blk_j % 8 == 0 and blk_k % 16 == 0
        self.mma_warp_groups = blk_i // 64
        self.num_threads = self.mma_warp_groups * 128
        self.num_threads_per_warp_group = 128
        self.ab_stage = 3
        self.buffer_align_bytes = 1024

    @cute.jit
    def __call__(
        self,
        mA: cute.Tensor,  # (M=N, K=N, L=B·D)  a[b,d,i,k] viewed i=M, k=K, (b,d)=L
        mB: cute.Tensor,  # (N=N, K=N, L=B·D)  b[b,d,j,k] viewed j=N, k=K, (b,d)=L
        mTri: cute.Tensor,  # (B, N, N, D) with D contiguous
        mStatsPartial: cute.Tensor,  # (D/blk_d, B*N*N, 2) fp32 partial (sum x, sum x^2) per token
        stream: cuda.CUstream,
    ):
        """Build the MMA, the swizzled SMEM layouts and the TMA atoms, then launch the kernel.

        Semantics:
            Everything computed here is layout / type algebra that the compiler folds away; the only
            emitted instruction is the launch. The operand major modes are DERIVED from the tensors
            (``LayoutEnum.from_tensor``), which is what lets the incoming direction be a transposed
            strided view of the same memory rather than a second kernel.

        Args:
            mA: The ``i``-side operand, ``(N, N, B*D)``. Its stride-1 axis decides the major mode:
                k-major for the outgoing direction, m-major for the incoming one. A non-16-byte
                aligned leading extent is rejected by the TMA descriptor build.
            mB: The ``j``-side operand, same shape and major mode as ``mA``. Mixing major modes
                between the two compiles but contracts the wrong axis.
            mTri: The einsum output, ``(B, N, N, D)``, ``D`` contiguous. A different innermost axis
                silently un-coalesces the block store and breaks the layout op 8 depends on.
            mStatsPartial: fp32 workspace, ``(D/blk_d, B*N*N, 2)``. Written, never read, so it may
                be uninitialised -- but it must be exactly this shape, since the row index is the
                CTA's feature chunk and every chunk writes every token.
            stream: The CUDA stream. Pass :func:`_fake_stream` at compile time.

        Returns:
            None. Its effect is ``mTri`` and ``mStatsPartial``.
        """
        a_layout = LayoutEnum.from_tensor(mA)
        b_layout = LayoutEnum.from_tensor(mB)
        blk_i, blk_j, blk_d, blk_k = self.blk_i, self.blk_j, self.blk_d, self.blk_k

        tiled_mma = sm90_helpers.make_trivial_tiled_mma(
            self.dtype,
            self.dtype,
            a_layout.sm90_mma_major_mode(),
            b_layout.sm90_mma_major_mode(),
            self.acc_dtype,
            (self.mma_warp_groups, 1, 1),
            (64, blk_j),
        )

        # Staged SMEM layouts for A (blk_i, blk_k) and B (blk_j, blk_k).
        a_smem_layout = fold_cp_ops_sm90.make_smem_layout(
            self.dtype, a_layout, (blk_i, blk_k), self.ab_stage
        )
        b_smem_layout = fold_cp_ops_sm90.make_smem_layout(
            self.dtype, b_layout, (blk_j, blk_k), self.ab_stage
        )

        # TMA atoms over the (M/N, K, L) tensors with a (blk_*, blk_k) box (no multicast).
        tma_atom_a, tma_tensor_a = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(),
            mA,
            cute.slice_(a_smem_layout, (None, None, 0)),
            (blk_i, blk_k),
            num_multicast=1,
        )
        tma_atom_b, tma_tensor_b = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(),
            mB,
            cute.slice_(b_smem_layout, (None, None, 0)),
            (blk_j, blk_k),
            num_multicast=1,
        )

        self.tma_copy_bytes = cute.size_in_bytes(
            self.dtype, cute.slice_(a_smem_layout, (None, None, 0))
        ) + cute.size_in_bytes(self.dtype, cute.slice_(b_smem_layout, (None, None, 0)))

        @cute.struct
        class SharedStorage:
            ab_pipeline_array_ptr: cute.struct.MemRange[cutlass.Int64, self.ab_stage * 2]
            sTri: cute.struct.Align[
                cute.struct.MemRange[self.dtype, blk_i * blk_j * blk_d], self.buffer_align_bytes
            ]
            sA: cute.struct.Align[
                cute.struct.MemRange[self.dtype, cute.cosize(a_smem_layout)],
                self.buffer_align_bytes,
            ]
            sB: cute.struct.Align[
                cute.struct.MemRange[self.dtype, cute.cosize(b_smem_layout)],
                self.buffer_align_bytes,
            ]

        self.shared_storage = SharedStorage

        grid = (
            (self.N + blk_i - 1) // blk_i,  # ceil_div: launch the partial M/N tile (predicated)
            (self.N + blk_j - 1) // blk_j,
            self.B * (self.D // blk_d),  # grid-z = B·(D/blk_d) feature chunks
        )
        self.kernel(
            tiled_mma,
            tma_atom_a,
            tma_tensor_a,
            tma_atom_b,
            tma_tensor_b,
            mTri,
            mStatsPartial,
            a_smem_layout,
            b_smem_layout,
        ).launch(
            grid=grid,
            block=[self.num_threads, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        tiled_mma: cute.TiledMma,
        tma_atom_a: cute.CopyAtom,
        mA: cute.Tensor,  # (M=N, K=N, L=B·D)
        tma_atom_b: cute.CopyAtom,
        mB: cute.Tensor,  # (N=N, K=N, L=B·D)
        mTri: cute.Tensor,  # (B, N, N, D)
        mStatsPartial: cute.Tensor,  # (D/blk_d, B*N*N, 2) fp32
        a_smem_layout: cute.ComposedLayout,
        b_smem_layout: cute.ComposedLayout,
    ):
        """One CTA: ``blk_d`` WGMMA einsums over one token-pair tile, staged and stored d-contiguous.

        Semantics:
            The feature loop is a ``range_constexpr``, so ``blk_d`` is baked and each iteration gets
            its own TMA views. Each feature is a SELF-CONTAINED k-pipeline: prefetch
            ``min(ab_stage, k_tile_cnt)`` tiles, run the steady state, drain. The pipeline STATES are
            monotonic across features (index = count % ab_stage), which is why the producer tracks a
            per-feature base rather than resetting its counter.

        Args:
            tiled_mma: The warpgroup MMA. Its accumulate flag is reset per feature.
            tma_atom_a, tma_atom_b: The G2S bulk-tensor atoms.
            mA, mB: The TMA-mapped operand views returned by ``make_tiled_tma_atom`` -- NOT the raw
                tensors; passing the raw ones makes the ``tma_partition`` coordinates wrong.
            mTri: The ``(B, N, N, D)`` output.
            mStatsPartial: The ``(D/blk_d, B*N*N, 2)`` fp32 statistics workspace.
            a_smem_layout, b_smem_layout: The swizzled staged layouts, used to view the SMEM
                allocation. Must be the same objects the TMA atoms were built from.

        Returns:
            None.
        """
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        tidx, _, _ = cute.arch.thread_idx()
        bi, bj, bd = cute.arch.block_idx()
        blk_i, blk_j, blk_d, blk_k = self.blk_i, self.blk_j, self.blk_d, self.blk_k
        D = const_expr(self.D)
        # Decompose the z-group into (batch b, feature chunk c).  n_dchunks = D/blk_d is PER-BATCH;
        # a blk_d chunk never crosses a batch boundary (D%blk_d==0).  B==1 -> b=0, c=bd.
        n_dchunks = const_expr(self.D // blk_d)
        b_idx = bd // n_dchunks
        c_chunk = bd % n_dchunks
        Boff = b_idx * D  # L-axis (B·D) offset of this batch's D features in mA/mB
        tok_boff = b_idx * (self.N * self.N)  # token-axis (B·N²) offset of this batch

        if warp_idx == 0:
            cpasync.prefetch_descriptor(tma_atom_a)
            cpasync.prefetch_descriptor(tma_atom_b)

        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)

        # AB pipeline (single TMA producer thread; consumer = all MMA warps).
        producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        num_warps = const_expr(self.num_threads // 32)
        consumer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, num_warps)
        ab_pipeline = pipeline.PipelineTmaAsync.create(
            barrier_storage=storage.ab_pipeline_array_ptr.data_ptr(),
            num_stages=self.ab_stage,
            producer_group=producer_group,
            consumer_group=consumer_group,
            tx_count=self.tma_copy_bytes,
            cta_layout_vmnk=cute.make_layout((1, 1, 1, 1)),
            defer_sync=True,
        )
        pipeline_init_arrive(cluster_shape_mn=(1, 1), is_relaxed=True)

        sA = storage.sA.get_tensor(a_smem_layout.outer, swizzle=a_smem_layout.inner)
        sB = storage.sB.get_tensor(b_smem_layout.outer, swizzle=b_smem_layout.inner)
        # tri staging: (blk_i, blk_j, blk_d), d contiguous (order=(2,1,0) -> stride d=1).
        s_tri = storage.sTri.get_tensor(
            cute.make_ordered_layout((blk_i, blk_j, blk_d), order=(2, 1, 0))
        )

        # MMA partitioning.  A/B SMEM fragments are partitioned by the warpgroup-base thread (WGMMA
        # is a warpgroup-collective read of SMEM); the per-thread C fragment and the SMEM store use
        # the real tidx slice so each thread writes ITS accumulator elements to the matching slot.
        warp_group_idx = cute.arch.make_warp_uniform(tidx // self.num_threads_per_warp_group)
        wg_thread_layout = cute.make_layout(
            self.mma_warp_groups, stride=self.num_threads_per_warp_group
        )
        thr_mma = tiled_mma.get_slice(wg_thread_layout(warp_group_idx))
        thr_mma_c = tiled_mma.get_slice(tidx)
        tCsA = thr_mma.partition_A(sA)
        tCsB = thr_mma.partition_B(sB)
        tCrA = tiled_mma.make_fragment_A(tCsA)
        tCrB = tiled_mma.make_fragment_B(tCsB)
        acc = cute.make_rmem_tensor(thr_mma_c.partition_shape_C((blk_i, blk_j)), self.acc_dtype)
        num_k_blocks = const_expr(cute.size(tCrA, mode=[2]))

        # ---- op 8's partial row statistics (per element, fp32) ----
        # Each accumulator element maps to a FIXED (i,j) across the feature loop (same partition_C
        # thread-value layout every iteration), so this CTA's partial sum(x)/sum(x²) per token is
        # just an elementwise accumulation of acc into two register fragments shaped like acc.  No
        # atomics and no coordinate reconstruction: element e always holds tri[*, i_e, j_e].
        rsum = cute.make_rmem_tensor(thr_mma_c.partition_shape_C((blk_i, blk_j)), self.acc_dtype)
        rsqsum = cute.make_rmem_tensor(thr_mma_c.partition_shape_C((blk_i, blk_j)), self.acc_dtype)
        acc_size = const_expr(cute.size(acc))
        for e in cutlass.range_constexpr(acc_size):
            rsum[e] = self.acc_dtype(0.0)
            rsqsum[e] = self.acc_dtype(0.0)

        # ceil_div: the last contraction tile may be partial (N % blk_k != 0); the TMA zero-fills
        # the out-of-bounds tail, and a zero-padded k contributes nothing to the WGMMA sum.
        k_tile_cnt = const_expr((self.N + blk_k - 1) // blk_k)

        pipeline_init_wait(cluster_shape_mn=(1, 1))

        # ---- outer loop over the blk_d features owned by this CTA ----
        ab_producer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.ab_stage
        )
        ab_read_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.ab_stage
        )
        ab_release_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.ab_stage
        )

        for dl in cutlass.range_constexpr(blk_d):
            d = Boff + c_chunk * blk_d + dl  # L-index into mA/mB (B·D axis)
            # Per-feature (M=blk_i, K, RestK) and (N=blk_j, K, RestK) k-tile views.
            mA_mk = mA[None, None, d]
            mB_nk = mB[None, None, d]
            gA = cute.local_tile(mA_mk, (blk_i, blk_k), (bi, None))  # (blk_i, blk_k, RestK)
            gB = cute.local_tile(mB_nk, (blk_j, blk_k), (bj, None))  # (blk_j, blk_k, RestK)

            tAsA, tAgA = cpasync.tma_partition(
                tma_atom_a,
                0,
                cute.make_layout(1),
                cute.group_modes(sA, 0, 2),
                cute.group_modes(gA, 0, 2),
            )
            tBsB, tBgB = cpasync.tma_partition(
                tma_atom_b,
                0,
                cute.make_layout(1),
                cute.group_modes(sB, 0, 2),
                cute.group_modes(gB, 0, 2),
            )

            # ---- per-feature k-pipeline: producer (warp 0) loads, consumer (all warps) WGMMAs ----
            # Pipeline states are monotonic across features (index = count % ab_stage); the
            # producer's local k-tile index is (count - base), so we track a per-feature base.
            prod_base = const_expr(dl * k_tile_cnt)
            read_base = const_expr(dl * k_tile_cnt)

            if warp_idx == 0:
                prefetch_cnt = const_expr(min(self.ab_stage, k_tile_cnt))
                for kt in cutlass.range_constexpr(prefetch_cnt):
                    ab_pipeline.producer_acquire(ab_producer_state)
                    bar = ab_pipeline.producer_get_barrier(ab_producer_state)
                    cute.copy(
                        tma_atom_a,
                        tAgA[(None, kt)],
                        tAsA[(None, ab_producer_state.index)],
                        tma_bar_ptr=bar,
                    )
                    cute.copy(
                        tma_atom_b,
                        tBgB[(None, kt)],
                        tBsB[(None, ab_producer_state.index)],
                        tma_bar_ptr=bar,
                    )
                    ab_pipeline.producer_commit(ab_producer_state)
                    ab_producer_state.advance()

            tiled_mma.set(warpgroup.Field.ACCUMULATE, False)
            peek = cutlass.Boolean(1)
            if 0 < k_tile_cnt:
                peek = ab_pipeline.consumer_try_wait(ab_read_state)
            for k_tile in cutlass.range(k_tile_cnt, unroll=1):
                ab_pipeline.consumer_wait(ab_read_state, peek)
                warpgroup.fence()
                for kb in cutlass.range_constexpr(num_k_blocks):
                    cute.gemm(
                        tiled_mma,
                        acc,
                        tCrA[(None, None, kb, ab_read_state.index)],
                        tCrB[(None, None, kb, ab_read_state.index)],
                        acc,
                    )
                    tiled_mma.set(warpgroup.Field.ACCUMULATE, True)
                warpgroup.commit_group()
                warpgroup.wait_group(0)
                ab_pipeline.consumer_release(ab_release_state)
                ab_read_state.advance()
                ab_release_state.advance()
                peek = cutlass.Boolean(1)
                if ab_read_state.count < read_base + k_tile_cnt:
                    peek = ab_pipeline.consumer_try_wait(ab_read_state)
                # refill: the producer loads the next not-yet-loaded k-tile of THIS feature.
                if warp_idx == 0:
                    kt_next = ab_producer_state.count - prod_base
                    if kt_next < k_tile_cnt:
                        ab_pipeline.producer_acquire(ab_producer_state)
                        bar = ab_pipeline.producer_get_barrier(ab_producer_state)
                        cute.copy(
                            tma_atom_a,
                            tAgA[(None, kt_next)],
                            tAsA[(None, ab_producer_state.index)],
                            tma_bar_ptr=bar,
                        )
                        cute.copy(
                            tma_atom_b,
                            tBgB[(None, kt_next)],
                            tBsB[(None, ab_producer_state.index)],
                            tma_bar_ptr=bar,
                        )
                        ab_pipeline.producer_commit(ab_producer_state)
                        ab_producer_state.advance()

            # ---- stage acc (cast to dtype) into s_tri[:, :, dl] ----
            # partition_C the 2-D (blk_i, blk_j) slice of s_tri with the SAME thread-value layout as
            # acc, so element e maps to the matching (i,j) SMEM slot with no coordinate arithmetic.
            s_tri_d = s_tri[None, None, dl]  # (blk_i, blk_j)
            tCsTri = thr_mma_c.partition_C(s_tri_d)
            for e in cutlass.range_constexpr(acc_size):
                v = acc[e]
                tCsTri[e] = v.to(self.dtype)
                # accumulate this feature's contribution into the per-token partials (fp32)
                rsum[e] += v
                rsqsum[e] += v * v
            cute.arch.barrier()

        # ---- write this CTA's partial (sum x, sum x²) per token to mStatsPartial ----
        # Recover (i_local, j_local) per element via partition_C of an identity tensor (the same
        # thread-value layout as acc).  token = the global (i,j) row-major into the B·N² token axis.
        # Each (feature chunk, token) is written by exactly ONE CTA and each (i,j) by exactly ONE
        # thread, so there is no race and no barrier or atomic is needed.
        N = const_expr(self.N)
        cId = cute.make_identity_tensor((blk_i, blk_j))
        tCcStat = thr_mma_c.partition_C(cId)
        for e in cutlass.range_constexpr(acc_size):
            i_local = tCcStat[e][0]
            j_local = tCcStat[e][1]
            gi = bi * blk_i + i_local
            gj = bj * blk_j + j_local
            # Compile-time branch: a tile-divisible N writes unconditionally (zero runtime ALU);
            # only the partial-tile case emits the bounds check that skips the token overhang.
            if const_expr(self.N % blk_i == 0 and self.N % blk_j == 0):
                mStatsPartial[c_chunk, tok_boff + gi * N + gj, 0] = rsum[e]
                mStatsPartial[c_chunk, tok_boff + gi * N + gj, 1] = rsqsum[e]
            elif gi < N and gj < N:
                mStatsPartial[c_chunk, tok_boff + gi * N + gj, 0] = rsum[e]
                mStatsPartial[c_chunk, tok_boff + gi * N + gj, 1] = rsqsum[e]

        # ---- coalesced store of the (blk_i, blk_j, blk_d) block, d contiguous ----
        # Slice the leading batch mode b_idx (keeping i, j, d as a view) -> (N, N, D).
        mTri_b = mTri[b_idx, None, None, None]
        gTri = cute.local_tile(mTri_b, (blk_i, blk_j, blk_d), (bi, bj, c_chunk))
        # Flatten to (blk_i*blk_j*blk_d,) and split across threads; d is innermost (stride 1).
        total = const_expr(blk_i * blk_j * blk_d)
        per_thread = const_expr((total + self.num_threads - 1) // self.num_threads)
        for t in cutlass.range_constexpr(per_thread):
            lin = t * self.num_threads + tidx
            if lin < total:
                dd = lin % blk_d
                rem = lin // blk_d
                jj = rem % blk_j
                ii = rem // blk_j
                # Compile-time branch: a tile-divisible N takes the unpredicated store (zero runtime
                # ALU); only the partial-tile case emits the (i,j) bounds check.
                if const_expr(self.N % blk_i == 0 and self.N % blk_j == 0):
                    gTri[ii, jj, dd] = s_tri[ii, jj, dd]
                else:
                    if bi * blk_i + ii < N and bj * blk_j + jj < N:
                        gTri[ii, jj, dd] = s_tri[ii, jj, dd]


class TokenPairGemmStatsContPipeSm90:
    """:class:`TokenPairGemmStatsSm90` with ONE continuous ``(feature, k)`` pipeline, not one per
    feature.

    Purpose:
        Remove the per-feature prefetch ramp. Where the baseline treats each of the ``blk_d``
        features as a self-contained pipeline -- prefetch, drain, RESET -- and therefore pays the
        ramp ``blk_d`` times while feature ``d+1``'s TMA loads cannot overlap feature ``d``'s WGMMA,
        this variant streams a SINGLE flattened tile sequence ``g = dl * k_tile_cnt + kt`` across all
        features and never resets.

    Functionality and semantics:
        Identical arithmetic and identical output to the baseline -- the accumulator is reset only at
        each feature's first k-tile (``ACCUMULATE=False`` when ``kt == 0``), and that feature's final
        accumulator is staged into ``s_tri[:, :, dl]`` plus the fp32 partials when its last k-tile
        completes. The coalesced ``(blk_i, blk_j, blk_d)`` block store is kept: it is NOT the
        bottleneck, and a strided per-feature immediate store was measured ~5x SLOWER (LSU-bound,
        HMMA down to 3%).

        MEASURED on H200: wins at N=512/D=512 (0.94x) and N=256 (0.93x) by removing the ramp, but
        REGRESSES at N=1024 (~1.20x) -- occupancy stays at 12.4% because ``s_tri`` still caps the
        CTA count, so HMMA actually drops 15.9% -> 12.9% there and the staging cost is not repaid
        once ``k_tile_cnt`` is large. That asymmetry is what the ``cont_pipe`` size heuristic
        encodes; see :func:`_heuristic_cont_pipe`.

        The one structural cost: the flattened tile sequence is a ``range_constexpr`` of
        ``blk_d * ceil(N/blk_k)`` iterations, so the unrolled body -- and the compile time -- grows
        with ``N``. That is the reason the heuristic's default flips away from this variant at large
        ``N`` anyway.

    Args:
        Identical to :class:`TokenPairGemmStatsSm90`; see there for every constraint, all of which
        apply here unchanged.

    Raises:
        AssertionError: On the same tile / extent constraints as the baseline.
    """

    def __init__(
        self,
        dtype: Type[cutlass.Numeric],
        N: int,
        D: int,
        blk_i: int,
        blk_j: int,
        blk_d: int,
        blk_k: int,
        B: int = 1,
    ):
        self.dtype = dtype
        self.acc_dtype = Float32
        self.N = N
        self.D = D
        # Batch B>1: token axis M = B·N²; operands viewed (M=N,K=N,L=B·D), grid-z = B·(D/blk_d), a
        # CTA derives b = zgroup//(D/blk_d), feature chunk c = zgroup%(D/blk_d).  B==1 identical.
        self.B = B
        self.blk_i = blk_i
        self.blk_j = blk_j
        self.blk_d = blk_d
        self.blk_k = blk_k
        assert N % 8 == 0, "N must be 16-byte aligned (N % 8 == 0 for bf16/fp16)"
        assert D % blk_d == 0, "D must be a multiple of blk_d (the feature group)"
        assert blk_i % 64 == 0, "WGMMA M must be a multiple of 64"
        assert blk_j % 8 == 0 and blk_k % 16 == 0
        self.mma_warp_groups = blk_i // 64
        self.num_threads = self.mma_warp_groups * 128
        self.num_threads_per_warp_group = 128
        self.ab_stage = 3
        self.buffer_align_bytes = 1024

    @cute.jit
    def __call__(
        self,
        mA: cute.Tensor,  # (M=N, K=N, L=B·D)
        mB: cute.Tensor,  # (N=N, K=N, L=B·D)
        mTri: cute.Tensor,  # (B, N, N, D) with D contiguous
        mStatsPartial: cute.Tensor,  # (D/blk_d, B*N*N, 2) fp32
        stream: cuda.CUstream,
    ):
        """Build the MMA, SMEM layouts and TMA atoms, then launch. See the baseline's ``__call__``.

        Args:
            mA, mB, mTri, mStatsPartial, stream: Exactly as
                :meth:`TokenPairGemmStatsSm90.__call__`, with the same constraints -- this variant
                differs only in how the launched kernel schedules its loads.

        Returns:
            None. Its effect is ``mTri`` and ``mStatsPartial``.
        """
        a_layout = LayoutEnum.from_tensor(mA)
        b_layout = LayoutEnum.from_tensor(mB)
        blk_i, blk_j, blk_d, blk_k = self.blk_i, self.blk_j, self.blk_d, self.blk_k

        tiled_mma = sm90_helpers.make_trivial_tiled_mma(
            self.dtype,
            self.dtype,
            a_layout.sm90_mma_major_mode(),
            b_layout.sm90_mma_major_mode(),
            self.acc_dtype,
            (self.mma_warp_groups, 1, 1),
            (64, blk_j),
        )

        a_smem_layout = fold_cp_ops_sm90.make_smem_layout(
            self.dtype, a_layout, (blk_i, blk_k), self.ab_stage
        )
        b_smem_layout = fold_cp_ops_sm90.make_smem_layout(
            self.dtype, b_layout, (blk_j, blk_k), self.ab_stage
        )

        tma_atom_a, tma_tensor_a = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(),
            mA,
            cute.slice_(a_smem_layout, (None, None, 0)),
            (blk_i, blk_k),
            num_multicast=1,
        )
        tma_atom_b, tma_tensor_b = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(),
            mB,
            cute.slice_(b_smem_layout, (None, None, 0)),
            (blk_j, blk_k),
            num_multicast=1,
        )

        self.tma_copy_bytes = cute.size_in_bytes(
            self.dtype, cute.slice_(a_smem_layout, (None, None, 0))
        ) + cute.size_in_bytes(self.dtype, cute.slice_(b_smem_layout, (None, None, 0)))

        @cute.struct
        class SharedStorage:
            ab_pipeline_array_ptr: cute.struct.MemRange[cutlass.Int64, self.ab_stage * 2]
            sTri: cute.struct.Align[
                cute.struct.MemRange[self.dtype, blk_i * blk_j * blk_d], self.buffer_align_bytes
            ]
            sA: cute.struct.Align[
                cute.struct.MemRange[self.dtype, cute.cosize(a_smem_layout)],
                self.buffer_align_bytes,
            ]
            sB: cute.struct.Align[
                cute.struct.MemRange[self.dtype, cute.cosize(b_smem_layout)],
                self.buffer_align_bytes,
            ]

        self.shared_storage = SharedStorage

        grid = (
            (self.N + blk_i - 1) // blk_i,  # ceil_div: launch the partial M/N tile (predicated)
            (self.N + blk_j - 1) // blk_j,
            self.B * (self.D // blk_d),  # grid-z = B·(D/blk_d) feature chunks
        )
        self.kernel(
            tiled_mma,
            tma_atom_a,
            tma_tensor_a,
            tma_atom_b,
            tma_tensor_b,
            mTri,
            mStatsPartial,
            a_smem_layout,
            b_smem_layout,
        ).launch(
            grid=grid,
            block=[self.num_threads, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        tiled_mma: cute.TiledMma,
        tma_atom_a: cute.CopyAtom,
        mA: cute.Tensor,  # (M=N, K=N, L=B·D)
        tma_atom_b: cute.CopyAtom,
        mB: cute.Tensor,  # (N=N, K=N, L=B·D)
        mTri: cute.Tensor,  # (B, N, N, D)
        mStatsPartial: cute.Tensor,  # (D/blk_d, B*N*N, 2) fp32
        a_smem_layout: cute.ComposedLayout,
        b_smem_layout: cute.ComposedLayout,
    ):
        """One CTA, one continuous pipeline over every ``(feature, k-tile)`` pair.

        Semantics:
            The producer never resets between features, so the loads for feature ``d+1`` overlap the
            WGMMA of feature ``d``. The consumer's flattened index ``g`` carries both coordinates:
            ``kt = g % k_tile_cnt`` decides the accumulate flag and ``dl = g // k_tile_cnt`` selects
            the staging slice. Because both are compile-time, the whole sequence is unrolled -- the
            price of the schedule, and why the body grows with ``N``.

        Args:
            tiled_mma, tma_atom_a, tma_atom_b, mA, mB, mTri, mStatsPartial, a_smem_layout,
            b_smem_layout: As :meth:`TokenPairGemmStatsSm90.kernel`, with the same requirement that
                ``mA``/``mB`` be the TMA-mapped views rather than the raw tensors.

        Returns:
            None.
        """
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        tidx, _, _ = cute.arch.thread_idx()
        bi, bj, bd = cute.arch.block_idx()
        blk_i, blk_j, blk_d, blk_k = self.blk_i, self.blk_j, self.blk_d, self.blk_k
        # z-group -> (batch b, feature chunk c); n_dchunks = D/blk_d per batch.  B==1 -> b=0, c=bd.
        n_dchunks = const_expr(self.D // blk_d)
        b_idx = bd // n_dchunks
        c_chunk = bd % n_dchunks
        Boff = b_idx * const_expr(self.D)  # L-axis (B·D) offset of this batch's D features
        tok_boff = b_idx * (self.N * self.N)  # token-axis (B·N²) offset of this batch

        if warp_idx == 0:
            cpasync.prefetch_descriptor(tma_atom_a)
            cpasync.prefetch_descriptor(tma_atom_b)

        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)

        producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        num_warps = const_expr(self.num_threads // 32)
        consumer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, num_warps)
        ab_pipeline = pipeline.PipelineTmaAsync.create(
            barrier_storage=storage.ab_pipeline_array_ptr.data_ptr(),
            num_stages=self.ab_stage,
            producer_group=producer_group,
            consumer_group=consumer_group,
            tx_count=self.tma_copy_bytes,
            cta_layout_vmnk=cute.make_layout((1, 1, 1, 1)),
            defer_sync=True,
        )
        pipeline_init_arrive(cluster_shape_mn=(1, 1), is_relaxed=True)

        sA = storage.sA.get_tensor(a_smem_layout.outer, swizzle=a_smem_layout.inner)
        sB = storage.sB.get_tensor(b_smem_layout.outer, swizzle=b_smem_layout.inner)

        warp_group_idx = cute.arch.make_warp_uniform(tidx // self.num_threads_per_warp_group)
        wg_thread_layout = cute.make_layout(
            self.mma_warp_groups, stride=self.num_threads_per_warp_group
        )
        thr_mma = tiled_mma.get_slice(wg_thread_layout(warp_group_idx))
        thr_mma_c = tiled_mma.get_slice(tidx)
        tCsA = thr_mma.partition_A(sA)
        tCsB = thr_mma.partition_B(sB)
        tCrA = tiled_mma.make_fragment_A(tCsA)
        tCrB = tiled_mma.make_fragment_B(tCsB)
        acc = cute.make_rmem_tensor(thr_mma_c.partition_shape_C((blk_i, blk_j)), self.acc_dtype)
        num_k_blocks = const_expr(cute.size(tCrA, mode=[2]))

        # ---- partial row statistics (per element, fp32) ----
        rsum = cute.make_rmem_tensor(thr_mma_c.partition_shape_C((blk_i, blk_j)), self.acc_dtype)
        rsqsum = cute.make_rmem_tensor(thr_mma_c.partition_shape_C((blk_i, blk_j)), self.acc_dtype)
        acc_size = const_expr(cute.size(acc))
        for e in cutlass.range_constexpr(acc_size):
            rsum[e] = self.acc_dtype(0.0)
            rsqsum[e] = self.acc_dtype(0.0)

        # identity coordinates (recover (i_local, j_local) per accumulator element).
        cId = cute.make_identity_tensor((blk_i, blk_j))
        tCcStat = thr_mma_c.partition_C(cId)

        # tri staging: (blk_i, blk_j, blk_d), d contiguous (coalesced block store at the end).
        s_tri = storage.sTri.get_tensor(
            cute.make_ordered_layout((blk_i, blk_j, blk_d), order=(2, 1, 0))
        )

        # ceil_div: the last k-tile may be partial (N % blk_k != 0); the TMA zero-fills the tail.
        k_tile_cnt = const_expr((self.N + blk_k - 1) // blk_k)
        N = const_expr(self.N)
        total_tiles = const_expr(blk_d * k_tile_cnt)  # ONE continuous pipeline over all (d, kt)

        pipeline_init_wait(cluster_shape_mn=(1, 1))

        ab_producer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.ab_stage
        )
        ab_read_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.ab_stage
        )
        ab_release_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.ab_stage
        )

        # Per-feature GMEM TMA views, precomputed (dl is compile-time, so this is just blk_d views).
        def _ab_views(dl):
            """The ``(A, B)`` GMEM TMA partitions for one local feature index.

            Args:
                dl: The local feature index, 0 <= dl < blk_d. A compile-time Python int -- this
                    helper runs during tracing, so a runtime value would not resolve.

            Returns:
                ``(tAgA, tBgB)``, the GMEM-side TMA partitions for that feature's operand slices.
            """
            d = Boff + c_chunk * blk_d + dl  # L-index into mA/mB (B·D axis)
            gA = cute.local_tile(mA[None, None, d], (blk_i, blk_k), (bi, None))
            gB = cute.local_tile(mB[None, None, d], (blk_j, blk_k), (bj, None))
            tAsA, tAgA = cpasync.tma_partition(
                tma_atom_a,
                0,
                cute.make_layout(1),
                cute.group_modes(sA, 0, 2),
                cute.group_modes(gA, 0, 2),
            )
            tBsB, tBgB = cpasync.tma_partition(
                tma_atom_b,
                0,
                cute.make_layout(1),
                cute.group_modes(sB, 0, 2),
                cute.group_modes(gB, 0, 2),
            )
            return tAgA, tBgB

        ab_views = [_ab_views(dl) for dl in range(blk_d)]

        # The SMEM-side TMA destinations are feature-independent (same sA/sB buffer); take them once.
        tAsA_buf, _ = cpasync.tma_partition(
            tma_atom_a,
            0,
            cute.make_layout(1),
            cute.group_modes(sA, 0, 2),
            cute.group_modes(cute.local_tile(mA[None, None, 0], (blk_i, blk_k), (bi, None)), 0, 2),
        )
        tBsB_buf, _ = cpasync.tma_partition(
            tma_atom_b,
            0,
            cute.make_layout(1),
            cute.group_modes(sB, 0, 2),
            cute.group_modes(cute.local_tile(mB[None, None, 0], (blk_j, blk_k), (bj, None)), 0, 2),
        )

        # ---- prologue prefetch: fill the pipeline (spans feature boundaries) ----
        prefetch_cnt = const_expr(min(self.ab_stage, total_tiles))
        if warp_idx == 0:
            for g in cutlass.range_constexpr(prefetch_cnt):
                dl_p = const_expr(g // k_tile_cnt)
                kt_p = const_expr(g % k_tile_cnt)
                tAgA, tBgB = ab_views[dl_p]
                ab_pipeline.producer_acquire(ab_producer_state)
                bar = ab_pipeline.producer_get_barrier(ab_producer_state)
                cute.copy(
                    tma_atom_a,
                    tAgA[(None, kt_p)],
                    tAsA_buf[(None, ab_producer_state.index)],
                    tma_bar_ptr=bar,
                )
                cute.copy(
                    tma_atom_b,
                    tBgB[(None, kt_p)],
                    tBsB_buf[(None, ab_producer_state.index)],
                    tma_bar_ptr=bar,
                )
                ab_pipeline.producer_commit(ab_producer_state)
                ab_producer_state.advance()

        # ---- continuous consumer loop; ACCUMULATE resets at each feature's first k-tile ----
        tiled_mma.set(warpgroup.Field.ACCUMULATE, False)
        peek = cutlass.Boolean(1)
        if 0 < total_tiles:
            peek = ab_pipeline.consumer_try_wait(ab_read_state)
        for g in cutlass.range_constexpr(total_tiles):
            kt = const_expr(g % k_tile_cnt)
            dl = const_expr(g // k_tile_cnt)
            tiled_mma.set(warpgroup.Field.ACCUMULATE, const_expr(kt != 0))
            ab_pipeline.consumer_wait(ab_read_state, peek)
            warpgroup.fence()
            for kb in cutlass.range_constexpr(num_k_blocks):
                cute.gemm(
                    tiled_mma,
                    acc,
                    tCrA[(None, None, kb, ab_read_state.index)],
                    tCrB[(None, None, kb, ab_read_state.index)],
                    acc,
                )
                tiled_mma.set(warpgroup.Field.ACCUMULATE, True)
            warpgroup.commit_group()
            warpgroup.wait_group(0)
            ab_pipeline.consumer_release(ab_release_state)
            ab_read_state.advance()
            ab_release_state.advance()
            peek = cutlass.Boolean(1)
            if const_expr(g + 1) < total_tiles:
                peek = ab_pipeline.consumer_try_wait(ab_read_state)
            # refill: the producer streams the next tile (continuous across features).
            if warp_idx == 0:
                g_next = const_expr(g + prefetch_cnt)
                if const_expr(g_next < total_tiles):
                    dl_n = const_expr(g_next // k_tile_cnt)
                    kt_n = const_expr(g_next % k_tile_cnt)
                    tAgA, tBgB = ab_views[dl_n]
                    ab_pipeline.producer_acquire(ab_producer_state)
                    bar = ab_pipeline.producer_get_barrier(ab_producer_state)
                    cute.copy(
                        tma_atom_a,
                        tAgA[(None, kt_n)],
                        tAsA_buf[(None, ab_producer_state.index)],
                        tma_bar_ptr=bar,
                    )
                    cute.copy(
                        tma_atom_b,
                        tBgB[(None, kt_n)],
                        tBsB_buf[(None, ab_producer_state.index)],
                        tma_bar_ptr=bar,
                    )
                    ab_pipeline.producer_commit(ab_producer_state)
                    ab_producer_state.advance()
            # when this feature's LAST k-tile just finished, stage acc -> s_tri[:, :, dl] + stats.
            if const_expr(kt == k_tile_cnt - 1):
                s_tri_d = s_tri[None, None, dl]
                tCsTri = thr_mma_c.partition_C(s_tri_d)
                for e in cutlass.range_constexpr(acc_size):
                    v = acc[e]
                    tCsTri[e] = v.to(self.dtype)
                    rsum[e] += v
                    rsqsum[e] += v * v

        # ---- write this CTA's partial (sum x, sum x²) per token to mStatsPartial ----
        for e in cutlass.range_constexpr(acc_size):
            i_local = tCcStat[e][0]
            j_local = tCcStat[e][1]
            gi = bi * blk_i + i_local
            gj = bj * blk_j + j_local
            # Compile-time branch: a tile-divisible N writes unconditionally (zero runtime ALU);
            # only the partial-tile case emits the bounds check that skips the token overhang.
            if const_expr(self.N % blk_i == 0 and self.N % blk_j == 0):
                mStatsPartial[c_chunk, tok_boff + gi * N + gj, 0] = rsum[e]
                mStatsPartial[c_chunk, tok_boff + gi * N + gj, 1] = rsqsum[e]
            elif gi < N and gj < N:
                mStatsPartial[c_chunk, tok_boff + gi * N + gj, 0] = rsum[e]
                mStatsPartial[c_chunk, tok_boff + gi * N + gj, 1] = rsqsum[e]

        # ---- coalesced store of the (blk_i, blk_j, blk_d) block, d contiguous ----
        cute.arch.barrier()
        # Slice the leading batch mode b_idx (keeping i, j, d as a view) -> (N, N, D).
        mTri_b = mTri[b_idx, None, None, None]
        gTri = cute.local_tile(mTri_b, (blk_i, blk_j, blk_d), (bi, bj, c_chunk))
        total = const_expr(blk_i * blk_j * blk_d)
        per_thread = const_expr((total + self.num_threads - 1) // self.num_threads)
        for t in cutlass.range_constexpr(per_thread):
            lin = t * self.num_threads + tidx
            if lin < total:
                dd = lin % blk_d
                rem = lin // blk_d
                jj = rem % blk_j
                ii = rem // blk_j
                # Compile-time branch: a tile-divisible N takes the unpredicated store (zero runtime
                # ALU); only the partial-tile case emits the (i,j) bounds check.
                if const_expr(self.N % blk_i == 0 and self.N % blk_j == 0):
                    gTri[ii, jj, dd] = s_tri[ii, jj, dd]
                else:
                    if bi * blk_i + ii < N and bj * blk_j + jj < N:
                        gTri[ii, jj, dd] = s_tri[ii, jj, dd]


class RowStatsReduceSm90:
    """Reduce the per-feature-chunk partial sums to one ``(mean, rstd)`` per token — op 8's stats.

    Purpose:
        Turn the ``(n_partials, n_tokens, 2)`` workspace the token-pair kernel filled into the
        ``(n_tokens, 2)`` statistics the normalize-on-load prologue reads.

    Functionality and semantics:
        Token-parallel and trivial: one thread owns one token, loops the ``n_partials`` rows summing
        ``sum(x)`` and ``sum(x^2)``, then computes ``mean = Sx/D``, ``var = Sxx/D - mean^2`` and
        ``rstd = rsqrt(var + eps)``, writing ``mStats[t] = (mean, rstd)``. The partials are fp32
        accumulations of the fp32 WGMMA accumulator -- NOT of the narrowed values stored to ``tri``
        -- so these statistics match an fp32 reference to roughly 1e-6, and differ slightly from
        statistics recomputed from the stored ``tri``. That is the correct behaviour, not an error;
        it matches an unfused LayerNorm applied to the exact product.

        ``var`` is computed from the raw second moment, which is why ``eps`` matters: for a token
        whose values are large and nearly equal, ``Sxx/D - mean^2`` is a difference of close
        quantities and can come out slightly negative, and ``eps`` is what keeps the ``rsqrt``
        finite.

    Args:
        n_tokens: The token count, ``B * N * N``. Must equal ``mStatsPartial``'s middle extent; a
            smaller value silently leaves the tail of ``mStats`` uninitialised, which then feeds
            garbage statistics into the projection.
        n_partials: The number of feature chunks, ``D // blk_d``. **Per batch**, not ``B*D/blk_d``:
            each token only carries its own batch's features. Getting this wrong sums another
            batch's partials into this token's mean.
        D: The feature count, used only as the divisor. Must be the true feature extent -- the
            partials cover exactly ``D`` values per token.
        num_threads: Threads per block, and hence tokens per block. Any positive multiple of the
            warp size; it only affects the grid split.

    Raises:
        Nothing. Every constraint above is trusted, because the caller builds all three workspaces.
    """

    def __init__(self, n_tokens: int, n_partials: int, D: int, num_threads: int = 256):
        self.n_tokens = n_tokens
        self.n_partials = n_partials
        self.D = D
        self.num_threads = num_threads

    @cute.jit
    def __call__(
        self,
        mStatsPartial: cute.Tensor,  # (n_partials, n_tokens, 2) fp32
        mStats: cute.Tensor,  # (n_tokens, 2) fp32 -> (mean, rstd)
        eps: Float32,
        stream: cuda.CUstream,
    ):
        """Launch the reduction over a ceil-div grid of token blocks.

        Args:
            mStatsPartial: The ``(n_partials, n_tokens, 2)`` fp32 workspace, fully written by the
                token-pair kernel. Reading it before that kernel completes on the same stream is a
                race; both launches go on one stream, which is what orders them.
            mStats: The ``(n_tokens, 2)`` fp32 output. Written unconditionally for every in-range
                token, so it may be uninitialised.
            eps: The LayerNorm epsilon, added to the variance before the reciprocal square root.
                Must be positive; a zero or negative one can make ``rstd`` infinite or NaN.
            stream: The CUDA stream. Pass :func:`_fake_stream` at compile time.

        Returns:
            None. Its effect is ``mStats``.
        """
        grid = ((self.n_tokens + self.num_threads - 1) // self.num_threads, 1, 1)
        self.kernel(mStatsPartial, mStats, eps).launch(
            grid=grid, block=[self.num_threads, 1, 1], stream=stream
        )

    @cute.kernel
    def kernel(
        self,
        mStatsPartial: cute.Tensor,  # (n_partials, n_tokens, 2)
        mStats: cute.Tensor,  # (n_tokens, 2)
        eps: Float32,
    ):
        """One thread, one token: sum the partials and publish ``(mean, rstd)``.

        Args:
            mStatsPartial: The partial-sums workspace.
            mStats: The ``(mean, rstd)`` output.
            eps: The variance epsilon.

        Returns:
            None. Threads past ``n_tokens`` return without writing.
        """
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        t = bidx * self.num_threads + tidx
        n_partials = const_expr(self.n_partials)
        inv_D = const_expr(1.0 / self.D)
        if t < self.n_tokens:
            sx = Float32(0.0)
            sxx = Float32(0.0)
            for dt in cutlass.range_constexpr(n_partials):
                sx += mStatsPartial[dt, t, 0]
                sxx += mStatsPartial[dt, t, 1]
            mean = sx * inv_D
            var = sxx * inv_D - mean * mean
            rstd = cute.math.rsqrt(var + eps, fastmath=True)
            mStats[t, 0] = mean
            mStats[t, 1] = rstd


class FeatureGemmPrologLnSm90:
    """Ops 9 and 12 with op 8's rescale folded into the load — the ``prolog_ln`` projection.

    Purpose:
        Contract the FEATURE axis of ``tri`` against the output projection while normalizing each
        staged A tile in place, so the LayerNorm never costs a separate pass over the ``O(N^2 * D)``
        activation.

    Functionality and semantics:
        Computes ``out[m, d'] = sum_d LN(tri)[m, d] * proj_w[d', d] + proj_b[d']``, optionally times
        ``gate3[m, d']``, where ``LN(tri)[m, d] = (tri[m, d] - mean[m]) * rstd[m] * norm_w[d] +
        norm_b[d]``. A is ``tri`` viewed ``(M=token, K=d)``, B is ``proj_w`` viewed ``(N=d', K=d)``,
        and the product is ``A @ B^T`` with fp32 accumulation.

        Normalizing a staged tile requires a ``fence_view_async_shared`` plus a barrier before the
        WGMMA may read it. A **1-deep software pipeline** keeps that fence off the critical path by
        lagging the WGMMA one tile behind the normalize::

            prologue: wait(s0) -> normalize(s0) -> fence -> barrier
            loop k:   WGMMA(stage k), commit                       (no drain)
                      if k >= 1: wait_group(1) [drain WGMMA(k-1)] -> release(k-1) -> refill
                      if k+1 < cnt: wait(s_{k+1}) -> normalize(s_{k+1}) -> fence -> barrier
                                                                   (overlaps WGMMA(k))
            epilogue: wait_group(0) -> release(last)

        Tile ``k+1``'s normalize writes a DIFFERENT stage buffer from the in-flight WGMMA's read of
        tile ``k``, so there is no shared-memory hazard -- which is exactly why ``ab_stage >= 2`` is
        required. The refill is bound to the release because warp 0 is BOTH the producer and a WGMMA
        consumer in this single-warpgroup design: acquiring only the just-released stage is what
        guarantees warp 0 never blocks on ``producer_acquire`` ahead of the warpgroup-wide
        ``wait_group``, which would deadlock.

        Two compile-time-gated predication paths handle a ``D`` that is a multiple of 8 but not 16.
        The affine vectors are padded to the next ``blk_k`` multiple with zeros, so a padded column
        normalizes to ``(0 - mu) * rstd * 0 + 0 = 0`` and contributes nothing -- no runtime predicate
        needed. The output side pads instead: the last N-tile's B rows are TMA zero-filled, so the
        padded output columns compute zero and are sliced off by the caller. When ``D % 16 == 0``
        both gates fold away and the generated code is identical to a build with no padding at all.

    Args:
        dtype: Element type of ``tri``, ``proj_w`` and ``out``. Must be the SAME for all three --
            there is no internal cast, and a mismatched weight is a launch-time type error.
        M: The token count, ``B * N * N``. Must be a multiple of ``blk_m``; the M axis is NOT
            predicated, so a non-multiple would read and write past the end.
        D: The feature extent, which is simultaneously the contraction length and the output width.
        blk_m: Token tile, i.e. the WGMMA M. Must be a multiple of 64.
        blk_n: Output-feature tile. Must be a multiple of 16 -- the shared-memory swizzle atom
            constraint. It need not divide ``D`` (the N-tile count is ceil-div and the tail is
            predicated), but a value that does divide it avoids the padded columns entirely.
        blk_k: Contraction tile over the feature axis. Must be a multiple of 16, and
            ``num_threads % (blk_k // vecsize)`` must be 0 so the normalize copy tiles evenly --
            which restricts it to {64, 32, 16} at the thread counts here.
        has_bias: Whether ``proj_b`` is supplied. False makes the epilogue add a staged zero (the
            bias scratch is zero-filled), so the two builds differ only in the host-side load.
        has_norm_bias: Whether ``norm_b``, the LayerNorm's own bias, is supplied. False stages a
            zero into ``s_normb`` exactly as the k-tile padding already does, so the normalize adds
            ``+ 0.0`` -- the identity, and the arithmetic is untouched. It is a separate flag from
            ``has_bias`` because the two biases enter at different points: this one inside the
            normalize prologue, that one in the epilogue.
        has_gate3: Whether the output gate is applied. False prunes the gate load and multiply via
            ``const_expr``, leaving an epilogue identical to a build compiled without the feature.

    Raises:
        AssertionError: If ``M % blk_m``, ``blk_m % 64``, ``blk_n % 16`` or ``blk_k % 16`` is
            violated, or if ``D`` is not a multiple of ``blk_n // 2`` and ``blk_k // 2``.
    """

    def __init__(
        self,
        dtype: Type[cutlass.Numeric],
        M: int,
        D: int,
        blk_m: int,
        blk_n: int,
        blk_k: int,
        has_bias: bool,
        has_gate3: bool = False,
        has_norm_bias: bool = True,
    ):
        self.dtype = dtype
        self.acc_dtype = Float32
        self.M = M
        self.D = D  # = K (the contraction) = N_out (d')
        self.blk_m = blk_m
        self.blk_n = blk_n
        self.blk_k = blk_k
        self.has_bias = has_bias
        self.has_gate3 = has_gate3
        self.has_norm_bias = has_norm_bias
        assert M % blk_m == 0 and D % (blk_n // 2) == 0 and D % (blk_k // 2) == 0
        assert blk_m % 64 == 0, "WGMMA M must be a multiple of 64"
        assert blk_n % 16 == 0 and blk_k % 16 == 0  # blk_n must be %16 for the swizzle layout
        self.mma_warp_groups = blk_m // 64
        self.num_threads = self.mma_warp_groups * 128
        self.num_threads_per_warp_group = 128
        self.ab_stage = 3
        self.buffer_align_bytes = 1024

    @cute.jit
    def __call__(
        self,
        mTri: cute.Tensor,  # (M, D) K(=d)-major  (the A operand)
        mProjW: cute.Tensor,  # (D, D) = (d', d) K(=d)-major  (the B operand)
        mStats: cute.Tensor,  # (M, 2) fp32 (mean, rstd) per token
        mNormW: cute.Tensor,  # (D,) fp32 LayerNorm gain
        mNormB: Optional[cute.Tensor],  # (D,) fp32 LayerNorm bias, or None
        mProjB: Optional[cute.Tensor],  # (D,) fp32 projection bias, or None
        mOut: cute.Tensor,  # (M, D_pad) row-major output
        eps: Float32,
        stream: cuda.CUstream,
        mGate3: Optional[cute.Tensor] = None,  # (M, D) row-major gate, or None
    ):
        """Build the MMA, SMEM layouts and TMA atoms, then launch the projection.

        Semantics:
            The affine scratch is padded to ``ceil(D / blk_k) * blk_k`` so the normalize can index a
            padded column without a runtime predicate; when ``D % blk_k == 0`` the padding is empty
            and the zero-fill loop is pruned.

        Args:
            mTri: The ``(M, D)`` activation with ``D`` contiguous. Its contents are consumed as
                UN-normalized values -- normalizing them beforehand would apply the affine twice.
            mProjW: The ``(D, D)`` projection weight, ``d`` contiguous. Must be ``dtype``; an fp32
                weight against a 16-bit activation is a type error at launch, deliberately, because
                casting per call would hide a conversion the caller should do once.
            mStats: The ``(M, 2)`` fp32 statistics, one row per token.
            mNormW: The ``(D,)`` fp32 LayerNorm gain. Must have exactly ``D`` entries: a shorter one
                is read past its end for the tail of every tile.
            mNormB: The ``(D,)`` fp32 LayerNorm bias, same requirement.
            mProjB: The ``(D,)`` fp32 projection bias, or None. Must be None exactly when the object
                was built with ``has_bias=False``; a mismatch either drops the bias or reads null.
            mOut: The ``(M, D_pad)`` output, where ``D_pad`` is ``D`` rounded up to a ``blk_n``
                multiple. Passing an un-padded buffer when ``D % blk_n != 0`` makes the last tile's
                TMA address dangle.
            eps: Accepted for signature symmetry with the statistics kernel; the reciprocal standard
                deviation arrives already computed in ``mStats``.
            stream: The CUDA stream. Pass :func:`_fake_stream` at compile time.
            mGate3: The ``(M, D)`` gate, or None. Must be None exactly when ``has_gate3`` is False.

        Returns:
            None. Its effect is ``mOut``.
        """
        a_layout = LayoutEnum.from_tensor(mTri)
        b_layout = LayoutEnum.from_tensor(mProjW)
        blk_m, blk_n, blk_k = self.blk_m, self.blk_n, self.blk_k
        D = self.D

        tiled_mma = sm90_helpers.make_trivial_tiled_mma(
            self.dtype,
            self.dtype,
            a_layout.sm90_mma_major_mode(),
            b_layout.sm90_mma_major_mode(),
            self.acc_dtype,
            (self.mma_warp_groups, 1, 1),
            (64, blk_n),
        )

        a_smem_layout = fold_cp_ops_sm90.make_smem_layout(
            self.dtype, a_layout, (blk_m, blk_k), self.ab_stage
        )
        b_smem_layout = fold_cp_ops_sm90.make_smem_layout(
            self.dtype, b_layout, (blk_n, blk_k), self.ab_stage
        )

        tma_atom_a, tma_tensor_a = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(),
            mTri,
            cute.slice_(a_smem_layout, (None, None, 0)),
            (blk_m, blk_k),
            num_multicast=1,
        )
        tma_atom_b, tma_tensor_b = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(),
            mProjW,
            cute.slice_(b_smem_layout, (None, None, 0)),
            (blk_n, blk_k),
            num_multicast=1,
        )

        self.tma_copy_bytes = cute.size_in_bytes(
            self.dtype, cute.slice_(a_smem_layout, (None, None, 0))
        ) + cute.size_in_bytes(self.dtype, cute.slice_(b_smem_layout, (None, None, 0)))

        # Pad the affine scratch to the next blk_k multiple so norm_w[k] = norm_b[k] = 0 for
        # k in [D, D_kpad).  The load loop writes [0, D) and explicitly zeros [D, D_kpad), which is
        # what lets _normalize_tile index k with no runtime predicate.
        D_kpad = ((D + blk_k - 1) // blk_k) * blk_k  # == D when D % blk_k == 0

        @cute.struct
        class SharedStorage:
            ab_pipeline_array_ptr: cute.struct.MemRange[cutlass.Int64, self.ab_stage * 2]
            s_stats: cute.struct.Align[
                cute.struct.MemRange[Float32, blk_m * 2], self.buffer_align_bytes
            ]
            s_normw: cute.struct.Align[
                cute.struct.MemRange[Float32, D_kpad], self.buffer_align_bytes
            ]
            s_normb: cute.struct.Align[
                cute.struct.MemRange[Float32, D_kpad], self.buffer_align_bytes
            ]
            s_proj_b: cute.struct.Align[
                cute.struct.MemRange[Float32, D_kpad], self.buffer_align_bytes
            ]
            sA: cute.struct.Align[
                cute.struct.MemRange[self.dtype, cute.cosize(a_smem_layout)],
                self.buffer_align_bytes,
            ]
            sB: cute.struct.Align[
                cute.struct.MemRange[self.dtype, cute.cosize(b_smem_layout)],
                self.buffer_align_bytes,
            ]

        self.shared_storage = SharedStorage

        grid = (self.M // blk_m, (self.D + blk_n - 1) // blk_n, 1)  # ceil_div N for a partial tile
        self.kernel(
            tiled_mma,
            tma_atom_a,
            tma_tensor_a,
            tma_atom_b,
            tma_tensor_b,
            mStats,
            mNormW,
            mNormB,
            mProjB,
            mOut,
            eps,
            a_smem_layout,
            b_smem_layout,
            mGate3,
        ).launch(
            grid=grid,
            block=[self.num_threads, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        tiled_mma: cute.TiledMma,
        tma_atom_a: cute.CopyAtom,
        mTri: cute.Tensor,  # (M, D)
        tma_atom_b: cute.CopyAtom,
        mProjW: cute.Tensor,  # (D, D)
        mStats: cute.Tensor,  # (M, 2)
        mNormW: cute.Tensor,  # (D,)
        mNormB: cute.Tensor,  # (D,)
        mProjB: Optional[cute.Tensor],  # (D,) or None
        mOut: cute.Tensor,  # (M, D_pad)
        eps: Float32,
        a_smem_layout: cute.ComposedLayout,
        b_smem_layout: cute.ComposedLayout,
        mGate3: Optional[cute.Tensor],  # (M, D) or None
    ):
        """One CTA: stage the loop-invariant scratch, then run the 1-deep normalize/WGMMA pipeline.

        Args:
            tiled_mma: The warpgroup MMA.
            tma_atom_a, tma_atom_b: The G2S bulk-tensor atoms.
            mTri, mProjW: The TMA-mapped operand views (not the raw tensors).
            mStats, mNormW, mNormB, mProjB, mOut, mGate3: As in ``__call__``.
            eps: Unused in the body; kept so the compiled signature matches the launcher.
            a_smem_layout, b_smem_layout: The swizzled staged layouts the atoms were built from.

        Returns:
            None. Its effect is ``mOut``.
        """
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        tidx, _, _ = cute.arch.thread_idx()
        bm, bn, _ = cute.arch.block_idx()
        blk_m, blk_n, blk_k = self.blk_m, self.blk_n, self.blk_k
        D = const_expr(self.D)
        D_kpad = const_expr(((self.D + self.blk_k - 1) // self.blk_k) * self.blk_k)
        NT = const_expr(self.num_threads)
        has_bias = const_expr(self.has_bias)
        has_gate3 = const_expr(self.has_gate3)
        has_norm_bias = const_expr(self.has_norm_bias)

        if warp_idx == 0:
            cpasync.prefetch_descriptor(tma_atom_a)
            cpasync.prefetch_descriptor(tma_atom_b)

        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)

        producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        num_warps = const_expr(self.num_threads // 32)
        consumer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, num_warps)
        ab_pipeline = pipeline.PipelineTmaAsync.create(
            barrier_storage=storage.ab_pipeline_array_ptr.data_ptr(),
            num_stages=self.ab_stage,
            producer_group=producer_group,
            consumer_group=consumer_group,
            tx_count=self.tma_copy_bytes,
            cta_layout_vmnk=cute.make_layout((1, 1, 1, 1)),
            defer_sync=True,
        )
        pipeline_init_arrive(cluster_shape_mn=(1, 1), is_relaxed=True)

        sA = storage.sA.get_tensor(a_smem_layout.outer, swizzle=a_smem_layout.inner)
        sB = storage.sB.get_tensor(b_smem_layout.outer, swizzle=b_smem_layout.inner)
        s_stats = storage.s_stats.get_tensor(cute.make_layout((blk_m, 2)))
        s_normw = storage.s_normw.get_tensor(cute.make_layout(D_kpad))
        s_normb = storage.s_normb.get_tensor(cute.make_layout(D_kpad))
        s_proj_b = storage.s_proj_b.get_tensor(cute.make_layout(D_kpad))

        # ---- stage the loop-invariant scratch (once per output tile) ----
        for r in cutlass.range_constexpr((blk_m + NT - 1) // NT):
            mm = r * NT + tidx
            if mm < blk_m:
                tok = bm * blk_m + mm
                s_stats[mm, 0] = mStats[tok, 0]
                s_stats[mm, 1] = mStats[tok, 1]
        for r in cutlass.range_constexpr((D_kpad + NT - 1) // NT):
            dd = r * NT + tidx
            if dd < D:
                s_normw[dd] = mNormW[dd]
                # An absent LayerNorm bias stages a zero, which is EXACTLY what the k-tile padding
                # below already writes -- so `+ s_normb[k]` in the normalize becomes `+ 0.0`, the
                # identity, and no arithmetic changes. With the bias present this is a
                # `const_expr(True)` branch, so the DSL prunes it to the unconditional load it
                # replaced and the biased build is unchanged.
                if const_expr(has_norm_bias):
                    s_normb[dd] = mNormB[dd]
                else:
                    s_normb[dd] = Float32(0.0)
                if const_expr(has_bias):
                    s_proj_b[dd] = mProjB[dd]
                else:
                    s_proj_b[dd] = Float32(0.0)
            # Zero the k-tile padding region [D, D_kpad).  _normalize_tile reads these for a padded
            # k: (0-mu)*rstd*0 + 0 = 0, i.e. zero contribution.  Compile-time gate: D_kpad == D
            # (D % blk_k == 0) eliminates this at compile time.
            if const_expr(D_kpad > D):
                if dd >= D:
                    if dd < D_kpad:
                        s_normw[dd] = Float32(0.0)
                        s_normb[dd] = Float32(0.0)
                        s_proj_b[dd] = Float32(0.0)
        cute.arch.barrier()

        # ---- MMA partitioning ----
        warp_group_idx = cute.arch.make_warp_uniform(tidx // self.num_threads_per_warp_group)
        wg_thread_layout = cute.make_layout(
            self.mma_warp_groups, stride=self.num_threads_per_warp_group
        )
        thr_mma = tiled_mma.get_slice(wg_thread_layout(warp_group_idx))
        thr_mma_c = tiled_mma.get_slice(tidx)
        tCsA = thr_mma.partition_A(sA)
        tCsB = thr_mma.partition_B(sB)
        tCrA = tiled_mma.make_fragment_A(tCsA)
        tCrB = tiled_mma.make_fragment_B(tCsB)
        acc = cute.make_rmem_tensor(thr_mma_c.partition_shape_C((blk_m, blk_n)), self.acc_dtype)
        num_k_blocks = const_expr(cute.size(tCrA, mode=[2]))

        vecsize = const_expr(min(blk_k, 128 // self.dtype.width))
        threads_per_row = const_expr(blk_k // vecsize)
        tiled_copy_norm = tiled_copy_2d(self.dtype, threads_per_row, NT, vecsize)
        thr_copy_norm = tiled_copy_norm.get_slice(tidx)

        k_tile_cnt = const_expr((D + blk_k - 1) // blk_k)  # ceil_div: TMA zero-pads a partial tile
        pipeline_init_wait(cluster_shape_mn=(1, 1))

        ab_producer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.ab_stage
        )
        ab_read_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.ab_stage
        )
        ab_release_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.ab_stage
        )

        gA = cute.local_tile(mTri, (blk_m, blk_k), (bm, None))
        gB = cute.local_tile(mProjW, (blk_n, blk_k), (bn, None))
        tAsA, tAgA = cpasync.tma_partition(
            tma_atom_a,
            0,
            cute.make_layout(1),
            cute.group_modes(sA, 0, 2),
            cute.group_modes(gA, 0, 2),
        )
        tBsB, tBgB = cpasync.tma_partition(
            tma_atom_b,
            0,
            cute.make_layout(1),
            cute.group_modes(sB, 0, 2),
            cute.group_modes(gB, 0, 2),
        )

        # ---- prefetch ----
        if warp_idx == 0:
            prefetch_cnt = const_expr(min(self.ab_stage, k_tile_cnt))
            for kt in cutlass.range_constexpr(prefetch_cnt):
                ab_pipeline.producer_acquire(ab_producer_state)
                bar = ab_pipeline.producer_get_barrier(ab_producer_state)
                cute.copy(
                    tma_atom_a,
                    tAgA[(None, kt)],
                    tAsA[(None, ab_producer_state.index)],
                    tma_bar_ptr=bar,
                )
                cute.copy(
                    tma_atom_b,
                    tBgB[(None, kt)],
                    tBsB[(None, ab_producer_state.index)],
                    tma_bar_ptr=bar,
                )
                ab_pipeline.producer_commit(ab_producer_state)
                ab_producer_state.advance()

        tiled_mma.set(warpgroup.Field.ACCUMULATE, False)

        # ---- prologue: wait tile 0, normalize it, fence + barrier (one-time, off the loop) ----
        peek = cutlass.Boolean(1)
        if 0 < k_tile_cnt:
            peek = ab_pipeline.consumer_try_wait(ab_read_state)
            ab_pipeline.consumer_wait(ab_read_state, peek)
            self._normalize_tile(
                sA[None, None, ab_read_state.index],
                s_stats,
                s_normw,
                s_normb,
                thr_copy_norm,
                Int32(0),
            )
            cute.arch.fence_view_async_shared()
            cute.arch.barrier()

        # ---- mainloop (1-deep pipeline): WGMMA(k) flies while normalize(k+1) runs under it ----
        # Order per k: WGMMA(k) on the tile normalized last iteration (or by the prologue at k=0);
        # advance the read state; at k>=1 wait_group(1) drains WGMMA(k-1) -> release(k-1) and REFILL
        # the freed stage; then wait + normalize tile k+1, overlapping WGMMA(k).  The refill is
        # bound to the release: warp 0 is BOTH producer and a WGMMA consumer, so it must never block
        # on producer_acquire ahead of the warpgroup-wide wait_group, and acquiring only the
        # just-released stage is what guarantees the stage is free.
        for k_tile in cutlass.range(k_tile_cnt, unroll=1):
            stage = ab_read_state.index
            warpgroup.fence()
            for kb in cutlass.range_constexpr(num_k_blocks):
                cute.gemm(
                    tiled_mma,
                    acc,
                    tCrA[(None, None, kb, stage)],
                    tCrB[(None, None, kb, stage)],
                    acc,
                )
                tiled_mma.set(warpgroup.Field.ACCUMULATE, True)
            warpgroup.commit_group()
            ab_read_state.advance()
            # Drain WGMMA(k-1) -> release its stage -> refill that freed stage (warp 0).
            if k_tile >= 1:
                warpgroup.wait_group(1)
                ab_pipeline.consumer_release(ab_release_state)
                ab_release_state.advance()
                if warp_idx == 0:
                    kt_next = ab_producer_state.count
                    if kt_next < k_tile_cnt:
                        ab_pipeline.producer_acquire(ab_producer_state)
                        bar = ab_pipeline.producer_get_barrier(ab_producer_state)
                        cute.copy(
                            tma_atom_a,
                            tAgA[(None, kt_next)],
                            tAsA[(None, ab_producer_state.index)],
                            tma_bar_ptr=bar,
                        )
                        cute.copy(
                            tma_atom_b,
                            tBgB[(None, kt_next)],
                            tBsB[(None, ab_producer_state.index)],
                            tma_bar_ptr=bar,
                        )
                        ab_pipeline.producer_commit(ab_producer_state)
                        ab_producer_state.advance()
            # Normalize tile k+1 into ITS stage (overlaps the WGMMA launched above).
            if ab_read_state.count < k_tile_cnt:
                peek = ab_pipeline.consumer_try_wait(ab_read_state)
                ab_pipeline.consumer_wait(ab_read_state, peek)
                self._normalize_tile(
                    sA[None, None, ab_read_state.index],
                    s_stats,
                    s_normw,
                    s_normb,
                    thr_copy_norm,
                    Int32(k_tile + 1),
                )
                cute.arch.fence_view_async_shared()
                cute.arch.barrier()
        # Drain the last (lagging) WGMMA and release its stage.
        if 0 < k_tile_cnt:
            warpgroup.wait_group(0)
            ab_pipeline.consumer_release(ab_release_state)
            ab_release_state.advance()

        # ---- epilogue: out[m, d'] = (acc + proj_b[d']) [* gate3[m, d']] ----
        gOut = cute.local_tile(mOut, (blk_m, blk_n), (bm, bn))
        tCgO = thr_mma_c.partition_C(gOut)
        cId = cute.make_identity_tensor((blk_m, blk_n))
        tCcO = thr_mma_c.partition_C(cId)
        # gate3 is (M, D) row-major with the SAME (token, feature) layout as the output, so its
        # (blk_m, blk_n) tile at (bm, bn) partitions exactly like the output.  It has exactly D
        # columns (no output padding), so its read is guarded by the SAME dprime<D predicate the
        # write uses.  const_expr-gated -> a no-op when the gate is off.
        if const_expr(has_gate3):
            gGate3 = cute.local_tile(mGate3, (blk_m, blk_n), (bm, bn))
            tCgGate3 = thr_mma_c.partition_C(gGate3)
        for e in cutlass.range_constexpr(cute.size(acc)):
            dprime = bn * blk_n + tCcO[e][1]
            # Padded output columns (dprime >= D) have acc = 0 (the WGMMA saw zero-padded B rows)
            # and s_proj_b[dprime] = 0, so val = 0.  The output is padded to a blk_n multiple, so
            # every tile write is in bounds; the caller slices the padded columns off.
            val = acc[e] + s_proj_b[dprime]
            if const_expr(has_gate3):
                # Read the gate only for in-bounds output columns; a padded column writes 0
                # regardless, so skipping its read avoids an out-of-bounds load.
                if dprime < D:
                    val = val * tCgGate3[e].to(Float32)
            tCgO[e] = val.to(mOut.element_type)

    @cute.jit
    def _normalize_tile(self, sA_stage, s_stats, s_normw, s_normb, thr_copy, k_tile):
        """Rewrite one staged ``(blk_m, blk_k)`` A tile in place as its LayerNorm.

        Semantics:
            ``sA[m, k] = (sA[m, k].f32 - mu_m) * rstd_m * norm_w[g_k] + norm_b[g_k]``, stored back
            in ``self.dtype``. ``g_k = k_tile * blk_k + local_k`` is the GLOBAL feature index, which
            is why ``k_tile`` is a parameter and not derived: a caller whose stage order does not
            match its global k order would otherwise apply the wrong slice of the gain, silently.
            ``norm_b`` is the LayerNorm bias and is ALWAYS applied; the projection bias is added in
            the epilogue instead.

            In place through registers, so the tile is DESTROYED: normalizing the same stage twice
            would apply the affine twice, and the caller must fence and barrier before the WGMMA
            reads it.

            For a ``D`` that is a multiple of 8 but not 16 the affine vectors are zero-padded to
            ``blk_k * k_tile_cnt``, so a padded ``k`` gives ``norm_w[k] = norm_b[k] = 0`` and hence
            ``(0 - mu) * rstd * 0 + 0 = 0`` -- no contribution and no runtime predicate.

        Args:
            sA_stage: The ``(blk_m, blk_k)`` staged tile, OVERWRITTEN.
            s_stats: The ``(blk_m, 2)`` per-token ``(mean, rstd)`` scratch for this tile's rows.
            s_normw: The zero-padded fp32 gain, covering at least ``(k_tile + 1) * blk_k`` columns.
            s_normb: The zero-padded fp32 bias, same extent requirement.
            thr_copy: This thread's slice of the normalize tiled copy.
            k_tile: The GLOBAL k-tile index this stage holds, as a runtime ``Int32``.

        Returns:
            None. Its effect is ``sA_stage``.
        """
        blk_m, blk_k = self.blk_m, self.blk_k
        cA = cute.make_identity_tensor((blk_m, blk_k))
        tAsA = thr_copy.partition_S(sA_stage)  # (CPY, CPY_M, CPY_N)
        tAcA = thr_copy.partition_S(cA)
        tArA = cute.make_rmem_tensor_like(tAsA)
        cute.autovec_copy(tAsA, tArA)
        for i in cutlass.range_constexpr(cute.size(tArA)):
            m = tAcA[i][0]
            k = k_tile * blk_k + tAcA[i][1]
            mu = s_stats[m, 0]
            rstd = s_stats[m, 1]
            val = (tArA[i].to(Float32) - mu) * rstd * s_normw[k] + s_normb[k]
            tArA[i] = val.to(self.dtype)
        cute.autovec_copy(tArA, tAsA)


def _run_token_pair_gemm(
    a: Tensor,  # (B, D, N, N)
    b: Tensor,  # (B, D, N, N)
    blk_i: int,
    blk_j: int,
    blk_d: int,
    blk_k: int,
    cont_pipe: bool = False,
    direction: str = "outgoing",
) -> Tuple[Tensor, Tensor]:
    """Compile and launch the token-pair einsum, returning ``tri`` and its partial statistics.

    Semantics:
        Computes, for the outgoing direction,
        ``tri[b, i, j, d] = sum_k a[b, d, i, k] * b[b, d, j, k]`` (contracting the CONTIGUOUS token
        axis), or for the incoming one
        ``tri[b, i, j, d] = sum_k a[b, d, k, i] * b[b, d, k, j]`` (contracting the STRIDED one),
        stored ``(B, N, N, D)`` with ``D`` contiguous. ``stats_partial`` holds each feature-chunk
        CTA's partial ``(sum x, sum x^2)`` per token.

        The incoming direction is NOT a second kernel: ``incoming(a, b) == outgoing(a^T, b^T)`` over
        the two token axes, so the operands are fed as a transposed STRIDED VIEW (``permute(2,1,0)``
        instead of ``(1,2,0)``). The contraction axis becomes the strided token and the output token
        becomes the contiguous operand axis, which flips both operands from k-major to mn-major. The
        kernel is already generic over the major mode, so this costs no kernel change and no extra
        pass over memory -- only the host-side view and a compile-cache key entry.

        The compiled callable is memoized in :data:`_COMPILE_CACHE`; without that memo every call
        recompiles and a benchmark loop measures the compiler.

    Args:
        a: The ``i``-side activation, ``(B, D, N, N)``, on CUDA. Made contiguous here; a
            non-contiguous input costs a copy but is not wrong.
        b: The ``j``-side activation, same shape and dtype as ``a``.
        blk_i, blk_j, blk_d, blk_k: The kernel's tile parameters; see
            :class:`TokenPairGemmStatsSm90` for each one's constraints.
        cont_pipe: Select the continuous-pipeline variant. A PURE scheduling choice -- both values
            compute identical numbers, so a wrong pick costs speed and never correctness.
        direction: ``"outgoing"`` or ``"incoming"``. Anything else is refused, because silently
            defaulting would contract the wrong axis and produce a plausible wrong answer.

    Returns:
        ``(tri, stats_partial)`` -- ``(B, N, N, D)`` in ``a``'s dtype and
        ``(D // blk_d, B*N*N, 2)`` fp32.

    Raises:
        AssertionError: If ``direction`` is not one of the two, if ``a`` and ``b`` disagree in
            shape, if the tensors are not 4-D with a square token block, or if they are not on CUDA.
    """
    assert direction in ("outgoing", "incoming")
    assert a.shape == b.shape and a.dim() == 4 and a.shape[0] >= 1
    B, D, N, N2 = a.shape
    assert N == N2
    assert a.is_cuda and b.is_cuda
    dtype = torch2cute_dtype_map[a.dtype]
    a = a.contiguous()
    b = b.contiguous()
    # tri is (B, N, N, D) (token axis B·N², D contiguous per token); the statistics workspace shares
    # that token axis.  n_partials = D/blk_d stays PER-BATCH -- each token carries only its own
    # batch's D features, so B·D/blk_d rows would sum another batch into this token's mean.
    tri = torch.empty(B, N, N, D, device=a.device, dtype=a.dtype)
    n_partials = D // blk_d
    stats_partial = torch.empty(n_partials, B * N * N, 2, device=a.device, dtype=torch.float32)

    # a is (B, D, N, N) row-major: a[b,d,p,q] with q at stride 1, p at N, d at N², b at D·N².
    # Flatten (B,D) -> a single L = B·D axis (stride N²) so the operand is (M=i, K=k, L=b·D+d).
    # outgoing: view (M=i, K=k, L) = permute (1,2,0) -> strides (N,1,N²), k-major.
    # incoming: view = permute (2,1,0) -> strides (1,N,N²), mn-major (i at stride 1; i=q, k=p).
    perm = (1, 2, 0) if direction == "outgoing" else (2, 1, 0)
    a3 = a.reshape(B * D, N, N).permute(*perm)  # (i=N, k=N, L=B·D)
    b3 = b.reshape(B * D, N, N).permute(*perm)  # (j=N, k=N, L=B·D)
    cls = TokenPairGemmStatsContPipeSm90 if cont_pipe else TokenPairGemmStatsSm90
    key = (
        "tokenpair_contpipe" if cont_pipe else "tokenpair",
        a.dtype,
        B,
        N,
        D,
        blk_i,
        blk_j,
        blk_d,
        blk_k,
        direction,
    )
    compiled = _COMPILE_CACHE.get(key)
    if compiled is None:
        op = cls(dtype, N, D, blk_i, blk_j, blk_d, blk_k, B=B)
        # The stride-1 operand axis in the FINAL (i, k, L) shape is k (axis 1) for outgoing and
        # i (axis 0) for incoming.  The L axis (B·D) carries stride N², divisibility 8.
        op_ld = 1 if direction == "outgoing" else 0
        fA = fake_tensor(dtype, (N, N, B * D), divisibility=8, leading_dim=op_ld)
        fB = fake_tensor(dtype, (N, N, B * D), divisibility=8, leading_dim=op_ld)
        fTri = fake_tensor(dtype, (B, N, N, D), divisibility=8, leading_dim=3)  # D contiguous
        fStats = fake_tensor(Float32, (n_partials, B * N * N, 2), divisibility=2, leading_dim=2)
        compiled = cute.compile(op, fA, fB, fTri, fStats, _fake_stream(), **_TVM_FFI)
        _COMPILE_CACHE[key] = compiled
    compiled(a3, b3, tri, stats_partial)
    return tri, stats_partial


def _run_row_stats_reduce(stats_partial: Tensor, D: int, eps: float) -> Tensor:
    """Compile and launch the statistics reduction, turning the partials into ``(mean, rstd)``.

    Args:
        stats_partial: The ``(n_partials, n_tokens, 2)`` fp32 workspace the token-pair kernel filled.
            Must be fully written; an uninitialised row poisons every token's mean.
        D: The feature count, used as the reduction divisor. Must be the true feature extent, since
            the kernel divides by it rather than by ``n_partials * blk_d``.
        eps: The variance epsilon. Must be positive, or ``rstd`` can be infinite or NaN.

    Returns:
        A ``(n_tokens, 2)`` fp32 tensor holding ``(mean, rstd)`` per token.
    """
    n_partials, n_tokens, _ = stats_partial.shape
    stats = torch.empty(n_tokens, 2, device=stats_partial.device, dtype=torch.float32)

    key = ("rowstats", n_tokens, n_partials, D)
    compiled = _COMPILE_CACHE.get(key)
    if compiled is None:
        op = RowStatsReduceSm90(n_tokens, n_partials, D)
        fStatsPartial = fake_tensor(
            Float32, (n_partials, n_tokens, 2), divisibility=2, leading_dim=2
        )
        fStats = fake_tensor(Float32, (n_tokens, 2), divisibility=2, leading_dim=1)
        compiled = cute.compile(op, fStatsPartial, fStats, Float32(0.0), _fake_stream(), **_TVM_FFI)
        _COMPILE_CACHE[key] = compiled
    compiled(stats_partial, stats, float(eps))
    return stats


def _run_feature_gemm(
    tri: Tensor,  # (B, N, N, D)
    stats: Tensor,  # (M, 2) fp32
    norm_w: Tensor,  # (D,)
    norm_b: Optional[Tensor],  # (D,) or None
    proj_w: Tensor,  # (D, D)
    proj_b: Optional[Tensor],  # (D,) or None
    gate3: Optional[Tensor],  # (M, D) or None
    eps: float,
    blk_m: int,
    blk_n: int,
    blk_kf: int,
) -> Tensor:
    """Compile and launch the normalize-on-load projection, returning the ``(M, D)`` output.

    Semantics:
        Views ``tri`` as ``(M = B*N*N, D)`` -- valid without a copy because the token-pair kernel
        already stored it ``D``-contiguous, which is the whole point of that kernel. The affine
        vectors and the projection bias are promoted to fp32 (the statistics and the LayerNorm
        affine stay fp32 even for a 16-bit activation), while ``proj_w`` must ALREADY be the
        activation dtype: casting it here would silently repeat a conversion the caller should do
        once, outside the loop.

        The output is over-allocated to the next ``blk_n`` multiple so the last N-tile's TMA address
        stays inside the allocation; the padded columns are never written with anything but zero and
        are sliced off before returning. When ``D % blk_n == 0`` the allocation is exactly ``D`` wide
        and no slice happens.

    Args:
        tri: The ``(B, N, N, D)`` einsum output, ``D`` contiguous.
        stats: The ``(M, 2)`` fp32 ``(mean, rstd)`` per token.
        norm_w: The ``(D,)`` LayerNorm gain, any float dtype (promoted to fp32 here).
        norm_b: The ``(D,)`` LayerNorm bias, any float dtype, or **None** to omit it. None compiles
            a build that stages a zero in its place, so the normalize adds the additive identity and
            no arithmetic changes -- and it is a SEPARATE compiled kernel, keyed as such below.
        proj_w: The ``(D, D)`` projection weight ``(d', d)``. Must already be ``tri``'s dtype.
        proj_b: The ``(D,)`` projection bias, or None to omit it.
        gate3: The ``(M, D)`` output gate, or None. Reshaped and cast to the activation dtype.
        eps: Passed through for signature symmetry; ``rstd`` is already computed.
        blk_m, blk_n, blk_kf: Tile parameters; see :class:`FeatureGemmPrologLnSm90`.

    Returns:
        A ``(M, D)`` tensor in ``tri``'s dtype.
    """
    B, N, _, D = tri.shape
    M = B * N * N
    dtype = torch2cute_dtype_map[tri.dtype]

    tri_2d = tri.reshape(M, D)  # (M, D), d contiguous -> k-major
    proj_w_c = proj_w.contiguous()  # (D, D) = (d', d), d contiguous -> k-major
    normw = norm_w.to(torch.float32).contiguous()
    normb = norm_b.to(torch.float32).contiguous() if norm_b is not None else None
    proj_bf = proj_b.to(torch.float32).contiguous() if proj_b is not None else None
    D_out = ((D + blk_n - 1) // blk_n) * blk_n
    out = torch.empty(M, D_out, device=tri.device, dtype=tri.dtype)

    has_gate3 = gate3 is not None
    gate3_2d = gate3.reshape(M, D).to(tri.dtype).contiguous() if has_gate3 else None
    has_bias = proj_bf is not None
    has_norm_bias = normb is not None

    # `has_norm_bias` IS PART OF THE KEY because it changes the generated kernel. When
    # `has_norm_bias=False` prunes the read, the two variants cannot share one cache entry.
    #
    # The measured failure mode if this term were dropped is a LOUD `TypeError` at the tvm-ffi
    # boundary in both orders (`762c723`), not a silent wrong answer.
    # `test_norm_bias_variants_do_not_share_a_compiled_kernel` is what fails if it is dropped again.
    key = ("featuregemm", tri.dtype, M, D, blk_m, blk_n, blk_kf, has_bias, has_gate3, has_norm_bias)
    compiled = _COMPILE_CACHE.get(key)
    if compiled is None:
        op = FeatureGemmPrologLnSm90(
            dtype,
            M,
            D,
            blk_m,
            blk_n,
            blk_kf,
            has_bias=has_bias,
            has_gate3=has_gate3,
            has_norm_bias=has_norm_bias,
        )
        fTri = fake_tensor(dtype, (M, D), divisibility=8, leading_dim=1)  # D(=k) contiguous
        fProjW = fake_tensor(dtype, (D, D), divisibility=8, leading_dim=1)
        fStats = fake_tensor(Float32, (M, 2), divisibility=2, leading_dim=1)
        fNormW = fake_tensor(Float32, (D,), divisibility=4, leading_dim=0)
        fNormB = (
            fake_tensor(Float32, (D,), divisibility=4, leading_dim=0) if has_norm_bias else None
        )
        fProjB = fake_tensor(Float32, (D,), divisibility=4, leading_dim=0) if has_bias else None
        fOut = fake_tensor(dtype, (M, D_out), divisibility=8, leading_dim=1)
        fGate3 = fake_tensor(dtype, (M, D), divisibility=8, leading_dim=1) if has_gate3 else None
        compiled = cute.compile(
            op,
            fTri,
            fProjW,
            fStats,
            fNormW,
            fNormB,
            fProjB,
            fOut,
            Float32(0.0),
            _fake_stream(),
            fGate3,
            **_TVM_FFI,
        )
        _COMPILE_CACHE[key] = compiled
    compiled(tri_2d, proj_w_c, stats, normw, normb, proj_bf, out, float(eps), gate3_2d)
    # Slice off the padding columns when D was rounded up; they were never written non-zero.
    return out if D_out == D else out[:, :D].contiguous()


class GemmLayerNormGemmSm90:
    """The whole chain as ONE resolved plan: gates checked, tiles picked, three kernels launched.

    Purpose:
        Put the front door's validation and its tile arithmetic in one place, so that "which tile
        does D=264 get" is answerable by reading one ``__init__`` rather than by tracing three
        launchers.

    Functionality and semantics:
        Construction is pure host-side arithmetic -- it allocates nothing and touches no device. It
        validates the two shape gates and resolves any tile left as ``None`` or left at a value that
        does not fit ``D``. Calling the instance runs the three kernels back to back on the ambient
        stream, which is what orders them: each reads the previous one's output.

        The tile resolution is what lets ``D`` need only ``D % 8 == 0``:

        * ``blk_n`` (the output-feature tile) must be a multiple of 16 for the shared-memory swizzle
          atom. For ``D % 16 == 0`` this takes the largest such divisor of ``D`` at most 256; for a
          ``D`` that is 8 mod 16 no such divisor exists, so it falls to 16 with a ceil-div N-tile
          count and a predicated output write.
        * ``blk_kf`` (the feature contraction tile) additionally needs ``num_threads`` divisible by
          ``blk_kf // vecsize``, which restricts it to {64, 32, 16}. A ``D`` that is 8 mod 16 gets 16
          with a partial last tile -- 8 real columns plus 8 TMA-zero-padded ones.
        * ``blk_m`` drops from 128 to 64 when ``M`` is not a multiple of 128. ``M = B*N^2`` is always
          a multiple of 64 given ``N % 8 == 0``, so 64 always fits.

        For ``D % 16 == 0`` every one of those gates folds away at compile time and the generated
        code is what an implementation with no padding support would emit.

    Args:
        a_dtype: The activation torch dtype. Must be a 16-bit float; the SM90 WGMMA atom this chain
            builds has no fp32 form, so an fp32 activation fails at MMA construction rather than at
            the front door.
        B: Batch. Any positive int.
        N: The token-axis extent. Must satisfy ``N % 8 == 0``.
        D: The feature extent. Must satisfy ``D % 8 == 0``.
        direction: ``"outgoing"`` or ``"incoming"``.
        cont_pipe: Which token-pair schedule to run. A pure perf knob.
        has_bias: Whether a projection bias will be supplied to ``__call__``.
        has_gate3: Whether an output gate will be supplied to ``__call__``.
        blk_i, blk_j, blk_d, blk_k: Token-pair tile parameters.
        blk_m: Token tile of the projection; lowered to 64 when ``M % 128 != 0``.
        blk_n: Output-feature tile, or None to resolve it from ``D``.
        blk_kf: Feature contraction tile; re-resolved when it does not divide ``D``.

    Raises:
        ValueError: If ``direction`` is not one of the two, or if ``N`` or ``D`` breaches its
            multiple-of-8 gate. Both are 16-byte alignment floors for a 16-bit dtype, and both are
            checked HERE rather than left to an assert deeper in, because a stripped assert (under
            ``python -O``) turns a clean refusal into a TMA fault or a wrong answer.
    """

    def __init__(
        self,
        a_dtype: torch.dtype,
        B: int,
        N: int,
        D: int,
        *,
        direction: str = "outgoing",
        cont_pipe: bool = False,
        has_bias: bool = False,
        has_gate3: bool = False,
        blk_i: int = 64,
        blk_j: int = 64,
        blk_d: int = 8,
        blk_k: int = 64,
        blk_m: int = 128,
        blk_n: Optional[int] = None,
        blk_kf: int = 64,
    ):
        if direction not in ("outgoing", "incoming"):
            raise ValueError(
                f"direction must be 'outgoing' or 'incoming', got {direction!r}. There is no "
                f"default: the two contract different token axes, so guessing would return a "
                f"plausible wrong answer instead of an error."
            )
        if a_dtype not in (torch.bfloat16, torch.float16):
            raise ValueError(
                f"the activation dtype must be bfloat16 or float16, got {a_dtype}. Both GEMMs in "
                f"this chain are SM90 WGMMA, whose f16/bf16 MMA atom has no 32-bit form -- an fp32 "
                f"activation has no kernel at all, so this is a capability boundary and not a "
                f"precision preference. Without this check it fails several frames down inside MMA "
                f"construction, naming neither the argument nor the dtype."
            )
        if N % 8 != 0:
            raise ValueError(
                f"N must be a multiple of 8 (16-byte TMA alignment for a 16-bit dtype), got N={N}. "
                f"N is otherwise unconstrained -- it need not be a power of two or a tile multiple."
            )
        if D % 8 != 0:
            raise ValueError(
                f"D must be a multiple of 8 (16-byte alignment for a 16-bit dtype), got D={D}. "
                f"D % 16 takes the unpredicated path; D % 8-not-16 takes the padded one."
            )
        self.a_dtype = a_dtype
        self.B, self.N, self.D = B, N, D
        self.direction = direction
        self.cont_pipe = cont_pipe
        self.has_bias = has_bias
        self.has_gate3 = has_gate3
        self.M = B * N * N
        self.blk_i, self.blk_j, self.blk_d, self.blk_k = blk_i, blk_j, blk_d, blk_k
        if blk_n is None:
            # Largest %16 divisor of D at most 256; for D % 8-not-16 none exists -> 16 with a
            # ceil-div N-tile count and a predicated output write.
            hi = min(D, 256)
            hi -= hi % 16
            blk_n = next((t for t in range(hi, 0, -16) if D % t == 0), 16)
        if D % blk_kf != 0:
            # blk_kf also feeds the normalize-on-load copy, which needs
            # num_threads % (blk_kf // vecsize) == 0 -> restricted to {64, 32, 16}.
            blk_kf = next((t for t in (64, 32, 16) if D % t == 0), 16)
        if self.M % blk_m != 0:
            blk_m = 64
        self.blk_m, self.blk_n, self.blk_kf = blk_m, blk_n, blk_kf

    def __call__(
        self,
        a: Tensor,  # (B, D, N, N)
        b: Tensor,  # (B, D, N, N)
        norm_w: Tensor,  # (D,)
        norm_b: Optional[Tensor],  # (D,) or None
        proj_w: Tensor,  # (D, D)
        proj_b: Optional[Tensor] = None,  # (D,)
        gate3: Optional[Tensor] = None,  # (M, D) or anything reshapeable to it
        eps: float = 1e-5,
    ) -> Tensor:
        """Run the three kernels and return ``(B, N, N, D)``.

        Semantics:
            The token-pair kernel writes ``tri`` and the partial statistics, the reduction turns
            those into ``(mean, rstd)``, and the projection consumes both. All three go on the
            ambient CUDA stream in that order, which is what makes the dependencies safe.

        Args:
            a: The ``i``-side activation, ``(B, D, N, N)``, on CUDA. Its dtype must match the one
                this object was built for; a mismatch produces a compile-cache miss and then a type
                error at launch.
            b: The ``j``-side activation, same shape and dtype.
            norm_w: The ``(D,)`` LayerNorm gain. Any float dtype -- promoted to fp32.
            norm_b: The ``(D,)`` LayerNorm bias, likewise, or None to omit it -- the bias-free
                build stages a zero in its place, so the normalize adds the additive identity.
            proj_w: The ``(D, D)`` projection weight ``(d', d)``. Must ALREADY be ``a``'s dtype.
            proj_b: The ``(D,)`` projection bias, or None. Must be None exactly when this object was
                built with ``has_bias=False``.
            gate3: The output gate, ``(M, D)`` or any shape reshapeable to it, with the SAME
                (token, feature) ordering as the output. Must be None exactly when ``has_gate3`` is
                False. A gate whose token order differs (for instance ``(B, D, N, N)``) reshapes
                without error and multiplies the wrong element into every output.
            eps: The LayerNorm epsilon. Must be positive.

        Returns:
            The ``(B, N, N, D)`` output in ``a``'s dtype.

        Raises:
            ValueError: If ``proj_w``'s dtype is not ``a``'s -- the kernel does no internal cast,
                and letting it reach the launch turns a one-line caller fix into a DSL type error.
                Also if any per-feature vector has the wrong extent, or ``gate3`` the wrong element
                count: neither is checked by the kernels, so a short one is a wrong answer with no
                fault raised.
        """
        if proj_w.dtype != a.dtype:
            raise ValueError(
                f"proj_w must already be the activation dtype: got proj_w={proj_w.dtype}, "
                f"a={a.dtype}. There is no internal cast -- casting per call would repeat a "
                f"conversion the caller should do once, outside the loop."
            )
        # Extent checks on the per-feature vectors.  None of these is checked by the kernels: the
        # affine and the bias are staged by a loop bounded by D, so a SHORT vector is read past its
        # end and whatever follows it in memory becomes the gain of every row's tail -- a wrong
        # answer with no fault and no diagnostic.
        D = self.D
        for name, t, want in (
            ("norm_w", norm_w, (D,)),
            ("norm_b", norm_b, (D,)),
            ("proj_w", proj_w, (D, D)),
            ("proj_b", proj_b, (D,)),
        ):
            if t is not None and tuple(t.shape) != want:
                raise ValueError(
                    f"{name} must have shape {want} for D={D}, got {tuple(t.shape)}. It is staged "
                    f"by a loop bounded by D, so a shorter one is read past its end and scales the "
                    f"tail of every row with unrelated memory -- silently."
                )
        if gate3 is not None and gate3.numel() != self.M * D:
            raise ValueError(
                f"gate3 must hold M*D = {self.M * D} elements for M={self.M}, D={D}, got "
                f"{gate3.numel()}. It is reshaped to (M, D), so a different count either fails the "
                f"reshape or silently re-indexes the gate against the wrong tokens."
            )
        tri, stats_partial = _run_token_pair_gemm(
            a,
            b,
            self.blk_i,
            self.blk_j,
            self.blk_d,
            self.blk_k,
            cont_pipe=self.cont_pipe,
            direction=self.direction,
        )
        stats = _run_row_stats_reduce(stats_partial, self.D, eps)
        out = _run_feature_gemm(
            tri,
            stats,
            norm_w,
            norm_b,
            proj_w,
            proj_b,
            gate3,
            eps,
            self.blk_m,
            self.blk_n,
            self.blk_kf,
        )
        return out.reshape(self.B, self.N, self.N, self.D)


# ───────────────────── autotune-free size heuristic for the one perf knob ─────────────────────
# `cont_pipe` is the single perf lever here: the token-pair GEMM is occupancy-capped (a structural
# ceiling), so the tile parameters are architecture constants and only the pipeline schedule moves.
# A swept grid {128, 256, 512, 1024} plus a bisection at N in {640, 768, 896} showed a
# MECHANISTICALLY SHARP crossover between N=640 and N=768:
#   * N <= 640: cont_pipe=True  -- removing the per-feature ramp gives 0.92-0.99x across every swept
#               cell at or below 640 (thin +1% at N=640 itself).
#   * N >= 768: cont_pipe=False -- the continuous pipeline LOWERS the einsum's HMMA share from 16%
#               to 13% and loses by a wide 16-20% margin at N=768/896 and 16-20% at N=1024.
# The threshold is the largest swept winning N. The mispick risk is ASYMMETRIC and this is the safe
# side: picking False inside the True region costs at most 8%, picking True inside the False region
# costs 16-20%, so the formula flips the moment the win turns marginal. BOTH values are valid at any
# shape -- the knob is purely a scheduling choice with no size constraint -- so there is no validity
# gate and no fallback to arrange.
# The crossover was bisected on `heuristic_arch.TUNED_ARCH`, so it is arch-DEPENDENT and keyed here;
# a non-tuned arch warns once and falls back to the tuned set. Because the knob has no
# arch-independent validity component, there is no correctness layer to preserve underneath.
_CONT_PIPE_BY_ARCH = {
    # The continuous pipeline wins for N <= CONT_PIPE_MAX_N; the per-feature one wins above it.
    "H200_SXM5": {"CONT_PIPE_MAX_N": 640},
}


def _heuristic_cont_pipe(N: int, device=None) -> bool:
    """Resolve ``cont_pipe`` from the token-axis extent, with no timing and no memory query.

    Semantics:
        A pure size-to-knob formula, which is what makes the default path free: no benchmark, no
        cache lookup, no device query unless a ``device`` is given. Both values run at any ``N``, so
        this never has to fall back.

    Args:
        N: The token-axis extent. Any positive int; only its magnitude matters.
        device: A torch device (or None) selecting the arch whose bisected threshold to use. None
            means the tuned arch is assumed and NO warning is emitted -- pass a device on a machine
            that may not be the tuned one, so the mismatch is reported once instead of silently
            using another arch's crossover.

    Returns:
        True to run the continuous-pipeline token-pair kernel, False for the per-feature one.
    """
    arch = heuristic_arch(device) if device is not None else TUNED_ARCH
    if arch != TUNED_ARCH:
        warn_arch_suboptimal_once(arch, "gemm_layernorm_gemm")
    perf = _CONT_PIPE_BY_ARCH.get(arch, _CONT_PIPE_BY_ARCH[TUNED_ARCH])
    return N <= perf["CONT_PIPE_MAX_N"]


#: The declared tunable axes. One knob, two points -- the token-pair GEMM is occupancy-capped, so
#: the tiles are architecture constants and the schedule is the only thing worth measuring.
GEMM_LAYERNORM_GEMM_TUNING_SPACE = AxisSpace(
    TuneAxis(
        "cont_pipe",
        domain=(
            "True or False at ANY shape -- the knob picks between two token-pair GEMM schedules "
            "that compute identical numbers, so it carries no size constraint and needs no "
            "validity rule"
        ),
        values=(False, True),
    )
)


@autotune(
    space=GEMM_LAYERNORM_GEMM_TUNING_SPACE,
    key=["direction"],
    gate="do_autotune",
)
def gemm_layernorm_gemm(
    a: Tensor,  # (B, D, N, N)
    b: Tensor,  # (B, D, N, N)
    norm_w: Tensor,  # (D,)
    norm_b: Optional[Tensor],  # (D,) or None
    proj_w: Tensor,  # (D, D) = (d', d), MUST be a.dtype
    proj_b: Optional[Tensor] = None,  # (D,)
    *,
    direction: str = "outgoing",
    eps: float = 1e-5,
    gate3: Optional[Tensor] = None,  # (M = B*N*N, D) row-major, or None
    blk_i: int = 64,
    blk_j: int = 64,
    blk_d: int = 8,
    blk_k: int = 64,
    blk_m: int = 128,
    blk_n: Optional[int] = None,
    blk_kf: int = 64,
    cont_pipe: Optional[bool] = None,
    do_autotune: bool = False,
) -> Tensor:
    """The token-pair einsum, its LayerNorm, the output projection and the gate — ops 7, 8, 9, 12.

    Semantics:
        Three kernel launches on the ambient stream (see :class:`GemmLayerNormGemmSm90`), producing
        ``out = (LayerNorm(tri) @ proj_w^T + proj_b) * gate3`` where ``tri`` is the token-pair
        einsum of ``a`` and ``b``.

        Batch ``B >= 1``: the token axis is ``M = B*N^2``. Only the einsum is batch-bound -- the
        contiguous ``(B, D, N, N)`` layout flattens ``(B, D)`` onto one L axis, so grid-z becomes
        ``B*(D/blk_d)`` and a CTA derives its batch and feature chunk from the z group. The other two
        kernels are already per-token over ``M``, and their shared parameters are identical across
        the batch. ``B == 1`` is byte-identical to an unbatched build.

        ``cont_pipe`` left at None resolves through :func:`_heuristic_cont_pipe` -- a size formula
        with no timing. Passing it explicitly overrides that. Passing ``do_autotune=True`` measures
        both values for this shape and functional variant instead; do NOT pass ``cont_pipe`` at the
        same time, since the measured winner and the executed kernel would then differ.

    Args:
        a: The ``i``-side activation, ``(B, D, N, N)``, 16-bit float on CUDA. ``N % 8 == 0`` and
            ``D % 8 == 0`` are the only shape constraints; both are 16-byte alignment floors and
            both are refused at this front door.
        b: The ``j``-side activation, same shape and dtype as ``a``. A different shape is refused.
        norm_w: The ``(D,)`` LayerNorm gain. Any float dtype -- promoted to fp32 internally, since
            the affine is applied in fp32 regardless of the activation's width.
        norm_b: The ``(D,)`` LayerNorm bias, likewise, or **None**. None is a genuinely different
            compiled kernel, not a runtime branch: it stages a zero into the affine scratch so the
            normalize's ``+ norm_b[k]`` becomes ``+ 0.0``, which is the additive identity and
            leaves the arithmetic untouched.
        proj_w: The ``(D, D)`` output-projection weight ``(d', d)``. **Must already be ``a``'s
            dtype**; an fp32 weight against a bf16 activation is refused rather than cast, so the
            conversion happens once at the caller instead of on every call.
        proj_b: The ``(D,)`` projection bias, or None. Any float dtype; promoted to fp32.
        direction: ``"outgoing"`` (contract the contiguous token axis) or ``"incoming"`` (contract
            the strided one). Anything else raises.
        eps: The LayerNorm epsilon, added to the variance before the reciprocal square root. Must be
            positive.
        gate3: The elementwise output gate, ``(M, D)`` or any shape reshapeable to it, with the SAME
            (token, feature) ordering as the output. None compiles the gate out entirely, giving a
            build identical to one that never had the feature. A gate in a different token order
            reshapes silently and multiplies the wrong element into every output.
        blk_i, blk_j, blk_d, blk_k: Token-pair GEMM tiles. Defaults are the tuned architecture
            constants; see :class:`TokenPairGemmStatsSm90` for the constraints on each.
        blk_m, blk_n, blk_kf: Projection tiles. ``blk_n=None`` and a ``blk_kf`` that does not divide
            ``D`` are resolved automatically; see :class:`GemmLayerNormGemmSm90`.
        cont_pipe: The token-pair schedule. None resolves it from ``N`` by the size heuristic. Both
            values compute identical numbers, so a wrong value costs speed and never correctness.
        do_autotune: Measure ``cont_pipe`` for this shape instead of resolving it. Keyword-only, and
            the gate the ``@autotune`` decorator reads.

    Returns:
        A ``(B, N, N, D)`` tensor in ``a``'s dtype.

    Raises:
        ValueError: For every caller mistake this front door can see -- ``a`` not 4-D, ``a`` and
            ``b`` disagreeing in shape, a non-square token block, ``direction`` not one of the two,
            ``N % 8 != 0`` or ``D % 8 != 0``, ``proj_w`` not in ``a``'s dtype, a per-feature vector
            of the wrong extent, or a ``gate3`` with the wrong element count. All of them are
            checked HERE rather than left to an assert deeper in, because ``python -O`` strips
            asserts and every one of these then becomes a TMA fault or a silently wrong answer.
        AssertionError: If ``a`` or ``b`` is not on CUDA.
    """
    if a.dim() != 4:
        raise ValueError(
            f"a must be 4-D (B, D, N, N), got shape {tuple(a.shape)}. The four axes are read "
            f"positionally to derive the batch, the feature extent and the token extent."
        )
    if tuple(b.shape) != tuple(a.shape):
        raise ValueError(
            f"a and b must have the same shape, got {tuple(a.shape)} and "
            f"{tuple(b.shape)}: they are the two operands of one square einsum."
        )
    # The dtype half of the same comparison. `a`'s dtype is validated on its own further down (the
    # WGMMA atom has no 32-bit form), and the SHAPE of the pair is validated one line above -- but
    # nothing compared their DTYPES, so a `b` of a different element type passed both and reached
    # the TVM-FFI ABI check as `Mismatched Tensor on argument #1 when calling: __call__(mA: ...`,
    # naming an FFI slot rather than the argument the caller passed. Kept adjacent to the shape
    # comparison on purpose: they are one question about one pair, and separating them is how the
    # second went missing.
    if b.dtype is not a.dtype:
        raise ValueError(
            f"a and b must share an element type, got a={a.dtype} and b={b.dtype}. They are the "
            f"two operands of one WGMMA, whose atom takes a single element type -- there is no "
            f"mixed-input form to fall back to."
        )
    B, D, N, N2 = a.shape
    if N != N2:
        raise ValueError(
            f"a's last two axes are the two token axes of a SQUARE einsum and must match, got "
            f"({N}, {N2})."
        )
    if cont_pipe is None:
        cont_pipe = _heuristic_cont_pipe(N, device=a.device)
    plan = GemmLayerNormGemmSm90(
        a.dtype,
        B,
        N,
        D,
        direction=direction,
        cont_pipe=cont_pipe,
        has_bias=proj_b is not None,
        has_gate3=gate3 is not None,
        blk_i=blk_i,
        blk_j=blk_j,
        blk_d=blk_d,
        blk_k=blk_k,
        blk_m=blk_m,
        blk_n=blk_n,
        blk_kf=blk_kf,
    )
    return plan(a, b, norm_w, norm_b, proj_w, proj_b, gate3, eps)


def gemm_layernorm_gemm_ref(
    a: Tensor,
    b: Tensor,
    norm_w: Tensor,
    norm_b: Optional[Tensor],
    proj_w: Tensor,
    proj_b: Optional[Tensor] = None,
    direction: str = "outgoing",
    eps: float = 1e-5,
    gate3: Optional[Tensor] = None,
) -> Tensor:
    """The fp32 oracle for :func:`gemm_layernorm_gemm` — ops 7, 8, 9, 12 in plain torch.

    Semantics:
        Every stage runs in fp32 with no intermediate narrowing, which is the point: the kernel
        rounds ``tri`` to the activation dtype before the projection reads it, so the two agree only
        to the activation's precision. Comparing against a bf16 torch chain instead would hide
        exactly the error this is meant to bound.

    Args:
        a: The ``i``-side activation, ``(B, D, N, N)``.
        b: The ``j``-side activation, same shape.
        norm_w: The ``(D,)`` LayerNorm gain.
        norm_b: The ``(D,)`` LayerNorm bias, or None -- passed straight to ``F.layer_norm``,
            which omits the shift for None exactly as the kernel's zero-staged build does.
        proj_w: The ``(D, D)`` projection weight. Unlike the kernel this accepts any float dtype,
            since it casts to fp32.
        proj_b: The ``(D,)`` projection bias, or None.
        direction: ``"outgoing"`` or ``"incoming"``.
        eps: The LayerNorm epsilon.
        gate3: The ``(M, D)`` gate, or None.

    Returns:
        A ``(B, N, N, D)`` fp32 tensor.
    """
    B, D, N, _ = a.shape
    if direction == "outgoing":  # tri[b,d,i,j] = sum_k a[b,d,i,k] * b[b,d,j,k]
        tri = torch.einsum("bdik,bdjk->bdij", a.float(), b.float())
    else:  # incoming: tri[b,d,i,j] = sum_k a[b,d,k,i] * b[b,d,k,j]
        tri = torch.einsum("bdki,bdkj->bdij", a.float(), b.float())
    tri_2d = tri.permute(0, 2, 3, 1).reshape(B * N * N, D)  # (M, D)
    ln = F.layer_norm(
        tri_2d, (D,), norm_w.float(), None if norm_b is None else norm_b.float(), eps=eps
    )
    out = F.linear(ln, proj_w.float(), proj_b.float() if proj_b is not None else None)
    if gate3 is not None:
        out = out * gate3.reshape(B * N * N, D).float()
    return out.reshape(B, N, N, D)


def _token_pair_gemm_stats(
    a: Tensor,
    b: Tensor,
    eps: float = 1e-5,
    blk_i: int = 64,
    blk_j: int = 64,
    blk_d: int = 8,
    blk_k: int = 64,
    cont_pipe: bool = False,
    direction: str = "outgoing",
) -> Tuple[Tensor, Tensor, Tensor]:
    """Run only the first two kernels — the einsum and the statistics — for rung-level testing.

    Purpose:
        Isolate ops 7 and 8's statistics from the projection, so a numerical failure can be
        attributed to one stage rather than to the whole chain.

    Args:
        a: The ``i``-side activation, ``(B, D, N, N)``.
        b: The ``j``-side activation, same shape.
        eps: The variance epsilon.
        blk_i, blk_j, blk_d, blk_k: Token-pair tiles; see :class:`TokenPairGemmStatsSm90`.
        cont_pipe: Which token-pair schedule to run.
        direction: ``"outgoing"`` or ``"incoming"``.

    Returns:
        ``(tri, mean, rstd)`` -- ``(B, N, N, D)`` in ``a``'s dtype, and two ``(B, N, N)`` fp32
        tensors. Note ``tri`` is the NARROWED value the kernel stored, whereas the statistics were
        accumulated from the fp32 accumulator, so recomputing the statistics from ``tri`` does not
        reproduce them exactly. That difference is correct, not a defect.
    """
    B, D, N, _ = a.shape
    tri, stats_partial = _run_token_pair_gemm(
        a, b, blk_i, blk_j, blk_d, blk_k, cont_pipe=cont_pipe, direction=direction
    )
    stats = _run_row_stats_reduce(stats_partial, D, eps)
    mean = stats[:, 0].reshape(B, N, N)
    rstd = stats[:, 1].reshape(B, N, N)
    return tri, mean, rstd
