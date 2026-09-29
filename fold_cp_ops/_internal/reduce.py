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

# Copyright (c) 2025, Tri Dao.
# This file has been modified by NVIDIA CORPORATION & AFFILIATES.

"""Row/block/cluster reductions (CuTe-DSL).

Trimmed to the ``row_reduce`` chain the LayerNorm kernel needs. ``online_softmax_reduce`` and
``sum_swap_shuffle`` (the softmax/cross-entropy path) are dropped until that kernel returns.
"""

import operator
from typing import Callable, Optional

import cutlass
import cutlass.cute as cute
from cutlass import Int32, Float32, const_expr

import fold_cp_ops._internal.utils as utils


@cute.jit
def block_reduce(
    val: cute.Numeric, op: Callable, reduction_buffer: cute.Tensor, init_val: cute.Numeric = 0.0
) -> cute.Numeric:
    """Reduce ``val`` across all warps of one block, via a shared-memory staging buffer.

    Two-step: each warp's lane 0 writes its partial to the buffer, then after a block barrier
    the first ``warps_per_row`` lanes read the partials back and a single warp shuffle finishes
    the reduction. Rows are independent -- a warp only ever combines with warps in its own row.

    Args:
        val: This warp's already warp-reduced partial.
        op: Binary combiner, e.g. ``operator.add``. Must be associative and commutative.
        reduction_buffer: Shared-memory staging tensor of shape
            ``(num_warps // warps_per_row, warps_per_row)``.
        init_val: Identity for ``op``; returned by lanes that hold no partial.

    Returns:
        The block-wide reduction, valid in every lane of the first ``warps_per_row`` lanes.
    """
    lane_idx, warp_idx = cute.arch.lane_idx(), cute.arch.warp_idx()
    warps_per_row = cute.size(reduction_buffer.shape[1])
    row_idx, col_idx = warp_idx // warps_per_row, warp_idx % warps_per_row
    if lane_idx == 0:
        reduction_buffer[row_idx, col_idx] = val
    cute.arch.barrier()
    block_reduce_val = init_val
    if lane_idx < warps_per_row:
        block_reduce_val = reduction_buffer[row_idx, lane_idx]
    return cute.arch.warp_reduction(block_reduce_val, op)


@cute.jit
def cluster_reduce(
    val: cute.Numeric,
    op: Callable,
    reduction_buffer: cute.Tensor,
    mbar_ptr: cute.Pointer,
    init_val: cute.Numeric = 0.0,
    phase: Optional[Int32] = None,
) -> cute.Numeric:
    """Reduce ``val`` across all warps of every CTA in the cluster, via distributed SMEM.

    The cluster analogue of :func:`block_reduce`. Each CTA pushes its partial into all
    ``cluster_n`` peers with ``utils.store_shared_remote`` and signals their mbarrier; after
    waiting, every CTA holds all partials locally and finishes with a warp shuffle. Warp 0
    elects one lane to arm the expected transaction count before any store is issued -- doing
    it after would race the peers' arrivals.

    Args:
        val: This warp's already warp-reduced partial.
        op: Binary combiner. Must be associative and commutative.
        reduction_buffer: Shared-memory tensor of shape
            ``(num_warps // warps_per_row, (warps_per_row, cluster_n))``.
        mbar_ptr: Local mbarrier pointer; remapped per peer inside ``store_shared_remote``.
        init_val: Identity for ``op``.
        phase: mbarrier phase to wait on; defaults to 0. Pass an explicit phase when the same
            barrier is reused across reduction stages.

    Returns:
        The cluster-wide reduction, valid in every participating lane.
    """
    cta_rank_in_cluster = cute.arch.block_idx_in_cluster()
    lane_idx, warp_idx = cute.arch.lane_idx(), cute.arch.warp_idx()
    rows_per_block, (warps_per_row, cluster_n) = reduction_buffer.shape
    row_idx, col_idx = warp_idx // warps_per_row, warp_idx % warps_per_row
    if warp_idx == 0:
        with cute.arch.elect_one():
            num_warps = rows_per_block * warps_per_row
            cute.arch.mbarrier_arrive_and_expect_tx(
                mbar_ptr,
                num_warps * cluster_n * reduction_buffer.element_type.width // 8,
            )
    if lane_idx < cluster_n:
        utils.store_shared_remote(
            val,
            utils.elem_pointer(reduction_buffer, (row_idx, (col_idx, cta_rank_in_cluster))),
            mbar_ptr,
            peer_cta_rank_in_cluster=lane_idx,
        )
    cute.arch.mbarrier_wait(mbar_ptr, phase=phase if phase is not None else 0)
    block_reduce_val = init_val
    num_iter = cute.ceil_div(warps_per_row * cluster_n, cute.arch.WARP_SIZE)
    for i in cutlass.range_constexpr(num_iter):
        idx = lane_idx + i * cute.arch.WARP_SIZE
        if idx < cute.size(reduction_buffer, mode=[1]):
            block_reduce_val = op(block_reduce_val, reduction_buffer[row_idx, idx])
    return cute.arch.warp_reduction(block_reduce_val, op)


@cute.jit
def block_or_cluster_reduce(
    val: cute.Numeric,
    op: Callable,
    reduction_buffer: cute.Tensor,
    mbar_ptr: Optional[cute.Pointer],
    phase: Optional[Int32] = None,
    init_val: cute.Numeric = 0.0,
) -> cute.Numeric:
    """Dispatch to :func:`block_reduce` or :func:`cluster_reduce` on ``mbar_ptr`` presence.

    The branch is ``const_expr``, so exactly one path is compiled -- a single-CTA build carries
    no cluster code at all, and ``utils.store_shared_remote`` is never reached when
    ``cluster_n == 1``.

    Args:
        val: This warp's already warp-reduced partial.
        op: Binary combiner. Must be associative and commutative.
        reduction_buffer: Shared-memory staging tensor.
        mbar_ptr: mbarrier pointer for the cluster path, or None to reduce within one block.
        phase: mbarrier phase for the cluster path.
        init_val: Identity for ``op``.

    Returns:
        The block-wide or cluster-wide reduction.
    """
    if const_expr(mbar_ptr is None):
        return block_reduce(val, op, reduction_buffer, init_val=init_val)
    else:
        return cluster_reduce(val, op, reduction_buffer, mbar_ptr, phase=phase, init_val=init_val)


@cute.jit
def row_reduce(
    x: cute.TensorSSA | cute.Numeric,
    op: cute.ReductionOp,
    threads_per_row: cutlass.Constexpr[int],
    reduction_buffer: Optional[cute.Tensor] = None,
    mbar_ptr: Optional[cute.Pointer] = None,
    phase: Optional[Int32] = None,
    init_val: cute.Numeric = 0.0,
    hook_fn: Optional[Callable] = None,
) -> cute.Numeric:
    """Reduce along a row, escalating through thread -> warp -> block/cluster as needed.

    Three tiers, each compiled in only when the shape calls for it:

    1. If ``x`` is a ``TensorSSA``, reduce its per-thread elements first.
    2. A butterfly warp shuffle over ``min(threads_per_row, WARP_SIZE)`` lanes. ``threads_per_row``
       must be a power of two -- the shuffle requires it.
    3. If ``reduction_buffer`` is given AND the row spans more than one warp or CTA, escalate to
       :func:`block_or_cluster_reduce`. A row that fits in one warp skips this entirely.

    Args:
        x: Per-thread values (``TensorSSA``) or an already-scalar partial.
        op: Reduction op; mapped to the matching warp combiner. ADD/MAX/MIN/MUL are supported.
        threads_per_row: Lanes cooperating on one row. Power of two.
        reduction_buffer: Shared-memory staging tensor of shape
            ``(num_warps // warps_per_row, (warps_per_row, cluster_n))``. None to stop after
            the warp shuffle.
        mbar_ptr: mbarrier pointer, required when ``cluster_n > 1``.
        phase: mbarrier phase to wait on.
        init_val: Identity for ``op``.
        hook_fn: Called between the warp shuffle and the block/cluster step. LayerNorm passes
            ``cluster_wait`` here so the cluster handshake overlaps the warp reduction instead
            of serializing after it.

    Returns:
        The row reduction, valid in every lane of the row.

    Raises:
        AssertionError: If ``cluster_n > 1`` but ``mbar_ptr`` is None.
    """
    if const_expr(isinstance(x, cute.TensorSSA)):
        val = x.reduce(op, init_val=init_val, reduction_profile=0)
    else:
        val = x
    warp_op = {
        cute.ReductionOp.ADD: operator.add,
        cute.ReductionOp.MAX: cute.arch.fmax if const_expr(x.dtype == Float32) else max,
        cute.ReductionOp.MIN: min,
        cute.ReductionOp.MUL: operator.mul,
    }[op]
    val = cute.arch.warp_reduction(
        val,
        warp_op,
        threads_in_group=min(threads_per_row, cute.arch.WARP_SIZE),
    )
    if const_expr(hook_fn is not None):
        hook_fn()
    if const_expr(reduction_buffer is not None):
        warps_per_row, cluster_n = reduction_buffer.shape[1]
        assert cluster_n == 1 or mbar_ptr is not None, (
            "mbar_ptr must be provided for cluster reduction"
        )
        if const_expr(warps_per_row > 1 or cluster_n > 1):
            val = block_or_cluster_reduce(
                val, warp_op, reduction_buffer, mbar_ptr, phase=phase, init_val=init_val
            )
    return val
