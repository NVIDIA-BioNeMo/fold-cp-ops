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

"""Low-level DSL primitives (CuTe-DSL) that have no better home.

Everything here emits real instructions — inline PTX, NVVM atomics, warp shuffles — which is why it
lives under ``_internal/`` rather than ``compile_time/``. Three groups, by who needs them:

* **Distributed shared memory** (``elem_pointer``, ``set_block_rank``, ``store_shared_remote``,
  ``store_shared_remote_x4``) — cluster-scoped writes for ``reduce.cluster_reduce`` and for the
  tile scheduler's cross-CTA work broadcast.
* **Fast math** (``sqrt``, ``ceil``) — the approximate PTX forms, deliberately not the IEEE ones.
* **Scheduling primitives** (``warp_prefix_sum``, ``atomic_inc_i32``, ``atomic_add_i32``,

Still trimmed relative to upstream: the ``f32x2``/``i64`` packing helpers and ``fmin`` are left out
until a kernel in this tree needs them.
"""

import math
from typing import Optional

import cutlass
import cutlass.cute as cute
from cutlass import Int32, Float32, const_expr
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass._mlir.dialects import llvm, nvvm


@dsl_user_op
def elem_pointer(x: cute.Tensor, coord: cute.Coord, *, loc=None, ip=None) -> cute.Pointer:
    """Return a pointer to the element of ``x`` at ``coord``.

    Resolves the coordinate through the tensor's layout, so it is correct for strided and
    hierarchical layouts, not only contiguous ones.

    Args:
        x: Tensor to index into.
        coord: Coordinate, matching ``x``'s layout profile (may be hierarchical).
        loc: Optional DSL source location, injected by ``@dsl_user_op``.
        ip: Optional DSL insertion point, injected by ``@dsl_user_op``.

    Returns:
        A ``cute.Pointer`` to that element, in ``x``'s address space.
    """
    return x.iterator + cute.crd2idx(coord, x.layout, loc=loc, ip=ip)


@dsl_user_op
def set_block_rank(
    smem_ptr: cute.Pointer, peer_cta_rank_in_cluster: Int32, *, loc=None, ip=None
) -> Int32:
    """Map the given smem pointer to the address at another CTA rank in the cluster."""
    smem_ptr_i32 = smem_ptr.toint(loc=loc, ip=ip).ir_value()
    return Int32(
        llvm.inline_asm(
            T.i32(),
            [smem_ptr_i32, peer_cta_rank_in_cluster.ir_value()],
            "mapa.shared::cluster.u32 $0, $1, $2;",
            "=r,r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def store_shared_remote(
    val: float | Float32 | Int32 | cutlass.Int64,
    smem_ptr: cute.Pointer,
    mbar_ptr: cute.Pointer,
    peer_cta_rank_in_cluster: cute.typing.Int,
    *,
    loc=None,
    ip=None,
) -> None:
    """Store ``val`` into a peer CTA's shared memory and signal that peer's mbarrier.

    This is the distributed-shared-memory write behind a cluster reduction: each CTA pushes its
    partial into every peer's reduction buffer, and the accompanying
    ``mbarrier::complete_tx`` arrival is what lets the peer know the byte count landed. Both the
    data pointer and the mbarrier pointer are remapped into the peer's address space via
    ``set_block_rank``, so the caller passes its OWN local pointers.

    Args:
        val: Value to store. A Python float is promoted to ``Float32``.
        smem_ptr: Local shared-memory destination pointer, remapped to the peer.
        mbar_ptr: Local mbarrier pointer, remapped to the peer and signalled on completion.
        peer_cta_rank_in_cluster: Destination CTA rank within the cluster.
        loc: Optional DSL source location, injected by ``@dsl_user_op``.
        ip: Optional DSL insertion point, injected by ``@dsl_user_op``.

    Returns:
        None. The asynchronous store is emitted as a side effect; completion is observed
        through the peer's mbarrier, not through this call.

    Raises:
        AssertionError: If ``val`` is not Float32, Int32, or Int64 -- the three types the
            ``st.async`` instruction has a suffix for.
    """
    remote_smem_ptr_i32 = set_block_rank(
        smem_ptr, peer_cta_rank_in_cluster, loc=loc, ip=ip
    ).ir_value()
    remote_mbar_ptr_i32 = set_block_rank(
        mbar_ptr, peer_cta_rank_in_cluster, loc=loc, ip=ip
    ).ir_value()
    if const_expr(isinstance(val, float)):
        val = Float32(val)
    assert isinstance(val, (Float32, Int32, cutlass.Int64)), "val must be Float32, Int32, or Int64"
    suffix = {Float32: "f32", Int32: "s32", cutlass.Int64: "s64"}[type(val)]
    constraint = {Float32: "f", Int32: "r", cutlass.Int64: "l"}[type(val)]
    llvm.inline_asm(
        None,
        [remote_smem_ptr_i32, val.ir_value(loc=loc, ip=ip), remote_mbar_ptr_i32],
        f"st.async.shared::cluster.mbarrier::complete_tx::bytes.{suffix} [$0], $1, [$2];",
        f"r,{constraint},r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


@dsl_user_op
def store_shared_remote_x4(
    val0: Float32 | Int32,
    val1: Float32 | Int32,
    val2: Float32 | Int32,
    val3: Float32 | Int32,
    smem_ptr: cute.Pointer,
    mbar_ptr: cute.Pointer,
    peer_cta_rank_in_cluster: cute.typing.Int,
    *,
    loc=None,
    ip=None,
) -> None:
    """Store four 32-bit values into a peer CTA's shared memory in ONE vectorized async store.

    The ``.v4`` form of :func:`store_shared_remote`. One instruction instead of four means one
    ``mbarrier::complete_tx`` arrival instead of four, which is why the tile scheduler uses it to
    broadcast a work tile (``m``, ``n``, ``l``, ``valid``) to every CTA in the cluster: four separate
    arrivals would have to be waited on separately.

    Args:
        val0: First value. Its runtime type selects the instruction suffix for all four, so it must
            be a ``Float32`` or ``Int32`` **instance** -- a Python ``float``/``int`` is not promoted
            here (unlike :func:`store_shared_remote`) and trips the assertion.
        val1: Second value. Converted to ``val0``'s type; a mismatched type is silently coerced, not
            rejected, so pass four values of one type.
        val2: Third value, same requirement as ``val1``.
        val3: Fourth value, same requirement as ``val1``.
        smem_ptr: Local shared-memory destination, remapped into the peer's window by
            ``set_block_rank``. Must be 16-byte aligned -- a ``.v4`` store to a misaligned address
            faults at runtime rather than being split by the hardware.
        mbar_ptr: Local mbarrier pointer, remapped the same way and signalled on completion.
        peer_cta_rank_in_cluster: Destination CTA rank within the cluster.
        loc: Optional DSL source location, injected by ``@dsl_user_op``.
        ip: Optional DSL insertion point, injected by ``@dsl_user_op``.

    Returns:
        None. The store is asynchronous; the peer observes it through its mbarrier.

    Raises:
        AssertionError: If ``val0`` is neither ``Float32`` nor ``Int32``.
    """
    remote_smem_ptr_i32 = set_block_rank(
        smem_ptr, peer_cta_rank_in_cluster, loc=loc, ip=ip
    ).ir_value()
    remote_mbar_ptr_i32 = set_block_rank(
        mbar_ptr, peer_cta_rank_in_cluster, loc=loc, ip=ip
    ).ir_value()
    assert isinstance(val0, (Float32, Int32)), "val must be Float32, or Int32"
    dtype = Float32 if isinstance(val0, Float32) else Int32
    suffix = {Float32: "f32", Int32: "s32"}[dtype]
    constraint = {Float32: "f", Int32: "r"}[dtype]
    llvm.inline_asm(
        None,
        [
            remote_smem_ptr_i32,
            remote_mbar_ptr_i32,
            dtype(val0).ir_value(loc=loc, ip=ip),
            dtype(val1).ir_value(loc=loc, ip=ip),
            dtype(val2).ir_value(loc=loc, ip=ip),
            dtype(val3).ir_value(loc=loc, ip=ip),
        ],
        "{\n\t"
        f".reg .v4 .{suffix} abcd;\n\t"
        f"mov.{suffix} abcd.x, $2;\n\t"
        f"mov.{suffix} abcd.y, $3;\n\t"
        f"mov.{suffix} abcd.z, $4;\n\t"
        f"mov.{suffix} abcd.w, $5;\n\t"
        f"st.async.shared::cluster.mbarrier::complete_tx::bytes.v4.{suffix} [$0], abcd, [$1];\n\t"
        "}\n",
        f"r,r,{constraint},{constraint},{constraint},{constraint}",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


@cute.jit
def load_scalar_or_pointer(x, dtype=Float32):
    """Read an epilogue scalar that may be a compile-time value OR a device pointer.

    Epilogue scalars like ``alpha`` and ``beta`` arrive one of two ways: baked in as a constant when
    the host knew the value at compile time, or as a GMEM pointer when it is only known at launch
    (a tensor-valued alpha). Both reach the same expression in the epilogue, so the branch is folded
    here rather than duplicated at every use site.

    Args:
        x: Either a ``cute.Pointer`` into GMEM (dereferenced, one element) or any value the DSL can
            use directly. The choice is a ``const_expr``, so the un-taken branch is pruned and no
            load is emitted for the constant case.
        dtype: Numeric type to read the pointed-to element as. Must match the type the pointer was
            created with -- nothing checks it, and a mismatch reinterprets the bits.

    Returns:
        A value of ``dtype`` when ``x`` is a pointer; otherwise ``x`` unchanged (NOT converted to
        ``dtype``).
    """
    if const_expr(isinstance(x, cute.Pointer)):
        return dtype(cute.make_tensor(x, cute.make_layout(1))[0])
    else:
        return x


@dsl_user_op
def sqrt(a: float | Float32, *, loc=None, ip=None) -> Float32:
    """Approximate square root via ``sqrt.approx.f32``.

    The fast hardware form, roughly 22-bit accurate, NOT IEEE-correctly-rounded. Used where the
    result feeds a heuristic (a tile-count estimate, a swizzle width), never where it is compared
    for equality against a CPU reference.

    Args:
        a: Non-negative value. A negative input yields NaN silently -- the instruction has no
            domain check.
        loc: Optional DSL source location, injected by ``@dsl_user_op``.
        ip: Optional DSL insertion point, injected by ``@dsl_user_op``.

    Returns:
        The approximate square root, as ``Float32``.
    """
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Float32(a).ir_value(loc=loc, ip=ip)],
            "sqrt.approx.f32 $0, $1;",
            "=f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def ceil(a: float | Float32, *, loc=None, ip=None) -> Int32:
    """Round a float UP to the next integer via ``cvt.rpi.ftz.s32.f32``.

    One instruction, versus a convert-compare-add sequence. ``.ftz`` flushes denormals to zero,
    which is harmless for the grid arithmetic this serves and would matter only for inputs within
    2^-126 of zero.

    Args:
        a: Value to round. Must be within ``int32`` range after rounding; outside it the conversion
            saturates rather than wrapping, so a huge input silently gives ``INT_MAX``.
        loc: Optional DSL source location, injected by ``@dsl_user_op``.
        ip: Optional DSL insertion point, injected by ``@dsl_user_op``.

    Returns:
        ``ceil(a)`` as ``Int32``.
    """
    return Int32(
        llvm.inline_asm(
            T.i32(),
            [Float32(a).ir_value(loc=loc, ip=ip)],
            "cvt.rpi.ftz.s32.f32 $0, $1;",
            "=r,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@cute.jit
def warp_prefix_sum(val: Int32, lane: Optional[Int32] = None) -> Int32:
    """Inclusive prefix sum of ``val`` across the 32 lanes of one warp, via shuffle-up.

    **Nothing in this tree calls it, and that is a PENDING state rather than a dead one.**
    Its one caller upstream is `VarlenMTileScheduler` (`main:_internal/tile_scheduler.py:1308`),
    which turns per-problem cluster counts into a cumulative tile index -- and that scheduler,
    `_internal/varlen_utils.py` and the ``gather_A`` load path are all still to be brought
    back. **Do not remove this on a no-callers finding alone**: such a finding describes this
    tree's state mid-migration, not this function.

    Five ``shuffle_sync_up`` rounds (log2 of the warp size), each adding the value from ``offset``
    lanes below. Lane ``i`` ends holding the sum of lanes ``0..i``.

    Args:
        val: This lane's contribution. Every lane of the warp must reach this call -- the shuffles
            are ``sync`` and a divergent warp deadlocks or reads garbage.
        lane: This thread's lane index, or None to read ``cute.arch.lane_idx()``. Pass it when the
            caller already has it; passing a WRONG value silently corrupts the result, because it
            decides which lanes skip an accumulation round.

    Returns:
        The inclusive prefix sum in every lane.

    Note:
        ``mask_and_clamp=0`` is load-bearing: it makes a shuffle from below lane 0 return the
        lane's own value unchanged, which combined with the ``lane >= offset`` guard is what stops
        the low lanes from folding in a neighbour's partial.
    """
    if const_expr(lane is None):
        lane = cute.arch.lane_idx()
    for i in cutlass.range_constexpr(int(math.log2(cute.arch.WARP_SIZE))):
        offset = 1 << i
        # Very important that we set mask_and_clamp to 0
        partial_sum = cute.arch.shuffle_sync_up(val, offset=offset, mask_and_clamp=0)
        if lane >= offset:
            val += partial_sum
    return val


@dsl_user_op
def atomic_inc_i32(a: int | Int32, gmem_ptr: cute.Pointer, *, loc=None, ip=None) -> Int32:
    """Atomically increment a GMEM counter with wraparound, returning the PREVIOUS value.

    NVVM's ``INC`` is not "add one": it computes ``old >= a ? 0 : old + 1``, i.e. it increments
    modulo ``a + 1``. That is exactly what a dynamic tile scheduler wants from a work counter it
    intends to reuse, and exactly not what a caller expecting ``+= 1`` wants -- pass a bound larger
    than any value the counter can reach if you want plain increment.

    Args:
        a: The wraparound bound described above, NOT an addend.
        gmem_ptr: Pointer to the 32-bit counter. Must be in global memory and 4-byte aligned; the
            atomic is device-scoped, so a shared-memory pointer is a silent correctness bug.
        loc: Optional DSL source location, injected by ``@dsl_user_op``.
        ip: Optional DSL insertion point, injected by ``@dsl_user_op``.

    Returns:
        The value the counter held **before** this increment.

    Note:
        Two NVVM spellings are supported because CUDA 12.9 requires an explicit result type on
        ``atomicrmw`` and later versions infer it. The version is read at trace time, not import
        time, so a mixed-toolchain environment picks per compile.
    """
    from cutlass import CUDA_VERSION

    if CUDA_VERSION.major == 12 and CUDA_VERSION.minor == 9:
        # Old API: requires explicit result type as first positional argument
        return nvvm.atomicrmw(
            res=T.i32(), op=nvvm.AtomicOpKind.INC, ptr=gmem_ptr.llvm_ptr, a=Int32(a).ir_value()
        )
    else:
        # New API: infers result type automatically
        return nvvm.atomicrmw(
            op=nvvm.AtomicOpKind.INC, ptr=gmem_ptr.llvm_ptr, a=Int32(a).ir_value()
        )


@dsl_user_op
def atomic_add_i32(a: int | Int32, gmem_ptr: cute.Pointer, *, loc=None, ip=None) -> Int32:
    """Atomically add to a GMEM 32-bit integer, returning the PREVIOUS value.

    Args:
        a: The addend. May be negative.
        gmem_ptr: Pointer to the counter. Must be in global memory and 4-byte aligned -- the atomic
            is device-scoped, so pointing it at shared memory is a silent correctness bug rather
            than an error.
        loc: Optional DSL source location, injected by ``@dsl_user_op``.
        ip: Optional DSL insertion point, injected by ``@dsl_user_op``.

    Returns:
        The value held **before** the add, which is what makes this usable as a ticket dispenser.

    Note:
        Same CUDA 12.9 vs later ``atomicrmw`` spelling split as :func:`atomic_inc_i32`.
    """
    from cutlass import CUDA_VERSION

    if CUDA_VERSION.major == 12 and CUDA_VERSION.minor == 9:
        # Old API: requires explicit result type as first positional argument
        return nvvm.atomicrmw(
            res=T.i32(), op=nvvm.AtomicOpKind.ADD, ptr=gmem_ptr.llvm_ptr, a=Int32(a).ir_value()
        )
    else:
        # New API: infers result type automatically
        return nvvm.atomicrmw(
            op=nvvm.AtomicOpKind.ADD, ptr=gmem_ptr.llvm_ptr, a=Int32(a).ir_value()
        )
