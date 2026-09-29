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

"""Runtime memory-copy helpers (CuTe-DSL).

These emit real instructions: ``copy`` issues the data movement, ``predicate_k`` materializes a
register-resident boolean mask (162 emitted ops, 8 producing runtime values). The *descriptors*
they consume -- ``get_copy_atom``, ``tiled_copy_2d`` -- are compile-time algebra and live in
``_internal/compile_time/copy_descriptors.py``.

Two groups, added as the kernels that need them arrived:

* **Element copies** -- ``copy``, ``predicate_k``, ``fill_oob``, ``cvt_copy``, ``sr_cvt_copy``.
* **TMA copy closures** -- ``tma_get_copy_fn`` and friends, which partition once at trace time and
  return a closure the mainloop calls per k-tile.

The ragged-view helpers (``create_ragged_tensor_for_tma`` / ``offset_ragged_tensor``) and the
indexed cp.async gather copies were removed with varlen and gather-A: the A2A-fused TriMul workflow
uses neither, and every operand here is dense with a batch mode. Restoring them means restoring
``VarlenManager`` too -- they are one feature, not four helpers.

Still trimmed relative to upstream: the swizzle-pointer helpers and ``tma_gather4_load`` are left
out until a kernel here needs them. The two atom pickers that ARE here --
``sm90_get_smem_load_op`` and ``sm90_get_smem_store_atom`` -- are the load and store legs of the
epilogue; the store one arrived with the gated post-activation, which is the first tile in this
tree narrower than the D store beside it.
"""

from typing import Callable, Optional, Tuple, Type

import cutlass
import cutlass.cute as cute
import cutlass.pipeline
from cutlass import Int32, Boolean, const_expr
from cutlass.cute.nvgpu import cpasync, warp
from cutlass.cutlass_dsl import dsl_user_op

from fold_cp_ops._internal.compile_time.copy_descriptors import get_copy_atom

#: Stand-in extent for the ragged (variable-length) mode of a TMA view. TMA descriptors are built at
#: compile time and need a static extent, but a varlen batch's true length is only known at launch.
#: 2**30 is large enough that no real sequence reaches it, small enough that ``BIG_INT * stride``
#: does not overflow the 64-bit address arithmetic.
BIG_INT = 2**30
#: The largest extent a TMA box mode may declare. Used for the synthetic wraparound modes, which
#: exist to make the address arithmetic wrap rather than to be iterated.
MAX_INT = 2**31 - 1
#: ``2**64 // BIG_INT``. Paired with ``-stride`` it makes the second synthetic mode subtract exactly
#: what the first added, modulo 2**64 -- which is how the wraparound lands back on the true base.
BIG_INT_INV = 2**64 // BIG_INT


@dsl_user_op
def copy(
    src: cute.Tensor,
    dst: cute.Tensor,
    *,
    pred: Optional[cute.Tensor] = None,
    is_async: bool = False,
    loc=None,
    ip=None,
    **kwargs,
) -> None:
    """Copy ``src`` to ``dst``, deriving the access width from ``src``'s own layout.

    The vector width is read off ``src.shape[0][0]`` (the innermost value mode of an already
    thread-partitioned tensor) rather than passed in, so callers cannot pick a width that
    disagrees with the partitioning they built.

    Args:
        src: Thread-partitioned source tensor. Its innermost value-mode extent sets the width.
        dst: Thread-partitioned destination tensor, same partitioning as ``src``.
        pred: Optional per-element predicate masking out-of-bounds lanes. Build it with
            ``predicate_k`` for a partial trailing tile; pass None when the extent divides evenly.
        is_async: If True issue an async ``cp.async`` copy; the caller is then responsible for
            the matching ``cp_async_commit_group`` / ``cp_async_wait_group``.
        loc: Optional DSL source location, injected by ``@dsl_user_op``.
        ip: Optional DSL insertion point, injected by ``@dsl_user_op``.
        **kwargs: Forwarded verbatim to ``cute.copy``.

    Returns:
        None. The copy is emitted as a side effect.
    """
    num_copy_elems = src.shape[0][0]
    copy_atom = get_copy_atom(src.element_type, num_copy_elems, is_async)
    cute.copy(copy_atom, src, dst, pred=pred, loc=loc, ip=ip, **kwargs)


@cute.jit
def predicate_k(tAcA: cute.Tensor, limit: Int32) -> cute.Tensor:
    """Build a boolean predicate masking the k (contiguous) dimension past ``limit``.

    This is what lets the kernel accept a feature extent that is not a multiple of the tile
    width: the trailing partial tile is masked rather than forbidden. Only the k dimension is
    predicated -- the mn dimension is guarded by a plain ``if`` on the row index, which is
    cheaper than materializing a second predicate.

    Args:
        tAcA: Thread-partitioned identity (coordinate) tensor, giving each element its global
            coordinate. Shape ``((v0, v1), rest_m, rest_k)``.
        limit: Exclusive upper bound on the k coordinate, i.e. the true feature extent.

    Returns:
        A register-resident ``Boolean`` tensor broadcast over the mn mode (stride 0), True where
        the element's k coordinate is ``< limit``. Pass it as ``copy(..., pred=...)``.
    """
    # Only compute predicates for the "k" dimension. For the mn dimension, we will use "if"
    tApA = cute.make_rmem_tensor(
        cute.make_layout(
            (cute.size(tAcA, mode=[0, 1]), cute.size(tAcA, mode=[1]), cute.size(tAcA, mode=[2])),
            stride=(cute.size(tAcA, mode=[2]), 0, 1),
        ),
        Boolean,
    )
    for rest_v in cutlass.range_constexpr(tApA.shape[0]):
        for rest_k in cutlass.range_constexpr(tApA.shape[2]):
            tApA[rest_v, 0, rest_k] = cute.elem_less(tAcA[(0, rest_v), 0, rest_k][1], limit)
    return tApA


@cute.jit
def fill_oob(tXX: cute.Tensor, tXpX: Optional[cute.Tensor], fill_value: cute.Numeric) -> None:
    """Overwrite the predicate-masked (out-of-bounds) vectors of a staged tile with ``fill_value``.

    **Why this exists: a predicated copy leaves the masked tail holding a value the caller did not
    choose, and the right value depends on what is reduced next.** A predicated ``cp.async``
    *zero-fills* the masked destination; a predicated register copy leaves it untouched. Zero is the
    identity for an ADD reduction, so a sum needs no fill at all -- which is exactly why this is
    easy to forget. It is *not* the identity for anything else:

    * ``MAX`` reads the zeros as real data on an all-negative row -- fill with ``-inf``.
    * a *centred* sum such as LayerNorm's ``sum (x - mean)^2`` reads each masked lane as
      ``(0 - mean)^2 = mean^2``, inflating the variance by ``(tile - N) * mean^2 / N``. That was a
      live silent-wrong-output bug (docs/refactor_and_fix.md §3.3); the fix fills the masked lanes
      of ``x - mean`` with ``0.0``.

    Granularity is a whole **vector**, not an element: ``predicate_k`` masks at the innermost
    value-mode's outer sub-mode, which is sound because the vector width divides ``N``, so a vector
    is either wholly in bounds or wholly out.

    Args:
        tXX: The tile to patch, thread-partitioned as ``((v0, v1), rest_m, rest_k)``. May live in
            SMEM or in registers -- both are written the same way. Modified **in place**. Its
            ``v1`` and ``rest_k`` extents must match ``tXpX``'s mode-0 and mode-2 extents, or the
            wrong vectors are filled; this is not checked, because both come from the same
            ``TiledCopy`` partitioning in every intended caller.
        tXpX: The predicate from :func:`predicate_k`, shape ``(v1, rest_m, rest_k)`` with the
            ``rest_m`` mode broadcast (stride 0). Pass **None** to fill the tile unconditionally --
            the whole tile, not just its tail -- which is how a caller zeroes a tile it is about to
            accumulate into. Passing None when a real predicate was meant destroys live data.
        fill_value: The value written to every masked position. Must be convertible to
            ``tXX.element_type``; a value that does not round-trip in that type (e.g. an fp32 row
            mean into a bf16 tile) is silently rounded, which reintroduces the very error the fill
            is meant to remove -- fill a float32 tensor in that case.

    Returns:
        None. ``tXX`` is written as a side effect.
    """
    # One fill vector, reused for every masked position: shape (v0, rest_m), i.e. one whole
    # innermost vector across all the rows this thread owns.
    tXrX_fill = cute.make_rmem_tensor_like(tXX[(None, 0), None, 0])
    tXrX_fill.fill(fill_value)
    for rest_v in cutlass.range_constexpr(tXX.shape[0][1]):
        for rest_k in cutlass.range_constexpr(tXX.shape[2]):
            if const_expr(tXpX is not None):
                if not tXpX[rest_v, 0, rest_k]:
                    cute.autovec_copy(tXrX_fill, tXX[(None, rest_v), None, rest_k])
            else:
                cute.autovec_copy(tXrX_fill, tXX[(None, rest_v), None, rest_k])


@dsl_user_op
def cvt_copy(
    tiled_copy: cute.TiledCopy,
    src: cute.Tensor,
    dst: cute.Tensor,
    *,
    pred: Optional[cute.Tensor] = None,
    retile: bool = False,
    loc=None,
    ip=None,
    **kwargs,
) -> None:
    """Copy register -> anywhere, converting the element type on the way if it differs.

    The epilogue accumulates in fp32 and stores in bf16/fp16. Rather than making every store site
    branch on that, this does the conversion into a scratch register tensor when the types differ
    and issues a plain copy when they do not -- the branch is a ``const_expr``, so the un-taken side
    emits nothing.

    Args:
        tiled_copy: The copy atom/tiling to issue through. Its element type must match ``dst``.
        src: Source tensor. **Must be register-resident** (``rmem``); asserted, because a conversion
            through ``make_rmem_tensor_like`` of a shared- or global-memory tensor would silently
            allocate a register file the caller did not budget for.
        dst: Destination tensor. Its ``element_type`` decides whether a conversion happens.
        pred: Optional predicate tensor masking the copy, forwarded to ``cute.copy``. Note the
            conversion is NOT predicated -- masked lanes convert garbage, which is harmless because
            they are not stored.
        retile: Whether to re-partition ``src`` through ``tiled_copy.retile`` first. Needed when
            ``src`` was partitioned for a different tiling than the one issuing the copy.
        loc: Optional DSL source location.
        ip: Optional DSL insertion point.
        **kwargs: Forwarded to ``cute.copy``.

    Returns:
        None; the copy is emitted as a side effect.

    Raises:
        AssertionError: If ``src`` is not an rmem tensor.
    """
    assert isinstance(src.iterator, cute.Pointer) and src.memspace == cute.AddressSpace.rmem
    if const_expr(src.element_type != dst.element_type):
        src_cvt = cute.make_rmem_tensor_like(src, dst.element_type)
        src_cvt.store(src.load().to(dst.element_type))
        src = src_cvt
    if const_expr(retile):
        src = tiled_copy.retile(src)
    cute.copy(tiled_copy, src, dst, pred=pred, loc=loc, ip=ip, **kwargs)


@dsl_user_op
def sr_cvt_copy(
    tiled_copy: cute.TiledCopy,
    src: cute.Tensor,
    dst: cute.Tensor,
    seed: Int32,
    tidx: Int32,
    *,
    loc=None,
    ip=None,
) -> None:
    """:func:`cvt_copy` with **stochastic** rounding for the fp32 -> bf16 narrowing.

    Round-to-nearest is biased when a value is repeatedly accumulated at low precision; stochastic
    rounding removes that bias by rounding up with probability proportional to the discarded
    remainder. The randomness comes from ``seed`` mixed with the thread index, so it is
    reproducible for a fixed seed and decorrelated across threads.

    Args:
        tiled_copy: The copy atom/tiling to issue through.
        src: Register-resident fp32 source. Asserted rmem, same reason as :func:`cvt_copy`.
        dst: Destination. Its element type must be bf16 -- this is not a general converter, and any
            other narrow type would be reinterpreted from the bf16 bit pattern.
        seed: Per-launch random seed. The SAME value must be passed by every thread of the copy;
            varying it per thread is not more random, it just makes results irreproducible.
        tidx: This thread's index, mixed with ``seed`` to decorrelate lanes.
        loc: Optional DSL source location.
        ip: Optional DSL insertion point.

    Returns:
        None; the copy is emitted as a side effect.

    Raises:
        AssertionError: If ``src`` is not an rmem tensor.

    Note:
        Unreachable from this package's public API today -- `gemm()` refuses ``RoundingMode.RS`` as
        SM100-only. Kept because the epilogue's rounding branch is shared machinery.
    """
    assert isinstance(src.iterator, cute.Pointer) and src.memspace == cute.AddressSpace.rmem
    from fold_cp_ops._internal.rounding import convert_f32_to_bf16_sr
    from cutlass.cute.tensor import TensorSSA

    src_cvt = cute.make_rmem_tensor_like(src, dst.element_type)
    src_vec = src.load()
    raw_vec = convert_f32_to_bf16_sr(src_vec, seed, tidx, loc=loc, ip=ip)
    src_cvt.store(TensorSSA(raw_vec, src_vec.shape, dst.element_type))
    src = src_cvt
    cute.copy(tiled_copy, src, dst, loc=loc, ip=ip)


@dsl_user_op
def sm90_get_smem_load_op(
    layout_c: cutlass.utils.LayoutEnum,
    elem_ty_c: Type[cutlass.Numeric],
    *,
    loc=None,
    ip=None,
) -> cute.CopyAtom:
    """Pick the widest SMEM->register load atom the output layout allows.

    For 16-bit outputs the ``ldmatrix`` instruction moves an 8x8 tile per warp in one go, and its
    transposing variant handles an m-major destination -- so the layout decides the atom, not just
    the type. For anything else there is no matrix-load form and it falls back to the universal
    (per-element) copy.

    Args:
        layout_c: Layout of the output tensor D. Only ``is_m_major_c()`` is consulted, to select the
            transposing ``ldmatrix``.
        elem_ty_c: Element type of D. Must be a cutlass ``Numeric`` **type**, not an instance.
        loc: Optional DSL source location.
        ip: Optional DSL insertion point.

    Returns:
        An ``ldmatrix`` 8x8x16b copy atom for 16-bit types, otherwise a universal copy atom.

    Raises:
        TypeError: If ``elem_ty_c`` is not a cutlass numeric type -- caught here because passing a
            torch dtype by mistake would otherwise produce an atom with the wrong element width.
    """
    if not isinstance(elem_ty_c, cutlass.cutlass_dsl.NumericMeta):
        raise TypeError(f"elem_ty_c must be a Numeric, but got {elem_ty_c}")
    is_m_major = layout_c.is_m_major_c()
    if elem_ty_c.width == 16:
        return cute.make_copy_atom(warp.LdMatrix8x8x16bOp(is_m_major, 4), elem_ty_c, loc=loc, ip=ip)
    else:
        return cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), elem_ty_c, loc=loc, ip=ip)


@dsl_user_op
def sm90_get_smem_store_atom(
    element_type: Type[cutlass.Numeric],
    transpose: bool = False,
    major_mode_size: Optional[int] = None,
    *,
    loc=None,
    ip=None,
) -> cute.CopyAtom:
    """Pick the register->SMEM store atom whose width matches the tile it will store.

    The counterpart of :func:`sm90_get_smem_load_op` for the store leg, and distinct from the D
    output's atom (``GemmSm90EpilogueMixin.epilog_smem_copy_atom``) in exactly one way that
    matters: it takes the tile's major-mode width rather than assuming it. ``stmatrix`` moves 1, 2
    or 4 8x8 matrices per instruction, and the count has to divide the tile -- an x4 atom aimed at
    an 8-wide tile partitions it wrongly.

    That case is not hypothetical: the gated epilogue HALVES its post-activation subtile
    (``2N -> N``), so a ``tile_N`` that is a multiple of 16 but not of 32 yields an 8-wide
    post-activation tile while the D store beside it is still 16-wide. Hardcoding x4 -- which is
    what ``cutlass``'s own ``sm90_get_smem_store_op`` does -- silently mis-partitions it.

    Args:
        element_type: Element type of the SMEM destination. Must be a cutlass ``Numeric`` **type**.
            Only 16-bit types get an ``stmatrix`` atom; anything else falls back to a universal
            copy, which is correct but slower.
        transpose: Whether the destination is m-major, selecting the transposing ``stmatrix``.
            Pass ``layout.is_m_major_c()``; getting it wrong stores a transposed tile with no error.
        major_mode_size: Extent of the tile's major mode, used to pick the matrix count: a multiple
            of 16 gives x4, of 8 gives x2, otherwise x1. **None means "assume x4"** -- pass the real
            width whenever the tile can be narrower than 16, or the partition is silently wrong.

    Returns:
        A ``cute.CopyAtom``: an ``stmatrix`` 8x8x16b atom for 16-bit types, else a universal copy
        atom sized to two elements (one when transposing).

    Raises:
        TypeError: If ``element_type`` is not a cutlass numeric type -- a torch dtype passed here
            would otherwise build an atom with the wrong element width.
    """
    if not isinstance(element_type, cutlass.cutlass_dsl.NumericMeta):
        raise TypeError(f"element_type must be a Numeric, but got {element_type}")
    if const_expr(element_type.width != 16):
        return cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            element_type,
            num_bits_per_copy=(2 if not transpose else 1) * element_type.width,
            loc=loc,
            ip=ip,
        )
    num_matrices = (
        4
        if major_mode_size is None or major_mode_size % 16 == 0
        else (2 if major_mode_size % 8 == 0 else 1)
    )
    return cute.make_copy_atom(
        warp.StMatrix8x8x16bOp(transpose=transpose, num_matrices=num_matrices),
        element_type,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def tma_get_copy_fn(
    atom: cute.CopyAtom,
    cta_coord: cute.Coord,
    cta_layout: cute.Layout,
    src_tensor: cute.Tensor,
    dst_tensor: cute.Tensor,
    filter_zeros: bool = False,
    single_stage: bool = False,
    *,
    loc=None,
    ip=None,
    **kwargs,
) -> Tuple[Callable, cute.Tensor, cute.Tensor]:
    """Partition a TMA copy once at trace time and return the per-tile closure that issues it.

    TMA partitioning (``cpasync.tma_partition``) is layout algebra: it costs nothing at runtime but
    is verbose, and doing it inside the mainloop would repeat it per k-tile. Hoisting it here and
    returning a closure keeps the mainloop to one call per tile.

    The direction is inferred, not passed: whichever of the two tensors lives in shared memory is
    the SMEM side, so the same helper serves both G2S loads and S2G stores.

    Args:
        atom: The TMA copy atom, built from the same layouts as the tensors below.
        cta_coord: This CTA's coordinate within the multicast cluster. Ignored when ``cta_layout``
            is trivial.
        cta_layout: Layout of the multicast group. Pass ``make_layout(1)`` for no multicast; a
            mismatch against the atom's multicast configuration is a hang, not an error.
        src_tensor: Source. Exactly one of src/dst must be in SMEM.
        dst_tensor: Destination.
        filter_zeros: Drop stride-0 modes from the partitioned views. Use when the operand is
            broadcast along a mode and the extra iteration would re-copy the same bytes.
        single_stage: True when the SMEM side has no pipeline-stage mode, which changes the returned
            closure's signature -- it then takes no indices at all.
        loc: Optional DSL source location.
        ip: Optional DSL insertion point.
        **kwargs: Bound into every issued copy (e.g. a multicast mask).

    Returns:
        ``(copy_fn, s, g)``. ``copy_fn(src_idx, dst_idx, **kw)`` issues one tile's copy -- or
        ``copy_fn(**kw)`` when ``single_stage``. ``s`` and ``g`` are the partitioned SMEM and GMEM
        views, returned because callers need their shapes to size barriers and predicates.
    """
    src_is_smem = const_expr(
        isinstance(src_tensor.iterator, cute.Pointer)
        and src_tensor.memspace == cute.AddressSpace.smem
    )
    smem_tensor, gmem_tensor = (src_tensor, dst_tensor) if src_is_smem else (dst_tensor, src_tensor)
    group_rank_smem = const_expr(cute.rank(smem_tensor) - (1 if not single_stage else 0))
    group_rank_gmem = const_expr(cute.rank(gmem_tensor) - (1 if not single_stage else 0))
    # ((atom_v, rest_v), STAGE), ((atom_v, rest_v), RestK)
    s, g = cpasync.tma_partition(
        atom,
        cta_coord,
        cta_layout,
        cute.group_modes(smem_tensor, 0, group_rank_smem),
        cute.group_modes(gmem_tensor, 0, group_rank_gmem),
        loc=loc,
        ip=ip,
    )
    if const_expr(filter_zeros):
        s = cute.filter_zeros(s)
        g = cute.filter_zeros(g)
    src, dst = (s, g) if src_is_smem else (g, s)

    @dsl_user_op
    def copy_tma(src_idx, dst_idx, *, loc=None, ip=None, **new_kwargs):
        """Issue one staged TMA copy.

        Args:
            src_idx: Index along the source's trailing (tile or stage) mode.
            dst_idx: Index along the destination's trailing mode.
            loc: Optional DSL source location.
            ip: Optional DSL insertion point.
            **new_kwargs: Per-call additions -- notably ``mbar_ptr``, the barrier the async copy
                completes against. Merged with the kwargs bound when the closure was built; a key in
                both is a duplicate-keyword ``TypeError``.

        Returns:
            None; the copy is asynchronous and completes against the barrier.
        """
        cute.copy(
            atom, src[None, src_idx], dst[None, dst_idx], **new_kwargs, **kwargs, loc=loc, ip=ip
        )

    @dsl_user_op
    def copy_tma_single_stage(*, loc=None, ip=None, **new_kwargs):
        """Issue the whole (unstaged) TMA copy; there is no index to advance.

        Args:
            loc: Optional DSL source location.
            ip: Optional DSL insertion point.
            **new_kwargs: Per-call additions, e.g. ``mbar_ptr``.

        Returns:
            None; asynchronous.
        """
        cute.copy(atom, src, dst, **new_kwargs, **kwargs, loc=loc, ip=ip)

    return (copy_tma if const_expr(not single_stage) else copy_tma_single_stage), s, g


@dsl_user_op
def tma_get_chunked_B_copy_fn(
    atom_up: cute.CopyAtom,
    atom_gate: cute.CopyAtom,
    gWp_blk: cute.Tensor,  # (G, tile_K, nblk_all, RestK) -- this CTA's up range (local_tile form)
    gWg_blk: cute.Tensor,  # (G, tile_K, nblk_all, RestK) -- this CTA's gate range (local_tile form)
    sB_z: cute.Tensor,  # ((G,tile_K), (nsub, 1, STAGE)) -- B smem zipped_divide by (G, tile_K)
    nblk: int,
    *,
    loc=None,
    ip=None,
    **kwargs,
) -> Callable:
    """Composite B-load for the block-interleaved dual-gated layout.

    The 2N-wide preact B tile is laid out as ``nblk`` blocks ``[up_G | gate_G | up_G | ...]``, so
    block ``j`` occupies SMEM sub-blocks ``2j`` (up) and ``2j+1`` (gate) but reads from two
    *separate* contiguous GMEM ranges. That cannot be one TMA copy, so this issues ``2 * nblk`` of
    them per k-tile, partitioning each once at trace time.

    Args:
        atom_up: TMA atom for the "up" half.
        atom_gate: TMA atom for the "gate" half. Usually identical to ``atom_up``; separate so the
            two halves may differ in dtype.
        gWp_blk: This CTA's up-operand GMEM range, ``local_tile``-shaped ``(G, tile_K, nblk, RestK)``.
        gWg_blk: The gate-operand counterpart, same shape.
        sB_z: The B SMEM tile, ``zipped_divide``d by ``(G, tile_K)`` so sub-block indexing is a
            single coordinate.
        nblk: Number of interleaved blocks. Must be a **static Python int**: it is the trip count of
            a trace-time loop, so a runtime value would fail to unroll.
        loc: Optional DSL source location.
        ip: Optional DSL insertion point.
        **kwargs: Bound into every issued copy.

    Returns:
        ``copy_chunked_B(k_tile, smem_idx, **kw)``, issuing all ``2 * nblk`` copies for one k-tile.

    Note:
        No multicast: the chunked path requires ``cluster_M == 1``, so ``cta_coord`` is 0 and
        ``cta_layout`` is trivial. Calling it under a wider cluster silently loads only this CTA's
        share.
    """
    cta_coord = 0
    cta_layout = cute.make_layout(1)
    s_up, g_up, s_gate, g_gate = [], [], [], []
    for j in range(nblk):
        # sB_z[None, (sub,0,None)] -> ((G,tile_K), STAGE) (box already grouped by zipped_divide).
        # gW*_blk[None,None,j,None] -> (G,tile_K,RestK); group box modes 0-1.
        s_u, g_u = cpasync.tma_partition(
            atom_up,
            cta_coord,
            cta_layout,
            sB_z[None, (2 * j, 0, None)],
            cute.group_modes(gWp_blk[None, None, j, None], 0, 2),
            loc=loc,
            ip=ip,
        )
        s_g, g_g = cpasync.tma_partition(
            atom_gate,
            cta_coord,
            cta_layout,
            sB_z[None, (2 * j + 1, 0, None)],
            cute.group_modes(gWg_blk[None, None, j, None], 0, 2),
            loc=loc,
            ip=ip,
        )
        s_up.append(s_u)
        g_up.append(g_u)
        s_gate.append(s_g)
        g_gate.append(g_g)

    @dsl_user_op
    def copy_chunked_B(k_tile, smem_idx, *, loc=None, ip=None, **new_kwargs):
        """Issue all ``2 * nblk`` block copies for one k-tile.

        Args:
            k_tile: Index of the k-tile to load, along the GMEM views' trailing mode.
            smem_idx: Destination pipeline stage.
            loc: Optional DSL source location.
            ip: Optional DSL insertion point.
            **new_kwargs: Per-call additions, e.g. ``mbar_ptr``. The SAME barrier receives all
                ``2 * nblk`` completions, so the caller's expected transaction count must cover
                every block, not one.

        Returns:
            None; the copies are asynchronous.
        """
        # nblk is a static python int -> this plain for-loop unrolls at trace time.
        for j in range(nblk):
            cute.copy(
                atom_up,
                g_up[j][None, k_tile],
                s_up[j][None, smem_idx],
                **new_kwargs,
                **kwargs,
                loc=loc,
                ip=ip,
            )
            cute.copy(
                atom_gate,
                g_gate[j][None, k_tile],
                s_gate[j][None, smem_idx],
                **new_kwargs,
                **kwargs,
                loc=loc,
                ip=ip,
            )

    return copy_chunked_B


def combine_copy_fns(copy_a: Callable, copy_b: Callable) -> Callable:
    """Fuse two copy closures into one that issues both with the same arguments.

    The mainloop has a single ``copy_B`` slot per pipeline stage; a kernel that needs two B operands
    (dual B plus a gate) fills it with one of these.

    Args:
        copy_a: First closure. Called first.
        copy_b: Second closure. Must accept the same signature as ``copy_a`` -- nothing checks it,
            and a mismatch surfaces as a ``TypeError`` at trace time.

    Returns:
        A closure forwarding ``*args, **kwargs`` to both, in order.

    Note:
        Deliberately a plain closure and NOT ``functools.partial``. ``partial`` is a registered
        pytree, so the DSL would flatten its captured arguments at a control-flow boundary and the
        copies would be issued against stale values.
    """

    def copy_fn(*args, **kwargs):
        """Issue both wrapped copies with identical arguments, ``copy_a`` first.

        Args:
            *args: Forwarded verbatim to both.
            **kwargs: Forwarded verbatim to both.

        Returns:
            None. Neither closure's return value is propagated, since both are side-effecting.
        """
        copy_a(*args, **kwargs)
        copy_b(*args, **kwargs)

    return copy_fn


def tma_producer_copy_fn(copy: Callable, pipeline: cutlass.pipeline.PipelineAsync) -> Callable:
    """Bind a TMA copy closure to a pipeline, so the caller supplies only the source index.

    Adapts :func:`tma_get_copy_fn`'s ``(src_idx, dst_idx, tma_bar_ptr)`` shape to what a producer
    warp actually has: a source tile index and its current pipeline state. The destination SMEM
    stage and the mbarrier to complete against both come from that state.

    Args:
        copy: A closure from :func:`tma_get_copy_fn` (the non-``single_stage`` form), accepting
            ``src_idx``, ``dst_idx`` and ``tma_bar_ptr`` as keywords.
        pipeline: The pipeline whose producer barrier the copy completes against. Must be the one
            the caller advances -- passing a different pipeline arms the wrong barrier and hangs.

    Returns:
        ``copy_fn(src_idx, producer_state, **kw)``.
    """

    def copy_fn(src_idx, producer_state: cutlass.pipeline.PipelineState, **new_kwargs):
        """Issue one copy into the stage the producer state currently points at.

        Args:
            src_idx: Index of the source tile to load.
            producer_state: The producer's pipeline state. Supplies BOTH the destination stage
                (``.index``) and the barrier, so the caller must not have advanced it past the
                stage it acquired.
            **new_kwargs: Forwarded to the wrapped copy.

        Returns:
            None; asynchronous.
        """
        copy(
            src_idx=src_idx,
            dst_idx=producer_state.index,
            tma_bar_ptr=pipeline.producer_get_barrier(producer_state),
            **new_kwargs,
        )

    return copy_fn
