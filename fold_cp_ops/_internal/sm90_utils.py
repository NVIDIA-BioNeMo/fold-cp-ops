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

"""SM90 WGMMA helpers: staging operands into SMEM and registers, and issuing the MMA.

Deliberately **mixed** compile-time and runtime, and kept in one module for that reason.
``make_smem_layout`` and ``partition_for_epilogue`` are pure layout algebra that is erased before
codegen; ``gemm``/``gemm_w_idx``/``partition_fragment_ABC`` emit real WGMMA instructions and
allocate registers. Splitting them would put four functions that are always used together in two
files -- the same judgement ``reduction_base.py`` records, for the same reason.

Everything here is Hopper-specific. The WGMMA atom is warpgroup-scoped (128 threads), reads both
operands from SMEM (or A from registers), and accumulates in place, which is why the helpers take a
``TiledMma`` rather than building one per call.
"""

from typing import Type, Union, Optional

import cutlass
import cutlass.cute as cute
import cutlass.utils.hopper_helpers as sm90_utils_og
from cutlass.cute.nvgpu import warpgroup
from cutlass.cutlass_dsl import Numeric, dsl_user_op
from cutlass import Float32, Int32, Boolean, const_expr
from cutlass.utils import LayoutEnum


@dsl_user_op
def make_smem_layout(
    dtype: Type[Numeric],
    layout: LayoutEnum,
    tile: cute.Tile,
    stage: Optional[int] = None,
    major_mode_size: Optional[int] = None,
    *,
    loc=None,
    ip=None,
) -> Union[cute.Layout, cute.ComposedLayout]:
    """Build the swizzled SMEM layout for one WGMMA operand (or epilogue tile).

    WGMMA reads its operands out of shared memory through a *swizzled* layout: the atom the hardware
    understands is chosen from the operand's dtype, major mode and the extent of its major dimension,
    then tiled up to the full tile shape. Getting the atom wrong does not fail loudly -- it produces
    bank conflicts, or a layout the MMA reads transposed.

    Args:
        dtype: Element type of the operand. Together with ``major_mode_size`` this picks the swizzle
            atom, so it must be the type the tile is actually stored as, not the accumulator's.
        layout: ``LayoutEnum`` for the operand, deciding m-major vs k/n-major. The mode ORDER of the
            result depends on it (``(1, 0, 2)`` vs ``(0, 1, 2)``), so passing the wrong one silently
            transposes the staged tile.
        tile: The tile shape to cover, e.g. ``(tile_M, tile_K)``. Flattened with ``product_each``,
            so a hierarchical tiler is accepted.
        stage: Number of pipeline stages to append as a third mode, or None for an unstaged layout.
            When None the ``order`` is truncated to two entries to match.
        major_mode_size: Extent of the major dimension, if it differs from the tile's. Defaults to
            the tile's own major extent. Overriding it is how a chunked or split operand keeps the
            swizzle its full-width sibling uses.
        loc: Optional DSL source location.
        ip: Optional DSL insertion point.

    Returns:
        A (usually composed/swizzled) layout of rank 2, or rank 3 when ``stage`` is given.
    """
    shape = cute.product_each(cute.shape(tile, loc=loc, ip=ip), loc=loc, ip=ip)
    if const_expr(major_mode_size is None):
        major_mode_size = shape[1] if layout.is_n_major_c() else shape[0]
    smem_layout_atom = warpgroup.make_smem_layout_atom(
        sm90_utils_og.get_smem_layout_atom(layout, dtype, major_mode_size),
        dtype,
    )
    order = (1, 0, 2) if const_expr(layout.is_m_major_c()) else (0, 1, 2)
    smem_layout_staged = cute.tile_to_shape(
        smem_layout_atom,
        cute.append(shape, stage) if const_expr(stage is not None) else shape,
        order=order if const_expr(stage is not None) else order[:2],
    )
    return smem_layout_staged


#: Alias of :func:`make_smem_layout`, kept because ``epi_utils.setup_epi_tensor`` selects between
#: this module and ``blackwell_helpers`` by architecture and calls the same name on either. The two
#: are genuinely the same operation on SM90; the alias is what lets the caller stay arch-agnostic.
make_smem_layout_epi = make_smem_layout


@dsl_user_op
def partition_for_epilogue(
    cT: cute.Tensor,
    epi_tile: cute.Tile,
    tiled_copy: cute.TiledCopy,
    tidx: Int32,
    reference_src: bool,  # do register tensors reference the src or dst layout of the tiled copy
    *,
    loc=None,
    ip=None,
) -> cute.Tensor:
    """Partition an epilogue tensor across one thread's share of the epilogue subtiles.

    Splits ``cT`` into epilogue-tile-sized pieces and hands back the slice this thread owns, laid
    out ``(CPY, CPY_M, CPY_N, EPI_M, EPI_N)`` -- per-copy value modes first, then which subtile.
    The epilogue loops over the trailing two modes.

    Args:
        cT: The tensor to partition. Usually an identity (coordinate) tensor or a GMEM view; it is
            not read here, only re-laid-out.
        epi_tile: The epilogue subtile shape. Must divide ``cT``'s shape -- ``flat_divide`` does not
            predicate, so a non-dividing tile silently produces a partial trailing subtile the
            caller has to mask itself.
        tiled_copy: The copy whose thread-value partitioning to apply.
        tidx: This thread's linear index within the copy's thread layout.
        reference_src: Whether the register fragment follows the copy's SOURCE layout
            (``partition_S``) or its DESTINATION layout (``partition_D``). Get this backwards and
            the partitioning is transposed relative to the data -- no error, wrong elements.
        loc: Optional DSL source location.
        ip: Optional DSL insertion point.

    Returns:
        This thread's partitioned view, rank 5 as described above.
    """
    thr_copy = tiled_copy.get_slice(tidx)
    cT_epi = cute.flat_divide(cT, epi_tile)
    # (CPY, CPY_M, CPY_N, EPI_M, EPI_N)
    if const_expr(reference_src):
        return thr_copy.partition_S(cT_epi, loc=loc, ip=ip)
    else:
        return thr_copy.partition_D(cT_epi, loc=loc, ip=ip)


@cute.jit
def gemm(
    tiled_mma: cute.TiledMma,
    acc: cute.Tensor,
    tCrA: cute.Tensor,
    tCrB: cute.Tensor,
    zero_init: cutlass.Constexpr[bool] = False,
    wg_wait: cutlass.Constexpr[int] = 0,
    # A_in_regs: cutlass.Constexpr[bool] = False,
    swap_AB: cutlass.Constexpr[bool] = False,
) -> None:
    """Issue the WGMMA chain for one k-block, accumulating into ``acc``.

    Loops over the operands' k mode issuing one ``cute.gemm`` per MMA instruction, with the
    accumulate flag cleared only on the first when ``zero_init``. A fresh ``mma_atom`` is built here
    rather than mutating ``tiled_mma``'s: setting the ACCUMULATE field on a shared atom makes the
    compiler report "operand #0 does not dominate this use".

    Args:
        tiled_mma: The tiled MMA whose op to issue. Not mutated.
        acc: Accumulator fragment, updated in place.
        tCrA: A operand fragment, ``(..., ..., k)``. Its k extent is the trip count, and it is read
            at trace time, so it must be static.
        tCrB: B operand fragment with a matching k extent.
        zero_init: If True the first instruction overwrites ``acc`` instead of accumulating -- which
            means ``acc`` need not be initialized, and a False here on an uninitialized ``acc`` is a
            garbage result rather than an error.
        wg_wait: How many warpgroup MMA groups to leave outstanding. 0 waits for all (the safe
            default); a negative value skips the wait entirely, leaving the caller to synchronize
            before reading ``acc``.
        swap_AB: Compute ``B @ A`` instead, by recursing with the operands exchanged. Used when the
            caller wants the transposed result without transposing memory.

    Returns:
        None; ``acc`` is updated in place.
    """
    if const_expr(swap_AB):
        gemm(tiled_mma, acc, tCrB, tCrA, zero_init=zero_init, wg_wait=wg_wait, swap_AB=False)
    else:
        warpgroup.fence()
        # We make a new mma_atom since we'll be modifying its attribute (accumulate).
        # Otherwise the compiler complains "operand #0 does not dominate this use"
        mma_atom = cute.make_mma_atom(tiled_mma.op)
        mma_atom.set(warpgroup.Field.ACCUMULATE, not zero_init)
        for k in cutlass.range_constexpr(cute.size(tCrA.shape[2])):
            cute.gemm(mma_atom, acc, tCrA[None, None, k], tCrB[None, None, k], acc)
            mma_atom.set(warpgroup.Field.ACCUMULATE, True)
        warpgroup.commit_group()
        if const_expr(wg_wait >= 0):
            warpgroup.wait_group(wg_wait)


def gemm_zero_init(
    tiled_mma: cute.TiledMma,
    shape: cute.Shape,
    tCrA: cute.Tensor,
    tCrB: cute.Tensor,
    A_idx: Optional[Int32] = None,
    B_idx: Optional[Int32] = None,
    wg_wait: int = -1,
    swap_AB: bool = False,
) -> cute.Tensor:
    """Allocate a zeroed accumulator of the right shape and run one WGMMA chain into it.

    The convenience form of :func:`gemm` for a caller that does not already hold an accumulator: it
    sizes the fragment from ``shape`` via the MMA's own partitioning, so the caller does not have to
    know the atom's register layout.

    Args:
        tiled_mma: The tiled MMA, which decides the accumulator's per-thread shape.
        shape: The ``(M, N)`` shape to accumulate over. Reversed internally when ``swap_AB``.
        tCrA: A operand fragment. May carry a leading selection mode indexed by ``A_idx``.
        tCrB: B operand fragment, likewise for ``B_idx``.
        A_idx: Optional index selecting one slice of ``tCrA``'s trailing mode -- for an operand
            staged once and consumed by several MMAs.
        B_idx: The same for ``tCrB``.
        wg_wait: Outstanding-group count, as in :func:`gemm`. Defaults to -1 (no wait), so the
            caller MUST synchronize before reading the result.
        swap_AB: Compute ``B @ A``; ``shape`` is reversed to match.

    Returns:
        A newly allocated fp32 accumulator fragment holding the product. Not zero-filled first --
        the first MMA overwrites it, which is what ``zero_init`` means here.
    """
    if const_expr(swap_AB):
        return gemm_zero_init(
            tiled_mma, shape[::-1], tCrB, tCrA, B_idx, A_idx, wg_wait, swap_AB=False
        )
    else:
        acc = cute.make_rmem_tensor(tiled_mma.partition_shape_C(shape), Float32)
        rA = tCrA if const_expr(A_idx is None) else tCrA[None, None, None, A_idx]
        rB = tCrB if const_expr(B_idx is None) else tCrB[None, None, None, B_idx]
        gemm(tiled_mma, acc, rA, rB, zero_init=True, wg_wait=wg_wait)
        return acc


def gemm_w_idx(
    tiled_mma: cute.TiledMma,
    acc: cute.Tensor,
    tCrA: cute.Tensor,
    tCrB: cute.Tensor,
    zero_init: Boolean,
    A_idx: Optional[Int32] = None,
    B_idx: Optional[Int32] = None,
    wg_wait: int = -1,
    swap_AB: bool = False,
) -> None:
    """:func:`gemm` with a **runtime** ``zero_init`` and optional operand selection.

    The difference from :func:`gemm` is entirely in ``zero_init``'s type: a ``Boolean`` rather than
    a Python bool, so whether the first instruction accumulates is decided at runtime. That is what
    the mainloop needs -- the first k-tile of each work tile overwrites, the rest accumulate, and
    which iteration is first is not known at trace time in a persistent kernel.

    Args:
        tiled_mma: The tiled MMA to issue.
        acc: Accumulator fragment, updated in place.
        tCrA: A operand fragment, optionally with a selectable trailing mode.
        tCrB: B operand fragment, likewise.
        zero_init: Runtime flag; True overwrites ``acc``, False accumulates into it.
        A_idx: Optional index into ``tCrA``'s trailing mode.
        B_idx: Optional index into ``tCrB``'s trailing mode.
        wg_wait: Outstanding-group count. Defaults to -1 (no wait) because the mainloop overlaps
            the MMA with the next tile's loads and waits explicitly.
        swap_AB: Compute ``B @ A``, exchanging the operands and their indices together.

    Returns:
        None; ``acc`` is updated in place.
    """
    if const_expr(swap_AB):
        gemm_w_idx(tiled_mma, acc, tCrB, tCrA, zero_init, B_idx, A_idx, wg_wait, swap_AB=False)
    else:
        rA = tCrA if const_expr(A_idx is None) else tCrA[None, None, None, A_idx]
        rB = tCrB if const_expr(B_idx is None) else tCrB[None, None, None, B_idx]
        gemm(tiled_mma, acc, rA, rB, zero_init=zero_init, wg_wait=wg_wait)


def partition_fragment_ABC(
    thr_mma: cute.ThrMma,
    shape_mnk: cute.Shape,
    sA: Optional[cute.Tensor],
    sB: Optional[cute.Tensor],
    swap_AB: bool = False,
):
    """Allocate the accumulator and the A/B register fragments for one thread's MMA.

    Three allocations in one call because they have to agree: the accumulator's shape comes from the
    MMA's C partitioning, and the operand fragments from its A and B partitionings, all for the same
    ``shape_mnk``. Deriving one of them separately is how a mismatch gets in.

    Whether A comes from SMEM or registers is read off the atom (``a_src``), not passed: an RS atom
    partitions A by *shape* because there is no SMEM tensor to slice.

    Args:
        thr_mma: This thread's slice of the tiled MMA.
        shape_mnk: The ``(M, N, K)`` shape being computed. Only the first two decide the
            accumulator; K sizes the operand fragments.
        sA: A's SMEM tile, or None when the atom sources A from registers. Passing None with an SS
            atom trips the assertion below rather than producing an unpartitioned fragment.
        sB: B's SMEM tile. Required in both cases -- B always comes from SMEM on SM90.
        swap_AB: Build the fragments for ``B @ A``: the accumulator's shape is transposed and the
            two SMEM tiles change roles.

    Returns:
        ``(acc, tCrA, tCrB)`` -- a newly allocated fp32 accumulator and the two operand fragments.

    Raises:
        AssertionError: If a required SMEM tile is None for the atom's operand source.
    """
    is_rs = thr_mma.op.a_src == warpgroup.OperandSource.RMEM
    if const_expr(not swap_AB):
        acc = cute.make_rmem_tensor(thr_mma.partition_shape_C(shape_mnk[:2]), Float32)
        if const_expr(not is_rs):
            assert sA is not None
            tCrA = thr_mma.make_fragment_A(thr_mma.partition_A(sA))
        else:
            tCrA = thr_mma.make_fragment_A(thr_mma.partition_shape_A((shape_mnk[0], shape_mnk[2])))
        assert sB is not None
        tCrB = thr_mma.make_fragment_B(thr_mma.partition_B(sB))
    else:
        acc = cute.make_rmem_tensor(
            thr_mma.partition_shape_C((shape_mnk[1], shape_mnk[0])), Float32
        )
        if const_expr(not is_rs):
            assert sB is not None
            tCrB = thr_mma.make_fragment_A(thr_mma.partition_A(sB))
        else:  # B in rmem
            tCrB = thr_mma.make_fragment_A(thr_mma.partition_shape_A((shape_mnk[1], shape_mnk[2])))
        assert sA is not None
        tCrA = thr_mma.make_fragment_B(thr_mma.partition_B(sA))
    return acc, tCrA, tCrB
