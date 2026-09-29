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

"""Layout-algebra helpers (CuTe-DSL).

Three helpers so far: ``expand`` for the LayerNorm broadcast views, and ``concat_layout`` /
``convert_layout_zero_stride`` for the GEMM epilogue's broadcast-vector partitioning. The upstream
module carries 14 more (accumulator relayouts, ldmatrix/stmatrix register permutes, vectorized
partition helpers); they return with the kernels that need them.

Everything here is pure layout algebra -- shapes and strides in, a ``!cute.layout`` out -- so it is
erased before codegen. Nothing allocates, nothing copies, and every result shares its input's
iterator.
"""

import cutlass.cute as cute
from cutlass import Int32, const_expr


def transpose_view(a: cute.Tensor) -> cute.Tensor:
    """Swap a tensor's first two modes, as a pure view.

    Purpose
        Lets one SMEM allocation be addressed in both orientations: a staging tile written
        ``[n, k]`` by the loading thread and read ``[k, n]`` by the storing thread is a
        transpose done in shared memory, which is what keeps BOTH the global read and the
        global write coalesced.

    Semantics
        Composition with a reordered layout -- no allocation, no copy, no instruction. The
        result aliases ``a``'s memory, so a write through the view is a write to ``a``. Modes
        past the first two are carried through unchanged.

    Args:
        a: Source tensor of rank >= 2. Typically an SMEM tensor; nothing here requires that,
            but transposing a GMEM tensor this way yields an uncoalesced access pattern rather
            than an error. Not modified -- the result shares its iterator.

    Returns:
        A view of ``a`` with modes 0 and 1 exchanged.
    """
    shape = (a.shape[1], a.shape[0], *a.shape[2:])
    order = (1, 0, *range(2, cute.rank(a)))
    return cute.composition(a, cute.make_ordered_layout(shape, order=order))


def expand(a: cute.Tensor, dim: int, size: Int32 | int) -> cute.Tensor:
    """Insert a broadcast (stride-0) mode of extent ``size`` at position ``dim``.

    A zero stride means every index along the new mode aliases the same memory, so this is a
    pure view -- no allocation, no copy. The LayerNorm kernel uses it twice: to broadcast the
    per-feature ``(N,)`` weight/bias across the tile's row mode, and to give the per-row
    ``(M,)`` rstd/mean outputs a column mode so they partition like the data tensor.

    Args:
        a: Source tensor. Not modified; the result shares its iterator.
        dim: Position at which to insert the new mode, in ``[0, rank(a)]``.
        size: Extent of the new mode. May be a runtime ``Int32``.

    Returns:
        A view of ``a`` with rank ``rank(a) + 1`` and stride 0 at ``dim``.
    """
    shape = (*a.shape[:dim], size, *a.shape[dim:])
    stride = (*a.layout.stride[:dim], 0, *a.layout.stride[dim:])
    return cute.make_tensor(a.iterator, cute.make_layout(shape, stride=stride))


def concat_layout(*layouts: cute.Layout) -> cute.Layout:
    """Nest several layouts side by side into one hierarchical layout.

    The result has one mode per input, each keeping its own shape and stride tuple. This is
    concatenation in the *hierarchical* sense -- rank goes to ``len(layouts)``, not to the sum of
    the input ranks -- which is what lets a partitioned tile and a broadcast vector be indexed by
    one coordinate.

    Args:
        *layouts: One or more layouts. Passing none produces a rank-0 layout, which is legal but
            almost certainly a caller bug. Extents and strides are used as-is; nothing is checked
            for compatibility, because there is nothing to be compatible with -- the modes are
            independent.

    Returns:
        A layout of rank ``len(layouts)`` whose mode ``i`` is ``layouts[i]``.
    """
    return cute.make_layout(
        tuple(l.shape for l in layouts),
        stride=tuple(l.stride for l in layouts),
    )


def convert_layout_zero_stride(
    input: cute.Tensor | cute.Layout, ref_layout: cute.Layout
) -> cute.Layout | cute.Tensor:
    """Regroup modes into ``(non-broadcast, broadcast)`` according to a reference layout.

    A broadcast operand -- a row vector added to every row of a tile -- is partitioned across
    threads with stride 0 along the broadcast modes. Iterating it naively would re-read the same
    element once per broadcast position. Splitting the modes into two groups, ordered by whether
    ``ref_layout`` gives them a zero stride, lets the epilogue loop over the first group only and
    read each distinct value exactly once.

    Args:
        input: The layout to regroup, or a tensor whose layout to regroup. When a tensor is passed a
            tensor is returned, sharing the same iterator -- this is a view, never a copy.
        ref_layout: The layout whose strides decide the grouping. Must have the same **flattened**
            rank as ``input``: the two are walked position by position, so a rank mismatch is an
            ``IndexError`` here and, worse, a silent mis-grouping if the ranks happen to line up
            while the modes do not correspond.

    Returns:
        Same kind as ``input`` (tensor in -> tensor out, layout in -> layout out), with rank exactly
        2: mode 0 gathers the modes ``ref_layout`` strides non-zero, mode 1 gathers the rest.

    Note:
        The all-broadcast case (every reference stride zero) would otherwise produce an empty first
        mode; it is given the degenerate ``shape (1,), stride (0,)`` instead, so downstream code can
        assume mode 0 exists and iterate it once.
    """
    layout = input.layout if const_expr(isinstance(input, cute.Tensor)) else input
    # Group the modes with non-zero stride in the ref_layout together,
    # and the modes with zero stride together
    layout_flat = cute.flatten(layout)
    ref_layout_flat = cute.flatten(ref_layout)
    nonzero_modes = [i for i in range(cute.rank(layout_flat)) if ref_layout_flat[i].stride != 0]
    zero_modes = [i for i in range(cute.rank(layout_flat)) if ref_layout_flat[i].stride == 0]
    # There's an edge case when all modes are zero stride
    new_shape = (
        tuple(layout_flat[i].shape for i in nonzero_modes) if len(nonzero_modes) > 0 else (1,),
        tuple(layout_flat[i].shape for i in zero_modes),
    )
    new_stride = (
        tuple(layout_flat[i].stride for i in nonzero_modes) if len(nonzero_modes) > 0 else (0,),
        tuple(layout_flat[i].stride for i in zero_modes),
    )
    out_layout = cute.make_layout(new_shape, stride=new_stride)
    if const_expr(isinstance(input, cute.Tensor)):
        return cute.make_tensor(input.iterator, out_layout)
    else:
        return out_layout
