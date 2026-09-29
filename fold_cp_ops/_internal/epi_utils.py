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
"""Epilogue utilities: shared helpers for epilogue mixin classes."""

import cutlass
import cutlass.cute as cute
import cutlass.utils.blackwell_helpers as sm100_utils

import fold_cp_ops._internal.sm90_utils as sm90_utils


def assume_stride_divisibility(tensor):
    """Assume all strides are divisible by 32 bits (except static strides).

    Used for broadcast vectors and similar tensors where stride alignment is guaranteed.
    Returns a new tensor with the assumed strides.
    """
    if tensor is None:
        return None
    new_stride = tuple(
        cute.assume(s, divby=32 // tensor.element_type.width) if not cute.is_static(s) else s
        for s in tensor.stride
    )
    return cute.make_tensor(tensor.iterator, cute.make_layout(tensor.shape, stride=new_stride))


def assume_broadcast_strides(*tensors):
    """Apply stride divisibility assumptions to multiple broadcast vectors.

    Returns a list with None preserved for None inputs.
    """
    return [assume_stride_divisibility(t) for t in tensors]


def setup_epi_tensor(gemm, tensor, epi_tile=None, op_type="store"):
    """Create TMA atom + smem layout for a supplemental epilogue tensor.

    Args:
        gemm: The GEMM object. Must already have run ``_setup_attributes`` -- ``epi_stage`` and
            ``epi_tile`` are read here, and are None before that.
        tensor: The global-memory tensor to build a descriptor for. Must be rank 3 ``(x, y, l)``
            with a 16-byte-aligned pitch and base address; TMA has no way to report a violation,
            it faults or reads the wrong window.
        epi_tile: Epilogue subtile shape. Defaults to ``gemm.epi_tile``. Must divide the CTA tile
            the epilogue partitions against, or the SMEM box and the register fragment disagree.
        op_type: ``"store"``, ``"add"`` or ``"load"`` -- which TMA reduction op the atom carries.

    Returns:
        ``(tma_atom, tma_tensor, smem_layout_staged, epi_tile)``. ``tma_tensor`` is the descriptor's
        view of ``tensor``, which is what the kernel must partition against -- partitioning the
        original instead gives coordinates the descriptor does not understand.
    """
    if epi_tile is None:
        epi_tile = gemm.epi_tile
    dtype = tensor.element_type
    layout = cutlass.utils.LayoutEnum.from_tensor(tensor)
    utils_cls = sm100_utils if gemm.arch >= 100 else sm90_utils
    smem_layout_staged = utils_cls.make_smem_layout_epi(dtype, layout, epi_tile, gemm.epi_stage)
    tma_atom, tma_tensor = gemm._make_tma_epi_atoms_and_tensors(
        tensor,
        smem_layout_staged,
        epi_tile,
        op_type=op_type,
    )
    return tma_atom, tma_tensor, smem_layout_staged, epi_tile
