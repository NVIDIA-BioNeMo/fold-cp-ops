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


"""Tests for ``fold_cp_ops._internal.sm90_utils`` -- WGMMA operand staging and issue.

Almost everything here needs an MLIR context or a device, so these tests pin the module's
*structure*: which helpers exist, that the epilogue alias is genuinely the same function, and the
swap_AB symmetry that a refactor is most likely to break.
"""

import inspect

import fold_cp_ops._internal.sm90_utils as sm90_utils


def test_make_smem_layout_epi_is_the_same_function_not_a_copy():
    """``epi_utils`` selects between this module and ``blackwell_helpers`` by name.

    The alias is what lets that caller stay architecture-agnostic. A second implementation under the
    same name would let the two drift while every call site still looked right.
    """
    assert sm90_utils.make_smem_layout_epi is sm90_utils.make_smem_layout


def test_the_module_carries_both_halves_of_the_mma_path():
    """Layout staging and instruction issue live together here, deliberately.

    The module is mixed compile-time/runtime -- ``make_smem_layout`` is erased before codegen while
    ``gemm`` emits WGMMA -- and is kept whole because the four functions are always used together.
    """
    for name in (
        "make_smem_layout",
        "partition_for_epilogue",
        "gemm",
        "gemm_zero_init",
        "gemm_w_idx",
        "partition_fragment_ABC",
    ):
        assert hasattr(sm90_utils, name), f"sm90_utils lost {name}"


def test_every_mma_entry_offers_swap_ab():
    """``B @ A`` without transposing memory is available on all three issue paths.

    A caller that has it on one and not another would have to materialize a transpose for the odd
    one out, which is a copy the swap exists to avoid.
    """
    for fn in (
        sm90_utils.gemm,
        sm90_utils.gemm_zero_init,
        sm90_utils.gemm_w_idx,
        sm90_utils.partition_fragment_ABC,
    ):
        assert "swap_AB" in inspect.signature(fn).parameters, fn.__name__


def test_zero_init_differs_between_the_two_issue_paths():
    """``gemm`` takes a Python bool; ``gemm_w_idx`` takes a runtime Boolean.

    That difference is the whole reason both exist: in a persistent kernel "is this the first
    k-tile" is not known at trace time, so the mainloop cannot use the const_expr form.
    """
    assert inspect.signature(sm90_utils.gemm).parameters["zero_init"].default is False
    assert "zero_init" in inspect.signature(sm90_utils.gemm_w_idx).parameters
    assert (
        inspect.signature(sm90_utils.gemm_w_idx).parameters["zero_init"].default
        is inspect.Parameter.empty
    ), (
        "a runtime zero_init must be required, not defaulted -- defaulting it would let a caller "
        "silently accumulate into an uninitialized accumulator"
    )
