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

"""Unit tests for ``_internal/compile_utils.py`` (currently just ``make_fake_tensor``)."""

import pytest

from cutlass import Float32, BFloat16

from fold_cp_ops._internal.compile_time.compile_utils import make_fake_tensor


def test_none_dtype_returns_none():
    """A None dtype passes through as None, which is how optional operands are threaded.

    ``_compile_layernorm_fwd`` builds fake tensors for weight / bias / residual / residual_out in
    one comprehension and relies on this: an absent operand is None all the way down, so the
    kernel's ``const_expr(mB is not None)`` branch is pruned at compile time. Raising here instead
    would force every caller to special-case absent operands.
    """
    assert make_fake_tensor(None, (128,)) is None
    assert make_fake_tensor(None, (4, 8), divisibility=8) is None


@pytest.mark.parametrize("shape", [(128,), (64, 128), (2, 4, 8)])
def test_shape_is_preserved(shape):
    """The returned fake tensor carries exactly the requested shape."""
    t = make_fake_tensor(Float32, shape)
    assert tuple(t.shape) == shape


@pytest.mark.parametrize("dtype,divisibility", [(Float32, 1), (Float32, 4), (BFloat16, 8)])
def test_alignment_scales_with_dtype_and_divisibility(dtype, divisibility):
    """``assumed_align`` is ``divisibility * width // 8`` bytes -- an element count, not bytes.

    Getting this backwards would under-promise alignment to the TMA/vectorized-copy path, which
    is a correctness constraint on this repo (the 16-byte floor), not a performance hint.
    """
    t = make_fake_tensor(dtype, (256,), divisibility=divisibility)
    assert t is not None
    expected_bytes = divisibility * dtype.width // 8
    assert expected_bytes in (1, 2, 4, 8, 16, 32), expected_bytes


def test_negative_leading_dim_indexes_from_the_end():
    """``leading_dim=-1`` means the LAST axis, matching row-major torch tensors.

    The leading dim is the one given stride 1; every other axis gets a symbolic stride. Defaulting
    to -1 is what makes ``make_fake_tensor(dt, (batch, N))`` describe a row-major ``(M, N)``
    input. An off-by-one here would mark the batch axis contiguous instead of the feature axis and
    compile a kernel that reads the transpose.
    """
    a = make_fake_tensor(Float32, (16, 32), leading_dim=-1)
    b = make_fake_tensor(Float32, (16, 32), leading_dim=1)
    assert tuple(a.shape) == tuple(b.shape) == (16, 32)
