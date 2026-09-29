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


"""Tests for ``fold_cp_ops._internal.gemm_tvm_ffi_utils`` -- operand description and validation.

Host-side and GPU-free. The two ``check_*`` functions are the interesting half: they are the front
door for the only shape constraint this GEMM has, and every case here is one that previously
surfaced as a TVM-FFI argument error naming an internal argument index.
"""

import pytest
import torch

from fold_cp_ops._internal.gemm_tvm_ffi_utils import (
    check_broadcast_alignment,
    check_tma_alignment,
    describe_operands,
    div_for_dtype,
    get_dtypes,
    get_major,
    get_majors,
    perm3d,
    perm3d_single,
)


@pytest.mark.parametrize(
    "dtype,expected",
    [(torch.float32, 4), (torch.bfloat16, 8), (torch.float16, 8), (torch.float8_e4m3fn, 16)],
)
def test_div_for_dtype_is_sixteen_bytes_in_elements(dtype, expected):
    """The divisibility floor is 16 bytes expressed in elements, so it narrows as the type widens."""
    from fold_cp_ops._internal.compile_time.cute_dsl_utils import torch2cute_dtype_map

    assert div_for_dtype(torch2cute_dtype_map[dtype]) == expected
    assert expected * torch2cute_dtype_map[dtype].width == 128


def test_perm3d_moves_the_batch_axis_last_and_leaves_2d_alone():
    """3-D operands are permuted ``(L, *, *) -> (*, *, L)``; anything not 3-D passes through."""
    A = torch.zeros(2, 4, 8)
    flat = torch.zeros(4, 8)
    assert perm3d(A, A, A, None)[0].shape == (4, 8, 2)
    assert perm3d(flat, A, A, None)[0].shape == (4, 8)
    assert perm3d(A, A, A, None)[3] is None, "an absent C stays absent"


def test_perm3d_permutes_all_four_operands_the_same_way():
    """Every operand takes the same rule, so no caller has to remember a per-operand exception."""
    A = torch.zeros(2, 4, 8)
    a, b, d, c = perm3d(A, A, A, A)
    assert a.shape == b.shape == d.shape == c.shape == (4, 8, 2)


def test_perm3d_single_returns_a_view_not_a_copy():
    """The permute must not copy: the caller keeps writing through the ORIGINAL tensor's storage.

    A copy here would make the kernel read stale operands and write into a buffer the caller never
    sees -- silently, since the shapes would still be right.
    """
    A = torch.zeros(2, 4, 8)
    out = perm3d_single(A)
    assert out.shape == (4, 8, 2)
    assert out.data_ptr() == A.data_ptr(), "perm3d_single must return a view"
    assert perm3d_single(None) is None
    flat = torch.zeros(4, 8)
    assert perm3d_single(flat) is flat, "a non-3-D tensor passes through untouched"


def test_get_major_reads_dim_one_and_falls_back_to_dim_zero():
    """Dim 1 is tested; dim 0 is the fallback, which is why a doubly-strided tensor must be refused first."""
    row = torch.zeros(4, 8)  # stride (8, 1)
    col = torch.zeros(8, 4).T  # stride (1, 8)
    assert get_major(row, "m", "k") == "k"
    assert get_major(col, "m", "k") == "m"


def test_get_majors_reports_all_four_and_none_for_absent_c():
    """Each operand gets its own pair of names; C reports None rather than a default.

    The input must be the **permuted** form -- ``get_major`` reads ``stride(1)``, which is the batch
    stride on an unpermuted ``(L, M, K)`` tensor and reports the wrong major without complaint. That
    is a real trap, so both forms are asserted here rather than only the correct one.
    """
    dense = torch.zeros(2, 4, 8)
    A_p, B_p, D_p, _ = perm3d(dense, dense, dense, None)
    assert get_majors(A_p, B_p, D_p, None) == ("k", "k", "n", None)
    assert get_majors(A_p, B_p, D_p, D_p)[3] == "n"
    assert get_majors(dense, dense, dense, None) == ("m", "n", "m", None), (
        "an UNPERMUTED tensor is mis-reported, silently -- which is why the docstring says permuted"
    )


def test_get_dtypes_maps_all_four():
    """The map is indexed directly, so an unmapped dtype is a KeyError -- callers check first."""
    import cutlass

    A = torch.zeros(1, dtype=torch.bfloat16)
    D = torch.zeros(1, dtype=torch.float32)
    assert get_dtypes(A, A, D, None) == (cutlass.BFloat16, cutlass.BFloat16, cutlass.Float32, None)
    with pytest.raises(KeyError):
        get_dtypes(torch.zeros(1, dtype=torch.int8), A, D, None)


def test_check_tma_alignment_accepts_the_shapes_that_actually_work():
    """M and the batch are unconstrained; N and K only need the 16-byte floor. Measured, not assumed."""
    for m, n in [(1, 8), (7, 64), (301, 200), (1001, 1000)]:
        check_tma_alignment("A", torch.zeros(m, n))  # fp32: floor is 4
    check_tma_alignment("C", None)  # an absent operand needs no guard at the call site


@pytest.mark.parametrize(
    "shape,dtype,expected",
    [
        ((128, 201), torch.bfloat16, r"16-byte alignment floor"),
        ((128, 129), torch.bfloat16, r"not multiples of 8 elements"),
        ((128, 4), torch.float8_e4m3fn, r"not multiples of 16 elements"),
    ],
)
def test_check_tma_alignment_names_the_extent_to_pad(shape, dtype, expected):
    """The refusal says which operand, which stride, and what multiple -- not an argument index."""
    with pytest.raises(ValueError, match=expected):
        check_tma_alignment("D", torch.zeros(*shape, dtype=dtype))


def test_check_tma_alignment_refuses_a_tensor_contiguous_in_neither_matrix_axis():
    """A TMA descriptor addresses a strided box, so one axis must have unit stride.

    Checked on ``strides[:2]`` and not the trailing pair: after ``perm3d`` the matrix axes are dims
    0 and 1, so an m-major operand -- whose stride 1 is at dim 0 -- must still pass.
    """
    doubly_strided = torch.zeros(64, 64)[::2, ::2]
    with pytest.raises(ValueError, match="contiguous along one of its two matrix axes"):
        check_tma_alignment("A", doubly_strided)
    m_major = torch.zeros(8, 128).T  # stride (1, 128): unit stride at dim 0
    check_tma_alignment("A", m_major)


def test_check_broadcast_alignment_imposes_four_elements_not_sixteen_bytes():
    """The bias load is 4-element vectorized regardless of dtype, so the floor is a literal 4.

    That is why a ``(l, m)`` column bias constrains M -- an extent otherwise entirely free.
    """
    check_broadcast_alignment("colvec_bias", torch.zeros(2, 304))
    check_broadcast_alignment("colvec_bias", torch.zeros(301))  # 1-D: no non-unit stride
    check_broadcast_alignment("rowvec_bias", None)
    with pytest.raises(ValueError, match=r"colvec_bias has stride"):
        check_broadcast_alignment("colvec_bias", torch.zeros(2, 301))


def test_describe_operands_agrees_with_the_functions_it_fuses():
    """The fused pass must return exactly what ``get_majors`` and ``get_dtypes`` do separately.

    It exists for launch latency: asking for the majors, the dtypes and the checks separately
    indexed the dtype map 13 times per launch and walked the strides twice, which is ~10% of a
    launch-bound GEMM. The risk that buys is a divergence nothing else would catch, so it is
    checked against both originals rather than against a hand-written expectation.
    """
    dense = torch.zeros(2, 4, 8, dtype=torch.bfloat16)
    A_p, B_p, D_p, _ = perm3d(dense, dense, dense, None)
    majors, dtypes = describe_operands(A_p, B_p, D_p, None)
    assert majors == get_majors(A_p, B_p, D_p, None)
    assert dtypes == get_dtypes(A_p, B_p, D_p, None)
    majors, dtypes = describe_operands(A_p, B_p, D_p, D_p)
    assert majors == get_majors(A_p, B_p, D_p, D_p)
    assert dtypes == get_dtypes(A_p, B_p, D_p, D_p)
    m_major = torch.zeros(8, 128, dtype=torch.bfloat16).T
    assert describe_operands(m_major, m_major, m_major, None)[0] == ("m", "n", "m", None)


def test_describe_operands_raises_where_check_tma_alignment_would():
    """Same refusals, same messages -- the checks are fused in, not weakened."""
    bad_align = torch.zeros(128, 201, dtype=torch.bfloat16)
    ok = torch.zeros(128, 128, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match=r"D violates the 16-byte alignment floor"):
        describe_operands(ok, ok, bad_align, None)
    with pytest.raises(ValueError, match=r"A must be contiguous"):
        describe_operands(torch.zeros(64, 64, dtype=torch.bfloat16)[::2, ::2], ok, ok, None)


def test_describe_operands_checks_the_operand_WIDTH_only_on_a_and_b():
    """A and B must be 16- or 8-bit; D and C may be fp32, and often are.

    The asymmetry is the point -- fp32 is a legitimate output and bias type, so a check applied to
    all four would refuse calls that work.
    """
    fp32 = torch.zeros(128, 128, dtype=torch.float32)
    bf16 = torch.zeros(128, 128, dtype=torch.bfloat16)
    describe_operands(bf16, bf16, fp32, fp32)  # fp32 D and C: fine
    with pytest.raises(ValueError, match=r"unsupported operand dtype torch.float32 for A"):
        describe_operands(fp32, bf16, bf16, None)
    with pytest.raises(ValueError, match=r"unsupported operand dtype torch.float32 for B"):
        describe_operands(bf16, fp32, bf16, None)


def test_describe_operands_reports_none_for_an_absent_operand():
    """An absent C must not be validated, and must get None for BOTH its major and its dtype."""
    ok = torch.zeros(128, 128, dtype=torch.bfloat16)
    majors, dtypes = describe_operands(ok, ok, ok, None)
    assert majors[3] is None and dtypes[3] is None
