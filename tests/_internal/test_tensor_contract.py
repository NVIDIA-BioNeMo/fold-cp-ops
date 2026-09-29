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
"""Unit tests for `fold_cp_ops._internal.tensor_contract`.

The shape check carries a fast path -- one C-level tuple compare in front of the per-axis loop --
added because the front door makes ~13 shape checks per call and this was the priciest of them.
A fast path is only safe while it is INDISTINGUISHABLE from the loop it skips, and nothing else in
the suite pins that: `check_tensor`'s behaviour is exercised incidentally by the kernel front doors,
which pass valid tensors and therefore never take the failing branches at all.

These tests exist to make the two paths' equivalence a property the suite enforces, so a future
edit to either one fails here rather than in a kernel whose error message quietly changed.
"""

import pytest
import torch

from fold_cp_ops._internal.tensor_contract import check_tensor, supported_dtypes


def _t(*shape, dtype=torch.bfloat16):
    """A CPU tensor of the given shape; `check_tensor` inspects metadata only, never device."""
    return torch.empty(shape, dtype=dtype)


def test_none_is_accepted_so_an_optional_argument_needs_no_guard_at_the_call_site():
    assert check_tensor("bg", None, expect_shape=(128,), expect_width=2, align_elems=8) is None


@pytest.mark.parametrize(
    "expect_shape",
    [(4, 8), (None, 8), (4, None), (None, None)],
    ids=["fully-specified", "free-rows", "free-cols", "all-free"],
)
def test_a_matching_shape_passes_whether_or_not_the_expectation_has_free_extents(expect_shape):
    """The fast path fires only for `(4, 8)`; the other three fall to the loop and must agree."""
    assert check_tensor("x", _t(4, 8), expect_shape=expect_shape) is None


@pytest.mark.parametrize(
    "expect_shape, axis, want, got",
    [((5, 8), 0, 5, 4), ((4, 9), 1, 9, 8), ((None, 9), 1, 9, 8)],
    ids=["axis0", "axis1", "axis1-with-free-axis0"],
)
def test_a_mismatched_extent_names_the_first_disagreeing_axis(expect_shape, axis, want, got):
    """The message is the contract: the fast path must not change WHICH axis is reported."""
    with pytest.raises(ValueError, match=rf"disagrees on axis {axis} \({got} vs {want}\)"):
        check_tensor("x", _t(4, 8), expect_shape=expect_shape)


def test_the_first_disagreeing_axis_is_reported_when_several_disagree():
    """Two bad axes, and axis 0 must win -- the loop is ordered and the fast path must not reorder."""
    with pytest.raises(ValueError, match=r"disagrees on axis 0 \(4 vs 9\)"):
        check_tensor("x", _t(4, 8), expect_shape=(9, 9))


@pytest.mark.parametrize("expect_shape", [(4,), (4, 8, 1)], ids=["too-few", "too-many"])
def test_a_rank_mismatch_is_refused_before_any_extent_is_compared(expect_shape):
    """Rank is checked first, so a rank error must never surface as an axis error."""
    with pytest.raises(ValueError, match=r"must be \d+-D"):
        check_tensor("x", _t(4, 8), expect_shape=expect_shape)


def test_rank_is_still_enforced_when_every_extent_is_free():
    """`(None, None)` constrains rank and nothing else -- the case a bare `*` reading would miss."""
    with pytest.raises(ValueError, match=r"must be 2-D"):
        check_tensor("x", _t(4, 8, 2), expect_shape=(None, None))


def test_an_unsupported_dtype_is_refused_and_the_message_lists_what_is_supported():
    with pytest.raises(ValueError, match=r"unsupported dtype for x: torch.int8"):
        check_tensor("x", _t(4, 8, dtype=torch.int8))
    assert "torch.bfloat16" in supported_dtypes()


def test_an_exact_dtype_requirement_refuses_a_merely_supported_one():
    """`norm_weight` is read in fp32; bf16 would be a silent precision loss, not a conversion."""
    with pytest.raises(ValueError, match=r"norm_weight must be torch.float32"):
        check_tensor("norm_weight", _t(8), expect_dtype=torch.float32)


def test_an_operand_width_requirement_refuses_a_wider_supported_dtype():
    with pytest.raises(ValueError, match=r"Wg must be 16-bit"):
        check_tensor("Wg", _t(4, 8, dtype=torch.float32), expect_width=2)


@pytest.mark.parametrize("trailing, ok", [(8, True), (16, True), (12, False), (9, False)])
def test_alignment_is_checked_on_the_trailing_extent_only(trailing, ok):
    """The 16-byte floor is this package's ONLY shape constraint, and it binds the last axis."""
    if ok:
        assert check_tensor("x", _t(3, trailing), align_elems=8) is None
    else:
        with pytest.raises(ValueError, match=r"violates the 16-byte alignment floor"):
            check_tensor("x", _t(3, trailing), align_elems=8)


def test_alignment_is_reached_only_after_the_shape_check_passes():
    """Both rules are violated; the shape message must win, because extents are checked first."""
    with pytest.raises(ValueError, match=r"disagrees on axis 1"):
        check_tensor("x", _t(3, 12), expect_shape=(3, 16), align_elems=8)


def test_a_list_expectation_behaves_the_same_as_a_tuple():
    """A list can never satisfy the tuple fast path, so it must fall through to the loop intact."""
    assert check_tensor("x", _t(4, 8), expect_shape=[4, 8]) is None
    with pytest.raises(ValueError, match=r"disagrees on axis 1"):
        check_tensor("x", _t(4, 8), expect_shape=[4, 9])
