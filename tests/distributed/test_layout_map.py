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

"""Tests for ``fold_cp_ops.distributed.layout_map`` -- the strided layout algebra.

**This file exists because the module had none.** `LayoutMap` was reachable from no test in the
suite: an API audit found `required_span_size` called by nothing anywhere, and following it up
showed the whole module unreferenced by `tests/`. What the class computes is a SIZE that a caller
allocates against, so being wrong is an under-allocated buffer rather than an exception.

The tests here are pure host algebra -- no process group, no device -- even though the package
directory is otherwise torchrun-collected. That is deliberate: nothing in `LayoutMap` is
distributed, and a test that needed four ranks to check an arithmetic identity would be a test
nobody runs.
"""

import itertools

import pytest

from fold_cp_ops.testing.kernel_matrix import matrix_exempt

from fold_cp_ops.testing.collective_guard import rank_invariant_skip

from fold_cp_ops.distributed.layout_map import LayoutLeftMap, LayoutMap, LayoutRightMap


def _brute_force_span(layout: LayoutMap) -> int:
    """The span of a layout, by ENUMERATING every coordinate rather than by formula.

    Purpose
        The independent oracle for :attr:`LayoutMap.required_span_size`. The class computes the
        span in closed form; this computes it by exhaustion, so agreement is evidence rather than
        a restatement.

    Args:
        layout: Any `LayoutMap` whose shape is small enough to enumerate -- the product of its
            extents is iterated in full, so keep test shapes tiny.

    Returns:
        ``max(flat_index) - offset + 1`` over every coordinate, i.e. the number of slots a
        contiguous buffer must have for the layout to be addressable. ``1`` for a rank-0 layout and
        ``0`` when any extent is zero, matching the degenerate cases the closed form special-cases.
    """
    shape = layout.shape
    if len(shape) == 0:
        return 1
    if any(s == 0 for s in shape):
        return 0
    hi = max(layout(ids) for ids in itertools.product(*(range(s) for s in shape)))
    return hi - layout.offset + 1


#: Layouts small enough to enumerate exhaustively, spanning the cases the closed form branches on:
#: a rank-0 layout, a zero extent, singleton axes (which the constructor's tie-breaking comment
#: singles out as the confounder for its uniqueness check), both canonical orders, and a non-zero
#: offset -- because the span must be measured FROM the offset, not from zero.
_CASES = [
    ("right_3d", LayoutRightMap((2, 3, 4))),
    ("left_3d", LayoutLeftMap((2, 3, 4))),
    ("right_singleton", LayoutRightMap((3, 1, 5))),
    ("left_singleton", LayoutLeftMap((3, 1, 5))),
    ("right_1d", LayoutRightMap((7,))),
    ("explicit_offset", LayoutMap((1, 3), (3, 2), offset=11)),
    ("explicit_permuted", LayoutMap((4, 1), (2, 4), offset=0)),
]


@matrix_exempt(
    "pure host LAYOUT algebra -- the subject is a LayoutRightMap's index arithmetic, which has no kernel, no device mesh and no dtype; the cases sweep layout shapes, which are the axis, and there is no second module that would need to relate to them"
)
@pytest.mark.parametrize("label,layout", _CASES, ids=[c[0] for c in _CASES])
def test_the_required_span_matches_an_exhaustive_enumeration(label, layout):
    """The closed-form span equals the largest flat index the layout can produce, plus one.

    **This is the property a caller allocates against**, so an over-estimate wastes a symmetric
    buffer and an under-estimate is an out-of-bounds write into whatever follows it -- neither of
    which raises anything at the point of the mistake.

    The oracle enumerates every coordinate instead of re-deriving the formula, which is what makes
    this a check rather than a tautology: the two agree only if
    ``1 + sum((shape[i] - 1) * strides[i])`` really is the maximum of ``sum(ids[i] * strides[i])``.

    ``offset`` is crossed deliberately. The span is a LENGTH measured from the offset, not an
    address, so a formula that leaked the offset into it would pass every zero-offset case and
    over-allocate by exactly the offset everywhere else.

    Args:
        label: The case name, for the failure message.
        layout: The layout under test.
    """
    assert layout.required_span_size == _brute_force_span(layout), (
        f"{label}: closed-form span {layout.required_span_size} disagrees with the enumerated "
        f"maximum for shape={layout.shape} strides={layout.strides} offset={layout.offset}"
    )


@matrix_exempt(
    "pure host LAYOUT algebra -- the subject is a LayoutRightMap's index arithmetic, which has no kernel, no device mesh and no dtype; the cases sweep layout shapes, which are the axis, and there is no second module that would need to relate to them"
)
@pytest.mark.parametrize("label,layout", _CASES, ids=[c[0] for c in _CASES])
def test_every_coordinate_lands_inside_the_span_it_reports(label, layout):
    """No coordinate maps outside ``[offset, offset + required_span_size)``.

    The complement of the test above: that one pins the span to the MAXIMUM, this one pins that
    nothing escapes it at all -- including below the offset, which a negative stride would allow
    and which the constructor is supposed to have already refused.

    Args:
        label: The case name.
        layout: The layout under test.
    """
    if len(layout.shape) == 0 or any(s == 0 for s in layout.shape):
        rank_invariant_skip(
            "degenerate layout has no coordinates to enumerate",
            because="the layout comes from a parametrized case, so every rank is handed the same "
            "one; this module is pure host algebra and issues no collective at all",
        )
    lo, span = layout.offset, layout.required_span_size
    for ids in itertools.product(*(range(s) for s in layout.shape)):
        flat = layout(ids)
        assert lo <= flat < lo + span, (
            f"{label}: coordinate {ids} maps to {flat}, outside [{lo}, {lo + span}) -- a caller "
            f"sizing an allocation from required_span_size would write out of bounds"
        )


@matrix_exempt(
    "pure host LAYOUT algebra -- the subject is a LayoutRightMap's index arithmetic, which has no kernel, no device mesh and no dtype; the cases sweep layout shapes, which are the axis, and there is no second module that would need to relate to them"
)
def test_a_rank_zero_layout_spans_one_slot_and_a_zero_extent_is_refused_outright():
    """Rank 0 spans one slot; a zero extent never reaches the span at all.

    **A zero extent is refused by the CONSTRUCTOR**, so
    ``_compute_required_span_size``'s ``if self._has_zero_shape: return 0`` is unreachable: a
    zero-extent shape trips the shape check first (or, through `LayoutRightMap`, the stride check,
    because the cumulative product puts a 0 in the strides). Both refusals are `ValueError`. This
    test pins the reachable contract rather than the branch, because asserting a span of 0 would
    require constructing something the class does not permit.

    Rank 0 is the case that IS reachable and is easy to get backwards -- it spans ONE slot, not
    zero, because a rank-0 layout addresses exactly its offset.
    """
    assert LayoutRightMap(()).required_span_size == 1
    assert LayoutMap((), ()).required_span_size == 1
    with pytest.raises(ValueError, match=r"zero values"):
        LayoutMap((3, 1), (0, 3))
    with pytest.raises(ValueError, match=r"non-integer or negative values"):
        LayoutRightMap((2, 0, 3))
