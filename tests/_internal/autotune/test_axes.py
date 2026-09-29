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

"""Tests for ``fold_cp_ops._internal.autotune.axes`` -- the declared tunable axes.

The point of this module is that the axes can be READ rather than reverse-engineered, so most of
these assert on what a reader (or a tool) can recover from a declaration -- names, domains, and why
a grid is sparse. CPU-only.
"""

import pytest

from fold_cp_ops._internal.autotune.axes import AxisSpace, TuneAxis


def test_the_product_is_generated_in_declaration_order():
    """Deterministic order is what makes the pool identical on every rank."""
    sp = AxisSpace(TuneAxis("a", "first", (1, 2)), TuneAxis("b", "second", (10, 20)))
    assert [c.all_kwargs() for c in sp.configs()] == [
        {"a": 1, "b": 10},
        {"a": 1, "b": 20},
        {"a": 2, "b": 10},
        {"a": 2, "b": 20},
    ]


def test_the_axes_and_their_domains_are_recoverable():
    """The whole gap this module closes: a flat pool cannot answer either question."""
    sp = AxisSpace(TuneAxis("tile_N", "multiple of 16, <= 256", (32, 64)))
    assert [a.name for a in sp.axes] == ["tile_N"]
    assert sp.axes[0].domain == "multiple of 16, <= 256"
    assert "multiple of 16" in sp.describe()


def test_a_sparse_grid_states_why_it_is_sparse():
    """An unexplained hole is indistinguishable from a forgotten combination."""
    sp = AxisSpace(
        TuneAxis("m", "tile M", (64, 128)),
        TuneAxis("n", "tile N", (128, 256)),
        exclude=lambda p: p["m"] == 64 and p["n"] == 256,
        because="exceeds the SMEM budget",
    )
    assert len(sp.configs()) == 3 and sp.excluded_count() == 1
    assert "exceeds the SMEM budget" in sp.describe()


def test_an_exclusion_without_a_reason_is_refused():
    """`because=` is mandatory precisely because the hole is otherwise unreadable."""
    with pytest.raises(ValueError, match=r"requires `because=`"):
        AxisSpace(TuneAxis("m", "tile M", (64, 128)), exclude=lambda p: p["m"] == 64)


def test_an_axis_without_a_domain_is_refused():
    """A pool with no stated claim cannot be judged under-covered."""
    with pytest.raises(ValueError, match=r"non-empty `domain`"):
        TuneAxis("m", "  ", (1, 2))


@pytest.mark.parametrize(
    "kwargs,match",
    [
        (dict(name="m", domain="d", values=()), r"no values"),
        (dict(name="m", domain="d", values=(1, 1)), r"duplicate values"),
    ],
)
def test_a_degenerate_axis_is_refused(kwargs, match):
    """An empty axis has nothing to try; a duplicate is benchmarked twice and can beat itself."""
    with pytest.raises(ValueError, match=match):
        TuneAxis(**kwargs)


def test_duplicate_axis_names_are_refused():
    """Two axes writing the same keyword would silently shadow one another."""
    with pytest.raises(ValueError, match=r"duplicate axis name"):
        AxisSpace(TuneAxis("m", "d", (1,)), TuneAxis("m", "d", (2,)))


def test_an_exclusion_that_empties_the_grid_is_refused():
    """Leaves the kernel with nothing to run -- caught at declaration, not at the first call."""
    with pytest.raises(ValueError, match=r"removed every point"):
        AxisSpace(TuneAxis("m", "d", (1, 2)), exclude=lambda p: True, because="all bad")
