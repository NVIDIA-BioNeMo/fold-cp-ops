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


"""Tests for ``fold_cp_ops._internal.rounding`` -- the epilogue's rounding modes.

Host-side. What is pinned is the enum's *contract* with the rest of the tree: ``RoundingMode.RN`` is
the SM90 default that ``GemmSm90.__init__`` installs, and ``RoundingMode.RS`` is the Blackwell mode
``kernels/gemm.gemm`` refuses by name.
"""

from fold_cp_ops._internal.rounding import RoundingMode


def test_rs_is_distinct_and_is_what_the_sm90_entry_refuses():
    """Stochastic rounding must be a different value from RN, or the refusal would reject everything."""
    assert RoundingMode.RS != RoundingMode.RN


def test_modes_are_ints_so_they_can_be_baked_in_as_constexpr():
    """The mode travels as a ``Constexpr[int]`` epilogue argument, so it has to compare as an int."""
    assert isinstance(RoundingMode.RN.value, int)
    assert int(RoundingMode.RN) == RoundingMode.RN
