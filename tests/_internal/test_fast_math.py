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


"""Tests for ``fold_cp_ops._internal.fast_math`` -- the divisor-carrying fast divmod.

Host-side: what is pinned here is the *reason the subclass exists* (it keeps the divisor) and its
marshalling, not the magic-number arithmetic, which belongs to cutlass.
"""

import cutlass

from fold_cp_ops._internal.fast_math import FastDivmod


def test_fast_divmod_is_a_cute_fast_divmod_divisor():
    """It must remain substitutable wherever the stock divisor is expected."""
    import cutlass.cute as cute

    assert issubclass(FastDivmod, cute.FastDivmodDivisor)


def test_the_subclass_exists_to_keep_the_divisor():
    """The stock class keeps only the magic number, from which the divisor cannot be recovered.

    The tile scheduler needs both -- the reciprocal to divide, the divisor itself to bound a loop --
    so this attribute is the entire point of the subclass. A refactor that dropped it would compile
    and then produce schedulers that walk the wrong number of tiles.
    """
    assert hasattr(FastDivmod, "__extract_mlir_values__")
    assert hasattr(FastDivmod, "__new_from_mlir_values__")


def test_marshalling_carries_the_magic_number_and_the_divisor():
    """Both values cross the boundary: the base's one, plus however many the divisor needs.

    Checked structurally rather than by running -- constructing one requires an MLIR context -- but
    the shape of the contract is what breaks when someone "simplifies" the override away.
    """
    import inspect

    src = inspect.getsource(FastDivmod.__extract_mlir_values__)
    assert "self._divisor" in src and "self.divisor" in src, (
        "extract must emit BOTH the magic number and the divisor"
    )
    src = inspect.getsource(FastDivmod.__new_from_mlir_values__)
    assert "values[0]" in src and "values[1:]" in src, (
        "rebuild must consume the magic number first, then the divisor's own values"
    )
    assert cutlass is not None
