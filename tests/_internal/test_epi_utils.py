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


"""Tests for ``fold_cp_ops._internal.epi_utils`` -- shared epilogue helpers.

``setup_epi_tensor`` needs a kernel object and an MLIR context; the stride-assumption helpers do
not, and they are where the interesting contract is.
"""

import inspect

from fold_cp_ops._internal import epi_utils


def test_assume_stride_divisibility_passes_none_through():
    """An absent broadcast vector must survive the helper unchanged, so call sites need no guard."""
    assert epi_utils.assume_stride_divisibility(None) is None


def test_assume_broadcast_strides_preserves_none_positionally():
    """The list is unpacked positionally by callers, so a dropped None would shift every later one."""
    assert epi_utils.assume_broadcast_strides(None, None) == [None, None]


def test_the_assumption_is_a_promise_not_a_check():
    """Nothing here validates the strides -- ``gemm()`` does, at the front door.

    Worth pinning because the name reads like a check. A caller that bypasses the host entry and
    breaks the promise gets a misaligned access rather than a diagnostic, and the only place that
    can be prevented is the entry point.
    """
    src = inspect.getsource(epi_utils.assume_stride_divisibility)
    assert "cute.assume" in src
    assert "raise" not in src and "assert" not in src


def test_setup_epi_tensor_defaults_the_tile_to_the_kernel_s():
    """A supplemental epilogue tensor shares D's subtile unless it explicitly needs another."""
    sig = inspect.signature(epi_utils.setup_epi_tensor)
    assert sig.parameters["epi_tile"].default is None
    assert sig.parameters["op_type"].default == "store"
