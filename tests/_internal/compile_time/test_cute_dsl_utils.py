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

"""Unit tests for ``_internal/cute_dsl_utils.py`` (currently just ``torch2cute_dtype_map``)."""

import torch

import cutlass

from fold_cp_ops._internal.compile_time.cute_dsl_utils import torch2cute_dtype_map


def test_dtype_map_entries_are_correct():
    """Each torch dtype maps to the cutlass type of the same width and kind.

    A wrong entry here is silent and catastrophic: the kernel would be compiled for one element
    width and handed a buffer of another, reading past the end of every row.

    The two fp8 entries are what let ``kernels/gemm.py`` reach ``GemmSm90``'s 8-bit WGMMA path. They
    are asserted by exact identity rather than by width, because ``Float8E4M3FN`` and
    ``Float8E4M3`` are both 8-bit and only the former matches torch's ``float8_e4m3fn``: a mix-up
    would compile a kernel whose NaN and saturation behaviour differs from the caller's tensor.
    """
    assert torch2cute_dtype_map == {
        torch.float16: cutlass.Float16,
        torch.bfloat16: cutlass.BFloat16,
        torch.float32: cutlass.Float32,
        torch.int32: cutlass.Int32,
        torch.int64: cutlass.Int64,
        torch.float8_e4m3fn: cutlass.Float8E4M3FN,
        torch.float8_e5m2: cutlass.Float8E5M2,
    }


def test_dtype_map_widths_match_torch():
    """The mapped cutlass type's ``.width`` (bits) equals torch's element size (bytes x 8).

    ``get_copy_atom`` divides 128 by exactly this ``.width`` to size a vectorized access, so a
    width that disagrees with the real element size silently mis-sizes every copy.
    """
    for torch_dt, cute_dt in torch2cute_dtype_map.items():
        assert cute_dt.width == torch.empty(0, dtype=torch_dt).element_size() * 8, torch_dt


def test_map_covers_every_dtype_layernorm_accepts():
    """The map must cover exactly the float dtypes ``layernorm_fwd`` admits.

    ``layernorm_fwd`` asserts ``x.dtype in [float16, bfloat16, float32]`` and then indexes this
    map. A dtype accepted by the assert but missing from the map would raise a bare ``KeyError``
    from inside the custom op instead of the actionable message the assert was written to give.
    """
    for dt in (torch.float16, torch.bfloat16, torch.float32):
        assert dt in torch2cute_dtype_map, f"{dt} is accepted by layernorm_fwd but unmapped"
