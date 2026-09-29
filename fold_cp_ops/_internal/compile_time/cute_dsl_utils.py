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

# Copyright (c) 2025, Tri Dao.
# This file has been modified by NVIDIA CORPORATION & AFFILIATES.

"""CuTe-DSL type mappings.

The one thing left in this module is ``torch2cute_dtype_map``: torch dtype -> cutlass numeric type,
consulted by every host entry that has to describe a caller's tensor to the DSL. It is a plain dict
lookup, so an unmapped dtype raises ``KeyError`` -- entries here are therefore the *definition* of
which element types a public entry accepts, and a front-door check that wants to say something
better has to run before the lookup.

Upstream's other contents live elsewhere in this tree:

* ``get_device_capacity`` / ``require_sm90`` / ``check_arch_supported`` -> ``_internal/arch.py``,
  with the SM90 capability boundary they enforce.
* ``ParamsBase`` / ``mlir_namedtuple`` / the TVM-FFI converter patch -> ``_internal/runtime_params``,
  the complement of ``compile_time/template_params``.
* ``StaticTypes`` -> ``compile_time/template_params.py``.
"""

import torch

import cutlass
from cutlass import Int32, Int64, Float16, BFloat16, Float32


#: torch dtype -> cutlass numeric type. The fp8 entries are what let ``kernels/gemm.py`` reach
#: ``GemmSm90``'s 8-bit WGMMA path; note the kernel accepts fp8 only in the **k-major** layout and
#: refuses anything else at its front door.
torch2cute_dtype_map = {
    torch.float16: Float16,
    torch.bfloat16: BFloat16,
    torch.float32: Float32,
    torch.int32: Int32,
    torch.int64: Int64,
    torch.float8_e4m3fn: cutlass.Float8E4M3FN,
    torch.float8_e5m2: cutlass.Float8E5M2,
}
