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

"""fold-cp-ops — A2A-fused distributed TriMul kernels and their cp=1 single-device fallback.

Deliberately does NOT eagerly import any kernel module.  Importing the package must stay cheap and
side-effect-free: the kernel modules pull in cutlass / CuTe-DSL and register global `torch.library`
ops, so an eager import would make `import fold_cp_ops` fail on a machine with no CUDA and would
drag the whole dependency tree in for a caller who only wanted the version.

There are NO public API re-exports here, and this paragraph used to claim three
(`trimul_autotuned`, `FusedTriMulCP`, `fused_trimul_dtensor`). None of those names exists anywhere
in the tree, so all three spellings raise `ImportError`. Import from the defining module instead:
`fold_cp_ops.distributed.workflows.trimul_autotuned` exports exactly `TriangularMultiplication`
and `trimul_a2a`.

Hardware: SM90 (H100/H200) only — see CLAUDE.md.
"""

from fold_cp_ops._version import __version__

__all__ = ["__version__"]
