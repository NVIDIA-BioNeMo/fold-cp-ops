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

"""Target-agnostic distributed bench harness (design: benchmark/distributed/HARNESS_DESIGN.md).

Core (this package) imports ZERO kernels — it knows only the BenchTarget protocol + the collective-symmetric
cell protocol. Kernels plug in via harness/targets/<name>.py that build BenchTargets + call register(name, fn).
"""
from benchmark.distributed.harness.target import BenchTarget, Ctx
from benchmark.distributed.harness.registry import REGISTRY, register, resolve
from benchmark.distributed.harness.driver import run_cell, run_matrix, norm_cell

__all__ = ["BenchTarget", "Ctx", "REGISTRY", "register", "resolve", "run_cell", "run_matrix", "norm_cell"]
