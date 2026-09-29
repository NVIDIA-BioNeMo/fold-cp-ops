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

"""The benchmark tree's name for the timing primitive, which now lives in the package.

The implementation moved to :mod:`fold_cp_ops._internal.bench_timing`. Nothing about the timing
changed; what changed is who can reach it.

**Why it had to move.** ``pyproject.toml`` packages ``fold_cp_ops*`` only, so ``benchmark/`` exists
in a source tree and not in an installed wheel. The autotuner's default timer is
:func:`benchmark_single` -- CLAUDE.md requires it, because a hand-rolled ``time.perf_counter`` loop
around an async launch measures host DISPATCH rather than device time and yields "bandwidths" above
the physical link. A required dependency that disappears on install is not a dependency; it is a
runtime error waiting for the first person to ``pip install`` the package and autotune. So the
primitive is now inside the package and this module re-exports it.

**Why this file remains rather than the imports being rewritten.** Every perf gate, harness target
and benchmark script names it ``bench_utils``, and that name is in the CLAUDE.md rule that governs
them ("benchmark via ``bench_utils`` (timing) + the ``harness`` (multi-cell)"). Re-exporting keeps
one obvious name for benchmark authors while the code sits where an installed package can find it.

The re-exported objects are the SAME objects, not wrappers: ``bench_utils.benchmark_single is
fold_cp_ops._internal.bench_timing.benchmark_single``. A wrapper would be a second place for the
measurement contract to drift, which is the thing this whole layer exists to prevent.
"""

from fold_cp_ops._internal.bench_timing import (
    MIN_COLD_COMPILE_SAMPLES,
    BenchResult,
    ColdCompileComparison,
    PairedResult,
    benchmark_extrapolated,
    benchmark_paired,
    benchmark_single,
    compare_cold_compile,
    host_dispatch_us,
    measure_launch_overhead,
    resolve_dist,
    time_callable,
)

__all__ = [
    "MIN_COLD_COMPILE_SAMPLES",
    "BenchResult",
    "ColdCompileComparison",
    "PairedResult",
    "benchmark_extrapolated",
    "benchmark_paired",
    "benchmark_single",
    "compare_cold_compile",
    "host_dispatch_us",
    "measure_launch_overhead",
    "resolve_dist",
    "time_callable",
]
