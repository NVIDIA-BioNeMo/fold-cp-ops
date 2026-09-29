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

"""``benchmark/distributed/bench_utils.py`` is a re-export shim; this proves it re-exports.

The measured behaviour of the timer is tested in ``tests/_internal/test_bench_timing.py``, next to
where it now lives. What is left here is the *shim's own* contract, and it is worth a file because
the failure it guards is quiet: if the re-export were a wrapper rather than an alias, the perf
gates and the autotuner would be calling two different objects, and a change to the measurement
contract could land in one and not the other.

The other half is coverage. Every perf gate says ``from benchmark.distributed import bench_utils``,
and CLAUDE.md names ``bench_utils`` as the timing life-line. If that import ever broke, the gates
would fail at collection with an ImportError far from the cause -- this fails first, with the reason.
"""

import pytest

from benchmark.distributed import bench_utils as BU
from fold_cp_ops._internal import bench_timing


@pytest.mark.parametrize(
    "name",
    [
        "benchmark_single",
        "benchmark_paired",
        "time_callable",
        "resolve_dist",
        "compare_cold_compile",
    ],
)
def test_the_shim_re_exports_the_same_object_not_a_wrapper(name):
    """``bench_utils.X is bench_timing.X`` -- identity, not equality.

    A wrapper would be a second place for the measurement contract to drift. Identity is what makes
    "the perf gates and the autotuner measure the same way" a fact rather than a convention: they
    call one function that happens to have two import paths.
    """
    assert getattr(BU, name) is getattr(bench_timing, name), (
        f"bench_utils.{name} is not the packaged object -- a wrapper here means the timer the perf "
        f"gates call and the timer the autotuner calls can diverge"
    )


@pytest.mark.parametrize("name", ["BenchResult", "PairedResult", "ColdCompileComparison"])
def test_the_result_types_are_shared(name):
    """The result dataclasses must be shared too, or an ``isinstance`` check across the two paths fails."""
    assert getattr(BU, name) is getattr(bench_timing, name)


def test_the_sample_floor_is_the_same_number_on_both_paths():
    """``MIN_COLD_COMPILE_SAMPLES`` is re-exported by identity, not copied.

    It is a constant rather than a function, which is exactly why it is worth pinning: a copied
    ``5`` in the shim would keep working right up until the floor was raised in one place, and a
    caller that imported the stale one would then claim a delta from too few samples with nothing
    to say so.
    """
    assert BU.MIN_COLD_COMPILE_SAMPLES is bench_timing.MIN_COLD_COMPILE_SAMPLES


def test_the_shim_advertises_exactly_what_it_re_exports():
    """``__all__`` and the module namespace agree, so a dropped re-export cannot pass silently.

    The literal set is spelled out rather than derived from the source module ON PURPOSE. Deriving it
    would make this test agree with whatever the shim happens to re-export, which is the one thing it
    exists to check -- a name added to `bench_timing` and forgotten here must FAIL, so that adding a
    timer entry point is a deliberate act at both ends rather than a silent widening of the surface.
    """
    for name in BU.__all__:
        assert hasattr(BU, name), f"__all__ names {name!r}, which the shim does not export"
    assert set(BU.__all__) == {
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
    }


def test_the_implementation_is_no_longer_under_benchmark():
    """The timer lives inside the installed package, which is the whole point of the move.

    ``pyproject.toml`` packages ``fold_cp_ops*`` only, so anything under ``benchmark/`` is absent
    from a wheel. The autotuner REQUIRES this timer -- CLAUDE.md forbids hand-rolling one, because a
    ``perf_counter`` loop around an async launch measures host dispatch and reports bandwidths above
    the physical link. A required dependency that disappears on ``pip install`` is a runtime error
    waiting for the first installed user, so this asserts where the code actually is.
    """
    assert (
        bench_timing.__file__.replace("\\", "/")
        .split("/fold_cp_ops/")[-1]
        .startswith("_internal/bench_timing.py")
    ), f"bench_timing must live inside the package; found {bench_timing.__file__}"
    assert "benchmark_single" not in BU.__file__, "sanity: __file__ is a path, not a symbol"
