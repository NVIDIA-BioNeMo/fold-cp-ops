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

"""Unit tests for ``_internal/reduction_base.py``.

``_num_threads`` and the ``_threads_per_row`` contract are pure Python and tested directly.
``_get_tiled_copy`` / ``_allocate_reduction_buffer_and_mbar`` / ``_initialize_cluster`` only have
meaning inside ``@cute.jit``, so they are exercised through the row-sum harness.
"""

import pytest
import torch

import cutlass

from fold_cp_ops._internal.reduction_base import ReductionBase

from fold_cp_ops.testing.numerics import (
    assert_elementwise,
    reduction_error_bound,
    reduction_reference,
)
from tests._internal._rowsum_kernel import row_sum


@pytest.mark.parametrize(
    "N,expected",
    [(1, 128), (1024, 128), (16384, 128), (16385, 256), (32768, 256), (262144, 256)],
)
def test_num_threads_switches_at_16384(N, expected):
    """Block size is 128 threads up to N=16384 inclusive and 256 above -- boundary pinned.

    Both sides of the boundary are parametrized because an off-by-one (``<`` vs ``<=``) changes
    the block size for exactly one shape, which a sampled test would miss. The block size feeds
    the tile geometry, so a shift here silently changes the compiled tile for that shape.
    """
    assert ReductionBase(cutlass.BFloat16, N, stage=1)._num_threads() == expected


def test_threads_per_row_is_abstract():
    """The base class refuses to guess a lane count: subclasses must supply the ladder.

    ``_threads_per_row`` is the one knob that has to be chosen per kernel; a silent default would
    give a subclass a plausible-but-untuned geometry with no signal that it was never set.
    """
    with pytest.raises(NotImplementedError):
        ReductionBase(cutlass.BFloat16, 1024, stage=1)._threads_per_row()


def test_default_cluster_n_is_one():
    """``cluster_n`` defaults to 1 and — crucially — EXISTS immediately after construction.

    It used to be established only by a ``_set_cluster_n()`` call made from the subclass's
    ``__call__``, so a freshly built functor did not have the attribute at all and
    ``_get_tiled_copy()`` raised ``AttributeError`` on it. Asserting presence-at-construction is
    the regression test for that; the default of 1 also means ``const_expr(cluster_n > 1)`` prunes
    every distributed-shared-memory path out of a plain build.
    """
    base = ReductionBase(cutlass.BFloat16, 1024, stage=1)
    assert base.cluster_n == 1
    assert "cluster_n" in base.param_dict()


def test_compile_time_params_are_immutable():
    """Rebinding a declared parameter raises — it is folded into the kernel as a constant.

    This is the guard that makes the old failure mode unrepresentable: configuration assigned by
    whichever method happened to run first. The message must point at the fix, so it is asserted.
    """
    base = ReductionBase(cutlass.BFloat16, 1024, stage=1)
    for name, value in (("cluster_n", 4), ("N", 2048), ("stage", 3), ("dtype", cutlass.Float32)):
        with pytest.raises(AttributeError, match="immutable after construction"):
            setattr(base, name, value)


def test_binding_params_twice_is_refused():
    """A second ``_bind_params`` would reintroduce two-phase configuration, so it raises."""
    base = ReductionBase(cutlass.BFloat16, 1024, stage=1)
    with pytest.raises(RuntimeError, match="called twice"):
        base._bind_params(dtype=cutlass.BFloat16, N=1024, stage=1)


def test_runtime_value_as_a_parameter_is_rejected():
    """A tensor passed where a compile-time constant belongs raises AT CONSTRUCTION.

    This is the whole point of the validation. A runtime value stashed on ``self`` does not raise
    at the assignment; it produces a kernel reading a stale MLIR value, and the symptom appears
    much later as a wrong result. Failing here, naming the field, is the difference.
    """
    x = torch.zeros(4, device="cuda") if torch.cuda.is_available() else torch.zeros(4)
    with pytest.raises(TypeError, match="RUNTIME values"):
        ReductionBase(dtype=x, N=1024, stage=1)


def test_constructor_records_its_arguments():
    """dtype / N / stage / reduction_dtype are stored verbatim and drive every derived quantity.

    ``stage`` in particular sizes the reduction buffer's third mode and the mbarrier array, so a
    dropped argument would under-allocate shared memory rather than raise.
    """
    base = ReductionBase(cutlass.BFloat16, 4096, stage=2)
    assert base.dtype is cutlass.BFloat16
    assert base.N == 4096
    assert base.stage == 2
    assert base.reduction_dtype is cutlass.Float32


requires_sm90 = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 9,
    reason="the tiled-copy / reduction-buffer path runs in an SM90 kernel",
)


@requires_sm90
@pytest.mark.parametrize("N", [1024, 16384, 16640, 32768])
def test_tile_geometry_is_correct_on_both_sides_of_the_block_size_switch(N):
    """A kernel built at each block size still reduces correctly.

    N=16384 and N=16640 straddle the ``_num_threads`` switch, so this compiles one kernel at 128
    threads and one at 256 and checks both. The tile extent is derived from the block size, so a
    geometry bug at the switch would corrupt exactly one of these two.
    """
    torch.manual_seed(0)
    x = torch.randn(16, N, device="cuda", dtype=torch.float32)
    # PER ROW against an fp64 oracle. A geometry bug at the block-size switch corrupts the
    # rows one tile covers, not all 16 uniformly -- so the pooled L2 this replaces was the
    # wrong statistic for the only defect the test names.
    assert_elementwise(
        row_sum(x),
        reduction_reference(x),
        reduction_error_bound(x, torch.float32),
        what=f"row sum N={N}",
    )
