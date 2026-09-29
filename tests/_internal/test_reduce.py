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

"""Unit tests for ``_internal/reduce.py``.

Driven through the row-sum harness, whose oracle (``x.sum(-1)``) is independent of any kernel
under test. ``cluster_n`` is forced rather than derived from N, so a test can select the
``block_reduce`` and ``cluster_reduce`` branches independently of the shape.
"""

import pytest
import torch

from fold_cp_ops.testing.numerics import (
    assert_elementwise,
    reduction_error_bound,
    reduction_reference,
)
from tests._internal._rowsum_kernel import row_sum

requires_sm90 = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 9,
    reason="reductions run in an SM90 kernel",
)


@requires_sm90
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize(
    "N",
    [
        64,  # threads_per_row ladder: <=64   -> 8 lanes/row (sub-warp: warp shuffle only)
        128,  #                         <=128  -> 16
        1024,  #                         <=3072 -> 32 (exactly one warp per row)
        4096,  #                         <=6144 -> 64 (crosses a warp -> block_reduce engages)
        8192,  #                         <=16384-> 128
        24576,  #                         else   -> 256
    ],
)
def test_row_reduce_across_the_threads_per_row_ladder(N, dtype):
    """Every rung of the ``threads_per_row`` ladder reduces correctly.

    The rungs are not cosmetic: at <=3072 a row fits in one warp and ``row_reduce`` stops after
    the butterfly shuffle, while at 4096+ the row spans several warps and it must escalate to
    ``block_reduce`` through the shared-memory staging buffer. A bug in the escalation boundary
    shows up as a sum that is right for small N and wrong for large N, which is exactly why the
    ladder is parametrized rather than sampled.
    """
    torch.manual_seed(0)
    x = torch.randn(64, N, device="cuda", dtype=dtype)
    # PER ROW against an fp64 oracle. The escalation boundary this test guards fails on a
    # SUBSET of rows -- a warp that fails to contribute moves those rows and no others --
    # which is exactly the shape a pooled L2 over 64 rows averages away.
    assert_elementwise(
        row_sum(x),
        reduction_reference(x),
        reduction_error_bound(x, torch.float32),
        what=f"row sum N={N} {dtype}",
    )


@requires_sm90
@pytest.mark.parametrize("N", [128, 1024, 4096])
def test_row_reduce_is_exact_on_representable_input(N):
    """On exactly-representable input the reduction is exact regardless of ordering.

    A row of ones sums to exactly N in float32 for these N, so any lost or double-counted partial
    (a mis-sized staging buffer, a warp that fails to contribute) is an exact integer mismatch
    rather than a tolerance judgement call.
    """
    got = row_sum(torch.ones(64, N, device="cuda", dtype=torch.float32))
    assert torch.equal(got, torch.full_like(got, float(N)))


@requires_sm90
@pytest.mark.parametrize("cluster_n", [1, 2, 4])
@pytest.mark.parametrize("N", [16384, 32768])
def test_cluster_reduce_matches_block_reduce(N, cluster_n):
    """The cluster path agrees with the single-CTA path on the same input.

    ``cluster_n == 1`` takes ``block_reduce``; ``> 1`` takes ``cluster_reduce``, which pushes
    partials into peer CTAs' shared memory and synchronises on an mbarrier. Comparing the two
    against ONE oracle is what makes this a test of the cluster handshake rather than of the
    arithmetic: if the mbarrier arrive/wait were mis-ordered, some peers' partials would be read
    before they land and only the cluster_n > 1 cells would drift.
    """
    torch.manual_seed(0)
    x = torch.randn(32, N, device="cuda", dtype=torch.float32)
    # PER ROW against an fp64 oracle. "Some peers' partials read before they land" is a
    # per-row corruption on whichever rows that CTA owned, so the pooled L2 this replaces
    # was averaging the mbarrier defect across the 32 rows that did land correctly.
    assert_elementwise(
        row_sum(x, cluster_n=cluster_n),
        reduction_reference(x),
        reduction_error_bound(x, torch.float32),
        what=f"cluster row sum N={N} cluster_n={cluster_n}",
    )


@requires_sm90
@pytest.mark.parametrize("cluster_n", [2, 4])
def test_cluster_reduce_is_exact_on_representable_input(cluster_n):
    """A dropped or double-counted peer partial is an exact mismatch, not a rounding difference.

    With every element 1.0 the true sum is exactly N. If one CTA of the cluster failed to
    contribute, the result would be off by a clean fraction of N -- visible here, invisible under
    a relative-error bound on random input.
    """
    N = 32768
    got = row_sum(torch.ones(32, N, device="cuda", dtype=torch.float32), cluster_n=cluster_n)
    assert torch.equal(got, torch.full_like(got, float(N))), (
        f"cluster_n={cluster_n}: expected exactly {N}, got {got.unique().tolist()}"
    )
