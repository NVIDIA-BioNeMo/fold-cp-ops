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

"""Unit tests for ``_internal/utils.py``.

The three retained primitives -- ``elem_pointer``, ``set_block_rank``, ``store_shared_remote`` --
are distributed-shared-memory operations with no host-callable form and no return value a host
can read. They are reachable ONLY through ``reduce.cluster_reduce``, i.e. only when
``cluster_n > 1``, so they are tested there: a ``cluster_n == 1`` run does not execute a single
line of this module.

That makes the coverage below indirect but not weak. ``store_shared_remote`` writes into a PEER
CTA's shared memory at an address computed by ``elem_pointer`` and remapped by ``set_block_rank``;
if any of the three were wrong the partial would land in the wrong CTA, at the wrong offset, or
in unmapped memory -- producing a wrong sum or a fault, both caught here.
"""

import pytest
import torch

from tests._internal._rowsum_kernel import row_sum

requires_sm90 = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 9,
    reason="distributed shared memory needs SM90 thread-block clusters",
)


@requires_sm90
@pytest.mark.parametrize("cluster_n", [2, 4])
@pytest.mark.parametrize("M", [1, 32, 199])
def test_store_shared_remote_delivers_every_peer_partial(M, cluster_n):
    """Every CTA's partial reaches every peer: the summed result is exact for all M.

    Each of the ``cluster_n`` CTAs owns a slice of the row and pushes its partial to all peers via
    ``store_shared_remote``. With all-ones input the exact answer is N, so a partial that landed
    at a wrong offset (bad ``elem_pointer``) or in the wrong CTA (bad ``set_block_rank``) shows up
    as an exact shortfall or excess. M is varied because the row count sets the grid, and a
    single-row grid is the degenerate case most likely to expose an off-by-one in the mapping.
    """
    N = 32768
    got = row_sum(torch.ones(M, N, device="cuda", dtype=torch.float32), cluster_n=cluster_n)
    assert got.shape == (M,)
    assert torch.equal(got, torch.full_like(got, float(N))), (
        f"M={M} cluster_n={cluster_n}: expected exactly {N}, got {got.unique().tolist()}"
    )


@requires_sm90
@pytest.mark.parametrize("cluster_n", [2, 4])
def test_cluster_partials_are_position_dependent(cluster_n):
    """Distinct per-row values survive the cluster round-trip, ruling out a uniform-value pass.

    An all-ones test can pass even if partials were broadcast to the wrong ROW, because every row
    expects the same answer. Here row ``m`` sums to exactly ``m``, so any cross-row contamination
    in the remote store is visible as a permuted or smeared result.
    """
    M, N = 64, 32768
    x = torch.zeros(M, N, device="cuda", dtype=torch.float32)
    x[:, 0] = torch.arange(M, device="cuda", dtype=torch.float32)
    got = row_sum(x, cluster_n=cluster_n)
    assert torch.equal(got, torch.arange(M, device="cuda", dtype=torch.float32)), (
        f"cluster_n={cluster_n}: rows contaminated; got {got[:8].tolist()}"
    )
