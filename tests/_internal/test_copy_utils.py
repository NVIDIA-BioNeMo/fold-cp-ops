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

"""Unit tests for ``_internal/copy_utils.py`` -- the RUNTIME copy helpers.

``copy`` issues real data movement, ``predicate_k`` materializes a register-resident boolean mask
and ``fill_oob`` writes through it, so all three are tested through the row-sum harness against an
independent torch oracle rather than on the host. The descriptors they consume are compile-time
algebra and are tested in ``tests/_internal/compile_time/test_copy_descriptors.py``.
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
    reason="the copy / predicate kernel path needs SM90",
)


@requires_sm90
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("M,N", [(64, 1024), (37, 760), (199, 1128), (8, 2048), (128, 384)])
def test_copy_roundtrip_via_row_sum(dtype, M, N):
    """``copy`` moves every element: the row sum over the copied tile matches the torch oracle.

    N=760, 1128 and 384 are not tile multiples, so ``predicate_k`` is engaged and the trailing
    partial tile is masked. A predicate wrong in either direction moves the sum -- too permissive
    reads past the row, too strict drops real elements.
    """
    torch.manual_seed(0)
    x = torch.randn(M, N, device="cuda", dtype=dtype)
    # PER ROW against an fp64 oracle, not a pooled L2 over all M rows: a predicate wrong
    # on ONE row is the defect this test exists for, and an L2 over 64 rows divides it
    # away. The bound is derived from the row's own |terms| (see reduction_error_bound),
    # so a row that cancels to near zero is judged on its own scale.
    assert_elementwise(
        row_sum(x),
        reduction_reference(x),
        reduction_error_bound(x, torch.float32),
        what=f"row sum M={M} N={N} {dtype}",
    )


@requires_sm90
@pytest.mark.parametrize("N", [760, 1128, 384, 1000, 2048])
def test_predicate_k_masks_exactly_the_tail(N):
    """The masked tail contributes EXACTLY zero, checked by exact equality rather than tolerance.

    A row of ones sums to exactly N. An over-read past the row would add whatever is in the
    neighbouring allocation; an under-read lands below N. Either shows up as an exact integer
    mismatch, so this pins the predicate boundary far more sharply than a relative-error bound.
    N=2048 is tile-aligned, so it is the control: there the predicate is not built at all.
    """
    x = torch.ones(64, N, device="cuda", dtype=torch.float32)
    got = row_sum(x)
    assert torch.equal(got, torch.full_like(got, float(N))), (
        f"N={N}: every row should sum to exactly {N}; got {got.unique().tolist()}"
    )


# ── fill_oob ───────────────────────────────────────────────────────────────────────────────────
def _tile_cols(N: int, cluster_n: int = 1) -> int:
    """Columns in one CTA tile times the cluster width, read back OUT of the kernel.

    Filling the whole tile with 1.0 discards every real element, so the row sum degenerates to a
    count of the tile's columns. Recovering the width this way rather than recomputing it from the
    ``_threads_per_row`` / ``vecsize`` ladders is the point: a hand-recomputed width would be a
    second source of truth, and a test that agrees with its own copy of the arithmetic proves
    nothing about the kernel's.

    Args:
        N: Feature extent to build the tile for. Positive; may be off-grid.
        cluster_n: Cluster width to force. The reported total spans the whole cluster.

    Returns:
        The tile's column count, always ``>= N``.
    """
    x = torch.zeros(8, N, device="cuda", dtype=torch.float32)
    return int(row_sum(x, cluster_n=cluster_n, oob_fill=1.0, fill_all=True)[0].item())


@requires_sm90
@pytest.mark.parametrize("N", [760, 1000, 1128, 384, 2048])
def test_fill_oob_touches_exactly_the_masked_tail(N):
    """A tail fill of ``v`` moves the row sum by exactly ``n_pad * v`` -- and by nothing else.

    Three exact (integer-valued) facts are pinned together, which is what makes this tight:

    1. ``oob_fill=0.0`` reproduces the unfilled sum. The tail is already zero-filled by the
       predicated ``cp.async``, so a fill that reaches an in-bounds element would show up here as
       a wrong total -- this is the "does not clobber real data" check.
    2. The offset is **linear** in the fill value: ``sum(v) - sum(0) == v * (sum(1) - sum(0))``.
       A predicate off by one vector would still be linear, so this alone is not enough, hence:
    3. The implied ``n_pad`` equals ``tile_cols - N``, where ``tile_cols`` is read back out of the
       kernel by :func:`_tile_cols`. That is what pins the boundary at the exact column.

    N=2048 and 384 are the controls: both tile evenly here, so ``n_pad`` must be 0 and the fill
    must be a no-op.
    """
    x = torch.ones(64, N, device="cuda", dtype=torch.float32)
    base = row_sum(x)
    assert torch.equal(row_sum(x, oob_fill=0.0), base), (
        f"N={N}: filling the tail with 0.0 changed the sum -- fill_oob wrote in-bounds elements"
    )
    off1 = (row_sum(x, oob_fill=1.0) - base)[0].item()
    off7 = (row_sum(x, oob_fill=7.0) - base)[0].item()
    assert off7 == 7 * off1, (
        f"N={N}: fill offset is not linear in the fill value ({off7} vs 7x{off1})"
    )
    assert off1 == _tile_cols(N) - N, (
        f"N={N}: fill reached {off1:.0f} elements, but the tile has {_tile_cols(N) - N} padding "
        f"columns -- the predicate boundary is off"
    )


@requires_sm90
@pytest.mark.parametrize("N", [1000, 2048])
def test_fill_oob_with_no_predicate_fills_everything(N):
    """``tXpX=None`` means "fill the whole tile", not "fill nothing".

    The distinction matters because the two calls are one argument apart and the wrong one is
    silent: passing None where a predicate was meant destroys the staged data instead of patching
    its tail. Here every real element is a 1 and the fill is a 3, so an unconditional fill must
    leave no 1s behind at all -- the sum is a clean multiple of 3.
    """
    x = torch.ones(64, N, device="cuda", dtype=torch.float32)
    got = row_sum(x, oob_fill=3.0, fill_all=True)
    expected = 3.0 * _tile_cols(N)
    assert torch.equal(got, torch.full_like(got, expected)), (
        f"N={N}: expected every row to be {expected} (3 x every tile column); "
        f"got {got.unique().tolist()}"
    )
