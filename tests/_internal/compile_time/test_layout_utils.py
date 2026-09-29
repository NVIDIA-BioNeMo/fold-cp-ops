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

"""Unit tests for ``_internal/layout_utils.py`` (currently just ``expand``).

Two layers. The host-side tests assert the returned LAYOUT directly (shape, stride, rank) using the
``mlir_ctx`` fixture; the kernel tests assert the observable BEHAVIOUR through the row-sum harness.
Both are worth having: the layout tests pin the contract precisely and run without a GPU, while the
harness tests prove the contract is the one the kernel actually relies on.

The host tests need a real ``cute.Tensor``. A ``make_fake_tensor`` will not do -- its ``.layout`` is
``None`` -- and a runtime tensor from ``from_dlpack`` raises ``NotImplementedError`` on ``.layout``.
``cute.make_identity_tensor`` gives a coordinate tensor with a genuine inspectable layout and no
backing memory, which is exactly what is needed to test layout algebra.
"""

import pytest
import torch

import cutlass.cute as cute

from fold_cp_ops._internal.compile_time.layout_utils import expand

from tests._internal._rowsum_kernel import row_sum


# ── host-side: assert the layout itself (no GPU) ───────────────────────────────────────────────
@pytest.mark.parametrize("dim,size", [(0, 4), (1, 7), (0, 1), (1, 128)])
def test_expand_inserts_a_zero_stride_mode(mlir_ctx, dim, size):
    """The new mode has the requested extent and stride EXACTLY 0 -- it broadcasts, not allocates.

    Stride 0 is the whole contract: every index along the new mode aliases the same element, so the
    result is a view over the source's memory. Any non-zero stride would walk off the end of a
    ``(N,)`` weight vector the moment it was indexed by row.
    """
    base = cute.make_identity_tensor((128,))
    out = expand(base, dim=dim, size=size)
    assert cute.rank(out.layout) == cute.rank(base.layout) + 1
    assert out.layout.shape[dim] == size
    assert out.layout.stride[dim] == 0


def test_expand_preserves_the_source_mode(mlir_ctx):
    """Expanding splices a mode in; it must not perturb the source's own extent or stride."""
    base = cute.make_identity_tensor((128,))
    out = expand(base, dim=0, size=8)
    assert out.layout.shape[1] == base.layout.shape[0]
    assert out.layout.stride[1] == base.layout.stride[0]


def test_expand_dim_selects_which_axis_broadcasts(mlir_ctx):
    """``dim`` places the broadcast mode; front and back placements are NOT interchangeable.

    LayerNorm depends on both: the weight is expanded at dim=0 (broadcast down rows) while
    rstd/mean are expanded at dim=1 (broadcast across the feature axis). Swapping them transposes
    the broadcast and corrupts every output, so the asymmetry is pinned here rather than assumed.
    """
    base = cute.make_identity_tensor((128,))
    front, back = expand(base, dim=0, size=4), expand(base, dim=1, size=4)
    assert front.layout.shape == (4, 128) and front.layout.stride[0] == 0
    assert back.layout.shape == (128, 4) and back.layout.stride[1] == 0
    assert front.layout.stride[1] != 0 and back.layout.stride[0] != 0


# ── kernel-side: assert the behaviour the kernel relies on ─────────────────────────────────────

requires_sm90 = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 9,
    reason="expand is exercised through an SM90 kernel",
)


@requires_sm90
@pytest.mark.parametrize("M,N", [(64, 1024), (199, 1128), (37, 760), (256, 512)])
def test_expand_places_each_row_result_at_its_own_index(M, N):
    """Row ``m`` is built to sum to exactly ``m``; assert ``out[m] == m`` for every row.

    This is the sharp test of ``expand``'s ``dim`` placement. The harness expands the ``(M,)``
    output to ``(M, N)`` with stride 0 on the new mode so it partitions like the data tensor. If
    the broadcast mode were inserted at the wrong ``dim``, or carried a non-zero stride, results
    would land in the wrong slots -- and because every row here has a DISTINCT expected value,
    that permutation is visible. A uniform-value test would pass under exactly that bug.
    """
    x = torch.zeros(M, N, device="cuda", dtype=torch.float32)
    x[:, 0] = torch.arange(M, device="cuda", dtype=torch.float32)  # row m sums to m
    got = row_sum(x)
    expected = torch.arange(M, device="cuda", dtype=torch.float32)
    assert torch.equal(got, expected), f"M={M} N={N}: row results misplaced; got {got[:8].tolist()}"


@requires_sm90
def test_expand_broadcast_is_a_view_not_a_copy():
    """A stride-0 mode must alias, so the harness needs no ``(M, N)`` output allocation.

    Checked by shape rather than by memory: the harness returns a ``(M,)`` tensor even though the
    kernel wrote through an ``(M, N)`` view. A non-zero stride on the broadcast mode would have
    required real ``(M, N)`` storage and the write would have run off the end of the allocation.
    """
    M, N = 128, 1024
    out = row_sum(torch.ones(M, N, device="cuda", dtype=torch.float32))
    assert out.shape == (M,)
    assert torch.equal(out, torch.full((M,), float(N), device="cuda"))
