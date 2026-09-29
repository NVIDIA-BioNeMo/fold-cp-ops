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


"""Tests for `benchmark.distributed.harness.shape_gate`."""

import pytest

from benchmark.distributed.harness.shape_gate import shape_check
from fold_cp_ops.testing.kernel_matrix import matrix_exempt

pytestmark = matrix_exempt(
    "the subject is a pure-integer HOST-side shape predicate, not a kernel: there is no kernel "
    "configuration to sweep, and the values parametrized here ARE the mandated sweep extents the "
    "gate must accept, so drawing them from a KernelMatrix would make the test assert against the "
    "same source it is meant to check"
)

# The mandated perf matrix's token extents, on the meshes a 2..16 cp ladder reaches. These are the
# shapes the front sweep will actually request, so pinning the gate ON THEM is what makes a future
# edit that narrows it fail here rather than silently emptying a sweep.
LEGAL = [
    (2048, 2, 1),
    (4096, 2, 1),
    (8192, 2, 1),
    (12288, 2, 1),  # on-grid ladder, 1-D mesh
    (2048, 4, 4),
    (4096, 4, 4),
    (8192, 16, 1),
    (12288, 16, 1),  # 2-D mesh and the widest cp
    (1088, 2, 1),
    (4032, 2, 1),
    (10048, 2, 1),
    (4128, 2, 1),  # the OFF-GRID pool (_OFFGRID_NTOK)
    (2048, 1, 1),  # degenerate cp=1
]

ILLEGAL = [
    ((2048, 0, 1), "invalid cp mesh"),  # a zero extent is a bad mesh, not a bad shape
    ((2048, 2, -1), "invalid cp mesh"),
    ((137, 2, 1), "not divisible"),  # odd N cannot shard
    ((2048, 2, 3), "not divisible"),  # divides one axis, not the other
    ((16, 4, 1), "16-B-aligned"),  # shards fine, but N/cp0 = 4 is under the 8-element floor
    ((64, 16, 1), "16-B-aligned"),  # N/cp0 = 4 again, at a larger N
]


@pytest.mark.parametrize("N,cp0,cp1", LEGAL, ids=lambda v: str(v))
def test_the_mandated_sweep_shapes_are_legal(N, cp0, cp1):
    """Every extent the front sweep requests passes the gate, on-grid and off-grid alike.

    The off-grid pool is the load-bearing half: those extents are deliberately NOT tile multiples
    (`%128` in {32, 64}) and exist to exercise the partial-last-tile path. A gate that rejected them
    would not fail loudly -- it would report them as `skip_shape` and quietly shrink the
    sweep to the aligned cells, which is the coverage loss this pool was added to prevent.
    """
    ok, reason = shape_check(N, cp0, cp1)
    assert ok, f"N={N} cp=({cp0},{cp1}) must be legal, got reason: {reason}"
    assert reason == "", f"a legal shape must carry no reason; got {reason!r}"


@pytest.mark.parametrize("args,expect", ILLEGAL, ids=lambda v: str(v))
def test_an_illegal_shape_is_refused_with_an_actionable_reason(args, expect):
    """Each refusal names WHICH extent failed, not merely that one did.

    The reason string is the only thing a user sees when a cell reports `skip_shape`, so a gate that
    returned a bare False would turn every skipped cell into an investigation. The three failure
    classes are distinguished on purpose: a bad mesh, an unshardable N, and a shardable N
    whose LOCAL extent breaks the 16-byte TMA floor are three different mistakes with three
    different fixes.
    """
    ok, reason = shape_check(*args)
    assert not ok, f"{args} must be refused"
    assert expect in reason, f"reason for {args} should name {expect!r}; got {reason!r}"


def test_the_gate_is_pure_so_every_rank_computes_the_same_verdict():
    """Same inputs, same verdict -- the property the driver relies on for collective symmetry.

    `supports()` runs independently on every rank and the driver does NOT reconcile the
    answers, so a gate that consulted the environment, a device, or any global would let one
    rank skip a cell while its peers built it -- and the peers would then block in the next
    rank skip a cell while its peers built it -- and the peers would then block in the next
    collective until a watchdog killed the job.
    """
    for args in [(2048, 2, 1), (137, 2, 1), (16, 4, 1)]:
        first = shape_check(*args)
        assert all(shape_check(*args) == first for _ in range(4)), f"{args} not deterministic"
