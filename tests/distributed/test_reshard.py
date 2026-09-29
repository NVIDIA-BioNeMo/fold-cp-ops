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
"""Tests for ``fold_cp_ops.distributed.reshard`` -- the host-side A2A pack/unpack views.

**The defect class this file exists for: a wrong permute produces the right shape, the right dtype,
and the right values in the wrong places.** Every method here is a `reshape` / `permute` /
`contiguous` chain, so nothing downstream can catch a transposition -- the buffer is the correct
size, every element is present exactly once, and the assertion a careless test would make (shape,
dtype, sum, mean) passes. Only a positional comparison sees it, which is why every check below is
`assert_bitwise` against a constructed ground truth.

**Pack and unpack model a DISTRIBUTED EXCHANGE, so a single-rank round trip would test nothing.**
`front_pack` on rank *r* produces a `send` whose slot *s* is destined for peer *s*; after the
all-to-all, rank *r*'s `recv` slot *s* holds what peer *s* sent it. A test that packed and unpacked
on one rank would exercise neither the slot layout nor the reassembly. So :func:`_simulate_a2a`
performs the exchange as a pure host permutation across all `cp` ranks in one process -- which is
also why this file needs **no GPU, no nvshmem and no process group**.

**Pool decision, and it is the third instance of the same lesson.** The matrix requires
``cp > 1``, ``B > 1`` and ``N_loc > 1``. Those are not conservatism: at ``cp == 1`` every permute is
the identity and every scramble passes; at ``B == 1`` a ``(B, cp, ...)`` transposition is invisible
because the transposed axis has extent 1; at ``N_loc == 1`` the token-block split degenerates the
same way. They are the convenient values a quick test reaches for first, and each is a point at
which the defect contributes exactly zero -- the same shape as ``torch.randn``'s zero mean hiding a
padded-tail variance bug through 598 tests, and as the identity ``pe_table`` hiding a sorting
builder in ``test_peer_tma_atoms.py``.

**Not covered here, and it is not a deferral:** `reshard.py` builds no TMA atom, translates no peer
pointer and touches no signal (measured: 0 hits for ``make_tiled_tma_atom``, ``get_peer_tensor``,
``signal_op``, ``assumed_align``, ``nvshmem``). Obligations A and B therefore do not belong to this
file at all and are pinned by FILENAME to ``gemm_sm90_a2a.py``, which builds the atoms.
"""

from typing import List

import pytest
import torch

from fold_cp_ops.testing.kernel_matrix import (
    Axis,
    KernelMatrix,
    computes_nothing_numeric,
    matrix_exempt,
    no_unsupported,
)
from fold_cp_ops.testing.numeric_guard import numeric_exempt
from fold_cp_ops.testing.numerics import assert_bitwise

RESHARD = KernelMatrix(
    kernel="reshard",
    axes=(
        Axis(
            name="geom",
            domain=(
                "any (cp, B, N, D, token_dim) with D and N divisible by the cp axes. The pool "
                "REQUIRES cp>1, B>1 and N_loc>1, and that is the discriminating decision rather "
                "than caution: at cp==1 every permute is the identity so any scramble passes; at "
                "B==1 a (B, cp, ...) transposition is invisible because the axis has extent 1; at "
                "N_loc==1 the token-block split degenerates the same way. Those are exactly the "
                "convenient values a quick test reaches for, and each is a point where this file's "
                "defect class contributes precisely nothing"
            ),
            values=(
                (2, 2, 4, 4, 1),
                (2, 2, 4, 4, 2),
                (4, 2, 8, 8, 1),
                (4, 2, 8, 8, 2),
                (4, 3, 8, 16, 1),
                (8, 2, 16, 8, 1),
            ),
            facets={
                "token_dim_1": lambda g: g[4] == 1,
                "token_dim_2": lambda g: g[4] == 2,
                "small_cp": lambda g: g[0] <= 2,
                "large_cp": lambda g: g[0] >= 4,
                # feat != D exercises the feat_width override path separately from D.
                "square_feat": lambda g: g[3] == g[2],
                "wide_feat": lambda g: g[3] != g[2],
            },
        ),
    ),
    computes=computes_nothing_numeric(
        because=(
            "every method here is a pure PERMUTATION -- reshape, permute, contiguous, with no "
            "arithmetic anywhere -- so no input distribution can hide a defect: the output is the "
            "input's bits in some order, and the only question is which order. That is also why "
            "the tests feed `arange` rather than `randn`: for a permutation the discriminating "
            "input property is DISTINCTNESS, not distribution. Random values risk collisions that "
            "would let a misplacement compare equal, which is the opposite of the usual concern"
        )
    ),
    unsupported=no_unsupported(
        because=(
            "every declared (cp, B, N, D, token_dim) satisfies the divisibility the constructor "
            "requires, so no combination of declared axis values is refused. The module's four "
            "raises -- D not divisible by cp, feat_width not divisible by cp, >2 cp axes, and a "
            "token_dim outside (1,2) -- all fire on MALFORMED constructor arguments rather than on "
            "any combination of pool values, and are covered directly by "
            "test_the_constructor_refuses_a_geometry_it_cannot_split"
        )
    ),
)


class _FakePeMap:
    """Minimal stand-in carrying only the five attributes `ReshardLayout` reads.

    Input requirements: `cp` > 0; `my_cp_rank` in ``[0, cp)``; `cp_axis_sizes` a 1- or 2-tuple whose
    product is `cp`; `cp_shard_tensor_dims` a tuple whose first entry is the split token dim (1 or
    2). A real `PeMap` is not used because building one needs a device mesh and a process group,
    and none of the properties under test depend on how the map was derived -- only on what
    `ReshardLayout` does with it.
    """

    def __init__(self, cp, my_cp_rank, token_dim):
        self.cp = cp
        self.my_cp_rank = my_cp_rank
        self.cp_axis_sizes = (cp,)
        self.cp_shard_tensor_dims = (token_dim,)


def _layouts(cp, B, N, D, token_dim, feat_width=None):
    """One `ReshardLayout` per rank, which is what a distributed exchange needs to be simulated."""
    from fold_cp_ops.distributed.reshard import ReshardLayout

    return [ReshardLayout(_FakePeMap(cp, r, token_dim), B, N, D, feat_width) for r in range(cp)]


def _simulate_a2a(sends: List[torch.Tensor], cp: int, rows: int) -> List[torch.Tensor]:
    """Perform the all-to-all as a pure host permutation: ``recv[r][s] = send[s][r]``.

    Purpose: the pack/unpack pair only means anything across ranks -- `send` slot ``s`` is destined
    for peer ``s``, and `recv` slot ``s`` holds what peer ``s`` sent. Simulating that here is what
    makes a round-trip test exercise the slot layout and the reassembly rather than a self-inverse
    pair of reshapes.

    Input requirements: `sends` has `cp` entries, each ``(cp*rows, cols)``; `rows` is
    ``rows_per_peer``. A mismatch raises from the slicing rather than silently truncating.

    Returns: `cp` receive buffers, each ``(cp*rows, cols)``.
    """
    return [
        torch.cat([sends[s][r * rows : (r + 1) * rows] for s in range(cp)], dim=0)
        for r in range(cp)
    ]


def _split_tokens(x: torch.Tensor, cp: int, token_dim: int) -> List[torch.Tensor]:
    """Split a full ``(B, N, N, feat)`` tensor into the per-rank token blocks the front consumes."""
    return list(torch.chunk(x, cp, dim=token_dim))


@RESHARD.parametrize("geom")
def test_the_front_exchange_reassembles_the_token_grid_for_my_feature_slice(geom):
    """front_pack -> A2A -> front_unpack gives each rank the FULL token grid for its D-slice.

    The ground truth is constructed independently: rank ``r``'s result must be exactly
    ``x[..., r*feat_loc:(r+1)*feat_loc]`` of the global tensor. Comparing against a slice of the
    original -- rather than against the pipeline's own intermediate -- is what stops the test from
    confirming that two wrong permutes cancel.
    """
    cp, B, N, D, token_dim = geom
    lays = _layouts(cp, B, N, D, token_dim)
    L0 = lays[0]
    x = torch.arange(B * N * N * L0.feat, dtype=torch.float32).reshape(B, N, N, L0.feat)
    locals_ = _split_tokens(x, cp, token_dim)

    sends = []
    for r in range(cp):
        s = torch.zeros(*lays[r].send_recv_shape, dtype=torch.float32)
        lays[r].front_pack(locals_[r].contiguous(), s)
        sends.append(s)
    recvs = _simulate_a2a(sends, cp, L0.rows_per_peer)

    for r in range(cp):
        got = lays[r].front_unpack(recvs[r])
        want = x[..., r * L0.feat_loc : (r + 1) * L0.feat_loc].contiguous()
        assert_bitwise(got, want, what=f"front_unpack on rank {r}")


@RESHARD.parametrize("geom")
def test_the_dmajor_unpack_agrees_with_the_plain_one(geom):
    """``front_unpack_dmajor`` equals ``front_unpack`` followed by the D-major transpose.

    This is the sharpest check in the file. ``front_unpack_dmajor`` FUSES two passes into one
    permute for speed, and a fused permute that silently reorders is exactly the class that survives
    every other test here -- the plain path stays correct, the fused path is self-consistent, and
    only a comparison BETWEEN them can tell that they disagree.
    """
    cp, B, N, D, token_dim = geom
    lays = _layouts(cp, B, N, D, token_dim)
    L0 = lays[0]
    x = torch.arange(B * N * N * L0.feat, dtype=torch.float32).reshape(B, N, N, L0.feat)
    locals_ = _split_tokens(x, cp, token_dim)

    sends = []
    for r in range(cp):
        s = torch.zeros(*lays[r].send_recv_shape, dtype=torch.float32)
        lays[r].front_pack(locals_[r].contiguous(), s)
        sends.append(s)
    recvs = _simulate_a2a(sends, cp, L0.rows_per_peer)

    for r in range(cp):
        plain = lays[r].front_unpack(recvs[r])
        # (B, N, N, feat_loc) -> (feat_loc, B*N*N), the layout GEMM1 contracts against.
        want = plain.permute(3, 0, 1, 2).reshape(L0.feat_loc, B * N * N).contiguous()
        got = lays[r].front_unpack_dmajor(recvs[r])
        assert_bitwise(got, want, what=f"front_unpack_dmajor vs plain+transpose on rank {r}")


@RESHARD.parametrize("geom")
def test_the_back_exchange_reassembles_the_full_feature_for_my_token_block(geom):
    """back_pack -> A2A -> back_unpack gives each rank the full feature D for its token block.

    The back half has its own frame and its own ``token_dim`` branch, so it is not covered by the
    front test even though the two are inverses in principle -- an error in either branch would
    otherwise be visible only in production.
    """
    cp, B, N, D, token_dim = geom
    lays = _layouts(cp, B, N, D, token_dim)
    L0 = lays[0]
    x = torch.arange(B * N * N * L0.feat, dtype=torch.float32).reshape(B, N, N, L0.feat)

    sends = []
    for r in range(cp):
        feat_shard = x[..., r * L0.feat_loc : (r + 1) * L0.feat_loc].contiguous()
        s = torch.zeros(*lays[r].send_recv_shape, dtype=torch.float32)
        lays[r].back_pack(feat_shard, s)
        sends.append(s)
    recvs = _simulate_a2a(sends, cp, L0.rows_per_peer)

    token_blocks = _split_tokens(x, cp, token_dim)
    for r in range(cp):
        got = lays[r].back_unpack(recvs[r])
        assert_bitwise(got, token_blocks[r].contiguous(), what=f"back_unpack on rank {r}")


@RESHARD.parametrize("geom")
@numeric_exempt(
    "asserts a declared SHAPE against the packer's actual geometry -- a contract between two "
    "host-side properties. The packed values are compared element-wise by the three round-trip "
    "tests above; repeating that here would test the permutation a fourth time and the shape "
    "contract not at all"
)
def test_the_declared_buffer_shape_is_the_one_pack_actually_writes(geom):
    """``send_recv_shape`` matches what ``front_pack`` fills -- other modules size buffers from it.

    `CpAllToAll` allocates symmetric buffers from this property, and a symmetric allocation is
    COLLECTIVE, so a shape that disagreed with the packer would not be a local error: every rank
    would allocate the wrong size together and the mismatch would surface as corruption or a hang
    rather than as a shape check.
    """
    cp, B, N, D, token_dim = geom
    lays = _layouts(cp, B, N, D, token_dim)
    L0 = lays[0]
    x = torch.arange(B * N * N * L0.feat, dtype=torch.float32).reshape(B, N, N, L0.feat)
    local0 = _split_tokens(x, cp, token_dim)[0].contiguous()

    s = torch.zeros(*L0.send_recv_shape, dtype=torch.float32)
    L0.front_pack(local0, s)  # raises if the declared shape cannot hold the packed data
    assert L0.send_recv_shape == (cp * L0.rows_per_peer, L0.feat_loc), (
        f"send_recv_shape {L0.send_recv_shape} disagrees with (cp*rows_per_peer, feat_loc) = "
        f"{(cp * L0.rows_per_peer, L0.feat_loc)}; callers size COLLECTIVE symmetric allocations "
        "from this property, so a disagreement is a whole-job failure rather than a local one"
    )


@matrix_exempt(
    "asserts constructor REFUSALS on malformed geometry -- inputs deliberately outside every "
    "declared axis value, so drawing them from the matrix is impossible by construction"
)
def test_the_constructor_refuses_a_geometry_it_cannot_split():
    """Each of the divisibility contracts raises with a message naming the offending numbers.

    These are the module's only refusals. They matter because the alternative to raising is a
    silent truncation: an indivisible D would leave a ragged final feature slice that every reshape
    downstream would happily accept at the wrong size.
    """
    from fold_cp_ops.distributed.reshard import ReshardLayout

    with pytest.raises(ValueError, match="must be divisible by cp"):
        ReshardLayout(_FakePeMap(4, 0, 1), B=2, N=8, D=6)
    with pytest.raises(ValueError, match="feat_width=.* must be divisible by cp"):
        ReshardLayout(_FakePeMap(4, 0, 1), B=2, N=8, D=8, feat_width=6)
    with pytest.raises(ValueError, match="must be divisible by both cp axes"):
        pm = _FakePeMap(4, 0, 1)
        pm.cp_axis_sizes = (4,)
        ReshardLayout(pm, B=2, N=6, D=8)
