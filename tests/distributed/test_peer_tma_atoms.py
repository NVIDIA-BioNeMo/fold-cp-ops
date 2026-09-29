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
"""Tests for ``fold_cp_ops.distributed.peer_tma_atoms`` -- the per-peer S2G atom builder.

**The property worth testing is WHICH peer each atom was baked against.** An S2G TMA descriptor
freezes its destination base pointer at atom-build time and cannot be retargeted at runtime, so
``atoms[r]`` is permanently bound to ``pe_table[r]``. If that mapping is permuted, the kernel stores
a tile to one PE and signals another -- a recorded anti-pattern, not a hypothesis, and one that
produces no error anywhere: the data lands somewhere valid, just not where the receiver waits.

**Two pool decisions carry the discriminating power, and both are the same mistake avoided twice.**

1. **``pe_table`` must be SCRAMBLED, never the identity ``(0, 1, 2, ...)``.** Under the identity
   table a builder that sorts, reverses, or re-derives the mapping from the loop index produces an
   observation identical to a correct one. Identity is the single pool value at which the defect
   contributes nothing -- the same shape as the LayerNorm variance bug that survived 598 tests
   because ``torch.randn`` is zero-mean and zero is the one row mean at which a padded-tail defect
   cancels exactly.
2. **The fake peer view must return a DISTINCT address per call.** With one shared address the test
   observes only the SEQUENCE of ``pe`` arguments, so a builder that calls in the right order and
   then pairs the results off by one -- or builds every atom from a single view -- passes unchanged.
   A constant address is to the pairing what the identity table is to the ordering.

**Why this needs no GPU, no nvshmem and no process group.** ``build_peer_store_atoms`` is plain
Python: it loops in host code and calls ``_get_peer_tensor_aligned`` per peer. Patching that
module-level name intercepts the only nvshmem-dependent step, and ``cutlass.Int32(n).value`` is
readable outside a trace (measured), so the ``pe`` each call received is observable host-side. The
remaining DSL work -- ``make_tiled_tma_atom`` -- emits layout algebra and needs only an MLIR
Context/Module/InsertionPoint, which the fixture supplies.

**NOT covered here, and where it is covered instead.** That the real ``nvshmem_ptr`` returns a
usable peer address, and that a store through one of these descriptors actually lands on the PE the
descriptor names. **Owner: port items 4/5.** This is obligation **B**, and it is deliberately
distinct from obligation **A** carried over from ``test_nvshmem_utils.py`` (that a peer view's
ALIGNMENT survives into a real TMA atom). A is about the pointer's annotation; B is about the
descriptor's identity, and a test for either passes while the other is broken. Item 2 covers B's
HOST-SIDE mapping half here; only B's runtime half defers.
"""

import cutlass
import cutlass.cute as cute
import pytest

from fold_cp_ops.testing.kernel_matrix import (
    Axis,
    KernelMatrix,
    computes_nothing_numeric,
    no_unsupported,
)
from fold_cp_ops.testing.numeric_guard import numeric_exempt

#: Base address and per-call stride for the fake peer views. The stride makes each call's address
#: distinct, which is what turns "the calls happened in this order" into "this atom came from that
#: call" -- see decision 2 in the module docstring.
_FAKE_BASE = 0x40000000
_FAKE_STRIDE = 0x100000

PEER_ATOMS = KernelMatrix(
    kernel="peer_tma_atoms",
    axes=(
        Axis(
            name="peer_case",
            domain=(
                "any (cp, pe_table) pair whose table length equals cp and whose entries are "
                "distinct global PE ids. The table is NOT required to be sorted or contiguous, and "
                "the pool deliberately weights SCRAMBLED tables: under the identity table a builder "
                "that sorts or re-derives the mapping from the loop index is indistinguishable from "
                "a correct one, so identity is the one value at which the defect this file exists "
                "for contributes nothing"
            ),
            values=(
                (2, (1, 0)),  # smallest scramble -- a pure swap
                (4, (3, 1, 0, 2)),  # no fixed point, not a rotation
                (4, (0, 1, 2, 3)),  # identity: the production-common case, kept for coverage only
                (8, (7, 3, 5, 1, 6, 2, 4, 0)),
                (8, (2, 3, 0, 1, 6, 7, 4, 5)),  # pairwise-swapped: a rotation bug would survive
                (16, tuple(range(15, -1, -1))),  # full reversal at the largest declared cp
            ),
            facets={
                "identity_order": lambda c: list(c[1]) == sorted(c[1]),
                "scrambled_order": lambda c: list(c[1]) != sorted(c[1]),
                "small_cp": lambda c: c[0] <= 4,
                "large_cp": lambda c: c[0] >= 8,
                # A table with no fixed point cannot be satisfied by "return the loop index".
                "no_fixed_point": lambda c: all(p != i for i, p in enumerate(c[1])),
            },
        ),
    ),
    computes=computes_nothing_numeric(
        because=(
            "the subject is a TMA DESCRIPTOR's identity -- which peer's base pointer each atom was "
            "baked against. Nothing is launched and no tensor is produced; the builder returns "
            "copy atoms and views, so there is no output whose element distribution could hide a "
            "defect. The defect this file guards produces perfectly valid numbers at the wrong PE"
        )
    ),
    unsupported=no_unsupported(
        because=(
            "ENUMERATED, not sampled: the module has exactly one public entry point "
            "(build_peer_store_atoms) and exactly two raise sites. The RuntimeError at :162 is "
            "UNREACHABLE BY CONSTRUCTION now that nvshmem_utils is ported -- HAS_PEER is True, so "
            "it can only fire on an image without nvshmem, where collection skips first. The "
            "ValueError at :167 is the only reachable refusal, and it fires on a MALFORMED input "
            "(a pe_table whose length disagrees with cp) rather than on any combination of "
            "declared axis values -- every (cp, pe_table) pair in the pool is valid by "
            "construction. It is covered directly by "
            "test_a_pe_table_that_does_not_match_cp_is_refused, with match= on front-door-only "
            "wording rather than a bare type"
        )
    ),
)


@pytest.fixture
def mlir_ctx():
    """An MLIR Context, Module and InsertionPoint -- what ``make_tiled_tma_atom`` emits into.

    Building a TMA atom is metaprogramming: it emits layout and type algebra the compiler folds
    away, but it still needs somewhere to emit. All three pieces are load-bearing -- the builder
    interns types in the **Context**, the ops it creates are owned by the **Module**, and the
    **InsertionPoint** says where they go. Supplying only the Context leaves the ops with nowhere to
    land, which shows up as heap corruption on a later call rather than an exception.

    Production never needs this: ``@cute.jit`` establishes all three for a trace, which is why
    ``build_peer_store_atoms`` carries no decorator and is still safe at its real call site.

    Yields: None. Used for its enter/exit side effect only.
    """
    from cutlass._mlir import ir

    with ir.Context(), ir.Location.unknown():
        module = ir.Module.create()
        with ir.InsertionPoint(module.body):
            yield


@pytest.fixture
def fake_peer_views(monkeypatch):
    """Intercept the only nvshmem-dependent step and hand back a DISTINCT view per call.

    Purpose: `build_peer_store_atoms` reaches nvshmem solely through the module-level
    ``_get_peer_tensor_aligned``. Patching that name makes the whole builder runnable host-side, and
    -- because the fake constructs the view -- lets the test know exactly which address every atom
    was built from.

    Semantics: each call returns a tensor based at ``_FAKE_BASE + _FAKE_STRIDE * call_index`` with
    the source layout preserved, mirroring what the real helper does (swap the base pointer, keep
    the layout). The per-call stride is deliberate: a shared address would let a builder that pairs
    results off by one, or reuses one view for every atom, pass unnoticed.

    Input requirements: none. `monkeypatch` restores the real symbol, which matters because the
    module object is process-wide and a leaked fake would silently disarm any later test.

    Yields: the list of ``pe`` values (as Python ints, read via ``Int32.value``) in call order, so a
        test can assert BOTH that the builder asked for the right peers and in the right sequence.
    """
    import fold_cp_ops.distributed.peer_tma_atoms as pta

    seen = []

    def _fake(tensor, pe, **kw):
        addr = _FAKE_BASE + _FAKE_STRIDE * len(seen)
        seen.append(int(pe.value) if hasattr(pe, "value") else int(pe))
        ptr = cute.make_ptr(
            tensor.element_type, cutlass.Int64(addr), cute.AddressSpace.gmem, assumed_align=16
        )
        return cute.make_tensor(ptr, tensor.layout)

    monkeypatch.setattr(pta, "_get_peer_tensor_aligned", _fake)
    return seen


def _recv_tensor(m=256, n=256):
    """A concrete-stride recv tensor standing in for the symmetric buffer. See T0.5 in the source."""
    ptr = cute.make_ptr(
        cutlass.BFloat16, cutlass.Int64(0x10000), cute.AddressSpace.gmem, assumed_align=16
    )
    return cute.make_tensor(ptr, cute.make_layout((m, n, 1)))


def _epi_layout_and_tile():
    """The base GEMM's epilogue SMEM layout and tile, bound the way the real caller binds them.

    ``cta_tile_k``, ``epi_tile`` and ``epi_smem_layout_staged`` are set by ``bind_operand_types``,
    not by construction -- reading them off a freshly constructed functor raises ``AttributeError``,
    which is a probe defect and not a statement about the builder.
    """
    from fold_cp_ops.kernels.gemm_sm90 import GemmSm90

    g = GemmSm90(cutlass.Float32, cutlass.BFloat16, (128, 128), (1, 1, 1))
    g.bind_operand_types(
        _recv_tensor(256, 64), _recv_tensor(256, 64), _recv_tensor(), None, None, None
    )
    return g.epi_smem_layout_staged, g.epi_tile


@PEER_ATOMS.parametrize("peer_case")
@numeric_exempt(
    "the subject is a TMA descriptor's baked peer identity, asserted on the builder's call record "
    "and its returned views. Nothing is launched and no tensor of values is produced, so there is "
    "no element-wise comparison to make"
)
def test_each_atom_is_built_against_its_own_peer(mlir_ctx, fake_peer_views, peer_case):
    """``atoms[r]`` is built from ``pe_table[r]`` -- in that order, one view each.

    This is the whole point of the file. The descriptor cannot be retargeted after the build, so a
    permuted mapping here means every store lands on the wrong PE for the rest of the run, silently.
    """
    import fold_cp_ops.distributed.peer_tma_atoms as pta

    cp, pe_table = peer_case
    epi_layout, epi_tile = _epi_layout_and_tile()
    atoms, peer_tensors = pta.build_peer_store_atoms(
        _recv_tensor(), cp, pe_table, epi_layout, epi_tile
    )
    assert len(atoms) == cp, f"expected {cp} atoms, got {len(atoms)}"
    assert len(peer_tensors) == cp, f"expected {cp} peer tensors, got {len(peer_tensors)}"
    assert fake_peer_views == list(pe_table), (
        f"the builder translated peers {fake_peer_views} but pe_table is {list(pe_table)}. "
        "Each atom's descriptor bakes its peer at build time and cannot be retargeted, so a "
        "permuted mapping stores every tile to the wrong PE with no error anywhere."
    )


@PEER_ATOMS.parametrize("peer_case")
@numeric_exempt(
    "asserts a REFUSAL at the front door, not a computed value; there is no output to compare"
)
def test_a_pe_table_that_does_not_match_cp_is_refused(peer_case):
    """A short or long ``pe_table`` raises ``ValueError`` naming both lengths, before any DSL work.

    The check is the module's only reachable refusal, and it runs before the loop, so a caller that
    passes a table for the wrong mesh gets a sentence rather than ``cp`` atoms of which some were
    built from garbage.
    """
    import fold_cp_ops.distributed.peer_tma_atoms as pta

    cp, pe_table = peer_case
    # `mRecv` is deliberately None: the length check runs BEFORE any operand is touched, so passing
    # a real tensor would only add an MLIR-context dependency this test does not otherwise need --
    # and a probe that dies constructing its input reports as a missing raise, which is how a
    # working front door reads as a broken one.
    with pytest.raises(ValueError, match="length must equal cp="):
        pta.build_peer_store_atoms(None, cp, pe_table[:-1], None, None)
