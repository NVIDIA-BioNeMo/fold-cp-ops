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
"""Tests for ``fold_cp_ops.distributed.dtensor_adapter`` -- the placement classifier.

**This module computes no values. It answers four questions about a `(mesh, placements)` pair**, and
every one of them is a decision the rest of the distributed path depends on:

* ``validate_trimul_sharding`` -- is this sharding legal, and which mesh axes are the cp axes?
* ``_token_split_factor`` -- is the einsum LOCAL (``== 1``) or does it need a reshard (``> 1``)?
  **This is the dispatch decision.** Wrong low, and a kernel runs on operands that were never
  gathered; wrong high, and the workflow pays for communication it did not need.
* ``_effective_placements`` -- which ``Shard``s are real and which sit on a size-1 mesh axis?
* ``_is_rows_sharded_canonical`` -- is this the one layout the real A2A path supports?

**No tensor value can affect any of those answers**, which is what ``computes_nothing_numeric``
records here -- not the weaker "it produces no output". So the pool is over MESH CONFIGURATIONS, and
its discriminating properties are structural.

**Three pool decisions carry the discrimination, and all three are the same lesson.**

1. **A ``Shard`` on a SIZE-1 mesh axis must be in the pool.** ``_effective_placements`` exists only
   to normalize those, and the module's own docstring names the case: *"dp=1 or cp1=1 in the
   canonical ``(dp=1, cp0=cp, cp1=1)`` mesh"*. A pool whose mesh axes are all > 1 leaves that
   function a no-op on every value -- untested, while being the one whose failure makes
   ``PeMap``/``ReshardLayout`` treat tensor dim 0 as a cp axis. The singleton case is the PRODUCTION
   mesh, not a corner.
2. **``_token_split_factor == 1`` must be reached BOTH ways.** Either no token axis is sharded at
   all, or token axes are sharded only on size-1 mesh axes. They exercise different code, and a pool
   producing only one of them tests the local-dispatch path for the wrong reason while looking
   complete.
3. **The 2-D cp grid must have ASYMMETRIC axis sizes.** ``_token_split_factor`` is a PRODUCT over cp
   axes, and at ``(2, 2)`` the product, the sum, and "twice the first axis" are all 4 -- so a bug
   that adds instead of multiplying passes. At ``(2, 4)``: product 8, sum 6, max 4, first 2, last 4,
   and every plausible wrong reduction separates at once. ``(2, 2)`` is the symmetric value at which
   distinct operators agree, exactly as the identity ``pe_table`` is the permutation at which sort,
   reverse and identity agree.

**No GPU and no process group.** Placements are real ``Shard``/``Replicate`` objects (constructible
without a group) and the mesh is a stand-in exposing only ``.mesh``, following ``test_pe_map.py`` --
the one module whose mesh axis needs no coverage ledger, because a single run reaches every case.
"""

import pytest
import torch
from torch.distributed.tensor import Replicate, Shard

from fold_cp_ops.distributed.dtensor_adapter import (
    _effective_placements,
    _is_rows_sharded_canonical,
    _token_split_factor,
    validate_trimul_sharding,
)
from fold_cp_ops.testing.kernel_matrix import (
    Axis,
    KernelMatrix,
    computes_nothing_numeric,
    matrix_exempt,
    no_unsupported,
)
from fold_cp_ops.testing.numeric_guard import numeric_exempt


class _FakeMesh:
    """Minimal ``DeviceMesh`` stand-in exposing only ``.mesh``, whose shape is what is read.

    Input requirements: `shape` a tuple of positive ints, one entry per mesh dim, matching the
    length of the placements it is paired with. A real `DeviceMesh` needs a process group; nothing
    under test reads anything but `.mesh.shape`, so constructing one would buy no fidelity and cost
    a launcher.
    """

    def __init__(self, shape):
        self.mesh = torch.empty(shape, dtype=torch.int32)

    @property
    def ndim(self):
        return self.mesh.ndim


def _pl(spec):
    """Decode a symbolic placement tuple into real DTensor placements.

    Args:
        spec: tuple of ``("S", dim)`` / ``("R",)`` pairs. Encoded symbolically because a matrix
            axis's values are compared and hashed by the machinery, and placement objects are
            awkward to render into a test id.

    Returns: tuple of `Shard`/`Replicate`.
    """
    return tuple(Shard(p[1]) if p[0] == "S" else Replicate() for p in spec)


ADAPTER = KernelMatrix(
    kernel="dtensor_adapter",
    axes=(
        Axis(
            name="mesh_case",
            domain=(
                "any (mesh_shape, placements, expected_factor) triple whose placement count equals "
                "the mesh rank. The pool is REQUIRED to include (a) Shards on SIZE-1 mesh axes, "
                "because _effective_placements exists only to normalize those and the canonical "
                "production mesh (dp=1, cp0=cp, cp1=1) has two of them, so an all-axes>1 pool "
                "leaves that function a no-op on every value; (b) factor==1 reached BOTH ways -- no "
                "token shard at all, and token shards only on size-1 axes -- since they exercise "
                "different code; and (c) an ASYMMETRIC 2-D cp grid, because _token_split_factor is "
                "a PRODUCT and at (2,2) the product, the sum and twice-the-first are all 4, so a "
                "bug that adds instead of multiplying would pass"
            ),
            values=(
                # (mesh_shape, placement_spec, expected token-split factor)
                ((2,), (("S", 1),), 2),  # 1-D cp, rows split
                ((2,), (("S", 2),), 2),  # 1-D cp, cols split -- NOT canonical
                ((2,), (("R",),), 1),  # factor 1, way ONE: nothing sharded
                ((2,), (("S", 0),), 1),  # batch-only shard: legal, no token comm
                ((1, 2, 1), (("S", 0), ("S", 1), ("S", 2)), 2),  # canonical: TWO singleton axes
                ((1, 1, 1), (("S", 0), ("S", 1), ("S", 2)), 1),  # factor 1, way TWO: all singletons
                ((2, 4), (("S", 1), ("S", 2)), 8),  # ASYMMETRIC 2-D: product 8 != sum 6 != max 4
            ),
            facets={
                "has_singleton_axis": lambda c: any(s == 1 for s in c[0]),
                "all_axes_real": lambda c: all(s > 1 for s in c[0]),
                "local_einsum": lambda c: c[2] == 1,
                "needs_reshard": lambda c: c[2] > 1,
                "one_cp_axis": lambda c: (
                    sum(1 for p in c[1] if p[0] == "S" and p[1] in (1, 2)) <= 1
                ),
                "two_cp_axes": lambda c: sum(1 for p in c[1] if p[0] == "S" and p[1] in (1, 2)) > 1,
            },
        ),
    ),
    computes=computes_nothing_numeric(
        because=(
            "the subject is a PLACEMENT CLASSIFIER: every answer is a function of (mesh_shape, "
            "placements) alone, so no tensor value -- and therefore no input distribution -- can "
            "affect any of them. This is stronger than 'produces no output': there is no input "
            "whose distribution could hide a defect, because the inputs are not data"
        )
    ),
    unsupported=no_unsupported(
        because=(
            "every declared triple is a legal (mesh, placements) pair by construction. The "
            "module's three refusals -- a placements/mesh-ndim mismatch, a sharded FEATURE dim, and "
            "no token axis sharded -- are MALFORMED inputs rather than combinations of declared "
            "axis values, and are covered directly by "
            "test_the_validator_refuses_a_sharding_trimul_cannot_run"
        )
    ),
)


@ADAPTER.parametrize("mesh_case")
@numeric_exempt(
    "asserts a dispatch DECISION -- whether the einsum is local -- computed from a mesh shape and "
    "placements. Nothing is launched and no tensor is produced"
)
def test_the_token_split_factor_is_the_product_over_real_cp_axes(mesh_case):
    """``_token_split_factor`` multiplies the sizes of mesh axes that split a TOKEN dim.

    **This is the dispatch decision**: ``1`` sends the workflow to the monolithic local kernel and
    anything greater sends it through the reshard seam. Getting it wrong low runs a kernel on
    operands that were never gathered -- a wrong answer with nothing raised.

    The asymmetric ``(2, 4)`` case is what makes this a test of a PRODUCT rather than of any
    reduction: 8 separates it from the sum (6), the max (4) and either single axis (2, 4).
    """
    mesh_shape, spec, expected = mesh_case
    got = _token_split_factor(_FakeMesh(mesh_shape), _pl(spec))
    assert got == expected, (
        f"mesh {mesh_shape} with {spec} gives factor {got}, expected {expected}. This is the "
        "local-vs-reshard dispatch: too low runs the einsum on un-gathered operands, too high pays "
        "for communication that was not needed."
    )


@ADAPTER.parametrize("mesh_case")
@numeric_exempt("asserts which placements survive normalization, not a computed value")
def test_a_shard_on_a_size_one_axis_is_normalized_away(mesh_case):
    """``_effective_placements`` replaces a ``Shard`` on a size-1 mesh axis with ``Replicate``.

    A ``Shard`` on a singleton axis splits nothing, and leaving it in makes ``PeMap`` and
    ``ReshardLayout`` treat that mesh dim as a real cp axis -- so a batch-dim ``Shard`` on ``dp=1``
    would be mistaken for a token split. The canonical production mesh has two such axes, so this
    is the common path rather than an edge.
    """
    mesh_shape, spec, _ = mesh_case
    eff = _effective_placements(_FakeMesh(mesh_shape), _pl(spec))
    for axis, (size, p) in enumerate(zip(mesh_shape, eff)):
        if size == 1:
            assert not getattr(p, "is_shard", lambda: False)(), (
                f"mesh axis {axis} has size 1 but its placement survived as {p}. A Shard that "
                "splits nothing is read downstream as a real cp axis."
            )
    # A real (size>1) Shard must be kept exactly as it was -- normalization must not over-reach.
    for axis, (size, orig, p) in enumerate(zip(mesh_shape, _pl(spec), eff)):
        if size > 1:
            assert type(p) is type(orig), (
                f"mesh axis {axis} has size {size} and its placement changed from {orig} to {p}; "
                "normalization must only touch axes that split nothing"
            )


@ADAPTER.parametrize("mesh_case")
@numeric_exempt("asserts the returned cp mesh axes, not a computed value")
def test_the_validator_accepts_every_legal_sharding_and_names_its_cp_axes(mesh_case):
    """Legal placements validate, and the returned axes are exactly those splitting a token dim.

    The return value is consumed as "which mesh axes must the all-to-all run over", so an extra
    axis means a collective on a dim that carries no data and a missing one means data that never
    moves.
    """
    mesh_shape, spec, _ = mesh_case
    placements = _pl(spec)
    token_axes = tuple(i for i, p in enumerate(spec) if p[0] == "S" and p[1] in (1, 2))
    if not token_axes:
        # NOT a skip: a pool value with no token axis is a case the validator must REFUSE, so the
        # branch asserts that rather than stepping aside. A `pytest.skip` here would also be a bare
        # skip under tests/distributed/, which the collective guard refuses.
        with pytest.raises(ValueError, match="at least one token axis"):
            validate_trimul_sharding(placements, len(mesh_shape))
        return
    got = validate_trimul_sharding(placements, len(mesh_shape))
    assert got == token_axes, (
        f"validator returned cp axes {got} for {spec}; the token-sharding axes are "
        f"{token_axes}. This tuple selects the mesh dims the all-to-all runs over."
    )


@matrix_exempt(
    "asserts the three REFUSALS on malformed inputs -- a length mismatch, a sharded feature dim, "
    "and no token axis -- which are deliberately outside every declared axis value"
)
@numeric_exempt("asserts raises, not computed values")
def test_the_validator_refuses_a_sharding_trimul_cannot_run():
    """Each of the module's three refusals fires, with a message naming the cause.

    The feature-dim case is the one that matters most: LayerNorm and the dual-gated GEMM reduce over
    the FULL feature width, so a sharded ``D`` produces a plausible-looking wrong answer rather than
    an error anywhere downstream.
    """
    with pytest.raises(ValueError, match="placements length"):
        validate_trimul_sharding((Shard(1),), 2)
    with pytest.raises(ValueError, match="feature dim"):
        validate_trimul_sharding((Shard(3),), 1)
    with pytest.raises(ValueError, match="at least one token axis"):
        validate_trimul_sharding((Replicate(),), 1)


@ADAPTER.parametrize("mesh_case")
@numeric_exempt("asserts a layout classification, not a computed value")
def test_only_a_row_split_on_one_axis_is_the_canonical_layout(mesh_case):
    """``_is_rows_sharded_canonical`` is True iff the ONLY token axis split is tensor dim 1.

    The real A2A path's ``ReshardLayout`` packs ``(B, N_loc, N, D)`` assuming rows are the split
    axis; anything else -- a column split, or a 2-D cp grid -- must fall back to the all-gather
    reference. A classifier that were too permissive would route a column-sharded mesh into a packer
    that silently mis-tiles it.
    """
    mesh_shape, spec, _ = mesh_case
    mesh, placements = _FakeMesh(mesh_shape), _pl(spec)
    real = [(p[1], mesh_shape[i]) for i, p in enumerate(spec) if p[0] == "S" and mesh_shape[i] > 1]
    want = bool(real) and all(dim == 1 for dim, _ in real)
    got = _is_rows_sharded_canonical(mesh, placements)
    assert got == want, (
        f"mesh {mesh_shape} with {spec}: canonical={got}, expected {want}. The real A2A packer "
        "assumes rows (tensor dim 1) are the split axis; routing any other layout into it mis-tiles "
        "the shard without raising."
    )
