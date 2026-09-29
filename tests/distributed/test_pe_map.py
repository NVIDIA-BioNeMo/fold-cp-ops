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

"""Unit tests for the (DeviceMesh, placements) -> PE-map layer (T1.1, pe_map.py).

The PE-map is **deterministic layout algebra**, so the bulk of correctness is
verified **CPU-side, no GPU**: for every rank of a mesh we recompute
``flat_cp_peer -> global_PE`` and assert it matches a reference. We cover
cp in {2, 4, 8}, both a **flattened 1-D cp** (``cp=(N,)``) and **N-D cp grids**
(``cp=(2,2)``, ``cp=(2,4)``, ``cp=(2,2,2)``), with and without a ``dp`` axis.

A separate 2-rank GPU smoke (``test_pe_map_smoke_2rank``) confirms the map
composes with a live ``DistributedManager`` PE map (gated on >=2 CUDA devices).
"""

from __future__ import annotations

from math import prod

import numpy as np
import pytest

from fold_cp_ops.testing.collective_guard import rank_invariant_skip
import torch

from fold_cp_ops.distributed.layout_map import LayoutRightMap
from fold_cp_ops.distributed.pe_map import PeMap
from fold_cp_ops.testing.numeric_guard import numeric_exempt
from fold_cp_ops.testing.kernel_matrix import (
    Axis,
    KernelMatrix,
    computes_nothing_numeric,
    matrix_exempt,
    no_unsupported,
)


# --- offline fakes (no torch.distributed) -----------------------------------
class _FakeMesh:
    """Minimal DeviceMesh stand-in exposing only ``.mesh`` (the rank tensor)."""

    def __init__(self, rank_tensor: torch.Tensor):
        self.mesh = rank_tensor

    @property
    def ndim(self) -> int:
        return self.mesh.ndim


class _Shard:
    """Duck-typed DTensor ``Shard(dim)`` (avoids importing torch.dtensor)."""

    def __init__(self, dim: int):
        self.dim = dim

    def is_shard(self) -> bool:
        return True

    def is_replicate(self) -> bool:
        return False


class _Replicate:
    """Duck-typed DTensor ``Replicate()``."""

    def is_shard(self) -> bool:
        return False

    def is_replicate(self) -> bool:
        return True


def _row_major_rank_tensor(shape: tuple[int, ...]) -> torch.Tensor:
    """LayoutRight rank tensor (rank == row-major index), shape == mesh shape."""
    return torch.arange(prod(shape), dtype=torch.int64).reshape(shape)


def _reference_cp_pes(rank_tensor_np: np.ndarray, placements, my_rank: int):
    """Reference flat-cp PE list + my_cp_rank, computed independently of PeMap.

    Independent re-derivation: cp axes = sharded dims; flatten row-major; fix
    non-cp coords at this rank's position; index the rank tensor.
    """
    ndim = rank_tensor_np.ndim
    cp_axes = tuple(i for i, p in enumerate(placements) if p.is_shard())
    cp_sizes = tuple(int(rank_tensor_np.shape[a]) for a in cp_axes)
    cp = int(np.prod(cp_sizes))
    my_coord = tuple(int(c) for c in np.argwhere(rank_tensor_np == my_rank)[0])
    flat = LayoutRightMap(cp_sizes)
    pes = []
    my_cp_rank = -1
    for r in range(cp):
        cp_coord = flat.unravel(r)
        full = list(my_coord)
        for axis, c in zip(cp_axes, cp_coord):
            full[axis] = c
        pe = int(rank_tensor_np[tuple(full)])
        pes.append(pe)
        if pe == my_rank:
            my_cp_rank = r
    return pes, my_cp_rank, cp


# (mesh_shape, placements, expected_cp) — placements are one per mesh dim.
#: The declared mesh pool for this module -- the ONLY sanctioned source of (mesh shape, placements)
#: pairs here. It replaces a hand-rolled ``_CASES`` list: when every test writes its own, nothing
#: relates them, so a configuration nobody considered is indistinguishable from one considered and
#: excluded. Placements are encoded symbolically (``("S", dim)`` / ``("R",)``) because a parametrize
#: value must be HASHABLE and produce a stable id, which a list of DTensor placement objects is not;
#: :func:`_placements` decodes them back.
#:
#: Unlike the mesh axis in ``test_distributed_manager.py``, this pool is NOT bounded by the launch:
#: every case is built from a synthetic rank tensor with no process group, so ONE run exercises all
#: of it and the coverage ledger has nothing to reconcile.
PE_MESH = KernelMatrix(
    kernel="pe_map",
    axes=(
        Axis(
            name="mesh_case",
            domain=(
                "any (mesh_shape, placements, expected_cp) triple whose placement count equals the "
                "mesh rank and whose sharded-axis product equals expected_cp; no constraint on "
                "axis count, ordering, or power-of-two sizes"
            ),
            values=(
                # flattened 1-D cp (no dp)
                (((2,), (("S", 1),), 2)),
                (((4,), (("S", 1),), 4)),
                (((8,), (("S", 1),), 8)),
                # 1-D cp with a dp axis in front (dp replicated)
                (((1, 2), (("R",), ("S", 1)), 2)),
                (((2, 4), (("R",), ("S", 1)), 4)),
                # 2-D cp grids (the (cp0, cp1) case), no dp
                (((2, 2), (("S", 1), ("S", 2)), 4)),
                (((2, 4), (("S", 1), ("S", 2)), 8)),
                (((4, 2), (("S", 1), ("S", 2)), 8)),
                # dp + 2-D cp = the design's (dp, cp0, cp1) mesh
                (((1, 2, 2), (("R",), ("S", 1), ("S", 2)), 4)),
                (((2, 2, 2), (("R",), ("S", 1), ("S", 2)), 4)),
                (((1, 2, 4), (("R",), ("S", 1), ("S", 2)), 8)),
                # 3-D cp flattened (cp in 2..8 on a non-square factorization)
                (((2, 2, 2), (("S", 1), ("S", 2), ("S", 3)), 8)),
            ),
            facets={
                "flat_cp": lambda c: sum(1 for p in c[1] if p[0] == "S") == 1,
                "nd_cp": lambda c: sum(1 for p in c[1] if p[0] == "S") > 1,
                "has_dp": lambda c: any(p[0] == "R" for p in c[1]),
                "no_dp": lambda c: all(p[0] == "S" for p in c[1]),
                # dp=1 is the production distributed-TriMul shape, and the degenerate end besides.
                "unit_dp": lambda c: any(p[0] == "R" and c[0][i] == 1 for i, p in enumerate(c[1])),
                "nonunit_dp": lambda c: any(
                    p[0] == "R" and c[0][i] > 1 for i, p in enumerate(c[1])
                ),
                "cp_small": lambda c: c[2] <= 4,
                "cp_large": lambda c: c[2] >= 8,
            },
        ),
    ),
    computes=computes_nothing_numeric(
        because=(
            "the subject is a PE ADDRESSING table -- which global PE a flat cp peer index maps to. "
            "It launches no kernel and produces no tensor, so no input distribution can hide "
            "anything in it"
        )
    ),
    unsupported=no_unsupported(
        because=(
            "every declared triple is a valid (mesh, placements) pair by construction, and PeMap "
            "is placement-generic: the cp axes are exactly the sharded ones, at any count and any "
            "ordering. The refusals PeMap DOES have -- a placement/ndim mismatch and a mesh with no "
            "sharded axis -- are not combinations of declared axis values (they are malformed "
            "inputs), and are covered directly by test_pe_map_raises_on_bad_input."
        )
    ),
)


def _placements(spec):
    """Decode a symbolic placement tuple back into DTensor placement objects.

    Args:
        spec: Tuple of ``("S", dim)`` / ``("R",)`` pairs, as stored in `PE_MESH`. Encoded because a
            parametrize value must be hashable and produce a stable test id, which a list of
            placement objects is not.

    Returns:
        A list of ``Shard``/``Replicate``, in the same order -- which is what `PeMap` expects, since
        placement position is what binds a placement to a mesh axis.
    """
    return [_Shard(p[1]) if p[0] == "S" else _Replicate() for p in spec]


@numeric_exempt(
    "compares an integer PE ADDRESSING table against an independently computed reference -- "
    "exact identity on ints, not an approximation of a computed tensor; the matrix says the same "
    "with computes_nothing_numeric"
)
@PE_MESH.parametrize("mesh_case")
def test_pe_map_cpu_all_ranks(mesh_case):
    mesh_shape, placement_spec, expected_cp = mesh_case
    placements = _placements(placement_spec)
    """For every rank: PeMap flat-cp PE table matches the independent reference."""
    rank_tensor = _row_major_rank_tensor(mesh_shape)
    rank_np = rank_tensor.numpy()
    world = prod(mesh_shape)
    cp_axes_expected = tuple(i for i, p in enumerate(placements) if p.is_shard())
    cp_sizes_expected = tuple(mesh_shape[a] for a in cp_axes_expected)

    for my_rank in range(world):
        pm = PeMap.from_mesh_placements(
            _FakeMesh(rank_tensor), placements, my_rank=my_rank, device=torch.device("cpu")
        )
        ref_pes, ref_my_cp_rank, ref_cp = _reference_cp_pes(rank_np, placements, my_rank)

        assert pm.cp == expected_cp == ref_cp, f"cp mismatch rank {my_rank}"
        assert pm.cp_axes == cp_axes_expected
        assert pm.cp_axis_sizes == cp_sizes_expected
        assert pm.cp_pe_table.dtype == torch.int32
        assert pm.cp_pe_table.shape == (expected_cp,)
        assert pm.cp_pe_table.tolist() == ref_pes, (
            f"rank {my_rank} mesh {mesh_shape} pl {placements}: "
            f"got {pm.cp_pe_table.tolist()} != ref {ref_pes}"
        )
        assert pm.my_cp_rank == ref_my_cp_rank
        # self index points back at this rank
        assert ref_pes[pm.my_cp_rank] == my_rank


@matrix_exempt(
    "pins the ONE identity between a 1-D and a 2-D cp grid of the same size; it names both meshes on purpose, so drawing either from the pool would destroy the comparison"
)
def test_pe_map_flattened_equals_2d_membership():
    """Flattened 1-D cp=(4,) and the same 4 ranks as cp=(2,2): each rank's cp
    PE *set* is identical (the design's 'flatten cp0*cp1' must not change who
    talks to whom — only the indexing order may differ)."""
    rank_1d = _row_major_rank_tensor((4,))  # ranks 0..3, one cp group
    rank_2d = _row_major_rank_tensor((2, 2))  # same 4 ranks, 2x2
    for my_rank in range(4):
        pm_1d = PeMap.from_mesh_placements(
            _FakeMesh(rank_1d), [_Shard(1)], my_rank=my_rank, device=torch.device("cpu")
        )
        pm_2d = PeMap.from_mesh_placements(
            _FakeMesh(rank_2d), [_Shard(1), _Shard(2)], my_rank=my_rank, device=torch.device("cpu")
        )
        assert pm_1d.cp == pm_2d.cp == 4
        # full world is one cp group in both -> identical membership set {0,1,2,3}
        assert set(pm_1d.cp_pe_table.tolist()) == set(pm_2d.cp_pe_table.tolist()) == {0, 1, 2, 3}


@matrix_exempt(
    "asserts that a dp axis PARTITIONS the cp groups, which needs one specific dp>1 mesh and its expected partition written side by side"
)
def test_pe_map_dp_splits_cp_groups():
    """With dp=2, cp=2 (mesh (2,2), dp on axis0): rank r's cp group is exactly
    the 2 ranks sharing its dp coord — disjoint cp groups across dp."""
    rank_tensor = _row_major_rank_tensor((2, 2))  # [[0,1],[2,3]]; axis0=dp, axis1=cp
    expected = {0: [0, 1], 1: [0, 1], 2: [2, 3], 3: [2, 3]}
    for my_rank in range(4):
        pm = PeMap.from_mesh_placements(
            _FakeMesh(rank_tensor),
            [_Replicate(), _Shard(1)],
            my_rank=my_rank,
            device=torch.device("cpu"),
        )
        assert pm.cp == 2
        assert pm.cp_pe_table.tolist() == expected[my_rank], f"rank {my_rank}"


@matrix_exempt(
    "checks that the shard TENSOR dims are carried through, a property of the placements alone; the mesh shape is incidental and sweeping it would repeat one assertion 12 times"
)
def test_pe_map_shard_tensor_dims_recorded():
    """cp_shard_tensor_dims records which TENSOR dim each cp axis shards."""
    rank_tensor = _row_major_rank_tensor((1, 2, 2))
    pm = PeMap.from_mesh_placements(
        _FakeMesh(rank_tensor),
        [_Replicate(), _Shard(1), _Shard(2)],
        my_rank=0,
        device=torch.device("cpu"),
    )
    assert pm.cp_axes == (1, 2)
    assert pm.cp_shard_tensor_dims == (1, 2)


@matrix_exempt(
    "feeds MALFORMED inputs (placement/ndim mismatch, no sharded axis) that are by construction outside the declared pool, which only holds well-formed triples"
)
def test_pe_map_raises_on_bad_input():
    """Validation: placement/mesh-ndim mismatch and no-shard both raise."""
    rank_tensor = _row_major_rank_tensor((2, 2))
    with pytest.raises(ValueError, match="placements length"):
        PeMap.from_mesh_placements(
            _FakeMesh(rank_tensor), [_Shard(1)], my_rank=0, device=torch.device("cpu")
        )
    with pytest.raises(ValueError, match="No sharded mesh axis"):
        PeMap.from_mesh_placements(
            _FakeMesh(rank_tensor),
            [_Replicate(), _Replicate()],
            my_rank=0,
            device=torch.device("cpu"),
        )


# --- 2-rank GPU smoke: composes with a live DistributedManager ---------------
@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 2,
    reason="needs >=2 CUDA devices",
)
@matrix_exempt(
    "an unconditional placeholder for GPU coverage that lives in a separate torchrun script; it runs no code of its own"
)
def test_pe_map_smoke_2rank():
    """2-rank smoke (run under the torchrun harness / spawn) is provided as the
    standalone script tests/distributed/_pe_map_smoke.py; this placeholder marks
    the GPU-side coverage and is exercised there (kept out of in-process pytest
    to avoid nvshmem-init side effects in the collector)."""
    rank_invariant_skip(
        "GPU 2-rank smoke runs via tests/distributed/_pe_map_smoke.py (torchrun)",
        because="an unconditional placeholder -- it skips on every rank of every run, so there is "
        "no predicate that could differ",
    )


@matrix_exempt(
    "sweeps the declared pool ITSELF in a loop rather than by parametrize, because the property is about the whole pool being self-consistent; one failing case should name the case, not split the assertion"
)
def test_the_declared_pe_table_layout_describes_the_table_it_is_for():
    """``cp_layout_shape_stride`` must agree with ``cp_pe_table``'s actual shape, at every mesh.

    **The property is a promise about a DIFFERENT object**, which is what makes it worth checking
    rather than reading. It hands a kernel the ``(shape, stride)`` to rebuild the PE-table layout
    with inside a trace context, where `cute.make_layout` can run and this host code cannot. So a
    disagreement between the declared shape and the table's real extent is not a Python error --
    it is a kernel indexing a table by a length that is not the table's, off the end of a
    symmetric-heap allocation.

    It was reachable from no test at all before this one, in a module whose other accessors are
    exercised only incidentally.

    Swept over the same declared pool (`PE_MESH`) the rest of this file uses, because the whole content
    property is ``self.cp``, and ``cp`` is exactly what varies with the mesh.
    """
    for mesh_shape, placement_spec, expected_cp in PE_MESH.axis("mesh_case").values:
        placements = _placements(placement_spec)
        rank_tensor = _row_major_rank_tensor(mesh_shape)
        pm = PeMap.from_mesh_placements(
            _FakeMesh(rank_tensor), placements, my_rank=0, device=torch.device("cpu")
        )
        shape, stride = pm.cp_layout_shape_stride
        assert shape == (pm.cp,) == (expected_cp,), (
            f"mesh {mesh_shape} {placements}: declared layout shape {shape} disagrees with "
            f"cp={pm.cp}"
        )
        assert stride == (1,), f"the PE table is 1-D and contiguous; got stride {stride}"
        assert tuple(pm.cp_pe_table.shape) == shape, (
            f"mesh {mesh_shape} {placements}: declared layout shape {shape} disagrees with the "
            f"table it describes, which is {tuple(pm.cp_pe_table.shape)} -- a kernel rebuilding "
            f"the layout from this would index past the table"
        )
