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

"""Torchrun-based unit tests for fold_cp_ops.distributed.DistributedManager (T0.1).

Exercises the DistributedManager public API surface under the T0.2 torchrun
harness (one process group + device_mesh per session; see conftest). Run with::

    CUDA_VISIBLE_DEVICES=2,3 torchrun --nproc_per_node=2 -m pytest -q \
        tests/distributed/test_distributed_manager.py

The device mesh is a PARAMETRIZED AXIS, declared once in :data:`DIST_MESH` below and applied per
test by the ``apply_mesh`` fixture. It is deliberately not a launch-time choice: when the mesh comes
from an env var or a CLI flag, the set of configurations a run covered is a property of the command
line, and a mesh nobody ever considered is indistinguishable from one considered and excluded. The
session builds only the world group; each test builds the grid it needs. nvshmem-dependent tests
skip cleanly when nvshmem init is unavailable (the harness itself only needs NCCL).

WORLD_SIZE still comes from the launch, so the pool spans mesh shapes for several world sizes and
each run covers the subset that fits, skipping the rest with a reason. (The world process group can
in fact be rebuilt in-process -- fresh ``MASTER_PORT`` per cycle, ``TORCHELASTIC_USE_AGENT_STORE``
dropped -- but rebuilding only the GRID is cheaper and needs neither, so that is what ``apply_mesh``
does when the mesh is the only thing varying.)

To sweep the whole pool::

    torchrun --nproc_per_node={2,4,8} -m pytest tests/distributed/test_distributed_manager.py

    # the 16-rank CROSS-NODE shapes -- 2 nodes x 8 GPUs, venue B. Both launchers, since
    # DistributedManager supports each and the `launch` axis has a cell for each:
    srun -N2 --ntasks-per-node=8 python -m pytest tests/distributed/test_distributed_manager.py
    srun -N2 --ntasks-per-node=1 torchrun --nnodes=2 --nproc_per_node=8 -m pytest <same>

The 16-rank cells skip on any single-node box with the reason naming what is missing (rank count or
node count), read from the ``topology`` fixture -- which is the manager's OWN
``_detect_nvshmem_traits()`` probe, so the suite and the nvshmem profile selection cannot form two
different opinions of the same hardware.

This is the torchrun-based DM unit test the project standardizes on: new
distributed tests should use this harness (torchrun + the session fixtures), not
``spawn_multiprocessing`` (kept only as a documented fallback for one-off meshes).
"""

from math import prod

import pytest
import torch
import torch.distributed as dist

from fold_cp_ops._internal.port_selection import ephemeral_floor
from fold_cp_ops.distributed import DistributedManager
from fold_cp_ops.distributed import distributed_manager as dist_manager_mod
from fold_cp_ops.testing.collective_guard import gated_skip, rank_invariant_skip
from fold_cp_ops.testing.kernel_matrix import (
    Axis,
    KernelMatrix,
    computes_nothing_numeric,
    matrix_exempt,
    no_unsupported,
)
from fold_cp_ops.testing.numerics import assert_elementwise, tolerance_bound
from fold_cp_ops.testing.numeric_guard import numeric_exempt


def _spec_numel(spec) -> int:
    """Ranks a mesh spec needs: the product of every group size, subgrids flattened.

    Args:
        spec: Tuple of ``(name, size)`` pairs; ``size`` is an ``int`` or a ``tuple[int, ...]``.

    Returns:
        The rank count. Used by the axis facets to tell a single-node shape from one that can only
        run across nodes, and by ``apply_mesh`` to skip a spec this launch cannot build.
    """
    total = 1
    for _, size in spec:
        total *= prod(size) if isinstance(size, tuple) else size
    return total


#: The declared device-mesh pool -- the ONLY sanctioned source of mesh configurations for this
#: module. A test may narrow it with ``only=``/``because=``; it may not invent a spec, because
#: ``parametrize`` rejects values outside the pool. That rejection IS the enforcement: adding a mesh
#: to a test means adding it to this pool, where the next reader can see the whole covered set at
#: once instead of reconstructing it from a dozen hand-rolled lists.
DIST_MESH = KernelMatrix(
    kernel="distributed_manager",
    axes=(
        Axis(
            name="mesh",
            domain=(
                "any OrderedDict of group-name -> int | tuple[int, ...] whose rank product equals "
                "WORLD_SIZE; a tuple value partitions that group into a subgrid, adding one "
                "<name>_axis_<i> group per axis. No constraint on group count, ordering, or "
                "power-of-two sizes -- a unit-size group is legal and is the common shape for "
                "distributed TriMul (dp=1)"
            ),
            values=(
                # flat, single group -- the 1-D context-parallel default, at three world sizes
                (("cp", 2),),
                (("cp", 4),),
                (("cp", 8),),
                # two groups, one of them unit-size: the production distributed-TriMul shape
                (("dp", 1), ("cp", 2)),
                (("dp", 1), ("cp", (2, 2))),
                # two groups, both non-unit: dp and cp genuinely split the world
                (("dp", 2), ("cp", 2)),
                (("dp", 2), ("cp", 4)),
                # subgrid without a sibling group -- cp alone factorized
                (("cp", (2, 2)),),
                (("cp", (2, 4)),),
                # unit-size LAST group: the degenerate end of the range, where cp spans one rank
                (("dp", 4), ("cp", 1)),
                # --- 16 ranks: the CROSS-NODE shapes. Only reachable on >=2 nodes (venue B), and
                # they are not just "bigger" -- under LayoutRight the last group's ranks are
                # contiguous, so each one places the IB boundary somewhere different:
                (("cp", 16),),  # one cp group spanning both nodes -> IB *inside* cp
                (("dp", 2), ("cp", 8)),  # cp == exactly one node -> cp all-NVLink, dp over IB
                (("dp", 8), ("cp", 2)),  # cp pairs, contiguous -> intra-node; dp straddles
                (("cp", (2, 8)),),  # subgrid: axis_0 crosses nodes, axis_1 stays on one
                # subgrid whose contiguous axis is HALF a node, not a whole one. Under LayoutRight
                # with 8 GPUs/node, cp_axis_1 = {0,1,2,3} sits INSIDE node 0 while cp_axis_0 =
                # {0,4,8,12} straddles -- so unlike (2,8), neither axis aligns with a node boundary.
                # That is the shape where a group is smaller than the NVLink domain it lives in.
                # LAUNCH IT WITH `srun`, NOT torchrun. This cell needs 16 ranks over 2 nodes, and
                # `srun -N2 --ntasks-per-node=8` reaches that directly. The torchrun path CAN do it
                # (CLAUDE.md documents the form) but only wrapped in an srun that places one agent
                # per node and derives --node-rank/--master-addr per task, so in practice the
                # `launch=srun` cell is the one that runs and `launch=torchrun` skips -- measured:
                # `mesh(('cp',(4,4)),)-launchsrun` passed, `-launchtorchrun` skipped, on the same run.
                # Do NOT pass --gpus-per-task with 8 tasks/node: each task then sees ONE GPU while
                # LOCAL_WORLD_SIZE=8, every test skips, and the run exits GREEN having tested nothing.
                (("cp", (4, 4)),),
            ),
            facets={
                "has_subgrid": lambda s: any(isinstance(sz, tuple) for _, sz in s),
                "flat": lambda s: all(isinstance(sz, int) for _, sz in s),
                "multi_group": lambda s: len(s) > 1,
                "single_group": lambda s: len(s) == 1,
                "has_unit_group": lambda s: any(sz == 1 for _, sz in s),
                # Needs more ranks than one DGX node holds, so it can only run multi-node.
                "cross_node": lambda s: _spec_numel(s) > 8,
                "single_node": lambda s: _spec_numel(s) <= 8,
            },
        ),
        Axis(
            name="launch",
            domain=(
                "DistributedManager supports two launchers: torchrun (env:// rendezvous, exports "
                "RANK/WORLD_SIZE/LOCAL_RANK) and a bare `srun --ntasks-per-node=N` where "
                "_initialize_slurm reads SLURM_PROCID/NTASKS/LOCALID directly. Both must yield "
                "identical rank identity and mesh topology"
            ),
            values=("torchrun", "srun"),
        ),
    ),
    # A mesh is refused for MISMATCHING WORLD_SIZE, and world size is not a declared axis -- it is
    # fixed by the launch, so the refusal is a property of the (mesh, launch) pair rather than of any
    # combination of declared axis values, and `where=` can only read declared axes. Every spec above
    # is legal at the world size it is selected for. The refusal itself is covered directly by
    # test_grid_group_refuses_mesh_that_mismatches_world_size.
    computes=computes_nothing_numeric(
        because=(
            "the subject is a mesh LAYOUT -- which rank sits where, which groups exist. "
            "It launches no kernel and produces no tensor, so no input distribution can "
            "hide anything in it"
        )
    ),
    unsupported=no_unsupported(
        because=(
            "every declared mesh shape is valid at the WORLD_SIZE it is selected for; the only "
            "refusal is a mesh/world-size mismatch, and world size is not an axis (the launch fixes "
            "it), so no combination of declared axis values is unsupported. The mismatch refusal is "
            "tested by test_grid_group_refuses_mesh_that_mismatches_world_size."
        )
    ),
)


# --------------------------------------------------------------------------- #
# Borg singleton + initialization state.
# --------------------------------------------------------------------------- #
@matrix_exempt(
    "asserts a manager property that does not vary with the device mesh -- rank/world identity, singleton state, nvshmem bring-up and the symmetric-memory backend are all one-per-process facts. Sweeping DIST_MESH here would rebuild the grid 15 times to re-assert the same value"
)
def test_borg_singleton_shares_state(dist_manager, rank):
    """Every DistributedManager() shares one ``_state`` (Borg), not per-instance."""
    a = DistributedManager()
    b = DistributedManager()
    assert a.__dict__ is b.__dict__, "Borg instances must share __dict__ (_state)"
    assert a.rank == b.rank == rank


@matrix_exempt(
    "asserts a manager property that does not vary with the device mesh -- rank/world identity, singleton state, nvshmem bring-up and the symmetric-memory backend are all one-per-process facts. Sweeping DIST_MESH here would rebuild the grid 15 times to re-assert the same value"
)
def test_is_initialized(dist_manager):
    """is_initialized() reports True once the session fixture has initialized."""
    assert DistributedManager.is_initialized() is True


@matrix_exempt(
    "asserts a manager property that does not vary with the device mesh -- rank/world identity, singleton state, nvshmem bring-up and the symmetric-memory backend are all one-per-process facts. Sweeping DIST_MESH here would rebuild the grid 15 times to re-assert the same value"
)
def test_methods_init_available_and_backend(dist_manager):
    """The init-method registry and the device->backend map are well-formed."""
    methods = DistributedManager.methods_init_available()
    assert "ENV" in methods, f"expected ENV in init methods, got {methods}"
    backends = DistributedManager.backend_for_device()
    assert isinstance(backends, dict) and backends, "backend_for_device() must be a non-empty dict"


# --------------------------------------------------------------------------- #
# Rank / device identity — DM agrees with torch.distributed and the env.
# --------------------------------------------------------------------------- #
@matrix_exempt(
    "asserts a manager property that does not vary with the device mesh -- rank/world identity, singleton state, nvshmem bring-up and the symmetric-memory backend are all one-per-process facts. Sweeping DIST_MESH here would rebuild the grid 15 times to re-assert the same value"
)
def test_rank_world_size_match_torch(dist_manager):
    """DM rank/world_size agree with the live torch.distributed group."""
    assert dist_manager.rank == dist.get_rank()
    assert dist_manager.world_size == dist.get_world_size()


@matrix_exempt(
    "asserts a manager property that does not vary with the device mesh -- rank/world identity, singleton state, nvshmem bring-up and the symmetric-memory backend are all one-per-process facts. Sweeping DIST_MESH here would rebuild the grid 15 times to re-assert the same value"
)
def test_local_rank_pins_device(dist_manager, local_rank, device):
    """DM.local_rank indexes the bound CUDA device (cuda:LOCAL_RANK)."""
    assert dist_manager.local_rank == local_rank
    assert dist_manager.device == device
    assert dist_manager.device.type == "cuda"
    assert dist_manager.device.index == local_rank


# --------------------------------------------------------------------------- #
# Device mesh + process groups.
# --------------------------------------------------------------------------- #
@DIST_MESH.parametrize("mesh")
@numeric_exempt("asserts a mesh's RANK COVERAGE, not a computed tensor")
def test_device_mesh_spans_world(apply_mesh, mesh, world_size):
    """Whatever its shape, a built device_mesh covers exactly WORLD_SIZE ranks."""
    manager = apply_mesh(mesh)
    assert manager.device_mesh.size() == world_size
    assert prod(tuple(manager.device_mesh.shape)) == world_size


@DIST_MESH.parametrize("mesh")
@numeric_exempt("asserts process-group object identity, not a computed tensor")
def test_repeated_mesh_spec_reuses_the_device_mesh_and_its_groups(apply_mesh, mesh):
    """Applying the SAME mesh spec twice reuses one ``DeviceMesh`` and one set of process groups.

    Functionality & semantics:
        Applies ``mesh``, records the ``DeviceMesh`` and the per-axis process-group objects, applies
        the identical spec again, and asserts every one is the SAME OBJECT. Object identity, not
        equality: two distinct NCCL communicators over the same ranks compare equal on every
        observable this test could otherwise check, and it is the second communicator's existence
        that is the defect.

        This is a LEAK test. ``torch``'s ``DeviceMesh`` has no construction cache -- a dim spanning
        the whole world early-returns the default PG, but a genuine submesh dim calls
        ``split_group``/``new_group`` on every construction. A per-test mesh therefore minted fresh
        NCCL communicators per test: measured at world 16, ``cp=(4,4)`` cost +250 MiB on each of
        three constructions, and a 1060-test session reached 68 GB of non-torch device memory and
        exhausted an 80 GB card. NCCL buffers are outside torch's allocator, so ``memory_reserved()``
        cannot see them and ``empty_cache()`` cannot release them.

    Input requirements:
        ``apply_mesh``: the fixture factory; called twice with the same spec, which is what a real
        session does across two tests that declare the same mesh. ``mesh``: a spec from the declared
        matrix whose rank product matches WORLD_SIZE (``apply_mesh`` skips the rest, collectively).

    Returns:
        None.
    """
    manager = apply_mesh(mesh)
    first_mesh = manager.device_mesh
    first_groups = {n: g for n, g in manager.group.items() if n != "world"}
    assert first_groups, "spec registered no non-world group; the test would assert nothing"

    manager = apply_mesh(mesh)
    assert manager.device_mesh is first_mesh, (
        "re-applying the same mesh spec built a NEW DeviceMesh; its submesh dims re-issue "
        "split_group/new_group, which is the per-call NCCL communicator leak"
    )
    for name, group in first_groups.items():
        assert manager.group[name] is group, (
            f"axis {name!r} got a different process-group object on the second apply of the same "
            "spec -- a fresh communicator, not a reuse"
        )


@numeric_exempt("asserts process-group object identity, not a computed tensor")
@matrix_exempt(
    "the subject is the SUBGRID path specifically -- a tuple-sized axis, which is the only shape "
    "that mints real submesh communicators -- so it must name one such spec directly rather than "
    "sweep a pool in which most entries are flat and exercise nothing"
)
def test_repeated_subgrid_spec_reuses_the_axis_groups(apply_mesh, world_size):
    """A repeated SUBGRID spec reuses its subgroups mesh and its per-axis process groups.

    Functionality & semantics:
        This is the case that actually leaked. A flat ``cp=N`` axis spans the whole world, and
        torch's ``DeviceMesh._init_process_groups`` early-returns the existing default PG for such a
        dim -- free, and shared with every other flat spec, which is correct rather than a
        collision. Only a TUPLE-sized axis builds genuine submeshes, via ``split_group``/``new_group``
        per dim, and torch has no construction cache for them. Measured at world 16 before the fix:
        ``cp=(4,4)`` cost +250 MiB on each of three constructions (+250/+500/+750); after, +250 once
        and then flat.

        Note ``device_mesh`` stays the FLAT product mesh for a tuple spec -- the 2-D structure lives
        in ``device_mesh_subgroups`` with ``<name>_axis_<i>`` groups -- so those are what this
        asserts on. Object identity, not equality: two distinct communicators over the same ranks
        compare equal on everything else, and it is the second one's existence that is the defect.

    Input requirements:
        Needs ``world_size % 4 == 0`` for a 2-D factorisation. WORLD_SIZE is one number for the whole
        job, so the skip is rank-invariant.

    Returns:
        None.
    """
    if world_size % 4:
        rank_invariant_skip(
            f"needs world_size % 4 == 0 for a 2-D cp factorisation; got {world_size}",
            because="WORLD_SIZE is one number for the whole job, so every rank computes this "
            "comparison identically",
        )
    spec = (("cp", (world_size // 4, 4)),)
    first = apply_mesh(spec)
    first_sub = first.device_mesh_subgroups
    first_axes = {n: g for n, g in first.group.items() if n.startswith("cp_axis_")}
    assert first_sub is not None, "a tuple-sized axis registered no subgroups mesh"
    assert first_axes, (
        "a tuple-sized axis registered no cp_axis_* groups; nothing would be asserted"
    )

    second = apply_mesh(spec)
    assert second.device_mesh_subgroups is first_sub, (
        "re-applying the same SUBGRID spec built a new subgroups DeviceMesh; its dims re-issue "
        "split_group/new_group, which is the per-call NCCL communicator leak"
    )
    for name, group in first_axes.items():
        assert second.group[name] is group, (
            f"axis {name!r} got a different process-group object on the second apply of the same "
            "subgrid spec -- a fresh communicator, not a reuse"
        )


@DIST_MESH.parametrize("mesh")
@numeric_exempt("asserts which process groups exist, not a computed tensor")
def test_declared_groups_are_registered(apply_mesh, mesh, world_size):
    """Every group named in the spec appears in group_ranks, with the size the spec asked for.

    The mesh axis earns its keep here: a flat ``cp=N`` and a subgrid ``cp=(a, b)`` take different
    branches of ``create_grid_group``, and only the subgrid one registers ``<name>_axis_<i>``
    groups. Sweeping the pool exercises both without a second test.
    """
    manager = apply_mesh(mesh)
    group_ranks = manager.group_ranks
    for name, size in mesh:
        assert name in group_ranks, f"group {name!r} missing from {sorted(group_ranks)}"
        assert len(group_ranks[name]) == (prod(size) if isinstance(size, tuple) else size)
        if isinstance(size, tuple):
            for i, axis_len in enumerate(size):
                axis_name = f"{name}_axis_{i}"
                assert axis_name in group_ranks, f"subgrid axis {axis_name!r} not registered"
                assert len(group_ranks[axis_name]) == axis_len


@DIST_MESH.parametrize("mesh")
@numeric_exempt("asserts a rank-to-group mapping, not a computed tensor")
def test_group_rank_is_consistent(apply_mesh, mesh, rank):
    """For every registered group, this rank's group-rank indexes it in group_ranks."""
    manager = apply_mesh(mesh)
    for name, ranks in manager.group_ranks.items():
        if rank in ranks:
            gr = manager.group_rank[name]
            assert ranks[gr] == rank, (
                f"group {name!r}: group_rank {gr} -> {ranks[gr]} != global rank {rank}"
            )


@DIST_MESH.parametrize(
    "mesh",
    only={"mesh": ((("dp", 1), ("cp", (2, 2))), (("cp", (2, 2)),), (("cp", (2, 4)),))},
    because="subgroup factorization only exists for a spec with a subgrid value",
)
@numeric_exempt("asserts subgroup membership, not a computed tensor")
def test_subgroups_consistent(apply_mesh, mesh, rank):
    """For a subgrid group (cp = cp_axis_0 x cp_axis_1) the factorization is consistent.

    ``subgroups_ranks[g]`` lists the per-mesh-dim subgroups passing THROUGH this rank -- the axes of
    the subgrid intersecting here -- so their sizes multiply to the parent group size and this rank
    lies on every axis. They do NOT tile the ranks; the axes overlap at this rank.
    """
    manager = apply_mesh(mesh)
    assert manager.has_subgroups, f"spec {mesh} declares a subgrid but has_subgroups is False"
    assert manager.device_mesh_subgroups is not None
    sg_ranks = manager.subgroups_ranks
    assert isinstance(sg_ranks, dict) and sg_ranks, "subgroups_ranks must be a non-empty dict"
    group_ranks = manager.group_ranks
    for name, axes in sg_ranks.items():
        parent_size = len(group_ranks[name]) if name in group_ranks else manager.world_size
        axis_lens = [len(axis) for axis in axes]
        assert prod(axis_lens) == parent_size, (
            f"subgroup {name!r}: axis sizes {axis_lens} don't multiply to parent size {parent_size}"
        )
        for i, axis in enumerate(axes):
            assert rank in axis, f"subgroup {name!r} axis {i} ({axis}) lacks this rank {rank}"


@DIST_MESH.parametrize(
    "mesh",
    "launch",
    only={
        "mesh": (
            (("cp", 16),),
            (("dp", 2), ("cp", 8)),
            (("dp", 8), ("cp", 2)),
            (("cp", (2, 8)),),
            (("cp", (4, 4)),),
        )
    },
    because="the IB boundary only exists for a mesh too large for one node; the <=8-rank shapes are covered by the tests above",
)
@numeric_exempt("asserts where the IB boundary falls in the layout, not a computed tensor")
def test_cross_node_mesh_places_the_ib_boundary_where_the_layout_says(
    apply_mesh, topology, mesh, launch, world_size
):
    """A 16-rank mesh must put each group on the node(s) the LayoutRight rank order implies.

    This is the case the local boxes cannot reach: with 16 ranks over 2 nodes, some groups are
    all-NVLink and some straddle the IB fabric, and WHICH is which follows from the rank layout
    rather than from the group's name. ``LayoutRightMap`` makes the LAST group's ranks contiguous,
    so ``dp=2,cp=8`` puts each cp group entirely on one node while dp crosses, and ``cp=16`` puts
    the boundary inside cp. Getting this wrong routes an A2A that assumed NVLink over IB, where a
    device-issued signal hangs -- so it is worth asserting rather than assuming.

    Runs under BOTH launchers. ``DistributedManager`` accepts torchrun (``env://`` rendezvous) and a
    bare ``srun --ntasks-per-node=N`` (``_initialize_slurm`` reads SLURM's task vars directly);
    the mesh they produce must be identical, since a rank identity that
    differs by launcher would silently reshard the model. The cell for the launcher not in use skips
    -- covering both means running this file twice on venue B, once per launcher.
    """
    if topology["launch"] != launch:
        rank_invariant_skip(
            f"launched via {topology['launch']!r}, not {launch!r}",
            because="the launcher is one property of the job; every rank of a torchrun run reports "
            "torchrun and every rank of an srun run reports srun",
        )
    if world_size != _spec_numel(mesh):
        rank_invariant_skip(
            f"mesh needs {_spec_numel(mesh)} ranks; WORLD_SIZE={world_size}",
            because="mesh is a declared matrix value and WORLD_SIZE is one number for the job, so "
            "every rank computes this comparison identically",
        )
    if topology["n_nodes"] < 2:
        rank_invariant_skip(
            f"needs >=2 nodes for an IB boundary; n_nodes={topology['n_nodes']} "
            f"(local GPUs={topology['n_local_gpus']}). Run on venue B: 2 nodes x 8 GPUs.",
            because="node count is one property of the allocation; `topology` is the manager's own "
            "_detect_nvshmem_traits() probe, which every rank runs against the same job",
        )

    manager = apply_mesh(mesh)
    per_node = topology["local_world_size"]
    node_of = lambda r: r // per_node
    group_ranks = manager.group_ranks

    straddling = {
        name: ranks
        for name, ranks in group_ranks.items()
        if name != "world" and len({node_of(r) for r in ranks}) > 1
    }
    assert "world" in group_ranks and len(group_ranks["world"]) == world_size
    if mesh == (("dp", 2), ("cp", 8)):
        # cp is the LAST group -> contiguous -> exactly one node each; dp must therefore cross.
        assert "cp" not in straddling, (
            f"cp {group_ranks['cp']} should be intra-node, per LayoutRight"
        )
        assert "dp" in straddling, f"dp {group_ranks['dp']} should cross the IB boundary"
    elif mesh == (("cp", 16),):
        assert "cp" in straddling, "a 16-rank cp group must span both nodes"
    elif mesh == (("dp", 8), ("cp", 2)):
        assert "cp" not in straddling, f"contiguous cp pairs {group_ranks['cp']} must stay on-node"
        assert "dp" in straddling, f"dp {group_ranks['dp']} should cross the IB boundary"
    else:
        # The two subgrid specs, (2,8) and (4,4). In BOTH, axis_0 is strided and therefore crosses,
        # while axis_1 is contiguous and does not -- but the contiguous axis differs in kind:
        # (2,8) gives axis_1 = exactly one node (8 ranks), (4,4) gives HALF a node (4 ranks). The
        # assertion is the same; what (4,4) adds is a group SMALLER than its NVLink domain, so a
        # peer-mapping that assumed "contiguous == the whole node" would pass (2,8) and fail here.
        stride = mesh[0][1][1]  # ranks between consecutive axis_0 members
        assert "cp_axis_0" in straddling, (
            f"cp_axis_0 (stride {stride}) must cross the node boundary; got {group_ranks['cp_axis_0']}"
        )
        assert "cp_axis_1" not in straddling, (
            f"cp_axis_1 (contiguous, {stride} ranks) must stay on one node; "
            f"got {group_ranks['cp_axis_1']}"
        )


@matrix_exempt(
    "the refusal is about a mesh/world-size MISMATCH, so the spec must come from outside the pool by construction"
)
def test_grid_group_refuses_mesh_that_mismatches_world_size(dist_manager, world_size):
    """A mesh whose rank product differs from WORLD_SIZE must be refused, not silently reshaped.

    This is the one refusal the matrix cannot express as an unsupported region (world size is not a
    declared axis), so it is tested directly. Building the mesh anyway would hand every downstream
    peer-indexed transfer a rank set that does not exist.
    """
    from collections import OrderedDict

    with pytest.raises(RuntimeError, match="does not match"):
        DistributedManager.create_grid_group(OrderedDict(cp=world_size + 1))


@matrix_exempt(
    "asserts a manager property that does not vary with the device mesh -- rank/world identity, singleton state, nvshmem bring-up and the symmetric-memory backend are all one-per-process facts. Sweeping DIST_MESH here would rebuild the grid 15 times to re-assert the same value"
)
def test_world_group_ranks_complete(dist_manager, world_size):
    """A 'world' group exists and its global ranks are exactly 0..world_size-1."""
    group_ranks = dist_manager.group_ranks
    assert isinstance(group_ranks, dict) and group_ranks, "group_ranks must be a non-empty dict"
    assert "world" in group_ranks, f"expected a 'world' group, got {list(group_ranks)}"
    assert sorted(group_ranks["world"]) == list(range(world_size))


# --------------------------------------------------------------------------- #
# Ad-hoc group creation.
# --------------------------------------------------------------------------- #
@matrix_exempt(
    "asserts a manager property that does not vary with the device mesh -- rank/world identity, singleton state, nvshmem bring-up and the symmetric-memory backend are all one-per-process facts. Sweeping DIST_MESH here would rebuild the grid 15 times to re-assert the same value"
)
def test_create_group_registers(dist_manager, world_size):
    """create_group registers a new named group spanning the given ranks.

    All ranks must call new_group collectively (torch requirement), so build a
    group over ALL ranks and check it lands in the group_ranks registry.
    """
    all_ranks = list(range(world_size))
    name = "ut_allranks"
    if name not in dist_manager.group_ranks:
        DistributedManager.create_group(name, all_ranks)
    assert name in dist_manager.group_ranks
    assert sorted(dist_manager.group_ranks[name]) == all_ranks


# --------------------------------------------------------------------------- #
# NVSHMEM PE maps — guarded; the harness needs only NCCL, so nvshmem may be off.
# --------------------------------------------------------------------------- #
@pytest.fixture
def nvshmem_ready(dist_manager):
    """Init nvshmem once for the session; skip the dependent tests if unavailable, ON EVERY RANK.

    The skip predicate here is **per-host, not per-job**, which is why it is routed through
    `CollectiveGate.should_skip` rather than calling `pytest.skip` directly. nvshmem can bootstrap
    on one node and fail on another for reasons that are genuinely local -- a missing
    ``libnvshmem_host.so`` on one image, no IB device on one host, a UID handshake that one rank
    times out of. A bare `pytest.skip` then fires on a SUBSET of ranks, and the ranks that did not
    skip walk into the next collective alone and block until a watchdog kills the job, with a
    traceback naming whatever test they were in rather than this fixture.

    That is not hypothetical here: `init_nvshmem` is itself a collective bootstrap, so the failing
    rank leaves at the moment its peers most need it to arrive.

    Semantics
        Every rank runs the init, records its own outcome as a reason-or-``None``, and then reaches
        `should_skip` UNCONDITIONALLY -- the reduction is the point, and a rank that returned early
        would be the exact defect this guards. If any rank has a reason, all ranks skip; the rank
        that did not observe the failure says so rather than claiming a condition it never saw.

    Args:
        dist_manager: The session `DistributedManager`. Its process group must already be up --
            `CollectiveGate` refuses an uninitialized group -- which this fixture's dependency
            guarantees.

    Returns:
        The `dist_manager`, once every rank agrees nvshmem is usable.

    Raises:
        Nothing directly. Calls `pytest.skip` on ALL ranks or none.
    """
    local_reason = None
    try:
        DistributedManager.init_nvshmem()
        if not dist_manager.nvshmem_initialized:
            local_reason = "nvshmem did not initialize"
    except Exception as e:  # noqa: BLE001 — any nvshmem/bootstrap failure -> skip, not error
        local_reason = f"nvshmem init unavailable: {type(e).__name__}: {e}"
    gated_skip(local_reason)
    return dist_manager


@matrix_exempt(
    "asserts a manager property that does not vary with the device mesh -- rank/world identity, singleton state, nvshmem bring-up and the symmetric-memory backend are all one-per-process facts. Sweeping DIST_MESH here would rebuild the grid 15 times to re-assert the same value"
)
def test_nvshmem_initialized_flag(nvshmem_ready):
    """nvshmem_initialized flips True after init_nvshmem()."""
    assert nvshmem_ready.nvshmem_initialized is True


@matrix_exempt(
    "asserts a manager property that does not vary with the device mesh -- rank/world identity, singleton state, nvshmem bring-up and the symmetric-memory backend are all one-per-process facts. Sweeping DIST_MESH here would rebuild the grid 15 times to re-assert the same value"
)
def test_init_nvshmem_idempotent(nvshmem_ready):
    """init_nvshmem() is idempotent (a second call must not raise / re-init)."""
    DistributedManager.init_nvshmem()
    assert nvshmem_ready.nvshmem_initialized is True


@matrix_exempt(
    "asserts a manager property that does not vary with the device mesh -- rank/world identity, singleton state, nvshmem bring-up and the symmetric-memory backend are all one-per-process facts. Sweeping DIST_MESH here would rebuild the grid 15 times to re-assert the same value"
)
def test_symmetric_backend_is_nvshmem(nvshmem_ready, device):
    """init_nvshmem() must leave the symmetric-memory backend on NVSHMEM, not torch's default.

    This is the load-bearing new invariant, and the reason it needs a test of its own is that the
    alternative fails SILENTLY. Measured on this build: with the default backend, an
    ``empty`` + ``rendezvous`` still SUCCEEDS and still returns non-NULL peer pointers (CUDA IPC) --
    while ``nvshmem.bindings.n_pes()`` stays 0, i.e. nvshmem was never initialized. Every in-kernel
    ``direct.my_pe()`` would then read 0 on every rank, and the buffers are single-node besides. No
    peer-pointer or allocation check catches that; only the backend name does.
    """
    import torch.distributed._symmetric_memory as symm_mem

    backend = symm_mem.get_backend(device)
    assert backend == "NVSHMEM", (
        f"symmetric-memory backend is {backend!r} after init_nvshmem(); a non-NVSHMEM backend "
        "cannot serve the in-kernel nvshmem device API and degrades silently"
    )


@matrix_exempt(
    "asserts a manager property that does not vary with the device mesh -- rank/world identity, singleton state, nvshmem bring-up and the symmetric-memory backend are all one-per-process facts. Sweeping DIST_MESH here would rebuild the grid 15 times to re-assert the same value"
)
def test_nvshmem_pe_identity_matches_torch_rank(nvshmem_ready, rank, world_size):
    """nvshmem's PE numbering must equal torch's rank numbering.

    ``PeMap`` builds ``cp_pe_table`` straight from the ``DeviceMesh`` rank tensor, so PE == global
    rank is the assumption the whole peer-indexed transfer path rests on. It used to hold BY
    CONSTRUCTION (``rank=`` was handed to ``nvshmem.core.init``); now torch's bootstrap assigns the
    numbers, so it is a property to be checked rather than one we set. A mismatch misroutes A2A
    payloads with no error raised anywhere.
    """
    nb = pytest.importorskip("nvshmem.bindings")

    assert int(nb.my_pe()) == rank, f"nvshmem PE {int(nb.my_pe())} != torch rank {rank}"
    assert int(nb.n_pes()) == world_size, f"nvshmem n_pes {int(nb.n_pes())} != world {world_size}"


@matrix_exempt(
    "asserts a manager property that does not vary with the device mesh -- rank/world identity, singleton state, nvshmem bring-up and the symmetric-memory backend are all one-per-process facts. Sweeping DIST_MESH here would rebuild the grid 15 times to re-assert the same value"
)
def test_symmetric_allocation_after_init_is_usable(
    nvshmem_ready, topology, device, rank, world_size
):
    """A symmetric tensor allocated AFTER init_nvshmem() rendezvouses with real peer mappings.

    init_nvshmem() drops its one-byte bootstrap tensor on purpose, so this pins the consequence that
    matters: the heap stays up and later allocations -- the ones the A2A kernels actually use -- map
    every peer this rank can reach directly. COLLECTIVE: every rank must reach both calls, in the
    same order, or it hangs rather than fails.

    **A NULL peer pointer is not a failure, and asserting otherwise is wrong.** ``nvshmem_ptr``
    returns a loadable address only for a peer in the same NVLink domain and **0 for an IB peer** --
    so on a hybrid 2-node job every rank correctly sees 8 mapped peers and 8 NULLs. Measured on
    venue B a recorded job: rank 0 got NULLs for ranks 8-15, rank 11 got NULLs for ranks 0-7, each the
    exact complement of its own node. That NULL-ness IS the venue probe the drain selection uses to
    tell NVLink from IB, so what this asserts is the direction that must hold: **every node-mate is
    directly addressable**. A cross-node peer may be either, and is reported rather than judged.
    """
    import torch.distributed._symmetric_memory as symm_mem

    t = symm_mem.empty(world_size, dtype=torch.int32, device=device)
    handle = symm_mem.rendezvous(t, torch.distributed.group.WORLD)
    peers = list(handle.buffer_ptrs)
    # Drop the symmetric allocation HERE, before the first assert, not at the end of the function.
    # `peers` is already a plain list of ints, so nothing below needs `handle` or `t`. The point is
    # the assert path: every assertion below is computed from THIS rank's peer map, so one rank can
    # fail while its peers pass, and a rank that raises leaves the frame without running a trailing
    # `del` -- moving the drop out of the exception's way is what keeps it at a program point every
    # rank reaches in the same order. Deliberately NOT converted to the pool idiom: `symm_mem.empty`
    # + `rendezvous` + `handle.buffer_ptrs` IS this test's subject, and a MemPool hands back no
    # handle to read `buffer_ptrs` from.
    del handle, t
    assert len(peers) == world_size, f"got {len(peers)} peer pointers, want {world_size}"

    per_node = topology["local_world_size"]
    mates = [p for p in range(world_size) if p // per_node == rank // per_node]
    unmapped_mates = [p for p in mates if peers[p] == 0]
    assert not unmapped_mates, (
        f"rank {rank}: node-mates {unmapped_mates} have NULL peer pointers, but a peer in the same "
        f"NVLink domain must be directly addressable. peers={[hex(p) for p in peers]}"
    )
    # Cross-node peers: NULL is expected (IB). Assert only that the split is coherent -- a mapped
    # cross-node peer is fine, a job that claims 2 nodes but maps everything means the ranks all
    # landed on ONE node and the run is not testing what it says it is.
    if topology["n_nodes"] > 1:
        assert any(peers[p] == 0 for p in range(world_size) if p not in mates), (
            f"rank {rank}: n_nodes={topology['n_nodes']} but every peer is directly addressable, so "
            f"all ranks are really on one node -- this run is not exercising the IB path"
        )


# --------------------------------------------------------------------------- #
# Symmetric-memory MemPool ownership.
#
# WHY THESE ARE NOT "does the property return an object" TESTS.
# The pool prevents a MEASURED deadlock, and it does so through a refcount property, not an API
# surface. `nvshmem_free` is COLLECTIVE; torch's `symm_mem.empty()` attaches a `from_blob` deleter
# that calls it unconditionally when the last reference dies (`SymmetricMemory.cpp`), so the moment
# of a collective is decided by each rank's interpreter. Two ranks decide differently, one enters
# the collective alone, and the job hangs -- observed with one rank in
# `nvshmem_free -> nvshmemi_barrier` while its peer sat in a teardown barrier.
#
# Routing through a held MemPool removes that: torch frees a pool's segments only once the pool
# reaches `use_count == 0` and enters `graph_pools_freeable`, so while DM holds the handle even the
# allocator's OOM-retry path cannot release them. Each test below pins one link of that chain, and
# each would go green against a property that merely existed -- which is why the assertions are
# about IDENTITY, USE COUNT, ADDRESS REUSE and SURVIVING `cleanup()` rather than about the type.
# --------------------------------------------------------------------------- #
@DIST_MESH.parametrize("mesh")
@numeric_exempt("asserts POOL OBJECT IDENTITY; allocates no tensor and computes no value")
def test_symmetric_mempool_is_one_handle_per_process(apply_mesh, mesh, device):
    """Every caller gets the SAME pool object, and the Borg instances agree on it -- under any mesh.

    Identity is the invariant, not "a pool exists": two pools for one device would mean two
    allocation states, and a rank whose peers used the other one is exactly the desync the pool is
    meant to prevent.

    Swept over the mesh because ``apply_mesh`` RESETS AND REBUILDS the process groups on every call.
    The pool must survive that: it is keyed on the device and owned by the process, so a regrouping
    must not silently mint a second one. A pool whose lifetime accidentally tracked the process
    GROUP would pass a single-mesh test and fail here.
    """
    apply_mesh(mesh)
    a = DistributedManager.symmetric_mempool(device)
    b = DistributedManager.symmetric_mempool(device)
    c = DistributedManager().symmetric_mempool(device)
    assert a is b is c, "symmetric_mempool must return one shared object per device"
    # Index and torch.device spellings must resolve to the same pool, or two call sites that
    # disagree only in spelling would silently hold different pools.
    assert DistributedManager.symmetric_mempool(device.index) is a


@DIST_MESH.parametrize("mesh")
@numeric_exempt("asserts an allocator REFCOUNT; allocates no tensor and computes no value")
def test_symmetric_mempool_use_count_stays_positive(apply_mesh, mesh, device):
    """``use_count`` never reaches 0 -- the condition torch requires before it may free segments.

    This is the load-bearing assertion. `release_cached_blocks` frees a pool's blocks only for pools
    in `graph_pools_freeable`, and `releasePool` inserts a pool there only when `--use_count == 0`.
    A positive count is therefore not a statistic; it is the proof that no implicit `nvshmem_free`
    can fire.

    Swept over the mesh: ``apply_mesh`` rebuilds the process groups, and the count must not be
    disturbed by that. A regrouping that incidentally dropped a reference would arm the free.
    """
    apply_mesh(mesh)
    pool = DistributedManager.symmetric_mempool(device)
    assert pool.use_count() >= 1, "a pool at use_count 0 is eligible for release -> implicit free"
    # An enter/exit of the context manager must return to a POSITIVE count, not to zero. `__exit__`
    # calls `releasePool`, and it is only safe because entry increments first.
    with torch.cuda.use_mem_pool(pool):
        inside = pool.use_count()
    assert inside >= 2, f"entering use_mem_pool should incref, got {inside}"
    assert pool.use_count() >= 1, "exiting use_mem_pool must not drop the count to 0"


@DIST_MESH.parametrize("mesh")
@numeric_exempt("asserts allocator address reuse, not a computed tensor")
def test_pooled_allocation_is_recycled_not_freed(apply_mesh, mesh, device):
    """A same-size allocation after the previous one dies must land on the SAME address.

    Address reuse is the observable that distinguishes "returned to the pool" from "freed and
    re-allocated". If the segment had been freed, `nvshmem_free` ran -- a collective, at a moment
    this rank chose alone -- and a fresh address would come back. Measured on the unpooled path:
    60 allocations produced 60 distinct addresses.

    COLLECTIVE-SAFE: every rank runs the identical sequence, so any collective underneath (a pool
    miss growing the pool) is reached by all ranks together.

    The mesh sweep is the point of interest here rather than a formality: a cross-node spec puts the
    symmetric group across InfiniBand, where a collective free is far more expensive to get wrong,
    and recycling must hold identically there.
    """
    import gc

    apply_mesh(mesh)
    pool = DistributedManager.symmetric_mempool(device)
    with torch.cuda.use_mem_pool(pool):
        x = torch.empty(1024, dtype=torch.float32, device=device)
    first = x.data_ptr()
    del x
    gc.collect()
    with torch.cuda.use_mem_pool(pool):
        y = torch.empty(1024, dtype=torch.float32, device=device)
    second = y.data_ptr()
    del y
    gc.collect()
    assert first == second, (
        f"pooled same-size realloc got {hex(second)} after {hex(first)}; a fresh address means the "
        "segment was FREED (collective nvshmem_free) rather than recycled"
    )


@DIST_MESH.parametrize("mesh")
@numeric_exempt("asserts device free memory, not a computed tensor")
def test_pooled_churn_does_not_grow_device_memory(apply_mesh, mesh, device):
    """Churning many same-size allocations through the pool must not consume memory per iteration.

    The complement of the address test: reuse could in principle coexist with a leak. Holding free
    memory flat over N allocations shows one segment is being recycled, so N-1 collective frees did
    NOT happen. Bounded deliberately -- a few iterations prove recycling; a long loop would only
    prove it more slowly.
    """
    import gc

    apply_mesh(mesh)
    pool = DistributedManager.symmetric_mempool(device)
    n_elem, iters = 4 * 1024 * 1024, 8  # 16 MiB x 8 = 128 MiB if nothing is recycled
    gc.collect()
    before = torch.cuda.mem_get_info(device.index)[0]
    for _ in range(iters):
        with torch.cuda.use_mem_pool(pool):
            t = torch.empty(n_elem, dtype=torch.float32, device=device)
        del t
        gc.collect()
    after = torch.cuda.mem_get_info(device.index)[0]
    grew_mib = (before - after) // (1024 * 1024)
    assert grew_mib <= 32, (
        f"{iters} pooled 16 MiB allocations consumed {grew_mib} MiB; recycling should cost about "
        "one segment, so this indicates each iteration took a fresh one"
    )


@DIST_MESH.parametrize("mesh")
@numeric_exempt("asserts pool-handle OWNERSHIP and release wiring; no tensor involved")
def test_the_pool_is_released_at_a_controlled_point_not_at_exit(apply_mesh, mesh, device):
    """The pool must be released DELIBERATELY, before nvshmem finalizes -- not left to shutdown.

    This replaces a test that asserted the OPPOSITE, and the correction is the point of it. Holding
    the pool forever does not avoid the collective free; it defers it to interpreter shutdown, which
    runs AFTER ``_finalize_nvshmem``. Measured before the fix: every rank exited 255 with ``NVSHMEM
    API called before NVSHMEM initialization has completed``. The invariant was never "never
    release" -- it is "release at exactly ONE point every rank reaches together".

    Asserts the WIRING rather than calling the release, which would free the pool out from under the
    rest of the session. What must hold: the release exists, the exit path calls it BEFORE the
    finalize, and the handle lives outside ``_state`` so no ``cleanup()`` drops it at a moment the
    ranks have not agreed on.
    """
    import inspect

    from fold_cp_ops.distributed import distributed_manager as dm_mod

    apply_mesh(mesh)
    pool = DistributedManager.symmetric_mempool(device)
    assert pool in dm_mod._SYMMETRIC_MEMPOOLS.values(), (
        "the pool must be held in the module-level registry, not in per-manager state"
    )
    assert not any(v is pool for v in DistributedManager._state.values()), (
        "the pool must NOT live in _state: cleanup() wipes it, which would drop the handle at a "
        "moment the ranks have not agreed on"
    )
    # Compare the CALL EXPRESSIONS, not bare names. Both names also occur in that function's
    # comments -- which is how an earlier version of this assertion failed on correct code: the
    # first `_finalize_nvshmem` hit was inside the comment explaining this very ordering.
    src = inspect.getsource(dm_mod.DistributedManager._drain_and_finalize_at_exit)
    release_call = "DistributedManager.release_symmetric_mempools()"
    finalize_call = "DistributedManager._finalize_nvshmem()"
    assert release_call in src, (
        "the exit path must release the pools explicitly; leaving it to interpreter shutdown puts "
        "the collective free after _finalize_nvshmem and every rank exits 255"
    )
    assert finalize_call in src, "expected _drain_and_finalize_at_exit to call _finalize_nvshmem()"
    assert src.index(release_call) < src.index(finalize_call), (
        "the pool release must come BEFORE _finalize_nvshmem -- freeing a symmetric segment after "
        "the library is down raises 'NVSHMEM API called before NVSHMEM initialization has completed'"
    )


@matrix_exempt(
    "reads the SOURCE of init_nvshmem; the allocator a statement is written against is a property "
    "of the file, not of the device mesh, so sweeping DIST_MESH would re-parse one function 15 times"
)
@numeric_exempt("parses a function's AST; allocates no tensor and computes no value")
def test_the_lazy_uid_bootstrap_probe_is_allocated_through_the_pool():
    """``init_nvshmem``'s one-byte bootstrap probe must come from the pool, not a bare ``empty``.

    Why this is asserted at all, and why here
        Every other symmetric allocation this package makes is under
        :meth:`DistributedManager.symmetric_mempool`, so its death recycles a segment instead of
        firing the COLLECTIVE ``nvshmem_free`` at whatever moment one rank's interpreter picks. The
        bootstrap probe is the ONE allocation that ran before that rule existed, and it is on the
        path EVERY rank takes FIRST -- the worst possible place for a rank-chosen collective.

    Why a bare ``symm_mem.empty`` is not already good enough
        It very nearly is, which is the trap. torch's ``symm_mem.empty`` routes through
        ``get_mem_pool`` -- the same pool object -- but only while ``_should_use_implicit_mempool()``
        holds, and that reads ``TORCH_SYMMMEM_IMPLICIT_POOL``, defaulting to ``"1"`` (torch v2.11.0,
        ``_symmetric_memory/__init__.py``). An operator exporting ``0`` moves the first allocation of
        every rank back onto the unpooled path, where the ``from_blob`` deleter frees collectively on
        the last reference drop. Nothing would report that; the job would simply hang one day. So the
        property worth pinning is that we do NOT depend on that env var.

    Why the AST and not a substring
        The sibling release-ordering test above searches source TEXT and carries a comment about how
        an earlier version of it matched its own explanatory comment instead of the call. Parsing
        removes that whole failure mode: a mention of ``symm_mem.empty`` in a comment is not a
        statement, and only statements are examined here.

    Raises:
        AssertionError: if the probe assignment is not lexically inside a
            ``torch.cuda.use_mem_pool(...)`` block, if that block's pool does not come from
            ``symmetric_mempool``, or if a bare ``symm_mem.empty(...)`` call survives anywhere in
            ``init_nvshmem``.
    """
    import ast
    import inspect
    import textwrap

    from fold_cp_ops.distributed import distributed_manager as dm_mod

    fn = ast.parse(textwrap.dedent(inspect.getsource(dm_mod.DistributedManager.init_nvshmem))).body[
        0
    ]

    def calls(node):
        return [ast.unparse(n.func) for n in ast.walk(node) if isinstance(n, ast.Call)]

    # No STATEMENT may call symm_mem.empty -- rendezvous/set_backend/get_backend are fine and stay.
    assert "symm_mem.empty" not in calls(fn), (
        "init_nvshmem() allocates its bootstrap probe with a bare symm_mem.empty(). That lands in "
        "the pool only while TORCH_SYMMMEM_IMPLICIT_POOL != '0', so an operator can silently put "
        "the first allocation every rank makes back on the path whose from_blob deleter calls the "
        "COLLECTIVE nvshmem_free at a moment this rank alone decides. Allocate it inside "
        "`with torch.cuda.use_mem_pool(DistributedManager.symmetric_mempool(device)):` instead."
    )

    pooled = [
        w
        for w in ast.walk(fn)
        if isinstance(w, ast.With)
        and any(
            isinstance(i.context_expr, ast.Call)
            and ast.unparse(i.context_expr.func) == "torch.cuda.use_mem_pool"
            for i in w.items
        )
    ]
    assert pooled, (
        "init_nvshmem() has no torch.cuda.use_mem_pool block, so nothing it allocates is pooled"
    )
    probe_blocks = [
        w
        for w in pooled
        if any(
            isinstance(s, ast.Assign) and any(getattr(t, "id", None) == "probe" for t in s.targets)
            for s in w.body
        )
    ]
    assert probe_blocks, (
        "init_nvshmem() assigns `probe` outside every use_mem_pool block. The bootstrap allocation "
        "is what fixes the backend and triggers torch's lazy UID handshake; allocating it off the "
        "pool strands a symmetric segment that release_symmetric_mempools() cannot reach."
    )
    # The pool must be OURS -- symm_mem.get_mem_pool() returns the same object but takes no permanent
    # reference through DM, and the held reference is what keeps use_count above zero.
    for w in probe_blocks:
        for i in w.items:
            if ast.unparse(i.context_expr.func) != "torch.cuda.use_mem_pool":
                continue
            inner = calls(i.context_expr)
            assert any("symmetric_mempool" in c for c in inner), (
                f"the probe's pool comes from {inner!r}, not DistributedManager.symmetric_mempool. "
                "Only the DM accessor stores the handle in the module-level registry, and that "
                "held reference is what keeps use_count >= 1 and the collective free disarmed."
            )


# --------------------------------------------------------------------------- #
# The one suppressed nvshmem warning -- a DRIFT DETECTOR, not a re-run of the suppression.
# --------------------------------------------------------------------------- #
@matrix_exempt(
    "the subject is a library-level log string and nvshmem4py's own module state; neither varies "
    "with the device mesh, so sweeping DIST_MESH would re-assert one process-wide fact 15 times"
)
@numeric_exempt("asserts a log string and a tracker's emptiness; no tensor involved")
def test_the_suppressed_nvshmem_warning_still_matches_upstream(dist_manager):
    """The one message ``_finalize_nvshmem`` filters must still be EXACTLY what nvshmem4py emits.

    Why this test exists
        ``_finalize_nvshmem`` silences exactly one warning -- "NVSHMEM Library is not initialized.
        Cannot free buffers" -- because it is unconditionally false under torch-owned init and it
        has twice been misread as evidence (once as deferred reclaim, once as a status flag that
        reported failure on runs that exited 0). Silencing a message creates two new ways to be
        wrong, and this test covers both.

    What it would catch
        1. **Upstream re-words the message.** Our filter is an EXACT string match, so a wording
           change makes it stop matching -- the warning returns and spooks the next reader, with no
           indication that a filter was ever involved. Asserting the literal against nvshmem4py's
           own source turns that into a named failure here.
        2. **The suppression starts hiding something real.** ``_free_all_buffers`` is a LEAK SWEEPER
           over ``_mr_references``. Skipping it is free today only because that tracker is EMPTY --
           torch owns every allocation. The moment anything allocates via nvshmem4py (e.g.
           ``nvshmem_cute.tensor``), the skipped sweep is a skipped leak check and this exact
           warning becomes its only symptom. A non-empty tracker therefore means: un-suppress and
           re-examine.

    Deliberately does NOT call ``_finalize_nvshmem``. That tears the library down for the whole
    process; a test that invoked it would strand every later test in the session.

    Args:
        dist_manager: session manager, for ordering only -- this asserts process-wide library state
            rather than anything rank-specific, so it needs no mesh and no collective.

    Raises:
        AssertionError: if the filtered literal no longer appears in either our source or
            nvshmem4py's, or if nvshmem4py has started tracking allocations of its own.
    """
    import inspect

    memory = pytest.importorskip("nvshmem.core.memory")
    tracking = pytest.importorskip("nvshmem.core._internal_tracking")

    from fold_cp_ops.distributed import distributed_manager as dm_mod

    expected = "NVSHMEM Library is not initialized. Cannot free buffers"

    ours = inspect.getsource(dm_mod.DistributedManager._finalize_nvshmem)
    assert expected in ours, (
        "_finalize_nvshmem no longer filters the expected literal; if the suppression was removed "
        "on purpose, delete this test with it"
    )
    upstream = inspect.getsource(memory._free_all_buffers)
    assert expected in upstream, (
        f"nvshmem4py no longer emits {expected!r} -- our EXACT-match filter in _finalize_nvshmem is "
        "now dead code, so the warning (or its replacement) will surface unexplained. Re-read "
        "nvshmem/core/memory.py::_free_all_buffers and re-sync the literal or drop the filter."
    )
    assert not tracking._mr_references, (
        f"nvshmem4py is now tracking {len(tracking._mr_references)} allocation(s), so "
        "_free_all_buffers() is no longer a vacuous sweep. Our filter hides the ONLY symptom of it "
        "being skipped -- UN-SUPPRESS it in _finalize_nvshmem and re-examine the teardown."
    )


@matrix_exempt(
    "argument validation: the refusal happens on the device argument before any group, mesh or "
    "allocation is touched, so it cannot vary with the mesh. The other five mempool tests ARE swept "
    "over DIST_MESH because they reach the allocator, where a cross-node group genuinely differs"
)
def test_symmetric_mempool_refuses_a_non_cuda_device(dist_manager):
    """A CPU device must RAISE, not return a pool that cannot back a peer-addressable tensor."""
    with pytest.raises(RuntimeError, match="requires a CUDA device"):
        DistributedManager.symmetric_mempool(torch.device("cpu"))


"""Repo-native (torchrun / srun-per-rank COLLECTED) pytest for the nvshmem device-state registration
co-residency cap (task #44). See ``docs/nvshmem4py_state_duplication_limit.md`` for the mechanism.

SELF-CONTAINED: the repro worker (the trivial non-GEMM nvshmem-device kernel + the build/library_init loop)
lives in THIS file as module-level fns and as an ``if __name__ == "__main__"`` runnable repro -- there is no
separate standalone script to keep in sync.

DESIGN (like the other ``tests/distributed`` tests): the test body runs IN-PROCESS per-rank under the
conftest ``dist_manager`` + a session-scoped ``_nvshmem`` fixture (``DistributedManager.init_nvshmem()``);
each rank builds K DISTINCT nvshmem-device-linked CUlibraries and ``library_init``-registers each (the
module-load path ``nvshmemx_culibrary_init -> nvshmemi_update_device_state -> cuMemcpyHtoD``). Consensus via
``all_reduce(MIN)`` on the last-successful build index across ranks.

The cap is IBGDA/cross-node-specific (confirmed on venue B sm90: single-node NVLink builds all K; 2-node
IBGDA SIGSEGVs at ~the 3rd co-resident ``library_init``; ``library_finalize``-each clears it). A cross-node
SIGSEGV is a hard C-level crash -- it CANNOT be asserted in-process (it would kill the rank). Worse, the
ACCUMULATE probe cannot cleanly coexist with anything else in one process (findings 2+3 below), so the
collected PYTEST is the ONE robust cell:
  * ``test_registration_finalize_no_cap``: finalize-each keeps the live registered set <=1 -> builds all K
                                         with NO cap on EITHER single- OR cross-node -> assert min_ok == K ->
                                         PASS everywhere. Pins the ``library_finalize`` workaround the
                                         autotuner relies on; if nvshmem ever breaks that path it FAILS.
The single-node-no-cap-vs-cross-node-cap CONTRAST is the ``__main__`` accumulate repro, NOT a pytest.

THREE nvshmem findings this file exercises (all on the ``update_device_state`` device-state machinery):
  1. INIT cap: accumulate (retain) -> the ~3rd co-resident ``library_init`` SIGSEGVs cross-node/IBGDA
     (``nvshmemx_culibrary_init``); single-node NVLink has no such cap (builds all K).
  2. batch-UNREGISTER wall: per-module ``library_finalize`` of a CO-RESIDENT batch HANGS/SIGSEGVs
     (``culibrary_finalize``, init_fini.py:453) -- a ``dist.barrier`` cannot fix it.
  3. finalize-CHURN pollution: a run of ``library_init``+``library_finalize`` pairs corrupts the device-state
     refcount machinery so a LATER ``library_init`` in the SAME process SIGSEGVs.
(2)+(3) are why the accumulate/no-cap check can't be a collected pytest -- clean only as the FIRST and ONLY
nvshmem-registration activity in a process (the ``__main__`` repro).

The single-node/NVLink LAUNCH must use a CLEAN env (no ``NVSHMEM_HCA_LIST`` / ``NVSHMEM_ENABLE_NIC_PE_MAPPING``,
``NVSHMEM_REMOTE_TRANSPORT=none``, IBGDA off); the cross-node LAUNCH sets IBGDA + the active ``NVSHMEM_HCA_LIST``.
That is a LAUNCHER concern (the per-launch sbatch env) -- do NOT leak the 2-node IB env into the single-node run.

Runnable repro (replaces the standalone): ``python tests/distributed/test_nvshmem_registration_cap.py
--mode {accumulate|finalize} --K 16`` under srun-per-rank / torchrun -- accumulate cross-node reproduces the cap.
"""


@matrix_exempt(
    "the subject is the symmetric-heap LIFECYCLE across a failed allocation -- there is no kernel, "
    "shape or dtype axis to sweep. The OOM size is derived from live free memory, not declared, "
    "because a fixed size would stop reproducing on a box with a different card"
)
@numeric_exempt("asserts that a collective still works after an OOM, not a computed value")
def test_a_symmetric_OOM_does_not_poison_the_NEXT_allocation(dist_manager):
    """H1: an allocation that OOMs must leave the symmetric heap usable for the next one.

    THE REPRODUCER for a defect that has cost this session more than any other. Observed shape, from
    the progress file of a real gate run::

        cp8   N4096 D512 outgoing   call passed
        cp8   N4096 D512 incoming   call SKIPPED     <- OOM skip
        cp2x4 N2048 D128 outgoing   setup passed, then IMA -> context dead, session over

    The cell AFTER an OOM-skipped one takes an illegal memory access, and a poisoned CUDA context is
    sticky, so the whole launch is lost. The same shape produced a 52-failure/192-error cascade, and
    killed a world-4 harvest and a verification run.

    Mechanism under test: `nvshmem_malloc` is COLLECTIVE, so an allocation that fails on one rank and
    succeeds on another leaves the symmetric heap ASYMMETRIC -- different ranks holding different
    offsets. `gated_skip` correctly all-reduces the SKIP decision (which is what stops the ranks
    deadlocking), but nothing re-symmetrises the heap, so the next `rendezvous` maps addresses that
    are valid on some ranks and not others.

    Why this is an infrastructure bug and not a test wart: a session is one process, and a cell that
    fails must leave the process as it found it. The same leak is reachable from the shipped API --
    any caller who OOMs one `TriangularMultiplication` and builds another in the same process
    inherits it. The
    per-cell isolation rule is currently covering for it, which makes isolation load-bearing for
    CORRECTNESS rather than for blast radius.

    What is asserted: after a deliberate symmetric OOM, a SMALL symmetric allocation rendezvouses and
    a collective over it completes. Not that the OOM is avoided -- OOM is a legitimate outcome and the
    repo's rule is to skip on it -- only that it does not leak.
    """
    dm = dist_manager
    dm.init_nvshmem()
    free, _total = torch.cuda.mem_get_info()
    # Sized to exceed what is free, so the allocation FAILS -- the point is the failure path, and a
    # size derived from the live figure fails on any box rather than only on an 80 GB one.
    doomed = int(free * 4)
    # Through the PRODUCTION seam, not a raw `torch.empty` in the pool: the fix is a precheck in
    # `_symmetric_empty`, and a test that bypasses it would keep passing while every real caller
    # aborts. This is the path `TriangularMultiplication` allocates its recv buffers on.
    from fold_cp_ops.distributed.workflows.trimul_autotuned import _symmetric_empty

    try:
        _symmetric_empty((doomed,), dtype=torch.uint8, device=torch.cuda.current_device())
        pytest.fail(f"expected a symmetric OOM for {doomed} bytes with {free} free")
    except torch.cuda.OutOfMemoryError:
        pass
    except RuntimeError as e:
        # A SYMMETRIC over-allocation does NOT raise `torch.cuda.OutOfMemoryError`. Measured, at
        # world 2: `RuntimeError: nvshmem_malloc failed`, with `NVSHMEMX_ERROR_INVALID_VALUE  Not
        # enough space for allocating memory` on stderr. Neither string contains "out of memory",
        # which is why the repo's OOM seams normalise it explicitly -- and why a handler that only
        # catches `OutOfMemoryError` never sees a symmetric OOM at all.
        if not any(
            k in str(e).lower() for k in ("out of memory", "nvshmem_malloc", "symmetric heap")
        ):
            raise
    # Every rank reached the same failure, so this is the moment `gated_skip` would agree to skip.
    dist.barrier()
    # The claim: the heap is still usable. A small allocation, a rendezvous, and a collective on it.
    import torch.distributed._symmetric_memory as symm_mem

    with torch.cuda.use_mem_pool(dm.symmetric_mempool()):
        probe = torch.zeros(1024, dtype=torch.float32, device="cuda")
    handle = symm_mem.rendezvous(probe, dist.group.WORLD)
    assert handle is not None, "rendezvous returned no handle after an OOM on the same heap"
    probe.fill_(float(dm.rank) + 1.0)
    dist.all_reduce(probe)
    torch.cuda.synchronize()
    expect = float(sum(r + 1 for r in range(dm.world_size)))
    assert_elementwise(
        probe,
        torch.full_like(probe, expect),
        tolerance_bound(torch.full_like(probe, expect)),
        what="all_reduce over a symmetric buffer allocated AFTER an OOM",
    )


import argparse  # noqa: E402
import os  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402

import pytest  # noqa: E402
import torch  # noqa: E402,F811

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


import cutlass  # noqa: E402
import cutlass.cute as cute  # noqa: E402
import cutlass.torch as cutlass_torch  # noqa: E402
import cuda.core  # noqa: E402
import nvshmem.core  # noqa: E402
import nvshmem.core.device.cute.direct as nvshmem_cute_direct  # noqa: E402

_K = int(
    os.environ.get("REGCAP_K", "16")
)  # >8 so the cap (if present) would trigger; keeps runs short


# --------------------------------------------------------------------------- #
# nvshmem bootstrap (session-scoped, collective) -- mirrors test_a2a_fusion._nvshmem.
# --------------------------------------------------------------------------- #
@pytest.fixture(autouse=False)
def _nvshmem(dist_manager):
    """``DistributedManager.init_nvshmem()`` once per session (collective: uid broadcast + barrier). The
    conftest ``dist_manager`` brings up torch.distributed but NOT nvshmem; the registration probe needs it."""
    from fold_cp_ops.distributed.distributed_manager import (
        DistributedManager,
    )  # lazy: keeps the __main__ repro fold_cp_ops-free

    DistributedManager.init_nvshmem()
    yield


# --------------------------------------------------------------------------- #
# Repro worker (module-level; folded from the former standalone docs/nvshmem4py_reg_repro.py).
# --------------------------------------------------------------------------- #
class _MinSignalKernel:
    """Trivial nvshmem device kernel: ONE lane issues one signal_op. ``tag`` is a compile-time constant baked
    into the IR so each distinct ``tag`` yields a DISTINCT compiled module / CUlibrary (mimics an autotuner
    with many distinct live kernels). NO MMA atom -> compiles wherever the nvshmem device bitcode links."""

    def __init__(self, tag: int, add_op: int):
        self.tag = tag
        self.add_op = add_op

    @cute.jit
    def __call__(self, mSignal: cute.Tensor, mPeTable: cute.Tensor, stream):
        self.kernel(mSignal, mPeTable).launch(
            grid=(1, 1, 1), block=(32, 1, 1), cluster=(1, 1, 1), stream=stream
        )

    @cute.kernel
    def kernel(self, mSignal: cute.Tensor, mPeTable: cute.Tensor):
        tag: cutlass.Constexpr[int] = self.tag
        if cute.arch.warp_idx() == cutlass.Int32(0):
            if cute.arch.lane_idx() == cutlass.Int32(0):
                cute.arch.fence_proxy("async.global")
                cute.arch.fence_acq_rel_sys()
                dst_pe = mPeTable[0]
                sig_slot = cute.local_tile(mSignal, (1,), (cutlass.Int32(0),))
                nvshmem_cute_direct.signal_op(
                    sig_slot, cutlass.Int64(tag + 1), cutlass.Int32(self.add_op), dst_pe
                )


def _resolve_bitcode(rank):
    """Default lookup (current arch); fall back to explicit archs if the env lacks the current-arch .bc."""
    try:
        return nvshmem.core.find_device_bitcode_library(), "default"
    except Exception as e_def:
        if rank == 0:
            print(
                f"[repro] default bitcode lookup failed: {e_def!r}; trying explicit archs",
                flush=True,
            )
        for a in ("90", "100", "89", "80"):
            try:
                return nvshmem.core.find_device_bitcode_library(arch=a), a
            except Exception:
                continue
    raise RuntimeError("no nvshmem device bitcode found (default or arch {90,100,89,80})")


def _build_one(tag, bitcode, stream, add_op, dev_id):
    """cute.compile(--link-libraries) -> .to(dev) -> library_init one distinct trivial nvshmem module."""
    op = _MinSignalKernel(tag, add_op)
    f_sig = cute.runtime.make_fake_tensor(cutlass.Int64, (1,), stride=(1,), assumed_align=8)
    f_pe = cute.runtime.make_fake_tensor(cutlass.Int32, (1,), stride=(1,), assumed_align=4)
    compiled = cute.compile(op, f_sig, f_pe, stream, options=f" --link-libraries={bitcode}")
    exec_fn = compiled.to(dev_id)
    jm = getattr(exec_fn, "jit_module", None)
    if jm is None or not getattr(jm, "cuda_library", None):
        raise RuntimeError(f"module tag={tag} has no cuda_library handle (jit_module={jm!r})")
    handle = int(jm.cuda_library[0])
    nv_obj = nvshmem.core.NvshmemKernelObject.from_handle(handle)
    nvshmem.core.library_init(nv_obj)  # <-- culibrary_init -> update_device_state -> cuMemcpyHtoD
    return nv_obj, handle


def _run_loop(mode, K, rank, world, local_rank):
    """Build K distinct nvshmem modules + library_init each; accumulate (retain -> cap) vs finalize
    (library_finalize each -> no cap). Returns the cross-rank min last-successful build count (== K iff
    every rank built all K). Finalizes retained modules before returning so the shared session nvshmem is
    NOT left with an accumulated registered set (this is in-process; do NOT finalize the session itself)."""
    bitcode, bc_arch = _resolve_bitcode(rank)
    if rank == 0:
        print(f"[repro] mode={mode} K={K} world={world} bitcode_arch={bc_arch}", flush=True)
    stream = cutlass_torch.current_stream()
    add_op = int(nvshmem.core.SignalOp.SIGNAL_ADD)
    dev_id = cuda.core.Device().device_id

    retained = []
    faulted_at = None
    t_all0 = time.time()
    for k in range(K):
        t0 = time.time()
        try:
            nv_obj, handle = _build_one(k, bitcode, stream, add_op, dev_id)
        except BaseException as e:  # noqa: BLE001 — surface any fault (incl. SIGSEGV-adjacent) + frame
            faulted_at = k
            print(
                f"[repro][rank{rank}] *** FAULT at build k={k} (dt={time.time() - t0:.2f}s) after {k} "
                f"successful builds ***",
                flush=True,
            )
            print(
                f"[repro][rank{rank}] error type: {type(e).__module__}.{type(e).__name__}",
                flush=True,
            )
            print(f"[repro][rank{rank}] error str : {e!r}", flush=True)
            traceback.print_exc()
            break
        dt = time.time() - t0
        if mode == "finalize":
            nvshmem.core.library_finalize(nv_obj)
        else:
            retained.append(nv_obj)
        if rank == 0:
            print(
                f"built {k}  wall={dt:.2f}s  handle={handle}  retained={len(retained)}", flush=True
            )

    local_ok = K if faulted_at is None else faulted_at
    t = torch.tensor([local_ok], device=f"cuda:{local_rank}")
    torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.MIN)
    min_ok = int(t.item())
    if rank == 0:
        if faulted_at is None:
            print(
                f"[repro] DONE mode={mode}: built ALL {K} modules, NO fault "
                f"(min_ok_across_ranks={min_ok}, total {time.time() - t_all0:.1f}s)",
                flush=True,
            )
        else:
            print(
                f"[repro] STOPPED mode={mode}: rank0 faulted at k={faulted_at} "
                f"(min successful build across ranks={min_ok})",
                flush=True,
            )

    # Cleanup: FINALIZE mode already library_finalize'd each module in-loop (``retained`` empty). ACCUMULATE
    # mode's ``retained`` modules are DELIBERATELY co-resident; do NOT batch per-module library_finalize them
    # -- nvshmem's ``culibrary_finalize`` (init_fini.py:453, the C ``nvshmemx_culibrary_finalize``) HANGS /
    # SIGSEGVs tearing down a CO-RESIDENT batch: a second, related ``update_device_state`` wall on the
    # UNREGISTER path (a C-level hang a ``dist.barrier`` cannot fix). They are process-local device state,
    # reclaimed cleanly by the session's GLOBAL ``nvshmem.core.finalize()`` (``hostlib_finalize``, NOT
    # per-module) + process exit -- so LEAVING them is non-hanging. Only the ``__main__`` accumulate repro
    # exercises this path (the accumulate/no-cap check is not a collected pytest -- see the module docstring).
    if mode == "accumulate" and retained and rank == 0:
        print(
            f"[repro] accumulate: leaving {len(retained)} co-resident modules for GLOBAL teardown "
            f"(per-module finalize of a co-resident batch hangs — secondary unregister wall)",
            flush=True,
        )
    try:
        torch.distributed.barrier()
    except BaseException:
        pass
    return min_ok


# --------------------------------------------------------------------------- #
# The ONE robust collected cell. The accumulate/no-cap probe is NOT a pytest here: it can't cleanly coexist
# with anything else in a process (finalize-churn pollution + co-resident-batch teardown hang -- the module
# docstring's findings 2+3), so it lives ONLY in the __main__ repro. finalize-each keeps the live registered
# set <=1, so it is cap-free and safe to collect on both single- and cross-node.
# --------------------------------------------------------------------------- #
@matrix_exempt(
    "asserts a manager property that does not vary with the device mesh -- rank/world identity, singleton state, nvshmem bring-up and the symmetric-memory backend are all one-per-process facts. Sweeping DIST_MESH here would rebuild the grid 15 times to re-assert the same value"
)
@pytest.mark.usefixtures("_nvshmem")
@pytest.mark.usefixtures("_module_skip__distributed__nvshmem_registration_cap")
def test_registration_finalize_no_cap(dist_manager, world_size):
    """library_finalize-each is the workaround: finalizing each module before building the next keeps the live
    registered set small (<=1), so ALL K build on BOTH single- and cross-node. Confirms the cap is co-resident
    ACCUMULATION, not a per-build defect (and pins the workaround the autotuner relies on)."""
    min_ok = _run_loop("finalize", _K, dist_manager.rank, world_size, dist_manager.local_rank)
    assert min_ok == _K, (
        f"finalize-each must build ALL {_K} (no cap) single- AND cross-node; min_ok={min_ok}"
    )


# --------------------------------------------------------------------------- #
# Runnable repro (replaces the standalone). Full bootstrap (torch.dist + nvshmem UID) since it is NOT under
# the pytest fixtures. accumulate cross-node reproduces the cap (SIGSEGV at ~the 3rd module).
# --------------------------------------------------------------------------- #
def _envint(name, *alts, default=None):
    for k in (name, *alts):
        v = os.environ.get(k)
        if v is not None and v != "":
            return int(v)
    if default is not None:
        return default
    raise KeyError(f"none of {(name, *alts)} set in env")


def _bootstrap_standalone():
    """torch.distributed env:// + nvshmem UID init (srun-per-rank or torchrun) for the __main__ repro."""
    import numpy as np
    import torch.distributed as dist

    rank = _envint("RANK", "SLURM_PROCID")
    world = _envint("WORLD_SIZE", "SLURM_NTASKS", "SLURM_NPROCS")
    local_rank = _envint("LOCAL_RANK", "SLURM_LOCALID", default=0)
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl", rank=rank, world_size=world)
    dev = cuda.core.Device(local_rank)
    dev.set_current()
    uid = nvshmem.core.get_unique_id(empty=(rank != 0))
    uid_bytes = uid._data.view(np.uint8).copy()
    uid_tensor = torch.from_numpy(uid_bytes).to(device)
    dist.broadcast(uid_tensor, src=0)
    dist.barrier()
    uid._data[:] = uid_tensor.cpu().numpy().view(uid._data.dtype)
    nvshmem.core.init(device=dev, uid=uid, rank=rank, nranks=world, initializer_method="uid")
    return rank, world, local_rank


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["accumulate", "finalize"], default="accumulate")
    ap.add_argument("--K", type=int, default=_K)
    args = ap.parse_args()
    _rank, _world, _local_rank = _bootstrap_standalone()
    if _rank == 0:
        _cap = torch.cuda.get_device_capability(_local_rank)
        print(
            f"[repro] nvshmem init OK: mode={args.mode} K={args.K} world={_world} sm={_cap} "
            f"dev={torch.cuda.get_device_name(_local_rank)}",
            flush=True,
        )
    _run_loop(args.mode, args.K, _rank, _world, _local_rank)
    try:
        nvshmem.core.finalize()
    except BaseException:
        pass
    try:
        torch.distributed.destroy_process_group()
    except BaseException:
        pass


@pytest.fixture(scope="session")
def _module_skip__distributed__nvshmem_registration_cap():
    pytest.importorskip("nvshmem.core")


def _fake_ib_sysfs(root, devices):
    """Build a fake ``/sys/class/infiniband`` tree so the RDMA probe can be tested off real hardware.

    Purpose: `_detect_nvshmem_traits` classifies HCAs by reading ``ports/<p>/link_layer`` and
    ``ports/<p>/state``. On any real host those files describe whatever fabric happens to be
    present, so the ACTIVE/DOWN and IB/RoCE branches cannot both be exercised. This fabricates them.

    Args:
        root: a ``pathlib.Path`` to populate (use pytest's ``tmp_path``). Created if absent.
        devices: mapping ``{device_name: (link_layer, state)}``, e.g.
            ``{"mlx5_0": ("InfiniBand", "4: ACTIVE"), "mlx5_1": ("InfiniBand", "1: DOWN")}``.
            Both values are written VERBATIM, because the production code matches the sysfs
            spelling (``"4: ACTIVE"``) rather than a bare word -- a test that wrote ``"ACTIVE"``
            would pass while the real format silently failed. Port ``1`` is always the one written;
            the probe reads ports ``1`` then ``2`` and stops at the first that exists.

    Returns:
        The ``root`` path, so a caller can pass it straight to the monkeypatched seam.
    """
    for name, (link_layer, state) in devices.items():
        port = root / name / "ports" / "1"
        port.mkdir(parents=True, exist_ok=True)
        (port / "link_layer").write_text(link_layer + "\n")
        (port / "state").write_text(state + "\n")
    return root


@matrix_exempt(
    "probes the sysfs RDMA classifier on a fabricated tree; no kernel, no mesh, no GPU -- the "
    "matrix's axes do not apply"
)
def test_a_down_ib_port_is_kept_out_of_the_nvshmem_hca_pin(tmp_path, monkeypatch):
    """A DOWN InfiniBand port must never reach ``NVSHMEM_HCA_LIST``.

    This is the failure the pin exists to prevent: a DOWN (or wrong-fabric) HCA in the list aborts
    nvshmem bootstrap at C level with exit 255 and no Python traceback -- only the bad-HCA rank
    dies, while every peer hangs. Before the port-state check the probe read ``link_layer`` alone,
    so a DOWN IB port was pinned exactly like an ACTIVE one.
    """
    root = _fake_ib_sysfs(
        tmp_path,
        {
            "mlx5_0": ("InfiniBand", "4: ACTIVE"),
            "mlx5_1": ("InfiniBand", "1: DOWN"),
            "mlx5_2": ("Ethernet", "4: ACTIVE"),
        },
    )
    monkeypatch.setattr(dist_manager_mod, "_IB_SYSFS_ROOT", str(root))
    monkeypatch.setenv("WORLD_SIZE", "16")
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "8")
    traits = DistributedManager._detect_nvshmem_traits()

    assert traits["mixed_fabric"] is True, "one Ethernet + two IB HCAs is a mixed fabric"
    assert traits["ib_hca_list"] == "mlx5_0:1", (
        f"the pin must carry ONLY the ACTIVE IB port; got {traits['ib_hca_list']!r}"
    )
    assert traits["n_active_ib_ports"] == 1


@matrix_exempt(
    "probes the sysfs RDMA classifier on a fabricated tree; no kernel, no mesh, no GPU -- the "
    "matrix's axes do not apply"
)
def test_all_ib_node_now_gets_a_pin_of_its_active_rails(tmp_path, monkeypatch):
    """An all-IB node is pinned to its ACTIVE rails.

    Restricting the pin only to ``mixed_fabric`` leaves an all-IB node unrestricted. That condition
    prevents use of a wrong-FABRIC NIC but says nothing about rail SPREAD: on a homogeneous 8-rail
    node the empty list cost 28.6x on a cross-node A2A (23.78 ms, against 0.83 ms once the list was
    supplied by hand). This test requires active-rail pinning on all-IB nodes.

    The DOWN rail must still be excluded: the whole point of the pin is that a DOWN port in it
    aborts nvshmem bootstrap C-level.
    """
    root = _fake_ib_sysfs(
        tmp_path,
        {
            "mlx5_0": ("InfiniBand", "4: ACTIVE"),
            "mlx5_1": ("InfiniBand", "4: ACTIVE"),
            "mlx5_2": ("InfiniBand", "1: DOWN"),
        },
    )
    monkeypatch.setattr(dist_manager_mod, "_IB_SYSFS_ROOT", str(root))
    monkeypatch.setenv("WORLD_SIZE", "16")
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "8")
    traits = DistributedManager._detect_nvshmem_traits()

    assert traits["mixed_fabric"] is False, "every HCA is InfiniBand -> not a mixed fabric"
    assert traits["ib_hca_list"] == "mlx5_0:1,mlx5_1:1", (
        f"an all-IB node must now be pinned to its ACTIVE rails; got {traits['ib_hca_list']!r}"
    )
    assert traits["n_active_ib_ports"] == 2, "the DOWN rail must not be counted as ACTIVE"


@matrix_exempt(
    "probes the sysfs RDMA classifier on a fabricated tree; no kernel, no mesh, no GPU -- the "
    "matrix's axes do not apply"
)
def test_a_node_with_no_active_ib_port_yields_an_empty_pin(tmp_path, monkeypatch):
    """No ACTIVE IB port -> empty pin, so nvshmem is left to its own device discovery.

    The pin must never be fabricated from nothing: exporting an empty NVSHMEM_HCA_LIST would be
    indistinguishable from "not set" to nvshmem, and exporting a DOWN port would abort bootstrap.
    This is also the state the init banner warns about on a multi-node job.
    """
    root = _fake_ib_sysfs(
        tmp_path,
        {"mlx5_0": ("InfiniBand", "1: DOWN"), "mlx5_1": ("Ethernet", "4: ACTIVE")},
    )
    monkeypatch.setattr(dist_manager_mod, "_IB_SYSFS_ROOT", str(root))
    monkeypatch.setenv("WORLD_SIZE", "16")
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "8")
    traits = DistributedManager._detect_nvshmem_traits()

    assert traits["ib_hca_list"] == "", "no ACTIVE IB port -> nothing to pin"
    assert traits["n_active_ib_ports"] == 0


# ---------------------------------------------------------------------------------------------
# MASTER_ADDR derivation under a bare `srun` (no torchrun). Host-only: pure text + os.environ, no
# process group, no GPU, so these run in any collection.
#
# REGRESSION. Taking MASTER_ADDR from SLURM_LAUNCH_NODE_IPADDR alone names the host where `srun`
# was invoked, which is rank 0's node only when srun runs inside the allocation. Invoked from a
# login shell it names the login node, and every rank then blocks forever in the env:// TCPStore
# rendezvous against a host with no store. A hang is the worst failure shape here: it reads as a
# slow kernel rather than a wrong address.
def _capture_setup(monkeypatch):
    """Run an initializer's PARSE and capture what it would hand to ``_setup``, without initializing.

    ``_setup`` is where the parse ends, and it calls ``init_process_group``, which BLOCKS in the
    env:// rendezvous rather than declining when the address does not resolve. Measured: a test that
    let it run hung until the outer timeout killed the session at 900 s, and the failure looked like
    a hang in the harness rather than a wrong assertion. There is deliberately no separate parse
    function to call instead (the initializers own their parse; see the refactor doc), so the parse
    is observed by intercepting its single consumer.
    """
    seen = {}

    def _fake_setup(*args, **kwargs):
        seen.update(kwargs)

    monkeypatch.setattr(DistributedManager, "_setup", staticmethod(_fake_setup))
    return seen


_FIRST_HOST_CASES = [
    ("node001", "node001"),
    ("n-[1-2]", "n-1"),
    ("n-[1,3]", "n-1"),
    ("n-[1-2,5]", "n-1"),
    ("n-[1-3,7]", "n-1"),
    ("n-[01-09]", "n-01"),  # zero padding is preserved -- n-1 would not resolve
    ("n-[1-2]-ib", "n-1-ib"),  # suffix after the bracket group
    ("a,b-[1-2]", "a"),  # top-level comma splits first
    ("b-[1-2],a", "b-1"),  # a comma INSIDE brackets does NOT split
    ("pool0-00171", "pool0-00171"),
    ("", None),
    ("   ", None),
]


@matrix_exempt("host-only: parses a Slurm nodelist string; no mesh, no group, no device")
@pytest.mark.parametrize(
    "nodelist,expected", _FIRST_HOST_CASES, ids=[c[0] or "empty" for c in _FIRST_HOST_CASES]
)
def test_slurm_first_host_parses_compact_nodelists(nodelist, expected):
    """The nodelist parser returns rank 0's host for every compact form Slurm emits.

    The two comma cases are the ones that matter: `a,b-[1-2]` must yield `a` while `b-[1-2],a` must
    yield `b-1`. A parser that splits on every comma silently returns a WRONG host for the second,
    and a wrong host hangs the rendezvous instead of raising.
    """
    assert DistributedManager._slurm_first_host(nodelist) == expected


@matrix_exempt("host-only: env derivation only; no mesh, no group, no device")
def test_derive_master_addr_prefers_step_nodelist_over_launch_node(monkeypatch):
    """MASTER_ADDR comes from the STEP's nodelist, never from the srun launch host.

    Simulates `salloc` (2 nodes) + `srun -N1` that Slurm placed on the SECOND node, launched from a
    login host. The launch-node value is present and WRONG; the step nodelist is present and right.
    """
    for k in (
        "MASTER_ADDR",
        "MASTER_PORT",
        "RANK",
        "WORLD_SIZE",
        "LOCAL_RANK",
        "LOCAL_WORLD_SIZE",
        "TORCHELASTIC_RUN_ID",
    ):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("SLURM_PROCID", "0")
    monkeypatch.setenv("SLURM_NTASKS", "8")
    monkeypatch.setenv("SLURM_LOCALID", "0")
    monkeypatch.setenv("SLURM_NTASKS_PER_NODE", "8")
    monkeypatch.setenv("SLURM_JOB_NODELIST", "n-[1-2]")  # the whole allocation
    monkeypatch.setenv("SLURM_STEP_NODELIST", "n-2")  # what THIS step got
    monkeypatch.setenv("SLURM_LAUNCH_NODE_IPADDR", "10.0.0.1")  # the login node
    seen = _capture_setup(monkeypatch)
    DistributedManager._initialize_slurm()
    assert seen["addr"] == "n-2"
    assert seen["addr"] != "10.0.0.1"
    # The identity too, from the SLURM namespace alone. This used to read os.environ, because the
    # derivation wrote there; the write now happens inside `_setup`, which is intercepted here.
    # That the export still reaches os.environ is a separate, launched assertion -- it needs a real
    # init, so a host-only test cannot make it.
    assert seen["rank"] == 0 and seen["world_size"] == 8 and seen["local_world_size"] == 8


@matrix_exempt("host-only: launcher detection only; no mesh, no group, no device")
def test_torchrun_selects_the_env_branch_even_though_slurm_vars_are_present(monkeypatch):
    """Under torchrun the launcher is ENV, however completely SLURM's variables are populated.

    ORIGINAL PURPOSE, PRESERVED: torchrun owns the env:// variables and nothing may overwrite them
    with Slurm-derived values. That used to be expressed as "the derivation is a no-op"; with the
    derivation gone it is expressed as "the SLURM branch is never selected", which is the same
    defect one level up and is now the thing that would actually break.

    Why the SLURM variables are set here and not omitted: torchrun spawns its workers INSIDE an
    srun step, so both namespaces are always populated on a cluster, and the SLURM ones describe
    the OUTER step. A detector that cannot tell these apart returns SLURM, every worker resolves
    rank 0 of a 1-rank world, and every collective silently becomes a no-op. Dropping the SLURM
    variables from this test would make it pass against a detector that has that bug.
    """
    monkeypatch.setenv("TORCHELASTIC_RUN_ID", "none")  # the literal string torchrun uses by default
    monkeypatch.setenv("RANK", "3")
    monkeypatch.setenv("WORLD_SIZE", "8")
    monkeypatch.setenv("SLURM_PROCID", "0")  # the OUTER step: one agent, not one worker
    monkeypatch.setenv("SLURM_NTASKS", "1")
    monkeypatch.setenv("SLURM_STEP_NODELIST", "n-2")
    assert DistributedManager._detect_launcher() == "ENV"


@matrix_exempt("host-only: launcher detection only; no mesh, no group, no device")
def test_a_stale_complete_env_set_under_srun_is_refused_in_favour_of_slurm(monkeypatch):
    """A complete but STALE env:// set does not capture a bare-srun launch.

    This is the case ordered fallback cannot express and the launcher identifier exists for. With
    `RANK=0 WORLD_SIZE=1 ...` left in a shell profile, a try-env-then-slurm resolver parses it
    successfully and every rank of a 16-way srun becomes rank 0 of a 1-rank world -- silently
    wrong, no error. `is_torchelastic_launched()` being False PROVES the launcher is not torchrun
    (torchrun always sets the variable), so a complete env:// set can only be stale and SLURM's
    variables are authoritative.
    """
    monkeypatch.delenv("TORCHELASTIC_RUN_ID", raising=False)
    for k, v in (
        ("RANK", "0"),
        ("WORLD_SIZE", "1"),
        ("LOCAL_RANK", "0"),
        ("LOCAL_WORLD_SIZE", "1"),
        ("MASTER_ADDR", "127.0.0.1"),
        ("MASTER_PORT", "29500"),
    ):
        monkeypatch.setenv(k, v)
    monkeypatch.setenv("SLURM_PROCID", "5")
    monkeypatch.setenv("SLURM_NTASKS", "16")
    assert DistributedManager._detect_launcher() == "SLURM"


@matrix_exempt("host-only: refusal path only; no mesh, no group, no device")
def test_create_grid_group_refuses_at_the_front_door_without_initialize():
    """An identity-dependent routine refuses where the caller can act on it, not three frames in.

    `create_grid_group` is the README's SECOND call, and before this guard the failure arrived from
    `_create_device_mesh_and_groups` as a torch.distributed complaint about an uninitialized default
    group -- which reads as a bug in the collective rather than as a missing `initialize()`.

    Scoped deliberately: `symmetric_mempool`, `release_symmetric_mempools` and
    `_barrier_with_timeout` are NOT guarded, because they sit on allocation and TEARDOWN paths and
    `cleanup()` wipes `_state`, so a guard there would fire during an atexit running after it and
    break a path that works today.
    """
    import collections

    if DistributedManager.is_initialized():
        rank_invariant_skip(
            "the manager is already initialized in this session, so the refusal cannot be observed",
            because="every rank of a launch calls initialize() together, so is_initialized() is "
            "job-uniform: either all ranks skip this or none do",
        )
    with pytest.raises(RuntimeError, match="initialize"):
        DistributedManager.create_grid_group(collections.OrderedDict(cp=2))


@matrix_exempt("host-only: launcher detection only; no mesh, no group, no device")
def test_no_launcher_is_reported_as_none_rather_than_guessed(monkeypatch):
    """With neither launcher present the detector returns None; `initialize()` turns that into the
    refusal. Kept separate from `initialize()` so the harness can ask the question and SKIP."""
    for k in (
        "TORCHELASTIC_RUN_ID",
        "RANK",
        "WORLD_SIZE",
        "SLURM_PROCID",
        "SLURM_NTASKS",
        "SLURM_NPROCS",
    ):
        monkeypatch.delenv(k, raising=False)
    assert DistributedManager._detect_launcher() is None


# The derivation is a FALLBACK, never a policy. `MASTER_ADDR` / `MASTER_PORT` are PyTorch's env://
# convention, not Slurm's -- Slurm exports neither, so whatever sets them is site tooling: an sbatch
# preamble, a TaskProlog, or a container hook. Such a site has picked an address and a port that its
# own fabric is known to route, and overriding it substitutes a guess for a known-good value.
#
# The torchrun no-op above does NOT cover this: it returns at the TORCHELASTIC_RUN_ID gate, several
# branches before the MASTER_* blocks, so the `not in os.environ` guards on those blocks are never
# reached by it. Measured -- replacing both guards with `if True:` leaves that test, and every other
# test in this file, GREEN. These cases are what fail the mutant.
_SITE_RENDEZVOUS_CASES = [
    # id, pre-set MASTER_ADDR, pre-set MASTER_PORT
    ("both", "site-host", "45001"),
    ("addr_only", "site-host", None),
    ("port_only", None, "45001"),
]


@matrix_exempt("host-only: env parsing only; no mesh, no group, no device")
@pytest.mark.parametrize(
    "addr,port",
    [(c[1], c[2]) for c in _SITE_RENDEZVOUS_CASES],
    ids=[c[0] for c in _SITE_RENDEZVOUS_CASES],
)
def test_env_branch_uses_the_site_rendezvous_verbatim(monkeypatch, addr, port):
    """On the env:// branch, a site's MASTER_ADDR/MASTER_PORT are used exactly as supplied.

    ORIGINAL PURPOSE, PRESERVED BUT NARROWED -- and the narrowing is the point, not an accident.
    This test used to assert that a site's rendezvous survived a Slurm-derived one under ANY
    launch, because one function read both namespaces and had to be stopped from clobbering. There
    is no such function now: each branch reads one namespace, so the guarantee is exactly "the
    env:// branch uses the env:// values" and nothing wider.

    What that costs, stated so nobody rediscovers it as a bug: a site whose env:// set is
    INCOMPLETE -- a container hook exporting RANK/LOCAL_RANK/WORLD_SIZE/MASTER_ADDR/MASTER_PORT but
    not LOCAL_WORLD_SIZE, which is a real, measured shape -- no longer reaches this branch at all.
    It is a bare srun, the SLURM branch claims it, and its address and port are re-derived. Such a
    site sets CPO_DISTRIBUTED_INIT_METHOD=ENV to get back here, and must then supply the missing
    variable.
    """
    # Neutralize the AMBIENT launcher, not just the variables under test. This file is host-only
    # but it is COLLECTED under srun, where SLURM_PROCID is set for real -- and a bare
    # `_detect_launcher()` would then correctly answer "SLURM" and this assertion would fail.
    # Measured: 3 failures on a 2-node srun run against a version that passed on a laptop, which is
    # the shape of every environment-dependent host-only test.
    for k in (
        "TORCHELASTIC_RUN_ID",
        "MASTER_ADDR",
        "MASTER_PORT",
        "SLURM_PROCID",
        "SLURM_NTASKS",
        "SLURM_NPROCS",
    ):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("RANK", "5")
    monkeypatch.setenv("LOCAL_RANK", "5")
    monkeypatch.setenv("WORLD_SIZE", "16")
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "8")  # COMPLETE here -- that is what selects this branch
    monkeypatch.setenv("MASTER_ADDR", addr if addr is not None else "site-host")
    monkeypatch.setenv("MASTER_PORT", port if port is not None else "45001")
    assert DistributedManager._detect_launcher() == "ENV"
    seen = _capture_setup(monkeypatch)
    DistributedManager._initialize_env()
    assert seen["addr"] == (addr if addr is not None else "site-host")
    assert seen["port"] == (port if port is not None else "45001")


@matrix_exempt("host-only: env parsing only; no mesh, no group, no device")
def test_env_branch_refuses_an_incomplete_env_set(monkeypatch):
    """The env:// branch requires ALL of its variables, which is what makes ordering work.

    The old form accepted RANK + WORLD_SIZE and let the rest default, so a five-of-seven container
    hook was accepted here with LOCAL_WORLD_SIZE left unset -- and LOCAL_WORLD_SIZE is what
    `_detect_nvshmem_traits` uses to tell a one-node job from a two-node one.
    """
    monkeypatch.delenv("TORCHELASTIC_RUN_ID", raising=False)
    monkeypatch.delenv("LOCAL_WORLD_SIZE", raising=False)
    for k, v in (
        ("RANK", "5"),
        ("WORLD_SIZE", "16"),
        ("LOCAL_RANK", "5"),
        ("MASTER_ADDR", "site-host"),
        ("MASTER_PORT", "45001"),
    ):
        monkeypatch.setenv(k, v)
    with pytest.raises(RuntimeError, match="LOCAL_WORLD_SIZE"):
        DistributedManager._initialize_env()


@matrix_exempt("host-only: env derivation only; no mesh, no group, no device")
def test_derived_port_stays_below_the_hosts_ephemeral_floor(monkeypatch):
    """The derived rendezvous port never lands in the range the kernel hands out to outbound sockets.

    REGRESSION. The port used to be `20000 + (job_id % 20000)` -> 20000-40000, which OVERLAPS the
    ephemeral range on a default Linux host (32768-60999). A rank whose derived port collided with
    an already-open outbound socket died `EADDRINUSE` while its peers ran, and it presents as rank 0
    never listening -- so it reads as a hang, not as a port collision.

    Swept over job ids rather than asserting one value: the property is what matters, and a single
    id can satisfy a formula that is wrong for the rest of the range. Reads the port from what
    `_setup` RECEIVES, not from `os.environ`, so it also pins that the value is PASSED --
    `_initialize_slurm` used to omit `port=` entirely, silently applying `_setup`'s "29500" default
    to every job on the cluster.
    """
    floor = ephemeral_floor() or 65536
    for k in ("TORCHELASTIC_RUN_ID", "RANK", "WORLD_SIZE", "LOCAL_RANK", "LOCAL_WORLD_SIZE"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("SLURM_PROCID", "0")
    monkeypatch.setenv("SLURM_NTASKS", "2")
    monkeypatch.setenv("SLURM_LOCALID", "0")
    monkeypatch.setenv("SLURM_STEP_NODELIST", "n-1")
    seen = _capture_setup(monkeypatch)
    for jid in (1, 7, 918787, 1970909, 1971231, 2**31 - 1):
        for sid in (0, 12, 22):
            monkeypatch.setenv("SLURM_JOB_ID", str(jid))
            monkeypatch.setenv("SLURM_STEP_ID", str(sid))
            DistributedManager._initialize_slurm()
            got = int(seen["port"])
            assert 1024 < got < floor, f"jid={jid} sid={sid} -> {got}, floor={floor}"
