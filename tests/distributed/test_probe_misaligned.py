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

"""TEMPORARY probe: what does the 16-B per-peer guard actually refuse at runtime?

**This file is a measurement, not a test, and it is meant to be deleted.** It exists because two
readings of the same source disagreed and neither of us could settle it by reading further. It
answers three questions in one launch, and each is printed rather than asserted so that a refusal
and an acceptance are equally reportable outcomes:

1. **The exact raise text** at the factored-misaligned cell, for the region's ``match=``. A pattern
   chosen from a docstring's wording rather than the raise is the weak-alternation trap; the
   docstring at ``:517`` says "each per-axis extent must be %8 (16-B TMA)", which appears in NO
   raise in the package.
2. **Is a FLAT misaligned cell refused or accepted?** If accepted, the region covers the 17 factored
   cells; if refused, it covers 48 and both source readings were wrong in different directions.
3. **Does the factored-misaligned cell raise WITHOUT ``pe_aligned_tiling``?** The guard reads
   ``if cp1 > 1 and self._pe_aligned_tiling``, so the flag gates it -- and it is a configure
   argument, not a declared matrix axis. If the answer is "no raise", a region written on
   ``(N, mesh)`` alone OVER-CLAIMS: it would make the matrix refuse to build grids that are legal.

A pytest unit rather than an ad-hoc script, deliberately: the conftest fixtures already solve
distributed bring-up, and a standalone script that re-implements it gets it wrong -- measured, twice,
on this team today.
"""

import traceback

import pytest
import torch
from cutlass import Float32
from torch.distributed.tensor import Shard

from fold_cp_ops._internal.compile_time.cute_dsl_utils import torch2cute_dtype_map
from fold_cp_ops.distributed.gemm_a2a_epi import GemmA2ASm90
from fold_cp_ops.distributed.pe_map import PeMap
from fold_cp_ops.testing.collective_guard import rank_invariant_skip
from fold_cp_ops.testing.kernel_matrix import matrix_exempt
from fold_cp_ops.testing.numeric_guard import numeric_exempt

# N=536 is one of 17 pool cells whose 2-D split misaligns: 536/2 = 268, 268 % 8 == 4.
N_MISALIGNED = 536


def _verdict(fn):
    """Run ``fn``; return ``"OK"`` or ``"<Type>@<file>:<line>: <message>"``, untruncated.

    Untruncated on purpose: a 210-char cap once hid the decisive argument list of a raise on this
    same kernel and cost two iterations.
    """
    try:
        fn()
        return "OK (no raise)"
    except Exception as e:  # noqa: BLE001 -- the verdict IS the exception
        tb = traceback.extract_tb(e.__traceback__)[-1]
        return f"{type(e).__name__}@{tb.filename.split('/')[-1]}:{tb.lineno}: {e}"


def _gemm():
    """An unconfigured `GemmA2ASm90` at a shipped tile."""
    return GemmA2ASm90(
        Float32,
        torch2cute_dtype_map[torch.bfloat16],
        (128, 128),
        (1, 1, 1),
        pingpong=False,
        is_persistent=True,
    )


@matrix_exempt("a temporary measurement of a guard's runtime behaviour, not a kernel property")
@numeric_exempt("prints refusal verdicts; computes no value")
@pytest.mark.parametrize("pe_aligned", [True, False], ids=["pe_aligned", "no_pe_aligned"])
def test_probe_report(dist_manager, apply_mesh, world_size, pe_aligned):
    """Print the three verdicts. Always passes -- the OUTPUT is the result."""
    if world_size != 4:
        rank_invariant_skip(
            f"the factored cp=(2,2) spec needs 4 ranks; WORLD_SIZE={world_size}",
            because="WORLD_SIZE is one number for the whole job, read identically by every rank",
        )
    manager = apply_mesh((("cp", (2, 2)),))
    mesh = getattr(manager, "device_mesh_subgroups", None) or manager.device_mesh
    placements = [Shard(i + 1) for i in range(mesh.ndim)]
    pm = PeMap.from_mesh_placements(mesh, placements, distributed_manager=manager)
    pe_table = tuple(int(v) for v in pm.cp_pe_table.tolist())

    def factored():
        _gemm().configure_a2a_sharded(
            mesh,
            placements,
            pe_map=pm,
            B=1,
            N=N_MISALIGNED,
            gemm_native=True,
            pe_aligned_tiling=pe_aligned,
        )

    def flat():
        _gemm().configure_a2a_gemm_native(
            cp=len(pe_table),
            my_cp_rank=int(pm.my_cp_rank),
            B=1,
            N_loc=N_MISALIGNED // len(pe_table),
            N=N_MISALIGNED,
            pe_table=pe_table,
            arbitrary_n=True,
            pe_aligned_tiling=pe_aligned,
        )

    # `configure_a2a_gemm_native` calls build_p2p_table -> nvshmem_team_translate_pe, which does
    # not raise without NVSHMEM: it hard-ABORTS the process at exit 255. The first probe run died
    # there and lost two of its three answers. The factored arm survives only because
    # configure_a2a_sharded runs its 16-B guard BEFORE the topology probe. Neutralised to an
    # all-NVLink verdict -- the misalignment question does not read the topology.
    import fold_cp_ops.distributed.gemm_sm90_a2a as G

    monkeypatch_target = G.build_p2p_table
    G.build_p2p_table = lambda pe_table: tuple([True] * len(tuple(pe_table)))
    try:
        _run_and_report(dist_manager, pe_aligned, factored, flat)
    finally:
        G.build_p2p_table = monkeypatch_target


def _run_and_report(dist_manager, pe_aligned, factored, flat):
    """Print the two verdicts on rank 0; every rank still executes both, so no rank diverges."""
    if dist_manager.rank == 0:
        print(f"\nPROBE pe_aligned_tiling={pe_aligned}", flush=True)
        print(f"PROBE  factored (2,2) N={N_MISALIGNED}: {_verdict(factored)}", flush=True)
        print(f"PROBE  flat cp=4     N={N_MISALIGNED}: {_verdict(flat)}", flush=True)
    else:
        _verdict(factored)
        _verdict(flat)
