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

"""Tests for ``fold_cp_ops.distributed.gemm_a2a_epi`` -- ``GemmA2ASm90``'s fused A2A stores.

Every test here constructs `GemmA2ASm90` and drives one of its epilogue A2A stores: a GEMM whose
output tile is written straight into a PEER's symmetric heap, with no local GMEM round-trip and no
separate all-to-all kernel. The configure-time REFUSALS of the same class hierarchy are a different
subject and live in ``test_gemm_sm90_a2a.py``; this file is about what the configured kernel
COMPUTES.

**The upstream carried these as FOUR separate files, pasted together.** The paste left four SPDX
headers and four module-level docstrings, only the first of which Python treated as the module
docstring -- the other three evaluated as bare strings and were discarded, so three of the four
subject descriptions below were invisible at runtime and to every documentation tool. They are
sections here instead. Each is otherwise the upstream's own text, and the sections keep the file's
four subjects legible without pretending they are four modules.

Section 1 -- back fused stores (the original "Wave-2 A2A-fusion correctness GATE")
    The back, GEMM-native (design-E) store: a BATCHED (L>1) GEMM ``tri[L] = A[L] @ B[L]^T`` whose
    CTA ``(i,j)`` tile of plane ``L`` is TMA-S2G-stored into a 5-D symmetric recv
    ``(cp, Dloc, B, N_loc, N) = [slot, d, b, i_local, j]`` on peer ``i//N_loc`` -- the ``S3 -> S1``
    token reshard, d-outer recv. And the back, plain (L=1) store, whose GEMM output ``(M,N)`` token
    rows scatter into a token-major recv with peer-select by the M-tile.

    The FRONT (gated postact) A2A store is NOT covered here. It was carried on a stagec-based class
    that production never used; production's fused front is the STAGED class, covered by its own
    file.

    ``cp`` is the session world size, and the reference reshards are cp-agnostic (they loop
    ``range(cp)`` over the PE-table peers), so a cp=4 launch collects and runs the same tests. The
    flag-OFF byte-identity test runs on every rank independently, with no comm, and guards the
    "default == local kernel" promise.

Section 2 -- composite-K back-READ, single-GPU (the original "DEADLOCK regression gate")
    A regression guard for the route2_ni incoming composite-K read: the back einsum contracts
    ``K = (cp, Xg_pad)`` from a per-rank-CONTIGUOUS D-major recv. It drives ONLY the read, in
    isolation -- a synthetic recv fed to ``GemmA2ASm90(_a2a_composite_k=True, recv=None)``, compared
    against a torch einsum. **No nvshmem and no torchrun**, so it is the one section here that is a
    plain single-GPU test.

    Why it exists: the composite-K K-hoist must route BOTH the producer (``load_AB``) AND the
    MMA-consumer k-counts through ``_k_tile_cnt``. Hooking only the producer makes the producer load
    ``cp*nt_within`` tiles while the consumer drains ``nt_within`` -- an AB-pipeline DEADLOCK. **The
    cp=2 aligned shape IS the deadlock trigger; keep it in the grid.**

Section 3 -- IB-ring back-A2A store (pe_aligned per-peer tiling over the GMEM-ring drain)
    The shipped pe_aligned coupled store is a SMEM->peer TMA-S2G, NVLink-only. ``ib_ring`` composes
    pe_aligned's per-peer M-tiling scheduler WITH the decoupled GMEM-ring ``put_nbi_warp`` drain:
    each peer write is sourced from local GMEM (the rotating ring) and issued by ``put_nbi_warp``,
    so NVSHMEM auto-selects NVLink P2P for a node-local peer and IB for a remote one -- one
    mechanism, hybrid transport.

    On a SINGLE node every peer is NVLink-reachable, so these validate the CORRECTNESS of the hybrid
    store mechanism and the dynamic ``N`` / ``arbitrary_n`` shape contract it must preserve;
    the IB transport itself auto-engages only across nodes. Anchor 640 is a straddle
    (``N_loc = 640//cp`` is not a multiple of 128) for cp in {2,4,8}.

Section 4 -- pe_aligned generalized to dynamic shape (+2-D, +dyn-cp)
    pe_aligned tiles each peer's ``N_loc`` rows at its own base (``peer*N_loc + k*tile_m``) so no
    output tile straddles a peer -- every store is a single fast TMA-S2G whose recv descriptor
    clamps the partial-last per-peer tile. These validate the GENERALIZED tile scheduler: an
    un-baked runtime ``N_loc`` means ONE compile, at a straddle anchor so ``arbitrary_n`` stays on
    and pe_aligned activates, serves MANY token counts. The 2-D cells cover square AND non-square
    ``cp0 x cp1`` -- the tile->peer route is the row-major ``cp0_coord*cp1 + cp1_coord`` flatten, so
    ``cp0`` need not equal ``cp1``.

The correctness metric
    The upstream gate was a POOLED scalar, ``max|got-ref| / max|ref| < 5e-2`` in sections 1-2 and
    ``rel_L2 = ||got-ref||/||ref|| < 2e-2`` in sections 3-4. Both are replaced by the element-wise
    comparisons in `fold_cp_ops.testing.numerics`, which is not a tightening of the bar so much as a
    change of what the bar is applied TO: a pooled metric averages away exactly the defects these
    stores produce -- a straddle row, a padded lane, an unwritten scatter cell -- and this file's own
    upstream comments say so, noting that a per-row view is what localizes a dropped token row.

    The per-row error histogram is KEPT, as a diagnostic. It reports the worst row's ``(b,i,j)`` and
    the outlier ratio, which is where a reader looks after a failure; it is simply no longer what
    decides pass or fail. Same for the recv coverage count (cells still holding the ``-99``
    sentinel), which the upstream also printed rather than asserted.

Launching
    Sections 1, 3 and 4 need torchrun or srun with >=2 ranks, plus nvshmem. Section 2 does not.
    See CLAUDE.md's "five launch forms" for the exact invocations; ``CPO_CACHE_ENABLED=0`` is
    required for a correctness run.

Note:
    NO ``from __future__ import annotations`` -- this module imports the CuTe-DSL kernel classes,
    and the import is forbidden project-wide for @cute-adjacent files (it stringizes ``Constexpr``
    parameter annotations, making them dynamic). NO ``cute.printf`` reaches any device store.
"""

import json
import os
from pathlib import Path

import pytest
import torch

import cutlass.torch as cutlass_torch
from cutlass import Float32, Int32
from cutlass.cute.runtime import from_dlpack
from torch.distributed.tensor import Shard

from fold_cp_ops._internal.arch import get_device_capacity, get_max_active_clusters
from fold_cp_ops._internal.compile_time.cute_dsl_utils import torch2cute_dtype_map
from fold_cp_ops._internal.gemm_tvm_ffi_utils import make_scheduler_args
from fold_cp_ops.distributed.gemm_a2a_epi import GemmA2ASm90
from fold_cp_ops.distributed.gemm_bitcode_compile import compile_gemm_with_bitcode
from fold_cp_ops.distributed.gemm_sm90_a2a import build_p2p_table
from fold_cp_ops.distributed.pe_map import PeMap
from fold_cp_ops.distributed import DistributedManager
from fold_cp_ops.testing.collective_guard import rank_invariant_skip
from fold_cp_ops.testing.capacity_guard import capacity_gate
from fold_cp_ops.testing.cubin_identity import assert_has_code, digest_export
from fold_cp_ops.testing.kernel_matrix import (
    Axis,
    KernelMatrix,
    Unsupported,
    front_door_raises,
    matrix_exempt,
)
from fold_cp_ops.testing.numeric_guard import numeric_exempt
from fold_cp_ops.testing.numerics import (
    assert_bitwise,
    assert_elementwise,
    assert_written,
    tolerance_bound,
)

from tests.distributed import topology
from tests.distributed.correctness_harness import compute_error_histogram

# Why every geometry skip below is rank-INVARIANT, declared once and shared. Not asserted from
# resemblance: `w8plan/epi_classify_skips.py` walked all 48 call sites, collected the free names of
# each guarding condition, and found ZERO on an exception path and ZERO reading a rank-local name.
# Each one is a function of the world size, the session mesh and the parametrized shape constants --
# values every rank of a launch reads identically -- so the ranks skip together or not at all. The
# two divergent shapes this rules out are the ones that would deadlock: a per-rank build failure
# caught in an `except`, and an OOM. Either of those must go through `CollectiveGate` instead.
_GEOMETRY_UNIFORM_BECAUSE = (
    "the predicate is a function of world_size, the session device mesh and the parametrized "
    "shape constants -- all job-uniform, all read identically by every rank -- so every rank "
    "reaches the same verdict and none can skip alone. Measured rather than assumed: no skip in "
    "this module sits on an exception path or reads a rank-local name"
)

#: Appended to the ``because=`` of every test that parametrizes ``mesh`` JOINTLY with ``N``.
#:
#: The mechanism, which is not obvious from the decorator: ``matrix.parametrize`` evaluates an
#: ``Unsupported`` region only when the SAME call parametrizes every axis that region reads. The
#: 16-B per-peer region reads N AND mesh, so stacking ``parametrize("N")`` above
#: ``parametrize("mesh")`` leaves neither call a superset of ``{N, mesh}``, the region check is
#: SKIPPED on both, and pytest then crosses them into the very combination the region forbids --
#: from a test that asserts SUCCESS. Measured three times in this file before it was understood as
#: one class: at ``pe_aligned_fastcfg`` (8 cells), at ``arbitrary_n``, and at ``pe_aligned_static_2d``
#: (1 cell, and only reachable at 16 ranks).
#:
#: The cells are DERIVED from ``_misaligned_2d`` rather than listed, because a frozen literal is
#: only correct on the day it is written: the sibling ``arbitrary_n`` list silently lost the two
#: newest mesh specs -- ``cp=(2,8)`` and ``cp=(4,4)``, the first FACTORED **and** CROSS-NODE pair,
#: i.e. the combination production actually runs -- and nothing reported it.
_JOINT_MESH_BECAUSE = (
    "mesh is parametrized JOINTLY with N rather than in a second decorator: a region is evaluated "
    "only when ONE call parametrizes every axis it reads, so stacking them silently skips the 16-B "
    "per-peer check and lets pytest cross this test into cells the matrix declares UNSUPPORTED. The "
    "cell list is DERIVED from _misaligned_2d so it cannot disagree with the region it is defined "
    "against, and so a mesh value added to the pool joins this test by construction"
)


# The declared test matrix for this module. `tests/perf/` has no counterpart here -- the subject is
# a distributed store's correctness, and its speed is measured by the harness rather than pinned.
def _mesh_ranks(spec):
    """Total ranks a mesh spec needs -- the product of every group size, tuples included.

    Used only by the mesh axis's facets, which have to answer "does this need more ranks than one
    node holds" without a live group. Kept a module function rather than a lambda because a facet
    predicate that has to compute is unreadable inline, and this one decides `cross_node`.
    """
    total = 1
    for _, size in spec:
        if isinstance(size, tuple):
            for s in size:
                total *= s
        else:
            total *= size
    return total


def _misaligned_2d(N, mesh):
    """True when a FACTORED mesh splits ``N`` into a per-axis extent that is not 16-B aligned.

    The 2-D store's per-peer boxes start at ``N//cp0`` and ``N//cp1`` element offsets, and TMA needs
    a 16-B-aligned box start -- 8 elements at bf16. A factored spec whose split lands off that
    boundary is refused at configure; the alternative is silent corruption, which the raise itself
    says.

    Args:
        N: The token extent, from the matrix pool.
        mesh: A mesh spec, ``(("cp", size), ...)``. A FLAT spec always returns False -- see the
            region's ``reason`` for why flat is a supported shape rather than an omission.

    Returns:
        True iff some axis of a factored spec divides ``N`` into a non-``%8`` extent, or fails to
        divide it at all.
    """
    for _, size in mesh:
        if not isinstance(size, tuple):
            continue
        return any(N % c != 0 or (N // c) % 8 != 0 for c in size)
    return False


EPI = KernelMatrix(
    kernel="gemm_a2a_epi",
    axes=(
        Axis(
            name="N",
            domain=(
                "the full token extent, i.e. the einsum's square i/j dimension. The pool is the "
                "UNION of the four concatenated suites' own token pools, kept whole rather than "
                "trimmed: each was chosen for a property the others do not have. 264/520/1032/1544 "
                "are the %8-but-not-%128 straddles (N_loc lands mid-tile at cp=2); 256/768 are "
                "%128 but straddle at cp>=4; 1032 and 1048 are the row-DIRTY phases (2N%32==16) "
                "whose clean neighbours 1040/1056 are the only single-variable comparison "
                "available; 640 is the straddle anchor for cp in {2,4,8}; 536 is the value whose "
                "2-D HALVES misalign (N_i=268, %8==4) and must therefore be REFUSED -- note it is "
                "536/2 that misaligns, not 536, which is itself %8 like every value here; and the "
                "2176..16896 rung is "
                "the multi-band cluster drain, which needs more bands than clusters to engage at "
                "all. Dropping any of them would silently narrow a facet"
            ),
            values=(
                256,
                264,
                384,
                512,
                520,
                536,
                576,
                640,
                768,
                776,
                896,
                960,
                1024,
                1032,
                1040,
                1048,
                1056,
                1152,
                1544,
                2048,
                2056,
                2176,
                4096,
                4352,
                5376,
                6400,
                8192,
                8448,
                10752,
                12800,
                16896,
            ),
            facets={
                "tile_aligned": lambda n: n % 128 == 0,
                "straddle": lambda n: n % 128 != 0,
                # NOT a 16-byte-alignment facet on N. Every value here is %8, and that is
                # structural rather than a pool choice: 16-byte alignment is the one shape
                # constraint the TMA hardware imposes, so a non-%8 bf16 token extent is not a
                # supported shape at all, and a pool containing one would be a pool of shapes the
                # kernel is entitled to refuse. The misalignment this file DOES test lives in the
                # DERIVED per-peer extent -- N=536 on a 2-D mesh gives N_i=268, which is %8==4 --
                # so it is a function of N AND the mesh, and no predicate on N alone expresses it.
                # Measured: an `n % 8 != 0` facet scored 0/31 and its complement 31/31, i.e. both
                # were pure noise and neither could ever have discriminated anything.
                "halves_to_misaligned": lambda n: (n // 2) % 8 != 0,
                "row_dirty_phase": lambda n: (2 * n) % 32 == 16,
                "small": lambda n: n <= 1024,
                "multi_band": lambda n: n >= 2176,
                "power_of_two": lambda n: n & (n - 1) == 0,
            },
        ),
        Axis(
            name="D",
            domain=(
                "the FULL feature width, so Dloc = D/cp is what one rank holds. Only the values a "
                "test can name as a constant appear: 4 and 8 are the small back-store widths (L=1 "
                "and L>1 at cp=2), 96 is LCM-chosen so Dloc is 6/4/3 at cp=16/24/32, giving an "
                "L>1 batched store at every multi-node cp from ONE value, and 128/256/384/512 are "
                "the target TriMul workflow's feature widths. D is never free: "
                "each test body skips unless D % cp == 0 (Dloc must be integral) and, on the 2-D "
                "shard, D % (cp0*cp1) == 0 -- so the pool is a set of constants and the (N, D) "
                "CELLS are a short coupled list, not a product. The pe_aligned suite is NOT in "
                "this pool on purpose -- it fixes Dloc=128 and derives D=128*cp, so its D is a "
                "function of the world size and cannot be a declared constant, which adding more "
                "constants here does not change"
            ),
            values=(4, 8, 96, 128, 256, 384, 512),
            facets={
                "l_is_one": lambda d: d <= 4,
                "l_gt_one": lambda d: d >= 8,
                "multi_node_width": lambda d: d >= 96,
                # Four target TriMul feature widths, declared as a FACET rather than left as
                # four literals in a cell list. The difference is not cosmetic: a facet fails the
                # module when every test narrows it away, and a literal just disappears. Measured
                # need -- before this, `coverage_problems` reported `multi_node_width` unreached
                # because D=96 was declared and emitted by NOTHING (ran: ['4', '8']), which is
                # precisely the rot a facet makes visible and a bare value does not.
                "workflow_D": lambda d: d in (128, 256, 384, 512),
            },
        ),
        Axis(
            name="B",
            domain=(
                "the TriMul batch extent. It is multiplicative on the work and, BY DESIGN, an "
                "input no kernel may depend on at compile time -- it reaches the back store only "
                "as a factor of the GEMM batch axis (`L = Dloc * B`), which the store decomposes "
                "in-kernel as `d = L // B`, `b = L % B`. Any B >= 1. The pool carries a "
                "NON-POWER-OF-TWO value on purpose: at B=2 the decode is a shift and a mask, so a "
                "wrong-but-aligned index can still land on a valid plane and read as correct; only "
                "a real division catches it. Declared as an axis rather than left as the `B = 1` "
                "literal it was, because `d`/`b` is the one place the batch extent enters the "
                "kernel at all, and a literal makes 'nobody swept B' indistinguishable from "
                "'B was swept and is fine'"
            ),
            values=(1, 2, 3),
            facets={
                "batched": lambda b: int(b) > 1,
                "non_pow2": lambda b: int(b) > 1 and (int(b) & (int(b) - 1)) != 0,
            },
        ),
        Axis(
            name="tile",
            domain=(
                "the CTA tile (tile_M, tile_N) the store's epilogue drains. tile_M=256 is the one "
                "value that produces MULTIPLE epilogue M-subtiles, which the multi-subtile drain "
                "exists for; tile_N=256 drives the 3-warpgroup producer-warp drain. A pool of "
                "(128,128) alone reaches neither"
            ),
            values=((128, 128), (128, 256), (256, 128), (256, 256)),
            facets={
                "single_epi_subtile": lambda t: t[0] == 128,
                "multi_epi_subtile": lambda t: t[0] == 256,
                # (256, 256) is the only cell where the m_sub row-walk and the 3-warpgroup
                # producer-warp drain are COMPOUNDED; each is validated alone at (256,128)
                # and (128,256), so a pool without it tests them only in isolation.
                "compounded": lambda t: t == (256, 256),
                "narrow_n": lambda t: t[1] == 128,
                "wide_n": lambda t: t[1] == 256,
            },
        ),
        Axis(
            name="pingpong",
            domain=(
                "the two-warpgroup alternating schedule. Both values are shipped configurations, "
                "and the store must be independent of the choice -- that independence is what the "
                "autotunability tests assert, so a one-value pool would make them vacuous"
            ),
            values=(False, True),
            facets={"cooperative": lambda p: not p, "two_warpgroup": lambda p: p},
        ),
        Axis(
            name="dyn_cp",
            domain=(
                "whether cp itself is a runtime value. False is every shipped path; True is a "
                "declared hard wall -- the reshard peer atoms are a compile-time host list with "
                "per-peer BAKED symmetric base addresses, and there is no runtime-base TMA-S2G. "
                "Both values are in the pool because a pool of False alone would make the region "
                "below look UNREACHABLE rather than REFUSED"
            ),
            values=(False, True),
            facets={"compile_per_cp": lambda v: not v, "runtime_cp": lambda v: v},
        ),
        Axis(
            name="mesh",
            domain=(
                "any OrderedDict-shaped spec of group-name -> int | tuple[int, ...] whose rank "
                "product equals WORLD_SIZE. The pool is PURE context-parallel and nothing else, "
                "because every mesh dim here carries the all-to-all: `_pe_map` builds "
                "[Shard(i) for i in range(mesh.ndim)], so a dp group would be shard dims the store "
                "does not route. Both FLAT and FACTORED cp are in the pool and the difference is "
                "not cosmetic -- the 2-D store's tile->peer UNRAVEL exercises both cp axes only "
                "under a factored spec, and a flat cp=4 collapses it to a single division, which "
                "is precisely the case that would pass while testing half the routing"
            ),
            values=(
                (("cp", 2),),
                (("cp", 4),),
                (("cp", 8),),
                (("cp", (2, 2)),),
                (("cp", (2, 4)),),
                (("cp", 16),),
                # --- FACTORED **and** CROSS-NODE. Until these, every factored spec was <=8 ranks
                # and every cross-node spec was flat, so the facet pair (factored, cross_node) was
                # UNREACHABLE: the 2-D tile->peer unravel was never once exercised with a peer on
                # the far side of InfiniBand. That is the combination production actually runs.
                # Under LayoutRight with 8 GPUs/node the two differ in where the fabric cuts:
                (("cp", (2, 8)),),  # axis_1 == exactly one node; axis_0 (stride 8) crosses IB
                (("cp", (4, 4)),),  # axis_1 == HALF a node -- a cp group SMALLER than its NVLink
                #                     domain, so peer routing that assumed "contiguous axis == the
                #                     whole node" passes (2,8) and fails here. Launch with
                #                     `srun -N2 --ntasks-per-node=8`; torchrun needs an srun
                #                     wrapper per node, and NEVER pass --gpus-per-task with 8
                #                     tasks/node (each task then sees 1 GPU, every cell skips, and
                #                     the run exits GREEN having tested nothing).
            ),
            facets={
                "flat": lambda s: all(isinstance(sz, int) for _, sz in s),
                "factored": lambda s: any(isinstance(sz, tuple) for _, sz in s),
                "square_2d": lambda s: any(
                    isinstance(sz, tuple) and len(set(sz)) == 1 for _, sz in s
                ),
                "nonsquare_2d": lambda s: any(
                    isinstance(sz, tuple) and len(set(sz)) > 1 for _, sz in s
                ),
                "single_node": lambda s: _mesh_ranks(s) <= 8,
                "cross_node": lambda s: _mesh_ranks(s) > 8,
            },
        ),
    ),
    computes=(
        # A batched GEMM whose result is scattered to peers: a contraction followed by pure data
        # movement. Neither trait implies a distribution property yet -- there is no reduction over
        # a normalized row and no saturating activation anywhere on this path, so the shape axes
        # already vary everything the error is sensitive to.
        "contraction",
        "data_movement",
    ),
    unsupported=(
        Unsupported(
            where=lambda N, mesh: _misaligned_2d(N, mesh),
            raises=ValueError,
            # TWO branches because TWO different guards refuse this geometry, depending on
            # `pe_aligned_tiling` -- and both are anchored so neither can be satisfied by an
            # unrelated raise. MEASURED at 4 ranks (N=536, cp=(2,2)):
            #   pe_aligned_tiling=True  -> :1350  "requires 16-B-aligned per-peer extents"
            #   pe_aligned_tiling=False -> :1238  "N_i_loc=N/cp0=268 must be a positive multiple..."
            # The second branch is anchored on `N_i_loc=N/cp0=` on purpose: the bare phrase
            # "must be a positive multiple of cta_tile_M" ALSO occurs at :555, the 1-D guard that
            # fires for any non-tile-multiple extent, aligned or not. An alternation is only as
            # strong as its weakest branch, so the generic branch is narrowed to the 2-D variable
            # name rather than left to match a different refusal.
            match=(
                r"requires 16-B-aligned per-peer extents"
                r"|N_i_loc=N/cp0=\d+ must be a positive multiple of cta_tile_M"
            ),
            reason=(
                "a FACTORED cp mesh splits N into per-peer boxes at N//cp0 and N//cp1 element "
                "offsets, and TMA requires a 16-B-aligned box start -- 8 elements at bf16. A split "
                "that lands off that boundary must be REFUSED at configure; the guard's own message "
                "says the alternative is silent corruption. UNCONDITIONAL in pe_aligned_tiling, "
                "which is why a third axis would have been WRONG rather than merely expensive: it "
                "would declare (536, (2,2), pe_aligned=False) LEGAL while that cell actually "
                "raises -- the matrix wrong in the dangerous direction, blessing a failing cell. "
                "Measured at 4 ranks, the cell raises "
                "with the flag ON (the 16-B guard) and with it OFF (the tile-multiple guard), by "
                "different guards with different messages. It also holds for every declared tile, "
                "because a misaligned N_i is 4 mod 8 while every tile_m in the pool is 0 mod 8, so "
                "N_i can never be a tile multiple -- checked across all 17 cells x 4 tiles, zero "
                "escapes. FLAT specs are deliberately NOT in the region: arbitrary_n exists to make "
                "a misaligned per-rank N_loc a SUPPORTED shape, its own 16-B requirement lands on N "
                "rather than N/cp, and every N in this pool is structurally %8 -- measured accepted "
                "at flat cp=4, N=536, N_loc=134"
            ),
        ),
        Unsupported(
            where=lambda dyn_cp: dyn_cp,
            raises=NotImplementedError,
            match=r"dynamic-cp",
            reason=(
                "dynamic cp is a hard wall for EVERY A2A store, not a gap: the reshard peer atoms "
                "are a compile-time host list carrying per-peer BAKED symmetric base addresses, "
                "and there is no runtime-base TMA-S2G to retarget them with. Pure-cp A2A also has "
                "cp == world, and a different world is a different nvshmem job, i.e. a different "
                "launch -- so compile-per-cp is not a workaround but the only coherent meaning. It "
                "must REFUSE at configure rather than bake one cp's addresses and store to them"
            ),
        ),
    ),
)


# --------------------------------------------------------------------------- #
# Correctness metric — IDENTICAL to the single-device parent kernels' tests
# (tests/test_dual_gated_gemm_stagec.py / test_layernorm_gemm_stagec.py): the bf16
# tensor-core relative-error bar ``max_abs_err / ref_abs_max < 5e-2``.
# --------------------------------------------------------------------------- #
REL_BAR__distributed__a2a_fusion = (
    5e-2  # bf16 tensor-core relative-error bar (the parent kernels' bar)
)


# the ONE topology guard for this file's COUPLED peer stores (this suite shipped with zero
# topology guards, §4.5). A coupled store — back ``configure_a2a_gemm_native``/``configure_a2a_sharded``
# without ``decoupled=True``/``ib_ring=True`` — TMA-S2Gs/STGs into a
# peer's symmetric heap via ``nvshmem_ptr``, which returns NULL for an IB peer, so a cross-node cell
# takes an illegal memory access. ``pe_aligned_tiling`` is NOT exempt: it faults independently through
# its own base_m / aligned-predicate / TMA-descriptor chain (measured isolated at cp=16). Cross-node
# coverage lives on the decoupled ib_ring / V6 drain. Rank-invariant predicate -> collective skip (E4).
def _skip_coupled_cross_node(pm, what):
    topology.skip_if_coupled_cross_node(
        tuple(int(x) for x in pm.cp_pe_table.tolist()),
        detail=f"{what} is a coupled store; cross-node is covered by the decoupled ib_ring / V6 drain.",
    )


def _assert_recv_matches(got, expected, *, bar, what):
    """Gate a recv against its fp32 reference PER ELEMENT, at the upstream's own bar.

    Purpose
        The one numeric gate in this module. It replaces two identical pooled helpers the upstream
        carried -- ``_rel_err`` and ``_rel``, one per concatenated file -- and the ``rel_l2`` term
        of the histogram gate.

    Why the bar does not move, and where it genuinely tightens
        The upstream's ``max|out-ref| / max|ref| < bar`` is ALREADY a statement about every element:
        it says no element's error exceeds ``bar * max|ref|``. So converting it to
        ``assert_elementwise`` with ``atol = bar * max|ref|`` is the SAME bar -- what changes is the
        report, which now names the worst offender's index and both values instead of one scalar.

        The real tightening is at the ``rel_l2`` call sites. ``||got-ref|| / ||ref||`` is a genuine
        average, and this module's own upstream comments say what that costs: one corrupted token
        row among thousands barely moves the norm. Those sites gate on this function now, so a
        single bad element fails at its own index rather than needing to dominate a norm.

    Args:
        got: The recv, as written by the store. Any float dtype; compared in fp64.
        expected: The fp32 reference, same shape. Its global max sets the absolute bound, so it
            must be the REFERENCE and not the output -- passing them the other way round scales the
            bound by a corrupted value and can only loosen it.
        bar: The relative bar, as a fraction of ``max|expected|``. The upstream's two values are
            5e-2 (sections 1-2, the parent kernels' bf16 tensor-core bar) and 2e-2 (sections 3-4).
            Must be positive.
        what: Short label carried into the failure message -- pass the cell's ``tag`` so a failure
            names the shape and cp without the reader going back to the parametrization.

    Returns:
        The worst observed ``|err| / bound`` ratio, as a float in ``[0, 1]`` on success. Printed by
        the callers: a value creeping toward 1.0 says the bar is about to become flaky.

        **It is normalized to the BOUND, so it fails at 1.0 and NOT at ``bar``.** Callers must say
        so when they print it. They used to render it as ``worst_ratio=0.109 (bar 0.05)``, which
        invites exactly one wrong reading -- a 2x bar violation reported as PASS -- and the person
        it eventually catches will be reading a FAILURE at 3am, not a pass. Note the contrast with
        this module's ``rel=...`` prints, where ``(bar ...)`` genuinely IS the threshold: two
        different quantities were being labelled the same way.

    Raises:
        AssertionError: If ``got`` holds non-finite values, or any element exceeds its bound. The
            message gives the violating count and the worst offenders with coordinates.
    """
    scale = expected.double().abs().max().item()
    return assert_elementwise(
        got,
        expected,
        tolerance_bound(expected, atol=bar * (scale if scale > 0 else 1.0), rtol=0.0),
        what=what,
    )


# --------------------------------------------------------------------------- #
# nvshmem bootstrap (session-scoped, collective).
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="session", autouse=False)
def _nvshmem__distributed__a2a_fusion(dist_manager):
    """Initialize nvshmem4py over the world group ONCE for the session.

    The conftest ``dist_manager`` fixture brings up torch.distributed + the device
    mesh but NOT nvshmem; the fused stores allocate symmetric recv buffers
    (``nvshmem.core.interop.torch.tensor``), which require nvshmem to be live.
    ``DistributedManager.init_nvshmem()`` is a COLLECTIVE (broadcasts a unique id +
    barriers), so it must run in lockstep on every rank — a session-scoped autouse
    fixture (depending on ``dist_manager``) does exactly that, before the first
    test. Idempotent; the conftest's session teardown ``DistributedManager.cleanup()``
    already finalizes nvshmem, so this fixture adds no teardown of its own.
    """
    from fold_cp_ops.distributed.distributed_manager import DistributedManager

    DistributedManager.init_nvshmem()
    yield


# --------------------------------------------------------------------------- #
# Shared nvshmem drive helpers (lifted from the smokes).
# --------------------------------------------------------------------------- #


def _nvshmem_barrier():
    import nvshmem.core

    nvshmem.core.barrier_all(stream=torch.cuda.current_stream())


def _drain():
    """Quiesce all in-flight puts + barrier + device sync (post-store, T2.3 caller)."""
    import nvshmem.core
    import nvshmem.core.rma as nvshmem_rma

    nvshmem_rma.quiet(stream=torch.cuda.current_stream())
    nvshmem.core.barrier_all(stream=torch.cuda.current_stream())
    torch.cuda.synchronize()


def _pe_map(dist_manager):
    """The cp PE-map over the SESSION mesh (whatever ``CPO_DIST_MESH`` declared).

    The conftest session mesh is a PURE context-parallel mesh (default ``cp=WORLD``,
    or e.g. ``cp=2*2``): EVERY mesh dim carries the all-to-all, so every dim is a cp
    (Shard) axis. We therefore build a placement of length == the session mesh ndim
    with each dim a distinct ``Shard`` — ``PeMap.from_mesh_placements`` flattens the
    Shard axes row-major into ``PeMap.cp == prod(mesh shape) == world_size`` with
    ``cp_pe_table`` mapping flat peer -> global PE. Mesh-ndim- AND cp-agnostic: a 1-D
    ``cp=2`` torchrun and a 2-D ``cp=2*2`` torchrun both resolve the correct cp peers
    (this test only consumes ``cp`` / ``my_cp_rank`` / ``cp_pe_table``, none of the
    shard-tensor-dim bookkeeping, so the dummy Shard dims are immaterial).
    """
    mesh = dist_manager.device_mesh
    placements = [Shard(i) for i in range(mesh.ndim)]
    return PeMap.from_mesh_placements(mesh, placements, distributed_manager=dist_manager)


def _session_cp_mesh(dist_manager):
    """The cp DeviceMesh whose ndim reflects the cp-axis structure of the session.

    Under ``CPO_DIST_MESH=cp=2*2`` the manager's flat ``device_mesh`` is the 1-D
    ``(4,)`` rank tensor, but its ``device_mesh_subgroups`` is the 2-D ``(2,2)`` grid —
    that 2-D mesh is what makes the back store's tile->peer UNRAVEL exercise BOTH cp
    axes (1-D ``cp=4`` would flatten to a single division). So the sharded-API tests
    drive the SUBGROUP mesh when present (mirrors the conftest ``device_mesh`` fixture's
    own subgroup preference), else the flat mesh (a true 1-D ``cp=N`` session).
    """
    sub = getattr(dist_manager, "device_mesh_subgroups", None)
    if getattr(dist_manager, "has_subgroups", False) and sub is not None:
        return sub
    return dist_manager.device_mesh


def _trimul_placements(dist_manager):
    """TriMul token-shard placements over the cp mesh: mesh dim i -> Shard(i+1).

    The generic ``configure_a2a_sharded`` entry takes a (DeviceMesh, placements) and
    rejects a feature-dim-3 shard via ``validate_trimul_sharding`` (which requires a
    Shard on a TOKEN tensor-dim 1 or 2). The bare ``[Shard(i)]`` of :func:`_pe_map`
    shards tensor dim 0 (batch) for a 1-D mesh, which that validator rejects — so for
    the sharded-API tests we map mesh dim ``i`` to TOKEN tensor-dim ``i + 1``:

      * 1-D ``cp`` mesh  -> ``[Shard(1)]``        (i-axis sharded, j full)  -- the 1-D case.
      * 2-D ``cp0×cp1``  -> ``[Shard(1), Shard(2)]`` (i over cp0, j over cp1) -- the 2-D case.

    Uses the SUBGROUP cp mesh (:func:`_session_cp_mesh`) so a ``cp=2*2`` session is seen
    as the 2-D ``(2,2)`` grid. The flatten (``cp`` / ``my_cp_rank`` / ``cp_pe_table``) is
    the row-major flatten of those axes (matches :func:`_pe_map`); ``cp_shard_tensor_dims``
    now carries the token dims. Returns ``(placements, pe_map)``.
    """
    mesh = _session_cp_mesh(dist_manager)
    if mesh.ndim > 2:
        rank_invariant_skip(
            f"sharded-API tests cover 1-D/2-D token shards; mesh ndim={mesh.ndim}",
            because=_GEOMETRY_UNIFORM_BECAUSE,
        )
    placements = [Shard(i + 1) for i in range(mesh.ndim)]
    pm = PeMap.from_mesh_placements(mesh, placements, distributed_manager=dist_manager)
    return placements, pm


def _cp_axis_sizes(dist_manager):
    """The per-cp-axis sizes for the cp mesh (e.g. (2,) for 1-D, (2,2) for a 2-D shard)."""
    mesh = _session_cp_mesh(dist_manager)
    return tuple(int(mesh.size(i)) for i in range(mesh.ndim))


# --------------------------------------------------------------------------- #
# BACK GEMM-native (design-E) — 5-D recv store.  Lifted from
# tests/distributed/_t32_gemm_native_back_store_smoke.py (the CURRENT post-refactor
# API: configure_a2a_gemm_native; recv on epi_args.recv; logical (M,N,L) mD).
# --------------------------------------------------------------------------- #


def _compile_back_gemm_native(
    A_t,
    B_t,
    recv_t,
    tile_shape_mn,
    *,
    gemm_native_cfg,
    pingpong=False,
    configure_fn=None,
    ring_t=None,
    pe_table_dev_t=None,
    role_count_t=None,
    mma_time_t=None,
    cluster_shape_mnk=(1, 1, 1),
    run_j_tiles=0,
):
    """Compile ``GemmA2ASm90`` over a BATCHED (L>1) operand set with the design-E
    5-D GEMM-native store enabled.

    ``A_t`` (L,M,K), ``B_t`` (L,N,K) bf16; ``recv_t`` is the 5-D symmetric recv
    ``(cp,Dloc,B,N_loc,N)``. The recv rides on ``epi_args.recv``; ``mD`` is the
    LOGICAL (M=cp*N_loc, N, L=Dloc*B) row-major view of that same memory (drives only
    GEMM shape/scheduler — the store writes the peer atoms built from ``recv``, so the
    logical layout never reaches GMEM). Returns ``(compiled, run_args, recv_view)``.

    ``configure_fn``: optional ``callable(gemm_obj)`` that enables the store INSTEAD of
    the default 1-D ``configure_a2a_gemm_native(**gemm_native_cfg)`` — used to drive the
    generic ``configure_a2a_sharded`` entry through the SAME compile path (the recv /
    operands / mD are identical, so any store-routing delta is isolated to the config).
    """
    device_capacity = get_device_capacity()
    assert device_capacity[0] == 9, f"SM90 only; got {device_capacity}"
    a_dtype = torch2cute_dtype_map[A_t.dtype]
    L, M, K = A_t.shape
    _, N, _ = B_t.shape
    A3 = A_t.permute(1, 2, 0)  # (M,K,L) — NON-degenerate L stride (never unsqueeze)
    B3 = B_t.permute(1, 2, 0)  # (N,K,L)

    gemm_obj = GemmA2ASm90(
        Float32, a_dtype, tile_shape_mn, cluster_shape_mnk, pingpong=pingpong, is_persistent=True
    )
    if configure_fn is not None:
        configure_fn(gemm_obj)
    else:
        gemm_obj.configure_a2a_gemm_native(
            cp=gemm_native_cfg["cp"],
            my_cp_rank=gemm_native_cfg["my_cp_rank"],
            B=gemm_native_cfg["B"],
            N_loc=gemm_native_cfg["N_loc"],
            pe_table=gemm_native_cfg["pe_table"],
        )
    # mD = the GEMM's LOGICAL (M, N, L) output extent — M = full token-i (A's M),
    # N = full token-j (B's N), L = Dloc*B. This drives ONLY the GEMM shape/scheduler;
    # the store writes the peer atoms built from ``recv``, so the logical layout never
    # reaches GMEM. Derive from the OPERANDS (correct for 1-D AND 2-D token shards) —
    # NOT from recv_t.shape, whose cp*N_i_loc == cp1*N != M when BOTH token axes are
    # cp-split. Built as an as_strided view of recv (any same-device buffer of >= M*N*L
    # elems) so from_dlpack provides the MLIR Context (cute.make_layout can't run in
    # bare host Python); the view is logical-only (never stored into).
    m_logical, l_logical = M, L
    cute_recv = from_dlpack(recv_t, assumed_align=16)
    D_logical = torch.as_strided(recv_t, (m_logical, N, l_logical), (N, 1, m_logical * N))
    cute_D = from_dlpack(D_logical, assumed_align=16)

    max_active_clusters = get_max_active_clusters(cluster_shape_mnk[0] * cluster_shape_mnk[1])
    # Decoupled-store extras (§7.16j full-buffer putwarp): the local-GMEM staging ring + the (cp,)
    # device PE table ride on EpilogueArguments. None on the default coupled path -> byte-identical.
    cute_ring = from_dlpack(ring_t, assumed_align=16) if ring_t is not None else None
    cute_pe_dev = (
        from_dlpack(pe_table_dev_t, assumed_align=4) if pe_table_dev_t is not None else None
    )
    cute_rc = from_dlpack(role_count_t, assumed_align=4) if role_count_t is not None else None
    cute_mt = from_dlpack(mma_time_t, assumed_align=8) if mma_time_t is not None else None
    epi_args = GemmA2ASm90.EpilogueArguments(
        alpha=None,
        beta=None,
        mRowVecBroadcast=None,
        mColVecBroadcast=None,
        add_to_output=False,
        rounding_mode=None,
        sr_seed=None,
        recv=cute_recv,
        pe_table_dev=cute_pe_dev,
        ring=cute_ring,
        role_count=cute_rc,
        mma_time=cute_mt,
    )
    scheduler_args = make_scheduler_args(max_active_clusters, Int32(8), None, None)
    # COALESCE (§0.9.6): bake run_j_tiles into the scheduler so each CTA walks a full i-band's j-tiles
    # contiguously (== the coalesce store's ncluster_n; MUST match _a2a_ncluster_n). The bitcode route
    # traces with THIS scheduler_args (cute.compile, no tvm-ffi ABI erasure -- gemm_bitcode_compile.py),
    # so a concrete int run_j_tiles is baked as a Constexpr; the runtime re-pass ignores it (Constexpr,
    # not a marshaled FFI arg). run_j_tiles=0 (default) -> None-normalized -> byte-identical.
    if run_j_tiles:
        scheduler_args = scheduler_args._replace(run_j_tiles=int(run_j_tiles))
    cute_A = from_dlpack(A3, assumed_align=16)
    cute_B = from_dlpack(B3, assumed_align=16)
    stream = cutlass_torch.current_stream()
    compiled = compile_gemm_with_bitcode(
        gemm_obj,
        cute_A,
        cute_B,
        cute_D,
        None,
        epi_args,
        scheduler_args,
        stream,
        None,
        register=True,
    )
    run_args = (cute_A, cute_B, cute_D, None, epi_args, scheduler_args, stream, None)
    return compiled, run_args


def _ref_back_gemm_native(A, Bt, pm, *, cp, Dloc, B, N_loc, N, world_size):
    """fp32 batched-GEMM-then-reshard reference for the design-E 5-D recv.

    ``tri[L] = A[L] @ B[L]^T`` (L = d*B+b). The store routes token-i block
    (i//N_loc) -> peer; on MY recv, slot ``s`` holds rank ``s``'s tri for the token
    block I own: ``recv[s,d,b,i_local,j] = peer_s.tri[d*B+b][my_cp_rank*N_loc+i_local, j]``.
    """
    import torch.distributed as dist

    tri = torch.bmm(A.float(), Bt.float().transpose(-1, -2)).to(torch.bfloat16)  # (L,M,N)
    gathered = [torch.empty_like(tri) for _ in range(world_size)]
    dist.all_gather(gathered, tri.contiguous())
    expected = torch.empty((cp, Dloc, B, N_loc, N), device=A.device, dtype=torch.bfloat16)
    for s in range(cp):
        peer_global = int(pm.cp_pe_table[s].item())
        src = gathered[peer_global]  # rank s's tri (L,M,N)
        for d in range(Dloc):
            for b in range(B):
                Lp = d * B + b
                expected[s, d, b, :, :] = src[
                    Lp, pm.my_cp_rank * N_loc : (pm.my_cp_rank + 1) * N_loc, :
                ]
    return expected


def _ref_back_gemm_native_2d(A, Bt, pm, *, cp0, cp1, Dloc, B, N_i_loc, N_j_loc, N, world_size):
    """fp32 batched-GEMM-then-2D-reshard reference for the design-E 5-D recv (2-D shard).

    ``tri[L] = A[L] @ B[L]^T`` over the FULL (N,N) token grid (each rank's D-slice). The
    back A2A routes token ``(i, j)`` to the peer owning it under ``(Shard(1),Shard(2))``:
    ``cp0_coord = i // N_i_loc``, ``cp1_coord = j // N_j_loc``, ``peer = cp0_coord*cp1 +
    cp1_coord`` (row-major). On MY recv, slot ``s`` holds peer ``s``'s tri restricted to
    MY 2-D token block (MY mesh coord ``(r0, r1) = unravel(my_cp_rank)``):
    ``recv[s,d,b,i_local,j_local] = peer_s.tri[d*B+b][r0*N_i_loc + i_local, r1*N_j_loc +
    j_local]``. Returns the 5-D ``(cp, Dloc, B, N_i_loc, N_j_loc)`` recv reference.
    """
    import torch.distributed as dist

    cp = cp0 * cp1
    tri = torch.bmm(A.float(), Bt.float().transpose(-1, -2)).to(torch.bfloat16)  # (L,N,N)
    gathered = [torch.empty_like(tri) for _ in range(world_size)]
    dist.all_gather(gathered, tri.contiguous())
    # MY mesh coord on the (cp0, cp1) grid (row-major unravel of my flat cp rank).
    r0 = pm.my_cp_rank // cp1
    r1 = pm.my_cp_rank % cp1
    i0, j0 = r0 * N_i_loc, r1 * N_j_loc
    expected = torch.empty((cp, Dloc, B, N_i_loc, N_j_loc), device=A.device, dtype=torch.bfloat16)
    for s in range(cp):
        peer_global = int(pm.cp_pe_table[s].item())
        src = gathered[peer_global]  # peer s's tri (L,N,N)
        for d in range(Dloc):
            for b in range(B):
                Lp = d * B + b
                expected[s, d, b, :, :] = src[Lp, i0 : i0 + N_i_loc, j0 : j0 + N_j_loc]
    return expected


# --------------------------------------------------------------------------- #
# BACK plain (L=1) — token-major recv store.  Lifted from
# tests/distributed/_t22_gemm_a2a_smoke.py (configure_a2a; recv on epi_args.recv).
# --------------------------------------------------------------------------- #


# --------------------------------------------------------------------------- #
# Tests.
# --------------------------------------------------------------------------- #


# (N, D) parametrizations. D = cp*Dloc must be divisible by cp; N_loc = N//cp must be
# a multiple of the CTA tile-M (128). N=256 -> N_loc=128 (cp=2) / 64-? — keep N a
# multiple of cp*128 so N_loc % 128 == 0 at both cp=2 and cp=4 (N=512 -> N_loc 256/128).
#
# THIS LIST IS NOW LIVE. It was a DEAD constant -- defined here and referenced by nothing, while the
# decorator below inlined the same literal -- so widening the pool would have edited the copy that
# does not run. Wired to the decorator in the same change that widened it, which is the only order
# in which the two cannot disagree. (`_BACK_PUTWARP_ND` at the rotating-ring test was dead the
# same way and is now wired too -- see its comment for why it keeps its narrow widths.)
#
# Each pair earns its place, under `D % cp == 0` AND `(N // cp) % 128 == 0` -- both enforced by
# rank_invariant_skips in the body, so an invalid pair costs a skip rather than a failure:
#   (256,   4)  L=1 at cp=2 -- the `l_is_one` facet, and the un-bake's degenerate case.
#   (512,   8)  L>1 batched at cp=2 -- the case `d = L//B` / `b = L%B` has to separate.
#   (512,  96)  the `multi_node_width` facet. It is here because that facet was MEASURED unreached:
#               `coverage_problems` reported D emitting only [4, 8], so 96 was declared by the axis
#               and swept by nothing. 96 % cp == 0 at every cp in {2,4,8}, so nothing but the
#               absence of a cell was keeping it out.
#   (512, 128) (512, 256) (512, 384) (512, 512)
#               four target TriMul feature widths, at the largest N that still gives
#               N_loc % 128 == 0 at BOTH cp=2 and cp=4. Verified running before being declared:
#               D=128 and D=512 at cp=2 passed the element-wise gate (worst ratio 0.055/0.109 and
#               0.053/0.056 against a threshold of 1.0) with `untouched == 0`, at 1.28 s / 0.65 s.
#   (1024, 128) (1024, 256) (1024, 384) (1024, 512)
#               cp=8-reachable workflow cells -- ALL FOUR of them. At N=512 EVERY cell skips at cp=8, because
#               512/8 = 64 is not a multiple of 128 -- so the binding constraint is the TOKEN
#               extent, not the feature width, and no D value alone can close it.
#
#               (1024, 512) came first and closed it for one width. (1024, 256) and (1024, 384)
#               followed after a runtime sweep MEASURED the hole still open for those: at ws=8 only
#               5 of 208 D-carrying cells executed, and D=256/D=384 skipped on
#               `N_loc=N/cp=64 not a multiple of cta_tile_M=128 at cp=8` -- the reason string, not
#               an inference. Before that, D=256 and D=384 each ran in exactly THREE cells, all
#               from this one test at N=512, at cp in {2, 4, (2,2)}.
#
#               (1024, 128) closed the LAST one, and it is the entry most easily mistaken for
#               redundant. D=128 already executed at 8 ranks -- but ONLY through
#               `test_back_gemm_native_2d_store_correct`, which divides by `cp0` rather than `cp`,
#               so `512/2 = 256` clears the tile there. That is a DIFFERENT store path on a
#               FACTORED mesh. On THIS test -- the 1-D store, dividing by the full cp -- D=128 had
#               never executed at cp=8 at all, and never at a flat `cp=8` on any path. A summary
#               that said "all four widths reach cp=8" was true only on the loosest reading; this
#               cell is what makes it true on the strict one.
#
#               CEILING, so the next reader does not rediscover it: cp=16 needs `N >= 2048`
#               (2048/16 = 128) AND a 16-rank launch. Neither is here, so the factored 16-rank
#               specs still skip on the same gate. Raising N further is the ONLY thing that moves
#               it -- widening D cannot, for the reason above.
_BACK_GEMM_NATIVE = [
    (256, 4),
    (512, 8),
    (512, 96),
    (512, 128),
    (512, 256),
    (512, 384),
    (512, 512),
    (1024, 128),
    (1024, 256),
    (1024, 384),
    (1024, 512),
]

#: (N, D) pairs the BATCH axis is widened over. B is orthogonal to the token/feature geometry --
#: it multiplies the GEMM's L axis and touches nothing else -- so sweeping it against every pair
#: would triple the cell count to re-measure an independence the store's own structure guarantees.
#: One narrow-D and one wide-D pair keep both `Dloc` regimes represented at B>1.
_BACK_GEMM_NATIVE_BATCHED = [(512, 128), (1024, 512)]


@pytest.mark.usefixtures("_nvshmem__distributed__a2a_fusion")
@pytest.mark.usefixtures("_module_skip__distributed__a2a_fusion")
@EPI.parametrize(
    "N",
    "D",
    "B",
    "mesh",
    cells=[
        (n, d, b, m)
        for (n, d) in _BACK_GEMM_NATIVE
        for b in ((1, 2, 3) if (n, d) in _BACK_GEMM_NATIVE_BATCHED else (1,))
        for m in EPI.axis("mesh").values
        if not _misaligned_2d(n, m)
    ],
    because=(
        "N and D are COUPLED here, not a product: D must be divisible by cp and N_loc=N/cp a "
        "multiple of the 128 CTA tile, so the valid pairs are a short list rather than a grid. "
        "The pairs are `_BACK_GEMM_NATIVE`, whose own comment gives each one's reason. B is "
        "widened only over `_BACK_GEMM_NATIVE_BATCHED` -- see that list's comment for why the "
        "batch axis does not need the full product. " + _JOINT_MESH_BECAUSE
    ),
)
def test_back_gemm_native_store_correct(
    apply_mesh, mesh, dist_manager, device, world_size, N, D, B
):
    """Design-E 5-D back store: coverage + parent rel-err vs fp32 batched-GEMM-reshard.

    B > 1 is the cell that exercises the in-kernel batch decode ``d = L // B`` / ``b = L % B``
    (`gemm_sm90_a2a.py:3985`) on the COUPLED default store -- the path production NVLink runs, and
    the one the `cluster_drain` suite's own B=2 cells do NOT reach (they drive the cluster producer
    at `:3069` instead).
    """
    apply_mesh(mesh)

    cp = world_size
    # Geometry constraints (skip rather than fail so a cp=4 torchrun collects the SAME
    # ids and runs the cells whose (N,D) are valid at that cp): D=cp*Dloc and the CTA
    # M-tile (128) must divide the per-peer token block N_loc=N/cp.
    if D % cp != 0:
        rank_invariant_skip(f"D={D} not divisible by cp={cp}", because=_GEOMETRY_UNIFORM_BECAUSE)
    if (N // cp) % 128 != 0:
        rank_invariant_skip(
            f"N_loc=N/cp={N // cp} not a multiple of cta_tile_M=128 at cp={cp}",
            because=_GEOMETRY_UNIFORM_BECAUSE,
        )
    Dloc = D // cp
    B = int(B)  # the declared batch axis; L = Dloc*B is the GEMM's batch extent
    N_loc = N // cp
    M = N  # GEMM M = full token-i = cp*N_loc
    K = 256
    L = Dloc * B
    pm = _pe_map(dist_manager)
    _skip_coupled_cross_node(pm, "back gemm-native store")

    torch.manual_seed(4321 + dist_manager.rank)
    A = torch.randn(L, M, K, device=device, dtype=torch.bfloat16) / (K**0.5)
    Bt = torch.randn(L, N, K, device=device, dtype=torch.bfloat16) / (K**0.5)
    with torch.cuda.use_mem_pool(DistributedManager.symmetric_mempool(device)):
        recv = torch.empty((cp, Dloc, B, N_loc, N), dtype=torch.bfloat16, device=device)
    recv.fill_(-99.0)  # sentinel for the coverage stamp

    pe_table = tuple(int(x) for x in pm.cp_pe_table.tolist())
    cfg = dict(cp=cp, my_cp_rank=int(pm.my_cp_rank), B=B, N_loc=N_loc, pe_table=pe_table)
    compiled, run_args = _compile_back_gemm_native(A, Bt, recv, (128, 128), gemm_native_cfg=cfg)
    try:
        _nvshmem_barrier()
        compiled(*run_args)
        _drain()

        expected = _ref_back_gemm_native(
            A, Bt, pm, cp=cp, Dloc=Dloc, B=B, N_loc=N_loc, N=N, world_size=world_size
        )
        # Coverage is ASSERTED, not printed. The upstream left it a diagnostic and gave its reason
        # -- an unwritten cell keeps the O(1) -99 sentinel and so blows the pooled rel-err anyway --
        # but that reasoning is downstream of the POOLED metric, which this tree does not use. The
        # element-wise gate does catch an unwritten cell at its own index, so the two agree today;
        # asserting it keeps them agreeing if the reference ever grows a masked region.
        untouched = int((recv == -99.0).all(dim=4).sum().item())
        n_cells = cp * Dloc * B * N_loc
        tag = f"back-native cp={cp} N={N} D={D} Dloc={Dloc} N_loc={N_loc}"
        # Correctness gate: the parent kernels' bar, applied per element.
        worst = _assert_recv_matches(recv, expected, bar=REL_BAR__distributed__a2a_fusion, what=tag)
        assert untouched == 0, (
            f"{tag}: {untouched}/{n_cells} recv cells still hold the -99 sentinel, so the store "
            "never reached them"
        )
        print(
            f"\n[{tag}] worst_ratio={worst:.3f} of its per-element bound (FAILS at 1.0; "
            f"bound = {REL_BAR__distributed__a2a_fusion}*max|ref|) "
            f"untouched={untouched}/{n_cells} PASS",
            flush=True,
        )
    finally:
        compiled.free()
        _nvshmem_barrier()


# --------------------------------------------------------------------------- #
# Shared (N, D) size grid for the decoupled putwarp-drain test(s).
#
# WIRED, and RENAMED off a kwarg that no longer exists. This was the SECOND dead constant in this
# file -- defined, referenced by nothing, while the decorator below inlined the same literal -- so
# the obvious edit (widen the constant) would have changed the copy that does not run and produced
# no new cells while looking like a working change. It is now what the decorator reads.
#
# The rename is not tidying. `full_buffer` is a REMOVED kwarg: `test_removed_hybrid_drain_kwargs_
# rejected` in this very file asserts that passing it raises TypeError, "the 'never existed'
# behavior". So wiring a list named `_BACK_FULLBUF` into a test whose own body comment reads "NO
# full_buffer" would have swapped one landmine for a subtler one -- a reader would take these pairs
# to be the grid for a mode the API refuses.
#
# NOT shared with the cubin test, which inlines the same two pairs for its OWN reason (its `because=`
# argues the CONFIG pool is its axis and the shape pool is not). One list behind two tests whose
# pairs agree by coincidence is how widening one silently widens the other.
#
# Same coupling as `_BACK_GEMM_NATIVE` above (`D % cp == 0`, `(N // cp) % 128 == 0`). NOT widened,
# deliberately: this test's subject is the ring-reuse handshake at a small bounded `ring_depth`, and
# the feature width is a passenger there. The full D sweep lives on the coupled-store test, where
# D is the axis under test rather than a passenger.
# --------------------------------------------------------------------------- #
_BACK_PUTWARP_ND = [(256, 4), (512, 8)]


# --------------------------------------------------------------------------- #
# Phase-3 ROTATING-RING putwarp drain — the parked IB-precursor seed: a SMALL
# bounded ring_depth (the producer<->consumer empty[s] reuse handshake RE-ENABLED). The
# slot reuse must NOT corrupt: same ERROR-HISTOGRAM (rel_L2 < 2e-2, n_outlier_rows == 0)
# + scatter coverage (untouched == 0) gate, at a couple of small rds.
# --------------------------------------------------------------------------- #
@pytest.mark.usefixtures("_nvshmem__distributed__a2a_fusion")
@pytest.mark.usefixtures("_module_skip__distributed__a2a_fusion")
@pytest.mark.parametrize("rd", [4, 16], ids=["rd4", "rd16"])
@pytest.mark.parametrize("cwg", [2], ids=["cwg2"])
@EPI.parametrize(
    "N",
    "D",
    "mesh",
    cells=[
        (n, d, m)
        for (n, d) in _BACK_PUTWARP_ND
        for m in EPI.axis("mesh").values
        if not _misaligned_2d(n, m)
    ],
    because=(
        "same coupling as the coupled-store test above -- D%cp==0 and N_loc%128==0 admit a "
        "short list, not a product; the pairs are `_BACK_PUTWARP_ND`, whose comment says why "
        "they keep their narrow widths. The ring depth and consumer-warpgroup count are swept on "
        "top as ordinary parameters: they are drain knobs rather than shape, so no matrix axis "
        "describes them. " + _JOINT_MESH_BECAUSE
    ),
)
def test_back_gemm_native_rotating_ring_putwarp(
    apply_mesh, mesh, dist_manager, device, world_size, N, D, cwg, rd
):
    """Rotating-ring (bounded ring_depth) putwarp drain with the reuse handshake ACTIVE."""
    apply_mesh(mesh)
    from tests.distributed.correctness_harness import compute_error_histogram

    cp = world_size
    if D % cp != 0:
        rank_invariant_skip(f"D={D} not divisible by cp={cp}", because=_GEOMETRY_UNIFORM_BECAUSE)
    if (N // cp) % 128 != 0:
        rank_invariant_skip(
            f"N_loc=N/cp={N // cp} not a multiple of cta_tile_M=128 at cp={cp}",
            because=_GEOMETRY_UNIFORM_BECAUSE,
        )
    Dloc = D // cp
    B = 1
    N_loc = N // cp
    M = N
    K = 256
    L = Dloc * B
    tile_shape_mn = (128, 128)
    pm = _pe_map(dist_manager)

    torch.manual_seed(4321 + dist_manager.rank)
    A = torch.randn(L, M, K, device=device, dtype=torch.bfloat16) / (K**0.5)
    Bt = torch.randn(L, N, K, device=device, dtype=torch.bfloat16) / (K**0.5)
    with torch.cuda.use_mem_pool(DistributedManager.symmetric_mempool(device)):
        recv = torch.empty((cp, Dloc, B, N_loc, N), dtype=torch.bfloat16, device=device)
    recv.fill_(-99.0)

    pe_table = tuple(int(x) for x in pm.cp_pe_table.tolist())
    max_active_clusters = get_max_active_clusters(1)

    # SMALL rotating ring: ring_depth=rd, NO full_buffer (the reuse handshake is live: the producer
    # waits empty[s] before reusing a slot the consumer has drained).
    def _cfg(g):
        g.configure_a2a_gemm_native(
            cp=cp,
            my_cp_rank=int(pm.my_cp_rank),
            B=B,
            N_loc=N_loc,
            pe_table=pe_table,
            decoupled=True,
            producer_tma=True,
            consumer_strided=True,
            consumer_strided_putwarp=True,
            ring_depth=rd,
            consumer_warpgroups=cwg,
        )

    grid_ctas = max_active_clusters
    # ring_t is the drain's put SOURCE; a cross-node IBGDA put DMA-reads it, and the NIC lkey
    # resolves ONLY symmetric-heap addresses (gemm_sm90_a2a.py:708) -> allocate it on the symmetric
    # heap, not plain torch (a plain source aborts device-side in ibgda_get_lkey, assert 0).
    with torch.cuda.use_mem_pool(DistributedManager.symmetric_mempool(device)):
        ring_t = torch.empty(
            (grid_ctas, rd, 128, tile_shape_mn[1]), dtype=torch.bfloat16, device=device
        )
    ring_t.fill_(0.0)
    pe_dev = pm.cp_pe_table.to(torch.int32).contiguous()

    compiled, run_args = _compile_back_gemm_native(
        A,
        Bt,
        recv,
        tile_shape_mn,
        gemm_native_cfg=None,
        configure_fn=_cfg,
        ring_t=ring_t,
        pe_table_dev_t=pe_dev,
    )
    try:
        _nvshmem_barrier()
        compiled(*run_args)
        _drain()
        expected = _ref_back_gemm_native(
            A, Bt, pm, cp=cp, Dloc=Dloc, B=B, N_loc=N_loc, N=N, world_size=world_size
        )
        got4d = recv.reshape(cp, Dloc * B, N_loc, N)
        exp4d = expected.reshape(cp, Dloc * B, N_loc, N)
        h = compute_error_histogram(got4d, exp4d)
        untouched = int((recv == -99.0).all(dim=4).sum().item())
        n_cells = cp * Dloc * B * N_loc
        tag = f"rotating-ring cp={cp} N={N} D={D} Dloc={Dloc} N_loc={N_loc} cwg={cwg} rd={rd}"
        print(
            f"\n[{tag}] rel_L2={h.rel_l2:.3e} "
            f"row_max/med={h.row_outlier_ratio:.1f}x outlier_rows={h.n_outlier_rows} "
            f"untouched={untouched}/{n_cells} (gate below)",
            flush=True,
        )
        _assert_recv_matches(got4d, exp4d, bar=2e-2, what=tag)
        assert h.n_outlier_rows == 0, (
            f"{tag}: {h.n_outlier_rows} outlier row(s) (ratio {h.row_outlier_ratio:.1f}x, "
            f"worst@{h.worst_row_index}); untouched={untouched}/{n_cells}"
        )
        assert untouched == 0, f"{tag}: {untouched}/{n_cells} recv cells unwritten (coverage gap)"
    finally:
        compiled.free()
        _nvshmem_barrier()


# ---- 2-D token-shard back store (cp = cp0 × cp1). N must split on BOTH axes so a CTA
# tile maps to ONE peer: N//cp0 % tile_M == 0 AND N//cp1 % tile_N == 0. With (cp0,cp1)=
# (2,2) + tile (128,128): N=256 -> N_i_loc=N_j_loc=128 (1 tile/block each axis); N=512 ->
# 256 (2 tiles/block, multi-tile peer block per axis = the non-trivial unravel regime).
_BACK_2D = [(256, 8), (512, 8)]


@pytest.mark.usefixtures("_nvshmem__distributed__a2a_fusion")
@pytest.mark.usefixtures("_module_skip__distributed__a2a_fusion")
@EPI.parametrize(
    "N",
    "D",
    "mesh",
    cells=[
        (n, d, m)
        for (n, d) in ((256, 8), (512, 8), (512, 128), (512, 512))
        for m in EPI.axis("mesh").values
        if not _misaligned_2d(n, m)
    ],
    because=(
        "the 2-D shard needs N divisible by BOTH cp axes and D by their product, so D=4 drops "
        "out at cp0*cp1>=4 and the pair list is shorter still than the 1-D one. TWO workflow "
        "widths rather than all four: the subject here is the tile->peer UNRAVEL, which reads the "
        "token coordinate and not the feature width, so D enters only through Dloc = "
        "D/(cp0*cp1) and the narrowest and widest bracket what the unravel can see. The full "
        "the full D sweep lives on the 1-D store test, where D is the axis under test rather than "
        "a passenger. " + _JOINT_MESH_BECAUSE
    ),
)
def test_back_gemm_native_2d_store_correct(
    apply_mesh, mesh, dist_manager, device, world_size, N, D
):
    """2-D token shard (Shard(1),Shard(2)) back store: the tile->peer UNRAVEL.

    Validates the design-E 5-D store under a 2-D ``(cp0, cp1)`` cp mesh: a GEMM ``(i, j)``
    tile routes to peer ``(i//N_i_loc)*cp1 + (j//N_j_loc)`` (row-major flatten of the 2-D
    cp coordinate). Gated against the fp32 batched-GEMM-then-2D-reshard oracle with the
    ERROR-HISTOGRAM (rel_L2 < bar AND n_outlier_rows == 0 — a localized i==0/j==0
    corruption that a scalar rel_L2 averages away) + scatter coverage (untouched == 0).
    Skips unless the session mesh is 2-D (run with CPO_DIST_MESH=cp=2*2, 4 GPUs).
    """
    apply_mesh(mesh)
    from tests.distributed.correctness_harness import compute_error_histogram

    axis_sizes = _cp_axis_sizes(dist_manager)
    if len(axis_sizes) != 2:
        rank_invariant_skip(
            f"2-D back test needs a 2-D cp mesh; session mesh is {axis_sizes}",
            because=_GEOMETRY_UNIFORM_BECAUSE,
        )
    cp0, cp1 = axis_sizes
    cp = cp0 * cp1
    if D % cp != 0:
        rank_invariant_skip(f"D={D} not divisible by cp={cp}", because=_GEOMETRY_UNIFORM_BECAUSE)
    if N % cp0 != 0 or N % cp1 != 0:
        rank_invariant_skip(
            f"N={N} not divisible by both cp axes {axis_sizes}", because=_GEOMETRY_UNIFORM_BECAUSE
        )
    N_i_loc, N_j_loc = N // cp0, N // cp1
    if N_i_loc % 128 != 0 or N_j_loc % 128 != 0:
        rank_invariant_skip(
            f"N_i_loc={N_i_loc}/N_j_loc={N_j_loc} not multiples of tile 128 at {axis_sizes}",
            because=_GEOMETRY_UNIFORM_BECAUSE,
        )
    Dloc, B, K = D // cp, 1, 256
    L = Dloc * B
    placements, pm = _trimul_placements(dist_manager)
    _skip_coupled_cross_node(pm, "back gemm-native 2-D store")
    mesh = _session_cp_mesh(dist_manager)

    torch.manual_seed(4321 + dist_manager.rank)
    A = torch.randn(L, N, K, device=device, dtype=torch.bfloat16) / (K**0.5)  # (L, i=N, K)
    Bt = torch.randn(L, N, K, device=device, dtype=torch.bfloat16) / (K**0.5)  # (L, j=N, K)
    with torch.cuda.use_mem_pool(DistributedManager.symmetric_mempool(device)):
        recv = torch.empty((cp, Dloc, B, N_i_loc, N_j_loc), dtype=torch.bfloat16, device=device)
    recv.fill_(-99.0)

    fn = lambda g: g.configure_a2a_sharded(mesh, placements, pe_map=pm, B=B, N=N, gemm_native=True)
    compiled, run_args = _compile_back_gemm_native(
        A, Bt, recv, (128, 128), gemm_native_cfg=None, configure_fn=fn
    )
    try:
        _nvshmem_barrier()
        compiled(*run_args)
        _drain()

        expected = _ref_back_gemm_native_2d(
            A,
            Bt,
            pm,
            cp0=cp0,
            cp1=cp1,
            Dloc=Dloc,
            B=B,
            N_i_loc=N_i_loc,
            N_j_loc=N_j_loc,
            N=N,
            world_size=world_size,
        )
        # Reshape recv [slot=cp, d=Dloc, b=B, i_local, j_local] and ref to (B, N_i, N_j, D)
        # with feature order (slot, d) == (cp, Dloc) -> the error-histogram per-row L2 reduces
        # over D, so a single corrupted token row (i==0/j==0 collapse) shows as an outlier.
        got_bijD = recv.permute(2, 3, 4, 0, 1).reshape(B, N_i_loc, N_j_loc, cp * Dloc)
        ref_bijD = expected.permute(2, 3, 4, 0, 1).reshape(B, N_i_loc, N_j_loc, cp * Dloc)
        h = compute_error_histogram(got_bijD, ref_bijD)
        untouched = int((recv == -99.0).all(dim=4).sum().item())
        n_cells = cp * Dloc * B * N_i_loc * N_j_loc
        tag = f"back-2D cp={cp0}x{cp1} N={N} D={D} N_i_loc={N_i_loc} N_j_loc={N_j_loc}"
        print(
            f"\n[{tag}] rank={dist_manager.rank} {h.summary()} untouched={untouched}/{n_cells} "
            f"(gate below)",
            flush=True,
        )
        _assert_recv_matches(got_bijD, ref_bijD, bar=REL_BAR__distributed__a2a_fusion, what=tag)
        assert h.n_outlier_rows == 0, (
            f"{tag}: {h.n_outlier_rows} outlier token rows (localized corruption); worst @ "
            f"{h.worst_row_index} ratio {h.row_outlier_ratio:.1f}x — 2-D unravel mis-routed a tile"
        )
        assert untouched == 0, f"{tag}: {untouched}/{n_cells} recv cells unwritten (coverage gap)"
    finally:
        compiled.free()
        _nvshmem_barrier()


@pytest.mark.usefixtures("_nvshmem__distributed__a2a_fusion")
@pytest.mark.usefixtures("_module_skip__distributed__a2a_fusion")
@matrix_exempt(
    "the subject is that a2a-OFF is BIT-IDENTICAL to the plain local kernel, at one fixed (M, N, K). It compares two kernels against each other rather than an output against a reference, so no shape axis changes what is being asserted -- and sweeping N would multiply the runtime of a claim that is about the code path, not the shape"
)
def test_flag_off_byte_identical(dist_manager, device, world_size):
    """§3.1 guarantee: a flag-OFF GemmA2ASm90 == the local GemmDefaultSm90, BIT-for-bit.

    No A2A config -> ``_a2a_enabled`` stays False -> ``build_D_copy_fn`` falls back to
    ``super()``. Compiled through the SAME bitcode route (no nvshmem op issued, so
    ``register=False``); compared against the local default-epilogue kernel compiled
    identically. Runs independently on every rank (no collectives). bit-exact (==),
    not a tolerance — the kernels must emit the same PTX store.
    """
    # `kernels/gemm_default_epi` is the UPSTREAM path; this tree split that module in two -- the
    # mixin to `_internal/epi_default.py`, the concrete class to `kernels/gemm.py`. Same class, same
    # bases (`GemmDefaultEpiMixin, GemmSm90`); only the module moved. Every other site in this tree
    # already imports it from here (gemm_bitcode_compile.py, test_gemm.py, test_gemm_sm90.py).
    from fold_cp_ops.kernels.gemm import GemmDefaultSm90

    torch.manual_seed(1234 + dist_manager.rank)
    M, N, K = 256, 256, 256
    tile_shape_mn = (128, 128)
    A = torch.randn(M, K, device=device, dtype=torch.bfloat16) / (K**0.5)
    B = torch.randn(N, K, device=device, dtype=torch.bfloat16) / (K**0.5)
    A3 = A.permute(1, 0).unsqueeze(-1).permute(1, 0, 2)  # (M,K,1) k-major

    def _compile_and_run(GemmCls):
        a_dtype = torch2cute_dtype_map[A.dtype]
        D = torch.zeros(M, N, device=device, dtype=torch.bfloat16)
        cA = from_dlpack(A.unsqueeze(-1), assumed_align=16)
        cB = from_dlpack(B.unsqueeze(-1), assumed_align=16)
        cD = from_dlpack(D.unsqueeze(-1), assumed_align=16)
        gemm_obj = GemmCls(
            Float32, a_dtype, tile_shape_mn, (1, 1, 1), pingpong=False, is_persistent=True
        )
        epi_args = GemmCls.EpilogueArguments(
            alpha=None,
            beta=None,
            mRowVecBroadcast=None,
            mColVecBroadcast=None,
            add_to_output=False,
            rounding_mode=None,
            sr_seed=None,
        )
        max_active_clusters = get_max_active_clusters(1)
        scheduler_args = make_scheduler_args(max_active_clusters, Int32(8), None, None)
        stream = cutlass_torch.current_stream()
        # Flag-off GemmA2ASm90 issues NO nvshmem op -> register=False (still bitcode-linked).
        compiled = compile_gemm_with_bitcode(
            gemm_obj,
            cA,
            cB,
            cD,
            None,
            epi_args,
            scheduler_args,
            stream,
            None,
            register=False,
        )
        compiled(
            from_dlpack(A.unsqueeze(-1), assumed_align=16),
            from_dlpack(B.unsqueeze(-1), assumed_align=16),
            from_dlpack(D.unsqueeze(-1), assumed_align=16),
            None,
            epi_args,
            scheduler_args,
            stream,
            None,
        )
        torch.cuda.synchronize()
        compiled.free()
        return D

    D_local = _compile_and_run(GemmDefaultSm90)
    D_a2a_off = _compile_and_run(GemmA2ASm90)  # NO configure_a2a -> flag OFF
    n_diff = int((D_local != D_a2a_off).sum().item())
    max_abs = (D_local.float() - D_a2a_off.float()).abs().max().item()
    print(
        f"\n[flag-off byte-identity] rank={dist_manager.rank} M={M} N={N} K={K} "
        f"n_diff={n_diff}/{M * N} max_abs={max_abs:.3e} "
        f"{'IDENTICAL' if n_diff == 0 else 'MISMATCH'}",
        flush=True,
    )
    # BIT patterns, not values. The upstream compared with `!=`, which reports -0.0 and 0.0 as
    # equal -- and a sign-of-zero flip is precisely the kind of thing a store path can lose while
    # every value still matches. On a test whose whole subject is byte-identity, the weaker
    # comparison is the one defect it cannot see.
    assert_bitwise(D_a2a_off, D_local, what="flag-OFF GemmA2ASm90 vs local GemmDefaultSm90")


# --------------------------------------------------------------------------- #
# Rung 1 — the generic ``configure_a2a_sharded`` entry on a 1-D mesh must reproduce
# the 1-D ``configure_a2a*`` store BIT-FOR-BIT (the cp_axis_sizes=(cp,) special case).
# Runs each of the three stores TWICE on the same inputs — once via the legacy 1-D
# entry, once via the (DeviceMesh, placements) sharded entry — into two symmetric recvs
# and asserts ``torch.equal``. 1-D mesh only (the 1-D entries don't exist for 2-D);
# pytest.skip on a 2-D session mesh so a cp=2x2 torchrun collects the SAME id.
# --------------------------------------------------------------------------- #


@pytest.mark.usefixtures("_nvshmem__distributed__a2a_fusion")
@pytest.mark.usefixtures("_module_skip__distributed__a2a_fusion")
@EPI.parametrize("mesh")
def test_configure_a2a_sharded_1d_byte_identical(
    apply_mesh, mesh, dist_manager, device, world_size
):
    """``configure_a2a_sharded`` on a 1-D mesh == the 1-D ``configure_a2a*`` recv, bit-exact."""
    apply_mesh(mesh)

    cp = world_size
    axis_sizes = _cp_axis_sizes(dist_manager)
    if len(axis_sizes) != 1:
        rank_invariant_skip(
            f"1-D byte-identity test; session mesh is {len(axis_sizes)}-D ({axis_sizes})",
            because=_GEOMETRY_UNIFORM_BECAUSE,
        )
    placements, pm = _trimul_placements(dist_manager)
    _skip_coupled_cross_node(pm, "back gemm-native store")
    mesh = _session_cp_mesh(dist_manager)
    pe_table = tuple(int(x) for x in pm.cp_pe_table.tolist())
    my = int(pm.my_cp_rank)
    K = 256
    torch.manual_seed(4321 + dist_manager.rank)

    results = {}  # tag -> (legacy_recv_clone, sharded_recv_clone)

    # ---- back-gemm-native (design-E 5-D) ----
    D = 4 * cp if (4 * cp) % cp == 0 else cp
    Dloc, Bb, N_loc, Nn = D // cp, 1, 256 // cp if (256 // cp) % 128 == 0 else 128, 256
    Nn = N_loc * cp
    L = Dloc * Bb
    A2 = torch.randn(L, Nn, K, device=device, dtype=torch.bfloat16) / (K**0.5)
    Bt2 = torch.randn(L, Nn, K, device=device, dtype=torch.bfloat16) / (K**0.5)
    _MODES = ("legacy", "sharded")
    if N_loc % 128 == 0:
        # HOISTED ALLOCATION — this is the F2 hang fix, and it is NOT cosmetic.
        # A pool-backed symmetric allocation is COLLECTIVE whenever it GROWS the segment (the retired
        # `symmetric_empty`/`nvshmem_torch.tensor` were collective on EVERY call). The previous form allocated inside
        # the per-mode loop, so mode 2's collective malloc sat BEHIND mode 1's host-blocking device
        # syncs (`_drain()` -> torch.cuda.synchronize, then `recv.clone()`) — the documented
        # interleaved-device-sync-between-collective-mallocs deadlock. Measured on the post-removal
        # tree: 3 of 10 isolated cp=8 runs wedged, with the faulthandler dump showing ONE rank inside
        # `nvshmem/core/nvshmem_types.py::allocate` while the others sat in `_drain`'s
        # `torch.cuda.synchronize` — one rank in the collective malloc, the rest past it.
        # (This also corrects the earlier localization of F2 to the stagec front block: that block is
        # gone and the hang survived it. The mechanism is the alloc/sync interleave, not the store.)
        # Both symmetric buffers are now allocated back-to-back with NO device sync between them, and
        # returned to the pool together. They are ~8 MB each at cp=8, so holding both is free.
        with torch.cuda.use_mem_pool(DistributedManager.symmetric_mempool(device)):
            recvs = [
                torch.empty((cp, Dloc, Bb, N_loc, Nn), dtype=torch.bfloat16, device=device)
                for _ in _MODES
            ]
        _nvshmem_barrier()
        compiled_all = []
        # A3: try/finally so a raise on the build/run path cannot LEAK the symmetric buffers — a
        # stray nvshmem allocation poisons the next cell on this node. All ranks build the same
        # shapes, so any raise here is rank-invariant and every rank still reaches the barrier.
        try:
            for mode, recv in zip(_MODES, recvs):
                recv.fill_(-99.0)
                if mode == "legacy":
                    cfg = dict(cp=cp, my_cp_rank=my, B=Bb, N_loc=N_loc, pe_table=pe_table)
                    compiled, run_args = _compile_back_gemm_native(
                        A2, Bt2, recv, (128, 128), gemm_native_cfg=cfg
                    )
                else:
                    fn = lambda g: g.configure_a2a_sharded(
                        mesh, placements, pe_map=pm, B=Bb, N=Nn, gemm_native=True
                    )
                    compiled, run_args = _compile_back_gemm_native(
                        A2, Bt2, recv, (128, 128), gemm_native_cfg=None, configure_fn=fn
                    )
                compiled_all.append(compiled)
                _nvshmem_barrier()
                compiled(*run_args)
                _drain()
                results.setdefault("back-native", []).append(recv.clone())
        finally:
            for c in compiled_all:
                c.free()
            _nvshmem_barrier()

    for tag, (legacy, sharded) in results.items():
        n_diff = int((legacy != sharded).sum().item())
        max_abs = (legacy.float() - sharded.float()).abs().max().item()
        print(
            f"\n[sharded-1D-identical {tag}] rank={dist_manager.rank} cp={cp} "
            f"n_diff={n_diff}/{legacy.numel()} max_abs={max_abs:.3e} "
            f"{'IDENTICAL' if n_diff == 0 else 'MISMATCH'}",
            flush=True,
        )
        # Bit patterns rather than `!=`, for the same reason as the flag-off gate above.
        assert_bitwise(
            sharded, legacy, what=f"{tag}: configure_a2a_sharded 1-D recv vs configure_a2a*"
        )


# --------------------------------------------------------------------------- #
# Per-path setup closures — build (compiled, run_callable, recv, expected) for one
# fused-store path so the determinism + autotunability tests dispatch the three
# stores uniformly (each reuses the SAME _compile_*/_ref_* helpers the correctness
# tests use; nothing about the store is re-implemented here).
# --------------------------------------------------------------------------- #


def _setup_back_gemm_native(dist_manager, device, world_size, *, N, D, K, tile_shape_mn, pingpong):
    """One back GEMM-native (design-E 5-D recv) store at (N,D,K,tile,pingpong).

    Returns ``(compiled, run, recv, untouched_fn, expected)``: ``run()`` invokes the
    kernel, ``untouched_fn()`` counts recv cells still holding the -99 sentinel (a
    full D-row of a (slot,d,b,i_local) cell unwritten), ``expected`` is the fp32
    batched-GEMM-then-reshard reference. Geometry constraints raise ``pytest.skip``.
    """

    cp = world_size
    if D % cp != 0:
        rank_invariant_skip(f"D={D} not divisible by cp={cp}", because=_GEOMETRY_UNIFORM_BECAUSE)
    if (N // cp) % tile_shape_mn[0] != 0:
        rank_invariant_skip(
            f"N_loc=N/cp={N // cp} not a multiple of tile_M={tile_shape_mn[0]} at cp={cp}",
            because=_GEOMETRY_UNIFORM_BECAUSE,
        )
    Dloc, B, N_loc, M, L = D // cp, 1, N // cp, N, (D // cp) * 1
    pm = _pe_map(dist_manager)
    torch.manual_seed(4321 + dist_manager.rank)
    A = torch.randn(L, M, K, device=device, dtype=torch.bfloat16) / (K**0.5)
    Bt = torch.randn(L, N, K, device=device, dtype=torch.bfloat16) / (K**0.5)
    with torch.cuda.use_mem_pool(DistributedManager.symmetric_mempool(device)):
        recv = torch.empty((cp, Dloc, B, N_loc, N), dtype=torch.bfloat16, device=device)
    recv.fill_(-99.0)
    pe_table = tuple(int(x) for x in pm.cp_pe_table.tolist())
    cfg = dict(cp=cp, my_cp_rank=int(pm.my_cp_rank), B=B, N_loc=N_loc, pe_table=pe_table)
    compiled, run_args = _compile_back_gemm_native(
        A, Bt, recv, tile_shape_mn, gemm_native_cfg=cfg, pingpong=pingpong
    )
    expected = _ref_back_gemm_native(
        A, Bt, pm, cp=cp, Dloc=Dloc, B=B, N_loc=N_loc, N=N, world_size=world_size
    )
    return (
        compiled,
        lambda: compiled(*run_args),
        recv,
        lambda: int((recv == -99.0).all(dim=4).sum().item()),
        expected,
    )


# --------------------------------------------------------------------------- #
# Determinism — the async-vec-race regime guard
# ([[reference_fold_cp_ops_epilogue_async_vec_race]]). Each fused store shares the
# parent's COOPERATIVE-epilogue async-vec race (>=2 async vec-loads + cooperative +
# multi-wave M/N -> non-deterministic corruption/deadlock; bit #29's back kernel:
# bit-exact on one pair, deadlock/50%-cov on another). Lifted from the
# trash_to_be_removed/_t30_*_determinism.py probes: run the SAME (shape,config)
# K=3 times and assert every run's recv is BIT-IDENTICAL (torch.equal) to the
# first AND fully covered (untouched == 0). NON-VACUOUS: a race that corrupts even
# one row breaks torch.equal across runs; a missed scatter cell breaks coverage.
# --------------------------------------------------------------------------- #

_DET_RUNS = 3  # K: number of repeated runs that must be bit-identical

# (path, pingpong) cells. The race scales with multi-wave M / tiles_per_peer, so the
# back paths drive a LARGER N (N=512 -> N_loc=256 = 2 CTA-M tiles/peer at cp=2) and
# the front drives a LARGER token block M (>= several CTA-M tiles -> multi-wave). Both
# COOPERATIVE (the regime where the race bites) and PINGPONG (the documented remedy)
# are covered per path. cp-agnostic: the per-path setup skips geometrically-invalid
# cells (e.g. N_loc % tile_M) so a cp=4 torchrun collects the SAME ids.
_DET_CELLS = [
    ("back-native", False),
    ("back-native", True),
]


@pytest.mark.usefixtures("_nvshmem__distributed__a2a_fusion")
@pytest.mark.usefixtures("_module_skip__distributed__a2a_fusion")
@pytest.mark.parametrize(
    "path,pingpong", _DET_CELLS, ids=[f"{p}-{'pp' if pp else 'coop'}" for (p, pp) in _DET_CELLS]
)
@matrix_exempt(
    "the subject is REPEATABILITY -- three runs of one configuration must be bit-identical -- so the axis that matters is the run index, which is not a shape. It sweeps the store path and the schedule because those are what could introduce a race; a shape sweep would triple the cost of a claim no shape changes"
)
def test_fused_store_deterministic(dist_manager, device, world_size, path, pingpong):
    """K=3 repeated fused-store runs must be BIT-IDENTICAL + fully covered.

    The async-vec-race signature is exactly cross-run non-determinism; ``torch.equal``
    across K runs of the SAME compiled kernel + SAME inputs is the strongest catch
    (stricter than the rel-err bar). Coverage (untouched == 0) keeps it non-vacuous —
    a store that no-ops every run would be trivially "deterministic". The store is also
    gated against its fp32 reference (``_rel_err < REL_BAR``) on the first run so a
    deterministically-WRONG store still fails. cp-agnostic via the per-path setup skips.
    """

    assert path == "back-native", path
    compiled, run, recv, untouched_fn, expected = _setup_back_gemm_native(
        dist_manager,
        device,
        world_size,
        N=512,
        D=8,
        K=256,
        tile_shape_mn=(128, 128),
        pingpong=pingpong,
    )
    mode = "pingpong" if pingpong else "cooperative"
    try:
        first = None
        for it in range(_DET_RUNS):
            recv.fill_(-99.0)  # re-stamp the sentinel so each run's coverage is independent
            _nvshmem_barrier()
            run()
            _drain()
            untouched = untouched_fn()
            snap = recv.clone()
            if it == 0:
                first = snap
            bit_identical = bool(torch.equal(snap, first))
            print(
                f"\n[det {path} {mode} cp={world_size}] rank={dist_manager.rank} iter={it} "
                f"untouched={untouched} bit_identical_to_run0={bit_identical}",
                flush=True,
            )
            # Coverage (non-vacuity) + bit-identity every run.
            assert untouched == 0, (
                f"{path} {mode}: run {it} left {untouched} recv cells unwritten (coverage gap "
                f"-> determinism would be vacuous)"
            )
            assert bit_identical, (
                f"{path} {mode}: run {it} recv differs from run 0 (async-vec-race "
                f"non-determinism: max_abs={(snap.float() - first.float()).abs().max().item():.3e})"
            )
        # A deterministically-wrong store still fails: gate run 0 against the fp32 ref. Bit-identity
        # across runs says the store is repeatable, which is silent about whether it is right --
        # a store that drops the same cell every time is perfectly deterministic.
        worst = _assert_recv_matches(
            recv,
            expected,
            bar=REL_BAR__distributed__a2a_fusion,
            what=f"{path} {mode} (deterministic but checked against the fp32 reference)",
        )
        print(
            f"\n[det {path} {mode} cp={world_size}] rank={dist_manager.rank} "
            f"DETERMINISTIC over {_DET_RUNS} runs, worst_ratio={worst:.3f} of bound "
            f"(FAILS at 1.0) PASS",
            flush=True,
        )
    finally:
        compiled.free()
        _nvshmem_barrier()


# --------------------------------------------------------------------------- #
# Autotunability — the config knobs (tile_shape_mn, pingpong) flow through
# ``super().__init__`` unchanged and the store stays correct at EACH config, so an
# autotuner could sweep them (the §3.1 "no hardcoded config" / thin-subclass
# property). Lifted from the trash_to_be_removed/_t{22,32}_autotune_audit.py probes:
# compile + run + re-gate (``_rel_err < REL_BAR``) per config at a fixed shape. This
# does NOT run fold_cp_ops's autotuner (multi-GPU unsupported) — it proves the knobs are
# accepted and correct, which is the autotunable precondition.
# --------------------------------------------------------------------------- #

# (tile_shape_mn, pingpong) grid — the audit set: baseline, bigger tile_N, +pingpong.
# All valid under the per-path shapes below (N_loc / rows_per_peer % tile_M == 0;
# tile_N divides the GEMM N; the front postact tile = tile_N//2 tiles Dloc cleanly).
_AUTOTUNE_CONFIGS = [((128, 128), False), ((128, 256), False), ((128, 128), True)]
_AT_CFG_IDS = [f"tile{tm}x{tn}-{'pp' if pp else 'coop'}" for ((tm, tn), pp) in _AUTOTUNE_CONFIGS]


@pytest.mark.usefixtures("_nvshmem__distributed__a2a_fusion")
@pytest.mark.usefixtures("_module_skip__distributed__a2a_fusion")
@EPI.parametrize(
    "tile",
    "pingpong",
    cells=[((128, 128), False), ((128, 256), False), ((128, 128), True)],
    because=(
        "the audit set: a baseline, a wider tile_N, and the same tile with pingpong on. The "
        "subject is that the store is INDEPENDENT of these knobs, which needs at least two "
        "values of each and does not need their product -- the diagonal separates a tile "
        "effect from a schedule effect at three cells instead of six. tile_M=256 is excluded "
        "because this path stages a single 128-row epilogue subtile"
    ),
)
@pytest.mark.parametrize("path", ["back-native"])
def test_fused_store_autotunable(dist_manager, device, world_size, path, tile, pingpong):
    """Each fused store passes the rel-err gate at >=2 (tile_shape_mn, pingpong) configs.

    Proves the config knobs reach ``super().__init__`` and the store stays correct per
    config (the autotunable / no-hardcoded-config property). Same ``_rel_err < REL_BAR``
    gate as the correctness tests; coverage (untouched == 0) printed as a diagnostic.
    cp-agnostic: the per-path setup skips geometrically-invalid (cp, tile) cells.
    """

    # Shapes chosen so the WHOLE config grid is valid: N=512 -> {tile_N 128,256} both
    # divide it and N_loc=512/cp % 128 == 0 at cp in {2,4}; front N=256 -> Dloc=128
    # tiles cleanly against postact tile {64,128}.
    assert path == "back-native", path
    compiled, run, recv, untouched_fn, expected = _setup_back_gemm_native(
        dist_manager,
        device,
        world_size,
        N=512,
        D=8,
        K=256,
        tile_shape_mn=tile,
        pingpong=pingpong,
    )
    n_cells_fn = lambda: world_size * (8 // world_size) * 1 * (512 // world_size)
    try:
        _nvshmem_barrier()
        run()
        _drain()
        untouched = untouched_fn()
        n_cells = n_cells_fn()
        tag = f"autotune {path} tile={tile} pp={pingpong} cp={world_size}"
        worst = _assert_recv_matches(
            recv,
            expected,
            bar=REL_BAR__distributed__a2a_fusion,
            what=f"{tag} (a failure here means the config knob did not flow through)",
        )
        # Asserted rather than printed, for the reason given at the coupled-store gate above.
        assert untouched == 0, (
            f"{tag}: {untouched}/{n_cells} recv cells still hold the -99 sentinel, so the store "
            "never reached them"
        )
        print(
            f"\n[{tag}] rank={dist_manager.rank} worst_ratio={worst:.3f} of its per-element "
            f"bound (FAILS at 1.0; bound = {REL_BAR__distributed__a2a_fusion}*max|ref|) "
            f"untouched={untouched}/{n_cells} PASS",
            flush=True,
        )
    finally:
        compiled.free()
        _nvshmem_barrier()


# --------------------------------------------------------------------------- #
# 2-D autotunability — the SAME no-hardcoded-config property under a 2-D ``(cp0, cp1)``
# token shard via ``configure_a2a_sharded``. Proves the (tile, pingpong) knobs flow
# through ``super().__init__`` AND the 2-D tile->peer unravel stays correct at each
# config. Skips unless the session mesh is 2-D (CPO_DIST_MESH=cp=2*2). Reuses the SAME
# correctness gate (error-histogram for back; rel-err for front).
# --------------------------------------------------------------------------- #


def _setup_back_gemm_native_sharded(
    dist_manager, device, world_size, *, N, D, K, tile_shape_mn, pingpong
):
    """One back GEMM-native store at (N,D,K,tile,pingpong) via ``configure_a2a_sharded``.

    Covers BOTH a 1-D ``(cp,)`` and a 2-D ``(cp0, cp1)`` session mesh (the generic entry).
    Returns ``(compiled, run, recv, untouched_fn, ref_5d, reshape_to_bijD)``. The recv is
    the flat-cp 5-D ``(cp, Dloc, B, N_i_loc, N_j_loc)`` (for 1-D, ``N_j_loc == N``);
    ``ref_5d`` is the fp32 reshard oracle; ``reshape_to_bijD`` maps either 5-D tensor to
    ``(B, N_i, N_j, D)`` for the error-histogram. Geometry constraints raise ``pytest.skip``.
    """

    axis_sizes = _cp_axis_sizes(dist_manager)
    cp0 = axis_sizes[0]
    cp1 = axis_sizes[1] if len(axis_sizes) > 1 else 1
    cp = cp0 * cp1
    if D % cp != 0 or N % cp0 != 0 or N % cp1 != 0:
        rank_invariant_skip(
            f"D={D}/N={N} not divisible by cp axes {axis_sizes}", because=_GEOMETRY_UNIFORM_BECAUSE
        )
    N_i_loc, N_j_loc = N // cp0, N // cp1
    if N_i_loc % tile_shape_mn[0] != 0 or N_j_loc % tile_shape_mn[1] != 0:
        rank_invariant_skip(
            f"N_i_loc={N_i_loc}/N_j_loc={N_j_loc} vs tile {tile_shape_mn} at {axis_sizes}",
            because=_GEOMETRY_UNIFORM_BECAUSE,
        )
    Dloc, B = D // cp, 1
    L = Dloc * B
    placements, pm = _trimul_placements(dist_manager)
    _skip_coupled_cross_node(pm, "back gemm-native sharded store")
    mesh = _session_cp_mesh(dist_manager)
    torch.manual_seed(4321 + dist_manager.rank)
    A = torch.randn(L, N, K, device=device, dtype=torch.bfloat16) / (K**0.5)
    Bt = torch.randn(L, N, K, device=device, dtype=torch.bfloat16) / (K**0.5)
    with torch.cuda.use_mem_pool(DistributedManager.symmetric_mempool(device)):
        recv = torch.empty((cp, Dloc, B, N_i_loc, N_j_loc), dtype=torch.bfloat16, device=device)
    recv.fill_(-99.0)
    fn = lambda g: g.configure_a2a_sharded(mesh, placements, pe_map=pm, B=B, N=N, gemm_native=True)
    compiled, run_args = _compile_back_gemm_native(
        A, Bt, recv, tile_shape_mn, gemm_native_cfg=None, configure_fn=fn, pingpong=pingpong
    )
    ref_5d = _ref_back_gemm_native_2d(
        A,
        Bt,
        pm,
        cp0=cp0,
        cp1=cp1,
        Dloc=Dloc,
        B=B,
        N_i_loc=N_i_loc,
        N_j_loc=N_j_loc,
        N=N,
        world_size=world_size,
    )
    reshape = lambda t: t.permute(2, 3, 4, 0, 1).reshape(B, N_i_loc, N_j_loc, cp * Dloc)
    return (
        compiled,
        lambda: compiled(*run_args),
        recv,
        lambda: int((recv == -99.0).all(dim=4).sum().item()),
        ref_5d,
        reshape,
    )


@pytest.mark.usefixtures("_nvshmem__distributed__a2a_fusion")
@pytest.mark.usefixtures("_module_skip__distributed__a2a_fusion")
@EPI.parametrize(
    "tile",
    "pingpong",
    cells=[((128, 128), False), ((128, 256), False), ((128, 128), True)],
    because=(
        "the audit set: a baseline, a wider tile_N, and the same tile with pingpong on. The "
        "subject is that the store is INDEPENDENT of these knobs, which needs at least two "
        "values of each and does not need their product -- the diagonal separates a tile "
        "effect from a schedule effect at three cells instead of six. tile_M=256 is excluded "
        "because this path stages a single 128-row epilogue subtile"
    ),
)
@pytest.mark.parametrize("path", ["back-native"])
@EPI.parametrize("mesh")
def test_fused_store_autotunable_sharded(
    apply_mesh, mesh, dist_manager, device, world_size, path, tile, pingpong
):
    """Each fused store passes the gate at >=2 (tile, pingpong) configs via the GENERIC entry.

    Proves the (tile, pingpong) knobs flow through ``super().__init__`` under
    ``configure_a2a_sharded`` AND the store (incl. the 2-D tile->peer unravel) stays
    correct per config — at BOTH a 1-D ``cp`` mesh (CPO_DIST_MESH=cp=2) AND a 2-D
    ``(cp0, cp1)`` mesh (cp=2*2). back-native uses the error-histogram (rel_L2 < bar AND
    n_outlier_rows == 0); front uses the flat-cp rel-err. The back-plain path has no 2-D
    variant (1-D token-major only), so only back-native is swept here.
    """
    apply_mesh(mesh)
    from tests.distributed.correctness_harness import compute_error_histogram

    # N=512 so the whole config grid is valid at 1-D cp in {2,4} AND 2-D (2,2):
    # N_i_loc=N//cp0, N_j_loc=N//cp1 -> tile_N in {128,256} divide both; back D=8.
    assert path == "back-native", path
    compiled, run, recv, untouched_fn, ref_5d, reshape = _setup_back_gemm_native_sharded(
        dist_manager,
        device,
        world_size,
        N=512,
        D=8,
        K=256,
        tile_shape_mn=tile,
        pingpong=pingpong,
    )
    try:
        _nvshmem_barrier()
        run()
        _drain()
        untouched = untouched_fn()
        h = compute_error_histogram(reshape(recv), reshape(ref_5d))
        rel, n_out = h.rel_l2, h.n_outlier_rows
        nd = len(_cp_axis_sizes(dist_manager))
        tag = f"autotune-sharded({nd}D) {path} tile={tile} pp={pingpong} cp={world_size}"
        print(
            f"\n[{tag}] rank={dist_manager.rank} rel={rel:.3e} (bar {REL_BAR__distributed__a2a_fusion}) outliers={n_out} "
            f"untouched={untouched} (gate below)",
            flush=True,
        )
        # The gate is PER ELEMENT at this section's own bar, not the histogram's pooled `rel_l2`.
        # Same bound, different reporting: `rel_l2` is a norm ratio, so a single mis-routed cell has
        # to dominate a sum over millions of elements before it moves -- which is exactly the defect
        # class a 2-D tile->peer unravel produces. The histogram is kept because it LOCALIZES a
        # failure once one happens; it no longer decides one.
        _assert_recv_matches(
            reshape(recv), reshape(ref_5d), bar=REL_BAR__distributed__a2a_fusion, what=tag
        )
        assert n_out == 0, f"{tag}: {n_out} outlier rows — 2-D unravel mis-routed at this config"
        assert untouched == 0, f"{tag}: {untouched} recv cells unwritten"
    finally:
        compiled.free()
        _nvshmem_barrier()


# --------------------------------------------------------------------------- #
# arbitrary_n=ON correctness gate (Tier-1 #1) — the design-E back store at a
# NON-%128 / sub-128 per-peer token block N_loc, across the COUPLED in-epilogue
# TMA-S2G store + the two SIMT putwarp drains (strided_putwarp / claim). Mirrors
# the PROVEN ad-hoc validators verbatim:
#   * coupled         -> benchmark/distributed/arb_n_coupled_validate._validate_one
#                        (configure_a2a_gemm_native(arbitrary_n=True, N=N); oracle compare;
#                         per-row L2 histogram).
#   * strided_putwarp -> benchmark/distributed/arb_n_simt_smoke._run_case(claim_drain=False)
#   * claim           -> benchmark/distributed/arb_n_simt_smoke._run_case(claim_drain=True)
# (the SIMT drains add the §7.16j write-once GMEM ring + the runtime (cp,) PE table).
#
# N list: the STRADDLE point is the per-peer token block N_loc = N//cp NOT a
# multiple of cta_tile_M=128. The %8-non-%128 values {264, 520, 1032, 1544} straddle at
# cp=2 (N_loc {132, 260, 516, 772}); the ALIGNED-N {256, 768} are %128 but at cp>=4
# give a sub-128 / non-%128 N_loc ({64, 192} at cp=4) — they ONLY straddle at cp>=4 so at
# cp=2 they reduce to the byte-identical aligned path (arbitrary_n=True is a no-op there).
# Skip a cell ONLY when N % cp != 0 (the reshard split must be integral); a sub-128
# N_loc is the WHOLE POINT, never skipped. The gate is the error-histogram (rel_L2 < 2e-2
# AND n_outlier_rows == 0) + scatter coverage (untouched == 0).
# --------------------------------------------------------------------------- #
_ARB_N_VARIANTS = ["coupled", "pe_aligned"]
# 384 = the canonical straddle of arb_n_simt_smoke (cp=2 -> N_loc=192, 192%128=64); 264/520/1032/1544
# the %8-non-%128 straddles of arb_n_coupled_validate (cp=2 -> N_loc 132/260/516/772). 256/768 are
# %128 but straddle at cp>=4 (sub-128 N_loc 64/96/192). Together: straddle + sub-128 across cp {2,4,8}.
_ARB_N_TOKENS = [256, 264, 384, 520, 576, 768, 1032, 1544]


@pytest.mark.usefixtures("_nvshmem__distributed__a2a_fusion")
@pytest.mark.usefixtures("_module_skip__distributed__a2a_fusion")
@EPI.parametrize(
    "N",
    "mesh",
    cells=[
        (n, m) for n in _ARB_N_TOKENS for m in EPI.axis("mesh").values if not _misaligned_2d(n, m)
    ],
    because=(
        "JOINT, not two stacked decorators, and that is load-bearing rather than style: "
        "`matrix.parametrize` evaluates an Unsupported region only when the SAME call "
        "parametrizes every axis the region reads. The 16-B per-peer region reads N AND mesh, "
        "so stacking `parametrize('N')` above `parametrize('mesh')` leaves neither call a "
        "superset of {N, mesh}, the region check is SKIPPED on both, and pytest then crosses "
        "them into the very combination the region forbids -- measured: 4 misaligned cells "
        "collected in this test's grid before this change. Listing the cells jointly is what "
        "lets the region refuse them. Do NOT simplify this back into two decorators. The "
        "cells are the straddle pool crossed with every mesh, MINUS the factored-misaligned "
        "combinations the region owns. "
        "That sentence used to describe a hand-written list of 40 cells, and the list had ROTTED: "
        "when `cp=(2,8)` and `cp=(4,4)` joined the mesh pool the literal was not updated, so 8 "
        "LEGAL cells -- {256, 384, 576, 768} x those two specs -- were silently absent. They are "
        "the worst possible ones to lose: the mesh axis's own comment calls that pair the first "
        "FACTORED **and** CROSS-NODE specs, i.e. the combination production actually runs, "
        "previously unreachable. So the list is now DERIVED from `_misaligned_2d`, the same "
        "predicate the region is declared with. A mesh value added to the pool joins this test by "
        "construction, and the cell list cannot disagree with the region it is defined against -- "
        "a frozen literal can only be right on the day it is written"
    ),
)
@pytest.mark.parametrize("variant", _ARB_N_VARIANTS, ids=_ARB_N_VARIANTS)
def test_back_gemm_native_arbitrary_n(
    apply_mesh, mesh, dist_manager, device, world_size, variant, N
):
    """arbitrary_n=ON design-E back store at a straddling/sub-128 N_loc (coupled + SIMT drains)."""
    apply_mesh(mesh)
    from tests.distributed.correctness_harness import compute_error_histogram

    cp = world_size
    # SKIP only on a non-integral reshard split (the kernel needs N % cp == 0 so
    # N_loc = N//cp is an integer). A sub-128 / non-%128 N_loc is the POINT -> never skipped.
    if N % cp != 0:
        rank_invariant_skip(
            f"N={N} not divisible by cp={cp} (reshard split must be integral)",
            because=_GEOMETRY_UNIFORM_BECAUSE,
        )
    N = N
    N_loc = N // cp
    # Small D (Dloc=2, B=1 -> L=2) for fast cold compile (cache OFF), as in arb_n_coupled_validate;
    # D % cp == 0 by construction. M = full token-i = cp*N_loc = N (square GEMM grid).
    D = 2 * cp
    Dloc, B = D // cp, 1
    M, K = N, 256
    L = Dloc * B
    tile_shape_mn = (128, 128)
    epi_m = 128  # gcd(128, cta_tile_M=128); N_loc straddles iff N_loc % epi_m != 0
    straddles = (N_loc % epi_m) != 0
    pm = _pe_map(dist_manager)
    # TOPOLOGY guard, BEFORE any symmetric allocation. BOTH variants are COUPLED stores (see
    # _cfg below: `pe_aligned` only adds pe_aligned_tiling on top of the same coupled config), and a
    # coupled store TMA-S2Gs / STGs straight into the peer's symmetric heap via nvshmem_ptr, which
    # returns NULL for an IB (non-P2P) peer -> a cross-node cell takes an illegal memory access.
    # `pe_aligned` is NOT a knock-on of `coupled`: it faults through its own base_m derivation, its own
    # `fast = aligned` predicate and its own TMA-descriptor store (gemm_sm90_a2a.py:3567, :3585, :3592),
    # measured independently in four isolated fresh processes at cp=16 (recorded jobs).
    # The SAME store at the SAME shapes on cp=8 / all-P2P passes 8/8, so this is a topology
    # boundary, not an off-grid addressing bug — the straddle hypothesis is refuted. Cross-node coverage
    # lives on the V6 / ib_drain decoupled path, which is structurally immune (recorded jobs).
    # Rank-invariant predicate (a pure function of pe_table) -> a plain skip is collective (E4).
    topology.skip_if_coupled_cross_node(
        tuple(int(x) for x in pm.cp_pe_table.tolist()),
        detail=f"back gemm-native arbitrary_n variant={variant} is a coupled store (both variants are); "
        f"cross-node is covered by the V6 decoupled drain.",
    )
    max_active_clusters = get_max_active_clusters(1)

    torch.manual_seed(4321 + dist_manager.rank)
    A = torch.randn(L, M, K, device=device, dtype=torch.bfloat16) / (K**0.5)
    Bt = torch.randn(L, N, K, device=device, dtype=torch.bfloat16) / (K**0.5)
    with torch.cuda.use_mem_pool(DistributedManager.symmetric_mempool(device)):
        recv = torch.empty((cp, Dloc, B, N_loc, N), dtype=torch.bfloat16, device=device)
    recv.fill_(-99.0)  # coverage sentinel
    pe_table = tuple(int(x) for x in pm.cp_pe_table.tolist())

    # Per-variant config closure. arbitrary_n=True + N is MANDATORY (the kernel raises without N).
    # coupled = the default in-epilogue TMA-S2G store; pe_aligned = per-peer ceil tiling
    # (pe_aligned_tiling). Both are COUPLED stores (no decoupled ring / PE table needed).
    def _cfg(g):
        kw = dict(
            cp=cp,
            my_cp_rank=int(pm.my_cp_rank),
            B=B,
            N_loc=N_loc,
            pe_table=pe_table,
            arbitrary_n=True,
            N=N,
        )
        if variant == "pe_aligned":
            kw.update(pe_aligned_tiling=True)
        g.configure_a2a_gemm_native(**kw)

    ring_t = pe_dev = None
    rd_probe = None

    compiled, run_args = _compile_back_gemm_native(
        A,
        Bt,
        recv,
        tile_shape_mn,
        gemm_native_cfg=None,
        configure_fn=_cfg,
        ring_t=ring_t,
        pe_table_dev_t=pe_dev,
    )
    try:
        _nvshmem_barrier()
        compiled(*run_args)
        _drain()

        expected = _ref_back_gemm_native(
            A, Bt, pm, cp=cp, Dloc=Dloc, B=B, N_loc=N_loc, N=N, world_size=world_size
        )
        # Error-histogram gate (mirrors the validators' per-row L2): map the 5-D recv
        # (cp,Dloc,B,N_loc,N) -> (cp, Dloc*B, N_loc, N) so the leading 3 dims (cp, Dloc*B, N_loc)
        # are the ROWS (scatter granularity) and the last dim N (j) the feature -> a mis-routed
        # straddle row shows as an outlier row a scalar rel_err would average away.
        got4d = recv.reshape(cp, Dloc * B, N_loc, N)
        exp4d = expected.reshape(cp, Dloc * B, N_loc, N)
        h = compute_error_histogram(got4d, exp4d)
        untouched = int((recv == -99.0).all(dim=4).sum().item())
        n_cells = cp * Dloc * B * N_loc
        rd_tag = f" rd={rd_probe}" if rd_probe is not None else ""
        tag = (
            f"arb-n[{variant}] cp={cp} N={N} N_loc={N_loc} straddle={straddles} Dloc={Dloc}{rd_tag}"
        )
        print(
            f"\n[{tag}] rel_L2={h.rel_l2:.3e} "
            f"row_max/med={h.row_outlier_ratio:.1f}x outlier_rows={h.n_outlier_rows} "
            f"untouched={untouched}/{n_cells} (gate below)",
            flush=True,
        )
        _assert_recv_matches(got4d, exp4d, bar=2e-2, what=tag)
        assert h.n_outlier_rows == 0, (
            f"{tag}: {h.n_outlier_rows} outlier row(s) (ratio {h.row_outlier_ratio:.1f}x, "
            f"worst@{h.worst_row_index}); untouched={untouched}/{n_cells} — straddle row mis-routed"
        )
        assert untouched == 0, f"{tag}: {untouched}/{n_cells} recv cells unwritten (coverage gap)"
    finally:
        compiled.free()
        _nvshmem_barrier()


# --------------------------------------------------------------------------- #
# route-2 DECOUPLED straddle store correctness (§A.2 Stage 0): the HIGH peer of a
# straddle subtile is served by a SECOND TMA-S2G (the reduced global_dim-shrink atom
# family) on the STOCK GEMM tiling -- NOT the per-row SIMT tail. The LOW peer reuses the
# clamped TMA. The high peer is FULLY TMA for EVERY split (even AND odd, #19): its valid
# SMEM rows [split, epi_m) are cooperatively repacked in CHUNK-row strips into a small
# even-aligned NON-swizzled scratch (sidestepping the odd-`split` 256-B SMEM-source
# misalignment) then TMA-S2G'd from scratch row 0 -> ZERO intra-NVLink SIMT at any cp.
# (cp{2,4} have even splits, cp=8 odd-N_loc has odd splits -- both all-TMA.) Same
# error-histogram gate as the arbitrary_n test:
# PASS = rel_L2 < 2e-2 AND n_outlier_rows == 0 AND untouched == 0 (full coverage). route2
# requires N_loc >= epi_m (a tile straddles <=1 boundary -> one high peer), so a cell with
# N_loc = N//cp < 128 is SKIPPED (the sub-epi_m multi-straddle regime is clamped/
# pe_aligned territory). At cp=2/4 the %8-non-%128 cells {1032,1544,2056,...} straddle with
# an even split; at cp=8 they straddle with an odd split (SIMT-high fallback exercised).
# --------------------------------------------------------------------------- #
_ROUTE2_N_TOKENS = [264, 384, 520, 576, 1032, 1544, 2056]


# CONFIG-CHANGE correctness (2040-fix): pe_aligned + the FASTER GEMM config tile(128,128) cluster(1,2)
# (cluster_M==1 -> pe_aligned-compatible). cluster_N=2 adds B-operand N-multicast (the dominant
# WGMMA-issue lever: bare GEMM S2040 390->553 TF) on the SAME validated (128,128) tile — (128,256)
# would give more TF but does NOT compile for the in-epilogue A2A store at D=256 (ptxas insufficient
# registers). The cluster + the A2A store (incl. the partial-N store on a straddle) MUST stay
# oracle-clean for the config to be adoptable as the default. Straddle shapes at cp4: N576/N520/N1032
# (N_loc%128!=0) + aligned N2048. Gate = the error-histogram (rel_L2<2e-2, 0 outliers, full coverage).
_CFG_FIX_TILE = (128, 128)
_CFG_FIX_CLUSTER = (1, 2, 1)  # cluster_N=2, cluster_M=1 (pe_aligned guard requires cluster_M==1)


@pytest.mark.usefixtures("_nvshmem__distributed__a2a_fusion")
@pytest.mark.usefixtures("_module_skip__distributed__a2a_fusion")
@EPI.parametrize(
    "N",
    "mesh",
    cells=[
        (n, m)
        for n in (520, 576, 1032, 2048)
        for m in EPI.axis("mesh").values
        if not _misaligned_2d(n, m)
    ],
    because=(
        "three straddles plus one aligned control at 2048. The subject is a CONFIG change "
        "(a faster tile and cluster) rather than a shape, so the pool only has to keep "
        "pe_aligned engaged -- which needs a straddle -- while the control shows the config "
        "change is not itself the thing being measured. "
        "N and mesh are parametrized JOINTLY rather than as two decorators, and that is the whole "
        "point of this cell list: pytest CROSSES separate parametrize marks, and the cross put "
        "(520|1032) x every FACTORED spec into this test -- 8 cells the matrix declares "
        "UNSUPPORTED. At those, N//cp0 is 4 mod 8, so the 2-D store's per-peer box start is not "
        "16-B aligned and configure MUST raise; a success-asserting test may not claim them. "
        "This is a re-parametrization and NOT a coverage drop: their refusal is owned by the "
        "region and exercised by "
        "test_a_declared_unsupported_configuration_is_refused_at_the_front_door, whose "
        "parametrize_unsupported('N', 'mesh', 'dyn_cp') sweep reaches _misaligned_2d in 72 cells. "
        "The list is DERIVED from _misaligned_2d rather than written out because that predicate is "
        "the single source of truth for which pairs are legal -- a mesh value added to the pool "
        "later then joins this test automatically, where a frozen literal would silently omit it"
    ),
)
def test_back_gemm_native_pe_aligned_fastcfg_correct(
    apply_mesh, mesh, dist_manager, device, world_size, N
):
    """pe_aligned + tile(128,256) cluster(1,2) (the 2040-fix config) oracle-clean — straddle
    N{520,576,1032} (N_loc%128!=0) + aligned N2048 (N_loc=512 %128==0; pe_aligned auto-reduces off)."""
    apply_mesh(mesh)
    from tests.distributed.correctness_harness import compute_error_histogram

    cp = world_size
    if N % cp != 0:
        rank_invariant_skip(f"N={N} not divisible by cp={cp}", because=_GEOMETRY_UNIFORM_BECAUSE)
    N = N
    N_loc = N // cp
    D = 2 * cp
    Dloc, B = D // cp, 1
    M, K = N, 256
    L = Dloc * B
    epi_m = 128
    straddles = (N_loc % epi_m) != 0
    pm = _pe_map(dist_manager)
    _skip_coupled_cross_node(pm, "back gemm-native pe_aligned store")
    torch.manual_seed(4321 + dist_manager.rank)
    A = torch.randn(L, M, K, device=device, dtype=torch.bfloat16) / (K**0.5)
    Bt = torch.randn(L, N, K, device=device, dtype=torch.bfloat16) / (K**0.5)
    with torch.cuda.use_mem_pool(DistributedManager.symmetric_mempool(device)):
        recv = torch.empty((cp, Dloc, B, N_loc, N), dtype=torch.bfloat16, device=device)
    recv.fill_(-99.0)
    pe_table = tuple(int(x) for x in pm.cp_pe_table.tolist())

    def _cfg(g):
        g.configure_a2a_gemm_native(
            cp=cp,
            my_cp_rank=int(pm.my_cp_rank),
            B=B,
            N_loc=N_loc,
            pe_table=pe_table,
            arbitrary_n=True,
            pe_aligned_tiling=True,
            N=N,
        )

    compiled, run_args = _compile_back_gemm_native(
        A,
        Bt,
        recv,
        _CFG_FIX_TILE,
        gemm_native_cfg=None,
        configure_fn=_cfg,
        cluster_shape_mnk=_CFG_FIX_CLUSTER,
    )
    try:
        _nvshmem_barrier()
        compiled(*run_args)
        _drain()
        expected = _ref_back_gemm_native(
            A, Bt, pm, cp=cp, Dloc=Dloc, B=B, N_loc=N_loc, N=N, world_size=world_size
        )
        got4d = recv.reshape(cp, Dloc * B, N_loc, N)
        exp4d = expected.reshape(cp, Dloc * B, N_loc, N)
        h = compute_error_histogram(got4d, exp4d)
        untouched = int((recv == -99.0).all(dim=4).sum().item())
        n_cells = cp * Dloc * B * N_loc
        tag = (
            f"pe-aligned-fastcfg{_CFG_FIX_TILE}c{_CFG_FIX_CLUSTER[:2]} cp={cp} N={N} "
            f"N_loc={N_loc} straddle={straddles}"
        )
        print(
            f"\n[{tag}] rel_L2={h.rel_l2:.3e} outlier_rows={h.n_outlier_rows} "
            f"untouched={untouched}/{n_cells} (gate below)",
            flush=True,
        )
        _assert_recv_matches(got4d, exp4d, bar=2e-2, what=tag)
        assert h.n_outlier_rows == 0, f"{tag}: {h.n_outlier_rows} outlier rows"
        assert untouched == 0, f"{tag}: {untouched}/{n_cells} unwritten"
    finally:
        compiled.free()
        _nvshmem_barrier()


@pytest.mark.usefixtures("_nvshmem__distributed__a2a_fusion")
@pytest.mark.usefixtures("_module_skip__distributed__a2a_fusion")
@numeric_exempt(
    "asserts a HOST-SIDE configuration predicate, not a computed tensor: that "
    "arbitrary_n=True auto-reduces to False on a fully-aligned N_loc. No kernel is launched and no "
    "output exists to compare -- the runtime two-store comparison this replaced was removed as "
    "redundant AND artifact-prone (it flagged a -99 coverage diff at cp>=4 between two separate "
    "symmetric recvs even when both configs had reduced to the SAME kernel). Byte-identity here is "
    "proven by the reduction, so the honest gate is the reduction"
)
@pytest.mark.parametrize("variant", _ARB_N_VARIANTS, ids=_ARB_N_VARIANTS)
@EPI.parametrize("mesh")
def test_back_gemm_native_arbitrary_n_aligned_byte_identical(
    apply_mesh, mesh, dist_manager, device, world_size, variant
):
    """arbitrary_n=True == arbitrary_n=False on an aligned N_loc (per-row reduces to per-tile)."""
    apply_mesh(mesh)
    cp = world_size
    N = 512
    N_loc = N // cp
    epi_m = 128
    if N_loc % epi_m != 0:
        rank_invariant_skip(
            f"byte-identity holds only on an aligned N_loc; N={N} cp={cp} -> N_loc={N_loc} straddles",
            because=_GEOMETRY_UNIFORM_BECAUSE,
        )
    D = 2 * cp
    Dloc, B = D // cp, 1
    M, K = N, 256
    L = Dloc * B
    tile_shape_mn = (128, 128)
    pm = _pe_map(dist_manager)
    _skip_coupled_cross_node(pm, "back gemm-native arbitrary_n store")

    torch.manual_seed(4321 + dist_manager.rank)
    A = torch.randn(L, M, K, device=device, dtype=torch.bfloat16) / (K**0.5)
    Bt = torch.randn(L, N, K, device=device, dtype=torch.bfloat16) / (K**0.5)
    pe_table = tuple(int(x) for x in pm.cp_pe_table.tolist())

    def _mk_cfg(arb):
        def f(g):
            kw = dict(
                cp=cp,
                my_cp_rank=int(pm.my_cp_rank),
                B=B,
                N_loc=N_loc,
                pe_table=pe_table,
                arbitrary_n=arb,
                N=N,
            )
            g.configure_a2a_gemm_native(**kw)

        return f

    # §0.8 perf-neutrality is BY CONSTRUCTION: configure auto-reduces arbitrary_n -> False on a FULLY
    # ALIGNED shape (N_loc % epi_m == 0 and N % tile_n == 0) -> the SAME compiled kernel as
    # arbitrary_n=False -> byte-identical output. We assert that reduction HOST-SIDE (deterministic).
    # (The earlier runtime two-store compare is redundant AND had a cp>=4 cross-rank artifact: even with
    # both configs auto-reduced to arbitrary_n=False — same kernel — comparing two stores into two
    # SEPARATE symmetric recvs flagged a -99 coverage diff; the host-side auto-reduction is the real proof.)
    g_arb = GemmA2ASm90(
        Float32,
        torch2cute_dtype_map[A.dtype],
        tile_shape_mn,
        (1, 1, 1),
        pingpong=False,
        is_persistent=True,
    )
    _mk_cfg(True)(g_arb)
    g_base = GemmA2ASm90(
        Float32,
        torch2cute_dtype_map[A.dtype],
        tile_shape_mn,
        (1, 1, 1),
        pingpong=False,
        is_persistent=True,
    )
    _mk_cfg(False)(g_base)
    tag = f"arb-n-autoreduce[{variant}] cp={cp} N={N} N_loc={N_loc}"
    if dist_manager.rank == 0:
        print(
            f"\n[{tag}] arbitrary_n=True -> _a2a_arbitrary_n={g_arb._a2a_arbitrary_n} "
            f"(base={g_base._a2a_arbitrary_n}) "
            f"-> {'PASS' if g_arb._a2a_arbitrary_n is False else 'FAIL'}",
            flush=True,
        )
    assert g_arb._a2a_arbitrary_n is False, (
        f"{tag}: arbitrary_n=True did NOT auto-reduce to =False on a fully-aligned N_loc -> NOT "
        f"byte-identical to arbitrary_n=False by construction (§0.8 perf-neutrality)"
    )
    assert g_base._a2a_arbitrary_n is False


# --------------------------------------------------------------------------- #
# UNSUPPORTED-SHAPE GUARD test — every host-side ``ValueError`` of
# ``configure_a2a_gemm_native`` must RAISE (a malformed config is a LOUD error, never a
# silent miss). The THREE guards (the unsupported-shape coverage the brief requires):
#   (b) arbitrary_n=True + N=None -> the partial-N P_j predicate (gj < N) needs N.
#   (c) arbitrary_n=True + N % 8 != 0 -> 16-B aligned stride-1 runs (bf16 needs N%8==0).
#   (d) arbitrary_n=False + N_loc % 128 != 0 -> the DEFAULT-path one-peer-per-tile constraint
#       (a CTA M-tile must map to ONE peer); arbitrary_n is the opt-in to relax it.
# Pure host-side config validation (no launch / compile / comm), so it's cheap; it takes the
# dist fixture only to read a real cp / my_cp_rank / pe_table. GemmA2ASm90 ctor is host-only.
# --------------------------------------------------------------------------- #
@pytest.mark.usefixtures("_nvshmem__distributed__a2a_fusion")
@pytest.mark.usefixtures("_module_skip__distributed__a2a_fusion")
@numeric_exempt(
    "pure host-side config validation: three `pytest.raises` on configure_a2a_gemm_native. Nothing "
    "is launched, allocated or computed, so there is no output to compare per element. The subject "
    "is that a malformed config is refused LOUDLY, and each raise is already pinned with `match=` "
    "so it cannot be satisfied by an unrelated ValueError"
)
@EPI.parametrize("mesh")
def test_back_gemm_native_arbitrary_n_guard_raises(
    apply_mesh, mesh, dist_manager, device, world_size
):
    """3 host-side guards RAISE: N=None, N%8!=0, default-path N_loc%128."""
    apply_mesh(mesh)
    cp = world_size
    pm = _pe_map(dist_manager)
    pe_table = tuple(int(x) for x in pm.cp_pe_table.tolist())
    my = int(pm.my_cp_rank)
    tile_shape_mn = (128, 128)
    # cta_tile_M=128, epi_m=gcd(128,128)=128 -> a straddling N_loc is any N_loc % 128 != 0.

    def _mk():
        return GemmA2ASm90(
            Float32,
            torch2cute_dtype_map[torch.bfloat16],
            tile_shape_mn,
            (1, 1, 1),
            is_persistent=True,
        )

    # (b) arbitrary_n=True with N=None -> ValueError (the partial-N P_j predicate needs N). Use an
    # ALIGNED N_loc (128) so the FIRST guard (N_loc % cta_tile_M) passes under arbitrary_n and we
    # reach the N-is-None guard specifically.
    # `match=` is REQUIRED, not decoration: a bare pytest.raises(ValueError) is satisfied by ANY
    # ValueError from configure, including the new cross-node topology guard (§5.2) — which would make
    # this assertion vacuously green on a cp=16 job, exactly the failure mode that hid the front's dead
    # 16-B alarm. The shape guards all fire earlier in configure than the topology one, so pinning the
    # message keeps this test asserting what it claims at EVERY mesh.
    with pytest.raises(ValueError, match="requires N"):
        _mk().configure_a2a_gemm_native(
            cp=cp,
            my_cp_rank=my,
            B=1,
            N_loc=128,
            pe_table=pe_table,
            arbitrary_n=True,
            N=None,
        )

    # (c) arbitrary_n=True with N % 8 != 0 -> ValueError (16-B aligned stride-1 runs). Aligned
    # N_loc=128 + a NON-%8 N (130) so only the N%8 guard trips.
    with pytest.raises(ValueError, match="N%8"):
        _mk().configure_a2a_gemm_native(
            cp=cp,
            my_cp_rank=my,
            B=1,
            N_loc=128,
            pe_table=pe_table,
            arbitrary_n=True,
            N=130,
        )

    # (d) arbitrary_n=False (the DEFAULT) with a NON-%128 N_loc -> ValueError: the default path
    # requires N_loc a positive multiple of cta_tile_M (one CTA M-tile -> one peer). This is the
    # constraint arbitrary_n exists to relax; without it a straddling N_loc must be a LOUD error.
    with pytest.raises(ValueError, match="positive multiple of cta_tile_M"):
        _mk().configure_a2a_gemm_native(
            cp=cp,
            my_cp_rank=my,
            B=1,
            N_loc=132,
            pe_table=pe_table,  # 132 % 128 != 0, arbitrary_n OFF
        )

    if dist_manager.rank == 0:
        print(
            "\n[arb-n guard] all 4 host-side ValueError guards RAISE "
            "(TMA-drain+straddle, N=None, N%8!=0, default-path N_loc%128!=0)",
            flush=True,
        )


# --------------------------------------------------------------------------- #
# Tier-2 — back PLAIN (L=1 token-major) store coverage. The plain store routes a CTA tile
# to its peer by the TOKEN-i (M) tile ONLY, so it is architecturally 1-D-token-shard ONLY:
# ``configure_a2a_sharded(gemm_native=False)`` explicitly RAISES ValueError on a 2-D cp mesh
# (the brief's "back-plain 2-D shard" item is unsupported BY DESIGN — a 2-D shard needs the
# GEMM-native d-outer recv, covered by ``test_back_gemm_native_2d_store_correct``). This test
# (a) on a 2-D session mesh asserts the plain+2-D guard RAISES (closes the "is plain 2-D
# silently wrong?" gap), and (b) on a 1-D session mesh runs the plain store via the generic
# sharded entry vs the fp32 oracle (the plain analogue of the 1-D gemm-native store test,
# through the (DeviceMesh, placements) API). One test, mesh-ndim-dispatched, so a cp=2 AND a
# cp=2*2 torchrun each collect the SAME id and exercise the regime valid at that mesh.
# --------------------------------------------------------------------------- #


# =================================================================================================== #
# BACK design-E N-PHASE gate — a cross-N flatness read on the real GemmA2ASm90 design-E store.
#
# ⚠ THE MECHANISM THIS TEST WAS NAMED AFTER IS REFUTED. Read this before trusting the name.
#
# It was originally written believing the design-E recv (cp, Dloc, B, N_i_loc, N_j_loc) pays +21-29 %
# because its token-i DESTINATION ROW STRIDE `2*N_j_loc` is 16-mod-32 for every N congruent to 8
# (mod 16). That came from a CROSS-N comparison — exactly the comparison below — which
# CANNOT isolate the row phase: at fixed cp/tile the row-stride phase, the trailing-j-tile width and
# the OPERAND leading dimension all move together with N. `2N % 32 == 16` <=> `N % 128` is an odd
# multiple of 8, so every "row-dirty" N in this family is also an N whose last j-tile is 8/24/40/…
# columns. No choice of N here separates them.
#
# Measured directly, at FIXED N=1040, varying only ONE byte quantity per probe:
#   * DESTINATION row stride (over-allocate the recv's innermost extent, store into `recv[..., :N]`):
#     0.987-1.002x at Dloc=8 and 0.999-1.002x at Dloc=128 (drift +-0.4 %).
#     ZERO. The destination row stride costs nothing.
#   * SOURCE operand leading dimension (over-allocate A/B's innermost K axis and slice back): 1.113-
#     1.132x at Dloc=8 and 1.213-1.215x at Dloc=128 — against a cross-N of
#     1.221-1.223x at the same Dloc. The MAINLOOP TMA-G2S walks `(M, K=N, L)` operands with an M-row
#     stride of `2*N` bytes; THAT is what this test's cross-N residual is made of.
# See the byte-phase law comment in `gemm_sm90_a2a.py` (beside the a_major="m" 16-B guard) for the
# full record and for why padding `N_j_loc` in the recv is the WRONG fix.
#
# WHAT THIS GATE IS THEREFORE WORTH. It is a real, cheap, single-process FLATNESS gate on the back
# store: a regression that makes any N congruent to 8 (mod 16) disproportionately slow still trips it.
# What it is NOT is evidence for a destination-side cause; do not re-derive that from a firing here.
# Two known defects in its construction, deliberately left in place by the record-correction pass so
# that pass changed no behaviour — fix them together with a decision on the bar, never separately:
#   (a) the N^3 normaliser does not apply. 1032 and 1040 launch the IDENTICAL 16x9x17xL CTA lattice
#       under per-peer ceil tiling, so the true work expectation is 1.000, not (1032/1040)^3 = 0.977;
#       the test inflates its own residual by ~2.3 %.
#   (b) the honest form of the read is the FIXED-N pad sweep (one variable, expectation exactly 1.000,
#       no work-ratio normalisation) on BOTH the destination and the operand stride.
# The rank-invariance assert below is sound as written and catches the FRONT store's class for free.
#
# K = N IS MANDATORY (CLAUDE.md HARD RULE): the back is the SQUARE einsum M=N=K=N. A
# hardcoded/decoupled K measures an O(N^2) thin-K matmul instead of the O(N^3) einsum and inverts every
# store-vs-compute conclusion. Operands are built `randn(L, N, N)` for that reason, and the work ratio
# below is (N_num/N_den)^3 because of it.
#
# Timing goes through `bench_utils.benchmark_single(mode="event")` (the shared, tested life-line — never
# a hand-rolled perf_counter loop, which would measure host DISPATCH, not device time). `reduce=None` so
# each rank keeps its OWN median: the reduction to a single number is what would hide a rank-dependent
# effect, and detecting one is half the point here.
# =================================================================================================== #
_BACK_PHASE_N = [
    1024,
    1032,
    1040,
    1048,
    1056,
]  # dirty: 1032, 1048 (2N % 32 == 16). clean: the rest.
_BACK_PHASE_BAR = 1.25  # the same bar family as the front gate. NEVER widen.


@pytest.mark.usefixtures("_nvshmem__distributed__a2a_fusion")
@pytest.mark.usefixtures("_module_skip__distributed__a2a_fusion")
@numeric_exempt(
    "a PERF gate that happens to live outside tests/perf/: the subject is a measured TIME ratio "
    "against the analytic N^3 work ratio (and its rank spread), compared to _BACK_PHASE_BAR. It "
    "launches kernels but never compares an output to a reference, so there is no per-element "
    "assertion to make -- the same reason tests/perf/ is excluded from this guard by directory. "
    "The one output property it does depend on IS asserted: `untouched == 0` per N, because a "
    "partial store would time FASTER and silently pass the ratio"
)
@EPI.parametrize("mesh")
def test_back_gemm_native_row_stride_phase(apply_mesh, mesh, dist_manager, device, world_size):
    """The back design-E path must be FLAT in N once the analytic N^3 work ratio is divided out.

    NOT a destination-row-stride gate despite the name — that mechanism is REFUTED (measured
    0.999-1.002x at fixed N, recorded jobs). The cross-N residual this reads is dominated by the
    SOURCE-side operand leading dimension (`2*N` byte M-row stride on the mainloop TMA-G2S), measured
    1.213-1.215x at Dloc=128 (a recorded job). See the header comment above and the byte-phase law in
    gemm_sm90_a2a.py.
    """
    apply_mesh(mesh)
    import torch.distributed as dist
    from benchmark.distributed import bench_utils as BU

    cp = world_size
    pm = _pe_map(dist_manager)
    _skip_coupled_cross_node(pm, "back gemm-native row-stride phase gate")
    ns = [n for n in _BACK_PHASE_N if n % cp == 0 and n % 8 == 0]
    if not ({1032, 1040} <= set(ns)):
        rank_invariant_skip(
            f"cp={cp} admits {ns}; the phase read needs BOTH a row-dirty N (2N%32==16, e.g. 1032) and a "
            f"row-clean neighbour (1040) — otherwise there is no single-variable comparison to make.",
            because=_GEOMETRY_UNIFORM_BECAUSE,
        )
    D = 8 * cp  # Dloc = 8 -> L = 8 planes; keeps the einsum ms-scale while the operands stay MB
    Dloc, B = D // cp, 1
    L = Dloc * B
    tile_shape_mn = (128, 128)
    pe_table = tuple(int(x) for x in pm.cp_pe_table.tolist())
    my_rank = int(dist_manager.rank)
    out = {}
    for N in ns:
        N_loc = N // cp
        torch.manual_seed(4321 + dist_manager.rank)
        # K = N — the SQUARE einsum. Never a decoupled K.
        A = torch.randn(L, N, N, device=device, dtype=torch.bfloat16) / (N**0.5)
        Bt = torch.randn(L, N, N, device=device, dtype=torch.bfloat16) / (N**0.5)
        with torch.cuda.use_mem_pool(DistributedManager.symmetric_mempool(device)):
            recv = torch.empty((cp, Dloc, B, N_loc, N), dtype=torch.bfloat16, device=device)
        recv.fill_(-99.0)

        def _cfg(g):
            g.configure_a2a_gemm_native(
                cp=cp,
                my_cp_rank=int(pm.my_cp_rank),
                B=B,
                N_loc=N_loc,
                pe_table=pe_table,
                arbitrary_n=True,
                N=N,
                pe_aligned_tiling=True,
            )

        compiled = None
        try:
            compiled, run_args = _compile_back_gemm_native(
                A, Bt, recv, tile_shape_mn, gemm_native_cfg=None, configure_fn=_cfg
            )
            _nvshmem_barrier()
            compiled(*run_args)
            _drain()
            untouched = int((recv == -99.0).all(dim=4).sum().item())
            res = BU.benchmark_single(
                lambda: compiled(*run_args),
                rounds=25,
                warmup=5,
                iters=1,
                device=device,
                dist=dist_manager,
                mode="event",
                reduce=None,
                label=f"back-phase N={N} cp={cp}",
            )
            out[N] = dict(
                ms=res.median_ms,
                untouched=untouched,
                N_loc=N_loc,
                row_b=(2 * N) % 32,
                slot_b=(Dloc * B * N_loc * N * 2) % 128,
            )
        finally:
            if compiled is not None:
                compiled.free()
            _nvshmem_barrier()
        del A, Bt
        torch.cuda.empty_cache()

    print(
        f"\n[back-phase rank={my_rank}] N   N_loc  rowstride%32  slotbase%128  untouched  median_ms",
        flush=True,
    )
    for N in sorted(out):
        r = out[N]
        print(
            f"  [rank={my_rank}] {N:>5} {r['N_loc']:>6} {r['row_b']:>13} {r['slot_b']:>13} "
            f"{r['untouched']:>10}  {r['ms']:.4f}",
            flush=True,
        )
    for N, r in out.items():
        assert r["untouched"] == 0, (
            f"back-phase N={N}: {r['untouched']} recv cells unwritten — the timing is meaningless if the "
            f"store did not cover the recv (a partial store would look FASTER, not slower)."
        )

    # Single-variable reads: dirty/clean at matched cp, matched per-peer tiling, <=3 % apart in work.
    reads = [
        (1032, 1040, "row-DIRTY (2N%32=16) vs row-CLEAN"),
        (1048, 1040, "row-DIRTY with a different N_loc vs the same clean reference"),
        (1056, 1040, "row-CLEAN vs row-CLEAN — a pure work-ratio NULL"),
    ]
    worst, worst_tag = 1.0, "none"
    for num, den, what in reads:
        if num not in out or den not in out:
            continue
        expected = (num / den) ** 3  # K = N => FLOPs = 2*L*N^3
        measured = out[num]["ms"] / out[den]["ms"]
        excess = measured / expected
        if excess > worst:
            worst, worst_tag = excess, f"N{num}/N{den} ({what})"
        print(
            f"[back-phase rank={my_rank}] N{num}/N{den}: measured {measured:.3f}x / expected(N^3) "
            f"{expected:.3f}x = {excess:.3f}x excess — {what}",
            flush=True,
        )

    # Rank-invariance: every rank runs identical shapes and identical work.
    worst_spread, spread_tag = 1.0, "none"
    for N in sorted(out):
        t = torch.tensor([out[N]["ms"]], device=device, dtype=torch.float32)
        allt = [torch.empty_like(t) for _ in range(world_size)]
        dist.all_gather(allt, t)
        vals = [float(v.item()) for v in allt]
        spread = max(vals) / max(min(vals), 1e-9)
        if spread > worst_spread:
            worst_spread, spread_tag = spread, f"N{N}: {['%.4f' % v for v in vals]}"
        if my_rank == 0:
            print(
                f"[back-phase rank-invariance] N={N}: max/min over ranks = {spread:.3f}x",
                flush=True,
            )

    assert worst < _BACK_PHASE_BAR, (
        f"back design-E path is NOT flat in N: {worst_tag} costs {worst:.3f}x its analytic N^3 work "
        f"ratio (bar {_BACK_PHASE_BAR}). Do NOT read this as a destination row-stride defect — that "
        f"mechanism is REFUTED (0.999-1.002x at fixed N, recorded jobs) and padding `N_j_loc` in "
        f"the recv fixes NOTHING. The measured cause of this cross-N residual is the SOURCE-side "
        f"operand leading dimension: the mainloop TMA-G2S walks `(M, K=N, L)` with a `2*N` byte M-row "
        f"stride, 16-mod-32 for every N congruent to 8 (mod 16) — 1.213-1.215x at Dloc=128 (job "
        f"601331). The lever is the back OPERAND's M-stride (the inner token extent of the "
        f"(Dloc,B,N_i,N_j) view); see the byte-phase law in gemm_sm90_a2a.py. Never widen this bar. "
        f"NOTE the null read N1056/N1040 must stay ~1.0 — if IT is the worst read, the measurement is "
        f"noisy and the cell needs isolating, not the bar moving."
    )
    assert worst_spread < _BACK_PHASE_BAR, (
        f"back store time is RANK-DEPENDENT ({worst_spread:.3f}x max/min, {spread_tag}) at shapes where "
        f"every rank does identical work — a per-rank destination-address effect (the class-1 base-phase "
        f"hazard). The back's own hazard is class-2 (row stride) and is uniform across ranks, so this "
        f"assertion firing means a SECOND, different alignment defect."
    )


@pytest.fixture(scope="session")
def _module_skip__distributed__a2a_fusion():
    pytest.importorskip("nvshmem.core")
    pytest.importorskip("fold_cp_ops.distributed.gemm_a2a_epi")


def _is_sm90():
    # torch's own capability query (NOT cutlass get_device_capacity) so this collection-time skip check
    # does NOT create a cutlass-driver CUDA context that then conflicts with torch's primary context
    # (the cutlass HardwareInfo path raises CUDA_ERROR_INVALID_CONTEXT if it runs before torch inits).
    if not torch.cuda.is_available():
        return False
    try:
        return torch.cuda.get_device_capability(0)[0] == 9
    except Exception:
        return False


REL_BAR__distributed__composite_k_backread = (
    5e-2  # bf16 tensor-core relative-error bar (the parent kernels' metric)
)
_COMPILED = {}  # cp -> compiled kernel (compile once per cp, reuse across shapes)


def _compiled_for(cp):
    if cp in _COMPILED:
        return _COMPILED[cp]
    import cutlass.cute as cute
    from cutlass import BFloat16, Float32, Int32
    from cutlass.cute.runtime import make_fake_stream
    from fold_cp_ops._internal.compile_time.compile_utils import make_fake_tensor as fake_tensor
    from fold_cp_ops._internal.gemm_tvm_ffi_utils import make_scheduler_args
    from fold_cp_ops._internal.arch import get_max_active_clusters
    from fold_cp_ops.distributed.gemm_a2a_epi import GemmA2ASm90

    g = GemmA2ASm90(Float32, BFloat16, (128, 128), (1, 1, 1), pingpong=False, is_persistent=True)
    g._a2a_composite_k = True
    g._a2a_cp = cp
    g._a2a_B = 1
    M = cute.sym_int()
    N = cute.sym_int()
    Xg = cute.sym_int()
    L = cute.sym_int()
    mA = fake_tensor(BFloat16, (M, Xg, L), leading_dim=1, divisibility=8)  # (M,Xg_pad,L) K-major
    mB = fake_tensor(BFloat16, (N, Xg, L), leading_dim=1, divisibility=8)
    mD = fake_tensor(BFloat16, (M, N, L), leading_dim=1, divisibility=8)  # (M,N,L) n-major
    epi = GemmA2ASm90.EpilogueArguments(
        alpha=None,
        beta=None,
        mRowVecBroadcast=None,
        mColVecBroadcast=None,
        add_to_output=False,
        rounding_mode=None,
        sr_seed=None,
        recv=None,
    )
    sched = make_scheduler_args(get_max_active_clusters(1), Int32(8), None, None)
    compiled = cute.compile(
        g,
        mA,
        mB,
        mD,
        None,
        epi,
        sched,
        make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi",
    )
    _COMPILED[cp] = compiled
    return compiled


# (cp, Xg, Xg_pad, Dloc, N_j): aligned (the DEADLOCK trigger); off-grid Xg (pad zeroed); non-pow2 cp;
# non-pow2 cp + off-grid Xg + odd Dloc + arbitrary N_j (the FIRST-PRINCIPLE generality bar).
_CASES = [
    (2, 128, 128, 4, 256),
    (2, 100, 128, 4, 256),
    (3, 128, 128, 6, 200),
    (3, 100, 128, 6, 264),
]


@pytest.mark.skipif(
    not _is_sm90(), reason="composite-K back-read is SM90-only (WGMMA sm_90a-gated); needs an H100"
)
@pytest.mark.parametrize(
    "cp,Xg,Xg_pad,Dloc,Nj",
    _CASES,
    ids=[f"cp{c}_Xg{x}_pad{p}_D{d}_N{n}" for (c, x, p, d, n) in _CASES],
)
@matrix_exempt(
    "a SINGLE-GPU test with no nvshmem and no peers, whose axes are the composite-K decode (cp, Xg, Xg_pad, Dloc, Nj) -- a synthetic recv geometry rather than the distributed store's token extent. Its cp is a loop bound over a fabricated contraction, not a world size, so the matrix's N and D mean different things here"
)
def test_composite_k_backread(cp, Xg, Xg_pad, Dloc, Nj):
    from cutlass import Int32
    from fold_cp_ops.distributed.gemm_a2a_epi import GemmA2ASm90
    from fold_cp_ops._internal.gemm_tvm_ffi_utils import make_scheduler_args
    from fold_cp_ops._internal.arch import get_max_active_clusters

    # RESTORE THE AMBIENT DEVICE ON THE WAY OUT. This test pins cuda:0 deliberately and that pin is
    # kept -- what is added is putting the current device back. It was written for a session that has
    # only one GPU; under a launcher it LEAKS current-device 0 into every later test in this process,
    # whose tensors live on cuda:<local_rank>, and `torch.Tensor.__dlpack__` then refuses with
    # "Can't export tensors on a different CUDA device index. Expected: N. Current device: 0."
    # Measured: that is the entirety of the test_ib_ring_cluster_drain_1d failure, at W=2 and W=8,
    # and it looks rank-asymmetric only because on rank 0 the expected and current indices agree.
    _dev_before = torch.cuda.current_device()
    try:
        dev = torch.device("cuda", 0)
        torch.cuda.set_device(dev)
        torch.zeros(
            1, device=dev
        )  # force torch's primary CUDA context BEFORE cutlass HardwareInfo queries
        torch.manual_seed(1234)
        compiled = _compiled_for(cp)

        ra = torch.randn(Dloc, cp, Nj, Xg_pad, device=dev, dtype=torch.bfloat16) / (Xg**0.5)
        rb = torch.randn(Dloc, cp, Nj, Xg_pad, device=dev, dtype=torch.bfloat16) / (Xg**0.5)
        if Xg_pad > Xg:  # zero the pad rows (front glu(0)=0) so the padded contraction is exact
            ra[:, :, :, Xg:] = 0
            rb[:, :, :, Xg:] = 0
        A3 = ra[:, 0].permute(1, 2, 0)  # (Nj, Xg_pad, Dloc) VIEW; kernel synthesizes the cp mode
        B3 = rb[:, 0].permute(1, 2, 0)
        mD_buf = torch.empty(Dloc, Nj, Nj, device=dev, dtype=torch.bfloat16)  # (d, j, J)
        mD = mD_buf.permute(1, 2, 0)  # (Nj, Nj, Dloc) n-major
        epi = GemmA2ASm90.EpilogueArguments(
            alpha=None,
            beta=None,
            mRowVecBroadcast=None,
            mColVecBroadcast=None,
            add_to_output=None,
            rounding_mode=None,
            sr_seed=None,
            recv=None,
        )
        sched = make_scheduler_args(get_max_active_clusters(1), Int32(8), None, None)
        compiled(A3, B3, mD, None, epi, sched)  # tvm-ffi: torch tensors direct
        torch.cuda.synchronize()

        ref = torch.einsum(
            "drjw,drJw->djJ", ra.float(), rb.float()
        )  # (Dloc, Nj, Nj) == mD_buf layout
        _assert_recv_matches(
            mD_buf,
            ref,
            bar=REL_BAR__distributed__composite_k_backread,
            what=f"composite-K back-read cp={cp} Xg={Xg} Xg_pad={Xg_pad} Dloc={Dloc} Nj={Nj}",
        )
    finally:
        torch.cuda.set_device(_dev_before)


K__distributed__ib_ring = 256
TILE = (128, 128)
ANCHOR__distributed__ib_ring = (
    640  # straddle anchor: N_loc = 640//cp is NOT a multiple of 128 for cp in {2,4,8}.
)
# Dynamic run shapes served by the ONE anchor compile (cp<=8): straddle + aligned + an UNSEEN N +
# a partial token-j shape (N % 128 != 0 -> exercises the runtime col-clamp). All % 8 == 0; the
# per-cp %cp filter runs in-test.
RUN_NS__distributed__ib_ring = [512, 640, 896, 1024, 776]
# MULTI-NODE (cp>8) straddle shapes: every N is %cp AND %8 AND N_loc=N/cp is NOT %128 (a straddle,
# so arbitrary_n stays on + pe_aligned activates). 960 is %128!=0 -> also exercises the partial
# token-j col-clamp multi-node. D=96 (=LCM(16,24,32) multiple) gives Dloc=6/4/3 at cp=16/24/32 ->
# an L>1 batched store (d/b un-bake) at every target cp with ONE D value.
MULTI_NS = {16: [640, 768, 960, 1024], 24: [768, 960], 32: [640, 768, 960, 1024]}
MULTI_D = 96
# 2-D (cp0,cp1) candidate token counts for the cluster_drain straddle tests: N_i_loc=N//cp0 straddles
# tile_m=128 (arbitrary_n on) with per-axis %8 (16-B TMA). Each test filters this pool for its mesh +
# (for the even-shard cluster_drain_2d) nt_j_pp % cluster_n == 0.
DIFF2D_NS = [640, 896, 1152]


@pytest.fixture(scope="session", autouse=False)
def _nvshmem__distributed__ib_ring(dist_manager):
    from fold_cp_ops.distributed.distributed_manager import DistributedManager

    DistributedManager.init_nvshmem()
    yield


def _mark(t):
    return from_dlpack(t, assumed_align=16).mark_layout_dynamic()


def _build_inputs(device, rank, N, cp, Dloc, B, shape_mode="dynamic"):
    """Operands + a fresh symmetric recv for token extent N.

    ``shape_mode="dynamic"`` (the default, and every existing caller) marks the returned views so
    the token extent is a RUNTIME value; ``"static"`` leaves them unmarked so it is baked. The
    distinction is not cosmetic here: a baked extent makes the recv's strides compile-time
    constants, and a constant stride participates in compile-time integer arithmetic that a runtime
    one does not -- which is exactly how the front A2A's peer offset came to be formed in 32 bits.
    """

    M, L, N_loc = N, Dloc * B, N // cp
    torch.manual_seed(4321 + rank)
    A = torch.randn(L, M, K__distributed__ib_ring, device=device, dtype=torch.bfloat16) / (
        K__distributed__ib_ring**0.5
    )
    Bt = torch.randn(L, N, K__distributed__ib_ring, device=device, dtype=torch.bfloat16) / (
        K__distributed__ib_ring**0.5
    )
    with torch.cuda.use_mem_pool(DistributedManager.symmetric_mempool(device)):
        recv = torch.empty((cp, Dloc, B, N_loc, N), dtype=torch.bfloat16, device=device)
    recv.fill_(-99.0)
    A3, B3 = A.permute(1, 2, 0), Bt.permute(1, 2, 0)
    D_logical = torch.as_strided(recv, (M, N, L), (N, 1, M * N))
    view = _mark if shape_mode == "dynamic" else (lambda t: from_dlpack(t, assumed_align=16))
    return A, Bt, recv, (view(A3), view(B3), view(D_logical))


def _epi_dyn(gemm, recv, ring_c, pe_dev_c):
    """EpilogueArguments for the dynamic ib_ring path: dynamic recv + the (static-shape) GMEM ring
    + the device PE table (both flat-in-N, so ONE alloc is reused across all token counts). Routed through
    the #57 Stage-1.1 dead-descriptor gate (make_differential_epi_args): on a single-node all-P2P COLLAPSE
    (has_ib_peers=False, cp<=8) the ring leaf is NULLED (collapsed store == pe_aligned, byte-identical); on a
    multi-node MIXED run (has_ib_peers=True, cp>8) the ring passes through unchanged (IB overlay live)."""
    return gemm.make_differential_epi_args(recv=_mark(recv), ring=ring_c, pe_table_dev=pe_dev_c)


def _epi_cluster(gemm, recv, stage_c, pe_dev_c, shape_mode="dynamic"):
    """EpilogueArguments for the CLUSTER-DRAIN (3.1b) path: dynamic recv + the per-cluster SYMMETRIC-heap
    staging buffer (n_clusters, epi_m, N_j_loc) + the device PE table. The staging is the IB put SOURCE
    (symmetric); one over-alloc reused across token counts (mark_layout_dynamic'd). No ring on this path.
    Routed through make_differential_epi_args: on the all-P2P COLLAPSE (has_ib_peers=False) the cluster_stage
    leaf is NULLED (collapsed store == nvdirect); on MIXED (has_ib_peers=True) it passes through unchanged."""
    _recv = _mark(recv) if shape_mode == "dynamic" else from_dlpack(recv, assumed_align=16)
    return gemm.make_differential_epi_args(recv=_recv, cluster_stage=stage_c, pe_table_dev=pe_dev_c)


def _epi_design_e(recv, pe_dev_c):
    """EpilogueArguments for the design-E COUPLED direct store (pe_aligned / nvdirect): dynamic recv + the
    device PE table, NO ring / NO cluster_stage. The per-peer TMA-S2G atoms are built internally at compile
    (from recv), so the caller passes only recv + pe_table_dev (byte-identical to the plain coupled path)."""
    return GemmA2ASm90.EpilogueArguments(
        alpha=None,
        beta=None,
        mRowVecBroadcast=None,
        mColVecBroadcast=None,
        add_to_output=False,
        rounding_mode=None,
        sr_seed=None,
        recv=_mark(recv),
        pe_table_dev=pe_dev_c,
    )


def _gate(recv, expected, cp, Dloc, B, N, tag, rank):
    got4d = recv.reshape(cp, Dloc * B, N // cp, N)
    exp4d = expected.reshape(cp, Dloc * B, N // cp, N)
    h = compute_error_histogram(got4d, exp4d)
    untouched = int((recv == -99.0).all(dim=4).sum().item())
    n_cells = cp * Dloc * B * (N // cp)
    # The element-wise gate decides; the histogram is printed alongside it because that is what
    # localizes a failure once one happens. Reaching the print means the gate already passed, so
    # there is no PASS/FAIL ternary to compute -- and recomputing a weaker pooled verdict next to a
    # stronger element-wise one would only invite a reader to trust the wrong number.
    worst = _assert_recv_matches(got4d, exp4d, bar=2e-2, what=tag)
    assert h.n_outlier_rows == 0, (
        f"{tag}: {h.n_outlier_rows} outlier row(s) (ratio {h.row_outlier_ratio:.1f}x, "
        f"worst@{h.worst_row_index}); untouched={untouched}/{n_cells}"
    )
    assert untouched == 0, f"{tag}: {untouched}/{n_cells} recv cells unwritten (coverage gap)"
    if rank == 0:
        print(
            f"\n[{tag}] worst_ratio={worst:.3f} rel_L2={h.rel_l2:.3e} "
            f"outlier_rows={h.n_outlier_rows} untouched={untouched}/{n_cells} PASS",
            flush=True,
        )


def _build_inputs_2d(device, rank, N, cp0, cp1, Dloc, B):
    """Task #13 2-D variant of _build_inputs: a fresh symmetric recv (cp,Dloc,B,N_i_loc,N_j_loc) for the
    2-D (cp0,cp1) token shard (N_i_loc=N//cp0, N_j_loc=N//cp1) + the logical (M,N,L)=(N,N,L) D view over
    its storage (the store copy_fn intercepts every write and routes it to the peer recv, so the D view
    only carries the problem shape). The recv storage == L*N*N (== the 1-D case), so the same as_strided
    (N,N,L)/(N,1,N*N) view fits."""

    cp = cp0 * cp1
    M, L, N_i_loc, N_j_loc = N, Dloc * B, N // cp0, N // cp1
    torch.manual_seed(4321 + rank)
    A = torch.randn(L, M, K__distributed__ib_ring, device=device, dtype=torch.bfloat16) / (
        K__distributed__ib_ring**0.5
    )
    Bt = torch.randn(L, N, K__distributed__ib_ring, device=device, dtype=torch.bfloat16) / (
        K__distributed__ib_ring**0.5
    )
    with torch.cuda.use_mem_pool(DistributedManager.symmetric_mempool(device)):
        recv = torch.empty((cp, Dloc, B, N_i_loc, N_j_loc), dtype=torch.bfloat16, device=device)
    recv.fill_(-99.0)
    A3, B3 = A.permute(1, 2, 0), Bt.permute(1, 2, 0)
    D_logical = torch.as_strided(recv, (M, N, L), (N, 1, M * N))
    return A, Bt, recv, (_mark(A3), _mark(B3), _mark(D_logical))


def _build_inputs_2d_kn(device, rank, N, cp0, cp1, Dloc, B):
    """K=N variant of _build_inputs_2d for the #52 cp2 TRIO: the REAL square back-TriMul einsum
    (M=N=K=N; operands O(N²) each), NOT the module K=256 correctness shortcut. Same symmetric recv +
    (M,N,L) D-view; only the contraction dim differs (K=N). At K=N large N OOMs -- caller catches it."""

    cp = cp0 * cp1
    M, L, N_i_loc, N_j_loc = N, Dloc * B, N // cp0, N // cp1
    Kn = N  # back-TriMul "lik,ljk->lij" contracts over N -> K == N (NOT the thin K=256)
    assert Kn == N, "back-TriMul einsum contracts over N; K must == N"
    torch.manual_seed(4321 + rank)
    A = torch.randn(L, M, Kn, device=device, dtype=torch.bfloat16) / (Kn**0.5)
    Bt = torch.randn(L, N, Kn, device=device, dtype=torch.bfloat16) / (Kn**0.5)
    with torch.cuda.use_mem_pool(DistributedManager.symmetric_mempool(device)):
        recv = torch.empty((cp, Dloc, B, N_i_loc, N_j_loc), dtype=torch.bfloat16, device=device)
    recv.fill_(-99.0)
    A3, B3 = A.permute(1, 2, 0), Bt.permute(1, 2, 0)
    D_logical = torch.as_strided(recv, (M, N, L), (N, 1, M * N))
    return A, Bt, recv, (_mark(A3), _mark(B3), _mark(D_logical))


def _gate_2d(recv, expected, cp, Dloc, B, N_i_loc, N_j_loc, tag, rank):
    """Task #13 2-D gate: same per-row L2 + 0-outlier + 0-untouched contract as _gate, over the 2-D recv
    (cp,Dloc,B,N_i_loc,N_j_loc). A mis-routed (i,j)->peer / mis-addressed j-column shows as an outlier
    row (n_outlier_rows>0) or an unwritten cell (untouched>0), vs the _ref_back_gemm_native_2d oracle."""
    got4d = recv.reshape(cp, Dloc * B, N_i_loc, N_j_loc)
    exp4d = expected.reshape(cp, Dloc * B, N_i_loc, N_j_loc)
    h = compute_error_histogram(got4d, exp4d)
    untouched = int((recv == -99.0).all(dim=4).sum().item())
    n_cells = cp * Dloc * B * N_i_loc
    worst = _assert_recv_matches(got4d, exp4d, bar=2e-2, what=tag)
    assert h.n_outlier_rows == 0, (
        f"{tag}: {h.n_outlier_rows} outlier row(s) (ratio {h.row_outlier_ratio:.1f}x, "
        f"worst@{h.worst_row_index}); untouched={untouched}/{n_cells}"
    )
    assert untouched == 0, f"{tag}: {untouched}/{n_cells} recv cells unwritten (coverage gap)"
    if rank == 0:
        print(
            f"\n[{tag}] worst_ratio={worst:.3f} rel_L2={h.rel_l2:.3e} "
            f"outlier_rows={h.n_outlier_rows} untouched={untouched}/{n_cells} PASS",
            flush=True,
        )


@pytest.mark.usefixtures("_nvshmem__distributed__ib_ring")
@pytest.mark.usefixtures("_module_skip__distributed__ib_ring")
@numeric_exempt(
    "COMPILE-CHECK only: the drain signal is STUBBED, so the kernel is configured and compiled but "
    "never launched and no recv is written. The single assertion is `compiled is not None` -- there "
    "is no output to compare per element, and a sanctioned assertion here could only be made by "
    "inventing a run this test deliberately does not do. The gated run is its sibling "
    "test_ib_ring_cluster_drain_2d. Surfaced only at WORLD_SIZE=16: the 2-D specs cp=(2,8)/(4,4) are "
    "the first meshes at which this cell is not rank-skipped, so no smaller launch could reach it"
)
@pytest.mark.parametrize("cluster_n", [2], ids=["cn2"])
@pytest.mark.parametrize("rd", [2], ids=["rd2"])
@pytest.mark.parametrize("dbmw", [(False, False)], ids=["sb"])
@EPI.parametrize("mesh")
def test_ib_ring_cluster_drain_2d_compile(
    apply_mesh, mesh, dist_manager, device, world_size, rd, cluster_n, dbmw
):
    """CLUSTER-DRAIN (3.1b) COMPILE-CHECK: configure + compile the cluster-cooperative bounded-band
    store over a 2-D (cp0,cp1) shard and assert it compiles clean. Signal STUBBED (probe #2) -> no run
    / no gate yet. Mirrors test_ib_ring_coalesce_dynamic_2d's host setup (dynamic anchor compile) but
    with an N-axis cluster (1, cluster_n, 1), cluster_drain=True + coalesce=False, and the per-cluster
    symheap staging (n_clusters, epi_m, N_j_loc) on the epilogue params."""
    apply_mesh(mesh)

    assert get_device_capacity()[0] == 9, "SM90 only"
    axis_sizes = _cp_axis_sizes(dist_manager)
    if len(axis_sizes) != 2:
        rank_invariant_skip(
            f"cluster_drain 2-D needs a 2-D cp mesh (CPO_DIST_MESH=cp=2*4/2*8); got {axis_sizes}",
            because=_GEOMETRY_UNIFORM_BECAUSE,
        )
    cp0, cp1 = axis_sizes
    db, mw = dbmw  # SPIKE (Phase 5): double-buffer + multi-warp toggles
    cp = cp0 * cp1
    Dloc, B = 2, 1
    run_ns = [
        N
        for N in DIFF2D_NS
        if N % cp0 == 0
        and N % cp1 == 0
        and (N // cp0) % 8 == 0
        and (N // cp1) % 8 == 0
        and (N // cp0) % TILE[0] != 0
        and ((N // cp1 + TILE[1] - 1) // TILE[1]) % cluster_n == 0
    ]
    if not run_ns:
        rank_invariant_skip(
            f"no 2-D even-shard straddle shapes for cp=({cp0},{cp1}) cn={cluster_n} in {DIFF2D_NS}",
            because=_GEOMETRY_UNIFORM_BECAUSE,
        )
    anchor = run_ns[0]
    pm = _trimul_placements(dist_manager)[1]
    pe_table = tuple(int(x) for x in pm.cp_pe_table.tolist())
    stream = cutlass_torch.current_stream()
    a_dtype = torch2cute_dtype_map[torch.bfloat16]
    # grid.z (num persistent clusters) <= get_max_active_clusters(cluster_n); the staging is indexed by
    # the physical cluster id bidx=block_idx()[0]+[2] in [0, grid.z) -> over-alloc to the max is safe.
    n_clusters = get_max_active_clusters(cluster_n)
    sched = make_scheduler_args(n_clusters, Int32(8), None, None)
    size_Nj = anchor // cp1
    # per-cluster (n_clusters, epi_m=128, N_j_loc) staging on the SYMMETRIC heap (the IB put SOURCE).
    with torch.cuda.use_mem_pool(DistributedManager.symmetric_mempool(device)):
        stage_buf = torch.empty(
            (n_clusters, rd, 128, size_Nj) if db else (n_clusters, 128, size_Nj),
            dtype=torch.bfloat16,
            device=device,
        )
    stage_buf.zero_()
    pe_dev_t = pm.cp_pe_table.to(torch.int32).contiguous()
    pe_dev_c = from_dlpack(pe_dev_t, assumed_align=4)
    stage_c = from_dlpack(stage_buf, assumed_align=16).mark_layout_dynamic()

    A0, Bt0, recv0, (cA0, cB0, cD0) = _build_inputs_2d(
        device, dist_manager.rank, anchor, cp0, cp1, Dloc, B
    )
    gemm = GemmA2ASm90(
        Float32, a_dtype, TILE, (1, cluster_n, 1), pingpong=False, is_persistent=True
    )
    gemm.configure_a2a_gemm_native(
        cp=cp,
        my_cp_rank=int(pm.my_cp_rank),
        B=B,
        N_loc=anchor // cp0,
        pe_table=pe_table,
        N=anchor,
        cp_axis_sizes=(cp0, cp1),
        dynamic=True,
        arbitrary_n=True,
        pe_aligned_tiling=True,
        ib_ring=True,
        ib_quiet=True,
        decoupled=True,
        producer_tma=True,
        consumer_strided=True,
        consumer_strided_putwarp=True,
        ring_depth=rd,
        cluster_drain=True,
        cluster_n=cluster_n,
    )
    try:
        compiled = compile_gemm_with_bitcode(
            gemm,
            cA0,
            cB0,
            cD0,
            None,
            _epi_cluster(gemm, recv0, stage_c, pe_dev_c),
            sched,
            stream,
            None,
            register=True,
        )
        assert compiled is not None, "cluster_drain 2-D compile returned None"
        if dist_manager.rank == 0:
            print(
                f"\n[cluster_drain 2-D COMPILE cp=({cp0},{cp1}) cn={cluster_n} rd={rd} @N={anchor} "
                f"run_j={gemm._a2a_cluster_run_j}] compile-clean PASS (signal STUBBED -> no run/gate)",
                flush=True,
            )
        compiled.free()
    finally:
        _nvshmem_barrier()


# --------------------------------------------------------------------------- #
# CLUSTER-DRAIN (Phase-3.1b) CORRECTNESS — the active cross-CTA mbarrier signal . The
# cluster_n CTAs cooperatively stage a peer_j width into the shared symheap buffer; cluster-rank-0
# waits its 'full' mbar (count cluster_n), drains with the coalesced put_warp, then arrives each CTA's
# 'empty' reuse gate. Anchor is an EVEN-divisibility straddle (nt_j_pp % cluster_n == 0 so
# runs_per_band=cp1; N_i_loc % 128 != 0 off-grid). Gated vs the fp32 2-D reshard oracle. STATIC run_j
# (3.1a) -> single-N (dynamic-N bounded band = 3.1c). A hang here = a signal bug (run under timeout).
# --------------------------------------------------------------------------- #
@pytest.mark.usefixtures("_nvshmem__distributed__ib_ring")
@pytest.mark.usefixtures("_module_skip__distributed__ib_ring")
@pytest.mark.parametrize("cluster_n", [2], ids=["cn2"])
@pytest.mark.parametrize("rd", [2], ids=["rd2"])
# SPIKE (Phase 5): A/B the single-buffer (sb) baseline vs double-buffer (db) vs double-buffer+multi-warp
# (dbmw). db allocs a 2x (rd-slot) staging; dbmw additionally splits rank-0's drain across n_consumer_warps.
@pytest.mark.parametrize("dbmw", [(False, False)], ids=["sb"])
@EPI.parametrize("mesh")
def test_ib_ring_cluster_drain_2d(
    apply_mesh, mesh, dist_manager, device, world_size, rd, cluster_n, dbmw
):
    """CLUSTER-DRAIN (3.1b) rung-1 correctness: configure + compile + run the cluster-cooperative
    bounded-band store over a 2-D (cp0,cp1) shard at an even-divisibility straddle anchor, and gate the
    recv vs _ref_back_gemm_native_2d (per-row L2<2e-2, 0 outlier/untouched). Exercises the active
    cross-CTA mbarrier signal + coalesced put drain. Off-grid (N_i_loc%128!=0) is the anchor itself."""
    apply_mesh(mesh)

    assert get_device_capacity()[0] == 9, "SM90 only"
    axis_sizes = _cp_axis_sizes(dist_manager)
    if len(axis_sizes) != 2:
        rank_invariant_skip(
            f"cluster_drain 2-D needs a 2-D cp mesh (CPO_DIST_MESH=cp=2*4/2*8); got {axis_sizes}",
            because=_GEOMETRY_UNIFORM_BECAUSE,
        )
    cp0, cp1 = axis_sizes
    db, mw = dbmw  # SPIKE (Phase 5): double-buffer + multi-warp toggles
    cp = cp0 * cp1
    Dloc, B = 2, 1
    D = Dloc * cp

    def _ok(N):
        Ni, Nj = N // cp0, N // cp1
        ntj = (Nj + TILE[1] - 1) // TILE[1]
        return (
            N % cp0 == 0
            and N % cp1 == 0
            and Ni % 8 == 0
            and Nj % 8 == 0
            and Ni % TILE[0] != 0
            and ntj % cluster_n == 0
        )

    run_ns = [N for N in DIFF2D_NS if _ok(N)]
    if not run_ns:
        rank_invariant_skip(
            f"no even-divisibility 2-D straddle shapes for cp=({cp0},{cp1}) cn={cluster_n}",
            because=_GEOMETRY_UNIFORM_BECAUSE,
        )
    anchor = run_ns[0]
    pm = _trimul_placements(dist_manager)[1]
    pe_table = tuple(int(x) for x in pm.cp_pe_table.tolist())
    stream = cutlass_torch.current_stream()
    a_dtype = torch2cute_dtype_map[torch.bfloat16]
    n_clusters = get_max_active_clusters(cluster_n)
    sched = make_scheduler_args(n_clusters, Int32(8), None, None)
    size_Nj = anchor // cp1
    with torch.cuda.use_mem_pool(DistributedManager.symmetric_mempool(device)):
        stage_buf = torch.empty(
            (n_clusters, rd, 128, size_Nj) if db else (n_clusters, 128, size_Nj),
            dtype=torch.bfloat16,
            device=device,
        )
    stage_buf.zero_()
    pe_dev_t = pm.cp_pe_table.to(torch.int32).contiguous()
    pe_dev_c = from_dlpack(pe_dev_t, assumed_align=4)
    stage_c = from_dlpack(stage_buf, assumed_align=16).mark_layout_dynamic()

    A0, Bt0, recv0, (cA0, cB0, cD0) = _build_inputs_2d(
        device, dist_manager.rank, anchor, cp0, cp1, Dloc, B
    )
    gemm = GemmA2ASm90(
        Float32, a_dtype, TILE, (1, cluster_n, 1), pingpong=False, is_persistent=True
    )
    gemm.configure_a2a_gemm_native(
        cp=cp,
        my_cp_rank=int(pm.my_cp_rank),
        B=B,
        N_loc=anchor // cp0,
        pe_table=pe_table,
        N=anchor,
        cp_axis_sizes=(cp0, cp1),
        dynamic=True,
        arbitrary_n=True,
        pe_aligned_tiling=True,
        ib_ring=True,
        ib_quiet=True,
        decoupled=True,
        producer_tma=True,
        consumer_strided=True,
        consumer_strided_putwarp=True,
        ring_depth=rd,
        cluster_drain=True,
        cluster_n=cluster_n,
    )
    compiled = compile_gemm_with_bitcode(
        gemm,
        cA0,
        cB0,
        cD0,
        None,
        _epi_cluster(gemm, recv0, stage_c, pe_dev_c),
        sched,
        stream,
        None,
        register=True,
    )
    try:
        N = anchor
        A, Bt, recv, (cA, cB, cD) = _build_inputs_2d(
            device, dist_manager.rank, N, cp0, cp1, Dloc, B
        )
        N_i_loc, N_j_loc = N // cp0, N // cp1
        _nvshmem_barrier()
        compiled(cA, cB, cD, None, _epi_cluster(gemm, recv, stage_c, pe_dev_c), sched, stream, None)
        _drain()
        exp = _ref_back_gemm_native_2d(
            A,
            Bt,
            pm,
            cp0=cp0,
            cp1=cp1,
            Dloc=Dloc,
            B=B,
            N_i_loc=N_i_loc,
            N_j_loc=N_j_loc,
            N=N,
            world_size=world_size,
        )
        tag = (
            f"ib_ring CLUSTER-DRAIN cp=({cp0},{cp1}) cn={cluster_n} D={D} rd={rd} @N={N} "
            f"(straddle N_i={N_i_loc} N_j={N_j_loc} run_j={gemm._a2a_cluster_run_j})"
        )
        _gate_2d(recv, exp, cp, Dloc, B, N_i_loc, N_j_loc, tag, dist_manager.rank)
    finally:
        compiled.free()
        _nvshmem_barrier()


# --------------------------------------------------------------------------- #
# CLUSTER-DRAIN 1-D (cp1==1) correctness — the KEEP-path 1-D IB drain the drain cut must RETAIN (FIRST
# PRINCIPLE: every supported shape hardened by pytest; the cut removed the only 1-D tests). Only the
# i / GEMM-M axis is cp-split; j is FULL-N (N_j_loc == N), so the drain's 2-D routing (const_expr(cp1>1))
# takes the 1-D arm. The anchor is a STRADDLE N_loc (%128!=0) so arbitrary_n stays on (an ALIGNED N_loc
# auto-reduces it off -> pe_aligned/cluster_drain would not engage). nt_j_pp is off the FULL N (N_j_loc==N).
# VERIFIED host-side (cp1==1, sm_120): cluster_drain configures CLEAN for cluster_n in {1,2} x multislot in
# {False,True} (no 2-D-only assumption). Run under a FLAT cp mesh (CPO_DIST_MESH=cp=16 -> cross-node IB;
# single-node cp<=8 collapses to the NVLink coupled store). Gated vs the fp32 1-D reshard oracle
# (rel_L2<2e-2, 0 outlier/untouched). A hang here = a producer<->drain mbar desync (run under timeout).
# --------------------------------------------------------------------------- #
# 1-D straddle candidates: N_loc=N//cp is %8 (16-B) but %128!=0 (arbitrary_n on); for cluster_n>1 the
# even-shard gate needs ceil(N/tile_n)%cluster_n==0 (nt_j_pp off the FULL N). Filtered per (cp, cluster_n).
DIFF1D_NS = [2176, 4352, 5376, 6400, 8448]
# LARGE 1-D straddle candidates for the MULTIBAND cell: bigger N_i_loc so nt_i_pp>=5 -> total_bands =
# cp*nt_i_pp*(Dloc*B) EXCEEDS n_clusters=get_max_active_clusters(cn) at cp16 (cn1~132, cn2~66) -> a physical
# cluster drains >1 band. That is the regime the 1-D fix restores: the producer must fire the per-band
# handshake at a band->band boundary where peer_j can't change (peer_j==0 at cp1==1). cn1's DEFAULT small
# anchor (2176 -> total_bands=64<132) is single-band-per-cluster, so it never exercised this pre-fix. All
# are even-ntj straddles (ntj%2==0) so they pass _ok for BOTH cn1 and cn2. Filtered dynamically by _multiband.
DIFF1D_NS_BIG = [8448, 10752, 12800, 16896]


def _nloc8_anchor(cp):
    """1-D straddle anchor with the per-peer i-extent N_loc = N//cp %8 != 0 -- the EXACT boundary the
    N=1000/cp2 e2e failure sits on (N_loc=500, %8=4). FIRST-PRINCIPLE legit (NOT a dropped shape): the
    recv store axis is the col N (== N_j_loc at cp1==1) which stays 16-B (N %8==0), so every coalesced
    put is aligned regardless of N_i_loc%8; the a_major="k" A-load rides the STRIDED M axis (M-stride=K,
    %8==0). Prefers N=1000 (the e2e shape, valid for cp in {2,4}); else the smallest N with N%cp==0,
    N%8==0 (recv col), N_loc%8 != 0, N_loc%128 != 0 (arbitrary_n straddle), N_loc>128 (multi-tile per
    peer, a clean single-boundary ceil-spill like the e2e). None if unreachable at this cp."""
    for N in [1000] + list(range(8, 8192, 8)):
        if N % cp == 0 and N % 8 == 0:
            nl = N // cp
            if nl % 8 != 0 and nl % TILE[0] != 0 and nl > TILE[0]:
                return N
    return None


@pytest.mark.usefixtures("_nvshmem__distributed__ib_ring")
@pytest.mark.usefixtures("_module_skip__distributed__ib_ring")
@pytest.mark.parametrize("bands", [False, True], ids=["b1", "multiband"])
@pytest.mark.parametrize("ms", [False, True], ids=["sb", "multislot"])
@pytest.mark.parametrize("cluster_n", [1, 2], ids=["cn1", "cn2"])
@pytest.mark.parametrize("rd", [2], ids=["rd2"])
@EPI.parametrize("mesh")
def test_ib_ring_cluster_drain_1d(
    apply_mesh, mesh, dist_manager, device, world_size, rd, cluster_n, ms, bands
):
    """CLUSTER-DRAIN 1-D (cp1==1) correctness: configure + compile + run the cluster-cooperative store over
    a FLAT (i-only) cp shard at a straddle anchor (N_loc%128!=0 so arbitrary_n stays on), for the base
    even-shard drain (ms=False) AND the degenerate one-peer_j multislot (ms=True), gated vs
    _ref_back_gemm_native (per-row L2<2e-2, 0 outlier/untouched). Off-grid (N_loc%128!=0) is the anchor.

    bands: b1 = the default small anchor (<=1 band per physical cluster). multiband = a LARGE-N anchor
    (DIFF1D_NS_BIG) forcing total_bands=cp*nt_i_pp*(Dloc*B) > n_clusters so a physical cluster drains >1
    band -> exercises the per-band producer handshake at a band->band boundary where peer_j can't change
    (the cn2-multislot 1-D fix's core mechanism; also hardens cn1, whose default anchor is single-band)."""
    apply_mesh(mesh)

    assert get_device_capacity()[0] == 9, "SM90 only"
    axis_sizes = _cp_axis_sizes(dist_manager)
    if len(axis_sizes) != 1:
        rank_invariant_skip(
            f"cluster_drain 1-D needs a FLAT cp mesh (CPO_DIST_MESH=cp=16); got {axis_sizes}",
            because=_GEOMETRY_UNIFORM_BECAUSE,
        )
    cp = axis_sizes[0]
    Dloc, B = 2, 1
    D = Dloc * cp

    def _ok(N):
        N_loc = N // cp
        ntj = (N + TILE[1] - 1) // TILE[1]  # nt_j_pp off the FULL N (N_j_loc == N at cp1==1)
        # 16-B (%8) constraint is on the recv STORE col N (== N_j_loc at cp1==1), NOT on N_loc: the
        # coalesced put runs along N_j_loc so every row start is 16-B aligned regardless of N_i_loc%8, and
        # the a_major="k" A-load rides the STRIDED M axis (M-stride=K, %8). The old `N_loc%8==0` gate was a
        # latent FIRST-PRINCIPLE violation (it skipped an in-principle-supported shape) that HID the N=1000
        # e2e boundary; N_loc%8!=0 is covered by test_ib_ring_cluster_drain_1d_nloc8_straddle below.
        return N % cp == 0 and N % 8 == 0 and N_loc % TILE[0] != 0 and ntj % cluster_n == 0

    n_clusters = get_max_active_clusters(cluster_n)

    def _multiband(N):
        # total_bands = cp*nt_i_pp*(Dloc*B) > n_clusters -> a physical cluster drains >1 band (the 1-D fix's
        # core regime: the producer must fire the per-band handshake at a band->band boundary where peer_j
        # can't change). cn1's DEFAULT small anchor is single-band; this forces the multi-band path for cn1.
        nt_i_pp = (N // cp + TILE[0] - 1) // TILE[0]
        return cp * nt_i_pp * (Dloc * B) > n_clusters

    if bands:
        cand = [N for N in DIFF1D_NS_BIG if _ok(N) and _multiband(N)]
        if not cand:
            rank_invariant_skip(
                f"no 1-D multiband anchor (total_bands>{n_clusters}) for cp={cp} cn={cluster_n}",
                because=_GEOMETRY_UNIFORM_BECAUSE,
            )
        anchor = cand[0]
        if dist_manager.rank == 0:  # self-document the regime in the gate log
            _ntip = (anchor // cp + TILE[0] - 1) // TILE[0]
            print(
                f"\n[1D MULTIBAND cp={cp} cn={cluster_n} ms={ms}] N={anchor} nt_i_pp={_ntip} "
                f"total_bands={cp * _ntip * (Dloc * B)} > n_clusters={n_clusters} (>1 band/cluster)",
                flush=True,
            )
    else:
        run_ns = [N for N in DIFF1D_NS if _ok(N)]
        if not run_ns:
            rank_invariant_skip(
                f"no 1-D straddle shapes for cp={cp} cn={cluster_n} in {DIFF1D_NS}",
                because=_GEOMETRY_UNIFORM_BECAUSE,
            )
        anchor = run_ns[0]
    pm = _pe_map(dist_manager)
    pe_table = tuple(int(x) for x in pm.cp_pe_table.tolist())
    stream = cutlass_torch.current_stream()
    a_dtype = torch2cute_dtype_map[torch.bfloat16]
    sched = make_scheduler_args(n_clusters, Int32(8), None, None)
    # 1-D: N_j_loc == N (j unsharded). SB drain -> (n_clusters, epi_m=128, N); multislot adds the rd
    # ring-slot axis -> (n_clusters, rd, epi_m=128, N), mirroring the 2D test's `if db` staging.
    with torch.cuda.use_mem_pool(DistributedManager.symmetric_mempool(device)):
        stage_buf = torch.empty(
            (n_clusters, rd, 128, anchor) if ms else (n_clusters, 128, anchor),
            dtype=torch.bfloat16,
            device=device,
        )
    stage_buf.zero_()
    pe_dev_t = pm.cp_pe_table.to(torch.int32).contiguous()
    pe_dev_c = from_dlpack(pe_dev_t, assumed_align=4)
    stage_c = from_dlpack(stage_buf, assumed_align=16).mark_layout_dynamic()

    A0, Bt0, recv0, (cA0, cB0, cD0) = _build_inputs(device, dist_manager.rank, anchor, cp, Dloc, B)
    gemm = GemmA2ASm90(
        Float32, a_dtype, TILE, (1, cluster_n, 1), pingpong=False, is_persistent=True
    )
    gemm.configure_a2a_gemm_native(
        cp=cp,
        my_cp_rank=int(pm.my_cp_rank),
        B=B,
        N_loc=anchor // cp,
        pe_table=pe_table,
        N=anchor,
        dynamic=True,
        arbitrary_n=True,
        pe_aligned_tiling=True,
        ib_ring=True,
        ib_quiet=True,
        decoupled=True,
        producer_tma=True,
        consumer_strided=True,
        consumer_strided_putwarp=True,
        ring_depth=rd,
        cluster_drain=True,
        cluster_n=cluster_n,
        cluster_multislot=ms,
    )
    compiled = compile_gemm_with_bitcode(
        gemm,
        cA0,
        cB0,
        cD0,
        None,
        _epi_cluster(gemm, recv0, stage_c, pe_dev_c),
        sched,
        stream,
        None,
        register=True,
    )
    try:
        N = anchor
        A, Bt, recv, (cA, cB, cD) = _build_inputs(device, dist_manager.rank, N, cp, Dloc, B)
        N_loc = N // cp
        _nvshmem_barrier()
        compiled(cA, cB, cD, None, _epi_cluster(gemm, recv, stage_c, pe_dev_c), sched, stream, None)
        _drain()
        exp = _ref_back_gemm_native(
            A,
            Bt,
            pm,
            cp=cp,
            Dloc=Dloc,
            B=B,
            N_loc=N_loc,
            N=N,
            world_size=world_size,
        )
        tag = (
            f"ib_ring CLUSTER-DRAIN 1D cp={cp} cn={cluster_n} ms={ms} D={D} rd={rd} @N={N} "
            f"(straddle N_loc={N_loc} run_j={gemm._a2a_cluster_run_j})"
        )
        _gate(recv, exp, cp, Dloc, B, N, tag, dist_manager.rank)
    finally:
        compiled.free()
        _nvshmem_barrier()


# --------------------------------------------------------------------------- #
# CLUSTER-DRAIN 1-D cluster_multislot at N_loc%8 != 0 — the EXACT N=1000/cp2 e2e boundary, in ISOLATION.
# This is the regression lock the old test_ib_ring_cluster_drain_1d._ok `N_loc%8==0` gate HID (a latent
# FIRST-PRINCIPLE violation: it skipped an in-principle-supported shape). It exercises the back store DIRECTLY
# on SYNTHETIC operands (_build_inputs, a_major="k"), so a PASS here isolates the back drain as CORRECT at
# N_loc%8!=0 (pointing any residual e2e failure UPSTREAM of the store); a FAIL here is the isolated back-store
# repro. Cross-node (cp2 on venue F = 2 nodes) engages the real IB drain. cn in {1,2} = the shipped multislot set.
# --------------------------------------------------------------------------- #
@pytest.mark.usefixtures("_nvshmem__distributed__ib_ring")
@pytest.mark.usefixtures("_module_skip__distributed__ib_ring")
@pytest.mark.parametrize("cluster_n", [1, 2], ids=["cn1", "cn2"])
@pytest.mark.parametrize("rd", [2], ids=["rd2"])
@EPI.parametrize("mesh")
def test_ib_ring_cluster_drain_1d_nloc8_straddle(
    apply_mesh, mesh, dist_manager, device, world_size, rd, cluster_n
):
    """cluster_multislot 1-D at N_loc = N//cp %8 != 0 (N=1000/cp2 -> N_loc=500), gated vs
    _ref_back_gemm_native (per-row L2<2e-2, 0 outlier/untouched). N_loc%8!=0 is 16-B-LEGIT: the store col
    N (=N_j_loc) is %8 so every coalesced put is aligned; the a_major="k" A-load rides the strided M axis
    (M-stride=K, %8). This shape was UNTESTED in isolation (the old _ok N_loc%8==0 gate skipped it) -> the
    N=1000 e2e is its first hit. A PASS EXONERATES the back store (bug is upstream); a FAIL is the repro."""
    apply_mesh(mesh)

    assert get_device_capacity()[0] == 9, "SM90 only"
    axis_sizes = _cp_axis_sizes(dist_manager)
    if len(axis_sizes) != 1:
        rank_invariant_skip(
            f"cluster_drain 1-D needs a FLAT cp mesh (e.g. CPO_DIST_MESH=cp=2); got {axis_sizes}",
            because=_GEOMETRY_UNIFORM_BECAUSE,
        )
    cp = axis_sizes[0]
    Dloc, B = 2, 1
    D = Dloc * cp
    anchor = _nloc8_anchor(cp)
    if anchor is None:
        rank_invariant_skip(
            f"no N_loc%8!=0 straddle anchor reachable for cp={cp}",
            because=_GEOMETRY_UNIFORM_BECAUSE,
        )
    N_loc = anchor // cp
    assert N_loc % 8 != 0 and N_loc % TILE[0] != 0 and anchor % 8 == 0, (
        f"bad N_loc%8 anchor {anchor} for cp={cp}: N_loc={N_loc} (%8={N_loc % 8}, %128={N_loc % TILE[0]})"
    )
    n_clusters = get_max_active_clusters(cluster_n)
    if dist_manager.rank == 0:
        print(
            f"\n[1D NLOC8-STRADDLE cp={cp} cn={cluster_n}] N={anchor} N_loc={N_loc} "
            f"(N_loc%8={N_loc % 8}, N_loc%128={N_loc % TILE[0]}, N%8={anchor % 8})",
            flush=True,
        )
    pm = _pe_map(dist_manager)
    pe_table = tuple(int(x) for x in pm.cp_pe_table.tolist())
    stream = cutlass_torch.current_stream()
    a_dtype = torch2cute_dtype_map[torch.bfloat16]
    sched = make_scheduler_args(n_clusters, Int32(8), None, None)
    with torch.cuda.use_mem_pool(DistributedManager.symmetric_mempool(device)):
        stage_buf = torch.empty((n_clusters, rd, 128, anchor), dtype=torch.bfloat16, device=device)
    stage_buf.zero_()
    pe_dev_t = pm.cp_pe_table.to(torch.int32).contiguous()
    pe_dev_c = from_dlpack(pe_dev_t, assumed_align=4)
    stage_c = from_dlpack(stage_buf, assumed_align=16).mark_layout_dynamic()

    A0, Bt0, recv0, (cA0, cB0, cD0) = _build_inputs(device, dist_manager.rank, anchor, cp, Dloc, B)
    gemm = GemmA2ASm90(
        Float32, a_dtype, TILE, (1, cluster_n, 1), pingpong=False, is_persistent=True
    )
    gemm.configure_a2a_gemm_native(
        cp=cp,
        my_cp_rank=int(pm.my_cp_rank),
        B=B,
        N_loc=anchor // cp,
        pe_table=pe_table,
        N=anchor,
        dynamic=True,
        arbitrary_n=True,
        pe_aligned_tiling=True,
        ib_ring=True,
        ib_quiet=True,
        decoupled=True,
        producer_tma=True,
        consumer_strided=True,
        consumer_strided_putwarp=True,
        ring_depth=rd,
        cluster_drain=True,
        cluster_n=cluster_n,
        cluster_multislot=True,
    )
    compiled = compile_gemm_with_bitcode(
        gemm,
        cA0,
        cB0,
        cD0,
        None,
        _epi_cluster(gemm, recv0, stage_c, pe_dev_c),
        sched,
        stream,
        None,
        register=True,
    )
    try:
        N = anchor
        A, Bt, recv, (cA, cB, cD) = _build_inputs(device, dist_manager.rank, N, cp, Dloc, B)
        _nvshmem_barrier()
        compiled(cA, cB, cD, None, _epi_cluster(gemm, recv, stage_c, pe_dev_c), sched, stream, None)
        _drain()
        exp = _ref_back_gemm_native(
            A,
            Bt,
            pm,
            cp=cp,
            Dloc=Dloc,
            B=B,
            N_loc=N_loc,
            N=N,
            world_size=world_size,
        )
        tag = (
            f"ib_ring CLUSTER-DRAIN 1D NLOC8 cp={cp} cn={cluster_n} D={D} rd={rd} @N={N} "
            f"(N_loc={N_loc} %8={N_loc % 8} straddle run_j={gemm._a2a_cluster_run_j})"
        )
        _gate(recv, exp, cp, Dloc, B, N, tag, dist_manager.rank)
    finally:
        compiled.free()
        _nvshmem_barrier()


# --------------------------------------------------------------------------- #
# CLUSTER-DRAIN #78 MULTI-EPI-SUBTILE (tile_m=256 -> m_sub_per_tile=2) — RUNG-1. The even-shard SINGLE-BUFFER
# cluster_drain at tile_m=256 stages the 2 epi M-subtile bands (rows 0-127, 128-255 of each 256-row CTA tile)
# into an m_sub axis of the (n_clusters, m_sub, epi_m=128, N_j_loc) staging, and cluster-rank-0's drain
# INNER-LOOPS the 2 bands (gi_base_s = tile_base + sub_m*128). The run-count / scheduler / mbar counts+phases
# are UNCHANGED vs tile_m=128 (the sub_m axis lives BELOW scheduler tile granularity). COVERAGE INSTRUMENT:
# pre-#78 subtile-1 (rows 128-255) was NEVER drained (untouched ~= half the M-rows, rel_L2~2240); the fix
# drives untouched -> 0. cn1 = degenerate 1-CTA cluster (producer + drain warps in the SAME CTA, no cross-CTA
# mbarrier) = the simplest handshake; cn2 exercises the proven cross-CTA even-shard signal + the subtile axis
# together. cp=(2,8) cross-node. ON-GRID M (N%512==0 -> N_i_loc%tile_m==0) so no i-peer-boundary straddle
# (straddle + cn4/cn8 are rung-2). A hang here = a producer<->drain mbar desync (run under timeout).
# --------------------------------------------------------------------------- #
TILE256 = (256, 128)  # #78 tile_m=256 -> m_sub_per_tile = 256//128 = 2


@pytest.mark.usefixtures("_nvshmem__distributed__ib_ring")
@pytest.mark.usefixtures("_module_skip__distributed__ib_ring")
@EPI.parametrize(
    "N",
    "tile",
    "mesh",
    cells=[
        (n, t, m)
        for (n, t) in (
            (2048, (256, 128)),
            (8192, (256, 128)),
            (2048, (256, 256)),
            (8192, (256, 256)),
        )
        for m in EPI.axis("mesh").values
        if not _misaligned_2d(n, m)
    ],
    because=(
        "tile_M is pinned at 256 because that IS the subject -- this is the multi-epi-subtile "
        "drain, which tile_M=128 does not reach at all. Both tile_N values are kept: 256 drives "
        "the 3-warpgroup producer-warp drain, so (256, 256) COMPOUNDS the m_sub row-walk with it "
        "while (256, 128) validates the row-walk alone. " + _JOINT_MESH_BECAUSE
    ),
)
@pytest.mark.parametrize(
    "cluster_n", [1, 2, 4], ids=["cn1", "cn2", "cn4"]
)  # cn8 now RAISES (never-deploy
# hyper-param guard in configure_a2a_gemm_native: cluster_n>=8 over-concentrates, never wins the perf curve).
def test_ib_ring_cluster_drain_2d_tile256(
    apply_mesh, mesh, dist_manager, device, world_size, cluster_n, N, tile
):
    """#78 RUNG-1/2(even): even-shard single-buffer cluster_drain at tile_m=256 (m_sub_per_tile=2) over a 2-D
    (cp0,cp1) shard, gated vs _ref_back_gemm_native_2d (per-row L2<2e-2, 0 outlier/UNTOUCHED). The untouched
    count is the coverage instrument: it flips from ~half (subtile-1 undrained, pre-#78) to 0. cn1..8 all use
    the SAME even-shard drain (my A' sub_m axis is cluster_n-agnostic); cn>=4 needs nt_j_pp % cluster_n == 0
    (else it's a STRADDLE -> multislot/roundpark, rung-2, skipped here). Needs cp=(2,8).

    #78 rung-3: `tile` sweeps tile_n — m256n256 validates the tile_n=256 3-WG producer-warp drain
    ([[reference_a2a_4wg_tile256_register_wall]]) COMPOUNDED with the even-shard m_sub=2 sub_m axis. The
    per-cluster staging N_j width (size_Nj=N//cp1) is tile_n-independent; tile_n changes nt_j_pp (the
    even-shard j-slice count) + the host-computed 3-WG warp layout only."""
    apply_mesh(mesh)

    assert get_device_capacity()[0] == 9, "SM90 only"
    axis_sizes = _cp_axis_sizes(dist_manager)
    if len(axis_sizes) != 2:
        rank_invariant_skip(
            f"#78 tile256 needs a 2-D cp mesh (CPO_DIST_MESH=cp=2*8); got {axis_sizes}",
            because=_GEOMETRY_UNIFORM_BECAUSE,
        )
    cp0, cp1 = axis_sizes
    if (cp0, cp1) != (2, 8):
        rank_invariant_skip(
            f"#78 rung-1 pinned to cp=(2,8); got ({cp0},{cp1})", because=_GEOMETRY_UNIFORM_BECAUSE
        )
    cp = cp0 * cp1
    Dloc, B = 2, 1
    D = Dloc * cp
    rd = 2
    m_sub = tile[0] // 128  # == 2
    N = N
    # ON-GRID M for tile_m=256: N_i_loc=N//cp0 % 256 == 0 (N%512==0) so no i-peer-boundary straddle. cn2 even-
    # shard also needs nt_j_pp % cluster_n == 0 (auto: nt_j_pp = ceil((N//cp1)/tile_n) is even for these Ns).
    assert N % 512 == 0 and (N // cp1) % 8 == 0, f"#78 rung-1 wants on-grid N%512==0; got {N}"
    # EVEN-SHARD requires nt_j_pp % cluster_n == 0 (else the cluster STRADDLES a peer_j boundary -> that is
    # the multislot/roundpark path = rung-2, not this even-shard test). Skip the straddle cells here.
    nt_j_pp = (N // cp1 + tile[1] - 1) // tile[1]  # tile_n -> j-tiles per peer_j
    if nt_j_pp % cluster_n != 0:
        rank_invariant_skip(
            f"even-shard cn={cluster_n} needs nt_j_pp({nt_j_pp}) % cluster_n == 0; "
            f"@N={N} nt_j_pp={nt_j_pp} is a STRADDLE (multislot/roundpark = rung-2)",
            because=_GEOMETRY_UNIFORM_BECAUSE,
        )
    pm = _trimul_placements(dist_manager)[1]
    pe_table = tuple(int(x) for x in pm.cp_pe_table.tolist())
    stream = cutlass_torch.current_stream()
    a_dtype = torch2cute_dtype_map[torch.bfloat16]
    n_clusters = get_max_active_clusters(cluster_n)
    sched = make_scheduler_args(n_clusters, Int32(8), None, None)
    size_Nj = N // cp1
    # #78 the per-cluster staging gains the m_sub axis: (n_clusters, m_sub_per_tile, epi_m=128, N_j_loc). The
    # atom builder + producer index the sub_m axis (same 4-D box-permute as the db slot); the drain reads it.
    with torch.cuda.use_mem_pool(DistributedManager.symmetric_mempool(device)):
        stage_buf = torch.empty(
            (n_clusters, m_sub, 128, size_Nj), dtype=torch.bfloat16, device=device
        )
    stage_buf.zero_()
    pe_dev_t = pm.cp_pe_table.to(torch.int32).contiguous()
    pe_dev_c = from_dlpack(pe_dev_t, assumed_align=4)
    stage_c = from_dlpack(stage_buf, assumed_align=16).mark_layout_dynamic()

    A0, Bt0, recv0, (cA0, cB0, cD0) = _build_inputs_2d(
        device, dist_manager.rank, N, cp0, cp1, Dloc, B
    )
    gemm = GemmA2ASm90(
        Float32, a_dtype, tile, (1, cluster_n, 1), pingpong=False, is_persistent=True
    )
    gemm.configure_a2a_gemm_native(
        cp=cp,
        my_cp_rank=int(pm.my_cp_rank),
        B=B,
        N_loc=N // cp0,
        pe_table=pe_table,
        N=N,
        cp_axis_sizes=(cp0, cp1),
        dynamic=True,
        arbitrary_n=True,
        pe_aligned_tiling=True,
        ib_ring=True,
        ib_quiet=True,
        decoupled=True,
        producer_tma=True,
        consumer_strided=True,
        consumer_strided_putwarp=True,
        ring_depth=rd,
        cluster_drain=True,
        cluster_n=cluster_n,
    )
    compiled = compile_gemm_with_bitcode(
        gemm,
        cA0,
        cB0,
        cD0,
        None,
        _epi_cluster(gemm, recv0, stage_c, pe_dev_c),
        sched,
        stream,
        None,
        register=True,
    )
    try:
        N_i_loc, N_j_loc = N // cp0, N // cp1
        A, Bt, recv, (cA, cB, cD) = _build_inputs_2d(
            device, dist_manager.rank, N, cp0, cp1, Dloc, B
        )
        _nvshmem_barrier()
        compiled(cA, cB, cD, None, _epi_cluster(gemm, recv, stage_c, pe_dev_c), sched, stream, None)
        _drain()
        exp = _ref_back_gemm_native_2d(
            A,
            Bt,
            pm,
            cp0=cp0,
            cp1=cp1,
            Dloc=Dloc,
            B=B,
            N_i_loc=N_i_loc,
            N_j_loc=N_j_loc,
            N=N,
            world_size=world_size,
        )
        tag = (
            f"ib_ring CLUSTER-DRAIN #78 tile{tile[0]}x{tile[1]} cp=({cp0},{cp1}) cn={cluster_n} D={D} "
            f"@N={N} (on-grid N_i={N_i_loc} N_j={N_j_loc} m_sub={m_sub} run_j={gemm._a2a_cluster_run_j})"
        )
        _gate_2d(recv, exp, cp, Dloc, B, N_i_loc, N_j_loc, tag, dist_manager.rank)
    finally:
        compiled.free()
        _nvshmem_barrier()


# --------------------------------------------------------------------------- #
# CLUSTER-DRAIN #76 (d) MULTISLOT — arbitrary cluster_n at a STRADDLE N (nt_j_pp % cluster_n != 0). The
# even-shard gate is BYPASSED: the cluster STRADDLES peer_j boundaries via per-tile peer-routing into 2
# rotating full-peer slots (slot = peer_j&1) over a FULL-BAND walk (run_j_dynamic). The producer real-
# arrives full[prev&1] on each peer-cross (+ the tail flush of the walk's last peer); rank-0 drains the 2
# slots alternating. CANARY: cp=(2,8), cluster_n=4, N s.t. nt_j_pp=6 (6%4=2 straddle, 6>4 CAP-ok) -> the
# 2-slot rotation across a straddle is the deadlock-prone bit. Gated vs the fp32 2-D reshard oracle
# (rel_L2<2e-2, 0 outlier/untouched). A hang here = a phase/reuse signal bug (run under timeout).
# --------------------------------------------------------------------------- #
# SHIP-LADDER (#76 (d)): cluster_multislot ships at cluster_n in {1,2} (the fix(a) per-peer serialize + drain
# fence_acq_rel_sys, DEFAULT-ON). cn=1 = degenerate 1-CTA multislot (intra-CTA, no cross-CTA race); cn=2 at a
# single-straddle (N=4672, nt_j=5) AND a MULTI/double-straddle (N=6528, nt_j=7, the walk crosses >1 peer_j
# boundary). cluster_n>=4 STRADDLE is DEFERRED -- the cross-CTA 2-slot coordination race with >2 producer CTAs
# that the per-peer serialize does NOT fully close (isolation cell cn4ms reproduces it under CPO_MS_SERIAL);
# unlocking it needs the whole-round full-buffer restructure (shared #45), pending the perf-curve go/no-go. It
# is NOT a dropped shape: every straddle shape ships via cluster_n<=2, and cluster_n>=4 ships EVEN-SHARD
# (cn4mse) cross-node today; cluster_n is an autotune knob, so cn>=4-straddle is one deferred autotune point,
# not an unsupported input size.
@pytest.mark.usefixtures("_nvshmem__distributed__ib_ring")
@pytest.mark.usefixtures("_module_skip__distributed__ib_ring")
@pytest.mark.parametrize(
    "cluster_n,N",
    [(1, 4672), (2, 4672), (2, 6528)],
    ids=["cn1_n4672", "cn2_n4672_straddle", "cn2_n6528_dblstraddle"],
)
@pytest.mark.parametrize("rd", [2], ids=["rd2"])
@EPI.parametrize("mesh")
def test_ib_ring_cluster_drain_2d_multislot(
    apply_mesh, mesh, dist_manager, device, world_size, rd, cluster_n, N
):
    """CLUSTER-DRAIN #76 (d) MULTISLOT correctness: configure + compile + run the cluster-cooperative
    FULL-BAND multislot store over a 2-D (cp0,cp1) shard at a STRADDLE anchor (nt_j_pp % cluster_n != 0,
    nt_j_pp > cluster_n) that the even-shard gate would REJECT, and gate the recv vs _ref_back_gemm_native_2d
    (per-row L2<2e-2, 0 outlier/untouched). Exercises the 2 rotating full-peer slots + real-arrive-on-cross
    + tail flush + per-slot empty reuse. Off-grid N_i_loc%128!=0 is the anchor itself. Needs cp=(2,8)."""
    apply_mesh(mesh)

    assert get_device_capacity()[0] == 9, "SM90 only"
    axis_sizes = _cp_axis_sizes(dist_manager)
    if len(axis_sizes) != 2:
        rank_invariant_skip(
            f"cluster_drain 2-D needs a 2-D cp mesh (CPO_DIST_MESH=cp=2*8); got {axis_sizes}",
            because=_GEOMETRY_UNIFORM_BECAUSE,
        )
    cp0, cp1 = axis_sizes
    cp = cp0 * cp1
    Dloc, B = 2, 1
    D = Dloc * cp

    Ni_chk, Nj_chk = N // cp0, N // cp1
    ntj_chk = (Nj_chk + TILE[1] - 1) // TILE[1]
    # Divisibility gate for THIS mesh (16-B TMA align + off-grid M anchor). cn>=2 ALSO needs the multislot
    # STRADDLE (ntj%cn!=0, ntj>cn) -- else the even-shard gate auto-reduces and multislot's slot rotation is
    # never exercised. cn=1 is the degenerate 1-CTA multislot: no peer-cross straddle exists (ntj%1==0 always)
    # so only the divisibility gate applies (confirms multislot degrades to cn=1 without a cross-CTA path).
    if not (
        N % cp0 == 0
        and N % cp1 == 0
        and Ni_chk % 8 == 0
        and Nj_chk % 8 == 0
        and Ni_chk % TILE[0] != 0
    ):
        rank_invariant_skip(
            f"anchor {N} not shape-valid for cp=({cp0},{cp1})", because=_GEOMETRY_UNIFORM_BECAUSE
        )
    if cluster_n >= 2 and not (ntj_chk % cluster_n != 0 and ntj_chk > cluster_n):
        rank_invariant_skip(
            f"anchor {N} not a multislot straddle for cp=({cp0},{cp1}) cn={cluster_n}",
            because=_GEOMETRY_UNIFORM_BECAUSE,
        )
    anchor = N
    pm = _trimul_placements(dist_manager)[1]
    pe_table = tuple(int(x) for x in pm.cp_pe_table.tolist())
    stream = cutlass_torch.current_stream()
    a_dtype = torch2cute_dtype_map[torch.bfloat16]
    n_clusters = get_max_active_clusters(cluster_n)
    print(
        f"[ckpt r{dist_manager.rank}] n_clusters={n_clusters} anchor={anchor} cn={cluster_n}",
        flush=True,
    )
    sched = make_scheduler_args(n_clusters, Int32(8), None, None)
    size_Nj = anchor // cp1
    # MULTISLOT reuses the rd-slot (db-shaped) staging (n_clusters, rd=2, epi_m=128, N_j_loc): the rd=2
    # slots ARE the 2 rotating full-peer slots. Symmetric-heap (IB put SOURCE), over-alloc reused.
    if dist_manager.rank == 0:
        print(f"[ckpt] before stage_buf alloc shape=({n_clusters},{rd},128,{size_Nj})", flush=True)
    with torch.cuda.use_mem_pool(DistributedManager.symmetric_mempool(device)):
        stage_buf = torch.empty((n_clusters, rd, 128, size_Nj), dtype=torch.bfloat16, device=device)
    stage_buf.zero_()
    _nvshmem_barrier()
    if dist_manager.rank == 0:
        print("[ckpt] after stage_buf alloc+zero+barrier", flush=True)
    pe_dev_t = pm.cp_pe_table.to(torch.int32).contiguous()
    pe_dev_c = from_dlpack(pe_dev_t, assumed_align=4)
    stage_c = from_dlpack(stage_buf, assumed_align=16).mark_layout_dynamic()

    A0, Bt0, recv0, (cA0, cB0, cD0) = _build_inputs_2d(
        device, dist_manager.rank, anchor, cp0, cp1, Dloc, B
    )
    _nvshmem_barrier()
    if dist_manager.rank == 0:
        print("[ckpt] after _build_inputs_2d recv0 alloc+barrier", flush=True)
    gemm = GemmA2ASm90(
        Float32, a_dtype, TILE, (1, cluster_n, 1), pingpong=False, is_persistent=True
    )
    gemm.configure_a2a_gemm_native(
        cp=cp,
        my_cp_rank=int(pm.my_cp_rank),
        B=B,
        N_loc=anchor // cp0,
        pe_table=pe_table,
        N=anchor,
        cp_axis_sizes=(cp0, cp1),
        dynamic=True,
        arbitrary_n=True,
        pe_aligned_tiling=True,
        ib_ring=True,
        ib_quiet=True,
        decoupled=True,
        producer_tma=True,
        consumer_strided=True,
        consumer_strided_putwarp=True,
        ring_depth=rd,
        cluster_drain=True,
        cluster_n=cluster_n,
        cluster_multislot=True,
    )
    if dist_manager.rank == 0:
        print("[ckpt] after configure; before compile", flush=True)
    compiled = compile_gemm_with_bitcode(
        gemm,
        cA0,
        cB0,
        cD0,
        None,
        _epi_cluster(gemm, recv0, stage_c, pe_dev_c),
        sched,
        stream,
        None,
        register=True,
    )
    if dist_manager.rank == 0:
        print("[ckpt] after compile", flush=True)
    try:
        N = anchor
        A, Bt, recv, (cA, cB, cD) = _build_inputs_2d(
            device, dist_manager.rank, N, cp0, cp1, Dloc, B
        )
        N_i_loc, N_j_loc = N // cp0, N // cp1
        ntj = (N_j_loc + TILE[1] - 1) // TILE[1]
        if dist_manager.rank == 0:
            print("[ckpt] after run recv alloc; before pre-kernel barrier", flush=True)
        _nvshmem_barrier()
        if dist_manager.rank == 0:
            print("[ckpt] after pre-kernel barrier; launching kernel", flush=True)
        compiled(cA, cB, cD, None, _epi_cluster(gemm, recv, stage_c, pe_dev_c), sched, stream, None)
        torch.cuda.synchronize()  # DEBUG: surface a kernel-run fault HERE (not deferred to _drain's barrier)
        if dist_manager.rank == 0:
            print(
                "[ckpt] after kernel launch + cuda.synchronize (kernel did NOT fault)", flush=True
            )
        _drain()
        if dist_manager.rank == 0:
            print("[ckpt] after _drain", flush=True)
        exp = _ref_back_gemm_native_2d(
            A,
            Bt,
            pm,
            cp0=cp0,
            cp1=cp1,
            Dloc=Dloc,
            B=B,
            N_i_loc=N_i_loc,
            N_j_loc=N_j_loc,
            N=N,
            world_size=world_size,
        )
        tag = (
            f"ib_ring CLUSTER MULTISLOT cp=({cp0},{cp1}) cn={cluster_n} D={D} rd={rd} @N={N} "
            f"(STRADDLE N_i={N_i_loc} N_j={N_j_loc} nt_j_pp={ntj} nt_j%cn={ntj % cluster_n})"
        )
        _gate_2d(recv, exp, cp, Dloc, B, N_i_loc, N_j_loc, tag, dist_manager.rank)
    finally:
        compiled.free()
        _nvshmem_barrier()


# --------------------------------------------------------------------------- #
# (d) MULTISLOT L=1 (Dloc*B==1) STRADDLE — the deadlock repro + regression gate.
# The shipped multislot test (test_ib_ring_cluster_drain_2d_multislot) fixes Dloc=2,B=1 -> L=Dloc*B=2, so
# L=1 (D<=cp, tiny D -> Dloc=1) was NEVER covered. A cross-node cp=(2,8) sweep FROZE at exactly this config
# (D=16 -> Dloc=1 -> L=1) at a straddle N (nt_j_pp odd). The FIRST PRINCIPLE forbids a D>=cp / L>=2 shape
# gate (D is an input dim; the only shape constraint is 16-B == N%8==0), so L=1 MUST RUN CORRECT, not raise.
# This test is (a) the deadlock REPRO (run under `timeout` + `compute-sanitizer --tool synccheck,racecheck`
# to name the divergent mbarrier) and (b) the fix's REGRESSION GATE (must PASS -- rel_L2<2e-2, 0 outlier --
# once the L=1 run deadlock is fixed). Dloc=1,B=1 is the ONLY delta vs the shipped multislot test; every other
# knob (2-slot rotation, real-arrive-on-cross, tail flush, per-slot reuse) is byte-identical. NEEDS cp=(2,8).
# One cell per torchrun job (a hang/fault poisons the process). synccheck cmd (run separately, on a dev cluster
# 2 nodes x 8 H100, 16 ranks):
#   CPO_CACHE_ENABLED=0 NVSHMEM_DISABLE_NVLS=1 CPO_DIST_MESH=cp=2*8 <srun/torchrun 16 ranks> \
#     compute-sanitizer --tool synccheck --target-processes all \
#     python -m pytest tests/distributed/test_ib_ring.py -x \
#       -k "test_ib_ring_cluster_drain_2d_multislot_L1 and cn2_n3008"
# racecheck-CLEAN + synccheck-ERRORS == a barrier/mbarrier divergence synccheck names directly.
# --------------------------------------------------------------------------- #
@pytest.mark.usefixtures("_nvshmem__distributed__ib_ring")
@pytest.mark.usefixtures("_module_skip__distributed__ib_ring")
@pytest.mark.parametrize(
    "cluster_n,N",
    [(2, 3008), (1, 3008), (2, 4672)],
    ids=["cn2_n3008_straddle", "cn1_n3008", "cn2_n4672_straddle"],
)
@pytest.mark.parametrize("rd", [2], ids=["rd2"])
@EPI.parametrize("mesh")
def test_ib_ring_cluster_drain_2d_multislot_L1(
    apply_mesh, mesh, dist_manager, device, world_size, rd, cluster_n, N
):
    """MULTISLOT at L=Dloc*B==1 (Dloc=1,B=1 -> D=cp): the tiny-D straddle that the shipped L=2 multislot test
    never exercised and that a cross-node sweep hung on. Same config as test_ib_ring_cluster_drain_2d_multislot
    with Dloc=1 (L=1). MUST run correct (no shape gate -- L=1 is a valid 16-B-aligned input). Gate vs
    _ref_back_gemm_native_2d (per-row L2<2e-2, 0 outlier). Run under timeout + compute-sanitizer for the
    deadlock localization (see the block comment above). Needs cp=(2,8)."""
    apply_mesh(mesh)

    assert get_device_capacity()[0] == 9, "SM90 only"
    axis_sizes = _cp_axis_sizes(dist_manager)
    if len(axis_sizes) != 2:
        rank_invariant_skip(
            f"cluster_drain 2-D needs a 2-D cp mesh (CPO_DIST_MESH=cp=2*8); got {axis_sizes}",
            because=_GEOMETRY_UNIFORM_BECAUSE,
        )
    cp0, cp1 = axis_sizes
    cp = cp0 * cp1
    Dloc, B = 1, 1  # L = Dloc*B == 1 (the tiny-D / D==cp case) -- the deadlock trigger
    D = Dloc * cp
    assert Dloc * B == 1, "this cell fixes the L=1 (Dloc*B==1) boundary the shipped L=2 test skips"

    Ni_chk, Nj_chk = N // cp0, N // cp1
    ntj_chk = (Nj_chk + TILE[1] - 1) // TILE[1]
    if not (
        N % cp0 == 0
        and N % cp1 == 0
        and Ni_chk % 8 == 0
        and Nj_chk % 8 == 0
        and Ni_chk % TILE[0] != 0
    ):
        rank_invariant_skip(
            f"anchor {N} not shape-valid (16-B straddle) for cp=({cp0},{cp1})",
            because=_GEOMETRY_UNIFORM_BECAUSE,
        )
    if cluster_n >= 2 and not (ntj_chk % cluster_n != 0 and ntj_chk > cluster_n):
        rank_invariant_skip(
            f"anchor {N} not a multislot straddle for cp=({cp0},{cp1}) cn={cluster_n}",
            because=_GEOMETRY_UNIFORM_BECAUSE,
        )
    anchor = N
    pm = _trimul_placements(dist_manager)[1]
    pe_table = tuple(int(x) for x in pm.cp_pe_table.tolist())
    stream = cutlass_torch.current_stream()
    a_dtype = torch2cute_dtype_map[torch.bfloat16]
    n_clusters = get_max_active_clusters(cluster_n)
    print(
        f"[ckpt r{dist_manager.rank}] L=1 multislot n_clusters={n_clusters} anchor={anchor} "
        f"cn={cluster_n} D={D}",
        flush=True,
    )
    sched = make_scheduler_args(n_clusters, Int32(8), None, None)
    size_Nj = anchor // cp1
    with torch.cuda.use_mem_pool(DistributedManager.symmetric_mempool(device)):
        stage_buf = torch.empty((n_clusters, rd, 128, size_Nj), dtype=torch.bfloat16, device=device)
    stage_buf.zero_()
    _nvshmem_barrier()
    pe_dev_t = pm.cp_pe_table.to(torch.int32).contiguous()
    pe_dev_c = from_dlpack(pe_dev_t, assumed_align=4)
    stage_c = from_dlpack(stage_buf, assumed_align=16).mark_layout_dynamic()

    A0, Bt0, recv0, (cA0, cB0, cD0) = _build_inputs_2d(
        device, dist_manager.rank, anchor, cp0, cp1, Dloc, B
    )
    _nvshmem_barrier()
    gemm = GemmA2ASm90(
        Float32, a_dtype, TILE, (1, cluster_n, 1), pingpong=False, is_persistent=True
    )
    gemm.configure_a2a_gemm_native(
        cp=cp,
        my_cp_rank=int(pm.my_cp_rank),
        B=B,
        N_loc=anchor // cp0,
        pe_table=pe_table,
        N=anchor,
        cp_axis_sizes=(cp0, cp1),
        dynamic=True,
        arbitrary_n=True,
        pe_aligned_tiling=True,
        ib_ring=True,
        ib_quiet=True,
        decoupled=True,
        producer_tma=True,
        consumer_strided=True,
        consumer_strided_putwarp=True,
        ring_depth=rd,
        cluster_drain=True,
        cluster_n=cluster_n,
        cluster_multislot=True,
    )
    compiled = compile_gemm_with_bitcode(
        gemm,
        cA0,
        cB0,
        cD0,
        None,
        _epi_cluster(gemm, recv0, stage_c, pe_dev_c),
        sched,
        stream,
        None,
        register=True,
    )
    try:
        N = anchor
        A, Bt, recv, (cA, cB, cD) = _build_inputs_2d(
            device, dist_manager.rank, N, cp0, cp1, Dloc, B
        )
        N_i_loc, N_j_loc = N // cp0, N // cp1
        ntj = (N_j_loc + TILE[1] - 1) // TILE[1]
        _nvshmem_barrier()
        compiled(cA, cB, cD, None, _epi_cluster(gemm, recv, stage_c, pe_dev_c), sched, stream, None)
        torch.cuda.synchronize()  # surface a kernel fault / hang HERE (deadlock localization)
        _drain()
        exp = _ref_back_gemm_native_2d(
            A,
            Bt,
            pm,
            cp0=cp0,
            cp1=cp1,
            Dloc=Dloc,
            B=B,
            N_i_loc=N_i_loc,
            N_j_loc=N_j_loc,
            N=N,
            world_size=world_size,
        )
        tag = (
            f"ib_ring CLUSTER MULTISLOT L=1 cp=({cp0},{cp1}) cn={cluster_n} D={D} rd={rd} @N={N} "
            f"(STRADDLE N_i={N_i_loc} N_j={N_j_loc} nt_j_pp={ntj} nt_j%cn={ntj % cluster_n} L={Dloc * B})"
        )
        _gate_2d(recv, exp, cp, Dloc, B, N_i_loc, N_j_loc, tag, dist_manager.rank)
    finally:
        compiled.free()
        _nvshmem_barrier()


# --------------------------------------------------------------------------- #
# FIX-1 REGRESSION LOCK — cluster_n>=4 cluster_multislot SAME-SLOT REUSE WAR fault.
# ROOT CAUSE (FIX1_ANALYSIS.md): the rd=2 slots are RECYCLED across a band's same-parity peers (even
# peers -> slot0, odd -> slot1). A producer refilling slot s (for peer P) must not race rank-0's NIC
# RDMA-read of slot s (for the earlier same-slot peer P-2). The per-slot WAR gate (REUSE-GATE,
# gemm_sm90_a2a.py :2714, waits empty[slot]) IS present, but its async-proxy ORDERING FENCE (:2751
# fence_proxy async.global) was `const_expr(_serial)`-GATED -> in the double-buffer perf mode
# (CPO_MS_SERIAL=0) the reuse-gate's empty[slot] acquire was NOT enforced against the async TMA-S2G
# write -> the write races the NIC read of the SAME slot -> CUDA_ERROR_LAUNCH_FAILED (a RACE, reliably
# exposed at cluster_n>=4 = 3+ concurrent producer CTAs). FIX: the fence is now UNCONDITIONAL. Different-
# slot overlap stays legal (concurrent RDMA-read(slotA) || TMA-write(slotB) on non-overlapping subspans
# of one registered MR is safe) -> the double-buffer perf is preserved. cluster_n=2 stays green.
# serial="0" = the double-buffer mode (RED pre-fix, GREEN post-fix); serial="1" = the fully-serialized
# mode (GREEN both, byte-identical). One cell per torchrun job. Needs cp=(2,8) cross-node. `_serial` is a
# COMPILE-time const_expr read from CPO_MS_SERIAL, so the env is set BEFORE compile (one-cell isolation).
# --------------------------------------------------------------------------- #
@pytest.mark.usefixtures("_nvshmem__distributed__ib_ring")
@pytest.mark.usefixtures("_module_skip__distributed__ib_ring")
@EPI.parametrize(
    "N",
    "mesh",
    cells=[
        (n, m) for n in (4096, 8192) for m in EPI.axis("mesh").values if not _misaligned_2d(n, m)
    ],
    because=(
        "the two token counts whose per-cluster band count divides cluster_n=4 on this mesh. "
        "The serial/double-buffer arm and the ring depth ride on top as ordinary parameters: "
        "they are drain-internal knobs, not shape. " + _JOINT_MESH_BECAUSE
    ),
)
@pytest.mark.parametrize("serial", ["0", "1"], ids=["dbuf_ms0", "serial_ms1"])
@pytest.mark.parametrize("cluster_n", [4], ids=["cn4"])
@pytest.mark.parametrize("rd", [2], ids=["rd2"])
def test_ib_ring_cluster_drain_2d_multislot_cn4(
    apply_mesh, mesh, dist_manager, device, world_size, rd, cluster_n, serial, N
):
    """FIX-1: cluster_multislot=True at cluster_n=4, EVEN-SHARD, LARGE-N multi-band, cp=(2,8). serial="0"
    (CPO_MS_SERIAL=0, the double-buffer perf mode) reproduces the same-slot-reuse WAR ULF on the pre-fix
    kernel and MUST go GREEN post-fix; serial="1" (fully-serialized) is GREEN both (byte-identical). Gate
    vs the fp32 2-D reshard oracle (rel_L2<2e-2, 0 outlier/untouched)."""
    apply_mesh(mesh)

    assert get_device_capacity()[0] == 9, "SM90 only"
    axis_sizes = _cp_axis_sizes(dist_manager)
    if len(axis_sizes) != 2:
        rank_invariant_skip(
            f"cluster_drain 2-D needs a 2-D cp mesh (CPO_DIST_MESH=cp=2*8); got {axis_sizes}",
            because=_GEOMETRY_UNIFORM_BECAUSE,
        )
    cp0, cp1 = axis_sizes
    if (cp0, cp1) != (2, 8):
        rank_invariant_skip(
            f"FIX-1 regression pinned to cp=(2,8); got ({cp0},{cp1})",
            because=_GEOMETRY_UNIFORM_BECAUSE,
        )
    cp = cp0 * cp1
    Dloc, B = 2, 1
    D = Dloc * cp
    N = N
    N_i_loc, N_j_loc = N // cp0, N // cp1
    nt_j_pp = (N_j_loc + TILE[1] - 1) // TILE[1]
    nt_i_pp = (N_i_loc + TILE[0] - 1) // TILE[0]
    # EVEN-SHARD (nt_j_pp % cn == 0) + 16-B (%8). MULTI-BAND (total_bands = cp0*nt_i_pp*L > n_clusters)
    # is the regime that recycles the 2 slots across many same-parity peers -> the WAR hazard.
    if not (
        N % cp0 == 0
        and N % cp1 == 0
        and N_i_loc % 8 == 0
        and N_j_loc % 8 == 0
        and nt_j_pp % cluster_n == 0
    ):
        rank_invariant_skip(
            f"anchor {N} not even-shard/16-B for cp=({cp0},{cp1}) cn={cluster_n}",
            because=_GEOMETRY_UNIFORM_BECAUSE,
        )
    a_dtype = torch2cute_dtype_map[torch.bfloat16]
    pm = _trimul_placements(dist_manager)[1]
    pe_table = tuple(int(x) for x in pm.cp_pe_table.tolist())
    n_clusters = get_max_active_clusters(cluster_n)
    total_bands = cp0 * nt_i_pp * (Dloc * B)
    if dist_manager.rank == 0:
        print(
            f"\n[FIX-1 cn{cluster_n} MS_SERIAL={serial} N={N}] nt_j_pp={nt_j_pp} total_bands={total_bands} "
            f"n_clusters={n_clusters} {'MULTI-BAND' if total_bands > n_clusters else 'single-band'}",
            flush=True,
        )
    stream = cutlass_torch.current_stream()
    sched = make_scheduler_args(n_clusters, Int32(8), None, None)
    size_Nj = N // cp1
    with torch.cuda.use_mem_pool(DistributedManager.symmetric_mempool(device)):
        stage_buf = torch.empty((n_clusters, rd, 128, size_Nj), dtype=torch.bfloat16, device=device)
    stage_buf.zero_()
    _nvshmem_barrier()
    pe_dev_t = pm.cp_pe_table.to(torch.int32).contiguous()
    pe_dev_c = from_dlpack(pe_dev_t, assumed_align=4)
    stage_c = from_dlpack(stage_buf, assumed_align=16).mark_layout_dynamic()

    A0, Bt0, recv0, (cA0, cB0, cD0) = _build_inputs_2d(
        device, dist_manager.rank, N, cp0, cp1, Dloc, B
    )
    _nvshmem_barrier()
    # _serial (the fence + serialize gate) is a COMPILE-time const_expr read from CPO_MS_SERIAL -> set it
    # BEFORE compile; restore after (one-cell-per-job isolation makes the global env safe).
    _saved = os.environ.get("CPO_MS_SERIAL")
    os.environ["CPO_MS_SERIAL"] = serial
    try:
        gemm = GemmA2ASm90(
            Float32, a_dtype, TILE, (1, cluster_n, 1), pingpong=False, is_persistent=True
        )
        gemm.configure_a2a_gemm_native(
            cp=cp,
            my_cp_rank=int(pm.my_cp_rank),
            B=B,
            N_loc=N // cp0,
            pe_table=pe_table,
            N=N,
            cp_axis_sizes=(cp0, cp1),
            dynamic=True,
            arbitrary_n=True,
            pe_aligned_tiling=True,
            ib_ring=True,
            ib_quiet=True,
            decoupled=True,
            producer_tma=True,
            consumer_strided=True,
            consumer_strided_putwarp=True,
            ring_depth=rd,
            cluster_drain=True,
            cluster_n=cluster_n,
            cluster_multislot=True,
        )
        compiled = compile_gemm_with_bitcode(
            gemm,
            cA0,
            cB0,
            cD0,
            None,
            _epi_cluster(gemm, recv0, stage_c, pe_dev_c),
            sched,
            stream,
            None,
            register=True,
        )
    finally:
        if _saved is None:
            os.environ.pop("CPO_MS_SERIAL", None)
        else:
            os.environ["CPO_MS_SERIAL"] = _saved
    try:
        A, Bt, recv, (cA, cB, cD) = _build_inputs_2d(
            device, dist_manager.rank, N, cp0, cp1, Dloc, B
        )
        _nvshmem_barrier()
        compiled(cA, cB, cD, None, _epi_cluster(gemm, recv, stage_c, pe_dev_c), sched, stream, None)
        torch.cuda.synchronize()  # surface the same-slot-WAR ULF HERE (pre-fix, serial="0")
        _drain()
        exp = _ref_back_gemm_native_2d(
            A,
            Bt,
            pm,
            cp0=cp0,
            cp1=cp1,
            Dloc=Dloc,
            B=B,
            N_i_loc=N_i_loc,
            N_j_loc=N_j_loc,
            N=N,
            world_size=world_size,
        )
        tag = (
            f"ib_ring FIX-1 MULTISLOT cn{cluster_n} MS_SERIAL={serial} cp=({cp0},{cp1}) D={D} rd={rd} "
            f"@N={N} (N_i={N_i_loc} N_j={N_j_loc} nt_j_pp={nt_j_pp} total_bands={total_bands})"
        )
        _gate_2d(recv, exp, cp, Dloc, B, N_i_loc, N_j_loc, tag, dist_manager.rank)
    finally:
        compiled.free()
        _nvshmem_barrier()


def _throwaway_gemm():
    a_dtype = torch2cute_dtype_map[torch.bfloat16]
    return GemmA2ASm90(Float32, a_dtype, TILE, (1, 1, 1), pingpong=False, is_persistent=True)


@pytest.mark.usefixtures("_nvshmem__distributed__ib_ring")
@pytest.mark.usefixtures("_module_skip__distributed__ib_ring")
@matrix_exempt(
    "asserts that a REMOVED configuration is refused. There is no shape: the claim is that a particular kwarg combination raises, whatever it is given"
)
def test_ib_ring_uniform_store_rejected():
    """G4: a COMPLETE ib_ring stack with no cluster_drain is rejected — the bare uniform ib_ring putwarp
    store (the §0.9.5 loser) was removed. Epic-close P2 reduces ib_ring to the cluster_drain wide band
    only. Locks in the cut."""
    g = _throwaway_gemm()
    with pytest.raises(ValueError, match="requires a WIDE-BAND drain"):
        g.configure_a2a_gemm_native(
            cp=2,
            my_cp_rank=0,
            B=1,
            N_loc=320,
            pe_table=(0, 1),
            N=640,
            arbitrary_n=True,
            pe_aligned_tiling=True,
            ib_ring=True,
            ib_quiet=True,
            decoupled=True,
            producer_tma=True,
            consumer_strided=True,
            consumer_strided_putwarp=True,
            ring_depth=2,
        )


@pytest.mark.usefixtures("_nvshmem__distributed__ib_ring")
@pytest.mark.usefixtures("_module_skip__distributed__ib_ring")
@matrix_exempt(
    "same as the uniform-store rejection -- a claim about removed kwargs, with no operand"
)
def test_removed_hybrid_drain_kwargs_rejected():
    """Epic-close P2 (cluster_multislot is the SOLE IB drain): every cut IB-drain variant's kwarg was
    REMOVED from configure_a2a_gemm_native's signature, so passing one is an UNKNOWN keyword argument ->
    TypeError (the 'never existed' behavior). Covers the earlier hybrid-drain kwargs (completion_drain /
    full_buffer) AND the epic-close-P2 cut set (coalesce / differential / cluster_roundpark /
    cluster_nvdirect / cluster_drain_db / cluster_drain_mw / ib_reap + their drain_warps / drain_cta_div /
    reap_cadence / min_drain / handshake_skip / local_world_size sub-levers)."""
    g = _throwaway_gemm()
    base = dict(cp=2, my_cp_rank=0, B=1, N_loc=256, pe_table=(0, 1), N=512)
    removed_kwargs = (
        dict(completion_drain=True),
        dict(full_buffer=True),
        dict(coalesce=True),
        dict(differential=True),
        dict(cluster_roundpark=True),
        dict(cluster_nvdirect=True),
        dict(cluster_drain_db=True),
        dict(cluster_drain_mw=True),
        dict(ib_reap=True),
        dict(drain_warps=1),
        dict(drain_cta_div=2),
        dict(min_drain=True),
        dict(handshake_skip=True),
        dict(local_world_size=8),
        dict(reap_cadence=1),
    )
    for removed in removed_kwargs:
        with pytest.raises(TypeError, match="unexpected keyword argument"):
            g.configure_a2a_gemm_native(**base, decoupled=True, **removed)


@pytest.fixture(scope="session")
def _module_skip__distributed__ib_ring():
    pytest.importorskip("nvshmem.core")
    pytest.importorskip("fold_cp_ops.distributed.gemm_a2a_epi")


# every live test in this file drives a COUPLED pe_aligned store (pe_aligned_tiling on
# top of configure_a2a_gemm_native / configure_a2a_sharded, no decoupled ring), which TMA-S2Gs into the
# peer's symmetric heap via nvshmem_ptr -> NULL for an IB peer -> illegal memory access on a cross-node
# mesh. §5.2/§5.4: `pe_aligned` faults INDEPENDENTLY of `coupled` (its own base_m / aligned predicate /
# store instruction), measured isolated at cp=16, while the same store at the same shapes on cp=8
# all-P2P passes. Rank-invariant predicate -> a plain skip stays collective (E4). The two configure-time
# reject tests (dyn_cp_wontfix, 2d_misaligned_xfail) never launch a store and are deliberately NOT
# guarded — guarding an xfail'd raise would report SKIPPED and kill the alarm.
_SKIP_DETAIL = "pe_aligned back store; cross-node is covered by the V6 decoupled drain."

K__distributed__pe_aligned_general = 256
DLOC = 128
Bsz = 1
TILE__distributed__pe_aligned_general = (128, 128)
ANCHOR__distributed__pe_aligned_general = (
    640  # straddle anchor (N_loc = 640//cp is NOT a multiple of 128 for cp in {2,4,8})
)
RUN_NS__distributed__pe_aligned_general = [
    512,
    640,
    896,
    1024,
]  # aligned + straddle mix (all % cp == 0, all % tile_n == 0)


@pytest.fixture(scope="session", autouse=False)
def _nvshmem__distributed__pe_aligned_general(dist_manager):
    from fold_cp_ops.distributed.distributed_manager import DistributedManager

    DistributedManager.init_nvshmem()
    yield


def _pe_map__distributed__pe_aligned_general(dist_manager):
    mesh = dist_manager.device_mesh
    placements = [Shard(i) for i in range(mesh.ndim)]
    return PeMap.from_mesh_placements(mesh, placements, distributed_manager=dist_manager)


def _drain__distributed__pe_aligned_general():
    import nvshmem.core
    import nvshmem.core.rma as nvshmem_rma

    nvshmem_rma.quiet(stream=torch.cuda.current_stream())
    nvshmem.core.barrier_all(stream=torch.cuda.current_stream())
    torch.cuda.synchronize()


def _barrier():
    import nvshmem.core

    nvshmem.core.barrier_all(stream=torch.cuda.current_stream())


def _mark__distributed__pe_aligned_general(t, dynamic):
    c = from_dlpack(t, assumed_align=16)
    return c.mark_layout_dynamic() if dynamic else c


def _build(device, rank, N, cp, dynamic):

    M, L, N_loc = N, DLOC * Bsz, N // cp
    torch.manual_seed(4321 + rank)
    A = torch.randn(
        L, M, K__distributed__pe_aligned_general, device=device, dtype=torch.bfloat16
    ) / (K__distributed__pe_aligned_general**0.5)
    Bt = torch.randn(
        L, N, K__distributed__pe_aligned_general, device=device, dtype=torch.bfloat16
    ) / (K__distributed__pe_aligned_general**0.5)
    with torch.cuda.use_mem_pool(DistributedManager.symmetric_mempool(device)):
        recv = torch.empty((cp, DLOC, Bsz, N_loc, N), dtype=torch.bfloat16, device=device)
    recv.fill_(-99.0)
    A3, B3 = A.permute(1, 2, 0), Bt.permute(1, 2, 0)
    D_logical = torch.as_strided(recv, (M, N, L), (N, 1, M * N))
    return (
        A,
        Bt,
        recv,
        (
            _mark__distributed__pe_aligned_general(A3, dynamic),
            _mark__distributed__pe_aligned_general(B3, dynamic),
            _mark__distributed__pe_aligned_general(D_logical, dynamic),
        ),
    )


def _epi(recv, dynamic):
    return GemmA2ASm90.EpilogueArguments(
        alpha=None,
        beta=None,
        mRowVecBroadcast=None,
        mColVecBroadcast=None,
        add_to_output=False,
        rounding_mode=None,
        sr_seed=None,
        recv=_mark__distributed__pe_aligned_general(recv, dynamic),
    )


def _ref(A, Bt, pm, N, cp, world_size):
    import torch.distributed as dist

    N_loc = N // cp
    tri = torch.bmm(A.float(), Bt.float().transpose(-1, -2)).to(torch.bfloat16)  # (L,M,N)
    gathered = [torch.empty_like(tri) for _ in range(world_size)]
    dist.all_gather(gathered, tri.contiguous())
    exp = torch.empty((cp, DLOC, Bsz, N_loc, N), device=A.device, dtype=torch.bfloat16)
    for s in range(cp):
        src = gathered[int(pm.cp_pe_table[s].item())]
        for d in range(DLOC):
            for b in range(Bsz):
                Lp = d * Bsz + b
                exp[s, d, b, :, :] = src[Lp, pm.my_cp_rank * N_loc : (pm.my_cp_rank + 1) * N_loc, :]
    return exp


@pytest.mark.usefixtures("_nvshmem__distributed__pe_aligned_general")
@pytest.mark.usefixtures("_module_skip__distributed__pe_aligned_general")
@matrix_exempt(
    "the subject is that ONE compile at a straddle anchor serves MANY token counts, so the several N values must be live simultaneously inside a single test body. Parametrizing would compile once per N and assert the opposite of what the test is for"
)
def test_pe_aligned_dynamic_1d(apply_mesh, dist_manager, device, world_size):
    """A1: ONE dynamic-shape compile (straddle anchor) serves many token counts, 1-D token shard."""

    assert get_device_capacity()[0] == 9, "SM90 only"
    cp = world_size
    if ANCHOR__distributed__pe_aligned_general % cp != 0 or any(
        N % cp != 0 for N in RUN_NS__distributed__pe_aligned_general
    ):
        rank_invariant_skip(
            f"ANCHOR/RUN_NS not divisible by cp={cp}", because=_GEOMETRY_UNIFORM_BECAUSE
        )
    N_loc0 = ANCHOR__distributed__pe_aligned_general // cp
    if N_loc0 % TILE__distributed__pe_aligned_general[0] == 0:
        rank_invariant_skip(
            f"anchor N_loc={N_loc0} not a straddle for cp={cp} (pe_aligned would auto-reduce)",
            because=_GEOMETRY_UNIFORM_BECAUSE,
        )
    # BUILD the mesh. `_pe_map__distributed__pe_aligned_general` reads `dist_manager.device_mesh`,
    # which only `apply_mesh` constructs -- so without this the test dies at
    # `'NoneType' object has no attribute 'ndim'` whenever it runs FIRST, and passes only when some
    # earlier test in the same session happened to build one. Measured: it fails under `-k` isolation
    # and passes in a whole-file run, i.e. it was order-dependent rather than green.
    apply_mesh((("cp", world_size),))
    pm = _pe_map__distributed__pe_aligned_general(dist_manager)
    pe_table = tuple(int(x) for x in pm.cp_pe_table.tolist())
    topology.skip_if_coupled_cross_node(pe_table, detail=_SKIP_DETAIL)
    stream = cutlass_torch.current_stream()
    a_dtype = torch2cute_dtype_map[torch.bfloat16]
    sched = make_scheduler_args(get_max_active_clusters(1), Int32(8), None, None)

    A0, Bt0, recv0, (cA0, cB0, cD0) = _build(
        device, dist_manager.rank, ANCHOR__distributed__pe_aligned_general, cp, True
    )
    gemm = GemmA2ASm90(
        Float32,
        a_dtype,
        TILE__distributed__pe_aligned_general,
        (1, 1, 1),
        pingpong=False,
        is_persistent=True,
    )
    gemm.configure_a2a_gemm_native(
        cp=cp,
        my_cp_rank=int(pm.my_cp_rank),
        B=Bsz,
        N_loc=N_loc0,
        pe_table=pe_table,
        dynamic=True,
        arbitrary_n=True,
        pe_aligned_tiling=True,
        N=ANCHOR__distributed__pe_aligned_general,
    )
    compiled = compile_gemm_with_bitcode(
        gemm, cA0, cB0, cD0, None, _epi(recv0, True), sched, stream, None, register=True
    )

    try:
        for N in RUN_NS__distributed__pe_aligned_general:
            A, Bt, recv, (cA, cB, cD) = _build(device, dist_manager.rank, N, cp, True)
            _barrier()
            compiled(cA, cB, cD, None, _epi(recv, True), sched, stream, None)
            _drain__distributed__pe_aligned_general()
            exp = _ref(A, Bt, pm, N, cp, world_size)
            got4d = recv.reshape(cp, DLOC * Bsz, N // cp, N)
            exp4d = exp.reshape(cp, DLOC * Bsz, N // cp, N)
            h = compute_error_histogram(got4d, exp4d)
            untouched = int((recv == -99.0).all(dim=4).sum().item())
            kind = (
                "straddle"
                if (N // cp) % TILE__distributed__pe_aligned_general[0] != 0
                else "aligned"
            )
            if dist_manager.rank == 0:
                print(
                    f"\n[pe_aligned dyn cp={cp} compile@{ANCHOR__distributed__pe_aligned_general} run@{N} ({kind} N_loc={N // cp})] "
                    f"rel_L2={h.rel_l2:.3e} outlier_rows={h.n_outlier_rows} untouched={untouched} "
                    f"(gate below)",
                    flush=True,
                )
            _assert_recv_matches(got4d, exp4d, bar=2e-2, what=f"cp={cp} N={N}")
            assert h.n_outlier_rows == 0, f"cp={cp} N={N}: {h.n_outlier_rows} outlier rows"
            assert untouched == 0, f"cp={cp} N={N}: {untouched} unwritten recv cells"
    finally:
        compiled.free()
        _barrier()


@pytest.mark.usefixtures("_nvshmem__distributed__pe_aligned_general")
@pytest.mark.usefixtures("_module_skip__distributed__pe_aligned_general")
@EPI.parametrize(
    "N",
    "mesh",
    cells=[
        (n, m)
        for n in (640, 1152, 1056)
        for m in EPI.axis("mesh").values
        if not _misaligned_2d(n, m)
    ],
    because=(
        "the three token counts whose per-axis split is legal on a 2-D mesh AND leaves at "
        "least one axis straddling, which is what makes pe_aligned engage. 1056 is the "
        "partial-token-j case (N%128!=0), so the runtime column clamp is exercised too. "
        "JOINT, not two stacked decorators: a region is evaluated only when the SAME call "
        "parametrizes every axis it reads, so stacking left the 16-B per-peer region unchecked "
        "and pytest crossed N=1056 with cp=(2,8) -- 1056//8 = 132, which is 4 mod 8, so the "
        "per-peer box start is not 16-B aligned and configure MUST raise. A success-asserting "
        "test may not claim it; the refusal is owned by the region and swept by "
        "test_a_declared_unsupported_configuration_is_refused_at_the_front_door. "
        "DERIVED from _misaligned_2d rather than listed, so a mesh value added to the pool joins "
        "this test by construction -- a frozen literal in a sibling test had already silently "
        "dropped the two newest specs before anyone noticed"
    ),
)
def test_pe_aligned_static_2d(apply_mesh, mesh, dist_manager, device, world_size, N):
    """A2: 2-D token shard (cp0×cp1) pe_aligned, STATIC. Per-peer ceil tiling on BOTH axes -> a
    STRADDLING N_i_loc/N_j_loc (NOT a multiple of the CTA tile) routes cleanly (which the aligned
    2-D store rejects). Gate: fp32 batched-GEMM-then-2D-reshard oracle, error-histogram + coverage.
    Needs a 2-D cp mesh -- SQUARE (CPO_DIST_MESH=cp=2*2, 4 GPUs) or NON-SQUARE (cp=2*4, 8 GPUs;
    cp=2*3, 6 GPUs -- Task-13 Phase 0). N=1056 is %96==0, keeping N_i_loc=N/cp0 and N_j_loc=N/cp1
    16-B-safe (%8==0) AND off-grid (%128!=0 on N and on both per-axis extents) across every cp0=2,
    cp1 in {2,3,4} mesh this file is launched under -- N=1200 (a naive 600/400 pick) is NOT safe
    here: it satisfies cp=2*3 but violates the %8 guard at cp=2*4 (N_j_loc=300, 300%8=4)."""
    apply_mesh(mesh)

    assert get_device_capacity()[0] == 9, "SM90 only"
    axis_sizes = _cp_axis_sizes(dist_manager)
    if len(axis_sizes) != 2:
        rank_invariant_skip(
            f"2-D pe_aligned test needs a 2-D cp mesh; session mesh is {axis_sizes}",
            because=_GEOMETRY_UNIFORM_BECAUSE,
        )
    cp0, cp1 = axis_sizes
    cp = cp0 * cp1
    if N % cp0 != 0 or N % cp1 != 0:
        rank_invariant_skip(
            f"N={N} not divisible by both cp axes {axis_sizes}", because=_GEOMETRY_UNIFORM_BECAUSE
        )
    N_i_loc, N_j_loc = N // cp0, N // cp1
    if N_i_loc % 128 == 0 and N_j_loc % 128 == 0:
        rank_invariant_skip(
            f"N_i_loc={N_i_loc}/N_j_loc={N_j_loc} both aligned — pe_aligned adds nothing here",
            because=_GEOMETRY_UNIFORM_BECAUSE,
        )
    D = 2 * cp
    Dloc, B = D // cp, 1
    L = Dloc * B
    placements, pm = _trimul_placements(dist_manager)
    mesh = _session_cp_mesh(dist_manager)
    topology.skip_if_coupled_cross_node(
        tuple(int(x) for x in pm.cp_pe_table.tolist()), detail=_SKIP_DETAIL
    )
    torch.manual_seed(4321 + dist_manager.rank)
    A = torch.randn(
        L, N, K__distributed__pe_aligned_general, device=device, dtype=torch.bfloat16
    ) / (K__distributed__pe_aligned_general**0.5)  # (L, i=N, K)
    Bt = torch.randn(
        L, N, K__distributed__pe_aligned_general, device=device, dtype=torch.bfloat16
    ) / (K__distributed__pe_aligned_general**0.5)  # (L, j=N, K)
    with torch.cuda.use_mem_pool(DistributedManager.symmetric_mempool(device)):
        recv = torch.empty((cp, Dloc, B, N_i_loc, N_j_loc), dtype=torch.bfloat16, device=device)
    recv.fill_(-99.0)

    fn = lambda g: g.configure_a2a_sharded(
        mesh, placements, pe_map=pm, B=B, N=N, gemm_native=True, pe_aligned_tiling=True
    )
    compiled, run_args = _compile_back_gemm_native(
        A, Bt, recv, (128, 128), gemm_native_cfg=None, configure_fn=fn
    )
    try:
        _barrier()
        compiled(*run_args)
        _drain__distributed__pe_aligned_general()
        expected = _ref_back_gemm_native_2d(
            A,
            Bt,
            pm,
            cp0=cp0,
            cp1=cp1,
            Dloc=Dloc,
            B=B,
            N_i_loc=N_i_loc,
            N_j_loc=N_j_loc,
            N=N,
            world_size=world_size,
        )
        got_bijD = recv.permute(2, 3, 4, 0, 1).reshape(B, N_i_loc, N_j_loc, cp * Dloc)
        ref_bijD = expected.permute(2, 3, 4, 0, 1).reshape(B, N_i_loc, N_j_loc, cp * Dloc)
        h = compute_error_histogram(got_bijD, ref_bijD)
        untouched = int((recv == -99.0).all(dim=4).sum().item())
        if dist_manager.rank == 0:
            print(
                f"\n[pe_aligned 2D cp={cp0}x{cp1} N={N} N_i_loc={N_i_loc} N_j_loc={N_j_loc}] "
                f"rel_L2={h.rel_l2:.3e} outlier_rows={h.n_outlier_rows} untouched={untouched} "
                f"(gate below)",
                flush=True,
            )
        _assert_recv_matches(got_bijD, ref_bijD, bar=2e-2, what=f"2D cp={cp0}x{cp1} N={N}")
        assert h.n_outlier_rows == 0, f"2D cp={cp0}x{cp1} N={N}: {h.n_outlier_rows} outlier rows"
        assert untouched == 0, f"2D cp={cp0}x{cp1} N={N}: {untouched} unwritten recv cells"
    finally:
        compiled.free()
        _barrier()


def _build_2d(device, rank, N, cp0, cp1, Dloc, dynamic):

    cp = cp0 * cp1
    M, L = N, Dloc * Bsz
    N_i_loc, N_j_loc = N // cp0, N // cp1
    torch.manual_seed(4321 + rank)
    A = torch.randn(
        L, N, K__distributed__pe_aligned_general, device=device, dtype=torch.bfloat16
    ) / (K__distributed__pe_aligned_general**0.5)  # (L, i=N, K)
    Bt = torch.randn(
        L, N, K__distributed__pe_aligned_general, device=device, dtype=torch.bfloat16
    ) / (K__distributed__pe_aligned_general**0.5)  # (L, j=N, K)
    with torch.cuda.use_mem_pool(DistributedManager.symmetric_mempool(device)):
        recv = torch.empty((cp, Dloc, Bsz, N_i_loc, N_j_loc), dtype=torch.bfloat16, device=device)
    recv.fill_(-99.0)
    A3, B3 = A.permute(1, 2, 0), Bt.permute(1, 2, 0)  # (i=N,K,L), (j=N,K,L)
    D_logical = torch.as_strided(
        recv, (M, N, L), (N, 1, M * N)
    )  # logical (M,N,L); store uses peer atoms
    return (
        A,
        Bt,
        recv,
        (
            _mark__distributed__pe_aligned_general(A3, dynamic),
            _mark__distributed__pe_aligned_general(B3, dynamic),
            _mark__distributed__pe_aligned_general(D_logical, dynamic),
        ),
    )


@pytest.mark.usefixtures("_nvshmem__distributed__pe_aligned_general")
@pytest.mark.usefixtures("_module_skip__distributed__pe_aligned_general")
@EPI.parametrize("mesh")
def test_pe_aligned_dynamic_2d(apply_mesh, mesh, dist_manager, device, world_size):
    """A3: 2-D token shard (cp0×cp1) pe_aligned, DYNAMIC. ONE compile at a straddle anchor serves many
    token counts on BOTH axes. Gate: fp32 2D-reshard oracle, error-histogram + coverage. 2-D mesh only
    -- SQUARE (cp=2*2) or NON-SQUARE (cp=2*4, cp=2*3 -- Task-13 Phase 0)."""
    apply_mesh(mesh)

    assert get_device_capacity()[0] == 9, "SM90 only"
    axis_sizes = _cp_axis_sizes(dist_manager)
    if len(axis_sizes) != 2:
        rank_invariant_skip(
            f"2-D pe_aligned dynamic test needs a 2-D cp mesh; session mesh is {axis_sizes}",
            because=_GEOMETRY_UNIFORM_BECAUSE,
        )
    cp0, cp1 = axis_sizes
    cp = cp0 * cp1
    # Base set serves cp0=2, cp1 in {2,4} (640 % 3 != 0 -> cp1=3 falls through to the %96==0 set below,
    # which keeps N_i_loc=N/cp0 and N_j_loc=N/cp1 both %8==0 for cp0=2, cp1 in {2,3,4} -- Task-13 Phase 0
    # non-square coverage). Square (2,2) and (2,4) are UNCHANGED (still hit the base set -> byte-identical
    # to the pre-Phase-0 anchor/runs).
    anchor, runs = 640, [640, 896, 1152, 1024]
    if anchor % cp0 != 0 or anchor % cp1 != 0 or any(N % cp0 != 0 or N % cp1 != 0 for N in runs):
        anchor, runs = 1152, [960, 1152, 1344, 1056]
    if anchor % cp0 != 0 or anchor % cp1 != 0 or any(N % cp0 != 0 or N % cp1 != 0 for N in runs):
        rank_invariant_skip(
            f"anchor/runs not divisible by cp axes {axis_sizes}", because=_GEOMETRY_UNIFORM_BECAUSE
        )
    if (anchor // cp0) % 128 == 0 and (anchor // cp1) % 128 == 0:
        rank_invariant_skip(
            f"anchor N_i/N_j not a straddle for {axis_sizes}", because=_GEOMETRY_UNIFORM_BECAUSE
        )
    D = 2 * cp
    Dloc = D // cp
    placements, pm = _trimul_placements(dist_manager)
    mesh = _session_cp_mesh(dist_manager)
    topology.skip_if_coupled_cross_node(
        tuple(int(x) for x in pm.cp_pe_table.tolist()), detail=_SKIP_DETAIL
    )
    stream = cutlass_torch.current_stream()
    a_dtype = torch2cute_dtype_map[torch.bfloat16]
    sched = make_scheduler_args(get_max_active_clusters(1), Int32(8), None, None)

    A0, Bt0, recv0, (cA0, cB0, cD0) = _build_2d(
        device, dist_manager.rank, anchor, cp0, cp1, Dloc, True
    )
    gemm = GemmA2ASm90(
        Float32,
        a_dtype,
        TILE__distributed__pe_aligned_general,
        (1, 1, 1),
        pingpong=False,
        is_persistent=True,
    )
    gemm.configure_a2a_sharded(
        mesh,
        placements,
        pe_map=pm,
        B=Bsz,
        N=anchor,
        gemm_native=True,
        dynamic=True,
        pe_aligned_tiling=True,
    )
    epi0 = GemmA2ASm90.EpilogueArguments(
        alpha=None,
        beta=None,
        mRowVecBroadcast=None,
        mColVecBroadcast=None,
        add_to_output=False,
        rounding_mode=None,
        sr_seed=None,
        recv=_mark__distributed__pe_aligned_general(recv0, True),
    )
    compiled = compile_gemm_with_bitcode(
        gemm, cA0, cB0, cD0, None, epi0, sched, stream, None, register=True
    )

    try:
        for N in runs:
            A, Bt, recv, (cA, cB, cD) = _build_2d(
                device, dist_manager.rank, N, cp0, cp1, Dloc, True
            )
            N_i_loc, N_j_loc = N // cp0, N // cp1
            epi = GemmA2ASm90.EpilogueArguments(
                alpha=None,
                beta=None,
                mRowVecBroadcast=None,
                mColVecBroadcast=None,
                add_to_output=False,
                rounding_mode=None,
                sr_seed=None,
                recv=_mark__distributed__pe_aligned_general(recv, True),
            )
            _barrier()
            compiled(cA, cB, cD, None, epi, sched, stream, None)
            _drain__distributed__pe_aligned_general()
            exp = _ref_back_gemm_native_2d(
                A,
                Bt,
                pm,
                cp0=cp0,
                cp1=cp1,
                Dloc=Dloc,
                B=Bsz,
                N_i_loc=N_i_loc,
                N_j_loc=N_j_loc,
                N=N,
                world_size=world_size,
            )
            got = recv.permute(2, 3, 4, 0, 1).reshape(Bsz, N_i_loc, N_j_loc, cp * Dloc)
            ref = exp.permute(2, 3, 4, 0, 1).reshape(Bsz, N_i_loc, N_j_loc, cp * Dloc)
            h = compute_error_histogram(got, ref)
            untouched = int((recv == -99.0).all(dim=4).sum().item())
            kind = "straddle" if (N_i_loc % 128 or N_j_loc % 128) else "aligned"
            if dist_manager.rank == 0:
                print(
                    f"\n[pe_aligned 2D-dyn cp={cp0}x{cp1} compile@{anchor} run@{N} ({kind} "
                    f"N_i={N_i_loc} N_j={N_j_loc})] rel_L2={h.rel_l2:.3e} "
                    f"outlier_rows={h.n_outlier_rows} untouched={untouched} "
                    f"(gate below)",
                    flush=True,
                )
            _assert_recv_matches(got, ref, bar=2e-2, what=f"2D-dyn N={N}")
            assert h.n_outlier_rows == 0, f"2D-dyn N={N}: {h.n_outlier_rows} outlier rows"
            assert untouched == 0, f"2D-dyn N={N}: {untouched} unwritten recv cells"
    finally:
        compiled.free()
        _barrier()


@pytest.mark.usefixtures("_nvshmem__distributed__pe_aligned_general")
@pytest.mark.usefixtures("_module_skip__distributed__pe_aligned_general")
@EPI.parametrize_unsupported("N", "mesh", "dyn_cp")
def test_a_declared_unsupported_configuration_is_refused_at_the_front_door(
    apply_mesh, mesh, dist_manager, N, dyn_cp, expected_error, expected_match
):
    """``dyn_cp=True`` must be refused at configure, and refused by our OWN front door.

    dynamic-cp is a hard wall for EVERY A2A store rather than a gap: the reshard peer atoms are a
    compile-time host list carrying per-peer BAKED symmetric base addresses, and there is no
    runtime-base TMA-S2G to retarget them with. Pure-cp A2A also has ``cp == world``, and a
    different world is a different nvshmem job -- so compile-per-cp is the only coherent meaning,
    not a workaround.

    **``front_door_raises`` rather than ``pytest.raises``, which is what the upstream used.** The
    two agree on the type and the message; only the former also rejects a raise that arrived from
    inside ``cute.compile``. That difference is the whole enforcement -- a check that fires during
    tracing is latent, costs the caller a compile before saying no, and reports the toolchain's
    message rather than one naming the argument to change. The refusal here is host-side and must
    stay host-side, because a caller who has to pay a compile to learn that dynamic cp is
    unsupported has been told too late to act on it.

    The sweep comes from the matrix's ``unsupported=`` region, so this test cannot silently stop
    covering the combination: removing the region from the matrix makes
    ``parametrize_unsupported`` refuse at import rather than leave a green test asserting nothing.
    """
    apply_mesh(mesh)
    # N % cp != 0 is a GEOMETRY PRECONDITION, not the refusal under test. `configure_a2a_sharded`
    # checks the reshard split before it reaches any declared region's guard, so at such an N it
    # raises `ValueError: N=... must be divisible by both cp axes` -- a refusal this cell was not
    # emitted to assert. Skipping matches how all 13 other sites in this file treat the same
    # predicate, and it costs no coverage: the region is still exercised at every divisible N on
    # the SAME mesh (23 of them at cp=16), so no facet is lost -- measured, not assumed.
    #
    # CAVEAT for whoever revisits this: the divisibility constraint stays UNDECLARED, so the matrix
    # still cannot see it. Declaring it as an `Unsupported` region is defensible, but it must land
    # AFTER the multi-region emission fix (kernel_matrix.parametrize_unsupported now admits every
    # matching region) -- declaring it before that would have added 8 more overlapping cells to a
    # mechanism that dropped all but the first match, which is the defect that produced 19 uncaught
    # exceptions and a 16-rank desync.
    axis_sizes = _cp_axis_sizes(dist_manager)
    cp_total = 1
    for _s in axis_sizes:
        cp_total *= _s
    if N % cp_total != 0:
        rank_invariant_skip(
            f"N={N} not divisible by cp={cp_total} (axes {axis_sizes}); the reshard split must be "
            f"integral before any declared region's guard is reached",
            because=_GEOMETRY_UNIFORM_BECAUSE,
        )
    placements, pm = _trimul_placements(dist_manager)
    mesh = _session_cp_mesh(dist_manager)
    a_dtype = torch2cute_dtype_map[torch.bfloat16]
    gemm = GemmA2ASm90(
        Float32,
        a_dtype,
        TILE__distributed__pe_aligned_general,
        (1, 1, 1),
        pingpong=False,
        is_persistent=True,
    )
    with front_door_raises(expected_error, expected_match):
        gemm.configure_a2a_sharded(
            mesh,
            placements,
            pe_map=pm,
            B=Bsz,
            N=N,
            gemm_native=True,
            dynamic=True,
            dyn_cp=dyn_cp,
            pe_aligned_tiling=True,
        )


@pytest.mark.usefixtures("_nvshmem__distributed__pe_aligned_general")
@pytest.mark.usefixtures("_module_skip__distributed__pe_aligned_general")
@pytest.mark.xfail(
    raises=ValueError,
    strict=True,
    reason="2-D 16-B partial-tile constraint: pe_aligned per-peer N_i/N_j must be "
    "%8==0 (bf16); guarded in configure_a2a_sharded (silent corruption else) — A2/#6",
)
@EPI.parametrize("mesh")
def test_pe_aligned_2d_misaligned_xfail(apply_mesh, mesh, dist_manager):
    """A2/#6 HARDENING (xfail): a 2-D pe_aligned config whose per-peer extent is %8!=0 must be
    REJECTED at configure (16-B TMA-S2G partial-tile alignment on both axes). N=536, cp0=2 -> N_i=268
    (%8==4) -> the guard raises ValueError. strict xfail on raises=ValueError: if the guard is ever
    removed (configure stops raising) this XPASSES -> strict turns that into a FAILURE, alerting the
    recurrence of the silent-2D-corruption pealignA proved fundamental (compile@N==run@N still fails)."""
    apply_mesh(mesh)
    axis_sizes = _cp_axis_sizes(dist_manager)
    if len(axis_sizes) != 2:
        rank_invariant_skip(
            f"2-D pe_aligned alignment guard needs a 2-D cp mesh; session mesh is {axis_sizes}",
            because=_GEOMETRY_UNIFORM_BECAUSE,
        )
    placements, pm = _trimul_placements(dist_manager)
    mesh = _session_cp_mesh(dist_manager)
    a_dtype = torch2cute_dtype_map[torch.bfloat16]
    gemm = GemmA2ASm90(
        Float32,
        a_dtype,
        TILE__distributed__pe_aligned_general,
        (1, 1, 1),
        pingpong=False,
        is_persistent=True,
    )
    gemm.configure_a2a_sharded(
        mesh, placements, pe_map=pm, B=Bsz, N=536, gemm_native=True, pe_aligned_tiling=True
    )


@pytest.fixture(scope="session")
def _module_skip__distributed__pe_aligned_general():
    pytest.importorskip("nvshmem.core")
    pytest.importorskip("fold_cp_ops.distributed.gemm_a2a_epi")


# --------------------------------------------------------------------------- #
# CUBIN byte identity -- the bring-back's first gate, measured on CODE rather
# than on output tensors.
#
# Five tests in this tree carry ``byte_identical`` in their name and every one of
# them compares OUTPUT; a kernel can compute the right numbers from different
# instructions, which is exactly how a missing race guard once survived an output
# comparison here. This is the first test that digests the emitted code.
#
# It does TWO things, and only the second needs `main`:
#   1. NON-VACUITY, in-tree and self-contained: the declared config pool must emit
#      DISTINGUISHABLE code. A sweep whose configs all digest the same is reporting
#      a key collision as identity, and that failure looks exactly like success.
#   2. HARVEST: with ``CPO_CUBIN_OUT`` set it writes one JSON digest per config, so
#      an out-of-tree driver can run the same test in `main`'s tree and diff.
#
# THE POOL IS `main`'s, NOT INVENTED HERE. ``fused_trimul_autotune._grid()`` is
# fully determined by ``(cp1, has_ib_peers, N_j_loc)``, and five distinct configs
# span the whole space: ``(128,128,{1,2,4})`` and ``(128,256,{1,2})``.
#
# ONE CONFIG PER PROCESS is the caller's job, not this test's. The disk cache must
# be off (``CPO_CACHE_ENABLED=0``, read once at import) AND the process fresh,
# because the in-process memo is a second, independent way to be handed something
# other than a fresh compile. `digest_export` refuses a cache hit by name, so a
# violated protocol fails loudly rather than reporting false identity.
# --------------------------------------------------------------------------- #
#: `main`'s declared byte-identity pool: (tile_m, tile_n, cluster_n).
_CUBIN_CONFIGS = ((128, 128, 1), (128, 128, 2), (128, 128, 4), (128, 256, 1), (128, 256, 2))


def _cubin_label(tile_m, tile_n, cluster_n, N, D):
    """Filesystem-safe identity for one harvested configuration.

    Purpose:
        The digest of two configurations must never land on one path -- `digest_export`
        documents that as the failure it exists to make visible, since the resulting
        "identical" verdict is indistinguishable from a real one.

    Args:
        tile_m, tile_n: the CTA tile shape; must be positive ints.
        cluster_n: the N-axis cluster width; must be a positive int.
        N, D: the token extent and feature width the compile was anchored at; positive ints.
            They are part of the label because a STATIC-shape compile bakes them, so the same
            config at two shapes is two artifacts, not one.

    Returns:
        A label containing every varying input, so no two cells of a sweep collide.
    """
    return f"t{tile_m}x{tile_n}_cn{cluster_n}_N{N}_D{D}"


@pytest.mark.usefixtures("_nvshmem__distributed__a2a_fusion")
@pytest.mark.usefixtures("_module_skip__distributed__a2a_fusion")
@pytest.mark.parametrize(
    "tile_m,tile_n,cluster_n",
    _CUBIN_CONFIGS,
    ids=[f"t{a}x{b}_cn{c}" for (a, b, c) in _CUBIN_CONFIGS],
)
@EPI.parametrize(
    "N",
    "D",
    "mesh",
    cells=[
        (n, d, m)
        for (n, d) in ((256, 4), (512, 8))
        for m in EPI.axis("mesh").values
        if not _misaligned_2d(n, m)
    ],
    because=(
        "byte identity is a property of the EMITTED CODE at a fixed anchor, and the axis being "
        "swept here is the CONFIG pool, not the shape pool -- the five configs are main's own "
        "`_grid()` span. Widening the shape pool multiplies compiles by the config count for no "
        "extra discrimination: a shape enters the cubin only through the baked extents, which the "
        "correctness tests above already sweep across the full N/D pool. The out-of-tree "
        "ours-vs-main driver re-anchors this same test at other shapes via -k when it wants them."
    ),
)
@numeric_exempt(
    "compares COMPILED CODE, not numbers: there is no computed tensor and no reference to "
    "compare it against. The subject is the digest of the cubin's code sections."
)
def test_back_gemm_native_cubin_distinguishable(
    apply_mesh, mesh, dist_manager, device, world_size, N, D, tile_m, tile_n, cluster_n, tmp_path
):
    """Digest the back A2A store's cubin, and prove the config pool is distinguishable.

    Functionality & semantics:
        Compiles ``GemmA2ASm90``'s design-E 5-D GEMM-native store at one declared config,
        exports the host object, and digests only its CUDA-ELF CODE sections (`code_sections`
        strips the mangled names, so a pure rename does not read as a difference). Asserts the
        artifact carries ``.text``. When ``CPO_CUBIN_OUT`` names a directory the digest is written
        there as JSON under `_cubin_label`, which is how the ours-vs-`main` comparison is fed;
        with the variable unset the export goes to ``tmp_path`` and is discarded.

        The DISTINGUISHABILITY half is deliberately NOT asserted inside one process: the protocol
        is one config per process, so a single test item sees exactly one digest. The driver
        compares the harvested JSONs, and an all-identical harvest is the collision signal.

    Input requirements:
        ``tile_m``/``tile_n`` must be a tile shape the SM90 GEMM accepts and ``cluster_n`` a
        cluster width whose ``(1, cluster_n, 1)`` shape has active clusters available, else the
        compile raises rather than mis-emitting. ``N`` must satisfy the store's geometry -- ``D %
        cp == 0`` and ``(N // cp) % 128 == 0`` -- which is checked and skipped rank-invariantly,
        because a per-rank skip here would deadlock every peer in the next collective. The disk
        cache MUST be off in a fresh process; `digest_export` raises `TypeError` naming the cache
        if it is not, so the protocol is enforced rather than trusted.

    Raises:
        ``TypeError`` from `digest_export` when handed a cache hit instead of a compiled object;
        ``RuntimeError`` from `code_sections` when the export carried no CUDA ELF.
    """
    apply_mesh(mesh)
    cp = world_size
    if D % cp != 0:
        rank_invariant_skip(f"D={D} not divisible by cp={cp}", because=_GEOMETRY_UNIFORM_BECAUSE)
    if (N // cp) % tile_m != 0:
        rank_invariant_skip(
            f"N_loc=N/cp={N // cp} not a multiple of cta_tile_M={tile_m} at cp={cp}",
            because=_GEOMETRY_UNIFORM_BECAUSE,
        )
    Dloc, B = D // cp, 1
    N_loc = N // cp
    M, K, L = N, 256, (D // cp) * 1
    pm = _pe_map(dist_manager)
    _skip_coupled_cross_node(pm, "back gemm-native cubin digest")

    torch.manual_seed(4321 + dist_manager.rank)
    A = torch.randn(L, M, K, device=device, dtype=torch.bfloat16) / (K**0.5)
    Bt = torch.randn(L, N, K, device=device, dtype=torch.bfloat16) / (K**0.5)
    with torch.cuda.use_mem_pool(DistributedManager.symmetric_mempool(device)):
        recv = torch.empty((cp, Dloc, B, N_loc, N), dtype=torch.bfloat16, device=device)
    recv.zero_()

    pe_table = tuple(int(x) for x in pm.cp_pe_table.tolist())
    cfg = dict(cp=cp, my_cp_rank=int(pm.my_cp_rank), B=B, N_loc=N_loc, pe_table=pe_table)
    compiled, _ = _compile_back_gemm_native(
        A,
        Bt,
        recv,
        (tile_m, tile_n),
        gemm_native_cfg=cfg,
        cluster_shape_mnk=(1, cluster_n, 1),
    )
    label = _cubin_label(tile_m, tile_n, cluster_n, N, D)
    out_dir = os.environ.get("CPO_CUBIN_OUT")
    dest = Path(out_dir) if out_dir else Path(str(tmp_path))
    dest.mkdir(parents=True, exist_ok=True)
    try:
        # RANK-UNIQUE path. Every rank compiles and exports; a shared filename makes two writers
        # race on one file and the reader then sees a TORN object, which surfaces as "no CUDA ELF
        # embedded" -- a message that points at the exporter rather than at the collision.
        digest = digest_export(compiled.compiled, str(dest / f"{label}.r{dist_manager.rank}.o"))
        assert_has_code(digest, f"back gemm-native {label}")
        if out_dir and dist_manager.rank == 0:
            (dest / f"{label}.json").write_text(
                json.dumps({"label": label, "cp": cp, "sections": digest}, indent=1, sort_keys=True)
            )
        print(
            f"\n[cubin {label} cp={cp}] "
            + " ".join(f"{k}:{v[0]}B:{v[1][:12]}" for k, v in sorted(digest.items())),
            flush=True,
        )
    finally:
        compiled.free()
        _nvshmem_barrier()


# --------------------------------------------------------------------------- #
# The back's 32-bit peer-offset exposure -- the front's defect, asked of the back.
#
# The front A2A's IB drain formed `dst_feat * epi_n * M_full` in 32 bits and wrapped
# at 2**31, faulting cross-node only, and only with a STATIC token extent. The back's
# decoupled putwarp drain reaches its peer recv by the SAME construction --
# `g_wide[(None, None, i_local, j_grp, Int32(my_cp_rank), d, b)]` then
# `.iterator.toint()` -- over a recv laid out (cp, Dloc, B, N_loc, N).
#
# The arithmetic says the two are the same question. The cp-mode stride is
# `Dloc*B*N_loc*N` ELEMENTS, indexed by `my_cp_rank`, and summing the cp, d and b
# terms gives a maximum offset of about `(2*Dloc - 1) * N_loc * N` -- which is just
# the recv's own element COUNT. So the threshold is not obscure: the drain is exposed
# exactly when the recv holds 2**31 elements or more, i.e. 4 GiB at bf16.
#
# THE CELLS ARE PRE-REGISTERED, and that is the point. Six successive models of the
# front defect each fit every measured row and each turned out false; what settled it
# was a cell where the surviving model and its rival predict OPPOSITE outcomes, called
# in advance. So the pair below is declared with its prediction attached rather than
# chosen to fit whatever the first run does:
#
#   under  Dloc=8, N=12280 -> 1.21e9 recv elements (0.56 x 2**31) -- PASS, both modes
#   over   Dloc=8, N=20000 -> 3.20e9 recv elements (1.49 x 2**31) -- PASS dynamic;
#                                                                    FAULT static, if
#                                                                    the back shares
#                                                                    the front's defect
#
# CROSS-NODE IS REQUIRED and is not a detail: the all-P2P arm reaches the recv through
# a TMA descriptor, which the hardware addresses in 64 bits, so a single-node run
# cannot fault however large the recv gets. The front measured exactly that -- one node
# passed at a 17.18 GB recv while two nodes faulted at 8.59 GB.
# --------------------------------------------------------------------------- #
#: (label, Dloc, N) with the recv element count each produces. See the block comment.
_I32_CELLS = (("under", 8, 12280), ("over", 8, 20000))


@matrix_exempt(
    "the swept axis is the recv's ELEMENT COUNT against the 2**31 i32 boundary, which no declared "
    "axis expresses: the values needed (12280, 20000) are far outside the N pool, and adding them "
    "there would enlarge every other test's grid to carry a threshold only this test reads. The "
    "mesh axis is not swept either, and that is not evasion -- the drain under test is the 1-D one, "
    "so every factored spec would emit a rank-invariant skip, and crossing an out-of-pool N with "
    "the mesh axis is precisely what the audit refused here (correctly: it lands in the "
    "_misaligned_2d region, which must RAISE, under a test that asserts success)."
)
@pytest.mark.usefixtures("_nvshmem__distributed__ib_ring")
@pytest.mark.usefixtures("_module_skip__distributed__ib_ring")
@pytest.mark.parametrize("shape_mode", ["dynamic", "static"])
@pytest.mark.parametrize("regime,Dloc,N", _I32_CELLS, ids=[c[0] for c in _I32_CELLS])
def test_ib_ring_cluster_drain_1d_addresses_a_recv_past_two_gig(
    apply_mesh, dist_manager, device, world_size, regime, Dloc, N, shape_mode
):
    """The 1-D cluster drain stores correctly into a recv whose element count exceeds 2**31.

    Functionality & semantics:
        Configures and runs the decoupled pe_aligned ib_ring cluster drain over a flat (i-only) cp
        shard at both an under- and an over-threshold recv, in both shape modes, and asserts every
        recv element was written. `assert_written` rather than a value comparison: the defect under
        test corrupts the DESTINATION ADDRESS, so its signature is an element that never receives a
        store (or an illegal address that kills the process outright), not a wrong number. A value
        gate at this size would also need an fp32 reference of twice the recv, which at the over
        cell is 12.8 GB of avoidable allocation.

    Input requirements:
        Needs a FLAT cp mesh; a factored one takes the 2-D path, whose static mode the kernel
        refuses at its front door, and the skip is rank-invariant because mesh shape is job-uniform.
        Needs the recv and the fp32-free staging to fit: the over cell allocates 6.4 GB of symmetric
        recv per rank, so an OutOfMemoryError is a legitimate skip and NOT a memory-estimate gate.
        Needs at least one IB peer to mean anything -- on an all-P2P job the drain is const_expr
        elided and this test would pass while measuring nothing, so that case is skipped by name.

    Raises:
        Nothing directly. A failure of the defect it guards is a CUDA illegal address, which kills
        the process rather than raising -- the run report's progress file names the test that died.
    """
    # The mesh is BUILT here, flat, rather than drawn from the matrix axis. `apply_mesh` is what
    # constructs the session cp mesh at all -- without a call the manager's `device_mesh` is None
    # and every cell dies in a fixture helper rather than in the kernel. Flat because the drain
    # under test is the 1-D one; sweeping the mesh axis would emit seven rank-invariant skips and
    # one duplicate of this.
    apply_mesh((("cp", world_size),))
    assert get_device_capacity()[0] == 9, "SM90 only"
    axis_sizes = _cp_axis_sizes(dist_manager)
    if len(axis_sizes) != 1:
        rank_invariant_skip(
            f"the 1-D cluster drain needs a FLAT cp mesh; got {axis_sizes}",
            because=_GEOMETRY_UNIFORM_BECAUSE,
        )
    cp, B, cluster_n, rd = axis_sizes[0], 1, 1, 2
    if N % cp or N % 8 or (N // cp) % TILE[0] == 0:
        rank_invariant_skip(
            f"N={N} is not a valid 1-D straddle anchor at cp={cp}",
            because=_GEOMETRY_UNIFORM_BECAUSE,
        )
    pm = _pe_map(dist_manager)
    if all(build_p2p_table(tuple(int(x) for x in pm.cp_pe_table.tolist()))):
        rank_invariant_skip(
            "every peer is NVLink-reachable, so the IB SIMT drain this test exists for is "
            "const_expr-elided and the run would pass while exercising nothing",
            because=_GEOMETRY_UNIFORM_BECAUSE,
        )
    pe_table = tuple(int(x) for x in pm.cp_pe_table.tolist())
    n_clusters = get_max_active_clusters(cluster_n)
    stream = cutlass_torch.current_stream()
    sched = make_scheduler_args(n_clusters, Int32(8), None, None)
    dyn = shape_mode == "dynamic"

    # `capacity_gate`, not a bare `except torch.OutOfMemoryError` + `rank_invariant_skip`, and the
    # replaced form was wrong in THREE independent ways:
    #   * it caught only the caching allocator's type, so the SYMMETRIC recv's
    #     `RuntimeError: nvshmem_malloc failed` FAILED the cell instead of skipping it;
    #   * `rank_invariant_skip` is UNILATERAL, and free memory is not job-uniform -- one rank skipped
    #     alone while its peers failed and the ranks that did allocate stored into a peer buffer that
    #     no longer existed, an illegal access that poisoned the context for nine later cells. The
    #     `because=` cited GEOMETRY uniformity for a MEMORY predicate;
    #   * its SCOPE ended here, but the largest allocation in this cell is the pair of reference
    #     buffers `assert_written` makes below (2 x the recv), which sat outside it entirely.
    def _alloc():
        with torch.cuda.use_mem_pool(DistributedManager.symmetric_mempool(device)):
            stage_buf = torch.empty((n_clusters, 128, N), dtype=torch.bfloat16, device=device)
        stage_buf.zero_()
        stage_c = from_dlpack(stage_buf, assumed_align=16)
        if dyn:
            stage_c = stage_c.mark_layout_dynamic()
        pe_dev_c = from_dlpack(pm.cp_pe_table.to(torch.int32).contiguous(), assumed_align=4)
        return (stage_buf, stage_c, pe_dev_c) + _build_inputs(
            device, dist_manager.rank, N, cp, Dloc, B, shape_mode=shape_mode
        )

    # `stage_buf` is unpacked and never read again ON PURPOSE: the cute view `stage_c` aliases its
    # storage, so dropping the reference would free memory the kernel still reads.
    stage_buf, stage_c, pe_dev_c, A, Bt, recv, (cA, cB, cD) = capacity_gate(  # noqa: RUF059
        f"recv+staging (symmetric) Dloc={Dloc} N={N} cp={cp}", _alloc
    )
    gemm = GemmA2ASm90(
        Float32,
        torch2cute_dtype_map[torch.bfloat16],
        TILE,
        (1, cluster_n, 1),
        pingpong=False,
        is_persistent=True,
    )
    gemm.configure_a2a_gemm_native(
        cp=cp,
        my_cp_rank=int(pm.my_cp_rank),
        B=B,
        N_loc=N // cp,
        pe_table=pe_table,
        N=N,
        dynamic=dyn,
        arbitrary_n=True,
        pe_aligned_tiling=True,
        ib_ring=True,
        ib_quiet=True,
        decoupled=True,
        producer_tma=True,
        consumer_strided=True,
        consumer_strided_putwarp=True,
        ring_depth=rd,
        cluster_drain=True,
        cluster_n=cluster_n,
    )
    epi = _epi_cluster(gemm, recv, stage_c, pe_dev_c, shape_mode=shape_mode)
    compiled = compile_gemm_with_bitcode(
        gemm, cA, cB, cD, None, epi, sched, stream, None, register=True
    )
    elems = cp * Dloc * B * (N // cp) * N
    tag = f"back-i32 {regime} {shape_mode} cp={cp} Dloc={Dloc} N={N} recv={elems / 2**31:.2f}x2^31"
    try:

        def run(out):
            """Pre-fill the SYMMETRIC recv from `out`, store into it, and bring the result back.

            `assert_written` allocates the two differently-pre-filled buffers itself, but the
            kernel can only store into the symmetric recv -- so the buffers are used as carriers:
            copied IN as the pre-fill, copied OUT as the result. Any element the drain never
            addresses keeps its pre-fill, and the two runs then disagree exactly there.
            """
            recv.copy_(out)
            _nvshmem_barrier()
            compiled(cA, cB, cD, None, epi, sched, stream, None)
            _drain()
            out.copy_(recv)

        # Inside the gate too: `assert_written` allocates TWO buffers of the recv's size on the
        # DEFAULT pool, which is the single largest allocation in this cell.
        capacity_gate(
            f"assert_written references (2x recv) {tag}",
            lambda: assert_written(run, recv.shape, recv.dtype, device, what=tag),
        )
        # DID-IT-RUN WITNESS, written to a file rather than printed. `-q` captures stdout for a
        # PASSING test, so a print is invisible in exactly the case that needs proving, and a bare
        # "1 passed" cannot distinguish this cell from any other item a -k expression also matched.
        # A pre-registered prediction is only evidence if the row it predicts is known to have run.
        witness = os.environ.get("CPO_I32_WITNESS")
        if witness:
            with open(witness, "a") as fh:
                fh.write(f"{tag} rank={dist_manager.rank} WRITTEN_OK\n")
        if dist_manager.rank == 0:
            print(f"\n[{tag}] every recv element written PASS", flush=True)
    finally:
        compiled.free()
        _nvshmem_barrier()


@matrix_exempt(
    "the swept axis is main's `_grid()` CONFIG pool -- (tile_m, tile_n, cluster_n) -- which the "
    "matrix does not declare, because it is an autotune grid rather than a shape. The mesh is not "
    "swept for the same reason as the i32 gate above: the drain under test is the 1-D one."
)
@pytest.mark.usefixtures("_nvshmem__distributed__ib_ring")
@pytest.mark.usefixtures("_module_skip__distributed__ib_ring")
@pytest.mark.parametrize("shape_mode", ["dynamic", "static"])
@pytest.mark.parametrize(
    "tile_m,tile_n,cluster_n",
    _CUBIN_CONFIGS,
    ids=[f"t{a}x{b}_cn{c}" for (a, b, c) in _CUBIN_CONFIGS],
)
@numeric_exempt("digests COMPILED CODE; no tensor is produced and none is compared")
def test_ib_ring_cluster_drain_1d_cubin(
    apply_mesh, dist_manager, device, world_size, tile_m, tile_n, cluster_n, shape_mode, tmp_path
):
    """Digest the CLUSTER-DRAIN cubin -- the code path the i64 widening actually changed.

    Functionality & semantics:
        The sibling harvest on the plain gemm-native store cannot cover this. That store is the
        COUPLED one; the 64-bit peer offsets live in `_cluster_drain_loop` and
        `_cluster_drain_loop_multislot`, which are `const_expr`-elided unless the full decoupled
        pe_aligned ib_ring stack is configured. Measured: the coupled harvest's ten digests are
        byte-for-byte unchanged across that edit, which is correct and is also exactly why it proves
        nothing about it. This configures the drain and digests what that emits.

        Writes one JSON digest per cell under ``CPO_CUBIN_OUT`` when set, for the out-of-tree
        ours-vs-`main` comparison; otherwise exports to ``tmp_path`` and discards.

    Input requirements:
        A FLAT cp mesh (built here). ``N`` must be a valid 1-D straddle anchor at this cp --
        ``N % cp == 0``, ``N % 8 == 0``, ``N_loc % tile_m != 0`` so ``arbitrary_n`` stays on -- and
        ``ntj % cluster_n == 0`` for the bounded band; a value failing either is skipped
        rank-invariantly, because a per-rank skip under ``tests/distributed/`` is a deadlock.
        **The compile is what is under test, so this must run on the venue whose codegen it means
        to capture**: ``has_ib_peers`` is a VENUE property, false on one node and true across
        nodes, and it gates four `const_expr` branches -- so the same config emits DIFFERENT code
        on the two venues and both are worth harvesting. The disk cache MUST be off in a fresh
        process; `digest_export` raises naming the cache otherwise.

    Raises:
        ``RuntimeError`` from `code_sections` if the export carried no CUDA ELF -- most often two
        ranks racing on one path, which is why the object name carries the rank.
    """
    apply_mesh((("cp", world_size),))
    assert get_device_capacity()[0] == 9, "SM90 only"
    axis_sizes = _cp_axis_sizes(dist_manager)
    if len(axis_sizes) != 1:
        rank_invariant_skip(
            f"the 1-D cluster drain needs a FLAT cp mesh; got {axis_sizes}",
            because=_GEOMETRY_UNIFORM_BECAUSE,
        )
    # Dloc defaults to 2 (a cheap compile); CPO_DRAIN_CUBIN_DLOC pins it so the harvest can reach
    # the DECLARED D pool, where D = cp*Dloc -- at cp=16 that is Dloc in {8,16,24,32} for
    # D in {128,256,384,512}. Same reason as CPO_DRAIN_CUBIN_N: a harness that picks its own
    # convenient shape cannot be pointed at the pool the gate is defined over.
    cp, B, rd = axis_sizes[0], 1, 2
    Dloc = int(os.environ.get("CPO_DRAIN_CUBIN_DLOC", 2))
    # SEARCH the anchor rather than picking from DIFF1D_NS. Measured: that five-value list contains
    # NO N satisfying the bounded-band gate at cluster_n in {2, 4}, so three of the five declared
    # configs harvested NOTHING and the sweep reported 4 of 10 labels while looking complete. The
    # constraints admit thousands of values (3495 for cn=2, 1755 for cn=4, within 8..60000 at cp=2),
    # so the gap was the pool, not the geometry -- which a search establishes and reading five
    # values cannot. Smallest feasible N keeps the compile cheap; this test digests code, not time.
    # CPO_DRAIN_CUBIN_N pins the anchor so the harvest can run at the DECLARED N_token ladder
    # rather than at a searched straddle. Both are wanted: the search finds the smallest cell that
    # exercises arbitrary_n, and the declared ladder is where the gates must be
    # demonstrated on. Measured: every declared N gives an ALIGNED N_loc (N/cp % 128 == 0) at both
    # cp=2 and cp=16, and the stack still configures with arbitrary_n True -- `ib_ring` suppresses
    # the auto-reduce that would otherwise turn it off and collapse the stack.
    _pinned = os.environ.get("CPO_DRAIN_CUBIN_N")
    N = (
        int(_pinned)
        if _pinned
        else next(
            (
                n
                for n in range(8, 60000, 8)
                if n % cp == 0
                and (n // cp) % tile_m != 0
                and n // cp > tile_m
                and (((n + tile_n - 1) // tile_n) % cluster_n == 0)
            ),
            None,
        )
    )
    if _pinned and (N % cp or (((N + tile_n - 1) // tile_n) % cluster_n)):
        rank_invariant_skip(
            f"pinned N={N} invalid at cp={cp} tile_n={tile_n} cn={cluster_n}",
            because=_GEOMETRY_UNIFORM_BECAUSE,
        )
    if N is None:
        rank_invariant_skip(
            f"no 1-D straddle anchor in {DIFF1D_NS} for cp={cp} tile=({tile_m},{tile_n}) "
            f"cn={cluster_n}",
            because=_GEOMETRY_UNIFORM_BECAUSE,
        )
    pm = _pe_map(dist_manager)
    pe_table = tuple(int(x) for x in pm.cp_pe_table.tolist())
    n_clusters = get_max_active_clusters(cluster_n)
    stream = cutlass_torch.current_stream()
    sched = make_scheduler_args(n_clusters, Int32(8), None, None)
    dyn = shape_mode == "dynamic"
    with torch.cuda.use_mem_pool(DistributedManager.symmetric_mempool(device)):
        stage_buf = torch.empty((n_clusters, 128, N), dtype=torch.bfloat16, device=device)
    stage_buf.zero_()
    stage_c = from_dlpack(stage_buf, assumed_align=16)
    if dyn:
        stage_c = stage_c.mark_layout_dynamic()
    pe_dev_c = from_dlpack(pm.cp_pe_table.to(torch.int32).contiguous(), assumed_align=4)
    _, _, recv, (cA, cB, cD) = _build_inputs(
        device, dist_manager.rank, N, cp, Dloc, B, shape_mode=shape_mode
    )
    gemm = GemmA2ASm90(
        Float32,
        torch2cute_dtype_map[torch.bfloat16],
        (tile_m, tile_n),
        (1, cluster_n, 1),
        pingpong=False,
        is_persistent=True,
    )
    gemm.configure_a2a_gemm_native(
        cp=cp,
        my_cp_rank=int(pm.my_cp_rank),
        B=B,
        N_loc=N // cp,
        pe_table=pe_table,
        N=N,
        dynamic=dyn,
        arbitrary_n=True,
        pe_aligned_tiling=True,
        ib_ring=True,
        ib_quiet=True,
        decoupled=True,
        producer_tma=True,
        consumer_strided=True,
        consumer_strided_putwarp=True,
        ring_depth=rd,
        cluster_drain=True,
        cluster_n=cluster_n,
    )
    epi = _epi_cluster(gemm, recv, stage_c, pe_dev_c, shape_mode=shape_mode)
    compiled = compile_gemm_with_bitcode(
        gemm, cA, cB, cD, None, epi, sched, stream, None, register=False
    )
    ib = bool(getattr(gemm, "_a2a_has_ib_peers", True))
    label = f"drain_t{tile_m}x{tile_n}_cn{cluster_n}_N{N}_Dloc{Dloc}_{shape_mode}_ib{int(ib)}"
    out_dir = os.environ.get("CPO_CUBIN_OUT")
    dest = Path(out_dir) if out_dir else Path(str(tmp_path))
    dest.mkdir(parents=True, exist_ok=True)
    try:
        digest = digest_export(compiled.compiled, str(dest / f"{label}.r{dist_manager.rank}.o"))
        assert_has_code(digest, f"cluster drain {label}")
        if out_dir and dist_manager.rank == 0:
            (dest / f"{label}.json").write_text(
                json.dumps({"label": label, "cp": cp, "sections": digest}, indent=1, sort_keys=True)
            )
        print(
            f"\n[cubin {label}] .text {digest['.text'][0]}B {digest['.text'][1][:12]}", flush=True
        )
    finally:
        compiled.free()
