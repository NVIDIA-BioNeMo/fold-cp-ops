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
"""Tests for ``fold_cp_ops.distributed.gemm_sm90_a2a`` -- the back-einsum A2A kernel.

Everything here is HOST-side: constructor guards, drain-variant validity, and one source-level
invariant. The kernel's own numerics belong to the fused-store path and are not re-asserted here.

**Four properties, and the last two are new work rather than a port.**

1. **A COUPLED back store must be REFUSED on a cross-node mesh.** The coupled store TMA-S2Gs into a
   peer's symmetric heap via ``nvshmem_ptr``, which returns NULL for an IB peer -- so cross-node it
   took an illegal memory access that surfaced later, at a barrier, unattributed. It is NVLink-only
   *by hardware*; the defect was the missing guard, and the supported cross-node path (the decoupled
   ring drain) already exists. Host-only by construction: ``build_p2p_table`` is monkeypatched, so
   the guard is asserted at the exact triggering topology without a 2-node allocation.
2. **With A2A unconfigured, the store seam delegates to the parent.** That is the
   "distributed OFF == parent" principle at the one seam that could break it.
3. **An atom's baked peer and its signal's PE come from the SAME index** -- see
   :func:`test_a_peer_atom_and_its_signal_are_indexed_together`. This is the SOURCE-LEVEL HALF of
   an obligation whose runtime half is still open; the docstring there says exactly what it does and
   does not cover.

4. **The token extent AND THE MESH reach the BAKED per-peer geometry, and the band it sizes
   COVERS.** The store derives every per-peer extent from one number and one shard shape --
   ``N_loc = N // cp0``, ``N_j_loc = N // cp1``, the per-peer tile counts, and the bounded drain
   band ``run_j`` -- at CONFIGURE time, on the host, with nothing allocated. So the whole declared
   ``N_token`` ladder is assertable here for the price of the smallest value, INCLUDING 12288. The
   large-shape memory carve-out does not reach a subject that allocates nothing.
   See :func:`test_the_token_extent_reaches_the_baked_per_peer_geometry`, and the ``N_token`` axis
   for why an OFF-GRID value is what makes the covering assertions able to fail at all.

**The MESH is an axis here, and it is a configure argument rather than a launch.** ``cp``,
``cp_axis_sizes`` and ``pe_table`` are plain arguments to ``configure_a2a_gemm_native``, so all
eight declared specs -- including the three needing 16 ranks -- are swept by ONE run on any SM90
box, with no allocation and no process group. That is the cheap half of a mesh; the expensive half
(what the geometry then ROUTES over a real fabric) belongs to the launching sibling
``test_gemm_a2a_epi.py``, which declares the same pool. Two things the axis therefore does NOT
claim, both stated at its declaration: ``cross_node`` names a rank count rather than a fabric that
was crossed, and this module does not write the shared mesh coverage ledger. Unpinning ``cp`` from
the 4 every call site used to hardcode is what let the third unsupported region reach a second
value, and what first let a WORKFLOW-LADDER ``N_token`` reach the covering assertions at all.

**On the drain-variant region.** ``tile_m=256`` on a DEFERRED drain variant must config-reject,
while every other declared ``(tile, drain_variant)`` pair must configure clean. That is a genuine
combination of declared axis values rather than a malformed input, so it is an ``Unsupported``
region rather than a direct raises-test -- the first in this bring-back where the machinery's
region form is the right one.

**NOT ported: ``main``'s ``test_orthogonality_gpu_sass_pointer``.** Its first statement is an
unconditional ``pytest.skip`` and its own docstring says the real proof runs off-box. A test that
never runs anywhere asserts nothing, and its green is indistinguishable from every other green in
the file. The property it was standing in for is recorded here instead: **the GPU SASS
byte-identity tier is established by digesting the compiled object, not by this suite** -- see the
route recorded in the plan (``export_to_c`` on what ``compile_gemm_with_bitcode`` returns, at the
declared configs, digested with ``cubin_identity.py``).

**Obligations A and B, stated so neither is mistaken for done.** A -- a test that fails if a peer
view's *alignment* downgrades into a real TMA atom -- is NOT here: measured, ``make_tiled_tma_atom``
accepts a 2-byte-aligned peer view without complaint, and neither the atom nor its tensor exposes
the alignment afterwards, so the property is only observable through a real ``cute.copy`` and
therefore through the kernel. B's host-side mapping half is in
``tests/distributed/test_peer_tma_atoms.py``; its source-level half is below; its runtime half is
open.
"""

import ast
import inspect
import math

import pytest
import torch

from cutlass import Float32

from fold_cp_ops._internal.arch import get_device_capacity
from fold_cp_ops._internal.compile_time.cute_dsl_utils import torch2cute_dtype_map
from fold_cp_ops.distributed.gemm_a2a_epi import GemmA2ASm90
from fold_cp_ops.testing.collective_guard import rank_invariant_skip
from fold_cp_ops.testing.kernel_matrix import (
    Axis,
    KernelMatrix,
    Unsupported,
    computes_nothing_numeric,
    front_door_raises,
    matrix_exempt,
    shape_mode_axis,
)
from fold_cp_ops.testing.numeric_guard import numeric_exempt


def _mesh_ranks(spec) -> int:
    """Ranks a mesh spec needs -- the product of every group size, subgrids flattened.

    Purpose
        Two jobs. It is the flat ``cp`` :func:`_configure_drain` passes, because this module's pool
        is pure context-parallel and every mesh dim carries the all-to-all; and it is what the
        ``mesh`` axis's ``single_node``/``cross_node`` facets read.

    Semantics
        Mirrors the identically-named helpers in ``test_dual_gated_gemm_a2a.py`` and
        ``test_gemm_a2a_epi.py``. Restated rather than imported: each of those is a test module's
        private helper, and importing one would couple this file's pool to a module it does not
        otherwise depend on.

    Args:
        spec: Tuple of ``(name, size)`` pairs, as the pool below declares them. ``size`` is an
            ``int`` or a ``tuple[int, ...]`` (a subgrid). A malformed spec is a declaration bug and
            raises here, at import, rather than producing a wrong rank count later.

    Returns:
        The rank count the spec requires.
    """
    total = 1
    for _, size in spec:
        total *= math.prod(size) if isinstance(size, tuple) else size
    return total


def _cp_axis_sizes(spec):
    """The ``cp_axis_sizes`` tuple a mesh spec configures with: ``(cp0,)`` flat, ``(cp0, cp1)`` 2-D.

    Purpose
        ONE translation from a declared mesh spec to the kernel's own argument, so the axis pool and
        the call cannot drift. ``configure_a2a_gemm_native`` refuses more than two cp axes, which is
        why the pool declares at most a 2-tuple.

    Args:
        spec: A value from the ``mesh`` pool. Sizes are flattened across groups in declaration
            order, so a pool that later grows a second group keeps working without a second spelling
            here.

    Returns:
        A tuple of ints whose product is :func:`_mesh_ranks`. Length 1 is the 1-D (``cp1 == 1``)
        identity case the kernel skips its whole per-axis block for; length 2 is the 2-D shard.
    """
    sizes = []
    for _, size in spec:
        sizes.extend(size if isinstance(size, tuple) else (size,))
    return tuple(sizes)


def _cp0_cp1(spec):
    """``(cp0, cp1)`` for a mesh spec, with ``cp1 == 1`` standing for "token-j unsharded".

    Purpose
        The two numbers every derived extent in this module comes from: ``N_i_loc = N // cp0``,
        ``N_j_loc = N // cp1``, and the ``cp1 > 1`` test that selects the whole 2-D block.

    Args:
        spec: A value from the ``mesh`` pool.

    Returns:
        ``(cp0, cp1)``. A flat spec returns ``(cp, 1)`` -- 1 rather than ``cp`` on purpose, because
        that is exactly what the kernel resolves ``cp_axis_sizes=(cp,)`` to, and the sentinel the
        1-D path keys on.
    """
    ax = _cp_axis_sizes(spec)
    return ax[0], (ax[1] if len(ax) > 1 else 1)


def _per_axis_misaligns(N_token, mesh):
    """Whether a 2-D mesh lands either per-peer token extent off the 16-byte floor.

    ONE definition, used by BOTH the ``Unsupported`` region below and the accepting tests' cell
    lists, because the two must agree exactly: a region that refuses a cell an accepting test still
    emits is a contradiction the machinery can only catch one cell at a time.

    **It reads the real ``cp0``/``cp1``, and that is a widening rather than a tidy-up.** Its
    predecessor hardcoded ``// 2`` because ``_configure_drain`` pinned the 2-D shard to
    ``cp_axis_sizes=(2, 2)`` at ``cp=4``, so the only reachable divisor was 2. With the ``mesh``
    axis the divisors span ``cp0 in {2, 4}`` and ``cp1 in {2, 4, 8}``, and the region gains a cell
    no hardcoded half could express: ``N_token=2080`` at ``cp=(2, 8)`` has ``N_i_loc = 1040``
    (aligned) and ``N_j_loc = 260`` (``260 % 8 == 4``), so a value that is fully supported at every
    previously reachable mesh must be REFUSED at that one. Measured -- the five refusing
    ``(mesh, N_token)`` pairs in the whole pool are ``(2,2)/2072``, ``(2,4)/2072``, ``(2,8)/2072``,
    ``(2,8)/2080`` and ``(4,4)/2072``.

    Args:
        N_token: A value from the ``N_token`` pool. Every pool value is ``% 8``; what varies is
            whether its per-peer QUOTIENTS are, which is the property here.
        mesh: A value from the ``mesh`` pool. A flat spec never misaligns -- ``cp1 == 1`` skips the
            kernel's whole per-axis block -- so the answer is a function of the PAIR and no
            predicate on ``N_token`` alone expresses it.

    Returns:
        True when ``configure_a2a_gemm_native`` must refuse the combination at its 16-B per-axis
        guard (``gemm_sm90_a2a.py:932``); False when it must configure.
    """
    cp0, cp1 = _cp0_cp1(mesh)
    return cp1 > 1 and ((N_token // cp0) % 8 != 0 or (N_token // cp1) % 8 != 0)


def _shard_is_uneven(N_token, mesh):
    """Whether a mesh divides ``N_token`` UNEVENLY, so ``N // cp`` is a floor rather than a shard.

    **This is not a kernel refusal, and the asymmetry is the reason it needs its own helper.** The
    2-D path REFUSES an uneven shard at the front door -- ``N_loc == N//cp0`` and ``N % cp1 == 0``
    are both checked (``gemm_sm90_a2a.py:885``, ``:891``). The 1-D path checks NEITHER: it accepts
    whatever ``N_loc`` the caller hands it and derives every extent from that, so a floor-divided
    ``N_loc`` configures clean while silently owning fewer tokens than exist. Measured: at
    ``cp=16``, ``N_token=2072`` gives ``N_loc = 129`` and ``129 * 16 = 2064``, so eight tokens
    belong to no peer and nothing raises.

    That state is UNTESTED here rather than asserted, and the exclusion is what says so. The
    geometry test's subject is what the kernel DERIVES from an even caller-supplied shard; an uneven
    shard is a different contract (production would hand different ranks different ``N_loc``), and
    the partition identity it asserts is not the property to check there. Reachable only because the
    ``mesh`` axis unpinned ``cp`` -- at the previous pin of 4, every declared ``N_token`` divided
    exactly, so no cell could reach it.

    Args:
        N_token: A value from the ``N_token`` pool.
        mesh: A value from the ``mesh`` pool.

    Returns:
        True when either cp axis fails to divide ``N_token`` exactly.
    """
    cp0, cp1 = _cp0_cp1(mesh)
    return N_token % cp0 != 0 or (cp1 > 1 and N_token % cp1 != 0)


A2A_KERNEL = KernelMatrix(
    kernel="gemm_sm90_a2a",
    axes=(
        Axis(
            name="tile",
            domain=(
                "any (tile_m, tile_n) the SM90 WGMMA atom can build. The pool spans BOTH tile_m "
                "values because tile_m=256 is what produces MULTIPLE epilogue M-subtiles, and the "
                "drain's validity turns on exactly that -- a pool of tile_m=128 alone could not "
                "reach the multi-subtile drain at all, and would report the deferred-variant region "
                "as unreachable rather than as refused"
            ),
            values=((128, 128), (128, 256), (256, 128), (256, 256)),
            facets={
                "single_epi_subtile": lambda t: t[0] == 128,
                "multi_epi_subtile": lambda t: t[0] == 256,
                "narrow_n": lambda t: t[1] == 128,
                "wide_n": lambda t: t[1] == 256,
            },
        ),
        Axis(
            name="drain_variant",
            domain=(
                "the cluster-drain kind: 'cluster_drain' is the even-shard single-buffer drain, "
                "'cluster_multislot' the surviving DEFERRED subtile-axis variant. Both are "
                "reachable configurations, and they differ in whether the multi-epi-subtile drain "
                "lands -- which is the whole content of the unsupported region below"
            ),
            values=("cluster_drain", "cluster_multislot"),
            facets={
                "single_buffer": lambda v: v == "cluster_drain",
                "deferred": lambda v: v == "cluster_multislot",
            },
        ),
        Axis(
            name="mesh",
            domain=(
                "any spec of group-name -> int | tuple[int, ...] whose rank product is the flat `cp` "
                "the store is configured at. The pool is PURE context-parallel and nothing else, "
                "because every mesh dim here carries the all-to-all: a `dp` group would add shard "
                "dims the back store does not route. Both FLAT and FACTORED cp are in the pool and "
                "the difference is not cosmetic -- a factored spec IS the 2-D token shard (cp1>1), "
                "whose per-peer j clamp, LayoutRight peer re-flatten "
                "(`_a2a_cp_unravel_shape_stride`) and per-axis 16-B floor a flat cp collapses to a "
                "single division, which is exactly the case that would pass while testing half the "
                "routing. At most TWO cp axes: `configure_a2a_gemm_native` refuses a third"
            ),
            # ---- WHY THIS AXIS EXISTS HERE AT ALL, AND WHAT ITS FACETS DO NOT CLAIM -------------
            # THE MESH IS A CONFIGURE ARGUMENT IN THIS MODULE, NOT A LAUNCH. `cp`, `cp_axis_sizes`
            # and `pe_table` are plain arguments to `configure_a2a_gemm_native`, which derives every
            # per-peer extent from them on the HOST with nothing allocated and no process group
            # required. So the whole pool -- including the three specs that need 16 ranks -- is
            # exercised by ONE run on any SM90 box. That is coverage the launching sibling
            # (`test_gemm_a2a_epi.py`, which declares this SAME 8-spec pool) can only buy with a
            # 16-rank cross-node allocation, and it is why the axis is declared here rather than
            # deferred to the ledger the way a launched mesh must be.
            #
            # THE COROLLARY IS WHAT THE FACET NAMES DO NOT CLAIM. `cross_node` names a rank count
            # that would REQUIRE crossing a fabric, not a fabric that was crossed. No IB peer exists
            # in this module -- `build_p2p_table` is monkeypatched to an all-NVLink verdict so the
            # drain verdict can be read without a live team -- and the property under test is the
            # BAKED GEOMETRY, not the routing that geometry later drives.
            #
            # AND FOR THE SAME REASON THIS AXIS DELIBERATELY DOES NOT WRITE THE COVERAGE LEDGER.
            # `fold_cp_ops/testing/coverage_ledger.py` is keyed `distributed_manager.mesh` off the
            # `apply_mesh` fixture and UNIONS across every distributed module, precisely so that
            # "which meshes has any launch ever built?" has one answer. Recording `cp=(4,4)` as
            # exercised from a host-side configure would put a mesh in that answer that no launch
            # ever built, and the hole would close on paper for every module at once. A host-side
            # sweep is a different KIND of coverage and is kept out of the ledger that tracks the
            # other kind. The TOPOLOGY half of the mesh -- which peers are P2P and which are IB --
            # is asserted separately over probe outcomes, in
            # `test_a_coupled_back_store_is_refused_on_a_cross_node_mesh`, because a p2p verdict is
            # a property of the FABRIC and is not derivable from a rank count.
            values=(
                (("cp", 2),),
                (("cp", 4),),
                (("cp", 8),),
                (("cp", (2, 2)),),
                (("cp", (2, 4)),),
                (("cp", 16),),
                # FACTORED *and* >8 ranks. Under LayoutRight with 8 GPUs/node the two below differ
                # in where a real fabric would cut, which is what the store's peer routing turns on:
                (("cp", (2, 8)),),  # axis_1 == exactly one node; axis_0 (stride 8) would cross IB
                (("cp", (4, 4)),),  # axis_1 == HALF a node -- a cp group SMALLER than its NVLink
                #                     domain, so peer math that assumed "contiguous axis == the
                #                     whole node" passes (2,8) and fails here. It is also the pool's
                #                     only spec with cp0 == 4 AND cp1 > 1, i.e. the only one whose
                #                     two derived extents are equal without being halves of N.
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
                # Named for the SIBLING pools' facets rather than for what this module does with
                # them, so the three matrices stay relatable: same names, same predicates, same
                # values. See the block above for what `cross_node` does and does not assert here.
                "single_node": lambda s: _mesh_ranks(s) <= 8,
                "cross_node": lambda s: _mesh_ranks(s) > 8,
            },
        ),
        Axis(
            name="N_token",
            domain=(
                "the FULL square token extent the back einsum runs at -- the value the caller "
                "hands `configure_a2a_gemm_native` as `N`, and from which every per-peer extent is "
                "DERIVED (`N_loc = N // cp0`, `N_j_loc = N // cp1`). Any value with `N % 8 == 0`: "
                "that is the 16-byte TMA/put floor on the stride-1 token axis and the ONLY shape "
                "constraint this store carries. The 2-D shard applies the SAME floor to the derived "
                "halves, which is a property of N AND the mesh rather than of N alone, so no "
                "predicate on N by itself expresses it"
            ),
            # 12288 belongs in this correctness pool because the large-shape memory carve-out does
            # not apply. The carve-out permits dropping 12288 because at D=512
            # it is 9.7 GB in and 9.7 GB out before any fp32 reference. This module allocates
            # NOTHING at any N -- every test here is a host-side configure verdict -- so the largest
            # declared token extent costs precisely what the smallest does, and dropping it would be
            # narrowing for a reason that does not reach this subject.
            #
            # 2080 is the OFF-GRID value, and it is load-bearing rather than decorative. Every
            # ladder value divides BOTH declared tile_n values exactly (2048/128=16, 2048/256=8 ...
            # 12288/256=48), so a floor-division where the kernel means ceil -- the run_j sub-band
            # silent-zero class the kernel's own comment names -- would be INVISIBLE on the ladder:
            # `nt_pp` and `cluster_run_j` come out identical either way and the covering assertions
            # below cannot fail. 2080 is off-grid on every declared tile and every declared cp, and
            # it configures clean at every mesh that does not refuse it, so it carries with no
            # narrowing anywhere.
            #
            # WHICH VALUES ACTUALLY REACH THE COVERING ASSERTIONS IS A PROPERTY OF (N_token, mesh),
            # and the mesh axis changed the answer. Measured over the surviving grid, counting cells
            # where `N_i_loc % tile_m != 0` (i.e. ceil and floor disagree):
            #     N=2048   2 of 32 cells   N=2072  12 of 12   N=2080  28 of 28
            #     N=4096   0 of 32         N=8192   0 of 32   N=12288  0 of 32
            # Two things follow that the previous cp=4 pin hid. First, 2072 is not only the refusing
            # value -- at every FLAT mesh it configures clean AND straddles (N_i_loc = 1036/518/259,
            # none a multiple of 128), so it exercises the covering assertions rather than merely
            # the refusal. Second, and this is what the mesh axis bought: at cp=16 the LADDER value
            # 2048 straddles too, N_i_loc = 128 against tile_m = 256, where a floor would return
            # nt_i = 0 and grid NOTHING. At the old pin of cp0 in {2, 4} that same value derived
            # 512/1024 and was exact on both tiles, so no ladder cell could reach the assertion at
            # all. A pool wide in N and pinned in cp was one point wide in the axis that decides
            # whether the division is exact.
            #
            # 2072 is the value whose 2-D per-peer extents misalign. Note it is 2072/2 == 1036
            # that is `% 8 == 4`, not 2072, which is itself `% 8` like every value here -- so the
            # refusal is a property of the (N_token, mesh) PAIR and no predicate on N alone
            # expresses it. At every FLAT mesh it configures CLEAN (cp1 == 1 skips the per-axis
            # block entirely; N_loc = 1036/518/259 straddles, which arbitrary_n lifts), so the value
            # is not simply a bad shape: it is a shape the store supports on one mesh and must
            # REFUSE on another. Without it the 16-B per-axis guard at `gemm_sm90_a2a.py:932` is
            # unreachable from this pool and the region could only be declared, never exercised.
            #
            # It is no longer the ONLY such value, and that is the mesh axis at work: the five
            # refusing pairs are (2,2)/2072, (2,4)/2072, (2,8)/2072, (2,8)/2080 and (4,4)/2072. The
            # 2080 pair is the one a halving cannot produce -- N_i_loc = 1040 clears the floor and
            # N_j_loc = 260 does not -- so the region now exercises BOTH of its divisors instead of
            # one. See `_per_axis_misaligns`.
            values=(2048, 2072, 2080, 4096, 8192, 12288),
            facets={
                "workflow_ladder": lambda v: v in (2048, 4096, 8192, 12288),
                "off_grid": lambda v: v % 128 != 0,
                # A predicate on the DERIVED half, not on N -- see the 2072 note above. Named after
                # the sibling module's facet of the same shape (`test_gemm_a2a_epi.py`), which was
                # added for the same reason on N=536.
                "halves_to_misaligned": lambda v: (v // 2) % 8 != 0,
                # Named as a facet precisely BECAUSE a memory-bounded suite is entitled to drop this
                # value elsewhere: with the facet declared, dropping it HERE fails instead of
                # passing quietly, and the reason above has to be re-argued rather than forgotten.
                "largest_workflow": lambda v: v == 12288,
                "power_of_two": lambda v: v > 0 and v & (v - 1) == 0,
            },
        ),
        # ---- THERE IS DELIBERATELY NO `D` (feature width) AXIS HERE. --------------------------
        # Workflow coverage includes N_token AND D, so D's absence here needs supporting evidence.
        #
        # THE EVIDENCE. `D` is not an input to any decision this module's SUBJECT makes.
        #   * `configure_a2a_gemm_native(self, cp, my_cp_rank, B, N_loc, pe_table, *, ...)`
        #     (`gemm_sm90_a2a.py:417`) has no feature-width parameter. `B` is the TriMul batch, not
        #     D: the back einsum's L = B * (D/cp) is decomposed IN-KERNEL as `d = L//B`, `b = L%B`,
        #     so D is a property of the recv TENSOR and never of the config.
        #   * Tokenised the whole kernel module with comments and string literals stripped: `D`,
        #     `Dloc` and `D_loc` occur ZERO times as code tokens. Every occurrence is prose.
        #   * The one host-side site that sees the feature width is
        #     `_build_peer_store_atoms_gemm_native` (`:3581`), which takes the live 5-D symmetric
        #     recv `(cp, Dloc, B, N_loc, N)` and peer-translates it -- it needs a real NVSHMEM
        #     allocation. At the largest declared corner (D=512, N_token=12288, cp=4) that recv is
        #     ~38 GB PER RANK, and this module allocates nothing at all.
        #
        # WHY THAT FORBIDS THE AXIS RATHER THAN MERELY EXCUSING IT. A declared axis has exactly two
        # possible shapes here and both are worse than the omission: declared-and-unswept is what
        # `coverage_problems` rule 1 calls decoration ("an axis nothing sweeps ... changes nothing
        # that runs"); and swept by a test asserting "the configure verdict is independent of D" is
        # VACUOUS BY CONSTRUCTION, since configure never receives D, so that test cannot fail. This
        # module's own header already refuses a ported test for exactly that defect -- a green that
        # is indistinguishable from every other green.
        #
        # AND THE MACHINERY NOW ENFORCES IT, WHICH STRENGTHENS THE CONCLUSION RATHER THAN
        # WEAKENING IT. This paragraph used to say the opposite -- that `coverage_problems` ran only
        # behind `_is_kernel_module` (a `kernels/` path PART) so a decorative axis here would pass
        # silently. That was true when it was written and is FALSE now: `_owned_matrices`
        # (`fold_cp_ops/testing/kernel_matrix.py`) re-scoped coverage onto matrix OWNERSHIP at any
        # depth on 2026-08-19, and `matrix_scope` puts `tests/distributed/` in the audit
        # (`tests/conftest.py`). So a declared-and-unswept `D` no longer passes quietly -- it FAILS
        # every item in this module at setup. Both shapes an axis could take here are therefore
        # unavailable, not merely unattractive: decorative is red, and vacuous cannot fail.
        #
        # Recorded rather than deleted, because "the check that would have caught this did not run"
        # and "it runs and this is still the right call" are different states, and a reader who
        # remembers the first sentence needs to be told the second.
        #
        # WHERE THE D OBLIGATION IS REACHABLE: `tests/distributed/test_gemm_a2a_epi.py`, the module
        # for the file `GemmA2ASm90` is defined in. Its tests LAUNCH, and its matrix declares
        # `D = (4, 8, 96, 128, 256, 384, 512)` -- a SUPERSET of {128, 256, 384, 512} --
        # swept alongside the same 8-spec `mesh` pool this file declares. The back A2A spans two
        # source files (`gemm_sm90_a2a.py` holds `GemmSm90A2A`; `gemm_a2a_epi.py` composes it into
        # `GemmA2ASm90`) and the one-test-file-per-source rule split its coverage with it, so the D
        # obligation is discharged there. Doing it here would move a number through a signature that
        # has no argument to receive it.
        shape_mode_axis(),
    ),
    computes=computes_nothing_numeric(
        because=(
            "every test here is a HOST-side configuration decision -- a topology guard, a drain "
            "validity check, a delegation, and a source-level invariant. No kernel is launched and "
            "no tensor is produced, so there is no output whose distribution could hide anything. "
            "The kernel's numerics are the fused-store path's concern and are asserted where that "
            "path is exercised"
        )
    ),
    unsupported=(
        Unsupported(
            where=lambda tile, drain_variant: (
                tile[0] == 256 and drain_variant == "cluster_multislot"
            ),
            raises=ValueError,
            # Both alternatives are UNIQUE to the front door's raise (measured: one hit each across
            # the whole package). A third alternative, "deferred", was dropped -- it occurs in 5
            # files, so a raise from anywhere in the chain whose message merely contained that word
            # would satisfy the region. An alternation is only as strong as its weakest branch, and
            # `raises=` alone cannot carry the guard: `cute.compile` itself raises `TypeError`, so a
            # region naming a bare type is satisfiable with no front-door check having run at all.
            # Dropping it was necessary, not merely tighter: the measured message says "unsupported
            # drain variant" and contains no "deferred" at all, so that branch was both the weakest
            # AND vacuous -- it could only ever have matched some OTHER raise.
            match=r"single-subtile drain|multi-epi-subtile",
            reason=(
                "tile_m=256 produces MULTIPLE epilogue M-subtiles, and the multi-epi-subtile drain "
                "lands only on the even-shard SINGLE-BUFFER cluster_drain. cluster_multislot is the "
                "surviving DEFERRED variant and is excluded from that path, so the combination must "
                "config-REJECT rather than configure and mis-drain at runtime"
            ),
        ),
        Unsupported(
            where=lambda mesh, shape_mode: _cp0_cp1(mesh)[1] > 1 and shape_mode == "static",
            raises=NotImplementedError,
            # TWO front doors refuse this, and the region is anchored on BOTH so neither can be
            # satisfied by the other's absence: the ib_ring 2-D gate and the cluster_drain 2-D
            # twin. The bare phrase "requires dynamic=True" was NOT usable -- it also occurs twice
            # as `dyn_cp=True requires dynamic=True`, a ValueError from an unrelated guard, so a
            # region matching it would be satisfiable without either 2-D check running. Narrowed to
            # the "2-D (cp1>1)" wording, which occurs only in these two raises.
            match=(
                r"2-D \(cp1>1\) ib_ring requires dynamic=True"
                r"|cluster_drain 2-D \(cp1>1\) requires dynamic=True"
            ),
            reason=(
                "a 2-D token shard needs the per-peer j column clamp, and that clamp reads N_j_loc "
                "off the recv's RUNTIME shape. With a STATIC extent there is no runtime shape to "
                "read, so the clamp would size itself from the full N and either garbage a "
                "straddling N_j_loc or run off the end -- the kernel refuses instead. Recorded as a "
                "REGION rather than a waiver because the refusal is real code at the front door, "
                "and because DYNAMIC-N is the path production actually runs (one compile serves "
                "many N_token); static-N 2-D is a documented follow-up, not a silent gap"
            ),
        ),
        Unsupported(
            where=_per_axis_misaligns,
            raises=ValueError,
            # Anchored on the "16-B-aligned per-peer extents" wording, which occurs ONCE in the
            # package. The obvious shorter alternatives were measured and rejected: "16-B" alone
            # appears in several unrelated alignment raises across the store, and "must both be
            # multiples of 8" is a sentence fragment a future reword would silently drop. The
            # pattern names the CONSTRAINT, which is what keeps a region from being satisfied by an
            # unrelated failure.
            match=r"2-D ib_ring drain requires 16-B-aligned per-peer extents",
            reason=(
                "the 2-D drain routes each row and each column band to a peer through per-row "
                "NVSHMEM puts over the recv, so BOTH derived per-peer extents -- N_i_loc = N//cp0 "
                "and N_j_loc = N//cp1 -- must clear the 16-byte floor, not just N itself. This is "
                "the FIRST PRINCIPLE's one permitted shape constraint applied to a DERIVED extent: "
                "N=2072 is 16-B aligned and fully supported at every FLAT mesh, and it is only the "
                "cp-division that puts 1036 (or 518, or 259) off the boundary. With the mesh axis "
                "the region reaches a SECOND value no halving could express -- N=2080 at cp=(2,8), "
                "where N_i_loc=1040 is aligned and N_j_loc=260 is not -- so the two divisors are "
                "independently load-bearing. A straddling per-row put would garbage a partial run "
                "rather than fault, which is why this must be a front-door refusal and not a "
                "runtime discovery -- the kernel's own comment records that only a scalar rel_L2 "
                "catches that corruption and the per-row outlier gate misses it"
            ),
        ),
    ),
)


def _requires_sm90():
    """Rank-invariant skip off SM90 -- a property of the box, identical for every rank of a launch.

    Not a bare ``pytest.skip``: under ``tests/distributed/`` a divergent skip is a deadlock rather
    than a skip, and the declaration is what makes "we know this is job-uniform" a checked state
    instead of a belief. The predicate here genuinely is uniform -- this cluster's nodes are
    homogeneous and a launch does not span architectures.
    """
    if not torch.cuda.is_available() or get_device_capacity()[0] != 9:
        rank_invariant_skip(
            "needs an SM90 (H100) device",
            because=(
                "compute capability is a property of the machine the launch landed on; the nodes "
                "of one job are homogeneous, so every rank reaches the same verdict and no rank "
                "can skip alone"
            ),
        )


def _gemm(tile=(128, 128)):
    """An unconfigured `GemmA2ASm90` at `tile` -- A2A is OFF until `configure_a2a_gemm_native`."""
    return GemmA2ASm90(
        Float32,
        torch2cute_dtype_map[torch.bfloat16],
        tile,
        (1, 1, 1),
        pingpong=False,
        is_persistent=True,
    )


def _configure_drain(
    monkeypatch, tile, drain_variant, *, mesh=(("cp", 4),), shape_mode="dynamic", n_token=2048
):
    """Configure the cluster drain at `tile` with `drain_variant`; raises if the combo is refused.

    **Every kwarg here is load-bearing, and the set was measured rather than read.** The cluster
    drain sits behind four earlier gates, each of which refuses before the subtile-count guard is
    ever consulted -- so a configure call missing any of them exercises a DIFFERENT guard while
    looking like it exercised this one:

    * ``cluster_drain`` rides only on the complete decoupled pe_aligned ib_ring stack (``decoupled``,
      ``producer_tma``, ``consumer_strided``, ``consumer_strided_putwarp``, ``arbitrary_n``,
      ``pe_aligned_tiling``, ``ib_quiet``) -- six flags, and the raise names whichever are missing;
    * ``ib_ring=True`` is separately required, because it SUPPRESSES the auto-reduce that would
      otherwise turn ``arbitrary_n`` off on a 128-aligned ``N_loc`` and collapse the stack;
    * ``N`` (the full token-j extent) must be supplied to size the bounded band;
    * ``N_loc`` must be a positive multiple of ``cta_tile_M`` unless ``arbitrary_n`` lifts it --
      it is lifted here (``arbitrary_n=True``), which is what lets ``n_token`` carry an OFF-GRID
      value whose ``N_loc`` straddles a CTA M-tile.

    ``build_p2p_table`` is neutralised to an all-NVLink verdict because it calls
    ``nvshmem_team_translate_pe``, which does not raise but hard-ABORTS the process (exit 255) when
    NVSHMEM is not initialised. The drain verdict does not read the topology, so without this the
    whole file would be gated on a live team for a property unrelated to one. The topology guard
    itself is asserted separately, in
    :func:`test_a_coupled_back_store_is_refused_on_a_cross_node_mesh`.

    Args:
        monkeypatch: pytest's fixture, used for the ``build_p2p_table`` patch. Required rather than
            optional: without it the process aborts instead of failing a test.
        tile: ``(tile_M, tile_N)`` from the matrix pool.
        drain_variant: ``"cluster_drain"`` or ``"cluster_multislot"`` from the matrix pool.
        mesh: A spec from the ``mesh`` pool. It supplies THREE arguments at once and they must
            move together, which is why one axis carries them: ``cp = _mesh_ranks(mesh)``,
            ``cp_axis_sizes = _cp_axis_sizes(mesh)`` (passed ONLY when factored, so a flat spec
            takes the kernel's own ``(cp,)`` default and stays on the byte-identical 1-D path), and
            ``pe_table = tuple(range(cp))`` -- ``configure_a2a_gemm_native`` raises when
            ``len(pe_table) != cp``, so a mesh swept without its table would fail on the table
            rather than on the mesh. A FLAT spec leaves ``cp1 == 1`` and the whole 2-D block is
            skipped; a FACTORED one sets ``cp0``/``cp1`` and turns on the per-axis 16-B guard, which
            is a joint constraint on ``n_token`` AND this argument (see ``n_token``). Defaults to
            ``(("cp", 4),)`` -- a DECLARED pool value, and the same ``cp=4`` every call site pinned
            before the axis existed, so a test that does not name a mesh is pinned to something the
            matrix knows about.
        n_token: The full square token extent, from the matrix pool -- passed as ``N``, with
            ``N_loc = n_token // cp0``, which is the same relation the caller has to satisfy for
            real: ``configure_a2a_gemm_native`` REJECTS a 2-D config whose ``N_loc != N // cp0``.
            The floor division is exact for every declared ``(n_token, mesh)`` pair EXCEPT
            ``cp=16`` with ``n_token=2072`` (2072/16 = 129.5); see :func:`_shard_is_uneven` for why
            that pair is excluded from the geometry test rather than asserted, and for the 1-D/2-D
            asymmetry that makes it configure clean. It defaults to ``2048`` -- a DECLARED pool
            value rather than a convenient literal, so the tests that pin it are pinned to something
            the matrix knows about and a pool edit cannot silently orphan the default.
        shape_mode: ``"dynamic"`` marks the token extents runtime (``dynamic=True``), ``"static"``
            bakes them. This is the pair the second unsupported region turns on: 2-D + static is
            refused, every other combination configures. DYNAMIC is the production path -- one
            compile serves many ``N_token`` -- and nothing here compiles anything, so covering both
            modes costs no device time and no cold-compile budget.

    Returns:
        The configured `GemmA2ASm90`.

    Raises:
        ValueError: From the kernel's own front door when the combination is refused -- which is the
            observable the unsupported-region test asserts.
    """
    import fold_cp_ops.distributed.gemm_sm90_a2a as G

    monkeypatch.setattr(G, "build_p2p_table", lambda pe_table: tuple([True] * len(tuple(pe_table))))
    cfg = dict(
        cluster_drain=True,
        cluster_n=1,
        decoupled=True,
        producer_tma=True,
        ib_ring=True,
        consumer_strided=True,
        consumer_strided_putwarp=True,
        arbitrary_n=True,
        pe_aligned_tiling=True,
        ib_quiet=True,
    )
    if drain_variant == "cluster_multislot":
        cfg["cluster_multislot"] = True
    cp = _mesh_ranks(mesh)
    cp0, cp1 = _cp0_cp1(mesh)
    # Passed ONLY when factored. A flat spec must leave the kwarg at None so the kernel resolves it
    # to its own `(cp,)`: handing it `(cp,)` explicitly would be the same numbers down a different
    # branch, and the 1-D path's whole claim is that it is the byte-identical one.
    if cp1 > 1:
        cfg["cp_axis_sizes"] = _cp_axis_sizes(mesh)
    cfg["dynamic"] = shape_mode == "dynamic"
    n_loc = n_token // cp0
    g = _gemm(tile)
    g.configure_a2a_gemm_native(
        cp=cp, my_cp_rank=0, B=1, N_loc=n_loc, N=n_token, pe_table=tuple(range(cp)), **cfg
    )
    return g


@A2A_KERNEL.parametrize(
    "tile",
    "drain_variant",
    cells=[
        ((128, 128), "cluster_drain"),
        ((128, 256), "cluster_drain"),
        ((256, 128), "cluster_drain"),
        ((256, 256), "cluster_drain"),
        ((128, 128), "cluster_multislot"),
        ((128, 256), "cluster_multislot"),
    ],
    because=(
        "the product's other two cells -- tile_m=256 x cluster_multislot -- ARE the declared "
        "unsupported region and must RAISE, and a correctness test cannot also claim they succeed; "
        "they are covered by the parametrize_unsupported() test directly below. Spelled as cells= "
        "and not drop=: dropping tile_m=256 would surrender the multi_epi_subtile facet on "
        "cluster_drain, and dropping cluster_multislot would surrender the deferred facet on "
        "tile_m=128 -- both are supported configurations that must stay covered here"
    ),
)
@numeric_exempt(
    "asserts a CONFIGURATION verdict -- whether a (tile, drain_variant) pair is accepted -- not a "
    "computed value. Nothing is launched"
)
def test_the_drain_accepts_exactly_the_tiles_its_subtile_count_supports(
    monkeypatch, tile, drain_variant
):
    """Every pair OUTSIDE the unsupported region configures clean.

    This body covers only the ACCEPTING side; the refusing side is the test below. Both halves
    matter, and neither implies the other: a guard that over-rejects would silently disable the
    shipped 128-row tiles, and one that under-rejects lets a deferred variant configure and
    mis-drain at runtime with nothing raised.

    Declaring the ``Unsupported`` region does **not** assert it. The region obliges a refusing test
    and makes this decorator refuse to build a grid containing a must-RAISE cell -- which is why
    the accepting cells are listed explicitly rather than swept as a product.

    **The second assertion is what makes the two axis values DISTINGUISHABLE, and without it this
    test could not tell them apart at all.** ``_a2a_cluster_drain`` is True for BOTH variants -- it
    records that the cluster drain is on, not WHICH one -- so a kernel that silently ignored
    ``cluster_multislot=True`` would pass every cell here while configuring the single-buffer drain
    twice. Measured across all 84 (mesh x N_token) cells at ``tile_m=128``: the two variants bake
    IDENTICAL geometry in every baked attribute -- ``cp``/``cp0``/``cp1``, ``N_i_loc``/``N_j_loc``,
    ``nt_pp``/``nt_i_pp``/``nt_j_pp``, ``cluster_run_j``, ``cluster_n``, the unravel -- and differ in
    ``_a2a_drain_tail`` ALONE, uniformly, at every one of the 84.

    That measurement is also why this test does not cross ``drain_variant`` with ``mesh``. The one
    observable that separates the variants is a mesh-INDEPENDENT boolean, so 84 more cells would
    re-assert the same flag 84 times and discriminate nothing further; the mesh is swept where it
    changes an answer, on the geometry test. A cross that varies a parameter the subject does not
    read is grid, not coverage.
    """
    _requires_sm90()
    g = _configure_drain(monkeypatch, tile, drain_variant)
    assert g._a2a_cluster_drain, (
        f"tile {tile} with {drain_variant} configured without enabling the cluster drain; the "
        "config was accepted but the path it was meant to select is off"
    )
    assert bool(g._a2a_drain_tail) == (drain_variant == "cluster_multislot"), (
        f"drain_variant={drain_variant} did not reach the kernel: _a2a_drain_tail is "
        f"{g._a2a_drain_tail!r}. That flag arms the MMA-warp tail hook and is the ONLY baked "
        f"difference between the two variants, so a configure call that dropped the variant would "
        f"leave both cells of this axis selecting the same drain -- and the axis would report "
        f"coverage it does not have"
    )


@A2A_KERNEL.parametrize_unsupported("tile", "drain_variant", "mesh", "shape_mode", "N_token")
@numeric_exempt("asserts a REFUSAL at the kernel's own front door, not a computed value")
def test_every_declared_unsupported_configuration_is_refused_at_the_front_door(
    monkeypatch,
    tile,
    drain_variant,
    mesh,
    shape_mode,
    N_token,
    expected_error,
    expected_match,
):
    """All three declared regions, swept together, refused by ``configure_a2a_gemm_native``.

    ONE test over all five axes rather than one per region, and that is the machinery's rule rather
    than a style choice: ``parametrize_unsupported`` requires the swept names to be a SUPERSET of
    every region's axes, because a region it cannot evaluate would be silently dropped and the test
    would cover less than it appears to. Splitting it would have meant two sweeps each blind to the
    other's region.

    The three refusals are independent and no one of them implies another:

    * ``tile_m=256`` x ``cluster_multislot`` -- multiple epilogue M-subtiles land only on the
      even-shard single-buffer drain, so the deferred variant must config-REJECT rather than
      configure and mis-drain at runtime. Pattern ``single-subtile drain|multi-epi-subtile``: both
      alternatives occur exactly once across the package. A third was dropped for occurring in five
      files -- an alternation is only as strong as its weakest branch.
    * a FACTORED mesh (``cp1>1``) x ``static`` extent -- the per-peer j column clamp reads
      ``N_j_loc`` off the
      recv's RUNTIME shape, and a baked extent leaves nothing to read. Pattern anchored on the
      ``2-D (cp1>1)`` wording, NOT on ``requires dynamic=True``: measured, the bare phrase occurs
      four times in the kernel, twice as an unrelated ``dyn_cp=True`` ``ValueError``, so matching it
      would make the region satisfiable without either 2-D check having run.
    * a FACTORED mesh x a token extent whose per-peer quotients misalign -- the 2-D drain routes
      rows and column bands by per-row NVSHMEM puts, so BOTH derived extents (``N//cp0``,
      ``N//cp1``) must clear the 16-byte floor and not just ``N``. The pairing is the whole content:
      ``N_token=2072`` configures CLEAN at every flat mesh and must REFUSE at four factored ones,
      and ``N_token=2080`` -- clean everywhere a halving could reach -- must REFUSE at ``cp=(2,8)``
      alone, where ``N_j_loc = 260``. So this is the FIRST PRINCIPLE's one permitted constraint
      applied to a DERIVED extent rather than a new constraint on the input, and the mesh is what
      makes BOTH divisors load-bearing instead of one.

    A cell may land in SEVERAL regions, and the expectation then admits EVERY one of their messages
    -- a region's claim is "this combination is refused", not "refused for my reason and no other".
    That is load-bearing here rather than theoretical: ``(2d, static, 2072)`` violates the second
    region AND the third, and the kernel's own guard ORDER decides which fires (the dynamic check at
    ``gemm_sm90_a2a.py:911`` precedes the 16-B check at ``:932``, so it raises
    ``NotImplementedError``). Pinning WHICH refusal appears would be pinning an internal check order
    the matrix has no business knowing.

    ``front_door_raises`` rather than ``pytest.raises``: the two agree on the type and on the
    message, and only the former also rejects a raise that came from inside ``cute.compile``. That
    distinction is the whole enforcement here -- the expected types are a plain ``ValueError`` and a
    plain ``NotImplementedError``, and a region asserted with bare ``pytest.raises`` can pass
    against a kernel carrying no front-door check at all.

    Nothing is launched and nothing is compiled: the refusal happens while configuring, so this
    sweep costs no device time and no cold-compile budget, and runs wherever the module imports.
    """
    _requires_sm90()
    with front_door_raises(expected_error, expected_match):
        # EVERY swept axis is threaded into the call, `n_token` included. A parameter that reaches
        # the test signature but not the subject makes the sweep look wider than it is: measured
        # here, leaving `n_token` at its default while sweeping it produced 6 DID-NOT-RAISE
        # failures, because the third region's cells were being configured at 2048 instead of 2072.
        _configure_drain(
            monkeypatch,
            tile,
            drain_variant,
            mesh=mesh,
            shape_mode=shape_mode,
            n_token=N_token,
        )


@matrix_exempt(
    "the subject is a TOPOLOGY guard parametrized over p2p-probe RESULTS, which no declared axis "
    "carries -- and specifically not the `mesh` axis, despite this test's name. A mesh spec is a "
    "rank COUNT and a factoring; which of those ranks are reached over NVLink and which over IB is "
    "a property of the FABRIC, not derivable from the spec, and is exactly what this test varies. "
    "Drawing tile or drain values from the matrix would vary something the guard does not read"
)
@numeric_exempt("asserts a refusal at the public API, not a computed value")
@pytest.mark.parametrize(
    "is_p2p,decoupled,expect_raise",
    [
        ((True, True, True, True), False, False),  # coupled, all-NVLink -> supported
        ((True, True, False, False), False, True),  # coupled, CROSS-NODE -> the defect: must RAISE
        ((True, False, True, False), False, True),  # interleaved IB peers -> same
        ((False, False, False, False), False, False),  # probe could not run -> no positive verdict
        ((True, True, False, False), True, False),  # decoupled ring cross-node -> supported path
        ((True, True, True, True), True, False),  # decoupled on all-NVLink -> collapses, supported
    ],
    ids=[
        "coupled-nvlink",
        "coupled-crossnode",
        "coupled-interleaved",
        "coupled-unprobed",
        "decoupled-crossnode",
        "decoupled-nvlink",
    ],
)
def test_a_coupled_back_store_is_refused_on_a_cross_node_mesh(
    monkeypatch, is_p2p, decoupled, expect_raise
):
    """A COUPLED back store must be rejected at the public API when any peer is reached over IB.

    The coupled store writes into a peer's symmetric heap through ``nvshmem_ptr``, which returns
    NULL for an IB peer -- so cross-node it produced an illegal memory access that surfaced later at
    a barrier, with no attribution. It is NVLink-only by hardware, the defect was the missing guard,
    and the supported cross-node path already exists.

    **The all-False row is the one that earns its place.** This rank's own PE is always in
    ``TEAM_SHARED``, so an all-False probe table means the probe could not run at all -- and the
    guard must NOT reject an all-NVLink job on that basis. A pool without it would let a guard that
    rejects on "no positive evidence" pass.
    """
    _requires_sm90()
    import fold_cp_ops.distributed.gemm_sm90_a2a as G

    monkeypatch.setattr(G, "build_p2p_table", lambda pe_table: is_p2p[: len(tuple(pe_table))])
    kw = dict(cp=4, my_cp_rank=0, B=1, N_loc=128, pe_table=(0, 1, 2, 3))
    if decoupled:
        kw.update(
            decoupled=True,
            producer_tma=True,
            consumer_strided=True,
            consumer_strided_putwarp=True,
            arbitrary_n=True,
            pe_aligned_tiling=True,
            N=512,
        )

    if expect_raise:
        with pytest.raises(ValueError, match="COUPLED"):
            _gemm().configure_a2a_gemm_native(**kw)
    else:
        g = _gemm()
        g.configure_a2a_gemm_native(**kw)
        assert getattr(g, "_a2a_is_p2p", None) is not None, (
            "the P2P probe must run on EVERY store kind -- it was decoupled-only, which is why the "
            "coupled store never saw its own topology"
        )


@matrix_exempt(
    "asserts the UNCONFIGURED state of a fresh instance, which does not vary with tile or drain "
    "variant -- parametrizing would construct four instances and assert the same thing"
)
@numeric_exempt("asserts a flag on an unconfigured instance, not a computed value")
def test_with_a2a_unconfigured_the_store_seam_delegates_to_the_parent():
    """A fresh instance has A2A OFF, so ``build_D_copy_fn``'s first branch returns the parent store.

    This is "distributed OFF == parent" at the one seam that could break it: the override's first
    statement is ``if const_expr(not self._a2a_enabled): return super().build_D_copy_fn(...)``, so
    an instance that came up with the flag set would silently take the peer path on a local run.
    """
    _requires_sm90()
    g = _gemm()
    assert getattr(g, "_a2a_enabled", False) is False, (
        "an unconfigured GemmA2ASm90 reports A2A ENABLED; the store seam would then take the peer "
        "path on a purely local run, where there is no peer"
    )


@matrix_exempt(
    "a SOURCE-level invariant over the store path's text -- it has no shape, dtype or tile to draw "
    "from, and holds for every configuration simultaneously"
)
@numeric_exempt("inspects source structure, not a computed value")
def test_a_peer_atom_and_its_signal_are_indexed_together():
    """Wherever a peer atom and a signal PE appear together, they are selected by the SAME index.

    **This is obligation B's SOURCE-LEVEL half, and it is deliberately narrow.** The recorded
    anti-pattern is an API that takes a peer-bound atom *and* a runtime ``pe`` for ``signal_op``: a
    TMA descriptor freezes its destination at build time and cannot be retargeted, so a mismatched
    ``pe`` lands the data on one PE and signals another. Nothing raises; the receiver waits on a
    buffer that was never written while some other rank's buffer holds data nobody is waiting for.

    **What this covers and what it does not.** It catches a subscript mismatch in the source --
    ``atoms[r]`` paired with ``pe_list[s]``. It cannot catch a mismatch produced at runtime, by a
    remapped table or an index computed rather than subscripted. That half is open, and it needs a
    2-rank test that writes through a peer atom and verifies the data landed where the signal went.
    Narrow and honest: a wrong index fails this, so it is not vacuous -- it is partial.
    """
    import fold_cp_ops.distributed.gemm_sm90_a2a as G

    src = inspect.getsource(G)
    tree = ast.parse(src)
    offenders = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        name = fn.attr if isinstance(fn, ast.Attribute) else getattr(fn, "id", "")
        if "signal_op" not in name:
            continue
        # The signal's PE argument and any atom subscript in the same call must share a subscript.
        subs = {
            ast.unparse(a.slice)
            for a in ast.walk(node)
            if isinstance(a, ast.Subscript) and "atom" in ast.unparse(a.value).lower()
        }
        pes = {
            ast.unparse(a.slice)
            for a in ast.walk(node)
            if isinstance(a, ast.Subscript) and "pe" in ast.unparse(a.value).lower()
        }
        if subs and pes and subs != pes:
            offenders.append(f"line {node.lineno}: atom index {subs} vs signal PE index {pes}")
    assert not offenders, (
        "a peer atom and its signal's PE are selected by DIFFERENT indices:\n  "
        + "\n  ".join(offenders)
        + "\nA TMA descriptor bakes its destination at build time, so this stores to one PE and "
        "signals another -- the receiver waits forever on a buffer nobody wrote."
    )


@A2A_KERNEL.parametrize(
    "mesh",
    "shape_mode",
    cells=[
        (m, s)
        for m in A2A_KERNEL.axis("mesh").values
        for s in A2A_KERNEL.axis("shape_mode").values
        if not (_cp0_cp1(m)[1] > 1 and s == "static")
    ],
    because=(
        "the full product MINUS the four cells (any FACTORED mesh, 'static'), which ARE the second "
        "declared unsupported region and must RAISE -- a test claiming they configure would be "
        "asserting the opposite of the matrix. They are covered by the parametrize_unsupported() "
        "test above. The exclusion is computed from the SAME quantity the region reads "
        "(`_cp0_cp1(mesh)[1] > 1`) rather than restated as a literal list, so an added mesh spec "
        "cannot land in an accepting grid the region also refuses. `tile` and `drain_variant` are "
        "pinned to one value each because the axis under test is the (mesh, shape_mode) PAIR and "
        "neither refusal reads the tile: crossing all four tiles would quadruple the cells and "
        "discriminate nothing. Both tile facets are exercised by the drain tests above, on the "
        "same matrix."
    ),
)
@numeric_exempt("a HOST-side configure decision; nothing is launched and no tensor is produced")
def test_the_mesh_and_shape_mode_pair_configures_where_it_is_supported(
    monkeypatch, mesh, shape_mode
):
    """Every (mesh, shape_mode) pair OUTSIDE the region configures clean, in BOTH modes.

    The accepting side is the half that keeps the region honest in the expensive direction. A guard
    that refused a FLAT mesh at static as well would disable the path ``FusedTriMul``'s own default
    reaches -- ``dynamic`` defaults to False there -- and the refusing test alone cannot see that,
    because an over-broad guard satisfies it just as well as a correct one. The same argument now
    covers a second way to be over-broad that the pinned ``cp=4`` could not express: a guard keyed
    on "more than four peers" rather than on ``cp1 > 1`` would refuse flat ``cp=8`` and ``cp=16``
    at static, and every cell of the old two-value axis would still have passed.

    Static is covered HERE, where it costs nothing, precisely because covering it where it is
    expensive is the thing to avoid: a static token extent is baked into the compile key, so a
    static correctness sweep pays a cold compile per shape. The configure path bakes nothing, so
    both modes are asserted for the price of neither -- and for the price of neither at all eight
    declared meshes, including the three that would need 16 ranks to launch.
    """
    _requires_sm90()
    g = _configure_drain(monkeypatch, (128, 128), "cluster_drain", mesh=mesh, shape_mode=shape_mode)
    assert g._a2a_cluster_drain, (
        f"mesh={mesh} shape_mode={shape_mode} configured without enabling the cluster "
        "drain; the config was accepted but the path it selects is off"
    )
    assert bool(g._a2a_dynamic) == (shape_mode == "dynamic"), (
        f"shape_mode={shape_mode} did not reach the kernel: _a2a_dynamic is {g._a2a_dynamic!r}. "
        "A mode the configure call silently drops would make both cells of this axis compile the "
        "same code, and the axis would report coverage it does not have"
    )
    cp, (cp0, cp1) = _mesh_ranks(mesh), _cp0_cp1(mesh)
    assert g._a2a_cp == cp and (g._a2a_cp0, g._a2a_cp1) == (cp0, cp1), (
        f"mesh={mesh} did not reach the kernel: got cp={g._a2a_cp}, (cp0, cp1)="
        f"({g._a2a_cp0}, {g._a2a_cp1}), expected cp={cp}, (cp0, cp1)=({cp0}, {cp1}). A mesh the "
        "configure call silently flattens would make several cells of this axis configure the same "
        "store, and the axis would report coverage it does not have -- the same failure the "
        "shape_mode assertion above exists to catch, on the axis that carries three arguments"
    )


@A2A_KERNEL.parametrize(
    "N_token",
    "mesh",
    "tile",
    cells=[
        (n, m, t)
        for n in A2A_KERNEL.axis("N_token").values
        for m in A2A_KERNEL.axis("mesh").values
        for t in A2A_KERNEL.axis("tile").values
        if not _per_axis_misaligns(n, m) and not _shard_is_uneven(n, m)
    ],
    because=(
        "the full product MINUS two disjoint exclusions, and they are excluded for OPPOSITE "
        "reasons.\n"
        "\n"
        "(1) The 20 cells where `_per_axis_misaligns` holds -- five (mesh, N_token) pairs across "
        "four tiles -- ARE the third declared unsupported region and must RAISE, so a test "
        "asserting they bake a valid geometry would be asserting the opposite of the matrix. They "
        "are covered by the parametrize_unsupported() test above. The exclusion is computed with "
        "the SAME predicate the region declares rather than restated, so the two cannot drift "
        "apart; a hand-written literal here is exactly how an accepting grid and a refusing region "
        "come to disagree about one cell.\n"
        "\n"
        "(2) The 4 cells where `_shard_is_uneven` holds -- cp=16 at N_token=2072, all four tiles -- "
        "are the OPPOSITE case: the kernel ACCEPTS them and no region declares them. 2072/16 is "
        "129.5, so the floor-divided N_loc=129 owns 2064 of 2072 tokens and this test's PARTITION "
        "assertion (`N_i_loc * cp0 == N_token`) is not the property to check there. The subject "
        "here is what the kernel DERIVES from an EVEN caller-supplied shard; an uneven shard is a "
        "different caller contract, and the 1-D path -- unlike the 2-D one, which refuses both "
        "`N_loc != N//cp0` and `N % cp1 != 0` at its front door -- checks neither. That asymmetry "
        "is recorded in `_shard_is_uneven` and left UNTESTED rather than silently satisfied; it is "
        "reachable at all only because the mesh axis unpinned cp"
    ),
)
@numeric_exempt(
    "asserts the BAKED per-peer geometry a configure call derives from the token extent -- host-side "
    "integers read off the instance, not a computed tensor. Nothing is launched and nothing is "
    "allocated at any pool value"
)
def test_the_token_extent_reaches_the_baked_per_peer_geometry(monkeypatch, N_token, mesh, tile):
    """One token extent in; every per-peer extent, tile count and drain band out -- and they COVER.

    ``configure_a2a_gemm_native`` takes the square token extent as a single ``N`` and derives the
    whole store geometry from it on the host: the per-peer i/j blocks (``N // cp0``, ``N // cp1``),
    the per-peer tile counts the pe_aligned scheduler grids on, and ``_a2a_cluster_run_j`` -- the
    BOUNDED drain band, in cluster-tile units. Every one of those is baked at config, so this whole
    ladder costs no device memory, no launch and no cold compile, which is why the largest declared
    ``N_token`` is in this pool rather than deferred to a perf cell.

    **Three claims, and none of them is a restatement of the kernel's formula.**

    * **Partition.** ``N_i_loc * cp0 == N_token`` (and, on a 2-D shard, ``N_j_loc * cp1 ==
      N_token``). A shard that lost or double-counted a token band satisfies no arithmetic identity
      with the extent it came from.
    * **The 1-D SENTINEL.** At ``cp1 == 1`` the kernel sets ``_a2a_N_j_loc = 0`` -- deliberately not
      ``N`` -- because the copy_fn reads that zero as "j unsharded, use ``tile_n``". A configure
      that helpfully filled in the full ``N`` there would route every 1-D column through the 2-D
      per-peer clamp, so the sentinel is load-bearing and is asserted as a value, not as a formula.
    * **Covering, and MINIMALLY covering.** The per-peer tile count must span its block and no more
      (``(nt-1)*tile < extent <= nt*tile``), and the bounded band must span one peer_j width the
      same way. That is the *property* a ceil expresses, asserted instead of the ceil: a
      floor-division leaves the last partial tile or the last partial cluster undrained -- the
      ``run_j`` sub-band silent-zero class the kernel's own comment names -- and fails the LEFT
      inequality, while an over-generous band fails the right one.

    **Whether a ceil and a floor can be told apart is a property of the (N_token, mesh) PAIR, and
    that is what the mesh axis bought here.** Every ladder value divides both declared ``tile_n``
    values exactly, so on the ladder alone a ceil and a floor agree everywhere and all three
    covering assertions hold vacuously. What decides whether the DERIVED extent is inexact is the
    divisor, i.e. the mesh. Measured over this grid, counting cells where ``N_i_loc % tile_m != 0``:

    ====== ============ ==========================================================================
    N       inexact      note
    ====== ============ ==========================================================================
    2048    2 of 32      ONLY at cp=16 x tile_m=256: N_i_loc=128, where a floor gives nt_i = 0
    2072   12 of 12      every surviving cell -- N_i_loc = 1036 / 518 / 259, none a tile multiple
    2080   28 of 28      every surviving cell -- the off-grid value, inexact at every divisor
    4096    0 of 32      exact everywhere
    8192    0 of 32      exact everywhere
    12288   0 of 32      exact everywhere
    ====== ============ ==========================================================================

    The 2048 row is the one worth reading twice. At the previous pinned ``cp=4`` that value derived
    ``N_loc`` 512 or 1024 and was exact on both tiles, so no WORKFLOW-LADDER cell could reach the
    covering assertion at all -- only the two off-grid values could, and a suite that lost them
    would have kept a green that asserted nothing on the shapes production runs. A pool wide in
    ``N_token`` and pinned in ``cp`` was one point wide in the axis that decides the question.

    **``drain_variant`` and ``shape_mode`` are PINNED, and the pins are the supported corner rather
    than a convenience.** ``cluster_drain`` is the even-shard single-buffer drain, the variant that
    accepts every declared tile; ``dynamic`` is the production path AND the one a factored mesh
    requires (factored + static is a declared unsupported region, so naming ``shape_mode`` here
    would build a grid containing a must-RAISE cell and the decorator would refuse it). Both axes
    are swept where they are the subject: the drain tests and the mesh/shape_mode pair test above,
    on this same matrix.
    """
    _requires_sm90()
    g = _configure_drain(monkeypatch, tile, "cluster_drain", mesh=mesh, n_token=N_token)
    cluster_n = 1  # _configure_drain passes cluster_n=1; the band is in cluster-tile units of it
    cp0, cp1 = _cp0_cp1(mesh)
    tile_m, tile_n = tile

    assert g._a2a_N == N_token, (
        f"N_token={N_token} did not reach the kernel: _a2a_N is {g._a2a_N!r}. Every per-peer "
        "extent below is derived from this one number, so a dropped N silently re-sizes all of them"
    )
    assert g._a2a_N_i_loc * cp0 == N_token, (
        f"the token-i shard does not PARTITION the extent: N_i_loc={g._a2a_N_i_loc} x cp0={cp0} = "
        f"{g._a2a_N_i_loc * cp0}, not N_token={N_token}. Rows are lost or double-owned"
    )
    if cp1 > 1:
        assert g._a2a_N_j_loc * cp1 == N_token, (
            f"the token-j shard does not PARTITION the extent: N_j_loc={g._a2a_N_j_loc} x "
            f"cp1={cp1} = {g._a2a_N_j_loc * cp1}, not N_token={N_token}"
        )
        nt_j = g._a2a_nt_j_pp
        assert (nt_j - 1) * tile_n < g._a2a_N_j_loc <= nt_j * tile_n, (
            f"nt_j_pp={nt_j} does not minimally cover N_j_loc={g._a2a_N_j_loc} at tile_n={tile_n} "
            f"(need {(nt_j - 1) * tile_n} < {g._a2a_N_j_loc} <= {nt_j * tile_n}); a floor leaves "
            "the last partial j-tile of every peer block undrained"
        )
    else:
        assert g._a2a_N_j_loc == 0, (
            f"a FLAT mesh must leave the j sentinel at 0, got _a2a_N_j_loc={g._a2a_N_j_loc}. The "
            "copy_fn reads that zero as 'j unsharded, use tile_n'; a filled-in N routes every 1-D "
            "column through the 2-D per-peer clamp instead"
        )

    nt_i = g._a2a_nt_pp
    assert (nt_i - 1) * tile_m < g._a2a_N_i_loc <= nt_i * tile_m, (
        f"nt_pp={nt_i} does not minimally cover N_i_loc={g._a2a_N_i_loc} at tile_m={tile_m} (need "
        f"{(nt_i - 1) * tile_m} < {g._a2a_N_i_loc} <= {nt_i * tile_m}); the per-peer M-tiling "
        "scheduler would then grid short of the peer block and leave its tail rows unstored"
    )
    assert g._a2a_nt_i_pp == nt_i, (
        f"_a2a_nt_i_pp={g._a2a_nt_i_pp} diverged from _a2a_nt_pp={nt_i}; the 2-D scheduler reads "
        "the former and the 1-D path the latter, so a divergence grids the two meshes differently"
    )

    n_j_eff = (N_token // cp1) if cp1 > 1 else N_token
    band = cluster_n * tile_n
    run_j = g._a2a_cluster_run_j
    assert (run_j - 1) * band < n_j_eff <= run_j * band, (
        f"cluster_run_j={run_j} does not minimally cover one peer_j width {n_j_eff} in "
        f"cluster-tile units of {band} (cluster_n={cluster_n} x tile_n={tile_n}); need "
        f"{(run_j - 1) * band} < {n_j_eff} <= {run_j * band}. Short is the run_j sub-band "
        "silent-zero class -- the band ends before the peer's last columns and they are never put"
    )


@matrix_exempt(
    "asserts a REFUSAL at the kernel's own front door on a (cp, N, N_loc) arithmetic relation -- "
    "there is no operand, no tile and no drain variant involved, so the matrix axes do not apply"
)
@numeric_exempt("asserts a refusal at the public API, not a computed value")
@pytest.mark.parametrize(
    "cp,N,N_loc,expect_raise",
    [
        # The measured defect: N/cp = 129.5, a caller floors it, 8 of 2072 tokens vanish.
        (16, 2072, 129, True),
        # And the ceiling, which over-covers -- equally not a partition.
        (16, 2072, 130, True),
        # A shallower mesh reaches it too; this is not a cp=16 property. Note the violation is
        # `cp*N_loc != N`, which a caller can hit even when `N % cp == 0` simply by passing the wrong
        # N_loc -- that is the (4, 1000, 249) row.
        (4, 1000, 250, False),
        (4, 1000, 249, True),
        # Every 1-D (cp, N, N_loc) the 40-cell acceptance grid actually elects (STEP 1,
        # w8plan/step1/variants.csv). These must keep passing: the guard is an addition, and a
        # guard that also rejects the production path is a regression wearing a fix's clothes.
        (2, 2176, 1088, False),
        (4, 2176, 544, False),
        (8, 2176, 272, False),
        (16, 2176, 136, False),
        (16, 12416, 776, False),
    ],
)
def test_the_1d_store_refuses_an_n_loc_that_does_not_partition_n(monkeypatch, cp, N, N_loc, expect_raise):
    """1-D ``configure_a2a_gemm_native`` must refuse peer blocks that do not TILE the token extent.

    The i axis is the only cp-split axis in a 1-D mesh, so the cp peer blocks must cover N exactly.
    Without the guard a caller that floors ``N/cp`` is ACCEPTED and the remainder is never visited --
    measured at ``cp=16, N=2072, N_loc=129``: ``16*129 = 2064``, and the last 8 tokens are dropped with
    no raise and no warning. Nothing downstream can notice, because the recv is SIZED from ``N_loc``:
    the missing rows are not short, they are absent from the geometry.

    The 2-D branch has carried the analogous guards all along (``N_loc == N//cp0`` and ``N % cp1 == 0``),
    which is what makes the 1-D omission an omission rather than a design choice.

    The guard is fail-SAFE -- a refusal replacing a wrong answer -- and the second half of the pool
    pins that the production path is untouched.

    ``build_p2p_table`` is monkeypatched to an all-NVLink table so the coupled/IB topology guard cannot
    fire first and satisfy the ``pytest.raises`` for the wrong reason.

    **Every N in the pool is a multiple of 8.** ``arbitrary_n`` carries its own 16-B alignment guard
    (``N % 8 == 0``) that runs EARLIER, and an N like 1002 trips it instead -- a ``pytest.raises`` with
    no ``match=`` would have been satisfied by the wrong guard and this test would have passed while
    proving nothing. The ``match=`` is what caught it; keep it.
    """
    _requires_sm90()
    import fold_cp_ops.distributed.gemm_sm90_a2a as G

    monkeypatch.setattr(G, "build_p2p_table", lambda pe_table: [True] * len(tuple(pe_table)))
    kw = dict(cp=cp, my_cp_rank=0, B=1, N_loc=N_loc, pe_table=tuple(range(cp)), N=N,
              arbitrary_n=True, dynamic=True)
    if expect_raise:
        with pytest.raises(ValueError, match=r"cp\*N_loc == N"):
            _gemm().configure_a2a_gemm_native(**kw)
    else:
        g = _gemm()
        g.configure_a2a_gemm_native(**kw)
        assert g._a2a_N_i_loc == N_loc and g._a2a_N_j_loc == 0, (
            "a 1-D configure must leave the j axis unsharded (the 0 sentinel) and record N_loc as "
            "the i-axis peer block"
        )


#: Every keyword `main`'s A2A-fused TriMul workflow passes to this kernel's two front doors, and the
#: values it passes them at. NOT transcribed from a signature -- MEASURED, by running `main`'s e2e over
#: all 18 tags of the acceptance grid with the two `configure_*` methods wrapped and their bound
#: arguments recorded (`w8plan/step1/`, STEP 1 of `docs/a2a_bringback.md`; raw records in
#: `w8plan/step1/main_e2e/<tag>/rank<N>.jsonl`, joined into `variants.csv`).
#:
#: A signature transcription would only prove our signature matches itself. This proves our signature
#: admits what the workflow ACTUALLY sends, which is the property N2 depends on.
_WORKFLOW_KWARGS_MEASURED = {
    "configure_a2a_gemm_native": {
        "B", "N", "N_loc", "a_major", "arbitrary_n", "cluster_drain", "cluster_multislot",
        "cluster_n", "consumer_strided", "consumer_strided_putwarp", "consumer_warpgroups", "cp",
        "cp_axis_sizes", "decoupled", "dyn_cp", "dynamic", "ib_quiet", "ib_ring", "my_cp_rank",
        "pe_aligned_tiling", "pe_table", "producer_tma", "ring_depth",
    },
    "configure_a2a_sharded": {
        "B", "N", "consumer_warpgroups", "device_mesh", "dyn_cp", "dynamic", "gemm_native",
        "ib_ring", "pe_aligned_tiling", "pe_map", "placements", "ring_depth", "rows_per_peer",
    },
}

#: Keywords this kernel deliberately REFUSES. Empty, and that is the finding: over the whole 40-cell
#: grid the workflow sent nothing we do not accept. The set exists so a future removal at the kernel
#: source has a declared home instead of turning the assertion below into a puzzle -- an entry here
#: must come with a front-door test proving the refusal (see
#: `tests/distributed/test_gemm_a2a_epi.py::test_removed_hybrid_drain_kwargs_rejected`, which pins the
#: the drain-consolidation removals, none of which the workflow sends).
_WORKFLOW_KWARGS_REFUSED: dict[str, set[str]] = {
    "configure_a2a_gemm_native": set(),
    "configure_a2a_sharded": set(),
}


@matrix_exempt(
    "compares two SIGNATURES against a measured keyword set -- there is no kernel launch, no "
    "operand and no shape, so no matrix axis applies"
)
@numeric_exempt("compares parameter names, not a computed value")
@pytest.mark.parametrize("method", sorted(_WORKFLOW_KWARGS_MEASURED))
def test_every_workflow_kwarg_is_accepted_or_refused(method):
    """Every keyword the workflow sends is a parameter of ours, or is declared REFUSED.

    This is the N0 gate. The workflow (`main:fused_trimul.py`) drives these two doors directly, so a
    keyword it sends that we do not accept is a `TypeError` the moment N2 ports the workflow -- and it
    would surface at rank 0 of a multi-node run, not here.

    The measured set is non-empty by assertion: an empty set would make this test pass while checking
    nothing, which is the failure mode a coverage test is most prone to.
    """
    measured = _WORKFLOW_KWARGS_MEASURED[method]
    refused = _WORKFLOW_KWARGS_REFUSED[method]
    assert measured, f"the measured keyword set for {method} is empty -- it would prove nothing"
    ours = {p for p in inspect.signature(getattr(GemmA2ASm90, method)).parameters if p != "self"}
    unclassified = measured - ours - refused
    assert not unclassified, (
        f"{method}: the workflow passes {sorted(unclassified)}, which is neither a parameter of our "
        f"signature nor a declared REFUSED keyword. Add the parameter, or declare the refusal AND "
        f"land a front-door test for it -- an undeclared gap becomes a TypeError at rank 0."
    )
    assert not (refused & ours), (
        f"{method}: {sorted(refused & ours)} is declared REFUSED but IS a parameter -- the "
        f"declaration and the signature disagree, so one of them is stale."
    )


@matrix_exempt(
    "audits the class's METHOD RESOLUTION -- which methods it defines at all -- so it launches no "
    "kernel and varies with no declared axis"
)
def test_the_a2a_back_does_not_copy_the_kernel():
    """Pin the back half of the NO-KERNEL-COPY claim that ``dual_gated_gemm_a2a.py:73`` mirrors.

    Purpose
    -------
    The front's comment says its mechanism is a "mirror of stagec a2a / the back ``GemmA2ASm90``",
    so the claim is made about BOTH halves and was pinned on neither. The front's pin lives in
    ``test_dual_gated_gemm_a2a.py``; this is the back's, in the file that mirrors its source, per
    the one-test-per-source rule.

    Functionality & semantics
    -------------------------
    Asserts ``GemmSm90A2A`` does not define ``kernel``, ``mma_warpgroup_role``,
    ``producer_warpgroup_role``, ``epilogue`` or ``__call__`` in its own ``vars()``. It legitimately
    overrides many hooks -- operand remaps, stage counts, extra warpgroups, the drain role -- and
    that is the design; what it must never do is re-implement the kernel entry or a warpgroup role,
    because such a copy keeps working while silently diverging from the base it was copied from.

    ``vars(cls)``, not ``hasattr``: ``hasattr`` is satisfied by the inherited attribute and would
    pass unconditionally.

    Input requirements
    ------------------
    None. Class-dict inspection; no GPU, no process group.

    Returns / Raises
    ----------------
    None. Raises ``AssertionError`` naming the copied method.
    """
    from fold_cp_ops.distributed.gemm_sm90_a2a import GemmSm90A2A

    own = set(vars(GemmSm90A2A))
    copied = {"kernel", "mma_warpgroup_role", "producer_warpgroup_role", "epilogue", "__call__"} & own
    assert not copied, (
        f"the A2A back re-implements {sorted(copied)}, so the kernel IS copied -- it will drift "
        f"from GemmSm90 silently, keeping working while ceasing to match."
    )
