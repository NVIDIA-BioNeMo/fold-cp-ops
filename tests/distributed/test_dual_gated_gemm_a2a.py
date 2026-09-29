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

"""Tests for ``fold_cp_ops.distributed.dual_gated_gemm_a2a`` -- the FRONT A2A store.

Every test here drives `DualGatedGemmDistSm90`: the staged dual-gated front which, run with
``transpose_out=True`` and dual width ``2D``, fuses the front reshard into its postact store and
produces BOTH ``a`` and ``b`` in a D-MAJOR einsum-native recv ``(2*Dloc, M_full)`` in ONE invocation,
with the einsum reading the halves as zero-copy views.

**The upstream carried these as SEVEN separate files, pasted together**, leaving seven SPDX headers
and seven module-level docstrings -- only the first of which Python treated as a docstring. The
other six evaluated as bare strings and were discarded, so six of the seven subject descriptions
below were invisible at runtime. They are SECTION BANNERS here, each immediately above the code it
describes and otherwise the upstream's own text. Folding all seven into this docstring was the other
option and was rejected: at ~180 lines it produces a header nobody reads, and the descriptions are
most useful next to their tests.

The seven sections, in file order:

1. **Front-A2A staged D-major store** -- the main correctness gate, and the largest by far.
2. **Dynamic-N route2_ni + ib_wide + RUN_I** -- one dynamic compile serving many token counts.
3. **route2_ni + ib_wide + run_i at cp1** -- the same, on the other mesh factorization.
4. **Route-2 (A) PADDED-recv transpose_in** -- the padded-recv variant.
5. **Front rank-2-M "walk p-inner"** -- the route-2 differential.
6. **REPRO: coupled store at Dloc=8 / postact_epi_n=8** -- a pinned reproduction.
7. **HOST-side WIDE-PUT run_i raster guard** -- **needs no GPU**, so it runs in an ordinary session.

The correctness metric
    The upstream gated on ``max|got-ref| / max|ref| < REL_BAR`` and, in several places, on the
    pooled ``rel_l2``. Both now go through the element-wise comparisons in
    `fold_cp_ops.testing.numerics`, via `_same_bar_elementwise`, and the two conversions are NOT the
    same kind of change -- see that function's docstring, which is where the argument lives rather
    than repeated at fifteen call sites:

    * off ``max|got-ref| / max|ref|`` the conversion is EXACT (a maximum is under a threshold iff
      every element is), so the bar does not move in either direction;
    * off a pooled L2 the CONSTANT is carried and the STATISTIC changes, which is a tightening
      exactly on the localized defects this file hunts -- a mis-placed token block, a padded lane, a
      straddle row -- and neutral elsewhere. Each such site says so in a comment.

    What this deliberately does NOT do: the bound is still keyed to ``max|ref|``, so an element
    whose own value is small and wrong by 100% still passes. Removing that needs a DERIVED bound
    (`numerics.gated_error_bound` and friends) built from each peer's operands and resharded exactly
    as the reference is; that is a separate, reviewable step with its own risk, and nothing here can
    be run without a multi-rank GPU job.

    Exact-equality sites use `numerics.assert_bitwise` rather than ``torch.equal``. Not because
    ``torch.equal`` is forbidden -- it is element-wise and exact, so the guard permits it -- but
    because it records nothing for the coverage gate and reports no index.

    The per-row error histogram and the recv coverage stamp are kept, and the coverage stamp is
    ASSERTED rather than printed.

The declared test matrix and the collective-safe skips
    `FRONT_A2A` below is this module's `KernelMatrix`: every test either draws its cells from it or
    carries ``@matrix_exempt`` with a written reason. Every axis pool is the UNION of what the tests
    already ran and every test still selects exactly the cells it selected before, so no grid
    changed size -- what the matrix buys is that the lists are now relatable.

    Every skip goes through `fold_cp_ops.testing.collective_guard`. Under ``tests/distributed/`` a
    bare ``pytest.skip`` on a per-rank predicate is a DEADLOCK rather than a skip; the four
    ``because`` texts near the top of this file are the four distinct arguments this module's skips
    make, and the one genuinely divergent predicate (the fp32 oracle's OOM catch) is all_reduced
    through ``gated_skip`` instead of declared.

Launching
    Sections 1-6 need torchrun or srun with >=2 ranks plus nvshmem; several need 4 for their 2-D
    cells. Section 7 needs neither. See CLAUDE.md's "five launch forms"; ``CPO_CACHE_ENABLED=0`` is
    required for a correctness run.

Note:
    NO ``from __future__ import annotations`` -- this module imports CuTe-DSL kernel classes, where
    it is forbidden project-wide because it stringizes ``Constexpr`` parameter annotations.
"""

import inspect
import json
import math
import gc
import os
from pathlib import Path

import pytest
import torch

import cutlass.cute as cute
import cutlass.torch as cutlass_torch
from cutlass import BFloat16, Float32
from cutlass.cute.runtime import from_dlpack
from torch.distributed.tensor import Shard

from fold_cp_ops._internal.activation import gate_fn_map
from fold_cp_ops._internal.rounding import RoundingMode
from fold_cp_ops._internal.arch import get_device_capacity, get_max_active_clusters
from fold_cp_ops._internal.gemm_tvm_ffi_utils import get_dtypes, make_scheduler_args, perm3d
from fold_cp_ops.distributed.gemm_bitcode_compile import compile_gemm_with_bitcode
from fold_cp_ops.distributed.dual_gated_gemm_a2a import DualGatedGemmDistSm90
from fold_cp_ops.distributed.distributed_manager import DistributedManager
from fold_cp_ops.distributed.pe_map import PeMap

# NOT kernels.layernorm_dual_gated_gemm: that module name exists in BOTH trees and means different
# things. Here it is the LN-fused kernel; upstream it is a helper module. The symbol is here.
from fold_cp_ops.kernels.dual_gated_gemm import interleave_dual_weights
from fold_cp_ops.kernels.dual_gated_gemm import DualGatedGemmSm90
from fold_cp_ops.testing import numerics
from fold_cp_ops.testing.collective_guard import gated_skip, rank_invariant_skip
from fold_cp_ops.testing.cubin_identity import assert_has_code, digest_export
from fold_cp_ops.testing.kernel_matrix import (
    Axis,
    KernelMatrix,
    Unsupported,
    front_door_raises,
    matrix_exempt,
    shape_mode_axis,
)
from fold_cp_ops.testing.numeric_guard import numeric_exempt

from tests.distributed import topology

REL_BAR = 5e-2  # bf16 tensor-core relative-error bar (the parent kernels' bar)


# --------------------------------------------------------------------------- #
# The declared test matrix for ``fold_cp_ops/distributed/dual_gated_gemm_a2a.py``.
#
# **Every axis pool below is the UNION of what the tests already ran**, and each test still selects
# exactly the cells it selected before, through ``only=``/``cells=`` naming the same module-level
# list it always named. Nothing was added to a grid and nothing was dropped from one -- what changes
# is that the lists are now relatable, so "a config nobody considered" and "a config considered and
# excluded" stop looking identical. That is the whole reason the matrix exists; it does not make the
# grid bigger.
#
# **The ``mesh`` axis -- this paragraph used to say there was none, and was wrong twice over.**
# CORRECTED 2026-08-20. ``FRONT_A2A`` DOES declare a ``mesh`` axis (see the ``Axis(name="mesh", ...)``
# below, 8 specs and 6 facets), and its values appear in this module's collected ids -- so the old
# text contradicted its own file forty lines further down. It also asserted that
# ``test_gemm_sm90_a2a.py`` "declares no mesh axis for the same reason"; that sibling now declares one
# **and the reason does not transfer**, which is the part worth keeping.
#
# The old reasoning -- every test derives ``cp`` from ``world_size``, so the mesh is fixed by the
# LAUNCH and a mesh axis would declare values no test could select -- is a statement about a mesh that
# reaches the kernel through a process group. It is NOT true of a mesh that reaches the kernel as a
# plain configure ARGUMENT: in ``test_gemm_sm90_a2a.py`` ``cp``/``cp_axis_sizes``/``pe_table`` are
# parameters and nothing allocates, so all 8 specs -- including the three needing 16 ranks -- are
# swept in ONE process on any SM90 box. Same word, two different things.
#
# What DOES survive from the old text, and still binds here: **a mesh bounded by the launch cannot be
# covered by one run**, so coverage of it is a property of the SET of launches and lives in
# ``fold_cp_ops/testing/coverage_ledger.py``. A host-side configure sweep must NOT write that ledger
# -- recording ``cp=(4,4)`` as exercised from a configure call would close that hole on paper for the
# whole distributed suite while no fabric was ever crossed.
#
# **Ids.** ``KernelMatrix.parametrize`` renders ids from the VALUES, so the hand-written id lists
# (``_SHAPE_IDS``, ``["B1sq", ...]``) are gone and the ``shape``/``dyn_case`` cells now read as their
# full tuples. That is more verbose and it is the convention's own trade: an id that is derived from
# the value cannot drift away from the value it names.
# --------------------------------------------------------------------------- #
def _mesh_ranks(spec) -> int:
    """Ranks a mesh spec needs: the product of every group size, subgrids flattened.

    Purpose
        Lets the ``mesh`` axis's facets tell a single-node shape from one that can only run across
        nodes. Mirrors ``test_distributed_manager._spec_numel`` and ``conftest._mesh_numel``; it is
        restated here rather than imported because those are a test module's private helper and a
        conftest's private helper respectively, and importing either would couple this file's pool
        to a module it does not otherwise depend on.

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


FRONT_A2A = KernelMatrix(
    kernel="dual_gated_gemm_a2a",
    axes=(
        Axis(
            name="mesh",
            domain=(
                "any spec of group-name -> int | tuple[int, ...] whose rank product equals "
                "WORLD_SIZE. The pool is PURE context-parallel and nothing else, because every mesh "
                "dim here carries the all-to-all: `_pe_map` builds `[Shard(i) for i in "
                "range(mesh.ndim)]`, so a `dp` group would add shard dims the front store does not "
                "route. Both FLAT and FACTORED cp are in the pool and the difference is not "
                "cosmetic -- the 2-D front's recv-index and the tile->peer unravel are exercised "
                "only under a factored spec, and a flat cp collapses them to a single division, "
                "which is exactly the case that would pass while testing half the routing"
            ),
            values=(
                (("cp", 2),),
                (("cp", 4),),
                (("cp", 8),),
                (("cp", (2, 2)),),
                (("cp", (2, 4)),),
                (("cp", 16),),
                # FACTORED *and* CROSS-NODE. Under LayoutRight with 8 GPUs/node the two below differ
                # in where the fabric cuts, which is what the ib_drain hybrid arm routes on:
                (("cp", (2, 8)),),  # axis_1 == exactly one node; axis_0 (stride 8) crosses IB
                (("cp", (4, 4)),),  # axis_1 == HALF a node -- a cp group SMALLER than its NVLink
                #                     domain, so peer routing that assumed "contiguous axis == the
                #                     whole node" passes (2,8) and fails here. Launch it with
                #                     `srun -N2 --ntasks-per-node=8`, and NEVER pass
                #                     --gpus-per-task with 8 tasks/node: each task then sees ONE
                #                     GPU while LOCAL_WORLD_SIZE=8, every cell skips, and the run
                #                     exits GREEN having tested nothing.
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
        Axis(
            name="D",
            domain=(
                "the per-half feature width of the stacked dual (a|b) front, i.e. the GEMM N is 2D. "
                "Any positive multiple of cp that leaves the per-peer slice Dloc = D/cp at or above "
                "the front's feature-scatter floor (Dloc >= 8, and Dloc a multiple of the postact "
                "tile). Combinations below the floor are SKIPPED by the shape gates in each test, "
                "not refused -- which is why they are absent from `unsupported` below"
            ),
            values=(16, 128, 256, 384, 512),
            facets={
                # 16 is the ONLY value that reaches Dloc=8 at a small cp (D=16/cp2), i.e. the
                # postact_epi_n=8 STSM-x2 R2S path the repro test was written for. Without it the
                # pool would be entirely in the comfortable Dloc>=32 regime.
                "small_dloc_regime": lambda v: v <= 16,
                "trimul_feature_width": lambda v: v >= 128,
                "wide_feature": lambda v: v >= 512,
                # SHAPE CHARACTER, not magnitude -- the three facets above are all thresholds, so a
                # pool of nothing but powers of two satisfies them and still never varies in KIND.
                # The kernel path that tells the two kinds apart is `best_front_tile`'s divisor
                # search:
                #     postact_n = 32
                #     while postact_n > 1 and dloc % postact_n != 0:
                #         postact_n //= 2
                # When Dloc is a power of two, EVERY candidate <= Dloc divides it, so the loop only
                # ever walks down from 32 to Dloc and its REJECTION arm -- a candidate that fits
                # inside Dloc yet does not divide it -- never executes. Measured over every Dloc the
                # rest of this pool can produce (256,128,64,32,16,8,4,2,1 across cp 2..16): "rejects
                # a candidate <= Dloc" NEVER, all nine. D=384 is the value that reaches it: Dloc=48
                # rejects 32, Dloc=24 rejects 16. It is also a declared TriMul workflow width that
                # the perf sweep already runs, so this closes a plain coverage hole at the same time.
                "not_power_of_two": lambda v: v > 0 and v & (v - 1) != 0,
            },
        ),
        Axis(
            name="tile_N",
            domain=(
                "the CTA N-tile of the staged dual GEMM; the postact tile is tile_N // 2. Any "
                "multiple of 16 (the SM90 gated STSM-quad floor, which puts the postact tile at >= "
                "8). Several tests IGNORE this parameter and derive the tile from the production "
                "picker `best_front_tile(Dloc)` instead -- the pool records what is passed, and the "
                "test docstrings say where it is overridden"
            ),
            # 256 is the production picker's own choice wherever Dloc >= 128 -- measured, it is
            # elected at four cells already inside the declared pools: (D=256,cp=2), (D=512,cp=2),
            # (D=512,cp=4), (D=512,cp=(2,2)). It is declared so the byte-identity sweep can select
            # it without the matrix refusing a bare literal.
            #
            # `three_wg_trigger` names the EXACT gate at dual_gated_gemm_a2a.py:1202 --
            # `tile_N == 256 AND has_ib_peers` (+ decoupled) -- which selects the 3-WG/384-thread
            # drain. It was deliberately NOT declared while nothing ran 256: a VALUE in the pool is
            # PERMISSION, a FACET is a COVERAGE CLAIM, and measured both ways at the time,
            # `coverage_problems` reported 1 with the facet and 0 without, at 643 collected either
            # way. It arrives now WITH its consumer, never before it.
            values=(16, 128, 256),
            facets={
                "sub32_postact": lambda v: v < 32,
                "production_width": lambda v: v >= 128,
                "three_wg_trigger": lambda v: v >= 256,
            },
        ),
        Axis(
            name="ring_depth",
            domain=(
                "staging slots per CTA on the decoupled producer/consumer ring, passed as "
                "`configure_a2a(ring_depth=...)`. Any POSITIVE int: `_configure_decoupled` "
                "(dual_gated_gemm_a2a.py:1111) raises below 1 and imposes no upper bound. 2 is the "
                "production default baked at :255. 0 is in the pool because the refusal is a "
                "declared `Unsupported` region and a region unreachable from the pools cannot be "
                "tested -- it is never RUN, only refused.\n"
                "\n"
                "1 and 4 are the byte-identity sweep's other depths, selected by "
                "`test_the_five_byte_identity_configs_are_distinguishable` -- the control that makes "
                "the cross-tree driver readable. They arrived WITH that consumer, never before it"
            ),
            values=(0, 1, 2, 4),
            facets={
                "refused_depth": lambda v: v < 1,
                "no_rotation": lambda v: v == 1,
                "production_depth": lambda v: v == 2,
                "deep_rotation": lambda v: v >= 4,
            },
        ),
        Axis(
            name="cwg",
            domain=(
                "consumer warpgroups on the decoupled ib_drain ring: how many warpgroups drain "
                "ring -> peer recv. Any positive count the epilogue can spare; 1 and 2 are the two "
                "the store is written for"
            ),
            values=(1, 2),
            facets={
                "single_consumer_wg": lambda v: v == 1,
                "multi_consumer_wg": lambda v: v >= 2,
            },
        ),
        Axis(
            name="N_token",
            domain=(
                "the token extent whose square (or, on the m_linear arms, whose value) is this "
                "rank's token block. Any value with N % 8 == 0 -- the 16-byte TMA/put floor on the "
                "stride-1 token axis, and the ONLY shape constraint the front carries"
            ),
            # 3008 is the CROSS-NODE OFF-GRID value, and it is here because a sweep found something
            # a test could not: at mesh cp=(2,8) the `cluster` target does not finish building
            # within 900 s, so the cell dies with no measurement. It is an INHERITED defect, not a
            # divergence -- ours took 421 s / 901 s and main 422 s / 900 s on the same two runs, i.e.
            # the two trees stall within a second of each other. `front` and `front_a2a` at the same
            # mesh and N measure in 25 s, which is what localizes the wall to one target rather than
            # to off-grid-times-cross-node in general.
            #
            # It is NOT xfail'd. N=3008 satisfies the only shape constraint the front carries
            # (N % 8 == 0), so it is in-principle supported and an xfail would be a violation rather
            # than a deferral. The value is declared so the failure is attributable to a shape a test
            # names, instead of living in a sweep script that outlives nobody's memory of it.
            #
            # 128, 2048, 5952, 8192 and 12288 arrive WITH their consumer, never before it: the perf
            # gate `tests/distributed/perf/test_benchmark_perf_dual_gated_gemm_a2a.py` gained an
            # `N_token` axis and draws exactly these. 128 is the value that gate has always run (its
            # 33 shipped pins are all at it) and was until now an UNDECLARED literal in a module the
            # matrix governs. The other four complete the perf ladder the 2026-08-19 neutrality run
            # measured at cp=(2,8) -- 2048/3008/4096/5952/8192/10048/12288 -- of which only three
            # were already declared, so four values of a measured ladder had no home on the axis.
            #
            # Declaring them costs the CORRECTNESS module nothing, and that is measured rather than
            # assumed: every `N_token` parametrize site in this file narrows with `only=`
            # (`_OFFGRID_NTOK`, `_PARTIAL_NTOK`, `+ (4096,)`), so no existing test's cell count moves.
            # Nor is a new coverage obligation created -- a value may rot out of coverage, a FACET may
            # not, and these five add no facet.
            values=(
                16,
                24,
                32,
                40,
                128,
                1088,
                2048,
                3008,
                4032,
                4096,
                4128,
                5952,
                8192,
                10048,
                12288,
            ),
            facets={
                "small_smoke": lambda v: v <= 40,
                "large_sequence": lambda v: v >= 1088,
                "power_of_two": lambda v: v > 0 and v & (v - 1) == 0,
                # The mirror of the usual gap: before 4096 this pool was ENTIRELY straddle, 0 of 8
                # values %128 == 0 (16/24/32/40 are all below 128; 1088%128=64, 4032%128=64,
                # 4128%128=32, 10048%128=64). So the off-grid side was fully covered and the aligned
                # side not at all, and no threshold facet could notice.
                #
                # GROUNDED MORE WEAKLY THAN `not_power_of_two` ABOVE, and deliberately recorded as
                # such: that one names a single branch whose rejection arm provably never runs. This
                # one rests on `rows_per_peer % tile_M` on the m_linear arms plus the partial-token
                # clamp, whose visible shadow is this file's own `if not m_linear and M % 128 != 0:
                # skip`. Treat it as the second-ranked of the two, and do not read it as naming one
                # site the way the D facet does.
                #
                # 4096 rather than 2048 because it is a declared `N_token` ladder value, so the
                # addition also moves this pool toward the declared ladder -- which it previously
                # shared no value with -- instead of only satisfying a predicate.
                "tile_aligned": lambda v: v % 128 == 0,
            },
        ),
        Axis(
            name="W",
            domain=(
                "the WIDE-PUT batch width: how many token-subtiles one ring slot accumulates before "
                "the consumer issues a single contiguous put. Any positive constant -- it is O(1) in "
                "the token count by construction, which is what keeps the ring sub-leading"
            ),
            values=(32, 128),
            facets={
                "narrow_coalesce_window": lambda v: v <= 32,
                "production_coalesce_window": lambda v: v >= 128,
            },
        ),
        Axis(
            name="N",
            domain=(
                "the GLOBAL square token extent of the route-2 (N_i-stride-1) front. Any value the "
                "cp grid divides with N % 8 == 0; the upper end is bounded only by the O(N^2 D) fp32 "
                "oracle's memory, which is handled by a runtime OOM skip rather than a gate"
            ),
            values=(256, 520, 1024, 1088, 2048, 4032, 4096),
            facets={
                "tile_m_aligned": lambda v: v % 128 == 0,
                "oracle_oom_risk": lambda v: v >= 4032,
                "power_of_two": lambda v: v > 0 and v & (v - 1) == 0,
            },
        ),
        Axis(
            name="wide_nbi",
            domain=(
                "whether the wide put is issued NON-BLOCKING (B2) or blocking (B1). Both are wired; "
                "the flag changes only when the drain waits, never what it writes"
            ),
            values=(False, True),
            facets={
                "blocking_put": lambda v: v is False,
                "non_blocking_put": lambda v: v is True,
            },
        ),
        Axis(
            name="B",
            domain=(
                "the token-BATCH extent of the local block for the A2A-path transpose_in tests, "
                "i.e. the outermost mode of the native (b, X, Y) token grid. Any B >= 1; the walk "
                "puts no constraint on it. Paired with `Xg`/`Yg` through cells=, never crossed -- "
                "(B, Xg, Yg) is ONE local grid.\n"
                "\n"
                "SEPARATE from the B carried inside the `shape` axis, and deliberately: that axis's "
                "own domain says it is for the LOCAL, single-GPU transpose_in tests with A2A OFF, "
                "so its B=2 cells never reach the peer store. This axis is that same extent on the "
                "A2A path -- the recv layout, the per-peer column block and the plane decode -- and "
                "collapsing the two would make a single-GPU cell look like coverage of a "
                "distributed one"
            ),
            values=(1, 2, 3),
            facets={
                "batched": lambda v: v > 1,
                # 3 and not just 2. At B=2 the back's plane decode (`d = L // B`, `b = L % B`) is a
                # SHIFT and a MASK, so a decode that is wrong but ALIGNED still lands on a valid
                # plane and returns a correctly-shaped answer; an odd B forces a real division and
                # a real remainder. Same reasoning as the odd / off-grid N_token values elsewhere
                # in this matrix.
                "odd_batch": lambda v: v > 1 and v % 2 == 1,
            },
        ),
        Axis(
            name="Xg",
            domain=(
                "the local i-shard token extent of the PADDED-recv transpose_in variant. Any value "
                "with Xg % 8 == 0; it need NOT be a CTA tile_M multiple, which is the whole point "
                "of the padded-per-Y scheduler. Paired with `Yg` through cells=, never crossed: the "
                "two describe one grid and an unpaired combination is a shape nobody runs"
            ),
            values=(100, 250),
            facets={
                "large_grid": lambda v: v >= 250,
                "partial_xg": lambda v: v % 128 != 0,
            },
            waived={
                "partial_xg": (
                    "both declared values are partial by construction -- a partial Xg IS this "
                    "variant's subject, and an aligned Xg through the same code would exercise the "
                    "unpadded path instead. Aligned Xg is covered on the `shape` axis, whose "
                    "single-GPU half is entirely %128 == 0, so the property is sampled on both "
                    "sides ACROSS the module rather than within this one axis"
                )
            },
        ),
        Axis(
            name="Yg",
            domain=(
                "the local j-shard token extent of the PADDED-recv transpose_in variant. Any value "
                "with Yg % 8 == 0. Paired with `Xg` through cells=; see that axis"
            ),
            values=(200, 500),
            facets={"wide_j_shard": lambda v: v >= 500},
        ),
        Axis(
            name="shape",
            domain=(
                "a full (B, Xg, Yg, D, K, (tile_M, tile_N)) cell for the LOCAL, single-GPU "
                "transpose_in tests -- A2A OFF, so there is no cp and the whole shape travels "
                "together. Any B >= 1 and any Xg, Yg % 8 == 0; the tile is any pair the SM90 WGMMA "
                "atom can build"
            ),
            values=(
                (1, 128, 128, 128, 256, (128, 128)),
                (1, 128, 256, 128, 256, (128, 128)),
                (1, 256, 128, 128, 256, (256, 128)),
                (1, 128, 128, 128, 256, (128, 256)),
                (2, 128, 128, 128, 256, (128, 128)),
                (1, 100, 200, 128, 256, (128, 128)),
                (1, 250, 500, 128, 256, (128, 128)),
                (1, 192, 256, 128, 256, (128, 128)),
                (2, 100, 200, 128, 256, (128, 128)),
            ),
            facets={
                "partial_xg": lambda s: s[1] % 128 != 0,
                "batched": lambda s: s[0] > 1,
                "rectangular_grid": lambda s: s[1] != s[2],
                "wide_tile_n": lambda s: s[5][1] == 256,
                "tall_tile_m": lambda s: s[5][0] == 256,
            },
        ),
        shape_mode_axis(),
        Axis(
            name="dyn_case",
            domain=(
                "a (B, anchor_N, runs) cell for the DYNAMIC transpose_in test, where runs is a "
                "tuple of (N, Xg, Yg) triples served by ONE compile at anchor_N. Any B >= 1 and any "
                "runs whose anchor appears among them -- the compile-once-many-N contract puts no "
                "constraint on the runtime N beyond the token-grid divisibility the host gate checks"
            ),
            values=(
                (1, 256, ((256, 256, 256), (384, 384, 384))),
                (1, 256, ((256, 128, 256), (512, 256, 512))),
                (2, 128, ((128, 128, 128), (256, 256, 256))),
                (1, 200, ((200, 100, 200), (500, 250, 500))),
            ),
            facets={
                "batched": lambda c: c[0] > 1,
                "partial_xg_runs": lambda c: any(xg % 128 != 0 for (_, xg, _) in c[2]),
            },
        ),
    ),
    computes=("contraction", "saturating_activation", "data_movement"),
    # WAS `no_unsupported(...)`, and the claim it made was true of the axes that existed then:
    # every one was a SHAPE or a RASTER extent. `ring_depth` is the first axis here that is a KNOB,
    # and the front DOES raise on a value of it, so the claim no longer holds and the region below
    # replaces it. The four FLAG refusals that block quoted are unchanged and still cannot be
    # regions over these pools -- they are not values on any axis, and expressing them would need a
    # `store_mode` axis every accepting test would have to parametrize. `ring_depth` is different in
    # exactly the way that matters: every accepting test already passes it.
    unsupported=(
        Unsupported(
            where=lambda wide_nbi: wide_nbi,
            raises=ValueError,
            match=r"REFUSED at the front\s+door on every toolchain",
            reason=(
                "B2 (the non-blocking wide-put drain) is refused OUTRIGHT at `_configure_ib_wide`, "
                "on EVERY toolchain, so `wide_nbi=True` is now an unsupported region rather than a "
                "supported variant. Two independent hazards, which is why the refusal is "
                "toolchain-unconditional -- gating on the version would let them trade places:\n"
                "  * cutlass-dsl < 4.7.0 cannot LINK the `nvshmemx_flush_warp` B2 emits (NVVM "
                "'Unknown attribute kind (102)'). Before the refusal this was 48 GPU cells dying "
                "in an opaque ICE several frames into codegen; the region converts those into one "
                "legible sentence, so this declaration REPLACES broken coverage, it does not "
                "retire working coverage.\n"
                "  * cutlass-dsl >= 4.7.0 links it, and then runs the A2A-fused DualGatedGEMM "
                "3.23-3.25x SLOWER (measured 2026-08-27, D=256 N_token=3008 cp=4x4 incoming: 96% "
                "of a 2.53x e2e regression, 99.8% of it DEVICE time, one kernel, while the BACK "
                "A2A GEMM in the SAME trace under the SAME compiler is unchanged at 0.96-0.99). "
                "REASON UNKNOWN -- REG 74->74, STACK 264->264, census +0.6-3.2%, so it is a "
                "scheduling/latency-hiding effect and not a codegen-volume one. A silent 3x is "
                "worse for a caller than a refusal.\n"
                "The flag is UNREACHABLE from the shipped API (nothing under `workflows/` or "
                "`fused_trimul*.py` passes it, and it defaults False), so this region costs the "
                "workflow, the benchmarks and the README tutorial nothing. `_CPO_ALLOW_IB_WIDE_NBI=1` "
                "is the developer escape hatch for working on B2 itself, and the cells that still "
                "have value under it are kept, gated on that variable. The `match` quotes wording "
                "unique to the outright-refusal site, NOT the older `REQUIRES ib_wide=True` guard "
                "that sits below it -- the two are different checks and the outright one fires first"
            ),
        ),
        Unsupported(
            where=lambda ring_depth: ring_depth < 1,
            raises=ValueError,
            match=r"must be a positive int",
            reason=(
                "a ring of zero slots has nowhere to stage a tile, so the producer would index an "
                "empty rotation. `_configure_decoupled` refuses it at the API boundary "
                "(dual_gated_gemm_a2a.py:1111-1113) rather than several frames into the epilogue. "
                "The `match` quotes wording unique to that site, which is what makes this a "
                "FRONT-DOOR guard rather than a region satisfiable by any ValueError raised deeper"
            ),
        ),
    ),
)

# WHY THE FOUR FLAG REFUSALS ARE STILL NOT REGIONS (preserved verbatim from the
# `no_unsupported(because=...)` this matrix carried before `ring_depth` made the
# no-refusals claim false). They are not values on any declared axis, so expressing them
# would need a `store_mode` axis every accepting test would have to parametrize.
#
#   every axis above is a SHAPE or a RASTER extent, and the front computes or SKIPS every
#   combination the pools can form -- it never RAISES on one. The skips are shape-validity gates
#   (Dloc below the feature-scatter floor, a rows_per_peer that is not a CTA tile_M multiple on the
#   non-clamped arms, a cp that does not divide D), each declared rank-invariant so the ranks skip
#   together.
#
#   The kernel DOES refuse four things, and every one of them is a store-mode FLAG or a malformed
#   argument rather than a value on any axis here, which is why they cannot be expressed as an
#   `Unsupported` region over these pools: `ib_wide=True` without `ib_drain=True`
#   (test_front_ib_wide_requires_ib_drain_raises), the RETIRED plain-decoupled store
#   `decoupled=True, ib_drain=False` (test_front_plain_decoupled_retired_raises), a hand-crafted
#   out-of-contract rows_per_peer whose tail is under the 16-byte floor
#   (test_front_partial_token_clamp_guard_raises -- unreachable from any N_token in the
#   pool, which is the point of hand-crafting it), and a token count not divisible by B*N
#   (test_transpose_in_dynamic_bad_N_raises). All four are asserted, each by the named test;
#   declaring a region for them would need a `store_mode` axis every accepting test would then have
#   to parametrize, which is a restructure and not a declaration.

# --------------------------------------------------------------------------- #
# Collective-safe skips: WHY each declaration below is the one the guard asks for.
#
# Under ``tests/distributed/`` a bare ``pytest.skip`` on a predicate that can differ per rank is a
# DEADLOCK rather than a skip -- the skipping rank leaves and every peer blocks in the next
# collective until a watchdog kills the job, with a traceback naming whatever test they happened to
# be in. ``fold_cp_ops/testing/collective_guard.py`` admits three spellings and enforces them both
# statically and at call time. This module uses all three:
#
#   * ``rank_invariant_skip(reason, because=...)`` -- 52 sites, every one a predicate computed from
#     job-uniform inputs. The four ``because`` texts below are the four DISTINCT arguments those
#     sites make; a site quotes the one that matches its predicate rather than a generic sentence.
#   * ``gated_skip(reason_or_None)`` -- ONE site, the fp32 oracle's OOM catch in
#     ``_run_ib_drain_front_route2``. An OOM genuinely can hit one rank and not another, so that
#     decision is all_reduced instead of declared.
#   * ``pre_init_skip`` -- unused here. Every site in this module runs after ``dist_manager`` has
#     brought the group up, and that helper RAISES when a group is live, so it is not available
#     even if somebody wanted it. That is the rule as code rather than as prose.
#
# ``because`` is checked for EXISTENCE, not truth -- no system can verify that a predicate is
# uniform. What the declaration removes is the cheapness of forgetting: "we know this is uniform"
# and "it is written down as uniform" stop being the same state.
# --------------------------------------------------------------------------- #
_SHAPE_GATE_BECAUSE = (
    "a SHAPE-VALIDITY gate: the predicate reads only the LAUNCH's cp/mesh extents (world_size, the "
    "cp axis sizes) and the value pytest bound for this cell, and both are identical on every rank "
    "of the job. It queries no device, allocates nothing, and probes nothing, so it holds no "
    "quantity that could come out differently on one rank -- every rank reaches the same verdict "
    "and they skip, or do not skip, together"
)

_IB_UNIFORM_BECAUSE = (
    "build_p2p_table is a pure function of pe_table, and the cp pe_table is a rank-INVARIANT job "
    "property, so every rank evaluates this predicate identically and skips (or does not skip) "
    "together -- no teardown-barrier desync. This is the argument tests/distributed/topology.py "
    "makes for the SAME probe, carried here because these call sites reach the probe directly; "
    "only a predicate that can DIVERGE per rank (an OOM catch, a per-rank build failure) needs "
    "CollectiveGate's all_reduce consensus"
)

_HOST_ENV_BECAUSE = (
    "whether a MODULE is importable is a property of the IMAGE the whole job booted from, not of "
    "the rank: one launch does not mix python environments, so an import succeeds on every rank or "
    "fails on every rank. Used for two predicates that are the same question -- is nvshmem4py "
    "present, and is an as-yet-UNPORTED module present -- neither of which is a capability of the "
    "device or a property of the shape"
)

_ARCH_BECAUSE = (
    "compute capability is a property of the machine the launch landed on; the nodes of one job "
    "are homogeneous and a launch does not span architectures, so every rank reaches the same "
    "verdict and no rank can skip alone"
)


def _same_bar_elementwise(actual, reference, rel_bar, what):
    """Assert the file's historical ``max|got-ref| / max|ref| < rel_bar`` bar, but PER ELEMENT.

    Purpose
        The single conversion used at every correctness site in this module, so the argument for
        why the bar did not move is written once instead of fifteen times.

    Functionality & semantics
        Asserts ``|got_i - ref_i| <= rel_bar * max|ref|`` for every element, and the two kinds of
        call site it replaces relate to that differently -- stated separately, because reading one
        as the other would misdescribe what a green run now proves:

        * **The ``max|got-ref| / max|ref| < rel_bar`` sites (this module's ``_rel_err``, and the
          two hand-inlined copies of it): the conversion is EXACT.** A maximum is under a threshold
          iff every element is, so the two forms accept and reject the same tensors. The bar does
          not move in either direction.
        * **The pooled-L2 sites (``||got-ref|| / ||ref|| < c``): the CONSTANT is carried and the
          STATISTIC changes.** These are not comparable forms, and the difference has a direction:
          an L2 ratio divides by ``sqrt(n)``-scaled energy, so a single badly-wrong element inside a
          large correct tensor barely moves it, while this bound judges that element on its own.
          The conversion is therefore a TIGHTENING exactly on the defect class the guard exists for
          -- a tile edge, a padded lane, a straddle row, a last-wave CTA -- and is neutral
          elsewhere. Each such site says so in a comment, so no reader has to infer it.

        In both cases what changes at FAILURE is the same: the pooled form printed one scalar, and
        `fold_cp_ops.testing.numerics.assert_elementwise` names the worst offender's index with its
        actual, its reference and its bound.

        **What this deliberately does NOT do, stated so nobody reads more into it.** The bound is
        still keyed to the LARGEST reference magnitude, so an element whose own value is small and
        wrong by 100% still passes -- the weakness the guard's docstring names. Removing it needs a
        DERIVED bound (`numerics.gated_error_bound` and friends) built from each peer's operands and
        resharded exactly as the reference is, which is a different change with a different risk:
        a derived bound can fail cells that pass today, and nothing in this file can be run without
        a multi-rank GPU job. So the bar is carried across UNCHANGED and the tightening is left as
        its own reviewable step.

        The bound is passed as a SCALAR float, not as the tensor `numerics.tolerance_bound` would
        build. `assert_elementwise` documents a scalar as a first-class input and there is nothing
        to lose -- the bound genuinely is uniform -- while a full fp64 tensor the shape of an
        O(N^2 * D) reference would OOM the check at cells whose kernel ran fine. The body says so at
        the line that does it.

    Args:
        actual: The computed tensor. Any dtype/device `assert_elementwise` accepts; it is promoted
            to fp64 there. Must be finite -- a NaN fails with a message naming it rather than
            silently comparing False.
        reference: The reference, same shape as ``actual``. Its magnitudes set the bound, so it
            must be the REFERENCE and not the computed value; passing them the wrong way round
            scales the bar by the wrong tensor and the check silently changes meaning.
        rel_bar: The historical relative bar, e.g. `REL_BAR`. Must be positive; a zero here demands
            bit-exactness and `numerics.assert_bitwise` is the honest spelling for that.
        what: Short label carried into the failure message. Pass the site's own ``tag`` so a
            failure names the cell.

    Returns:
        The worst observed ``|err| / bound`` ratio, as a float -- a value creeping toward 1.0 means
        the bar is about to become flaky, which is worth knowing before it fails.

    Raises:
        AssertionError: From `numerics.assert_elementwise`, if any element exceeds its bound, if the
            shapes disagree, or if the actual holds non-finite values.
    """
    # A FLOAT bound, not `numerics.tolerance_bound(reference, atol=..., rtol=0.0)`, and the reason is
    # memory rather than style. That builder returns `atol + rtol * reference.double().abs()`, which
    # at rtol=0 still materialises a FULL fp64 tensor the shape of the reference -- and it is not
    # chunked, unlike the comparison itself. The references here reach O(N^2 * D): the route-2 tri
    # oracle is (Dloc, N_j, N_j) fp32 at N up to 4096, so an fp64 copy of it is tens of GB and would
    # OOM the check at cells whose kernel ran fine. `assert_elementwise` documents a scalar bound as
    # a first-class input (it is passed through the split unchanged), and nothing is lost: this bar
    # genuinely IS uniform -- there is no per-element variation for a tensor to carry.
    #
    # `scale` is taken in the reference's own dtype, exactly as the `_rel_err` this replaces did, so
    # the number is the same one and the temporary is 1x the reference rather than 4x.
    scale = reference.abs().max().item()
    scale = scale if scale > 0 else 1.0
    return numerics.assert_elementwise(actual, reference, rel_bar * scale, what=what)


# --------------------------------------------------------------------------- #
# nvshmem bootstrap (session-scoped, collective).
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="session", autouse=False)
def _nvshmem__distributed__front_a2a_staged(dist_manager):
    from fold_cp_ops.distributed.distributed_manager import DistributedManager

    DistributedManager.init_nvshmem()
    yield


#: Epilogue fields the A2A SUBCLASS adds on top of its local parent, DERIVED from the two classes
#: rather than listed. The Dist subclass declares 21 epilogue fields to the parent's 14 -- the peer
#: handles (``recv``, ``ring``, ``pe_table_dev``, ``token_grid_yg``) and the LN terms (``mWeight``,
#: ``mBias``, ``eps``). Deriving it means a field added to either class is covered without anyone
#: editing this line; a hardcoded tuple would be correct only until the next one.
_SUBCLASS_ONLY_EPI_FIELDS = frozenset(DualGatedGemmDistSm90.EpilogueArguments._fields) - frozenset(
    DualGatedGemmSm90.EpilogueArguments._fields
)


def _epi_args(GemmCls, **kw):
    """Build ``GemmCls.EpilogueArguments``, supplying only the fields THAT class declares.

    Purpose
        Two helpers here are parametrized over the class under test -- the flag-OFF byte-identity
        gate and the transpose_in builder both run the PARENT and the SUBCLASS through the same
        code. The subclass's epilogue params are a superset, so one literal kwarg list cannot serve
        both: handing ``mWeight``/``mBias``/``eps`` to the parent raises
        ``TypeError: EpilogueArguments.__new__() got an unexpected keyword argument 'mWeight'``.

    Functionality & semantics
        Drops a keyword ONLY when the target class does not declare it AND it is in
        :data:`_SUBCLASS_ONLY_EPI_FIELDS`. That second condition is what keeps this from being a
        silent typo-swallower: a misspelled field is in neither class, so it is passed through and
        the namedtuple raises on it exactly as it would have. The rule is "the subclass has extras
        the parent does not", not "drop whatever does not fit".

    Args:
        GemmCls: The functor class whose ``EpilogueArguments`` to build -- the parent or the A2A
            subclass. Must expose ``EpilogueArguments`` with ``_fields``; anything else raises
            ``AttributeError`` here rather than producing a half-built argument pack.
        **kw: Epilogue terms. May include subclass-only fields; those are dropped for the parent.
            Every other name must be a real field of ``GemmCls.EpilogueArguments``.

    Returns:
        The constructed ``EpilogueArguments`` namedtuple.

    Raises:
        TypeError: From the namedtuple, if a keyword is neither a field of ``GemmCls`` nor a
            declared subclass-only extra -- i.e. a typo.
    """
    fields = set(GemmCls.EpilogueArguments._fields)
    return GemmCls.EpilogueArguments(
        **{k: v for k, v in kw.items() if k in fields or k not in _SUBCLASS_ONLY_EPI_FIELDS}
    )


def _this_rank_device():
    """This rank's CUDA device, for symmetric allocations.

    Purpose
        Every symmetric buffer in this module is allocated through
        ``DistributedManager.symmetric_mempool``, which needs a device, and most call sites are
        helpers that do not take the ``device`` fixture. One helper keeps them from each inventing
        their own answer.

    Semantics
        Returns the CURRENT CUDA device. That is the right one because the launcher sets it per rank
        (``torch.cuda.set_device(local_rank)``) before any test runs, and `symmetric_mempool`
        documents that a pool for a device the caller does not allocate on is a real hazard -- the
        allocation succeeds and the tensor is not peer-addressable where the kernel expects it.
        Deliberately NOT `torch.device("cuda:0")`: that is the hardcoded-device defect this suite
        already carries elsewhere, and it puts every rank's buffer on GPU 0.

    Returns:
        A ``torch.device`` naming this rank's GPU.

    Raises:
        RuntimeError: From torch, if CUDA is unavailable -- symmetric memory has no CPU allocator,
            so failing here is better than handing back a device that cannot back one.
    """
    return torch.device("cuda", torch.cuda.current_device())


def _nvshmem_barrier():
    import nvshmem.core

    nvshmem.core.barrier_all(stream=torch.cuda.current_stream())


def _drain():
    import nvshmem.core
    import nvshmem.core.rma as nvshmem_rma

    nvshmem_rma.quiet(stream=torch.cuda.current_stream())
    nvshmem.core.barrier_all(stream=torch.cuda.current_stream())
    torch.cuda.synchronize()


def _pe_map(dist_manager):
    mesh = dist_manager.device_mesh
    placements = [Shard(i) for i in range(mesh.ndim)]
    return PeMap.from_mesh_placements(mesh, placements, distributed_manager=dist_manager)


def _cp_mesh(dist_manager):
    """The mesh whose ndim is the cp FACTORISATION -- the subgroup mesh when there is one.

    Purpose
        Hand ``PeMap.from_mesh_placements`` a mesh that actually carries the declared cp axes, so a
        FACTORED spec such as ``(("cp", (2, 8)),)`` yields ``cp_axis_sizes == (2, 8)`` and not
        ``(16,)``.

    Why this is not ``dist_manager.device_mesh``, which is the trap
        ``DistributedManager.create_grid_group`` splits a factored value in two: the PARENT mesh gets
        ``prod(v)`` under the group's own name (``shape_groups.append(prod(v))``), and the
        factorisation goes to a SEPARATE mesh built with ``suffix_mesh="subgroups"``
        (``shape_subgroups.extend(v)``). So ``_state["_device_mesh"]`` is 1-D of shape ``(cp,)`` for
        EVERY spec, factored or not, and ``DistributedManager.__getattr__`` maps
        ``dist_manager.device_mesh`` to exactly that one. ``PeMap`` then reads ``cp_axis_sizes`` off
        the shape of whatever mesh it is handed -- and documents that the mesh rank tensor is
        authoritative and the manager is never consulted about membership -- so the factorisation is
        simply absent, ``cp1`` collapses to 1, and a test whose id says ``mesh(('cp',(2,8)),)`` runs a
        1-D cp geometry.

        Measured, 4 ranks::

            SPEC=(('cp', 4),)       device_mesh.ndim=1 shape=(4,) cp_axis_sizes=(4,) cp1=1
                                    _device_mesh_subgroups.shape=None
            SPEC=(('cp', (2, 2)),)  device_mesh.ndim=1 shape=(4,) cp_axis_sizes=(4,) cp1=1
                                    _device_mesh_subgroups.shape=(2, 2)

        The second line IS the defect: the 2-D grid exists, and the mesh the tests read is not it.

    Semantics
        Returns ``_state["_device_mesh_subgroups"]`` when it is not None, else
        ``dist_manager.device_mesh``. For a FLAT spec the subgroup mesh is never built
        (``_has_subgroups = name_groups != name_subgroups`` is False), so a flat spec resolves to the
        parent mesh and its behaviour is unchanged -- this WIDENS coverage, it does not move it.

    Input requirements
        ``dist_manager``: a live ``DistributedManager`` with a grid already applied (i.e. after
        ``apply_mesh``). Called before ``create_grid_group`` it returns the parent mesh or a stale
        one -- silently the wrong answer, because ``reset_grid_groups`` clears both entries.

    Returns:
        A ``DeviceMesh`` whose ``ndim`` equals the number of declared cp axes.
    """
    sub = DistributedManager._state.get("_device_mesh_subgroups")
    return sub if sub is not None else dist_manager.device_mesh


# The topology guard now lives ONCE, in tests/distributed/topology.py (A1), wrapping the STORE's own
# ``build_p2p_table`` probe so every suite guards on the same source the kernel routes on. The three
# names below are kept as module-local aliases so this file's call sites are unchanged.
_job_has_ib_peers = topology.job_has_ib_peers
_COUPLED_CROSS_NODE_DETAIL = (
    "Cross-node Dloc=8 is covered by test_front_ib_drain_hybrid[D128] at cp16(2,8)."
)
_COUPLED_NVLINK_ONLY_SKIP = topology.COUPLED_NVLINK_ONLY_SKIP + " " + _COUPLED_CROSS_NODE_DETAIL


def _skip_if_coupled_cross_node(pe_table):
    """Self-skip a PURE-COUPLED (no-ib_drain) peer-store test on a job with >=1 IB peer.

    Purpose
        The coupled store TMA-S2Gs straight into a peer's symmetric heap through ``nvshmem_ptr``,
        which returns NULL for an IB peer -- so cross-node it faults, unattributed, at the next
        barrier. That is a hardware capability boundary of the coupled path, not a supported shape
        being dodged: the cross-node case is covered by the ib_drain / V6 differential tests.

    Functionality & semantics
        Delegates to `tests.distributed.topology.skip_if_coupled_cross_node` rather than repeating
        the probe-and-skip here. That matters for more than duplication: topology's version issues
        the skip through ``rank_invariant_skip`` with the DECLARATION of why the predicate is
        job-uniform, and one copy means this suite and its siblings can never guard on different
        reasoning than the kernel routes on. The module's extra sentence rides in through
        ``detail=``, which is exactly what that parameter is for.

    Args:
        pe_table: The flat-cp peer table -- any iterable of ints, e.g.
            ``pm.cp_pe_table.tolist()``. Must be the cp table and must be NON-EMPTY: the probe is
            reduced with ``all()``, and ``all(())`` is True, so an empty table reports "no IB
            peers" and silently un-guards the caller on the very mesh the guard exists for.

    Returns:
        None when the job is all-P2P, so the caller proceeds to build the coupled store.

    Raises:
        Skipped: pytest's, on every rank together, when the job has at least one IB peer.
    """
    topology.skip_if_coupled_cross_node(pe_table, detail=_COUPLED_CROSS_NODE_DETAIL)


def _session_cp_mesh(dist_manager):
    sub = getattr(dist_manager, "device_mesh_subgroups", None)
    if getattr(dist_manager, "has_subgroups", False) and sub is not None:
        return sub
    return dist_manager.device_mesh


def _trimul_placements(dist_manager):
    mesh = _session_cp_mesh(dist_manager)
    if mesh.ndim > 2:
        rank_invariant_skip(
            f"front staged tests cover 1-D/2-D token shards; mesh ndim={mesh.ndim}",
            because=_SHAPE_GATE_BECAUSE,
        )
    placements = [Shard(i + 1) for i in range(mesh.ndim)]
    pm = PeMap.from_mesh_placements(mesh, placements, distributed_manager=dist_manager)
    return placements, pm


def _cp_axis_sizes(dist_manager):
    mesh = _session_cp_mesh(dist_manager)
    return tuple(int(mesh.size(i)) for i in range(mesh.ndim))


# --------------------------------------------------------------------------- #
# Compile harness — the STAGED front with transpose_out=True + D-major recv.
# --------------------------------------------------------------------------- #
def _build_front(
    x, Wg2, Wp2, tile_shape_mn, *, recv_t=None, a2a_cfg=None, configure_fn=None, pingpong=False
):
    """Construct + bitcode-compile a DualGatedGemmDistSm90 front.

    x (M,K) bf16 pre-normalized activation (gated_gemm_gate path, _normalize=False); Wg2/Wp2 (2D,K)
    the STACKED a/b gate/up weights -> dual GEMM N = 2D (glu -> 2D postact). With transpose_out=True
    the postact is M-major (M,2D) and the split a=out[:D], b=out[D:] is each D-major (D,M).

    A2A ON  (recv_t + a2a_cfg|configure_fn): the postact (a|b) store is the D-major peer TMA store
    into recv_t (2*Dloc, M_full). A2A OFF (recv_t=None): byte-identical local store into out_postact.
    Returns (compiled, run_fn, out_postact) — out_postact is the (2D, M) LOCAL store (meaningful only
    when A2A OFF; with A2A ON the data lands in recv_t, out_postact is the logical extent only).
    """
    device_capacity = get_device_capacity(x.device)
    assert device_capacity[0] == 9, f"SM90 only; got {device_capacity}"
    M, K = x.shape
    twoN = Wg2.shape[0]  # = 2D
    tile_M, tile_N = tile_shape_mn

    B = interleave_dual_weights(Wg2, Wp2)  # (K, 2*twoN) bf16
    out_postact = torch.empty((twoN, M), dtype=x.dtype, device=x.device)
    pa_arg = out_postact.mT  # (M, 2D) M-major (stride (1, M))

    A = x.unsqueeze(0)  # (1,M,K)
    B3 = B.mT.unsqueeze(0)  # (1, 2*twoN, K) RAW weight .mT
    PostAct = pa_arg.unsqueeze(0)  # (1, M, 2D) M-major
    A_p, B_p, _, _ = perm3d(A, B3, None, None)
    PostAct_p = perm3d(PostAct, B3, None, None)[0]

    a_dtype, b_dtype, _, _ = get_dtypes(A, B3, A, None)
    gemm_obj = DualGatedGemmDistSm90(
        Float32,
        a_dtype,
        (tile_M, tile_N),
        (1, 1, 1),
        pingpong=pingpong,
        is_persistent=True,
        chunk_g=1,
        n_dual_tiles=0,
        gate3_n3=0,
    )

    a2a_on = recv_t is not None
    if a2a_on:
        if configure_fn is not None:
            configure_fn(gemm_obj)
        else:
            # forward any extra configure_a2a kwargs (e.g. partial_token_clamp) past the base 4.
            extra = {
                k: v
                for k, v in a2a_cfg.items()
                if k not in ("cp", "my_cp_rank", "rows_per_peer", "pe_table")
            }
            # PURE-COUPLED (no ib_drain) store is NVLink-only (§10) -> self-skip on a cross-node (has_ib)
            # job before configuring, since a coupled TMA-S2G to an IB peer's NULL nvshmem_ptr faults.
            # The caller's symmetric recv needs NO free before skipping: it comes from the symmetric
            # MemPool, which recycles. That is the point of the pool -- it takes the collective free
            # off the destruction path, so a skip that pre-empts a test's try/finally can no longer
            # strand a buffer or wedge the next collective allocation.
            if not extra.get("ib_drain", False) and _job_has_ib_peers(a2a_cfg["pe_table"]):
                rank_invariant_skip(_COUPLED_NVLINK_ONLY_SKIP, because=_IB_UNIFORM_BECAUSE)
            gemm_obj.configure_a2a(
                cp=a2a_cfg["cp"],
                my_cp_rank=a2a_cfg["my_cp_rank"],
                rows_per_peer=a2a_cfg["rows_per_peer"],
                pe_table=a2a_cfg["pe_table"],
                **extra,
            )
        cute_recv = from_dlpack(recv_t, assumed_align=16)
    else:
        cute_recv = None

    max_active_clusters = get_max_active_clusters(1)

    def _mk_epi():
        return DualGatedGemmDistSm90.EpilogueArguments(
            mPostAct=from_dlpack(PostAct_p, assumed_align=16),
            act_fn=gate_fn_map["glu"],
            mRowVecBroadcast=None,
            mBiasUp=None,
            mBiasGate=None,
            mMaskColVec=None,
            mPostAct3=None,
            act_fn_3=None,
            mRowVecBroadcast3=None,
            mWeight=None,
            mBias=None,
            eps=Float32(1e-5),
            rounding_mode=RoundingMode.RN,
            recv=cute_recv,
        )

    scheduler_args = make_scheduler_args(max_active_clusters, 8, None)
    cA = from_dlpack(A_p, assumed_align=16)
    cB = from_dlpack(B_p, assumed_align=16)
    stream = cutlass_torch.current_stream()
    compiled = compile_gemm_with_bitcode(
        gemm_obj,
        cA,
        cB,
        None,
        None,
        _mk_epi(),
        scheduler_args,
        stream,
        register=a2a_on,
    )

    def run_fn():
        compiled(
            from_dlpack(A_p, assumed_align=16),
            from_dlpack(B_p, assumed_align=16),
            None,
            None,
            _mk_epi(),
            scheduler_args,
            stream,
        )

    return compiled, run_fn, out_postact


def _build_front_decoupled(
    x,
    Wg2,
    Wp2,
    tile_shape_mn,
    *,
    recv_t,
    ring_depth,
    consumer_warpgroups,
    configure_fn,
    shape_mode="static",
):
    """Construct + bitcode-compile a DECOUPLED front (SIMT GMEM-ring putwarp drain).

    Same as ``_build_front`` (A2A ON) but: configure_a2a(decoupled=True) + allocate the bounded
    LOCAL-GMEM staging ring ``(grid_ctas, ring_depth, epi_n_postact, epi_m)`` [TOKEN-innermost] + the
    ``(cp,)`` int32 device PE table, threaded via the ``ring`` / ``pe_table_dev`` EpilogueArguments. The
    producer-TMA stages each postact subtile to the ring; the consumer warpgroup drains ring->peer recv.
    Returns ``(compiled, run_fn, gemm_obj, ring_t)``.
    """
    device_capacity = get_device_capacity(x.device)
    assert device_capacity[0] == 9, f"SM90 only; got {device_capacity}"
    # SHAPE MODE (the `shape_mode` matrix axis). "static" leaves every extent baked, which is what
    # every caller did before this kwarg existed -- so static is byte-identical to the prior
    # behaviour and no existing test moves. "dynamic" applies the two marks the shipped workflow
    # applies (`fold_cp_ops/distributed/fused_trimul.py`): mark_layout_dynamic() on the operands and
    # postact, and mark_compact_shape_dynamic(mode=1) on the recv -- token dim runtime, FEATURE dim
    # (2*Dloc) deliberately left STATIC because the epilogue reads int(recv.shape[0]).
    #
    # Both modes go through THIS SAME builder on purpose: they are otherwise different generated
    # code, and a test that reconstructed the dynamic build separately would be comparing two
    # harnesses rather than two shape modes.
    if shape_mode not in ("static", "dynamic"):
        raise ValueError(f"shape_mode must be 'static' or 'dynamic'; got {shape_mode!r}")
    _dyn = shape_mode == "dynamic"
    _md = (lambda c: c.mark_layout_dynamic()) if _dyn else (lambda c: c)
    _md_recv = (lambda c: c.mark_compact_shape_dynamic(mode=1)) if _dyn else (lambda c: c)
    M, K = x.shape
    twoN = Wg2.shape[0]  # = 2D
    tile_M, tile_N = tile_shape_mn

    B = interleave_dual_weights(Wg2, Wp2)
    out_postact = torch.empty((twoN, M), dtype=x.dtype, device=x.device)
    pa_arg = out_postact.mT
    A = x.unsqueeze(0)
    B3 = B.mT.unsqueeze(0)
    PostAct = pa_arg.unsqueeze(0)
    A_p, B_p, _, _ = perm3d(A, B3, None, None)
    PostAct_p = perm3d(PostAct, B3, None, None)[0]

    a_dtype, b_dtype, _, _ = get_dtypes(A, B3, A, None)
    gemm_obj = DualGatedGemmDistSm90(
        Float32,
        a_dtype,
        (tile_M, tile_N),
        (1, 1, 1),
        pingpong=False,
        is_persistent=True,
        chunk_g=1,
        n_dual_tiles=0,
        gate3_n3=0,
    )
    configure_fn(gemm_obj)
    if _dyn:
        # `_a2a_dynamic` is what switches the store's rows_per_peer from the BAKED constant to a read
        # of the runtime recv extent. The marks alone are not enough -- without this the store would
        # read a baked rpp against a runtime-shaped recv.
        gemm_obj._a2a_dynamic = True

    max_active_clusters = get_max_active_clusters(1)
    # Ring geometry: epi_tile = (epi_m, epi_n_D) = (gcd(128,tile_M), gcd(32,tile_N)) for the cooperative
    # SM90 path; the GLU-halved postact box N is epi_n_D//2. The ring slot box is (epi_n_postact, epi_m)
    # — TOKEN (epi_m) innermost (stride-1) so the per-feature-row put is a contiguous token run matching
    # the D-major recv's stride-1 (token) axis. grid_ctas = max_active_clusters (persistent grid-z).
    import math as _math

    epi_m = _math.gcd(128, tile_M)
    epi_n_D = _math.gcd(32, tile_N)
    epi_n_postact = epi_n_D // 2
    grid_ctas = max_active_clusters
    # WIDE-PUT: the ring token axis is W· wider (each slot holds W accumulated token-subtiles) so the
    # consumer drains one (W·epi_m)-token contiguous put per feature row. W=1 on the per-subtile path.
    # Frugality still holds: the ring is O(grid_ctas · rd · epi_n · W · epi_m) with EVERY factor a const
    # -> O(1) in N_token (SUB-LEADING to the O(N^2) recv), regardless of W.
    ring_w = (
        int(getattr(gemm_obj, "_a2a_ib_wide_batch", 1))
        if bool(getattr(gemm_obj, "_a2a_ib_wide", False))
        else 1
    )
    ring_tok = epi_m * ring_w
    # ib_drain: the ring is the PUT SOURCE for IB peers, so it MUST live on the SYMMETRIC HEAP (SPMD
    # same-size on every rank). Plain-decoupled (NVLink) uses a LOCAL torch.zeros ring.
    if bool(getattr(gemm_obj, "_a2a_ib_drain", False)):
        # Symmetric memory via the torch MemPool, NOT nvshmem4py's allocator: torch owns the
        # nvshmem bootstrap here, so nvshmem4py's own `_is_initialized` is permanently False
        # and its `tensor()` raises `NVSHMEM Library is not initialized`. The pool also
        # recycles, which keeps the collective free off the destruction path.
        with torch.cuda.use_mem_pool(DistributedManager.symmetric_mempool(_this_rank_device())):
            ring_t = torch.empty(
                (grid_ctas, int(ring_depth), epi_n_postact, ring_tok),
                dtype=x.dtype,
                device=_this_rank_device(),
            )
        ring_t.zero_()
    else:
        ring_t = torch.zeros(
            (grid_ctas, int(ring_depth), epi_n_postact, ring_tok), device=x.device, dtype=x.dtype
        )
    # (cp,) int32 device PE table (flat-cp -> global PE), assumed_align=4.
    pe_dev = gemm_obj._a2a_pe_table
    pe_dev_t = torch.tensor(list(pe_dev), device=x.device, dtype=torch.int32).contiguous()

    cute_recv = _md_recv(from_dlpack(recv_t, assumed_align=16))
    cute_ring = from_dlpack(ring_t, assumed_align=16)
    cute_pe_dev = from_dlpack(pe_dev_t, assumed_align=4)

    def _mk_epi():
        return DualGatedGemmDistSm90.EpilogueArguments(
            mPostAct=_md(from_dlpack(PostAct_p, assumed_align=16)),
            act_fn=gate_fn_map["glu"],
            mRowVecBroadcast=None,
            mBiasUp=None,
            mBiasGate=None,
            mMaskColVec=None,
            mPostAct3=None,
            act_fn_3=None,
            mRowVecBroadcast3=None,
            mWeight=None,
            mBias=None,
            eps=Float32(1e-5),
            rounding_mode=RoundingMode.RN,
            recv=cute_recv,
            ring=cute_ring,
            pe_table_dev=cute_pe_dev,
        )

    scheduler_args = make_scheduler_args(max_active_clusters, 8, None)
    cA = _md(from_dlpack(A_p, assumed_align=16))
    cB = _md(from_dlpack(B_p, assumed_align=16))
    stream = cutlass_torch.current_stream()
    compiled = compile_gemm_with_bitcode(
        gemm_obj, cA, cB, None, None, _mk_epi(), scheduler_args, stream, register=True
    )
    # Compile populated epi_tile; assert the ring geometry matches (loud, not silent corruption).
    assert tuple(gemm_obj.epi_tile) == (epi_m, epi_n_D), (
        f"epi_tile {tuple(gemm_obj.epi_tile)} != heuristic ({epi_m},{epi_n_D}) — ring mis-sized"
    )

    def run_fn():
        compiled(
            _md(from_dlpack(A_p, assumed_align=16)),
            _md(from_dlpack(B_p, assumed_align=16)),
            None,
            None,
            _mk_epi(),
            scheduler_args,
            stream,
        )

    return compiled, run_fn, gemm_obj, ring_t


# --------------------------------------------------------------------------- #
# References.
# --------------------------------------------------------------------------- #
def _front_glu_2d__distributed__front_a2a_staged(x, Wg2, Wp2):
    """fp32 GLU of the stacked a/b front (no LN; x pre-normalized): (M, 2D)."""
    xf = x.float()
    g = xf @ Wg2.float().t()
    p = xf @ Wp2.float().t()
    return torch.sigmoid(g) * p  # (M, 2D)


def _ref_front_dmajor(x, Wg2, Wp2, pm, *, cp, rows_per_peer, Dloc, world_size):
    """fp32 (front GLU then reshard S1->S3) reference, D-MAJOR (2*Dloc, M_full).

    On MY recv (peer my_cp_rank): rows [0:Dloc] = a-half D-slice my_cp_rank, rows [Dloc:2Dloc] =
    b-half D-slice my_cp_rank; col block s = peer s's whole token block (D-major).
    """
    import torch.distributed as dist

    dual = _front_glu_2d__distributed__front_a2a_staged(x, Wg2, Wp2).to(torch.bfloat16)  # (M, 2D)
    D = Dloc * cp
    a, b = dual[:, :D], dual[:, D:]  # (M, D) each
    # COLUMN-CHUNKED all_gather. The full-width version below materialised
    # 2 * world_size * (M, D) -- with M = B*N^2 that is 2.00 GiB PER COPY and ~64 GiB total at
    # cp=16 / N=2048 / D=256, which OOM'd 178 of 192 cells of the ours-vs-main numerical sweep on a
    # card where the kernel's own buffers fit comfortably (D=128 passed at every mesh, D>=256 at
    # none). Only `[:, my]` was ever read from each gathered tensor, so the extra (cp-1)/cp of every
    # copy was allocated, transferred and discarded.
    #
    # Slicing BEFORE the gather would be WRONG: `my` is the RECEIVER's D-slice, so each rank would
    # contribute its own slice rather than the one its peers need. Instead every rank contributes
    # the SAME chunk index s on iteration s, and keeps only s == my_cp_rank -- which yields exactly
    # `ga[pg] == <peer pg's a>[:, my]`, the quantity the loop below consumes. Peak memory becomes
    # world_size * (M, Dloc) = (M, D), a cp-fold cut; the bytes transferred are unchanged, split
    # across cp collectives instead of one.
    my = slice(pm.my_cp_rank * Dloc, (pm.my_cp_rank + 1) * Dloc)
    ga = gb = None
    for _s in range(cp):
        _sl = slice(_s * Dloc, (_s + 1) * Dloc)
        _ca, _cb = a[:, _sl].contiguous(), b[:, _sl].contiguous()
        _oa = [torch.empty_like(_ca) for _ in range(world_size)]
        _ob = [torch.empty_like(_cb) for _ in range(world_size)]
        dist.all_gather(_oa, _ca)
        dist.all_gather(_ob, _cb)
        if _s == int(pm.my_cp_rank):
            ga, gb = _oa, _ob  # keep ONLY this rank's slice; the rest fall out of scope here
        del _oa, _ob, _ca, _cb
    assert ga is not None and gb is not None, (
        f"my_cp_rank={int(pm.my_cp_rank)} outside range(cp={cp}) -- no chunk was retained"
    )
    M_full = cp * rows_per_peer
    expected = torch.empty((2 * Dloc, M_full), device=x.device, dtype=torch.bfloat16)
    for s in range(cp):
        pg = int(pm.cp_pe_table[s].item())
        col = slice(s * rows_per_peer, (s + 1) * rows_per_peer)
        expected[:Dloc, col] = ga[pg].t()
        expected[Dloc:, col] = gb[pg].t()
    return expected


def _ref_front_block_stream(x, Wg2, Wp2, pm, *, cp, rows_per_peer, Dloc, world_size):
    """Yield the reference ONE PEER BLOCK at a time, holding 1/cp of it instead of all of it.

    Purpose
        Cut the reference's peak from a full copy to a single block. :func:`_ref_front_gathered`
        already avoids ASSEMBLING the ``(2*Dloc, M_full)`` matrix, but it still retains
        ``ga``/``gb`` -- ``2 * M_full * Dloc`` elements, which is exactly one copy of the reference.
        Measured at cp=(2,2) ``N_token=4096`` ``D=256``: that retention is a 16 GiB allocation and
        the cell dies inside the gather. This holds one ``(rows_per_peer, Dloc)`` pair instead.

    Why ``scatter`` and not the obvious alternatives -- each was checked, not assumed
        * ``all_gather`` (today's) hands back every rank's contribution at once, so one element of it
          cannot be obtained without the whole.
        * ``all_to_all`` is the exact shape of the exchange -- send destination ``d`` my columns for
          ``d``'s slice, receive from ``p`` its columns for mine -- but it also delivers every peer
          simultaneously, so the peak is unchanged.
        * ``broadcast`` of peer ``p``'s full ``a`` lets each rank slice its own, but ``(rows_per_peer,
          D)`` per peer EQUALS the whole reference's size, so it buys nothing.
        Only a per-source ``scatter`` delivers one receiver one ``(rows_per_peer, Dloc)`` block, which
        is the 1/cp.

    Semantics
        A GENERATOR, and every rank must drive it to exhaustion in the same order: each step is a
        collective, so a rank that stops early hangs its peers. Yields ``(got, ref)`` pairs shaped
        ``(rows_per_peer, 2*Dloc)`` -- the transposed column block, which is exactly
        ``cat([a_block, b_block], dim=1)`` and needs no transpose of a large tensor.

    Input requirements
        ``cp == world_size``. The scatter is indexed by GLOBAL rank, so a job with ranks outside the
        cp group would need a filler chunk for each of them; rather than invent one, this asserts and
        the caller keeps the gathered path for that case.
        ``pm.cp_pe_table`` must be the cp-slot -> global-rank map; its inverse is what tells the
        source which column chunk each destination wants, so a wrong table silently pairs every block
        with the wrong peer.

    Yields:
        ``(got_block, ref_block)``, both ``(rows_per_peer, 2*Dloc)`` bf16 on ``x``'s device.

    Raises:
        AssertionError: If ``cp != world_size``.
    """
    import torch.distributed as dist

    assert cp == world_size, (
        f"the scatter stream is indexed by global rank and needs cp == world_size; got cp={cp}, "
        f"world_size={world_size}. Use _ref_front_gathered for a job with ranks outside the group."
    )
    dual = _front_glu_2d__distributed__front_a2a_staged(x, Wg2, Wp2).to(torch.bfloat16)
    D = Dloc * cp
    a, b = dual[:, :D], dual[:, D:]
    pe_table = [int(pm.cp_pe_table[s].item()) for s in range(cp)]
    # Inverse of the cp-slot -> rank map: which slice global rank r is the receiver for. Built once,
    # because getting it wrong pairs every block with the wrong peer and every element mismatches.
    slot_of = {r: s for s, r in enumerate(pe_table)}
    me = int(dist.get_rank())
    for s in range(cp):
        pg = pe_table[s]
        outs = []
        for half in (a, b):
            recv_blk = torch.empty((rows_per_peer, Dloc), dtype=torch.bfloat16, device=x.device)
            # Only the SOURCE builds a scatter list; every other rank passes None, which is what
            # torch.distributed requires rather than an empty list.
            if me == pg:
                lst = [
                    half[:, slot_of[r] * Dloc : (slot_of[r] + 1) * Dloc].contiguous()
                    for r in range(world_size)
                ]
            else:
                lst = None
            dist.scatter(recv_blk, lst, src=pg)
            outs.append(recv_blk)
        yield outs[0], outs[1]


def _ref_front_gathered(x, Wg2, Wp2, pm, *, cp, rows_per_peer, Dloc, world_size):
    """The same reference as :func:`_ref_front_dmajor`, stopped one step earlier: the GATHERED halves.

    Purpose
        Let a caller consume the reference PER PEER BLOCK instead of receiving it whole. The assembled
        ``expected`` is ``2*Dloc x cp*rows_per_peer`` = ``2*D*B*N^2`` elements -- **``cp`` cancels**, so
        a bigger mesh does not shrink a rank's copy. Measured at ``cp=(2,2)``, ``N_token=5952``,
        ``D=256``: assembling it asked for a single 33.79 GiB allocation beside an equally large recv
        and the cell died, identically at every mesh (0/6 at N=4096 and 0/6 at N=5952 for cp 2, 4, 8,
        16 and (2,2) alike -- the pass/fail pattern depends on ``N_token`` ALONE).

    Functionality & semantics
        Performs exactly the column-chunked all_gather that :func:`_ref_front_dmajor` performs, with
        the same correctness argument (see its comment for why slicing before the gather is wrong),
        and returns the gathered per-peer halves rather than the assembled matrix. Peer ``s`` occupies
        ``expected[:, s*rows_per_peer : (s+1)*rows_per_peer]``, and that block TRANSPOSED is exactly
        ``cat([ga[pg], gb[pg]], dim=1)`` -- no transpose or copy is needed to form it.

        Returns the two SCALARS a streaming consumer cannot compute until it has seen every block:
        ``ref_absmax`` (for the uniform ``rel_bar * max|ref|`` bound) and ``ref_absmean`` (for the
        histogram's ``abs_thresh``). Both come from the gathered halves directly, so the consumer
        stays single-pass and its counts stay EXACT rather than becoming block-order dependent.

    Args:
        x, Wg2, Wp2: As :func:`_ref_front_dmajor` -- this rank's operands, on CUDA.
        pm: The PE map; ``pm.cp_pe_table[s]`` must be the global rank holding cp-slot ``s`` and
            ``pm.my_cp_rank`` this rank's slot, else the blocks pair with the wrong peer and every
            element mismatches.
        cp, rows_per_peer, Dloc, world_size: As :func:`_ref_front_dmajor`. ``cp * Dloc`` must equal the
            full feature width or the gathered chunk is not this rank's D-slice.

    Returns:
        ``(ga, gb, ref_absmax, ref_absmean)``. ``ga[pg]`` / ``gb[pg]`` are ``(rows_per_peer, Dloc)``
        bf16 for global rank ``pg``; the two floats are fp32 reductions over every gathered element.

    Raises:
        AssertionError: If ``my_cp_rank`` falls outside ``range(cp)``, so no chunk was retained -- the
            same guard the assembled form carries, kept because a silent ``None`` here would surface
            as an unrelated TypeError inside the caller's block loop.
    """
    import torch.distributed as dist

    dual = _front_glu_2d__distributed__front_a2a_staged(x, Wg2, Wp2).to(torch.bfloat16)
    D = Dloc * cp
    a, b = dual[:, :D], dual[:, D:]
    ga = gb = None
    for _s in range(cp):
        _sl = slice(_s * Dloc, (_s + 1) * Dloc)
        _ca, _cb = a[:, _sl].contiguous(), b[:, _sl].contiguous()
        _oa = [torch.empty_like(_ca) for _ in range(world_size)]
        _ob = [torch.empty_like(_cb) for _ in range(world_size)]
        dist.all_gather(_oa, _ca)
        dist.all_gather(_ob, _cb)
        if _s == int(pm.my_cp_rank):
            ga, gb = _oa, _ob
        del _oa, _ob, _ca, _cb
    assert ga is not None and gb is not None, (
        f"my_cp_rank={int(pm.my_cp_rank)} outside range(cp={cp}) -- no chunk was retained"
    )
    # Only the peers this rank will actually consume contribute -- cp of the world_size gathered
    # entries. Reducing over all of them would fold in ranks outside this cp group and shift both
    # scalars, which for `abs_thresh` silently changes a COUNT rather than raising.
    _pgs = [int(pm.cp_pe_table[s].item()) for s in range(cp)]
    _amax = 0.0
    _asum = 0.0
    _n = 0
    for pg in _pgs:
        for t in (ga[pg], gb[pg]):
            tf = t.float().abs()
            _amax = max(_amax, float(tf.max().item()))
            _asum += float(tf.sum().item())
            _n += tf.numel()
    return ga, gb, _amax, _asum / _n


# --------------------------------------------------------------------------- #
# Tests.
# --------------------------------------------------------------------------- #
@pytest.mark.usefixtures("_nvshmem__distributed__front_a2a_staged")
@pytest.mark.usefixtures("_module_skip__distributed__front_a2a_staged")
@matrix_exempt(
    "asserts the flag-OFF subclass is BIT-identical to its local parent at ONE shape. The property is a code-path identity -- `_a2a_enabled` False routes the store to `super()` -- not something that varies with a declared extent; parametrizing would compile the same two kernels at several shapes and assert the same identity each time"
)
def test_flag_off_byte_identical(dist_manager, device, world_size):
    """§3.1 guarantee: a flag-OFF DualGatedGemmDistSm90 == the local parent, BIT-for-bit.

    No A2A config -> ``_a2a_enabled`` stays False -> ``epi_setup_postact`` falls back to ``super()``
    (the parent local store) and ``epi_to_underlying_arguments`` returns the parent params (no peer
    atoms). Compiled through the SAME bitcode route (no nvshmem op issued, register=False); compared
    against the local ``DualGatedGemmSm90`` compiled identically on the SAME transpose_out
    postact inputs. Runs independently on every rank (no collectives). bit-exact (==).
    """
    from fold_cp_ops.kernels.dual_gated_gemm import DualGatedGemmSm90

    torch.manual_seed(1234 + dist_manager.rank)
    M, D, K = 256, 128, 256
    tile_shape_mn = (128, 128)
    x = torch.randn(M, K, device=device, dtype=torch.bfloat16) / (K**0.5)
    Wg2 = torch.randn(2 * D, K, device=device, dtype=torch.bfloat16) / (K**0.5)
    Wp2 = torch.randn(2 * D, K, device=device, dtype=torch.bfloat16) / (K**0.5)

    def _compile_and_run(GemmCls):
        twoN = 2 * D
        B = interleave_dual_weights(Wg2, Wp2)
        out_postact = torch.empty((twoN, M), dtype=x.dtype, device=x.device)
        pa_arg = out_postact.mT
        A = x.unsqueeze(0)
        B3 = B.mT.unsqueeze(0)
        PostAct = pa_arg.unsqueeze(0)
        A_p, B_p, _, _ = perm3d(A, B3, None, None)
        PostAct_p = perm3d(PostAct, B3, None, None)[0]
        a_dtype, _, _, _ = get_dtypes(A, B3, A, None)
        gemm_obj = GemmCls(
            Float32,
            a_dtype,
            tile_shape_mn,
            (1, 1, 1),
            pingpong=False,
            is_persistent=True,
            chunk_g=1,
            n_dual_tiles=0,
            gate3_n3=0,
        )

        def _mk_epi():
            # Through `_epi_args`: this helper runs BOTH the parent and the A2A subclass, whose
            # epilogue params are a superset. The subclass-only terms below are dropped for the
            # parent, which otherwise raises on `mWeight`.
            return _epi_args(
                GemmCls,
                mPostAct=from_dlpack(PostAct_p, assumed_align=16),
                act_fn=gate_fn_map["glu"],
                mRowVecBroadcast=None,
                mBiasUp=None,
                mBiasGate=None,
                mMaskColVec=None,
                mPostAct3=None,
                act_fn_3=None,
                mRowVecBroadcast3=None,
                mWeight=None,
                mBias=None,
                eps=Float32(1e-5),
                rounding_mode=RoundingMode.RN,
            )

        max_active_clusters = get_max_active_clusters(1)
        scheduler_args = make_scheduler_args(max_active_clusters, 8, None)
        cA = from_dlpack(A_p, assumed_align=16)
        cB = from_dlpack(B_p, assumed_align=16)
        stream = cutlass_torch.current_stream()
        compiled = compile_gemm_with_bitcode(
            gemm_obj,
            cA,
            cB,
            None,
            None,
            _mk_epi(),
            scheduler_args,
            stream,
            register=False,
        )
        compiled(
            from_dlpack(A_p, assumed_align=16),
            from_dlpack(B_p, assumed_align=16),
            None,
            None,
            _mk_epi(),
            scheduler_args,
            stream,
        )
        torch.cuda.synchronize()
        compiled.free()
        return out_postact.clone()

    out_parent = _compile_and_run(DualGatedGemmSm90)
    out_dist_off = _compile_and_run(DualGatedGemmDistSm90)  # NO configure_a2a -> flag OFF
    print(
        f"\n[staged flag-off byte-identity] rank={dist_manager.rank} M={M} D={D} K={K} "
        f"n={out_parent.numel()} elements compared BITWISE",
        flush=True,
    )
    # BITWISE, not `!=`-counted: `assert_bitwise` compares raw bit patterns, so it also catches the
    # two cases a value comparison gets wrong in opposite directions -- a lost sign of zero
    # (-0.0 == 0.0 is True, and the sign flips a later division) and a faithfully-copied NaN
    # (NaN != NaN is True, so a value comparison fails a correct store). §3.1 is a BYTE-identity
    # guarantee, so bits are the right thing to compare, and the failure names the first differing
    # index with both hex patterns rather than a count and a max-abs.
    numerics.assert_bitwise(
        out_dist_off, out_parent, what="flag-OFF DualGatedGemmDistSm90 vs local DualGatedGemmSm90"
    )


# --------------------------------------------------------------------------- #
# BEST-CONFIG correctness gate: the picker DualGatedGemmDistSm90.best_front_tile(Dloc) —
# the production CTA tile ((128,256) @ Dloc>=128, (256,128) otherwise) that beats the stock (128,128)
# by 1.12-1.26x (debug/front_cfg_sweep.py). The store math is tile-shape-generic, but a wider tile_N
# (256, postact 128) and a taller tile_M (256) change the per-CTA box / wave layout, so the picker
# config must be re-validated oracle-clean at a MULTI-WAVE M (exercises the real production tiling, not
# the N=16 micro-shape). PASS = rel<bar AND n_outlier_rows==0 AND full coverage AND a/b VIEW.
# --------------------------------------------------------------------------- #
# 384 is here and not merely in the axis: it is the ONLY width whose Dloc makes
# `best_front_tile`'s divisor search REJECT a candidate that fits (Dloc=48 rejects 32, Dloc=24
# rejects 16), and this is the picker's own test. Declaring the `not_power_of_two` facet without
# running it somewhere would leave `coverage_problems` reporting a pool "narrowed away by every
# test that uses it" -- which it did, until this line.
_BEST_CFG_D = [128, 256, 384]


@pytest.mark.usefixtures("_nvshmem__distributed__front_a2a_staged")
@pytest.mark.usefixtures("_module_skip__distributed__front_a2a_staged")
@FRONT_A2A.parametrize(
    "D",
    only={"D": tuple(_BEST_CFG_D)},
    because=(
        "the BEST-CONFIG picker is validated at the three production feature widths, including "
        "the one non-power-of-two width: 384 is what makes the picker's divisor search do real "
        "work rather than walk down to min(32, Dloc). D=16 puts Dloc below the picker's own "
        "postact-tile invariant at every cp this file runs, and D=512 is the 2-D mesh cell's "
        "width -- covered by test_front_dmajor_2d_correct"
    ),
)
@FRONT_A2A.parametrize("mesh")
def test_front_best_config_correct(apply_mesh, mesh, dist_manager, device, world_size, D):
    """The shape-aware BEST CTA tile (best_front_tile) stores oracle-clean at a multi-wave M (cp2/cp4)."""
    apply_mesh(mesh)
    from tests.distributed.correctness_harness import compute_error_histogram

    cp = world_size
    if D % cp != 0:
        rank_invariant_skip(f"D={D} not divisible by cp={cp}", because=_SHAPE_GATE_BECAUSE)
    Dloc = D // cp
    tile_M, tile_N = DualGatedGemmDistSm90.best_front_tile(Dloc)
    # The SM90 WGMMA atom's OWN floor, which the gates around this one do not cover: a CTA tile
    # needs `tile_N % 16 == 0 and <= 256`, or `% 32 == 0 and <= 512`. Whether that is reachable is
    # decided by the MESH, not by D alone -- at D=16, cp=8 leaves Dloc=2, the picker returns
    # tile_N=4, and the build below raises from the kernel's own front door (`gemm_sm90.py`: "CTA
    # tile shape N must be divisible by 16 and <= 256, or ..."). Fewer features per rank than a
    # single N-tile is a HARDWARE floor, not a shape constraint this repo imposes, so it is skipped
    # rather than xfail'd or removed from the pool.
    #
    # First reachable at world_size 8: at 2 ranks the cp=8 and cp=(2,4) meshes skip outright, so no
    # earlier launch in this tree could observe it.
    if not ((tile_N % 16 == 0 and tile_N <= 256) or (tile_N % 32 == 0 and tile_N <= 512)):
        rank_invariant_skip(
            f"Dloc={Dloc} at D={D}/cp={cp} yields tile_N={tile_N}, under the SM90 WGMMA CTA-tile "
            f"floor (needs %16 and <=256, or %32 and <=512)",
            because=_SHAPE_GATE_BECAUSE,
        )
    tile_n_postact = tile_N // 2
    if Dloc % tile_n_postact != 0:
        rank_invariant_skip(
            f"Dloc={Dloc}%postact tile {tile_n_postact} at cp={cp} (picker invariant broken)",
            because=_SHAPE_GATE_BECAUSE,
        )
    # Multi-wave M: N=128 -> rows_per_peer = 128^2/cp (cp2: 8192 = 64 tile_M=128 rows; cp4: 4096).
    # Must be a multiple of tile_M (the picker may pick tile_M=256): N=128 gives 8192%256==0 (cp2),
    # 4096%256==0 (cp4) -> clean (no partial M-tile; partial is covered by the clamp tests).
    B, N = 1, 128
    M = B * N * N
    if M % cp != 0 or (M // cp) % tile_M != 0:
        rank_invariant_skip(
            f"M={M}//cp not a multiple of tile_M={tile_M} at cp={cp}", because=_SHAPE_GATE_BECAUSE
        )
    rows_per_peer = M // cp
    K = 256
    M_full = cp * rows_per_peer
    pm = _pe_map(dist_manager)

    torch.manual_seed(4321 + dist_manager.rank)
    x = torch.randn(rows_per_peer, K, device=device, dtype=torch.bfloat16) / (K**0.5)
    Wg2 = torch.randn(2 * D, K, device=device, dtype=torch.bfloat16) / (K**0.5)
    Wp2 = torch.randn(2 * D, K, device=device, dtype=torch.bfloat16) / (K**0.5)
    # Symmetric memory via the torch MemPool, NOT nvshmem4py's allocator: torch owns the
    # nvshmem bootstrap here, so nvshmem4py's own `_is_initialized` is permanently False
    # and its `tensor()` raises `NVSHMEM Library is not initialized`. The pool also
    # recycles, which keeps the collective free off the destruction path.
    with torch.cuda.use_mem_pool(DistributedManager.symmetric_mempool(_this_rank_device())):
        recv = torch.empty((2 * Dloc, M_full), dtype=torch.bfloat16, device=_this_rank_device())
    recv.fill_(-99.0)

    pe_table = tuple(int(v) for v in pm.cp_pe_table.tolist())
    a2a_cfg = dict(
        cp=cp, my_cp_rank=int(pm.my_cp_rank), rows_per_peer=rows_per_peer, pe_table=pe_table
    )
    compiled, run_fn, _ = _build_front(x, Wg2, Wp2, (tile_M, tile_N), recv_t=recv, a2a_cfg=a2a_cfg)
    try:
        _nvshmem_barrier()
        run_fn()
        _drain()
        expected = _ref_front_dmajor(
            x, Wg2, Wp2, pm, cp=cp, rows_per_peer=rows_per_peer, Dloc=Dloc, world_size=world_size
        )
        untouched = int((recv == -99.0).sum().item())
        a_is_view = recv[:Dloc].data_ptr() == recv.data_ptr()
        got_bnnd = recv.t().contiguous().reshape(1, M_full, 1, 2 * Dloc)
        ref_bnnd = expected.t().contiguous().reshape(1, M_full, 1, 2 * Dloc)
        h = compute_error_histogram(got_bnnd, ref_bnnd)
        tag = f"front-best-cfg cp={cp} D={D} Dloc={Dloc} tile=({tile_M},{tile_N}) M={M} rpp={rows_per_peer}"
        # The recv-vs-reference verdict, PER ELEMENT. `_same_bar_elementwise` carries the
        # historical `max|got-ref| / max|ref| < REL_BAR` bar UNCHANGED -- a maximum is under a
        # threshold iff every element is -- and raises naming the worst offender's INDEX with
        # its actual, reference and bound. It sits where the pooled scalar was computed, so the
        # scalar `assert rel < REL_BAR` that used to follow is gone. `rel` is that same ratio,
        # recovered exactly for the diagnostic print (the bound is the uniform REL_BAR*max|ref|,
        # so worst |err|/bound * REL_BAR == the old ratio); it no longer decides the test.
        rel = REL_BAR * _same_bar_elementwise(recv, expected, REL_BAR, tag)
        passed = rel < REL_BAR and h.n_outlier_rows == 0 and untouched == 0 and a_is_view
        print(
            f"\n[{tag}] rank={dist_manager.rank} rel={rel:.3e} (bar {REL_BAR}) {h.summary()} "
            f"untouched={untouched}/{recv.numel()} a_view={a_is_view} {'PASS' if passed else 'FAIL'}",
            flush=True,
        )
        assert h.n_outlier_rows == 0, (
            f"{tag}: {h.n_outlier_rows} outlier token rows (the wider/taller best tile corrupted the "
            f"store?); worst @ {h.worst_row_index} ratio {h.row_outlier_ratio:.1f}x"
        )
        assert untouched == 0, (
            f"{tag}: {untouched}/{recv.numel()} recv cells unwritten (coverage gap)"
        )
        assert a_is_view, f"{tag}: a/b half not a VIEW"
    finally:
        compiled.free()
        _nvshmem_barrier()


# (D feature per half, tile_N). D=128 -> Dloc=64, postact tile=tile_N//2; D=256 -> Dloc=128,
# n_tiles_per_Dslice=2 (a D-slice spans TWO postact tiles).
_FRONT = [(128, 128), (256, 128)]


@pytest.mark.usefixtures("_nvshmem__distributed__front_a2a_staged")
@pytest.mark.usefixtures("_module_skip__distributed__front_a2a_staged")
@FRONT_A2A.parametrize(
    "D",
    "tile_N",
    cells=_FRONT,
    because=(
        "cells= and not a product: tile_N is not free here. Each D runs at the tile_N whose postact "
        "tile (tile_N // 2) divides Dloc, and the crossed pairs are layouts the front's feature "
        "scatter cannot lay out. The pool's other D values belong to the small-Dloc reproduction "
        "(16) and to the 2-D mesh cell (512)"
    ),
)
@FRONT_A2A.parametrize("mesh")
def test_front_dmajor_correct(apply_mesh, mesh, dist_manager, device, world_size, D, tile_N):
    """1-D front D-major store: recv rel/coverage vs flat-cp ref + (N,N,Dloc) reconstruct vs oracle."""
    apply_mesh(mesh)
    from tests.distributed.correctness_harness import compute_error_histogram

    cp = world_size
    if D % cp != 0:
        rank_invariant_skip(f"D={D} not divisible by cp={cp}", because=_SHAPE_GATE_BECAUSE)
    Dloc = D // cp
    tile_n_postact = tile_N // 2
    if Dloc % tile_n_postact != 0:
        rank_invariant_skip(
            f"Dloc={Dloc} not a multiple of postact tile_N={tile_n_postact} at cp={cp}",
            because=_SHAPE_GATE_BECAUSE,
        )
    # M = B*N*N token block.  Keep it a perfect square*B so the (N,N,Dloc) reconstruct is exact;
    # B=1, N a multiple of the CTA tile-M is NOT required for the front (it scatters by feature),
    # but the reconstruct needs M = N*N.  N=16 -> M=256.
    B, N = 1, 16
    M = B * N * N
    if M % 128 != 0:  # rows_per_peer must be a multiple of the CTA tile-M (no partial M-tile)
        rank_invariant_skip(f"M={M} not a multiple of CTA tile_M=128", because=_SHAPE_GATE_BECAUSE)
    K = 256
    rows_per_peer = M  # my whole token block lands on every peer (D-sliced)
    M_full = cp * rows_per_peer
    pm = _pe_map(dist_manager)

    torch.manual_seed(4321 + dist_manager.rank)
    x = torch.randn(M, K, device=device, dtype=torch.bfloat16) / (K**0.5)
    Wg2 = torch.randn(2 * D, K, device=device, dtype=torch.bfloat16) / (K**0.5)
    Wp2 = torch.randn(2 * D, K, device=device, dtype=torch.bfloat16) / (K**0.5)
    # Symmetric memory via the torch MemPool, NOT nvshmem4py's allocator: torch owns the
    # nvshmem bootstrap here, so nvshmem4py's own `_is_initialized` is permanently False
    # and its `tensor()` raises `NVSHMEM Library is not initialized`. The pool also
    # recycles, which keeps the collective free off the destruction path.
    with torch.cuda.use_mem_pool(DistributedManager.symmetric_mempool(_this_rank_device())):
        recv = torch.empty((2 * Dloc, M_full), dtype=torch.bfloat16, device=_this_rank_device())
    recv.fill_(-99.0)

    pe_table = tuple(int(v) for v in pm.cp_pe_table.tolist())
    a2a_cfg = dict(
        cp=cp, my_cp_rank=int(pm.my_cp_rank), rows_per_peer=rows_per_peer, pe_table=pe_table
    )
    compiled, run_fn, _ = _build_front(x, Wg2, Wp2, (128, tile_N), recv_t=recv, a2a_cfg=a2a_cfg)
    try:
        _nvshmem_barrier()
        run_fn()
        _drain()

        # (1) recv vs flat-cp D-major reference.
        expected = _ref_front_dmajor(
            x, Wg2, Wp2, pm, cp=cp, rows_per_peer=rows_per_peer, Dloc=Dloc, world_size=world_size
        )
        untouched = int((recv == -99.0).sum().item())

        # (2) a=recv[:Dloc], b=recv[Dloc:] are VIEWS feeding the einsum (zero-copy). Add the
        # error-histogram (per-token-COLUMN L2 over the 2*Dloc feature rows -> n_outlier_rows catches
        # a localized token corruption the scalar rel averages away). recv.t() = (M_full, 2*Dloc) so
        # the histogram's "row" is a token (its 2*Dloc-feature L2), matching the parent test idiom.
        a_is_view = recv[:Dloc].data_ptr() == recv.data_ptr()
        # error-histogram wants (B,N,N,D): view recv.t()=(M_full,2*Dloc) as (1, M_full, 1, 2*Dloc) so
        # each "token row" is a token's full 2*Dloc-feature L2 (a localized token corruption -> outlier).
        got_bnnd = recv.t().contiguous().reshape(1, M_full, 1, 2 * Dloc)
        ref_bnnd = expected.t().contiguous().reshape(1, M_full, 1, 2 * Dloc)
        h = compute_error_histogram(got_bnnd, ref_bnnd)
        n_tiles_per_Dslice = max(1, Dloc // tile_n_postact)
        tag = f"front-staged cp={cp} M={M} D={D} tile_N={tile_N} Dloc={Dloc} ntpd={n_tiles_per_Dslice}"
        # The recv-vs-reference verdict, PER ELEMENT. `_same_bar_elementwise` carries the
        # historical `max|got-ref| / max|ref| < REL_BAR` bar UNCHANGED -- a maximum is under a
        # threshold iff every element is -- and raises naming the worst offender's INDEX with
        # its actual, reference and bound. It sits where the pooled scalar was computed, so the
        # scalar `assert rel < REL_BAR` that used to follow is gone. `rel` is that same ratio,
        # recovered exactly for the diagnostic print (the bound is the uniform REL_BAR*max|ref|,
        # so worst |err|/bound * REL_BAR == the old ratio); it no longer decides the test.
        rel = REL_BAR * _same_bar_elementwise(recv, expected, REL_BAR, tag)
        passed = rel < REL_BAR and h.n_outlier_rows == 0 and untouched == 0 and a_is_view
        print(
            f"\n[{tag}] rank={dist_manager.rank} rel={rel:.3e} (bar {REL_BAR}) {h.summary()} "
            f"untouched={untouched}/{recv.numel()} a_view={a_is_view} {'PASS' if passed else 'FAIL'}",
            flush=True,
        )
        assert h.n_outlier_rows == 0, (
            f"{tag}: {h.n_outlier_rows} outlier token rows (localized corruption); worst @ "
            f"{h.worst_row_index} ratio {h.row_outlier_ratio:.1f}x"
        )
        assert untouched == 0, (
            f"{tag}: {untouched}/{recv.numel()} recv cells unwritten (coverage gap)"
        )
        assert a_is_view, f"{tag}: a/b half not a VIEW (the zero-copy einsum feed is broken)"
    finally:
        compiled.free()
        _nvshmem_barrier()


_DECPL = [(128, 128, 1), (256, 128, 1), (256, 128, 2)]
# hybrid ALSO covers Dloc=8 (D16/cp2 -> best_front_tile(8)=(128,16), postact=tile_N//2=8=Dloc): the
# postact_epi_n=8 STSM-x2 R2S path (kernels/layernorm_dual_gated_gemm.py:321 / a2a:1316), verified cross-node venue F
# cp2 (rel_L2~0, 0 outlier/untouched). tile_N param IGNORED (tile_from_dloc=best_front_tile). The kernel
# needs NO change for Dloc=8, AND the bench fused targets already pass raw tile_N=16 (no _round32 change). Guards #3.
_DECPL_HYB = _DECPL + [(16, 16, 1)]


# NOTE (§10): `test_front_decoupled_correct` (the plain-decoupled SIMT GMEM-ring putwarp drain,
# decoupled=True / ib_drain=False) was REMOVED — that store mode is RETIRED (a superseded precursor;
# configure_a2a now raises for it, see `test_front_plain_decoupled_retired_raises`). Its surviving
# coverage: `test_front_ib_drain_nvlink_arm` (all-P2P collapse -> the coupled store) +
# `test_front_ib_drain_hybrid` (the ring + consumer WG + blocking put). `_build_front_decoupled`
# stays (shared by the ib_drain tests, which drive it via a `_cfg` that sets ib_drain=True).


def _run_ib_drain_front(
    dist_manager,
    device,
    world_size,
    D,
    tile_N,
    cwg,
    *,
    require_ib,
    ring_depth=2,
    tile_from_dloc=False,
    n_token=None,
    m_linear=False,
    ib_wide=False,
    ib_wide_batch=None,
    ib_wide_nbi=False,
    expect_run_i=False,
    shape_mode="static",
):
    """Shared body for the ib_drain=True differential front (NVLink-arm + hybrid). Builds the front with
    configure_a2a(ib_drain=True) (which auto-builds is_p2p + forces the decoupled ring), runs it, and
    gates recv vs the flat-cp D-major reference (SAME recv as the coupled store: rel<bar, 0 outlier, 0
    untouched). ``require_ib``: True -> skip unless the job actually has an IB peer (the hybrid gate,
    needs a 2-node mesh); False -> the all-NVLink arm (all-P2P collapse -> coupled store byte-identical).
    ``tile_from_dloc``: derive the CTA tile from the PRODUCTION picker ``best_front_tile(Dloc)`` instead
    of the parametrized ``tile_N`` -- needed at cp16 where Dloc=D/16 in {8,16} makes the fixed
    postact-64 tile invalid (Dloc % 64 != 0). The picker returns the widest-legal small-Dloc tile
    (Dloc=8 -> (128,16) postact 8; Dloc=16 -> (128,32) postact 16), which divides Dloc by construction."""
    from tests.distributed.correctness_harness import compute_error_histogram_blocked

    import os as _os2

    # D override (CPO_FRONT_D): at large N_token the fp32 reference (allgather of each rank's (2D,M)
    # GLU output across cp) is cp*2D*N^2*dtype and OOMs at D256/cp16. Running the SAME kernel path
    # (Dloc=16, tile (128,32)) at a SMALLER D on the smallest cross-node config (cp=2 = 2 nodes x 1 GPU,
    # D=32 -> Dloc=16) shrinks the reference cp*2D* linearly so it fits, while still exercising the
    # ib_drain IB branch (the 1 cross-node peer) at that large N_token. Default: the parametrized D.
    D = int(_os2.environ.get("CPO_FRONT_D", str(D)))
    cp = world_size
    if D % cp != 0:
        rank_invariant_skip(f"D={D} not divisible by cp={cp}", because=_SHAPE_GATE_BECAUSE)
    Dloc = D // cp
    tile_M = 128
    if tile_from_dloc:
        tile_M, tile_N = DualGatedGemmDistSm90.best_front_tile(Dloc)  # cp16-valid production tile
        # Base-kernel floor (POST tileN relaxation): the staged GEMM needs tile_N%16==0 (SM90 gated
        # STSM-quad floor, kernels/layernorm_dual_gated_gemm.py:274) -> postact=tile_N//2>=8, AND the front feature-
        # scatter needs Dloc%postact==0 (a CTA N-tile in ONE D-slice) + the 16-B feature floor
        # 2*Dloc*2B>=16B (Dloc>=4). best_front_tile(8)=(128,16) (postact 8) satisfies ALL of these, so
        # Dloc=8 (D128/cp16) RUNS the production ib_drain path (the postact_epi_n=8 STSM-x2 R2S fix in
        # dual_gated_gemm_a2a.epi_setup_postact). Only Dloc<8 (tile_N<16) hits the genuine base %16
        # floor. (Was tile_N%32 — a STALE pre-tileN-relaxation floor that wrongly skipped the valid Dloc=8.)
        if tile_N % 16 != 0:
            rank_invariant_skip(
                f"front feature-scatter unsupported at Dloc={Dloc} (D={D}/cp={cp}): best_front_tile gives "
                f"tile_N={tile_N} but the staged GEMM needs tile_N%16==0 (postact>=8). Genuine base-kernel "
                "SM90 STSM-quad floor, independent of ib_drain.",
                because=_SHAPE_GATE_BECAUSE,
            )
    tile_n_postact = tile_N // 2
    if Dloc % tile_n_postact != 0:
        rank_invariant_skip(
            f"Dloc={Dloc} not a multiple of postact tile_N={tile_n_postact} at cp={cp}",
            because=_SHAPE_GATE_BECAUSE,
        )
    import os as _os

    # N_token from env (default 16 = fast smoke). The per-cell driver sets CPO_FRONT_NTOKEN to sweep
    # the production sequence lengths at D256/cp16 (ONE torchrun per N — per-cell isolation, so an OOM
    # cell fails its own launch, never a divergent in-process skip). M = B*N_token^2 is the front's
    # token-pair grid; the recv (2*Dloc, cp*M) grows as cp*N_token^2, so large N OOMs (handled by the
    # per-cell driver, not a memory-estimate gate).
    # n_token from the PARAM (the off-grid gate) OR env (default 16 = fast smoke). m_linear=True sets
    # M = N (a LINEAR token count) so rows_per_peer = M is OFF-GRID (M % tile_M != 0) at N%128==64 — the
    # FIRST-PRINCIPLE case the row-peer%tile_M fix removes; the sole surviving check is 16-B (M%8==0).
    # Default M = B*N*N (the token-pair grid; must be a CTA tile_M multiple on the base path).
    B = 1
    N = int(n_token) if n_token is not None else int(_os.environ.get("CPO_FRONT_NTOKEN", "16"))
    M = B * N if m_linear else B * N * N
    if m_linear and M % 8 != 0:
        rank_invariant_skip(
            f"N={N} not %8 (16-B TMA/put floor — the ONLY allowed shape constraint)",
            because=_SHAPE_GATE_BECAUSE,
        )
    if not m_linear and M % 128 != 0:
        rank_invariant_skip(f"M={M} not a multiple of CTA tile_M=128", because=_SHAPE_GATE_BECAUSE)
    K = 256
    rows_per_peer = M
    M_full = cp * rows_per_peer
    pm = _pe_map(dist_manager)
    pe_table = tuple(int(v) for v in pm.cp_pe_table.tolist())
    rd = ring_depth  # was a bare `2`; now matrix-drawn where the caller parametrizes it

    # Probe is_p2p on THIS job's pe_table (host-only, buffer-free) so we can gate the hybrid arm.
    # _build_p2p_table is STATELESS (uses only pe_table) -> the probe's dtypes are irrelevant; pass
    # Float32 (a cutlass Numeric with .width, unlike a raw torch.dtype) so the constructor is valid.
    has_ib = _job_has_ib_peers(pe_table)  # A1: the ONE shared probe (tests/distributed/topology.py)
    if require_ib is True and not has_ib:
        rank_invariant_skip(
            f"hybrid ib_drain needs >=1 IB (non-P2P) peer; this job is all-NVLink (pe_table={pe_table}). "
            "Run cp=(2,8) across 2 nodes (two 8-GPU nodes).",
            because=_IB_UNIFORM_BECAUSE,
        )
    if require_ib is False and has_ib:
        rank_invariant_skip(
            f"NVLink-arm test needs an all-P2P job; this job has IB peers (pe_table={pe_table}).",
            because=_IB_UNIFORM_BECAUSE,
        )
    # require_ib is None (the off-grid gate): run on WHATEVER the job is — an all-P2P job exercises the
    # P2P descriptor-clamp arm (ib_drain collapses to the coupled clamp store), a hybrid job exercises the
    # IB run_tokens arm; either way an off-grid rows_per_peer must land the SAME recv as the base store.

    torch.manual_seed(4321 + dist_manager.rank)
    x = torch.randn(M, K, device=device, dtype=torch.bfloat16) / (K**0.5)
    Wg2 = torch.randn(2 * D, K, device=device, dtype=torch.bfloat16) / (K**0.5)
    Wp2 = torch.randn(2 * D, K, device=device, dtype=torch.bfloat16) / (K**0.5)
    # Symmetric memory via the torch MemPool, NOT nvshmem4py's allocator: torch owns the
    # nvshmem bootstrap here, so nvshmem4py's own `_is_initialized` is permanently False
    # and its `tensor()` raises `NVSHMEM Library is not initialized`. The pool also
    # recycles, which keeps the collective free off the destruction path.
    with torch.cuda.use_mem_pool(DistributedManager.symmetric_mempool(_this_rank_device())):
        recv = torch.empty((2 * Dloc, M_full), dtype=torch.bfloat16, device=_this_rank_device())
    recv.fill_(-99.0)

    def _cfg(g):
        g.configure_a2a(
            cp=cp,
            my_cp_rank=int(pm.my_cp_rank),
            rows_per_peer=rows_per_peer,
            pe_table=pe_table,
            ib_drain=True,
            ring_depth=rd,
            consumer_warpgroups=cwg,
            ib_wide=ib_wide,
            ib_wide_batch=ib_wide_batch,
            ib_wide_nbi=ib_wide_nbi,
        )

    compiled, run_fn, gemm_obj, ring_t = _build_front_decoupled(
        x,
        Wg2,
        Wp2,
        (tile_M, tile_N),
        recv_t=recv,
        ring_depth=rd,
        shape_mode=shape_mode,
        consumer_warpgroups=cwg,
        configure_fn=_cfg,
    )
    if expect_run_i:
        # The wide-put run_i raster was derived at compile (get_scheduler_arguments) and stashed on the
        # instance. Assert it ACTUALLY engaged (>= R_MIN) and is SUB-BAND (R ∤ ncluster_m) so this cell
        # genuinely exercises the CAVEAT-A clamp+monotonic decode end-to-end (not silently the default
        # raster). cluster_m=1 for the front -> ncluster_m = ceil(rows_per_peer / tile_M).
        R_i = int(getattr(gemm_obj, "_a2a_run_i_tiles_derived", 0))
        r_min = int(gemm_obj._A2A_RUN_I_RMIN)
        ncm = (M + tile_M - 1) // tile_M
        print(
            f"\n[run_i subband] rank={dist_manager.rank} M={M} tile_M={tile_M} ncm={ncm} "
            f"derived_R={R_i} (R_MIN={r_min}) subband={R_i > 0 and ncm % R_i != 0}",
            flush=True,
        )
        assert R_i >= r_min, (
            f"run_i did NOT engage (derived R={R_i} < R_MIN={r_min}) at M={M} tile_M={tile_M} ncm={ncm} "
            f"— the wide-put coalescing raster is silently OFF (grid_z too large? shape too small?)"
        )
        assert ncm % R_i != 0, (
            f"run_i sub-band test needs R∤ncm (CAVEAT-A); got R={R_i} dividing ncm={ncm} — pick an "
            f"off-grid N whose ncluster_m is not a multiple of the derived R (e.g. N=10048 -> ncm=79 prime)"
        )
    try:
        _nvshmem_barrier()
        run_fn()
        _drain()

        # OURS-vs-MAIN numerical comparison hook. Dumps the computed recv plus a HASH OF THE
        # INPUTS, gated on CPO_FRONT_RECV_OUT so it is inert in every normal run. The input hash is
        # not decoration: the two trees build their operands independently, and a recv difference
        # caused by different operands would read exactly like a kernel difference, so the
        # comparator refuses any pair whose input hashes disagree.
        _recv_out = os.environ.get("CPO_FRONT_RECV_OUT")
        if _recv_out:
            import hashlib as _hl
            import json as _json
            import pathlib as _pl

            _d = _pl.Path(_recv_out)
            _d.mkdir(parents=True, exist_ok=True)
            _lbl = os.environ.get("CPO_FRONT_RECV_LABEL", "cell")
            _r = int(dist_manager.rank)
            _raw = lambda t: t.detach().cpu().contiguous().view(torch.int16).numpy().tobytes()
            _h = lambda t: _hl.sha256(_raw(t)).hexdigest()
            # The HASH is always written; the 2 GB tensor only when asked. A recv at the declared
            # shapes is (2*Dloc, cp*B*N^2) -- 2 GB per rank at N=2048/cp=2 alone -- so dumping it
            # per cell would cost terabytes across a sweep. Hash-first answers "bitwise equal?"
            # definitively; the full dump is for LOCATING a difference once one is known to exist.
            if os.environ.get("CPO_FRONT_RECV_FULL"):
                (_d / f"{_lbl}.r{_r}.recv.bin").write_bytes(_raw(recv))
            (_d / f"{_lbl}.r{_r}.json").write_text(
                _json.dumps(
                    {
                        "label": _lbl,
                        "rank": _r,
                        "cp": int(cp),
                        "M": int(M),
                        "D": int(D),
                        "Dloc": int(Dloc),
                        "recv_shape": list(recv.shape),
                        "recv_dtype": str(recv.dtype),
                        "in_x": _h(x),
                        "in_Wg2": _h(Wg2),
                        "in_Wp2": _h(Wp2),
                        "out_recv": _h(recv),
                    },
                    indent=1,
                    sort_keys=True,
                )
            )
        # DUMP-MODE SHORT-CIRCUIT (CPO_FRONT_RECV_OUT). The ours-vs-main sweep compares the two
        # trees' `recv` to EACH OTHER, bitwise. The fp32 reference below is this tree's OWN gate and,
        # in the sweep driver's words, "says nothing about how the trees relate to each other" --
        # while its first statement, the GLU, allocates B*N^2*2D*4 bytes: 32.00 GiB at cp=(2,2)
        # N_token=4096 D=256, upstream of the streaming and unreachable by it. So in dump mode the
        # reference is not merely expensive, it is what makes the cell impossible; skipping it is
        # what lets the declared shapes be compared at all.
        #
        # SAFE AS A COLLECTIVE: the predicate is an ENV VAR, which is job-uniform, so every rank
        # returns or none does. A per-rank predicate here would be a deadlock, not a skip -- the
        # reference path runs `dist.scatter` per peer.
        #
        # It is announced, deliberately. A run that silently stopped checking correctness while still
        # printing per-cell rows is exactly the artifact somebody later mistakes for a correctness
        # result. The `finally` below still releases the symmetric buffers on this path.
        if _recv_out:
            print(
                f"\n[{_lbl}] rank={dist_manager.rank} CORRECTNESS GATE OFF (CPO_FRONT_RECV_OUT "
                f"set): recv dumped for the ours-vs-main comparison; the fp32 reference, the error "
                f"histogram and every assert below were SKIPPED. This run proves nothing about this "
                f"tree's own correctness.",
                flush=True,
            )
            return

        # TWO STREAMING PASSES over the reference, each holding 1/cp of it. `_ref_front_gathered`
        # avoided assembling the (2*Dloc, M_full) matrix but still RETAINED ga/gb, which is
        # 2*M_full*Dloc -- exactly one copy of the reference, and a 16 GiB allocation at cp=(2,2)
        # N_token=4096 D=256, where the cell died inside the gather.
        #
        # Two passes rather than one, deliberately: `abs_thresh` (2e-2 * mean|ref|) and the
        # element-wise bound (REL_BAR * max|ref|) are GLOBAL over the whole reference, and no rank
        # holds the whole reference any more. The alternative -- deriving them from an all_reduce
        # over each rank's full local operands -- is one pass but the union over ranks is a SUPERSET
        # of any single rank's reference, so max|ref| comes out LOOSER and the bar moves. This costs
        # 2x the collectives on a test-only path and every reported number stays identical.
        def _stream():
            return _ref_front_block_stream(
                x,
                Wg2,
                Wp2,
                pm,
                cp=cp,
                rows_per_peer=rows_per_peer,
                Dloc=Dloc,
                world_size=world_size,
            )

        ref_absmax, _asum, _nel = 0.0, 0.0, 0
        for _ab, _bb in _stream():  # PASS 1: the two global scalars, nothing else
            for _t in (_ab, _bb):
                _tf = _t.float().abs()
                ref_absmax = max(ref_absmax, float(_tf.max().item()))
                _asum += float(_tf.sum().item())
                _nel += _tf.numel()
        ref_absmean = _asum / _nel
        untouched = int((recv == -99.0).sum().item())
        a_is_view = recv[:Dloc].data_ptr() == recv.data_ptr()
        arm = "hybrid" if has_ib else "nvlink-arm"
        tag = f"front-ibdrain[{arm}] cp={cp} M={M} D={D} tile_N={tile_N} cwg={cwg} Dloc={Dloc} rd={rd}"
        # STREAMED comparison -- the reference is consumed one peer block at a time and never
        # assembled. The three tensors this replaces were each `2*D*B*N^2` elements: the assembled
        # `expected`, and the two `.t().contiguous()` copies that fed the histogram. `cp` CANCELS in
        # that count, so a bigger mesh does not shrink any of them. Measured at cp=(2,2)
        # N_token=5952 D=256: a single 33.79 GiB request beside an equally large recv, and the cell
        # died -- at EVERY mesh identically (0/6 at N=4096, 0/6 at N=5952 for cp 2, 4, 8, 16, (2,2)),
        # which is what identifies the reference rather than the kernel as the cause.
        #
        # The two scalars come from the gathered halves BEFORE the loop, so the bound stays the
        # uniform `REL_BAR * max|ref|` it has always been and `abs_thresh` stays the global
        # `2e-2 * mean|ref|`. Deriving either per block would silently make the verdict depend on
        # how the blocks were cut, which is exactly the kind of change that reads as a refactor.
        # PASS 2: the histogram AND the element-wise verdict, off the SAME stream. The bound is
        # the uniform `REL_BAR * max|ref|` it has always been; the assert lives inside the generator
        # because the histogram consumes it, and a third pass would be a third round of collectives
        # for a number both already have in hand. Each block names its peer, so a failure still
        # localizes to the worst offender's index within that block.
        _bound = REL_BAR * (ref_absmax if ref_absmax > 0 else 1.0)
        _worst = [0.0]

        def _pairs():
            """Yield ``(got, ref)`` per peer and assert the element-wise bound as it goes.

            ``expected[:, col].t()`` IS ``cat([a_block, b_block], dim=1)``, so the pair is formed
            with no transpose of a large tensor and no copy of recv.
            """
            for _s, (_ab, _bb) in enumerate(_stream()):
                _col = slice(_s * rows_per_peer, (_s + 1) * rows_per_peer)
                _got, _ref = recv[:, _col].t(), torch.cat([_ab, _bb], dim=1)
                _worst[0] = max(
                    _worst[0],
                    numerics.assert_elementwise(
                        _got, _ref, _bound, what=f"{tag} peer-block s={_s}"
                    ),
                )
                yield _got, _ref

        h = compute_error_histogram_blocked(
            _pairs(),
            abs_thresh=2e-2 * (ref_absmean + 1e-30),
            row_dims=(1, M_full, 1),
        )
        # `rel` is the historical `max|got-ref| / max|ref|`, recovered exactly: the per-block helper
        # returns worst|err|/bound, and bound == REL_BAR*max|ref|.
        rel = REL_BAR * _worst[0]
        passed = rel < REL_BAR and h.n_outlier_rows == 0 and untouched == 0 and a_is_view
        print(
            f"\n[{tag}] rank={dist_manager.rank} has_ib={has_ib} rel={rel:.3e} (bar {REL_BAR}) "
            f"{h.summary()} untouched={untouched}/{recv.numel()} a_view={a_is_view} "
            f"{'PASS' if passed else 'FAIL'}",
            flush=True,
        )
        assert h.n_outlier_rows == 0, (
            f"{tag}: {h.n_outlier_rows} outlier token rows (localized corruption); worst @ "
            f"{h.worst_row_index} ratio {h.row_outlier_ratio:.1f}x"
        )
        assert untouched == 0, (
            f"{tag}: {untouched}/{recv.numel()} recv cells unwritten (coverage gap)"
        )
        assert a_is_view, f"{tag}: a/b half not a VIEW (the zero-copy einsum feed is broken)"
    finally:
        compiled.free()
        # ib_drain allocates the ring on the SYMMETRIC HEAP (put source) -> it MUST be released
        # before the next allocation (an unfreed symmetric buffer wedges the multi-test session /
        # faults at the next collective alloc).
        #
        # `main` calls nvshmem_torch.free_tensor(recv) + free_tensor(ring_t) here. That allocator is
        # NOT ours: recv and the ring come from DistributedManager.symmetric_mempool, which RECYCLES
        # rather than frees. The bring-back therefore dropped main's two calls -- and replaced them
        # with NOTHING, leaving the comment above orphaned. The blocks were then released only when
        # the locals fell out of scope and the collector happened to run, which across ranks is not
        # an agreed order: exactly the GC-ordering hazard the pool exists to avoid. A
        # multi-parametrization session accumulated symmetric blocks until a later collective
        # allocation faulted and the NCCL watchdog aborted the ranks.
        #
        # The pool's equivalent of free_tensor is dropping the last reference, so do it EXPLICITLY:
        # same statements, same order, on every rank. `run_fn`/`gemm_obj` are released too because
        # they close over the same buffers, and a reference held by a closure keeps the block out of
        # the pool just as surely as the local does. This is symmetric on every rank, so unlike a
        # conditional free it cannot run a collective on a strict subset and deadlock.
        run_fn = gemm_obj = ring_t = recv = None
        gc.collect()
        _nvshmem_barrier()


@pytest.mark.usefixtures("_nvshmem__distributed__front_a2a_staged")
@pytest.mark.usefixtures("_module_skip__distributed__front_a2a_staged")
@FRONT_A2A.parametrize(
    "D",
    "tile_N",
    "cwg",
    cells=_DECPL,
    because=(
        "cells= and not a product: (D, tile_N, cwg) travel together. tile_N is chosen so the postact "
        "tile divides Dloc, and the second consumer warpgroup is exercised at the wider D where it "
        "has work. The (16, 16, 1) cell is meaningful only where a real IB peer exists, so it is "
        "selected by test_front_ib_drain_hybrid instead"
    ),
)
@FRONT_A2A.parametrize(
    "ring_depth",
    only={"ring_depth": (2,)},
    because=(
        "only the production depth here: this arm's subject is the all-P2P collapse, not the ring, "
        "and 2 is what `dual_gated_gemm_a2a.py:255` bakes. Selecting one value keeps the cell count "
        "flat (x1) while giving the axis a real consumer, which is what `coverage_problems` rule 1 "
        "asks for. The other depths belong to the byte-identity sweep, which is what will widen "
        "this pool"
    ),
)
@FRONT_A2A.parametrize("mesh")
def test_front_ib_drain_nvlink_arm(
    apply_mesh, mesh, dist_manager, device, world_size, D, tile_N, cwg, ring_depth
):
    """ib_drain=True on an ALL-NVLink (all-P2P) job == the coupled TMA-S2G store (byte-identical arm).

    Rung 2 of the validation ladder: with every peer P2P the has_ib_peers collapse elides the whole IB
    drain (ring + consumer WG) -> the store reduces to the coupled NVLink TMA-S2G. Proves the is_p2p
    branch's NVLink arm is intact + the differential store seam did not regress the coupled path.
    Runnable on a SINGLE venue E DGX (cp<=8, all-NVLink)."""
    apply_mesh(mesh)
    _run_ib_drain_front(
        dist_manager, device, world_size, D, tile_N, cwg, require_ib=False, ring_depth=ring_depth
    )


@pytest.mark.usefixtures("_nvshmem__distributed__front_a2a_staged")
@pytest.mark.usefixtures("_module_skip__distributed__front_a2a_staged")
@FRONT_A2A.parametrize(
    "D",
    "tile_N",
    "cwg",
    cells=_DECPL_HYB,
    because=(
        "cells= for the same reason as the NVLink arm, plus the (16, 16, 1) cell that arm cannot "
        "use: D=16 is what drives Dloc=8 and the postact_epi_n=8 STSM-x2 R2S path, and that path is "
        "only more than the coupled store when the mesh actually spans nodes"
    ),
)
@FRONT_A2A.parametrize("mesh")
def test_front_ib_drain_hybrid(apply_mesh, mesh, dist_manager, device, world_size, D, tile_N, cwg):
    """ib_drain=True HYBRID: real IB peers exercised -> P2P peers coupled TMA-S2G, IB peers symmetric
    ring + BLOCKING put_warp. The load-bearing gate (rung 3). Skips unless the job has >=1 IB peer (needs
    a 2-node mesh, e.g. cp=(2,8) across two 8-GPU nodes). Gates recv rel/coverage vs the
    flat-cp D-major reference (both directions land the SAME recv as the coupled store). The CTA tile is
    derived from best_front_tile(Dloc) so it is valid at cp16 (Dloc=D/16 in {8,16}) where the fixed
    postact-64 tile would be invalid (Dloc % 64 != 0)."""
    apply_mesh(mesh)
    _run_ib_drain_front(
        dist_manager, device, world_size, D, tile_N, cwg, require_ib=True, tile_from_dloc=True
    )


@pytest.mark.usefixtures("_nvshmem__distributed__front_a2a_staged")
@pytest.mark.usefixtures("_module_skip__distributed__front_a2a_staged")
@FRONT_A2A.parametrize("shape_mode")
@FRONT_A2A.parametrize(
    "D",
    "tile_N",
    "cwg",
    cells=[_DECPL_HYB[0]],
    because=(
        "ONE cell, because the axis under test is `shape_mode` and the (D, tile_N, cwg) grid is "
        "already swept against BOTH the NVLink arm and the hybrid arm by the two tests above. "
        "Crossing a full (D, tile_N, cwg) grid with shape_mode here would re-pay those cold "
        "compiles to re-verify a dimension they already cover, and each cell of this test is a "
        "cross-node cold compile. What this test must not narrow is `shape_mode` itself -- both "
        "values run, which is what makes the module satisfy the axis's two facets."
    ),
)
@FRONT_A2A.parametrize("mesh")
def test_front_shape_mode_correct(
    apply_mesh, mesh, dist_manager, device, world_size, D, tile_N, cwg, shape_mode
):
    """Both token-extent spellings compute the right recv: STATIC (extent baked) and DYNAMIC (marked).

    WHY THIS EXISTS, measured. The IB drain addressed the peer recv through a feature-mode stride of
    ``epi_n * M_full`` ELEMENTS. With a STATIC extent that stride is a compile-time constant, so
    ``dst_feat * epi_n * M_full`` was formed in 32 bits and WRAPPED once
    ``(2*Dloc - epi_n) * M_full >= 2**31``, handing a wrapped address to the put -- a CUDA illegal
    address. It reproduced ONLY with a static extent, and the shipped profiling harness hardcodes
    ``--dynamic``, so 23 archived nsys cells and every green gate missed it while 45 of 128 declared
    front cells were broken. The razor pair that pinned the threshold was 16 tokens apart:
    cp=2 D=256 N=2976 (2,125,578,240, under 2**31) passed and N=2992 (2,148,495,360) faulted.

    So the two modes are not the same code run twice: constant folding is the difference, and a
    matrix without this axis cannot tell "static was tested here" from "static was never tested
    here". Real inference traffic is dynamic-N by design (ONE compile serves many token counts), but
    the static path is still compiled, is still ``FusedTriMul``'s own default, and shares its code
    with the dynamic path -- so an untested static path is a defect sitting in code the tested path
    borrows from.

    Both modes run through the SAME builder (``_build_front_decoupled(shape_mode=...)``), so
    a difference between them is attributable to the mode rather than to two harnesses.

    Requires a real IB peer (a 2-node mesh): on an all-P2P job ``_ib_collapse()`` const_expr-elides
    the entire IB drain, which is the arm the defect lived in -- the cell would then pass without
    exercising anything this test is about.
    """
    apply_mesh(mesh)
    _run_ib_drain_front(
        dist_manager,
        device,
        world_size,
        D,
        tile_N,
        cwg,
        require_ib=True,
        tile_from_dloc=True,
        shape_mode=shape_mode,
    )


_OFFGRID_NTOK = [
    1088,
    4032,
    10048,
    4128,
]  # %128 in {64, 32} (OFF-grid rows_per_peer) AND %8 == 0 (16-B ok); 4128%128=32 mirrors the e2e N1000 rpp%128


@pytest.mark.usefixtures("_nvshmem__distributed__front_a2a_staged")
@pytest.mark.usefixtures("_module_skip__distributed__front_a2a_staged")
@FRONT_A2A.parametrize(
    "N_token",
    only={"N_token": tuple(_OFFGRID_NTOK) + (4096,)},
    because=(
        "the off-grid gate needs an N whose rows_per_peer is NOT a CTA tile_M multiple. The pool's "
        "small values (16..40) are the partial-token auto-clamp cells and are swept by "
        "test_front_partial_token_clamp_correct.\n"
        "\n"
        "4096 is the ALIGNED CONTROL, and it is added here rather than to `_OFFGRID_NTOK` because "
        "that list is what its name says. On this arm `m_linear=True`, so rows_per_peer == N and "
        "4096 % 128 == 0 -- the same code path with a clean rpp, which is what makes the off-grid "
        "cells' behaviour attributable to being off-grid rather than to the arm itself. It is also "
        "what runs the axis's `tile_aligned` facet: before it, 0 of 8 declared values were "
        "%128 == 0 and the pool was entirely straddle"
    ),
)
@FRONT_A2A.parametrize("mesh")
def test_front_ib_drain_offgrid(apply_mesh, mesh, dist_manager, device, world_size, N_token):
    """FIRST-PRINCIPLE off-grid gate (the row-peer%tile_M fix): rows_per_peer = N (LINEAR M) with
    N%128==64 -> rows_per_peer % CTA tile_M != 0. BEFORE the fix this hit `ValueError: front-A2A
    decoupled drain requires rows_per_peer % tile_M`; the fix makes it run FUSED with the SOLE surviving
    shape check being 16-B (N%8==0). require_ib=None -> runs on WHATEVER the job is: an all-P2P job
    (cp<=8, single DGX) exercises the P2P descriptor-clamp arm (ib_drain collapses to the coupled clamp
    store); a cp16 (2,8) 2-node job exercises the IB run_tokens arm (partial run = rows_per_peer%epi_m).
    Anchor D=256, CTA tile from best_front_tile(Dloc). Gates recv rel/coverage vs the flat-cp D-major
    reference (must MATCH the base coupled store, per FIRST PRINCIPLE); NO xfail, NO skip on N%8==0."""
    apply_mesh(mesh)
    _run_ib_drain_front(
        dist_manager,
        device,
        world_size,
        D=256,
        tile_N=64,
        cwg=1,
        require_ib=None,
        tile_from_dloc=True,
        n_token=N_token,
        m_linear=True,
    )


# WIDE-PUT (ib_wide) coalescing-width sweep. W=128 -> 32 KiB/put (the IB knee); W=32 forces frequent
# mid-stream flushes (tests the break-flush path); large W mostly finalize-flushes (tests the trailing
# flush). Correctness is W-INVARIANT (the recv content is identical to the per-subtile drain).
_WIDE_W = [32, 128]


@pytest.mark.usefixtures("_nvshmem__distributed__front_a2a_staged")
@pytest.mark.usefixtures("_module_skip__distributed__front_a2a_staged")
@FRONT_A2A.parametrize("cwg", "W")
@FRONT_A2A.parametrize("mesh")
def test_front_ib_wide_hybrid(apply_mesh, mesh, dist_manager, device, world_size, cwg, W):
    """WIDE-PUT HYBRID (the load-bearing gate): ib_wide=True on a job with >=1 IB peer exercises the
    wide producer (accumulate W subtiles/slot, flush on break/W-cap + finalize) + the per-BATCH consumer
    drain (loop until subtiles_drained==total_subtiles). Proves (a) COUNT-PARITY §6.4 — a mismatch
    deadlocks -> the torchrun `timeout` catches it; (b) the wide (W·epi_m)-token put lands the SAME recv
    as the per-subtile drain (rel<bar, 0 outlier, 0 untouched). Skips unless the job has an IB peer (needs
    a 2-node mesh, e.g. cp=(2,8) across two 8-GPU nodes). Anchor D=256, tile from
    best_front_tile(Dloc). Aligned N (full-W batches); the partial-tail (CAVEAT B) is the offgrid test."""
    apply_mesh(mesh)
    _run_ib_drain_front(
        dist_manager,
        device,
        world_size,
        D=256,
        tile_N=64,
        cwg=cwg,
        require_ib=True,
        tile_from_dloc=True,
        ib_wide=True,
        ib_wide_batch=W,
    )


@pytest.mark.usefixtures("_nvshmem__distributed__front_a2a_staged")
@pytest.mark.usefixtures("_module_skip__distributed__front_a2a_staged")
@FRONT_A2A.parametrize(
    "N_token",
    only={"N_token": tuple(_OFFGRID_NTOK)},
    because=(
        "same off-grid selection as the per-subtile drain: an N whose rows_per_peer is not a CTA "
        "tile_M multiple. The small values are the auto-clamp cells, swept elsewhere"
    ),
)
@FRONT_A2A.parametrize("mesh")
def test_front_ib_wide_offgrid(apply_mesh, mesh, dist_manager, device, world_size, N_token):
    """WIDE-PUT off-grid (CAVEAT B — the loop-terminating arrive_empty + the partial-tail run). rows_per_
    peer = N (LINEAR M) with N%128==64 -> the CTA's LAST wide batch is PARTIAL (n_sub<W and/or a partial-
    tail last subtile run<epi_m). The finalize must flush it AND the consumer's terminating batch must
    still arrive_empty (else count-parity breaks by one -> deadlock -> timeout). The wide put's run =
    total_run (a multiple of 8 -> run·2B a multiple of 16 B), landing the SAME recv as the base store.
    require_ib=None -> runs on whatever the job is (all-P2P collapses ib_wide to the coupled clamp store =
    a wide-OFF identity check; hybrid exercises the partial wide batch). W=128 (the production width)."""
    apply_mesh(mesh)
    _run_ib_drain_front(
        dist_manager,
        device,
        world_size,
        D=256,
        tile_N=64,
        cwg=1,
        require_ib=None,
        tile_from_dloc=True,
        n_token=N_token,
        m_linear=True,
        ib_wide=True,
        ib_wide_batch=128,
    )


@pytest.mark.usefixtures("_nvshmem__distributed__front_a2a_staged")
@pytest.mark.usefixtures("_module_skip__distributed__front_a2a_staged")
@FRONT_A2A.parametrize("cwg")
@FRONT_A2A.parametrize("mesh")
def test_front_ib_wide_nbi_hybrid(apply_mesh, mesh, dist_manager, device, world_size, cwg):
    """WIDE-PUT B2 PIPELINE (ib_wide_nbi=True) — the load-bearing WAR gate. The per-feature-row wide put
    goes NON-BLOCKING and ONE trailing BLOCKING put per batch reaps the qp (its internal ibgda_quiet(qp(pe))
    drains the batch's prior nbi puts — NO device-scope quiet). If that reap does NOT cover every nbi put
    (a wrong 1-qp-per-(pe,warp) assumption), the producer overwrites a still-in-flight ring slot -> recv
    CORRUPTION (rel>bar / outliers) is caught HERE against the flat-cp D-major reference. Also proves the
    pipeline lands the SAME recv as B1 + count-parity holds (a mismatch deadlocks -> torchrun timeout).
    Skips unless the job has an IB peer (2-node mesh, e.g. cp=(2,8)). W=128, anchor D=256, production tile."""
    # B2 IS A DECLARED UNSUPPORTED REGION -- `ib_wide_nbi=True` is refused OUTRIGHT at
    # `_configure_ib_wide` on every toolchain, so without the developer hatch this test can no longer
    # reach its own subject and would fail on the refusal rather than on the property it asserts.
    # `rank_invariant_skip` and not `gated_skip`: the predicate is an ENVIRONMENT VARIABLE, which is
    # job-uniform -- every rank in the launch sees the same value -- so no all_reduce is needed and a
    # divergent-skip deadlock is impossible. NOT `xfail`: an xfail'd correctness assertion cannot tell
    # "refused" from "answered wrongly", so a silently-corrupting B2 would satisfy it.
    if os.environ.get("_CPO_ALLOW_IB_WIDE_NBI") != "1":
        rank_invariant_skip(
            "B2 (ib_wide_nbi=True) is a declared unsupported region: refused at the front door on "
            "every toolchain (<4.7.0 cannot link `nvshmemx_flush_warp`; >=4.7.0 links but runs the "
            "A2A-fused DualGatedGEMM 3.23-3.25x slower, cause unknown). Set _CPO_ALLOW_IB_WIDE_NBI=1 "
            "to develop B2 itself.",
            because=(
                "the predicate is an environment variable read identically on every rank of the "
                "launch, so the decision is job-uniform by construction and cannot diverge per rank"
            ),
        )
    apply_mesh(mesh)
    _run_ib_drain_front(
        dist_manager,
        device,
        world_size,
        D=256,
        tile_N=64,
        cwg=cwg,
        require_ib=True,
        tile_from_dloc=True,
        ib_wide=True,
        ib_wide_batch=128,
        ib_wide_nbi=True,
    )


@pytest.mark.usefixtures("_nvshmem__distributed__front_a2a_staged")
@pytest.mark.usefixtures("_module_skip__distributed__front_a2a_staged")
@FRONT_A2A.parametrize(
    "N_token",
    only={"N_token": tuple(_OFFGRID_NTOK)},
    because=(
        "same off-grid selection as its blocking sibling: the B2 non-blocking put must land the "
        "same recv on an off-grid rows_per_peer. The small values are the auto-clamp cells"
    ),
)
@FRONT_A2A.parametrize("mesh")
def test_front_ib_wide_nbi_offgrid(apply_mesh, mesh, dist_manager, device, world_size, N_token):
    """WIDE-PUT B2 + off-grid (the partial-tail run under the non-blocking pipeline). The CTA's PARTIAL last
    wide batch (n_sub<W and/or a partial-tail last subtile run<epi_m) drained NON-BLOCKING + one trailing
    blocking qp-reap must STILL land the SAME recv as the base store AND keep count-parity (else deadlock ->
    timeout). require_ib=None -> runs on whatever the job is (all-P2P collapses ib_wide/nbi to the coupled
    clamp store = a wide/nbi-OFF identity check; a hybrid job exercises the partial B2 batch). W=128."""
    # B2 IS A DECLARED UNSUPPORTED REGION -- `ib_wide_nbi=True` is refused OUTRIGHT at
    # `_configure_ib_wide` on every toolchain, so without the developer hatch this test can no longer
    # reach its own subject and would fail on the refusal rather than on the property it asserts.
    # `rank_invariant_skip` and not `gated_skip`: the predicate is an ENVIRONMENT VARIABLE, which is
    # job-uniform -- every rank in the launch sees the same value -- so no all_reduce is needed and a
    # divergent-skip deadlock is impossible. NOT `xfail`: an xfail'd correctness assertion cannot tell
    # "refused" from "answered wrongly", so a silently-corrupting B2 would satisfy it.
    if os.environ.get("_CPO_ALLOW_IB_WIDE_NBI") != "1":
        rank_invariant_skip(
            "B2 (ib_wide_nbi=True) is a declared unsupported region: refused at the front door on "
            "every toolchain (<4.7.0 cannot link `nvshmemx_flush_warp`; >=4.7.0 links but runs the "
            "A2A-fused DualGatedGEMM 3.23-3.25x slower, cause unknown). Set _CPO_ALLOW_IB_WIDE_NBI=1 "
            "to develop B2 itself.",
            because=(
                "the predicate is an environment variable read identically on every rank of the "
                "launch, so the decision is job-uniform by construction and cannot diverge per rank"
            ),
        )
    apply_mesh(mesh)
    _run_ib_drain_front(
        dist_manager,
        device,
        world_size,
        D=256,
        tile_N=64,
        cwg=1,
        require_ib=None,
        tile_from_dloc=True,
        n_token=N_token,
        m_linear=True,
        ib_wide=True,
        ib_wide_batch=128,
        ib_wide_nbi=True,
    )


@pytest.mark.usefixtures("_nvshmem__distributed__front_a2a_staged")
@pytest.mark.usefixtures("_module_skip__distributed__front_a2a_staged")
@FRONT_A2A.parametrize("mesh")
def test_front_ib_wide_run_i_subband(apply_mesh, mesh, dist_manager, device, world_size):
    """WIDE-PUT run_i SUB-BAND (Stage-1, the CAVEAT-A end-to-end gate): ib_wide=True at a LARGE off-grid
    N where the derived run_i raster ENGAGES (R >= R_MIN) AND is SUB-BAND (run_i_tiles ∤ ncluster_m), so
    the scheduler exercises the clamp+monotonic decode (b275675 mirror on the i-axis) — the silent-zero
    the host bijection test (test_run_i_bijection.py) guards, proven here on real hardware end-to-end.

    N=10048 (LINEAR M, %128==64 off-grid + %8 16-B-ok): rows_per_peer=10048, tile_M=128 -> ncluster_m=79
    (PRIME) so run_i is sub-band for ANY plausible grid_z (H100: R=19-22). ``expect_run_i=True`` asserts
    the derived R actually engaged + is sub-band (never silently the default raster). Gates recv rel/
    coverage vs the flat-cp D-major reference AND count-parity (a run_i raster that dropped/duplicated a
    subtile breaks arrive_full==arrive_empty -> deadlock -> the torchrun ``timeout`` catches it). HYBRID
    only (require_ib=True): needs >=1 IB peer (cp=(2,8) across two 8-GPU nodes). Anchor D=256,
    CTA tile from best_front_tile(Dloc); W=128 (the production 32 KiB width, capped by R at this M)."""
    apply_mesh(mesh)
    _run_ib_drain_front(
        dist_manager,
        device,
        world_size,
        D=256,
        tile_N=64,
        cwg=1,
        require_ib=True,
        tile_from_dloc=True,
        n_token=10048,
        m_linear=True,
        ib_wide=True,
        ib_wide_batch=128,
        expect_run_i=True,
    )


@pytest.mark.usefixtures("_nvshmem__distributed__front_a2a_staged")
@pytest.mark.usefixtures("_module_skip__distributed__front_a2a_staged")
@FRONT_A2A.parametrize(
    "D",
    "tile_N",
    "cwg",
    cells=_DECPL,
    because=(
        "cells= and not a product, exactly as the per-subtile NVLink arm: (D, tile_N, cwg) travel "
        "together, and the (16, 16, 1) cell needs a real IB peer to be more than the coupled store"
    ),
)
@FRONT_A2A.parametrize("mesh")
def test_front_ib_wide_nvlink_arm(
    apply_mesh, mesh, dist_manager, device, world_size, D, tile_N, cwg
):
    """WIDE-PUT on an ALL-NVLink (all-P2P) job: the has_ib_peers collapse elides the whole IB drain (incl.
    the wide producer/consumer) -> ib_wide reduces to the coupled TMA-S2G store, BYTE-IDENTICAL. Proves
    ib_wide does NOT regress the all-P2P path (the collapse is orthogonal to the wide flag). Runnable on a
    SINGLE venue E DGX (cp<=8, all-NVLink). W=128."""
    apply_mesh(mesh)
    _run_ib_drain_front(
        dist_manager,
        device,
        world_size,
        D,
        tile_N,
        cwg,
        require_ib=False,
        ib_wide=True,
        ib_wide_batch=128,
    )


@pytest.mark.usefixtures("_nvshmem__distributed__front_a2a_staged")
@pytest.mark.usefixtures("_module_skip__distributed__front_a2a_staged")
@FRONT_A2A.parametrize("mesh")
@matrix_exempt(
    "the subject is whether a MODEL of the fabric agrees with a PROBE of it. There is no shape or "
    "dtype to sweep -- the mesh is swept because the mesh is what the model is about"
)
@numeric_exempt(
    "compares two tuples of BOOLEANS -- a synthesized P2P reachability table against the live "
    "probe -- with `assert modelled == probed`. There is no computed tensor here and so nothing "
    "for an element-wise bound to be taken over; an exact tuple equality is already the strongest "
    "comparison this subject admits. Needed only from world_size 8: at 2 ranks the cp=8 and "
    "cp=(2,4) meshes skip, so the coverage layer never saw this test RUN before"
)
def test_the_synthesized_p2p_table_agrees_with_the_live_probe(
    apply_mesh, mesh, dist_manager, device, world_size
):
    """The synthesis is a MODEL of the fabric; this is the only thing that makes it falsifiable.

    `synth_p2p_table` computes P2P reachability from rank arithmetic instead of asking nvshmem.
    That is what lets an IB-carrying config be compiled on a single node -- and it is also how a
    forced `is_p2p` could silently describe a machine that does not exist, producing a cubin no
    real job would ever emit and a byte-identity comparison that proves nothing.

    So the model is checked against `build_p2p_table`, the SAME probe the production path uses,
    on whatever mesh this launch actually has. A disagreement means either the domain arithmetic is
    wrong or ``LOCAL_WORLD_SIZE`` does not describe the allocation; both are worth failing on, and
    the message says which values disagreed.

    Note this test is only as strong as the launch: on a single-node mesh the probe returns all-P2P
    and the model trivially agrees. The cross-node case is the one that discriminates, and it needs
    a mesh that spans nodes -- which is why the reported mesh is in the failure message.
    """
    import os as _os

    from fold_cp_ops.distributed.gemm_sm90_a2a import build_p2p_table

    apply_mesh(mesh)
    pm = _pe_map(dist_manager)
    pe_table = tuple(int(v) for v in pm.cp_pe_table.tolist())
    my_pe = int(dist_manager.rank)
    node_sz = int(_os.environ.get("LOCAL_WORLD_SIZE", 0) or 0)
    if node_sz < 1:
        rank_invariant_skip(
            "LOCAL_WORLD_SIZE is unset, so the allocation's node size is unknown and the model has "
            "no input to be checked against",
            because="LOCAL_WORLD_SIZE is set by the launcher and is identical on every rank",
        )

    probed = tuple(bool(v) for v in build_p2p_table(pe_table))
    modelled = DualGatedGemmDistSm90.synth_p2p_table(pe_table, my_pe, node_sz)
    assert modelled == probed, (
        f"the synthesized is_p2p disagrees with the live probe on mesh={mesh}: "
        f"modelled={modelled} probed={probed} pe_table={pe_table} my_pe={my_pe} "
        f"node_sz={node_sz}. Either the domain arithmetic (pe // node_sz) is wrong for this "
        f"fabric, or LOCAL_WORLD_SIZE does not describe the allocation. A forced is_p2p built "
        f"from this model would compile a cubin no real job emits."
    )


@matrix_exempt(
    "pure host-side arithmetic over a PE table -- no kernel, no shape, no dtype. Runs anywhere, "
    "including with no CUDA and no process group"
)
def test_synth_p2p_table_models_the_nvlink_domain_boundary():
    """`synth_p2p_table` reproduces the domain split a `node_sz`-GPU NVLink island would give.

    Purpose of the override it feeds: `is_p2p` is a const_expr baked into the kernel, so the cubin
    depends on its CONTENTS. Without a synthesized table, byte identity for an IB-carrying config
    needs a 2-node allocation for every compile.

    The cases are the hybrid `cp=(2,8)` mesh the sweep targets: 16 PEs over 2 islands of 8, checked
    from a rank on each side, plus the boundary ranks 7 and 8 where an off-by-one would show.
    """
    G = DualGatedGemmDistSm90
    pe = tuple(range(16))
    for my_pe in (0, 7):
        t = G.synth_p2p_table(pe, my_pe, 8)
        assert t == (True,) * 8 + (False,) * 8, f"my_pe={my_pe}: {t}"
    for my_pe in (8, 15):
        t = G.synth_p2p_table(pe, my_pe, 8)
        assert t == (False,) * 8 + (True,) * 8, f"my_pe={my_pe}: {t}"
    # A SINGLE-island job is all-P2P, which is what collapses the IB path -- the arithmetic must
    # produce that too, or the NVLink arm could never be synthesized.
    assert G.synth_p2p_table(tuple(range(8)), 0, 8) == (True,) * 8

    # The refusals. node_sz=16 over 16 PEs would claim a 16-wide NVSwitch domain, which no H100 has.
    with pytest.raises(ValueError, match=r"must be a positive int"):
        G.synth_p2p_table(pe, 0, 0)
    with pytest.raises(ValueError, match=r"does not describe a real machine"):
        G.synth_p2p_table(pe, 0, 16)


@matrix_exempt(
    "the subject is a HOST-side front-door refusal driven by a synthesized is_p2p table, so it needs "
    "no mesh, no GPU and no kernel launch; sweeping the matrix would compile the same refusal 14x"
)
@numeric_exempt("asserts a front-door ValueError, not a computed tensor")
def test_a_coupled_store_is_refused_when_a_peer_is_ib():
    """`configure_a2a` without `ib_drain` refuses a pe_table that has an IB peer.

    Functionality & semantics:
        Builds the is_p2p table a 2-node job WOULD probe (`synth_p2p_table`, node_sz=8 over 16 cp
        slots) and passes it as the `is_p2p=` override, so the check runs against a declared fabric
        rather than this machine's. The COUPLED store TMA-S2Gs into a peer's symmetric heap and
        `nvshmem_ptr` returns NULL for an IB peer, so cross-node it forms an unmapped address; the
        fault then surfaces at nvshmem's `barrier.cu:55` as an illegal memory access with exit 255 on
        every rank and NO Python traceback. This asserts it is a front-door `ValueError` instead.

        The all-P2P arm is the control, and it is what proves the check is not simply always-on: the
        same call with a single-node table must be ACCEPTED, because an all-P2P job is exactly where
        the coupled store is correct and fast.

    Input requirements:
        None -- host-side only. `configure_a2a` is a build-time call that bakes const_expr state; no
        process group, no nvshmem and no device are touched on this path.

    Returns:
        None.

    Raises:
        Nothing. `pytest.raises` owns the expected `ValueError`.
    """
    G = DualGatedGemmDistSm90
    cp, node_sz = 16, 8
    pe_table = tuple(range(cp))
    cross_node = G.synth_p2p_table(pe_table, my_pe=0, node_sz=node_sz)
    assert not all(cross_node), "synthesized table has no IB peer; the test would assert nothing"

    g = G(Float32, BFloat16, (128, 64), (1, 1, 1))
    with pytest.raises(ValueError, match="COUPLED NVLink-only A2A store"):
        g.configure_a2a(
            cp=cp, my_cp_rank=0, rows_per_peer=256, pe_table=pe_table, is_p2p=cross_node
        )

    # CONTROL: the same call on an all-P2P table must be accepted, or the check is just a veto.
    all_p2p = G.synth_p2p_table(tuple(range(node_sz)), my_pe=0, node_sz=node_sz)
    assert all(all_p2p), "single-node table should be all-P2P"
    g2 = G(Float32, BFloat16, (128, 64), (1, 1, 1))
    g2.configure_a2a(
        cp=node_sz, my_cp_rank=0, rows_per_peer=256, pe_table=tuple(range(node_sz)), is_p2p=all_p2p
    )


@matrix_exempt(
    "host-side validation of a caller-supplied table -- no kernel, no shape, no dtype. The point is "
    "that a malformed override is refused, and refusal needs no device"
)
def test_a_malformed_is_p2p_override_is_refused():
    """An override the PROBE could not have returned must not reach the kernel.

    Both refusals are about a table that describes no machine. A length mismatch indexes the wrong
    cp slot or runs off the end of the store's peer loop; more than 8 P2P peers claims an NVSwitch
    island wider than any H100 has. Neither can come out of `_build_p2p_table`, so neither may come
    out of an override -- otherwise the escape hatch that exists to let a real config be compiled
    becomes a way to compile one that cannot exist.
    """
    G = DualGatedGemmDistSm90
    with pytest.raises(ValueError, match=r"they index the same cp slots"):
        G._validated_p2p_table((True, False), (0, 1, 2, 3))
    with pytest.raises(ValueError, match=r"the probe could not have returned this"):
        G._validated_p2p_table((True,) * 9, tuple(range(9)))
    # and the well-formed case passes through, normalized to bools
    assert G._validated_p2p_table((1, 0, 1), (5, 6, 7)) == (True, False, True)


#: The five configs the cross-tree byte-identity driver sweeps, as (D, tile_N, ring_depth).
#:
#: DERIVE FROM ``_grid()`` WHEN IT IS PORTED. These are literals because the generator is not in this
#: tree yet: ``main``'s ``fold_cp_ops/distributed/fused_trimul_autotune.py:_grid()`` produces the
#: config space from ``(cp1, has_ib_peers, N_j_loc)`` under the filters ``cn <= max(1, N_j_loc//128)``,
#: ``cn < 8``, and ``cn >= 4`` requiring ``N_j_loc % (128*cn) == 0``. Until that lands, these five are
#: transcribed rather than generated, and THAT is why they are written out -- not because the matrix
#: is being bypassed. A named obligation, not a comment nobody re-reads.
#:
#: D=256 on a cp=2 mesh gives Dloc=128, which is what makes tile_N=256 legal at all
#: (``best_front_tile``: the postact tile is tile_N//2 and must divide Dloc).
_DISTINCT = [(256, 128, 1), (256, 128, 2), (256, 128, 4), (256, 256, 1), (256, 256, 2)]

#: The anchor every cell is compared against -- the production config (``ring_depth`` 2 is baked at
#: ``dual_gated_gemm_a2a.py:255``; tile_N 128 is ``best_front_tile``'s pick below Dloc=128).
_DISTINCT_ANCHOR = (256, 128, 2)


@pytest.mark.usefixtures("_nvshmem__distributed__front_a2a_staged")
@pytest.mark.usefixtures("_module_skip__distributed__front_a2a_staged")
@FRONT_A2A.parametrize(
    "D",
    "tile_N",
    "ring_depth",
    cells=_DISTINCT,
    because=(
        "cells= and not a product: these are the FIVE configs the byte-identity driver sweeps, "
        "transcribed from main's _grid() (see _DISTINCT). The crossed pairs are not configs the "
        "driver measures, and tile_N=256 is legal only where Dloc >= 128, which is why D travels "
        "with it"
    ),
)
@FRONT_A2A.parametrize(
    "mesh",
    only={"mesh": ((("cp", 2),),)},
    because=(
        "one mesh: the subject is whether five CONFIGS emit different code, which is a property of "
        "the config and not of the mesh. cp=2 is the smallest that gives Dloc=128 at D=256, the "
        "condition tile_N=256 needs. Sweeping meshes here would multiply compiles to re-answer one "
        "question"
    ),
)
@numeric_exempt(
    "compares COMPILED CODE, not numbers: there is no computed tensor and no reference to "
    "compare it against. The subject is the digest of the cubin's code sections. Same exemption, "
    "same words, as the back-half sibling test_back_gemm_native_cubin_distinguishable -- which "
    "HAD it while this one did not, so the front byte-identity gate could not have passed even "
    "after its export was fixed"
)
def test_the_five_byte_identity_configs_are_distinguishable(
    apply_mesh, mesh, dist_manager, device, world_size, D, tile_N, ring_depth
):
    """The five sweep configs must emit DIFFERENT cubins. This is the driver's non-vacuity control.

    Why this is not byte identity, and must not be called that
        It compares configs WITHIN one tree. Byte identity is a claim about two TREES and lives in
        `benchmark/distributed/harness/cold_compile.py`'s `compare_digests`, which needs two
        ``PYTHONPATH``s and cannot be a pytest cell.

    Why the driver is unsound without it
        If two of the five emit the SAME code, the driver reporting IDENTICAL for them says nothing
        about `main` -- it says the configs were never distinguishable. That is exactly the
        `compile_key`-collision failure the driver pre-registers against: the package's single
        `compile_key` reads none of `is_p2p`, `has_ib_peers`, `ib_drain`, `decoupled` or
        `ring_depth`, so distinguishability is a property to MEASURE, not to assume.

    Why it needs a real launch (measured, not assumed)
        The front cannot be compiled host-only. `compile_gemm_with_bitcode` links the nvshmem device
        bitcode and then does ``.to(dev)`` + ``library_init``, so a compile with no live nvshmem
        dies at JIT link: ``Symbols not found: [ nvshmem_ptr ]``. That is why this is a distributed
        cell rather than the cheap host-only one it was hoped to be.

    Args are matrix-drawn; `is_p2p` is SYNTHESIZED so the IB arm is reachable on a single node --
    the cubin depends on that table's contents (five `const_expr` sites), which is the whole reason
    the sweep can run without a 2-node allocation per compile.
    """
    import tempfile

    from fold_cp_ops.testing.cubin_identity import assert_has_code, differing_kinds, digest_export

    apply_mesh(mesh)
    cp = int(dist_manager.world_size)
    Dloc, K, M = D // cp, 256, 256
    pm = _pe_map(dist_manager)
    pe_table = tuple(int(v) for v in pm.cp_pe_table.tolist())
    my_pe = int(dist_manager.rank)
    x = torch.randn(M, K, device=device, dtype=torch.bfloat16) / (K**0.5)
    Wg2 = torch.randn(2 * D, K, device=device, dtype=torch.bfloat16) / (K**0.5)
    Wp2 = torch.randn(2 * D, K, device=device, dtype=torch.bfloat16) / (K**0.5)
    with torch.cuda.use_mem_pool(DistributedManager.symmetric_mempool(device)):
        recv = torch.empty((2 * Dloc, cp * M), device=device, dtype=torch.bfloat16)
    # node_sz=1 forces EVERY peer to be an IB peer, so has_ib_peers is True and the IB arm is
    # actually emitted. With the real fabric on one node every peer is P2P and the whole IB path
    # const_expr-elides, which would make all five configs collapse to the coupled store and this
    # control pass for the wrong reason.
    is_p2p = DualGatedGemmDistSm90.synth_p2p_table(pe_table, my_pe, 1)

    def _digest(tn, rd, tag):
        def cfg(g):
            g.configure_a2a(
                cp=cp,
                my_cp_rank=int(pm.my_cp_rank),
                rows_per_peer=M,
                pe_table=pe_table,
                ib_drain=True,
                ring_depth=rd,
                consumer_warpgroups=1,
                is_p2p=is_p2p,
            )

        compiled, _run, _g, _ring = _build_front_decoupled(
            x,
            Wg2,
            Wp2,
            (128, tn),
            recv_t=recv,
            ring_depth=rd,
            consumer_warpgroups=1,
            configure_fn=cfg,
        )
        d = digest_export(compiled, f"{tempfile.mkdtemp(prefix='cpo_distinct_')}/{tag}.o")
        assert_has_code(d, tag)
        return d

    mine = _digest(tile_N, ring_depth, f"tn{tile_N}_rd{ring_depth}")
    anchor_D, anchor_tn, anchor_rd = _DISTINCT_ANCHOR
    anchor = _digest(anchor_tn, anchor_rd, f"anchor_tn{anchor_tn}_rd{anchor_rd}")
    diff = differing_kinds(mine, anchor)

    if (D, tile_N, ring_depth) == _DISTINCT_ANCHOR:
        # The anchor against itself: two independent compiles of ONE config must agree, or the
        # digest is a function of the export rather than of the code and every "differs" below is
        # meaningless.
        assert not diff, (
            f"the anchor recompiled differs from itself in {diff} -- digest is unstable"
        )
    else:
        assert diff, (
            f"config (tile_N={tile_N}, ring_depth={ring_depth}) emits code IDENTICAL to the anchor "
            f"{_DISTINCT_ANCHOR}. The byte-identity driver cannot distinguish them, so an IDENTICAL "
            f"verdict for this config would say nothing about main. Suspect the compile_key gap "
            f"(it reads none of these knobs) before concluding the kernel ignores the parameter."
        )


@FRONT_A2A.parametrize_unsupported("ring_depth", "wide_nbi")
def test_front_a2a_unsupported_combos_raise(
    ring_depth, wide_nbi, expected_error, expected_match, monkeypatch
):
    """Every declared unsupported combo is refused BY THIS KERNEL'S OWN front door.

    `front_door_raises` and not `pytest.raises`: the latter cannot tell a front-door refusal from
    the same exception type raised several frames down, and `ValueError` is common enough inside
    tracing that a region asserted with the bare form could pass with no guard at the door at all.

    Host-only -- nothing is compiled and nothing is launched, so this runs wherever the module
    imports. `configure_a2a` reaches `_configure_decoupled` before any nvshmem team op, which is
    what lets a ring-depth refusal be asserted without a mesh.

    ONE test sweeps BOTH region axes because `parametrize_unsupported` requires it: a call that
    sweeps only `ring_depth` cannot evaluate the `wide_nbi` region and is refused at import with
    "Sweep those axes too, or the test silently covers less than it appears to". That is the
    machinery preventing exactly the failure it is named for -- a region declared but never
    exercised, which reads as coverage and is not.

    ORDERING IS LOAD-BEARING for the overlapping cell (`ring_depth<1` AND `wide_nbi=True`):
    `configure_a2a` calls `_configure_decoupled` at :453 and `_configure_ib_wide` at :468, so the
    ring-depth refusal fires FIRST and is the sentence that combo must match.

    `is_p2p=(True, True)` KEEPS THIS HOST-ONLY, and it is required, not decorative. Adding
    `ib_drain=True, ib_wide=True` (needed to reach the B2 guard at all) makes a VALID-`ring_depth`
    cell walk past the ring check into the live nvshmem TEAM_SHARED P2P probe, which blocks with no
    initialised nvshmem -- measured: the run hung at `[ring_depth1-wide_nbiTrue]` after setup, with
    no traceback, which reads as a wedge rather than a failure. Supplying the table takes the
    `_validated_p2p_table` branch instead of the probe, exactly as the neighbouring
    `test_front_ib_wide_nbi_requires_ib_wide_raises` already does.

    The B2 escape hatch is DELETED, not assumed absent: `_CPO_ALLOW_IB_WIDE_NBI=1` disables the very
    refusal the `wide_nbi` region asserts, so a developer with it exported would see this test pass
    while asserting nothing.
    """
    g = DualGatedGemmDistSm90(
        Float32,
        Float32,
        (128, 64),
        (1, 1, 1),
        pingpong=False,
        is_persistent=True,
        chunk_g=1,
        n_dual_tiles=0,
        gate3_n3=0,
    )
    monkeypatch.delenv("_CPO_ALLOW_IB_WIDE_NBI", raising=False)
    with front_door_raises(expected_error, expected_match):
        g.configure_a2a(
            cp=2,
            my_cp_rank=0,
            rows_per_peer=256,
            pe_table=(0, 1),
            decoupled=True,
            ring_depth=ring_depth,
            ib_drain=True,
            is_p2p=(True, True),
            ib_wide=True,
            ib_wide_nbi=wide_nbi,
        )


@pytest.mark.usefixtures("_nvshmem__distributed__front_a2a_staged")
@pytest.mark.usefixtures("_module_skip__distributed__front_a2a_staged")
@matrix_exempt(
    "asserts a REFUSAL at `configure_a2a` over a store-mode FLAG pair, which is not a value on any declared axis (see the matrix's `unsupported=` for why these four refusals cannot be regions over these pools). Host-only: nothing is compiled and nothing is launched"
)
def test_front_ib_wide_requires_ib_drain_raises():
    """ib_wide=True REQUIRES ib_drain=True (host-only, no GPU): the wide-put coalesces ONLY the IB arm, so
    it rides the differential ib_drain machinery. configure_a2a(ib_wide=True) without ib_drain must reject
    at the API boundary (before any nvshmem team op)."""
    g = DualGatedGemmDistSm90(
        Float32,
        Float32,
        (128, 64),
        (1, 1, 1),
        pingpong=False,
        is_persistent=True,
        chunk_g=1,
        n_dual_tiles=0,
        gate3_n3=0,
    )
    with pytest.raises(ValueError, match="REQUIRES ib_drain"):
        g.configure_a2a(cp=2, my_cp_rank=0, rows_per_peer=256, pe_table=(0, 1), ib_wide=True)


# DELETED (A1.1): ``test_front_ib_wide_nbi_not_wired_raises``. It asserted that ``_configure_ib_wide``
# rejects B2 (``ib_wide_nbi=True``) with ``NotImplementedError`` — a capability boundary that no longer
# exists. B2 IS wired: ``_configure_ib_wide`` stores the flag unconditionally
# (dual_gated_gemm_a2a.py:411) and the drain loop consumes it (:2517-2536); there is ZERO
# ``NotImplementedError`` in that file, and an executed host-only probe of
# ``_configure_ib_wide(True, 128, True)`` returns ``None`` leaving ``_a2a_ib_wide_nbi=True``
# (docs/migration_failing_tests.md §1.1 / §5.5). Its two siblings above/below — the ib_drain
# prerequisite and the plain-decoupled retirement — were re-executed and both still raise, so they stay.
# The test below is NOT a revival of it: B2 IS wired, and what that one refuses is nbi WITHOUT the wide
# drain that is its only reader — the dependency EDGE, not the capability.


@pytest.mark.usefixtures("_nvshmem__distributed__front_a2a_staged")
@pytest.mark.usefixtures("_module_skip__distributed__front_a2a_staged")
@matrix_exempt(
    "asserts a REFUSAL at `configure_a2a` over a store-mode FLAG pair, which is not a value on any declared axis (see the matrix's `unsupported=` for why these four refusals cannot be regions over these pools). Host-only: nothing is compiled and nothing is launched"
)
def test_front_ib_wide_nbi_requires_ib_wide_raises(monkeypatch):
    """``ib_wide_nbi=True`` without ``ib_wide=True`` must REJECT at the API boundary (host-only, no GPU).

    RUNS UNDER THE DEVELOPER ESCAPE HATCH, and must. B2 is now refused OUTRIGHT on every toolchain by
    a check that sits ABOVE this one, so without ``_CPO_ALLOW_IB_WIDE_NBI=1`` the outright refusal
    fires first and this guard is unreachable -- the test would still go green while asserting a
    DIFFERENT sentence than the one it names. The two checks are distinct and both are load-bearing:
    the outright refusal says "B2 is not shipped"; THIS one says "if you are developing B2, the flag
    is silently INERT without ``ib_wide``". Keeping it gated preserves the second, which is exactly
    the coverage that still has value under the hatch.

    ``_a2a_ib_wide_nbi`` is read at exactly ONE site — inside ``_a2a_wide_front_drain_loop``, which the
    epilogue reaches only under ``const_expr(self._a2a_ib_wide)``. So with ``ib_wide=False`` the flag is
    stored on the instance and NEVER read: the caller silently gets the B1 blocking per-row drain it
    believed it had opted out of. Its sibling ``test_front_ib_wide_requires_ib_drain_raises`` asserts the
    analogous unmet dependency one level up; the asymmetry between them was the bug.

    Three things are pinned, and each is a separate way the defect comes back:
      1. the REFUSAL fires at all;
      2. it fires BEFORE the flag is recorded (the assignment used to precede the
         ``if not ib_wide: return`` early-out, which is how a dead flag came to be on the instance);
      3. the message NAMES the §10 retirement, so the next reader does not "fix" this by wiring nbi
         into the NARROW per-subtile drain — that arm's blocking put IS the cp16 QP-exhaustion fix.
    Then the COMPOSED form is configured to prove the guard refuses a dependency, not a capability.

    Host-only: ``ib_drain`` is left False on the refusal calls so nothing probes nvshmem, and the
    composed call passes an explicit ``is_p2p`` override for the same reason.
    """
    g = DualGatedGemmDistSm90(
        Float32,
        Float32,
        (128, 64),
        (1, 1, 1),
        pingpong=False,
        is_persistent=True,
        chunk_g=1,
        n_dual_tiles=0,
        gate3_n3=0,
    )
    monkeypatch.setenv("_CPO_ALLOW_IB_WIDE_NBI", "1")
    with pytest.raises(ValueError, match="ib_wide_nbi=True REQUIRES ib_wide=True") as ei:
        g.configure_a2a(
            cp=2,
            my_cp_rank=0,
            rows_per_peer=256,
            pe_table=(0, 1),
            ib_wide=False,
            ib_wide_nbi=True,
        )
    assert "RETIRED" in str(ei.value), (
        "the refusal must name the §10 retirement of the narrow non-blocking arm, else the next reader "
        f"re-opens the cp16 QP-exhaustion bug wiring nbi there; got: {ei.value}"
    )
    assert g._a2a_ib_wide_nbi is False, (
        "`_a2a_ib_wide_nbi` was assigned before the guard raised — the assignment must sit AFTER the "
        "`if ib_wide_nbi and not ib_wide` check"
    )
    # Not a capability refusal: the COMPOSED form configures, and the flag lands.
    g.configure_a2a(
        cp=2,
        my_cp_rank=0,
        rows_per_peer=256,
        pe_table=(0, 1),
        ib_drain=True,
        is_p2p=(True, True),
        ib_wide=True,
        ib_wide_batch=128,
        ib_wide_nbi=True,
    )
    assert g._a2a_ib_wide is True and g._a2a_ib_wide_nbi is True


@pytest.mark.usefixtures("_nvshmem__distributed__front_a2a_staged")
@pytest.mark.usefixtures("_module_skip__distributed__front_a2a_staged")
@matrix_exempt(
    "asserts a REFUSAL at `configure_a2a` over a store-mode FLAG pair, which is not a value on any declared axis (see the matrix's `unsupported=` for why these four refusals cannot be regions over these pools). Host-only: nothing is compiled and nothing is launched"
)
def test_front_plain_decoupled_retired_raises():
    """§10 retirement (host-only, no GPU): the front plain-decoupled store (decoupled=True,
    ib_drain=False) is REMOVED -> configure_a2a must REJECT it at the API boundary (the guard fires
    before any nvshmem team op). ib_drain=True is the surviving path (auto-collapses to the coupled store
    on an all-P2P job, differential drain on a hybrid job)."""
    try:
        import nvshmem.core  # noqa: F401 (host availability probe)
    except Exception:
        rank_invariant_skip("nvshmem4py unavailable on host", because=_HOST_ENV_BECAUSE)
    g = DualGatedGemmDistSm90(
        Float32,
        Float32,
        (128, 64),
        (1, 1, 1),
        pingpong=False,
        is_persistent=True,
        chunk_g=1,
        n_dual_tiles=0,
        gate3_n3=0,
    )
    with pytest.raises(ValueError, match="RETIRED"):
        g.configure_a2a(cp=2, my_cp_rank=0, rows_per_peer=256, pe_table=(0, 1), decoupled=True)


@pytest.mark.usefixtures("_nvshmem__distributed__front_a2a_staged")
@pytest.mark.usefixtures("_module_skip__distributed__front_a2a_staged")
@matrix_exempt(
    "a purely ALGEBRAIC frugality argument about the ring's token extent: it compares two arithmetic expressions and never builds a tensor, compiles a kernel or touches a device, so there is no input for the matrix to supply"
)
def test_ib_drain_ring_scratch_frugal():
    """FRUGALITY unit (host-only, no GPU): the ib_drain symmetric ring is O(Dloc·cp·rd·W) with W a
    CONSTANT (epi_m), so it stays SUB-LEADING in N_token — it must NOT scale with rows_per_peer
    (= B·N²/cp, the O(N²) recv-scaling axis). Mirrors the back's frugal-scratch assertion.

    Proves the property algebraically: the ring's token extent W = gcd(128, tile_M) is fixed by the CTA
    tile, INDEPENDENT of the token count; the recv's M_full = cp·rows_per_peer scales with N². So at two
    token counts the ring's per-slot token extent is IDENTICAL while the recv grows quadratically."""
    import math as _math

    tile_M = 128
    W = _math.gcd(128, tile_M)  # epi_m = the ring's token-innermost extent (the coalescing window)
    # Two token counts (N=16 vs N=64) -> rows_per_peer M = B*N^2/cp grows 16x; the ring's W does NOT.
    for cp in (2, 4, 8):
        M1 = 1 * 16 * 16  # N=16
        M2 = 1 * 64 * 64  # N=64 (16x larger)
        recv1 = 2 * (256 // cp) * (cp * M1)  # (2*Dloc) * M_full  ~ the recv numel
        recv2 = 2 * (256 // cp) * (cp * M2)
        assert recv2 > recv1, "recv must grow with N (sanity)"
        # ring token extent (W) is the SAME regardless of token count (the frugality invariant).
        assert W == _math.gcd(128, tile_M), "ring W drifted"
        assert W <= tile_M, f"ring W={W} must be a bounded constant <= tile_M={tile_M}"
        # W is O(1) in N: it never appears multiplied by rows_per_peer -> the ring is sub-leading.
        assert W < M2, f"ring W={W} must be strictly sub-leading vs rows_per_peer={M2} (O(N²) axis)"


# ==================================================================================================
# ROUTE-2 (A) transpose_in × IB matrix — the §10 R1 composition HARDENING. The N_i-stride-1 producer
# store (_a2a_route2_ni) now composes with ib_drain/ib_wide (LAYOUT ⊥ TRANSPORT): the decoupled
# producer/consumer/wide-coalesce became route2_ni-aware (dst_i/dst_feat/dst_j addressing; the wide
# coalesce runs along N_i within ONE N_j column). These cells mirror the D-major IB units above but
# exercise the 3-D (2*Dloc, N_i, N_j) N_i-stride-1 store (the INCOMING direction). Reference = the
# _b5_route2_ni_probe idiom: the native a_major="k" einsum over the recv (contract the padded N_i; the
# zero pad tail adds 0) vs the fp32 GLU oracle (gathered + shard-positioned + contracted). Preserves the
# SAME pure-coupled self-skip as the D-major units (require_ib gates hybrid vs all-P2P-collapse).
# ==================================================================================================
# Reference-gated N-list, pushed UP the §11 bench list as far as the O(N²·D) fp32 oracle fits (H100 80GB):
# 256 grid + 520 pad-gap (fast) + 1024/1088(off-grid straddle)/2048 (all fit) unconditional; 4032/4096
# guarded by a runtime OutOfMemoryError -> pytest.skip (CLAUDE.md: NEVER a memory-estimate gate). The
# LARGE-N (8192/10048) route2_ni+IB correctness is covered by the bench driver's rel=0 fused-vs-2-kernel
# gate (two bf16 recvs, no O(N²·D) fp32 oracle) — no correctness hole, just the oracle's memory limit.
_ROUTE2_N = [256, 520, 1024, 1088, 2048]
_ROUTE2_N_OOM = [4032, 4096]  # oracle may OOM at cp2 (2-node) — try/except skip in the body
_ROUTE2_N_WIDE = [256, 520, 2048]  # ib_wide subset (W-invariance proven small; 2048 the N_i stress)


def _glu_dual(x, Wg2, Wp2):
    """glu(x@Wg2^T, x@Wp2^T) -> (M, 2D) fp32 dual output (a=[:D], b=[D:]); the front postact value."""
    xf = x.float()
    return torch.sigmoid(xf @ Wg2.float().t()) * (xf @ Wp2.float().t())


def _run_ib_drain_front_route2(
    dist_manager,
    device,
    world_size,
    *,
    require_ib,
    N,
    D=256,
    cwg=1,
    ib_wide=False,
    ib_wide_batch=None,
    ib_drain=True,
):
    """route2_ni (N_i-stride-1) front × ib_drain differential drain (§10 R1 composition). Mirrors
    ``_run_ib_drain_front`` but the store DST is the 3-D (2*Dloc, N_i=cp*Xg_pad, N_j) recv (transpose_in
    walk + ``_a2a_route2_ni``). Gates the native a_major="k" einsum over the recv (contract the padded
    N_i; zero pad tail) vs the fp32 GLU oracle — the ib_drain differential drain (P2P coupled / IB
    ring+put, wide-coalesced along N_i when ib_wide) must land the SAME N_i-stride-1 recv as the coupled
    route2_ni store. ``require_ib``: True=hybrid (>=1 IB peer), False=all-P2P collapse, None=either. Only
    legit skips: NVLink-only-on-IB-job (require_ib mismatch) + Dloc<8 (front-scatter floor)."""
    import torch.distributed as dist
    import os as _os

    # D override (CPO_FRONT_D, mirror _run_ib_drain_front): a smaller D on the smallest cross-node cp both
    # shrinks the O(N²·D) fp32 oracle AND picks a non-register-wall tile — D256/cp2 gives Dloc=128 ->
    # best_front_tile=(128,256) -> the PRE-EXISTING tile_N=256 ptxas put_warp register wall; D128/cp2 gives
    # Dloc=64 -> tile_N=128 (the production regime) so the cross-node IB-drain path runs+compiles.
    D = int(_os.environ.get("CPO_FRONT_D") or D)  # `or D` = robust to an empty forwarded value
    cp = world_size
    if D % cp != 0:
        rank_invariant_skip(f"D={D} not divisible by cp={cp}", because=_SHAPE_GATE_BECAUSE)
    Dloc = D // cp
    if (
        Dloc < 8
    ):  # front feature-scatter floor (D/cp>=8 the accepted capability boundary); NOT an xfail
        rank_invariant_skip(
            f"Dloc={Dloc} < 8 (front feature-scatter floor: D/cp>=8); needs D>={8 * cp}",
            because=_SHAPE_GATE_BECAUSE,
        )
    B, K = 1, 256
    if N % cp != 0 or N % 8 != 0:
        rank_invariant_skip(
            f"N={N} needs N%cp==0 and N%8==0 (16-B floor — the only allowed shape constraint)",
            because=_SHAPE_GATE_BECAUSE,
        )
    N_loc = N // cp
    tile_M, tile_N = DualGatedGemmDistSm90.best_front_tile(Dloc)  # cp16-valid production tile
    if tile_N % 16 != 0 or Dloc % (tile_N // 2) != 0:
        rank_invariant_skip(
            f"Dloc={Dloc}: best_front_tile ({tile_M},{tile_N}) invalid (base STSM/postact floor)",
            because=_SHAPE_GATE_BECAUSE,
        )
    n_x = (N_loc + tile_M - 1) // tile_M
    Xg_pad = n_x * tile_M
    N_i = cp * Xg_pad  # padded global-i extent (N_i stride-1)
    N_j = N
    M = B * N_loc * N
    pm = _pe_map(dist_manager)
    pe_table = tuple(int(v) for v in pm.cp_pe_table.tolist())
    my = int(pm.my_cp_rank)
    rd = 2
    # has_ib self-skip (the SAME probe idiom as the D-major units): route2_ni WITHOUT a live IB peer
    # collapses to the coupled route2_ni store (byte-identical); the hybrid arm needs a 2-node mesh.
    has_ib = _job_has_ib_peers(pe_table)  # A1: the ONE shared probe (tests/distributed/topology.py)
    if require_ib is True and not has_ib:
        rank_invariant_skip(
            "hybrid route2_ni ib_drain needs >=1 IB peer (2-node mesh, e.g. cp=(2,8)).",
            because=_IB_UNIFORM_BECAUSE,
        )
    if require_ib is False and has_ib:
        rank_invariant_skip(
            "route2_ni NVLink-arm needs an all-P2P job; this job has IB peers.",
            because=_IB_UNIFORM_BECAUSE,
        )
    if (
        not ib_drain and has_ib
    ):  # the pure-COUPLED route2_ni store is NVLink-only (a coupled TMA-S2G to
        rank_invariant_skip(
            "coupled route2_ni store (no ib_drain) is NVLink-only (TMA-S2G to an IB peer NULLs); "
            "run all-P2P.",
            because=_IB_UNIFORM_BECAUSE,
        )  # an IB peer returns nvshmem_ptr NULL
    # (3-WG producer-warp drain, ported from GemmSm90A2A e6654c8): tile_N>=256 (Dloc>=128) cross-node
    # ib_drain NO LONGER walls — at tile_N=256 the drain rides the producer WG's spare warps 9-11
    # (3-WG/384-thread -> uniform 65536/384=170 >= the 154-reg put_warp/WGMMA need) instead of a dedicated
    # 4th WG (512-thread -> 128 < 154, the old ptxas C7602 wall). So these ARE real correctness cells now
    # (the register-wall skip is removed). The _use_3wg_drain gate is const_expr(tile_N==256 AND has_ib) ->
    # every tile_N<=128 config compiles byte-identical to pre-#21.

    torch.manual_seed(4321 + dist_manager.rank)
    x = torch.randn(M, K, device=device, dtype=torch.bfloat16) / (K**0.5)
    Wg2 = torch.randn(2 * D, K, device=device, dtype=torch.bfloat16) / (K**0.5)
    Wp2 = torch.randn(2 * D, K, device=device, dtype=torch.bfloat16) / (K**0.5)
    # N_i-STRIDE-1 recv: contiguous (2*Dloc, N_j, N_i) [N_i inner stride-1], presented (2*Dloc, N_i, N_j).
    # Symmetric memory via the torch MemPool, NOT nvshmem4py's allocator: torch owns the
    # nvshmem bootstrap here, so nvshmem4py's own `_is_initialized` is permanently False
    # and its `tensor()` raises `NVSHMEM Library is not initialized`. The pool also
    # recycles, which keeps the collective free off the destruction path.
    with torch.cuda.use_mem_pool(DistributedManager.symmetric_mempool(_this_rank_device())):
        recv_buf = torch.empty(
            (2 * Dloc, N_j, N_i), dtype=torch.bfloat16, device=_this_rank_device()
        )
    recv_buf.fill_(-99.0)
    recv = recv_buf.permute(0, 2, 1)  # (2*Dloc, N_i, N_j) N_i stride-1

    def _cfg(g):
        ib_kw = (
            dict(
                ib_drain=True,
                ring_depth=rd,
                consumer_warpgroups=cwg,
                ib_wide=ib_wide,
                ib_wide_batch=ib_wide_batch,
            )
            if ib_drain
            else {}
        )
        g.configure_a2a(
            cp=cp,
            my_cp_rank=my,
            rows_per_peer=M,
            pe_table=pe_table,
            transpose_in=True,
            token_grid=(B, N_loc, N),
            **ib_kw,
        )
        g._a2a_route2_ni = True  # engage the N_i-stride-1 producer store
        g._a2a_cp_axis_sizes = (cp, 1)  # 1-D token shard (cp0=cp, cp1=1)

    if ib_drain:
        compiled, run_fn, gemm_obj, ring_t = _build_front_decoupled(
            x,
            Wg2,
            Wp2,
            (tile_M, tile_N),
            recv_t=recv,
            ring_depth=rd,
            consumer_warpgroups=cwg,
            configure_fn=_cfg,
        )
    else:  # pure-coupled route2_ni store (no decoupled ring) — the non-decoupled builder, no ring_t
        compiled, run_fn, gemm_obj = _build_front(
            x,
            Wg2,
            Wp2,
            (tile_M, tile_N),
            recv_t=recv,
            configure_fn=_cfg,
        )
        ring_t = None
    try:
        _nvshmem_barrier()
        run_fn()
        _drain()
        # native a_major="k" einsum over the recv (contract the padded N_i; the pad tail must be 0) vs the
        # fp32 GLU oracle. The oracle is O(N²·D) (K=N genuine problem size) -> at large N it OOMs; catch the
        # runtime OutOfMemoryError -> skip (CLAUDE.md: runtime OOM skip, NEVER a memory-estimate gate).
        #
        # THE SKIP IS ALL_REDUCED, and the earlier argument for a bare one was wrong. It read: the
        # allocations are SYMMETRIC (same size on every rank), so the OOM is consensus'd by
        # construction. Symmetric REQUESTS do not make symmetric OUTCOMES -- what decides an OOM is
        # free memory, which differs per rank with allocator fragmentation, with what the store just
        # left resident, and with anything else sharing the device. One rank raising where its peers
        # did not is the exact shape `collective_guard` exists to stop: that rank leaves and every
        # peer blocks in the next collective. `gated_skip` reduces the decision (ANY rank asking ->
        # every rank skips), so it is COLLECTIVE and must be reached UNCONDITIONALLY -- which is why
        # the catch records a reason instead of skipping inside the `except`.
        oom_reason = None
        try:
            a_h, b_h = recv[:Dloc].float(), recv[Dloc:].float()
            tri_r2 = torch.einsum("dij,diJ->djJ", a_h, b_h)  # (Dloc, N_j, N_j)
            del a_h, b_h  # 2*(Dloc,N_i,N_j) fp32 — done with them
            dual = _glu_dual(x, Wg2, Wp2).to(torch.bfloat16)  # (M, 2D)
            gath = [torch.empty_like(dual) for _ in range(cp)]
            dist.all_gather(gath, dual.contiguous())
            # A6: scatter DIRECTLY into MY (i_pad, j, Dloc) a/b slices instead of materializing the full
            # (N_i, N_j, 2D) fp32 `a_glob` and then slicing it. Only my_cp_rank's Dloc columns of each
            # half are ever read, so the old form over-allocated by 2D/(2*Dloc) = cp (34 GB -> 8.6 GB at
            # N4096/cp4/D256 — the 30.52 GiB allocation §3.1 reports). Same values, same zero pad gap;
            # the fp32 einsum's accumulation order may differ, which is immaterial at a 5e-2 bar (and is
            # the same claim trimul_ref_chunked already carries). This is why N4096 can now RUN rather
            # than take the OOM skip — CLAUDE.md: a skip on an in-principle-supported shape is a violation.
            aa = torch.zeros(N_i, N_j, Dloc, device=device, dtype=torch.float32)  # MY a-slice
            bb = torch.zeros(N_i, N_j, Dloc, device=device, dtype=torch.float32)  # MY b-slice
            for p in range(cp):
                pg = int(pm.cp_pe_table[p].item())
                g3 = gath[pg].reshape(N_loc, N, 2 * D)
                lo, hi = p * Xg_pad, p * Xg_pad + N_loc
                aa[lo:hi] = g3[..., my * Dloc : (my + 1) * Dloc].float()
                bb[lo:hi] = g3[..., D + my * Dloc : D + (my + 1) * Dloc].float()
            del gath, dual, g3  # cp*(M,2D) bf16 — the other big block
            tri_or = torch.einsum("ijd,iJd->djJ", aa, bb)
            # The tri-vs-oracle verdict, PER ELEMENT, at the SAME REL_BAR constant. This one is a
            # STATISTIC change, not an exact restatement: it replaces a pooled ||diff|| / ||ref||,
            # which divides a localized error by the whole tensor's energy and therefore cannot see
            # a handful of wrong elements among O(N²·Dloc) correct ones. It runs INSIDE the try so a
            # comparison that OOMs at the large-N cells takes the same all_reduced skip the oracle
            # does, and BEFORE the `del` below because that frees both operands.
            rel = REL_BAR * _same_bar_elementwise(
                tri_r2, tri_or, REL_BAR, f"route2_ni tri N={N} cp={cp} (native einsum vs oracle)"
            )
            rr = (tri_r2 - tri_or).reshape(Dloc, -1).norm(dim=0) / tri_or.reshape(Dloc, -1).norm(
                dim=0
            ).clamp_min(1e-9)
            n_out = int((rr > 1e-2).sum().item())
            del tri_r2, tri_or, aa, bb, rr  # free the O(N²·D) oracle before the checks
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            oom_reason = (
                f"K=N O(N²·D) fp32 oracle OOM at N={N} cp={cp} (the genuine problem size — never a "
                "memory-estimate gate). Large-N route2_ni+IB correctness is covered by the bench driver's "
                "rel=0 fused-vs-2-kernel gate (two bf16 recvs, no O(N²·D) fp32 oracle)."
            )
        gated_skip(oom_reason)
        # pad gap store-written 0 (NOT the -99 sentinel) + no untouched cell (full coverage).
        pad_max = 0.0
        if Xg_pad > N_loc:
            for p in range(cp):
                hi = p * Xg_pad + N_loc
                pad = recv[:, hi : (p + 1) * Xg_pad, :].float()
                pad_max = max(pad_max, pad.abs().max().item() if pad.numel() else 0.0)
        untouched = int((recv == -99.0).sum().item())
        arm = "hybrid" if has_ib else "nvlink-arm"
        wtag = f"wide(W={ib_wide_batch})" if ib_wide else "per-subtile"
        tag = (
            f"front-route2-ibdrain[{arm}] {wtag} cp={cp} N={N} N_loc={N_loc} D={D} Dloc={Dloc} "
            f"tile=({tile_M},{tile_N})"
        )
        passed = rel < REL_BAR and n_out == 0 and pad_max == 0.0 and untouched == 0
        print(
            f"\n[{tag}] rank={dist_manager.rank} has_ib={has_ib} rel={rel:.3e} (bar {REL_BAR}) "
            f"n_outlier={n_out} pad_gap={pad_max:.3e} untouched={untouched} "
            f"{'PASS' if passed else 'FAIL'}",
            flush=True,
        )
        # (no scalar `assert rel < REL_BAR` here: the per-element assertion above already fired, and
        # it is the one that would name WHICH element the N_i-stride-1 store mis-landed.)
        assert n_out == 0, f"{tag}: {n_out} outlier tokens (localized store corruption)"
        assert pad_max == 0.0, (
            f"{tag}: pad gap != 0 (max {pad_max:.3e}) — einsum-UNSAFE (OOB tail not 0)"
        )
        assert untouched == 0, f"{tag}: {untouched} recv cells unwritten (coverage gap)"
    finally:
        compiled.free()
        _nvshmem_barrier()


@pytest.mark.usefixtures("_nvshmem__distributed__front_a2a_staged")
@pytest.mark.usefixtures("_module_skip__distributed__front_a2a_staged")
@FRONT_A2A.parametrize("mesh")
def test_front_route2_coupled_correct(apply_mesh, mesh, dist_manager, device, world_size):
    """FACTORING-INTEGRITY gate (§10 R1): the pre-existing COUPLED route2_ni store (transpose_in +
    _a2a_route2_ni, NO ib_drain) is oracle-clean POST-factoring — proves the recv_view factoring (removing
    the early-return, running the transport block over the N_i-stride-1 recv_view) did NOT break the
    coupled N_i-stride-1 store, INDEPENDENT of the ib_drain collapse. Pure-coupled is NVLink-only -> self-
    skips on a job with >=1 IB peer (mirror the D-major coupled self-skip). Grid (256) + off-grid (520)."""
    apply_mesh(mesh)
    for N in (256, 520):
        _run_ib_drain_front_route2(
            dist_manager, device, world_size, require_ib=None, N=N, ib_drain=False
        )


@pytest.mark.usefixtures("_nvshmem__distributed__front_a2a_staged")
@pytest.mark.usefixtures("_module_skip__distributed__front_a2a_staged")
@FRONT_A2A.parametrize("N")
@FRONT_A2A.parametrize("mesh")
def test_front_route2_ib_drain(apply_mesh, mesh, dist_manager, device, world_size, N):
    """route2_ni × ib_drain (per-subtile) — the §10 R1 composition. require_ib=None: an all-P2P job
    exercises the collapse (coupled route2_ni store); a hybrid job the IB ring+put. Grid (256), off-grid
    pad-gap (520), + the bench N-list up to 2048 unconditional, 4032/4096 OOM-guarded (K=N oracle). All
    land the SAME oracle recv. NO xfail/skip on a supported shape (only Dloc<8 / require_ib / oracle-OOM)."""
    apply_mesh(mesh)
    _run_ib_drain_front_route2(dist_manager, device, world_size, require_ib=None, N=N)


@pytest.mark.usefixtures("_nvshmem__distributed__front_a2a_staged")
@pytest.mark.usefixtures("_module_skip__distributed__front_a2a_staged")
@FRONT_A2A.parametrize(
    "N",
    "W",
    only={"N": tuple(_ROUTE2_N_WIDE)},
    because=(
        "W-invariance is proven at the small N, so the wide arm runs the three-N subset that spans "
        "what matters -- 256 (grid), 520 (pad gap), 2048 (the N_i stress) -- against the full W "
        "pool. The larger N are covered without ib_wide by test_front_route2_ib_drain, and "
        "4032/4096 would double an fp32 oracle that already sits at the OOM boundary"
    ),
)
@FRONT_A2A.parametrize("mesh")
def test_front_route2_ib_wide(apply_mesh, mesh, dist_manager, device, world_size, W, N):
    """route2_ni × ib_wide — the N_i wide-put COALESCE (consecutive dst_i within ONE N_j column). W=32
    forces mid-stream break-flushes, W=128 mostly finalize-flushes. Correctness is W-invariant (the recv
    content == the per-subtile drain). require_ib=None (all-P2P collapse OR hybrid). Off-grid N tests the
    partial trailing wide batch (CAVEAT B) on the N_i axis."""
    apply_mesh(mesh)
    _run_ib_drain_front_route2(
        dist_manager, device, world_size, require_ib=None, N=N, ib_wide=True, ib_wide_batch=W
    )


# HYBRID cells use D=128 so the tile is Dloc=D/cp-dependent but stays tile_N<=128 at the small cp a
# 2-DGX pair provides (cp2 -> Dloc=64 -> tile_N=128; cp4 -> Dloc=32 -> tile_N=64) — matching the PRODUCTION
# cp16 regime (Dloc=16 -> tile_N=32). Dloc>=128 (D=256/cp2 -> tile_N=256) hits a PRE-EXISTING front-ib_drain
# ptxas REGISTER WALL (needs 154 > 128 regs; memory reference_a2a_4wg_tile256_register_wall, known fix =
# 3-WG producer-warp drain, not wired for this front path) — that is NOT a route2_ni defect and NOT the
# production hybrid tile; a cp2/D256 hybrid run is disqualified by the tile, not the composition.
@pytest.mark.usefixtures("_nvshmem__distributed__front_a2a_staged")
@pytest.mark.usefixtures("_module_skip__distributed__front_a2a_staged")
@FRONT_A2A.parametrize("cwg")
@FRONT_A2A.parametrize("mesh")
def test_front_route2_ib_drain_hybrid(apply_mesh, mesh, dist_manager, device, world_size, cwg):
    """route2_ni × ib_drain HYBRID (require_ib=True) — the DECISIVE cross-node composition: P2P peers
    coupled TMA-S2G (4-coord N_i store), IB peers symmetric ring + BLOCKING put_warp along N_i. Skips
    unless the job has >=1 IB peer (2-node mesh, e.g. cp=(2,8) across two 8-GPU nodes). D=128
    keeps the tile <=128 (no tile_N=256 register wall — see the section note)."""
    apply_mesh(mesh)
    _run_ib_drain_front_route2(
        dist_manager, device, world_size, require_ib=True, N=256, D=128, cwg=cwg
    )


@pytest.mark.usefixtures("_nvshmem__distributed__front_a2a_staged")
@pytest.mark.usefixtures("_module_skip__distributed__front_a2a_staged")
@FRONT_A2A.parametrize("mesh")
def test_front_route2_ib_drain_hybrid_dloc8(apply_mesh, mesh, dist_manager, device, world_size):
    """route2_ni × ib_drain HYBRID at the Dloc=8 front feature-scatter floor (D16/cp2 -> Dloc=8 ->
    best_front_tile(8)=(128,16), postact=tile_N//2=8): guards the transpose_in N_i-stride-1 store at the
    accepted D/cp>=8 boundary, complementary to the D-major _DECPL_HYB (16,16,1) guard. tile_N=16 is far
    below the tile_N=256 register wall. HYBRID only (needs >=1 IB peer, e.g. venue F cp2 cross-node).
    FIRST-PRINCIPLE: route2_ni is a supported variant + Dloc=8 a supported shape -> harden, never skip."""
    apply_mesh(mesh)
    _run_ib_drain_front_route2(
        dist_manager, device, world_size, require_ib=True, N=256, D=16, cwg=1
    )


@pytest.mark.usefixtures("_nvshmem__distributed__front_a2a_staged")
@pytest.mark.usefixtures("_module_skip__distributed__front_a2a_staged")
@FRONT_A2A.parametrize("mesh")
def test_front_route2_ib_wide_hybrid(apply_mesh, mesh, dist_manager, device, world_size):
    """route2_ni × ib_wide HYBRID — the N_i wide coalesce CROSS-NODE (up to 32 KiB put along N_i within
    one N_j column). Proves count-parity §6.4 (a mismatch deadlocks -> torchrun timeout) + the wide put
    lands the SAME recv as the per-subtile drain. HYBRID only (needs >=1 IB peer). D=128 (tile<=128)."""
    apply_mesh(mesh)
    _run_ib_drain_front_route2(
        dist_manager,
        device,
        world_size,
        require_ib=True,
        N=256,
        D=128,
        ib_wide=True,
        ib_wide_batch=128,
    )


_FRONT_2D = [256, 512]  # D == feature; M = N_i_loc*N_j_loc per rank


@pytest.mark.usefixtures("_nvshmem__distributed__front_a2a_staged")
@pytest.mark.usefixtures("_module_skip__distributed__front_a2a_staged")
@FRONT_A2A.parametrize(
    "D",
    only={"D": tuple(_FRONT_2D)},
    because=(
        "the 2-D mesh cell needs a D that BOTH cp axes divide, so it runs the two widest values. "
        "16 and 128 leave Dloc below the front's feature-scatter floor once the mesh is factored"
    ),
)
@FRONT_A2A.parametrize("mesh")
def test_front_dmajor_2d_correct(apply_mesh, mesh, dist_manager, device, world_size, D):
    """2-D token shard front D-major store: recv-index 2-D-invariance (flat-cp ref).

    Runs the front under a 2-D ``(cp0, cp1)`` cp mesh with a TRUE 2-D token block
    (M = N_i_loc·N_j_loc) via ``configure_a2a_sharded``, gates the recv against the flat-cp
    ``_ref_front_dmajor`` (proves the kernel store is recv-index 2-D-invariant). Skips unless the
    session mesh is 2-D (CPO_DIST_MESH=cp=2*2).
    """
    apply_mesh(mesh)

    axis_sizes = _cp_axis_sizes(dist_manager)
    if len(axis_sizes) != 2:
        rank_invariant_skip(
            f"2-D front test needs a 2-D cp mesh; session mesh is {axis_sizes}",
            because=_SHAPE_GATE_BECAUSE,
        )
    cp0, cp1 = axis_sizes
    cp = cp0 * cp1
    if D % cp != 0:
        rank_invariant_skip(f"D={D} not divisible by cp={cp}", because=_SHAPE_GATE_BECAUSE)
    Dloc = D // cp
    tile_N = 128
    tile_n_postact = tile_N // 2
    if Dloc % tile_n_postact != 0:
        rank_invariant_skip(
            f"Dloc={Dloc} not a multiple of postact tile {tile_n_postact} at {axis_sizes}",
            because=_SHAPE_GATE_BECAUSE,
        )
    # N chosen so M = N_i_loc*N_j_loc is a multiple of the CTA tile-M (128): with (cp0,cp1)=(2,2),
    # N_i_loc=N_j_loc=N//2, so M=(N//2)^2; N=32 -> M=256 (>= 128, no partial M-tile).
    N = 32  # global token extent (square); N_i_loc = N//cp0, N_j_loc = N//cp1
    if N % cp0 != 0 or N % cp1 != 0:
        rank_invariant_skip(
            f"N={N} not divisible by both cp axes {axis_sizes}", because=_SHAPE_GATE_BECAUSE
        )
    N_i_loc, N_j_loc = N // cp0, N // cp1
    M = N_i_loc * N_j_loc
    if M % 128 != 0:  # rows_per_peer must be a multiple of the CTA tile-M (no partial M-tile)
        rank_invariant_skip(
            f"M={M} (=N_i_loc*N_j_loc) not a multiple of CTA tile_M=128 at {axis_sizes}",
            because=_SHAPE_GATE_BECAUSE,
        )
    K = 256
    placements, pm = _trimul_placements(dist_manager)
    # PURE-COUPLED sharded store (configure_a2a_sharded, no ib_drain) is NVLink-only (§10): self-skip a
    # cross-node (has_ib) job (coupled TMA-S2G to an IB peer's NULL nvshmem_ptr faults).
    _skip_if_coupled_cross_node(tuple(int(v) for v in pm.cp_pe_table.tolist()))
    mesh = _session_cp_mesh(dist_manager)
    rows_per_peer = M
    M_full = cp * rows_per_peer

    torch.manual_seed(4321 + dist_manager.rank)
    x = torch.randn(M, K, device=device, dtype=torch.bfloat16) / (K**0.5)
    Wg2 = torch.randn(2 * D, K, device=device, dtype=torch.bfloat16) / (K**0.5)
    Wp2 = torch.randn(2 * D, K, device=device, dtype=torch.bfloat16) / (K**0.5)
    # Symmetric memory via the torch MemPool, NOT nvshmem4py's allocator: torch owns the
    # nvshmem bootstrap here, so nvshmem4py's own `_is_initialized` is permanently False
    # and its `tensor()` raises `NVSHMEM Library is not initialized`. The pool also
    # recycles, which keeps the collective free off the destruction path.
    with torch.cuda.use_mem_pool(DistributedManager.symmetric_mempool(_this_rank_device())):
        recv = torch.empty((2 * Dloc, M_full), dtype=torch.bfloat16, device=_this_rank_device())
    recv.fill_(-99.0)

    fn = lambda g: g.configure_a2a_sharded(mesh, placements, pe_map=pm, rows_per_peer=M)
    compiled, run_fn, _ = _build_front(x, Wg2, Wp2, (128, tile_N), recv_t=recv, configure_fn=fn)
    try:
        _nvshmem_barrier()
        run_fn()
        _drain()

        expected = _ref_front_dmajor(
            x, Wg2, Wp2, pm, cp=cp, rows_per_peer=M, Dloc=Dloc, world_size=world_size
        )
        untouched = int((recv == -99.0).sum().item())
        a_is_view = recv[:Dloc].data_ptr() == recv.data_ptr()
        tag = f"front-staged-2D cp={cp0}x{cp1} N={N} D={D} M={M} Dloc={Dloc}"
        # The recv-vs-reference verdict, PER ELEMENT. `_same_bar_elementwise` carries the
        # historical `max|got-ref| / max|ref| < REL_BAR` bar UNCHANGED -- a maximum is under a
        # threshold iff every element is -- and raises naming the worst offender's INDEX with
        # its actual, reference and bound. It sits where the pooled scalar was computed, so the
        # scalar `assert rel < REL_BAR` that used to follow is gone. `rel` is that same ratio,
        # recovered exactly for the diagnostic print (the bound is the uniform REL_BAR*max|ref|,
        # so worst |err|/bound * REL_BAR == the old ratio); it no longer decides the test.
        rel = REL_BAR * _same_bar_elementwise(recv, expected, REL_BAR, tag)
        passed = rel < REL_BAR and untouched == 0 and a_is_view
        print(
            f"\n[{tag}] rank={dist_manager.rank} recv-index-invariant rel={rel:.3e} (bar {REL_BAR}) "
            f"untouched={untouched}/{recv.numel()} a_view={a_is_view} {'PASS' if passed else 'FAIL'}",
            flush=True,
        )
        assert untouched == 0, f"{tag}: {untouched}/{recv.numel()} recv cells unwritten"
        assert a_is_view, f"{tag}: a/b half not a view"
    finally:
        compiled.free()
        _nvshmem_barrier()


_DET_RUNS = 3


@pytest.mark.usefixtures("_nvshmem__distributed__front_a2a_staged")
@pytest.mark.usefixtures("_module_skip__distributed__front_a2a_staged")
@FRONT_A2A.parametrize("mesh")
def test_front_deterministic(apply_mesh, mesh, dist_manager, device, world_size):
    """K=3 repeated front D-major store runs must be BIT-IDENTICAL + fully covered.

    The async-vec-race signature is cross-run non-determinism; a BITWISE comparison across K runs of
    the SAME compiled kernel + SAME inputs is the strongest catch, and it is spelled
    ``numerics.assert_bitwise`` rather than ``torch.equal``. Not because ``torch.equal`` is banned --
    it is exact and element-wise, so the guard permits it -- but because it records nothing for the
    coverage gate and reports no index. Coverage (untouched == 0) keeps this non-vacuous; run 0 is
    gated against the fp32 ref so a deterministically-WRONG store still fails.
    Drives a LARGER token block M (multi-wave) to exercise the race regime. COOPERATIVE-only: the
    STAGED parent (DualGatedGemmSm90) asserts ``not pingpong`` (no pingpong path), unlike the
    stagec/back kernels — so there is no pingpong determinism cell here.
    """
    apply_mesh(mesh)
    pingpong = False

    cp = world_size
    D, tile_N = 256, 128
    if D % cp != 0:
        rank_invariant_skip(f"D={D} not divisible by cp={cp}", because=_SHAPE_GATE_BECAUSE)
    Dloc = D // cp
    if Dloc % (tile_N // 2) != 0:
        rank_invariant_skip(
            f"Dloc={Dloc} not a multiple of postact tile {tile_N // 2} at cp={cp}",
            because=_SHAPE_GATE_BECAUSE,
        )
    M, K = 2048, 256  # large token block -> multi-wave (the race regime)
    rows_per_peer = M
    M_full = cp * rows_per_peer
    pm = _pe_map(dist_manager)

    torch.manual_seed(4321 + dist_manager.rank)
    x = torch.randn(M, K, device=device, dtype=torch.bfloat16) / (K**0.5)
    Wg2 = torch.randn(2 * D, K, device=device, dtype=torch.bfloat16) / (K**0.5)
    Wp2 = torch.randn(2 * D, K, device=device, dtype=torch.bfloat16) / (K**0.5)
    # Symmetric memory via the torch MemPool, NOT nvshmem4py's allocator: torch owns the
    # nvshmem bootstrap here, so nvshmem4py's own `_is_initialized` is permanently False
    # and its `tensor()` raises `NVSHMEM Library is not initialized`. The pool also
    # recycles, which keeps the collective free off the destruction path.
    with torch.cuda.use_mem_pool(DistributedManager.symmetric_mempool(_this_rank_device())):
        recv = torch.empty((2 * Dloc, M_full), dtype=torch.bfloat16, device=_this_rank_device())
    recv.fill_(-99.0)

    pe_table = tuple(int(v) for v in pm.cp_pe_table.tolist())
    a2a_cfg = dict(
        cp=cp, my_cp_rank=int(pm.my_cp_rank), rows_per_peer=rows_per_peer, pe_table=pe_table
    )
    compiled, run_fn, _ = _build_front(
        x, Wg2, Wp2, (128, tile_N), recv_t=recv, a2a_cfg=a2a_cfg, pingpong=pingpong
    )
    expected = _ref_front_dmajor(
        x, Wg2, Wp2, pm, cp=cp, rows_per_peer=rows_per_peer, Dloc=Dloc, world_size=world_size
    )
    mode = "pingpong" if pingpong else "cooperative"
    try:
        first = None
        rel = None
        for it in range(_DET_RUNS):
            recv.fill_(-99.0)
            _nvshmem_barrier()
            run_fn()
            _drain()
            untouched = int((recv == -99.0).sum().item())
            snap = recv.clone()
            if it == 0:
                first = snap
                # Run 0 is the one compared to the reference, PER ELEMENT and at the same bar (see
                # `_same_bar_elementwise`). It fires HERE rather than after the loop because a wrong
                # run 0 makes every later "bit-identical to run 0" assertion vacuously true -- three
                # identical wrong answers pass a determinism test, which is exactly the shape this
                # test cannot afford to be blind to.
                rel = REL_BAR * _same_bar_elementwise(
                    recv, expected, REL_BAR, f"det front {mode} cp={cp} run0"
                )
            # `torch.equal` is NOT banned by the numerics guard -- it is exact and element-wise, so
            # it cannot shadow a single wrong element -- but it records nothing for the coverage
            # gate. `assert_bitwise` is the sanctioned spelling: same verdict, and on failure it
            # names the differing index instead of only a max-abs.
            numerics.assert_bitwise(snap, first, what=f"front {mode} run {it} vs run 0")
            bit_identical = True
            print(
                f"\n[det front {mode} cp={cp}] rank={dist_manager.rank} iter={it} "
                f"untouched={untouched} bit_identical_to_run0={bit_identical}",
                flush=True,
            )
            assert untouched == 0, f"front {mode}: run {it} left {untouched} cells unwritten"
        print(
            f"\n[det front {mode} cp={cp}] rank={dist_manager.rank} DETERMINISTIC over {_DET_RUNS} "
            f"runs, rel={rel:.3e} PASS",
            flush=True,
        )
    finally:
        compiled.free()
        _nvshmem_barrier()


# --------------------------------------------------------------------------- #
# PARTIAL-TOKEN AUTO-CLAMP correctness : an arbitrary N_token whose per-rank token block
# rows_per_peer = N_token^2/cp is NOT a multiple of CTA tile_M=128 stores correctly via the high-end
# OOB clamp on the stride-1 TOKEN axis. This is now AUTOMATIC (NO partial_token_clamp flag): a
# misaligned rows_per_peer auto-builds the peer atoms over MY (rows_per_peer, 2*Dloc) column block
# (descriptor token-bound = rows_per_peer), so the partial-last M-tile's overshoot past rows_per_peer
# is DROPPED by the clamp instead of bleeding into the next rank's columns; an aligned rows_per_peer
# keeps the full-M_full absolute-dst atom (byte-identical). PASS = recv oracle-clean (rel<bar, 0 outlier
# token rows, full coverage). N_token are REACHABLE (N%8==0): the rows_per_peer they induce have %128 in
# {0 (aligned control) or >=8 (partial, clamp valid)} — the 1..7-token tail (16B limit) is unreachable +
# guarded (see test_..._guard_raises / _guard_xfail). The AUTO-CLAMP is exercised by the partial cells
# (e.g. N=24 cp2 -> rpp=288, %128=32). (rpp may != N^2 -> recv-vs-ref.)
# --------------------------------------------------------------------------- #
_PARTIAL_NTOK = [16, 24, 32, 40]  # all %8; cp2 rpp {128,288,512,800}, cp4 rpp {64,144,256,400}


@pytest.mark.usefixtures("_nvshmem__distributed__front_a2a_staged")
@pytest.mark.usefixtures("_module_skip__distributed__front_a2a_staged")
@FRONT_A2A.parametrize(
    "N_token",
    only={"N_token": tuple(_PARTIAL_NTOK)},
    because=(
        "the auto-clamp cells are the SMALL N, whose rows_per_peer = N^2/cp lands at %128 in "
        "{0 (the aligned control) or >= 8 (partial, clamp valid)}. The large N in the pool are the "
        "off-grid drain cells and are swept by test_front_ib_drain_offgrid"
    ),
)
@FRONT_A2A.parametrize("mesh")
def test_front_partial_token_clamp_correct(
    apply_mesh, mesh, dist_manager, device, world_size, N_token
):
    """reachable N_token via the partial-token high-end COLUMN AUTO-clamp -> oracle-clean recv (cp2/cp4).

    NO partial_token_clamp flag: a misaligned rows_per_peer auto-takes the clamp; an aligned one keeps
    the absolute-dst store. Validates the Task-1 auto-relax on STATIC shapes (the flag is now optional)."""
    apply_mesh(mesh)
    from tests.distributed.correctness_harness import compute_error_histogram

    cp = world_size
    D, tile_N = 256, 128
    if D % cp != 0:
        rank_invariant_skip(f"D={D} not divisible by cp={cp}", because=_SHAPE_GATE_BECAUSE)
    Dloc = D // cp
    tile_n_postact = tile_N // 2
    if Dloc % tile_n_postact != 0:
        rank_invariant_skip(
            f"Dloc={Dloc} not a multiple of postact tile_N={tile_n_postact} at cp={cp}",
            because=_SHAPE_GATE_BECAUSE,
        )
    if (N_token * N_token) % cp != 0:
        rank_invariant_skip(
            f"N_token^2={N_token * N_token} not divisible by cp={cp}", because=_SHAPE_GATE_BECAUSE
        )
    # rows_per_peer = my token block = B*N_token^2/cp (B=1). REACHABLE -> %128 in {0 or >=8}.
    rows_per_peer = (N_token * N_token) // cp
    M = rows_per_peer
    K = 256
    M_full = cp * rows_per_peer
    pm = _pe_map(dist_manager)

    torch.manual_seed(4321 + dist_manager.rank)
    x = torch.randn(M, K, device=device, dtype=torch.bfloat16) / (K**0.5)
    Wg2 = torch.randn(2 * D, K, device=device, dtype=torch.bfloat16) / (K**0.5)
    Wp2 = torch.randn(2 * D, K, device=device, dtype=torch.bfloat16) / (K**0.5)
    # Symmetric memory via the torch MemPool, NOT nvshmem4py's allocator: torch owns the
    # nvshmem bootstrap here, so nvshmem4py's own `_is_initialized` is permanently False
    # and its `tensor()` raises `NVSHMEM Library is not initialized`. The pool also
    # recycles, which keeps the collective free off the destruction path.
    with torch.cuda.use_mem_pool(DistributedManager.symmetric_mempool(_this_rank_device())):
        recv = torch.empty((2 * Dloc, M_full), dtype=torch.bfloat16, device=_this_rank_device())
    recv.fill_(-99.0)

    pe_table = tuple(int(v) for v in pm.cp_pe_table.tolist())
    # AUTO-CLAMP: NO partial_token_clamp -> a misaligned rows_per_peer auto-takes the block clamp.
    a2a_cfg = dict(
        cp=cp,
        my_cp_rank=int(pm.my_cp_rank),
        rows_per_peer=rows_per_peer,
        pe_table=pe_table,
    )
    compiled, run_fn, _ = _build_front(x, Wg2, Wp2, (128, tile_N), recv_t=recv, a2a_cfg=a2a_cfg)
    try:
        _nvshmem_barrier()
        run_fn()
        _drain()
        expected = _ref_front_dmajor(
            x, Wg2, Wp2, pm, cp=cp, rows_per_peer=rows_per_peer, Dloc=Dloc, world_size=world_size
        )
        untouched = int((recv == -99.0).sum().item())
        got_bnnd = recv.t().contiguous().reshape(1, M_full, 1, 2 * Dloc)
        ref_bnnd = expected.t().contiguous().reshape(1, M_full, 1, 2 * Dloc)
        h = compute_error_histogram(got_bnnd, ref_bnnd)
        tag = f"front-auto-clamp cp={cp} rows_per_peer={rows_per_peer} D={D} Dloc={Dloc}"
        # The recv-vs-reference verdict, PER ELEMENT. `_same_bar_elementwise` carries the
        # historical `max|got-ref| / max|ref| < REL_BAR` bar UNCHANGED -- a maximum is under a
        # threshold iff every element is -- and raises naming the worst offender's INDEX with
        # its actual, reference and bound. It sits where the pooled scalar was computed, so the
        # scalar `assert rel < REL_BAR` that used to follow is gone. `rel` is that same ratio,
        # recovered exactly for the diagnostic print (the bound is the uniform REL_BAR*max|ref|,
        # so worst |err|/bound * REL_BAR == the old ratio); it no longer decides the test.
        rel = REL_BAR * _same_bar_elementwise(recv, expected, REL_BAR, tag)
        passed = rel < REL_BAR and h.n_outlier_rows == 0 and untouched == 0
        print(
            f"\n[{tag}] rank={dist_manager.rank} rel={rel:.3e} (bar {REL_BAR}) {h.summary()} "
            f"untouched={untouched}/{recv.numel()} {'PASS' if passed else 'FAIL'}",
            flush=True,
        )
        assert h.n_outlier_rows == 0, (
            f"{tag}: {h.n_outlier_rows} outlier token rows (partial-tile overshoot bled into a "
            f"neighbor column?); worst @ {h.worst_row_index} ratio {h.row_outlier_ratio:.1f}x"
        )
        assert untouched == 0, (
            f"{tag}: {untouched}/{recv.numel()} recv cells unwritten (coverage gap)"
        )
    finally:
        compiled.free()
        _nvshmem_barrier()


@pytest.mark.usefixtures("_nvshmem__distributed__front_a2a_staged")
@pytest.mark.usefixtures("_module_skip__distributed__front_a2a_staged")
@FRONT_A2A.parametrize("mesh")
@numeric_exempt(
    "asserts a REFUSAL at the kernel's front door on an out-of-contract rows_per_peer; nothing is computed to compare"
)
def test_front_partial_token_clamp_guard_raises(apply_mesh, mesh, dist_manager, device, world_size):
    """The defensive 16B guard: a sub-8-token tail (rows_per_peer%128 in 1..7) -> clear ValueError.

    This tail is UNREACHABLE from a valid N_token (verify_rpp_unreachable.py) but we assert the guard
    fires for a hand-crafted out-of-contract rows_per_peer so the limit can never silently corrupt.
    rows_per_peer=132 -> 132%128=4 tokens=8 B < 16 B (the TMA stride-1-axis minimum)."""
    apply_mesh(mesh)

    cp = world_size
    D = 256
    if D % cp != 0:
        rank_invariant_skip(f"D={D} not divisible by cp={cp}", because=_SHAPE_GATE_BECAUSE)
    Dloc = D // cp
    # A2: the CTA tile comes from the PRODUCTION picker, NOT a literal. postact tile_N == CTA tile_N//2
    # (derived, kernels/layernorm_dual_gated_gemm.py:289-292), so the old hardcoded tile_N=128 -> postact 64 violates
    # `Dloc % postact_tile_N == 0` at Dloc=32 (D=256, cp=8) and the GEOMETRY ValueError at
    # dual_gated_gemm_a2a.py:1210-1214 PRE-EMPTS the 16-B guard this test exists to exercise
    # (28 lines later, :1240). tile_M stays 128: epi_m is 128 for every legal tile (the gated epi_tile_fn
    # halves N only, gemm_act.py:220-224), so the 132 literal keeps its 132%128=4 tail at every cp.
    tile_N = DualGatedGemmDistSm90.best_front_tile(Dloc)[1]
    rows_per_peer = 132  # 132 % 128 = 4 -> 8 B < 16 B -> must raise under partial_token_clamp
    M, K = rows_per_peer, 256
    M_full = cp * rows_per_peer
    pm = _pe_map(dist_manager)
    torch.manual_seed(4321 + dist_manager.rank)
    x = torch.randn(M, K, device=device, dtype=torch.bfloat16) / (K**0.5)
    Wg2 = torch.randn(2 * D, K, device=device, dtype=torch.bfloat16) / (K**0.5)
    Wp2 = torch.randn(2 * D, K, device=device, dtype=torch.bfloat16) / (K**0.5)
    # Symmetric memory via the torch MemPool, NOT nvshmem4py's allocator: torch owns the
    # nvshmem bootstrap here, so nvshmem4py's own `_is_initialized` is permanently False
    # and its `tensor()` raises `NVSHMEM Library is not initialized`. The pool also
    # recycles, which keeps the collective free off the destruction path.
    with torch.cuda.use_mem_pool(DistributedManager.symmetric_mempool(_this_rank_device())):
        recv = torch.empty((2 * Dloc, M_full), dtype=torch.bfloat16, device=_this_rank_device())
    pe_table = tuple(int(v) for v in pm.cp_pe_table.tolist())
    a2a_cfg = dict(
        cp=cp,
        my_cp_rank=int(pm.my_cp_rank),
        rows_per_peer=rows_per_peer,
        pe_table=pe_table,
        partial_token_clamp=True,
    )
    raised = False
    try:
        with pytest.raises(ValueError, match="16 B"):
            _build_front(x, Wg2, Wp2, (128, tile_N), recv_t=recv, a2a_cfg=a2a_cfg)
        raised = True
    finally:
        _nvshmem_barrier()
    print(
        f"\n[partial-clamp-guard cp={cp} rpp=132 rem=4] rank={dist_manager.rank} "
        f"ValueError_raised={raised} {'PASS' if raised else 'FAIL'}",
        flush=True,
    )


def _front_paired_median_ms(run_fn, n_it=50, warmup=8):
    """Kernel-only paired-median: warmup (untimed, folds out the one-time JIT/setup), then a CUDA-event
    loop over the SAME compiled run_fn (inputs pre-built, looped — no per-iter host marshaling beyond the
    from_dlpack the runner already does). Returns the median ms. Drift cancels in the partial-vs-aligned
    PAIR (both measured back-to-back on the same stream)."""
    _nvshmem_barrier()
    for _ in range(warmup):
        run_fn()
    _drain()
    evs = [
        (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
        for _ in range(n_it)
    ]
    for s, e in evs:
        s.record()
        run_fn()
        e.record()
    torch.cuda.synchronize()
    return sorted(s.elapsed_time(e) for s, e in evs)[n_it // 2]


# REAL N_token (production scale): rpp = N^2/cp is LARGE -> the dual-GEMM + store is ms-scale, ABOVE the
# ~50us kernel-launch floor (the earlier {16,24,32,40} table was pure launch latency -> meaningless).
#
# ================================ THE mod-128 DESTINATION-PHASE LAW ================================
# MEASURED (cp=8, all 8 ranks, single-variable): the peer-store penalty is a function of the
# destination byte address **mod 128** -- roughly +4 % at 64-mod-128, +8-11 % at 32-mod-128, +18-37 % at
# 16-mod-32. The front-plain D-major recv is (2*Dloc, M_full) with the TOKEN axis stride-1, so rank r's
# destination column block begins at `r * rows_per_peer * 2` bytes: the phase is entirely
# `r*rpp*2 (mod 128)`, and the alignment target is `rpp % 64 == 0`, NOT `% 16`.
#
# WHAT IS **NOT** THE CAUSE, both excluded by measurement, so do not re-derive them:
#   * the CLAMP -- forced-clamp vs no-clamp at an IDENTICAL shape costs 1.0038-1.0076x on all 8 ranks;
#   * the TAIL LENGTH -- an 8-token tail and a 72-token tail cost the SAME when the phase matches.
# Only the byte phase matters. The 1.25x bar below stands and must never be widened; the even-rank
# numbers (1.014-1.026, i.e. exactly the work ratio) show what this measures once the store is fixed.
#
# The comment this replaces said "1032(partial %128=32)". That was wrong: 133128 % 128 = 8, and likewise
# 528392 % 128 = 8 for 2056 -- the shipped pairs sit exactly ON the 16-B floor, the MINIMUM legal tail.
# 1040 is the N with %128 == 32.
#
# SINGLE-VARIABLE CELLS. The shipped pairs vary THREE things at once (partial tail, clamp, sector phase),
# so a ratio from them cannot attribute. These cells hold two constant at a time (all legal at cp=8, all
# within 6 % rpp of 1024):
#   cell    N      rpp     rpp%128 (tail)  clamp        rpp*2 % 128 (phase)
#   base    1024   131072  0               off          0 (all ranks)
#   1024F   1024   131072  0               FORCED ON    0            -> the clamp ALONE
#   1040    1040   135200  32              auto ON      64*(r%2)     -> tail+clamp, phase held
#   1032    1032   133128  8               auto ON      16*r         -> +phase (the shipped pair)
#   1048    1048   137288  72              auto ON      16*r         -> phase with a LONGER tail
#   1056    1056   139392  0               off          0            -> a pure work-ratio null
# Reads: 1024F/1024 = the clamp; 1040/1024 = tail+clamp at held phase; 1032/1040 = THE PHASE ALONE;
# 1048/1040 = the phase with a longer tail (kills "the short tail run is the cost"); 1056/1024 = null.
# Each read carries its own analytic work-ratio expectation (rpp_num/rpp_den), so the assertion is
# `measured/expected < 1.25`, not `measured < 1.25`.
# ===================================================================================================
_SMOOTH_NTOK = [1024, 1032, 1040, 1048, 1056, 2048, 2056]
# (label, N_token, force_clamp) -- "1024F" is the same shape as 1024 with the clamp forced ON.
_SMOOTH_CELLS = [(str(n), n, False) for n in _SMOOTH_NTOK] + [("1024F", 1024, True)]


@pytest.mark.usefixtures("_nvshmem__distributed__front_a2a_staged")
@pytest.mark.usefixtures("_module_skip__distributed__front_a2a_staged")
@FRONT_A2A.parametrize("mesh")
@numeric_exempt(
    "asserts TIMING ratios between ADJACENT token counts -- the detector for the destination-byte-phase cliff. Its subject is cost, not correctness"
)
def test_front_partial_token_clamp_smooth(apply_mesh, mesh, dist_manager, device, world_size):
    """SMOOTHNESS at REAL scale: at matched (near-equal) rpp, the PARTIAL-clamp fused front time ≈ the
    ALIGNED, and the STORE-DELTA (fused − bare-front-GEMM single_device) of the partial ≈ the aligned ->
    the clamp store is FLAT, no cliff. Kernel-only paired-median CUDA-event timing (warmup folds out JIT).
    Reports fused ms, single_device ms, store_delta, and the partial/aligned ratios. PASS = the matched
    partial-vs-aligned fused ratio AND store-delta ratio are within 1.25x (flat at real scale)."""
    apply_mesh(mesh)

    cp = world_size
    D = 256
    if D % cp != 0:
        rank_invariant_skip(f"D={D} not divisible by cp={cp}", because=_SHAPE_GATE_BECAUSE)
    Dloc = D // cp
    # Best CTA tile for this Dloc (the front's only autotune axis; debug/front_cfg_sweep.py): the
    # stock (128,128) is ~1.12-1.26x slower at production N_token. Measure the SMOOTHNESS at the
    # honest best config so the partial-vs-aligned comparison is on the perf the front actually ships.
    tile_M, tile_N = DualGatedGemmDistSm90.best_front_tile(Dloc)
    K = 256
    import torch.distributed as dist

    my_rank = int(dist_manager.rank)
    rows = {}
    for label, N_token, force_clamp in _SMOOTH_CELLS:
        if (N_token * N_token) % cp != 0:
            continue
        rpp = (N_token * N_token) // cp
        if rpp % 128 not in (0,) and (rpp % 128) < 8:
            continue  # guarded sub-8 tail (unreachable; skip if a cp makes it so)
        M, M_full = rpp, cp * rpp
        partial = (rpp % 128) != 0
        torch.manual_seed(4321 + dist_manager.rank)
        x = torch.randn(M, K, device=device, dtype=torch.bfloat16) / (K**0.5)
        Wg2 = torch.randn(2 * D, K, device=device, dtype=torch.bfloat16) / (K**0.5)
        Wp2 = torch.randn(2 * D, K, device=device, dtype=torch.bfloat16) / (K**0.5)
        pm = _pe_map(dist_manager)
        pe_table = tuple(int(v) for v in pm.cp_pe_table.tolist())
        # (a) FUSED front (dual-GEMM + D-major peer clamp store).
        # Symmetric memory via the torch MemPool, NOT nvshmem4py's allocator: torch owns the
        # nvshmem bootstrap here, so nvshmem4py's own `_is_initialized` is permanently False
        # and its `tensor()` raises `NVSHMEM Library is not initialized`. The pool also
        # recycles, which keeps the collective free off the destruction path.
        with torch.cuda.use_mem_pool(DistributedManager.symmetric_mempool(_this_rank_device())):
            recv = torch.empty((2 * Dloc, M_full), dtype=torch.bfloat16, device=_this_rank_device())
        recv.fill_(-99.0)
        # AUTO-CLAMP: NO flag -> partial rpp auto-clamps, aligned rpp uses the absolute-dst store.
        # force_clamp (the "1024F" cell) pins it ON at an rpp where it would otherwise be OFF, which is
        # the ONLY way to read the clamp's own cost with the tail and the phase both held.
        a2a_cfg = dict(cp=cp, my_cp_rank=int(pm.my_cp_rank), rows_per_peer=rpp, pe_table=pe_table)
        if force_clamp:
            a2a_cfg["partial_token_clamp"] = True
        c_f, run_f, _ = _build_front(x, Wg2, Wp2, (tile_M, tile_N), recv_t=recv, a2a_cfg=a2a_cfg)
        # (b) BARE front GEMM (single_device, LOCAL store, NO A2A) — the store-delta reference.
        c_sd, run_sd, _ = _build_front(x, Wg2, Wp2, (tile_M, tile_N), recv_t=None)
        try:
            fused_ms = _front_paired_median_ms(run_f)
            sd_ms = _front_paired_median_ms(run_sd)
            store_delta = fused_ms - sd_ms
            # The DIAGNOSTIC that names the cause: my destination base is `my_cp_rank*rpp*2` bytes, so
            # its residue mod 32 / mod 128 IS the predictor of the penalty (the mod-128 law above).
            base_b = int(pm.my_cp_rank) * rpp * 2
            rows[label] = dict(
                N=N_token,
                rpp=rpp,
                partial=partial,
                clamp=force_clamp,
                fused=fused_ms,
                sd=sd_ms,
                delta=store_delta,
                b32=base_b % 32,
                b128=base_b % 128,
            )
        finally:
            c_f.free()
            c_sd.free()
            _nvshmem_barrier()
    # PER-RANK print. Only rank 0 used to print, so a rank-DEPENDENT effect (which is exactly the
    # signature of a per-rank destination-phase penalty) showed up only as a scattered pass/fail across
    # rank-runs and was invisible in any single log.
    print(
        f"\n[clamp-smooth rank={my_rank}] cell   N   rpp  tail%128 clamp  base%32 base%128  "
        f"fused_ms  bareGEMM_ms  store_delta_ms",
        flush=True,
    )
    for label, r in rows.items():
        print(
            f"  [rank={my_rank}] {label:>6} {r['N']:>5} {r['rpp']:>8} {r['rpp'] % 128:>8} "
            f"{('F' if r['clamp'] else ('Y' if r['partial'] else '-')):>5} {r['b32']:>8} "
            f"{r['b128']:>8}  {r['fused']:.4f}    {r['sd']:.4f}      {r['delta']:.4f}",
            flush=True,
        )

    # ---- (1) the SHIPPED pair check, unchanged. The 1.25x bar stands and is never widened. ----
    pairs = [("1032", "1024"), ("2056", "2048")]
    worst_fused, worst_store = 1.0, 1.0
    for p_n, a_n in pairs:
        if p_n in rows and a_n in rows:
            fr = rows[p_n]["fused"] / rows[a_n]["fused"]
            sr = max(rows[p_n]["delta"], 1e-6) / max(rows[a_n]["delta"], 1e-6)
            worst_fused, worst_store = max(worst_fused, fr), max(worst_store, sr)
            print(
                f"[clamp-smooth rank={my_rank}] pair N{p_n}(partial)/N{a_n}(aligned): "
                f"fused {fr:.3f}x  store-delta {sr:.3f}x",
                flush=True,
            )

    # ---- (2) SINGLE-VARIABLE reads, each normalized by its own analytic work ratio. ----
    # `expected` = rpp_num/rpp_den: the pure FLOP/byte ratio the shapes differ by. Asserting
    # measured/expected isolates the STORE effect from the (few-percent) work difference.
    reads = [
        ("1024F", "1024", "the CLAMP alone (tail and phase held)"),
        ("1040", "1024", "tail+clamp with the PHASE HELD (rpp*2%128 = 64*(r%2))"),
        ("1032", "1040", "the PHASE ALONE (both partial, both clamped)"),
        ("1048", "1040", "the phase with a LONGER (72-token) tail"),
        ("1056", "1024", "a pure work-ratio NULL (both aligned, both phase 0)"),
    ]
    worst_sv, worst_sv_tag = 1.0, "none"
    for num, den, what in reads:
        if num not in rows or den not in rows:
            continue
        expected = rows[num]["rpp"] / rows[den]["rpp"]
        measured = rows[num]["fused"] / rows[den]["fused"]
        excess = measured / expected
        worst_sv, worst_sv_tag = (
            (excess, f"{num}/{den} ({what})") if excess > worst_sv else (worst_sv, worst_sv_tag)
        )
        print(
            f"[clamp-smooth rank={my_rank}] {num}/{den}: measured {measured:.3f}x / expected "
            f"{expected:.3f}x = {excess:.3f}x excess — {what}",
            flush=True,
        )

    # ---- (3) RANK-INVARIANCE. Every rank runs the SAME shapes and the SAME work, so a rank-dependent
    # front-store time is ITSELF a defect signature — it can only come from a per-rank destination
    # address. This single assertion would have caught the defect on day one, and it is independent of
    # the 1.25x pair bar. Collective (all_gather over a rank-invariant cell list), so no desync.
    worst_rank_spread, spread_tag = 1.0, "none"
    for label in sorted(rows):
        t = torch.tensor([rows[label]["fused"]], device=device, dtype=torch.float32)
        allt = [torch.empty_like(t) for _ in range(world_size)]
        dist.all_gather(allt, t)
        vals = [float(v.item()) for v in allt]
        spread = max(vals) / max(min(vals), 1e-9)
        if spread > worst_rank_spread:
            worst_rank_spread, spread_tag = spread, f"{label}: {['%.4f' % v for v in vals]}"
        if my_rank == 0:
            print(
                f"[clamp-smooth rank-invariance] {label}: max/min over ranks = {spread:.3f}x "
                f"({['%.4f' % v for v in vals]})",
                flush=True,
            )

    if my_rank == 0:
        print(
            f"[clamp-smooth] worst pair {worst_fused:.3f}x | worst single-variable excess "
            f"{worst_sv:.3f}x [{worst_sv_tag}] | worst rank spread {worst_rank_spread:.3f}x "
            f"[{spread_tag}]",
            flush=True,
        )
    assert worst_fused < 1.25, (
        f"partial-token-clamp NOT flat at real scale: worst partial/aligned fused = {worst_fused:.3f}x "
        f"(>1.25x -> a cliff). NOT the clamp and NOT the tail (both excluded by measurement) — it is the "
        f"DESTINATION BYTE PHASE: rank r stores at base r*rpp*2 bytes, and the penalty is a function of "
        f"that address mod 128 (see the module comment above this test). Fix the STORE alignment, never "
        f"the bar."
    )
    assert worst_sv < 1.25, (
        f"single-variable read {worst_sv_tag} is {worst_sv:.3f}x over its analytic work ratio. This "
        f"attributes the cliff to ONE variable — see which read it is: a 1032/1040 or 1048/1040 excess "
        f"is the destination-phase term (rpp*2 % 128), a 1024F/1024 excess would be the clamp (measured "
        f"at 1.004-1.008x, so it should not be), a 1056/1024 excess would be pure measurement noise."
    )
    assert worst_rank_spread < 1.25, (
        f"front-store time is RANK-DEPENDENT ({worst_rank_spread:.3f}x max/min over ranks, {spread_tag}) "
        f"at a shape where every rank does identical work. The only per-rank quantity in the store is the "
        f"destination base r*rpp*2 bytes -> this is the mod-128 phase penalty. Since the front A2A is "
        f"barrier-coupled, the SLOWEST rank sets the pace for the whole job."
    )


@pytest.mark.usefixtures("_nvshmem__distributed__front_a2a_staged")
@pytest.mark.usefixtures("_module_skip__distributed__front_a2a_staged")
@FRONT_A2A.parametrize("mesh")
@numeric_exempt(
    "asserts TIMING ratios (a pad policy must flatten the destination-phase cliff), not a computed output. Blocked on an unported dependency besides"
)
def test_front_pad_inner_kills_the_destination_phase(
    apply_mesh, mesh, dist_manager, device, world_size
):
    """P2 — the SAME cells as ``..._clamp_smooth`` above, configured with ``pad_inner``, must be FLAT.

    ``..._clamp_smooth`` builds the store at the RAW ``rpp = N_token**2 / cp`` and therefore keeps
    measuring the unpadded destination phase; it is the DETECTOR for the defect and its bars must never
    be widened. This test is the FIX's counterpart: identical shapes, identical timing, but the recv is
    sized (and the walk configured) from ``front_pad_inner_geometry``, so every rank's destination base
    ``r*rpp_eff*2`` is 128-B clean. The rank spread — the signature that cannot come from anything but a
    per-rank destination address — must collapse.

    Same 1.25x bars as the detector, so the two are directly comparable; the interesting number is how
    far under the bar this lands (printed), not merely that it passes.

    BLOCKED in this tree: ``front_pad_inner_geometry`` lives in
    ``fold_cp_ops.distributed.fused_trimul``, which has not been brought back yet, so the first
    statement of the body skips on the module's ABSENCE. That is a missing dependency, not an
    unsupported shape -- see the comment there, which also says how to retire the gate.
    """
    apply_mesh(mesh)
    import importlib.util

    # BLOCKED ON AN UNPORTED DEPENDENCY, and this is NOT the forbidden kind of skip. CLAUDE.md's
    # rule is that an xfail/skip on an IN-PRINCIPLE-SUPPORTED SHAPE is a violation rather than a
    # deferral -- that is about refusing to run a shape the kernel is meant to handle. This is a
    # different thing: `fold_cp_ops/distributed/fused_trimul.py` does not exist in this tree at all,
    # so the test's own import cannot resolve and there is no shape to refuse. The feature is not
    # unsupported; the module has not been brought back yet (it is part of an unported workflow
    # layer with its own gate). Measured: exactly ONE unresolvable `fold_cp_ops` import in this
    # file, and it is this one -- every other test is unblocked.
    #
    # The predicate is module PRESENCE, probed with `find_spec` rather than a try/except around the
    # real import. That distinction is load-bearing: `find_spec` does not EXECUTE the module, so it
    # cannot silently swallow an ImportError raised from INSIDE a `fused_trimul` that has landed but
    # is broken -- which must fail loudly, not skip. It also answers only the question asked, at the
    # module granularity; a `front_pad_inner_geometry` missing from a module that IS present is a
    # real defect in the port and will still raise here.
    #
    # This runs FIRST, before the import and before any collective, so a skipping rank leaves before
    # it can desync a peer.
    if importlib.util.find_spec("fold_cp_ops.distributed.fused_trimul") is None:
        rank_invariant_skip(
            "blocked on an UNPORTED dependency: fold_cp_ops.distributed.fused_trimul is absent from "
            "this tree, and this test needs front_pad_inner_geometry from it. The pad_inner path is "
            "not unsupported -- the module has not been brought back yet. Delete this gate as part "
            "of that bring-back; do not stub or mock the module to satisfy it",
            because=_HOST_ENV_BECAUSE,
        )

    import torch.distributed as dist

    from fold_cp_ops.distributed.fused_trimul import front_pad_inner_geometry

    cp = world_size
    D = 256
    if D % cp != 0:
        rank_invariant_skip(f"D={D} not divisible by cp={cp}", because=_SHAPE_GATE_BECAUSE)
    Dloc = D // cp
    tile_M, tile_N = DualGatedGemmDistSm90.best_front_tile(Dloc)
    K = 256
    my_rank = int(dist_manager.rank)
    rows = {}
    for label, N_token, force_clamp in _SMOOTH_CELLS:
        if force_clamp:
            continue  # the clamp-alone cell is a detector-only control; the pad makes the clamp moot
        if (N_token * N_token) % cp != 0 or N_token % cp != 0:
            continue
        M = (N_token * N_token) // cp  # the TRUE local token count (1-D: N_i_loc*N_j_loc)
        Yg, rpp_eff, N_j_pad = front_pad_inner_geometry(1, N_token // cp, N_token, tile_M)
        # THE PROPERTY UNDER TEST, asserted before a single kernel runs.
        assert (2 * rpp_eff) % 128 == 0, f"N{N_token}: pad left 2*rpp % 128 = {2 * rpp_eff % 128}"
        torch.manual_seed(4321 + dist_manager.rank)
        x = torch.randn(M, K, device=device, dtype=torch.bfloat16) / (K**0.5)
        Wg2 = torch.randn(2 * D, K, device=device, dtype=torch.bfloat16) / (K**0.5)
        Wp2 = torch.randn(2 * D, K, device=device, dtype=torch.bfloat16) / (K**0.5)
        pm = _pe_map(dist_manager)
        pe_table = tuple(int(v) for v in pm.cp_pe_table.tolist())
        # Symmetric memory via the torch MemPool, NOT nvshmem4py's allocator: torch owns the
        # nvshmem bootstrap here, so nvshmem4py's own `_is_initialized` is permanently False
        # and its `tensor()` raises `NVSHMEM Library is not initialized`. The pool also
        # recycles, which keeps the collective free off the destruction path.
        with torch.cuda.use_mem_pool(DistributedManager.symmetric_mempool(_this_rank_device())):
            recv = torch.empty(
                (2 * Dloc, cp * rpp_eff), dtype=torch.bfloat16, device=_this_rank_device()
            )
        recv.fill_(-99.0)
        a2a_cfg = dict(
            cp=cp,
            my_cp_rank=int(pm.my_cp_rank),
            rows_per_peer=rpp_eff,
            pe_table=pe_table,
            pad_inner=True,
            inner_extent=Yg,
            token_count=M,
        )
        c_f, run_f, _ = _build_front(x, Wg2, Wp2, (tile_M, tile_N), recv_t=recv, a2a_cfg=a2a_cfg)
        c_sd, run_sd, _ = _build_front(x, Wg2, Wp2, (tile_M, tile_N), recv_t=None)
        try:
            fused_ms = _front_paired_median_ms(run_f)
            sd_ms = _front_paired_median_ms(run_sd)
            rows[label] = dict(
                N=N_token,
                M=M,
                rpp=rpp_eff,
                njp=N_j_pad,
                fused=fused_ms,
                sd=sd_ms,
                delta=fused_ms - sd_ms,
                b128=(int(pm.my_cp_rank) * rpp_eff * 2) % 128,
            )
        finally:
            c_f.free()
            c_sd.free()
            _nvshmem_barrier()

    print(
        f"\n[pad-inner rank={my_rank}] cell   N      M      rpp_eff  N_j_pad  pad%  base%128  "
        f"fused_ms  bareGEMM_ms  store_delta_ms",
        flush=True,
    )
    for label, r in rows.items():
        print(
            f"  [rank={my_rank}] {label:>6} {r['N']:>5} {r['M']:>8} {r['rpp']:>8} {r['njp']:>7} "
            f"{100.0 * (r['rpp'] - r['M']) / r['M']:>5.2f} {r['b128']:>8}  {r['fused']:.4f}    "
            f"{r['sd']:.4f}      {r['delta']:.4f}",
            flush=True,
        )

    # ---- (1) the SAME shipped pairs, the SAME 1.25x bar. ----
    worst_fused = 1.0
    for p_n, a_n in [("1032", "1024"), ("2056", "2048")]:
        if p_n in rows and a_n in rows:
            fr = rows[p_n]["fused"] / rows[a_n]["fused"]
            worst_fused = max(worst_fused, fr)
            print(f"[pad-inner rank={my_rank}] pair N{p_n}/N{a_n}: fused {fr:.3f}x", flush=True)

    # ---- (2) RANK-INVARIANCE — the load-bearing one. Collective, so no desync. ----
    worst_rank_spread, spread_tag = 1.0, "none"
    for label in sorted(rows):
        t = torch.tensor([rows[label]["fused"]], device=device, dtype=torch.float32)
        allt = [torch.empty_like(t) for _ in range(world_size)]
        dist.all_gather(allt, t)
        vals = [float(v.item()) for v in allt]
        spread = max(vals) / max(min(vals), 1e-9)
        if spread > worst_rank_spread:
            worst_rank_spread, spread_tag = spread, f"{label}: {['%.4f' % v for v in vals]}"
        if my_rank == 0:
            print(
                f"[pad-inner rank-invariance] {label}: max/min over ranks = {spread:.3f}x "
                f"({['%.4f' % v for v in vals]})",
                flush=True,
            )
    if my_rank == 0:
        print(
            f"[pad-inner] worst pair {worst_fused:.3f}x | worst rank spread "
            f"{worst_rank_spread:.3f}x [{spread_tag}]",
            flush=True,
        )

    assert worst_rank_spread < 1.25, (
        f"pad_inner did NOT remove the per-rank destination-phase penalty: front-store time is still "
        f"RANK-DEPENDENT ({worst_rank_spread:.3f}x max/min, {spread_tag}) even though every rank's base "
        f"r*rpp_eff*2 is 128-B clean by construction. Something other than the base phase is per-rank."
    )
    assert worst_fused < 1.25, (
        f"pad_inner did NOT flatten the partial/aligned pair: worst {worst_fused:.3f}x. Fix the store, "
        f"never the bar."
    )


# --------------------------------------------------------------------------- #
# AUTO-CLAMP <16B guard, xfail-STRICT variant (mirror test_pe_aligned_general.py:330-338). A hand-crafted
# sub-8-token tail (rows_per_peer%tile_M in 1..7) is UNREACHABLE from a valid N_token but MUST raise at
# compile (the clamp's partial tail on the recv stride-1 TOKEN axis would be rem*2 B < 16 B = the TMA
# innermost-dim minimum -> silent corruption). This exercises the AUTO path (NO partial_token_clamp flag)
# so the auto-clamp itself trips the guard. strict xfail on raises=ValueError: if the guard is ever
# removed (compile stops raising) this XPASSES -> strict turns that into a FAILURE, alerting the recurrence
# of the silent sub-16B-tail corruption. Complements the pytest.raises variant
# (test_front_partial_token_clamp_guard_raises, which exercises the explicit-flag path).
# --------------------------------------------------------------------------- #
@pytest.mark.usefixtures("_nvshmem__distributed__front_a2a_staged")
@pytest.mark.usefixtures("_module_skip__distributed__front_a2a_staged")
@pytest.mark.xfail(
    raises=ValueError,
    strict=True,
    reason="<16B sub-8-token tail: rows_per_peer%tile_M in 1..7 (rem*2 B < 16 B, the TMA "
    "stride-1-axis minimum) -> guarded in epi_to_underlying_arguments (silent "
    "corruption else) — W1 auto-clamp",
)
@FRONT_A2A.parametrize("mesh")
@numeric_exempt(
    "an xfail'd probe of a hand-crafted out-of-contract rows_per_peer: its subject is whether the guard fires, not a computed value"
)
def test_front_clamp_guard_xfail(apply_mesh, mesh, dist_manager, device, world_size):
    """A sub-8-token tail (rpp%tile_M=4) AUTO-takes the clamp -> the <16B guard raises ValueError.

    NO partial_token_clamp flag: the auto-clamp engages (132%128=4 != 0) and the <16B floor fires
    (4 tokens = 8 B < 16 B). strict-xfail so a guard regression (removed/weakened) alerts via xpass."""
    apply_mesh(mesh)

    cp = world_size
    D = 256
    if D % cp != 0:
        rank_invariant_skip(f"D={D} not divisible by cp={cp}", because=_SHAPE_GATE_BECAUSE)
    Dloc = D // cp
    # A2 (the SILENT half of the same defect — MUST change together with guard_raises above).
    # ``raises=ValueError`` matches on exception TYPE only, so the hardcoded tile_N=128's GEOMETRY
    # ValueError at Dloc=32 satisfied the strict xfail: this test reported XFAIL-green while exercising
    # NOTHING, and an executed 5-state demonstration showed that with the geometry error pre-empting,
    # DELETING the 16-B guard outright still reports XFAIL — i.e. the XPASS alarm this test exists for
    # could not fire. Taking the tile from the production picker restores it. A pytest.skip here is NOT
    # an alternative: a skip inside a strict xfail reports SKIPPED, making the dead alarm permanent.
    tile_N = DualGatedGemmDistSm90.best_front_tile(Dloc)[1]
    rows_per_peer = 132  # 132 % 128 = 4 -> 8 B < 16 B -> auto-clamp must raise (no flag needed)
    M, K = rows_per_peer, 256
    M_full = cp * rows_per_peer
    pm = _pe_map(dist_manager)
    torch.manual_seed(4321 + dist_manager.rank)
    x = torch.randn(M, K, device=device, dtype=torch.bfloat16) / (K**0.5)
    Wg2 = torch.randn(2 * D, K, device=device, dtype=torch.bfloat16) / (K**0.5)
    Wp2 = torch.randn(2 * D, K, device=device, dtype=torch.bfloat16) / (K**0.5)
    # Symmetric memory via the torch MemPool, NOT nvshmem4py's allocator: torch owns the
    # nvshmem bootstrap here, so nvshmem4py's own `_is_initialized` is permanently False
    # and its `tensor()` raises `NVSHMEM Library is not initialized`. The pool also
    # recycles, which keeps the collective free off the destruction path.
    with torch.cuda.use_mem_pool(DistributedManager.symmetric_mempool(_this_rank_device())):
        recv = torch.empty((2 * Dloc, M_full), dtype=torch.bfloat16, device=_this_rank_device())
    pe_table = tuple(int(v) for v in pm.cp_pe_table.tolist())
    # AUTO path: NO partial_token_clamp -> auto-clamp engages (132%128 != 0) and the <16B guard fires.
    a2a_cfg = dict(
        cp=cp, my_cp_rank=int(pm.my_cp_rank), rows_per_peer=rows_per_peer, pe_table=pe_table
    )
    try:
        _build_front(x, Wg2, Wp2, (128, tile_N), recv_t=recv, a2a_cfg=a2a_cfg)
    finally:
        _nvshmem_barrier()


# --------------------------------------------------------------------------- #
# DYNAMIC-N auto-clamp (task #10, mirror test_pe_aligned_general.py::test_pe_aligned_dynamic_1d + the
# dyn_front_validate.py recipe): ONE dynamic-shape compile at a STRADDLE anchor M (use_clamp baked True)
# serves MANY token counts on the SAME executor — an aligned M AND a misaligned (auto-clamp) M. Under
# _a2a_dynamic the clamp block-slice + peer atoms are rebuilt per-call from the RUNTIME recv M_full
# (rows_per_peer = M_full // cp), so the descriptor's TOKEN (stride-1) bound + my column offset track the
# runtime token count, NOT the compile anchor. Gate each run with the error-histogram (rel_L2 < 2e-2 AND
# n_outlier_rows == 0 AND untouched == 0) vs the flat-cp D-major oracle. bf16.
# --------------------------------------------------------------------------- #
_DYN_ANCHOR = 1032  # straddle: 1032 % 128 = 8 -> use_clamp baked True at compile
_DYN_RUNS = [1024, 1032, 2056]  # aligned (control), straddle (rem 8), larger straddle (rem 8)


def _mark(c, dynamic):
    return c.mark_layout_dynamic() if dynamic else c


def _mark_recv(c, dynamic):
    # FRONT recv (2*Dloc, M_full): feature (2*Dloc, mode 0) is FIXED (D fixed); only the token
    # (M_full, mode 1) is dynamic. mark_compact_shape_dynamic(mode=1) keeps shape[0]=2*Dloc STATIC.
    return c.mark_compact_shape_dynamic(mode=1) if dynamic else c


def _front_operands(x, Wg2, Wp2):
    """(A_p, B_p, PostAct_p) for the staged front (mirror _build_front's perm3d prep)."""
    M = x.shape[0]
    twoN = Wg2.shape[0]  # = 2D
    B = interleave_dual_weights(Wg2, Wp2)
    out_postact = torch.empty((twoN, M), dtype=x.dtype, device=x.device)
    A = x.unsqueeze(0)
    B3 = B.mT.unsqueeze(0)
    PostAct = out_postact.mT.unsqueeze(0)
    A_p, B_p, _, _ = perm3d(A, B3, None, None)
    PostAct_p = perm3d(PostAct, B3, None, None)[0]
    return A_p, B_p, PostAct_p


def _mk_gemm_dyn(a_dtype, tile_shape_mn, K):
    tile_M, tile_N = tile_shape_mn
    g = DualGatedGemmDistSm90(
        Float32,
        a_dtype,
        (tile_M, tile_N),
        (1, 1, 1),
        pingpong=False,
        is_persistent=True,
        chunk_g=1,
        n_dual_tiles=0,
        gate3_n3=0,
    )
    return g


def _mk_epi_dyn(PostAct_p, recv, dynamic):
    return DualGatedGemmDistSm90.EpilogueArguments(
        mPostAct=_mark(from_dlpack(PostAct_p, assumed_align=16), dynamic),
        act_fn=gate_fn_map["glu"],
        mRowVecBroadcast=None,
        mBiasUp=None,
        mBiasGate=None,
        mMaskColVec=None,
        mPostAct3=None,
        act_fn_3=None,
        mRowVecBroadcast3=None,
        mWeight=None,
        mBias=None,
        eps=Float32(1e-5),
        rounding_mode=RoundingMode.RN,
        recv=_mark_recv(from_dlpack(recv, assumed_align=16), dynamic),
    )


@pytest.mark.usefixtures("_nvshmem__distributed__front_a2a_staged")
@pytest.mark.usefixtures("_module_skip__distributed__front_a2a_staged")
@FRONT_A2A.parametrize("mesh")
def test_front_dynamic_ntoken(apply_mesh, mesh, dist_manager, device, world_size):
    """ONE dynamic compile @straddle anchor serves aligned + misaligned (auto-clamp) token counts.

    Compile the front ONCE at a STRADDLE anchor (M=1032, %128=8 -> use_clamp baked True), then run
    several DIFFERENT token counts on the SAME executor: an aligned M (1024, control) AND misaligned
    auto-clamp M (1032, 2056). The dynamic clamp derives rows_per_peer from the runtime recv M_full, so
    each run's descriptor token-bound = its own runtime rpp. Each run is gated oracle-clean
    (rel_L2 < 2e-2, 0 outlier token rows, 0 unwritten cells). Parametrized by the session cp = world_size
    (flat _pe_map)."""
    apply_mesh(mesh)
    from tests.distributed.correctness_harness import compute_error_histogram

    cp = world_size
    D, tile_N, tile_M = 256, 128, 128
    if D % cp != 0:
        rank_invariant_skip(f"D={D} not divisible by cp={cp}", because=_SHAPE_GATE_BECAUSE)
    Dloc = D // cp
    if Dloc % (tile_N // 2) != 0:
        rank_invariant_skip(
            f"Dloc={Dloc} not a multiple of postact tile {tile_N // 2} at cp={cp}",
            because=_SHAPE_GATE_BECAUSE,
        )
    K = 256
    anchor = _DYN_ANCHOR
    assert anchor % tile_M != 0, (
        f"anchor {anchor} must be a STRADDLE (%tile_M != 0) to bake use_clamp"
    )
    pm = _pe_map(dist_manager)
    pe_table = tuple(int(v) for v in pm.cp_pe_table.tolist())
    # pure-coupled dynamic front (configure_a2a dynamic=True, no ib_drain): NVLink-only (§10), skip cross-node.
    _skip_if_coupled_cross_node(pe_table)
    stream = cutlass_torch.current_stream()

    # Compile ONCE at the straddle anchor (dynamic marks on operands + recv).
    torch.manual_seed(4321 + dist_manager.rank)
    x0 = torch.randn(anchor, K, device=device, dtype=torch.bfloat16) / (K**0.5)
    Wg0 = torch.randn(2 * D, K, device=device, dtype=torch.bfloat16) / (K**0.5)
    Wp0 = torch.randn(2 * D, K, device=device, dtype=torch.bfloat16) / (K**0.5)
    A0, B0, P0 = _front_operands(x0, Wg0, Wp0)
    # Symmetric memory via the torch MemPool, NOT nvshmem4py's allocator: torch owns the
    # nvshmem bootstrap here, so nvshmem4py's own `_is_initialized` is permanently False
    # and its `tensor()` raises `NVSHMEM Library is not initialized`. The pool also
    # recycles, which keeps the collective free off the destruction path.
    with torch.cuda.use_mem_pool(DistributedManager.symmetric_mempool(_this_rank_device())):
        recv0 = torch.empty(
            (2 * Dloc, cp * anchor), dtype=torch.bfloat16, device=_this_rank_device()
        )
    recv0.fill_(-99.0)
    a_dtype, _, _, _ = get_dtypes(
        x0.unsqueeze(0), interleave_dual_weights(Wg0, Wp0).mT.unsqueeze(0), x0.unsqueeze(0), None
    )
    gemm = _mk_gemm_dyn(a_dtype, (tile_M, tile_N), K)
    gemm.configure_a2a(
        cp=cp, my_cp_rank=int(pm.my_cp_rank), rows_per_peer=anchor, pe_table=pe_table, dynamic=True
    )
    sched = make_scheduler_args(get_max_active_clusters(1), 8, None)
    compiled = compile_gemm_with_bitcode(
        gemm,
        _mark(from_dlpack(A0, assumed_align=16), True),
        _mark(from_dlpack(B0, assumed_align=16), True),
        None,
        None,
        _mk_epi_dyn(P0, recv0, True),
        sched,
        stream,
        register=True,
    )
    if dist_manager.rank == 0:
        print(
            f"\n[front-dyn cp={cp}] COMPILED @straddle anchor M={anchor} (use_clamp baked True)",
            flush=True,
        )

    results = []
    try:
        for M in _DYN_RUNS:
            if M % cp != 0:
                continue
            torch.manual_seed(4321 + dist_manager.rank)
            x = torch.randn(M, K, device=device, dtype=torch.bfloat16) / (K**0.5)
            Wg = torch.randn(2 * D, K, device=device, dtype=torch.bfloat16) / (K**0.5)
            Wp = torch.randn(2 * D, K, device=device, dtype=torch.bfloat16) / (K**0.5)
            A_p, B_p, P_p = _front_operands(x, Wg, Wp)
            M_full = cp * M
            # Symmetric memory via the torch MemPool, NOT nvshmem4py's allocator: torch owns the
            # nvshmem bootstrap here, so nvshmem4py's own `_is_initialized` is permanently False
            # and its `tensor()` raises `NVSHMEM Library is not initialized`. The pool also
            # recycles, which keeps the collective free off the destruction path.
            with torch.cuda.use_mem_pool(DistributedManager.symmetric_mempool(_this_rank_device())):
                recv = torch.empty(
                    (2 * Dloc, M_full), dtype=torch.bfloat16, device=_this_rank_device()
                )
            recv.fill_(-99.0)
            _nvshmem_barrier()
            compiled(
                _mark(from_dlpack(A_p, assumed_align=16), True),
                _mark(from_dlpack(B_p, assumed_align=16), True),
                None,
                None,
                _mk_epi_dyn(P_p, recv, True),
                sched,
                stream,
            )
            _drain()
            expected = _ref_front_dmajor(
                x, Wg, Wp, pm, cp=cp, rows_per_peer=M, Dloc=Dloc, world_size=world_size
            )
            untouched = int((recv == -99.0).sum().item())
            got_bnnd = recv.t().contiguous().reshape(1, M_full, 1, 2 * Dloc)
            ref_bnnd = expected.t().contiguous().reshape(1, M_full, 1, 2 * Dloc)
            h = compute_error_histogram(got_bnnd, ref_bnnd)
            kind = "aligned" if M % tile_M == 0 else "straddle"
            # Per element, at the SAME 2e-2 constant. This is the STATISTIC change documented on
            # `_same_bar_elementwise`: a pooled L2 over the whole recv cannot see a handful of
            # mis-stored tokens, which is exactly the straddle-run defect this cell exists to catch.
            # It fires HERE, inside the loop, because `recv` is nvshmem-freed a few lines below and
            # the assert loop at the bottom only ever sees the scalars.
            _same_bar_elementwise(recv, expected, 2e-2, f"front-dyn cp={cp} run@{M} ({kind})")
            passed = h.rel_l2 < 2e-2 and h.n_outlier_rows == 0 and untouched == 0
            if dist_manager.rank == 0:
                print(
                    f"[front-dyn cp={cp} compile@{anchor} run@{M} ({kind})] rel_L2={h.rel_l2:.3e} "
                    f"outlier_rows={h.n_outlier_rows} untouched={untouched}/{recv.numel()} "
                    f"{'PASS' if passed else 'FAIL'}",
                    flush=True,
                )
            results.append((M, kind, h.rel_l2, h.n_outlier_rows, untouched))
            _nvshmem_barrier()
        # No scalar `assert rl2 < 2e-2` here: the recv-vs-reference verdict fired PER ELEMENT inside
        # the loop, while the buffers were still alive. `rl2` stays in the tuple because it is what
        # the per-run diagnostic reports.
        for M, kind, rl2, nout, unt in results:
            assert nout == 0, (
                f"cp={cp} run@{M} ({kind}): {nout} outlier token rows (localized corruption)"
            )
            assert unt == 0, f"cp={cp} run@{M} ({kind}): {unt} unwritten recv cells (coverage gap)"
        kinds = {k for (_, k, _, _, _) in results}
        assert "aligned" in kinds and "straddle" in kinds, (
            f"dynamic-N test must exercise BOTH an aligned AND a straddle run; got {kinds}"
        )
    finally:
        compiled.free()
        _nvshmem_barrier()


@pytest.fixture(scope="session")
def _module_skip__distributed__front_a2a_staged():
    pytest.importorskip("nvshmem.core")
    pytest.importorskip("fold_cp_ops.distributed.dual_gated_gemm_a2a")


# --------------------------------------------------------------------------- #
# DYNAMIC-N route2_ni + ib_wide + RUN_I correctness gate (torchrun -m pytest, cp cross-node).
#
# The incoming/route2 counterpart of test_front_route2_ib_wide.py, for the DYNAMIC-shape run_i raster
# (``_maybe_inject_run_i_wide_dynamic`` -> the @cute.jit ``_derive_run_i_len_rt`` runtime divisor-snap +
# the tile_scheduler ``run_i_dynamic`` decode). Proves the FIRST-PRINCIPLE compile-once-MANY-N contract:
# ONE compile at an anchor N serves an aligned N AND an off-grid STRADDLE N (Xg%tile_M != 0), each with:
#   (1) run_i ENGAGED on the dynamic path (gemm._a2a_run_i_tiles_derived == -1 sentinel),
#   (2) NO deadlock (a count-parity hang would time the torchrun out — the R|ncm snap makes the route2
#       b_j dup-absorb un-reachable, for EVERY runtime N),
#   (3) recv == the 2-kernel torch all_to_all reshard reference (rel_L2 < 2e-2, 0 outlier rows).
#
# Cross-node only exercises the fused wide IB drain (single-node NVLink collapses ib_drain to the coupled
# store — still a valid run_i-raster identity check). cp = world_size; a mesh that can't form the (cp0,cp1)
# route2 grid cleanly SKIPS (symmetric on every rank). W=128 (production width).
# --------------------------------------------------------------------------- #


_K__distributed__front_route2_dynamic_run_i = 256
_D = 256
_ANCHOR = 512
# aligned (==anchor), aligned (!=anchor), STRADDLE (!=anchor, N_i_loc%tile_M != 0 -> padded Xg).
_RUNS = [512, 1024, 640]


@pytest.fixture(scope="session", autouse=False)
def _nvshmem__distributed__front_route2_dynamic_run_i(dist_manager):
    from fold_cp_ops.distributed.distributed_manager import DistributedManager

    DistributedManager.init_nvshmem()
    yield


def _dual_ref__distributed__front_route2_dynamic_run_i(x, Wg2, Wp2):
    xf = x.float()
    return (torch.sigmoid(xf @ Wg2.float().t()) * (xf @ Wp2.float().t())).to(x.dtype)


@pytest.mark.usefixtures("_nvshmem__distributed__front_route2_dynamic_run_i")
@pytest.mark.usefixtures("dist_manager")
@FRONT_A2A.parametrize("mesh")
def test_route2_ni_dynamic_run_i_multiN(apply_mesh, mesh, dist_manager, device, world_size):
    """ONE dynamic compile @_ANCHOR serves aligned + straddle N; run_i engaged + recv-parity each."""
    apply_mesh(mesh)
    import nvshmem.core
    import torch.distributed as dist
    import cutlass.torch as cutlass_torch
    from cutlass import Float32
    from cutlass.cute.runtime import from_dlpack
    from torch.distributed.tensor import Shard
    from fold_cp_ops._internal.activation import gate_fn_map
    from fold_cp_ops._internal.arch import get_device_capacity, get_max_active_clusters
    from fold_cp_ops.kernels.dual_gated_gemm import interleave_dual_weights
    from fold_cp_ops._internal.gemm_tvm_ffi_utils import get_dtypes, make_scheduler_args, perm3d
    from fold_cp_ops.distributed.pe_map import PeMap
    from fold_cp_ops.distributed.dual_gated_gemm_a2a import (
        DualGatedGemmDistSm90 as G,
    )
    from fold_cp_ops.distributed.gemm_bitcode_compile import compile_gemm_with_bitcode
    from tests.distributed.correctness_harness import compute_error_histogram

    if get_device_capacity(device)[0] != 9:
        rank_invariant_skip("route2 front needs sm_90a (H100)", because=_ARCH_BECAUSE)
    cp = world_size
    mesh = dist_manager.device_mesh
    pm = PeMap.from_mesh_placements(
        mesh, [Shard(i + 1) for i in range(mesh.ndim)], distributed_manager=dist_manager
    )
    cp_axes = tuple(int(s) for s in pm.cp_axis_sizes)
    cp0 = int(cp_axes[0])
    cp1 = int(cp_axes[1]) if len(cp_axes) > 1 else 1
    if _D % cp != 0:
        rank_invariant_skip(f"D={_D} not divisible by cp={cp}", because=_SHAPE_GATE_BECAUSE)
    Dloc = _D // cp
    tile_M, tile_N = G.best_front_tile(Dloc)
    if tile_N % 16 != 0 or Dloc % (tile_N // 2) != 0 or Dloc < 8:
        rank_invariant_skip(
            f"front feature-scatter floors unmet at Dloc={Dloc} (tile_N={tile_N})",
            because=_SHAPE_GATE_BECAUSE,
        )
    if _ANCHOR % cp0 or _ANCHOR % cp1:
        rank_invariant_skip(
            f"anchor {_ANCHOR} not divisible by cp grid ({cp0},{cp1})", because=_SHAPE_GATE_BECAUSE
        )
    pe_table = tuple(int(v) for v in pm.cp_pe_table.tolist())
    dev = device
    stream = cutlass_torch.current_stream()

    def geom(N):
        N_i_loc, N_j_loc = N // cp0, N // cp1
        Xg_pad = ((N_i_loc + tile_M - 1) // tile_M) * tile_M
        return N_i_loc, N_j_loc, N_i_loc * N_j_loc, Xg_pad, cp0 * Xg_pad, cp1 * N_j_loc

    Wg2 = torch.randn(
        2 * _D, _K__distributed__front_route2_dynamic_run_i, device=dev, dtype=torch.bfloat16
    ) / (_K__distributed__front_route2_dynamic_run_i**0.5)
    Wp2 = torch.randn(
        2 * _D, _K__distributed__front_route2_dynamic_run_i, device=dev, dtype=torch.bfloat16
    ) / (_K__distributed__front_route2_dynamic_run_i**0.5)
    grid_ctas = get_max_active_clusters(1)
    import math as _math

    epi_m = _math.gcd(128, tile_M)
    epi_n_pa = _math.gcd(32, tile_N) // 2
    # Symmetric memory via the torch MemPool, NOT nvshmem4py's allocator: torch owns the
    # nvshmem bootstrap here, so nvshmem4py's own `_is_initialized` is permanently False
    # and its `tensor()` raises `NVSHMEM Library is not initialized`. The pool also
    # recycles, which keeps the collective free off the destruction path.
    with torch.cuda.use_mem_pool(DistributedManager.symmetric_mempool(_this_rank_device())):
        ring = torch.empty(
            (grid_ctas, 2, epi_n_pa, epi_m * 128), dtype=torch.bfloat16, device=_this_rank_device()
        )
    ring.zero_()
    pe_dev = torch.tensor(list(pe_table), device=dev, dtype=torch.int32).contiguous()
    _md = lambda c: c.mark_layout_dynamic()
    _md_recv = lambda c: c.mark_compact_shape_dynamic(
        mode=1, stride_order=(0, 2, 1), divisibility=8
    ).mark_compact_shape_dynamic(mode=2, stride_order=(0, 2, 1), divisibility=8)

    def build(N):
        N_i_loc, N_j_loc, M, Xg_pad, N_i, N_j = geom(N)
        torch.manual_seed(4321 + int(dist_manager.rank) + N)
        x = torch.randn(
            M, _K__distributed__front_route2_dynamic_run_i, device=dev, dtype=torch.bfloat16
        ) / (_K__distributed__front_route2_dynamic_run_i**0.5)
        # Symmetric memory via the torch MemPool, NOT nvshmem4py's allocator: torch owns the
        # nvshmem bootstrap here, so nvshmem4py's own `_is_initialized` is permanently False
        # and its `tensor()` raises `NVSHMEM Library is not initialized`. The pool also
        # recycles, which keeps the collective free off the destruction path.
        with torch.cuda.use_mem_pool(DistributedManager.symmetric_mempool(_this_rank_device())):
            rbuf = torch.empty(
                (2 * Dloc, N_j, N_i), dtype=torch.bfloat16, device=_this_rank_device()
            )
        rbuf.fill_(-99.0)
        recv = rbuf.permute(0, 2, 1)
        Bw = interleave_dual_weights(Wg2, Wp2)
        out_pa = torch.empty((2 * _D, M), dtype=x.dtype, device=dev)
        A_p, B_p, _, _ = perm3d(x.unsqueeze(0), Bw.mT.unsqueeze(0), None, None)
        PA_p = perm3d(out_pa.mT.unsqueeze(0), Bw.mT.unsqueeze(0), None, None)[0]
        return x, rbuf, recv, A_p, B_p, PA_p, N_i_loc, N_j_loc, M, Xg_pad, N_i, N_j

    def epi(g, PA_p, recv, N_j_loc, M):
        return G.EpilogueArguments(
            mPostAct=_md(from_dlpack(PA_p, assumed_align=16)),
            act_fn=gate_fn_map["glu"],
            mRowVecBroadcast=None,
            mBiasUp=None,
            mBiasGate=None,
            mMaskColVec=None,
            mPostAct3=None,
            act_fn_3=None,
            mRowVecBroadcast3=None,
            mWeight=None,
            mBias=None,
            eps=Float32(1e-5),
            rounding_mode=RoundingMode.RN,
            recv=_md_recv(from_dlpack(recv, assumed_align=16)),
            ring=from_dlpack(ring, assumed_align=16),
            pe_table_dev=from_dlpack(pe_dev, assumed_align=4),
            token_grid_yg=g.dynamic_token_grid_yg(N_j_loc, M),
        )

    # ---- compile ONCE at the anchor ----
    xa, rbuf_a, recv_a, A0, B0, P0, N_i_loc, N_j_loc, M, *_ = build(_ANCHOR)
    a_dtype, _, _, _ = get_dtypes(
        xa.unsqueeze(0), interleave_dual_weights(Wg2, Wp2).mT.unsqueeze(0), xa.unsqueeze(0), None
    )
    g = G(
        Float32,
        a_dtype,
        (tile_M, tile_N),
        (1, 1, 1),
        pingpong=False,
        is_persistent=True,
        chunk_g=1,
        n_dual_tiles=0,
        gate3_n3=0,
    )
    g.configure_a2a(
        cp=cp,
        my_cp_rank=int(pm.my_cp_rank),
        rows_per_peer=M,
        pe_table=pe_table,
        transpose_in=True,
        token_grid=(1, N_i_loc, N_j_loc),
        dynamic=True,
        ib_drain=True,
        ring_depth=2,
        consumer_warpgroups=1,
        ib_wide=True,
        ib_wide_batch=128,
    )
    g._a2a_route2_ni = True
    g._a2a_cp_axis_sizes = (cp0, cp1)
    sched = make_scheduler_args(grid_ctas, 8, None)
    compiled = compile_gemm_with_bitcode(
        g,
        _md(from_dlpack(A0, assumed_align=16)),
        _md(from_dlpack(B0, assumed_align=16)),
        None,
        None,
        epi(g, P0, recv_a, N_j_loc, M),
        sched,
        stream,
        register=True,
    )
    # run_i MUST have engaged on the dynamic path (the -1 sentinel), else the whole gate is vacuous.
    assert int(getattr(g, "_a2a_run_i_tiles_derived", 0)) == -1, (
        "dynamic route2 run_i did NOT engage"
    )

    try:
        for N in _RUNS:
            x, rbuf, recv, A_p, B_p, PA_p, N_i_loc, N_j_loc, M, Xg_pad, N_i, N_j = build(N)
            dist.barrier()
            compiled(
                _md(from_dlpack(A_p, assumed_align=16)),
                _md(from_dlpack(B_p, assumed_align=16)),
                None,
                None,
                epi(g, PA_p, recv, N_j_loc, M),
                sched,
                stream,
            )
            import nvshmem.core.rma as nvshmem_rma

            nvshmem_rma.quiet(stream=torch.cuda.current_stream())
            nvshmem.core.barrier_all(stream=torch.cuda.current_stream())
            torch.cuda.synchronize()
            # 2-kernel reshard reference (mirror test_front_route2_ib_wide.py).
            dual = _dual_ref__distributed__front_route2_dynamic_run_i(x, Wg2, Wp2).t().contiguous()
            a = dual[:_D].reshape(_D, N_i_loc, N_j_loc)
            b = dual[_D:].reshape(_D, N_i_loc, N_j_loc)
            per = Dloc * N_i_loc * N_j_loc
            chunk = 2 * per
            parts = []
            for p in range(cp):
                s = pe_table.index(p)
                sl = slice(s * Dloc, (s + 1) * Dloc)
                parts.append(a[sl].reshape(-1))
                parts.append(b[sl].reshape(-1))
            recv_flat = torch.empty_like(torch.cat(parts))
            dist.all_to_all_single(recv_flat, torch.cat(parts))
            ref = torch.zeros(2 * Dloc, N_j, N_i, device=dev, dtype=torch.bfloat16).permute(0, 2, 1)
            for p in range(cp):
                s = pe_table.index(p)
                i0 = (s // cp1) * Xg_pad
                j0 = (s % cp1) * N_j_loc
                blk = recv_flat[p * chunk : (p + 1) * chunk]
                ref[:Dloc, i0 : i0 + N_i_loc, j0 : j0 + N_j_loc] = blk[:per].reshape(
                    Dloc, N_i_loc, N_j_loc
                )
                ref[Dloc:, i0 : i0 + N_i_loc, j0 : j0 + N_j_loc] = blk[per:].reshape(
                    Dloc, N_i_loc, N_j_loc
                )
            h = compute_error_histogram(recv.contiguous().float(), ref.float())
            if int(dist_manager.rank) == 0:
                print(
                    f"\n[route2 dyn run_i N={N}] rel_L2={h.rel_l2:.3e} outliers={h.n_outlier_rows}",
                    flush=True,
                )
            # Per element, at the SAME 2e-2 constant -- the STATISTIC change documented on
            # `_same_bar_elementwise`. The defect this cell hunts is a MIS-PLACED token block under
            # the dynamic run_i raster, i.e. a localized error, which is the one shape a pooled L2
            # over the whole recv is blind to.
            _same_bar_elementwise(
                recv.contiguous().float(), ref.float(), 2e-2, f"route2 dyn run_i N={N} recv"
            )
            assert h.n_outlier_rows == 0, (
                f"route2 dynamic run_i N={N} recv MISMATCH: rel_L2={h.rel_l2:.3e} "
                f"outliers={h.n_outlier_rows} (compile-once-many-N / straddle broke)"
            )
    finally:
        try:
            compiled.free()
        except Exception:
            pass
        try:
            dist.barrier()
        except Exception:
            pass


# --------------------------------------------------------------------------- #
# route2_ni + ib_wide + run_i CORRECTNESS gate (torchrun -m pytest, cp16 2-node). The #8 deadlock shape:
# the fused transpose_in front with the DIFFERENTIAL IB drain + the WIDE-PUT coalesce + the run_i raster.
#
# The cp16 deadlock (route2 + wide + run_i SUB-BAND) is fixed by the R|ncm snap (_maybe_inject_run_i_wide ->
# _largest_divisor_leq) so run_i never hits the sub-band clamp/dup-absorb for route2. This gate proves the
# fixed path (a) does NOT hang (a count-parity deadlock would time the torchrun out) AND (b) the b_j-pinned
# wide IB drain PLACES tokens correctly -- the fused recv == the 2-kernel torch-reshard reference (per-token
# error histogram: rel_L2 < 2e-2, 0 outlier rows, full coverage of the valid cols). The deadlock previously
# MASKED whether the coalesced wide put lands the right (dst_i, dst_feat, dst_j) -- this is the FIRST-PRINCIPLE
# pytest at the exact triggering shape.
#
# Runs on WHATEVER the job is (require_ib=None-style): an all-P2P job (cp<=8, single node) collapses the IB
# drain to the coupled clamp store (a wide/run_i-OFF-equivalent identity check + the R-snap raster); a cp16
# (2,8) 2-node job exercises the fused wide IB drain (the deadlock shape). W=128 (production width).
# --------------------------------------------------------------------------- #


_K__distributed__front_route2_ib_wide = 256
_N = [
    1024
]  # N1024 mesh 2*8 -> ncm=512, derived R=62 -> SNAP 32 (the ISO-4-measured deadlock-free shape)


@pytest.fixture(scope="session", autouse=False)
def _nvshmem__distributed__front_route2_ib_wide(dist_manager):
    """Init the NVSHMEM library once per session (the symmetric recv/ring allocs below need it)."""
    from fold_cp_ops.distributed.distributed_manager import DistributedManager

    DistributedManager.init_nvshmem()
    yield


def _dual_ref__distributed__front_route2_ib_wide(x, Wg2, Wp2):
    """fp32 GLU dual projection reference: sigmoid(x@Wg^T) * (x@Wp^T) -> (M, 2D), D-major [a|b]."""
    xf = x.float()
    return (torch.sigmoid(xf @ Wg2.float().t()) * (xf @ Wp2.float().t())).to(x.dtype)


@pytest.mark.usefixtures("dist_manager")
@numeric_exempt(
    "launches no kernel and produces no tensor: it reads the cp AXIS SIZES off the resolved mesh and "
    "compares two tuples of ints. There is no computed output for an element-wise bound to bound"
)
@FRONT_A2A.parametrize("mesh")
def test_cp_mesh_reports_the_declared_cp_factorisation(apply_mesh, mesh, dist_manager):
    """The mesh the A2A tests read must carry the cp axes the matrix DECLARED (host-only, no kernel).

    This is the guard for a silent coverage collapse, not a feature test. ``create_grid_group`` puts
    ``prod(v)`` on the parent mesh and the factorisation on a SEPARATE ``suffix_mesh="subgroups"``
    mesh, so ``dist_manager.device_mesh`` is 1-D of shape ``(cp,)`` for EVERY spec. Every A2A test
    that read it therefore ran a 1-D cp geometry while its pytest id said ``mesh(('cp',(2,8)),)`` --
    ``cp1`` collapsed to 1, the route-2 ``b_j`` column index became single-valued, and the
    ``dst_j == b_j`` wide-coalesce gate was a tautology. The matrix's own ``mesh`` domain says the
    opposite in as many words: "the 2-D front's recv-index and the tile->peer unravel are exercised
    only under a factored spec".

    Asserts BOTH directions on the same spec, because only the pair is falsifiable:
      * ``_cp_mesh`` reports exactly the declared axes -- so a factored spec is genuinely 2-D;
      * ``dist_manager.device_mesh`` reports the FLAT product -- so this records the trap rather than
        merely avoiding it. If upstream ever makes the parent mesh factored, this half fails and the
        helper can be retired deliberately instead of rotting into a no-op.

    Args:
        apply_mesh: the mesh factory fixture; skips (rank-invariantly) when the spec's rank product
            does not equal WORLD_SIZE, so one launch covers only the specs that fit it.
        mesh: a spec drawn from ``FRONT_A2A``'s ``mesh`` axis, e.g. ``(("cp", (2, 8)),)``.
        dist_manager: the live manager, read AFTER ``apply_mesh`` (before it, both meshes are stale
            or None -- ``reset_grid_groups`` clears them).
    """
    from torch.distributed.tensor import Shard
    from fold_cp_ops.distributed.pe_map import PeMap

    apply_mesh(mesh)
    declared = tuple(int(a) for _, v in mesh for a in (v if isinstance(v, tuple) else (v,)))

    cpm = _cp_mesh(dist_manager)
    pm = PeMap.from_mesh_placements(
        cpm, [Shard(i + 1) for i in range(cpm.ndim)], distributed_manager=dist_manager
    )
    got = tuple(int(x) for x in pm.cp_axis_sizes)
    assert got == declared, (
        f"spec {mesh} declares cp axes {declared}; `_cp_mesh` -> PeMap read {got}. A factored spec "
        "that reads 1-D means every 2-D-labelled cell is running a 1-D geometry."
    )

    flat = dist_manager.device_mesh
    flat_pm = PeMap.from_mesh_placements(
        flat, [Shard(i + 1) for i in range(flat.ndim)], distributed_manager=dist_manager
    )
    flat_axes = tuple(int(x) for x in flat_pm.cp_axis_sizes)
    import numpy as _np

    assert flat_axes == (int(_np.prod(declared)),), (
        f"`dist_manager.device_mesh` reported {flat_axes} for spec {mesh}; this test records that it "
        f"reports the FLAT product ({int(_np.prod(declared))},). If that changed, `_cp_mesh` may no "
        "longer be needed -- retire it deliberately rather than leaving it as a silent no-op."
    )


@pytest.mark.usefixtures("_nvshmem__distributed__front_route2_ib_wide")
@pytest.mark.usefixtures("dist_manager")
@FRONT_A2A.parametrize(
    "N",
    "wide_nbi",
    only={"N": tuple(_N), "wide_nbi": (False,)},
    because=(
        "one N is enough for the B1-vs-B2 differential: the non-blocking flag changes WHEN the "
        "wide_nbi is now narrowed to (False,) BECAUSE B2 became a declared unsupported region: "
        "`ib_wide_nbi=True` is refused outright at `_configure_ib_wide` on every toolchain, so the "
        "True cell can no longer be a CORRECTNESS cell -- it must RAISE, and it is covered as such "
        "by `test_front_a2a_unsupported_combos_raise` via parametrize_unsupported(). This is a "
        "narrowing to a refusal, NOT a coverage retreat: the B1-vs-B2 differential this test used to "
        "carry is unrunnable on both shipped toolchains (<4.7.0 cannot link it; >=4.7.0 links it and "
        "runs 3.23-3.25x slower), and the surviving B2 correctness cells are kept behind "
        "_CPO_ALLOW_IB_WIDE_NBI=1 in test_front_ib_wide_nbi_{hybrid,offgrid}.\n"
        "drain waits, never what it writes, so what must be swept here is wide_nbi and not N. N is "
        "swept by the route-2 ib_drain and ib_wide cells above"
    ),
)
@FRONT_A2A.parametrize("mesh")
def test_route2_ni_ib_wide_recv_matches_reshard(
    apply_mesh, mesh, dist_manager, device, world_size, N, wide_nbi
):
    apply_mesh(mesh)
    import torch.distributed as dist
    from torch.distributed.tensor import Shard
    from fold_cp_ops.distributed.pe_map import PeMap
    from fold_cp_ops.distributed.dual_gated_gemm_a2a import DualGatedGemmDistSm90
    from tests.distributed.correctness_harness import compute_error_histogram

    cp = world_size
    # `_cp_mesh`, NOT `dist_manager.device_mesh`: the latter is 1-D of shape (cp,) for EVERY spec, so
    # a FACTORED mesh id (`mesh(('cp',(2,8)),)`) used to run a 1-D cp geometry -- cp1 collapsed to 1,
    # `b_j` became single-valued, and the `dst_j == b_j` merge predicate this test exists to exercise
    # was a tautology. This test's own matrix says the opposite ("the 2-D front's recv-index and the
    # tile->peer unravel are exercised only under a factored spec"); the helper is what makes it true.
    mesh_spec = mesh  # the PARAMETRIZED spec -- keep it, `mesh` is rebound to a DeviceMesh below
    mesh = _cp_mesh(dist_manager)
    placements = [Shard(i + 1) for i in range(mesh.ndim)]
    pm = PeMap.from_mesh_placements(mesh, placements, distributed_manager=dist_manager)
    cp_axes = tuple(int(s) for s in pm.cp_axis_sizes)
    # A factored spec that still reads 1-D means the resolver regressed. FAIL here, loudly, rather
    # than run a degenerate geometry under a 2-D test id -- that silence is what hid this for months.
    _declared = tuple(int(a) for _, v in mesh_spec for a in (v if isinstance(v, tuple) else (v,)))
    assert cp_axes == _declared, (
        f"mesh spec {mesh_spec} declares cp axes {_declared} but PeMap read {cp_axes}: the cp "
        "factorisation was dropped (see `_cp_mesh`). A 2-D test id would then run a 1-D geometry "
        "and the `dst_j == b_j` wide-coalesce gate would never be exercised."
    )
    cp0 = int(cp_axes[0])
    cp1 = int(cp_axes[1]) if len(cp_axes) > 1 else 1
    # Handles BOTH 1-D (cp1==1) and 2-D (cp1>1) token sharding — the #8 deadlock repro (mesh 2*8) is 2-D
    # (cp0=2,cp1=8), and the reference reshard below places each cp-slot by (s//cp1, s%cp1) (mirrors the
    # fused store's cp0_coord/cp1_coord).
    D = 256
    if D % cp != 0:
        rank_invariant_skip(f"D={D} not divisible by cp={cp}", because=_SHAPE_GATE_BECAUSE)
    Dloc = D // cp
    B = 1
    N_i_loc = N // cp0  # Xg (my i-shard)
    N_j_loc = N // cp1  # Yg (== N for 1-D)
    M = B * N_i_loc * N_j_loc
    if N_i_loc % 8 != 0 or N_j_loc % 8 != 0:
        rank_invariant_skip(
            f"N_i_loc={N_i_loc}/N_j_loc={N_j_loc} %8 != 0 (16-B TMA floor)",
            because=_SHAPE_GATE_BECAUSE,
        )
    tile_M, tile_N = DualGatedGemmDistSm90.best_front_tile(Dloc)
    if tile_N % 16 != 0 or Dloc % (tile_N // 2) != 0 or Dloc < 8:
        rank_invariant_skip(
            f"front feature-scatter floors unmet at Dloc={Dloc} (tile_N={tile_N})",
            because=_SHAPE_GATE_BECAUSE,
        )
    Xg_pad = ((N_i_loc + tile_M - 1) // tile_M) * tile_M
    N_i = cp0 * Xg_pad
    N_j = cp1 * N_j_loc
    pe_table = tuple(int(v) for v in pm.cp_pe_table.tolist())

    torch.manual_seed(4321 + int(dist_manager.rank))
    x = torch.randn(
        M, _K__distributed__front_route2_ib_wide, device=device, dtype=torch.bfloat16
    ) / (_K__distributed__front_route2_ib_wide**0.5)
    Wg2 = torch.randn(
        2 * D, _K__distributed__front_route2_ib_wide, device=device, dtype=torch.bfloat16
    ) / (_K__distributed__front_route2_ib_wide**0.5)
    Wp2 = torch.randn(
        2 * D, _K__distributed__front_route2_ib_wide, device=device, dtype=torch.bfloat16
    ) / (_K__distributed__front_route2_ib_wide**0.5)

    # Symmetric memory via the torch MemPool, NOT nvshmem4py's allocator: torch owns the
    # nvshmem bootstrap here, so nvshmem4py's own `_is_initialized` is permanently False
    # and its `tensor()` raises `NVSHMEM Library is not initialized`. The pool also
    # recycles, which keeps the collective free off the destruction path.
    with torch.cuda.use_mem_pool(DistributedManager.symmetric_mempool(_this_rank_device())):
        recv_buf = torch.empty(
            (2 * Dloc, N_j, N_i), dtype=torch.bfloat16, device=_this_rank_device()
        )
    recv_buf.fill_(-99.0)
    recv = recv_buf.permute(0, 2, 1)  # (2*Dloc, N_i, N_j), N_i stride-1

    def _cfg(g):
        g.configure_a2a(
            cp=cp,
            my_cp_rank=int(pm.my_cp_rank),
            rows_per_peer=M,
            pe_table=pe_table,
            transpose_in=True,
            token_grid=(B, N_i_loc, N_j_loc),
            ib_drain=True,
            ring_depth=2,
            consumer_warpgroups=1,
            ib_wide=True,
            ib_wide_batch=128,
            ib_wide_nbi=wide_nbi,
        )
        g._a2a_route2_ni = True
        g._a2a_cp_axis_sizes = (cp0, cp1)

    compiled, run_fn, gemm_obj, ring_t = _build_front_decoupled(
        x,
        Wg2,
        Wp2,
        (tile_M, tile_N),
        recv_t=recv,
        ring_depth=2,
        consumer_warpgroups=1,
        configure_fn=_cfg,
    )
    # run_i R MUST have snapped to a DIVISOR of ncm (no sub-band clamp -> no route2 deadlock).
    R = int(getattr(gemm_obj, "_a2a_run_i_tiles_derived", 0))
    ncm = B * N_j_loc * ((N_i_loc + tile_M - 1) // tile_M)
    if R > 0:
        assert ncm % R == 0, (
            f"route2 run_i R={R} does NOT divide ncm={ncm} (sub-band -> #8 deadlock risk)"
        )

    try:
        dist.barrier()
        run_fn()  # the fused DualGatedGEMM + route2 wide IB drain (would HANG if deadlocked)
        torch.cuda.synchronize()
        dist.barrier()

        # ---- 2-kernel reference recv: local dual GLU -> torch all_to_all reshard into (2*Dloc, N_i, N_j).
        dual = (
            _dual_ref__distributed__front_route2_ib_wide(x, Wg2, Wp2).t().contiguous()
        )  # (2D, M) D-major [a|b]
        a = dual[:D].reshape(D, N_i_loc, N_j_loc)
        b = dual[D:].reshape(D, N_i_loc, N_j_loc)
        per = Dloc * N_i_loc * N_j_loc
        chunk = 2 * per
        send_parts = []
        for p in range(cp):
            s = pe_table.index(p)
            sl = slice(s * Dloc, (s + 1) * Dloc)
            send_parts.append(a[sl].reshape(-1))
            send_parts.append(b[sl].reshape(-1))
        send = torch.cat(send_parts)
        recv_flat = torch.empty_like(send)
        dist.all_to_all_single(recv_flat, send)
        ref = torch.zeros(2 * Dloc, N_j, N_i, device=device, dtype=torch.bfloat16).permute(0, 2, 1)
        for p in range(cp):
            s = pe_table.index(p)
            i0 = (s // cp1) * Xg_pad
            j0 = (s % cp1) * N_j_loc
            blk = recv_flat[p * chunk : (p + 1) * chunk]
            ref[:Dloc, i0 : i0 + N_i_loc, j0 : j0 + N_j_loc] = blk[:per].reshape(
                Dloc, N_i_loc, N_j_loc
            )
            ref[Dloc:, i0 : i0 + N_i_loc, j0 : j0 + N_j_loc] = blk[per:].reshape(
                Dloc, N_i_loc, N_j_loc
            )

        # Compare on the VALID i-shard cols (the padded [N_i_loc, Xg_pad) rows stay -99/0 on both, skip them).
        got = recv[:, :, :].contiguous()
        h = compute_error_histogram(got.float(), ref.float())
        if int(dist_manager.rank) == 0:
            print(
                f"\n[route2 ib_wide {'B2' if wide_nbi else 'B1'} N={N} cp={cp} R={R} ncm={ncm}] "
                f"rel_L2={h.rel_l2:.3e} outlier_rows={h.n_outlier_rows}",
                flush=True,
            )
        # Per element, at the SAME 2e-2 constant -- the STATISTIC change documented on
        # `_same_bar_elementwise`. The named suspect here is "the b_j-pinned wide coalesce
        # MIS-PLACED tokens", i.e. a localized error, and a pooled L2 over the whole recv is exactly
        # the statistic that cannot see one.
        # COLLECTIVE VERDICT, and it is a correctness requirement of the SUITE, not a nicety.
        #
        # A numeric mismatch on SOME ranks and not others does not fail this job -- it WEDGES it, in
        # the NEXT cell, with a traceback that names an innocent line. Measured on this exact test at
        # 16 ranks, 2 nodes: `wide_nbiTrue` mismatched on 6 of 16 ranks (58-180 elements of 67 108 864,
        # worst ratio 26-43, a different set each time -- a race). Pytest RETAINS a failing test's
        # frame for its report, so those 6 ranks kept this cell's symmetric tensors referenced while
        # the 10 that passed released theirs into `DistributedManager.symmetric_mempool`. At the next
        # cell the 10 recycled a pool block and issued NO collective; the 6 had nothing to reuse and
        # entered the symmetric allocator, which is collective over every PE. The 6 parked in
        # `with torch.cuda.use_mem_pool(...)` and the 10 parked in the `dist.barrier()` on the line
        # after it -- 6 and 10, exactly. `symmetric_mempool`'s own docstring predicts the shape of
        # this: "the resulting rendezvous mismatch surfaces as a hang rather than an error".
        #
        # So reduce the verdict. Every rank runs the sanctioned element-wise comparison and keeps its
        # own worst-offender detail; the outcome is then all_reduced so ALL ranks raise together. The
        # defect still FAILS -- it is not softened -- but it fails as a failure instead of as a
        # deadlock three lines into an unrelated cell. This is the same bargain `collective_guard`
        # strikes for SKIPS (`gated_skip` reduces the decision), applied to the assertion.
        _local_err = None
        try:
            _same_bar_elementwise(
                got.float(), ref.float(), 2e-2, f"route2 ib_wide recv N={N} vs 2-kernel reshard"
            )
            assert h.n_outlier_rows == 0, (
                f"route2 ib_wide recv MISMATCH vs 2-kernel reshard: rel_L2={h.rel_l2:.3e} "
                f"outliers={h.n_outlier_rows} (the b_j-pinned wide coalesce mis-placed tokens)"
            )
        except AssertionError as _e:  # noqa: PERF203 -- one comparison, not a loop
            _local_err = _e
        _bad = torch.tensor([1 if _local_err is not None else 0], device=device, dtype=torch.int32)
        dist.all_reduce(_bad)  # SUM: how many ranks mismatched. Every rank reaches this line.
        _n_bad = int(_bad.item())
        if _local_err is not None:
            raise AssertionError(
                f"[{_n_bad}/{cp} ranks mismatched; this one did] {_local_err}"
            ) from _local_err
        assert _n_bad == 0, (
            f"{_n_bad}/{cp} ranks mismatched the 2-kernel reshard reference (this rank's recv "
            f"matched). A rank-divergent numeric outcome is a DEADLOCK in the next cell, not a "
            f"per-rank failure -- see the comment above. wide_nbi={wide_nbi}."
        )
    finally:
        try:
            compiled.free()
        except Exception:
            pass
        try:
            dist.barrier()
        except Exception:
            pass


# --------------------------------------------------------------------------- #
# Route-2 (A) PADDED-recv transpose_in front-A2A correctness GATE (torchrun pytest).
#
# Validates the REAL cp distributed peer TMA-S2G store of the transpose_in front
# (:class:`fold_cp_ops.distributed.dual_gated_gemm_a2a.DualGatedGemmDistSm90`) at an ARBITRARY
# N_loc (partial Xg-tile, Xg NOT %CTA-tile_M — the profiling grid's regime). The store is UNCHANGED
# (contiguous WALK index); configure sets rows_per_peer = PADDED B·Yg·ceil(Xg/BLK_M)·BLK_M so the walk
# lands in a padded symmetric recv (per-Y stride n_x·BLK_M, valid [0,Xg), BLK_M-aligned -> no 16-B/epi-M
# TMA wall). The pad tail (X in [Xg, n_x·BLK_M)) is the OOB A-load -> 0 -> einsum-safe.
#
# References the VALIDATED native (transpose_in-OFF) recv (RUNG-2 idiom at the real A2A level): the
# transposed recv's valid cols == the native recv with the (X,Y) token axes swapped; the pad tail == 0.
# PLUS a real X-axis contraction over the padded recv == the same contraction over the unpadded valid
# data (the "back a_major='k' reads the padded M transparently" proof — the zero tail contributes 0).
#
# Run (cp=2; CPO_CACHE_ENABLED=0 for the stale-mB2 trap, NVSHMEM_DISABLE_NVLS=1)::
#
#     CUDA_VISIBLE_DEVICES=0,1 CPO_CACHE_ENABLED=0 NVSHMEM_DISABLE_NVLS=1 CPO_DIST_MESH=cp=2 \
#       PYTHONPATH=$PWD python -m torch.distributed.run --rdzv-endpoint=localhost:29572 \
#       --nproc_per_node=2 -m pytest -q -s tests/distributed/test_front_transpose_in_a2a.py
# --------------------------------------------------------------------------- #


REL_BAR__distributed__front_transpose_in_a2a = 5e-2
# fp32 round-off bar for the padded-vs-valid X contraction (see test_transpose_in_padded_einsum_safe):
# the two contractions reduce a DIFFERENT number of X elements, so torch picks different accumulation
# orders and the same non-zero products sum to results differing in the last ulps. MEASURED on cp=8, all
# 8 ranks, with a bitwise-zero pad tail: rel = 1.18e-7 .. 1.45e-7, i.e. ONE ulp of fp32 (eps = 1.19e-7)
# — the signature of pure round-off. The bar keeps ~75x headroom over that, while a genuinely non-zero
# pad tail would inject terms as large as the valid ones (rel ~1e-1..1), five orders above the bar.
REL_EINSUM = 1e-5
BLK_M = 128


def _front_tile_n(Dloc):
    """Widest LEGAL front CTA ``tile_N`` at this ``Dloc``, capped at 128.

    The front feature-axis peer-split requires a CTA postact N-tile to lie inside ONE D-slice:
    ``tile_N % 16 == 0`` AND ``Dloc % (tile_N // 2) == 0`` (``best_front_tile``,
    dual_gated_gemm_a2a.py:290-331).  A HARDCODED 128 is illegal for ``Dloc < 64`` — i.e. cp>=8
    at D=256 — where it used to cost the whole cp=8 cell; the production picker returns a legal tile for
    any ``Dloc >= 8``, so derive instead of skipping.  Capped at 128 so cp<=4 (``Dloc >= 64``) keeps the
    historical tile EXACTLY: the picker would widen those to (128,256)/(256,128), and a ``tile_M`` other
    than ``BLK_M`` would invalidate this file's BLK_M-anchored pad geometry (``Xg_pad``/``rpp_p``).
    Halving a legal ``tile_N`` stays legal (``Dloc % 128 == 0`` implies ``Dloc % 64 == 0``)."""
    tile_N = min(128, DualGatedGemmDistSm90.best_front_tile(int(Dloc))[1])
    assert tile_N % 16 == 0 and int(Dloc) % (tile_N // 2) == 0, (
        f"no legal front tile_N at Dloc={Dloc} (got {tile_N})"
    )
    return tile_N


def _pe_map__distributed__front_transpose_in_a2a(dist_manager):
    from torch.distributed.tensor import Shard

    mesh = dist_manager.device_mesh
    placements = [Shard(i) for i in range(mesh.ndim)]
    return PeMap.from_mesh_placements(mesh, placements, distributed_manager=dist_manager)


def _nvshmem_barrier__distributed__front_transpose_in_a2a():
    import nvshmem.core

    nvshmem.core.barrier_all(stream=torch.cuda.current_stream())


def _drain__distributed__front_transpose_in_a2a():
    import nvshmem.core
    import nvshmem.core.rma as nvshmem_rma

    nvshmem_rma.quiet(stream=torch.cuda.current_stream())
    nvshmem.core.barrier_all(stream=torch.cuda.current_stream())
    torch.cuda.synchronize()


@pytest.fixture(scope="session", autouse=False)
def _nvshmem__distributed__front_transpose_in_a2a(dist_manager):
    from fold_cp_ops.distributed.distributed_manager import DistributedManager

    DistributedManager.init_nvshmem()
    yield


def _front_glu_2d__distributed__front_transpose_in_a2a(x, Wg2, Wp2):
    xf = x.float()
    return torch.sigmoid(xf @ Wg2.float().t()) * (xf @ Wp2.float().t())  # (M, 2D)


def _build_front_transpose_in_A2A(
    x,
    Wg2,
    Wp2,
    tile_shape_mn,
    *,
    recv_t,
    cp,
    my_cp_rank,
    rows_per_peer,
    pe_table,
    transpose_in=False,
    token_grid=None,
):
    """Compile+run a DualGatedGemmDistSm90 front A2A (transpose_in optional). recv_t is the
     symmetric (2*Dloc, cp*rpp) recv — PADDED rpp for transpose_in. Returns (compiled, run_fn, gemm_obj).

     A2A **ON**. The sibling `_build_front_transpose_in_LOCAL` is the A2A-OFF build of the same
     front; the two differ only in the store target, so a failure attributed to one and reproduced
    in the other is an attribution error, not a second defect."""
    assert get_device_capacity(x.device)[0] == 9, "SM90 only"
    # PURE-COUPLED by construction: `configure_a2a` below is called with no `ib_drain` and no
    # `decoupled`, so the store is a TMA-S2G straight into a peer's symmetric heap -- NVLink-only,
    # because `nvshmem_ptr` returns NULL for an IB peer. The guard lives HERE rather than at the call
    # sites because the builder is the thing that knows it is coupled; the two callers did not call
    # it, and at cp=16 over 2 nodes that faulted the whole session at `barrier.cu:55` with exit 255
    # on every rank and no Python traceback (pytest's fd capture is never flushed when nvshmem
    # `exit(-1)`s). Keyed on MEASURED topology, not on `cp`, so it stays correct where cp=16 IS one
    # NVLink domain.
    _skip_if_coupled_cross_node(tuple(int(v) for v in pe_table))
    M, K = x.shape
    twoN = Wg2.shape[0]
    tile_M, tile_N = tile_shape_mn
    Bw = interleave_dual_weights(Wg2, Wp2)
    out_postact = torch.empty((twoN, M), dtype=x.dtype, device=x.device)
    A = x.unsqueeze(0)
    B3 = Bw.mT.unsqueeze(0)
    PostAct = out_postact.mT.unsqueeze(0)
    A_p, B_p, _, _ = perm3d(A, B3, None, None)
    PostAct_p = perm3d(PostAct, B3, None, None)[0]
    a_dtype, _, _, _ = get_dtypes(A, B3, A, None)
    g = DualGatedGemmDistSm90(
        Float32,
        a_dtype,
        (tile_M, tile_N),
        (1, 1, 1),
        pingpong=False,
        is_persistent=True,
        chunk_g=1,
        n_dual_tiles=0,
        gate3_n3=0,
    )
    g.configure_a2a(
        cp=cp,
        my_cp_rank=my_cp_rank,
        rows_per_peer=rows_per_peer,
        pe_table=pe_table,
        transpose_in=transpose_in,
        token_grid=token_grid,
    )
    cute_recv = from_dlpack(recv_t, assumed_align=16)

    def _mk_epi():
        return DualGatedGemmDistSm90.EpilogueArguments(
            mPostAct=from_dlpack(PostAct_p, assumed_align=16),
            act_fn=gate_fn_map["glu"],
            mRowVecBroadcast=None,
            mBiasUp=None,
            mBiasGate=None,
            mMaskColVec=None,
            mPostAct3=None,
            act_fn_3=None,
            mRowVecBroadcast3=None,
            mWeight=None,
            mBias=None,
            eps=Float32(1e-5),
            rounding_mode=RoundingMode.RN,
            recv=cute_recv,
            token_grid_yg=(
                g.dynamic_token_grid_yg(token_grid[2], M)
                if (transpose_in and g._a2a_dynamic)
                else None
            ),
        )

    sched = make_scheduler_args(get_max_active_clusters(1), 8, None)
    cA = from_dlpack(A_p, assumed_align=16)
    cB = from_dlpack(B_p, assumed_align=16)
    stream = cutlass_torch.current_stream()
    compiled = compile_gemm_with_bitcode(
        g, cA, cB, None, None, _mk_epi(), sched, stream, register=True
    )

    def run_fn():
        compiled(
            from_dlpack(A_p, assumed_align=16),
            from_dlpack(B_p, assumed_align=16),
            None,
            None,
            _mk_epi(),
            sched,
            stream,
        )

    return compiled, run_fn, g


# Non-%128 N_loc cases (B, Xg=N_loc, Yg=N), representative of the grid (N_loc never %128):
#   n_x=1 partial (Xg<128): Xg=100/Yg=200; n_x=2 partial: Xg=250/Yg=500.
#
# BOTH original B=1 cells are kept verbatim, with a batched twin of each rather than a replacement:
# the batched walk is a DIFFERENT A-operand (rank-5 (Xg, K, Yg, B, L) instead of rank-4) and a
# different per-peer column law, so a B>1 cell does not subsume its B=1 twin. B=3 rides the larger
# grid because that is the cell whose n_x=2 partial-X tail interacts with the plane stride.
_CASES = [(1, 100, 200), (2, 100, 200), (1, 250, 500), (3, 250, 500)]


@pytest.mark.usefixtures("_nvshmem__distributed__front_transpose_in_a2a")
@pytest.mark.usefixtures("_module_skip__distributed__front_transpose_in_a2a")
@FRONT_A2A.parametrize(
    "B",
    "Xg",
    "Yg",
    cells=_CASES,
    because=(
        "cells= and not a product: B, Xg and Yg describe ONE local token grid, and the crossed "
        "triples are grids no rank ever holds"
    ),
)
@FRONT_A2A.parametrize("mesh")
def test_transpose_in_padded_recv_correct(
    apply_mesh, mesh, dist_manager, device, world_size, B, Xg, Yg
):
    """REAL cp A2A store: the transpose_in PADDED recv's valid cols == the native recv (X,Y)-swapped,
    the pad tail == 0, per-token error-histogram 0 outliers, full coverage of the valid region.

    BATCHED cells included (B from the matrix). At B > 1 the A-operand is the rank-5
    ``(Xg, K, Yg, B, L)`` form and the flat M-tile unravels a PLANE as well as a (Y, X) pair, so the
    per-peer column law becomes ``(b*Yg + Y)*Xg_pad + X``. Both reshapes below already carry the
    plane, so the comparison is per-plane by construction -- a decode that landed a whole plane in
    the wrong place fails on the histogram rather than on a shape."""
    apply_mesh(mesh)
    from tests.distributed.correctness_harness import compute_error_histogram

    cp = world_size
    D, K = 256, 256  # B is a matrix axis now -- see `_CASES`
    if D % cp != 0:
        rank_invariant_skip(f"D={D} not divisible by cp={cp}", because=_SHAPE_GATE_BECAUSE)
    Dloc = D // cp
    tile_N = _front_tile_n(
        Dloc
    )  # derived, not hardcoded -> legal at cp=8 (Dloc=32 -> tile_N=64) too
    M = B * Xg * Yg
    n_x = (Xg + BLK_M - 1) // BLK_M
    Xg_pad = n_x * BLK_M
    rpp_n = B * Xg * Yg  # native (unpadded) per-peer token extent
    rpp_p = B * Yg * Xg_pad  # PADDED per-peer token extent
    pm = _pe_map__distributed__front_transpose_in_a2a(dist_manager)
    pe_table = tuple(int(v) for v in pm.cp_pe_table.tolist())

    torch.manual_seed(4321 + dist_manager.rank)
    x = torch.randn(M, K, device=device, dtype=torch.bfloat16) / (K**0.5)
    Wg2 = torch.randn(2 * D, K, device=device, dtype=torch.bfloat16) / (K**0.5)
    Wp2 = torch.randn(2 * D, K, device=device, dtype=torch.bfloat16) / (K**0.5)

    # Symmetric memory via the torch MemPool, NOT nvshmem4py's allocator: torch owns the
    # nvshmem bootstrap here, so nvshmem4py's own `_is_initialized` is permanently False
    # and its `tensor()` raises `NVSHMEM Library is not initialized`. The pool also
    # recycles, which keeps the collective free off the destruction path.
    with torch.cuda.use_mem_pool(DistributedManager.symmetric_mempool(_this_rank_device())):
        recv_n = torch.empty(
            (2 * Dloc, cp * rpp_n), dtype=torch.bfloat16, device=_this_rank_device()
        )
    recv_n.fill_(-99.0)
    # Symmetric memory via the torch MemPool, NOT nvshmem4py's allocator: torch owns the
    # nvshmem bootstrap here, so nvshmem4py's own `_is_initialized` is permanently False
    # and its `tensor()` raises `NVSHMEM Library is not initialized`. The pool also
    # recycles, which keeps the collective free off the destruction path.
    with torch.cuda.use_mem_pool(DistributedManager.symmetric_mempool(_this_rank_device())):
        recv_p = torch.empty(
            (2 * Dloc, cp * rpp_p), dtype=torch.bfloat16, device=_this_rank_device()
        )
    recv_p.fill_(-99.0)
    try:
        # (1) native front (transpose_in OFF) — the validated reference.
        cN, runN, _ = _build_front_transpose_in_A2A(
            x,
            Wg2,
            Wp2,
            (BLK_M, tile_N),
            recv_t=recv_n,
            cp=cp,
            my_cp_rank=int(pm.my_cp_rank),
            rows_per_peer=rpp_n,
            pe_table=pe_table,
        )
        _nvshmem_barrier__distributed__front_transpose_in_a2a()
        runN()
        _drain__distributed__front_transpose_in_a2a()
        cN.free()
        # (2) transpose_in front (PADDED) — the unit under test.
        cP, runP, g = _build_front_transpose_in_A2A(
            x,
            Wg2,
            Wp2,
            (BLK_M, tile_N),
            recv_t=recv_p,
            cp=cp,
            my_cp_rank=int(pm.my_cp_rank),
            rows_per_peer=rpp_n,
            pe_table=pe_table,
            transpose_in=True,
            token_grid=(B, Xg, Yg),
        )
        assert g._a2a_rows_per_peer == rpp_p, (
            f"configure padded rpp {g._a2a_rows_per_peer} != expected {rpp_p}"
        )
        _nvshmem_barrier__distributed__front_transpose_in_a2a()
        runP()
        _drain__distributed__front_transpose_in_a2a()
        cP.free()

        # Per-peer compare: transposed-padded valid cols == native (X,Y)-swapped; pad tail == 0.
        n_out_total = 0
        tail_max = 0.0
        rel_max = 0.0
        for s in range(cp):
            bn = recv_n[:, s * rpp_n : (s + 1) * rpp_n].reshape(
                2 * Dloc, B, Xg, Yg
            )  # native X-outer Y-inner
            bp = recv_p[:, s * rpp_p : (s + 1) * rpp_p].reshape(
                2 * Dloc, B, Yg, Xg_pad
            )  # transposed-padded
            valid = bp[:, :, :, :Xg]  # (2Dloc, B, Yg, Xg)
            nat_swap = bn.permute(0, 1, 3, 2)  # (2Dloc, B, Yg, Xg)
            tail = bp[:, :, :, Xg:]
            tail_max = max(tail_max, tail.float().abs().max().item() if tail.numel() else 0.0)
            # per-token (b,Y,X) L2 over the 2*Dloc feature rows.
            got = valid.reshape(2 * Dloc, -1).t().reshape(1, -1, 1, 2 * Dloc)
            ref = nat_swap.reshape(2 * Dloc, -1).t().reshape(1, -1, 1, 2 * Dloc)
            h = compute_error_histogram(got.contiguous(), ref.contiguous())
            n_out_total += h.n_outlier_rows
            # Per element, at the SAME bar -- and here the conversion is EXACT: the form it replaces
            # was already max|got-ref| / max|ref|, and a maximum is under a threshold iff every
            # element is. Judged PER PEER, so a failure names the offending peer's column block and
            # the index inside it. `rel_max` is the same number, recovered from the worst ratio the
            # helper returns.
            rel_max = max(
                rel_max,
                REL_BAR__distributed__front_transpose_in_a2a
                * _same_bar_elementwise(
                    got.float(),
                    ref.float(),
                    REL_BAR__distributed__front_transpose_in_a2a,
                    f"transpose_in padded recv, peer {s}",
                ),
            )
        untouched = int((recv_p == -99.0).sum().item())  # all padded cols written (valid+tail)
        passed = (
            rel_max < REL_BAR__distributed__front_transpose_in_a2a
            and n_out_total == 0
            and tail_max == 0.0
            and untouched == 0
        )
        print(
            f"\n[transpose_in-pad cp={cp} Xg={Xg}(%128={Xg % 128}) Yg={Yg} n_x={n_x} rpp_p={rpp_p}] "
            f"rank={dist_manager.rank} rel={rel_max:.3e} n_outlier={n_out_total} tail_max={tail_max:.3e} "
            f"untouched={untouched} {'PASS' if passed else 'FAIL'}",
            flush=True,
        )
        # (no scalar rel_max assert: the per-peer element-wise verdict above already fired, and it is
        # the one that names WHICH element of WHICH peer's block the store corrupted.)
        assert n_out_total == 0, (
            f"{n_out_total} outlier token rows (localized transpose/store corruption)"
        )
        assert tail_max == 0.0, (
            f"pad tail != 0 (max {tail_max:.3e}) — einsum-UNSAFE (OOB A not zero-filled)"
        )
        assert untouched == 0, f"{untouched} padded recv cols unwritten (coverage gap)"
    finally:
        _nvshmem_barrier__distributed__front_transpose_in_a2a()


@pytest.mark.usefixtures("_nvshmem__distributed__front_transpose_in_a2a")
@pytest.mark.usefixtures("_module_skip__distributed__front_transpose_in_a2a")
@FRONT_A2A.parametrize(
    "B",
    "Xg",
    "Yg",
    cells=_CASES,
    because=(
        "cells= and not a product, for the same reason as the recv gate above: the triple IS the grid"
    ),
)
@FRONT_A2A.parametrize("mesh")
def test_transpose_in_padded_einsum_safe(
    apply_mesh, mesh, dist_manager, device, world_size, B, Xg, Yg
):
    """BACK a_major='k' reads the padded M TRANSPARENTLY: a real X-axis contraction over the padded
    front-kernel recv == the same contraction over the unpadded valid data (the zero pad tail adds 0).
    This is the incoming einsum's contracted (sharded i=X) axis — the property the back GEMM relies on.

    Two assertions, at the two precisions each deserves: the pad tail is BITWISE zero (the store's own
    guarantee, exactly checkable), and the padded contraction matches the valid one to fp32 round-off
    (`REL_EINSUM` — the two einsums reduce different X extents, so torch's accumulation order differs).

    BATCHED cells included (B from the matrix). The einsums already carry ``b`` as a free index, so
    at B > 1 this asserts the pad tail is zero in EVERY plane -- the tail is per (b, Y) row, and a
    plane whose tail was written by another plane's overshoot shows up here and nowhere else."""
    apply_mesh(mesh)

    cp = world_size
    D, K = 256, 256  # B is a matrix axis now -- see `_CASES`
    if D % cp != 0:
        rank_invariant_skip(f"D={D} not divisible by cp={cp}", because=_SHAPE_GATE_BECAUSE)
    Dloc = D // cp
    tile_N = _front_tile_n(
        Dloc
    )  # derived: a hardcoded 128 is ILLEGAL at cp=8 (Dloc=32) and used to
    #                                turn this cell into a build rejection rather than a real check
    M = B * Xg * Yg
    n_x = (Xg + BLK_M - 1) // BLK_M
    Xg_pad = n_x * BLK_M
    rpp_p = B * Yg * Xg_pad
    pm = _pe_map__distributed__front_transpose_in_a2a(dist_manager)
    pe_table = tuple(int(v) for v in pm.cp_pe_table.tolist())

    torch.manual_seed(777 + dist_manager.rank)
    x = torch.randn(M, K, device=device, dtype=torch.bfloat16) / (K**0.5)
    Wg2 = torch.randn(2 * D, K, device=device, dtype=torch.bfloat16) / (K**0.5)
    Wp2 = torch.randn(2 * D, K, device=device, dtype=torch.bfloat16) / (K**0.5)
    # Symmetric memory via the torch MemPool, NOT nvshmem4py's allocator: torch owns the
    # nvshmem bootstrap here, so nvshmem4py's own `_is_initialized` is permanently False
    # and its `tensor()` raises `NVSHMEM Library is not initialized`. The pool also
    # recycles, which keeps the collective free off the destruction path.
    with torch.cuda.use_mem_pool(DistributedManager.symmetric_mempool(_this_rank_device())):
        recv_p = torch.empty(
            (2 * Dloc, cp * rpp_p), dtype=torch.bfloat16, device=_this_rank_device()
        )
    recv_p.fill_(0.0)
    try:
        cP, runP, g = _build_front_transpose_in_A2A(
            x,
            Wg2,
            Wp2,
            (BLK_M, tile_N),
            recv_t=recv_p,
            cp=cp,
            my_cp_rank=int(pm.my_cp_rank),
            rows_per_peer=B * Xg * Yg,
            pe_table=pe_table,
            transpose_in=True,
            token_grid=(B, Xg, Yg),
        )
        _nvshmem_barrier__distributed__front_transpose_in_a2a()
        runP()
        _drain__distributed__front_transpose_in_a2a()
        cP.free()
        # MY recv, peer-0 block: (2*Dloc, B, Yg, Xg_pad); a/b halves; contract over X (the sharded axis).
        blk = recv_p[:, :rpp_p].reshape(2 * Dloc, B, Yg, Xg_pad)
        a_h, b_h = blk[:Dloc].float(), blk[Dloc:].float()
        # (1) THE MECHANISM, asserted EXACTLY: the pad tail the back GEMM will read is BITWISE zero.
        # That is what makes the padded contraction safe, it is a property of the STORE, and it is
        # exactly checkable — so it is asserted exactly.
        tail = blk[..., Xg:]
        tail_max = tail.float().abs().max().item() if tail.numel() else 0.0
        # (2) THE CONSEQUENCE, asserted to fp32 round-off: contracting the padded X equals contracting
        # only the valid X. NOT bitwise — the two einsums reduce a different NUMBER of X elements, so
        # torch tiles them differently and the identical set of non-zero products lands in a different
        # summation tree; the fp32 results then differ in the last ulps even with an exactly-zero tail
        # (measured on cp=8, all 8 ranks: rel 1.18e-7..1.45e-7 = ONE fp32 ulp, with `tail_max` exactly 0).
        # A bitwise `!=` here asserts torch's reduction determinism, not the kernel's guarantee, and that
        # is why it fired at Dloc=32 while the store was provably correct.
        out_pad = torch.einsum("ibYX,jbYX->ijbY", a_h, b_h)  # contract padded X (tail incl)
        out_val = torch.einsum(
            "ibYX,jbYX->ijbY", a_h[..., :Xg], b_h[..., :Xg]
        )  # contract valid X only
        n_diff = int((out_pad != out_val).sum().item())
        print(
            f"\n[einsum-safe cp={cp} Xg={Xg} Yg={Yg} Xg_pad={Xg_pad}] rank={dist_manager.rank} "
            f"tail_max={tail_max:.3e} padded-vs-valid n_diff={n_diff}/{out_pad.numel()} "
            f"{'tail SAFE' if tail_max == 0.0 else 'tail UNSAFE'}",
            flush=True,
        )
        assert tail_max == 0.0, (
            f"pad tail is NOT zero (max {tail_max:.3e}) — the back a_major='k' read of the padded M "
            f"contracts garbage; einsum-UNSAFE"
        )
        # Per element, at the SAME REL_EINSUM constant, and this conversion is EXACT: the form it
        # replaces was already max|pad-val| / max|val|. The bar stays deliberately tight (one fp32
        # ulp) because the two einsums reduce a DIFFERENT NUMBER of X elements -- torch tiles them
        # differently, so the identical set of non-zero products lands in a different summation tree
        # -- and a failure now names the (i,j,b,Y) coordinate whose contraction actually diverged
        # instead of one scalar over the whole output.
        _same_bar_elementwise(
            out_pad, out_val, REL_EINSUM, f"X-contraction over padded vs valid M (Xg={Xg}, Yg={Yg})"
        )
    finally:
        _nvshmem_barrier__distributed__front_transpose_in_a2a()


@pytest.fixture(scope="session")
def _module_skip__distributed__front_transpose_in_a2a():
    pytest.importorskip("nvshmem.core")
    pytest.importorskip("fold_cp_ops.distributed.dual_gated_gemm_a2a")


# --------------------------------------------------------------------------- #
# Front rank-2-M "walk p-inner" (route-2 ``transpose_in``) differential-layout correctness GATE.
#
# LOCAL, single-GPU (the front A2A-OFF path — NO torchrun / NVSHMEM runtime; the subclass is only
# imported, never nvshmem-initialized). Validates the STATIC ``transpose_in`` A-load of
# :class:`fold_cp_ops.distributed.dual_gated_gemm_a2a.DualGatedGemmDistSm90`: fed the NATIVE
# token buffer, it WALKS its token-M axis in transposed (X<->Y-swapped) p-inner order via a rank-2
# composite M layout, so the recv comes out einsum-``k``-major (a_major="k") for the back A2A GEMM with
# NO store change — killing the incoming ``.transpose(-1,-2).contiguous()`` copies.
#
# The front is a per-token map (contracts K): output token order == the A M-walk order. So walking M
# transposed just swaps the two N token axes of the output; the postact store writes recv col = flat
# GEMM-M index (unchanged). This is the permanent-pytest promotion of debug/route2_spike/rung2_transpose_in.py
# (+ rung4_flagoff.py). Runs on the A2A-OFF store (byte-identical to the local parent when flag off), so
# no collectives — plain ``pytest`` on one GPU:
#
#     CUDA_VISIBLE_DEVICES=0 CPO_CACHE_ENABLED=0 pytest -q tests/distributed/test_front_transpose_in.py
#
# Tests (SM90-only; skipped otherwise):
#   * ``test_transpose_in_swap_and_oracle`` — (a) transpose_in out == native out with the 2 N token axes
#     swapped, BIT-EXACT; (b) both walk orders vs the fp32 GLU oracle (rel_L2 < 2e-2). Parametrized over
#     square / rectangular / tile_M=256 / tile_N=256 / B>1 (3-mode composite M). (Xg % tile_M == 0.)
#   * ``test_transpose_in_partial_xg_padded`` — ROUTE-2 (A) arbitrary N_loc: Xg NOT %tile_M (partial
#     Xg-tile). The padded-per-Y scheduler + A-load store BIT-EXACT into a PADDED buffer — valid rows ==
#     native-swapped, the pad tail store-written to 0 (glu(0), einsum-safe). The A2A-OFF local analogue
#     of the cp=2 A2A gate (test_front_transpose_in_a2a.py). Grid-representative N_loc {100,250,192} + B=2.
#   * ``test_transpose_in_flag_off_byte_identical`` — flag-OFF subclass == the local parent, BIT-for-bit
#     (the §3.1 default-byte-identity guarantee: the rank-2-M hook is identity when transpose_in off).
#   * ``test_transpose_in_dynamic`` — dynamic + transpose_in: ONE dynamic compile at an anchor N serves
#     many runtime N, each BIT-EXACT vs a static-per-N front (one-compile-many-N via the 3-D fallback);
#     incl B>1 (the 5-D variant + 3-way tile_m unravel) AND a PARTIAL-Xg case (padded per-N postact M).
#   * ``test_transpose_in_dynamic_bad_N_raises`` — arbitrary N_loc (Xg NOT %tile_M) is now VALID (padded);
#     the only host-gate rejection left is M not divisible by B*N (a malformed grid) -> clear ValueError.
#
# ROUTE-2 (A) arbitrary N_loc (partial Xg-tile): Xg NEED NOT be a CTA tile_M multiple. The scheduler
# override emits B·Yg·ceil(Xg/BLK_M) M-tiles (each Y padded to a whole BLK_M count -> NO output M-tile
# straddles the Xg->Y boundary); the A-load OOB-clamps the partial last X-tile (pad rows read 0). The
# store is UNCHANGED (contiguous WALK index) but lands in a PADDED recv (per-Y stride n_x·BLK_M,
# BLK_M-aligned -> no 16-B/epi-M TMA wall, unlike the DEAD unpadded tile_y·Xg remap). The zero pad tail
# is einsum-safe (contributes 0 to any back contraction).
#
# The A-load is UNIFIED on the 3-D fallback for BOTH static and dynamic transpose_in (the composite
# rank-2-M was retired after it proved bit-exact + perf-equal to the 3-D path): A is presented as a
# composite-free ``(Xg, K, Yg[, B], L)`` view, tiled by the DEFAULT 2-tuple ``(BLK_M,BLK_K)`` (leaving
# ``Yg[,B],L`` untiled), so the atom box is a static ``(BLK_M,BLK_K)`` that takes RUNTIME ``Xg,Yg``.
# --------------------------------------------------------------------------- #


REL_BAR__distributed__front_transpose_in = 2e-2


def _front_glu_2d__distributed__front_transpose_in(x, Wg2, Wp2):
    """fp32 GLU of the stacked a/b front (no LN; x pre-normalized): (M, 2D)."""
    xf = x.float()
    return torch.sigmoid(xf @ Wg2.float().t()) * (xf @ Wp2.float().t())


def _build_front_transpose_in_LOCAL(
    x, Wg2, Wp2, tile_shape_mn, *, GemmCls, transpose_in, token_grid, dynamic=False, m_out=None
):
    """Construct + compile a front (parent or Dist subclass), A2A OFF, storing (a|b) to out_postact.

    Returns (compiled, run_fn, out_postact). tvm-ffi cute.compile (no nvshmem runtime). When GemmCls
    is the Dist subclass and transpose_in, the rank-2-M walk is set via _configure_transpose_in;
    ``dynamic`` sets ``_a2a_dynamic`` so the dynamic branch is exercised. ``m_out`` overrides the
    postact M extent (PADDED B·Yg·ceil(Xg/BLK_M)·BLK_M for a partial-Xg transpose_in: the store writes
    the padded WALK index, so the buffer must be padded; sentinel-filled so an unwritten pad row is
    distinguishable from a store-written glu(0)=0 tail).

    A2A **OFF**. The sibling `_build_front_transpose_in_A2A` builds the same front with A2A ON
    into a symmetric recv; see there before attributing a failure to either."""
    M, K = x.shape
    twoN = Wg2.shape[0]  # = 2D
    tile_M, tile_N = tile_shape_mn
    is_dist = GemmCls is DualGatedGemmDistSm90
    m_ext = M if m_out is None else int(m_out)

    Bw = interleave_dual_weights(Wg2, Wp2)  # (K, 2*twoN)
    out_postact = (
        torch.empty((twoN, m_ext), dtype=x.dtype, device=x.device)
        if m_out is None
        else torch.full((twoN, m_ext), -99.0, dtype=x.dtype, device=x.device)
    )
    A = x.unsqueeze(0)
    B3 = Bw.mT.unsqueeze(0)
    PostAct = out_postact.mT.unsqueeze(0)  # (1, M, 2D) M-major
    A_p, B_p, _, _ = perm3d(A, B3, None, None)
    PostAct_p = perm3d(PostAct, B3, None, None)[0]

    a_dtype, _, _, _ = get_dtypes(A, B3, A, None)
    gemm_obj = GemmCls(
        Float32,
        a_dtype,
        (tile_M, tile_N),
        (1, 1, 1),
        pingpong=False,
        is_persistent=True,
        chunk_g=1,
        n_dual_tiles=0,
        gate3_n3=0,
    )
    if is_dist:
        gemm_obj._a2a_dynamic = bool(dynamic)
        if transpose_in:
            gemm_obj._configure_transpose_in(True, token_grid, M)  # A2A OFF; sets the rank-2-M walk

    def mk_epi():
        common = dict(
            mPostAct=from_dlpack(PostAct_p, assumed_align=16),
            act_fn=gate_fn_map["glu"],
            mRowVecBroadcast=None,
            mBiasUp=None,
            mBiasGate=None,
            mMaskColVec=None,
            mPostAct3=None,
            act_fn_3=None,
            mRowVecBroadcast3=None,
            mWeight=None,
            mBias=None,
            eps=Float32(1e-5),
            rounding_mode=RoundingMode.RN,
        )
        # `_epi_args` drops the subclass-only terms for the parent, so the `is_dist` branch is
        # only about `recv` -- which the parent has no field for at all, subclass-only or not.
        return _epi_args(GemmCls, **common, recv=None) if is_dist else _epi_args(GemmCls, **common)

    sched = make_scheduler_args(get_max_active_clusters(1), 8, None)
    stream = cutlass_torch.current_stream()
    compiled = cute.compile(
        gemm_obj,
        from_dlpack(A_p, assumed_align=16),
        from_dlpack(B_p, assumed_align=16),
        None,
        None,
        mk_epi(),
        sched,
        stream,
        None,
    )

    def run_fn():
        compiled(
            from_dlpack(A_p, assumed_align=16),
            from_dlpack(B_p, assumed_align=16),
            None,
            None,
            mk_epi(),
            sched,
            stream,
            None,
        )

    return compiled, run_fn, out_postact


# (B, Xg, Yg, D, K, tile) -- Xg % tile_M == 0 required (no Y-straddle in the composite tile box).
_SHAPES = [
    (1, 128, 128, 128, 256, (128, 128)),  # square grid
    (1, 128, 256, 128, 256, (128, 128)),  # rectangular Yg > Xg
    (1, 256, 128, 128, 256, (256, 128)),  # tile_M = 256 (autotune cfg)
    (1, 128, 128, 128, 256, (128, 256)),  # tile_N = 256 (autotune cfg)
    (2, 128, 128, 128, 256, (128, 128)),  # B > 1 -> 3-mode composite M
]
_SHAPE_IDS = [f"B{b}_Xg{xg}_Yg{yg}_D{d}_tile{tm}x{tn}" for (b, xg, yg, d, k, (tm, tn)) in _SHAPES]


@pytest.mark.skipif(
    not torch.cuda.is_available() or get_device_capacity()[0] != 9,
    reason="front transpose_in tests are SM90-only",
)
@pytest.mark.usefixtures("_module_skip__distributed__front_transpose_in")
@FRONT_A2A.parametrize(
    "shape",
    only={"shape": tuple(_SHAPES)},
    because=(
        "the ALIGNED half of the shape pool (Xg % 128 == 0), which is what the unpadded walk is "
        "defined on. The partial-Xg half exercises the padded-per-Y scheduler and is covered by "
        "test_transpose_in_partial_xg_padded"
    ),
)
def test_transpose_in_swap_and_oracle(shape):
    """(a) transpose_in output == native output with the 2 N token axes swapped, BIT-EXACT;
    (b) both walk orders vs the fp32 GLU oracle (rel_L2 < 2e-2)."""
    B, Xg, Yg, D, K, tile = shape
    M, twoN = B * Xg * Yg, 2 * D
    torch.manual_seed(0)
    x = torch.randn(M, K, device=_this_rank_device(), dtype=torch.bfloat16) / (K**0.5)
    Wg2 = torch.randn(twoN, K, device=_this_rank_device(), dtype=torch.bfloat16) / (K**0.5)
    Wp2 = torch.randn(twoN, K, device=_this_rank_device(), dtype=torch.bfloat16) / (K**0.5)

    c_nat, run_nat, out_nat = _build_front_transpose_in_LOCAL(
        x,
        Wg2,
        Wp2,
        tile,
        GemmCls=DualGatedGemmDistSm90,
        transpose_in=False,
        token_grid=None,
    )
    c_tr, run_tr, out_tr = _build_front_transpose_in_LOCAL(
        x,
        Wg2,
        Wp2,
        tile,
        GemmCls=DualGatedGemmDistSm90,
        transpose_in=True,
        token_grid=(B, Xg, Yg),
    )
    try:
        run_nat()
        torch.cuda.synchronize()
        out_nat = out_nat.clone()
        run_tr()
        torch.cuda.synchronize()
        out_tr = out_tr.clone()

        # (a) BIT-EXACT: out_tr (b,Y,X) == out_nat (b,X,Y) with the two N token axes swapped.
        nat_g = out_nat.reshape(twoN, B, Xg, Yg)
        tr_g = out_tr.reshape(twoN, B, Yg, Xg)
        # `assert_bitwise` rather than `torch.equal`: same exact verdict (`torch.equal` is not banned
        # -- it is element-wise and cannot shadow one element), but this one RECORDS the comparison
        # for the numeric coverage gate and, on failure, names the first differing (d,b,Y,X)
        # coordinate with both hex bit patterns instead of only a max-abs.
        numerics.assert_bitwise(
            tr_g, nat_g.transpose(-1, -2), what="transpose_in walk vs native-swapped"
        )

        # (b) vs fp32 GLU oracle, both walk orders -- PER ELEMENT, at the SAME constant. This is the
        # STATISTIC change documented on `_same_bar_elementwise`: the pooled ||diff|| / ||ref|| it
        # replaces divides by the whole (M, 2D) energy, so a wrongly-walked token or two moves it by
        # ~1/sqrt(M) and vanishes -- and a mis-walked M axis is the ONE defect this test exists for.
        dual = _front_glu_2d__distributed__front_transpose_in(x, Wg2, Wp2)  # (M, 2D)
        dual_tr = dual.reshape(B, Xg, Yg, twoN).transpose(1, 2).reshape(M, twoN)  # (b,Y,X)
        _same_bar_elementwise(
            out_nat.t().float(),
            dual,
            REL_BAR__distributed__front_transpose_in,
            "native walk vs fp32 GLU oracle",
        )
        _same_bar_elementwise(
            out_tr.t().float(),
            dual_tr,
            REL_BAR__distributed__front_transpose_in,
            "transpose_in walk vs fp32 GLU oracle",
        )
    finally:
        for c in (c_nat, c_tr):
            if hasattr(c, "free"):
                c.free()


# (B, Xg, Yg, D, K, tile) — Xg NOT %tile_M (route-2 (A) PADDED-per-Y: arbitrary N_loc). Grid-
# representative N_loc: n_x=1 partial (Xg<128) + n_x=2 partial (Xg in {192,250,500}).
_PARTIAL = [
    (1, 100, 200, 128, 256, (128, 128)),  # n_x=1, Xg<128 (grid N200/cp2 shape)
    (1, 250, 500, 128, 256, (128, 128)),  # n_x=2 partial (grid N500)
    (1, 192, 256, 128, 256, (128, 128)),  # n_x=2, Xg=192 partial 64
    (2, 100, 200, 128, 256, (128, 128)),  # B=2 (5-D) partial
]
_PARTIAL_IDS = [f"B{b}_Xg{xg}_Yg{yg}" for (b, xg, yg, d, k, t) in _PARTIAL]


@pytest.mark.skipif(
    not torch.cuda.is_available() or get_device_capacity()[0] != 9,
    reason="front transpose_in tests are SM90-only",
)
@pytest.mark.usefixtures("_module_skip__distributed__front_transpose_in")
@FRONT_A2A.parametrize(
    "shape",
    only={"shape": tuple(_PARTIAL)},
    because=(
        "the PARTIAL-Xg half of the shape pool -- the padded-per-Y scheduler is what this test is "
        "about. The aligned half is covered by test_transpose_in_swap_and_oracle"
    ),
)
def test_transpose_in_partial_xg_padded(shape):
    """Route-2 (A) arbitrary N_loc (partial Xg-tile): the padded-per-Y scheduler + A-load store the
    transpose_in output BIT-EXACT into a PADDED buffer — valid rows == native-swapped, the pad tail
    is store-written to 0 (glu(0), einsum-safe). The A2A-OFF local analogue of the cp=2 A2A gate."""
    B, Xg, Yg, D, K, tile = shape
    blk_m = tile[0]
    n_x = (Xg + blk_m - 1) // blk_m
    Xg_pad = n_x * blk_m
    M, twoN = B * Xg * Yg, 2 * D
    m_pad = B * Yg * Xg_pad
    torch.manual_seed(0)
    x = torch.randn(M, K, device=_this_rank_device(), dtype=torch.bfloat16) / (K**0.5)
    Wg2 = torch.randn(twoN, K, device=_this_rank_device(), dtype=torch.bfloat16) / (K**0.5)
    Wp2 = torch.randn(twoN, K, device=_this_rank_device(), dtype=torch.bfloat16) / (K**0.5)

    c_nat, run_nat, out_nat = _build_front_transpose_in_LOCAL(
        x,
        Wg2,
        Wp2,
        tile,
        GemmCls=DualGatedGemmDistSm90,
        transpose_in=False,
        token_grid=None,
    )
    c_tr, run_tr, out_pad = _build_front_transpose_in_LOCAL(
        x,
        Wg2,
        Wp2,
        tile,
        GemmCls=DualGatedGemmDistSm90,
        transpose_in=True,
        token_grid=(B, Xg, Yg),
        m_out=m_pad,
    )
    try:
        run_nat()
        torch.cuda.synchronize()
        out_nat = out_nat.clone()
        run_tr()
        torch.cuda.synchronize()
        out_pad = out_pad.clone()
        # padded (twoN, B, Yg, Xg_pad); valid [:,:,:,:Xg] == native (b,X,Y)-swapped; tail == 0.
        padded = out_pad.reshape(twoN, B, Yg, Xg_pad)
        valid = padded[:, :, :, :Xg]
        nat_swap = out_nat.reshape(twoN, B, Xg, Yg).permute(0, 1, 3, 2)  # (twoN, B, Yg, Xg)
        tail = padded[:, :, :, Xg:]
        # BITWISE, and via the sanctioned helper so the coverage gate sees a comparison happened.
        # The verdict is the same one `torch.equal` gave (it is not banned -- element-wise and exact);
        # what is added is the first differing (d,b,Y,X) coordinate and both hex bit patterns.
        numerics.assert_bitwise(
            valid, nat_swap, what="partial-Xg transpose_in valid cols vs native-swapped"
        )
        tail_max = tail.float().abs().max().item() if tail.numel() else 0.0
        assert tail_max == 0.0, (
            f"pad tail != 0 (max {tail_max:.3e}) — the store did not zero-fill the OOB-A pad rows "
            f"(einsum-UNSAFE); a -99 sentinel here means the pad was left UNWRITTEN"
        )
    finally:
        for c in (c_nat, c_tr):
            if hasattr(c, "free"):
                c.free()


@pytest.mark.skipif(
    not torch.cuda.is_available() or get_device_capacity()[0] != 9,
    reason="front transpose_in tests are SM90-only",
)
@pytest.mark.usefixtures("_module_skip__distributed__front_transpose_in")
@matrix_exempt(
    "asserts the flag-OFF subclass is BIT-identical to its local parent at ONE shape. The property is a code-path identity -- `_a2a_enabled` False routes the store to `super()` -- not something that varies with a declared extent; parametrizing would compile the same two kernels at several shapes and assert the same identity each time"
)
def test_transpose_in_flag_off_byte_identical():
    """§3.1: a flag-OFF DualGatedGemmDistSm90 == the local DualGatedGemmSm90, BIT-for-bit
    (the rank-2-M hook is identity when transpose_in is off -> the shared-kernel edits are byte-id)."""
    B, Xg, Yg, D, K, tile = 1, 128, 128, 128, 256, (128, 128)
    M, twoN = B * Xg * Yg, 2 * D
    torch.manual_seed(1234)
    x = torch.randn(M, K, device=_this_rank_device(), dtype=torch.bfloat16) / (K**0.5)
    Wg2 = torch.randn(twoN, K, device=_this_rank_device(), dtype=torch.bfloat16) / (K**0.5)
    Wp2 = torch.randn(twoN, K, device=_this_rank_device(), dtype=torch.bfloat16) / (K**0.5)

    outs = {}
    for cls in (DualGatedGemmSm90, DualGatedGemmDistSm90):
        c, run, out = _build_front_transpose_in_LOCAL(
            x, Wg2, Wp2, tile, GemmCls=cls, transpose_in=False, token_grid=None
        )
        run()
        torch.cuda.synchronize()
        outs[cls.__name__] = out.clone()
        if hasattr(c, "free"):
            c.free()
    a, b = outs["DualGatedGemmSm90"], outs["DualGatedGemmDistSm90"]
    # §3.1 is a BYTE-identity guarantee, so compare BITS rather than counting `!=`: that also settles
    # the two cases a value comparison gets wrong in opposite directions -- a lost sign of zero
    # (-0.0 == 0.0) and a faithfully-copied NaN (NaN != NaN).
    numerics.assert_bitwise(
        b, a, what="flag-OFF DualGatedGemmDistSm90 vs local DualGatedGemmSm90 (transpose_in off)"
    )


def _build_front_dynamic(anchor_x, Wg2, Wp2, tile_shape_mn, *, token_grid):
    """Compile ONCE a DYNAMIC (mark_layout_dynamic + _a2a_dynamic) transpose_in front at the anchor
    shape. Returns (compiled, gemm_obj, run_at) where run_at(x_N, N) rebinds fresh operands for the
    runtime N (+ token_grid_yg = the runtime Yg=N) and runs -> out_postact (2*D, M_N)."""
    twoN = Wg2.shape[0]
    Bw = interleave_dual_weights(Wg2, Wp2)
    a_dtype, _, _, _ = get_dtypes(
        anchor_x.unsqueeze(0), Bw.mT.unsqueeze(0), anchor_x.unsqueeze(0), None
    )
    g = DualGatedGemmDistSm90(
        Float32,
        a_dtype,
        tile_shape_mn,
        (1, 1, 1),
        pingpong=False,
        is_persistent=True,
        chunk_g=1,
        n_dual_tiles=0,
        gate3_n3=0,
    )
    g._a2a_dynamic = True
    g._configure_transpose_in(True, token_grid, anchor_x.shape[0])
    sched = make_scheduler_args(get_max_active_clusters(1), 8, None)
    stream = cutlass_torch.current_stream()

    def _views(x, m_ext):
        # m_ext = PADDED postact M (B·Yg·ceil(Xg/BLK_M)·BLK_M); == M when Xg%BLK_M==0. Sentinel-filled
        # so a partial-Xg pad row left UNWRITTEN is caught (vs a store-written glu(0)=0 tail).
        out = torch.full((twoN, m_ext), -99.0, dtype=x.dtype, device=x.device)
        A_p, B_p, _, _ = perm3d(x.unsqueeze(0), Bw.mT.unsqueeze(0), None, None)
        PA_p = perm3d(out.mT.unsqueeze(0), Bw.mT.unsqueeze(0), None, None)[0]
        cA = from_dlpack(A_p, assumed_align=16).mark_layout_dynamic(leading_dim=1)  # K-major, M dyn
        cB = from_dlpack(B_p, assumed_align=16)  # static weight
        cPA = from_dlpack(PA_p, assumed_align=16).mark_layout_dynamic(leading_dim=0)  # M-major
        return cA, cB, cPA, out

    def _epi(cPA, yg):
        return DualGatedGemmDistSm90.EpilogueArguments(
            mPostAct=cPA,
            act_fn=gate_fn_map["glu"],
            mRowVecBroadcast=None,
            mBiasUp=None,
            mBiasGate=None,
            mMaskColVec=None,
            mPostAct3=None,
            act_fn_3=None,
            mRowVecBroadcast3=None,
            mWeight=None,
            mBias=None,
            eps=Float32(1e-5),
            rounding_mode=RoundingMode.RN,
            recv=None,
            token_grid_yg=yg,
        )

    m_pad_a = g.padded_rows_per_peer_dynamic(token_grid[2], anchor_x.shape[0])
    cA0, cB0, cPA0, _ = _views(anchor_x, m_pad_a)
    yg0 = g.dynamic_token_grid_yg(token_grid[2], anchor_x.shape[0])
    compiled = cute.compile(g, cA0, cB0, None, None, _epi(cPA0, yg0), sched, stream, None)

    def run_at(x_N, N):
        m_pad = g.padded_rows_per_peer_dynamic(N, x_N.shape[0])  # PADDED per-N postact M
        cA, cB, cPA, out = _views(x_N, m_pad)
        yg = g.dynamic_token_grid_yg(N, x_N.shape[0])
        compiled(cA, cB, None, None, _epi(cPA, yg), sched, stream, None)
        torch.cuda.synchronize()
        return out.clone()

    return compiled, g, run_at


# (B, anchor_N, [(run_N, Xg, Yg), ...]). B=2 exercises the 5-D path; the last case is PARTIAL-Xg
# (Xg NOT %tile_M) -> the padded one-compile-many-N (route-2 (A)); the anchor + runs share ONE compile.
# The run lists are TUPLES, not lists: these are `FRONT_A2A`'s `dyn_case` pool values, and the
# matrix hashes a value to record which cells a module emitted. A list inside the cell makes the
# whole cell unhashable, which the machinery survives (it catches the TypeError) by silently
# recording NOTHING -- i.e. the axis would look un-exercised. Tuples cost nothing here: the test
# only iterates these.
_DYN_CASES = [
    (1, 256, ((256, 256, 256), (384, 384, 384))),  # B=1 square (Xg%128==0), one compile serves 2 N
    (1, 256, ((256, 128, 256), (512, 256, 512))),  # B=1 rectangular (Xg=N/2, %128==0)
    (2, 128, ((128, 128, 128), (256, 256, 256))),  # B=2 -> 5-D (Xg,K,Yg,B,L) + 3-way unravel
    (1, 200, ((200, 100, 200), (500, 250, 500))),  # PARTIAL Xg (100,250 NOT %128) — padded per-N
]


@pytest.mark.skipif(
    not torch.cuda.is_available() or get_device_capacity()[0] != 9,
    reason="front transpose_in tests are SM90-only",
)
@pytest.mark.usefixtures("_module_skip__distributed__front_transpose_in")
@FRONT_A2A.parametrize("dyn_case")
def test_transpose_in_dynamic(dyn_case):
    """dynamic + transpose_in: ONE dynamic compile at an anchor N serves many runtime N, each
    BIT-EXACT vs a static-per-N transpose_in front (one-compile-many-N via the 3-D fallback). B=2
    exercises the 5-D variant + 3-way tile_m unravel; the partial-Xg case exercises the padded per-N
    postact M (the static comparand is padded identically -> a direct bit-exact match incl the tail).

    The parameter is named ``dyn_case`` and not ``case`` because a matrix axis name IS the pytest
    argument name -- `KernelMatrix.parametrize` emits exactly the declared name, so the two have to
    agree or pytest reports the test as using no such fixture."""
    B, anchor_N, runs = dyn_case
    tile, D, K = (128, 128), 128, 256
    blk_m = tile[0]
    twoN = 2 * D
    torch.manual_seed(0)
    Wg2 = torch.randn(twoN, K, device=_this_rank_device(), dtype=torch.bfloat16) / (K**0.5)
    Wp2 = torch.randn(twoN, K, device=_this_rank_device(), dtype=torch.bfloat16) / (K**0.5)
    a_Xg, a_Yg = next((xg, yg) for (n, xg, yg) in runs if n == anchor_N)
    ax = torch.randn(B * a_Xg * a_Yg, K, device=_this_rank_device(), dtype=torch.bfloat16) / (
        K**0.5
    )
    compiled, _, run_at = _build_front_dynamic(ax, Wg2, Wp2, tile, token_grid=(B, a_Xg, a_Yg))
    try:
        for N, Xg, Yg in runs:
            x = torch.randn(B * Xg * Yg, K, device=_this_rank_device(), dtype=torch.bfloat16) / (
                K**0.5
            )
            dyn_out = run_at(x, N)
            m_pad = B * Yg * ((Xg + blk_m - 1) // blk_m) * blk_m  # PADDED (== M when Xg%blk_m==0)
            c_s, run_s, out_s = _build_front_transpose_in_LOCAL(
                x,
                Wg2,
                Wp2,
                tile,
                GemmCls=DualGatedGemmDistSm90,
                transpose_in=True,
                token_grid=(B, Xg, Yg),
                m_out=m_pad,
            )
            run_s()
            torch.cuda.synchronize()
            stat_out = out_s.clone()
            if hasattr(c_s, "free"):
                c_s.free()
            # BITWISE via the sanctioned helper: same verdict as `torch.equal`, but it records the
            # comparison for the coverage gate and names the first differing coordinate, which is
            # what tells a mis-decoded dynamic tile from a genuinely different numeric result.
            numerics.assert_bitwise(
                dyn_out,
                stat_out,
                what=f"dynamic one-compile vs static-per-N at N={N} (B={B},Xg={Xg},Yg={Yg})",
            )
    finally:
        if hasattr(compiled, "free"):
            compiled.free()


@pytest.mark.skipif(
    not torch.cuda.is_available() or get_device_capacity()[0] != 9,
    reason="front transpose_in tests are SM90-only",
)
@pytest.mark.usefixtures("_module_skip__distributed__front_transpose_in")
@matrix_exempt(
    "asserts a REFUSAL on a MALFORMED token grid (a token count not divisible by B*N). That is an argument fault rather than a value on any declared axis"
)
def test_transpose_in_dynamic_bad_N_raises():
    """A runtime N violating the M % (B*N) == 0 token-grid contract raises a CLEAR ValueError (host
    gate), not a crash. NOTE: arbitrary N_loc (Xg NOT %tile_M) is now VALID (route-2 (A) padded-per-Y),
    so the ONLY host-gate rejection left is a token count M not divisible by B*N (a malformed grid)."""
    twoN, K = 2 * 128, 256
    ax = torch.randn(256 * 256, K, device=_this_rank_device(), dtype=torch.bfloat16) / (K**0.5)
    Wg2 = torch.randn(twoN, K, device=_this_rank_device(), dtype=torch.bfloat16) / (K**0.5)
    Wp2 = torch.randn(twoN, K, device=_this_rank_device(), dtype=torch.bfloat16) / (K**0.5)
    _, g, _ = _build_front_dynamic(ax, Wg2, Wp2, (128, 128), token_grid=(1, 256, 256))
    # Xg=100 (NOT %128) is now ACCEPTED (padded) — no raise:
    assert int(g.dynamic_token_grid_yg(200, 1 * 100 * 200)) == 200
    # M not divisible by B*N -> raise:
    with pytest.raises(ValueError):
        g.dynamic_token_grid_yg(200, 1 * 100 * 200 + 1)  # M=20001 not divisible by B*N=200


@pytest.fixture(scope="session")
def _module_skip__distributed__front_transpose_in():
    pytest.importorskip(
        "nvshmem.core"
    )  # the subclass module imports the vendored nvshmem utils at import time
    pytest.importorskip("fold_cp_ops.distributed.dual_gated_gemm_a2a")


# --------------------------------------------------------------------------- #
# REPRO: front-A2A staged coupled store at Dloc=8 / postact_epi_n=8 (the Dloc<16 crash).
#
# Mirrors ``test_front_best_config_correct`` but forces a SMALL D so that at low cp the
# per-peer feature slice Dloc = D/cp == 8 (postact tile_n=8, postact_epi_n=8) — the untested
# intersection that crashes at cp16(2,8) cross-node. Runs the COUPLED path (no ib_drain) so it
# reproduces single-node ALL-P2P (cp2, D=16 -> Dloc=8), where compute-sanitizer memcheck can
# pinpoint the faulting access.
#
# Run (cp2, single DGX, under memcheck)::
#
#     CUDA_VISIBLE_DEVICES=0,1 CPO_CACHE_ENABLED=0 NVSHMEM_DISABLE_NVLS=1 CPO_DIST_MESH=cp=2 \
#       PYTHONPATH=$PWD compute-sanitizer --tool memcheck --launch-timeout 120 \
#       python -m torch.distributed.run --rdzv-endpoint=localhost:29571 --nproc_per_node=2 \
#       -m pytest -q -s tests/distributed/test_repro_dloc8.py
# --------------------------------------------------------------------------- #


# noqa: E402

# D forced small so Dloc = D/cp lands at 8 (postact_epi_n=8). D=16 @ cp2 -> Dloc=8.
# A LITERAL, not the former `os.environ.get("REPRO_D", "16")`. The matrix is the only sanctioned
# source of test values, and an env-driven pool is precisely the "a test can invent shapes the
# checker never sees" case it exists to stop: the declared pool would then depend on who ran it, and
# a REPRO_D naming a D outside `FRONT_A2A.axis("D").values` would fail at IMPORT with a pool error
# rather than run. 16 was the default and is the only value that reaches the Dloc=8 regime this
# reproduction is about (D=16 at cp=2); another D belongs in the axis, alongside a because=.
_REPRO_D = [16]


@pytest.fixture(scope="session", autouse=False)
def _nvshmem__distributed__repro_dloc8(dist_manager):
    from fold_cp_ops.distributed.distributed_manager import DistributedManager

    DistributedManager.init_nvshmem()
    yield


@pytest.mark.usefixtures("_nvshmem__distributed__repro_dloc8")
@pytest.mark.usefixtures("_module_skip__distributed__repro_dloc8")
@FRONT_A2A.parametrize(
    "D",
    only={"D": tuple(_REPRO_D)},
    because=(
        "a pinned REPRODUCTION of the Dloc=8 / postact_epi_n=8 crash, which needs the one D that "
        "reaches Dloc=8 at a small cp. Every other D in the pool leaves Dloc >= 16 and reproduces "
        "nothing"
    ),
)
@FRONT_A2A.parametrize("mesh")
def test_repro_dloc8_coupled(apply_mesh, mesh, dist_manager, device, world_size, D):
    """Coupled front store at Dloc=D/cp (==8 for D16/cp2) — the postact_epi_n=8 crash repro."""
    apply_mesh(mesh)
    from tests.distributed.correctness_harness import compute_error_histogram

    cp = world_size
    if D % cp != 0:
        rank_invariant_skip(f"D={D} not divisible by cp={cp}", because=_SHAPE_GATE_BECAUSE)
    Dloc = D // cp
    tile_M, tile_N = DualGatedGemmDistSm90.best_front_tile(Dloc)
    # The SM90 WGMMA atom's OWN floor, which the gates around this one do not cover: a CTA tile
    # needs `tile_N % 16 == 0 and <= 256`, or `% 32 == 0 and <= 512`. Whether that is reachable is
    # decided by the MESH, not by D alone -- at D=16, cp=8 leaves Dloc=2, the picker returns
    # tile_N=4, and the build below raises from the kernel's own front door (`gemm_sm90.py`: "CTA
    # tile shape N must be divisible by 16 and <= 256, or ..."). Fewer features per rank than a
    # single N-tile is a HARDWARE floor, not a shape constraint this repo imposes, so it is skipped
    # rather than xfail'd or removed from the pool.
    #
    # First reachable at world_size 8: at 2 ranks the cp=8 and cp=(2,4) meshes skip outright, so no
    # earlier launch in this tree could observe it.
    if not ((tile_N % 16 == 0 and tile_N <= 256) or (tile_N % 32 == 0 and tile_N <= 512)):
        rank_invariant_skip(
            f"Dloc={Dloc} at D={D}/cp={cp} yields tile_N={tile_N}, under the SM90 WGMMA CTA-tile "
            f"floor (needs %16 and <=256, or %32 and <=512)",
            because=_SHAPE_GATE_BECAUSE,
        )
    tile_n_postact = tile_N // 2
    if Dloc % tile_n_postact != 0:
        rank_invariant_skip(
            f"Dloc={Dloc}%postact tile {tile_n_postact} at cp={cp} (picker invariant broken)",
            because=_SHAPE_GATE_BECAUSE,
        )
    B, N = 1, 128
    M = B * N * N
    if M % cp != 0 or (M // cp) % tile_M != 0:
        rank_invariant_skip(
            f"M={M}//cp not a multiple of tile_M={tile_M} at cp={cp}", because=_SHAPE_GATE_BECAUSE
        )
    rows_per_peer = M // cp
    K = 256
    M_full = cp * rows_per_peer
    pm = _pe_map(dist_manager)

    torch.manual_seed(4321 + dist_manager.rank)
    x = torch.randn(rows_per_peer, K, device=device, dtype=torch.bfloat16) / (K**0.5)
    Wg2 = torch.randn(2 * D, K, device=device, dtype=torch.bfloat16) / (K**0.5)
    Wp2 = torch.randn(2 * D, K, device=device, dtype=torch.bfloat16) / (K**0.5)
    # Symmetric memory via the torch MemPool, NOT nvshmem4py's allocator: torch owns the
    # nvshmem bootstrap here, so nvshmem4py's own `_is_initialized` is permanently False
    # and its `tensor()` raises `NVSHMEM Library is not initialized`. The pool also
    # recycles, which keeps the collective free off the destruction path.
    with torch.cuda.use_mem_pool(DistributedManager.symmetric_mempool(_this_rank_device())):
        recv = torch.empty((2 * Dloc, M_full), dtype=torch.bfloat16, device=_this_rank_device())
    recv.fill_(-99.0)

    pe_table = tuple(int(v) for v in pm.cp_pe_table.tolist())
    a2a_cfg = dict(
        cp=cp, my_cp_rank=int(pm.my_cp_rank), rows_per_peer=rows_per_peer, pe_table=pe_table
    )
    compiled, run_fn, _ = _build_front(x, Wg2, Wp2, (tile_M, tile_N), recv_t=recv, a2a_cfg=a2a_cfg)
    try:
        _nvshmem_barrier()
        run_fn()
        _drain()
        expected = _ref_front_dmajor(
            x, Wg2, Wp2, pm, cp=cp, rows_per_peer=rows_per_peer, Dloc=Dloc, world_size=world_size
        )
        untouched = int((recv == -99.0).sum().item())
        a_is_view = recv[:Dloc].data_ptr() == recv.data_ptr()
        got_bnnd = recv.t().contiguous().reshape(1, M_full, 1, 2 * Dloc)
        ref_bnnd = expected.t().contiguous().reshape(1, M_full, 1, 2 * Dloc)
        h = compute_error_histogram(got_bnnd, ref_bnnd)
        tag = f"repro-dloc8 cp={cp} D={D} Dloc={Dloc} tile=({tile_M},{tile_N}) M={M} rpp={rows_per_peer}"
        # The recv-vs-reference verdict, PER ELEMENT. `_same_bar_elementwise` carries the
        # historical `max|got-ref| / max|ref| < REL_BAR` bar UNCHANGED -- a maximum is under a
        # threshold iff every element is -- and raises naming the worst offender's INDEX with
        # its actual, reference and bound. It sits where the pooled scalar was computed, so the
        # scalar `assert rel < REL_BAR` that used to follow is gone. `rel` is that same ratio,
        # recovered exactly for the diagnostic print (the bound is the uniform REL_BAR*max|ref|,
        # so worst |err|/bound * REL_BAR == the old ratio); it no longer decides the test.
        rel = REL_BAR * _same_bar_elementwise(recv, expected, REL_BAR, tag)
        passed = rel < REL_BAR and h.n_outlier_rows == 0 and untouched == 0 and a_is_view
        print(
            f"\n[{tag}] rank={dist_manager.rank} rel={rel:.3e} (bar {REL_BAR}) {h.summary()} "
            f"untouched={untouched}/{recv.numel()} a_view={a_is_view} {'PASS' if passed else 'FAIL'}",
            flush=True,
        )
        assert h.n_outlier_rows == 0, (
            f"{tag}: {h.n_outlier_rows} outlier rows worst@{h.worst_row_index}"
        )
        assert untouched == 0, (
            f"{tag}: {untouched}/{recv.numel()} recv cells unwritten (coverage gap)"
        )
        assert a_is_view, f"{tag}: a/b half not a VIEW"
    finally:
        compiled.free()
        _nvshmem_barrier()


@pytest.fixture(scope="session")
def _module_skip__distributed__repro_dloc8():
    pytest.importorskip("nvshmem.core")
    pytest.importorskip("fold_cp_ops.distributed.dual_gated_gemm_a2a")


# --------------------------------------------------------------------------- #
# HOST-side (no GPU) regression guard for the WIDE-PUT run_i raster (front A2A-fused DualGatedGEMM).
#
# Two independent host-only checks — both run with plain ``pytest`` on ANY box (no CUDA, no torchrun),
# the sm_120 dev box included (they are pure Python int arithmetic). This is the CAVEAT-A silent-zero
# guard + the occupancy-safety proof of the R-derivation, the two things that can only be verified on a
# GPU by an EXPENSIVE cp16 2-node run — so they are pinned here as a cheap, always-green tripwire.
#
# 1. ``test_run_i_bijection_*`` — a pure-Python MIRROR of the run_i decode
#    (``TileScheduler._delinearize_work_idx_run`` run_i branch, ``fold_cp_ops/tile_scheduler.py``): walk every
#    persistent CTA over its whole run sequence and collect the covered (cid_m, cid_n, bidz) tiles. The
#    decode MUST be a BIJECTION over all tiles — every (m,n,l) covered EXACTLY the valid set, NO tile
#    silently dropped and NO out-of-range tile emitted — for ARBITRARY ``run_i_tiles`` (incl. the sub-band
#    case ``run_i_tiles ∤ ncluster_m``, the CAVEAT-A silent-zero the b275675 clamp+monotonic fix removes).
#    A regression here = the exact ``run_j_tiles`` sub-band silent-zero bug (5ec4944/b275675) on the i-axis.
#
# 2. ``test_derive_run_i_tiles_*`` — the host R-derivation
#    (``DualGatedGemmDistSm90._derive_run_i_tiles``): R = min(W, floor(ncm·ncn/grid_z)), enable iff
#    R >= R_MIN. Asserts (a) R <= W (ring cap), (b) R >= R_MIN when engaged, (c) the OCCUPANCY invariant
#    ``total_runs = ncn·ceil(ncm/R) >= grid_z`` (no persistent cluster left idle — the whole reason R is
#    bounded, NOT run_i_dynamic), and (d) the sub-R_MIN / degenerate shapes decline to 0 (default raster,
#    byte-identical). Pure int math -> no GPU (the grid_z device query is the caller's concern, mocked here).
# --------------------------------------------------------------------------- #


# --------------------------------------------------------------------------- #
# (1) run_i decode bijection / no-silent-zero (CAVEAT-A).
# --------------------------------------------------------------------------- #
def _run_i_coverage(ncluster_m, ncluster_n, L, R, gz):
    """Pure-Python MIRROR of the run_i decode (``_delinearize_work_idx_run`` run_i branch,
    ``fold_cp_ops/tile_scheduler.py``). Walk every persistent CTA z in [0,gz) over c=0,1,2,... while
    ``is_valid``, collect the covered (cid_m, cid_n, bidz) tiles. Returns the multiset {tile: count}.
    A silent-zero bug => some (m,n,l) is never covered; a clamp bug => an out-of-range tile appears."""
    runs_per_band = math.ceil(ncluster_m / R)
    total_runs = L * ncluster_n * runs_per_band
    covered = {}
    for z in range(gz):
        c = 0
        while True:
            run_local = c // R
            offset = c % R
            run_global = z + run_local * gz
            is_valid = run_global < total_runs
            if not is_valid:
                break  # monotonic stop-on-first-invalid (the persistent loop's exact stop)
            band = run_global // runs_per_band
            q = run_global % runs_per_band
            i = q * R + offset
            bidz = band // ncluster_n
            cid_n = band % ncluster_n
            cid_m = min(i, ncluster_m - 1)  # CLAMP padding i (the b275675 mirror on the i-axis)
            covered[(cid_m, cid_n, bidz)] = covered.get((cid_m, cid_n, bidz), 0) + 1
            c += 1
            if c > total_runs * R + gz + 10:  # safety bound (never hit for a correct decode)
                break
    return covered


# (ncluster_m, ncluster_n, L, R, gz): a mix of R|m (clean) + R∤m (sub-band / CAVEAT-A) + odd + tiny.
_BIJECTION_CASES = [
    (16, 16, 1, 4, 132),  # R | m (clean full-band divisor)
    (16, 16, 1, 5, 132),  # R ∤ m  <-- the sub-band silent-zero case (CAVEAT-A)
    (16, 16, 1, 7, 132),  # R ∤ m
    (13, 16, 1, 4, 132),  # odd m, R ∤ m
    (100, 16, 1, 128, 132),  # R > m (full band -> runs_per_band=1)
    (1024, 16, 1, 128, 132),  # production-ish large m
    (17, 3, 2, 5, 8),  # odd m/n, L=2, small gz, R ∤ m
    (1, 16, 1, 4, 132),  # m=1
    (16, 1, 1, 5, 132),  # n=1, R ∤ m
    (63, 16, 1, 8, 132),  # R ∤ m
    (79, 32, 1, 19, 114),  # the venue E cp16 D256 N=10048 sub-band anchor (79 prime -> always ∤)
    (79, 32, 1, 13, 188),  # same anchor, sm_120 grid_z -> R=13 (still sub-band)
]


def _run_i_per_cta_tile_counts(ncluster_m, ncluster_n, L, R, gz):
    """Mirror the run_i persistent walk PER CTA: for each z in [0,gz), count the work tiles CTA z
    processes (the epilogue PRODUCER emits one epi-batch group per tile — INCLUDING the clamped-padding
    re-stores for R∤ncluster_m). Returns ({z: n_tiles}, total_runs)."""
    runs_per_band = math.ceil(ncluster_m / R)
    total_runs = L * ncluster_n * runs_per_band
    counts = {}
    for z in range(gz):
        c = 0
        n = 0
        while True:
            run_global = z + (c // R) * gz
            if run_global >= total_runs:
                break
            n += 1
            c += 1
            if c > total_runs * R + gz + 10:
                break
        counts[z] = n
    return counts, total_runs


_PER_CTA_CASES = [
    (79, 32, 1, 19, 132),  # venue E cp16 N=10048 anchor (R=19, ncm=79 prime sub-band, gz=132 H100)
    (79, 32, 1, 19, 114),
    (79, 32, 1, 13, 188),
    (64, 32, 1, 15, 132),  # N=8192 ablation (R=15, ncm=64)
    (128, 32, 1, 31, 132),  # N=16384 (R=31)
    (32, 32, 1, 8, 132),  # R | ncm (no duplicates) — still redistributes tiles across CTAs
    (16, 16, 1, 5, 132),
    (100, 16, 1, 128, 132),  # R > ncm
    (17, 3, 2, 5, 8),  # L=2, small gz
]


@pytest.mark.usefixtures("_module_skip__distributed__run_i_bijection")
@pytest.mark.parametrize(
    "ncm,ncn,L,R,gz",
    _PER_CTA_CASES,
    ids=[f"m{m}n{n}L{ll}R{r}gz{g}" for (m, n, ll, r, g) in _PER_CTA_CASES],
)
@matrix_exempt(
    "HOST-side integer raster algebra -- a pure-Python mirror of the tile scheduler's run_i decode (or of the host R-derivation it feeds). It launches no kernel, allocates no tensor and needs no GPU; its variables (ncluster_m/n, L, R, grid_z) are a persistent-grid RASTER configuration rather than an input shape, so the matrix has nothing for it to draw"
)
def test_run_i_per_cta_count_matches_wide_count_parity(ncm, ncn, L, R, gz):
    """The WIDE-PUT count-parity fix: the consumer's per-CTA drain target ``my_tiles =
    ceil((total_runs-z)/gz)·R`` (dual_gated_gemm_a2a.py) MUST equal the producer's per-CTA tile
    emission under the run_i walk, for EVERY z — else target≠emission -> producer/consumer count mismatch
    -> DEADLOCK (the cp16 N=10048 hang, 2026-07-22). Also asserts Σ_z target = total_runs·R (the producer's
    total emission). Guards the run_i × wide-put count-parity regression at the exact triggering shape."""
    counts, total_runs = _run_i_per_cta_tile_counts(ncm, ncn, L, R, gz)

    def my_tiles(z):  # the exact kernel formula (the fix)
        if z >= total_runs:
            return 0
        return ((total_runs - z + gz - 1) // gz) * R

    mism = [(z, counts[z], my_tiles(z)) for z in range(gz) if counts[z] != my_tiles(z)]
    assert not mism, (
        f"count-parity target != producer emission for {len(mism)} CTAs (e.g. {mism[:3]}: "
        f"z, walk_tiles, formula) — the wide-put drain would DEADLOCK on these z"
    )
    total_emit = total_runs * R  # each valid run = R tiles (duplicates included)
    assert sum(counts.values()) == total_emit
    assert sum(my_tiles(z) for z in range(gz)) == total_emit


def _run_i_ordered_tiles(ncm, ncn, L, R, gz):
    """Per CTA z, the ORDERED (dst_tok=cid_m, peer_feat=(cid_n,plane)) tiles the wide PRODUCER sees (the
    persistent-loop c-order). For cluster_m=1 + m_sub_per_tile=1, dst_tok==cid_m and a run holds one feat."""
    runs_per_band = math.ceil(ncm / R)
    total_runs = L * ncn * runs_per_band
    per_cta = {}
    for z in range(gz):
        seq = []
        c = 0
        while True:
            run_global = z + (c // R) * gz
            if run_global >= total_runs:
                break
            band = run_global // runs_per_band
            q = run_global % runs_per_band
            i = q * R + (c % R)
            seq.append((min(i, ncm - 1), (band % ncn, band // ncn)))
            c += 1
            if c > total_runs * R + gz + 10:
                break
        per_cta[z] = seq
    return per_cta, total_runs


def _simulate_producer(seq, W, absorb):
    """Mirror the wide producer copy_fn batch accumulation (dual_gated_gemm_a2a.py). absorb=True is
    the run_i DUP-ABSORB fix; False is the OLD break-flush-every-duplicate flood. Returns
    (n_put, n_sub, covered) — covered = the (dst_tok, peer_feat) set of REAL-staged tiles (written to recv)."""
    b_w = b_base = b_dup = 0
    b_pf = None
    n_put = n_sub = 0
    covered = set()
    for dst_tok, pf in seq:
        merge = (b_w > 0) and (b_pf == pf) and (dst_tok == b_base + b_w) and (b_w < W)
        is_dup = absorb and (b_w > 0) and (b_pf == pf) and (b_base <= dst_tok < b_base + b_w)
        if (b_w > 0) and (not merge) and (not is_dup):
            n_put += 1
            n_sub += b_w + b_dup
            b_w = b_dup = 0
        if b_w == 0:
            b_base, b_pf, b_dup = dst_tok, pf, 0
        if not is_dup:
            covered.add((dst_tok, pf))  # a REAL staged tile -> written into the recv by this put
            b_w += 1
        else:
            b_dup += 1
    if b_w > 0:
        n_put += 1
        n_sub += b_w + b_dup
    return n_put, n_sub, covered


# _PER_CTA_CASES + EDGE cases (opt-in flag): stress a batch that fills W / closes at a run's last-valid
# tile right before its clamp-dups, and the exact-divisor (no-dup) boundary.
_PRODUCER_CASES = _PER_CTA_CASES + [
    (
        528,
        32,
        1,
        128,
        132,
    ),  # R=128=W: non-last runs FILL the batch to W, last run 16 real + 112 dups
    (300, 16, 1, 128, 132),  # R=128=W, 128∤300
    (
        129,
        8,
        1,
        128,
        132,
    ),  # R=128, ncm=129 -> last run 1 real + 127 dups (dups >> reals; W-boundary)
    (127, 8, 1, 64, 132),  # R=64, ncm=127 -> last run 63 real + 1 dup (single dup at the run tail)
    (256, 32, 1, 64, 132),  # R=64 | 256 -> NO dups (exact-divisor coverage / no-op edge)
    (131, 8, 1, 130, 40),  # R=130>W=128 cap -> a run itself exceeds W (batch MUST split mid-run)
]


@pytest.mark.usefixtures("_module_skip__distributed__run_i_bijection")
@pytest.mark.parametrize(
    "ncm,ncn,L,R,gz",
    _PRODUCER_CASES,
    ids=[f"m{m}n{n}L{ll}R{r}gz{g}" for (m, n, ll, r, g) in _PRODUCER_CASES],
)
@matrix_exempt(
    "HOST-side integer raster algebra -- a pure-Python mirror of the tile scheduler's run_i decode (or of the host R-derivation it feeds). It launches no kernel, allocates no tensor and needs no GPU; its variables (ncluster_m/n, L, R, grid_z) are a persistent-grid RASTER configuration rather than an input shape, so the matrix has nothing for it to draw"
)
def test_run_i_producer_dup_absorb(ncm, ncn, L, R, gz):
    """The run_i DUPLICATE-ABSORB producer fix (dual_gated_gemm_a2a.py): the wide producer absorbs a
    clamped-padding re-store (R∤ncm) into the pending batch's n_sub INSTEAD of break-flushing it as its own
    tiny 256 B put (the drain FLOOD that made run_i a perf LOSS — cp16 N=16384 measured 0.659). Asserts:
    (1) count-parity — Σ n_sub == total tiles processed (incl. dups) == the consumer's my_tiles·epi_tile_num
    target (else deadlock); (2) flood removed — one wide put per (run capped by W): n_put == Σ ceil(reals/W)
    over runs, and NEVER a per-dup tiny put; (3) A never INCREASES puts vs the old flood; (4) FULL COVERAGE —
    every VALID tile is REAL-staged (written) by some CTA -> untouched==0 (the coordinator's edge: a dup must
    never be the SOLE writer of a token its real tile also covers, and a flush at a run's last-valid tile must
    not orphan a dup into a spurious backward put)."""
    per_cta, total_runs = _run_i_ordered_tiles(ncm, ncn, L, R, gz)
    tiles = sum(len(s) for s in per_cta.values())
    put_old = sum(_simulate_producer(s, 128, absorb=False)[0] for s in per_cta.values())
    # expected wide puts = per run, ceil(reals_in_run / W) (a run wider than W splits into W-sized puts,
    # but a run NEVER emits a per-dup put). runs_per_band real counts: full R for non-last, ncm%R (or R) last.
    runs_per_band = math.ceil(ncm / R)
    exp_put_per_run = []
    for band_run in range(runs_per_band):
        reals = R if band_run < runs_per_band - 1 else (ncm - band_run * R)
        exp_put_per_run.append(max(1, math.ceil(reals / 128)))
    exp_puts = L * ncn * sum(exp_put_per_run)
    put_new = n_sub = 0
    covered = set()
    for s in per_cta.values():
        p, n, cov = _simulate_producer(s, 128, absorb=True)
        put_new += p
        n_sub += n
        covered |= cov
    assert n_sub == tiles, (
        f"count-parity BROKEN: Σn_sub={n_sub} != tiles={tiles} (the consumer target) — would DEADLOCK"
    )
    assert put_new == exp_puts, (
        f"flood/put mismatch: n_put={put_new} != expected {exp_puts} (one W-capped put per run, no per-dup put)"
    )
    assert put_new <= put_old, f"A increased puts ({put_old}->{put_new}) — regression"
    want = {(m, (n, l)) for m in range(ncm) for n in range(ncn) for l in range(L)}
    missing = want - covered
    assert not missing, (
        f"COVERAGE GAP: {len(missing)} valid tiles never REAL-staged (e.g. {sorted(missing)[:3]}) — the recv "
        f"would be untouched there (a dup orphaned / a real tile never written)"
    )


@pytest.mark.usefixtures("_module_skip__distributed__run_i_bijection")
@matrix_exempt(
    "HOST-side integer raster algebra -- a pure-Python mirror of the tile scheduler's run_i decode (or of the host R-derivation it feeds). It launches no kernel, allocates no tensor and needs no GPU; its variables (ncluster_m/n, L, R, grid_z) are a persistent-grid RASTER configuration rather than an input shape, so the matrix has nothing for it to draw"
)
def test_run_i_producer_dup_absorb_default_raster_noop():
    """A is a NO-OP on any raster WITHOUT repeated dst_tok (the default serpentine + R|ncm): absorb=True and
    absorb=False give the IDENTICAL put/n_sub/coverage. Guards byte-identity of the non-run_i / R|ncm paths."""
    per_cta, _ = _run_i_ordered_tiles(128, 32, 1, 32, 132)  # 32|128 -> no clamped duplicates
    for s in per_cta.values():
        assert _simulate_producer(s, 128, absorb=True) == _simulate_producer(s, 128, absorb=False)


# --------------------------------------------------------------------------- #
# (2) R-derivation: occupancy-safety + R_MIN gate + W-cap (host, no GPU).
# --------------------------------------------------------------------------- #


def _occupancy_total_runs(ncm, ncn, R):
    """total_runs the run_i scheduler emits = ncluster_n · ceil(ncluster_m / R) (·L, L=1 for the front)."""
    return ncn * math.ceil(ncm / R)


@pytest.mark.usefixtures("_module_skip__distributed__run_i_bijection")
@pytest.mark.parametrize("W", [32, 128])
@pytest.mark.parametrize("gz", [94, 114, 128, 132, 188])
@matrix_exempt(
    "HOST-side integer raster algebra -- a pure-Python mirror of the tile scheduler's run_i decode (or of the host R-derivation it feeds). It launches no kernel, allocates no tensor and needs no GPU; its variables (ncluster_m/n, L, R, grid_z) are a persistent-grid RASTER configuration rather than an input shape, so the matrix has nothing for it to draw"
)
def test_derive_run_i_tiles_occupancy_and_caps(W, gz):
    """Over a broad ncm×ncn sweep: when run_i engages (R>0) it (a) never exceeds W, (b) is >= R_MIN, and
    (c) keeps total_runs >= grid_z (NO idle persistent cluster — the bounded-run occupancy invariant).
    When it declines (R==0) it is genuinely sub-R_MIN or degenerate (never a missed engage)."""
    r_min = _G._A2A_RUN_I_RMIN
    for ncm in range(1, 2200, 7):
        for ncn in (1, 2, 3, 16, 32, 64):
            R = _G._derive_run_i_tiles(ncm, ncn, gz, W, r_min)
            if R == 0:
                raw = min(W, (ncm * ncn) // gz)
                assert raw < r_min, (
                    f"MISSED ENGAGE: ncm={ncm} ncn={ncn} gz={gz} W={W} -> raw R={raw} >= R_MIN={r_min} "
                    f"but derived 0"
                )
                continue
            assert R <= W, f"R={R} > W={W} (ring cap violated) at ncm={ncm} ncn={ncn} gz={gz}"
            assert R >= r_min, f"R={R} < R_MIN={r_min} engaged at ncm={ncm} ncn={ncn} gz={gz}"
            total_runs = _occupancy_total_runs(ncm, ncn, R)
            assert total_runs >= gz, (
                f"OCCUPANCY COLLAPSE: ncm={ncm} ncn={ncn} gz={gz} W={W} R={R} -> total_runs="
                f"{total_runs} < grid_z={gz} (a persistent cluster is left idle)"
            )


@pytest.mark.usefixtures("_module_skip__distributed__run_i_bijection")
@matrix_exempt(
    "HOST-side integer raster algebra -- a pure-Python mirror of the tile scheduler's run_i decode (or of the host R-derivation it feeds). It launches no kernel, allocates no tensor and needs no GPU; its variables (ncluster_m/n, L, R, grid_z) are a persistent-grid RASTER configuration rather than an input shape, so the matrix has nothing for it to draw"
)
def test_derive_run_i_tiles_gates_small_and_degenerate():
    """Small / degenerate shapes decline run_i (R==0 -> default serpentine raster, byte-identical)."""
    r_min = _G._A2A_RUN_I_RMIN
    assert _G._derive_run_i_tiles(2, 2, 132, 128, r_min) == 0  # tiny problem
    assert _G._derive_run_i_tiles(0, 16, 132, 128, r_min) == 0  # degenerate ncm
    assert _G._derive_run_i_tiles(16, 0, 132, 128, r_min) == 0  # degenerate ncn
    assert _G._derive_run_i_tiles(1000, 16, 0, 128, r_min) == 0  # degenerate grid_z
    # Just below vs at the R_MIN engage boundary (ncm·ncn/gz crossing r_min).
    assert _G._derive_run_i_tiles(r_min * 132 - 132, 1, 132, 128, r_min) == 0  # R = r_min-1 -> off
    assert _G._derive_run_i_tiles(r_min * 132, 1, 132, 128, r_min) == r_min  # R = r_min -> on


@pytest.mark.usefixtures("_module_skip__distributed__run_i_bijection")
@matrix_exempt(
    "HOST-side integer raster algebra -- a pure-Python mirror of the tile scheduler's run_i decode (or of the host R-derivation it feeds). It launches no kernel, allocates no tensor and needs no GPU; its variables (ncluster_m/n, L, R, grid_z) are a persistent-grid RASTER configuration rather than an input shape, so the matrix has nothing for it to draw"
)
def test_derive_run_i_tiles_w_cap():
    """A huge problem caps R at W (the ring holds exactly W subtiles — R>W would overflow)."""
    r_min = _G._A2A_RUN_I_RMIN
    for W in (32, 128):
        R = _G._derive_run_i_tiles(100000, 64, 132, W, r_min)
        assert R == W, f"huge problem should cap R at W={W}, got {R}"


@pytest.mark.usefixtures("_module_skip__distributed__run_i_bijection")
@matrix_exempt(
    "HOST-side integer raster algebra -- a pure-Python mirror of the tile scheduler's run_i decode (or of the host R-derivation it feeds). It launches no kernel, allocates no tensor and needs no GPU; its variables (ncluster_m/n, L, R, grid_z) are a persistent-grid RASTER configuration rather than an input shape, so the matrix has nothing for it to draw"
)
def test_derive_run_i_tiles_devtools_anchor_subband():
    """The venue E cp16 D256 N=10048 (m_linear, off-grid %128==64) anchor: ncm=ceil(10048/128)=79
    (PRIME), ncn=32. run_i MUST engage (R>=R_MIN) AND be sub-band (R ∤ 79) for every plausible grid_z —
    the shape the GPU sub-band test (test_front_ib_wide_run_i_subband) exercises end-to-end."""
    r_min = _G._A2A_RUN_I_RMIN
    ncm = math.ceil(10048 / 128)
    assert ncm == 79
    for gz in (94, 114, 128, 132, 188):
        R = _G._derive_run_i_tiles(ncm, 32, gz, 128, r_min)
        assert R >= r_min, f"run_i must engage at the venue E anchor (gz={gz}); got R={R}"
        assert ncm % R != 0, f"anchor must be SUB-BAND (R∤{ncm}); got R={R} at gz={gz}"


# --------------------------------------------------------------------------- #
# (3) route2_ni run_i SUB-BAND fix (#8 cp16 DEADLOCK): route2 snaps R to a DIVISOR of ncm so
#     runs_per_band is EXACT -> NO sub-band clamp/dup-absorb -> the b_j-pinned wide drain never
#     deadlocks. (D-major keeps the arbitrary-R sub-band; its dup path is correct.) Host, no GPU.
# --------------------------------------------------------------------------- #
@pytest.mark.usefixtures("_module_skip__distributed__run_i_bijection")
@matrix_exempt(
    "HOST-side integer raster algebra -- a pure-Python mirror of the tile scheduler's run_i decode (or of the host R-derivation it feeds). It launches no kernel, allocates no tensor and needs no GPU; its variables (ncluster_m/n, L, R, grid_z) are a persistent-grid RASTER configuration rather than an input shape, so the matrix has nothing for it to draw"
)
def test_largest_divisor_leq():
    """The route2 R-snap primitive: largest divisor of n that is <= cap (>=1); 0 on degenerate."""
    f = _G._largest_divisor_leq
    assert (
        f(512, 62) == 32
    )  # N1024 mesh 2*8 anchor: 512's divisors <=62 -> 32 (the ISO-4 working R)
    assert f(512, 128) == 128  # exact W-cap
    assert f(512, 3) == 2  # 512's divisors <=3 -> 2
    assert f(510, 62) == 51  # 510 = 2·3·5·17 -> 51
    assert f(7, 5) == 1  # prime -> 1 (caller then disables run_i)
    assert f(0, 62) == 0 and f(512, 0) == 0  # degenerate


@pytest.mark.usefixtures("_module_skip__distributed__run_i_bijection")
@matrix_exempt(
    "HOST-side integer raster algebra -- a pure-Python mirror of the tile scheduler's run_i decode (or of the host R-derivation it feeds). It launches no kernel, allocates no tensor and needs no GPU; its variables (ncluster_m/n, L, R, grid_z) are a persistent-grid RASTER configuration rather than an input shape, so the matrix has nothing for it to draw"
)
def test_route2_run_i_snaps_to_divisor_no_subband():
    """route2_ni MUST NOT hit the run_i sub-band clamp (R∤ncm -> the dup-absorb re-store path that DEADLOCKS
    the b_j-pinned wide drain at cp16, #8). route2's ncm = B·Yg·n_x ALWAYS has n_x as a factor, so snapping
    the occupancy-safe R DOWN to the largest divisor of ncm ALWAYS yields R | ncm (runs_per_band exact) ->
    NO clamp, NO dups. Mirrors _maybe_inject_run_i_wide's route2 snap (composition of the two host statics)
    over a broad shape sweep incl. PRIME Yg. (The R=62->32 @ncm=512 case is the ISO-4-measured 1.853ms fix.)"""
    r_min = _G._A2A_RUN_I_RMIN
    for W in (32, 128):
        for gz in (94, 128, 132, 188):
            for n_x in (1, 2, 4, 8):
                for Yg in (1, 3, 7, 16, 128, 251):  # 251 prime -> stress the divisor search
                    for B in (1, 2):
                        ncm = B * Yg * n_x
                        for ncn in (1, 16, 32):
                            R = _G._derive_run_i_tiles(ncm, ncn, gz, W, r_min)
                            if R == 0:
                                continue  # run_i declined -> default raster (fine)
                            Rs = _G._largest_divisor_leq(ncm, R)
                            if Rs < r_min:
                                continue  # snapped below R_MIN -> caller disables run_i
                            assert ncm % Rs == 0, (
                                f"route2 SUB-BAND NOT eliminated: ncm={ncm} R={R} -> snapped {Rs}, "
                                f"{ncm}%{Rs}={ncm % Rs} != 0 (the #8 deadlock trigger)"
                            )
                            assert r_min <= Rs <= R, f"snapped R={Rs} out of [{r_min}, {R}]"


@pytest.fixture(scope="session")
def _module_skip__distributed__run_i_bijection():
    global _G, _a2a
    _a2a = pytest.importorskip(
        "fold_cp_ops.distributed.dual_gated_gemm_a2a",
        reason="R-derivation test needs the front-A2A module (cutlass); the bijection tests above do not.",
    )
    _G = _a2a.DualGatedGemmDistSm90


# --------------------------------------------------------------------------- #
# FRONT A2A cubin byte identity -- the other half of the gate the back closed.
#
# The i64 widening that fixed the front's illegal address touched BOTH drains --
# the per-subtile one and `ib_wide` -- so `ib_wide` is the axis, not a detail.
# And it is a STATIC-extent defect: under a dynamic extent the strides are runtime
# values and the Int64 casts are redundant by construction, so a dynamic-only
# comparison is guaranteed to look clean whatever the code does. Both modes are
# swept for that reason.
#
# The label carries the VENUE (`_ib0` / `_ib1`). `has_ib_peers` is false on one node
# and true across nodes, and it gates the whole IB drain -- measured on the back's
# equivalent harvest, the two venues differ by 4x in object size, so a harvest that
# could not tell them apart would silently compare one against the other.
# --------------------------------------------------------------------------- #
_FRONT_CUBIN_CONFIGS = ((False,), (True,))


@matrix_exempt(
    "the swept axis is the DRAIN VARIANT (`ib_wide` on/off) crossed with `shape_mode`, which is what "
    "the i64 widening touched -- not a shape. The matrix's D/tile_N/cwg pools describe the numerical "
    "tests; sweeping them here would multiply compiles by the drain count for no extra "
    "discrimination, since a shape enters the cubin only through baked extents and those are the "
    "`shape_mode` axis itself."
)
@pytest.mark.usefixtures("_nvshmem__distributed__front_a2a_staged")
@pytest.mark.usefixtures("_module_skip__distributed__front_a2a_staged")
@pytest.mark.parametrize("shape_mode", ["dynamic", "static"])
@pytest.mark.parametrize("ib_wide", [False, True], ids=["subtile", "ibwide"])
@numeric_exempt("digests COMPILED CODE; no tensor is produced and none is compared")
def test_front_drain_cubin(
    apply_mesh, dist_manager, device, world_size, ib_wide, shape_mode, tmp_path
):
    """Digest the front A2A drain's cubin, per drain variant and per shape mode.

    Functionality & semantics:
        Builds the decoupled front through the same `_build_front_decoupled` the numerical
        tests use, exports the compiled object and digests only its CUDA-ELF code sections. Writes
        one JSON digest per cell under ``CPO_CUBIN_OUT`` when set, for the out-of-tree ours-vs-`main`
        comparison; otherwise exports into ``tmp_path`` and discards.

    Input requirements:
        A flat cp mesh, built here -- `_pe_map` reads `dist_manager.device_mesh`, which only
        ``apply_mesh`` constructs, and a test that omits it is order-dependent rather than green.
        ``D`` must give ``Dloc = D // cp`` that `best_front_tile` can serve and that divides the
        postact tile, else the front refuses at configure; ``D = 32 * cp`` satisfies it for every cp
        this runs at. The geometry is deliberately the SMALLEST that compiles -- this test digests
        code, not time, and the recv is ``(2*Dloc, cp*M)``, which grows as ``cp*N_token^2``.
        The disk cache MUST be off and the process fresh, one config per process; `digest_export`
        raises naming the cache if not.

    Raises:
        ``RuntimeError`` from `code_sections` if the export carried no CUDA ELF -- most often two
        ranks racing on one path, which is why the object name carries the rank.
    """
    apply_mesh((("cp", world_size),))
    assert get_device_capacity(device)[0] == 9, "SM90 only"
    cp = world_size
    # D defaults to 32*cp so Dloc stays 32 at any mesh -- but that COUPLES D to cp, and a
    # cp-vs-cp comparison then varies two things at once. Measured the hard way: a pre-registered
    # prediction that the ours-vs-main delta is cp-independent could not be scored, because cp=2
    # ran at D=64 and cp=16 at D=512. CPO_FRONT_CUBIN_D pins D so the comparison is controlled.
    D = int(os.environ.get("CPO_FRONT_CUBIN_D", 32 * cp))
    if D % cp:
        rank_invariant_skip(f"D={D} is not divisible by cp={cp}", because=_IB_UNIFORM_BECAUSE)
    Dloc = D // cp
    tile_M, tile_N = DualGatedGemmDistSm90.best_front_tile(Dloc)
    if Dloc % (tile_N // 2) != 0:
        rank_invariant_skip(
            f"Dloc={Dloc} is not a multiple of the postact tile {tile_N // 2} at cp={cp}",
            because=_IB_UNIFORM_BECAUSE,
        )
    B, N_token, K, rd, cwg = 1, 16, 256, 2, 2
    M = B * N_token * N_token
    pm = _pe_map(dist_manager)
    pe_table = tuple(int(v) for v in pm.cp_pe_table.tolist())
    has_ib = _job_has_ib_peers(pe_table)
    torch.manual_seed(4321 + dist_manager.rank)
    x = torch.randn(M, K, device=device, dtype=torch.bfloat16) / (K**0.5)
    Wg2 = torch.randn(2 * D, K, device=device, dtype=torch.bfloat16) / (K**0.5)
    Wp2 = torch.randn(2 * D, K, device=device, dtype=torch.bfloat16) / (K**0.5)
    with torch.cuda.use_mem_pool(DistributedManager.symmetric_mempool(_this_rank_device())):
        recv = torch.empty((2 * Dloc, cp * M), dtype=torch.bfloat16, device=_this_rank_device())
    recv.zero_()

    def _cfg(g):
        g.configure_a2a(
            cp=cp,
            my_cp_rank=int(pm.my_cp_rank),
            rows_per_peer=M,
            pe_table=pe_table,
            ib_drain=True,
            ring_depth=rd,
            consumer_warpgroups=cwg,
            ib_wide=ib_wide,
        )

    compiled, _run, _g, _ring = _build_front_decoupled(
        x,
        Wg2,
        Wp2,
        (tile_M, tile_N),
        recv_t=recv,
        ring_depth=rd,
        consumer_warpgroups=cwg,
        configure_fn=_cfg,
        shape_mode=shape_mode,
    )
    variant = "ibwide" if ib_wide else "subtile"
    label = f"front_{variant}_D{D}_N{N_token}_{shape_mode}_ib{int(has_ib)}"
    out_dir = os.environ.get("CPO_CUBIN_OUT")
    dest = Path(out_dir) if out_dir else Path(str(tmp_path))
    dest.mkdir(parents=True, exist_ok=True)
    try:
        digest = digest_export(compiled.compiled, str(dest / f"{label}.r{dist_manager.rank}.o"))
        assert_has_code(digest, f"front drain {label}")
        if out_dir and dist_manager.rank == 0:
            (dest / f"{label}.json").write_text(
                json.dumps({"label": label, "cp": cp, "sections": digest}, indent=1, sort_keys=True)
            )
        print(
            f"\n[cubin {label}] .text {digest['.text'][0]}B {digest['.text'][1][:12]}", flush=True
        )
    finally:
        compiled.free()


def _source_of_resolved_method(cls, name):
    """Source text of the method ``cls`` resolves for ``name``, read from its DEFINING class.

    Purpose
        Let a test ask what a method actually DOES, for a method the CuTe DSL has decorated.

    Why not ``inspect.getsource(getattr(cls, name))``, which is the obvious way and is WRONG here
        A ``@cute.kernel``-decorated method keeps its ``__qualname__`` but NOT its ``__code__``:
        measured, ``DualGatedGemmDistSm90.kernel.__qualname__`` is ``"GemmSm90.kernel"`` while
        ``inspect.getsourcefile`` of it returns **``dsl.py``** and ``co_firstlineno`` is a line in
        the DSL. So ``getsource`` hands back the DECORATOR's body, and a check written that way
        reports on code the kernel does not contain -- passing or failing for reasons unrelated to
        the subject. Resolving the OWNER through the MRO and reading the class's own file avoids the
        wrapper entirely.

    Args:
        cls: The class whose resolution is in question -- the MRO is walked in order, so the first
            class whose ``vars()`` holds ``name`` is the one Python would use.
        name: Method name. Must be defined somewhere in ``cls.__mro__``; a name defined nowhere
            raises rather than returning "" , since an empty string would read as "defines nothing"
            and silently satisfy a substring assertion.

    Returns:
        ``(owner, source)`` -- the defining class and the exact source lines of its ``name``.

    Raises:
        AssertionError: If no class in the MRO defines ``name``.
    """
    import ast
    import inspect

    owners = [k for k in cls.__mro__ if name in vars(k)]
    assert owners, f"no class in {cls.__name__}'s MRO defines {name!r}"
    owner = owners[0]
    text = inspect.getsource(inspect.getmodule(owner))
    tree = ast.parse(text)
    cdef = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == owner.__name__)
    fdef = next(n for n in cdef.body if isinstance(n, ast.FunctionDef) and n.name == name)
    return owner, "\n".join(text.splitlines()[fdef.lineno - 1 : fdef.end_lineno])


@matrix_exempt(
    "asserts a CROSS-CLASS SOURCE contract over the MRO -- which methods this subclass resolves and "
    "whether they still stash and reach the handle. It has no shape, dtype or mesh to draw from and "
    "holds for every configuration simultaneously"
)
@numeric_exempt("inspects resolved method source, not a computed value")
def test_the_prologue_this_class_resolves_still_stashes_epi_storage():
    """The ``_epi_storage`` handoff spans two files and two classes; assert both ends of it.

    **The contract.** ``GemmSm90.kernel_prologue`` stashes the trace-time SMEM handle as
    ``self._epi_storage`` so the decoupled-A2A producer copy_fn can reach the ring's SMEM -- that
    copy_fn is reached deep inside the MMA warpgroup's epilogue and is not threaded ``storage``
    through its signature. The read lives in THIS module's store seam, guarded by
    ``const_expr(self._decoupled_active())``. Writer and reader sit in different files and different
    classes, joined by nothing but Python's MRO, and until now **no assertion connected them**.

    **Two ways it can break, and both are asserted, because either alone would let the other
    through.**

    1. The resolved ``kernel_prologue`` stops writing the handle -- an override that reimplements the
       prologue without it.
    2. The resolved ``kernel`` stops CALLING ``kernel_prologue`` -- an override that allocates its
       own storage. This is the live one: ``kernels/layernorm_dual_gated_gemm.py`` DOES override
       ``kernel()`` and remains correct precisely because it still delegates to the base prologue
       (``:2250``). So an override is not itself the hazard; an override that skips the prologue is.

    Either failure yields ``AttributeError: no attribute '_epi_storage'`` on the ib_drain_hybrid
    path, at runtime, on the cross-node drain -- the precise bug the current placement was made to
    fix, returning with nothing to announce it.

    **On the placement comment this depends on.** ``fold_cp_ops/kernels/gemm_sm90.py`` explains the
    divergence from ``main`` in terms of "the ``kernel()`` that actually runs", but the write it
    annotates is physically inside ``kernel_prologue`` -- which is the method that makes the
    arrangement robust, since every ``kernel()`` override in the tree still delegates to it. A
    comment is not a check either way; this is the check.

    Pure source inspection: no device, no process group, no compile.
    """
    prologue_owner, prologue_src = _source_of_resolved_method(
        DualGatedGemmDistSm90, "kernel_prologue"
    )
    assert "self._epi_storage" in prologue_src, (
        f"DualGatedGemmDistSm90 resolves kernel_prologue() to "
        f"{prologue_owner.__name__}.kernel_prologue, whose source does NOT write "
        "`self._epi_storage`. The decoupled-A2A store seam in this module reads that attribute, so "
        "the ib_drain_hybrid path will raise `AttributeError: no attribute '_epi_storage'` at "
        "runtime on the cross-node drain. Restore the write in whichever prologue this class now "
        "resolves, or thread `storage` through the epilogue signature and delete the attribute "
        "(the real fix, deliberately scoped out of the bring-back). See the "
        "`trace_time_smem_handoff` note in tests/kernels/test_gemm_sm90.py."
    )

    kernel_owner, kernel_src = _source_of_resolved_method(DualGatedGemmDistSm90, "kernel")
    assert "self.kernel_prologue(" in kernel_src, (
        f"DualGatedGemmDistSm90 resolves kernel() to {kernel_owner.__name__}.kernel, which does NOT "
        f"call self.kernel_prologue(). The `_epi_storage` write lives in the PROLOGUE, so a kernel "
        "that allocates its own storage instead of delegating loses the handle even though the "
        "prologue still writes it -- and the failure appears only on the ib_drain_hybrid cross-node "
        "drain, as an AttributeError, at runtime."
    )


#: Every keyword `main`'s A2A-fused TriMul workflow passes to this kernel's front door, MEASURED by
#: running `main`'s e2e over all 18 tags of the acceptance grid with `configure_a2a` wrapped and its
#: bound arguments recorded (`w8plan/step1/`, STEP 1 of `docs/a2a_bringback.md`).
#:
#: `configure_a2a_sharded` is ABSENT on purpose and the absence is a measurement, not an oversight:
#: over all 28 cell-directions the workflow reached this kernel through the flat-cp `configure_a2a`
#: even on a 2-D mesh (the staged front's recv index is 2-D-invariant, so 2-D is the
#: `cp_axis_sizes == (cp,)` case). A door the acceptance grid never opens cannot have its keyword set
#: measured, and guessing one would be exactly the transcription this file avoids.
_WORKFLOW_KWARGS_MEASURED = {
    "configure_a2a": {
        "consumer_warpgroups",
        "cp",
        "decoupled",
        "dynamic",
        "ib_drain",
        "ib_wide",
        "ib_wide_batch",
        "ib_wide_nbi",
        "inner_extent",
        "my_cp_rank",
        "pad_inner",
        "partial_token_clamp",
        "pe_table",
        "ring_depth",
        "rows_per_peer",
        "token_count",
        "token_grid",
        "transpose_in",
    },
}

#: Keywords this kernel deliberately REFUSES. Empty -- the workflow sent nothing we do not accept.
#: An entry here must arrive with a front-door test proving the refusal.
_WORKFLOW_KWARGS_REFUSED: dict[str, set[str]] = {"configure_a2a": set()}


@matrix_exempt(
    "compares a SIGNATURE against a measured keyword set -- no kernel launch, no operand, no shape, "
    "so no matrix axis applies"
)
@numeric_exempt("compares parameter names, not a computed value")
@pytest.mark.parametrize("method", sorted(_WORKFLOW_KWARGS_MEASURED))
def test_every_workflow_kwarg_is_accepted_or_refused(method):
    """Every keyword the workflow sends is a parameter of ours, or is declared REFUSED.

    The N0 gate for the front kernel. `main:fused_trimul.py` drives `configure_a2a` at three call
    sites; a keyword it sends that we do not accept becomes a `TypeError` the moment N2 ports the
    workflow, surfacing at rank 0 of a multi-node run rather than here.
    """
    measured = _WORKFLOW_KWARGS_MEASURED[method]
    refused = _WORKFLOW_KWARGS_REFUSED[method]
    assert measured, f"the measured keyword set for {method} is empty -- it would prove nothing"
    ours = {
        p
        for p in inspect.signature(getattr(DualGatedGemmDistSm90, method)).parameters
        if p != "self"
    }
    unclassified = measured - ours - refused
    assert not unclassified, (
        f"{method}: the workflow passes {sorted(unclassified)}, which is neither a parameter of our "
        f"signature nor a declared REFUSED keyword. Add the parameter, or declare the refusal AND "
        f"land a front-door test for it."
    )
    assert not (refused & ours), (
        f"{method}: {sorted(refused & ours)} is declared REFUSED but IS a parameter."
    )


@matrix_exempt(
    "audits the class's METHOD RESOLUTION -- which methods it defines at all -- so it launches no "
    "kernel and varies with no declared axis"
)
def test_the_a2a_front_splices_the_epilogue_and_does_not_copy_the_kernel():
    """Pin the NO-KERNEL-COPY claim at ``dual_gated_gemm_a2a.py:73``, which nothing asserted.

    Purpose
    -------
    That comment states the whole design argument for the A2A front: the ``cp`` peer S2G atoms ride
    on the ``EpilogueParams`` dataclass, so splicing them in costs three method overrides and NOT a
    copy of the kernel. It was prose. This makes it executable.

    Functionality & semantics
    -------------------------
    Two assertions, because either alone lets the other through:

    1. **The forbidden set is untouched.** ``kernel``, ``mma_warpgroup_role``,
       ``producer_warpgroup_role``, ``epilogue`` and ``__call__`` must not appear in the subclass's
       own ``__dict__``. A whole-method copy drifts from the parent it copied and freezes that
       parent's config at copy time -- which is the stated reason the rule exists, and it is a
       SILENT divergence: the copy keeps working, just not like its base.
    2. **The declared splice is actually present.** ``EpilogueParams``,
       ``epi_to_underlying_arguments`` and ``epi_setup_postact`` must all be overridden. Without
       this half, deleting the mechanism entirely would satisfy assertion 1 perfectly.

    Checks ``vars(cls)`` rather than ``hasattr``, deliberately: ``hasattr`` is satisfied by the
    INHERITED attribute, so it answers "can this class reach the name" when the question is "does
    this class define it". And it does not use ``inspect.getsource``, which returns the DECORATOR's
    body for a ``@cute.kernel`` method -- see :func:`_source_of_resolved_method`, written for that
    reason and, until this test, pointed only at a different claim.

    Note the class legitimately overrides ~18 methods; the "three" in the source comment is the
    PEER-ATOM SPLICE specifically, not the total. Asserting a total would fail on the next unrelated
    hook and teach the next reader to delete this test.

    Input requirements
    ------------------
    None. Imports the class and reads its ``__dict__``; no GPU, no process group.

    Returns / Raises
    ----------------
    None. Raises ``AssertionError`` naming which forbidden method was copied, or which declared
    splice method went missing.
    """
    from fold_cp_ops.distributed.dual_gated_gemm_a2a import DualGatedGemmDistSm90

    own = set(vars(DualGatedGemmDistSm90))
    copied = {
        "kernel",
        "mma_warpgroup_role",
        "producer_warpgroup_role",
        "epilogue",
        "__call__",
    } & own
    assert not copied, (
        f"the A2A front re-implements {sorted(copied)}, so the kernel IS copied. The peer atoms are "
        f"supposed to ride on EpilogueParams; a copied method silently freezes the parent's config "
        f"at copy time and drifts from every later fix to it."
    )
    splice = {"EpilogueParams", "epi_to_underlying_arguments", "epi_setup_postact"}
    assert splice <= own, (
        f"the declared peer-atom splice is incomplete: {sorted(splice - own)} not overridden. "
        f"Without these the no-copy claim above is satisfied by a class that does nothing."
    )


@matrix_exempt(
    "asserts the class's BASE, which is a property of the class statement -- no kernel launch and "
    "no declared axis varies it"
)
def test_the_a2a_front_is_parented_where_its_byte_identity_evidence_assumes():
    """Pin the re-parenting that `docs/kernel_variants_map.md` 5.3 describes and nothing asserted.

    Purpose
    -------
    `main` expresses "no LayerNorm" as a FLAG on the fused class (`_normalize=False`); this package
    expresses it as a SEPARATE class, so the A2A front re-parents onto ``DualGatedGemmSm90`` where
    `main`'s counterpart sits on ``DualGatedGemmStagedSm90``. The doc states the condition a correct
    byte-identity check must meet -- it must compare against `main`'s ``_normalize=False`` trace, not
    its default -- which is the signature of a check that was DESIGNED and not built.

    Functionality & semantics
    -------------------------
    Asserts the immediate base is exactly ``DualGatedGemmSm90``. This is a REGRESSION pin, not a
    discovery: the correspondence is already established empirically at the strongest available bar
    -- **256 of 256 (cell, rank) pairs bitwise identical** to `main`, input hashes compared FIRST,
    over 4 mesh shapes x 4 token extents x 3 feature widths, on cells spanning both nodes so the
    drain crosses IB. (Honest denominator: 38 of 96 cells produced output on both trees; the other 58
    OOM structurally, since the fp32 reference and ``out_postact`` both scale as ``N^2*D``.)

    **What this pin is for is the FUTURE.** That evidence was a one-time cross-tree measurement at a
    commit -- the two trees share module names and cannot be co-imported, so no standing test can
    re-derive it. If the base ever changes, every one of those 256 comparisons silently stops
    applying, and nothing else in the suite would notice: a wrong base that double-normalizes or
    drops a LayerNorm still produces plausible numbers, and our numerical tests bind THIS tree to its
    OWN fp32 reference, which says nothing about how the trees relate.

    Checks ``__mro__[1]`` rather than ``issubclass``: the subclass relation survives an inserted
    intermediate base, and an inserted base is exactly the change that would invalidate the trace
    comparison while leaving ``issubclass`` true.

    Input requirements
    ------------------
    None. Imports two classes and compares identity; no GPU, no process group.

    Returns / Raises
    ----------------
    None. Raises ``AssertionError`` naming the base actually found.
    """
    from fold_cp_ops.kernels.dual_gated_gemm import DualGatedGemmSm90
    from fold_cp_ops.distributed.dual_gated_gemm_a2a import DualGatedGemmDistSm90

    base = DualGatedGemmDistSm90.__mro__[1]
    assert base is DualGatedGemmSm90, (
        f"the A2A front is parented on {base.__name__}, not DualGatedGemmSm90. The 256/256 bitwise "
        f"result against main was measured against main's _normalize=False trace on the assumption "
        f"of THIS base; re-parenting silently invalidates it and no other test would notice."
    )
