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

"""Unit for ``fold_cp_ops/distributed/workflows/trimul_autotuned.py`` -- the A2A-fused TriMul engine.

N2. The module is a BRING-BACK: the upstream's engine, moved into this tree's layout with the
class names the owner's rule requires (``DualGatedGemmDistStore``, ``GemmA2AStore``,
``TriMulAutotuned``) and with three call sites TRANSLATED rather than copied, because the kernels
they call were restructured during extraction. The tests are shaped around exactly those three
seams, since everything else is the upstream's code unchanged and is covered by the kernels' own
suites.

The seams:

1. **The front config resolver.** The upstream sourced it from
   ``dual_gated_gemm_stagec._stagec_heuristic_config``; no symbol of that name exists here, because
   extraction renamed the ``stagec`` fusion to ``alg_fold`` and folded both dual-gated fusions into
   one functor. `alg_fold_heuristic_config` answers the same question with the same return shape.
   That is a claim about BEHAVIOUR, so it is pinned against what `main` actually resolved over the
   40-cell acceptance grid -- measured in STEP 1, ``w8plan/step1/variants.csv``.
2. **The out-gate consumer.** One functor with a compile-time ``fusion_variant`` instead of two
   module-level entries, and the entry takes the output tensor and the CTA tile positionally rather
   than allocating and heuristing internally.
3. **The back config resolver**, unchanged, but pinned against the same measurement.

**Seam 2 was described here and NOT pinned, and it was inverted for the life of the port.**
``consumer="stagec"`` -- the default -- dispatched to ``prolog_ln``, the upstream's Stage D, where
the upstream runs Stage C; measured 1.24-1.28x on the out-gate. This paragraph used to say
``stagec`` had been renamed ``prolog_ln``, which is the inversion in prose. Describing a translation
is not testing it: seams 1 and 3 were pinned against measured values and were right, seam 2 got a
sentence and was wrong. `test_the_out_gate_variant_is_the_upstream_entry_it_replaces` now pins it,
and pins it on the functor's MECHANISM rather than on the variant string, so a rename cannot
satisfy it and a re-inversion cannot hide behind one.

**Why the pins are the measured values and not the source's constants.** Reading `main`'s resolver
and asserting it returns what its own code says would test the transcription, not the port. The
grid ran; the configs it elected are recorded; those are the numbers.
"""

from __future__ import annotations

import ast
import warnings

import pytest
import torch

from fold_cp_ops._internal.arch import get_device_capacity
from fold_cp_ops.distributed.workflows.trimul_autotuned import (
    DualGatedGemmDistStore,
    GemmA2AStore,
    TriMulAutotuned,
    _consumer_fusion_variant,
    _consumer_tile_n,
    _resolve_back_config,
    _resolve_front_config,
    _resolve_front_tile_n,
    _resolve_sm90_ib_config,
    incoming_store_variant,
)
from fold_cp_ops.testing.collective_guard import rank_invariant_skip
from fold_cp_ops.testing.kernel_matrix import matrix_exempt

#: Module-level, because the numeric guard's COVERAGE layer asks whether a comparison HAPPENED, not
#: whether a forbidden one was avoided -- and a file with no matrix-parametrized test has nothing for
#: it to check. Declaring it here rather than per test is the spelling the guard requires: "adding a
#: file must not be a way to opt out of element-wise comparison", so the opt-out is one visible line
#: with a reason rather than a decorator repeated until nobody reads it.
NUMERIC_EXEMPT = (
    "this module is `matrix_exempt`, so it has NO matrix-parametrized test for the numeric COVERAGE "
    "gate to inspect, and the guard requires an explicit declaration rather than silence in that "
    "case. That is the whole mechanism. It is NOT that the static pass objects to anything here: "
    "measured 2026-08-21, stripping this declaration leaves the static pass reporting ZERO findings. "
    "Most assertions compare resolved tile integers and class identity. The one that DOES compare a "
    "computed tensor -- `test_the_fused_chain_matches_the_fp32_oracle` -- uses `assert_elementwise` "
    "with `tolerance_bound`, i.e. the sanctioned element-wise form, so nothing pooled is hidden by "
    "this exemption. An earlier version of this text claimed the module compares no computed tensor, "
    "which was false and would have told a reader not to look."
)

pytestmark = matrix_exempt(
    "the subject is the workflow's host-side config resolution and its class surface -- the kernels "
    "it composes carry their own matrices, and duplicating their axes here would sweep the same "
    "tiles twice while testing the dispatch once"
)

#: What `main` ACTUALLY resolved, per cp, over the 40-cell acceptance grid at D=256. Read off
#: ``w8plan/step1/variants.csv`` (STEP 1's ``_front_cfg`` / ``_back_cfg`` columns), which recorded
#: the constructor's stored config on every rank of all 18 tags -- with zero rank disagreements.
#:
#: ``front`` is ``(tile_M, tile_N_effective, pingpong)`` AFTER `_resolve_front_tile_n` narrows the
#: candidate to the per-peer D-slice; ``back`` is ``(tile_M, tile_N, pingpong, cluster_shape)``.
_MEASURED = {
    # cp:  (front)                 (back 1-D)                    (back 2-D)
    2:  ((128, 128, False), (128, 128, False, (1, 2, 1)), None),
    4:  ((128, 64, False), (128, 128, False, (1, 2, 1)), (128, 128, False, (1, 1, 1))),
    8:  ((128, 32, False), (128, 128, False, (1, 2, 1)), (128, 128, False, (1, 1, 1))),
    # cp=16 is a HAS-IB mesh: its back config comes from the baked table, not the heuristic
    # (see test_the_baked_ib_table_matches_what_main_ran_across_the_nodes).
    16: ((128, 32, False), (128, 128, False, (1, 1, 1)), (128, 128, False, (1, 1, 1))),
}
_D = 256
_M = 2097152  # the front operand M extent at the grid's smallest cell; the heuristic keys on K, not M


def _requires_sm90():
    """Rank-invariant skip off SM90.

    Not a bare ``pytest.skip``: under ``tests/distributed/`` a divergent skip is a deadlock rather
    than a skip. Compute capability is a property of the machine the launch landed on and the nodes
    of one job are homogeneous, so every rank reaches the same verdict.
    """
    if not torch.cuda.is_available() or get_device_capacity()[0] != 9:
        rank_invariant_skip(
            "needs an SM90 (H100/H200) device",
            because=(
                "compute capability is a property of the machine the launch landed on; the nodes of "
                "one job are homogeneous, so no rank can reach a different verdict"
            ),
        )


@pytest.mark.parametrize("cp", sorted(_MEASURED))
def test_the_front_tile_matches_what_main_resolved(cp):
    """D2.4 -- the front resolver's effective tile equals `main`'s, per cp.

    Composition of `_resolve_front_config` (asked about the FULL D, since the plain dual's
    contraction is K=D) and `_resolve_front_tile_n` (which narrows to the per-peer D-slice). Pinning
    the composition rather than either half is deliberate: the split between them is an
    implementation detail, and only the product is what the kernel runs at.
    """
    _requires_sm90()
    dev = torch.device("cuda")
    tm, tn_cand, _pp_heuristic = _resolve_front_config(_M, _D, dev)
    tn = _resolve_front_tile_n(_D // cp, tn_cand)
    exp_tm, exp_tn, _ = _MEASURED[cp][0]
    assert (tm, tn) == (exp_tm, exp_tn), (
        f"cp={cp}: resolved front tile ({tm}, {tn}); `main` ran at ({exp_tm}, {exp_tn}) "
        f"(w8plan/step1/variants.csv)"
    )


def test_the_front_is_cooperative_whatever_the_heuristic_says():
    """The heuristic's ``pingpong`` is DISCARDED, and that is why the pins say False.

    Both trees' heuristics return ``{"pingpong": True}`` at ``K=D=256`` (their ``PINGPONG_MAX_K`` is
    512 in each). The staged front kernel is cooperative-only -- it asserts ``not pingpong`` -- so
    the workflow overrides the pick. Without this test the measured ``False`` pins look like a
    heuristic difference between the trees, which is what they first appeared to be.
    """
    _requires_sm90()
    _tm, _tn, pp = _resolve_front_config(_M, _D, torch.device("cuda"))
    assert pp is True, (
        "the heuristic itself still prefers ping-pong at K=256; if this flips, the pins below are "
        "no longer testing the override"
    )
    assert all(cfg[0][2] is False for cfg in _MEASURED.values()), (
        "every measured front config must record pingpong=False -- the workflow forces it"
    )


@pytest.mark.parametrize("cp", [2, 4, 8])
def test_the_back_tile_matches_what_main_resolved_on_nvlink(cp):
    """D2.4 -- the back HEURISTIC's tile and cluster equal `main`'s on an all-NVLink 1-D mesh.

    cp 2/4/8 fit one node, so the job has no IB peer and the baked table declines (below). This is
    therefore the path where `_resolve_back_config` is what actually decides.
    """
    _requires_sm90()
    got = _resolve_back_config(1, 2048, _D, cp, torch.bfloat16, torch.device("cuda"))
    exp = _MEASURED[cp][1]
    assert tuple(got) == exp, (
        f"cp={cp}: resolved back config {tuple(got)}; `main` ran at {exp} "
        f"(w8plan/step1/variants.csv)"
    )


@pytest.mark.parametrize("cp0,cp1", [(16, 1), (2, 8), (4, 4)])
def test_the_baked_ib_table_matches_what_main_ran_across_the_nodes(cp0, cp1):
    """D2.4 -- on a has-IB venue the BAKED table decides, and it matches the measurement.

    This is the path every cross-node cell of the acceptance grid took, and it is a different
    function from the one above: `_resolve_sm90_ib_config` returns a harvested ``(front, back)`` pair
    and short-circuits both heuristics. Pinning only `_resolve_back_config` would have left the three
    IB meshes -- half the elected variants -- untested, which is exactly what the first version of
    this file did: it asserted the heuristic's ``(1, 2, 1)`` against the measured ``(1, 1, 1)`` and
    failed, because at cp=16 the heuristic is not consulted at all.

    The front half is pinned too. STEP 1 recorded ``ib_wide_batch=256`` on every IB cell, which is
    the baked front's fourth element -- an independent corroboration that this table, and not the
    heuristic, is what ran.
    """
    _requires_sm90()
    baked = _resolve_sm90_ib_config(_D, cp0, cp1, True, True, False)
    assert baked is not None, (
        f"cp=({cp0},{cp1}) is a has-IB mesh of the acceptance grid; the baked table must answer"
    )
    front, back = baked
    assert front == (128, 32, False, 256), f"baked front {front}"
    assert back == (128, 128, False, (1, 1, 1)), f"baked back {back}"


@pytest.mark.parametrize("cp0", [2, 4, 8])
def test_the_baked_table_declines_without_ib_peers(cp0):
    """The baked table is venue-gated: no IB peer, no baked config.

    Without this, the test above would pass just as well if the table answered unconditionally --
    and an all-NVLink job would silently run a config harvested on a fabric it does not have.
    """
    _requires_sm90()
    assert _resolve_sm90_ib_config(_D, cp0, 1, True, False, False) is None


@pytest.mark.parametrize("D,expect", [(256, 128), (128, 128), (384, 128), (512, 128), (48, 96), (16, 32)])
def test_the_out_gate_tile_divides_the_two_activation_width(D, expect):
    """The translated consumer's tile rule: largest mult-of-32 <= 128 dividing ``2D``.

    Mirrors `_resolve_front_config`'s own fallback deliberately -- two different answers to "what
    tile does a 2D-wide dual take" is how a tile becomes a shape constraint. The small-D rows are
    the ones that would expose a divergence; at the production widths both rules say 128.
    """
    tn = _consumer_tile_n(D)
    assert tn == expect, f"D={D}: got tile_N={tn}, expected {expect}"
    assert (2 * D) % tn == 0 or tn == 128, f"tile_N={tn} must divide 2D={2 * D} unless it fell back"


@pytest.mark.parametrize(
    "consumer,variant,a_sweeps,a_in_regs",
    [
        # stagec  = upstream Stage C: A read RAW and ONCE, rank-one repair in the epilogue, and its
        #           two-A class asserts `not a_in_regs`.
        ("stagec", "alg_fold", 1, False),
        # staged_a_in_regs = upstream Stage D: A NORMALIZED in shared memory before the WGMMA, so
        #           the producer sweeps it twice; carries the register-source mainloop the name says.
        ("staged_a_in_regs", "prolog_ln", 2, True),
    ],
)
def test_the_out_gate_variant_is_the_upstream_entry_it_replaces(
    consumer, variant, a_sweeps, a_in_regs
):
    """`consumer` selects the functor that does what the upstream entry of that name does.

    **This is the seam this module names as translated and did not pin, and it was INVERTED.**
    ``consumer="stagec"`` -- the default, and what STEP 1 measured on all 28 cell-directions --
    dispatched to ``prolog_ln``, so every forward ran the upstream's Stage D where the upstream runs
    Stage C. Measured price at the workflow's own cell (D=256, MN-major value, one idle H100,
    `bench_timing.benchmark_single`): ``1.241x`` at N=2048 and ``1.284x`` at N=4096, against `main`'s
    `dual_gated_gemm_stagec`. Nothing failed; the output was right to two bf16 ULP, because the two
    variants compute the same function by different arithmetic.

    **The assertion is on the MECHANISM, not on the string.** Pinning ``_consumer_fusion_variant
    ("stagec") == "alg_fold"`` would restate the implementation and would pass just as happily if
    both names moved. So this resolves the name to the functor class the kernel module would build
    and interrogates the two attributes that DEFINE the upstream distinction:

    * ``_A_SWEEPS`` -- how many times the producer stages each k-tile. Stage C is 1, Stage D is 2.
      This is the whole speed argument, and it is the attribute a future refactor would have to
      change to re-create the defect.
    * ``_SUPPORTS_A_IN_REGS`` -- whether the register-source value mainloop exists. The upstream's
      Stage-C x_gate class asserts ``not self.a_in_regs``; its Stage-D class computes
      ``_value_a_in_regs`` from the value major, which is what the name ``staged_a_in_regs`` refers
      to.

    A rename that kept the behaviour passes this; a re-inversion fails it whatever the names are.

    Args:
        consumer: an upstream consumer name; both keys of the mapping are swept.
        variant: the `fusion_variant` it must produce.
        a_sweeps: the producer's sweeps over A in the functor that variant builds.
        a_in_regs: whether that functor declares the register-source value mainloop.
    """
    from fold_cp_ops.kernels.layernorm_dual_gated_gemm import _xgate_functor_for

    got = _consumer_fusion_variant(consumer)
    assert got == variant, (
        f"consumer={consumer!r} selected fusion_variant={got!r}, expected {variant!r}. The upstream "
        f"dispatches this name to its Stage-{'C' if consumer == 'stagec' else 'D'} entry."
    )
    cls = _xgate_functor_for(got)
    assert cls._A_SWEEPS == a_sweeps, (
        f"consumer={consumer!r} -> {got!r} -> {cls.__name__} sweeps A {cls._A_SWEEPS}x, expected "
        f"{a_sweeps}x. Sweep count IS the Stage-C/Stage-D distinction: one sweep reads A raw and "
        f"repairs in the epilogue, two normalize it in shared memory first."
    )
    assert cls._SUPPORTS_A_IN_REGS is a_in_regs, (
        f"consumer={consumer!r} -> {got!r} -> {cls.__name__} declares "
        f"_SUPPORTS_A_IN_REGS={cls._SUPPORTS_A_IN_REGS}, expected {a_in_regs}. The upstream's "
        f"Stage-C x_gate class asserts `not a_in_regs`; only its Stage-D class has the path."
    )


def test_the_out_gate_refuses_a_consumer_it_cannot_map():
    """An unknown `consumer` raises instead of falling through to a default.

    The inverted mapping was spelled as a ternary with an ``else``, so EVERY name that was not
    ``"staged_a_in_regs"`` -- including a typo, and including the correct ``"stagec"`` -- landed on
    one branch. A default is what let a mis-translation run as a kernel instead of failing at the
    front door, so the replacement has none and this pins that.

    ``"torch"`` is deliberately among the refused names here: `_consume` handles it before the
    mapping is reached, so the mapping accepting it would mean two places claimed the same
    decision.
    """
    for bad in ("stage_c", "staged", "torch", "", "ALG_FOLD"):
        with pytest.raises(ValueError, match="names no out-gate kernel"):
            _consumer_fusion_variant(bad)


def test_the_renamed_classes_are_the_ones_the_owner_named():
    """The three classes carry their functor names, and the upstream names are gone.

    The rename is by SYMBOL, never by substring: ``front_door`` means the API boundary and
    ``*_unpack`` names a reshard direction, so a blind ``front->`` substitution would break the
    testing machinery. This asserts the result of the symbol rename, and the module source is
    checked for the upstream class names to catch a half-applied one.
    """
    import fold_cp_ops.distributed.workflows.trimul_autotuned as mod

    for cls in (DualGatedGemmDistStore, GemmA2AStore, TriMulAutotuned):
        assert isinstance(cls, type), cls
    src = open(mod.__file__).read()
    for gone in ("_FrontFusedStoreStaged", "_BackFusedStore", "class FusedTriMul",
                 "DualGatedGemmStagedDistSm90"):
        assert gone not in src, f"{gone!r} survived the rename"
    # The three that must NOT have been renamed.
    assert "front_unpack" in src and "back_unpack" in src, (
        "*_unpack names a RESHARD DIRECTION, not a workflow half -- a substring rename would have "
        "eaten it"
    )


# --------------------------------------------------------------------------- #
# D2.1 / D2.2 -- the stores, driven end to end and compared PER ELEMENT.
# --------------------------------------------------------------------------- #

@pytest.fixture(scope="session", autouse=False)
def _nvshmem__distributed__workflows__trimul_autotuned(dist_manager):
    """Bring up nvshmem4py over the world group ONCE for the session.

    The conftest ``dist_manager`` fixture starts torch.distributed but NOT nvshmem, and the fused
    stores allocate symmetric recv buffers -- so without this the first store call dies at C level
    with ``nvshmem_team_translate_pe: NVSHMEM API called before NVSHMEM initialization has
    completed``, exit 255 and no Python traceback. Measured, twice.

    ``init_nvshmem()`` is COLLECTIVE (it broadcasts a unique id and barriers), so it must run in
    lockstep on every rank; a session-scoped fixture depending on ``dist_manager`` does exactly that,
    before the first test. Idempotent, and the conftest's session teardown already finalizes nvshmem,
    so this adds no teardown of its own.

    The name is the module's own path, per this suite's convention: a session fixture shared by two
    modules would tie their lifetimes together, and nvshmem cannot be re-initialized once finalized.
    """
    from fold_cp_ops.distributed.distributed_manager import DistributedManager

    DistributedManager.init_nvshmem()
    yield


_E2E_B, _E2E_N, _E2E_D = 1, 256, 128
_E2E_SEED = 20260821

#: Mesh specs the mesh-parametrized GPU tests draw from -- the equivalence gate and the
#: batch-plane gate, which must agree about which meshes count. `apply_mesh` SKIPS, rank-uniformly, any spec whose
#: rank product is not this launch's WORLD_SIZE, so one pool serves a 2-, 4-, 8- or 16-rank run and
#: the 2-D entries are what give `route2_ni` coverage -- the incoming fast path the flat specs never
#: reach. `(2, 4)` and `(4, 2)` are both here because `cp1` is the DISCRIMINATOR in
#: `incoming_store_variant`, so they are different cases and not a transposition of one.
_EQ_MESH_SPECS = (
    (("cp", 2),), (("cp", 4),), (("cp", 8),), (("cp", 16),),
    (("cp", (2, 2)),), (("cp", (2, 4)),), (("cp", (4, 2)),),
    (("cp", (4, 4)),), (("cp", (2, 8)),), (("cp", (8, 2)),),
)


def _pe_map(dist_manager):
    """This job's `PeMap`, from the manager's own device mesh with one `Shard` per mesh dim.

    Prefers the SUBGROUPS mesh, and that is a repair rather than a preference. A spec like
    ``(("cp", (4, 4)),)`` makes `create_grid_group` build TWO meshes: ``device_mesh`` over the
    named GROUPS -- one dim, ``cp`` of 16 -- and ``device_mesh_subgroups`` over the subgrid AXES,
    ``(4, 4)``. Reading the first gives ``cp_axis_sizes == (16,)``, so ``cp1 == 1``, so
    `incoming_store_variant` answers ``composite_k`` for EVERY spec in `_EQ_MESH_SPECS` and the
    2-D entries select nothing that the flat entries do not. Measured at 4 ranks on
    ``(("cp", (2, 2)),)``: ``device_mesh.ndim=1 shape=(4,) -> cp_axis_sizes=(4,)`` from the group
    mesh, ``(2, 2)`` from the subgroups mesh. Both this module's docstrings and the `_EQ_MESH_SPECS`
    comment already claimed the 2-D specs were what gave ``route2_ni`` coverage; before this they
    did not, and a mesh axis that changes no behaviour is decoration that satisfies a coverage
    check while testing one case repeatedly.

    Args:
        dist_manager: a live `DistributedManager` whose grid group has been built for THIS test
            (`apply_mesh` does that). Must be the object `apply_mesh` returned, not the session
            fixture -- the session one leaves ``device_mesh`` unset.

    Returns:
        ``(PeMap, mesh, placements)``. ``mesh`` and ``placements`` are returned alongside because
        every caller forwards them to `TriMulAutotuned` / `DTensor.from_local`, and re-deriving
        them at the call site is how a test comes to disagree with its own PeMap about the shard.

    Raises:
        AttributeError: if the manager has no device mesh at all, i.e. `apply_mesh` was not called.
    """
    from torch.distributed.tensor import Shard

    from fold_cp_ops.distributed.pe_map import PeMap

    # `reset_grid_groups` sets `_device_mesh_subgroups` back to None between meshes, so a flat spec
    # that follows a 2-D one cannot inherit the 2-D mesh -- the `is not None` test is exact, not a
    # heuristic. `is not None` rather than truthiness: DeviceMesh does not define __bool__ and an
    # implicit conversion would be an unnecessary bet on that staying true.
    sub = getattr(dist_manager, "device_mesh_subgroups", None)
    mesh = sub if sub is not None else dist_manager.device_mesh
    placements = [Shard(i + 1) for i in range(mesh.ndim)]
    return PeMap.from_mesh_placements(mesh, placements, distributed_manager=dist_manager), mesh, placements


def _local_shard(x_global, pm, rank):
    """This rank's ``(B, N_i_loc, N_j_loc, D)`` slab of the global input.

    1-D (``cp1 == 1``): the i axis is split ``cp`` ways and j is full. 2-D: both are split. Derived
    from the pe_map rather than from the rank arithmetic so the test cannot disagree with the kernel
    about which shard it owns -- the failure that looks like a numerics bug and is not.
    """
    cp0 = int(pm.cp_axis_sizes[0])
    cp1 = int(pm.cp_axis_sizes[1]) if len(pm.cp_axis_sizes) > 1 else 1
    N = x_global.shape[1]
    i_blk, j_blk = N // cp0, N // cp1
    r = int(pm.my_cp_rank)
    ri, rj = (r // cp1, r % cp1) if cp1 > 1 else (r, 0)
    return x_global[:, ri * i_blk:(ri + 1) * i_blk, rj * j_blk:(rj + 1) * j_blk, :].contiguous()


# (bias, batch) as explicit CELLS, not a cross product. Each cell builds a `TriMulAutotuned` with
# its own symmetric recvs, and `_symmetric_free` is a deliberate no-op (the MemPool recycles by
# refcount), so per-process cell count is charged against the symmetric heap. Measured: the full
# 2x3 product took this module from 56 tests to 62 and the SESSION aborted mid-run at the 7th
# engine -- while the same 12 cells pass in isolation. Batch is orthogonal to bias, so one B=2 cell
# buys the layout coverage that matters (`reshard.py`'s slot/b transpose) at +2 cells, not +6.
@pytest.mark.parametrize(
    "bias,batch",
    [("none", 1), ("out", 1), ("all", 1), ("none", 2)],
    ids=["none-B1", "out-B1", "all-B1", "none-B2"],
)
@pytest.mark.parametrize("direction", ["outgoing", "incoming"])
def test_the_fused_chain_matches_the_fp32_oracle(
    _nvshmem__distributed__workflows__trimul_autotuned, apply_mesh, world_size, device, direction,
    bias, batch,
):
    """D2.1 + D2.2 -- one rank's fused output equals the fp32 oracle on its own shard, PER ELEMENT.

    The ``batch`` axis is not decoration. `reshard.py`'s 1-D front unpack transposed the recv's
    (slot, b) modes for every B > 1 -- degenerate at B == 1, so a pool one point wide in the batch
    could not see it, and did not for the life of this port. The self-consistency test
    (`test_one_engine_serves_every_batch_extent`) catches that class too, but only by comparing the
    fused path against ITSELF; this one compares it against an INDEPENDENT fp32 reference computed
    from the un-sharded global input, which is the check that cannot be satisfied by two wrong
    halves agreeing.

    This is the test the whole port exists to pass: it drives `TriMulAutotuned.forward` through the
    real A2A stores on real symmetric memory, and compares against a reference computed from the
    GLOBAL input rather than from assembled per-rank results -- so a resharding bug cannot cancel
    itself.

    Element-wise, not pooled, and the bound is derived. A global rel_L2 averages over B*N*N*D
    elements, so a single collapsed token row -- the defect class these stores actually produce --
    barely moves the norm. `assert_elementwise` fails at the offending index with both values.

    Shapes are the smallest that still exercise the store: ``N=256`` splits at every cp of the
    acceptance grid, and ``D=128`` is one of the four production feature widths and was a live blind
    spot. Larger cells are N5/N6's job; this one has to run inside a unit suite.

    Args:
        bias: which projection biases the cell carries.

            * ``"none"`` -- the wave-1 contract; only the LayerNorm affine is non-trivial.
            * ``"out"`` -- adds ``p_out_b`` / ``g_out_b``, fused as ``bp`` / ``bg`` on
              `layernorm_dual_gated_gemm`.
            * ``"all"`` -- adds ``p_in_b`` / ``g_in_b`` as well, which reach the FRONT store's
              epilogue as the interleaved ``mRowVecBroadcast``.

            Each arm is a distinct compiled artifact: the bias terms are ``const_expr``-pruned when
            absent, so ``"none"`` does not run the adds against zeros, it does not contain them.
            All three are compared against the same fp32 oracle, which applies every bias it is
            given -- so an arm that dropped one would diverge rather than agree cheaply.
    """
    _requires_sm90()
    from fold_cp_ops.testing.numerics import assert_elementwise, tolerance_bound
    from tests.distributed.correctness_harness import global_oracle, make_weights

    # The mesh is built PER TEST through `apply_mesh`, not by `dist_manager`: the fixture leaves
    # `device_mesh` unset (measured: AttributeError on None.ndim) because the mesh is a parametrized
    # axis, not a launch-time choice. A flat cp mesh over this launch's world size is the shape the
    # acceptance grid's 1-D cells use.
    dist_manager = apply_mesh((("cp", int(world_size)),))
    pm, mesh, placements = _pe_map(dist_manager)
    cp0 = int(pm.cp_axis_sizes[0])
    cp1 = int(pm.cp_axis_sizes[1]) if len(pm.cp_axis_sizes) > 1 else 1
    if _E2E_N % cp0 or _E2E_N % cp1 or _E2E_D % int(pm.cp):
        rank_invariant_skip(
            f"N={_E2E_N} / D={_E2E_D} do not shard at cp=({cp0},{cp1})",
            because=(
                "the mesh shape is fixed by the launcher and identical on every rank, so the "
                "divisibility verdict is the same everywhere"
            ),
        )

    dt = torch.bfloat16
    weights = make_weights(
        _E2E_D, in_bias=(bias == "all"), out_bias=(bias in ("out", "all")),
        seed=_E2E_SEED, device=device,
    )
    g = torch.Generator(device="cpu").manual_seed(_E2E_SEED + 1)
    # Built on the CPU from an explicit generator, then moved: every rank must hold the SAME global
    # input, and a device randn would not agree across ranks even at a fixed seed.
    x_global = torch.randn(
        batch, _E2E_N, _E2E_N, _E2E_D, generator=g, dtype=torch.float32
    ).to(device)
    x_local = _local_shard(x_global, pm, int(pm.my_cp_rank)).to(dt)

    ftm = TriMulAutotuned(
        pm, batch, _E2E_N, _E2E_D, weights, dt,
        dynamic=True, device_mesh=mesh, placements=placements,
    )
    try:
        got = ftm.forward(x_local, direction)
    finally:
        try:
            ftm.free()
        except Exception:
            pass

    ref_global = global_oracle(
        x_global, weights, direction=direction, mask_global=None, eps=ftm.eps, row_tile=None
    )
    ref_local = _local_shard(ref_global, pm, int(pm.my_cp_rank))
    # bf16 through a two-GEMM chain with an fp32 accumulator: the bound scales with the reference,
    # which is what keeps it correct as D grows rather than silently loosening.
    bound = tolerance_bound(ref_local, atol=2e-2, rtol=6e-2)
    assert_elementwise(
        got.float(), ref_local, bound, what=f"trimul_autotuned {direction} B={batch}"
    )


# --------------------------------------------------------------------------- #
# N4 -- the public DTensor API.
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("mesh_spec", _EQ_MESH_SPECS, ids=lambda s: str(s[0][1]).replace(" ", ""))
@pytest.mark.parametrize("batch", [2, 3])
@pytest.mark.parametrize("direction", ["outgoing", "incoming"])
def test_one_engine_serves_every_batch_extent(
    _nvshmem__distributed__workflows__trimul_autotuned, apply_mesh, world_size, device, batch,
    direction, mesh_spec,
):
    """P0-L3 -- ONE compiled engine serves every batch extent, and each plane stays independent.

    Two claims, and the second is what makes this more than a shape check:

    1. **One compile.** ``TriMulAutotuned.__init__`` is entered EXACTLY once while a single instance
       forwards at B=1 and again at ``batch``. The batch extent is a runtime input now -- the kernel
       reads it off recv mode 4, the host derives ``M`` from the tensor in hand -- so a second batch
       must not produce a second engine. Before this landed, B=2 raised
       ``RuntimeError: shape '[131072, 256]' is invalid for input of size 67108864``.

    2. **Batch independence, PER ELEMENT.** Plane ``b`` of a ``batch``-wide forward must equal what
       that plane computes ALONE. A shape assertion cannot make this claim: the back store decodes
       its plane as ``d = L // B`` / ``b = L % B``, and a decode that is wrong but ALIGNED still
       writes a correctly-shaped output filled with another plane's values. Running the same rows
       singly is what separates those two outcomes.

    The two paths run the SAME kernels on the SAME weights and differ only in how many planes the
    GEMM's L axis carries, so the e2e tolerances used here are generous on purpose: the failure this
    guards against is a whole plane of another plane's values, not a last-bit drift.

    BOTH DIRECTIONS, and the incoming one is the point. ``direction="incoming"`` routes through the
    fast-path store the mesh selects -- ``composite_k`` on a 1-D token shard, ``route2_ni`` on a 2-D
    one (`incoming_store_variant`) -- which is where the batch plane had to be built. Those two
    recvs used to carry no plane at all and refused ``B > 1`` at construction, so this test was
    outgoing-only and the refusal was the only thing anyone could have asserted.

    **The variant kwarg is passed EXPLICITLY, and that is a repair, not a flourish.** The paragraph
    above was already written when this test constructed a bare ``TriMulAutotuned`` -- whose
    ``route2_ni`` and ``composite_k`` both default to ``False``, and which does NOT apply
    `incoming_store_variant` itself (only `trimul_a2a` and `TriangularMultiplication` do). So every
    "incoming" cell was running the PLAIN store, and the fast incoming path had NO ``B > 1``
    coverage anywhere in the suite while a docstring said it did. A claim in prose and a kwarg in
    the constructor are not the same artifact; this is the second.

    **MESH-PARAMETRIZED, because the mesh IS the variant.** ``cp1`` is the discriminator, so a
    flat-mesh-only run exercises ``composite_k`` and never ``route2_ni``. The two have different
    recv layouts (``(2*Dloc, B, cp*rpp_b)`` vs ``(2*Dloc, B, N_j, N_i)``) and therefore different
    plane strides, and on a CROSS-NODE launch they take different drain arms, so one proves nothing
    about the other. `apply_mesh` rank-uniformly skips any spec whose rank product is not this
    launch's ``WORLD_SIZE``, so the pool serves a 2-, 4-, 8- or 16-rank run unchanged.

    **What this covers that a single-node run cannot.** On an all-P2P mesh the IB drain
    ``const_expr``-collapses to the coupled TMA-S2G store, so the arm that carries the plane through
    ring METADATA is never executed. At 16 ranks over two nodes half the peers are IB, the drain is
    live, and the producer's plane stamp, the wide batch's plane pin and the consumer's plane index
    are all on the path from these inputs to these outputs.

    The variant attribute is asserted to have SURVIVED construction. A cross-node build at
    ``B > 1`` used to be silently demoted to the plain store by `TriMulAutotuned.__init__` -- which
    would leave every assertion below passing, on the wrong kernel. Reading the attribute back is
    what separates "the fast store is correct at B>1" from "something correct ran".

    What makes the incoming arm a real check rather than a shape check: BOTH B>1 defects on that
    path produced correctly-shaped, finite, plausible-magnitude output. One reshaped the recv with
    no batch mode (an exact 2x size error, which at least raised); the other sized the front recv at
    the CONSTRUCTOR's batch while the back read the runtime one, which did not raise -- it returned
    another plane's rows. Only the per-plane comparison below separates those from correct.

    ``engine`` is built at B=1 and then forwarded at ``batch`` with NO recompile, so this also pins
    that the batch extent stays a RUNTIME input on the fast paths: the transpose_in walk reads it
    off ``EpilogueArguments.token_grid_b`` rather than baking it, and ``rebind_M`` re-sizes the
    symmetric recv for it. That REBIND arm is reachable only through direct `TriMulAutotuned` use --
    both shipped entry points key their engine cache on the batch -- so it is asserted here rather
    than assumed to be covered by the front door.
    """
    _requires_sm90()
    from fold_cp_ops.testing.numerics import assert_elementwise, tolerance_bound
    from tests.distributed.correctness_harness import make_weights

    dist_manager = apply_mesh(mesh_spec)
    pm, mesh, placements = _pe_map(dist_manager)
    cp0 = int(pm.cp_axis_sizes[0])
    cp1 = int(pm.cp_axis_sizes[1]) if len(pm.cp_axis_sizes) > 1 else 1
    if _E2E_N % cp0 or _E2E_N % cp1 or _E2E_D % int(pm.cp):
        rank_invariant_skip(
            f"N={_E2E_N} / D={_E2E_D} do not shard at cp=({cp0},{cp1})",
            because=(
                "the mesh spec is a parametrized value every rank receives in the same order, so "
                "the divisibility verdict is identical on every rank"
            ),
        )
    # The store this configuration is SUPPOSED to run. A bare `TriMulAutotuned` does not apply the
    # rule itself, so it is applied here -- otherwise "incoming" silently means the plain store.
    variant = incoming_store_variant(direction, cp1, batch)
    variant_kw = {} if variant == "plain" else {variant: True}

    dt = torch.bfloat16
    weights = make_weights(_E2E_D, seed=_E2E_SEED, device=device)
    g = torch.Generator(device="cpu").manual_seed(_E2E_SEED + 5)
    # This rank's LOCAL shard for every plane, built on the CPU so the value does not depend on
    # which device happened to generate it.
    x_local = (
        torch.randn(batch, _E2E_N // cp0, _E2E_N // cp1, _E2E_D, generator=g, dtype=torch.float32)
        .to(device)
        .to(dt)
    )

    n_built = [0]
    original_init = TriMulAutotuned.__init__

    def counting_init(self, *args, **kwargs):
        n_built[0] += 1
        return original_init(self, *args, **kwargs)

    TriMulAutotuned.__init__ = counting_init
    try:
        engine = TriMulAutotuned(
            pm, 1, _E2E_N, _E2E_D, weights, dt,
            device_mesh=mesh, placements=placements, dynamic=True, **variant_kw,
        )
        try:
            built = (
                "composite_k" if engine.composite_k
                else ("route2_ni" if engine.route2_ni else "plain")
            )
            assert built == variant, (
                f"asked for the {variant!r} incoming store and the engine built {built!r}. A "
                f"cross-node build at B>1 used to be DEMOTED to the plain store here; if that is "
                f"back, every per-element assertion below is checking the wrong kernel and will "
                f"pass while the fast path stays unverified. hybrid_ib={engine.hybrid_ib}"
            )
            singly = [
                engine.forward(x_local[b : b + 1].contiguous(), direction).float().clone()
                for b in range(batch)
            ]
            together = engine.forward(x_local, direction).float()
        finally:
            engine.free()
    finally:
        TriMulAutotuned.__init__ = original_init

    assert n_built[0] == 1, (
        f"{n_built[0]} engines were constructed while forwarding at B=1 and B={batch} through one "
        f"instance; the batch extent is a runtime input and must not key a compile"
    )
    worst = 0.0
    for b in range(batch):
        ref = singly[b][0]
        worst = max(worst, float(assert_elementwise(
            together[b], ref, tolerance_bound(ref, atol=2e-2, rtol=6e-2),
            what=f"{direction} {variant} batch plane {b} of B={batch} vs the same rows run alone",
        )))
    # The two runs execute the SAME compiled kernels on the SAME weights and differ only in how many
    # planes the L axis carries, so the expected worst ratio is 0.0 -- bitwise. The bound above is
    # kept generous because the defect class this guards is a whole plane of another plane's values,
    # not a last-bit drift, and a bitwise assertion would make an unrelated scheduling change read
    # as a batch defect. The observed ratio is reported so a drift away from 0.0 is visible in the
    # run report rather than absorbed silently by the tolerance.
    print(f"[batch-plane] {direction} {variant} cp=({cp0},{cp1}) B={batch} worst ratio {worst:.6g}")


def test_the_incoming_store_variant_rule_is_total_and_depends_on_nothing_else():
    """The variant rule, enumerated over its WHOLE domain -- 2 directions x 2 mesh classes x 2
    batch classes.

    This is not a sample. `incoming_store_variant` is total over three small inputs, so the table
    below IS the function, and a change to the rule cannot pass by landing outside a sampled grid.

    The DISCRIMINATOR between the two fast variants is ``cp1`` alone, and it is now the ONLY input
    that moves the answer. ``batch`` used to gate whether either fast variant ran at all -- neither
    recv layout carried a batch plane -- and the right-hand half of this table read ``"plain"``.
    Both layouts grew one (``route2_ni``: ``(2*Dloc, B, N_j, N_i)``; ``composite_k``:
    ``(2*Dloc, B, cp*rpp_b)``, plane-major then slot), so the four ``batch=2`` rows below now match
    their ``batch=1`` twins EXACTLY. That equality is the assertion: it is what a reader checks to
    see that the limitation is gone rather than merely moved.

    ``batch`` stays in the signature, validated but inert, because deleting a positional parameter
    silently re-binds every call site already written.

    Before this rule was a function it lived inline in `trimul_a2a` and could only be exercised by
    building an engine -- so every case here previously cost a compile and a symmetric allocation.
    """
    from fold_cp_ops.distributed.workflows.trimul_autotuned import (
        INCOMING_STORE_VARIANTS,
        incoming_store_variant,
    )

    expected = {
        ("outgoing", 1, 1): "plain",
        ("outgoing", 1, 2): "plain",
        ("outgoing", 4, 1): "plain",
        ("outgoing", 4, 2): "plain",
        ("incoming", 1, 1): "composite_k",
        ("incoming", 1, 2): "composite_k",
        ("incoming", 4, 1): "route2_ni",
        ("incoming", 4, 2): "route2_ni",
    }
    got = {k: incoming_store_variant(*k) for k in expected}
    assert got == expected, f"variant rule changed: {got} != {expected}"
    assert set(got.values()) <= set(INCOMING_STORE_VARIANTS)
    # The batch column is INERT, stated as an equality over the whole domain rather than as four
    # more literals -- a rule that started reading `batch` again would fail here even at a value
    # this table does not list.
    for d in ("outgoing", "incoming"):
        for c in (1, 4):
            at1 = incoming_store_variant(d, c, 1)
            for bch in (2, 3, 7):
                assert incoming_store_variant(d, c, bch) == at1, (
                    f"batch moved the variant at ({d}, cp1={c}): batch={bch} gave "
                    f"{incoming_store_variant(d, c, bch)!r}, batch=1 gave {at1!r}"
                )

    # A typo must RAISE, not fall through to "plain" -- that would cost ~2x on incoming, silently.
    with pytest.raises(ValueError, match="outgoing"):
        incoming_store_variant("incomming", 1, 1)
    with pytest.raises(ValueError, match="cp1"):
        incoming_store_variant("incoming", 0, 1)
    with pytest.raises(ValueError, match="batch"):
        incoming_store_variant("incoming", 1, 0)

def test_a_misspelled_engine_knob_is_refused_at_the_call_the_caller_wrote():
    """`trimul_a2a` must reject an unknown ``fused_kwargs`` key HERE, not inside the lazy build.

    The engine is constructed on a cache MISS, well after the PeMap is built and nvshmem is up, so
    a typo used to surface as ``TypeError: __init__() got an unexpected keyword argument`` from a
    constructor the caller never named -- on a cluster, having already spent the allocation. The
    check compares against `TriMulAutotuned.__init__`'s LIVE keyword-only signature, so renaming a
    knob on the engine cannot leave a stale allow-list behind, and it suggests the nearest legal
    name because the thing being caught is a spelling.

    GPU-free: it raises before any device work, which is the whole point.
    """
    import torch as _torch

    from fold_cp_ops.distributed.workflows.trimul_autotuned import trimul_a2a

    # D=4 is deliberately NOT 16-byte aligned for a 16-bit activation, so the legal-knob case below
    # stops at a NAMED shape guard instead of wandering into an empty weight dict -- the assertion
    # is about WHICH error arrives, so the error has to be predictable.
    x = _torch.zeros(1, 8, 8, 4)
    with pytest.raises(TypeError, match=r"hybird_ib.*did you mean 'hybrid_ib'"):
        trimul_a2a(x, {}, _torch.bfloat16, hybird_ib=True)
    with pytest.raises(TypeError, match=r"front_tile_mm.*did you mean 'front_tile_mn'"):
        trimul_a2a(x, {}, _torch.bfloat16, front_tile_mm=(128, 128))
    # A LEGAL knob must pass this gate and fail later on its own merits, or the guard is a wall.
    with pytest.raises(ValueError, match=r"16-byte aligned"):
        trimul_a2a(x, {}, _torch.bfloat16, consumer="stagec")

def test_the_public_surface_is_exactly_two_names():
    """``__all__`` is the API, and the engine is NOT in it.

    The upstream shipped TWO public entries -- a raw-tensor engine under a DTensor wrapper -- and
    this tree deliberately carries one. A name leaking back into ``__all__`` is how the second API
    returns under a different label.
    """
    import fold_cp_ops.distributed.workflows.trimul_autotuned as mod

    assert set(mod.__all__) == {"TriangularMultiplication", "trimul_a2a"}, mod.__all__
    assert "TriMulAutotuned" not in mod.__all__, "the engine stays internal"


def test_the_retired_upstream_names_are_gone():
    """None of the ``fused_trimul`` family survives as a symbol.

    Checked against the module SOURCE as well as its namespace: a name can persist in a docstring
    or a deferred import and still be what a reader copies.
    """
    import fold_cp_ops.distributed.workflows.trimul_autotuned as mod

    for gone in ("FusedTriMulCP", "fused_trimul_dtensor", "_get_fused_trimul", "_trimul_cp1"):
        assert not hasattr(mod, gone), f"{gone!r} is still bound"
        assert gone not in open(mod.__file__).read(), f"{gone!r} survives in the source"


def test_the_shipped_tree_never_selects_the_prolog_ln_out_gate():
    """No shipped caller sets ``consumer="staged_a_in_regs"``, so ``prolog_ln`` is off the cp>1 path.

    Purpose
    -------
    Pin a REACHABILITY claim that `docs/kernel_variants_map.md` asserts in prose and nothing
    executed. The prose has already flipped once and carries a correction notice, which is the
    argument for making it a test: a paragraph that has been wrong before will be read as right
    again.

    Functionality & semantics
    -------------------------
    Scans the SOURCE of the shipped package and the benchmark harness for a `consumer=` keyword
    bound to a string literal, and asserts none of them is ``"staged_a_in_regs"`` -- the only name
    that maps to ``prolog_ln``. Scans source rather than calling anything, so it needs no GPU and no
    process group, the same bargain `test_the_retired_upstream_names_are_gone` strikes.

    Deliberately does NOT assert the option is unreachable in principle. It is a public constructor
    argument and an explicit caller may still pass it; `_CONSUMER_VARIANT` maps it, and
    `test_the_out_gate_variant_is_the_upstream_entry_it_replaces` pins where it lands. The claim
    here is narrower and is the one the docs make: nothing WE ship selects it.

    **Why this claim is load-bearing.** `stagec` is both the default and the only consumer any
    shipped caller passes, so the two arms of the mapping are not symmetric in exposure: one takes
    100% of the traffic and the other takes none. That is precisely the arrangement in which an
    inverted mapping cannot be caught by use -- and it was inverted, for the life of the port. If a
    caller ever DOES start passing ``staged_a_in_regs``, this test failing is the notice that the
    unexercised arm just became live and needs its own coverage.

    Input requirements
    ------------------
    None. Reads files under the repo root, resolved from this test's own location, so it does not
    depend on the working directory.

    Returns / Raises
    ----------------
    None. Raises ``AssertionError`` naming the file, line and consumer value of any shipped call
    site that selects the prolog_ln out-gate.
    """
    import re
    from pathlib import Path

    root = Path(__file__).resolve().parents[3]
    pat = re.compile(r"""(?<![_\w])consumer\s*=\s*["']([A-Za-z_][A-Za-z0-9_]*)["']""")
    offenders, scanned = [], 0
    for sub in ("fold_cp_ops", "benchmark"):
        for f in (root / sub).rglob("*.py"):
            scanned += 1
            for n, line in enumerate(f.read_text(errors="replace").splitlines(), 1):
                if line.lstrip().startswith("#"):
                    continue
                for m in pat.finditer(line):
                    if m.group(1) == "staged_a_in_regs":
                        offenders.append(f"{f.relative_to(root)}:{n}: {line.strip()}")

    # A scan that matched nothing because it scanned nothing is not evidence; the repo has produced
    # that failure more than once. Assert the denominator before asserting the finding.
    assert scanned > 50, f"only {scanned} files scanned -- the walk is broken, not the tree clean"
    assert not offenders, (
        "a shipped caller selects the prolog_ln out-gate, which the docs say is off the cp>1 "
        "path:\n  " + "\n  ".join(offenders)
    )


def test_the_weight_key_sets_agree_at_import():
    """The CP weight dict and the single-device entry's weight kwargs are the SAME set.

    The cp=1 fallback splats ``w`` straight through, so a rename on either side would silently drop
    a weight at the first single-device call. The module asserts this at IMPORT; this test is what
    makes that assertion's failure legible rather than an import error in someone else's traceback.
    """
    import fold_cp_ops.distributed.workflows.trimul_autotuned as mod

    from fold_cp_ops.distributed.trimul_weights import W_KEYS, W_PROJ_BIAS_KEYS

    assert set(mod._W_ALL_KEYS) == set(W_KEYS) | set(W_PROJ_BIAS_KEYS)
    assert set(mod._W_ALL_KEYS) == set(mod._TRIMUL_W_KWARGS)


@pytest.mark.parametrize("x", ["plain_tensor"])
def test_a_plain_tensor_routes_to_the_single_device_path(x):
    """No mesh at all means cp=1: the dispatch must not build a process group or symmetric memory.

    This is D4.2's property. It is asserted on the PREDICATE rather than by running the kernel,
    because the thing being checked is that nothing distributed happens -- and a test that runs the
    kernel to find out has already paid for the thing it is checking is absent.
    """
    import fold_cp_ops.distributed.workflows.trimul_autotuned as mod

    assert mod._is_single_device(torch.zeros(1, 4, 4, 8)) is True


# ── the OOM precheck's decision must be RANK-UNIFORM, or its refusal is a hang ─────────────────
@matrix_exempt(
    "the subject is the symmetric-allocation PRECHECK's collective behaviour -- there is no kernel, "
    "shape pool or dtype axis, only a divergent free-memory reading"
)
def test_a_DIVERGENT_free_memory_reading_refuses_on_EVERY_rank(dist_manager, monkeypatch):
    """The failure this prevents is a DEADLOCK, not a wrong number.

    `_refuse_unsatisfiable_symmetric_request` compares the request against THIS rank's free memory.
    Every rank computes the same requested size and reads its own free figure, so two ranks can
    disagree -- and `nvshmem_malloc` is COLLECTIVE, so a rank that refuses alone leaves its peers
    blocked in the allocator until a watchdog kills the job.

    The earlier answer to this was "the raise flows into the caller's `gated_skip`, which all-reduces
    the decision". That holds on the TEST path and not on the production one: `gated_skip` is a
    pytest facility, and `_symmetric_empty` has eighteen call sites in the workflow's own
    constructors. Without the reduce inside the precheck, the guard covers exactly the path that does
    not ship.

    **Two assertions, and the first is the control.** A test in which every rank refuses anyway would
    pass against a completely unreduced implementation, so this first proves the LOCAL decisions
    would have diverged -- rank 0 short, every peer comfortable -- and only then that the reduced
    outcome is unanimous. Without the control the test cannot tell "the reduce worked" from "there
    was nothing to reduce".

    Bounded by an explicit deadline because the failure mode under test IS a hang: a regression here
    does not fail, it stops, and a stopped test is indistinguishable from a slow one.
    """
    import torch.distributed as dist

    from fold_cp_ops.distributed.workflows import trimul_autotuned as ta

    dm = dist_manager
    dm.init_nvshmem()
    rank = dist.get_rank()
    dev = torch.cuda.current_device()

    # A modest request every rank can really satisfy, so nothing is actually allocated short.
    n_bytes = 1 << 20
    real_free, real_total = torch.cuda.mem_get_info(dev)

    # Rank 0 alone sees a device with almost nothing left. Every peer sees the truth. This is the
    # divergence, injected at the only place the function reads per-rank state.
    def _skewed(device=None):
        return (n_bytes // 2, real_total) if rank == 0 else (real_free, real_total)

    monkeypatch.setattr(torch.cuda, "mem_get_info", _skewed)

    # CONTROL: the local decisions really do disagree, so the reduce has something to do.
    local_fits = n_bytes <= _skewed(dev)[0]
    assert local_fits == (rank != 0), (
        f"rank {rank}: the skew did not produce the intended disagreement (local_fits={local_fits})"
    )
    votes = torch.tensor([1 if local_fits else 0], device=dev, dtype=torch.int32)
    gathered = [torch.zeros_like(votes) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, votes)
    seen = {int(g.item()) for g in gathered}
    assert seen == {0, 1}, (
        f"the control failed: every rank voted the same ({seen}), so this test would pass against an "
        f"unreduced implementation and proves nothing"
    )

    # THE ASSERTION: every rank refuses, together, and nobody enters the collective allocator.
    with pytest.raises(torch.cuda.OutOfMemoryError) as ei:
        ta._symmetric_empty((n_bytes,), dtype=torch.uint8, device=dev)
    msg = str(ei.value)
    if rank == 0:
        assert "this rank has" in msg, msg
    else:
        assert "a PEER rank could not fit it" in msg, (
            f"rank {rank} refused, but reported the refusal as its OWN shortage. A rank with memory "
            f"to spare must say a peer was short, or the first person to read this log looks for a "
            f"leak on the wrong rank.\n{msg}"
        )
    # Reaching here on every rank is the real result: the run did not hang.
    dist.barrier()


@matrix_exempt("the subject is the precheck's fallback with no process group; no kernel axis")
def test_a_measurement_FAILURE_votes_fits_rather_than_refusing_its_peers(dist_manager, monkeypatch):
    """A rank that cannot read its free memory must not refuse the job on everyone's behalf.

    It also must not skip the reduce, which is the subtler half: an early `return` on the error path
    would leave that rank out of a collective its peers are already in, reintroducing the very hang
    the reduce exists to remove. So the failure votes FITS and still participates.
    """
    import torch.distributed as dist

    from fold_cp_ops.distributed.workflows import trimul_autotuned as ta

    dm = dist_manager
    dm.init_nvshmem()
    dev = torch.cuda.current_device()

    def _broken(device=None):
        raise RuntimeError("mem_get_info unavailable on this driver")

    monkeypatch.setattr(torch.cuda, "mem_get_info", _broken)
    t = ta._symmetric_empty((1 << 10,), dtype=torch.uint8, device=dev)
    assert t.numel() == 1 << 10
    dist.barrier()  # if any rank had skipped the reduce, this is where the survivors would stop


# --------------------------------------------------------------------------- #
# TriangularMultiplication -- the constructor contract
#
# Host-side only: every test below CONSTRUCTS and inspects. None runs a forward, so none needs a
# GPU, a mesh or a process group, and none skips -- which is what keeps them inside the
# `tests/distributed/**` skip rules without needing a gate.
# --------------------------------------------------------------------------- #

_TRIMUL_SUBMODULE_NAMES = ("norm_in", "p_in", "g_in", "norm_out", "p_out", "g_out")


def _layer(d=16, *, dtype=torch.float32):
    """A minimal TriMul-shaped layer. Deliberately a bare ``nn.Module``, NOT the reference class."""
    import torch.nn as nn

    m = nn.Module()
    m.norm_in = nn.LayerNorm(d, eps=1e-5).to(dtype)
    m.norm_out = nn.LayerNorm(d).to(dtype)
    m.p_in = nn.Linear(d, 2 * d, bias=False).to(dtype)
    m.g_in = nn.Linear(d, 2 * d, bias=False).to(dtype)
    m.p_out = nn.Linear(d, d, bias=False).to(dtype)
    m.g_out = nn.Linear(d, d, bias=False).to(dtype)
    return m.eval()


def _tm(**kw):
    """Construct a `TriangularMultiplication` with CPU-only defaults, overridable per keyword.

    Defaults `layer` to `_layer()`, `direction` to ``"outgoing"`` and both `device_mesh` and
    `distributed_manager` to ``None`` -- the cp=1 shape, which needs no group, no mesh and no
    nvshmem. Any keyword given wins, so a test that is ABOUT one of those passes just that one.
    """
    from fold_cp_ops.distributed.workflows.trimul_autotuned import TriangularMultiplication

    kw.setdefault("layer", _layer())
    kw.setdefault("direction", "outgoing")
    kw.setdefault("device_mesh", None)
    kw.setdefault("distributed_manager", None)
    return TriangularMultiplication(**kw)


def test_the_module_exposes_the_six_canonical_submodule_names():
    """The names ARE the drop-in contract: the CP reference wrappers set exactly these
    six, so anything walking a model by attribute must not be able to tell the two apart."""
    mod = _tm()
    for name in _TRIMUL_SUBMODULE_NAMES:
        assert hasattr(mod, name), f"missing {name}"
    assert {n for n, _ in mod.named_children()} == set(_TRIMUL_SUBMODULE_NAMES)


def test_a_layer_that_is_not_the_reference_class_is_accepted():
    """`layer` is duck-typed on purpose -- that is what keeps any foreign layer off this
    package's dependency list. `_layer` is a bare `nn.Module`; if this ever starts requiring the reference
    type, the decoupling has been lost."""
    import torch.nn as nn

    assert type(_layer()) is nn.Module
    _tm()  # must not raise


@pytest.mark.parametrize("bad", ["out", "OUTGOING", "forward", ""])
def test_an_unrecognised_direction_is_refused_at_the_front_door(bad):
    """`direction` selects the back-half einsum. A typo must not reach a kernel and quietly compute
    a different contraction."""
    with pytest.raises(ValueError, match="direction"):
        _tm(direction=bad)


def test_a_missing_submodule_is_named_rather_than_guessed():
    """A layer short one projection fails HERE, naming it, not with a shape error frames later."""
    lyr = _layer()
    del lyr.g_out
    with pytest.raises(AttributeError, match="g_out"):
        _tm(layer=lyr)


def test_the_layernorm_parameters_stay_fp32_whatever_dtype_is_asked_for():
    """REGRESSION. `dtype` is the COMPUTE dtype, not a cast over the weight dict.

    An earlier version forwarded it into `weights_from_trimul_module`, which casts every key --
    including the LN gains/biases, which `tensor_contract.check_tensor` requires at fp32 because a
    narrower gain is a silent loss rather than a conversion. That reached the cluster and failed
    every cross-node cell with

        ValueError: norm_weight must be torch.float32; got torch.bfloat16

    Coercion is UPWARD only: a bf16 layer is promoted, and an fp32 layer is never round-tripped
    through bf16 (which would mangle a real checkpoint's gain while leaving `torch.ones` intact --
    i.e. invisible to a synthetic-weight test)."""
    from fold_cp_ops.distributed.workflows.trimul_autotuned import _W_LN_KEYS

    for layer_dtype in (torch.float32, torch.bfloat16):
        mod = _tm(layer=_layer(dtype=layer_dtype), dtype=torch.bfloat16)
        for key in _W_LN_KEYS:
            assert mod._w[key].dtype is torch.float32, f"{key} at layer dtype {layer_dtype}"


def test_an_fp32_layers_norm_gain_is_carried_bitwise_not_round_tripped():
    """The upward-only rule with teeth: a non-trivial fp32 gain must survive EXACTLY. Casting to
    bf16 and back would leave `torch.ones` unchanged, so a test using default LayerNorm weights
    cannot see the bug it is meant to catch."""
    lyr = _layer()
    with torch.no_grad():
        lyr.norm_in.weight.copy_(torch.linspace(0.9, 1.1, lyr.norm_in.weight.numel()))
    from fold_cp_ops.testing.numerics import assert_bitwise

    mod = _tm(layer=lyr, dtype=torch.bfloat16)
    assert_bitwise(mod._w["norm_in_w"], lyr.norm_in.weight.detach(), what="norm_in_w carried")


def test_the_constructor_mesh_is_checked_against_the_input_not_dispatched_on():
    """`device_mesh` records what the instance was built for. A DTensor arriving on another mesh
    raises rather than resharding onto a PE map the compiled instance was not built for -- and the
    check fires BEFORE any dispatch, so no nvshmem symbol is touched on the error path."""

    class _FakeMesh:
        """A 1-device mesh stand-in: enough for the constructor to derive cp == 1 (so no PeMap and
        no nvshmem), and distinct by identity so the forward check has something to disagree with.
        A bare string no longer works -- `__init__` now READS the mesh (`.ndim`, `.mesh`) to resolve
        the placements and the cp map, which is the whole point of the hoist."""

        def __init__(self, tag):
            """`tag` is the identity the equality check compares; the rest is the minimum a mesh
            must expose for `_resolve_placements` / `_resolve_cp` to read it as 1-device."""
            self.tag, self.ndim, self.mesh = tag, 1, torch.zeros(1, dtype=torch.int64)

        def __eq__(self, other):
            """Equal iff same tag -- so two differently-tagged stand-ins disagree the way two real
            meshes would, which is what the forward check under test needs."""
            return isinstance(other, _FakeMesh) and other.tag == self.tag

        __hash__ = None

    class _FakeDT:
        """A DTensor stand-in carrying only the attribute `forward`'s mesh check reads, so the
        check is reached without a real distributed tensor."""

        device_mesh = _FakeMesh("A")

    mod = _tm(device_mesh=_FakeMesh("B"))
    with pytest.raises(ValueError, match="device_mesh"):
        mod(_FakeDT())


def test_the_constructor_refuses_a_device_mesh_it_cannot_read():
    """`__init__` derives the placements, cp and the PeMap from `device_mesh`, so an object that
    merely compares equal is no longer sufficient -- and the refusal must name the argument rather
    than surfacing as an `AttributeError` from inside a private resolver."""
    with pytest.raises(TypeError, match="device_mesh must be a DeviceMesh"):
        _tm(device_mesh="mesh-B")


def test_a_three_dim_mesh_refuses_to_default_its_placements():
    """The convention `[Shard(i + 1) ...]` would put `Shard(3)` on the FEATURE axis of a 3-D mesh.
    `validate_trimul_sharding` would refuse that with a message about LayerNorm locality -- true,
    but it names the wrong cause for a caller who passed no placements at all."""

    class _M3:
        """A 3-D mesh stand-in: only `.ndim` and `.mesh` are read before the refusal fires."""

        ndim, mesh = 3, torch.zeros(8, dtype=torch.int64)

    with pytest.raises(ValueError, match="cannot default the placements"):
        _tm(device_mesh=_M3())


def test_explicit_placements_must_have_one_entry_per_mesh_dim():
    """There is no padding rule: a short sequence is a caller error, not a prefix."""
    from torch.distributed.tensor import Shard

    class _M2:
        """A 2-D mesh stand-in; the length check fires before anything else is read."""

        ndim, mesh = 2, torch.zeros(4, dtype=torch.int64)

    with pytest.raises(ValueError, match="one placement per mesh dim|placements has 1 entries"):
        _tm(device_mesh=_M2(), placements=[Shard(1)])


def test_free_is_a_no_op_on_a_module_that_never_ran():
    """`free()` is called unconditionally in teardown paths. On a module whose cache is empty it
    must neither raise nor touch nvshmem."""
    mod = _tm()
    mod.free()
    mod.free()
    assert mod._engines == {}


# --------------------------------------------------------------------------- #
# TriangularMultiplication -- the shipped API, FORWARD, carrying every bias it accepts
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("direction", ["outgoing", "incoming"])
def test_the_shipped_module_matches_the_oracle_with_every_bias_it_accepts(
    _nvshmem__distributed__workflows__trimul_autotuned, apply_mesh, world_size, device, direction
):
    """`TriangularMultiplication` DTensor-in/DTensor-out, with all FOUR accepted biases non-zero.

    Purpose
        Close the gap between "the engine handles these biases" and "the shipped API does". Every
        other test on this class CONSTRUCTS and inspects -- none ran a forward -- and the one that
        drives the fused chain (`test_the_fused_chain_matches_the_fp32_oracle`) calls
        `TriMulAutotuned` directly, one layer beneath. So the DTensor front door had never executed
        with a bias of any kind, and the two entry points do NOT share their weight handling: this
        one goes through `weights_from_trimul_module` plus an fp32 coercion of the LayerNorm keys that
        the direct path does not perform.

    Semantics
        Builds a TriMul-shaped layer from a weight dict whose LayerNorm affine is off the identity
        and whose OUT projections carry biases, wraps the local shard as a `DTensor` on this
        launch's mesh, and compares the returned DTensor's local part to the fp32 oracle computed
        from the GLOBAL input -- so a resharding error cannot cancel itself.

        The four biases this covers are the four the workflow wires: ``norm_in_b`` (into
        `layernorm_fwd`), ``norm_out_b``, ``p_out_b`` and ``g_out_b`` (into
        `layernorm_dual_gated_gemm` as ``norm_bias``/``bp``/``bg``). ``p_in_b``/``g_in_b`` are the
        only two that are not, and `test_trimul_weights` covers their refusal.

        **The fp32 coercion is the specific thing worth exercising here.** The constructor casts the
        LayerNorm keys upward because `layernorm_dual_gated_gemm` refuses a non-fp32 gain, and the
        engine's own ``self._norm_out_b = w["norm_out_b"]`` performs no cast -- it trusts its caller.
        At a zero bias that contract is unobservable in the output.

    Args:
        direction: which axis op 7 contracts; both are separate compiled artifacts.
    """
    _requires_sm90()
    from torch.distributed.tensor import DTensor

    from fold_cp_ops.distributed.trimul_weights import trimul_module_from_weights
    from fold_cp_ops.distributed.workflows.trimul_autotuned import TriangularMultiplication
    from fold_cp_ops.testing.numerics import assert_elementwise, tolerance_bound
    from tests.distributed.correctness_harness import global_oracle, make_weights

    dist_manager = apply_mesh((("cp", int(world_size)),))
    pm, mesh, placements = _pe_map(dist_manager)
    cp0 = int(pm.cp_axis_sizes[0])
    cp1 = int(pm.cp_axis_sizes[1]) if len(pm.cp_axis_sizes) > 1 else 1
    if _E2E_N % cp0 or _E2E_N % cp1 or _E2E_D % int(pm.cp):
        rank_invariant_skip(
            f"N={_E2E_N} / D={_E2E_D} do not shard at cp=({cp0},{cp1})",
            because=(
                "the mesh shape is fixed by the launcher and identical on every rank, so the "
                "divisibility verdict is the same everywhere"
            ),
        )

    dt = torch.bfloat16
    weights = make_weights(_E2E_D, out_bias=True, seed=_E2E_SEED, device=device)
    assert weights["p_out_b"] is not None and weights["g_out_b"] is not None
    layer = trimul_module_from_weights(weights)

    g = torch.Generator(device="cpu").manual_seed(_E2E_SEED + 1)
    x_global = torch.randn(
        _E2E_B, _E2E_N, _E2E_N, _E2E_D, generator=g, dtype=torch.float32
    ).to(device)
    x_local = _local_shard(x_global, pm, int(pm.my_cp_rank)).to(dt)
    x_d = DTensor.from_local(x_local, mesh, placements)

    tm = TriangularMultiplication(layer, direction, mesh, dist_manager, dtype=dt)
    try:
        got_d = tm(x_d)
        got = got_d.to_local() if hasattr(got_d, "to_local") else got_d
    finally:
        for inst in list(getattr(tm, "_engines", {}).values()):
            try:
                inst.free()
            except Exception:
                pass

    ref_global = global_oracle(
        x_global, weights, direction=direction, mask_global=None, eps=1e-5, row_tile=None
    )
    ref_local = _local_shard(ref_global, pm, int(pm.my_cp_rank))
    assert got.shape == ref_local.shape, (tuple(got.shape), tuple(ref_local.shape))
    bound = tolerance_bound(ref_local, atol=2e-2, rtol=6e-2)
    assert_elementwise(
        got.float(), ref_local, bound, what=f"TriangularMultiplication {direction} with biases"
    )


@matrix_exempt("a constructor contract on weights; no kernel shape axis applies")
@pytest.mark.parametrize("sub", ["p_in", "g_in", "p_out", "g_out"])
def test_the_shipped_module_accepts_a_bias_on_every_projection(sub):
    """The constructor accepts a bias on ANY of the four projections -- none is refused.

    This test previously asserted the opposite for ``p_in``/``g_in``, and the flip is the point: the
    front store now fuses them as the interleaved ``mRowVecBroadcast``, so a refusal here would
    reject a weight the engine can compute. Kept as a test rather than deleted because "the API
    accepts every bias the layer can carry" is the contract a caller relies on, and it is exactly the
    thing that was silently false for the OUTPUT pair for the whole port.

    Host-side -- weights are read before any mesh or device work -- so this needs no GPU and no
    process group and cannot skip.
    """
    import torch.nn as nn

    from fold_cp_ops.distributed.workflows.trimul_autotuned import TriangularMultiplication

    d = 16
    layer = _layer(d)
    width = 2 * d if sub in ("p_in", "g_in") else d
    setattr(getattr(layer, sub), "bias", nn.Parameter(torch.zeros(width), requires_grad=False))
    tm = TriangularMultiplication(layer, "outgoing", None, None)
    assert tm._w[f"{sub}_b"] is not None, f"{sub}_b was dropped on the way in"


# --------------------------------------------------------------------------- #
# R6/R7: the module path and the functional path must build the SAME engine
# --------------------------------------------------------------------------- #
def _capture_engine_ctor(monkeypatch):
    """Install a `TriMulAutotuned.__init__` wrapper that records its arguments; return the sink.

    Purpose
        `§4.2` moved the engine's argument construction out of `trimul_a2a` and into
        `TriangularMultiplication.__init__` + `prepare`. "The same arguments" is the entire
        correctness claim of that move, and it is not observable from the output alone -- two
        different configurations can agree numerically and differ in which store, which drain and
        which tile they used. So it is read off the constructor.

    Functionality & semantics
        Records the positional shape/dtype facts and every keyword, JSON-reduced so a tensor
        contributes its shape rather than its values (the two paths pass the SAME weight dict
        object, so comparing values would be a tautology; comparing shapes still catches a dropped
        key). Sink is cleared by the caller between paths.

    Args:
        monkeypatch: pytest's fixture. The patch is reverted at teardown, so a failure cannot leave
            a recording constructor installed for the rest of the session.

    Returns:
        A dict that gains ``{"args": {...}, "kwargs": {...}}`` on each construction.
    """
    import functools

    from fold_cp_ops.distributed.workflows import trimul_autotuned as T

    sink: dict = {}
    orig = T.TriMulAutotuned.__init__

    def _red(v):
        """JSON-reduce one captured value so the two paths compare by VALUE, not by identity.

        A tensor contributes its shape and dtype rather than its bytes: both paths are handed the
        SAME weight dict object, so comparing values would be a tautology, while comparing shapes
        still catches a dropped or renamed key. Anything else unrecognised becomes its type name,
        which is enough to notice a substitution and not enough to make the comparison fragile.
        """
        if isinstance(v, (int, float, str, bool)) or v is None:
            return v
        if isinstance(v, (tuple, list)):
            return [_red(x) for x in v]
        if isinstance(v, torch.dtype):
            return str(v)
        if isinstance(v, torch.Tensor):
            return ("__tensor__", tuple(v.shape), str(v.dtype))
        if isinstance(v, dict):
            return {k: _red(x) for k, x in sorted(v.items())}
        return f"<{type(v).__name__}>"

    # `functools.wraps` is LOAD-BEARING, not cosmetic: `_reject_unknown_engine_kwargs` derives the
    # legal knob set from `inspect.signature(TriMulAutotuned.__init__)`'s keyword-only parameters,
    # and a bare `(self, *args, **kwargs)` wrapper has none -- so an unwrapped patch makes the guard
    # report "Legal engine knobs are []" and refuse every legitimate kwarg. `wraps` sets
    # `__wrapped__`, which `inspect.signature` follows back to the real signature.
    @functools.wraps(orig)
    def wrapper(self, *args, **kwargs):
        """Record this construction into `sink`, then build the engine for real."""
        # positional order is (pe_map, B, N, D, w, dt)
        sink["args"] = {"B": args[1], "N": args[2], "D": args[3], "dt": str(args[5]),
                        "w_keys": sorted(k for k, v in args[4].items() if v is not None)}
        sink["kwargs"] = {k: _red(v) for k, v in sorted(kwargs.items())}
        return orig(self, *args, **kwargs)

    monkeypatch.setattr(T.TriMulAutotuned, "__init__", wrapper)
    return sink


def _engine_resolved(e):
    """Snapshot the state an engine RESOLVED to -- read off the instance, never recomputed.

    Purpose
        The companion to `_capture_engine_ctor`: the constructor arguments say what was ASKED for,
        this says what the engine DID with them. A refactor that passes the same kwargs but reaches
        a different store class or a different tile would pass the first check and fail this one.

    Functionality & semantics
        Field-for-field the same set `tests/distributed/workflows/trimul_engine_oracle.py` freezes
        into `trimul_engine_args.json`, so a discrepancy found here has the same shape as one found
        by the cross-launch oracle diff and can be read the same way. Everything is coerced to a
        plain Python value so the comparison is by VALUE and a class is compared by NAME.

    Args:
        e: A constructed `TriMulAutotuned`. Must be past `__init__` -- every attribute read here is
            set there, so a partially-built engine raises `AttributeError` rather than reporting a
            default.

    Returns:
        A JSON-shaped dict.
    """
    sig = getattr(e, "_front_sig", None)
    return {
        "front_cfg": list(e._front_cfg), "back_cfg": list(e._back_cfg),
        "hybrid_ib": bool(e.hybrid_ib), "has_ib_peers": bool(e._has_ib_peers),
        "route2_ni": bool(e.route2_ni), "composite_k": bool(e.composite_k),
        "has_mask": bool(e.has_mask), "dynamic": bool(e.dynamic), "back_store": e.back_store,
        "use_signal_drain": bool(e._use_signal_drain),
        "drain_obj": type(sig).__name__ if sig is not None else None,
        "front_pad_inner": bool(e.front_pad_inner), "front_pad_eager": bool(e.front_pad_eager),
        "geometry": {"cp": int(e.cp), "cp0": int(e.cp0), "cp1": int(e.cp1),
                     "N_i_loc": int(e.N_i_loc), "N_j_loc": int(e.N_j_loc), "M": int(e.M),
                     "D_loc": int(e.D_loc), "L": int(e._back.L)},
        "store_classes": {
            "front": type(e._front).__name__, "back": type(e._back).__name__,
            "front_ni": type(e._front_ni).__name__ if e._front_ni is not None else None,
            "front_comp": type(e._front_comp).__name__ if e._front_comp is not None else None},
    }




@pytest.mark.parametrize("mesh_spec", _EQ_MESH_SPECS, ids=lambda s: str(s[0][1]).replace(" ", ""))
@pytest.mark.parametrize("direction", ["outgoing", "incoming"])
@pytest.mark.parametrize("masked", [False, True])
def test_the_module_builds_the_same_engine_the_functional_path_does(
    _nvshmem__distributed__workflows__trimul_autotuned, apply_mesh, world_size, device,
    monkeypatch, direction, masked, mesh_spec,
):
    """`TriangularMultiplication` and `trimul_a2a` must reach an IDENTICAL engine, and identical bits.

    Purpose
        `§4.2` of `docs/refactor_dtensor_api.md` moved the mesh unwrap, the sharding validation, the
        PeMap build and the store-variant selection out of `trimul_a2a` (per forward) and into
        `TriangularMultiplication.__init__` (once). That is a RELOCATION -- the same predicates over
        the same inputs, evaluated earlier -- and this is the assertion that it stayed one.

        The output check alone would not be enough. Two configurations can agree numerically and
        differ in which front store, which drain object and which tile they used; the incoming
        fast-path variants exist precisely because a ~2x-slower store computes the same answer. So
        the constructor arguments and the resolved state are compared as well, and the message names
        the field that differs.

    Semantics
        Runs the SAME input through both entry points in one process, on this launch's mesh, and
        compares three things: the recorded `TriMulAutotuned.__init__` arguments, the resolved
        engine state (`_engine_resolved`, the field set the cross-launch oracle freezes), and the
        output tensors BITWISE. `torch.equal` is the right comparison here and not a tolerance --
        the two paths are supposed to run the same compiled kernel on the same bytes, so any
        difference at all is the finding, not a magnitude to be bounded.

        Parametrized over three axes because each selects a different compiled artifact:
        `direction` picks the back-half einsum, `masked` is a compile-time front-store variant that
        keys the engine dict, and `mesh_spec` decides `cp1` -- which IS the discriminator in
        `incoming_store_variant`, so a flat-mesh-only run would never exercise `route2_ni` at all
        and the whole 2-D half of the rule would be untested here.

    Args:
        direction: which contraction op 7 runs; separate compiled artifacts.
        masked: whether a mask DTensor is passed; a compile-time front-store variant.
        mesh_spec: one entry of `_EQ_MESH_SPECS`; skipped rank-uniformly by `apply_mesh` when its
            rank product is not this launch's WORLD_SIZE.
    """
    _requires_sm90()
    from torch.distributed.tensor import DTensor

    from fold_cp_ops.distributed.trimul_weights import trimul_module_from_weights
    from fold_cp_ops.distributed.workflows.trimul_autotuned import (
        TriangularMultiplication, trimul_a2a,
    )
    from tests.distributed.correctness_harness import make_weights

    dist_manager = apply_mesh(mesh_spec)
    pm, mesh, placements = _pe_map(dist_manager)
    cp0 = int(pm.cp_axis_sizes[0])
    cp1 = int(pm.cp_axis_sizes[1]) if len(pm.cp_axis_sizes) > 1 else 1
    if _E2E_N % cp0 or _E2E_N % cp1 or _E2E_D % int(pm.cp):
        rank_invariant_skip(
            f"N={_E2E_N} / D={_E2E_D} do not shard at cp=({cp0},{cp1})",
            because=(
                "the mesh spec is a parametrized value every rank receives in the same order, so "
                "the divisibility verdict is identical on every rank"
            ),
        )

    dt = torch.bfloat16
    weights = make_weights(_E2E_D, out_bias=True, seed=_E2E_SEED, device=device)
    layer = trimul_module_from_weights(weights)
    g = torch.Generator(device="cpu").manual_seed(_E2E_SEED + 7)
    x_local = torch.randn(
        _E2E_B, _E2E_N // cp0, _E2E_N // cp1, _E2E_D, generator=g, dtype=torch.float32
    ).to(device).to(dt)
    x_d = DTensor.from_local(x_local.contiguous(), mesh, placements)
    mask_d = None
    if masked:
        ml = torch.ones(_E2E_B, _E2E_N // cp0, _E2E_N // cp1, device=device, dtype=dt)
        mask_d = DTensor.from_local(ml, mesh, placements)

    sink = _capture_engine_ctor(monkeypatch)

    # --- the FUNCTIONAL path -------------------------------------------------------------
    fcache: dict = {}
    sink.clear()
    try:
        # `eps` is passed EXPLICITLY here because the module resolves it from `layer.norm_in.eps`
        # and forwards it (R3), while a bare `trimul_a2a` leaves the engine on its own default.
        # Both land on 1e-5 for this layer, so omitting it compares equal numerically and UNEQUAL
        # in the kwargs -- which is a difference in the test's setup, not in the paths, and this
        # gate is sharp enough to say so. Keep them matched.
        out_f = trimul_a2a(x_d, dict(weights), dt, direction=direction, mask=mask_d,
                           eps=float(getattr(layer.norm_in, "eps", 1e-5)),
                           distributed_manager=dist_manager, _cache=fcache)
        assert "args" in sink, "trimul_a2a did not construct an engine -- nothing to compare"
        args_f, kw_f = dict(sink["args"]), dict(sink["kwargs"])
        res_f = _engine_resolved(next(iter(fcache.values())))
        out_f = out_f.to_local().clone()
    finally:
        for inst in fcache.values():
            inst.free()
        fcache.clear()

    # --- the MODULE path -----------------------------------------------------------------
    sink.clear()
    tm = TriangularMultiplication(layer, direction, mesh, dist_manager, dtype=dt)
    try:
        out_m = tm(x_d, mask_d)
        assert "args" in sink, "the module did not construct an engine -- nothing to compare"
        args_m, kw_m = dict(sink["args"]), dict(sink["kwargs"])
        res_m = _engine_resolved(next(iter(tm._engines.values())))
        out_m = out_m.to_local().clone()
        assert tm.incoming_variant == (
            "composite_k" if res_m["composite_k"] else
            ("route2_ni" if res_m["route2_ni"] else "plain")
        ), f"incoming_variant={tm.incoming_variant!r} disagrees with the engine it reports on"
        # ...and the engine it reports on is the one the RULE asked for. The assertion above is
        # self-consistency: it holds just as well when the module quietly built something slower
        # than it was asked for, because it compares the witness to the same demoted engine. This
        # one compares against `incoming_store_variant`, which is where the answer is DECIDED, and
        # it is the assertion that separates "fixed" from "still demoting": a cross-node
        # (`hybrid_ib`) build used to fall back to the plain store, and the pair of assertions
        # would both have passed while reporting `"plain"` on a mesh whose rule says otherwise.
        assert tm.incoming_variant == incoming_store_variant(direction, cp1, _E2E_B), (
            f"the rule says {incoming_store_variant(direction, cp1, _E2E_B)!r} for "
            f"direction={direction!r} cp1={cp1} B={_E2E_B}, but the module BUILT "
            f"{tm.incoming_variant!r}. A store the selector did not choose is a silent downgrade "
            f"(~2x on the incoming path) whatever the numbers say; the outputs below will still "
            f"match, because the slow store is correct."
        )
    finally:
        tm.free()

    for name, a, b in (("init positional", args_f, args_m), ("init kwargs", kw_f, kw_m),
                       ("resolved state", res_f, res_m)):
        diff = sorted(k for k in set(a) | set(b) if a.get(k, "<absent>") != b.get(k, "<absent>"))
        assert not diff, (
            f"{name} differs between trimul_a2a and TriangularMultiplication on {diff}: "
            + "; ".join(f"{k}: functional={a.get(k, '<absent>')!r} module={b.get(k, '<absent>')!r}"
                        for k in diff)
        )
    assert torch.equal(out_f, out_m), (
        "the two entry points produced different bits from the same input; they are supposed to "
        "run the same compiled kernel, so this is a configuration difference the field comparisons "
        "above did not cover"
    )



def test_a_second_batch_extent_warns_once_and_builds_a_second_engine(
    _nvshmem__distributed__workflows__trimul_autotuned, apply_mesh, world_size, device,
):
    """A second `(batch, masked)` pair costs a second compile and a second set of symmetric
    buffers. It is SUPPORTED, so it must not raise -- and it used to happen in total silence, so it
    must say so.

    Purpose
        `§3.5.2` established that a compiled instance is bound to its batch extent: the front
        operand's M is `batch * N_i_loc * N_j_loc` and the back receive buffer is shaped with it. A
        caller varying batch per call therefore pays a multi-second compile and a fresh allocation
        every time a new extent appears, with nothing in the log connecting the memory growth to the
        cause. R8 makes the second build announce itself.

    Semantics
        Outgoing, so the plain store runs and the fast-path `batch == 1` limitation is not in play
        -- this test is about the MODULE's engine bookkeeping, not about the incoming variants.

        Three assertions, and the third is the one worth having: the first forward warns NOT AT ALL
        (a warning on the expected build would be noise), the second warns EXACTLY once naming both
        keys, and a THIRD forward at the second extent is silent -- the module holds the engine and
        the warning is once per module, not once per miss. A per-miss warning would itself become
        noise in a loop that legitimately alternates.

    Args:
        world_size: this launch's rank count; the mesh is the flat cp spec.
    """
    _requires_sm90()
    from torch.distributed.tensor import DTensor

    from fold_cp_ops.distributed.trimul_weights import trimul_module_from_weights
    from fold_cp_ops.distributed.workflows.trimul_autotuned import TriangularMultiplication
    from tests.distributed.correctness_harness import make_weights

    dist_manager = apply_mesh((("cp", int(world_size)),))
    pm, mesh, placements = _pe_map(dist_manager)
    cp0 = int(pm.cp_axis_sizes[0])
    cp1 = int(pm.cp_axis_sizes[1]) if len(pm.cp_axis_sizes) > 1 else 1
    if _E2E_N % cp0 or _E2E_N % cp1 or _E2E_D % int(pm.cp):
        rank_invariant_skip(
            f"N={_E2E_N} / D={_E2E_D} do not shard at cp=({cp0},{cp1})",
            because="the mesh shape is fixed by the launcher and identical on every rank",
        )

    dt = torch.bfloat16
    layer = trimul_module_from_weights(
        make_weights(_E2E_D, seed=_E2E_SEED, device=device)
    )

    def _x(b):
        return DTensor.from_local(
            torch.randn(b, _E2E_N // cp0, _E2E_N // cp1, _E2E_D,
                        device=device, dtype=torch.float32).to(dt).contiguous(),
            mesh, placements,
        )

    tm = TriangularMultiplication(layer, "outgoing", mesh, dist_manager, dtype=dt)
    try:
        with warnings.catch_warnings(record=True) as first:
            warnings.simplefilter("always")
            tm(_x(1))
        assert not [w for w in first if issubclass(w.category, RuntimeWarning)], (
            "the FIRST engine build is the expected one and must be silent; warning on it would "
            f"make the second-build warning unreadable. Got: {[str(w.message) for w in first]}"
        )
        with warnings.catch_warnings(record=True) as second:
            warnings.simplefilter("always")
            tm(_x(2))
        hits = [w for w in second if issubclass(w.category, RuntimeWarning)
                and "SECOND compiled engine" in str(w.message)]
        assert len(hits) == 1, (
            f"expected exactly one second-engine RuntimeWarning, got {len(hits)}: "
            f"{[str(w.message) for w in second]}"
        )
        with warnings.catch_warnings(record=True) as third:
            warnings.simplefilter("always")
            tm(_x(2))
        assert not [w for w in third if issubclass(w.category, RuntimeWarning)
                    and "SECOND compiled engine" in str(w.message)], (
            "the warning is once per MODULE, not once per miss -- a loop alternating between two "
            "batch extents would otherwise drown its own output"
        )
        assert sorted(tm._engines) == [(1, False), (2, False)], (
            f"the module should hold one engine per batch extent; got {sorted(tm._engines)}"
        )
    finally:
        tm.free()

# --------------------------------------------------------------------------- #
# R9: the README tutorial -- STATIC half here, EXECUTION in its own launch
# --------------------------------------------------------------------------- #
def test_the_readme_tutorial_is_extractable_and_every_name_in_it_resolves():
    """The README `Use` block parses, and every module and attribute it names EXISTS.

    Purpose
        `§1.5` of `docs/refactor_dtensor_api.md` recorded the defect this guards: the README's
        `Use` block showed `from fold_cp_ops.distributed import FusedTriMulCP` and
        `from fold_cp_ops import trimul_autotuned`, and **neither name has ever existed** -- the
        published example raised `ImportError` on line 1. That is a documentation defect a reader
        hits before anything else, and it needs no GPU to catch.

    Semantics
        Three checks, cheapest first: the block is extractable and unique, it compiles, and every
        `from X import a, b` in it resolves -- the module imports and each name is an attribute of
        it. The last is the one that would have caught `FusedTriMulCP`.

        **This does NOT run the block.** Execution is `run_readme_tutorial.py`, launched under
        torchrun/srun of its own, and that module's docstring records why it cannot live here: the
        block ends with `DistributedManager.cleanup()`, and running it in a child process to
        contain that HANGS -- a second nvshmem runtime cannot come up inside a process tree whose
        parent already holds one. Measured at cp=2 locally and at cp=16 on two nodes over IB, with
        `cudaHostRegister with IoMemory failed with error=800` in the child's captured stdout.

    Raises:
        AssertionError: on a missing/duplicated block, a syntax error, or a name the block imports
            that its module does not define.
    """
    import importlib

    from tests.distributed.workflows.run_readme_tutorial import readme_python_block

    src = readme_python_block()
    tree = ast.parse(src, "README.md#Use")

    missing = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            try:
                mod = importlib.import_module(node.module)
            except Exception as e:  # noqa: BLE001 -- the failure IS the finding
                missing.append(f"{node.module!r} does not import ({type(e).__name__}: {e})")
                continue
            for alias in node.names:
                if not hasattr(mod, alias.name):
                    missing.append(f"{node.module}.{alias.name} does not exist")
        elif isinstance(node, ast.Import):
            for alias in node.names:
                try:
                    importlib.import_module(alias.name)
                except Exception as e:  # noqa: BLE001
                    missing.append(f"{alias.name!r} does not import ({type(e).__name__}: {e})")
    assert not missing, (
        "the README `Use` block names things that do not exist, so a reader copying it fails "
        "before reaching any of this package's behaviour: " + "; ".join(missing)
    )


def test_the_readme_tutorial_declares_the_lifecycle_it_documents():
    """The block must still contain the four lifecycle calls its comments make claims about.

    Purpose
        `§2.1` audited this block and found four wrong claims, **three of them in the four-line
        teardown** -- setup is exercised by every test in the repo, so a wrong claim there fails
        immediately, while teardown is exercised by process exit, where a wrong claim is invisible.
        The corrected comments now say precisely what `free()` and `cleanup()` do and do not do.

        A comment that describes a call the block no longer makes is worse than no comment: it
        reads as a checked claim. This pins the four calls those comments are about, so deleting
        one forces the prose to be revisited.

    Semantics
        Presence only -- ordering and behaviour are the runner's job (`run_readme_tutorial.py`),
        which executes the block under its own launch. This is the GPU-free half.

    Raises:
        AssertionError: naming each lifecycle call the block has stopped making.
    """
    from tests.distributed.workflows.run_readme_tutorial import readme_python_block

    src = readme_python_block()
    required = {
        "DistributedManager.initialize(": "brings up the process group AND sets this rank's device",
        "DistributedManager.init_nvshmem(": "the one-way door the atexit finalize is paired with",
        ".free()": "releases the COMPILED kernel modules, not symmetric memory",
        "DistributedManager.cleanup(": "barrier -> release CUDA libraries -> destroy the group",
    }
    gone = [f"{k!r} ({why})" for k, why in required.items() if k not in src]
    assert not gone, (
        "the README tutorial no longer makes lifecycle calls its own comments describe, so the "
        "prose now documents something the block does not do: " + "; ".join(gone)
    )


# ── the flat signal-kernel compile entry point ─────────────────────────────────────────────────
def test_the_signal_kernel_declares_its_two_compile_time_constants():
    """``_DeviceSignalKernel`` must key on ``cp`` and ``add_op``, because both are folded in.

    ``cp`` becomes the launch grid extent and ``add_op`` the ``SignalOp`` baked into the device
    ``signal_op``. Before this kernel had a parameter pack it had no ``compile_key()`` at all, so a
    shared compile keyed on its TYPE alone would have handed a cp=8 grid to a cp=16 drain -- CTA 8
    reading past a table sized 8, which is an illegal address rather than an exception.

    Host-only: constructing the functor traces nothing and touches no device.
    """
    from fold_cp_ops.distributed.workflows.trimul_autotuned import _DeviceSignalKernel

    assert _DeviceSignalKernel(8, 0).compile_key() == {"cp": 8, "add_op": 0}
    assert _DeviceSignalKernel(8, 0).compile_key() != _DeviceSignalKernel(16, 0).compile_key()
    assert _DeviceSignalKernel(8, 0).compile_key() != _DeviceSignalKernel(8, 5).compile_key()


def test_the_signal_kernel_params_are_immutable_after_construction():
    """The two constants must be frozen, which is what makes the key trustworthy.

    A key is only worth having if the value it recorded is the value the kernel was compiled with.
    ``TemplateParamsMixin`` enforces that by refusing the write; this pins it for THIS functor,
    because the whole point of moving it onto the paradigm was to get that guarantee.
    """
    from fold_cp_ops.distributed.workflows.trimul_autotuned import _DeviceSignalKernel

    k = _DeviceSignalKernel(8, 0)
    with pytest.raises(AttributeError):
        k.cp = 16


def test_the_flat_signal_entry_point_takes_only_scalars():
    """Every argument that decides the emitted signal kernel must be a scalar PARAMETER.

    This is the flat-entry-point property stated as a check rather than as prose: if a compile
    surface takes an already-constructed functor, its key depends on whatever that functor was
    configured with elsewhere, and "elsewhere" is where an unkeyed knob hides. A signature of
    scalars cannot have that problem -- there is nowhere else for a knob to be.

    ``persist`` is exempt because it selects a STORAGE backend, not emitted code: the same kernel
    with and without it is byte-identical, which is exactly why the artifact cache can be optional.
    """
    import inspect

    from fold_cp_ops.distributed.workflows.trimul_autotuned import compile_device_signal_kernel

    sig = inspect.signature(compile_device_signal_kernel)
    assert [p for p in sig.parameters] == ["cp", "add_op", "persist"], (
        f"the flat entry point grew a parameter: {list(sig.parameters)}. A non-scalar one moves the "
        "key back onto an object the caller configured somewhere else"
    )
    for name in ("cp", "add_op"):
        assert sig.parameters[name].annotation is int, (
            f"{name} must be a plain int so it can be part of a hashable key"
        )


def _free_quietly(*engines):
    """Release every non-None engine, swallowing whatever a teardown raises.

    Purpose
        Cleanup for a test that may have failed PART WAY through construction. A half-built engine
        can raise from ``free()`` for a reason that has nothing to do with the assertion under test,
        and letting that propagate replaces the real failure with a teardown traceback -- the
        original error is then gone from the report.

    Functionality & semantics
        Frees in the order given. Callers should pass the most recently built engine FIRST, so a
        shared compile's LAST holder is the one that was created first -- which keeps the finalize
        order the same as it would be without sharing.

    Args:
        *engines: `TriMulAutotuned` instances or None. None entries are skipped, so a caller can
            pass names that were never assigned because construction raised.

    Returns:
        None. Never raises -- that is the entire contract, and it is why the blind ``except`` here
        is deliberate rather than lazy.
    """
    for eng in engines:
        if eng is None:
            continue
        try:
            eng.free()
        except Exception:  # noqa: BLE001 -- see the docstring; a teardown must not mask the failure
            pass


def test_a_second_engine_in_one_process_reuses_the_first_ones_compiles(
    _nvshmem__distributed__workflows__trimul_autotuned, apply_mesh, world_size, device
):
    """Two identically-configured engines in one process must compile ONCE, not twice.

    This is the whole point of the in-process reuse cache, and it is the measurement the module
    docstring's own warning asked for: "building a SECOND compiled engine ... a separate
    multi-second compile". Before `reuse=`, every distributed compile ran `cute.compile`
    unconditionally -- ``persist`` defaults to None, no shipped caller sets one, and ``@jit_cache``
    has no users under ``distributed/``.

    Asserted on the HIT COUNTER, never on wall clock. A timing assertion here would be a perf gate
    wearing a correctness gate's clothes: it would fail on a loaded box and pass on a fast one
    whatever the cache did, and this box is not the one the perf pins were harvested on.

    The second engine must record at least one hit AND no new miss. Both halves are needed --
    hits>0 alone is satisfied by a cache that hits on some kernels and recompiles the rest, which is
    exactly the shape a partially-complete key produces.

    Engine COUNT is deliberately two. `_symmetric_free` is a no-op (the MemPool recycles by
    refcount), so every live engine is charged against the symmetric heap, and this module has
    already measured a session abort at the seventh.
    """
    _requires_sm90()
    import fold_cp_ops.distributed.gemm_bitcode_compile as gbc
    from tests.distributed.correctness_harness import make_weights

    dist_manager = apply_mesh((("cp", int(world_size)),))
    pm, mesh, placements = _pe_map(dist_manager)
    cp0 = int(pm.cp_axis_sizes[0])
    cp1 = int(pm.cp_axis_sizes[1]) if len(pm.cp_axis_sizes) > 1 else 1
    if _E2E_N % cp0 or _E2E_N % cp1 or _E2E_D % int(pm.cp):
        rank_invariant_skip(
            f"N={_E2E_N} / D={_E2E_D} do not shard at cp=({cp0},{cp1})",
            because=(
                "the mesh shape is fixed by the launcher and identical on every rank, so the "
                "divisibility verdict is the same everywhere"
            ),
        )

    dt = torch.bfloat16
    weights = make_weights(_E2E_D, seed=_E2E_SEED, device=device)

    def build():
        return TriMulAutotuned(
            pm, 1, _E2E_N, _E2E_D, weights, dt,
            dynamic=True, device_mesh=mesh, placements=placements,
        )

    first = second = None
    try:
        first = build()
        first.forward(torch.zeros(1, _E2E_N // cp0, _E2E_N // cp1, _E2E_D, device=device, dtype=dt),
                      "outgoing")
        before = dict(gbc.REUSE_STATS)
        second = build()
        second.forward(torch.zeros(1, _E2E_N // cp0, _E2E_N // cp1, _E2E_D, device=device, dtype=dt),
                       "outgoing")
        after = dict(gbc.REUSE_STATS)
    finally:
        _free_quietly(second, first)

    hits = after["hits"] - before["hits"]
    misses = after["misses"] - before["misses"]
    assert hits > 0, (
        f"the second engine took {hits} reuse hits and {misses} misses -- it recompiled everything. "
        "Either `reuse=True` is not reaching the call sites, or the key carries something that "
        "differs between two identically-configured engines"
    )
    assert misses == 0, (
        f"the second engine took {hits} hits but ALSO {misses} misses. A partial hit rate is the "
        "signature of a key that separates two identical configurations on one component -- find "
        "which by diffing `gbc._reuse_key(...)` for the two engines, not by loosening this bound"
    )


def test_freeing_one_engine_does_not_invalidate_a_second_holding_the_same_compile(
    _nvshmem__distributed__workflows__trimul_autotuned, apply_mesh, world_size, device
):
    """The reuse hazard, on the real object: engine A's ``free()`` must not break engine B.

    With sharing on, two engines hold ONE `CompiledGemmBitcode`. If A's ``free()`` finalized it, B's
    next launch would run against a registration nvshmem has let go of and a CUDA library the DSL is
    free to unload -- a FAULT several frames from the ``free()`` that caused it, not an exception.

    The proof is a forward pass AFTER the first engine is freed. Asserting the holder count instead
    would test the bookkeeping, which the host-side cell in
    ``tests/distributed/test_gemm_bitcode_compile.py`` already does; only an actual launch proves
    the registration is still live.
    """
    _requires_sm90()
    from tests.distributed.correctness_harness import make_weights

    dist_manager = apply_mesh((("cp", int(world_size)),))
    pm, mesh, placements = _pe_map(dist_manager)
    cp0 = int(pm.cp_axis_sizes[0])
    cp1 = int(pm.cp_axis_sizes[1]) if len(pm.cp_axis_sizes) > 1 else 1
    if _E2E_N % cp0 or _E2E_N % cp1 or _E2E_D % int(pm.cp):
        rank_invariant_skip(
            f"N={_E2E_N} / D={_E2E_D} do not shard at cp=({cp0},{cp1})",
            because=(
                "the mesh shape is fixed by the launcher and identical on every rank, so the "
                "divisibility verdict is the same everywhere"
            ),
        )

    dt = torch.bfloat16
    weights = make_weights(_E2E_D, seed=_E2E_SEED, device=device)
    x = torch.zeros(1, _E2E_N // cp0, _E2E_N // cp1, _E2E_D, device=device, dtype=dt)

    def build():
        return TriMulAutotuned(
            pm, 1, _E2E_N, _E2E_D, weights, dt,
            dynamic=True, device_mesh=mesh, placements=placements,
        )

    first = second = None
    try:
        first, second = build(), build()
        first.forward(x, "outgoing")
        second.forward(x, "outgoing")
        first.free()
        first = None
        # The launch that would fault if the shared registration had been finalized underneath it.
        out = second.forward(x, "outgoing")
        assert out.shape == x.shape, f"second engine returned {tuple(out.shape)} after the free"
    finally:
        _free_quietly(second, first)
