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

"""Perf gate for the A2A-fused BACK einsum store (``gemm_sm90_a2a``) -- the O(N^3) half.

Why this file exists at all
    Before it, the back had **no perf evidence of any kind**. Measured on 2026-08-19: the
    distributed perf directory held ONE gate (the front), the shipped pin file's targets were
    ``{'front': 21, 'front_ibdrain': 12}`` -- zero back cells in all 33 -- and the perf-neutrality
    sweep that reported 56/56 within 1.25% swept ``front`` and ``front_a2a`` only, at ``cp=(2,8)``
    only. The back is proven CORRECT (``test_gemm_a2a_epi.py::test_back_gemm_native_store_correct``)
    and proven BYTE-IDENTICAL to ``main``, and was entirely unproven on SPEED -- on the half where a
    regression costs most, because the einsum is O(N^3) while the front projection is O(N^2).

K == N_token, and that is a HARD RULE rather than a shape choice
    The back einsum is ``out[b,i,j,d] = sum_k a[b,i,k,d]*b[b,j,k,d]`` with ``i == j == k ==
    N_token``. A benchmark that decouples K -- e.g. the ``K=256`` the CORRECTNESS module uses -- times
    an O(N^2) thin-K matmul instead, which is comm-bound where the real thing is compute-bound, and
    silently inverts every overlap / crossover / "GEMM hides the A2A" conclusion drawn from it. So
    the operands here come from ``back_a2a_store_bench._build_inputs_2d_kn``, which asserts
    ``K == N`` in its own body, and NOT from the correctness module's builder. The gate additionally
    reports TFLOP/s per cell, computed from ``2*L*M*N*K``: a decoupled K would show up immediately as
    an impossible number, so the rule is self-evidencing in the run's own output rather than only in
    this docstring.

The grid is ``(mesh, N_token, D, variant)``, DECLARED IN FULL before its pins exist
    That is the whole design brief. Every cell is collected from day one and a cell with no pin
    SKIPS with a stated reason (``PINS`` membership, exactly as the front gate does) rather than
    being absent -- a declared-but-unpinned cell must be visible AS unpinned, because a cell that is
    simply missing from the file looks identical to a cell nobody thought of.

Where each axis comes from, and why it takes TWO matrices
    ``N_token`` and ``drain_variant`` are drawn from :data:`A2A_KERNEL`, the matrix
    ``tests/distributed/test_gemm_sm90_a2a.py`` declares for this kernel -- which the audit also
    requires this gate to hold by identity, so the timed and the configure-tested token extents
    cannot drift.

    ``mesh`` and ``D`` are drawn from :data:`EPI`, and they are NOT on ``A2A_KERNEL`` for a reason
    that module states with its evidence: ``configure_a2a_gemm_native`` has no feature-width
    parameter at all, the kernel module contains zero ``D``/``Dloc`` code tokens, and every test in
    that module is a host-side configure verdict that allocates nothing. Adding a ``D`` axis THERE
    would be either decoration (declared, swept by nothing -- which ``coverage_problems`` rule 1
    fails) or a test asserting that a verdict is independent of an argument the function never
    receives, which cannot fail. ``EPI`` declares ``mesh`` and ``D`` for the same subject and
    already sweeps them in a LAUNCHING test (``test_back_gemm_native_store_correct``), which is
    where the feature width is genuinely an input. Two matrices, each consulted for the axis it
    actually owns, is the honest spelling; one matrix with a vacuous axis is not.

Two targets, mutually exclusive BY FABRIC -- the same arrangement the front gate uses
    The module docstring of the front gate records why one process may time only one target: run
    after ``front`` in the same process the back median moved 15.00 -> 25.96 ms and the two trees
    elected DIFFERENT autotune configs, i.e. contamination does not merely add noise, it changes
    which config wins. The two cells here cannot contaminate each other because they cannot both
    run: the COUPLED store is refused on a mesh that spans NVLink domains
    (``topology.skip_if_coupled_cross_node``), and the CLUSTER drain is the cross-node path and
    skips when the job has no IB peer -- on an all-NVLink job it would collapse to the coupled store
    and report a different measurement under this file's name. They share a module because the
    matrix audit derives a gate's owning correctness module from its FILENAME; a second file would
    search for a ``test_<other>.py`` that does not exist and error every one of its cells at setup.

State
    The pin file ships EMPTY, so every cell skips until harvested. The harvest protocol -- one
    isolated launch per cell -- is written into the JSON's ``measured_on`` as it is filled.
"""

from __future__ import annotations

import os

import pytest
import torch

from benchmark.distributed import bench_utils as BU

from fold_cp_ops.distributed.distributed_manager import DistributedManager
from fold_cp_ops.testing.collective_guard import gated_skip, rank_invariant_skip
from fold_cp_ops.testing.capacity_guard import capacity_gate
from fold_cp_ops.testing.kernel_matrix import matrix_exempt

# The two matrices, each for the axes it owns. `A2A_KERNEL` is ALSO what the audit checks this gate
# holds by identity (`audit_test_module` check 4), so this import is load-bearing twice over.
from tests.distributed.test_gemm_sm90_a2a import A2A_KERNEL
from tests.distributed.test_gemm_a2a_epi import (
    EPI,
    _compile_back_gemm_native,
    _mesh_ranks,
    _misaligned_2d,
    _pe_map,
    _skip_coupled_cross_node,
)
from tests.distributed.topology import job_has_ib_peers

# The mesh->label function the FRONT gate defines, imported rather than copied so the two gates key a
# mesh identically BY CONSTRUCTION. Two independent spellings that happened to agree would share a
# pin the moment they stopped agreeing, and a shared pin is silently wrong rather than loudly so.
from tests.distributed.perf.test_benchmark_perf_dual_gated_gemm_a2a import _mesh_label

from tests.perf.calibration import assert_cell
from tests.perf.pins import load as load_pins

#: Required by this directory's ``pytest_collectstart``. A collective timed with ``mode="device"``
#: picks its repetition count PER RANK, so the ranks run different iteration counts and desync --
#: producing numbers rather than an error, which is why the conftest refuses it rather than warns.
TIMING = {"mode": "event", "reduce": "max"}

PINS = load_pins(__file__)

#: Rounds/warmup for one cell. Fixed, not adaptive: a fixed count is what keeps every rank in
#: lockstep through the in-kernel collective inside the timed region.
_ROUNDS = 15
_WARMUP = 3

#: The CTA tile every cell here is timed at. Pinned to one value ON PURPOSE, and the consequence is
#: worth stating: because ``tile`` is not parametrized, no :class:`Unsupported` region of
#: ``A2A_KERNEL`` is evaluable over this gate's grid (both regions read ``tile`` or ``shape_mode``),
#: so ``parametrize`` cannot exclude a must-raise cell for us. At ``tile_m=128`` neither declared
#: region applies -- the first needs ``tile_m == 256``, the second ``shape_mode == "static"`` -- so
#: there is nothing to exclude. Widening this to the matrix's ``tile`` axis therefore MUST move
#: ``tile`` into the same ``parametrize`` call as ``drain_variant``, or the grid will silently
#: contain ``(256, *) x cluster_multislot``, which the kernel is required to REFUSE.
_TILE = (128, 128)

#: The four feature widths used by the target TriMul workflow. ``EPI``'s ``D`` pool also carries
#: 4/8/96, which exist there so a small back store can be checked for CORRECTNESS at a width a test
#: can name as a constant; timing them would measure a launch-bound cell of a kernel whose subject
#: is throughput.
_WORKFLOW_D = (128, 256, 384, 512)

#: Why the ``D`` pool is narrowed to :data:`_WORKFLOW_D`. Hoisted to a constant because both cells
#: give the same reason and a copy-pasted ``because=`` is one edit away from saying two things.
_D_BECAUSE = (
    "the timed subject is the O(N^3) einsum's throughput at the target workflow's feature widths "
    "128/256/384/512. EPI's 4/8/96 are correctness widths -- "
    "4 and 8 are the L=1 and L>1 back-store minimums and 96 is LCM-chosen so Dloc stays integral at "
    "cp 16/24/32 -- and at those widths a cell is launch-bound, so its median reports the dispatch "
    "floor rather than the kernel. They stay covered where they were chosen for: the LAUNCHING "
    "correctness test on the same matrix"
)


@pytest.fixture(scope="session", autouse=True)
def _nvshmem_up(dist_manager):
    """Bring nvshmem up once for this module, session-scoped and collective.

    The correctness modules declare an equivalent fixture, but MODULE-PRIVATE: a ``@pytest.fixture``
    in a test module is not visible from another module, only conftest fixtures are. Depending on
    one via ``usefixtures`` collects fine and then errors at SETUP with "fixture not found", which is
    how the front gate first ran -- 40 errors, zero measured.

    Args:
        dist_manager: The session's DistributedManager. Required rather than decorative --
            ``init_nvshmem`` needs the process group already up, and requesting the fixture is what
            orders the two.

    Yields:
        None.
    """
    DistributedManager.init_nvshmem()
    yield


def _back_flops(N: int, D: int, cp: int, B: int = 1) -> float:
    """FLOPs of one back-einsum call, ``2 * L * M * N * K`` with ``L = B*(D/cp)`` and ``M = K = N``.

    Purpose
        Turn the K==N rule into something the RUN's own output can be checked against. A cell built
        with a decoupled K would report a TFLOP/s far outside anything an H100 can do (or far below,
        depending which way the error went), so the number is a tripwire and not decoration.

    Args:
        N: The full square token extent. Must be the same value used to build BOTH operands --
            passing the per-peer extent here understates the count by ``cp`` and makes a slow cell
            look fast.
        D: The full feature width. Must divide ``cp``; a non-integral ``D/cp`` is refused by the
            caller before this is reached, so no check is repeated here.
        cp: The context-parallel rank count, i.e. ``world_size``.
        B: The TriMul batch. 1 everywhere in this gate, spelled as an argument so a future batched
            cell cannot forget the factor.

    Returns:
        The FLOP count as a float. Never raises.
    """
    L = B * (D // cp)
    return 2.0 * L * N * N * N


def _skip_unless_pinned(key) -> None:
    """Skip a cell that has no pin yet, unless this launch IS the harvest.

    Purpose
        A gate authored before its harvest must compare against nothing rather than against invented
        numbers, and the skip must be VISIBLE -- a declared-but-unpinned cell that simply vanished
        would be indistinguishable from a cell nobody declared.

    Why the decision is made here and not inside ``assert_cell``
        That helper skips with a bare ``pytest.skip``, which is correct for ``tests/perf/`` and
        FORBIDDEN under ``tests/distributed/**``: a skip on a predicate that could differ per rank is
        a deadlock, and the collective guard cannot know this particular predicate is safe. Measured
        on the front gate -- its first clean run failed all 8 runnable cells with "collective
        symmetry violation ... bare pytest.skip" AFTER measuring them, so the numbers were real and
        the verdict was still refused.

    Why the harvest qualifier is the whole point
        During a harvest (``CPO_PERF_MEASURE=1``) the pin is absent BY DEFINITION -- absent is what
        the run exists to fix -- so an unconditional skip makes the harvest structurally impossible.
        Measured on the front gate: the first harvest launch reported ``rc=0`` and ``harvested=0`` on
        every cell, 15 launches that would have written an empty pin file and looked clean.

    Args:
        key: The cell's pin key, a tuple of JSON-safe scalars. Must be the SAME tuple passed to
            ``assert_cell`` -- a key that differs by so much as an int-vs-str would skip a cell that
            is in fact pinned, or measure one that is not.

    Returns:
        None when the cell is pinned or this is a harvest run. Otherwise never returns.

    Raises:
        Skipped: through ``rank_invariant_skip``. Rank-invariant because ``PINS`` is read from a file
            shipped in the repo and is byte-identical on every rank.
    """
    if os.environ.get("CPO_PERF_MEASURE", "0") == "1" or tuple(key) in PINS:
        return
    rank_invariant_skip(
        f"no pin for {key}; harvest with CPO_PERF_MEASURE=1 and ONE ISOLATED LAUNCH PER CELL -- see "
        f"this module's docstring for why two targets cannot share a process",
        because="PINS is loaded from a file shipped in the repo, identical on every rank",
    )


def _skip_geometry(
    *, D: int, cp: int, N_token: int, cp0: int, mesh, arbitrary_n: bool = False
) -> None:
    """Skip the cells this launch's world size cannot form a legal back-store geometry for.

    Purpose
        Collect the FULL declared grid on every launch and let the illegal cells say why they did not
        run, instead of narrowing the grid per world size -- a narrowed grid reports 100% of what it
        chose to attempt, which is the number that hides a shape nobody covers anywhere.

    Semantics
        Four predicates, every one job-uniform, hence ``rank_invariant_skip`` rather than a reduce:

        * ``D % cp`` -- the per-rank feature shard ``Dloc = D/cp`` must be integral.
        * ``(N_token // cp0) % 128`` -- the per-peer token block must be a multiple of the CTA
          M-tile. The coupled builder does not pass ``arbitrary_n``, so an off-grid token extent
          (``A2A_KERNEL``'s 2072 / 2080, which exist to reach the store's inexact-division paths)
          straddles a CTA tile and is refused rather than mis-tiled.
        * ``_misaligned_2d`` -- a FACTORED mesh whose per-axis extent is not 16-B aligned. This is
          ``EPI``'s own declared unsupported region, applied by hand because the region is evaluated
          only when ONE ``parametrize`` call spans every axis it reads, and here ``N_token`` and
          ``mesh`` come from different matrices. Omitting it would cross this gate into cells the
          matrix declares UNSUPPORTED and time a configure that must raise.
        * ``Dloc == 0`` is impossible once ``D % cp == 0`` holds for a positive ``D``, so it is not
          separately tested.

    Args:
        D: The full feature width from the pool.
        cp: This launch's rank count.
        N_token: The full square token extent from the pool.
        cp0: The token-i shard extent -- ``cp`` on the 1-D coupled path.
        mesh: The mesh spec, passed only to ``_misaligned_2d``.

    Returns:
        None when the cell is legal here.

    Raises:
        Skipped: through ``rank_invariant_skip``, naming which predicate refused it.
    """
    uniform = (
        "D, N_token and the mesh are declared matrix values and world_size is fixed by the "
        "launcher, so every rank evaluates this identically without needing to reduce"
    )
    if D % cp != 0:
        rank_invariant_skip(
            f"D={D} does not divide cp={cp}; the per-rank feature shard Dloc is not integral",
            because=uniform,
        )
    if not arbitrary_n and (N_token // cp0) % 128 != 0:
        rank_invariant_skip(
            f"N_loc = N_token/cp0 = {N_token // cp0} is not a multiple of the CTA M-tile 128 at "
            f"N_token={N_token}, cp0={cp0}; the coupled builder does not lift it with arbitrary_n",
            because=uniform,
        )
    if _misaligned_2d(N_token, mesh):
        rank_invariant_skip(
            f"mesh {mesh!r} splits N_token={N_token} into a per-axis extent that is not 16-B "
            f"aligned; EPI declares this combination UNSUPPORTED and the store refuses it",
            because=uniform,
        )


def _measure(run_fn, dist_manager) -> float:
    """Time one call through ``bench_utils``, returning the slowest PE's median in ms.

    Purpose
        The ONE timing spelling in this module, so the two cells cannot drift into two
        configurations. Hand-rolling a ``perf_counter`` loop around a kernel launch is forbidden
        repo-wide: a raw launch is async, so the host returns before the device finishes and the
        derived "bandwidth" can exceed the physical link.

    Semantics
        ``mode="event"`` runs a FIXED iteration count -- which is what keeps every rank in lockstep
        through the in-kernel put -- and ``reduce="max"`` reports the slowest PE, which is what a
        collective's latency actually is. Both are spelled from :data:`TIMING` rather than splatted,
        so the declared dict stays the single source of truth while each argument still type-checks
        at its own parameter.

    Args:
        run_fn: A zero-argument callable that issues exactly one back-einsum call. It must not
            allocate: an allocation inside the timed window is measured as kernel time.
        dist_manager: Passed to ``bench_utils`` as ``dist=``, which supplies rank / world_size /
            device for the ``all_reduce(MAX)`` consensus.

    Returns:
        The median over ``_ROUNDS`` event-timed windows, reduced to the slowest PE. Bit-identical on
        every rank, which is what lets each rank compare against the same pin independently.
    """
    return BU.benchmark_single(
        run_fn,
        rounds=_ROUNDS,
        warmup=_WARMUP,
        iters=1,
        dist=dist_manager,
        mode=TIMING["mode"],
        reduce=TIMING["reduce"],
    ).median_ms


def _cp_axes(spec, cp: int) -> tuple[int, int]:
    """The ``(cp0, cp1)`` token shard a mesh spec asks the STORE for.

    Purpose
        Make the mesh axis reach the store's own shard arithmetic, not only the DeviceMesh the test
        runs on. A FLAT spec is the 1-D shard -- token-i split ``cp`` ways, token-j whole -- and a
        FACTORED one splits BOTH token axes, which is the case whose tile->peer UNRAVEL exercises
        both cp axes; a flat spec collapses that to a single division and would pass while testing
        half the routing.

    Why it matters beyond coverage
        ``_misaligned_2d`` -- EPI's declared unsupported region, which this gate applies by hand --
        is defined on the FACTORED reading of the spec. Forcing a 1-D shard while applying that
        guard makes the guard and the store disagree. It also decides comparability: the harness's
        ``cluster`` cells derive ``cp0``/``cp1`` from the same spec, so a 1-D store timed under a
        factored mesh's name would be compared against ``main``'s number for a different shard.

    Args:
        spec: The mesh spec, ``(("cp", size), ...)``. ``size`` is an int (flat) or a tuple of ints
            (factored). Only the FIRST entry is read, which is exact for this pool: every declared
            value is a single pure-cp axis. A spec with a second axis would silently have it
            ignored, so it raises instead.
        cp: This launch's rank count. The derived product must equal it; a mismatch means the spec
            and the launch disagree, which would shard the tokens a different number of ways than
            there are peers and store into the wrong ranks rather than failing.

    Returns:
        ``(cp0, cp1)`` with ``cp0 * cp1 == cp``.

    Raises:
        ValueError: If the spec has more than one axis, or the product does not equal ``cp``.
    """
    if len(spec) != 1:
        raise ValueError(f"mesh spec {spec!r} has {len(spec)} axes; this gate reads a pure-cp mesh")
    size = spec[0][1]
    if isinstance(size, int):
        cp0, cp1 = size, 1
    else:
        cp0 = size[0]
        cp1 = 1
        for e in size[1:]:
            cp1 *= e
    if cp0 * cp1 != cp:
        raise ValueError(f"mesh spec {spec!r} needs {cp0 * cp1} ranks but this launch has {cp}")
    return cp0, cp1


def _assert_store_ran(recv, *, what: str) -> None:
    """Assert the timed store actually WROTE every recv cell, using the builder's own sentinel.

    Purpose
        Close the "did-not-run" hole. A perf gate reports a median whether or not the kernel did the
        work, and a store that silently wrote nothing is FASTER -- so the failure mode makes the
        number look better, which is the worst direction for a gate to be wrong in. Nothing else in
        this module witnesses the store: the timed callable returns a duration, and the harness's own
        correctness gate reported ``"correct": null`` on the cells measured beside these.

    Semantics
        ``_build_inputs_2d_kn`` fills the recv with ``-99.0`` before the run, so a cell still holding
        that value along its whole last axis was never reached. This is a COVERAGE stamp, not a
        numerical comparison -- there is no reference and no tolerance -- which is why it is exact
        and why it does not belong to the sanctioned element-wise comparison set.

    Why it is outside the timed region and costs nothing
        It runs after ``assert_cell`` has returned, i.e. after 15 samples x 18 windows, so it reads a
        buffer the measurement has already finished with. One full read of the recv is a few
        milliseconds against a cell that has just spent tens of seconds on the device.

    Args:
        recv: The symmetric recv, shape ``(cp, Dloc, B, N_loc, N)``, as returned by the builder that
            filled it with the sentinel. Passing a recv from any other builder makes this vacuous --
            an unfilled buffer holds arbitrary bytes and will not equal the sentinel.
        what: A short cell description for the failure message.

    Returns:
        None.

    Raises:
        AssertionError: Naming how many of the ``cp*Dloc*B*N_loc`` cells the store never reached.
    """
    untouched = int((recv == -99.0).all(dim=4).sum().item())
    assert untouched == 0, (
        f"{what}: {untouched} of {recv.shape[0] * recv.shape[1] * recv.shape[2] * recv.shape[3]} "
        f"recv cells still hold the -99.0 sentinel, so the store this gate just TIMED never reached "
        f"them -- the median is of a kernel that did less work than the cell claims"
    )


@EPI.parametrize("mesh", "D", only={"D": _WORKFLOW_D}, because=_D_BECAUSE)
@A2A_KERNEL.parametrize("N_token")
@matrix_exempt(
    "a perf gate's subject is the kernel's measured SPEED at a declared cell, not a numerical "
    "result. It parametrizes from the SAME KernelMatrix objects the correctness modules declare, "
    "which is what the convention asks of a gate; there is no separate pool here to audit"
)
def test_back_coupled_median_is_within_its_pin(
    apply_mesh, mesh, D, N_token, dist_manager, device, world_size, mesh_or_skip
):
    """The COUPLED 5-D back store's slowest-PE median stays within its pinned value at one cell.

    Semantics
        Builds the real back einsum -- ``K == N_token``, both operands O(N^2), the design-E 5-D
        symmetric recv ``(cp, Dloc, B, N_loc, N)`` -- then times one fused call through
        ``bench_utils`` with ``mode="event", reduce="max"``. Exactly one target is timed in this
        process; see the module docstring for the measurement that makes that a requirement.

    Input requirements
        ``mesh`` and ``D`` are declared :data:`EPI` pool values; ``N_token`` a declared
        :data:`A2A_KERNEL` one. The cell skips -- coherently, never through a bare ``pytest.skip``,
        which under ``tests/distributed/**`` is a deadlock -- when this launch's ``WORLD_SIZE``
        cannot supply the mesh, when the geometry is illegal at this ``cp`` (see
        :func:`_skip_geometry`), when the job spans NVLink domains (the coupled store is refused
        cross-node, and the cluster-drain cell below is that fabric's path), when the K==N operands
        do not fit, or when the cell has no pin yet.
    """
    mesh_or_skip(mesh, _mesh_ranks(mesh), kernel="gemm_sm90_a2a")
    apply_mesh(mesh)

    cp = world_size
    # The coupled store shards token-i only: cp0 == cp, cp1 == 1. `_pe_map` flattens every Shard
    # axis row-major, so a FACTORED mesh spec still presents a flat cp here -- the spec changes the
    # DeviceMesh this runs on, not the store's own shard arithmetic. Same relation the launching
    # correctness test uses, which is what makes the two comparable.
    cp0, cp1 = cp, 1
    # 1-D DELIBERATELY, and NOT `_cp_axes(mesh, cp)` as the cluster cell below uses. This mirrors
    # the LAUNCHING correctness test exactly -- `test_back_gemm_native_store_correct` takes
    # `cp = world_size`, `N_loc = N/cp` at every mesh spec, so the coupled store's 2-D shard has
    # never been exercised anywhere in this tree. Timing a configuration no correctness test covers
    # would be pinning a path nothing proves correct.
    _skip_geometry(D=D, cp=cp, N_token=N_token, cp0=cp0, mesh=mesh)

    pm = _pe_map(dist_manager)
    # The coupled peer store writes over NVLink only. On a job with an IB peer it is REFUSED, and the
    # cluster-drain cell below is that fabric's path -- which is also why the two cells in this
    # module can never contaminate each other's process.
    _skip_coupled_cross_node(pm, "back gemm-native coupled store")

    Dloc, B = D // cp, 1
    key = ("back_coupled", _mesh_label(mesh), D, N_token)
    _skip_unless_pinned(key)

    # Imported at CALL time, not module scope: `back_a2a_store_bench` pulls in cutlass and the
    # bitcode compile path, and this module must stay importable for collection on a box with no
    # SM90 silicon -- the whole grid has to be VISIBLE even where none of it can run.
    from benchmark.distributed.back_a2a_store_bench import _build_inputs_2d_kn

    A = Bt = recv = compiled = None
    try:
        # THE K==N BUILDER. `_build_inputs_2d_kn` asserts `K == N` in its own body; the
        # correctness module's builder uses K=256 and is labelled correctness-only, so importing
        # THAT here is the documented benchmark-invalidating defect and not a shortcut.
        A, Bt, recv, _cute = capacity_gate(
            f"back_coupled operands N={N_token} D={D} cp={cp}",
            lambda: _build_inputs_2d_kn(device, int(dist_manager.rank), N_token, cp0, cp1, Dloc, B),
        )

        pe_table = tuple(int(x) for x in pm.cp_pe_table.tolist())
        cfg = dict(
            cp=cp,
            my_cp_rank=int(pm.my_cp_rank),
            B=B,
            N_loc=N_token // cp0,
            pe_table=pe_table,
        )
        compiled, run_args = _compile_back_gemm_native(A, Bt, recv, _TILE, gemm_native_cfg=cfg)

        def _run() -> None:
            compiled(*run_args)

        assert_cell(
            lambda: _measure(_run, dist_manager),
            key,
            PINS,
            __file__,
            flops=_back_flops(N_token, D, cp, B),
        )
        _assert_store_ran(recv, what=f"back_coupled N={N_token} D={D} cp={cp}")
    finally:
        # The recv rides the SYMMETRIC heap, so it must be released before the next allocation -- the
        # same per-run free the correctness path does. `main` frees both; the bring-back dropped it
        # once already (fixed in 6461fe3), so it is spelled out rather than assumed.
        if compiled is not None:
            compiled.free()
        # DROPPING THE REFERENCE MEANS ALL OF THEM. `del A, Bt, recv` releases nothing while other
        # names in this frame still alias the same storage: `_cute` holds the marked views
        # (`D_logical` is `as_strided` OVER recv; `A3`/`B3` are permutes of A/Bt) and `run_args`
        # holds the `from_dlpack` CuTe tensors built over recv, which `_run` closes over. The
        # identical gap in `harness/targets/cluster_drain.py::_build` measured 27.39 GiB RETAINED
        # per rejected config and forced a second symmetric segment -- a COLLECTIVE
        # `nvshmem_malloc` -- at the next one.
        #
        # ASSIGNMENT, not `del`: only A/Bt/recv/compiled are pre-initialised above, so `del _cute`
        # would raise when the builder itself was what failed. Assignment BINDS a local and cannot
        # raise on an unbound name, which is what makes it safe in a `finally` that runs on every
        # path.
        A = Bt = recv = compiled = None
        _cute = run_args = _run = None
        # Return the freed blocks to the DRIVER. nvshmem grows its symmetric heap with `cuMemCreate`,
        # which needs PHYSICAL pages; torch's caching allocator holding freed-but-cached blocks is
        # what turned the next cell's growth into `mem_heap.cpp:2135 ... CUDA_ERROR_OUT_OF_MEMORY`
        # on an idle 80 GiB card.
        torch.cuda.empty_cache()


@EPI.parametrize("mesh", "D", only={"D": _WORKFLOW_D}, because=_D_BECAUSE)
@A2A_KERNEL.parametrize("N_token", "drain_variant")
@matrix_exempt(
    "a perf gate's subject is the kernel's measured SPEED at a declared cell, not a numerical "
    "result. It parametrizes from the SAME KernelMatrix objects the correctness modules declare, "
    "which is what the convention asks of a gate; there is no separate pool here to audit"
)
def test_back_cluster_drain_median_is_within_its_pin(
    apply_mesh, mesh, D, N_token, drain_variant, dist_manager, device, world_size, mesh_or_skip
):
    """The CROSS-NODE cluster-drain back store's slowest-PE median stays within its pinned value.

    Semantics
        Builds the same K==N back einsum, then configures the decoupled ``cluster_drain`` stack --
        the production cross-node store -- through the harness's own ``_build_cluster``, so the
        timed path is the one the sweep and the bench already drive rather than a second spelling of
        it. ``drain_variant`` selects the staging shape and the config kwarg: ``cluster_drain`` is
        the even-shard single buffer, ``cluster_multislot`` the rotating ``ring_depth``-slot one.

    Why the timed callable ends in a QUIET
        The cluster drain issues NON-blocking IB puts, so the kernel returning does not mean the
        store landed. Timing the launch alone would report the GEMM and charge the communication to
        the next window, which is precisely the "GEMM hides the A2A" conclusion this gate exists to
        measure rather than assume.

    Input requirements
        ``mesh`` and ``D`` are declared :data:`EPI` pool values; ``N_token`` and ``drain_variant``
        declared :data:`A2A_KERNEL` ones. The cell skips -- coherently, never through a bare
        ``pytest.skip`` -- when this launch cannot supply the mesh, when the geometry is illegal at
        this ``cp``, when the job has NO IB peer (the drain would collapse to the coupled store and
        measure something else under this file's name), when the K==N operands do not fit, or when
        the cell has no pin yet.
    """
    mesh_or_skip(mesh, _mesh_ranks(mesh), kernel="gemm_sm90_a2a_cluster_drain")
    apply_mesh(mesh)

    cp = world_size
    # From the SPEC, not a fixed 1-D split -- see `_cp_axes` for why the guard and the store must
    # agree on which reading of the mesh is in force, and why comparability to `main` depends on it.
    cp0, cp1 = _cp_axes(mesh, cp)
    # arbitrary_n=True because `_build_cluster` configures it, and that LIFTS the CTA M-tile floor.
    # Applying the coupled path's predicate here would skip the OFF-GRID token extents (2072, 2080)
    # that the cluster drain genuinely supports -- and those two values are the only ones in the
    # pool whose per-peer division is inexact, i.e. the only ones that can catch a floor-division
    # where the kernel means ceil. Skipping them would be narrowing away the very coverage they
    # were declared for.
    _skip_geometry(D=D, cp=cp, N_token=N_token, cp0=cp0, mesh=mesh, arbitrary_n=True)

    pm = _pe_map(dist_manager)
    pe_table = tuple(int(x) for x in pm.cp_pe_table.tolist())
    # The fabric gate, and it is the SAME probe the correctness path uses, so this gate and that one
    # agree by construction rather than by two copies of the same arithmetic.
    if not job_has_ib_peers(pe_table):
        rank_invariant_skip(
            f"the cluster drain is the cross-node store; this job is all-NVLink "
            f"(pe_table={pe_table}), where it collapses to the coupled store the cell above times. "
            f"Harvest and enforce these cells on a >=2-node allocation.",
            because="the peer table is derived from the job's own rank->node map, identical on "
            "every rank",
        )

    Dloc, B = D // cp, 1
    key = (f"back_{drain_variant}", _mesh_label(mesh), D, N_token)
    _skip_unless_pinned(key)

    # Call-time import, as in the cell above: this module must stay importable for collection on a
    # box that cannot run any of it.
    import nvshmem.core.rma as nvshmem_rma
    from cutlass.cute.runtime import from_dlpack

    from fold_cp_ops._internal.arch import get_max_active_clusters
    from benchmark.distributed.back_a2a_store_bench import (
        _build_cluster,
        _build_inputs_2d_kn,
        _cluster_forced,
        _symmetric_empty,
    )

    A = Bt = recv = stage_buf = compiled = None
    try:
        A, Bt, recv, (cA, cB, cD) = capacity_gate(
            f"back_{drain_variant} operands N={N_token} D={D} cp={cp}",
            lambda: _build_inputs_2d_kn(device, int(dist_manager.rank), N_token, cp0, cp1, Dloc, B),
        )

        # KEEP a reference: the cute view below aliases this tensor's storage, and letting it fall
        # out of scope frees memory the kernel still reads.
        pe_dev_t = pm.cp_pe_table.to(torch.int32).contiguous()
        pe_dev_c = from_dlpack(pe_dev_t, assumed_align=4)

        cluster_n, ring_depth = 1, 4
        N_j_loc = N_token // cp1
        nt_j_pp = (N_j_loc + _TILE[1] - 1) // _TILE[1]
        n_clusters = get_max_active_clusters(cluster_n)
        # `_cluster_forced` owns the staging SHAPE and the config kwarg for a variant; deriving them
        # here instead would be a second copy of a mapping the harness already validates.
        force = None if drain_variant == "cluster_drain" else "multislot"
        stage_shape, cfg_kw, _path = _cluster_forced(
            force, cluster_n, nt_j_pp, cp1, N_j_loc, n_clusters, ring_depth
        )
        stage_buf = capacity_gate(
            f"back_{drain_variant} staging (symmetric) {stage_shape}",
            lambda: _symmetric_empty(stage_shape, torch.bfloat16, device),
        )
        stage_buf.zero_()
        stage_view = from_dlpack(stage_buf, assumed_align=16).mark_layout_dynamic()

        compiled, run_args, _use_3wg = _build_cluster(
            A,
            Bt,
            cA,
            cB,
            cD,
            recv,
            stage_view,
            pe_dev_c,
            cluster_n=cluster_n,
            cfg_kw=cfg_kw,
            cp=cp,
            cp0=cp0,
            cp1=cp1,
            my_cp_rank=int(pm.my_cp_rank),
            B=B,
            N_loc=N_token // cp0,
            pe_table=pe_table,
            N=N_token,
            rd=ring_depth,
        )

        def _run() -> None:
            compiled(*run_args)
            # Stream-ordered, so it lies INSIDE the CUDA-event window and adds no host cost: the
            # window closes when the puts have landed, which is when the store is actually done.
            nvshmem_rma.quiet(stream=torch.cuda.current_stream())

        assert_cell(
            lambda: _measure(_run, dist_manager),
            key,
            PINS,
            __file__,
            flops=_back_flops(N_token, D, cp, B),
        )
        _assert_store_ran(recv, what=f"back_{drain_variant} N={N_token} D={D} cp={cp}")
    finally:
        # The staging buffer is the IB put SOURCE and lives on the SYMMETRIC heap; the mempool
        # recycles on the last reference, so dropping it here is the release.
        if compiled is not None:
            compiled.free()
        # Same aliasing rule as the coupled cell above: `cA`/`cB`/`cD` are the marked views (cD
        # over recv) and `stage_view`/`ring_view`/`run_args` wrap the staging buffer, so dropping
        # only the four base names retains every block.
        A = Bt = recv = stage_buf = compiled = None
        cA = cB = cD = stage_view = ring_view = run_args = _run = None
        torch.cuda.empty_cache()
