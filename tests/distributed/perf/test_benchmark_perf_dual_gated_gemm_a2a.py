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

"""Perf gate for the A2A-fused front (``dual_gated_gemm_a2a``), ONE TARGET PER MODULE.

Why one target per module, which is the only unusual thing here
    A sibling target timed earlier IN THE SAME PROCESS changes this one's number, and by enough to
    invent a regression. Measured on venue A, mesh ``cp=(2,8)``,
    ``N_token=4096``, ``D=512``, the back (``cluster``) target:

    ======================================  ===========  ==========
    how it was run                          ours         main
    ======================================  ===========  ==========
    after ``front``, same process           25.96 ms     21.69 ms
    alone, its own process                  15.00 ms     15.01 ms
    ======================================  ===========  ==========

    Three repeats alone agreed to +-0.02 ms ACROSS BOTH TREES and elected the same autotune config
    every time; run after ``front`` the two trees elected different configs and landed 20% apart in
    opposite directions. So contamination does not merely add noise -- it changes which config wins,
    which is what made it look like an ours-vs-main divergence for most of a day. A gate that timed
    front and cluster in one pytest process would bake that artifact into its pins.

    Hence: this module gates the FRONT only. The back gets its own module, and the harvest runs one
    launch per (mesh, D, target). Do not "consolidate" them.

What it gates
    ``winner.time_ms``-equivalent: this rank's median over ``rounds`` event-timed windows, then
    ``all_reduce(MAX)`` across ranks -- the SLOWEST PE, which is the real distributed cost and is
    bit-identical on every rank. ``bench_utils`` does the reduce; ``TIMING`` below declares the mode
    the directory's conftest requires, and ``mode="device"`` is refused there because its per-rank
    adaptive repetition count DESYNCS a collective without failing.

The ``N_token`` axis, and why it is not free
    Both cells parametrize ``N_token`` and NARROW it to :data:`_PERF_NTOK`. Before that they carried
    a bare ``B, N = 1, 128``, so all 33 shipped pins sat at one shape whose medians are 52-190 us
    against a ~22 us host dispatch floor -- a gate that cannot see a regression which appears only at
    production ``M``. The small rungs the correctness pool declares (16/24/32/40) are EXCLUDED here
    and the exclusion is stated rather than silent: at ``rows_per_peer = N_token**2 / cp`` they land
    at 16..800 rows against a CTA ``tile_M`` of 128, so they either self-skip on the tile gate or
    measure the launcher. The large end is bounded by a runtime OOM skip, never a memory estimate.

State
    Fully harvested at ``N_token=128``: 33 of 33 reachable ``(target, mesh, D)`` cells, venue A
    venue A. The pin key now carries ``N_token`` as a fourth element,
    so those cells are keyed ``(..., 128)``. The larger rungs are pinned only where a harvest has
    reached them; every unpinned cell skips with the key it wanted, which is what tells you what to
    harvest. The harvest protocol is written into the JSON's ``measured_on``.
"""

from __future__ import annotations

import contextlib
import os

import pytest
import torch

from benchmark.distributed import bench_utils as BU

from fold_cp_ops._internal import bench_timing
from fold_cp_ops.distributed.distributed_manager import DistributedManager
from fold_cp_ops.testing.collective_guard import gated_skip, rank_invariant_skip
from fold_cp_ops.testing.capacity_guard import (
    HEAP_EXHAUSTED,
    capacity_gate,
    is_capacity_error,
)
from fold_cp_ops.testing.kernel_matrix import matrix_exempt

from fold_cp_ops.distributed.dual_gated_gemm_a2a import DualGatedGemmDistSm90

from tests.distributed.test_dual_gated_gemm_a2a import (
    FRONT_A2A,
    _build_front,
    _build_front_decoupled,
    _pe_map,
    _this_rank_device,
)
from tests.distributed.topology import job_has_ib_peers
from tests.perf.calibration import assert_cell
from tests.perf.pins import load as load_pins

#: Required by ``tests/distributed/perf/conftest.py``'s ``pytest_collectstart``. Not decoration: a
#: collective timed with ``mode="device"`` picks its repetition count per rank, so the ranks run
#: different iteration counts and desync -- producing numbers rather than an error.
TIMING = {"mode": "event", "reduce": "max"}

PINS = load_pins(__file__)

#: Rounds/warmup for one cell. Fixed, not adaptive: a fixed count is what keeps every rank in
#: lockstep through the collective inside the timed region.
_ROUNDS = 15
_WARMUP = 3

#: The ``N_token`` values this gate measures, a NARROWING of the declared ``FRONT_A2A`` pool.
#:
#: 128 is where every shipped pin sits and is kept so the existing harvest stays reachable. The
#: other seven are the ladder the 2026-08-19 perf-neutrality run measured at cp=(2,8), which is the
#: only ladder for which medians already exist on disk -- reusing them is the whole reason this axis
#: could be added without a fresh multi-allocation harvest.
#:
#: The correctness pool's small rungs (16, 24, 32, 40) are deliberately ABSENT, and this is the
#: "documented exclusion" rather than a silent omission. At ``rows_per_peer = N_token**2 / cp`` they
#: give 16..800 rows against ``tile_M = 128``: cp>=2 makes ``(M // cp) % tile_M != 0`` for every one
#: of them, so the shape gate would skip them anyway, and where it did not the cell would be almost
#: entirely the ~22 us host dispatch floor -- a pin on the launcher, not on the kernel. 1088, 4032
#: and 4128 are absent for a weaker reason: no measurement exists for them yet. They are a harvest
#: away, not a design decision, and adding them here without one would only manufacture skips.
_PERF_NTOK = (128, 2048, 3008, 4096, 5952, 8192, 10048, 12288)

#: Why the pool is narrowed, quoted into every ``parametrize(only=...)`` below so the two cells
#: cannot drift apart in their justification.
_PERF_NTOK_BECAUSE = (
    "a perf cell costs an isolated launch, so the axis is narrowed to the values a harvest has "
    "reached or can reach: 128 (where all 33 shipped pins sit) plus the cp=(2,8) neutrality ladder "
    "whose medians are already on disk. The small rungs 16/24/32/40 are excluded because "
    "rows_per_peer = N_token**2/cp puts them under the CTA tile_M of 128 and inside the ~22 us host "
    "dispatch floor -- they would pin the launcher. See _PERF_NTOK for the full accounting"
)


def _mesh_label(spec) -> str:
    """Flatten a mesh spec to one hashable scalar, e.g. ``(('cp', (2, 4)),) -> "cp2x4"``.

    Purpose
        A pin key must survive a JSON round-trip AND stay hashable. A mesh spec is a tuple of
        tuples, which is hashable in Python and comes back from JSON as a LIST -- so
        ``PinFile.__init__``'s ``key in self._cells`` raises ``TypeError: unhashable type: 'list'``
        the moment such a pin is loaded. Measured: the first written pin file refused to load at all.
        Every shipped pin file therefore keys on flat scalars, and this is how a mesh becomes one.

    Args:
        spec: The mesh spec, a tuple of ``(axis, extent)`` pairs where ``extent`` is an int or a
            tuple of ints. Anything else raises rather than producing a label that collides with
            another mesh's -- two meshes sharing a label would silently share a pin.

    Returns:
        A label built from the axis names and extents, stable across runs and JSON-safe.

    Raises:
        TypeError: If an extent is neither an int nor a tuple of ints.
    """
    parts = []
    for axis, extent in spec:
        if isinstance(extent, int):
            parts.append(f"{axis}{extent}")
        elif isinstance(extent, tuple):
            for e in extent:
                if not isinstance(e, int):
                    raise TypeError(f"mesh extent {extent!r} holds a non-int {e!r}")
            parts.append(f"{axis}" + "x".join(str(e) for e in extent))
        else:
            raise TypeError(f"mesh extent {extent!r} is neither int nor tuple")
    return "_".join(parts)


def _mesh_ranks(spec) -> int:
    """Rank count a mesh spec needs, e.g. ``(('cp', (2, 8)),) -> 16``.

    Args:
        spec: A mesh spec as declared on :data:`FRONT_A2A`'s ``mesh`` axis -- a tuple of
            ``(axis_name, extent_or_tuple)`` pairs. An extent may be an int or a tuple of ints;
            anything else is a malformed spec and raises rather than silently counting as 1, because
            a wrong count mis-reports mesh coverage instead of failing.

    Returns:
        The product of every axis extent.

    Raises:
        TypeError: If an extent is neither an int nor a tuple of ints.
    """
    total = 1
    for _axis, extent in spec:
        if isinstance(extent, int):
            total *= extent
        elif isinstance(extent, tuple):
            for e in extent:
                if not isinstance(e, int):
                    raise TypeError(f"mesh extent {extent!r} holds a non-int {e!r}")
                total *= e
        else:
            raise TypeError(f"mesh extent {extent!r} is neither int nor tuple")
    return total


def _front_geometry_or_skip(N_token, D, cp):
    """Derive ``(Dloc, tile_M, tile_N, rows_per_peer)`` for one cell, or skip it coherently.

    Purpose
        Both cells need the same four numbers and the same three shape gates, and before this
        existed they carried two copies that had already drifted -- one derived the tile before
        allocating and one after, and only one of them gated the token block against ``tile_M``.
        Deriving once is what keeps "which shapes does this gate measure" a single answer.

    Semantics
        Pure arithmetic plus the production tile picker; allocates nothing and issues no collective
        of its own, so it is safe to call before the operands exist. Every gate here is
        RANK-INVARIANT by construction -- ``N_token``, ``D`` and ``world_size`` are job-uniform, so
        every rank walks the identical branch and either all skip or none do. That is why
        ``rank_invariant_skip`` is correct here and ``gated_skip`` (which costs a collective) is not;
        the OOM gate is the opposite case and uses the opposite primitive.

    Input requirements
        N_token: A declared ``FRONT_A2A`` pool value, narrowed to :data:`_PERF_NTOK`. Must be
            positive; ``B*N_token**2`` is the full token block and ``//cp`` this rank's share, so a
            value whose square does not divide ``cp`` skips rather than silently truncating.
        D: A declared feature width. Must already be known to divide ``cp`` -- the caller checks
            that first, because the message it wants to print names the feature shard, not the tile.
        cp: The rank count, i.e. ``world_size``. Must be >= 1 and identical on every rank; a
            per-rank value here would make the skip decision divergent and deadlock the group.

    Returns:
        ``(Dloc, tile_M, tile_N, rows_per_peer)``. Never returns on a gated shape -- it raises
        pytest's ``Skipped`` on every rank simultaneously.

    Raises:
        Skipped: Via ``rank_invariant_skip``, when the token block does not divide the mesh, when
            this rank's share is not a whole number of CTA M-tiles, or when the picked ``tile_N``
            falls under the SM90 WGMMA CTA-tile floor.
    """
    Dloc = D // cp
    # ASK THE PICKER for the tile; do not hardcode one. A frozen (128, 128) looks harmless and is
    # not: the front-A2A requires `Dloc % postact_tile_N == 0` so a CTA N-tile lies inside ONE
    # D-slice (the per-CTA single-peer select depends on it), and at Dloc=32 a hardcoded tile_N=128
    # gives postact tile_N=64, which the kernel refuses at its front door. Measured on this gate's
    # first run: every runnable cell failed with
    # "front-A2A requires Dloc (=32) % postact tile_N (=64) == 0".
    tile_M, tile_N = DualGatedGemmDistSm90.best_front_tile(Dloc)
    # The SM90 WGMMA CTA-tile floor the correctness module gates too: the picker's answer can still
    # fall under the atom's own minimum when the mesh shards D far enough (Dloc=2 -> tile_N=4). A
    # hardware floor, so skipped rather than failed.
    if not ((tile_N % 16 == 0 and tile_N <= 256) or (tile_N % 32 == 0 and tile_N <= 512)):
        rank_invariant_skip(
            f"Dloc={Dloc} at D={D}/cp={cp} yields tile_N={tile_N}, under the SM90 WGMMA CTA-tile "
            f"floor (needs %16 and <=256, or %32 and <=512)",
            because="D and world_size are job-uniform, so every rank derives the same tile",
        )
    B = 1
    M = B * N_token * N_token
    # The gate the correctness module states at test_dual_gated_gemm_a2a.py:1524 and this gate did
    # not: rows_per_peer must be a WHOLE number of CTA M-tiles, because a partial last M-tile is the
    # clamp path's subject and not this one's. It bites the moment N_token becomes an axis -- at
    # N_token=16 and cp=16 the share is ONE row against tile_M=128 -- and without it such a cell
    # reaches the kernel's own front door instead of this gate's.
    if M % cp != 0 or (M // cp) % tile_M != 0:
        rank_invariant_skip(
            f"N_token={N_token} gives M={M}; M//cp={M // cp if M % cp == 0 else 'n/a'} is not a "
            f"whole number of CTA tile_M={tile_M} rows at cp={cp}",
            because="N_token and D are declared matrix values and world_size is fixed by the "
            "launch, so every rank derives the same M, tile and remainder",
        )
    return Dloc, tile_M, tile_N, M // cp


#: Substrings that identify a SYMMETRIC-heap exhaustion, which does NOT arrive as an
#: ``OutOfMemoryError``.
#:
#: Measured, not guessed. Running the ``front`` cell at ``N_token=12288, D=256, cp=2`` -- where the
#: recv is ``(256, 150994944)`` = 77 GB -- the failure was::
#:
#:     RuntimeError: nvshmem_malloc failed
#:     .../src/host/mem/mem_heap.cpp:2056: cuMemCreate failed
#:     .../src/host/mem/mem_heap.cpp:2135: error status: 7 (NVSHMEMX_ERROR_INTERNAL)
#:
#: The recv lives on the symmetric heap, so its capacity limit is nvshmem's rather than the caching
#: allocator's, and catching only ``torch.cuda.OutOfMemoryError`` leaves the large end of the ladder
#: FAILING where CLAUDE.md requires it to SKIP. Matched on the MESSAGE and not on ``RuntimeError``
#: itself, deliberately: a bare ``except RuntimeError`` here would swallow every real build failure
#: the gate exists to surface, and a capacity gate that hides defects is worse than no gate.
# The capacity guard now lives in `fold_cp_ops.testing.capacity_guard`, imported here as the SAME
# objects rather than re-declared. It was duplicated into two perf modules and the copies had already
# diverged into different functions of the same name -- one taking an already-caught exception, so it
# could only be reached from an `except`, where the non-raising ranks never enter the reduction.
_HEAP_EXHAUSTED = HEAP_EXHAUSTED
_is_capacity_error = is_capacity_error
_oom_gate = capacity_gate

_DIAG_PREDICATE_LOGGED = False


def _log_diag_predicate_once(dist_manager) -> None:
    """Print, once per process, whether the per-rank timing diagnostic is ON for THIS rank.

    Purpose
        Make "did ``CPO_BENCH_RANK_DIAG`` reach the ranks?" answerable **independently** of whether
        anything downstream surfaces an argmax. Without it, a run that produces no argmax has two
        indistinguishable explanations -- the env never arrived, or it arrived and the value was
        discarded -- and a re-run cannot separate them either.

    Why it prints in BOTH states, which is the whole design
        An emit that fired only when the diagnostic is ON would reproduce the exact ambiguity it
        exists to remove: absence would again mean either "off" or "never ran". So it prints ``ON``
        or ``OFF`` and the LINE's presence proves the check ran.

    Why ``print`` and not logging
        Same mechanism as ``calibration.assert_cell``'s ``median=... rel_std=...`` line, which is
        known to reach the srun cell logs of this gate. Matching the emit that is observed to arrive
        is a checkable argument; guessing at pytest's capture flags is not.

    Args:
        dist_manager: The gate's manager fixture, used only for ``.rank``. Any object with a
            ``rank`` attribute works; a missing attribute falls back to ``?`` rather than raising,
            since a diagnostic that can abort a perf cell is worse than one that is vague.

    Returns:
        None. Guarded by a module-level flag so this is one line per PROCESS, not per cell -- 16
        ranks x every cell would bury the numbers the gate exists to print.
    """
    global _DIAG_PREDICATE_LOGGED
    if _DIAG_PREDICATE_LOGGED:
        return
    _DIAG_PREDICATE_LOGGED = True
    state = "ON" if bench_timing.rank_diag_enabled() else "OFF"
    raw = os.environ.get(bench_timing.RANK_DIAG_ENV)
    print(
        f"    [rank-diag] rank={getattr(dist_manager, 'rank', '?')} "
        f"{bench_timing.RANK_DIAG_ENV}={raw!r} -> diagnostic {state}",
        flush=True,
    )


#: Env var selecting the arm of the missing-barrier A/B. Unset/``0`` -> ARM A, the SHIPPED path
#: (``dist=<DistributedManager>``), whose ``da.barrier()`` is a silent no-op. Set -> ARM B, which
#: passes ``dist=None`` so ``resolve_dist`` takes its ``torch.distributed`` branch and the
#: per-window barrier is LIVE. Read at call time so one launch's env decides one launch's arm.
_BARRIER_AB_ENV = "CPO_BENCH_LIVE_BARRIER"


def _timer_dist(dist_manager):
    """Return the ``dist=`` argument for ``benchmark_single``, per the barrier A/B arm.

    Purpose
        Flip ``benchmark_single``'s per-window barrier between DEAD and LIVE **without editing the
        library**, so the A/B compares two paths that already exist rather than a path against a
        patch. The measured defect: ``_DistAdapter.barrier`` (``bench_timing.py:164-170``) calls the
        manager's own ``barrier()`` if it has one, else ``torch.distributed`` if ``_use_torch_dist``
        -- and ``resolve_dist``'s manager branch (``:273``) leaves ``_use_torch_dist`` False while
        ``DistributedManager`` has no ``barrier``. Both arms of the ``if`` are False, so the method
        returns None. Proven with two positive controls: ``dist=None`` -> 1 call,
        ``dist=<duck with .barrier>`` -> 1 call, ``dist=<DistributedManager>`` -> 0 calls.

    Semantics
        The ONLY behavioural delta between the arms is the barrier. ``all_reduce_max`` and
        ``all_gather_max_diag`` test ``torch.distributed.is_initialized()`` directly rather than
        ``_use_torch_dist``, so the ``reduce="max"`` consensus is identical on both. ``device`` /
        ``rank`` / ``world_size`` resolve to the same values (``_default_device()`` reads
        ``torch.cuda.current_device()``, which the ``dist_manager`` fixture has already set to this
        rank's local device) -- ``_barrier_witness`` prints all three so a divergence shows in the
        run's own output instead of being assumed away.

    Input requirements
        dist_manager: The gate's manager fixture. Returned unchanged in ARM A. Not inspected in
            ARM B, where ``None`` makes ``resolve_dist`` re-derive rank/world_size from the live
            group -- which requires that group to be UP, i.e. this must be called from inside a
            test that has already requested ``dist_manager``.

    Returns:
        ``dist_manager`` (ARM A) or ``None`` (ARM B). Never raises; an unset env is ARM A, so the
        shipped path is what a launch that knows nothing about this experiment gets.
    """
    if os.environ.get(_BARRIER_AB_ENV, "").strip().lower() not in ("", "0", "false", "no"):
        return None
    return dist_manager


@contextlib.contextmanager
def _barrier_witness(dist_manager):
    """Count ``torch.distributed.barrier`` calls across a measurement and print the total on rank 0.

    Purpose
        Assert the arm's identity by CONTENT at run time. A log that merely carries the env var says
        which arm was REQUESTED; this says which arm RAN. The expected counts are exact and derived,
        not guessed: ``benchmark_single``'s event path issues ``warmup`` + ``2 * rounds`` barriers
        per call (``bench_timing.py:529/543/546``), so at ``_WARMUP=3`` / ``_ROUNDS=15`` that is 33
        per sample and ``33 * N_SAMPLES_PIN_TIME = 495`` per harvested cell. **ARM A must print 0
        and ARM B 495**; anything else is a finding, which is why the raw number is printed rather
        than a boolean.

    Semantics
        Monkeypatches the module attribute ``torch.distributed.barrier`` for the duration of the
        block and restores it in a ``finally``, so an exception inside the measurement cannot leave
        the spy installed. ``bench_timing`` reaches the function through that attribute at call
        time, so the patch intercepts it; a ``from torch.distributed import barrier`` would not have
        been interceptable this way.

    Why this does not perturb the measurement
        The barrier is issued BETWEEN windows, never inside one -- ``time_callable`` records its
        CUDA events itself, after ``da.barrier()`` has returned. So the wrapper's cost (one dict
        increment per call, at most 495 of them) lies wholly outside every timed region. It is also
        installed only around the measurement, so collectives run by fixtures are not counted.

    Input requirements
        dist_manager: Used only for ``.rank``, to keep the emit to one line per launch. A missing
            ``rank`` falls back to ``?`` rather than raising -- a witness that can abort a perf cell
            is worse than one that is vague.

    Yields:
        None. Emits two stdout lines on rank 0 (the arm before, the count after); other ranks are
        silent, because 16 identical copies bury the numbers the gate exists to print.
    """
    arm = "B" if _timer_dist(dist_manager) is None else "A"
    rank = getattr(dist_manager, "rank", "?")
    calls = {"n": 0}
    real = torch.distributed.barrier

    def _spy(*a, **kw):
        calls["n"] += 1
        return real(*a, **kw)

    if rank == 0:
        print(
            f"    [ab-arm] arm={arm} {_BARRIER_AB_ENV}="
            f"{os.environ.get(_BARRIER_AB_ENV)!r} expect td_barrier_calls="
            f"{'495' if arm == 'B' else '0'}"
            + ("   <<< ARM B: its PINHARVEST lines are NOT pinnable" if arm == "B" else ""),
            flush=True,
        )
    torch.distributed.barrier = _spy
    try:
        yield
    finally:
        torch.distributed.barrier = real
        if rank == 0:
            da = BU.resolve_dist(_timer_dist(dist_manager))
            print(
                f"    [ab-witness] arm={arm} td_barrier_calls={calls['n']} "
                f"device={da.device} rank={da.rank} world_size={da.world_size}",
                flush=True,
            )


@pytest.fixture(scope="session", autouse=True)
def _nvshmem_up(dist_manager):
    """Bring nvshmem up once for this module, session-scoped and collective.

    The correctness module declares an equivalent fixture, but it is MODULE-PRIVATE -- a
    `@pytest.fixture` in a test module is not visible from another module, only conftest fixtures
    are. Depending on it via `usefixtures` collects fine and then errors at SETUP with
    "fixture not found", which is how this file first ran: 40 errors, zero measured.

    Args:
        dist_manager: The session's DistributedManager. Required -- `init_nvshmem` needs the process
            group already up, and requesting the fixture is what orders the two.
    """
    DistributedManager.init_nvshmem()
    yield


@FRONT_A2A.parametrize("N_token", only={"N_token": _PERF_NTOK}, because=_PERF_NTOK_BECAUSE)
@FRONT_A2A.parametrize("mesh", "D")
@matrix_exempt(
    "a perf gate's subject is the kernel's measured SPEED at a declared cell, not a numerical "
    "result. It parametrizes from the SAME KernelMatrix the correctness module declares, which is "
    "what the convention asks of a gate; there is no separate pool here to audit"
)
def test_front_a2a_median_is_within_its_pin(
    apply_mesh, mesh, D, N_token, dist_manager, device, world_size, mesh_or_skip
):
    """The A2A-fused front's slowest-PE median stays within its pinned value at one declared cell.

    Semantics
        Builds the front once, then times ``run_fn`` through ``bench_utils.benchmark_single`` with
        ``mode="event", reduce="max"`` -- so the reported number is this rank's median over the
        timed windows, reduced to the slowest PE. Exactly one target is timed in this process; see
        the module docstring for the measurement that makes that a requirement rather than a
        preference.

    Input requirements
        ``mesh``, ``D`` and ``N_token`` come from :data:`FRONT_A2A`, so they are declared pool
        values; ``N_token`` is narrowed to :data:`_PERF_NTOK`. The cell is skipped -- coherently,
        through ``mesh_or_skip`` / ``rank_invariant_skip`` / ``gated_skip``, never a bare
        ``pytest.skip`` -- when this launch's ``WORLD_SIZE`` cannot supply the mesh, when ``D`` does
        not divide by the rank count (a non-integral per-rank feature shard), when the token block
        does not tile (``(B*N_token**2 // cp) % tile_M != 0``), and when the operands do not fit,
        which is decided by CATCHING ``OutOfMemoryError`` -- never by a memory estimate.
    """
    _log_diag_predicate_once(dist_manager)
    mesh_or_skip(mesh, _mesh_ranks(mesh), kernel="dual_gated_gemm_a2a")
    apply_mesh(mesh)

    cp = world_size
    if D % cp != 0:
        # rank_invariant_skip, NOT pytest.skip: under tests/distributed/** a skip on a predicate
        # that could differ per rank is a DEADLOCK, and the collective guard refuses the bare form
        # by directory. D and world_size are both job-uniform, so every rank decides identically.
        rank_invariant_skip(
            f"D={D} does not divide cp={cp}; the per-rank feature shard is not integral",
            because="D comes from the declared matrix and world_size is fixed by the launch, so "
            "both are identical on every rank",
        )
    Dloc, tile_M, tile_N, rows_per_peer = _front_geometry_or_skip(N_token, D, cp)
    K = 256
    pm = _pe_map(dist_manager)

    torch.manual_seed(1234 + int(dist_manager.rank))

    def _alloc():
        x = torch.randn(rows_per_peer, K, device=device, dtype=torch.bfloat16) / (K**0.5)
        Wg2 = torch.randn(2 * D, K, device=device, dtype=torch.bfloat16) / (K**0.5)
        Wp2 = torch.randn(2 * D, K, device=device, dtype=torch.bfloat16) / (K**0.5)
        with torch.cuda.use_mem_pool(DistributedManager.symmetric_mempool(_this_rank_device())):
            recv = torch.empty((2 * Dloc, cp * rows_per_peer), dtype=torch.bfloat16, device=device)
        return x, Wg2, Wp2, recv

    x, Wg2, Wp2, recv = _oom_gate(f"operands + recv at N_token={N_token}, D={D}", _alloc)

    a2a_cfg = dict(
        cp=cp,
        my_cp_rank=int(pm.my_cp_rank),
        rows_per_peer=rows_per_peer,
        pe_table=tuple(int(v) for v in pm.cp_pe_table.tolist()),
    )
    # The build allocates too -- `_build_front` materializes out_postact, which is (2D, M) and at the
    # top of the ladder is the single largest tensor in the cell -- so it goes through the same OOM
    # gate. Gating only the operands would move the failure one line down and out of the skip.
    compiled, run_fn, _ = _oom_gate(
        f"front build at N_token={N_token}, D={D}",
        lambda: _build_front(x, Wg2, Wp2, (tile_M, tile_N), recv_t=recv, a2a_cfg=a2a_cfg),
    )
    try:

        def _measure() -> float:
            return BU.benchmark_single(
                run_fn,
                rounds=_ROUNDS,
                warmup=_WARMUP,
                iters=1,
                # `_timer_dist`, not `dist_manager` -- default is byte-identical (it RETURNS
                # dist_manager unless CPO_BENCH_LIVE_BARRIER is set); see its docstring.
                dist=_timer_dist(dist_manager),
                # Spelled from TIMING rather than `**TIMING` so the declared dict stays the single
                # source of truth while each argument still type-checks at its own parameter.
                mode=TIMING["mode"],
                reduce=TIMING["reduce"],
            ).median_ms

        # The no-pin case is decided HERE, not inside assert_cell. That helper skips with a bare
        # `pytest.skip`, which is correct for tests/perf/ and forbidden under tests/distributed/**:
        # the collective guard refuses it because a skip on a predicate that could differ per rank
        # is a deadlock, and the guard cannot know this particular predicate is safe. Measured -- the
        # first clean run of this gate failed all 8 runnable cells with "collective symmetry
        # violation ... bare pytest.skip" AFTER measuring them, so the numbers were real and the
        # verdict was still refused.
        #
        # Rank-invariant because the pin file is read from disk at import and is byte-identical on
        # every rank, so every rank reaches the same membership answer without reducing.
        #
        # Gated on NOT-harvesting, and that qualifier is the whole point: during a harvest
        # (``CPO_PERF_MEASURE=1``) the pin is absent BY DEFINITION -- absent is what the run exists
        # to fix -- so an unconditional skip here makes the harvest structurally impossible. Measured
        # the hard way: the first harvest launch reported ``rc=0`` and ``harvested=0`` on every cell,
        # 15 launches that would have written an empty pin file and looked like a clean run.
        key = ("front", _mesh_label(mesh), D, N_token)
        if os.environ.get("CPO_PERF_MEASURE", "0") != "1" and tuple(key) not in PINS:
            rank_invariant_skip(
                f"no pin for {key}; harvest with CPO_PERF_MEASURE=1 and one launch per "
                f"(mesh, D, N_token, target) -- see this module's docstring for why the targets "
                f"cannot share a process",
                because="PINS is loaded from a file shipped in the repo, identical on every rank",
            )
        # `_barrier_witness` proves WHICH arm ran from the log's own bytes (ARM A -> 0 calls,
        # ARM B -> 495). A no-op wrapper on the default path: it counts a function nothing calls.
        with _barrier_witness(dist_manager):
            assert_cell(_measure, key, PINS, __file__)
    finally:
        # The ring lives on the SYMMETRIC heap, so it must be released before the next allocation --
        # the same per-run free the correctness path does, for the same reason.
        compiled.free()


# ─────────────────────────────────────────────────────────────────────────────
# The CROSS-NODE store. Same target (front), different STORE, and the two are mutually exclusive by
# FABRIC: the coupled store above is NVLink-only and self-skips on a cross-node job, while ib_drain
# below requires >=1 IB peer and skips on a single-node one. So they never time each other in one
# process, which is the property that forced one-target-per-module in the first place -- and they
# live together because the matrix audit derives a gate's owning correctness module from its
# FILENAME. A separate `..._a2a_ibdrain.py` searched for `test_dual_gated_gemm_a2a_ibdrain.py`,
# which does not exist, and errored all 40 of its cells at setup.
# ─────────────────────────────────────────────────────────────────────────────
_RING_DEPTH = 2
_CWG = 1


@FRONT_A2A.parametrize("N_token", only={"N_token": _PERF_NTOK}, because=_PERF_NTOK_BECAUSE)
@FRONT_A2A.parametrize("mesh", "D")
@matrix_exempt(
    "a perf gate's subject is the kernel's measured SPEED at a declared cell, not a numerical "
    "result. It parametrizes from the SAME KernelMatrix the correctness module declares, which is "
    "what the convention asks of a gate; there is no separate pool here to audit"
)
def test_front_ibdrain_median_is_within_its_pin(
    apply_mesh, mesh, D, N_token, dist_manager, device, world_size, mesh_or_skip
):
    """The cross-node ib_drain front's slowest-PE median stays within its pinned value.

    Semantics
        Builds the front through ``configure_a2a(ib_drain=True)`` -- which auto-builds ``is_p2p``
        and forces the decoupled ring -- then times ``run_fn`` with ``mode="event", reduce="max"``,
        so the number is this rank's median over the timed windows reduced to the SLOWEST PE.
        Exactly one target is timed per process; see the module docstring.

    The shape this builds, and the one it used to build
        ``x`` is ``(rows_per_peer, K)`` -- this rank's OWN token block, so ``GEMM M ==
        rows_per_peer``. That is the relation the coupled cell above holds, the one the correctness
        module holds at ``test_dual_gated_gemm_a2a.py:1806`` (``rows_per_peer = M``), and the one
        the harness target that produced this gate's pinned medians holds
        (``harness/targets/front_a2a.py:78``). It did NOT hold here: this cell built ``x`` with
        ``M = B*N_token**2`` rows while setting ``rows_per_peer = M // cp``, so at cp=16 the GEMM
        was 16x wider than the A2A plan it fed and 15/16 of its output was computed and dropped.
        The line came from ``_ib_drain_body`` (which has ``rows_per_peer = M``) while
        ``rows_per_peer`` came from the coupled sibling -- one line from each source, and neither
        source's invariant. It measured a shape no production path builds and no other site agrees
        with, which is also what made its medians incomparable to the harness cells at the same
        ``(N_token, cp, D)``.

    Input requirements
        ``mesh``, ``D`` and ``N_token`` are declared :data:`FRONT_A2A` pool values, with ``N_token``
        narrowed to :data:`_PERF_NTOK`. The cell skips -- coherently, through ``mesh_or_skip`` /
        ``rank_invariant_skip`` / ``gated_skip``, never a bare ``pytest.skip``, which under
        ``tests/distributed/**`` is a deadlock -- when this launch cannot supply the mesh, when the
        job has NO IB peer (``ib_drain`` would collapse to the coupled store and measure something
        else under this file's name), when ``D`` does not divide the rank count, when the picked
        ``tile_N`` falls under the SM90 WGMMA CTA-tile floor, when the token block is not a whole
        number of CTA M-tiles, or when the operands do not fit -- the last decided by CATCHING
        ``OutOfMemoryError``, never by a memory estimate.
    """
    _log_diag_predicate_once(dist_manager)
    mesh_or_skip(mesh, _mesh_ranks(mesh), kernel="dual_gated_gemm_a2a_ibdrain")
    apply_mesh(mesh)

    cp = world_size
    if D % cp != 0:
        rank_invariant_skip(
            f"D={D} does not divide cp={cp}; the per-rank feature shard is not integral",
            because="D is a declared matrix value and world_size is fixed by the launch, so both "
            "are identical on every rank",
        )
    pm = _pe_map(dist_manager)
    pe_table = tuple(int(v) for v in pm.cp_pe_table.tolist())

    # The fabric gate. `job_has_ib_peers` is the ONE shared probe the correctness path uses, so this
    # gate and that one agree by construction rather than by two copies of the same arithmetic.
    if not job_has_ib_peers(pe_table):
        rank_invariant_skip(
            f"ib_drain needs >=1 IB (non-P2P) peer; this job is all-NVLink (pe_table={pe_table}). "
            f"Harvest and enforce this file on a 2-node allocation.",
            because="the peer table is derived from the job's own rank->node map, identical on "
            "every rank",
        )

    Dloc, tile_M, tile_N, rows_per_peer = _front_geometry_or_skip(N_token, D, cp)
    K = 256
    torch.manual_seed(4321 + int(dist_manager.rank))

    def _alloc():
        # `rows_per_peer` rows, NOT `cp * rows_per_peer` -- see this test's docstring for what the
        # old spelling measured and why nothing else in the tree agrees with it.
        x = torch.randn(rows_per_peer, K, device=device, dtype=torch.bfloat16) / (K**0.5)
        Wg2 = torch.randn(2 * D, K, device=device, dtype=torch.bfloat16) / (K**0.5)
        Wp2 = torch.randn(2 * D, K, device=device, dtype=torch.bfloat16) / (K**0.5)
        with torch.cuda.use_mem_pool(DistributedManager.symmetric_mempool(_this_rank_device())):
            recv = torch.empty((2 * Dloc, cp * rows_per_peer), dtype=torch.bfloat16, device=device)
        return x, Wg2, Wp2, recv

    x, Wg2, Wp2, recv = _oom_gate(f"operands + recv at N_token={N_token}, D={D}", _alloc)

    def _cfg(g):
        g.configure_a2a(
            cp=cp,
            my_cp_rank=int(pm.my_cp_rank),
            rows_per_peer=rows_per_peer,
            pe_table=pe_table,
            ib_drain=True,
            ring_depth=_RING_DEPTH,
            consumer_warpgroups=_CWG,
        )

    # Through the same OOM gate as the operands: this builder allocates out_postact (2D, M) AND the
    # symmetric staging ring, and at the top of the ladder out_postact is the largest single tensor
    # in the cell. Gating only the operands would move the failure one line down, out of the skip.
    compiled, run_fn, _gemm_obj, ring_t = _oom_gate(
        f"ib_drain build at N_token={N_token}, D={D}",
        lambda: _build_front_decoupled(
            x,
            Wg2,
            Wp2,
            (tile_M, tile_N),
            recv_t=recv,
            ring_depth=_RING_DEPTH,
            consumer_warpgroups=_CWG,
            configure_fn=_cfg,
        ),
    )
    try:

        def _measure() -> float:
            return BU.benchmark_single(
                run_fn,
                rounds=_ROUNDS,
                warmup=_WARMUP,
                iters=1,
                # See the sibling cell: default returns `dist_manager` unchanged.
                dist=_timer_dist(dist_manager),
                mode=TIMING["mode"],
                reduce=TIMING["reduce"],
            ).median_ms

        # Decided HERE, not inside assert_cell, which skips with a bare `pytest.skip` -- correct for
        # tests/perf/ and refused under tests/distributed/**. Gated on NOT-harvesting: during a
        # harvest the pin is absent BY DEFINITION, and an unconditional skip makes the harvest
        # structurally impossible (measured on the sibling: rc=0, harvested=0, on every cell).
        key = ("front_ibdrain", _mesh_label(mesh), D, N_token)
        if os.environ.get("CPO_PERF_MEASURE", "0") != "1" and tuple(key) not in PINS:
            rank_invariant_skip(
                f"no pin for {key}; harvest on a 2-node allocation with CPO_PERF_MEASURE=1 and one "
                f"launch per (mesh, D, N_token)",
                because="PINS is loaded from a file shipped in the repo, identical on every rank",
            )
        # `_barrier_witness` proves WHICH arm ran from the log's own bytes (ARM A -> 0 calls,
        # ARM B -> 495). A no-op wrapper on the default path: it counts a function nothing calls.
        with _barrier_witness(dist_manager):
            assert_cell(_measure, key, PINS, __file__)
    finally:
        # ib_drain puts the ring on the SYMMETRIC heap (it is the put SOURCE), so it must be
        # released before the next allocation -- an unfreed symmetric buffer wedges the session or
        # faults at the next collective alloc. `main` frees both; the bring-back dropped it once
        # already (fixed in 6461fe3), so it is spelled out here rather than assumed.
        compiled.free()
        del ring_t
