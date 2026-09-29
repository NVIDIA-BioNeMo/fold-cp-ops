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


"""PERF GATE: the A2A-fused TriMul WORKFLOW -- ``workflows.trimul_autotuned.trimul_a2a``.

Subject is the COMPOSITE, not a kernel. The two A2A kernel gates in this directory pin the front
projection and the back einsum separately, 408 live cells between them, and were all green while
the composite ran 1.378x vs ``main`` -- because a per-kernel pin cannot see interleave, per-call
dispatch cost, or a variant-selection inversion. The out-gate inversion this repo fixed was exactly
that class and was found by reading two trees side by side, not by any gate.

``tests/perf/test_benchmark_perf_trimul_autotune.py`` is the cp=1 workflow and cannot stand in: its
own docstring records that it uses ``mode="device"`` because the cp=1 path is collective-free, and
that "the distributed A2A-fused path must use ``mode="event", reduce="max"`` instead". A collective
under an adaptive per-rank rep count DESYNCS.

**AUTOTUNE IS ON, and that is the point.** The gate calls the user-facing entry with
``CPO_DIST_AUTOTUNE=1`` so each fused build autotunes its own back/front stores internally via the
freeze-cached ``trimul_autotune_policy``, exactly as ``harness/targets/trimul_e2e.py`` builds it.
Pinning a hand-picked config would guarantee performance for a path no user runs.

**WHY THE MATRIX IS DECLARED HERE** and not in the correctness module, against the usual rule. The
correctness module ``tests/distributed/workflows/test_trimul_autotuned.py`` is
``pytestmark = matrix_exempt(...)`` at MODULE level, so a matrix placed there would be decorative --
and the coverage audit now rides matrix OWNERSHIP, which makes a decorative axis under
``tests/distributed/**`` RED rather than silent. Declaring it in the gate keeps the axes owned by
the module that parametrizes from them. Moving it is a follow-up that must move the exemption too.

**Off-grid N_token is absent at cp16 BY CONSTRAINT, not by omission.** cp16 derives ``N/16`` and the
per-peer extent must clear the 16-byte floor (bf16 -> %8), so cp16 forces ``N % 128 == 0``. Every
value legal on all eight meshes is therefore a multiple of 128, and the genuinely off-grid extents
(3008, 5952, 10048 -- none a multiple of 128) are legal on the other seven.
"""

import json
import contextlib
import os

import pytest
import torch

from fold_cp_ops.testing.collective_guard import gated_skip, rank_invariant_skip
from fold_cp_ops.testing.capacity_guard import is_capacity_error
import tests.perf.pins as pins_mod
from tests.perf import calibration
from tests.perf.calibration import assert_cell, rounds_for_cell
from tests.perf.pins import load as load_pins

from fold_cp_ops.testing.kernel_matrix import (
    Axis,
    KernelMatrix,
    Unsupported,
    computes_nothing_numeric,
    matrix_exempt,
)
from fold_cp_ops.testing.numeric_guard import numeric_exempt

# --------------------------------------------------------------------------- #
# The declared coverage. 8 meshes x 7 N_token x 4 D x 2 directions = 448, less the
# 24 off-grid-at-cp16 combinations the region below refuses = 424 legal cells.
# --------------------------------------------------------------------------- #
_MESHES = ("cp2", "cp4", "cp8", "cp16", "cp2x2", "cp2x4", "cp2x8", "cp4x4")
_MESH_SHAPE = {
    "cp2": (2, 1),
    "cp4": (4, 1),
    "cp8": (8, 1),
    "cp16": (16, 1),
    "cp2x2": (2, 2),
    "cp2x4": (2, 4),
    "cp2x8": (2, 8),
    "cp4x4": (4, 4),
}
#: ``apply_mesh`` takes a mesh SPEC (the ("cp", shape) form the matrix axis uses), not the label.
_MESH_SPEC = {m: (("cp", (a, b) if b > 1 else a),) for m, (a, b) in _MESH_SHAPE.items()}
_ON_GRID = (2048, 4096, 8192, 12288)
_OFF_GRID = (3008, 5952, 10048)


def _mesh_world(mesh):
    """Rank count a mesh label needs. Input must be a key of ``_MESH_SHAPE``; KeyError otherwise."""
    a, b = _MESH_SHAPE[mesh]
    return a * b


def _is_off_grid(n):
    """True when ``n`` is NOT a multiple of 128 -- the extents cp16 cannot derive an aligned shard from."""
    return int(n) % 128 != 0


TRIMUL_WORKFLOW = KernelMatrix(
    kernel="trimul_a2a_workflow",
    axes=(
        Axis(
            name="mesh",
            domain=(
                "every 1-D and 2-D cp sharding at world <= 16. The 1-D pool spans NVLink-only "
                "(cp2/cp4/cp8 on one node) AND the hybrid NVLink-intra/IB-inter shape (cp16, two "
                "nodes); the 2-D pool spans a node-local factorisation (cp2x2, cp2x4) and two that "
                "must cross the fabric (cp2x8, cp4x4). A pool of 1-D alone would never exercise the "
                "per-axis derived extents that the 2-D drain routes on"
            ),
            values=_MESHES,
            facets={
                "two_dim": lambda m: "x" in m,
                "cross_node": lambda m: _mesh_world(m) > 8,
            },
        ),
        Axis(
            name="N_token",
            domain=(
                "any token extent whose per-peer shard clears the 16-byte floor on every cp axis. "
                "The pool deliberately carries BOTH multiples of 128 and extents that are not: the "
                "aligned ones are what production runs, and the unaligned ones are the only values "
                "that reach the partial-tile and straddle paths at all"
            ),
            values=_ON_GRID + _OFF_GRID,
            facets={
                "off_grid": _is_off_grid,
                "large": lambda n: int(n) >= 8192,
            },
        ),
        Axis(
            name="D",
            domain=(
                "the four TriMul feature widths. 128 and 384 were live blind spots that hid real "
                "defects, so a pool of {256, 512} is not a subset -- it is the one that missed them"
            ),
            values=(128, 256, 384, 512),
            facets={"narrow": lambda d: int(d) <= 256},
        ),
        Axis(
            name="direction",
            domain=(
                "the two A2A directions. They are not symmetric: the incoming path carries an "
                "internal transpose temp the outgoing one does not, so timing one and inferring the "
                "other would miss an algorithmic cost rather than a constant factor"
            ),
            values=("outgoing", "incoming"),
            facets={"incoming": lambda v: v == "incoming"},
        ),
    ),
    computes=computes_nothing_numeric(
        because=(
            "a perf gate's subject is the composite workflow's measured SPEED at a declared cell, "
            "not a numerical result. The workflow's numerics are asserted in "
            "tests/distributed/workflows/test_trimul_autotuned.py, which owns that question; "
            "duplicating the comparison here would give two places to update and one to forget"
        )
    ),
    unsupported=(
        Unsupported(
            where=lambda mesh, N_token: mesh == "cp16" and _is_off_grid(N_token),
            raises=ValueError,
            match=r"16-B|multiple of|not token-shardable|per-peer",
            reason=(
                "cp16 shards the token-i axis 16 ways, so the per-peer extent is N/16 and must "
                "clear the 16-byte floor (bf16 -> 8 elements). That forces N % 128 == 0, which the "
                "off-grid pool values (3008, 5952, 10048) are chosen to violate. Their absence at "
                "cp16 is the constraint doing its job -- at every other mesh in the pool they are "
                "supported and MUST be timed, which is why they are in the pool at all"
            ),
        ),
    ),
)


# --------------------------------------------------------------------------- #
# The gate. Two questions, and the second is the one a timing-only gate cannot ask.
# --------------------------------------------------------------------------- #
PINS = load_pins(__file__)

#: Lockstep timing, required by this directory's ``pytest_collectstart`` and correct on the merits:
#: the workflow is collective-bearing, so ``mode="device"``'s adaptive per-rank rep count would
#: DESYNC the ranks, and ``reduce="max"`` makes the median a property of the slowest PE -- i.e. of
#: the JOB -- rather than of whichever rank happened to finish first.
TIMING = {"mode": "event", "reduce": "max"}

#: The probe pair for the two-point solve: `device` is obtained from `iters=n1` and `iters=n2`
#: instead of paying `iters=20` to shrink the per-call host-dispatch bias (see
#: `bench_timing.benchmark_extrapolated`). THE DEFAULT, because the pins are harvested in this
#: regime -- a gate whose default configuration cannot read its own pin file has no default.
#:
#: `CPO_PERF_EXTRAPOLATE` overrides the pair, or disables the correction entirely with `0` / `off`
#: (which then reports `single` and can only be run against a `single`-regime pin file).
#:
#: COLON-separated, not comma: `srun --export` takes a COMMA-separated list, so a comma inside a
#: value is eaten by the launcher and the variable arrives truncated to its first field. Measured --
#: `CPO_PERF_EXTRAPOLATE=1,4` reached the test as `1` and every cell died on `n1, n2 = probe_iters`.
#:
#: `(1, 4)` is chosen on cost/noise, not accuracy: a test asserts the extrapolate does not depend on
#: the pair. 5 launches per window against `iters=20`'s 20, at ~1.37x noise amplification -- ~0.82%
#: against a 2.5% band.
_DEFAULT_PROBE_ITERS = (1, 4)
_EXTRAPOLATE_ENV = os.environ.get("CPO_PERF_EXTRAPOLATE")
_EXTRAPOLATE = (
    None
    if _EXTRAPOLATE_ENV in ("0", "off", "none")
    else tuple(int(v) for v in _EXTRAPOLATE_ENV.split(":"))
    if _EXTRAPOLATE_ENV
    else _DEFAULT_PROBE_ITERS
)

#: Lazily-measured `C` for THIS process, as a one-element list so the closure can fill it.
#: Enabled by `CPO_PERF_PROBE_C=1`. `None` means the mode is off; `[value]` means measured.
#:
#: One probe per PROCESS, not per cell: `C` is the callable's Python dispatch cost and does not vary
#: with tensor extent (323 us from a cheap shape vs 335 us weighted from the full-size cells).
#: That is what removes the second timing point the two-point solve pays on every cell.
_PROBE_C = [] if os.environ.get("CPO_PERF_PROBE_C") == "1" else None

#: THREE regimes, not two. I first wrote that the two corrected estimators "report the same quantity
#: so they share a pin regime". Measured, they do not: a global `C` leaves each cell a residual of
#: `(C_cell - C_probe) / device`, and at cp8 that is 2.4% on the smallest cell against a 2.5% band --
#: 12 of 13 pass, the smallest fails at 1.033x. So probing gets its OWN label, and the estimator
#: guard refuses to compare it against pins harvested by the per-cell solve.
_ESTIMATOR = (
    pins_mod.ESTIMATOR_PROBED
    if _PROBE_C is not None
    else pins_mod.ESTIMATOR_EXTRAPOLATED
    if _EXTRAPOLATE
    else pins_mod.ESTIMATOR_SINGLE
)


class _Cell:
    """The minimal ``ctx`` the harness target's builders read.

    Purpose
        ``benchmark/distributed/harness/targets/trimul_e2e.py`` owns the construction of a fused
        workflow cell -- mesh, weights, local shard, DTensor wrap -- and is the path the e2e
        benchmark actually runs. Reusing it is what makes this gate time the SAME thing the
        benchmark does; re-deriving the build here would create a second definition of "the
        workflow" and one of them would drift.

    Input requirements
        ``cp0``/``cp1`` must multiply to the launch's WORLD_SIZE (checked by the caller via
        ``_skip_unless_this_world``); ``Dloc`` is the PER-PEER feature width, so the full ``D`` the
        matrix declares is ``Dloc * cp0 * cp1`` and the caller divides. ``Dloc >= 8`` is the front
        feature-scatter floor -- at D=128/cp=16 it is exactly 8, which is why 128 stays in the pool.
    """

    def __init__(self, mesh, N_token, D, dm, B=1):
        a, b = _MESH_SHAPE[mesh]
        self.cp0, self.cp1 = a, b
        self.cp = a * b
        self.N, self.B = int(N_token), int(B)
        self.Dloc = int(D) // self.cp
        self.cfg = {}
        # `dm` and `rank` are read by the builder (trimul_e2e.py:363, :388). Enumerated from the
        # target's own `ctx.` accesses rather than discovered one AttributeError at a time -- the
        # first smoke died on `.dm` alone, and fixing only that surfaced `.cp` on the next launch.
        #
        # The DM must come from the `apply_mesh` FIXTURE, not from `DistributedManager()`: the Borg
        # singleton exists without a device mesh, and `_fused_mesh` reads `dm.device_mesh.ndim`
        # straight into an AttributeError. The fixture is what builds the mesh on the live world
        # group -- prior art that already handles the rank-invariant skip when the world cannot
        # form the spec.
        self.dm = dm
        self.rank = int(dm.rank)


#: The probe shape, FIXED rather than inherited from the cell. Chosen from a measured sweep, not
#: guessed: at cp8 the two-point solve returns 328.6/328.6 us here across independent runs, against a
#: 335 us reference solved on the full-size cells. See `measure_workflow_dispatch_cost` for the two
#: regimes that make an arbitrary shape return a plausible wrong number instead of an error.
#: Target DEVICE seconds for one whole harvest cell (all `N_SAMPLES_PIN_TIME` samples), from which
#: `calibration.rounds_for_cell` derives the timed-round count. 60 s because it leaves every cell
#: this gate has pinned at N<=4096 untouched -- the count only drops above ~23 ms, which is 51 of
#: the 125 -- while cutting the N=8192/12288 ladder, where one cell reaches 56 min at the ceiling,
#: by ~4.3x.
#:
#: A BUDGET rather than a fixed count, because the thing being bounded is TIME and the thing being
#: chosen is a repetition count. Pinning the count is the defect this replaces: a 1.76 ms cell and a
#: 245 ms cell both fired 2925 launches.
_ROUNDS_BUDGET_S = float(os.environ.get("CPO_PERF_ROUNDS_BUDGET_S", "60"))

#: This gate's pin and its run are routinely taken on DIFFERENT MACHINES -- a pin harvested in one
#: A multi-node allocation is gated in the next, on whichever nodes Slurm hands out --
#: so its band needs the cross-machine term `calibration._CROSS_SESSION_REL` measures. Opted in HERE
#: rather than defaulted on, because a single-device gate whose pin and run share a box must not
#: silently inherit a widening measured on a two-node A2A workflow.
_CROSS_SESSION_REL = calibration._CROSS_SESSION_REL

#: Launches one timed round issues, for the budget arithmetic. `benchmark_extrapolated` times the
#: callable at BOTH probe points inside ONE round, so it is `n1 + n2` -- not `max`, and not `iters`.
#: Wrong here and the budget is off by that factor while still returning a plausible count.
_LAUNCHES_PER_ROUND = sum(_EXTRAPOLATE) if _EXTRAPOLATE else 1

_PROBE_D = 256
_PROBE_N_PER_CP = 128


def measure_workflow_dispatch_cost(
    mesh, D=None, direction="outgoing", dm=None, *, n_token=None, rounds=15
):
    """Measure ``C`` -- the event window's per-call bias -- with the window itself, at a FIXED shape.

    Purpose
        ``C`` is the host cost of submitting ``fn()``: `time_callable` records the start event before
        the Python call runs, so the device idles one dispatch before the first kernel lands, and
        `measured(N) = device + C/N` divides that gap by ``N``. Solving it lets the gate run at
        ``iters=1`` instead of paying ``iters=20`` to shrink it.

    Semantics
        Times the SAME callable the gate times, at two cheap ``iters`` points, and solves
        ``C = (m1 - m4) / (1/1 - 1/4)`` exactly.

        **`C` is DEFINED by the event window, so the event window is what can measure it.** Two other
        instruments were tried and neither measures this quantity:

        * `bench_timing.host_dispatch_us` (re-exported by `bench_utils`, and still reachable as
          `calibration.host_dispatch_us`) -- the repo's sanctioned host-dispatch primitive, and the
          right tool for what it is for. Its contract requires "a callable issuing exactly ONE
          launch"; `_run_fusedcp` issues many plus in-kernel collectives, so backpressure hits
          immediately and it returns DEVICE time. Measured: 1074 us at D=128/N=1024 rising to
          38601 us at D=384/N=4096, which is that cell's device time to three digits.
        * a `perf_counter` span with a per-call `synchronize` -- forbidden by CLAUDE.md, and wrong
          anyway: ~1030 us, 3x high, because syncing first measures cold-queue latency and the span
          covers all of ``fn()`` rather than the submit the window waits on.

    Input requirements
        The shape is FIXED (`_PROBE_D`, `_PROBE_N_PER_CP`) and deliberately not taken from the cell,
        because the solve has two regimes that return a plausible float rather than an error:

        * device time BELOW ``C`` -> host-bound, ``m1 ~ m4``, the difference collapses. Measured at
          D=128/N=1024: ``C`` reads 43 us. **This one shipped**: the gate probed at the cell's own
          D=128, subtracted 56 us where ~335 us was needed, and failed 7 of 13 cells.
        * device time far ABOVE ``C`` -> ``C`` is a small difference of two ~30 ms numbers, i.e.
          noise. Measured at D=256/N=4096: -25 us.

        ``D`` and ``direction`` are accepted and IGNORED for the shape; they remain in the signature
        so the call site reads like the cell it is probing for.

    Returns:
        ``C`` in milliseconds, from this rank. Callers subtracting it from a collective-bearing
        measurement should reduce it, as the timer does for its own medians.

    Raises:
        RuntimeError: if the solve lands outside the regime it is valid in -- ``C`` non-positive, or
            larger than the ``iters=4`` reading itself. Refused rather than returned: a wrong ``C``
            is subtracted from every cell and is invisible in the result.
    """
    from benchmark.distributed.bench_utils import time_callable
    from benchmark.distributed.harness.targets import trimul_e2e as T
    import statistics as _st

    a, b = _MESH_SHAPE[mesh]
    n = int(n_token) if n_token else _PROBE_N_PER_CP * (a * b)
    ctx = _Cell(mesh, n, _PROBE_D, dm)
    handle = T._build_fusedcp(direction)(ctx)
    try:
        dev = torch.device("cuda", torch.cuda.current_device())
        run = lambda: T._run_fusedcp(handle)  # noqa: E731
        for _ in range(3):
            time_callable(run, 4, dev)
        m1 = _st.median([time_callable(run, 1, dev) for _ in range(rounds)])
        m4 = _st.median([time_callable(run, 4, dev) for _ in range(rounds)])
        c = (m1 - m4) / (1.0 / 1 - 1.0 / 4)
        if not (0.0 < c < m4):
            raise RuntimeError(
                f"probe C = {c * 1000:.1f} us is outside the regime the two-point solve is valid in "
                f"(m1={m1 * 1000:.1f} us, m4={m4 * 1000:.1f} us at mesh={mesh} N={n} D={_PROBE_D}). "
                f"C <= 0 means the run was host-bound and the difference collapsed; C >= m4 means "
                f"the device time dwarfs C and the difference is noise. Refused rather than "
                f"returned: a wrong C is subtracted from EVERY cell and is invisible in the result."
            )
        return c
    finally:
        T._teardown_fusedcp(handle)


@contextlib.contextmanager
def _built_workflow_cell(mesh, N_token, D, direction, dm, rounds=None):
    """Build the cell ONCE and yield a zero-arg ``measure()`` that only TIMES it.

    Purpose
        `assert_cell` calls its measure callable ``N_SAMPLES_TEST_TIME`` (or ``_PIN_TIME``) times.
        Before this, `measure()` rebuilt the whole cell on EVERY call, so the
        build is paid 9 or 15 times per cell. MEASURED at cp8, floor timing: a rebuild is 5.65 s, of
        which `cute.compile` is 1.33 s x ~3 calls (72%) and the local-shard alloc 1.24 s (22%) --
        i.e. the SAME kernels recompiled 27 times in one process. That build cost, not the
        benchmarking, was 47% of a gate cell.

    Semantics
        Builds once, yields a closure that runs only ``benchmark_single``, and tears down on exit.
        The outer sampling is NOT weakened: `calibration` takes those samples to reject "an excursion
        that spans a WHOLE `benchmark_single` call" -- a property of MACHINE state, which independent
        timing calls still sample. What is dropped is re-JIT-ing and re-allocating identical objects
        between them.

    Input requirements
        As the gate test's cell parameters. The caller MUST keep this inside its
        `torch.cuda.OutOfMemoryError` handler: the build moved into the ``with``, so a build OOM now
        raises at ``__enter__`` rather than inside `assert_cell`.

    Yields:
        A zero-arg callable returning the slowest-PE median in ms.

    Raises:
        torch.cuda.OutOfMemoryError: from the build, for the caller's OOM skip.
    """
    # The pins encode an ENVIRONMENT this test does not create. `measured_on` in the pin file says
    # "AUTOTUNE ON (CPO_DIST_AUTOTUNE=1) with CPO_DIST_AUTOTUNE_FREEZE_DIR=...", and
    # `trimul_autotuned.py` reads that variable straight off `os.environ`. This module's own
    # docstring asserts in bold that "the gate calls the user-facing entry with
    # CPO_DIST_AUTOTUNE=1" -- but nothing set it and nothing checked it, so the claim described the
    # launcher's habits rather than the code's behaviour.
    #
    # MEASURED COST of leaving that unchecked, 2026-08-27: a full sweep ran with the flag unset, so
    # every cell built the HEURISTIC config while the pins record a per-shape-TUNED one. It produced
    # a stable ~9% "regression" on cp8/D512/N4096 that was nearly attributed first to a kernel change
    # and then to a venue offset -- and, worse, an `e2e_a_hyb` "2 passed". **The false GREEN is the
    # dangerous half**: a failure gets investigated, a pass is never revisited.
    #
    # Asserted rather than SET, deliberately. Setting it here would paper over a launcher that forgot
    # and make the pin file's `measured_on` unverifiable from the run. The gate should refuse to
    # produce a number it cannot compare.
    if os.environ.get("CPO_DIST_AUTOTUNE") != "1":
        raise RuntimeError(
            "perf gate requires CPO_DIST_AUTOTUNE=1; got "
            f"{os.environ.get('CPO_DIST_AUTOTUNE')!r}. The pins were harvested with autotune ON "
            "(see `measured_on` in test_benchmark_perf_trimul_autotuned.json), so without it every "
            "cell builds the HEURISTIC config and is compared against a per-shape-TUNED pin -- a "
            "comparison this run is not entitled to make, in either direction. Export it (and "
            "CPO_DIST_AUTOTUNE_FREEZE_DIR to the pins' freeze dir) from the launcher."
        )
    from benchmark.distributed.bench_utils import benchmark_extrapolated, benchmark_single
    from benchmark.distributed.harness.targets import trimul_e2e as T

    if _PROBE_C is not None and not _PROBE_C:
        # Measured once, on the first cell of the process, and reused for every later one.
        _PROBE_C.append(measure_workflow_dispatch_cost(mesh, D, direction, dm))
        if int(getattr(dm, "rank", 0)) == 0:
            print(f"\n[probe] C = {_PROBE_C[0] * 1000:.1f} us (once per process)", flush=True)

    ctx = _Cell(mesh, N_token, D, dm)
    handle = T._build_fusedcp(direction)(ctx)

    def _timed():
        """One measurement of the built cell, by whichever estimator is selected."""
        run = lambda: T._run_fusedcp(handle)  # noqa: E731
        if _PROBE_C is not None:
            # subtract a `C` measured ONCE per process from a cheap-shape probe of this same
            # callable, and time at iters=1. Reports the same `device` term the two-point solve
            # does -- validated at 323 us vs the 335 us weighted mean of the cells that constrain
            # `C` -- for a fifth of the launches, because there is no second timing point per cell.
            return float(
                benchmark_single(run, iters=1, subtract_overhead=_PROBE_C[0], **TIMING).median_ms
            )
        if _EXTRAPOLATE:
            # Solve `device` from two CHEAP points instead of paying iters=20 to hide the
            # per-call host dispatch -- see `bench_timing.benchmark_extrapolated`.
            kw = {} if rounds is None else {"rounds": rounds}
            return float(
                benchmark_extrapolated(run, probe_iters=_EXTRAPOLATE, reduce="max", **kw).median_ms
            )
        return float(benchmark_single(run, **TIMING).median_ms)

    try:
        yield _timed
    finally:
        T._teardown_fusedcp(handle)


_POISON_SIG = (
    "illegal memory access",
    "unspecified launch failure",
    "cudaerrorillegaladdress",
    "unhandled cuda error",
    "unhandledcudaerror",
)


def _context_is_dead():
    """Ask the DEVICE whether its context still works, instead of pattern-matching a message.

    Purpose
        A poisoned CUDA context fails every subsequent call, so the cheapest possible operation is a
        complete test: if a 1-element allocation and sync cannot succeed, nothing can.

    Semantics
        Measured, and this is why the guard could not stay string-based: an IMA killed a world-4
        harvest, but the text `illegal memory access` NEVER REACHED PYTHON -- it was printed by
        nvshmem's C layer to stderr, while the exception pytest caught said
        `ncclUnhandledCudaError: ... Failed to CUDA calloc async 8 bytes`. A string list can only
        ever match the spellings someone has already seen; the probe matches the CONDITION.

        ("Failed to CUDA calloc async 8 bytes" is itself the tell -- an 8-byte allocation does not
        fail for want of memory.)

    Input requirements
        None. Safe to call from an exception handler: it catches everything and answers rather than
        raising, because a guard that can itself raise turns one failure into two.

    Returns:
        True if the context is unusable, False if a trivial op still succeeds (including when CUDA
        was never initialised, which is not a poisoned context).
    """
    try:
        import torch

        if not torch.cuda.is_available():
            return False
        torch.zeros(1, device="cuda").add_(1.0)
        torch.cuda.synchronize()
        return False
    except BaseException:  # noqa: BLE001 -- ANY failure of a 1-element op means the context is gone
        return True


def _abort_session_if_context_is_poisoned(exc):
    """End the SESSION on an unrecoverable CUDA fault instead of letting it cascade.

    Purpose
        A CUDA context that has taken an illegal memory access is STICKY: every later CUDA call in
        that process fails. So this is not a cell that failed, it is a process that is over -- and
        pytest faithfully reports every remaining cell as broken.

    Semantics
        Measured on this gate: one unpinned large cell exhausted the symmetric heap, the failure
        surfaced as an IMA inside an nvshmem barrier, and the run produced **52 failures and 192
        `torch.AcceleratorError`s** across cells that never really ran, exiting 255. The first real
        failure was line 28 of a 186-line progress file; the rest was noise generated by the fault.

        Decided by :func:`_context_is_dead` -- a probe, not a message match. The message list is kept
        only as a fast path for the spellings already seen; a LATER harvest was killed by an IMA whose
        Python-visible exception was `ncclUnhandledCudaError` and contained none of them.

        NOT a substitute for per-cell isolation, which CLAUDE.md requires for a multi-cell sweep and
        which is what actually contains the blast radius. This makes the failure LEGIBLE when several
        cells do share a process.

        Rank-uniform without gating: the fault is a property of the device context, so every rank that
        touches it sees it and no rank exits alone.

    Input requirements
        ``exc`` is any exception caught around a cell. Anything that leaves the context WORKING
        returns normally, so the CALLER must re-raise; this never swallows.

    Returns:
        None, when the context still works.

    Raises:
        Exception: via :func:`pytest.exit`, ending the session with a non-zero status.
    """
    msg = (str(exc) or "").lower()
    if not any(sig in msg for sig in _POISON_SIG) and not _context_is_dead():
        return
    pytest.exit(
        f"CUDA context is dead; every later cell in this process would report a failure it did not "
        f"cause: {exc!r}\n"
        f"Re-run the remaining cells in a FRESH process, one launch per cell (CLAUDE.md's per-cell "
        f"isolation rule). On a large cell the underlying cause is most likely symmetric-heap "
        f"exhaustion, which surfaces as an IMA rather than as torch.cuda.OutOfMemoryError and so "
        f"never reaches the OOM skip.",
        returncode=2,
    )


def _skip_unless_this_world(mesh):
    """Skip when the launch's WORLD_SIZE cannot form ``mesh``.

    A launch fixes one rank count, so a mesh needing a different one is unreachable in THIS job
    rather than unsupported. The decision is job-uniform -- every rank reads the same WORLD_SIZE --
    so it is a ``rank_invariant_skip`` and cannot diverge a collective.

    Args:
        mesh: A label from the declared pool; anything else is a KeyError rather than a silent pass.
    """
    need = _mesh_world(mesh)
    have = int(os.environ.get("WORLD_SIZE", "0"))
    if have != need:
        rank_invariant_skip(
            f"mesh {mesh} needs WORLD_SIZE={need}, this launch has {have}",
            because="WORLD_SIZE is a launch property; every rank reads the same value",
        )


_OFF_GRID_MESHES = tuple(m for m in _MESHES if m != "cp16")
_OFF_BECAUSE = (
    "cp16 is excluded because the declared region refuses it: cp16 derives N/16 and the per-peer "
    "extent must clear the 16-byte floor, which forces N % 128 == 0 -- and every value in the "
    "off-grid pool is chosen NOT to be a multiple of 128. Narrowing the MESH rather than dropping "
    "the N values keeps off-grid coverage on the seven meshes that do support it"
)


@TRIMUL_WORKFLOW.parametrize(
    "mesh",
    "N_token",
    "D",
    "direction",
    only={"N_token": _ON_GRID},
    because="the on-grid half of the token pool, legal on every declared mesh",
)
def test_the_fused_workflow_median_is_within_its_pin(mesh, N_token, D, direction, apply_mesh):
    """The AUTOTUNED workflow, end to end, at a declared cell.

    Times ``TriangularMultiplication`` -- the shipped nn.Module, via the harness target's
    ``_build_fusedcp`` -- with ``CPO_DIST_AUTOTUNE=1``, so what is pinned is the config a user
    actually gets. (It reaches ``trimul_a2a`` underneath, which is what this line used to say; the
    distinction matters because the sibling target ``trimul_fused`` times the RAW engine instead.) Pinning a hand-chosen store would guarantee a number
    for a path nobody runs, and would go on passing after the heuristic drifted away from it.

    ``mode="event", reduce="max"``: the workflow is collective-bearing, and ``mode="device"``'s
    adaptive per-rank rep count DESYNCS collectives. The slowest-PE reduction is what makes the
    median a property of the JOB rather than of whichever rank finished first.

    One-sided, like every gate here: a pass means "not slower than the pin", never "unchanged".
    """
    _skip_unless_this_world(mesh)
    dm = apply_mesh(_MESH_SPEC[mesh])
    key = ("workflow", mesh, D, N_token, direction)
    # Derived from THIS cell's own pin, before the closure is built, so the closure and the pin
    # cannot disagree -- `assert_cell` refuses them if they do. Pure arithmetic on bytes every rank
    # already has, so every rank computes the same count without having to agree on one, which is
    # what makes it safe for a gate whose every round carries a collective.
    rounds = rounds_for_cell(
        PINS, key, budget_s=_ROUNDS_BUDGET_S, launches_per_round=_LAUNCHES_PER_ROUND
    )
    oom_reason = None  # reduced AFTER the try; see the handler below
    try:
        with _built_workflow_cell(mesh, N_token, D, direction, dm, rounds=rounds) as measure:
            assert_cell(
                measure,
                key,
                PINS,
                __file__,
                estimator=_ESTIMATOR,
                rounds_used=rounds,
                cross_session_rel=_CROSS_SESSION_REL,
            )
    except BaseException as exc:  # capacity is recorded; anything else re-raises
        # Operands are O(N_token**2 * D / cp), so the largest cells legitimately do not fit at low
        # cp. A runtime capacity skip is the repo's rule -- never a memory-estimate gate, which would
        # encode today's box into the matrix. `is_capacity_error` covers BOTH heaps: the caching
        # allocator raises `OutOfMemoryError`, the symmetric heap raises
        # `RuntimeError: nvshmem_malloc failed`, and catching only the first left the top of the
        # ladder FAILING rather than skipping.
        #
        # The reason is RECORDED here and reduced AFTER the try, never skipped from inside this
        # handler. `gated_skip` all-reduces, so reaching it only on the ranks that raised leaves them
        # blocked in a collective their peers never enter -- a hang, not a skip. Reaching it
        # unconditionally is the property that makes the reduction safe.
        if not is_capacity_error(exc):
            # A poisoned context ends the session here rather than cascading; anything else is this
            # cell's own failure and propagates untouched.
            _abort_session_if_context_is_poisoned(exc)
            raise
        oom_reason = f"OOM building the cell mesh={mesh} N={N_token} D={D} dir={direction}"
    gated_skip(oom_reason)


@TRIMUL_WORKFLOW.parametrize(
    "mesh",
    "N_token",
    "D",
    "direction",
    only={"N_token": _OFF_GRID, "mesh": _OFF_GRID_MESHES},
    because=_OFF_BECAUSE,
)
def test_the_fused_workflow_median_is_within_its_pin_off_grid(
    mesh, N_token, D, direction, apply_mesh
):
    """The same timed cell at token extents that are NOT multiples of 128.

    Separated from the on-grid test only so cp16 can be narrowed away: the declared region refuses
    cp16 with off-grid N, and `parametrize` correctly refuses to emit a cell inside a region. Doing
    it by narrowing the MESH keeps these extents timed on the seven meshes that DO support them --
    dropping the N values instead would have removed the only cells that reach the partial-tile and
    straddle paths at all.
    """
    _skip_unless_this_world(mesh)
    dm = apply_mesh(_MESH_SPEC[mesh])
    key = ("workflow", mesh, D, N_token, direction)
    # Derived from THIS cell's own pin, before the closure is built, so the closure and the pin
    # cannot disagree -- `assert_cell` refuses them if they do. Pure arithmetic on bytes every rank
    # already has, so every rank computes the same count without having to agree on one, which is
    # what makes it safe for a gate whose every round carries a collective.
    rounds = rounds_for_cell(
        PINS, key, budget_s=_ROUNDS_BUDGET_S, launches_per_round=_LAUNCHES_PER_ROUND
    )
    oom_reason = None  # reduced AFTER the try; see the handler below
    try:
        with _built_workflow_cell(mesh, N_token, D, direction, dm, rounds=rounds) as measure:
            assert_cell(
                measure,
                key,
                PINS,
                __file__,
                estimator=_ESTIMATOR,
                rounds_used=rounds,
                cross_session_rel=_CROSS_SESSION_REL,
            )
    except BaseException as exc:  # capacity is recorded; anything else re-raises
        # Operands are O(N_token**2 * D / cp), so the largest cells legitimately do not fit at low
        # cp. A runtime capacity skip is the repo's rule -- never a memory-estimate gate, which would
        # encode today's box into the matrix. `is_capacity_error` covers BOTH heaps: the caching
        # allocator raises `OutOfMemoryError`, the symmetric heap raises
        # `RuntimeError: nvshmem_malloc failed`, and catching only the first left the top of the
        # ladder FAILING rather than skipping.
        #
        # The reason is RECORDED here and reduced AFTER the try, never skipped from inside this
        # handler. `gated_skip` all-reduces, so reaching it only on the ranks that raised leaves them
        # blocked in a collective their peers never enter -- a hang, not a skip. Reaching it
        # unconditionally is the property that makes the reduction safe.
        if not is_capacity_error(exc):
            # A poisoned context ends the session here rather than cascading; anything else is this
            # cell's own failure and propagates untouched.
            _abort_session_if_context_is_poisoned(exc)
            raise
        oom_reason = f"OOM building the cell mesh={mesh} N={N_token} D={D} dir={direction}"
    gated_skip(oom_reason)


#: Fraction by which the autotuner's runner-up must be slower before a heuristic that picked it is
#: called wrong. Set at the venue's own reproducibility floor rather than at 0: the pins in this
#: module are harvested at rel_std 0.005, so two configs within 2% are not distinguishable by the
#: instrument that would have to prove the difference. A disagreement UNDER this bar is a tie; a
#: disagreement with NO recorded margin is a FAILURE, because "cannot be shown to be a tie" and "is
#: a tie" are different states and only one of them is evidence.
_TIE_REL = 0.02


def dm_of(apply_mesh):
    """The ``DistributedManager`` the ``apply_mesh`` fixture built, whatever shape the fixture returns.

    Purpose
        ``apply_mesh`` is the prior art that builds a device mesh on the live world group and handles
        the rank-invariant skip when the world cannot form the spec. Unwrapping it lives in ONE place
        -- a second spelling is how a test ends up holding a bare ``DistributedManager()`` whose
        ``device_mesh`` is None, which surfaces as an AttributeError several frames into
        ``_fused_mesh`` rather than as "you passed the wrong object".

    Args:
        apply_mesh: The fixture value. Either the manager itself, or an object/sequence carrying one.

    Returns:
        The manager.

    Raises:
        TypeError: naming what it actually got, rather than returning something that fails later.
    """
    from fold_cp_ops.distributed.distributed_manager import DistributedManager

    if isinstance(apply_mesh, DistributedManager):
        return apply_mesh
    for cand in (getattr(apply_mesh, "dm", None), getattr(apply_mesh, "manager", None)):
        if isinstance(cand, DistributedManager):
            return cand
    if isinstance(apply_mesh, (tuple, list)):
        for cand in apply_mesh:
            if isinstance(cand, DistributedManager):
                return cand
    raise TypeError(
        f"apply_mesh did not yield a DistributedManager; got {type(apply_mesh).__name__}"
    )


def _configs_elected_and_heuristic(mesh, N_token, D, direction, dm):
    """Build the SAME cell twice -- autotune on, then off -- and return both resolved perf-configs.

    Purpose
        Give the heuristic-agreement test the two objects it must compare, read back from the built
        engine rather than re-derived. ``TriMulAutotuned`` records what it resolved on
        ``_front_cfg`` / ``_back_cfg`` (``trimul_autotuned.py:2484``); reading those is what makes
        the comparison about the config a USER receives. Re-calling ``_resolve_front_config`` /
        ``autotune_front_config`` here would compare two private resolvers and could agree perfectly
        while the engine used a third value -- the precedence chain in front of them (explicit
        override > autotune > ``_baked`` > heuristic, plus the ``_resolve_front_tile_n`` clamp and
        the design_e straddle downgrade) is exactly where a disagreement would hide.

    Semantics
        The two builds differ ONLY in ``CPO_DIST_AUTOTUNE``, restored in a ``finally``. The autotune
        build is freeze-cached per shape, so on a warm freeze dir it costs a lookup rather than a
        sweep; the heuristic build never consults the freeze at all. Each build is torn down before
        the next is made, because both hold a symmetric recv allocation and the pair does not fit at
        the larger cells.

    Args:
        mesh: A label from the declared pool -- KeyError on anything else.
        N_token, D: The cell's token and FULL feature extents; ``D`` must be divisible by the mesh's
            rank count (the matrix's declared pools guarantee it).
        direction: ``"outgoing"`` or ``"incoming"``; selects which A2A the workflow runs. It does
            NOT enter the perf-config key -- see the caller's narrowing.
        dm: The ``DistributedManager`` from the ``apply_mesh`` fixture, WITH a device mesh built. A
            bare ``DistributedManager()`` has ``device_mesh is None`` and dies in ``_fused_mesh``.

    Returns:
        ``(elected, heuristic)``, each ``{"front": (...), "back": (...)}`` of plain tuples.

    Raises:
        torch.cuda.OutOfMemoryError: propagated from either build for the caller's OOM skip.
    """
    from benchmark.distributed.harness.targets import trimul_e2e as T

    def _read(ctx):
        handle = T._build_fusedcp(direction)(ctx)
        try:
            eng = next(iter(handle["mod"]._engines.values()))
            return {"front": tuple(eng._front_cfg), "back": tuple(eng._back_cfg)}
        finally:
            T._teardown_fusedcp(handle)

    prev = os.environ.get("CPO_DIST_AUTOTUNE")
    try:
        os.environ["CPO_DIST_AUTOTUNE"] = "1"
        elected = _read(_Cell(mesh, N_token, D, dm))
        os.environ["CPO_DIST_AUTOTUNE"] = "0"
        heuristic = _read(_Cell(mesh, N_token, D, dm))
    finally:
        if prev is None:
            os.environ.pop("CPO_DIST_AUTOTUNE", None)
        else:
            os.environ["CPO_DIST_AUTOTUNE"] = prev
    return elected, heuristic


def _freeze_margin(mesh, N_token, D):
    """The autotuner's own best-vs-runner-up margin for this cell, or ``None`` if it did not record one.

    Purpose
        Decide whether a heuristic/autotune DISAGREEMENT is a defect or a tie. The sweep already
        timed every candidate and ``_margin_of`` kept the runner-up gap as ``margin["rel"]`` -- the
        fraction by which second place is slower. A disagreement under a near-zero ``rel`` is a
        coin-flip between configs the venue cannot separate; under a large ``rel`` it is the
        heuristic genuinely picking a slower config.

    Input requirements
        Must be called AFTER the autotune build, so the freeze entry for this shape exists on disk.
        Keys are reconstructed exactly as the policy builds them -- front ``(cp, N, D)``, back
        ``(cp, N, D, cp1)``, both with ``N`` put through ``anchor_n(..., dynamic=True)`` because the
        workflow builder passes ``dynamic=True``. A key built any other way silently misses and
        returns ``None``, which would read as "no margin recorded" rather than as a lookup bug.

    Returns:
        ``{"front": rel_or_None, "back": rel_or_None}``. A value is ``None`` when the entry is
        absent, or when fewer than two configs were timed (a one-candidate grid has no runner-up).
    """
    from fold_cp_ops.distributed.workflows.trimul_autotune_policy import _freeze_load, anchor_n

    a, b = _MESH_SHAPE[mesh]
    axes = (a, b) if b > 1 else (a,)
    n = anchor_n(int(N_token), axes, True)
    cp1 = (tuple(int(s) for s in axes) + (1,))[1]
    out = {}
    for kind, key in (("front", (a * b, n, int(D))), ("back", (a * b, n, int(D), cp1))):
        rec = _freeze_load(kind, key)
        # `_freeze_load` returns the CFG, not the record, so the margin comes from the file itself.
        try:
            with open(_freeze_path_for(kind, key)) as f:
                out[kind] = (json.load(f).get("margin") or {}).get("rel")
        except (OSError, ValueError):
            out[kind] = None
        del rec
    return out


def _freeze_path_for(kind, key):
    """Path of one freeze entry -- the policy's own ``_freeze_path``, re-exported for readability.

    Args:
        kind: ``"front"`` or ``"back"``; anything else raises inside the policy.
        key: The tuple built by the caller; joined into the filename verbatim, so a key whose
            element ORDER differs from the policy's produces a path that simply does not exist.

    Returns:
        An absolute path string. The file need not exist.
    """
    from fold_cp_ops.distributed.workflows.trimul_autotune_policy import _freeze_path

    return _freeze_path(kind, key)


@TRIMUL_WORKFLOW.parametrize(
    "mesh",
    "N_token",
    "D",
    "direction",
    only={"N_token": _ON_GRID, "direction": ("outgoing",)},
    because="on-grid is the production pool; and the perf config is keyed on (cp, N, D) ALONE -- autotune_front_config keys (cp, N, D) and autotune_back_config (cp, N, D, cp1), neither carrying direction -- so timing the incoming direction would build a second cell to re-derive a config already asserted, at no additional coverage",
)
def test_the_heuristic_picks_what_the_autotuner_measures_fastest(
    mesh, N_token, D, direction, apply_mesh
):
    """The shipped heuristic must select the config the autotuner finds fastest at this cell.

    This is the question a timing gate structurally cannot ask. A gate that times whatever the
    heuristic picked stays green while the heuristic drifts, because it re-pins the drifted choice
    as the new normal -- the measurement and the thing being measured move together.

    It is also how ``main`` derived the heuristic in the first place: sweep, then encode the winner.
    Re-asserting that correspondence is what keeps the encoded table honest as shapes are added.

    Agreement is asserted on the ELECTED CONFIG, not on a time -- but a config difference alone is
    not the verdict. Two configs whose medians sit inside the venue's own spread are not a defect,
    so a disagreement is checked against the margin the autotuner ITSELF recorded while sweeping
    (``margin["rel"]``, best vs runner-up); at or under ``_TIE_REL`` it is a tie. That reuses a
    measurement already taken instead of timing both configs again, which would double every cell.

    **"Heuristic" here means WHATEVER AUTOTUNE-OFF RESOLVES TO, which on this venue is often not the
    size-heuristic.** The precedence is explicit override > autotune > ``_baked`` > size-heuristic,
    and ``_resolve_sm90_ib_config`` returns a non-None ``_baked`` exactly when the venue matches its
    harvest venue -- sm90, cross-node IB, autotune off. Every one of those holds on the cluster this
    gate runs on. So a failure here names a stale BAKED TABLE at least as often as a drifted
    heuristic, and the message prints both configs rather than a verdict about which is at fault.
    """
    _skip_unless_this_world(mesh)
    # `apply_mesh` is a FACTORY fixture -- it must be CALLED with a mesh spec, exactly as the two
    # timing tests above do. Handing the factory itself to `dm_of` made this test raise
    # `TypeError: apply_mesh did not yield a DistributedManager; got function` on every cell it ever
    # selected, so the one check a timing gate structurally cannot make has never actually run.
    # Measured on 6 selected cp8 cells, 6 failed, all with that TypeError.
    dm = apply_mesh(_MESH_SPEC[mesh])
    oom_reason = None  # reduced AFTER the try; see the handler below
    try:
        elected, heuristic = _configs_elected_and_heuristic(mesh, N_token, D, direction, dm_of(dm))
    except BaseException as exc:  # capacity is recorded; anything else re-raises
        # Recorded and reduced AFTER the try, never skipped from inside the handler: `gated_skip`
        # all-reduces, so reaching it only on the ranks that raised strands them in a collective
        # their peers never enter. `is_capacity_error` covers the symmetric heap too.
        if not is_capacity_error(exc):
            raise
        oom_reason = f"OOM building the cell twice mesh={mesh} N={N_token} D={D}"
    gated_skip(oom_reason)

    agree = {half: elected[half] == heuristic[half] for half in ("front", "back")}
    if all(agree.values()):
        return

    # A disagreement is a defect only if the autotuner's win was DECISIVE. `margin["rel"]` is the
    # fraction by which the runner-up is slower, measured during the sweep itself -- so a coin-flip
    # reads ~0.00 and a real win reads ~0.3. Below the bar the two are configs this venue cannot
    # separate, which is not something a heuristic can be asked to get right.
    margins = _freeze_margin(mesh, N_token, D)
    verdict = []
    for half, ok in agree.items():
        if ok:
            continue
        rel = margins.get(half)
        if rel is not None and rel <= _TIE_REL:
            continue
        verdict.append(
            f"{half}: heuristic picked {heuristic[half]}, autotuner elected {elected[half]}; "
            f"runner-up margin rel={rel!r} (tie bar {_TIE_REL}) -- "
            + (
                "no margin recorded, so the win cannot be shown to be a tie"
                if rel is None
                else f"the autotuner's win is decisive at {rel:.3f}"
            )
        )
    assert not verdict, (
        f"the shipped heuristic disagrees with the autotuner at mesh={mesh} "
        f"N_token={N_token} D={D}:\n  " + "\n  ".join(verdict)
    )


#: `C` solved from the REAL cp8 cells, with the error bar each carries. Only the first two constrain
#: it: `C` is a DIFFERENCE of two large numbers, so its precision degrades with cell size, and the
#: 11.3/20.9 ms entries are consistent with almost anything. Fitting a trend through all six -- which
#: I did once, R^2 0.865 -- is fitting noise.
_C_FROM_REAL_CELLS_US = ((368, 17), (287, 39), (218, 72), (247, 78), (115, 102), (-367, 185))


@matrix_exempt(
    "the subject is the TIMER's bias term, not a kernel: one cheap cell is measured to characterise "
    "`C` for the callable, and sweeping shapes is precisely what it exists to make unnecessary"
)
@numeric_exempt("asserts a timing-model parameter against its own error bars, not a tensor")
def test_the_workflow_probe_reproduces_the_C_solved_on_the_REAL_cells(apply_mesh):
    """G2: a cheap-shape probe of the SAME callable must give the same `C` as the full-size cells.

    `C` is the host cost of dispatching `fn()` once, and `fn` here is a DTensor round trip that runs
    the same number of PYTHON operations regardless of tensor extent. If that reasoning is right, the
    workflow at its smallest legal shape yields the same `C` as the workflow at N=4096 -- and one
    cheap probe per process then replaces the second timing point this gate currently pays on EVERY
    cell (`43 x (1+4)` launches -> `43 x 1`).

    PASS is agreement with the cells that actually CONSTRAIN `C`: the two smallest, 368 +- 17 us and
    287 +- 39 us, inverse-variance weighted to 335 us. The larger cells carry +-100..185 us and are
    excluded deliberately -- including them would let a wrong probe pass by hiding inside their error
    bars, which is the same mistake as fitting a trend through them.

    A FAILURE here is a real result, not a broken test: it would mean `C` depends on something the
    cheap shape does not reproduce, and the per-cell two-point solve stays. Either way the number is
    recorded.
    """
    _skip_unless_this_world("cp8")
    dm = apply_mesh(_MESH_SPEC["cp8"])
    c_ms = measure_workflow_dispatch_cost("cp8", 256, "outgoing", dm)
    c_us = c_ms * 1000.0
    constraining = _C_FROM_REAL_CELLS_US[:2]
    lo = min(v - 2 * e for v, e in constraining)
    hi = max(v + 2 * e for v, e in constraining)
    print(
        f"\nPROBE C = {c_us:.1f} us; constraining cells {constraining}; 2-sigma window [{lo}, {hi}]"
    )
    assert lo <= c_us <= hi, (
        f"the cheap-shape probe gives C = {c_us:.1f} us, outside the [{lo}, {hi}] us window set by "
        f"the two cells that constrain C ({constraining}). `C` therefore depends on something the "
        f"small shape does not reproduce, and the per-cell two-point solve must stay."
    )


@matrix_exempt(
    "characterises the PROBE itself -- its shape is fixed by `_PROBE_D`/`_PROBE_N_PER_CP`, so there "
    "is nothing for a matrix to sweep; the point is that the probe is REPRODUCIBLE, not that it "
    "covers a grid"
)
@numeric_exempt("asserts a timing-model parameter against its reference, not a tensor")
def test_the_probe_C_is_reproducible_and_matches_the_cells_that_constrain_it(apply_mesh):
    """G2/J2: the probe must return the same `C` every time, and the right one.

    TWO ASSERTIONS, because either alone passes a broken probe. A probe that returns a stable wrong
    number passes reproducibility; one that lands near the reference by luck passes accuracy. The
    first version of this test checked accuracy at ONE configuration, called the design validated,
    and a 6x error then shipped into the gate.

    Reference: `C` solved on the full-size cp8 cells, 368 +- 17 and 287 +- 39 us, inverse-variance
    weighted to 335 us. Only those two constrain it -- the larger cells carry +-72..185 us, and
    admitting them would let a wrong probe hide inside their bars.

    Measured at the fixed probe shape across independent runs: 330.7 / 331.4 / 331.1 us. The shape is
    fixed precisely BECAUSE the sweep that produced those numbers also found two regimes where the
    solve returns a plausible float rather than an error -- N_token=512 reads 70 us (host-bound,
    device time below `C`) and N_token=4096 reads 294..465 us (device time dwarfs `C`, so the
    difference is noise). Neither is an error condition the solve can see from the inside, which is
    why the shape is a constant and not a parameter.
    """
    _skip_unless_this_world("cp8")
    dm = apply_mesh(_MESH_SPEC["cp8"])
    runs = [measure_workflow_dispatch_cost("cp8", dm=dm) * 1000 for _ in range(3)]
    print(
        f"\nPROBE C (3 runs, fixed shape D={_PROBE_D} N={_PROBE_N_PER_CP * 8}): "
        f"{['%.1f' % v for v in runs]} us",
        flush=True,
    )
    spread = (max(runs) - min(runs)) / min(runs)
    assert spread <= 0.10, (
        f"probe `C` is not reproducible: {runs} us, {spread * 100:.1f}% spread. A `C` that moves "
        f"between measurements is subtracted from every cell and cannot be told from a real change."
    )
    c = sum(runs) / len(runs)
    lo, hi = 368 - 2 * 17, 287 + 2 * 39
    lo, hi = min(lo, 287 - 2 * 39), max(hi, 368 + 2 * 17)
    assert lo <= c <= hi, (
        f"probe `C` = {c:.1f} us is outside the [{lo}, {hi}] us window set by the two cells that "
        f"constrain `C` (368 +- 17, 287 +- 39). The probe is not measuring the same quantity the "
        f"full-size cells do, so the per-cell two-point solve must stay."
    )
