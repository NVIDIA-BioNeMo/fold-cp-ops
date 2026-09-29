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

"""Torchrun-based pytest harness for fold-cp-ops distributed tests (T0.2).

Pattern (Megatron-LM / Megatron-core): pytest is invoked *under* torchrun::

    torchrun --nproc_per_node=N -m pytest -q tests/distributed/<file>

Every torchrun rank is its own pytest process. Each TEST brings up its own
``torch.distributed`` process group and its own device mesh through the
``dist_manager`` fixture, and releases both afterwards, so no test inherits
distributed state from another. Unlike the reference baseline's ``spawn_multiprocessing``, the
PROCESSES are not re-spawned per test — only the group is — so isolation costs a
rendezvous per test rather than a process launch per test.

Neutralizing pytest's single-process side effects under torchrun
----------------------------------------------------------------
pytest assumes a single process; under torchrun N identical pytest processes
collect and run the same items in lockstep. This conftest:

* reads ``RANK`` / ``WORLD_SIZE`` / ``LOCAL_RANK`` from the torchrun env and
  pins ``torch.cuda.set_device(LOCAL_RANK)`` BEFORE any process-group init
  (so each rank lands on a distinct GPU);
* initializes the process group per test (``dist_manager``), each with its own
  rendezvous port and with torchrun's agent store dropped — the two things that
  make a second rendezvous in one process work;
* exposes ``rank`` / ``world_size`` / ``local_rank`` / ``device`` /
  ``device_mesh`` fixtures and ``rank_zero_first`` / ``run_on_rank`` helpers so
  rank-aware tests stay readable;
* installs a ``pytest_runtest_teardown`` barrier so ranks cannot drift between
  tests (a fast rank starting test N+1 while a slow rank is still in test N
  would desync collectives);
* tears the group down after each test, behind a barrier.

The session builds the WORLD GROUP ONLY. The mesh config (names + shape) is a
PARAMETRIZED AXIS: each test module declares its pool in a ``KernelMatrix`` and
each test applies one spec through the ``apply_mesh`` fixture, which rebuilds the
grid on the live world group. There is no ``--dist-mesh`` flag and no
``CPO_DIST_MESH`` env var, deliberately — when the mesh comes from the command
line, "which configurations did this run cover" is answerable only by reading the
launch, and a mesh nobody considered looks exactly like one considered and
excluded.

WORLD_SIZE is still a launch property — one torchrun/srun per WORLD SIZE, not per
mesh — and each run covers every declared spec whose rank product matches,
skipping the others with a reason.

The world process group itself CAN be torn down and rebuilt in-process, but only
with a fresh ``MASTER_PORT`` per cycle and ``TORCHELASTIC_USE_AGENT_STORE``
dropped; reusing torchrun's agent store on the same port desynchronizes the ranks
(one raises, the rest hang). Rebuilding only the GRID needs none of that, which is
why ``apply_mesh`` rebuilds the grid rather than the group.

Fallback: for one-off mesh configs that don't fit "run everything in one
group", the upstream ``spawn_multiprocessing`` launcher (the CP fork's per-test
spawn) remains available and is documented in ``README.md``.
"""

import gc
import os
import warnings
from datetime import timedelta
from collections import OrderedDict

import pytest

# ---------------------------------------------------------------------------
# Wedge auto-dump: if any rank's process HANGS (an in-kernel nvshmem A2A
# deadlock, a collective that never returns, a runtime hang at a large shape),
# dump EVERY thread's Python traceback so the wedged host-side handshake names
# itself -- no py-spy, no scancel (the sbatch -t / bash `timeout` then kills
# clean). install_wedge_watchdog dumps from a Python thread every
# CPO_FAULTHANDLER_TIMEOUT seconds (default 300; <=0 disables) and REPEATs, so a
# drain/collective stuck for > timeout self-reports to stderr -> the .out.
# `kill -USR1 <pid>` also dumps on demand. Pure diagnostic: no behavior change
# on a passing run. (Added for the N4096 e2e runtime-hang isolation.)
#
# It dumps from a PYTHON thread, holding the GIL, deliberately: the previous
# faulthandler.dump_traceback_later() dumps from a C thread that does not hold
# the GIL and so races the frame chain of every running thread, segfaulting
# HEALTHY sessions. See tests/wedge_watchdog.py for the measured failure.
#
# escalate_after=2 is NOT boilerplate, and this file needs it where tests/perf
# does not. Measured: a rank parked in cudaStreamSynchronize behind a hung
# in-kernel A2A leaves the GIL FREE, so the Python thread reports it (87 of 88
# nominal dumps still landed), and NCCL likewise. But nvshmem.pyx wraps
# `barrier` / `barrier_on_stream` `with nogil:` and does NOT wrap `barrier_all`
# / `barrier_all_on_stream` -- those hold the GIL for the whole call, and
# `barrier_all` is what cleanup() calls. A rank hung in a cross-node
# barrier_all is therefore a wedge NO Python thread can ever report.
#
# So a dead-man timer backstops it: ITIMER_REAL armed for 2 ticks and re-armed
# on every tick, so SIGALRM fires only once Python has stopped being scheduled
# at all. That dump IS the racy signal-context one -- accepted deliberately,
# because by then the process is wedged and headed for the sbatch/`timeout`
# kill anyway. This is what covers the unattended cross-node case, where
# SIGUSR1 has no operator to send it.
#
# NOTE: this claims SIGALRM/ITIMER_REAL process-wide. Do not enable pytest-timeout's
# `signal` method in this directory; one of the two would silently lose.
#
# SIGUSR1 stays registered for on-demand dumps in an attended session.
import signal as _signal

from tests.wedge_watchdog import install_wedge_watchdog
from fold_cp_ops.testing.collective_guard import pre_init_skip, rank_invariant_skip
from fold_cp_ops.testing.coverage_ledger import record_exercised, record_skipped

# The dumps go to a FILE, not to stderr, and that is the whole point of this block.
#
# `wedge_watchdog`'s own docstring records the limitation: under pytest's default `fd` capture the
# dumps land in the CAPTURED stderr of whichever test is running and are shown "only if that test
# fails". **A wedged test never fails -- it never completes** -- so on the SIGKILL that ends a hung
# rank the captured buffer is discarded and every dump goes with it. The instrument is therefore
# blind in exactly the scenario it exists for, and it was written down as a note rather than treated
# as a defect.
#
# Measured, which is why this is code and not a comment: a 13-minute two-rank wedge run with
# `CPO_FAULTHANDLER_TIMEOUT=45` produced **0 dump markers** in its log. At that period it should have
# emitted ~17. The mechanism was armed and firing; the output was simply swallowed.
#
# A real file bypasses capture entirely, survives the kill, and is the same bargain `run_progress.txt`
# already makes for the crash case. One file per PID so concurrent ranks never interleave, and the
# path is PRINTED at import so it is discoverable without reading this comment.
#
# **`CPO_WEDGE_DIR` exists because `TMPDIR` is NODE-LOCAL on a real cluster.** Measured on a multi-node cluster:
# a compute node's `/tmp` is `/dev/md0`, its own root filesystem -- so on a 2-node run each rank's
# dump lands on a different machine and reading them means an `srun` onto each node, which is exactly
# the friction that stops a diagnostic being read. Point `CPO_WEDGE_DIR` at shared storage in the
# LAUNCHER and both ranks' files land side by side.
#
# It is an env var and not a path in this file on purpose: which filesystem is shared is a property
# of the cluster, not of the repo, and the repo does not hardcode an absolute base. Same reasoning
# that keeps NIC/HCA selection in the launcher.
import pathlib as _pathlib

_wedge_user = os.environ.get("USER", "x")
_wedge_base = os.environ.get("CPO_WEDGE_DIR") or os.environ.get("TMPDIR", "/tmp")
_wedge_dir = _pathlib.Path(_wedge_base) / f"cpo-wedge-of-{_wedge_user}"
_wedge_dir.mkdir(parents=True, exist_ok=True)
_wedge_rank = os.environ.get("RANK", "0")
_wedge_path = _wedge_dir / f"wedge_rank{_wedge_rank}_pid{os.getpid()}.txt"
# Line-buffered: a dump that reaches the OS only at flush is a dump a SIGKILL still eats.
_wedge_file = open(_wedge_path, "w", buffering=1)  # noqa: SIM115 -- lives for the session by design
print(f"wedge dumps (survive a SIGKILL; empty on a healthy run): {_wedge_path}", flush=True)
install_wedge_watchdog(escalate_after=2, file=_wedge_file)
if hasattr(_signal, "SIGUSR1"):
    import faulthandler as _faulthandler

    _faulthandler.register(_signal.SIGUSR1, all_threads=True)

# --------------------------------------------------------------------------- #
# Lazy import of the distributed manager.
#
# fold_cp_ops.distributed.DistributedManager is being built in parallel (T0.1). To
# avoid hard-crashing collection before it lands, import it lazily inside the
# session fixture and skip the whole session with a clear message if it (or
# nvshmem / multiple GPUs) is unavailable. The API we depend on mirrors the reference baseline's
# DistributedManager: ``.initialize(grid_group_sizes, device_type, backend)``,
# the ``DistributedManager()`` Borg singleton, ``.rank`` / ``.world_size`` /
# ``.local_rank`` / ``.device`` / ``.device_mesh`` / ``.device_mesh_subgroups``,
# and ``.cleanup()``.
# --------------------------------------------------------------------------- #


def _import_distributed_manager():
    """Return the DistributedManager class, or ``None`` if unavailable.

    Upstream this preferred its own manager but FELL BACK to
    the CP fork's ``distributed.manager.DistributedManager`` (the reference implementation whose API it
    mirrors), so the harness was testable before the port landed. That fallback is deliberately
    NOT carried over: the port has landed, and a fallback here means an import error in
    ``fold_cp_ops.distributed`` silently reroutes every DistributedManager test onto a
    third-party class — they would pass while proving nothing about this repo. Returning
    ``None`` makes the gate SKIP visibly instead.
    """
    try:
        from fold_cp_ops.distributed import DistributedManager  # type: ignore

        return DistributedManager
    except (ImportError, AttributeError):
        return None


# --------------------------------------------------------------------------- #
# CLI / env: choose the (names, shape) device-mesh for this session.
# --------------------------------------------------------------------------- #


def pytest_addoption(parser):
    """Add distributed-mesh options.

    Uses a ``dist`` option group with names distinct from the top-level
    ``tests/conftest.py`` (which owns ``--compile-only``) so there is no
    option collision when both conftests load.
    """
    group = parser.getgroup("distributed", "fold-cp-ops distributed (torchrun) tests")
    # NOTE: there is deliberately no --dist-mesh / CPO_DIST_MESH knob. The device mesh is declared
    # in each test module's KernelMatrix and applied per test by `apply_mesh`, so the set of
    # configurations a run covered is a property of the SOURCE rather than of the command line.
    group.addoption(
        "--dist-backend",
        action="store",
        default=None,
        help="torch.distributed backend ('nccl' for cuda, 'gloo' for cpu). "
        "Default: auto by device type.",
    )
    group.addoption(
        "--dist-device-type",
        action="store",
        default="cuda",
        help="Device type for the mesh: 'cuda' (default) or 'cpu'.",
    )


def _mesh_numel(grid_group_sizes) -> int:
    """Ranks a mesh spec needs: the product of every group size, subgrids flattened.

    Args:
        grid_group_sizes: Mapping of group name -> ``int`` or ``tuple[int, ...]``.

    Returns:
        The rank count. ``apply_mesh`` compares it against WORLD_SIZE and skips rather than fails on
        a mismatch, because one declared pool serves 2-, 4- and 8-rank launches.
    """
    from math import prod

    total = 1
    for v in grid_group_sizes.values():
        total *= prod(v) if isinstance(v, tuple) else v
    return total


# --------------------------------------------------------------------------- #
# Torchrun env helpers.
# --------------------------------------------------------------------------- #


def _launcher_vars() -> tuple[str, str, str]:
    """Names of the (rank, world size, local rank) variables for THIS launch, in one place.

    Asks ``DistributedManager._detect_launcher()`` which namespace owns this process rather than
    guessing from which variables happen to be set. That matters BEFORE ``initialize()``: under a
    bare ``srun`` the ``env://`` names are simply absent until the manager exports them, so a
    helper that only reads ``RANK`` would report "no launcher" and every test in this directory
    would skip while the run exited green.

    Only the three IDENTITY variables are mapped here. The rendezvous address and port are NOT --
    deriving those lives solely in ``_initialize_slurm``, and a second copy of that mapping is the
    "two things, one name" failure this harness must not reintroduce.
    """
    _dm = _import_distributed_manager()
    if _dm is not None and _dm._detect_launcher() == "SLURM":
        return ("SLURM_PROCID", "SLURM_NTASKS", "SLURM_LOCALID")
    return ("RANK", "WORLD_SIZE", "LOCAL_RANK")


def _is_under_torchrun() -> bool:
    """True when SOME launcher provided this process's identity (torchrun or a bare srun)."""
    _dm = _import_distributed_manager()
    if _dm is None:  # the manager itself is unimportable -- the gate below skips visibly
        return "RANK" in os.environ and "WORLD_SIZE" in os.environ
    return _dm._detect_launcher() is not None


def _env_rank() -> int:
    return int(os.environ.get(_launcher_vars()[0], "0"))


def _env_world_size() -> int:
    ws, _ = _launcher_vars()[1], None
    return int(os.environ.get(ws) or os.environ.get("SLURM_NPROCS", "1"))


def _env_local_rank() -> int:
    # torchrun always sets LOCAL_RANK; a bare srun uses SLURM_LOCALID; else fall back to the rank.
    lr = _launcher_vars()[2]
    return int(os.environ.get(lr, os.environ.get(_launcher_vars()[0], "0")))


# --------------------------------------------------------------------------- #
# Session-scoped distributed setup — init ONCE, reused by every test.
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="session")
def _rendezvous_store():
    """ONE ``TCPStore`` for the whole session, handed to every test's ``init_process_group``.

    Passing ``store=`` bypasses ``env://`` entirely: no rendezvous handler, no per-test bind, no
    port to rotate, and nothing for ``destroy_process_group`` to tear down -- the store is owned by
    this fixture and outlives every group built on it. That removes the rebind race instead of
    timing around it.

    Each test wraps this in a ``PrefixStore`` so successive groups cannot collide on keys.
    """
    import torch.distributed as dist

    port = int(os.environ.get("MASTER_PORT", "29500")) + 1  # NOT the agent's own port
    yield dist.TCPStore(
        host_name=os.environ.get("MASTER_ADDR", "127.0.0.1"),
        port=port,
        world_size=_env_world_size(),
        is_master=(_env_rank() == 0),
        timeout=timedelta(seconds=300),
        multi_tenant=True,
    )


def _pg_options_workaround(backend: str, world_size: int):
    """Options that make a SECOND ``init_process_group`` in one process survive making a subgroup.

    THIS IS REQUIRED, NOT AN OPTIMIZATION. Without it the second test that builds a mesh with a real
    subgroup SEGFAULTS on every rank -- a NULL dereference inside ``ProcessGroupNCCL::split``.

    Root cause, in torch's own source (identical in v2.11.0 and v2.13.0, and duplicated verbatim in
    ProcessGroupGloo)::

        static std::atomic<size_t> process_group_id = 0;   // per-process, only ever increments
        ...  local_id_(process_group_id++)

        const std::vector<uint64_t>& ProcessGroupNCCL::groupRanks() const {
          if (options_->global_ranks_in_group.empty() && local_id_ == 0) {
            static std::vector<uint64_t> globalRanks(size_);   // identity fallback
            std::iota(globalRanks.begin(), globalRanks.end(), 0);
            return globalRanks;
          }
          return options_->global_ranks_in_group;             // <- EMPTY for a fresh world group
        }

    ``split()`` then does ``globalRanksInGroup.emplace_back(groupRanks()[rank])``, and indexing an
    empty vector reads ``data() == nullptr``. The ``local_id_ == 0`` guard uses "first process group
    in this process" as a proxy for "the default group", which stops being true the moment the
    default group is destroyed and re-created. So the fallback fires for the first world group and
    never again.

    Setting ``global_ranks_in_group`` explicitly makes ``groupRanks()`` return it directly and never
    consult the guard. For a world group the correct value is exactly what the fallback would have
    synthesised: ``range(world_size)``.

    **Scoped to the tests on purpose.** Production initializes once per process and never trips this;
    only a per-test lifecycle does. Rather than carry an upstream workaround inside
    ``DistributedManager``, it is passed in through ``initialize(**kwargs_init_pg)``, which forwards
    to ``init_process_group``.

    Args:
        backend: ``"nccl"`` or ``"gloo"``. Selects the Options CLASS; the field itself lives on the
            base ``c10d::Backend::Options``, so both backends need and accept it.
        world_size: Rank count; the group's ranks are ``range(world_size)``.

    Returns:
        A backend-appropriate ``Options``, or ``None`` for a backend with no Options type, in which
        case a second subgroup-creating test is expected to fail.
    """
    import torch.distributed as dist

    cls = {
        "nccl": getattr(dist, "ProcessGroupNCCL", None),
        "gloo": getattr(dist, "ProcessGroupGloo", None),
    }.get(backend)
    if cls is None or not hasattr(cls, "Options"):
        return None
    opts = cls.Options()
    opts.global_ranks_in_group = list(range(world_size))
    return opts


@pytest.fixture
def dist_manager_recycled(request, _rendezvous_store):
    """OPT-IN: bring up torch.distributed FOR THIS TEST and tear it down after it.

    **Not used by default, and NOT mixable with** ``dist_manager`` **in one session** -- the manager
    is a Borg singleton, so a per-test teardown would pull the group out from under a session-scoped
    user. Request one or the other, never both.

    Costs 3.5-5x the wall clock of the session-scoped fixture (measured: world 4, 33s -> 114s; world
    8, ~27s -> 138s), because every test pays a full ``init_process_group`` + NCCL communicator
    setup + ``destroy_process_group``. The gap WIDENS with rank count, since communicator setup
    scales with it. Use it for tests whose subject IS the DM lifecycle.

    Each test is an independent program: it initializes its own process group through the
    DistributedManager, builds whatever mesh it needs, and releases the group afterwards. Nothing
    distributed survives between tests, so no test can be made to pass -- or fail -- by state an
    earlier one left behind.

    nvshmem is the one exception, and it is a hard one: it CANNOT be re-initialized (torch guards its
    bootstrap with a static bool nothing resets), so it comes up on the first test that needs it and
    stays up. ``cleanup()`` deliberately does not touch it; the finalize is an ``atexit`` hook the
    manager registers itself.

    Three things are required to re-initialize in one process, and all three are load-bearing --
    dropping any one makes the SECOND test hang or segfault:

    * ``TORCHELASTIC_USE_AGENT_STORE`` dropped and a ``MASTER_PORT`` of our own, so each test stands
      up its own rendezvous instead of re-attaching to torchrun's agent store -- measured: keeping
      the agent store HANGS on the second test, and a reused port fails under a full suite;
    * ``pg_options`` carrying ``global_ranks_in_group`` -- see ``_pg_options_workaround``.

    Skips (with a reason) when not launched under torchrun/srun, when the DistributedManager import
    is unavailable, or when there are fewer local CUDA devices than ranks.
    """
    DistributedManager = _import_distributed_manager()
    if DistributedManager is None:
        pre_init_skip(
            "DistributedManager unavailable: fold_cp_ops.distributed (T0.1, in "
            "progress) nor the CP fork's distributed.manager importable.",
            because="import probe inside dist_manager_recycled, before initialize(); an import can "
            "fail on one host only, and there is no group yet to reduce the decision over",
        )
    # No env:// pre-population here any more. The helpers below ask DistributedManager which
    # launcher this is and read that launcher's own variables, so a bare `srun` needs nothing to be
    # written into os.environ before initialize() runs.
    if not _is_under_torchrun():
        rank_invariant_skip(
            "Not launched under torchrun or srun (RANK/WORLD_SIZE unset, and no SLURM task env). Run "
            "e.g.: torchrun --nproc_per_node=2 -m pytest tests/distributed/... OR srun "
            "--ntasks-per-node=8 -N2 --mpi=pmix python -m pytest tests/distributed/...",
            because="a launcher either set RANK/WORLD_SIZE in every task's env or in none; there is "
            "no launch in which one rank is under torchrun and another is not",
        )

    device_type = request.config.getoption("--dist-device-type")
    world_size = _env_world_size()
    local_rank = _env_local_rank()

    import torch

    if device_type == "cuda":
        if not torch.cuda.is_available():
            pre_init_skip(
                "device-type=cuda but torch.cuda.is_available() is False",
                because="runs inside dist_manager_recycled before initialize(); CUDA can be absent "
                "on ONE host (a bad driver, a claimed GPU) so this is genuinely divergent, and "
                "there is no group yet to reduce it over: reducing it needs a vote channel that "
                "predates the process group, which does not exist yet",
            )
        n_dev = torch.cuda.device_count()
        # Multi-node: each node hosts LOCAL_WORLD_SIZE ranks (= nproc_per_node), NOT the full
        # WORLD_SIZE, so gate on the per-node rank count (torchrun sets LOCAL_WORLD_SIZE; fall back
        # to world_size for a single-node run -> unchanged there). Gating on world_size wrongly
        # skipped every multi-node run ("need >= 16 CUDA devices, have 8").
        local_ws = int(os.environ.get("LOCAL_WORLD_SIZE", str(world_size)))
        if n_dev < local_ws:
            pre_init_skip(
                f"need >= {local_ws} local CUDA devices (LOCAL_WORLD_SIZE), have {n_dev}",
                because="runs inside dist_manager_recycled before initialize(); device COUNT differs "
                "per host on a heterogeneous job, and there is no group yet to reduce it over",
            )
        # Pin the device BEFORE init_process_group so NCCL binds each rank to a
        # distinct GPU (LOCAL_RANK indexes into CUDA_VISIBLE_DEVICES).
        torch.cuda.set_device(local_rank)

    backend = request.config.getoption("--dist-backend")

    # ENV init method: torchrun exports RANK/WORLD_SIZE/LOCAL_RANK/MASTER_*.
    os.environ.setdefault("CPO_DISTRIBUTED_INIT_METHOD", "ENV")
    # This test's own key namespace inside the shared store.
    import torch.distributed as dist

    _store = dist.PrefixStore(f"cpo/{request.node.nodeid}/", _rendezvous_store)

    # NO grid here -- the world group only. The device mesh is a PARAMETRIZED AXIS supplied per test
    # by the ``mesh`` fixture from a declared KernelMatrix, not a session-wide choice made by the
    # launcher. Building it here would pin the whole session to one configuration and make "which
    # meshes were covered" a property of the command line, which is precisely the state where a mesh
    # nobody considered is indistinguishable from one considered and excluded.
    # pg_options is REQUIRED for the second and later tests -- see _pg_options_workaround.
    _opts = _pg_options_workaround(
        backend or ("gloo" if device_type == "cpu" else "nccl"), world_size
    )
    DistributedManager.initialize(
        None,
        device_type=device_type,
        backend=backend,
        store=_store,
        **({"pg_options": _opts} if _opts else {}),
    )

    # initialize() SWALLOWS a failed init_process_group: it warns and "default initializes", leaving
    # _initialized True with no process group. Under torchrun that is how a rendezvous failure turns
    # into a HANG rather than an error -- the rank that failed sails on and its peers block forever
    # waiting for it. Convert it into a failure here, on the rank that actually failed.
    if not torch.distributed.is_initialized():
        raise RuntimeError(
            f"DistributedManager.initialize() returned without a process group for "
            f"{request.node.nodeid} (MASTER_ADDR={os.environ.get('MASTER_ADDR')} "
            f"MASTER_PORT={os.environ.get('MASTER_PORT')}). initialize() swallows the underlying "
            "error and default-initializes; the warning above this line carries the real cause."
        )
    manager = DistributedManager()

    yield manager

    # Ordinary fixture teardown: cleanup() barriers and destroys the process group, and does NOT
    # touch nvshmem -- init_nvshmem registers its own atexit hook for that, so the one-way-door
    # finalize happens exactly once, at exit, whether or not anyone calls cleanup(). That is what
    # makes cleanup() safe to call per test.
    DistributedManager.cleanup()
    gc.collect()  # release the TCPStore so the next test rebinds a FRESH daemon, not the dying one


@pytest.fixture(scope="session")
def dist_manager(request):
    """Bring up torch.distributed ONCE for the session. The default.

    No ``pg_options`` workaround either: this initializes ONCE, so it is the first process group in
    the process and torch's own identity fallback applies. That workaround belongs only to the
    recycled fixture.

    Uses the launcher's own ``env://`` rendezvous -- one group, one rendezvous, nothing to recycle,
    so none of the store/port machinery :func:`dist_manager_recycled` needs applies here.

    Same bring-up otherwise, but the group is built once and shared, because
    the per-test lifecycle costs 3.5-5x wall clock and almost no test's subject is the lifecycle
    itself. Per-test ISOLATION of what matters is still there: each test rebuilds its own device
    mesh and groups through ``apply_mesh``.
    """
    DistributedManager = _import_distributed_manager()
    if DistributedManager is None:
        pre_init_skip(
            "DistributedManager unavailable",
            because="import probe inside dist_manager, before initialize(); per-process failure "
            "with no group yet to reduce over",
        )
    if not _is_under_torchrun():
        rank_invariant_skip(
            "Not launched under torchrun or srun (RANK/WORLD_SIZE unset)",
            because="the launcher sets RANK/WORLD_SIZE in every task's env or in none",
        )

    import torch
    import torch.distributed as dist

    device_type = request.config.getoption("--dist-device-type")
    backend = request.config.getoption("--dist-backend")
    world_size, local_rank = _env_world_size(), _env_local_rank()
    if device_type == "cuda":
        if not torch.cuda.is_available():
            pre_init_skip(
                "device-type=cuda but torch.cuda.is_available() is False",
                because="runs inside dist_manager before initialize(); CUDA can be absent on ONE "
                "host, and there is no group yet to reduce the decision over",
            )
        local_ws = int(os.environ.get("LOCAL_WORLD_SIZE", str(world_size)))
        if torch.cuda.device_count() < local_ws:
            pre_init_skip(
                f"need >= {local_ws} local CUDA devices, have {torch.cuda.device_count()}",
                because="runs inside dist_manager before initialize(); device COUNT differs per "
                "host on a heterogeneous job, with no group yet to reduce it over",
            )
        torch.cuda.set_device(local_rank)

    os.environ.setdefault("CPO_DISTRIBUTED_INIT_METHOD", "ENV")
    DistributedManager.initialize(
        None,
        device_type=device_type,
        backend=backend,
    )
    if not dist.is_initialized():
        raise RuntimeError(
            "DistributedManager.initialize() returned without a process group; it swallows the "
            "underlying error and default-initializes, so the warning above carries the cause."
        )
    yield DistributedManager()
    DistributedManager.cleanup()


# Function-scoped like everything downstream of dist_manager: the manager is per-test now, so a
# session-cached probe would outlive the state it describes. The probe is env+sysfs only, so
# repeating it per test is cheap.
@pytest.fixture
def topology(dist_manager):
    """What this run's hardware and launcher actually are, as the DistributedManager sees them.

    Purpose
        Give tests one place to ask "can this machine run this case?" so a cell that needs two nodes
        or an IB fabric skips with a reason instead of failing on hardware that was never going to
        satisfy it.

    Semantics
        Wraps ``DistributedManager._detect_nvshmem_traits()`` -- the manager's own host-only probe
        (env + sysfs; no nvshmem or CUDA-kernel calls) -- so the test suite and the nvshmem profile
        selection read the SAME topology rather than two hand-rolled opinions that can disagree.
        Adds three things the traits do not carry:

        * ``n_local_gpus``  -- ``torch.cuda.device_count()``, what this box can see.
        * ``n_nodes``       -- ``world_size // local_world_size``.
        * ``launch``        -- ``"torchrun"`` or ``"srun"``, discriminated by ``TORCHELASTIC_RUN_ID``
          rather than by ``method_init``: a bare ``srun`` is ALSO reported as ``"ENV"``, because
          ``_initialize_slurm`` reads ``SLURM_PROCID``/``NTASKS``/``LOCALID`` and ``_setup`` then
          exports the same ``env://`` names. Reading ``method_init`` would now distinguish them, but
          ``TORCHELASTIC_RUN_ID`` remains the direct question and needs no manager.

        Keys from the traits include ``single_node``, ``ib_present``, ``n_ib_devices``,
        ``mixed_fabric``, ``peermem``, ``nic_curated``, ``world_size``, ``local_world_size``.

    Input requirements
        Requires an initialized manager (the ``dist_manager`` fixture). Session-scoped: none of
        these can change inside a run, and re-probing sysfs per test would be pure cost.

    Returns
        A dict. Read-only by convention -- mutating it would silently change later tests' skips.
    """
    import torch

    DistributedManager = _import_distributed_manager()
    traits = dict(DistributedManager._detect_nvshmem_traits())
    lws = traits.get("local_world_size") or 1
    traits["n_local_gpus"] = torch.cuda.device_count() if torch.cuda.is_available() else 0
    traits["n_nodes"] = max(1, traits.get("world_size", 1) // lws)
    traits["launch"] = "torchrun" if os.environ.get("TORCHELASTIC_RUN_ID") else "srun"
    return traits


#: Matrix ``kernel`` name the mesh axis is recorded under. A constant rather than a read off the
#: test module, because ``apply_mesh`` is shared by every distributed module and the ledger must
#: union across them -- two spellings would produce two half-empty ledgers that never merge.
_MESH_KERNEL = "distributed_manager"


@pytest.fixture
def apply_mesh(dist_manager, world_size):
    """Factory: build one device-mesh configuration on the live world group, for this test only.

    Purpose
        Make the device mesh a PARAMETRIZED AXIS rather than a launch-time choice. Tests draw mesh
        specs from a declared ``KernelMatrix`` and hand each one here.

    Semantics
        Returns a callable ``apply(spec) -> DistributedManager``. Per call it builds the mesh and
        per-dim groups from ``spec``, and refreshes the nvshmem PE maps when nvshmem is up. It
        resets the grid first, which is required whenever the world group outlives the test. The world process group and
        nvshmem are untouched. The group COULD be rebuilt per test (fresh ``MASTER_PORT``, no agent
        store) but need not be when only the mesh varies; nvshmem cannot be re-initialized once
        finalized (torch's init guard is a static bool it never resets).

        A spec whose rank product does not equal ``WORLD_SIZE`` is SKIPPED, not failed: the matrix
        pool spans mesh shapes for several world sizes, and only the ones that fit this launch can
        run. That keeps one declared pool usable from a 2-, 4- or 8-rank torchrun.

    Input requirements
        ``spec`` is a tuple of ``(name, size)`` pairs, where ``size`` is an int or a tuple of ints
        (a subgrid, per ``DistributedManager.grid_group_sizes``). A tuple-of-pairs rather than a
        dict because pytest parametrize values must be hashable and produce a stable id.

        COLLECTIVE: ``create_grid_group`` calls ``new_group``, which every rank must reach in the
        same order. Every rank must therefore apply the same specs in the same sequence -- which is
        what parametrizing all ranks from the same matrix guarantees. A rank-conditional skip here
        would hang the others rather than fail them.

    Returns
        The factory. Calling it returns the ``DistributedManager`` singleton with the mesh built.
    """
    DistributedManager = _import_distributed_manager()

    def apply(spec):
        grid = OrderedDict(spec)
        numel = _mesh_numel(grid)
        # Record what this LAUNCH did with the spec, exercised or declined. One launch fixes one
        # WORLD_SIZE, so no single run can cover a pool that spans several -- the ledger is what
        # lets the union across runs answer that, without failing any run for its own topology.
        # See fold_cp_ops/testing/coverage_ledger.py.
        if numel != world_size:
            record_skipped(
                _MESH_KERNEL,
                "mesh",
                spec,
                f"needs {numel} ranks; this launch has WORLD_SIZE={world_size}",
            )
        else:
            record_exercised(_MESH_KERNEL, "mesh", spec)
        if numel != world_size:
            # One launch = one WORLD SIZE. A spec that needs a different rank count skips here and is
            # covered by a separate torchrun/srun at that size. Reshaping the mesh WITHIN a world
            # size is supported and is the point of this fixture (see below); reshaping the world is
            # not, because the process group cannot be rebuilt in-process without tripping an
            # upstream NCCL bug, and running a mesh on a SUBSET of ranks cannot be made safe while
            # nvshmem's symmetric allocation is collective over every PE.
            rank_invariant_skip(
                f"mesh {dict(grid)} needs {numel} ranks; WORLD_SIZE={world_size}",
                because="the mesh spec is a module-level constant and WORLD_SIZE is one number for "
                "the whole job, so every rank computes this comparison identically",
            )

        DistributedManager.reset_grid_groups()
        DistributedManager.create_grid_group(grid)
        return DistributedManager()

    yield apply
    # No teardown: dist_manager's cleanup() releases the whole group for this test.


@pytest.fixture
def rank(dist_manager) -> int:
    """Global rank of this process."""
    return dist_manager.rank


@pytest.fixture
def world_size(dist_manager) -> int:
    """Total number of ranks in the process group."""
    return dist_manager.world_size


@pytest.fixture
def local_rank(dist_manager) -> int:
    """Node-local rank (== device index into CUDA_VISIBLE_DEVICES)."""
    return dist_manager.local_rank


@pytest.fixture
def device(dist_manager):
    """``torch.device`` this rank computes on."""
    return dist_manager.device


@pytest.fixture  # function scope: the mesh is per-test now, so a session-cached value would be stale
def device_mesh(dist_manager):
    """The ``DeviceMesh`` currently applied (subgroup mesh if the spec had subgrids).

    Returns ``device_mesh_subgroups`` when the applied spec declared subgrids (e.g. ``cp=(2, 2)`` ->
    a 2-D cp grid exposed as separate mesh dims), else the flat ``device_mesh``. This is the mesh
    tests pass to ``distribute_tensor``.

    Input requirements
        A mesh must already be applied for this test, via the ``apply_mesh`` fixture. The session no
        longer builds one -- the mesh is a parametrized axis -- so requesting this fixture without
        having applied a spec raises with that instruction rather than handing back ``None``, which
        would surface much later as an opaque attribute error inside ``distribute_tensor``.

    Raises
        RuntimeError: If no mesh has been applied for this test.
    """
    sub = getattr(dist_manager, "device_mesh_subgroups", None)
    if getattr(dist_manager, "has_subgroups", False) and sub is not None:
        return sub
    mesh = dist_manager.device_mesh
    if mesh is None:
        raise RuntimeError(
            "no device mesh is applied. The session builds only the world group -- the mesh is a "
            "parametrized axis. Request the `apply_mesh` fixture and call it with a spec drawn from "
            "the module's KernelMatrix, e.g. `@DIST_MESH.parametrize('mesh')` + `apply_mesh(mesh)`."
        )
    return mesh


# --------------------------------------------------------------------------- #
# Rank-aware helpers (keep rank-conditional test logic readable).
# --------------------------------------------------------------------------- #


@pytest.fixture
def rank_zero_first(dist_manager):
    """Context manager: rank 0 runs the body first, others wait, then proceed.

    Use for setup that must happen once (e.g. a rank-0 download / file write)
    before all ranks read it::

        with rank_zero_first():
            if rank == 0:
                prepare_shared_file()
    """
    import contextlib

    import torch

    @contextlib.contextmanager
    def _ctx():
        is_dist = torch.distributed.is_available() and torch.distributed.is_initialized()
        if is_dist and dist_manager.rank != 0:
            torch.distributed.barrier()
        try:
            yield
        finally:
            if is_dist and dist_manager.rank == 0:
                torch.distributed.barrier()

    return _ctx


@pytest.fixture
def run_on_rank(dist_manager):
    """Return ``lambda r: dist_manager.rank == r`` for rank-conditional asserts."""
    return lambda r: dist_manager.rank == r


# --------------------------------------------------------------------------- #
# Inter-test barrier — keep ranks lockstep so collectives never desync.
# --------------------------------------------------------------------------- #


# `tryfirst`, and the ordering is the whole correctness of this hook -- do NOT flip it back.
#
# This was `trylast=True`. What that costs is EXACTLY ONE barrier -- the last one -- and the
# numbers below replace an earlier version of this comment that claimed it "never executed a single
# barrier". That claim was wrong. It was inferred from a 1-passes/2-hangs signature and asserted as
# measured; it had never been measured. It now has been, on 2 ranks over
# `test_distributed_manager.py` (67 teardowns: 20 passed + 47 skipped), by counting how often the
# hook body was entered, got past the guard, and completed the barrier:
#
#     arm         entered   guard_passed   barrier_done
#     tryfirst      67          67             67
#     trylast       67          66             66
#
# Both ranks identical, and under `trylast` the event trace ends on a bare `entered` with no
# `guard_passed` after it -- so the ONE lost barrier is the FINAL test's, not a representative one.
#
# The mechanism, corrected. `teardown_exact(nextitem)` finalizes only the fixtures whose scope has
# ended. For a non-last test that excludes the SESSION-scoped `dist_manager`, so the group is still
# alive and this hook's barrier runs normally -- 66 times. Only the last test's
# `teardown_exact(None)` finalizes session scope, running `DistributedManager.cleanup()` ->
# `destroy_process_group` (COLLECTIVE) before our impl, after which `dist.is_initialized()` is False
# and the guard below returns early.
#
# So the hazard is on the EXIT path: without the final barrier the ranks enter the collective
# `destroy_process_group` unsynchronized. `tryfirst` runs it while the group is still alive, so they
# enter together. A barrier AFTER the collective it protects cannot protect it, and one that runs
# after the group is destroyed cannot even run.
#
# KNOWN GAP, stated rather than papered over: the 2-rank/2-test hang that originally prompted this
# change (rank 1 inside `destroy_process_group`, rank 0 still in its test body) is NOT fully
# explained by a single lost exit barrier, and has not been re-measured since. The old comment
# explained it via "no barriers at all", which the table above refutes. The fix is still correct --
# the final barrier is real and is load-bearing for a clean shutdown -- but do not treat the
# original signature as accounted for.
@pytest.hookimpl(tryfirst=True)
def pytest_runtest_teardown(item, nextitem):
    """Barrier after each distributed test so ranks stay in lockstep, and FAIL if they are not.

    Without this, a rank that finishes test N quickly could enter test N+1 and
    issue a collective while a slow rank is still draining test N — a
    cross-test desync / hang. Only fires when the process group is live (i.e.
    the ``dist_manager`` session fixture actually initialized), so non-torchrun
    / skipped sessions are unaffected.

    **A timeout ends the session rather than proceeding.** The barrier used to be an unbounded
    ``torch.distributed.barrier()`` under ``except Exception: pass``, which is the anti-pattern this
    hook exists to prevent, one level up: a desync was swallowed here and surfaced later as a hang
    whose traceback named an unrelated test N tests further on. `CollectiveGate.barrier` bounds the
    wait and RAISES, and `BarrierTimeout`'s own contract is that the caller must not enter another
    collective afterwards -- and running the next test IS entering another collective. So the only
    action consistent with that contract is to stop, naming the test that diverged.

    Stopping is safe precisely because the bound is large (``CPO_BARRIER_TIMEOUT_S``, 600 s by
    default, deliberately well above the slowest legitimate cell): reaching it means a peer is DEAD,
    not slow, so there is no live peer left to strand by exiting.

    Args:
        item: The test that just ran. Named in the exit message -- without it the operator gets a
            desync with no location, which is the situation this replaces.
        nextitem: Unused; part of the hook signature.

    Returns:
        None.

    Raises:
        Only via `pytest.exit`, on a barrier timeout. Every other exception is swallowed with a
        warning, because a teardown barrier must never mask the test's own outcome -- a broken
        barrier is a worse diagnostic than the failure it would hide.
    """
    try:
        import torch

        if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
            return
    except Exception:  # noqa: BLE001 — torch import/state probe must never break teardown
        return

    from fold_cp_ops.distributed.collective_symmetry import BarrierTimeout, CollectiveGate

    try:
        CollectiveGate().barrier()
    except BarrierTimeout as e:
        msg = (
            f"ranks desynchronized at teardown of {item.nodeid}: {e} "
            f"Every later test would enter a collective on a group that is already broken, so the "
            f"session stops here rather than reporting N unrelated failures or hanging."
        )
        # EMIT BEFORE UNWINDING. `pytest.exit`'s text is printed at SESSION END, and the unwind has
        # to survive every fixture finalizer to get there -- which is exactly what it could not do:
        # measured per-heartbeat on a wedged pair, rank0 was in `CollectiveGate.barrier` at 45 s and
        # 90 s and in `cuda_jit_executor.__del__ -> unload` by 135 s, so the bound FIRED and the
        # message died in teardown. The diagnostic must not be a payload of the unwind that the
        # diagnostic itself is reporting on.
        #
        # stderr is unbuffered-flushed here rather than left to pytest's capture, because a wedged
        # session's captured buffer is discarded on the kill -- the same reason the wedge dumps go
        # to a file.
        import sys as _sys

        print(f"\nNAMED DESYNC: {msg}", file=_sys.stderr, flush=True)
        # AND TO A FILE, because stderr alone is not delivery. It travels a pipe through
        # `srun` -> `ssh`, and a timeout-kill discards whatever is still in flight -- measured, that
        # is exactly what ate an entire control run's output. This commit's predecessor cited the
        # wedge dumps' file-based design as its reason and then did not copy it, which left an
        # ABSENCE with four readings (the raise did not happen / the run did not exercise it / the
        # fix is broken / delivery was lost) where a file leaves three. Open-append-close per write,
        # the same containment that kept a refcount readout alive through a run that died.
        try:
            # STAMPED, because the file is APPEND and accumulates across every iteration of a run.
            # Without a stamp, co-presence of two ranks' files is unreadable as co-occurrence in
            # TIME -- measured: two files existing was read (by me) as "both ranks timed out on each
            # other", when they held three entries written minutes apart across separate iterations.
            # The file mtime only dates the LAST write, so it cannot date the others. One field per
            # entry makes the same artifact answer a question it currently cannot.
            import time as _time

            _stamp = _time.strftime("%H:%M:%S")
            with open(_wedge_dir / f"desync_rank{_wedge_rank}.txt", "a") as _fh:
                _fh.write(f"[{_stamp} pid={os.getpid()} rank={_wedge_rank}] {msg}\n")
        except Exception:  # noqa: BLE001 — a diagnostic write must never mask the desync it reports
            pass
        # Tell the finalizers not to touch the group. `cleanup()` would otherwise run a pre-destroy
        # barrier and `destroy_process_group`, both COLLECTIVE, on a group this timeout just declared
        # broken -- which `BarrierTimeout`'s own contract forbids and which is where the unwind parked.
        try:
            from fold_cp_ops.distributed.distributed_manager import DistributedManager

            DistributedManager._state["_group_broken"] = True
        except Exception:  # noqa: BLE001 — never let the flag-set mask the desync being reported
            pass
        pytest.exit(msg, returncode=1)
    except Exception as e:  # noqa: BLE001 — see Raises: never mask the test's own outcome
        warnings.warn(f"teardown barrier raised (ignored): {e!r}", stacklevel=2)


@pytest.fixture(autouse=True)
def _return_freed_blocks_to_the_driver():
    """Return torch's cached-but-free CUDA blocks to the driver after every distributed test.

    Purpose
        nvshmem grows its symmetric heap with ``cuMemCreate``, which needs PHYSICAL pages. Torch's
        caching allocator does not return freed blocks to the driver on its own, so a cell that has
        finished can still deny the next one its pages -- observed as
        ``mem_heap.cpp:2135 ... CUDA_ERROR_OUT_OF_MEMORY`` on an IDLE 80 GiB card. Distributed
        operands here are ``O(N_token**2 * D)``, which is why this directory needs it and
        ``tests/kernels/`` does not.

    Semantics
        Teardown only. ``empty_cache()`` releases blocks nothing references; it CANNOT free memory
        the test still holds. Two consequences worth knowing:

        * A PASSING test's tensors are already released when its frame is destroyed at return, so
          this fixture only returns the blocks -- it is not what drops the references.
        * A FAILING test's frame is kept alive by pytest's traceback, so its tensors survive this
          teardown BY DESIGN. No fixture can change that, which is why the sites with the largest
          operands also null their aliases explicitly rather than relying on scope exit.

    Scope
        ``tests/distributed/`` only, deliberately. Measured on 8xH20: ~3 ms per test, which is
        ~24 s across the 8000-test kernel module for no benefit there, since those tests are not
        under memory pressure.

    Returns:
        Nothing; this is a teardown-only autouse fixture. Silent when CUDA is unavailable, so the
        GPU-free band tests are unaffected.
    """
    yield
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001 — a teardown must never mask the test's own outcome
        pass
