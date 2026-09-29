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

"""Group-requiring tests for the `persist=` artifact cache: the per-rank half.

``tests/_internal/test_artifact_cache.py`` covers the store's host logic with no GPU. What CANNOT
be checked there, and is checked here, is everything that only exists once a process group and an
nvshmem context do:

  * artifacts are PER-RANK -- N ranks produce N distinct shas at N distinct paths;
  * a rank handed ANOTHER rank's artifact REFUSES it, before ``library_init``. Unguarded that is
    ``CUDA_ERROR_ILLEGAL_ADDRESS`` inside ``nvshmem init.cu:2183``, and the CUDA context is poisoned
    for everything after it -- so the run either refuses cleanly or dies, with no third outcome;
  * a reloaded kernel is BITWISE identical to the freshly compiled one it replaces;
  * reuse is real ACROSS PROCESSES, which one pytest session cannot show on its own (see the
    two-phase note on :func:`test_a_second_process_reuses_the_artifact_this_one_minted`).

WHY THIS FILE MATTERS MOST ON IB, NOT NVLINK. ``nvshmemx_culibrary_init`` does strictly more work on
the IB path -- under ``#if defined(NVSHMEM_IBGDA_SUPPORT)`` it resolves and allocates per-module
transport state -- so a single-node run never exercises the state these claims concern. Run this
file at ``WORLD_SIZE=16`` across two IB-connected nodes; a 2-rank NVLink run is a smoke test of the
same code, not a validation of it.

EVERY SKIP HERE IS DECLARED, per the directory's rule: ``rank_invariant_skip`` for predicates that
are job-uniform (the image, the arch, the world size) and nothing else. A bare ``pytest.skip`` on a
per-rank predicate is a deadlock, not a skip.
"""

import hashlib
import json
import os
import time
from pathlib import Path

import pytest
import torch

from fold_cp_ops._internal import artifact_cache as ac
from fold_cp_ops.testing.collective_guard import gated_skip, rank_invariant_skip
from fold_cp_ops.testing.kernel_matrix import (
    Axis,
    KernelMatrix,
    computes_nothing_numeric,
    no_unsupported,
)
from fold_cp_ops.testing.numeric_guard import numeric_exempt

try:
    import nvshmem.core  # noqa: F401

    _HAS_NVSHMEM = True
except ImportError:  # pragma: no cover - non-nvshmem image
    _HAS_NVSHMEM = False

#: A module-level mark, not a `skip` call: both predicates are properties of the IMAGE and the BOX,
#: which are uniform across a homogeneous launch, so every rank reaches the same verdict and none can
#: leave alone. `collective_guard` does not see marks, so this reasoning is written rather than
#: enforced -- exactly as `test_gemm_bitcode_compile.py` records for the same pair.
pytestmark = pytest.mark.skipif(
    not _HAS_NVSHMEM, reason="needs nvshmem4py (the artifact route links the device bitcode)"
)

#: The world sizes this file is meaningful at, PARAMETRIZED -- every test draws from it and skips
#: the values this launch cannot build. That is the mesh-axis shape, not an ordinary pool: one
#: launch fixes WORLD_SIZE, so coverage is a property of the SET of launches and is reconciled
#: through `coverage_ledger` rather than asserted by any single run. An axis nothing parametrizes is
#: decoration -- it satisfies the diversity floor and changes nothing that runs -- and the matrix
#: audit fails a module for it, which is how this file's first draft was caught.
#: 16 is the value that matters: the only one that spans two nodes, hence the only one reaching
#: `nvshmemx_culibrary_init`'s IB-only branch, where it allocates per-module transport state a
#: single-node run never touches.
ARTIFACT_MATRIX = KernelMatrix(
    kernel="artifact_cache_distributed",
    axes=(
        Axis(
            name="world",
            domain=(
                "PE counts a launch can have. 2 is a workstation smoke; 8 fills one node over "
                "NVLink; 16 is the only value that spans two nodes and so the only one that reaches "
                "the per-module IB transport state a single-node run never allocates"
            ),
            values=(2, 8, 16),
            facets={
                "single_node": lambda w: w <= 8,
                "cross_node": lambda w: w > 8,
                "minimal": lambda w: w == 2,
            },
        ),
    ),
    computes=computes_nothing_numeric(
        because=(
            "the subject is an artifact STORE. The only number it produces is the output of a "
            "kernel it did not author, compared BITWISE against the same kernel freshly compiled -- "
            "so there is no tolerance to derive and no input distribution that could hide a defect"
        )
    ),
    unsupported=no_unsupported(
        because=(
            "every declared world size is a legal launch for this store; the rank scoping is "
            "defined for all of them. The store's refusals are about artifact CONTENT (wrong rank, "
            "wrong backend, corrupt bytes) rather than about a combination of declared axis values, "
            "and each is covered by a test that asserts the refusal directly"
        )
    ),
)


@pytest.fixture
def nvshmem_ready(dist_manager):
    """Bring nvshmem up once, or skip the dependent tests ON EVERY RANK.

    Every kernel in this file links the nvshmem device bitcode and registers it with
    ``library_init``, and ``library_init`` against an UNINITIALISED nvshmem SEGFAULTS -- measured:
    without this fixture the first test died at exitcode -11 with `setup passed` and no `call` line,
    which is what a fault looks like from outside.

    `gated_skip`, not `rank_invariant_skip`: the predicate is PER-HOST, not per-job. nvshmem can
    bootstrap on one node and fail on another for genuinely local reasons -- a missing
    ``libnvshmem_host.so`` in one image, no IB device on one host, a UID handshake one rank times out
    of. A bare skip then fires on a SUBSET of ranks and the rest block in the next collective until
    a watchdog kills the job, blaming whatever test they happened to be in. Every rank therefore runs
    the init, records its own outcome, and reaches the reduction UNCONDITIONALLY.

    Args:
        dist_manager: the session manager. Its group must already be up -- ``CollectiveGate`` refuses
            an uninitialised group, and this dependency is what guarantees it.

    Returns:
        The ``dist_manager``, once every rank agrees nvshmem is usable.

    Raises:
        Nothing directly. Skips on ALL ranks or none.
    """
    from fold_cp_ops.distributed.distributed_manager import DistributedManager

    local_reason = None
    try:
        DistributedManager.init_nvshmem()
        if not dist_manager.nvshmem_initialized:
            local_reason = "nvshmem did not initialize"
    except Exception as e:  # noqa: BLE001 -- any bootstrap failure is a skip, not an error
        local_reason = f"nvshmem init unavailable: {type(e).__name__}: {e}"
    gated_skip(local_reason)
    return dist_manager


def _dm_for(world, apply_mesh, world_size):
    """Gate this launch against the parametrized world size, then build a flat ``cp = world`` grid.

    One launch fixes WORLD_SIZE, so at most one declared value is buildable per run and the rest
    must be skipped. `rank_invariant_skip` rather than a bare `pytest.skip`: the predicate is
    "declared != this launch's world size", and WORLD_SIZE is set identically on every rank by the
    launcher -- so every rank compares the same two numbers, reaches the same verdict, and none can
    leave a collective alone. The skipped values are what the coverage ledger reconciles across the
    SET of launches; no single run can cover this axis.

    A helper rather than a fixture because `KernelMatrix.parametrize` has no `indirect=` -- the
    value arrives as a test PARAMETER, which a fixture cannot take as an argument.

    Args:
        world: the parametrized value.
        apply_mesh: the conftest's mesh builder. Used rather than initialising a manager here so
            this file shares ONE process group with every other distributed test in the session --
            a second `DistributedManager.initialize` would leave ranks in different collectives.
        world_size: the conftest's live world size.

    Returns:
        The ``DistributedManager`` with the grid applied.

    Raises:
        Skips, on EVERY rank or none, when this launch is a different size.
    """
    # CALLED ONLY WHEN THE PREDICATE HOLDS. `rank_invariant_skip(reason, because=)` ALWAYS skips --
    # it is a plain `pytest.skip` with a declaration attached. Passing `None` as the reason skips
    # with an opaque `<Skipped instance>` rather than not skipping; that is `gated_skip`'s
    # convention, not this one. Conflating them made all 18 cells skip while the session still
    # exited 0, which reads exactly like a healthy run.
    if world != world_size:
        rank_invariant_skip(
            f"declared world={world}, launch has {world_size}",
            because=(
                "WORLD_SIZE comes from the launcher and is identical on every rank of a job, so "
                "every rank compares the same two numbers and reaches the same verdict"
            ),
        )
    return apply_mesh((("cp", world),))


def _agreed_artifact_root(tmp_path_factory) -> str:
    """An artifact root EVERY RANK AGREES ON -- which `tmp_path_factory` alone does NOT give.

    Purpose
        Both `PersistSpec` fixtures below promise a shared directory, and both derived it from
        `tmp_path_factory.getbasetemp()`, which is **per PROCESS**: at world 16 the ranks landed in
        `/tmp/pytest-of-<user>/pytest-0`, `pytest-1`, `pytest-2`, ... so no rank could see a peer's
        artifact. `test_a_wrong_rank_artifact_is_refused_before_library_init` then failed on "peer
        artifact absent: the ranks did not all write" -- at EVERY world size, measured on base
        `9001bf9` at world 2, 8 and 16. The fixtures' docstrings were right about the intent and the
        implementation never delivered it.

    Semantics
        Prefers `CPO_JIT_ARTIFACT_DIR` when the launcher names one -- that is already agreed. Failing
        that, rank 0's basetemp is BROADCAST, so every rank uses the same path rather than each
        computing its own and hoping they coincide. Collective, and reached by every rank
        unconditionally, so it cannot strand one.

    Args:
        tmp_path_factory: pytest's factory, used only for rank 0's fallback root.

    Returns:
        A path string identical on every rank. The directory exists on return.
    """
    if _AGREED_ROOT:
        # MEMOISED, and that is a correctness property rather than a saving: the probe below is
        # COLLECTIVE, so re-running it once per fixture that wants a root would make the number of
        # barriers depend on which fixtures a test happens to request. Every rank runs the same
        # tests, so the counts would still match today -- but a test that used one root fixture and
        # not the other would be one refactor away from a hang. One agreement per session.
        return _AGREED_ROOT[0]
    root = os.environ.get("CPO_JIT_ARTIFACT_DIR")
    token = "solo"
    import torch.distributed as dist

    live = dist.is_available() and dist.is_initialized()
    if not root:
        root = str(tmp_path_factory.getbasetemp() / "artifacts")
    if live:
        # Broadcast the TOKEN as well as the path. The token is what stops a probe file left by an
        # EARLIER session under a persistent `CPO_JIT_ARTIFACT_DIR` from answering this session's
        # visibility question -- a stale probe would report "shared" on a root that is not.
        box = [(root, f"{os.getpid()}-{time.time_ns()}") if dist.get_rank() == 0 else None]
        dist.broadcast_object_list(box, src=0)
        root, token = box[0]
    os.makedirs(root, exist_ok=True)
    _ROOT_IS_SHARED[root] = _probe_root_is_shared(root, token)
    _AGREED_ROOT.append(root)
    return root


#: The one agreed root for this session, appended by :func:`_agreed_artifact_root` on first call.
_AGREED_ROOT: list = []

#: Verdict per root from :func:`_probe_root_is_shared`, filled by :func:`_agreed_artifact_root`.
#: Read through the `artifact_root_is_shared` fixture; never written by a test.
_ROOT_IS_SHARED: dict = {}


def _probe_root_is_shared(root: str, token: str) -> bool:
    """Does every rank of this group SEE the same directory, or merely spell it the same way?

    Purpose
        Agreeing on a PATH is not sharing a DIRECTORY, and the difference is invisible until a job
        spans nodes. Measured on venue B at world 16, 2 nodes x 8:
        ``test_a_wrong_rank_artifact_is_refused_before_library_init[world16]`` failed on rank 7 and
        rank 15 -- and ONLY those two -- because the agreed root was under node-local ``/tmp``, so
        each node had its own directory of that name. The test reads rank ``(me+1) % ws``'s
        artifact, and ``(me+1) % ws`` crosses the node boundary at exactly two ranks: the last local
        rank of each node. At world 2 and world 8 the whole job is on one node, every rank's
        neighbour is local, and the defect is unreachable.

        The failure mode is worse than the miss: `-x` made the two failing ranks leave while the
        other fourteen blocked in the next collective, so it surfaced as a 600 s NCCL timeout and an
        abort, not as an assertion.

    Semantics
        Every rank writes ``.sharedprobe_<pe>_<token>``, barriers, then looks for its neighbour's.
        The verdict is reduced with MIN, so it is IDENTICAL on every rank -- which is what makes a
        `gated_skip` on it legal under this directory's skip rule rather than a per-rank divergence.
        Collective and unconditional: every rank reaches every collective in it.

        Probe files are LEFT IN PLACE. They live under a session tmp root (or a launcher-named cache
        dir) and are named by token, so nothing reads them again; removing them would add a delete
        to a path whose whole purpose is to be cheap and side-effect-free.

    Args:
        root: the agreed root, already created on every rank.
        token: a session-unique string, identical on every rank (broadcast from rank 0). A repeated
            token across sessions would let a stale probe answer for a root that is not shared.

    Returns:
        True if every rank can see its neighbour's probe, or if there is no live group / the world
        is 1 (nothing to share with). False otherwise, identically on every rank.
    """
    import torch
    import torch.distributed as dist

    if not (dist.is_available() and dist.is_initialized()):
        return True
    ws, me = dist.get_world_size(), dist.get_rank()
    if ws == 1:
        return True
    Path(os.path.join(root, f".sharedprobe_{me}_{token}")).write_text(str(me))
    dist.barrier()
    peer = os.path.join(root, f".sharedprobe_{(me + 1) % ws}_{token}")
    seen = torch.tensor([1 if os.path.exists(peer) else 0], dtype=torch.int32)
    if dist.get_backend() == "nccl":
        seen = seen.cuda()
    dist.all_reduce(seen, op=dist.ReduceOp.MIN)
    return bool(int(seen.item()))


@pytest.fixture
def artifact_root_is_shared(dist_manager, tmp_path_factory) -> bool:
    """True when the artifact root is visible to EVERY rank -- the precondition for a peer read.

    Args:
        dist_manager: forces group initialisation, so the probe inside the root helper is collective.
        tmp_path_factory: passed through to :func:`_agreed_artifact_root`.

    Returns:
        The reduced verdict from :func:`_probe_root_is_shared`, identical on every rank. Defaults to
        True when no verdict was recorded, so a caller that never agreed a root is not skipped.
    """
    return _ROOT_IS_SHARED.get(_agreed_artifact_root(tmp_path_factory), True)


@pytest.fixture
def spec(request, tmp_path_factory, dist_manager):
    """A `PersistSpec` in a directory EVERY RANK AGREES ON, which is not what `tmp_path` gives.

    Args:
        tmp_path_factory: pytest's factory. Its per-rank base differs between processes on a
            multi-node launch, so the path is derived from a shared root instead.
        dist_manager: forces group initialisation first.

    Returns:
        A ``PersistSpec`` under ``$CPO_JIT_ARTIFACT_DIR`` when set (the launcher's choice, and the
        only option that works across two nodes), else a session-local directory.

    Note:
        Deriving the directory per rank would make every rank's artifact invisible to every other,
        so the "16 distinct shas" assertion would pass for the wrong reason -- each rank would be
        looking only at its own file. The shared root is what makes that assertion meaningful.
    """
    root = _agreed_artifact_root(tmp_path_factory)
    # THE TEST NAME IS IN THE KEY, so no two tests share an artifact FILE. Required, and the reason
    # is an OPEN QUESTION rather than a tidiness preference: with one shared key, a test that
    # `free()`d its wrappers and let them fall out of scope was followed by a test that loaded the
    # SAME file, and rank 1 took a SIGSEGV. `free()` calls `library_finalize` but does not release
    # the module, so the CUDA library unloads later at GC -- and whether re-loading that same path in
    # the same process before or after that unload is safe has NOT been established here.
    # Per-test keys make each test independent so the suite measures the cache rather than that
    # interaction. The interaction itself is recorded as open; it is a real question about the
    # design, not about the tests, and it deserves its own investigation.
    # ISOLATION IS BY DIRECTORY, NOT BY KEY -- and that changed at schema 2.
    #
    # An earlier revision isolated tests by folding the test name and a run tag INTO `spec.key`, because the key
    # was the caller's to compose. It no longer is: `program_key` DERIVES identity from the functor's
    # class, its `compile_key()`, the operand signature and the compile options. Two tests compiling
    # the same kernel therefore SHOULD produce the same key -- that is the mechanism working -- so
    # key-based isolation is now both impossible and wrong: it would perturb the very value under
    # test.
    #
    # Directory isolation is orthogonal. It separates the STORES without touching what a key means,
    # so a test still starts cold without lying about identity.
    #
    # The run tag stays, as a DIRECTORY component rather than a key component. The launcher has to
    # supply it (`CPO_ARTIFACT_RUN_TAG`): only the launcher knows where one run ends and the next
    # begins, and every in-process candidate was measured wrong -- `MASTER_PORT` fails under srun
    # because `DistributedManager` derives it from the job id, so both phases of a two-phase protocol
    # in one allocation read the identical value.
    #
    # Whatever the tag is, it must be RANK-UNIFORM. A per-rank token (a pid, a timestamp) puts each
    # rank in its own directory and quietly destroys the cross-rank assertions this file exists to
    # make, even though both "differ per run".
    node = request.node.name.split("[")[0]
    launch = os.environ.get("CPO_ARTIFACT_RUN_TAG", "norun")
    d = os.path.join(root, f"{launch}__{node}")
    os.makedirs(d, exist_ok=True)
    return ac.PersistSpec(dir=d)


@pytest.fixture
def shared_spec(tmp_path_factory, dist_manager):
    """A spec STABLE ACROSS LAUNCHES -- the only one that can express cross-process reuse.

    Everything else keys per launch so each run starts cold. This one deliberately does not: the
    claim "a compile survives process death" is unmakeable if the key changes when the process does.

    Args:
        tmp_path_factory: fallback root when the launcher names no artifact directory.
        dist_manager: supplies the world size, and forces group initialisation first.

    Returns:
        A ``PersistSpec`` whose key depends only on the workload and the world size.

    Note:
        A session-local fallback directory makes this spec cold on EVERY run, so the test that uses
        it can only assert the mint half there and says so. Point ``$CPO_JIT_ARTIFACT_DIR`` at a
        directory that survives between launches to exercise the reuse half.
    """
    root = _agreed_artifact_root(tmp_path_factory)
    return ac.PersistSpec(key=("dist_test_shared", "bf16", int(dist_manager.world_size)), dir=root)


@pytest.fixture
def rank_scope(dist_manager):
    """This process's ``RankScope``, the thing the store keys and checks artifacts by.

    Args:
        dist_manager: the session manager; supplies the live world size and rank. Taken from the
            LAUNCH rather than from the parametrized `world`, because an artifact is scoped by what
            nvshmem actually reports and a test that scoped it by a declared value could pass while
            the store keyed on something else.
    """
    return ac.RankScope(world_size=int(dist_manager.world_size), pe=int(dist_manager.rank))


def _compiled_signal_kernel(dm, persist, *, register=True):
    """Compile a small in-kernel-nvshmem kernel through the shipped `compile_nvshmem(persist=)`.

    The device-signal kernel is used rather than a full A2A store on purpose: it is the smallest
    thing in the repo that genuinely issues an nvshmem DEVICE op, so it needs the bitcode link and
    the ``library_init`` registration -- the two properties under test -- while compiling in seconds
    rather than minutes. A cache test that takes five minutes per assertion does not get run.

    Args:
        dm: the initialised manager (supplies ``cp`` and the device).
        persist: the ``PersistSpec``, or None for an uncached compile.
        register: whether to hand the module to ``library_init``. **The pure cache-bookkeeping tests
            pass False, and that is a correctness requirement on a cross-node launch, not a
            shortcut.** A device-issued nvshmem signal is NVLink-only on this fabric, and at
            ``WORLD_SIZE=16`` across two nodes the register-and-free cycles on this kernel
            segfaulted every rank. Registration is exercised where it belongs -- in
            `test_a_reloaded_kernel_is_registered_and_bitwise_identical`, which runs single-node --
            while hit/miss, mode and cross-process reuse are properties of the STORE and need no
            nvshmem state at all. Testing them with `register=False` makes them meaningful on both
            fabrics instead of only one.

    Returns:
        The ``CompiledGemmBitcode``. The CALLER MUST KEEP IT ALIVE for as long as the kernel may
        launch; dropping it unloads a CUDA library nvshmem still holds a raw handle to, and the next
        launch faults rather than raises.
    """
    import cutlass
    import cutlass.cute as cute
    import cutlass.torch as cutlass_torch
    import nvshmem.core

    from fold_cp_ops.distributed.gemm_bitcode_compile import compile_nvshmem
    from fold_cp_ops.distributed.workflows.trimul_autotuned import _DeviceSignalKernel

    # NO MLIR-VERSION SKIP ANY MORE. This used to `rank_invariant_skip` when the DSL had no
    # `to_precompiled_mlir`, which on the pinned 4.4.2 is ALWAYS -- so every persist= cell in this
    # file skipped, and the artifact cache shipped with its distributed coverage entirely inert.
    # `resolve_program_key` now falls back to the flat key (functor `compile_key()` + operand
    # layouts), which is complete because `compile_key()` reads the post-construction gates. The
    # cells run here on 4.4.2 and take the MLIR key on 4.7.0+, so both routes are exercised by
    # running this file on both DSLs rather than by skipping one.
    cp = int(dm.world_size)
    op = _DeviceSignalKernel(cp, int(nvshmem.core.SignalOp.SIGNAL_ADD))
    f_sig = cute.runtime.make_fake_tensor(cutlass.Int64, (cp,), stride=(1,), assumed_align=8)
    f_pe = cute.runtime.make_fake_tensor(cutlass.Int32, (cp,), stride=(1,), assumed_align=4)
    # op_factory is REQUIRED under persist= (a later change deleted the composed-key fallback): the program key
    # is the hash of the emitted MLIR, taking it means TRACING, and `template_params` refuses a
    # second trace of an instance -- so the key must come from an object that is never compiled.
    return compile_nvshmem(
        op, f_sig, f_pe, cutlass_torch.current_stream(), register=register, persist=persist,
        op_factory=lambda: _DeviceSignalKernel(cp, int(nvshmem.core.SignalOp.SIGNAL_ADD)),
    )


# ---------------------------------------------------------------------------
# Per-rank identity -- the claim whose violation is a fault, not a wrong answer.
# ---------------------------------------------------------------------------
@ARTIFACT_MATRIX.parametrize("world")
@numeric_exempt("asserts artifact identity across ranks, not a computed tensor")
def test_each_rank_mints_its_own_artifact(world, apply_mesh, world_size, spec, rank_scope, nvshmem_ready):
    """N ranks must mint N artifacts at N distinct PATHS, each meta naming its own PE.

    THE ASSERTION IS ABOUT PATHS AND META, NOT ABOUT BYTES, and an earlier draft of this test got
    that wrong. It asserted N distinct SHAS and failed with "1 distinct shas across 2 ranks" --
    correctly, because ``_DeviceSignalKernel``'s every compile-time parameter (``cp``, the signal op)
    is rank-UNIFORM, so every rank compiles the identical program. Rank-varying bytes were measured
    on the A2A store kernels, which bake in a peer layout; they are a property of THOSE kernels, not
    of the cache.

    So the invariant the store actually provides, and the one that matters, is: each rank reads and
    writes ITS OWN FILE. That holds whether or not the contents coincide, it is what stops N ranks
    racing onto one path, and it is what makes the meta ``pe`` check meaningful. The cache cannot
    know which kind of kernel it is holding -- so it scopes every one by rank, which costs extra
    misses for the rank-uniform kind and prevents a fault for the other.

    The sha-set size is recorded rather than asserted, because both 1 and N are legitimate answers
    and which one appears is the KERNEL's property.
    """
    dm = _dm_for(world, apply_mesh, world_size)
    import torch.distributed as dist

    compiled = _compiled_signal_kernel(dm, spec)
    assert compiled is not None  # keeps the module alive for the rest of the test

    o_path = compiled.artifact_path
    assert o_path is not None and o_path.exists(), (
        f"rank {rank_scope.pe} minted nothing at {o_path}"
    )
    assert f".pe{rank_scope.pe}.of{rank_scope.world_size}" in o_path.name

    # This rank's meta must name THIS rank. That is the field the wrong-rank guard reads, so a
    # store that wrote someone else's pe here would defeat the guard silently.
    meta = json.loads(o_path.with_name(o_path.stem + ".meta.json").read_text())
    assert meta["pe"] == rank_scope.pe
    assert meta["world_size"] == rank_scope.world_size

    sha = hashlib.sha256(o_path.read_bytes()).hexdigest()
    gathered = [None] * int(dm.world_size)
    dist.all_gather_object(gathered, (sha, o_path.name, meta["pe"]))
    names = {g[1] for g in gathered}
    pes = {g[2] for g in gathered}
    assert len(names) == int(dm.world_size), (
        f"only {len(names)} distinct artifact PATHS across {dm.world_size} ranks -- "
        "ranks are sharing a file and will race on it"
    )
    assert pes == set(range(int(dm.world_size))), f"pe coverage {sorted(pes)} != every rank"
    # Recorded, not asserted: 1 means the kernel's compile is rank-uniform, N means it is not, and
    # both are correct. Anything strictly between would mean the key is unstable, which IS a defect.
    n_shas = len({g[0] for g in gathered})
    assert n_shas in (1, int(dm.world_size)), (
        f"{n_shas} distinct shas across {dm.world_size} ranks is neither all-same nor all-different, "
        "which for a deterministic compiler means the artifact key is unstable"
    )
    compiled.free()


@ARTIFACT_MATRIX.parametrize("world")
@numeric_exempt("asserts a refusal, not a computed value")
def test_a_wrong_rank_artifact_is_refused_before_library_init(
    world, apply_mesh, world_size, spec, rank_scope, nvshmem_ready, artifact_root_is_shared
):
    """Hand this rank its neighbour's artifact; require a clean REJECT and a quarantine.

    This is the control the whole rank-scoping design exists for. Without the meta ``pe`` check, the
    artifact reaches ``library_init`` and the measured result is ``CUDA_ERROR_ILLEGAL_ADDRESS`` at
    ``nvshmem init.cu:2183`` plus a poisoned CUDA context -- meaning every later test in the session
    fails too, for reasons that look nothing like this one. So the check must precede the
    registration, and this test asserts it does by never letting a DSL call happen at all.
    """
    dm = _dm_for(world, apply_mesh, world_size)
    import torch.distributed as dist

    # NO COMPILE HERE, DELIBERATELY, and both halves of that are load-bearing.
    #
    # The subject is the meta `pe` check, which is pure store logic -- what makes this a DISTRIBUTED
    # test is that the pe comes from a real rank in a real group, not that a kernel was built. So the
    # artifacts are written directly, with bytes that need not be a real image.
    #
    # Compiling here was tried and is wrong twice over. A cache HIT maps the file via `load_module`,
    # and the in-place rewrite below then changes bytes under a loaded CUDA library -- a SIGSEGV
    # (the store never does this: it publishes with `os.replace` and quarantines with a rename, both
    # of which leave an open mapping on the old inode). Guarding that with "assert the key is cold"
    # then made the test FAIL on any warm directory, which is exactly the second phase of the
    # cross-process protocol this file is run under.
    own = ac.PersistSpec(dir=os.path.join(spec.dir, "wrongrank"))
    os.makedirs(own.dir, exist_ok=True)
    ws, me = int(rank_scope.world_size), int(rank_scope.pe)

    # WHICH NEIGHBOUR, and why it is not always `(me + 1) % ws`.
    #
    # This test reads a PEER'S FILE, so it needs a root every rank can SEE -- and agreeing on a
    # path is not sharing a directory. Measured on venue B at world 16 (2 nodes x 8) with the root
    # under node-local `/tmp`: this failed on rank 7 and rank 15 and NOWHERE ELSE, because
    # `(me + 1) % ws` crosses the node boundary at exactly the last local rank of each node. `-x`
    # then took those two ranks out while the other fourteen blocked in the next collective, so it
    # presented as a 600 s NCCL timeout and an abort rather than as an assertion.
    #
    # The SUBJECT is the meta `pe` check, and any pe != me exercises it. So when the root is not
    # shared, wrap WITHIN THE NODE instead of across the job: same claim, same strength, and the
    # world-16 cell keeps running instead of skipping. When the root IS shared -- which is what
    # `$CPO_JIT_ARTIFACT_DIR` on a cluster filesystem gives -- take the true neighbour, because
    # that additionally proves the cross-node path.
    #
    # The rule is computed identically on every rank from job-uniform inputs, so no rank picks a
    # different scheme than its peers.
    local_ws = int(os.environ.get("LOCAL_WORLD_SIZE", ws))
    if artifact_root_is_shared or local_ws >= ws:
        peer_pe = (me + 1) % ws
    elif local_ws > 1:
        node, local = divmod(me, local_ws)
        peer_pe = node * local_ws + (local + 1) % local_ws
    else:
        gated_skip(
            "artifact root is not shared across nodes and LOCAL_WORLD_SIZE=1, so no peer's "
            "artifact is visible; point $CPO_JIT_ARTIFACT_DIR at a shared filesystem"
        )
    neighbour = ac.RankScope(world_size=ws, pe=peer_pe)

    # Keys built directly, which this test MAY do precisely because it compiles nothing: there is no
    # implementation key to stay in step with. The tests that DO compile read the key off the
    # returned wrapper instead, so they cannot drift from `compile_nvshmem`.
    #
    # ONE program key, TWO artifact keys -- which is the split under test: both ranks agree on the
    # program, and only `pe` separates the files.
    # A LITERAL program key. This test compiles nothing, so it has no MLIR to hash and no functor to
    # hash it from -- and it does not need one: its subject is the meta `pe` check, which reads the
    # ARTIFACT key. Deriving the program half through the product would couple a store test to a
    # derivation that has already been replaced once without its claim changing.
    prog = hashlib.sha256(f"wrongrank-{ws}".encode()).hexdigest()
    mine = ac.artifact_key(prog, rank_scope)
    theirs = ac.artifact_key(prog, neighbour)
    assert mine != theirs, "the key must separate ranks, or nothing below can be tested"

    payload = f"artifact-for-pe{me}".encode()
    ac.write_artifact(
        payload, mine, own, prefix="k", backend=ac.BACKEND_DUMP_OBJECT, rank=rank_scope
    )
    o_path, m_path = ac.artifact_paths(mine, own, rank_scope)
    n_o, n_m = ac.artifact_paths(theirs, own, neighbour)
    dist.barrier()  # every rank has written before any rank reads its neighbour's
    assert n_o.exists() and n_m.exists(), (
        f"peer artifact absent at {n_o}: rank {me} cannot see pe {peer_pe}'s file. Either the "
        f"ranks did not all write, or the root is not shared "
        f"(artifact_root_is_shared={artifact_root_is_shared}, LOCAL_WORLD_SIZE={local_ws}, "
        f"world={ws}) -- the second is a HARNESS fault, not a store fault"
    )

    # READ the neighbour's artifact BEFORE anyone starts planting, then barrier. Without that
    # barrier this is a torn read at scale: rank r rewrites its OWN meta (the plant below) while
    # rank r-1 is still reading it, and `write_text` truncates before it writes -- so the reader
    # sees an empty file and a JSONDecodeError. Measured: passed at WORLD_SIZE=2 and failed at 16,
    # which is how a race of this shape usually announces itself.
    #
    # The store itself is immune to exactly this, because it publishes with `os.replace`. The test
    # is not, because it deliberately writes in place to simulate a corrupted entry.
    peer_bytes = n_o.read_bytes()
    meta = json.loads(n_m.read_text())
    dist.barrier()

    # Plant the neighbour's bytes AND meta at my path, with the sha corrected so the ONLY thing
    # wrong is the rank. A test that also corrupted the bytes would be satisfied by the sha check
    # and would never reach the rank check it is named for.
    o_path.write_bytes(peer_bytes)
    meta["sha256"] = hashlib.sha256(o_path.read_bytes()).hexdigest()
    meta["bytes"] = o_path.stat().st_size
    m_path.write_text(json.dumps(meta))
    assert meta["pe"] != me, "the planted artifact must belong to a different PE"

    ac.STATS.reset()
    got = ac.read_artifact(mine, own, prefix="k", backend=ac.BACKEND_DUMP_OBJECT, rank=rank_scope)
    assert got is None, "a peer's artifact was ACCEPTED; the next library_init would fault"
    assert ac.STATS.rejects == 1
    assert any("pe mismatch" in r for r in ac.STATS.reject_reasons), ac.STATS.reject_reasons
    assert not o_path.exists(), "the offending entry must be quarantined, not merely skipped"


# ---------------------------------------------------------------------------
# The reloaded kernel is the same kernel.
# ---------------------------------------------------------------------------
@ARTIFACT_MATRIX.parametrize("world")
@numeric_exempt("compares two device outputs BITWISE via torch.equal, which numerics.py permits")
def test_a_reloaded_kernel_is_registered_and_bitwise_identical(world, apply_mesh, world_size, spec, rank_scope, nvshmem_ready, topology):
    """A disk-reloaded kernel must register with nvshmem and produce byte-identical output.

    Registration is checked BEFORE any launch, because the failure mode of an unregistered module is
    a fault rather than an exception -- an in-kernel nvshmem call resolves against unwritten device
    state. ``torch.equal``, not a tolerance: the two kernels are the same program, so anything short
    of exact equality is a defect and not a rounding difference.

    THE LAUNCH HALF IS SINGLE-NODE ONLY, and that is a property of the KERNEL, not of the cache.
    ``_DeviceSignalKernel`` issues its ``signal_op`` from the DEVICE, and a device-issued nvshmem
    signal is NVLink-only on this fabric -- the IB-capable path is the HOST-issued one. Measured on
    venue B at ``WORLD_SIZE=16`` across two nodes: the two artifact-identity tests passed and this
    one failed, then tasks 5-6 took a segfault. So on a multi-node launch the registration and
    artifact assertions still run -- they are what this file is for -- and only the launch-and-
    compare is skipped, naming the reason. Skipping the whole test would give up the cross-node
    coverage that is the only reason to run here at all.
    """
    dm = _dm_for(world, apply_mesh, world_size)
    fresh = reloaded = again = None
    try:
        fresh = _compiled_signal_kernel(dm, None)  # uncached, so this is genuinely a fresh compile
        assert fresh.from_cache is False

        reloaded = _compiled_signal_kernel(dm, spec)  # mints
        again = _compiled_signal_kernel(dm, spec)  # and now hits, in THIS process
        assert again.from_cache is True, "a second compile at the same key must load"
        assert again.module is not None, "a hit must retain its module or the CUDA library unloads"
        assert again.nvshmem_kernel_obj is not None, "a hit must be registered before any launch"
        _launch_and_compare(dm, fresh, again, topology)
    finally:
        # THE `finally` IS THE FIX, AND THE LEAK IT CLOSES WAS A CROSS-NODE SIGSEGV.
        # `_launch_and_compare` skips on a multi-node launch -- `pytest.skip` RAISES -- and the
        # release used to sit after it, so on two nodes all THREE `library_init`-ed modules leaked
        # to the exit sweep. Measured at WORLD_SIZE=16 over IB: the whole file gave 8 passed, 22
        # skipped, 0 failures and then SIGSEGV on all 16 ranks before pytest's summary, while
        # `-k "not reloaded"` gave the same 8 passed and exited 0.
        #
        # It was unreachable until the artifact cache started working on cutlass-dsl 4.4.2: every
        # persist cell here skipped on `mlir_keying_available()`, so this test never got past its
        # second line on the pinned DSL.
        for c in (again, reloaded, fresh):
            if c is not None:
                c.free()


def _launch_and_compare(dm, fresh, again, topology):
    """Launch both kernels into a symmetric pad and require bit-identical results.

    Purpose
        The launch half of `test_a_reloaded_kernel_is_registered_and_bitwise_identical`, split out
        so the caller can wrap the whole test in one `finally` -- this function SKIPS, and a skip
        raises, so anything after it in the caller's body does not run.

    Functionality & semantics
        Allocates a symmetric ``(cp,)`` int64 pad, barriers so no rank zeroes another's pad
        mid-flight, launches ``fresh`` then ``again``, and compares the two results with
        ``torch.equal``.

    Args:
        dm: the live `DistributedManager`; supplies ``world_size`` for the pad extent.
        fresh: the freshly compiled signal kernel.
        again: the disk-reloaded one. Must be the SAME program as ``fresh`` or the comparison is
            meaningless rather than failing.
        topology: the conftest's topology dict; ``n_nodes`` decides the skip.

    Returns:
        None.

    Raises:
        Skipped: on a multi-node launch, rank-invariantly. NOT a defect -- see below.
    """
    import cutlass.torch as cutlass_torch
    from cutlass.cute.runtime import from_dlpack

    if int(topology.get("n_nodes", 1)) > 1:
        rank_invariant_skip(
            f"device-issued nvshmem signal is NVLink-only; this launch spans "
            f"{topology['n_nodes']} nodes. The artifact assertions above already ran.",
            because=(
                "the node count is derived from WORLD_SIZE // LOCAL_WORLD_SIZE, both set by the "
                "launcher and identical on every rank, so every rank reaches the same verdict"
            ),
        )

    # THE SIGNAL PAD MUST BE SYMMETRIC MEMORY. `signal_op` writes into a PEER's copy at a symmetric
    # address, so a plain `torch.zeros` destination is an illegal access on the device -- measured:
    # this test's first version faulted with `cudaErrorIllegalAddress` at the synchronize after the
    # FRESH launch, which reads like a defect in the reloaded kernel and is nothing of the kind.
    # `_symmetric_empty` is the same allocator the real drain uses.
    from fold_cp_ops.distributed.workflows.trimul_autotuned import _symmetric_empty, _symmetric_free

    cp = int(dm.world_size)
    pad = _symmetric_empty((cp,), dtype=torch.int64)
    pad.zero_()
    pe_tab = torch.arange(cp, device="cuda", dtype=torch.int32)
    c_sig = from_dlpack(pad, assumed_align=8)
    c_pe = from_dlpack(pe_tab, assumed_align=4)
    # The kernel signals PEERS, so every rank must have its pad allocated and zeroed before any rank
    # launches, or an early rank writes into a pad a later rank then zeroes.
    import torch.distributed as dist

    dist.barrier()

    fresh(c_sig, c_pe, cutlass_torch.current_stream())
    torch.cuda.synchronize()
    dist.barrier()                       # every peer's signal has landed before anyone reads
    after_fresh = pad.clone()

    pad.zero_()
    torch.cuda.synchronize()
    dist.barrier()                       # and before anyone zeroes for the second round
    again(c_sig, c_pe, cutlass_torch.current_stream())
    torch.cuda.synchronize()
    dist.barrier()
    after_reloaded = pad.clone()

    assert torch.equal(after_fresh, after_reloaded), (
        f"reloaded kernel diverged: fresh={after_fresh.tolist()} reloaded={after_reloaded.tolist()}"
    )
    _symmetric_free(pad)


@ARTIFACT_MATRIX.parametrize("world")
@numeric_exempt("asserts cache bookkeeping, not a computed value")
def test_a_hit_and_a_miss_are_distinguishable(world, apply_mesh, world_size, spec, rank_scope, nvshmem_ready):
    """A store that silently never hits satisfies every positive assertion made about it.

    Reuse is ``hits==1 AND rejects==0 AND writes==0``, not merely that a read happened: a verified
    read whose load then fails quarantines and recompiles -- correct behaviour that a naive
    ``hits==1`` check reports as success. That is not hypothetical; it happened once on the
    export_c backend and the probe passed.
    """
    dm = _dm_for(world, apply_mesh, world_size)
    ac.STATS.reset()
    first = _compiled_signal_kernel(dm, spec, register=False)
    assert first.from_cache is False
    assert (ac.STATS.misses, ac.STATS.writes) == (1, 1), vars(ac.STATS)

    ac.STATS.reset()
    second = _compiled_signal_kernel(dm, spec, register=False)
    assert second.from_cache is True
    assert (ac.STATS.hits, ac.STATS.rejects, ac.STATS.writes) == (1, 0, 0), vars(ac.STATS)
    first.free()
    second.free()


@ARTIFACT_MATRIX.parametrize("world")
@numeric_exempt("asserts cache bookkeeping, not a computed value")
def test_mode_r_consumes_without_minting(world, apply_mesh, world_size, spec, rank_scope, nvshmem_ready):
    """A perf gate must be able to read entries without creating them.

    A gate that mints makes its own first and second runs different measurements, which is the one
    way a cache can corrupt a perf number while leaving every correctness test green.
    """
    dm = _dm_for(world, apply_mesh, world_size)
    minted = _compiled_signal_kernel(dm, spec, register=False)
    ro = ac.PersistSpec(key=spec.key, dir=spec.dir, mode="r")

    ac.STATS.reset()
    used = _compiled_signal_kernel(dm, ro, register=False)
    assert used.from_cache is True
    assert ac.STATS.writes == 0

    # and a MISS under mode="r" leaves nothing behind
    cold = ac.PersistSpec(key=spec.key + ("unseen",), dir=spec.dir, mode="r")
    ac.STATS.reset()
    fresh = _compiled_signal_kernel(dm, cold, register=False)
    assert fresh.from_cache is False
    assert ac.STATS.writes == 0, "mode='r' minted an entry"
    for c in (minted, used, fresh):
        c.free()


# ---------------------------------------------------------------------------
# Cross-rank agreement -- the checks whose whole value is in the FAILURE mode.
# ---------------------------------------------------------------------------
@ARTIFACT_MATRIX.parametrize("world")
@numeric_exempt("asserts a collective refusal, not a computed value")
def test_all_ranks_agree_on_the_program_key(
    world, apply_mesh, world_size, spec, rank_scope, nvshmem_ready
):
    """`agree_on_program_key` must NOT raise when the ranks agree -- the positive half, and it is cheap.

    A rank-free key is the expected state, and a mismatch means the ranks are about to compile or
    load different programs into one collective. This proves the check is quiet when it should be;
    the next one proves it fires when it should not.

    The key here is a rank-uniform LITERAL, not a derived one. M5 deleted the composed key, and the
    real derivation (`all_rank_program_key`) is COLLECTIVE and already has its own test -- so
    deriving it here would test that function twice and this one not at all. What is under test is
    the gather-and-compare, which takes any string.
    """
    import torch.distributed as dist

    dm = _dm_for(world, apply_mesh, world_size)
    prog = hashlib.sha256(f"agree-{int(dm.world_size)}".encode()).hexdigest()
    gathered = [None] * int(dm.world_size)
    dist.all_gather_object(gathered, prog)
    assert len(set(gathered)) == 1, f"{len(set(gathered))} distinct program keys across ranks"

    from fold_cp_ops.distributed.gemm_bitcode_compile import agree_on_program_key

    agree_on_program_key(prog)  # must not raise when the ranks agree


@ARTIFACT_MATRIX.parametrize("world")
@numeric_exempt("asserts a collective refusal, not a computed value")
def test_a_forced_one_rank_key_divergence_makes_EVERY_rank_raise(
    world, apply_mesh, world_size, spec, rank_scope, nvshmem_ready
):
    """Diverge ONE rank's key and require every rank to fail -- and the run not to hang.

    **The entire value of the agreement check is this failure mode**, so a happy-path test proves
    nothing about it. The mechanism under test is that the decision is REDUCED before anyone raises:
    a rank that detected the mismatch and raised alone would leave its peers blocked in the next
    collective until a watchdog killed the job, which is the deadlock the check exists to replace.

    So the assertion is two-sided by rank: the diverging rank must see `CollectiveFailure` (it
    failed), every other rank must see `PeerFailure` (it did not fail, and stops anyway). A run that
    HANGS here fails by timeout rather than by assertion, which is why the file is launched with a
    bounded faulthandler.
    """
    from fold_cp_ops.distributed.collective_symmetry import CollectiveFailure, PeerFailure
    from fold_cp_ops.distributed.gemm_bitcode_compile import agree_on_program_key

    dm = _dm_for(world, apply_mesh, world_size)
    # Rank-uniform LITERAL, for the same reason as the test above: the subject is the reduction, and
    # the derivation it used to call was deleted with the composed key.
    prog = hashlib.sha256(f"agree-{int(dm.world_size)}".encode()).hexdigest()
    diverging = int(dm.world_size) - 1  # the LAST rank, so rank 0's broadcast value is the good one
    mine = prog + "-DIVERGED" if int(dm.rank) == diverging else prog

    with pytest.raises((CollectiveFailure, PeerFailure)) as excinfo:
        agree_on_program_key(mine)
    if int(dm.rank) == diverging:
        assert isinstance(excinfo.value, CollectiveFailure)
        # the message must name what to fix, not merely that the keys differ
        assert "program_key MISMATCH" in str(excinfo.value.__cause__ or excinfo.value)
    else:
        assert isinstance(excinfo.value, PeerFailure), (
            "a rank whose key MATCHED must still stop -- otherwise it walks into the next "
            "collective alone, which is the deadlock this check replaces"
        )


# ---------------------------------------------------------------------------
# Cross-process reuse -- the claim one session cannot make about itself.
# ---------------------------------------------------------------------------
@ARTIFACT_MATRIX.parametrize("world")
@numeric_exempt("asserts cache bookkeeping across processes, not a computed value")
def test_a_second_process_reuses_the_artifact_this_one_minted(world, apply_mesh, world_size, shared_spec, rank_scope, nvshmem_ready):
    """Reuse ACROSS processes, checked by running this file twice against one directory.

    ONE SESSION CANNOT PROVE THIS. An in-process hit could come from anything the process still
    holds; the claim is that a compile survives process death. So the test keys off whether the
    artifact was already on disk when the session STARTED, which is recorded once at first use:

        phase 1 (cold dir)  every rank misses and mints        -> this test asserts "minted"
        phase 2 (same dir)  every rank finds it pre-existing   -> this test asserts "reused"

    The launcher decides which phase by whether ``$CPO_JIT_ARTIFACT_DIR`` already has entries. When
    it cannot tell -- a fresh session-local directory -- the test asserts the mint half only and
    says so, rather than silently checking nothing.
    """
    dm = _dm_for(world, apply_mesh, world_size)
    # Probe the path WITHOUT compiling, by asking the store where this spec's entries live and
    # whether any exists for this rank. The key itself comes off the wrapper afterwards.
    pre_existing = any(
        f.name.endswith(f"{rank_scope.infix}.o") for f in Path(shared_spec.dir).glob("*.o")
    )

    ac.STATS.reset()
    compiled = _compiled_signal_kernel(dm, shared_spec, register=False)
    try:
        if pre_existing:
            assert compiled.from_cache is True, (
                "the artifact was on disk before this PROCESS started and was not used -- "
                f"stats={vars(ac.STATS)}"
            )
            assert ac.STATS.rejects == 0, ac.STATS.reject_reasons
            assert ac.STATS.writes == 0, "a reuse must not rewrite the entry"
        else:
            assert compiled.from_cache is False
            assert ac.STATS.writes == 1, "a cold directory must be populated for the next process"
            assert compiled.artifact_path is not None and compiled.artifact_path.exists()
    finally:
        compiled.free()


# ---------------------------------------------------------------------------
# The ALL-RANK program key, on a real kernel through a real group.
# ---------------------------------------------------------------------------
@ARTIFACT_MATRIX.parametrize("world")
@numeric_exempt("asserts key identity across ranks, not a computed tensor")
def test_the_all_rank_program_key_is_IDENTICAL_on_every_rank(
    world, apply_mesh, world_size, nvshmem_ready
):
    """The invariant the whole design rests on: one key, agreed by construction.

    `tests/_internal/test_artifact_cache.py` proves the COMBINATION rules against a faked gather --
    ordering, world size, failure reduction -- because those are arithmetic and need no GPU. This
    proves the thing that fake cannot: that a REAL kernel traced through a REAL process group yields
    a key every rank lands on.

    `_DeviceSignalKernel` is the subject for the same reason the other tests here use it: it is the
    smallest thing in the repo issuing a genuine nvshmem device op, so it exercises the real compile
    path in seconds. Its compile-time parameters are rank-UNIFORM, so its per-rank hashes agree
    before the combine -- which is the WEAKER of the two cases and is stated as such. The
    rank-VARYING case was measured directly on `GemmA2ASm90` (ee4cb228 vs d9846317 at world 2,
    docs/jit_cache_survival.md M4); reproducing it here would mean standing up a full cp=2 workflow
    inside a cache test.
    """
    import cutlass
    import cutlass.cute as cute
    import cutlass.torch as cutlass_torch
    import nvshmem.core
    import torch.distributed as dist

    from fold_cp_ops._internal.artifact_cache import all_rank_program_key
    from fold_cp_ops.distributed.workflows.trimul_autotuned import _DeviceSignalKernel

    if not hasattr(cute.compile, "to_precompiled_mlir"):
        rank_invariant_skip(
            "cutlass has no to_precompiled_mlir; MLIR keying needs 4.7.0+",
            because="the DSL version is a property of the image, identical on every rank of a job",
        )

    # GATE on the parametrized world, via the same helper every other test in this file uses.
    # Without it `world` is a DECORATIVE axis: measured before this line existed, world2/world8/
    # world16 all reported PASSED in a WORLD_SIZE=2 launch, because nothing compared the declared
    # value to the launch's. Three identical runs of one configuration read exactly like three
    # configurations covered.
    _dm_for(world, apply_mesh, world_size)
    cp = int(world_size)
    f_sig = cute.runtime.make_fake_tensor(cutlass.Int64, (cp,), stride=(1,), assumed_align=8)
    f_pe = cute.runtime.make_fake_tensor(cutlass.Int32, (cp,), stride=(1,), assumed_align=4)
    args = (f_sig, f_pe, cutlass_torch.current_stream())
    factory = lambda: _DeviceSignalKernel(cp, int(nvshmem.core.SignalOp.SIGNAL_ADD))  # noqa: E731

    key = all_rank_program_key(factory, args)
    seen = [None] * cp
    dist.all_gather_object(seen, key)
    assert len(set(seen)) == 1, f"ranks disagreed on the all-rank program key: {sorted(set(seen))}"
    assert len(key) == 64, key

    # Stable across a second formation -- a key that varied run to run would miss every time.
    again = all_rank_program_key(factory, args)
    assert again == key, f"the key moved between two formations: {key} -> {again}"


@ARTIFACT_MATRIX.parametrize("world")
@numeric_exempt("asserts a collective refusal, not a computed tensor")
def test_ONE_ranks_trace_failure_refuses_on_EVERY_rank_rather_than_hanging(
    world, apply_mesh, world_size, nvshmem_ready
):
    """The failure mode under test is a HANG, so this is bounded and asserted on every rank.

    `all_gather_object` sits on the compile path. A rank whose trace raises and leaves alone blocks
    every peer inside it until a watchdog kills the job -- the same hazard, on the same kind of path,
    that `_refuse_unsatisfiable_symmetric_request` had to fix. The local failure is therefore carried
    INTO the gather and reduced.

    Rank 0 alone is given a factory that raises. Every rank must see `PeerTraceError`, and the ranks
    that traced fine must see it too -- that is the whole point, and a test asserting only on rank 0
    would pass against an implementation that hangs its peers.
    """
    import torch.distributed as dist

    from fold_cp_ops._internal.artifact_cache import PeerTraceError, all_rank_program_key

    _dm_for(world, apply_mesh, world_size)  # gate on the parametrized world -- see the note above

    def boom():
        raise RuntimeError("synthetic trace failure on rank 0")

    rank = dist.get_rank()
    factory = boom if rank == 0 else (lambda: None)
    with pytest.raises(PeerTraceError, match=r"could not produce an MLIR hash"):
        all_rank_program_key(factory, ())
    dist.barrier()  # if any rank had been left inside the gather, this is where it would stop
