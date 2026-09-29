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

"""Enforcement for the DISTRIBUTED perf gates -- the pinned medians of the two A2A-fused kernels.

Why this directory exists rather than more files under ``tests/perf/``: the gates need
``dist_manager`` / ``apply_mesh`` / ``world_size``, and those fixtures are defined ONLY in
``tests/distributed/conftest.py``. pytest conftests apply BY DIRECTORY, so a gate placed in
``tests/perf/`` cannot see them. Importing them across directories via ``pytest_plugins`` was
rejected: it makes ``pytest tests/perf/`` behave differently under torchrun, which then has to be
explained before any of its numbers can be read.

What is carried over from ``tests/perf/conftest.py``, and what deliberately is not:

* **carried, unchanged in spirit** -- the ``PINS`` requirement, the shared-session skip, the
  best-effort clock lock, the GPU warm-up, and the wedge watchdog. The watchdog is worth MORE here:
  a hung rank is the characteristic distributed failure, and a hang with no traceback is the one
  outcome that tells you nothing at all.
* **NOT carried -- the dispatch-pin requirement.** That gate exists because ``tests/perf/`` times
  with ``mode="device"``, which reports the kernel and nothing about the Python that submits it, so
  the submit path goes unwatched unless something insists. It does not transfer: a collective must
  be timed with ``mode="event"`` (fixed iteration count, hence lockstep across ranks), and an event
  window INCLUDES the host submit. Requiring a separate dispatch pin here would demand a pin for a
  cost the main measurement already contains.
* **added, and it is this directory's real hazard** -- :func:`pytest_collectstart` refuses a gate
  that does not declare ``mode="event", reduce="max"``. ``mode="device"`` picks its repetition count
  adaptively PER RANK; with a collective in the loop the ranks then run different iteration counts
  and DESYNC. That does not fail loudly, it produces numbers, which is why it needs a guard rather
  than a comment.

Skips in here are governed by ``fold_cp_ops.testing.collective_guard`` like everything else under
``tests/distributed/`` -- the guard scopes by directory name at any depth, so this directory was
covered before it had a file in it. Use :func:`mesh_or_skip`, never a bare ``pytest.skip``.
"""

from __future__ import annotations

import os
import pathlib

import pytest

from fold_cp_ops.testing import coverage_ledger
from fold_cp_ops.testing.collective_guard import rank_invariant_skip
from tests.perf._clocklock import clock_lock
from tests.wedge_watchdog import install_wedge_watchdog

# Wedge auto-dump: a hung rank is THE distributed failure mode, and an outer `timeout` that kills a
# wedged job leaves no evidence of where it was. This dumps every thread's traceback from a Python
# thread (holding the GIL for the walk) -- see tests/wedge_watchdog.py for why the faulthandler
# variant segfaulted healthy sessions.
install_wedge_watchdog()

#: Seconds of sustained matmul before the first timed cell, matching ``tests/perf``'s ``_gpu_warm``.
#: Not re-derived here: the measurement that set it (across-process spread 0.71% -> 0.29%) is a
#: property of the part, not of how many ranks are running.
_WARM_SECONDS = 10.0

#: The only timing configuration a collective may be measured with. ``mode="event"`` runs a FIXED
#: iteration count, so every rank performs the same number of collectives; ``reduce="max"`` reports
#: the slowest PE, which is what a collective's latency actually is. See CLAUDE.md's bench_utils
#: rule -- ``mode="device"`` adapts the repetition count per rank and desyncs collectives.
_REQUIRED_TIMING = {"mode": "event", "reduce": "max"}


def _timed_gate(path: pathlib.Path) -> bool:
    """Whether a collected module is a TIMED gate in this directory.

    Purpose: separate the modules that measure (and so must be pinned, mode-checked and
    session-isolated) from the unit tests of the gates' own machinery, which measure nothing and
    must run under ``pytest tests/`` like any other unit test.

    Args:
        path: The collected module's path. Only a file whose parent is THIS directory and whose
            name starts with ``test_benchmark_perf_`` is a timed gate; anything else -- including a
            timed gate in a sibling directory -- returns False, because a conftest may only speak
            for its own directory.

    Returns:
        True for a timed gate in this directory, False otherwise.
    """
    return path.parent == pathlib.Path(__file__).parent and path.name.startswith(
        "test_benchmark_perf_"
    )


def pytest_collectstart(collector):
    """Refuse a timed distributed gate that is unpinned or timed in a way that desyncs its ranks.

    Purpose
        Two things must be true of every number this directory produces, and neither is visible by
        reading a passing run: it came from a pin file rather than a module constant, and it was
        measured in lockstep across ranks. Both are checked here, at collection, so a gate that
        stopped satisfying them cannot ship quietly.

    Semantics
        Applies only to ``test_benchmark_perf_*`` modules whose parent is this directory. Each must
        expose:

        * ``PINS`` -- a :class:`tests.perf.pins.PinFile` built by ``pins.load(__file__)``, which by
          construction reads ``<same name>.json``. Same pin format as the single-device gates on
          purpose: one harvest/merge toolchain, not two.
        * ``TIMING`` -- a mapping equal to ``{"mode": "event", "reduce": "max"}``, or a module-level
          ``TIMING_EXEMPT`` string giving the reason. The exemption exists for a genuinely
          collective-free distributed kernel (a per-rank-independent cell), which is the only case
          where ``mode="device"`` is safe.

        It FAILS rather than skips. A skip is how a gate that stopped checking anything ships
        quietly, and the single-device directory has already had that happen once.

    Args:
        collector: The pytest collector. Only :class:`pytest.Module` collectors are inspected;
            anything else passes through untouched.

    Returns:
        None.

    Raises:
        pytest.UsageError: Naming the module and the exact declaration it is missing.
    """
    if not isinstance(collector, pytest.Module):
        return
    path = pathlib.Path(str(collector.path))
    if not _timed_gate(path):
        return
    from tests.perf.pins import PinFile

    module = collector.obj
    if not isinstance(getattr(module, "PINS", None), PinFile):
        raise pytest.UsageError(
            f"{path.name} is a timed distributed perf gate but does not expose a PINS loaded from "
            f"its pin file. Add `from tests.perf.pins import load as load_pins` and "
            f"`PINS = load_pins(__file__)`. A gate that keeps its numbers in module constants "
            f"cannot be told apart from one that guessed them."
        )
    if getattr(module, "TIMING_EXEMPT", None):
        return
    if getattr(module, "TIMING", None) != _REQUIRED_TIMING:
        raise pytest.UsageError(
            f"{path.name} must declare TIMING = {_REQUIRED_TIMING!r} (bench_utils' only "
            f"collective-safe configuration) or a TIMING_EXEMPT reason. mode='device' picks its "
            f"repetition count adaptively PER RANK, so ranks run different iteration counts and "
            f"DESYNC -- which yields numbers rather than an error, and is why this is checked."
        )


def pytest_collection_modifyitems(session, config, items):
    """Skip the timed gates when the session also collected non-perf tests.

    Purpose
        A pinned median is only meaningful under the conditions it was pinned in. The single-device
        directory measured this directly: the tightest cell read +11.5% against its pin under
        ``pytest tests/`` and reproduced to 0.1% under ``pytest tests/perf/``, same kernel. The
        mechanism was never isolated -- a controlled 1030-test in-process replay moved it only
        +1.0% -- but the two invocations disagree by more than the band, which is enough to make the
        isolated one the only supported way to read these numbers.

        Here there is a second reason, specific to distributed: correctness tests under
        ``tests/distributed/`` build and tear down process groups and symmetric pools. A gate that
        runs after them measures a different allocator state, not a different kernel.

    Semantics
        Skips only ``test_benchmark_perf_*`` items in THIS directory, and only when the session
        collected at least one item from somewhere else. ``CPO_PERF_MEASURE=1`` (harvesting) and
        ``CPO_PERF_ALLOW_SHARED_SESSION=1`` both override.

        The skip is applied as a MARKER at collection, before any process group exists and
        identically on every rank -- it is a function of the collected item set, which is
        rank-invariant under one-launch-per-file. It is therefore not a divergent skip; a
        per-rank predicate here would deadlock the job rather than skip it.

    Args:
        session: The pytest session. Unused; part of the hook signature.
        config: The pytest config. Unused.
        items: Every collected item, mutated in place to carry the skip marker.

    Returns:
        None.
    """
    if os.environ.get("CPO_PERF_MEASURE", "0") == "1":
        return
    if os.environ.get("CPO_PERF_ALLOW_SHARED_SESSION", "0") == "1":
        return
    here, other = [], False
    for it in items:
        if _timed_gate(pathlib.Path(str(getattr(it, "path", "")))):
            here.append(it)
        else:
            other = True
    if not (here and other):
        return
    mark = pytest.mark.skip(
        reason="distributed perf gate: session also collected non-perf tests, so a pinned median "
        "would be read under conditions it was not pinned in. Run `pytest tests/distributed/perf/"
        "<file>.py` alone, or set CPO_PERF_ALLOW_SHARED_SESSION=1."
    )
    for it in here:
        it.add_marker(mark)


@pytest.fixture(scope="session", autouse=True)
def clock_locked():
    """Lock GPU clocks to max for the session; reset at teardown. Best-effort, one locker per node.

    Purpose: give every cell in this directory the same clock state, without making the run depend
    on having clock control. ``_clocklock`` warns and proceeds UNLOCKED when the attempt is denied,
    which is the common case without root and the case on both venues used here.

    Semantics: session-scoped and autouse -- locking per test would add seconds per cell for no
    benefit. Only ``LOCAL_RANK`` 0 touches the clocks, because every rank on a node would otherwise
    issue the same node-wide call and the last teardown would reset clocks another rank still
    wanted.

    IMPORTANT -- the repo harvests and gates FREE-RUNNING (CLAUDE.md: "do NOT lock the GPU clock").
    A locked pin compared against a free-running measurement compares two different distributions.
    This fixture is carried so the two directories behave alike and so a dedicated bench host is
    supported; it must not become the assumed state.

    Yields:
        bool -- True if the clocks were actually locked, False if the attempt was denied. A test may
        read it to widen its tolerance; none currently does.
    """
    if os.environ.get("LOCAL_RANK", "0") != "0":
        yield False
        return
    with clock_lock() as locked:
        yield locked


@pytest.fixture(scope="session", autouse=True)
def _gpu_warm(clock_locked, dist_manager):
    """Run sustained matmul before the first timed cell, so nothing is measured on a ramping part.

    Purpose: remove the "first cell measured while the clock ramps" failure mode. Ordered after
    ``clock_locked`` by depending on it, so the soak happens in the clock state the cells will see.

    Semantics: session-scoped and autouse; runs on EVERY rank, because every rank has its own GPU
    and an unwarmed rank is the one that sets a ``reduce="max"`` median. Allocates a single square
    fp16 operand and frees it before yielding, so the soak leaves no footprint for a cell to
    inherit. Skips silently when torch has no CUDA device -- collection on a GPU-free box must not
    fail here.

    **``dist_manager`` IS A DEPENDENCY FOR ORDERING, NOT FOR USE -- and dropping it silently
    re-breaks this fixture.** ``dist_manager`` is what calls ``torch.cuda.set_device(local_rank)``.
    Without that dependency this fixture ran BEFORE any device was pinned, so
    ``torch.cuda.current_device()`` was still **0 on every rank** and the soak ran on device 0 --
    correct code at the wrong moment. Measured with a ``pytest_fixture_setup`` hookwrapper::

        [fx] _gpu_warm      dev 0->0   ctx []  -> ['<device 0>']     <- BOTH ranks
        [fx] dist_manager   dev 0->1   ctx [.] -> [., '<device 1>']  <- pinned AFTERWARDS

    The consequences were two, and the second is worse. Every rank left a primary CUDA context on
    device 0 that outlives the soak (``empty_cache()`` reclaims the operand, never the context), so
    at 8 ranks/node device 0 carried 8 foreign contexts for the whole run; and the soak itself put
    8 ranks on one GPU for ``_WARM_SECONDS``, measured at 690/700 W with ``throttle 0x4 SW Power
    Cap`` while every other GPU idled at ~75 W. Meanwhile **the fixture was not doing its job at
    all** -- it warmed device 0 eight times and left every rank's real GPU cold, which is exactly
    the ramp it exists to prevent.

    **ORDER MATTERS AND IS LOAD-BEARING: ``clock_locked`` MUST stay first in this signature.** Its
    own docstring says it is deliberately ahead of the soak so the soak happens in the clock state
    the cells will see. pytest sets up same-scope independent fixtures in signature order, so
    swapping these two arguments silently inverts that -- no error, no test failure, just a soak in
    the wrong clock state.

    **On the soak's strength after the fix:** the wall time is unchanged, but each rank now has a
    whole GPU for it instead of a contended eighth of one, so the warming is stronger per rank, not
    weaker. There is no need to raise ``_WARM_SECONDS`` to compensate; if anything the previous
    value was calibrated against a contended device.

    Yields:
        None.
    """
    import torch

    if not torch.cuda.is_available():
        yield
        return
    import time

    dev = torch.device("cuda", torch.cuda.current_device())
    a = torch.randn(4096, 4096, device=dev, dtype=torch.float16)
    end = time.perf_counter() + _WARM_SECONDS
    while time.perf_counter() < end:
        for _ in range(20):
            a @ a
        torch.cuda.synchronize(dev)
    del a
    torch.cuda.empty_cache()
    yield


@pytest.fixture
def mesh_or_skip(world_size):
    """Return a callable that skips COHERENTLY when this launch cannot supply a mesh, and ledgers it.

    Purpose
        One launch fixes one ``WORLD_SIZE``, so no single run can cover the mesh axis. Coverage of
        that axis is a property of the SET of launches, which is what
        :mod:`fold_cp_ops.testing.coverage_ledger` makes inspectable -- every mesh spec is recorded
        as exercised or skipped-with-reason, and ``union_ledgers`` merges the runs.

    Semantics
        The returned callable takes a mesh spec and the rank count it needs. When this launch's
        ``WORLD_SIZE`` cannot supply it, the call records the spec as SKIPPED and raises through
        ``rank_invariant_skip`` -- rank-invariant because ``WORLD_SIZE`` is job-uniform by
        definition, so every rank reaches the same decision without needing to reduce. When it can,
        the spec is recorded as EXERCISED and the call returns None.

        A bare ``pytest.skip`` here would be a DEADLOCK, not a skip: the skipping rank leaves while
        its peers block in the next collective until a watchdog kills the job. That is what
        ``collective_guard`` enforces for everything under ``tests/distributed/``.

    Args:
        world_size: Injected fixture -- this launch's rank count, from ``dist_manager``.

    Returns:
        A callable ``(spec, needs, *, kernel) -> None``, where ``spec`` is the mesh (any value the
        ledger can key, e.g. ``("cp", (2, 8))``), ``needs`` is the rank count it requires (must be a
        positive int; a wrong value silently mis-reports coverage rather than failing), and
        ``kernel`` names the kernel whose ledger the record belongs to.

    Raises:
        Skipped: via ``rank_invariant_skip``, when ``needs != world_size``.
    """

    def _need(spec, needs: int, *, kernel: str) -> None:
        if needs != world_size:
            coverage_ledger.record_skipped(
                kernel, "mesh", spec, f"needs {needs} ranks, launch has {world_size}"
            )
            rank_invariant_skip(
                f"mesh {spec!r} needs {needs} ranks, this launch has {world_size}",
                because="WORLD_SIZE is fixed by the launcher and identical on every rank, so this "
                "predicate cannot differ between ranks",
            )
        coverage_ledger.record_exercised(kernel, "mesh", spec)

    return _need
