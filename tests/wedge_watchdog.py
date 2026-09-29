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

"""Periodic all-thread traceback dump for a WEDGED test session, without the crash.

A long test session that hangs should name the wedge itself -- which thread, in which frame -- so a
stuck collective or an in-kernel deadlock is diagnosable from the job's stderr with no py-spy and no
second run. That is what this module provides, and it is a drop-in for the
``faulthandler.dump_traceback_later(timeout, repeat=True)`` the conftests used to arm.

**Why that API had to go: it segfaults healthy runs.** ``dump_traceback_later`` starts a *C* thread
(``Modules/faulthandler.c:faulthandler_thread``) which every ``timeout`` seconds calls
``_Py_DumpTracebackThreads(fd, interp, NULL)`` -- deliberately **without taking the GIL**, because
the same routine has to be safe to call from a fatal-signal handler. On CPython 3.11+ interpreter
frames live in chunked data stacks that are reallocated and freed as Python executes, so that walk
races every thread still running. Under a signal handler the race is harmless (the process is
already dying and nothing else is running); armed on a *healthy* process on a repeating timer it is
a live data race that eventually reads a torn frame and dereferences garbage.

Measured, on CPython 3.12.13: with the racy timer at a 1 ms period plus four threads doing ordinary
frame churn, the interpreter SIGSEGVs in **0.29-0.58 s, 3 runs of 3**, in
``dump_frame -> dump_traceback -> _Py_DumpTracebackThreads -> faulthandler_thread`` with the frame's
code pointer read as ``0x4``. At the shipped 300 s period the same race fired roughly once per 37
full test sessions -- rare enough to look like a kernel bug and expensive enough to lose a suite.

**The fix is the GIL.** ``faulthandler.dump_traceback()`` -- the immediate form -- is a plain
Python-callable C function, so the caller holds the GIL for the whole walk. No other thread can be
executing bytecode, hence none can be pushing or popping the frames being read. So the watchdog here
is an ordinary Python :class:`threading.Thread` that sleeps and then calls that safe form.

**What this costs, stated plainly.** A Python thread cannot run while some C extension holds the
GIL, so a wedge *of that kind* is invisible to the safe path alone. That is a real gap, not a
theoretical one -- it is measured below -- and it is why :meth:`WedgeWatchdog` takes an
``escalate_after``. Where the gap is real (the distributed conftest) escalation closes it; where it
is not (``tests/perf``, which never calls into nvshmem), escalation stays off so nothing claims
``SIGALRM``.

**Two things that were measured rather than assumed, because both are the obvious objection.**

*It does not perturb what it watches.* One dump holds the GIL for a **median 0.595 ms at 65 threads**
(0.002 ms at 1, 0.137 at 16, 0.902 at 128) -- against a 300 s period, and against perf gates that
median 9-15 samples per cell. The C-thread version held the GIL for none of that, so this is a real
difference; it is five orders of magnitude below the period.

*It still fires on time.* At a 0.1 s period over 5 s it dumped **49 of a nominal 50** with the main
thread parked in ``time.sleep`` (the GIL-free wedge case) and **47 of 50** against a tight Python
loop, so ordinary GIL contention does not starve it. Note separately that under pytest's default
``fd`` capture the dumps land in the *captured* stderr of whichever test is running and are shown
only if that test fails -- true of the API this replaces as well, since both resolve ``sys.stderr``
at arm time.

**The GIL-held wedge is real, and ``escalate_after`` is the answer to it.** Measured on the two
blocking calls a distributed rank actually wedges in: the host thread parked in
``cudaStreamSynchronize`` behind a hung in-kernel A2A leaves the GIL FREE (87 of 88 nominal dumps
still landed), and so does NCCL (4 of 5). But ``nvshmem/bindings/nvshmem.pyx`` shows
``cpdef barrier(team)`` and ``barrier_on_stream`` wrapped ``with nogil:`` while
**``barrier_all()`` and ``barrier_all_on_stream()`` are NOT** -- they hold the GIL for the whole
call. ``barrier_all`` is what ``DistributedManager.cleanup()`` calls, so a rank hung in a
cross-node ``barrier_all`` is precisely a wedge no Python thread can report.

So escalation is a **dead-man timer, kicked from the safe path**: a one-shot ``ITIMER_REAL`` armed
for ``escalate_after * period`` seconds and re-armed on every tick. While Python is being scheduled
the timer is always reset before it expires and ``SIGALRM`` never fires -- zero exposure on a
healthy run. When Python stops being scheduled the kernel delivers ``SIGALRM`` regardless of who
holds the GIL, and ``faulthandler``'s signal-context handler dumps. That dump uses the same
non-GIL walk this module exists to avoid, **deliberately**: it fires once, only after the process
has demonstrably stopped running Python, i.e. on a process already wedged and headed for the outer
``timeout``. Verified both properties directly: ``setitimer`` works from the watchdog's own
(non-main) thread, and with ``sys.setswitchinterval(10000)`` plus a tight loop -- a GIL no other
thread can take -- the ``SIGALRM`` dump still lands.
"""

from __future__ import annotations

import faulthandler
import os
import signal
import threading
from typing import IO, Optional

#: Seconds between dumps when the environment does not say otherwise.
DEFAULT_TIMEOUT_SECONDS = 300.0

#: Environment variable naming the dump period in seconds. ``<= 0`` disables the watchdog entirely;
#: an unparseable value falls back to :data:`DEFAULT_TIMEOUT_SECONDS` rather than raising, because a
#: typo in a launcher must not take down a test session.
TIMEOUT_ENV_VAR = "CPO_FAULTHANDLER_TIMEOUT"

#: Prefix of the line written immediately before each dump. Kept greppable and distinct so a dump in
#: a job log is attributable to this watchdog and not to a real fault.
DUMP_HEADER_PREFIX = "WedgeWatchdog"


class WedgeWatchdog:
    """A daemon thread that periodically dumps every thread's Python traceback, GIL-safely.

    Semantics: after each ``period`` of wall time it calls
    ``faulthandler.dump_traceback(file=..., all_threads=True)``. That call holds the GIL for the
    duration of the walk, which is the entire point -- see the module docstring. The dump is
    unconditional, exactly as the ``dump_traceback_later`` it replaces was: it does not attempt to
    detect whether the session is actually making progress, so a healthy long run emits a periodic
    heartbeat rather than nothing.

    The thread is a **daemon**, so it never delays interpreter shutdown and never needs joining.
    Writes go to the file's underlying **file descriptor** (``faulthandler`` bypasses Python-level
    buffering), which is what makes the output survive a process that is later killed.

    Args:
        period: Seconds between dumps. Must be ``> 0``; a zero or negative period would busy-spin
            the interpreter and is rejected, since "disabled" is expressed by not constructing one
            (see :func:`install_wedge_watchdog`).
        file: Destination, defaulting to ``sys.stderr`` when None. Must be a real file object with
            a working ``fileno()``; ``faulthandler`` writes to that descriptor, so an object that
            only implements ``write()`` (an ``io.StringIO``, a pytest capture shim) raises
            ``io.UnsupportedOperation`` here rather than at the first dump, where it would surface
            as a silently dead watchdog.
        escalate_after: Ticks of no progress after which the SIGALRM dead-man dump fires (see the
            module docstring). Must be ``>= 1`` or None. **Pass it only where a GIL-holding wedge
            is real**: it claims ``SIGALRM`` and ``ITIMER_REAL`` for the whole process, so it
            collides with anything else using them -- notably ``pytest-timeout``'s ``signal``
            method -- and the loser of that collision fails silently, whichever it is. None
            (the default) touches neither.

    Raises:
        ValueError: If ``period`` is not strictly positive, if ``escalate_after`` is ``< 1``, or if
            ``escalate_after`` is given on a platform without ``signal.setitimer``.
        io.UnsupportedOperation, AttributeError: If ``file`` has no usable file descriptor.
    """

    def __init__(
        self,
        period: float,
        file: Optional[IO[str]] = None,
        escalate_after: Optional[int] = None,
    ) -> None:
        if period <= 0:
            raise ValueError(f"WedgeWatchdog period must be > 0, got {period!r}")
        if escalate_after is not None and escalate_after < 1:
            raise ValueError(f"escalate_after must be >= 1 or None, got {escalate_after!r}")
        if escalate_after is not None and not hasattr(signal, "setitimer"):
            raise ValueError("escalate_after needs signal.setitimer, which this platform lacks")
        if file is None:
            import sys

            file = sys.stderr
        # Resolve the descriptor NOW so a file that cannot be dumped to fails at construction, where
        # the traceback names the caller, and not inside the daemon thread where it would be
        # swallowed and leave a watchdog that silently never dumps.
        self._fd = file.fileno()
        self._file = file
        self._period = float(period)
        self._escalate_after = escalate_after
        self._stop = threading.Event()
        self._dumps = 0
        self._thread = threading.Thread(
            target=self._run, name="fold-cp-ops-wedge-watchdog", daemon=True
        )

    def _kick(self) -> None:
        """Restart the kernel dead-man timer. No-op unless escalation is enabled.

        Semantics: arms a ONE-SHOT ``ITIMER_REAL`` for ``escalate_after * period`` seconds,
        replacing any previous arming. Called once at :meth:`start` and again on every tick, so on a
        healthy process the timer is always reset before it can expire and SIGALRM never fires.

        Returns:
            None. Emits a process-wide ``setitimer`` syscall as a side effect.
        """
        if self._escalate_after is not None:
            signal.setitimer(signal.ITIMER_REAL, self._escalate_after * self._period)

    @property
    def dump_count(self) -> int:
        """Number of dumps written so far.

        Returns:
            A monotonically non-decreasing count, readable from any thread. Written only by the
            watchdog thread and read as a plain int, so it needs no lock -- a reader may observe a
            value one dump stale, which is all any test needs.
        """
        return self._dumps

    def start(self) -> "WedgeWatchdog":
        """Start the daemon thread.

        Returns:
            ``self``, so construction and start compose on one line.

        Raises:
            RuntimeError: If called twice -- :class:`threading.Thread` refuses a restart. Calling it
                twice would otherwise silently mean two watchdogs interleaving dumps.
        """
        if self._escalate_after is not None:
            # chain=False: dump and continue. The process is wedged, not faulting; the outer
            # `timeout` is what ends it, and re-raising here would only obscure that.
            faulthandler.register(signal.SIGALRM, file=self._file, all_threads=True, chain=False)
            self._kick()
        self._thread.start()
        return self

    def cancel(self, timeout: float = 5.0) -> None:
        """Stop dumping and wait briefly for the thread to notice.

        Semantics: sets the stop event, which the thread observes at its next wake -- so a dump
        already in progress completes rather than being torn. Idempotent.

        Args:
            timeout: Seconds to wait for the thread to exit. Must be non-negative. A wait that
                expires is NOT an error and does not raise: the thread is a daemon, so the worst
                case is one further dump, never a hang at exit.

        Returns:
            None.
        """
        self._stop.set()
        if self._escalate_after is not None:
            signal.setitimer(signal.ITIMER_REAL, 0)  # disarm; a stale timer would dump post-cancel
        if self._thread.is_alive():
            self._thread.join(timeout)

    def _run(self) -> None:
        """Thread body: sleep, dump, repeat until cancelled.

        ``Event.wait`` returns True when the event is set, so the loop exits promptly on
        :meth:`cancel` instead of sleeping out the remaining period. Any exception from the dump
        ends the loop rather than propagating -- an unhandled exception in a diagnostic thread
        would print a confusing secondary traceback into the very log the dump is meant to clarify.

        Returns:
            None.
        """
        while not self._stop.wait(self._period):
            # Kick FIRST: reaching here proves Python is still being scheduled, which is exactly
            # the condition the dead-man timer tests for.
            self._kick()
            try:
                # CUMULATIVE elapsed, not the period. This used to print `self._period` on every
                # heartbeat, so a dump with three heartbeats read `45s / 45s / 45s` and a reader
                # wanting "how long was this rank parked here" had to DERIVE it as period x index.
                # That derivation is sound but it is not an observation, and the number it produces
                # is the one that decides whether a timeout was exceeded -- measured live: a rank
                # sat in `CollectiveGate.barrier` for two heartbeats against a 90 s bound, and
                # "at 90 s" was computed while being quoted as read. A diagnostic should not make
                # its reader do arithmetic to reach the fact it exists to report.
                elapsed = self._period * (self._dumps + 1)
                os.write(
                    self._fd,
                    f"\n{DUMP_HEADER_PREFIX}: {elapsed:g}s elapsed "
                    f"(heartbeat {self._dumps + 1}, period {self._period:g}s); "
                    f"dumping all thread tracebacks (diagnostic, not a fault)\n".encode(),
                )
                faulthandler.dump_traceback(file=self._file, all_threads=True)
            except Exception:
                return
            self._dumps += 1


def resolve_timeout(env: Optional[dict] = None) -> float:
    """Read the watchdog period from the environment.

    Args:
        env: Mapping to read :data:`TIMEOUT_ENV_VAR` from; ``os.environ`` when None. Taking it as an
            argument is what lets the unit test cover the parsing without mutating global state.

    Returns:
        The configured period in seconds. :data:`DEFAULT_TIMEOUT_SECONDS` when the variable is
        absent or unparseable; ``<= 0`` means the caller should not install a watchdog at all.
    """
    if env is None:
        env = os.environ
    raw = env.get(TIMEOUT_ENV_VAR)
    if raw is None:
        return DEFAULT_TIMEOUT_SECONDS
    try:
        return float(raw)
    except (TypeError, ValueError):
        return DEFAULT_TIMEOUT_SECONDS


def install_wedge_watchdog(
    period: Optional[float] = None,
    file: Optional[IO[str]] = None,
    escalate_after: Optional[int] = None,
) -> Optional[WedgeWatchdog]:
    """Enable fatal-signal dumping and start the periodic watchdog, if it is enabled.

    This is the whole conftest-facing API: it replaces the ``faulthandler.enable()`` +
    ``faulthandler.dump_traceback_later(t, repeat=True)`` pair. ``faulthandler.enable()`` is kept
    unchanged and unconditionally -- installing handlers for SIGSEGV/SIGBUS/SIGFPE/SIGABRT/SIGILL is
    safe (they only run once the process is already faulting) and is what puts a Python-level stack
    next to the native one.

    Args:
        period: Seconds between dumps. When None, read from :data:`TIMEOUT_ENV_VAR` via
            :func:`resolve_timeout`. A value ``<= 0`` installs no watchdog and is the supported way
            to switch it off; it is not an error.
        file: Destination for the dumps, defaulting to ``sys.stderr``. Must have a real
            ``fileno()`` -- see :class:`WedgeWatchdog`.
        escalate_after: Ticks of no progress after which the SIGALRM dead-man dump fires; None
            disables escalation. **Only pass this where a GIL-holding wedge is real**, because it
            claims ``SIGALRM``/``ITIMER_REAL`` process-wide -- see :class:`WedgeWatchdog`.

    Returns:
        The started :class:`WedgeWatchdog`, or None when the period is ``<= 0``. Callers that need
        to stop it (tests) keep the handle; conftests deliberately do not, because the thread is a
        daemon and outlives them harmlessly.
    """
    faulthandler.enable()
    if period is None:
        period = resolve_timeout()
    if period <= 0:
        return None
    return WedgeWatchdog(period, file=file, escalate_after=escalate_after).start()
