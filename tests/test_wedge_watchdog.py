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

"""Tests for ``tests/wedge_watchdog.py``.

The load-bearing test here is :func:`test_the_watchdog_survives_concurrent_frame_churn`, and it is
written as a **controlled** experiment rather than a bare assertion. The condition being guarded --
"dumping all thread tracebacks on a timer does not segfault a healthy process" -- is only meaningful
while the underlying CPython race is live, so the control arms the OLD, racy API first and the safe
assertion runs only once the control has demonstrated the crash. That is what keeps the guard from
passing vacuously on a CPython where the race has been fixed (at which point the control skips and
says so), and what keeps it from failing spuriously there either.

No GPU, no torch, no ``fold_cp_ops`` import: the defect these cover is pure CPython threading.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
import time

import pytest

from fold_cp_ops.testing.kernel_matrix import matrix_exempt
from tests.wedge_watchdog import (
    DEFAULT_TIMEOUT_SECONDS,
    DUMP_HEADER_PREFIX,
    TIMEOUT_ENV_VAR,
    WedgeWatchdog,
    install_wedge_watchdog,
    resolve_timeout,
)

#: Seconds the churn payload runs. The racy API crashes in 0.29-0.58 s (measured, 3 of 3, CPython
#: 3.12.13), so this is ~10x the observed worst case -- long enough that a surviving run is evidence
#: rather than luck, short enough to stay a unit test.
_CHURN_SECONDS = 5.0

#: Dump period inside the churn payload. Far below anything shipped (300 s): the point is to raise
#: the number of dump events by ~5 orders of magnitude so a race that fires once per ~37 real
#: sessions fires within seconds here.
_CHURN_PERIOD = 0.001

#: Payload run in a SUBPROCESS, because the failure mode under test is a SIGSEGV -- it cannot be
#: caught in-process, only observed as a return code. ``{arm}`` is the line that starts the dumper.
_CHURN_SOURCE = textwrap.dedent(
    """
    import faulthandler, sys, threading, time
    sys.path.insert(0, {repo!r})
    sink = open("/dev/null", "w")
    {arm}

    def churn(depth):
        if depth <= 0:
            return sum(x * x for x in range(8))
        return churn(depth - 1) + len([y for y in range(4)])

    def worker(deadline):
        while time.time() < deadline:
            for d in (1, 7, 31, 97, 211):
                churn(d)

    deadline = time.time() + {seconds!r}
    ts = [threading.Thread(target=worker, args=(deadline,), daemon=True) for _ in range(3)]
    for t in ts:
        t.start()
    worker(deadline)
    for t in ts:
        t.join()
    print("SURVIVED")
    """
)

_ARM_RACY = "faulthandler.dump_traceback_later({period!r}, repeat=True, file=sink)"
_ARM_SAFE = (
    "from tests.wedge_watchdog import install_wedge_watchdog\n"
    "install_wedge_watchdog({period!r}, file=sink)"
)


def _run_churn(tmp_path, arm: str, seconds: float, period: float):
    """Run the frame-churn payload in a subprocess with the given dumper armed.

    Args:
        tmp_path: pytest tmp dir to write the payload into. Must be writable.
        arm: Source line(s) starting the dumper, with ``{period}`` already formatted in. An empty
            string arms nothing, which is the no-dumper control.
        seconds: Wall seconds the payload churns for. Must be > 0.
        period: Dump period passed into ``arm``.

    Returns:
        The ``subprocess.CompletedProcess``. ``returncode == 0`` means the payload printed
        SURVIVED; ``-11`` (``-signal.SIGSEGV``) means it faulted, which is the condition under test.

    Raises:
        subprocess.TimeoutExpired: If the payload outlives ``seconds`` by a wide margin, which would
            mean it wedged rather than either surviving or crashing -- a third outcome the caller
            must not silently read as success.
    """
    repo = str(__import__("pathlib").Path(__file__).resolve().parent.parent)
    src = _CHURN_SOURCE.format(repo=repo, arm=arm.format(period=period), seconds=seconds)
    script = tmp_path / "churn_payload.py"
    script.write_text(src)
    return subprocess.run(
        [sys.executable, str(script)],
        capture_output=True,
        text=True,
        timeout=seconds + 120,
    )


@matrix_exempt("a CPython threading race; there is no kernel, shape or dtype to sweep")
def test_the_watchdog_survives_concurrent_frame_churn(tmp_path):
    """The replacement does not fault where the API it replaces does.

    This is the regression test at the exact triggering condition of the SIGSEGV that killed
    ``pytest tests/perf/``: an all-thread traceback dump firing on a timer while other threads run
    Python code. The control (the old ``faulthandler.dump_traceback_later``) must crash for the
    assertion to mean anything; if CPython has fixed the race, the control survives and this skips
    rather than passing on no evidence.
    """
    control = _run_churn(tmp_path, _ARM_RACY, _CHURN_SECONDS, _CHURN_PERIOD)
    if control.returncode == 0:
        pytest.skip(
            "positive control did not reproduce: faulthandler.dump_traceback_later survived "
            f"{_CHURN_SECONDS}s of frame churn on {sys.version.split()[0]}. The CPython race this "
            "guards appears fixed here, so the safe-path assertion would prove nothing."
        )
    assert control.returncode < 0, (
        "the control was expected to die on a signal (SIGSEGV) or survive; it exited "
        f"{control.returncode} instead:\n{control.stderr[-2000:]}"
    )

    safe = _run_churn(tmp_path, _ARM_SAFE, _CHURN_SECONDS, _CHURN_PERIOD)
    assert safe.returncode == 0, (
        f"install_wedge_watchdog faulted under the same churn the control died on "
        f"(rc={safe.returncode}); the GIL-held dump is supposed to make this safe.\n"
        f"{safe.stderr[-4000:]}"
    )
    assert "SURVIVED" in safe.stdout


@matrix_exempt("a source-level convention guard; there is no kernel, shape or dtype to sweep")
def test_the_racy_timer_is_not_armed_anywhere_in_the_repo():
    """No module may re-arm ``faulthandler.dump_traceback_later``.

    The crash it causes is rare (~1 in 37 sessions), lands on an unrelated test, and looks exactly
    like a kernel bug -- it cost a full investigation once. A grep is the only cheap way to keep it
    from coming back, since nothing about re-adding the call fails at the time it is written.

    Matched on the **AST**, not with a substring scan, so the conftests may keep explaining in a
    comment why they do not use it -- that prose is the reason the ban is legible at all. This file
    is excluded because it arms the call deliberately, as a positive control.
    """
    import ast
    import pathlib

    root = pathlib.Path(__file__).resolve().parent.parent
    offenders = []
    for py in list((root / "tests").rglob("*.py")) + list((root / "fold_cp_ops").rglob("*.py")):
        if py.resolve() == pathlib.Path(__file__).resolve():
            continue
        for node in ast.walk(ast.parse(py.read_text())):
            if not isinstance(node, ast.Call):
                continue
            fn = node.func
            called = fn.attr if isinstance(fn, ast.Attribute) else getattr(fn, "id", None)
            if called == "dump_traceback_later":
                offenders.append(f"{py.relative_to(root)}:{node.lineno}")
    assert not offenders, (
        "faulthandler.dump_traceback_later dumps from a C thread WITHOUT the GIL and races the "
        "frame chain of every running thread, segfaulting healthy sessions. Use "
        "tests.wedge_watchdog.install_wedge_watchdog instead. Offenders: " + ", ".join(offenders)
    )


@matrix_exempt("exercises a thread's own behaviour; there is no kernel, shape or dtype to sweep")
def test_the_watchdog_dumps_periodically_and_cancel_stops_it(tmp_path):
    """It actually dumps, on a period, and stops when cancelled.

    Guards the failure mode a purely negative test would miss: a watchdog that never dumps also
    never crashes, and would satisfy every other assertion here while providing no diagnostic.
    """
    sink_path = tmp_path / "dumps.txt"
    with open(sink_path, "w") as sink:
        wd = WedgeWatchdog(0.05, file=sink).start()
        deadline = time.time() + 10.0
        while wd.dump_count < 3 and time.time() < deadline:
            time.sleep(0.02)
        wd.cancel()
        after_cancel = wd.dump_count
        time.sleep(0.3)
        assert wd.dump_count == after_cancel, "the watchdog kept dumping after cancel()"

    text = sink_path.read_text()
    assert after_cancel >= 3, (
        f"expected >= 3 dumps within 10s at a 0.05s period, got {after_cancel}"
    )
    assert text.count(DUMP_HEADER_PREFIX) >= 3
    # all_threads=True, so this test's own thread must appear in the dump.
    assert "Thread 0x" in text or "Current thread 0x" in text


@matrix_exempt("validates argument handling; there is no kernel, shape or dtype to sweep")
def test_a_non_positive_period_is_refused_by_the_class_and_disables_the_installer():
    """``<= 0`` means "off" at the installer and is an error at the class.

    The two differ on purpose: a launcher setting the env var to 0 is asking for no watchdog, while
    code constructing ``WedgeWatchdog(0)`` directly has written a busy-spin and should hear about it.
    """
    for bad in (0, -1, -0.5):
        with pytest.raises(ValueError, match="must be > 0"):
            WedgeWatchdog(bad)
        assert install_wedge_watchdog(bad) is None


@matrix_exempt("validates env parsing; there is no kernel, shape or dtype to sweep")
def test_the_period_comes_from_the_env_and_a_bad_value_falls_back():
    """A typo in a launcher must not take down a test session, so parsing never raises."""
    assert resolve_timeout({}) == DEFAULT_TIMEOUT_SECONDS
    assert resolve_timeout({TIMEOUT_ENV_VAR: "12.5"}) == 12.5
    assert resolve_timeout({TIMEOUT_ENV_VAR: "0"}) == 0.0
    for junk in ("", "300s", "abc", "None"):
        assert resolve_timeout({TIMEOUT_ENV_VAR: junk}) == DEFAULT_TIMEOUT_SECONDS


@matrix_exempt("validates argument handling; there is no kernel, shape or dtype to sweep")
def test_a_file_without_a_descriptor_fails_at_construction():
    """A sink faulthandler cannot write to must fail loudly at install, not silently at first dump.

    ``faulthandler`` writes to the underlying descriptor, so an ``io.StringIO`` would raise inside
    the daemon thread, be swallowed by the loop's guard, and leave a watchdog that looks installed
    and never dumps -- the exact silent-diagnostic failure this check exists to prevent.
    """
    import io

    with pytest.raises((io.UnsupportedOperation, AttributeError)):
        WedgeWatchdog(1.0, file=io.StringIO())


#: Escalation payload. ``sys.setswitchinterval(10000)`` plus a tight loop is a GIL no other Python
#: thread can take -- a faithful stand-in for a rank parked inside ``nvshmem_barrier_all()``, which
#: nvshmem.pyx does NOT wrap ``with nogil:``. ``{hold}`` picks wedged (tight loop) vs healthy
#: (``time.sleep``, which releases the GIL).
_ESCALATE_SOURCE = textwrap.dedent(
    """
    import sys, time
    sys.path.insert(0, {repo!r})
    from tests.wedge_watchdog import WedgeWatchdog
    sink = open({out!r}, "w")
    wd = WedgeWatchdog(0.1, file=sink, escalate_after=2).start()   # dead-man deadline 0.2s
    {hold}
    sink.flush()
    print("done")
    """
)
_HOLD_WEDGED = (
    "sys.setswitchinterval(10000)\n"
    "t0 = time.time()\n"
    "while time.time() - t0 < 2.0: pass\n"
    "sys.setswitchinterval(0.005)"
)
_HOLD_HEALTHY = "time.sleep(2.0)\nwd.cancel()"


def _run_escalation(tmp_path, hold: str):
    """Run the escalation payload in a subprocess and return its dump text.

    Args:
        tmp_path: pytest tmp dir; must be writable.
        hold: Source for what the main thread does -- :data:`_HOLD_WEDGED` or
            :data:`_HOLD_HEALTHY`.

    Returns:
        ``(completed_process, dump_text)``. ``dump_text`` is everything the watchdog and any
        SIGALRM handler wrote, which is what the caller asserts against.
    """
    import pathlib

    out = tmp_path / "escalate_dumps.txt"
    script = tmp_path / "escalate_payload.py"
    repo = str(pathlib.Path(__file__).resolve().parent.parent)
    script.write_text(_ESCALATE_SOURCE.format(repo=repo, out=str(out), hold=hold))
    proc = subprocess.run(
        [sys.executable, str(script)], capture_output=True, text=True, timeout=180
    )
    return proc, (out.read_text() if out.exists() else "")


@matrix_exempt("a GIL/signal-delivery property; there is no kernel, shape or dtype to sweep")
def test_escalation_dumps_when_python_stops_being_scheduled(tmp_path):
    """The dead-man timer reports a wedge that no Python thread can report.

    This is the case `tests/distributed/conftest.py` needs and the plain watchdog cannot cover:
    ``nvshmem``'s ``barrier_all()`` holds the GIL, so a rank hung there never lets the watchdog
    thread run. The kernel delivers SIGALRM regardless, and faulthandler's signal-context handler
    dumps. Recognised by an UNHEADERED traceback: the periodic path always writes a
    ``WedgeWatchdog:`` header first, the SIGALRM path never does.
    """
    proc, text = _run_escalation(tmp_path, _HOLD_WEDGED)
    assert proc.returncode == 0, f"payload died: {proc.stderr[-2000:]}"
    dumps = text.count("Current thread 0x")
    headers = text.count(DUMP_HEADER_PREFIX)
    assert dumps > headers, (
        "expected at least one UNHEADERED (SIGALRM) dump while the GIL was held, but every dump "
        f"carried a periodic header ({dumps} dumps, {headers} headers). Escalation did not fire.\n"
        f"{text[:2000]}"
    )


@matrix_exempt("a GIL/signal-delivery property; there is no kernel, shape or dtype to sweep")
def test_escalation_never_fires_on_a_healthy_run(tmp_path):
    """Zero exposure while Python is being scheduled -- the whole point of kicking the timer.

    Without this, escalation would just be the racy dumper on a longer period, which is the defect
    this module exists to remove. 2.0 s at a 0.1 s period is 10x the 0.2 s dead-man deadline, so a
    timer that were not being re-armed would have fired many times over.
    """
    proc, text = _run_escalation(tmp_path, _HOLD_HEALTHY)
    assert proc.returncode == 0, f"payload died: {proc.stderr[-2000:]}"
    dumps = text.count("Current thread 0x")
    headers = text.count(DUMP_HEADER_PREFIX)
    assert dumps >= 5, f"the periodic watchdog should have dumped many times, got {dumps}"
    assert dumps == headers, (
        f"{dumps - headers} unheadered dump(s) -- SIGALRM fired on a HEALTHY run, so the dead-man "
        f"timer is not being kicked and the racy walk is running on a live process.\n{text[:2000]}"
    )


@matrix_exempt("validates argument handling; there is no kernel, shape or dtype to sweep")
def test_escalate_after_is_refused_below_one_and_defaults_off():
    """Escalation claims SIGALRM process-wide, so it is opt-in and its argument is validated."""
    for bad in (0, -1):
        with pytest.raises(ValueError, match="escalate_after"):
            WedgeWatchdog(1.0, escalate_after=bad)
    wd = WedgeWatchdog(1.0)
    assert wd._escalate_after is None, "escalation must be OFF unless asked for"
