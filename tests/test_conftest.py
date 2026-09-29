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
"""Tests for ``tests/conftest.py`` -- specifically its two session-record mechanisms.

The subject is the pair ``pytest_report_teststatus`` (the live progress record) and
``pytest_terminal_summary`` (the end-of-session report). Both are pytest hooks, so they cannot be
called directly with any fidelity: what they promise is about what a *session* leaves on disk,
including a session that dies. Every test here therefore runs a real pytest in a SUBPROCESS over a
generated test file and inspects the files that survive it.

The conftest is loaded into the subprocess with ``-p tests.conftest`` rather than by placing the
generated file under ``tests/``. Two reasons: the generated module would otherwise be collected by
ordinary runs of the suite, and a file physically under ``tests/kernels`` or ``tests/perf`` would be
picked up by the kernel-matrix audit and fail for an unrelated reason.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

#: Filename written by the conftest's live-progress hook, kept in sync with `conftest._PROGRESS_NAME`
#: by :func:`test_the_progress_filename_matches_the_conftest`.
PROGRESS_NAME = "run_progress.txt"


def _run_pytest(body: str, tmp_path: Path, *, extra_args=()):
    """Run a real pytest subprocess over a generated test module, with our conftest loaded.

    Args:
        body: Source of the generated test module. Written verbatim to ``tmp_path/test_generated.py``.
            It must be self-contained -- the subprocess does not import the repo's test helpers --
            and any test in it that is meant to crash the session should call ``os._exit`` so that
            no exit handler, and therefore no terminal-summary hook, gets a chance to run.
        tmp_path: A writable directory, used both for the generated module and (under ``bt/``) as
            pytest's ``--basetemp``. Pinning basetemp is what makes the report and progress paths
            predictable; without it they land in a numbered directory chosen by the child.
        extra_args: Extra pytest arguments, appended after the generated module.

    Returns:
        ``(completed_process, basetemp)`` where ``basetemp`` is the ``Path`` the child used. Note
        the child's numbered run directory is ``basetemp/<name>-0`` style only when pytest picks it;
        with an explicit ``--basetemp`` the directory IS ``basetemp``, which is why the progress and
        report paths below are formed relative to it directly.

    Raises:
        Nothing. A non-zero exit status is an expected outcome for several of these tests and is
        returned for the caller to assert on.
    """
    mod = tmp_path / "test_generated.py"
    mod.write_text(body)
    basetemp = tmp_path / "bt"
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        str(mod),
        "-p",
        "tests.conftest",
        "-q",
        "-p",
        "no:cacheprovider",
        f"--basetemp={basetemp}",
        *extra_args,
    ]
    env = dict(os.environ, PYTHONPATH=str(REPO_ROOT))
    proc = subprocess.run(cmd, cwd=REPO_ROOT, env=env, capture_output=True, text=True, timeout=300)
    return proc, basetemp


def _progress_lines(basetemp: Path):
    """Read the live-progress record a child session left behind.

    Args:
        basetemp: The ``--basetemp`` handed to the child.

    Returns:
        A list of non-empty lines. An absent file yields ``[]`` rather than raising, so a test can
        assert the difference between "no record" and "an incomplete record" itself.
    """
    path = basetemp / PROGRESS_NAME
    if not path.exists():
        return []
    return [ln for ln in path.read_text().splitlines() if ln.strip()]


def test_the_progress_filename_matches_the_conftest():
    """The name this module reads is the name the conftest writes.

    A test that reads a hardcoded filename passes forever after the conftest renames the file --
    it would find no record and, by the convention in :func:`_progress_lines`, see an empty list,
    which several assertions below would misread. Bind the two names together instead.
    """
    from tests import conftest

    assert conftest._PROGRESS_NAME == PROGRESS_NAME


def test_a_passing_session_records_every_test_as_it_goes(tmp_path):
    """Each test appears in the progress file, in execution order, with setup before call."""
    proc, basetemp = _run_pytest(
        "def test_one(): pass\ndef test_two(): pass\ndef test_three(): pass\n", tmp_path
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    lines = _progress_lines(basetemp)
    ids = [ln.split()[-1].split("::")[-1] for ln in lines]
    assert ids == ["test_one", "test_one", "test_two", "test_two", "test_three", "test_three"]
    assert [ln.split()[0] for ln in lines[:2]] == ["setup", "call"]
    assert all(ln.split()[1] == "passed" for ln in lines)


def test_a_passing_teardown_is_not_recorded(tmp_path):
    """Two lines per passing test, not three -- the passing teardown carries nothing new.

    This is the one filtering rule in the hook, so it is asserted directly rather than inferred
    from a count: if the rule is dropped the file grows by 50% and this test says which rule went.
    """
    _, basetemp = _run_pytest("def test_only(): pass\n", tmp_path)
    assert [ln.split()[0] for ln in _progress_lines(basetemp)] == ["setup", "call"]


def test_a_failure_is_recorded_with_its_outcome(tmp_path):
    """A failing call is recorded as failed.

    The filter is on the (phase, outcome) PAIR, not on the test's verdict, so a failing test whose
    teardown succeeds still gets no teardown line. Asserted explicitly because the natural misreading
    -- "failures are recorded in full" -- is wrong, and a later reader chasing a missing teardown
    line should find the answer here rather than in the hook.
    """
    _, basetemp = _run_pytest("def test_bad(): assert False\n", tmp_path)
    lines = _progress_lines(basetemp)
    phases = {ln.split()[0]: ln.split()[1] for ln in lines}
    assert phases == {"setup": "passed", "call": "failed"}


def test_a_skip_is_recorded(tmp_path):
    """A skip reports at setup, and that is the line that lands."""
    _, basetemp = _run_pytest(
        "import pytest\n@pytest.mark.skip(reason='deliberate')\ndef test_s(): pass\n", tmp_path
    )
    lines = _progress_lines(basetemp)
    assert any(ln.startswith("setup") and "skipped" in ln for ln in lines)


def test_the_record_survives_a_session_that_dies_mid_test(tmp_path):
    """The point of the whole mechanism: a hard exit leaves the record, and names the victim.

    ``os._exit`` skips every exit handler, so pytest's terminal-summary hook never runs -- the same
    situation as the core dump that motivated this. Two things must hold, and the second is the one
    that saves the time: the progress file exists at all, and its LAST line names the test that was
    executing rather than the last one that finished.
    """
    body = (
        "import os\n"
        "def test_first(): pass\n"
        "def test_second(): pass\n"
        "def test_boom(): os._exit(70)\n"
        "def test_never_runs(): pass\n"
    )
    proc, basetemp = _run_pytest(body, tmp_path)
    assert proc.returncode == 70, f"expected the hard exit, got {proc.returncode}"

    lines = _progress_lines(basetemp)
    assert lines, "the session died and left no progress record at all"
    assert lines[-1].split()[-1].endswith("::test_boom"), (
        f"the last line should name the test that was RUNNING, got: {lines[-1]!r}"
    )
    assert lines[-1].split()[0] == "setup", "the surviving line for the victim is its setup line"
    assert not any("test_never_runs" in ln for ln in lines)
    assert any(ln.split()[-1].endswith("::test_first") for ln in lines)


def test_the_end_of_session_report_does_not_survive_that_same_crash(tmp_path):
    """The control for the test above -- otherwise it proves nothing new.

    If the terminal-summary report happened to be written anyway, the live record would be
    redundant and this whole item would be unjustified. It is not: the same crashing session leaves
    no report, which is precisely the gap. Asserting it here means that if pytest ever starts
    writing the summary on a hard exit, this test fails and says the mechanism is now redundant --
    a much better outcome than quietly carrying a second file forever.
    """
    body = "import os\ndef test_boom(): os._exit(70)\n"
    _, basetemp = _run_pytest(body, tmp_path)
    assert not (basetemp / "run_report.txt").exists()
    assert _progress_lines(basetemp), "but the live record IS there"


def test_the_two_files_are_independent(tmp_path):
    """A completed session writes both, and neither is a prefix or a rewrite of the other.

    The design constraint from the plan was that adding the live record must not leave two files to
    keep in sync. This asserts the shape that makes that true: separate names, separate content,
    the summary written whole at the end and the progress file never rewritten.
    """
    proc, basetemp = _run_pytest("def test_ok(): pass\ndef test_bad(): assert 0\n", tmp_path)
    assert proc.returncode != 0
    report = (basetemp / "run_report.txt").read_text()
    progress = (basetemp / PROGRESS_NAME).read_text()
    assert "## counts" in report and "## counts" not in progress
    assert "call      failed" in progress.replace("   ", "   ") or "failed" in progress
    assert not report.startswith(progress) and not progress.startswith(report)


@pytest.mark.parametrize("phase", ["setup", "teardown"])
def test_an_error_outside_the_call_phase_is_recorded(tmp_path, phase):
    """Fixture errors land too -- they are the failures that a call-only record would drop."""
    body = (
        "import pytest\n"
        "@pytest.fixture\n"
        "def f():\n"
        f"    {'raise RuntimeError(1)' if phase == 'setup' else 'yield'}\n"
        f"    {'' if phase == 'setup' else 'raise RuntimeError(1)'}\n"
        "def test_uses(f): pass\n"
    )
    _, basetemp = _run_pytest(body, tmp_path)
    lines = _progress_lines(basetemp)
    assert any(ln.startswith(phase) and "failed" in ln for ln in lines), lines


@pytest.mark.parametrize("rank_var", ["LOCAL_RANK", "RANK", "WORLD_SIZE"])
def test_xdist_is_refused_inside_a_torchrun_session(tmp_path, rank_var, monkeypatch):
    """``-n`` under torchrun must be REFUSED, because both of them assign GPUs.

    Purpose
        The conftest hands each xdist worker its own GPU through ``CUDA_VISIBLE_DEVICES``; torchrun
        maps ``LOCAL_RANK`` to a device. Two assigners in one process tree do not compose -- neither
        knows about the other, so ranks land on the same GPU. That is an OOM at best and a silently
        shared device at worst, which corrupts every measurement taken on it.

    Semantics
        Asserts the REFUSAL, not the assignment, and that is the point: a silent skip would leave
        every worker on device 0 under torchrun, which is the same resource conflict with the
        diagnostic removed. Each of the three torchrun variables is checked separately because a
        launcher that sets only one of them must still trip the guard.

        The message is asserted too, since a `UsageError` that does not name ``torchrun`` sends the
        reader looking at their ``-n`` value instead of at how they launched.

    Args:
        tmp_path: Writable directory for the generated module and basetemp.
        rank_var: Which torchrun variable to set. Any one must be sufficient.
        monkeypatch: Used to set the variable for the CHILD process via the inherited environment.

    Returns:
        None.
    """
    monkeypatch.setenv(rank_var, "0")
    monkeypatch.setenv("PYTEST_XDIST_WORKER", "gw0")
    proc, _ = _run_pytest("def test_ok(): pass\n", tmp_path)
    assert proc.returncode != 0, "xdist inside a torchrun session must not run"
    combined = proc.stdout + proc.stderr
    assert "torchrun" in combined, f"the refusal must name torchrun; got:\n{combined[-600:]}"


def test_the_session_gets_its_own_autotune_result_cache(tmp_path, monkeypatch):
    """A test session must never read or write the developer's real autotune result cache.

    Purpose
        The result cache persists sweep outcomes across processes and defaults ON, matching `main`.
        For a suite that is wrong twice over: a run would be handed whatever timings the developer
        last measured, and it would write FAKE ones back -- ``tests/_internal/autotune/test_tuner.py``
        builds kernels whose candidates cost 1.0 and 2.0 by construction, and its fake kernels share
        a name, a shape and a candidate set, which is the entire cache key.

    Semantics
        Measured, not argued: that file passes 19/19 into a fresh directory and fails 3 on an
        immediate re-run against the same one. This asserts the mechanism that prevents it -- the
        child session points ``CPO_AUTOTUNE_CACHE_DIR`` at a directory of its own, not at the
        default under ``$CPO_HOME`` -- and that an operator's explicit setting is left alone, since
        a conftest that overrode it would make the variable undebuggable.

    Args:
        tmp_path: Writable directory for the generated module and basetemp.
        monkeypatch: Used to clear, then to pin, the variable in the CHILD's inherited environment.

    Returns:
        None.
    """
    body = (
        "import os\n\n\n"
        "def test_report():\n"
        "    print('AUTOTUNE_DIR=' + os.environ.get('CPO_AUTOTUNE_CACHE_DIR', '<unset>'))\n"
    )
    monkeypatch.delenv("CPO_AUTOTUNE_CACHE_DIR", raising=False)
    proc, _ = _run_pytest(body, tmp_path, extra_args=("-s",))
    line = [ln for ln in proc.stdout.splitlines() if ln.startswith("AUTOTUNE_DIR=")]
    assert line, f"the generated test did not report; got:\n{proc.stdout[-600:]}"
    got = line[0].split("=", 1)[1]
    assert got not in ("", "<unset>"), "the session must PIN the autotune cache dir, not inherit it"
    assert "fold_cp_ops_autotune_" in got, f"expected a session-private directory, got {got!r}"

    pinned = tmp_path / "operator_pinned"
    monkeypatch.setenv("CPO_AUTOTUNE_CACHE_DIR", str(pinned))
    # The marker is deliberately LEFT INHERITED. Clearing it here would make this assertion pass
    # under the old constant-"1" marker too, which is the whole defect -- a regression test that
    # cannot fail on the bug it was written for.
    second = tmp_path / "second"
    second.mkdir()
    proc, _ = _run_pytest(body, second, extra_args=("-s",))
    line = [ln for ln in proc.stdout.splitlines() if ln.startswith("AUTOTUNE_DIR=")]
    assert line and line[0].split("=", 1)[1] == str(pinned), (
        "an explicitly-set CPO_AUTOTUNE_CACHE_DIR must be left alone; got "
        f"{line[0] if line else '<no line>'}"
    )


def test_a_child_inheriting_OUR_marker_mints_its_own_autotune_cache(tmp_path, monkeypatch):
    """The mirror of the test above, and the reason the marker carries a PATH rather than "1".

    Purpose
        Pin the xdist half. `pytest_configure` gives a session its own autotune result cache and
        uses ``CPO_AUTOTUNE_CACHE_IS_OURS`` to tell "we minted this" from "the operator pinned it".
        Until this test existed that distinction lived only in a comment, and the comment was not
        enough: the marker was the constant ``"1"``, every descendant inherited it, and the
        neighbouring test -- which asserts a pin IS respected -- failed on every run, including on
        clean ``main``.

    Semantics
        Both cases are a child that inherits a SET ``CPO_AUTOTUNE_CACHE_DIR``; only the marker
        separates them, so both must be asserted or a "fix" satisfies one by breaking the other:

        * marker == the directory  -> the child is one of ours (an xdist worker) and must mint a
          FRESH cache. Sharing one is not hypothetical -- measured 2026-08-31, two
          ``tests/_internal/autotune/test_tuner.py`` tests failed 3 of 3 runs under ``-n 8`` while
          the same scope passed 1266/1266 serial, because a sibling worker's timings were loaded
          and the injected failures never fired.
        * marker != the directory  -> somebody pinned it; leave it alone (the test above).

    Args:
        tmp_path: Writable directory for the generated module and basetemp.
        monkeypatch: Sets the inherited environment the child sees.

    Returns:
        None.
    """
    body = (
        "import os\n\n\n"
        "def test_report():\n"
        "    print('AUTOTUNE_DIR=' + os.environ.get('CPO_AUTOTUNE_CACHE_DIR', '<unset>'))\n"
    )
    # Exactly what an xdist worker inherits: a directory the controller minted, vouched for by a
    # marker naming that same directory.
    inherited = tmp_path / "controller_minted"
    monkeypatch.setenv("CPO_AUTOTUNE_CACHE_DIR", str(inherited))
    monkeypatch.setenv("CPO_AUTOTUNE_CACHE_IS_OURS", str(inherited))

    proc, _ = _run_pytest(body, tmp_path, extra_args=("-s",))
    line = [ln for ln in proc.stdout.splitlines() if ln.startswith("AUTOTUNE_DIR=")]
    assert line, f"the generated test did not report; got:\n{proc.stdout[-600:]}"
    got = line[0].split("=", 1)[1]
    assert got != str(inherited), (
        "a child inheriting OUR OWN marker must mint a fresh cache, not reuse the inherited one -- "
        "that is how xdist workers end up sharing one result cache and reading each other's timings"
    )
    assert "fold_cp_ops_autotune_" in got, f"expected a freshly minted directory, got {got!r}"
