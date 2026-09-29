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

"""Tests for `fold_cp_ops.testing.collective_guard`.

Every rejection test is paired with an ACCEPTANCE test on its nearest legal neighbour. A guard that
flagged everything would satisfy a one-sided suite and then fail the whole distributed directory on
landing, so "it refuses the bad form" is only half the property -- "and admits the good one" is the
other half, and it is the half that a too-eager pattern breaks first.

No GPU and no process group needed: the AST half is pure text, and the runtime half is exercised by
driving `_group_is_live` through a monkeypatched ``torch.distributed`` rather than by standing up a
real group. That is deliberate -- a guard whose own tests need a 2-rank launch is a guard nobody runs.
"""

from __future__ import annotations

import textwrap

import pytest

from fold_cp_ops.testing import collective_guard as cg
from fold_cp_ops.testing.kernel_matrix import matrix_exempt
from fold_cp_ops.testing.numeric_guard import numeric_exempt


def _write(tmp_path, relpath: str, body: str):
    """Write a module at a path whose PARTS decide scope, and return it.

    Args:
        tmp_path: pytest's per-test temp dir.
        relpath: Path relative to it, e.g. ``"tests/distributed/test_x.py"``. The directory names
            matter -- `collective_scope` reads them -- so a test that wants out-of-scope behaviour
            must say so in the path, not in a flag.
        body: Module source; dedented before writing, so callers can indent it naturally.

    Returns:
        The `pathlib.Path` written.
    """
    p = tmp_path / relpath
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(textwrap.dedent(body))
    return p


# ── scope ─────────────────────────────────────────────────────────────────────────────────────────


@matrix_exempt("pure path algebra on the guard's own scope rule; no kernel, shape or dtype")
@numeric_exempt("scope is a boolean about a path, not a computed tensor")
@pytest.mark.parametrize(
    "relpath,expected",
    [
        ("tests/distributed/test_x.py", True),
        # The point of directory scoping: a subdirectory that does not exist yet is already covered,
        # which is where the A2A kernel tests will land.
        ("tests/distributed/kernels/test_x_a2a.py", True),
        ("tests/distributed/kernels/deeper/test_y.py", True),
        ("tests/kernels/test_gemm.py", False),
        ("tests/perf/test_benchmark_perf_gemm.py", False),
    ],
)
def test_scope_follows_the_directory_at_any_depth(relpath, expected):
    """A module is in scope iff some path component is a governed directory."""
    assert cg.collective_scope(relpath) is expected


# ── the AST half ──────────────────────────────────────────────────────────────────────────────────


@matrix_exempt("pure AST inspection; no kernel, shape or dtype to sweep")
@numeric_exempt("the subject is a list of problem strings, not a computed tensor")
@pytest.mark.parametrize(
    "call",
    ["pytest.skip('x')", "skip('x')"],
    ids=["dotted", "bare-imported"],
)
def test_a_bare_skip_in_scope_is_flagged(tmp_path, call):
    """Both spellings are caught. ``from pytest import skip`` would otherwise walk straight past."""
    p = _write(
        tmp_path,
        "tests/distributed/test_bad.py",
        f"""
        import pytest
        def test_x():
            {call}
        """,
    )
    problems = cg.divergent_skip_problems(p)
    assert len(problems) == 1, problems
    assert "bare pytest.skip()" in problems[0]
    assert "gated_skip" in problems[0], "the message must name the fix, not just the violation"


@matrix_exempt("pure AST inspection; no kernel, shape or dtype to sweep")
@numeric_exempt("the subject is a list of problem strings, not a computed tensor")
@pytest.mark.parametrize(
    "call",
    [
        "gated_skip(None)",
        "rank_invariant_skip('x', because='WORLD_SIZE is one number for the job')",
        "pre_init_skip('x', because='runs before initialize()')",
    ],
    ids=sorted(cg.SANCTIONED_SKIPS),
)
def test_every_sanctioned_spelling_is_admitted(tmp_path, call):
    """The acceptance half. A guard that flagged these would fail the directory on landing."""
    p = _write(
        tmp_path,
        "tests/distributed/test_good.py",
        f"""
        from fold_cp_ops.testing.collective_guard import (
            gated_skip, pre_init_skip, rank_invariant_skip,
        )
        def test_x():
            {call}
        """,
    )
    assert cg.divergent_skip_problems(p) == []


@matrix_exempt("pure AST inspection; no kernel, shape or dtype to sweep")
@numeric_exempt("the subject is a list of problem strings, not a computed tensor")
def test_a_bare_skip_OUTSIDE_the_directory_is_not_this_guard_s_business(tmp_path):
    """Scope is a real boundary: the same source is clean under tests/kernels/.

    Without this, "the guard flags bare skips" and "the guard flags bare skips in scope" would be
    indistinguishable, and the first would fail ~90 sites across the suite.
    """
    body = """
        import pytest
        def test_x():
            pytest.skip('x')
        """
    assert cg.divergent_skip_problems(_write(tmp_path, "tests/kernels/test_k.py", body)) == []
    assert cg.divergent_skip_problems(_write(tmp_path, "tests/distributed/test_d.py", body)) != []


@matrix_exempt("pure AST inspection; no kernel, shape or dtype to sweep")
@numeric_exempt("the subject is a list of problem strings, not a computed tensor")
def test_a_syntax_error_is_pytest_s_to_report_not_the_guard_s(tmp_path):
    """An unparseable module yields no problems, so the guard cannot mask a collection error."""
    p = _write(tmp_path, "tests/distributed/test_broken.py", "def test_x(: pass\n")
    assert cg.divergent_skip_problems(p) == []


# ── the declaration ───────────────────────────────────────────────────────────────────────────────


@matrix_exempt("argument validation; no kernel, shape or dtype to sweep")
@numeric_exempt("the subject is a raised ValueError, not a computed tensor")
@pytest.mark.parametrize("fn", [cg.rank_invariant_skip, cg.pre_init_skip])
@pytest.mark.parametrize("because", ["", "   "], ids=["empty", "whitespace"])
def test_a_declaration_with_no_reason_is_refused(fn, because):
    """``because=`` must be written. Existence is checked; truth cannot be, and is not claimed."""
    with pytest.raises(ValueError, match="reason"):
        fn("x", because=because)


# ── the rule: pre_init_skip is self-policing ───────────────────────────────────────────────


@matrix_exempt("drives a monkeypatched dist state; no kernel, shape or dtype to sweep")
@numeric_exempt("the subject is a raised AssertionError, not a computed tensor")
def test_pre_init_skip_REFUSES_to_run_once_a_group_exists(monkeypatch):
    """The rule, as code: past ``initialize()`` with a live group, the exemption does not apply.

    This is what makes ``pre_init_skip`` an exemption that cannot be misapplied by writing a
    convincing ``because=``. A free-text reason is checked by a reader who may never come; this is
    checked by the interpreter, every run.
    """
    monkeypatch.setattr(cg, "_group_is_live", lambda: True)
    with pytest.raises(AssertionError, match="not pre-init"):
        cg.pre_init_skip("x", because="claims to be pre-init but a group is up")


@matrix_exempt("drives a monkeypatched dist state; no kernel, shape or dtype to sweep")
@numeric_exempt("the subject is a raised Skipped, not a computed tensor")
def test_pre_init_skip_DOES_skip_when_there_is_genuinely_no_group(monkeypatch):
    """The acceptance half: with no group it is an ordinary skip, which is the whole point."""
    monkeypatch.setattr(cg, "_group_is_live", lambda: False)
    with pytest.raises(BaseException) as exc:
        cg.pre_init_skip("no group here", because="runs before initialize()")
    assert exc.typename == "Skipped", f"expected a pytest skip, got {exc.typename}"


# ── the runtime tripwire ──────────────────────────────────────────────────────────────────────────


@matrix_exempt("drives the tripwire directly; no kernel, shape or dtype to sweep")
@numeric_exempt("the subject is the guard's violation buffer, not a computed tensor")
def test_the_tripwire_records_a_bare_skip_only_when_a_group_is_LIVE(monkeypatch):
    """Conditional by design: a bare skip with no group is legal and must stay silent.

    Both directions are asserted from one place because the conditional IS the behaviour -- a
    tripwire that fired unconditionally would fail every collection-time skip in the directory, and
    one that never fired would be indistinguishable from not being installed.
    """
    cg.install_skip_tripwire()
    import pytest as _pytest

    monkeypatch.setattr(cg, "_group_is_live", lambda: False)
    cg.reset_violations()
    with pytest.raises(BaseException):
        _pytest.skip("no group -> allowed")
    assert list(cg.drain_violations()) == [], "a bare skip without a group must not be recorded"

    monkeypatch.setattr(cg, "_group_is_live", lambda: True)
    cg.reset_violations()
    with pytest.raises(BaseException):
        _pytest.skip("live group -> recorded")
    recorded = list(cg.drain_violations())
    assert len(recorded) == 1, recorded
    assert "LIVE torch.distributed group" in recorded[0]
    assert "pre_init_skip` does not apply" in recorded[0]


@matrix_exempt("drives the tripwire directly; no kernel, shape or dtype to sweep")
@numeric_exempt("the subject is the guard's violation buffer, not a computed tensor")
def test_a_sanctioned_helper_is_not_recorded_even_under_a_live_group(monkeypatch):
    """The re-entry flag: the helpers call ``pytest.skip`` themselves and must not trip their own wire.

    Without this the guard would fail every correctly-written site, which is the failure mode that
    reads as "the guard works" right up until the suite is red.
    """
    cg.install_skip_tripwire()
    monkeypatch.setattr(cg, "_group_is_live", lambda: True)
    cg.reset_violations()
    with pytest.raises(BaseException):
        cg.rank_invariant_skip("uniform", because="one number for the whole job")
    assert list(cg.drain_violations()) == []


@matrix_exempt("drives the tripwire directly; no kernel, shape or dtype to sweep")
@numeric_exempt("the subject is the guard's violation buffer, not a computed tensor")
def test_the_reentry_flag_is_cleared_even_when_the_helper_raises(monkeypatch):
    """``_IN_SANCTIONED_SKIP`` must not latch: the helper's exit is a raise, every time.

    It is set around a call that ALWAYS raises ``Skipped``, so a non-``finally`` release would leave
    it True forever and silently disarm the tripwire for the rest of the session -- passing this
    file's other tests while enforcing nothing.
    """
    cg.install_skip_tripwire()
    monkeypatch.setattr(cg, "_group_is_live", lambda: True)
    with pytest.raises(BaseException):
        cg.rank_invariant_skip("uniform", because="one number for the whole job")
    assert cg._IN_SANCTIONED_SKIP is False

    cg.reset_violations()
    import pytest as _pytest

    with pytest.raises(BaseException):
        _pytest.skip("a bare skip AFTER the helper must still be caught")
    assert len(list(cg.drain_violations())) == 1, "the flag latched; the tripwire is now disarmed"


@matrix_exempt("drives the tripwire directly; no kernel, shape or dtype to sweep")
@numeric_exempt("the subject is a status string, not a computed tensor")
def test_disarming_the_tripwire_is_itself_reported():
    """The anti-evasion check, and its acceptance half.

    ``monkeypatch.setattr(pytest, "skip", orig)`` in a test would turn the guard off for every test
    after it; checking at teardown attributes that to the test that did it. Restoration is done and
    undone by hand here rather than with monkeypatch, because monkeypatch's own teardown would
    reverse it before the assertion could run.
    """
    cg.install_skip_tripwire()
    assert cg.skip_tripwire_problem() is None, "armed tripwire must report clean"

    owner, attr, poison, original = cg._POISONED["pytest.skip"]
    setattr(owner, attr, original)
    try:
        problem = cg.skip_tripwire_problem()
        assert problem is not None and "disarmed" in problem
    finally:
        setattr(owner, attr, poison)
    assert cg.skip_tripwire_problem() is None, "re-arming must clear the report"


@matrix_exempt("pure bookkeeping on the guard's buffer; no kernel, shape or dtype to sweep")
@numeric_exempt("the subject is the guard's violation buffer, not a computed tensor")
def test_draining_attributes_a_violation_to_ONE_test():
    """A drained violation does not reappear, or every later test inherits the first one's failure."""
    cg._VIOLATIONS.append("synthetic")
    assert list(cg.drain_violations()) == ["synthetic"]
    assert list(cg.drain_violations()) == []


# ── the real tree ─────────────────────────────────────────────────────────────────────────────────


@matrix_exempt("scans the repo's own distributed tests; no kernel, shape or dtype to sweep")
@numeric_exempt("the subject is a list of problem strings, not a computed tensor")
def test_the_repo_s_own_distributed_tests_are_clean():
    """Non-vacuity plus a landing gate.

    Non-vacuous because the directory is scanned for real rather than through a fixture: if the guard
    ever stops finding files, this fails on an empty scan rather than passing on one. And it fails
    the moment somebody adds a bare skip, which is the whole point of the exercise.
    """
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent / "distributed"
    files = sorted(root.rglob("test_*.py"))
    assert files, f"scanned {root} and found no test modules; the guard would pass vacuously"
    problems = [p for f in files for p in cg.divergent_skip_problems(f)]
    assert problems == [], "\n".join(problems)
