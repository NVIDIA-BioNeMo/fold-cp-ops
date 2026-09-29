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
"""Tests for ``tests/distributed/conftest.py`` -- specifically, that its barrier RUNS.

**This file exists because the inter-test barrier silently did not run, and nothing noticed.**
Measured on a 2-rank launch: rank 1 sat in `destroy_process_group` while rank 0 was still inside a
test body, and the barrier whose whole purpose is to prevent exactly that had never executed a
single `barrier()` in the session. One cell passed, two hung, twelve hung.

The mechanism is three innocuous facts that only bite in combination:

1. the hook is declared ``@pytest.hookimpl(trylast=True)``, so it runs AFTER pytest's own
   ``pytest_runtest_teardown``;
2. pytest's own implementation is the one that calls ``teardown_exact``, i.e. it runs the FIXTURE
   FINALIZERS -- including ``dist_manager``'s ``DistributedManager.cleanup()``;
3. that cleanup calls ``destroy_process_group``, so by the time the barrier hook is reached its own
   ``if not dist.is_initialized(): return`` guard is TRUE and it returns without barriering.

Each of those is defensible alone. Together the barrier is a no-op in precisely the configuration it
was written for, and every symptom is a hang somewhere else entirely.

**What is asserted here, and why it is these two things.** Asserting the ordering alone would pin a
pytest implementation detail; asserting "the barrier ran" alone would not say why it stopped. So the
first test measures the ORDER (the cause) and the second measures the CONSEQUENCE (a guarded hook
after a de-initializing finalizer never runs its body). The second is the one that would have caught
the original defect, and it will catch the next reordering regardless of which of the three facts
changes.

Both run in an ordinary session: no GPU, no process group, no torchrun. They use pytest's
``pytester`` fixture, which runs a real nested pytest with real hook dispatch -- a mock of the
ordering would only assert what its author believed pluggy does, which is the belief that produced
the defect.
"""

import re

from fold_cp_ops.testing.kernel_matrix import matrix_exempt
from fold_cp_ops.testing.numeric_guard import numeric_exempt

pytest_plugins = ["pytester"]

# A nested session that mirrors the real arrangement: a fixture finalizer that de-initializes some
# state, and a `trylast` teardown hook guarded on that state. Deliberately NOT importing torch --
# the defect is about hook order and a guard, and reproducing it with a plain boolean shows that
# nothing about CUDA or NCCL is load-bearing.
_NESTED_CONFTEST = '''
import pytest

ORDER_FLAG = "__ORDER_FLAG__"

STATE = {"initialized": False, "events": [], "barriers": 0}


@pytest.fixture
def resource():
    """Stands in for `dist_manager`: initializes, yields, then DE-INITIALIZES in its finalizer."""
    STATE["initialized"] = True
    yield "r"
    STATE["initialized"] = False
    STATE["events"].append("fixture-teardown")


@pytest.hookimpl(**{ORDER_FLAG: True})
def pytest_runtest_teardown(item, nextitem):
    """Mirrors the real barrier hook: declared trylast, and guarded on the state above."""
    STATE["events"].append("barrier-hook")
    if not STATE["initialized"]:
        return
    STATE["barriers"] += 1
'''

_NESTED_TESTS = """
from conftest import STATE


def test_one(resource):
    assert resource == "r"


def test_two(resource):
    assert resource == "r"


def test_report():
    # Runs last; by now both earlier teardowns have completed.
    print("ORDER=" + ",".join(STATE["events"][:2]))
    print("BARRIERS=" + str(STATE["barriers"]))
"""


def _run_nested(pytester, order_flag="trylast"):
    """Run the nested session and return ``(order, barriers)`` parsed from its output.

    Args:
        pytester: pytest's own fixture for running a nested pytest in a temp directory.
        order_flag: ``"trylast"`` (the broken arrangement) or ``"tryfirst"`` (the fix). The
            SAME nested harness proves both, so the two outcomes are directly comparable
            rather than being two differently-written probes.

    Returns:
        ``(order, barriers)`` where ``order`` is the two-element event list as a comma-joined
        string and ``barriers`` is how many times the guarded hook body executed.

    Raises:
        AssertionError: If the nested session did not run all three tests, or did not emit the two
            report lines -- either means the probe measured nothing, which must not be mistaken for
            a clean result.
    """
    pytester.makeconftest(_NESTED_CONFTEST.replace("__ORDER_FLAG__", order_flag))
    pytester.makepyfile(test_nested=_NESTED_TESTS)
    result = pytester.runpytest("-s", "-p", "no:randomly")
    result.assert_outcomes(passed=3)
    lines = "\n".join(result.outlines)

    # SUBSTRING, not startswith: under `-s` the nested prints are streamed inline and land
    # concatenated onto pytest's own progress line ("test_nested.py ..ORDER=..."). A startswith
    # parser found nothing and the guard below correctly refused to report -- which is the guard
    # working, but the parser was assuming a format rather than reading one.
    def _field(name):
        m = re.search(rf"{name}=(\S+)", lines)
        return m.group(1) if m else None

    order, barriers = _field("ORDER"), _field("BARRIERS")
    assert order is not None and barriers is not None, (
        f"the nested session did not report; it measured nothing:\n{lines}"
    )
    return order, int(barriers)


@matrix_exempt(
    "the subject is pytest HOOK ORDERING inside a nested session -- there is no kernel, no shape "
    "and no mesh, so no axis of the distributed matrix varies what is being asserted"
)
@numeric_exempt("asserts an event order, not a computed value")
def test_a_trylast_teardown_hook_runs_after_fixture_finalizers(pytester):
    """A ``trylast`` teardown hook runs AFTER the fixture finalizers, not before.

    This is the CAUSE. pytest's own ``pytest_runtest_teardown`` is the implementation that calls
    ``teardown_exact`` and therefore runs the finalizers, and ``trylast`` orders our hook after it.

    Pinned as a test rather than a comment because it is a property of pluggy that a reader would
    otherwise have to take on faith -- and taking it on faith in the other direction is what put a
    dead barrier in the harness.
    """
    order, _ = _run_nested(pytester)
    assert order == "fixture-teardown,barrier-hook", (
        f"observed {order!r}. A trylast teardown hook running BEFORE the finalizers would make the "
        "barrier able to protect them; running after, it cannot."
    )


@matrix_exempt(
    "same subject as the ordering test above -- a nested pytest session's hook dispatch, which no "
    "declared axis of this directory's matrix describes"
)
@numeric_exempt("asserts how many times a hook body executed, not a computed value")
def test_a_guarded_barrier_after_a_deinitializing_finalizer_never_runs(pytester):
    """The CONSEQUENCE, and the assertion that would have caught the real defect.

    A hook that (a) runs after the finalizers and (b) guards its body on state those finalizers tear
    down executes its body **zero** times across a whole session -- while looking, from the outside,
    exactly like a barrier that ran and found nothing to do.

    That is the shape of the real bug: `pytest_runtest_teardown` in the distributed conftest is
    declared ``trylast`` and opens with ``if not dist.is_initialized(): return``, while
    ``dist_manager``'s finalizer calls ``destroy_process_group``. The barrier never barriered, the
    ranks drifted, and the symptom surfaced as a hang inside an unrelated collective two tests later.

    **The number is the assertion.** "The hook ran" is true and useless -- it ran and returned. What
    matters is how many times the guarded BODY executed, which is the only quantity that
    distinguishes a working barrier from a decorative one.
    """
    _, barriers = _run_nested(pytester)
    assert barriers == 0, (
        f"expected 0, observed {barriers}. This is SYNTHETIC and pins a pluggy property, so it "
        "stays 0 forever regardless of what our conftest does -- do NOT 'update it with the fix'. "
        "A change here means pluggy's ordering semantics moved, which is a much larger finding."
    )


@matrix_exempt(
    "same nested-pytest subject as the two tests above -- hook dispatch, which no declared axis of "
    "this directory's matrix describes"
)
@numeric_exempt("asserts how many times a hook body executed, not a computed value")
def test_a_tryfirst_barrier_actually_runs_once_per_test(pytester):
    """The FIX, witnessed: with ``tryfirst`` the barrier body runs once per test.

    This is the assertion that fails if anyone reverts the ordering flag. The two tests above
    describe the defect; without this one the file would only DOCUMENT the bug and nothing would
    notice its return.

    Two tests in the nested session, so the expected count is 2 -- a per-test barrier that ran once
    for the whole session would be just as broken as one that never ran, and only a count can tell
    those apart.
    """
    order, barriers = _run_nested(pytester, order_flag="tryfirst")
    assert order == "barrier-hook,fixture-teardown", (
        f"observed {order!r}; with tryfirst the barrier must precede the finalizer, or it is again "
        "unable to protect the collective that finalizer performs."
    )
    assert barriers == 2, (
        f"expected the barrier body to run once per test (2), observed {barriers}. Zero means the "
        "ordering regressed to the original defect; one means it is no longer per-test."
    )


@matrix_exempt(
    "the subject is the REAL conftest's decorator, read statically -- there is no kernel, shape or "
    "mesh, and no declared axis varies what is asserted"
)
@numeric_exempt("asserts a decorator's keyword, not a computed value")
def test_the_real_teardown_barrier_is_declared_tryfirst():
    """**The actual gate.** Our own ``pytest_runtest_teardown`` must be declared ``tryfirst``.

    The three tests above are a mechanism DEMONSTRATION on a synthetic session: they pin what
    pluggy does, permanently, and they would all stay green if someone flipped the real hook back
    to ``trylast`` tomorrow. That is the property that let the original defect survive, so this
    test asserts the subject rather than the mechanism.

    **Read by AST, not by grep**, for a reason measured twice today: a text search cannot tell code
    from documentation. A ``grep trylast=True`` to confirm the fix returned exactly one hit -- the
    comment explaining the change -- and trusting it would have said the fix had not applied. The
    decorator node cannot be confused with prose about the decorator.

    Importing the conftest is also avoided deliberately: it installs a signal handler, a wedge
    watchdog and process-wide state at import, none of which a static assertion needs.
    """
    import ast
    import pathlib

    def _flags(source):
        """The hookimpl keywords on a module-level ``pytest_runtest_teardown``, or None if absent."""
        fn = next(
            (
                n
                for n in ast.parse(source).body
                if isinstance(n, ast.FunctionDef) and n.name == "pytest_runtest_teardown"
            ),
            None,
        )
        if fn is None:
            return None
        return {
            kw.arg: getattr(kw.value, "value", None)
            for d in fn.decorator_list
            if isinstance(d, ast.Call) and ast.unparse(d.func).endswith("hookimpl")
            for kw in d.keywords
        }

    # POSITIVE CONTROL, first: a gate that has never failed proves nothing, and this one reads a
    # file it does not control. Feed it the BROKEN declaration and the comment that would fool a
    # grep, and require that it says trylast -- so a passing verdict below is a measurement rather
    # than an extraction that quietly found nothing.
    broken = (
        "import pytest\n"
        "# historical note: this used to be trylast=True and that was the bug\n"
        "@pytest.hookimpl(trylast=True)\n"
        "def pytest_runtest_teardown(item, nextitem):\n"
        "    pass\n"
    )
    assert _flags(broken) == {"trylast": True}, (
        f"the control returned {_flags(broken)!r}; the extraction cannot see a decorator it is "
        "meant to reject, so its verdict on the real file means nothing."
    )
    assert _flags("def pytest_runtest_teardown(i, n): pass\n") == {}, (
        "an undecorated hook must read as no flags, not as correctly ordered."
    )

    src = pathlib.Path(__file__).with_name("conftest.py").read_text()
    fn = next(
        (
            n
            for n in ast.parse(src).body
            if isinstance(n, ast.FunctionDef) and n.name == "pytest_runtest_teardown"
        ),
        None,
    )
    assert fn is not None, (
        "conftest.py defines no module-level pytest_runtest_teardown. If the barrier moved, move "
        "this assertion with it -- an absent hook is not the same as a correctly-ordered one."
    )
    flags = {
        kw.arg: getattr(kw.value, "value", None)
        for d in fn.decorator_list
        if isinstance(d, ast.Call) and ast.unparse(d.func).endswith("hookimpl")
        for kw in d.keywords
    }
    assert flags.get("tryfirst") is True, (
        f"the teardown barrier is declared {flags or '<no hookimpl>'}. It MUST be tryfirst: pytest's "
        "own pytest_runtest_teardown runs the fixture finalizers, and dist_manager's finalizer "
        "calls the COLLECTIVE destroy_process_group. Declared trylast, the barrier runs after the "
        "group is destroyed and its own is_initialized() guard returns early -- so it never "
        "barriers at all, the ranks drift, and the symptom is a hang in an unrelated collective "
        "two tests later. Measured: 1 test passed, 2 hung, 12 hung."
    )
