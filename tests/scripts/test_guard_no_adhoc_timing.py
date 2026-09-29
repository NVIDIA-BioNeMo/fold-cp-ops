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

"""Tests for ``scripts/guard_no_adhoc_timing.py`` -- the hand-rolled-timing tripwire.

**The load-bearing tests are the ones that prove it FIRES**, and second to those, the ones that
prove it stays SILENT on legitimate wall-clock use. A guard that never fires and a guard that fires
on everything fail the same way in practice: the first gates nothing, the second is suppressed
wholesale within a week. This repo has one of each as a standing example -- the de-branding guard is
unwired and exits 1 on both trees on prose alone, so its exit code carries no signal at all.

The guard was written for a defect that had already happened twice; when it was first run over this
tree it found a THIRD, in `tests/perf/test_benchmark_perf_gemm.py`, where a local dispatch loop put
its trailing `synchronize()` inside the timed span and so returned device time at one cell. That
cell's one-sided `<= ceiling` assertion had therefore been passing unconditionally. The fixture
`test_the_real_defect_this_guard_was_written_for_is_caught` below is that shape, reduced.

No GPU, no torch: the guard is an AST pass, which is why it can run in pre-commit on every file.
"""

import subprocess
import sys
from pathlib import Path

from fold_cp_ops.testing.kernel_matrix import matrix_exempt
from fold_cp_ops.testing.numeric_guard import numeric_exempt

pytestmark = matrix_exempt(
    "the subject is a source-scanning GUARD -- it compiles nothing and has no shape/dtype/tile axes "
    "for a KernelMatrix to declare"
)

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "guard_no_adhoc_timing.py"


def _write(tmp_path, body: str, name: str = "probe.py") -> Path:
    """Write `body` as a module under `tmp_path` and return its path.

    Args:
        tmp_path: pytest's per-test directory.
        body: module source. Written verbatim, so the caller controls indentation exactly -- a
            dedent here would silently change what the AST sees, which is the thing under test.
        name: the file name. Only matters for the sanctioned-path test, which keys on the suffix.

    Returns:
        The written path.
    """
    p = tmp_path / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(body)
    return p


def _run(*paths):
    """Invoke the guard as a SUBPROCESS; returns ``(returncode, stderr)``.

    A subprocess because the exit STATUS is half the contract -- pre-commit reads nothing else --
    and importing `main()` would exercise the message while skipping the signal.
    """
    r = subprocess.run(
        [sys.executable, str(_SCRIPT), *[str(p) for p in paths]],
        capture_output=True, text=True, timeout=120,
    )
    return r.returncode, r.stderr


@numeric_exempt("asserts a guard verdict, not a computed value")
def test_the_real_defect_this_guard_was_written_for_is_caught(tmp_path):
    """THE control, reduced from the live finding: a local dispatch loop dividing by its count.

    This is the shape that had already shipped three times in this repo -- twice during the perf-gate
    rebuild and once, undetected, inside a perf gate's own assertion. The `/ n` is what makes it a
    MEASUREMENT rather than a wall-clock report, and it is what the guard keys on.
    """
    f = _write(tmp_path, """
import time
def _host_us(fn, n=500):
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    return (time.perf_counter() - t0) / n * 1e6
""")
    rc, err = _run(f)
    assert rc == 1, f"the guard must fire on the defect it was written for\n{err}"
    assert "[P1]" in err
    assert "bench_utils" in err, "the message must name the fix, not merely the finding"


@numeric_exempt("asserts a guard verdict, not a computed value")
def test_a_derived_bandwidth_is_caught(tmp_path):
    """`volume / elapsed` is the shape CLAUDE.md names explicitly: a >line-rate value is the tell."""
    f = _write(tmp_path, """
import time
def bw(nbytes, launch):
    t0 = time.perf_counter()
    launch()
    elapsed = time.perf_counter() - t0
    return nbytes / elapsed
""")
    rc, err = _run(f)
    assert rc == 1, err
    assert "[P2]" in err


@numeric_exempt("asserts a guard verdict, not a computed value")
def test_a_three_link_chain_is_still_caught(tmp_path):
    """`t0 = clock()` -> `el = t1 - t0` -> `per = el / n` binds across three statements.

    A single binding pass resolves only the first link, so this is the case that decides whether the
    guard sees a real span or only the toy one-liner. It is also the spelling most real code uses.
    """
    f = _write(tmp_path, """
import time
def run(fn, n):
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    t1 = time.perf_counter()
    el = t1 - t0
    per_call = el / n
    return per_call
""")
    rc, err = _run(f)
    assert rc == 1, err


@numeric_exempt("asserts a guard verdict, not a computed value")
def test_a_raw_wall_clock_report_does_not_fire(tmp_path):
    """A compile timer, a phase timer and a watchdog all take wall clock legitimately.

    They report or compare the elapsed value; they never divide it into a per-call cost or a rate.
    This is the test that keeps the guard wireable: it is green on this tree only because these are
    silent, and there are dozens of them.
    """
    f = _write(tmp_path, """
import time
def compile_and_report(compile_fn, deadline_s):
    t0 = time.perf_counter()
    compile_fn()
    elapsed = time.perf_counter() - t0
    print(f"compiled in {elapsed:.2f}s")
    if elapsed > deadline_s:
        raise RuntimeError("too slow")
    return elapsed * 1e3
""")
    rc, err = _run(f)
    assert rc == 0, f"a wall-clock report must not fire\n{err}"


@numeric_exempt("asserts a guard verdict, not a computed value")
def test_a_unit_conversion_does_not_fire(tmp_path):
    """`elapsed / 1e6` converts units; `elapsed / n` measures. Only the second is a finding.

    Dividing by a numeric LITERAL is the discriminator, and it has to be, because `/ 60` for minutes
    and `/ iters` for a per-call cost are indistinguishable without it.
    """
    f = _write(tmp_path, """
import time
def minutes(fn):
    t0 = time.perf_counter()
    fn()
    return (time.perf_counter() - t0) / 60
""")
    rc, err = _run(f)
    assert rc == 0, err


@numeric_exempt("asserts a guard verdict, not a computed value")
def test_a_stamp_does_not_leak_into_a_SIBLING_function(tmp_path):
    """`t0` in one function and `elapsed / n` in another are not one span.

    The first draft scoped by MODULE and matched a fixture's `t0` against a division 200 lines away.
    A guard with cross-function false hits gets an `exclude:` line in pre-commit, which is how a
    wired guard becomes an unwired one without anyone deciding to unwire it.
    """
    f = _write(tmp_path, """
import time
def a():
    t0 = time.perf_counter()
    return t0

def b(total, n):
    return total / n
""")
    rc, err = _run(f)
    assert rc == 0, err


@numeric_exempt("asserts a guard verdict, not a computed value")
def test_a_declared_escape_hatch_silences_it(tmp_path):
    """A span that is genuinely not a launch says so, in one line, and the guard believes it.

    Existence-checked, not truth-checked -- no static tool can verify that a callable does not
    launch. What the marker removes is the cheapness of forgetting.
    """
    f = _write(tmp_path, """
import time
def rate(items, work):
    t0 = time.perf_counter()
    work()
    el = time.perf_counter() - t0
    return items / el  # adhoc-timing-ok: `work` is a pure-Python parse, no device involved
""")
    rc, err = _run(f)
    assert rc == 0, err


@numeric_exempt("asserts a guard verdict, not a computed value")
def test_an_EMPTY_escape_hatch_does_NOT_silence_it(tmp_path):
    """A bare marker is a suppression with no author and no argument. That is the state to make dear.

    This is the half that decides whether the escape hatch is a declaration or a mute button: if a
    reasonless marker worked, the cheapest way past the guard would be to type six words, and every
    suppression in the tree would eventually be one.
    """
    f = _write(tmp_path, """
import time
def rate(items, work):
    t0 = time.perf_counter()
    work()
    el = time.perf_counter() - t0
    return items / el  # adhoc-timing-ok:
""")
    rc, err = _run(f)
    assert rc == 1, f"a reasonless marker must not suppress\n{err}"


@numeric_exempt("asserts a guard verdict, not a computed value")
def test_the_escape_hatch_may_sit_on_the_enclosing_def(tmp_path):
    """A function that is wall-clock by nature declares it once, not on every division inside it."""
    f = _write(tmp_path, """
import time
def rate(items, work):  # adhoc-timing-ok: pure host-side parse throughput, no launches anywhere
    t0 = time.perf_counter()
    work()
    el = time.perf_counter() - t0
    return items / el
""")
    rc, err = _run(f)
    assert rc == 0, err


@numeric_exempt("asserts a guard verdict, not a computed value")
def test_the_sanctioned_timers_are_exempt(tmp_path):
    """`bench_timing.py` and `calibration.py` hand-roll the span BY JOB; flagging them flags the fix.

    Keyed on the path suffix so a worktree, an editable install and a staged absolute path all
    resolve the same -- a prefix match would exempt the file in one checkout and flag it in another.
    """
    body = """
import time
def host_dispatch_us(fn, n):
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    return (time.perf_counter() - t0) / n * 1e6
"""
    assert _run(_write(tmp_path, body, "fold_cp_ops/_internal/bench_timing.py"))[0] == 0
    assert _run(_write(tmp_path, body, "tests/perf/calibration.py"))[0] == 0
    # ... and the SAME body anywhere else is still a finding, or the exemption is doing no work
    assert _run(_write(tmp_path, body, "tests/perf/somewhere_else.py"))[0] == 1


@numeric_exempt("asserts a guard verdict, not a computed value")
def test_an_unparseable_file_does_not_mask_a_finding_beside_it(tmp_path):
    """One broken file must not turn a pre-commit run green over every other staged file."""
    broken = _write(tmp_path, "def (:\n", "broken.py")
    bad = _write(tmp_path, """
import time
def m(fn, n):
    t0 = time.perf_counter()
    fn()
    return (time.perf_counter() - t0) / n
""", "bad.py")
    rc, err = _run(broken, bad)
    assert rc == 1, f"a syntax error must not mask the finding next to it\n{err}"


@numeric_exempt("asserts a usage refusal, not a computed value")
def test_a_bare_invocation_is_a_USAGE_error_not_a_clean_verdict():
    """Exit 2, distinct from both verdicts.

    The de-branding guard is the counter-example this copies away from: a bare run there scans the
    whole tree and exits 1 on prose, so its status means nothing and nobody gates on it. A status
    that cannot be mistaken for "clean" is the point.
    """
    r = subprocess.run([sys.executable, str(_SCRIPT)], capture_output=True, text=True, timeout=60)
    assert r.returncode == 2
    assert "usage:" in r.stderr


@numeric_exempt("asserts a guard verdict over the live tree, not a computed value")
def test_the_guard_is_GREEN_over_the_whole_live_tree():
    """A wired guard must pass on the tree it is wired into, or it is unwired by the first bypass.

    Not a tautology with the tests above: those use fixtures. This one asserts the property that
    makes the pre-commit line honest, and it is the test that will fail the day someone adds a real
    one -- which is the intended way to find out.
    """
    root = Path(__file__).resolve().parents[2]
    files = [
        p for d in ("tests", "benchmark", "fold_cp_ops", "scripts")
        for p in (root / d).rglob("*.py")
        if "trash_to_be_removed" not in p.parts
    ]
    assert files, "found no sources to scan -- the walk is wrong, not the tree"
    rc, err = _run(*files)
    assert rc == 0, f"the live tree must be clean for the pre-commit hook to be honest\n{err}"


@numeric_exempt("asserts the hook's wiring, not a computed value")
def test_the_hook_is_WIRED_into_pre_commit_and_its_filter_selects_real_files():
    """An unwired guard is worth nothing, and this repo has the standing example.

    This repo had the standing example: a de-branding guard believed to be enforced and never wired
    into `.pre-commit-config.yaml`, so no commit was ever gated on it. It was eventually deleted
    rather than wired, which is the cheaper of the two honest endings but still a whole invariant
    lost to a missing four lines of config.

    So the wiring is asserted here rather than eyeballed once. Checked structurally -- the hook
    exists, its `entry` names a script that exists, and its `files`/`exclude` regexes actually SELECT
    this repo's sources -- because `pre-commit run` itself needs network access to fetch the remote
    hook repos in the same config and cannot run in a sandboxed session. A green `pre-commit run`
    would be the stronger proof; a filter that silently matches nothing is the failure this catches,
    and it is the one that looks identical to a passing hook.
    """
    import re

    import yaml

    root = Path(__file__).resolve().parents[2]
    cfg = yaml.safe_load((root / ".pre-commit-config.yaml").read_text())
    hooks = [h for r in cfg["repos"] for h in r.get("hooks", []) if h.get("id") == "no-adhoc-timing"]
    assert len(hooks) == 1, f"expected exactly one no-adhoc-timing hook, found {len(hooks)}"
    h = hooks[0]
    assert h["pass_filenames"] is True, "the guard takes filenames; a bare run is a usage error"
    script = h["entry"].split()[-1]
    assert (root / script).is_file(), f"the hook's entry points at a missing script: {script}"
    assert Path(script) == _SCRIPT.relative_to(root), "the hook runs a DIFFERENT script than tested"

    files_re, exclude_re = re.compile(h["files"]), re.compile(h["exclude"])
    tracked = subprocess.run(
        ["git", "ls-files"], cwd=root, capture_output=True, text=True, timeout=120
    ).stdout.splitlines()
    selected = [f for f in tracked if files_re.search(f) and not exclude_re.search(f)]
    assert len(selected) > 100, (
        f"the hook's file filter selects only {len(selected)} tracked files -- a filter that matches "
        f"nothing passes identically to a guard that works"
    )
    assert "tests/perf/test_benchmark_perf_gemm.py" in selected, (
        "the file the live defect was found in is not selected by the hook's own filter"
    )
    assert not any("trash_to_be_removed" in f for f in selected), "the trash must stay excluded"
