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

#!/usr/bin/env python
"""Pre-commit guard: no hand-rolled `perf_counter` timing of a kernel or collective launch.

Enforces the CLAUDE.md HARD RULE "benchmark via `bench_utils` (timing) + the `harness` (multi-cell);
NEVER roll your own". A raw launch is ASYNC, so a `time.perf_counter()` span around it measures host
DISPATCH, not device time -- and `volume / dispatch_time` then yields a "bandwidth" that can exceed
the physical link, which is the tell. `bench_utils` times with CUDA events, L2-flushes, and
`all_reduce(MAX)` for the slowest PE.

**Why this guard exists at all.** Prose did not hold. During the perf-gate rebuild the repo's own
sanctioned primitive was re-invented twice inside one week -- once as a CUDA-event two-point solve
that shipped a 6x-wrong constant into the gate, once as exactly the forbidden `perf_counter` loop,
which read 3x high and stable enough to look right. Neither author was unaware of the rule; a new
instrument was simply faster to WRITE than the existing one was to FIND.

WHAT IT LOOKS FOR, and why not simply "any use of `perf_counter`". A blanket ban would be red on
this tree on day one -- cold-compile timers, the wedge watchdog, phase timers and progress reports
all take wall clock legitimately, and CLAUDE.md's de-branding guard is the standing counter-example
of what a permanently-red guard is worth (nothing: it has gated no commit, ever). So the guard fires
on the two shapes that mean "this elapsed time is being used AS a measurement":

    P1  per-call time  <elapsed> / <non-literal>   e.g. `elapsed / n`, `(t1 - t0) / iters`
    P2  derived rate   <anything> / <elapsed>      e.g. `nbytes / elapsed`  -- the >line-rate tell

`<elapsed>` means a value this guard can PROVE came from `time.perf_counter()`, `time.time()` or
`time.monotonic()` by subtraction inside the same function, directly or through one assignment.
Dividing by a numeric literal (`elapsed / 1e6`, `elapsed / 60`) is a UNIT conversion and never
fires; a raw elapsed that is merely printed, logged, or compared against a deadline never fires.

That is a deliberately narrow reading of "a timing span around a callable", chosen because it is the
shape that produces a WRONG NUMBER rather than merely a wall-clock report -- and because a guard
that fires on a legitimate site teaches people to suppress it, which costs more than it saves.

ESCAPE HATCH, declared rather than silent: put `# adhoc-timing-ok: <reason>` on the offending line
or on the enclosing `def`. The reason is checked for EXISTENCE, not for truth -- no static tool can
verify that a span is legitimate. What the declaration removes is the cheapness of forgetting: "we
know this is not a kernel timing" and "it is written down as not a kernel timing" stop being the
same state. Same bargain `rank_invariant_skip(because=...)` strikes in the collective guard.

Usage (pre-commit passes the staged files): guard_no_adhoc_timing.py <files...>
A BARE run with no arguments is a usage error (exit 2), not a whole-tree scan -- the de-branding
guard's bare-run exit code carries no signal precisely because nobody agreed what it scanned.
"""

import ast
import sys
from pathlib import Path

#: Files whose JOB is to hand-roll the host-side span. Both are the sanctioned primitives the rest
#: of the repo is required to call instead of writing their own, so a guard that flagged them would
#: be flagging the fix. Matched on the path SUFFIX so a worktree, a `-e` install and a staged
#: absolute path all resolve the same.
_SANCTIONED = (
    "fold_cp_ops/_internal/bench_timing.py",  # host_dispatch_us, and the event-window timer
    "tests/perf/calibration.py",  # imports host_dispatch_us back under its original name
)

#: The clock sources whose difference is a duration. `time.process_time` and `time.thread_time` are
#: deliberately ABSENT: they measure CPU time, cannot be mistaken for device time, and nothing in
#: this repo uses them.
_CLOCKS = ("perf_counter", "time", "monotonic", "perf_counter_ns", "monotonic_ns", "time_ns")

_MARKER = "adhoc-timing-ok:"


def _is_clock_call(node) -> bool:
    """True if `node` is a call to one of `_CLOCKS`, spelled `time.X()` or a bare imported `X()`.

    Purpose
        The single place that decides what counts as reading a clock, so the qualified and unbound
        spellings cannot drift apart.

    Args:
        node: any AST node. A non-`Call` returns False rather than raising, so callers can hand it
            arbitrary expression nodes.

    Returns:
        True for `time.perf_counter()`, `perf_counter()`, `time.time()`, and the `_ns` variants.
        `time.time()` is included even though the attribute and module share a name -- `Attribute`
        matching is on the ATTR, so `time.time` and a bare `time()` both resolve.
    """
    if not isinstance(node, ast.Call):
        return False
    f = node.func
    if isinstance(f, ast.Attribute):
        return f.attr in _CLOCKS and isinstance(f.value, ast.Name) and f.value.id == "time"
    return isinstance(f, ast.Name) and f.id in _CLOCKS


def _own_body(node):
    """The statements of `node` that belong to ITS scope, i.e. every one that is not a nested `def`.

    `_walk_own_scope` stops at a function boundary while DESCENDING, but a scope's body statement can
    BE a function definition, and then the walk starts inside it -- which is how a module scan
    reached into every function it contained. The outer pass visits each function as its own scope
    anyway, so dropping them here loses nothing and is what keeps one scope's names its own.
    """
    return [
        st
        for st in getattr(node, "body", [])
        if not isinstance(st, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]


def _walk_own_scope(node):
    """Yield `node`'s descendants, NOT descending into a nested `def`/`async def`/`lambda`.

    Purpose
        `ast.walk` has no scope notion: from a module it walks straight through every function body,
        so a module-level scan sees names bound INSIDE functions and divisions written inside them.
        That is not a hypothetical -- the first version used `ast.walk` with a `continue` on
        `FunctionDef`, which skips the node while still yielding its whole subtree, and it reported
        every in-function finding a SECOND time at module scope with `lineno = 0` for the enclosing
        def. The visible symptom was that an escape-hatch comment on a `def` line stopped working,
        because the duplicate finding had no def line to look at.

    Args:
        node: any AST node. Yielded first, then its descendants breadth-first.

    Returns:
        A generator. A nested function node IS yielded (so a caller can find it) but its children
        are not, which is what makes each scope's names its own.
    """
    from collections import deque

    q = deque([node])
    while q:
        cur = q.popleft()
        yield cur
        for child in ast.iter_child_nodes(cur):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
                continue
            q.append(child)


class _Scope:
    """Collect elapsed-derived divisions within ONE function body.

    Purpose
        A function is the unit because a `t0` in one function and an `elapsed` in another are not
        the same span, and treating the module as one scope produced cross-function false hits on
        the first draft (a `t0` in a fixture matched an `elapsed` in a test 200 lines away).

    Semantics
        Two passes over the same body. The first binds names: a name assigned a clock READING
        (`t0 = time.perf_counter()`) becomes a *stamp*; a name assigned a difference of a stamp or
        of two clock readings becomes an *elapsed*. The second looks for `BinOp(Div)` whose
        numerator or denominator is an elapsed value. Both passes walk nested nodes, so a span
        inside a `with`, a loop or a comprehension is seen; a nested `def` gets its OWN scope, so a
        closure cannot inherit its parent's stamps.

    Input requirements
        `node` must be a function-like AST node (`FunctionDef`, `AsyncFunctionDef`) or `Module`.
        Nothing else is meaningful: the constructor does not validate it, and handing it an
        expression yields an empty finding list rather than an error.

    Returns:
        Nothing; findings accumulate in `self.findings` as `(lineno, pattern, kind)` triples.
    """

    def __init__(self):
        self.stamps: set[str] = set()
        self.elapsed: set[str] = set()
        self.findings: list[tuple[int, str, str]] = []

    def _is_elapsed(self, e) -> bool:
        """True if `e` is a duration this guard can trace back to a clock reading."""
        if isinstance(e, ast.Name):
            return e.id in self.elapsed
        if isinstance(e, ast.BinOp) and isinstance(e.op, ast.Sub):
            for side in (e.left, e.right):
                if _is_clock_call(side) or (isinstance(side, ast.Name) and side.id in self.stamps):
                    return True
        # `(t1 - t0) * 1e3` and `(t1 - t0) / n` both keep the duration; unwrap one level of scaling
        if isinstance(e, ast.BinOp) and isinstance(e.op, (ast.Mult, ast.Div)):
            return self._is_elapsed(e.left) or (
                isinstance(e.op, ast.Mult) and self._is_elapsed(e.right)
            )
        return False

    def bind(self, node):
        """First pass: record which names hold a clock stamp and which hold a duration."""
        for n in _walk_own_scope(node):
            if not isinstance(n, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
                continue
            targets = n.targets if isinstance(n, ast.Assign) else [n.target]
            names = [t.id for t in targets if isinstance(t, ast.Name)]
            if n.value is None:
                continue
            if _is_clock_call(n.value):
                self.stamps.update(names)
            elif self._is_elapsed(n.value):
                self.elapsed.update(names)


def _scan_function(node, src_lines) -> list[tuple[int, str]]:
    """Find the P1/P2 divisions inside one function body.

    Args:
        node: the function-like node to scan.
        src_lines: the file's lines, 0-indexed, used only to read the escape-hatch comment.

    Returns:
        A list of `(lineno, pattern)` for each unsuppressed finding, `pattern` being "P1" or "P2".
    """
    sc = _Scope()
    # Iterate to a fixpoint: `t0 = perf_counter()` then `el = t1 - t0` then `per = el / n` is a
    # three-link chain, and a single pass binds `el` only if `t0` was already known. Two passes
    # suffice for every shape in this repo; a third costs nothing and removes the assumption.
    for _ in range(3):
        for stmt in _own_body(node):
            sc.bind(stmt)
    out = []
    for stmt in _own_body(node):
        for n in _walk_own_scope(stmt):
            if not (isinstance(n, ast.BinOp) and isinstance(n.op, ast.Div)):
                continue
            # Pattern 2 first: a rate is the more dangerous shape, and `elapsed / elapsed` is neither.
            if sc._is_elapsed(n.right):
                out.append((n.lineno, "P2"))
            elif sc._is_elapsed(n.left) and not isinstance(n.right, ast.Constant):
                out.append((n.lineno, "P1"))
    return out


def _suppressed(lineno: int, src_lines: list[str], fn_lineno: int) -> bool:
    """True if the finding carries a `# adhoc-timing-ok: <reason>` with a NON-EMPTY reason.

    The reason must be non-empty because a bare marker is a suppression with no author and no
    argument, which is the state this whole guard exists to make expensive. Checked on the finding's
    own line and on the enclosing `def` line, so a function that is wall-clock by nature declares it
    once rather than on every division.
    """
    for ln in (lineno, fn_lineno):
        if 1 <= ln <= len(src_lines):
            line = src_lines[ln - 1]
            if _MARKER in line and line.split(_MARKER, 1)[1].strip():
                return True
    return False


def scan(path: Path) -> list[tuple[int, str]]:
    """Scan one file; returns `(lineno, pattern)` for every unsuppressed finding.

    Args:
        path: a `.py` file. A file that does not parse returns no findings rather than raising --
        a syntax error is another tool's job to report, and crashing here would mask every OTHER
        file in the same pre-commit invocation.

    Returns:
        The findings, possibly empty. A sanctioned path always returns empty.
    """
    p = str(path).replace("\\", "/")
    if any(p.endswith(s) for s in _SANCTIONED):
        return []
    try:
        src = path.read_text()
        tree = ast.parse(src)
    except (OSError, SyntaxError, UnicodeDecodeError):
        return []
    lines = src.splitlines()
    found = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Module)):
            continue
        fn_lineno = getattr(node, "lineno", 0)
        for lineno, pat in _scan_function(node, lines):
            if not _suppressed(lineno, lines, fn_lineno):
                found.append((lineno, pat))
    return sorted(set(found))


_EXPLAIN = {
    "P1": "an elapsed wall-clock span divided by a call/iteration count -- that is a per-call TIME",
    "P2": "something divided BY an elapsed wall-clock span -- that is a derived RATE",
}


def main(argv) -> int:
    """Entry point. Exits 1 on a finding, 2 on a bare invocation, 0 otherwise."""
    files = [Path(a) for a in argv[1:] if a.endswith(".py")]
    if not files:
        sys.stderr.write(
            "usage: guard_no_adhoc_timing.py <files...>\n"
            "This guard takes filenames (pre-commit's convention). A bare run scans nothing on "
            "purpose: an exit code whose scope nobody agreed on carries no signal.\n"
        )
        return 2
    bad = {}
    for f in files:
        hits = scan(f)
        if hits:
            bad[str(f)] = hits
    if not bad:
        return 0
    sys.stderr.write("\nERROR: hand-rolled wall-clock timing of what looks like a launch:\n")
    for p, hits in sorted(bad.items()):
        for lineno, pat in hits:
            sys.stderr.write(f"  {p}:{lineno}: [{pat}] {_EXPLAIN[pat]}\n")
    sys.stderr.write(
        "\nA raw kernel/collective launch is ASYNC: the host returns before the device finishes, so "
        "a `perf_counter` span measures host DISPATCH, not device time -- and `volume / elapsed` "
        "then reports a bandwidth that can EXCEED the physical link.\n"
        "Time it with `bench_utils.benchmark_single` / `benchmark_paired` (CUDA events, L2 flush, "
        "all_reduce(MAX) for the slowest PE), or `host_dispatch_us` if host dispatch is genuinely "
        "the subject. If a new primitive is needed, ADD it to `bench_timing.py` with a test.\n"
        "If this span times something that is NOT a launch (a compile, a phase, a deadline), say so "
        "on the line or on the enclosing def:  # adhoc-timing-ok: <reason>\n"
        "See CLAUDE.md HARD RULE: benchmark via `bench_utils` + the `harness`; NEVER roll your own.\n"
    )
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
