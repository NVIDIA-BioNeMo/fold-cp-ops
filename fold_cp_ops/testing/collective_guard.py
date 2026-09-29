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

"""Every divergent-rank decision under ``tests/distributed/`` goes through `CollectiveGate`.

Purpose
    A ``pytest.skip`` on a predicate that can differ per rank is a DEADLOCK, not a skip. The
    skipping rank leaves; every other rank walks into the next collective and blocks until a
    watchdog kills the job, with a traceback naming whatever test they happened to be in. This
    module makes the safe spelling the only spelling that survives collection and execution, so a
    future ``tests/distributed/kernels/test_<x>_a2a.py`` cannot invent its own.

    It is the third instance of a pattern this repo has twice already: `kernel_matrix`'s mandatory
    ``unsupported=`` and `numeric_guard`'s sanctioned-assertion whitelist. All three exist for the
    same reason -- to make "nobody considered this" look different from "somebody considered it and
    decided".

Semantics -- three sanctioned spellings, and NO free-text escape hatch
    * `gated_skip` -- the group is up, so reduce the decision and skip together. Wraps
      `CollectiveGate.should_skip`; it is one call rather than two because the second line of the
      two-line form is a bare `pytest.skip` no static pass can tell from the dangerous kind.
    * `rank_invariant_skip` -- the predicate is job-uniform (a mesh size, a launcher name), so every
      rank reaches it with the same answer and a bare skip is already symmetric. Costs one written
      ``because=``.
    * `pre_init_skip` -- the ONLY exemption, and it is **self-policing rather than declared**: it
      asserts at call time that ``torch.distributed`` is NOT initialized, and RAISES if it is. A
      site that runs after ``DistributedManager.initialize()`` with a live group therefore cannot
      use it, whatever reason its author writes. That is the owner's rule expressed as code:
      pre-init is the only circumstance under which a rank may decide alone, because there is no
      group to reduce over yet.

    There is deliberately no module-level exemption attribute. `numeric_guard` has one because its
    own test file must spell the forbidden forms to prove the tripwire fires; here the equivalent
    need is served by `pre_init_skip` plus the guard's own tests calling the internals directly.

Why BOTH an AST pass and a runtime tripwire
    They fail on disjoint inputs and neither subsumes the other:

    * The AST pass sees code that never runs -- a divergent skip on a branch this launch did not
      take is still a landmine for the next mesh, and the tripwire will never fire on it.
    * The tripwire sees calls the pass cannot resolve: a skip inside a helper module under a
      different directory, one reached through an alias, one built by ``getattr``. It is also the
      layer that enforces the owner's rule, because only at call time is ``dist.is_initialized()``
      knowable.

Input requirements
    Scope is by DIRECTORY (`DIST_DIRS`), not by whether a file imports ``torch.distributed``. A
    host-only module today grows a collective tomorrow, and scoping on imports would silently drop
    it from the guard at exactly that moment. Paths are matched on their parts under ``tests/``, so
    any depth below ``tests/distributed/`` is covered -- which is the point, since the A2A kernel
    tests will land in a subdirectory that does not exist yet.

Raises
    `AssertionError` from the tripwire and from `pre_init_skip` when the rule is broken; the audit
    functions RETURN problem strings rather than raising, so the caller can attribute them to the
    offending module without aborting the session.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any, Iterable, List, Optional, Sequence, Set

#: Directories under ``tests/`` whose modules are in scope, matched at ANY depth. A single entry
#: today; a tuple because the front-half A2A tests may land beside rather than below it.
DIST_DIRS = frozenset({"distributed"})

#: The three sanctioned spellings of a skip in scope. Matched on the ATTRIBUTE/function name, so
#: ``gate.should_skip(...)``, ``self.gate.should_skip(...)`` and a bare imported
#: ``rank_invariant_skip(...)`` all resolve.
SANCTIONED_SKIPS = frozenset({"gated_skip", "rank_invariant_skip", "pre_init_skip"})

#: Set while a sanctioned helper is calling ``pytest.skip`` on the caller's behalf, so the tripwire
#: can tell its own re-entry from a bare call. A plain module global rather than a context var: the
#: test session is single-threaded per rank, and a context var would not be visible to the poisoned
#: ``pytest.skip`` installed in another module's namespace.
_IN_SANCTIONED_SKIP = False

#: Populated by :func:`install_skip_tripwire`; read by :func:`skip_tripwire_problem` to detect a
#: test that restored the original and thereby disarmed the guard for every test after it.
_POISONED: dict = {}

#: Problems recorded by the tripwire during a test call, drained per test by the conftest hook.
_VIOLATIONS: List[str] = []


def _group_is_live() -> bool:
    """Whether a ``torch.distributed`` process group exists in this process right now.

    This is the predicate the owner's rule turns on: with a live group a rank CAN reduce, so
    deciding alone is never necessary; without one it has no choice.

    Returns:
        ``True`` only when torch is importable, distributed support is available AND a group is
        initialized. Any import failure returns ``False`` -- treating "no torch" as "no group",
        which is the conservative answer for a CPU-only collection pass.
    """
    try:
        import torch.distributed as dist

        return bool(dist.is_available() and dist.is_initialized())
    except Exception:  # noqa: BLE001 - a probe must never break collection
        return False


def rank_invariant_skip(reason: str, *, because: str) -> None:
    """Skip on a predicate every rank computes IDENTICALLY, declaring that it is uniform.

    Semantics
        A plain `pytest.skip`, with the declaration recorded at the call site. No collective is
        issued -- that is the whole point: a job-uniform predicate needs no reduction, and routing
        one through `CollectiveGate.should_skip` would spend a collective to learn what every rank
        already knows.

    Args:
        reason: Shown by pytest, as for a bare skip.
        because: WHY the predicate is rank-invariant, in the author's own words -- "derived from
            WORLD_SIZE, which every rank reads identically", not "it's fine". Keyword-only and
            non-empty; an empty or whitespace string raises, because a declaration nobody wrote is
            the state this function exists to make impossible. It is not checked for truth -- no
            system can do that -- only for existence, which is what removes the cheapness of
            forgetting.

    Returns:
        Never returns; raises pytest's ``Skipped``.

    Raises:
        ValueError: If ``because`` is empty or whitespace.
    """
    if not because or not because.strip():
        raise ValueError(
            "rank_invariant_skip(..., because=...) needs a written reason naming WHY every rank "
            "computes this predicate identically. If it can differ per rank -- a per-host import, "
            "a per-GPU capability, an OOM -- it is not rank-invariant and must go through "
            "CollectiveGate.should_skip so the ranks skip together."
        )
    global _IN_SANCTIONED_SKIP
    import pytest

    _IN_SANCTIONED_SKIP = True
    try:
        pytest.skip(reason)
    finally:
        _IN_SANCTIONED_SKIP = False


def gated_skip(local_reason: Optional[str], *, group=None, device=None) -> None:
    """Reduce a possibly-divergent skip decision across the group, then skip together or not at all.

    Semantics
        The sanctioned spelling for the case the guard exists for. Wraps
        `CollectiveGate.should_skip`, which all_reduces a 0/1 flag, and issues the `pytest.skip` only
        if the group agreed. **COLLECTIVE and unconditional**: every rank must reach this call on
        every path, which is exactly the property being enforced -- a rank that returns early before
        it is the defect.

        This exists as one call rather than the two-line ``agreed = gate.should_skip(r)`` /
        ``pytest.skip(agreed)`` because the second line is a bare `pytest.skip` that no static pass
        can distinguish from the dangerous kind. Folding them removes the ambiguity instead of
        carving an exception for it.

    Args:
        local_reason: THIS rank's reason to skip, or ``None`` to proceed. Only the local view --
            passing a reason another rank computed defeats the reduction.
        group: Process group to reduce over; ``None`` means the default group. Every rank must pass
            the SAME group, or the reduction deadlocks on mismatched collectives.
        device: Device for the reduction tensor; ``None`` derives ``cuda:<local_rank>``. Under NCCL
            this must be the calling rank's own device or the collective raises.

    Returns:
        ``None`` when the group agreed nobody skips. Otherwise never returns -- raises pytest's
        ``Skipped`` on EVERY rank, with this rank's own reason when it had one and a note that a
        peer asked when it did not, so a log never claims a condition the local rank did not see.

    Raises:
        RuntimeError: From `CollectiveGate` if no process group is initialized. That is the correct
            failure: with no group there is nothing to reduce over, and the site wants
            :func:`pre_init_skip` instead.
    """
    from fold_cp_ops.distributed.collective_symmetry import CollectiveGate

    agreed = CollectiveGate(group=group, device=device).should_skip(local_reason)
    if agreed is None:
        return
    global _IN_SANCTIONED_SKIP
    import pytest

    _IN_SANCTIONED_SKIP = True
    try:
        pytest.skip(agreed)
    finally:
        _IN_SANCTIONED_SKIP = False


def pre_init_skip(reason: str, *, because: str) -> None:
    """Skip BEFORE any process group exists -- the one circumstance a rank may decide alone.

    Semantics
        **Self-policing, not declared.** The call asserts that no ``torch.distributed`` group is
        initialized and raises if one is, so this cannot be used to wave through a divergent skip in
        a test that runs after ``DistributedManager.initialize()``. That is deliberate and is the
        rule this module was asked to enforce: once a group exists a rank can always reduce, so
        deciding alone is a choice rather than a necessity, and `CollectiveGate.should_skip` is the
        only correct spelling.

        A pre-init divergent skip is genuinely unfixable in-place today -- reducing the decision
        needs a vote channel that predates the group, and that is chicken-and-egg: you cannot
        all-reduce "should we init?" over the group you have not initialized. A gloo side channel
        would have to BE the default group, which collides with `DistributedManager.initialize()`
        over ownership; a `TCPStore` vote sidesteps that and is the open design question. Every call site here is therefore visible, enumerated debt that closes when
        that channel lands, rather than an invisible hazard.

    Args:
        reason: Shown by pytest, as for a bare skip.
        because: Why this site is unavoidably pre-init -- name the fixture or phase, e.g. "runs
            inside dist_manager before initialize(); no group to reduce over". Keyword-only and
            non-empty.

    Returns:
        Never returns; raises pytest's ``Skipped``.

    Raises:
        ValueError: If ``because`` is empty or whitespace.
        AssertionError: If a process group IS initialized. The message names the fix rather than
            the violation, because the fix is a one-line substitution.
    """
    if not because or not because.strip():
        raise ValueError(
            "pre_init_skip(..., because=...) needs a written reason naming the fixture or phase "
            "that makes this site unavoidably pre-init."
        )
    if _group_is_live():
        raise AssertionError(
            "pre_init_skip() was called while a torch.distributed process group IS initialized, so "
            "this site is not pre-init and the exemption does not apply to it. With a live group a "
            "rank can always reduce its decision, which makes deciding alone a choice rather than a "
            "necessity -- and a divergent skip here hangs every peer in the next collective. Use "
            "`gated_skip(reason_or_None)` instead; if the predicate is job-uniform, use "
            "`rank_invariant_skip(reason, because=...)` and say why it cannot differ per rank."
        )
    global _IN_SANCTIONED_SKIP
    import pytest

    _IN_SANCTIONED_SKIP = True
    try:
        pytest.skip(reason)
    finally:
        _IN_SANCTIONED_SKIP = False


def collective_scope(path) -> bool:
    """Whether a test module is governed by this guard.

    Args:
        path: Path to a test module, absolute or relative. Matched on the path PARTS, so
            ``tests/distributed/kernels/test_x_a2a.py`` is in scope without the guard knowing that
            directory exists.

    Returns:
        ``True`` when any path component is in `DIST_DIRS`. Directory-scoped deliberately -- see the
        module docstring on why scoping by "does it import torch.distributed" fails silently at
        exactly the moment a host-only module grows a collective.
    """
    return bool(DIST_DIRS & set(Path(path).parts))


def _is_pytest_skip(node: ast.AST) -> bool:
    """Whether a node is a call to pytest's skip, in either spelling.

    Args:
        node: Any AST node.

    Returns:
        ``True`` for ``pytest.skip(...)`` and for a bare ``skip(...)`` imported from pytest. A bare
        ``skip`` is included because ``from pytest import skip`` is legal and would otherwise walk
        straight past the guard; the cost is that a local helper named ``skip`` is also flagged,
        which is the safe direction to be wrong in.
    """
    if not isinstance(node, ast.Call):
        return False
    fn = node.func
    if isinstance(fn, ast.Attribute) and fn.attr == "skip":
        return isinstance(fn.value, ast.Name) and fn.value.id == "pytest"
    return isinstance(fn, ast.Name) and fn.id == "skip"


def divergent_skip_problems(path) -> List[str]:
    """Every bare ``pytest.skip`` in a module governed by this guard.

    Semantics
        A syntactic pass, and it does NOT try to decide whether a predicate is rank-invariant -- no
        static analysis can, and one that guessed would either wave through the dangerous cases or
        drown the safe ones. It asks only whether the author DECLARED which kind it is. That is the
        same bargain `unsupported=` strikes on `KernelMatrix`.

        The guard's own source is skipped: this module names the forbidden form in order to detect
        it, and flagging itself would make the check un-runnable.

    Args:
        path: Test module to scan. Out-of-scope paths return ``[]`` without being read, so calling
            this on every collected file is cheap.

    Returns:
        One message per offending call, each naming the line and the three legal substitutions.
        Empty when the module is out of scope, unreadable, or clean. A syntax error yields ``[]``
        rather than a problem -- pytest reports that far better than this guard would.
    """
    p = Path(path)
    if not collective_scope(p) or p.name == "collective_guard.py":
        return []
    try:
        tree = ast.parse(p.read_text())
    except (OSError, SyntaxError):
        return []

    problems: List[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not _is_pytest_skip(node):
            continue
        problems.append(
            f"{p.name}:{node.lineno}: bare pytest.skip() under tests/distributed/. A skip on a "
            f"predicate that can differ per rank is a DEADLOCK: the skipping rank leaves and every "
            f"peer blocks in the next collective. Use one of "
            f"`gated_skip(reason_or_None)` (group is up -- reduce the decision), "
            f"`rank_invariant_skip(reason, because=...)` (every rank computes it identically), or "
            f"`pre_init_skip(reason, because=...)` (no group exists yet; raises if one does)."
        )
    problems.extend(_unreachable_gated_skip_problems(p, tree))
    return problems


def _gated_skip_helpers(tree: ast.Module) -> set:
    """Names of module-local functions whose body calls ``gated_skip``.

    A call to one of these is treated as a ``gated_skip`` by
    :func:`_unreachable_gated_skip_problems`, because wrapping the reduction in a one-line helper is
    exactly how the placement bug hid: a module defined ``_oom_gate(exc, what)`` that called
    ``gated_skip``, and every call site passed an already-caught exception -- so it was reachable
    only from an ``except`` block, and a scan looking for the literal name saw nothing.

    Args:
        tree: The parsed module.

    Returns:
        The set of such function names. One level deep only; a helper calling a helper is not
        followed, which is a deliberate floor rather than an oversight -- the check is a tripwire for
        an easy mistake, not a call-graph analysis.
    """
    helpers = set()
    for fn in ast.walk(tree):
        if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for call in ast.walk(fn):
            if isinstance(call, ast.Call) and getattr(call.func, "id", None) == "gated_skip":
                helpers.add(fn.name)
                break
    return helpers


def _unreachable_gated_skip_problems(p: Path, tree: ast.Module) -> List[str]:
    """Every ``gated_skip`` that only SOME ranks can reach.

    Semantics
        ``gated_skip`` all-reduces, so **every rank must reach it on every path**. Its own docstring
        says so. Placing it inside an ``except`` handler breaks that in the most damaging way
        available: the ranks that raised enter the reduction and the ranks that did not never
        arrive, so the raisers block on peers who have moved on. NCCL matches collectives by issue
        order, not by tag, so the result is a hang or a mismatched reduction -- not a skip.

        This is not hypothetical and it is not rare. Seven sites in this repo placed it that way,
        and three of them carried a comment correctly explaining why ``gated_skip`` was the right
        primitive before putting it where the reduction cannot happen. The hazard was understood;
        the placement defeated it. That is why this is a check and not a paragraph.

        The sanctioned shape is to RECORD the reason in the handler and call ``gated_skip(reason)``
        after the ``try``, where every rank arrives with either a reason or ``None``.

    What is NOT flagged
        A ``gated_skip`` inside an ``if`` is allowed. A rank-uniform predicate makes the reduction a
        no-op, and this pass cannot decide uniformity -- guessing would either wave through the
        dangerous cases or drown the safe ones, the same reason the bare-skip pass above declines to
        judge predicates. The ``except`` case needs no judgement: an exception is never job-uniform.

    Args:
        p: Module path, for the message.
        tree: The parsed module.

    Returns:
        One message per offending call site, naming the line and the fix. Empty when clean.
    """
    helpers = _gated_skip_helpers(tree)
    problems: List[str] = []
    for handler in ast.walk(tree):
        if not isinstance(handler, ast.ExceptHandler):
            continue
        for call in ast.walk(handler):
            if not isinstance(call, ast.Call):
                continue
            name = getattr(call.func, "id", None) or getattr(call.func, "attr", None)
            if name != "gated_skip" and name not in helpers:
                continue
            via = "" if name == "gated_skip" else f" (via `{name}`, which calls it)"
            problems.append(
                f"{p.name}:{call.lineno}: `gated_skip`{via} inside an `except` handler. It "
                f"ALL-REDUCES, so every rank must reach it on every path -- here only the ranks "
                f"that RAISED do, and they block in a collective their peers never enter. Record "
                f"the reason in the handler and call `gated_skip(reason_or_None)` AFTER the `try`, "
                f"so every rank arrives with either a reason or None."
            )
    return problems


def audit_collective_module(path, module: Any = None) -> List[str]:
    """Every collective-symmetry problem in one test module, for the collection-time hook.

    Args:
        path: Test module path.
        module: The imported module, accepted for symmetry with
            `numeric_guard.audit_numeric_module` and currently unused -- the checks here are all
            syntactic or runtime, with nothing to read off the module object.

    Returns:
        The list from :func:`divergent_skip_problems`; empty for an out-of-scope or clean module.
        Never raises, so one non-conforming file marks its own items and does not abort the session
        (the failure mode an earlier version of the matrix audit had, which turned ``pytest tests/``
        into "no tests ran").
    """
    return divergent_skip_problems(path)


def install_skip_tripwire() -> None:
    """Poison ``pytest.skip`` for the session so a bare call under a LIVE group fails.

    Semantics
        The runtime half, and the layer that actually enforces the owner's rule -- only at call time
        is ``dist.is_initialized()`` knowable, so only here can "you had a group and decided alone"
        be distinguished from "there was nothing to reduce over".

        The poison is conditional, not absolute: a bare skip with NO live group passes straight
        through. That keeps the guard silent for the pure-host modules in the directory
        (``test_layout_map`` touches ``dist`` zero times) and for collection-time skips, while still
        failing the case that hangs a job. A sanctioned helper sets `_IN_SANCTIONED_SKIP` around its
        own call, so the tripwire sees its re-entry and lets it through.

        Poisons the ATTRIBUTE on the ``pytest`` module rather than a name, so a call built through
        ``getattr(pytest, "sk" + "ip")`` fails too. Idempotent -- installing twice keeps the first
        poison, so a second call cannot capture the poisoned function as "the original".

    Returns:
        None; mutates ``pytest`` in place and is deliberately never restored.
        :func:`skip_tripwire_problem` treats restoration as a violation.
    """
    import pytest

    if "pytest.skip" in _POISONED:
        return
    original = pytest.skip

    def _guarded(reason="", *args, **kwargs):
        if not _IN_SANCTIONED_SKIP and _group_is_live():
            _VIOLATIONS.append(
                f"bare pytest.skip({reason!r}) with a LIVE torch.distributed group. If this "
                f"predicate can differ per rank, the ranks that did not skip block in the next "
                f"collective until a watchdog kills the job. Reduce the decision with "
                f"`gated_skip(reason_or_None)`, or -- if every rank computes it "
                f"identically -- say so with `rank_invariant_skip(reason, because=...)`. "
                f"`pre_init_skip` does not apply: a group is already initialized."
            )
        return original(reason, *args, **kwargs)

    _POISONED["pytest.skip"] = (pytest, "skip", _guarded, original)
    pytest.skip = _guarded


def skip_tripwire_problem() -> Optional[str]:
    """Report whether the tripwire is still armed, naming what put it back if not.

    **The anti-evasion check.** ``monkeypatch.setattr(pytest, "skip", orig)`` inside a test would
    disarm the guard for that test and, with no explicit restore, for every test after it. Checking
    at each teardown attributes the disarming to the test that did it rather than leaving a suite
    that quietly stopped enforcing.

    Returns:
        ``None`` when the poisoned attribute is still the guard's callable, or when the tripwire was
        never installed (a session with no distributed tests is not failed for opting out).
        Otherwise a message naming what was restored.
    """
    restored = [
        key
        for key, (owner, attr, poison, _orig) in _POISONED.items()
        if getattr(owner, attr, None) is not poison
    ]
    if not restored:
        return None
    return (
        f"the collective-skip tripwire was disarmed during this test: {', '.join(sorted(restored))} "
        f"is no longer the guard's callable. Restoring the original turns the guard off for every "
        f"test after this one."
    )


def drain_violations() -> Sequence[str]:
    """Take and clear the violations the tripwire recorded, for the per-test hook.

    Returns:
        The recorded messages, in call order; the buffer is empty afterwards so the next test starts
        clean. Draining rather than reading is what attributes a violation to the test that caused
        it instead of to every test after it.
    """
    out = list(_VIOLATIONS)
    _VIOLATIONS.clear()
    return out


def reset_violations() -> None:
    """Clear the violation buffer without reporting, for use at test SETUP.

    A skip raised during collection or during another test's teardown would otherwise be attributed
    to whichever test ran next. Called from the setup hook so each test is judged only on what it
    itself did.

    Returns:
        None.
    """
    _VIOLATIONS.clear()


def sanctioned_skip_names() -> Iterable[str]:
    """The sanctioned spellings, for an error message or a test to enumerate.

    Returns:
        The contents of `SANCTIONED_SKIPS`, sorted, so a message listing them is stable across runs.
    """
    return sorted(SANCTIONED_SKIPS)


def scoped_dirs() -> Set[str]:
    """The directory names this guard governs.

    Returns:
        A copy of `DIST_DIRS` as a mutable set, so a caller can inspect or extend it locally without
        mutating the module constant.
    """
    return set(DIST_DIRS)
