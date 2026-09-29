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

"""Every numerical comparison in the suite goes through :mod:`fold_cp_ops.testing.numerics`.

**The defect this exists to stop.** Five e2e tests asserted ``max|diff| / max|ref| <= 0.02``. That
divides the worst deviation by the LARGEST reference magnitude, so an element whose own value is
small but wrong by 100% is invisible. Demonstrated rather than argued: corrupting ONE element of a
``(1,128,128,128)`` output left the pooled metric at 0.01121, comfortably inside the 0.02 bar, while
the element-wise form reported ``worst ratio 1.121`` *with the offending index*. Every defect class
this repo actually hits -- a tile edge, a padded lane, a straddle row, a last-wave CTA -- is a
handful of elements out of millions, which is precisely what a pooled statistic cannot see.

**Whitelist, not blacklist.** Pooling has unbounded spellings (``.max()``, ``.mean()``, ``.norm()``,
``.sum()``, ``.quantile()``, a hand-rolled loop), so enumerating the bad forms is whack-a-mole. The
sanctioned set is finite and already exists: :data:`SANCTIONED_ASSERTIONS`. Everything else is a
violation.

**Three layers, because each catches what the others cannot.**

1. :func:`forbidden_comparison_problems` -- a static AST pass, run at collection. Catches the known
   spellings and the aliased imports that would hide them from a runtime attribute patch.
2. :func:`install_tripwire` -- poisons ``torch.allclose`` and ``torch.testing.assert_close`` for the
   duration of the session. Catches calls the AST pass cannot see: built through ``getattr``,
   reached from a helper module, or synthesised at runtime.
3. :func:`numeric_scope` + :func:`record_assertion` -- the COVERAGE gate. A matrix-parametrized test
   that runs a kernel and records NO sanctioned assertion fails. **This is the layer that catches
   the original defect**, because it asks "did a sanctioned comparison happen" rather than "was a
   forbidden one avoided" -- and the five pooled tests would have been caught by it even if their
   spelling had been novel.

**On ``torch.testing.assert_close`` specifically, so nobody re-derives this wrongly.** It is *not*
a pooled comparison -- it checks every element and reports the worst with its index. It is banned
for a different and weaker reason: its bound is a hand-written ``atol + rtol * |expected|`` rather
than one of this repo's DERIVED error bounds, and a hand-written bar is exactly where a too-loose
tolerance hides. Converting a call site is mechanical -- see
:func:`fold_cp_ops.testing.numerics.tolerance_bound`.

**Anti-evasion.** The guard is default-deny and self-verifying: a module in a numeric directory with
no in-scope test and no declared exemption fails, an exemption without a literal reason fails, and
:func:`tripwire_problem` fails the test that restored ``torch.allclose`` rather than letting the
restoration go unnoticed.

**Failure scoping.** Nothing here aborts a session. Problems are returned as strings for the caller
to attach to the offending module's items, exactly as ``kernel_matrix.audit_test_module`` does. An
earlier audit in this repo raised at collection and one non-conforming file turned ``pytest tests/``
into "no tests ran", destroying the signal from every unrelated test.

**Known limit, stated rather than discovered later:** none of this constrains a too-LOOSE bound.
``assert_elementwise(got, ref, 1e9)`` satisfies every layer. Review is the backstop, helped by the
one-line corruption experiment above.
"""

import ast
import pathlib
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set

import pytest

#: The comparison helpers a test is allowed to use. All live in
#: :mod:`fold_cp_ops.testing.numerics` and all judge PER ELEMENT, reporting the worst offender with
#: its index and both values. Adding a name here widens what the coverage gate accepts, so it is the
#: one place that decision is made.
SANCTIONED_ASSERTIONS = frozenset(
    {
        "assert_elementwise",
        "assert_bitwise",
        "assert_gemm_exact",
        "assert_gemm_close",
        "assert_written",
    }
)

#: Leaf names that are a numerical comparison no matter what they are spelled through.
#: ``torch.allclose``, ``np.allclose``, ``torch.testing.assert_close``,
#: ``np.testing.assert_allclose``, and any local rebinding of them.
FORBIDDEN_LEAVES = frozenset({"allclose", "assert_close", "assert_allclose"})

#: Reduction methods that collapse a tensor to a scalar. Forbidden only when applied to an
#: expression containing a subtraction inside an ``assert`` -- that combination IS the pooled-error
#: idiom, whereas ``median(times)`` in a perf harness is legitimate and has no subtraction.
POOLING_REDUCTIONS = frozenset(
    {"max", "mean", "norm", "sum", "amax", "median", "std", "var", "quantile", "item"}
)

#: Directories under ``tests/`` whose modules are kernel/workflow correctness tests, and therefore
#: subject to the module-level default-deny rule. A module here that neither has an in-scope test
#: nor declares :data:`MODULE_EXEMPT_ATTR` is refused -- otherwise adding a file would be a way to
#: opt out of the whole guard.
#:
#: **Matched on path PARTS at any depth**, so ``tests/distributed/kernels/`` and
#: ``tests/distributed/workflows/`` are governed before those directories exist. The earlier
#: ``parent.name`` spelling covered them only because their leaf names happen to collide with the
#: ones written for ``tests/kernels`` -- a coincidence, not a decision, and one that would have
#: evaporated on a rename. Same rule and spelling as :data:`collective_guard.DIST_DIRS`.
NUMERIC_DIRS = frozenset({"kernels", "workflows"})

#: Directories excluded from the coverage gate. A perf gate imports the same matrix and parametrizes
#: from it, so every syntactic test says "in scope" -- but its subject is a median against a pin, not
#: an output it produces. Matched on parts, so a sibling ``tests/distributed/perf/`` is excluded for
#: the same reason ``tests/perf/`` is, rather than needing this list edited when it appears.
NUMERIC_EXCLUDED_DIRS = frozenset({"perf"})

#: Module-level constant a test module sets to opt out of the STATIC pass and the default-deny
#: rule, e.g. ``NUMERIC_EXEMPT = "pure dispatch-validation module; launches no kernel"``. Must be a
#: non-empty string literal, for the same reason ``matrix_exempt``'s reason must be: it is read from
#: the AST, so a name lookup reads as no reason at all. It does NOT waive the coverage gate, so an
#: exempt kernel module still has to compare its output with a sanctioned assertion.
MODULE_EXEMPT_ATTR = "NUMERIC_EXEMPT"

#: Sanctioned assertions that have run since the last :func:`reset_assertions`. A plain module-level
#: list rather than a fixture value because :mod:`fold_cp_ops.testing.numerics` must be able to
#: record without importing pytest or knowing a test is in progress.
_RECORDED: List[str] = []

#: What :func:`install_tripwire` replaced, as ``"<module>.<attr>" -> (owner, attr, poison)``. The
#: owner and attribute are kept alongside the callable so :func:`tripwire_problem` can re-read the
#: live attribute and tell "still poisoned" from "somebody put the original back" -- the evasion the
#: guard has to catch rather than tolerate.
_POISONED: dict = {}


def record_assertion(name: str) -> None:
    """Record that a sanctioned comparison ran, for the coverage gate to read at teardown.

    Called by every function in :data:`SANCTIONED_ASSERTIONS`. Cheap by construction (one list
    append) because it is on the path of assertions that already move megabytes.

    Args:
        name: The assertion's function name. Not validated against
            :data:`SANCTIONED_ASSERTIONS` -- an unknown name recorded here would only ever *satisfy*
            the gate for a caller who already imported the module, and rejecting it would put a
            failure inside an assertion helper, where a raise is indistinguishable from the
            comparison failing.

    Returns:
        None; appends to the process-global record.
    """
    _RECORDED.append(name)


def assertions_recorded() -> Sequence[str]:
    """The sanctioned assertions that have run since the last reset.

    Returns:
        A tuple of names, in call order. Empty when nothing sanctioned has run, which is exactly
        what the coverage gate fails on for an in-scope test.
    """
    return tuple(_RECORDED)


def reset_assertions() -> None:
    """Clear the record, so one test's assertions cannot satisfy the next test's coverage gate.

    Called at the start of every test's call phase. Forgetting it would make the gate pass for
    every test after the first sanctioned one in the session -- a silently vacuous check, which is
    worse than no check because it reads as coverage.

    Returns:
        None.
    """
    _RECORDED.clear()


def numeric_exempt(reason: str):
    """Mark a test as legitimately making no element-wise numerical comparison.

    For matrix-parametrized tests whose subject is not a computed value -- a launch-count assertion,
    a cache-key property, a shape/dtype contract, a raise. Without this the coverage gate fails the
    test for recording no sanctioned assertion; with it, the exemption is a sentence in the source
    a reviewer can disagree with. Same shape as ``kernel_matrix.matrix_exempt`` on purpose: an
    escape hatch that costs a written reason is one people use honestly.

    Args:
        reason: Why no numerical comparison applies. Must be a non-empty **string literal at the
            decoration site** -- the audit reads it from the AST, so an f-string or a name lookup
            reads as no reason at all and is rejected there.

    Returns:
        A pytest marker. Purely declarative: it changes nothing about how the test runs.

    Raises:
        ValueError: If ``reason`` is empty or whitespace.
    """
    if not (reason and reason.strip()):
        raise ValueError("numeric_exempt() requires a non-empty reason")
    return pytest.mark.numeric_exempt(reason)


def _decorator_sources(fn: ast.FunctionDef) -> List[str]:
    """Unparse a function's decorators once, so the callers below can match on text.

    Args:
        fn: The function node.

    Returns:
        One source string per decorator, in source order.
    """
    return [ast.unparse(d) for d in fn.decorator_list]


def _is_matrix_parametrized(decs: Iterable[str]) -> bool:
    """Whether a test draws its cases from a :class:`~fold_cp_ops.testing.kernel_matrix.KernelMatrix`.

    ``parametrize_unsupported`` is deliberately excluded: those cases assert that the kernel
    REFUSES a combo, so there is no computed value to compare and requiring one would make the
    coverage gate fail correct tests.

    Args:
        decs: Unparsed decorator sources.

    Returns:
        True for ``<matrix>.parametrize(...)``, False for a bare ``pytest.mark.parametrize`` (which
        is the hand-rolled list the matrix exists to replace) and for ``parametrize_unsupported``.
    """
    return any(
        ".parametrize(" in d and not d.startswith("pytest.mark.parametrize") for d in decs
    ) and not any(".parametrize_unsupported(" in d for d in decs)


def _has_exempt(decs: Iterable[str]) -> bool:
    """Whether a test carries :func:`numeric_exempt` (with any argument).

    Args:
        decs: Unparsed decorator sources.

    Returns:
        True if any decorator names ``numeric_exempt``. Whether its reason is a literal is checked
        separately, by :func:`forbidden_comparison_problems`, so a bad reason is reported as its own
        problem rather than silently un-exempting the test.
    """
    return any("numeric_exempt(" in d for d in decs)


def _contains_subtraction(node: ast.AST) -> bool:
    """Whether an expression subtree contains a ``-`` between two operands.

    The distinguishing feature of the pooled-error idiom: ``(got - ref).abs().max()`` reduces a
    DIFFERENCE, where a perf harness's ``median(times)`` reduces a sample.

    Args:
        node: Any AST node; walked in full.

    Returns:
        True if a ``BinOp`` with ``Sub`` appears anywhere beneath it.
    """
    return any(isinstance(n, ast.BinOp) and isinstance(n.op, ast.Sub) for n in ast.walk(node))


#: The reductions that can ORIGINATE a pooled scalar. :data:`POOLING_REDUCTIONS` minus ``item``,
#: because ``.item()`` does not pool anything -- it unwraps a tensor that is ALREADY a scalar, so
#: the pooling verb is whichever call produced that scalar. Keeping ``item`` as an originator makes
#: ``(row_sum(x, fill=1.0) - base)[0].item()`` -- an indexed read of ONE element -- look like a
#: reduction. It still PROPAGATES, so ``(a - b).abs().max().item()`` is caught by its ``max``.
_SCALAR_ORIGINATORS = POOLING_REDUCTIONS - {"item"}

#: Reductions whose FIRST POSITIONAL argument is a dim. Used with :data:`_SCALAR_ORIGINATORS` to
#: decide whether a call collapses to a scalar at all.
_DIM_TAKING = frozenset({"max", "sum", "mean", "norm", "amax", "median", "std", "var", "quantile"})


def _keeps_an_axis(call: ast.Call) -> bool:
    """Whether a reduction names a dim/axis, and therefore does NOT collapse to a scalar.

    Purpose
        A per-axis reduction cannot shadow an element the way a scalar does -- it is the localized
        error statistic this repo deliberately keeps. Measured instance:
        ``rr = (got - ref).norm(dim=0) / ref.norm(dim=0); n_out = int((rr > 1e-2).sum());
        assert n_out == 0`` is a PER-COLUMN outlier count, and flagging it would fail the one check
        that catches a localized store corruption.

    Args:
        call: A reduction call whose method name is in :data:`_SCALAR_ORIGINATORS`.

    Returns:
        True when a ``dim=``/``axis=`` keyword is present, or when a positional argument is given to
        a reduction that takes a dim first. Conservative in the SAFE direction: a reduction whose
        first positional argument is something else reads as axis-preserving and is not reported,
        which loses a finding rather than inventing one.
    """
    if any(k.arg in ("dim", "axis") for k in call.keywords):
        return True
    return bool(call.args) and call.func.attr in _DIM_TAKING


def _difference_is_already_consumed(receiver: ast.AST) -> bool:
    """Whether every subtraction under ``receiver`` has already stopped being a raw magnitude.

    Purpose
        Two measured FALSE POSITIVES, one rule. Both are reductions applied to something that is no
        longer a difference by the time they run, so calling them "a pooled error" is simply wrong:

        * **an axis-keeping reduction upstream.**
          ``int((((got - ref).norm(dim=0) / ref.norm(dim=0)) > 1e-2).sum())`` is a PER-COLUMN
          outlier COUNT -- the localized-corruption statistic this repo deliberately keeps. The
          ``dim=0`` is right there, and the outer ``.sum()`` cannot see it. Before this, the
          statistic was protected only when the author happened to bind it to a temp name first,
          which is not a property anybody would predict.
        * **a comparison upstream.**
          ``((got - ref).abs() > tol).sum()`` reduces BOOLEANS. It is a genuinely per-element gate
          -- one bad element makes the count non-zero -- and the old rule called it pooled, which is
          backwards.

    Semantics
        Walks from ``receiver`` down to each ``Sub`` and asks whether the path passes through an
        axis-keeping reduction call or a ``Compare``. Every subtraction must be consumed for the
        reduction to be cleared: a receiver mixing a consumed difference with a raw one is still
        reducing a raw one.

    Args:
        receiver: The expression a reduction is applied to (the call's ``func.value``).

    Returns:
        True when there is at least one subtraction and all of them are consumed. False when a raw
        difference survives, and False when there is no subtraction at all -- the caller has already
        established there is one, and "no subtraction" is not this function's business.
    """

    def consumed(node: ast.AST, shielded: bool) -> tuple:
        """Returns ``(saw_sub, all_consumed)`` for the subtree under ``node``."""
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Sub):
            return True, shielded
        shield = shielded or isinstance(node, ast.Compare)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr in POOLING_REDUCTIONS and _keeps_an_axis(node):
                shield = True
        saw, ok = False, True
        for child in ast.iter_child_nodes(node):
            s, o = consumed(child, shield)
            saw = saw or s
            ok = ok and o
        return saw, ok

    saw_sub, all_consumed = consumed(receiver, False)
    return saw_sub and all_consumed


def _pooled_calls(node: ast.AST) -> List[ast.Call]:
    """Every call beneath ``node`` that reduces a DIFFERENCE to a scalar.

    Args:
        node: Any AST node; walked in full.

    Returns:
        The qualifying calls, in walk order. A call qualifies on FOUR conditions together -- the
        method is a scalar originator, its receiver contains a subtraction, it does not itself keep
        an axis, and that subtraction has not ALREADY been consumed upstream by an axis-keeping
        reduction or a comparison (:func:`_difference_is_already_consumed`). Each condition alone is
        common and legitimate; the fourth is what makes the rule survive being written inline rather
        than through a temp variable.
    """
    return [
        c
        for c in ast.walk(node)
        if isinstance(c, ast.Call)
        and isinstance(c.func, ast.Attribute)
        and c.func.attr in _SCALAR_ORIGINATORS
        and _contains_subtraction(c.func.value)
        and not _keeps_an_axis(c)
        and not _difference_is_already_consumed(c.func.value)
    ]


def _names(node: ast.AST) -> Set[str]:
    """Every bare name appearing beneath ``node``.

    Args:
        node: Any AST node.

    Returns:
        The set of ``ast.Name`` identifiers, load or store alike -- the caller decides which it
        wants by where it looks.
    """
    return {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}


def _callee_names(node: ast.AST) -> Set[str]:
    """The leaf name of every call beneath ``node``, so a call to a local helper is recognisable.

    Args:
        node: Any AST node.

    Returns:
        Bare-call identifiers and attribute leaves together (``f(...)`` -> ``f``, ``o.f(...)`` ->
        ``f``). The leaf is enough here: this only ever asks whether the name matches a function
        DEFINED IN THE SAME MODULE, and an unrelated method that happens to share the name would at
        worst produce a finding a reader can dismiss.
    """
    return {
        (c.func.id if isinstance(c.func, ast.Name) else getattr(c.func, "attr", ""))
        for c in ast.walk(node)
        if isinstance(c, ast.Call)
    }


#: Bound on both taint fixpoints. Each iteration can only ADD names, and a test module's call graph
#: is shallow, so this converges in two or three passes in practice; the bound exists so a
#: pathological module costs a missed finding rather than a hung collection.
_TAINT_PASSES = 8


def _pooled_returning_functions(funcs: Sequence[ast.FunctionDef]) -> Set[str]:
    """The functions whose RETURN value is a pooled scalar, transitively.

    Purpose
        This is the half that was missing, and it is where the original defect lived::

            def _rel_err(out, ref):
                return (out.float() - ref.float()).abs().max().item() / scale
            ...
            rel = _rel_err(recv, expected)
            assert rel < REL_BAR

        The reduction is in a ``return``, so an ``assert``-anchored check reported nothing at all.

    Args:
        funcs: Every ``FunctionDef`` in the module.

    Returns:
        Function NAMES, not nodes -- the caller matches them against call sites, which is a name
        comparison. Transitive: a function returning a call to a tainted function is tainted too,
        computed to a fixpoint.
    """
    tainted: Set[str] = set()
    for _ in range(_TAINT_PASSES):
        grew = False
        for fn in funcs:
            if fn.name in tainted:
                continue
            local = _tainted_locals(fn, tainted)
            for node in ast.walk(fn):
                if not isinstance(node, ast.Return) or node.value is None:
                    continue
                if (
                    _pooled_calls(node.value)
                    or (_names(node.value) & set(local))
                    or (_callee_names(node.value) & tainted)
                ):
                    tainted.add(fn.name)
                    grew = True
                    break
        if not grew:
            break
    return tainted


def _tainted_locals(fn: ast.FunctionDef, tainted_funcs: Set[str]) -> Dict[str, tuple]:
    """Local names holding a pooled scalar, mapped to where that scalar came from.

    Semantics
        A name is tainted when it is assigned from a pooled call, from another tainted name, or
        from a call to a function in ``tainted_funcs``. Iterated to a fixpoint so an intermediate
        (``a = pooled(); b = a; assert b < bar``) is followed. Flow-insensitive by design: it does
        not model rebinding or branches, which for a test module costs nothing and keeps this small
        enough to read.

    Args:
        fn: The function to scan.
        tainted_funcs: Names of module functions that return a pooled scalar.

    Returns:
        ``name -> (lineno, verb)`` naming the ORIGIN, so a report can point at the reduction rather
        than only at the assert that consumed it.
    """
    local: Dict[str, tuple] = {}
    for _ in range(_TAINT_PASSES):
        before = dict(local)
        for node in ast.walk(fn):
            if not isinstance(node, ast.Assign):
                continue
            pooled = _pooled_calls(node.value)
            if pooled:
                origin = (pooled[0].lineno, f".{pooled[0].func.attr}()")
            else:
                origin = next(
                    (local[n] for n in sorted(_names(node.value)) if n in local),
                    next(
                        (
                            (node.lineno, f"{f}()")
                            for f in sorted(_callee_names(node.value))
                            if f in tainted_funcs
                        ),
                        None,
                    ),
                )
            if origin is None:
                continue
            for target in node.targets:
                for t in ast.walk(target):
                    if isinstance(t, ast.Name):
                        local[t.id] = origin
        if local == before:
            break
    return local


def _pooled_reduction_problem(
    node: ast.Assert, local: Dict[str, tuple], tainted_funcs: Set[str]
) -> Optional[str]:
    """Describe the pooled scalar this ``assert`` decides on, however it got there.

    Semantics
        Reports when the assert's test holds a pooled scalar -- computed inline, reached through a
        local name, or returned by a module function. One finding per ASSERT, not per reduction, so
        the count stays one-per-offending-verdict.

        **An assert whose BOTH sides are pooled is NOT reported.** That shape is an error-vs-error
        claim rather than a threshold gate: measured instance,
        ``e0 = (a - ref).abs().max(); e1 = (b - ref).abs().max(); assert e1 <= 2 * e0 + 1e-6``
        asserts that one variant is not LESS accurate than another, and no per-element bound
        expresses that. The banned shape is a pooled scalar against a CONSTANT.

    Args:
        node: The ``assert`` statement.
        local: This function's tainted locals, from :func:`_tainted_locals`.
        tainted_funcs: From :func:`_pooled_returning_functions`.

    Returns:
        A human-readable description naming the origin and the route, or None when the assert is
        clean. The caller wraps it with the filename and the remedy.
    """

    def _tainted(expr: ast.AST) -> bool:
        return bool(
            _pooled_calls(expr)
            or (_names(expr) & set(local))
            or (_callee_names(expr) & tainted_funcs)
        )

    test = node.test
    if isinstance(test, ast.Compare) and len(test.comparators) == 1:
        if _tainted(test.left) and _tainted(test.comparators[0]):
            return None
    if not _tainted(test):
        return None

    inline = _pooled_calls(test)
    if inline:
        return f"reduces a difference with .{inline[0].func.attr}() in the assert itself"
    for name in sorted(_names(test) & set(local)):
        lineno, verb = local[name]
        where = "on this line" if lineno == node.lineno else f"on line {lineno}"
        return f"decides on `{name}`, a scalar pooled by {verb} {where}"
    for fname in sorted(_callee_names(test) & tainted_funcs):
        return f"decides on the return value of {fname}(), which pools a difference to a scalar"
    return None


def forbidden_comparison_problems(path) -> List[str]:
    """Statically flag every unsanctioned numerical comparison in one test module.

    Three checks, each catching a different evasion:

    1. **Calls** to a :data:`FORBIDDEN_LEAVES` name, however qualified. ``torch.allclose``,
       ``tt.assert_close``, and a bare ``allclose`` all reduce to the same leaf.
    2. **Imports** that rebind one of those names -- ``from torch import allclose as ac`` would make
       check 1 miss the call site, so the import itself is the violation.
    3. **Pooled reductions** inside an ``assert``: a reduction applied to a difference. This is the
       original defect's exact shape and has no sanctioned spelling.

    Also validates that every ``numeric_exempt`` reason is a non-empty string literal, since an
    exemption whose reason cannot be read is not an exemption.

    Args:
        path: Path to the test module. Read and parsed; must be real source, not a temporary copy
            under a different name, because the reported problems quote its filename.

    Returns:
        A list of human-readable problems, empty when the module conforms. Each names the line and
        the replacement to use.

    Raises:
        SyntaxError: If the module does not parse. Callers that must not abort a session should
            wrap this, as ``tests/conftest.py`` does.
    """
    path = pathlib.Path(path)
    tree = ast.parse(path.read_text())
    problems: List[str] = []
    # A module may legitimately define its own `assert_close` that delegates to this module's
    # element-wise helpers -- three do, deliberately, so their call sites read as they always have.
    # A BARE call to a locally-defined name provably is not torch's, since the only ways torch's can
    # be reached are qualified (flagged below), imported (flagged below) or via getattr (caught by
    # the runtime tripwire, which poisons the attribute rather than the name). The local definition
    # is audited by the same pass, so a shadow that pools is still caught -- by its own body.
    local_defs = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}

    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            spelled = ast.unparse(node.func)
            leaf = spelled.split(".")[-1]
            if leaf in FORBIDDEN_LEAVES and not (spelled == leaf and leaf in local_defs):
                problems.append(
                    f"{path.name}:{node.lineno}: {ast.unparse(node.func)}(...) is not a sanctioned "
                    f"comparison. Use fold_cp_ops.testing.numerics.assert_elementwise(actual, "
                    f"reference, bound) -- for a plain atol/rtol pair, "
                    f"bound=numerics.tolerance_bound(reference, atol, rtol)."
                )
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                if alias.name.split(".")[-1] in FORBIDDEN_LEAVES:
                    problems.append(
                        f"{path.name}:{node.lineno}: importing {alias.name!r} binds a forbidden "
                        f"comparison under a name the call-site check cannot see. Import from "
                        f"fold_cp_ops.testing.numerics instead."
                    )
    # The pooled-scalar pass needs the module's taint map, so it runs as its own walk rather than
    # inside the one above. Anchoring on the ASSERT and following the value BACKWARDS is what fixed
    # the hole: the check used to look only INSIDE the assert's own expression, so a reduction
    # computed in a `return` (or in an assignment two lines up) was invisible. Measured before this
    # change: 0 findings across the whole `tests/` tree, on a tree containing four of them.
    funcs = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
    tainted_funcs = _pooled_returning_functions(funcs)
    for fn in funcs:
        # A PER-TEST `@numeric_exempt("why")` waives this finding, and ONLY this finding. A test
        # whose subject IS the pooled gate -- proving it is weaker than the element-wise one -- has
        # to spell the forbidden form, and converting it would delete the comparison the test
        # exists to make. The alternative, a module-level NUMERIC_EXEMPT, is refused by CLAUDE.md
        # for the whole file and rightly: it would switch the static pass off for every line to
        # silence two, and "each new entry checks one module less". The forbidden-LEAF and import
        # checks above are deliberately NOT waivable this way, so `torch.allclose` stays banned in
        # an exempt test too -- the exemption is for the shape of a comparison, not for reaching
        # past the sanctioned set.
        if _has_exempt(_decorator_sources(fn)):
            continue
        local = _tainted_locals(fn, tainted_funcs)
        for node in ast.walk(fn):
            if not isinstance(node, ast.Assert):
                continue
            described = _pooled_reduction_problem(node, local, tainted_funcs)
            if described:
                problems.append(
                    f"{path.name}:{node.lineno}: this assert {described}, which is a POOLED error "
                    f"-- one badly-wrong element inside a large correct tensor cannot move it. "
                    f"Judge per element with fold_cp_ops.testing.numerics.assert_elementwise, or "
                    f'-- if the pooled form IS this test\'s subject -- @numeric_exempt("why").'
                )

    for fn in [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]:
        for dec in fn.decorator_list:
            if not (isinstance(dec, ast.Call) and ast.unparse(dec.func).endswith("numeric_exempt")):
                continue
            arg = dec.args[0] if dec.args else None
            if not (
                isinstance(arg, ast.Constant) and isinstance(arg.value, str) and arg.value.strip()
            ):
                problems.append(
                    f"{path.name}::{fn.name} is @numeric_exempt but its reason is not a non-empty "
                    f"string literal; it is read from the AST, so it must be legible in the source."
                )
    return problems


def numeric_scope(path) -> Set[str]:
    """Which test functions in a module must record a sanctioned assertion.

    A test is in scope when it draws cases from a ``KernelMatrix`` -- i.e. it launches a kernel over
    shapes, which is the only thing that produces a value worth comparing -- and is not
    :func:`numeric_exempt`. Structural tests are out of scope by construction, so this costs no
    decoration on the tests that legitimately compare nothing.

    **``tests/perf/`` is excluded wholesale**, and that is a rule about the subject rather than a
    convenience. A perf gate's subject is a kernel's measured SPEED: it imports the same matrix and
    parametrizes from it, so it looks in-scope by every syntactic test, but the thing it asserts is
    a median against a pin. Measured before this exclusion existed: 21 perf tests across 9 files
    would have been failed for not comparing an output they never produce.

    Args:
        path: Path to the test module. Parsed, not imported, so this is usable at collection time
            before a module's fixtures exist. Its PATH PARTS are read, so this must be the module's
            real location rather than a temporary copy at a different depth.

    Returns:
        The set of function names. Empty for a module with no matrix-parametrized tests, which is
        what the module-level default-deny rule then reacts to, and empty for anything under a
        ``perf/`` directory at any depth (:data:`NUMERIC_EXCLUDED_DIRS`).

    Raises:
        SyntaxError: If the module does not parse.
    """
    path = pathlib.Path(path)
    if NUMERIC_EXCLUDED_DIRS & set(path.parts):
        return set()
    tree = ast.parse(path.read_text())
    scope: Set[str] = set()
    for fn in [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]:
        if not fn.name.startswith("test_"):
            continue
        decs = _decorator_sources(fn)
        if _is_matrix_parametrized(decs) and not _has_exempt(decs):
            scope.add(fn.name)
    return scope


def audit_numeric_module(path, module: Any = None) -> List[str]:
    """Every static check this guard makes about one test module, in one call.

    The single entry point ``tests/conftest.py`` and ``tests/testing/test_numeric_guard.py`` share,
    so the collection hook and the meta-test can never disagree about the rules.

    Args:
        path: Path to the test module. Its path PARTS decide whether the module-level default-deny
            rule applies, so it must be the module's real location and at its real depth.
        module: The imported module object, or None. Only used for the module-level exemption, and
            the exemption is read from the AST as well, so passing None loses nothing today; the
            parameter exists so the signature matches ``kernel_matrix.audit_test_module`` and a
            future check that needs the object has somewhere to go.

    Returns:
        A list of human-readable problems, empty when the module conforms.
    """
    path = pathlib.Path(path)
    if module_exemption(path):
        # The module said why the static pass does not apply to it. The COVERAGE gate is not
        # waived by this -- `numeric_scope` is computed independently -- so an exempt kernel module
        # still has to compare its output with a sanctioned assertion. What the exemption buys is
        # the right to SPELL a forbidden form, which exactly one module in this repo legitimately
        # needs: the one that proves the tripwire fires.
        return []
    problems = forbidden_comparison_problems(path)
    parts = set(path.parts)
    if (NUMERIC_DIRS & parts) and not (NUMERIC_EXCLUDED_DIRS & parts) and not numeric_scope(path):
        # Name the module's real directory, not just the matched component: "sits under workflows/"
        # is ambiguous the moment there are two of them, and the whole point of the parts rule is
        # that there now are.
        pp = path.parts
        i = pp.index("tests") if "tests" in pp else max(0, len(pp) - 2)
        problems.append(
            f"{path.name} sits under {'/'.join(pp[i:-1])}/ but has "
            f"no matrix-parametrized test "
            f"for the numeric coverage gate to check, and declares no "
            f'{MODULE_EXEMPT_ATTR} = "<why>" at module level. Adding a file must not be a way '
            f"to opt out of element-wise comparison."
        )
    return problems


def module_exemption(path) -> Optional[str]:
    """The module-level exemption reason a test module declares, if any.

    Read from the AST rather than by importing, so it is available at collection time and so an
    exemption computed at runtime (``NUMERIC_EXEMPT = _pick_reason()``) reads as no exemption at
    all -- the same rule ``matrix_exempt``'s reason follows, and for the same reason: a reason that
    a reviewer cannot read in the source is not one.

    Args:
        path: Path to the test module. Parsed, not imported.

    Returns:
        The declared reason, or None when the module declares none (or declares a non-literal,
        empty, or whitespace-only one, all of which are treated as absent).

    Raises:
        SyntaxError: If the module does not parse.
    """
    tree = ast.parse(pathlib.Path(path).read_text())
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if not any(isinstance(t, ast.Name) and t.id == MODULE_EXEMPT_ATTR for t in node.targets):
            continue
        if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            if node.value.value.strip():
                return node.value.value
    return None


def install_tripwire() -> None:
    """Make ``torch.allclose`` and ``torch.testing.assert_close`` raise for the whole session.

    The runtime half of the guard. It catches what the AST pass cannot: a call built through
    ``getattr(torch, "all" + "close")``, a call inside a shared helper module the pass did not scan,
    and a call reached through an alias bound before the pass ran. Poisoning the ATTRIBUTE rather
    than the name is what makes the dynamic form fail too.

    Idempotent: installing twice keeps the first poison, so a second call cannot accidentally
    capture the poisoned function as "the original".

    Returns:
        None; mutates ``torch`` and ``torch.testing`` in place. Deliberately never restored -- the
        session is a test session, and :func:`tripwire_problem` treats restoration as a violation.
    """
    import torch

    for owner, attr in ((torch, "allclose"), (torch.testing, "assert_close")):
        key = f"{owner.__name__}.{attr}"
        if key in _POISONED:
            continue

        def _poisoned(*_args, _key=key, **_kwargs):
            raise AssertionError(
                f"{_key} is not a sanctioned comparison in this suite. Judge per element with "
                f"fold_cp_ops.testing.numerics.assert_elementwise(actual, reference, bound); for a "
                f"plain atol/rtol pair use numerics.tolerance_bound(reference, atol, rtol). See "
                f"fold_cp_ops/testing/numeric_guard.py for why."
            )

        _POISONED[key] = (owner, attr, _poisoned)
        setattr(owner, attr, _poisoned)


def tripwire_problem() -> Optional[str]:
    """Report whether the tripwire is still armed, naming what put it back if not.

    **This is the anti-evasion check.** ``monkeypatch.setattr(torch, "allclose", orig)`` inside a
    test would silently disarm the guard for that test and, without an explicit restore, for every
    test after it. Checking at each teardown attributes the disarming to the test that did it,
    rather than leaving a suite that quietly stopped enforcing.

    Returns:
        None when every poisoned attribute is still the poisoned callable, otherwise a message
        naming the attributes that were restored. Returns None when the tripwire was never
        installed, so a caller that opts out (a session with no numeric tests) is not failed for it.
    """
    restored = [
        key
        for key, (owner, attr, poison) in _POISONED.items()
        if getattr(owner, attr, None) is not poison
    ]
    if not restored:
        return None
    return (
        f"the numeric tripwire was disarmed during this test: {', '.join(sorted(restored))} is no "
        f"longer the guard's callable. Restoring a forbidden comparison turns the guard off for "
        f"every test after this one; if a test genuinely needs the original, mark it "
        f"@numeric_exempt and say why."
    )
