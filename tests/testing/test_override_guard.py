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
"""Tests for ``fold_cp_ops.testing.override_guard``.

The guard's own subject is a defect class with no local cause -- a call in one file that cannot
satisfy an override in another -- so its tests have to prove BOTH directions on synthetic sources:
it must FIRE on a stale override and stay SILENT on the adaptations that are correct. A guard proven
in one direction only is the shape that let the real defect through: the earlier audits all passed
while ``epi_setup_postact`` was failing 83 blocks.
"""

import textwrap

import pytest

from fold_cp_ops.testing.kernel_matrix import matrix_exempt
from fold_cp_ops.testing.numeric_guard import numeric_exempt
from fold_cp_ops.testing.override_guard import (
    _SIGNATURE_PRESERVING_DECORATORS,
    duplicate_class_names,
    module_function_call_problems,
    override_acceptance_problems,
    override_arity_problems,
    reachable_override_pairs,
    resolved_module_calls,
    scanned_module_count,
    stale_override_params,
)


def _tree(tmp_path, source: str):
    """Write one synthetic module into a scannable root.

    Args:
        tmp_path: pytest's per-test directory.
        source: Module source. Dedented, so tests can use triple-quoted blocks at their own indent.

    Returns:
        The root path to hand to the guard.
    """
    root = tmp_path / "pkg"
    root.mkdir(parents=True, exist_ok=True)
    (root / "m.py").write_text(textwrap.dedent(source))
    return root


@matrix_exempt(
    "scans this package's SOURCE for a call/override arity mismatch; the result is a property of "
    "the files on disk and cannot vary with a kernel's shape axes"
)
@numeric_exempt("parses ASTs and compares parameter names; launches nothing and computes no tensor")
def test_the_package_has_no_unsatisfiable_override_call():
    """No ``self.X(...)`` in this package can fail to fill a reachable override's parameters.

    The live assertion. Two real violations motivated it -- ``_gA_local_tile`` keeping the ported
    3-parameter shape, and ``epi_setup_postact`` keeping ``varlen_manager`` -- and each cost a
    multi-rank gate run to find, while this check finds them at import.

    The scan count is asserted first and separately: a walk that reads zero files reports zero
    problems, and that is indistinguishable from a clean tree unless the count is checked.
    """
    n = scanned_module_count()
    assert n > 50, f"the package scan read {n} modules; a near-empty scan makes this vacuous"
    problems = override_arity_problems()
    assert not problems, "unsatisfiable override call(s):\n  " + "\n  ".join(problems)


@matrix_exempt(
    "asserts class-name uniqueness in this package; no kernel and no shape axis involved"
)
@numeric_exempt("asserts a name mapping; computes no value")
def test_top_level_class_names_are_unique_enough_to_resolve_ancestry():
    """Duplicated top-level class names make the guard's name-based ancestry ambiguous.

    Not a correctness failure by itself, which is why the guard reports it separately rather than
    folding it into a verdict. It is pinned here because the guard resolves receivers BY NAME: a
    duplicate silently makes "which class is this" unanswerable, and a verdict computed against the
    wrong one reads exactly like a verdict computed against the right one.

    Known and accepted: ``_AlreadyOrderedBarrier`` is declared in two kernel modules. The assertion
    pins the SET so a new duplicate is a visible edit rather than a silent widening -- the same
    bargain the numeric guard's exempt-module list strikes.
    """
    assert set(duplicate_class_names()) <= {"_AlreadyOrderedBarrier"}, (
        f"new duplicate top-level class name(s): {duplicate_class_names()}. The override guard "
        "resolves ancestry by name, so a duplicate makes its verdict about that class unreliable."
    )


@matrix_exempt("scans package SOURCE for override/parent signature divergence; no shape axis")
@numeric_exempt("compares parameter names; launches nothing and computes no tensor")
def test_the_set_of_overrides_requiring_more_than_their_parent_is_pinned():
    """The second check ships as a PINNED REPORT, and that is a deliberate downgrade.

    ``stale_override_params`` asks whether an override requires a parameter its parent does not
    accept. That is the signature of the real defect (``epi_setup_postact`` keeping
    ``varlen_manager``), but it is **not** sufficient for it: a specialized variant that owns its
    own call path may legitimately require more. Measured -- all four rows below are correct, and
    ``override_arity_problems`` reports NONE against the same tree, which is the evidence that no
    caller is broken by any of them.

    So this asserts the SET rather than emptiness. An assertion of emptiness would fail on correct
    code today, and a check that fires on correct code is removed rather than repaired -- the
    failure mode that let the original defect survive three audits. Pinning means a NEW divergence
    is a visible edit with a reason attached, the same bargain the numeric guard's exempt-module
    list and the wide-seam allowlist both strike.
    """
    got = {p.split(" ")[0] for p in stale_override_params()}
    assert got == {
        "SmemColVecBroadcast.__init__",
        "LayerNormTransposeSm90.__init__",
        "LayerNormTransposeSm90.__call__",
        "LayerNormTransposeSm90.kernel",
    }, (
        f"the set of overrides requiring more than their parent changed: {sorted(got)}. Each is "
        "safe ONLY while the subclass is reached through its own call path -- check "
        "override_arity_problems() for the same tree before accepting a new entry."
    )


def test_stale_override_params_fires_on_a_required_extra_and_not_a_defaulted_one(tmp_path):
    """Both directions for the report: a REQUIRED extra is named, a DEFAULTED one is not.

    The distinction is the whole precision of this check. Widening a hook with a defaulted
    parameter is how the A2A token-grid seam was landed and must stay silent; keeping a required
    parameter the parent dropped is the defect and must be named.
    """
    required = _tree(
        tmp_path / "a",
        """
        class Base:
            def hook(self, x):
                return None

        class Sub(Base):
            def hook(self, x, stale):
                return None
        """,
    )
    problems = stale_override_params(required)
    assert len(problems) == 1 and "['stale']" in problems[0], problems

    defaulted = _tree(
        tmp_path / "b",
        """
        class Base:
            def hook(self, x):
                return None

        class Sub(Base):
            def hook(self, x, widened=None):
                return None
        """,
    )
    assert stale_override_params(defaulted) == [], (
        "a DEFAULTED extra needs no caller change and must not be reported; flagging it would "
        "fire on the very widening this guard's own package uses"
    )


def test_it_fires_when_an_override_keeps_a_parameter_the_caller_does_not_pass(tmp_path):
    """POSITIVE control, in the exact shape of the real defect.

    ``Sub`` keeps ``varlen_manager`` in the middle of its signature; the caller passes four
    positionals and ``epi_gate3=`` as a keyword. The four fill through ``varlen_manager``, the
    keyword fills the LAST parameter, and ``tidx`` is left empty -- the parameter the runtime error
    names, three positions from the one at fault.

    **A count would call this satisfied**: four positional plus one keyword is five supplied against
    five required. Only tracking names sees the hole, which is why this control asserts on the
    reported NAME rather than on the fact that something was reported.
    """
    root = _tree(
        tmp_path,
        """
        class Base:
            def epi(self, params, tiled_copy, tile_coord, tidx, epi_gate3=False):
                return None

            def caller(self):
                return self.epi(1, 2, 3, 4, epi_gate3=True)

        class Sub(Base):
            def epi(self, params, tiled_copy, tile_coord, varlen_manager, tidx, epi_gate3=False):
                return None
        """,
    )
    problems = override_arity_problems(root)
    assert len(problems) == 1, problems
    assert "['tidx']" in problems[0], f"must name the UNFILLED parameter, got: {problems[0]}"
    assert "Sub.epi" in problems[0], "must name the receiving override"


def test_it_is_silent_when_the_override_dropped_the_parameter_correctly(tmp_path):
    """NEGATIVE control: the adaptation that three of the four real overrides made.

    Same shape as above with ``varlen_manager`` removed from the subclass -- which is what
    ``build_D_copy_fn``, ``get_scheduler_arguments`` and ``epi_setup_postact`` (after the fix) all
    do. A guard that flagged this would fire on correct code, and a guard that fires on correct code
    gets deleted rather than fixed.
    """
    root = _tree(
        tmp_path,
        """
        class Base:
            def epi(self, params, tiled_copy, tile_coord, tidx, epi_gate3=False):
                return None

            def caller(self):
                return self.epi(1, 2, 3, 4, epi_gate3=True)

        class Sub(Base):
            def epi(self, params, tiled_copy, tile_coord, tidx, epi_gate3=False):
                return None
        """,
    )
    assert override_arity_problems(root) == []


def test_it_catches_a_mismatch_across_the_inheritance_GRAPH_not_just_a_direct_pair(tmp_path):
    """The receiver set is the DESCENDANTS, so a grandchild's override is still checked.

    This is what lets the guard catch a call written in ``_internal/`` that breaks on an override in
    ``distributed/`` -- the two files never mention each other, and the classes are three links
    apart. A direct base/subclass check would miss it and report the tree clean.
    """
    root = _tree(
        tmp_path,
        """
        class A:
            def hook(self, x, y):
                return None

            def caller(self):
                return self.hook(1, 2)

        class B(A):
            pass

        class C(B):
            def hook(self, x, y, extra):
                return None
        """,
    )
    problems = override_arity_problems(root)
    assert len(problems) == 1 and "['extra']" in problems[0], problems
    assert "C.hook" in problems[0]


def test_self_versus_cls_is_not_reported_as_a_parameter_difference(tmp_path):
    """A classmethod overridden by an instance method differs in a binding name, not a parameter.

    Without this filter the guard produced two false rows on ``_compute_stages``. Measured, not
    anticipated: the filter exists because the rows appeared.
    """
    root = _tree(
        tmp_path,
        """
        class Base:
            @classmethod
            def build(cls, a, b):
                return None

            def caller(self):
                return self.build(1, 2)

        class Sub(Base):
            def build(self, a, b):
                return None
        """,
    )
    assert override_arity_problems(root) == []


@pytest.mark.parametrize("spelling", ["*args", "**kwargs"])
def test_a_dynamic_argument_list_is_skipped_rather_than_guessed(tmp_path, spelling):
    """A call built with ``*args``/``**kwargs`` is unresolvable statically, so it is not reported.

    Losing a finding is the safe direction here and inventing one is not: a false positive on a
    correct kernel is what makes a guard get deleted. Both spellings are swept because they take
    different branches at the call site.
    """
    root = _tree(
        tmp_path,
        f"""
        class Base:
            def hook(self, x, y):
                return None

            def caller(self, *args, **kwargs):
                return self.hook({spelling})

        class Sub(Base):
            def hook(self, x, y, extra):
                return None
        """,
    )
    assert override_arity_problems(root) == []


def test_an_empty_scan_reports_no_problems_which_is_why_the_count_is_asserted(tmp_path):
    """The guard's own vacuous case, pinned so nobody reads a clean result as evidence.

    ``override_arity_problems`` on a directory with no sources returns ``[]`` -- identical to a
    fully-scanned clean tree. Three separate zeros in this project came from a walk that read
    nothing, which is why :func:`scanned_module_count` exists and why the live test asserts it
    before asserting the verdict.
    """
    empty = tmp_path / "empty"
    empty.mkdir()
    assert override_arity_problems(empty) == []
    assert scanned_module_count(empty) == 0, (
        "the witness must report zero here; if it cannot distinguish an empty scan, the live "
        "assertion above it is unfalsifiable"
    )


def _pkg(tmp_path, files: dict, name: str = "pkg"):
    """Write a multi-module synthetic package into a scannable root.

    The module-level check resolves imports, so its subjects need at least two files -- a definition
    module and a calling module -- which the single-file :func:`_tree` cannot express.

    Args:
        tmp_path: pytest's per-test directory.
        files: ``{relative_path: source}``. Each source is dedented, so tests can write triple-quoted
            blocks at their own indent. Parent directories are created as needed.
        name: The root directory name, which BECOMES the package name the guard resolves imports
            against -- a source saying ``import pkg.helpers`` needs this to be ``"pkg"``.

    Returns:
        The root path to hand to the guard.
    """
    root = tmp_path / name
    for rel, source in files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(textwrap.dedent(source))
    return root


#: The definition the extraction re-signatured: ``main``'s leading ``arch`` is gone because this
#: package is SM90-only by decision. Shared by the fires/silent pair so both run against the SAME
#: callee and the only difference between them is the call site.
_RESIGNATURED_HELPER = """
    def sm90_get_smem_store_atom(element_type, transpose=False, major_mode_size=None, *,
                                 loc=None, ip=None):
        return (element_type, transpose, major_mode_size)
    """


@matrix_exempt(
    "scans this package's SOURCE for a call that cannot bind to a module-level definition; the "
    "result is a property of the files on disk and cannot vary with a kernel's shape axes"
)
@numeric_exempt("parses ASTs and compares parameter names; launches nothing and computes no tensor")
def test_the_package_has_no_unbindable_module_function_call():
    """No call to a module-level function in this package can fail to bind to its definition.

    The third live assertion, and the one the other two are structurally unable to make: they reason
    about methods, and a module-level function has no override and no parent.

    Both witnesses are asserted before the verdict. ``scanned_module_count`` proves files were read;
    ``resolved_module_calls`` proves calls were CHECKED, which is the stronger of the two here --
    most calls in this package go to methods, third-party functions or dynamic targets and are
    skipped by design, so a resolver that quietly resolved nothing would report a clean tree.
    """
    n = scanned_module_count()
    assert n > 50, f"the package scan read {n} modules; a near-empty scan makes this vacuous"
    checked = resolved_module_calls()
    assert checked > 100, (
        f"only {checked} call sites resolved to a checkable module-level definition; the verdict "
        "below is near-vacuous if the resolver stopped resolving"
    )
    problems = module_function_call_problems()
    assert not problems, "call(s) that cannot bind to their definition:\n  " + "\n  ".join(problems)


@pytest.mark.parametrize("where", ["module-level", "function-local"])
def test_it_fires_on_the_dropped_leading_argument_and_names_the_collision(tmp_path, where):
    """The known-failing control: the real defect, reproduced, must be reported by NAME.

    This is the defect the check was written for -- a call site that still passes ``main``'s leading
    ``arch`` to a helper this tree re-signatured without it. The extra positional lands on
    ``transpose`` and collides with the keyword that names it. **A count check passes this in both
    states**, which is why the assertion demands the parameter name in the message.

    The import placement is swept because the live instance hides behind a FUNCTION-LOCAL import
    (``dual_gated_gemm_a2a.py:1697``). A resolver that walks ``tree.body`` instead of the whole tree
    resolves the module-level spelling and silently misses the one that actually occurred.
    """
    call = """
        class D:
            def epi(self, params):
                {local}
                return helpers.sm90_get_smem_store_atom(
                    self.arch,
                    self.postact_dtype,
                    transpose=True,
                    major_mode_size=params.n,
                )
        """
    top = "" if where == "function-local" else "import pkg.helpers as helpers\n"
    local = "import pkg.helpers as helpers" if where == "function-local" else ""
    root = _pkg(
        tmp_path,
        {
            "helpers.py": _RESIGNATURED_HELPER,
            "caller.py": top + textwrap.dedent(call).format(local=local),
        },
    )
    problems = module_function_call_problems(root)
    assert len(problems) == 1, f"expected exactly one finding, got {problems}"
    assert "transpose" in problems[0], (
        f"the finding must NAME the colliding parameter, not just count arguments: {problems[0]}"
    )


def test_it_is_silent_on_the_repaired_call_which_was_still_resolved(tmp_path):
    """The other direction -- and the silence is only evidence because the call still RESOLVED.

    Same callee as the test above, one argument removed. A guard proven in the firing direction only
    is the shape that lets a defect through, and a guard whose silence comes from resolving nothing
    is worse still: it passes on every tree, including a broken one. So this asserts the resolution
    count alongside the empty verdict.
    """
    root = _pkg(
        tmp_path,
        {
            "helpers.py": _RESIGNATURED_HELPER,
            "caller.py": """
                import pkg.helpers as helpers

                class D:
                    def epi(self, params):
                        return helpers.sm90_get_smem_store_atom(
                            self.postact_dtype,
                            transpose=True,
                            major_mode_size=params.n,
                        )
                """,
        },
    )
    assert resolved_module_calls(root) == 1, (
        "the repaired call must still be RESOLVED; a silence produced by failing to resolve it "
        "would pass on the broken form too"
    )
    assert module_function_call_problems(root) == []


@pytest.mark.parametrize(
    "call,expected",
    [
        ("helpers.f(1)", "'b'"),
        ("helpers.f(1, 2, 3)", "3 positional"),
        ("helpers.f(1, 2, zz=3)", "'zz'"),
    ],
    ids=["required-unfilled", "too-many-positional", "unknown-keyword"],
)
def test_every_binding_direction_is_reported_and_each_names_the_parameter(tmp_path, call, expected):
    """All three ways an argument list fails to bind, each naming the parameter at fault.

    ``override_arity_problems`` needs only the unfilled direction, because a stale override is always
    LONGER than its caller expects. A renamed free function can be SHORTER, so the extra positional
    has somewhere to land -- which is the collision and the too-many cases. Enumerated here so a
    future simplification cannot quietly drop the two directions the method check never needed.

    Args:
        call: The call-site expression under test.
        expected: A substring the finding must contain, always a parameter NAME or an argument
            count, never a bare "does not bind".
    """
    root = _pkg(
        tmp_path,
        {
            "helpers.py": """
                def f(a, b):
                    return a + b
                """,
            "caller.py": f"""
                import pkg.helpers as helpers

                def go():
                    return {call}
                """,
        },
    )
    problems = module_function_call_problems(root)
    assert len(problems) == 1, f"expected one finding for {call!r}, got {problems}"
    assert expected in problems[0], f"finding must name {expected}: {problems[0]}"


@pytest.mark.parametrize("decorator", ["jit_cache", "autotune", "some.unknown_wrapper"])
def test_a_callee_behind_an_unrecognized_decorator_is_skipped_rather_than_guessed(
    tmp_path, decorator
):
    """A wrapper may rebuild the argument list, so its callee is skipped, not assumed.

    ``jit_cache`` and ``autotune`` are the two live cases in this package (13 and 8 call sites at the
    time of writing). Reporting through them would fire on correct code, and a guard that fires on
    correct code is deleted rather than repaired -- the failure mode that let the original defect
    survive three audits.
    """
    root = _pkg(
        tmp_path,
        {
            "helpers.py": f"""
                @{decorator}
                def f(a, b):
                    return a + b
                """,
            "caller.py": """
                import pkg.helpers as helpers

                def go():
                    return helpers.f(1)
                """,
        },
    )
    assert module_function_call_problems(root) == []
    assert resolved_module_calls(root) == 0, "an unchecked callee must not be counted as checked"


@matrix_exempt("pins a module constant; no kernel and no shape axis involved")
@numeric_exempt("asserts a set of decorator names; computes no value")
def test_the_signature_preserving_decorator_set_is_pinned():
    """The allowlist is pinned, because widening it silently reduces what the guard checks.

    Every name added here declares "this wrapper does not change the signature a caller must
    satisfy". That is a claim about a third-party decorator, not something the guard can verify, so
    it must be an explicit edit rather than a convenience. An un-pinned allowlist degrades the same
    way an un-pinned exemption list does: each entry checks one thing less, and the end state passes
    identically to a clean one.
    """
    assert _SIGNATURE_PRESERVING_DECORATORS == frozenset(
        {
            "cute.jit",
            "cute.kernel",
            "dsl_user_op",
            "lru_cache",
            "functools.lru_cache",
            "contextlib.contextmanager",
            "torch.library.custom_op",
            "staticmethod",
        }
    ), (
        "the signature-preserving decorator allowlist changed. Each entry asserts that the wrapper "
        "does not rebuild its callee's argument list; verify that before widening it."
    )


def test_a_call_through_a_rebound_name_is_skipped(tmp_path):
    """A name assigned anywhere in the file cannot be attributed to its import with confidence.

    Deliberately coarse -- the check does not model scopes -- because over-skipping loses a finding
    while under-skipping invents one, and only the second gets a guard removed.
    """
    root = _pkg(
        tmp_path,
        {
            "helpers.py": """
                def f(a, b):
                    return a + b
                """,
            "caller.py": """
                import pkg.helpers as helpers

                def go(other):
                    helpers = other
                    return helpers.f(1)
                """,
        },
    )
    assert module_function_call_problems(root) == []


@pytest.mark.parametrize(
    "callee,call",
    [
        ("def f(a, *rest):\n    return a", "helpers.f()"),
        ("def f(a, b):\n    return a", "helpers.f(*xs)"),
    ],
    ids=["callee-takes-star-args", "call-spreads-star-args"],
)
def test_a_dynamic_argument_list_is_skipped_on_the_module_path_too(tmp_path, callee, call):
    """``*args`` on either side makes the positional count meaningless, so no verdict is issued.

    The method check already refuses both spellings; this pins that the module-level path does the
    same, since the two resolve their callees through completely different code.
    """
    root = _pkg(
        tmp_path,
        {
            "helpers.py": callee,
            "caller.py": f"""
                import pkg.helpers as helpers

                def go(xs):
                    return {call}
                """,
        },
    )
    assert module_function_call_problems(root) == []


def test_an_empty_scan_resolves_nothing_which_is_why_the_count_is_asserted(tmp_path):
    """The new check's own vacuous case, pinned next to the one it mirrors.

    ``module_function_call_problems`` on an empty directory returns ``[]``, identical to a clean
    tree. ``resolved_module_calls`` is what tells those apart, and the live assertion checks it
    before believing the verdict.
    """
    empty = tmp_path / "empty_mod"
    empty.mkdir()
    assert module_function_call_problems(empty) == []
    assert resolved_module_calls(empty) == 0, (
        "the witness must report zero here; if it cannot distinguish an empty scan, the live "
        "assertion is unfalsifiable"
    )


@matrix_exempt(
    "scans this package's SOURCE for a call a reachable override cannot accept; the result is a "
    "property of the files on disk and cannot vary with a kernel's shape axes"
)
@numeric_exempt("parses ASTs and compares parameter names; launches nothing and computes no tensor")
def test_the_package_has_no_call_a_reachable_override_cannot_accept():
    """No ``self.X(...)`` passes an argument some reachable override of ``X`` would refuse.

    The fourth live assertion, and the one that fires when a seam is WIDENED rather than when an
    override goes stale. Its motivating defect: ``mainloop_remap_mA`` grew ``mA_mkl=None,
    batch_idx=None`` on the parent and at the call site while ``GemmSm90A2A``'s two-parameter
    override was left alone, and 203 blocks raised ``got an unexpected keyword argument 'mA_mkl'``
    with every other check in this module reporting zero.

    The pair count is asserted first for the usual reason, and it is the load-bearing witness here:
    the receiver set is NARROWED (see ``_reachable_receivers``), so a narrowing bug would empty the
    walk and produce a clean verdict from nothing.
    """
    pairs = reachable_override_pairs()
    assert pairs > 100, (
        f"only {pairs} call/override pairs were examined; the receiver narrowing may have emptied "
        "the walk, which would make the verdict below vacuous"
    )
    problems = override_acceptance_problems()
    assert not problems, "call(s) a reachable override cannot accept:\n  " + "\n  ".join(problems)


def _widened_seam(tmp_path, caller_override: str):
    """A parent seam widened with a defaulted parameter, and a subclass override left behind.

    The shape of the live regression, reduced: the base declares ``hook(self, extra=None)`` and
    passes ``extra=``; the subclass still declares ``hook(self)``.

    Args:
        tmp_path: pytest's per-test directory.
        caller_override: Extra source for ``Sub``, used to vary whether ``Sub`` overrides the
            CALLING method -- which is what decides whether the base's call site is reachable at all.

    Returns:
        The root path to hand to the guard.
    """
    return _tree(
        tmp_path,
        f"""
        class Base:
            def caller(self):
                return self.hook(mA_mkl=1)

            def hook(self, mA_mkl=None):
                return None

        class Sub(Base):
            def hook(self):
                return None
        {caller_override}
        """,
    )


def test_it_fires_when_a_widened_seam_leaves_an_override_behind(tmp_path):
    """The known-failing control: the Option C regression, reduced, must be named by parameter.

    A defaulted addition on the PARENT does not make an OVERRIDE accept it. That sentence is the
    whole defect, and it was asserted to be safe on the strength of a check that asked a different
    question -- whether ``super()`` calls bind.
    """
    problems = override_acceptance_problems(_widened_seam(tmp_path, ""))
    assert len(problems) == 1, f"expected exactly one finding, got {problems}"
    assert "mA_mkl" in problems[0], (
        f"the finding must NAME the keyword the override refuses: {problems[0]}"
    )


def test_the_arity_check_is_silent_on_the_widening_the_acceptance_check_reports(tmp_path):
    """The two directions are not redundant, demonstrated on one tree rather than argued.

    ``override_arity_problems`` asks whether the call passes too FEW; a widened seam passes too
    MANY, so it has nothing to report and correctly says so. Keeping this as a test means a future
    "simplification" that folds one check into the other has to fail here first.
    """
    root = _widened_seam(tmp_path, "")
    assert override_arity_problems(root) == [], (
        "the too-few check must stay silent here; if it starts firing, the two checks have been "
        "conflated and this file no longer proves they are independent"
    )
    assert len(override_acceptance_problems(root)) == 1


@pytest.mark.parametrize(
    "caller_override,fires",
    [
        ("", True),
        ("\n            def caller(self):\n                return None", False),
        ("\n            def caller(self):\n                return super().caller()", True),
    ],
    ids=["subclass-does-not-override-caller", "overrides-caller", "overrides-caller-with-super"],
)
def test_a_receiver_that_never_runs_the_calling_body_is_not_a_reachable_receiver(
    tmp_path, caller_override, fires
):
    """The narrowing that makes this check an assertion instead of a pinned report.

    A call site in ``Base.caller`` cannot dispatch anywhere for an instance of ``Sub`` when ``Sub``
    overrides ``caller`` and does not delegate -- the base body never runs. Measured on the real
    package: without this, the check reports exactly one row, and that row is a false positive of
    the kind ``stale_override_params`` already documents.

    The ``super()`` case is the half that keeps the narrowing honest. Delegating brings the base
    body back into play, so the finding must return -- a narrowing that fired on neither would be
    indistinguishable from one that had simply stopped working.

    Args:
        caller_override: Extra source appended to ``Sub``.
        fires: Whether the base's call site is reachable on ``Sub``.
    """
    problems = override_acceptance_problems(_widened_seam(tmp_path, caller_override))
    assert bool(problems) is fires, f"expected fires={fires}, got {problems}"


def test_the_too_many_positional_direction_is_reported_with_both_counts(tmp_path):
    """The other half of "cannot accept": more positionals than the override declares.

    Reported alongside the keyword case because a widened seam can be threaded either way, and the
    message carries both counts so the reader does not have to open two files to see the gap.
    """
    root = _tree(
        tmp_path,
        """
        class Base:
            def caller(self):
                return self.hook(1, 2, 3)

            def hook(self, a, b=None, c=None):
                return None

        class Sub(Base):
            def hook(self, a):
                return None
        """,
    )
    problems = override_acceptance_problems(root)
    assert len(problems) == 1, f"expected one finding, got {problems}"
    assert "3 positional" in problems[0] and "1 ['a']" in problems[0], problems[0]


@pytest.mark.parametrize(
    "callee,call",
    [
        ("def hook(self, **kwargs):\n                return None", "self.hook(extra=3)"),
        ("def hook(self, *args):\n                return None", "self.hook(1, 2)"),
        (
            "def hook(self, *args, **kwargs):\n                return None",
            "self.hook(1, 2, extra=3)",
        ),
    ],
    ids=["kwargs-absorbs-the-keyword", "star-args-absorbs-the-positionals", "both"],
)
def test_an_override_that_absorbs_the_argument_kind_passed_is_not_reported(tmp_path, callee, call):
    """``**kwargs`` absorbs any KEYWORD and ``*args`` any POSITIONAL count -- each, not both.

    Reporting through a genuine absorber would fire on correct code, which is how a guard gets
    deleted rather than repaired. Each absorber is paired with the argument kind it actually takes,
    because the first version of this test paired them with a call passing BOTH and the check was
    right to reject it: ``hook(1, 2, extra=3)`` against ``def hook(self, **kwargs)`` is a real
    ``TypeError``. The test was wrong, not the guard -- worth keeping in the record, since a
    "false positive" that turns out to be a true one is the cheapest way to weaken a check.

    Args:
        callee: The subclass override's source.
        call: The call-site expression, passing only what that override can absorb.
    """
    root = _tree(
        tmp_path,
        f"""
        class Base:
            def caller(self):
                return {call}

            def hook(self, a=None, b=None, extra=None):
                return None

        class Sub(Base):
            {callee}
        """,
    )
    assert override_acceptance_problems(root) == []


def test_an_empty_scan_examines_no_pairs_which_is_why_the_count_is_asserted(tmp_path):
    """The acceptance check's own vacuous case, pinned beside the two it mirrors."""
    empty = tmp_path / "empty_acc"
    empty.mkdir()
    assert override_acceptance_problems(empty) == []
    assert reachable_override_pairs(empty) == 0, (
        "the witness must report zero here; if it cannot distinguish an empty scan, the live "
        "assertion is unfalsifiable"
    )
