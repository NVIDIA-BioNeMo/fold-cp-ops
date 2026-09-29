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

"""Unit tests for ``fold_cp_ops/testing/numeric_guard.py`` -- and proof it discriminates.

**Every rejection test here is paired with an acceptance test on the nearest legal neighbour.** A
guard with no proof it discriminates is decoration: a `forbidden_comparison_problems` that returned
every module would satisfy any test that only checks "the bad module is flagged", and would then
fail the whole suite the moment it landed. So each pair changes exactly the one thing the rule is
about.

GPU-free by construction -- the static half is AST over strings, and the runtime half is checked by
calling the poisoned attribute, which raises before it would touch a device.

The last test is the repo-wide lock: every test module under ``tests/`` is clean *today*. Without
it the guard could be quietly weakened and nothing would notice, because a guard that flags nothing
looks exactly like a codebase with nothing to flag.
"""

import pathlib

import pytest
import torch

from fold_cp_ops.testing.numeric_guard import (
    MODULE_EXEMPT_ATTR,
    SANCTIONED_ASSERTIONS,
    assertions_recorded,
    audit_numeric_module,
    forbidden_comparison_problems,
    install_tripwire,
    module_exemption,
    numeric_exempt,
    numeric_scope,
    record_assertion,
    reset_assertions,
    tripwire_problem,
)

#: This module has to SPELL the forbidden forms -- a tripwire nobody triggers is a tripwire nobody
#: has tested. Exempting it from the static pass is the only way to write that proof, and the
#: exemption is itself locked below by :func:`test_the_exempt_modules_are_exactly_the_expected_ones`
#: so a second module cannot pick it up quietly.
NUMERIC_EXEMPT = "the guard's own tests must spell the forbidden comparisons to prove it fires"

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]


def _module(tmp_path, source, name="test_probe.py", subdir=None):
    """Write a throwaway test module and return its path.

    Args:
        tmp_path: pytest's per-test temp dir.
        source: The module source. Written verbatim, so indentation matters.
        name: Filename. Must start with ``test_`` for the audit's own filters to treat it as a test
            module.
        subdir: Directory under ``tmp_path`` to place it in, or None for ``tmp_path`` itself. The
            audit keys its module-level default-deny rule on the PATH PARTS, so a probe for that
            rule must pass ``"kernels"`` (or a nested ``"distributed/kernels"``) here; a probe for
            anything else must not, or it picks up an unrelated second problem.

    Returns:
        The path written.
    """
    parent = tmp_path if subdir is None else tmp_path / subdir
    parent.mkdir(parents=True, exist_ok=True)
    path = parent / name
    path.write_text(source)
    return path


# ── scope: WHERE the default-deny rule applies ─────────────────────────────────────────────────


@pytest.mark.parametrize(
    "subdir,denied",
    [
        ("kernels", True),
        ("workflows", True),
        # The rows with teeth. Neither directory existed when this rule was written; under the old
        # `parent.name` spelling the first was covered by COINCIDENCE (its leaf happens to be
        # "kernels") and the second was covered for the same accidental reason -- neither by
        # decision, and both would have evaporated on a rename.
        ("distributed/kernels", True),
        ("distributed/workflows", True),
        ("distributed/kernels/deeper", True),
        # A perf gate parametrizes from the same matrix but compares a median against a pin, so it
        # is excluded -- at any depth, which is what lets tests/distributed/perf/ exist without
        # this list being edited.
        ("perf", False),
        ("distributed/perf", False),
        # Not a governed directory at all.
        ("_internal", False),
        (None, False),
    ],
)
def test_the_default_deny_rule_follows_the_directory_at_any_depth(tmp_path, subdir, denied):
    """A matrix-less module is refused iff it sits under a governed directory, at any depth.

    The module written here declares no matrix-parametrized test and no ``NUMERIC_EXEMPT``, which
    is precisely what the default-deny rule reacts to. So the ONLY thing varying across rows is the
    directory -- if the rule were still keyed on the immediate parent, the ``distributed/*/deeper``
    row would flip and the ``distributed/perf`` row would flip the other way.
    """
    path = _module(tmp_path, "def test_x():\n    assert True\n", subdir=subdir)
    problems = audit_numeric_module(path)
    hit = [p for p in problems if "no matrix-parametrized test" in p]
    assert bool(hit) is denied, f"subdir={subdir!r} expected denied={denied}, got {problems}"


@pytest.mark.parametrize("subdir", ["perf", "distributed/perf", "distributed/perf/deeper"])
def test_a_perf_module_is_out_of_the_coverage_scope_at_any_depth(tmp_path, subdir):
    """``numeric_scope`` is empty under any ``perf/``, so the coverage gate never fires there.

    Measured before the exclusion existed: 21 perf tests across 9 files would have been failed for
    not comparing an output they never produce. The depth-independence is what keeps that true for
    ``tests/distributed/perf/``, which has to be a separate directory because its gates need a live
    process group.
    """
    path = _module(
        tmp_path, "@GEMM.parametrize('M', 'N')\ndef test_x(M, N):\n    pass\n", subdir=subdir
    )
    assert numeric_scope(path) == set()


# ── layer 1: the static pass ───────────────────────────────────────────────────────────────────


def test_a_qualified_torch_comparison_is_refused(tmp_path):
    """``torch.testing.assert_close`` is the spelling the converted call sites used to carry."""
    path = _module(
        tmp_path,
        "import torch\n\n\ndef test_x():\n    torch.testing.assert_close(a, b, atol=1, rtol=1)\n",
    )
    problems = forbidden_comparison_problems(path)
    assert len(problems) == 1 and "assert_close" in problems[0]
    assert "assert_elementwise" in problems[0], "the message must name the replacement"


def test_the_sanctioned_spelling_of_the_same_check_is_accepted(tmp_path):
    """The acceptance half of the pair above: same comparison, sanctioned helper, no problem.

    This is what proves the rule is about WHICH function, not about the presence of a comparison.
    """
    path = _module(
        tmp_path,
        "from fold_cp_ops.testing.numerics import assert_elementwise, tolerance_bound\n\n\n"
        "def test_x():\n    assert_elementwise(a, b, tolerance_bound(b, 1, 1))\n",
    )
    assert forbidden_comparison_problems(path) == []


def test_an_aliased_import_is_refused_at_the_import(tmp_path):
    """``from torch import allclose as ac`` would make the call-site check miss ``ac(...)``.

    So the import is the violation, not the call. Catching it here is what keeps the static pass
    from being defeated by a rename.
    """
    path = _module(
        tmp_path, "from torch import allclose as ac\n\n\ndef test_x():\n    assert ac(a, b)\n"
    )
    problems = forbidden_comparison_problems(path)
    assert len(problems) == 1 and "allclose" in problems[0]


def test_an_unrelated_import_from_the_same_module_is_accepted(tmp_path):
    """The acceptance half: importing from ``torch`` is not itself suspicious."""
    path = _module(tmp_path, "from torch import zeros as z\n\n\ndef test_x():\n    z(3)\n")
    assert forbidden_comparison_problems(path) == []


def test_a_pooled_reduction_over_a_difference_is_refused(tmp_path):
    """The original defect, in its original spelling: worst error over LARGEST reference.

    No blacklist of function names would have caught this one -- it is built from ``-``, ``.abs()``
    and ``.max()``, which are not forbidden anywhere else -- which is why the rule keys on the
    combination of a reduction and a subtraction.
    """
    path = _module(
        tmp_path,
        "def test_x():\n    assert (got - ref).abs().max() / ref.abs().max() <= 0.02\n",
    )
    problems = forbidden_comparison_problems(path)
    assert len(problems) == 1 and "POOLED" in problems[0]


def test_a_reduction_with_no_difference_under_it_is_accepted(tmp_path):
    """The acceptance half, and the reason ``tests/perf`` is not collateral damage.

    ``median(times)`` reduces a SAMPLE, not an error, and a perf harness is built out of exactly
    that. Requiring a subtraction beneath the reduction is what separates the two.
    """
    path = _module(tmp_path, "def test_x():\n    assert times.median() <= budget\n")
    assert forbidden_comparison_problems(path) == []


def test_a_bare_call_to_a_locally_defined_helper_is_accepted(tmp_path):
    """Three kernel test modules define their own element-wise ``assert_close``; they are fine.

    A bare call to a name the module DEFINES provably is not torch's: torch's can only arrive
    qualified (refused above), imported (refused above), or through ``getattr`` (caught by the
    tripwire, which poisons the attribute rather than the name). The local definition is audited by
    the same pass, so a shadow that pools is still caught -- by its own body, as the next test
    shows.
    """
    path = _module(
        tmp_path,
        "from fold_cp_ops.testing.numerics import assert_elementwise\n\n\n"
        "def assert_close(d, ref, bound):\n    assert_elementwise(d, ref, bound)\n\n\n"
        "def test_x():\n    assert_close(d, ref, bound)\n",
    )
    assert forbidden_comparison_problems(path) == []


def test_a_local_helper_that_actually_pools_is_still_refused(tmp_path):
    """The exemption above is for the NAME, not for the behaviour. This is what makes that safe."""
    path = _module(
        tmp_path,
        "def assert_close(d, ref, tol):\n    assert (d - ref).abs().max() <= tol\n\n\n"
        "def test_x():\n    assert_close(d, ref, 0.02)\n",
    )
    problems = forbidden_comparison_problems(path)
    assert len(problems) == 1 and "POOLED" in problems[0]


# ── the reach of the pooled detector: WHERE the reduction is allowed to sit ────────────────────
#
# The detector used to look only INSIDE an ``assert``'s own expression. That anchor was a proxy for
# "this scalar is the verdict", and the proxy is what let the original defect through: the reduction
# sat in a ``return``, so the check reported nothing. Measured at the time of this change: ZERO
# findings across the whole ``tests/`` tree, on a tree that contained four real ones.
#
# The rule now anchors on the ASSERT and follows the value BACKWARDS. Each case below is paired with
# an acceptance case, because a detector that flags everything is worth no more than one that flags
# nothing -- and widening this one turned up four legitimate shapes that had to be excluded by
# construction rather than by luck.


def test_a_pooled_scalar_reached_through_a_local_name_is_refused(tmp_path):
    """The commonest real shape: pool into a name on one line, decide on it on the next.

    Measured in this repo at the time of the widening: three modules under ``tests/_internal/``
    carry exactly this, spelled ``rel = ((got - ref).norm() / ref.norm()).item()`` followed by
    ``assert rel < 2e-2``. None was reported by the assert-anchored rule.
    """
    path = _module(
        tmp_path,
        "def test_x():\n"
        "    rel = ((got - ref).norm() / ref.norm().clamp_min(1e-6)).item()\n"
        "    assert rel < 2e-2\n",
    )
    problems = forbidden_comparison_problems(path)
    assert len(problems) == 1 and "POOLED" in problems[0]
    assert "`rel`" in problems[0], "the message must name the variable the assert decided on"
    assert ".norm()" in problems[0], "the message must name the verb that pooled it"


def test_a_pooled_scalar_returned_by_a_helper_is_refused(tmp_path):
    """THE ORIGINAL HOLE, in its original shape: the reduction lives in a ``return``.

    ``_rel_err`` returned ``max|out-ref| / max|ref|`` and every caller asserted on the result. The
    reduction is nowhere near an ``assert``, so an assert-anchored walk could not see it -- and did
    not, on a file that carried thirteen call sites.
    """
    path = _module(
        tmp_path,
        "def _rel_err(out, ref):\n"
        "    return (out.float() - ref.float()).abs().max().item() / ref.abs().max().item()\n\n\n"
        "def test_x():\n"
        "    rel = _rel_err(recv, expected)\n"
        "    assert rel < 5e-2\n",
    )
    problems = forbidden_comparison_problems(path)
    assert len(problems) == 1 and "POOLED" in problems[0]
    assert "_rel_err()" in problems[0], "the message must name the helper the scalar came from"


def test_a_per_axis_reduction_is_accepted(tmp_path):
    """A reduction that KEEPS an axis cannot shadow an element, so it is not a pooled scalar.

    This is the localized-error statistic the repo deliberately keeps -- a per-column L2 with a
    threshold, counting violators. Flagging it would fail the one check that catches a localized
    store corruption, which is the opposite of what the guard is for.
    """
    path = _module(
        tmp_path,
        "def test_x():\n"
        "    rr = (got - ref).norm(dim=0) / ref.norm(dim=0).clamp_min(1e-9)\n"
        "    n_out = int((rr > 1e-2).sum().item())\n"
        "    assert n_out == 0\n",
    )
    assert forbidden_comparison_problems(path) == []


def test_an_indexed_single_element_read_is_accepted(tmp_path):
    """``.item()`` on an INDEXED element reads one element; it pools nothing.

    ``.item()`` unwraps a tensor that is already a scalar, so the pooling verb is whatever produced
    that scalar. Treating ``item`` as an originator made ``(a - b)[0].item()`` look like a
    reduction; it still propagates, so ``(a - b).abs().max().item()`` is caught by its ``max``.
    """
    path = _module(
        tmp_path,
        "def test_x():\n"
        "    off1 = (row_sum(x, oob_fill=1.0) - base)[0].item()\n"
        "    off7 = (row_sum(x, oob_fill=7.0) - base)[0].item()\n"
        "    assert off7 == 7 * off1\n",
    )
    assert forbidden_comparison_problems(path) == []


def test_an_error_versus_error_comparison_is_accepted(tmp_path):
    """Comparing two pooled errors to EACH OTHER is a relative-accuracy claim, not a gate.

    ``assert e1 <= 2 * e0 + 1e-6`` says one variant must not be LESS accurate than another, and no
    per-element bound expresses that -- there is no reference to be within a bound OF. The banned
    shape is a pooled scalar against a CONSTANT, which is what having an untainted side means.
    """
    path = _module(
        tmp_path,
        "def test_x():\n"
        "    e0 = (outs[0].double() - ref).abs().max().item()\n"
        "    e1 = (outs[1].double() - ref).abs().max().item()\n"
        "    assert e1 <= 2 * e0 + 1e-6\n",
    )
    assert forbidden_comparison_problems(path) == []


def test_a_pooled_scalar_used_only_in_a_message_is_accepted(tmp_path):
    """A pooled scalar a test PRINTS is a diagnostic; only one it DECIDES on is a verdict.

    The rule reports a value that reaches an ``assert``'s test, so a max-abs carried into a
    ``what=`` message -- beside a sanctioned assertion doing the actual judging -- is left alone.
    """
    path = _module(
        tmp_path,
        "from fold_cp_ops.testing.numerics import assert_bitwise\n\n\n"
        "def test_x():\n"
        "    assert_bitwise(early, late, what=f'{(early - late).abs().max().item():.3e}')\n",
    )
    assert forbidden_comparison_problems(path) == []


def test_a_reference_built_from_a_mean_of_a_difference_is_accepted(tmp_path):
    """The LayerNorm formula literally contains a mean of a squared difference.

    Building a REFERENCE is not measuring an error, and the two are indistinguishable by syntax
    alone -- which is why the rule asks where the value GOES rather than what it looks like.
    """
    path = _module(
        tmp_path,
        "import torch\n"
        "from fold_cp_ops.testing.numerics import assert_elementwise\n\n\n"
        "def test_x():\n"
        "    xd = x.double()\n"
        "    ref = 1.0 / torch.sqrt(((xd - xd.mean(-1, keepdim=True)) ** 2).mean(-1) + eps)\n"
        "    assert_elementwise(rstd.double(), ref, 1e-4 * ref.abs())\n",
    )
    assert forbidden_comparison_problems(path) == []


# ── exemptions ─────────────────────────────────────────────────────────────────────────────────


def test_an_exemption_without_a_literal_reason_is_refused(tmp_path):
    """The reason is read from the AST, so a name lookup reads as no reason at all."""
    path = _module(
        tmp_path,
        "WHY = 'because'\n\n\n@numeric_exempt(WHY)\ndef test_x():\n    pass\n",
    )
    problems = forbidden_comparison_problems(path)
    assert len(problems) == 1 and "string literal" in problems[0]


def test_an_exemption_with_a_literal_reason_is_accepted(tmp_path):
    """The acceptance half of the pair above."""
    path = _module(
        tmp_path,
        '@numeric_exempt("asserts a launch COUNT, not a value")\ndef test_x():\n    pass\n',
    )
    assert forbidden_comparison_problems(path) == []


def test_an_empty_exemption_reason_raises_at_the_decoration(tmp_path):
    """Caught at import rather than at audit, so an empty reason cannot even be written."""
    with pytest.raises(ValueError, match="non-empty reason"):
        numeric_exempt("   ")


# ── layer 3: what the coverage gate is scoped to ───────────────────────────────────────────────


def test_a_matrix_parametrized_test_is_in_the_coverage_scope(tmp_path):
    """Drawing shapes from a matrix means launching a kernel, which means producing a value."""
    path = _module(tmp_path, "@GEMM.parametrize('M', 'N')\ndef test_x(M, N):\n    pass\n")
    assert numeric_scope(path) == {"test_x"}


def test_an_unsupported_sweep_is_out_of_scope(tmp_path):
    """``parametrize_unsupported`` asserts a REFUSAL, so there is no computed value to compare.

    Requiring one would fail a correct test, which is the fastest way to get a guard disabled.
    """
    path = _module(tmp_path, "@GEMM.parametrize_unsupported()\ndef test_x(**kw):\n    pass\n")
    assert numeric_scope(path) == set()


def test_a_hand_rolled_parametrize_is_out_of_scope(tmp_path):
    """A bare ``pytest.mark.parametrize`` is the hand-rolled list the matrix replaces.

    It is refused by the kernel-matrix audit, which is where that belongs; the numeric guard does
    not double-report it.
    """
    path = _module(tmp_path, "@pytest.mark.parametrize('M', [1])\ndef test_x(M):\n    pass\n")
    assert numeric_scope(path) == set()


def test_an_exempt_test_leaves_the_coverage_scope(tmp_path):
    """The escape hatch has to actually release the test, or nobody can use it."""
    path = _module(
        tmp_path,
        "@GEMM.parametrize('M')\n@numeric_exempt(\"asserts a launch COUNT\")\n"
        "def test_x(M):\n    pass\n",
    )
    assert numeric_scope(path) == set()


# ── default-deny at the module level ───────────────────────────────────────────────────────────


def test_a_kernel_module_with_no_in_scope_test_and_no_exemption_is_refused(tmp_path):
    """Adding a file must not be a way to opt out of the whole guard."""
    path = _module(
        tmp_path, "def test_x():\n    assert True\n", name="test_pk.py", subdir="kernels"
    )
    problems = audit_numeric_module(path)
    assert len(problems) == 1 and MODULE_EXEMPT_ATTR in problems[0]


def test_the_same_module_with_a_declared_exemption_is_accepted(tmp_path):
    """The acceptance half: the cost of opting out is one sentence somebody had to write."""
    path = _module(
        tmp_path,
        f'{MODULE_EXEMPT_ATTR} = "pure dispatch validation; launches no kernel"\n\n\n'
        "def test_x():\n    assert True\n",
        name="test_pk.py",
        subdir="kernels",
    )
    assert audit_numeric_module(path) == []


def test_a_module_outside_the_numeric_directories_is_not_default_denied(tmp_path):
    """``tests/_internal`` holds helper tests that legitimately compute nothing to compare.

    The default-deny is deliberately scoped rather than global: applied everywhere it would demand
    a declaration from every AST-parsing and cache-key test in the suite, which is churn that buys
    nothing and teaches people to add the marker without reading it.
    """
    path = _module(tmp_path, "def test_x():\n    assert True\n", name="test_pk.py")
    assert audit_numeric_module(path) == []


# ── layer 2: the runtime tripwire and its integrity check ──────────────────────────────────────


def test_the_tripwire_refuses_a_call_and_names_the_replacement():
    """``torch.allclose`` must fail loudly wherever it is reached from, including a helper module.

    The tripwire is installed by ``tests/conftest.py`` at ``pytest_configure``, so it is already
    armed here; calling ``install_tripwire`` again is a no-op by design and is exercised for that.
    """
    install_tripwire()
    with pytest.raises(AssertionError, match="assert_elementwise"):
        torch.allclose(torch.zeros(1), torch.zeros(1))


def test_the_tripwire_survives_the_dynamic_spelling():
    """``getattr(torch, "all" + "close")`` is the evasion a static pass cannot see.

    It works because the tripwire poisons the ATTRIBUTE, not a name binding -- which is the whole
    reason layer 2 exists alongside layer 1 rather than being folded into it.
    """
    install_tripwire()
    fn = getattr(torch, "all" + "close")
    with pytest.raises(AssertionError, match="not a sanctioned comparison"):
        fn(torch.zeros(1), torch.zeros(1))


def test_restoring_the_original_is_detected_and_attributed():
    """A test that puts ``torch.allclose`` back fails, naming what it restored.

    Without this the guard would switch off for every test after the offending one, and the suite
    would keep reporting green -- the failure mode that makes an unverified guard worse than none.

    Restored in a ``finally`` rather than with ``monkeypatch``, and that is not a preference:
    monkeypatch undoes at FIXTURE teardown, which runs after ``pytest_runtest_call`` -- so the
    conftest hook would see the disarmed state and fail this very test for the disarming it is
    deliberately staging. Putting the poison back inside the test body is what keeps the probe from
    tripping the thing it is probing.
    """
    assert tripwire_problem() is None, "the guard must be armed before this probe means anything"
    poison = torch.allclose
    try:
        torch.allclose = lambda *a, **k: True
        problem = tripwire_problem()
    finally:
        torch.allclose = poison
    assert problem is not None and "torch.allclose" in problem
    assert tripwire_problem() is None, "the probe must leave the guard armed for the next test"


# ── the recorder the coverage gate reads ───────────────────────────────────────────────────────


def test_the_recorder_round_trips_and_resets():
    """One test's assertions must never satisfy the next test's gate."""
    reset_assertions()
    assert assertions_recorded() == ()
    record_assertion("assert_elementwise")
    assert assertions_recorded() == ("assert_elementwise",)
    reset_assertions()
    assert assertions_recorded() == ()


def test_every_sanctioned_assertion_actually_records():
    """The gate reads the recorder, so a sanctioned helper that forgot to record would fail tests.

    Checked by calling the real functions on trivially-passing inputs rather than by reading their
    source: a ``record_assertion`` line that is present but unreachable would pass a source scan.
    """
    from fold_cp_ops.testing import numerics

    a = torch.ones(4)
    # NOT zeros: `numerics` refuses an all-zero reference as vacuous, which is itself a check worth
    # not defeating here. ones(2,3) @ ones(2,3)^T is exactly 3, well inside the integer-exact window
    # both GEMM gates require.
    ab, d = torch.ones(2, 3), torch.full((2, 2), 3.0)
    checks = {
        "assert_elementwise": lambda: numerics.assert_elementwise(a, a, 1.0),
        "assert_bitwise": lambda: numerics.assert_bitwise(a, a),
        "assert_gemm_exact": lambda: numerics.assert_gemm_exact(d, ab, ab),
        "assert_gemm_close": lambda: numerics.assert_gemm_close(d, ab, ab),
        # `assert_written` takes a fill callable rather than a tensor: it runs it twice into two
        # differently pre-filled buffers and requires them to agree, so an element the callable
        # never touches keeps two different pre-fills and is reported.
        "assert_written": lambda: numerics.assert_written(
            lambda out: out.fill_(7.0), (2, 2), torch.float32, "cpu"
        ),
    }
    assert set(checks) == set(SANCTIONED_ASSERTIONS), (
        "a sanctioned assertion was added without a recording check here, so the coverage gate "
        "would silently accept a helper that never records"
    )
    for name, call in checks.items():
        reset_assertions()
        call()
        assert name in assertions_recorded(), f"{name} ran but recorded nothing"


# ── the repo-wide lock ─────────────────────────────────────────────────────────────────────────


def test_every_test_module_in_the_repo_is_clean():
    """No module under ``tests/`` holds an unsanctioned comparison. GPU-free; runs everywhere.

    This is the lock that keeps the guard honest in both directions. A weakened rule shows up as
    this test passing while a deliberately-bad probe above also passes -- which is why the probes
    are paired with acceptance cases rather than standing alone.
    """
    problems = []
    for path in sorted((_REPO_ROOT / "tests").rglob("test_*.py")):
        problems.extend(audit_numeric_module(path))
    assert not problems, "unsanctioned numerical comparisons:\n  - " + "\n  - ".join(problems)


def test_the_exempt_modules_are_exactly_the_expected_ones():
    """A module-level exemption turns the static pass OFF, so the set of them must be reviewed.

    Without this lock the previous test degrades quietly: every new exemption makes it check one
    module less, and a suite where nothing is checked passes identically to one where everything is
    clean. Pinning the set means widening it is a visible edit with a reason attached.
    """
    exempt = {
        str(p.relative_to(_REPO_ROOT)): module_exemption(p)
        for p in sorted((_REPO_ROOT / "tests").rglob("test_*.py"))
        if module_exemption(p)
    }
    #: Each entry is here because it is `matrix_exempt` AND therefore has no matrix-parametrized
    #: test for the COVERAGE gate to inspect -- at which point the guard requires an explicit
    #: declaration rather than silence. That is the mechanism; it is NOT "the static pass flags
    #: these files". Measured 2026-08-21: strip the exemption and the static pass reports ZERO
    #: findings in both, so the exemption buys nothing against pooled comparisons and hides nothing.
    #: Both were checked for what they actually compare: the policy module compares NO tensor at
    #: all, and the workflow module's one tensor comparison
    #: (`test_the_fused_chain_matches_the_fp32_oracle`) uses `assert_elementwise` + `tolerance_bound`.
    #: Widening this set is a decision, and it was made on 2026-08-21.
    #:
    #: `test_trimul_tuning.py` was added 2026-08-26 by `8bf7203`, which gave it a module-level
    #: `NUMERIC_EXEMPT` and did NOT add it here -- so this assertion has been RED at every commit
    #: since, and a red guard-test is precisely how a new violation stops being noticed. Checked on
    #: the same terms as the two above rather than waved through: that module is `matrix_exempt`, it
    #: constructs frozen dataclasses and calls `to_engine_kwargs`, it launches no kernel and produces
    #: no tensor, so there is nothing for the coverage gate to inspect and nothing a pooled
    #: comparison could hide. The exemption is correct; omitting it from the reviewed set was the
    #: defect.
    allowed = {
        "tests/testing/test_numeric_guard.py",
        "tests/distributed/workflows/test_trimul_autotuned.py",
        "tests/distributed/workflows/test_trimul_autotune_policy.py",
        "tests/distributed/workflows/test_trimul_tuning.py",
    }
    assert set(exempt) == allowed, (
        f"the numeric guard's static pass is switched off for {sorted(exempt)}, which is not the "
        f"reviewed set -- each exemption checks one module less, and "
        f"a suite where nothing is checked passes identically to one where everything is clean. A "
        f"test that must spell a forbidden form carries the PER-TEST @numeric_exempt instead. "
        f"Reasons given: {exempt}"
    )


def test_a_perf_gate_is_never_in_the_coverage_scope(tmp_path):
    """A perf gate parametrizes from the SAME matrix, and must still not be asked to compare.

    Its subject is a kernel's measured SPEED: it imports the correctness module's matrix, sweeps the
    same cells, and asserts a median against a pin. Every syntactic test for "in scope" says yes,
    and it is the only one in the suite where that answer is wrong -- so the exclusion is by
    directory, checked here on a module that is otherwise identical to an in-scope one.

    Measured before the exclusion existed: 21 perf tests across 9 files would have failed for not
    comparing an output they never produce.
    """
    body = "@GEMM.parametrize('M', 'N')\ndef test_x(M, N):\n    pass\n"
    assert numeric_scope(_module(tmp_path, body, name="test_pf.py", subdir="kernels")) == {"test_x"}
    assert numeric_scope(_module(tmp_path, body, name="test_pf.py", subdir="perf")) == set()
