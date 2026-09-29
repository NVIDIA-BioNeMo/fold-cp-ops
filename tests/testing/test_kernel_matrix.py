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

"""Unit tests for ``tests/kernel_matrix.py`` -- and the two guards that give it teeth.

The last two tests are the reason the module exists. GPU-free, so they run everywhere:

* :func:`test_every_declared_axis_is_diverse` scores every pool against its facets and fails when
  one is degenerate -- the "domain says any integer, pool is all powers of two" case.
* :func:`test_every_kernel_test_draws_on_the_matrix` walks the AST of every kernel and perf test
  and fails any that neither parametrizes from a matrix nor carries a written exemption.
"""

import importlib.util
import pathlib
import types

import pytest

from fold_cp_ops.testing.kernel_matrix import (
    API_LEVEL_ERRORS,
    MIN_FACET_ENTROPY,
    Axis,
    KernelMatrix,
    Unsupported,
    audit_test_module,
    binary_entropy,
    front_door_problem,
    front_door_raises,
    computes_nothing_numeric,
    matrix_exempt,
    matrix_scope,
    no_unsupported,
    unsupported_cell_problem,
)

#: Every matrix built in this file is a PROBE of the machinery, not a kernel: it declares axes to
#: exercise one rule and launches nothing. Sharing one sentinel keeps that fact in a single place
#: rather than repeating a reason at every construction, and keeps the probes from having to model
#: an input property they do not have.
_PROBE = computes_nothing_numeric(because="probe matrix for the machinery; launches nothing")

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]


#: Where a matrix-declaring test module can live. ``kernels`` holds one kernel each; ``workflows``
#: holds the dispatchers that compose them. The second was added when `trimul_autotune` moved there
#: and its matrix silently stopped being scored -- a matrix outside the scanned directories is not
#: reported as unscored, it simply never appears, which is the failure mode this file exists to
#: prevent one level down.
_MATRIX_DIRS = ("kernels", "workflows", "distributed")


def _kernel_test_modules():
    """Import every matrix-declaring test module and return ``(path, module)`` for each.

    Discovery rather than a registry: a registry is only populated once something imports the
    module that fills it, so a guard reading one would pass vacuously when run on its own. This
    imports the modules itself, so the guards below hold no matter what else the invocation
    collected.

    Returns:
        A list of ``(pathlib.Path, module)`` pairs, sorted by path, over every directory in
        :data:`_MATRIX_DIRS`, **recursively**. The recursion is the point: ``tests/distributed/``
        holds ``kernels/`` and ``workflows/`` subdirectories, and a non-recursive glob would scan
        neither while still passing the non-vacuity assert below on the modules it did find --
        green, and blind to everything the A2A bring-back adds.

    Raises:
        AssertionError: If no test modules are found at all, which would make every guard below
            vacuous -- the same failure mode as a CWD-relative glob.
    """
    import importlib

    found = []
    for sub in _MATRIX_DIRS:
        for p in sorted((_REPO_ROOT / "tests" / sub).rglob("test_*.py")):
            # Derived from the path rather than assembled from `sub`, because with rglob the module
            # can sit any number of levels below it. `tests/` is a real package all the way down
            # (see the __init__.py note in CLAUDE.md), so the dotted name is just the relative path.
            dotted = ".".join(p.relative_to(_REPO_ROOT).with_suffix("").parts)
            found.append((p, importlib.import_module(dotted)))
    assert found, f"no matrix test modules under {_MATRIX_DIRS}"
    return found


_KERNEL_MODULES = _kernel_test_modules()
_ALL_MATRICES = {
    m.kernel: m
    for _, mod in _KERNEL_MODULES
    for m in vars(mod).values()
    if isinstance(m, KernelMatrix)
}


def _matrix(name="probe", **kw):
    """Build a throwaway matrix without polluting the shared REGISTRY for other tests.

    Args:
        name: Kernel name; each caller passes a distinct one, since registering two different
            matrices under one name is itself an error the machinery raises.
        **kw: Forwarded to :class:`KernelMatrix`.

    Returns:
        The matrix. Removed from ``REGISTRY`` by the caller is unnecessary -- the diversity guard
        below only walks the matrices declared in ``tests/_matrices.py``.
    """
    kw.setdefault("unsupported", no_unsupported(because="probe matrix; not a real kernel"))
    kw.setdefault("computes", _PROBE)
    return KernelMatrix(kernel=name, **kw)


# ── entropy ───────────────────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize(
    "p,expected", [(0.0, 0.0), (1.0, 0.0), (0.5, 1.0), (0.05, 0.2864), (0.25, 0.8113)]
)
@matrix_exempt("tests a pure scalar function; no kernel shape axes apply")
def test_binary_entropy(p, expected):
    """H is 0 at both extremes and 1 when balanced; 0.05 sits just above the default threshold.

    The 0.05 row is the one that matters in practice: it is the "one representative in twenty"
    case, and it must clear ``MIN_FACET_ENTROPY`` so that a deliberately-rare config (an odd N, a
    single production-scale M) is not reported as missing coverage.
    """
    assert binary_entropy(p) == pytest.approx(expected, abs=1e-4)
    assert binary_entropy(0.05) > MIN_FACET_ENTROPY


@matrix_exempt("tests a pure scalar function's error path; no kernel shape axes apply")
def test_binary_entropy_rejects_a_non_fraction():
    """A fraction outside [0,1] is a caller bug, not something to clamp.

    Clamping would report a coverage score that does not correspond to the pool it came from --
    the one failure mode a coverage checker must not have.
    """
    with pytest.raises(ValueError, match="fraction"):
        binary_entropy(1.5)


# ── Axis ──────────────────────────────────────────────────────────────────────────────────────
@matrix_exempt("tests the matrix machinery itself, which has no kernel shape axes")
def test_axis_rejects_pools_that_would_skew_the_score():
    """An empty pool, a duplicate value, or a waiver for a facet that does not exist all raise.

    Duplicates are the subtle one: they change every facet's fraction without adding coverage, so
    a pool padded with repeats would score as more diverse than it is.
    """
    with pytest.raises(ValueError, match="pool is empty"):
        Axis(name="N", domain="any int", values=())
    with pytest.raises(ValueError, match="duplicate value"):
        Axis(name="N", domain="any int", values=(8, 8))
    with pytest.raises(ValueError, match="unknown facet"):
        Axis(name="N", domain="any int", values=(8,), facets={}, waived={"nope": "reason"})
    # A waiver suppresses a coverage failure, so a reasonless one is just a deleted check.
    with pytest.raises(ValueError, match="give no reason"):
        Axis(
            name="N", domain="d", values=(8,), facets={"odd": lambda v: v % 2}, waived={"odd": " "}
        )


@matrix_exempt("tests the matrix machinery itself, which has no kernel shape axes")
def test_axis_diversity_catches_an_all_powers_of_two_pool():
    """**The motivating case.** A pool that is entirely powers of two scores H=0 on that facet.

    This is the example the config system was asked for: the domain claims any integer, the pool
    only lists powers of two, and the checker must say so rather than pass.
    """
    axis = Axis(
        name="N",
        domain="any positive int",
        values=(64, 128, 256, 512),
        facets={"power_of_two": lambda v: v & (v - 1) == 0, "odd": lambda v: v % 2},
    )
    scores = {s.facet: s for s in axis.diversity()}
    assert scores["power_of_two"].entropy == 0.0 and scores["power_of_two"].hits == 4
    assert scores["odd"].entropy == 0.0 and scores["odd"].hits == 0
    # ...and adding one off-grid odd value repairs both facets at once.
    repaired = Axis(name="N", domain=axis.domain, values=(64, 128, 256, 999), facets=axis.facets)
    assert all(s.entropy > MIN_FACET_ENTROPY for s in repaired.diversity())


@matrix_exempt("tests the matrix machinery itself, which has no kernel shape axes")
def test_axis_waiver_exempts_a_facet_but_keeps_it_visible():
    """A waived facet still scores and still prints -- it is exempt, not deleted.

    Deleting an unreachable facet would erase the fact that the axis has an unreachable region.
    The waiver keeps the reason in the report where the next reader will see it.
    """
    axis = Axis(
        name="N",
        domain="any positive int",
        values=(64, 128),
        facets={"odd": lambda v: v % 2},
        waived={"odd": "16-bit copy atom fails IR verification at odd N"},
    )
    (score,) = axis.diversity()
    assert score.entropy == 0.0 and score.waived.startswith("16-bit")
    assert "waived:" in str(score)


# ── parametrize ───────────────────────────────────────────────────────────────────────────────
@matrix_exempt("tests the matrix machinery itself, which has no kernel shape axes")
def test_parametrize_builds_the_cross_product():
    """Unrestricted, the decorator yields the full product with readable ids."""
    m = _matrix("probe_product", axes=(Axis("N", "any int", (8, 16)), Axis("M", "any int", (1, 2))))
    mark = m.parametrize("N", "M")
    assert mark.args[0] == "N,M"
    assert list(mark.args[1]) == [(8, 1), (8, 2), (16, 1), (16, 2)]
    assert mark.kwargs["ids"] == ["N8-M1", "N8-M2", "N16-M1", "N16-M2"]


@matrix_exempt("tests the matrix machinery itself, which has no kernel shape axes")
def test_restricting_without_a_reason_is_refused():
    """**This is the lock.** ``only=``/``drop=``/``cells=`` without ``because=`` raises at import.

    A restriction and a forgotten config produce the same grid, so the only thing that can tell
    them apart is a sentence someone had to write. Failing at import rather than at run time means
    the author sees it immediately, not in CI.
    """
    m = _matrix("probe_reason", axes=(Axis("N", "any int", (8, 16, 32)),))
    with pytest.raises(ValueError, match="requires because="):
        m.parametrize("N", only={"N": [8]})
    with pytest.raises(ValueError, match="requires because="):
        m.parametrize("N", drop={"N": [8]})
    # ...and a reason with nothing to justify is equally a mistake.
    with pytest.raises(ValueError, match="nothing is restricted"):
        m.parametrize("N", because="describes nothing")
    # The sanctioned form works and narrows the grid.
    ok = m.parametrize("N", only={"N": [8, 32]}, because="the 16 case is covered by the perf gate")
    assert list(ok.args[1]) == [8, 32]


@matrix_exempt("tests the matrix machinery itself, which has no kernel shape axes")
def test_a_test_cannot_invent_values_outside_the_pool():
    """Selecting an undeclared value raises, and the message says to add it to the matrix.

    Without this the diversity check is hollow: a test could quietly use shapes the checker never
    scores, which is precisely the state the matrix replaced.
    """
    m = _matrix("probe_pool", axes=(Axis("N", "any int", (8, 16)),))
    with pytest.raises(ValueError, match=r"not in the declared pool"):
        m.parametrize("N", only={"N": [8, 999]}, because="probing")
    with pytest.raises(KeyError, match="no axis 'nope'"):
        m.parametrize("nope")


@matrix_exempt("tests the matrix machinery itself, which has no kernel shape axes")
def test_cells_take_an_explicit_grid_but_still_check_every_value():
    """``cells=`` parametrizes a list of tuples rather than a product, and still validates them.

    A perf gate pins one median per cell, so a cross product would be hundreds of timed runs; but
    the components must still come from the pool, or timing and correctness would drift apart.
    """
    m = _matrix("probe_cells", axes=(Axis("N", "any int", (8, 16)), Axis("M", "any int", (1, 2))))
    mark = m.parametrize("N", "M", cells=[(8, 2), (16, 1)], because="perf pins are per-cell")
    assert list(mark.args[1]) == [(8, 2), (16, 1)]
    with pytest.raises(ValueError, match="not in the declared pool"):
        m.parametrize("N", "M", cells=[(8, 99)], because="probing")
    with pytest.raises(ValueError, match="arity"):
        m.parametrize("N", "M", cells=[(8,)], because="probing")
    with pytest.raises(ValueError, match="excludes only=/drop="):
        m.parametrize("N", cells=[(8,)], only={"N": [8]}, because="probing")


@matrix_exempt("tests the matrix machinery itself, which has no kernel shape axes")
def test_matrix_exempt_requires_a_reason():
    """An exemption with no reason is exactly the silence the guard exists to break."""
    with pytest.raises(ValueError, match="non-empty reason"):
        matrix_exempt("")


# ── GUARD 1: every declared pool is diverse ───────────────────────────────────────────────────
@pytest.mark.parametrize("kernel", sorted(_ALL_MATRICES), ids=lambda k: k)
@matrix_exempt("scores the matrices themselves; parametrized over kernels, not over shapes")
def test_every_declared_axis_is_diverse(kernel, capsys):
    """Every non-waived facet of every axis clears ``MIN_FACET_ENTROPY``.

    **What this catches:** an axis whose ``domain`` claims a broad space while its pool samples one
    corner of it. The failure names the facet and the count, so the fix is always "add a value of
    this kind", never a hunt.

    The full report is printed on failure (and under ``-s``) so a near-miss is visible before it
    becomes a regression.
    """
    matrix = _ALL_MATRICES[kernel]
    scores = [s for axis in matrix.axes for s in axis.diversity()]
    assert scores, f"{kernel}: no facets declared on any axis -- nothing is being checked"

    degenerate = [s for s in scores if s.waived is None and s.entropy < MIN_FACET_ENTROPY]
    if degenerate:
        report = "\n  ".join(str(s) for s in scores)
        offenders = "\n  ".join(
            f"{s.axis}.{s.facet}: {s.hits}/{s.total} values, H={s.entropy:.2f}\n"
            f"      domain claims: {matrix.axis(s.axis).domain}"
            for s in degenerate
        )
        pytest.fail(
            f"{kernel}: {len(degenerate)} facet(s) below H={MIN_FACET_ENTROPY} -- the pool samples "
            f"one corner of the domain it claims:\n  {offenders}\n\n"
            f"Add values of the missing kind to the matrix in its kernel's test module, or waive "
            f"the facet there with a reason if it is genuinely unreachable.\n\n"
            f"Full report:\n  {report}"
        )
    with capsys.disabled():
        print(f"\n[matrix {kernel}] " + " | ".join(str(s) for s in scores))


# ── scope: WHERE the audit applies ────────────────────────────────────────────────────────────


@matrix_exempt("pure path algebra on the audit's own scope rule; no kernel, shape or dtype")
@pytest.mark.parametrize(
    "relpath,expected",
    [
        ("tests/kernels/test_gemm.py", True),
        ("tests/perf/test_benchmark_perf_gemm.py", True),
        ("tests/workflows/test_trimul_autotune.py", True),
        ("tests/distributed/test_pe_map.py", True),
        # The two that motivated this rule. Neither directory existed when it was written, and
        # under the old `parent.name` spelling the first passed by COINCIDENCE (its leaf happens to
        # be "kernels") while the second -- its sibling -- did not pass at all.
        ("tests/distributed/kernels/test_gemm_sm90_a2a.py", True),
        ("tests/distributed/workflows/test_trimul_autotuned.py", True),
        ("tests/distributed/perf/test_benchmark_perf_gemm_sm90_a2a.py", True),
        ("tests/distributed/kernels/deeper/test_y.py", True),
        # Out of scope: not a governed directory, or not a test module.
        ("tests/test_conftest.py", False),
        ("tests/testing/test_kernel_matrix.py", False),
        ("tests/distributed/kernels/conftest.py", False),
        ("tests/distributed/kernels/helper.py", False),
        ("tests/_internal/test_autotuner.py", False),
    ],
)
def test_matrix_scope_follows_the_directory_at_any_depth(relpath, expected):
    """A module is audited iff some path component is a governed directory AND it is a ``test_*.py``.

    Matched on PARTS rather than the immediate parent, which is what makes an unbuilt subdirectory
    governed in advance. The `tests/distributed/{kernels,workflows}` rows are the ones with teeth:
    they are where the A2A bring-back lands, and the previous spelling covered one of them only
    because its leaf name collided with a literal written for `tests/kernels/`.
    """
    assert matrix_scope(relpath) is expected


@matrix_exempt("path algebra over a synthetic module; no kernel, shape or dtype")
@pytest.mark.parametrize(
    "kernel,kernels_dir,expected_gate",
    [
        ("gate_probe_local", "tests/kernels", "tests/perf"),
        ("gate_probe_dist", "tests/distributed/kernels", "tests/distributed/perf"),
    ],
)
def test_the_required_perf_gate_is_the_sibling_perf_directory(
    tmp_path, kernel, kernels_dir, expected_gate
):
    """A kernel's perf gate is looked for in the ``perf/`` SIBLING of its ``kernels/`` directory.

    Computed rather than hardcoded to ``tests/perf/``, and that generalization is what the
    distributed tree needs: a distributed gate has to be launched under torchrun/srun with a live
    process group, so it cannot sit in ``tests/perf/``, which is collected in a plain isolated
    session. Pinned here because that message is the only place a developer learns where to put it.
    """
    import types

    path = tmp_path / kernels_dir / f"test_{kernel}.py"
    path.parent.mkdir(parents=True)
    path.write_text(f"@{kernel}.parametrize('M')\ndef test_a(M):\n    pass\n")
    module = types.ModuleType(f"probe_{kernel}")
    setattr(
        module,
        kernel,
        _matrix(
            name=kernel,
            axes=(
                Axis(
                    name="M",
                    domain="probe",
                    values=(16, 32),
                    facets={
                        "small": lambda v: v < 24,
                        "big": lambda v: v >= 24,
                        "bad_dtype": lambda v: False,
                        "bad_extent": lambda v: False,
                    },
                    waived={"bad_dtype": "probe matrix", "bad_extent": "probe matrix"},
                ),
            ),
        ),
    )
    problems = [p for p in audit_test_module(path, module) if "there is no" in p]
    assert problems, "the perf-gate check did not fire at all -- the probe proves nothing"
    assert f"{expected_gate}/test_benchmark_perf_{kernel}.py" in problems[0], problems[0]


@matrix_exempt("pure path algebra; asserts the conftest hook and this file share one predicate")
def test_the_collection_hook_scopes_the_audit_through_this_same_predicate():
    """`tests/conftest.py` must call :func:`matrix_scope`, not re-spell the rule.

    The audit's RULES are already shared between the collection hook and this file through
    `audit_test_module`; its SCOPE was not, and a scope that disagrees is a rule that silently does
    not run. Source-read rather than behavioural because the hook needs a live pytest session to
    exercise, and a check that expensive is one nobody keeps.
    """
    src = (_REPO_ROOT / "tests" / "conftest.py").read_text()
    assert "matrix_scope" in src, (
        "tests/conftest.py no longer scopes the matrix audit through kernel_matrix.matrix_scope. "
        "A locally re-spelled directory list is how tests/distributed/workflows/ came to be "
        "outside the audit while tests/workflows/ was inside it."
    )


# ── GUARD 2: every kernel test draws on a matrix ──────────────────────────────────────────────
_SCANNED = sorted(
    p
    for d in ("tests/kernels", "tests/perf", "tests/workflows", "tests/distributed")
    for p in (_REPO_ROOT / d).rglob("test_*.py")
)
assert _SCANNED, "found no kernel/perf test modules to scan -- the guard below would be vacuous"


@pytest.mark.parametrize("path", _SCANNED, ids=lambda p: p.name)
@matrix_exempt("scans test sources; parametrized over modules, not over shapes")
def test_every_kernel_test_draws_on_the_matrix(path):
    """Every kernel and perf test module satisfies :func:`audit_test_module`.

    **The same function ``tests/conftest.py`` runs at collection**, so the two enforcement points
    cannot disagree. The conftest hook is the one that cannot be skipped -- running a single test
    file still triggers it -- while this test exists so a violation reads as a test failure with a
    full report rather than a usage error.

    Checked: every ``def test_*`` parametrizes from a matrix or is exempt with a literal reason;
    each ``tests/kernels/test_<k>.py`` declares exactly one matrix named ``<k>`` and has a perf
    gate; each ``tests/perf/test_benchmark_perf_<k>.py`` imports that same object.
    """
    import importlib

    # Anchor on the nearest ancestor named `tests`, NOT on `path.parent.name`. The two agree for
    # every gate directly under `tests/` and DISAGREE for a nested one: the distributed A2A gate
    # lives at `tests/distributed/perf/`, where `path.parent.name` is "perf", so the old form built
    # `tests.perf.test_benchmark_perf_dual_gated_gemm_a2a` -- a module that does not exist -- and
    # the test died with ModuleNotFoundError instead of auditing anything. This is the SAME defect
    # `audit_test_module` already fixed on its own side (see its `tests_root` comment); the fix had
    # not been carried across, so it reappeared the moment the first nested gate was authored.
    #
    # THE GENERAL SHAPE, worth more than this one fix: an artifact placed one directory deeper than
    # any existing one silently leaves the auditor's reach, and the auditor reports NOTHING because
    # it never sees the file -- silence, so no run turns red. It happened twice from
    # one cause: this dotted-name derivation, and `_KERNEL_DIRS == {"kernels"}` excluding all of
    # `tests/distributed/` from `coverage_problems` (see that constant's comment). Both were found
    # while doing something else. When adding a module or a gate at a NEW depth, verify the auditor
    # can still NAME it before trusting that it audits it.
    tests_root = next((q for q in path.parents if q.name == "tests"), path.parent.parent)
    dotted = ".".join((tests_root.name, *path.relative_to(tests_root).with_suffix("").parts))
    problems = audit_test_module(path, importlib.import_module(dotted))
    assert not problems, f"{path.name}:\n  - " + "\n  - ".join(problems)


# ── unsupported regions: the combos a kernel must REFUSE ──────────────────────────────────────
@matrix_exempt("tests the matrix machinery itself, which has no kernel shape axes")
def test_unsupported_is_mandatory():
    """**Omitting ``unsupported=`` raises.** A silent default is the whole failure mode.

    A kernel that never states which combos it refuses is a kernel that can silently accept one and
    return a wrong answer. Making the field mandatory does not discover unknown-unknowns -- nothing
    can -- but it converts "nobody knew" into "nobody wrote it down", which a reviewer can see.
    An empty tuple is rejected for the same reason: it reads as "not filled in".
    """
    ax = (Axis("N", "any int", (8, 9)),)
    with pytest.raises(ValueError, match="is REQUIRED"):
        KernelMatrix(kernel="probe_unsup_missing", axes=ax)
    with pytest.raises(ValueError, match="is REQUIRED"):
        KernelMatrix(kernel="probe_unsup_empty", axes=ax, computes=_PROBE, unsupported=())
    # "there are none" is sayable -- but only with the reasoning.
    with pytest.raises(ValueError, match="requires because"):
        no_unsupported(because="  ")
    KernelMatrix(
        kernel="probe_unsup_none",
        axes=ax,
        computes=_PROBE,
        unsupported=no_unsupported(because="every N in the pool computes; nothing is refused"),
    )


@matrix_exempt("tests the matrix machinery itself, which has no kernel shape axes")
def test_unsupported_region_must_be_reachable_and_explained():
    """A region that matches nothing, or explains nothing, is rejected at construction.

    An unreachable region is decoration: no test can exercise it, so the kernel could stop raising
    and nothing would notice. A region naming an axis that does not exist is a typo that would
    otherwise match nothing, i.e. the same thing with a friendlier appearance.
    """
    with pytest.raises(ValueError, match="reason must say why"):
        Unsupported(where=lambda N: N % 2 == 1, raises=ValueError, match="m", reason="")
    with pytest.raises(ValueError, match="non-empty regex"):
        Unsupported(where=lambda N: N % 2 == 1, raises=ValueError, match="", reason="why")
    # An internal explosion is the kernel FAILING, not REFUSING. A region cannot be satisfied by
    # one -- that is what obliges the developer to add a front-door check.
    for internal in (Exception, RuntimeError, ArithmeticError):
        with pytest.raises(ValueError, match="front door"):
            Unsupported(where=lambda N: N % 2 == 1, raises=internal, match="x", reason="r")
    reachable = Unsupported(
        where=lambda N: N % 2 == 1, raises=ValueError, match="boom", reason="odd is refused"
    )
    with pytest.raises(ValueError, match="matches NO combo"):
        KernelMatrix(
            "probe_unreach",
            axes=(Axis("N", "d", (8, 16)),),
            computes=_PROBE,
            unsupported=(reachable,),
        )
    with pytest.raises(ValueError, match="undeclared axes"):
        KernelMatrix(
            "probe_badaxis",
            axes=(Axis("N", "d", (8, 9)),),
            computes=_PROBE,
            unsupported=(
                Unsupported(where=lambda Q: Q > 0, raises=ValueError, match="b", reason="r"),
            ),
        )


@matrix_exempt("tests the matrix machinery itself, which has no kernel shape axes")
def test_a_supported_test_cannot_cover_an_unsupported_cell():
    """``parametrize`` refuses to emit a cell inside an unsupported region.

    Asserting correctness on a combo declared to RAISE is a contradiction, and nothing else would
    surface it -- the correctness test would simply fail later, looking like a kernel bug.
    """
    m = KernelMatrix(
        "probe_contradiction",
        axes=(Axis("N", "d", (8, 9)),),
        computes=_PROBE,
        unsupported=(
            Unsupported(
                where=lambda N: N % 2 == 1, raises=ValueError, match="b", reason="odd refused"
            ),
        ),
    )
    with pytest.raises(ValueError, match="lies in the unsupported region"):
        m.parametrize("N")
    ok = m.parametrize(
        "N", only={"N": [8]}, because="9 is unsupported and covered by the raises test"
    )
    assert list(ok.args[1]) == [8]


@matrix_exempt("tests the matrix machinery itself, which has no kernel shape axes")
def test_parametrize_unsupported_sweeps_exactly_the_refused_cells():
    """It yields the matching cells plus each one's expected error pattern, and nothing else."""
    m = KernelMatrix(
        "probe_sweep",
        axes=(Axis("N", "d", (8, 9, 11)),),
        computes=_PROBE,
        unsupported=(
            Unsupported(
                where=lambda N: N % 2 == 1, raises=ValueError, match="ICE", reason="odd refused"
            ),
        ),
    )
    mark = m.parametrize_unsupported("N")
    assert mark.args[0] == "N,expected_error,expected_match"
    assert list(mark.args[1]) == [(9, ValueError, "ICE"), (11, ValueError, "ICE")]
    # A kernel with no regions has nothing to sweep, and saying otherwise would never run.
    none = _matrix("probe_sweep_none", axes=(Axis("N", "d", (8,)),))
    with pytest.raises(ValueError, match="nothing for parametrize_unsupported"):
        none.parametrize_unsupported("N")


@matrix_exempt("tests the matrix machinery itself, which has no kernel shape axes")
def test_a_declared_region_must_actually_be_tested(tmp_path):
    """``audit_test_module`` fails a module that declares a region but never sweeps it.

    Declaring and testing have to be one act; otherwise the kernel could stop raising and the
    declaration would sit there asserting otherwise.
    """
    # must live under a directory named "kernels" -- that is how audit_test_module knows the
    # per-kernel rules apply at all.
    (tmp_path / "kernels").mkdir()
    mod = tmp_path / "kernels" / "test_probekernel.py"
    mod.write_text("def test_x():\n    pass\n")

    class _Fake:
        pass

    fake = _Fake()
    fake.M = KernelMatrix(
        "probekernel",
        axes=(Axis("N", "d", (8, 9)),),
        computes=_PROBE,
        unsupported=(
            Unsupported(
                where=lambda N: N % 2 == 1, raises=ValueError, match="b", reason="odd refused"
            ),
        ),
    )
    problems = audit_test_module(mod, fake)
    assert any("parametrize_unsupported()" in p for p in problems), problems


@matrix_exempt("tests the matrix machinery itself, which has no kernel shape axes")
def test_an_assert_based_guard_cannot_satisfy_a_region():
    """``AssertionError`` is excluded from :data:`API_LEVEL_ERRORS`, and the reason is demonstrated.

    A region promises "this combo is refused". A guard written as ``assert cond, msg`` does not
    keep that promise, because ``python -O`` deletes it -- the kernel then sails past the check and
    fails however it was going to fail internally, which is the state the region existed to
    prevent. So the region must name an exception a ``raise`` produces.

    The subprocess below is the evidence rather than the claim: it runs the same guard with and
    without ``-O`` and shows the assert-based one silently stops firing. Without it this test would
    just be restating the allowlist back to itself.
    """
    import subprocess
    import sys

    assert AssertionError not in API_LEVEL_ERRORS
    with pytest.raises(ValueError, match="front door"):
        Unsupported(where=lambda N: N % 2 == 1, raises=AssertionError, match="x", reason="r")

    guard = (
        "import sys\n"
        "def f(n):\n"
        "    assert n % 2 == 0, 'odd n unsupported'\n"
        "    return n\n"
        "try:\n"
        "    f(3); print('NO_RAISE')\n"
        "except AssertionError:\n"
        "    print('RAISED')\n"
    )
    plain = subprocess.run([sys.executable, "-c", guard], capture_output=True, text=True)
    optimized = subprocess.run([sys.executable, "-O", "-c", guard], capture_output=True, text=True)
    assert plain.stdout.strip() == "RAISED", plain
    assert optimized.stdout.strip() == "NO_RAISE", (
        "python -O no longer strips asserts, so the rationale for excluding AssertionError from "
        f"API_LEVEL_ERRORS may not hold: {optimized}"
    )


@matrix_exempt("tests the matrix machinery itself, which has no kernel shape axes")
def test_a_region_needs_a_test_that_actually_asserts_the_raise(tmp_path):
    """Declaring a region obliges a test that BOTH sweeps it and asserts the raise.

    Two ways to appear compliant while guaranteeing nothing, both measured as holes before this
    check existed:

    * the name ``parametrize_unsupported`` appearing only in a comment or docstring -- a substring
      search over the source is satisfied by it, which is why the check reads the AST instead;
    * the decorator present but a body that never calls the kernel inside
      ``pytest.raises(expected_error, ...)`` -- it sweeps every refused combo and asserts nothing
      about any of them.

    Only ONE such test is required per kernel, and that is deliberate: ``parametrize_unsupported``
    refuses to run unless its axes can evaluate every declared region, so a single test covers them
    all. Other tests are not merely excused from the refused combos -- ``parametrize()`` actively
    refuses to emit one, so a correctness test cannot wander into a region by accident.
    """
    (tmp_path / "kernels").mkdir()
    (tmp_path / "perf").mkdir()
    (tmp_path / "perf" / "test_benchmark_perf_pkregion.py").write_text("")

    class _Fake:
        pass

    fake = _Fake()
    fake.M = KernelMatrix(
        "pkregion",
        axes=(Axis("N", "d", (8, 9)),),
        computes=_PROBE,
        unsupported=(
            Unsupported(where=lambda N: N % 2 == 1, raises=ValueError, match="b", reason="odd"),
        ),
    )
    # filename and matrix kernel name MUST agree, or the name-mismatch check fires first
    # and the region check below is never reached.
    mod = tmp_path / "kernels" / "test_pkregion.py"

    def probe(src):
        mod.write_text(src)
        return [p for p in audit_test_module(mod, fake) if "unsupported region" in p]

    assert probe('# @M.parametrize_unsupported("N") -- someday\ndef test_a(): pass\n'), (
        "a mention in a comment must not satisfy the check"
    )
    assert probe(
        '@M.parametrize_unsupported("N")\n'
        "def test_u(N, expected_error, expected_match):\n    pass\n"
    ), "the decorator alone, with a body that asserts nothing, must not satisfy the check"
    assert not probe(
        '@M.parametrize_unsupported("N")\n'
        "def test_u(N, expected_error, expected_match):\n"
        "    with pytest.raises(expected_error, match=expected_match):\n        f()\n"
    ), "a real sweep asserting the raise must satisfy it"


# ── front_door_raises: the raise must come from the entry point, not from inside the compiler ──
# `API_LEVEL_ERRORS` makes the exception TYPE checkable and stops there. These cover the half it
# cannot: that the author wrote an explicit check instead of letting the DSL explode.
#
# The pure-predicate tests come first and use SYNTHETIC frame lists, which is what lets rule 1 be
# isolated from rule 2 -- the case that matters most (our own code raising during tracing) cannot be
# built as a real traceback without putting a deliberately-broken `@cute.jit` function into the
# shipped package.

_PKG = str(pathlib.Path(__file__).resolve().parents[2] / "fold_cp_ops")
_DSL = (
    "/opt/conda/lib/python3.12/site-packages/nvidia_cutlass_dsl/python_packages/cutlass/"
    "utils/hopper_helpers.py"
)


@matrix_exempt("tests the matrix machinery itself, which has no kernel shape axes")
def test_front_door_problem_accepts_a_raise_from_the_entry_module():
    """The plain case: the entry raised, nothing else is on the path."""
    assert front_door_problem([f"{_PKG}/kernels/gemm.py"]) is None


@matrix_exempt("tests the matrix machinery itself, which has no kernel shape axes")
def test_front_door_problem_accepts_a_shared_helper_under_the_package_root():
    """A front door that DELEGATES its check is still a front door.

    This is the answer to "what if the check calls a helper?", and it is yes -- ``gemm()`` refuses a
    bad dtype through ``tensor_contract.check_tensor`` and a bad stride through
    ``gemm_tvm_ffi_utils._raise_misaligned``, both shared by several entries. Requiring the literal
    ``raise`` in the entry module would force every check to be inlined, which is worse code and
    buys nothing: what makes these front-door checks is that nothing has been traced yet.
    """
    assert (
        front_door_problem(
            [
                "/repo/tests/kernels/test_gemm.py",
                f"{_PKG}/kernels/gemm.py",
                f"{_PKG}/_internal/tensor_contract.py",
            ]
        )
        is None
    )


@matrix_exempt("tests the matrix machinery itself, which has no kernel shape axes")
def test_front_door_problem_rejects_a_raise_from_inside_the_dsl():
    """A DSL frame on the path means the kernel was already compiling -- a latent check."""
    problem = front_door_problem(
        ["/repo/tests/kernels/test_x.py", f"{_PKG}/kernels/gemm_layernorm_gemm.py", _DSL]
    )
    assert problem and "LATENT check" in problem
    assert "NECESSARY BUT NOT SUFFICIENT" in problem, "the message must teach, not just fail"
    assert "entry point" in problem, "and must say what to do about it"


@matrix_exempt("tests the matrix machinery itself, which has no kernel shape axes")
def test_front_door_problem_rejects_a_latent_raise_even_when_our_own_code_raised_it():
    """**The control that proves rule 1 is not redundant with rule 2.**

    Here the RAISING frame is under the package root -- our own ``@cute.jit`` body -- so the
    "is it our code" rule passes it. It is still a latent check: a DSL frame sits on the path, so
    the caller only reached this by paying a compile. Measured shape: the real fp32 traceback has
    ``.../fold_cp_ops/kernels/gemm_layernorm_gemm.py:703 in __call__`` one frame ABOVE the raise,
    for exactly this reason -- had that line raised instead of ``hopper_helpers``, rule 2 alone
    would have let it through.
    """
    frames = [
        "/repo/tests/kernels/test_x.py",
        f"{_PKG}/kernels/gemm.py",
        _DSL.replace("utils/hopper_helpers.py", "base_dsl/dsl.py"),
        f"{_PKG}/kernels/gemm_sm90.py",  # our code, but executing INSIDE cute.compile
    ]
    assert front_door_problem(frames[-1:]) is None, (
        "the raising frame alone passes rule 2 -- which is the point: rule 2 cannot see this case"
    )
    problem = front_door_problem(frames)
    assert problem and "LATENT check" in problem


@matrix_exempt("tests the matrix machinery itself, which has no kernel shape axes")
def test_front_door_problem_rejects_a_third_party_raising_frame():
    """A ``ValueError`` out of torch is not this package refusing the combination."""
    problem = front_door_problem(
        ["/repo/tests/kernels/test_x.py", "/opt/conda/lib/python3.12/site-packages/torch/_meta.py"]
    )
    assert problem and "not under" in problem and "third-party" in problem


@matrix_exempt("tests the matrix machinery itself, which has no kernel shape axes")
def test_front_door_problem_rejects_a_helper_that_failed_inside_the_test():
    """A test helper that dies before reaching the kernel is not a refusal either.

    ``tests/`` sits beside the package, not under it, so this falls out of rule 2 rather than
    needing its own rule -- and it is worth a cell because it is the likeliest way a region test
    passes while testing nothing: mis-build an input, get a ``ValueError`` from the builder, and
    the ``match`` happens to be loose enough.
    """
    problem = front_door_problem(["/repo/tests/kernels/test_x.py", "/repo/tests/kernels/helper.py"])
    assert problem and "not under" in problem


@matrix_exempt("tests the matrix machinery itself, which has no kernel shape axes")
def test_front_door_problem_matches_a_path_component_and_not_a_substring():
    """``/home/me/cutlass_notes/mykernel.py`` is not the toolchain.

    A substring test for ``"cutlass"`` would fail that path, and it would do so only on the machine
    whose directories happened to be named that way -- the worst kind of check to debug.
    """
    assert front_door_problem([f"{_PKG}/kernels/gemm.py"]) is None
    assert front_door_problem(["/home/me/cutlass_notes/x.py"]), "not ours, so rule 2 fires"
    assert front_door_problem([f"{_PKG}/notes_about_cutlass_things.py"]) is None, (
        "a FILE whose name contains the word is still ours; only a path COMPONENT is the toolchain"
    )


@matrix_exempt("tests the matrix machinery itself, which has no kernel shape axes")
def test_front_door_problem_refuses_an_empty_traceback():
    """Nothing to judge is refused, not passed -- passing would green-light every raise."""
    with pytest.raises(ValueError, match="at least one traceback frame"):
        front_door_problem([])


@matrix_exempt("tests the matrix machinery itself, which has no kernel shape axes")
def test_front_door_raises_keeps_every_guarantee_pytest_raises_already_gave():
    """DID NOT RAISE, the wrong type and a non-matching message all still fail, and first.

    Provenance is judged LAST on purpose: a failure message should describe the most specific thing
    that is wrong, not complain about where an exception of the wrong type came from.
    """
    with pytest.raises(pytest.fail.Exception, match="DID NOT RAISE"):
        with front_door_raises(ValueError, "anything"):
            pass
    with pytest.raises(TypeError):
        with front_door_raises(ValueError, "anything"):
            raise TypeError("the wrong type propagates out, exactly as with pytest.raises")
    with pytest.raises(AssertionError, match="Regex pattern did not match"):
        with front_door_raises(ValueError, "the constraint"):
            raise ValueError("a different message")


@matrix_exempt("tests the matrix machinery itself, which has no kernel shape axes")
def test_front_door_raises_passes_a_raise_from_this_package_and_yields_the_excinfo():
    """A real package-level raise passes, and ``as excinfo`` still works."""
    from fold_cp_ops._internal.tensor_contract import check_tensor

    import torch

    with front_door_raises(ValueError, r"unsupported dtype for probe") as excinfo:
        check_tensor("probe", torch.zeros(4, dtype=torch.float64))
    assert excinfo.type is ValueError


@matrix_exempt("tests the matrix machinery itself, which has no kernel shape axes")
def test_front_door_raises_rejects_a_latent_raise_that_pytest_raises_accepts():
    """**Negative control 1, end to end, on a real kernel.**

    ``front_door_problem`` is exercised above with synthetic frames; this drives the whole thing
    through an actual latent raise so the two cannot drift. The exception is synthesised here rather
    than by calling a kernel, because every kernel in this tree now HAS its front-door check -- the
    frame list is taken verbatim from the measured fp32 traceback (recorded in
    ``front_door_raises``'s docstring), so what is being tested is the same judgement on the same
    paths.
    """
    frames = [f"{_PKG}/kernels/gemm_layernorm_gemm.py", _DSL]
    # pytest.raises is satisfied: the TYPE is in API_LEVEL_ERRORS and the message matches.
    msg = "unsupported a_dtype and b_dtype, got Float32 and Float32"
    with pytest.raises(TypeError, match=r"unsupported.*dtype"):
        raise TypeError(msg)
    # front_door_raises is NOT.
    problem = front_door_problem(frames)
    assert problem is not None and "LATENT check" in problem, (
        "the measured fp32 traceback must be rejected; a TypeError from hopper_helpers is the "
        "exact deep explosion a region exists to forbid"
    )


@matrix_exempt("tests the matrix machinery itself, which has no kernel shape axes")
def test_the_audit_still_recognises_a_region_test_written_with_front_door_raises(tmp_path):
    """``front_door_raises`` satisfies ``_has_working_unsupported_test``, and that is PINNED.

    The audit recognises a real region test by finding a call whose function name ends in
    ``raises`` and whose first argument is ``expected_error``. ``front_door_raises`` satisfies that
    by its NAME, which is a coupling nobody would notice breaking -- rename it to
    ``assert_front_door`` and every module using it silently reads as having no region test at all.
    This is the test that would fail instead.
    """
    (tmp_path / "kernels").mkdir()
    (tmp_path / "perf").mkdir()
    (tmp_path / "perf" / "test_benchmark_perf_pkfd.py").write_text("")

    class _Fake:
        pass

    fake = _Fake()
    fake.M = KernelMatrix(
        "pkfd",
        axes=(
            Axis(
                "arg_fault",
                "a malformed tensor argument, or none",
                ("none", "w_dtype", "w_extent"),
                facets={
                    "bad_dtype": lambda v: v.endswith("_dtype"),
                    "bad_extent": lambda v: v.endswith("_extent"),
                },
            ),
        ),
        computes=_PROBE,
        unsupported=(
            Unsupported(
                where=lambda arg_fault: arg_fault != "none",
                raises=ValueError,
                match="b",
                reason="a malformed argument is refused at the front door",
            ),
        ),
    )
    mod = tmp_path / "kernels" / "test_pkfd.py"
    mod.write_text(
        '@M.parametrize_unsupported("arg_fault")\n'
        "def test_u(arg_fault, expected_error, expected_match):\n"
        "    with front_door_raises(expected_error, expected_match):\n"
        "        call(arg_fault)\n"
    )
    problems = [p for p in audit_test_module(mod, fake) if "unsupported region" in p]
    assert not problems, (
        f"front_door_raises must satisfy the region-test check; it no longer does: {problems}"
    )


# ── computes= and the input properties it implies ──────────────────────────────────────────────


@matrix_exempt("tests the matrix machinery itself, which has no kernel shape axes")
def test_computes_is_required_and_sayable_as_nothing():
    """A matrix must state what its subject computes, because that is what implies the property.

    Shape coverage is not distribution coverage. ``torch.randn`` is zero-mean, which is the single
    row mean at which a padded-tail variance defect contributes exactly nothing -- so a pool can be
    wide in every extent and one point wide in the axis that would have caught the bug. Making the
    field mandatory turns "nobody considered the distribution" into "nobody wrote it down".
    """
    ax = (Axis("N", "any int", (8, 9)),)
    unsup = no_unsupported(because="every N computes")
    with pytest.raises(ValueError, match="`computes=` is REQUIRED"):
        KernelMatrix(kernel="probe_comp_missing", axes=ax, unsupported=unsup)
    with pytest.raises(ValueError, match="`computes=` is REQUIRED"):
        KernelMatrix(kernel="probe_comp_empty", axes=ax, unsupported=unsup, computes=())
    with pytest.raises(ValueError, match="requires because"):
        computes_nothing_numeric(because="   ")
    KernelMatrix(
        kernel="probe_comp_none",
        axes=ax,
        unsupported=unsup,
        computes=computes_nothing_numeric(because="returns a config; launches nothing"),
    )


@matrix_exempt("tests the matrix machinery itself, which has no kernel shape axes")
def test_an_undeclared_trait_is_refused_rather_than_ignored():
    """A trait not in :data:`SENSITIVITY` implies nothing, so accepting it would be silent coverage.

    This is the failure mode a free-text field would have: ``computes=("layernormish",)`` would look
    like a declaration and require nothing at all.
    """
    with pytest.raises(ValueError, match="unknown computation trait"):
        KernelMatrix(
            kernel="probe_comp_unknown",
            axes=(Axis("N", "d", (8, 9)),),
            unsupported=no_unsupported(because="none"),
            computes=("layernormish",),
        )


@matrix_exempt("tests the matrix machinery itself, which has no kernel shape axes")
def test_a_required_property_must_be_an_axis_or_a_waiver_with_a_reason():
    """``row_reduction`` obliges the pool to sample ``row_mean`` -- as an axis, or explicitly not.

    All three states are checked in one place because it is the CONTRAST that matters: absent is
    refused, an axis satisfies it, and a waiver satisfies it only while it carries prose. An empty
    waiver is treated as absent, so ``property_waivers={"row_mean": ""}`` cannot be used to make the
    error go away without saying anything.
    """
    unsup = no_unsupported(because="none")
    with pytest.raises(ValueError, match="requires sampling 'row_mean'"):
        KernelMatrix(
            kernel="probe_prop_missing",
            axes=(Axis("N", "d", (8, 9)),),
            unsupported=unsup,
            computes=("row_reduction",),
        )
    with pytest.raises(ValueError, match="requires sampling 'row_mean'"):
        KernelMatrix(
            kernel="probe_prop_blank",
            axes=(Axis("N", "d", (8, 9)),),
            unsupported=unsup,
            computes=("row_reduction",),
            property_waivers={"row_mean": "   "},
        )
    KernelMatrix(
        kernel="probe_prop_axis",
        axes=(Axis("N", "d", (8, 9)), Axis("row_mean", "the row offset", (0.0, 100.0))),
        unsupported=unsup,
        computes=("row_reduction",),
    )
    KernelMatrix(
        kernel="probe_prop_waived",
        axes=(Axis("N", "d", (8, 9)),),
        unsupported=unsup,
        computes=("row_reduction",),
        property_waivers={
            "row_mean": "measured insensitive: worst ratio 0.34 -> 0.73 to mu/sd 1182"
        },
    )


@matrix_exempt("tests the matrix machinery itself, which has no kernel shape axes")
def test_a_waiver_for_a_property_nothing_requires_is_refused():
    """A waiver reads as "we considered this and measured it away", so a stray one is a false claim."""
    with pytest.raises(ValueError, match="which nothing requires"):
        KernelMatrix(
            kernel="probe_prop_stray",
            axes=(Axis("N", "d", (8, 9)),),
            unsupported=no_unsupported(because="none"),
            computes=("contraction",),
            property_waivers={"row_mean": "not applicable"},
        )


@matrix_exempt("tests the matrix machinery itself, which has no kernel shape axes")
def test_required_properties_are_the_union_over_traits_without_duplicates():
    """Two traits implying the same property must oblige it once, not twice.

    Trivial arithmetic, but a duplicate would make the "stray waiver" check above reject a legal
    waiver, which is the kind of interaction that only shows up once somebody hits it.
    """
    m = KernelMatrix(
        kernel="probe_prop_union",
        axes=(Axis("N", "d", (8, 9)), Axis("row_mean", "the row offset", (0.0, 100.0))),
        unsupported=no_unsupported(because="none"),
        computes=("row_reduction", "contraction", "data_movement"),
    )
    assert m.required_properties() == ("row_mean",)
    none = KernelMatrix(
        kernel="probe_prop_union_none",
        axes=(Axis("N", "d", (8, 9)),),
        unsupported=no_unsupported(because="none"),
        computes=computes_nothing_numeric(because="launches nothing"),
    )
    assert none.required_properties() == ()


@matrix_exempt("walks every declared matrix rather than one kernel's shapes")
def test_every_waived_input_property_in_the_repo_is_one_of_the_expected_ones():
    """The waiver set is PINNED, so widening it is a visible edit with a reason attached.

    A waiver switches a requirement off. Without this, the mechanism degrades exactly the way an
    un-pinned exemption list always does: each new waiver checks one property less, and a matrix
    where nothing is required looks identical to one where everything is sampled.
    """
    waived = {
        (m.kernel, prop): reason
        for m in _ALL_MATRICES.values()
        for prop, reason in m.property_waivers.items()
    }
    assert set(waived) == {
        # The ONLY waiver in the repo, and it is structural rather than measured: the row this
        # kernel normalizes is the token-pair einsum's OUTPUT, so no caller-side value sets its
        # mean. Every other kernel that computes a row reduction declares a `row_mean` AXIS --
        # including the two that were candidates for a measured waiver, because measuring one and
        # then waiving it leaves nothing running to notice when the measurement stops holding.
        ("gemm_layernorm_gemm", "row_mean"),
    }, f"unexpected input-property waivers: {sorted(waived)}"
    for key, reason in waived.items():
        assert "measured" in reason.lower(), (
            f"{key} is waived without naming its evidence: {reason!r}. A waiver is a MEASURABLE "
            f"claim; say what was swept and what the result was."
        )


def test_a_stacked_parametrize_that_emits_an_unsupported_cell_is_caught():
    """The per-item check catches a forbidden cell `parametrize` structurally cannot see.

    `KernelMatrix.parametrize` evaluates a region only when ONE call parametrizes every axis that
    region reads, because one call is all it is given. Separate decorators are CROSSED by pytest,
    so a region reading two axes is skipped by each call in turn and the forbidden pair is emitted
    anyway -- present, correct-looking, never firing. Measured in a real port: four declared-
    unsupported cells sat in an accepting test's grid that way.

    `unsupported_cell_problem` reads the ACTUAL bound cell instead, which is why it does not care
    how the cell was produced.

    The three cases are one positive control and two negatives, and all three are load-bearing:
    a guard that has never fired proves nothing, and a guard that fires on everything is worse than
    none. The third is the subtle one -- `parametrize_unsupported` EXISTS to emit region cells, so
    flagging those would fail the very test that discharges the region.
    """
    m = _matrix(
        name="probe_stacked_region",
        axes=(
            Axis(
                name="n",
                domain="token extent",
                values=(8, 12),
                facets={"even8": lambda v: v % 8 == 0},
            ),
            Axis(
                name="split", domain="mesh split", values=(1, 2), facets={"flat": lambda v: v == 1}
            ),
        ),
        unsupported=(
            Unsupported(
                where=lambda n, split: split > 1 and n % 8 != 0,
                raises=ValueError,
                match=r"not 16-B aligned",
                reason="a split extent must stay 8-aligned",
            ),
        ),
    )
    mod = types.SimpleNamespace(MATRIX=m)

    problem = unsupported_cell_problem(mod, {"n": 12, "split": 2})
    assert problem is not None, (
        "the forbidden cell (n=12, split=2) was NOT reported. This is the positive control: "
        "without it a passing check cannot be told from one that never evaluates anything."
    )
    assert "probe_stacked_region" in problem and "8-aligned" in problem, (
        f"the report must name the kernel and quote the region's reason so the failure is "
        f"actionable without opening the matrix; got {problem!r}"
    )
    assert unsupported_cell_problem(mod, {"n": 8, "split": 2}) is None, (
        "a legal cell was flagged; a guard that fires on everything is worse than none"
    )
    assert (
        unsupported_cell_problem(mod, {"n": 12, "split": 2, "expected_error": ValueError}) is None
    ), (
        "a parametrize_unsupported cell was flagged. Those exist to land in a region and assert "
        "the raise, so flagging them fails the very test that discharges the region."
    )


def _region_probe(name, *, with_region):
    """A two-axis probe matrix, optionally carrying the (n, split) 16-B region.

    Args:
        name: Kernel name. Distinct per call -- registering two different matrices under one name
            is itself an error the machinery raises, so a shared name would fail at construction
            rather than at the assertion under test.
        with_region: True to declare the ``split > 1 and n % 8 != 0`` region; False for a matrix
            that declares none, standing in for a sibling matrix a gate merely IMPORTS.

    Returns:
        The matrix. Not registered anywhere the diversity guard walks, so it needs no cleanup.
    """
    return KernelMatrix(
        kernel=name,
        axes=(
            Axis(
                name="n",
                domain="token extent",
                values=(8, 12),
                facets={"even8": lambda v: v % 8 == 0},
            ),
            Axis(
                name="split", domain="mesh split", values=(1, 2), facets={"flat": lambda v: v == 1}
            ),
        ),
        computes=_PROBE,
        unsupported=(
            (
                Unsupported(
                    where=lambda n, split: split > 1 and n % 8 != 0,
                    raises=ValueError,
                    match=r"not 16-B aligned",
                    reason="a split extent must stay 8-aligned",
                ),
            )
            if with_region
            else no_unsupported(because="probe matrix; declares no regions")
        ),
    )


def test_the_unsupported_cell_check_reads_every_matrix_a_module_holds_not_only_a_lone_one():
    """A module holding TWO matrices is still checked -- it used to be a silent no-op.

    **The bug this pins.** The check began ``if len(matrices) != 1: return None``. A COUNT is not
    the question the check asks; the question is "does ANY matrix in scope declare this cell
    unsupported?". Any module holding two matrices therefore disabled the guard entirely, and a
    perf gate is exactly that shape once it draws some axes from its own kernel's matrix and others
    from a sibling's -- which the back A2A gate does, holding ``A2A_KERNEL`` and ``EPI``. Nothing
    was wrong with that gate; what was missing is the machinery that would have caught it if the
    hand-written skip beside it had been wrong. Same shape as the ``_owned_matrices`` fix, which
    likewise replaced a count with a question.

    **Why the second and third cases are not padding.** A guard that fires on everything is worse
    than none, and a fix that only ever adds reports cannot be told from one that broke the legal
    path. The ``expected_error`` case is the subtle one and is the reason the exemption must be
    checked BEFORE the matrices are gathered: ``parametrize_unsupported`` exists to emit region
    cells, so flagging those would fail the very test that discharges the region.

    **The second-position case is the one a naive fix passes by accident.** Iterating and returning
    the first hit looks correct when the owning matrix happens to come first in ``vars()``; the
    region has to fire when it is the SECOND matrix that owns it, which is also the real gate's
    layout.
    """
    owns = _region_probe("probe_multi_owner", with_region=True)
    plain = _region_probe("probe_multi_sibling", with_region=False)
    forbidden = {"n": 12, "split": 2}

    one = types.SimpleNamespace(MATRIX=owns)
    first = types.SimpleNamespace(MATRIX=owns, OTHER=plain)
    second = types.SimpleNamespace(OTHER=plain, MATRIX=owns)

    baseline = unsupported_cell_problem(one, forbidden)
    assert baseline is not None, (
        "the single-matrix path stopped reporting a forbidden cell -- this is the REGRESSION "
        "control, and without it a two-matrix fix could pass while breaking the case that worked"
    )
    for label, mod in (("region-owner first", first), ("region-owner second", second)):
        problem = unsupported_cell_problem(mod, forbidden)
        assert problem is not None, (
            f"a module holding TWO matrices ({label}) did NOT report the forbidden cell "
            f"{forbidden}. The check bailed on the matrix COUNT instead of asking whether any "
            f"matrix in scope declares the cell unsupported, so the guard was a silent no-op on "
            f"every perf gate that imports a sibling's matrix."
        )
        assert "probe_multi_owner" in problem and "8-aligned" in problem, (
            f"the report must name the matrix that OWNS the region and quote its reason, so the "
            f"failure is actionable without opening either matrix; got {problem!r}"
        )

    assert unsupported_cell_problem(first, {"n": 8, "split": 2}) is None, (
        "a LEGAL cell was flagged in a two-matrix module; widening the scan must not widen what "
        "counts as forbidden"
    )
    assert unsupported_cell_problem(first, {**forbidden, "expected_error": ValueError}) is None, (
        "a parametrize_unsupported cell was flagged in a two-matrix module. That exemption has to "
        "be applied BEFORE the matrices are gathered, or widening the scan re-breaks the tests "
        "whose whole job is to assert the refusal."
    )
    assert unsupported_cell_problem(types.SimpleNamespace(), forbidden) is None, (
        "a module with NO matrix was flagged; there is nothing to check and nothing to report"
    )

    aliased = types.SimpleNamespace(MATRIX=owns, ALIAS=owns)
    assert unsupported_cell_problem(aliased, forbidden) == baseline, (
        "one matrix bound to two names produced a different report than the same matrix bound to "
        "one. It is ONE matrix; deduping by identity is what keeps the message from depending on "
        "which name `vars()` happens to yield first."
    )


@matrix_exempt("tests the matrix machinery itself, which has no kernel shape axes")
def test_a_module_level_pytestmark_exempts_every_test_but_only_with_a_literal_reason(tmp_path):
    """A module-level ``pytestmark = matrix_exempt("...")`` exempts the whole module.

    **Why the module-level form exists.** A wholesale-ported test module can have a subject that is
    not a kernel at all -- the harness driver's cell protocol launches nothing and has no shape axis
    -- so the same sentence would otherwise be repeated once per test. Repeating it 45 times does not
    make a reader likelier to disagree with it; it makes the file harder to diff against the tree it
    was ported from, which is a cost with no matching benefit.

    **What must NOT weaken.** The escape hatch is only as good as the two things it still refuses, so
    both are asserted here rather than left to the reader:

    1. a module with NO exemption at all is still flagged, exactly as before; and
    2. a non-literal reason does not merely warn -- it fails to exempt, so every test in the module
       is reported as well. The AST is what is read, so a name or an f-string is no reason, and
       granting the exemption anyway would let a module opt out with a reason nobody can see.
    """
    good = tmp_path / "test_probe_good.py"
    good.write_text(
        "from fold_cp_ops.testing.kernel_matrix import matrix_exempt\n"
        'pytestmark = matrix_exempt("subject is not a kernel; no shape axis exists here")\n'
        "def test_a():\n    pass\n"
    )
    assert audit_test_module(good) == [], audit_test_module(good)

    bare = tmp_path / "test_probe_bare.py"
    bare.write_text("def test_a():\n    pass\n")
    assert any("neither parametrizes" in p for p in audit_test_module(bare)), (
        "a module with no exemption must still be flagged -- the module-level form must not turn "
        "the check off for everyone"
    )

    named = tmp_path / "test_probe_named.py"
    named.write_text(
        "from fold_cp_ops.testing.kernel_matrix import matrix_exempt\n"
        '_R = "a name, not a literal"\n'
        "pytestmark = matrix_exempt(_R)\n"
        "def test_a():\n    pass\n"
    )
    problems = audit_test_module(named)
    assert any("LITERAL reason" in p for p in problems), problems
    assert any("neither parametrizes" in p for p in problems), (
        "a non-literal reason must NOT exempt the module -- otherwise a module opts out with a "
        "reason no reader can see"
    )


# ── the coverage scope: OWNERSHIP, not a directory ────────────────────────────────────────────
#: Source of a probe module that OWNS a matrix. ``{kernel}`` names the matrix, ``{swept}`` is the
#: argument list the one test draws -- ``"a"`` leaves axis ``b`` declared and swept by nothing,
#: ``"a", "b"`` sweeps both. Written to disk and IMPORTED for real rather than faked with a stub
#: object, because the check under test reads the module's AST AND its live attributes: a stub whose
#: attribute names merely happened to match would pass while the correspondence it asserts is
#: fictional.
_OWNED_MATRIX_MODULE = """\
from fold_cp_ops.testing.kernel_matrix import (
    Axis,
    KernelMatrix,
    computes_nothing_numeric,
    no_unsupported,
)

PROBE = KernelMatrix(
    kernel="{kernel}",
    axes=(Axis("a", "probe axis", (1, 2)), Axis("b", "probe axis", (1, 2))),
    unsupported=no_unsupported(because="probe module for the machinery; not a kernel"),
    computes=computes_nothing_numeric(because="probe module; launches nothing"),
)


@PROBE.parametrize({swept})
def test_probe({args}):
    pass
"""


def _import_probe_module(path):
    """Import a probe module from an arbitrary path, without touching ``sys.path``.

    Args:
        path: A ``pathlib.Path`` to a written ``.py`` file. Its parent need not be a package and
            must NOT be on ``sys.path`` -- the point of the probe is that the module lives somewhere
            the normal collection rules would never reach.

    Returns:
        The imported module object, whose ``PROBE`` attribute is the live matrix its source
        constructs.
    """
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "sweeps_b,expect_problem", [(False, True), (True, False)], ids=["b-unswept", "b-swept"]
)
@matrix_exempt("tests the machinery's own scoping rule; it has no kernel shape axes")
def test_coverage_is_checked_on_a_matrix_owner_outside_any_kernels_directory(
    tmp_path, sweeps_b, expect_problem
):
    """A module that OWNS a matrix is coverage-checked wherever it lives -- here, not under ``kernels/``.

    **This is a test of the SPLIT, not of the audit, and the directory is what makes it one.**
    Coverage used to ride ``_is_kernel_module``, which requires a ``kernels/`` path PART; it is now
    scoped on :func:`_owned_matrices`, i.e. on whether the module CONSTRUCTS the matrix. The probe
    is therefore written under ``distributed/`` -- a directory the OLD predicate skipped outright.
    Every other ``tmp_path`` probe in this file uses ``tmp_path / "kernels"`` because the old rules
    required it, and a probe placed there would pass for the old reason and assert nothing about the
    change. **Reverting the split must make this test RED**, and that is the only property it has;
    verified by doing exactly that before it was committed.

    Both arms run because one alone proves half of it. ``b-unswept`` shows the check FIRES on a
    decorative axis; ``b-swept`` shows it goes SILENT when a test draws that axis, which is what
    keeps the first arm from being satisfiable by any always-on complaint.

    Args:
        tmp_path: pytest's per-test directory. The probe module is written under a ``distributed/``
            subdirectory of it, deliberately NOT ``kernels/``.
        sweeps_b: Whether the probe's one test draws axis ``b`` as well as ``a``.
        expect_problem: Whether ``audit_test_module`` must report the coverage problem naming ``b``.
    """
    kernel = f"probe_owner_{'swept' if sweeps_b else 'unswept'}"
    (tmp_path / "distributed").mkdir()
    path = tmp_path / "distributed" / f"test_{kernel}.py"
    path.write_text(
        _OWNED_MATRIX_MODULE.format(
            kernel=kernel,
            swept='"a", "b"' if sweeps_b else '"a"',
            args="a, b" if sweeps_b else "a",
        )
    )
    assert "kernels" not in path.parts, (
        f"the probe landed at {path}, which contains a 'kernels' component -- the OLD predicate "
        "would then cover it too and this test would pass without the split"
    )

    problems = audit_test_module(path, _import_probe_module(path))
    naming_b = [x for x in problems if "axis 'b'" in x]
    if expect_problem:
        assert naming_b, (
            f"axis 'b' is declared and swept by nothing, but audit_test_module reported no coverage "
            f"problem naming it. Coverage is not reaching a matrix owner outside kernels/. "
            f"Reported: {problems}"
        )
        assert "NO test parametrizes it" in naming_b[0], naming_b
    else:
        assert not problems, (
            f"the probe sweeps every axis it declares, so it must audit clean; got {problems}"
        )
