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

"""Tests for ``fold_cp_ops.testing.template_param_guard``.

A guard with no proof it DISCRIMINATES is decoration: one that flagged nothing and one that flagged
everything would both look green from here. So every refusal case below is paired with an
acceptance case, and the acceptance cases are not invented -- they are the eight attributes
``test_dual_gated_gemm_a2a.py`` legitimately sets post-construction, each verified to be read by the
kernel under that exact spelling.

GPU-free: the guard is an AST pass over source plus an introspection of already-imported classes.
"""

import pathlib
import types

import pytest

from fold_cp_ops.testing.template_param_guard import (
    functor_self_names,
    audit_template_param_module,
    declared_template_params,
    template_param_problems,
)

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]


class _Params:
    """Stand-in for a functor's construction-phase declaration."""

    @classmethod
    def field_names(cls):
        """The declared construction-phase parameter names."""
        return frozenset({"chunk_g", "n_dual_tiles", "gate3_n3", "pingpong"})


class _CallParams:
    """Stand-in for a functor's call-phase declaration."""

    @classmethod
    def field_names(cls):
        """The declared call-phase parameter names."""
        return frozenset({"epi_tile", "d_dtype"})


class _Functor:
    """A minimal functor exposing both declared phases, as the real kernels do."""

    Params = _Params
    CallParams = _CallParams


def _module_with_functor():
    """A fake imported test module whose namespace holds a functor class.

    Returns:
        A ``types.ModuleType`` with ``_Functor`` bound, which is all
        :func:`declared_template_params` reads.
    """
    m = types.ModuleType("fake_test_module")
    m._Functor = _Functor
    return m


def _source(tmp_path, body, name="test_probe.py"):
    """Write a probe module and return its path.

    Args:
        tmp_path: pytest's ``tmp_path``.
        body: Module source.
        name: File name; must start with ``test_`` so it reads like a real test module.

    Returns:
        The written ``pathlib.Path``.
    """
    p = tmp_path / name
    p.write_text(body)
    return p


# ── what counts as declared ────────────────────────────────────────────────────────────────────


def test_declared_names_come_from_the_classes_the_module_imported():
    """Both phases are collected, and from the module's OWN namespace.

    Scoping per module is what keeps the guard from flagging ``x.pingpong = True`` in a module that
    never imports a kernel -- a global union over the package could not tell the two apart.
    """
    assert declared_template_params(_module_with_functor()) == {
        "chunk_g",
        "n_dual_tiles",
        "gate3_n3",
        "pingpong",
        "epi_tile",
        "d_dtype",
    }


def test_a_module_importing_no_functor_declares_nothing():
    """The common case, and it must cost nothing rather than guess."""
    assert declared_template_params(types.ModuleType("empty")) == set()


def test_a_missing_module_is_a_no_op_rather_than_an_error():
    """Collection hands over modules that failed to import; the guard must not add a second error."""
    assert declared_template_params(None) == set()


# ── the FATAL half: assigning a declared parameter ─────────────────────────────────────────────


def test_assigning_a_declared_parameter_is_refused(tmp_path):
    """``g.chunk_g = 1`` -- the failure the 2-rank gate reported, caught statically."""
    path = _source(tmp_path, "def test_x():\n    g = build()\n    g.chunk_g = 1\n")
    problems = template_param_problems(path, _module_with_functor())
    assert len(problems) == 1, problems
    assert "chunk_g" in problems[0] and "DECLARED template parameter" in problems[0]
    assert "pass chunk_g= to the" in problems[0], "the message must name the fix"


def test_a_call_phase_parameter_is_refused_too(tmp_path):
    """Both phases are folded into the kernel, so a near-miss on either is equally inert."""
    path = _source(tmp_path, "def test_x():\n    g = build()\n    g.epi_tile = (128, 32)\n")
    problems = template_param_problems(path, _module_with_functor())
    assert len(problems) == 1 and "epi_tile" in problems[0]


# ── the INERT half: the near-miss, and the reason this guard exists ────────────────────────────


def test_an_underscore_near_miss_of_a_declared_parameter_is_refused(tmp_path):
    """``g._n_dual_tiles = 0`` SUCCEEDS at runtime and configures nothing. That is the defect.

    Measured in this repo: after ``g._n_dual_tiles = 99`` the real ``g.n_dual_tiles`` is still 0,
    because the kernel reads the declared name. The write is invisible until the value it failed to
    set differs from the default -- which is exactly when nobody is looking.
    """
    path = _source(tmp_path, "def test_x():\n    g = build()\n    g._n_dual_tiles = 0\n")
    problems = template_param_problems(path, _module_with_functor())
    assert len(problems) == 1, problems
    assert "NOTHING reads" in problems[0], "the message must say the write is inert"
    assert "`n_dual_tiles`" in problems[0], "the message must name the REAL parameter"


def test_every_declared_name_has_its_near_miss_covered(tmp_path):
    """The rule is derived from the declared set, not a hand-kept list of known-bad spellings."""
    body = "def test_x():\n    g = build()\n" + "".join(
        f"    g._{n} = 0\n" for n in sorted(declared_template_params(_module_with_functor()))
    )
    problems = template_param_problems(_source(tmp_path, body), _module_with_functor())
    assert len(problems) == 6, f"one per declared name, got {len(problems)}: {problems}"


# ── the ACCEPTANCE half: the eight attributes that are REAL configuration ──────────────────────

#: The attributes ``test_dual_gated_gemm_a2a.py`` legitimately sets after construction. Each was
#: checked against the kernel source to be read under THIS spelling -- e.g. ``self._has_gate3`` at
#: ``dual_gated_gemm.py:329,511``. They are the reason the rule keys on the declared set and its
#: underscore near-misses rather than on "any underscore attribute", which would flag all eight.
_LEGITIMATE = (
    "_a2a_cp_axis_sizes",
    "_a2a_dynamic",
    "_a2a_route2_ni",
    "_has_gate3",
)
# ``STATS_TPR``, ``_do_normalize``, ``_gemm_K`` and ``_stats_mode`` USED to sit in the tuple above,
# on the reading that they were real configuration read under that spelling. Re-measured on this
# tree: 8-10 writes each and ZERO reads, because the A2A front's parent here is the plain
# ``DualGatedGemmSm90`` rather than `main`'s LayerNorm-fused one -- there is no normalization to
# disable and no ``_gemm_K`` to override. They are pinned as FLAGGED below instead. Leaving them
# whitelisted would have kept the guard blessing four names that configure nothing.


@pytest.mark.parametrize("attr", _LEGITIMATE)
def test_a_legitimate_post_construction_attribute_is_accepted(tmp_path, attr):
    """None of the eight real ones is flagged. A rule that also flagged these would be worse.

    ``_has_gate3`` is the one that makes the point: it sits in the same block as the two inert
    writes, differs from them only in that the kernel really does read it, and no syntactic rule
    about leading underscores could tell them apart. The declared set can.
    """
    path = _source(tmp_path, f"def test_x():\n    g = build()\n    g.{attr} = 0\n")
    assert template_param_problems(path, _module_with_functor()) == []


def test_a_constructor_keyword_is_not_an_assignment(tmp_path):
    """Passing the parameter the RIGHT way must not be flagged -- that is the fix being recommended."""
    path = _source(
        tmp_path,
        "def test_x():\n    g = Functor(acc, a, tile, cluster, chunk_g=1, n_dual_tiles=0)\n",
    )
    assert template_param_problems(path, _module_with_functor()) == []


def test_a_read_is_not_an_assignment(tmp_path):
    """Reading a declared parameter is ordinary and common -- only writes are the defect."""
    path = _source(
        tmp_path,
        "def test_x():\n    g = build()\n    assert tuple(g.epi_tile) == (128, 32)\n    n = g.chunk_g\n",
    )
    assert template_param_problems(path, _module_with_functor()) == []


def test_a_module_that_imports_no_functor_is_never_walked(tmp_path):
    """With no declared names, even a source that spells the forbidden form is passed.

    This is the scoping decision made visible: the guard is about functors, and a module with none
    has nothing it could be getting wrong.
    """
    path = _source(tmp_path, "def test_x():\n    g = build()\n    g.chunk_g = 1\n")
    assert template_param_problems(path, types.ModuleType("no_functor")) == []
    assert audit_template_param_module(path, None) == []


# ── the repo-wide lock ─────────────────────────────────────────────────────────────────────────


def test_no_test_module_in_the_repo_assigns_a_template_parameter():
    """No module under ``tests/`` configures a functor by assigning to it after construction.

    The lock that keeps the guard honest in both directions, and the one that would have caught the
    ``_n_dual_tiles`` / ``_gate3_n3`` pair the day they were written. Modules are imported here
    because the declared names come from the classes they hold; an import failure is reported as
    such rather than silently skipping the file, since a skipped module is an unchecked one.
    """
    import importlib

    problems, checked = [], 0
    for path in sorted((_REPO_ROOT / "tests").rglob("test_*.py")):
        dotted = ".".join(path.relative_to(_REPO_ROOT).with_suffix("").parts)
        try:
            module = importlib.import_module(dotted)
        except Exception as exc:  # noqa: BLE001 - report, never skip silently
            problems.append(f"{path.name}: could not import to audit it ({exc!r})")
            continue
        checked += 1
        problems.extend(audit_template_param_module(path, module))
    assert checked, "no test modules were imported -- the lock would be vacuous"
    assert not problems, "template-parameter violations:\n  - " + "\n  - ".join(problems)


# ── the one carve-out, and the proof it is not a hole ──────────────────────────────────────────


def test_an_assignment_asserted_to_raise_is_accepted(tmp_path):
    """A write wrapped in ``pytest.raises`` is ASSERTING the refusal, not performing one.

    Three real sites do exactly this -- ``test_template_params.py`` twice and
    ``test_dual_gated_gemm.py`` once -- and each is the machinery's own proof that the functor
    refuses. A guard that flagged them would be refusing the tests that prove the thing it depends
    on.
    """
    path = _source(
        tmp_path,
        "import pytest\n\n\ndef test_x():\n    g = build()\n"
        '    with pytest.raises(AttributeError, match="immutable"):\n        g.chunk_g = 1\n',
    )
    assert template_param_problems(path, _module_with_functor()) == []


def test_the_raises_carve_out_does_not_leak_to_the_rest_of_the_test(tmp_path):
    """Only statements INSIDE the block are exempt; a real write elsewhere still fires.

    This is what stops the carve-out being an escape hatch. It is narrower than a module-level
    exemption by construction, and narrower still than it looks: a write that does NOT raise fails
    its own enclosing ``pytest.raises`` with "DID NOT RAISE", so the block cannot be used to smuggle
    a working configuration write past the guard.
    """
    path = _source(
        tmp_path,
        "import pytest\n\n\ndef test_x():\n    g = build()\n"
        "    with pytest.raises(AttributeError):\n        g.chunk_g = 1\n"
        "    g._n_dual_tiles = 0\n",
    )
    problems = template_param_problems(path, _module_with_functor())
    assert len(problems) == 1, problems
    assert "_n_dual_tiles" in problems[0] and "NOTHING reads" in problems[0]


# ── the dynamic spelling, added because it is the natural workaround to this guard's own error ──


def test_setattr_with_a_literal_name_is_refused(tmp_path):
    """``setattr(g, "chunk_g", 1)`` is the SAME write and must not be a way around the rule.

    Not added because it has occurred -- it had not. It is added because it is precisely what
    somebody reaches for when attribute assignment starts raising, which is exactly what this guard
    causes. A rule steppable by the most natural response to its own error message has a hole aimed
    at its own users.
    """
    path = _source(tmp_path, 'def test_x():\n    g = build()\n    setattr(g, "chunk_g", 1)\n')
    problems = template_param_problems(path, _module_with_functor())
    assert len(problems) == 1 and "chunk_g" in problems[0]


def test_setattr_reaches_the_inert_near_miss_too(tmp_path):
    """The dynamic spelling gets the full rule, not just the fatal half."""
    path = _source(tmp_path, 'def test_x():\n    g = build()\n    setattr(g, "_n_dual_tiles", 0)\n')
    problems = template_param_problems(path, _module_with_functor())
    assert len(problems) == 1 and "NOTHING reads" in problems[0]


def test_setattr_with_a_computed_name_is_left_alone(tmp_path):
    """Only a LITERAL name is resolvable; a computed one is not guessed at.

    The safe direction: a missed finding rather than an invented one. Flagging every three-argument
    ``setattr`` would fire on ordinary reflective code that has nothing to do with a functor.
    """
    path = _source(tmp_path, "def test_x():\n    g = build()\n    setattr(g, name, 1)\n")
    assert template_param_problems(path, _module_with_functor()) == []


def test_a_setattr_asserted_to_raise_is_accepted(tmp_path):
    """The carve-out covers the dynamic spelling too, or it would protect one form and refuse the other."""
    path = _source(
        tmp_path,
        "import pytest\n\n\ndef test_x():\n    g = build()\n"
        '    with pytest.raises(AttributeError):\n        setattr(g, "chunk_g", 1)\n',
    )
    assert template_param_problems(path, _module_with_functor()) == []


def test_a_read_only_property_is_refused_naming_where_it_is_declared(tmp_path):
    """The FOURTH case: the name IS the one the kernel reads, and is still not assignable.

    ``_has_gate3`` is read at ``dual_gated_gemm.py:329,511`` under exactly that spelling, and is a
    ``@property`` with no setter derived as ``self.n_dual_tiles > 0``. It passes every name test and
    the assignment still raises -- which is why the discriminator is SETTABILITY, not name-matching.
    A three-way taxonomy of declared / near-miss / real misses it, and did.
    """

    class _WithProperty(_Functor):
        @property
        def _derived(self):
            """Read-only, like the real `_has_gate3`."""
            return True

    m = types.ModuleType("fake")
    m._WithProperty = _WithProperty
    path = _source(tmp_path, "def test_x():\n    g = build()\n    g._derived = False\n")
    problems = template_param_problems(path, m)
    assert len(problems) == 1, problems
    assert "READ-ONLY property" in problems[0] and "_WithProperty._derived" in problems[0]
    assert "DERIVED" in problems[0], "the message must say to arrange the inputs instead"


# ── the FIFTH case: an attribute this tree's kernel never mentions ─────────────────────────────


def test_an_attribute_the_functor_never_mentions_is_flagged(tmp_path):
    """The bring-back residue: a write that was load-bearing against a DIFFERENT parent.

    ``_do_normalize`` is the measured instance. Under `main`'s LayerNorm-fused parent it disables a
    normalization; on the extraction branch the parent is the plain dual, which has none, so the
    write creates an attribute nobody reads and the test believes it configured something. Neither
    the declared-name test nor the near-miss test can see it -- the name matches nothing precisely
    because nothing is there -- which is why the discriminator has to be "does the kernel mention
    this at all" rather than "does this name resemble a parameter".
    """
    body = "def test_x():\n    g = _Functor()\n    g._do_normalize = False\n"
    problems = template_param_problems(_source(tmp_path, body), _module_with_functor())
    assert len(problems) == 1, f"expected exactly one finding, got {problems}"
    assert "NEVER MENTIONS" in problems[0], problems[0]
    assert "_do_normalize" in problems[0]


@pytest.mark.parametrize("attr", ("_do_normalize", "_stats_mode", "_gemm_K", "STATS_TPR"))
def test_the_four_measured_inert_front_attributes_are_flagged(tmp_path, attr):
    """All four names the front-A2A builders carried over, pinned individually.

    They were deleted from ``test_dual_gated_gemm_a2a.py`` (32 writes, zero reads) once the parent
    question was settled in favour of the plain ``DualGatedGemmSm90``. Pinning them here is what
    stops the next port re-introducing the class silently: a re-added write fails at the guard
    rather than passing as configuration nobody applied.
    """
    body = f"def test_x():\n    g = _Functor()\n    g.{attr} = 1\n"
    problems = template_param_problems(_source(tmp_path, body), _module_with_functor())
    assert len(problems) == 1 and "NEVER MENTIONS" in problems[0], problems


def test_an_attribute_the_test_reads_back_is_not_flagged(tmp_path):
    """Writing an undeclared attribute AND asserting it is the point of some tests.

    ``tests/_internal/compile_time/test_template_params.py`` proves an undeclared name stays
    settable, which requires writing one. The finding is about a write nobody observes, so a write
    the file observes is by construction not it. The exception is tied to evidence in the source
    rather than to a pragma, because a pragma is copied and evidence is not.
    """
    body = (
        "def test_x():\n    g = _Functor()\n    g._some_cache = 5\n    assert g._some_cache == 5\n"
    )
    assert template_param_problems(_source(tmp_path, body), _module_with_functor()) == []


def test_an_attribute_on_something_that_is_not_a_functor_is_not_flagged(tmp_path):
    """Scope: only names bound to a functor CONSTRUCTOR are subject to this finding.

    Without the restriction the check fires on every attribute assignment in the file -- measured
    on this tree, 7 findings, all correct about the name and all wrong about the subject: a
    monkeypatched class method, attributes stuffed onto a module object by a fixture, and locals
    that were never kernels. Each would have sent a reader to delete a working line.
    """
    body = (
        "def test_x():\n"
        "    other = helper()\n"
        "    other._do_normalize = False\n"
        "    Cls.method = lambda self: None\n"
    )
    assert template_param_problems(_source(tmp_path, body), _module_with_functor()) == []


def test_the_finding_is_skipped_when_the_functor_source_cannot_be_read(tmp_path):
    """Fail-SAFE: unknown means silent, never "flag everything".

    `functor_self_names` returns None when any class in an MRO has no retrievable source, and the
    caller then skips this finding entirely. A false positive here costs a developer deleting a
    line that was doing real work, which is worse than the silence the guard is trying to end. The
    probe uses a dynamically built class, whose source `inspect.getsource` cannot recover.
    """
    dynamic = types.new_class("_DynamicFunctor")
    dynamic.Params = _Params
    dynamic.CallParams = _CallParams
    m = types.ModuleType("fake_dynamic_module")
    m._Functor = dynamic
    assert functor_self_names(m) is None
    body = "def test_x():\n    g = _Functor()\n    g._do_normalize = False\n"
    assert template_param_problems(_source(tmp_path, body), m) == []
