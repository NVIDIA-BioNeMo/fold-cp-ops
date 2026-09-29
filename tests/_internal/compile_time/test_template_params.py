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

"""Unit tests for ``_internal/compile_time/template_params.py``.

The last test in this file is the one with teeth: it walks the AST of every shipped kernel functor
and fails if a ``@cute.jit`` / ``@cute.kernel`` method reads a ``self`` attribute that is not a
declared compile-time parameter. That is what locks future kernels into the paradigm instead of
relying on everyone remembering it.
"""

import ast
import functools
import pathlib
from typing import Optional, Type

import pytest
import torch

import cutlass

from fold_cp_ops._internal.compile_time.template_params import (
    canonical_key_bytes,
    StaticTypes,
    TemplateParams,
    TemplateParamsMixin,
    is_compile_time_value,
)


class _Params(TemplateParams):
    """Minimal pack used by the tests below."""

    dtype: Type[cutlass.Numeric]
    N: int
    flag: bool = False
    mode: Optional[str] = None


class _Functor(TemplateParamsMixin):
    """Minimal functor binding :class:`_Params`."""

    Params = _Params

    def __init__(self, **kw):
        self._bind_params(**kw)


# ── the classifier ────────────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize(
    "value,expected",
    [
        (1, True),
        (True, True),
        ("smem", True),
        (1.5, True),
        (None, True),
        (cutlass.BFloat16, True),  # a cutlass numeric TYPE, via NumericMeta
        ((128, 4), True),  # tuple of constants — shapes must be admissible
        ((128, None), True),
    ],
)
def test_is_compile_time_value_accepts_constants(value, expected):
    """Scalars, numeric types, None and tuples of those are foldable into a kernel."""
    assert is_compile_time_value(value) is expected


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a tensor to reject")
def test_is_compile_time_value_rejects_runtime_values():
    """A tensor is not foldable — and neither is a tuple that merely CONTAINS one.

    The tuple case is the subtle one: ``isinstance((tensor, 4), StaticTypes)`` is False only
    because ``tuple`` is absent from ``StaticTypes``; once tuples are admitted for shapes, a
    non-recursive check would wave a tensor straight through inside one.
    """
    x = torch.zeros(4, device="cuda")
    assert is_compile_time_value(x) is False
    assert is_compile_time_value((128, x)) is False
    # Tuples are handled by RECURSION, not by isinstance -- `tuple` is deliberately absent from
    # StaticTypes, so a non-recursive check would reject (128, 4) outright and, if tuple were
    # simply added, would wave a tensor through inside one.
    assert tuple not in StaticTypes


def test_an_enum_member_is_a_compile_time_value():
    """``LayoutEnum`` and ``RoundingMode`` are foldable, and that is required, not convenient.

    A GEMM's operand major mode decides the WGMMA atom and every SMEM swizzle -- it is as much a
    template parameter as the tile shape. ``cutlass.utils.LayoutEnum`` is a plain ``Enum``, so
    without ``enum.Enum`` in ``StaticTypes`` the four major modes could not be declared and the
    functor's compile-time surface would be incomplete by exactly the fields that matter most to
    the compiled artifact.

    ``RoundingMode`` is an ``IntEnum`` and was already admitted by ``int``; it is checked here so
    that "enums are admissible" is not silently resting on that coincidence.
    """
    import enum

    from cutlass.utils import LayoutEnum

    from fold_cp_ops._internal.rounding import RoundingMode

    assert enum.Enum in StaticTypes
    assert is_compile_time_value(LayoutEnum.ROW_MAJOR) is True
    assert is_compile_time_value(RoundingMode.RN) is True
    assert is_compile_time_value((LayoutEnum.COL_MAJOR, 128)) is True


def test_an_enum_field_survives_the_pack_round_trip():
    """A declared enum field reads back as the SAME member, not a copy.

    ``param_dict``/``compile_key`` go through ``dataclasses.asdict``, which deep-copies non-dataclass
    values. Identity matters because every downstream branch is an ``is``/``==`` against the module
    singleton; a copied member would compare unequal and silently take the other branch.
    """
    from cutlass.utils import LayoutEnum

    class P(TemplateParams):
        """A pack whose one field is an enum member."""

        major: LayoutEnum = LayoutEnum.ROW_MAJOR

    class K(TemplateParamsMixin):
        """A functor binding it."""

        Params = P

        def __init__(self, **kw):
            self._bind_params(**kw)

    k = K(major=LayoutEnum.COL_MAJOR)
    assert k.major is LayoutEnum.COL_MAJOR
    assert k.param_dict()["major"] is LayoutEnum.COL_MAJOR


# ── the pack ──────────────────────────────────────────────────────────────────────────────────
def test_subclass_is_frozen_and_keyword_only():
    """Subclasses become frozen kw-only dataclasses without repeating the decorator.

    Keyword-only is not cosmetic: a base with defaulted fields followed by a subclass with required
    ones is a ``TypeError`` for an ordinary dataclass, which would make ``ReductionParams`` ->
    ``LayerNormParams`` impossible to express.
    """
    p = _Params(dtype=cutlass.BFloat16, N=1024)
    assert p.N == 1024 and p.flag is False and p.mode is None
    with pytest.raises(Exception):  # FrozenInstanceError (a subclass of AttributeError)
        p.N = 2048
    with pytest.raises(TypeError):
        _Params(cutlass.BFloat16, 1024)  # positional — rejected


def test_declared_fields_are_reported():
    """``field_names`` reports the declared pack, including inherited fields."""
    assert _Params.field_names() == frozenset({"dtype", "N", "flag", "mode"})
    assert TemplateParams.field_names() == frozenset()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a tensor to reject")
def test_runtime_value_is_rejected_naming_the_field():
    """Construction fails, and the message names the offending field and its type."""
    x = torch.zeros(4, device="cuda")
    with pytest.raises(TypeError, match=r"RUNTIME values.*N.*Tensor"):
        _Params(dtype=cutlass.BFloat16, N=x)


# ── the mixin ─────────────────────────────────────────────────────────────────────────────────
def test_params_read_back_as_plain_attributes():
    """Binding exposes each field as ``self.<name>`` so call sites stay ergonomic."""
    f = _Functor(dtype=cutlass.BFloat16, N=1024, flag=True, mode="smem")
    assert (f.dtype, f.N, f.flag, f.mode) == (cutlass.BFloat16, 1024, True, "smem")
    assert f.param_dict() == {"dtype": cutlass.BFloat16, "N": 1024, "flag": True, "mode": "smem"}


def test_declared_params_are_immutable_but_other_attributes_are_not():
    """Only declared parameters are protected; ordinary attributes stay writable.

    Deliberate. The guarantee is about values that reach the kernel as compile-time constants.
    Freezing the whole instance also breaks subclass ``__init__`` ordering in ways that are tedious
    to work around, which is measured -- the first attempt at this did exactly that.
    """
    f = _Functor(dtype=cutlass.BFloat16, N=1024)
    with pytest.raises(AttributeError, match="immutable after construction"):
        f.N = 2048
    f._some_cache = 5  # not declared -> allowed
    assert f._some_cache == 5


def test_incomplete_or_unknown_params_are_refused():
    """A missing required field or an unknown keyword is a TypeError from the pack constructor."""
    with pytest.raises(TypeError):
        _Functor(dtype=cutlass.BFloat16)  # N missing
    with pytest.raises(TypeError):
        _Functor(dtype=cutlass.BFloat16, N=8, nope=1)  # unknown


# ── the guard: kernels may read only declared parameters ──────────────────────────────────────
# Anchored to THIS FILE, not to the working directory. A CWD-relative glob here is not a style
# nit: run pytest from anywhere but the repo root and it matches nothing, `parametrize` gets an
# empty list, and the guard below silently collapses to a single skipped placeholder -- it reports
# as a pass, protects nothing, and the test count barely moves (measured: 886 -> 885 collected).
# The assert is the other half: an empty list must fail loudly rather than vanish.
_REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
_KERNEL_MODULES = sorted((_REPO_ROOT / "fold_cp_ops" / "kernels").glob("*.py"))
assert _KERNEL_MODULES, (
    f"no kernel modules found under {_REPO_ROOT / 'fold_cp_ops' / 'kernels'} -- the paradigm guard "
    f"below would be vacuous. Did the tree move relative to this file?"
)


def _traced_methods(tree: ast.AST):
    """Yield ``(class_name, func_node)`` for every ``@cute.jit`` / ``@cute.kernel`` method.

    Args:
        tree: A parsed module AST.

    Yields:
        Pairs of the enclosing class name and the decorated function node. Module-level decorated
        functions are skipped -- they have no ``self`` and so cannot commit this mistake.
    """
    for cls in [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]:
        for fn in [n for n in cls.body if isinstance(n, ast.FunctionDef)]:
            decs = " ".join(ast.unparse(d) for d in fn.decorator_list)
            if "cute.jit" in decs or "cute.kernel" in decs:
                yield cls.name, fn


def _readable_names(functor) -> set:
    """The ``self.X`` names a traced method of ``functor`` may legitimately read.

    Purpose
        Encodes the paradigm's actual rule, which is broader than "is a declared field" and much
        narrower than "is any attribute".

    Semantics
        Four admissible classes, each safe for a different reason:

        * fields of ``Params`` -- bound in ``__init__``, frozen;
        * fields of ``CallParams`` -- bound at the top of ``__call__``, frozen;
        * ``property`` / ``cached_property`` anywhere in the MRO -- DERIVED from the packs, so they
          cannot desynchronize from what they were derived from;
        * class attributes holding a compile-time value -- shared constants rather than per-instance
          state, and foldable by the same test the packs apply to their own fields.

        Callables are handled by the caller: a method is a pure function of the bound parameters
        evaluated at trace time, not stored state.

    Args:
        functor: The kernel functor CLASS, not an instance. Must expose ``Params``; ``CallParams``
            is optional and defaults to the empty pack, which is the single-phase case.

    Returns:
        The set of admissible attribute names.
    """
    names = set(functor.Params.field_names())
    names |= set(getattr(functor, "CallParams", TemplateParams).field_names())
    for klass in functor.__mro__:
        for name, member in vars(klass).items():
            if isinstance(member, (property, functools.cached_property)):
                names.add(name)
            elif not name.startswith("__") and is_compile_time_value(member):
                names.add(name)
    return names


def test_readable_names_admits_the_three_classes_and_nothing_else():
    """The guard's admission rule, checked directly rather than only through the kernels it scans.

    Without this, a bug in `_readable_names` that admitted everything would make the guard vacuous
    while it kept reporting green -- the same failure shape as the empty-glob one the module-level
    assert above exists to prevent.
    """

    class P(TemplateParams):
        """Construction pack."""

        tile: int = 128

    class C(TemplateParams):
        """Call pack."""

        a_dtype: type = None

    class K(TemplateParamsMixin):
        """A functor with one of each admissible class, plus one loose attribute."""

        Params = P
        CallParams = C
        occupancy = 1  # class constant -> admissible

        @functools.cached_property
        def derived(self):
            """Derived from the packs -> admissible."""
            return self.tile * 2

        @property
        def plain_property(self):
            """A plain property is derived too."""
            return 3

        def __init__(self):
            self._bind_params(tile=128)
            self.loose = 7  # instance state -> NOT admissible

    names = _readable_names(K)
    assert {"tile", "a_dtype", "derived", "plain_property", "occupancy"} <= names
    assert "loose" not in names, (
        "instance attributes must NOT be admitted -- they are exactly the mutable state that can "
        "desynchronize from an already-compiled kernel, which is what the guard exists to catch"
    )


@pytest.mark.parametrize("path", _KERNEL_MODULES, ids=lambda p: p.name)
def test_traced_methods_read_only_declared_params(path):
    """A traced method may read a declared param, a DERIVED property, or a class constant.

    **This is the rule that locks the paradigm in.** Inside ``@cute.jit`` / ``@cute.kernel``, a
    ``self.X`` read is resolved at trace time and folded into the kernel as a constant. If ``X`` is
    none of those three it is either (a) a runtime value, which will not survive the boundary and
    fails silently, or (b) mutable state that can desynchronize from an already-compiled kernel.
    Both are the bug class ``TemplateParams`` exists to prevent, and neither raises on its own.

    **Derived properties are admissible, and that correction is what made the rule usable.** An
    earlier version allowed only declared fields, which forced a false choice: an MMA atom or a SMEM
    layout is an OBJECT, so ``is_compile_time_value`` rejects it and it can never be a field -- yet
    a functor that computed one was then in violation for reading it. So a functor either declared
    every attribute (impossible) or declared none, and ``GemmSm90`` declared none for exactly that
    reason. Expressing "derived from the packs" as a ``cached_property`` is strictly stronger than
    stashing the value in ``__init__``: it cannot drift from its inputs, and it cannot be computed
    before they exist.

    Method calls are exempt: ``self._num_threads()`` is a pure function of the bound parameters,
    evaluated at trace time, not stored state.
    """
    import importlib

    tree = ast.parse(path.read_text())
    module = importlib.import_module(f"fold_cp_ops.kernels.{path.stem}")

    violations = []
    for cls_name, fn in _traced_methods(tree):
        functor = getattr(module, cls_name, None)
        if functor is None or not hasattr(functor, "Params"):
            continue
        allowed = _readable_names(functor)
        for node in ast.walk(fn):
            if not (
                isinstance(node, ast.Attribute)
                and isinstance(node.value, ast.Name)
                and node.value.id == "self"
                and not isinstance(getattr(node, "ctx", None), ast.Store)
            ):
                continue
            if node.attr in allowed:
                continue
            # A method is fine: `self._num_threads()` / `self.kernel(...)` are pure functions of
            # the bound parameters evaluated at trace time, not stored state.
            if callable(getattr(functor, node.attr, None)):
                continue
            violations.append(f"{path.name}:{cls_name}.{fn.name} reads self.{node.attr}")

    assert not violations, (
        "traced methods may read only a DECLARED parameter, a DERIVED property, or a class "
        "constant; these do not:\n  "
        + "\n  ".join(sorted(set(violations)))
        + "\nDeclare them in the functor's Params/CallParams, express them as a cached_property "
        "over those, or pass them as explicit kernel arguments."
    )


# ── the COMPILE KEY: a compile-time gate must be able to split it ─────────────────────────────
#: The calls whose argument is evaluated at TRACE time and therefore selects which code is emitted.
#: `const_expr` prunes a branch; `range_constexpr` fixes an unroll count. Both mean "the value of
#: this expression decided what the artifact contains".
_COMPILE_TIME_GATES = frozenset(
    {"cutlass.const_expr", "const_expr", "cutlass.range_constexpr", "range_constexpr"}
)

#: Every module that can define a functor. `kernels/` is the NEGATIVE CONTROL and must stay in the
#: list: if the rule below were unsatisfiable, or the scanner broken, these would fail too, and a
#: guard that only ever fails on the modules it was written for proves nothing about either.
_FUNCTOR_MODULES = sorted((_REPO_ROOT / "fold_cp_ops" / "kernels").glob("*.py")) + sorted(
    (_REPO_ROOT / "fold_cp_ops" / "distributed").glob("*.py")
)
assert len(_FUNCTOR_MODULES) > len(_KERNEL_MODULES), (
    f"the distributed half of {_REPO_ROOT / 'fold_cp_ops'} contributed nothing -- the compile-key "
    f"guard below would run on kernels/ alone, which is exactly the half that already passes"
)


def _gate_tested_self_names(cls_node: ast.ClassDef) -> dict:
    """Every ``self`` attribute whose VALUE is read inside a compile-time gate, with a line number.

    Purpose
        Answers "which attributes of this functor decide what gets compiled" from source alone, so
        the check costs no GPU and no compile.

    Functionality & semantics
        Walks the class body for a call to any name in :data:`_COMPILE_TIME_GATES`, then walks that
        call's ARGUMENTS for two spellings, which is the whole subtlety:

        * ``self._flag`` -- an ``ast.Attribute`` on ``self``;
        * ``getattr(self, "_flag", <default>)`` -- an ``ast.Call``, invisible to an attribute scan.

        **Both are required and the second is not hypothetical.** The A2A functors use the ``getattr``
        form for exactly the flags that may not exist yet, so an attribute-only scanner found 2 of
        the 4 flags this guard was written for and reported the other 2 as absent. A scan that finds
        nothing is a claim that needs its own check, which is why
        `test_the_gate_scanner_sees_both_spellings` exists.

        The whole class body is scanned, not only ``@cute.jit`` methods: a gate in an untraced helper
        that a traced method calls prunes the same branch, and `_remap_A_operand_layout` is one.

    Args:
        cls_node: A parsed ``ast.ClassDef``. Must come from the module the functor is defined in;
            a node from another file yields line numbers that point at the wrong source.

    Returns:
        ``{attribute_name: first_line_number}``. First occurrence wins, so the line names one
        representative gate rather than all of them.
    """
    found: dict = {}
    for node in ast.walk(cls_node):
        if not (isinstance(node, ast.Call) and ast.unparse(node.func) in _COMPILE_TIME_GATES):
            continue
        for arg in list(node.args) + [kw.value for kw in node.keywords]:
            for a in ast.walk(arg):
                if (
                    isinstance(a, ast.Attribute)
                    and isinstance(a.value, ast.Name)
                    and a.value.id == "self"
                ):
                    found.setdefault(a.attr, a.lineno)
                elif (
                    isinstance(a, ast.Call)
                    and isinstance(a.func, ast.Name)
                    and a.func.id == "getattr"
                    and len(a.args) >= 2
                    and isinstance(a.args[0], ast.Name)
                    and a.args[0].id == "self"
                    and isinstance(a.args[1], ast.Constant)
                    and isinstance(a.args[1].value, str)
                ):
                    found.setdefault(a.args[1].value, a.lineno)
    return found


def _gated_over_mro(functor) -> set:
    """Every compile-time-gated ``self`` attribute in ``functor``'s class body OR any base's.

    Purpose
        ``COMPILE_GATED_ATTRS`` is a class attribute, so a subclass inherits its parent's names.
        Comparing an inherited declaration against a scan of the subclass's own body alone would
        report every inherited name as stale -- so the scan has to cover the same span the
        declaration does, which is the MRO.

    Functionality & semantics
        Walks ``functor.__mro__``, resolves each class to its defining file via
        ``inspect.getsourcefile``, parses it, and unions `_gate_tested_self_names` over the matching
        ``ClassDef``. Classes with no resolvable source (builtins, C extensions) are skipped, and a
        file is parsed at most once per call.

    Args:
        functor: The kernel functor CLASS, not an instance.

    Returns:
        The union of gated attribute NAMES (line numbers are dropped -- a name can be gated in
        several bases and only one line could be reported).
    """
    import inspect

    names, parsed = set(), {}
    for klass in functor.__mro__:
        try:
            src = inspect.getsourcefile(klass)
        except TypeError:
            continue  # `object` and any other C-level base: no Python source to scan
        if src is None or not pathlib.Path(src).exists():
            continue
        if src not in parsed:
            parsed[src] = ast.parse(pathlib.Path(src).read_text())
        for node in ast.walk(parsed[src]):
            if isinstance(node, ast.ClassDef) and node.name == klass.__name__:
                names |= set(_gate_tested_self_names(node))
    return names


def _keyed_names(functor) -> set:
    """The names that actually appear in ``functor.compile_key()``, and nothing else.

    Purpose
        Deliberately NARROWER than `_readable_names`: that one answers "may a traced method read
        this", this one answers "does this reach the cache key". The gap between the two is the
        defect class this guard exists for -- an attribute can be perfectly legal to read and still
        be invisible to the key.

    Functionality & semantics
        ``compile_key()`` merges ``dataclasses.asdict(self.params)`` with ``asdict(self.call_params)``,
        so its key set is exactly the two packs' declared fields. Computed from the CLASS, so no
        instance is built and no phase needs binding.

    Args:
        functor: The functor CLASS. Must expose a ``Params`` that subclasses `TemplateParams`;
            ``CallParams`` is optional and defaults to the empty pack. Passing an instance raises
            ``AttributeError`` on ``Params``.

    Returns:
        The set of field names ``compile_key()`` would contain once both phases are bound.
    """
    return set(functor.Params.field_names()) | set(
        getattr(functor, "CallParams", TemplateParams).field_names()
    )


def test_the_gate_scanner_sees_both_spellings():
    """`_gate_tested_self_names` must find the ``getattr`` form, not only ``self.X``.

    Without this the guard below degrades silently: it keeps reporting findings, just fewer of them,
    and the flags it misses are the ones written defensively -- which correlates with the flags that
    are set after construction, i.e. precisely the risky ones. Measured: an attribute-only scanner
    reported 2 of the 4 flags this guard was written for.
    """
    tree = ast.parse(
        "class K:\n"
        "    @cute.jit\n"
        "    def f(self):\n"
        "        if cutlass.const_expr(self.plain):\n"
        "            pass\n"
        "        if const_expr(getattr(self, 'defensive', False)):\n"
        "            pass\n"
        "        for i in cutlass.range_constexpr(self.trips):\n"
        "            pass\n"
        "        if self.ungated:\n"
        "            pass\n"
    )
    cls = next(n for n in ast.walk(tree) if isinstance(n, ast.ClassDef))
    seen = _gate_tested_self_names(cls)
    assert set(seen) == {"plain", "defensive", "trips"}, (
        f"the scanner must see the attribute form, the getattr form and the unroll-count form, and "
        f"must NOT claim a plain runtime `if` is a compile-time gate; saw {sorted(seen)}"
    )


#: THE 37-GATE COMPILE-KEY GAP IS CLOSED. This block used to hold `_COMPILE_KEY_GAP`, a
#: per-module `strict=True` xfail waiving `gemm_sm90.py` (2 gates), `dual_gated_gemm_a2a.py` (13)
#: and `gemm_sm90_a2a.py` (22) -- 37 `const_expr` gates reading attributes `compile_key()` could not
#: see, so two functors emitting DIFFERENT code returned the SAME key.
#:
#: It was fixed rather than deleted. `TemplateParamsMixin.COMPILE_GATED_ATTRS` declares each such
#: name and `compile_key()` reads it, so the key now covers the post-construction configure surface
#: that `Params` / `CallParams` structurally cannot -- those freeze at binding, and every
#: `configure_a2a*` writes AFTER construction by design.
#:
#: The waiver's own note said "strict=True so the day it is fixed this cell fails with 'remove me'".
#: This is that removal. `_FUNCTOR_PARAMS` collapsed into `_FUNCTOR_MODULES` with it: with nothing
#: waived there is no second list to keep in step, and a marks-wrapper over an empty waiver set is a
#: place for a future waiver to be added without anyone noticing it was ever empty.
_FUNCTOR_PARAMS = _FUNCTOR_MODULES


#: The TOKEN-COUNT-derived attributes: keyed only when the compile is STATIC, absent when it is
#: dynamic. Classified 2026-08-22 by reading every one of the 37 gates at its site, and the line is
#: sharper than "shape" -- MESH and BATCH (`_a2a_cp`, `_a2a_cp0`, `_token_grid_b`) are baked in BOTH
#: modes and are always keyed. `dynamic` means dynamic in N, not in the mesh or the batch.
#:
#: Keying these unconditionally would make the artifact key SHAPE-DEPENDENT -- one artifact per
#: `N_token` -- destroying the measured property that one compiled kernel serves every N
#: (byte-identical at 2048 / 3008 / 4096). Omitting them in dynamic mode is SOUND because they sit in
#: the STATIC arm and never reach codegen there. THIS TEST IS WHAT KEEPS THAT SOUND: it is the
#: difference between an argument in a design doc and a property the interpreter checks.
_TOKEN_DERIVED_GATES = {
    "_a2a_N_loc", "_a2a_N_i_loc", "_a2a_N_j_loc",
    "_a2a_nt_pp", "_a2a_nt_i_pp", "_a2a_nt_j_pp",
    "_pad_inner_x", "_pad_inner_y", "_token_grid_x", "_token_grid_y",
}

#: The flag whose branch those gates must sit under.
_DYNAMIC_FLAG = "_a2a_dynamic"


def _reads_under_dynamic_branch(cls_node: ast.ClassDef) -> set:
    """Attribute names read anywhere INSIDE an ``if``/``else`` whose test mentions `_a2a_dynamic`.

    Args:
        cls_node: the parsed functor class.

    Returns:
        The set of ``self`` attribute names read within such a branch, either arm. Either arm is
        correct: what matters is that the read is GUARDED by the dynamic flag, not which side it is
        on -- the static values live in the ``else`` and the runtime ones in the ``if``.

    Note:
        **ONE LEVEL OF CALL IS FOLLOWED, and it is required rather than a refinement.** The static
        arm often does not read the attribute inline -- it calls a helper that does. Measured:
        ``gemm_sm90_a2a.py:3490`` is ``a_shift = self._pe_tiled_a_row_shift(...)`` inside the ``else``
        of a ``_a2a_dynamic`` branch, and the ``const_expr(self._a2a_N_loc)`` lives inside that
        helper. A check that did not follow the call reported `_a2a_N_loc` as unguarded -- a FALSE
        POSITIVE on correct code, and a guard that cries wolf gets switched off.

        Deliberately coarse beyond that. It answers "is this read governed by the dynamic flag at
        all", not "does it sit in the correct arm" -- the latter needs dataflow this does not do, and
        a check that over-claims its precision is worse than one that states its limit. The narrower
        question is answered by the key itself: a mis-armed read would change codegen in dynamic
        mode, which `ir_sha` would expose as a collision in the offline audit.
    """
    # method name -> the self attributes its body reads, so a call under a dynamic branch can be
    # resolved to the attributes it transitively guards.
    by_method: dict = {}
    for fn in [n for n in ast.walk(cls_node) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]:
        reads = set()
        for a in ast.walk(fn):
            if isinstance(a, ast.Attribute) and isinstance(a.value, ast.Name) and a.value.id == "self":
                reads.add(a.attr)
        by_method[fn.name] = reads

    guarded: set = set()
    for node in ast.walk(cls_node):
        if not isinstance(node, ast.If):
            continue
        if _DYNAMIC_FLAG not in ast.unparse(node.test):
            continue
        for stmt in list(node.body) + list(node.orelse):
            # a call to self.<helper> under this branch guards whatever that helper reads
            for a in ast.walk(stmt):
                if (
                    isinstance(a, ast.Call)
                    and isinstance(a.func, ast.Attribute)
                    and isinstance(a.func.value, ast.Name)
                    and a.func.value.id == "self"
                ):
                    guarded |= by_method.get(a.func.attr, set())
            for a in ast.walk(stmt):
                if isinstance(a, ast.Attribute) and isinstance(a.value, ast.Name) and a.value.id == "self":
                    guarded.add(a.attr)
                elif (
                    isinstance(a, ast.Call)
                    and ast.unparse(a.func) == "getattr"
                    and a.args
                    and isinstance(a.args[0], ast.Name)
                    and a.args[0].id == "self"
                    and len(a.args) > 1
                    and isinstance(a.args[1], ast.Constant)
                ):
                    guarded.add(a.args[1].value)
    return guarded


@pytest.mark.parametrize("path", _FUNCTOR_MODULES, ids=lambda p: p.name)
def test_token_derived_gates_sit_under_a_dynamic_branch(path):
    """Every TOKEN-COUNT-derived gate must be governed by `_a2a_dynamic`.

    Purpose
        This is the enforcement half of the group-(a)/(b) split that the artifact key relies on.
        Group (b) -- the token-count-derived quantities -- is keyed ONLY when the compile is static,
        and omitting it in dynamic mode is sound *because* those reads sit in the static arm of a
        dynamic branch and never reach codegen otherwise.

        That soundness was established by reading all 37 gates by hand. Hand-reading does not stay
        true: move one of those `const_expr` reads out from under its branch and the key silently
        stops covering something that now decides codegen, with no test failing. This makes the
        argument mechanical.

    Functionality & semantics
        For each functor module, finds the gated attribute names (`_gate_tested_self_names`) and the
        names read under a `_a2a_dynamic` branch, and requires every gated name that is in
        :data:`_TOKEN_DERIVED_GATES` to appear in the second set.

        A module with no token-derived gates passes trivially, which is most of them -- the check is
        scoped by CONTENT rather than by filename, so a new functor that grows one is covered the
        day it does.

    Args:
        path: a ``.py`` under ``kernels/`` or ``distributed/``.

    Raises:
        AssertionError: naming every token-derived gate that is NOT under a dynamic branch, with the
            consequence spelled out -- because the reader has to decide between moving the read back
            under the branch and promoting the attribute to always-keyed.
    """
    tree = ast.parse(path.read_text())
    problems = []
    for cls in [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]:
        gated = _gate_tested_self_names(cls)
        token_gated = {n: ln for n, ln in gated.items() if n in _TOKEN_DERIVED_GATES}
        if not token_gated:
            continue
        guarded = _reads_under_dynamic_branch(cls)
        for name, line in sorted(token_gated.items()):
            if name not in guarded:
                problems.append(f"  {path.name}:{line} gates on self.{name} OUTSIDE a {_DYNAMIC_FLAG} branch")
    assert not problems, (
        f"{len(problems)} token-count-derived gate(s) are not governed by {_DYNAMIC_FLAG}:\n"
        + "\n".join(problems)
        + "\n\nThese attributes are keyed ONLY when the compile is static, which is sound exactly "
        "because they never reach codegen in dynamic mode. A read outside the branch breaks that: "
        "it decides codegen in BOTH modes while the key sees it in only one, so two different "
        "programs can share an artifact key.\n"
        "Either move the read back under the dynamic branch, or promote the attribute to "
        "always-keyed (remove it from _TOKEN_DERIVED_GATES and declare it in Params)."
    )


@pytest.mark.parametrize("path", _FUNCTOR_PARAMS, ids=lambda p: p.name)
def test_const_expr_gates_reach_the_compile_key(path):
    """An attribute that decides what gets COMPILED must be able to split the compile KEY.

    Purpose
        **This is the first question a new gating flag gets, before any test of what it does.**
        `compile_key` documents itself as the functor's ENTIRE compile-time surface -- "a compile
        cache key built from this cannot drift from what the kernel was actually compiled with".
        A ``const_expr`` gate read off an attribute that is in neither pack falsifies that sentence:
        two functors that emit DIFFERENT code return the SAME key.

        The consequence is not a crash. `jit_cache` is persistent and CROSS-PROCESS, so one
        configuration can be served the other's artifact, and every numeric check still passes --
        the numbers are correct for the artifact that ran, it is simply the wrong artifact. Nothing
        downstream can detect that, which is why the check has to be structural and up here.

    Functionality & semantics
        For each functor class in ``path`` that binds a `TemplateParams` pack, every attribute read
        inside a compile-time gate must be one of:

        * a field of ``Params`` or ``CallParams`` -- in the key by construction;
        * a ``property`` / ``cached_property`` or a compile-time class constant -- DERIVED, so it is
          a pure function of things that are in the key, or shared by every instance;
        * a method -- a pure function of the bound parameters, evaluated at trace time.

        The admissible set is `_readable_names`, i.e. the same rule
        `test_traced_methods_read_only_declared_params` applies to what a traced method may READ.
        This test asks the narrower question of what may GATE, over a wider region: the whole class
        body rather than the traced methods alone.

        **Known softness, stated rather than implied:** a derived property is admitted without
        checking what it derives FROM. One that reads an unkeyed instance attribute is still a hole,
        and closing it needs dataflow this test does not do. Every finding below is direct.

        Modules are imported only when the AST found a gate, so a module with no functor is never
        imported and cannot fail on an unrelated import error.

    Args:
        path: A ``.py`` under ``fold_cp_ops/kernels/`` or ``fold_cp_ops/distributed/``, supplied by
            `_FUNCTOR_MODULES`. Must be importable host-only; every module reached today is.

    Raises:
        AssertionError: Listing ``file:line`` and the attribute for every gate the key cannot see.
    """
    import importlib

    tree = ast.parse(path.read_text())
    package = path.parent.name
    module = None
    violations, n_unseen = [], 0
    for cls in [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]:
        gated = _gate_tested_self_names(cls)
        if not gated:
            continue
        if module is None:
            module = importlib.import_module(f"fold_cp_ops.{package}.{path.stem}")
        functor = getattr(module, cls.name, None)
        params = getattr(functor, "Params", None)
        if not (isinstance(params, type) and issubclass(params, TemplateParams)):
            continue
        # `COMPILE_GATED_ATTRS` is the THIRD admissible class, added when the 37-gate gap was
        # closed: a name declared there is read by `compile_key()` via `getattr`, so the key CAN
        # see it even though it is neither a pack field nor a property. Admitting it here is not a
        # loophole -- `test_declared_compile_gated_attrs_match_the_ast_scan` below pins the tuple
        # against this same scan, so a declaration that does not correspond to a real gate fails
        # there, and a real gate that nobody declared still fails HERE.
        # `compile_gated_attrs()` and NOT the raw class attribute: a subclass declares only its
        # OWN new gates and the classmethod unions the MRO, so reading the attribute directly
        # here would report every gate inherited from a base as unkeyed.
        admissible = _readable_names(functor) | set(functor.compile_gated_attrs())
        unseen = [
            (attr, lineno)
            for attr, lineno in sorted(gated.items())
            if attr not in admissible and not callable(getattr(functor, attr, None))
        ]
        if unseen:
            n_unseen += len(unseen)
            violations.append(f"{cls.name} -- keyed today: {sorted(_keyed_names(functor))}")
            violations.extend(
                f"  {path.name}:{lineno} gates on self.{attr}" for attr, lineno in unseen
            )

    assert not violations, (
        f"{n_unseen} compile-time gate(s) read an attribute that compile_key() cannot see, "
        f"so two configurations that emit different code hash identically:\n  "
        + "\n  ".join(violations)
        + "\nDeclare each in the functor's Params/CallParams so it lands in the key, derive it as a "
        "cached_property over fields that are already there, or -- for a knob written AFTER "
        "construction by a configure_* method -- add it to the class's COMPILE_GATED_ATTRS."
    )


@pytest.mark.parametrize("path", _FUNCTOR_MODULES, ids=lambda p: p.name)
def test_declared_compile_gated_attrs_match_the_ast_scan(path):
    """``COMPILE_GATED_ATTRS`` must equal the gates the class actually has, both directions.

    Purpose
        The guard above admits a name because it is DECLARED. This one checks the declaration is
        true, so that admission cannot be bought by writing a name down.

    Functionality & semantics
        For every functor class in the module, scans the class body AND every base's class body
        (`_gated_over_mro`) for compile-time gates, subtracts the names the key can already see for
        another reason (`_readable_names`: pack fields, properties, compile-time class attributes)
        and the ones that resolve to methods, then requires the remainder to equal
        ``COMPILE_GATED_ATTRS`` EXACTLY.

        **The MRO walk is what makes the comparison well-formed.** ``COMPILE_GATED_ATTRS`` is a
        class attribute and therefore INHERITED, so a subclass reports its parent's names too; a
        scan of the subclass's own body alone would report every inherited name as stale. Bases are
        parsed from ``inspect.getsourcefile``, so a base in another module is included without this
        test needing to know where it lives.

        Both directions are checked and they fail for different reasons:

        * a scanned name that is NOT declared is an UNKEYED gate -- two configurations emitting
          different code hash identically, which is the defect the whole guard exists for;
        * a declared name that is NOT scanned is ROT -- a gate that was removed or renamed, leaving
          a key component that no longer corresponds to anything. Harmless to the key, but it is how
          a list stops describing the code it is supposed to describe.

    Args:
        path: A ``.py`` under ``fold_cp_ops/kernels/`` or ``fold_cp_ops/distributed/``, from
            `_FUNCTOR_MODULES`. Must be importable host-only.

    Raises:
        AssertionError: naming the undeclared and the stale names separately, because the fixes
            differ -- add the name, versus delete it.
    """
    import importlib

    tree = ast.parse(path.read_text())
    module, problems = None, []
    for cls in [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]:
        if not _gate_tested_self_names(cls):
            continue
        if module is None:
            module = importlib.import_module(f"fold_cp_ops.{path.parent.name}.{path.stem}")
        functor = getattr(module, cls.name, None)
        params = getattr(functor, "Params", None)
        if not (isinstance(params, type) and issubclass(params, TemplateParams)):
            continue
        admissible = _readable_names(functor)
        expected = {
            a
            for a in _gated_over_mro(functor)
            if a not in admissible and not callable(getattr(functor, a, None))
        }
        # The MRO UNION, not the shadowing class attribute -- and `admissible` is subtracted from
        # the declared side too. A name declared on a BASE that a SUBCLASS promotes to a `Params`
        # field is redundant there rather than stale, and forcing the subclass to re-declare the
        # difference would make every such promotion a two-file edit.
        declared = {
            a
            for a in functor.compile_gated_attrs()
            if a not in admissible and not callable(getattr(functor, a, None))
        }
        if declared != expected:
            problems.append(
                f"{cls.name}: undeclared gates {sorted(expected - declared)} | "
                f"declared but not gated {sorted(declared - expected)}"
            )

    assert not problems, (
        "COMPILE_GATED_ATTRS does not describe the class's actual compile-time gates:\n  "
        + "\n  ".join(problems)
        + "\nAdd an undeclared name so compile_key() records it; delete a stale one so the "
        "declaration keeps describing the code."
    )


# ── the CALL phase: operand facts, bound at trace time ────────────────────────────────────────
class _CallP(TemplateParams):
    """Operand facts a functor can only learn when it is handed real tensors."""

    a_dtype: type = None
    a_major: str = "k"


def _two_phase_cls():
    """Build the two-phase functor class fresh, so each test gets its own packs."""

    class P(TemplateParams):
        tile: int = 128

    class K(TemplateParamsMixin):
        Params = P
        CallParams = _CallP

        def __init__(self, tile):
            self._bind_params(tile=tile)

        def call(self, a_dtype, a_major):
            self._bind_call_params(a_dtype=a_dtype, a_major=a_major)

        @functools.cached_property
        def derived(self):
            return f"{self.tile}:{self.a_major}"

    return K


def test_the_call_phase_binds_operand_facts_after_construction():
    """The whole point: a functor can be configured before it knows its operand types."""
    k = _two_phase_cls()(128)
    assert k.tile == 128
    assert not hasattr(k, "a_major") or "call_params" not in k.__dict__
    k.call(a_dtype=cutlass.BFloat16, a_major="m")
    assert k.a_dtype is cutlass.BFloat16 and k.a_major == "m"


def test_call_phase_parameters_are_immutable_once_bound():
    """Same guarantee as construction params: they are folded in, so a rewrite desynchronizes."""
    k = _two_phase_cls()(128)
    k.call(a_dtype=cutlass.BFloat16, a_major="k")
    with pytest.raises(AttributeError, match=r"call-phase template parameter"):
        k.a_major = "m"


def test_the_call_phase_binds_only_once():
    """One functor is traced once; a second binding means it is being reused across operand types."""
    k = _two_phase_cls()(128)
    k.call(a_dtype=cutlass.BFloat16, a_major="k")
    with pytest.raises(RuntimeError, match=r"_bind_call_params\(\) called twice"):
        k.call(a_dtype=cutlass.Float16, a_major="k")


def test_the_call_phase_rejects_a_runtime_value():
    """The mistake this phase invites: binding the TENSOR instead of its element type."""
    k = _two_phase_cls()(128)
    with pytest.raises(TypeError, match=r"RUNTIME values|not compile-time"):
        k.call(a_dtype=object(), a_major="k")


def test_compile_key_is_the_union_of_both_phases():
    """A cache key built from this cannot drift from what the kernel was compiled with."""
    k = _two_phase_cls()(64)
    assert k.compile_key() == {"tile": 64}, "before __call__, the construction params alone"
    k.call(a_dtype=cutlass.BFloat16, a_major="m")
    assert k.compile_key() == {"tile": 64, "a_dtype": cutlass.BFloat16, "a_major": "m"}


def test_a_derived_value_is_a_cached_property_not_a_parameter():
    """Derived values are objects, not constants, so they can never be params -- and need not be.

    ``cached_property`` writes ``__dict__`` directly, so the guarded ``__setattr__`` does not block
    it, and a value derived from frozen params cannot desynchronize from them.
    """
    k = _two_phase_cls()(64)
    k.call(a_dtype=cutlass.BFloat16, a_major="m")
    assert k.derived == "64:m"
    assert "derived" in k.__dict__, "cached_property must be able to store its result"
    assert "derived" not in k.compile_key(), "a derived value is not part of the compile surface"


def test_a_single_phase_functor_is_unaffected():
    """CallParams defaults to an empty pack, so an existing one-phase functor behaves as before."""
    k = _Functor(dtype=cutlass.BFloat16, N=256)
    assert k.compile_key() == k.param_dict()
    assert "call_params" not in k.__dict__


@pytest.mark.parametrize(
    "a,b,why",
    [
        (True, 1, "bool and int are == in Python and would share a cache entry untagged"),
        (1, 1.0, "int and float are == in Python"),
        (True, 1.0, "all three of True/1/1.0 are mutually =="),
        ((1, 2), frozenset({1, 2}), "both encode as a MessagePack array untagged"),
        ("1", 1, "the decimal text of an int must not collide with that string"),
        (0.0, -0.0, "0.0 == -0.0 but they are different doubles"),
    ],
)
def test_values_that_are_python_equal_still_key_apart(a, b, why):
    """The whole reason the normalizer emits a type TAG.

    Each pair is ``==`` (or encodes identically) in the naive form, so an untagged MessagePack dump
    would hand them one cache entry -- and a cache entry is a compiled kernel, so the failure is a
    wrong artifact rather than a slow run.
    """
    assert canonical_key_bytes(a) != canonical_key_bytes(b), f"{a!r} and {b!r} collided: {why}"


def test_the_key_encoding_is_identical_in_a_FRESH_process():
    """Determinism across processes is the property the disk cache actually depends on.

    A key computed in one process names a file another process reads, so an encoding that varied by
    interpreter run would silently disable the cache (every lookup a miss) or, worse, vary only for
    SOME values. This is why the encoding is MessagePack over text/bytes rather than ``pickle``,
    whose output is stable only within one interpreter version and protocol.

    Run with ``PYTHONHASHSEED`` explicitly randomized, since set and dict iteration order is the
    most likely source of run-to-run drift -- and a frozenset key is in the sample for that reason.
    """
    import os
    import subprocess
    import sys

    prog = (
        "import cutlass;"
        "from fold_cp_ops._internal.compile_time.template_params import canonical_key_bytes;"
        "k=('m', 128, True, 1.5, b'x', None, (1,(2,3)), frozenset({3,1,2}), cutlass.Float32);"
        "print(canonical_key_bytes(k).hex())"
    )
    env = dict(os.environ, PYTHONHASHSEED="random", PYTHONPATH=str(_REPO_ROOT))
    outs = {
        subprocess.run(
            [sys.executable, "-c", prog], capture_output=True, text=True, env=env, check=True
        ).stdout.strip()
        for _ in range(3)
    }
    assert len(outs) == 1, f"the encoding differed across fresh processes: {outs}"
    assert outs != {""}, "the probe produced no output, so it asserted nothing"
