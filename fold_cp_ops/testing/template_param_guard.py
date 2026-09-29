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

"""A test may not configure a kernel functor by assigning to it after construction.

**The defect this exists to stop, and it is the SILENT half that matters.** A functor's compile-time
parameters are declared (``Params`` / ``CallParams``) and folded into the kernel as constants, so
``TemplateParamsMixin.__setattr__`` refuses a write to one after its phase is bound. That refusal is
loud and self-explaining -- a test doing ``g.chunk_g = 1`` gets an ``AttributeError`` naming the fix.
Nothing here is needed for that case.

What IS needed is the case one letter away. ``g._n_dual_tiles = 0`` is NOT a declared parameter, so
the write SUCCEEDS -- it creates an ordinary attribute nobody reads, while the kernel goes on
reading the real ``self.n_dual_tiles``. The test believes it configured something; it configured
nothing. **That is only ever observable when the value it failed to set differs from the default,
which is exactly when nobody is looking.** Measured in this repo: after ``g._n_dual_tiles = 99`` the
real ``g.n_dual_tiles`` is still ``0``.

**It has recurred twice within one day**, which is why it is machinery and not a review note:

* ``test_dual_gated_gemm_a2a.py`` set ``_n_dual_tiles`` and ``_gate3_n3`` (litter, silent) beside
  ``_has_gate3`` (real -- the kernel does read that one). One block, three attributes, two of them
  inert, and only the ``chunk_g`` line in the same block ever announced itself.
* The recorded front-A2A case, where a post-construction ``_do_normalize = False`` at every site
  including production would have run a LayerNorm every test believed disabled -- and passed.

**The discriminator is SETTABILITY, not name-matching, and that sentence is here to stop the next
reader re-deriving the taxonomy and getting the same three cases.** Sorting a name into "declared
parameter" / "inert near-miss" / "real, the kernel reads it" looks complete and is not. There is a
FOURTH case: the attribute is genuinely the one the kernel reads, and is STILL not assignable.
Measured -- ``_has_gate3`` is read at ``dual_gated_gemm.py:329,511`` under exactly that spelling, so
it passes every name test, and it is a ``@property`` with no setter derived as
``self.n_dual_tiles > 0``. Eight sites assigned it. Asking "is this name settable on this class"
subsumes all four; asking "does this name match something" does not.

**And there is a FIFTH case, which the bring-back produced rather than reasoning finding.** The
attribute is not a declared parameter, not a near-miss of one, and not a read-only property -- it is
not part of this tree's kernel at all. ``g._do_normalize = False`` is the measured instance: under
`main`'s LayerNorm-fused parent it disables a normalization, and on the extraction branch the parent
is the PLAIN dual, which has none. Four such names (``_do_normalize``, ``_stats_mode``, ``_gemm_K``,
``STATS_TPR``) carried 32 writes and ZERO reads here, and every static audit read them as meaningful
configuration precisely BECAUSE they are meaningful somewhere. Settability cannot separate this one
either -- the write succeeds, as it is supposed to. What separates it is whether the kernel ever
mentions the name, so `functor_self_names` walks the functor MRO's SOURCE and the finding is "this
tree's kernel never mentions it". Two restrictions keep it honest, both measured: it applies only to
names bound to a functor CONSTRUCTOR (unrestricted, it produced 7 findings that were right about the
name and wrong about the subject), and it is skipped when the file READS the attribute back (a test
proving an undeclared name stays settable must write one).

Same bargain as ``kernel_matrix``'s mandatory ``unsupported=`` and ``numeric_guard``'s coverage
layer: it cannot find a parameter nobody thought about, and it does not try. What it removes is the
cheapness of the near-miss -- "I set it" and "the kernel reads it" stop being the same state.

**Scope is per MODULE, from what that module itself imported.** The declared names come from the
functor classes in the test module's own namespace, so the guard cannot drift from the classes and
needs no import of its own; a module importing no functor is checked against an empty set and is
free by construction. A GLOBAL union over the package was the obvious alternative and is worse: it
would flag ``x.pingpong = True`` on some unrelated object in a module that never imports a kernel.

**Failure scoping.** Nothing here aborts a session. Problems are returned as strings for the caller
to attach to the offending module's items, exactly as ``kernel_matrix.audit_test_module`` and
``numeric_guard.audit_numeric_module`` do. An earlier audit in this repo raised at collection, and
one non-conforming file turned ``pytest tests/`` into "no tests ran" -- destroying the signal from
every unrelated test. Do not "harden" this into a collection error.
"""

import ast
import inspect
import pathlib
import textwrap
from typing import Any, Dict, List, Optional, Set

#: Attribute names on a functor class that hold a declared-parameter dataclass. Both phases are
#: protected by ``TemplateParamsMixin.__setattr__``, and both are folded into the compiled kernel,
#: so a near-miss on either is equally inert.
_PARAM_HOLDERS = ("Params", "CallParams")


def declared_template_params(module: Any) -> Set[str]:
    """Every declared parameter name reachable from the functor classes a module imported.

    Purpose
        Gives the audit a name set derived from the REAL classes rather than a hand-kept list, so a
        parameter added to a kernel is protected in the test modules that use it without anyone
        editing this file.

    Semantics
        Scans the module's own namespace for classes exposing a ``Params`` / ``CallParams`` with a
        ``field_names()`` classmethod, and unions those names. Duck-typed on ``field_names`` rather
        than on an isinstance check against ``TemplateParams``, so a functor that grows its own
        declaration mechanism is still covered and this module keeps importing nothing from the
        kernel tree.

    Args:
        module: An imported test module, or None. None yields an empty set, which makes every
            caller a no-op rather than an error -- collection hands modules that failed to import.

    Returns:
        The union of declared names. Empty when the module imported no functor class, which is the
        common case and costs nothing.
    """
    names: Set[str] = set()
    if module is None:
        return names
    for value in vars(module).values():
        if not isinstance(value, type):
            continue
        for holder in _PARAM_HOLDERS:
            declared = getattr(value, holder, None)
            fields = getattr(declared, "field_names", None)
            if callable(fields):
                try:
                    names |= set(fields())
                except Exception:  # noqa: BLE001 - a probe must never break collection
                    continue
    return names


def _asserted_to_raise(tree: ast.AST) -> Set[int]:
    """Ids of the ``Assign`` nodes sitting inside a ``with ...raises(...):`` block.

    Purpose
        An assignment a test wraps in ``pytest.raises`` is ASSERTING the functor's refusal, not
        performing a misconfiguration -- it is the machinery's own proof that the guard fires.
        Flagging it would make this guard refuse the tests that prove the thing it depends on.

    Semantics
        Matches any ``with`` whose context expression is a call whose name ENDS IN ``raises``, so
        ``pytest.raises`` and the repo's own ``front_door_raises`` are both recognised without this
        module importing either. Nested statements are covered: the whole ``with`` body is walked.

        **This is not an escape hatch.** It requires the assignment to be inside a block asserting
        that it raises -- a test cannot use it to sneak in a real configuration write, because a
        write that does NOT raise fails the enclosing ``pytest.raises`` with "DID NOT RAISE". The
        two spellings that would have been escape hatches -- a module-level exemption constant, and
        skipping the machinery's own test file by name -- were both rejected for that reason: they
        turn the check off for a whole file, where this turns it off for exactly the statements
        already proven to explode.

    Args:
        tree: A parsed module AST.

    Returns:
        A set of ``id()`` values. Ids rather than line numbers because one line can hold several
        statements and the identity is exact.
    """
    inside: Set[int] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.With):
            continue
        raises = any(
            isinstance(item.context_expr, ast.Call)
            and ast.unparse(item.context_expr.func).split(".")[-1].endswith("raises")
            for item in node.items
        )
        if not raises:
            continue
        for stmt in node.body:
            for child in ast.walk(stmt):
                # Calls too, not just assignments: a test proving the refusal may spell it
                # `setattr(g, "chunk_g", 1)`, and the carve-out has to cover the same shapes the
                # rule does or it protects one spelling and refuses the other.
                if isinstance(child, (ast.Assign, ast.Call)):
                    inside.add(id(child))
    return inside


def unsettable_attributes(module: Any) -> Dict[str, str]:
    """Attributes of a module's functor classes that CANNOT be assigned, and why.

    Purpose
        The fourth case, and the one that got past a hand-rolled classification. Sorting a name into
        "declared parameter" / "inert near-miss" / "real, the kernel reads it" is not enough,
        because a name can be **genuinely the one the kernel reads and still not assignable**.
        Measured instance: ``_has_gate3`` is read at ``dual_gated_gemm.py:329,511`` under exactly
        that spelling -- so it passes every name test -- and it is a ``@property`` with no setter,
        derived as ``self.n_dual_tiles > 0``. ``g._has_gate3 = False`` raises
        ``property ... has no setter``.

        The right discriminator is therefore not "does this name match something" but **"is this
        name settable on this class"**, which subsumes all four cases: a declared parameter is
        refused by ``__setattr__``, a read-only property is refused by the descriptor, an inert
        near-miss matches nothing at all, and a real settable attribute is fine.

    Semantics
        Walks the MRO of every functor class the module imported and collects ``property`` objects
        with ``fset is None``. Restricted to functor classes for the same reason the declared-name
        scan is: an unrelated imported class with a read-only ``shape`` must not make ``obj.shape =
        ...`` a finding somewhere else in the file.

    Args:
        module: An imported test module, or None. None yields an empty mapping.

    Returns:
        ``attribute -> "<Class>.<attr>"`` naming where the read-only property is declared, so the
        message can point at the definition rather than only at the symptom.
    """
    out: Dict[str, str] = {}
    if module is None:
        return out
    for value in vars(module).values():
        if not isinstance(value, type):
            continue
        if not any(getattr(value, h, None) is not None for h in _PARAM_HOLDERS):
            continue
        for klass in value.__mro__:
            for name, attr in vars(klass).items():
                if isinstance(attr, property) and attr.fset is None:
                    out.setdefault(name, f"{klass.__name__}.{name}")
    return out


def attributes_read_back(tree: ast.AST) -> Set[tuple]:
    """``(object, attribute)`` pairs the TEST ITSELF reads, not only writes.

    Purpose
        One legitimate reason to set an undeclared attribute is to prove that setting it WORKS --
        ``tests/_internal/compile_time/test_template_params.py`` writes ``f._some_cache = 5`` and
        immediately asserts it reads back 5, which is the whole point of that test. The UNREAD
        finding is about a write nobody observes; a write the test observes is by definition not
        that. Encoding the exception as "the file reads it back" rather than as a pragma keeps the
        escape hatch tied to evidence in the source instead of to a comment a reader can copy.

    Semantics
        Collects every ``NAME.attr`` that appears in a LOAD context. Assignment targets are stores
        and are not collected, so a file that only ever writes the attribute yields nothing.

    Args:
        tree: the parsed AST of the test file.

    Returns:
        ``{(object_name, attribute_name)}`` for every attribute the file reads.
    """
    out: Set[tuple] = set()
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and isinstance(node.ctx, ast.Load)
        ):
            out.add((node.value.id, node.attr))
    return out


def functor_instance_names(module: Any, tree: ast.AST) -> Set[str]:
    """Local names in a test file that are bound to a FUNCTOR CONSTRUCTOR call.

    Purpose
        The UNREAD finding below is a "not in a known set" test, so unlike the other three it is
        not made safe by a narrow name list -- on its own it fires on ANY attribute assignment to
        ANY object. Measured on this tree before this restriction: 7 findings that were all correct
        about the NAME and all wrong about the SUBJECT -- a monkeypatched class method
        (``G.build_p2p_table = ...``), attributes stuffed onto a MODULE object by the guard's own
        fixtures, and locals that were never kernels at all. Every one would have sent a reader to
        delete a line doing real work.

    Semantics
        Collects ``name = SomeFunctorClass(...)`` where the call's terminal identifier is one of the
        functor classes the module imported (same ``Params``/``CallParams`` test the declared-name
        scan uses). File-scoped rather than function-scoped: a name reused for a non-functor
        elsewhere in the same file would be over-included, which costs a possible false positive on
        that one name and buys not having to model scopes.

        A functor obtained from a HELPER (``g = _build(...)``) is deliberately not collected. That
        is the fail-safe direction: the check goes silent rather than guessing.

    Args:
        module: the imported test module, or None (yields an empty set, so nothing is flagged).
        tree: the already-parsed AST of the same file, so it is not re-read.

    Returns:
        The set of local names known to hold a functor instance.
    """
    if module is None:
        return set()
    classes = {
        name
        for name, value in vars(module).items()
        if isinstance(value, type)
        and any(getattr(value, h, None) is not None for h in _PARAM_HOLDERS)
    }
    if not classes:
        return set()
    out: Set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Call):
            continue
        func = node.value.func
        called = func.id if isinstance(func, ast.Name) else getattr(func, "attr", None)
        if called not in classes:
            continue
        for target in node.targets:
            if isinstance(target, ast.Name):
                out.add(target.id)
    return out


def functor_self_names(module: Any) -> Optional[Set[str]]:
    """Every attribute name a module's functor classes actually touch on ``self``.

    Purpose
        The FIFTH case, and the one the front-A2A bring-back produced: an attribute that is neither
        a declared parameter, nor an underscore near-miss of one, nor a read-only property -- it
        simply is not part of this tree's kernel at all. ``g._do_normalize = False`` is the worked
        example. Under `main`'s LayerNorm-fused parent it is load-bearing; on the extraction branch
        the parent is the PLAIN dual, which has no normalization to disable, so the write creates an
        attribute nobody reads and every static audit still reads it as meaningful configuration.
        Measured on this tree: ``_do_normalize``, ``_stats_mode``, ``_gemm_K`` and ``STATS_TPR``
        have 8-10 writes each and ZERO reads.

        A name-matching rule cannot see it -- the name matches nothing precisely because nothing is
        there. What separates it from a real setting is whether the kernel ever mentions it.

    Semantics
        Walks the MRO of every functor class the module imported and parses each class's SOURCE,
        collecting: every ``self.<name>``; every string literal handed to ``getattr``/``setattr``/
        ``hasattr`` with ``self`` as the target (the kernels reach optional a2a knobs that way, and
        missing it would flag every one of them); and every class-level name. Class attributes are
        included because a value set only on the class is still read as ``self.<name>`` elsewhere.

    Fail-SAFE, deliberately
        If ANY class in any MRO has no retrievable source -- a C extension, a dynamically built
        class, a stripped install -- this returns None, meaning "unknown", and the caller SKIPS the
        check entirely rather than flagging on partial knowledge. The cost of a false positive here
        is a developer deleting a line that was doing real work, which is strictly worse than the
        silence this guard is trying to end.

    Args:
        module: An imported test module, or None. None yields None (unknown), so a caller handed an
            un-importable module never manufactures findings.

    Returns:
        The set of names the functors touch on ``self``, or None when it could not be determined
        completely.
    """
    if module is None:
        return None
    names: Set[str] = set()
    saw_functor = False
    for value in vars(module).values():
        if not isinstance(value, type):
            continue
        if not any(getattr(value, h, None) is not None for h in _PARAM_HOLDERS):
            continue
        saw_functor = True
        for klass in value.__mro__:
            if klass is object:
                continue
            names.update(vars(klass).keys())
            try:
                src = textwrap.dedent(inspect.getsource(klass))
                tree = ast.parse(src)
            except (OSError, TypeError, SyntaxError, IndentationError):
                return None
            for node in ast.walk(tree):
                if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
                    if node.value.id == "self":
                        names.add(node.attr)
                elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                    if node.func.id in ("getattr", "setattr", "hasattr") and len(node.args) >= 2:
                        tgt, key = node.args[0], node.args[1]
                        if (
                            isinstance(tgt, ast.Name)
                            and tgt.id == "self"
                            and isinstance(key, ast.Constant)
                            and isinstance(key.value, str)
                        ):
                            names.add(key.value)
    return names if saw_functor else None


def _assignments(tree: ast.AST):
    """Every ``<name>.<attr> = ...`` in a module, as ``(lineno, obj, attr)``.

    Args:
        tree: A parsed module AST.

    Returns:
        A list of triples. Only ``Name.attr`` targets are reported: a deeper target
        (``a.b.c = ...``) is not the shape this guard is about, and including it would flag
        configuration of a nested object that has nothing to do with a functor. Assignments a test
        asserts will RAISE are excluded -- see :func:`_asserted_to_raise`.
    """
    skip = _asserted_to_raise(tree)
    out = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign) or id(node) in skip:
            continue
        for target in node.targets:
            if isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name):
                out.append((node.lineno, target.value.id, target.attr))
    # `setattr(g, "chunk_g", 1)` is the SAME write, and it is specifically the workaround somebody
    # reaches for when attribute assignment starts raising -- which is exactly what this guard makes
    # happen. A rule that can be stepped around by the most natural response to its own error message
    # has a hole aimed at its own users, so the dynamic spelling is read too. Only a LITERAL name is
    # resolvable; `setattr(g, name, v)` is not, and is left alone rather than guessed at.
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or id(node) in skip:
            continue
        if getattr(node.func, "id", None) != "setattr" or len(node.args) != 3:
            continue
        obj, attr = node.args[0], node.args[1]
        if (
            isinstance(obj, ast.Name)
            and isinstance(attr, ast.Constant)
            and isinstance(attr.value, str)
        ):
            out.append((node.lineno, obj.id, attr.value))
    return out


def template_param_problems(path, module: Any = None) -> List[str]:
    """Flag post-construction writes to a declared parameter, and the near-misses that are inert.

    Semantics
        Two findings, and the second is the one worth having:

        * **FATAL** -- assigning a DECLARED parameter. The functor already raises on this at
          runtime, so the value here is that the whole suite is checked statically rather than one
          cell at a time on whatever hardware happened to reach it.
        * **INERT** -- assigning ``_<declared>`` where ``<declared>`` is a real parameter. This
          SUCCEEDS at runtime and configures nothing. It is invisible until the value it failed to
          set differs from the default.

        A module importing no functor class has no declared names and is passed without its source
        being walked at all.

    Args:
        path: Path to the test module; read and parsed. Its name appears in the messages.
        module: The imported module object, or None. **Required in practice** -- the declared names
            come from it, so passing None makes this a silent no-op rather than an error. That is
            the safe direction for a collection hook, which is handed un-importable modules.

    Returns:
        A list of human-readable problems, empty when the module conforms. Each names the line, the
        attribute, and the fix.

    Raises:
        SyntaxError: If the module does not parse. Callers that must not abort a session should
            wrap this, as ``tests/conftest.py`` does.
    """
    declared = declared_template_params(module)
    readonly = unsettable_attributes(module)
    touched = functor_self_names(module)
    if not declared and not readonly and touched is None:
        return []
    path = pathlib.Path(path)
    tree = ast.parse(path.read_text())
    near: Dict[str, str] = {f"_{name}": name for name in declared}
    instances = functor_instance_names(module, tree)
    read_back = attributes_read_back(tree)
    problems: List[str] = []
    for lineno, obj, attr in _assignments(tree):
        if attr in readonly:
            problems.append(
                f"{path.name}:{lineno}: `{obj}.{attr} = ...` assigns a READ-ONLY property "
                f"({readonly[attr]}), which raises `property ... has no setter`. It is DERIVED, so "
                f"there is nothing to set: arrange the inputs it is computed from instead. This is "
                f"the case a name-matching rule cannot see -- the attribute really is the one the "
                f"kernel reads, and is still not assignable."
            )
        elif attr in declared:
            problems.append(
                f"{path.name}:{lineno}: `{obj}.{attr} = ...` assigns a DECLARED template parameter "
                f"after construction. It is folded into the compiled kernel as a constant, so the "
                f"write cannot affect an already-compiled kernel -- pass {attr}= to the "
                f"constructor instead."
            )
        elif attr in near:
            real = near[attr]
            problems.append(
                f"{path.name}:{lineno}: `{obj}.{attr} = ...` writes an attribute NOTHING reads. "
                f"The declared parameter is `{real}` (no leading underscore), so this succeeds, "
                f"configures nothing, and leaves `{real}` at its default -- which is invisible "
                f"until the value you meant to set differs from that default. Pass {real}= to the "
                f"constructor."
            )
        elif (
            touched is not None
            and obj in instances
            and attr not in touched
            and (obj, attr) not in read_back
        ):
            problems.append(
                f"{path.name}:{lineno}: `{obj}.{attr} = ...` sets an attribute this tree's kernel "
                f"NEVER MENTIONS -- `{attr}` appears nowhere as `self.{attr}` (nor through "
                f"getattr/hasattr/setattr) anywhere in the functor's MRO. The write succeeds and "
                f"configures nothing. This is the shape a BRING-BACK leaves behind: the line was "
                f"load-bearing against the class it was written for and is inert against the one "
                f"it was ported onto, so it reads as real configuration to every audit. Delete it, "
                f"or wire the attribute into the kernel if the behaviour was meant to come with it."
            )
    return problems


def audit_template_param_module(path, module: Any = None) -> List[str]:
    """Every template-parameter problem in one test module, for the collection-time hook.

    The single entry point ``tests/conftest.py`` and
    ``tests/testing/test_template_param_guard.py`` share, so the hook and the meta-test can never
    disagree about the rules.

    Args:
        path: Path to the test module.
        module: The imported module object, or None.

    Returns:
        The list from :func:`template_param_problems`; empty for a module that imports no functor.
        Never raises on a normal module, so one non-conforming file marks its own items and does not
        abort the session.
    """
    return template_param_problems(path, module)
