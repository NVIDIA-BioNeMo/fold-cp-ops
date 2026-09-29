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
"""Can every call in this package bind to the definition it will actually reach?

**The failure this exists to catch has no local cause**, which is why it cost a day of cluster time
before a static check found it in seconds. A hook is called in one file and overridden in another;
the override carries one parameter the caller does not pass; every later positional argument shifts
left; and Python reports a parameter several positions away from the one at fault. Reading either
file alone shows nothing wrong, and every runtime probe confirms that everything in view is correct.

Two real instances, both in the A2A bring-back:

* ``DualGatedGemmDistSm90._gA_local_tile`` kept the ported ``(mA_mkl, tile_coord_mnkl, batch_idx)``
  while this tree's parent takes ``(mA_mk, tile_coord_mnkl)``.
* ``DualGatedGemmDistSm90.epi_setup_postact`` kept ``varlen_manager`` -- a parameter this tree
  removed everywhere else -- in position 6. The caller passes six positionals and ``epi_gate3=`` as
  a keyword; the six fill ``params … varlen_manager``, the keyword fills the LAST slot, and ``tidx``
  is left unfilled. **The error said "missing 'tidx'" while the stale parameter was
  ``varlen_manager``, three positions away.**

**Classify by parameter NAME, never by count.** The first version of this check counted
``6 positional + 1 keyword >= 7 required`` and reported zero problems while the defect was failing
83 test blocks. A keyword argument fills the parameter it NAMES, which may be the last one, so a
count says "satisfied" for an argument list that leaves a middle parameter empty. Counting is how
the defect is made; it cannot also be how it is found.

**Dispatch is dynamic, so the receiver set is the DESCENDANTS.** A ``self.X(...)`` written in class
``C`` may execute against any subclass of ``C``, so the call is safe only if it satisfies every
override reachable below it. That is what lets this catch a call in ``_internal/`` that breaks on an
override in ``distributed/`` -- the two files never mention each other.

**``self`` / ``cls`` must be excluded.** A classmethod overridden by an instance method (or the
reverse) differs in that first name only, which is a binding convention rather than a parameter
difference; without the filter it reports every such pair. Measured: two false rows on
``_compute_stages`` before the filter.

**Four questions, and they are not interchangeable.**
:func:`override_arity_problems` asks whether a ``self.X(...)`` passes ENOUGH for every override of
``X`` below it; :func:`override_acceptance_problems` asks the opposite direction, whether every such
override ACCEPTS what the call passes, which is what a WIDENED seam breaks;
:func:`stale_override_params` asks whether an override still matches the parent it overrides. All
three reason about METHODS, so all three are structurally blind to a module-level function -- it has
no override and no parent. :func:`module_function_call_problems` is the fourth, and it caught the
last two structural defects of the A2A bring-back after the others were green.

**The two method directions stay SEPARATE on purpose.**
``test_the_arity_check_is_silent_on_the_widening_the_acceptance_check_reports`` fails if they are
folded together, and it earned that role: extending the too-few check to cover the widening as well
passes its own non-vacuity control and is still wrong, because the work already existed under the
other name. A redundancy that passes its own control is exactly the kind that lands.

**One direction is checked for module-level calls and deliberately NOT for method dispatch: the
positional/keyword COLLISION**, where an extra positional lands on a parameter the call also names
as a keyword. :func:`_binding_failure` tests it; the two method checks test too-few and
too-many/unknown-keyword, and would report such a call as neither. No instance has been observed at
method level in this tree, and this package does not add a check against a defect it has not seen --
so the gap is named here to keep it a decision on record rather than assumed coverage.

**The rename hides the re-signature.** A symbol that MOVED during the extraction was usually also
re-signatured, and fixing the name produces a different error at the same line -- which reads as a
new defect and is the same one. ``main``'s ``copy_utils.get_smem_store_atom(arch, element_type,
...)`` is this tree's ``sm90_get_smem_store_atom(element_type, ...)``, the leading ``arch`` dropped
because the package is SM90-only by decision; a ported call site passed it anyway, so the dtype
landed on ``transpose`` and Python reported a collision on a parameter the author never touched.
**Re-read the signature in the same edit that repoints the name.**

Scope note: this is a STATIC approximation. It cannot see a call whose arguments are built
dynamically (``*args`` / ``**kwargs``), and it deliberately skips those rather than guessing --
losing a finding is safe here, inventing one is not. Each check names its own exclusions; none of
them is silent about having any.
"""

import ast
import collections
import pathlib
from typing import Any, Dict, List, Optional, Set, Tuple

#: Names that differ only by the binding convention of the first parameter, never by content.
_BINDING_NAMES = frozenset({"self", "cls"})

#: Retired code is MOVED here rather than deleted, so every walk must skip it at any depth. A stale
#: override left behind would be reported against callers that can no longer reach it.
_TRASH_DIR = "trash_to_be_removed"


def _package_root() -> pathlib.Path:
    """The ``fold_cp_ops`` package directory, derived from this module rather than the CWD.

    Returns:
        The package root, so the scan is identical under any launcher or working directory.
    """
    return pathlib.Path(__file__).resolve().parent.parent


def _load_classes(root: pathlib.Path) -> Dict[str, Dict[str, Any]]:
    """Every TOP-LEVEL class under ``root``, with its bases and its directly-defined methods.

    Nested classes are skipped: they are class-scoped, never appear in a base chain, and
    ``EpilogueArguments`` alone is declared many times over.

    Args:
        root: Directory to walk. Unparseable files are skipped rather than raising -- this guard
            must not be the reason a syntax error surfaces as an import failure somewhere unrelated.

    Returns:
        ``{class_name: {"file": str, "bases": [str], "methods": {name: FunctionDef}}}``. A class
        name declared twice silently keeps the last one seen; see :func:`duplicate_class_names`,
        which reports that separately rather than letting it corrupt a verdict here.
    """
    out: Dict[str, Dict[str, Any]] = {}
    for path in sorted(root.rglob("*.py")):
        if _TRASH_DIR in path.parts:
            continue
        try:
            tree = ast.parse(path.read_text())
        except Exception:  # pragma: no cover - a syntax error is someone else's failure to report
            continue
        for node in tree.body:
            if not isinstance(node, ast.ClassDef):
                continue
            out[node.name] = {
                "file": str(path.relative_to(root)),
                "bases": [b.id for b in node.bases if isinstance(b, ast.Name)],
                "methods": {
                    m.name: m
                    for m in node.body
                    if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef))
                },
            }
    return out


def duplicate_class_names(root: Optional[pathlib.Path] = None) -> Dict[str, List[str]]:
    """Top-level class names declared in more than one module.

    Not a failure on its own -- but this guard resolves ancestry BY NAME, so a duplicated name makes
    "which class is this" ambiguous and any verdict about it unreliable. Reported so a caller can
    decide, rather than silently folded into the result.

    Args:
        root: Package root, or None for :func:`_package_root`.

    Returns:
        ``{name: [file, ...]}`` for names appearing more than once; empty when all are unique.
    """
    root = root or _package_root()
    seen: Dict[str, List[str]] = collections.defaultdict(list)
    for path in sorted(root.rglob("*.py")):
        if _TRASH_DIR in path.parts:
            continue
        try:
            tree = ast.parse(path.read_text())
        except Exception:  # pragma: no cover
            continue
        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                seen[node.name].append(str(path.relative_to(root)))
    return {k: v for k, v in seen.items() if len(v) > 1}


def _descendants(kids: Dict[str, Set[str]], cls: str) -> Set[str]:
    """Every class below ``cls`` in the name-resolved inheritance graph.

    Args:
        kids: ``{base_name: {subclass_name, ...}}``.
        cls: The class to walk down from.

    Returns:
        The transitive subclasses, excluding ``cls`` itself. Cycles cannot occur in a legal class
        graph and are guarded against anyway, so a malformed tree loops finitely rather than hangs.
    """
    out: Set[str] = set()
    stack = [cls]
    while stack:
        for kid in kids.get(stack.pop(), ()):
            if kid not in out:
                out.add(kid)
                stack.append(kid)
    return out


def _signature(fn: ast.AST) -> Tuple[List[str], Set[str], bool, Set[str]]:
    """A callee's parameters, as NAMES.

    Args:
        fn: The ``FunctionDef`` of the override being dispatched to.

    Returns:
        ``(positional_names_in_order, required_names, has_vararg, accepted_names)``. ``self``/``cls``
        is dropped from the positional list, so index 0 is the first real argument. Required names
        include keyword-only parameters that have no default.
    """
    a = fn.args
    positional = [p.arg for p in a.posonlyargs] + [p.arg for p in a.args]
    positional = [p for p in positional if p not in _BINDING_NAMES]
    kwonly = [p.arg for p in a.kwonlyargs]
    required = set(positional[: len(positional) - len(a.defaults)])
    required |= {k for k, d in zip(kwonly, a.kw_defaults) if d is None}
    return positional, required, bool(a.vararg), set(positional) | set(kwonly)


def override_arity_problems(root: Optional[pathlib.Path] = None) -> List[str]:
    """Every ``self.X(...)`` that cannot satisfy an override of ``X`` it may dispatch to.

    Args:
        root: Package root to scan, or None for :func:`_package_root`. A directory containing no
            ``.py`` files yields an empty list, which is why callers should assert the scan saw
            something -- see the module docstring's note on witnesses.

    Returns:
        Human-readable problems, empty when every call site satisfies every reachable override. Each
        names the calling ``class.method``, the callee, the receiving subclass and its file, and the
        parameter names left UNFILLED -- the last being the part a count cannot give.
    """
    root = root or _package_root()
    classes = _load_classes(root)
    kids: Dict[str, Set[str]] = collections.defaultdict(set)
    for name, info in classes.items():
        for base in info["bases"]:
            kids[base].add(name)

    problems: List[str] = []
    seen: Set[Tuple[str, str, str, str]] = set()
    for cls_name, info in classes.items():
        receivers = {cls_name} | _descendants(kids, cls_name)
        for meth_name, meth in info["methods"].items():
            for node in ast.walk(meth):
                if not (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and isinstance(node.func.value, ast.Name)
                    and node.func.value.id == "self"
                ):
                    continue
                if any(isinstance(x, ast.Starred) for x in node.args):
                    continue  # dynamic argument list -- skipped, never guessed
                if any(k.arg is None for k in node.keywords):
                    continue  # **kwargs at the call site, same reason
                called = node.func.attr
                n_pos = len(node.args)
                kw_names = {k.arg for k in node.keywords if k.arg}
                for recv in sorted(receivers):
                    callee = classes.get(recv, {}).get("methods", {}).get(called)
                    if callee is None:
                        continue
                    positional, required, has_vararg, accepted = _signature(callee)
                    if has_vararg:
                        continue  # *args absorbs any positional count
                    filled = set(positional[:n_pos]) | (kw_names & accepted)
                    missing = sorted(required - filled)
                    if not missing:
                        continue
                    key = (cls_name, meth_name, called, recv)
                    if key in seen:
                        continue
                    seen.add(key)
                    problems.append(
                        f"{cls_name}.{meth_name} calls self.{called}() with {n_pos} positional "
                        f"+ {sorted(kw_names)} keyword, which leaves {missing} UNFILLED on "
                        f"{recv}.{called} ({classes[recv]['file']}). Python will report whichever "
                        f"parameter ends up empty, which is usually NOT the one at fault -- check "
                        f"whether the override still carries a parameter this tree removed."
                    )
    return problems


def _calls_super(fn: ast.AST, method_name: str) -> bool:
    """Does this override delegate to ``super().<method_name>(...)`` anywhere in its body?

    Args:
        fn: The overriding ``FunctionDef``.
        method_name: The method being overridden, which is also the attribute the ``super()`` call
            must name -- a ``super().other()`` does not bring this method's body back into play.

    Returns:
        True when the base implementation can still run for an instance of this class.
    """
    for node in ast.walk(fn):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == method_name
            and isinstance(node.func.value, ast.Call)
            and isinstance(node.func.value.func, ast.Name)
            and node.func.value.func.id == "super"
        ):
            return True
    return False


def _reachable_receivers(
    classes: Dict[str, Dict[str, Any]],
    kids: Dict[str, Set[str]],
    cls: str,
    calling_method: str,
) -> List[str]:
    """Which classes a ``self.X(...)`` written in ``cls.calling_method`` can actually execute for.

    Starts from ``cls`` plus every descendant -- the model :func:`override_arity_problems` uses --
    and then removes a descendant that OVERRIDES ``calling_method`` without delegating to
    ``super()``. For such a receiver the base body never runs, so a call site inside it cannot
    dispatch anywhere.

    **This narrowing is what lets :func:`override_acceptance_problems` be an assertion instead of a
    pinned report.** Measured on this package: without it the check reports exactly one row,
    ``LayerNorm.__call__ -> LayerNormTransposeSm90.kernel`` (13 positional against 12 parameters),
    which is a FALSE POSITIVE of the kind :func:`stale_override_params` documents --
    ``LayerNormTransposeSm90`` declares its own ``__call__`` and never runs the base's. With the
    narrowing that row disappears and both directions are genuinely empty.

    Only the receiver's OWN override is inspected, not an intermediate class's, so the narrowing is
    weaker than it could be. That is the safe direction: a receiver kept in the set can only produce
    an extra finding to triage, never a missed one. The residual unsoundness is an explicit unbound
    call (``Base.method(self, ...)``) or a ``getattr``-dispatched one, either of which could run the
    base body for a receiver excluded here -- and both would cost a MISSED finding, which this
    module prefers to an invented one.

    Args:
        classes: The table from :func:`_load_classes`.
        kids: ``{base_name: {subclass_name, ...}}``.
        cls: The class whose method holds the call site.
        calling_method: The method holding the call site.

    Returns:
        The receiver names to check, sorted, always including ``cls`` itself.
    """
    out = []
    for recv in sorted({cls} | _descendants(kids, cls)):
        override = classes.get(recv, {}).get("methods", {}).get(calling_method)
        if recv != cls and override is not None and not _calls_super(override, calling_method):
            continue
        out.append(recv)
    return out


def _iter_dispatch_pairs(root: pathlib.Path):
    """Yield every ``(call site, reachable override)`` pair this package can produce.

    Args:
        root: Package root to scan.

    Yields:
        ``(caller_class, caller_method, call_node, receiver_class, callee_FunctionDef, classes)``.
        Calls spreading ``*args``/``**kwargs`` are skipped, since their argument list is not
        statically known.
    """
    classes = _load_classes(root)
    kids: Dict[str, Set[str]] = collections.defaultdict(set)
    for name, info in classes.items():
        for base in info["bases"]:
            kids[base].add(name)
    for cls_name, info in classes.items():
        for meth_name, meth in info["methods"].items():
            for node in ast.walk(meth):
                if not (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and isinstance(node.func.value, ast.Name)
                    and node.func.value.id == "self"
                ):
                    continue
                if any(isinstance(x, ast.Starred) for x in node.args):
                    continue
                if any(k.arg is None for k in node.keywords):
                    continue
                for recv in _reachable_receivers(classes, kids, cls_name, meth_name):
                    callee = classes.get(recv, {}).get("methods", {}).get(node.func.attr)
                    if callee is not None:
                        yield cls_name, meth_name, node, recv, callee, classes


def override_acceptance_problems(root: Optional[pathlib.Path] = None) -> List[str]:
    """Every ``self.X(...)`` passing something a reachable override of ``X`` cannot ACCEPT.

    The inverse of :func:`override_arity_problems`, and the two are not redundant -- they fail on
    opposite edits.

    ===================================  ==============================================
    edit                                 which check sees it
    ===================================  ==============================================
    an override keeps a STALE parameter   :func:`override_arity_problems` -- the call is
    the caller no longer passes           one argument short and a slot is left UNFILLED
    a seam is WIDENED and the call site   this one -- the extra keyword or positional
    passes the new arguments, while       reaches an override that never grew to take it
    another override still has the old
    signature
    ===================================  ==============================================

    **The second row is a live instance, not a hypothesis.** Widening
    ``mainloop_remap_mA`` with ``mA_mkl=None, batch_idx=None`` and passing them at the call site left
    ``GemmSm90A2A``'s two-parameter override untouched, and every A2A GEMM raised
    ``got an unexpected keyword argument 'mA_mkl'`` -- 203 blocks. **A defaulted addition on the
    PARENT does not make an OVERRIDE accept it**, and no check in this module asked that question
    while the defect was live: the arity check looks for too FEW, and the ``super()`` binds audit
    asks whether ``super()`` calls bind, which is a different call.

    This is the direction that fires whenever somebody widens a seam, which is what a bring-back does
    constantly. Adding a parameter to a hook is a THREE-place edit -- the parent, the call site, and
    every override -- and the first two are the ones you are already looking at.

    Args:
        root: Package root to scan, or None for :func:`_package_root`. Assert
            :func:`reachable_override_pairs` is non-zero before believing an empty result.

    Returns:
        Human-readable problems, empty when every reachable override accepts every call. Each names
        the calling ``class.method``, the receiving subclass and its file, and the offending
        keyword NAME or positional count.

    Notes:
        Receiver reachability is narrowed by :func:`_reachable_receivers`; a callee taking
        ``**kwargs`` accepts any keyword and one taking ``*args`` any positional count, so neither
        is reported.
    """
    root = root or _package_root()
    problems: List[str] = []
    seen: Set[Tuple[str, str, str, str, str]] = set()
    for cls_name, meth_name, node, recv, callee, classes in _iter_dispatch_pairs(root):
        positional, _, has_vararg, accepted = _signature(callee)
        kw_names = {k.arg for k in node.keywords if k.arg}
        n_pos = len(node.args)
        why = None
        unknown = sorted(kw_names - accepted)
        if unknown and not callee.args.kwarg:
            why = (
                f"passes keyword(s) {unknown} that {recv}.{node.func.attr} does not accept "
                f"(it takes {sorted(accepted)})"
            )
        elif not has_vararg and n_pos > len(positional):
            why = (
                f"passes {n_pos} positional argument(s) where {recv}.{node.func.attr} declares "
                f"{len(positional)} {positional}"
            )
        if why is None:
            continue
        key = (cls_name, meth_name, node.func.attr, recv, why[:40])
        if key in seen:
            continue
        seen.add(key)
        problems.append(
            f"{cls_name}.{meth_name} calls self.{node.func.attr}() and {why} "
            f"({classes[recv]['file']}). If the seam was just WIDENED, the override was not widened "
            f"with it -- a defaulted parameter on the parent does not make an override accept it."
        )
    return problems


def reachable_override_pairs(root: Optional[pathlib.Path] = None) -> int:
    """How many ``(call site, reachable override)`` pairs :func:`override_acceptance_problems` checked.

    The witness that makes its silence readable, for the same reason
    :func:`resolved_module_calls` exists: an empty verdict from a walk that resolved nothing is
    indistinguishable from an empty verdict from a clean tree.

    Args:
        root: Package root, or None for :func:`_package_root`.

    Returns:
        The number of pairs examined.
    """
    return sum(1 for _ in _iter_dispatch_pairs(root or _package_root()))


def _ancestors(classes: Dict[str, Dict[str, Any]], cls: str) -> List[str]:
    """Base classes of ``cls``, nearest first, resolved by name.

    Args:
        classes: The table from :func:`_load_classes`.
        cls: Class to walk up from.

    Returns:
        The transitive bases in DFS order. Not a C3 linearization -- it answers "does an ancestor
        define this name", which is what a hook lookup needs, and must not be used to decide which
        implementation would actually win.
    """
    out: List[str] = []
    seen: Set[str] = set()
    stack = list(classes.get(cls, {}).get("bases", []))
    while stack:
        base = stack.pop(0)
        if base in seen or base not in classes:
            continue
        seen.add(base)
        out.append(base)
        stack.extend(classes[base]["bases"])
    return out


def stale_override_params(root: Optional[pathlib.Path] = None) -> List[str]:
    """Overrides that REQUIRE a parameter their own parent does not have.

    The second half of this guard, and it answers a different question from
    :func:`override_arity_problems`. That one asks whether the call sites we can SEE satisfy the
    override; this one asks whether the override's signature still matches the parent it overrides,
    which holds even when no call site is statically visible -- a method reached dynamically, from a
    test, or from outside the package.

    Both were needed in practice. The extraction that motivated this guard dropped ``varlen_*``
    parameters from the parents in ``kernels/``/``_internal/`` and adapted three of the four
    subclass overrides in ``distributed/``. The fourth kept the parameter, and **only a comparison
    against OUR parent shows that** -- comparing against the upstream's parent shows the same
    difference for all four and reads as "expected".

    Only REQUIRED extras are reported. An override may legitimately add a DEFAULTED parameter (the
    A2A token-grid seam does exactly that), because no caller has to supply it and no argument
    shifts; a required one cannot be satisfied by an unmodified caller and is the defect signature.

    **This is a REPORT, not an assertion, and the difference is measured.** A required extra is only
    a defect when the subclass is reached through a call site it SHARES with its parent. A
    specialized variant that owns its own call path may legitimately require more --
    ``LayerNormTransposeSm90`` declares its own ``self.kernel(...)`` (``kernels/layernorm.py:1450``)
    distinct from ``LayerNorm``'s (``:380``) and so takes eight extra parameters with no caller
    broken anywhere. Four such rows exist in this package today and every one is correct, which is
    why the test PINS the known set rather than asserting emptiness: a check that fires on correct
    code gets deleted rather than fixed. :func:`override_arity_problems` is the assertion, because
    it asks the question that actually fails -- can a real caller satisfy a real override.

    Args:
        root: Package root to scan, or None for :func:`_package_root`.

    Returns:
        Human-readable problems, empty when every override's required parameters are a subset of
        its parent's. Each names the class, the method, the offending parameter NAMES, and the
        parent whose signature it diverged from.
    """
    root = root or _package_root()
    classes = _load_classes(root)
    problems: List[str] = []
    for cls_name, info in classes.items():
        for meth_name, meth in info["methods"].items():
            parent = next(
                (
                    (a, classes[a]["methods"][meth_name])
                    for a in _ancestors(classes, cls_name)
                    if meth_name in classes[a]["methods"]
                ),
                None,
            )
            if parent is None:
                continue  # subclass-introduced; nothing to diverge from
            parent_name, pdef = parent
            _, required, _, _ = _signature(meth)
            _, _, _, parent_accepts = _signature(pdef)
            extra = sorted(required - parent_accepts)
            if not extra:
                continue
            problems.append(
                f"{cls_name}.{meth_name} ({info['file']}) REQUIRES {extra}, which "
                f"{parent_name}.{meth_name} does not accept. If this is a parameter the tree "
                f"removed, the override was not adapted with its parent and every caller is now "
                f"one argument short -- the resulting error names whichever slot ends up empty, "
                f"not this one."
            )
    return problems


def scanned_module_count(root: Optional[pathlib.Path] = None) -> int:
    """How many ``.py`` files a scan of ``root`` would read.

    Exists so a caller can prove the scan was non-empty. A guard that returns "no problems" because
    it read no files is indistinguishable from one that read everything and found nothing, and this
    package has produced that exact false negative more than once.

    Args:
        root: Package root, or None for :func:`_package_root`.

    Returns:
        The count of non-trash ``.py`` files.
    """
    root = root or _package_root()
    return sum(1 for p in root.rglob("*.py") if _TRASH_DIR not in p.parts)


#: Decorators that provably do NOT change the signature a caller must satisfy, so a function
#: carrying only these can still be checked. Anything else -- ``jit_cache``, ``autotune`` -- is
#: SKIPPED rather than assumed, because a wrapper that rebuilds the argument list would make every
#: call site through it a false positive, and a guard that fires on correct code gets deleted rather
#: than fixed. Pinned by a test so the set cannot grow without a decision.
_SIGNATURE_PRESERVING_DECORATORS = frozenset(
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
)


def _dotted(root: pathlib.Path, path: pathlib.Path) -> str:
    """The importable module name of ``path``, rooted at the package directory.

    Derived from ``root.name`` rather than a hardcoded ``fold_cp_ops`` so a synthetic root under a
    temp directory resolves its own imports the same way the real package does.

    Args:
        root: The package directory being scanned.
        path: A ``.py`` file beneath it.

    Returns:
        The dotted name, with a trailing ``__init__`` dropped so a package resolves to itself.
    """
    parts = list(path.relative_to(root).with_suffix("").parts)
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join([root.name] + parts)


def _module_functions(root: pathlib.Path) -> Dict[str, Dict[str, ast.AST]]:
    """Every TOP-LEVEL function in every module beneath ``root``.

    Methods are excluded by construction (only ``tree.body`` is inspected), because a method is
    already covered by :func:`override_arity_problems` and resolving one from a call site requires
    the receiver's type, which is not statically known here.

    Args:
        root: Package directory. Unparseable files are skipped rather than raising, so this guard is
            never the reason a syntax error surfaces somewhere unrelated.

    Returns:
        ``{dotted_module: {function_name: FunctionDef}}``, including modules that define none.
    """
    out: Dict[str, Dict[str, ast.AST]] = {}
    for path in sorted(root.rglob("*.py")):
        if _TRASH_DIR in path.parts:
            continue
        try:
            tree = ast.parse(path.read_text())
        except Exception:  # pragma: no cover - someone else's failure to report
            continue
        out[_dotted(root, path)] = {
            n.name: n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
    return out


def _import_bindings(
    tree: ast.AST, cur_mod: str, funcs: Dict[str, Dict[str, ast.AST]]
) -> Tuple[Dict[str, str], Dict[str, Tuple[str, str]]]:
    """What each imported NAME in one module refers to.

    **Walks the whole tree, not ``tree.body``.** A function-local import is invisible to a
    body-only scan, and the defect this check exists to catch is behind one: the offending
    ``import fold_cp_ops._internal.copy_utils as copy_utils`` sits inside the method that calls it.

    Args:
        tree: The parsed module.
        cur_mod: Its own dotted name, used to resolve relative imports.
        funcs: The table from :func:`_module_functions`, used to reject anything outside the package
            -- a third-party name must never be resolved against a same-named module here.

    Returns:
        ``({alias: module}, {name: (module, function)})``. The first covers ``import pkg.m as m``
        and ``from pkg import m``; the second covers ``from pkg.m import f``. Names that do not
        resolve inside the package are absent from both.
    """
    aliases: Dict[str, str] = {}
    direct: Dict[str, Tuple[str, str]] = {}
    pkg_parts = cur_mod.split(".")[:-1]
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for al in node.names:
                if al.asname and al.name in funcs:
                    aliases[al.asname] = al.name
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ""
            if node.level:
                head = pkg_parts[: len(pkg_parts) - node.level + 1]
                base = ".".join(head + ([base] if base else []))
            for al in node.names:
                bound = al.asname or al.name
                if f"{base}.{al.name}" in funcs:
                    aliases[bound] = f"{base}.{al.name}"
                elif base in funcs and al.name in funcs[base]:
                    direct[bound] = (base, al.name)
    return aliases, direct


def _rebound_names(tree: ast.AST) -> Set[str]:
    """Every name ASSIGNED or bound as a parameter anywhere in a module.

    A call through such a name cannot be resolved to an import with any confidence, so it is
    skipped. Deliberately coarse -- it does not model scopes -- because over-skipping loses a
    finding while under-skipping invents one, and only the second is dangerous.

    Args:
        tree: The parsed module.

    Returns:
        The set of shadowing names.
    """
    out: Set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            out.add(node.id)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            a = node.args
            out.update(p.arg for p in a.posonlyargs + a.args + a.kwonlyargs)
            if a.vararg:
                out.add(a.vararg.arg)
            if a.kwarg:
                out.add(a.kwarg.arg)
    return out


def _free_signature(fn: ast.AST) -> Tuple[List[str], Set[str], bool, bool, Set[str]]:
    """A module-level function's parameters, as NAMES.

    Distinct from :func:`_signature` in one way that matters: it does NOT drop ``self``/``cls``,
    because a free function has no binding convention and a parameter so named is an ordinary one.

    Args:
        fn: The ``FunctionDef`` being called.

    Returns:
        ``(positional_names_in_order, required_names, has_vararg, has_kwarg, accepted_names)``.
    """
    a = fn.args
    positional = [p.arg for p in a.posonlyargs] + [p.arg for p in a.args]
    kwonly = [p.arg for p in a.kwonlyargs]
    required = set(positional[: len(positional) - len(a.defaults)])
    required |= {k for k, d in zip(kwonly, a.kw_defaults) if d is None}
    return positional, required, bool(a.vararg), bool(a.kwarg), set(positional) | set(kwonly)


def _binding_failure(fn: ast.AST, n_pos: int, kw_names: Set[str]) -> Optional[str]:
    """Why this argument list cannot bind to this definition, or None when it can.

    Checks the three ways a POSITIONAL SHIFT surfaces, which is one more than
    :func:`override_arity_problems` needs. That check looks only for an unfilled parameter, because
    a stale override is always LONGER than its caller expects. A renamed free function can be
    SHORTER -- this package dropped ``get_smem_store_atom``'s leading ``arch`` because it is
    SM90-only -- and then the extra positional lands on the next parameter and COLLIDES with the
    keyword that names it. Counting cannot tell those apart; both are "one argument out".

    Args:
        fn: The callee's ``FunctionDef``.
        n_pos: Number of positional arguments at the call site. The caller must already have
            excluded ``*args`` spreads, which make this number meaningless.
        kw_names: Keyword argument names at the call site, ``**kwargs`` spreads already excluded.

    Returns:
        A sentence naming the offending PARAMETER, or None. A callee taking ``*args`` always returns
        None -- it absorbs any positional count, so no static verdict is possible.
    """
    positional, required, has_vararg, has_kwarg, accepted = _free_signature(fn)
    if has_vararg:
        return None
    if n_pos > len(positional):
        return (
            f"passes {n_pos} positional argument(s) to {len(positional)} parameter(s) {positional}"
        )
    filled = set(positional[:n_pos])
    collision = sorted(filled & kw_names)
    if collision:
        return (
            f"fills {sorted(filled)} positionally AND names {collision} as a keyword -- Python "
            f"reports \"got multiple values for argument '{collision[0]}'\", which names the "
            f"COLLISION and not the extra argument that caused it"
        )
    unknown = sorted(kw_names - accepted)
    if unknown and not has_kwarg:
        return f"passes unknown keyword(s) {unknown}; the definition accepts {sorted(accepted)}"
    missing = sorted(required - (filled | (kw_names & accepted)))
    if missing:
        return f"leaves {missing} UNFILLED"
    return None


def _iter_resolved_calls(root: pathlib.Path):
    """Yield every call in the package that resolves to a CHECKABLE module-level definition.

    Args:
        root: Package directory to scan.

    Yields:
        ``(path, call_node, module_name, function_name, FunctionDef)``.
    """
    funcs = _module_functions(root)
    for path in sorted(root.rglob("*.py")):
        if _TRASH_DIR in path.parts:
            continue
        try:
            tree = ast.parse(path.read_text())
        except Exception:  # pragma: no cover
            continue
        cur = _dotted(root, path)
        aliases, direct = _import_bindings(tree, cur, funcs)
        shadowed = _rebound_names(tree)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if any(isinstance(x, ast.Starred) for x in node.args):
                continue  # dynamic argument list -- skipped, never guessed
            if any(k.arg is None for k in node.keywords):
                continue  # **kwargs at the call site, same reason
            target = None
            func = node.func
            if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
                mod = aliases.get(func.value.id)
                if mod and func.value.id not in shadowed and func.attr in funcs[mod]:
                    target = (mod, func.attr)
            elif isinstance(func, ast.Name) and func.id not in shadowed:
                if func.id in direct:
                    target = direct[func.id]
                elif func.id in funcs.get(cur, {}):
                    target = (cur, func.id)
            if target is None:
                continue
            fn = funcs[target[0]][target[1]]
            names = {ast.unparse(d).split("(")[0] for d in fn.decorator_list}
            if not names <= _SIGNATURE_PRESERVING_DECORATORS:
                continue
            yield path, node, target[0], target[1], fn


def module_function_call_problems(root: Optional[pathlib.Path] = None) -> List[str]:
    """Every call to a module-level function in this package that cannot bind to its definition.

    The third question in this module, and the one neither of the other two can ask.
    :func:`override_arity_problems` and :func:`stale_override_params` both reason about METHODS --
    an override against its parent, a call site against a reachable override. **A module-level
    function has no override and no parent**, so a call to ``copy_utils.f(...)`` whose definition
    was re-signatured during the extraction is invisible to both.

    **A rename and a re-signature arrive together, and the first HIDES the second.** The instance
    this was written for: ``main``'s ``copy_utils.get_smem_store_atom(arch, element_type, ...)``
    became this tree's ``sm90_get_smem_store_atom(element_type, ...)`` -- the leading ``arch`` gone
    because the package is SM90-only by decision -- while a faithfully-ported call site still passed
    it. Repointing the moved name produced a DIFFERENT error at the SAME line, which reads like a
    new defect and is the same one. **Whoever repoints a moved symbol must re-read its signature in
    the same edit**; a rerun will not tell them the two are connected.

    **Classify by NAME, never by count.** The defect above is a pure positional shift: the argument
    count was right both before and after, and a count check passes it in both states.

    Args:
        root: Package root to scan, or None for :func:`_package_root`. A directory with no ``.py``
            files yields an empty list, which is why :func:`resolved_module_calls` exists -- assert
            it is non-zero, or "no problems" cannot be distinguished from "nothing was read".

    Returns:
        Human-readable problems, empty when every resolvable call binds. Each names the calling
        file and line, the callee's module and function, and the PARAMETER at fault.

    Notes:
        Deliberately NOT covered, and silent about each because a guess is worse than a gap:

        * a call reached by ``getattr(mod, name)(...)`` or through a dynamically-imported module;
        * a call spreading ``*args`` / ``**kwargs``, whose argument list is not statically known;
        * a callee taking ``*args``, which absorbs any positional count;
        * a callee whose decorators are not all in ``_SIGNATURE_PRESERVING_DECORATORS`` (today
          ``jit_cache`` and ``autotune``), since a wrapper may rebuild the argument list;
        * a call through a name that is ASSIGNED anywhere in the same file, which cannot be
          attributed to the import with confidence.
    """
    root = root or _package_root()
    problems: List[str] = []
    for path, node, mod, name, fn in _iter_resolved_calls(root):
        why = _binding_failure(fn, len(node.args), {k.arg for k in node.keywords if k.arg})
        if why is None:
            continue
        problems.append(
            f"{path.relative_to(root)}:{node.lineno} calls {mod}.{name}() and {why}. "
            f"If this symbol was recently renamed, check whether it was also RE-SIGNATURED: the "
            f"rename is what sends you to this line, and the signature change is what broke it."
        )
    return problems


def resolved_module_calls(root: Optional[pathlib.Path] = None) -> int:
    """How many call sites :func:`module_function_call_problems` actually resolved and checked.

    The witness that makes its silence readable. Most calls in this package resolve to methods,
    third-party functions or dynamic targets and are skipped by design, so an empty problem list is
    only evidence if a non-trivial number of calls were checked to produce it. A test that asserts
    zero problems without asserting this number proves nothing.

    Args:
        root: Package root, or None for :func:`_package_root`.

    Returns:
        The count of resolved, checkable call sites.
    """
    return sum(1 for _ in _iter_resolved_calls(root or _package_root()))


__all__ = [
    "override_arity_problems",
    "override_acceptance_problems",
    "stale_override_params",
    "module_function_call_problems",
    "duplicate_class_names",
    "scanned_module_count",
    "resolved_module_calls",
    "reachable_override_pairs",
]
