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
"""Refuse an in-tree import whose MODULE does not exist.

Why this is machinery and not a review note
    A FUNCTION-LOCAL import is invisible to everything this repo already runs. Ruff's F401 is about
    unused imports, not absent ones. pytest collection imports the module but never enters the
    function, so the file collects clean and the suite is green. `python -c "import X"` succeeds for
    the same reason. So a module MOVE leaves the import behind with NO signal at all, and the target
    is silently dead until somebody runs that one code path on that one venue.

    **Measured on this branch, twice, both found by accident while chasing something else:**

    * `harness/targets/front_route2.py` imported `dual_gated_gemm_staged_a2a` /
      `DualGatedGemmStagedDistSm90` -- neither of which exists here, because the `staged` scheme was
      retired. Found while auditing an unrelated IMA.
    * `harness/targets/trimul_e2e.py` imported `fold_cp_ops.trimul_autotune`, which moved to
      `fold_cp_ops.workflows.trimul_autotune`. Found while following up a rename.

    Two instances of one shape is a class, and the repo's answer to a class is a guard that names it.

What it checks, and what it deliberately does not
    For every `from <pkg>.<mod> import ...` and `import <pkg>.<mod>` naming an IN-TREE package
    (`fold_cp_ops`, `benchmark`, `tests`, `scripts`), the module path must resolve to a `.py` file or
    a package directory. Third-party and stdlib imports are NOT checked -- their resolution depends
    on the installed environment, which a source scan cannot see and should not guess at.

    It resolves by PATH, not by importing. Importing would execute module-level code, which for this
    package means CUDA initialization and an nvshmem bootstrap -- far too much to ask of a
    pre-commit hook, and it would make the guard's verdict depend on having a GPU.

Waivers are DECLARED, with a reason
    `WAIVED` maps a module path to why its absence is expected. Same bargain as `KernelMatrix`'s
    mandatory `unsupported=`: the guard cannot know that a module is absent ON PURPOSE, so saying so
    costs one line and makes "we know this is missing" and "it is recorded as missing" stop being
    separable states. A waiver with an empty reason is refused.

Input requirements
    Paths on argv (pre-commit's convention). A path that is not a `.py` file under a checked root is
    skipped rather than erroring, so passing the whole tree is safe. A file that does not parse is
    reported as a finding rather than crashing the hook -- an unparseable file cannot be checked, and
    silently passing it is the failure mode this guard exists to remove.

Returns
    0 when every in-tree import resolves or is waived; 1 otherwise, listing file, line, and module.
"""

import ast
import pathlib
import sys

#: In-tree top-level packages. An import naming one of these must resolve to a path in the repo;
#: anything else is a third-party or stdlib import whose resolution is an environment question.
IN_TREE_ROOTS = ("fold_cp_ops", "benchmark", "tests", "scripts")

#: module path -> why it is expected to be absent. An empty reason is refused (see `_check_waivers`).
WAIVED = {
    "fold_cp_ops.distributed.fused_trimul": (
        "the workflow layer is not ported yet and is not in the docs/a2a_bringback.md section 5.2 "
        "port order; sized at ~4600 lines in section 18.22. Resolves when that lands."
    ),
    "fold_cp_ops.distributed.fused_trimul_cp": (
        "same unported workflow layer as fused_trimul; this is the FusedTriMulCP public API."
    ),
    "benchmark.distributed.bench_front_a2a_staged": (
        "a benchmark module never brought over in the extraction. Whether it should be is a scope "
        "question for the owner, not a fix -- so it is recorded rather than quietly repointed."
    ),
}


def _module_exists(dotted: str, repo_root: pathlib.Path) -> bool:
    """Whether `dotted` resolves to a module or package PATH inside `repo_root`.

    Args:
        dotted: A dotted module path, e.g. ``fold_cp_ops.distributed.pe_map``. Only its leading
            segments are used; an empty string yields False rather than raising.
        repo_root: Directory the packages live under. Must be the repo root, not a subdirectory --
            resolving against the wrong root reports every in-tree import as dead.

    Returns:
        True if ``<root>/<a>/<b>.py`` is a file, or ``<root>/<a>/<b>/`` is a directory.

        **A bare DIRECTORY counts, and that is not laxness.** `benchmark/distributed/` and
        `scripts/` carry no ``__init__.py`` and are imported as PEP 420 namespace packages -- they
        resolve at runtime, so refusing them reported 11 WORKING imports as dead on the first run.
        Requiring ``__init__.py`` is a real rule for `tests/` (pytest's prepend mode needs it), but
        it belongs to the collection convention rather than to "does this module exist".
    """
    if not dotted:
        return False
    rel = pathlib.Path(*dotted.split("."))
    return (repo_root / rel).with_suffix(".py").is_file() or (repo_root / rel).is_dir()


def _findings(path: pathlib.Path, repo_root: pathlib.Path):
    """Every unresolvable in-tree import in one file, as ``(lineno, module)`` pairs.

    Walks the whole AST rather than only ``tree.body``, which is the entire point: a module-level
    import would have been caught by any import of the file, and the ones that hide are nested
    inside functions.

    Args:
        path: The file to scan. Must be readable UTF-8 Python; a parse failure is returned as a
            finding rather than raised, so one bad file cannot make the hook pass by crashing.
        repo_root: Passed through to `_module_exists`.

    Returns:
        A list of ``(lineno, dotted_module)``, empty when the file is clean.
    """
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (OSError, SyntaxError, UnicodeDecodeError) as exc:
        return [(0, f"<unparseable: {type(exc).__name__}>")]
    out = []
    for node in ast.walk(tree):
        mods = []
        if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            mods.append(node.module)
        elif isinstance(node, ast.Import):
            mods.extend(a.name for a in node.names)
        for mod in mods:
            if mod.split(".")[0] not in IN_TREE_ROOTS:
                continue
            if mod in WAIVED or _module_exists(mod, repo_root):
                continue
            out.append((node.lineno, mod))
    return out


def _check_waivers():
    """Refuse a waiver with no reason. Returns a list of offending module paths."""
    return [m for m, why in WAIVED.items() if not (why or "").strip()]


def main(argv):
    """Scan the given paths; return 0 when clean, 1 with a report otherwise."""
    repo_root = pathlib.Path(__file__).resolve().parent.parent
    empty = _check_waivers()
    if empty:
        sys.stderr.write(f"\nERROR: WAIVED entries with no reason: {sorted(empty)}\n")
        return 1
    bad = {}
    for raw in argv[1:]:
        path = pathlib.Path(raw)
        if path.suffix != ".py" or not path.is_file():
            continue
        hits = _findings(path, repo_root)
        if hits:
            bad[raw] = hits
    if not bad:
        return 0
    sys.stderr.write("\nERROR: import of a module that does not exist in this tree:\n")
    for p, hits in sorted(bad.items()):
        for lineno, mod in hits:
            sys.stderr.write(f"  {p}:{lineno}: {mod}\n")
    sys.stderr.write(
        "\nA FUNCTION-LOCAL import of an absent module is invisible to ruff, to pytest collection "
        "and to importing the file -- so the target stays dead until someone runs that one code "
        "path. Repoint it if the module MOVED, or add it to WAIVED in this script WITH A REASON if "
        "it is expected to be absent.\n"
    )
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
