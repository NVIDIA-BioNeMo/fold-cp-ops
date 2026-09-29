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
"""Behaviour of `scripts/guard_no_dead_module_import.py`.

The half that earns its keep is the FIRING half. A guard that only ever passes is
indistinguishable from one that checks nothing, and both look identical in CI -- so each test below
reproduces a case the repo actually hit, or a way the guard could be silently wrong.
"""

import importlib.util
import pathlib
import sys

import pytest

_ROOT = pathlib.Path(__file__).resolve().parents[2]
_SPEC = importlib.util.spec_from_file_location(
    "guard_no_dead_module_import", _ROOT / "scripts" / "guard_no_dead_module_import.py"
)
guard = importlib.util.module_from_spec(_SPEC)
sys.modules["guard_no_dead_module_import"] = guard
_SPEC.loader.exec_module(guard)


def _probe(tmp_path, body, name="probe.py"):
    """Write a probe file and return its path as a string.

    Args:
        tmp_path: pytest's ``tmp_path``.
        body: Python source. Need not be runnable -- the guard parses, never imports.
        name: File name; must end in ``.py`` or the guard skips it by design.
    """
    p = tmp_path / name
    p.write_text(body)
    return str(p)


def test_a_function_local_import_of_an_absent_module_is_caught(tmp_path):
    """THE case this guard exists for, and the one nothing else in the repo catches.

    Reproduces `harness/targets/trimul_e2e.py`: the module moved to
    `fold_cp_ops.workflows.trimul_autotune` and the import was left behind. Nested inside a
    function, so ruff sees a used import, pytest collection imports the file happily, and the
    target stays dead until that one code path runs on that one venue.
    """
    body = "def f():\n    from fold_cp_ops.trimul_autotune import trimul_autotuned\n    return trimul_autotuned\n"
    assert guard.main(["prog", _probe(tmp_path, body)]) == 1


def test_a_module_level_import_of_an_absent_module_is_caught(tmp_path):
    """The easy case, checked so the walk is not accidentally function-only."""
    body = "from fold_cp_ops.distributed.no_such_module import thing\n"
    assert guard.main(["prog", _probe(tmp_path, body)]) == 1


def test_a_real_in_tree_import_passes(tmp_path):
    """The control that keeps the guard from being trivially satisfiable by failing everything."""
    body = "def f():\n    from fold_cp_ops.distributed.pe_map import PeMap\n    return PeMap\n"
    assert guard.main(["prog", _probe(tmp_path, body)]) == 0


def test_a_namespace_package_without_init_passes(tmp_path):
    """`benchmark/distributed/` and `scripts/` are PEP 420 namespace packages.

    Requiring ``__init__.py`` reported ELEVEN working imports as dead on the guard's first run --
    every `tests/perf/*` module that imports `benchmark.distributed`. A guard whose first act is to
    condemn working code gets disabled, so this is pinned rather than left to the docstring.
    """
    body = "def f():\n    from benchmark.distributed import bench_utils\n    return bench_utils\n"
    assert guard.main(["prog", _probe(tmp_path, body)]) == 0


def test_third_party_and_stdlib_imports_are_not_checked(tmp_path):
    """Out of scope on purpose: their resolution is an environment question, not a source one."""
    body = "import torch\nimport definitely_not_installed_xyz\nfrom cutlass import Float32\n"
    assert guard.main(["prog", _probe(tmp_path, body)]) == 0


def test_a_waived_module_passes_and_the_waiver_must_carry_a_reason(tmp_path, monkeypatch):
    """A waiver silences the finding; an EMPTY reason is refused.

    Same bargain as ``KernelMatrix``'s mandatory ``unsupported=``: the guard cannot know a module is
    absent on purpose, so the declaration is what separates "we know" from "it is written down".
    """
    body = "def f():\n    from fold_cp_ops.distributed.fused_trimul import FusedTriMul\n    return FusedTriMul\n"
    assert guard.main(["prog", _probe(tmp_path, body)]) == 0

    monkeypatch.setitem(guard.WAIVED, "fold_cp_ops.distributed.fused_trimul", "  ")
    assert guard.main(["prog", _probe(tmp_path, body)]) == 1


def test_an_unparseable_file_is_a_finding_not_a_silent_pass(tmp_path):
    """A file the guard cannot read must FAIL, not be skipped.

    Skipping it would let a syntax error buy a clean verdict, which is the shape of failure this
    guard is meant to remove rather than add.
    """
    assert guard.main(["prog", _probe(tmp_path, "def f(:\n")]) == 1


def test_non_python_and_missing_paths_are_skipped(tmp_path):
    """pre-commit passes whatever changed, so a non-.py path must not error."""
    (tmp_path / "notes.md").write_text("from fold_cp_ops.nope import x\n")
    assert guard.main(["prog", str(tmp_path / "notes.md"), str(tmp_path / "gone.py")]) == 0


@pytest.mark.parametrize("mod", sorted(guard.WAIVED))
def test_every_waived_module_is_still_actually_absent(mod):
    """A waiver for a module that now EXISTS is stale and must be removed.

    Without this, a waiver outlives the absence it documents: the module lands, the guard keeps
    silently skipping it, and the next move of that module goes uncaught. The waiver list decays
    into a blindspot exactly as an un-pinned exemption list does.
    """
    assert not guard._module_exists(mod, _ROOT), (
        f"{mod} now EXISTS, so its WAIVED entry in scripts/guard_no_dead_module_import.py is stale "
        "and is silently exempting a live module from the check. Delete the entry."
    )
