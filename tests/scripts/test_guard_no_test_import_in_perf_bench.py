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

"""The guard's own tests -- it had none, which is how one of its two arms went dead unnoticed.

The guard forbids a ``benchmark/`` file importing a correctness test's K-baking operand builder or
its frozen K constant, because that turns the back-TriMul einsum from O(N^3) into a thin-K O(N^2)
matmul and silently invalidates every compute-vs-comm conclusion drawn from it.

**Why these tests exist at all.** The guard matched the constant by the literal name ``K``. The
constants were later renamed to this repo's ``K__<module-suffix>`` convention, at which point that
arm matched nothing that exists -- a perf bench could import the literal 256 and the guard would
pass it. Nothing noticed, because no test drove the guard. That is the same shape as the recorded
``startswith("P1.")`` trap: a rename lands, the code that PARSES the name stops matching, and the
silence reads as success.

So the load-bearing test here is not any single case below -- it is
`test_the_banned_constant_names_still_exist_in_the_tree`, which fails if the names the guard hunts
for stop existing. A guard whose pattern no longer matches reality passes everything, and passing
everything is indistinguishable from a clean tree.
"""

import pathlib
import subprocess
import sys

import pytest

from fold_cp_ops.testing.kernel_matrix import matrix_exempt

pytestmark = matrix_exempt(
    "the subject is a source-scanning pre-commit guard; it launches no kernel and has no shape axis"
)

_REPO = pathlib.Path(__file__).resolve().parents[2]
_GUARD = _REPO / "scripts" / "guard_no_test_import_in_perf_bench.py"


def _run(tmp_path, source: str) -> int:
    """Run the guard over one staged `benchmark/` file holding `source`; return its exit code.

    The guard filters on a path starting with ``benchmark/`` and takes filenames on argv, so the
    file is written under that relative path and the guard invoked from `tmp_path`. Returns the
    process exit status: 0 clean, 1 refused.
    """
    f = tmp_path / "benchmark" / "distributed" / "bench_probe.py"
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(source)
    rel = "benchmark/distributed/bench_probe.py"
    return subprocess.run(
        [sys.executable, str(_GUARD), rel], cwd=tmp_path, capture_output=True
    ).returncode


def test_the_renamed_frozen_K_constant_is_caught(tmp_path):
    """`K__<suffix>` is refused. This is the arm that was dead: it matched only the bare name `K`."""
    src = "from tests.distributed.test_gemm_a2a_epi import K__distributed__ib_ring\n"
    assert _run(tmp_path, src) == 1, (
        "a perf bench importing the frozen K=256 constant must be refused -- this is the exact "
        "defect vector the guard exists for, and it passed before the K__ prefix was added"
    )


def test_the_bare_K_spelling_is_still_caught(tmp_path):
    """The original spelling keeps working; the fix widened the match rather than moving it."""
    assert _run(tmp_path, "from tests.distributed.test_gemm_a2a_epi import K\n") == 1


def test_the_K_baking_operand_builder_is_still_caught(tmp_path):
    """The other arm, unchanged -- a regression check on the half that was never broken."""
    assert _run(tmp_path, "from tests.distributed.test_gemm_a2a_epi import _build_inputs\n") == 1


def test_a_caller_owns_K_builder_is_allowed(tmp_path):
    """`_build_front_decoupled` is NOT refused, and the distinction is the whole rule.

    It takes `x`, `Wg2`, `Wp2` and derives `M, K = x.shape`, so the CALLER manufactures the operands
    and owns K -- the same property that makes the K-agnostic oracle importable by the guard's own
    docstring. The banned builder takes no K argument and reaches for a module constant instead.
    Widening the guard to ban every `tests.distributed.test_*` import would forbid this sharing for
    no safety gain, and the sharing has already paid: an operand-orientation fix landed on the bench
    and the test together because both go through one builder.
    """
    src = "from tests.distributed.test_dual_gated_gemm_a2a import _build_front_decoupled\n"
    assert _run(tmp_path, src) == 0


def test_a_file_outside_benchmark_is_not_scanned(tmp_path):
    """Scope is `benchmark/**` only -- a test importing its own helpers is not this guard's business."""
    f = tmp_path / "tests" / "probe.py"
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text("from tests.distributed.test_gemm_a2a_epi import _build_inputs\n")
    r = subprocess.run(
        [sys.executable, str(_GUARD), "tests/probe.py"], cwd=tmp_path, capture_output=True
    )
    assert r.returncode == 0


@pytest.mark.parametrize("name", ["K__distributed__ib_ring", "K__distributed__pe_aligned_general"])
def test_the_banned_constant_names_still_exist_in_the_tree(name):
    """The names the guard hunts for must still EXIST, or its pattern is dead and passes everything.

    This is the test that would have caught the original defect. A pattern-matching guard has no
    "did not fire" state to notice: once the thing it matches is renamed, it reports clean forever
    and a reader cannot distinguish that from a tree with nothing to find.
    """
    src = (_REPO / "tests" / "distributed" / "test_gemm_a2a_epi.py").read_text()
    assert f"\n{name} = " in src, (
        f"{name} no longer exists, so the guard's K__ arm may match nothing. If the constant was "
        f"renamed, update BANNED_PREFIX in scripts/guard_no_test_import_in_perf_bench.py to the new "
        f"spelling -- do not just delete this test, or the guard goes silently dead again."
    )
