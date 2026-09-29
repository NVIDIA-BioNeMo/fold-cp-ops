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

"""Tests for ``scripts/audit_artifact_keys.py`` -- the offline collision sweep.

**The control that matters is `test_a_forced_collision_is_REPORTED`.** Everything else in the
artifact-cache design is a positive assertion about a cache working; this audit exists to catch the
one failure those cannot see -- a composed `program_key` that merged two genuinely different
programs. An audit that never reports anything passes every "clean cache" test identically to one
that works, so the tests here are weighted toward proving it FIRES.

No GPU, no torch, no nvshmem: the audit reads meta sidecars, which is exactly why it can run
nightly over a cache directory rather than on the compile path.
"""

import json
import subprocess
import sys
from pathlib import Path

from fold_cp_ops.testing.kernel_matrix import matrix_exempt
from fold_cp_ops.testing.numeric_guard import numeric_exempt

pytestmark = matrix_exempt(
    "the subject is an offline audit SCRIPT over meta sidecars -- it compiles nothing and has no "
    "shape/dtype/tile axes for a KernelMatrix to declare"
)

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "audit_artifact_keys.py"


def _entry(d: Path, name: str, *, program_key: str, ir_sha=None) -> Path:
    """Write one artifact + meta sidecar pair into ``d``.

    Args:
        d: the directory to write into.
        name: the artifact stem.
        program_key: the composed key this entry claims.
        ir_sha: the witness, or ``None`` to simulate an entry written before the witness existed.

    Returns:
        The meta path.
    """
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{name}.o").write_bytes(b"artifact")
    meta = {"schema": 2, "backend": "dump_object", "prefix": "k", "program_key": program_key}
    if ir_sha is not None:
        meta["ir_sha"] = ir_sha
    m = d / f"{name}.meta.json"
    m.write_text(json.dumps(meta))
    return m


def _run(*dirs):
    """Invoke the audit as a SUBPROCESS and return ``(returncode, stdout)``.

    Run as a subprocess rather than by import because the exit STATUS is half its contract -- a
    nightly job keys off it, and importing `main()` would test the printing while skipping the thing
    the caller actually reads.
    """
    r = subprocess.run(
        [sys.executable, str(_SCRIPT), *[str(d) for d in dirs]],
        capture_output=True, text=True, timeout=120,
    )
    return r.returncode, r.stdout


@numeric_exempt("asserts an audit verdict, not a computed value")
def test_a_forced_collision_is_REPORTED(tmp_path):
    """THE control. Two entries share a program_key and differ in ir_sha -- the audit must say so.

    This is the shape a real defect takes: a `const_expr` flag that `compile_key()` cannot see (the
    37 gates) makes two functors emitting DIFFERENT code hash the same. The cache is cross-process,
    so one would be served the other's artifact with every numeric check still passing. Nothing
    downstream can detect that, which is why this sweep exists -- and an audit that cannot report a
    collision is indistinguishable from a clean cache, which is why this test is the load-bearing one.
    """
    _entry(tmp_path, "a", program_key="P" * 64, ir_sha="1" * 64)
    _entry(tmp_path, "b", program_key="P" * 64, ir_sha="2" * 64)
    rc, out = _run(tmp_path)
    assert rc == 1, f"a collision must exit 1, got {rc}\n{out}"
    assert "COLLISION" in out
    assert "2 distinct ir_sha" in out
    # the report must name what to FIX, not merely that something is wrong
    assert "compile_key" in out


@numeric_exempt("asserts an audit verdict, not a computed value")
def test_a_clean_cache_is_reported_clean(tmp_path):
    """Distinct programs under distinct keys, and the SAME program twice under one key, are fine.

    The second half matters: every rank of a job mints an artifact for the same program, and a
    rank-uniform compile gives them identical `ir_sha` -- measured, 1 distinct sha across 16 ranks.
    An audit that flagged that would fire on every healthy distributed run.
    """
    _entry(tmp_path, "a", program_key="P" * 64, ir_sha="1" * 64)
    _entry(tmp_path, "b", program_key="P" * 64, ir_sha="1" * 64)  # same program, another rank
    _entry(tmp_path, "c", program_key="Q" * 64, ir_sha="3" * 64)  # a different program
    rc, out = _run(tmp_path)
    assert rc == 0, out
    assert "no program_key holds two different programs" in out


@numeric_exempt("asserts an audit verdict, not a computed value")
def test_entries_without_a_witness_are_reported_as_UNCHECKED_not_clean(tmp_path):
    """An entry the audit cannot check must never be counted as one it checked and passed.

    That is the failure mode an audit must not have: reporting "clean" over a set it never examined.
    Such entries are real -- anything written before `ir_sha` existed, or by a backend that could not
    produce one.
    """
    _entry(tmp_path, "a", program_key="P" * 64, ir_sha=None)
    _entry(tmp_path, "b", program_key="P" * 64, ir_sha=None)
    rc, out = _run(tmp_path)
    assert rc == 0, "unwitnessed entries are not a collision"
    assert "could not be checked" in out, out


@numeric_exempt("asserts an audit verdict, not a computed value")
def test_the_sweep_reaches_one_level_of_subdirectory(tmp_path):
    """Both the test suite and the launcher nest per-run subdirectories under the artifact root.

    A sweep that only globbed the top level would report a clean cache while every entry sat one
    level below -- a false clean, which is worse than no audit at all.
    """
    _entry(tmp_path / "run1__testA", "a", program_key="P" * 64, ir_sha="1" * 64)
    _entry(tmp_path / "run1__testA", "b", program_key="P" * 64, ir_sha="2" * 64)
    rc, out = _run(tmp_path)
    assert rc == 1, f"a collision one level down must still be found\n{out}"


@numeric_exempt("asserts an audit verdict, not a computed value")
def test_an_unparseable_sidecar_does_not_stop_the_sweep(tmp_path):
    """A torn or quarantined entry must not hide a collision sitting beside it."""
    (tmp_path / "broken.meta.json").write_text("{not json")
    _entry(tmp_path, "a", program_key="P" * 64, ir_sha="1" * 64)
    _entry(tmp_path, "b", program_key="P" * 64, ir_sha="2" * 64)
    rc, out = _run(tmp_path)
    assert rc == 1, f"one bad sidecar must not mask a real finding\n{out}"


@numeric_exempt("asserts an audit verdict, not a computed value")
def test_an_empty_directory_is_not_an_error(tmp_path):
    """Nothing to check is a clean verdict, not a failure -- a nightly job must not go red on it."""
    rc, out = _run(tmp_path)
    assert rc == 0, out
    assert "swept 0 artifact" in out


@numeric_exempt("asserts a usage refusal, not a computed value")
def test_no_argument_is_a_usage_error():
    """Exit 2, distinct from both verdicts, so a driver cannot mistake a mis-invocation for clean."""
    r = subprocess.run([sys.executable, str(_SCRIPT)], capture_output=True, text=True, timeout=60)
    assert r.returncode == 2
