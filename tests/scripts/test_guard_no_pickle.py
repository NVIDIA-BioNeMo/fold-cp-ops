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

"""Unit tests for ``scripts/guard_no_pickle.py``.

**The detection tests are the file's whole value.** A guard that matched nothing would pass the
repo scan below identically to one that works, so every banned form is fed to it explicitly -- and
so is every form that must NOT be flagged, because a guard that fires on the word "pickle" in a
comment gets disabled by the first person it annoys, and then protects nothing.
"""

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

from fold_cp_ops.testing.kernel_matrix import matrix_exempt

pytestmark = matrix_exempt(
    "the subject is a repo-wide source guard, not a kernel: no shape, dtype or tile axis exists "
    "for a KernelMatrix to declare"
)

REPO_ROOT = Path(__file__).resolve().parents[2]
GUARD = REPO_ROOT / "scripts" / "guard_no_pickle.py"


@pytest.fixture(scope="module")
def guard():
    """The guard module, loaded by path because ``scripts/`` is not an importable package."""
    spec = importlib.util.spec_from_file_location("guard_no_pickle", GUARD)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize(
    "label,src",
    [
        ("plain import", "import pickle\n"),
        ("aliased import", "import pickle as p\np.loads(b'')\n"),
        ("from-import of the callable", "from pickle import loads\n"),
        ("submodule import", "import pickle.foo\n"),
        ("cPickle", "import cPickle\n"),
        ("dill", "import dill\n"),
        ("shelve", "import shelve\n"),
        ("torch.load without weights_only", "import torch\ntorch.load('f')\n"),
    ],
)
def test_every_banned_form_is_caught(guard, label, src):
    """Each form separately, because they fail for different reasons.

    The ALIASED and FROM-IMPORT cases are why this is an AST pass rather than a grep: neither
    leaves a ``pickle.`` token in the file, so a regex over the source sees a clean module while
    ``p.loads(buf)`` deserializes attacker-chosen bytes.
    """
    assert guard._offences("x.py", src), f"{label} was not caught"


@pytest.mark.parametrize(
    "label,src",
    [
        ("the word in a docstring", '"""pickle was removed from the wire."""\n'),
        ("the word in a comment", "# pickle.loads used to be here\n"),
        ("the word in a string", "MSG = 'not picklable'\n"),
        ("torch.load with weights_only=True", "import torch\ntorch.load('f', weights_only=True)\n"),
        ("an unrelated .load", "import json\njson.load(fh)\n"),
    ],
)
def test_prose_and_safe_forms_are_NOT_flagged(guard, label, src):
    """False positives are how a guard gets deleted.

    This repo records WHY the codec changed in comments and docstrings that necessarily name
    ``pickle``. A guard that forbade the explanation would take the reasoning out with the risk,
    and ``torch.load(weights_only=True)`` is the safe spelling the guard exists to steer toward --
    flagging it would leave no compliant option at all.
    """
    assert not guard._offences("x.py", src), f"{label} was flagged"


def test_the_guard_scans_a_real_and_nonzero_file_set(guard):
    """A guard that walks nothing exits 0 exactly like a guard that passes.

    Measured on this repo at the time of writing: 239 files. The assertion is a floor rather than
    an equality so ordinary growth does not fail it, but a walk that collapses to a handful -- a
    renamed package root, a bad exclusion -- does.
    """
    rc = subprocess.run(
        [sys.executable, str(GUARD)], cwd=REPO_ROOT, capture_output=True, text=True
    )
    assert rc.returncode == 0, f"the tree is not clean:\n{rc.stderr}"

    import os

    scanned = 0
    for root in ("fold_cp_ops", "tests"):
        for dirpath, dirnames, filenames in os.walk(REPO_ROOT / root):
            dirnames[:] = [d for d in dirnames if d != "trash_to_be_removed"]
            scanned += sum(1 for f in filenames if f.endswith(".py"))
    assert scanned > 150, f"the guard would only have scanned {scanned} files; the walk is broken"


def test_the_guard_fails_on_a_planted_violation(tmp_path):
    """End-to-end, through the real CLI: a violating file must produce a non-zero exit.

    Run against a file in ``tmp_path`` rather than by editing the repo, which is what keeps this
    test from needing an entry in the guard's own EXEMPT list -- an exemption list with entries is
    the first step toward one that covers the thing you cared about.
    """
    bad = tmp_path / "fold_cp_ops" / "sneaky.py"
    bad.parent.mkdir(parents=True)
    bad.write_text("import pickle as p\n\n\ndef f(b):\n    return p.loads(b)\n")
    rc = subprocess.run(
        [sys.executable, str(GUARD), "fold_cp_ops/sneaky.py"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert rc.returncode == 1, "a planted `import pickle as p` did not fail the guard"
    assert "sneaky.py" in rc.stderr and "import pickle as p" in rc.stderr, rc.stderr
