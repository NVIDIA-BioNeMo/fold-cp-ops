#!/usr/bin/env python3
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

"""Sweep an artifact cache and report any `program_key` that holds MORE THAN ONE program.

WHY THIS EXISTS. `program_key` is COMPOSED -- source fingerprint, arch, functor class,
`compile_key()`, operand signature, normalised options. Composed means it can be incomplete, and the
repo has already measured one way it is: 37 `const_expr` gates across three A2A functors read
attributes `compile_key()` cannot see, so two functors that emit DIFFERENT code can hash the same.

The consequence of that is not a crash and not a miss. The cache is cross-process, so one
configuration is served the other's artifact and every numeric check still passes -- the numbers are
correct for the artifact that ran, it is simply the wrong artifact. Nothing downstream can detect it.

`ir_sha` is what makes it detectable. It is the DSL's own notion of program identity -- a hash of the
traced MLIR, complete BY CONSTRUCTION -- recorded on the miss path where the module is already built
and it costs nothing. Two entries that agree on `program_key` and DISAGREE on `ir_sha` are proof the
composed key merged two different programs.

WHAT A FINDING MEANS, AND WHAT IT DOES NOT. A collision here is a real defect in the key, and the fix
is to add the missing component (usually: declare the flag so `compile_key()` can see it -- S9).
A collision is NOT a wrong answer that has already happened: it means one COULD be served. Nothing
here can say whether it was.

WHY IT IS OFFLINE AND NOT AN ASSERT. Detecting a collision needs TWO entries, so no single process
can see it at the moment it mints one. It also needs no GPU, no nvshmem and no recompilation -- it
reads meta sidecars -- so it belongs in a nightly sweep rather than on the compile path, where it
would cost every run to catch a defect that only exists across runs.

Usage:
    python scripts/audit_artifact_keys.py <artifact-dir> [<artifact-dir> ...]

Exit status:
    0  no collision found (or nothing to check)
    1  at least one `program_key` holds two or more distinct `ir_sha`
    2  usage error
"""

import collections
import json
import pathlib
import sys


def _load(meta_path):
    """Read one meta sidecar, or None if it is unreadable.

    Args:
        meta_path: path to a ``*.meta.json``.

    Returns:
        The parsed dict, or ``None`` -- an unparseable sidecar is skipped rather than fatal, because
        a torn or quarantined entry must not stop the sweep from checking every other one.
    """
    try:
        return json.loads(meta_path.read_text())
    except Exception:
        return None


def audit(dirs):
    """Group every artifact by `program_key` and report groups with more than one `ir_sha`.

    Args:
        dirs: iterable of directories to sweep, non-recursively plus one level down (the test suite
            and the launcher both nest per-run subdirectories, and a sweep that missed them would
            report a clean cache while the entries sat one level below).

    Returns:
        ``(collisions, n_entries, n_keys)`` where ``collisions`` maps a program key to the list of
        ``(ir_sha, path)`` it holds. Entries with no ``ir_sha`` (written before the witness existed,
        or by a backend that could not produce one) are counted but cannot participate -- they are
        reported separately rather than silently treated as agreeing.
    """
    by_key = collections.defaultdict(list)
    n_entries = n_witnessless = 0
    for d in dirs:
        root = pathlib.Path(d)
        for meta_path in sorted(list(root.glob("*.meta.json")) + list(root.glob("*/*.meta.json"))):
            meta = _load(meta_path)
            if meta is None:
                continue
            n_entries += 1
            prog, ir = meta.get("program_key"), meta.get("ir_sha")
            if not prog:
                continue
            if not ir:
                n_witnessless += 1
                continue
            by_key[prog].append((ir, str(meta_path)))
    collisions = {k: v for k, v in by_key.items() if len({ir for ir, _ in v}) > 1}
    return collisions, n_entries, len(by_key), n_witnessless


def main(argv):
    """Sweep the given directories and print a verdict.

    Args:
        argv: command-line arguments; one or more directories.

    Returns:
        The process exit status (see the module docstring).
    """
    if len(argv) < 2:
        print(__doc__.strip().splitlines()[-4], file=sys.stderr)
        print("usage: audit_artifact_keys.py <artifact-dir> [...]", file=sys.stderr)
        return 2
    collisions, n_entries, n_keys, n_witnessless = audit(argv[1:])
    print(f"swept {n_entries} artifact(s) under {len(argv) - 1} director(ies): {n_keys} program key(s)")
    if n_witnessless:
        # Stated, never silently folded into "clean": an entry with no witness is one this audit
        # CANNOT check, and reporting it as checked is the failure mode an audit must not have.
        print(f"  {n_witnessless} entr(ies) carry NO ir_sha and could not be checked")
    if not collisions:
        print("no program_key holds two different programs")
        return 0
    print(f"\nCOLLISION: {len(collisions)} program key(s) hold MORE THAN ONE program.")
    print("The composed key merged two different programs, so one can be served the other's")
    print("artifact cross-process with every numeric check still passing. Add the missing key")
    print("component -- usually a flag `compile_key()` cannot see (S9).\n")
    for prog, entries in sorted(collisions.items()):
        print(f"  program_key {prog[:16]}...  {len({ir for ir, _ in entries})} distinct ir_sha:")
        for ir, path in sorted(entries):
            print(f"    ir_sha {ir[:16]}...  {path}")
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
