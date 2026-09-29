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



"""Diff a fresh engine-argument capture against the frozen oracle, field by field.

Purpose
    The DTensor-API refactor is components-wiring: it may change WHO computes a config value, never
    WHAT it resolves to. So its gate is argument identity, and this is the tool that decides it.
    Pair with `trimul_engine_oracle.py`, which produces the captures.

Functionality & semantics
    Three kinds of difference are reported SEPARATELY, because they mean different things and a
    single "changed" count would hide the third:

    * ``VALUE``    -- a field resolved differently. The fence was crossed.
    * ``APPEARED`` / ``VANISHED`` -- a key one side has and the other does not. This is not
      pedantry: ``composite_k`` is ABSENT rather than ``False`` on outgoing cells, because
      `trimul_a2a` only sets it for incoming. A value-only comparison therefore cannot see a change
      that starts passing it explicitly, and would call that refactor clean.
    * cells present on one side only, reported as a count so a partial re-capture (say, local-only
      when the cross-node allocation has expired) reads as partial rather than as agreement.

Input requirements
    ``fixture``: path to the committed oracle, either the ``{"meta":…, "cells":[…]}`` document or a
    bare list of cells. ``pattern``: a glob over fresh capture shards. Both sides are keyed by
    ``(mesh, direction, D, masked)``; a duplicate key in either input is a caller error and is not
    checked for here, because the capture harness writes one shard per key.

Returns / Raises
    Exit status 0 when there are no differences, 1 otherwise -- so it can gate a shell step.
"""

from __future__ import annotations

import collections
import glob as globmod
import json
import sys
from typing import Any


def flatten(d: dict, prefix: str = "") -> dict[str, str]:
    """Flatten a nested dict to ``{dotted.path: json}``.

    Values are JSON-encoded with sorted keys so two structurally equal values compare equal as
    strings, which keeps the diff insensitive to dict ordering.

    Args:
        d: Any JSON-able mapping. Nested dicts recurse; everything else terminates a path.
        prefix: Path prefix for the recursion; callers pass ``""``.

    Returns:
        One entry per leaf.
    """
    out: dict[str, str] = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out.update(flatten(v, f"{prefix}{k}."))
        else:
            out[f"{prefix}{k}"] = json.dumps(v, sort_keys=True)
    return out


def load_fixture(path: str) -> list[dict[str, Any]]:
    """Read the committed oracle, accepting either the documented or the bare-list form.

    Args:
        path: Path to the fixture JSON. A ``{"meta":…, "cells":[…]}`` document yields its cells; a
            bare list is returned as-is, so an older capture still compares.

    Returns:
        The list of cell records.
    """
    doc = json.load(open(path))
    return doc["cells"] if isinstance(doc, dict) else doc


def cell_key(record: dict[str, Any]) -> tuple:
    """The identity of one cell: ``(mesh, direction, D, masked)``.

    ``mesh`` falls back to ``str(cp)`` so captures taken before the mesh label existed still key
    consistently with flat-mesh cells taken after it.
    """
    c = record["cell"]
    return (c.get("mesh", str(c["cp"])), c["direction"], c["D"], c["masked"])


def compare(fixture_path: str, pattern: str) -> int:
    """Compare a fresh capture against the fixture and print the report.

    Args:
        fixture_path: The committed oracle.
        pattern: Glob over fresh shards.

    Returns:
        The number of differences found; 0 means the fence held.
    """
    ref = {cell_key(r): r for r in load_fixture(fixture_path)}
    new: dict[tuple, dict] = {}
    for f in sorted(globmod.glob(pattern)):
        for r in json.load(open(f)):
            new[cell_key(r)] = r

    shared = sorted(set(ref) & set(new))
    diffs: dict[str, list] = collections.defaultdict(list)
    for k in shared:
        a = {**flatten(ref[k]["resolved"], "resolved."), **flatten(ref[k]["init_args"], "init.")}
        b = {**flatten(new[k]["resolved"], "resolved."), **flatten(new[k]["init_args"], "init.")}
        for f in sorted(set(a) | set(b)):
            if f not in b:
                diffs["VANISHED"].append((k, f, a[f]))
            elif f not in a:
                diffs["APPEARED"].append((k, f, b[f]))
            elif a[f] != b[f]:
                diffs["VALUE"].append((k, f, f"{a[f]} -> {b[f]}"))

    not_recaptured = sorted(set(ref) - set(new))
    unexpected = sorted(set(new) - set(ref))
    print(
        f"compared {len(shared)} cells (fixture {len(ref)}, fresh {len(new)}); "
        f"not re-captured this run: {len(not_recaptured)}"
    )
    if unexpected:
        print(f"CELLS NOT IN FIXTURE: {unexpected}")
    total = sum(len(v) for v in diffs.values())
    for kind in ("VALUE", "APPEARED", "VANISHED"):
        if diffs[kind]:
            print(f"\n{kind} ({len(diffs[kind])}):")
            for k, f, d in diffs[kind][:20]:
                print(f"   {k} :: {f} :: {d}")
    print(f"\nTOTAL DIFFERENCES: {total}")
    return total


def main() -> int:
    """CLI: ``trimul_engine_oracle_diff.py <fixture.json> '<fresh/*.json>'``."""
    if len(sys.argv) != 3:
        print(main.__doc__)
        return 2
    return 1 if compare(sys.argv[1], sys.argv[2]) else 0


if __name__ == "__main__":
    sys.exit(main())
