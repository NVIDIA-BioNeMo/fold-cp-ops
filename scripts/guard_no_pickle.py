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

"""Refuse a direct or aliased ``pickle`` import/call under ``fold_cp_ops/`` or ``tests/``.

WHY A GUARD AND NOT A CONVENTION. ``pickle.loads`` constructs whatever the incoming bytes describe,
which includes calling arbitrary importable objects, so every place this project deserializes is a
place a hostile payload becomes code execution. The project removed pickle from all four of its own
serialization paths -- the distributed config broadcast, the pre-compile worker pipe, the JIT cache
key and the artifact key -- and none of those removals is self-defending: re-adding
``import pickle`` is one line, reads as ordinary, and nothing fails.

WHAT IT MATCHES, and why each form is here rather than assumed absent:

* ``import pickle`` / ``import cPickle`` / ``import dill`` -- the obvious form.
* ``import pickle as p`` -- the ALIAS is why this is an AST pass and not a grep. A regex for
  ``pickle\\.`` sees nothing in ``p.loads(buf)``.
* ``from pickle import loads`` -- imports the dangerous callable under a bare name, which likewise
  leaves no ``pickle.`` token anywhere in the file.
* ``torch.load`` without ``weights_only=True`` -- pickle by another name. Torch's own default
  changed across versions, so relying on the default is relying on the installed torch.

WHAT IT DELIBERATELY DOES NOT MATCH: the word "pickle" in a comment, a docstring or a string. Those
are how this repo RECORDS why the codec changed, and a guard that forbade the explanation would
delete the reasoning along with the risk.

Usage (pre-commit passes changed files; a bare run scans the whole tree):
    python scripts/guard_no_pickle.py [path ...]
"""

import ast
import os
import sys

#: Modules whose entire purpose is to reconstruct arbitrary objects from bytes.
BANNED_MODULES = {"pickle", "cPickle", "_pickle", "dill", "shelve"}

#: Files allowed to name the banned modules, with the reason. Keep this EMPTY unless a genuine need
#: appears: an exemption list that grows is a guard that checks nothing, and the one exemption this
#: project would have needed (the guard's own test) instead builds its samples in a tmp directory.
EXEMPT = {}


def _offences(path, src):
    """Every banned import or call in *src*, as ``(lineno, description)``."""
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return []
    out, aliases = [], set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name.split(".")[0] in BANNED_MODULES:
                    aliases.add(a.asname or a.name)
                    out.append((node.lineno, f"import {a.name}"
                                             + (f" as {a.asname}" if a.asname else "")))
        elif isinstance(node, ast.ImportFrom):
            if node.module and node.module.split(".")[0] in BANNED_MODULES:
                names = ", ".join(a.name for a in node.names)
                out.append((node.lineno, f"from {node.module} import {names}"))
        elif isinstance(node, ast.Call):
            f = node.func
            # `torch.load(...)` without an explicit weights_only=True is pickle by another name.
            if isinstance(f, ast.Attribute) and f.attr == "load":
                base = f.value
                if isinstance(base, ast.Name) and base.id == "torch":
                    kw = {k.arg for k in node.keywords}
                    if "weights_only" not in kw:
                        out.append((node.lineno, "torch.load without weights_only=True"))
    return out


def main(argv):
    """Scan *argv* (or the whole tree) and report every offence. Returns a process exit status."""
    paths = [p for p in argv[1:] if p.endswith(".py")]
    if not paths:
        for root in ("fold_cp_ops", "tests"):
            for dirpath, dirnames, filenames in os.walk(root):
                dirnames[:] = [d for d in dirnames if d != "trash_to_be_removed"]
                paths += [os.path.join(dirpath, f) for f in filenames if f.endswith(".py")]
    bad = {}
    for path in paths:
        norm = path.replace(os.sep, "/")
        if "trash_to_be_removed" in norm or norm in EXEMPT:
            continue
        if not (norm.startswith("fold_cp_ops/") or norm.startswith("tests/")):
            continue
        try:
            src = open(path, encoding="utf-8").read()
        except OSError:
            continue
        hits = _offences(path, src)
        if hits:
            bad[norm] = hits
    if bad:
        sys.stderr.write("\nERROR: unsafe deserialization reintroduced:\n")
        for p, hits in sorted(bad.items()):
            for lineno, what in hits:
                sys.stderr.write(f"  {p}:{lineno}: {what}\n")
        sys.stderr.write(
            "\n`pickle.loads` builds whatever the incoming bytes describe, so any path that "
            "deserializes untrusted or merely un-authenticated data is a code-execution path. This "
            "project uses MessagePack with no object/extension hook for the config broadcast, the "
            "pre-compile worker pipe, and both cache keys.\n"
            "If you need a NEW serialized channel, use msgpack.packb/unpackb with raw=False, "
            "use_list=False, strict_map_key=True and no hooks.\n"
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
