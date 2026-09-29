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

#!/usr/bin/env python
"""Pre-commit guard: a perf harness must NOT import a correctness-test INPUT-BUILDER.

Enforces the CLAUDE.md HARD-RULE "perf/roofline/NCU benchmarks of the back-TriMul einsum MUST use
K = N_token". The 2026-07-15 K=256 bug entered because `benchmark/` perf harnesses `import`ed
`_build_inputs` (a correctness-only builder that bakes a fixed K=256) / the constant `K` from
`tests/distributed/test_*`. That cross-import IS the defect vector: it silently gives a perf bench a
thin-K O(N^2) GEMM instead of the real O(N^3) TriMul einsum, flipping every compute-vs-comm / overlap
/ crossover / SoL conclusion.

Fails (exit 1) if any staged `benchmark/**.py` imports `_build_inputs*` (the K-baking builder) or a
bare `K` from `tests.distributed.test_*`. Perf harnesses must own their K=N operand builder.
(The oracle `_ref_back_gemm_native*` is K-agnostic — it validates whatever operands it is handed — so
importing THAT from a test is allowed; only the operand-builder + the K constant bake the wrong K.)
Usage (pre-commit passes the staged files): guard_no_test_import_in_perf_bench.py <files...>
"""

import re
import sys

# The module part is `\w+`, NOT `test_\w+`, and that is a fix rather than a widening.
# `tests/distributed/correctness_harness.py` does not match `test_*`, so it was INVISIBLE to
# this guard -- and `benchmark/.../trimul_e2e.py` imports `make_weights` from it, an OPERAND
# BUILDER, which is exactly the class this guard exists to police. Two of the ten cross-import
# sites in `benchmark/` were unseen for that reason alone.
#
# Widening `BANNED_PREFIX` instead would be the WRONG fix: it would flag
# `front_a2a.py:244`'s `_build_front_decoupled`, whose own comment records that the
# front is K=D-exempt and that this is a COMPILE builder, not a K-baking input builder. A
# guard that fires on a deliberate, reasoned import teaches people to suppress the guard.
# `BANNED_PREFIX` still decides what is banned; this only decides where to look.
IMPORT_RE = re.compile(
    r"from\s+tests\.distributed\.\w+\s+import\s+(\([^)]*\)|[^\n(]+)", re.DOTALL
)
BANNED_PREFIX = (
    "_build_inputs",  # the K-baking operand builder (the oracle is K-agnostic → allowed)
    # The frozen-K CONSTANTS. They were named `K` when this guard was written and have since been
    # renamed to the repo's `K__<module-suffix>` convention, at which point the `n == "K"` arm below
    # stopped matching anything that exists -- a perf bench could import the literal 256 and this
    # guard would pass it. `n == "K"` is kept for the bare spelling; the prefix catches the live one.
    "K__",
)


def _imported_names(block):
    block = block.strip().lstrip("(").rstrip(")")
    return [n.strip() for n in re.split(r"[,\s]+", block) if n.strip()]


def main(argv):
    bad = {}
    for path in argv[1:]:
        if not (path.startswith("benchmark/") and path.endswith(".py")):
            continue
        try:
            src = open(path, encoding="utf-8").read()
        except OSError:
            continue
        for m in IMPORT_RE.finditer(src):
            names = _imported_names(m.group(1))
            hit = [n for n in names if n == "K" or any(n.startswith(p) for p in BANNED_PREFIX)]
            if hit:
                bad.setdefault(path, []).extend(hit)
    if bad:
        sys.stderr.write(
            "\nERROR: perf harness imports a correctness-test input-builder (the K=256 defect vector):\n"
        )
        for p, names in bad.items():
            sys.stderr.write(f"  {p}: imports {sorted(set(names))} from tests/distributed/*\n")
        sys.stderr.write(
            "\nA benchmark/ file must NOT import _build_inputs (the K-baking operand builder) or the "
            "constant K from a correctness test — those bake a fixed correctness-only K (e.g. K=256), "
            "making a perf bench measure a thin-K O(N^2) GEMM instead of the real O(N^3) back-TriMul "
            "einsum (K=N_token). Give the perf harness its OWN K=N operand builder.\n"
            "See CLAUDE.md HARD-RULE: perf benchmarks of the back-TriMul einsum MUST use K = N_token.\n"
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
