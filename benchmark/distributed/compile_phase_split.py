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

"""Measure the TRACE vs BACKEND split of a real A2A compile. Not a test -- it asserts nothing.

WHY IT DECIDES AN ARCHITECTURE. The artifact cache can key one of two ways:

  * the DSL's own MLIR hash -- complete BY CONSTRUCTION, immune to undeclared attributes, globals and
    monkeypatches -- but obtainable only AFTER tracing, so it costs a trace on the cache-HIT path,
    which currently skips `cute.compile` entirely;
  * a composed key -- free on the hit path, but blind to non-scalar attributes and to anything
    outside the functor.

If trace is a small fraction of a real compile, the MLIR hash is nearly free and is the better key by
a wide margin. If trace dominates, the composed key earns its holes. A plain GEMM answers neither:
its whole compile is ~1 s with trace at ~50%, while a workflow cell's kernel compile is 50-70 s.

The split is stamped at the DSL's OWN boundary: `get_module_hash` is called immediately after
`build_ir_module()` returns and before the module is compiled (`dsl.py:1715`), so this reads the
DSL's notion of "tracing is done" rather than a guess. The patch records and delegates.

Run (2 ranks is enough -- the question is about compile phases, not about the fabric):
    CPO_CACHE_ENABLED=0 CPO_JIT_ARTIFACT_ENABLED=0 PYTHONPATH=$PWD \\
      python -m torch.distributed.run --nproc_per_node=2 --master_port=29650 \\
      -m benchmark.distributed.compile_phase_split
"""

import os
import sys
import time
from collections import OrderedDict

_STAMPS = []
_ROWS = []
_PRE_ERR = []


def _install_probe():
    """Patch the DSL's trace/compile boundary and `cute.compile` to record a per-compile split."""
    from cutlass.base_dsl import dsl as _dsl

    orig_hash = _dsl.BaseDSL.get_module_hash

    def timed_hash(self, module, function_name):
        _STAMPS.append((function_name, time.perf_counter()))
        return orig_hash(self, module, function_name)

    _dsl.BaseDSL.get_module_hash = timed_hash

    import cutlass.cute as cute

    orig_compile = cute.compile

    def timed_compile(*args, **kwargs):
        # NOTE: timing `to_precompiled_mlir` here was REMOVED. It traces the functor, and the
        # real compile below then traces it AGAIN, which the functors forbid --
        # `DualGatedGemmDistSm90._bind_call_params() called twice`. That failure is the
        # finding, not an obstacle to it: an MLIR-keyed lookup cannot be followed by a
        # compile of the same instance, and `cute.compile` refuses the precompiled
        # artifact as an input (CALL_NOT_CALLABLE), so there is no way to continue from it.
        t_pre = None
        t0, mark = time.perf_counter(), len(_STAMPS)
        try:
            return orig_compile(*args, **kwargs)
        finally:
            t1 = time.perf_counter()
            if len(_STAMPS) > mark:
                name, th = _STAMPS[mark]
                _ROWS.append((name, th - t0, t1 - th, t1 - t0, t_pre))

    cute.compile = timed_compile


def main():
    """Build the real fused workflow and report where its compile time went."""
    _install_probe()
    import torch  # noqa: F401  (imported after the probe so nothing compiles first)

    from benchmark.distributed.harness.targets import trimul_e2e as T
    from fold_cp_ops.distributed.distributed_manager import DistributedManager
    from tests.distributed.perf.test_benchmark_perf_trimul_autotuned import _Cell

    DistributedManager.initialize(None, device_type="cuda", backend=None)
    ws = int(os.environ.get("WORLD_SIZE", "2"))
    DistributedManager.reset_grid_groups()
    DistributedManager.create_grid_group(OrderedDict((("cp", ws),)))
    dm = DistributedManager()

    D = int(os.environ.get("PROBE_D", "128"))
    N = int(os.environ.get("PROBE_N", "2048"))
    t0 = time.perf_counter()
    T._build_fusedcp("outgoing")(_Cell(f"cp{ws}", N, D, dm))
    wall = time.perf_counter() - t0

    if int(dm.rank) != 0:
        return 0
    tr = sum(r[1] for r in _ROWS)
    be = sum(r[2] for r in _ROWS)
    tot = sum(r[3] for r in _ROWS)
    pre = sum(r[4] for r in _ROWS if r[4] is not None)
    n_pre = sum(1 for r in _ROWS if r[4] is not None)
    print(f"\ncp{ws} D={D} N={N}   {len(_ROWS)} cute.compile call(s), build wall {wall:.1f}s")
    print(f"{'kernel':46s} {'full_s':>8s} {'precomp_s':>10s} {'precomp%':>9s}")
    for name, a, b, c, pc in sorted(_ROWS, key=lambda r: -r[3])[:14]:
        pct = f"{100 * pc / c:8.1f}%" if pc is not None else "       --"
        pcs = f"{pc:10.2f}" if pc is not None else "        --"
        print(f"{name[:46]:46s} {c:8.2f} {pcs} {pct}")
    print(f"\nTOTAL full={tot:.2f}s  trace={tr:.2f}s  backend={be:.2f}s")
    if n_pre:
        print(f"      to_precompiled_mlir={pre:.2f}s over {n_pre}/{len(_ROWS)} calls "
              f"= {100 * pre / tot:.1f}% OF THE FULL COMPILE")
        print(f"\nMLIR-keyed cache HIT would cost ~{pre:.2f}s; composed-key hit ~0.15s; "
              f"cold compile {tot:.2f}s")
        print(f"  => MLIR keying saves {100 * (tot - pre) / tot:.0f}% of compile, "
              f"composed saves ~{100 * (tot - 0.15) / tot:.0f}%")
    if _PRE_ERR:
        print(f"\nto_precompiled_mlir FAILED on {len(_PRE_ERR)} call(s): {_PRE_ERR[0][:140]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
