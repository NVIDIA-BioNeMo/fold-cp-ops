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

"""CLI (HARNESS_DESIGN §4): python -m benchmark.distributed.harness --targets a,b --baselines x,y --N-list ...
--rounds 5. Auto-imports the target modules (populating the REGISTRY), resolves names, dist-inits, and runs the
generic driver. The core imports zero kernels; only the target modules (imported here) do."""
import argparse
import importlib
import os
import pkgutil

from benchmark.distributed.harness import registry as reg
from benchmark.distributed.harness import driver as drv
from benchmark.distributed.harness.target import Ctx


def _import_all_targets():
    """Import every module under harness/targets/ so its register(...) calls populate REGISTRY."""
    import benchmark.distributed.harness.targets as pkg
    for m in pkgutil.iter_modules(pkg.__path__):
        if not m.name.startswith("_"):
            importlib.import_module(f"benchmark.distributed.harness.targets.{m.name}")


def _shapes(spec):
    return [int(x) for x in spec.split(",") if x.strip()]


class _SoloDM:
    """Single-device stand-in for ``DistributedManager`` (the ``--single-device`` cp=1 mode).

    The cp=1 cell is the DENOMINATOR of every strong-/weak-scaling speedup (docs/strong_weak_scaling_plan.md
    §6-§7), so it must be timed by the SAME driver + ``bench_utils`` path as every cp>=2 cell — but it needs
    no process group and no nvshmem ("cp=1 cells run single-process", §6). Rather than fake a 1-PE
    distributed job (which still pays an nvshmem init and can fail on a box with no fabric), hand the driver
    a manager-SHAPED object exposing exactly what the driver and ``bench_utils.resolve_dist`` duck-type:
    ``.device`` / ``.rank`` / ``.world_size`` / ``.local_rank`` / ``.barrier()``. ``world_size == 1`` makes
    ``resolve_dist`` yield a NON-distributed adapter, so ``all_reduce_max`` is the identity and the
    per-round barriers are no-ops — the measurement reduces to the plain single-GPU event window.
    """

    def __init__(self, device):
        self.device, self.rank, self.world_size, self.local_rank = device, 0, 1, 0

    def barrier(self):
        pass


def _main_single_device(args, targets):
    """cp=1 path: no torch.distributed, no nvshmem — the generic driver over a solo manager.

    Everything downstream of the seams is UNCHANGED (same run_matrix, same run_cell config protocol, same
    bench_utils event-window timer, same incremental JSON schema), so a t(1) produced here is directly
    comparable to a t(cp) produced by a torchrun'd cell. ``Dloc`` is the full ``D`` because cp == 1.
    """
    import torch
    from benchmark.distributed import bench_utils
    if not torch.cuda.is_available():
        raise SystemExit("--single-device requires a CUDA device")
    torch.cuda.set_device(0)   # one GPU per job; the launcher pins it via CUDA_VISIBLE_DEVICES
    dm = _SoloDM(torch.device("cuda", torch.cuda.current_device()))
    drv._barrier = lambda ctx: None            # no peers to synchronize
    drv._gate_output = lambda out, ref: True   # perf sweep (mirrors the distributed path's default)
    drv._MAX_CONFIGS = int(args.max_configs)
    ctx = Ctx(dm=dm, pm=None, da=bench_utils.resolve_dist(dm), N=0, cp0=1, cp1=1, Dloc=args.D, B=args.B,
              rd=args.rd, device=dm.device, rank=0)
    shapes = _shapes(args.N_list)
    print(f"[harness] SINGLE-DEVICE cp=1 D={args.D} targets={[t.name for t in targets]} shapes={shapes} "
          f"gpu={torch.cuda.get_device_name(0)}", flush=True)
    drv.run_matrix(targets, shapes, ctx, rounds=args.rounds, warmup=args.warmup, do_gate=False,
                   first_n=None, out_dir=args.out_dir, rank0=True)
    return 0


def main():
    ap = argparse.ArgumentParser("benchmark.distributed.harness")
    ap.add_argument("--targets", default="", help="COMMA-SEP registered target names (role=target)")
    ap.add_argument("--baselines", default="", help="COMMA-SEP registered baseline names (role=baseline)")
    ap.add_argument("--N-list", required=True, help="COMMA-SEP token extents N")
    ap.add_argument("--D", type=int, default=16, help="feature dim; Dloc = D // cp")
    ap.add_argument("--B", type=int, default=1)
    ap.add_argument("--rd", type=int, default=2)
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--do-gate", action="store_true", help="run the correctness gate at the anchor N")
    ap.add_argument("--max-configs", type=int, default=0,
                    help="truncate each target's autotune grid to the first N configs (0=all; smoke=1)")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--list", action="store_true", help="list registered targets and exit (no GPU/dist)")
    ap.add_argument("--single-device", action="store_true",
                    help="cp=1 mode: one GPU, NO process group, NO nvshmem (the scaling denominator t(1))")
    args = ap.parse_args()

    _import_all_targets()
    if args.list:
        print("registered:", sorted(reg.REGISTRY))
        return 0
    targets = reg.resolve([t for t in args.targets.split(",") if t.strip()], role="target")
    targets += reg.resolve([b for b in args.baselines.split(",") if b.strip()], role="baseline")
    if not targets:
        raise SystemExit("no --targets/--baselines given (see --list)")
    if args.single_device:   # cp=1: short-circuit BEFORE any dist/nvshmem import or init
        return _main_single_device(args, targets)

    # ---- dist init (mirrors back_a2a_store_bench.main) ----
    import torch
    import torch.distributed as dist
    from collections import OrderedDict
    from fold_cp_ops.distributed.distributed_manager import DistributedManager
    from benchmark.distributed import bench_utils
    from benchmark.distributed.harness.shape_gate import shape_check
    # Proven dist plumbing (barrier + placements + cp-axis sizes), taken from the SAME module the
    # upstream's roundpark bench takes them from. These are NOT operand builders or oracles -- they
    # carry no K, so this is not the cross-import the perf-K=N guard forbids (see that guard's own
    # docstring, which blesses importing K-agnostic helpers from a test module).
    #
    # KNOWN INHERITED DEFECT: `_trimul_placements` raises a pytest skip when the cp mesh has
    # ndim > 2, which in a bench run surfaces as a `Skipped` exception rather than a clean error.
    # The upstream has the identical hazard (its copy calls a bare `pytest.skip`) and its bench
    # imports the function anyway, so this is carried, not introduced -- and not silently fixed
    # here, because diverging from the upstream is a maintainer decision. Only bites at mesh ndim > 2.
    from tests.distributed.test_gemm_a2a_epi import (
        _cp_axis_sizes,
        _nvshmem_barrier,
        _trimul_placements,
    )

    if hasattr(DistributedManager, "_derive_dist_env_from_slurm"):
        DistributedManager._derive_dist_env_from_slurm()
    _local = int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", "0")))
    torch.cuda.set_device(_local)
    spec = os.environ.get("CPO_DIST_MESH", "")
    if spec:
        mesh = OrderedDict()
        for tok in spec.split(","):
            nm, sz = tok.strip().split("=", 1)
            parts = [int(p) for p in sz.strip().split("*")]
            mesh[nm.strip()] = tuple(parts) if len(parts) > 1 else parts[0]
    else:
        mesh = OrderedDict(cp=int(os.environ["WORLD_SIZE"]))
    os.environ.setdefault("CPO_DISTRIBUTED_INIT_METHOD", "ENV")
    DistributedManager.initialize(mesh, device_type="cuda")
    dm = DistributedManager()
    DistributedManager.init_nvshmem()
    # STOPGAP (cluster-family compile-blowup): raise the torch.distributed default-PG timeout so a slow
    # (not-hung) cold compile degrades to slow-not-CRASH instead of the 600s DistStoreError desync, while
    # the <=5s compile HARD-RULE fix lands. Env CPO_HARNESS_STORE_TIMEOUT_S (default 3600; 0 disables).
    from benchmark.distributed.harness._timeout import raise_store_timeout
    _sto = raise_store_timeout()
    if dm.rank == 0 and _sto:
        print(f"[harness] raised dist-store timeout to {_sto:.0f}s (STOPGAP for slow cluster compiles)",
              flush=True)
    pm = _trimul_placements(dm)[1]
    da = bench_utils.resolve_dist(dm)
    axis = _cp_axis_sizes(dm)
    cp0, cp1 = (int(axis[0]), int(axis[1])) if len(axis) == 2 else (int(axis[0]), 1)
    cp = cp0 * cp1
    Dloc = max(1, args.D // cp)

    # inject the REAL barrier + correctness gate into the driver seams (timing now via bench_utils in _timeit)
    drv._barrier = lambda ctx: _nvshmem_barrier()

    def _real_gate(out, ref):
        return True  # perf sweep default; a target with output/ref + a real _gate can override per-target
    drv._gate_output = _real_gate
    drv._MAX_CONFIGS = int(args.max_configs)   # smoke/quick mode: 1 config/target (§8 minimal smoke)

    ctx = Ctx(dm=dm, pm=pm, da=da, N=0, cp0=cp0, cp1=cp1, Dloc=Dloc, B=args.B, rd=args.rd,
              device=dm.device, rank=dm.rank)
    shapes = _shapes(args.N_list)
    first_n = min([n for n in shapes if shape_check(n, cp0, cp1)[0]], default=None)
    if dm.rank == 0:
        print(f"[harness] cp=({cp0},{cp1}) D={args.D} Dloc={Dloc} targets={[t.name for t in targets]} "
              f"shapes={shapes}", flush=True)
    drv.run_matrix(targets, shapes, ctx, rounds=args.rounds, warmup=args.warmup, do_gate=args.do_gate,
                   first_n=first_n, out_dir=args.out_dir, rank0=(dm.rank == 0))
    if dist.is_available() and dist.is_initialized():
        dist.barrier(device_ids=[_local])
    DistributedManager.cleanup()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
