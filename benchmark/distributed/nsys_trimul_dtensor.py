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


"""Unified nsys overlap profiler: fused ``TriMulAutotuned`` (dynamic-N) vs a selectable baseline.

ONE process per (D, cp): builds ONE ``TriMulAutotuned`` at an anchor N and drives EACH runtime N
(default 1000/2000/4000) through ``forward()``, for the requested ``--direction`` (``both`` sweeps
outgoing+incoming), alongside a ``--baseline`` path. Shape mode: ``--dynamic`` (DEFAULT; compile ONCE,
run any N — REQUIRED for a multi-N ``--Ns`` sweep) vs ``--no-dynamic`` (STATIC, bakes the shape ->
SINGLE ``--Ns``; faithful to the old ``nsys_trimul_e2e`` path — the byte-identical production kernel,
perf-NEUTRAL vs dynamic at fixed cp, so the e2e shim passes ``--no-dynamic``):

* ``--baseline none``    — fused-ONLY (the clean path; no baseline stall).
* ``--baseline dtensor`` — the NON-fused DTensor reference
  (``t2_0_dtensor_baseline.distributed_trimul_fwd`` — torch DTensor ``redistribute`` all-to-all,
  the §5 correctness ref (b)). This is the historical default of this driver.
* ``--baseline dtensor_baseline_ring_reducescatter``   — the FAIR published reference CP baseline that NATIVELY matches the
  sharding (``benchmark.distributed.trimul_dtensor_baseline`` -> the reference ``TriangularMultiplication*1D`` ring
  for a 1-D ``cp`` session, or ``TriangleMultiplication*2D`` + ``Ring2DComm`` for a 2-D
  ``cp0*cp1`` session), run AS-IS on the SAME weights. NO all-to-all — a 1-D/2-D ring.
  **THIS is THE perf-grid baseline** — reference CP is the chosen perf reference, so the profiling
  grid passes ``--baseline dtensor_baseline_ring_reducescatter`` explicitly; ``dtensor`` above is the CORRECTNESS ref only.
  (``--baseline`` defaults to ``dtensor`` purely for back-compat with this driver's old callers.)

(This file MERGES the former ``nsys_trimul_e2e.py`` — vs the reference — INTO this driver. ``nsys_trimul_e2e.py``
is now a thin shim that calls this with ``--baseline dtensor_baseline_ring_reducescatter``. Filename kept ``*_dtensor`` for its live
callers ``run_2d_incoming_grid.py`` / ``run_route2_ni_grid.py`` and the ``assemble_all.py`` /
``assemble_2d.py`` [LAT]-stdout + ``fused[...]``/``dtensor[...]`` NVTX-tag parsers.)

Each (N, direction) is its own NVTX range for both paths (``fused[...]`` and ``dtensor[...]`` /
``dtensor_baseline_ring_reducescatter[...]``), so ``nsys profile --trace=cuda,nvtx --gpu-metrics-devices=all`` captures a single
timeline where per-range GPU-metrics (tensor-pipe %, NVLink TX/RX) are attributable per cell.

DE-INTERLEAVED loop (critical): per (N, direction) ALL fused iters run as a contiguous block, then a
cross-rank ``torch.distributed.barrier``, then ALL baseline iters as a block. Interleaving fused +
baseline PER ITER DESYNCS the ranks: the baseline's collective completes at slightly different times
per rank and ``torch.cuda.synchronize`` is rank-LOCAL (no cross-rank barrier), so the next fused
iter's in-kernel nvshmem ``barrier_on_stream`` absorbs the drift as stall-time — inflating the fused
median ~8x (a MEASUREMENT artifact, not kernel cost). Each path as a block stays in lockstep via its
OWN collectives; the barrier separates the blocks. (``none`` needs no barrier — fused self-syncs.)

Invalid cells (guard violations: N%cp, N%8, 2-D per-axis %8, D%cp) are printed as ``N/A(reason)`` and
SKIPPED (never crash the sweep). The anchor is the FIRST valid N for this (D, cp).

INCOMING pays the (C) ``.contiguous()`` on the back operand (the pe_aligned mainloop_remap_mA
a_major="m" workaround, task #16) — the SHIPPED path, honest to profile; W4 notes it + (B) as the opt.

Run (per (D,cp); cp=2 on GPUs 4,5; CPO_CACHE_ENABLED=0 NVSHMEM_DISABLE_NVLS=1)::

    CUDA_VISIBLE_DEVICES=4,5 CPO_CACHE_ENABLED=0 NVSHMEM_DISABLE_NVLS=1 \
      CPO_DIST_MESH=cp=2 NVSHMEM_SYMMETRIC_SIZE=17179869184 PYTHONPATH=$PWD \
      nsys profile --trace=cuda,nvtx --gpu-metrics-devices=all --sample=none \
        --capture-range=cudaProfilerApi --capture-range-end=stop --force-overwrite=true \
        -o profiling/distributed/trimul/trimul_H200_NV18_D256_cp2 \
        python -m torch.distributed.run --nproc_per_node=2 \
        benchmark/distributed/nsys_trimul_dtensor.py --D 256 --cp 2 --baseline dtensor_baseline_ring_reducescatter

The ``[LAT]`` stdout lines (per N,direction: fused_ms, <baseline>_ms) + ``[NA]`` lines feed the
summary assemblers (assemble_all.py / assemble_2d.py) alongside the nsys-stats GPU-metrics.
"""

import argparse
import contextlib
import os
from collections import OrderedDict

import torch
from torch.distributed.tensor import Shard, distribute_tensor
from torch.distributed.device_mesh import init_device_mesh


@contextlib.contextmanager
def _nvtx(name):
    torch.cuda.nvtx.range_push(name)
    try:
        yield
    finally:
        torch.cuda.nvtx.range_pop()


def _all_reduce_max(value: float) -> float:
    """Reduce a per-rank scalar to the SLOWEST PE's value.

    The true latency of a multi-process workflow is its slowest participant, not any one rank's local
    view (CLAUDE.md, and `bench_utils`' ``reduce="max"``). Kept as a tiny local helper rather than an
    import so this driver's dependency surface is unchanged.

    NaN-safe: a fused-only cell has ``bmed = nan`` and ``ReduceOp.MAX`` would propagate it to every
    rank, so nan is passed straight through without touching the collective. That also keeps the
    number of collectives IDENTICAL across ranks -- a rank that skipped the reduce while others
    entered it would desync the group and hang, which is the failure this guard exists to prevent.
    """
    if value != value:  # nan
        return value
    if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
        return value
    t = torch.tensor([value], dtype=torch.float64, device="cuda")
    torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.MAX)
    return float(t.item())


def _valid(N, cp0, cp1, D, cp):
    """Guard check (mirrors TriMulAutotuned.__init__): returns (ok, reason)."""
    if D % cp != 0:
        return False, f"D%cp!=0 (D={D},cp={cp})"
    if N % cp0 != 0 or N % cp1 != 0:
        return False, f"N%cp!=0 (N={N},cp0={cp0},cp1={cp1})"
    if N % 8 != 0:
        return False, f"N%8!=0 (N={N})"
    if cp1 > 1 and ((N // cp0) % 8 != 0 or (N // cp1) % 8 != 0):
        return False, f"2-D per-axis %8 (N_i={N // cp0},N_j={N // cp1})"
    return True, ""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--D", type=int, required=True)
    ap.add_argument("--cp", type=int, default=None, help="1-D cp world size")
    ap.add_argument("--mesh2d", type=str, default=None, help="2-D 'cp0,cp1'")
    ap.add_argument("--Ns", type=str, default="1000,2000,4000")
    ap.add_argument("--B", type=int, default=1)
    ap.add_argument("--consumer", default="stagec")
    ap.add_argument(
        "--direction",
        default="both",
        choices=("outgoing", "incoming", "both"),
        help="which direction(s) to profile; 'both' sweeps outgoing+incoming (default).",
    )
    ap.add_argument(
        "--baseline",
        default="dtensor_baseline_a2a",
        choices=("none", "dtensor_baseline_a2a", "dtensor_baseline_ring_reducescatter"),
        help="baseline path profiled alongside fused: 'none' (fused-only), 'dtensor' "
        "(DTensor redistribute all-to-all — the CORRECTNESS ref, historical default), "
        "'dtensor_baseline_ring_reducescatter' (the reference CP native ring — THE PERF-GRID baseline; the profiling "
        "grid passes this explicitly). De-interleaved from fused.",
    )
    ap.add_argument(
        "--warmup",
        type=int,
        default=8,
        help="un-profiled; burns baseline cold specs + fused rebind",
    )
    ap.add_argument("--iters", type=int, default=5, help="profiled iters per (N,direction,path)")
    ap.add_argument("--seed", type=int, default=20260622)
    ap.add_argument(
        "--route2_ni",
        action="store_true",
        help="route-2 (A) no-copy INCOMING: front writes N_i-stride-1 recv so the back reads "
        "a_major='k' NATIVE (NO .transpose().contiguous()). 1-D + B==1 only; OUTGOING "
        "unchanged (bit-identical to plain). The static route2_ni front recompiles per N.",
    )
    ap.add_argument(
        "--composite_k",
        action="store_true",
        help="composite-K FAST incoming (the 2.08x fix, §9.4; SUPERSEDES the slow route2_ni): "
        "config-only D-major per-rank-contiguous recv (32KB puts, no b_j-pin Xg cap) + "
        "GemmSm90A2A composite-K back read K=(cp,Xg_pad). 1-D + B==1; OUTGOING byte-identical "
        "(incoming-only, default-off). Mutually exclusive with --route2_ni.",
    )
    ap.add_argument(
        "--dynamic",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="DYNAMIC-shape TriMulAutotuned (compile ONCE at the anchor N, run any N — REQUIRED "
        "for a multi-N --Ns sweep; the grid callers rely on this default). --no-dynamic = "
        "STATIC (dynamic_shape=False; bakes the shape -> SINGLE --Ns only) — faithful to "
        "the old nsys_trimul_e2e single-N path (the e2e shim passes --no-dynamic). Static "
        "is the byte-identical production kernel; perf-neutral vs dynamic at fixed cp.",
    )
    ap.add_argument(
        "--no-dtensor",
        dest="no_dtensor",
        action="store_true",
        help="DEPRECATED alias for '--baseline none' (kept for run_route2_ni_grid.py / "
        "run_2d_incoming_grid.py which pass it). Wins over --baseline if both given.",
    )
    ap.add_argument(
        "--a2a-fused-dtensor-api",
        dest="a2a_fused_dtensor_api",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="G7 FAIRNESS: profile the SHIPPED DTensor API TriangularMultiplication(x_dt, mask_dt)."
        ".to_local() — DTensor-in/DTensor-out, paying the from_local/to_local host "
        "overhead the reference CP baseline ALSO pays (DEFAULT — the fair DTensor-vs-DTensor "
        "head-to-head). --no-dtensor-api reverts to the RAW TriMulAutotuned on the local "
        "shard (legacy raw-vs-DTensor path).",
    )
    ap.add_argument(
        "--mask",
        dest="mask",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Build a ones pair-mask for the fused path (DEFAULT ON — the reference CP baseline "
        "ALWAYS runs mask-ON, so mask-ON is the apples-to-apples headline; the fused "
        "epilogue mask is ~free). --no-mask disables it (legacy mask-off reference).",
    )
    args = ap.parse_args()
    # `--no-dynamic` has nowhere to land on the DTensor path: TriangularMultiplication is ALWAYS
    # dynamic (one compiled instance per direction serves every N -- see its docstring). REFUSED
    # rather than accepted-and-ignored: a silently dropped flag reports a static run that never ran.
    assert not (args.a2a_fused_dtensor_api and not args.dynamic), (
        "--no-dynamic is incompatible with --a2a-fused-dtensor-api: TriangularMultiplication is always "
        "dynamic. Use --no-dynamic WITHOUT --a2a-fused-dtensor-api (the raw TriMulAutotuned path, which does "
        "take a static shape), or drop --no-dynamic."
    )
    assert not (args.route2_ni and args.composite_k), (
        "--route2_ni and --composite_k are mutually exclusive"
    )

    # Resolve the baseline (--no-dtensor is the deprecated 'none' switch; it wins over --baseline).
    baseline = "none" if args.no_dtensor else args.baseline
    # NVTX-tag + [LAT]-column label. 'none'/'dtensor' -> "dtensor" (BYTE-COMPAT with the parsers:
    # 'none' emits dtensor_ms=nan); 'dtensor_baseline_ring_reducescatter' -> "dtensor_baseline_ring_reducescatter" (non-colliding). See module docstring.
    base_label = "dtensor_baseline_ring_reducescatter" if baseline == "dtensor_baseline_ring_reducescatter" else "dtensor_baseline_a2a"
    directions = ["outgoing", "incoming"] if args.direction == "both" else [args.direction]

    world = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", os.environ["RANK"]))
    torch.cuda.set_device(local_rank)
    os.environ.setdefault("CPO_DISTRIBUTED_INIT_METHOD", "ENV")

    if args.mesh2d:
        cp0, cp1 = (int(x) for x in args.mesh2d.split(","))
    else:
        cp0, cp1 = (args.cp or world), 1
    cp = cp0 * cp1
    assert cp == world, f"cp={cp} != world={world}"
    Ns = [int(x) for x in args.Ns.split(",")]
    B, D, dt = args.B, args.D, torch.bfloat16
    cp_label = f"{cp0}x{cp1}" if cp1 > 1 else f"{cp0}"

    from fold_cp_ops.distributed import DistributedManager
    import datetime

    cp_spec = (cp0, cp1) if cp1 > 1 else cp0
    # PG collective timeout 30min (torch default ~10min): a ROBUSTNESS guard against a transient WALL-CLOCK
    # DESYNC at cold start -- fast ranks finish the ~16-28s cold construct + race into the warmup collective
    # while a straggler lags -> the fast ranks blow NCCL's ~10min watchdog -> rc=143 (seen ONCE at cp16/N2000).
    # The longer timeout lets them wait through it (does NOT corrupt profiled iters, unlike warm-cache).
    # NOTE (2026-07-25, verified on-venue + full-TriMulAutotuned cute.compile census): this is NOT a ~20min compile
    # cliff. Steady-state cold construct is ~16s single-rank / ~23-28s @16-rank, LOW variance, on BOTH venue B
    # (default-cache, cold, 1:45) and venue E (1:48 at the full sweep config); census = NO hidden
    # compile, max single-kernel ~5-9s (<=5s rule satisfied, the reference_user_override_autotune_5s_compile_bar
    # accepted cost), ptxas 13.2.78 byte-identical conda/enroot. The one-time ~20min/rc=143 was a TRANSIENT
    # wall-clock desync -- DGX clock skew (measured ~14.6min on the venue E repro, which itself COMPLETED
    # cleanly at 1:48) + cold-compile variance + fused/baseline interleaving tripped the ~10min watchdog on the
    # ORIGINAL venue B run; PG-timeout + de-interleaving make it robust. NOT a persistent compile -> no
    # kernel/config fix. See memory reference_cp16_cliff_not_mlir_gen_sm120_verdict.
    DistributedManager.initialize(
        OrderedDict(cp=cp_spec),
        device_type="cuda",
        backend="nccl",
        timeout=datetime.timedelta(minutes=30),
    )
    dm = DistributedManager()
    DistributedManager.init_nvshmem()
    dev = dm.device

    from fold_cp_ops.distributed.workflows.trimul_autotuned import TriMulAutotuned
    from fold_cp_ops.distributed.trimul_weights import trimul_module_from_weights
    from fold_cp_ops.distributed.workflows.trimul_autotuned import TriangularMultiplication
    from fold_cp_ops.distributed.workflows.trimul_tuning import (
        OutGateTuning,
        TriMulTuning,
    )
    from fold_cp_ops.distributed.layout_map import LayoutRightMap
    from fold_cp_ops.distributed.pe_map import PeMap
    from tests.distributed.correctness_harness import make_weights

    weights = make_weights(D, seed=args.seed, device=dev)

    # FUSED cp mesh (subgroup if 2-D) + pe_map (mirror the e2e test).
    sub = getattr(dm, "device_mesh_subgroups", None)
    fmesh = sub if (getattr(dm, "has_subgroups", False) and sub is not None) else dm.device_mesh
    fpl = [Shard(i + 1) for i in range(fmesh.ndim)]
    pm = PeMap.from_mesh_placements(fmesh, fpl, distributed_manager=dm)
    axis = tuple(int(s) for s in pm.cp_axis_sizes)
    unravel = LayoutRightMap(axis)
    coord = unravel.unravel(int(pm.my_cp_rank))
    ib = coord[0]
    jb = coord[1] if len(coord) > 1 else 0

    # BASELINE setup (collective; ALL ranks build identically). Only the selected baseline is built.
    distributed_trimul_fwd = None
    dmesh = token_pl = None
    dtensor_baseline_ring_reducescatter_modules = {}  # {direction: (module, mesh, placements, n_axes)}
    if baseline == "dtensor_baseline_a2a":
        from benchmark.distributed.t2_0_dtensor_baseline import distributed_trimul_fwd

        # DTensor (dp,cp0,cp1) mesh + token placements (the §5 ref-b convention).
        dmesh = init_device_mesh("cuda", (1, cp0, cp1), mesh_dim_names=("dp", "cp0", "cp1"))
        token_pl = [Shard(0), Shard(1), Shard(2)]
    elif baseline == "dtensor_baseline_ring_reducescatter":
        from benchmark.distributed.trimul_dtensor_baseline import (
            build_trimul_dtensor_baseline,
            make_trimul_dtensor_baseline_local_fn_sharded,
        )

        # N-agnostic native reference module per direction (collective init_device_mesh inside).
        for d in directions:
            dtensor_baseline_ring_reducescatter_modules[d] = build_trimul_dtensor_baseline(dm, D=D, direction=d, weights=weights, dt=dt)

    # Valid-N filter + N/A reporting; anchor = first valid N.
    valid, na = [], []
    for N in Ns:
        ok, reason = _valid(N, cp0, cp1, D, cp)
        (valid if ok else na).append((N, reason))
    if dm.rank == 0:
        for N, reason in na:
            for direction in directions:
                print(f"[NA] D={D} cp={cp_label} N={N} dir={direction} reason={reason}", flush=True)
    if not valid:
        if dm.rank == 0:
            print(f"[skip] D={D} cp={cp_label}: no valid N in {Ns}", flush=True)
        DistributedManager.cleanup()
        return 0
    anchor_N = valid[0][0]
    # STATIC (--no-dynamic) bakes the shape at the anchor -> it can only run that ONE N. Guard a
    # multi-N static sweep (would run the anchor-baked kernel at the wrong N). Dynamic has no such limit.
    if not args.dynamic and len(valid) > 1:
        if dm.rank == 0:
            print(
                f"[skip] D={D} cp={cp_label}: --no-dynamic (static) bakes the shape -> a SINGLE valid N "
                f"only; got {[N for N, _ in valid]}. Use --dynamic for a multi-N sweep.",
                flush=True,
            )
        DistributedManager.cleanup()
        return 0

    # Explicit incoming-store overrides (default: let TriangularMultiplication AUTO-select the fast
    # incoming path —
    # composite_k for 1-D, route2_ni for 2-D — exactly as the shipped API does).
    fused_kw = {}
    if args.route2_ni:
        fused_kw["route2_ni"] = True
    if args.composite_k:
        fused_kw["composite_k"] = True

    # NVTX-mark the (dynamic) construct so a whole-process nsys capture (no --capture-range) shows the
    # construction / cute.compile phase as a named range — "collect the dynamic compilation". With the
    # DTensor API (default) TriangularMultiplication compiles LAZILY on first forward (in warmup,
    # un-profiled); the
    # RAW path compiles eagerly here. Either way compile is OUTSIDE the cudaProfilerApi capture.
    ftm = None
    fused_modules = {}  # {direction: TriangularMultiplication}  (a2a-fused DTensor-API path; one per direction)
    with _nvtx(f"ftm_construct_compile[{'dynamic' if args.dynamic else 'static'}]"):
        if args.a2a_fused_dtensor_api:
            # FAIRNESS: the SHIPPED DTensor API. direction is baked per module, so build one per
            # direction. dynamic=True => ONE compiled instance/direction serves all N. mask-ON keys
            # has_mask=True internally. AUTO transport (hybrid_ib default) + AUTO incoming fast path.
            # `--route2_ni` / `--composite_k` force a DERIVED choice, and the shipped module no
            # longer accepts one: `TriangularMultiplication` auto-selects the incoming store from
            # (direction, cp1). Refuse loudly rather than silently profiling the auto pick under a
            # flag that says otherwise -- and the combination is self-contradictory anyway, since
            # `--a2a-fused-dtensor-api` exists to profile what the SHIPPED API selects. The RAW path below
            # still takes both overrides.
            if fused_kw:
                raise SystemExit(
                    f"--a2a-fused-dtensor-api cannot force an incoming store: {sorted(fused_kw)} are derived "
                    f"from (direction, cp1) and the shipped TriangularMultiplication does not "
                    f"expose them. Use --no-dtensor-api to override them on the raw TriMulAutotuned "
                    f"path, or drop the flag to profile what the shipped API actually picks."
                )
            for d in directions:
                fused_modules[d] = TriangularMultiplication(
                    trimul_module_from_weights(weights),
                    d,
                    fmesh,
                    dm,
                    dtype=dt,
                    tuning=TriMulTuning(out_gate=OutGateTuning(consumer=args.consumer)),
                )
        else:
            ftm = TriMulAutotuned(
                pm,
                B,
                anchor_N,
                D,
                weights,
                dt,
                consumer=args.consumer,
                device_mesh=fmesh,
                placements=fpl,
                dynamic=args.dynamic,
                route2_ni=args.route2_ni,
                composite_k=args.composite_k,
            )

    # Per-N inputs. ONE global x per N (same on every rank) -> fused slices its native block; the
    # DTensor baseline distributes it; the reference CP baseline distributes it onto ITS native mesh.
    # baseline==none: build THIS rank's local block DIRECTLY (no full global x_g) — halves the peak
    # and lets the least-sharded D512/cp2 cells fit (profiling needs no oracle-consistency; correctness
    # is gated by tests/distributed/test_trimul_route2_ni.py). The returned ``payload`` is what the
    # baseline call needs: None (none) / x_dt (dtensor) / x_global (dtensor_baseline_ring_reducescatter — kept for its own distribute).
    def make_inputs(N):
        N_i, N_j = N // cp0, N // cp1
        if baseline in ("none", "dtensor_baseline_ring_reducescatter"):
            # SHARDED-ONLY (real-app premise): build THIS rank's local block directly, NEVER the full
            # global (B,N,N,D). Both the fused AND reference workflows hold only sharded tensors in
            # production, so the probe/profiling max-N reflects the TRUE per-rank footprint (the old
            # the reference path built a cp-INDEPENDENT global on every rank -> understated max N and flattened
            # the cp16-vs-cp8 gap). the reference wraps this same shard via DTensor.from_local
            # (make_trimul_dtensor_baseline_local_fn_sharded); its native shard shape matches xl exactly. Random.
            gx = torch.Generator(device="cpu").manual_seed(args.seed + dm.rank)
            xl = torch.randn(B, N_i, N_j, D, generator=gx, dtype=torch.float32).to(dt).to(dev)
            return xl, (xl if baseline == "dtensor_baseline_ring_reducescatter" else None)
        # dtensor = the CORRECTNESS reference (NOT the perf baseline) -> still distributes a global.
        gx = torch.Generator(device="cpu").manual_seed(args.seed)
        xg = torch.randn(B, N, N, D, generator=gx, dtype=torch.float32).to(dt).to(dev)
        xl = xg[:, ib * N_i : (ib + 1) * N_i, jb * N_j : (jb + 1) * N_j, :].contiguous()
        if baseline == "dtensor_baseline_a2a":
            x_dt = distribute_tensor(xg, dmesh, token_pl)
            del xg
            return xl, x_dt
        # dtensor_baseline_ring_reducescatter: keep the global x to distribute onto the reference native mesh (per-direction fn below).
        return xl, xg

    def make_base_fn(direction, payload):
        """Return the no-arg baseline callable for this (direction, per-N payload), or None."""
        if baseline == "none":
            return None
        if baseline == "dtensor_baseline_a2a":
            x_dt = payload
            return lambda: distributed_trimul_fwd(x_dt, None, weights, direction, dt).to_local()
        module, bmesh, bpl, _ = dtensor_baseline_ring_reducescatter_modules[direction]
        return make_trimul_dtensor_baseline_local_fn_sharded(module, bmesh, bpl, payload, dt)

    from torch.distributed.tensor import DTensor

    def _rowmaj_stride(shape):
        st = [1] * len(shape)
        for i in range(len(shape) - 2, -1, -1):
            st[i] = st[i + 1] * int(shape[i + 1])
        return tuple(st)

    def _dt_from_local(local, global_shape):
        """Wrap THIS rank's local shard as a DTensor on the fused mesh/placements WITHOUT allocating the
        O(N²) global (mirror fused_trimul_cp / trimul_e2e). Explicit global shape+stride => no collective."""
        gs = torch.Size(tuple(int(s) for s in global_shape))
        return DTensor.from_local(
            local.contiguous(),
            fmesh,
            list(fpl),
            shape=gs,
            stride=_rowmaj_stride(gs),
            run_check=False,
        )

    def make_fused_fn(direction, xl, N):
        """No-arg FUSED callable for (direction, xl, N). --a2a-fused-dtensor-api (DEFAULT): wrap xl (+ ones mask)
        as DTensors and drive the SHIPPED TriangularMultiplication(x_dt, mask_dt).to_local() —
        DTensor-in/out, the
        fair head-to-head vs the reference baseline's DTensor path. --no-dtensor-api: the RAW TriMulAutotuned on the local
        shard. mask-ON by default for BOTH (the reference baseline always runs mask-ON)."""
        N_i, N_j = int(xl.shape[1]), int(xl.shape[2])
        if args.a2a_fused_dtensor_api:
            x_dt = _dt_from_local(xl, (B, N, N, D))
            mask_dt = None
            if args.mask:
                ml = torch.ones(B, N_i, N_j, device=xl.device, dtype=dt)
                mask_dt = _dt_from_local(ml, (B, N, N))
            mod = fused_modules[direction]
            return lambda: mod(x_dt, mask_dt).to_local()
        mask_l = torch.ones(B, N_i, N_j, device=xl.device, dtype=dt) if args.mask else None
        return lambda: ftm.forward(xl, direction, mask_local=mask_l)

    # MEMORY: build+free inputs PER-N (NOT pre-cached for all N) — pre-caching all N's x_global (bf16)
    # + the baseline shard OOMs the GPU at heavy cells (D512/N4000), and a one-rank OOM hangs the rest
    # on a collective. Peak now = ONE N's footprint. Warmup (per-N, freed) burns each N's cold baseline
    # sharding-prop spec (process-level cache persists across the free) + the fused rebind.
    for N, _ in valid:
        xl, payload = make_inputs(N)
        for (
            direction
        ) in directions:  # per-direction: keep at most ONE baseline shard alive (frugal)
            fused_fn = make_fused_fn(direction, xl, N)
            base_fn = make_base_fn(direction, payload)
            for _ in range(args.warmup):
                fused_fn()
                if base_fn is not None:
                    base_fn()
            del fused_fn, base_fn
        del xl, payload
        torch.cuda.empty_cache()
    torch.cuda.synchronize(dev)
    torch.distributed.barrier(device_ids=[local_rank])

    import statistics

    rows = []
    torch.cuda.profiler.start()  # profiled-only capture (small trace); per-N rebuild (warm caches) + free
    for N, _ in valid:
        xl, payload = make_inputs(N)
        for direction in directions:
            tag = f"N{N}_D{D}_cp{cp_label}_{direction}"
            fused_fn = make_fused_fn(direction, xl, N)
            base_fn = make_base_fn(direction, payload)
            # DE-INTERLEAVE: cross-rank resync so the FUSED block starts in lockstep. Kept PER-CELL
            # (not only post-warmup) so EVERY fused block starts synced regardless of --baseline —
            # the pre-fused resync the e2e fix relied on.
            torch.distributed.barrier(device_ids=[local_rank])
            # ---- FUSED block: ALL iters contiguous (lockstep via the in-kernel nvshmem barrier). ----
            fused_ms = []
            for _ in range(args.iters):
                e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                with _nvtx(f"fused[{tag}]"):
                    e0.record()
                    fused_fn()
                    e1.record()
                    torch.cuda.synchronize(dev)
                fused_ms.append(e0.elapsed_time(e1))
            # ---- BASELINE block: DE-INTERLEAVED — resync, then ALL baseline iters as a block. ----
            base_ms = []
            if base_fn is not None:
                torch.distributed.barrier(
                    device_ids=[local_rank]
                )  # resync after fused, before baseline
                for _ in range(args.iters):
                    e2, e3 = (
                        torch.cuda.Event(enable_timing=True),
                        torch.cuda.Event(enable_timing=True),
                    )
                    with _nvtx(f"{base_label}[{tag}]"):
                        e2.record()
                        base_fn()
                        e3.record()
                        torch.cuda.synchronize(dev)
                    base_ms.append(e2.elapsed_time(e3))
            bmed = statistics.median(base_ms) if base_ms else float("nan")
            fmed = statistics.median(fused_ms)
            # SLOWEST-PE reduction, reported ALONGSIDE the rank-0 median rather than replacing it.
            #
            # Every rank measures its own 4 iters and takes its own median; only rank 0 prints, so the
            # historical `fused_ms` is ONE rank's local view. Under lockstep that equals the slowest
            # PE (identical durations => median == max) and the two agree. They diverge only under
            # rank SKEW: a collective cannot finish before its last participant arrives, so an EARLY
            # rank sits inside it and records a LONG duration while a LATE rank records a SHORT one.
            # Rank 0 then reports whichever position it happened to occupy -- neither best nor worst
            # case. `all_reduce(MAX)` is the honest multi-process cost (first-ready to finished) and
            # is what CLAUDE.md mandates for anything containing a collective.
            #
            # Emitting BOTH makes the divergence measurable in a single run: fmax/fmed ~= 1 proves the
            # ranks were in lockstep and the rank-0 number was already correct; a large ratio localises
            # the venue's skew. Replacing the column outright would have made historical numbers
            # incomparable and hidden exactly that check.
            fmax = _all_reduce_max(fmed)
            bmax = _all_reduce_max(bmed) if base_ms else float("nan")
            rows.append((N, direction, fmed, bmed, fmax, bmax))
        del xl, payload
        torch.cuda.empty_cache()
    torch.cuda.profiler.stop()
    torch.distributed.barrier(device_ids=[local_rank])

    if dm.rank == 0:
        for N, direction, fmed, bmed, fmax, bmax in rows:
            speedup = bmed / fmed if bmed == bmed else float("nan")  # bmed!=bmed => nan
            # The historical [LAT] line is UNCHANGED, byte-for-byte, so every existing parser
            # (assemble_all.py / assemble_2d.py / assemble_latency_summary.py) and every archived
            # summary.csv stays comparable. The slowest-PE reduction is a SEPARATE line.
            print(
                f"[LAT] D={D} cp={cp_label} N={N} dir={direction} "
                f"fused_ms={fmed:.4f} {base_label}_ms={bmed:.4f} speedup={speedup:.3f}",
                flush=True,
            )
            # skew = slowest-PE / rank-0-median. ~1.00 proves the ranks ran in lockstep, so the
            # rank-0 number on the line above IS the slowest-PE number and the historical
            # measurement was sound. A large ratio means rank 0's local view understated the real
            # multi-process cost, and by exactly this factor.
            smax = bmax / fmax if bmax == bmax else float("nan")
            fskew = fmax / fmed if fmed else float("nan")
            bskew = (bmax / bmed) if (bmed == bmed and bmed) else float("nan")
            print(
                f"[MAXPE] D={D} cp={cp_label} N={N} dir={direction} "
                f"fused_ms={fmax:.4f} {base_label}_ms={bmax:.4f} speedup={smax:.3f} "
                f"fused_skew={fskew:.3f} {base_label}_skew={bskew:.3f}",
                flush=True,
            )
        # Config readout: RAW path reads ftm directly; DTensor-API path pulls a cached TriMulAutotuned
        # instance out of a module's cache (built during warmup).
        if args.a2a_fused_dtensor_api:
            _inst = next(iter(fused_modules[directions[0]]._engines.values()), None)
            _front_cfg = getattr(_inst, "_front_cfg", None)
            _back_cfg = getattr(_inst, "_back_cfg", None)
        else:
            _front_cfg, _back_cfg = ftm._front_cfg, ftm._back_cfg
        print(
            f"[done] D={D} cp={cp_label} a2a_fused_dtensor_api={args.a2a_fused_dtensor_api} mask={args.mask} "
            f"route2_ni={args.route2_ni} baseline={baseline} dynamic={args.dynamic} "
            f"direction={args.direction} front_cfg={_front_cfg} back_cfg={_back_cfg} "
            f"anchor_N={anchor_N} valid={[N for N, _ in valid]}",
            flush=True,
        )

    if args.a2a_fused_dtensor_api:
        for m in fused_modules.values():
            m.free()
    else:
        ftm.free()
    DistributedManager.cleanup()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
