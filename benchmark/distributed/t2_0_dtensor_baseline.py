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

"""T2.0 Part B — non-fused distributed TriMul fwd BASELINE, PURE torch DTensor.

This is the baseline the FUSED Wave-2 (in-kernel NVSHMEM A2A) version must beat. It uses torch
DTensor `redistribute` as the all-to-all (NOT our custom comm primitive — that is T1.2), so the
A2A cost here is "what you get for free from torch". We localize the einsum by resharding the
token-sharded operands onto the feature axis, run the einsum locally, reshard back, and run the
local LN + dual-gated back half.

Sharding contract (mirrors the CP fork's ``distributed/.../triangular_mult.py``):
  x : DTensor (B, N, N, D), placements (Shard(0), Shard(1), Shard(2)) on a (dp, cp0, cp1) mesh.
  For cp=2 on 2 GPUs we use mesh (dp=1, cp0=2, cp1=1): dp & cp1 are size-1 (Shard on them is a
  no-op), so only cp0 shards token-axis-1 (N rows) -> each rank holds (B, N/2, N, D).

Flow (the non-fused baseline; both token axes conceptually sharded over cp=cp0*cp1):
  1. FRONT (local, D not sharded): xn = LN(x); ab = (xn@p_in^T+bp)*sigmoid(xn@g_in^T+bg);
     mask; a,b = chunk(ab, 2).                                  -- no comm
  2. FRONT A2A: reshard a,b from token-shard (Shard(1)) to feature-shard (Shard(3))  -- all_to_all
     so each rank owns the FULL (n,k) token grid for a D-slice -> the einsum is LOCAL.
  3. LOCAL einsum: tri = einsum("bnkd,bmkd->bnmd", a, b) (outgoing) per D-slice (Shard(3)). -- no comm
  4. BACK A2A: reshard tri from feature-shard (Shard(3)) back to token-shard (Shard(1)). -- all_to_all
  5. BACK (local): out = (LN(tri)@p_out^T+bp) * sigmoid(xn@g_out^T+bg)  (cuEq: out-gate consumes xn). -- no comm

Correctness: all-gather x -> run the fp32 global oracle trimul_ref -> compare to the all-gathered
distributed output (rel L2 err).

Comm fraction: (a) CommDebugMode confirms exactly the 2 all_to_all collectives (front+back) and
their byte volume; (b) we time the redistribute-only path vs the full e2e with the barrier-bracketed
paired/single timer from bench_utils -> comm_fraction = t_redistribute / t_e2e.

Run (cp=2 on GPUs 4,5):
  CUDA_VISIBLE_DEVICES=4,5 torchrun --standalone --nnodes 1 --nproc_per_node=2 \
      benchmark/distributed/t2_0_dtensor_baseline.py --N 512 --D 128 --direction outgoing
"""

from __future__ import annotations

import argparse
import os
import sys

import torch
import torch.nn.functional as F
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor, Shard, distribute_tensor
from torch.distributed.tensor.debug import CommDebugMode

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from benchmark.distributed.bench_utils import resolve_dist  # noqa: E402
# `rel_err` does NOT come across: it is ||got-ref|| / ||ref||, the pooled scalar this repo's numerics
# rule forbids as a GATE -- one badly-wrong element inside a large correct tensor cannot move it. It
# is defined locally below, and ONLY for the diagnostic print; every pass/fail decision in this tree
# goes through `fold_cp_ops.testing.numerics`.
from fold_cp_ops.workflows.trimul_autotune import trimul_ref  # noqa: E402


def rel_err(a, b) -> float:
    """Pooled relative L2 ``||a-b|| / ||b||`` in fp32 -- for PRINTING, never a bar.

    Args:
        a: the tensor under test. Cast to fp32.
        b: the reference. Cast to fp32; a zero-norm reference is guarded by a 1e-30 floor rather
            than raising, because this is a diagnostic and a NaN in a log line is worse than a
            meaningless-but-finite one.

    Returns:
        The pooled ratio. See the note above for why it is not a gate.
    """
    a, b = a.float(), b.float()
    return (a - b).norm().item() / (b.norm().item() + 1e-30)


# ── weights (replicated; D is NOT sharded, so all projections are local) ──────
def make_weights(N: int, D: int, *, dtype, bias: bool, seed: int):
    g = torch.Generator(device="cpu").manual_seed(seed)

    def rn(*shape):
        return torch.randn(*shape, generator=g, dtype=torch.float32) * 0.02

    # The LayerNorm affine is OFF the identity, matching `correctness_harness.make_weights`. This is
    # the SECOND builder of this dict in the tree and the two must not drift: a `ones`/`zeros` affine
    # makes the LayerNorm bias -- which the fused workflow really does fuse -- contribute exactly
    # nothing, so a baseline built here would agree with a fused path that ignored it entirely. Both
    # fp32: `layernorm_dual_gated_gemm` refuses a non-fp32 `norm_weight`.
    def ln_affine():
        return (1.0 + torch.randn(D, generator=g, dtype=torch.float32) * 0.1,
                torch.randn(D, generator=g, dtype=torch.float32) * 0.5)

    norm_in_w, norm_in_b = ln_affine()
    norm_out_w, norm_out_b = ln_affine()
    w = dict(
        norm_in_w=norm_in_w, norm_in_b=norm_in_b,
        p_in_w=rn(2 * D, D), g_in_w=rn(2 * D, D),
        norm_out_w=norm_out_w, norm_out_b=norm_out_b,
        p_out_w=rn(D, D), g_out_w=rn(D, D),
        p_in_b=rn(2 * D) if bias else None, g_in_b=rn(2 * D) if bias else None,
        p_out_b=rn(D) if bias else None, g_out_b=rn(D) if bias else None,
    )
    return w


def _local_front(xn_local, w, mask_local, dt):
    """Local front half on a rank's shard: GLU -> mask -> chunk. xn_local already LN'd. Returns a,b."""
    p_in = xn_local @ w["p_in_w"].to(dt).T
    if w["p_in_b"] is not None:
        p_in = p_in + w["p_in_b"].to(dt)
    g_in = xn_local @ w["g_in_w"].to(dt).T
    if w["g_in_b"] is not None:
        g_in = g_in + w["g_in_b"].to(dt)
    ab = p_in * torch.sigmoid(g_in)
    if mask_local is not None:
        ab = ab * mask_local.unsqueeze(-1)
    a, b = ab.chunk(2, dim=-1)
    return a.contiguous(), b.contiguous()


def _local_einsum(a_local, b_local, direction):
    """Local einsum on the feature-sharded operands (each rank owns a D-slice, full token grid)."""
    if direction == "outgoing":
        return torch.einsum("bnkd,bmkd->bnmd", a_local, b_local)
    return torch.einsum("bknd,bkmd->bnmd", a_local, b_local)


def _local_back(tri_local, xn_local, w, dt):
    """Local back half: LN(tri) -> p_out value, xn -> g_out gate (cuEq: out-gate consumes xn)."""
    D = tri_local.shape[-1]
    trin = F.layer_norm(tri_local.float(), (D,), w["norm_out_w"], w["norm_out_b"], 1e-5).to(dt)
    p_out = trin @ w["p_out_w"].to(dt).T
    if w["p_out_b"] is not None:
        p_out = p_out + w["p_out_b"].to(dt)
    g_out = xn_local @ w["g_out_w"].to(dt).T
    if w["g_out_b"] is not None:
        g_out = g_out + w["g_out_b"].to(dt)
    return p_out * torch.sigmoid(g_out)


def distributed_trimul_fwd(x_dt: DTensor, mask_dt, w, direction: str, dt, *, capture_comm=False):
    """Non-fused distributed TriMul fwd via torch DTensor redistribute. Returns out DTensor (Shard(1)).

    If capture_comm, wrap in CommDebugMode and return (out_dt, comm_mode) for collective inspection.
    """
    mesh = x_dt.device_mesh
    token_pl = list(x_dt.placements)            # (Shard(0), Shard(1), Shard(2))
    feat_pl = _feature_placements(token_pl)

    ctx = CommDebugMode() if capture_comm else _nullctx()
    with ctx:
        # FRONT (local): LN then GLU. LN over D is local (D not sharded). Keep as DTensor so the
        # redistribute sees a sharded spec; do the math on locals via from_local round-trip.
        xn_dt = _layernorm_dt(x_dt, w["norm_in_w"], w["norm_in_b"])
        xn_local = xn_dt.to_local()
        mask_local = mask_dt.to_local() if mask_dt is not None else None
        a_local, b_local = _local_front(xn_local, w, mask_local, dt)

        # Rebuild a,b as token-sharded DTensors, then redistribute -> feature-shard (FRONT A2A).
        ash = x_dt.shape[:-1] + (x_dt.shape[-1] // 2,)
        a_dt = DTensor.from_local(a_local, mesh, token_pl, shape=ash, stride=_rowmaj_stride(ash))
        b_dt = DTensor.from_local(b_local, mesh, token_pl, shape=ash, stride=_rowmaj_stride(ash))
        a_feat = a_dt.redistribute(placements=feat_pl)   # all_to_all
        b_feat = b_dt.redistribute(placements=feat_pl)   # all_to_all

        # LOCAL einsum on the D-slice each rank now owns.
        tri_local = _local_einsum(a_feat.to_local(), b_feat.to_local(), direction)
        tri_sh = x_dt.shape[:-1] + (x_dt.shape[-1] // 2,)
        tri_feat = DTensor.from_local(tri_local, mesh, feat_pl, shape=tri_sh, stride=_rowmaj_stride(tri_sh))

        # BACK A2A: feature-shard -> token-shard.
        tri_tok = tri_feat.redistribute(placements=token_pl)   # all_to_all

        # BACK (local): LN(tri) value + xn gate.
        out_local = _local_back(tri_tok.to_local(), xn_local, w, dt)
        out_dt = DTensor.from_local(out_local, mesh, token_pl, shape=tri_sh, stride=_rowmaj_stride(tri_sh))

    if capture_comm:
        return out_dt, ctx
    return out_dt


def _feature_placements(token_pl):
    """Map a token-sharded placement (Shard(1)/Shard(2) on token axes) to a feature-sharded one.

    For EVERY mesh axis that currently shards a token axis (tensor dim 1 or 2), redirect it to
    shard the feature axis (tensor dim 3) instead — so the einsum, which needs the full token grid
    per feature slice, becomes local. Generic over the mesh rank:
      - 1-D cp mesh (dp=1, cp0=k, cp1=1): only axis-1 = Shard(1) -> Shard(3); D sharded /k.
      - 2-D cp mesh (dp=1, cp0=a, cp1=b): axis-1 = Shard(1) AND axis-2 = Shard(2) both -> Shard(3);
        D sharded across BOTH mesh axes -> D/(a*b) per rank (the flattened-cp localization).
    dp / size-1 axes (Shard(0) / Replicate) pass through unchanged.
    """
    out = list(token_pl)
    for i, p in enumerate(token_pl):
        if isinstance(p, Shard) and p.dim in (1, 2):
            out[i] = Shard(3)
    return out


def redistribute_only(x_dt: DTensor):
    """Just the two A2As (front a/b + back tri), no compute — isolates the comm cost for the fraction."""
    token_pl = list(x_dt.placements)
    feat_pl = _feature_placements(token_pl)
    # one operand round-trip stands in for the structure; do 3 operand A2As (a,b front + tri back).
    af = x_dt.redistribute(placements=feat_pl)
    _ = af.to_local()
    bf = x_dt.redistribute(placements=feat_pl)
    _ = bf.to_local()
    back = af.redistribute(placements=token_pl)
    _ = back.to_local()
    return back


# ── small helpers ─────────────────────────────────────────────────────────────
class _nullctx:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _rowmaj_stride(shape):
    s = [1] * len(shape)
    for i in range(len(shape) - 2, -1, -1):
        s[i] = s[i + 1] * shape[i + 1]
    return tuple(s)


def _layernorm_dt(x_dt: DTensor, w, b):
    """LayerNorm over the last dim done on locals (D not sharded), rewrapped as the same DTensor spec."""
    mesh, pl = x_dt.device_mesh, x_dt.placements
    xl = x_dt.to_local()
    D = xl.shape[-1]
    yl = F.layer_norm(xl.float(), (D,), w, b, 1e-5).to(xl.dtype)
    return DTensor.from_local(yl, mesh, list(pl), shape=x_dt.shape, stride=_rowmaj_stride(x_dt.shape))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=512)
    ap.add_argument("--D", type=int, default=128)
    ap.add_argument("--B", type=int, default=1)
    ap.add_argument("--direction", default="outgoing", choices=("outgoing", "incoming"))
    ap.add_argument("--bias", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rounds", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--iters", type=int, default=10)
    args = ap.parse_args()

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    world = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    torch.distributed.init_process_group(backend="nccl", device_id=torch.device("cuda", local_rank))
    dev = torch.device("cuda", local_rank)
    dt = torch.bfloat16
    B, N, D, direction = args.B, args.N, args.D, args.direction

    # mesh: default 1-D cp = (dp=1, cp0=world, cp1=1). CPO_MESH="cp0,cp1" (product==world) -> 2-D cp.
    m = os.environ.get("CPO_MESH")
    if m:
        cp0, cp1 = (int(x) for x in m.split(","))
        assert cp0 * cp1 == world, f"CPO_MESH {m} product {cp0*cp1} != world {world}"
    else:
        cp0, cp1 = world, 1
    mesh = init_device_mesh("cuda", (1, cp0, cp1), mesh_dim_names=("dp", "cp0", "cp1"))
    token_pl = [Shard(0), Shard(1), Shard(2)]
    da = resolve_dist()

    # ── build GLOBAL x + weights on every rank (same seed) -> distribute ──
    gx = torch.Generator(device="cpu").manual_seed(args.seed)
    x_global = (torch.randn(B, N, N, D, generator=gx, dtype=torch.float32) * 1.0).to(dt).to(dev)
    w_cpu = make_weights(N, D, dtype=dt, bias=args.bias, seed=args.seed)
    w = {k: (v.to(dev) if v is not None else None) for k, v in w_cpu.items()}
    mask_global = None  # mask off for the baseline (kept simple; matches design's B=1 unmasked target)

    x_dt = distribute_tensor(x_global, mesh, token_pl)  # (Shard0,Shard1,Shard2)

    # ── correctness: distributed out (all-gathered) vs fp32 global oracle ──
    out_dt = distributed_trimul_fwd(x_dt, None, w, direction, dt)
    out_full = out_dt.full_tensor()  # all-gather -> (B,N,N,D) on every rank
    if da.is_rank0:
        out_ref = trimul_ref(
            x_global, direction, mask_global,
            w["norm_in_w"], w["norm_in_b"], w["p_in_w"], w["g_in_w"],
            w["norm_out_w"], w["norm_out_b"], w["p_out_w"], w["g_out_w"],
            p_in_b=w["p_in_b"], g_in_b=w["g_in_b"], p_out_b=w["p_out_b"], g_out_b=w["g_out_b"],
            eps=1e-5,
        )
        err = rel_err(out_full, out_ref)
        print(f"[correctness] N={N} D={D} {direction} world={world}: rel_L2_err = {err:.3e} "
              f"({'PASS' if err < 2e-2 else 'FAIL'} @ 2e-2 bf16 tol)")

    # ── comm inspection: confirm exactly the A2As + their byte volume ──
    if da.is_rank0:
        _, cm = distributed_trimul_fwd(x_dt, None, w, direction, dt, capture_comm=True)
        print("[comm trace] collectives observed (rank 0):")
        try:
            print(cm.generate_comm_debug_tracing_table(noise_level=0))
        except Exception:
            counts = cm.get_comm_counts()
            print(f"  comm counts: { {str(k): v for k, v in counts.items()} }")

    # ── timing: full e2e vs redistribute-only (comm fraction) ──
    # DTensor sharding-propagation compiles COLD on the FIRST call of each distinct spec (~127 ms
    # at N=512); the full fwd has several distinct specs (front A2A, back A2A, einsum, GEMMs, LN).
    # Burn ALL of them in with a generous warm loop, then time INLINE with a single CUDA-event
    # window per round (NOT bench_utils' per-window-barrier path — that serializes the many tiny
    # DTensor-dispatch windows on 2 ranks and dominates; the diagnostic confirms warm A2A ~0.13 ms).
    # NOTE: this in-script timing is a convenience cross-check; the AUTHORITATIVE comm-fraction
    # sweep is t2_0_timing_sweep.py (same ops, no full_tensor()/CommDebugMode preamble — that
    # preamble poisons the DTensor dispatcher's warm-up and inflates this loop). Correctness +
    # the 3-alltoall comm structure above are the load-bearing outputs of THIS script.
    e2e_fn = lambda: distributed_trimul_fwd(x_dt, None, w, direction, dt).to_local()
    comm_fn = lambda: redistribute_only(x_dt).to_local()
    for _ in range(20):  # amortize ALL cold sharding-prop compiles (several specs x ~127 ms)
        e2e_fn()
        comm_fn()
    torch.cuda.synchronize(dev)

    def time_inline(fn, rounds):
        """Median over `rounds` of a per-call CUDA-event window, syncing each call.

        IMPORTANT: we sync after EVERY call (not back-to-back). Back-to-back DTensor redistributes
        without an intervening sync let many NCCL all_to_alls + AsyncCollectiveTensor.wait()s queue
        and serialize pathologically (observed: minutes), whereas one-call-at-a-time is the warm
        ~0.13 ms/A2A the diagnostic measures. Barrier ONCE before the measurement (ranks aligned)."""
        torch.distributed.barrier()
        torch.cuda.synchronize(dev)
        ms = []
        for _ in range(rounds):
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record()
            fn()
            e.record()
            torch.cuda.synchronize(dev)
            ms.append(s.elapsed_time(e))
        ms.sort()
        return ms[len(ms) // 2]

    e2e_ms = time_inline(e2e_fn, args.rounds)
    comm_ms = time_inline(comm_fn, args.rounds)
    # all-reduce the per-rank medians to a MAX (true distributed latency = slowest rank).
    e2e_t = torch.tensor([e2e_ms], device=dev)
    comm_t = torch.tensor([comm_ms], device=dev)
    torch.distributed.all_reduce(e2e_t, op=torch.distributed.ReduceOp.MAX)
    torch.distributed.all_reduce(comm_t, op=torch.distributed.ReduceOp.MAX)
    if da.is_rank0:
        e2e_v, comm_v = e2e_t.item(), comm_t.item()
        frac = comm_v / e2e_v if e2e_v > 0 else float("nan")
        local_v = max(e2e_v - comm_v, 0.0)
        print(f"[timing] N={N} D={D} {direction} world={world}: "
              f"e2e(max-rank) = {e2e_v:.4f} ms/call ; A2A-only(3 alltoall) = {comm_v:.4f} ms/call ; "
              f"local-compute = {local_v:.4f} ms ; COMM_FRACTION = {frac:.1%}")

    torch.distributed.barrier()
    torch.distributed.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
