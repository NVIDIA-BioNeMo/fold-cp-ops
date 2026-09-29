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

"""Back-A2A STORE-VARIANT bench — footprint and time per store variant (docs/gemm_a2a_*).

Per N_token (cp=(2,8) 2-D shard, K=N square einsum) times the back-A2A store variants and reports each
variant's GMEM SCRATCH footprint (analytical + torch-measured).

HISTORY, because the file was called ``roundpark_footprint_bench`` until 2026-08-22 and its dispatch
table still names the variant. The bench was built to weigh ONE variant -- ``cluster_roundpark``, whose
cp1-full-buffer staging bought speed for a large scratch footprint -- against the alternatives. That
comparison is CLOSED: epic-close P2 made ``cluster_multislot`` the sole production IB drain and removed
``cluster_roundpark`` from ``configure_a2a_gemm_native``, so no shape or workflow variant can select it
and the footprint-vs-gain question it existed to answer can no longer be posed. The name survives in
``_VARIANT_TABLE`` ONLY so the CLI can refuse it with a reason instead of failing as an opaque
``TypeError``; it is deliberately absent from ``VARIANT_NAMES``, and
``tests/benchmark/distributed/test_back_a2a_store_bench.py`` asserts that absence four ways.

WHAT IT IS NOW, and why it did not become dead code with the variant. Two pieces are load-bearing for
the back A2A perf gate, which imports them directly:
  * ``_build_inputs_2d_kn`` -- the K==N operand builder. Every back A2A pin in
    ``tests/distributed/perf/test_benchmark_perf_gemm_sm90_a2a.json`` was measured through it.
  * ``_shape_check`` -- the (N, cp0, cp1) gate: token-shardability then the 16-B per-axis floor.
Consumers that take the builder without the gate get a truncated recv and an ``as_strided`` that
overruns it, which reads as a harness bug rather than an unsupported shape.

The variants:

  1. cluster_drain @ cluster_n=1  — degenerate 1-CTA cluster (single-buffer staging).
  2. cluster_drain @ cluster_n=2  — straddle => 2-slot MULTISLOT (rd-slot staging); even-shard => single-buffer.
  3. cluster_drain @ cluster_n=4  — straddle => REFUSED (this was round-park; the kernel parameter is gone);
                                    even-shard => single-buffer (small).
  4. cluster_drain @ cluster_n=8  — same shape-dispatch as (3).
  5. coalesce_dyn                 — the general per-CTA full-N band drain baseline (double-buffered ring).
  6. 2-kernel GEMM+NCCL A2A       — UNFUSED floor: cuBLAS batched GEMM (K=N einsum) -> local (L,N,N) tri in
                                    GMEM, then a SEPARATE torch.distributed all_to_all_single 2-D reshard.

DISPATCH BY SHAPE (the footprint comparison's crux). nt_j_pp = ceil((N/cp1)/tile_n):
  * cluster_n>=4 + STRADDLE (nt_j_pp % cluster_n != 0)  -> was cluster_roundpark=True; now REFUSED (removed).
  * cluster_n>=4 + EVEN-SHARD (nt_j_pp % cluster_n == 0) -> even-shard single-buffer (small).  NEVER force
    round-park on an even-shard point (that would hide the footprint tax — both would carry the big buffer).
  * cluster_n==2: straddle -> 2-slot multislot; even-shard -> single-buffer.
  * cluster_n==1: always single-buffer (nt_j_pp % 1 == 0 always; the degenerate 1-CTA cluster).

HARD RULES honored:
  * K = N_token (the square back-TriMul einsum M=N=K=N_token). Operands built HERE (assert K==N) — NO import
    of any tests/-side operand builder (the K=256 taint vector).  arbitrary_n=True so the ib_ring/cluster
    drain engages at ANY N (aligned or straddle N_i_loc) — verified vs cn4mse (N=8192 aligned) passing.
  * Correctness BEFORE timing: each variant is gated ONCE vs the fp32 reshard oracle (rel_L2 < 2e-2 AND
    0 outlier rows AND 0 untouched recv cells) — a fast-but-wrong variant is DISQUALIFIED, not ranked.
  * OOM guard: a runtime torch.cuda.OutOfMemoryError skips the cell (NO memory-estimate gate).
  * Timing is a barrier-bracketed, per-round-drained, drift-cancelled paired-median with all_reduce(MAX)
    slowest-PE consensus — NOT gate_c's back-to-back benchmark_paired (which desyncs cross-node -> hangs).

RUN (venue D / venue E, cp=(2,8) 2-node, torchrun- OR srun-per-rank-launched; ONE process, all cells):
  CPO_CACHE_ENABLED=0 NVSHMEM_DISABLE_NVLS=1 CPO_DIST_MESH=cp=2*8 \
    <launcher> python -u benchmark/distributed/back_a2a_store_bench.py \
      --N-list 4800,8192 --cluster-list 1,2,4,8 --D 16 --out-dir results/roundpark_fp
  Add --check-only for a fast correctness+footprint smoke (no timing). Default: correctness-gate + time.

NOTE (harness kind): this is a STANDALONE PERF DRIVER launched under torchrun / srun-per-rank (the
benchmark/distributed/gate_c_overlap.py pattern) — NOT a torchrun-collected pytest correctness unit (those
live in tests/distributed/test_ib_ring.py) and NOT a kernel definition (it INSTANTIATES the existing thin
GemmA2ASm90 subclass with the standard bench tile TILE=(128,128), exactly as the tests + gate_c do). The
timing reuses bench_utils' validated event-window primitive (time_callable) + all_reduce(MAX) slowest-PE
consensus, but adds a per-round nvshmem drain that benchmark_paired lacks (the anti-hang the coordinator
required — gate_c's benchmark_paired desyncs cross-node on straddle). torch.cuda memory queries below are
FOOTPRINT ACCOUNTING (reporting only) — never a combo-selection / OOM-estimate gate (capacity is a runtime
OutOfMemoryError skip).
"""
import argparse
import json
import os
import statistics
import sys
import time
import traceback
from collections import OrderedDict

# Repo root on sys.path (so `python benchmark/distributed/back_a2a_store_bench.py` works without
# PYTHONPATH; the launcher's PYTHONPATH=$CPO_REPO also covers it). Mirrors tests/distributed/test_ib_ring.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import torch.distributed as dist

import cutlass.torch as cutlass_torch
from cutlass import Float32, Int32
from cutlass.base_dsl.common import DSLBaseError
from cutlass.cute.runtime import from_dlpack

from fold_cp_ops._internal.arch import get_device_capacity, get_max_active_clusters
from fold_cp_ops._internal.compile_time.cute_dsl_utils import torch2cute_dtype_map
from fold_cp_ops._internal.gemm_tvm_ffi_utils import make_scheduler_args
from fold_cp_ops.distributed.gemm_a2a_epi import GemmA2ASm90
from fold_cp_ops.distributed.gemm_bitcode_compile import compile_gemm_with_bitcode
from fold_cp_ops.distributed.distributed_manager import DistributedManager

# Proven low-level helpers (epilogue-arg builders + dist barrier/drain/placements). These are NOT operand
# builders / oracles (no K=256 taint) — the K=N operands + the fp32 oracle are OWNED below in this file.
from tests.distributed.test_gemm_a2a_epi import _epi_cluster, _epi_dyn, _mark, TILE
from tests.distributed.test_gemm_a2a_epi import (
    _drain, _nvshmem_barrier, _trimul_placements, _cp_axis_sizes,
)
from tests.distributed.correctness_harness import compute_error_histogram
# Shared distributed-bench harness (H3): the validated atomic event-window unit + the slowest-PE
# all_reduce(MAX) consensus adapter. We reuse these (NOT hand-rolled events) and add a per-round nvshmem
# drain — benchmark_paired lacks it and desyncs cross-node on straddle (the gate_c hang the maintainer hit).
from benchmark.distributed.bench_utils import time_callable, resolve_dist

# The N-lists for a non-curated-NIC fabric (D=16, cp=(2,8), K=N). STRADDLE: nt_j_pp ODD -> round-park for cluster_n>=4
# (the big cp1 full-buffer under test). EVEN-SHARD: nt_j_pp % 8 == 0 -> single-buffer for cluster_n>=4 (small).
STRADDLE_NS = [1024, 2624, 4800, 6528, 8768, 10304, 13056, 15104, 16512, 18560, 20608, 23552]
EVEN_NS = [8192, 16384, 24576]

# tile-autotune grid (--tile-sweep): the fused GEMM cta tile is swept so each variant runs its BEST
# tile per (N, cluster_n) — the 2-kernel cuBLAS baseline already autotunes its tile per shape, so freezing
# the fused side at 128x128 is the perf-neutrality bug this stage fixes. Grid = tile_m × tile_n × pingpong.
# swizzle / a_in_regs are NOT parent GemmSm90.__init__ constructor knobs (swizzle is derived internally from
# the SmemLayoutAtomKind; there is no a_in_regs on the GEMM) so they are NOT swept. The INVALID-by-construction
# grid points are the PINGPONG ones at tile_m=256 / tile_n>208 (pingpong forbids both -> ValueError at build)
# -> the per-config compile-gate (_guarded_run_cell) SKIPS them cleanly (ValueError / ptxas C7602 / CUDA 701)
# and the sweep continues. #78 UPDATE: tile_m=256 at pingpong=FALSE (INCLUDING 256x256) now RUNS CORRECTLY
# via the 3-WG producer-warp drain (validated 2-node IB cp=(2,8), §0.9.15 / test_ib_ring.py) -- it is NOT a
# register-wall skip; the earlier note that "256x256 busts even the 3-WG layout" is SUPERSEDED (the 3-WG drain
# is exactly the fix that fits tile_n=256). So all four pingpong=False tiles (128x128/128x256/256x128/256x256)
# are expected to gate correct + time; only the pingpong=True tile_m=256 points skip.
TILE_GRID = [
    (128, 128, False), (128, 256, False), (256, 128, False), (256, 256, False),
    (128, 128, True),  (128, 256, True),  (256, 128, True),  (256, 256, True),
]

# AUTOTUNE-VALIDATED tile grid: the tiles that COMPILE clean on SM90 (no ValueError). EXCLUDES (a) 256×256
# (register-spilling, §0.9.17) and (b) EVERY pingpong tile: tile_m=256 pp fails "M must be 64/128/192 if
# pingpong", and 128×256 pp also ValueError'd on a NIC-curated fabric (register/SMEM). Gating them OUT means the autotune
# never even ATTEMPTS a config that can crash the JIT under 16-way parallel cache-off compile (a compile
# ABORT on some ranks — vs a caught ValueError on others — is the cross-rank desync). The 3 survivors are
# all pp=False, all validated (256×128 pp=False runs via the 3-WG drain, §0.9.15). Override via --tile-grid.
AUTOTUNE_TILE_GRID = [(128, 128, False), (128, 256, False), (256, 128, False)]


def _parse_tile_grid(spec):
    """Parse a --tile-grid spec into a [(tile_m, tile_n, pingpong), ...] list. Token = "MxN" or "MxN:pp"
    (pp in {0,1}, default 0). Empty/None -> the full TILE_GRID (the shipped perf-grid default). Used to SCOPE
    the venue E correctness run to a compile-affordable subset (e.g. "128x128,128x256" = the straddle wide-tile
    proof) without touching the default — the ~195s straddle bitcode-compile cliff (task #72) makes the full
    8-pt grid cache-off intractable in one job, but venue E only needs the NEW capability's correctness."""
    if not spec:
        return list(TILE_GRID)
    grid = []
    for tok in spec.split(","):
        tok = tok.strip()
        if not tok:
            continue
        pp = False
        if ":" in tok:
            mn, pps = tok.split(":", 1)
            pp = pps.strip() in ("1", "true", "True", "T")
        else:
            mn = tok
        m, n = mn.lower().split("x")
        grid.append((int(m), int(n), pp))
    return grid


# --------------------------------------------------------------------------- #
# CUT-VS-KEEP variant spec: the 7 live back-A2A drain variants -> (family, force_cluster, differential).
# Autotune knobs per variant: completion {ib_quiet, ib_reap} (ib_reap REQUIRES coalesce=True per the config
# gate -> only coalesce/differential); cluster_n {1,2,4,8} (cluster family only, else 1). The min_drain
# register ladder is EXCLUDED (dead lever) -- never emitted here.
# --------------------------------------------------------------------------- #
# "cluster_roundpark" is NOT here: the kernel parameter it needs was removed at the source, so the
# name can only produce a TypeError. `_VARIANT_SPEC` keeps the entry so an old sbatch naming it gets
# the explanatory ValueError from `_cluster_forced` instead of an unknown-variant error that says
# nothing about why.
VARIANT_NAMES = ("coalesce", "differential", "strided_putwarp", "cluster", "cluster_multislot",
                 "2kernel")
_VARIANT_SPEC = {
    # name -> (family, force_cluster, differential)
    "coalesce":          ("coalesce", None,        False),
    "differential":      ("coalesce", None,        True),   # coalesce band + differential (Task-14) modifier
    "strided_putwarp":   ("strided",  None,        False),  # per-row proto drain (coalesce=False)
    "cluster":           ("cluster",  None,        False),  # shape-adaptive (_cluster_dispatch by nt_j_pp%cn)
    "cluster_multislot": ("cluster",  "multislot", False),  # FORCED 2-slot rotating staging
    "cluster_roundpark": ("cluster",  "roundpark", False),  # FORCED cp1-full-buffer staging
    "2kernel":           ("2kernel",  None,        False),  # cuBLAS GEMM + NCCL a2a baseline
}


def _variant_family(v):
    if v in _VARIANT_SPEC:
        return _VARIANT_SPEC[v][0]
    if v.startswith("cn"):  # legacy cnX (shape-adaptive cluster) -> back-compat for old sbatches
        return "cluster"
    raise ValueError(f"unknown variant {v!r}; known: {VARIANT_NAMES} (or legacy cn1/cn2/cn4/cn8)")


def _variant_completions(v):
    # ib_reap REQUIRES coalesce=True (gemm_sm90_a2a.py: 'ib_reap requires coalesce'). strided/cluster/2kernel
    # keep the single blocking-quiet completion; only coalesce/differential get the {quiet, reap} knob.
    return ["ib_quiet", "ib_reap"] if v in ("coalesce", "differential") else ["ib_quiet"]


def _variant_cluster_ns(v, cluster_ns):
    return list(cluster_ns) if _variant_family(v) == "cluster" else [1]


def _pp_disallowed(fam):
    # FIX-2: BOTH A2A drain families are pingpong-INCOMPATIBLE (mirrors gate_c_overlap.py::_cfg_ok, which
    # gates the same two mechanisms as 'cluster_drain'/'coalesce_dyn'). pingpong runs the epilogue PER
    # MMA-warpgroup = 2 concurrent in-flight runs, breaking each family's single-in-flight-run assumption:
    #   * cluster (cluster_drain, incl. cluster_multislot/cluster_roundpark) DEADLOCKS -- its cross-CTA
    #     'full' mbarrier is init count=cluster_n (one arrive per cluster CTA per run); the 2 warpgroups
    #     interleave 'full' arrives -> the mbar phase desyncs -> mbarrier_wait spins forever (GPU-100%).
    #   * coalesce (coalesce_dyn, incl. differential) is NUMERICALLY WRONG -- its intra-CTA ring shares ONE
    #     per-CTA producer counter across the 2 pingpong warpgroups -> ring-slot routing races (wrong
    #     output, no deadlock -- the fast-but-wrong trap, so this must be gated, not left to the oracle).
    # Gated UNCONDITIONALLY (every N, every tile): TILE_GRID's (128,128,True) is construction-valid (doesn't
    # trip the base-GEMM pingpong ValueError) so, absent this gate, run_tile_sweep/run_autotune_cell's
    # DEFAULT (--tile-grid unset) walk reaches it for cluster/coalesce -- e.g. at N=8192 (EVEN_NS).
    return fam in ("cluster", "coalesce")


# --------------------------------------------------------------------------- #
# ROBUST STRUCTURED STATUS/SHAPE helpers. Every cell result carries {status, error_class, error_msg, reason};
# status in {ok, error, skip_shape, oom, timeout, disqualified_wrong, check_ok, skip_*}. Arbitrary N + cp are
# ACCEPTED; the ONLY legit skip is the 16-B-alignment HARD RULE (N//cp per-axis must be %8 for bf16 TMA) ->
# emit a LOUD structured skip_shape (never a silent continue, never a crash).
# --------------------------------------------------------------------------- #
def _shape_check(N, cp0, cp1):
    """(valid, reason). Arbitrary N + arbitrary cp (incl 1-D cp1==1, non-pow2 cp) are supported; the ONLY
    invalid case is the 16-byte-alignment HARD RULE: N shardable on both axes AND N//cp0, N//cp1 each %8==0
    (bf16 -> 8 elems = 16 B, the TMA minimum). Returns a human reason so the user SEES why a shape skipped."""
    if cp0 <= 0 or cp1 <= 0:
        return False, f"invalid cp mesh cp0={cp0} cp1={cp1}"
    if N % cp0 != 0 or N % cp1 != 0:
        return False, f"N={N} not divisible by cp0={cp0} and/or cp1={cp1} (not token-shardable)"
    ni, nj = N // cp0, N // cp1
    if ni % 8 != 0 or nj % 8 != 0:
        return False, (f"N_i_loc=N/cp0={ni} (%8={ni % 8}) / N_j_loc=N/cp1={nj} (%8={nj % 8}) not %8 == not "
                       f"16-B-aligned (bf16 TMA HARD RULE); pick N with N/cp0 and N/cp1 both multiples of 8")
    return True, ""


def _err_fields(exc):
    """Structured {error_class, error_msg} from an exception (msg truncated)."""
    return {"error_class": type(exc).__name__, "error_msg": repr(exc)[:1500]}


def _print_run_summary(all_cells, variant_list, Ns):
    """End-of-run SUMMARY (rank 0): the (N × variant) -> status matrix + counts, so one glance shows what
    ran / didn't / why. Statuses collapsed to a 2-char glyph; a legend + per-status counts follow."""
    glyph = {"ok": "ok", "error": "ER", "oom": "OM", "skip_shape": "sk", "timeout": "TO",
             "disqualified_wrong": "XX", "autotune_ok": "ok", "check_ok": "ok", "skip_pp_unsupported": "sk"}
    by = {}
    counts = {}
    for c in all_cells:
        st = str(c.get("status", "?"))
        counts[st] = counts.get(st, 0) + 1
        by[(c.get("N"), c.get("variant"))] = st
    print("\n================ RUN SUMMARY (N × variant → status) ================", flush=True)
    hdr = f"{'N':>7} | " + " ".join(f"{v[:9]:>9}" for v in variant_list)
    print(hdr, flush=True)
    print("-" * len(hdr), flush=True)
    for N in Ns:
        row = f"{N:>7} | "
        for v in variant_list:
            st = by.get((N, v))
            row += f"{glyph.get(st, (str(st)[:2] if st else '--')):>9} "
        print(row, flush=True)
    print("legend: ok / ER=error / OM=oom / sk=skip_shape,skip_pp / TO=timeout / XX=disqualified_wrong",
          flush=True)
    print("counts: " + "  ".join(f"{k}={v}" for k, v in sorted(counts.items())), flush=True)
    print("====================================================================\n", flush=True)


def _norm_cell(cell):
    """Ensure a cell dict carries the structured-report keys {status, error_class, error_msg, reason}. Maps
    legacy statuses (ERROR/OOM/DISQUALIFIED_wrong) to the normalized lower-case set. Idempotent + never
    raises. Also lifts an 'err' string into error_msg when error_class is absent."""
    if not isinstance(cell, dict):
        return {"status": "error", "error_class": "BadCell", "error_msg": repr(cell)[:200], "reason": ""}
    st = str(cell.get("status", "ok"))
    _map = {"ERROR": "error", "OOM": "oom", "DISQUALIFIED_wrong": "disqualified_wrong"}
    cell["status"] = _map.get(st, st)
    cell.setdefault("reason", "")
    if "error_class" not in cell:
        _e = cell.get("err")
        if _e:
            cell["error_class"] = str(_e).split(":", 1)[0]
            cell["error_msg"] = str(_e)[:1500]
        else:
            cell["error_class"] = None
            cell["error_msg"] = None
    return cell


# --------------------------------------------------------------------------- #
# K=N operand builder (OWNED here; the assert is the K==N guard). NOT importable from tests/ (K=256 taint).
# --------------------------------------------------------------------------- #
def _symmetric_empty(shape, dtype, device):
    """Allocate a SYMMETRIC buffer through the torch MemPool that owns nvshmem in this tree.

    Purpose: hand the back-A2A benches a symmetric tensor WITHOUT nvshmem4py's own allocator.
    `DistributedManager` performs the nvshmem bootstrap here, so nvshmem4py's `_is_initialized`
    is permanently False and its `tensor()` raises `NvshmemInvalid: NVSHMEM Library is not
    initialized` -- silently, per config, which the harness isolates into a `status: error` row.
    `targets/front_a2a.py` already allocates this way; this is the same conversion for the back.

    Semantics: the pool draws from the nvshmem symmetric heap and RECYCLES -- dropping the last
    reference returns the block, so there is NO per-tensor free to call. That is deliberate: it
    keeps the collective `nvshmem_free` off the object-destruction path, where a GC-ordering
    difference across ranks deadlocks it.

    Input requirements:
        shape: tuple of ints, IDENTICAL on every rank. The underlying `nvshmem_malloc` is
            COLLECTIVE, so a rank-varying shape desynchronizes it and HANGS the job -- it does
            not raise.
        dtype: a torch dtype (every current caller passes `torch.bfloat16`).
        device: the calling rank's CUDA device. A CPU device raises inside the pool.

    Returns:
        An UNINITIALIZED symmetric tensor. Callers must `zero_()`/`fill_()` it themselves --
        recycled blocks carry the previous tenant's bytes.
    """
    from fold_cp_ops.distributed.distributed_manager import DistributedManager
    with torch.cuda.use_mem_pool(DistributedManager.symmetric_mempool(device)):
        return torch.empty(shape, dtype=dtype, device=device)


def _build_inputs_2d_kn(device, rank, N, cp0, cp1, Dloc, B):
    """The REAL back-TriMul einsum c = einsum("lik,ljk->lij", a, b): K == N_token (operands O(N²) each), NOT
    a thin K=256. Symmetric recv (cp,Dloc,B,N_i_loc,N_j_loc) + the (M,N,L) logical D-view over its storage."""
    cp = cp0 * cp1
    M, L, N_i_loc, N_j_loc = N, Dloc * B, N // cp0, N // cp1
    K = N  # back-TriMul einsum contracts over N_token -> K == N (a K=256 shortcut INVALIDATES the perf read)
    assert K == N, "back-TriMul einsum contracts over N_token; K must == N (K=256 is correctness-only)"
    torch.manual_seed(4321 + rank)
    A = torch.randn(L, M, K, device=device, dtype=torch.bfloat16) / (K ** 0.5)
    Bt = torch.randn(L, N, K, device=device, dtype=torch.bfloat16) / (K ** 0.5)
    recv = _symmetric_empty((cp, Dloc, B, N_i_loc, N_j_loc), torch.bfloat16, device)
    recv.fill_(-99.0)
    A3, B3 = A.permute(1, 2, 0), Bt.permute(1, 2, 0)
    D_logical = torch.as_strided(recv, (M, N, L), (N, 1, M * N))
    return A, Bt, recv, (_mark(A3), _mark(B3), _mark(D_logical))


# --------------------------------------------------------------------------- #
# fp32 reshard oracle (OWNED here; correctness reference only). Mirrors test_a2a_fusion._ref_back_gemm_native_2d
# EXACTLY: tri = bmm(A, Bt^T) over the full (N,N) grid; recv[s,d,b,i,j] = peer_s.tri[d*B+b][r0*Ni+i, r1*Nj+j].
# --------------------------------------------------------------------------- #
def _oracle_2d(A, Bt, pm, *, cp0, cp1, Dloc, B, N_i_loc, N_j_loc, N, world_size):
    cp = cp0 * cp1
    tri = torch.bmm(A.float(), Bt.float().transpose(-1, -2)).to(torch.bfloat16)  # (L,N,N)
    gathered = [torch.empty_like(tri) for _ in range(world_size)]
    dist.all_gather(gathered, tri.contiguous())
    r0, r1 = pm.my_cp_rank // cp1, pm.my_cp_rank % cp1
    i0, j0 = r0 * N_i_loc, r1 * N_j_loc
    expected = torch.empty((cp, Dloc, B, N_i_loc, N_j_loc), device=A.device, dtype=torch.bfloat16)
    for s in range(cp):
        peer_global = int(pm.cp_pe_table[s].item())
        src = gathered[peer_global]
        for d in range(Dloc):
            for b in range(B):
                Lp = d * B + b
                expected[s, d, b, :, :] = src[Lp, i0 : i0 + N_i_loc, j0 : j0 + N_j_loc]
    return expected, tri


# --------------------------------------------------------------------------- #
# Variant 6 pieces: the NCCL all_to_all_single 2-D reshard + a per-N tri buffer. The reshard reproduces the
# fused kernel's permutation: token (i,j) -> global peer; recv slot s holds peer s's tri for MY 2-D block.
# --------------------------------------------------------------------------- #
def _reshard_2d(pm, *, cp0, cp1, Dloc, B, N_i_loc, N_j_loc, device):
    cp = cp0 * cp1
    L = Dloc * B
    pe_table = [int(pm.cp_pe_table[s].item()) for s in range(cp)]
    chunk = L * N_i_loc * N_j_loc

    def reshard(tri):  # tri: (L, N, N) bf16 -> (cp, Dloc, B, N_i_loc, N_j_loc) recv
        send_parts = []
        for p in range(cp):
            s = pe_table.index(p)  # global PE p's flat cp slot -> its (r0,r1) block
            i0, j0 = (s // cp1) * N_i_loc, (s % cp1) * N_j_loc
            blk = tri[:, i0 : i0 + N_i_loc, j0 : j0 + N_j_loc].contiguous()
            send_parts.append(blk.reshape(-1))
        send = torch.cat(send_parts)
        recv_flat = torch.empty_like(send)
        dist.all_to_all_single(recv_flat, send)
        recv = torch.empty((cp, Dloc, B, N_i_loc, N_j_loc), device=device, dtype=tri.dtype)
        for p in range(cp):
            s = pe_table.index(p)
            part = recv_flat[p * chunk : (p + 1) * chunk].reshape(L, N_i_loc, N_j_loc)
            recv[s] = part.reshape(Dloc, B, N_i_loc, N_j_loc)
        return recv

    return reshard


# --------------------------------------------------------------------------- #
# Per-cluster_n / per-shape dispatch: (staging shape, config kwargs, path label). This IS the footprint
# comparison — round-park's cp1-full-buffer vs the even-shard single-buffer, routed by nt_j_pp % cluster_n.
# --------------------------------------------------------------------------- #
def _cluster_dispatch(cluster_n, nt_j_pp, cp1, N_j_loc, n_clusters, rd):
    even_shard = (nt_j_pp % cluster_n == 0)
    m_sub = TILE[0] // 128  # #78: epi M-subtiles per CTA tile (1 at tile_m<=128 -> rank-3, byte-identical)
    if cluster_n == 1 or even_shard:
        # single-buffer even-shard drain (small footprint). cluster_n==1 is degenerate (nt_j%1==0 always).
        # #78 multi-epi-subtile: tile_m>128 (m_sub>1) needs the m_sub staging AXIS
        # (n_clusters, m_sub, 128, N_j_loc) — the drain's _build_cluster_stage_atom applies the 4-D box-
        # permute mode=[2,3,1,0] when m_sub>1 (the m_sub axis plays the db slot's role), so a rank-3 staging
        # raises "invalid mode element for input of rank 3". m_sub==1 keeps the shipped rank-3 (byte-
        # identical). Mirrors test_ib_ring.py's stage_buf = (n_clusters, m_sub, 128, size_Nj).
        shape = (n_clusters, m_sub, 128, N_j_loc) if m_sub > 1 else (n_clusters, 128, N_j_loc)
        return shape, {}, ("even_shard" if cluster_n != 1 else "cn1_degenerate")
    if cluster_n == 2:
        # straddle -> 2-slot MULTISLOT (rd-slot rotating full-peer staging).
        return (n_clusters, rd, 128, N_j_loc), {"cluster_multislot": True}, "multislot_2slot"
    # cluster_n >= 4, straddle. The historical route here was WHOLE-ROUND PARK (cp1 full-buffer slots,
    # one per peer_j -- the big buffer, `cluster_roundpark=True`).
    #
    # THAT ROUTE IS DEAD AND WAS EMITTING AN IMPOSSIBLE CONFIG. A later change made cluster_multislot the
    # SOLE production IB drain and REMOVED `cluster_roundpark` from `configure_a2a_gemm_native`'s
    # signature (see gemm_sm90_a2a.py's "IB-drain variant is REMOVED at the source" block; the removal
    # is pinned by tests/distributed/test_gemm_a2a_epi.py::test_removed_hybrid_drain_kwargs_rejected,
    # which asserts TypeError). This dispatcher was never updated, so every cluster_n>=4 STRADDLE cell
    # built a kwargs dict the kernel cannot accept and died with
    #     TypeError: ... unexpected keyword argument 'cluster_roundpark'
    # -- measured 2026-08-20: cluster_n=4 gave 8 of 12 valid, cluster_n=8 gave 0 ok / 288 err. It fires
    # on the SHAPE-ADAPTIVE "cluster" variant too, not only on the explicit cluster_roundpark name, which
    # is why the whole cn>=4 straddle region of the sweep was empty.
    #
    # Route to the surviving drain instead. The kernel's own cap is `cluster_n <= nt_j_pp` and its error
    # text says "the dispatcher must pick cluster_n<=nt_j_pp", i.e. this is the dispatcher's job. Where
    # the cap does not hold we deliberately DO NOT silently drop to cluster_n=1: the kernel raises a named
    # ValueError, _guarded_run_cell turns it into a graceful ERROR skip, and the cell is reported rather
    # than quietly re-measured at a different cluster_n. The label changes so a result row can never be
    # mistaken for the old round-park measurement.
    return (n_clusters, rd, 128, N_j_loc), {"cluster_multislot": True}, "multislot_cn_ge4"


def _cluster_forced(force, cluster_n, nt_j_pp, cp1, N_j_loc, n_clusters, rd):
    """(staging shape, cfg kwargs, path label) for a cluster VARIANT. force=None -> the shape-adaptive
    _cluster_dispatch (base even-shard / multislot / roundpark by nt_j_pp%cluster_n, the 'cluster' name).
    force='multislot'/'roundpark' -> the FORCED staging + cfg REGARDLESS of shape (the explicit
    cluster_multislot / cluster_roundpark cut-vs-keep names). configure_a2a_gemm_native still validates
    (multislot 2-slot cap cluster_n<=nt_j_pp; roundpark drops it) -> an invalid (variant,cn,shape) raises
    at build -> _guarded_run_cell catches it -> graceful ERROR skip (the sweep continues)."""
    if force is None:
        return _cluster_dispatch(cluster_n, nt_j_pp, cp1, N_j_loc, n_clusters, rd)
    if force == "multislot":
        return (n_clusters, rd, 128, N_j_loc), {"cluster_multislot": True}, "forced_multislot"
    raise ValueError(
        "the 'cluster_roundpark' variant was REMOVED at the kernel source (epic-close P2: "
        "cluster_multislot is the sole production IB drain), so configure_a2a_gemm_native has no "
        "cluster_roundpark parameter and forcing it can only raise TypeError. Use 'cluster' "
        "(shape-adaptive) or 'cluster_multislot'."
    )


def _build_cluster(A, Bt, cA, cB, cD, recv, stage_view, pe_dev_c, *, cluster_n, cfg_kw, cp, cp0, cp1,
                   my_cp_rank, B, N_loc, pe_table, N, rd, pingpong=False):
    a_dtype = torch2cute_dtype_map[A.dtype]
    n_clusters = get_max_active_clusters(cluster_n)
    sched = make_scheduler_args(n_clusters, Int32(8), None, None)
    stream = cutlass_torch.current_stream()
    g = GemmA2ASm90(Float32, a_dtype, TILE, (1, cluster_n, 1), pingpong=pingpong, is_persistent=True)
    g.configure_a2a_gemm_native(
        cp=cp, my_cp_rank=my_cp_rank, B=B, N_loc=N_loc, pe_table=pe_table, N=N,
        cp_axis_sizes=(cp0, cp1), dynamic=True, arbitrary_n=True, pe_aligned_tiling=True, ib_ring=True,
        ib_quiet=True, decoupled=True, producer_tma=True, consumer_strided=True,
        consumer_strided_putwarp=True, ring_depth=rd,
        cluster_drain=True, cluster_n=cluster_n, **cfg_kw,
    )
    epi = _epi_cluster(g, recv, stage_view, pe_dev_c)
    c = compile_gemm_with_bitcode(g, cA, cB, cD, None, epi, sched, stream, None, register=True)
    # use_3wg: whether the STAGE-1 3-WG producer-warp drain auto-activated (host-computable post-configure;
    # True for tile_n=256 / register-heavy tiles on the has_ib_peers path -> the wide tile that STAGE-1 unlocked).
    return c, (cA, cB, cD, None, epi, sched, stream, None), bool(g._use_3wg_drain())


def _build_coalesce(A, Bt, cA, cB, cD, recv, ring_view, pe_dev_c, *, cp, cp0, cp1, my_cp_rank, B, N_loc,
                    pe_table, N, rd, pingpong=False, reap=None, differential=False, lws=None):
    a_dtype = torch2cute_dtype_map[A.dtype]
    sched = make_scheduler_args(get_max_active_clusters(1), Int32(8), None, None)
    stream = cutlass_torch.current_stream()
    g = GemmA2ASm90(Float32, a_dtype, TILE, (1, 1, 1), pingpong=pingpong, is_persistent=True)
    # LEVER-1 nbi_reap A/B: CPO_A2A_IB_REAP=1 -> the coalesce drain uses nbi + periodic QP_ALL reap
    # (overlaps the GEMM) instead of the per-put blocking quiet; ib_quiet flips to False (reap needs nbi).
    # CPO_A2A_REAP_CADENCE=<int> overrides the reap window (default = ring_depth).
    # LEVER-2 min_drain A/B: CPO_A2A_MIN_DRAIN=1 -> the coalesce drain runs on ONE warp (warp 9) to
    # restore the parent register schedule (256x192 stops spilling). It REQUIRES the nbi reap, so it forces
    # ib_reap on (the nbi is what makes 1 warp sufficient).
    _min = os.environ.get("CPO_A2A_MIN_DRAIN", "0") == "1"
    _minp = os.environ.get("CPO_A2A_MIN_DRAIN_PRECOMPUTE", "0") == "1"
    # SPLIT (LEVER-2 register): w12=PUT-only / w13=REAP-only so their live sets never coexist -> drain-WG
    # setmaxnreg cap <=56 -> the MMA .inc(208) fits. Requires the precompute drain (CPO_A2A_MIN_DRAIN_PRECOMPUTE).
    _mins = os.environ.get("CPO_A2A_MIN_DRAIN_SPLIT", "0") == "1"
    # completion: the autotune passes `reap` explicitly (True=ib_reap / False=ib_quiet); when None fall back
    # to the env A/B (back-compat for the sbatch launcher). min_drain (env, default off) always forces reap.
    _env_reap = os.environ.get("CPO_A2A_IB_REAP", "0") == "1"
    _reap = (bool(reap) if reap is not None else _env_reap) or _min
    _cad = os.environ.get("CPO_A2A_REAP_CADENCE")
    g.configure_a2a_gemm_native(
        cp=cp, my_cp_rank=my_cp_rank, B=B, N_loc=N_loc, pe_table=pe_table, N=N,
        cp_axis_sizes=(cp0, cp1), dynamic=True, arbitrary_n=True, pe_aligned_tiling=True, ib_ring=True,
        ib_quiet=(not _reap), decoupled=True, producer_tma=True, consumer_strided=True,
        consumer_strided_putwarp=True, ring_depth=rd, coalesce=True,
        ib_reap=_reap, reap_cadence=(int(_cad) if _cad else None), min_drain=_min,
        min_drain_precompute=_minp, min_drain_split=_mins,
        differential=differential, local_world_size=lws,  # differential variant (Task-14); None/False = off
    )
    if _min:
        # LEVER-2 num_regs_mma cap for the extra-drain-WG layout. Default 0 -> auto ~208 (precompute drain
        # <=40). For the NO-precompute layout de-risk (drain@168), set CPO_A2A_MIN_DRAIN_MMA_REGS=152 so
        # the MMA .inc does not over-allocate (drain@168 pins the 4th WG -> MMA 152).
        g._a2a_min_drain_mma_regs = int(os.environ.get("CPO_A2A_MIN_DRAIN_MMA_REGS", "0"))
    epi = _epi_dyn(g, recv, ring_view, pe_dev_c)
    c = compile_gemm_with_bitcode(g, cA, cB, cD, None, epi, sched, stream, None, register=True)
    return c, (cA, cB, cD, None, epi, sched, stream, None), bool(g._use_3wg_drain())


def _build_strided_putwarp(A, Bt, cA, cB, cD, recv, ring_view, pe_dev_c, *, cp, cp0, cp1, my_cp_rank, B,
                           N_loc, pe_table, N, rd, pingpong=False):
    """STRIDED per-row putwarp drain (variant 'strided_putwarp'): coalesce=False +
    consumer_strided_putwarp=True -> the §7.16 per-row _decoupled_drain_loop_strided_putwarp (the ~3.2x
    proto follow-up, the natural 'cut' candidate). BLOCKING quiet only (ib_quiet=True; ib_reap requires
    coalesce). The ring slot is a per-TILE box (rd, epi_m=128, tile_n), NOT the coalesce full-N band.
    NOTE: no prior harness reference builds this pure (non-cluster) strided path -> best-effort; a build
    error is caught by _guarded_run_cell -> graceful ERROR skip."""
    a_dtype = torch2cute_dtype_map[A.dtype]
    sched = make_scheduler_args(get_max_active_clusters(1), Int32(8), None, None)
    stream = cutlass_torch.current_stream()
    g = GemmA2ASm90(Float32, a_dtype, TILE, (1, 1, 1), pingpong=pingpong, is_persistent=True)
    g.configure_a2a_gemm_native(
        cp=cp, my_cp_rank=my_cp_rank, B=B, N_loc=N_loc, pe_table=pe_table, N=N,
        cp_axis_sizes=(cp0, cp1), dynamic=True, arbitrary_n=True, pe_aligned_tiling=True, ib_ring=True,
        ib_quiet=True, decoupled=True, producer_tma=True, consumer_strided=True,
        consumer_strided_putwarp=True, ring_depth=rd, coalesce=False,  # -> per-row strided drain
    )
    epi = _epi_dyn(g, recv, ring_view, pe_dev_c)
    c = compile_gemm_with_bitcode(g, cA, cB, cD, None, epi, sched, stream, None, register=True)
    return c, (cA, cB, cD, None, epi, sched, stream, None), bool(g._use_3wg_drain())


# --------------------------------------------------------------------------- #
# Correctness gate: rel_L2 < 2e-2 AND 0 outlier rows AND 0 untouched recv cells (the _gate_2d contract).
# --------------------------------------------------------------------------- #
def _gate(recv, expected, *, cp, Dloc, B, N_i_loc, N_j_loc):
    got4d = recv.reshape(cp, Dloc * B, N_i_loc, N_j_loc)
    exp4d = expected.reshape(cp, Dloc * B, N_i_loc, N_j_loc)
    h = compute_error_histogram(got4d, exp4d)
    untouched = int((recv == -99.0).all(dim=4).sum().item())
    passed = (h.rel_l2 < 2e-2) and (h.n_outlier_rows == 0) and (untouched == 0)
    return passed, {"rel_l2": float(h.rel_l2), "n_outlier_rows": int(h.n_outlier_rows),
                    "untouched": untouched}


# --------------------------------------------------------------------------- #
# SAFE distributed timer (H3-compliant): each round is barrier-before / event-window / drain-after, in
# lockstep (fixed identical iters=1 on every rank -> no adaptive-rep desync). The atomic event window is
# bench_utils.time_callable (the validated, sync-bracketed unit); the drift-cancelled per-rank median is
# reduced to the SLOWEST PE via the shared adapter's all_reduce(MAX). The per-round _drain() (nvshmem quiet
# + barrier + sync) is the ANTI-HANG benchmark_paired lacks: ib_quiet blocking puts complete in-kernel, but
# the drain finishes cross-node completion + resyncs before the ring/staging slot is reused next round.
# --------------------------------------------------------------------------- #
def _time_variant(call_fn, device, da, *, rounds, warmup):
    for _ in range(warmup):
        _nvshmem_barrier()
        call_fn()
        _drain()
    samples = []
    for _ in range(rounds):
        _nvshmem_barrier()
        samples.append(time_callable(call_fn, 1, device))  # shared atomic event-window unit
        _drain()  # complete cross-node puts + resync before slot reuse (the anti-hang)
    med = statistics.median(samples)
    med_max = da.all_reduce_max(med) if da.is_distributed else med
    return med_max, med, samples


# --------------------------------------------------------------------------- #
# One (N, variant) cell: build K=N operands, gate correctness vs oracle, then (unless --check-only) time it,
# and account the GMEM scratch footprint (analytical exact from the symheap alloc + torch peak delta).
# --------------------------------------------------------------------------- #
def run_cell(dm, pm, da, N, variant, *, cp0, cp1, Dloc, B, rd, rounds, warmup, timed, pingpong=False,
             perf_only=False, first_n=None, cluster_n=1, completion=None, lws=None):
    # FIX-2 PP1 GATE: host-deterministic on (variant, pingpong) alone -- checked FIRST, before dm/pm/nvshmem/
    # CUDA are touched, so every rank skips in lockstep and the sweep never attempts a deadlocking / wrong-
    # answer build. See _pp_disallowed for why cluster/coalesce are pingpong-incompatible.
    fam = _variant_family(variant)
    if pingpong and _pp_disallowed(fam):
        return {"N": N, "variant": variant, "status": "skip_pp_unsupported", "pingpong": True,
                "reason": "pingpong=True gated for cluster/coalesce families (cross-CTA mbar deadlock / "
                          "intra-CTA ring wrong-answer at pp1; see _pp_disallowed)"}
    cp = cp0 * cp1
    device = dm.device
    N_i_loc, N_j_loc = N // cp0, N // cp1
    N_loc = N_i_loc
    tile_n = TILE[1]
    nt_j_pp = (N_j_loc + tile_n - 1) // tile_n
    pe_table = tuple(int(x) for x in pm.cp_pe_table.tolist())
    pe_dev_t = pm.cp_pe_table.to(torch.int32).contiguous()  # KEEP a ref alive (cute view below aliases it)
    pe_dev_c = from_dlpack(pe_dev_t, assumed_align=4)
    grid_ctas = get_max_active_clusters(1)
    dtsize = 2  # bf16
    # CUT-VS-KEEP variant resolution: the forced-cluster mode + differential + the completion knob (fam above).
    force_cluster = _VARIANT_SPEC[variant][1] if variant in _VARIANT_SPEC else None
    differential = _VARIANT_SPEC[variant][2] if variant in _VARIANT_SPEC else False
    # completion=None (legacy / single-cell) -> reap=None -> _build_coalesce env-fallback (back-compat with
    # the launcher's CPO_A2A_IB_REAP). The autotune passes 'ib_quiet'/'ib_reap' -> explicit reap.
    reap = None if completion is None else (completion == "ib_reap")
    if variant.startswith("cn"):  # legacy shape-adaptive cnX -> cluster_n from the name (back-compat)
        cluster_n = int(variant[2:])
    out = {"N": N, "variant": variant, "cp0": cp0, "cp1": cp1, "cp": cp, "Dloc": Dloc, "B": B,
           "N_i_loc": N_i_loc, "N_j_loc": N_j_loc, "nt_j_pp": nt_j_pp,
           "tile_m": TILE[0], "tile_n": TILE[1], "pingpong": pingpong, "use_3wg": None,
           "cluster_n": (cluster_n if fam == "cluster" else None), "completion": completion,
           "i_straddle": (N_i_loc % TILE[0] != 0), "status": "ok"}
    # PERF-ONLY oracle gate (lead-required): full check-mode ALWAYS gates; perf-only gates correctness ONCE
    # per (variant,tile) at the SMALLEST N (`first_n`, oracle cheap there) and SKIPS the oracle at every
    # larger N (times only) — the kernels are dynamic-shape (correct@small-N ⟹ correct@any-N), so one gate
    # covers all N while the big-N ~18GB fp32 O(N³) oracle is never built.
    do_gate = (not perf_only) or (first_n is not None and N == first_n)

    # NOTE (2026-07-20): the earlier "2-slot MULTISLOT straddle HANGS at Dloc*B==1" skip-guard was REMOVED.
    # It was an INFERRED (never synccheck-confirmed) attribution from a MULTI-CELL cut-keep SWEEP wedge, and
    # is REFUTED by direct measurement: `test_ib_ring_cluster_drain_2d_multislot_L1` (cn2, straddle N=3008
    # nt_j_pp=3 AND N=4672 nt_j_pp=5, Dloc=1,B=1 -> L=1) PASSES on venue E 2-DGX cp=(2,8) cross-node
    # (correct reshard, per-row L2<2e-2), and compute-sanitizer synccheck reports NO barrier divergence in
    # the multislot kernel. So L=1 is NOT a kernel deadlock (the per-peer serialize is L-independent and the
    # drain posts empty[slot] every peer). Per the FIRST PRINCIPLE, L=1 (D<=cp) is a valid 16-B-aligned
    # input and MUST run -- so it is NO LONGER skipped here. If a MULTI-CELL sweep wedges, that is the
    # harness's cross-rank barrier-variance / process-teardown class (run ONE cell per job), not this kernel.

    recv = None
    compiled = None
    stage_buf = None
    scratch_bytes = 0
    scratch_formula = ""
    # Initialized HERE and not at its explanatory comment inside the try (below), because the
    # `finally` now READS it, and everything between `try:` and that comment can raise -- an OOM in
    # `_build_inputs_2d_kn` is exactly the path this whole block exists for. Reading an unbound
    # local raises UnboundLocalError, which inside a `finally` would REPLACE the real error with a
    # bogus one. (Assigning to an unbound local is fine; only reading is not.)
    build_err = None
    try:
        # Operands + fp32 oracle INSIDE the try so a large-N OOM (e.g. the oracle's world-wide all_gather)
        # still frees the symheap recv in the finally (no leak that would cascade into the next cell's OOM).
        A, Bt, recv, (cA, cB, cD) = _build_inputs_2d_kn(device, dm.rank, N, cp0, cp1, Dloc, B)
        recv_bytes = recv.numel() * dtsize
        out["recv_bytes"] = recv_bytes  # the O(N²) output every variant materializes (NOT scratch)
        # PERF-ONLY (venue D grid): skip the fp32 O(N³) reshard ORACLE + its O(N²·cp) all_gather (~18 GB @
        # N=24576) and the per-variant correctness run. Compile + time ONLY. Correctness is proven separately
        # on venue E; a perf run must not pay the oracle (lead-required for the large-N grid).
        expected = None
        passed = gate = None  # perf-only safety: assigned ONLY in the do_gate branches; the reads below are
        # do_gate-guarded, but init keeps them bound on every path (no UnboundLocalError if a branch skips).
        # BLOCKER-1 consensus: a per-config COMPILE can fail on SOME ranks but not others (transient MLIR /
        # resource under 16-way parallel cache-off) -> the collective do_gate below (oracle all_gather +
        # kernel puts + drain) would then run on the survivors while the failures skip it -> cross-rank
        # desync -> collective hang. So capture the build failure, all-reduce it, and run the gate ONLY if
        # EVERY rank built. build_err None == this rank built. (Its `= None` initialization sits
        # ABOVE the `try:` -- see the note there; the `finally` reads it and this line is inside the
        # window that can raise.)
        call_fn = None
        if do_gate:
            expected, tri = _oracle_2d(A, Bt, pm, cp0=cp0, cp1=cp1, Dloc=Dloc, B=B, N_i_loc=N_i_loc,
                                       N_j_loc=N_j_loc, N=N, world_size=cp)
        # ---- build + (unless perf_only) one correctness run per variant ----
        if fam == "2kernel":
            reshard = _reshard_2d(pm, cp0=cp0, cp1=cp1, Dloc=Dloc, B=B, N_i_loc=N_i_loc, N_j_loc=N_j_loc,
                                  device=device)
            L = Dloc * B
            # scratch = the GEMM output tri (O(N²)) + the reshard send + recv_flat + recv_nccl (each O(N²)).
            tri_bytes = L * N * N * dtsize
            a2a_bytes = 3 * L * N_i_loc * N_j_loc * cp * dtsize  # send + recv_flat + recv_nccl (== 3·L·N²)
            scratch_bytes = tri_bytes + a2a_bytes
            scratch_formula = f"tri(L·N²={tri_bytes}) + a2a(3·L·N²={a2a_bytes})"
            # FAIR baseline: the TIMED GEMM MUST match the fused kernel's WGMMA precision — bf16 in, fp32
            # accumulate (cuBLAS tensor-core). An fp32 bmm (A.float()) runs the fp32/TF32 path at ~1/4-1/8
            # the bf16 rate; at K=N (O(N³) compute-bound) the GEMM dominates the 2-kernel wall-clock, so fp32
            # would inflate the floor 4-8x at large N and overstate the fused win. fp32 stays oracle-ONLY.
            # (the 2kernel correctness gate moved to the shared post-consensus gate block below)
            def _call_2kernel():
                # naive 2-kernel: cuBLAS batched bf16 GEMM (K=N einsum, fp32 accum -> matches fused WGMMA
                # precision, FAIR wall-clock) -> local tri, then NCCL all_to_all A2A reshard.
                _tri = torch.bmm(A, Bt.transpose(-1, -2))
                reshard(_tri)

            call_fn = _call_2kernel  # de-shadowed: unique def name, explicit bind (Pyright-clean)
        elif fam == "coalesce":
            # coalesce band ring (grid, rd, cp1, tile_m, N_j_loc). 'differential' = the SAME ring + the
            # node-local-peer-skip modifier (Task-14). completion {ib_quiet, ib_reap} via `reap`. #78
            # grow-rows: the ring's M extent is tile_m (== TILE[0]) so the drain walks all tile_m band rows.
            ring_buf = _symmetric_empty((grid_ctas, rd, cp1, TILE[0], N_j_loc), torch.bfloat16, device)
            ring_buf.zero_()
            stage_buf = ring_buf
            scratch_bytes = ring_buf.numel() * dtsize
            scratch_formula = f"ring(grid={grid_ctas}·rd={rd}·cp1={cp1}·{TILE[0]}·N_j_loc={N_j_loc})"
            ring_view = from_dlpack(ring_buf, assumed_align=16).mark_layout_dynamic()
            try:  # BLOCKER-1: isolate the compile (may fail non-deterministically) for the consensus below
                compiled, run_args, use_3wg = _build_coalesce(
                    A, Bt, cA, cB, cD, recv, ring_view, pe_dev_c, cp=cp, cp0=cp0, cp1=cp1,
                    my_cp_rank=int(pm.my_cp_rank), B=B, N_loc=N_loc, pe_table=pe_table, N=N, rd=rd,
                    pingpong=pingpong, reap=reap, differential=differential, lws=lws)
                out["path"] = "differential_coalesce" if differential else "coalesce_dyn"
                out["use_3wg"] = use_3wg
            except (DSLBaseError, ValueError, RuntimeError, torch.cuda.OutOfMemoryError) as _be:
                build_err = _be
        elif fam == "strided":
            # strided per-row putwarp: the ring slot is a per-TILE box (grid, rd, epi_m=128, tile_n), NOT the
            # coalesce full-N band. Best-effort (no prior harness reference; a build error -> graceful skip).
            ring_buf = _symmetric_empty((grid_ctas, rd, 128, tile_n), torch.bfloat16, device)
            ring_buf.zero_()
            stage_buf = ring_buf
            scratch_bytes = ring_buf.numel() * dtsize
            scratch_formula = f"ring(grid={grid_ctas}·rd={rd}·128·tile_n={tile_n})"
            ring_view = from_dlpack(ring_buf, assumed_align=16).mark_layout_dynamic()
            try:  # BLOCKER-1: isolate the compile for the consensus below
                compiled, run_args, use_3wg = _build_strided_putwarp(
                    A, Bt, cA, cB, cD, recv, ring_view, pe_dev_c, cp=cp, cp0=cp0, cp1=cp1,
                    my_cp_rank=int(pm.my_cp_rank), B=B, N_loc=N_loc, pe_table=pe_table, N=N, rd=rd,
                    pingpong=pingpong)
                out["path"] = "strided_putwarp"
                out["use_3wg"] = use_3wg
            except (DSLBaseError, ValueError, RuntimeError, torch.cuda.OutOfMemoryError) as _be:
                build_err = _be
        else:  # fam == "cluster": 'cluster' (shape-adaptive) / cluster_multislot / cluster_roundpark / cnX
            n_clusters = get_max_active_clusters(cluster_n)
            stage_shape, cfg_kw, path = _cluster_forced(force_cluster, cluster_n, nt_j_pp, cp1, N_j_loc,
                                                        n_clusters, rd)
            stage_buf = _symmetric_empty(stage_shape, torch.bfloat16, device)
            stage_buf.zero_()
            scratch_bytes = stage_buf.numel() * dtsize
            scratch_formula = f"stage{tuple(stage_shape)} (n_clusters={n_clusters})"
            stage_view = from_dlpack(stage_buf, assumed_align=16).mark_layout_dynamic()
            try:  # BLOCKER-1: isolate the compile for the consensus below
                compiled, run_args, use_3wg = _build_cluster(
                    A, Bt, cA, cB, cD, recv, stage_view, pe_dev_c, cluster_n=cluster_n, cfg_kw=cfg_kw,
                    cp=cp, cp0=cp0, cp1=cp1, my_cp_rank=int(pm.my_cp_rank), B=B, N_loc=N_loc,
                    pe_table=pe_table, N=N, rd=rd, pingpong=pingpong)
                out["path"] = path
                out["n_clusters"] = n_clusters
                out["use_3wg"] = use_3wg
            except (DSLBaseError, ValueError, RuntimeError, torch.cuda.OutOfMemoryError) as _be:
                build_err = _be

        # SHARED fused call_fn for ALL fused variants (coalesce / differential / strided / cluster; NOT
        # 2kernel, which set its own call_fn above). Gated on build_err is None (a failed compile leaves
        # run_args unbound -> the closure must not capture it). The correctness GATE moved BELOW the consensus.
        if fam != "2kernel" and build_err is None:
            def _call_fused(_c=compiled, _a=run_args):
                _c(*_a)

            call_fn = _call_fused  # de-shadowed: unique def name, explicit bind (Pyright-clean)

        # BLOCKER-1 CONSENSUS: all-reduce the per-config build failure. If ANY rank failed to compile, ALL
        # ranks skip the collective gate + timing UNIFORMLY (status=ERROR) -> no rank enters the do_gate
        # kernel/oracle collective while a peer skipped it (the cross-rank desync -> hang). Every rank reaches
        # this all-reduce whether or not it built, so it never itself desyncs. (A rank that ABORTS the compile
        # process — uncatchable — is prevented upstream by AUTOTUNE_TILE_GRID, which never attempts a
        # crash-prone config.)
        _fail = 1.0 if build_err is not None else 0.0
        if da.is_distributed:
            _fail = float(da.all_reduce_max(_fail))
        if _fail > 0.0:
            out["status"] = "ERROR"
            out["err"] = (type(build_err).__name__ + ": " + repr(build_err)[:1500]) if build_err is not None \
                else "peer_build_fail (a peer rank's config compile failed; skipped in lockstep)"
            return out

        # GATE (do_gate) — reached ONLY when EVERY rank built, so the kernel/oracle collective is desync-safe.
        if do_gate:
            if fam == "2kernel":
                tri_bf16 = torch.bmm(A, Bt.transpose(-1, -2))  # bf16 in, fp32 accum — the fair cuBLAS floor
                _nvshmem_barrier()
                recv_nccl = reshard(tri_bf16)  # gate the ACTUAL timed-path bf16 output (not the fp32 oracle)
                passed, gate = _gate(recv_nccl, expected, cp=cp, Dloc=Dloc, B=B, N_i_loc=N_i_loc,
                                     N_j_loc=N_j_loc)
            else:
                recv.fill_(-99.0)
                _nvshmem_barrier()
                compiled(*run_args)
                _drain()
                passed, gate = _gate(recv, expected, cp=cp, Dloc=Dloc, B=B, N_i_loc=N_i_loc, N_j_loc=N_j_loc)

        out["scratch_bytes"] = scratch_bytes
        out["scratch_formula"] = scratch_formula
        out["scratch_over_recv"] = round(scratch_bytes / recv_bytes, 4) if recv_bytes else None
        if do_gate:
            out["correct"] = bool(passed)
            out.update({f"gate_{k}": v for k, v in gate.items()})
            if not passed:
                out["status"] = "DISQUALIFIED_wrong"
                return out
        if not timed:
            out["status"] = "check_ok"
            return out

        # ---- timing (correctness already passed) ----
        # FOOTPRINT ACCOUNTING (reporting only, NOT combo selection): torch peak-alloc delta around the
        # timed call = the torch-side per-call scratch (the 2-kernel's tri + NCCL buffers; ~0 for the fused
        # symheap variants, whose staging/recv live on the nvshmem symmetric heap, untracked by torch).
        torch.cuda.reset_peak_memory_stats(device)
        base_alloc = torch.cuda.memory_allocated(device)
        med_max, med_local, samples = _time_variant(call_fn, device, da, rounds=rounds, warmup=warmup)
        peak_delta = torch.cuda.max_memory_allocated(device) - base_alloc
        out["time_ms"] = med_max
        out["time_ms_local"] = med_local
        out["time_samples_ms"] = [round(x, 4) for x in samples]
        out["torch_peak_delta_bytes"] = int(peak_delta)  # torch-side per-call scratch (symheap NOT counted)
    finally:
        if compiled is not None:
            compiled.free()
        # NO per-tensor free: stage_buf/recv come from the recycling symmetric MemPool (see
        # _symmetric_empty). Dropping the reference returns the block; nvshmem4py's free_tensor
        # would be the wrong allocator AND would put a collective free on the GC path.
        #
        # DROPPING THE REFERENCE MEANS ALL OF THEM. Nulling `stage_buf`/`recv` alone does not
        # release the block, because several other names in this frame still alias the same
        # storage: `cD` is a view over recv, `stage_view`/`ring_view` over the staging buffer,
        # `ring_buf` IS the staging buffer, and `run_args`/`call_fn` close over the views. That is
        # the same gap `harness/targets/cluster_drain.py::_build` had -- measured there at 27.39 GiB
        # retained per rejected config, which forced a SECOND symmetric segment (a COLLECTIVE
        # nvshmem_malloc) at the next config. `main` is immune because its `free_tensor(recv)`
        # returns the heap regardless of who still references the tensor.
        #
        # Assigning to a name that this branch never bound is safe -- assignment BINDS a local, and
        # only READING an unbound one raises UnboundLocalError -- so no pre-initialization is needed
        # and none of these lines can turn a teardown into a NameError.
        stage_buf = None
        recv = None
        ring_buf = stage_view = ring_view = cD = run_args = call_fn = None
        # And the frames of `_build_cluster` / `_build_coalesce` / `_build_strided_putwarp`, which
        # take recv / cD / the staging view as PARAMETERS and are pinned by `build_err`'s traceback
        # for as long as `build_err` lives -- which, because `build_err` and that traceback form a
        # reference CYCLE through this very frame, is until the CYCLIC collector runs, at a moment
        # each rank picks for itself. `clear_frames` skips THIS frame (a frame that is executing
        # cannot be cleared), which is why the explicit nulls above are still required.
        if build_err is not None:
            traceback.clear_frames(build_err.__traceback__)
        _nvshmem_barrier()
    return out


# --------------------------------------------------------------------------- #
# per-config COMPILE-GATE (the crux): run_cell wrapped so a ptxas C7602 ("insufficient registers"),
# CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES (701), or ANY compile/launch error is CAUGHT and returned as a
# status=ERROR cell (reason truncated to ~1500 chars) — NEVER crashes the sweep. OOM is a separate skip.
# Register-busting tiles (pingpong tile_m=256/tile_n>208 -> ValueError at build; 256x256 -> 701 at launch
# even in the 3-WG layout) thus skip cleanly and the sweep continues to the next config. The catch set +
# reason-format MATCH main()'s per-cell handler exactly (the un-truncation in commit ae01723 was needed to
# surface the real tile_n=256 register cause) so the single-cell and sweep paths report failures identically.
# The failure is HOST-DETERMINISTIC from the config (same tile+shape -> same ValueError/C7602/701 on every
# rank), so all cp ranks skip the same tiles in lockstep -> no collective desync across the gate.
def _guarded_run_cell(dm, pm, da, N, variant, *, pingpong, cp0, cp1, Dloc, B, rd, rounds, warmup, timed,
                      perf_only=False, first_n=None, cluster_n=1, completion=None, lws=None):
    try:
        return _norm_cell(run_cell(dm, pm, da, N, variant, cp0=cp0, cp1=cp1, Dloc=Dloc, B=B, rd=rd,
                                   rounds=rounds, warmup=warmup, timed=timed, pingpong=pingpong,
                                   perf_only=perf_only, first_n=first_n, cluster_n=cluster_n,
                                   completion=completion, lws=lws))
    except torch.cuda.OutOfMemoryError as e:
        torch.cuda.empty_cache()  # OOM is the one recoverable capacity skip (never a memory-estimate gate)
        return _norm_cell({"N": N, "variant": variant, "status": "oom", "reason": "CUDA OutOfMemoryError",
                           **_err_fields(e)})
    except Exception as e:  # ANY per-config failure (ValueError / DSL / RuntimeError / harness bug) -> ISOLATE
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
        return _norm_cell({"N": N, "variant": variant, "status": "error",
                           "reason": "per-config build/run raised (isolated; sweep continues)",
                           **_err_fields(e)})


# --------------------------------------------------------------------------- #
# TILE SWEEP: for ONE (N, variant) walk TILE_GRID, rebinding the module TILE + pingpong per grid point
# (the same TILE-global rebind the single --tile-m/--tile-n override uses), run each through the compile-gate,
# and pick the fastest VALID (correct + timed) tile. Operands + the fp32 oracle are rebuilt per tile inside
# run_cell (identical across tiles: seeded builder -> deterministic; the finally frees the symheap recv each
# tile so there is no growth) — robustness over sweep speed on the correctness venue; the venue D perf run is
# the. Emits the per-tile survival list (status/use_3wg/correct/time) + the winner per (N,variant).
def run_tile_sweep(dm, pm, da, N, variant, *, cp0, cp1, Dloc, B, rd, rounds, warmup, timed, grid=None,
                   perf_only=False, first_n=None, disq_tiles=None):
    global TILE
    saved = TILE
    _grid = grid if grid is not None else TILE_GRID
    # BELT-AND-SUSPENDERS (#77): a per-(variant,tile) QUARANTINE set carried across N. A tile DISQUALIFIED_wrong
    # at the anchor gate N (correct=False) is quarantined so it can NEVER be timed/picked at a LARGER N (where
    # --perf-only skips the gate -> run_cell would return status='ok'+time_ms and a silent-wrong tile could win
    # the sweep, tainting the perf answer). Closes the taint CLASS even if a future config-guard is missed.
    _disq = disq_tiles if disq_tiles is not None else set()
    tiles = []
    try:
        for (tm, tn, pp) in _grid:
            TILE = (tm, tn)
            key = (variant, tm, tn, pp)
            _nvshmem_barrier()  # lockstep entry: every rank starts this tile together (symmetric skip below)
            if key in _disq:  # quarantined by the anchor-N gate -> never time/pick at a larger N
                sub = {"status": "skip_anchor_disq", "tile_m": tm, "tile_n": tn, "pingpong": pp,
                       "tile_wall_s": 0.0}
                tiles.append(sub)
                if dm.rank == 0:
                    print(f"    [tile] N={N:>6} {variant:>9} tile=({tm},{tn}) pp={int(pp)} "
                          f"status= skip_anchor_disq (wrong@anchor N={first_n})", flush=True)
                continue
            t0 = time.perf_counter()
            sub = _guarded_run_cell(dm, pm, da, N, variant, pingpong=pp, cp0=cp0, cp1=cp1, Dloc=Dloc, B=B,
                                    rd=rd, rounds=rounds, warmup=warmup, timed=timed, perf_only=perf_only,
                                    first_n=first_n)
            # Stamp the config on EVERY sub (the ERROR/OOM branches return a minimal dict without them).
            sub["tile_m"], sub["tile_n"], sub["pingpong"] = tm, tn, pp
            sub["tile_wall_s"] = round(time.perf_counter() - t0, 2)
            if sub.get("correct") is False:  # DISQUALIFIED_wrong at the gate -> quarantine for all larger N
                _disq.add(key)
            tiles.append(sub)
            if dm.rank == 0:
                print(f"    [tile] N={N:>6} {variant:>9} tile=({tm},{tn}) pp={int(pp)} "
                      f"status={sub.get('status'):>16} use3wg={sub.get('use_3wg', '-')} "
                      f"correct={sub.get('correct', '-')} relL2={sub.get('gate_rel_l2', '-')} "
                      f"time_ms={sub.get('time_ms', '-')} err={str(sub.get('err', ''))[:80]}", flush=True)
    finally:
        TILE = saved  # restore the module global whatever happened (never leak a swept TILE to later cells)
    # Winner = fastest tile that both PASSED correctness and was TIMED (status 'ok'), EXCLUDING any tile
    # quarantined at the anchor gate. In --check-only there is no time, so fall back to reporting the correct
    # tiles (the correctness-venue answer to "which tiles route + are correct", NOT a perf pick).
    valid = [t for t in tiles if t.get("status") == "ok" and t.get("time_ms") is not None
             and (variant, t.get("tile_m"), t.get("tile_n"), t.get("pingpong")) not in _disq]
    if valid:
        w = min(valid, key=lambda t: t["time_ms"])
        winner = {"tile_m": w["tile_m"], "tile_n": w["tile_n"], "pingpong": w["pingpong"],
                  "time_ms": w["time_ms"], "use_3wg": w.get("use_3wg")}
    else:
        correct = [(t["tile_m"], t["tile_n"], t["pingpong"]) for t in tiles if t.get("correct")]
        winner = {"correct_tiles": correct} if correct else None
    return {"N": N, "variant": variant, "sweep": True, "tiles": tiles, "winner": winner,
            "n_valid": len(valid), "n_correct": sum(1 for t in tiles if t.get("correct")),
            "status": "sweep_ok"}


# --------------------------------------------------------------------------- #
# CUT-VS-KEEP AUTOTUNE: for ONE (N, variant) sweep the applicable knobs -- tile (the passed grid) x cluster_n
# {1,2,4,8} (cluster family only, else [1]) x completion {ib_quiet, ib_reap} (coalesce/differential only,
# else [ib_quiet]) -- through the compile-gate, and pick the fastest VALID (correct+timed) config as the
# variant's autotuned wall-time for this cell. rd is FIXED (footprint knob, not the perf lever). Emits every
# sub-config's survival + the winner. Mirrors run_tile_sweep + the two extra knob loops + the anchor-quarantine.
# --------------------------------------------------------------------------- #
def run_autotune_cell(dm, pm, da, N, variant, *, cp0, cp1, Dloc, B, rd, rounds, warmup, timed, grid=None,
                      perf_only=False, first_n=None, disq_tiles=None, cluster_ns=(1,), lws=None):
    global TILE
    saved = TILE
    # ARBITRARY-N + arbitrary-cp: a shape-INVALID (N,cp) (16-B HARD RULE only) -> a LOUD structured skip_shape
    # cell (never silent, never a crash). Other N still run. NO GPU touched on this path.
    _ok, _reason = _shape_check(N, cp0, cp1)
    if not _ok:
        return _norm_cell({"N": N, "variant": variant, "cp0": cp0, "cp1": cp1, "Dloc": Dloc, "B": B,
                           "autotune": True, "configs": [], "winner": None, "n_valid": 0, "n_correct": 0,
                           "status": "skip_shape", "reason": _reason})
    _grid = grid if grid is not None else TILE_GRID
    # 2kernel = cuBLAS GEMM + NCCL a2a: TILE is irrelevant (no fused kernel) -> ONE tile (avoid N redundant
    # timings of the same cuBLAS). The fused variants sweep the full grid.
    if _variant_family(variant) == "2kernel":
        _grid = [(128, 128, False)]
    _disq = disq_tiles if disq_tiles is not None else set()
    cns = _variant_cluster_ns(variant, cluster_ns)
    comps = _variant_completions(variant)
    configs = []
    try:
        for cn in cns:
            for comp in comps:
                for (tm, tn, pp) in _grid:
                    TILE = (tm, tn)
                    tkey = (variant, tm, tn, pp)  # anchor-quarantine is tile-scoped (config-independent)
                    _nvshmem_barrier()            # lockstep entry: every rank starts this config together
                    if tkey in _disq:
                        configs.append({"status": "skip_anchor_disq", "cluster_n": cn, "completion": comp,
                                        "tile_m": tm, "tile_n": tn, "pingpong": pp, "cfg_wall_s": 0.0})
                        continue
                    t0 = time.perf_counter()
                    sub = _guarded_run_cell(dm, pm, da, N, variant, pingpong=pp, cp0=cp0, cp1=cp1, Dloc=Dloc,
                                            B=B, rd=rd, rounds=rounds, warmup=warmup, timed=timed,
                                            perf_only=perf_only, first_n=first_n, cluster_n=cn,
                                            completion=comp, lws=lws)
                    sub["cluster_n"], sub["completion"] = cn, comp
                    sub["tile_m"], sub["tile_n"], sub["pingpong"] = tm, tn, pp
                    sub["cfg_wall_s"] = round(time.perf_counter() - t0, 2)
                    if sub.get("correct") is False:  # DISQUALIFIED_wrong at the anchor gate -> quarantine tile
                        _disq.add(tkey)
                    configs.append(_norm_cell(sub))  # structured {status, error_class, error_msg, reason}
                    if dm.rank == 0:
                        # cfg_wall_s = compile + (gate) + timing. With cache=1 the COLD compile is the FIRST
                        # (variant,cn,comp,tile) occurrence at the SMALLEST N (timing cheap there); a >5s wall
                        # AT THE ANCHOR N flags the ≤5s-compile HARD RULE (constexpr-loop-nested-if). At large
                        # N the O(N³) timing dominates the wall (NOT a compile signal) -> read the anchor N.
                        _slow = "  <<5s-CHECK-COMPILE" if (sub.get("cfg_wall_s", 0) > 5.0) else ""
                        print(f"    [cfg] N={N:>6} {variant:>17} cn={cn} {comp:>8} tile=({tm},{tn}) "
                              f"pp={int(pp)} status={str(sub.get('status')):>16} "
                              f"correct={sub.get('correct', '-')} time_ms={sub.get('time_ms', '-')} "
                              f"wall={sub.get('cfg_wall_s', '-')}s{_slow} "
                              f"err={str(sub.get('err', ''))[:70]}", flush=True)
    finally:
        TILE = saved  # never leak a swept TILE to later cells
    valid = [c for c in configs if c.get("status") == "ok" and c.get("time_ms") is not None
             and (variant, c.get("tile_m"), c.get("tile_n"), c.get("pingpong")) not in _disq]
    if valid:
        w = min(valid, key=lambda c: c["time_ms"])
        winner = {"cluster_n": w.get("cluster_n"), "completion": w.get("completion"), "tile_m": w["tile_m"],
                  "tile_n": w["tile_n"], "pingpong": w["pingpong"], "time_ms": w["time_ms"],
                  "path": w.get("path"), "use_3wg": w.get("use_3wg")}
    else:
        correct = [(c.get("cluster_n"), c.get("completion"), c["tile_m"], c["tile_n"], c["pingpong"])
                   for c in configs if c.get("correct")]
        winner = {"correct_configs": correct} if correct else None
    # CELL status: ok if >=1 config timed valid; else error with the aggregated reason (every config failed/
    # skipped) so a whole-cell blowup is REPORTED, not silent. The cell lifts the FIRST failing config's
    # structured error into its own {error_class, error_msg} so the summary/reduce show the WHY at cell level.
    _extra = {}
    if valid:
        cell_status, reason = "ok", ""
    else:
        _errs = [c for c in configs if _norm_cell(c).get("status") in ("error", "oom")]
        reason = (f"all {len(configs)} config(s) failed/skipped; first error: "
                  f"{(_errs[0].get('error_msg') or '')[:200]}" if _errs
                  else f"no valid+timed config among {len(configs)} (all disq/skip)")
        cell_status = "error" if _errs else "ok"  # all-disq/skip is not itself an error (data absent, not broken)
        if _errs:
            _extra = {"error_class": _errs[0].get("error_class"), "error_msg": _errs[0].get("error_msg")}
    return _norm_cell({"N": N, "variant": variant, "autotune": True, "configs": configs, "winner": winner,
                       "n_valid": len(valid), "n_correct": sum(1 for c in configs if c.get("correct")),
                       "status": cell_status, "reason": reason, **_extra})


def _check_config():
    """DRY config-mapping assertion (no GPU / no nvshmem / no compile) -- the sm_120 local dry-check. Asserts
    each of the 7 --variants names resolves to the intended (family, force_cluster, differential) + the right
    autotune knob sets. Run: python back_a2a_store_bench.py --check-config."""
    expect = {  # name -> (family, force_cluster, differential, completions, is_cluster)
        "coalesce":          ("coalesce", None,        False, ["ib_quiet", "ib_reap"], False),
        "differential":      ("coalesce", None,        True,  ["ib_quiet", "ib_reap"], False),
        "strided_putwarp":   ("strided",  None,        False, ["ib_quiet"],            False),
        "cluster":           ("cluster",  None,        False, ["ib_quiet"],            True),
        "cluster_multislot": ("cluster",  "multislot", False, ["ib_quiet"],            True),
        "cluster_roundpark": ("cluster",  "roundpark", False, ["ib_quiet"],            True),
        "2kernel":           ("2kernel",  None,        False, ["ib_quiet"],            False),
    }
    ok = True
    for name in VARIANT_NAMES:
        fam, force, diff = _VARIANT_SPEC[name]
        comps = _variant_completions(name)
        cns = _variant_cluster_ns(name, [1, 2, 4, 8])
        is_cluster = (cns == [1, 2, 4, 8])
        exp = expect[name]
        got = (fam, force, diff, comps, is_cluster)
        want = (exp[0], exp[1], exp[2], exp[3], exp[4])
        good = (got == want)
        ok = ok and good
        print(f"  {'OK ' if good else 'BAD'} {name:>17} -> family={fam:>9} force={str(force):>9} "
              f"differential={diff!s:>5} completion={comps} cluster_n={cns}")
    # legacy cnX still routes to the cluster family (back-compat)
    for legacy in ("cn1", "cn2", "cn4", "cn8"):
        good = (_variant_family(legacy) == "cluster")
        ok = ok and good
        print(f"  {'OK ' if good else 'BAD'} {legacy:>17} -> family=cluster (legacy back-compat)")
    print("CHECK_CONFIG_OK" if ok else "CHECK_CONFIG_FAIL")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N-list", default=None,
                    help="COMMA-SEP N sweep. Default: the straddle+even-shard venue D lists.")
    ap.add_argument("--cluster-list", default="1,2,4,8", help="COMMA-SEP cluster_n values (default 1,2,4,8).")
    ap.add_argument("--variants", default="cluster,coalesce,2kernel",
                    help="COMMA-SEP drain variants. CUT-VS-KEEP names: coalesce, differential, strided_putwarp, "
                    "cluster, cluster_multislot, cluster_roundpark, 2kernel (the 7 live variants). Legacy "
                    "cn1/cn2/cn4/cn8 (shape-adaptive cluster) still accepted. Default: cluster,coalesce,2kernel.")
    ap.add_argument("--autotune", action="store_true", help="CUT-VS-KEEP autotune: for each (N, variant) sweep "
                    "tile x cluster_n (cluster family) x completion {ib_quiet,ib_reap} (coalesce/differential) "
                    "and report the fastest VALID config as the variant's wall-time. Subsumes --tile-sweep "
                    "(owns the tile loop); rd stays fixed. EXCLUDES the known-spilling 256x256 tile by default.")
    ap.add_argument("--local-world-size", type=int, default=8, help="GPUs per node (differential variant peer "
                    "classification: is_local(peer)=pe_table[peer]//lws==my_node). venue D = 8.")
    ap.add_argument("--check-config", action="store_true", help="DRY config-mapping assertion (no GPU/nvshmem/"
                    "compile): assert each --variants name -> the intended config + knobs. The sm_120 dry-check.")
    ap.add_argument("--D", type=int, default=16, help="total feature dim D; per-rank Dloc = D // cp (L=Dloc·B).")
    ap.add_argument("--tile-m", type=int, default=None, help="override the fused GEMM cta_tile_M (default: the "
                    "module TILE[0]=128). WS-A tile-autotune probe: does the drain run+benefit at a wider tile?")
    ap.add_argument("--tile-n", type=int, default=None, help="override the fused GEMM cta_tile_N (default: the "
                    "module TILE[1]=128). tile_n changes nt_j_pp=ceil(N_j_loc/tile_n) -> the even_shard/straddle route.")
    ap.add_argument("--B", type=int, default=1)
    ap.add_argument("--rd", type=int, default=2)
    ap.add_argument("--rounds", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--check-only", action="store_true", help="correctness + footprint only (no timing).")
    ap.add_argument("--tile-sweep", action="store_true", help="WS-S2 compile-gated tile autotune: for each "
                    "(N, variant) sweep tile_m×tile_n×pingpong (TILE_GRID), SKIP register-busters via the "
                    "per-config compile-gate, and pick the fastest surviving tile. Additive/default-off; "
                    "IGNORES --tile-m/--tile-n (the sweep owns TILE). Combine with --check-only for a "
                    "correctness+3WG-route smoke (no timing).")
    ap.add_argument("--perf-only", action="store_true", help="PERF/venue D grid mode: SKIP the fp32 O(N³) "
                    "reshard oracle + its O(N²·cp) all_gather (~18GB @N=24576) + the correctness gate; compile "
                    "+ TIME only. Correctness is proven separately on venue E. Requires timed (not --check-only).")
    ap.add_argument("--tile-grid", default=None, help="COMMA-SEP tile-sweep grid override: 'MxN' or 'MxN:pp' "
                    "(pp in {0,1}, default 0), e.g. '128x128,128x256'. Default (None) = the full 8-pt TILE_GRID. "
                    "SCOPES the sweep to a compile-affordable subset for the venue E correctness venue (the "
                    "~195s straddle bitcode cliff / task #72 makes the full grid cache-off intractable in one "
                    "job); the full perf grid is the coordinator's venue D run.")
    ap.add_argument("--out-dir", default=None, help="write one JSON per N (roundpark_fp_D<D>_N<N>.json).")
    args = ap.parse_args()

    # DRY config-mapping check (no GPU / no dist init) -- the sm_120 local dry-check. Exits immediately.
    if args.check_config:
        import sys as _sys
        _sys.exit(_check_config())

    # WS-A tile-autotune probe: rebind the module TILE global so run_cell + _build_cluster/_build_coalesce
    # (which read TILE) build the fused GEMM at the requested cta tile. The perf harness MUST sweep the tile
    # (the 2-kernel's cuBLAS autotunes per shape; freezing the fused side at 128x128 is a perf-neutrality bug).
    global TILE
    if args.tile_m is not None or args.tile_n is not None:
        TILE = (args.tile_m if args.tile_m is not None else TILE[0],
                args.tile_n if args.tile_n is not None else TILE[1])
    tile_grid = _parse_tile_grid(args.tile_grid)  # scoped tile-sweep grid (None -> full 8-pt TILE_GRID)
    # --autotune uses the AUTOTUNE-VALIDATED grid (no pingpong / no 256×256 that ValueError on SM90) so it
    # never ATTEMPTS a crash-prone config; --tile-grid overrides it explicitly.
    autotune_grid = tile_grid if args.tile_grid else AUTOTUNE_TILE_GRID

    # STANDALONE dist-init (mirrors gate_c_overlap.main / conftest): SLURM/torchrun env -> set_device ->
    # DistributedManager.initialize(2-D mesh) -> init_nvshmem.
    if hasattr(DistributedManager, "_derive_dist_env_from_slurm"):
        DistributedManager._derive_dist_env_from_slurm()
    _local = int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", "0")))
    torch.cuda.set_device(_local)
    spec = os.environ.get("CPO_DIST_MESH", "")
    if spec:
        mesh = OrderedDict()
        for tok in spec.split(","):
            name, size = tok.strip().split("=", 1)
            parts = [int(p) for p in size.strip().split("*")]
            mesh[name.strip()] = tuple(parts) if len(parts) > 1 else parts[0]
    else:
        mesh = OrderedDict(cp=int(os.environ["WORLD_SIZE"]))
    os.environ.setdefault("CPO_DISTRIBUTED_INIT_METHOD", "ENV")
    DistributedManager.initialize(mesh, device_type="cuda")
    dm = DistributedManager()
    DistributedManager.init_nvshmem()
    pm = _trimul_placements(dm)[1]
    da = resolve_dist(dm)  # shared bench adapter: barrier + all_reduce(MAX) slowest-PE consensus
    axis = _cp_axis_sizes(dm)
    # ARBITRARY cp: 2-D (cp0,cp1) OR 1-D (cp,) -> (cp0=cp, cp1=1); non-pow2 cp is fine. The kernel's cp1==1
    # path is the 1-D base (§ generic 1D/2D sharding); the shape gate (per-axis %8) handles validity per N.
    if len(axis) == 2:
        cp0, cp1 = int(axis[0]), int(axis[1])
    elif len(axis) == 1:
        cp0, cp1 = int(axis[0]), 1
    else:
        raise ValueError(f"cp mesh must be 1-D (cp=N) or 2-D (cp=CP0*CP1); got {axis}")
    cp = cp0 * cp1
    assert get_device_capacity()[0] == 9, "SM90 only"
    assert args.D % cp == 0, f"--D {args.D} must be divisible by cp={cp}"
    Dloc = args.D // cp
    assert Dloc >= 1, f"--D {args.D} gives Dloc={Dloc} < 1"

    Ns = [int(x) for x in args.N_list.split(",") if x.strip()] if args.N_list else (STRADDLE_NS + EVEN_NS)
    # PERF-ONLY correctness-gate anchor: the SMALLEST shape-valid N (oracle cheap there). In --perf-only the
    # correctness gate runs ONLY at this N per (variant,tile); every larger N times with the oracle SKIPPED.
    # #78: the anchor gate N must be NON-PARTIAL for the WIDEST swept tile. At N_j_loc < tile_n (or
    # N_i_loc < tile_m) the widest tile is a PARTIAL tile AT THE GATE and can spuriously DISQUALIFY, which
    # then quarantines it at every larger N (the tile never times). Require N_i_loc >= max(tile_m) AND
    # N_j_loc >= max(tile_n) across the swept grid (tile_n=256 @ cp1=8 -> N >= 2048; the grid's 2624 works).
    # Single-tile (no --tile-sweep) uses the current TILE.
    # --autotune ALSO sweeps the tile grid (up to 256) -> the anchor gate N must be non-partial for the
    # WIDEST swept tile (else a 256-tile is PARTIAL at the anchor and spuriously DISQUALIFIES -> quarantined
    # at every larger N). So widen the anchor sizing for --autotune too (not just --tile-sweep).
    _sweep_tiles = ((autotune_grid if args.autotune else tile_grid)
                    if (args.tile_sweep or args.autotune) else [(TILE[0], TILE[1])])
    _max_tm = max(t[0] for t in _sweep_tiles)
    _max_tn = max(t[1] for t in _sweep_tiles)
    _valid_ns = [N for N in Ns if not (N % cp0 or N % cp1 or (N // cp0) % 8 or (N // cp1) % 8)
                 and (N // cp0) >= _max_tm and (N // cp1) >= _max_tn]
    perf_gate_n = min(_valid_ns) if _valid_ns else None
    # PERF-ONLY: process the anchor gate N FIRST so the per-(variant,tile) DISQUALIFIED quarantine is populated
    # BEFORE any larger N (a wrong tile must be quarantined before it can be timed-as-ok at a no-gate large N).
    if args.perf_only and perf_gate_n is not None:
        Ns = [perf_gate_n] + [N for N in Ns if N != perf_gate_n]
    disq_tiles = set()  # (variant, tile_m, tile_n, pingpong) DISQUALIFIED_wrong at the anchor gate (#77 net)
    cluster_ns = [int(x) for x in args.cluster_list.split(",") if x.strip()]
    _req = [v.strip() for v in args.variants.split(",") if v.strip()]
    # variant list. --autotune: the CUT-VS-KEEP names pass through DIRECTLY (the cluster family autotunes
    # cluster_n INTERNALLY per cell -> no cnX expansion). Legacy (non-autotune): the old cluster->cnX
    # expansion + the individual names as single cells (back-compat for existing sbatches).
    variant_list = []
    if args.autotune:
        variant_list = [v for v in VARIANT_NAMES if v in _req] + [v for v in _req if v not in VARIANT_NAMES]
    else:
        inc = set(_req)
        if "cluster" in inc:
            variant_list += [f"cn{c}" for c in cluster_ns]
        for name in ("coalesce", "differential", "strided_putwarp", "cluster_multislot",
                     "cluster_roundpark", "2kernel"):
            if name in inc:
                variant_list += [name]

    if dm.rank == 0:
        print(f"[roundpark-fp] cp=({cp0},{cp1}) D={args.D} Dloc={Dloc} B={args.B} rd={args.rd} "
              f"rounds={args.rounds} warmup={args.warmup} check_only={args.check_only}", flush=True)
        print(f"[roundpark-fp] Ns={Ns}", flush=True)
        print(f"[roundpark-fp] variants={variant_list}", flush=True)

    all_cells = []  # (N,variant) cells across the whole run -> the end-of-run SUMMARY table
    for N in Ns:
        # N-LOOP ISOLATION: one bad N (shape / barrier / any exception) -> logged + skipped -> the sweep
        # proceeds to the rest and still writes JSONs for the good N. Only a fatal OOM/timeout ends the job.
        try:
            _ok, _reason = _shape_check(N, cp0, cp1)
            if not _ok:
                # ARBITRARY-N: shape-INVALID (16-B HARD RULE) -> LOUD structured skip_shape (one cell per
                # variant so the matrix shows WHY), never a silent continue. Other N still run.
                cells = [_norm_cell({"N": N, "variant": v, "cp0": cp0, "cp1": cp1, "Dloc": Dloc, "B": args.B,
                                     "status": "skip_shape", "reason": _reason}) for v in variant_list]
                all_cells += cells
                if dm.rank == 0:
                    print(f"[skip_shape] N={N} cp=({cp0},{cp1}): {_reason}", flush=True)
                    if args.out_dir:
                        os.makedirs(args.out_dir, exist_ok=True)
                        with open(os.path.join(args.out_dir, f"roundpark_fp_D{args.D}_N{N}.json"), "w") as f:
                            json.dump({"N": N, "cp0": cp0, "cp1": cp1, "D": args.D, "Dloc": Dloc,
                                       "B": args.B, "cells": cells}, f, indent=2)
                continue
            cells = []
            for variant in variant_list:
                if dm.rank == 0:
                    print(f"  [start] N={N} variant={variant} (nt_j={(N // cp1 + TILE[1] - 1) // TILE[1]})"
                          f"{' [autotune]' if args.autotune else (' [tile-sweep]' if args.tile_sweep else '')}",
                          flush=True)
                _nvshmem_barrier()  # host-deterministic entry -> every rank starts this variant together
                t0 = time.perf_counter()
                # VARIANT-LOOP ISOLATION: run_autotune_cell already returns (never raises) on a per-config
                # failure; this try catches a whole-cell blowup (a harness bug / non-config exception) so ONE
                # bad variant -> ERROR cell + the next variant continues (host-deterministic -> lockstep).
                try:
                    if args.autotune:
                        cell = run_autotune_cell(dm, pm, da, N, variant, cp0=cp0, cp1=cp1, Dloc=Dloc, B=args.B,
                                                 rd=args.rd, rounds=args.rounds, warmup=args.warmup,
                                                 timed=not args.check_only, grid=autotune_grid,
                                                 perf_only=args.perf_only, first_n=perf_gate_n,
                                                 disq_tiles=disq_tiles, cluster_ns=cluster_ns,
                                                 lws=args.local_world_size)
                    elif args.tile_sweep:
                        cell = run_tile_sweep(dm, pm, da, N, variant, cp0=cp0, cp1=cp1, Dloc=Dloc, B=args.B,
                                              rd=args.rd, rounds=args.rounds, warmup=args.warmup,
                                              timed=not args.check_only, grid=tile_grid,
                                              perf_only=args.perf_only, first_n=perf_gate_n,
                                              disq_tiles=disq_tiles)
                    else:
                        cell = _guarded_run_cell(dm, pm, da, N, variant, pingpong=False, cp0=cp0, cp1=cp1,
                                                 Dloc=Dloc, B=args.B, rd=args.rd, rounds=args.rounds,
                                                 warmup=args.warmup, timed=not args.check_only,
                                                 perf_only=args.perf_only, first_n=perf_gate_n)
                except torch.cuda.OutOfMemoryError as e:
                    torch.cuda.empty_cache()
                    cell = {"N": N, "variant": variant, "status": "oom",
                            "reason": "cell-level OOM (isolated)", **_err_fields(e)}
                except Exception as e:
                    cell = {"N": N, "variant": variant, "status": "error",
                            "reason": "cell-level exception (isolated; next variant continues)",
                            **_err_fields(e)}
                cell = _norm_cell(cell)
                cell["wall_s"] = round(time.perf_counter() - t0, 2)
                cells.append(cell)
                all_cells.append(cell)
                if dm.rank == 0:
                    st = cell.get("status")
                    if cell.get("autotune"):
                        print(f"  N={N:>6} {variant:>17} AUTOTUNE status={st} n_valid={cell.get('n_valid','-')} "
                              f"n_correct={cell.get('n_correct','-')} winner={cell.get('winner')} "
                              f"reason={cell.get('reason','')[:60]} wall={cell['wall_s']}s", flush=True)
                    elif cell.get("sweep"):
                        print(f"  N={N:>6} {variant:>9} SWEEP n_valid={cell.get('n_valid','-')} "
                              f"n_correct={cell.get('n_correct','-')} winner={cell.get('winner')} "
                              f"wall={cell['wall_s']}s", flush=True)
                    else:
                        print(f"  N={N:>6} {variant:>9} status={st:>16} path={cell.get('path','-'):>16} "
                              f"correct={cell.get('correct','-')} time_ms={cell.get('time_ms','-')} "
                              f"reason={cell.get('reason','')[:50]} wall={cell['wall_s']}s", flush=True)
                    # INCREMENTAL write after EACH cell -> a later hang/OOM/timeout preserves the cells so far.
                    if args.out_dir:
                        os.makedirs(args.out_dir, exist_ok=True)
                        p = os.path.join(args.out_dir, f"roundpark_fp_D{args.D}_N{N}.json")
                        with open(p, "w") as f:
                            json.dump({"N": N, "cp0": cp0, "cp1": cp1, "D": args.D, "Dloc": Dloc,
                                       "B": args.B, "cells": cells}, f, indent=2)
        except Exception as e:  # N-level blowup (never kills the sweep) -> log + continue to the next N
            if dm.rank == 0:
                print(f"[N-ERROR] N={N} cp=({cp0},{cp1}) skipped: {type(e).__name__}: {repr(e)[:200]}",
                      flush=True)
            all_cells.append(_norm_cell({"N": N, "variant": "*", "status": "error",
                                         "reason": "N-loop exception (isolated)", **_err_fields(e)}))
            continue

    if dm.rank == 0:
        _print_run_summary(all_cells, variant_list, Ns)
    if dist.is_available() and dist.is_initialized():
        dist.barrier(device_ids=[_local])
    DistributedManager.cleanup()


if __name__ == "__main__":
    main()
