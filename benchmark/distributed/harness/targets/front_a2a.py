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

"""FRONT-A2A adapter (HARNESS_DESIGN §7) — the front DualGatedGEMM+A2A as BenchTargets, thin plug-ins over
the EXISTING bench_front_a2a_staged builders (`_compile_front_staged`, `_front_a2a_plan_dmajor`). NO kernel
logic is duplicated here. Three names:

  * `front`         (role=target)   — the FUSED front: DualGatedGemmDistSm90 (configure_a2a) with the
                                      in-epilogue D-major peer TMA-S2G store. run() = the single fused call.
                                      comm_bytes = V_a2a -> the fused effective BW (comm_bytes/t_fused), the
                                      wide-put throughput as the app sees it (refinement 1).
  * `front_2kernel` (role=baseline) — the TWO-KERNEL head-to-head: the SAME kernel WITHOUT the fused store
                                      (local D-major store) THEN a torch all_to_all feature reshard.
                                      comm_bytes = V_a2a (the reshard is a separate timed leg); cell_meta
                                      = gemm_cfg (the heuristic DualGatedGEMM config, Table-B column).
  * `front_gemm`    (role=baseline) — GEMM-ONLY: the LOCAL DualGatedGEMM projection (enable_a2a=False), NO
                                      all_to_all -> the cell's time_ms IS t_gemm in isolation (so Table B
                                      reports t_gemm, t_a2a, total WITHOUT cross-run subtraction). cell_meta
                                      = gemm_cfg.
  * `front_a2a`     (role=baseline) — COMM-ONLY: just dist.all_to_all_single over the front reshard shapes,
                                      so the cell's time_ms IS t_a2a in isolation -> BW = V_a2a / t_a2a.

FRONT vs BACK contract. The front reshards over the FEATURE dim (K=D, projection) — it is EXEMPT from the
K=N_token HARD RULE (that guards the BACK einsum). FAITHFUL strong-scaling convention: rows_per_peer =
B·N²/cp (BOTH token axes sharded), so M_full = B·N² EXACTLY matches the §0.2 roofline (M_full = B·N_token²).
The earlier linear-token proxy (rows_per_peer = B·N) is RETIRED — it measured a weak-on-tokens O(N) GEMM,
not the true strong-scaling front; B(i)/B(ii) are re-benched faithful, so D-major and route2_ni differ ONLY
in store layout at the SAME M=N²/cp (a genuine head-to-head). O(N²) operands -> the large-N/small-cp cells
OOM (runtime OutOfMemoryError-skip, NEVER a memory gate). comm_bytes is the ACTUAL off-diagonal reshard
volume 2·Dloc·rows_per_peer·(cp-1)·dtype (derived from the real send buffer).

Front feature-scatter floor (memory front_feature_scatter_dloc16_floor): Dloc = D/cp >= 16 AND the
best_front_tile's tile_N%32==0 AND Dloc % (tile_N//2) == 0 — a DETERMINISTIC skip_shape on every rank.
"""

import os

from benchmark.distributed.harness.shape_gate import shape_check
from benchmark.distributed.harness.registry import register
from benchmark.distributed.harness.target import BenchTarget

_DTYPE_BYTES = 2  # bf16
_K = 256  # front contraction (feature/hidden) — the O(N²) projection K, NOT K=N_token
_EPS = 1e-5


def _front_geom(ctx):
    """(cp, Dloc, D, M_full, rows_per_peer) for THIS cell. FAITHFUL strong-scaling: rows_per_peer =
    B·N_i_loc·N_j_loc = B·N²/cp (BOTH token axes sharded — the §0.2 roofline pair grid), so M_full =
    cp·rows_per_peer = B·N² EXACTLY matches the roofline M_full = B·N_token². (The linear-token proxy
    rows_per_peer = B·N is RETIRED — it measured a weak-on-tokens O(N) GEMM, not the true strong-scaling
    front; B(i)/B(ii) are re-benched faithful too, so D-major vs route2_ni differ ONLY in store layout at
    the SAME M.) O(N²) operands -> the large-N/small-cp cells OOM (runtime OutOfMemoryError-skip, NEVER a
    memory gate). Pure shape math -> IDENTICAL on every rank."""
    cp0, cp1 = int(ctx.cp0), int(ctx.cp1)
    cp = cp0 * cp1
    Dloc = int(ctx.Dloc)
    D = Dloc * cp
    N = int(ctx.N)
    rows_per_peer = (
        int(ctx.B) * (N // cp0) * (N // cp1)
    )  # = B·N²/cp (pair grid; mirrors _route2_geom)
    M_full = cp * rows_per_peer  # = B·N²  (the §0.2 roofline)
    return cp, Dloc, D, M_full, rows_per_peer


def _front_comm_bytes(ctx):
    """Per-rank off-diagonal A2A volume (bytes) = 2·Dloc·rows_per_peer·(cp-1)·dtype — derived from the ACTUAL
    reshard tensors: _front_a2a_plan_dmajor sends 2·Dloc·rows_per_peer elements to EACH of (cp-1) remote
    peers (the 1/cp diagonal stays local). rows_per_peer = B·N²/cp (faithful strong-scaling) -> the reported
    bytes are EXACTLY what run() transfers at this (rows_per_peer, Dloc, cp)."""
    cp, Dloc, _, _, rpp = _front_geom(ctx)
    if cp <= 1:
        return 0
    return int(2 * Dloc * rpp * (cp - 1) * _DTYPE_BYTES)


# ------------------------------------------------------------------------------------------- supports() ---
def _front_baseline_supports(ctx):
    """BASELINE gate (front_gemm / front_2kernel = LOCAL (2D,M) store + torch all_to_all, NO fused peer
    feature-scatter): 16-B TMA gate + the auto-clamp M-floor ONLY. The Dloc>=16 / tile_N%32 / Dloc%postact
    floors are the FUSED PEER-store's constraints (memory front_feature_scatter_dloc16_floor) — the local
    store does NOT do the feature scatter, so they are NOT applied here (=> D=128/cp16 Dloc=8 runs the
    baseline, per the Table B footnote). Deterministic on every rank."""
    ok, reason = shape_check(ctx.N, ctx.cp0, ctx.cp1)
    if not ok:
        return reason
    cp, Dloc, _, _, M = _front_geom(ctx)  # rows_per_peer == M == B·N²/cp (faithful strong-scaling)
    from fold_cp_ops.distributed.dual_gated_gemm_a2a import DualGatedGemmDistSm90

    tile_M, _ = DualGatedGemmDistSm90.best_front_tile(Dloc)
    rem = (
        M % tile_M
    )  # the store auto-clamps a partial last M-tile (>=8), but a 1..7 remainder hits the
    if (
        rem != 0 and rem < 8
    ):  # kernel's defensive 16-B floor -> gate it out deterministically (bench:303)
        return f"M=ctx.N={M} % tile_M={tile_M}={rem} in 1..7 (16-B auto-clamp floor)"
    return None


def _front_supports(ctx):
    """FUSED-front gate: the baseline gate PLUS the peer feature-scatter floors (Dloc>=8, best tile
    tile_N%16==0, Dloc%postact==0) that the in-epilogue D-major peer store requires. Deterministic.
    Dloc>=8 (was >=16) since tile_N%32->%16 (bedfe71) + the a2a store-op x4->x2 fix (d1864c3):
    postact=tile_N/2 reaches 8 -> Dloc=8 / D128-cp16 runs FUSED via ib_drain."""
    reason = _front_baseline_supports(ctx)
    if reason is not None:
        return reason
    cp, Dloc, _, _, M = _front_geom(ctx)
    if Dloc < 8:
        return f"Dloc={Dloc}<8 (front feature-scatter floor D/cp>=8 post-fix; postact=tile_N/2 store <16 B below Dloc=8)"
    from fold_cp_ops.distributed.dual_gated_gemm_a2a import DualGatedGemmDistSm90

    tile_M, tile_N = DualGatedGemmDistSm90.best_front_tile(Dloc)
    if tile_N % 16 != 0:
        return f"best_front_tile tile_N={tile_N}%16 != 0 (Dloc={Dloc}; STSM-quad floor, was %32)"
    postact = tile_N // 2
    if postact == 0 or Dloc % postact != 0:
        return f"Dloc={Dloc} % postact_tile={postact} != 0 (feature-peer straddle)"
    return None


_UNPORTED_BASELINE = (
    "benchmark/distributed/bench_front_a2a_staged.py is not ported to this tree, so the "
    "LOCAL-projection builders this baseline needs (_compile_front_staged / _front_a2a_plan_dmajor) "
    "do not exist here"
)


def _unported_baseline_supports(ctx):
    """Skip the two LOCAL-projection baselines: their builders live in a module this tree lacks.

    A structured skip rather than an ImportError at build time, so the cell appears in the status
    matrix as `sk` with a reason naming exactly what is missing. A baseline that is absent from the
    matrix reads as a baseline that was never requested; one that is present and skipped reads as
    work still to do. Constant, so it is trivially identical on every rank -- the determinism
    `supports` requires for collective symmetry.

    Args:
        ctx: The driver `Ctx`. Unused -- the gate is a property of this tree, not of the shape.

    Returns:
        The skip reason. Never None, so these targets never reach `build`.
    """
    return _UNPORTED_BASELINE


def _front_comm_supports(ctx):
    """Lighter gate for the COMM-ONLY probe: only the 16-B TMA gate (no GEMM tile floor); M_full=cp·N is
    trivially cp-shardable."""
    ok, reason = shape_check(ctx.N, ctx.cp0, ctx.cp1)
    if not ok:
        return reason
    if int(ctx.Dloc) < 1:
        return f"Dloc={ctx.Dloc}<1"
    return None


# --------------------------------------------------------------------------------------------- configs ---
def _round32(tn):
    """GemmGatedSm90 requires tileN %% 32 == 0 (a CORE kernel constraint, not fused-only). best_front_tile
    can return the FUSED per-peer feature slice tileN=16 at Dloc=8 (D=128/cp16), which the kernel REJECTS;
    the LOCAL baseline GEMM's feature dim is the full 2D (>=32), so round tileN UP to the nearest multiple
    of 32 (min 32) -> valid AND a better local tile. No-op for the fused's Dloc>=16 tiles (already %%32)."""
    return max(32, ((int(tn) + 31) // 32) * 32)


def _front_configs(ctx):
    """The front's ONLY autotune axis: the CTA (tile_M, tile_N) from best_front_tile(Dloc), tile_N rounded
    to a valid multiple of 32 (see _round32). Deterministic on every rank -> the cell records which tile ran."""
    from fold_cp_ops.distributed.dual_gated_gemm_a2a import DualGatedGemmDistSm90

    tile_M, tile_N = DualGatedGemmDistSm90.best_front_tile(ctx.Dloc)
    return [{"tile_M": int(tile_M), "tile_N": _round32(tile_N)}]


def _pick_tile(ctx, Dloc):
    from fold_cp_ops.distributed.dual_gated_gemm_a2a import DualGatedGemmDistSm90

    tm = ctx.cfg.get("tile_M")
    tn = ctx.cfg.get("tile_N")
    if tm is None or tn is None:
        tm, tn = DualGatedGemmDistSm90.best_front_tile(Dloc)
    return int(tm), _round32(tn)


def _front_fused_configs(ctx):
    """FUSED-front autotune axis: the RAW best_front_tile(Dloc) CTA (tile_M, tile_N) — NO _round32. The fused
    per-peer feature-scatter store needs the raw tile_N, which is %16-valid AND satisfies the peer-store
    straddle Dloc % (tile_N//2) == 0 BY CONSTRUCTION (best_front_tile caps tile_N to the widest legal for the
    Dloc). _round32 is a LOCAL-baseline-only fixup (its feature dim is the full 2D >= 32); applying it to the
    fused path bumps tile_N 16->32 at Dloc=8 (D128/cp16) -> postact tile_N=16 -> Dloc(8)%16 != 0 -> the kernel
    ValueErrors at build. Deterministic on every rank -> the cell records the tile that actually ran."""
    from fold_cp_ops.distributed.dual_gated_gemm_a2a import DualGatedGemmDistSm90

    tile_M, tile_N = DualGatedGemmDistSm90.best_front_tile(ctx.Dloc)
    return [{"tile_M": int(tile_M), "tile_N": int(tile_N)}]


def _pick_tile_fused(ctx, Dloc):
    """Raw tile for the FUSED peer store (no _round32; see _front_fused_configs)."""
    from fold_cp_ops.distributed.dual_gated_gemm_a2a import DualGatedGemmDistSm90

    tm = ctx.cfg.get("tile_M")
    tn = ctx.cfg.get("tile_N")
    if tm is None or tn is None:
        tm, tn = DualGatedGemmDistSm90.best_front_tile(Dloc)
    return int(tm), int(tn)


# ----------------------------------------------------------------------------------- build / run / teardown
def _front_operands(ctx, cp, Dloc, D, rpp):
    """Build the front GEMM operands (x, Wg2, Wp2) + the a2a_cfg. rows = rpp (per-rank M_rank)."""
    import torch

    device = ctx.device
    torch.manual_seed(4321 + int(ctx.rank))
    x = torch.randn(rpp, _K, device=device, dtype=torch.bfloat16) / (_K**0.5)
    Wg2 = torch.randn(2 * D, _K, device=device, dtype=torch.bfloat16) / (_K**0.5)
    Wp2 = torch.randn(2 * D, _K, device=device, dtype=torch.bfloat16) / (_K**0.5)
    pe_table = tuple(int(v) for v in ctx.pm.cp_pe_table.tolist())
    a2a_cfg = dict(cp=cp, my_cp_rank=int(ctx.pm.my_cp_rank), rows_per_peer=rpp, pe_table=pe_table)
    return x, Wg2, Wp2, a2a_cfg


def _front_build(ctx):
    """FUSED front, CROSS-NODE: DualGatedGemmDistSm90 with the DIFFERENTIAL IB drain
    (configure_a2a(ib_drain=True) — an NVLink peer keeps the coupled in-epilogue TMA-S2G store; an IB
    (non-P2P) peer drains a symmetric-ring blocking put). This is the ONLY cross-node-capable fused path:
    the plain coupled store is NVLink-only (cannot reach an off-node IB peer), so at cp16 (2,8) [8 IB peers]
    the fused kernel MUST use ib_drain. Built via the anchor's proven builder (mirrors
    bench_front_ib_drain_ablation.py's fused_full). MAY raise (isolated + consensus'd by the driver)."""
    import torch

    from fold_cp_ops.distributed.distributed_manager import DistributedManager

    # LAZY import of the proven ib_drain builder (module-load stays tests-free; only a real GPU cell touches
    # it). The FRONT is K=D-exempt from the K=N perf rule, and this is a compile BUILDER, not a K=256
    # input-builder — not the cross-import the perf-K=N guard forbids.
    from tests.distributed.test_dual_gated_gemm_a2a import _build_front_decoupled

    cp, Dloc, D, M_full, rpp = _front_geom(ctx)
    tile_M, tile_N = _pick_tile_fused(ctx, Dloc)
    x, Wg2, Wp2, a2a_cfg = _front_operands(ctx, cp, Dloc, D, rpp)
    # Symmetric memory via the torch MemPool, NOT nvshmem4py's allocator: torch owns the nvshmem
    # bootstrap here, so nvshmem4py's own `_is_initialized` is permanently False and its `tensor()`
    # raises. The pool also recycles, which keeps the collective free off the destruction path.
    with torch.cuda.use_mem_pool(DistributedManager.symmetric_mempool(ctx.device)):
        recv = torch.empty((2 * Dloc, M_full), dtype=torch.bfloat16, device=ctx.device)
    recv.fill_(-99.0)

    # Wide-put A/B knob (CPO_FRONT_WIDE_W): unset/empty -> 256B drain (default, byte-identical); int W ->
    # ib_wide=True at W token-subtiles/put (run_i auto-engages the coalesce at large N). fused wall(OFF)/
    # fused wall(ON) = the drain gain (comm-bound at large N).
    _wide_w = os.environ.get("CPO_FRONT_WIDE_W")
    _wide_kw = dict(ib_wide=True, ib_wide_batch=int(_wide_w)) if _wide_w else {}
    # B2 pipeline knob (CPO_FRONT_WIDE_NBI): non-blocking per-row wide put + one trailing blocking qp-drain
    # per batch (hides the per-row put latency). Only meaningful WITH ib_wide; default off -> B1 blocking.
    if _wide_w and os.environ.get("CPO_FRONT_WIDE_NBI"):
        _wide_kw["ib_wide_nbi"] = True

    def _cfg_ibdrain(g):
        g.configure_a2a(
            cp=cp,
            my_cp_rank=int(ctx.pm.my_cp_rank),
            rows_per_peer=rpp,
            pe_table=a2a_cfg["pe_table"],
            ib_drain=True,
            ring_depth=2,
            consumer_warpgroups=1,
            **_wide_kw,
        )

    h = {"recv": recv, "compiled": None, "run": None, "ring": None}
    try:
        compiled, run_fn, _gemm, ring = _build_front_decoupled(
            x,
            Wg2,
            Wp2,
            (tile_M, tile_N),
            recv_t=recv,
            ring_depth=2,
            consumer_warpgroups=1,
            configure_fn=_cfg_ibdrain,
        )
        h["compiled"], h["run"], h["ring"] = compiled, run_fn, ring
        return h
    except Exception:
        _front_teardown(h)
        raise


def _front_run(h):
    h["run"]()  # the single fused DualGatedGEMM+A2A call (the TIMED hot call)


def _front_teardown(h):
    """Release the compiled artifact. The symmetric buffers are NOT freed here.

    `recv` and the ring come from `DistributedManager.symmetric_mempool`, which RECYCLES rather than
    frees -- that is what keeps the collective `nvshmem_free` off the object-destruction path, where
    a GC-ordering difference across ranks would deadlock it. Dropping the last reference returns the
    block to the pool; there is no per-tensor free to call, and nvshmem4py's `free_tensor` would be
    the wrong allocator besides."""
    c = h.get("compiled")
    if c is not None:
        try:
            c.free()
        except Exception:
            pass
    # DROP the symmetric buffers, which is what the docstring above says to do and what this function
    # did not do. The pool RECYCLES, so the release IS dropping the last reference -- and a reference
    # held in the handle keeps the block out of the pool just as surely as a live local would.
    # It matters WITHIN a cell, not just across cells: a `cluster` cell builds 12 configs and a front
    # cell several, each allocating recv + ring, so an unreleased handle accumulates symmetric blocks
    # until a later collective allocation faults or hangs in teardown. Same statements on every rank,
    # so this stays symmetric and cannot run a collective on a strict subset.
    for _k in ("recv", "ring", "run"):
        if _k in h:
            h[_k] = None


def _front_local_build(ctx):
    """Compile the LOCAL DualGatedGEMM projection (enable_a2a=False -> byte-identical local D-major (2D, M)
    store, NO comm). Returns h={compiled, run, dual, reshard}. Shared by `front_gemm` (GEMM-only) and
    `front_2kernel` (GEMM + torch reshard)."""
    from benchmark.distributed.bench_front_a2a_staged import _compile_front_staged

    cp, Dloc, D, M_full, rpp = _front_geom(ctx)
    tile_M, tile_N = _pick_tile(ctx, Dloc)
    x, Wg2, Wp2, a2a_cfg = _front_operands(ctx, cp, Dloc, D, rpp)
    h = {"compiled": None, "run": None, "dual": None, "reshard": None}
    try:
        compiled, run_fn, dual_gmem = _compile_front_staged(
            x,
            Wg2,
            Wp2,
            None,
            (tile_M, tile_N),
            eps=_EPS,
            a2a_cfg=a2a_cfg,
            enable_a2a=False,
        )
        h["compiled"], h["run"], h["dual"] = compiled, run_fn, dual_gmem
        return h
    except Exception:
        _front_local_teardown(h)
        raise


def _front_local_teardown(h):
    c = h.get("compiled")
    if c is not None:
        try:
            c.free()
        except Exception:
            pass


def _front_gemm_run(h):
    h["run"]()  # LOCAL DualGatedGEMM projection ONLY (the TIMED t_gemm in isolation, NO comm)


def _front2k_build(ctx):
    """TWO-KERNEL baseline: the LOCAL projection (enable_a2a=False) + the torch all_to_all D-major reshard
    plan. run() executes BOTH legs (GEMM then reshard)."""
    from benchmark.distributed.bench_front_a2a_staged import _front_a2a_plan_dmajor

    cp, Dloc, D, M_full, rpp = _front_geom(ctx)
    h = _front_local_build(ctx)
    try:
        h["reshard"] = _front_a2a_plan_dmajor(
            ctx.pm, cp=cp, M=rpp, D=D, Dloc=Dloc, device=ctx.device
        )
        return h
    except Exception:
        _front_local_teardown(h)
        raise


def _front2k_run(h):
    h["run"]()  # local D-major (2D, M) store (kernel leg)
    h["reshard"](
        h["dual"]
    )  # torch all_to_all feature reshard (comm leg) -> the TIMED 2-kernel pair


def _gemm_cfg_meta(ctx):
    """The local DualGatedGEMM's HEURISTIC config for the Table-B 'autotuned DualGatedGEMM config' column:
    the best_front_tile(Dloc) picker output + the staged parent's fixed knobs (cooperative, no pingpong,
    (1,1,1) cluster — the front's ONLY perf axis is the CTA tile). Pure shape math -> deterministic on every
    rank. Attached to front_gemm + front_2kernel (the t_gemm-producing cells)."""
    cp, Dloc, D, M_full, rpp = _front_geom(ctx)
    tm, tn = _pick_tile(ctx, Dloc)
    return {
        "gemm_cfg": {
            "tile_M": tm,
            "tile_N": tn,
            "cluster": [1, 1, 1],
            "pingpong": False,
            "persistent": True,
            "picker": "best_front_tile",
            "K": _K,
            "M": rpp,
        }
    }


def _a2a_build(ctx):
    """COMM-ONLY probe: symmetric-free plain torch buffers for the front feature reshard. send is (cp,
    2*Dloc*rows_per_peer) -> all_to_all_single scatters the off-diagonal (cp-1)/cp -> the cell's time_ms is
    t_a2a in isolation (BW = comm_bytes / t_a2a)."""
    import torch

    cp, Dloc, _, _, rpp = _front_geom(ctx)
    per_peer = 2 * Dloc * rpp
    send = torch.randn(cp, per_peer, device=ctx.device, dtype=torch.bfloat16)
    recv = torch.empty_like(send)
    return {"send": send, "recv": recv}


def _a2a_run(h):
    import torch.distributed as dist

    dist.all_to_all_single(h["recv"], h["send"])  # the TIMED front feature reshard (isolated t_a2a)


# --------------------------------------------------------------------------------------------- registry ---
def _factory():
    return [
        BenchTarget(
            "front",
            build=_front_build,
            run=_front_run,
            role="target",
            supports=_front_supports,
            teardown=_front_teardown,
            configs=_front_fused_configs,
            comm_bytes=_front_comm_bytes,
        ),
        # Both LOCAL-projection baselines are gated OFF by `supports` until
        # benchmark/distributed/bench_front_a2a_staged.py is ported -- see _unported_baseline_supports.
        # They stay REGISTERED so the status matrix shows them skipped rather than omitting them.
        BenchTarget(
            "front_2kernel",
            build=_front2k_build,
            run=_front2k_run,
            role="baseline",
            supports=_unported_baseline_supports,
            teardown=_front_local_teardown,
            configs=_front_configs,
            comm_bytes=_front_comm_bytes,
            cell_meta=_gemm_cfg_meta,
        ),
        BenchTarget(
            "front_gemm",
            build=_front_local_build,
            run=_front_gemm_run,
            role="baseline",
            supports=_unported_baseline_supports,
            teardown=_front_local_teardown,
            configs=_front_configs,
            cell_meta=_gemm_cfg_meta,
        ),
        BenchTarget(
            "front_a2a",
            build=_a2a_build,
            run=_a2a_run,
            role="baseline",
            supports=_front_comm_supports,
            comm_bytes=_front_comm_bytes,
        ),
    ]


for _nm in ("front", "front_2kernel", "front_gemm", "front_a2a"):
    register(_nm, (lambda nn: lambda: [t for t in _factory() if t.name == nn])(_nm))
register("front_bundle", _factory)
