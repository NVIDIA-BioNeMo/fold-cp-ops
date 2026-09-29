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

"""FRONT route2_ni / transpose_in (INCOMING) A2A adapter — Table B(iii)/B(iv). The transpose_in front
DualGatedGEMM+A2A as BenchTargets, thin plug-ins over the SAME proven builders the D-major `front_a2a.py`
target uses (`_build_front_decoupled` / `_build_front` / `_compile_front_staged`). NO kernel
logic is duplicated. Three names (+ the shared local-GEMM helper):

  * `front_route2`         (role=target)   — the FUSED transpose_in front with the DIFFERENTIAL IB drain
                                             (configure_a2a(transpose_in=True, ib_drain=True) + _a2a_route2_ni).
                                             run() = the single fused call. On an all-P2P (cp{2,4,8}) job the
                                             drain COLLAPSES to the coupled store -> MUST be byte-identical to
                                             `front_route2_coupled`. comm_bytes = V_a2a -> the fused effective
                                             wide-put BW (comm_bytes/t_fused) (refinement 1).
  * `front_route2_coupled` (role=baseline) — the NVLink-only COUPLED transpose_in fused (ib_drain=False), the
                                             "transpose_in-NVLink ref" column on cp{2,4,8}. comm_bytes = V_a2a
                                             (effective NVLink store BW). SKIPS on any job with an IB peer (the
                                             coupled TMA-S2G store cannot reach a non-P2P peer) — a
                                             cp{2,4,8}-only baseline by construction.
  * `front_route2_2kernel` (role=baseline) — the TWO-KERNEL head-to-head: the SAME kernel WITHOUT the fused
                                             store (local (2D,M) store) THEN a torch all_to_all reshard that
                                             lands the SAME N_i-stride-1 (2*Dloc, N_i, N_j) recv. comm_bytes =
                                             V_a2a; cell_meta = gemm_cfg (Table-B column).

M = B·N²/cp (the FAITHFUL strong-scaling pair grid — SAME M as the faithful D-major front_a2a target, so the
two differ ONLY in store LAYOUT: a genuine head-to-head). transpose_in is a 2-D (i_loc, j) token grid by
construction: the front GEMM has M = B·N_i_loc·N_j_loc = (N/cp0)·(N/cp1) = N²/cp rows (the pair projection —
mirrors tests/distributed/test_dual_gated_gemm_a2a._run_ib_drain_front_route2 and fused_trimul.py:278-291), and
the recv is the 3-D N_i-STRIDE-1 (2*Dloc, N_i=cp0·Xg_pad, N_j=cp1·N_j_loc) buffer (vs the D-major flat
(2*Dloc, M_full) recv — the ONLY difference). GEMM-dominated, O(N²) operands -> OOMs the large-N/small-cp
cells, handled by the harness runtime OutOfMemoryError-skip (NEVER a memory gate). FRONT contracts over
K=D -> EXEMPT from the K=N_token HARD RULE.

2-D dst addressing (matches dual_gated_gemm_staged_a2a.py:1389-1390 EXACTLY, validated by the rel=0
fused-vs-2-kernel gate): a flat cp-slot s decomposes cp0_coord = s // cp1, cp1_coord = s % cp1; its block
lands at recv i-base cp0_coord·Xg_pad, j-base cp1_coord·N_j_loc. 1-D (cp1==1): i-base s·Xg_pad, j-base 0,
N_j_loc==N -> the reference-oracle layout.
"""

import os

import benchmark.distributed.back_a2a_store_bench as rp
from benchmark.distributed.harness.registry import register
from benchmark.distributed.harness.target import BenchTarget
from benchmark.distributed.harness.targets.front_a2a import _round32

_DTYPE_BYTES = 2  # bf16
_K = 256  # front contraction (feature/hidden) — the O(N²) projection K, NOT K=N_token
_EPS = 1e-5


def _route2_geom(ctx):
    """(cp, cp0, cp1, Dloc, D, N, B, N_i_loc, N_j_loc, M, Xg_pad, N_i, N_j, tile_M, tile_N) for THIS cell.
    M = B·N_i_loc·N_j_loc = N²/cp (the transpose_in pair grid). Pure shape math -> IDENTICAL on every rank."""
    from fold_cp_ops.distributed.dual_gated_gemm_a2a import DualGatedGemmDistSm90

    cp0, cp1 = int(ctx.cp0), int(ctx.cp1)
    cp = cp0 * cp1
    Dloc = int(ctx.Dloc)
    D = Dloc * cp
    N = int(ctx.N)
    B = int(ctx.B)
    N_i_loc = N // cp0  # local i extent (Xg)
    N_j_loc = N // cp1  # local j extent (Yg); 1-D (cp1==1) -> N
    M = B * N_i_loc * N_j_loc  # pair rows = N²/cp
    tile_M, tile_N = DualGatedGemmDistSm90.best_front_tile(Dloc)
    Xg_pad = ((N_i_loc + tile_M - 1) // tile_M) * tile_M  # BLK_M-padded local-i extent
    N_i = cp0 * Xg_pad  # padded global-i extent (N_i stride-1)
    N_j = cp1 * N_j_loc  # full global-j extent (== N)
    return dict(
        cp=cp,
        cp0=cp0,
        cp1=cp1,
        Dloc=Dloc,
        D=D,
        N=N,
        B=B,
        N_i_loc=N_i_loc,
        N_j_loc=N_j_loc,
        M=M,
        Xg_pad=Xg_pad,
        N_i=N_i,
        N_j=N_j,
        tile_M=int(tile_M),
        tile_N=int(tile_N),
    )


def _route2_comm_bytes(ctx):
    """Per-rank off-diagonal A2A volume = 2·Dloc·M·(cp-1)·dtype (each rank sends 2·Dloc·M elements to EACH of
    (cp-1) remote peers; the diagonal cp-slot stays local). M = B·N_i_loc·N_j_loc — derived from the ACTUAL
    reshard send buffer, NOT a symbolic formula."""
    g = _route2_geom(ctx)
    if g["cp"] <= 1:
        return 0
    return int(2 * g["Dloc"] * g["M"] * (g["cp"] - 1) * _DTYPE_BYTES)


def _has_ib_peer(pe_table):
    """True iff ANY peer is NOT NVLink/P2P-reachable (an IB peer) -> the coupled TMA-S2G store can't reach it.
    Replicates DualGatedGemmDistSm90._build_p2p_table's BUFFER-FREE host query (nvshmem
    team_translate_pe(WORLD, pe, SHARED) < 0 => IB peer) WITHOUT constructing the kernel (no hardcoded
    tile/pingpong)."""
    import nvshmem.core.teams as _nvt

    try:
        from nvshmem.core.nvshmem_types import Team_id as _Tid

        team_world, team_shared = _Tid.TEAM_WORLD, _Tid.TEAM_SHARED
    except Exception:
        from nvshmem.core.nvshmem_types import Teams as _Teams

        _reg = dict(_Teams.items())
        team_world, team_shared = _reg.get("TEAM_WORLD", 0), _reg.get("TEAM_SHARED", 1)
    for pe in pe_table:
        try:
            r = _nvt.team_translate_pe(team_world, int(pe), team_shared)
            if not (r is not None and int(r) >= 0):
                return True  # not SHARED/NVLink => IB peer
        except Exception:
            return True
    return False


# ------------------------------------------------------------------------------------------- supports() ---
#: Every target in this file reaches `_route2_local_build`, which imports
#: `benchmark.distributed.bench_front_a2a_staged._compile_front_staged` -- a module that is NOT in
#: this tree. Without this gate a cell reaches `build` and dies with ImportError, which the driver
#: isolates into `status: error` and the cell exits rc=0 with nothing measured: a whole target family
#: reporting clean while measuring nothing, the same shape as the dead back-A2A bench path (2f972db).
#: `front_a2a.py` already carries the identical constant for the same reason; this file did not.
_UNPORTED_BUILDER = (
    "benchmark/distributed/bench_front_a2a_staged.py is not ported to this tree, so the "
    "LOCAL-projection builder every front_route2 target needs (_compile_front_staged) does not "
    "exist here. Remove this gate when that module lands."
)


def _route2_baseline_supports(ctx):
    """BASELINE gate (front_route2_2kernel = LOCAL (2D,M) store + torch reshard): the 16-B TMA gate ONLY (the
    local store does NOT do the peer feature-scatter, so the Dloc/tile floors are the FUSED store's, applied
    in _route2_supports). M=N²/cp is trivially tile-shardable (auto-clamp handles a partial last M-tile).
    Deterministic on every rank."""
    return _UNPORTED_BUILDER
    ok, reason = rp._shape_check(ctx.N, ctx.cp0, ctx.cp1)
    if not ok:
        return reason
    g = _route2_geom(ctx)
    if g["N_i_loc"] % 8 != 0 or g["N_j_loc"] % 8 != 0:  # 16-B (bf16) on BOTH local token axes
        return f"N_i_loc={g['N_i_loc']} / N_j_loc={g['N_j_loc']} %8 != 0 (16-B TMA floor)"
    return None


def _route2_supports(ctx):
    """FUSED-front gate: the baseline gate PLUS the peer feature-scatter floors (Dloc>=8, best tile
    tile_N%16==0, Dloc%(tile_N//2)==0) the in-epilogue peer store requires (memory
    front_feature_scatter_dloc16_floor; Dloc>=8 post tile_N%16 + a2a x4->x2). Deterministic."""
    return _UNPORTED_BUILDER
    reason = _route2_baseline_supports(ctx)
    if reason is not None:
        return reason
    g = _route2_geom(ctx)
    if g["Dloc"] < 8:
        return f"Dloc={g['Dloc']}<8 (front feature-scatter floor D/cp>=8)"
    if g["tile_N"] % 16 != 0:
        return f"best_front_tile tile_N={g['tile_N']}%16 != 0 (Dloc={g['Dloc']}; STSM-quad floor)"
    postact = g["tile_N"] // 2
    if postact == 0 or g["Dloc"] % postact != 0:
        return f"Dloc={g['Dloc']} % postact_tile={postact} != 0 (feature-peer straddle)"
    return None


def _route2_coupled_supports(ctx):
    """COUPLED (NVLink-only) gate: the fused gate PLUS a has-IB self-skip — the coupled TMA-S2G store returns
    a NULL nvshmem_ptr for a non-P2P peer, so it is NVLink-only. Skips on any job with an IB peer (cp16) ->
    a cp{2,4,8}-only baseline. The probe is the SAME idiom as _run_ib_drain_front_route2 (build cheap, no GPU
    alloc); robust to a probe failure (falls through -> the fused build's is_p2p handles it)."""
    return _UNPORTED_BUILDER
    reason = _route2_supports(ctx)
    if reason is not None:
        return reason
    try:
        pe_table = tuple(int(v) for v in ctx.pm.cp_pe_table.tolist())
        if _has_ib_peer(pe_table):
            return "transpose_in NVLink-only coupled needs all-P2P (job has IB peers -> use front_route2)"
    except Exception:
        pass
    return None


# --------------------------------------------------------------------------------------------- configs ---
def _route2_configs(ctx):
    """The reported cfg column: best_front_tile(Dloc) with tile_N %32-rounded (the LOCAL baseline GEMM's
    feature dim is the full 2D >= 32; matches B(i)/B(ii)'s reported (128,32)). Deterministic."""
    g = _route2_geom(ctx)
    return [{"tile_M": g["tile_M"], "tile_N": _round32(g["tile_N"])}]


def _route2_fused_configs(ctx):
    """FUSED autotune axis: the RAW best_front_tile(Dloc) CTA (NO _round32) — the per-peer feature-scatter
    store needs the raw %16-valid tile_N (and Dloc % (tile_N//2) == 0 by construction). Deterministic."""
    g = _route2_geom(ctx)
    return [{"tile_M": g["tile_M"], "tile_N": g["tile_N"]}]


def _pick_tile(ctx, g):
    tm = ctx.cfg.get("tile_M")
    tn = ctx.cfg.get("tile_N")
    if tm is None or tn is None:
        tm, tn = g["tile_M"], g["tile_N"]
    return int(tm), _round32(tn)


def _pick_tile_fused(ctx, g):
    tm = ctx.cfg.get("tile_M")
    tn = ctx.cfg.get("tile_N")
    if tm is None or tn is None:
        tm, tn = g["tile_M"], g["tile_N"]
    return int(tm), int(tn)


# ----------------------------------------------------------------------------------- build / run / teardown
def _route2_operands(ctx, g):
    """Build the front GEMM operands (x, Wg2, Wp2). rows = M = N²/cp (the pair grid)."""
    import torch

    device = ctx.device
    torch.manual_seed(4321 + int(ctx.rank))
    x = torch.randn(g["M"], _K, device=device, dtype=torch.bfloat16) / (_K**0.5)
    Wg2 = torch.randn(2 * g["D"], _K, device=device, dtype=torch.bfloat16) / (_K**0.5)
    Wp2 = torch.randn(2 * g["D"], _K, device=device, dtype=torch.bfloat16) / (_K**0.5)
    return x, Wg2, Wp2


def _route2_recv(ctx, g):
    """The symmetric N_i-STRIDE-1 recv: contiguous (2*Dloc, N_j, N_i) presented as (2*Dloc, N_i, N_j) (N_i
    inner stride-1). Mirror _run_ib_drain_front_route2."""
    import torch

    # rp._symmetric_empty, NOT nvshmem_torch.tensor. This tree gave symmetric-memory ownership to
    # torch, so nvshmem4py's `_is_initialized` is permanently False and its `tensor()` raises
    # `NvshmemInvalid: NVSHMEM Library is not initialized` -- meaning the whole front_route2 family
    # could not BUILD here at all. Same dead allocator 2f972db removed from cluster_drain.py and
    # back_a2a_store_bench.py; this file was missed then because nothing exercised it.
    recv_buf = rp._symmetric_empty(
        (2 * g["Dloc"], g["N_j"], g["N_i"]),
        torch.bfloat16,
        torch.device("cuda", torch.cuda.current_device()),
    )
    recv_buf.fill_(-99.0)
    return recv_buf, recv_buf.permute(0, 2, 1)


def _route2_cfg_fn(ctx, g, pe_table, *, ib_drain):
    """The configure_a2a closure for the transpose_in store — ib_drain (fused) or coupled (NVLink-only).

    ``pe_table`` is materialized by the CALLER (a ``cp_pe_table.tolist()`` d2h CUDA host-sync) *before* the
    first collective nvshmem malloc (the recv), and passed IN — the closure NEVER recomputes it. This is the
    #B malloc-rendezvous-deadlock fix: the old body did the ``.tolist()`` HERE, and because this cfg_fn is
    evaluated as the ``configure_fn=`` ARGUMENT (i.e. AFTER the recv malloc, BEFORE the ring malloc inside
    ``_build_front_decoupled``), that N-scaled host-sync landed BETWEEN the two collective symmetric-
    mallocs (recv, ring). It smeared the ranks across the two malloc barriers -> the nvshmem_malloc rendezvous
    never converged at large N (the N2048 cp16 DECOUPLED hang; fine at N1024 where the sync is 4x shorter).
    The D-major ``front_a2a._front_operands`` already hoists it identically (line 180, before its recv) —
    which is exactly why the same-builder D-major target does NOT hang."""
    _wide_w = os.environ.get("CPO_FRONT_WIDE_W")
    _wide_kw = dict(ib_wide=True, ib_wide_batch=int(_wide_w)) if _wide_w else {}
    # B2 pipeline knob (CPO_FRONT_WIDE_NBI): non-blocking per-row wide put + one trailing blocking qp-drain
    # per batch — hides the per-row put latency that caps route2_ni's small-run wide put. Off -> B1 blocking.
    if _wide_w and os.environ.get("CPO_FRONT_WIDE_NBI"):
        _wide_kw["ib_wide_nbi"] = True

    def _cfg(gemm):
        ib_kw = (
            dict(ib_drain=True, ring_depth=2, consumer_warpgroups=1, **_wide_kw) if ib_drain else {}
        )
        gemm.configure_a2a(
            cp=g["cp"],
            my_cp_rank=int(ctx.pm.my_cp_rank),
            rows_per_peer=g["M"],
            pe_table=pe_table,
            transpose_in=True,
            token_grid=(g["B"], g["N_i_loc"], g["N_j_loc"]),
            **ib_kw,
        )
        gemm._a2a_route2_ni = True
        gemm._a2a_cp_axis_sizes = (g["cp0"], g["cp1"])

    return _cfg


def _route2_build(ctx):
    """FUSED transpose_in front, CROSS-NODE-capable: DIFFERENTIAL IB drain (an NVLink peer keeps the coupled
    in-epilogue TMA-S2G store; an IB peer drains a symmetric-ring blocking put). On an all-P2P job it
    collapses to the coupled store (byte-identical to front_route2_coupled). MAY raise (isolated + consensus'd
    by the driver)."""
    from tests.distributed.test_dual_gated_gemm_a2a import _build_front_decoupled

    g = _route2_geom(ctx)
    tile_M, tile_N = _pick_tile_fused(ctx, g)
    # #B rendezvous-deadlock fix: materialize pe_table (a cp_pe_table.tolist() d2h host-sync) HERE, BEFORE the
    # FIRST collective nvshmem malloc (the recv). Passing the precomputed tuple into the cfg closure means NO
    # CUDA host-sync lands BETWEEN the two collective symmetric-mallocs (recv, then the ring inside the
    # builder) -> the nvshmem_malloc rendezvous converges at large N (fixes the N2048 cp16 decoupled hang).
    # Mirrors the D-major front_a2a._front_operands idiom (which never hung for exactly this reason).
    pe_table = tuple(int(v) for v in ctx.pm.cp_pe_table.tolist())
    x, Wg2, Wp2 = _route2_operands(ctx, g)
    recv_buf, recv = _route2_recv(
        ctx, g
    )  # COLLECTIVE alloc -- symmetric on all ranks (before the try)
    h = {"recv_buf": recv_buf, "compiled": None, "run": None, "ring": None}
    local_ok, _exc = True, None
    try:
        compiled, run_fn, _gemm, ring = _build_front_decoupled(
            x,
            Wg2,
            Wp2,
            (tile_M, tile_N),
            recv_t=recv,
            ring_depth=2,
            consumer_warpgroups=1,
            configure_fn=_route2_cfg_fn(ctx, g, pe_table, ib_drain=True),
        )
        h["compiled"], h["run"], h["ring"] = compiled, run_fn, ring
    except Exception as e:
        local_ok, _exc = False, e
    _route2_consensus_teardown(
        ctx, h, local_ok, _exc
    )  # #2 fix: consensus BEFORE any collective nvshmem free
    return h


def _route2_coupled_build(ctx):
    """NVLink-only COUPLED transpose_in fused (ib_drain=False) — the transpose_in-NVLink ref column. Uses the
    NON-decoupled builder (no ring). Defensive has-IB raise (supports() already skips on cp16)."""
    from tests.distributed.test_dual_gated_gemm_a2a import _build_front

    g = _route2_geom(ctx)
    pe_table = tuple(int(v) for v in ctx.pm.cp_pe_table.tolist())
    if _has_ib_peer(pe_table):
        raise RuntimeError(
            "front_route2_coupled is NVLink-only; this job has IB peers (use front_route2)"
        )
    tile_M, tile_N = _pick_tile_fused(ctx, g)
    x, Wg2, Wp2 = _route2_operands(ctx, g)
    recv_buf, recv = _route2_recv(
        ctx, g
    )  # COLLECTIVE alloc -- symmetric on all ranks (before the try)
    h = {"recv_buf": recv_buf, "compiled": None, "run": None, "ring": None}
    local_ok, _exc = True, None
    try:
        compiled, run_fn, _gemm = _build_front(
            x,
            Wg2,
            Wp2,
            (tile_M, tile_N),
            recv_t=recv,
            configure_fn=_route2_cfg_fn(
                ctx, g, pe_table, ib_drain=False
            ),  # #B: reuse the pre-recv pe_table
        )
        h["compiled"], h["run"] = compiled, run_fn
    except Exception as e:
        local_ok, _exc = False, e
    _route2_consensus_teardown(
        ctx, h, local_ok, _exc
    )  # #2 fix: consensus BEFORE any collective nvshmem free
    return h


def _route2_run(h):
    h["run"]()  # the single fused DualGatedGEMM+A2A call (the TIMED hot call)


def _route2_teardown(h):
    """Release the compiled kernel and DROP the symmetric tensors this cell holds.

    The tensors are dropped rather than freed through an allocator, and the distinction is the whole
    point: this tree gave symmetric-memory ownership to torch (`DistributedManager.symmetric_mempool`),
    so nvshmem4py's `_is_initialized` is permanently False and its `free_tensor` raises
    `NvshmemInvalid: NVSHMEM Library is not initialized` on EVERY call. This function used to call it
    inside a bare `except Exception: pass`, so the ring and recv_buf were never released and the
    failure was silent -- a teardown that reads as if it frees. Same defect class as the dead
    allocator removed from `cluster_drain.py` and `back_a2a_store_bench.py` in 2f972db, and the
    same shape as the per-run release the bring-back dropped in 6461fe3.

    Dropping the reference is the correct release here: the pool recycles, and keeping a collective
    `nvshmem_free` off the object-destruction path is precisely why the pool owns them (a GC-ordered
    collective free across ranks is what wedged the sweep before).

    Args:
        h: The cell handle. Missing keys are tolerated -- teardown runs after a build that MAY have
            raised partway, so `compiled` or `ring` can legitimately be absent.
    """
    c = h.get("compiled")
    if c is not None:
        try:
            c.free()
        except Exception:
            pass
    for k in ("ring", "recv_buf", "run"):
        if k in h:
            h[k] = None


def _route2_consensus_teardown(ctx, h, local_ok, exc):
    """#2 asymmetric-collective (anti-pattern #1) fix. An IB-peer decoupled build can RAISE on SOME ranks but
    not others (observed cp16=(2,8) N2048). The old per-target `except: _route2_teardown(h); raise` did a
    COLLECTIVE nvshmem free (recv/ring) on the RAISING rank BEFORE the driver's consensus -> it desynced
    against the surviving ranks (heading to the driver's all_reduce / their own ring alloc) -> deadlock; the
    driver's own `if handle is not None: teardown` is likewise asymmetric when some ranks built and some did
    not. Fix: CONSENSUS the failure FIRST (ctx.da.all_reduce_max -- the driver's primitive), so EITHER all
    ranks symmetric-teardown+raise OR all return. The collective nvshmem free then never runs on a strict
    subset of ranks. (recv_buf is allocated by ALL ranks before the try, so the symmetric free is safe; the
    ring -- allocated inside _build_front_decoupled -- is only freed where h['ring'] is set, i.e. on a
    rank that got past the ring alloc, and is None-skipped elsewhere.)"""
    da = getattr(ctx, "da", None)
    if da is not None and getattr(da, "is_distributed", False):
        global_fail = float(da.all_reduce_max(0.0 if local_ok else 1.0))
    else:
        global_fail = 0.0 if local_ok else 1.0
    if global_fail >= 0.5:
        # Free ONLY recv_buf: it is allocated by ALL ranks BEFORE the try, so nvshmem_free(recv_buf) is a
        # SYMMETRIC collective. Do NOT free the ring or the compiled kernel here -- the ring is allocated
        # INSIDE _build_front_decoupled (h['ring'] set only on the SUCCESS return) and c.free()
        # (library_finalize) only exists on a rank that reached library_init; freeing either would run the
        # collective on a STRICT SUBSET of ranks = the very asymmetry we are fixing. A symmetric small leak on
        # a failed cell (rare; the sweep continues) beats an asymmetric-collective hang.
        # DROP the reference; do not call nvshmem4py's free_tensor. It raises here (wrong allocator)
        # and the bare `except` would swallow it, so the buffer was never released and the teardown
        # only LOOKED like it freed. The pool recycles, which also keeps a COLLECTIVE free off the
        # object-destruction path -- the GC-ordered collective free is what wedged the sweep before.
        if h.get("recv_buf") is not None:
            h["recv_buf"] = None
        if not local_ok:
            raise exc
        raise RuntimeError(
            "front_route2: peer rank build failed (collective-symmetric lockstep skip)."
        )


# ------------------------------------------------------------------------- 2-kernel baseline (local + torch)
def _front_a2a_plan_route2ni(pm, g, device):
    """all_to_all_single plan for the transpose_in reshard: local (2D, M) [a|b], M=N_i_loc·N_j_loc -> recv
    (2*Dloc, N_i, N_j) [N_i stride-1]. Same feature-scatter as _front_a2a_plan_dmajor but reassembled into
    the 2-D (i,j) grid: cp-slot s -> i-base (s//cp1)·Xg_pad, j-base (s%cp1)·N_j_loc (matches the fused store's
    cp0_coord/cp1_coord). The rel=0 fused-vs-2-kernel gate validates this on GPU."""
    import torch
    import torch.distributed as dist

    cp, cp1 = g["cp"], g["cp1"]
    Dloc, D = g["Dloc"], g["D"]
    N_i_loc, N_j_loc, N_i, N_j = g["N_i_loc"], g["N_j_loc"], g["N_i"], g["N_j"]
    Xg_pad = g["Xg_pad"]
    per = Dloc * N_i_loc * N_j_loc  # per-peer a (or b) payload
    chunk = 2 * per  # per-peer a+b payload
    pe_table = [int(pm.cp_pe_table[s].item()) for s in range(cp)]

    def reshard(dual_dm):  # dual_dm: (2D, M) D-major [a (D,M) | b (D,M)]
        a = dual_dm[:D].reshape(
            D, N_i_loc, N_j_loc
        )  # (D, i_loc, j_loc) — i_loc outer (token_grid order)
        b = dual_dm[D:].reshape(D, N_i_loc, N_j_loc)
        send_parts = []
        for p in range(cp):  # p = GLOBAL PE (all_to_all_single ordering)
            s = pe_table.index(p)  # cp-slot for global PE p
            sl = slice(s * Dloc, (s + 1) * Dloc)
            send_parts.append(a[sl].reshape(-1))  # my a, peer p's Dloc-slice
            send_parts.append(b[sl].reshape(-1))
        send = torch.cat(send_parts)
        recv_flat = torch.empty_like(send)
        dist.all_to_all_single(recv_flat, send)
        recv_buf = torch.zeros(
            2 * Dloc, N_j, N_i, device=device, dtype=dual_dm.dtype
        )  # N_i-inner backing
        recv = recv_buf.permute(0, 2, 1)  # (2*Dloc, N_i, N_j) N_i stride-1 (pad rows stay 0)
        for p in range(cp):
            s = pe_table.index(p)  # cp-slot of sender p -> its (i,j) grid position
            i0 = (s // cp1) * Xg_pad
            j0 = (s % cp1) * N_j_loc
            blk = recv_flat[p * chunk : (p + 1) * chunk]
            recv[:Dloc, i0 : i0 + N_i_loc, j0 : j0 + N_j_loc] = blk[:per].reshape(
                Dloc, N_i_loc, N_j_loc
            )
            recv[Dloc:, i0 : i0 + N_i_loc, j0 : j0 + N_j_loc] = blk[per:].reshape(
                Dloc, N_i_loc, N_j_loc
            )
        return recv

    return reshard


def _route2_local_build(ctx):
    """Compile the LOCAL DualGatedGEMM projection (enable_a2a=False -> local D-major (2D, M) store, NO comm).
    Shared by front_route2_2kernel (GEMM + reshard)."""
    from benchmark.distributed.bench_front_a2a_staged import _compile_front_staged

    g = _route2_geom(ctx)
    tile_M, tile_N = _pick_tile(ctx, g)
    x, Wg2, Wp2 = _route2_operands(ctx, g)
    pe_table = tuple(int(v) for v in ctx.pm.cp_pe_table.tolist())
    a2a_cfg = dict(
        cp=g["cp"], my_cp_rank=int(ctx.pm.my_cp_rank), rows_per_peer=g["M"], pe_table=pe_table
    )
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
        _route2_local_teardown(h)
        raise


def _route2_local_teardown(h):
    c = h.get("compiled")
    if c is not None:
        try:
            c.free()
        except Exception:
            pass


def _route2_2k_build(ctx):
    """TWO-KERNEL baseline: the LOCAL projection + the transpose_in all_to_all reshard plan."""
    g = _route2_geom(ctx)
    h = _route2_local_build(ctx)
    try:
        h["reshard"] = _front_a2a_plan_route2ni(ctx.pm, g, ctx.device)
        return h
    except Exception:
        _route2_local_teardown(h)
        raise


def _route2_2k_run(h):
    h["run"]()  # local D-major (2D, M) store (kernel leg)
    h["reshard"](
        h["dual"]
    )  # torch all_to_all transpose_in reshard (comm leg) -> the TIMED 2-kernel pair


def _gemm_cfg_meta(ctx):
    """The local DualGatedGEMM's HEURISTIC config for the Table-B(iii)/(iv) 'cfg' column. Pure shape math."""
    g = _route2_geom(ctx)
    tm, tn = _pick_tile(ctx, g)
    return {
        "gemm_cfg": {
            "tile_M": tm,
            "tile_N": tn,
            "cluster": [1, 1, 1],
            "pingpong": False,
            "persistent": True,
            "picker": "best_front_tile",
            "K": _K,
            "M": g["M"],
        }
    }


# --------------------------------------------------------------- COMPOSITE-K variant (§9; the FAST incoming) -
# The composite-K front is CONFIG-ONLY on top of the SAME transpose_in front store as route2_ni — the ONLY
# differences from route2_ni are (a) the recv is a 2-D PER-RANK-CONTIGUOUS D-major (2*Dloc, cp*rpp_padded)
# instead of the N_i-stride-1 3-D, and (b) `_a2a_route2_ni` is NOT set, so the plain D-MAJOR postact store
# runs (coalesces the full per-rank block -> wide 32-KiB puts, not the b_j-pinned 640-B cap). Everything else
# (operands, tile, supports/configs/comm_bytes, run, teardown, consensus) is route2's, reused verbatim.
# MEASURED (front_a2a paired-median, cp16 N5120 D256): route2_ni 83.37ms -> composite 40.03ms = 2.08x.
def _composite_recv(ctx, g):
    """2-D per-rank-CONTIGUOUS D-major recv (2*Dloc, cp*rpp_padded), rpp_padded = B*N_j_loc*Xg_pad — reshapes
    (Dloc, cp, N_j, Xg_pad) = the composite K=(cp,Xg_pad) the back reads. SAME element count as route2_ni's
    3-D (2*Dloc, N_i, N_j), per-rank contiguous instead of N_i-stride-1."""
    import torch

    rpp = g["B"] * g["N_j_loc"] * g["Xg_pad"]
    # See the note at the other allocation site: nvshmem4py's allocator is not this tree's.
    buf = rp._symmetric_empty(
        (2 * g["Dloc"], g["cp"] * rpp),
        torch.bfloat16,
        torch.device("cuda", torch.cuda.current_device()),
    )
    buf.fill_(-99.0)
    return buf, buf  # 2-D; NO permute — the D-major store writes it directly


def _composite_cfg_fn(ctx, g, pe_table):
    """configure_a2a for the COMPOSITE-K front: transpose_in (padded i_loc-inner walk) + the D-MAJOR store
    (do NOT set `_a2a_route2_ni`). Mirrors _route2_cfg_fn's #B pre-recv pe_table hoist + the ib_wide knobs."""
    _wide_w = os.environ.get("CPO_FRONT_WIDE_W")
    _wide_kw = dict(ib_wide=True, ib_wide_batch=int(_wide_w)) if _wide_w else {}
    if _wide_w and os.environ.get("CPO_FRONT_WIDE_NBI"):
        _wide_kw["ib_wide_nbi"] = True

    def _cfg(gemm):
        gemm.configure_a2a(
            cp=g["cp"],
            my_cp_rank=int(ctx.pm.my_cp_rank),
            rows_per_peer=g["M"],
            pe_table=pe_table,
            transpose_in=True,
            token_grid=(g["B"], g["N_i_loc"], g["N_j_loc"]),
            ib_drain=True,
            ring_depth=2,
            consumer_warpgroups=1,
            **_wide_kw,
        )
        # NO gemm._a2a_route2_ni = True -> the plain D-MAJOR store runs (composite-K). NO _a2a_cp_axis_sizes
        # (flat-cp; composite is 1-D + B==1, matching the validated back-read).

    return _cfg


def _composite_build(ctx):
    """COMPOSITE-K fused transpose_in front (the FAST incoming: D-major per-rank-contiguous recv + wide put).
    Same decoupled builder + consensus teardown as _route2_build; only the recv + cfg differ."""
    from tests.distributed.test_dual_gated_gemm_a2a import _build_front_decoupled

    g = _route2_geom(ctx)
    tile_M, tile_N = _pick_tile_fused(ctx, g)
    pe_table = tuple(int(v) for v in ctx.pm.cp_pe_table.tolist())
    x, Wg2, Wp2 = _route2_operands(ctx, g)
    recv_buf, recv = _composite_recv(
        ctx, g
    )  # COLLECTIVE alloc — symmetric on all ranks (before the try)
    h = {"recv_buf": recv_buf, "compiled": None, "run": None, "ring": None}
    local_ok, _exc = True, None
    try:
        compiled, run_fn, _gemm, ring = _build_front_decoupled(
            x,
            Wg2,
            Wp2,
            (tile_M, tile_N),
            recv_t=recv,
            ring_depth=2,
            consumer_warpgroups=1,
            configure_fn=_composite_cfg_fn(ctx, g, pe_table),
        )
        h["compiled"], h["run"], h["ring"] = compiled, run_fn, ring
    except Exception as e:
        local_ok, _exc = False, e
    _route2_consensus_teardown(ctx, h, local_ok, _exc)
    return h


# --------------------------------------------------------------------------------------------- registry ---
def _factory():
    return [
        BenchTarget(
            "front_route2",
            build=_route2_build,
            run=_route2_run,
            role="target",
            supports=_route2_supports,
            teardown=_route2_teardown,
            configs=_route2_fused_configs,
            comm_bytes=_route2_comm_bytes,
        ),
        BenchTarget(
            "front_composite",
            build=_composite_build,
            run=_route2_run,
            role="target",
            supports=_route2_supports,
            teardown=_route2_teardown,
            configs=_route2_fused_configs,
            comm_bytes=_route2_comm_bytes,
        ),
        BenchTarget(
            "front_route2_coupled",
            build=_route2_coupled_build,
            run=_route2_run,
            role="baseline",
            supports=_route2_coupled_supports,
            teardown=_route2_teardown,
            configs=_route2_fused_configs,
            comm_bytes=_route2_comm_bytes,
        ),
        BenchTarget(
            "front_route2_2kernel",
            build=_route2_2k_build,
            run=_route2_2k_run,
            role="baseline",
            supports=_route2_baseline_supports,
            teardown=_route2_local_teardown,
            configs=_route2_configs,
            comm_bytes=_route2_comm_bytes,
            cell_meta=_gemm_cfg_meta,
        ),
    ]


for _nm in ("front_route2", "front_composite", "front_route2_coupled", "front_route2_2kernel"):
    register(_nm, (lambda nn: lambda: [t for t in _factory() if t.name == nn])(_nm))
register("front_route2_bundle", _factory)
