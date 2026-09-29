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
# Copyright (c) 2025, Wentao Guo, Tri Dao.
# Based on the cute-dsl example:
# https://github.com/NVIDIA/cutlass/blob/main/examples/python/CuTeDSL/hopper/dense_gemm.py
# This file has been modified by NVIDIA CORPORATION & AFFILIATES.

"""Front-A2A fused into the STAGED DualGatedGEMM postact store — D-major per-half (design-E front).

The production front-A2A store. (It began as the STAGED analogue of a STAGEC-based front
that was removed once this class superseded it, one day younger, in 2026-06.) It
produces the einsum-native D-MAJOR recv in ONE invocation (BOTH ``a`` and ``b``). It
subclasses the local ``fold_cp_ops/kernels/dual_gated_gemm.py::DualGatedGemmSm90`` (the
local kernel stays UNTOUCHED — design doc §3.1) and overrides ONLY the postact-store seam
so the gated dual output is stored SMEM->PEER-symmetric-GMEM (the CP fork's ``put_signal_nbi_tma_peer``
-style S2G TMA) instead of SMEM->local-GMEM — fusing the FRONT A2A (``a,b`` reshard
``S(0,1,2) -> S(0,3,3)``) into the store the kernel already does.

WHY STAGED (not stagec) + transpose_out=True + dual-width 2D
============================================================
The single-device front contract we reproduce is
``gated_gemm_gate(x_norm, Wg=[Wg_a;Wg_b], Wp=[Wp_a;Wp_b], transpose_out=True, split_out_half=True)``
== the LN-fused dual-gated GEMM with the prologue normalize OFF: the GEMM "N" is ``2D`` (the caller stacks the a/b
projection weights), the glu epilogue halves ``2*2D -> 2D``, and ``transpose_out=True`` makes the
postact **M-major** ((M, 2D) with token stride-1, stored into a (2D, M) buffer). The split
``a = out[:D], b = out[D:]`` is then each a CONTIGUOUS ``(D, M)`` = D-MAJOR view (feature outer, token
stride-1) — exactly what ``_gemm1`` reads (``(D, M)`` -> ``.reshape(D*B, N, N)`` view).

This M-major postact is the KEY enabler of the D-major peer store: the SMEM postact tile's stride-1
leading dim is the TOKEN (M), and the D-major recv ``(2*Dloc, M_full)`` is C-contiguous with the token
(M_full) as its stride-1 dim. So — exactly like the BACK design-E (which kept ``j``=GEMM-N stride-1 in
BOTH the recv and the SMEM box) — the SMEM box and the GMEM box agree on the stride-1 leading dim and
the TMA-S2G atom builds with NO transpose at the store. (The stagec front used a TOKEN-major recv
``(cp*rpp, Dloc)`` where the feature D was stride-1; that is the un-transposed postact's natural layout.
Here we deliberately TRANSPOSE the postact so the recv is D-major / einsum-native, eliding the
token<->D transpose the non-fused front would pay.)

THE FRONT D-MAJOR PER-HALF STORE ADDRESSING
===========================================
Run the front with ``transpose_out=True``, dual width ``2D`` (one invocation emits a|b). The postact is
the M-major ``(M, 2D)`` tile; the CTA's postact N-tile (``tile_coord_mnkl[1]``) is a FEATURE tile. Each
consecutive group of ``n_tiles_per_half = D // tile_n_postact`` feature N-tiles is one half (a then b);
within a half, each group of ``n_tiles_per_Dslice = Dloc // tile_n_postact`` feature N-tiles maps to one
peer's D-slice. The DST in peer ``p``'s D-major recv ``(2*Dloc, M_full)`` [feature row OUTER, token col
stride-1]::

    n_tiles_per_half   = D    // tile_n_postact      # feature N-tiles per half (a or b)
    n_tiles_per_Dslice = Dloc // tile_n_postact      # feature N-tiles per peer's Dloc slice
    N_tile = tile_coord_mnkl[1]
    half   = N_tile // n_tiles_per_half                                       # 0 = a, 1 = b
    peer   = (N_tile %  n_tiles_per_half) // n_tiles_per_Dslice               # per-half Dloc -> peer
    dst_row = half*Dloc + ((N_tile % n_tiles_per_half) % n_tiles_per_Dslice)*n_sub + sub_feat  # feature
    dst_col = my_cp_rank*rows_per_peer + M_tile*m_sub_per_tile + sub_tok                        # token

So ``recv = (2*Dloc, M_full)`` D-major, ``M_full = cp*rows_per_peer = B*N*N``. ``a = recv[:Dloc]``,
``b = recv[Dloc:]``, each ``(Dloc, M_full)`` -> ``.reshape(Dloc*B, N, N)`` is a VIEW feeding the einsum.

THE NO-KERNEL-COPY MECHANISM (mirror of stagec a2a / the back GemmA2ASm90): the kernel is NOT copied.
The ``cp`` peer S2G atoms + peer tensors ride on the ``EpilogueParams`` dataclass (which already crosses
the @cute.jit -> @cute.kernel region carrying the postact TMA atom). The subclass regenerates
``EpilogueParams`` with two extra fields (``postact_peer_atoms``/``postact_peer_tensors``, default None),
attaches them host-side in ``epi_to_underlying_arguments``, and reads them in the in-kernel
``epi_setup_postact``. The splice is just three method overrides — no kernel-arg threading.

REUSES (proven, do NOT re-derive): the T2.0d ``compile_gemm_with_bitcode`` (tvm-ffi OFF +
--link-libraries + library_init), the ASK-2 ``peer_tma_atoms.build_peer_store_atoms`` (cp peer S2G
atoms, gotcha-correct) — here over a peer recv view PERMUTED to (token, feature) so the box modes match
the M-major SMEM box (analogue of the back design-E ``(N_loc,N,...)`` permute) — and the store recipe:
``cute.flat_divide`` (NOT zipped), ``tma_partition(group_modes ...)``, const_expr-unrolled per-peer
select, 4-coord ``cute.copy`` L=1, SMEM Align-128, NO ``cute.printf`` in the epilogue store.
"""

import dataclasses
import os
import typing
from typing import Callable, NamedTuple, Optional

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, const_expr
from cutlass.cute.nvgpu import cpasync

from fold_cp_ops._internal.arch import get_max_active_clusters
from fold_cp_ops._internal.runtime_params import mlir_namedtuple
from fold_cp_ops._internal.rounding import RoundingMode
from fold_cp_ops.kernels.dual_gated_gemm import DualGatedGemmSm90

from fold_cp_ops._internal.epi_composable import _make_epi_params
from fold_cp_ops.distributed.peer_tma_atoms import build_peer_store_atoms

try:
    import nvshmem.core  # noqa: F401 (availability probe)

    # Reuse the back kernel's wide-aligned (align<16>) int16 put wire-ins (the same rebuilt FFI
    # prototypes for nvshmemx_int16_put_{nbi_,}warp; §7.16b) so the front's per-row drain matches the
    # call/prototype alignment exactly — no duplicate FFI, one source of truth.  The NON-nbi
    # ``_put_warp_int16_a16`` is BLOCKING: its IBGDA device path ``ibgda_quiet``-s after posting ->
    # in-kernel completion + backpressure (the cp16 QP-exhaustion fix) — the IB-drain (ib_drain=True)
    # per-window put; on NVLink it is a cheap direct store.
    from fold_cp_ops.distributed.gemm_sm90_a2a import (
        _flush_warp_device,
        _put_nbi_warp_int16_a16,
        _put_warp_int16_a16,
    )

    HAS_NVSHMEM = True
except ImportError:  # pragma: no cover - non-nvshmem host
    HAS_NVSHMEM = False
    _put_nbi_warp_int16_a16 = None
    _flush_warp_device = None
    _put_warp_int16_a16 = None


class DualGatedGemmDistSm90(DualGatedGemmSm90):
    """STAGED DualGatedGEMM whose postact (a|b) store is a FRONT-A2A D-MAJOR peer TMA store
    when ``configure_a2a`` has been called; byte-identical to the local kernel otherwise
    (all A2A paths const_expr(self._a2a_enabled)-gated).

    MUST be run with ``transpose_out=True`` (the postact M-major makes the recv D-major
    einsum-native) and the dual width ``2D`` (one invocation emits both halves). Compile via
    ``fold_cp_ops.distributed.gemm_bitcode_compile.compile_gemm_with_bitcode`` (tvm-ffi OFF +
    --link-libraries + library_init) — the postact store issues an nvshmem device op (the
    peer S2G), which needs the bitcode linked.
    """

    #: Post-construction ``const_expr`` gates -- see :attr:`GemmSm90A2A.COMPILE_GATED_ATTRS` for the
    #: reasoning. 13 of these were unkeyed as of the 2026-08-18 measurement. Pinned against an AST
    #: scan of this class body by ``tests/_internal/compile_time/test_template_params.py``.
    #:
    #: ``_DECOUPLED_BSTATE_FIELDS`` is a class CONSTANT and ``cta_tile_shape_mnk`` is DERIVED, so
    #: neither can vary between two instances of one build; both are listed because the scan finds
    #: them, and keeping the declaration equal to the scan removes the exempt table an omission
    #: would otherwise hide in.
    COMPILE_GATED_ATTRS = (
        "_a2a_decoupled",
        "_a2a_dynamic",
        "_a2a_ring_depth",
        "_a2a_route2_ni",
        "_pad_inner",
        "_pad_inner_x",
        "_pad_inner_y",
        "_token_grid_b",
        "_token_grid_b_dynamic",
        "_token_grid_x",
        "_token_grid_y",
        "_transpose_in",
    )

    # EpilogueArguments: the parent's NamedTuple (fields verbatim) + a ``recv`` field (the
    # symmetric D-major recv buffer (2*Dloc, M_full)).  The caller passes the M-major (M, 2D)
    # postact LOGICAL view as ``mPostAct`` (parent shape/scheduler/STSM machinery unchanged) and
    # the real D-major recv via ``recv``.  Mirrors GemmA2ASm90's ``recv`` extension; default None
    # -> flag-off (parent local store).  Re-declared in full (NamedTuple subclassing can't append
    # fields cleanly); kept in lock-step with DualGatedGemmSm90.EpilogueArguments.
    @mlir_namedtuple
    class EpilogueArguments(NamedTuple):
        mPostAct: cute.Tensor = None
        act_fn: cutlass.Constexpr[Optional[Callable]] = None
        alpha: Optional[Float32 | cute.Tensor] = None
        beta: Optional[Float32 | cute.Tensor] = None
        mRowVecBroadcast: Optional[cute.Tensor] = None
        mColVecBroadcast: Optional[cute.Tensor] = None
        mMaskColVec: Optional[cute.Tensor] = None
        mBiasUp: Optional[cute.Tensor] = None
        mBiasGate: Optional[cute.Tensor] = None
        mPostAct3: Optional[cute.Tensor] = None
        act_fn_3: cutlass.Constexpr[Optional[Callable]] = None
        mRowVecBroadcast3: Optional[cute.Tensor] = None
        rounding_mode: cutlass.Constexpr[int] = RoundingMode.RN
        sr_seed: Optional[Int32 | cute.Tensor] = None
        mWeight: Optional[cute.Tensor] = None
        mBias: Optional[cute.Tensor] = None
        eps: Float32 = Float32(1e-5)
        recv: Optional[cute.Tensor] = None  # symmetric D-major recv (2*Dloc, M_full); None -> local
        # ---- decoupled SIMT GMEM-ring putwarp drain ----
        # The caller's bounded staging ring (grid_CTAs, ring_depth, epi_n_feat, epi_m_token) — TOKEN-
        # innermost so a per-row put is a contiguous token run matching the D-major recv's stride-1
        # (token) axis. None unless configure_a2a(decoupled=True). The MMA warpgroup's epilogue stages
        # each postact subtile here; the consumer warpgroup drains ring->peer recv. On the ib_drain
        # differential path the ring is the PUT SOURCE for IB peers, so it MUST be allocated on the
        # SYMMETRIC HEAP (nvshmem_torch.tensor), SPMD same-size on every rank (an IB put's source must be
        # symmetric-addressable); a LOCAL torch.zeros ring works only for the plain-decoupled NVLink path.
        ring: Optional[cute.Tensor] = None
        # (cp,) int32 device PE table (flat-cp -> global PE) so the consumer routes the per-row
        # put_nbi_warp to peer ``peer`` at RUNTIME -> PE-transparent (NVLink OR IB). None on the
        # coupled path (byte-identical).
        pe_table_dev: Optional[cute.Tensor] = None
        # ROUTE-2 dynamic transpose_in (one-compile-many-N): the RUNTIME 2nd token axis Yg (=N).
        # The rank-2-M remap reads this (a dynamic scalar arg -> rebinds per launch) and derives
        # Xg = M // (B * Yg) (exact; M=mA.shape[0] runtime). None -> static grid (baked) / flag-off.
        # Only consumed HOST-side in _remap_A_operand_layout (never reaches the epilogue params).
        token_grid_yg: Optional[Int32] = None
        # ROUTE-2 dynamic transpose_in, BATCH half (one-compile-many-B): the RUNTIME token-batch
        # extent B. Same seam and the same lifetime as ``token_grid_yg`` -- consumed HOST-side in
        # ``_remap_A_operand_layout`` / ``get_scheduler_arguments`` and never reaching the epilogue
        # params. Required iff the functor was configured ``b_dynamic=True``; ignored otherwise, in
        # which case ``_token_grid_b`` is the baked extent and the flag-off path is byte-identical.
        token_grid_b: Optional[Int32] = None

    # Regenerate EpilogueParams with the 2 peer fields APPENDED to the parent's extra fields
    # (act_fn/act_fn_3).  __init_subclass__ only auto-regens when EpilogueParams is not already
    # in a base's __dict__ (the parent HAS it), so set both explicitly.  The peer fields default
    # None -> the parent's epi_to_underlying_arguments still builds a valid params (flag-off);
    # the override below fills them when A2A is on.  _epi_ops / _epi_param_bases inherited.
    _extra_param_fields = tuple(DualGatedGemmSm90._extra_param_fields) + (
        ("postact_peer_atoms", typing.Any, None),
        ("postact_peer_tensors", typing.Any, None),
        # ---- decoupled SIMT GMEM-ring putwarp drain ----
        ("decoupled_ring", typing.Any, None),  # local-GMEM staging ring (grid,rd,epi_n,epi_m)
        ("ring_store_atom", typing.Any, None),  # producer-TMA S2G atom SMEM-postact-box -> ring
        ("ring_store_tensor", typing.Any, None),  # the box-permuted ring view (epi_m,epi_n,rd,grid)
        ("recv_local", typing.Any, None),  # THIS rank's LOCAL recv perm (token,feature) put dst
        ("pe_table_dev", typing.Any, None),  # (cp,) device PE table (runtime put-by-PE routing)
    )
    EpilogueParams = _make_epi_params(
        DualGatedGemmSm90._epi_ops,
        _extra_param_fields,
        DualGatedGemmSm90._epi_param_bases,
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Master A2A gate (default OFF -> byte-identical to the local kernel).
        self._a2a_enabled = False
        self._a2a_cp = 1
        self._a2a_my_cp_rank = 0
        self._a2a_rows_per_peer = 0
        self._a2a_pe_table = ()
        # Generic 1D/2D DTensor token-sharding (Part A).  The FRONT A2A scatters by FEATURE
        # (D-slice) -> peer-select is by the postact N-tile, INDEPENDENT of how the token axes
        # are sharded; my whole local token block (rows_per_peer rows) lands on every peer at
        # the recv col offset my_cp_rank*rows_per_peer.  So the store math is cp_axis_sizes-
        # AGNOSTIC: rows_per_peer is just my local token count and the recv col offset is the
        # flat my_cp_rank — the 2-D (i,j) decomposition only matters for the host-side unpack,
        # NOT the kernel store.  cp_axis_sizes is recorded for parity with the back kernel.
        self._a2a_cp_axis_sizes = (1,)
        # Dynamic-shape store: read the token-scaling recv COLUMN offset off the recv's RUNTIME
        # M_full instead of baking it (one compile serves many token counts). The feature side
        # (D/Dloc/tile_n/n_tiles_per_*) is FIXED, so only rows_per_peer (token count) goes runtime.
        # Default off => byte-identical static path. CP-geometry (cp/my_cp_rank) stays baked.
        self._a2a_dynamic = False
        # ---- DECOUPLED producer/consumer SIMT GMEM-ring putwarp drain -----------------
        # Mirror of the back GemmSm90A2A decoupled path (docs/gemm_a2a_design.md §7.16j/k). ON =>
        # the postact D-major store becomes a PRODUCER (the MMA warpgroup's epilogue writes each
        # postact subtile to a bounded LOCAL-GMEM ring via a retargeted TMA-S2G + signals a dedicated
        # CONSUMER warpgroup through a two-color mbarrier) and the CONSUMER drains ring->peer recv via
        # a per-row NVSHMEM put_nbi_warp routed by RUNTIME PE (pe_table_dev[peer]) — PE-transparent,
        # hence the IB-swappable precursor (coupled TMA-S2G is NVLink-only). The front's peer is per
        # FEATURE-tile = COMPILE-TIME-uniform per CTA (simpler than the back's per-token-row dynamic
        # peer). Adds ONE consumer warpgroup (_num_extra_warpgroups) + the GMEM ring (bounded,
        # rd=2 rotating, ~MiB). All gated const_expr(self._a2a_decoupled); default OFF => byte-identical
        # to the coupled in-epilogue TMA-S2G store (and the local kernel when A2A off).
        self._a2a_decoupled = False
        # ---- IB-DRAIN candidate 1 (is_p2p DIFFERENTIAL drain; mirror back cluster_multislot + #57) ----
        # ib_drain=True makes the SymMEM drain DIFFERENTIAL per-peer by interconnect type:
        #   * NVLink (P2P) peer -> the EXISTING fast coupled in-epilogue TMA-S2G store (byte-identical).
        #   * IB (non-P2P) peer -> stage the postact tile into a SYMMETRIC-heap ring + a drain warp does
        #     COALESCED-BLOCKING put_warp of a bounded token window to the peer's D-major recv.
        # Composed ON TOP of the decoupled producer/consumer ring machinery (the raw material). The IB
        # branch is mostly orthogonal to the NVLink branch (a per-peer const_expr is_p2p select at the
        # store seam) so NVLink perf is untouched; only IB peers pay the staging path. Default OFF ->
        # byte-identical (every branch const_expr(self._a2a_ib_drain)-gated).
        self._a2a_ib_drain = False
        # #57 CONSOLIDATION: the precise P2P/NVLink connectivity table (nvshmem TEAM_SHARED, buffer-free) +
        # has_ib_peers. Default has_ib_peers=True (conservative) -> the IB machinery stays present until
        # configure_a2a(ib_drain=True) computes is_p2p; an ALL-P2P job (no IB peer, cp<=8) then sets
        # has_ib_peers=False -> the whole IB drain (ring + consumer WG) is const_expr-ELIDED -> the store
        # reduces to the pure coupled NVLink TMA-S2G = byte-identical/fast. is_p2p None until configure.
        self._a2a_is_p2p = None
        self._a2a_has_ib_peers = True
        # Ring depth (staging slots per CTA). rd=2 rotating is the §7.16k production choice (the
        # producer's empty[s] reuse-wait is live; cheap GMEM footprint). The caller allocates the
        # GMEM ring at (grid_CTAs, ring_depth, epi_n_feat, epi_m_token) — NOTE token-innermost so the
        # per-row put is a contiguous token run matching the D-major recv's stride-1 (token) axis.
        self._a2a_ring_depth = 2
        # producer_tma: the MMA warp issues the SAME async TMA-S2G the coupled store uses, retargeted
        # at the GMEM ring slot (O(1) MMA-warp occupancy -> no ab-stage theft, MMA keeps single-device
        # speed), then a cheap local commit+wait before signaling full[s]. Forced ON on the decoupled
        # path (the SIMT producer ring-write would serialize the MMA warp). Mirrors the back default.
        self._a2a_producer_tma = True
        # Number of dedicated CONSUMER warpgroups (4 drainer warps each). The SIMT GMEM->peer drain is
        # parallelized across all of them; scaling adds draining warps. Default 1 (4 warps).
        self._a2a_consumer_warpgroups = 1
        # ---- ROUTE-2 differential-layout: rank-2-M "walk p-inner" A-load (incoming k-major) ---------
        # transpose_in ON => the front WALKS its token-M axis in transposed (X<->Y-swapped) p-inner
        # order via a rank-2 (Xg,Yg) M layout, so the recv comes out einsum-k-major (a_major="k") for
        # the back A2A GEMM WITHOUT the incoming .transpose(-1,-2).contiguous() copies. Default OFF =>
        # _remap_A_operand_layout is identity => the A-load + store are byte-identical to today. It is a
        # COMPILE-CACHE KEY (a distinct traced kernel) — the caller compiles the transpose_in=True variant
        # separately (the bitcode route already compiles incoming vs outgoing separately anyway). The
        # token grid (B, Xg=N_i_loc row-axis, Yg=N_j col-axis; native order (b,X,Y) with Y stride-1) is
        # baked here so the remap can build the composite M layout; None until configure_a2a sets it.
        self._transpose_in = False
        self._token_grid_b = 1
        self._token_grid_x = 0  # Xg = first N token axis (row/r; the transpose's INNER walk axis)
        self._token_grid_y = 0  # Yg = second N token axis (col/c; stride-1 in the native buffer)
        # B-DYNAMIC transpose_in (one-compile-many-BATCH). OFF (default) => ``_token_grid_b`` is a
        # const_expr and B keys the compile, exactly as before -- the whole B==1 tree is then
        # byte-identical. ON => the batch extent arrives per launch on
        # ``EpilogueArguments.token_grid_b`` and the A-operand ALWAYS takes the 5-D
        # (Xg, K, Yg, B, L) arm (an extent-1 mode at B==1) so ONE compiled front serves every batch.
        # It must be a separate flag rather than "B > 1": the whole point is that a functor built
        # while B==1 can later be launched at B==2, which no test of the baked value can express.
        self._token_grid_b_dynamic = False
        # BATCH-PLANE recv LAYOUT (see _a2a_b_mode). Separate from _token_grid_b_dynamic on purpose:
        # a standalone transpose_in front with a BAKED B>1 is perfectly correct writing the old
        # slot-major-then-plane 2-D recv, and does. The plane only has to move OUTSIDE the cp slot
        # for the composite / route2 BACK READ, whose L=(Dloc,B) needs one stride -- so the layout is
        # the CONSUMER's requirement, requested explicitly, not a consequence of B alone.
        self._a2a_b_plane = False
        # ROUTE-2 (A) N_i-STRIDE-1 PRODUCER STORE (user's producer-side no-copy path; requires
        # transpose_in). Default OFF => the peer-block-concat store (byte-identical). ON => the front
        # store DST targets a 3-D recv (2*Dloc, N_i=cp*Xg_pad, N_j=Yg) with N_i STRIDE-1: each rank
        # writes its i-shard GLOBAL-i-positioned (i_loc STRIDE-1 innermost box @ offset my_cp_rank*Xg_pad,
        # j @ stride N_i, feature @ stride N_i*N_j). global-N_i becomes CONTIGUOUS by construction (not
        # peer-block-concat) => the incoming einsum reads a_major="k" NATIVE (no copy, no composite-K, no
        # back-GEMM change). 16-B-safe: box-start my_cp_rank*Xg_pad is BLK_M-aligned (padded shard); the
        # j-stride N_i needs N%8. The transpose_in walk already makes i_loc the SMEM inner (matching the
        # stride-1 GMEM innermost); the pad gap between shards is einsum-safe (glu(0)=0 tail).
        self._a2a_route2_ni = False
        # ---- WIDE-PUT IB drain (docs/layernorm_dual_gated_gemm_gemm_wide_put_plan.md) --------------
        # ib_wide=True COALESCES the IB (non-P2P) drain: instead of one 256 B per-feature-row put per
        # postact subtile, the producer ACCUMULATES W consecutive-dst_tok / same-(peer,dst_feat) subtiles
        # into ONE wide ring slot (token axis grows W×) and the consumer drains it as one (W·epi_m)-token
        # contiguous put per feature row (32 KiB at W=128 -> at/above the IB coalescing knee). Rides ON TOP
        # of ib_drain (changes ONLY the IB arm; the P2P/NVLink coupled store is UNTOUCHED -> perf-neutral).
        # Batching is RASTER-AGNOSTIC via dynamic break-detection (the producer flushes a batch when the
        # next subtile's (peer,feat) differs OR dst_tok is non-consecutive OR W is reached), and count-
        # parity is GUARANTEED-BY-CONSTRUCTION: the consumer drains until subtiles_drained==total_subtiles
        # (summing the per-batch subtile count), so producer-arrive_full == consumer-arrive_empty on ANY
        # raster (§6.4). Default OFF -> every branch const_expr(self._a2a_ib_wide)-gated -> byte-identical
        # to the per-subtile ib_drain. The caller allocates the ring W× wider (grid,rd,epi_n,W·epi_m).
        self._a2a_ib_wide = False
        # W: token-subtiles coalesced per wide put. 128 -> W·epi_m·2B = 128·128·2 = 32768 = 32 KiB EXACTLY
        # (the IB knee). A COMPILE-TIME CONST -> the ring is O(1) in N_token (frugality); the per-put run is
        # RUNTIME (b_w·epi_m) so no nested-if in a range_constexpr W-loop (compile-gate safe).
        self._a2a_ib_wide_batch = 128
        # B2 variant: non-blocking put_nbi + ONE trailing quiet (pipelines the epi_n per-row puts). Default
        # OFF -> B1 blocking put (self-quiet per put, WAR-safe immediately). Benched against B1.
        self._a2a_ib_wide_nbi = False

    # ------------------------------------------------------------------
    # Shape-aware CTA-tile picker (the front's ONLY autotune axis).
    # ------------------------------------------------------------------
    @staticmethod
    def best_front_tile(Dloc):
        """Best CTA ``(tile_M, tile_N)`` for the front tall-skinny dual-GEMM at this ``Dloc``.

        The front GEMM is ``(M_rank, 2D, K)`` with ``M_rank = B·N_token²/cp`` HUGE, GEMM-N = ``2D``,
        ``K=256`` — a tall-skinny shape whose optimum is the CTA tile (the front's ONLY perf axis:
        the cooperative-only staged parent has no pingpong; the A2A path forces ``streaming`` stats →
        no cluster, since ``_do_normalize=False`` and the streaming TMA loads are non-multicast, so the
        back's ``cluster(1,2)`` does NOT port). Determined by a single-device paired-median sweep
        (``debug/front_cfg_sweep.py``, cp 2/4 × D 128/256). At ``N_token=2048`` the stock ``(128,128)``
        runs **412–423 TFLOP/s** while the wider tiles hold **460–533 TFLOP/s** (the small tile's
        wave-quantization tail widens the gap with ``N_token``): the best valid tile beats ``(128,128)``
        by **1.13–1.26×**.

        The front feature-axis peer-split requires ``Dloc % (tile_N//2) == 0`` (a CTA postact N-tile
        must lie within ONE D-slice — so a wider ``tile_N`` never straddles a feature-peer boundary by
        construction), so ``tile_N=256`` (postact 128) needs ``Dloc ≥ 128``. The PER-SHAPE best:
          * ``Dloc ≥ 128`` (D256/cp2, D512/cp≤4) → ``(128, 256)`` — the measured best (468 TF @ D256,
            vs 464 for ``(256,128)`` — the ~1% the picker captures over a flat ``(256,128)``).
          * ``Dloc ∈ [64, 128)`` (D128/cp2, D256/cp4) → ``(256, 128)`` (533/466 TF; ``tile_N=256`` is
            invalid here — ``Dloc < 128``).
          * ``Dloc < 64`` (cp8 / small-D — ``tile_N=128`` ILLEGAL, ``Dloc % 64 != 0``) → ``(128, 2·Dloc)``
            (widest legal ``tile_N``; ``tile_M=256`` lift needs ``tile_N≥128``). A runnability floor,
            not a perf pick — keeps cp8 runnable where ``(256,128)`` cannot exist.
        EVERY branch has ``tile_M ∈ {128, 256}`` (a multiple of 128) → **partial-token-safe**:
        ``rows_per_peer % tile_M ∈ {0 or ≥8}`` follows from the existing ``rows_per_peer % 128`` proof
        (``debug/verify_rpp_unreachable.py``) — no fresh unreachability verification. (``(192,128)`` is
        deliberately NOT used: ``192 ∤ 128`` would need a fresh proof.) The returned tile is always a
        runnable config (caps ``tile_N`` to the largest valid for the ``Dloc``); the
        ``Dloc % (tile_N//2)`` store constraint is also asserted in ``epi_to_underlying_arguments``.
        Parent-untouched, ``partial_token_clamp``-compatible."""
        dloc = int(Dloc)
        if dloc % 128 == 0:  # Dloc>=128: the wider tile_N=256 (postact 128) is the per-shape best.
            return (128, 256)
        if dloc % 64 == 0:  # Dloc in [64,128): tile_N capped at 128 → the taller tile_M=256 wins.
            return (256, 128)
        # Dloc < 64 (cp8 / small-D): tile_N=128 ILLEGAL. Validity floor — widest legal postact-N
        # = largest pow2 dividing Dloc, tile_N = 2*that, tile_M=128.
        postact_n = 32
        while postact_n > 1 and dloc % postact_n != 0:
            postact_n //= 2
        return (128, 2 * postact_n)

    def configure_a2a(
        self,
        cp,
        my_cp_rank,
        rows_per_peer,
        pe_table,
        *,
        dynamic=False,
        decoupled=False,
        ring_depth=None,
        consumer_warpgroups=None,
        partial_token_clamp=False,
        transpose_in=False,
        token_grid=None,
        b_plane=False,
        b_dynamic=False,
        ib_drain=False,
        is_p2p=None,
        ib_wide=False,
        ib_wide_batch=None,
        ib_wide_nbi=False,
        pad_inner=False,
        inner_extent=None,
        token_count=None,
    ):
        """Enable the front-A2A D-major peer TMA store + bake the cp geometry (host-side).

        Mirrors ``GemmSm90A2A.configure_a2a``.
        Call on the instance BEFORE compiling/launching.

        Parameters
        ----------
        cp : int
            Flat cp peer count.
        my_cp_rank : int
            This rank's flat-cp index in ``[0, cp)``.
        rows_per_peer : int
            Token rows of my local block (``B*N_loc*N``) — the postact M extent.  The front
            reshard sends my WHOLE token block to EVERY peer (D-sliced), landing at this peer's
            recv COLUMN range ``[my_cp_rank*rows_per_peer : ...]`` (token = stride-1 col).
        pe_table : tuple of int
            ``cp``-length flat-peer -> global-PE map (``PeMap.cp_pe_table.tolist()``).
        """
        if not HAS_NVSHMEM:
            raise RuntimeError(
                "DualGatedGemmDistSm90.configure_a2a requires nvshmem4py + the vendored nvshmem utils."
            )
        if len(pe_table) != cp:
            raise ValueError(f"pe_table {pe_table} length must equal cp={cp}.")
        if rows_per_peer <= 0:
            raise ValueError(f"rows_per_peer={rows_per_peer} must be positive.")
        self._a2a_enabled = True
        self._a2a_cp = int(cp)
        self._a2a_my_cp_rank = int(my_cp_rank)
        self._a2a_rows_per_peer = int(rows_per_peer)
        self._a2a_dynamic = bool(dynamic)
        self._a2a_pe_table = tuple(int(p) for p in pe_table)
        self._a2a_cp_axis_sizes = (int(cp),)
        # partial-trailing-token-tile CLAMP is now AUTOMATIC (see _a2a_should_clamp): any rows_per_peer
        # that is not a CTA tile_M multiple auto-builds the peer atoms over MY column BLOCK
        # (rows_per_peer, 2*Dloc) view (descriptor token-bound = rows_per_peer) so the partial-last
        # M-tile's overshoot is dropped by the high-end OOB clamp on the stride-1 TOKEN axis (vs the
        # back's outer-axis row clamp); the store dst_tok is RELATIVE to my block. An aligned
        # rows_per_peer keeps the full-M_full absolute-dst atom -> byte-identical. ``partial_token_clamp``
        # is now an EXPLICIT OVERRIDE (force the clamp even when aligned; a no-op if auto already covers
        # it), kept for back-compat; forced OFF on the decoupled/ib_drain path (it reaches the clamp via
        # the AUTO rows_per_peer%tile_M!=0 path instead — the P2P arm reuses the coupled descriptor clamp,
        # the IB arm run-clamps in software (§2/§3), so an off-grid rows_per_peer now runs FUSED, no reject).
        self._a2a_partial_token_clamp = bool(partial_token_clamp and not decoupled)
        self._configure_transpose_in(
            transpose_in, token_grid, rows_per_peer, b_plane=b_plane, b_dynamic=b_dynamic
        )
        self._configure_pad_inner(pad_inner, inner_extent, token_count, rows_per_peer)
        self._configure_decoupled(decoupled, ring_depth, consumer_warpgroups)
        self._configure_ib_drain(ib_drain, is_p2p)
        # §10: the plain-decoupled front store (decoupled=True, ib_drain=False) is RETIRED — a superseded
        # precursor with NO production caller (the harness `front` target + fused_trimul use ib_drain=True;
        # an all-P2P job auto-collapses ib_drain to the coupled store). Its egress arms (the all-nbi
        # producer ring-write + the consumer non-blocking put) were deleted, so reaching the producer/
        # consumer with it would silently drop the store. Reject it at the API boundary (its capability is
        # fully covered by ib_drain=True); decoupled= stays in the signature only paired with ib_drain=True
        # (which forces _a2a_decoupled on).
        if self._a2a_decoupled and not self._a2a_ib_drain:
            raise ValueError(
                "front-A2A plain-decoupled store (decoupled=True, ib_drain=False) is RETIRED (§10) — a "
                "superseded precursor. Pass ib_drain=True instead: an all-P2P/NVLink job auto-collapses to "
                "the coupled TMA-S2G store (byte-identical), a hybrid job gets the differential drain."
            )
        self._configure_ib_wide(ib_wide, ib_wide_batch, ib_wide_nbi)
        # BATCH-MODE recv x DECOUPLED/IB drain: WIRED. The b-mode recv gains a trailing BATCH grid
        # mode, so the coupled TMA coord grows one coordinate; the decoupled/ib drain has no TMA coord,
        # so the plane travels in the ring METADATA instead -- one trailing int32 the producer stamps
        # and the consumer indexes (see _DECOUPLED_META_B_INDEX), plus one bstate slot pinning the
        # pending WIDE batch to a plane so no coalesced put ever spans two of them. This used to
        # RAISE here; the refusal is gone because the record now carries what the store needs.

    def _configure_ib_wide(self, ib_wide, ib_wide_batch, ib_wide_nbi):
        """Bake the WIDE-PUT coalescing knobs (plan §1/§5). ``ib_wide`` rides ON TOP of ib_drain (it
        only changes the IB arm), so it REQUIRES ib_drain. ``ib_wide_batch`` (W, default 128 -> 32 KiB
        at epi_m=128) is the token-subtiles coalesced per wide put — a COMPILE-TIME CONST so the ring
        stays O(1) in N_token (frugality). ``ib_wide_nbi`` selects the B2 non-blocking put variant
        and REQUIRES ``ib_wide`` for the same reason (it is read ONLY by the wide drain loop; alone it
        is silently inert). Default OFF -> byte-identical to the per-subtile ib_drain.

        Args:
            ib_wide: Enable the wide-put coalescing arm. Requires ``ib_drain=True`` AND a PERSISTENT
                grid; both are refused below. Falsy leaves the per-subtile ib_drain untouched.
            ib_wide_batch: W, the token-subtiles coalesced per wide put, or None to keep the current
                value (default 128). Must be a positive int -- a W<1 ring has no slot to accumulate
                into, so the drain would put zero bytes and the peer recv would keep its fill value.
            ib_wide_nbi: Select the B2 non-blocking put pipeline. Must NOT be set without ``ib_wide``.
                Was measured DEFECTIVE and is now FIXED -- see the note below before changing it.

        .. note:: B2 (``ib_wide_nbi=True``) corrupted data until the completion point was added.

            Kept because a race that has been fixed once is the kind that comes back, and the next
            person to touch this arm needs to know which idiom was wrong.

            **The defect.** B2 posted every feature row non-blocking EXCEPT the last one a warp owns,
            and relied on that trailing blocking put's internal ``ibgda_quiet(qp(pe))`` to reap the
            prior non-blocking puts "on the SAME qp" before ``arrive_empty`` handed the ring slot back
            to the producer. It does not. Measured, 16 ranks over IB, ``cp=16 D=256 N=1024``,
            route2_ni, ``CPO_FRONT_WIDE_FENCE=1``: B2 mismatched the 2-kernel reshard reference in
            **6 of 6 runs**, 2 to 9 of 16 ranks each time, 58-239 elements of 67 108 864, worst
            per-element ratio 26.9-43.3, a different element set every run. B1 passed **8 of 8** on
            the same three mesh shapes (flat ``cp16``, ``(2,8)``, ``(4,4)``).

            **The fix.** One ``nvshmemx_flush_warp()`` per batch, after the row loop and before the
            fence and the empty-arrive -- the vendor's exact primitive for this hazard: "wait until
            all source buffers used by preceding non-blocking puts ... are safe to reuse". It assumes
            nothing about which QP a put landed on, and is documented as a no-op on pure P2P, so the
            NVLink arm pays nothing.

            **It REQUIRES cutlass-dsl >= 4.7.0, and that is a real constraint.** NVVM ships with
            cutlass-dsl and is the READER of the nvshmem bitcode. Measured on one box, same nvshmem
            3.7 bitcode, with a minimal kernel whose only body is the call::

                cutlass-dsl 4.4.2 -> LINK_FAIL   (NVVM Compilation Error; the full kernel reports
                                                  NVVM_ERROR_INVALID_IR, "Unknown attribute kind
                                                  (102)", Producer 'LLVM20.0.0git' / Reader 'LLVM 20.0.0')
                cutlass-dsl 4.7.0 -> LINK_OK     (flush_warp AND quiet)

            On an older NVVM the B2 arm therefore fails to COMPILE rather than silently corrupting.
            That is deliberate and is the right direction: loud beats silent, and it is not a
            fallback to B1 -- the configuration is never refused, the correct code is simply emitted
            and an old reader declines it.

            An earlier revision of this note blamed the BITCODE (the sm_90 ``.bc`` is ~19 MB larger
            than sm_80 and carries the GPUNetIO/DOCA path). **That inference was wrong** -- nvshmem
            3.7.2 ships the same ~50 MB sm_90 bitcode -- and it is recorded here because it is the
            plausible-but-false explanation a reader is likely to re-derive from the size alone.

            Four things worth keeping, because each cost real time to establish:

            * **A fence is not a completion.** ``fence_acq_rel_sys`` (the #9 guard) ORDERS the NIC's
              read against the producer's write; it never waits for that read to finish. It was ON in
              every run that corrupted. Ordering was necessary and present; completion was missing.
            * **A POOLED statistic cannot see this.** The same run printed ``rel_L2=3.663e-05,
              outlier_rows=0`` while the element-wise comparison reported 121/67108864 at worst ratio
              43.3. Do not re-validate this path with an L2.
            * **The first symptom is a HANG, not a failure.** A rank-divergent numeric outcome makes
              pytest retain the failing ranks' frames, so their symmetric tensors stay referenced, so
              the NEXT cell's allocation in ``DistributedManager.symmetric_mempool`` needs a new
              segment -- collective over every PE -- on exactly those ranks and recycles on the
              others. Measured: 6 ranks parked in the allocator, 10 in the barrier after it.
            * **A blocking put does not complete the nbi puts before it.** That was the old arm's
              entire correctness argument and it is the assumption that failed. The trailing blocking
              put is still there -- it bounds the QP for its own row -- but nothing rests on it now.
            * **Importability is not callability.** Both primitives import with well-formed
              zero-argument prototypes under BOTH toolchains; only a compile tells them apart. A
              Python-level `dir()` probe reported success on the toolchain where the call cannot be
              emitted at all.

            The shipped ladder never ran B2: ``ib_wide_nbi`` appears nowhere under
            ``fold_cp_ops/distributed/workflows/``, ``TriMulAutotuned.__init__`` has no such
            parameter, and ``_ib_kw`` passes only ``ib_drain``/``ib_wide``/``ib_wide_batch``/
            ``consumer_warpgroups``/``ring_depth``. Its published numbers are unaffected either way.

        Raises:
            ValueError: ``ib_wide_batch < 1``; ``ib_wide_nbi`` without ``ib_wide``; ``ib_wide``
                without ``ib_drain``; ``ib_wide`` on a non-persistent grid.
        """
        self._a2a_ib_wide = bool(ib_wide)
        if ib_wide_batch is not None:
            if int(ib_wide_batch) < 1:
                raise ValueError(f"ib_wide_batch (W) = {ib_wide_batch} must be a positive int.")
            self._a2a_ib_wide_batch = int(ib_wide_batch)
        # ib_wide_nbi WITHOUT ib_wide is SILENTLY INERT, so refuse it here rather than record it.
        # `_a2a_ib_wide_nbi` is read at exactly ONE site -- inside `_a2a_wide_front_drain_loop`, which is
        # reachable only under `const_expr(self._a2a_ib_wide)`. Set it alone and the flag is stored, never
        # read, and the caller silently gets the B1 blocking drain it believed it had opted out of. Note
        # the assignment now sits AFTER this guard: it used to precede the `if not ib_wide: return` below,
        # which is precisely how a dead flag came to be recorded on the instance. Its sibling below raises
        # for the analogous unmet dependency (ib_wide without ib_drain); the asymmetry was the bug, not a
        # deliberate looseness. (NOT the claim of the deleted `test_front_ib_wide_nbi_not_wired_raises` --
        # B2 IS wired; what is refused here is nbi WITHOUT the wide drain that is its only reader.)
        #
        # DO NOT resolve this by wiring nbi into the NARROW (per-subtile) drain instead. That arm's
        # non-blocking put was deliberately RETIRED (§10): the BLOCKING `_put_warp_int16_a16` IBGDA-quiets
        # after posting, which gives in-kernel completion + backpressure, and THAT is the cp16
        # QP-exhaustion fix. Re-introducing a narrow nbi arm re-opens a closed bug.
        # FRONT-DOOR REFUSAL — B2 is refused OUTRIGHT, on every toolchain, and the reason is PERF, not
        # linking. Set `_CPO_ALLOW_IB_WIDE_NBI=1` in the environment only to develop B2 itself.
        #
        # The flag is UNREACHABLE from the shipped API: no caller under `workflows/` or
        # `fused_trimul*.py` passes it, and it defaults False here and at `__init__`. So this refusal
        # costs the e2e workflow, the benchmarks and the README tutorial NOTHING -- none of them has
        # ever emitted `nvshmemx_flush_warp`, which is why none of them has ever hit the link error.
        # That is the property being PRESERVED here, not one being introduced.
        #
        # Why refuse rather than let it through on a toolchain that links it: measured 2026-08-27,
        # cutlass-dsl 4.7.0 runs the front dual-gated GEMM A2A **3.23-3.25x slower** than 4.4.2 --
        # 96% of a 2.53x e2e regression, 99.8% of it DEVICE time, one kernel, with the back A2A GEMM
        # in the same trace under the same compiler at 0.96-0.99. Cause UNKNOWN: the emitted code is
        # essentially unchanged (REG 74->74, STACK 264->264, census +0.6-3.2%), so it is a scheduling
        # or latency-hiding effect, not a code-volume one. Handing a user a silent 3x is worse than
        # refusing, and refusing on BOTH toolchains keeps the two failure modes from trading places.
        if ib_wide_nbi and os.environ.get("_CPO_ALLOW_IB_WIDE_NBI") != "1":
            raise ValueError(
                "front-A2A ib_wide_nbi=True (the B2 non-blocking-put drain) is REFUSED at the front "
                "door on every toolchain, and it is NOT reachable from the shipped API — nothing "
                "under workflows/ or fused_trimul*.py passes it, so the e2e workflow, the benchmarks "
                "and the README tutorial are unaffected.\n"
                "  * cutlass-dsl < 4.7.0: B2 emits `nvshmemx_flush_warp`, which that NVVM CANNOT LINK "
                "(NVVM Compilation Error, 'Unknown attribute kind (102)'). This refusal exists so you "
                "get this sentence instead of an ICE several frames into codegen.\n"
                "  * cutlass-dsl >= 4.7.0: it compiles and links, but the A2A-fused DualGatedGEMM "
                "kernel then runs 3.23-3.25x SLOWER (measured 2026-08-27, D=256 N_token=3008 cp=4x4 "
                "incoming; 96% of a 2.53x end-to-end regression, 99.8% of it device time, while the "
                "BACK A2A GEMM in the same trace under the same compiler is unchanged at 0.96-0.99). "
                "REASON UNKNOWN — the emitted code is essentially identical across the two "
                "toolchains, so it is a scheduling/latency-hiding effect, not a codegen-volume one.\n"
                "Drop ib_wide_nbi; the shipped default is the B1 blocking per-row drain. To develop "
                "B2 itself, set _CPO_ALLOW_IB_WIDE_NBI=1 and accept both hazards above."
            )
        if ib_wide_nbi and not ib_wide:
            raise ValueError(
                "front-A2A ib_wide_nbi=True REQUIRES ib_wide=True — the B2 non-blocking put variant is "
                "read ONLY by the wide-put drain loop, so with ib_wide=False it is SILENTLY INERT: the "
                "flag would be stored, never read, and you would get the B1 blocking per-row drain. Pass "
                "ib_wide=True, or drop ib_wide_nbi. It is NOT wired into the narrow per-subtile drain and "
                "must not be — that arm's non-blocking put was RETIRED (§10) because the blocking put's "
                "in-kernel IBGDA quiet is the cp16 QP-exhaustion fix."
            )
        self._a2a_ib_wide_nbi = bool(ib_wide_nbi)
        if not ib_wide:
            return
        # ib_wide_nbi (B2) IS wired (drain loop): EVERY per-feature-row put goes NON-BLOCKING and the batch
        # is completed by ONE nvshmem_quiet() before the fence + empty-arrive. This hides the per-row put
        # LATENCY that dominates a SMALL-run drain (route2_ni's N_i-stride-1 recv caps the per-put run at the
        # i-shard span -> ~1 KiB puts deep in the IB latency knee); the D-major 32 KiB run already amortizes
        # the latency, so B2 is ~neutral there. Default OFF -> B1 per-row blocking.
        #
        # The sentence that USED to be here was wrong and had corrupted data: that the trailing blocking put's
        # ibgda_quiet(qp(pe)) "reaps ALL the prior nbi puts on the SAME (pe) qp". It does not -- B2 mismatched
        # in 6 of 6 measured runs where B1 passed 8 of 8. Completion now comes from one nvshmemx_flush_warp()
        # Completion now comes from ONE nvshmemx_flush_warp() per batch (gemm_sm90_a2a._flush_warp_device),
        # which needs cutlass-dsl >= 4.7.0 to link. The same old comment also said no device-scope quiet was
        # needed because it "IS un-callable on this nvshmem-FFI env" -- right outcome on the OLD toolchain,
        # wrong reason: the symbol IS bound, it just failed to LINK under NVVM 4.4.2 and links under 4.7.0.
        # The `.. note::` above carries the measurement, the version A/B and one refuted explanation.
        if not self._a2a_ib_drain:
            raise ValueError(
                "front-A2A ib_wide=True REQUIRES ib_drain=True — the wide-put coalesces ONLY the IB "
                "(non-P2P) arm; it rides on the differential ib_drain machinery. Pass ib_drain=True."
            )
        # Frugality HARD RULE: the symmetric ring is grid_CTAs·rd·epi_n·W·epi_m·2B. That is O(1) in
        # N_token ONLY if grid_CTAs is the SM-count CONST (a PERSISTENT grid). On a NON-persistent grid
        # (grid_CTAs ∝ tiles ∝ N²) the ring would be O(N²) = FORBIDDEN. Assert persistent here so the
        # frugality never silently regresses.
        if not bool(getattr(self, "is_persistent", False)):
            raise ValueError(
                "front-A2A ib_wide=True requires a PERSISTENT grid (is_persistent=True): the wide "
                "symmetric ring is O(grid_CTAs·rd·epi_n·W·epi_m), which is O(1) in N_token ONLY when "
                "grid_CTAs is the SM-count const. A non-persistent grid_CTAs ∝ N² -> O(N²) ring "
                "(frugality violation). The front A2A kernel is persistent; do not disable it here."
            )

    def _configure_ib_drain(self, ib_drain, is_p2p=None):
        """Bake the IB-drain (is_p2p DIFFERENTIAL) knob + the P2P connectivity table.

        ``ib_drain`` makes the SymMEM drain differential per-peer by interconnect type (NVLink peer ->
        coupled TMA-S2G, IB peer -> symmetric-ring + blocking put_warp). It COMPOSES with the decoupled
        producer/consumer ring machinery (the raw material), so it forces ``_a2a_decoupled`` on. OFF =>
        byte-identical (the coupled store; is_p2p left None, has_ib_peers True but unused). Builds the
        #57 is_p2p table (nvshmem TEAM_SHARED translate, buffer-free, post-init) + has_ib_peers so the
        all-P2P collapse can const_expr-elide the whole IB path.

        Args:
            ib_drain: Whether to enable the differential drain. Falsy returns immediately, leaving
                ``is_p2p`` None and ``has_ib_peers`` True-but-unused.
            is_p2p: OVERRIDE for the connectivity table, or None to PROBE nvshmem. Non-None is for
                COMPILING a config whose fabric is absent -- the table is a const_expr, so the cubin
                depends on its contents. Must be one bool per cp slot in `pe_table` order;
                `_validated_p2p_table` refuses a length mismatch or more than 8 P2P peers, both of
                which the probe could not have produced. Build it with `synth_p2p_table` rather than
                by hand: a hand-written table can describe a machine that does not exist, and a
                cubin compiled from one proves nothing when compared.

        Raises:
            ValueError: From `_validated_p2p_table`, on a malformed override.
        """
        self._a2a_ib_drain = bool(ib_drain)
        if not ib_drain:
            self._refuse_coupled_cross_node(is_p2p)
            return
        # ib_drain rides the decoupled producer/consumer ring machinery.
        self._a2a_decoupled = True
        # `is_p2p=` OVERRIDES the nvshmem probe. It exists so a fabric-dependent config can be
        # COMPILED where that fabric is absent: the table is a const_expr baked into the kernel, so
        # the cubin depends on its CONTENTS, and byte identity for an IB-carrying config is
        # otherwise unreachable without a 2-node allocation for every compile.
        #
        # THE HAZARD, stated because the check for it cannot live here: a table that could not
        # arise on any real job produces a cubin that could not arise either, and comparing two
        # such cubins proves nothing. `synth_p2p_table` derives the tuple a real job WOULD probe,
        # and the cross-check against `_build_p2p_table` on a live mesh is what makes the synthesis
        # falsifiable. Passing a hand-written tuple bypasses both and is the caller's own risk.
        #
        # NOT plumbed through `configure_a2a_dtensor`, deliberately: that entry is the production
        # path and has a live fabric by construction, so an override there would only ever be a way
        # to compile a lie.
        self._a2a_is_p2p = (
            self._build_p2p_table(self._a2a_pe_table)
            if is_p2p is None
            else self._validated_p2p_table(is_p2p, self._a2a_pe_table)
        )
        self._a2a_has_ib_peers = not all(self._a2a_is_p2p)

    def _refuse_coupled_cross_node(self, is_p2p=None):
        """Refuse a COUPLED (no-ib_drain) A2A store on a job that has an InfiniBand peer.

        Purpose
            The coupled store TMA-S2Gs straight into a peer's symmetric heap, which requires the peer
            to be P2P/NVLink-reachable -- ``nvshmem_ptr`` returns NULL for an IB peer. Issuing it
            cross-node forms an unmapped address and the fault surfaces LATER, at nvshmem's
            ``barrier.cu:55``, as ``cuda failed with an illegal memory access`` plus ``exit 255`` on
            every rank with no Python traceback. This turns that into a front-door ``ValueError``.

        Why HERE and not one layer up
            ``trimul_autotuned`` already refuses the same combination, but every direct
            ``configure_a2a()`` caller bypassed it -- and the differential NVLink/IB mechanism lives
            entirely inside the ``ib_drain=True`` branch, which returns before probing when
            ``ib_drain`` is false. So the coupled path is not a graceful degradation of the
            differential one; it is an unconditional TMA-S2G to every peer, with the probe that would
            have caught this sitting in the branch that was not taken.

        WHY A FAILED PROBE MUST NOT RAISE, which is the whole subtlety
            :meth:`_build_p2p_table` appends ``False`` on ANY exception, so a job where nvshmem is
            not up yet produces an all-False table -- indistinguishable, by value, from a job where
            every peer really is IB. Raising on that would refuse a perfectly good single-node
            configure. The discriminator is THIS RANK'S OWN SLOT: a PE is always
            P2P-reachable from itself, so ``is_p2p[my_cp_rank]`` False means the probe did not work
            rather than that self is remote. In that case we return silently and leave the behaviour
            exactly as it was before this check existed.

        Args:
            is_p2p: The caller's connectivity OVERRIDE, or None to probe. Honored when given, so the
                documented compile-where-the-fabric-is-absent path (a config compiled for a fabric
                this machine does not have) is unaffected -- it passes its own table and gets checked
                against that rather than against the local machine.

        Returns:
            None, both when every peer is P2P-reachable and when the probe could not be trusted.

        Raises:
            ValueError: When the table is trustworthy and names at least one non-P2P peer.
        """
        pe_table = self._a2a_pe_table
        if not pe_table:
            return
        try:
            table = (
                self._build_p2p_table(pe_table)
                if is_p2p is None
                else self._validated_p2p_table(is_p2p, pe_table)
            )
        except Exception:  # noqa: BLE001 - an unusable probe must not refuse a valid config
            return
        my = self._a2a_my_cp_rank
        if not (0 <= my < len(table)) or not table[my]:
            return  # self is always P2P-reachable; a False here means the probe, not the fabric
        if all(table):
            return
        n_ib = sum(1 for v in table if not v)
        raise ValueError(
            f"This job has {n_ib} cross-node IB peer(s) (not all cp peers are P2P/NVLink-reachable, "
            f"is_p2p={tuple(table)}, pe_table={tuple(pe_table)}), but this configure_a2a() selects "
            "the COUPLED NVLink-only A2A store, which faults with a CUDA illegal address on a "
            "TMA-S2G to an IB peer -- reported later at nvshmem's barrier.cu:55 as an illegal memory "
            "access, with exit 255 and no traceback. Pass ib_drain=True for the differential drain "
            "(NVLink peers keep the coupled TMA-S2G; IB peers go through the symmetric ring), which "
            "collapses back to the coupled store byte-identically on an all-P2P job."
        )

    @staticmethod
    def synth_p2p_table(pe_table, my_pe, node_sz):
        """The is_p2p table a job with `node_sz` GPUs per NVLink domain WOULD probe, by arithmetic.

        Purpose
            Let a fabric-dependent config be COMPILED where that fabric is absent. ``is_p2p`` is a
            const_expr baked into the kernel, so the cubin depends on its CONTENTS; without this, a
            byte-identity check on any IB-carrying config needs a 2-node allocation per compile.

        Semantics
            PEs are numbered globally and NVSHMEM's ``TEAM_SHARED`` is the NVLink-local set, so PE
            ``p`` sits in domain ``p // node_sz`` and is P2P-reachable from ``my_pe`` exactly when
            the two domains match. `_build_p2p_table` obtains the same predicate by asking
            ``team_translate_pe``; this COMPUTES it, and is therefore a MODEL of the fabric rather
            than an observation of one. That difference is the whole risk, and it is why the
            cross-check test against the probe exists.

        Args:
            pe_table: Flat cp-slot -> GLOBAL PE mapping, as `configure_a2a` receives it. A cp-LOCAL
                table would put every peer in domain 0 and synthesize an all-P2P job, which
                const_expr-elides the entire IB path -- a silently wrong cubin, not an error.
            my_pe: This rank's GLOBAL PE, from the same numbering as `pe_table`.
            node_sz: GPUs per NVLink domain (8 on an H100 NVSwitch node). Must be positive, and must
                match the ALLOCATION: a wrong value moves the node boundary and the resulting table
                is internally consistent, so nothing downstream can notice.

        Returns:
            ``tuple[bool]``, one entry per cp slot, parallel to `pe_table`.

        Raises:
            ValueError: If `node_sz` is not positive, or if the P2P set exceeds 8 -- the H100
                NVSwitch bound `_build_p2p_table` also asserts. A table breaking it describes no
                real machine.
        """
        if int(node_sz) < 1:
            raise ValueError(f"node_sz={node_sz} must be a positive int.")
        dom = int(my_pe) // int(node_sz)
        out = tuple(int(pe) // int(node_sz) == dom for pe in pe_table)
        if sum(out) > 8:
            raise ValueError(
                f"synth_p2p_table: |P2P| = {sum(out)} > 8 (the H100 NVSwitch domain is <= 8). "
                f"node_sz={node_sz} does not describe a real machine for pe_table={tuple(pe_table)}."
            )
        return out

    @staticmethod
    def _validated_p2p_table(is_p2p, pe_table):
        """Check a caller-supplied ``is_p2p`` is shaped like something the probe could return.

        Args:
            is_p2p: The override. One bool per cp slot, same length and ORDER as `pe_table` -- a
                short tuple runs off the end of the store's peer loop and a long one silently
                ignores entries.
            pe_table: The table it must line up with.

        Returns:
            ``tuple[bool]``.

        Raises:
            ValueError: On a length mismatch, or on more than 8 P2P peers. The probe could not have
                returned either, so neither may an override.
        """
        out = tuple(bool(v) for v in is_p2p)
        if len(out) != len(tuple(pe_table)):
            raise ValueError(
                f"is_p2p has {len(out)} entries but pe_table has {len(tuple(pe_table))}; they index "
                f"the same cp slots, so a mismatch reads the wrong peer or runs off the end."
            )
        if sum(out) > 8:
            raise ValueError(
                f"is_p2p declares {sum(out)} P2P peers > 8 (the H100 NVSwitch domain is <= 8); the "
                f"probe could not have returned this, so neither may an override."
            )
        return out

    def _build_p2p_table(self, pe_table):
        """#57 P2P/NVLink connectivity table (mirror of GemmSm90A2A._build_p2p_table). is_p2p[flat_cp
        slot] = True iff peer pe_table[slot] is P2P/NVLink-reachable == TMA-S2G-able. Uses nvshmem's
        NVSHMEM_TEAM_SHARED (the NVLink-local PE set) via team_translate_pe(WORLD, pe, SHARED) != -1 --
        BUFFER-FREE, host-side, post-init. P2P is a STATIC job property -> a const_expr tuple baked once.
        Asserts |P2P| <= 8 (H100 NVSwitch domain). Returns tuple[bool]."""
        import nvshmem.core.teams as _nvt

        try:
            from nvshmem.core.nvshmem_types import Team_id as _Tid

            team_world, team_shared = _Tid.TEAM_WORLD, _Tid.TEAM_SHARED
        except Exception:
            from nvshmem.core.nvshmem_types import Teams as _Teams

            _reg = dict(_Teams.items())
            team_world, team_shared = _reg.get("TEAM_WORLD", 0), _reg.get("TEAM_SHARED", 1)
        is_p2p = []
        for pe in pe_table:
            try:
                r = _nvt.team_translate_pe(team_world, int(pe), team_shared)
                is_p2p.append(
                    bool(r is not None and int(r) >= 0)
                )  # -1/raise => not SHARED => IB peer
            except Exception:
                is_p2p.append(False)
        n_p2p = sum(is_p2p)
        assert n_p2p <= 8, (
            f"IB-drain: |P2P/NVLink peers| = {n_p2p} > 8 (H100 NVSwitch domain is <=8). "
            f"is_p2p={is_p2p}, pe_table={tuple(pe_table)}."
        )
        return tuple(is_p2p)

    def _ib_collapse(self) -> bool:
        """#57 all-P2P COLLAPSE: ib_drain requested but NO IB peer (a node-local cp<=8 job, every peer
        NVLink-reachable). The whole IB drain (symmetric ring + consumer WG + blocking put) is then
        USELESS -> const_expr-ELIDED so the store reduces to the pure coupled NVLink TMA-S2G = BYTE-
        IDENTICAL to today's coupled path (the load-bearing "NVLink arm unchanged" guarantee). Pure fn of
        the baked ib_drain/has_ib_peers state (host + kernel agree). getattr-safe (parent __init__ calls
        the warpgroup-count hook before configure sets the flags)."""
        return bool(
            getattr(self, "_a2a_ib_drain", False) and not getattr(self, "_a2a_has_ib_peers", True)
        )

    def _decoupled_active(self) -> bool:
        """Is the decoupled producer/consumer ring machinery LIVE (extra WG + ring SMEM + drain)? True
        for the PLAIN decoupled path (ib_drain off) AND the ib_drain DIFFERENTIAL path WITH >=1 IB peer.
        False when decoupled is off OR at the all-P2P collapse (-> coupled store, byte-identical). Every
        decoupled gate (extra-WG / SMEM struct / reg-adjust / store seam / consumer role) keys off THIS so
        the collapse elides the whole path uniformly."""
        return bool(getattr(self, "_a2a_decoupled", False)) and not self._ib_collapse()

    def _configure_transpose_in(
        self, transpose_in, token_grid, rows_per_peer, b_plane=False, b_dynamic=False
    ):
        """Bake the route-2 rank-2-M walk-p-inner knob + the token grid (shared by configure_a2a /
        _sharded; also directly settable in a LOCAL A2A-off unit test).

        ``transpose_in`` makes the front WALK its token-M axis in transposed (X<->Y-swapped) p-inner
        order (recv comes out a_major="k" for the back). ``token_grid`` = ``(B, Xg, Yg)`` the LOCAL
        block's token grid: native M rows in ``(b, X, Y)`` order with ``Y`` stride-1 (Xg = the first
        N/row axis = the transpose's INNER walk axis; Yg = the second N/col axis). Must satisfy
        ``B*Xg*Yg == rows_per_peer``. OFF/None => byte-identical (identity remap). It is a
        compile-cache key (distinct traced kernel).

        ``b_dynamic`` makes the BATCH extent a RUNTIME value instead of a compile constant: the
        A-operand always takes the 5-D (Xg, K, Yg, B, L) arm and reads B off
        ``EpilogueArguments.token_grid_b`` per launch, so ONE compiled front serves every batch.
        ``token_grid[0]`` is then only the ANCHOR extent (it still has to satisfy the product check
        against the anchor ``rows_per_peer``, and it still sizes the anchor's padded per-peer count);
        every later launch supplies its own. Default False => B keys the compile, byte-identical.
        Requires ``dynamic=True`` at the same ``configure_a2a`` call -- the static walk bakes
        (Xg, Yg) as const_expr, so a runtime B could not be combined with them; passing
        ``b_dynamic=True`` without ``dynamic=True`` raises rather than silently baking B."""
        self._transpose_in = bool(transpose_in)
        self._token_grid_b_dynamic = bool(transpose_in and b_dynamic)
        # LAYOUT flag, ORTHOGONAL to the runtime-extent one and implied by it. b_dynamic forces it
        # because a functor that may be launched at B>1 must address the plane even while B==1.
        self._a2a_b_plane = bool(transpose_in and (b_plane or b_dynamic))
        if b_dynamic and transpose_in and not bool(getattr(self, "_a2a_dynamic", False)):
            raise ValueError(
                "b_dynamic=True requires dynamic=True: the static transpose_in walk bakes (Xg, Yg) "
                "as const_expr, so a runtime batch extent has nothing to derive Xg from."
            )
        if not transpose_in:
            return
        if token_grid is None or len(token_grid) != 3:
            raise ValueError(
                f"transpose_in=True requires token_grid=(B, Xg, Yg); got {token_grid!r}."
            )
        B, Xg, Yg = (int(token_grid[0]), int(token_grid[1]), int(token_grid[2]))
        if B <= 0 or Xg <= 0 or Yg <= 0:
            raise ValueError(f"token_grid={token_grid} entries must be positive.")
        if B * Xg * Yg != int(rows_per_peer):
            raise ValueError(
                f"token_grid {token_grid} product {B * Xg * Yg} != rows_per_peer={rows_per_peer} "
                f"(the local token block M extent)."
            )
        # ARBITRARY Xg (route-2 (A) PADDED-per-Y): Xg NEED NOT be a CTA tile_M multiple. The
        # get_scheduler_arguments override emits B·Yg·ceil(Xg/BLK_M) M-tiles (each Y padded to a whole
        # BLK_M count) so NO output M-tile straddles the Xg->Y boundary; the A-load OOB-clamps the
        # partial last X-tile (Xg-bounded (Xg,K) view -> the pad rows read 0 -> acc=0 -> glu(0)=0 tail).
        # The store is UNCHANGED (contiguous WALK index) but now lands in a PADDED recv whose per-peer
        # token extent is the padded count -> OVERRIDE _a2a_rows_per_peer to it. Every per-Y block then
        # starts BLK_M-aligned (n_x·BLK_M stride) so the store box coord is 16-B/epi-M-aligned (NO wall,
        # vs the unpadded tile_y·Xg remap which is misaligned for any Xg not %epi_M). The zero pad tail
        # is einsum-safe (contributes 0 to any back contraction). CONTRACT: the caller (test / workflow)
        # MUST size the symmetric recv (2*Dloc, cp·rpp) from self._a2a_rows_per_peer (now PADDED) AFTER
        # configure — non-transpose leaves it unchanged (byte-identical).
        blk_m = int(self.tile_shape_mn[0])
        n_x = (Xg + blk_m - 1) // blk_m
        self._a2a_rows_per_peer = int(
            B * Yg * n_x * blk_m
        )  # PADDED per-Y token extent (recv sized from this)
        self._token_grid_b, self._token_grid_x, self._token_grid_y = B, Xg, Yg

    def dynamic_token_grid_b(self, B):
        """The runtime token-batch extent for ``EpilogueArguments.token_grid_b``.

        Returns ``Int32(B)`` when the functor was configured ``b_dynamic=True``, else ``None`` --
        so a caller can pass the result unconditionally and a NON-b_dynamic build keeps the field
        unset (byte-identical: nothing reads it). Raises ``ValueError`` on a build that bakes B if
        the requested extent differs from the baked one, because that combination is silently wrong
        rather than merely unsupported: the baked walk would unravel the flat M-tile with the wrong
        ``Xg`` and deliver another plane's rows in exactly the right shape.

        Args:
            B: the batch extent of THIS launch; must be a positive int.
        """
        B = int(B)
        if B <= 0:
            raise ValueError(f"token batch extent B={B} must be positive.")
        if not bool(getattr(self, "_token_grid_b_dynamic", False)):
            if bool(getattr(self, "_transpose_in", False)) and B != int(self._token_grid_b):
                raise ValueError(
                    f"this transpose_in front baked B={int(self._token_grid_b)} and was asked to "
                    f"run at B={B}. Build it with b_dynamic=True (one compile, every batch) -- a "
                    f"baked B unravels the walk with the wrong Xg and returns another plane's rows."
                )
            return None
        return Int32(B)

    def dynamic_token_grid_yg(self, N, M, B=None):
        """HOST-side contract check for dynamic transpose_in — the runtime analogue of
        _configure_transpose_in's static assert. Returns ``Int32(N)`` for
        ``EpilogueArguments.token_grid_yg`` after validating that ``Xg = M // (B*N)`` is a CTA
        ``tile_M`` multiple (no Y-straddle in the TMA box). The caller (workflow / test) calls this
        per launch with the CURRENT (N, M) and raises a CLEAR ValueError on a bad N (the kernel
        can't raise, so this host gate is where the contract is enforced for one-compile-many-N).

        ``B`` is THIS launch's batch extent; ``None`` falls back to the configured
        ``_token_grid_b``, which is the only correct source on a build that bakes B."""
        B = int(self._token_grid_b) if B is None else int(B)
        N, M = int(N), int(M)
        if B * N <= 0 or M % (B * N) != 0:
            raise ValueError(
                f"dynamic transpose_in: M={M} not divisible by B*N={B * N} (bad token grid)."
            )
        # ARBITRARY Xg (route-2 (A) PADDED-per-Y): NO Xg%BLK_M requirement. The scheduler override pads
        # each Y to a whole BLK_M count (n_x=ceil(Xg/BLK_M)); the DYNAMIC store derives the padded
        # rows_per_peer from the runtime recv M_full//cp, so the caller MUST size the per-N symmetric
        # recv PADDED via padded_rows_per_peer_dynamic(N, M). (M % (B*N) == 0 above stays the genuine
        # floor — a bad token grid.)
        return Int32(N)

    def padded_rows_per_peer_dynamic(self, N, M, B=None):
        """PADDED per-peer token extent at runtime (N=Yg, M=B·Xg·Yg) for a dynamic transpose_in front:
        ``B·N·ceil((M/(B·N))/BLK_M)·BLK_M``. The caller sizes the per-N symmetric recv
        ``(2*Dloc, cp·this)`` from it (the store walks a padded contiguous recv; the pad tail is 0).
        The STATIC path bakes the padded extent into self._a2a_rows_per_peer at configure — this is the
        one-compile-many-N analogue (per-launch, since the recv size varies with N).

        ``B`` is THIS launch's batch extent; ``None`` falls back to the configured
        ``_token_grid_b``."""
        B = int(self._token_grid_b) if B is None else int(B)
        N, M = int(N), int(M)
        if B * N <= 0 or M % (B * N) != 0:
            raise ValueError(f"padded_rows_per_peer_dynamic: M={M} not divisible by B*N={B * N}.")
        Xg = M // (B * N)
        blk_m = int(self.tile_shape_mn[0])
        n_x = (Xg + blk_m - 1) // blk_m
        return int(B * N * n_x * blk_m)

    def _configure_pad_inner(self, pad_inner, inner_extent, token_count, rows_per_peer):
        """P2 — bake the PAD-the-native-INNER-token-axis walk (the MIRROR of ``transpose_in``).

        ``transpose_in`` pads the walk's native-OUTER axis (``Xg``) and walks it inner; ``pad_inner``
        keeps the NATIVE order (inner axis stays inner) and pads the native-INNER axis ``Yg`` to a whole
        CTA ``BLK_M`` count, so every ``(b, i)`` token row starts ``BLK_M``-aligned in the flat GEMM-M
        index. The recv column of the flat-M index is then ``(b*N_i_loc + i)*Yg_pad + j``, i.e. the
        per-rank block stride becomes ``rpp_pad = B*N_i_loc*Yg_pad`` and every rank's destination base
        ``r*rpp_pad*2`` bytes is 128-B clean (``Yg_pad % BLK_M == 0`` and ``BLK_M ∈ {128,256}``).
        THAT is the whole point: it removes the ``2*rpp mod 128`` destination-phase penalty
        (:1239-1255) WITHOUT breaking the zero-copy contract, because ``rpp_pad`` factorises through
        ``N_i_loc`` exactly, so the global-i axis stays ONE mode of stride ``Yg_pad`` and every
        downstream reader is still a strided VIEW (see ``reshard.front_unpack_dmajor_2d``). Padding
        ``rpp`` itself does NOT factorise and is the expensive composite variant (:1257-1268).

        Parameters
        ----------
        inner_extent : int
            ``Yg`` — the WALK's inner token extent. Two host-selected regimes, ONE kernel:
            * **PAD** (the shape carries the phase penalty): ``Yg = N_j_loc``, so ``Xg = M//Yg =
              B*N_i_loc`` and the scheduler emits ``Xg*ceil(Yg/BLK_M)`` M-tiles.
            * **DECLINE** (the shape is ALREADY phase-clean — ``rpp % 64 == 0``): ``Yg = M``, so
              ``Xg = 1``, ``tile_y == tile_m``, ``tile_x == 0`` and the walk/tile-count collapse
              EXACTLY onto the unpadded flat-M walk (``A row == tile_m*BLK_M + r``). The pad must be
              declined there or it would COST work for zero benefit (e.g. ``N_j_loc = 1088`` is
              ``%64==0`` — already clean — but ``%128==64``, so a blind pad buys nothing).
            The regime is a per-N HOST decision precisely because ``dynamic_shape`` compiles ONCE for
            many N: the kernel takes ``Yg`` at RUNTIME (``EpilogueArguments.token_grid_yg``) so one
            compile serves both regimes.
        token_count : int
            ``M = B*N_i_loc*N_j_loc``, the TRUE (unpadded) A-operand row count. ``Xg = M // Yg``.
        rows_per_peer : int
            The recv's per-peer column stride the CALLER sized the symmetric recv from — ``rpp_pad``
            in the PAD regime, ``M`` in the DECLINE regime. Validated against both here so a caller
            that sizes the recv one way and configures the walk the other is rejected at the API
            boundary rather than silently corrupting a peer's column block.
        """
        self._pad_inner = bool(pad_inner)
        if not pad_inner:
            return
        if bool(getattr(self, "_transpose_in", False)):
            raise ValueError(
                "pad_inner and transpose_in are mutually-exclusive token walks (one pads the native-"
                "INNER axis and keeps it inner, the other pads the native-OUTER axis and walks it inner)."
            )
        if inner_extent is None or token_count is None:
            raise ValueError("pad_inner=True requires inner_extent (Yg) and token_count (M).")
        Yg, M = int(inner_extent), int(token_count)
        if Yg <= 0 or M <= 0 or M % Yg != 0:
            raise ValueError(
                f"pad_inner: token_count={M} must be a positive multiple of inner_extent={Yg}."
            )
        Xg = M // Yg
        blk_m = int(self.tile_shape_mn[0])
        n_y = (Yg + blk_m - 1) // blk_m
        rpp = int(rows_per_peer)
        if rpp not in (M, Xg * n_y * blk_m):
            raise ValueError(
                f"pad_inner: rows_per_peer={rpp} must be either the PADDED extent "
                f"Xg*ceil(Yg/BLK_M)*BLK_M={Xg * n_y * blk_m} (pad regime) or the raw token_count={M} "
                f"(decline regime); Xg={Xg}, Yg={Yg}, BLK_M={blk_m}."
            )
        self._pad_inner_x, self._pad_inner_y = Xg, Yg

    def pad_inner_token_grid_yg(self, Yg, M):
        """HOST-side contract check for dynamic pad_inner — the runtime analogue of
        ``_configure_pad_inner``'s static assert. Returns ``Int32(Yg)`` for
        ``EpilogueArguments.token_grid_yg`` after validating ``M % Yg == 0`` (so ``Xg = M//Yg`` is
        exact). The kernel cannot raise, so this host gate is where the contract is enforced for
        one-compile-many-N. Mirror of ``dynamic_token_grid_yg``."""
        Yg, M = int(Yg), int(M)
        if Yg <= 0 or M <= 0 or M % Yg != 0:
            raise ValueError(f"dynamic pad_inner: token_count M={M} not divisible by Yg={Yg}.")
        return Int32(Yg)

    def padded_rows_per_peer_inner(self, Yg, M):
        """PADDED per-peer token extent ``(M//Yg)*ceil(Yg/BLK_M)*BLK_M`` for a pad_inner front.

        The host sizes the per-N symmetric recv ``(2*Dloc, cp*this)`` from it. The STATIC path bakes
        the same number into ``_a2a_rows_per_peer`` at configure; this is the one-compile-many-N
        analogue (the recv size varies with N). Mirror of ``padded_rows_per_peer_dynamic``."""
        Yg, M = int(Yg), int(M)
        if Yg <= 0 or M % Yg != 0:
            raise ValueError(f"padded_rows_per_peer_inner: M={M} not divisible by Yg={Yg}.")
        blk_m = int(self.tile_shape_mn[0])
        return int((M // Yg) * ((Yg + blk_m - 1) // blk_m) * blk_m)

    def _use_3d_A_load(self):
        """UNIFIED (GATE 2: 3-D-static == composite bit-exact AND perf-equal, ratio 0.993): the 3-D
        A-load is used for ALL transpose_in (static AND dynamic) — the composite rank-2-M is retired.
        The 3-D form (A as 4-D (Xg,K,Yg,L) / 5-D (Xg,K,Yg,B,L), tiled by the DEFAULT 2-tuple
        (BLK_M,BLK_K) leaving Yg[,B],L untiled) is composite-free so its box is static (BLK_M,BLK_K)
        regardless of Xg,Yg -> the atom takes RUNTIME Xg,Yg (the composite V-map could not).

        pad_inner (P2) uses the SAME 3-D machinery with the two token modes SWAPPED: A as
        (Yg, K, Xg, L) with Yg (the native-INNER axis) at mode-0 = the tiled/padded one."""
        return bool(getattr(self, "_transpose_in", False)) or bool(
            getattr(self, "_pad_inner", False)
        )

    def _remap_A_operand_layout(self, mA, epi_args=None):
        """Route-2 override: present A so the GEMM WALKS its token-M axis in transposed (X<->Y-swapped)
        p-inner order — flat GEMM-M index ``i`` reads native row ``b*Xg*Yg + X*Yg + Y``; the postact
        store writes recv col = flat-M index, so the recv auto-comes-out einsum-k-major for the back
        with NO store change. Default-off => identity (byte-identical).

        UNIFIED 3-D form (static + dynamic): a composite-free (Xg, K, Yg[, B], L) tensor — K kept at
        mode-1 (parent shape[1]=K uses stay byte-identical); token axes = mode-0 (Xg), mode-2 (Yg) and
        (B>1) mode-3 (B). The atom is built with the DEFAULT 2-tuple cta_tiler (BLK_M,BLK_K) which
        tiles (Xg,K) and leaves Yg[,B],L untiled, so its box is a static (BLK_M,BLK_K) -> the descriptor
        takes RUNTIME Xg,Yg. The kernel (_gA_local_tile) selects Y=tile_y[,B=tile_b],L=batch then tiles.
        STATIC: (Xg,Yg) baked. DYNAMIC (one-compile-many-N): Yg=epi_args.token_grid_yg, Xg=M//(B*Yg)
        (exact). All strides are rs-multiples => 16-B TMA aligned any N."""
        if const_expr(getattr(self, "_pad_inner", False)):
            # pad_inner: (Yg, K, Xg, L) — the MIRROR of the transpose_in layout below. Mode-0 is the
            # native-INNER token axis Yg (stride = the native row stride rs, i.e. it stays inner) and
            # mode-2 is the native-OUTER axis Xg (stride Yg*rs). Flat native row = X*Yg + Y, so the walk
            # order is the NATIVE one — the ONLY change is that mode-0 (not the flat M) is what
            # _gA_local_tile tiles by BLK_M, which is what makes every X row start BLK_M-aligned in the
            # flat GEMM-M index. B is FOLDED INTO Xg (native m = (b*N_i_loc + i)*N_j_loc + j collapses
            # exactly), so there is no 5-D form and pad_inner is B-general.
            rs_p = mA.stride[0]
            fs_p = mA.stride[1]
            ls_p = mA.stride[2]
            K_p = mA.shape[1]
            if const_expr(getattr(self, "_a2a_dynamic", False)):
                assert epi_args is not None and epi_args.token_grid_yg is not None, (
                    "dynamic + pad_inner requires EpilogueArguments.token_grid_yg (runtime Yg); "
                    "build it via self.pad_inner_token_grid_yg(Yg, M)."
                )
                Yg_p = Int32(epi_args.token_grid_yg)  # runtime inner walk extent
                Xg_p = Int32(mA.shape[0]) // Yg_p  # runtime, exact (M = Xg*Yg)
            else:
                Xg_p = const_expr(self._pad_inner_x)
                Yg_p = const_expr(self._pad_inner_y)
            lay_p = cute.make_layout(
                (Yg_p, K_p, Xg_p, mA.shape[2]), stride=(rs_p, fs_p, Yg_p * rs_p, ls_p)
            )
            return cute.make_tensor(mA.iterator, lay_p)
        if const_expr(not getattr(self, "_transpose_in", False)):
            return mA
        Bt = const_expr(self._token_grid_b)
        b_dyn = const_expr(bool(getattr(self, "_token_grid_b_dynamic", False)))
        rs = mA.stride[0]  # native row stride (element units; = K for token-contiguous A)
        fs = mA.stride[1]  # feature (K) stride (= 1)
        ls = mA.stride[2]  # L (batch/group) stride
        K = mA.shape[1]
        if const_expr(getattr(self, "_a2a_dynamic", False)):
            assert epi_args is not None and epi_args.token_grid_yg is not None, (
                "dynamic + transpose_in requires EpilogueArguments.token_grid_yg (runtime Yg=N); "
                "build it via self.dynamic_token_grid_yg(N, M)."
            )
            Yg = Int32(epi_args.token_grid_yg)  # runtime 2nd token axis (=N)
            if const_expr(b_dyn):
                assert epi_args.token_grid_b is not None, (
                    "b_dynamic + transpose_in requires EpilogueArguments.token_grid_b (runtime B); "
                    "build it via self.dynamic_token_grid_b(B)."
                )
                Bv = Int32(epi_args.token_grid_b)  # runtime token-batch extent
            else:
                Bv = Int32(Bt)
            Xg = Int32(mA.shape[0]) // (Bv * Yg)  # runtime, exact (M=B*Xg*Yg)
        else:
            Bv = Int32(Bt)
            Xg = const_expr(self._token_grid_x)
            Yg = const_expr(self._token_grid_y)
        if const_expr(b_dyn):
            # B-DYNAMIC: the batch takes the DEGENERATE L slot rather than adding a fifth mode, so
            # the TMA descriptor stays RANK 4 -- (Xg, K, Yg, B) with the batch stride Xg*Yg*rs where
            # the L stride used to sit. This front's A is always (M, K, 1) (the caller builds it as
            # `x_norm_2d.reshape(1, M, K).permute(1, 2, 0)`), so nothing is lost: the L coordinate
            # was always 0. MEASURED, and this is why it is written this way: as a genuine 5-D
            # operand the b_dynamic front cost +1.4% at B==1 against the baked-B build, while the
            # b-plane STORE cost only +0.29% -- i.e. essentially all of the runtime-batch tax was
            # the descriptor RANK, not the extra arithmetic.
            #
            # `mainloop_remap_mA` must select this mode with the PLANE index instead of
            # `batch_idx`; the two are the same value (0) at B==1 and only the plane is right above
            # it, which is what keeps the B==1 addressing identical.
            lay = cute.make_layout((Xg, K, Yg, Bv), stride=(Yg * rs, fs, rs, Xg * Yg * rs))
        elif const_expr(Bt > 1):
            # 5-D (Xg, K, Yg, B, L): token-batch B is mode-3 (native m = b*Xg*Yg + X*Yg + Y).
            lay = cute.make_layout(
                (Xg, K, Yg, Bt, mA.shape[2]), stride=(Yg * rs, fs, rs, Xg * Yg * rs, ls)
            )
        else:
            lay = cute.make_layout((Xg, K, Yg, mA.shape[2]), stride=(Yg * rs, fs, rs, ls))
        return cute.make_tensor(mA.iterator, lay)

    def mainloop_remap_mA(self, mA_mk, tile_coord_mnkl, mA_mkl=None, batch_idx=None):
        """SELECT half of the 3-D A-load: unravel the flat scheduler tile_m, pick the fixed
        Y (and B), and return a plain 2-D ``(Xg, K)`` — the TILE half is :meth:`_gA_local_tile`.

        **Why the work is split across two hooks here when the port's source did it in one.**
        This tree factors the A-load into ``select_batch`` -> ``mainloop_remap_mA`` ->
        ``_gA_local_tile``; the kernel this ports from had a single 3-parameter
        ``_gA_local_tile`` doing select-and-tile together. The two halves land on the two hooks
        that already exist, and **not** on ``select_batch``: that one is operand-GENERIC (the
        loader calls it for ``mA``, ``mB`` and ``mB2``), so putting the A-operand unravel there
        would silently apply it to the B operands. This hook is A-specific, which is why it
        takes the pre-selection ``mA_mkl``/``batch_idx`` — the modes it must index are already
        collapsed in ``mA_mk``.

        Args:
            mA_mk: A after ``select_batch``. Returned unchanged when the 3-D load is off, which
                is the flag-off byte-identical path.
            tile_coord_mnkl: Scheduler tile coord; ``[0]`` is the flat token-grid tile index.
            mA_mkl: The pre-selection ``(Xg, K, Yg[, B], L)`` A. **Required** when the 3-D load
                is on -- ``None`` there is a caller that did not pass it, and indexing it raises
                rather than silently reading the wrong tile.
            batch_idx: The L coordinate, same requirement.

        Returns:
            ``(Xg, K)`` (or ``(Yg, K)`` under ``_pad_inner``) for the tile half to cut.
        """
        if const_expr(not self._use_3d_A_load()):
            return mA_mk
        blk_m = cute.select(self.cta_tile_shape_mnk, [0, 2])[0]
        tile_m = tile_coord_mnkl[0]
        if const_expr(getattr(self, "_pad_inner", False)):
            # pad_inner: mode-0 is Yg (the native-INNER axis), so the flat tile_m unravels
            # Y-INNER and it is X that is SELECTED here.
            n_y_p = cute.ceil_div(mA_mkl.shape[0], blk_m)
            return mA_mkl[None, None, tile_m // n_y_p, batch_idx]  # (Yg, K)
        n_x = cute.ceil_div(mA_mkl.shape[0], blk_m)  # runtime x-tile count (Xg dynamic)
        # Three operand shapes, and the select has to match whichever `_remap_A_operand_layout`
        # built:
        #   * b_dynamic  -> RANK 4 (Xg, K, Yg, B): the batch occupies the degenerate L slot, so the
        #     last coordinate is the PLANE, not `batch_idx` (which is 0 here -- the front's A is
        #     (M, K, 1) -- so the two agree at B==1 and differ only above it).
        #   * baked B>1  -> rank 5 (Xg, K, Yg, B, L): plane AND L.
        #   * otherwise  -> rank 4 (Xg, K, Yg, L), the unchanged path.
        if const_expr(bool(getattr(self, "_token_grid_b_dynamic", False))):
            n_y = mA_mkl.shape[2]  # Yg -- consumed by this select, hence read BEFORE it
            rem = tile_m // n_x
            return mA_mkl[None, None, rem % n_y, rem // n_y]  # (Xg, K)
        if const_expr(self._token_grid_b > 1):
            n_y = mA_mkl.shape[2]  # Yg -- consumed by this select, hence read BEFORE it
            rem = tile_m // n_x
            return mA_mkl[None, None, rem % n_y, rem // n_y, batch_idx]  # (Xg, K)
        return mA_mkl[None, None, tile_m // n_x, batch_idx]  # (Xg, K)

    def _gA_local_tile(self, mA_mk, tile_coord_mnkl):
        """TILE half of the 3-D A-load: cut ``(BLK_M, BLK_K)`` at the X-inner tile index.

        Both unravel branches tile at ``tile_m % n``, and ``n`` is recoverable from the
        SELECTED tensor: ``mainloop_remap_mA`` collapses the Y/B/L modes but leaves mode-0
        (``Xg``, or ``Yg`` under ``_pad_inner``) intact, so ``mA_mk.shape[0]`` is the same
        extent the select half divided by. That is what lets the two halves sit on separate
        hooks without threading state between them.

        Args:
            mA_mk: The 2-D operand from :meth:`mainloop_remap_mA`.
            tile_coord_mnkl: Scheduler tile coord; ``[0]`` is the flat token-grid tile index.

        Returns:
            ``(BLK_M, BLK_K, RestK)``.
        """
        if const_expr(not self._use_3d_A_load()):
            return super()._gA_local_tile(mA_mk, tile_coord_mnkl)
        blk_mk = cute.select(self.cta_tile_shape_mnk, [0, 2])  # (BLK_M, BLK_K)
        n = cute.ceil_div(mA_mk.shape[0], blk_mk[0])
        return cute.local_tile(mA_mk, blk_mk, (tile_coord_mnkl[0] % n, None))

    # ---- WIDE-PUT run_i raster (Stage-1 STATIC) ------------------------------------------------
    # Minimum consecutive-M run length R below which the wide-put run_i raster is NOT worth engaging
    # (a sub-R_MIN coalesce batches < R_MIN·epi_m·2B per put -> keep the default serpentine raster,
    # BYTE-IDENTICAL). A tunable PERF-heuristic threshold, NOT a shape constraint (any R is correct;
    # below R_MIN we just decline the small coalescing win). R_MIN=2 (was 8): a direct-probe small-N
    # sweep (D256/cp16 venue D, cache=0) placed the coalesce engage-boundary at R>=2. R=1 is a genuine
    # NO-OP (one tile, nothing to stitch across: N1024 0.2465 engaged vs 0.2492 declined, ~noise), but
    # R>=2 already wins: N1088(R=2) 0.252->0.164 = 1.54× (reproducible), N2560(R=4) 0.349->0.176 = 2.0×
    # (N2048(R=3) a smaller ~6%, re-confirmed by the rounds-11 grid). Old R_MIN=8 stranded the whole
    # R=2..7 band (N~1088-8000) on 256B puts. The R>=8 control (N10048, R=19) is unchanged -> no
    # regression; R's occupancy cap (_derive_run_i_tiles) is untouched and the W× ring is allocated
    # regardless of R (already-paid; USING it is strictly better, no new footprint). R=1 (N<~1088)
    # correctly stays on the default raster.
    _A2A_RUN_I_RMIN = 2

    @staticmethod
    def _derive_run_i_tiles(ncluster_m, ncluster_n, grid_z, W, r_min):
        """Occupancy-preserving consecutive-M run length R for the wide-put run_i raster (the front's
        TOKEN = GEMM-M coalescing axis). ``R = min(W, floor(ncluster_m·ncluster_n / grid_z))`` — the
        LONGEST R-tile run per persistent CTA that STILL keeps ``total_runs = ncluster_n·ceil(ncluster_m
        / R) >= grid_z`` so NO persistent cluster is left idle (occupancy proof: ``R <= ncm·ncn/gz`` =>
        ``ncn·ceil(ncm/R) >= ncn·ncm/R >= gz``; the W-cap only shrinks R, so it still holds). ``R <
        r_min`` => return 0 (default raster, byte-identical): the coalesced put would batch < r_min
        subtiles — not worth the W× ring footprint. Pure host int arithmetic -> unit-testable with NO
        GPU. Stage-1 bakes R as a compile-const (static shape); the dynamic-N runtime-R form is Stage-2."""
        if grid_z <= 0 or ncluster_m <= 0 or ncluster_n <= 0:
            return 0
        R = min(int(W), (int(ncluster_m) * int(ncluster_n)) // int(grid_z))
        return int(R) if R >= int(r_min) else 0

    @staticmethod
    def _largest_divisor_leq(n, cap):
        """Largest divisor of ``n`` that is <= ``cap`` (>=1). Used to snap the route2_ni run_i R to a
        DIVISOR of ncluster_m so ``runs_per_band = ncm/R`` is EXACT — the sub-band clamp (R∤ncm ->
        dup-absorb re-stores) DEADLOCKS the route2 b_j-pinned wide drain (the dst_j==b_j dup gate; D-major's
        identical sub-band is fine). O(cap) with cap <= W (<=128) -> cheap host int arithmetic."""
        n, cap = int(n), int(cap)
        if n <= 0 or cap <= 0:
            return 0
        d = min(cap, n)
        while d > 1 and (n % d) != 0:
            d -= 1
        return d

    @staticmethod
    def _a2a_run_i_grid_z(scheduler_args, cluster_size):
        """The persistent grid_z ceiling (HW max-active clusters) as a Python int, for the run_i
        R-derivation. PRIMARY: the EXACT value the launch's ``get_grid_shape`` uses to bound grid_z —
        ``int(scheduler_args.max_active_clusters)`` (a STATIC Int32 wrapping the caller's Python int ->
        int() yields it, NO device query, definitionally the launch's ceiling for whatever cluster_size
        the caller sized it at). FALLBACK: a fresh ``get_max_active_clusters(cluster_size)`` device query
        (needs a CUDA context; the gotcha-1 stash). Returns 0 if NEITHER yields a usable int -> the
        caller then DECLINES run_i (default raster, byte-identical) rather than derive from a guessed
        grid_z (a too-small guess would over-size R -> occupancy collapse; declining is always safe)."""
        mac = getattr(scheduler_args, "max_active_clusters", None)
        if mac is not None:
            try:
                return int(mac)
            except Exception:
                pass
        try:
            return int(get_max_active_clusters(cluster_size))
        except Exception:
            return 0

    def _maybe_inject_run_i_wide(self, args, scheduler_args):
        """Inject the wide-put run_i (consecutive-M-at-fixed-N) raster into the scheduler args for the
        ib_wide path. STATIC (Stage-1): derive a COMPILE-CONST R from ``args.problem_shape_ntile_mnl``
        (real ints; route2 snaps R|ncm). DYNAMIC (Stage-2): the runtime ntile is a symbolic Int32
        (mark_layout_dynamic), so the occupancy-optimal R is derived at RUNTIME (in the traced scheduler-
        args build) from the runtime ncluster_m and injected as ``run_i_dynamic`` + ``run_i_run_len`` (a
        runtime Int32) — a compile-const R baked from the ANCHOR over-sizes at a smaller runtime ncm ->
        total_runs<grid_z -> idle clusters -> a NON-anchor perf regression (the wide-put's occupancy proof
        needs R<=ncm·ncn/grid_z at the ACTUAL ncm; measured -34% cp=2 D256). Delegated to
        :meth:`_maybe_inject_run_i_wide_dynamic`. Default-off leaves ``args`` UNCHANGED -> byte-identical
        serpentine raster. The derived R (or -1 sentinel on the runtime path) is stashed on self
        (``_a2a_run_i_tiles_derived``) so a test can assert run_i ACTUALLY engaged."""
        self._a2a_run_i_tiles_derived = 0
        # DEBUG/A-B knob: CPO_FRONT_NO_RUN_I disables the run_i raster injection. NOTE the check is
        # `if os.environ.get(...)` so ANY non-empty value (incl "0") DISABLES — pass EMPTY/unset to ENABLE.
        if os.environ.get("CPO_FRONT_NO_RUN_I"):
            return args
        if not getattr(self, "_a2a_ib_wide", False):
            return args  # ib_wide off -> no run_i (byte-identical serpentine raster)
        if getattr(self, "_a2a_dynamic", False):
            # DYNAMIC (Stage-2): the runtime ncm demands a RUNTIME R (occupancy-optimal at every N); the
            # dynamic inject dispatches per-variant inside _derive_run_i_len_rt (D-major arbitrary-R /
            # route2_ni divisor-snap).
            return self._maybe_inject_run_i_wide_dynamic(args, scheduler_args)
        # STATIC (Stage-1): real int ntile -> compile-const R (byte-identical to today).
        ntile = getattr(args, "problem_shape_ntile_mnl", None)
        if ntile is None or ntile[0] is None or not isinstance(ntile[0], int):
            return args  # ntile_m None, or a dynamic Int32 extent -> default raster (safe)
        cm = int(args.cluster_shape_mnk[0])
        cn = int(args.cluster_shape_mnk[1])
        ncm = (int(ntile[0]) + cm - 1) // cm  # ncluster_m = ceil(ntile_m / cluster_m)
        ncn = (int(ntile[1]) + cn - 1) // cn  # ncluster_n = ceil(ntile_n / cluster_n)
        # grid_z = the persistent cluster count the launch bounds by the HW max-active clusters (the
        # launch's get_grid_shape does min(problem, max_active); using max_active is the SATURATED,
        # conservative choice — can only SHRINK R, never overshoot occupancy). cluster_size = cm·cn.
        grid_z = self._a2a_run_i_grid_z(scheduler_args, cm * cn)
        if grid_z <= 0:
            return args  # couldn't size grid_z -> decline run_i (byte-identical, never a guessed R)
        R = self._derive_run_i_tiles(
            ncm, ncn, grid_z, self._a2a_ib_wide_batch, self._A2A_RUN_I_RMIN
        )
        # route2_ni SUB-BAND FIX (#8): snap static R to the largest divisor of ncm <= the occupancy-safe R
        # (runs_per_band EXACT -> NO clamp/dup -> deadlock-free b_j-pinned drain). D-major keeps the
        # arbitrary-R sub-band (its dup-absorb re-store path is correct).
        if getattr(self, "_a2a_route2_ni", False) and R > 0:
            R = self._largest_divisor_leq(ncm, R)
            if R < self._A2A_RUN_I_RMIN:
                R = 0
        # DIAGNOSTIC override: CPO_FRONT_RUN_I_R forces R (static path only; the dynamic R is runtime).
        _force_r = os.environ.get("CPO_FRONT_RUN_I_R")
        if _force_r and R > 0:
            R = int(_force_r)
        self._a2a_run_i_tiles_derived = int(R)
        if R > 0:
            args = dataclasses.replace(args, run_i_tiles=int(R))
        return args

    def _maybe_inject_run_i_wide_dynamic(self, args, scheduler_args):
        """STAGE-2 DYNAMIC-N wide-put run_i (BOTH variants). Under dynamic-shape compile ``ntile_m`` is a
        RUNTIME Int32, so the occupancy-optimal run-length R is derived HERE at runtime (in the traced
        scheduler-args build) and injected as ``run_i_dynamic`` + ``run_i_run_len`` (a runtime Int32) —
        NOT baked from the anchor (a compile-const R over-sizes at a smaller runtime ncm -> total_runs <
        grid_z -> idle clusters -> the measured -34% non-anchor regression, cp=2 D256). The scheduler's
        run_i_dynamic decode (FastDivmod(R)) + the wide drain's runtime-R count-parity consume it. The
        per-variant R-derivation lives in ``_derive_run_i_len_rt`` (const_expr-gated on ``_a2a_route2_ni``):
        route2_ni snaps R to a DIVISOR of the runtime ncm (runs_per_band=ncm/R EXACT -> NO b_dup re-store
        -> the route2 b_j-pinned wide drain can't hang, the #8 deadlock); D-major uses ARBITRARY R (its
        dup-absorb sub-band re-stores clamped-pad tiles idempotently -> R need NOT divide ncm).
        ``CPO_FRONT_RUN_I_R`` is not honored (R is a runtime Int32, not a host const)."""
        ntile = getattr(args, "problem_shape_ntile_mnl", None)
        if ntile is None or ntile[0] is None:
            return args  # ntile_m None -> default raster (safe)
        cm = int(args.cluster_shape_mnk[0])
        cn = int(args.cluster_shape_mnk[1])
        grid_z = self._a2a_run_i_grid_z(scheduler_args, cm * cn)  # host int (SM-count ceiling)
        if grid_z <= 0:
            return args  # couldn't size grid_z -> decline run_i (byte-identical, never a guessed R)
        # runtime ncluster_m/n (ntile_m runtime Int32; ntile_n runtime/static-feature -> Int32() coerces).
        # PLAIN Int32 arithmetic only here (this method is NOT @cute.jit -> the preprocessor does NOT run,
        # so range_constexpr / the runtime min/max/ternary-select would raise "should be preprocessed"). The
        # scan + min/max live in the @cute.jit _derive_run_i_len_rt (preprocessed).
        ncm = (Int32(ntile[0]) + Int32(cm - 1)) // Int32(cm)
        ncn = (Int32(ntile[1]) + Int32(cn - 1)) // Int32(cn)
        run_len = self._derive_run_i_len_rt(ncm, ncn, Int32(grid_z))
        self._a2a_run_i_tiles_derived = -1  # sentinel: dynamic run_i engaged (R is a runtime Int32)
        args = dataclasses.replace(args, run_i_dynamic=True, run_i_run_len=run_len)
        return args

    @cute.jit
    def _derive_run_i_len_rt(self, ncm, ncn, grid_z):
        """@cute.jit runtime run-length R for the DYNAMIC wide-put run_i (BOTH variants; preprocessed so
        the runtime min/max + the range_constexpr scan are legal — a plain method can host neither).
        Common cap: R_cap = min(W, floor(ncm·ncn/grid_z)) keeps ncn·ceil(ncm/R) >= grid_z so NO cluster
        idles at any N (the runtime mirror of the host ``_derive_run_i_tiles`` — the fix for the
        anchor-baked-R non-anchor regression). const_expr-gated on ``_a2a_route2_ni``:
        * route2_ni: R = largest DIVISOR of the runtime ncm that is <= R_cap (BOUNDED scan d=R_MIN..W, W a
          COMPILE-CONST -> unrolled; FLAT ternary-select body, NO nested if -> NOT the range_constexpr
          blowup class; measured 0.36 s). R | ncm -> runs_per_band=ncm/R EXACT -> NO sub-band clamp/b_dup
          re-store -> the route2 b_j-pinned wide drain can't hang (#8). Clamp >=1 (1 divides ANY ncm) ->
          deadlock-safe even for a no-divisor shape.
        * D-major: R = R_cap directly (ARBITRARY R — its dup-absorb sub-band re-stores the clamped-pad
          tiles idempotently and the wide drain's count-parity is raster-agnostic in R, so R need NOT
          divide ncm; route2 needs the snap for its b_j-pinned drain, D-major does not).
        Clamp >=1 (R=1 = run_i-off-equivalent; NEVER a div-by-zero)."""
        W: cutlass.Constexpr[int] = int(self._a2a_ib_wide_batch)
        R_occ = (ncm * ncn) // grid_z
        R_cap = cutlass.min(Int32(W), R_occ)
        if cutlass.const_expr(bool(getattr(self, "_a2a_route2_ni", False))):
            r_min: cutlass.Constexpr[int] = int(self._A2A_RUN_I_RMIN)
            R = Int32(0)
            for d in cutlass.range_constexpr(r_min, W + 1):
                dd = Int32(d)
                R = (
                    dd if ((dd <= R_cap) and ((ncm % dd) == Int32(0))) else R
                )  # keep the LARGEST qualifier
            return cutlass.max(R, Int32(1))
        return cutlass.max(R_cap, Int32(1))  # D-major: arbitrary R (dup-absorb handles R∤ncm)

    def get_scheduler_arguments(self, mA, mB, mD, scheduler_args, epilogue_args):
        """Route-2 arbitrary-N_loc (partial Xg-tile): for transpose_in, present the scheduler a
        PADDED-M A so it emits B·Yg·ceil(Xg/BLK_M) M-tiles — each Y padded to a whole BLK_M count, so
        NO output M-tile straddles the Xg→Y boundary (the fix for Xg NOT %tile_M). The A-load unravels
        tile_m X-inner and OOB-clamps the partial last X-tile (Xg-bounded (Xg,K) view); the store maps
        the padded tile → contiguous recv col + per-Y clamp. base uses only mA.shape[0] (the tile count)
        so a fake padded-shape mA is safe. transpose_in off → the staged parent scheduler (byte-id).

        WIDE-PUT run_i: after building the base args, ``_maybe_inject_run_i_wide`` injects the
        consecutive-M-at-fixed-N run_i raster on the ib_wide path so the wide put's W-batch fills to R
        token-subtiles (the 32 KiB coalescing win). STATIC derives a compile-const R from the real ntile;
        DYNAMIC (D-major) derives R at RUNTIME from the runtime ncm (occupancy-optimal at every N ->
        run_i_dynamic + run_i_run_len). ib_wide off / dynamic route2 -> unchanged."""
        if const_expr(getattr(self, "_pad_inner", False)):
            # pad_inner: present a PADDED-M A so the scheduler emits Xg*ceil(Yg/BLK_M) M-tiles — each
            # X row padded to a whole BLK_M count, so no output M-tile straddles the Yg->X boundary and
            # every X row starts BLK_M-aligned in the flat M index. base uses only mA.shape[0] (the tile
            # count), so a fake padded-shape mA is safe (same contract as the transpose_in arm).
            blk_m_p = const_expr(self.cta_tile_shape_mnk[0])
            if const_expr(getattr(self, "_a2a_dynamic", False)):
                Yg_s = Int32(epilogue_args.token_grid_yg)
                Xg_s = Int32(mA.shape[0]) // Yg_s
                # SCALAR ceil_div (NOT cute.ceil_div — that builds a cute.tile op -> illegal
                # 'cute.derefine' in the @jit scheduler-args trace); Yg is a runtime Int32.
                n_y_s = (Yg_s + Int32(blk_m_p - 1)) // Int32(blk_m_p)
                m_padded_p = Xg_s * n_y_s * Int32(blk_m_p)  # runtime padded per-X M
            else:
                Xg_s = const_expr(self._pad_inner_x)
                Yg_s = const_expr(self._pad_inner_y)
                n_y_s = const_expr((Yg_s + blk_m_p - 1) // blk_m_p)
                m_padded_p = const_expr(Xg_s * n_y_s * blk_m_p)  # static padded per-X M
            mA_sched_p = cute.make_tensor(
                mA.iterator,
                cute.make_layout((m_padded_p, mA.shape[1], mA.shape[2]), stride=mA.stride),
            )
            args = DualGatedGemmSm90.get_scheduler_arguments(
                self, mA_sched_p, mB, mD, scheduler_args, epilogue_args
            )
            return self._maybe_inject_run_i_wide(args, scheduler_args)
        if const_expr(getattr(self, "_transpose_in", False)):
            blk_m = const_expr(self.cta_tile_shape_mnk[0])
            Bt = const_expr(self._token_grid_b)
            if const_expr(getattr(self, "_a2a_dynamic", False)):
                Yg = Int32(epilogue_args.token_grid_yg)
                # b_dynamic: the tile count scales with the RUNTIME batch, so the padded-M the
                # scheduler is shown must too -- a baked Bt here would emit Bt/B_rt of the tiles.
                if const_expr(bool(getattr(self, "_token_grid_b_dynamic", False))):
                    Bv = Int32(epilogue_args.token_grid_b)
                else:
                    Bv = Int32(Bt)
                Xg = Int32(mA.shape[0]) // (Bv * Yg)
                # SCALAR ceil_div (NOT cute.ceil_div — that builds a cute.tile op -> illegal
                # 'cute.derefine' in the @jit scheduler-args trace); Xg is a runtime Int32.
                n_x = (Xg + Int32(blk_m - 1)) // Int32(blk_m)
                m_padded = Bv * Yg * n_x * Int32(blk_m)  # runtime padded per-Y M
            else:
                Xg = const_expr(self._token_grid_x)
                Yg = const_expr(self._token_grid_y)
                n_x = const_expr((Xg + blk_m - 1) // blk_m)
                m_padded = const_expr(Bt * Yg * n_x * blk_m)  # static padded per-Y M
            mA_sched = cute.make_tensor(
                mA.iterator,
                cute.make_layout((m_padded, mA.shape[1], mA.shape[2]), stride=mA.stride),
            )
            args = DualGatedGemmSm90.get_scheduler_arguments(
                self, mA_sched, mB, mD, scheduler_args, epilogue_args
            )
        else:
            args = DualGatedGemmSm90.get_scheduler_arguments(
                self, mA, mB, mD, scheduler_args, epilogue_args
            )
        return self._maybe_inject_run_i_wide(args, scheduler_args)

    def _configure_decoupled(self, decoupled, ring_depth, consumer_warpgroups):
        """Bake the decoupled SIMT-ring-drain knobs (shared by configure_a2a / _sharded).

        ``decoupled`` turns on the producer-ring + dedicated consumer-warpgroup putwarp drain (the
        IB-swappable precursor); ``ring_depth`` (default 2) sets the rotating-ring depth (the caller
        allocates the GMEM ring at this depth); ``consumer_warpgroups`` (default 1) the drainer-WG
        count. All OFF/default => byte-identical to the coupled TMA-S2G store."""
        self._a2a_decoupled = bool(decoupled)
        if ring_depth is not None:
            if int(ring_depth) < 1:
                raise ValueError(f"ring_depth={ring_depth} must be a positive int.")
            self._a2a_ring_depth = int(ring_depth)
        if consumer_warpgroups is not None:
            self._a2a_consumer_warpgroups = int(consumer_warpgroups)

    # ==================================================================
    # DECOUPLED producer/consumer SIMT-ring drain — warpgroup-split hooks (mirror GemmSm90A2A).
    # ==================================================================
    # Per-slot metadata the producer writes for the consumer (the D-major recv index + peer + run):
    #   [0]=peer  [1]=dst_tok_sub (token recv-col tile, epi_m units; RELATIVE to my column block under
    #   auto-clamp)  [2]=dst_feat_sub (feature recv-row, epi_n units)  [3]=run_tokens (the valid token
    #   count of this — possibly partial / fully-OOB — subtile; the IB SIMT put's clamped run, 0 => skip;
    #   §2). 4 Int32 per ring slot. The front's peer is feature-tile-uniform per CTA (NOT the back's
    #   per-token-row dynamic peer). meta[3] adds +rd*4 B SMEM (frugal; 0 GMEM).
    #   WIDE-PUT (ib_wide): the per-slot metadata becomes PER-BATCH (a wide slot holds W accumulated
    #   subtiles): [0]=peer(-1 skip)  [1]=base_tok (the batch's FIRST subtile dst_tok, epi_m units)
    #   [2]=dst_feat  [3]=total_run (the batch's total VALID token count = the wide put length; the OOB
    #   suffix is excluded)  [4]=n_sub (subtiles absorbed by this batch, for the consumer's count-parity
    #   accounting: subtiles_drained += n_sub). 5 Int32/slot -> +rd*4 B SMEM vs per-subtile (frugal).
    #   B-MODE (_a2a_b_mode): a TRAILING ``tile_b`` field — the recv's batch PLANE. The coupled store
    #   carries the plane as an extra TMA store coordinate; the drain has no descriptor, so the plane
    #   has to travel in this record or the consumer addresses plane 0 for every plane. ALWAYS LAST, so
    #   every non-b build keeps its exact record (byte-identical) and the index is `nfields-1`.
    _DECOUPLED_META_FIELDS_BASE = 4
    _DECOUPLED_META_FIELDS_WIDE = 5
    # WIDE-PUT per-CTA batch state (SMEM, single-writer producer lane): [bc, b_w, b_base, b_peer,
    # b_feat, b_run, b_dup] — persists the pending wide batch across per-subtile copy_fn calls (see
    # _extra_smem_struct). ``b_dup`` (run_i DUPLICATE-ABSORB) counts the clamped-padding re-store subtiles
    # (R∤ncluster_m) that fall WITHIN the current batch's already-covered token range — absorbed into the
    # batch's n_sub for count-parity WITHOUT a redundant tiny put (the perf fix for the R∤ncm flood). 0 on
    # the default/no-run_i raster (no repeated dst_tok -> is_dup never fires). Only present on ib_wide.
    # route2_ni adds ``b_j`` (field 7) — a batch coalesces only within ONE N_j column, so the merge key
    # gains a dst_j==b_j test (D-major has no N_j axis -> 7 fields, byte-identical bstate).
    # b-mode adds ``b_b`` (LAST field) — the pending batch's recv PLANE. A batch coalesces only within
    # ONE plane (planes are not adjacent along the coalesce axis), so the merge AND duplicate-absorb keys
    # both gain a tile_b==b_b test. UNLIKE b_j this is needed by BOTH variants: composite_k has no N_j
    # column to accidentally separate the planes with. Not b-mode -> byte-identical bstate.
    _DECOUPLED_BSTATE_FIELDS_BASE = 7

    @property
    def _DECOUPLED_BSTATE_FIELDS(self):
        """Wide-put batch-state int32 count: 7, **+1 for route2_ni** (b_j, the N_j column the batch is
        pinned to), **+1 for b-mode** (b_b, the recv batch plane the batch is pinned to). A plain int
        (host/compile-time) -> flows into cutlass.Constexpr[int] at every use.

        Input requirements: none (a property). Reads only baked configure state via ``getattr`` so it is
        safe before ``configure_a2a`` has run (the parent ``__init__`` calls the warpgroup hooks first).

        Returns:
            ``int`` >= 7. The optional fields are appended in a FIXED order (b_j then b_b) so an index
            derived from this count is stable; :attr:`_DECOUPLED_BSTATE_B_INDEX` is the only supported
            way to address the b-mode slot.
        """
        base = self._DECOUPLED_BSTATE_FIELDS_BASE
        if getattr(self, "_a2a_route2_ni", False):
            base += 1
        if self._a2a_b_mode():
            base += 1
        return base

    @property
    def _DECOUPLED_BSTATE_B_INDEX(self):
        """Index of the pending batch's PLANE (``b_b``) inside the wide-put batch state.

        The plane slot is appended LAST, so it is ``_DECOUPLED_BSTATE_FIELDS - 1`` by construction --
        derived from the same property that sizes the SMEM allocation, so the reader and the allocator
        can never disagree about where it lives.

        Input requirements: only meaningful when :meth:`_a2a_b_mode` is True; every read site is
        ``const_expr``-gated on that, so a non-b build never evaluates the index.

        Returns:
            ``int`` -- the int32 slot index. On a non-b build this aliases the LAST real field
            (``b_dup`` or ``b_j``), which is why the gate is mandatory rather than advisory.
        """
        return self._DECOUPLED_BSTATE_FIELDS - 1

    @property
    def _DECOUPLED_META_FIELDS(self):
        """Per-slot metadata int32 count: 5 on the wide-put path (adds n_sub for count-parity), else 4.
        **+1 for route2_ni** — the N_i-stride-1 recv put is 3-coord (dst_i, dst_feat, dst_j), so the
        producer stamps an EXTRA ``dst_j`` field (the N_j grid column) the consumer indexes; D-major
        (route2_ni off) is unchanged -> byte-identical meta layout. **+1 for b-mode** — the recv gains a
        batch PLANE mode, so the producer stamps a TRAILING ``tile_b`` the consumer indexes; not b-mode
        -> byte-identical meta layout. A plain int (host/compile-time) so it flows into
        ``cutlass.Constexpr[int]`` at every use.

        Input requirements: none (a property). ``getattr``-safe before ``configure_a2a`` has run.

        Returns:
            ``int`` >= 4. Optional fields are appended in a FIXED order (dst_j then tile_b), so
            :attr:`_DECOUPLED_META_B_INDEX` derives the plane slot from this same count.
        """
        base = (
            self._DECOUPLED_META_FIELDS_WIDE
            if getattr(self, "_a2a_ib_wide", False)
            else self._DECOUPLED_META_FIELDS_BASE
        )
        if getattr(self, "_a2a_route2_ni", False):
            base += 1
        if self._a2a_b_mode():
            base += 1
        return base

    @property
    def _DECOUPLED_META_B_INDEX(self):
        """Index of the batch PLANE (``tile_b``) inside the per-slot ring metadata record.

        Appended LAST, so it is ``_DECOUPLED_META_FIELDS - 1`` -- derived from the same property that
        sizes the record, so producer and consumer index the identical slot by construction. This is the
        whole reason the plane goes last rather than beside ``dst_tok``: an interior insert would shift
        every field after it and silently re-point a consumer built from the other count.

        Input requirements: only meaningful when :meth:`_a2a_b_mode` is True; every read/write site is
        ``const_expr``-gated on that.

        Returns:
            ``int`` -- the int32 field index within one ring slot's record.
        """
        return self._DECOUPLED_META_FIELDS - 1

    # A free named-barrier id (> the staged kernel's epilogue/stats ids) for the consumer WG's
    # internal sync; disjoint so there is no cross-warpgroup aliasing.
    _DECOUPLED_CONSUMER_BARRIER_ID = 9

    @property
    def _DECOUPLED_CONSUMER_WARPS(self):
        """Total consumer warps = 4 * consumer_warpgroups (the SIMT GMEM->peer drain is parallelized
        across all of them). Used for the empty[s] arrive count + the per-row drain stride.

        #21 WS-B 3-WG wide-tile (port of GemmSm90A2A e6654c8): at tile_N=256 the drain rides the
        PRODUCER WG's spare warps (its 4 warps minus the ``num_ab_load_warps`` TMA-load warp(s) == 3),
        NOT a dedicated 4th warpgroup -> the 4-warp drain FOLDS onto 3."""
        if self._drain_on_producer_wg():
            return 4 - int(self.num_ab_load_warps)
        return 4 * int(getattr(self, "_a2a_consumer_warpgroups", 1))

    def _num_extra_warpgroups(self) -> int:
        """Add ONE dedicated consumer warpgroup ONLY on the LIVE decoupled A2A path (flag-gated; getattr
        because the parent __init__ calls this hook BEFORE the subclass sets the flags — the launch
        recomputes threads_per_cta in _setup_attributes once the flags are set).

        #57: keys off ``_decoupled_active()`` so the all-P2P collapse (ib_drain + no IB peer) ELIDES the
        consumer WG -> threads_per_cta byte-identical to the coupled store.
        #21: the 3-WG wide-tile drain rides the producer WG's spare warps -> NO dedicated extra WG (0)."""
        if const_expr(getattr(self, "_a2a_enabled", False) and self._decoupled_active()):
            if self._use_3wg_drain():
                return 0
            return int(getattr(self, "_a2a_consumer_warpgroups", 1))
        return 0

    def _use_3wg_drain(self) -> bool:
        """#21 (port of GemmSm90A2A._use_3wg_drain, e6654c8): the shipped 4-WG (512-thread) decoupled-
        drain layout's uniform register cap (65536/512 = 128) is below the tile_N=256 (Dloc>=128)
        put_warp/WGMMA ~154-reg need -> ptxas C7602 register wall (PROVEN; no nvvm.maxnreg/setmaxnreg
        escapes the 512-thread launch reservation). Drop the dedicated 4th warpgroup and run the drain
        on the PRODUCER WG's spare warps 9-11 -> 3-WG/384-thread -> uniform cap 65536/384 = 170 >= 154
        (fits, NO register tricks). const_expr host-computable. Gated on the EXACT wall trigger
        ``tile_N==256 AND has_ib_peers`` (+ decoupled) so EVERY other config const_expr-ELIDES the 3-WG
        code -> byte-identical compiled kernel/PTX to pre-#21 (tile_N<=128 folds to the 4-WG path)."""
        return bool(
            getattr(self, "_a2a_enabled", False)
            and getattr(self, "_a2a_decoupled", False)
            and getattr(self, "_a2a_has_ib_peers", True)
            and int(self.cta_tile_shape_mnk[1]) >= 256
        )

    def _drain_on_producer_wg(self) -> bool:
        return self._use_3wg_drain()

    def _drain_first_warp(self) -> int:
        """First warp index of the drain (consumer) role. 4-WG: the dedicated extra warpgroup's first
        warp ``(mma_warp_groups+1)*4`` (== 12). 3-WG (_use_3wg_drain): the producer WG's first SPARE
        warp ``mma_warp_groups*4 + num_ab_load_warps`` (== 9), after the sole TMA-load warp 8. The
        drain loops anchor their warp-gate + wid math on this."""
        if self._use_3wg_drain():
            return self.mma_warp_groups * 4 + int(self.num_ab_load_warps)
        return (self.mma_warp_groups + 1) * 4

    def _extra_wg_reg_adjust(self, n_extra_wg: int) -> None:
        """UNIFORM-allocate the decoupled-drain extra-WG layout (default non-A2A path untouched).

        Drops the fragile Hopper warp-specialized setmaxnreg realloc for the decoupled-drain config
        (the MMA .inc target can be unreachable from the launch baseline -> deadlock; see
        GemmSm90A2A._extra_wg_reg_adjust + project memory). The front parent kernel gates BOTH its
        load setmaxregister_decrease and its MMA setmaxregister_increase on _skip_warpgroup_reg_realloc
        (the two hooks added alongside this work), so setting it here elides the pair -> ptxas uses one
        uniform allocation. No-op (byte-identical) on every non-decoupled path."""
        if const_expr(n_extra_wg <= 0):
            return
        if const_expr(not (getattr(self, "_a2a_enabled", False) and self._decoupled_active())):
            return
        self._skip_warpgroup_reg_realloc = True

    def _extra_smem_struct(self):
        """Decoupled ring SMEM: full/empty mbarriers + per-slot metadata + producer counter.

        Flag-gated (_a2a_decoupled); else the parent default (0-size, byte-identical). The GMEM ring
        itself lives in GMEM (passed via epi_params); only the rd-deep mbarriers + meta + pcount are
        SMEM. At the default rd=2 this is ~76 B (absorbed by alignment), so the parent SharedStorage
        footprint is effectively unchanged on the coupled path."""
        if const_expr(not (getattr(self, "_a2a_enabled", False) and self._decoupled_active())):
            return super()._extra_smem_struct()
        rd: cutlass.Constexpr[int] = self._a2a_ring_depth
        nfields: cutlass.Constexpr[int] = self._DECOUPLED_META_FIELDS

        if const_expr(getattr(self, "_a2a_ib_wide", False)):
            # WIDE-PUT: the producer accumulates W subtiles per ring slot, so it carries a PERSISTENT
            # per-CTA BATCH STATE across copy_fn calls (single-writer, elected producer lane): bstate =
            #   [0]=bc (batch counter; slot=bc%rd, full-phase=(bc//rd)&1)
            #   [1]=b_w (subtiles in the pending batch; 0 => none; == n_sub at flush)
            #   [2]=b_base (the pending batch's first-subtile dst_tok, epi_m units)
            #   [3]=b_peer (pending batch peer; -1 => P2P skip batch, no drain)
            #   [4]=b_feat (pending batch dst_feat, epi_n units)
            #   [5]=b_run (pending batch total VALID token run; the wide put length)
            #   [6]=b_dup (run_i clamped-padding re-stores absorbed for count-parity)
            #   then, OPTIONALLY and in this order, [7]=b_j (route2_ni's pinned N_j column) and
            #   [+1]=b_b (b-mode's pinned recv PLANE) -- see _DECOUPLED_BSTATE_B_INDEX. Both are
            #   MERGE KEYS: a wide put is one contiguous run along the recv stride-1 axis, and
            #   neither a different N_j column nor a different plane is adjacent to the current run.
            # 7-9 Int32/CTA SMEM (frugal, O(1) in N_token -- it is per-CTA state, not per-token).
            # pcount is unused on this path (kept for the base-path init) — the batch machinery
            # tracks everything via bstate.
            nb: cutlass.Constexpr[int] = self._DECOUPLED_BSTATE_FIELDS

            @cute.struct
            class DecoupledSmem:
                full: cute.struct.MemRange[cutlass.Int64, rd]
                empty: cute.struct.MemRange[cutlass.Int64, rd]
                meta: cute.struct.MemRange[cutlass.Int32, rd * nfields]
                pcount: cute.struct.MemRange[cutlass.Int32, 1]
                bstate: cute.struct.MemRange[cutlass.Int32, nb]

            return DecoupledSmem

        @cute.struct
        class DecoupledSmem:
            full: cute.struct.MemRange[cutlass.Int64, rd]
            empty: cute.struct.MemRange[cutlass.Int64, rd]
            meta: cute.struct.MemRange[cutlass.Int32, rd * nfields]
            pcount: cute.struct.MemRange[cutlass.Int32, 1]

        return DecoupledSmem

    def _init_extra_smem(self, storage, warp_idx) -> None:
        """Init the ring's full/empty mbarriers + zero the producer counter (pre-warp-split window;
        the cluster-wide pipeline_init_wait orders it before any producer/consumer use). PLAIN method
        (the storage struct cannot cross a @cute.jit boundary): extract the SMEM pointers and hand
        them to the @cute.jit _init_decoupled_mbarriers."""
        if const_expr(not (self._a2a_enabled and self._decoupled_active())):
            return
        # WIDE-PUT: also hand the per-CTA batch state so the init zeroes bc + b_w (a fresh CTA starts
        # with no pending batch). None on the per-subtile path (byte-identical).
        bstate = None
        if const_expr(getattr(self, "_a2a_ib_wide", False)):
            bstate = storage.decoupled.bstate.get_tensor((self._DECOUPLED_BSTATE_FIELDS,))
        self._init_decoupled_mbarriers(
            storage.decoupled.full.data_ptr(),
            storage.decoupled.empty.data_ptr(),
            storage.decoupled.pcount.get_tensor((1,)),
            warp_idx,
            bstate,
        )

    @cute.jit
    def _init_decoupled_mbarriers(self, full_ptr, empty_ptr, pcount, warp_idx, bstate=None) -> None:
        rd: cutlass.Constexpr[int] = self._a2a_ring_depth
        # full[s]: the SINGLE producer warp (32 threads) arrives -> count 32.
        # empty[s]: ALL consumer-warpgroup threads do the SIMT put (warpgroup-parallel) -> count
        #   consumer_warps*32. The producer waits (does not arrive) on empty; the consumer waits on full.
        full_cnt: cutlass.Constexpr[int] = 32
        empty_cnt: cutlass.Constexpr[int] = self._DECOUPLED_CONSUMER_WARPS * 32
        lane = cute.arch.lane_idx()
        if warp_idx == Int32(0) and lane == Int32(0):
            for s in cutlass.range_constexpr(rd):
                cute.arch.mbarrier_init(full_ptr + s, full_cnt)
                cute.arch.mbarrier_init(empty_ptr + s, empty_cnt)
            pcount[0] = Int32(0)
            if const_expr(bstate is not None):
                for bi in cutlass.range_constexpr(self._DECOUPLED_BSTATE_FIELDS):
                    bstate[bi] = Int32(0)
        cute.arch.mbarrier_init_fence()

    def _compute_stages(self, *args, **kwargs):
        """Reserve the decoupled ring's mbarrier/meta SMEM before the parent sizes ab_stage.

        DEFENSE-IN-DEPTH (FIRST-PRINCIPLE: a VALID CTA tile must NEVER overflow SMEM at launch).
        Reserve the decoupled struct's WORST-CASE ALIGNED footprint at rd >= 2 (NOT just rd>2). The
        raw struct is only ~60 B at rd=2, but the SharedStorage allocator inserts it right before the
        1024-B-aligned ``sA`` field, so its true cost rounds UP to a full ``buffer_align_bytes`` (1024)
        granule. On the tightest tile — (128,32) Dloc=16, where ab_stage fills SMEM to the ~228 KB
        sm_90 cap with NO slack — that +1024 tips the launch over (observed 233472 > 232448). At rd=2
        the earlier ``> 2`` gate reserved 0 on the assumption the struct is "absorbed by alignment";
        that is FALSE at the tightest tile. Reserving the aligned cost forces the parent to leave room
        -> ab_stage drops by EXACTLY 1 (frees a full ~20 KB stage -> large headroom) ONLY on tiles too
        tight to absorb it, and is byte-identical on tiles that already have >=1024 B of slack. The
        default-off (no ib_drain / no decoupled) path is untouched (gate is False) -> byte-identical.
        arg[7] is the SMEM capacity."""
        if const_expr(
            getattr(self, "_a2a_enabled", False)
            and self._decoupled_active()
            and self._a2a_ring_depth >= 2
        ):
            align = int(getattr(self, "buffer_align_bytes", 1024))
            raw = int(
                self._extra_smem_struct().size_in_bytes()
            )  # rd*16 + rd*nfields*4 + 4, aligned
            reserve = ((raw + align - 1) // align) * align  # >= the struct's true aligned SMEM cost
            args = list(args)
            args[7] = args[7] - reserve
            args = tuple(args)
        return super()._compute_stages(*args, **kwargs)

    def configure_a2a_sharded(
        self,
        device_mesh,
        placements,
        *,
        pe_map=None,
        rows_per_peer=None,
        dynamic=False,
        decoupled=False,
        ring_depth=None,
        consumer_warpgroups=None,
        ib_drain=False,
        ib_wide=False,
        ib_wide_batch=None,
        ib_wide_nbi=False,
    ):
        """Generic 1D/2D DTensor token-sharding entry for the front-A2A D-major postact store.

        Parses a torch ``DeviceMesh`` + DTensor ``placements`` into the cp geometry (via
        :class:`fold_cp_ops.distributed.pe_map.PeMap`) and bakes the flat-cp scalars + my local token
        count. Subsumes the 1-D :meth:`configure_a2a` as the ``cp_axis_sizes == (cp,)`` special
        case (byte-identical recv).

        The FRONT A2A scatters by the FEATURE/D axis, so — unlike the back store — the per-CTA
        peer-select (by postact N-tile) and the recv-index (col-offset = my_cp_rank*rows_per_peer,
        D-row within the half/slice) are INDEPENDENT of whether the token grid is sharded on one
        axis (1-D ``cp``) or two (2-D ``(cp0, cp1)``): the recv buffer is ``(2*Dloc, M_full)`` with
        ``rows_per_peer`` = my local token count (= ``prod`` of the per-axis token extents), and the
        2-D ``(i, j)`` token decomposition only enters the HOST-side unpack, not the kernel store.
        So this entry just flattens the cp axes to ``cp`` / ``my_cp_rank`` / ``pe_table`` and forwards
        to :meth:`configure_a2a` (the recv-index 2-D-invariance is empirically confirmed in the
        Part-A validation ladder, rung 3, and inherited by the D-major store unchanged).

        Parameters
        ----------
        device_mesh : torch.distributed.device_mesh.DeviceMesh
            The cp mesh (1-D ``(cp,)`` or 2-D ``(cp0, cp1)``).
        placements : Sequence
            One placement per mesh dim; the ``Shard`` dims are the cp (token) axes. The feature
            dim (3) MUST NOT be sharded (rejected).
        pe_map : PeMap, optional
            Prebuilt PeMap; if ``None`` it is built from (device_mesh, placements).
        rows_per_peer : int
            My local token count (``B * N_i_loc * N_j_loc``) — the postact M extent.
        """
        if not HAS_NVSHMEM:
            raise RuntimeError("DualGatedGemmDistSm90.configure_a2a_sharded requires nvshmem4py.")
        from fold_cp_ops.distributed.dtensor_adapter import validate_trimul_sharding
        from fold_cp_ops.distributed.pe_map import PeMap

        validate_trimul_sharding(placements, len(placements))
        if pe_map is None:
            pe_map = PeMap.from_mesh_placements(device_mesh, placements)
        if rows_per_peer is None:
            raise ValueError("configure_a2a_sharded requires rows_per_peer (my local token count).")
        cp = int(pe_map.cp)
        pe_table = tuple(int(x) for x in pe_map.cp_pe_table.tolist())
        self.configure_a2a(
            cp,
            int(pe_map.my_cp_rank),
            int(rows_per_peer),
            pe_table,
            dynamic=dynamic,
            decoupled=decoupled,
            ring_depth=ring_depth,
            consumer_warpgroups=consumer_warpgroups,
            ib_drain=ib_drain,
            ib_wide=ib_wide,
            ib_wide_batch=ib_wide_batch,
            ib_wide_nbi=ib_wide_nbi,
        )
        self._a2a_cp_axis_sizes = tuple(int(s) for s in pe_map.cp_axis_sizes)

    def _a2a_b_mode(self) -> bool:
        """Does the peer recv carry an explicit token-BATCH mode? (host + kernel agree, pure fn)

        Purpose
            ONE predicate for "this store addresses a batch plane", read by the peer-view build and
            by every ``copy_fn``, so the descriptor RANK and the copy COORD can never disagree.

        Semantics
            True exactly when ``configure_a2a`` was given ``b_plane=True`` (or ``b_dynamic=True``,
            which implies it). NOT implied by ``B > 1`` alone: a standalone transpose_in front with
            a baked ``B=2`` writes the old slot-major-then-plane recv and is correct doing so --
            only the composite / route2 BACK READ needs the plane moved out. False for every plain
            build and for every front whose consumer did not ask, which is what keeps those
            byte-identical: the whole b-mode tree is ``const_expr``-pruned there.

            When True the recv gains a leading BATCH mode INSIDE the feature mode and OUTSIDE the
            cp slot -- ``(2*Dloc, B, ...)`` -- because that is the only placement for which the back
            operand's ``L = (Dloc, B)`` collapses to one stride: ``Dloc``'s stride must be exactly
            ``B x`` ``B``'s. With the cp slot between them (the natural store order) no flat L mode
            exists at any ``cp > 1``, so the store, not the reader, is what has to move.

        Returns
            ``bool`` -- a plain Python bool, usable in ``cutlass.const_expr``.
        """
        return bool(getattr(self, "_a2a_b_plane", False))

    def _a2a_should_clamp(self) -> bool:
        """AUTO-CLAMP decision — a pure fn of the baked cp/postact-tile state (host + kernel agree).

        ``True`` => build the per-rank column-BLOCK clamp atom + a RELATIVE ``dst_tok`` (the partial
        last M-tile's overshoot past ``rows_per_peer`` is dropped by the descriptor's high-end OOB clamp
        on the recv stride-1 TOKEN axis). Taken AUTOMATICALLY whenever ``rows_per_peer`` is not a CTA
        ``tile_M`` multiple (a partial last M-tile), OR forced by the explicit ``partial_token_clamp``
        override (back-compat; a no-op if the auto path already covers it). The descriptor AUTO-CLAMP is
        valid for ANY TMA-S2G egress — the coupled base store AND the ib_drain differential P2P arm (it
        reuses the coupled peer atoms); the IB SIMT put has no descriptor and run-clamps in software via
        ``meta[3]`` (§2), NOT via this flag, so the clamp decision is the SAME pure fn of ``rows_per_peer``
        for the coupled and decoupled paths. Aligned ``rows_per_peer`` (rem==0, no override) => ``False``
        => the full-``M_full`` absolute-dst atom (byte-identical to pre-relax)."""
        tile_m = int(self.cta_tile_shape_postact_mn[0])
        return (self._a2a_rows_per_peer % tile_m != 0) or bool(
            getattr(self, "_a2a_partial_token_clamp", False)
        )

    # ------------------------------------------------------------------
    # Host-side: attach the cp peer postact atoms to the EpilogueParams (runs in __call__ via
    # self.epi_to_underlying_arguments, BEFORE the kernel launch).
    # ------------------------------------------------------------------
    def epi_to_underlying_arguments(self, args, *, loc=None, ip=None):
        p = super().epi_to_underlying_arguments(args, loc=loc, ip=ip)
        if not self._a2a_enabled:
            return p
        # FRONT D-major reshard validity.  args.mPostAct is the M-major postact logical view
        # (M, 2D, L) -> we store into THIS rank's symmetric D-major recv (2*Dloc, M_full), passed
        # via the peer atoms built from `recv`.  The recv is built from args.mPostAct's MEMORY: with
        # transpose_out the host allocates (2D, M) and passes args.mPostAct = (M,2D) M-major; the recv
        # is that SAME (2D, M) buffer reinterpreted as the D-major (2*Dloc, M_full) symmetric tensor.
        # The peer-atom recv view is supplied separately (the symmetric recv tensor) — see configure;
        # here we read it off the recv attribute the host set on args.
        recv = getattr(args, "recv", None)
        if recv is None:
            raise ValueError(
                "front-A2A staged requires args.recv (the symmetric D-major recv (2*Dloc, M_full))."
            )
        # A CTA postact N-tile must lie wholly within ONE D-slice (Dloc % tile_n_postact == 0) AND
        # within ONE half (D % tile_n_postact == 0).  recv shape[0] = 2*Dloc -> Dloc = shape[0]//2.
        two_dloc = int(recv.shape[0])
        assert two_dloc % 2 == 0, f"recv shape[0]={two_dloc} (=2*Dloc) must be even."
        dloc = two_dloc // 2
        tile_n_postact = int(self.cta_tile_shape_postact_mn[1])
        if dloc % tile_n_postact != 0:
            raise ValueError(
                f"front-A2A requires Dloc (={dloc}) % postact tile_N (={tile_n_postact}) == 0 "
                "(a CTA N-tile must lie within ONE D-slice for the per-CTA single-peer select)."
            )
        # MECHANISM the clamp addresses: the token (M) recv column dst is subtile-granular (dst_tok_sub =
        # [my_col_off +] M_tile*m_sub_per_tile + sub_m); a partial last M-tile (rows_per_peer % tile_M != 0)
        # makes the GEMM emit a fractional tile whose UNCLAMPED store coords overshoot the peer's column
        # block -> silent cross-block corruption (validated: M=64 < tile_M=128 corrupts ALL blocks). The
        # auto-clamp below drops that overshoot, so rows_per_peer need NOT be a tile_M multiple (16-B only).
        # AUTO-CLAMP (default): a non-multiple rows_per_peer (a partial last M-tile) AUTOMATICALLY takes
        # the per-rank column-BLOCK clamp atom (its overshoot dropped by the descriptor's high-end OOB
        # clamp on the recv stride-1 TOKEN axis); an aligned rows_per_peer keeps the full-M_full
        # absolute-dst atom (byte-identical). This now fires for the COUPLED store AND the decoupled/
        # ib_drain differential P2P arm (which reuses these atoms) AND is mirrored in software by the IB
        # SIMT run-clamp (§2/meta[3]) -> the old "rpp%tile_M!=0 -> raise on the decoupled drain" reject is
        # GONE: an off-grid rows_per_peer runs FUSED, the sole surviving shape check being the 16-B floor
        # below. partial_token_clamp still forces the clamp (back-compat). See _a2a_should_clamp.
        use_clamp = self._a2a_should_clamp()
        # 16-B FLOOR (the SOLE surviving shape check besides Dloc%tile_n): both egress arms transfer at the
        # epi_m (subtile/box) grain; the partial-last subtile carries rows_per_peer % epi_m valid tokens
        # (the P2P descriptor OOB-clamps to it; the IB SIMT put run-clamps to it via meta[3]). That run
        # must be >= 16 B = 8 bf16 (the TMA stride-1 / warp-put minimum). rows_per_peer % epi_m in {0} or
        # {>=8, multiple of 8} for any valid N (N%8==0 && N%cp==0 -> rows_per_peer%128 in {0} or >=8;
        # debug/verify_rpp_unreachable.py, 4608 combos, 0 violations) -> a DEFENSIVE guard, never fires for
        # an in-contract shape; a 1..7-token tail (< 16 B) raises only on a hand-crafted off-contract
        # rows_per_peer. NOTE %epi_m (the drain grain), NOT %tile_M: at tile_M=256 (epi_m=128) a tile-tail
        # of e.g. 132 hides a 4-token subtile-tail that %tile_M would miss.
        # ====== mod-128 DESTINATION-PHASE LAW — PERFORMANCE, strictly ABOVE this 16-B LEGALITY floor ===
        # The guard below (and every other alignment comment in this file) reasons only to 16 B. That is
        # the TMA/put LEGALITY minimum and says NOTHING about speed. MEASURED (cp=8, all 8
        # ranks, single-variable), the peer-store penalty is a function of the DESTINATION BYTE ADDRESS
        # mod 128: ~+4 % at 64-mod-128, ~+8-11 % at 32-mod-128, ~+18-37 % at 16-mod-32.
        #
        # For THIS store the recv is (2*Dloc, M_full) with the TOKEN axis stride-1, so rank r's block
        # begins at `r * rows_per_peer * 2` bytes and the phase is entirely `r*rpp*2 (mod 128)`. The
        # alignment target for FULL speed is `rpp % 64 == 0`, i.e. 32x tighter than the 16-B floor here.
        # At cp=8 (one node -- the most common production mesh) HALF of all legal N land half the ranks
        # at 16-mod-32: the slowest rank then takes 0.5612 ms against 0.3766 ms for a phase-0 rank AT
        # THE SAME SHAPE (1.49x), and because the front A2A is barrier-coupled the slowest rank sets the
        # pace for the whole job.
        #
        # EXCLUDED BY MEASUREMENT, so do not re-derive them: the CLAMP (forced vs not at an identical
        # shape = 1.0038-1.0076x on all 8 ranks) and the TAIL LENGTH (8-token and 72-token tails cost the
        # same at matched phase). Only the byte phase matters.
        #
        # NOT FIXED HERE, deliberately. Padding to `rpp_stride = ceil(rpp/tile_M)*tile_M` costs
        # `4*D*pad` bytes -- O(1) in N_token (8 KiB at pad=16, D=256, at EVERY N from 1032 to 8200), so
        # memory frugality is NOT the blocker. The blocker is the downstream ZERO-COPY CONTRACT: the
        # plain D-major recv's value is that `cp*rpp == B*N*N` exactly, which makes the back operand a
        # free view (`reshard.py:288`, `fused_trimul.py:1437-1438`). Any pad makes the global-i axis a
        # 2-mode composite `(cp, N_i_loc)` with outer stride `rpp_stride`, and a `.contiguous()` there
        # would be an O(Dloc*N^2) I/O-order copy -- exactly the tax the frugality rule forbids. The
        # machinery to consume the composite exists (`gemm_sm90_a2a._composite_remap` +
        # `_pe_aligned_tiling`) but only for 1-D + B==1 + back_store="pe_aligned" + incoming, so the fix
        # is: rpp_stride here (~5 lines) + the recv sizing + GENERALISING that back read to outgoing,
        # B>1 and 2-D. A smaller pad buys nothing: any pad at all breaks the reshape, so the choice is
        # binary. See docs/migration_failing_tests.md §6.2. Measured every run by
        # tests/distributed/test_dual_gated_gemm_a2a.py::test_front_partial_token_clamp_smooth,
        # whose 1.25x bar must NEVER be widened.
        # ==============================================================================================
        epi_m_floor = int(p.epi_tile_mPostAct[0])
        rpp_rem_epi = self._a2a_rows_per_peer % epi_m_floor
        if 1 <= rpp_rem_epi <= 7:
            raise ValueError(
                f"front-A2A: rows_per_peer (={self._a2a_rows_per_peer}) % epi_m (={epi_m_floor}) = "
                f"{rpp_rem_epi} in 1..7 -> the partial-tail token run is {rpp_rem_epi * 2} B < 16 B (the "
                "TMA stride-1 / warp-put minimum) -> corruption. UNREACHABLE from a valid N_token "
                "(rows_per_peer%128 is 0 or >=8 for N%8==0 && N%cp==0); pass N_token%8==0."
            )
        # §10 R1 FACTORING — the LAYOUT (transpose_in / route2_ni) axis is ⊥ the TRANSPORT
        # (ib_drain / ib_wide) axis. Compute the recv VIEW + peer-store target (atom_recv) ONCE here,
        # branching ONLY on the layout, then run the SAME transport block (is_p2p null-proof + peer atoms
        # + decoupled ring) over it below. The early-return that used to FUSE route2_ni with the
        # coupled-only path is GONE: route2_ni now composes with ib_drain/ib_wide (the transport is
        # layout-agnostic). route2_ni WITHOUT ib_drain (decoupled off) => the ring block below is elided
        # => `atom_recv`-only return == today's coupled route2_ni store (all decoupled_* fields default
        # None, so byte-identical).
        route2_ni: cutlass.Constexpr[bool] = bool(getattr(self, "_a2a_route2_ni", False))
        if const_expr(route2_ni):
            # ROUTE-2 (A) N_i-STRIDE-1 recv: 3-D (2*Dloc, N_i=cp*Xg_pad, N_j) with N_i STRIDE-1. Present
            # (N_i, 2*Dloc, N_j) [box modes (N_i=i_loc, feature) FIRST; N_j the untiled grid mode] so the
            # postact SMEM box (epi_m=i_loc walk, epi_n=feature) maps to the stride-1 N_i innermost + the
            # feature row. build_peer_store_atoms is 3-D-generic (tiles the first 2 modes, leaves N_j a
            # grid mode). NO token clamp: the padded i-shard makes each i_loc block a whole BLK_M count
            # (zero tail); the copy_fn positions global-i by shard. Requires transpose_in (i_loc inner).
            if not const_expr(getattr(self, "_transpose_in", False)):
                raise ValueError("_a2a_route2_ni requires transpose_in (the i_loc-inner walk).")
            if const_expr(self._a2a_b_mode()):
                # B-MODE: the recv is 4-D (2*Dloc, B, N_i, N_j) -- the batch plane sits INSIDE the
                # feature and OUTSIDE both token axes (see _a2a_b_mode for why nothing else works).
                # Present (N_i, 2*Dloc, N_j, B): the box modes stay first and B becomes a trailing
                # untiled grid mode, so shape[2] is still the PER-PLANE global-j extent and every
                # existing derivation off it (n_x, the cp1 j-base) is unchanged.
                atom_recv = cute.make_tensor(
                    recv.iterator, cute.select(recv.layout, mode=[2, 0, 3, 1])
                )
            else:
                atom_recv = cute.make_tensor(
                    recv.iterator, cute.select(recv.layout, mode=[1, 0, 2])
                )
        elif const_expr(self._a2a_b_mode()):
            # COMPOSITE-K B-MODE: the recv is 3-D (2*Dloc, B, cp*rpp_b) -- per-plane blocks of cp
            # peer slots, batch OUTSIDE the slot. Present (cp*rpp_b, 2*Dloc, B) so `shape[0] // cp`
            # is the PER-PLANE per-peer token extent the copy_fn needs and B is a trailing grid mode.
            if use_clamp:
                raise ValueError(
                    "front-A2A b-mode store cannot combine with the partial-token clamp: the clamp "
                    "atom is a CONTIGUOUS (rows_per_peer, 2*Dloc) column block, and a batch plane "
                    "makes my block cp-strided. Reachable only from a rows_per_peer that is not a "
                    "tile_M multiple, which the transpose_in pad rules out (rpp = B*Yg*n_x*BLK_M)."
                )
            atom_recv = cute.make_tensor(recv.iterator, cute.select(recv.layout, mode=[2, 0, 1]))
        else:
            # Build the cp peer S2G atoms over MY local symmetric D-major recv buffer.  REUSE the
            # postact's OWN epi smem layout + epi_tile (the M-major STSM box gotchas inherited) — the
            # SMEM box is (epi_m=token, epi_n=feature).  The recv (2*Dloc, M_full) is permuted to
            # (M_full, 2*Dloc) = (token, feature) so the box modes (token, feature) come FIRST and the
            # token (M_full) stride-1 leading dim matches the M-major SMEM box (gotcha #1).
            recv_tok_feat = cute.make_tensor(recv.iterator, cute.select(recv.layout, mode=[1, 0]))
            # AUTO-CLAMP: build the peer atoms over MY column BLOCK (rows_per_peer, 2*Dloc) — a token-axis
            # slice of recv_tok_feat at row offset my_cp_rank*rows_per_peer — so the descriptor's TOKEN
            # (stride-1) bound is rows_per_peer (NOT the full M_full). The partial-last M-tile then
            # overshoots PAST rows_per_peer -> dropped by the high-end OOB clamp on the token axis, instead
            # of bleeding into the NEXT rank's columns. The store's dst_tok becomes RELATIVE to my block
            # (my_col_off removed). use_clamp False -> full recv (absolute dst).
            atom_recv = recv_tok_feat
            if use_clamp:
                two_dloc_c = int(
                    recv.shape[0]
                )  # (2*Dloc) = recv mode-0 -> STATIC even under dynamic
                if self._a2a_dynamic:
                    # DYNAMIC clamp: derive rows_per_peer from the RUNTIME recv M_full (recv_tok_feat
                    # mode-0 = M_full; the static compile anchor self._a2a_rows_per_peer differs at other
                    # token counts) so the descriptor's TOKEN (stride-1) bound + my column offset track the
                    # runtime token count -> ONE compile serves many token counts (mirror copy_fn :731).
                    rpp = recv_tok_feat.shape[0] // self._a2a_cp
                    my_off = self._a2a_my_cp_rank * rpp
                else:
                    rpp = int(self._a2a_rows_per_peer)
                    my_off = int(self._a2a_my_cp_rank) * rpp
                # token (M_full) is recv_tok_feat mode-0 (stride-1); feature is mode-1 (stride M_full).
                # Slice to (rpp, 2*Dloc) at token offset my_off: shift the iterator + shrink the extent.
                blk_layout = cute.make_layout((rpp, two_dloc_c), stride=recv_tok_feat.layout.stride)
                atom_recv = cute.make_tensor(recv_tok_feat.iterator + my_off, blk_layout)
        # IB-DRAIN atom-build safety: nvshmem_ptr(addr, pe) returns NULL for a non-P2P (IB) peer, so
        # building a coupled S2G TMA descriptor over that null base can fault cuTensorMapEncodeTiled at
        # HOST build. The differential producer NEVER issues a coupled store to an IB peer (const_expr
        # is_p2p[r] guards it), so substitute MY OWN PE (pe_table[my_cp_rank]) for every IB slot -> a
        # valid (self-based) descriptor that is never used. pe_table_dev (the runtime put routing) keeps
        # the REAL peer PEs. Non-ib_drain / all-P2P -> the real pe_table verbatim (byte-identical).
        atom_pe_table = self._a2a_pe_table
        if self._a2a_ib_drain and self._a2a_is_p2p is not None:
            my_pe = int(self._a2a_pe_table[self._a2a_my_cp_rank])
            atom_pe_table = tuple(
                int(pe) if self._a2a_is_p2p[r] else my_pe for r, pe in enumerate(self._a2a_pe_table)
            )
        atoms, tensors = build_peer_store_atoms(
            atom_recv,
            self._a2a_cp,
            atom_pe_table,
            p.epi_mPostAct_smem_layout_staged,
            p.epi_tile_mPostAct,
        )
        # ---- DECOUPLED SIMT-ring drain plumbing ----
        # The MMA warpgroup's epilogue stages each postact SMEM subtile to a bounded LOCAL-GMEM ring
        # (producer-TMA, retargeted at the ring slot); the dedicated consumer warpgroup drains the
        # ring -> peer recv via a per-row put_nbi_warp routed by RUNTIME PE. Build the producer-TMA
        # ring-store atom + forward the local recv view (token,feature) + the device PE table. All
        # None on the coupled path (byte-identical).
        ring = getattr(args, "ring", None)
        ring_store_atom, ring_store_tensor = None, None
        recv_local, pe_table_dev = None, None
        if const_expr(self._a2a_decoupled):
            if ring is None:
                raise ValueError(
                    "front-A2A decoupled requires args.ring (the bounded LOCAL-GMEM staging ring "
                    "(grid_CTAs, ring_depth, epi_n_feat, epi_m_token))."
                )
            ring_store_atom, ring_store_tensor = self._build_postact_ring_store_atom(
                ring, p.epi_mPostAct_smem_layout_staged, p.epi_tile_mPostAct
            )
            # recv_local is MY column BLOCK (atom_recv) so the consumer's flat_divide + RELATIVE dst_tok
            # (producer §3b) land in MY block — mirroring the P2P arm's clamp atom. use_clamp False =>
            # atom_recv == recv_tok_feat (full recv, absolute dst) => byte-identical to pre-relax.
            recv_local = (
                atom_recv  # THIS rank's recv put dst (clamp block under auto-clamp, else full)
            )
            pe_table_dev = getattr(args, "pe_table_dev", None)
            if pe_table_dev is None:
                raise ValueError(
                    "front-A2A decoupled requires args.pe_table_dev (the (cp,) int32 device PE table "
                    "for the runtime put-by-PE routing)."
                )
        return dataclasses.replace(
            p,
            postact_peer_atoms=atoms,
            postact_peer_tensors=tensors,
            decoupled_ring=ring,
            ring_store_atom=ring_store_atom,
            ring_store_tensor=ring_store_tensor,
            recv_local=recv_local,
            pe_table_dev=pe_table_dev,
        )

    def _build_postact_ring_store_atom(self, ring, epi_smem_layout_staged, epi_tile):
        """Build the producer-TMA S2G atom: SMEM postact box -> LOCAL-GMEM ring slot (front analogue
        of GemmSm90A2A._build_ring_store_atom). The SAME async TMA-S2G the coupled store uses, just a
        different (LOCAL-GMEM) destination -> the MMA warp's ring-write is O(1) instead of an O(tile)
        SIMT copy. ``ring`` is (grid_CTAs, ring_depth, epi_n_feat, epi_m_token) — note the front
        allocates the ring TOKEN-innermost (epi_m last) so the slot box's stride-1 is the TOKEN, the
        SAME stride-1 the D-major recv has (token = recv col); the consumer's per-feature-row put is
        then a contiguous token run on BOTH ends. We permute the box modes (epi_m=token, epi_n=feature)
        FIRST -> (epi_m, epi_n, ring_depth, grid_CTAs) and build a CopyBulkTensorTileS2GOp whose
        descriptor covers the whole ring; the kernel selects the live (slot, cta) per store.
        Returns (atom, tensor); tensor is the box-permuted ring view the copy_fn flat_divides."""
        epi_smem_layout = cute.slice_(epi_smem_layout_staged, (None, None, 0))
        # (grid_CTAs, ring_depth, epi_n_feat, epi_m_token) -> (epi_m_token, epi_n_feat, rd, grid).
        # mode order [3,2,1,0]: dim3=epi_m_token (the stride-1 axis) FIRST, then dim2=epi_n_feat.
        ring_view = cute.make_tensor(ring.iterator, cute.select(ring.layout, mode=[3, 2, 1, 0]))
        d_cta_v_layout = cute.composition(cute.make_identity_layout(ring_view.shape), epi_tile)
        op = cpasync.CopyBulkTensorTileS2GOp()
        atom, tensor = cpasync.make_tiled_tma_atom(op, ring_view, epi_smem_layout, d_cta_v_layout)
        return atom, tensor

    # ------------------------------------------------------------------
    # The postact-store seam override.  epi_setup_postact (parent line 291) returns
    # (tiled_copy_postact_r2s, tRS_sPostAct, copy_postact); we keep the R2S stage identical and
    # redirect ONLY copy_postact to the D-major peer store when _a2a_enabled (and NOT gate3 —
    # mPostAct3 is the Replicate gate output, not resharded; keep it local).
    # ------------------------------------------------------------------
    def epi_setup_postact(
        self,
        params,
        epi_smem_tensors,
        tiled_copy_r2s,
        tiled_copy_t2r,
        tile_coord_mnkl,
        tidx,
        epi_gate3=False,
    ):
        # flag-off OR gate3 region -> parent's local store verbatim (byte-identical).
        if const_expr((not self._a2a_enabled) or epi_gate3):
            return super().epi_setup_postact(
                params,
                epi_smem_tensors,
                tiled_copy_r2s,
                tiled_copy_t2r,
                tile_coord_mnkl,
                tidx,
                epi_gate3=epi_gate3,
            )
        # --- A2A dual postact: same R2S stage as the parent, D-major peer-store copy_postact ---
        import fold_cp_ops._internal.copy_utils as copy_utils

        name = "mPostAct"
        sPostAct = epi_smem_tensors[self._epi_smem_map[name]]
        pa_layout = self.postact_layout
        # STSM num_matrices from the postact epi-tile N-width (x2 for the glu-halved 8-wide tail at
        # postact_epi_n=8, x4 for >=16) — MUST match the parent (kernels/layernorm_dual_gated_gemm.py, `_postact_epi_n`): the
        # bare hopper sm90_get_smem_store_op HARDCODES x4, which mis-partitions an 8-wide sPostAct at
        # Dloc=8 (postact tile_n=8 -> postact_epi_n=8) -> wrong R2S -> garbage recv (rel~2, memcheck-clean
        # single-node all-P2P). At postact_epi_n>=16 (Dloc>=16) major_mode_size%16==0 -> num_matrices=4 ->
        # BYTE-IDENTICAL to the old op (the tileN store-op relaxation the parent already proved neutral).
        copy_atom_postact_r2s = copy_utils.sm90_get_smem_store_atom(
            # `arch` is not a parameter here: this package is SM90-only by decision, so the helper
            # dropped main's leading arch argument. Passing it shifts every later positional.
            self.postact_dtype,
            transpose=pa_layout.is_m_major_c(),
            major_mode_size=params.epi_tile_mPostAct[1],
        )
        tiled_copy_postact_r2s = cute.make_tiled_copy_S(copy_atom_postact_r2s, tiled_copy_r2s)
        tRS_sPostAct = tiled_copy_postact_r2s.get_slice(tidx).partition_D(sPostAct)
        if const_expr(self._decoupled_active()):
            # DECOUPLED producer: the MMA warpgroup's epilogue stages each postact subtile to the ring +
            # writes the per-slot recv-index/peer metadata + signals full[s] (the consumer warpgroup
            # drains ring->peer recv). ib_drain makes it DIFFERENTIAL: an NVLink (P2P) peer's tile takes
            # the coupled TMA-S2G store DIRECTLY (byte-identical data path) while an IB peer's tile stages
            # into the SYMMETRIC-heap ring for the blocking-put drain (the store seam const_expr-selects
            # per-peer on is_p2p). The all-P2P collapse routes here as coupled-only (never reached: it is
            # elided by _decoupled_active()==False -> the coupled store below). Ring/meta SMEM come off
            # the storage struct (passed in).
            copy_postact = self._a2a_decoupled_producer_copy_fn(
                params,
                self.cta_tile_shape_postact_mn,
                params.epi_tile_mPostAct,
                sPostAct,
                tile_coord_mnkl,
                self._epi_storage,
            )
            return tiled_copy_postact_r2s, tRS_sPostAct, copy_postact
        # D-major peer-store copy_postact: select half + peer by the N(feature)-block, S2G store at
        # the front-reshard D-major recv offset.  The cp peer atoms/tensors ride on params (attached
        # host-side in epi_to_underlying_arguments) — they cross the @jit->@kernel region as part of
        # the EpilogueParams kernel arg, so the partition below is region-local.
        copy_postact, _, _ = self._a2a_postact_peer_store_copy_fn(
            params.postact_peer_atoms,
            params.postact_peer_tensors,
            self.cta_tile_shape_postact_mn,
            params.epi_tile_mPostAct,
            sPostAct,
            tile_coord_mnkl,
        )
        return tiled_copy_postact_r2s, tRS_sPostAct, copy_postact

    def _a2a_postact_peer_store_copy_fn(
        self, atoms, tensors, tile_shape_mn, epi_tile, sPostAct, tile_coord_mnkl
    ):
        """Build copy_postact = select half + peer by the CTA N(feature)-block + D-major peer S2G
        store at the front-reshard recv offset (token-major-column / feature-major-row).

        MIRRORS the partition/copy mechanics of the proven stores (the KEY FIXES — flat_divide NOT
        zipped, group_modes box-fold, no cute.printf), with a 3-COORD copy (the recv is 2-D -> NO L
        mode, unlike the L=1 back store's 4-tuple).  Two deltas vs the stagec (token-major) front:
          1. PER-HALF: the dual width is 2D, so the postact N-tile selects the half (a|b) first, then
             the peer within the half (the stagec front had width D = one half).
          2. D-MAJOR dst: the recv is (2*Dloc, M_full) [feature row OUTER, token col stride-1]; the
             peer tensor is permuted to (M_full, 2*Dloc) = (token, feature) [token stride-1] so the
             box modes (epi_m=token, epi_n=feature) come first and match the M-major SMEM box.

        FRONT D-major reshard (module header):
          n_tiles_per_half   = D    // tile_n   (tile_n = postact tile N = BLK_N//2)
          n_tiles_per_Dslice = Dloc // tile_n
          N_tile  = tile_coord_mnkl[1]
          half    = N_tile // n_tiles_per_half                                       # 0=a, 1=b
          peer    = (N_tile %  n_tiles_per_half) // n_tiles_per_Dslice               # per-half -> peer
          dst_feat_sub = half*(Dloc/epi_n) + ((N_tile % n_tiles_per_half) % n_tiles_per_Dslice)*n_sub + sub_n
          dst_tok_sub  = my_cp_rank*(rpp/epi_m) + tile_coord_mnkl[0]*(tile_m/epi_m) + sub_m
        Returns ``(copy_fn, s0, g0)`` (the epilogue only consumes ``copy_fn``).
        """
        cp: cutlass.Constexpr[int] = self._a2a_cp
        my_cp_rank: cutlass.Constexpr[int] = self._a2a_my_cp_rank
        # B10 2-D token-axis decomposition (route2_ni store ONLY; the FEATURE peer loop keeps flat cp).
        # cp_axis_sizes is (cp,) for the flat/1-D configure and (cp0, cp1) when the route2_ni front
        # overrides it. cp0 = i-axis blocks, cp1 = j-axis blocks; my (cp0_coord, cp1_coord) row-major.
        # 1-D (cp1==1): cp0_coord==my_cp_rank, cp1_coord==0 -> the addressing collapses to the flat path.
        _cp_axes = self._a2a_cp_axis_sizes
        cp0: cutlass.Constexpr[int] = int(_cp_axes[0])
        cp1: cutlass.Constexpr[int] = int(_cp_axes[1]) if len(_cp_axes) > 1 else 1
        cp0_coord: cutlass.Constexpr[int] = my_cp_rank // cp1
        cp1_coord: cutlass.Constexpr[int] = my_cp_rank % cp1
        tile_m: cutlass.Constexpr[int] = tile_shape_mn[0]
        tile_n: cutlass.Constexpr[int] = tile_shape_mn[1]  # postact (glu-halved) tile N
        epi_m: cutlass.Constexpr[int] = epi_tile[0]
        epi_n: cutlass.Constexpr[int] = epi_tile[1]
        m_sub_per_tile: cutlass.Constexpr[int] = tile_m // epi_m
        n_sub_per_tile: cutlass.Constexpr[int] = tile_n // epi_n
        # Token-scaling recv COLUMN offset (my_cp_rank * rows_per_peer) // epi_m: baked const_expr
        # (static -> byte-identical) OR runtime from the recv's M_full (tensors[0] permuted to
        # (M_full, 2*Dloc) -> shape[0]=M_full; rows_per_peer = M_full // cp). One compile, many token counts.
        if const_expr(self._a2a_dynamic):
            # tensors[0]==atom_recv: the per-rank column-BLOCK clamp atom (shape[0]==rpp, ALREADY
            # per-peer) under auto-clamp, ELSE the full recv (shape[0]==M_full -> //cp). //cp on the
            # clamp block HALVES rows_per_peer -> the run_tokens tail-clamp drops the upper half of every
            # block (the N1000 dynamic-auto-clamp mis-deliver). Branch on the SAME should_clamp const.
            if const_expr(self._a2a_should_clamp()):
                rows_per_peer = tensors[0].shape[0]  # clamp block: already per-peer rpp
            else:
                rows_per_peer = tensors[0].shape[0] // cp  # full recv M_full // cp
            my_col_sub_off = (my_cp_rank * rows_per_peer) // epi_m  # runtime
        else:
            # B-MODE (static): _a2a_rows_per_peer counts ALL planes (B*Yg*n_x*BLK_M) but the recv's
            # cp slots stride by ONE plane's worth, so the column offset must divide the batch out.
            # Under _a2a_dynamic the same value falls out of shape[0]//cp for free -- the presented
            # peer view is (cp*rpp_b, 2*Dloc, B), i.e. its token mode is already per-plane.
            rows_per_peer: cutlass.Constexpr[int] = self._a2a_rows_per_peer // (
                int(self._token_grid_b) if self._a2a_b_mode() else 1
            )
            my_col_sub_off: cutlass.Constexpr[int] = (my_cp_rank * rows_per_peer) // epi_m
        # D-major geometry.  The per-peer recv tensors[r] are permuted to (M_full, 2*Dloc[, L=1]):
        # shape[1] = 2*Dloc holds MY D-slice's a-half (rows [0:Dloc]) + b-half (rows [Dloc:2*Dloc]).
        # The full postact feature width is D = cp*Dloc (the CTA N-tile ranges over [0, 2*D/tile_n)).
        two_dloc: cutlass.Constexpr[int] = int(tensors[0].shape[1])
        dloc: cutlass.Constexpr[int] = two_dloc // 2
        n_tiles_per_Dslice: cutlass.Constexpr[int] = (
            dloc // tile_n
        )  # postact N-tiles per peer's Dloc
        n_tiles_per_half: cutlass.Constexpr[int] = (
            cp * n_tiles_per_Dslice
        )  # = D//tile_n (D=cp*Dloc)
        dloc_sub: cutlass.Constexpr[int] = (
            dloc // epi_n
        )  # feature epi-subtiles per half (recv row blk)

        # Per-peer tma_partition (the proven recipe): FLAT_divide the peer GMEM tensor (NOT zipped);
        # group_modes(.,0,2) folds the (epi_m,epi_n) box leaving the (nt_tok, nt_feat) grid modes ->
        # g_r indexed g_r[(None, dst_tok, dst_feat)] (box=None + 2 grid coords; the recv is 2-D, so —
        # unlike the L=1 back store's 4-tuple — there is NO L mode, hence a 3-tuple copy coord).
        s_views = []
        g_views = []
        for r in range(cp):
            gD = cute.flat_divide(tensors[r], epi_tile)  # (epi_m, epi_n, nt_tok, nt_feat)
            s_r, g_r = cpasync.tma_partition(
                atoms[r],
                0,
                cute.make_layout(1),
                cute.group_modes(sPostAct, 0, cute.rank(sPostAct) - 1),
                cute.group_modes(gD, 0, 2),
            )
            s_views.append(s_r)
            g_views.append(g_r)

        # ROUTE-2 (A) N_i-STRIDE-1 store constants (const_expr): the padded x-tile count n_x per Y and
        # the flag. When set, the copy_fn unravels the flat M-tile X-inner (tile_x=i-tile, tile_y=j) and
        # positions global-i by shard (my_cp_rank*Xg_pad); g_views[r] is the 3-D (N_i, feat, N_j) grid.
        route2_ni: cutlass.Constexpr[bool] = bool(getattr(self, "_a2a_route2_ni", False))
        # BATCH-MODE recv (see _a2a_b_mode): the peer view carries a TRAILING batch grid mode, so the
        # copy coord gains one coordinate and the flat M-tile index has a plane component to peel off.
        # False for every plain / B==1-baked build -> the whole tree below is const_expr-pruned there.
        b_mode: cutlass.Constexpr[bool] = bool(self._a2a_b_mode())
        if const_expr(route2_ni):
            # B10 my global-j block base = cp1_coord*N_j_loc into the FULL-global-N recv N_j (the store
            # writes only my j-block; 1-D cp1_coord==0 -> j base 0 -> byte-identical).
            if const_expr(self._a2a_dynamic):
                # RUNTIME n_x AND j-base (one-compile-many-N): the per-N rebound recv is 3-D
                # (N_i=cp0*Xg_pad, 2*Dloc, N_j=full global-N) with BOTH token axes marked
                # mark_compact_shape_dynamic (mode 1 = N_i, mode 2 = N_j). tensors[0] is that peer view
                # -> shape[0]=N_i, shape[2]=N_j (runtime). n_x = N_i//(cp0*BLK_M); N_j_loc = N_j//cp1;
                # my j-base = cp1_coord*N_j_loc. Deriving BOTH from the runtime shape (not the anchor
                # const) is REQUIRED: a straddle N whose N_j_loc != the anchor's garbles otherwise
                # (rel_L2~1.2 at N=1040 vs anchor 1024). Mirrors the scheduler/A-load runtime n_x
                # (get_scheduler_arguments :501, _gA_local_tile :472) + dynamic rows_per_peer (:995).
                n_x_r2_dyn = Int32(tensors[0].shape[0]) // Int32(cp0 * tile_m)
                j_off_r2_dyn = Int32(cp1_coord) * (Int32(tensors[0].shape[2]) // Int32(cp1))
                # B-MODE: the walk's flat j-tile is (b*Yg_loc + y), so the plane and the local j have
                # to be split apart before the cp1 shard base is added. shape[2] is the PER-PLANE
                # global-j extent (the plane is its own mode), so Yg_loc = shape[2]//cp1 -- the same
                # quantity j_off_r2_dyn already divides out, and no runtime B is needed.
                yg_loc_r2_dyn = Int32(tensors[0].shape[2]) // Int32(cp1)
            else:
                # STATIC single-shape front: n_x from the walk Xg (token_grid X); my j-base from the walk
                # Yg = N_j_loc (token_grid Y) baked const (correct for the single compiled shape).
                Xg_r2: cutlass.Constexpr[int] = int(self._token_grid_x)
                n_x_r2: cutlass.Constexpr[int] = (
                    Xg_r2 + tile_m - 1
                ) // tile_m  # tile_m == BLK_M (static)
                n_j_loc_r2: cutlass.Constexpr[int] = int(self._token_grid_y)
                j_off_r2: cutlass.Constexpr[int] = cp1_coord * n_j_loc_r2
        # B-MODE, HOISTED. Every quantity below is CTA-UNIFORM (a pure function of
        # `tile_coord_mnkl[0]`), and `copy_fn` runs ONCE PER EPILOGUE SUBTILE -- so leaving the plane
        # unravel inside it pays a RUNTIME INTEGER DIVIDE per subtile on hardware with no
        # integer-divide unit. Computing it here, in the same @cute.jit scope the existing
        # `n_x_r2_dyn` / `j_off_r2_dyn` constants already live in, leaves `copy_fn` with one add. The
        # non-b_mode arms below are untouched and keep their arithmetic where it was, so their
        # emitted code does not move.
        if const_expr(b_mode):
            if const_expr(route2_ni):
                if const_expr(self._a2a_dynamic):
                    _tx = tile_coord_mnkl[0] % n_x_r2_dyn
                    _ty_raw = tile_coord_mnkl[0] // n_x_r2_dyn
                    _tile_b = _ty_raw // yg_loc_r2_dyn
                    _dst_j = j_off_r2_dyn + (_ty_raw - _tile_b * yg_loc_r2_dyn)
                    _dst_i_tile = (Int32(cp0_coord) * n_x_r2_dyn + _tx) * Int32(m_sub_per_tile)
                else:
                    _tx = tile_coord_mnkl[0] % Int32(n_x_r2)
                    _ty_raw = tile_coord_mnkl[0] // Int32(n_x_r2)
                    _tile_b = _ty_raw // Int32(n_j_loc_r2)
                    _dst_j = Int32(j_off_r2) + (_ty_raw - _tile_b * Int32(n_j_loc_r2))
                    _dst_i_tile = (Int32(cp0_coord * n_x_r2) + _tx) * Int32(m_sub_per_tile)
            else:
                # COMPOSITE-K: the walk's flat M-tile is (b*Yg + y)*n_x + x and the recv strides the
                # PLANE outside the cp slot, so the plane leaves the column index and becomes its own
                # store coordinate; what remains is the per-plane tile index. tiles_per_b is exact --
                # rows_per_peer is Yg*n_x*BLK_M here (the transpose_in pad), no partial trailing tile.
                _tiles_per_b = Int32(rows_per_peer) // Int32(tile_m)
                _tile_b = tile_coord_mnkl[0] // _tiles_per_b
                _dst_tok_tile = Int32(my_col_sub_off) + (
                    tile_coord_mnkl[0] - _tile_b * _tiles_per_b
                ) * Int32(m_sub_per_tile)

        @cute.jit
        def copy_fn(src_idx, dst_idx, **kwargs):
            sub_m, sub_n = dst_idx[0], dst_idx[1]
            n_tile = tile_coord_mnkl[1]
            # PER-HALF: the postact N(feature)-tile selects the half (a=0, b=1), then the peer's
            # D-slice within the half; my whole token block lands on EVERY peer at my COLUMN offset.
            half = n_tile // Int32(n_tiles_per_half)
            n_in_half = n_tile % Int32(n_tiles_per_half)
            peer = n_in_half // Int32(n_tiles_per_Dslice)
            n_in_slice = n_in_half % Int32(n_tiles_per_Dslice)
            dst_feat_sub = half * Int32(dloc_sub) + n_in_slice * Int32(n_sub_per_tile) + sub_n
            # const_expr-unrolled peer select (cp is compile-time); avoids the MLIR alias mis-fold
            # on runtime-conditional aliased access.  NO cute.printf here (device deadlock under nvshmem).
            if const_expr(route2_ni):
                # N_i-STRIDE-1: unravel the flat M-tile X-inner -> (tile_x=i-tile, tile_y=j-in-block).
                # global-i is positioned by the i-shard (cp0_coord): dst_i (epi_m units) =
                # (cp0_coord*n_x + tile_x)*m_sub_per_tile + sub_m (i.e. cp0_coord*Xg_pad//epi_m). global-j
                # is positioned by the j-shard: dst_j = cp1_coord*N_j_loc + tile_y (into the full-N recv
                # N_j). 1-D: cp0_coord==my_cp_rank, cp1_coord==0 -> byte-identical. 4-coord (box+i+feat+j).
                if const_expr(b_mode):
                    dst_i_sub = _dst_i_tile + sub_m  # HOISTED above; only sub_m is per-subtile
                    for r in cutlass.range_constexpr(cp):
                        if peer == Int32(r):
                            cute.copy(
                                atoms[r],
                                s_views[r][(None, src_idx)],
                                g_views[r][(None, dst_i_sub, dst_feat_sub, _dst_j, _tile_b)],
                            )
                elif const_expr(self._a2a_dynamic):
                    # DYNAMIC: runtime n_x + runtime j-base (n_x_r2_dyn, j_off_r2_dyn) -> one compile any N.
                    tile_x = tile_coord_mnkl[0] % n_x_r2_dyn
                    tile_y = tile_coord_mnkl[0] // n_x_r2_dyn
                    dst_i_sub = (Int32(cp0_coord) * n_x_r2_dyn + tile_x) * Int32(
                        m_sub_per_tile
                    ) + sub_m
                    dst_j = j_off_r2_dyn + tile_y  # cp1_coord*N_j_loc(runtime) + local j-tile
                    for r in cutlass.range_constexpr(cp):
                        if peer == Int32(r):
                            cute.copy(
                                atoms[r],
                                s_views[r][(None, src_idx)],
                                g_views[r][(None, dst_i_sub, dst_feat_sub, dst_j)],
                            )
                else:
                    # STATIC: const n_x + const j-base (byte-identical to the shipped 4f9351d path at cp1==1).
                    tile_x = tile_coord_mnkl[0] % Int32(n_x_r2)
                    tile_y = tile_coord_mnkl[0] // Int32(n_x_r2)
                    dst_i_sub = (Int32(cp0_coord * n_x_r2) + tile_x) * Int32(m_sub_per_tile) + sub_m
                    dst_j = Int32(j_off_r2) + tile_y  # cp1_coord*N_j_loc(const) + local j-tile
                    for r in cutlass.range_constexpr(cp):
                        if peer == Int32(r):
                            cute.copy(
                                atoms[r],
                                s_views[r][(None, src_idx)],
                                g_views[r][(None, dst_i_sub, dst_feat_sub, dst_j)],
                            )
            else:
                # token = stride-1 col; feature = outer row (half offset + intra-slice feature subtile).
                # PARTIAL-TOKEN CLAMP: the atom is on MY (rows_per_peer, 2*Dloc) column block, so dst_tok
                # is RELATIVE to my block (no my_col_sub_off); the partial-last M-tile's rows past
                # rows_per_peer are dropped by the high-end clamp. Default: absolute (my_col_sub_off added).
                if const_expr(b_mode):
                    dst_tok_sub = _dst_tok_tile + sub_m  # HOISTED above
                elif const_expr(self._a2a_should_clamp()):
                    dst_tok_sub = tile_coord_mnkl[0] * Int32(m_sub_per_tile) + sub_m
                else:
                    dst_tok_sub = (
                        Int32(my_col_sub_off) + tile_coord_mnkl[0] * Int32(m_sub_per_tile) + sub_m
                    )
                # COMPILE TIME, not style. A nested `if` inside a `cutlass.range_constexpr` loop
                # makes the compiler unroll the loop AND compile-time-prune the nested branch PER
                # ITERATION -- the combinatorial blowup CLAUDE.md names as the first thing to check
                # when a cold compile goes over the 5 s bar.
                #
                # WHAT IS MEASURED, and it is narrower than an earlier version of this comment said.
                # OBSERVED: with the two-arm form, the cp=8 perf selection went from 14 cells in
                # 348 s to 2 cells in 900 s. That is real and it read as a hang, which it was not.
                # NOT REPRODUCED, and the instrument that was supposed to show it is BLIND.
                # A direct A/B of the known-slow tree against the known-good one, both on the same
                # narrowed selection and both paying cold init, reads **1.002x** on pure-compile
                # (`heuristic_picks`) cells. **That metric cannot separate a tree that HAS the
                # defect from one that does not**, so a green reading from it certifies nothing --
                # "control proves it WORKS, silence proves it was NEEDED", and this control never
                # went red. An earlier 1.34x from the same cells was an artifact of comparing a
                # WARM reference against a COLD arm; matched, it vanishes.
                # **Do not quote ~18x, and do not cite compile-time parity as evidence the
                # blowup is gone.** The workflow cells were tested too: known-BAD vs known-GOOD is
                # **1.001x** there, as it was 1.002x on pure-compile. **Neither cell type can
                # separate the trees**, so on that venue `fb846de == d1f1e37` is TRIVIALLY TRUE and
                # certifies nothing about this hoist.
                #
                # SETTLED -- and it settles AGAINST this hoist being the cause.
                # The one arm every earlier version of this comment called missing -- the known-BAD
                # tree, on the venue where the 900 s was actually seen -- has now been run.
                # `650a780`, §7.3 cp8 selection, freeze dir POPULATED (48 entries):
                # **14 executed cells in 310.11 s**, against **328.70 s** for `fb846de` and
                # **348.8 s** for `d1f1e37` on the same venue and the same selection.
                # **The 900 s does not reproduce.**
                #
                # The inference that closes this does NOT depend on which rival account is right.
                # The nested-branch structure is HELD CONSTANT between the original 900 s run and
                # this one -- same commit, same source, verified by reading it -- and this one is
                # fast. **So the structure is not SUFFICIENT to produce the slowdown**, and
                # `range_constexpr` pruning cannot be the attribution whatever else was going on.
                # The freeze-MISS account survives as the leading rival (a MISS makes the autotuner
                # sweep and compile many configs per cell, ~450 s/cell falls out with no compiler
                # pathology, and a miss is worth a measured 2.13x), but it no longer has to be
                # excluded for the conclusion to hold.
                #
                # So the hoist is HYGIENE, not a fix, and the compile-blowup issue is closed as
                # **NOT REPRODUCED** rather than as repaired. Do not cite this edit as having fixed
                # a blowup, and do not re-open the issue on the strength of this comment.
                #
                # The hoist stands on its own regardless: a nested branch in an unrolled loop is
                # the named blowup shape, and the repaired form is byte-identical where
                # `_a2a_b_mode()` is False.
                #
                # Hoisting the COORD out leaves ONE `cute.copy` in the unrolled body, so the loop
                # emits cp store sites instead of cp x 2 pruned pairs. Same store, same arguments --
                # a non-b build assembles exactly the 3-tuple the old `else` arm passed, which is
                # what keeps it byte-identical (Gate 4 measures this).
                crd = (
                    (None, dst_tok_sub, dst_feat_sub, _tile_b)
                    if const_expr(b_mode)
                    else (None, dst_tok_sub, dst_feat_sub)
                )
                for r in cutlass.range_constexpr(cp):
                    if peer == Int32(r):
                        cute.copy(atoms[r], s_views[r][(None, src_idx)], g_views[r][crd])

        return copy_fn, s_views[0], g_views[0]

    # ==================================================================
    # DECOUPLED producer/consumer SIMT-ring drain — the in-epilogue producer + the consumer WG.
    # ==================================================================
    def _a2a_decoupled_producer_copy_fn(
        self, params, tile_shape_mn, epi_tile, sPostAct, tile_coord_mnkl, storage
    ):
        """Build the PRODUCER copy_fn (decoupled store): SMEM postact box -> LOCAL-GMEM ring + signal.

        The MMA warpgroup's epilogue calls this per postact subtile. Instead of the coupled (awaited)
        peer TMA-S2G, it: (1) waits the slot's ``empty`` mbarrier (skip on first use of the slot),
        (2) issues the producer-TMA S2G ``sPostAct[:,:,src_idx]`` -> the CTA's GMEM ring slot (the SAME
        async store the coupled path uses, retargeted -> O(1) MMA-warp occupancy) + a CHEAP local
        commit+wait (local HBM, NOT the NVLink stall), (3) writes the per-slot peer/recv-index metadata
        for the consumer, (4) arrives ``full[s]`` + bumps the producer counter. The MMA warpgroup then
        advances WITHOUT waiting for the peer put -> the put latency hides behind the next subtile's MMA.

        The peer / dst_tok_sub / dst_feat_sub math is IDENTICAL to the coupled
        :meth:`_a2a_postact_peer_store_copy_fn`; here it is computed to WRITE the metadata the consumer
        reads (the consumer issues the actual put). The front's peer is feature-tile-uniform per CTA."""
        # §10: the producer path only ever runs on the differential ib_drain store (the plain-decoupled
        # all-ring-write store is RETIRED at configure). The egress + is_p2p tables below assume it ->
        # assert loudly at trace if the invariant is ever violated (belt-and-suspenders vs the configure gate).
        assert bool(getattr(self, "_a2a_ib_drain", False)), (
            "front-A2A decoupled producer requires ib_drain=True (plain-decoupled store retired, §10)."
        )
        # WIDE-PUT dispatch: the wide (coalescing) producer is a SEPARATE builder (own batch-accumulation
        # copy_fn). This host-side branch keeps the per-subtile builder below 100% BYTE-IDENTICAL when
        # ib_wide is OFF (the default-off identity is trivially this `if`).
        if bool(getattr(self, "_a2a_ib_wide", False)):
            return self._a2a_wide_producer_copy_fn(
                params, tile_shape_mn, epi_tile, sPostAct, tile_coord_mnkl, storage
            )
        cp: cutlass.Constexpr[int] = self._a2a_cp
        my_cp_rank: cutlass.Constexpr[int] = self._a2a_my_cp_rank
        tile_m: cutlass.Constexpr[int] = tile_shape_mn[0]
        tile_n: cutlass.Constexpr[int] = tile_shape_mn[1]
        epi_m: cutlass.Constexpr[int] = epi_tile[0]
        epi_n: cutlass.Constexpr[int] = epi_tile[1]
        m_sub_per_tile: cutlass.Constexpr[int] = tile_m // epi_m
        n_sub_per_tile: cutlass.Constexpr[int] = tile_n // epi_n
        rd: cutlass.Constexpr[int] = self._a2a_ring_depth
        nfields: cutlass.Constexpr[int] = self._DECOUPLED_META_FIELDS
        # Recv geometry off the host-built peer tensors[0] permuted to (M_full, 2*Dloc): col offset +
        # feature half/slice math (same as the coupled copy_fn). Dynamic (one compile, many token
        # counts) reads rows_per_peer off the runtime M_full; static bakes it.
        tensors = params.postact_peer_tensors
        if const_expr(self._a2a_dynamic):
            # tensors[0]==atom_recv is the per-rank column-BLOCK clamp atom (shape[0]==rpp, ALREADY
            # per-peer) under auto-clamp, ELSE the full recv (shape[0]==M_full -> //cp). //cp on the
            # clamp block HALVES rows_per_peer -> run_tokens tail-clamp drops the upper half of every
            # block (the N1000 dynamic-auto-clamp mis-deliver). Branch on the SAME should_clamp const.
            if const_expr(self._a2a_should_clamp()):
                rows_per_peer = tensors[0].shape[0]
            else:
                rows_per_peer = tensors[0].shape[0] // cp
            my_col_sub_off = (my_cp_rank * rows_per_peer) // epi_m
        else:
            # B-MODE (static): _a2a_rows_per_peer counts ALL planes (B*Yg*n_x*BLK_M) but the recv's cp
            # slots stride by ONE plane's worth, so the column offset must divide the batch out -- the
            # SAME correction the coupled builder makes. Not b-mode -> `// 1` -> byte-identical.
            rows_per_peer: cutlass.Constexpr[int] = self._a2a_rows_per_peer // (
                int(self._token_grid_b) if self._a2a_b_mode() else 1
            )
            my_col_sub_off: cutlass.Constexpr[int] = (my_cp_rank * rows_per_peer) // epi_m
        two_dloc: cutlass.Constexpr[int] = int(tensors[0].shape[1])
        dloc: cutlass.Constexpr[int] = two_dloc // 2
        n_tiles_per_Dslice: cutlass.Constexpr[int] = dloc // tile_n
        n_tiles_per_half: cutlass.Constexpr[int] = cp * n_tiles_per_Dslice
        dloc_sub: cutlass.Constexpr[int] = dloc // epi_n
        # ROUTE-2 (A) N_i-stride-1 coords (mirror the coupled copy_fn :1391-1411): the flat M-tile unravels
        # X(i)-inner (tile_x=i-tile, tile_y=j); global-i positioned by the i-shard (cp0_coord), global-j by
        # the j-shard (cp1_coord). The FEATURE math above (dloc/half/slice) is layout-agnostic (tensors[0]
        # shape[1]=2*Dloc for BOTH the 2-D D-major recv and the 3-D N_i recv). Default D-major -> unused.
        route2_ni: cutlass.Constexpr[bool] = bool(getattr(self, "_a2a_route2_ni", False))
        if const_expr(route2_ni):
            _cp_axes = self._a2a_cp_axis_sizes
            cp0: cutlass.Constexpr[int] = int(_cp_axes[0])
            cp1: cutlass.Constexpr[int] = int(_cp_axes[1]) if len(_cp_axes) > 1 else 1
            cp0_coord: cutlass.Constexpr[int] = my_cp_rank // cp1
            cp1_coord: cutlass.Constexpr[int] = my_cp_rank % cp1
            if const_expr(self._a2a_dynamic):
                # RUNTIME n_x + j-base off the 3-D recv (shape[0]=N_i, shape[2]=N_j) -> one compile any N.
                n_x_r2_dyn = Int32(tensors[0].shape[0]) // Int32(cp0 * tile_m)
                j_off_r2_dyn = Int32(cp1_coord) * (Int32(tensors[0].shape[2]) // Int32(cp1))
            else:
                Xg_r2: cutlass.Constexpr[int] = int(self._token_grid_x)
                n_x_r2: cutlass.Constexpr[int] = (Xg_r2 + tile_m - 1) // tile_m
                n_j_loc_r2: cutlass.Constexpr[int] = int(self._token_grid_y)
                j_off_r2: cutlass.Constexpr[int] = cp1_coord * n_j_loc_r2

        # B-MODE, HOISTED (mirrors the coupled copy_fn's hoist, for the same reason). The recv gains a
        # batch PLANE mode, so the walk's flat M-tile index carries a plane component that must be peeled
        # off before what remains can index the token axis -- and the plane itself must reach the
        # CONSUMER, which has no store descriptor to carry it and would otherwise address plane 0 for
        # every plane. Every quantity here is CTA-UNIFORM (a pure function of `tile_coord_mnkl[0]`) while
        # `copy_fn` runs ONCE PER EPILOGUE SUBTILE, so leaving the unravel inside it would pay a runtime
        # integer DIVIDE per subtile on hardware with no integer-divide unit. `const_expr`-pruned whole
        # when b_mode is False, so the non-b arms below keep their arithmetic exactly where it was.
        b_mode: cutlass.Constexpr[bool] = bool(self._a2a_b_mode())
        mb_idx: cutlass.Constexpr[int] = self._DECOUPLED_META_B_INDEX  # plane field; b_mode only
        if const_expr(b_mode):
            if const_expr(route2_ni):
                # ROUTE-2: the flat j-tile is (b*Yg_loc + y), so the plane and the local j split apart
                # before the cp1 shard base is added. shape[2] is the PER-PLANE global-j extent (the
                # plane is its own trailing mode), so Yg_loc = shape[2]//cp1 -- no runtime B needed.
                if const_expr(self._a2a_dynamic):
                    _tx = tile_coord_mnkl[0] % n_x_r2_dyn
                    _ty_raw = tile_coord_mnkl[0] // n_x_r2_dyn
                    _yg_loc = Int32(tensors[0].shape[2]) // Int32(cp1)
                    _tile_b = _ty_raw // _yg_loc
                    _dst_j = j_off_r2_dyn + (_ty_raw - _tile_b * _yg_loc)
                    _dst_i_tile = (Int32(cp0_coord) * n_x_r2_dyn + _tx) * Int32(m_sub_per_tile)
                else:
                    _tx = tile_coord_mnkl[0] % Int32(n_x_r2)
                    _ty_raw = tile_coord_mnkl[0] // Int32(n_x_r2)
                    _tile_b = _ty_raw // Int32(n_j_loc_r2)
                    _dst_j = Int32(j_off_r2) + (_ty_raw - _tile_b * Int32(n_j_loc_r2))
                    _dst_i_tile = (Int32(cp0_coord * n_x_r2) + _tx) * Int32(m_sub_per_tile)
            else:
                # COMPOSITE-K: the walk's flat M-tile is (b*Yg + y)*n_x + x and the recv strides the
                # PLANE outside the cp slot, so the plane leaves the column index and becomes its own
                # store coordinate; what remains is the per-plane tile index. tiles_per_b is exact --
                # rows_per_peer is the PER-PLANE Yg*n_x*BLK_M (the transpose_in pad), no partial tile.
                _tiles_per_b = Int32(rows_per_peer) // Int32(tile_m)
                _tile_b = tile_coord_mnkl[0] // _tiles_per_b
                _dst_tok_tile = Int32(my_col_sub_off) + (
                    tile_coord_mnkl[0] - _tile_b * _tiles_per_b
                ) * Int32(m_sub_per_tile)

        full_ptr = storage.decoupled.full.data_ptr()
        empty_ptr = storage.decoupled.empty.data_ptr()
        meta = storage.decoupled.meta.get_tensor((rd * nfields,))
        pcount = storage.decoupled.pcount.get_tensor((1,))

        # Producer-TMA ring write: tma_partition the ring-store atom (box <-> sPostAct), mirroring the
        # coupled store but targeting the GMEM ring. ring_store_tensor is (epi_m, epi_n, rd, grid).
        ring_atom = params.ring_store_atom
        gRing = cute.flat_divide(params.ring_store_tensor, epi_tile)  # (em,en,1,1,rd,grid)
        s_ring, g_ring = cpasync.tma_partition(
            ring_atom,
            0,
            cute.make_layout(1),
            cute.group_modes(sPostAct, 0, cute.rank(sPostAct) - 1),
            cute.group_modes(gRing, 0, 2),
        )
        bidx = cute.arch.block_idx()[0] + cute.arch.block_idx()[2]
        # IB-DRAIN differential: build the cp COUPLED peer-store views (the SAME per-peer TMA-S2G
        # partition the coupled store uses) so an NVLink (P2P) peer's subtile takes the coupled store
        # DIRECTLY (byte-identical data path) while an IB peer stages into the ring. is_p2p is a
        # const_expr tuple -> the per-peer arm is compile-time-selected inside the runtime peer match.
        ib_drain: cutlass.Constexpr[bool] = bool(self._a2a_ib_drain)
        s_pviews, g_pviews = [], []
        if const_expr(ib_drain):
            p_atoms = params.postact_peer_atoms
            p_tensors = params.postact_peer_tensors
            for r in range(cp):
                gD = cute.flat_divide(p_tensors[r], epi_tile)  # (epi_m, epi_n, nt_tok, nt_feat)
                s_pr, g_pr = cpasync.tma_partition(
                    p_atoms[r],
                    0,
                    cute.make_layout(1),
                    cute.group_modes(sPostAct, 0, cute.rank(sPostAct) - 1),
                    cute.group_modes(gD, 0, 2),
                )
                s_pviews.append(s_pr)
                g_pviews.append(g_pr)
        is_p2p = self._a2a_is_p2p if const_expr(ib_drain) else None

        @cute.jit
        def copy_fn(src_idx, dst_idx, **kwargs):
            sub_m, sub_n = dst_idx[0], dst_idx[1]
            n_tile = tile_coord_mnkl[1]
            half = n_tile // Int32(n_tiles_per_half)
            n_in_half = n_tile % Int32(n_tiles_per_half)
            peer = n_in_half // Int32(n_tiles_per_Dslice)
            n_in_slice = n_in_half % Int32(n_tiles_per_Dslice)
            # dst_tok_sub = the subtile index along the recv STRIDE-1 axis (token for D-major, N_i=i_loc
            # for route2_ni — the SAME role, §10 R1 layout-factoring). dst_j = the N_j grid column (route2_ni
            # 3rd store coord; 0 / unused for D-major). run_tokens = the put's valid run.
            if const_expr(route2_ni):
                # N_i-STRIDE-1 (mirror the coupled copy_fn :1432-1443): unravel the flat M-tile X(i)-inner;
                # global-i by the i-shard (cp0_coord), global-j by the j-shard (cp1_coord). run = epi_m
                # always — the padded i-shard is zero-tail (pad rows glu(0)=0, safe to put), no partial.
                if const_expr(b_mode):
                    # HOISTED above: the plane is peeled off the flat j-tile, so `_dst_j` is the LOCAL
                    # j column WITHIN this plane and only `sub_m` remains per-subtile.
                    dst_tok_sub = _dst_i_tile + sub_m
                    dst_j = _dst_j
                elif const_expr(self._a2a_dynamic):
                    tile_x = tile_coord_mnkl[0] % n_x_r2_dyn
                    tile_y = tile_coord_mnkl[0] // n_x_r2_dyn
                    dst_tok_sub = (Int32(cp0_coord) * n_x_r2_dyn + tile_x) * Int32(
                        m_sub_per_tile
                    ) + sub_m
                    dst_j = j_off_r2_dyn + tile_y
                else:
                    tile_x = tile_coord_mnkl[0] % Int32(n_x_r2)
                    tile_y = tile_coord_mnkl[0] // Int32(n_x_r2)
                    dst_tok_sub = (Int32(cp0_coord * n_x_r2) + tile_x) * Int32(
                        m_sub_per_tile
                    ) + sub_m
                    dst_j = Int32(j_off_r2) + tile_y
                run_tokens = Int32(epi_m)
            else:
                # dst_tok_sub is RELATIVE to my column block under auto-clamp (mirror the coupled copy_fn
                # :1228): the P2P arm's clamp atom AND the consumer's block-sliced recv_local both address MY
                # block, so dst_tok drops my_col_sub_off. Aligned (no clamp) -> absolute (my_col_sub_off
                # added) -> byte-identical. run_tokens (§2): epi_m for a full subtile, rows_per_peer%epi_m
                # for the partial tail, 0 for a fully-OOB subtile (tile_M>epi_m). Branch-free ternary select.
                if const_expr(b_mode):
                    # HOISTED above: the plane is peeled out of the flat M-tile, leaving the PER-PLANE
                    # token tile. ABSOLUTE (my_col_sub_off already folded in) -- b-mode and the partial-
                    # token clamp are mutually exclusive, refused at the recv build, so there is no
                    # relative form to choose here. `dst_tok_local` drives the run clamp against the
                    # PER-PLANE rows_per_peer, which is what makes the clamp exact under b-mode.
                    dst_tok_sub = _dst_tok_tile + sub_m
                    dst_tok_local = dst_tok_sub - Int32(my_col_sub_off)
                elif const_expr(self._a2a_should_clamp()):
                    dst_tok_sub = (
                        tile_coord_mnkl[0] * Int32(m_sub_per_tile) + sub_m
                    )  # RELATIVE to my block
                    dst_tok_local = dst_tok_sub
                else:
                    dst_tok_sub = (
                        Int32(my_col_sub_off) + tile_coord_mnkl[0] * Int32(m_sub_per_tile) + sub_m
                    )
                    dst_tok_local = dst_tok_sub - Int32(my_col_sub_off)
                rem_tok = Int32(rows_per_peer) - dst_tok_local * Int32(
                    epi_m
                )  # tokens left from subtile
                rem_tok = (
                    Int32(epi_m) if rem_tok > Int32(epi_m) else rem_tok
                )  # cap at a full subtile
                run_tokens = Int32(0) if rem_tok < Int32(0) else rem_tok  # floor at 0 (fully-OOB)
                dst_j = Int32(
                    0
                )  # D-major has no N_j grid coord (unused; keeps dst_j defined for meta)
            dst_feat_sub = half * Int32(dloc_sub) + n_in_slice * Int32(n_sub_per_tile) + sub_n
            # PEER-STORE COORD for the P2P (coupled TMA-S2G) arm, assembled ONCE so the arms cannot
            # drift: box + the recv stride-1 tile (token D-major / N_i route2_ni) + the feature tile,
            # then route2_ni's N_j grid column, then b-mode's batch PLANE. Same coordinate the coupled
            # store builds -- this arm writes the SAME descriptor, so it needs the SAME rank. All host-
            # side `const_expr`, so a non-route2 non-b build assembles exactly `(None, dst_tok_sub,
            # dst_feat_sub)` and traces the identical slice op it did before.
            p_crd = (None, dst_tok_sub, dst_feat_sub)
            if const_expr(route2_ni):
                p_crd = p_crd + (dst_j,)
            if const_expr(b_mode):
                p_crd = p_crd + (_tile_b,)
            lane = cute.arch.lane_idx()
            # producer counter -> slot + wrap phase (single-writer warp; uniform read across lanes).
            t = pcount[0]
            slot = t % Int32(rd)
            k = t // Int32(rd)
            # Wait the slot's prior occupant to be drained (skip on first use, gated k>=1). try_wait
            # SPIN-LOOP (not blocking mbarrier_wait — the safer pattern under the producer-TMA-deposit
            # ring + cross-warpgroup handshake; see reference_mbarrier_wait_hangs_tma_tx).
            if k >= Int32(1):
                for ss in cutlass.range_constexpr(rd):
                    if slot == Int32(ss):
                        done = cute.arch.mbarrier_try_wait(
                            empty_ptr + ss, (k - Int32(1)) & Int32(1)
                        )
                        while not done:
                            done = cute.arch.mbarrier_try_wait(
                                empty_ptr + ss, (k - Int32(1)) & Int32(1)
                            )
            # DIFFERENTIAL producer-TMA store (one async TMA-S2G, one commit+wait covers whichever arm
            # fired). read=False = full completion before full[s].arrive: for the IB arm the consumer
            # reads the ring slot from GMEM; for the P2P arm it lets the NVLink store finish before the
            # SMEM box is reused. §10: the plain-decoupled (ib_drain-off) all-ring-write arm is RETIRED
            # (configure_a2a rejects decoupled without ib_drain; the builder asserts _a2a_ib_drain), so
            # ib_drain is const_expr True here -> ONE differential per-peer is_p2p branch, no all-nbi else.
            with cute.arch.elect_one():
                if const_expr(ib_drain):
                    # const_expr-unrolled peer match; the is_p2p[r] arm is compile-time-selected.
                    for r in cutlass.range_constexpr(cp):
                        if peer == Int32(r):
                            if const_expr(is_p2p[r]):
                                # NVLink (P2P): coupled TMA-S2G DIRECTLY to the peer recv atom; NO ring, NO
                                # drain -> meta gets the -1 skip sentinel below. The store coord is
                                # `p_crd` above: (box, dst_tok/dst_i, dst_feat)[, dst_j][, tile_b], with
                                # the RELATIVE dst_tok under auto-clamp (descriptor OOB-drops the partial
                                # tail). Mirrors the coupled copy_fn's per-layout store.
                                cute.copy(
                                    p_atoms[r],
                                    s_pviews[r][(None, src_idx)],
                                    g_pviews[r][p_crd],
                                )
                            else:
                                # IB (non-P2P): stage the FULL postact box into the SYMMETRIC-heap ring
                                # slot (the garbage tail stays LOCAL — only the consumer's run-clamped put
                                # egresses it) for the consumer's blocking-put drain.
                                for ss in cutlass.range_constexpr(rd):
                                    if slot == Int32(ss):
                                        cute.copy(
                                            ring_atom,
                                            s_ring[(None, src_idx)],
                                            g_ring[(None, Int32(0), Int32(0), Int32(ss), bidx)],
                                        )
                cute.arch.cp_async_bulk_commit_group()
                cute.arch.cp_async_bulk_wait_group(0, read=False)
            cute.arch.sync_warp()
            # meta_peer = -1 (skip sentinel, P2P handled by the coupled store) if the runtime peer is a
            # P2P slot, else the real peer (IB, consumer drains it). Built with a const_expr-unrolled
            # TERNARY-select (NOT a scalar reassign inside a runtime `if`, which the DSL would not
            # propagate out): the `for`/`if const_expr` unroll to straight-line, leaving only the runtime
            # `select(peer==r, -1, meta_peer)`. A P2P slot still signals full/empty -> uniform handshake.
            meta_peer = peer
            if const_expr(ib_drain):
                for r in cutlass.range_constexpr(cp):
                    if const_expr(is_p2p[r]):
                        meta_peer = Int32(-1) if (peer == Int32(r)) else meta_peer
            # Per-slot metadata (one lane) so the consumer can address the peer recv (or skip on -1), run-
            # clamp the put, and skip a fully-OOB (run==0) subtile. meta[3]=run_tokens is written in this
            # SAME lane==0 block before the sync_warp below -> visible with the other 3 fields (§5.4).
            if lane == Int32(0):
                base = slot * Int32(nfields)
                meta[base + Int32(0)] = meta_peer
                meta[base + Int32(1)] = dst_tok_sub
                meta[base + Int32(2)] = dst_feat_sub
                meta[base + Int32(3)] = run_tokens
                if const_expr(route2_ni):
                    meta[base + Int32(4)] = (
                        dst_j  # extra N_i-stride-1 store coord (the N_j grid column)
                    )
                if const_expr(b_mode):
                    # The recv BATCH PLANE. The coupled arm carries it as a TMA store coordinate; the
                    # drain has no descriptor, so it travels here or the consumer writes plane 0 for
                    # every plane -- silently, since the planes are in-bounds of each other.
                    meta[base + Int32(mb_idx)] = _tile_b
            cute.arch.sync_warp()  # data + meta visible before signaling full
            # Signal full[slot] (the consumer waits it). The whole producer warp arrives -> count 32.
            for ss in cutlass.range_constexpr(rd):
                if slot == Int32(ss):
                    cute.arch.mbarrier_arrive(full_ptr + ss)
            if lane == Int32(0):
                pcount[0] = t + Int32(1)
            cute.arch.sync_warp()

        return copy_fn

    def _a2a_wide_producer_copy_fn(
        self, params, tile_shape_mn, epi_tile, sPostAct, tile_coord_mnkl, storage
    ):
        """WIDE-PUT producer (plan §3). Instead of one 256 B per-subtile put, ACCUMULATE W consecutive-
        dst_tok / same-(peer,dst_feat) IB subtiles into ONE wide ring slot (token axis W· wider) so the
        consumer drains one (b_w·epi_m)-token contiguous put per feature row. The P2P (NVLink) arm keeps
        the coupled TMA-S2G store, BYTE-IDENTICAL to the per-subtile path.

        SINGLE FLUSH POINT + finalize (count-parity §6.4): every subtile joins exactly one batch; a batch
        is flushed (meta stamp + arrive full[bc%rd]) by the FIRST subtile that does NOT continue it (a full
        batch has ``b_w==W`` so ``merge`` is False -> it flushes on the next subtile), and the CTA's
        TRAILING batch is flushed by :meth:`_a2a_wide_producer_finalize` after the MMA loop. So
        Σ(per-batch n_sub) == total_subtiles == the consumer's drain target -> producer-arrive_full ==
        consumer-arrive_empty on ANY raster (raster-agnostic; run_i widens the batches, never a correctness
        dep). Batch state persists across copy_fn calls in ``storage.decoupled.bstate`` (single-writer
        warp-0/lane-0, the SAME warp that runs this copy_fn under the epilogue's ``is_tma_warp`` gate)."""
        # TODO(#22, DEFERRED — pre-existing, NOT route2_ni-introduced): this wide producer copy_fn
        # COLD-COMPILES ~38s (> the 5s bar). Root cause = cute-DSL Pitfall #26: the BASE-CLASS epilogue's
        # ``range_constexpr`` subtile loop (``self.epilogue``, ``epi_tile_num=(tile_m//epi_m)*(tile_n//epi_n)``
        # ≈ 8-16) INLINES this copy_fn per subtile, so the tracer cost = OUTER_unroll(≈16) × THIS-FN'S
        # OP-COUNT — the coalesce state machine (merge/is_dup/flush/start/add) + the ``for r in
        # range_constexpr(cp)`` peer-select + the ``for ss in range_constexpr(rd)`` slot-select (both are
        # Pitfall-#5 workarounds: cute-DSL can't dynamic-index the STATIC Python lists p_atoms[r]/ring-slots,
        # since TMA atoms are compile-time objects). It is W-INDEPENDENT (W32≈W128) and compile-NEUTRAL
        # (D-major wide == route2 wide). NOT fixable by trimming branches HERE: they are INHERENT (coalescing)
        # or DSL-FORCED, and flattening the dynamic-if depth measured WORSE (43s — both arms traced).
        # THE FIX BELONGS IN THE BASE GEMM CLASS, not here: make the epilogue subtile loop RUNTIME
        # (``cutlass.range(epi_tile_num, unroll=1)``) so this fn is traced ONCE, not ≈16×. That is base-class
        # BLAST RADIUS (all SM90 GEMMs share ``self.epilogue``) AND it breaks the STSM R2S store's static-SMEM
        # ``src_idx`` (needs a compile-time subtile index) — so it needs a base-class change or an A2A-subclass
        # epilogue override. The BACK A2A GEMM (gemm_sm90_a2a.py) AVOIDED this by coalescing in a compiled-ONCE
        # consumer drain loop (``_cluster_drain_loop``) instead of an inlined producer copy_fn. MITIGATION
        # until then: the 38s is a ONE-TIME-per-config CACHED compile (warm-cache benches run fine).
        assert bool(getattr(self, "_a2a_ib_drain", False)), (
            "front-A2A wide producer requires ib_drain=True (rides the differential drain)."
        )
        cp: cutlass.Constexpr[int] = self._a2a_cp
        my_cp_rank: cutlass.Constexpr[int] = self._a2a_my_cp_rank
        tile_m: cutlass.Constexpr[int] = tile_shape_mn[0]
        tile_n: cutlass.Constexpr[int] = tile_shape_mn[1]
        epi_m: cutlass.Constexpr[int] = epi_tile[0]
        epi_n: cutlass.Constexpr[int] = epi_tile[1]
        m_sub_per_tile: cutlass.Constexpr[int] = tile_m // epi_m
        n_sub_per_tile: cutlass.Constexpr[int] = tile_n // epi_n
        rd: cutlass.Constexpr[int] = self._a2a_ring_depth
        nfields: cutlass.Constexpr[int] = self._DECOUPLED_META_FIELDS
        W: cutlass.Constexpr[int] = int(self._a2a_ib_wide_batch)
        tensors = params.postact_peer_tensors
        if const_expr(self._a2a_dynamic):
            # tensors[0]==atom_recv is the per-rank column-BLOCK clamp atom (shape[0]==rpp, ALREADY
            # per-peer) under auto-clamp, ELSE the full recv (shape[0]==M_full -> //cp). //cp on the
            # clamp block HALVES rows_per_peer -> run_tokens tail-clamp drops the upper half of every
            # block (the N1000 dynamic-auto-clamp mis-deliver). Branch on the SAME should_clamp const.
            if const_expr(self._a2a_should_clamp()):
                rows_per_peer = tensors[0].shape[0]
            else:
                rows_per_peer = tensors[0].shape[0] // cp
            my_col_sub_off = (my_cp_rank * rows_per_peer) // epi_m
        else:
            # B-MODE (static): _a2a_rows_per_peer counts ALL planes (B*Yg*n_x*BLK_M) but the recv's cp
            # slots stride by ONE plane's worth, so the column offset must divide the batch out -- the
            # SAME correction the coupled builder makes. Not b-mode -> `// 1` -> byte-identical.
            rows_per_peer: cutlass.Constexpr[int] = self._a2a_rows_per_peer // (
                int(self._token_grid_b) if self._a2a_b_mode() else 1
            )
            my_col_sub_off: cutlass.Constexpr[int] = (my_cp_rank * rows_per_peer) // epi_m
        two_dloc: cutlass.Constexpr[int] = int(tensors[0].shape[1])
        dloc: cutlass.Constexpr[int] = two_dloc // 2
        n_tiles_per_Dslice: cutlass.Constexpr[int] = dloc // tile_n
        n_tiles_per_half: cutlass.Constexpr[int] = cp * n_tiles_per_Dslice
        dloc_sub: cutlass.Constexpr[int] = dloc // epi_n
        # ROUTE-2 (A) N_i-stride-1 coords (mirror the coupled / per-subtile producer): the flat M-tile
        # unravels X(i)-inner; global-i by the i-shard (cp0_coord), global-j by the j-shard (cp1_coord).
        # The wide coalesce runs along N_i (dst_i) WITHIN one N_j column (b_j). Default D-major -> unused.
        route2_ni: cutlass.Constexpr[bool] = bool(getattr(self, "_a2a_route2_ni", False))
        if const_expr(route2_ni):
            _cp_axes = self._a2a_cp_axis_sizes
            cp0: cutlass.Constexpr[int] = int(_cp_axes[0])
            cp1: cutlass.Constexpr[int] = int(_cp_axes[1]) if len(_cp_axes) > 1 else 1
            cp0_coord: cutlass.Constexpr[int] = my_cp_rank // cp1
            cp1_coord: cutlass.Constexpr[int] = my_cp_rank % cp1
            if const_expr(self._a2a_dynamic):
                n_x_r2_dyn = Int32(tensors[0].shape[0]) // Int32(cp0 * tile_m)
                j_off_r2_dyn = Int32(cp1_coord) * (Int32(tensors[0].shape[2]) // Int32(cp1))
            else:
                Xg_r2: cutlass.Constexpr[int] = int(self._token_grid_x)
                n_x_r2: cutlass.Constexpr[int] = (Xg_r2 + tile_m - 1) // tile_m
                n_j_loc_r2: cutlass.Constexpr[int] = int(self._token_grid_y)
                j_off_r2: cutlass.Constexpr[int] = cp1_coord * n_j_loc_r2

        # B-MODE, HOISTED (mirrors the coupled copy_fn's hoist, for the same reason). The recv gains a
        # batch PLANE mode, so the walk's flat M-tile index carries a plane component that must be peeled
        # off before what remains can index the token axis -- and the plane itself must reach the
        # CONSUMER, which has no store descriptor to carry it and would otherwise address plane 0 for
        # every plane. Every quantity here is CTA-UNIFORM (a pure function of `tile_coord_mnkl[0]`) while
        # `copy_fn` runs ONCE PER EPILOGUE SUBTILE, so leaving the unravel inside it would pay a runtime
        # integer DIVIDE per subtile on hardware with no integer-divide unit. `const_expr`-pruned whole
        # when b_mode is False, so the non-b arms below keep their arithmetic exactly where it was.
        b_mode: cutlass.Constexpr[bool] = bool(self._a2a_b_mode())
        mb_idx: cutlass.Constexpr[int] = self._DECOUPLED_META_B_INDEX  # plane field; b_mode only
        if const_expr(b_mode):
            if const_expr(route2_ni):
                # ROUTE-2: the flat j-tile is (b*Yg_loc + y), so the plane and the local j split apart
                # before the cp1 shard base is added. shape[2] is the PER-PLANE global-j extent (the
                # plane is its own trailing mode), so Yg_loc = shape[2]//cp1 -- no runtime B needed.
                if const_expr(self._a2a_dynamic):
                    _tx = tile_coord_mnkl[0] % n_x_r2_dyn
                    _ty_raw = tile_coord_mnkl[0] // n_x_r2_dyn
                    _yg_loc = Int32(tensors[0].shape[2]) // Int32(cp1)
                    _tile_b = _ty_raw // _yg_loc
                    _dst_j = j_off_r2_dyn + (_ty_raw - _tile_b * _yg_loc)
                    _dst_i_tile = (Int32(cp0_coord) * n_x_r2_dyn + _tx) * Int32(m_sub_per_tile)
                else:
                    _tx = tile_coord_mnkl[0] % Int32(n_x_r2)
                    _ty_raw = tile_coord_mnkl[0] // Int32(n_x_r2)
                    _tile_b = _ty_raw // Int32(n_j_loc_r2)
                    _dst_j = Int32(j_off_r2) + (_ty_raw - _tile_b * Int32(n_j_loc_r2))
                    _dst_i_tile = (Int32(cp0_coord * n_x_r2) + _tx) * Int32(m_sub_per_tile)
            else:
                # COMPOSITE-K: the walk's flat M-tile is (b*Yg + y)*n_x + x and the recv strides the
                # PLANE outside the cp slot, so the plane leaves the column index and becomes its own
                # store coordinate; what remains is the per-plane tile index. tiles_per_b is exact --
                # rows_per_peer is the PER-PLANE Yg*n_x*BLK_M (the transpose_in pad), no partial tile.
                _tiles_per_b = Int32(rows_per_peer) // Int32(tile_m)
                _tile_b = tile_coord_mnkl[0] // _tiles_per_b
                _dst_tok_tile = Int32(my_col_sub_off) + (
                    tile_coord_mnkl[0] - _tile_b * _tiles_per_b
                ) * Int32(m_sub_per_tile)

        full_ptr = storage.decoupled.full.data_ptr()
        empty_ptr = storage.decoupled.empty.data_ptr()
        meta = storage.decoupled.meta.get_tensor((rd * nfields,))
        bstate = storage.decoupled.bstate.get_tensor((self._DECOUPLED_BSTATE_FIELDS,))
        bb_idx: cutlass.Constexpr[int] = self._DECOUPLED_BSTATE_B_INDEX  # plane slot; b_mode only

        # WIDE ring view: the caller sizes the ring token-axis W· wider -> flat_divide exposes an extra
        # W (token-tile) mode; the producer writes subtile at token-tile position ``b_w`` (= token offset
        # b_w·epi_m). ring_store_tensor is (W·epi_m, epi_n, rd, grid) -> gRing (epi_m,epi_n,W,1,rd,grid).
        ring_atom = params.ring_store_atom
        gRing = cute.flat_divide(params.ring_store_tensor, epi_tile)  # (em,en,W,1,rd,grid)
        s_ring, g_ring = cpasync.tma_partition(
            ring_atom,
            0,
            cute.make_layout(1),
            cute.group_modes(sPostAct, 0, cute.rank(sPostAct) - 1),
            cute.group_modes(gRing, 0, 2),
        )
        bidx = cute.arch.block_idx()[0] + cute.arch.block_idx()[2]
        # Per-peer COUPLED views (P2P arm, byte-identical to the per-subtile / coupled store).
        p_atoms = params.postact_peer_atoms
        p_tensors = params.postact_peer_tensors
        s_pviews, g_pviews = [], []
        for r in range(cp):
            gD = cute.flat_divide(p_tensors[r], epi_tile)
            s_pr, g_pr = cpasync.tma_partition(
                p_atoms[r],
                0,
                cute.make_layout(1),
                cute.group_modes(sPostAct, 0, cute.rank(sPostAct) - 1),
                cute.group_modes(gD, 0, 2),
            )
            s_pviews.append(s_pr)
            g_pviews.append(g_pr)
        is_p2p = self._a2a_is_p2p

        @cute.jit
        def copy_fn(src_idx, dst_idx, **kwargs):
            sub_m, sub_n = dst_idx[0], dst_idx[1]
            n_tile = tile_coord_mnkl[1]
            half = n_tile // Int32(n_tiles_per_half)
            n_in_half = n_tile % Int32(n_tiles_per_half)
            peer = n_in_half // Int32(n_tiles_per_Dslice)
            n_in_slice = n_in_half % Int32(n_tiles_per_Dslice)
            # dst_tok_sub = subtile index along the recv STRIDE-1 axis (token D-major / N_i route2_ni); dst_j
            # = the N_j grid column (route2_ni 3rd store coord; the wide batch is pinned to ONE dst_j).
            if const_expr(route2_ni):
                if const_expr(b_mode):
                    # HOISTED above: the plane is peeled off the flat j-tile, so `_dst_j` is the LOCAL
                    # j column WITHIN this plane and only `sub_m` remains per-subtile.
                    dst_tok_sub = _dst_i_tile + sub_m
                    dst_j = _dst_j
                elif const_expr(self._a2a_dynamic):
                    tile_x = tile_coord_mnkl[0] % n_x_r2_dyn
                    tile_y = tile_coord_mnkl[0] // n_x_r2_dyn
                    dst_tok_sub = (Int32(cp0_coord) * n_x_r2_dyn + tile_x) * Int32(
                        m_sub_per_tile
                    ) + sub_m
                    dst_j = j_off_r2_dyn + tile_y
                else:
                    tile_x = tile_coord_mnkl[0] % Int32(n_x_r2)
                    tile_y = tile_coord_mnkl[0] // Int32(n_x_r2)
                    dst_tok_sub = (Int32(cp0_coord * n_x_r2) + tile_x) * Int32(
                        m_sub_per_tile
                    ) + sub_m
                    dst_j = Int32(j_off_r2) + tile_y
                run_tokens = Int32(epi_m)  # zero-tail padded i-shard -> full subtile always
            else:
                if const_expr(b_mode):
                    # HOISTED above: the plane is peeled out of the flat M-tile, leaving the PER-PLANE
                    # token tile. ABSOLUTE (my_col_sub_off already folded in) -- b-mode and the partial-
                    # token clamp are mutually exclusive, refused at the recv build, so there is no
                    # relative form to choose here. `dst_tok_local` drives the run clamp against the
                    # PER-PLANE rows_per_peer, which is what makes the clamp exact under b-mode.
                    dst_tok_sub = _dst_tok_tile + sub_m
                    dst_tok_local = dst_tok_sub - Int32(my_col_sub_off)
                elif const_expr(self._a2a_should_clamp()):
                    dst_tok_sub = (
                        tile_coord_mnkl[0] * Int32(m_sub_per_tile) + sub_m
                    )  # RELATIVE to my block
                    dst_tok_local = dst_tok_sub
                else:
                    dst_tok_sub = (
                        Int32(my_col_sub_off) + tile_coord_mnkl[0] * Int32(m_sub_per_tile) + sub_m
                    )
                    dst_tok_local = dst_tok_sub - Int32(my_col_sub_off)
                rem_tok = Int32(rows_per_peer) - dst_tok_local * Int32(epi_m)
                rem_tok = Int32(epi_m) if rem_tok > Int32(epi_m) else rem_tok
                run_tokens = Int32(0) if rem_tok < Int32(0) else rem_tok
                dst_j = Int32(
                    0
                )  # D-major has no N_j grid coord (unused; keeps dst_j defined below)
            dst_feat_sub = half * Int32(dloc_sub) + n_in_slice * Int32(n_sub_per_tile) + sub_n
            # PEER-STORE COORD for the P2P (coupled TMA-S2G) arm, assembled ONCE so the arms cannot
            # drift: box + the recv stride-1 tile (token D-major / N_i route2_ni) + the feature tile,
            # then route2_ni's N_j grid column, then b-mode's batch PLANE. Same coordinate the coupled
            # store builds -- this arm writes the SAME descriptor, so it needs the SAME rank. All host-
            # side `const_expr`, so a non-route2 non-b build assembles exactly `(None, dst_tok_sub,
            # dst_feat_sub)` and traces the identical slice op it did before.
            p_crd = (None, dst_tok_sub, dst_feat_sub)
            if const_expr(route2_ni):
                p_crd = p_crd + (dst_j,)
            if const_expr(b_mode):
                p_crd = p_crd + (_tile_b,)
            lane = cute.arch.lane_idx()
            # is_ib = NOT is_p2p[peer], as Int32 1/0 (const_expr-unrolled ternary-select; the meta_peer
            # idiom). A P2P subtile (is_ib==0) never coalesces -> its own size-1 skip batch.
            is_ib = Int32(1)
            for r in cutlass.range_constexpr(cp):
                if const_expr(is_p2p[r]):
                    is_ib = Int32(0) if (peer == Int32(r)) else is_ib
            # Read the pending batch state (all lanes; lane-0's prior writeback visible via the trailing
            # sync_warp of the previous call — the SAME single-writer pattern as pcount). All lanes track
            # bc/b_w/... uniformly in registers -> the collective arrive/wait branches stay in lockstep.
            bc = bstate[0]
            b_w = bstate[1]
            b_base = bstate[2]
            b_peer = bstate[3]
            b_feat = bstate[4]
            b_run = bstate[5]
            b_dup = bstate[
                6
            ]  # run_i clamped-padding re-stores absorbed into this batch (the perf fix)
            b_j = (
                bstate[7] if route2_ni else Int32(0)
            )  # route2_ni: the N_j column the batch is pinned to
            b_b = (
                bstate[bb_idx] if b_mode else Int32(0)
            )  # b-mode: the recv batch PLANE the batch is pinned to
            # merge: does THIS subtile CONTINUE the pending IB batch? consecutive dst_tok, same peer+feat,
            # room left (b_w<W). A P2P subtile (is_ib==0) never merges; a P2P pending batch (b_peer==-1)
            # never matches an IB peer(>=0). Uniform Boolean across lanes.
            merge = (
                (b_w > Int32(0))
                and (is_ib == Int32(1))
                and (peer == b_peer)
                and (dst_feat_sub == b_feat)
                and (dst_tok_sub == b_base + b_w)
                and (b_w < Int32(W))
            )
            # run_i DUPLICATE-ABSORB (the R∤ncluster_m perf fix): a clamped-padding re-store maps to a
            # dst_tok ALREADY covered by the pending batch's [b_base, b_base+b_w) range at the SAME
            # peer+feat. WITHOUT this it fails `merge` -> break-flushes as its own tiny 256 B put (the drain
            # FLOOD that made run_i a LOSS). ABSORB it instead: count it in b_dup (for count-parity — the
            # consumer's my_tiles·epi_tile_num target INCLUDES these clamped tiles) but do NO put + NO
            # b_w/ring advance (its token was already staged by the real tile it clamps onto). The DEFAULT /
            # no-run_i raster never repeats a dst_tok -> is_dup is ALWAYS False -> byte-identical. Uniform.
            is_dup = (
                (b_w > Int32(0))
                and (is_ib == Int32(1))
                and (peer == b_peer)
                and (dst_feat_sub == b_feat)
                and (dst_tok_sub >= b_base)
                and (dst_tok_sub < b_base + b_w)
            )
            # route2_ni: the wide coalesce runs along N_i WITHIN one N_j column, so a subtile only
            # continues / duplicates the batch when it lands on the SAME dst_j. const_expr-gated so the
            # D-major merge/is_dup are byte-identical (the reassign is pruned when route2_ni is off).
            if const_expr(route2_ni):
                merge = merge and (dst_j == b_j)
                is_dup = is_dup and (dst_j == b_j)
            # B-MODE: a wide put is ONE contiguous run along the recv stride-1 axis, and the planes are
            # NOT adjacent along it -- so a batch may only coalesce WITHIN one plane. Both keys need the
            # test and they must change TOGETHER:
            #   * `merge` alone would let one put span two planes. Under composite_k the plane stride
            #     EQUALS the token extent, so a cross-plane run is perfectly contiguous and in bounds --
            #     no clamp, no OOB, no length mismatch. It would silently overwrite plane b's tail with
            #     plane b+1's bytes and nothing downstream could notice.
            #   * `is_dup` alone is WORSE, and fixing only `merge` converts one into the other: two
            #     subtiles from different planes at the same WITHIN-plane tile index are identical in
            #     every field `is_dup` inspects (peer, feat, dst_tok range, dst_j), so the second is
            #     ABSORBED -- counted into n_sub for parity and never put. Plane b+1's data is dropped,
            #     count-parity still balances, and the drain neither hangs nor warns.
            # Today each is arguably unreachable by a raster accident (a plane change shows up as a
            # BACKWARD token jump for composite, and always moves dst_j for route2), but both are
            # properties of the walk, and the run_i R-derivation is exactly the kind of thing that gets
            # retuned. Correctness by construction, not by raster.
            if const_expr(b_mode):
                merge = merge and (_tile_b == b_b)
                is_dup = is_dup and (_tile_b == b_b)
            # SINGLE FLUSH POINT: a pending batch this subtile does NOT continue is COMPLETE -> flush it
            # (stamp per-batch meta + arrive full[bc%rd], whole warp -> count 32). A full batch flushes
            # here on the NEXT subtile (merge False via b_w==W). A duplicate SUPPRESSES the flush.
            if (b_w > Int32(0)) and (not merge) and (not is_dup):
                if lane == Int32(0):
                    mbase = (bc % Int32(rd)) * Int32(nfields)
                    meta[mbase + Int32(0)] = b_peer  # -1 => P2P skip batch (consumer does no put)
                    meta[mbase + Int32(1)] = b_base  # batch base dst_tok (epi_m units)
                    meta[mbase + Int32(2)] = b_feat  # dst_feat (epi_n units)
                    meta[mbase + Int32(3)] = b_run  # total VALID token run = the wide put length
                    meta[mbase + Int32(4)] = (
                        b_w + b_dup
                    )  # n_sub (real + absorbed dups) -> count-parity
                    if const_expr(route2_ni):
                        meta[mbase + Int32(5)] = (
                            b_j  # the N_j column this batch's wide put lands in
                        )
                    if const_expr(b_mode):
                        meta[mbase + Int32(mb_idx)] = b_b  # the recv PLANE this wide put lands in
                cute.arch.sync_warp()
                for ss in cutlass.range_constexpr(rd):
                    if (bc % Int32(rd)) == Int32(ss):
                        cute.arch.mbarrier_arrive(full_ptr + ss)
                bc = bc + Int32(1)
                b_w = Int32(0)
                b_dup = Int32(0)  # the flushed batch's dups are stamped -> reset for the next batch
            # START a new batch on this subtile if none is pending. empty-wait the slot's prior occupant
            # (rd batches ago) -> WAR-safe reuse (spin try_wait; gated bc>=rd). All lanes spin (uniform).
            # A duplicate keeps b_w>0 so it never starts a batch.
            if b_w == Int32(0):
                kk = bc // Int32(rd)
                if kk >= Int32(1):
                    for ss in cutlass.range_constexpr(rd):
                        if (bc % Int32(rd)) == Int32(ss):
                            done = cute.arch.mbarrier_try_wait(
                                empty_ptr + ss, (kk - Int32(1)) & Int32(1)
                            )
                            while not done:
                                done = cute.arch.mbarrier_try_wait(
                                    empty_ptr + ss, (kk - Int32(1)) & Int32(1)
                                )
                b_base = dst_tok_sub
                b_peer = peer if (is_ib == Int32(1)) else Int32(-1)
                b_feat = dst_feat_sub
                b_run = Int32(0)
                b_dup = Int32(0)  # a fresh batch has no absorbed dups yet
                if const_expr(route2_ni):
                    b_j = dst_j  # pin the batch to THIS subtile's N_j column (the coalesce is intra-column)
                if const_expr(b_mode):
                    b_b = _tile_b  # pin the batch to THIS subtile's recv plane (coalesce is intra-plane)
            # ADD this subtile at token-tile position b_w — SKIPPED for a duplicate (token already staged).
            # IB -> async TMA-S2G into the wide ring slot at token offset b_w·epi_m; P2P -> coupled TMA-S2G
            # DIRECTLY to the peer recv (no ring). One commit+wait(read=False) covers whichever arm fired ->
            # full completion before the flush's arrive (consumer reads the ring / SMEM reuse is safe).
            if not is_dup:
                with cute.arch.elect_one():
                    if is_ib == Int32(1):
                        for ss in cutlass.range_constexpr(rd):
                            if (bc % Int32(rd)) == Int32(ss):
                                cute.copy(
                                    ring_atom,
                                    s_ring[(None, src_idx)],
                                    g_ring[(None, b_w, Int32(0), Int32(ss), bidx)],
                                )
                    else:
                        for r in cutlass.range_constexpr(cp):
                            if peer == Int32(r):
                                if const_expr(is_p2p[r]):
                                    # `p_crd` above: (box, dst_i/dst_tok, dst_feat)[, dst_j][, tile_b].
                                    cute.copy(
                                        p_atoms[r],
                                        s_pviews[r][(None, src_idx)],
                                        g_pviews[r][p_crd],
                                    )
                    cute.arch.cp_async_bulk_commit_group()
                    cute.arch.cp_async_bulk_wait_group(0, read=False)
                cute.arch.sync_warp()
                # Extend the batch's valid run (b_w is the position BEFORE increment). Only a valid IB
                # subtile (run>0) advances b_run; a fully-OOB subtile (run==0) is COUNTED (n_sub) but not
                # drained; a P2P subtile leaves b_run at 0 (its batch is skip-only, meta[3] ignored @ -1).
                if (is_ib == Int32(1)) and (run_tokens > Int32(0)):
                    b_run = b_w * Int32(epi_m) + run_tokens
                b_w = b_w + Int32(1)
            else:
                b_dup = b_dup + Int32(
                    1
                )  # absorbed clamped-padding re-store: count for parity, no put
            # Write the batch state back (lane-0; visible to the next call's all-lane read via sync_warp).
            if lane == Int32(0):
                bstate[0] = bc
                bstate[1] = b_w
                bstate[2] = b_base
                bstate[3] = b_peer
                bstate[4] = b_feat
                bstate[5] = b_run
                bstate[6] = b_dup
                if const_expr(route2_ni):
                    bstate[7] = b_j  # the pending batch's pinned N_j column
                if const_expr(b_mode):
                    bstate[bb_idx] = b_b  # the pending batch's pinned recv plane
            cute.arch.sync_warp()

        return copy_fn

    def _a2a_wide_producer_finalize(self, storage) -> None:
        """Flush the CTA's TRAILING wide batch after the MMA tile loop (plan §6.4, CAVEAT B). Called from
        the staged kernel's post-loop ``if is_tma_warp:`` block -> runs on WARP 0 (the SAME producer warp
        that owns bstate). PLAIN method (the storage struct cannot cross a @cute.jit boundary): pull the
        SMEM pointers, hand them to the @cute.jit flush. No-op unless the wide A2A path is live."""
        if const_expr(
            not (
                self._a2a_enabled
                and self._decoupled_active()
                and getattr(self, "_a2a_ib_wide", False)
            )
        ):
            return
        rd: cutlass.Constexpr[int] = self._a2a_ring_depth
        nfields: cutlass.Constexpr[int] = self._DECOUPLED_META_FIELDS
        self._a2a_wide_flush_trailing(
            storage.decoupled.full.data_ptr(),
            storage.decoupled.meta.get_tensor((rd * nfields,)),
            storage.decoupled.bstate.get_tensor((self._DECOUPLED_BSTATE_FIELDS,)),
        )

    @cute.jit
    def _a2a_wide_flush_trailing(self, full_ptr, meta, bstate) -> None:
        """Trailing-batch flush (warp 0, all 32 lanes -> full[bc%rd] count 32). If a pending batch exists
        (b_w>0) stamp its per-batch meta (lane 0) + arrive full[bc%rd]; else no-op (the CTA had no pending
        batch — e.g. my_tiles==0, or the last subtile exactly filled+flushed a batch).

        The stamped record is the SAME shape the in-loop flush writes, optional fields included: route2_ni's
        ``b_j`` and b-mode's ``b_b`` (the recv PLANE). Both must be written here too -- this is a real
        batch that a consumer will drain, and a trailing batch missing its plane would put the CTA's LAST
        wide run onto plane 0 while every earlier run landed correctly, which is the hardest version of
        this bug to see: correct almost everywhere.

        Args:
            full_ptr: base of the ring's ``full`` mbarrier array (rd entries); the whole warp arrives on
                ``full[bc % rd]``, so all 32 lanes must reach this call.
            meta: the ``(rd * _DECOUPLED_META_FIELDS,)`` SMEM int32 metadata tensor. Written by lane 0
                only, then made visible by the ``sync_warp`` before the arrive.
            bstate: the ``(_DECOUPLED_BSTATE_FIELDS,)`` SMEM int32 batch state this CTA's producer warp
                has been accumulating. Read by all lanes; must be the SAME tensor the producer wrote.

        Returns:
            None. Emits the meta stamp and the mbarrier arrive as side effects.
        """
        rd: cutlass.Constexpr[int] = self._a2a_ring_depth
        nfields: cutlass.Constexpr[int] = self._DECOUPLED_META_FIELDS
        route2_ni: cutlass.Constexpr[bool] = bool(getattr(self, "_a2a_route2_ni", False))
        b_mode: cutlass.Constexpr[bool] = bool(self._a2a_b_mode())
        mb_idx: cutlass.Constexpr[int] = self._DECOUPLED_META_B_INDEX  # plane field; b_mode only
        bb_idx: cutlass.Constexpr[int] = self._DECOUPLED_BSTATE_B_INDEX  # plane slot; b_mode only
        lane = cute.arch.lane_idx()
        bc = bstate[0]
        b_w = bstate[1]
        b_base = bstate[2]
        b_peer = bstate[3]
        b_feat = bstate[4]
        b_run = bstate[5]
        b_dup = bstate[6]  # run_i absorbed clamped-padding re-stores (count-parity, no extra put)
        b_j = (
            bstate[7] if route2_ni else Int32(0)
        )  # route2_ni: the trailing batch's pinned N_j column
        b_b = (
            bstate[bb_idx] if b_mode else Int32(0)
        )  # b-mode: the trailing batch's pinned recv PLANE
        if b_w > Int32(0):
            if lane == Int32(0):
                mbase = (bc % Int32(rd)) * Int32(nfields)
                meta[mbase + Int32(0)] = b_peer
                meta[mbase + Int32(1)] = b_base
                meta[mbase + Int32(2)] = b_feat
                meta[mbase + Int32(3)] = b_run
                meta[mbase + Int32(4)] = b_w + b_dup  # n_sub = real + absorbed dups (count-parity)
                if const_expr(route2_ni):
                    meta[mbase + Int32(5)] = b_j  # the N_j column this trailing wide put lands in
                if const_expr(b_mode):
                    meta[mbase + Int32(mb_idx)] = b_b  # the PLANE this trailing wide put lands in
            cute.arch.sync_warp()
            for ss in cutlass.range_constexpr(rd):
                if (bc % Int32(rd)) == Int32(ss):
                    cute.arch.mbarrier_arrive(full_ptr + ss)
            cute.arch.sync_warp()

    def consumer_warpgroup_role(self, warp_idx, storage, epilogue_params, tile_sched_params):
        """The dedicated consumer warpgroup's per-CTA drain: LOCAL-GMEM ring -> peer recv via per-row
        put_nbi_warp (routed by RUNTIME PE -> IB-swappable). PLAIN method (the storage struct cannot
        cross a @cute.jit boundary): extract the SMEM pointers + build the drain views (host-trace
        tensor algebra), then hand the flatten-able DSL values to the @cute.jit drain loop."""
        if const_expr(not (self._a2a_enabled and self._decoupled_active())):
            return
        cp: cutlass.Constexpr[int] = self._a2a_cp
        # GLU-HALVED postact geometry: the postact STSM box is (epi_m, epi_n_postact = epi_n_D//2) and
        # the epilogue calls copy_postact (the producer) (tile_m//epi_m)*(BLK_N//epi_n_D) times per
        # CTA-tile (gemm_sm90.py:1427, using the FULL D epi_tile + full BLK_N). The consumer must walk
        # the SAME slot count + the SAME (epi_m, epi_n_postact) box (else it drains the wrong #slots /
        # wrong box). epi_n_postact == params.epi_tile_mPostAct[1] == self.epi_tile[1]//2.
        epi_m: cutlass.Constexpr[int] = self.epi_tile[0]
        epi_n_D: cutlass.Constexpr[int] = self.epi_tile[1]
        epi_n: cutlass.Constexpr[int] = epi_n_D // 2  # postact (glu-halved) feature box width
        tile_m: cutlass.Constexpr[int] = self.cta_tile_shape_mnk[0]
        tile_n_full: cutlass.Constexpr[int] = self.cta_tile_shape_mnk[
            1
        ]  # GEMM BLK_N (full, un-halved)
        epi_tile_num: cutlass.Constexpr[int] = (tile_m // epi_m) * (tile_n_full // epi_n_D)
        total_clusters = Int32(cute.size(tile_sched_params.problem_shape_ncluster_mnl))
        rd: cutlass.Constexpr[int] = self._a2a_ring_depth
        nfields: cutlass.Constexpr[int] = self._DECOUPLED_META_FIELDS
        full_ptr = storage.decoupled.full.data_ptr()
        empty_ptr = storage.decoupled.empty.data_ptr()
        meta = storage.decoupled.meta.get_tensor((rd * nfields,))
        # The CTA's ring sub-view (rd, epi_m_token, epi_n_feat) — the producer-TMA wrote each slot as
        # the (epi_m, epi_n) box; the underlying GMEM is token-innermost (epi_m stride-1).
        ring = epilogue_params.decoupled_ring  # (grid_CTAs, rd, epi_n_feat, epi_m_token)
        bidx = cute.arch.block_idx()[0] + cute.arch.block_idx()[2]
        ring_cta = ring[bidx, None, None, None]  # (rd, epi_n_feat, epi_m_token)
        recv_local = epilogue_params.recv_local  # (M_full_token, 2*Dloc_feat), token stride-1
        # Recast to int16 for the put_nbi_warp wire-in (bf16 -> int16 same bits).
        if const_expr(ring.element_type is cutlass.BFloat16):
            ring_cta_put = cute.recast_tensor(ring_cta, cutlass.Int16)
            recv_put = cute.recast_tensor(recv_local, cutlass.Int16)
        else:
            ring_cta_put = ring_cta
            recv_put = recv_local
        # flat_divide the recv (token, feature) by the (epi_m_token, epi_n_feat) box -> the consumer
        # indexes (None, None, tok_tile, feat_tile) to land the box; per-feature-row put is a token run.
        g_box = cute.flat_divide(recv_put, (epi_m, epi_n))  # (epi_m, epi_n, nt_tok, nt_feat)
        # WIDE-PUT dispatch: the wide drain walks per-BATCH (loop until subtiles_drained==total_subtiles
        # summing meta[4]=n_sub), each batch = one (total_run≤W·epi_m)-token contiguous put per feature
        # row. Same views (ring_cta_put now (rd,epi_n,W·epi_m); g_box unchanged). Default-off -> the
        # per-subtile drain below, BYTE-IDENTICAL.
        if const_expr(getattr(self, "_a2a_ib_wide", False)):
            # run_i count-parity: the wide drain target must reflect the run_i per-CTA walk (see the
            # drain loop's my_tiles). STATIC run_i_tiles>0 -> total_runs (runtime) + R (const); DYNAMIC-N
            # run_i (D-major arbitrary-R / route2 divisor-snap) -> run_i_dynamic flag + run_i_len (runtime
            # R). run_i off -> total_runs 0/unused (const_expr-elided) -> the default walk.
            run_i_tiles: cutlass.Constexpr[int] = const_expr(
                getattr(tile_sched_params, "run_i_tiles", 0) or 0
            )
            run_i_dynamic: cutlass.Constexpr[bool] = const_expr(
                bool(getattr(tile_sched_params, "run_i_dynamic", False))
            )
            run_i_len = tile_sched_params.run_i_len if const_expr(run_i_dynamic) else Int32(0)
            self._a2a_wide_front_drain_loop(
                warp_idx,
                ring_cta_put,
                g_box,
                full_ptr,
                empty_ptr,
                meta,
                epilogue_params.pe_table_dev,
                total_clusters,
                Int32(epi_tile_num),
                tile_sched_params.total_runs,
                run_i_tiles,
                run_i_dynamic,
                run_i_len,
            )
        else:
            self._decoupled_front_drain_loop(
                warp_idx,
                ring_cta_put,
                g_box,
                full_ptr,
                empty_ptr,
                meta,
                epilogue_params.pe_table_dev,
                total_clusters,
                Int32(epi_tile_num),
            )

    @cute.jit
    def _decoupled_front_drain_loop(
        self,
        warp_idx,
        ring_cta,
        g_box,
        full_ptr,
        empty_ptr,
        meta,
        pe_table_dev,
        total_clusters,
        epi_tile_num,
    ):
        """Runtime drain: per ring slot, read meta (peer, dst_tok_sub, dst_feat_sub, run), then for each
        FEATURE subrow [0,epi_n) put a contiguous (run,) TOKEN run ring_slot[f,:run] -> recv[dst_tok:
        dst_tok+run, dst_feat+f] routed by pe_table_dev[peer] (recv addressed by MY column block; dst_tok
        RELATIVE under auto-clamp). run = the subtile's valid token count (§2): epi_m for a full subtile,
        rows_per_peer%epi_m for the partial tail (a multiple of 8 => run*2 B a multiple of 16 B), 0 for a
        fully-OOB subtile. The token run is stride-1 on BOTH ends (ring token-innermost; recv token =
        stride-1 col) -> 16-B-aligned wide put (a full run epi_m=128 bf16 = 256 B). Warp-strided over the
        epi_n feature rows.

        ib_drain DIFFERENTIAL: a slot's meta peer == -1 is the P2P SKIP sentinel (that subtile was stored
        coupled TMA-S2G by the producer, NOT staged) OR run==0 is a fully-OOB subtile -> the consumer does
        NO put but STILL arrives empty[slot] to keep the uniform rotation/handshake count (the #1 deadlock
        trap). An IB slot (peer>=0, run>0) drains via the BLOCKING ``_put_warp_int16_a16`` (IBGDA-quiets ->
        in-kernel completion + backpressure, the cp16 QP-exhaustion fix). §10: the plain-decoupled non-
        blocking arm is retired -> ib_drain is the only store mode here (the primitive itself is kept)."""
        cp: cutlass.Constexpr[int] = self._a2a_cp
        ib_drain: cutlass.Constexpr[bool] = bool(self._a2a_ib_drain)
        route2_ni: cutlass.Constexpr[bool] = bool(getattr(self, "_a2a_route2_ni", False))
        rd: cutlass.Constexpr[int] = self._a2a_ring_depth
        nfields: cutlass.Constexpr[int] = self._DECOUPLED_META_FIELDS
        # B-MODE: the recv gains a trailing batch PLANE mode, so `recv_local` (== `atom_recv`) is
        # ALREADY plane-carrying and this loop's `flat_divide` therefore has one MORE grid mode than on
        # a non-b build. `b_stride_mode` names that mode: g_box is (epi_m, epi_n, nt_tok, nt_feat[, N_j],
        # B), so the plane is mode 4 on the D-major recv and mode 5 on route2_ni's 3-D one. All
        # `const_expr` -> pruned whole on a non-b build (byte-identical drain).
        b_mode: cutlass.Constexpr[bool] = bool(self._a2a_b_mode())
        mb_idx: cutlass.Constexpr[int] = self._DECOUPLED_META_B_INDEX  # plane field; b_mode only
        b_stride_mode: cutlass.Constexpr[int] = 5 if route2_ni else 4
        epi_m: cutlass.Constexpr[int] = self.epi_tile[0]
        epi_n: cutlass.Constexpr[int] = (
            self.epi_tile[1] // 2
        )  # postact (glu-halved) feature box width
        n_consumer_warps: cutlass.Constexpr[int] = self._DECOUPLED_CONSUMER_WARPS
        first_extra_warp: cutlass.Constexpr[int] = self._drain_first_warp()
        if warp_idx >= Int32(first_extra_warp):
            wid = warp_idx - Int32(first_extra_warp)  # consumer-warp index [0, n_consumer_warps)
            gz = Int32(cute.arch.grid_dim()[2])
            z = Int32(cute.arch.block_idx()[2])
            my_tiles = Int32(0)
            if z < total_clusters:
                my_tiles = (total_clusters - z + gz - Int32(1)) // gz  # ceil_div(total - z, gz)
            n_groups = my_tiles * epi_tile_num  # one slot per postact subtile (grp_sz==1)
            g = Int32(0)
            while g < n_groups:
                slot = g % Int32(rd)
                k = g // Int32(rd)
                # try_wait SPIN-LOOP (not blocking mbarrier_wait): a blocking wait can WEDGE here under
                # the producer-TMA-deposit ring + cross-warpgroup handshake (reference_mbarrier_wait_
                # hangs_tma_tx). Spin on the counted try_wait until the producer's full[slot] phase lands.
                for ss in cutlass.range_constexpr(rd):
                    if slot == Int32(ss):
                        done = cute.arch.mbarrier_try_wait(full_ptr + ss, k & Int32(1))
                        while not done:
                            done = cute.arch.mbarrier_try_wait(full_ptr + ss, k & Int32(1))
                base = slot * Int32(nfields)
                peer = meta[base + Int32(0)]
                dst_tok = meta[
                    base + Int32(1)
                ]  # token recv-col tile (epi_m units; RELATIVE under clamp)
                dst_feat = meta[base + Int32(2)]  # feature recv-row tile (epi_n units)
                run = meta[
                    base + Int32(3)
                ]  # valid token count of this subtile (§2; 0 => fully-OOB skip)
                # ib_drain: peer==-1 is the P2P SKIP sentinel (coupled store already wrote it) OR run==0 is
                # a fully-OOB subtile (a tile_M>epi_m partial tail) -> NO put, but STILL arrive empty[slot]
                # (below) so the rotation/handshake count stays uniform. peer/run are uniform meta reads ->
                # do_drain is warp-uniform (no sync_warp divergence).
                do_drain = True
                if const_expr(ib_drain):
                    do_drain = (peer >= Int32(0)) and (run > Int32(0))
                if do_drain:
                    pe = pe_table_dev[peer]  # runtime flat-cp -> global PE (branch-free)
                    # ring slot box (epi_n_feat, epi_m_stride1): index the feature row, take the run. epi_m
                    # is the recv STRIDE-1 axis in BOTH layouts (token D-major / N_i=i_loc route2_ni), so the
                    # ring (epi_m-innermost) drains a contiguous run on both ends without re-lay.
                    src_slot = ring_cta[slot, None, None]  # (epi_n_feat, epi_m_stride1)
                    # recv box: D-major g_box[(None,None,dst_tok,dst_feat)]; route2_ni g_box[(None,None,
                    # dst_i,dst_feat,dst_j)] — the 3-D recv gives g_box the extra N_j grid mode, indexed by
                    # the extra meta[4]=dst_j (the N_j column). dst_tok holds dst_i for route2_ni.
                    # B-MODE: the batch PLANE the producer stamped (meta's trailing field). Read
                    # before the coords below so both recv layouts can carry it.
                    tile_b = meta[base + Int32(mb_idx)] if b_mode else Int32(0)
                    if const_expr(route2_ni):
                        dst_j = meta[base + Int32(4)]
                        if const_expr(b_mode):
                            dst_box = g_box[(None, None, dst_tok, dst_feat, dst_j, tile_b)]
                        else:
                            dst_box = g_box[
                                (None, None, dst_tok, dst_feat, dst_j)
                            ]  # (epi_m_i, epi_n_feat)
                    elif const_expr(b_mode):
                        dst_box = g_box[(None, None, dst_tok, dst_feat, tile_b)]
                    else:
                        dst_box = g_box[
                            (None, None, dst_tok, dst_feat)
                        ]  # (epi_m_token, epi_n_feat)
                    f = wid  # warp wid owns feature rows {wid, wid+nwarps, ...} < epi_n
                    while f < Int32(epi_n):
                        # src: ring_slot[f, :] = (epi_m,) token-contiguous (epi_m stride-1).
                        sr0 = src_slot[(f, None)]  # (epi_m_token,) stride-1
                        # dst: recv[dst_tok box, dst_feat+f] = (epi_m,) token run (token stride-1 col).
                        # 64-BIT DESTINATION ADDRESS -- the fix for a ROOT-CAUSED illegal address.
                        # `g_box` is flat_divide(recv, (epi_m, epi_n)); its FEATURE-mode stride is
                        # `epi_n * M_full` ELEMENTS. `dst_feat` is an Int32 from `meta`, so with a
                        # STATIC token extent that stride is a compile-time constant and the product
                        # `dst_feat * epi_n * M_full` was formed in 32 bits -- it WRAPS once
                        # `(2*Dloc - epi_n) * M_full >= 2**31`, and the wrapped value became the put
                        # destination. Re-form the SAME offset with every term widened to Int64.
                        #
                        # Cross-node only, because the P2P arm reaches this recv through a TMA
                        # DESCRIPTOR (hardware 64-bit). `--dynamic` hid it by making the stride a
                        # runtime value, which is why the shipped profiling harness never saw it.
                        # Threshold measured on a razor pair 16 tokens apart: cp=2 D=256 N=2976
                        # (2,125,578,240, under) PASSES and N=2992 (2,148,495,360, over) FAULTED.
                        # `* 2` is elements -> bytes (the recv is recast to Int16 for the put).
                        _off_el = (
                            cutlass.Int64(dst_tok) * cutlass.Int64(g_box.layout.stride[2])
                            + cutlass.Int64(dst_feat) * cutlass.Int64(g_box.layout.stride[3])
                            + cutlass.Int64(f) * cutlass.Int64(g_box.layout.stride[1])
                        )
                        if const_expr(route2_ni):
                            # route2_ni's recv is 3-D, so g_box carries a FIFTH mode (the N_j column).
                            # Omitting it here would address the right feature row of the wrong column.
                            _off_el = _off_el + cutlass.Int64(dst_j) * cutlass.Int64(
                                g_box.layout.stride[4]
                            )
                        if const_expr(b_mode):
                            # The batch PLANE is g_box's LAST grid mode (`b_stride_mode`). THIS is the
                            # line the whole record widening exists for: omit it and every plane's put
                            # lands on plane 0 -- in bounds, correctly aligned, right feature row, right
                            # token run. Under composite_k the plane stride equals the token extent, so
                            # the wrong destination is contiguous with the right one; nothing downstream
                            # can distinguish it. Int64 for the same reason the terms above are.
                            _off_el = _off_el + cutlass.Int64(tile_b) * cutlass.Int64(
                                g_box.layout.stride[b_stride_mode]
                            )
                        dst_addr = g_box.iterator.toint() + _off_el * cutlass.Int64(2)
                        aligned = (
                            sr0.iterator.toint() % cutlass.Int64(16) == cutlass.Int64(0)
                        ) and (dst_addr % cutlass.Int64(16) == cutlass.Int64(0))
                        if aligned:
                            sp = cute.make_ptr(
                                cutlass.Int16,
                                sr0.iterator.toint(),
                                cute.AddressSpace.gmem,
                                assumed_align=16,
                            )
                            dp = cute.make_ptr(
                                cutlass.Int16,
                                dst_addr,
                                cute.AddressSpace.gmem,
                                assumed_align=16,
                            )
                            # run-CLAMPED put: only the subtile's VALID token prefix (run tokens, a
                            # multiple of 8 => run*2 B a multiple of 16 B) egresses; the partial tail's
                            # garbage [run, epi_m) stays in the LOCAL ring (never reaches the peer recv).
                            # At an aligned N run==epi_m -> byte-identical to the fixed-epi_m put.
                            sr = cute.make_tensor(sp, cute.make_layout((run,)))
                            dr = cute.make_tensor(dp, cute.make_layout((run,)))
                            # IB peer -> BLOCKING put (IBGDA-quiets -> in-kernel completion + backpressure,
                            # the cp16 QP-exhaustion fix). §10: the plain-decoupled non-blocking arm is
                            # retired -> ib_drain-only blocking put (the _put_nbi_warp_int16_a16 PRIMITIVE
                            # is KEPT for the wide-put B2 variant; only this store-mode branch is gone).
                            _put_warp_int16_a16(dr, sr, pe)
                        f += Int32(n_consumer_warps)
                cute.arch.sync_warp()
                for ss in cutlass.range_constexpr(rd):
                    if slot == Int32(ss):
                        cute.arch.mbarrier_arrive(empty_ptr + ss)
                g += Int32(1)

    @cute.jit
    def _a2a_wide_front_drain_loop(
        self,
        warp_idx,
        ring_cta,
        g_box,
        full_ptr,
        empty_ptr,
        meta,
        pe_table_dev,
        total_clusters,
        epi_tile_num,
        total_runs,
        run_i_tiles,
        run_i_dynamic=False,
        run_i_len=None,
    ):
        """WIDE-PUT drain (plan §3): walk per-BATCH (not per-subtile). Loop until subtiles_drained ==
        total_subtiles (== the producer's emitted subtile count == my_tiles·epi_tile_num), summing the
        per-batch subtile count meta[4]=n_sub. So arrive_empty fires EXACTLY once per producer flush ->
        producer-arrive_full == consumer-arrive_empty on ANY raster (§6.4 count-parity by construction).
        Each batch drains ONE (run≤W·epi_m)-token contiguous put per feature row (run = meta[3]=total_run;
        a SINGLE feature row -> no >128-ROW IBGDA hang, run·2B a multiple of 16 B).

        CAVEAT B (loop-terminating arrive_empty): arrive_empty is issued INSIDE the loop body BEFORE the
        ``sd += n_sub`` that trips the exit, so the FINAL (possibly partial) batch's empty-arrival always
        fires -> count-parity never breaks by one. A P2P skip batch (peer==-1) or an all-OOB batch
        (run==0) does NO put but STILL arrives empty (uniform handshake)."""
        cp: cutlass.Constexpr[int] = self._a2a_cp
        rd: cutlass.Constexpr[int] = self._a2a_ring_depth
        nfields: cutlass.Constexpr[int] = self._DECOUPLED_META_FIELDS
        route2_ni: cutlass.Constexpr[bool] = bool(getattr(self, "_a2a_route2_ni", False))
        # B-MODE: the recv gains a trailing batch PLANE mode, so `recv_local` (== `atom_recv`) is
        # ALREADY plane-carrying and this loop's `flat_divide` therefore has one MORE grid mode than on
        # a non-b build. `b_stride_mode` names that mode: g_box is (epi_m, epi_n, nt_tok, nt_feat[, N_j],
        # B), so the plane is mode 4 on the D-major recv and mode 5 on route2_ni's 3-D one. All
        # `const_expr` -> pruned whole on a non-b build (byte-identical drain).
        b_mode: cutlass.Constexpr[bool] = bool(self._a2a_b_mode())
        mb_idx: cutlass.Constexpr[int] = self._DECOUPLED_META_B_INDEX  # plane field; b_mode only
        b_stride_mode: cutlass.Constexpr[int] = 5 if route2_ni else 4
        # B2 PIPELINE (ib_wide_nbi): per-feature-row puts NON-BLOCKING, one trailing BLOCKING put per batch
        # to reap the qp. Hides the per-row put LATENCY that dominates a SMALL-run drain (route2_ni's
        # N_i-stride-1 recv caps run at the i-shard span -> ~1 KiB puts, deep in the IB latency knee; the
        # D-major 32 KiB run already amortizes it). Default OFF -> the B1 per-row blocking self-quiet.
        ib_wide_nbi: cutlass.Constexpr[bool] = bool(getattr(self, "_a2a_ib_wide_nbi", False))
        # #9 same-slot ring-reuse race fix: the IB put's NIC RDMA read of the ring slot is a SYSTEM-scope
        # agent that no .cta/.gpu fence (nor the DSL's .cta acq_rel / async proxy, which cover only the
        # same-CTA producer/consumer) orders against the producer's slot-reuse write. A fence_acq_rel_sys
        # before the empty release closes it (cp16 route2 sub-band: 0/10 hangs with, intermittent without).
        # Runtime-guarded (CPO_FRONT_WIDE_FENCE, default ON) ONLY so a paired fenced-vs-unfenced bench can
        # isolate the per-batch fence cost on one allocation; ships default-ON (correctness fence).
        wide_fence: cutlass.Constexpr[bool] = bool(
            getattr(self, "_a2a_wide_fence", os.environ.get("CPO_FRONT_WIDE_FENCE", "1") != "0")
        )
        epi_m: cutlass.Constexpr[int] = self.epi_tile[0]
        epi_n: cutlass.Constexpr[int] = (
            self.epi_tile[1] // 2
        )  # postact (glu-halved) feature box width
        n_consumer_warps: cutlass.Constexpr[int] = self._DECOUPLED_CONSUMER_WARPS
        first_extra_warp: cutlass.Constexpr[int] = self._drain_first_warp()
        if warp_idx >= Int32(first_extra_warp):
            wid = warp_idx - Int32(first_extra_warp)  # consumer-warp index [0, n_consumer_warps)
            gz = Int32(cute.arch.grid_dim()[2])
            z = Int32(cute.arch.block_idx()[2])
            # RUN_I count-parity: run_i changes the per-CTA persistent walk from the DEFAULT (CTA z does
            # tiles z, z+gz, … -> ceil((total_clusters-z)/gz) tiles) to R tiles per RUN over
            # ceil((total_runs-z)/gz) runs. It BOTH redistributes tiles across CTAs AND re-stores the
            # clamped padding tiles when R∤ncluster_m (the sub-band re-stores the last i-cluster; those
            # are is_valid work tiles the epilogue PRODUCER emits too). So the drain target MUST be the
            # run_i per-CTA tile count·R -- else target≠(producer emission) -> count mismatch -> DEADLOCK.
            # Σ_z (ceil((total_runs-z)/gz)·R) = total_runs·R = the producer's total emission (each valid
            # run = R tiles; Σ_z num_runs_z = total_runs). run_i OFF (run_i_tiles==0 AND run_i_dynamic
            # False, const_expr-elided) -> the default per-CTA walk, BYTE-IDENTICAL. STATIC bakes R (const
            # run_i_tiles); DYNAMIC-N sources R from run_i_len (runtime Int32, either variant) — the
            # count-parity Σ_z(ceil((total_runs-z)/gz)·R)=total_runs·R is raster-agnostic in R (const or runtime).
            my_tiles = Int32(0)
            if const_expr(run_i_tiles > 0 or run_i_dynamic):
                R_i = run_i_len if const_expr(run_i_dynamic) else Int32(run_i_tiles)
                if z < total_runs:
                    num_runs_z = (
                        total_runs - z + gz - Int32(1)
                    ) // gz  # ceil_div(total_runs - z, gz)
                    my_tiles = num_runs_z * Int32(R_i)
            else:
                if z < total_clusters:
                    my_tiles = (total_clusters - z + gz - Int32(1)) // gz  # ceil_div(total - z, gz)
            total_subtiles = (
                my_tiles * epi_tile_num
            )  # count-parity target (== producer emitted subtiles)
            sd = Int32(0)  # subtiles accounted (Σ n_sub) -> loop until == total_subtiles
            g = Int32(0)  # batch index -> slot rotation + full-phase
            while sd < total_subtiles:
                slot = g % Int32(rd)
                k = g // Int32(rd)
                # try_wait SPIN (not blocking mbarrier_wait — WEDGE trap; reference_mbarrier_wait_hangs_
                # tma_tx). Spin the counted try_wait until the producer's per-BATCH full[slot] phase lands.
                for ss in cutlass.range_constexpr(rd):
                    if slot == Int32(ss):
                        done = cute.arch.mbarrier_try_wait(full_ptr + ss, k & Int32(1))
                        while not done:
                            done = cute.arch.mbarrier_try_wait(full_ptr + ss, k & Int32(1))
                base = slot * Int32(nfields)
                peer = meta[
                    base + Int32(0)
                ]  # -1 => P2P skip batch (coupled store already wrote it)
                base_tok = meta[
                    base + Int32(1)
                ]  # batch base token tile (epi_m units; RELATIVE under clamp)
                dst_feat = meta[base + Int32(2)]  # feature recv-row tile (epi_n units)
                run = meta[
                    base + Int32(3)
                ]  # total VALID token run of the batch (the wide put length)
                n_sub = meta[base + Int32(4)]  # subtiles in this batch (count-parity accounting)
                # do_drain: an IB batch with valid tokens. peer==-1 (P2P skip) or run==0 (all-OOB) -> no
                # put but STILL arrive empty (uniform handshake). Uniform meta reads -> warp-uniform.
                do_drain = (peer >= Int32(0)) and (run > Int32(0))
                if do_drain:
                    pe = pe_table_dev[peer]  # runtime flat-cp -> global PE (branch-free)
                    src_slot = ring_cta[slot, None, None]  # (epi_n_feat, W·epi_m_stride1)
                    # recv box at the batch BASE tile along the recv STRIDE-1 axis (token D-major / N_i
                    # route2_ni): the (run,) layout below extends it contiguously across the whole wide run.
                    # route2_ni indexes the extra N_j grid column meta[5]=dst_j (base_tok holds base dst_i).
                    # B-MODE: the batch PLANE the producer pinned this whole wide batch to (meta's
                    # trailing field). One batch == one plane by the merge/is_dup keys above.
                    tile_b = meta[base + Int32(mb_idx)] if b_mode else Int32(0)
                    if const_expr(route2_ni):
                        dst_j = meta[base + Int32(5)]
                        if const_expr(b_mode):
                            dst_box = g_box[(None, None, base_tok, dst_feat, dst_j, tile_b)]
                        else:
                            dst_box = g_box[
                                (None, None, base_tok, dst_feat, dst_j)
                            ]  # (epi_m_i, epi_n_feat)
                    elif const_expr(b_mode):
                        dst_box = g_box[(None, None, base_tok, dst_feat, tile_b)]
                    else:
                        dst_box = g_box[
                            (None, None, base_tok, dst_feat)
                        ]  # (epi_m_token, epi_n_feat)
                    f = wid  # warp wid owns feature rows {wid, wid+nwarps, ...} < epi_n
                    while f < Int32(epi_n):
                        sr0 = src_slot[
                            (f, None)
                        ]  # (W·epi_m_token,) stride-1 (ring token-innermost)
                        # 64-BIT DESTINATION ADDRESS -- the fix for a ROOT-CAUSED illegal address.
                        # `g_box` is flat_divide(recv, (epi_m, epi_n)); its FEATURE-mode stride is
                        # `epi_n * M_full` ELEMENTS. `dst_feat` is an Int32 from `meta`, so with a
                        # STATIC token extent that stride is a compile-time constant and the product
                        # `dst_feat * epi_n * M_full` was formed in 32 bits -- it WRAPS once
                        # `(2*Dloc - epi_n) * M_full >= 2**31`, and the wrapped value became the put
                        # destination. Re-form the SAME offset with every term widened to Int64.
                        #
                        # Cross-node only, because the P2P arm reaches this recv through a TMA
                        # DESCRIPTOR (hardware 64-bit). `--dynamic` hid it by making the stride a
                        # runtime value, which is why the shipped profiling harness never saw it.
                        # Threshold measured on a razor pair 16 tokens apart: cp=2 D=256 N=2976
                        # (2,125,578,240, under) PASSES and N=2992 (2,148,495,360, over) FAULTED.
                        # `* 2` is elements -> bytes (the recv is recast to Int16 for the put).
                        _off_el = (
                            cutlass.Int64(base_tok) * cutlass.Int64(g_box.layout.stride[2])
                            + cutlass.Int64(dst_feat) * cutlass.Int64(g_box.layout.stride[3])
                            + cutlass.Int64(f) * cutlass.Int64(g_box.layout.stride[1])
                        )
                        if const_expr(route2_ni):
                            # route2_ni's recv is 3-D, so g_box carries a FIFTH mode (the N_j column).
                            # Omitting it here would address the right feature row of the wrong column.
                            _off_el = _off_el + cutlass.Int64(dst_j) * cutlass.Int64(
                                g_box.layout.stride[4]
                            )
                        if const_expr(b_mode):
                            # The batch PLANE is g_box's LAST grid mode (`b_stride_mode`). THIS is the
                            # line the whole record widening exists for: omit it and every plane's put
                            # lands on plane 0 -- in bounds, correctly aligned, right feature row, right
                            # token run. Under composite_k the plane stride equals the token extent, so
                            # the wrong destination is contiguous with the right one; nothing downstream
                            # can distinguish it. Int64 for the same reason the terms above are.
                            _off_el = _off_el + cutlass.Int64(tile_b) * cutlass.Int64(
                                g_box.layout.stride[b_stride_mode]
                            )
                        dst_addr = g_box.iterator.toint() + _off_el * cutlass.Int64(2)
                        aligned = (
                            sr0.iterator.toint() % cutlass.Int64(16) == cutlass.Int64(0)
                        ) and (dst_addr % cutlass.Int64(16) == cutlass.Int64(0))
                        if aligned:
                            sp = cute.make_ptr(
                                cutlass.Int16,
                                sr0.iterator.toint(),
                                cute.AddressSpace.gmem,
                                assumed_align=16,
                            )
                            dp = cute.make_ptr(
                                cutlass.Int16,
                                dst_addr,
                                cute.AddressSpace.gmem,
                                assumed_align=16,
                            )
                            # WIDE run: run = total_run up to W·epi_m -> ONE contiguous (≥32 KiB at W=128)
                            # put per feature row (a SINGLE 1-D row -> NOT the >128-ROW IBGDA blocking-put
                            # hang; run·2 B a multiple of 16 B).
                            sr = cute.make_tensor(sp, cute.make_layout((run,)))
                            dr = cute.make_tensor(dp, cute.make_layout((run,)))
                            if const_expr(ib_wide_nbi):
                                # B2 PIPELINE: NON-BLOCKING put for every feature row except the LAST one
                                # this warp owns, which is BLOCKING.
                                #
                                #
                                # It used to make only the LAST row a warp owns blocking, on the theory that
                                # that put's internal `ibgda_quiet(qp(pe))` "reaps ALL the prior nbi puts on
                                # the SAME qp". **That theory is FALSE and it silently corrupted data.**
                                # Measured, 16 ranks over IB, cp=16 D=256 N=1024 route2_ni, fence ON: B2
                                # mismatched the 2-kernel reshard reference in 6 of 6 runs -- 2 to 9 of 16
                                # ranks each time, 58-239 elements of 67 108 864, worst per-element ratio
                                # 26.9-43.3 -- while B1 (every put blocking) passed 8 of 8 on the SAME three
                                # mesh shapes. The wrong elements were a CONTIGUOUS run along the recv's
                                # stride-1 axis at one (feature, N_j), i.e. exactly one wide put's
                                # destination, holding other plausible bf16 values rather than the -99 fill:
                                # the slot was REFILLED by the producer while the NIC was still reading it.
                                # Since the only difference between the arms is whether the non-last puts
                                # self-quiet, the trailing blocking put demonstrably does NOT provide
                                # completion for them. The SOURCE-REUSE WAIT after this loop is what
                                # supplies it; the trailing blocking put below is KEPT because it is
                                # not the bug and still bounds the QP for its own row.
                                if (f + Int32(n_consumer_warps)) < Int32(epi_n):
                                    _put_nbi_warp_int16_a16(dr, sr, pe)
                                else:
                                    _put_warp_int16_a16(dr, sr, pe)
                            else:
                                # B1 BLOCKING put: self-quiets -> in-kernel completion + backpressure -> the
                                # ring slot is provably drained the instant the put returns -> arrive_empty
                                # below is immediately WAR-safe.
                                _put_warp_int16_a16(dr, sr, pe)
                        f += Int32(n_consumer_warps)
                    if const_expr(ib_wide_nbi):
                        # BATCH COMPLETION POINT -- the B2 fix. const_expr-pruned on B1, so the non-nbi
                        # cubin is untouched and the shipped ladder (which only ever runs B1) is unmoved.
                        #
                        # The rows above were posted non-blocking, so the NIC may still be READING this
                        # ring slot. `arrive_empty` below hands the slot back to the producer to refill
                        # `rd` batches later: a write-after-read on the slot, and the one that produced
                        # the measured corruption (6/6 runs, 2-9 of 16 ranks, a contiguous run of wrong
                        # elements at one (dst_feat, dst_j) holding other plausible bf16 values).
                        #
                        # A FENCE cannot do this job and neither can the trailing blocking put. The #9
                        # `fence_acq_rel_sys` below ORDERS the NIC read against the producer's write; it
                        # does not WAIT for it, and it was ON in every run that corrupted. The blocking
                        # put completes its OWN transfer only. `nvshmemx_flush_warp` is the vendor's
                        # primitive for exactly this: "wait until all source buffers used by preceding
                        # non-blocking puts ... are safe to reuse".
                        #
                        # Warp-COLLECTIVE, so it must be reached warp-uniformly: `do_drain` is derived
                        # from meta every lane reads, which is what makes this placement legal. Free on
                        # an all-NVLink job -- the vendor documents flush as a no-op on pure P2P.
                        _flush_warp_device()
                cute.arch.sync_warp()
                if const_expr(wide_fence):
                    # #9 do_drain-GATE (back-a2a race model): only a PUT batch has a NIC RDMA read to order;
                    # a no-put/skip batch (peer==-1 P2P-skip / run==0 all-OOB) has none, so the fence is
                    # unneeded there. Every put batch KEEPS the fence -> the 0/10-hang validation holds.
                    # Strict improvement (fewer fences, correctness-equivalent). The .sys scope + per-batch
                    # presence are irreducible: the NIC read is a SYSTEM-scope agent a .gpu/.cta fence can't
                    # order, and each empty-arrive gates a slot-reuse rd batches later (per-recycle ordering
                    # required). do_drain is defined at the batch top (peer>=0 and run>0).
                    #
                    # PERF: this fence is the cost of the ONLY correct cp16-IB drain -- the unfenced kernel
                    # DEADLOCKS (18-min hang -> 0/10 with fence), so it is NOT a valid faster baseline.
                    # Measured ~+7% ungated / ~+6% gated on the nbi champion @ N=1024 cp=2*8 D=256 (Dloc=16)
                    # (do_drain-gate recovered only ~21% MEASURED, not the ~47% the P2P-skip fraction analysis
                    # predicted -- the actual skip fraction / a per-fence fixed cost differ; still a strict
                    # improvement so it stays). ZERO cost on cp<=8 (NVLink -> coupled store, fence unreached)
                    # and on D-major (32 KiB puts swamp it). Scaling: fence_% ~ 1/(Dloc*W) and N_token-INDEPENDENT
                    # -> the ~6% is flat across N but concentrated at the Dloc=16 floor (shrinks ~1/Dloc as D
                    # grows) and only on route2/nbi small (~1 KiB) puts. do_drain-gate below drops the fence on
                    # no-put/NVLink-skip batches (a PURE-IB cp0=16 job has no P2P-skip -> the gate is a no-op).
                    # TODO(#9 perf, future -- recover the residual ~4% ONLY in the cp16-IB/route2/nbi/Dloc~16 corner):
                    #   1. RELEASE-ONLY fence: only the release edge is needed (the producer's empty-WAIT supplies
                    #      the acquire), so a fence_release_sys would ~halve the barrier. back-a2a: MARGINAL +
                    #      needs a new DSL primitive (the .sys cross-agent round-trip dominates, not acq/rel).
                    #   2. WIDER BATCHES (raise W): fence_% ~ 1/(Dloc*W), so a coarser drain raster / bigger W
                    #      claws back most of the residual here. RISKY: can regress coalescing/occupancy/the
                    #      run_i raster (#4/#8) -> needs its own re-validate + re-bench. Zero benefit outside
                    #      the cp16-IB/route2/nbi/small-Dloc corner.
                    if do_drain:
                        cute.arch.fence_acq_rel_sys()  # #9: order the put's NIC RDMA read before empty release
                for ss in cutlass.range_constexpr(rd):
                    if slot == Int32(ss):
                        cute.arch.mbarrier_arrive(empty_ptr + ss)
                sd = (
                    sd + n_sub
                )  # AFTER arrive_empty (CAVEAT B: the terminating batch still fires empty)
                g = g + Int32(1)
