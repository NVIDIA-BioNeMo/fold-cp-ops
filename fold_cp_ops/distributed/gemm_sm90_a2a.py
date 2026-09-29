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

# Based on the cute-dsl example:
# https://github.com/NVIDIA/cutlass/blob/main/examples/python/CuTeDSL/hopper/dense_gemm.py
# This file has been modified by NVIDIA CORPORATION & AFFILIATES.

"""T2.2 / #32 — back-A2A fused into the GemmSm90 D-store, as a THIN SUBCLASS.

``GemmSm90A2A`` extends the local ``fold_cp_ops.kernels.gemm_sm90.GemmSm90`` (the local kernel
stays UNTOUCHED — design doc §3.1) and overrides ONLY the D-store-build seam
(``build_D_copy_fn``) so the GEMM1 output is stored SMEM->PEER-symmetric-GMEM
(``put_signal_nbi_tma_peer``-style S2G TMA) instead of SMEM->local-GMEM —
fusing the BACK A2A (tri reshard ``S3 -> S1`` on the GEMM M/token axis) into the
store the kernel already does, with NO local GMEM round-trip and NO separate A2A
kernel launch. Flag-off (no ``configure_a2a*`` call) is byte-identical to the
local kernel (all A2A paths ``const_expr(self._a2a_enabled)``-gated).

WHAT WAS A WHOLE-FILE COPY IS NOW INHERITANCE.  Previously this file duplicated the
entire 2120-line ``GemmSm90`` (kernel/mma/load_AB/epilogue/_compute_stages/_make_*/
all epi_* hooks) just to splice the store inline + thread peer atoms as kernel args.
Now: the parent factored the store-build into the overridable ``build_D_copy_fn``
hook (the ONLY change to ``gemm_sm90.py``, a pure extract-method); this subclass
overrides that hook and inherits everything else.

PEER-ATOM ROUTING (the one hard piece) — front-consistent epi-params route, NO
kernel/__call__ override.  The ``cp`` peer-pinned S2G atoms (built host-side from
``get_peer_tensor``) must reach the in-kernel store.  The parent already carries an
``epilogue_params`` (built in ``__call__`` via ``epi_to_underlying_arguments``, an
overridable @cute.jit method) across the @jit -> @kernel region as a kernel arg.  So
— mirroring the FRONT (``dual_gated_gemm_staged_a2a.py``) — the PUBLIC composed class
``fold_cp_ops.distributed.gemm_a2a_epi.GemmA2ASm90`` (= ``GemmDefaultEpiMixin`` + this class)
owns the epi-params plumbing: it extends the mixin ``EpilogueArguments`` with a
``recv`` field, regenerates the composable ``EpilogueParams`` with two extra
``peer_atoms``/``peer_tensors`` fields, and overrides ``epi_to_underlying_arguments``
to build the peer atoms from ``args.recv`` (inside ``__call__``'s jit trace) and stash
them in the params.  THIS class owns the kernel-side seam: ``build_D_copy_fn`` returns
the peer-store copy_fn when ``self._a2a_enabled`` (reading the peer atoms off
``epi_params``), else ``super().build_D_copy_fn(...)`` (the byte-identical local store).
The caller passes the LOGICAL ``(M,N,L)`` GEMM view as ``mD`` (so the parent's
shape/scheduler/local-atom machinery is unchanged) and the real recv buffer (3-D plain
or 5-D gemm-native) via the ``recv`` epi-arg.  The peer-atom build + the per-peer
``tma_partition`` / ``cute.copy`` mechanics are the PROVEN T2.2/#32 store recipe, KEPT
VERBATIM from the prior copy.

REUSES (proven, do NOT re-derive): the T2.0d ``compile_gemm_with_bitcode`` (tvm-ffi
OFF + --link-libraries + library_init) compile route, the fork's alignment-preserving
``get_peer_tensor``, and the store recipe (``cute.flat_divide`` NOT zipped,
``tma_partition(group_modes(...))``, const_expr-unrolled per-peer select, NO
``cute.printf`` in the epilogue — deadlocks under nvshmem).
"""

import math
import os

import cutlass
import cutlass.cute as cute
from cutlass import Int32, const_expr
from cutlass.cute.nvgpu import cpasync

import fold_cp_ops._internal.sm90_utils as fold_cp_ops_sm90_utils
from fold_cp_ops.kernels.gemm_sm90 import GemmSm90
from fold_cp_ops.distributed.layout_map import LayoutRightMap

# nvshmem device + peer-translation helper for the back-A2A peer TMA store
# (import lazily-tolerant so this module still imports without nvshmem; configure_a2a
# raises clearly if used there). get_peer_tensor is VENDORED into fold_cp_ops
# (fold_cp_ops/distributed/nvshmem_utils.py, ported from the CP fork) and is alignment-PRESERVING (the
# upstream nvshmem.core.device.cute.mem.get_peer_tensor strips assumed_align -> breaks
# the TMA 16 B-min alignment); see fold_cp_ops/distributed/nvshmem_utils.py.
try:
    import nvshmem.core  # noqa: F401  (bare import = availability probe for HAS_NVSHMEM)
    import nvshmem.core.device.cute.rma as nvshmem_cute_rma  # §7.16 strided drain warp put
    from fold_cp_ops.distributed.nvshmem_utils import get_peer_tensor as _get_peer_tensor_aligned

    HAS_NVSHMEM = True
except ImportError:
    HAS_NVSHMEM = False
    nvshmem_cute_rma = None

# §7.16b — a custom align<16> int16 warp-put FFI for the strided putwarp drain. The stock
# nvshmem.core.device.cute.rma.put_nbi_warp builds its FFI prototype from _CutePtrType(Int16)
# with the dtype-DEFAULT align<2> (rendered no-align). When the drain stamps its (provably
# 16-B-aligned) ring/recv row pointers assumed_align=16, the call type ptr<i16,align<16>>
# does NOT match the prototype ptr<i16,align<2>> -> "External prototype types mismatch". Rather
# than DOWN-stamp the call to align<2> (which works but caps the device-side vectorization the
# linked bitcode can emit), we REBUILD the FFI prototype WITH align<16> so the wide-aligned call
# matches AND the align<16> hint can propagate into the inlined bitcode put (wider GMEM->SYMMEM
# vector stores). Same extern symbol (nvshmemx_int16_put_nbi_warp); only the caller-side pointer
# alignment annotation differs. Built lazily (needs the cute MLIR context); guarded by HAS_NVSHMEM.
_int16_put_nbi_warp_a16 = None
if HAS_NVSHMEM:
    try:
        from nvshmem.bindings.device.cute._cuteast import _CutePtrType as _NvshmemCutePtrType

        _int16_put_nbi_warp_a16 = cute.ffi(
            name="nvshmemx_int16_put_nbi_warp",
            params_types=[
                _NvshmemCutePtrType(cutlass.Int16, alignment=16),
                _NvshmemCutePtrType(cutlass.Int16, alignment=16),
                cutlass.Uint64,
                cutlass.Int32,
            ],
        )
    except Exception:
        _int16_put_nbi_warp_a16 = None


@cute.jit
def _put_nbi_warp_int16_a16(dst: cute.Tensor, src: cute.Tensor, pe: Int32) -> None:
    """int16 ``put_nbi_warp`` over an align<16> FFI prototype (the wide-aligned wire-in).

    ``dst``/``src`` must already carry assumed_align=16 iterators (the caller stamps them; the
    addresses are 16-B aligned by construction). Transfers ``min(size(dst),size(src))`` int16."""
    nelems = cute.size(dst.layout)
    if cute.size(src.layout) < nelems:
        nelems = cute.size(src.layout)
    _int16_put_nbi_warp_a16(dst.iterator, src.iterator, cutlass.Uint64(nelems), pe)


@cute.jit
def _put_nbi_raw_int16_a16(dst_addr, src_addr, nelem: Int32, pe: Int32) -> None:
    """RAW-pointer int16 ``put_nbi_warp`` (LEVER-2 register shave): make_ptr straight from i64 byte
    addresses + call the FFI directly -- NO make_tensor, so the per-put path holds NO CuTe tensor-object
    registers (the sr/dr layout). ``dst_addr``/``src_addr`` are 16-B-aligned int16 BYTE addresses;
    ``nelem`` int16 elements. Semantically identical to :func:`_put_nbi_warp_int16_a16` (same extern +
    same wide-aligned prototype), just without the tensor wrapping."""
    dp = cute.make_ptr(cutlass.Int16, dst_addr, cute.AddressSpace.gmem, assumed_align=16)
    sp = cute.make_ptr(cutlass.Int16, src_addr, cute.AddressSpace.gmem, assumed_align=16)
    _int16_put_nbi_warp_a16(dp, sp, cutlass.Uint64(nelem), pe)


# BLOCKING (non-nbi) int16 warp put, align<16> -- the IN-KERNEL cross-node completion for the decoupled
# drain (the cp16 IBGDA QP-exhaustion fix). Same REBUILT align<16> FFI mechanism as the nbi put above
# (which compiles + runs cross-node -- probe rung3), just the NON-nbi symbol ``nvshmemx_int16_put_warp``:
# its IBGDA device path calls ``ibgda_quiet(qp)`` INTERNALLY after posting (ibgda_device.cuh:2237-2240,
# 2418-2419), so each put COMPLETES + backpressures in-kernel -> the completion queue is reaped, the QP
# never fills, and the ring slot is safe to reuse -- WITHOUT any separate device-``quiet`` API (which is
# NOT reachable from @cute.kernel). On NVLink it is a direct store (the non-nbi path is a cheap no-op past
# the store). Serializes per-put (perf follow-up); CORRECT-first. Built lazily; guarded by HAS_NVSHMEM.
_int16_put_warp_a16 = None
if HAS_NVSHMEM:
    try:
        from nvshmem.bindings.device.cute._cuteast import _CutePtrType as _NvshmemCutePtrType2

        _int16_put_warp_a16 = cute.ffi(
            name="nvshmemx_int16_put_warp",
            params_types=[
                _NvshmemCutePtrType2(cutlass.Int16, alignment=16),
                _NvshmemCutePtrType2(cutlass.Int16, alignment=16),
                cutlass.Uint64,
                cutlass.Int32,
            ],
        )
    except Exception:
        _int16_put_warp_a16 = None


@cute.jit
def _put_warp_int16_a16(dst: cute.Tensor, src: cute.Tensor, pe: Int32) -> None:
    """int16 BLOCKING ``put_warp`` (non-nbi) over an align<16> FFI prototype. Identical wire-in to
    :func:`_put_nbi_warp_int16_a16` but the IBGDA device path ``ibgda_quiet``-s after posting -> in-kernel
    completion + backpressure (the cp16 QP-exhaustion fix)."""
    nelems = cute.size(dst.layout)
    if cute.size(src.layout) < nelems:
        nelems = cute.size(src.layout)
    _int16_put_warp_a16(dst.iterator, src.iterator, cutlass.Uint64(nelems), pe)


# SOURCE-REUSE COMPLETION for the non-blocking (nbi) put path -- the fix for the wide drain's WAR
# hazard on its ring slot. `nvshmemx_flush_warp` is the vendor's exact primitive for a staging buffer
# about to be recycled, and its header contract IS the hazard, in their words:
#
#   nvshmemx_flush - Wait until all source buffers used by preceding non-blocking puts issued from
#   this thread are safe to reuse. Guarantees reusability only: the source buffer may be overwritten
#   or freed after this call returns. Does NOT guarantee that the data is visible at the remote PE...
#   For NVLink (P2P) puts: st.global stores are blocking at the instruction level, so the source is
#   already consumed when put_nbi returns. This call is a no-op on pure-P2P deployments.
#   For network (IB/RoCE, EFA, proxy) puts: waits for the transport to confirm that the source buffer
#   has been DMA'd. Does not issue __threadfence_system.
#       -- nvidia/nvshmem/include/device/nvshmemx_defines.h:744-758
#
# REQUIRES cutlass-dsl >= 4.7.0, and that is a REAL constraint, not a preference. NVVM ships with
# cutlass-dsl, and it is the READER of the nvshmem bitcode. Measured on ONE box, same nvshmem 3.7
# bitcode, a minimal kernel whose only body is this call:
#     cutlass-dsl 4.4.2 -> LINK_FAIL  (NVVM Compilation Error; the full kernel reports
#                                      NVVM_ERROR_INVALID_IR "Unknown attribute kind (102)",
#                                      Producer 'LLVM20.0.0git' / Reader 'LLVM 20.0.0')
#     cutlass-dsl 4.7.0 -> LINK_OK    (both flush_warp and quiet)
# So the older NVVM rejects an attribute the bitcode's flush/quiet call graph carries; the put
# wire-ins above avoid it and link under both. An earlier note here blamed the BITCODE (the sm_90
# .bc is ~19 MB larger than sm_80 and carries the GPUNetIO/DOCA path). That inference was WRONG:
# 3.7.2 keeps the same ~50 MB sm_90 bitcode, and the variable that actually moves is the reader.
_nvshmem_flush_warp_dev = None
if HAS_NVSHMEM:
    try:
        from nvshmem.bindings.device.cute import flush_warp as _nvshmem_flush_warp_dev
    except Exception:
        _nvshmem_flush_warp_dev = None


@cute.jit
def _flush_warp_device() -> None:
    """``nvshmemx_flush_warp()`` -- wait until this WARP's outstanding nbi-put SOURCE buffers are reusable.

    Purpose
        Close the write-after-read hazard on a staging buffer that non-blocking puts read from and a
        producer then refills. Call it immediately before releasing such a buffer.

    Functionality & semantics
        Warp-scope, covering the non-blocking puts issued by that warp. Guarantees SOURCE
        REUSABILITY ONLY -- explicitly NOT remote visibility, so it does not replace the mbarrier
        handshake or ``fence_acq_rel_sys`` for cross-agent ordering, nor a quiet where remote
        completion is what is needed. Documented as a no-op on a pure-P2P/NVLink deployment (those
        stores are already blocking at the instruction level), so the arm that does not need it pays
        nothing.

        Neither a FENCE nor a trailing BLOCKING PUT substitutes for it. A fence orders and does not
        wait. A blocking put was MEASURED not to complete the nbi puts that preceded it on the same
        warp -- see the note on ``DualGatedGemmDistSm90._configure_ib_wide``, which carries the 6/6
        vs 8/8 measurement that established it.

    Input requirements
        Must be called WARP-UNIFORMLY -- it is a warp-collective, so a divergent call hangs the warp.
        Requires ``HAS_NVSHMEM``, the nvshmem device bitcode LINKED into the kernel, and
        **cutlass-dsl >= 4.7.0** (see the block above); on an older NVVM the compile fails at
        MLIR->cubin rather than silently doing nothing.

    Returns:
        None. A missing binding leaves ``_nvshmem_flush_warp_dev`` None, so callers must
        const_expr-gate on it rather than discover it inside the kernel.
    """
    _nvshmem_flush_warp_dev()


def build_p2p_table(pe_table):
    """#57: the compile-time P2P/NVLink connectivity table. is_p2p[flat_cp_slot] = True iff peer
    pe_table[slot] is P2P/NVLink-reachable == TMA-S2G-able (a peer TMA-S2G store REQUIRES the P2P-mapped
    peer address). Uses nvshmem's NVSHMEM_TEAM_SHARED (the shared-memory / NVLink-local PE set) via
    team_translate_pe(WORLD, pe, SHARED) != -1 -- BUFFER-FREE (no symmetric alloc needed at configure),
    host-side, post-init. Replaces the //LWS "8" heuristic with the PRECISE topology query (correct for
    non-contiguous rank maps / NIC-PE mapping / NVL72). P2P is a STATIC job property -> a const_expr
    tuple baked once. Asserts |P2P| <= 8 (H100 NVSwitch domain; the user's NVLink-<=8 sanity). The
    TEAM_SHARED membership is cross-checked against get_peer_buffer(recv,pe)!=NULL on the 2-node probe
    (they MUST match; if they ever diverge, fall back to the get_peer_buffer probe). Returns tuple[bool].

    Module-level (NOT solely a ``GemmSm90A2A`` method) so callers OUTSIDE the kernel object -- notably the
    distributed autotuner's VENUE gate (``fold_cp_ops.distributed.fused_trimul_autotune``) -- probe the IDENTICAL
    topology source the store uses (``has_ib_peers = not all(build_p2p_table(pe_table))``); a divergent
    copy would tune the wrong (NVLink vs IB) kernel for the venue."""
    import nvshmem.core.teams as _nvt

    # nvshmem4py 3.7 exposes the predefined teams as the Team_id enum (NOT module attrs);
    # venue E-probed: Team_id.TEAM_WORLD=0, TEAM_SHARED=1 (the shared-memory / NVLink-local PE set ==
    # the P2P/TMA-able set); team_translate_pe accepts the enum (or its int handle) and returns the
    # dest-team rank (>=0) or -1 for a non-member. Fall back to the Teams-registry name->handle if the
    # enum is ever absent (still no RuntimeError -- the enum is the confirmed reference).
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
            )  # -1 / raise => not SHARED => IB peer
        except Exception:
            is_p2p.append(False)
    n_p2p = sum(is_p2p)
    assert n_p2p <= 8, (
        f"#57: |P2P/NVLink peers| = {n_p2p} > 8 (H100 NVSwitch domain is <=8; a larger set means a "
        f"topology surprise or NVL72/NVLS out-of-scope for SM90). is_p2p={is_p2p}, pe_table={tuple(pe_table)}."
    )
    return tuple(is_p2p)


# ---------------------------------------------------------------------------------------------------
class GemmSm90A2A(GemmSm90):
    """Hopper batched GEMM whose D-store is a BACK-A2A peer TMA store when
    ``configure_a2a`` / ``configure_a2a_gemm_native`` has been called; byte-identical
    to the local ``GemmSm90`` otherwise (all A2A paths ``const_expr(self._a2a_enabled)``-
    gated).

    ALL config knobs (tile_shape_mn, cluster_shape_mnk, pingpong, is_persistent,
    fp8_fast_accum, gather_A, ...) flow through ``super().__init__`` unchanged — this
    class re-declares NONE of them, so an autotuner can vary them freely.

    Compile via ``fold_cp_ops.distributed.gemm_bitcode_compile.compile_gemm_with_bitcode``
    (tvm-ffi OFF + --link-libraries + library_init), NOT the stock --enable-tvm-ffi
    route — the peer store issues an nvshmem device op (the peer S2G), which needs the
    bitcode linked.
    """

    #: Post-construction ``const_expr`` gates -- every compile-time read this class makes that is
    #: NOT a declared field of ``Params`` / ``CallParams``. ``configure_a2a*`` writes these AFTER
    #: construction, so ``param_dict()`` cannot see them; :meth:`compile_key` picks them up from
    #: here, which is what makes two differently-configured functors return different keys.
    #:
    #: MEASURED 2026-08-18: 22 of these were unkeyed, and the collision was demonstrated -- one
    #: composed key, two distinct emitted-MLIR hashes. Pinned against an AST scan of this class body
    #: by ``tests/_internal/compile_time/test_template_params.py``, so a gate added without a
    #: declaration fails that test rather than silently sharing an artifact.
    #:
    #: ``cta_tile_shape_mnk`` is DERIVED (tile_shape_mn + cluster) and therefore redundant with the
    #: pack fields; it is listed anyway because the scan finds it and an exempt table is a place for
    #: a real omission to hide. A redundant key component costs nothing.
    COMPILE_GATED_ATTRS = (
        "_a2a_N_i_loc",
        "_a2a_N_j_loc",
        "_a2a_N_loc",
        "_a2a_arbitrary_n",
        "_a2a_cluster_multislot",
        "_a2a_composite_k",
        "_a2a_consumer_strided",
        "_a2a_consumer_strided_putwarp",
        "_a2a_cp",
        "_a2a_cp0",
        "_a2a_decoupled",
        "_a2a_decoupled_store",
        "_a2a_dynamic",
        "_a2a_gemm_native",
        "_a2a_has_ib_peers",
        "_a2a_nt_i_pp",
        "_a2a_nt_j_pp",
        "_a2a_nt_pp",
        "_a2a_ring_depth",
        "_pe_aligned_tiling",
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Master gate; all A2A branches are const_expr(self._a2a_enabled)-gated.
        self._a2a_enabled = False
        self._a2a_arbitrary_n = (
            False  # opt-in: N_loc not a multiple of cta_tile_M (straddling peer-blocks)
        )
        # opt-in PE-boundary-aware per-peer M-tiling (spec v2): tile each peer's N_loc rows
        # independently (bases peer*N_loc + k*128) so NO output tile straddles a peer -> every store
        # is a uniform single TMA-S2G (the partial-last per-peer tile is high-end-clamped by the recv
        # descriptor). Default off -> uniform tiling -> byte-identical. cluster_M must be 1 (no
        # B-multicast-on-M, which per-peer bases would break). Supersedes the straddle SIMT path.
        self._pe_aligned_tiling = False
        # IB-RING variant (design doc §0.9): pe_aligned per-peer tiling routed through the DECOUPLED
        # GMEM-ring put_nbi_warp drain (GMEM-source -> NVSHMEM hybrid NVLink+IB auto-transport) instead
        # of the NVLink-only SMEM->peer TMA-S2G. Composes pe_aligned + decoupled (which are otherwise
        # mutually exclusive). Default off -> byte-identical. Set by configure_a2a_gemm_native(ib_ring=).
        self._pe_aligned_ib_ring = False
        # CLUSTER-DRAIN (Phase-3.1): a COALESCE-family MODIFIER -- an N-axis A-multicast cluster
        # (1, cluster_n, 1) cooperatively owns a BOUNDED per-peer_j band (run_j_tiles =
        # ceil(N_j_loc/(cluster_n*tile_n)), CLUSTER-tile units) instead of coalesce's full-N band, so the
        # GEMM stays near-stock (Phase-3.1a screen: cluster_n=2, k<=8 -> ratio 0.95-0.9975 vs autotuned
        # REF, bit-identical). Default off -> byte-identical (every branch const_expr-elided). 3.1a wires
        # the schedule (drain OFF -> coupled store); the cooperative GMEM-staging drain (design §3) is 3.1b.
        self._a2a_cluster_drain = False
        self._a2a_cluster_n = 0  # 0 sentinel => not configured; else the N-axis cluster width
        self._a2a_cluster_run_j = 0  # baked bounded-band run length (CLUSTER-tile units)
        # (d): MULTI-SLOT decouple. Removes the even-shard gate (nt_j_pp % cluster_n == 0) by letting a
        # cluster STRADDLE a peer_j boundary — the cluster_n CTAs compute their REAL N-tiles (zero GEMM
        # waste, A-multicast/cluster-barrier untouched) and ROUTE each output to a per-PEER staging SLOT
        # (2 rotating full-peer slots, d-fullpeer). Null-arrive keeps each slot's full-mbar count fixed at
        # cluster_n; the last-arriver-with-data puts. Default off -> the single-slot even-shard path,
        # BYTE-IDENTICAL. (Producer per-tile peer-routing + per-slot completion is the WIP core.)
        self._a2a_cluster_multislot = False
        # #57 CONSOLIDATION: the precise P2P/NVLink connectivity table (nvshmem TEAM_SHARED, buffer-free) +
        # the const_expr cp<=8 COLLAPSE gate. Default has_ib_peers=True -> the 4 IB-machinery gates
        # (_num_extra_warpgroups / _decoupled_sring_bytes / _extra_wg_reg_adjust / consumer role) stay
        # UNCHANGED -> byte-identical default. configure_a2a_gemm_native sets is_p2p + has_ib_peers=(not
        # all-P2P) on the ring (decoupled) paths: an ALL-P2P job (no IB peer) -> has_ib_peers=False -> the
        # whole IB machinery is const_expr-ELIDED -> every variant reduces to the pure pe_aligned NVLink TMA
        # store = FAST (the 3.2x fix, done structurally = same SASS as pe_aligned). is_p2p None until configure.
        self._a2a_is_p2p = None
        self._a2a_has_ib_peers = True
        self._a2a_cp = 1
        self._a2a_my_cp_rank = 0
        self._a2a_rows_per_peer = 0
        self._a2a_pe_table = ()
        # design-E GEMM-native 5-D store
        self._a2a_gemm_native = False
        self._a2a_B = 1
        self._a2a_N_loc = 0
        # Full token-j (GEMM-N) extent, set by configure_a2a_gemm_native when known. Under
        # arbitrary_n the store paths use it as the P_j (gj < N) col predicate to drop the last
        # partial N-tile's garbage cols. 0 sentinel => "N unknown" (the aligned/default path never
        # needs it -> the TMA descriptor clamps; the SIMT per-row drains require it under arbitrary_n).
        self._a2a_N = 0
        # ---- generic 1D/2D DTensor token-sharding (Part A) ----------------------
        # cp_axis_sizes = the per-cp-axis sizes (e.g. (cp,) for 1-D, (cp0, cp1) for a
        # 2-D token shard); product == cp. _a2a_cp_unravel_shape_stride is the
        # row-major LayoutRightMap (shape, stride) the kernel const_expr-builds to
        # re-flatten a 2-axis cp coordinate to the flat peer index. The DEFAULT (and
        # the 1-D configure_a2a* path) is cp_axis_sizes=(cp,) -> the unravel is the
        # identity, so the back-store peer math collapses to today's single division.
        self._a2a_cp_axis_sizes = (1,)
        self._a2a_cp_unravel_shape_stride = ((1,), (1,))
        # For the GEMM-native 2-D store the two TOKEN axes (i = GEMM-M, j = GEMM-N)
        # are each cp-split; N_i_loc / N_j_loc are the per-peer-block token extents.
        # 1-D (cp1 == 1) -> N_j_loc == N (j unsharded), N_i_loc == N // cp.
        self._a2a_N_i_loc = 0
        self._a2a_N_j_loc = 0
        # ---- Track A A2: 2-D pe_aligned per-peer tiles-per-block (ceil, both axes) ----
        # Set by configure_a2a_sharded(pe_aligned_tiling=True). nt_i_pp = ceil(N_i_loc/tile_m),
        # nt_j_pp = ceil(N_j_loc/tile_n) (N_j_loc = full N when cp1==1). The 2-D grid becomes
        # cp0*nt_i_pp x cp1*nt_j_pp; the per-peer A-row / B-col shifts re-base the mainloop loads.
        # 0 sentinel => not configured (the 1-D configure_a2a_gemm_native pe_aligned uses _a2a_nt_pp).
        self._a2a_nt_i_pp = 0
        self._a2a_nt_j_pp = 0
        # When True the GEMM-native store reads the TOKEN-scaling extents (tiles_per_i/j_block,
        # from N_i_loc/N_j_loc) off the recv tensor's RUNTIME shape instead of baking them
        # const_expr — so ONE dynamic-shape compile (mark_layout_dynamic recv + operands) serves
        # many token counts. CP-geometry (cp/my_cp_rank/B/cp0/cp1/cp1_stride) + tile sizes stay
        # baked. Default False => byte-identical static path (per-shape compile, today's behavior).
        self._a2a_dynamic = False
        # a_major of the A operand fed to the back GEMM (a caller DECLARATION, not a device-path
        # switch: the parent GemmSm90 auto-detects the operand's major from its stride and picks the
        # TMA atom / SMEM swizzle / WGMMA accordingly, so the device code is major-agnostic). "k"
        # (default) = K-unit-stride (M=token-i strided); the pe_aligned per-peer A-row shift rides the
        # STRIDED M axis at stride K==N (16-B aligned by the N%8 check). "m" = M-unit-stride (the
        # INCOMING transposed operand VIEW fed WITHOUT .contiguous()); the per-peer shift then lands in
        # the CONTIGUOUS M axis, so the per-peer base peer*N_loc must be 16-B aligned (N_loc%8==0). The
        # only effect of this flag is to gate that alignment LOUDLY at configure (a misaligned m-major
        # A-load otherwise dies with an inscrutable cudaErrorIllegalInstruction). Default "k" ->
        # byte-identical (no gate, current behavior).
        self._a2a_a_major = "k"
        # ---- route2_ni COMPOSITE-K read (§9, incoming) --------------------------------------------
        # The incoming front A2A writes a per-rank-CONTIGUOUS D-major recv (fast linear big-puts, no
        # b_j-pin) instead of the slow N_i-stride-1 recv; the back einsum then contracts K = (cp, Xg_pad)
        # as a rank-2 COMPOSITE. When ON, the A AND B operands (SYMMETRIC — both token operands from the
        # front recv) arrive 3-D (X, Xg_pad, L) and the composite hooks below present them 4-D
        # (X, Xg_pad, L, cp) so the (BLK, BLK_K) atom box tiles (X, Xg_pad) and cp rides as an untiled TMA
        # batch mode; the K-loop iterates cp*nt_within tiles (within inner, rank outer). KEEPS a_major="k"
        # + arbitrary-N via the existing Xg_pad (NO new shape constraint). Default off -> the parent rank-3
        # (X, K, L) path is BYTE-IDENTICAL (every hook const_expr-gated). cp = self._a2a_cp (set by
        # configure_a2a*). B=1 only (as route2_ni today).
        self._a2a_composite_k = False
        # ---- dynamic-cp flag (spec §B2): would make the peer-enumeration loops runtime-cp so ONE
        # compile serves many cp values. WONTFIX for EVERY A2A store (see the configure-time gate): the
        # reshard peer atoms are a compile-time host list of TMA descriptors with per-peer BAKED symmetric
        # bases, and a pure-cp A2A has cp==world so a different cp = a different nvshmem world = a different
        # binary -> use compile-per-cp. Retained as an always-raise guard (dyn_cp=True refuses loudly);
        # default False => byte-identical.
        self._a2a_dyn_cp = False
        # ---- DECOUPLED producer/consumer peer store (design doc §7) -----------------
        # ON => the design-E GEMM-native store becomes a PRODUCER (MMA warpgroup writes each
        # epilogue tile to a bounded LOCAL-GMEM ring + signals a dedicated CONSUMER warpgroup
        # via a two-color mbarrier) and a CONSUMER (the extra warpgroup drains ring->peer SymMEM
        # via a NON-BLOCKING SIMT put + a coarse host quiet()), so the NVLink put latency hides
        # behind the next tile's MMA instead of stalling it. Adds ONE warpgroup
        # (_num_extra_warpgroups) + the local-GMEM ring (bounded, flat-in-N). Default OFF =>
        # byte-identical to the coupled TMA-S2G store (and to the local GemmSm90 when A2A off).
        self._a2a_decoupled = False
        # STEP-2 gate: when set, the store ROUTES to the producer/consumer ring (the MMA
        # warpgroup stages to local GMEM + signals; the consumer warpgroup drains to peer).
        # STEP 1 leaves this False (warpgroup present but idle, coupled store) so the layout
        # proof is isolated from the ring wiring. configure_a2a_gemm_native(decoupled=True) sets
        # BOTH (full decoupled path); the step-1 driver flips only _a2a_decoupled.
        self._a2a_decoupled_store = False
        # Ring depth (staging slots per CTA). The footprint is grid_CTAs * ring_depth *
        # epi_m * epi_n * dtype_bytes (GMEM ring, SIMT drain) or ring_depth * epi_m * epi_n *
        # dtype_bytes of SMEM (SMEM ring, TMA drain). Independent of token count.
        self._a2a_ring_depth = 2
        # I2b PRODUCER refinement (user): the producer's ring-WRITE mechanism, orthogonal to the
        # consumer drain. OFF (default) => the MMA warp does a SYNCHRONOUS vectorized SIMT copy
        # SMEM->ring (it executes every LDS+STG -> O(tile) MMA-warp occupancy -> defeats decoupling).
        # ON => the MMA warp issues the SAME async TMA-S2G the single-device GEMM uses, retargeted to
        # the GMEM ring slot (fire one bulk descriptor -> the TMA engine moves the tile), then a CHEAP
        # local commit+wait_group before full[s].arrive() -> the MMA store cost ≈ the local-store
        # ceiling and the slow SymMEM put lives entirely on the consumer. Requires a GMEM ring (NOT the
        # SMEM ring -> incompatible with _a2a_decoupled_tma, which the consumer reads from SMEM).
        self._a2a_producer_tma = False
        # Number of dedicated CONSUMER warpgroups (each 4 warps). Scaling this adds draining warps
        # (the SIMT GMEM->peer drain may be under-warped per CTA). Costs registers (the extra
        # warpgroups shave num_regs_mma) but NO SMEM -> ab_stage untouched. Default 1 (4 warps).
        self._a2a_consumer_warpgroups = 1
        # ---- §7.16 WARP-STRIDED + WIDEN-the-put consumer drain (the headline win) ----------
        # The original lockstep SIMT consumer drain was the SOLE drain wall (best decpl/sym 2.61):
        # all consumer warps walk the SAME per-slot full[s]/empty[s] sequence in lockstep, draining a
        # NARROW (epi_m, epi_n)=128x32 box (64-byte rows) -> 33-43 GB/s (11-15% of NVLink). The
        # de-risked winning pattern (benchmark/distributed/proto_drain_bw.py): warp-strided DISJOINT
        # rows (no cross-warp per-slot lockstep) + WIDEN the put past ~256 B -> 320 GB/s (109% of the
        # a2a.py bar). ON => the producer-TMA ring becomes TILE-GROUPED (one slot per CTA tile, the
        # n_sub_per_tile j-subtiles laid contiguous along the slot's tile_n cols) so the consumer
        # issues per-row tile_n-WIDE (>=256 B) put_nbi_warp's, warp-strided over the tile's rows, with
        # ONE coarse host quiet at the kernel boundary; full[tile_slot]/empty[tile_slot] signal at TILE
        # (not subtile) granularity. Requires the GMEM ring producer-TMA (NOT the SMEM-ring decoupled_tma).
        # Default OFF => the lockstep per-subtile SIMT drain (byte-identical to today). See design §7.15/7.16.
        self._a2a_consumer_strided = False
        # §7.16j IB-swappable drain selector: ON => the strided consumer drains via per-row NVSHMEM
        # ``put_nbi_warp`` (``_decoupled_drain_loop_strided_putwarp``, the GMEM-SOURCE warp put routed by
        # RUNTIME PE ``pe_table_dev[peer]`` -> PE-TRANSPARENT, hence IB-capable) instead of the default
        # WIDE peer-pinned universal STG drain (REMOVED; was a P2P-mapped peer pointer -> NVLink-ONLY).
        # Only consulted on the strided path; the putwarp drain is now the sole survivor (default OFF was
        # the removed STG drain). SWITCH NVLink->IB: allocate ``recv`` on an IB-reachable team + a
        # ``pe_table_dev`` with the remote PEs — NO kernel change (the put is already routed by runtime PE;
        # the runtime picks IB for a non-P2P peer). The earlier lockstep SIMT drain (cute.copy to a P2P
        # peer ptr, NOT IB-swappable) was also removed.
        self._a2a_consumer_strided_putwarp = False

    def configure_a2a_gemm_native(
        self,
        cp,
        my_cp_rank,
        B,
        N_loc,
        pe_table,
        *,
        dynamic=False,
        decoupled=False,
        producer_tma=False,
        consumer_strided=False,
        consumer_strided_putwarp=False,
        ring_depth=None,
        consumer_warpgroups=None,
        arbitrary_n=False,
        pe_aligned_tiling=False,
        ib_ring=False,
        ib_quiet=True,
        dyn_cp=False,
        N=None,
        a_major="k",
        cp_axis_sizes=None,
        cluster_drain=False,
        cluster_n=None,
        cluster_multislot=False,
    ):
        """Enable the design-E GEMM-native 5-D back-A2A store (host-side config).

        The back GEMM (``_gemm1``) is one batched GEMM with L = d*B + b (the D-slice as
        the leading batch axis). Each CTA owns a ``(tile_i, tile_j)`` tile of ONE plane
        ``L``. This store routes that tile, by the TOKEN-i axis, into a 5-D GEMM-native
        symmetric recv ``(cp, Dloc, B, N_loc, N)`` = ``[slot, d, b, i_local, j]``:

          * peer  = ``i_global // N_loc``           (token-i block -> peer; NOT d/feature)
          * slot  = ``my_cp_rank``                  (which source feature-slice sent it)
          * d     = ``L // B``  (the recv d-axis, indexed directly -- the un-bake target)
          * b     = ``L % B``
          * i_local = ``i_global - peer*N_loc`` ;  j = the GEMM N-axis (innermost stride-1)

        The per-CTA TMA-S2G box is the FULL ``(N_loc, N)`` tile (j-innermost, >=16 B ->
        valid box; the degenerate 1-col store is a silent no-op, see
        ``benchmark/distributed/t30_f_epi_n1_box_probe.py``). The transpose to D-inner
        moves to the CONSUMER's input load (``recv.reshape(D, M).t()``; LN+DualGatedGEMM
        reads it natively via ``get_major`` auto-detect). Unifies L=1 & L>1; flag-off and
        the L=1 plain back path (``configure_a2a``) stay byte-identical.

        Parameters
        ----------
        cp : int
            Flat cp peer count (== the recv's leading ``slot`` extent).
        my_cp_rank : int
            This rank's flat-cp index in ``[0, cp)`` (the recv ``slot`` it writes).
        B : int
            TriMul batch -- splits the GEMM L axis: ``d = L // B``, ``b = L % B``.
        N_loc : int
            Token-i rows per peer block (``N // cp`` on the cp-split token axis). The
            per-peer recv's ``N_loc`` extent; a CTA M-tile must map to ONE peer, so
            ``N_loc`` must be a positive multiple of ``cta_tile_M``.
        pe_table : tuple of int
            ``cp``-length flat-peer -> global-PE map (``PeMap.cp_pe_table.tolist()``).
        consumer_strided_putwarp : bool
            §7.16j IB-swappable drain. ``False`` (default) -> on the strided path the
            consumer drains via the WIDE peer-pinned universal STG (NVLink-only). ``True``
            -> drain via per-row NVSHMEM ``put_nbi_warp`` over the LOCAL recv view, peer
            routed by RUNTIME PE (``pe_table_dev[peer]``) -> PE-transparent, hence
            IB-capable (the runtime picks IB for a non-P2P peer; NO kernel change to
            switch NVLink->IB, just allocate ``recv`` on an IB-reachable team + a
            ``pe_table_dev`` with the remote PEs). Forces ``decoupled`` + ``consumer_strided``.
        ib_ring : bool
            §0.9 HYBRID NVLink+IB back-A2A store. ``False`` (default) -> byte-identical (the
            ``pe_aligned_tiling`` / ``decoupled`` mutual-exclusion is unchanged). ``True`` ->
            COMPOSE pe_aligned's per-peer M-tiling scheduler WITH the decoupled GMEM-ring
            ``put_nbi_warp`` drain (which the default forbids): the store routes each peer write
            through the local-GMEM ring + a GMEM-source ``put_nbi_warp``, so NVSHMEM auto-selects
            NVLink P2P for node-local peers and IB for remote ones (one mechanism, hybrid
            transport). REQUIRES the full decoupled putwarp stack (``decoupled=True``,
            ``producer_tma=True``, ``consumer_strided=True``, ``consumer_strided_putwarp=True``),
            ``pe_aligned_tiling=True``, and a STRADDLING ``N_loc`` (so ``arbitrary_n`` stays on).
            Preserves pe_aligned's dynamic ``N_token`` (``dynamic=True``) + ``arbitrary_n`` shape
            contract. A misconfigured request (missing any required flag) raises loudly.
        ring_depth : int, optional
            ROTATING-RING depth override. ``None`` (default) -> the default depth ``2``. An explicit
            ``int`` (only meaningful when ``decoupled``) sets ``_a2a_ring_depth`` to a SMALL rotating
            ring so the producer's ``empty[s]`` reuse-WAIT is RE-ENABLED (the bounded-ring wrap gate
            ``k = g // rd >= 1`` is live). The caller MUST allocate the GMEM ring at this SAME depth
            ``(grid_CTAs, ring_depth, epi_m, tile_n)``. Ignored on the non-decoupled paths.
        consumer_warpgroups : int, optional
            Number of dedicated consumer warpgroups (4 drainer warps each), exposed so a
            sweep can pick {1, 2}. ``None`` (default) leaves ``_a2a_consumer_warpgroups``
            unchanged (1). Only meaningful on the decoupled path.
        N : int, optional
            Full token-j (GEMM-N) extent. Required for ``arbitrary_n`` (the partial-N column
            predicate ``gj < N``).
        cp_axis_sizes : tuple of int, optional
            Task #13 2-D token sharding. ``None`` (default) -> 1-D ``(cp,)`` (only the GEMM-M / token-i
            axis is cp-split, j full N; byte-identical). ``(cp0, cp1)`` -> BOTH token axes are split (i
            over ``cp0``, j over ``cp1``); ``N_loc`` MUST equal ``N // cp0`` and ``N_j_loc = N // cp1``.
            The coalesce_dyn drain then routes each row/band to the row-major flat peer
            ``peer = peer_i*cp1 + peer_j``. Supported for the coalesce_dyn drain; ``cp0*cp1`` must
            equal ``cp`` and each per-axis extent must be ``%8`` (16-B TMA).

        Raises
        ------
        RuntimeError
            If nvshmem4py / the vendored nvshmem utils are not importable.
        ValueError
            If ``N_loc`` is not a positive multiple of ``cta_tile_M`` or
            ``len(pe_table) != cp``.
        """
        if not HAS_NVSHMEM:
            raise RuntimeError(
                "GemmSm90A2A.configure_a2a_gemm_native requires nvshmem4py + the vendored nvshmem utils."
            )
        # ---- cluster_multislot is the SOLE production IB drain. Every OTHER
        # IB-drain variant is REMOVED at the source -- coalesce / differential-FLAG / cluster_roundpark /
        # cluster_nvdirect / cluster_drain_db / cluster_drain_mw / ib_reap (+ their drain_warps /
        # drain_cta_div / reap_cadence / min_drain* / handshake_skip / local_world_size sub-levers) no
        # longer exist as params, so a caller passing one gets a TypeError. KEPT: cluster_drain=True +
        # cluster_multislot=True (cluster_n in {1,2,4}) riding the decoupled / consumer_strided_putwarp /
        # arbitrary_n / pe_aligned_tiling / ib_quiet stack + the #57 is_p2p auto-routing. The one surviving
        # guard (re-homed from the Stage-1 rejects): consumer_strided is KEPT (multislot rides it WITH
        # putwarp), but the non-putwarp strided drain (the removed WIDE peer-pinned STG) is gone, so guard
        # consumer_strided-without-putwarp so it never silently no-op-drains.
        if consumer_strided and not consumer_strided_putwarp:
            raise ValueError(
                "cluster_multislot is the sole production IB drain; the non-putwarp strided "
                "consumer drain (consumer_strided=True WITHOUT consumer_strided_putwarp -- the "
                "removed WIDE peer-pinned STG drain) is removed (epic-close P2). Pass "
                "consumer_strided_putwarp=True (the IB-capable put_nbi_warp drain multislot rides)."
            )
        cta_tile_m = self.tile_shape_mn[0]
        # arbitrary_n: opt-in support for N_loc NOT a multiple of cta_tile_M (a CTA M-tile may
        # straddle >=2 peer-blocks). The store paths const_expr-branch on the misalignment: aligned
        # stays the single-store/per-tile byte-identical path; misaligned takes the per-row SIMT drain
        # / partial-M multi-TMA coupled store. Default OFF -> the original one-peer-per-tile constraint.
        if N_loc <= 0 or (N_loc % cta_tile_m != 0 and not arbitrary_n):
            raise ValueError(
                f"N_loc={N_loc} must be a positive multiple of cta_tile_M={cta_tile_m} "
                f"(a CTA output M-tile must map to ONE peer); pass arbitrary_n=True to allow "
                f"a straddling N_loc (per-row SIMT drain / partial-M multi-TMA coupled store)."
            )
        if arbitrary_n:
            # arbitrary_n now means BOTH "N_loc may straddle peer-blocks" AND "PARTIAL M/N tiles are
            # handled" (the last CTA M-tile / N-tile may extend past M=N_loc*cp / past N). The only
            # remaining requirement is N_token%8==0 (16-B alignment, so the SIMT per-row puts stay
            # 16-B aligned) and N_token%cp==0 (the caller's reshard split; here it is implied because
            # N_loc=N_token//cp is an integer). The TWO predicates that make this work:
            #   P_i: gi < M=N_loc*cp  (drop garbage rows of the last partial M-tile; else peer=gi//N_loc
            #        is OOB) -> SIMT per-row drains guard it explicitly; TMA paths clamp via the N_loc
            #        descriptor extent at the last peer.
            #   P_j: gj < N            (drop garbage cols of the last partial N-tile) -> SIMT puts mask
            #        the col count; TMA paths clamp via the N descriptor extent.
            epi_m = math.gcd(
                128, cta_tile_m
            )  # the epi-subtile M extent (see _sm90_compute_tile_shape)
            # N (full token-j extent) is REQUIRED under arbitrary_n: the SIMT per-row P_j predicate masks
            # the last partial N-tile via gj < N. Without N, has_partial_n would be False -> garbage-col
            # writes on a partial-N shape. Make it a LOUD error, never a silent miss.
            if N is None:
                raise ValueError(
                    "arbitrary_n requires N (the full token-j extent) for the partial-N column "
                    "predicate (gj < N); got N=None."
                )
            # 16-B alignment of every stride-1 (N-axis) run -> bf16 needs N%8==0 (8*2 B = 16 B).
            # The arbitrary_n stores (SIMT per-row puts / coalesced band puts over an N-pitch recv)
            # require N%8==0.
            if int(N) % 8 != 0:
                raise ValueError(
                    f"arbitrary_n requires N={N} (token-j) to be a multiple of 8 (16-B aligned "
                    f"stride-1 runs); got N%8={int(N) % 8}."
                )
            # AUTO-REDUCE (PLAIN single-TMA store ONLY): the arbitrary_n path is a NO-OP on a FULLY-aligned
            # shape — N_loc % epi_m == 0 (no straddle; epi_m == cta_tile_m for the supported tile => M =
            # N_loc*cp % cta_tile_m == 0, no partial token-i) AND N % cta_tile_n == 0 (no partial token-j).
            # Reduce to the PROVEN default store so arbitrary_n=True is byte-identical to =False there
            # (§0.8 perf-neutrality, BY CONSTRUCTION). Straddle (N_loc%epi_m!=0) or partial-N keep it True.
            # ---- Task #13 GUARD: the auto-reduce is correct ONLY for the plain single-TMA store. The
            # pe_aligned-DRAIN stores (ib_ring / coalesce) and a 2-D (cp1>1) mesh REQUIRE the
            # arbitrary_n/pe_aligned path EVEN at an aligned N_loc: the 2-D j-axis routing lives in the
            # pe_aligned drain. At an aligned N_loc that drain is the
            # nt_pp-EXACT, no-straddle sub-case of the VALIDATED straddle path (no new code path: the
            # straddle double-store / ceil-spill / partial-N predicates all degenerate to no-ops when each
            # 128-row tile base peer*N_loc+k*128 is peer-aligned and within one peer) -> correct +
            # perf-neutral. So SUPPRESS the auto-reduce for those; the plain 1-D store still reduces
            # (byte-identical to today -> the §0.8 optimization is preserved). `cp1` is read from the raw
            # cp_axis_sizes kwarg (self._a2a_cp1 is resolved later, at the 2-D block below).
            _cp1_req = (
                int(cp_axis_sizes[1])
                if (cp_axis_sizes is not None and len(cp_axis_sizes) > 1)
                else 1
            )
            _needs_pe_aligned = bool(ib_ring or _cp1_req > 1)
            if not _needs_pe_aligned and N_loc % epi_m == 0 and int(N) % self.tile_shape_mn[1] == 0:
                arbitrary_n = False
        if len(pe_table) != cp:
            raise ValueError(f"pe_table {pe_table} length must equal cp={cp}.")
        # ---- 2-D token sharding (Task #13): cp_axis_sizes=(cp0,cp1) splits BOTH token axes (i over cp0,
        # j over cp1). Default None => the 1-D (cp,) case (j full N, byte-identical). The i-axis extent the
        # caller passes as N_loc MUST equal N//cp0; the j-axis per-peer block is N_j_loc = N//cp1. The
        # coalesce_dyn DRAIN routes each row/band to the row-major flat peer
        # peer = peer_i*cp1 + peer_j (mirrors the producer copy_fn :2577-2579 and the 2-D coupled store's
        # _ref_back_gemm_native_2d oracle). 1-D keeps cp1==1 -> every 2-D branch below is const_expr-elided.
        if cp_axis_sizes is None:
            _cp_axis = (int(cp),)
        else:
            _cp_axis = tuple(int(s) for s in cp_axis_sizes)
        if len(_cp_axis) > 2:
            raise ValueError(
                f"configure_a2a_gemm_native supports 1-D or 2-D token sharding; got "
                f"cp_axis_sizes={_cp_axis} ({len(_cp_axis)} cp axes)."
            )
        _cp0 = _cp_axis[0]
        _cp1 = _cp_axis[1] if len(_cp_axis) > 1 else 1
        if _cp0 * _cp1 != int(cp):
            raise ValueError(
                f"cp_axis_sizes={_cp_axis} must multiply to cp={cp} (cp0*cp1={_cp0 * _cp1})."
            )
        if _cp1 > 1 and N is None:
            raise ValueError(
                f"2-D configure_a2a_gemm_native (cp_axis_sizes={_cp_axis}) requires N (the square token "
                f"extent) to derive N_j_loc = N//cp1."
            )
        self._a2a_cp0 = int(_cp0)
        self._a2a_cp1 = int(_cp1)
        self._a2a_enabled = True
        self._a2a_gemm_native = True
        self._a2a_cp = int(cp)
        self._a2a_my_cp_rank = int(my_cp_rank)
        self._a2a_B = int(B)
        self._a2a_N_loc = int(N_loc)
        # Remember the full token-j extent N when supplied (used by the arbitrary_n P_j col predicate
        # in the SIMT per-row drains; the TMA paths clamp via the descriptor and never read it).
        if N is not None:
            self._a2a_N = int(N)
        self._a2a_arbitrary_n = bool(arbitrary_n)
        # a_major of the A operand (see __init__). "m" (INCOMING transposed VIEW) makes the pe_aligned
        # per-peer A-row shift ride the CONTIGUOUS M axis -> the per-peer base peer*N_loc must be 16-B
        # aligned (N_loc%8==0); "k" (default) rides the STRIDED M axis (byte-identical current path).
        self._a2a_a_major = str(a_major)
        if self._a2a_a_major not in ("k", "m"):
            raise ValueError(f"a_major must be 'k' or 'm'; got {a_major!r}.")
        # IB-RING variant (hybrid NVLink+IB back-A2A, design doc §0.9): compose pe_aligned's per-peer
        # M-tiling scheduler with the DECOUPLED GMEM-ring producer + put_nbi_warp drain (GMEM-source ->
        # NVSHMEM auto-selects NVLink P2P for node-local peers, IB for remote). The default pe_aligned /
        # decoupled mutual-exclusion (`and not decoupled` below) FORBIDS this compose; ib_ring lifts it
        # for exactly this stack. Requires the full decoupled putwarp path (decoupled + producer_tma is
        # forced + consumer_strided_putwarp) AND arbitrary_n (the per-row gi->peer route + gi_base meta
        # field 5). Default OFF => the `and not decoupled` term is unchanged => byte-identical.
        self._pe_aligned_ib_ring = bool(
            ib_ring
            and pe_aligned_tiling
            and self._a2a_arbitrary_n
            and decoupled
            and consumer_strided_putwarp
        )
        if ib_ring and not self._pe_aligned_ib_ring:
            # ib_ring requested but its required stack is incomplete: fail LOUDLY (a silent fall-through
            # to the default coupled/decoupled store would run a DIFFERENT mechanism than asked).
            if not self._a2a_arbitrary_n:
                raise ValueError(
                    "ib_ring=True requires a STRADDLING N_loc (arbitrary_n stays True; an aligned "
                    "N_loc auto-reduces arbitrary_n off -> the plain single-TMA store already serves "
                    "it). Got arbitrary_n resolved False."
                )
            if not (decoupled and consumer_strided_putwarp):
                raise ValueError(
                    "ib_ring=True requires the decoupled GMEM-ring putwarp drain: pass decoupled=True, "
                    "producer_tma=True, consumer_strided=True, consumer_strided_putwarp=True."
                )
            if not pe_aligned_tiling:
                raise ValueError("ib_ring=True requires pe_aligned_tiling=True.")
        # §0.9.5 verdict: the BARE uniform ib_ring putwarp store loses to NCCL 1.6-6x (per-row 512 B-
        # granularity puts) and is REMOVED. A complete ib_ring stack MUST select the cluster-cooperative
        # WIDE band (cluster_drain=True) -> reject the uniform loser LOUDLY so it is never silently the store.
        if self._pe_aligned_ib_ring and not cluster_drain:
            raise ValueError(
                "ib_ring=True requires a WIDE-BAND drain (cluster_drain=True): the bare uniform ib_ring "
                "putwarp store (§0.9.5) is removed (it loses to NCCL 1.6-6x). Pass cluster_drain=True "
                "(cluster-cooperative band)."
            )
        # opt-in PE-boundary-aware per-peer M-tiling (spec v2). Active ONLY when the straddle path is
        # active (arbitrary_n stayed True, i.e. N_loc % 128 != 0) and on the COUPLED TMA store -- OR on
        # the ib_ring decoupled putwarp path (which explicitly composes pe_aligned + decoupled). On an
        # aligned N_loc the auto-reduce already turned arbitrary_n off -> pe_aligned_tiling inert (the
        # per-peer bases == uniform bases anyway -> byte-identical). cluster_M must be 1: per-peer bases
        # are non-contiguous in M, which would break a B-multicast-on-M (num_mcast_ctas_b = cluster_M).
        self._pe_aligned_tiling = bool(
            pe_aligned_tiling
            and self._a2a_arbitrary_n
            and (not decoupled or self._pe_aligned_ib_ring)
        )
        if self._pe_aligned_tiling:
            if self.cluster_shape_mnk[0] != 1:
                raise ValueError(
                    f"pe_aligned_tiling requires cluster_M==1 (no B-multicast-on-M); got "
                    f"cluster_shape_mnk={self.cluster_shape_mnk}."
                )
            # nt_pp = tiles per peer = ceil(N_loc / cta_tile_M). Total m-tiles = cp*nt_pp (vs the
            # uniform ceil(M/cta_tile_M)). Const for the static compile.
            cta_tile_m_c = self.tile_shape_mn[0]
            self._a2a_nt_pp = (int(N_loc) + cta_tile_m_c - 1) // cta_tile_m_c
            # 2-D (Task #13): per-peer tile counts on BOTH axes for the 2-D-sharded ib_ring drain.
            # nt_i_pp == nt_pp (i-axis, ceil(N_i_loc/tile_m)); nt_j_pp = ceil(N_j_loc/tile_n) where
            # N_j_loc = N//cp1 (only set for cp1>1). 1-D (cp1==1) never reads nt_j_pp (the scheduler
            # keeps ps[1]/the N-axis unchanged) -> byte-identical.
            self._a2a_nt_i_pp = self._a2a_nt_pp
            if self._a2a_cp1 > 1:
                cta_tile_n_c = self.tile_shape_mn[1]
                _N_j_pp = int(self._a2a_N) // int(self._a2a_cp1)
                self._a2a_nt_j_pp = (_N_j_pp + cta_tile_n_c - 1) // cta_tile_n_c
            # a_major="m" (INCOMING transposed VIEW) 16-B ALIGNMENT WALL: the per-peer A-row shift
            # peer*N_loc rides the CONTIGUOUS M axis (stride 1), so each peer's TMA-load box starts at
            # element offset peer*N_loc. TMA requires a 16-B-aligned box start => bf16 needs
            # N_loc % 8 == 0 (8 elems * 2 B). A misaligned N_loc CRASHES the A-load with an inscrutable
            # cudaErrorIllegalInstruction (empirically N_loc=260 crash / 520 OK; the positive-shift
            # reformulation does NOT help -- it yields the same box coord/alignment). Fail LOUDLY here.
            # (a_major="k" rides the STRIDED M axis at stride K==N, %8 by the N%8 check -> always safe.)
            # Dynamic path: N_loc here is the compile ANCHOR; the CALLER must also ensure the RUNTIME
            # N_loc % 8 == 0 (the recv shape the kernel reads at run time).
            if self._a2a_a_major == "m" and int(N_loc) % 8 != 0:
                raise ValueError(
                    f"a_major='m' (transposed incoming operand VIEW) with pe_aligned_tiling requires "
                    f"N_loc % 8 == 0 (16-B TMA alignment of the contiguous per-peer A-row base "
                    f"peer*N_loc); got N_loc={N_loc} (N_loc%8={int(N_loc) % 8}). Use a_major='k' "
                    f"(materialize the transpose with .contiguous()) for a misaligned N_loc."
                )
        # ============ mod-128 BYTE-PHASE LAW (PERFORMANCE, above the 16-B LEGALITY floor) =============
        # Alignment comments in this file reason to 16 B, the TMA/put LEGALITY minimum, which says
        # nothing about SPEED. The performance granularity above it is stated once, here.
        #
        # (1) DESTINATION base phase is REAL, and is a peer-STORE effect: the penalty tracks the
        #     destination byte address mod 128 -- ~+4% at 64-mod-128, ~+8-11% at 32-mod-128, ~+18-37%
        #     at 16-mod-32. Live today as the front's rank-parity 1.54x at 16-mod-32
        #     (`test_front_partial_token_clamp_smooth`). The lever is the FRONT recv's token pitch, not
        #     anything in this file.
        #
        # (2) The ROW-STRIDE clause is REFUTED for THIS store's recv. An earlier revision blamed the
        #     design-E recv's token-i row stride `2*N_j_loc` for +21-29%. That came from a CROSS-N
        #     comparison (N=1032 vs 1040), which cannot separate row phase from trailing-tile width or
        #     operand stride -- all three move with N. At FIXED N=1040, varying ONLY the recv's
        #     innermost extent: 0.987-1.002 at Dloc=8, 0.999-1.002 at Dloc=128. Zero.
        #
        # (3) The +21-29% is the SOURCE operand leading dimension. `A_t`/`B_t` are `(L, M, K=N)`
        #     permuted to `(M, K, L)` (`fused_trimul.py:1444`), so the mainloop TMA-G2S walks an M-row
        #     stride of `2*N` bytes -- 2064 B (16-mod-32) at N=1032 vs 2080 B at N=1040. Same fixed-N
        #     control on the OPERANDS: 1.113-1.132 at Dloc=8, 1.213-1.215 at Dloc=128, against a
        #     cross-N of 1.221-1.223. A phase effect, not a size effect -- a LARGER `opad48` buffer is
        #     FASTER than `opad8` -- and only the 16-mod-32 tier shows on the source side, consistent
        #     with a 32-B sector granularity on the A/B TMA vs the 128-B one governing the store.
        #
        # (4) THEREFORE do NOT pad `N_j_loc` here. Zero benefit, and it breaks the downstream ZERO-COPY
        #     contract -- `reshard.back_unpack_gemm_native` is `recv.reshape(D, M).t()`, a pure view
        #     needing the j axis compact -- forcing the `a_major="m"` path and its own 16-B wall.
        #     The real lever is the back OPERAND's M-row stride (the inner token extent of the
        #     `(Dloc, B, N_i, N_j)` view; `fused_trimul.py:1437-1438`): make it a multiple of 64 bf16
        #     elements. The consumer already accepts one -- the `opad` probe ran a padded operand
        #     M-stride on the production pe_aligned + arbitrary_n store and wrote the full recv
        #     (`untouched == 0`) on all 8 ranks. Per route: plain incoming materialises with
        #     `.contiguous()` (local, cheap); composite_k / route2_ni are clean by construction (K is
        #     already `Xg_pad`); outgoing is a zero-copy VIEW of the front recv, so padding it means
        #     padding that recv's inner token extent, which also fixes (1) -- a `.contiguous()` there
        #     would be an O(Dloc*N^2) I/O-order copy, forbidden.
        # ==============================================================================================
        # The distinct-split set + the N_loc>=epi_m guard need epi_tile, which is computed at
        # compile (still None here). Defer both to _build_peer_store_atoms_gemm_native (runs host-
        # side at compile, epi_tile set) -> it sets _a2a_route2_splits before the kernel traces.
        # arbitrary_n per-row routing needs the epi-box's GLOBAL token-i base in the ring metadata so the
        # drain can compute, per row, peer = gi//N_loc and i_local = gi%N_loc (a straddling box's rows fan
        # out to >=2 peers). Carry it as ONE extra Int32 meta field (index 5) -> bump _DECOUPLED_META_FIELDS
        # to 6 only on this opt-in path (instance attr shadows the class default 5; the SMEM meta MemRange,
        # the producer fill, and every drain all read self._DECOUPLED_META_FIELDS -> they pick it up
        # uniformly). Default OFF keeps the class 5 -> SMEM sizing + metadata byte-identical.
        if self._a2a_arbitrary_n:
            self._DECOUPLED_META_FIELDS = 6
        self._a2a_dynamic = bool(dynamic)
        # ---- TRACK-B dynamic-cp flag (spec §B2): runtime cp peer loops; implies dynamic-shape. ----
        if dyn_cp and not self._a2a_dynamic:
            raise ValueError("dyn_cp=True requires dynamic=True (dyn_cp implies dynamic-shape).")
        # dyn-cp is WONTFIX for EVERY A2A store (pe_aligned + the coupled TMA-S2G store): the reshard
        # peer atoms are a compile-time host list of TMA descriptors with per-peer BAKED symmetric base
        # addresses (no runtime-base TMA-S2G), and pure-cp A2A has cp==world so a different cp is a
        # different nvshmem world = a different launch. Use compile-per-cp. Refuse loudly here (the dynamic-cp
        # verdict) so a dyn_cp=True request never silently no-ops.
        if dyn_cp:
            raise NotImplementedError(
                "dynamic-cp is WONTFIX for A2A stores: use compile-per-cp. Peer atoms are a "
                "compile-time host list with per-peer baked symmetric bases, and pure-cp cp==world."
            )
        self._a2a_dyn_cp = bool(dyn_cp)
        self._a2a_decoupled = bool(decoupled)
        self._a2a_decoupled_store = bool(decoupled)  # full path: warpgroup + ring store
        # ib_quiet: the putwarp drain's cross-node completion mode. True (default) -> BLOCKING put_warp
        # (ibgda_quiet-s in-kernel per put; correct + safe under ring reuse, serializes 1-deep -> slower).
        # False -> non-blocking put_nbi_warp (fast, self-bounded by IBGDA reserve backpressure; completion
        # by the post-kernel host quiet). NBI is correctness-safe ONLY when the ring does NOT wrap OR the
        # transport reads the source at issue (NVLink); over IB with a WRAPPING ring the producer can
        # overwrite a slot before the NIC finishes the async source-read (a reuse hazard the host quiet
        # does NOT cover). Exposed so the bench/tests can A/B nbi-vs-blocking correctness + perf.
        self._a2a_ib_quiet = bool(ib_quiet)
        # §7.16 warp-strided + widen consumer: a GMEM-ring SIMT consumer variant. Forces the GMEM-ring
        # producer-TMA write (the contiguous tile-wide source the wide put needs) and is INCOMPATIBLE
        # with the SMEM-ring TMA drain (decoupled_tma). The IB-swappable putwarp drain REQUIRES the
        # strided path -> force consumer_strided when consumer_strided_putwarp is set.
        self._a2a_consumer_strided = bool(
            decoupled and (consumer_strided or consumer_strided_putwarp)
        )
        # §7.16j IB-swappable drain selector (only consulted on the strided path). Routes the strided
        # consumer to put_nbi_warp (GMEM-source, runtime-PE -> IB-capable) instead of the WIDE STG.
        self._a2a_consumer_strided_putwarp = bool(
            self._a2a_consumer_strided and consumer_strided_putwarp
        )
        # producer_tma (GMEM-ring producer-TMA store) is INCOMPATIBLE with decoupled_tma (SMEM-ring
        # TMA consumer drain): the producer-TMA writes a GMEM ring; the SMEM-ring consumer reads SMEM.
        # consumer_strided REQUIRES the GMEM ring -> force producer_tma on that path.
        self._a2a_producer_tma = bool(decoupled and (producer_tma or self._a2a_consumer_strided))
        # Expose the consumer-warpgroup count (4 drainer warps each) so a sweep can pick {1,2}. None
        # leaves the default (1) untouched.
        if consumer_warpgroups is not None:
            self._a2a_consumer_warpgroups = int(consumer_warpgroups)
        # Phase-3 ROTATING-RING putwarp drain: an explicit SMALL bounded depth. The producer's
        # empty[s] reuse-WAIT is live (the wrap gate k>=1) so a depth-`ring_depth` ring rotates: the
        # producer waits the consumer to drain a slot before reusing it. Costs only
        # `grid_CTAs * ring_depth * epi_m * tile_n * dtype_bytes` of GMEM. The caller MUST allocate
        # the GMEM ring at this SAME depth.
        if decoupled and ring_depth is not None:
            if int(ring_depth) < 1:
                raise ValueError(f"ring_depth={ring_depth} must be a positive int.")
            self._a2a_ring_depth = int(ring_depth)
        self._a2a_rows_per_peer = int(N_loc)  # the token-i peer block extent
        self._a2a_pe_table = tuple(int(p) for p in pe_table)
        # Token-shard geometry (Task #13). 1-D (cp1==1): only the GEMM-M (i) axis is cp-split, j (GEMM-N)
        # is FULL -> cp_axis_sizes=(cp,), identity unravel, N_j_loc sentinel 0 (today's path, byte-
        # identical). 2-D (cp1>1): the LayoutRightMap over (cp0,cp1) is the authoritative row-major flatten
        # the drain uses to re-form peer = peer_i*cp1 + peer_j; N_j_loc = N//cp1 is the per-peer j-block the
        # drain routes columns by (mirrors _ref_back_gemm_native_2d + the 2-D coupled store).
        self._a2a_cp_axis_sizes = _cp_axis
        if self._a2a_cp1 > 1:
            if int(N_loc) != int(self._a2a_N) // self._a2a_cp0:
                raise ValueError(
                    f"2-D configure_a2a_gemm_native requires N_loc == N//cp0 (the i-axis peer block); got "
                    f"N_loc={N_loc}, N={self._a2a_N}, cp0={self._a2a_cp0} "
                    f"(N//cp0={int(self._a2a_N) // self._a2a_cp0})."
                )
            if int(self._a2a_N) % self._a2a_cp1 != 0:
                raise ValueError(
                    f"2-D configure_a2a_gemm_native requires N%cp1==0 (the j-axis reshard split); got "
                    f"N={self._a2a_N}, cp1={self._a2a_cp1}."
                )
            unravel = LayoutRightMap(_cp_axis)
            self._a2a_cp_unravel_shape_stride = (
                tuple(int(s) for s in unravel.shape),
                tuple(int(s) for s in unravel.strides),
            )
            self._a2a_N_i_loc = int(N_loc)
            self._a2a_N_j_loc = int(self._a2a_N) // self._a2a_cp1
        else:
            self._a2a_cp_unravel_shape_stride = ((int(cp),), (1,))
            self._a2a_N_i_loc = int(N_loc)
            self._a2a_N_j_loc = 0  # 0 sentinel => j unsharded (full N); the copy_fn uses tile_n
        # ---- 2-D (Task #13) gate + per-axis 16-B guard. Placed at the END so every variant flag is
        # resolved. cp1>1 is supported for coalesce_dyn (the option-D peer_j-MAJOR band; §0.9.12).
        # 1-D (cp1==1) skips this whole block -> byte-identical.
        if self._a2a_cp1 > 1:
            # DYNAMIC-N required for 2-D: the per-peer j col-clamp reads N_j_loc off the RUNTIME recv
            # (recv.shape[4] via dyn_ib); a STATIC-N 2-D would clamp to the full N_full (mis-sized on the
            # j-sharded recv -> OOB/garbage on a straddling N_j_loc). Static-N 2-D is a follow-up.
            if not self._a2a_dynamic:
                raise NotImplementedError(
                    f"2-D (cp1>1) ib_ring requires dynamic=True: the per-peer j col-clamp reads N_j_loc "
                    f"off the runtime recv; static-N 2-D is a follow-up. Got cp_axis_sizes={_cp_axis}."
                )
            if not cluster_drain:
                raise NotImplementedError(
                    f"2-D (cp1>1) ib_ring is supported ONLY for the cluster_drain variant "
                    f"(cluster_drain=True). Got a uniform/other store with cp_axis_sizes={_cp_axis}."
                )
            # per-axis 16-B TMA guard (bf16 -> %8). DYNAMIC caller-contract: this static check sees only the
            # compile-anchor N -> the caller MUST keep (N//cp0)%8==0 AND (N//cp1)%8==0 at EVERY runtime N
            # (necessary-not-sufficient, same as the 2-D coupled path). Else a per-row put straddle-garbages
            # a partial run (only scalar rel_L2 catches it; the per-row outlier gate misses it).
            _Ni = int(self._a2a_N) // self._a2a_cp0
            _Nj = int(self._a2a_N) // self._a2a_cp1
            if _Ni % 8 != 0 or _Nj % 8 != 0:
                raise ValueError(
                    f"2-D ib_ring drain requires 16-B-aligned per-peer extents: N_i_loc=N//cp0={_Ni} and "
                    f"N_j_loc=N//cp1={_Nj} must both be multiples of 8 (bf16); got N_i_loc%8={_Ni % 8}, "
                    f"N_j_loc%8={_Nj % 8}."
                )
        # _a2a_drain_tail: the MMA-warp tail hook (tail_drain_role) is enabled for cluster_multislot ONLY
        # (set True in the cluster_drain block below). Default False -> the parent skips the tail hook.
        self._a2a_drain_tail = False
        # #57 CONSOLIDATION: on the RING (decoupled) paths (cluster_drain / coalesce_dyn / differential --
        # the variants that carry the IB machinery), build the PRECISE P2P/NVLink connectivity table (nvshmem
        # TEAM_SHARED via _build_p2p_table, buffer-free) and derive has_ib_peers = NOT all-P2P. An ALL-P2P
        # job (cp<=8 single NVLink domain, no IB peer) -> has_ib_peers=False -> the 4 const_expr gates ELIDE
        # the decoupled ring / consumer-warpgroup / cluster-staging -> every variant reduces to the pure
        # pe_aligned NVLink TMA store (the 3.2x fix, structurally). Non-ring paths keep the ctor default
        # has_ib_peers=True -> gates UNCHANGED -> byte-identical. (Stage 1 = the collapse; the mixed-cp>8
        # per-tile is_p2p routing + ring counter re-param is the tracked Stage 2.)
        # §5.2 / B1 — the P2P probe is now UNCONDITIONAL (it was decoupled-only, so a COUPLED store
        # never probed its own topology and silently took an IMA cross-node), and a coupled store on a
        # cross-node mesh is REJECTED here. See _guard_coupled_nvlink_only.
        self._guard_coupled_nvlink_only(pe_table, entry="configure_a2a_gemm_native")
        if self._a2a_decoupled:
            self._a2a_has_ib_peers = not all(self._a2a_is_p2p)
        # CLUSTER-DRAIN (Phase-3.1a): record intent + bake the BOUNDED per-peer_j band length (CLUSTER-tile
        # units). Placed at the END so cp0/cp1/N/tile are resolved. Default off -> byte-identical. The
        # caller passes cluster_shape_mnk=(1, cluster_n, 1) to the ctor (num_mcast_ctas_a=cluster_N ->
        # A-multicast; cluster_M=1 so B is not multicast on the non-contiguous per-peer M bases); here we
        # validate that + bake run_j. get_scheduler_arguments injects run_j; the coupled store (drain OFF)
        # is unchanged in 3.1a (build_D_copy_fn falls through). The cooperative cluster drain is 3.1b.
        self._a2a_cluster_drain = bool(cluster_drain)
        if cluster_drain:
            _cn = int(cluster_n) if cluster_n is not None else int(self.cluster_shape_mnk[1])
            if _cn < 1:
                raise ValueError(f"cluster_drain cluster_n must be >=1; got {_cn}.")
            # UNREASONABLE-HYPER-PARAMETER guard (NOT an input-shape constraint). cluster_n is an AUTOTUNE
            # knob (concentration width), orthogonal to the input tensor sizes; guarding a never-deploy value
            # restricts NO N/D/cp shape (the FIRST PRINCIPLE's 16-B-only shape rule is untouched). cluster_n>=8
            # NEVER wins the perf curve (over-concentrates: too few NIC put-issuers starve the O(N²) drain
            # AND too little GEMM parallelism starves the O(N³) compute -> ties/loses at every N; measured
            # cross-node K=N, [[project_76d_multislot_outcome]] "cluster_n=8 NEVER wins, ties@24k"). The
            # deployable autotune set is cluster_n in {1,2,4}; reject >=8 LOUDLY so a dispatcher/autotuner
            # never spends a compile+run on it (and never over-concentrates in production). This is a
            # hyper-param reject like "don't pick pingpong at tiny M", never a capability boundary.
            if _cn >= 8:
                raise ValueError(
                    f"cluster_drain cluster_n={_cn} is an unreasonable autotune hyper-parameter (>=8 "
                    f"over-concentrates: NEVER wins the cross-node K=N perf curve, ties/loses at every N -- "
                    f"project_76d_multislot_outcome). The deployable cluster_n autotune set is {{1,2,4}}; "
                    f"pick cluster_n<=4. (This rejects an autotune KNOB, not an input shape -- every "
                    f"N_token/D/cp stays supported at 16-B alignment via cluster_n<=4.)"
                )
            if int(self.cluster_shape_mnk[0]) != 1 or int(self.cluster_shape_mnk[1]) != _cn:
                raise ValueError(
                    f"cluster_drain requires an N-axis cluster cluster_shape_mnk=(1, {_cn}, 1) "
                    f"(cluster_M=1 so B is not multicast on the non-contiguous per-peer M bases; "
                    f"cluster_N=cluster_n for A-multicast); got cluster_shape_mnk={self.cluster_shape_mnk}."
                )
            self._a2a_cluster_n = _cn
            # BOUNDED band = one peer_j width in CLUSTER-tile units = ceil(N_j_loc/(cluster_n*tile_n)).
            # N_j_loc = N//cp1 (2-D) or full N (1-D); tile_n = the CTA N-tile. The scheduler counts
            # j-CLUSTERS (ncluster_n = ceil(ntile_n/cluster_n) = cp1*run_j) so run_j | ncluster_n
            # (runs_per_band = cp1) -- the divisor the sub-band decode requires (5ec4944 / b275675).
            _tile_n = int(self.tile_shape_mn[1])
            _N_j_loc = (
                (int(self._a2a_N) // self._a2a_cp1) if self._a2a_cp1 > 1 else int(self._a2a_N)
            )
            if _N_j_loc <= 0:
                raise ValueError(
                    "cluster_drain requires N (the full token-j extent) at config time to size the bounded "
                    f"band; got _a2a_N={self._a2a_N}."
                )
            self._a2a_cluster_run_j = (_N_j_loc + _cn * _tile_n - 1) // (_cn * _tile_n)
            # (d) MULTI-SLOT: removes the even-shard gate by per-tile peer-routing into 2 rotating
            # full-peer slots (d-fullpeer). REQUIRES cluster_drain (it re-parameterizes the cluster
            # producer/drain). Orthogonal to db/mw for now (single-buffer per-slot). Default off ->
            # single-slot even-shard path byte-identical. (Producer/drain restructure is WIP.)
            self._a2a_cluster_multislot = bool(cluster_multislot)
            # (d): the 2-slot producer arrives full[prev_peer&1] on each peer-cross, but the walk's VERY
            # LAST peer has no next tile to cross into -> the MMA-warp tail (tail_drain_role) flushes it.
            # Enable the parent's post-mainloop tail hook (default off elsewhere -> byte-identical). ROUNDPARK
            # detects band completion via the per-CTA subtile counter (last_in_band fires on EVERY band incl.
            # the last) so its tail is a no-op; the hook stays wired (harmless) for uniformity.
            if self._a2a_cluster_multislot:
                self._a2a_drain_tail = True
            # (d) 2-SLOT CAP (config bound, NOT a shape gate): the 2 live slots (slot = peer_j & 1)
            # are sufficient ONLY when a cluster's cluster_n CONSECUTIVE N-tiles span <=2 ADJACENT peers,
            # which requires cluster_n <= nt_j_pp (per-peer_j tile count). At cluster_n > nt_j_pp a round
            # spans >2 peers -> peer_j&1 COLLIDES (peers j, j+2 -> slot 0) -> corruption. The DISPATCHER
            # (heuristic/autotuner) must never pick cluster_n > nt_j_pp; cluster_n=1 always valid so ANY N
            # runs (up to nt_j_pp available; at large N nt_j_pp>=16 so cluster_n up to 8 always available).
            # It's a config bound like tile_m<=M, not an N constraint. Asserted at the compile-anchor
            # nt_j_pp; DYNAMIC-N caller-contract: keep cluster_n <= nt_j_pp at EVERY runtime N.
            if self._a2a_cluster_multislot:
                _ntj_anchor = (_N_j_loc + _tile_n - 1) // _tile_n
                if _cn > _ntj_anchor:
                    raise ValueError(
                        f"cluster_multislot 2-slot cap: cluster_n ({_cn}) must be <= nt_j_pp "
                        f"({_ntj_anchor} = ceil(N_j_loc={_N_j_loc}/tile_n={_tile_n})) so a cluster spans "
                        f"<=2 adjacent peers; the dispatcher must pick cluster_n<=nt_j_pp (cluster_n=1 "
                        f"always valid -> any N runs). Under dynamic-N keep it <= nt_j_pp at every N."
                    )
            # 3.1b composition: the cluster producer TMA-S2G's into the per-cluster staging + signals
            # cluster-rank-0's consumer-warpgroup drain, which put_warp's over the LOCAL recv. That needs
            # the decoupled consumer warpgroup (decoupled=True), consumer_strided=True (so recv_local +
            # pe_table_dev are forwarded to the drain), arbitrary_n=True (meta field-5 gi_base; nfields==6),
            # pe_aligned_tiling=True (the 2-D per-peer scheduler grid), and ib_quiet=True (BLOCKING put so
            # the drained slice is safe to reuse). Validated loudly here (config-time Python).
            _need = {
                "decoupled": bool(self._a2a_decoupled_store),
                "consumer_strided": bool(self._a2a_consumer_strided),
                "consumer_strided_putwarp": bool(
                    getattr(self, "_a2a_consumer_strided_putwarp", False)
                ),
                "arbitrary_n": bool(self._a2a_arbitrary_n),
                "pe_aligned_tiling": bool(self._pe_aligned_tiling),
                "ib_quiet": bool(getattr(self, "_a2a_ib_quiet", False)),
            }
            _missing = [k for k, v in _need.items() if not v]
            if _missing:
                raise ValueError(
                    "cluster_drain=True requires the decoupled pe_aligned ib_ring stack (decoupled=True, "
                    "consumer_strided=True, consumer_strided_putwarp=True, arbitrary_n=True, "
                    f"pe_aligned_tiling=True, ib_quiet=True); missing/false: {_missing}."
                )
            if self._a2a_cp1 > 1 and not bool(self._a2a_dynamic):
                raise NotImplementedError(
                    "cluster_drain 2-D (cp1>1) requires dynamic=True (the per-tile column/gi decode reads "
                    "N_i_loc/N_j_loc off the runtime recv); static-N 2-D bounded band is a follow-up (3.1c)."
                )
            # EVEN-SHARD bounded band (LOUD gate, not a convention): "one run == one peer_j" holds only
            # when cluster_n*run_j == nt_j_pp, i.e. nt_j_pp % cluster_n == 0. With run_j =
            # ceil(nt_j_pp/cluster_n), an ODD nt_j_pp gives cluster_n*run_j > nt_j_pp so a run OVERSHOOTS
            # into the next peer_j -> the drain's single-peer_j-per-run meta mis-routes (the run_j sub-band
            # silent-zero class). Static check at the compile anchor; cluster_drain 2-D is DYNAMIC, so this
            # is a CALLER-CONTRACT: EVERY runtime N MUST keep ceil((N//cp1)/tile_n) % cluster_n == 0.
            # Arbitrary nt_j_pp is 3.1c. (16-B/%8 shard floors are unchanged; this is an even-tile-count
            # gate, surfaced-and-gated, not a silent constraint.)
            # (d) MULTISLOT bypasses the even-shard gate: the per-tile peer-routing into the 2
            # rotating full-peer slots (slot=peer_j&1) + the FULL-BAND cluster walk (run_j_dynamic below)
            # let a cluster STRADDLE peer_j boundaries at ARBITRARY nt_j_pp (only the 2-slot CAP cluster_n
            # <= nt_j_pp applies, asserted above). So the even-shard gate is the SINGLE-slot bounded-band
            # path's constraint ONLY -> skip it for multislot.
            _nt_j_pp = (_N_j_loc + _tile_n - 1) // _tile_n
            if (not self._a2a_cluster_multislot) and _nt_j_pp % _cn != 0:
                raise NotImplementedError(
                    f"cluster_drain: per-peer_j tile count nt_j_pp={_nt_j_pp} must be divisible by "
                    f"cluster_n={_cn} (even-shard bounded band); odd nt_j_pp straddles peer_j boundaries "
                    f"(one run overshoots into the next peer_j -> the single-peer_j-per-run drain "
                    f"mis-routes). Arbitrary nt_j_pp is #76 (d) cluster_multislot=True. Caller-contract "
                    f"(cluster_drain 2-D is dynamic): keep (N//cp1) s.t. ceil((N//cp1)/tile_n) % "
                    f"cluster_n == 0 at EVERY runtime N."
                )

        # STOP-THE-LINE tile_m guard (grid): the DECOUPLED fused drain (cluster_drain OR coalesce)
        # builds its epilogue store + peer-route + the 128-row symmetric staging for a SINGLE epilogue M-subtile.
        # A CTA tile with tile_m > epi_tile_m spans ceil(tile_m/epi_tile_m) subtiles, and epi_tile_m > 128 is a
        # subtile bigger than the 128-row staging — the single-subtile drain MIS-ROUTES either case →
        # INVOCABLE-BUT-WRONG (it compiles, launches, and returns garbage rel_L2~2240 instead of raising, and
        # under --perf-only at large N there is no correctness gate so it would time as "ok" and could win the
        # sweep, tainting the perf answer). Reject it at CONFIG (mirroring the pingpong tile_n>208 ValueError)
        # so the compile-gate records ERROR/skip → it never compiles-to-garbage, never times, never wins.
        # Building the multi-epi-subtile drain to make tile_m>128 actually WORK is a separate perf lever,
        # deliberately NOT done here. epi_tile_m mirrors the parent _sm90_compute_tile_shape_or_override.
        if self._a2a_cluster_drain:
            _tm = int(self.tile_shape_mn[0])
            _atom_m = int(self.atom_layout_mnk[0])
            if _tm % 128 == 0 and _atom_m > 1:
                _epi_m = math.gcd(128, _tm)
            elif _tm % 192 == 0 and _atom_m > 1:
                _epi_m = math.gcd(192, _tm)
            else:
                _epi_m = math.gcd(64, _tm)
            # #78 MULTI-EPI-SUBTILE drain: tile_m > epi_tile_m (m_sub_per_tile>1) is now SUPPORTED on:
            #  (A) the EVEN-SHARD SINGLE-BUFFER cluster_drain — the per-cluster staging gains an m_sub AXIS
            #      (n_clusters, m_sub, epi_m, N_j_loc) and the drain inner-loops the m_sub subtile-bands
            #      (gi_base_s = tile_base + sub_m*epi_m) [Option A']; and
            #  (B) the COALESCE drain (non-handshake_skip) — the per-CTA rotating ring GROWS its M extent
            #      epi_m -> m_sub*epi_m [grow-rows Approach (a); a 5-D atom is avoided since the ring already
            #      carries slot/cp1 axes], the producer writes epi-box sub_m to ring row-band sub_m, and the
            #      drain walks m_sub*epi_m rows. Both keep the run-count / scheduler / mbar counts+phases
            #      UNCHANGED (deadlock-safe; sub_m lives below scheduler tile granularity).
            # Still REJECTED for: coalesce + handshake_skip (#57 2b per-band is_band_remote needs sub_m=0
            # across the band — see the :1505 assert), the db/mw/multislot/roundpark cluster variants (their
            # subtile-axis is a later rung), and epi_tile_m > 128 (a subtile taller than the staging) always.
            _even_shard_sb = self._a2a_cluster_drain and not getattr(
                self, "_a2a_cluster_multislot", False
            )
            # #78: tile_m>epi_m (multi-epi-subtile) is validated on the even-shard single-buffer cluster_drain
            # AND the non-handshake_skip coalesce, at BOTH tile_n<=128 (256×128) and tile_n=256 (256×256),
            # 2-node IB cp=(2,8). tile_n=256 drives the 3-WG producer-warp drain
            # ([[reference_a2a_4wg_tile256_register_wall]]) — validated in isolation at 128×256 (m_sub=1) and
            # COMPOUNDED with the m_sub=2 row-walk at 256×256; the coalesce drain walks the full-N band
            # row-by-row (N_j width from recv, tile_n-independent), so tile_n only changes the producer's
            # n_sub_per_tile staging + the host-computed 3-WG warp layout. No tile_n gate (honest: validated).
            _tn = int(self.tile_shape_mn[1])
            _msub_ok = _even_shard_sb and _epi_m <= 128 and _tm % _epi_m == 0
            if _epi_m > 128 or (_tm > _epi_m and not _msub_ok):
                raise ValueError(
                    "fused A2A drain (cluster_drain/coalesce) requires the CTA tile to be a SINGLE epilogue "
                    "M-subtile of <=128 rows (single-subtile drain + 128-row staging), OR — for the even-shard "
                    "single-buffer cluster_drain / non-handshake_skip coalesce — a tile_m that is an exact "
                    f"multiple of epi_tile_m<=128 (#78 multi-epi-subtile drain, any tile_n); tile_m={_tm} "
                    f"tile_n={_tn} epi_tile_m={_epi_m} spans {-(-_tm // _epi_m)} subtile(s) on an unsupported "
                    "drain variant (silent-wrong if compiled; db/mw/multislot/roundpark + handshake_skip)."
                )
        # 1-D EXACT-PARTITION guard. LAST in this method, and the position is load-bearing.
        #
        # The i axis is the only cp-split axis in a 1-D mesh, so the cp peer blocks must TILE the token
        # extent: `cp * N_loc == N`. Without this a caller that floors `N/cp` is ACCEPTED and the store
        # never visits the remainder -- MEASURED at cp=16, N=2072 (N/cp = 129.5): N_loc=129 covers
        # 16*129 = 2064 of 2072 tokens and the last 8 are SILENTLY DROPPED. Nothing downstream can
        # notice, because the recv is SIZED from N_loc: the missing rows are not short, they are absent
        # from the geometry. The cp1>1 branch above has carried the analogous guards all along
        # (`N_loc == N//cp0`, `N % cp1 == 0`), which is what makes the 1-D omission an omission.
        #
        # WHY LAST, and not beside its sibling in the 1-D branch (where it was first written): a config
        # can be unsupported for SEVERAL reasons at once, and the earliest guard is the one whose
        # message the caller sees. Placed early it PREEMPTED the tile/subtile refusal above -- measured:
        # 4 tests of the declared-unsupported region
        # (tile 256x{128,256} x cluster_multislot x cp=16 x N_token=2072) started failing with
        # "Regex pattern did not match. Expected: 'single-subtile drain|multi-epi-subtile'", because
        # 2072 does not partition at cp=16 either. Those regions declare a TILE refusal; taking their
        # message away turns a specific diagnosis into a generic one.
        #
        # Reject incomplete 1-D token partitions before any tokens can be dropped. This guard is
        # fail-SAFE (a refusal replacing a wrong answer), and it rejects NOTHING on the production
        # path: all eight 1-D (cp, N, N_loc)
        # triples the 40-cell acceptance grid elects satisfy cp*N_loc == N exactly (STEP 1's
        # w8plan/step1/variants.csv), because the workflow PADS N before sharding -- which is
        # presumably why `main` never tripped over the missing check.
        if self._a2a_cp1 <= 1 and int(self._a2a_N) and int(cp) * int(N_loc) != int(self._a2a_N):
            raise ValueError(
                f"1-D configure_a2a_gemm_native requires the cp peer blocks to TILE the token "
                f"extent exactly: cp*N_loc == N; got cp={int(cp)}, N_loc={int(N_loc)} "
                f"(cp*N_loc={int(cp) * int(N_loc)}), N={int(self._a2a_N)} -- "
                f"{abs(int(self._a2a_N) - int(cp) * int(N_loc))} token(s) would be dropped "
                f"silently. Pass an N divisible by cp (N//cp={int(self._a2a_N) // int(cp)})."
            )

    def configure_a2a_sharded(
        self,
        device_mesh,
        placements,
        *,
        pe_map=None,
        B=1,
        N=None,
        gemm_native=True,
        rows_per_peer=None,
        dynamic=False,
        consumer_warpgroups=None,
        pe_aligned_tiling=False,
        dyn_cp=False,
        ib_ring=False,
        ring_depth=None,
    ):
        """Generic 1D/2D DTensor token-sharding entry for the back-A2A store.

        Parses a torch ``DeviceMesh`` + DTensor ``placements`` into the cp geometry
        (via :class:`fold_cp_ops.distributed.pe_map.PeMap`) and bakes BOTH the flat-cp
        scalars (``cp`` / ``my_cp_rank`` / ``pe_table``) AND the per-axis token-shard
        geometry the store needs to route a CTA tile to its peer. Subsumes the 1-D
        :meth:`configure_a2a_gemm_native` / :meth:`configure_a2a` as the
        ``cp_axis_sizes == (cp,)`` special case (byte-identical recv).

        Token sharding (TriMul): the i (GEMM-M) and j (GEMM-N) token axes are sharded
        by the placement ``Shard`` dims. For a 1-D ``cp`` mesh sharding tensor-dim 1
        only -> ``cp_axis_sizes == (cp,)``, i alone is split (``N_i_loc = N // cp``),
        j is full. For a 2-D ``(cp0, cp1)`` mesh sharding ``(Shard(1), Shard(2))`` ->
        ``cp_axis_sizes == (cp0, cp1)``, the i axis is split into ``cp0`` blocks of
        ``N_i_loc = N // cp0`` and the j axis into ``cp1`` blocks of ``N_j_loc = N //
        cp1``. The recv stays the flat-cp 5-D ``(cp, Dloc, B, N_i_loc, N_j_loc)`` (the
        2-D mesh is invisible to the symmetric alloc — same shape on every rank); the
        2-D structure lives ONLY in the tile->peer + recv-index map in the copy_fn.

        Parameters
        ----------
        device_mesh : torch.distributed.device_mesh.DeviceMesh
            The cp mesh (1-D ``(cp,)`` or N-D ``(cp0, cp1, ...)``).
        placements : Sequence
            One placement per mesh dim. The ``Shard`` dims are the cp (token) axes;
            feature dim (3) MUST NOT be sharded (rejected). Flattened row-major.
        pe_map : PeMap, optional
            Prebuilt PeMap; if ``None`` it is built from (device_mesh, placements).
        B : int
            TriMul batch (splits the GEMM L axis ``d = L // B``, ``b = L % B``) for the
            GEMM-native store.
        N : int
            The (square) global token extent. Required for ``gemm_native`` (sets the
            per-axis ``N_i_loc = N // cp0`` and ``N_j_loc = N // cp1``).
        gemm_native : bool
            ``True`` (default) -> the design-E 5-D GEMM-native store; ``False`` -> the
            plain L=1 token-major back store (1-D only; ``rows_per_peer`` required).
        rows_per_peer : int, optional
            For ``gemm_native=False`` only: token rows per peer block on the GEMM M
            axis (``M // cp``).

        Raises
        ------
        RuntimeError
            If nvshmem4py / the vendored nvshmem utils are not importable.
        ValueError
            If the feature dim is sharded, a per-axis block straddles a CTA tile, or
            (plain) ``rows_per_peer`` is missing.
        """
        if not HAS_NVSHMEM:
            raise RuntimeError(
                "GemmSm90A2A.configure_a2a_sharded requires nvshmem4py + the vendored nvshmem utils."
            )
        pe_map = self._resolve_pe_map(device_mesh, placements, pe_map)
        cp = int(pe_map.cp)
        my_cp_rank = int(pe_map.my_cp_rank)
        pe_table = tuple(int(x) for x in pe_map.cp_pe_table.tolist())
        cp_axis_sizes = tuple(int(s) for s in pe_map.cp_axis_sizes)
        if not gemm_native:
            raise ValueError(
                "configure_a2a_sharded(gemm_native=False) (the plain L=1 token-major back store) "
                "was removed; only the design-E GEMM-native store remains."
            )
        if N is None:
            raise ValueError("configure_a2a_sharded(gemm_native=True) requires N (token extent).")
        # Per-axis token-block extents from the cp-axis sizes (1-D => cp1 absent => j full).
        cp0 = cp_axis_sizes[0]
        cp1 = cp_axis_sizes[1] if len(cp_axis_sizes) > 1 else 1
        if len(cp_axis_sizes) > 2:
            raise ValueError(
                f"back GEMM-native store supports 1-D or 2-D token sharding; got "
                f"cp_axis_sizes={cp_axis_sizes} ({len(cp_axis_sizes)} cp axes)."
            )
        N = int(N)
        if N % cp0 != 0 or N % cp1 != 0:
            raise ValueError(f"N={N} must be divisible by both cp axes {cp_axis_sizes}.")
        N_i_loc = N // cp0
        N_j_loc = N // cp1
        cta_tile_m, cta_tile_n = self.tile_shape_mn[0], self.tile_shape_mn[1]
        # HALO-CACHE / PE_ALIGNED handle STRADDLING per-axis blocks (N_i_loc/N_j_loc need NOT be
        # tile-aligned). halo_cache absorbs the overflow; pe_aligned re-bases the tile schedule per peer
        # on BOTH axes (ceil tiles/block) so no tile straddles + the recv descriptor clamps the partial
        # last per-peer tile. So the alignment asserts are RELAXED for both (arbitrary 2-D straddle).
        if not pe_aligned_tiling:
            if N_i_loc % cta_tile_m != 0:
                raise ValueError(
                    f"N_i_loc=N/cp0={N_i_loc} must be a positive multiple of cta_tile_M="
                    f"{cta_tile_m} (a CTA M-tile must map to ONE peer block on the i axis). "
                    f"Pass halo_cache=True for a straddling per-axis block."
                )
            if cp1 > 1 and N_j_loc % cta_tile_n != 0:
                raise ValueError(
                    f"N_j_loc=N/cp1={N_j_loc} must be a positive multiple of cta_tile_N="
                    f"{cta_tile_n} (a CTA N-tile must map to ONE peer block on the j axis). "
                    f"Pass halo_cache=True for a straddling per-axis block."
                )
        if len(pe_table) != cp:
            raise ValueError(f"pe_table {pe_table} length must equal cp={cp}.")
        # IB-RING (§0.9 hybrid NVLink+IB store): 1-D token shard delegates to the 1-D
        # configure_a2a_gemm_native ib_ring path (it owns the full decoupled putwarp + pe_aligned
        # wiring — no duplication). NOTE (§0.9.5 cleanup): this delegation is now a DEAD PATH — it
        # passes NO coalesce, so configure_a2a_gemm_native's ib_ring gate rejects the bare uniform store
        # (removed as the §0.9.5 loser). The keeper hybrid store (coalesce_dyn) is reached via
        # configure_a2a_gemm_native DIRECTLY; plumbing coalesce
        # + a 2-D (i,j)->peer unravel through configure_a2a_sharded is a separate follow-up. The 2-D
        # refusal below still fires first for a 2-D mesh (never a silent wrong store).
        if ib_ring:
            if cp1 > 1:
                raise NotImplementedError(
                    "ib_ring is 1-D-token-shard only for now (cp1==1); the decoupled put_nbi_warp "
                    "drain routes rows by a 1-D gi->peer map. 2-D (cp0,cp1) ib_ring needs a 2-axis "
                    "tile->peer unravel over the ring drain (follow-up). Got cp_axis_sizes="
                    f"{cp_axis_sizes}."
                )
            self.configure_a2a_gemm_native(
                cp=cp,
                my_cp_rank=my_cp_rank,
                B=int(B),
                N_loc=N_i_loc,
                pe_table=pe_table,
                dynamic=dynamic,
                arbitrary_n=True,
                pe_aligned_tiling=True,
                ib_ring=True,
                decoupled=True,
                producer_tma=True,
                consumer_strided=True,
                consumer_strided_putwarp=True,
                ring_depth=ring_depth,
                consumer_warpgroups=consumer_warpgroups,
                N=N,
            )
            return
        # LayoutRightMap over cp_axis_sizes: the authoritative row-major flatten of the
        # (cp0, cp1) coordinate -> the flat peer index (matches PeMap's own flatten).
        unravel = LayoutRightMap(cp_axis_sizes)
        self._a2a_enabled = True
        self._a2a_gemm_native = True
        self._a2a_cp = cp
        self._a2a_my_cp_rank = my_cp_rank
        self._a2a_B = int(B)
        self._a2a_dynamic = bool(dynamic)
        # ---- TRACK-B dynamic-cp flag (spec §B2): runtime cp peer loops; implies dynamic-shape. ----
        if dyn_cp and not self._a2a_dynamic:
            raise ValueError("dyn_cp=True requires dynamic=True (dyn_cp implies dynamic-shape).")
        # dyn-cp is WONTFIX for EVERY A2A store (pe_aligned + the coupled TMA-S2G store): the reshard
        # peer atoms are a compile-time host list of TMA descriptors with per-peer BAKED symmetric base
        # addresses (no runtime-base TMA-S2G), and pure-cp A2A has cp==world so a different cp is a
        # different nvshmem world = a different launch. Use compile-per-cp. Refuse loudly here (the dynamic-cp
        # verdict) so a dyn_cp=True request never silently no-ops.
        if dyn_cp:
            raise NotImplementedError(
                "dynamic-cp is WONTFIX for A2A stores: use compile-per-cp. Peer atoms are a "
                "compile-time host list with per-peer baked symmetric bases, and pure-cp cp==world."
            )
        self._a2a_dyn_cp = bool(dyn_cp)
        self._a2a_N_loc = N_i_loc  # back-compat alias (the i-axis peer-block extent)
        self._a2a_N_i_loc = N_i_loc
        self._a2a_N_j_loc = N_j_loc if cp1 > 1 else 0  # 0 => j unsharded (1-D)
        self._a2a_rows_per_peer = N_i_loc
        self._a2a_pe_table = pe_table
        self._a2a_cp_axis_sizes = cp_axis_sizes
        self._a2a_cp_unravel_shape_stride = (
            tuple(int(s) for s in unravel.shape),
            tuple(int(s) for s in unravel.strides),
        )
        if N is not None:
            self._a2a_N = int(N)
        # ---- Track A A2: 2-D pe_aligned per-peer tiling (BOTH token axes). Every CTA (i,j) tile maps to
        # ONE peer block via ceil tiles-per-block; the mainloop A-row / B-col shifts re-base the loads to
        # the per-peer origin (peer_i*N_i_loc, peer_j*N_j_loc); the store rides the CLEAN non-arbitrary_n
        # 2-D copy_fn (LayoutRightMap peer unravel) whose recv descriptor clamps the partial-last tile.
        # cluster_M must be 1 (per-peer M bases are non-contiguous -> would break a B-multicast-on-M).
        # 1-D (cp1==1) collapses to byte-identical (nt_j_pp=ceil(N/tile_n)=uniform, peer_j=0 -> shift 0;
        # aligned N_i_loc -> nt_i_pp=floor, A-shift 0). Does NOT set arbitrary_n (uses the non-arb store).
        if pe_aligned_tiling:
            if self.cluster_shape_mnk[0] != 1:
                raise ValueError(
                    f"pe_aligned_tiling requires cluster_M==1 (no B-multicast-on-M); got "
                    f"cluster_shape_mnk={self.cluster_shape_mnk}."
                )
            self._pe_aligned_tiling = True
            N_j_loc_eff = N_j_loc if cp1 > 1 else N  # cp1==1 -> j full (peer_j always 0)
            self._a2a_nt_i_pp = (N_i_loc + cta_tile_m - 1) // cta_tile_m
            self._a2a_nt_j_pp = (N_j_loc_eff + cta_tile_n - 1) // cta_tile_n
        if consumer_warpgroups is not None:
            self._a2a_consumer_warpgroups = int(consumer_warpgroups)
        # ---- 2-D TMA-S2G 16-B partial-tile guard (correctness) — placed at the END of configure so
        # _pe_aligned_tiling is resolved (@ the A2 branch). Each 2-D pe_aligned store writes a straddling
        # box whose INNERMOST (stride-1) origin must be 16-B aligned -> the per-peer extent must be %8==0
        # (bf16): the partial per-peer tile on BOTH axes, guarded as a SAFE SUPERSET (exact for the square
        # TriMul grid). DYNAMIC caller-contract: this static check sees only the compile-anchor N
        # (necessary-not-sufficient) -> the caller MUST ensure (N//cp0)%8==0 AND (N//cp1)%8==0 at EVERY
        # runtime N. Otherwise the store SILENTLY corrupts (uniform-scale garbage; the per-row outlier/
        # coverage gate misses it, only scalar rel_L2 catches it).
        if cp1 > 1 and self._pe_aligned_tiling:
            if N_i_loc % 8 != 0 or N_j_loc % 8 != 0:
                raise ValueError(
                    f"2-D pe_aligned store requires 16-B-aligned per-peer extents: N_i_loc=N//cp0="
                    f"{N_i_loc} and N_j_loc=N//cp1={N_j_loc} must both be multiples of 8 (bf16); got "
                    f"N_i_loc%8={N_i_loc % 8}, N_j_loc%8={N_j_loc % 8} (silent corruption otherwise)."
                )
        # §5.2 / B1 — TOPOLOGY guard, LAST so the store kind is fully resolved. This arm never sets
        # _a2a_decoupled (the ib_ring arm returned above after delegating to configure_a2a_gemm_native,
        # which guards itself), so everything reaching here is a COUPLED store — including
        # pe_aligned_tiling, which faults independently through its own descriptor path. See
        # _guard_coupled_nvlink_only for why this is the fix and not a trigger-gate dodge.
        self._guard_coupled_nvlink_only(pe_table, entry="configure_a2a_sharded")

    @staticmethod
    def _resolve_pe_map(device_mesh, placements, pe_map):
        """Build (or validate) the cp PeMap for the sharded entry; reject feature shard."""
        from fold_cp_ops.distributed.dtensor_adapter import validate_trimul_sharding
        from fold_cp_ops.distributed.pe_map import PeMap

        # Reject a feature-dim-3 shard (TriMul keeps the feature axis replicated); a
        # placement-generic guard reused from the DTensor adapter (also requires >=1
        # token axis sharded). ndim_mesh == len(placements).
        validate_trimul_sharding(placements, len(placements))
        if pe_map is None:
            pe_map = PeMap.from_mesh_placements(device_mesh, placements)
        return pe_map

    # ------------------------------------------------------------------
    # The D-store-build seam override (parent gemm_sm90.py build_D_copy_fn).  Returns the
    # peer-store copy_fn when _a2a_enabled, else super()'s byte-identical local store.
    # The peer atoms/tensors ride on epi_params (attached host-side above).
    # ------------------------------------------------------------------
    def build_D_copy_fn(
        self,
        tma_atom_d,
        mD_mnl,
        batch_idx,
        sD,
        tile_coord_mnkl,
        epi_params,
        storage=None,
    ):
        if const_expr(not self._a2a_enabled):
            # Byte-identical local store (the parent default).
            return super().build_D_copy_fn(
                tma_atom_d,
                mD_mnl,
                batch_idx,
                sD,
                tile_coord_mnkl,
                epi_params,
                storage,
            )
        # #57: AND-in has_ib_peers -- an ALL-P2P job (no IB peer, cp<=8) SKIPS the decoupled/cluster/
        # differential PRODUCER block (whose ring/consumer is const_expr-elided by the gates) and FALLS
        # THROUGH to the design-E _a2a_peer_store_copy_fn_gemm_native = the shared NVLink store (= pe_aligned).
        # The atom build (_build_peer_store_atoms_gemm_native) likewise builds REAL peer atoms (not None) when
        # not has_ib_peers, so the fall-through store has its TMA atoms. This is the store side of the collapse.
        if const_expr(
            self._a2a_enabled
            and self._a2a_gemm_native
            and self._a2a_decoupled_store
            and getattr(self, "_a2a_has_ib_peers", True)
        ):
            # CLUSTER-DRAIN (3.1b): the cluster-cooperative producer writes each subtile into the
            # SHARED per-cluster staging (cluster_n CTAs tile up one peer_j width) + signals
            # cluster-rank-0, which drains. Its own producer copy_fn; no differential/ring path.
            if const_expr(getattr(self, "_a2a_cluster_drain", False)):
                return self._a2a_cluster_producer_copy_fn(
                    epi_params,
                    self.cta_tile_shape_mnk[:2],
                    self.epi_tile,
                    sD,
                    tile_coord_mnkl,
                    storage,
                )
            # DECOUPLED STORE (step 2): the MMA warpgroup (producer) writes this epilogue
            # tile to a bounded local-GMEM ring + signals the dedicated consumer warpgroup
            # (which drains ring->peer). Gated on the SEPARATE _a2a_decoupled_store flag so
            # STEP 1 (warpgroup present + idle, _a2a_decoupled only) keeps the COUPLED store and
            # isolates the layout-viability proof from the ring wiring.
            decoupled_copy = self._a2a_decoupled_producer_copy_fn(
                epi_params, self.cta_tile_shape_mnk[:2], self.epi_tile, sD, tile_coord_mnkl, storage
            )
            return decoupled_copy
        # design-E: per-peer S2G into the 5-D GEMM-native recv (the ONLY remaining coupled store; the
        # plain L=1 path was removed). Each CTA's (i,j) tile of plane L lands at recv[my_cp_rank, L//B,
        # L%B, i_local, j] on peer i//N_loc (full (N_loc,N) box; the L-coord un-bake is the (d,b) SELECT).
        copy_D, _, _ = self._a2a_peer_store_copy_fn_gemm_native(
            epi_params.peer_atoms,
            epi_params.peer_tensors,
            self.cta_tile_shape_mnk[:2],
            self.epi_tile,
            sD,
            tile_coord_mnkl,
            raw_tensors=epi_params.peer_raw_tensors,
            route2_atoms=epi_params.route2_atoms,
            route2_tensors=epi_params.route2_tensors,
            route2_dynstrip_ws=epi_params.route2_dynstrip_ws,
            route2_cache_ws=epi_params.route2_cache_ws,
            storage=storage,
        )
        return copy_D

    # ==================================================================
    # DECOUPLED producer/consumer peer store — warpgroup-split hooks.
    # ==================================================================
    # Per-slot metadata fields the producer writes for the consumer (peer + recv index):
    #   [0]=peer  [1]=i_tile  [2]=j_tile  [3]=d  [4]=b   (5 Int32 per ring slot).
    # arbitrary_n (opt-in) appends [5]=gi_base (the epi-box's GLOBAL token-i base) and sets the INSTANCE
    # attribute to 6 (see configure_a2a_gemm_native) so the per-row SIMT drain can route each row to
    # peer = gi//N_loc / i_local = gi%N_loc; the default-off path keeps this class default (5).
    _DECOUPLED_META_FIELDS = 5
    # A free named-barrier id (> NamedBarrierGemm.TmemPtr=7) for the consumer warpgroup's internal
    # sync. The MMA/epilogue use ids 1..7; this is disjoint so there is no cross-warpgroup aliasing.
    _DECOUPLED_CONSUMER_BARRIER_ID = 9

    @property
    def _DECOUPLED_CONSUMER_WARPS(self):
        """Total consumer warps = 4 * consumer_warpgroups. The SIMT GMEM->peer drain is parallelized
        across ALL of them (a2a.py reaches ~295 GB/s only with many warps in flight). Used for the
        empty[s] arrival count + the put tiled-copy thread count. Scaling adds draining warps.

        WS-B 3-WG wide-tile: the drain rides the producer WG's spare warps (its 4 warps minus the
        num_ab_load_warps TMA-load warp(s) == 3), NOT a dedicated 4-warp warpgroup."""
        if self._drain_on_producer_wg():
            return 4 - int(self.num_ab_load_warps)
        return 4 * int(getattr(self, "_a2a_consumer_warpgroups", 1))

    def _decoupled_sring_bytes(self):
        """Extra-SMEM bytes to RESERVE in _compute_stages so ab_stage shrinks (else total SMEM
        overflows the 228 KB cap -> CUDA_ERROR_INVALID_VALUE at launch). Covers:
          * Option B TMA-drain: the SMEM ring sRing = rd slots.
          * A′ double-TMA consumer: the SMEM bounce = bounce_bufs slots.
          * Deep rotating ring (ring_depth > 2): the rd-deep full/empty mbarriers + per-slot meta
            (the ``decoupled`` struct itself), which is O(rd) -> NON-negligible at a large rd. On the
            bounded ring (rd~=2) it is ~76 B (absorbed by alignment) -> ignored (byte-identical).
        Per-slot bytes == one epi stage (cute.size(epi_tile)*d_bytes), the parent's d_bytes_per_stage.
        0 on the SIMT-drain / non-decoupled paths (GMEM ring, no SMEM -> byte-identical).
        #57: ALSO 0 when _a2a_has_ib_peers is False (all-P2P job, cp<=8) -> the ring is elided -> ab_stage
        byte-identical to pe_aligned = the cp<=8 collapse."""
        if const_expr(
            not (
                getattr(self, "_a2a_enabled", False)
                and getattr(self, "_a2a_decoupled", False)
                and getattr(self, "_a2a_has_ib_peers", True)
            )
        ):
            return 0
        # Rotating-ring putwarp seed: the GMEM-ring decoupled struct (full+empty Int64 mbars + meta
        # Int32 + pcount + next_slot) is rd*(8+8) + rd*nfields*4 + 8 bytes. Reserve it (+128 align pad)
        # when ring_depth>2 so the ring's mbarrier/meta SMEM does not overflow ab_stage. At rd<=2 it is
        # ~76 B (absorbed by alignment) -> reserve 0 to keep the default path byte-identical.
        nfields = self._DECOUPLED_META_FIELDS
        if const_expr(self._a2a_ring_depth > 2):
            return self._a2a_ring_depth * (16 + nfields * 4) + 4 + 128
        return 0  # producer-TMA-only bounded ring (rd<=2): GMEM ring, no SMEM theft

    def _compute_stages(self, *args, **kwargs):
        """Reserve the TMA-drain sRing SMEM before the parent sizes ab_stage.

        The parent (@classmethod) fills ab_stage to consume ~all of smem_capacity; it does not know
        about the subclass sRing. Shrink the smem_capacity it sees by the sRing footprint so the
        total (ab + epi + sRing + mbars) fits. Default (no TMA drain) reserves 0 -> byte-identical."""
        reserve = self._decoupled_sring_bytes()
        if const_expr(reserve > 0):
            # smem_capacity is the 8th positional arg of the parent _compute_stages signature.
            args = list(args)
            args[7] = args[7] - reserve
            args = tuple(args)
        return GemmSm90._compute_stages.__func__(self, *args, **kwargs)

    def _precompute_max_irun(self):
        """PROVABLE upper bound on valid i-runs per band for the LEVER-2 descriptor cache (O(1) in N ->
        FIRST-PRINCIPLE clean, never O(N)). Each drain run ends at a 128-row put-cap boundary OR a peer_i
        boundary, and only cp0 peers are valid, so #valid-runs <= ceil(band_rows/128) + cp0. +4 margin.
        (CPU-sim'd max over all N = 3-4 for tile_m=256/cp0=2 -> the bound is ~2x, tiny SMEM.)"""
        band_rows = int(self.cta_tile_shape_mnk[0])
        cp0 = int(getattr(self, "_a2a_cp0", int(getattr(self, "_a2a_cp", 1))))
        return (band_rows + 127) // 128 + cp0 + 4

    def _precompute_slot_ints(self):
        """Int32s per ring slot in the descriptor cache: MAX_IRUN {peer_i,i_local,run,row} quads + 1 count."""
        return self._precompute_max_irun() * 4 + 1

    def _a2a_walk_state_len(self) -> int:
        """#76 (d) MULTISLOT per-CTA walk-state length (``next_slot`` size). Base multislot = 3
        ([0]=prev_peer, [1]=fill[slot0], [2]=fill[slot1]). ==== 1-D FIX (CONFIRMED: cp16 1-D gate 4/4 green) ====
        AT ``cp1==1`` grow to 5, adding [3]=prev_m0 (last M-tile coord) + [4]=prev_L (last L-plane) so the
        producer can detect a NEW drain-band when ``peer_j`` CAN'T change (peer_j is ALWAYS 0 at cp1==1, so
        the ``peer_j != prev_peer`` cross-detector never fires across bands -> the per-band full-arrive /
        reuse-gate / meta-write are skipped -> the cn2 1-D fault). 2-D multislot (cp1>1) stays 3 -> the SMEM
        struct is BYTE-IDENTICAL for 2-D; non-multislot stays 1 (vestigial). Single source of truth for all
        four ns_n sites (_extra_smem_struct / _init_extra_smem / producer copy_fn / tail_drain_role)."""
        if not bool(getattr(self, "_a2a_cluster_multislot", False)):
            return 1
        return 5 if int(getattr(self, "_a2a_cp1", 1)) == 1 else 3

    def _extra_smem_struct(self):
        """Decoupled ring SMEM: full/empty mbarriers + per-slot metadata + producer counter.

        Flag-gated (``_a2a_decoupled``); else the parent default (0-size, byte-identical). #57
        orthogonality: ALSO gated on ``_a2a_has_ib_peers`` -- the all-P2P collapse (cp<=8, no IB peer)
        ELIDES this rd-sized ring struct so the collapsed store's SMEM layout is byte-identical to
        pe_aligned + INVARIANT under IB knobs (rd). Safe at collapse: the consumer role is untraced
        (_num_extra_warpgroups==0), build_D falls through to design-E, tail_drain_role is gated -> nothing
        references storage.decoupled."""
        if const_expr(
            not (
                getattr(self, "_a2a_enabled", False)
                and getattr(self, "_a2a_decoupled", False)
                and getattr(self, "_a2a_has_ib_peers", True)
            )
        ):
            return super()._extra_smem_struct()
        rd: cutlass.Constexpr[int] = self._a2a_ring_depth
        nfields: cutlass.Constexpr[int] = self._DECOUPLED_META_FIELDS
        # GMEM-ring putwarp drain: full/empty mbarriers + per-slot metadata + producer counter in SMEM
        # (the ring itself lives in GMEM). ``next_slot`` is a vestigial 1-Int32 counter kept for struct
        # stability (init'd, never read on the putwarp path).

        # (d) MULTISLOT: repurpose next_slot as the per-CTA persistent walk state -- [0]=prev_peer
        # (last REAL peer_j staged; -1 == none/padding), [1]=fill[slot0], [2]=fill[slot1] (per-slot fill
        # counts driving the empty[slot] reuse-wait phase). Sized 3 (or 5 at cp1==1, the 1-D FIX -- see
        # _a2a_walk_state_len) ONLY under multislot; default keeps the 1-Int32 vestigial next_slot -> the
        # struct is BYTE-IDENTICAL for every non-multislot path (AND for 2-D multislot, which stays 3).
        ns_n: cutlass.Constexpr[int] = self._a2a_walk_state_len()

        # LEVER-2 PRECOMPUTE: an i-run DESCRIPTOR cache {peer_i, i_local, run} (3 Int32/run) + a per-slot
        # count, so the register-rich producer computes the heavy i-axis band-walk ONCE and the light drain
        # warp just reads a descriptor + reconstructs the cheap runtime cp1 j-loop. Bounded by band_rows =
        # tile_m (a COMPILE constant, O(tile_m), N-INDEPENDENT -> FIRST-PRINCIPLE clean, NEVER O(N)). Off ->
        # the descriptor field is ABSENT -> the struct is BYTE-IDENTICAL to today.
        @cute.struct
        class DecoupledSmem:
            full: cute.struct.MemRange[cutlass.Int64, rd]
            empty: cute.struct.MemRange[cutlass.Int64, rd]
            meta: cute.struct.MemRange[cutlass.Int32, rd * nfields]
            pcount: cute.struct.MemRange[cutlass.Int32, 1]
            next_slot: cute.struct.MemRange[cutlass.Int32, ns_n]

        return DecoupledSmem

    def _init_extra_smem(self, storage, warp_idx) -> None:
        """Init the ring's full/empty mbarriers (count 32 each side) + zero the producer counter.

        PLAIN method (the DSL cannot flatten the ``storage`` struct through a ``@cute.jit``
        boundary): it only extracts the SMEM POINTERS (attribute access, no control flow) and hands
        them to the ``@cute.jit`` :meth:`_init_decoupled_mbarriers` (pointers/ints ARE flatten-able),
        which does the (runtime-conditional) init. Done in the pre-warp-split window; the cluster-wide
        pipeline_init_wait (right after) orders it before any producer/consumer use. #57 orthogonality:
        ALSO gated on ``_a2a_has_ib_peers`` -- the all-P2P collapse elides the ring struct (see
        _extra_smem_struct), so there is nothing to init (byte-identical to pe_aligned)."""
        if const_expr(
            not (
                self._a2a_enabled
                and self._a2a_decoupled
                and getattr(self, "_a2a_has_ib_peers", True)
            )
        ):
            return
        rd: cutlass.Constexpr[int] = self._a2a_ring_depth
        # ``next_slot`` is a vestigial per-CTA counter (init'd to 0, never read on the putwarp path).
        # (d) MULTISLOT repurposes it as a length-3 walk state (prev_peer + fill[2]); size the view
        # to match _extra_smem_struct. The removed dtma/claim/cache drains had produced/mma_done -> None.
        ns_n: cutlass.Constexpr[int] = self._a2a_walk_state_len()
        next_slot_view = storage.decoupled.next_slot.get_tensor((ns_n,))
        produced_view = None
        mma_done_view = None
        self._init_decoupled_mbarriers(
            storage.decoupled.full.data_ptr(),
            storage.decoupled.empty.data_ptr(),
            storage.decoupled.pcount.get_tensor((1,)),  # tensor view (Pointer has no item-assign)
            next_slot_view,
            warp_idx,
            produced_view,
            mma_done_view,
        )

    @cute.jit
    def _init_decoupled_mbarriers(
        self, full_ptr, empty_ptr, pcount, next_slot, warp_idx, produced=None, mma_done=None
    ) -> None:
        rd: cutlass.Constexpr[int] = self._a2a_ring_depth
        # full[s]: the SINGLE producer warp (32 threads) arrives -> count 32.
        # empty[s]: who arrives depends on the drain mechanism:
        #   * SIMT drain (I2a): ALL consumer-warpgroup threads do the STG (warpgroup-parallel) ->
        #     count = consumer_warps * 32.
        #   * TMA drain (I2b Option B): ONE consumer warp issues the async TMA-S2G (the engine does
        #     the BW; a warpgroup-parallel TMA issue is pointless) -> count 32.
        # The producer waits (does not arrive) on empty; consumers wait (do not arrive) on full.
        # TMA-based consumers (Option B SMEM-ring drain, A′ double-TMA) issue from ONE warp -> empty
        # count 32. The SIMT warpgroup-parallel drain uses all consumer warps -> count = warps*32.
        # TRACK-1 gmem_cache: the drain uses ONE consumer warp (32 lanes cooperate on the repack +
        # warp-collective TMA-S2G) -> empty[slot] arrive count = 32 (like the TMA-drain consumers).
        # HALO-CACHE: the SIMT-masked residual drain uses ONE consumer warp (32 lanes) per straddle
        # slot (start simple; can parallelize in the ablation) -> empty arrive count 32.
        # putwarp drain: the whole consumer warpgroup arrives empty[s] (warpgroup-parallel).
        # CLUSTER-DRAIN (3.1b) reuses full[s]/empty[s] as the cross-CTA mbars (no struct change): full =
        # staging-ready, count == cluster_n (one arrive per cluster CTA's staging warp lane 0); empty =
        # the per-CTA reuse gate. SINGLE-BUFFER / db-no-mw: empty count == 1 (rank-0's ONE drain warp
        # arrives per run). SPIKE db: the range_constexpr(rd) loop below already inits BOTH slots -> the
        # rotating producer/drain use full[slot]/empty[slot]. SPIKE mw: rank-0's n_consumer_warps drain
        # warps EACH arrive empty[slot] -> empty count == n_consumer_warps (the mbar doubles as the
        # "all-drain-warps-done" barrier; no named-barrier id). full count stays cluster_n (mw is a DRAIN
        # change; the producers still arrive full once per CTA).
        _cluster: cutlass.Constexpr[bool] = bool(getattr(self, "_a2a_cluster_drain", False))
        _multislot: cutlass.Constexpr[bool] = bool(getattr(self, "_a2a_cluster_multislot", False))
        full_cnt: cutlass.Constexpr[int] = int(self._a2a_cluster_n) if _cluster else 32
        empty_cnt: cutlass.Constexpr[int] = 1 if _cluster else self._DECOUPLED_CONSUMER_WARPS * 32
        lane = cute.arch.lane_idx()
        if warp_idx == Int32(0) and lane == Int32(0):
            for s in cutlass.range_constexpr(rd):
                cute.arch.mbarrier_init(full_ptr + s, full_cnt)
                cute.arch.mbarrier_init(empty_ptr + s, empty_cnt)
            pcount[0] = Int32(0)
            # CLAIM DRAIN: zero the per-CTA work-claim counter on the SAME elected thread (warp_idx==0,
            # lane==0), pre-warp-split -> ordered before any claim by the existing pipeline_init_wait.
            # (d) MULTISLOT: next_slot is the length-3 walk state [prev_peer, fill0, fill1]. prev_peer
            # starts -1 (NO real peer staged yet -> the first peer's cross does NOT arrive a stale slot);
            # the fill counts start 0 (first fill of each slot skips the reuse-wait). 1-D FIX (cp1==1): the
            # length-5 state adds [3]=prev_m0, [4]=prev_L, both -1 (an impossible tile/plane coord) so the
            # FIRST tile's band/plane-change detector fires (tile_coord != -1) -> its meta is written.
            _cp1_init: cutlass.Constexpr[int] = int(getattr(self, "_a2a_cp1", 1))
            if const_expr(next_slot is not None):
                next_slot[0] = Int32(-1) if const_expr(_multislot) else Int32(0)
                if const_expr(_multislot):
                    next_slot[1] = Int32(0)
                    next_slot[2] = Int32(0)
                    if const_expr(_cp1_init == 1):
                        next_slot[3] = Int32(-1)
                        next_slot[4] = Int32(-1)
            # §7.16o REDESIGN: zero the dtma tail's produced-group counter on the same elected thread.
            if const_expr(produced is not None):
                produced[0] = Int32(0)
            # §7.16p clean-switch: zero the per-CTA MMA-done flag (cwg polls it; MMA warps set it).
            if const_expr(mma_done is not None):
                mma_done[0] = Int32(0)
        cute.arch.mbarrier_init_fence()

    def _build_p2p_table(self, pe_table):
        """#57: the compile-time P2P/NVLink connectivity table. Delegates to the module-level
        :func:`build_p2p_table` so callers OUTSIDE the kernel (e.g. the distributed autotuner venue gate)
        probe the SAME topology source; see :func:`build_p2p_table` for the full contract."""
        return build_p2p_table(pe_table)

    def _guard_coupled_nvlink_only(self, pe_table, *, entry):
        """§5.2 — probe the cp topology and REJECT a COUPLED back store on a cross-node mesh.

        THE DEFECT THIS CLOSES. The back A2A had the only ``nvshmem_ptr``-derived peer-pointer sites in
        the tree with **no** ``is_p2p`` / NULL guard (the front store has one at
        ``dual_gated_gemm_staged_a2a.py:1298-1310``; a host copy-engine A2A alternative also existed and
        was removed as an unadopted implementation).
        ``_a2a_is_p2p`` was computed only under ``if self._a2a_decoupled``, so on a coupled store the
        probe never ran and a cross-node caller of these PUBLIC entries got a silent illegal memory
        access instead of an error. Measured at cp=16 in four isolated fresh processes; the identical
        store at the identical shapes on cp=8 / all-P2P passes 8/8, so it is purely topological.

        THIS IS A FIX, NOT A TRIGGER-GATE DODGE. A TMA-S2G (and the raw-memref STG the uniform coupled
        tiling falls into) requires a **P2P-mapped virtual address**, and ``nvshmem_ptr`` returns NULL
        for an IB peer — a cross-node peer has no such address, by hardware. The coupled store is
        therefore NVLink-only as a capability, and the defect was the MISSING guard turning a real
        boundary into a silent fault. The supported cross-node path exists and production already routes
        to it (``fused_trimul.py`` auto-detects ``hybrid_ib`` from this same probe and raises the
        analogous error on an explicit ``hybrid_ib=False``); ``pe_aligned`` reaches the fault through its
        own base_m / aligned-predicate / TMA-descriptor chain, so BOTH coupled variants are covered here.

        DO **NOT** "fix" this by porting the front's ``atom_pe_table`` substitution (that store swaps an
        IB peer's PE for ``my_pe`` so the coupled copy targets a local address). It is safe there ONLY
        because the front has an alternative transport for those peers — the ``const_expr is_p2p[r]``
        differential drain — so the substituted atom is never used. The back coupled store has no such
        arm: ``cute.copy(atoms[r], ...)`` is its ONLY mechanism, so substituting ``my_pe`` would make the
        store SUCCEED into my own recv at the peer's offset — silent wrong output instead of a loud
        fault. Strictly worse.

        The probe is buffer-free and host-side, so calling it unconditionally costs no allocation and no
        collective. ``_a2a_has_ib_peers`` deliberately keeps its decoupled-only derivation at the call
        site: it feeds ~9 ``const_expr`` gates and, while most are conjoined with a decoupled flag,
        ``_extra_epilogue_tail_role`` conjoins it with ``_a2a_cluster_multislot`` instead — and
        ``cluster_drain``/``cluster_multislot`` are validated independently of ``decoupled`` — so
        widening it could move generated code on a coupled+multislot config. ``_a2a_is_p2p`` has no other
        reader here, so setting it changes no ``const_expr``.

        An all-False table means the probe could not run (nvshmem not initialised): my OWN pe is always a
        member of ``TEAM_SHARED``, so a live job always has at least one True. No positive determination
        -> no raise, rather than a spurious rejection of an all-NVLink job.
        """
        self._a2a_is_p2p = self._build_p2p_table(pe_table)
        is_p2p = self._a2a_is_p2p
        if self._a2a_decoupled or all(is_p2p) or not any(is_p2p):
            return
        n_ib = sum(1 for v in is_p2p if not v)
        raise ValueError(
            f"{entry}: this job has {n_ib} cross-node IB peer(s) (not all cp peers are P2P/NVLink-"
            f"reachable: is_p2p={tuple(bool(v) for v in is_p2p)}, pe_table={tuple(int(p) for p in pe_table)}), "
            "but the requested store is COUPLED (decoupled=False) — an in-epilogue TMA-S2G / STG straight "
            "into the peer's symmetric heap. nvshmem_ptr returns NULL for an IB peer, so that store takes "
            "a CUDA illegal memory access (it surfaces later, at the next barrier). The coupled store is "
            "NVLink-only BY HARDWARE (a TMA-S2G needs a P2P-mapped virtual address). Use the decoupled "
            "ring drain for a cross-node mesh — configure_a2a_gemm_native(decoupled=True, "
            "producer_tma=True, consumer_strided=True, consumer_strided_putwarp=True, ib_ring=True) or "
            "configure_a2a_sharded(ib_ring=True); via FusedTriMul, hybrid_ib=None auto-detects it."
        )

    def _num_extra_warpgroups(self) -> int:
        """Add ONE dedicated consumer warpgroup ONLY on the decoupled A2A path.

        Flag-gated: returns 0 unless ``self._a2a_decoupled`` (and A2A enabled), so the
        block dim is byte-identical to the parent everywhere else.  Uses ``getattr`` because
        the parent ``__init__`` calls this hook BEFORE the subclass sets the A2A flags
        (they are assigned after ``super().__init__()`` / in ``configure_a2a_*``); the launch
        recomputes ``threads_per_cta`` once the flags are set.

        #57: ALSO gated on ``_a2a_has_ib_peers`` (default True) -- an ALL-P2P job (no IB peer, cp<=8) has
        NO cross-node put to drain, so the consumer warpgroup is ELIDED (returns 0) -> threads_per_cta
        byte-identical to pe_aligned = the cp<=8 collapse (this cascades: _extra_wg_reg_adjust early-returns
        on n_extra_wg<=0 and the parent's consumer-role split is dead code at count 0)."""
        if const_expr(
            getattr(self, "_a2a_enabled", False)
            and getattr(self, "_a2a_decoupled", False)
            and getattr(self, "_a2a_has_ib_peers", True)
        ):
            # WS-B 3-WG wide-tile: the drain rides the PRODUCER warpgroup's spare warps 9-11 (no
            # dedicated 4th WG -> 384 threads -> reqntid cap 170 >= the m64n256 WGMMA's ~154) instead of
            # a 4th warpgroup (512 threads -> cap 128 < 154, the proven register wall). Return 0 extra WGs.
            if self._use_3wg_drain():
                return 0
            return int(getattr(self, "_a2a_consumer_warpgroups", 1))
        return 0

    def _use_3wg_drain(self) -> bool:
        """True when the shipped 4-WG (512-thread) decoupled-A2A layout's uniform register cap
        (65536/512 = 128) is below the MMA accumulator's per-thread need -> the wide tile (tile_n=256,
        m64n256 WGMMA ~154 regs) cannot fit 4-WG at 1 block/SM (PROVEN: 4-WG busts C7602 128, and no
        nvvm.maxnreg escapes the 512-thread launch reservation). Drop the dedicated 4th warpgroup and run
        the drain on the producer WG's spare warps 9-11 -> 384 threads -> reqntid cap 170 >= 154 (fits,
        NO register tricks). Host-computable (config only). False -> the shipped 4-WG layout, byte-
        identical for tile_n<=128 (accumulator ~64 << 128) and every non-decoupled path."""
        if not (
            getattr(self, "_a2a_enabled", False)
            and getattr(self, "_a2a_decoupled", False)
            and getattr(self, "_a2a_has_ib_peers", True)
        ):
            return False
        regs_acc = math.prod(self.cta_tile_shape_mnk[:2]) // (
            math.prod(self.atom_layout_mnk) * self.num_threads_per_warp_group
        )
        threads_4wg = (
            self.mma_warp_groups + 2
        ) * self.num_threads_per_warp_group  # MMA + load + 1 extra
        uniform_4wg = (65536 // threads_4wg) // 8 * 8
        return regs_acc >= uniform_4wg

    def _drain_on_producer_wg(self) -> bool:
        return self._use_3wg_drain()

    def _extra_link_bitcode(self):
        """``compile_gemm_with_bitcode`` hook: extra device ``.bc`` to link ALONGSIDE the stock nvshmem
        bitcode. The A2A drain links NOTHING extra -> byte-identical single-bitcode link."""
        return []

    def _drain_first_warp(self) -> int:
        """First warp index of the drain (consumer) role. 4-WG: the dedicated extra warpgroup's first
        warp (mma_warp_groups+1)*4 (== 12). 3-WG (_use_3wg_drain): the producer WG's first SPARE warp
        ab_load_warp_id + num_ab_load_warps (== mma_warp_groups*4 + 1 == 9), after the sole TMA-load warp
        8. The drain loops anchor their warp-gate + wid math on this."""
        if self._use_3wg_drain():
            return self.mma_warp_groups * 4 + int(self.num_ab_load_warps)
        return (self.mma_warp_groups + 1) * 4

    def _persistent_grid_adjust(self, grid):
        """Persistent-grid z-extent hook: identity (returns ``grid`` unchanged) -> byte-identical to the
        parent default. Retained as the parent ``__call__``'s grid hook (no A2A grid reduction)."""
        return grid

    def _extra_wg_reg_adjust(self, n_extra_wg: int) -> None:
        """UNIFORM-allocate the decoupled-drain extra-WG layout (default non-A2A path untouched).

        Eliminates an entire deadlock class: the Hopper warp-specialized ``setmaxnreg`` realloc
        (load/consumer warps ``.dec`` to ``num_regs_load``, MMA warps ``.inc`` to ``num_regs_mma``)
        is FRAGILE in the extra-WG layout. The MMA ``.inc`` (PTX ISA) *blocks until the CTA register
        pool has enough free registers*, and the pool is fed only by the ``.dec``s; whenever ptxas
        keeps an asymmetric realloc whose ``.inc`` target is unreachable from the launch REGCOUNT
        baseline, the ``.inc`` spins forever (§7.16m). ELF-confirmed in the low-register-pressure
        configs (idle-cwg "config-5" REGCOUNT=96 + EIATTR_REG_RECONFIG; TMA-engine tail same
        signature) — but the trigger is ptxas-dependent, so rather than enumerate which configs are
        safe, we DROP the realloc for *every* decoupled-drain config and let ptxas use one uniform
        allocation sized to actual usage. No pool dance -> no unreachable ``.inc`` -> no deadlock.

        Cost (the load-bearing assumption, verified in §7.16n): the only effect of dropping the
        realloc is that the MMA warps get the uniform per-thread register count instead of the
        realloc's high ``num_regs_mma``. This costs nothing IFF the MMA is not register-bound (the
        uniform count does not spill / does not lower occupancy enough to matter). The SIMT
        GMEM->SymMEM (putwarp) drain itself is UNAFFECTED by the MMA reg count. For the headline
        SIMT-drain configs ptxas already auto-DROPPED the realloc (REGCOUNT=90, no reconfig), so the
        explicit skip is the SAME uniform allocation it was already emitting — i.e. byte-identical to
        the prior behavior on those configs, and a no-op deadlock-out everywhere else.

        We do NOT try to make the split "uniform" via equal ``num_regs_*`` — a ``.dec``/``.inc`` to a
        target not on the right side of the entry baseline is ISA-UB; skipping the pair is the clean
        no-op. ``self._skip_warpgroup_reg_realloc`` -> the parent gates out BOTH the load/consumer
        ``setmaxregister_decrease`` and the MMA ``setmaxregister_increase`` together. The non-A2A base
        path never sets the attr (the guard below returns first) -> byte-identical PTX."""
        if const_expr(n_extra_wg <= 0):
            return
        if const_expr(
            not (getattr(self, "_a2a_enabled", False) and getattr(self, "_a2a_decoupled", False))
        ):
            return
        # UNIFORM allocation for ALL decoupled-drain configs: drop the fragile warp-specialized
        # setmaxnreg realloc (deadlock-out for the whole class). The headline SIMT-putwarp drains
        # already have ptxas auto-dropping it, so this is byte-identical for them; it additionally
        # covers the low-register-pressure idle-cwg + TMA-tail configs that would otherwise wedge.
        self._skip_warpgroup_reg_realloc = True

    def consumer_warpgroup_role(self, warp_idx, storage, epilogue_params, tile_sched_params):
        """The dedicated consumer warpgroup's per-CTA work on the decoupled A2A path.

        STEP 1 (de-risk the warp layout): does NOTHING yet — the extra warpgroup is
        present (block dim +128, its own role branch) but immediately exits its role, so
        we can confirm the structural change COMPILES via the bitcode route and produces
        byte-identical / correct output with the idle extra warpgroup present, BEFORE any
        ring/handoff is wired (STEP 2). Only one warp of the warpgroup need act; the rest
        fall through. No-op body."""
        if const_expr(not (self._a2a_enabled and self._a2a_decoupled)):
            return
        if const_expr(not self._a2a_decoupled_store):
            # STEP 1: warpgroup present but idle (the store stays coupled). No work — proves the
            # warp layout is viable before the ring is wired.
            return
        # STEP 2: drain the local-GMEM ring -> peer SymMEM. This (plain) method does the HOST-TRACE
        # extraction (tensor algebra, pointer/scalar prep — no runtime control flow), then hands the
        # FLATTEN-ABLE DSL values to the @cute.jit drain loops (the storage struct itself
        # cannot cross a @cute.jit boundary, so we never pass it). Mirrors how the parent extracts
        # sA/sB/pipelines in @cute.kernel and passes them to @cute.jit mma()/epilogue().
        cp: cutlass.Constexpr[int] = self._a2a_cp
        epi_tile = self.epi_tile
        epi_m: cutlass.Constexpr[int] = epi_tile[0]
        epi_n: cutlass.Constexpr[int] = epi_tile[1]
        tile_m: cutlass.Constexpr[int] = self.cta_tile_shape_mnk[0]
        tile_n: cutlass.Constexpr[int] = self.cta_tile_shape_mnk[1]
        epi_tile_num: cutlass.Constexpr[int] = (tile_m // epi_m) * (tile_n // epi_n)

        # per-CTA total subtiles (STATIC persistent scheduler): CTA z does work z, z+gz, ... < total.
        total_clusters = Int32(cute.size(tile_sched_params.problem_shape_ncluster_mnl))
        rd: cutlass.Constexpr[int] = self._a2a_ring_depth
        nfields: cutlass.Constexpr[int] = self._DECOUPLED_META_FIELDS
        full_ptr = storage.decoupled.full.data_ptr()
        empty_ptr = storage.decoupled.empty.data_ptr()
        meta = storage.decoupled.meta.get_tensor((rd * nfields,))

        # CLUSTER-DRAIN (3.1b): cluster-rank-0 drains the per-cluster staging (epi_m, N_j_loc) with the
        # coalesced put_warp. Placed BEFORE the ring dispatch (it reads cluster_stage, not decoupled_ring
        # -- which may be None on this variant). Default off -> const_expr-elided -> byte-identical.
        if const_expr(getattr(self, "_a2a_cluster_drain", False)):
            stage = epilogue_params.cluster_stage
            recv_local = epilogue_params.recv_local
            if const_expr(stage.element_type is cutlass.BFloat16):
                stage_put = cute.recast_tensor(stage, cutlass.Int16)
                recv_put = cute.recast_tensor(recv_local, cutlass.Int16)
            else:
                stage_put = stage
                recv_put = recv_local
            self._cluster_drain_loop(
                warp_idx,
                stage_put,
                recv_put,
                full_ptr,
                empty_ptr,
                meta,
                epilogue_params.pe_table_dev,
                total_clusters,
            )
            return

        if const_expr(self._a2a_consumer_strided):
            # ---- §7.16 STRIDED + WIDEN drain ----
            # The producer staged a TILE-WIDE ring slot (epi_m, tile_n) per GROUP (the n_sub_per_tile
            # j-subtiles laid contiguous along tile_n -> a (epi_m, tile_n) contiguous-per-row box). The
            # consumer drains a group with a tile_n-WIDE (256 B at tile_n=128 bf16) put per row (past the
            # ~256-B coalescing knee §7.15) -> NVLink BW vs the 33-43 GB/s NARROW 64-B lockstep box. Two
            # drain mechanisms, toggled by _a2a_consumer_strided_putwarp:
            #   * default: WIDE peer-pinned universal STG (the existing kernel mechanism, tiled-copy over
            #     the (epi_m, tile_n) box) -> ROBUST (no FFI prototype), the wired-in default.
            #   * putwarp: warp-strided per-row NVSHMEM put_nbi_warp (the de-risked proto mechanism) ->
            #     blocked by an FFI-prototype-align interaction w/ the dynamic-shape GEMM compile (see
            #     _decoupled_drain_loop_strided_putwarp). Kept for follow-up; OFF by default.
            ring = epilogue_params.decoupled_ring  # (grid_CTAs, rd_tiles, epi_m, tile_n)
            bidx = cute.arch.block_idx()[0] + cute.arch.block_idx()[2]
            ring_cta = ring[bidx, None, None, None]  # (rd_tiles, epi_m, tile_n)
            if const_expr(getattr(self, "_a2a_consumer_strided_putwarp", False)):
                recv_local = (
                    epilogue_params.recv_local
                )  # THIS rank's local recv (cp,Dloc,B,N_loc,N)
                if const_expr(ring.element_type is cutlass.BFloat16):
                    ring_cta_put = cute.recast_tensor(ring_cta, cutlass.Int16)
                    recv_put = cute.recast_tensor(recv_local, cutlass.Int16)
                else:
                    ring_cta_put = ring_cta
                    recv_put = recv_local
                recv_perm = cute.make_tensor(
                    recv_put.iterator, cute.select(recv_put.layout, mode=[3, 4, 0, 1, 2])
                )
                # arbitrary_n: divide the i-axis by 1 so EVERY row is addressable (the per-row drain
                # indexes by i_local directly); a straddling i_local would overflow the
                # floor(N_loc/epi_m) subtile axis of the (epi_m, tile_n) box.
                if const_expr(getattr(self, "_a2a_arbitrary_n", False)):
                    g_wide = cute.flat_divide(recv_perm, (1, tile_n))
                else:
                    g_wide = cute.flat_divide(recv_perm, (epi_m, tile_n))
                recv_n_j_val = recv_perm.shape[1]
                # recv_n_j == the recv's token-j extent N (RUNTIME under dynamic); the drain uses it as the
                # partial-N col clamp on the dynamic ib_ring path. Unused (const_expr-elided) on the static /
                # default putwarp path -> byte-identical.
                self._decoupled_drain_loop_strided_putwarp(
                    warp_idx,
                    ring_cta_put,
                    g_wide,
                    full_ptr,
                    empty_ptr,
                    meta,
                    epilogue_params.pe_table_dev,
                    total_clusters,
                    Int32(epi_tile_num),
                    recv_n_j=recv_n_j_val,
                )
                return

    @cute.jit
    def tail_drain_role(self, warp_idx, storage, epilogue_params, tile_sched_params):
        """MMA-warp tail hook. Default no-op (the putwarp drain lives in the consumer warpgroup).

        #57 Stage-2b (Approach C): when handshake_skip, the producer (MMA warp 0) arrives ONE DONE
        sentinel here so the drain's remote-only loop terminates. g_ib = next_slot = n_remote for this
        CTA (copy_fn bumped it per remote band). Rides the IDENTICAL arrival discipline as a real band:
        reuse-wait empty[slot] (phase (k-1)&1, only if the slot wrapped) -> write sentinel meta
        (gi_base field = -1) -> full-arrive[slot] (32 lanes = full_cnt). Zero-remote CTA -> g_ib=0 ->
        still arrives (drain does 0 drains + terminates on it). ONE producer warp (warp 0)."""
        # (d) MULTISLOT ALSO uses this MMA-warp tail (to flush the walk's FINAL real peer's full-arrive
        # -- it has no next tile to cross into). Gate on handshake_skip OR multislot (either needs the tail).
        _ms: cutlass.Constexpr[bool] = bool(getattr(self, "_a2a_cluster_multislot", False))
        if const_expr(
            not (
                getattr(self, "_a2a_enabled", False)
                and _ms
                and getattr(self, "_a2a_has_ib_peers", True)
            )
        ):
            return  # #57 orthogonality: no sentinel at the all-P2P collapse (the ring struct is elided)
        rd: cutlass.Constexpr[int] = self._a2a_ring_depth
        nfields: cutlass.Constexpr[int] = self._DECOUPLED_META_FIELDS
        full_ptr = storage.decoupled.full.data_ptr()
        empty_ptr = storage.decoupled.empty.data_ptr()
        meta = storage.decoupled.meta.get_tensor((rd * nfields,))
        ns_n: cutlass.Constexpr[int] = self._a2a_walk_state_len()
        next_slot = storage.decoupled.next_slot.get_tensor((ns_n,))
        lane = cute.arch.lane_idx()
        if const_expr(_ms):
            # (d) MULTISLOT final-peer flush. walk[0] == the LAST REAL peer_j this CTA staged (-1 if it
            # ended on a padding tile -> already arrived full[prev&1] at that cross). ONE lane arrives rank-
            # 0's full[prev&1] cross-CTA (count == cluster_n; this CTA's contribution for the final peer).
            if warp_idx == Int32(0):
                prev_peer = next_slot[Int32(0)]
                cp1: cutlass.Constexpr[int] = int(getattr(self, "_a2a_cp1", 1))
                prev_real = (prev_peer >= Int32(0)) & (prev_peer < Int32(cp1))
                if prev_real:
                    cute.arch.fence_acq_rel_gpu()
                    cute.arch.sync_warp()
                    if lane == Int32(0):
                        for ss in cutlass.range_constexpr(rd):
                            if (prev_peer % Int32(2)) == Int32(ss):
                                cute.arch.mbarrier_arrive(full_ptr + ss, peer_cta_rank_in_cluster=0)

    @cute.jit
    def _decoupled_drain_loop_strided_putwarp(
        self,
        warp_idx,
        ring_cta,
        g_wide,
        full_ptr,
        empty_ptr,
        meta,
        pe_table_dev,
        total_clusters,
        epi_tile_num,
        recv_n_j=None,
    ):
        """§7.16 put_nbi_warp variant (the de-risked 320-GB/s proto mechanism; FOLLOW-UP, default OFF).

        Same tile-grouped ring as the default, but drains via per-row NVSHMEM ``put_nbi_warp`` over the
        LOCAL recv view (peer by RUNTIME PE from ``pe_table_dev``) — the proto_drain_bw / a2a.py
        mechanism measured at 320 GB/s in isolation. The put is BRANCH-FREE (runtime ``slot`` indexes the
        ring; the dst is always the local recv view, peer routed by ``pe = pe_table_dev[peer]``; NO
        const_expr (slot,peer) unroll around the put) -> ONE call site.

        ALIGN BLOCKER RESOLVED (§7.16c), but does NOT land the <1 win. The earlier "External prototype
        mismatch" was an EXACT FFI type-equality check (`cute/ffi.py:_type_check`): the call was
        ``ptr<i16,align<16>>`` (ring+recv via ``from_dlpack(assumed_align=16)``) while the STOCK
        ``rma.put_nbi_warp`` int16 prototype is ``ptr<i16>`` = dtype-default ``align<2>`` (rendered
        no-align) — the doc had the two sides inverted. FIX (kept here, the wide-aligned plan): a REBUILT
        align<16> FFI for the same extern symbol (module-level ``_int16_put_nbi_warp_a16`` +
        ``_put_nbi_warp_int16_a16``) so call and prototype BOTH carry ``align<16>`` -> they match. SOUND:
        every row addr IS 16-B aligned (per-row stride ``tile_n*2 = 256 B``; heaps 256-B aligned). Compiles
        + bit-identical to the coupled recv. BUT measured ~3.2-3.4x str/sym (WORSE than the WIDE-STG drain's
        ~2.0x; align<16> == align<2> within noise -> the pointer-align hint is NOT a BW lever, the extern's
        device vectorization is fixed in the precompiled bitcode). ROOT CAUSE: the peer recv box
        ``(epi_m, tile_n)`` is a sub-tile of design-E's ``(cp,Dloc,B,N_loc,N)`` recv -> consecutive rows are
        N-STRIDED, so each ``put_nbi_warp`` is one ``(1,tile_n)=256 B`` row = 8 B/lane (vs the STG drain's
        STG.128 = 16 B/lane over 128 threads), AND only the consumer warpgroup drains behind a per-group
        handshake (shallow in-flight) — the proto's 320 GB/s assumed a CONTIGUOUS dst + grid-wide
        no-handshake blasting, neither of which holds in the fused epilogue. The <1 win needs a recv-layout
        change (contiguous per-tile peer box) or the TMA-engine drain, NOT this SIMT swap. Reachable via
        ``self._a2a_consumer_strided_putwarp=True``; the WIDE-STG drain (§7.16a) remains the default."""
        cp: cutlass.Constexpr[int] = self._a2a_cp
        my_cp_rank: cutlass.Constexpr[int] = self._a2a_my_cp_rank
        # 2-D (Task #13): cp0/cp1 split the cp peers into (i,j) axes. The per-row route re-forms the flat
        # peer = peer_i*cp1 + peer_j. 1-D (cp1==1): cp0==cp, peer_j==0 -> every 2-D term folds ->
        # byte-identical.
        cp1: cutlass.Constexpr[int] = int(getattr(self, "_a2a_cp1", 1))
        cp0: cutlass.Constexpr[int] = int(getattr(self, "_a2a_cp0", cp))
        rd: cutlass.Constexpr[int] = self._a2a_ring_depth
        nfields: cutlass.Constexpr[int] = self._DECOUPLED_META_FIELDS
        epi_m: cutlass.Constexpr[int] = self.epi_tile[0]
        epi_n: cutlass.Constexpr[int] = self.epi_tile[1]
        tile_n_c: cutlass.Constexpr[int] = self.cta_tile_shape_mnk[1]
        n_sub_per_tile: cutlass.Constexpr[int] = tile_n_c // epi_n
        n_consumer_warps: cutlass.Constexpr[int] = self._DECOUPLED_CONSUMER_WARPS
        # arbitrary_n: route rows PER-ROW (peer = gi//N_loc, i_local = gi%N_loc) instead of the single
        # per-tile (peer, i_tile). Mirrors the putwarp drain's per-row branch.
        arbitrary_n: cutlass.Constexpr[bool] = bool(getattr(self, "_a2a_arbitrary_n", False))
        N_loc: cutlass.Constexpr[int] = self._a2a_N_loc
        # IB-RING (§0.9): pe_aligned per-peer M-tiling routed through this GMEM-ring put drain. Two
        # deltas vs the default (design-E global M-tiling) drain: (1) the per-peer ceil tiling ALWAYS
        # spills the last per-peer tile past the peer boundary on the straddle path, so the LAST peer's
        # last tile spills past M=cp*N_loc EVEN when M % tile_m == 0 -> the P_i guard must be live
        # whenever arbitrary_n (NOT only when M % tile_m != 0). (2) dynamic-shape (one-compile-many-N)
        # reads N_loc / M / N off the recv at RUNTIME, not the baked compile anchor.
        pe_aligned_ib: cutlass.Constexpr[bool] = bool(getattr(self, "_pe_aligned_ib_ring", False))
        dyn_ib: cutlass.Constexpr[bool] = pe_aligned_ib and bool(self._a2a_dynamic)
        # IN-KERNEL cross-node completion: use the BLOCKING put_warp (non-nbi) instead of put_nbi_warp so
        # the IBGDA device path ibgda_quiet-s after each post -> the CQ is reaped + the QP backpressures
        # in-kernel (the cp16 QP-exhaustion fix), WITHOUT any device-quiet API (unreachable from
        # @cute.kernel). Default ON for the putwarp drain (this is the only store reaching here); flip to
        # the nbi put via _a2a_ib_quiet=False (intra-NVLink A/B: nbi is faster where there is no QP). No-op
        # if the blocking FFI failed to build. Perf-serializes per put -> a perf follow-up; CORRECT-first.
        ib_quiet: cutlass.Constexpr[bool] = bool(
            getattr(self, "_a2a_ib_quiet", True) and _int16_put_warp_a16 is not None
        )
        # P_i / P_j extents (partial M/N tiles). has_partial_* are
        # COMPILE-TIME -> aligned shapes elide all predication (byte-identical PTX, perf-neutral).
        tile_m: cutlass.Constexpr[int] = self.cta_tile_shape_mnk[0]
        # M (token-i extent) = N_i_loc * cp0 (Task #13: cp0, NOT cp -- the i-axis is split over cp0 only;
        # 1-D cp0==cp -> identical). N_full is the i-axis peer-block count's product, used for the P_i drop.
        M_full: cutlass.Constexpr[int] = N_loc * cp0
        N_full: cutlass.Constexpr[int] = int(getattr(self, "_a2a_N", 0))
        has_partial_m: cutlass.Constexpr[bool] = (
            arbitrary_n if pe_aligned_ib else (arbitrary_n and (M_full % tile_m) != 0)
        )
        has_partial_n: cutlass.Constexpr[bool] = (
            arbitrary_n and N_full > 0 and (N_full % tile_n_c) != 0
        )
        # Dynamic ib_ring: the col-clamp is ALWAYS applied at runtime (an aligned runtime N leaves
        # ncol == tile_n -> a no-op full-width put). use_ncol drives both the ncol compute + the put.
        use_ncol: cutlass.Constexpr[bool] = has_partial_n or dyn_ib
        # Runtime recv extents for the dynamic ib_ring path (one compile serves many token counts):
        # N_loc from g_wide's i-axis (mode 2 after flat_divide(recv_perm,(1,tile_n))), M = N_loc*cp,
        # and the token-j extent threaded in (recv_n_j = recv_perm.shape[1]). const_expr(dyn_ib)-gated
        # so nothing runtime is traced on the static / default path.
        if const_expr(dyn_ib):
            N_loc_rt = Int32(g_wide.shape[2])
            M_full_rt = N_loc_rt * Int32(cp0)  # Task #13: cp0 (i-axis peer count), NOT cp
            N_j_rt = Int32(recv_n_j)
        first_extra_warp: cutlass.Constexpr[int] = self._drain_first_warp()
        if warp_idx >= Int32(first_extra_warp):
            wid = warp_idx - Int32(first_extra_warp)  # consumer-warp index [0, n_consumer_warps)
            gz = Int32(cute.arch.grid_dim()[2])
            z = Int32(cute.arch.block_idx()[2])
            my_tiles = Int32(0)
            if z < total_clusters:
                my_tiles = (total_clusters - z + gz - Int32(1)) // gz  # ceil_div(total - z, gz)
            total_sub = my_tiles * epi_tile_num
            n_groups = total_sub // Int32(n_sub_per_tile)  # one tile-wide slot per group
            g = Int32(0)
            while g < n_groups:
                slot = g % Int32(rd)
                k = g // Int32(rd)
                # Bounded rotating ring: static-offset range_constexpr(rd) slot/phase match (rd small).
                for ss in cutlass.range_constexpr(rd):
                    if slot == Int32(ss):
                        cute.arch.mbarrier_wait(full_ptr + ss, k & Int32(1))
                base = slot * Int32(nfields)
                peer = meta[base + Int32(0)]
                i_tile = meta[base + Int32(1)]
                j_base = meta[base + Int32(2)]
                d = meta[base + Int32(3)]
                b = meta[base + Int32(4)]
                j_grp = j_base // Int32(n_sub_per_tile)  # j_base (epi_n units) -> tile_n-tile index
                # PE for the per-tile route; SKIPPED on the arbitrary_n path (meta peer may be >= cp on a
                # straddle/partial-M tile -> OOB device read). The arbitrary path uses per-row pe_row.
                if const_expr(not arbitrary_n):
                    pe = pe_table_dev[peer]  # runtime flat-cp -> global PE (branch-free)
                # RUNTIME slot index into the ring + the (always-local) dst recv box -> ONE put call site.
                src_box = ring_cta[slot, None, None]  # (epi_m, tile_n) ring slot (runtime slot)
                src_t = cute.local_tile(src_box, (1, tile_n_c), (None, None))  # (1,tn,em,1)
                # P_j: cols present at this tile_n-group = min(tile_n, N - j_grp*tile_n) (see claim drain).
                # static: const-gated on has_partial_n (ALIGNED N elides it -> byte-identical full-width
                # put). dynamic ib_ring: always clamp with the RUNTIME token-j extent (aligned N -> ncol
                # == tile_n -> full-width, no-op).
                if const_expr(use_ncol):
                    n_ext = N_j_rt if const_expr(dyn_ib) else Int32(N_full)
                    rem = n_ext - j_grp * Int32(tile_n_c)
                    ncol = rem if rem < Int32(tile_n_c) else Int32(tile_n_c)
                if const_expr(not arbitrary_n):
                    dst_box = g_wide[
                        (None, None, i_tile, j_grp, Int32(my_cp_rank), d, b)
                    ]  # (epi_m,tn)
                    dst_t = cute.local_tile(dst_box, (1, tile_n_c), (None, None))
                else:
                    gi_base = meta[
                        base + Int32(5)
                    ]  # epi-box GLOBAL token-i base (arbitrary_n only)
                row = wid  # warp wid owns rows {wid,wid+nw,..} < epi_m
                while row < Int32(epi_m):
                    sr0 = src_t[(None, None, row, Int32(0))]  # (1, tile_n) stride-1
                    do_put = (
                        True  # Python const default -> elided unless has_partial_m makes it runtime
                    )
                    if const_expr(not arbitrary_n):
                        dr0 = dst_t[(None, None, row, Int32(0))]
                        pe_row = pe
                    else:
                        # PER-ROW route: gi -> peer/i_local (a straddling box fans rows out to >=2 peers).
                        # g_wide's i-axis was divided by 1 -> index by i_local directly (see the claim
                        # drain); take the 1-row box's only row.
                        gi = gi_base + row
                        # dynamic ib_ring reads N_loc off the recv at RUNTIME (one-compile-many-N);
                        # static / default use the baked const N_loc.
                        if const_expr(dyn_ib):
                            peer_row = gi // N_loc_rt
                            i_local = gi % N_loc_rt
                        else:
                            peer_row = gi // Int32(N_loc)
                            i_local = gi % Int32(N_loc)  # in [0,N_loc) -> g_wide i-index in-bounds
                        # P_i: a garbage row past M=cp*N_loc (the last peer's per-peer spill, or the last
                        # partial M-tile) makes peer_row >= cp -> pe_table_dev[peer_row] would be an OOB
                        # device read (a real crash). CLAMP with %cp (identity for valid rows; the put is
                        # skipped via do_put). const-gated on has_partial_m -> aligned M elides it. Under
                        # pe_aligned_ib has_partial_m is live on the whole straddle path (the per-peer
                        # spill), with M read at RUNTIME on the dynamic path.
                        if const_expr(has_partial_m):
                            if const_expr(dyn_ib):
                                do_put = gi < M_full_rt
                            else:
                                do_put = gi < Int32(M_full)
                            peer_row = peer_row % Int32(
                                cp0
                            )  # Task #13: %cp0 (i-axis peer count; ==cp @1-D)
                        # 2-D (Task #13): re-form the flat destination peer = peer_i*cp1 + peer_j. peer_i is
                        # the per-row i-peer (peer_row, post-%cp0); peer_j = meta peer % cp1 is CONSTANT
                        # across the tile (pe_aligned maps a tile to ONE j-block; the per-peer N_j_loc
                        # col-clamp + the double-store cover the j-spill). 1-D: cp1==1 -> peer_j==0 ->
                        # pe_table_dev[peer_row] (byte-identical PTX -- the mul/add are const_expr-elided).
                        if const_expr(cp1 > 1):
                            peer_j = peer % Int32(cp1)
                            pe_row = pe_table_dev[peer_row * Int32(cp1) + peer_j]
                        else:
                            pe_row = pe_table_dev[peer_row]
                        dst_box_r = g_wide[
                            (None, None, i_local, j_grp, Int32(my_cp_rank), d, b)
                        ]  # (1,tn)
                        dst_t_r = cute.local_tile(dst_box_r, (1, tile_n_c), (None, None))
                        dr0 = dst_t_r[(None, None, Int32(0), Int32(0))]
                    # §7.16b FIX — the WIDE-ALIGNED (align<16>) put_nbi_warp wire-in. The stock
                    # rma.put_nbi_warp's int16 FFI prototype is ptr<i16,gmem> with the dtype-DEFAULT
                    # align<2> (MLIR renders it no-align); the call here is align<16> (the ring+recv come
                    # in via from_dlpack(assumed_align=16)) -> "External prototype types mismatch". The doc
                    # mis-stated which side was which; the real wall is the EXACT type-equality the FFI
                    # verifier enforces. FIX: call our REBUILT align<16> FFI (_put_nbi_warp_int16_a16,
                    # same extern symbol, prototype declared align<16>) so the wide-aligned call MATCHES
                    # AND the align<16> hint propagates into the linked-bitcode put (wider GMEM->SYMMEM
                    # vector stores than align<2> would admit). SOUND: every ring/recv row addr is 16-B
                    # aligned (per-row stride tile_n*2 = 256 B; ring/recv heaps 256-B aligned). The runtime
                    # guard is a no-op safety tripwire: it ASSERTS 16-B alignment (always true here) so a
                    # future layout regression skips the put -> shows as a missing row in the gate
                    # histogram (loud) rather than a silent mis-store.
                    aligned = (sr0.iterator.toint() % cutlass.Int64(16) == cutlass.Int64(0)) and (
                        dr0.iterator.toint() % cutlass.Int64(16) == cutlass.Int64(0)
                    )
                    put_ok = aligned
                    if const_expr(has_partial_m):
                        put_ok = (
                            aligned & do_put
                        )  # bitwise (both runtime Booleans; no __bool__ at trace)
                    if put_ok:
                        sp = cute.make_ptr(
                            cutlass.Int16,
                            sr0.iterator.toint(),
                            cute.AddressSpace.gmem,
                            assumed_align=16,
                        )
                        dp = cute.make_ptr(
                            cutlass.Int16,
                            dr0.iterator.toint(),
                            cute.AddressSpace.gmem,
                            assumed_align=16,
                        )
                        if const_expr(use_ncol):
                            sr = cute.make_tensor(sp, cute.make_layout((1, ncol)))
                            dr = cute.make_tensor(dp, cute.make_layout((1, ncol)))
                        else:
                            sr = cute.make_tensor(sp, sr0.layout)
                            dr = cute.make_tensor(dp, dr0.layout)
                        # ib_quiet -> BLOCKING put_warp (ibgda_quiet-s in-kernel: reaps the CQ + bounds
                        # the QP -> the cp16 cross-node fix); else the nbi put (NVLink A/B, faster).
                        if const_expr(ib_quiet):
                            _put_warp_int16_a16(dr, sr, pe_row)
                        else:
                            _put_nbi_warp_int16_a16(dr, sr, pe_row)
                    row += Int32(n_consumer_warps)
                cute.arch.sync_warp()
                # Bounded rotating ring: static-offset range_constexpr(rd) empty-arrive.
                for ss in cutlass.range_constexpr(rd):
                    if slot == Int32(ss):
                        cute.arch.mbarrier_arrive(empty_ptr + ss)
                g += Int32(1)

    @cute.jit
    def _cluster_drain_loop(
        self,
        warp_idx,
        cluster_stage,
        recv_put,
        full_ptr,
        empty_ptr,
        meta,
        pe_table_dev,
        total_clusters,
    ):
        """CLUSTER-DRAIN (3.1b) DRAIN: cluster-rank-0 reads the FULL per-cluster staging (epi_m, N_j_loc)
        and fires ONE coalesced ``put_warp`` per same-peer_i i-run for the run's peer_j (from meta).

        Mirrors the 2-D branch of :meth:`_decoupled_coalesce_drain_loop` with three deltas: (a) ONE
        peer_j per run (no cp1 sub-band loop -- ``runs_per_band=cp1`` emits one run per peer_j, its
        index carried in meta[0]); (b) reads the CLUSTER-shared staging ``cluster_stage[bidx]`` (not a
        per-CTA rotating ring); (c) ONLY cluster-rank-0 (``block_idx_in_cluster()==0``) drains -- the
        other ranks' consumer warps fall through (their producers arrive rank-0's mbar cross-CTA, and
        rank-0 arrives their 'empty' after the drain). ``cluster_stage`` is int16
        ``(n_clusters, epi_m, N_j_loc)``; ``recv_put`` the LOCAL recv int16 ``(cp,Dloc,B,N_i_loc,N_j_loc)``.

        SIGNAL (Phase-3.1b, probe #2 / task #42 — WIRED, not a stub). The per-run 'full' wait
        (acquire, count cluster_n) is the real mbarrier_wait (:2111 db / :2115 sb); the 'empty' arrive
        (release, cross-CTA to every cluster CTA) is the real mbarrier_arrive (:2173-2178). Ran
        correctness-clean at K=256 (Gate-C); K=N (real O(N^3) einsum) validation IN PROGRESS (task #51:
        first K=N launch SIGSEGVs — the meta read :2118-2119 vs the producer's meta-write/full-arrive
        ordering is under investigation). Default-off -> byte-identical; no other path reaches here."""
        cp: cutlass.Constexpr[int] = self._a2a_cp
        my_cp_rank: cutlass.Constexpr[int] = self._a2a_my_cp_rank
        cp1: cutlass.Constexpr[int] = int(getattr(self, "_a2a_cp1", 1))
        cp0: cutlass.Constexpr[int] = int(getattr(self, "_a2a_cp0", cp))
        rd: cutlass.Constexpr[int] = self._a2a_ring_depth
        nfields: cutlass.Constexpr[int] = self._DECOUPLED_META_FIELDS
        epi_m: cutlass.Constexpr[int] = self.epi_tile[0]
        # #78 MULTI-EPI-SUBTILE: epi M-subtiles per CTA tile (==1 unless tile_m>epi_m). Per full-signal the
        # drain inner-loops these bands (each staged in staging[bidx, sub_m]; gi_base_s=tile_base+sub_m*epi_m).
        m_sub_per_tile: cutlass.Constexpr[int] = int(self.cta_tile_shape_mnk[0]) // epi_m
        run_j: cutlass.Constexpr[int] = int(
            self._a2a_cluster_run_j
        )  # run length (cluster-tile units, STATIC)
        cluster_n: cutlass.Constexpr[int] = int(self._a2a_cluster_n)
        tile_n: cutlass.Constexpr[int] = self.cta_tile_shape_mnk[
            1
        ]  # (d): nt_j_pp_rt / ncluster_n_rt
        # (d) MULTISLOT: FULL-BAND walk drain (rank-0 walks its bands, cp1 peers/band, slots alternate).
        multislot: cutlass.Constexpr[bool] = bool(getattr(self, "_a2a_cluster_multislot", False))
        n_consumer_warps: cutlass.Constexpr[int] = self._DECOUPLED_CONSUMER_WARPS
        ib_quiet: cutlass.Constexpr[bool] = bool(
            getattr(self, "_a2a_ib_quiet", True) and _int16_put_warp_a16 is not None
        )
        first_extra_warp: cutlass.Constexpr[int] = self._drain_first_warp()
        # (d) MULTISLOT: FULL-BAND walk drain -> a SEPARATE method (its band/peer loop replaces the
        # even-shard run loop). Dispatch at the METHOD level (const_expr -> a method-level return, SAFE;
        # unlike a return inside the runtime `if is_drain:`). Default -> the even-shard bounded-band drain.
        if const_expr(multislot):
            self._cluster_drain_loop_multislot(
                warp_idx,
                cluster_stage,
                recv_put,
                full_ptr,
                empty_ptr,
                meta,
                pe_table_dev,
                total_clusters,
            )
            return
        # Only cluster-rank-0 drains: single-warp by default (its FIRST consumer warp); mw = ALL
        # n_consumer_warps of rank-0 (job-split, the 'empty' mbar count doubles as the barrier -- see below).
        # Every other warp/rank falls through (no early return -- not traceable in CuTe DSL). The other ranks'
        # producers arrive rank-0's 'full' mbar cross-CTA; rank-0 arrives their 'empty'.
        lane = cute.arch.lane_idx()
        is_rank0 = cute.arch.block_idx_in_cluster() == Int32(0)
        # DSL-idiom: a COMPOUND `(A) & (B)` fed STRAIGHT into `if` does NOT hit the dynamic-if AST
        # rewrite (which handles a single Compare/Name test) -> falls to Python __bool__ -> DSLRuntimeError.
        # Assign the `&` to a var first, then `if <name>:` -- the proven pattern the differential drain
        # uses (do_put_i = do_put_i & (...); if do_put_i:). Single-warp rank-0 gate (first consumer warp).
        # SINGLE-WARP (default / db-no-mw): only rank-0's FIRST consumer warp drains. MULTI-WARP (mw): ALL
        # n_consumer_warps consumer warps of rank-0 drain (job-split by wid); the cross-CTA 'empty' mbar
        # count == n_consumer_warps (each drain warp arrives it) doubles as the "all-warps-done" barrier.
        is_drain = (warp_idx == Int32(first_extra_warp)) & is_rank0
        if is_drain:
            wid = warp_idx - Int32(
                first_extra_warp
            )  # consumer-warp idx [0, n_consumer_warps); ==0 single-warp
            gz = Int32(cute.arch.grid_dim()[2])  # == n_clusters (grid (1, cluster_n, n_clusters))
            z = Int32(cute.arch.block_idx()[2])  # this cluster's z-slot
            bidx = cute.arch.block_idx()[0] + cute.arch.block_idx()[2]  # staging index (== z here)
            # RUNTIME per-peer extents off the recv (cluster_drain 2-D is dynamic). total_runs =
            # total_clusters // run_j == ncluster_m*cp1*L (one run per (i-band, peer_j, plane); the
            # cluster N-tile count ncluster_n == cp1*run_j so total_clusters == ncluster_m*cp1*run_j*L).
            N_i_loc_rt = Int32(recv_put.shape[3])
            N_j_loc_rt = Int32(recv_put.shape[4])
            total_runs = total_clusters // Int32(run_j)
            n_groups = Int32(0)
            if z < total_runs:
                n_groups = (total_runs - z + gz - Int32(1)) // gz  # ceil_div(total_runs - z, gz)
            g = Int32(0)
            while g < n_groups:
                # CLUSTER-DRAIN full-wait : block until all cluster_n producers
                # have staged their column-slices of THIS run (full mbar count == cluster_n). Plain-arrive
                # mbar -> blocking mbarrier_wait is fine (no try_wait).
                # DOUBLE-BUFFER (db): rotate the staging slot = g%rd (mirrors the coalesce ring :1788), wait
                # full[slot] at phase (g//rd)&1, read cluster_stage[bidx, slot]. SINGLE-BUFFER (default):
                # slot 0, phase g&1, cluster_stage[bidx] -- only the else branch traces when db off ->
                # byte-identical.
                cute.arch.mbarrier_wait(full_ptr, g & Int32(1))
                base = Int32(0)  # single synchronous buffer -> slot 0 (correctness-first)
                gi_base = meta[base + Int32(5)]  # band's GLOBAL token-i base (pe_aligned)
                peer_j = meta[base + Int32(0)]  # this run's peer_j (peer = peer_i*cp1 + peer_j)
                d = meta[base + Int32(3)]
                b = meta[base + Int32(4)]
                recv_blk = recv_put[my_cp_rank, d, b, None, None]  # (N_i_loc, N_j_loc) int16
                # 64-BIT PEER OFFSET -- the leading (cp, Dloc, B) terms, formed ONCE per band.
                # recv_put is (cp, Dloc, B, N_i_loc, N_j_loc) int16 and its cp-mode stride is
                # Dloc*B*N_i_loc*N_j_loc ELEMENTS. Under a BAKED token extent that stride is a
                # compile-time constant, so `my_cp_rank * stride` is formed in 32 bits and WRAPS once
                # the recv holds 2**31 elements (4 GiB at bf16) -- the wrapped value then becomes the
                # put destination. Same mechanism and same fix as the front A2A's IB drain. CROSS-NODE
                # ONLY: the P2P arm reaches this recv through a TMA descriptor, addressed in 64 bits by
                # the hardware. MEASURED on 2 nodes x 1 rank, cp=2, Dloc=8: N=12280 (0.56x 2**31) passes
                # in both shape modes and N=20000 (1.49x) passes DYNAMIC and faulted STATIC -- one
                # variable, called in advance.
                _recv_off64 = (
                    cutlass.Int64(my_cp_rank) * cutlass.Int64(recv_put.layout.stride[0])
                    + cutlass.Int64(d) * cutlass.Int64(recv_put.layout.stride[1])
                    + cutlass.Int64(b) * cutlass.Int64(recv_put.layout.stride[2])
                )
                # #78 MULTI-EPI-SUBTILE: drain the m_sub_per_tile epi row-bands of THIS run. Each was staged
                # into staging[bidx, sub_m] by the producer; its GLOBAL token-i base = the TILE base (meta[5],
                # written at first_in_run == sub_m 0) + sub_m*epi_m. m_sub==1 -> range_constexpr(1) folds to
                # the single-band path (db slot / single-buffer cluster_stage[bidx]) -> byte-identical. ONE
                # empty-arrive per full-signal (after ALL sub_m bands) mirrors the producer's per-run reuse
                # gate -> mbar counts/phases UNCHANGED (deadlock-safe).
                for sub_m in cutlass.range_constexpr(m_sub_per_tile):
                    if const_expr(m_sub_per_tile > 1):
                        stage_blk = cluster_stage[
                            bidx, sub_m, None, None
                        ]  # (epi_m, N_j_loc) sub_m band
                        gi_base_s = gi_base + Int32(sub_m * epi_m)
                    else:
                        stage_blk = cluster_stage[
                            bidx, None, None
                        ]  # (epi_m, N_j_loc) row-packed int16
                        gi_base_s = gi_base
                    row = Int32(0)
                    job_idx = Int32(
                        0
                    )  # i-run counter for the mw round-robin (unused/DCE'd when mw off)
                    while row < Int32(epi_m):
                        gi = gi_base_s + row
                        peer_i = gi // N_i_loc_rt
                        i_local = gi % N_i_loc_rt
                        to_peer = N_i_loc_rt - i_local  # rows to the next i-peer boundary (>=1)
                        to_band = Int32(epi_m) - row  # rows to the band end (>=1)
                        run = to_peer if to_peer < to_band else to_band  # single-i-peer run (>=1)
                        do_put = peer_i < Int32(cp0)  # drop the past-M i-axis ceil spill
                        if do_put:  # single-warp -> every run
                            peer = peer_i * Int32(cp1) + peer_j
                            pe_row = pe_table_dev[peer]
                            src_row = stage_blk[row, None]  # (N_j_loc,) -> run*N_j_loc contig
                            dst_row = recv_blk[i_local, None]  # (N_j_loc,) -> run*N_j_loc contig
                            # ...and the i term, added in 64 bits. `dst_row` is kept for its LAYOUT; its iterator
                            # is not used for the put, because that offset was already folded at 32 bits.
                            _dst_addr64 = recv_put.iterator.toint() + (
                                _recv_off64
                                + cutlass.Int64(i_local) * cutlass.Int64(recv_put.layout.stride[3])
                            ) * cutlass.Int64(2)
                            nelem = run * N_j_loc_rt
                            sp = cute.make_ptr(
                                cutlass.Int16,
                                src_row.iterator.toint(),
                                cute.AddressSpace.gmem,
                                assumed_align=16,
                            )
                            dp = cute.make_ptr(
                                cutlass.Int16,
                                _dst_addr64,
                                cute.AddressSpace.gmem,
                                assumed_align=16,
                            )
                            sr = cute.make_tensor(sp, cute.make_layout((Int32(1), nelem)))
                            dr = cute.make_tensor(dp, cute.make_layout((Int32(1), nelem)))
                            if const_expr(ib_quiet):
                                _put_warp_int16_a16(dr, sr, pe_row)
                            else:
                                _put_nbi_warp_int16_a16(dr, sr, pe_row)
                        job_idx = job_idx + Int32(1)
                        row = row + run
                cute.arch.sync_warp()
                # CLUSTER-DRAIN empty-arrive : the shared staging is fully drained (the BLOCKING
                # put_warp completed the DMA -> reuse is safe) -> release each cluster CTA's 'empty' reuse
                # gate so its producer may overwrite the slice for the next run. First-class cross-CTA
                # arrive (peer_cta_rank_in_cluster=r; r==0 is rank-0 itself).
                # MULTI-WARP (mw): EACH of rank-0's n_consumer_warps drain warps (its lane 0) arrives
                # empty[slot] -> the per-CTA 'empty' mbar count == n_consumer_warps DOUBLES as the
                # "all-drain-warps-done" barrier (no named-barrier id): the producer's reuse-wait returns
                # only after every drain warp finished its BLOCKING puts. SINGLE-WARP: only the one drain
                # warp arrives (count 1). DOUBLE-BUFFER (db): arrive rotates to empty[slot]; else empty[0].
                if lane == Int32(0):
                    for r in range(cluster_n):
                        cute.arch.mbarrier_arrive(empty_ptr, peer_cta_rank_in_cluster=r)
                g += Int32(1)

    @cute.jit
    def _cluster_drain_loop_multislot(
        self,
        warp_idx,
        cluster_stage,
        recv_put,
        full_ptr,
        empty_ptr,
        meta,
        pe_table_dev,
        total_clusters,
    ):
        """#76 (d) MULTISLOT drain: cluster-rank-0 walks its FULL-BAND bands (z, z+gz, ..., < total_bands ==
        ncluster_m*L). Each band == one i-band's full N-walk == cp1 REAL peer_j in order (0..cp1-1) -> the 2
        rotating full-peer slots alternate (slot = peer_j&1). Per (band, pj): wait full[slot] @ the per-slot
        phase (df_slot&1), read meta[slot] (gi_base/d/b), fire the coalesced run*N_j_loc put per same-peer_i
        i-run to recv[peer_i*cp1+pj], arrive empty[slot] cross-CTA (release the producer reuse-gate). The
        producer arrives full[slot] EXACTLY cluster_n times per peer (real-arrive-on-cross + the tail flush
        of the walk's last peer) -> the wait count + the alternating-slot phase stay in lockstep, no null-
        arrive / no data_present. Single-warp rank-0 (concentration-preserving). Padding peers (peer_j>=cp1,
        the ceil-spill) are NEVER produced/drained (the producer skips them). ``cluster_stage`` int16
        ``(n_clusters, rd, epi_m, N_j_loc)`` (rd=2 slots); ``recv_put`` the LOCAL recv int16."""
        cp: cutlass.Constexpr[int] = self._a2a_cp
        my_cp_rank: cutlass.Constexpr[int] = self._a2a_my_cp_rank
        cp1: cutlass.Constexpr[int] = int(getattr(self, "_a2a_cp1", 1))
        cp0: cutlass.Constexpr[int] = int(getattr(self, "_a2a_cp0", cp))
        rd: cutlass.Constexpr[int] = self._a2a_ring_depth
        nfields: cutlass.Constexpr[int] = self._DECOUPLED_META_FIELDS
        epi_m: cutlass.Constexpr[int] = self.epi_tile[0]
        tile_n: cutlass.Constexpr[int] = self.cta_tile_shape_mnk[1]
        cluster_n: cutlass.Constexpr[int] = int(self._a2a_cluster_n)
        ib_quiet: cutlass.Constexpr[bool] = bool(
            getattr(self, "_a2a_ib_quiet", True) and _int16_put_warp_a16 is not None
        )
        first_extra_warp: cutlass.Constexpr[int] = self._drain_first_warp()
        lane = cute.arch.lane_idx()
        is_rank0 = cute.arch.block_idx_in_cluster() == Int32(0)
        # Single-warp rank-0 drain (concentration-preserving): rank-0's FIRST consumer warp only.
        is_drain = (warp_idx == Int32(first_extra_warp)) & is_rank0
        if is_drain:
            gz = Int32(cute.arch.grid_dim()[2])  # == n_clusters (grid (1, cluster_n, n_clusters))
            z = Int32(cute.arch.block_idx()[2])  # this cluster's z-slot
            bidx = cute.arch.block_idx()[0] + cute.arch.block_idx()[2]  # staging index (== z here)
            N_i_loc_rt = Int32(recv_put.shape[3])
            N_j_loc_rt = Int32(recv_put.shape[4])
            # total_bands == ncluster_m*L == total_clusters // ncluster_n (cluster units); ncluster_n ==
            # ceil(cp1*nt_j_pp / cluster_n) (== the scheduler's ncluster_n) computed off the runtime recv.
            nt_j_pp_rt = (N_j_loc_rt + Int32(tile_n) - Int32(1)) // Int32(tile_n)
            ncluster_n_rt = (Int32(cp1) * nt_j_pp_rt + Int32(cluster_n) - Int32(1)) // Int32(
                cluster_n
            )
            total_bands = total_clusters // ncluster_n_rt
            band = z
            df0 = Int32(0)  # per-slot drain-fill counters -> full[slot] wait phase == df_slot & 1
            df1 = Int32(0)
            while band < total_bands:
                pj = Int32(0)
                while pj < Int32(cp1):
                    slot = pj % Int32(2)
                    # full[slot] wait @ (fills of THIS slot so far)&1. Pick df0/df1 by arithmetic (slot in
                    # {0,1}) -> no runtime ternary; rd(=2)-match for the runtime slot mbar ptr.
                    cur_fill = df0 * (Int32(1) - slot) + df1 * slot
                    phase = cur_fill & Int32(1)
                    for ss in cutlass.range_constexpr(rd):
                        if slot == Int32(ss):
                            cute.arch.mbarrier_wait(full_ptr + ss, phase)
                    base = slot * Int32(nfields)
                    gi_base = meta[base + Int32(5)]  # this (band, peer_j)'s GLOBAL token-i base
                    d = meta[base + Int32(3)]
                    b = meta[base + Int32(4)]
                    # 64-BIT PEER OFFSET -- the leading (cp, Dloc, B) terms, formed ONCE per band.
                    # recv_put is (cp, Dloc, B, N_i_loc, N_j_loc) int16 and its cp-mode stride is
                    # Dloc*B*N_i_loc*N_j_loc ELEMENTS. Under a BAKED token extent that stride is a
                    # compile-time constant, so `my_cp_rank * stride` is formed in 32 bits and WRAPS once
                    # the recv holds 2**31 elements (4 GiB at bf16) -- the wrapped value then becomes the
                    # put destination. Same mechanism and same fix as the front A2A's IB drain. CROSS-NODE
                    # ONLY: the P2P arm reaches this recv through a TMA descriptor, addressed in 64 bits by
                    # the hardware. MEASURED on 2 nodes x 1 rank, cp=2, Dloc=8: N=12280 (0.56x 2**31) passes
                    # in both shape modes and N=20000 (1.49x) passes DYNAMIC and faulted STATIC -- one
                    # variable, called in advance.
                    _recv_off64 = (
                        cutlass.Int64(my_cp_rank) * cutlass.Int64(recv_put.layout.stride[0])
                        + cutlass.Int64(d) * cutlass.Int64(recv_put.layout.stride[1])
                        + cutlass.Int64(b) * cutlass.Int64(recv_put.layout.stride[2])
                    )
                    recv_blk = recv_put[my_cp_rank, d, b, None, None]  # (N_i_loc, N_j_loc) int16
                    stage_blk = cluster_stage[
                        bidx, slot, None, None
                    ]  # this slot's (epi_m, N_j_loc)
                    row = Int32(0)
                    while row < Int32(epi_m):
                        gi = gi_base + row
                        peer_i = gi // N_i_loc_rt
                        i_local = gi % N_i_loc_rt
                        to_peer = N_i_loc_rt - i_local  # rows to the next i-peer boundary (>=1)
                        to_band = Int32(epi_m) - row  # rows to the band end (>=1)
                        run = to_peer if to_peer < to_band else to_band  # single-i-peer run (>=1)
                        do_put = peer_i < Int32(cp0)  # drop the past-M i-axis ceil spill
                        if do_put:
                            peer = peer_i * Int32(cp1) + pj  # pj == this peer_j (runtime)
                            pe_row = pe_table_dev[peer]
                            src_row = stage_blk[row, None]  # (N_j_loc,) -> run*N_j_loc contig
                            dst_row = recv_blk[i_local, None]  # (N_j_loc,) -> run*N_j_loc contig
                            # ...and the i term, added in 64 bits. `dst_row` is kept for its LAYOUT; its iterator
                            # is not used for the put, because that offset was already folded at 32 bits.
                            _dst_addr64 = recv_put.iterator.toint() + (
                                _recv_off64
                                + cutlass.Int64(i_local) * cutlass.Int64(recv_put.layout.stride[3])
                            ) * cutlass.Int64(2)
                            nelem = run * N_j_loc_rt
                            sp = cute.make_ptr(
                                cutlass.Int16,
                                src_row.iterator.toint(),
                                cute.AddressSpace.gmem,
                                assumed_align=16,
                            )
                            dp = cute.make_ptr(
                                cutlass.Int16,
                                _dst_addr64,
                                cute.AddressSpace.gmem,
                                assumed_align=16,
                            )
                            sr = cute.make_tensor(sp, cute.make_layout((Int32(1), nelem)))
                            dr = cute.make_tensor(dp, cute.make_layout((Int32(1), nelem)))
                            if const_expr(ib_quiet):
                                _put_warp_int16_a16(dr, sr, pe_row)
                            else:
                                _put_nbi_warp_int16_a16(dr, sr, pe_row)
                        row = row + run
                    cute.arch.sync_warp()
                    # (d) SERIALIZE (option a): ORDER the completed blocking put (its ibgda_quiet
                    # completed the NIC RDMA-read of this slot) BEFORE the CROSS-CTA empty-arrive, so the
                    # producer CTA that waits empty[slot] observes the NIC-read completion before it reuses/
                    # overwrites. The NIC RDMA-read is a SYSTEM-scope memory agent, so a GPU-scope fence does
                    # NOT order it to a cross-CTA consumer (measured: rank-0 clean but sibling ranks still
                    # fault). fence_acq_rel_SYS orders the NIC read system-wide before the empty release.
                    cute.arch.fence_acq_rel_sys()
                    cute.arch.sync_warp()
                    # arrive empty[slot] cross-CTA to EVERY cluster CTA -> release its producer reuse-gate
                    # (the BLOCKING put completed the DMA -> the slice is safe to overwrite for peer+2).
                    if lane == Int32(0):
                        for r in range(cluster_n):
                            for ss in cutlass.range_constexpr(rd):
                                if slot == Int32(ss):
                                    cute.arch.mbarrier_arrive(
                                        empty_ptr + ss, peer_cta_rank_in_cluster=r
                                    )
                    # advance the per-slot fill (df0 if slot0 else df1) -> next same-slot wait phase.
                    df0 = df0 + (Int32(1) - slot)
                    df1 = df1 + slot
                    pj = pj + Int32(1)
                band = band + gz

    def _a2a_decoupled_producer_copy_fn(
        self, epi_params, tile_shape_mn, epi_tile, sD, tile_coord_mnkl, storage
    ):
        """Build the PRODUCER copy_fn (T2.3 decoupled store): SMEM box -> local-GMEM ring + signal.

        The MMA warpgroup's ``is_tma_warp`` calls this per epilogue subtile. Instead of the
        coupled (awaited) peer TMA-S2G, it: (1) waits the slot's ``empty`` mbarrier (skipped on
        first use), (2) copies the swizzled SMEM box ``sD[:,:,src_idx]`` into the CTA's local-GMEM
        ring slot (a fast LOCAL store, no NVLink), (3) writes the per-slot peer/recv-index metadata
        for the consumer, (4) arrives the slot's ``full`` mbarrier + bumps the producer counter.
        The MMA warpgroup then advances WITHOUT waiting for the peer put -> the put latency hides
        behind the next tile's MMA (the consumer warpgroup drains the ring overlapped).

        The peer/recv-index math (cp0/cp1 -> peer, i_tile/j_tile/d/b) is IDENTICAL to the coupled
        ``_a2a_peer_store_copy_fn_gemm_native``; here it is computed to WRITE the metadata the
        consumer reads (the consumer issues the actual put)."""
        cp: cutlass.Constexpr[int] = self._a2a_cp
        my_cp_rank: cutlass.Constexpr[int] = self._a2a_my_cp_rank
        # The batch extent must NOT be a compile-time constant: `B` reaches this kernel ONLY as the
        # divisor of the GEMM batch coord (`d = L // B`, `b = L % B`), so baking it made one
        # compiled kernel serve one batch extent for no design reason. It is READ instead: mode 4
        # of the permuted peer recv (N_i_loc, N_j_loc, cp, Dloc, B) IS `B`. Runtime `Int32` wherever
        # the caller marked the recv dynamic (the production `TriMulAutotuned` path), a folded
        # Python int on a per-shape static compile -- the same split `N_loc_rt` has at :3903, and
        # why there is no branch here. There is deliberately NO static-B mode.
        B = epi_params.peer_tensors[0].shape[4]
        cp_axis_sizes = self._a2a_cp_axis_sizes
        cp1: cutlass.Constexpr[int] = cp_axis_sizes[1] if len(cp_axis_sizes) > 1 else 1
        cp1_stride: cutlass.Constexpr[int] = cp1
        tile_m: cutlass.Constexpr[int] = tile_shape_mn[0]
        tile_n: cutlass.Constexpr[int] = tile_shape_mn[1]
        epi_m: cutlass.Constexpr[int] = epi_tile[0]
        epi_n: cutlass.Constexpr[int] = epi_tile[1]
        m_sub_per_tile: cutlass.Constexpr[int] = tile_m // epi_m
        n_sub_per_tile: cutlass.Constexpr[int] = tile_n // epi_n
        rd: cutlass.Constexpr[int] = self._a2a_ring_depth
        nfields: cutlass.Constexpr[int] = self._DECOUPLED_META_FIELDS
        # §7.16 STRIDED+WIDEN: GROUP the n_sub_per_tile j-subtiles (same sub_m, adjacent j -> contiguous
        # in the peer box, j-innermost) into ONE tile-wide ring slot (epi_m, tile_n); the consumer puts
        # the slot's rows as tile_n-WIDE (>=256 B) runs. The producer counter t advances per subtile;
        # group g = t // grp_sz, slot = g % rd, j-within-group = t % grp_sz == sub_n (j-first iteration).
        strided: cutlass.Constexpr[bool] = self._a2a_consumer_strided
        # COALESCE (§0.9.6): widen the group from ONE CTA tile's n_sub_per_tile subtiles (a tile_n-wide
        # slot) to the WHOLE token-i band's ncluster_n*n_sub_per_tile subtiles (a full-N (epi_m, N)
        # slot). run_j_tiles=ncluster_n makes the band's ncluster_n j-tiles CONSECUTIVE in the producer
        # counter t, so t%grp_sz sweeps [0, N/epi_n) exactly (col_base = (t%grp_sz)*epi_n = the token-j
        # column; PROVEN for the only supported tile m_sub_per_tile==1). Default (coalesce off) is
        # byte-identical.
        # #78 grow-rows: for tile_m>epi_m the band has m_sub_per_tile*ncluster_n*n_sub subtiles (each tile
        # emits m_sub*n_sub epi-boxes, N-major), so grp_sz *= m_sub_per_tile -> first/last_in_grp still fire
        # ONCE per band -> 1 full-arrive/band (handshake count UNCHANGED). m_sub==1 -> ×1 byte-identical.
        grp_sz: cutlass.Constexpr[int] = n_sub_per_tile if strided else 1
        # token-scaling extents (peer block -> tiles): from the peer recv runtime/static shape.
        # arbitrary_n (1-D token shard, cp1==1): the per-peer block may NOT be a multiple of the CTA tile
        # (partial last M/N-tile) -> CEIL-divide so EVERY global tile (incl the partial last) maps into the
        # single per-axis block. For 1-D this makes cp1_coord = tile_coord_N // tiles_per_j_block == 0 and
        # tile_in_j_block == tile_coord_N for ALL j-tiles (so the consumer's j_grp reaches the partial
        # tile). Default (aligned / 2-D) keeps FLOOR — identical when divisible -> byte-identical PTX.
        arb_n_p: cutlass.Constexpr[bool] = bool(getattr(self, "_a2a_arbitrary_n", False))
        peer0 = epi_params.peer_tensors[0]  # permuted peer recv (N_i_loc, N_j_loc, cp, Dloc, B)
        if const_expr(self._a2a_dynamic):
            if const_expr(arb_n_p):
                tiles_per_i_block = (peer0.shape[0] + tile_m - 1) // tile_m
                tiles_per_j_block = (peer0.shape[1] + tile_n - 1) // tile_n
            else:
                tiles_per_i_block = peer0.shape[0] // tile_m
                tiles_per_j_block = peer0.shape[1] // tile_n
        elif const_expr(arb_n_p):
            tiles_per_i_block: cutlass.Constexpr[int] = (int(peer0.shape[0]) + tile_m - 1) // tile_m
            tiles_per_j_block: cutlass.Constexpr[int] = (int(peer0.shape[1]) + tile_n - 1) // tile_n
        else:
            tiles_per_i_block: cutlass.Constexpr[int] = int(peer0.shape[0]) // tile_m
            tiles_per_j_block: cutlass.Constexpr[int] = int(peer0.shape[1]) // tile_n
        full_ptr = storage.decoupled.full.data_ptr()
        empty_ptr = storage.decoupled.empty.data_ptr()
        meta = storage.decoupled.meta.get_tensor((rd * nfields,))  # tensor view (indexed write)
        pcount = storage.decoupled.pcount.get_tensor((1,))
        next_slot = storage.decoupled.next_slot.get_tensor((1,))  # SMEM walk-state view
        # §7.16o REDESIGN: the dtma tail consumer's ONLY producer->consumer signal — a per-CTA
        # produced-group counter. Present only on the tail-double-TMA struct (None elsewhere). The
        # producer bumps it (release) right after the ring-write LANDS (the existing cp_async_bulk_
        # wait_group(0) fence below); the tail spins until produced > g (acquire). Decouples the drain
        # from any shared/hand-rolled mbarrier (the cause of the prior dtma hang).
        tail_dtma: cutlass.Constexpr[bool] = False  # TMA-tail / double-TMA drains removed
        produced_ptr = None
        tma_drain: cutlass.Constexpr[bool] = False  # SMEM-ring TMA (decoupled_tma) drain removed
        producer_tma: cutlass.Constexpr[bool] = self._a2a_producer_tma
        prod_pipelined: cutlass.Constexpr[bool] = False  # producer ablation knobs removed
        prod_no_handshake: cutlass.Constexpr[bool] = False
        # arbitrary_n: append the epi-box global-i base (meta field 5) so the per-row SIMT drain can
        # route each row to its own peer/i_local (a straddling box fans rows out to >=2 peers). Default
        # OFF -> nfields stays 5 and this meta write is const_expr-elided (byte-identical metadata).
        arbitrary_n: cutlass.Constexpr[bool] = bool(getattr(self, "_a2a_arbitrary_n", False))
        # IB-RING (§0.9): the per-peer M-tiling scheduler is active (pe_aligned), so tile_coord_mnkl[0]
        # is the PER-PEER linear tile index m_linear in [0, cp*nt_pp), NOT a global M-tile. The epi-box's
        # global token-i base is then peer*N_loc + k*tile_m (peer=m_linear//nt_pp, k=m_linear%nt_pp),
        # which the gi_base meta must carry (else the drain's gi->peer route mis-scatters for peer>=1).
        pe_aligned_ib: cutlass.Constexpr[bool] = bool(getattr(self, "_pe_aligned_ib_ring", False))

        # ---- PRODUCER ring-write mechanism ----
        if const_expr(producer_tma):
            # I2b producer refinement: issue the SAME async TMA-S2G the single-device GEMM uses,
            # retargeted at the GMEM ring slot (O(1) MMA-warp occupancy). tma_partition the ring atom
            # (box <-> sD), mirroring design-E. ring_store_tensor is (epi_m, epi_n, ring_depth, grid_CTAs).
            ring_atom = epi_params.ring_store_atom
            gRing = cute.flat_divide(epi_params.ring_store_tensor, epi_tile)  # (em,en,1,1,rd,grid)
            s_ring, g_ring = cpasync.tma_partition(
                ring_atom,
                0,
                cute.make_layout(1),
                cute.group_modes(sD, 0, cute.rank(sD) - 1),
                cute.group_modes(gRing, 0, 2),
            )
            bidx = cute.arch.block_idx()[0] + cute.arch.block_idx()[2]
        elif const_expr(tma_drain):
            # TMA drain (Option B): a depth-rd SMEM ring (same swizzled epi layout as sD); the producer
            # SIMT-copies sD[buf] -> sRing[slot] (SMEM->SMEM), the consumer TMA-S2G's sRing[slot]->peer.
            ring_layout = fold_cp_ops_sm90_utils.make_smem_layout_epi(
                self.d_dtype, self.d_layout, self.epi_tile, rd
            )
            sRing = storage.decoupled.sring.get_tensor(ring_layout.outer, swizzle=ring_layout.inner)
        else:
            # SIMT drain (I2a): a local-GMEM ring (passed via epi_params); SMEM->GMEM de-swizzle copy.
            ring = epi_params.decoupled_ring  # (grid_CTAs, ring_depth, epi_m, epi_n)
            bidx = cute.arch.block_idx()[0] + cute.arch.block_idx()[2]  # cluster(1,1,*): only z
            ring_cta = ring[bidx, None, None, None]  # (ring_depth, epi_m, epi_n)
        # 32-lane vectorized SIMT copy (de-swizzle on read). Only used on the SIMT-producer paths.
        elem_ty = sD.element_type
        elem_bits: cutlass.Constexpr[int] = int(elem_ty.width)
        vec: cutlass.Constexpr[int] = max(1, min(128 // elem_bits, epi_n))
        vec_groups: cutlass.Constexpr[int] = epi_n // vec
        rows_of_lanes: cutlass.Constexpr[int] = max(1, min(epi_m, 32 // max(1, vec_groups)))
        cp_atom = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(), elem_ty, num_bits_per_copy=vec * elem_bits
        )
        cp_tiled = cute.make_tiled_copy_tv(
            cp_atom,
            cute.make_ordered_layout((rows_of_lanes, vec_groups), order=(1, 0)),
            cute.make_layout((1, vec)),
        )

        @cute.jit
        def copy_fn(src_idx, dst_idx, **kwargs):
            sub_m, sub_n = dst_idx[0], dst_idx[1]
            L = tile_coord_mnkl[3]
            d = L // Int32(B)
            b = L % Int32(B)
            cp0_coord = tile_coord_mnkl[0] // Int32(tiles_per_i_block)
            cp1_coord = tile_coord_mnkl[1] // Int32(tiles_per_j_block)
            peer = cp0_coord * Int32(cp1_stride) + cp1_coord
            tile_in_i_block = tile_coord_mnkl[0] % Int32(tiles_per_i_block)
            tile_in_j_block = tile_coord_mnkl[1] % Int32(tiles_per_j_block)
            i_tile = tile_in_i_block * Int32(m_sub_per_tile) + sub_m
            j_tile = tile_in_j_block * Int32(n_sub_per_tile) + sub_n
            # arbitrary_n: the GLOBAL token-i base of this epi-box, which the per-row drain routes by
            # (gi = gi_base + row -> peer = gi//N_loc, i_local = gi%N_loc).
            #   * DEFAULT (design-E global M-tiling): tile_coord_mnkl[0] is a GLOBAL M-tile index, so the
            #     base is tile_coord_M*tile_m + sub_m*epi_m.
            #   * IB-RING (pe_aligned per-peer M-tiling): tile_coord_mnkl[0] is a PER-PEER linear tile
            #     index m_linear; the global base is peer*N_loc + k*tile_m + sub_m*epi_m (the
            #     _pe_tiled_base_m formula). cp0_coord (== m_linear//nt_pp == peer) and tile_in_i_block
            #     (== m_linear%nt_pp == k) are already computed above (tiles_per_i_block == nt_pp on the
            #     1-D pe_aligned ceil path), and peer0.shape[0] == N_loc (RUNTIME under dynamic). This
            #     reproduces pe_aligned's reshard mapping over the GMEM-ring drain. const_expr-gated so
            #     the default path traces the identical single-mul base (byte-identical PTX).
            if const_expr(arbitrary_n):
                if const_expr(pe_aligned_ib):
                    gi_base = (
                        cp0_coord * Int32(peer0.shape[0])
                        + tile_in_i_block * Int32(tile_m)
                        + sub_m * Int32(epi_m)
                    )
                else:
                    gi_base = tile_coord_mnkl[0] * Int32(tile_m) + sub_m * Int32(epi_m)

            lane = cute.arch.lane_idx()
            # producer counter -> slot + phase (single-writer warp; uniform read across lanes).
            t = pcount[0]
            # STRIDED: the slot/handshake granularity is the GROUP (grp_sz subtiles -> one tile-wide
            # slot), so slot/phase index by group g=t//grp_sz; the j-within-group selects the slot's
            # tile_n col-band. Non-strided (grp_sz==1): g==t -> today's per-subtile path verbatim.
            g = t // Int32(grp_sz)
            j_in_grp = t % Int32(grp_sz)  # == sub_n on the j-first iteration
            last_in_grp = j_in_grp == Int32(grp_sz - 1)
            slot = g % Int32(rd)
            k = g // Int32(rd)
            first_in_grp = j_in_grp == Int32(0)
            # wait the slot's prior occupant to be drained (skip on first use of the slot).
            # A′ ablation (i): no_handshake => NO consumer warpgroup exists, so there is nobody to
            # arrive empty[s] -> skip the reuse-wait (the deferred wait_group(rd-1) below still
            # serializes ring-write reuse at the TMA level, which is the real producer cost).
            # STRIDED: gate the reuse-wait on the FIRST subtile of the group (one wait per slot reuse).
            if const_expr(not prod_no_handshake):
                # STRIDED: only the FIRST subtile of a group reuses the slot -> wait there once. The
                # later j-bands write into the SAME (already-empty) slot, no extra wait.
                do_reuse_wait = first_in_grp if const_expr(strided) else cutlass.Boolean(True)
                if k >= Int32(1):
                    if do_reuse_wait:
                        for ss in cutlass.range_constexpr(rd):
                            if slot == Int32(ss):
                                cute.arch.mbarrier_wait(empty_ptr + ss, (k - Int32(1)) & Int32(1))
            # Write the epilogue tile -> the ring slot:
            #   producer_tma (I2b): async TMA-S2G sD[buf] -> GMEM-ring[slot,cta] (O(1) MMA-warp), then
            #     a CHEAP local commit+wait_group (local HBM, not the NVLink stall) so the slot is
            #     fully written before full[s]. This restores the MMA store cost to the single-device
            #     ceiling; the slow SymMEM put lives entirely on the consumer.
            #   tma_drain (Option B): SIMT SMEM->SMEM (sRing[slot]).
            #   SIMT drain (I2a): SIMT SMEM->local-GMEM (ring_cta[slot], de-swizzle).
            if const_expr(producer_tma):
                with cute.arch.elect_one():
                    # g_ring grid modes after flat_divide(ring_view,(em,en)) + group:
                    #   non-strided ring (em,en,rd,grid): (box, nt_i=1, nt_j=1, rd, grid) ->
                    #     coord (None, 0, 0, slot, cta).
                    #   STRIDED tile-wide ring (em,tile_n,rd,grid): (box, nt_i=1, n_sub, rd, grid)
                    #     -> coord (None, 0, sub_n, slot, cta): the j-band sub_n of the tile slot.
                    # index the ring's rd mode by the RUNTIME slot (the bounded ring's strided j-band
                    # match, grp_sz = n_sub_per_tile ~4, stays unrolled -- cheap).
                    for ss in cutlass.range_constexpr(rd):
                        if slot == Int32(ss):
                            if const_expr(strided):
                                for jj in cutlass.range_constexpr(grp_sz):
                                    if j_in_grp == Int32(jj):
                                        cute.copy(
                                            ring_atom,
                                            s_ring[(None, src_idx)],
                                            g_ring[(None, Int32(0), Int32(jj), Int32(ss), bidx)],
                                        )
                            else:
                                cute.copy(
                                    ring_atom,
                                    s_ring[(None, src_idx)],
                                    g_ring[(None, Int32(0), Int32(0), Int32(ss), bidx)],
                                )
                    cute.arch.cp_async_bulk_commit_group()
                    if const_expr(prod_pipelined):
                        # A′ PIPELINED (the §7.14 ablation / keep-worthy change): DEFER the wait like
                        # the single-device epi_store_pipeline. Keep rd-1 ring-write bulk commits in
                        # flight; wait_group(rd-1, read=False) only awaits the slot about to be REUSED.
                        # With a LARGE ring the producer never stalls on reuse -> the ring write hides
                        # behind the next tile's MMA exactly like a normal pipelined TMA store. The
                        # consumer (when on) must still observe the LANDED slot, which the full[s]
                        # arrive after this deferred wait guarantees for the slot rd-1 tiles ago; the
                        # most-recent rd-1 slots may still be in flight at full[s].arrive, so the
                        # PIPELINED mode is correctness-safe ONLY for ring_depth >= consumer lookahead
                        # (here used for drain-elided timing, where correctness is not required).
                        cute.arch.cp_async_bulk_wait_group(rd - 1, read=False)
                    else:
                        # WAIT FOR THE WRITE TO LAND (read=False), not just source-reuse: the consumer
                        # warpgroup reads this ring slot from GMEM, so the bulk store must be COMPLETE
                        # (visible) before full[s].arrive — read=True would only guarantee the SMEM
                        # source is reusable and let the consumer read a half-written slot (corruption).
                        # This is a cheap LOCAL-HBM completion wait, NOT the NVLink stall. SERIALIZES.
                        cute.arch.cp_async_bulk_wait_group(0, read=False)
                cute.arch.sync_warp()
            else:
                s_box = cute.slice_(sD, (None, None, src_idx))  # (epi_m, epi_n) swizzled
                thr = cp_tiled.get_slice(lane)
                for ss in cutlass.range_constexpr(rd):
                    if slot == Int32(ss):
                        if const_expr(tma_drain):
                            dst_slot = cute.slice_(sRing, (None, None, ss))  # (epi_m, epi_n) SMEM
                        else:
                            dst_slot = ring_cta[ss, None, None]  # (epi_m, epi_n) local GMEM
                        cute.copy(cp_tiled, thr.partition_S(s_box), thr.partition_D(dst_slot))
            # write per-slot metadata (one lane) so the consumer can address the peer recv.
            # A′ ablation (i): no_handshake => no consumer reads meta -> skip the metadata writes too
            # (pure ring-write cost). The peer/i_tile/etc. math above is still traced (it is cheap
            # integer arithmetic, identical to the real path) so the timing is representative.
            cute.arch.sync_warp()
            # STRIDED: ONE meta entry per group, written at the first j-band; j_tile here is the group's
            # FIRST j-subtile (j_base) -> the consumer reconstructs the tile_n-wide dst box from it.
            do_meta = first_in_grp if const_expr(strided) else cutlass.Boolean(True)
            if const_expr(not prod_no_handshake):
                if lane == Int32(0):
                    if do_meta:
                        base = slot * Int32(nfields)
                        meta[base + Int32(0)] = peer
                        meta[base + Int32(1)] = i_tile
                        meta[base + Int32(2)] = j_tile
                        meta[base + Int32(3)] = d
                        meta[base + Int32(4)] = b
                        # arbitrary_n: the per-row drain needs the box's GLOBAL token-i base (field 5).
                        if const_expr(arbitrary_n):
                            meta[base + Int32(5)] = gi_base
            # TMA drain (SMEM-ring consumer): fence the SMEM ring write into the async proxy so the
            # consumer's TMA-S2G (cp.async.bulk reads SMEM via the async proxy) observes it after full.
            if const_expr(tma_drain):
                cute.arch.fence_view_async_shared()
            cute.arch.sync_warp()  # all stores (data + meta) visible before signaling full
            # A′ ablation (i): no_handshake => no consumer warpgroup waits on full[s] -> skip arrive.
            # STRIDED: signal full only at the LAST j-band (the whole tile slot is now landed).
            do_full = last_in_grp if const_expr(strided) else cutlass.Boolean(True)
            if const_expr(not prod_no_handshake):
                if do_full:
                    # Bounded rotating ring: signal full[slot] via the static-offset range_constexpr(rd)
                    # match (the consumer's wait tracks the phase k&1).
                    for ss in cutlass.range_constexpr(rd):
                        if slot == Int32(ss):
                            cute.arch.mbarrier_arrive(full_ptr + ss)
            # §7.16o REDESIGN dtma path: the ring-write for this GROUP has LANDED in GMEM (the
            # producer_tma branch above did cp_async_bulk_wait_group(0, read=False) before this, and
            # the sync_warp at meta-write made the data+meta visible). Bump the per-CTA produced-group
            # counter with RELEASE so any cta-scope acquirer (the tail consumer) that observes
            # produced > g is guaranteed to see this group's landed ring bytes + meta. ONE lane.
            # Groups are produced strictly in order (write-once: g monotonically increases with t), so
            # produced == #completed-groups -> the tail's `produced > g` gate is exact.
            if const_expr(tail_dtma):
                if do_full:
                    if lane == Int32(0):
                        cute.arch.atomic_add(produced_ptr, Int32(1), sem="release", scope="cta")
            if lane == Int32(0):
                pcount[0] = t + Int32(1)
            cute.arch.sync_warp()

        return copy_fn

    def _a2a_cluster_producer_copy_fn(
        self, epi_params, tile_shape_mn, epi_tile, sD, tile_coord_mnkl, storage
    ):
        """CLUSTER-DRAIN (3.1b) PRODUCER — the cluster-cooperative variant of the producer_tma write.

        Each cluster-rank r's MMA warpgroup TMA-S2G's its tile's (epi_m, epi_n) epilogue subtiles into
        the SHARED per-cluster staging ``(n_clusters, epi_m, N_j_loc)`` at column
        ``col = tile_in_j*tile_n + sub_n*epi_n`` (``tile_in_j = tile_coord_N % nt_j_pp``), so the
        ``cluster_n`` CTAs of one cluster tile up the full peer_j width ``N_j_loc``. After it has staged
        its whole RUN (``run_j = ceil(nt_j_pp/cluster_n)`` tiles == ``grp_sz//n_sub`` subtiles) it
        SIGNALS cluster-rank-0 that its column-slice is ready; cluster-rank-0's drain (``_cluster_drain_
        loop``) then reads the full ``(epi_m, N_j_loc)`` and fires the coalesced put.

        SIGNAL (Phase-3.1b, cross-CTA mbarrier de-risk probe #2 / task #42 — WIRED, not a stub). The
        cross-CTA "full" arrive is the real handshake: ``fence_acq_rel_gpu`` (:2708) then lane-0
        ``mbarrier_arrive(full, peer_cta_rank_in_cluster=0)`` (:2710-2716, count == cluster_n); the
        "empty" reuse-gate wait is real (:2652-2659). meta is written at first_in_run (:2685-2698)
        BEFORE the fence+full-arrive. Ran correctness-clean at K=256 (Gate-C); K=N validation IN
        PROGRESS (task #51). Default-off; no other path reaches here."""
        cp: cutlass.Constexpr[int] = self._a2a_cp
        # The batch extent must NOT be a compile-time constant: `B` reaches this kernel ONLY as the
        # divisor of the GEMM batch coord (`d = L // B`, `b = L % B`), so baking it made one
        # compiled kernel serve one batch extent for no design reason. It is READ instead: mode 4
        # of the permuted peer recv (N_i_loc, N_j_loc, cp, Dloc, B) IS `B`. Runtime `Int32` wherever
        # the caller marked the recv dynamic (the production `TriMulAutotuned` path), a folded
        # Python int on a per-shape static compile -- the same split `N_loc_rt` has at :3903, and
        # why there is no branch here. There is deliberately NO static-B mode.
        B = epi_params.peer_tensors[0].shape[4]
        cp_axis_sizes = self._a2a_cp_axis_sizes
        cp1: cutlass.Constexpr[int] = cp_axis_sizes[1] if len(cp_axis_sizes) > 1 else 1
        cp0: cutlass.Constexpr[int] = cp_axis_sizes[0] if len(cp_axis_sizes) > 1 else cp
        cluster_n: cutlass.Constexpr[int] = int(self._a2a_cluster_n)
        # (d) MULTISLOT: per-tile peer-routing into 2 rotating full-peer slots (slot=peer_j&1), real-
        # arrive-on-peer-cross signaling (NO null-arrive / NO data_present). Default off -> the even-shard
        # single/db bounded-band path traces byte-identical.
        multislot: cutlass.Constexpr[bool] = bool(getattr(self, "_a2a_cluster_multislot", False))
        # (d) FIX (round-barrier / option a): SERIALIZE the cross-CTA producer/drain so a producer CTA's
        # TMA-S2G never runs concurrently with rank-0's cross-node IBGDA RDMA-read of the sibling slot (the
        # cross-CTA rd=2 double-buffer overlap FAULTS on IB — [[reference_dbuf_overlap_crossnode_ibgda_fault]]).
        # At each peer-cross, AFTER arriving full[prev&1], wait empty[prev&1] for the PREVIOUS peer's drain to
        # complete before staging the next peer. Removes the write-while-RDMA-read; KEEPS both slots (straddle).
        # Loses the cross-round producer/drain overlap (perf-neutral vs the shipped single-buffer baseline; the
        # double-buffer overlap is #45's separate, cross-node-broken perf-opt). Default ON for multislot;
        # CPO_MS_SERIAL=0 reproduces the fault (A/B). Mirrors the reuse-gate phase (safe, no new mbar).
        _serial: cutlass.Constexpr[bool] = multislot and os.environ.get("CPO_MS_SERIAL", "1") == "1"
        tile_m: cutlass.Constexpr[int] = tile_shape_mn[0]
        tile_n: cutlass.Constexpr[int] = tile_shape_mn[1]
        epi_m: cutlass.Constexpr[int] = epi_tile[0]
        epi_n: cutlass.Constexpr[int] = epi_tile[1]
        n_sub_per_tile: cutlass.Constexpr[int] = tile_n // epi_n
        # #78 MULTI-EPI-SUBTILE: epi M-subtiles per CTA tile (== 1 unless tile_m>epi_m). Even-shard SB
        # stages each sub_m band into the m_sub axis of the (n_clusters, m_sub, epi_m, N_j_loc) staging.
        m_sub_per_tile: cutlass.Constexpr[int] = tile_m // epi_m
        rd: cutlass.Constexpr[int] = self._a2a_ring_depth
        nfields: cutlass.Constexpr[int] = self._DECOUPLED_META_FIELDS
        # run_j (cluster-tile units, STATIC — baked at the compile anchor in 3.1a; dynamic-N bounded
        # band is 3.1c) == the number of tiles this CTA stages per RUN; grp_sz == its subtiles/run.
        # MATCHES the scheduler's baked run_j_tiles (get_scheduler_arguments), so the run boundary the
        # producer signals lines up with the run the drain reads.
        run_j: cutlass.Constexpr[int] = int(self._a2a_cluster_run_j)
        # #78: a RUN spans run_j j-tiles; each tile emits m_sub_per_tile*n_sub_per_tile epi-subtile copy_fn
        # calls (N-major order), so grp_sz counts them -> first_in_run / last_in_run land on TILE
        # boundaries. m_sub_per_tile==1 -> run_j*n_sub_per_tile (byte-identical to the shipped path).
        grp_sz: cutlass.Constexpr[int] = run_j * m_sub_per_tile * n_sub_per_tile
        # peer recv (N_i_loc, N_j_loc, cp, Dloc, B); nt_{i,j}_pp = ceil(N_{i,j}_loc/tile_{m,n}) RUNTIME
        # (cluster_drain 2-D requires dynamic) -> the per-tile column / gi decode below.
        peer0 = epi_params.peer_tensors[0]
        nt_j_pp_rt = (peer0.shape[1] + tile_n - 1) // tile_n
        nt_i_pp_rt = (peer0.shape[0] + tile_m - 1) // tile_m

        full_ptr = storage.decoupled.full.data_ptr()
        empty_ptr = storage.decoupled.empty.data_ptr()
        meta = storage.decoupled.meta.get_tensor(
            (rd * nfields,)
        )  # slot 0 (single synchronous buffer)
        pcount = storage.decoupled.pcount.get_tensor((1,))
        # (d) MULTISLOT walk state next_slot: [0]=prev_peer (last REAL peer_j staged; -1 == none/padding),
        # [1]=fill[slot0], [2]=fill[slot1] (per-slot fill counts -> empty[slot] reuse phase); AT cp1==1 (1-D
        # FIX) also [3]=prev_m0, [4]=prev_L (band/plane-change detector -- see _a2a_walk_state_len).
        ns_n: cutlass.Constexpr[int] = self._a2a_walk_state_len()
        walk = storage.decoupled.next_slot.get_tensor((ns_n,))

        stage_atom = epi_params.cluster_stage_atom
        # (epi_m, N_j_loc, n_clusters) -> flat_divide(epi_tile) -> (em, en, 1, N_j_loc/en, n_clusters).
        gStage = cute.flat_divide(epi_params.cluster_stage_tensor, epi_tile)
        s_stage, g_stage = cpasync.tma_partition(
            stage_atom,
            0,
            cute.make_layout(1),
            cute.group_modes(sD, 0, cute.rank(sD) - 1),
            cute.group_modes(gStage, 0, 2),
        )
        # PHYSICAL cluster id: grid is (1, cluster_n, n_clusters) so block_idx()[0]+[2] == the cluster's
        # z-slot (shared by all cluster_n ranks), block_idx_in_cluster() == the rank in [0, cluster_n).
        bidx = cute.arch.block_idx()[0] + cute.arch.block_idx()[2]

        @cute.jit
        def copy_fn(src_idx, dst_idx, **kwargs):
            sub_m, sub_n = dst_idx[0], dst_idx[1]
            L = tile_coord_mnkl[3]
            d = L // Int32(B)
            b = L % Int32(B)
            # 2-D pe_aligned: tile_coord_N in [0, cp1*nt_j_pp). peer_j = //nt_j_pp, tile_in_j = %nt_j_pp.
            # Int32-wrap the runtime nt_*_pp_rt at use (matches the coalesce producer's Int32(tiles_per_*)).
            peer_j = tile_coord_mnkl[1] // Int32(nt_j_pp_rt)
            tile_in_j = tile_coord_mnkl[1] % Int32(nt_j_pp_rt)
            # global token-i base of this epi-box (pe_aligned per-peer M-tiling) == the coalesce
            # producer's gi_base: peer_i*N_i_loc + k*tile_m + sub_m*epi_m (N_i_loc == peer0.shape[0]).
            peer_i = tile_coord_mnkl[0] // Int32(nt_i_pp_rt)
            k_i = tile_coord_mnkl[0] % Int32(nt_i_pp_rt)
            gi_base = peer_i * Int32(peer0.shape[0]) + k_i * Int32(tile_m) + sub_m * Int32(epi_m)
            # j_within = the epi_n-subtile column index into the (epi_m, N_j_loc) staging.
            j_within = tile_in_j * Int32(n_sub_per_tile) + sub_n

            lane = cute.arch.lane_idx()
            if const_expr(multislot):
                # ===== #76 (d) MULTISLOT per-tile peer-routing (real-arrive-on-cross; NO null-arrive) =====
                # This CTA's REAL (un-clustered) peer_j -> its rotating full-peer slot = peer_j&1. is_real
                # drops the FULL-BAND walk's ceil-spill PADDING tiles (peer_j >= cp1; the scheduler clamps
                # them to a benign double-store -> we skip stage+signal). The CAP cluster_n<=nt_j_pp makes
                # each CTA cross each peer exactly once (peers complete in order -> slots alternate), so
                # full[peer&1] gets exactly cluster_n real arrives per peer -> no null-arrive, no data_present.
                slot = peer_j % Int32(2)
                is_real = peer_j < Int32(cp1)
                prev_peer = walk[Int32(0)]
                entered_new = (
                    peer_j != prev_peer
                )  # crossed into a new peer segment (incl. band-reset)
                # ==== 1-D FIX (CONFIRMED by the cp16 1-D gate: cn1-sb / cn1-multislot / cn2-sb /
                # cn2-multislot ALL PASS; cn2-multislot rel_L2=3.27e-05, 0 outlier, 0 untouched, 16 ranks).
                # memcheck was inconclusive by INSTRUMENTATION (the OOB manifests in the nvshmem device-put
                # RDMA target, which compute-sanitizer does not instrument) -> the green gate is the
                # confirmation of record for this root cause. ================================================
                # At cp1==1 peer_j is ALWAYS 0, so `peer_j != prev_peer` NEVER fires across BANDS (the i-base
                # / L-plane change that starts the next drain-band keeps peer_j 0->0). Detect the new
                # drain-band DIRECTLY: a change in the M-tile coord (tile_coord_mnkl[0], the i-base) OR the
                # L-plane (tile_coord_mnkl[3], the (d,b) plane) IS a fresh slot-fill unit == the per-band
                # full-arrive / reuse-gate / meta the drain expects. Without it, a physical cluster that
                # processes >1 band (total_bands > n_clusters == the cn2 regime) skips the per-band handshake
                # -> the drain's 2nd band reads stale meta -> OOB put (the CUDA_ERROR_LAUNCH_FAILURE). 2-D
                # (cp1>1) does NOT trace this branch: within a band tile_coord_mnkl[0]/[3] are CONSTANT while
                # peer_j cycles, and at the 2-D band boundary peer_j already changes -> the OR is a no-op ->
                # BYTE-IDENTICAL. `entered_new | new_band` stays a Name fed to `if` (the safe dynamic-if form).
                if const_expr(cp1 == 1):
                    new_band = (tile_coord_mnkl[0] != walk[Int32(3)]) | (
                        tile_coord_mnkl[3] != walk[Int32(4)]
                    )
                    entered_new = entered_new | new_band
                if entered_new:
                    # (1) COMPLETE the PREVIOUS real peer: fence its (prior-round) staged slice, then arrive
                    #     rank-0's full[prev&1] cross-CTA (one lane -> count == cluster_n). prev == -1 (init
                    #     or after a padding cross) -> nothing to complete.
                    prev_real = (prev_peer >= Int32(0)) & (prev_peer < Int32(cp1))
                    if prev_real:
                        cute.arch.fence_acq_rel_gpu()
                        cute.arch.sync_warp()
                        if lane == Int32(0):
                            for ss in cutlass.range_constexpr(rd):
                                if (prev_peer % Int32(2)) == Int32(ss):
                                    cute.arch.mbarrier_arrive(
                                        full_ptr + ss, peer_cta_rank_in_cluster=0
                                    )
                        # (d) SERIALIZE (option a): wait for the PREVIOUS peer's cross-node RDMA-read to
                        # COMPLETE (rank-0 arrives empty[prev&1] after its blocking put) BEFORE this CTA stages
                        # the next peer -> no producer-TMA-S2G ∥ NIC-RDMA-read on the shared staging. gp =
                        # slot prev&1's fill count (>=1: prev is real+staged); wait @(gp-1)&1 mirrors the
                        # reuse-gate = the gp-th drain of that slot (prev's drain). Warp-wide (all lanes).
                        if const_expr(_serial):
                            gp = walk[Int32(1) + (prev_peer % Int32(2))]
                            if gp >= Int32(1):
                                for ss in cutlass.range_constexpr(rd):
                                    if (prev_peer % Int32(2)) == Int32(ss):
                                        cute.arch.mbarrier_wait(
                                            empty_ptr + ss, (gp - Int32(1)) & Int32(1)
                                        )
                    if is_real:
                        # (2) REUSE-GATE: before overwriting `slot` for this NEW real peer, wait its prior
                        #     occupant (peer-2) drained. fill[slot] == #times this CTA already filled the
                        #     slot -> wait empty[slot] @ (fill-1)&1 (skip the slot's first fill). Mirrors the
                        #     even-shard `if kk>=1: wait empty[(kk-1)&1]`.
                        f = walk[Int32(1) + slot]
                        if f >= Int32(1):
                            for ss in cutlass.range_constexpr(rd):
                                if slot == Int32(ss):
                                    cute.arch.mbarrier_wait(
                                        empty_ptr + ss, (f - Int32(1)) & Int32(1)
                                    )
                        # (3) per-peer meta (peer_j, gi_base, d, b) + advance walk state (lane 0). Written
                        #     ONCE per peer (at the cross) -> stable until drained (the reuse-gate on the
                        #     NEXT peer-on-this-slot is what frees it). cluster-rank-0's copy is drained.
                        if lane == Int32(0):
                            mbase = slot * Int32(nfields)
                            meta[mbase + Int32(0)] = peer_j
                            meta[mbase + Int32(3)] = d
                            meta[mbase + Int32(4)] = b
                            meta[mbase + Int32(5)] = gi_base
                            walk[Int32(1) + slot] = f + Int32(1)
                            walk[Int32(0)] = peer_j  # prev_peer <- this real peer
                            if const_expr(cp1 == 1):
                                # 1-D FIX: record THIS band's identity (i-base + L-plane) so the NEXT tile's
                                # band-change detector fires only when the band actually advances, not every
                                # subtile of the same band. lane-0 only, mirroring the other walk writes.
                                walk[Int32(3)] = tile_coord_mnkl[0]
                                walk[Int32(4)] = tile_coord_mnkl[3]
                    else:
                        # crossed INTO a padding peer: prev completed above; mark prev INVALID so the tail
                        # flush does not re-arrive it (padding is not staged / not a drain peer).
                        if lane == Int32(0):
                            walk[Int32(0)] = Int32(-1)
                    cute.arch.sync_warp()
                # (4) STAGE this subtile into slot=peer_j&1 (every REAL subtile; padding skipped). Runtime
                #     slot index into the rd-slot staging (== the db path's g_stage indexing). Reuse-gate (2)
                #     already cleared the slot for a NEW peer; same-peer later subtiles reuse the clean slot.
                if is_real:
                    # (d) PRODUCER PROXY FENCE — FIX-1 : the TMA-S2G rides the ASYNC proxy, a
                    # separate pipe from the generic-proxy REUSE-GATE `empty[slot]` acquire above (:2714, which
                    # waits for rank-0's NIC RDMA-read of THIS slot to COMPLETE before re-staging it). Without
                    # an async-proxy fence the async write can LAUNCH before that generic acquire is observed
                    # -> the async store races the NIC read of the SAME slot -> the same-slot reuse hazard
                    # (CUDA_ERROR_LAUNCH_FAILED). fence_proxy("async.global") orders this CTA's async-proxy
                    # write AFTER the generic-proxy acquire, making the per-slot reuse-gate EFFECTIVE.
                    # ---- THE BUG (was `if const_expr(_serial):`) ----  the reuse-gate (:2714) ALWAYS runs,
                    # so its acquire MUST always be enforced against the write -> the fence MUST be
                    # UNCONDITIONAL. It was _serial-gated, so at _serial=0 (the double-buffer perf mode) the
                    # reuse-gate acquire was NOT ordered before the async write -> same-slot overlap -> the ULF,
                    # a RACE reliably exposed at cluster_n>=4 (3+ concurrent producer CTAs). Now unconditional:
                    # BYTE-IDENTICAL at _serial=1 (the fence already ran there); FIXES the _serial=0 double-
                    # buffer. Different-slot overlap stays legal (the fence only orders THIS CTA's write after
                    # THIS slot's reuse-gate acquire; it does not serialize distinct slots) -> perf preserved.
                    cute.arch.fence_proxy(kind="async.global")
                    with cute.arch.elect_one():
                        cute.copy(
                            stage_atom,
                            s_stage[(None, src_idx)],
                            g_stage[(None, Int32(0), j_within, slot, bidx)],
                        )
                        cute.arch.cp_async_bulk_commit_group()
                        cute.arch.cp_async_bulk_wait_group(0, read=False)
                    cute.arch.sync_warp()
            else:
                t = pcount[0]
                g = t // Int32(grp_sz)  # run index (per CTA)
                j_in_run = t % Int32(grp_sz)
                first_in_run = j_in_run == Int32(0)
                last_in_run = j_in_run == Int32(grp_sz - 1)
                slot = Int32(0)  # single synchronous buffer -> slot 0
                kk = g

                # CLUSTER-DRAIN reuse-gate : before this CTA overwrites slot `slot` for a NEW run
                # (first_in_run, kk>=1), WAIT its 'empty' mbar -- cluster-rank-0 arrives it cross-CTA after
                # draining slot's PRIOR run (run g-rd), so the producer never clobbers a slice rank-0 is
                # still put'ing from. Plain-arrive mbar -> blocking mbarrier_wait is fine. Phase (kk-1)&1.
                # db: rotating empty[slot]; SINGLE-BUFFER: empty[0] (kk==g -> byte-identical).
                if first_in_run:
                    if kk >= Int32(1):
                        cute.arch.mbarrier_wait(empty_ptr, (kk - Int32(1)) & Int32(1))

                # WRITE this subtile -> the SHARED per-cluster staging at (j_within, [slot/sub_m,] cluster=bidx).
                with cute.arch.elect_one():
                    if const_expr(m_sub_per_tile > 1):
                        # #78: (n_clusters, m_sub, epi_m, N_j_loc) staging -> index the epi M-subtile sub_m.
                        cute.copy(
                            stage_atom,
                            s_stage[(None, src_idx)],
                            g_stage[(None, Int32(0), j_within, Int32(sub_m), bidx)],
                        )
                    else:
                        cute.copy(
                            stage_atom,
                            s_stage[(None, src_idx)],
                            g_stage[(None, Int32(0), j_within, bidx)],
                        )
                    cute.arch.cp_async_bulk_commit_group()
                    # WAIT the S2G to LAND (read=False): rank-0 reads the staging from GMEM, so the bulk
                    # store must be COMPLETE before the "full" signal. Cheap LOCAL-HBM completion, NOT the
                    # NVLink stall.
                    cute.arch.cp_async_bulk_wait_group(0, read=False)
                cute.arch.sync_warp()

                # per-run meta (written ONCE at first_in_run by lane 0): the drain routes by these. All
                # cluster_n ranks agree (same i-band + peer_j + L), so each writes its OWN SMEM copy;
                # cluster-rank-0's copy is the one its drain reads.
                if lane == Int32(0):
                    if first_in_run:
                        meta[Int32(0)] = peer_j
                        meta[Int32(3)] = d
                        meta[Int32(4)] = b
                        meta[Int32(5)] = gi_base
                cute.arch.sync_warp()

                # CLUSTER-DRAIN full-signal : once this CTA has staged its WHOLE run
                # (last_in_run), RELEASE its GMEM staging (+ the SMEM meta) then signal cluster-rank-0's
                # 'full' mbar. The bulk store already LANDED (cp_async_bulk_wait_group(0) above) and the warp
                # is sync'd, so fence_acq_rel_gpu (the proven fence, GPU scope) orders the staging + meta
                # before the arrive; the sync_warp orders every lane's fence before lane-0's arrive; ONE
                # thread (lane 0) arrives rank-0's mbar (peer_cta_rank_in_cluster=0) so the count ==
                # cluster_n (one arrive per cluster CTA). Replaces the ring full-signal.
                if last_in_run:
                    cute.arch.fence_acq_rel_gpu()
                    cute.arch.sync_warp()
                    if lane == Int32(0):
                        cute.arch.mbarrier_arrive(full_ptr, peer_cta_rank_in_cluster=0)

                if lane == Int32(0):
                    pcount[0] = t + Int32(1)
                cute.arch.sync_warp()

        return copy_fn

    def _build_ring_store_atom(self, ring, epi_smem_layout_staged, epi_tile):
        """Build the producer-TMA S2G atom SMEM->local-GMEM ring (I2b producer refinement).

        ``ring`` is the caller's local-GMEM staging tensor ``(grid_CTAs, ring_depth, epi_m, epi_n)``.
        We permute the box modes (epi_m, epi_n) FIRST -> ``(epi_m, epi_n, ring_depth, grid_CTAs)`` (the
        design-E recipe; epi_n stays stride-1) and build a ``CopyBulkTensorTileS2GOp`` atom whose
        descriptor covers the whole ring; the kernel selects the live ``(slot, cta)`` per store. This
        is the SAME TMA store the single-device GEMM uses, just pointed at the ring -> the MMA warp
        fires one bulk descriptor (O(1)) instead of an O(tile) SIMT copy. Returns ``(atom, tensor)``;
        ``tensor`` is the box-permuted ring view the copy_fn ``flat_divide``s + indexes.

        2-D (Task #13 option D — coalesce only): the ring is peer_j-MAJOR
        ``(grid_CTAs, ring_depth, cp1, epi_m, N_j_loc)``; permute the box modes ``(epi_m, N_j_loc)``
        FIRST -> ``(epi_m, N_j_loc, cp1, ring_depth, grid_CTAs)`` (a 5-D box-permute), so the producer
        indexes ``(box, i_tile, j_within, peer_j, slot, cta)`` and the
        ``N_j_loc`` descriptor extent clamps each peer_j's ceil-spill. The 2-D DIFFERENTIAL ring stays 4-D
        ``(grid, rd, epi_m, tile_n)`` (only coalesce relayouts the band) -> gated on ``cp1>1 AND coalesce``."""
        epi_smem_layout = cute.slice_(epi_smem_layout_staged, (None, None, 0))
        # (grid_CTAs, ring_depth, epi_m, epi_n) -> (epi_m, epi_n, ring_depth, grid_CTAs).
        ring_view = cute.make_tensor(ring.iterator, cute.select(ring.layout, mode=[2, 3, 1, 0]))
        d_cta_v_layout = cute.composition(cute.make_identity_layout(ring_view.shape), epi_tile)
        op = cpasync.CopyBulkTensorTileS2GOp()
        atom, tensor = cpasync.make_tiled_tma_atom(op, ring_view, epi_smem_layout, d_cta_v_layout)
        return atom, tensor

    def _build_ring_load_atom(self, ring, epi_smem_layout_staged, epi_tile):
        """Build the A′ consumer G2S atom GMEM-ring -> SMEM bounce (same box-permuted ring view as the
        store atom, but a ``CopyBulkTensorTileG2SOp``)."""
        epi_smem_layout = cute.slice_(epi_smem_layout_staged, (None, None, 0))
        ring_view = cute.make_tensor(
            ring.iterator,
            cute.select(ring.layout, mode=[2, 3, 1, 0]),  # (epi_m, epi_n, rd, grid)
        )
        d_cta_v_layout = cute.composition(cute.make_identity_layout(ring_view.shape), epi_tile)
        op = cpasync.CopyBulkTensorTileG2SOp()
        atom, tensor = cpasync.make_tiled_tma_atom(op, ring_view, epi_smem_layout, d_cta_v_layout)
        return atom, tensor

    def _build_cluster_stage_atom(self, cluster_stage, epi_smem_layout_staged, epi_tile):
        """CLUSTER-DRAIN (3.1b): producer TMA-S2G atom SMEM epi-box -> the per-cluster staging buffer.

        ``cluster_stage`` is the caller's SYMMETRIC-heap staging tensor ``(n_clusters, epi_m, N_j_loc)``
        (n_clusters == max persistent clusters == grid_ctas//cluster_n). We permute the box modes
        (epi_m, N_j_loc) FIRST -> ``(epi_m, N_j_loc, n_clusters)`` (N_j_loc stays stride-1, the design-E
        recipe) and build a ``CopyBulkTensorTileS2GOp`` atom whose descriptor covers the whole staging;
        the kernel selects the live ``(j_within, cluster_idx)`` per store (cluster-rank r writes its
        ``tile_in_j`` tile_n-wide column-slice of the shared ``(epi_m, N_j_loc)`` cluster region). The
        ``N_j_loc`` descriptor extent clamps the per-peer_j ceil-spill exactly like the coalesce ring.
        Mirrors :meth:`_build_ring_store_atom`; returns ``(atom, box-permuted tensor)``."""
        epi_smem_layout = cute.slice_(epi_smem_layout_staged, (None, None, 0))
        # #78 MULTI-EPI-SUBTILE: tile_m>epi_m stages m_sub bands into a (n_clusters, m_sub, epi_m, N_j_loc)
        # staging -> the SAME 4-D box-permute as db/multislot (the m_sub axis plays the db slot's role).
        _m_sub = int(self.cta_tile_shape_mnk[0]) // int(epi_tile[0])
        # (d) MULTISLOT reuses the db rd-slot staging layout (n_clusters, rd, epi_m, N_j_loc): the rd=2
        # slots ARE the 2 rotating full-peer slots (slot=peer_j&1). Same 4-D box-permute as db.
        if bool(getattr(self, "_a2a_cluster_multislot", False)) or _m_sub > 1:
            # DOUBLE-BUFFER (Phase-5 spike): (n_clusters, rd, epi_m, N_j_loc) -> (epi_m, N_j_loc, rd,
            # n_clusters) -- the SAME 4-D box-permute as _build_ring_store_atom (mode=[2,3,1,0]); the
            # producer indexes (box, i_tile, j_within, slot, cluster) and the drain reads [cluster, slot].
            # #78: for the m_sub staging the 2nd axis is sub_m (not rd) -- same permute, same indexing.
            stage_view = cute.make_tensor(
                cluster_stage.iterator, cute.select(cluster_stage.layout, mode=[2, 3, 1, 0])
            )
        else:
            # SINGLE-BUFFER: (n_clusters, epi_m, N_j_loc) -> (epi_m, N_j_loc, n_clusters).
            stage_view = cute.make_tensor(
                cluster_stage.iterator, cute.select(cluster_stage.layout, mode=[1, 2, 0])
            )
        d_cta_v_layout = cute.composition(cute.make_identity_layout(stage_view.shape), epi_tile)
        op = cpasync.CopyBulkTensorTileS2GOp()
        atom, tensor = cpasync.make_tiled_tma_atom(op, stage_view, epi_smem_layout, d_cta_v_layout)
        return atom, tensor

    @cute.jit
    def _pe_tiled_a_row_shift(self, m_linear: Int32) -> Int32:
        """PE-boundary-aware per-peer M-tiling: row shift for A's TMA-load at per-peer tile m_linear.

        m_linear in [0, cp*nt_pp): peer = m_linear // nt_pp, k = m_linear % nt_pp. The desired A-row
        origin is base_m = peer*N_loc + k*cta_tile_M; the natural local_tile(.,(m_linear,None)) reads
        from m_linear*cta_tile_M, so the shift is base_m - m_linear*cta_tile_M = peer*(N_loc -
        nt_pp*cta_tile_M) (<= 0, since nt_pp*cta_tile_M >= N_loc). Same formula the store base_m uses.
        """
        nt_pp = cutlass.const_expr(self._a2a_nt_pp)
        tile_m = cutlass.const_expr(self.cta_tile_shape_mnk[0])
        N_loc = cutlass.const_expr(self._a2a_N_loc)
        peer = m_linear // Int32(nt_pp)
        return peer * Int32(N_loc - nt_pp * tile_m)

    def _pe_tiled_base_m(self, m_linear: Int32) -> Int32:
        """Per-peer base_m = peer*N_loc + k*cta_tile_M for per-peer tile index m_linear (store side)."""
        nt_pp = cutlass.const_expr(self._a2a_nt_pp)
        tile_m = cutlass.const_expr(self.cta_tile_shape_mnk[0])
        N_loc = cutlass.const_expr(self._a2a_N_loc)
        peer = m_linear // Int32(nt_pp)
        k = m_linear % Int32(nt_pp)
        return peer * Int32(N_loc) + k * Int32(tile_m)

    def get_scheduler_arguments(self, mA, mB, mD, scheduler_args, epilogue_args):
        """CHILD override: PE-boundary-aware per-peer M-tiling overrides ONLY the M-tile COUNT
        (cp*nt_pp vs the uniform ceil(M/tile_m)) so no output tile straddles a peer. Calls the
        parent (byte-identical) and dataclasses.replace's problem_shape_ntile_mnl when the flag is
        on; flag-off returns the parent args unchanged. Parent gemm_sm90 untouched (§3.1)."""
        args = super().get_scheduler_arguments(mA, mB, mD, scheduler_args, epilogue_args)
        if const_expr(getattr(self, "_pe_aligned_tiling", False)):
            import dataclasses

            ps = args.problem_shape_ntile_mnl
            if const_expr(getattr(self, "_a2a_arbitrary_n", False)):
                # A1 (1-D gemm_native pe_aligned, arbitrary_n copy_fn): M-tile count -> cp*nt_pp (each
                # peer's N_loc rows tiled independently). N/L unchanged.
                # dynamic-shape: un-bake nt_pp -> ceil(N_loc/tile_m) from the RUNTIME M (=cp*N_loc,
                # mA.shape[0]); the persistent STATIC scheduler caps the grid at max_active_clusters so a
                # runtime total-tile count is fine (same as the base dynamic GEMM's mA.shape[0] path).
                # const_expr fast-path for the static case keeps flag-off byte-identical (perf §★ #1).
                # 2-D ib_ring (Task #13): cp1>1 takes the A1-2D branch (per-peer grid on BOTH axes,
                # mirroring the A2/A3 non-arb branch); cp1==1 keeps the 1-D code below verbatim.
                _cp1_sched: cutlass.Constexpr[int] = int(
                    self._a2a_cp_axis_sizes[1] if len(self._a2a_cp_axis_sizes) > 1 else 1
                )
                if const_expr(_cp1_sched > 1):
                    # A1-2D: M-tiles = cp0*nt_i_pp, N-tiles = cp1*nt_j_pp so the producer copy_fn decodes
                    # peer_i = m//nt_i_pp, peer_j = n//nt_j_pp and the coalesce_dyn drain routes peer =
                    # peer_i*cp1 + peer_j. dynamic un-bakes nti/ntj from
                    # the RUNTIME M (mA.shape[0]) / N (mB.shape[0]).
                    _cp0_sched: cutlass.Constexpr[int] = int(self._a2a_cp_axis_sizes[0])
                    if const_expr(getattr(self, "_a2a_dynamic", False)):
                        tile_m_pe = cutlass.const_expr(self.cta_tile_shape_mnk[0])
                        tile_n_pe = cutlass.const_expr(self.cta_tile_shape_mnk[1])
                        N_i_rt = Int32(mA.shape[0]) // Int32(_cp0_sched)
                        N_j_rt = Int32(mB.shape[0]) // Int32(_cp1_sched)
                        nti = (N_i_rt + Int32(tile_m_pe - 1)) // Int32(tile_m_pe)
                        ntj = (N_j_rt + Int32(tile_n_pe - 1)) // Int32(tile_n_pe)
                        new_ps = (Int32(_cp0_sched) * nti, Int32(_cp1_sched) * ntj, ps[2])
                    else:
                        new_ps = (
                            Int32(_cp0_sched * self._a2a_nt_i_pp),
                            Int32(_cp1_sched * self._a2a_nt_j_pp),
                            ps[2],
                        )
                elif const_expr(getattr(self, "_a2a_dynamic", False)):
                    tile_m_pe = cutlass.const_expr(self.cta_tile_shape_mnk[0])
                    cp_pe = cutlass.const_expr(self._a2a_cp)
                    # Plain Int32 arithmetic (cute.ceil_div on a scalar Int32 emits an illegal derefine).
                    N_loc_rt = Int32(mA.shape[0]) // Int32(cp_pe)
                    nt_pp_rt = (N_loc_rt + Int32(tile_m_pe - 1)) // Int32(tile_m_pe)
                    new_ps = (Int32(cp_pe) * nt_pp_rt, ps[1], ps[2])
                else:
                    # STATIC pe_aligned arbitrary_n: emit the PER-PEER tiling cp*nt_pp (each peer's N_loc
                    # rows tiled independently); N/L unchanged. (Sub-one-tile M<=tile_m emits phantom extra
                    # tiles -- a pre-existing task#4 limitation, unchanged here.)
                    new_ps = (Int32(self._a2a_cp * self._a2a_nt_pp), ps[1], ps[2])
            else:
                # A2 (2-D sharded pe_aligned, non-arb copy_fn): per-peer grid on BOTH axes ->
                # M-tiles = cp0*nt_i_pp, N-tiles = cp1*nt_j_pp. 1-D (cp1==1) -> N-tiles = ceil(N/tile_n)
                # (== the uniform count) so ps[1] is unchanged -> byte-identical. A3 (dynamic) below.
                cp_axis = self._a2a_cp_axis_sizes
                cp0 = cutlass.const_expr(cp_axis[0])
                cp1 = cutlass.const_expr(cp_axis[1] if len(cp_axis) > 1 else 1)
                if const_expr(getattr(self, "_a2a_dynamic", False)):
                    tile_m_pe = cutlass.const_expr(self.cta_tile_shape_mnk[0])
                    tile_n_pe = cutlass.const_expr(self.cta_tile_shape_mnk[1])
                    N_i_rt = Int32(mA.shape[0]) // Int32(cp0)
                    N_j_rt = Int32(mB.shape[0]) // Int32(cp1)
                    nti = (N_i_rt + Int32(tile_m_pe - 1)) // Int32(tile_m_pe)
                    ntj = (N_j_rt + Int32(tile_n_pe - 1)) // Int32(tile_n_pe)
                    new_ps = (Int32(cp0) * nti, Int32(cp1) * ntj, ps[2])
                else:
                    new_ps = (Int32(cp0 * self._a2a_nt_i_pp), Int32(cp1 * self._a2a_nt_j_pp), ps[2])
            args = dataclasses.replace(args, problem_shape_ntile_mnl=new_ps)
        # CLUSTER-DRAIN (Phase-3.1a): inject the STATIC bounded per-peer_j band (CLUSTER-tile units, baked
        # in configure). run_j_tiles | ncluster_n (=cp1*run_j) -> runs_per_band=cp1, the divisor the
        # sub-band decode requires. Default off -> const_expr-elided -> byte-identical. (Dynamic-N bounded
        # band with runtime runs_per_band=cp1 is 3.1c; here N is the static compile anchor.)
        if const_expr(getattr(self, "_a2a_cluster_drain", False)):
            import dataclasses

            # (d) MULTISLOT: FULL-BAND cluster walk (run_j_dynamic, runs_per_band=1) instead of the
            # even-shard BOUNDED band. Each cluster then owns a WHOLE (plane, i-band) N-walk (all cp1
            # peer_j in j-order), so every peer_j is SELF-CONTAINED in one cluster's staging -> the 2
            # rotating full-peer slots can assemble it. The bounded band (runs_per_band=cp1) would SPLIT a
            # straddling peer_j across different physical clusters (run_global=z+run_local*gz -> two halves
            # land on z, z+1). Mirrors the coalesce_dyn dynamic scheduler; the cluster_n CTAs cooperate per
            # ROUND (one j-cluster = cluster_n consecutive N-tiles). Default (non-multislot) keeps the baked
            # bounded run_j_tiles -> byte-identical.
            if const_expr(getattr(self, "_a2a_cluster_multislot", False)):
                args = dataclasses.replace(args, run_j_dynamic=True)
            else:
                args = dataclasses.replace(
                    args, run_j_tiles=int(getattr(self, "_a2a_cluster_run_j", 0))
                )
        return args

    def mainloop_remap_mA(self, mA_mk, tile_coord_mnkl, mA_mkl=None, batch_idx=None):
        """CHILD override: shift A's row origin for per-peer M-tiling so the mainloop TMA-load reads
        from base_m = peer*N_loc + k*tile_m (tile_coord_mnkl[0] is the per-peer linear tile index).
        domain_offset is a pointer shift; the partial-last per-peer tile reads into the next peer's
        A rows (or zero-fill past M via the TMA descriptor) -> double-compute, MASKED by the store's
        recv-descriptor clamp. Flag-off -> parent pass-through (byte-identical)."""
        if const_expr(getattr(self, "_pe_aligned_tiling", False)):
            tile_m_pe = cutlass.const_expr(self.cta_tile_shape_mnk[0])
            if const_expr(getattr(self, "_a2a_arbitrary_n", False)):
                # A1 (gemm_native ib_ring): a_shift = peer_i*(N_i_loc - nt_i_pp*tile_m). Static keeps the
                # const_expr helper (nt_pp==nt_i_pp, N_loc==N_i_loc -> correct for BOTH 1-D and 2-D;
                # byte-identical, perf §★ #1). Dynamic derives N_i_loc from M//cp0 -- Task #13: cp0 (NOT
                # cp) so the 2-D i-axis extent is N//cp0=N_i_loc, not N//(cp0*cp1). 1-D: cp0==cp -> same.
                if const_expr(getattr(self, "_a2a_dynamic", False)):
                    cp0_pe = cutlass.const_expr(self._a2a_cp0)
                    N_loc_rt = Int32(mA_mk.shape[0]) // Int32(cp0_pe)
                    nt_pp_rt = (N_loc_rt + Int32(tile_m_pe - 1)) // Int32(tile_m_pe)
                    peer_pe = tile_coord_mnkl[0] // nt_pp_rt
                    a_shift = peer_pe * (N_loc_rt - nt_pp_rt * Int32(tile_m_pe))
                else:
                    a_shift = self._pe_tiled_a_row_shift(tile_coord_mnkl[0])
            else:
                # A2/A3 (2-D sharded): peer_i = m_linear // nt_i_pp; a_shift = peer_i*(N_i_loc -
                # nt_i_pp*tile_m). Static bakes nt_i_pp/N_i_loc; dynamic reads N_i_loc from M/cp0.
                cp_axis = self._a2a_cp_axis_sizes
                cp0 = cutlass.const_expr(cp_axis[0])
                if const_expr(getattr(self, "_a2a_dynamic", False)):
                    N_i_rt = Int32(mA_mk.shape[0]) // Int32(cp0)
                    nti = (N_i_rt + Int32(tile_m_pe - 1)) // Int32(tile_m_pe)
                    peer_i = tile_coord_mnkl[0] // nti
                    a_shift = peer_i * (N_i_rt - nti * Int32(tile_m_pe))
                else:
                    nti = cutlass.const_expr(self._a2a_nt_i_pp)
                    N_i = cutlass.const_expr(self._a2a_N_i_loc)
                    peer_i = tile_coord_mnkl[0] // Int32(nti)
                    a_shift = peer_i * Int32(N_i - nti * tile_m_pe)
            # rank-general offset: (a_shift, 0[, 0...]) padded to mA_mk's rank. Rank-2 default ->
            # (a_shift, 0) BYTE-IDENTICAL; the composite-K read's rank-3 (M, Xg_pad, cp) mA_mk ->
            # (a_shift, 0, 0) so the M-row shift lands on the STRIDED M axis and leaves Xg_pad + cp.
            _zpad = (Int32(0),) * (cute.rank(mA_mk) - 1)
            mA_mk = cute.domain_offset((a_shift,) + _zpad, mA_mk)
        return mA_mk

    def mainloop_remap_mB(self, mB_nk, tile_coord_mnkl):
        """CHILD override: shift B's N-origin for 2-D per-peer-j tiling so the mainloop TMA-load reads
        from base_n = peer_j*N_j_loc + kj*tile_n (tile_coord_mnkl[1] is the per-peer-j linear tile
        index). Only the 2-D-sharded pe_aligned path (the 1-D arbitrary_n pe_aligned has cp1==1 -> no
        j-split -> pass-through). Static bakes nt_j_pp/N_j_loc; dynamic reads N_j_loc from N/cp1. The
        partial-last per-peer-j tile's spilled cols are MASKED by the store's recv-descriptor clamp.
        Flag-off / 1-D -> parent pass-through (byte-identical).

        Task #13: the B re-base now also applies to the arbitrary_n ib_ring 2-D drain (cp1>1) -- the same
        per-peer-j shift the coupled 2-D store uses, so the mainloop loads peer_j's own columns and the
        coalesce_dyn drain routes them. The 1-D arbitrary_n path (cp1==1) stays pass-through (the gate's
        `not arbitrary_n or cp1>1` is False there) -> byte-identical."""
        cp_axis = getattr(self, "_a2a_cp_axis_sizes", (1,))
        cp1 = cutlass.const_expr(cp_axis[1] if len(cp_axis) > 1 else 1)
        if const_expr(
            getattr(self, "_pe_aligned_tiling", False)
            and (not getattr(self, "_a2a_arbitrary_n", False) or cp1 > 1)
        ):
            tile_n_pe = cutlass.const_expr(self.cta_tile_shape_mnk[1])
            if const_expr(getattr(self, "_a2a_dynamic", False)):
                N_j_rt = Int32(mB_nk.shape[0]) // Int32(cp1)
                ntj = (N_j_rt + Int32(tile_n_pe - 1)) // Int32(tile_n_pe)
                peer_j = tile_coord_mnkl[1] // ntj
                b_shift = peer_j * (N_j_rt - ntj * Int32(tile_n_pe))
            else:
                ntj = cutlass.const_expr(self._a2a_nt_j_pp)
                N_j = cutlass.const_expr(self._a2a_N_j_loc if cp1 > 1 else 0)
                peer_j = tile_coord_mnkl[1] // Int32(ntj)
                b_shift = peer_j * Int32(N_j - ntj * tile_n_pe)
            # rank-general offset (mirror mainloop_remap_mA): rank-2 default -> (b_shift, 0)
            # BYTE-IDENTICAL; composite-K rank-3 (N, Xg_pad, cp) -> (b_shift, 0, 0).
            _zpad = (Int32(0),) * (cute.rank(mB_nk) - 1)
            mB_nk = cute.domain_offset((b_shift,) + _zpad, mB_nk)
        return mB_nk

    # ==================================================================
    # route2_ni COMPOSITE-K read (§9) — the A/B operand K-hoist overrides. const_expr-gated on
    # _a2a_composite_k so the parent rank-3 (X, K, L) default is BYTE-IDENTICAL when off (proven by the
    # GemmSm90 default-path PTX-identity gate). A and B are SYMMETRIC (both token operands from the front
    # recv), so both get the identical treatment; the K-loop trip (_k_tile_cnt) is shared by copy_A/copy_B.
    # ==================================================================
    def _composite_remap(self, mX):
        """Present a 3-D (X, Xg_pad, L) operand as 4-D (X, Xg_pad, L, cp): SYNTHESIZE the cp
        contraction-rank mode. The per-rank (X, Xg_pad) block is CONTIGUOUS in the recv (X stride = Xg_pad,
        Xg_pad stride = 1) and ranks are stacked at stride X*Xg_pad, so cp's descriptor stride = the block
        size X*Xg_pad. L stays at mode-2 (Dloc stride = cp*X*Xg_pad) so offset_batch_{A,B}
        ([None,None,l]) selects it, leaving (X, Xg_pad, cp). The (BLK, BLK_K) atom box then tiles
        (X, Xg_pad); cp rides as an untiled TMA batch mode (composite MODE is illegal -> the 4-D
        presentation). Mirrors the staged front's rank-hoist, on the K-axis (see dual_gated_gemm_staged_
        a2a._remap_A_operand_layout)."""
        cp = cutlass.const_expr(self._a2a_cp)
        sX = mX.stride[0]  # X (token) stride == Xg_pad (per-rank (X,Xg_pad) block contiguous)
        sK = mX.stride[1]  # Xg_pad (K-within-rank) stride == 1
        sL = mX.stride[2]  # L (Dloc) stride == cp*X*Xg_pad
        rank_stride = mX.shape[0] * sX  # X*Xg_pad == the per-rank block size == the cp stride
        lay = cute.make_layout(
            (mX.shape[0], mX.shape[1], mX.shape[2], cp), stride=(sX, sK, sL, rank_stride)
        )
        return cute.make_tensor(mX.iterator, lay)

    def _composite_local_tile(self, mX_k, blk_xk, tile_x):
        """Tile (X, Xg_pad) of the L-offset 3-D (X, Xg_pad, cp) by (BLK, BLK_K) at tile_x, KEEP cp, and
        GROUP (nt_within, cp) into ONE composite K-loop mode -> (bX, bK, (nt_within, cp)). The flat k_tile
        then unravels (within inner, rank outer); the composite mode's strides (BLK_K on Xg_pad, X*Xg_pad
        on cp) give each k-tile's TMA coord. No K-loop / load_AB change (it iterates k_tile_cnt and indexes
        g[None, k_tile], raster-agnostic)."""
        gX = cute.local_tile(mX_k, blk_xk, (tile_x, None, None))  # (bX, bK, nt_within, cp)
        return cute.group_modes(gX, 2, cute.rank(gX))  # (bX, bK, (nt_within, cp))

    def _remap_A_operand_layout(self, mA, epi_args=None):
        if cutlass.const_expr(not getattr(self, "_a2a_composite_k", False)):
            return super()._remap_A_operand_layout(mA, epi_args)
        return self._composite_remap(mA)

    def _remap_B_operand_layout(self, mB, epi_args=None):
        if cutlass.const_expr(not getattr(self, "_a2a_composite_k", False)):
            return super()._remap_B_operand_layout(mB, epi_args)
        return self._composite_remap(mB)

    def _gA_local_tile(self, mA_mk, tile_coord_mnkl):
        if cutlass.const_expr(not getattr(self, "_a2a_composite_k", False)):
            return super()._gA_local_tile(mA_mk, tile_coord_mnkl)
        blk_mk = cute.select(self.cta_tile_shape_mnk, [0, 2])  # (BLK_M, BLK_K)
        return self._composite_local_tile(mA_mk, blk_mk, tile_coord_mnkl[0])

    def _gB_local_tile(self, mB_nk, tile_coord_mnkl):
        if cutlass.const_expr(not getattr(self, "_a2a_composite_k", False)):
            return super()._gB_local_tile(mB_nk, tile_coord_mnkl)
        blk_nk = cute.select(self.cta_tile_shape_mnk, [1, 2])  # (BLK_N, BLK_K)
        return self._composite_local_tile(mB_nk, blk_nk, tile_coord_mnkl[1])

    def _k_tile_cnt(self, len_k):
        if cutlass.const_expr(not getattr(self, "_a2a_composite_k", False)):
            return super()._k_tile_cnt(len_k)
        # len_k = Xg_pad (ONE rank's K = mA_mkl.shape[1]); the composite K-loop spans all cp ranks ->
        # cp * nt_within tiles (nt_within = ceil(Xg_pad / BLK_K)).
        cp = cutlass.const_expr(self._a2a_cp)
        return cp * cute.ceil_div(len_k, self.cta_tile_shape_mnk[2])

    # ==================================================================
    # A2A-specific store machinery — KEPT VERBATIM (the proven T2.2/#32 recipe).
    # ==================================================================
    def _build_peer_store_atoms_gemm_native(self, recv_5d, epi_smem_layout_staged, epi_tile):
        """Build the ``cp`` peer-pinned S2G atoms for the design-E 5-D GEMM-native store.

        ``recv_5d`` is THIS rank's local symmetric recv ``(cp, Dloc, B, N_loc, N)``. Each
        peer atom's descriptor is pinned to peer ``pe_table[r]``'s heap. The store target
        for a CTA is a ``(N_loc, N)`` sub-tensor at fixed ``(slot, d, b)``; since all
        peers share the SAME recv layout (only the base ptr differs) and the box is the
        SAME ``epi_tile`` per peer, we build the atom on the peer's FULL ``(N_loc, N)``
        slice at ``(slot=my_cp_rank, d=0, b=0)`` (the descriptor's base + the layout are
        what matter; the kernel selects the live ``(d, b, i_tile, j_tile)`` grid coord per
        CTA -- the d/b stride is carried by the 5-D tensor's layout, see the copy_fn).

        Returns ``(atoms, tensors)`` -- length-``cp`` Python lists; ``tensors[r]`` is the
        peer's FULL 5-D recv view (the kernel ``flat_divide``s the ``(N_loc, N)`` box out).

        On the DECOUPLED putwarp-drain path (``_a2a_decoupled_store``) the consumer needs a REAL
        GMEM memref (with an iterator) to ``partition_D`` -- the TMA-atom's ``tensor_r`` is a
        coord/identity tensor and ``cute.copy`` rejects it. So in that mode ``tensors[r]`` is the
        real peer ``(N_loc, N, cp, Dloc, B)`` memref (``peer_view``) and ``atoms[r]`` is ``None``.

        When ``self._a2a_arbitrary_n`` is set (and the default TMA store) we ALSO return the
        raw ``peer_view`` memref per peer (``raw_tensors[r]``) ALONGSIDE the TMA atom/tiled
        tensor: the arbitrary-N coupled store keeps the single TMA-S2G for subtiles that land
        in ONE peer at an ``epi_m``-aligned offset (the common case), and per-row SIMT-puts the
        FEW straddling/misaligned subtiles (which need the raw memref to ``partition_D``). On
        every other path ``raw_tensors`` is ``None`` (byte-identical).
        """
        cp = self._a2a_cp
        pe_table = self._a2a_pe_table
        # arbitrary_n hybrid (default TMA store): collect the raw peer memref too (the SIMT
        # straddle fallback's partition_D source). Only on the TMA store path (not simt_put /
        # not the decoupled SIMT drain, which already return the raw memref as tensors[r]).
        # #57: the cp<=8 COLLAPSE (not has_ib_peers) routes a decoupled/cluster_drain config to the design-E
        # store (build_D_copy_fn fall-through), which under arbitrary_n needs the raw peer memrefs for the
        # straddle SIMT fallback -> want_raw True when collapsing too (else :3371 "arbitrary_n needs the raw
        # peer memrefs" fires). has_ib_peers True default -> the term is False -> byte-identical.
        want_raw = const_expr(
            getattr(self, "_a2a_arbitrary_n", False)
            and (not self._a2a_decoupled_store or not getattr(self, "_a2a_has_ib_peers", True))
        )
        raw_tensors = [] if want_raw else None
        epi_smem_layout = cute.slice_(epi_smem_layout_staged, (None, None, 0))
        # Mirror the L=1 _build_peer_store_atoms recipe (avoids the TMA top-level-shape
        # mismatch, cute-dsl:debug "Descriptor Mismaps" #1): the V-map must be 2D
        # (epi_m, epi_n) matching the 2D SMEM box. PERMUTE the 5-D recv so the (N_loc, N)
        # box modes are FIRST -> (N_loc, N, cp, Dloc, B); then composition(identity, epi_tile)
        # (a 2-mode tiler) tiles (N_loc, N) leaving (cp, Dloc, B) as trailing modes -- exactly
        # the L=1 pattern where epi_tile tiles (M,N) and L rides along. flat_divide in the
        # copy_fn then yields (box, nt_i, nt_j, cp, Dloc, B); index (slot, d, b) into the
        # trailing modes. j innermost (stride-1) is preserved by the permute (N is mode 1).
        op = cpasync.CopyBulkTensorTileS2GOp()
        atoms = []
        tensors = []
        # ---- dynamic-cp WALL, documented here ----------------------------------------
        # cp MUST stay compile-time-constant: this HOST loop builds a length-cp Python list of TMA-S2G
        # atoms, one per peer, each descriptor BAKING peer r's heap base address (get_peer_tensor at
        # pe_table[r]); the copy_fn then const_expr-unrolls `for r in range_constexpr(cp): if peer==r`
        # to pick atoms[r] by a COMPILE-TIME index. A runtime cp would need (a) a runtime-count array of
        # TMA descriptors and (b) a runtime base address per store -- cutlass-dsl 4.4.2 has neither; the
        # only device path is a per-store tensormap.replace.global_address edit (copy_tensormap + replace
        # + release/acquire fences per subtile), which is BOTH new tech debt AND a per-store overhead
        # that regresses the baked-atom TMA (fails the §★ perf gate). NOTE the SCHEDULER is NOT the wall:
        # the persistent STATIC grid = min(total_tiles, max_active_clusters) is already runtime (A1/A3
        # prove a runtime total-tile count works). So dynamic-cp takes the SPEC FALLBACK: dynamic-shape +
        # 2D (A1-A3) ship; the cp axis stays compile-per-cp (cp is a deployment constant = the GPU count).
        for r in range(cp):
            peer_5d = _get_peer_tensor_aligned(recv_5d, Int32(pe_table[r]))  # (cp,Dloc,B,N_loc,N)
            # -> (N_loc, N, cp, Dloc, B): box modes (N_loc,N) first, j (N) stays stride-1.
            peer_view = cute.make_tensor(
                peer_5d.iterator,
                cute.select(peer_5d.layout, mode=[3, 4, 0, 1, 2]),
            )
            # Decoupled putwarp drain: hand the consumer the REAL peer GMEM memref (peer_view, via
            # tensors[r]); no TMA atom. The coupled store (decoupled_store False) builds TMA atoms below.
            # #57: build REAL peer TMA atoms (simt_drain False) when NOT has_ib_peers -- an all-P2P job
            # collapses to the design-E store (build_D_copy_fn fall-through), which needs real atoms not None.
            simt_drain = const_expr(
                self._a2a_decoupled_store and getattr(self, "_a2a_has_ib_peers", True)
            )
            if const_expr(simt_drain):
                # SIMT drain (I1 simt_put / I2a decoupled-SIMT): hand the copy_fn / consumer the REAL
                # peer GMEM memref (peer_view) so it can flat_divide + index it for a warp put; no TMA
                # atom on these paths. The TMA-drain decoupled path (I2b Option B) DOES build real TMA
                # atoms below (the consumer issues a TMA-S2G from the SMEM ring), like the coupled store.
                atoms.append(None)
                tensors.append(peer_view)
                continue
            d_cta_v_layout = cute.composition(cute.make_identity_layout(peer_view.shape), epi_tile)
            atom_r, tensor_r = cpasync.make_tiled_tma_atom(
                op, peer_view, epi_smem_layout, d_cta_v_layout
            )
            atoms.append(atom_r)
            tensors.append(tensor_r)  # tiled (N_loc,N,cp,Dloc,B); copy_fn flat_divides it
            if const_expr(want_raw):
                # arbitrary_n: keep the raw (N_loc,N,cp,Dloc,B) memref too (SIMT straddle fallback).
                raw_tensors.append(peer_view)
        if const_expr(want_raw):
            return atoms, tensors, raw_tensors
        return atoms, tensors

    def _build_peer_store_atoms(self, tensor_d, epi_smem_layout_staged, epi_tile):
        """Build the ``cp`` peer-pinned S2G TMA atoms + peer tensors for the D-store.

        For each flat peer ``r``, peer-translate ``tensor_d`` (this rank's local
        symmetric recv buffer) to peer ``r``'s heap via ``get_peer_tensor`` and build
        an S2G atom whose descriptor base is pinned to peer ``r``. The SMEM box +
        layout are identical across peers (same ``epi_tile``) -- each atom is the
        local store atom retargeted to a different peer.

        Returns ``(atoms, tensors)`` -- Python lists (compile-time static structure)
        of length ``cp``; each element is a real ``cute.CopyAtom`` / ``cute.Tensor``.
        """
        cp = self._a2a_cp
        pe_table = self._a2a_pe_table
        epi_smem_layout = cute.slice_(epi_smem_layout_staged, (None, None, 0))
        d_cta_v_layout = cute.composition(cute.make_identity_layout(tensor_d.shape), epi_tile)
        op = cpasync.CopyBulkTensorTileS2GOp()
        atoms = []
        tensors = []
        for r in range(cp):
            peer_view = _get_peer_tensor_aligned(tensor_d, Int32(pe_table[r]))
            atom_r, tensor_r = cpasync.make_tiled_tma_atom(
                op, peer_view, epi_smem_layout, d_cta_v_layout
            )
            atoms.append(atom_r)
            tensors.append(tensor_r)
        return atoms, tensors

    def _a2a_peer_store_copy_fn_gemm_native(
        self,
        atoms,
        tensors,
        tile_shape_mn,
        epi_tile,
        sD,
        tile_coord_mnkl,
        raw_tensors=None,
        route2_atoms=None,
        route2_tensors=None,
        route2_dynstrip_ws=None,
        route2_cache_ws=None,
        storage=None,
    ):
        """Build copy_D for the design-E GEMM-native 5-D store (1D AND 2D token shard).

        ``tensors[r]`` is peer ``r``'s FULL 5-D recv ``(cp, Dloc, B, N_i_loc, N_j_loc)``
        view (carried on epi_params -> region-local). For this CTA's plane
        ``L = tile_coord_mnkl[3]``: ``d = L // B``, ``b = L % B``.

        **1-D token shard** (``cp_axis_sizes == (cp,)``, ``N_j_loc == 0`` sentinel =>
        j-axis FULL): only the GEMM-M (i) token axis is cp-split. The M-tile routes to
        peer ``tm // tiles_per_i_block`` (token-i block); within that peer's recv it lands
        at slot ``my_cp_rank``, ``(d, b)`` plane, i_local sub-tile
        ``(tm % tiles_per_i_block)*m_sub_per_tile + sub_m`` and j sub-tile
        ``tn*n_sub_per_tile + sub_n`` (j is full, ``N_j_loc == N``).

        **2-D token shard** (``cp_axis_sizes == (cp0, cp1)``): BOTH token axes are
        cp-split. The i-tile -> cp0 coord ``tm // tiles_per_i_block`` and the j-tile ->
        cp1 coord ``tn // tiles_per_j_block``; the flat peer is the row-major
        re-flatten ``cp0_coord*cp1 + cp1_coord`` (``cp1`` = the LayoutRightMap stride of
        the cp0 axis, the AUTHORITATIVE PeMap flatten). Within that peer's recv it lands
        at slot ``my_cp_rank``, ``(d, b)`` plane, i_local sub-tile
        ``(tm % tiles_per_i_block)*m_sub_per_tile + sub_m`` and j_local sub-tile
        ``(tn % tiles_per_j_block)*n_sub_per_tile + sub_n``. A CTA tile maps to ONE peer
        on BOTH axes (``N_i_loc % tile_m == 0`` AND ``N_j_loc % tile_n == 0``, asserted
        host-side), so the cp0/cp1 coords are constant across a CTA's subtiles.

        The 1-D path is the ``cp1 == 1`` special case of the 2-D math (``cp1_coord`` is
        always 0, ``j_local == j_global``), so a single unified branch covers both; the
        ``const_expr(cp1 > 1)`` only selects the recv's j extent ``N_j_loc`` (vs full N)
        for ``tiles_per_j_block``.

        The peer 5-D tensor is sliced at the runtime ``(slot, d, b)`` to a
        ``(N_i_loc, N_j_loc)`` box (j-innermost stride-1 = shape-matched to ``sD``);
        ``flat_divide`` + ``tma_partition`` then ``cute.copy`` the box at the local tile.
        Returns ``(copy_fn, s0, g0)`` (the epilogue consumes only ``copy_fn``).
        """
        cp: cutlass.Constexpr[int] = self._a2a_cp
        my_cp_rank: cutlass.Constexpr[int] = self._a2a_my_cp_rank
        # The batch extent must NOT be a compile-time constant: `B` reaches this kernel ONLY as the
        # divisor of the GEMM batch coord (`d = L // B`, `b = L % B`), so baking it made one
        # compiled kernel serve one batch extent for no design reason. It is READ instead: mode 4
        # of the permuted peer recv (N_i_loc, N_j_loc, cp, Dloc, B) IS `B`. Runtime `Int32` wherever
        # the caller marked the recv dynamic (the production `TriMulAutotuned` path), a folded
        # Python int on a per-shape static compile -- the same split `N_loc_rt` has at :3903, and
        # why there is no branch here. There is deliberately NO static-B mode.
        B = tensors[0].shape[4]
        cp_axis_sizes = self._a2a_cp_axis_sizes
        cp0: cutlass.Constexpr[int] = cp_axis_sizes[0]
        cp1: cutlass.Constexpr[int] = cp_axis_sizes[1] if len(cp_axis_sizes) > 1 else 1
        tile_m: cutlass.Constexpr[int] = tile_shape_mn[0]
        tile_n: cutlass.Constexpr[int] = tile_shape_mn[1]
        epi_m: cutlass.Constexpr[int] = epi_tile[0]
        epi_n: cutlass.Constexpr[int] = epi_tile[1]
        # row-major flatten stride of the cp0 axis (== cp1); 1 for the 1-D special case.
        cp1_stride: cutlass.Constexpr[int] = cp1
        m_sub_per_tile: cutlass.Constexpr[int] = tile_m // epi_m
        n_sub_per_tile: cutlass.Constexpr[int] = tile_n // epi_n
        # Token-scaling extents (tiles_per_i/j_block from N_i_loc/N_j_loc): baked const_expr
        # (static -> byte-identical) OR read off the recv's RUNTIME shape (dynamic-shape compile:
        # one compile serves many token counts). tensors[0] is the peer recv permuted to
        # (N_i_loc, N_j_loc, cp, Dloc, B) -> shape[0]=N_i_loc, shape[1]=N_j_loc (== N for 1-D).
        # The copy_fn divmod uses tiles_per_*_block identically for both (Int32(const) | Int32(rt)).
        # CEIL tiles/block on BOTH axes so every CTA (i,j) tile maps into ONE peer block (the mainloop
        # A-row/B-col shifts re-base the loads; the recv descriptor clamps the partial-last per-peer tile).
        # This is CEIL on ALL pe_aligned paths (2-D pe_aligned_2d below AND 1-D arbitrary_n via the else):
        # the copy_fn (:3439/3442) DOES consume tiles_per_* (cp0_coord = m_linear // tiles_per_i_block), so
        # the earlier "1-D copy_fn ignores tiles_per_*, keep FLOOR" assumption was WRONG -> FLOOR gave
        # N_loc//128=0 for N_loc<128 -> a div-0-poison OOB store (the rank1 small-N bug). Aligned N_i/N_j
        # (N%128==0, incl. the non-pe_aligned design-E path) -> ceil==floor -> byte-identical.
        # Task #13: the arbitrary_n ib_ring path uses the 1-D SIMT-hybrid store below, BUT its 2-D (cp1>1)
        # coupled TMA-S2G reuses the CLEAN 2-D copy_fn (per-peer CEIL
        # tiling + the fully-OOB guard + recv clamp) -- so pe_aligned_2d is ALSO True for arbitrary_n +
        # cp1>1 (routes it there). 1-D arbitrary_n (cp1==1) keeps pe_aligned_2d False (SIMT-hybrid path).
        pe_aligned_2d: cutlass.Constexpr[bool] = bool(
            getattr(self, "_pe_aligned_tiling", False)
            and (not getattr(self, "_a2a_arbitrary_n", False) or cp1 > 1)
        )
        if const_expr(self._a2a_dynamic):
            if const_expr(pe_aligned_2d):
                tiles_per_i_block = (tensors[0].shape[0] + tile_m - 1) // tile_m  # runtime ceil
                tiles_per_j_block = (tensors[0].shape[1] + tile_n - 1) // tile_n
            else:
                tiles_per_i_block = tensors[0].shape[0] // tile_m  # runtime
                tiles_per_j_block = tensors[0].shape[1] // tile_n
        elif const_expr(pe_aligned_2d):
            N_i_loc: cutlass.Constexpr[int] = int(tensors[0].shape[0])
            N_j_loc: cutlass.Constexpr[int] = int(tensors[0].shape[1])
            tiles_per_i_block: cutlass.Constexpr[int] = (N_i_loc + tile_m - 1) // tile_m
            tiles_per_j_block: cutlass.Constexpr[int] = (N_j_loc + tile_n - 1) // tile_n
        else:
            N_i_loc: cutlass.Constexpr[int] = self._a2a_N_i_loc
            N_j_loc: cutlass.Constexpr[int] = int(tensors[0].shape[1])
            # CEIL (= nt_pp), NOT floor. On the pe_aligned arbitrary_n path tile_coord[0] is the PER-PEER
            # m_linear, so tiles_per_i_block must be the per-peer tile count ceil(N_loc/tile_m). FLOOR gave
            # N_loc//128 = 0 for N_loc<128 -> cp0_coord = m_linear//0 = DIV-0 POISON -> garbage i_tile -> OOB
            # store that corrupts the recv (the rank1 small-N all-zero bug; div-0 poison pattern). Same for
            # j (1-D: N_j_loc = full N -> N//128 = 0 for N<128 -> cp1_coord = tile_j//0 div-0). Aligned
            # (N_i/N_j % 128 == 0, incl. the non-pe_aligned design-E path that also reaches this else) ->
            # ceil == floor -> BYTE-IDENTICAL. Only pe_aligned straddle changes, and ceil=nt_pp is correct.
            tiles_per_i_block: cutlass.Constexpr[int] = (N_i_loc + tile_m - 1) // tile_m
            tiles_per_j_block: cutlass.Constexpr[int] = (N_j_loc + tile_n - 1) // tile_n
        # A2/A3: recv i/j extents for the fully-OOB epi-subtile guard (the per-peer ceil tiling emits
        # n_sub_per_tile epi-boxes per CTA tile; those ENTIRELY past N_i_loc/N_j_loc must be SKIPPED --
        # the partial ones straddling the edge are descriptor-clamped, but a fully-past box would index
        # the flat_divide beyond its tile grid). Only referenced under const_expr(pe_aligned_2d).
        if const_expr(pe_aligned_2d):
            pe_i_ext = (
                tensors[0].shape[0] if const_expr(self._a2a_dynamic) else int(tensors[0].shape[0])
            )
            pe_j_ext = (
                tensors[0].shape[1] if const_expr(self._a2a_dynamic) else int(tensors[0].shape[1])
            )

        # Per-peer partition ONCE, host-side (mirrors the L=1 path). tensors[r] is the tiled
        # peer recv permuted to (N_i_loc, N_j_loc, cp, Dloc, B); flat_divide by the 2D
        # epi_tile tiles only (N_i_loc, N_j_loc) -> (epi_m, epi_n, nt_i, nt_j, cp, Dloc, B).
        # group_modes(.,0,2) folds the (epi_m,epi_n) box -> g_r = (box, nt_i, nt_j, cp,
        # Dloc, B); cute.copy indexes g_r[(None, i_tile, j_tile, slot, d, b)].
        s_views = []
        g_views = []
        gD_coord_views = []  # flat_divided TMA coord tensors (clamped-TMA straddle: element-origin box)
        for r in range(cp):
            gD = cute.flat_divide(tensors[r], epi_tile)  # (epi_m,epi_n, nt_i,nt_j, cp,Dloc,B)
            gD_coord_views.append(gD)
            s_r, g_r = cpasync.tma_partition(
                atoms[r],
                0,
                cute.make_layout(1),
                cute.group_modes(sD, 0, cute.rank(sD) - 1),
                cute.group_modes(gD, 0, 2),
            )
            s_views.append(s_r)
            g_views.append(g_r)

        # =================================================================================
        # ARBITRARY-N (straddling N_loc): SIMT-HYBRID coupled store. 1-D token shard ONLY.
        # ---------------------------------------------------------------------------------
        # When N_loc is NOT a multiple of cta_tile_M, a CTA's epi-subtile [base_m, base_m+epi_m)
        # may (a) straddle >=2 peer-blocks, or (b) land in ONE peer at an offset
        # (base_m - peer*N_loc) NOT a multiple of epi_m -> the per-epi-tile TMA partition (which
        # addresses i_local in epi_m-row units) cannot place it. For such subtiles we drain the
        # rows via a per-row SIMT put (each global row gi -> peer gi//N_loc at i_local gi%N_loc);
        # the COMMON single-peer + epi_m-aligned subtile keeps the single TMA-S2G (reusing
        # s_views/g_views above). Gated on _a2a_arbitrary_n -> aligned N stays the byte-identical
        # single-store path below. 2-D shard arbitrary_n is unsupported HERE (would need j-straddle too);
        # Task #13 routes the 2-D (cp1>1) coupled store to the CLEAN 2-D copy_fn below (the
        # gate is `arbitrary_n and cp1==1`, so cp1>1 falls through to the pe_aligned_2d clean store).
        if const_expr(getattr(self, "_a2a_arbitrary_n", False) and cp1 == 1):
            assert raw_tensors is not None, (
                "arbitrary_n needs the raw peer memrefs (peer_raw_tensors)."
            )
            N_loc_c: cutlass.Constexpr[int] = self._a2a_N_loc
            # Fix B (dynamic N_loc, route-2): the per-row peer routing (p_lo/p_hi/off_lo/split) must use
            # the RUNTIME N_loc, not the compile anchor. raw_tensors[r] is the peer recv (N_loc, N, ...);
            # shape[0] is N_loc -- a runtime Int32 when _a2a_dynamic (mark_layout_dynamic). On the static
            # path it equals N_loc_c, so N_loc_rt is byte-identical there. The Constexpr partial-tile
            # flags (M_c/N_c/has_partial_*) stay on N_loc_c (route-2 elides the SIMT tail they gate).
            if const_expr(self._a2a_dynamic):
                N_loc_rt = Int32(raw_tensors[0].shape[0])
            else:
                N_loc_rt = Int32(N_loc_c)
            # P_i / P_j extents (partial M/N tiles): M = N_loc*cp is the TRUE token-i count; N (the true
            # token-j extent) is read off the peer memref's shape[1] (= recv N axis), const for the static
            # compile. The SLOW per-row store guards gi < M (drops the last partial M-tile's garbage rows)
            # and masks cols gj < N (drops the last partial N-epi-tile's garbage cols). The FAST single-TMA
            # store needs NO guard: the TMA descriptor is on the true (N_loc, N) -> partial N clamps, and a
            # garbage row gi>=M makes p_lo=gi//N_loc>=cp so the const_expr peer loop matches nobody (no store).
            # PERF-NEUTRALITY: compile-time partial-tile flags (M=N_loc*cp vs tile_m; N vs tile_n) ->
            # aligned shapes elide the P_i guard + P_j mask (byte-identical). N uses tile_n (the CTA
            # N-tile), NOT epi_n: the epilogue visits n_sub_per_tile epi_n-chunks of the LAST CTA N-tile,
            # so when N % tile_n != 0 (partial CTA N-tile) its TRAILING epi_n-chunks are fully OOB
            # (cols >= N) and MUST be masked -- even when N % epi_n == 0. (N=576, tile_n=128, epi_n=32:
            # 576%32==0 but 576%128=64 -> the chunks at cols 576..639 are garbage; using epi_n here
            # silently dropped the mask -> unmasked OOB col writes corrupted the recv: the square
            # partial-M / N%tile_n!=0 / N%epi_n==0 correctness bug. has_partial_m_c already uses tile_m.)
            # Fix B (dynamic N_loc): under _a2a_dynamic the shapes are RUNTIME -> int() would crash.
            # route-2 dynamic ELIDES the SIMT tail (do_row forced False), and the FAST single-TMA store
            # needs no partial flag, so the SIMT-masking flags are off the critical path. Compile-time
            # FALSE under dynamic route-2 (SIMT dead) avoids both the int(dynamic-shape) AND tracing the
            # dead per-lane predicate. The static path keeps the exact Constexpr flags (byte-identical).
            dyn_path: cutlass.Constexpr[bool] = bool(self._a2a_dynamic)
            if const_expr(dyn_path):
                # SIMT-masking flags compile-time False: dynamic arbitrary_n is supported ONLY via
                # pe_aligned (Track A) -- the per-peer-aligned bases make every store the FAST
                # single-TMA-S2G whose recv descriptor clamps the partial-last per-peer tile, so there
                # is no live SIMT tail to mask (the const_expr peer-match drops gi>=M rows of the last
                # peer; the fast-path TMA descriptor clamps partial N).
                pe_tiled_cf: cutlass.Constexpr[bool] = bool(
                    getattr(self, "_pe_aligned_tiling", False)
                )
                assert pe_tiled_cf, (
                    "dynamic arbitrary_n is only supported with pe_aligned_tiling (Track A)."
                )
                has_partial_m_c: cutlass.Constexpr[bool] = False
                has_partial_n_c: cutlass.Constexpr[bool] = False
            else:
                M_c: cutlass.Constexpr[int] = N_loc_c * cp
                N_c: cutlass.Constexpr[int] = int(raw_tensors[0].shape[1])
                has_partial_m_c: cutlass.Constexpr[bool] = (M_c % tile_m) != 0
                has_partial_n_c: cutlass.Constexpr[bool] = (N_c % tile_n) != 0
            # Raw per-peer flat_divide views for the SIMT straddle drain (epi_m,epi_n, nt_i,nt_j,
            # cp,Dloc,B) — same layout the simt_put path uses; index the (epi_m,epi_n) box then
            # slice ONE row. nt_i = ceil(N_loc/epi_m) (N_loc need not be a multiple of epi_m, but
            # flat_divide of the (N_loc,N,...) memref tiles N_loc into epi_m boxes; the last partial
            # box's OOB rows are never selected — i_local < N_loc always).
            g_raw_views = [cute.flat_divide(raw_tensors[r], epi_tile) for r in range(cp)]
            # 32-lane single-ROW store. epi_n is a cooperative epi-tile-N (gcd(32,tile_n) -> a
            # multiple of 32, =32 at the 128x128 default): all 32 lanes tile the row's epi_n cols,
            # each lane owning vec = epi_n//32 contiguous cols (STG.{16,32,...}). lanes_n*vec == epi_n
            # EXACTLY -> no OOB lane (every lane has a valid partition; no iteration). The assert
            # guards the (opt-in) arbitrary_n path against an exotic epi_n not a multiple of 32.
            assert epi_n % 32 == 0, f"arbitrary_n SIMT row store needs epi_n%32==0 (got {epi_n})."
            elem_ty = sD.element_type
            elem_bits: cutlass.Constexpr[int] = int(elem_ty.width)
            lanes_n: cutlass.Constexpr[int] = 32
            vec: cutlass.Constexpr[int] = epi_n // lanes_n
            row_atom = cute.make_copy_atom(
                cute.nvgpu.CopyUniversalOp(), elem_ty, num_bits_per_copy=vec * elem_bits
            )
            # thr (1, 32) over the row's columns; val (1, vec) gives each lane vec contiguous cols.
            row_tiled = cute.make_tiled_copy_tv(
                row_atom,
                cute.make_ordered_layout((1, lanes_n), order=(1, 0)),
                cute.make_layout((1, vec)),
            )
            # P_j (partial N): identity over the (1, epi_n) row so each lane knows the IN-EPI-TILE col of
            # its vec element; the per-lane predicate is (j_tile*epi_n + col) < N. Built once (host trace);
            # only consumed in copy_fn when has_partial_n_c. Each lane owns ``vec`` contiguous cols; N%8==0
            # + epi_n%32==0 -> vec divides (N - j0) so a lane never straddles the N boundary (whole-lane).
            if const_expr(has_partial_n_c):
                cRow = cute.make_identity_tensor((1, epi_n))

            @cute.jit
            def copy_fn(src_idx, dst_idx, **kwargs):
                sub_m, sub_n = dst_idx[0], dst_idx[1]
                L = tile_coord_mnkl[3]
                d = L // Int32(B)
                b = L % Int32(B)
                slot = Int32(my_cp_rank)
                # Global row start of THIS epi-subtile + the j epi-tile index (j is FULL for 1-D).
                # PE-boundary-aware per-peer M-tiling: tile_coord_mnkl[0] is the per-peer linear tile
                # index -> base_m = _pe_tiled_base_m(.) (peer*N_loc + k*tile_m). Every subtile then lies
                # in ONE peer at an epi_m-aligned i_local -> the FAST single-TMA path; the partial-last
                # per-peer tile's rows past N_loc are clamped by the recv descriptor (case-(c)).
                if const_expr(getattr(self, "_pe_aligned_tiling", False)):
                    # A1 (dynamic-shape): runtime per-peer base = peer*N_loc + k*tile_m (N_loc_rt from the
                    # recv shape, already computed above). Static keeps the const_expr helper (perf §★ #1).
                    if const_expr(self._a2a_dynamic):
                        nt_pp_rt = (N_loc_rt + Int32(tile_m - 1)) // Int32(tile_m)
                        peer_pe = tile_coord_mnkl[0] // nt_pp_rt
                        # k = m_linear - peer*nt_pp (mul-sub, avoids a 2nd runtime int-divide; perf §★).
                        k_pe = tile_coord_mnkl[0] - peer_pe * nt_pp_rt
                        base_m = (
                            peer_pe * N_loc_rt + k_pe * Int32(tile_m) + Int32(sub_m) * Int32(epi_m)
                        )
                    else:
                        base_m = self._pe_tiled_base_m(tile_coord_mnkl[0]) + Int32(sub_m) * Int32(
                            epi_m
                        )
                else:
                    base_m = tile_coord_mnkl[0] * Int32(tile_m) + Int32(sub_m) * Int32(epi_m)
                j_tile = tile_coord_mnkl[1] * Int32(n_sub_per_tile) + Int32(sub_n)
                p_lo = base_m // N_loc_rt
                p_hi = (base_m + Int32(epi_m) - Int32(1)) // N_loc_rt
                off_lo = base_m - p_lo * N_loc_rt  # i_local of base_m on peer p_lo
                aligned = (off_lo % Int32(epi_m)) == Int32(0)
                # bitwise & (NOT python `and`) so neither operand's __bool__ is invoked at trace time
                # (a dynamic Boolean rejects bool()); `if <runtime Boolean>:` lowers to scf.if cleanly.
                pe_tiled: cutlass.Constexpr[bool] = bool(getattr(self, "_pe_aligned_tiling", False))
                # PE-tiled: base_m is per-peer-aligned so off_lo is ALWAYS epi_m-aligned and the subtile
                # lies in ONE peer p_lo; the partial-last per-peer tile's rows i_local >= N_loc (p_hi
                # computes as p_lo+1) are GARBAGE (double-compute) and the recv descriptor (true N_loc
                # extent) CLAMPS them on the single TMA -> take the FAST path on `aligned` alone (NOT
                # p_lo==p_hi). This is the whole point: every store is one clean TMA, no straddle branch.
                fast = aligned if const_expr(pe_tiled) else ((p_lo == p_hi) & aligned)
                if fast:
                    # FAST PATH: whole subtile -> ONE peer, epi_m-aligned -> single TMA-S2G.
                    i_tile = off_lo // Int32(epi_m)
                    for r in cutlass.range_constexpr(cp):
                        if p_lo == Int32(r):
                            cute.copy(
                                atoms[r],
                                s_views[r][(None, src_idx)],
                                g_views[r][(None, i_tile, j_tile, slot, d, b)],
                            )
                else:
                    # STRADDLE (or single-peer non-epi_m-aligned) subtile.
                    # Per-row SIMT drain. Each global row gi -> peer gi//N_loc at i_local gi%N_loc;
                    # store the row's epi_n columns (vectorized 32-lane). No TMA op issued on this
                    # branch (clamped_tma's TMA above is a SEPARATE bulk op committed in the same
                    # per-tile group) -> SMEM reuse ordered by the LDS retiring before the next
                    # epilogue_barrier; remote completion by the host quiet(). Under clamped_tma the
                    # loop SKIPS p_lo's rows (gi < (p_lo+1)*N_loc) -- the TMA already wrote them.
                    # route-2 ELIDES the SIMT tail (every peer is served by a TMA above) -> no
                    # intra-NVLink SIMT. The per-row do_row predicate is const-forced False under
                    # route2 so the dynamic row loop stores NOTHING (the SIMT cute.copy is dead);
                    # the loop setup stays in scope (DSL: control-flow-body vars don't escape).
                    s_box = cute.slice_(sD, (None, None, src_idx))  # (epi_m, epi_n)
                    lane = cute.arch.lane_idx()
                    thr = row_tiled.get_slice(lane)
                    # P_j: this epi-tile's GLOBAL col base; under has_partial_n_c build the per-lane
                    # in-bounds mask (j0 + in-epi col) < N (cheap; lane-uniform j0). j0 + tDcR are
                    # const-gated so an ALIGNED N never traces the predicate fragment (byte-identical).
                    # DYNAMIC row loop (NOT range_constexpr): the 128-row × cp-peer full unroll was a
                    # compile-time trace EXPLOSION (128*cp nested dynamic scf.if) that both blew up
                    # compile time (>15 min cold) AND tripped the DSL dynamic-Boolean lowering at cp>=8.
                    # A dynamic loop traces ONE body -> small trace, fast compile, no nesting blowup.
                    for lr in cutlass.range(epi_m):
                        gi = base_m + Int32(lr)
                        # P_i: skip garbage rows of the last partial M-tile (gi >= M=N_loc*cp) -> the
                        # const_expr peer match below would also drop them (peer_r>=cp matches no r), but
                        # the explicit guard is clearer + future-proof. const-gated on has_partial_m_c ->
                        # ALIGNED M traces NO guard (every gi < M) -> byte-identical to the pre-arb path.
                        do_row = (gi < Int32(M_c)) if const_expr(has_partial_m_c) else True
                        if do_row:
                            peer_r = gi // N_loc_rt
                            i_local = gi - peer_r * N_loc_rt
                            s_row = cute.slice_(s_box, (Int32(lr), None))  # (epi_n,) SMEM row lr
                            # (1, epi_n) so the tiled copy's 2-mode (rows=1, cols) thr layout matches.
                            s_row2 = cute.make_tensor(s_row.iterator, cute.make_layout((1, epi_n)))
                            tSsR = thr.partition_S(s_row2)
                            i_box = i_local // Int32(epi_m)
                            i_in_box = i_local % Int32(epi_m)
                            for r in cutlass.range_constexpr(cp):
                                if peer_r == Int32(r):
                                    g_box = g_raw_views[r][
                                        (None, None, i_box, j_tile, slot, d, b)
                                    ]  # (epi_m, epi_n) on peer r's heap
                                    g_row = cute.slice_(g_box, (i_in_box, None))  # (epi_n,)
                                    g_row2 = cute.make_tensor(
                                        g_row.iterator, cute.make_layout((1, epi_n))
                                    )
                                    if const_expr(has_partial_n_c):
                                        # mask each lane's vec cols: (j0 + in-epi col) < N. tDcR[v][1] is
                                        # the identity col coord. Whole-lane in/out (vec divides N-j0 by
                                        # N%8==0 + epi_n%32==0 -> no straddle).
                                        j0 = j_tile * Int32(epi_n)
                                        tDcR = thr.partition_D(cRow)
                                        ncol_per_lane: cutlass.Constexpr[int] = cute.size(tDcR)
                                        tApR = cute.make_rmem_tensor(ncol_per_lane, cutlass.Boolean)
                                        for v in cutlass.range_constexpr(ncol_per_lane):
                                            tApR[v] = (j0 + Int32(tDcR[v][1])) < Int32(N_c)
                                        cute.copy(
                                            row_tiled, tSsR, thr.partition_D(g_row2), pred=tApR
                                        )
                                    else:
                                        cute.copy(row_tiled, tSsR, thr.partition_D(g_row2))

            return copy_fn, s_views[0], g_views[0]

        @cute.jit
        def copy_fn(src_idx, dst_idx, **kwargs):
            sub_m, sub_n = dst_idx[0], dst_idx[1]
            L = tile_coord_mnkl[3]
            d = L // Int32(B)
            b = L % Int32(B)
            # i-tile -> cp0 coord + i-local tile; j-tile -> cp1 coord + j-local tile.
            # For 1-D (cp1 == 1) tiles_per_j_block spans ALL j tiles -> cp1_coord == 0 and
            # tile_in_j_block == tile_coord_mnkl[1] (j full), reproducing the 1-D math.
            cp0_coord = tile_coord_mnkl[0] // Int32(tiles_per_i_block)
            cp1_coord = tile_coord_mnkl[1] // Int32(tiles_per_j_block)
            peer = cp0_coord * Int32(cp1_stride) + cp1_coord
            tile_in_i_block = tile_coord_mnkl[0] % Int32(tiles_per_i_block)
            tile_in_j_block = tile_coord_mnkl[1] % Int32(tiles_per_j_block)
            i_tile = tile_in_i_block * Int32(m_sub_per_tile) + sub_m
            j_tile = tile_in_j_block * Int32(n_sub_per_tile) + sub_n
            # const_expr-unrolled peer select (cp compile-time). slot = my_cp_rank (which
            # source feature-slice sent it); (d, b) select the recv plane; (i_tile, j_tile)
            # the box within the (N_i_loc, N_j_loc) tile. The L-coord un-bake is the (d,b) select.
            if const_expr(pe_aligned_2d):
                # SKIP epi-subtiles ENTIRELY past the peer block (per-peer ceil tiling emits them);
                # the edge-straddling box is descriptor-clamped, but a fully-past box must not store.
                do_store = (i_tile * Int32(epi_m) < Int32(pe_i_ext)) & (
                    j_tile * Int32(epi_n) < Int32(pe_j_ext)
                )
                for r in cutlass.range_constexpr(cp):
                    if (peer == Int32(r)) & do_store:
                        cute.copy(
                            atoms[r],
                            s_views[r][(None, src_idx)],
                            g_views[r][(None, i_tile, j_tile, Int32(my_cp_rank), d, b)],
                        )
            else:
                for r in cutlass.range_constexpr(cp):
                    if peer == Int32(r):
                        cute.copy(
                            atoms[r],
                            s_views[r][(None, src_idx)],
                            g_views[r][(None, i_tile, j_tile, Int32(my_cp_rank), d, b)],
                        )

        return copy_fn, s_views[0], g_views[0]
