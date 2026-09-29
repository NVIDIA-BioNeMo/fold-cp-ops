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
# This file has been modified by NVIDIA CORPORATION & AFFILIATES.

"""Back-A2A GEMM-epilogue fusion (T2.2): public entry + default epilogue mix.

This is the Wave-2 *back* all-to-all (design doc §2 step 3, §3.1): the GEMM1
``tri = einsum(a, b)`` output store — normally a local SMEM->GMEM **TMA store** —
becomes an in-kernel SMEM -> **peer's symmetric-heap GMEM** TMA S2G store (NVLink),
written **token-major** so the reshard ``(S0,S3,S3) -> (S0,S1,S2)`` (feature-shard
-> token-shard) happens *inside* the epilogue with NO local GMEM round-trip and NO
separate A2A kernel launch (the T2.0c wall: the token<->D transpose + GMEM staging
cost ~= the einsum).

Code layout (§3.1, hard rule). The kernel mechanics live in
:class:`fold_cp_ops.distributed.gemm_sm90_a2a.GemmSm90A2A` — an EDITABLE copy of the local
``GemmSm90`` placed in the distributed subdir (the local ``fold_cp_ops/gemm_sm90.py`` stays
UNTOUCHED). The copy threads ``cp`` peer-pinned S2G TMA atoms through its
``__call__`` / ``kernel`` signature (a TMA descriptor pins ONE peer base at build
time, and the back-A2A maps different output M-blocks to different peers, so per-peer
atoms are mandatory and — being host-built runtime objects — must be threaded as
kernel args, NOT stashed on ``self``: self-stash does not cross the
``@cute.jit`` -> ``@cute.kernel`` region). The epilogue store seam selects atom ``r``
for the CTA whose M-block targets peer ``r`` and stores at the back-A2A reshard row
offset. All A2A branches are ``const_expr(self._a2a_enabled)``-gated, default OFF, so
a flag-off kernel is byte-identical to the local :class:`GemmSm90`.

This module provides the public class :class:`GemmA2ASm90` (the default-epilogue mix
over the distributed kernel) and re-exports :class:`GemmSm90A2A`. Enable the back-A2A
store via :meth:`GemmSm90A2A.configure_a2a` before compiling. Compile via the bitcode
route (``--link-libraries={find_device_bitcode_library()}``, tvm-ffi OFF) — see the
test's compile shim / the T2.0d ``fold_cp_ops.distributed.gemm_bitcode_compile`` wrapper.

Mechanism = the upstream ``put_signal_nbi_tma_peer`` DECOMPOSED: a peer-pinned S2G atom
(descriptor built from the upstream nvshmem ``get_peer_tensor``) + the SMEM source.
It does NOT use ``put_nbi_warp`` (GMEM-source-only, cannot read epilogue SMEM) and
does NOT stage through local GMEM. Drain/signal = T2.3 (the caller drains via
``nvshmem.core.rma.quiet`` / ``barrier_all`` after the kernel).
"""

import dataclasses
import typing
from typing import NamedTuple, Optional

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32

from fold_cp_ops._internal.runtime_params import mlir_namedtuple
from fold_cp_ops.distributed.gemm_sm90_a2a import GemmSm90A2A
from fold_cp_ops._internal.epi_composable import _make_epi_params
from fold_cp_ops._internal.epi_default import GemmDefaultEpiMixin
from fold_cp_ops._internal.rounding import RoundingMode

__all__ = ["GemmSm90A2A", "GemmA2ASm90"]


class GemmA2ASm90(GemmDefaultEpiMixin, GemmSm90A2A):
    """SM90 dense GEMM (default epilogue) whose D-store is a back-A2A peer TMA store.

    Identical to :class:`~fold_cp_ops.kernels.gemm_default_epi.GemmDefaultSm90` until
    :meth:`GemmSm90A2A.configure_a2a` (or :meth:`configure_a2a_gemm_native`) flips the
    A2A flag (then the GEMM1 result is stored SMEM -> peer-symmetric-heap, reshard
    S3->S1, instead of locally).

    PEER-ATOM ROUTING (front-consistent, design doc §"SIMPLIFICATION FOUND"): the cp
    peer-pinned S2G atoms ride on the composable ``EpilogueParams`` (which already
    crosses the @cute.jit -> @cute.kernel region as a kernel arg). This class extends
    the mixin ``EpilogueArguments`` with a ``recv`` field (THIS rank's symmetric recv
    buffer — the back-A2A scatter target; default None -> flag-off), regenerates
    ``EpilogueParams`` with two extra peer fields, and overrides
    ``epi_to_underlying_arguments`` to build the peer atoms from ``args.recv`` and
    attach them. The kernel-side store redirect (``build_D_copy_fn``) lives on
    ``GemmSm90A2A`` and reads the peer atoms off the params. NO kernel/__call__ copy.
    """

    # EpilogueArguments: mixin's NamedTuple + a ``recv`` field (the symmetric recv
    # buffer). The caller passes the LOGICAL (M,N,L) GEMM view as ``mD`` (parent
    # shape/scheduler/local-atom machinery unchanged) and the real recv (3-D plain
    # (M,N,L) or 5-D gemm-native (cp,Dloc,B,N_loc,N)) via ``recv``.
    @mlir_namedtuple
    class EpilogueArguments(NamedTuple):
        alpha: Optional[Float32 | cute.Tensor] = None
        beta: Optional[Float32 | cute.Tensor] = None
        mRowVecBroadcast: Optional[cute.Tensor] = None
        mColVecBroadcast: Optional[cute.Tensor] = None
        add_to_output: cutlass.Constexpr[bool] = False
        rounding_mode: cutlass.Constexpr[int] = RoundingMode.RN
        sr_seed: Optional[Int32 | cute.Tensor] = None
        recv: Optional[cute.Tensor] = None  # symmetric recv buffer; None -> local store
        # §7.16 strided put_nbi_warp drain: optional (cp,) int32 device PE table (flat-cp -> global PE)
        # so the consumer routes the warp put to peer `peer` at RUNTIME (one branch-free put call site
        # -> consistent align<16> -> no FFI prototype mismatch). None on every non-putwarp path.
        pe_table_dev: Optional[cute.Tensor] = None
        # Decoupled peer-store local-GMEM staging ring (grid_CTAs, ring_depth, epi_m, epi_n);
        # None unless configure_a2a_gemm_native(decoupled=True). The caller allocates it (bounded,
        # flat-in-N); the MMA warpgroup stages each epilogue tile here, the consumer warpgroup drains.
        ring: Optional[cute.Tensor] = None
        # DIAGNOSTIC (claim_instrument): a 2-elem int32 GMEM per-role drain counter ([0]=in-MMA/cwg
        # overlap, [1]=post-MMA/tail) and a grid-sized int64 GMEM per-CTA MMA-active-duration buffer
        # (globaltimer at MMA-WG entry vs tile-loop-exit). None unless _a2a_claim_instrument; read back
        # host-side after the kernel. Default None -> byte-identical.
        role_count: Optional[cute.Tensor] = None
        mma_time: Optional[cute.Tensor] = None
        # route-2 DYNSTRIP straddle store: a per-CTA GMEM TMA-descriptor workspace, (grid_CTAs, 16)
        # int64 = grid_CTAs x 128 B (one runtime-editable TMA descriptor per concurrent CTA -> no
        # cross-CTA races). Only the high-peer PARTIAL last strip uses its slot (copy_tensormap +
        # tensormap.replace.global row global_dim -> h_s + fences + cute.copy(tma_desc_ptr=)). None
        # unless configure_a2a_gemm_native(route2_dynstrip=True); default None -> byte-identical.
        route2_dynstrip_ws: Optional[cute.Tensor] = None
        # TRACK-1 GMEM edge-rider: the per-CTA GMEM cache RING (grid_CTAs, rd, epi_m, epi_n), dtype =
        # d_dtype (bf16). Straddle HIGH-peer remainders are cached here by the MMA warp + drained by
        # the spare consumer warpgroup (spec §2). rd == _a2a_ring_depth (the cache depth). Zero-init not
        # required (only slots < pcount are read, keyed by per-slot meta). None unless
        # configure_a2a_gemm_native(route2_gmem_cache=True); default None -> byte-identical.
        route2_cache_ws: Optional[cute.Tensor] = None
        # CLUSTER-DRAIN (Phase-3.1b): the per-cluster SYMMETRIC-heap staging buffer
        # (n_clusters, epi_m, N_j_loc) bf16. The cluster_n CTAs cooperatively write their
        # (epi_m, tile_n) column-slices into their cluster's row-packed (epi_m, N_j_loc) region;
        # cluster-rank-0 drains it with one coalesced put_warp per same-peer_i run. It is the IB
        # put SOURCE, so it MUST live on the symmetric heap. None unless
        # configure_a2a_gemm_native(cluster_drain=True); default None -> byte-identical.
        cluster_stage: Optional[cute.Tensor] = None

    # Regenerate EpilogueParams with the 2 peer fields appended to the mixin's extra
    # fields. __init_subclass__ only auto-regens when EpilogueParams is not already in a
    # base's __dict__ (the mixin auto-generated it), so set it explicitly. The peer fields
    # default None -> the flag-off path builds a valid params (and build_D_copy_fn falls
    # back to super()). _epi_ops / _epi_param_bases are inherited from the mixin.
    _extra_param_fields = tuple(GemmDefaultEpiMixin._extra_param_fields) + (
        ("peer_atoms", typing.Any, None),
        ("peer_tensors", typing.Any, None),
        (
            "peer_raw_tensors",
            typing.Any,
            None,
        ),  # arbitrary_n: raw peer memrefs (SIMT straddle fallback)
        (
            "route2_atoms",
            typing.Any,
            None,
        ),  # route-2 high-peer reduced (global_dim-shrink) atom family [r][s]
        ("route2_tensors", typing.Any, None),  # route-2 per-(peer,split) tiled coord views [r][s]
        ("decoupled_ring", typing.Any, None),  # local-GMEM staging ring (decoupled store)
        ("ring_store_atom", typing.Any, None),  # producer-TMA: S2G atom SMEM->GMEM-ring (I2b)
        ("ring_store_tensor", typing.Any, None),  # producer-TMA: the box-permuted ring view
        ("ring_load_atom", typing.Any, None),  # A′ consumer: G2S atom GMEM-ring->SMEM bounce
        ("ring_load_tensor", typing.Any, None),  # A′ consumer: the box-permuted ring view (load)
        (
            "recv_local",
            typing.Any,
            None,
        ),  # §7.16 strided drain: THIS rank's LOCAL recv (put_nbi_warp dst)
        ("pe_table_dev", typing.Any, None),  # §7.16 strided put_nbi_warp: (cp,) device PE table
        ("role_count_dev", typing.Any, None),  # DIAGNOSTIC: 2-elem int32 per-role drain counter
        (
            "mma_time_dev",
            typing.Any,
            None,
        ),  # DIAGNOSTIC: grid-sized int64 per-CTA MMA-active duration
        ("route2_dynstrip_ws", typing.Any, None),  # dynstrip: per-CTA GMEM TMA-descriptor workspace
        (
            "route2_cache_ws",
            typing.Any,
            None,
        ),  # TRACK-1: per-CTA GMEM cache ring (straddle-remainder drain)
        (
            "cluster_stage",
            typing.Any,
            None,
        ),  # CLUSTER-DRAIN (3.1b): symheap (n_clusters, epi_m, N_j_loc)
        (
            "cluster_stage_atom",
            typing.Any,
            None,
        ),  # CLUSTER-DRAIN: producer TMA-S2G atom SMEM->cluster stage
        (
            "cluster_stage_tensor",
            typing.Any,
            None,
        ),  # CLUSTER-DRAIN: the box-permuted cluster-stage view
    )
    EpilogueParams = _make_epi_params(
        GemmDefaultEpiMixin._epi_ops,
        _extra_param_fields,
        GemmDefaultEpiMixin._epi_param_bases,
    )

    def epi_to_underlying_arguments(self, args, *, loc=None, ip=None):
        # Mixin builds the standard params (alpha/beta/vecs/sr_seed); then attach peers.
        p = super().epi_to_underlying_arguments(args, loc=loc, ip=ip)
        recv = getattr(args, "recv", None)
        if not self._a2a_enabled or recv is None:
            return p
        peer_raw_tensors = None
        route2_atoms = None
        route2_tensors = None
        # INVARIANT: reaching here implies gemm-native. `_a2a_enabled` is set True at exactly two
        # sites on this class (gemm_sm90_a2a.py:596 configure_a2a_gemm_native, :1227
        # configure_a2a_sharded) and BOTH set `_a2a_gemm_native = True` on the very next line; the
        # ctor default (:228) leaves `_a2a_enabled` False, which the guard above already returned on.
        # The old `else` arm here (the token-major L=1 store via _build_peer_store_atoms) was
        # therefore unreachable — the plain `configure_a2a` its docstrings named no longer exists on
        # this class — and was removed. Asserted rather than silently assumed.
        assert self._a2a_gemm_native, (
            "GemmA2ASm90: _a2a_enabled without _a2a_gemm_native — a new configure entry was added "
            "that does not set gemm-native; the non-native store arm no longer exists."
        )
        built = self._build_peer_store_atoms_gemm_native(
            recv, self.epi_smem_layout_staged, self.epi_tile
        )
        # arbitrary_n returns a 3rd list (raw peer memrefs for the SIMT straddle fallback);
        # route2_straddle returns 2 MORE (the per-(peer,split) reduced atom/tensor family).
        if len(built) == 5:
            atoms, tensors, peer_raw_tensors, route2_atoms, route2_tensors = built
        elif len(built) == 3:
            atoms, tensors, peer_raw_tensors = built
        else:
            atoms, tensors = built
        # Decoupled store: forward the caller's local-GMEM staging ring into the params so both
        # the producer copy_fn and the consumer warpgroup (which receive epi_params) can reach it.
        ring = getattr(args, "ring", None)
        # Producer-TMA (I2b): build a TMA-S2G atom retargeted at the GMEM ring (the SAME async store
        # the single-device GEMM uses, just a different GMEM destination) so the MMA warp's ring-write
        # is O(1) instead of an O(tile) SIMT copy. Built here (host trace) and ridden on epi_params.
        ring_store_atom, ring_store_tensor = None, None
        ring_load_atom, ring_load_tensor = None, None
        if getattr(self, "_a2a_producer_tma", False) and ring is not None:
            ring_store_atom, ring_store_tensor = self._build_ring_store_atom(
                ring, self.epi_smem_layout_staged, self.epi_tile
            )
            if getattr(self, "_a2a_consumer_double_tma", False) or getattr(
                self, "_a2a_tail_double_tma", False
            ):
                # A′ / §7.16m tail double-TMA: G2S atom GMEM-ring -> SMEM bounce (the consumer's /
                # tail's first hop). Epi-tile-boxed (the tile-wide ring slot is drained as
                # n_sub_per_tile epi-subtile G2S/S2G copies reusing the coupled peer_atoms).
                ring_load_atom, ring_load_tensor = self._build_ring_load_atom(
                    ring, self.epi_smem_layout_staged, self.epi_tile
                )
        # §7.16 STRIDED drain: forward THIS rank's LOCAL recv view so the consumer can issue
        # put_nbi_warp(local_recv[dst_coord], src, peer_pe) (the de-risked proto/a2a.py mechanism:
        # warp put over the LOCAL symmetric view, peer selected by PE). The ONLY reader is the strided
        # consumer_warpgroup_role, which is const_expr-elided at the all-P2P COLLAPSE
        # (_num_extra_warpgroups()==0 when not has_ib_peers). #57 Stage-1.1 completion: AND-in
        # has_ib_peers so recv_local + its downstream pe_table_dev param (:below) also NULL on the collapse
        # -- the 2 INTERNALLY-forwarded dead leaves the caller-side make_differential_epi_args gate couldn't
        # reach (safe: no collapse-reachable path derefs them; design-E fall-through == pe_aligned, which
        # already runs with both None). Mixed / real-IB (has_ib_peers True) unchanged -> byte-identical.
        _want_recv_local = getattr(self, "_a2a_consumer_strided", False) and getattr(
            self, "_a2a_has_ib_peers", True
        )
        recv_local = recv if _want_recv_local else None
        pe_table_dev = getattr(args, "pe_table_dev", None) if _want_recv_local else None
        # The claim_instrument diagnostic probes and the route-2 dynstrip / GMEM-cache / halo-cache
        # workspaces are always absent: `4ab8f05` (2026-07-04) "collapse back A2A store surface to
        # {pe_aligned, ...}" deleted the kwargs that set them, so nothing in fold_cp_ops/, tests/ or
        # benchmark/ assigns `_a2a_claim_instrument`, `_a2a_route2_dynstrip`, `_a2a_route2_gmem_cache`
        # or `_a2a_halo_cache`. The gates that read them were residue of that removal and are gone;
        # these four params stay (always None) because their in-kernel readers const_expr-elide on
        # None, so the traced kernel is unchanged.
        role_count_dev = None
        mma_time_dev = None
        route2_dynstrip_ws = None
        route2_cache_ws = None
        # CLUSTER-DRAIN (3.1b): forward the per-cluster symheap staging buffer + build its producer
        # TMA-S2G atom (SMEM epi-box -> the cluster's (epi_m, N_j_loc) region), mirroring ring_store_atom.
        # None unless cluster_drain -> byte-identical.
        cluster_stage = (
            getattr(args, "cluster_stage", None)
            if getattr(self, "_a2a_cluster_drain", False)
            else None
        )
        cluster_stage_atom, cluster_stage_tensor = None, None
        if getattr(self, "_a2a_cluster_drain", False) and cluster_stage is not None:
            cluster_stage_atom, cluster_stage_tensor = self._build_cluster_stage_atom(
                cluster_stage, self.epi_smem_layout_staged, self.epi_tile
            )
        return dataclasses.replace(
            p,
            peer_atoms=atoms,
            peer_tensors=tensors,
            peer_raw_tensors=peer_raw_tensors,
            route2_atoms=route2_atoms,
            route2_tensors=route2_tensors,
            decoupled_ring=ring,
            ring_store_atom=ring_store_atom,
            ring_store_tensor=ring_store_tensor,
            ring_load_atom=ring_load_atom,
            ring_load_tensor=ring_load_tensor,
            recv_local=recv_local,
            pe_table_dev=pe_table_dev,
            role_count_dev=role_count_dev,
            mma_time_dev=mma_time_dev,
            route2_dynstrip_ws=route2_dynstrip_ws,
            route2_cache_ws=route2_cache_ws,
            cluster_stage=cluster_stage,
            cluster_stage_atom=cluster_stage_atom,
            cluster_stage_tensor=cluster_stage_tensor,
        )

    def make_differential_epi_args(
        self, recv, *, ring=None, cluster_stage=None, pe_table_dev=None, **extra
    ):
        """#57 Stage-1.1 caller-side dead-descriptor gate: build the differential / coalesce_dyn /
        cluster_drain :class:`EpilogueArguments` with the IB-only descriptors (``ring``, ``cluster_stage``)
        NULLED whenever the run is an all-P2P COLLAPSE (``self._a2a_has_ib_peers is False`` — cp<=8, a single
        NVLink domain with no cross-node peer). On the mixed / real-IB path (``has_ib_peers=True``) it is a
        PASS-THROUGH: byte-identical to the explicit ``EpilogueArguments(...)`` the callers built before.

        WHY caller-side (a build-side gate cannot fix this). ``EpilogueArguments`` is an ``@mlir_namedtuple``
        — ``cute.compile`` FLATTENS it and turns EVERY ``cute.Tensor`` leaf into a compiled-signature input,
        INCLUDING ``ring`` / ``cluster_stage``, regardless of what :meth:`epi_to_underlying_arguments` forwards
        or drops. So a live ``ring`` tensor survives as a DEAD descriptor in the collapsed kernel's signature
        (occupancy-identical, but NOT byte-identical to pe_aligned). Nulling the top-level leaf HERE removes it
        AND cascades: the derived ``ring_store_atom`` / ``cluster_stage_atom`` in
        :meth:`epi_to_underlying_arguments` are built only ``if <leaf> is not None``, so the whole descriptor
        chain drops → the collapsed kernel's flattened signature == pe_aligned's.

        Keys ONLY on ``_a2a_has_ib_peers`` (host-known after :meth:`configure_a2a_gemm_native`; ``getattr``
        default True so an unconfigured / non-collapse obj passes descriptors through unchanged) — NEVER on the
        tensor/stage shape, so it stays robust to any later ``cluster_stage`` restructure. ``recv`` /
        ``pe_table_dev`` are common to every store variant (incl. pe_aligned) → never gated. ``recv`` is passed
        through as-is (mark it 16-B before the call); ``**extra`` overrides any other field (e.g. alpha/beta or
        a flag-gated diagnostic buffer) — do NOT route ``ring`` / ``cluster_stage`` through ``extra`` (they are
        the gated named params). Thin wrapper: does not touch EpilogueArguments' fields or epi_to_underlying_.
        """
        has_ib = getattr(self, "_a2a_has_ib_peers", True)
        fields = dict(
            alpha=None,
            beta=None,
            mRowVecBroadcast=None,
            mColVecBroadcast=None,
            add_to_output=False,
            rounding_mode=None,
            sr_seed=None,
            recv=recv,
            ring=(ring if has_ib else None),
            cluster_stage=(cluster_stage if has_ib else None),
            pe_table_dev=pe_table_dev,
        )
        fields.update(extra)
        return self.EpilogueArguments(**fields)
