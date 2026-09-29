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
"""The SM90 GEMM's operand-load layer: descriptors, addressing, and the producer warpgroup.

Everything between "here are the operand tensors" and "a k-tile is staged in SMEM and its
mbarrier has arrived". The MMA layer consumes what this produces and never looks at a global
tensor.

**This is where the A2A and composite-K derivations extend.** Six methods default to the identity
and exist to be overridden:

===========================  =================================================================
``select_batch``             which batch of a batched operand this tile reads
``mainloop_remap_mA/mB``     shift the operand's origin (an A2A per-peer base row/column)
``_remap_A/B_operand_layout``the view the TMA DESCRIPTOR is built from, which may differ
``_gA/_gB_local_tile``       how a tile is cut out, including a composite ``RestK``
``_k_tile_cnt``              the mainloop trip count
===========================  =================================================================

``_k_tile_cnt`` carries a coupling worth stating once: the MMA layer uses the same value to size
its drain loop. Overriding it here without the consumer following suit leaves the producer
staging more k-tiles than anyone consumes, which is a **pipeline deadlock**, not a wrong answer.
"""

from typing import Tuple, Type, Callable, Optional, Union, Literal  # noqa: F401
import math  # noqa: F401

import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
from cutlass.cute.nvgpu import cpasync, warp, warpgroup  # noqa: F401
import cutlass.utils.hopper_helpers as sm90_utils  # noqa: F401
from cutlass import Int32, Float32, Float16, Boolean, const_expr  # noqa: F401
from cutlass.utils import LayoutEnum  # noqa: F401

from fold_cp_ops._internal.pipeline import make_pipeline_state
import fold_cp_ops._internal.copy_utils as copy_utils
import fold_cp_ops._internal.sm90_utils as fold_cp_ops_sm90_utils
from fold_cp_ops._internal.rounding import RoundingMode  # noqa: F401


class GemmSm90LoadMixin:
    @cute.jit
    def select_batch(self, mX: cute.Tensor, batch_idx: Int32) -> cute.Tensor:
        """Select one batch of a batched operand, rank-generally.

        Replaces ``VarlenManager.offset_batch_*``'s dense branch, which is all that survives now
        that ragged M/K are gone. The select is written rank-generally rather than as a plain
        ``mX[None, None, batch_idx]`` so a composite-K operand keeps its trailing mode: the A2A
        route-2 K-hoist passes a 4-D ``(M, Xg_pad, L, cp)`` A and needs the ``cp`` mode preserved.

        Args:
            mX: A batched tensor of rank >= 3 whose mode 2 is the batch (L) axis. Rank 2 or lower
                indexes out of range -- there is no batch mode to select and the call raises rather
                than silently returning the whole tensor.
            batch_idx: Which batch, in ``[0, L)``. Out of range reads past the tensor; it is a
                coordinate, not a checked index.

        Returns:
            A view -- no copy -- of that batch, with the batch mode dropped and every mode past it
            retained.
        """
        return mX[(None, None, batch_idx) + (None,) * (cute.rank(mX) - 3)]

    def mainloop_remap_mA(
        self,
        mA_mk: cute.Tensor,
        tile_coord_mnkl: cute.Coord,
        mA_mkl: cute.Tensor = None,
        batch_idx: Int32 = None,
    ) -> cute.Tensor:
        """Overridable A-row remap hook for the mainloop TMA-load (extract-method seam).

        DEFAULT: pass-through (returns ``mA_mk`` unchanged) -> the stock GEMM A-load is
        BYTE-IDENTICAL (zero device cost; this is a host-trace identity). A subclass may
        override it to shift A's row origin for a custom M-tiling (e.g. the A2A
        PE-boundary-aware per-peer tiling, where ``tile_coord_mnkl[0]`` is a per-peer linear
        tile index and A must be read from ``peer*N_loc + k*tile_m`` instead of
        ``m_linear*tile_m``). Mirrors the ``build_D_copy_fn`` extract-method precedent.

        Args:
            mA_mk: A after :meth:`select_batch`, i.e. the 2-D ``(M, K)`` view for this batch.
            tile_coord_mnkl: The scheduler's tile coordinate.
            mA_mkl: The PRE-selection A, still carrying its batch (and any token-grid) modes.
                Defaulted and **unused by this default body**, so the local path is unchanged;
                it exists because a token-grid subclass has to do its own batch/Y/B selection
                from the un-collapsed tensor and cannot recover those modes from ``mA_mk``.
            batch_idx: The batch this tile belongs to, for the same reason.

        Why the extra arguments live HERE and not on ``select_batch``
            ``select_batch`` is operand-GENERIC -- the loader calls it for ``mA``, ``mB`` and
            ``mB2`` -- so a token-grid override placed there would apply the A operand's
            unravel to the B operands as well, which is silently wrong output rather than a
            failure. This hook is A-SPECIFIC (``mainloop_remap_mB`` is its N-axis twin), so it
            is the only seam that both sees the right operand and can be given the right
            information. Adding parameters a default body ignores costs nothing at trace time.
        """
        return mA_mk

    def mainloop_remap_mB(self, mB_nk: cute.Tensor, tile_coord_mnkl: cute.Coord) -> cute.Tensor:
        """Overridable B-col remap hook for the mainloop TMA-load (extract-method seam).

        DEFAULT: pass-through (returns ``mB_nk`` unchanged) -> BYTE-IDENTICAL (host-trace
        identity, zero device cost). The N-axis analogue of :meth:`mainloop_remap_mA`: a
        subclass may override it to shift B's N-origin for a custom N-tiling (the A2A 2-D
        per-peer-j tiling, where ``tile_coord_mnkl[1]`` is a per-peer-j linear tile index and
        B must be read from ``peer_j*N_j_loc + kj*tile_n`` instead of ``n_linear*tile_n``).
        """
        return mB_nk

    def _remap_A_operand_layout(self, mA: cute.Tensor, epi_args=None) -> cute.Tensor:
        """Overridable A-operand LAYOUT hook for the ATOM build (extract-method seam; distinct from the
        shape-PRESERVING ``mainloop_remap_mA`` row-shift). DEFAULT: pass-through (returns ``mA`` unchanged)
        -> the stock GEMM's A TMA-atom build is BYTE-IDENTICAL (host-trace identity, zero device cost). A
        subclass may return a HIGHER-RANK A view so the ``(BLK_M, BLK_K)`` box tiles the first two modes and
        the trailing modes ride as untiled TMA modes -- e.g. the route2 composite-K read presents A as 4-D
        ``(M, Xg_pad, cp, L)`` (cp = the resharded contraction blocks, stride ``N_j*Xg_pad``) so K reads
        ``(cp, Xg_pad)`` composite. Mirrors the staged front's ``_remap_A_operand_layout`` (rank-2-M hoist)."""
        return mA

    def _remap_B_operand_layout(self, mB: cute.Tensor, epi_args=None) -> cute.Tensor:
        """Overridable B-operand LAYOUT hook for the ATOM build (extract-method seam; the B-symmetric
        twin of :meth:`_remap_A_operand_layout`). DEFAULT: pass-through (returns ``mB`` unchanged) -> the
        stock GEMM's B TMA-atom build is BYTE-IDENTICAL (host-trace identity, zero device cost). A subclass
        may return a HIGHER-RANK B view so the ``(BLK_N, BLK_K)`` box tiles the first two modes and the
        trailing modes ride as untiled TMA modes -- e.g. the route2 composite-K read presents B as 4-D
        ``(N, Xg_pad, L, cp)`` so K reads ``(cp, Xg_pad)`` composite, identical to the A operand (the back
        einsum's A and B are BOTH token operands from the front recv, so both need the K-hoist)."""
        return mB

    def _gA_local_tile(self, mA_mk: cute.Tensor, tile_coord_mnkl: cute.Coord) -> cute.Tensor:
        """Overridable A ``local_tile`` hook for the mainloop partition (extract-method seam). DEFAULT: the
        stock ``(bM, bK, RestK)`` partition -> BYTE-IDENTICAL. A subclass may produce a COMPOSITE RestK from
        a higher-rank ``mA_mk`` -- e.g. route2 composite-K tiles ``(M, Xg_pad)`` leaving cp -> ``(bM, bK,
        nt_within, cp)`` grouped so the K-loop unravels ``(within, rank)`` (the TMA coord addresses the
        cp-mode). Mirrors the staged front's ``_gA_local_tile``."""
        return cute.local_tile(
            mA_mk,
            cute.select(self.cta_tile_shape_mnk, [0, 2]),
            (tile_coord_mnkl[0], None),
        )

    def _gB_local_tile(self, mB_nk: cute.Tensor, tile_coord_mnkl: cute.Coord) -> cute.Tensor:
        """Overridable B ``local_tile`` hook for the mainloop partition (extract-method seam; the
        B-symmetric twin of :meth:`_gA_local_tile`). DEFAULT: the stock ``(bN, bK, RestK)`` partition ->
        BYTE-IDENTICAL. A subclass may produce a COMPOSITE RestK from a higher-rank ``mB_nk`` -- e.g.
        route2 composite-K tiles ``(N, Xg_pad)`` leaving cp -> ``(bN, bK, nt_within, cp)`` grouped so the
        K-loop unravels ``(within, rank)`` (the TMA coord addresses the cp-mode)."""
        return cute.local_tile(
            mB_nk,
            cute.select(self.cta_tile_shape_mnk, [1, 2]),
            (tile_coord_mnkl[1], None),
        )

    def _k_tile_cnt(self, len_k):
        """Overridable K-loop trip-count hook (extract-method seam). DEFAULT: ceil_div(len_k, BLK_K) ->
        BYTE-IDENTICAL. A subclass may return a COMPOSITE trip -- e.g. route2 composite-K iterates
        cp*ceil_div(Xg_pad, BLK_K) = cp*nt_within tiles (unraveling (within, rank) for the rank-2 K read),
        since a 4-D A presents K-shape Xg_pad (one rank) but the reduction spans all cp rank-blocks."""
        return cute.ceil_div(len_k, self.cta_tile_shape_mnk[2])

    def make_mainloop_tma(self, mA, mB, mB2=None, epilogue_args=None):
        """Build the TMA atoms and descriptor views for the mainloop operands.

        The AB pipeline's per-stage transaction count is ``GemmSm90.num_tma_load_bytes``, derived
        from the same staged SMEM layouts these descriptors are built from. It used to be assigned
        here; deriving it instead is what makes "the count and the descriptors agree" structural
        rather than a convention -- a count that disagrees with what the descriptors move is a
        HANG, not a wrong answer.

        **The A and B descriptors go through ``_remap_A/_remap_B_operand_layout`` first**, which
        default to the identity. That is the seam a subclass uses to point the descriptor at a
        different view of the same memory -- the A2A per-peer base row, a composite-K hoist --
        without touching this method.

        Args:
            mA: A operand, strides already re-declared. Must be rank 3 ``(M, K, L)`` unless a
                subclass's remap hook says otherwise.
            mB: B operand, ``(N, K, L)``.
            mB2: Second B operand for the two-tensor gated load, or None.

        Returns:
            ``(tma_atom_a, tma_tensor_a, tma_atom_b, tma_tensor_b, tma_atom_b2, tma_tensor_b2)``.
            The two ``b2`` entries are None unless ``two_tensor_B``. **The kernel must partition
            against the returned ``tma_tensor_*``, not against the inputs** -- the descriptor has
            its own view and coordinates into the original mean something else.

        Note:
            Requires ``_setup_attributes`` to have run: the staged SMEM layouts and the CTA tile
            shape's K dimension are read here and are None before that.
        """
        a_smem_layout = cute.slice_(self.a_smem_layout_staged, (None, None, 0))
        b_smem_layout = cute.slice_(self.b_smem_layout_staged, (None, None, 0))
        tma_atom_a, tma_tensor_a = self._make_tma_atoms_and_tensors(
            self._remap_A_operand_layout(mA, epilogue_args),
            a_smem_layout,
            (self.cta_tile_shape_mnk[0], self.cta_tile_shape_mnk[2]),
            self.cluster_shape_mnk[1],
        )
        tma_atom_b2, tma_tensor_b2 = None, None
        if const_expr(not self.two_tensor_B):
            tma_atom_b, tma_tensor_b = self._make_tma_atoms_and_tensors(
                self._remap_B_operand_layout(mB),
                b_smem_layout,
                (self.cta_tile_shape_mnk[1], self.cta_tile_shape_mnk[2]),
                self.cluster_shape_mnk[0],
            )
        else:
            # One TMA atom per (Wp up, Wg gate) tensor, each with a (G, tile_K) SMEM box. cluster_M==1
            # was asserted in bind_operand_types so B is never multicast here -> plain G2S atoms over
            # (N,K,L) tensors. B is k-major (K contiguous): the smem-layout-atom swizzle is over the K
            # (inner) mode, so a standalone (G, tile_K) smem layout is bit-compatible with each G-row
            # stripe of the full (tile_N, tile_K) B smem tile (the destination sub-blocks).
            G = self.chunk_g
            b_box_smem_layout = fold_cp_ops_sm90_utils.make_smem_layout(
                self.b_dtype, self.b_layout, (G, self.cta_tile_shape_mnk[2])
            )
            tma_atom_b, tma_tensor_b = self._make_tma_atoms_and_tensors(
                mB, b_box_smem_layout, (G, self.cta_tile_shape_mnk[2]), 1
            )
            tma_atom_b2, tma_tensor_b2 = self._make_tma_atoms_and_tensors(
                mB2, b_box_smem_layout, (G, self.cta_tile_shape_mnk[2]), 1
            )
        return (
            tma_atom_a,
            tma_tensor_a,
            tma_atom_b,
            tma_tensor_b,
            tma_atom_b2,
            tma_tensor_b2,
        )

    @staticmethod
    def _make_tma_atoms_and_tensors(
        tensor: cute.Tensor,
        smem_layout: cute.ComposedLayout,
        smem_tile: Tuple[int, int],
        mcast_dim: int,
    ) -> Tuple[cute.CopyAtom, cute.Tensor]:
        """Create TMA atoms and tensors for input tensors.

        :param tensor: Input tensor (A or B)
        :type tensor: cute.Tensor
        :param smem_layout: Shared memory layout for the tensor
        :type smem_layout: cute.ComposedLayout
        :param smem_tile: Shared memory tile shape
        :type smem_tile: Tuple[int, int]
        :param mcast_dim: Multicast dimension
        :type mcast_dim: int

        :return: TMA atom and tensor
        :rtype: Tuple[cute.CopyAtom, cute.Tensor]
        """
        op = (
            cpasync.CopyBulkTensorTileG2SOp()
            if mcast_dim == 1
            else cpasync.CopyBulkTensorTileG2SMulticastOp()
        )
        tma_atom, tma_tensor = cpasync.make_tiled_tma_atom(
            op,
            tensor,
            smem_layout,
            smem_tile,
            num_multicast=mcast_dim,
        )
        return tma_atom, tma_tensor

    @cute.jit
    def load_AB(
        self,
        ab_pipeline: cutlass.pipeline.PipelineAsync,
        ab_producer_state: cutlass.pipeline.PipelineState,
        copy_A: Optional[Callable],
        copy_B: Callable,
        k_tile_cnt: Int32,
        # These are for Sm100 blockscaled gemm
        copy_SFA: Optional[Callable] = None,
        copy_SFB: Optional[Callable] = None,
    ) -> cutlass.pipeline.PipelineState:
        """Producer loop: stage every k-tile of A and B into SMEM through the AB pipeline.

        One iteration per k-tile: acquire an empty stage, issue the TMA copies into it, commit. The
        commit is implicit for TMA -- the copy itself arrives on the barrier -- so there is no
        explicit producer_commit here.

        Args:
            ab_pipeline: The A/B pipeline. Its stage count bounds how far ahead this runs.
            ab_producer_state: The producer state, advanced once per k-tile and returned.
            copy_A: Closure issuing A's copy, or None when A is loaded elsewhere (the gather path,
                or a fused kernel that produces A in registers).
            copy_B: Closure issuing B's copy. Never None.
            k_tile_cnt: Number of k-tiles for this work tile.
            src_idx_prev: Starting source index, for a persistent kernel resuming mid-operand.
            is_tma_warp: Whether this warp issues the TMA arrive. Exactly one producer warp may
                pass True; more over-arrives the barrier and the consumer reads unwritten SMEM.

        Returns:
            The advanced producer state, for the next work tile.
        """
        blockscaled = const_expr(copy_SFA is not None)
        if const_expr(blockscaled):
            assert copy_SFB is not None
        # Peek (try_wait) AB buffer empty for k_block = prefetch_k_tile_cnt
        peek_ab_empty_status = Boolean(True)
        if 0 < k_tile_cnt:
            peek_ab_empty_status = ab_pipeline.producer_try_acquire(ab_producer_state)
        # TMA load
        for k_tile in cutlass.range(k_tile_cnt, unroll=1):
            # Wait for A/B buffers to be empty before loading into them
            # Also sets the transaction barrier for the A/B buffers
            ab_pipeline.producer_acquire(ab_producer_state, peek_ab_empty_status)
            tma_bar_ptr = ab_pipeline.producer_get_barrier(ab_producer_state)
            smem_idx = ab_producer_state.index
            if const_expr(copy_A is not None):
                copy_A(k_tile, smem_idx, tma_bar_ptr=tma_bar_ptr)
            copy_B(k_tile, smem_idx, tma_bar_ptr=tma_bar_ptr)
            if const_expr(blockscaled):
                copy_SFA(k_tile, smem_idx, tma_bar_ptr=tma_bar_ptr)
                copy_SFB(k_tile, smem_idx, tma_bar_ptr=tma_bar_ptr)
            # Mainloop pipeline's producer commit is a NOP
            ab_pipeline.producer_commit(ab_producer_state)
            ab_producer_state.advance()
            peek_ab_empty_status = Boolean(True)
            if k_tile + 1 < k_tile_cnt:
                peek_ab_empty_status = ab_pipeline.producer_try_acquire(ab_producer_state)
        return ab_producer_state

    @cute.jit
    def producer_warpgroup_role(
        self,
        warp_idx,
        cluster_layout_mnk,
        mA_mkl,
        mB_nkl,
        mB2_nkl,
        tma_atom_a,
        tma_atom_b,
        tma_atom_b2,
        epilogue_params,
        tile_sched_params,
        TileSchedulerCls: cutlass.Constexpr[Callable],
        ab_pipeline,
        len_k,
        sA,
        sB,
        storage,
    ):
        """The AB-load warpgroup: walk work tiles, address the operands, issue the TMA loads.

        Runs only on ``warp_idx >= ab_load_warp_id``; the guard is kept inside so this reads as a
        complete role and a subclass can override it without re-deriving the split. It lowers its
        register budget first (the load warps need few), then loops the scheduler, and for each
        tile partitions A and B through the overridable addressing seams before handing the copy
        closures to :meth:`load_AB`.

        **The addressing seams are the extension point that matters.** ``mainloop_remap_mA/mB``,
        ``_gA/_gB_local_tile`` and ``_k_tile_cnt`` each default to the identity, so a subclass
        redirects where operands are read from -- a per-peer A2A base row, a composite-K hoist --
        without touching this loop. ``_k_tile_cnt`` must be overridden in lockstep with the
        consumer's use of it: the producer loading more k-tiles than the consumer drains is a
        pipeline deadlock, not a wrong answer.

        Args:
            warp_idx: Warp-uniform index; the role guard reads it.
            ctx: The :class:`_KernelContext` from :meth:`kernel_prologue`.
            cluster_layout_mnk: Cluster layout, for the multicast masks.
            mA_mkl: A operand as the TMA descriptor sees it.
            mB_nkl: B operand.
            mB2_nkl: Second B operand for the two-tensor gated load, or None.
            tma_atom_a: TMA atom for A.
            tma_atom_b: TMA atom for B.
            tma_atom_b2: TMA atom for the second B, or None.
            epilogue_params: Forwarded to the consumer-role hook for a subclass's extra warpgroup.
            tile_sched_params: Likewise.

        Returns:
            None. Its effect is the SMEM staging the MMA warpgroups consume.
        """
        if warp_idx >= self.ab_load_warp_id:
            # _skip_warpgroup_reg_realloc (const_expr, default absent->False): a subclass sets
            # this to SUPPRESS the warpgroup register realloc when the extra-WG layout makes the
            # paired MMA setmaxregister_increase unreachable from the launch REGCOUNT baseline
            # (the §7.16m idle-extra-WG .inc deadlock). Default OFF -> byte-identical.
            if const_expr(not getattr(self, "_skip_warpgroup_reg_realloc", False)):
                cute.arch.setmaxregister_decrease(self.num_regs_load)
            if (
                warp_idx >= self.ab_load_warp_id
                and warp_idx < self.ab_load_warp_id + self.num_ab_load_warps
            ):
                is_tma_warp = self.num_ab_load_warps == 1 or warp_idx == self.ab_load_warp_id
                # Get mcast mask
                cta_rank_in_cluster = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
                block_in_cluster_coord_mnk = cluster_layout_mnk.get_flat_coord(cta_rank_in_cluster)
                a_mcast_mask = cute.make_layout_image_mask(
                    cluster_layout_mnk, block_in_cluster_coord_mnk, mode=1
                )
                b_mcast_mask = cute.make_layout_image_mask(
                    cluster_layout_mnk, block_in_cluster_coord_mnk, mode=0
                )
                a_mcast_mask = a_mcast_mask if self.is_a_mcast else 0
                b_mcast_mask = b_mcast_mask if self.is_b_mcast else 0

                # Persistent tile scheduling loop
                is_scheduler_warp = self.num_ab_load_warps == 1 or warp_idx == self.ab_load_warp_id
                if const_expr(cute.size(cluster_layout_mnk) > 1):
                    is_scheduler_warp = is_scheduler_warp and cute.arch.block_idx_in_cluster() == 0
                tile_scheduler = TileSchedulerCls()
                work_tile = tile_scheduler.initial_work_tile_info()
                ab_producer_state = make_pipeline_state(
                    pipeline.PipelineUserType.Producer, self.ab_stage
                )
                while work_tile.is_valid_tile:
                    tile_coord_mnkl = work_tile.tile_idx
                    batch_idx = tile_coord_mnkl[3]
                    # Local_tile partition global tensors
                    mA_mk = self.select_batch(mA_mkl, batch_idx)
                    # Overridable A-row remap hook (extract-method; DEFAULT byte-identical = returns
                    # mA_mk unchanged). The A2A PE-boundary-aware per-peer M-tiling child overrides it
                    # to shift A's rows so local_tile(.,(m_linear,None)) reads from the per-peer
                    # base_m; the stock GEMM's default is a pure pass-through (zero device cost).
                    mA_mk = self.mainloop_remap_mA(
                        mA_mk, tile_coord_mnkl, mA_mkl=mA_mkl, batch_idx=batch_idx
                    )
                    # (bM, bK, RestK) — extract-method seam (DEFAULT byte-identical); a subclass may
                    # produce a COMPOSITE RestK (route2 composite-K K-hoist). See _gA_local_tile.
                    gA_mk = self._gA_local_tile(mA_mk, tile_coord_mnkl)
                    # (bN, bK, RestK)
                    gB_nk = None
                    gWp_blk, gWg_blk = None, None
                    if const_expr(not self.two_tensor_B):
                        # Overridable B-col remap hook (extract-method; DEFAULT byte-identical). The A2A
                        # 2-D PE-boundary-aware per-peer N-tiling child shifts B's N-origin so
                        # local_tile(.,(n_linear,None)) reads from the per-peer-j base (mirror of
                        # mainloop_remap_mA on the M axis); the stock GEMM's default is a pass-through.
                        mB_nk = self.mainloop_remap_mB(
                            self.select_batch(mB_nkl, batch_idx), tile_coord_mnkl
                        )
                        # (bN, bK, RestK) — extract-method seam (DEFAULT byte-identical); a subclass may
                        # produce a COMPOSITE RestK (route2 composite-K K-hoist, B-symmetric to A). See
                        # _gB_local_tile.
                        gB_nk = self._gB_local_tile(mB_nk, tile_coord_mnkl)
                    else:
                        # Two-tensor: tile Wp (up) and Wg (gate) by (G, tile_K) along their postact-N
                        # dim. nblk = tile_N//(2G) blocks come from the contiguous postact-N range
                        # [tile_coord_n*nblk : ...] in G-block units. -> (G, tile_K, nblk, RestK).
                        G = const_expr(self.chunk_g)
                        nblk = const_expr(self.cta_tile_shape_mnk[1] // (2 * G))
                        gk = (G, self.cta_tile_shape_mnk[2])
                        gWp_all = cute.local_tile(
                            self.select_batch(mB_nkl, batch_idx), gk, (None, None)
                        )  # (G, tile_K, N//G, RestK)
                        gWg_all = cute.local_tile(
                            self.select_batch(mB2_nkl, batch_idx), gk, (None, None)
                        )
                        base = tile_coord_mnkl[1] * nblk
                        gWp_blk = cute.domain_offset((0, 0, base, 0), gWp_all)
                        gWg_blk = cute.domain_offset((0, 0, base, 0), gWg_all)
                    #  TMA load A partition_S/D
                    copy_A, _, _ = copy_utils.tma_get_copy_fn(
                        tma_atom_a,
                        cta_coord=block_in_cluster_coord_mnk[1],
                        cta_layout=cute.make_layout(
                            cute.slice_(cluster_layout_mnk, (0, None, 0)).shape
                        ),
                        src_tensor=gA_mk,
                        dst_tensor=sA,
                        mcast_mask=a_mcast_mask,
                    )
                    # TMA load B partition_S/D
                    if const_expr(not self.two_tensor_B):
                        copy_B, _, _ = copy_utils.tma_get_copy_fn(
                            tma_atom_b,
                            cta_coord=block_in_cluster_coord_mnk[0],
                            cta_layout=cute.make_layout(
                                cute.slice_(cluster_layout_mnk, (None, 0, 0)).shape
                            ),
                            src_tensor=gB_nk,
                            dst_tensor=sB,
                            mcast_mask=b_mcast_mask,
                        )
                    else:
                        # sB zipped_divide by (G, tile_K) -> ((G,tile_K), (nsub, 1, STAGE)) so the box
                        # is a single grouped mode 0 (what tma_partition wants); selecting a sub-block
                        # then leaves ((G,tile_K), STAGE).  gWp_blk/gWg_blk are already (G,tile_K,
                        # nblk_all, RestK) (local_tile form), indexed per block.  nsub = tile_N//G.
                        G = const_expr(self.chunk_g)
                        nblk = const_expr(self.cta_tile_shape_mnk[1] // (2 * G))
                        sB_z = cute.zipped_divide(sB, (G, self.cta_tile_shape_mnk[2]))
                        copy_B = copy_utils.tma_get_chunked_B_copy_fn(
                            tma_atom_b, tma_atom_b2, gWp_blk, gWg_blk, sB_z, nblk
                        )
                    # extract-method seam (DEFAULT byte-identical = ceil_div(len_k, BLK_K)); the composite-K
                    # subclass returns cp*nt_within (rank-2 K-loop trip). See _k_tile_cnt.
                    k_tile_cnt = self._k_tile_cnt(len_k)
                    ab_producer_state = self.load_AB(
                        ab_pipeline, ab_producer_state, copy_A, copy_B, k_tile_cnt
                    )
                    tile_scheduler.advance_to_next_work(is_scheduler_warp=is_scheduler_warp)
                    work_tile = tile_scheduler.get_current_work()
                    # End of persistent scheduler loop
                if const_expr(self.pingpong):
                    # Need to write the tile_idx to smem for the next WG in the pingpong mode
                    if is_scheduler_warp:
                        tile_scheduler.write_work_tile_to_smem(work_tile)
                    work_tile = tile_scheduler.get_current_work()
                ab_pipeline.producer_tail(ab_producer_state)
                if is_scheduler_warp:
                    tile_scheduler.producer_tail()
            elif const_expr(self._num_extra_warpgroups() > 0 or self._drain_on_producer_wg()):
                # The consumer-role warps: warp_idx in [ab_load_warp_id + num_ab_load_warps,
                # threads_per_cta//32) -- i.e. either the dedicated EXTRA warpgroup (warps 12-15) OR,
                # when _drain_on_producer_wg(), the PRODUCER warpgroup's spare warps 9-11 (no extra WG,
                # smaller block). They already did setmaxregister_decrease(num_regs_load) above; route
                # them to the consumer-role hook (default no-op). Lives OUTSIDE the AB-load inner-if, so
                # it never touches the AB pipeline / scheduler. Gated so the branch does not exist when
                # neither a dedicated extra WG nor a producer-WG drain is requested.
                if warp_idx >= self.ab_load_warp_id + self.num_ab_load_warps:
                    self.consumer_warpgroup_role(
                        warp_idx, storage, epilogue_params, tile_sched_params
                    )

    def make_ab_pipeline(
        self,
        tiled_mma: cute.TiledMma,
        cluster_layout_vmnk: cute.Layout,
        ab_pipeline_mbar_ptr: cute.Pointer,
    ):
        # Threads/warps participating in this pipeline
        """Construct the A/B mainloop pipeline, sized for this kernel's warp roles.

        Args:
            tiled_mma: The tiled MMA, whose warpgroup count sets the consumer arrive count.
            cluster_layout_vmnk: The cluster layout, which decides the multicast masks.
            ab_pipeline_mbar_ptr: SMEM pointer to the barrier array. Must have room for
                ``ab_stage`` barrier pairs.

        Returns:
            A :class:`PipelineTmaAsync` with ``ab_stage`` stages. Both operands arrive by TMA, so
            there is a single producer thread and the arrive count is the transaction byte count.
        """
        ab_pipeline_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, 1)
        # Each warp will contribute to the arrive count with the number of mcast size
        mcast_size = self.num_mcast_ctas_a + self.num_mcast_ctas_b - 1
        consumer_arrive_cnt = mcast_size * tiled_mma.size // cute.arch.WARP_SIZE
        ab_pipeline_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, consumer_arrive_cnt
        )
        return pipeline.PipelineTmaAsync.create(
            barrier_storage=ab_pipeline_mbar_ptr,
            num_stages=self.ab_stage,
            producer_group=ab_pipeline_producer_group,
            consumer_group=ab_pipeline_consumer_group,
            tx_count=self.num_tma_load_bytes,
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )
